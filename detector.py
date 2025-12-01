import numpy as np
from scipy.stats import t

class ChangeDetector:
    """
    Manual Implementation of Bayesian Online Change Point Detection (BOCPD).
    Optimized for high-dimensional vectors using Diagonal Covariance (Fast).
    """
    def __init__(self, input_dim, threshold=0.3, hazard=0.01):
        self.input_dim = input_dim
        self.threshold = threshold
        self.hazard = hazard
        
        # Initialize Priors (Normal-Gamma for unknown mean & precision)
        self.reset()

    def reset(self):
        # T = Current Run Length
        # We track parameters for run lengths r=0, r=1, ... r=t
        # To keep it fast, we only track the "most likely" run length (approximate)
        # or we track the sufficient statistics for the current segment.
        
        # Sufficient Statistics for the current segment
        self.n = 0              # Count
        self.mean = np.zeros(self.input_dim)
        self.sum_sq = np.zeros(self.input_dim) # Sum of squares
        
        # Score smoothing
        self.change_prob = 0.0

    def update(self, deter_state):
        """
        Input: deter_state (Tensor or Numpy array)
        Output: (bool is_change, float score)
        """
        # 1. Convert to Numpy
        if hasattr(deter_state, 'detach'):
            x = deter_state.detach().cpu().numpy()
        else:
            x = deter_state
            
        # Handle Batch: Take mean if batch dimension exists
        if x.ndim > 1:
            x = x.mean(axis=0) # [512]
            
        # 2. Calculate Predictive Probability (Student-T)
        # We calculate the probability of this new point 'x' belonging to the 
        # current distribution we have been tracking.
        
        score = 0.0
        
        if self.n > 2:
            # We have enough data to form a belief
            # Posterior parameters
            df = self.n - 1
            
            # Variance calculation (Welford's algorithm style or direct)
            # var = (sum_sq - n * mean^2) / (n - 1)
            var = (self.sum_sq - self.n * (self.mean**2)) / (self.n - 1)
            var = np.maximum(var, 1e-6) # Avoid division by zero
            scale = np.sqrt(var * (1 + 1/self.n))
            
            # Calculate negative log likelihood (Anomaly Score)
            # Higher NLL = Lower probability = Higher chance of change
            # We sum NLL across dimensions (assuming diagonal/independent dims)
            
            # Standardize x
            t_score = (x - self.mean) / scale
            
            # Student-T Log PDF
            # log_pdf = t.logpdf(x, df, loc=self.mean, scale=scale)
            # Using t_score simplifies:
            log_pdf = t.logpdf(t_score, df) - np.log(scale)
            
            # The "Change Score" is the negative log probability averaged over dims
            score = -np.mean(log_pdf)
        
        # 3. Update Statistics with new point
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.sum_sq += x**2
        
        # 4. Determine Change
        # Simple threshold logic on the Negative Log Likelihood
        is_change = False
        
        # Heuristic: If the score spikes significantly above the threshold
        if score > self.threshold and self.n > 10: # Wait for burn-in (10 steps)
            is_change = True
            
        return is_change, score