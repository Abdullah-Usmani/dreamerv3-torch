import numpy as np
from scipy.stats import t

class ChangeDetector:
    """
    Adaptive 3-Sigma Anomaly Detector.
    Calculates the Negative Log Likelihood (Score) of the current state
    against the history of the current task.
    """
    def __init__(self, input_dim, history_window=500):
        self.input_dim = input_dim
        
        # 1. Distribution Stats (The "Normal" Baseline)
        self.n = 0
        self.mean = np.zeros(input_dim)
        self.sum_sq = np.zeros(input_dim)
        
        # 2. Threshold Stats (The Noise Floor)
        self.history_window = history_window
        self.score_history = [] 
        self.score_mean = 0.0
        self.score_std = 1.0 

    def update(self, deter_state):
        # Convert Input
        if hasattr(deter_state, 'detach'):
            x = deter_state.detach().cpu().numpy()
        else:
            x = deter_state
        if x.ndim > 1:
            x = x.mean(axis=0) # Average batch
            
        score = 0.0
        
        # Calculate Anomaly Score (Student-T NLL)
        if self.n > 5:
            df = self.n - 1
            # Variance = E[x^2] - (E[x])^2
            # Clamped to avoid division by zero
            var = (self.sum_sq - self.n * (self.mean**2)) / (self.n - 1)
            var = np.maximum(var, 1e-6) 
            
            scale = np.sqrt(var * (1 + 1/self.n))
            
            # Standardize x (Z-score relative to distribution)
            t_score = (x - self.mean) / scale
            
            # Log Probability
            log_pdf = t.logpdf(t_score, df) - np.log(scale)
            
            # Score = Negative Mean Log Likelihood
            # High Score = High Surprise = Anomaly
            score = -np.mean(log_pdf)

        # Update Distribution Stats
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.sum_sq += x**2
        
        # Check for Change (3-Sigma Rule)
        is_change = False
        
        # Burn-in: Wait 500 steps before triggering
        if self.n > 500:
            # Minimum noise floor (prevent infinite sensitivity)
            min_std = 0.1
            effective_std = max(self.score_std, min_std)
            
            # Calculate Z-Score of the Anomaly Score itself
            z_score = (score - self.score_mean) / effective_std
            
            # Trigger if > 3 Standard Deviations
            if z_score > 3.0: 
                is_change = True
        
        # Update History (Only if no change, to keep baseline clean)
        if not is_change:
            self.score_history.append(score)
            if len(self.score_history) > self.history_window:
                self.score_history.pop(0)
            
            if len(self.score_history) > 2:
                self.score_mean = np.mean(self.score_history)
                self.score_std = np.std(self.score_history)

        return is_change, score

    def reset(self):
        # Reset Distribution Stats (New Task = New Mean/Var)
        self.n = 0
        self.mean = np.zeros(self.input_dim)
        self.sum_sq = np.zeros(self.input_dim)
        
        # Note: We KEEP score_history to maintain a stable noise threshold
        # across tasks, assuming the agent's internal noise level is constant.