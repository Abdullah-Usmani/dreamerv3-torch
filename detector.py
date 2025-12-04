import numpy as np
from scipy.stats import t

class ChangeDetector:
    """
    Adaptive BOCPD with 3-Sigma Thresholding.
    Faithful implementation of statistical outlier detection.
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        
        # 1. Detection Statistics (Student-T Logic)
        self.n = 0
        self.mean = np.zeros(input_dim)
        self.sum_sq = np.zeros(input_dim)
        
        # 2. Adaptive Threshold Statistics (3-Sigma Logic)
        self.score_history = [] 
        self.score_mean = 0.0
        self.score_std = 1.0 
        
        # --- THIS IS THE VARIABLE YOU WERE MISSING ---
        self.history_window = 500  # Track last 500 scores for robust baseline
        # ---------------------------------------------

    def update(self, deter_state):
        # 1. Convert to Numpy
        if hasattr(deter_state, 'detach'):
            x = deter_state.detach().cpu().numpy()
        else:
            x = deter_state
        if x.ndim > 1:
            x = x.mean(axis=0)
            
        score = 0.0
        if self.n > 5:
            # Student-T NLL Calculation
            df = self.n - 1
            var = (self.sum_sq - self.n * (self.mean**2)) / (self.n - 1)
            var = np.maximum(var, 1e-6)
            scale = np.sqrt(var * (1 + 1/self.n))
            t_score = (x - self.mean) / scale
            log_pdf = t.logpdf(t_score, df) - np.log(scale)
            score = -np.mean(log_pdf)

        # Update Distribution Stats
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.sum_sq += x**2
        
        # 2. Adaptive Decision (3-Sigma)
        is_change = False
        
        # Burn-in: Wait 500 steps to establish a baseline
        if self.n > 500:
            # Enforce minimum noise floor to prevent infinite sensitivity
            min_std = 0.1
            effective_std = max(self.score_std, min_std)
            
            z_score = (score - self.score_mean) / effective_std
            
            # 3-Sigma Rule
            if z_score > 5.0: 
                is_change = True
        
        # Update Score History (Rolling Window)
        if not is_change:
            self.score_history.append(score)
            if len(self.score_history) > self.history_window:
                self.score_history.pop(0) # Remove oldest
            
            if len(self.score_history) > 2:
                self.score_mean = np.mean(self.score_history)
                self.score_std = np.std(self.score_history)

        return is_change, score

    def reset(self):
        # LLCD Strategy: Reset distribution belief, but KEEP score history
        # This ensures the threshold remains stable across tasks
        self.n = 0
        self.mean = np.zeros(self.input_dim)
        self.sum_sq = np.zeros(self.input_dim)