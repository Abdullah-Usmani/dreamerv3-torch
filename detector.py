import numpy as np
from scipy.stats import t

class ChangeDetector:
    """
    Adaptive BOCPD with 3-Sigma Thresholding (Faithful to LLCD Paper).
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        
        # 1. Detection Statistics (Student-T Logic)
        self.n = 0
        self.mean = np.zeros(input_dim)
        self.sum_sq = np.zeros(input_dim)
        
        # 2. Adaptive Threshold Statistics (3-Sigma Logic)
        # We track the history of the 'scores' themselves to find outliers
        self.score_history = [] 
        self.score_mean = 0.0
        self.score_std = 1.0 # Initialize non-zero to avoid div/0
        self.history_window = 100 # Moving window for adaptability

    def update(self, deter_state):
        # --- 1. Calculate Raw Anomaly Score (Same as before) ---
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
            score = -np.mean(log_pdf) # Raw Anomaly Score

        # Update Distribution Stats
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.sum_sq += x**2
        
        # --- 2. Adaptive Decision (The 3-Sigma Rule) ---
        is_change = False
        
        # We need a "Burn-in" period to learn what normal noise looks like
        if self.n > 20:
            # Calculate Z-Score: How many sigmas away is this score?
            z_score = (score - self.score_mean) / (self.score_std + 1e-8)
            
            # The "3-Sigma" Rule (LLCD Method)
            # If the score is 3 standard deviations higher than average -> CHANGE
            if z_score > 3.0: 
                is_change = True
        
        # Update Score History (Rolling Window)
        # We only update stats if it's NOT a change (to avoid polluting "normal" stats with anomalies)
        if not is_change:
            self.score_history.append(score)
            if len(self.score_history) > self.history_window:
                self.score_history.pop(0)
            
            # Recalculate Score Stats
            if len(self.score_history) > 2:
                self.score_mean = np.mean(self.score_history)
                self.score_std = np.std(self.score_history)

        return is_change, score

    def reset(self):
        # When a change is confirmed, we reset the Distribution Stats
        # BUT we keep the Score Stats (Score history) to maintain threshold stability?
        # Actually, LLCD resets the "Run Length", which implies resetting the distribution belief.
        
        self.n = 0
        self.mean = np.zeros(self.input_dim)
        self.sum_sq = np.zeros(self.input_dim)
        # We do NOT reset score_history, because we still need to know what "scores" generally look like.