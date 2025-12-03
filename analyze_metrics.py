import json
import sys
import pandas as pd

def calculate_convergence(log_file, threshold=300):
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Filter for evaluation steps
    tasks = [0, 1, 2, 3]
    convergence_results = {}
    
    for t in tasks:
        col = f"task_{t}_eval_return"
        if col in df.columns:
            # Find first step where return > threshold
            converged_row = df[df[col] >= threshold].sort_values("step").head(1)
            if not converged_row.empty:
                step = converged_row["step"].values[0]
                convergence_results[f"Task {t}"] = step
                print(f"Task {t} converged at step {step}")
            else:
                print(f"Task {t} did not converge (max: {df[col].max()})")
    
    return convergence_results

if __name__ == "__main__":
    log_path = "./logdir/dmc_crl_cartpole_full_run_2/metrics.jsonl"
    calculate_convergence(log_path, threshold=400)