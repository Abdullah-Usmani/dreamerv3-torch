import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# --- CONFIGURATION ---
experiments = {
    # Baseline Seeds
    "Cartpole Base 0": "./logdir/cartpole_fast_precision16_actr4", 
    "Cartpole Base 1": "./logdir/cartpole_fast_base_1", 
    "Cartpole Base 2": "./logdir/cartpole_fast_base_2", 
    
    # Hybrid Seeds
    "Cartpole LLCD 0": "./logdir/cartpole_llcd_diffadap_test1", # LLCD 0
    # "Cartpole LLCD 1": "./logdir/cartpole_llcd_diffadap_test2", # LLCD 1
    # "Cartpole LLCD 2": "./logdir/cartpole_llcd_diffadap_test4", # LLCD 2

    "Cartpole Other": "./logdir/cartpole_llcd_diffadap_test0", # LLCD 0
}

# The metrics you want in the grid
metrics_to_plot = [
    "task_0_eval_return", 
    "task_1_eval_return", 
    "task_2_eval_return", 
    "task_3_eval_return",
    "crl_avg_reward_all_tasks", 
    "crl_cf_task_0",
    "crl_cf_task_1",
    "crl_cf_task_2",
    "model_loss",
]

# Binning size: Round steps to nearest X to force alignment
# If your log_every is 5000 or 10000, set this to match.
STEP_BIN_SIZE = 10000 

action_repeat = 4
prefill_step_size = 2500 * action_repeat
task_switches = [100000 + prefill_step_size, 200000 + prefill_step_size, 300000 + prefill_step_size, 400000 + prefill_step_size]

# ---------------------

def load_data(experiments, metrics):
    all_data = []

    for label, path in experiments.items():
        if "Base" in label:
            group = "Baseline (Memory Only)"
        elif "LLCD" in label:
            group = "Hybrid (LLCD + Memory)"
        else:
            group = "Other"

        file_path = pathlib.Path(path) / "metrics.jsonl"
        if not file_path.exists():
            print(f"⚠️ Warning: Skipping {label}, file not found.")
            continue

        print(f"Loading {label}...")
        
        with open(file_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    raw_step = entry.get("step")
                    if raw_step is None: continue
                    
                    # --- FIX: Round the step to force alignment ---
                    # e.g. 5001 -> 5000, 5200 -> 5000, 7800 -> 10000
                    # This ensures Seed 1 and Seed 2 land on the same X-value.
                    aligned_step = round(raw_step / STEP_BIN_SIZE) * STEP_BIN_SIZE
                    
                    for metric in metrics:
                        if metric in entry:
                            all_data.append({
                                "Step": aligned_step, # Use aligned step
                                "Value": entry[metric],
                                "Metric": metric,
                                "Group": group,
                                "Run": label
                            })
                except: continue

    return pd.DataFrame(all_data)

def print_text_analysis(df):
    """
    Calculates and prints:
    1. Model Loss summary.
    2. Comprehensive Forgetting Analysis for Task 0, 1, and 2.
    """
    print("\n" + "="*50)
    print("       NUMERICAL ANALYSIS REPORT       ")
    print("="*50)
    
    groups = df["Group"].unique()
    max_step_available = df["Step"].max()
    
    # --- 1. Model Loss Analysis ---
    print(f"\n[ Model Loss Summary ]")
    loss_data = df[df["Metric"] == "model_loss"]
    if not loss_data.empty:
        for group in groups:
            g_data = loss_data[loss_data["Group"] == group]
            max_step = g_data["Step"].max()
            # Avg loss over last 50k steps
            final_loss = g_data[g_data["Step"] >= max_step - 50000]["Value"].mean()
            print(f"  > {group}: Final Avg Loss ~ {final_loss:.4f}")
    else:
        print("  (No model_loss data found)")

    # --- 2. General Forgetting Loop ---
    # We define the "End Step" for each task to establish the Baseline.
    # Task 0 ends at 100k, Task 1 at 200k, Task 2 at 300k, Task 3 at Max.
    
    task_definitions = [
        # (Task Label, Metric Name, End Step Value, Task Index)
        ("Task 0", "task_0_eval_return", task_switches[0], 0),
        ("Task 1", "task_1_eval_return", task_switches[1], 1),
        ("Task 2", "task_2_eval_return", task_switches[2], 2),
    ]
    
    # Map task index to its end step for lookup
    switch_map = {
        0: task_switches[0], # 100k
        1: task_switches[1], # 200k
        2: task_switches[2], # 300k
        3: task_switches[3] # End of experiment
    }

    for task_name, metric, baseline_step, task_idx in task_definitions:
        print(f"\n[ {task_name} Forgetting Analysis ]")
        
        t_data = df[df["Metric"] == metric]
        if t_data.empty:
            print(f"  (No data found for {metric})")
            continue

        for group in groups:
            g_data = t_data[t_data["Group"] == group]
            
            # 1. Get Baseline Score (Score at the end of THIS task)
            # We look at the window [baseline_step - 10000, baseline_step]
            base_window = g_data[ (g_data["Step"] <= baseline_step) & (g_data["Step"] > baseline_step - 10000) ]
            
            if base_window.empty:
                # Fallback: find nearest if exact window is empty
                if g_data.empty: baseline_score = 0.0
                else:
                    nearest_idx = (g_data["Step"] - baseline_step).abs().idxmin()
                    baseline_score = g_data.loc[nearest_idx, "Value"]
            else:
                baseline_score = base_window["Value"].mean()

            print(f"  > {group}:")
            print(f"      Baseline (End of {task_name}): {baseline_score:.2f}")

            # 2. Calculate Forgetting for all SUBSEQUENT tasks
            # If we are analyzing Task 0 (idx 0), we check after T1, T2, T3.
            # If we are analyzing Task 1 (idx 1), we check after T2, T3.
            
            for future_idx in range(task_idx + 1, 4): # Loop up to Task 3
                future_step = switch_map.get(future_idx)
                future_label = f"Task {future_idx}"
                
                # Get score at the end of that future task
                future_window = g_data[ (g_data["Step"] <= future_step) & (g_data["Step"] > future_step - 10000) ]
                
                if future_window.empty:
                    if g_data.empty: current_score = 0.0
                    else:
                        nearest_idx = (g_data["Step"] - future_step).abs().idxmin()
                        current_score = g_data.loc[nearest_idx, "Value"]
                else:
                    current_score = future_window["Value"].mean()
                
                forgetting = current_score - baseline_score 
                print(f"      Forgetting after {future_label}: {forgetting:.2f}")

    print("="*50 + "\n")
    
def plot_grid(df):
    num_plots = len(metrics_to_plot)
    cols = 3
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    sns.set_theme(style="whitegrid")
    palette = {"Baseline (Memory Only)": "#e74c3c", "Hybrid (LLCD + Memory)": "#2ecc71", "Other": "#3498db"}

    print("\n--- Generating Grid Plot ---")
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        subset = df[df["Metric"] == metric]
        
        if subset.empty:
            ax.set_title(f"{metric} (No Data)")
            continue

        # Plot
        sns.lineplot(
            data=subset,
            x="Step",
            y="Value",
            hue="Group",
            style="Group",
            palette=palette,
            linewidth=2,
            errorbar='sd', # Standard Deviation Shading
            ax=ax,
            legend=(i == 0) 
        )
        
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xlabel("Env Steps")
        ax.set_ylabel("Value")
        
        for switch in task_switches:
            ax.axvline(x=switch, color='black', linestyle=':', alpha=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_path = "final_results_grid_shaded.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Comparison Grid Saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    df = load_data(experiments, metrics_to_plot)
    if not df.empty:
        # Debug check: See if we actually have multiple points per step
        counts = df.groupby(["Metric", "Group", "Step"]).size()
        print(f"Debug: Max seeds found per step: {counts.max()}")
        if counts.max() < 2:
            print("⚠️ WARNING: Data is still not aligning! Check your log directories.")
            
        # 1. Print textual stats
        print_text_analysis(df)
        
        # 2. Plot graph
        plot_grid(df)
    else:
        print("❌ No data found.")