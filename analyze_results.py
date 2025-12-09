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
    "Cartpole Test": "./logdir/cartpole_llcd_diffadap_test2", 
    "Cartpole Mock": "./logdir/cartpole_llcd_diffadap_test4", 

    "Cartpole Other": "./logdir/cartpole_llcd_diffadap_test11", 
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
STEP_BIN_SIZE = 10000 

action_repeat = 4
prefill_step_size = 2500 * action_repeat
task_switches = [100000 + prefill_step_size, 200000 + prefill_step_size, 300000 + prefill_step_size, 400000 + prefill_step_size]

# Convergence Threshold (e.g., 0.90 means "steps to reach 90% of max score")
CONVERGENCE_THRESHOLD = 0.90

# ---------------------

def load_data(experiments, metrics):
    all_data = []

    for label, path in experiments.items():
        if "Base" in label:
            group = "Continual-DreamerV3"
        elif "LLCD" in label:
            group = "Hybrid (LLCD + Memory)"
        elif "Test" in label:
            group = "LLCD Hybrid (Constant)"
        elif "Mock" in label:
            group = "LLCD Hybrid (Z-Score)"
        else:
            group = "LLCD Hybrid (Score)"

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
                    
                    aligned_step = round(raw_step / STEP_BIN_SIZE) * STEP_BIN_SIZE
                    
                    for metric in metrics:
                        if metric in entry:
                            all_data.append({
                                "Step": aligned_step, 
                                "Value": entry[metric],
                                "Metric": metric,
                                "Group": group,
                                "Run": label
                            })
                except: continue

    return pd.DataFrame(all_data)

def calculate_convergence(df, metric_name, group, start_step, end_step):
    """
    Finds the first step where performance >= 90% of the Max Performance within the task window.
    Returns: (Steps Taken to Converge, Max Score)
    """
    # Filter for specific group and metric within the task time window
    subset = df[
        (df["Metric"] == metric_name) & 
        (df["Group"] == group) & 
        (df["Step"] >= start_step) & 
        (df["Step"] <= end_step)
    ]
    
    if subset.empty:
        return "N/A", 0.0

    # We smooth the data to avoid spikes triggering "fake" convergence
    # Group by step to handle multiple seeds, then take mean
    mean_trajectory = subset.groupby("Step")["Value"].mean()
    
    if mean_trajectory.empty:
        return "N/A", 0.0

    max_score = mean_trajectory.max()
    target_score = max_score * CONVERGENCE_THRESHOLD
    
    # Find first step where value >= target
    # We subtract start_step to get "Steps SINCE task start"
    converged_steps = mean_trajectory[mean_trajectory >= target_score].index
    
    if len(converged_steps) > 0:
        first_success_step = converged_steps[0]
        speed = first_success_step - start_step
        # Ensure speed isn't negative due to binning alignment
        return max(0, speed), max_score
    else:
        return "Not Reached", max_score


def print_text_analysis(df):
    print("\n" + "="*60)
    print("             NUMERICAL ANALYSIS REPORT             ")
    print("="*60)
    
    groups = df["Group"].unique()
    
    # --- 1. Model Loss Analysis ---
    print(f"\n[ Model Loss Summary ]")
    loss_data = df[df["Metric"] == "model_loss"]
    if not loss_data.empty:
        for group in groups:
            g_data = loss_data[loss_data["Group"] == group]
            max_step = g_data["Step"].max()
            final_loss = g_data[g_data["Step"] >= max_step - 50000]["Value"].mean()
            print(f"  > {group}: Final Avg Loss ~ {final_loss:.4f}")
    else:
        print("  (No model_loss data found)")

    # --- 2. Convergence Analysis ---
    print(f"\n[ Convergence Speed (Steps to reach {int(CONVERGENCE_THRESHOLD*100)}% of Max) ]")
    
    task_windows = [
        ("Task 0", "task_0_eval_return", 0, task_switches[0]),
        ("Task 1", "task_1_eval_return", task_switches[0], task_switches[1]),
        ("Task 2", "task_2_eval_return", task_switches[1], task_switches[2]),
        ("Task 3", "task_3_eval_return", task_switches[2], task_switches[3]),
    ]
    
    for task_name, metric, start, end in task_windows:
        print(f"\n  -- {task_name} --")
        for group in groups:
            speed, max_score = calculate_convergence(df, metric, group, start, end)
            if speed == "N/A":
                print(f"    {group:<25}: No Data")
            elif speed == "Not Reached":
                print(f"    {group:<25}: Did not converge (Max: {max_score:.1f})")
            else:
                print(f"    {group:<25}: {speed} steps (Max: {max_score:.1f})")

    # --- 3. General Forgetting Loop ---
    task_definitions = [
        ("Task 0", "task_0_eval_return", task_switches[0], 0),
        ("Task 1", "task_1_eval_return", task_switches[1], 1),
        ("Task 2", "task_2_eval_return", task_switches[2], 2),
    ]
    switch_map = {0: task_switches[0], 1: task_switches[1], 2: task_switches[2], 3: task_switches[3]}

    for task_name, metric, baseline_step, task_idx in task_definitions:
        print(f"\n[ {task_name} Forgetting Analysis ]")
        t_data = df[df["Metric"] == metric]
        if t_data.empty: continue

        for group in groups:
            g_data = t_data[t_data["Group"] == group]
            
            # Baseline (End of Task X)
            base_window = g_data[(g_data["Step"] <= baseline_step) & (g_data["Step"] > baseline_step - 10000)]
            if base_window.empty:
                if g_data.empty: baseline_score = 0.0
                else: baseline_score = g_data.loc[(g_data["Step"] - baseline_step).abs().idxmin(), "Value"]
            else:
                baseline_score = base_window["Value"].mean()

            print(f"  > {group}: Baseline {baseline_score:.2f}")

            for future_idx in range(task_idx + 1, 4):
                future_step = switch_map.get(future_idx)
                future_label = f"Task {future_idx}"
                
                future_window = g_data[(g_data["Step"] <= future_step) & (g_data["Step"] > future_step - 10000)]
                if future_window.empty:
                    if g_data.empty: current_score = 0.0
                    else: current_score = g_data.loc[(g_data["Step"] - future_step).abs().idxmin(), "Value"]
                else:
                    current_score = future_window["Value"].mean()
                
                forgetting = current_score - baseline_score 
                print(f"      Forgetting after {future_label}: {forgetting:.2f}")

    print("="*60 + "\n")
    
def plot_grid(df):
    num_plots = len(metrics_to_plot)
    cols = 3
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    sns.set_theme(style="whitegrid")
    unique_groups = df["Group"].unique()
    palette = {group: sns.color_palette("husl", len(unique_groups))[i] for i, group in enumerate(unique_groups)}

    print("\n--- Generating Grid Plot ---")
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        subset = df[df["Metric"] == metric]
        if subset.empty:
            ax.set_title(f"{metric} (No Data)")
            continue

        sns.lineplot(
            data=subset, x="Step", y="Value", hue="Group", style="Group",
            palette=palette, linewidth=2, errorbar='sd', ax=ax, legend=(i == 0) 
        )
        
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xlabel("Env Steps")
        ax.set_ylabel("Value")
        for switch in task_switches:
            ax.axvline(x=switch, color='black', linestyle=':', alpha=0.5)

        # Save individual plot
        plt.figure()
        sns.lineplot(
            data=subset, x="Step", y="Value", hue="Group", style="Group",
            palette=palette, linewidth=2, errorbar='sd'
        )
        plt.title(metric, fontsize=14, fontweight='bold')
        plt.xlabel("Env Steps")
        plt.ylabel("Value")
        for switch in task_switches:
            plt.axvline(x=switch, color='black', linestyle=':', alpha=0.5)
        plt.tight_layout()
        individual_path = f"new_imgs/{metric}_plot.png"
        plt.savefig(individual_path, dpi=300)
        plt.close()
        print(f"✅ Saved individual plot for {metric} to: {individual_path}")

    for j in range(i + 1, len(axes)): axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("new_imgs/final_results_grid_shaded.png", dpi=300)
    print(f"✅ Comparison Grid Saved to: final_results_grid_shaded.png")
    plt.show()

if __name__ == "__main__":
    df = load_data(experiments, metrics_to_plot)
    if not df.empty:
        # Debug check
        counts = df.groupby(["Metric", "Group", "Step"]).size()
        if counts.max() < 2:
            print("⚠️ WARNING: Data alignment check failed (Need multiple seeds for stats).")
            
        print_text_analysis(df)
        plot_grid(df)
    else:
        print("❌ No data found.")