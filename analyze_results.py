import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

# --- CONFIGURATION ---
experiments = {
    # Baseline Seeds
    "Cartpole Base 0": "./logdir/cartpole_fast_precision16_actr4", 
    "Cartpole Base 1": "./logdir/cartpole_fast_base_1", 
    "Cartpole Base 2": "./logdir/cartpole_fast_base_2", 
    
    # Hybrid Seeds
    "Cartpole LLCD 0": "./logdir/cartpole_llcd_diffadap_test2", 
    "Cartpole LLCD 1": "./logdir/cartpole_llcd_diffadap_test1", 
    "Cartpole LLCD 2": "./logdir/cartpole_llcd_diffadap_test4", 

    # # Baseline Seeds
    # "Walker Base 0": "./logdir/walker_tr64_0", 
    # "Walker Base 1": "./logdir/walker_tr64_1", 
    
    # # Hybrid Seeds
    # "Walker LLCD 0": "./logdir/walker_llcd_tr64_0", 
    # "Walker LLCD 1": "./logdir/walker_llcd_tr64_1", 
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
# STEP_BIN_SIZE = 60000 # for Walker

action_repeat = 4
prefill_step_size = 2500 * action_repeat
task_switches = [100000 + prefill_step_size, 200000 + prefill_step_size, 300000 + prefill_step_size, 400000 + prefill_step_size]
# task_switches = [600000 + prefill_step_size, 1200000 + prefill_step_size, 1800000 + prefill_step_size, 2400000 + prefill_step_size] # for Walker

# Convergence Threshold (e.g., 0.90 means "steps to reach 90% of max score")
CONVERGENCE_THRESHOLD = 0.90

# ---------------------

def load_data(experiments, metrics):
    all_data = []

    for label, path in experiments.items():
        if "Base" in label:
            group = "Continual-DreamerV3"
        elif "LLCD" in label:
            group = "LLCD-Dreamer"
        elif "Test" in label:
            group = "LLCD Hybrid (Constant)"
        elif "Mock" in label:
            group = "LLCD Hybrid (NLL-Score)"
        else:
            group = "LLCD Hybrid (Z-Score)"

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
                                "Return": entry[metric],
                                "Metric": metric,
                                "Group": group,
                                "Run": label
                            })
                except: continue

    return pd.DataFrame(all_data)

def get_stats_string(values):
    """Helper to return 'Mean ± Std' string from a list of values."""
    if len(values) == 0:
        return "N/A"
    mean_val = np.mean(values)
    std_val = np.std(values) if len(values) > 1 else 0.0
    return f"{mean_val:.2f} ± {std_val:.2f}"

def print_text_analysis(df):
    print("\n" + "="*80)
    print(f"{'NUMERICAL ANALYSIS REPORT (Mean ± Std)':^80}")
    print("="*80)
    
    groups = df["Group"].unique()
    
    # --- 1. Model Loss Analysis ---
    print(f"\n[ Model Loss Summary (Final 50k steps) ]")
    loss_data = df[df["Metric"] == "model_loss"]
    
    if not loss_data.empty:
        for group in groups:
            # Get all runs (seeds) for this group
            group_runs = loss_data[loss_data["Group"] == group]["Run"].unique()
            final_losses = []
            
            for run in group_runs:
                # Isolate data for THIS seed
                run_data = loss_data[loss_data["Run"] == run]
                max_step = run_data["Step"].max()
                # Average loss for this specific seed over its last 50k steps
                final_val = run_data[run_data["Step"] >= max_step - 50000]["Return"].mean()
                if not pd.isna(final_val):
                    final_losses.append(final_val)
            
            print(f"  > {group:<25}: {get_stats_string(final_losses)}")
    else:
        print("  (No model_loss data found)")

    # --- 2. Return Analysis (UPDATED: Max Return) ---
    print(f"\n[ Return Analysis (Max Return Achieved in Task Window) ]")
    
    task_windows = [
        ("Task 0", "task_0_eval_return", 0, task_switches[0]),
        ("Task 1", "task_1_eval_return", task_switches[0], task_switches[1]),
        ("Task 2", "task_2_eval_return", task_switches[1], task_switches[2]),
        ("Task 3", "task_3_eval_return", task_switches[2], task_switches[3]),
    ]

    for task_name, metric, start, end in task_windows:
        print(f"\n  -- {task_name} --")
        for group in groups:
            group_runs = df[(df["Group"] == group) & (df["Metric"] == metric)]["Run"].unique()
            run_maxs = []

            for run in group_runs:
                subset = df[
                    (df["Metric"] == metric) & 
                    (df["Run"] == run) & 
                    (df["Step"] >= start) & 
                    (df["Step"] <= end)
                ]
                if not subset.empty:
                    # CHANGED: Use .max() instead of .mean()
                    # We want to know if the agent EVER solved the task
                    run_maxs.append(subset["Return"].max())

            print(f"    {group:<25}: {get_stats_string(run_maxs)}")

    # --- 3. Average Reward Analysis ---
    print(f"\n[ Average Reward Across All Tasks (Final Snapshot) ]")
    avg_reward_data = df[df["Metric"] == "crl_avg_reward_all_tasks"]
    if not avg_reward_data.empty:
        for group in groups:
            group_runs = avg_reward_data[avg_reward_data["Group"] == group]["Run"].unique()
            final_vals = []
            
            for run in group_runs:
                g_data = avg_reward_data[avg_reward_data["Run"] == run]
                if not g_data.empty:
                    latest_step = g_data["Step"].max()
                    val = g_data[g_data["Step"] == latest_step]["Return"].mean()
                    final_vals.append(val)
            
            print(f"  > {group:<25}: {get_stats_string(final_vals)}")
    else:
        print("  (No crl_avg_reward_all_tasks data found)")

    # --- 4. Convergence Analysis ---
    print(f"\n[ Convergence Speed (Steps to reach {int(CONVERGENCE_THRESHOLD*100)}% of Max) ]")
    
    for task_name, metric, start, end in task_windows:
        print(f"\n  -- {task_name} --")
        for group in groups:
            group_runs = df[(df["Group"] == group) & (df["Metric"] == metric)]["Run"].unique()
            convergence_speeds = []
            max_scores = []
            
            for run in group_runs:
                subset = df[
                    (df["Metric"] == metric) & 
                    (df["Run"] == run) & 
                    (df["Step"] >= start) & 
                    (df["Step"] <= end)
                ]
                
                if subset.empty: continue
                
                trajectory = subset.groupby("Step")["Return"].mean()
                
                max_score = trajectory.max()
                max_scores.append(max_score)
                
                target_score = max_score * CONVERGENCE_THRESHOLD
                converged_indices = trajectory[trajectory >= target_score].index
                
                if len(converged_indices) > 0:
                    speed = converged_indices[0] - start
                    convergence_speeds.append(max(0, speed))

            speed_str = get_stats_string(convergence_speeds)
            max_str = get_stats_string(max_scores)
            
            if len(convergence_speeds) == 0:
                 print(f"    {group:<25}: Not Reached (Max: {max_str})")
            else:
                 print(f"    {group:<25}: {speed_str} steps (Max: {max_str})")

    # --- 5. General Forgetting Loop ---
    print(f"\n[ Forgetting Analysis (Baseline - Current) ]")
    
    task_definitions = [
        ("Task 0", "task_0_eval_return", task_switches[0], 0),
        ("Task 1", "task_1_eval_return", task_switches[1], 1),
        ("Task 2", "task_2_eval_return", task_switches[2], 2),
    ]
    switch_map = {0: task_switches[0], 1: task_switches[1], 2: task_switches[2], 3: task_switches[3]}

    for task_name, metric, baseline_step, task_idx in task_definitions:
        print(f"\n[ {task_name} Forgetting ]")
        
        if df[df["Metric"] == metric].empty: continue

        for group in groups:
            group_runs = df[(df["Group"] == group) & (df["Metric"] == metric)]["Run"].unique()
            baselines = []
            forgetting_results = {} 

            for run in group_runs:
                run_data = df[(df["Run"] == run) & (df["Metric"] == metric)]
                
                base_window = run_data[(run_data["Step"] <= baseline_step) & (run_data["Step"] > baseline_step - 10000)]
                if base_window.empty:
                    if run_data.empty: continue
                    baseline_val = run_data.iloc[(run_data["Step"] - baseline_step).abs().argmin()]["Return"]
                else:
                    baseline_val = base_window["Return"].mean()
                
                baselines.append(baseline_val)
                
                for future_idx in range(task_idx + 1, 4):
                    future_step = switch_map.get(future_idx)
                    future_label = f"Task {future_idx}"
                    
                    if future_label not in forgetting_results: forgetting_results[future_label] = []

                    future_window = run_data[(run_data["Step"] <= future_step) & (run_data["Step"] > future_step - 10000)]
                    
                    if future_window.empty:
                        current_val = run_data.iloc[(run_data["Step"] - future_step).abs().argmin()]["Return"]
                    else:
                        current_val = future_window["Return"].mean()
                    
                    forgetting_results[future_label].append(baseline_val - current_val)

            print(f"  > {group:<25}: Baseline {get_stats_string(baselines)}")
            
            sorted_tasks = sorted(forgetting_results.keys())
            for future_label in sorted_tasks:
                vals = forgetting_results[future_label]
                print(f"      After {future_label:<7}: {get_stats_string(vals)}")

    print("="*80 + "\n")
    
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
            data=subset, x="Step", y="Return", hue="Group", style="Group",
            palette=palette, linewidth=2, errorbar='sd', ax=ax, legend=(i == 0) 
        )
        
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xlabel("Env Steps")
        ax.set_ylabel("Return")
        for switch in task_switches:
            ax.axvline(x=switch, color='black', linestyle=':', alpha=0.5)

        plt.figure()
        sns.lineplot(
            data=subset, x="Step", y="Return", hue="Group", style="Group",
            palette=palette, linewidth=2, errorbar='sd'
        )
        plt.title(metric, fontsize=14, fontweight='bold')
        plt.xlabel("Env Steps")
        plt.ylabel("Return")
        for switch in task_switches:
            plt.axvline(x=switch, color='black', linestyle=':', alpha=0.5)
        plt.tight_layout()
        individual_path = f"run_imgs1/{metric}_plot.png"
        plt.savefig(individual_path, dpi=300)
        plt.close()
        print(f"✅ Saved individual plot for {metric} to: {individual_path}")

    for j in range(i + 1, len(axes)): axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("run_imgs1/final_results_grid_shaded.png", dpi=300)
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