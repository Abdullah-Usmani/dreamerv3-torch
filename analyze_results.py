import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# --- CONFIGURATION ---
# Map your folder paths here. 
# The script looks for "Base" or "LLCD" in the key name to group them.
experiments = {
    # Baseline Seeds
    "Cartpole Base 0": "./logdir/cartpole_fast_precision16_actr4", 
    "Cartpole Base 1": "./logdir/cartpole_fast_base_1", 
    "Cartpole Base 2": "./logdir/cartpole_fast_base_2", 
    
    # Hybrid Seeds
    "Cartpole LLCD 0": "./logdir/cartpole_llcd_diffadap_test1", # LLCD 0
    "Cartpole LLCD 1": "./logdir/cartpole_llcd_diffadap_test2", # LLCD 1
    # "Cartpole LLCD 2": "./logdir/cartpole_llcd_diffadap_test4", # LLCD 2
}

# The metrics you want in the grid
metrics_to_plot = [
    "task_0_eval_return", 
    "task_1_eval_return", 
    "task_2_eval_return", 
    "task_3_eval_return",
    "crl_avg_reward_all_tasks", # Added this summary metric
    "crl_cf_task_0",
    "crl_cf_task_1",
    "crl_cf_task_2",
    "model_loss",
]

# Task Switch Steps (Vertical Lines)
task_switches = [100000, 200000, 300000]

# ---------------------

def load_data(experiments, metrics):
    all_data = []

    for label, path in experiments.items():
        # Determine Group for Coloring/Shading
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
                    step = entry.get("step")
                    if step is None: continue
                    
                    for metric in metrics:
                        if metric in entry:
                            all_data.append({
                                "Step": step,
                                "Value": entry[metric],
                                "Metric": metric,
                                "Group": group,
                                "Run": label
                            })
                except: continue

    return pd.DataFrame(all_data)

def plot_grid(df):
    # Setup Grid
    num_plots = len(metrics_to_plot)
    cols = 3
    rows = math.ceil(num_plots / cols)
    
    # Create Figure
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    # Professional Style
    sns.set_theme(style="whitegrid")
    palette = {"Baseline (Memory Only)": "#e74c3c", "Hybrid (LLCD + Memory)": "#2ecc71"} # Red vs Green

    print("\n--- Generating Grid Plot ---")
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        subset = df[df["Metric"] == metric]
        
        if subset.empty:
            ax.set_title(f"{metric} (No Data)")
            continue

        # Plot with Shaded Error Bars (Mean +/- SD)
        sns.lineplot(
            data=subset,
            x="Step",
            y="Value",
            hue="Group",
            style="Group",
            palette=palette,
            linewidth=2,
            errorbar='sd', # Draws the shaded region
            ax=ax,
            legend=(i == 0) # Only show legend on first plot to save space
        )
        
        # Styling
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xlabel("Env Steps")
        ax.set_ylabel("Value")
        
        # Add Vertical Lines for Task Switches
        for switch in task_switches:
            ax.axvline(x=switch, color='black', linestyle=':', alpha=0.5)

    # Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    # Save Combined Image
    save_path = "final_results_grid.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Comparison Grid Saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    df = load_data(experiments, metrics_to_plot)
    if not df.empty:
        plot_grid(df)
    else:
        print("❌ No data found. Check your paths.")