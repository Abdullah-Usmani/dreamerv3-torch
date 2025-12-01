import argparse
import pathlib
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import ruamel.yaml as yaml

# Add current directory to path so we can import internal modules
sys.path.append(str(pathlib.Path(__file__).parent))

import tools
from dreamer import Dreamer
import envs.dmc as dmc

def test_detection(config, checkpoint_path=None):
    print("--- 🧪 INITIALIZING LLCD DETECTION TEST ---")
    
    # 1. Setup Environment
    print(f"Creating Environment: {config.task}")
    env = dmc.ContinualCartPole(
        task="balance", 
        action_repeat=config.action_repeat, 
        size=config.size
    )
    
    # Set num_actions
    act_space = env.action_space
    if hasattr(act_space, "n"):
        config.num_actions = act_space.n
    else:
        config.num_actions = act_space.shape[0]
    print(f"Detected Action Space: {config.num_actions}")
    
    # 2. Setup Agent
    print("Initializing Dreamer Agent...")
    dummy_logger = tools.Logger(pathlib.Path("./logdir/test_dummy"), 0)
    
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger=dummy_logger,
        dataset=None,
    ).to(config.device)
    
    # Load Checkpoint
    if checkpoint_path and pathlib.Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        agent.load_state_dict(checkpoint["agent_state_dict"])
    else:
        print("⚠️ WARNING: No checkpoint loaded. Running with Random Weights.")

    # 3. Test Loop
    print("\n--- STARTING SIMULATION ---")
    print("Phase 1: Task 0 (Standard Gravity) for 500 steps")
    print("Phase 2: Task 3 (Jupiter Gravity) for 500 steps")
    
    obs = env.reset()
    state = None
    done = False
    
    scores = []
    tasks = []
    
    # Initialize action/is_first
    agent_action = torch.zeros((1, 1, config.num_actions), device=config.device)
    
    for t in range(1000):
        # --- THE INTERVENTION ---
        if t == 500:
            print("\n[TEST] 🚨 FORCE SWITCHING GRAVITY TO JUPITER (Task 3) 🚨\n")
            env.set_task(3) 
        
        # 1. Preprocess Obs
        obs_tensor = agent._wm.preprocess(obs)
        
        # --- CRITICAL FIX: Add Batch Dimension [1, ...] ---
        # Encoder expects [Batch, C, H, W]
        obs_tensor = {k: v.unsqueeze(0) for k, v in obs_tensor.items()}
        # --------------------------------------------------
        
        # 2. Run Forward Pass (World Model)
        with torch.no_grad():
            # Embed Image -> Output [Batch, Dim]
            embed = agent._wm.encoder(obs_tensor)
            
            # Add Time Dimension -> [Batch, Time, Dim] = [1, 1, Dim]
            embed = embed.unsqueeze(1) 
            
            # RSSM Observe
            if state is None:
                action = torch.zeros((1, 1, config.num_actions), device=config.device)
                is_first = torch.ones((1, 1), device=config.device)
            else:
                action = agent_action
                is_first = torch.zeros((1, 1), device=config.device)

            # Run Dynamics
            post, _ = agent._wm.dynamics.observe(
                embed, action, is_first, state
            )
            
            # Extract Deterministic State
            deter = post["deter"][0, -1, :] 
            
            # Update Agent State for next step
            state = {k: v[:, -1] for k, v in post.items()}

            # 3. Get Action
            feat = agent._wm.dynamics.get_feat(post)
            actor_out = agent._task_behavior.actor(feat)
            agent_action = actor_out.sample()
            
            env_action = agent_action.detach().cpu().numpy()[0, 0]

        # 4. RUN DETECTOR
        is_change, score = agent._wm.change_detector.update(deter)
        
        scores.append(score)
        tasks.append(env.task_phase)
        
        if is_change:
            print(f"Step {t}: 🔔 Detector Triggered! Score: {score:.4f}")
        elif t % 100 == 0:
            print(f"Step {t}: Normal. Score: {score:.4f}")

        # 5. Environment Step
        obs, reward, done, info = env.step(env_action)
        if done:
            obs = env.reset()
            state = None
            # Reset action to zero on reset
            agent_action = torch.zeros((1, 1, config.num_actions), device=config.device)

    # 4. Visualization
    plot_results(scores, tasks)

def plot_results(scores, tasks):
    print("\n--- PLOTTING RESULTS ---")
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label="Detector Score (NLL)", color="blue")
    plt.axhline(y=0.3, color="red", linestyle="--", label="Threshold") # Default 0.3
    plt.axvline(x=500, color="green", linestyle="-", linewidth=2, label="Gravity Switch")
    plt.title("LLCD Detector Sensitivity Test")
    plt.xlabel("Step")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("detection_test_result.png")
    print(f"Graph saved to detection_test_result.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["dmc_crl"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args, remaining = parser.parse_known_args()
    
    yaml_path = pathlib.Path(__file__).parent / "configs.yaml"
    try:
        from ruamel.yaml import YAML
        yaml = YAML(typ='safe', pure=True)
        configs = yaml.load(yaml_path.read_text())
    except ImportError:
        import yaml
        configs = yaml.safe_load(yaml_path.read_text())
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
        
    class Config: pass
    config = Config()
    for k, v in defaults.items():
        setattr(config, k, v)
        
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.batch_size = 1 
    config.batch_length = 1 
    config.envs = 1
    
    test_detection(config, args.checkpoint)