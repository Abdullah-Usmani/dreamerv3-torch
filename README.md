# Continual Learning with DreamerV3 + LLCD

This repository is a fork of the [DreamerV3-Torch](https://github.com/NM512/dreamerv3-torch) repository (original README.md text is further below), modified to support \***\*Continual Reinforcement Learning (CRL)\*\*** scenarios. It integrates the \***\*Learning Latent and Changing Dynamics (LLCD)\*\*** algorithm to detect and adapt to sudden changes in environment physics (e.g., gravity, friction, wind) without catastrophic forgetting.

\***\*Project Links:\*\*** [GitHub](https://github.com/Abdullah-Usmani/dreamerv3-torch/) | [GitHub (Old)](https://github.com/Abdullah-Usmani/ARS-CW2526/)

---

## ⚡ Key Additions

1. \***\*Continual Reinforcement Learning (CRL) Loop\*\***:

   - Automated task switching based on step counts.

   - Environment wrappers that modify physics parameters on-the-fly.

2. \***\*LLCD Integration\*\***:

   - \***\*Anomaly Detection\*\***: Adaptive 3-Sigma detector using Negative Log Likelihood (NLL) of latent dynamics.

   - \***\*Adaptation Module\*\***: A secondary "context encoder" that learns a latent variable ($z$) to capture physics parameters.

   - \***\*Gated Adaptation\*\***: Mechanism to switch adaptation on/off to prevent overfitting to noise.

3. \***\*Reservoir Replay Buffer\*\***:

   - Replaces standard FIFO memory with Reservoir Sampling to retain long-term experiences for CRL.

---

## 🛠️ Installation & Setup

This codebase requires \***\*Python 3.11\*\***.

### 1. Create Virtual Environment

Bash

```
# Create venv
```

```
python -m venv venv
```

```
# Activate (Windows)
```

```
venvScriptsactivate
```

```
# Activate (Linux/Mac)
```

```
source venv/bin/activate
```

### **2. Install Dependencies**

Ensure you have the modified requirements.txt present.

Bash

```
pip install -r requirements.txt
```

**Key Requirements:**

- torch (PyTorch)
- dm_control, mujoco (Physics Engine)
- gym, gymnasium (Environment interfaces)
- matplotlib, seaborn, pandas (Analysis)

Windows Note: If you encounter rendering issues with MuJoCo, ensure you set the backend before running:

```
set MUJOCO_GL=glfw
```

---

## **📂 Changed Files & Architecture**

### **Core Logic**

- **dreamer.py**:
  - **main()**: Contains the master CRL loop that tracks steps and switches tasks defined in the schedule.
  - **make_env()**: Updated to initialize Continual\* environments.
  - **policy()**: Includes logic to query the LLCD detector and trigger adaptation modes.
- **models.py**:
  - **\_train()**: Modified loss function to include **Adaptation Loss** (reconstruction + KL of context) when LLCD is active.
- **detector.py**:
  - Implements the **Adaptive 3-Sigma Anomaly Detector**.
  - Calculates the **Score** (NLL of the current observation given the model's prediction).
  - Maintains a sliding window of historical scores to define "normal" behavior.
- **tools.py**:
  - Added **enforce_reservoir_limit()**: Implements Reservoir Sampling. When the buffer is full, new episodes replace random old episodes (instead of oldest-first) to preserve data diversity across tasks.

### **Environments & Configs**

- **envs/dmc.py**:
  - Added ContinualCartPole, ContinualCupCatch, and ContinualPendulumSwingup.
  - These classes expose a set_task(task_id) method to alter MuJoCo physics (Gravity, Mass, Friction, Wind) at runtime.
- **configs.yaml**:
  - Added CRL-specific configs (dmc_crl_cartpole, dmc_pendulum_proprio, etc.).
  - Defines task schedules (crl_tasks: [0, 1, 2, 3]) and switch intervals (crl_steps_per_task).

---

## **🚀 Running Experiments**

### **Basic Command**

To run a standard Continual Learning experiment (e.g., CartPole):

Bash

python dreamer.py --configs dmc_crl_cartpole --task dmc_cartpole_balance --llcd True --seed 0

### **Important Flags**

- --llcd True/False: Enables or disables the LLCD adaptation module.
- --configs: Loads a specific block from configs.yaml.
- --seed: Random seed (Crucial for reproducibility).

### **Critical Hyperparameters (in configs.yaml)**

| Parameter              | Value (Example)         | Description                                                                                    |
| :--------------------- | :---------------------- | :--------------------------------------------------------------------------------------------- |
| **train_ratio**        | 4 (or 512 for CartPole) | **Crucial:** How many env steps per gradient update. Use 4 for complex physics (Cup/Pendulum). |
| **action_repeat**      | 2 or 4                  | How many simulation steps per agent decision.                                                  |
| **prefill**            | 2500 or 10000           | Random steps collected before training starts (Warmup).                                        |
| **crl_steps_per_task** | 25000                   | Number of _agent steps_ before switching physics.                                              |

---

## **📊 Analysis & Results**

### **analyze_results.py**

This script generates tables and plots for **Catastrophic Forgetting**, **Convergence Speed**, and **Model Loss**.

**Usage:**

1. Ensure your logs are in logdir/.
2. Open analyze_results.py and update the experiments dictionary to point to your specific log folders.
3. Run:
4. Bash

python analyze_results.py

5.
6.

**Outputs:**

- **Numerical Report:** Prints Mean ± Std for Returns, Convergence Steps, and Forgetting metrics per task.
- **Plots:** Saves final_results_grid_shaded.png (Grid of all metrics) and individual plots in new_imgs/.

### **Helper Scripts**

- **test_detector.py**: Unit test to verify the 3-Sigma logic on dummy data.
- **verify_physics.py**: Runs the environments without an agent to visually/numerically confirm that gravity/mass actually changes when set_task() is called.

---

## **🐛 Troubleshooting**

- **Video Logging Crash:** If you see KeyError: 'image', it means the logger is trying to make a video for a vector-only environment. Set video_pred_log: false in configs.yaml.
- **0 Rewards in Proprioception:** Check train_ratio. If set to 32 or higher, the agent learns too slowly for dynamic tasks like Ball-in-Cup. Lower it to 4.

---

# Original README

# dreamerv3-torch

Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

### Method 1: Manual

Get dependencies with python 3.11:

```
pip install -r requirements.txt
```

Run training on DMC Vision:

```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```

Monitor results:

```
tensorboard --logdir ./logdir
```

To set up Atari or Minecraft environments, please check the scripts located in [env/setup_scripts](https://github.com/NM512/dreamerv3-torch/tree/main/envs/setup_scripts).

### Method 2: Docker

Please refer to the Dockerfile for the instructions, as they are included within.

## Benchmarks

So far, the following benchmarks can be used for testing.
| Environment | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [Atari 100k](https://github.com/openai/atari-py) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Minecraft](https://github.com/minerllabs/minerl) | Image and State |Discrete |100M| Vast 3D open world.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|

## Results

#### DMC Proprio

![dmcproprio](imgs/dmcproprio.png)

#### DMC Vision

![dmcvision](imgs/dmcvision.png)

#### Atari 100k

![atari100k](imgs/atari100k.png)

#### Crafter

<img src="https://github.com/NM512/dreamerv3-torch/assets/70328564/a0626038-53f6-4300-a622-7ac257f4c290" width="300" height="150" />

## Acknowledgments

This code is heavily inspired by the following works:

- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
