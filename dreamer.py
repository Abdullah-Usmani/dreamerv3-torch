import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "glfw"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()

# run command: python dreamer.py --configs dmc_crl --task dmc_crl_cartpole_balance `
# >>   --crl_steps_per_task 1000 `
# >>   --eval_every 1000 `
# >>   --log_every 1000 `
# >>   --steps 10000 `
# >>   --logdir ./logdir/dmc_crl_cartpole_test_v3  

class MetricCaptureLogger:
    """Wraps the logger to capture specific metrics for CF calculations."""
    def __init__(self, logger, task_prefix, capture_dict):
        self._logger = logger
        self._prefix = task_prefix
        self._capture_dict = capture_dict

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def scalar(self, name, value):
        # Capture the value if it's the return
        if name == "eval_return":
            self._capture_dict[self._prefix] = value
        
        # Log with prefix (e.g. "task_0_eval_return") so TensorBoard separates them
        self._logger.scalar(f"{self._prefix}_{name}", value)

    def video(self, name, value):
        self._logger.video(f"{self._prefix}_{name}", value)

    def write(self, **kwargs):
        self._logger.write(**kwargs)

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        
        # 1. Forward the dynamics
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        
        if getattr(self._config, "llcd", False) and training:
            deter = latent["deter"]
            
            # 1. Update Detector & Get Score
            has_changed, score = self._wm.change_detector.update(deter)
            
            # 2. Calculate Z-Score (Technique 2)
            # We do this OUTSIDE the detector so we can see it even if it doesn't trigger
            z_score = 0.0
            detector = self._wm.change_detector
            
            # Check if detector has enough history to calculate Z-score
            if detector.n > 50: 
                safe_std = max(detector.score_std, 0.001)
                z_score = (score - detector.score_mean) / safe_std
            
            # Save for logging
            self._wm.llcd_score_log = score
            self._wm.llcd_z_score_log = z_score

            if has_changed:
                print(f"[LLCD] 🚨 LIVE DETECT! Score: {score:.2f} (Z: {z_score:.2f}) | Adapting...")
                
                # 3. LATCH the Z-Score (Save it for the 50-step window)
                # We clamp it to 0.0 min to avoid negative weights if something weird happens
                self._wm.latched_z_score = max(0.0, z_score)
                self._wm.current_adaptation_score = score 
                
                self._wm.adaptation_timer = 50 
                self._wm.change_detector.reset()

        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        # We iterate through metrics to find LLCD ones and write them raw
        current_step = self._logger.step
        for key, value in metrics.items():
            if key.startswith("llcd_"):
                # Write directly to TensorBoard, bypassing the 10k averaging window
                self._logger.offline_scalar(key, value, current_step)
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    # --- FIX: Custom parsing for Continual Learning Suite ---
    if config.task.startswith("dmc_crl"):
        parts = config.task.split("_")
        
        # Handling "ball_in_cup" which has underscores in the name
        # Task string format: dmc_crl_ball_in_cup_catch
        # parts: ['dmc', 'crl', 'ball', 'in', 'cup', 'catch']
        
        if "ball" in parts:
            domain = "ball_in_cup"
            task_name = parts[-1] # "catch"
        else:
            domain = parts[2]
            task_name = parts[3]
        
        import envs.dmc as dmc
        
        use_vision = "image" in config.encoder.get("cnn_keys", [])
        
        if domain == "cartpole":
            env = dmc.ContinualCartPole(
                task=task_name, 
                action_repeat=config.action_repeat,
                size=config.size,
                seed=config.seed + id,
                vision=use_vision
            )
        elif domain == "walker":
            env = dmc.ContinualWalker(
                task=task_name,
                action_repeat=config.action_repeat,
                size=config.size,
                seed=config.seed + id,
                vision=use_vision
            )
        # -----------------
        else:
            raise NotImplementedError(f"Unknown CRL domain: {domain}")

        env = wrappers.NormalizeActions(env)
        env = wrappers.TimeLimit(env, config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
        return env

    # Standard Dreamer logic for other suites
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "gymnasium":
        from envs.gymnasium import GymEnv
        vision = getattr(config, "vision", False) or getattr(config, "pixel_obs", False)
        env = GymEnv(
            task,
            action_repeat=config.action_repeat,
            size=config.size,
            seed=config.seed + id,
            vision=vision,
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    if suite not in ["gymnasium"]:
        env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # --- CRL INITIALIZATION ---
    crl_active = hasattr(config, "crl_tasks")
    current_task_idx = -1  # Start at -1 to force initial update
    peak_returns = {}      # Track best performance per task
    
    if crl_active:
        crl_tasks = config.crl_tasks
        steps_per_task = config.crl_steps_per_task
        print(f"[CRL] Continual Learning Mode Active. Schedule: {crl_tasks}, Switch every {steps_per_task} steps.")

    while agent._step < config.steps + config.eval_every:
        logger.write()
        
        # --- DYNAMIC TASK SWITCHING ---
        if crl_active:
            
            prefill_steps = getattr(config, "prefill", 0)
            effective_step = max(0, agent._step - prefill_steps)
            cycle_length = steps_per_task * len(crl_tasks)
            cycle_step = effective_step % cycle_length
            expected_idx = cycle_step // steps_per_task

            print(f"[CRL] Step {agent._step} | Effective Task: {expected_idx}")

            desired_task_id = crl_tasks[expected_idx]
            
            # Only update if the task has CHANGED
            if current_task_idx != desired_task_id:
                print(f"[CRL] Step {agent._step}: Switching dynamics to Task {desired_task_id}")
                
                # Update all training environments
                for env in train_envs:
                    # Case 1: Parallel / Damy Wrapper
                    if hasattr(env, "call"):
                        env.call("set_task", desired_task_id)
                        
                    # Case 2: Standard Wrapper (Fallback)
                    else:
                        real_env = env
                        while hasattr(real_env, "_env") or hasattr(real_env, "env"):
                            if hasattr(real_env, "set_task"):
                                break
                            real_env = getattr(real_env, "_env", getattr(real_env, "env", None))
                        
                        if hasattr(real_env, "set_task"):
                            real_env.set_task(desired_task_id)
                            
                # Update local tracker
                current_task_idx = desired_task_id
        # ------------------------------

        if config.eval_episode_num > 0:
            print("Start evaluation on ALL tasks (for CF metrics).")
            eval_policy = functools.partial(agent, training=False)
            current_eval_results = {}
            tasks_to_eval = crl_tasks if crl_active else [0]
            
            for task_id in tasks_to_eval:
                # 1. Switch Eval Envs to this task
                for env in eval_envs:
                    if hasattr(env, "call"):
                         env.call("set_task", task_id)
                    else:
                        real_env = env
                        while hasattr(real_env, "_env") or hasattr(real_env, "env"):
                            if hasattr(real_env, "set_task"): break
                            real_env = getattr(real_env, "_env", getattr(real_env, "env", None))
                        if hasattr(real_env, "set_task"):
                            real_env.set_task(task_id)
                
                # 2. Wrap Logger
                task_prefix = f"task_{task_id}"
                proxy_logger = MetricCaptureLogger(logger, task_prefix, current_eval_results)
                
                # 3. Run Simulation
                tools.simulate(
                    eval_policy,
                    eval_envs,
                    eval_eps,
                    config.evaldir,
                    proxy_logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )

            # --- CALCULATE METRICS ---
            all_scores = list(current_eval_results.values())
            avg_reward = sum(all_scores) / len(all_scores) if all_scores else 0
            logger.scalar("crl_avg_reward_all_tasks", avg_reward)
            
            total_forgetting = 0
            tasks_seen = 0
            
            for task_id_str, score in current_eval_results.items():
                tid = int(task_id_str.split("_")[1])
                
                previous_peak = peak_returns.get(tid, -float('inf'))
                if score > previous_peak:
                    peak_returns[tid] = score
                
                # Calculate Forgetting only for tasks we have passed
                if crl_active and tid < current_task_idx:
                    forgetting = peak_returns[tid] - score
                    logger.scalar(f"crl_cf_task_{tid}", forgetting)
                    total_forgetting += forgetting
                    tasks_seen += 1
            
            if tasks_seen > 0:
                logger.scalar("crl_avg_forgetting", total_forgetting / tasks_seen)
                
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
            
            logger.write()

        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
