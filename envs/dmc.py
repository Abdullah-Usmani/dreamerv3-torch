import gym
import numpy as np
import os
from dm_control import suite


class DeepMindControl:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
   
            os.environ["MUJOCO_GL"] = "glfw"
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)

class ContinualWalker:
    def __init__(self, task="walk", action_repeat=2, size=(64, 64), seed=0, vision=True):
        os.environ["MUJOCO_GL"] = "glfw"
        self._env = suite.load("walker", task, task_kwargs={"random": seed})
        self._action_repeat = action_repeat
        self._size = tuple(size)
        self._vision = vision # <--- This enables Proprioception speed
        self.camera_id = 0
        
        # Continual Learning State
        self.task_phase = 0
        self._wind_force = 0.0
        self._orig_gravity = self._env.physics.model.opt.gravity.copy()

    def set_task(self, task_id):
        """Switch Physics for Walker."""
        self.task_phase = task_id
        
        # Reset to Baseline
        self._env.physics.model.opt.gravity[:] = self._orig_gravity
        self._wind_force = 0.0

        print(f"[ContinualWalker] Setting Task {task_id}...")

        if task_id == 0:
            pass # Standard
        elif task_id == 1:
            # Walker is heavier/sturdier than CartPole, so we need more wind force
            self._wind_force = 5.0 
            print(f"   > Task 1: Windy (Force {self._wind_force}N)")
        elif task_id == 2:
            self._env.physics.model.opt.gravity[2] = -1.62
            print("   > Task 2: Moon Gravity")
        elif task_id == 3:
            self._env.physics.model.opt.gravity[2] = -25.0
            print("   > Task 3: Jupiter Gravity")

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        reward = 0
        
        for _ in range(self._action_repeat):
            # --- APPLY WIND TO TORSO ---
            if self._wind_force != 0.0:
                self._env.physics.named.data.xfrc_applied["torso", 0] = self._wind_force
            
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        
        obs = self._get_obs(time_step)
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def _get_obs(self, time_step):
        # 1. Get Proprioceptive Data (Vectors)
        obs = dict(time_step.observation)
        obs = {key: [val] if np.isscalar(val) else val for key, val in obs.items()}
        
        # 2. Only Render if Vision is ON
        if self._vision:
            obs["image"] = self.render()
            
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def reset(self):
        time_step = self._env.reset()
        return self._get_obs(time_step)
    
    def render(self, mode="rgb_array"):
        return self._env.physics.render(*self._size, camera_id=self.camera_id)

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            shape = (1,) if len(value.shape) == 0 else value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        
        # Only declare image space if vision is enabled
        if self._vision:
            spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
            
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

class ContinualCartPole:
    def __init__(self, task="balance", action_repeat=1, size=(64, 64), seed=0, vision=True):
        # Ensure we use GLFW for rendering on Windows
        os.environ["MUJOCO_GL"] = "glfw"
        
        self._env = suite.load("cartpole", task, task_kwargs={"random": seed})
        self._action_repeat = action_repeat
        # --- FIX: Ensure size is a tuple so we can add (3,) to it later ---
        # self._size = size
        self._size = tuple(size) 
        # ------------------------------------------------------------------
        self.camera_id = 0
        self._vision = vision
        
        # Continual Learning State
        self.task_phase = 0
        self._wind_force = 0.0
        self._orig_gravity = self._env.physics.model.opt.gravity.copy()

    def set_task(self, task_id):
        """Switches the physics dynamics of the environment."""
        self.task_phase = task_id
        
        # Reset physics to baseline
        self._env.physics.model.opt.gravity[:] = self._orig_gravity
        self._wind_force = 0.0

        if task_id == 0:
            print(f"[Continual] Task {task_id}: Standard")
        elif task_id == 1:
            self._wind_force = 1.0
            print(f"[Continual] Task {task_id}: Windy")
        elif task_id == 2:
            # Moon Gravity (approx 1/6 of Earth)
            self._env.physics.model.opt.gravity[2] = -1.62
            print(f"[Continual] Task {task_id}: Moon Gravity")
        elif task_id == 3:
            # High Gravity
            self._env.physics.model.opt.gravity[2] = -25.0
            print(f"[Continual] Task {task_id}: Jupiter Gravity")

        # --- VERIFICATION BLOCK ---
        # Read values DIRECTLY from the physics engine to confirm they changed
        # current_gravity = self._env.physics.model.opt.gravity[2]
        # print(f"   > CONFIRMED GRAVITY (Z): {current_gravity}")
        
        # if self._wind_force > 0:
        #     print(f"   > CONFIRMED WIND FORCE SETTING: {self._wind_force} (Will apply on step)")
        # else:
        #     print(f"   > CONFIRMED WIND: None")
        # print("--------------------------------------------------\n")

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        reward = 0
        
        for _ in range(self._action_repeat):
            # Apply Wind Force if active
            if self._wind_force != 0.0:
                self._env.physics.named.data.xfrc_applied["pole_1", 0] = self._wind_force
            
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        
        # --- VERIFICATION (Print once per 1000 steps to avoid spam) ---
        # We check xfrc_applied on the pole to see if the engine registered the force
        # if self._wind_force != 0.0 and np.random.rand() < 0.001: 
            # actual_force = self._env.physics.named.data.xfrc_applied["pole_1", 0]
            # print(f"[PHYSICS CHECK] Step wind force on pole: {actual_force}")

        obs = self._get_obs(time_step)
        done = time_step.last()
        
        # Add discount info for Dreamer
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        return self._get_obs(time_step)

    def _get_obs(self, time_step):
        # Get raw vector obs from DM Control
        obs = dict(time_step.observation)
        # Ensure values are at least 1D arrays
        obs = {key: [val] if np.isscalar(val) else val for key, val in obs.items()}
        
        # Add Image Observation (Channels-Last: 64, 64, 3)
        if self._vision:
            obs["image"] = self.render()
        
        # Add Dreamer-specific flags
        # In DMC, first() is true at reset. discount==0 means terminal/crash.
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        
        return obs

    def render(self, mode="rgb_array"):
        # Returns (H, W, 3)
        return self._env.physics.render(*self._size, camera_id=self.camera_id)

    @property
    def observation_space(self):
        spaces = {}
        # Replicate vector spaces from underlying env
        for key, value in self._env.observation_spec().items():
            shape = (1,) if len(value.shape) == 0 else value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        
        # Image Space: (64, 64, 3) -> Channels LAST
        if self._vision:
            spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)