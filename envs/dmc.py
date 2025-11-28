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


class ContinualCartPole:
    def __init__(self, task="balance", action_repeat=1, size=(64, 64), seed=0):
        # Load the standard DM Control Cartpole
        
        # Add the MuJoCo_GL = glfw setting
        os.environ["MUJOCO_GL"] = "glfw"

        self._env = suite.load("cartpole", task, task_kwargs={"random": seed})
        self._action_repeat = action_repeat
        self._size = size
        self.camera_id = 0
        
        # --- CONTINUAL LEARNING STATE ---
        self.task_phase = 0  # 0: Standard, 1: Windy, 2: Heavy, 3: Low Gravity
        self._wind_force = 0.0
        
        # Cache original physics values to allow resetting
        self._orig_gravity = self._env.physics.model.opt.gravity.copy()
        # "pole_1" is the name of the pole geom in dm_control cartpole.xml
        self._orig_mass = self._env.physics.named.model.body_mass["pole_1"].copy()

    def set_task(self, task_id):
        """
        Call this method to switch the dynamics of the environment.
        Task 0: Standard
        Task 1: Windy (External force applied to pole)
        Task 2: Heavy Pole (Mass increased)
        Task 3: Low Gravity (Moon gravity)
        """
        self.task_phase = task_id
        
        # Reset to defaults first
        self._env.physics.model.opt.gravity[:] = self._orig_gravity
        self._env.physics.named.model.body_mass["pole_1"] = self._orig_mass
        self._wind_force = 0.0

        # Apply Task Dynamics
        if task_id == 1: # WINDY
            # Will apply force in self.step()
            self._wind_force = 1.0  # Adjust magnitude as needed
            print(f"SWITCHED TO TASK {task_id}: WINDY")
            
        elif task_id == 2: # HEAVY POLE
            # Double the mass of the pole
            self._env.physics.named.model.body_mass["pole_1"] = self._orig_mass * 3.0
            print(f"SWITCHED TO TASK {task_id}: HEAVY POLE")
            
        elif task_id == 3: # LOW GRAVITY
            # Reduce gravity by half
            self._env.physics.model.opt.gravity[2] = -4.0 # Standard is -9.81
            print(f"SWITCHED TO TASK {task_id}: LOW GRAVITY")
            
        else:
            print(f"SWITCHED TO TASK {task_id}: STANDARD")

    def step(self, action):
        # Normalize action (Dreamer outputs [-1, 1])
        action = np.clip(action, -1.0, 1.0)
        
        total_reward = 0.0
        for _ in range(self._action_repeat):
            # --- APPLY WIND FORCE ---
            if self.task_phase == 1 and self._wind_force != 0.0:
                # Apply varying wind force to the pole's center of mass
                # We use a sine wave to make it 'gusty' or constant for steady wind
                # named.data.xfrc_applied index: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
                # Applying Force in X direction
                self._env.physics.named.data.xfrc_applied["pole_1", 0] = self._wind_force * np.random.normal(1.0, 0.5)
            
            time_step = self._env.step(action)
            
            if time_step.reward is not None:
                total_reward += time_step.reward
            if time_step.last():
                break
        
        obs = self._get_obs(time_step)
        done = time_step.last()
        return obs, total_reward, done, {}

    def _get_obs(self, time_step):
        # Render image for Dreamer
        img = self._env.physics.render(height=self._size[0], width=self._size[1], camera_id=self.camera_id)
        return img.transpose(2, 0, 1) # Channel-first for PyTorch

    def reset(self):
        time_step = self._env.reset()
        return self._get_obs(time_step)

    @property
    def observation_space(self):
        # Mock space for Dreamer init
        import gym
        return gym.spaces.Box(low=0, high=255, shape=(3, self._size[0], self._size[1]), dtype=np.uint8)

    @property
    def action_space(self):
        import gym
        spec = self._env.action_spec()
        return gym.spaces.Box(low=spec.minimum, high=spec.maximum, dtype=np.float32)