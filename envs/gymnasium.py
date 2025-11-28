import gymnasium as gym
import numpy as np
import cv2

class GymEnv:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0, vision=False):
        self.name = name
        self.vision = vision
        self._action_repeat = action_repeat
        self._size = size
        self.seed = seed

        # ALWAYS enable RGB rendering (required for DreamerV3 video logging)
        self.env = gym.make(name, render_mode="rgb_array")
        self.env.reset(seed=seed)

        # Detect discrete env
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self._discrete_n = self.env.action_space.n
            # Dreamer requires continuous Box actions, even for discrete policies
            self.action_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self._discrete_n,), dtype=np.float32
            )
        else:
            self._discrete_n = None
            self.action_space = self.env.action_space

        # Build Dreamer-style observation space
        if vision:
            self.observation_space = gym.spaces.Dict({
                "image": gym.spaces.Box(0, 255, size + (3,), dtype=np.uint8),
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            })
        else:
            obs, _ = self.env.reset()
            obs_shape = np.asarray(obs, np.float32).shape
            self.observation_space = gym.spaces.Dict({
                "proprio": gym.spaces.Box(-np.inf, np.inf, obs_shape, dtype=np.float32),
                "image": gym.spaces.Box(0, 255, size + (3,), dtype=np.uint8),  # REQUIRED
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            })

    # ---------------------------------------------------
    # RESET
    # ---------------------------------------------------
    def reset(self):
        obs, info = self.env.reset(seed=self.seed)

        frame = self.render()

        if self.vision:
            return {
                "image": frame,
                "is_first": True,
                "is_terminal": False,
            }
        else:
            return {
                "proprio": np.asarray(obs, np.float32),
                "image": frame,
                "is_first": True,
                "is_terminal": False,
            }

    # ---------------------------------------------------
    # STEP
    # ---------------------------------------------------
    def step(self, action):
        # If discrete, convert NN output to integer action
        if self._discrete_n is not None:
            action = int(np.argmax(action))

        total_reward = 0
        terminated = False
        truncated = False

        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        done = terminated or truncated
        frame = self.render()

        if self.vision:
            obs_out = {
                "image": frame,
                "is_first": False,
                "is_terminal": bool(terminated),
            }
        else:
            obs_out = {
                "proprio": np.asarray(obs, np.float32),
                "image": frame,
                "is_first": False,
                "is_terminal": bool(terminated),
            }

        return obs_out, total_reward, done, info

    # ---------------------------------------------------
    # RENDER
    # ---------------------------------------------------
    def render(self, *args, **kwargs):
        frame = self.env.render()

        if frame is None:
            raise ValueError("Environment did not return an rgb_array frame.")

        frame = np.asarray(frame, dtype=np.uint8)

        # ✅ Force resize here before anything else can use it
        frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)

        return frame
