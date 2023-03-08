import gym
import numpy as np

from wrappers.common import TimeStep


class RobosuiteWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._max_episode_steps = self.env.horizon
    
    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        if self.env._check_success():
            done = True

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        return self.env.reset()

