from typing import Dict

import flax.linen as nn
import gym
import numpy as np
from tqdm import trange


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': [], 'success': []}

    for _ in trange(num_episodes, desc='evaluation', leave=False):
        observation, done = env.reset(), False

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
