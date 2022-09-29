import numpy as np
import JaxPref.reward_transform as r_tf

class StepSampler(object):

    def __init__(self, env, max_traj_length=1000, reward_trans=None, act_flag=False, act_coeff=1e-3):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()
        self._reward_trans = reward_trans
        self._act_flag = act_flag
        self._act_coeff = act_coeff
        
    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            action = policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
            next_observation, reward, done, info = self.env.step(action)
            observations.append(observation)
            actions.append(action)
            if self._reward_trans is not None:
                if self._act_flag:
                    reward_run = reward + self._act_coeff*np.square(action).sum()
                    new_reward = self._reward_trans(reward_run, np.square(action).sum())
                else:
                    new_reward = self._reward_trans(reward)
                reward = new_reward
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env


class TrajSampler(object):

    def __init__(self, env, max_traj_length=1000, loco_flag=True):
        self.max_traj_length = max_traj_length
        self._env = env
        self._loco_flag = loco_flag
        if not self._loco_flag:
            self.goal = r_tf.get_goal(env.unwrapped.spec.id)

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            rewards_run = []
            rewards_ctrl = []
            next_observations = []
            dones = []
            distance = []

            observation = self.env.reset()

            for _ in range(self.max_traj_length):
                action = policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                next_observation, reward, done, info = self.env.step(action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                if self._loco_flag:
                    rewards_run.append(info['reward_run'])
                    rewards_ctrl.append(info['reward_ctrl'])
                else:
                    xy = next_observation[:2]
                    distance.append(np.linalg.norm(xy-self.goal))
                dones.append(done)
                next_observations.append(next_observation)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done
                    )

                observation = next_observation

                if done:
                    break

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                rewards_run=np.array(rewards_run, dtype=np.float32),
                rewards_ctrl=np.array(rewards_ctrl, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
                distance=np.array(distance, dtype=np.float32)
            ))

        return trajs

    @property
    def env(self):
        return self._env
