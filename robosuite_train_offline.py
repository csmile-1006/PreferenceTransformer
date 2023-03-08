import datetime
import os
import pickle
from typing import Tuple

import gym
import numpy as np
from tqdm import tqdm
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from tensorboardX import SummaryWriter


import robosuite as suite
from robosuite.wrappers import GymWrapper
import robomimic.utils.env_utils as EnvUtils

import wrappers
from JaxPref.reward_transform import qlearning_robosuite_dataset
from dataset_utils import D4RLDataset, RelabeledDataset, reward_from_preference, reward_from_preference_transformer, split_into_trajectories
from evaluation import evaluate
from learner import Learner


# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.40'

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './logs/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('use_reward_model', False, 'Use reward model for relabeling reward.')
flags.DEFINE_string('model_type', 'MLP', 'type of reward model.')
flags.DEFINE_string('ckpt_dir',
                    './logs/pref_reward',
                    'ckpt path for reward model.')
flags.DEFINE_string('comment',
                    'base',
                    'comment for distinguishing experiments.')
flags.DEFINE_integer('seq_len', 25, 'sequence length for relabeling reward in Transformer.')
flags.DEFINE_bool('use_diff', False, 'boolean whether use difference in sequence for reward relabeling.')
flags.DEFINE_string('label_mode', 'last', 'mode for relabeling reward with tranformer.')
flags.DEFINE_string('pref_attn_type', 'max', 'mode for preference attention with tranformer.')
flags.DEFINE_integer('max_episode_steps', 500, 'max_episode_steps for rollout.')
flags.DEFINE_string('robosuite_dataset_path', './data', 'hdf5 dataset path for demonstrations')
flags.DEFINE_string('robosuite_dataset_type', 'ph', 'dataset type for robosuite')
# flags.DEFINE_list(
#     'obs_keys',
#     ["robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel", "object"],
#     'obs keys for using in making observations.'
# )

config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset, env_name, max_episode_steps=1000):
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)
    trj_mapper = []
    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="chunk trajectories"):
        traj_len = len(traj)

        for _ in range(traj_len):
            trj_mapper.append((trj_idx, traj_len))

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    sorted_trajs = sorted(trajs, key=compute_returns)
    min_return, max_return = compute_returns(sorted_trajs[0]), compute_returns(sorted_trajs[-1])

    normalized_rewards = []
    for i in range(dataset.size):
        _reward = dataset.rewards[i]
        if 'antmaze' in env_name:
            _, len_trj = trj_mapper[i]
            _reward -= min_return / len_trj
        _reward /= max_return - min_return
        # if ('halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name):
        _reward *= max_episode_steps
        normalized_rewards.append(_reward)

    dataset.rewards = np.array(normalized_rewards)


def make_env_and_dataset(env_name: str,
                         seed: int,
                         dataset_path: str,
                         max_episode_steps: int = 500) -> Tuple[gym.Env, D4RLDataset]:


    ds = qlearning_robosuite_dataset(dataset_path)
    dataset = RelabeledDataset(ds['observations'], ds['actions'], ds['rewards'], ds['terminals'], ds['next_observations'])

    ds['env_meta']['env_kwargs']['horizon'] = max_episode_steps
    env = EnvUtils.create_env_from_metadata(
        env_meta=ds['env_meta'],
        render=False,            # no on-screen rendering
        render_offscreen=False,   # off-screen rendering to support rendering video frames
    ).env
    env.ignore_done = False

    env._max_episode_steps = env.horizon
    env = GymWrapper(env)
    env = wrappers.RobosuiteWrapper(env)
    env = wrappers.EpisodeMonitor(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if FLAGS.use_reward_model:
        reward_model = initialize_model()
        if FLAGS.model_type == "MR":
            dataset = reward_from_preference(FLAGS.env_name, dataset, reward_model, batch_size=FLAGS.batch_size)
        else:
            dataset = reward_from_preference_transformer(
                FLAGS.env_name,
                dataset,
                reward_model,
                batch_size=FLAGS.batch_size,
                seq_len=FLAGS.seq_len,
                use_diff=FLAGS.use_diff,
                label_mode=FLAGS.label_mode
            )
        del reward_model

    if FLAGS.use_reward_model:
        normalize(dataset, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps)
        # if 'antmaze' in FLAGS.env_name:
        #     dataset.rewards -= 1.0
        if ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name or 'hopper' in FLAGS.env_name):
            dataset.rewards += 0.5
    else:
        if 'antmaze' in FLAGS.env_name:
            dataset.rewards -= 1.0
            # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
            # but I found no difference between (x - 0.5) * 4 and x - 1.0
        elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name or 'hopper' in FLAGS.env_name):
            normalize(dataset, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps)

    if 'pen' in FLAGS.env_name or 'hammer' in FLAGS.env_name:
        trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)
        trj_cumsum = np.cumsum([len(traj) for traj in trajs])
        split_point = trj_cumsum[int(len(trajs) // 2)]
        dataset.observations = dataset.observations[:split_point]
        dataset.actions = dataset.actions[:split_point]
        dataset.rewards = dataset.rewards[:split_point]
        dataset.masks = dataset.masks[:split_point]
        dataset.dones_float = dataset.dones_float[:split_point]
        dataset.next_observations = dataset.next_observations[:split_point]
        dataset.size = len(dataset.observations)

    return env, dataset


def initialize_model():
    if os.path.exists(os.path.join(FLAGS.ckpt_dir, "best_model.pkl")):
        model_path = os.path.join(FLAGS.ckpt_dir, "best_model.pkl")
    else:
        model_path = os.path.join(FLAGS.ckpt_dir, "model.pkl")

    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt['reward_model']
    if FLAGS.model_type == "PrefTransformer":
        reward_model.trans.config.pref_attn_type = FLAGS.pref_attn_type
    return reward_model


def main(_):
    save_dir = os.path.join(FLAGS.save_dir, 'tb',
                        FLAGS.env_name,
                            f"reward_{FLAGS.use_reward_model}_{FLAGS.model_type}" if FLAGS.use_reward_model else "original",
                            f"{FLAGS.comment}",
                            str(FLAGS.seed),
                            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    summary_writer = SummaryWriter(save_dir,
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    dataset_path = os.path.join(FLAGS.robosuite_dataset_path, FLAGS.env_name.lower(), FLAGS.robosuite_dataset_type, "low_dim.hdf5")
    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, dataset_path, max_episode_steps=FLAGS.max_episode_steps)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    eval_returns = []
    for i in tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(save_dir, 'progress.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

    # save IQL agent for last timestep.
    checkpoints.save_checkpoint(os.path.join(save_dir, "actor"), target=agent.actor, step=FLAGS.max_steps)
    checkpoints.save_checkpoint(os.path.join(save_dir, "critic"), target=agent.critic, step=FLAGS.max_steps)
    checkpoints.save_checkpoint(os.path.join(save_dir, "value"), target=agent.value, step=FLAGS.max_steps)
    checkpoints.save_checkpoint(os.path.join(save_dir, "target_critic"), target=agent.actor, step=FLAGS.max_steps)

if __name__ == '__main__':
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    app.run(main)
