import os
import pickle

import gym
import imageio
import jax
import numpy as np
from absl import app, flags
from tqdm import tqdm, trange

import d4rl
from JaxPref.reward_transform import load_queries_with_indices

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "antmaze-medium-diverse-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./video/", "saving dir.")
flags.DEFINE_string("query_path", "./human_label/", "query path")
flags.DEFINE_integer("num_query", 1000, "number of query.")
flags.DEFINE_integer("query_len", 100, "length of each query.")
flags.DEFINE_integer("label_type", 1, "label type.")
flags.DEFINE_integer("seed", 3407, "seed for reproducibility.")

video_size = {"medium": (500, 500), "large": (600, 450)}


def set_seed(env, seed):
    np.random.seed(seed)
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)


def qlearning_mujoco_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    xy_ = []
    done_bef_ = []

    qpos_ = []
    qvel_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i]) or episode_step == env._max_episode_steps - 1
        xy = dataset["infos/qpos"][i][:2].astype(np.float32)

        qpos = dataset["infos/qpos"][i]
        qvel = dataset["infos/qvel"][i]

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
            next_final_timestep = dataset["timeouts"][i + 1]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
            next_final_timestep = episode_step == env._max_episode_steps - 2

        done_bef = bool(next_final_timestep)

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        xy_.append(xy)
        done_bef_.append(done_bef)

        qpos_.append(qpos)
        qvel_.append(qvel)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "xys": np.array(xy_),
        "dones_bef": np.array(done_bef_),
        "qposes": np.array(qpos_),
        "qvels": np.array(qvel_),
    }


class Dataset(object):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        qposes: np.ndarray,
        qvels: np.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.qposes = qposes
        self.qvels = qvels
        self.size = size


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        dataset = qlearning_mujoco_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(dataset["observations"][i + 1] - dataset["next_observations"][i]) > 1e-5
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            qposes=dataset["qposes"].astype(np.float32),
            qvels=dataset["qvels"].astype(np.float32),
            size=len(dataset["observations"]),
        )


def visualize_query(
    gym_env, dataset, batch, query_len, num_query, width=500, height=500, save_dir="./video", verbose=False
):
    save_dir = os.path.join(save_dir, gym_env.spec.id)
    os.makedirs(save_dir, exist_ok=True)

    for seg_idx in trange(num_query):
        start_1, start_2 = (
            batch["start_indices"][seg_idx],
            batch["start_indices_2"][seg_idx],
        )
        frames = []
        frames_2 = []

        start_indices = range(start_1, start_1 + query_len)
        start_indices_2 = range(start_2, start_2 + query_len)

        gym_env.reset()

        if verbose:
            print(f"start pos of first one: {dataset['qposes'][start_indices[0]][:2]}")
            print("=" * 50)
            print(f"start pos of second one: {dataset['qposes'][start_indices_2[0]][:2]}")

        camera_name = "track"

        for t in trange(query_len, leave=False):
            gym_env.set_state(dataset["qposes"][start_indices[t]], dataset["qvels"][start_indices[t]])
            curr_frame = gym_env.sim.render(width=width, height=height, mode="offscreen", camera_name=camera_name)
            frames.append(np.flipud(curr_frame))
        gym_env.reset()
        for t in trange(query_len, leave=False):
            gym_env.set_state(
                dataset["qposes"][start_indices_2[t]],
                dataset["qvels"][start_indices_2[t]],
            )
            curr_frame = gym_env.sim.render(width=width, height=height, mode="offscreen", camera_name=camera_name)
            frames_2.append(np.flipud(curr_frame))

        video = np.concatenate((np.array(frames), np.array(frames_2)), axis=2)

        writer = imageio.get_writer(os.path.join(save_dir, f"./idx{seg_idx}.mp4"), fps=30)
        for frame in tqdm(video, leave=False):
            writer.append_data(frame)
        writer.close()

    print("save query indices.")
    with open(
        os.path.join(save_dir, f"human_indices_numq{num_query}_len{query_len}_s{FLAGS.seed}.pkl"),
        "wb",
    ) as f:
        pickle.dump(batch["start_indices"], f)
    with open(
        os.path.join(
            save_dir,
            f"human_indices_2_numq{num_query}_len{query_len}_s{FLAGS.seed}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(batch["start_indices_2"], f)


def main(_):
    gym_env = gym.make(FLAGS.env_name)
    if "medium" in FLAGS.env_name:
        width, height = video_size["medium"]
    elif "large" in FLAGS.env_name:
        width, height = video_size["large"]
    set_seed(gym_env, FLAGS.seed)
    ds = qlearning_mujoco_dataset(gym_env)

    base_path = os.path.join(FLAGS.query_path, FLAGS.env_name)
    human_indices_2_file, human_indices_1_file, _ = sorted(os.listdir(base_path))
    with open(os.path.join(base_path, human_indices_1_file), "rb") as fp:   # Unpickling
        human_indices = pickle.load(fp)
    with open(os.path.join(base_path, human_indices_2_file), "rb") as fp:   # Unpickling
        human_indices_2 = pickle.load(fp)
    human_labels = None
    batch = load_queries_with_indices(
        gym_env,
        ds,
        saved_indices=[human_indices, human_indices_2],
        saved_labels=human_labels,
        num_query=FLAGS.num_query,
        len_query=FLAGS.query_len,
        label_type=FLAGS.label_type,
        scripted_teacher=True
    )
    visualize_query(
        gym_env, ds, batch, FLAGS.query_len, FLAGS.num_query, width=width, height=height, save_dir=FLAGS.save_dir
    )


if __name__ == "__main__":
    app.run(main)
