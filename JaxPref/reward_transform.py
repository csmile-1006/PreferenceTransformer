import os
import h5py
import pickle
from tqdm import tqdm
import numpy as np
import ujson as json
import jax.numpy as jnp


def get_goal(name):
    if 'large' in name:
        return (32.0, 24.0)
    elif 'medium' in name:
        return (20.0, 20.0)
    elif 'umaze' in name:
        return (0.0, 8.0)
    return None


def new_get_trj_idx(env, terminate_on_end=False, **kwargs):

    if not hasattr(env, 'get_dataset'):
        dataset = kwargs['dataset']
    else:
        dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]
    
    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    start_idx, data_idx = 0, 0
    trj_idx_list = []
    for i in range(N-1):
        if env.spec and 'maze' in env.spec.id:
            done_bool = sum(dataset['infos/goal'][i+1] - dataset['infos/goal'][i]) > 0
        else:
            done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx-1])
            start_idx = data_idx
            continue  
        if done_bool or final_timestep:
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx])
            start_idx = data_idx + 1
            
        episode_step += 1
        data_idx += 1
        
    trj_idx_list.append([start_idx, data_idx])
    
    return trj_idx_list


def get_queries_from_multi(env, dataset, num_query, len_query, data_dir=None, balance=False, label_type=0, skip_flag=0):
    
    os.makedirs(data_dir, exist_ok=True)
    trj_idx_list = new_get_trj_idx(env, dataset=dataset) # get_nonmdp_trj_idx(env)
    labeler_info = np.zeros(len(trj_idx_list) - 1)
    
    # to-do: parallel implementation
    trj_idx_list = np.array(trj_idx_list)
    trj_len_list = trj_idx_list[:,1] - trj_idx_list[:,0] + 1

    assert max(trj_len_list) > len_query
    
    total_reward_seq_1, total_reward_seq_2 = np.zeros((num_query, len_query)), np.zeros((num_query, len_query))

    observation_dim = dataset["observations"].shape[-1]
    total_obs_seq_1, total_obs_seq_2 = np.zeros((num_query, len_query, observation_dim)), np.zeros((num_query, len_query, observation_dim))
    total_next_obs_seq_1, total_next_obs_seq_2 = np.zeros((num_query, len_query, observation_dim)), np.zeros((num_query, len_query, observation_dim))

    action_dim = dataset["actions"].shape[-1]
    total_act_seq_1, total_act_seq_2 = np.zeros((num_query, len_query, action_dim)), np.zeros((num_query, len_query, action_dim))

    total_timestep_1, total_timestep_2 = np.zeros((num_query, len_query), dtype=np.int32), np.zeros((num_query, len_query), dtype=np.int32)

    start_indices_1, start_indices_2 = np.zeros(num_query), np.zeros(num_query)
    time_indices_1, time_indices_2 = np.zeros(num_query), np.zeros(num_query)

    indices_1_filename = os.path.join(data_dir, f"indices_num{num_query}_q{len_query}")
    indices_2_filename = os.path.join(data_dir, f"indices_2_num{num_query}_q{len_query}")
    label_dummy_filename = os.path.join(data_dir, f"label_dummy")
    
    if not os.path.exists(indices_1_filename) or not os.path.exists(indices_2_filename):
        for query_count in tqdm(range(num_query), desc="get queries"):
            temp_count = 0
            labeler = -1
            while(temp_count < 2):
                trj_idx = np.random.choice(np.arange(len(trj_idx_list) - 1)[np.logical_not(labeler_info)])
                len_trj = trj_len_list[trj_idx]
                
                if len_trj > len_query and (temp_count == 0 or labeler_info[trj_idx] == labeler):
                    labeler = labeler_info[trj_idx]
                    time_idx = np.random.choice(len_trj - len_query + 1)
                    start_idx = trj_idx_list[trj_idx][0] + time_idx
                    end_idx = start_idx + len_query

                    assert end_idx <= trj_idx_list[trj_idx][1] + 1

                    reward_seq = dataset['rewards'][start_idx:end_idx]
                    obs_seq = dataset['observations'][start_idx:end_idx]
                    next_obs_seq = dataset['next_observations'][start_idx:end_idx]
                    act_seq = dataset['actions'][start_idx:end_idx]
                    # timestep_seq = np.arange(time_idx + 1, time_idx + len_query + 1)
                    timestep_seq = np.arange(1, len_query + 1)

                    # skip flag 1: skip queries with equal rewards.
                    if skip_flag == 1 and temp_count == 1:
                        if np.sum(total_reward_seq_1[-1]) == np.sum(reward_seq):
                            continue
                    # skip flag 2: keep queries with equal reward until 50% of num_query.
                    if skip_flag == 2 and temp_count == 1 and query_count < int(0.5*num_query):
                        if np.sum(total_reward_seq_1[-1]) == np.sum(reward_seq):
                            continue
                    # skip flag 3: keep queries with equal reward until 20% of num_query.
                    if skip_flag == 3 and temp_count == 1 and query_count < int(0.2*num_query):
                        if np.sum(total_reward_seq_1[-1]) == np.sum(reward_seq):
                            continue

                    if temp_count == 0:
                        start_indices_1[query_count] = start_idx
                        time_indices_1[query_count] = time_idx
                        total_reward_seq_1[query_count] = reward_seq
                        total_obs_seq_1[query_count] = obs_seq
                        total_next_obs_seq_1[query_count] = next_obs_seq
                        total_act_seq_1[query_count] = act_seq
                        total_timestep_1[query_count] = timestep_seq
                    else:
                        start_indices_2[query_count] = start_idx
                        time_indices_2[query_count] = time_idx
                        total_reward_seq_2[query_count] = reward_seq
                        total_obs_seq_2[query_count] = obs_seq
                        total_next_obs_seq_2[query_count] = next_obs_seq
                        total_act_seq_2[query_count] = act_seq
                        total_timestep_2[query_count] = timestep_seq

                    temp_count += 1
                
        seg_reward_1 = total_reward_seq_1.copy()
        seg_reward_2 = total_reward_seq_2.copy()
        
        seg_obs_1 = total_obs_seq_1.copy()
        seg_obs_2 = total_obs_seq_2.copy()
        
        seg_next_obs_1 = total_next_obs_seq_1.copy()
        seg_next_obs_2 = total_next_obs_seq_2.copy()
        
        seq_act_1 = total_act_seq_1.copy()
        seq_act_2 = total_act_seq_2.copy()

        seq_timestep_1 = total_timestep_1.copy()
        seq_timestep_2 = total_timestep_2.copy()
        
        if label_type == 0: # perfectly rational
            sum_r_t_1 = np.sum(seg_reward_1, axis=1)
            sum_r_t_2 = np.sum(seg_reward_2, axis=1)
            binary_label = 1*(sum_r_t_1 < sum_r_t_2)
            rational_labels = np.zeros((len(binary_label), 2))
            rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        elif label_type == 1:
            sum_r_t_1 = np.sum(seg_reward_1, axis=1)
            sum_r_t_2 = np.sum(seg_reward_2, axis=1)
            binary_label = 1*(sum_r_t_1 < sum_r_t_2)
            rational_labels = np.zeros((len(binary_label), 2))
            rational_labels[np.arange(binary_label.size), binary_label] = 1.0
            margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= 0).reshape(-1)
            rational_labels[margin_index] = 0.5

        start_indices_1 = np.array(start_indices_1, dtype=np.int32)
        start_indices_2 = np.array(start_indices_2, dtype=np.int32)
        time_indices_1 = np.array(time_indices_1, dtype=np.int32)
        time_indices_2 = np.array(time_indices_2, dtype=np.int32)
        
        batch = {}
        batch['labels'] = rational_labels
        batch['observations'] = seg_obs_1 # for compatibility, remove "_1"
        batch['next_observations'] = seg_next_obs_1
        batch['actions'] = seq_act_1
        batch['observations_2'] = seg_obs_2
        batch['next_observations_2'] = seg_next_obs_2
        batch['actions_2'] = seq_act_2
        batch['timestep_1'] = seq_timestep_1
        batch['timestep_2'] = seq_timestep_2
        batch['start_indices'] = start_indices_1
        batch['start_indices_2'] = start_indices_2

        # balancing data with zero_labels
        if balance:
            nonzero_condition = np.any(batch["labels"] != [0.5, 0.5], axis=1)
            nonzero_idx, = np.where(nonzero_condition)
            zero_idx, = np.where(np.logical_not(nonzero_condition))
            selected_zero_idx = np.random.choice(zero_idx, len(nonzero_idx))
            for key, val in batch.items():
                batch[key] = val[np.concatenate([selected_zero_idx, nonzero_idx])]
            print(f"size of batch after balancing: {len(batch['labels'])}")

        with open(indices_1_filename, "wb") as fp, open(indices_2_filename, "wb") as gp, open(label_dummy_filename, "wb") as hp:
            pickle.dump(batch['start_indices'], fp)
            pickle.dump(batch['start_indices_2'], gp)
            pickle.dump(np.ones_like(batch['labels']), hp)
    else:
        with open(indices_1_filename, "rb") as fp, open(indices_2_filename, "rb") as gp:
            indices_1, indices_2 = pickle.load(fp), pickle.load(gp)

        return load_queries_with_indices(
            env, dataset, num_query, len_query, 
            label_type=label_type, saved_indices=[indices_1, indices_2], 
            saved_labels=None, balance=balance, scripted_teacher=True
        )

    return batch


def find_time_idx(trj_idx_list, idx):
    for (start, end) in trj_idx_list:
        if start <= idx <= end:
            return idx - start


def load_queries_with_indices(env, dataset, num_query, len_query, label_type, saved_indices, saved_labels, balance=False, scripted_teacher=False):
    
    trj_idx_list = new_get_trj_idx(env, dataset=dataset) # get_nonmdp_trj_idx(env)
    
    # to-do: parallel implementation
    trj_idx_list = np.array(trj_idx_list)
    trj_len_list = trj_idx_list[:, 1] - trj_idx_list[:, 0] + 1
    
    assert max(trj_len_list) > len_query
    
    total_reward_seq_1, total_reward_seq_2 = np.zeros((num_query, len_query)), np.zeros((num_query, len_query))

    observation_dim = dataset["observations"].shape[-1]
    action_dim = dataset["actions"].shape[-1]

    total_obs_seq_1, total_obs_seq_2 = np.zeros((num_query, len_query, observation_dim)), np.zeros((num_query, len_query, observation_dim))
    total_next_obs_seq_1, total_next_obs_seq_2 = np.zeros((num_query, len_query, observation_dim)), np.zeros((num_query, len_query, observation_dim))
    total_act_seq_1, total_act_seq_2 = np.zeros((num_query, len_query, action_dim)), np.zeros((num_query, len_query, action_dim))
    total_timestep_1, total_timestep_2 = np.zeros((num_query, len_query), dtype=np.int32), np.zeros((num_query, len_query), dtype=np.int32)

    query_range = np.arange(len(saved_labels) - num_query, len(saved_labels))
    for query_count, i in enumerate(tqdm(query_range, desc="get queries from saved indices")):
        temp_count = 0
        while(temp_count < 2):                
            start_idx = saved_indices[temp_count][i]
            end_idx = start_idx + len_query

            reward_seq = dataset['rewards'][start_idx:end_idx]
            obs_seq = dataset['observations'][start_idx:end_idx]
            next_obs_seq = dataset['next_observations'][start_idx:end_idx]
            act_seq = dataset['actions'][start_idx:end_idx]
            timestep_seq = np.arange(1, len_query + 1)

            if temp_count == 0:
                total_reward_seq_1[query_count] = reward_seq
                total_obs_seq_1[query_count] = obs_seq
                total_next_obs_seq_1[query_count] = next_obs_seq
                total_act_seq_1[query_count] = act_seq
                total_timestep_1[query_count] = timestep_seq
            else:
                total_reward_seq_2[query_count] = reward_seq
                total_obs_seq_2[query_count] = obs_seq
                total_next_obs_seq_2[query_count] = next_obs_seq
                total_act_seq_2[query_count] = act_seq
                total_timestep_2[query_count] = timestep_seq
                    
            temp_count += 1
            
    seg_reward_1 = total_reward_seq_1.copy()
    seg_reward_2 = total_reward_seq_2.copy()
    
    seg_obs_1 = total_obs_seq_1.copy()
    seg_obs_2 = total_obs_seq_2.copy()
    
    seg_next_obs_1 = total_next_obs_seq_1.copy()
    seg_next_obs_2 = total_next_obs_seq_2.copy()
    
    seq_act_1 = total_act_seq_1.copy()
    seq_act_2 = total_act_seq_2.copy()

    seq_timestep_1 = total_timestep_1.copy()
    seq_timestep_2 = total_timestep_2.copy()
 
    if label_type == 0: # perfectly rational
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1*(sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
    elif label_type == 1:
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1*(sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= 0).reshape(-1)
        rational_labels[margin_index] = 0.5

    batch = {}
    if scripted_teacher:
        # counter part of human label for comparing with human label.
        batch['labels'] = rational_labels
    else:
        human_labels = np.zeros((len(saved_labels), 2))
        human_labels[np.array(saved_labels)==0,0] = 1.
        human_labels[np.array(saved_labels)==1,1] = 1.
        human_labels[np.array(saved_labels)==-1] = 0.5
        human_labels = human_labels[query_range]
        batch['labels'] = human_labels
    batch['script_labels'] = rational_labels

    batch['observations'] = seg_obs_1 # for compatibility, remove "_1"
    batch['next_observations'] = seg_next_obs_1
    batch['actions'] = seq_act_1
    batch['observations_2'] = seg_obs_2
    batch['next_observations_2'] = seg_next_obs_2
    batch['actions_2'] = seq_act_2
    batch['timestep_1'] = seq_timestep_1
    batch['timestep_2'] = seq_timestep_2
    batch['start_indices'] = saved_indices[0]
    batch['start_indices_2'] = saved_indices[1]

    if balance:
        nonzero_condition = np.any(batch["labels"] != [0.5, 0.5], axis=1)
        nonzero_idx, = np.where(nonzero_condition)
        zero_idx, = np.where(np.logical_not(nonzero_condition))
        selected_zero_idx = np.random.choice(zero_idx, len(nonzero_idx))
        for key, val in batch.items():
            batch[key] = val[np.concatenate([selected_zero_idx, nonzero_idx])]
        print(f"size of batch after balancing: {len(batch['labels'])}")

    return batch


def qlearning_ant_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
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

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    goal_ = []
    xy_ = []
    done_bef_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        goal = dataset['infos/goal'][i].astype(np.float32)
        xy = dataset['infos/qpos'][i][:2].astype(np.float32)

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
            next_final_timestep = dataset['timeouts'][i+1]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
            next_final_timestep = (episode_step == env._max_episode_steps - 2)
            
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
        goal_.append(goal)
        xy_.append(xy)
        done_bef_.append(done_bef)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'goals': np.array(goal_),
        'xys': np.array(xy_),
        'dones_bef': np.array(done_bef_)
    }


def qlearning_robosuite_dataset(dataset_path, terminate_on_end=False, **kwargs):
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
    f = h5py.File(dataset_path, 'r')

    # N = dataset['rewards'].shape[0]
    demos = list(f['data'].keys())
    N = len(demos)
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    traj_idx_ = []
    seg_idx_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    # if 'timeouts' in dataset:
    #     use_timeouts = True

    episode_step = 0
    obs_keys = kwargs.get("obs_key", ["object", "robot0_joint_pos", "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel"])
    for ep in tqdm(demos, desc="load robosuite demonstrations"):
        ep_grp = f[f"data/{ep}"]
        traj_len = ep_grp["actions"].shape[0]
        for i in range(traj_len - 1):
            total_obs = ep_grp["obs"]
            obs = np.concatenate([total_obs[key][i].tolist() for key in obs_keys], axis=0)
            new_obs = np.concatenate([total_obs[key][i + 1].tolist() for key in obs_keys], axis=0)
            action = ep_grp["actions"][i]
            reward = ep_grp["rewards"][i]
            done_bool = bool(ep_grp["dones"][i])

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            traj_idx_.append(int(ep[5:]))
            seg_idx_.append(i)

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'env_meta': json.loads(f["data"].attrs["env_args"]),
        'traj_indices': np.array(traj_idx_),
        'seg_indices': np.array(seg_idx_),
    }
