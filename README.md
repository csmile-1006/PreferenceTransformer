# Preference Transformer: Modeling Human Preferences using Transformers for RL (ICLR 2023)

Official Jax/Flax implementation of **[Preference Transformer: Modeling Human Preferences using Transformers for RL](https://openreview.net/forum?id=Peot1SFDX0)** by [Changyeon Kim*](https://changyeon.page)<sup>,1</sup>, [Jongjin Park*](https://pjj4288.github.io/)<sup>,1</sup>, [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html)<sup>1</sup>, [Honglak Lee](https://web.eecs.umich.edu/~honglak/)<sup>2,3</sup>, [Pieter Abbeel](http://people.eecs.berkeley.edu/~pabbeel/)<sup>4</sup>, [Kimin Lee](https://sites.google.com/view/kiminlee)<sup>5</sup>

<sup>1</sup>KAIST, <sup>2</sup>University of Michigan <sup>3</sup>LG AI Research <sup>4</sup>UC Berkeley <sup>5</sup>Google Research

**TL;DR**: We introduce a transformer-based architecture for preference-based RL considering non-Markovian rewards.

[paper](https://openreview.net/pdf?id=Peot1SFDX0)

<p align="center">
    <img src=figures/arch.png width="900"> 
</p>
Overview of Preference Transformer. We first construct hidden embeddings $\{\mathbf{x}_t\}$ through the causal transformer, where each represents the context information from the initial timestep to timestep $t$. The preference attention layer with a bidirectional self-attention computes the non-Markovian rewards $\{\hat{r}_t\} and their convex combinations $\{z_t \}$ from those hidden embeddings, then we aggregate $\{z_t \}$ for modeling the weighted sum of non-Markovian rewards $\sum_{t}{w_t \hat{r}_t }$.


## NOTICE

In this new version, we release the **real human preference** for various dataset in D4RL and Robosuite.
<!-- replace the human label with the dummy label (all labels are masked with constant 1), so you can only check how our implementation works. We will publicly release the collected real human preferences. -->

## How to run the code

### Install dependencies

```
conda create -y -n offline python=3.8
conda activate offline

pip install --upgrade pip
conda install -y -c conda-forge cudatoolkit=11.1 cudnn=8.2.1
pip install -r requirements.txt
cd d4rl
pip install -e .
cd ..

# Installs the wheel compatible with Cuda 11 and cudnn 8.
pip install "jax[cuda11_cudnn805]>=0.2.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install protobuf==3.20.1 gym==0.18.0 distrax==0.1.2 wandb==0.12.20
pip install transformers
```

## D4RL
### Run Training Reward Model

```python
# Preference Transfomer (PT)
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main --use_human_label True --comment {experiment_name} --transformer.embd_dim 256 --transformer.n_layer 1 --transformer.n_head 4 --env {D4RL env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query {number of query} --query_len 100 --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type PrefTransformer

# Non-Markovian Reward (NMR)
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main --use_human_label True --comment {experiment_name} --env {D4RL env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query {number of query} --query_len 100 --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type NMR

# Markovian Reward (MR)
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main --use_human_label True --comment {experiment_name} --env {D4RL env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query {number of query} --query_len 100 --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type MR
```

### Run IQL with learned Reward Model

```python
# Preference Transfomer (PT)
CUDA_VISIBLE_DEVICES=0 python train_offline.py --seq_len {sequence length in reward prediction} --comment {experiment_name} --eval_interval {5000: mujoco / 100000: antmaze / 50000: adroit} --env_name {d4rl env name} --config {configs/(mujoco|antmaze|adroit)_config.py} --eval_episodes {100 for ant , 10 o.w.} --use_reward_model True --model_type PrefTransformer --ckpt_dir {reward_model_path} --seed {seed}

# Non-Markovian Reward (NMR)
CUDA_VISIBLE_DEVICES=0 python train_offline.py --seq_len {sequence length in reward prediction} --comment {experiment_name} --eval_interval {5000: mujoco / 100000: antmaze / 50000: adroit} --env_name {d4rl env name} --config {configs/(mujoco|antmaze|adroit)_config.py} --eval_episodes {100 for ant , 10 o.w.} --use_reward_model True --model_type NMR --ckpt_dir {reward_model_path} --seed {seed}

# Markovian Reward (MR)
CUDA_VISIBLE_DEVICES=0 python train_offline.py --comment {experiment_name} --eval_interval {5000: mujoco / 100000: antmaze / 50000: adroit} --env_name {d4rl env name} --config {configs/(mujoco|antmaze|adroit)_config.py} --eval_episodes {100 for ant , 10 o.w.} --use_reward_model True --model_type MR --ckpt_dir {reward_model_path} --seed {seed}
```

## Robosuite

### Preliminaries
You must download the robomimic (https://robomimic.github.io/) dataset. <br/>
Please refer to this website: https://robomimic.github.io/docs/datasets/robomimic_v0.1.html
### Run Training Reward Model

```bash
# Preference Transfomer (PT)
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main --use_human_label True --comment {experiment_name} --robosuite True --robosuite_dataset_type {dataset_type} --robosuite_dataset_path {path for robomimic demonstrations} --transformer.embd_dim 256 --transformer.n_layer 1 --transformer.n_head 4 --env {Robosuite env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query {number of query} --query_len {100|50} --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type PrefTransformer

# Non-Markovian Reward (NMR)
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main --use_human_label True --comment {experiment_name} --robosuite True --robosuite_dataset_type {dataset_type} --robosuite_dataset_path {path for robomimic demonstrations} --env {Robosuite env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query {number of query} --query_len {100|50} --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type NMR

# Markovian Reward (MR)
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main --use_human_label True --comment {experiment_name} --robosuite True --robosuite_dataset_type {dataset_type} --robosuite_dataset_path {path for robomimic demonstrations} --env {Robosuite env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query 100000 --query_len {100|50} --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type MR
```

### Run IQL with learned Reward Model

```bash
# Preference Transfomer (PT)
CUDA_VISIBLE_DEVICES=0 python robosuite_train_offline.py --seq_len {sequence length in reward prediction} --comment {experiment_name} --eval_interval 100000 --env_name {Robosuite env name} --robosuite_dataset_type {ph|mh} --robosuite_dataset_path {path for robomimic demonstrations} --config configs/adroit_config.py --eval_episodes 10 --use_reward_model True --model_type PrefTransformer --ckpt_dir {reward_model_path} --seed {seed}

# Non-Markovian Reward (NMR)
CUDA_VISIBLE_DEVICES=0 python robosuite_train_offline.py --seq_len {sequence length in reward prediction} --comment {experiment_name} --eval_interval 100000 --env_name {Robosuite env name} --robosuite_dataset_type {ph|mh} --robosuite_dataset_path {path for robomimic demonstrations} --config configs/adroit_config.py --eval_episodes 10 --use_reward_model True --model_type NMR --ckpt_dir {reward_model_path} --seed {seed}

# Markovian Reward (MR)
CUDA_VISIBLE_DEVICES=0 python robosuite_train_offline.py --comment {experiment_name} --eval_interval 100000 --env_name {Robosuite env name} --robosuite_dataset_type {ph|mh} --robosuite_dataset_path {path for robomimic demonstrations} --config configs/adroit_config.py --eval_episodes 10 --use_reward_model True --model_type MR --ckpt_dir {reward_model_path} --seed {seed}
```

## Citation

```
@inproceedings{
kim2023preference,
title={Preference Transformer: Modeling Human Preferences using Transformers for {RL}},
author={Changyeon Kim and Jongjin Park and Jinwoo Shin and Honglak Lee and Pieter Abbeel and Kimin Lee},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=Peot1SFDX0}
}
```

## Acknowledgments

Our code is based on the implementation of [Flaxmodels](https://github.com/matthias-wright/flaxmodels) and [IQL](https://github.com/ikostrikov/implicit_q_learning). 