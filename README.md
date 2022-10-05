# Preference Transformer: Modeling Human Preferences using Transformers for RL
Implementation of the PT in JAX and FLAX. Our code is based on the implementation of [Flaxmodels](https://github.com/matthias-wright/flaxmodels) and [IQL](https://github.com/ikostrikov/implicit_q_learning). 

## NOTICE
In this version, we replace the human label with the dummy label (all labels are masked with constant 1), so you can only check how our implementation works. We will publicly release the collected real human preferences.
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

### Run Training Reward Model

```python
# Preference Transfomer (PT)
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main --use_human_label True --comment {experiment_name} --transformer.embd_dim 256 --transformer.n_layer 1 --transformer.n_head 4 --env {D4RL env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query {number of query} --query_len 100 --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type PrefTransformer

# Non-Markovian Reward (NMR)
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main --use_human_label True --comment {experiment_name} --env {D4RL env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query 100000 --query_len 25 --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type NMR

# Markovian Reward (MR)
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main --use_human_label True --comment {experiment_name} --env {D4RL env name} --logging.output_dir './logs/pref_reward' --batch_size 256 --num_query 100000 --query_len 25 --n_epochs 10000 --skip_flag 0 --seed {seed} --model_type MR
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
