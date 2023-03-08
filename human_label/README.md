# Generating your own human preferences
Based on the collected indices for queries in this folder, you could also generate your own real human preferences.

## Generating Videos
First, you have to generate videos for queries by running codes below.
```python
python -m JaxPref.human_label_preprocess_antmaze --env_name {AntMaze env name} --query_path ./human_label --save_dir {video folder to save} --num_query {number of query} --query_len {query length}

python -m JaxPref.human_label_preprocess_mujoco --env_name {Mujoco env name} --query_path ./human_label  --save_dir {video folder to save} --num_query {number of query} --query_len {query length}

python -m JaxPref.human_label_preprocess_robosuite --dataset /mnt/changyeon/ICLR2023_rebuttal/robosuite --dataset_type ph --env {Lift/Can/Square} --use-obs --video_path {video folder to save} --render_image_names agentview_image --indices_path ./human_label/ --query_len {query length} --num_query {number of query}
```

## Labeling Human Preferences
After generating videos, You could use `label_program.ipynb` for collecting human preferences.