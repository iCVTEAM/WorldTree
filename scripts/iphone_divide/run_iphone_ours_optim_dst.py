import os
import pandas as pd
import numpy as np
import csv
import yaml
import sys
import copy

############################################################################################
# * DyCheck
############################################################################################
exp_setting = sys.argv[1]
version = sys.argv[2]
abla_setting = sys.argv[3]

data_root_dir = '<data_dir of iphone dataset>'
exp_root_dir = f'./exp/iphone_divide_{version}'
profile_root_dir = f'./profile/iphone_divide_{version}'

if exp_setting == 'native':
    prep_cfg_file = f"{profile_root_dir}/iphone_fit_dst_{abla_setting}.yaml"
else:
    print('No such exp_setting.')
    exit(-1)

with open(prep_cfg_file) as file:
    optim_cfg_dict = yaml.load(file.read(), Loader=yaml.FullLoader)
    file.close()

GPU_ID = optim_cfg_dict['gpu_id']
exp_id = optim_cfg_dict['exp_id']
skip_training = optim_cfg_dict['skip_training']
skip_eval = optim_cfg_dict['skip_eval']
skip_stats = optim_cfg_dict['skip_stats']
gt_cam_idx_list = optim_cfg_dict['gt_cam_idx_list']
scene_names = optim_cfg_dict['scene_names']
stack_depth_max = optim_cfg_dict['stack_depth_max']

exp_dir = os.path.join(exp_root_dir, f'{str(exp_id).zfill(6)}')

# Optimization
for scene in scene_names:
    print(scene)
    scene_dir = os.path.join(exp_dir, exp_setting, scene)
    for dst_height in range(1, stack_depth_max+1):
        if exp_setting == 'native':
            dst_cfg_file = f"{scene_dir}/iphone_fit_dst_{abla_setting}_h_{dst_height}.yaml"
        else:
            print('No such exp_setting.')
            exit(-1)
        dst_cfg_dist = copy.deepcopy(optim_cfg_dict)
        dst_cfg_dist['stack_depth_max'] = dst_height
        dst_cfg_dist['eval_tree_height_start'] = dst_height
        with open(dst_cfg_file, 'w') as file:
            yaml.dump(dst_cfg_dist, file, default_flow_style=False)
            file.close()
        if not skip_training:
            os.makedirs(scene_dir, exist_ok=True)
            scene_log_dir = os.path.join(scene_dir, "train_log.txt")
            pycode = f"CUDA_VISIBLE_DEVICES={GPU_ID} python mosca_reconstruct.py --ws {data_root_dir}/{scene} --cfg {dst_cfg_file}  --model_path {scene_dir} >> {scene_log_dir} 2>&1"
            rst = os.system(pycode)
            assert rst == 0
        if not skip_eval:
            scene_log_dir = os.path.join(scene_dir, "eval_log.txt")
            pycode = f"CUDA_VISIBLE_DEVICES={GPU_ID} python mosca_onlyeval.py --ws {data_root_dir}/{scene} --cfg {dst_cfg_file}  --model_path {scene_dir} >> {scene_log_dir} 2>&1"
            rst = os.system(pycode)
            assert rst == 0
