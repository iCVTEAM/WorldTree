import os
import yaml
import sys

############################################################################################
# * NVIDIA
############################################################################################
version = sys.argv[1]

data_root_dir = '<data_dir of iphone dataset>'
profile_root_dir = f'./profile/iphone_divide_{version}'

# Preprocess
prep_cfg_file = f"{profile_root_dir}/iphone_prep.yaml"
with open(prep_cfg_file) as file:
    prep_cfg_dict = yaml.load(file.read(), Loader=yaml.FullLoader)
    file.close()
GPU_ID = prep_cfg_dict['gpu_id']
scene_names = prep_cfg_dict['scene_names']
for scene in scene_names:
    logdir = os.path.join(data_root_dir, scene, "preprocess_log.txt")
    pycode = f"CUDA_VISIBLE_DEVICES={GPU_ID} python mosca_precompute.py --ws {data_root_dir}/{scene} --cfg {prep_cfg_file} >> {logdir} 2>&1"
    rst = os.system(pycode)
    assert rst == 0
