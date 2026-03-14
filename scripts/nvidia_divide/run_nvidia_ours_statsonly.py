import os
import pandas as pd
import numpy as np
import csv
import yaml
import sys

############################################################################################
# * NVIDIA
############################################################################################

def stats_all():
    exp_setting = sys.argv[1]
    version = sys.argv[2]

    exp_root_dir = f'./exp/nvidia_divide_{version}'
    profile_root_dir = f'./profile/nvidia_divide_{version}'

    if exp_setting == 'native':
        prep_cfg_file = f"{profile_root_dir}/nvidia_stats.yaml"
    else:
        print('No such exp_setting.')
        exit(-1)

    with open(prep_cfg_file) as file:
        optim_cfg_dict = yaml.load(file.read(), Loader=yaml.FullLoader)
        file.close()

    exp_id = optim_cfg_dict['exp_id']
    scene_names = optim_cfg_dict['scene_names']

    exp_dir = os.path.join(exp_root_dir, f'{str(exp_id).zfill(6)}')
    has_mask_results = True

    tree_height = optim_cfg_dict['stack_depth_max']

    scene_metrics = np.zeros((tree_height+1, 3)).astype(np.float32)
    scene_metrics_dict = {}
    if has_mask_results:
        scene_mask_metrics = np.zeros((tree_height+1, 3)).astype(np.float32)
        scene_mask_metrics_dict = {}
    for scene in scene_names:
        scene_dir = os.path.join(exp_dir, exp_setting, scene)
        all_logdirs = sorted(os.listdir(scene_dir))
        latest_idx = -1
        while all_logdirs[latest_idx].endswith(".txt"):
            latest_idx -= 1
        scene_dir = os.path.join(scene_dir, all_logdirs[latest_idx])

        for max_tree_height in range(tree_height+1):
            scene_height_metric_dir = os.path.join(scene_dir, f"rendering_results_max_height_{max_tree_height}", "tto_cam001_test_report", "nvidia_render_metrics.xlsx")
            df = pd.read_excel(scene_height_metric_dir)
            each_metric = [[row['psnr'], row['ssim'], row['lpips']] for index, row in df[df['image_name'] == "all"].iterrows()]
            if max_tree_height in scene_metrics_dict.keys():
                scene_metrics_dict[max_tree_height].append([scene] + each_metric[0])
            else:
                scene_metrics_dict[max_tree_height] = [[scene] + each_metric[0]]
            scene_metrics[max_tree_height] += np.array(each_metric[0])
            if has_mask_results:
                scene_height_mask_metric_dir = os.path.join(scene_dir, f"rendering_results_max_height_{max_tree_height}", "tto_cam001_test_report_masked", "nvidia_render_metrics.xlsx")
                df = pd.read_excel(scene_height_mask_metric_dir)
                each_mask_metric = [[row['mpsnr'], row['mssim'], row['mlpips']] for index, row in df[df['image_name'] == "all"].iterrows()]
                if max_tree_height in scene_mask_metrics_dict.keys():
                    scene_mask_metrics_dict[max_tree_height].append([scene] + each_mask_metric[0])
                else:
                    scene_mask_metrics_dict[max_tree_height] = [[scene] + each_mask_metric[0]]
                scene_mask_metrics[max_tree_height] += np.array(each_mask_metric[0])
    scene_metrics /= len(scene_names)
    if has_mask_results:
        scene_mask_metrics /= len(scene_names)
    for max_tree_height in range(tree_height+1):
        scene_metrics_dict[max_tree_height].append(["average"] + scene_metrics[max_tree_height].tolist())
        if has_mask_results:
            scene_mask_metrics_dict[max_tree_height].append(["average"] + scene_mask_metrics[max_tree_height].tolist())
    for max_tree_height in range(scene_metrics.shape[0]):
        avg_height_metric_dir = os.path.join(exp_dir, exp_setting, f"nvidia_render_metrics_mh_{max_tree_height}.csv")
        all_df = pd.DataFrame(scene_metrics_dict[max_tree_height], columns=["scene", "psnr", "ssim", "lpips"])
        all_df.to_csv(avg_height_metric_dir, index=False)
        if has_mask_results:
            avg_height_mask_metric_dir = os.path.join(exp_dir, exp_setting, f"nvidia_render_metrics_masked_mh_{max_tree_height}.csv")
            all_df = pd.DataFrame(scene_mask_metrics_dict[max_tree_height], columns=["scene", "mpsnr", "mssim", "mlpips"])
            all_df.to_csv(avg_height_mask_metric_dir, index=False)
        

if __name__ == "__main__":
    stats_all()
