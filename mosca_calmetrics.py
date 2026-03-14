import numpy as np
import torch
import sys, os, os.path as osp
from omegaconf import OmegaConf

from eval_utils.eval_nvidia import eval_nvidia_ours_dir
from eval_utils.eval_nvidia import eval_nvidia_mask_ours_dir

from lib_mosca.photo_recon_utils import DynamicSegTreeBFS
from eval_utils.eval_dyncheck import eval_dycheck

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoSca-V2 Reconstruction")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--model_path", type=str, help="Model folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument("--no_viz", action="store_true", help="no viz")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # logdir = setup_recon_ours(args.model_path, fit_cfg=cfg)
    all_logdirs = sorted(os.listdir(args.model_path))
    latest_idx = -1
    while all_logdirs[latest_idx].endswith(".txt"):
        latest_idx -= 1
    logdir = os.path.join(args.model_path, all_logdirs[latest_idx])

    import re
    def parse_interval_from_dir(dir_name):
        """从目录名解析L和R的值，例如dst_L100_R200 -> (100, 200)"""
        match = re.match(r'dst_L(\d+)_R(\d+)', dir_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None
    file_list = sorted(os.listdir(logdir))
    leaves_list = []
    for filename in file_list:
        interval = parse_interval_from_dir(filename)
        if interval:
            leaves_list.append(
                DynamicSegTreeBFS(
                    left_bound=interval[0], 
                    right_bound=interval[1], 
                    depth=0, 
                )
            )

    for each_leaf in leaves_list:
        left_bound = each_leaf.left_bound
        right_bound = each_leaf.right_bound
        leaf_dir = os.path.join(logdir, f"dst_L{str(left_bound).zfill(3)}_R{str(right_bound).zfill(3)}")
        print(leaf_dir)
        # * EVAL AND VIZ
        datamode = getattr(cfg, "mode", "iphone")

        if datamode in ["nvidia_ours_v3"]:
            eval_prefix = "tto_"
            if args.ws.endswith("/"):
                data_root = args.ws[:-1]
            else:
                data_root = args.ws
            images_dir = osp.join(data_root, "images")
            images_name_list = sorted(os.listdir(images_dir))
            N_images = len(images_name_list)
            gt_cam_idx_list = getattr(cfg, "gt_cam_idx_list")
            gt_dir = osp.join(data_root, 'gt_images')
            mask_dir = osp.join(data_root, 'sam_results', 'masks')
            for gt_cam_idx in gt_cam_idx_list:
                override_eval = getattr(args, "override_eval", True)
                if override_eval or (not osp.exists(osp.join(leaf_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test_report"))):
                    eval_nvidia_ours_dir(
                        gt_dir=gt_dir,
                        pred_dir=osp.join(leaf_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test"),
                        report_dir=osp.join(leaf_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test_report"),
                        fixed_view_id=gt_cam_idx,
                        N_images=N_images,
                        left_bound=left_bound, 
                        right_bound=right_bound, 
                    )
                override_eval_mask = getattr(args, "override_eval_mask", True)
                if osp.exists(mask_dir) and (override_eval_mask or (not osp.exists(osp.join(leaf_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test_report_masked")))):
                    eval_nvidia_mask_ours_dir(
                        gt_dir=gt_dir,
                        mask_dir=mask_dir, 
                        pred_dir=osp.join(leaf_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test"),
                        report_dir=osp.join(leaf_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test_report_masked"),
                        fixed_view_id=gt_cam_idx,
                        N_images=N_images,
                        left_bound=left_bound, 
                        right_bound=right_bound, 
                    )
        elif datamode == "iphone_ours_v3":
            eval_prefix = "tto_"
            if args.ws.endswith("/"):
                data_root = args.ws[:-1]
            else:
                data_root = args.ws
            gt_dir = osp.join(data_root, "test_images")
            gt_cam_idx_list = [1] # NOTE: placeholder; NOTE: hard-code
            eval_also_dyncheck_non_masked = False # NOTE: hard-code
            for gt_cam_idx in gt_cam_idx_list:
                override_eval = getattr(args, "override_eval", True)
                if override_eval or (not osp.exists(osp.join(leaf_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test_report"))):
                    eval_dycheck(
                        save_dir=leaf_dir,
                        gt_rgb_dir=gt_dir,
                        gt_mask_dir=osp.join(data_root, "test_covisible"),
                        pred_dir=osp.join(leaf_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test"),
                        save_prefix=eval_prefix,
                        strict_eval_all_gt_flag=True,  # ! only support full len now!!
                        eval_non_masked=eval_also_dyncheck_non_masked,
                    )
        else:
            print('Not support these datamode!')
            exit(-1)
