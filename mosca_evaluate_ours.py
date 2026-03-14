import torch
import os, os.path as osp
import logging
import numpy as np
from tqdm import tqdm
from typing import Literal, Optional, Tuple
from pytorch3d.ops import knn_points

from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian

from eval_utils.eval_nvidia import eval_nvidia_dir
from eval_utils.eval_nvidia import eval_nvidia_ours_dir
from eval_utils.eval_nvidia import eval_nvidia_mask_ours_dir
from eval_utils.eval_dyncheck import eval_dycheck, eval_dycheck_ours

from eval_utils.campose_alignment import align_ate_c2b_use_a2b
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

import imageio
from omegaconf import OmegaConf
from data_utils.iphone_helpers import load_iphone_gt_poses
from data_utils.nvidia_helpers import load_nvidia_gt_pose, get_nvidia_dummy_test
from data_utils.nvidia_helpers import load_nvidia_ours_gt_pose_v3, get_nvidia_ours_dummy_test_v3

from lib_render.render_helper import render, render_cam_pcl
from tqdm import tqdm
import imageio
from matplotlib import pyplot as plt
import cv2 as cv
from lib_render.render_helper import GS_BACKEND
import time
import copy

from recon_utils import (
    SEED,
    seed_everything,
)

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

#########
# test helper
#########


def render_test_tto_ours_v3(
    H,
    W,
    cams: MonocularCameras,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    gt_rgb_dir,
    save_pose_fn,
    ##
    tto_steps=25,
    decay_start=15,
    lr_p=0.003,
    lr_q=0.003,
    lr_final=0.0001,
    ###
    gt_mask_dir=None,
    save_dir=None,
    fn_list=None,
    focal=None,
    cxcy_ratio=None,
    # dbg
    use_sgd=False,
    loss_type="psnr",
    # boost
    initialize_from_previous_camera=True,
    initialize_from_previous_step_factor=10,
    initialize_from_previous_lr_factor=0.1,
    fg_mask_th=0.1,
    all_cams_list=[], 
    all_train_camera_T_wi=None, 
    all_test_camera_T_wi=None, 
    test_camera_pick_mask=None, 
    fachain_d_model_list=[], 
):
    '''
    all_cams_list=[], [MonocularCameras, MonocularCameras, ..., MonocularCameras] (NOTE: some MonocularCameras is supposed to be split before input this func.)
    all_train_camera_T_wi=None, Tensor (NOTE: just simply input the original poses, do not select the child interval)
    all_test_camera_T_wi=None, Tensor (NOTE: just simply input the original poses, do not select the child interval)
    test_camera_pick_mask=None, Tensor (NOTE: the pick mask for child interval)
    '''

    if test_camera_T_wi is None:
        return None

    # * Optimize the test camera pose, nost simply do the global sim(3) alignment
    s_model.eval()
    d_model.eval()
    for param in s_model.parameters():
        param.requires_grad_(False)
    for param in d_model.parameters():
        param.requires_grad_(False)
    for param in d_model.scf.parameters():
        param.requires_grad_(False)
    if len(fachain_d_model_list) > 0:
        for fachain_d_model_infodict in fachain_d_model_list:
            fachain_d_model_infodict["d_model"].eval()
            for param in fachain_d_model_infodict["d_model"].parameters():
                param.requires_grad_(False)
            for param in fachain_d_model_infodict["d_model"].scf.parameters():
                param.requires_grad_(False)

    assert gt_mask_dir is None, "THIS IS NOT CORRECT, SHOULD NOT USE GT MASK DURING TTO"

    device = s_model.device

    if len(all_cams_list) == 0:
        # first align the camera
        with torch.no_grad():
            solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
            aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
                traj_a=train_camera_T_wi,
                traj_b=solved_cam_T_wi.detach().cpu(),
                traj_c=test_camera_T_wi,
            )
    else:
        assert len(all_cams_list) > 0 and (all_train_camera_T_wi is not None) and (all_test_camera_T_wi is not None) and (test_camera_pick_mask is not None)
        # first align the camera
        with torch.no_grad():
            solved_cam_T_wi_list = []
            for each_cam in all_cams_list:
                solved_cam_T_wi_list.extend([each_cam.T_wc(i).clone() for i in range(each_cam.T)])
            solved_cam_T_wi = torch.stack(solved_cam_T_wi_list, 0)
            all_train_camera_T_wi_clone = all_train_camera_T_wi.detach().cpu().clone()
            all_test_camera_T_wi_clone = all_test_camera_T_wi.detach().cpu().clone()
            all_aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
                traj_a=all_train_camera_T_wi_clone,
                traj_b=solved_cam_T_wi.detach().cpu(),
                traj_c=all_test_camera_T_wi_clone,
            )
            aligned_test_camera_T_wi = all_aligned_test_camera_T_wi[test_camera_pick_mask > 0, ...]

    # render
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if focal is None:
        focal = cams.rel_focal
    if cxcy_ratio is None:
        cxcy_ratio = cams.cxcy_ratio

    L = min(H, W)
    fx = focal * L / 2.0
    fy = focal * L / 2.0
    cx = W * cxcy_ratio[0]
    cy = H * cxcy_ratio[1]
    cam_K = torch.eye(3).to(device)
    cam_K[0, 0] = cam_K[0, 0] * 0 + float(fx)
    cam_K[1, 1] = cam_K[1, 1] * 0 + float(fy)
    cam_K[0, 2] = cam_K[0, 2] * 0 + float(cx)
    cam_K[1, 2] = cam_K[1, 2] * 0 + float(cy)

    test_ret = []
    solved_pose_list = []
    for i in tqdm(range(len(test_camera_tid))):
        if initialize_from_previous_camera and i == 0:
            step_factor = initialize_from_previous_step_factor
            lr_factor = 1.0
        else:
            step_factor = 1
            lr_factor = initialize_from_previous_lr_factor

        working_t = test_camera_tid[i]
        # load gt rgb and mask
        gt_rgb = imageio.imread(osp.join(gt_rgb_dir, f"{fn_list[i]}.png")) / 255.0
        gt_rgb = gt_rgb[..., :3]
        if gt_mask_dir is None:
            gt_mask = np.ones_like(gt_rgb[..., 0])
        else:
            raise RuntimeError("Should not use this during TTO!!")
            gt_mask = imageio.imread(osp.join(gt_mask_dir, f"{fn_list[i]}.png")) / 255.0
        gt_rgb = torch.tensor(gt_rgb, device=device).float()
        gt_mask = torch.tensor(gt_mask, device=device).float()
        gt_mask_sum = gt_mask.sum()


        T_cw_init = torch.linalg.inv(aligned_test_camera_T_wi[i]).to(device)
        T_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        t_init = torch.nn.Parameter(T_cw_init[:3, 3].detach())
        q_init = torch.nn.Parameter(matrix_to_quaternion(T_cw_init[:3, :3]).detach())
        if use_sgd:
            optimizer_type = torch.optim.SGD
        else:
            optimizer_type = torch.optim.Adam
        optimizer = optimizer_type(
            [
                {"params": t_init, "lr": lr_p * lr_factor},
                {"params": q_init, "lr": lr_q * lr_factor},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tto_steps * step_factor - decay_start,
            eta_min=lr_final * lr_factor,
        )

        loss_list = []

        for _step in range(tto_steps * step_factor):
            optimizer.zero_grad()
            _T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
            T_cw = torch.cat([_T_cw, T_bottom[None]], 0)
            with torch.no_grad():
                gs5 = [s_model(), d_model(working_t)]  # ! this does not change
            if len(fachain_d_model_list) == 0:
                render_dict = render(gs5, H, W, cam_K, T_cw=T_cw)
            else:
                with torch.no_grad():
                    for fa_d_model_infodict_idx in range(0, len(fachain_d_model_list)):
                        fa_d_model_infodict = fachain_d_model_list[fa_d_model_infodict_idx]
                        ref_time_mask = None
                        if fa_d_model_infodict["do_fachain_ref_time_mask"]:
                            if fa_d_model_infodict_idx == 0:
                                ref_time_mask = torch.logical_or(
                                    fa_d_model_infodict["d_model"].ref_time < fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["left_bound"], 
                                    fa_d_model_infodict["d_model"].ref_time > fa_d_model_infodict["leaf_right_bound"] - fa_d_model_infodict["left_bound"], 
                                )
                            else:
                                ref_time_mask = torch.logical_or(
                                    fa_d_model_infodict["d_model"].ref_time < fachain_d_model_list[fa_d_model_infodict_idx-1]["left_bound"] - fa_d_model_infodict["left_bound"], 
                                    fa_d_model_infodict["d_model"].ref_time > fachain_d_model_list[fa_d_model_infodict_idx-1]["right_bound"] - fa_d_model_infodict["left_bound"], 
                                )
                            ref_time_extend_mask = torch.logical_and(
                                fa_d_model_infodict["d_model"].ref_time >= fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["extend_length"], 
                                fa_d_model_infodict["d_model"].ref_time <= fa_d_model_infodict["leaf_right_bound"] + fa_d_model_infodict["extend_length"], 
                            )
                            ref_time_mask = torch.logical_and(ref_time_extend_mask, ref_time_mask)
                        if ref_time_mask is not None:
                            fa_d_model_infodict["ref_time_mask"] = ref_time_mask
                        if (ref_time_mask is None) or (ref_time_mask is not None and ref_time_mask.sum() > 0):
                            gs5.append(fa_d_model_infodict["d_model"](working_t + fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["left_bound"], ref_time_mask=ref_time_mask))
                render_dict = render(gs5, H, W, cam_K, T_cw=T_cw)
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)

            rendered_mask = render_dict["alpha"].squeeze(-1).squeeze(0) > fg_mask_th

            if loss_type == "abs":
                raise RuntimeError("Should not use this")
                rgb_loss_i = torch.abs(pred_rgb - gt_rgb) * gt_mask[..., None]
                rgb_loss = rgb_loss_i.sum() / gt_mask_sum
            elif loss_type == "psnr":
                mse = ((pred_rgb - gt_rgb) ** 2)[rendered_mask].mean()
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                rgb_loss = -psnr

            else:
                raise ValueError(f"Unknown loss tyoe {loss_type}")

            loss = rgb_loss
            loss.backward()
            optimizer.step()
            if _step >= decay_start:
                scheduler.step()

            loss_list.append(loss.item())
        
        solved_T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
        solved_T_cw = torch.cat([solved_T_cw, T_bottom[None]], 0)
        solved_pose_list.append(solved_T_cw.detach().cpu().numpy())
        with torch.no_grad():

            if len(fachain_d_model_list) == 0:
                render_dict = render(
                    [s_model(), d_model(working_t)], H, W, cam_K, T_cw=T_cw
                )
            else:
                gs_list = [s_model(), d_model(working_t)]
                for fa_d_model_infodict_idx in range(0, len(fachain_d_model_list)):
                    fa_d_model_infodict = fachain_d_model_list[fa_d_model_infodict_idx]
                    ref_time_mask = None
                    if fa_d_model_infodict["do_fachain_ref_time_mask"]:
                        if fa_d_model_infodict_idx == 0:
                            ref_time_mask = torch.logical_or(
                                fa_d_model_infodict["d_model"].ref_time < fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["left_bound"], 
                                fa_d_model_infodict["d_model"].ref_time > fa_d_model_infodict["leaf_right_bound"] - fa_d_model_infodict["left_bound"], 
                            )
                        else:
                            ref_time_mask = torch.logical_or(
                                fa_d_model_infodict["d_model"].ref_time < fachain_d_model_list[fa_d_model_infodict_idx-1]["left_bound"] - fa_d_model_infodict["left_bound"], 
                                fa_d_model_infodict["d_model"].ref_time > fachain_d_model_list[fa_d_model_infodict_idx-1]["right_bound"] - fa_d_model_infodict["left_bound"], 
                            )
                        ref_time_extend_mask = torch.logical_and(
                            fa_d_model_infodict["d_model"].ref_time >= fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["extend_length"], 
                            fa_d_model_infodict["d_model"].ref_time <= fa_d_model_infodict["leaf_right_bound"] + fa_d_model_infodict["extend_length"], 
                        )
                        ref_time_mask = torch.logical_and(ref_time_extend_mask, ref_time_mask)
                    if ref_time_mask is not None:
                        fa_d_model_infodict["ref_time_mask"] = ref_time_mask
                    if (ref_time_mask is None) or (ref_time_mask is not None and ref_time_mask.sum() > 0):
                        gs_list.append(fa_d_model_infodict["d_model"](working_t + fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["left_bound"], ref_time_mask=ref_time_mask))
                render_dict = render(
                    gs_list, H, W, cam_K, T_cw=T_cw
                )
            rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
            rgb = np.clip(rgb, 0, 1)  # ! important
            test_ret.append(rgb)
        if save_dir:
            imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb)
        print(f"TTO {fn_list[i]}: {loss_list[0]:.3f}->{loss_list[-1]:.3f}")
        if initialize_from_previous_camera and i < len(test_camera_tid) - 1:
            aligned_test_camera_T_wi[i + 1] = torch.linalg.inv(solved_T_cw).to(
                aligned_test_camera_T_wi
            )
    if os.path.exists(save_pose_fn):
        before_poses = np.load(save_pose_fn)['poses']
        before_poses_list = [before_poses[before_pose_idx] for before_pose_idx in range(before_poses.shape[0])]
        all_solved_poses_list = before_poses_list.extend(solved_pose_list)
    else:
        all_solved_poses_list = solved_pose_list
    np.savez(save_pose_fn, poses=all_solved_poses_list)
    return test_ret



def test_main_ours_v3(
    cfg,
    saved_dir,
    data_root,
    device,
    tto_flag,
    eval_also_dyncheck_non_masked=False,
    skip_test_gen=False,
    left_bound=-1, 
    right_bound=-1, 
    node_depth=-1, 
    layer_cams_list=[], 
    rendering_dir=None, 
    use_fachain=False, 
    segtree=[], 
):
    '''
    20250625 created: for layer evaluation with global cam alignment.
    20250626 update: add use_fachain, default False
    '''
    # ! this func can be called at the end of running, or run seperately after trained

    # get cfg
    if data_root.endswith("/"):
        data_root = data_root[:-1]
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
        OmegaConf.set_readonly(cfg, True)

    dataset_mode = getattr(cfg, "mode", "iphone")
    # max_sph_order = getattr(cfg, "max_sph_order", 1)
    logging.info(f"Dataset mode: {dataset_mode}")

    ######################################################################
    ######################################################################

    cams = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(saved_dir, "photometric_cam.pth"), map_location=torch.device('cpu'))
    ).to(device)
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_s_model_{GS_BACKEND.lower()}.pth"), 
            map_location=torch.device('cpu'),
        ),
        device=device,
    )
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_d_model_{GS_BACKEND.lower()}.pth"), 
            map_location=torch.device('cpu'),
        ),
        device=device,
    )

    cams.to(device)
    cams.eval()
    d_model.to(device)
    d_model.eval()
    s_model.to(device)
    s_model.eval()

    if left_bound == -1 and right_bound == -1:
        left_bound = 0
        right_bound = cams.T - 1
    if len(layer_cams_list) == 0: # NOTE: namely only the node interval, processed in render_test_tto_ours_v3
        pass
    assert rendering_dir

    fachain_d_model_list = []
    if use_fachain:
        node_idx = None
        for idx in range(1, len(segtree)):
            if segtree[idx] == os.path.basename(saved_dir):
                node_idx = idx
                break
        assert node_idx is not None
        fa_node_idx = node_idx // 2
        while fa_node_idx != 0: # NOTE: 1 at bfs_list is root
            founded_fa_node_name = segtree[fa_node_idx]
            from recon_utils import parse_interval_from_dir
            founded_fa_node_interval = parse_interval_from_dir(founded_fa_node_name)
            assert founded_fa_node_interval
            founded_fa_node_left_bound, founded_fa_node_right_bound = founded_fa_node_interval
            extend_length = getattr(cfg, "extend_length", 0)
            do_fachain_optim = getattr(cfg, "fachain_optim", False)
            do_fachain_ref_time_mask = getattr(cfg, "fachain_ref_time_mask", False)
            founded_fa_d_model_infodict = {
                "d_model_dir": os.path.join(saved_dir, founded_fa_node_name, f"photometric_d_model_{GS_BACKEND.lower()}.pth"), 
                "left_bound": founded_fa_node_left_bound, 
                "right_bound": founded_fa_node_right_bound, 
                "leaf_left_bound": left_bound, 
                "leaf_right_bound": right_bound, 
                "extend_length": extend_length,
                "do_fachain_optim": do_fachain_optim, 
                "do_fachain_ref_time_mask": do_fachain_ref_time_mask, 
            }
            if not os.path.exists(founded_fa_d_model_infodict["d_model_dir"]): break
            fachain_d_model_list.append(founded_fa_d_model_infodict)
            fa_node_idx = fa_node_idx // 2
        if len(fachain_d_model_list) > 0:
            for fa_d_model_infodict in fachain_d_model_list:
                fa_d_model_dir = fa_d_model_infodict["d_model_dir"]
                fa_d_model_infodict["d_model"] = DynSCFGaussian.load_from_ckpt(
                    torch.load(fa_d_model_dir, map_location=torch.device('cpu')),
                    device=device, 
                )
                fa_d_model_infodict["d_model"].to(device)
                fa_d_model_infodict["d_model"].eval()
                for param in fa_d_model_infodict["d_model"].parameters():
                    param.requires_grad_(False)
                for param in fa_d_model_infodict["d_model"].scf.parameters():
                    param.requires_grad_(False)
        fachain_test_length = getattr(cfg, "fachain_test_length", -1)
        if fachain_test_length != -1:
            if fachain_test_length > 0 and fachain_test_length < len(fachain_d_model_list):
                print("TESTING: Set fachain_test_length:", fachain_test_length)
                fachain_d_model_list = fachain_d_model_list[:fachain_test_length]
            else:
                print("Incorrect fachain_test_length:", fachain_test_length)
                exit(-1)
    ######################################################################
    ######################################################################

    render_func = None
    if dataset_mode == "nvidia_ours_v3":
        images_dir = osp.join(data_root, "images")
        images_name_list = sorted(os.listdir(images_dir))
        N_images = len(images_name_list)
        gt_cam_idx_list = getattr(cfg, "gt_cam_idx_list")
        
        # gt_training_cam_T_wi = cams.T_wc_list().detach().cpu()
        # gt_training_fov = cams.fov

        (gt_training_cam_T_wi, gt_training_fov, gt_training_cxcy_ratio) = (
            load_nvidia_ours_gt_pose_v3(
                osp.join(data_root, "poses_bounds.npy"), 
                N_images=N_images, 
            )
        )

        gt_dir = osp.join(data_root, 'gt_images')
        mask_dir = osp.join(data_root, 'sam_results', 'masks')
        (
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_testing_fov_list,
            gt_testing_cxcy_ratio_list,
        ) = get_nvidia_ours_dummy_test_v3(
            gt_training_cam_T_wi, 
            gt_training_fov, 
            gt_dir, 
            osp.join(data_root, "poses_bounds.npy"), 
            N_images=N_images, 
        )

        try:
            all_train_camera_T_wi = copy.deepcopy(gt_training_cam_T_wi)
            all_test_camera_T_wi_list = copy.deepcopy(gt_testing_cam_T_wi_list)
            test_camera_pick_mask_list = [torch.zeros(gt_testing_cam_T_wi_list[0].shape[0])]
            test_camera_pick_mask_list[0][left_bound:right_bound+1] = 1.
            gt_training_cam_T_wi = gt_training_cam_T_wi[left_bound:right_bound+1, ...]
            gt_testing_cam_T_wi_list[0] = gt_testing_cam_T_wi_list[0][left_bound:right_bound+1, ...]
            gt_testing_tids_list[0] = gt_testing_tids_list[0][left_bound:right_bound+1] - left_bound
            gt_testing_fns_list[0] = gt_testing_fns_list[0][left_bound:right_bound+1]
            gt_testing_fov_list = gt_testing_fov_list
            gt_testing_cxcy_ratio_list = gt_testing_cxcy_ratio_list
        except:
            import pdb; pdb.set_trace()

        print(left_bound, right_bound, gt_testing_fns_list)

        # * cfg
        # gt_testing_fov_list[0] = gt_testing_fov_list[0][0] # NOTE: we use the gt pose of training cam, since the test camera 1 is not included in training cameras 2-12
        tto_steps = getattr(cfg, "tto_steps", 30) # NOTE: DEBUG
        decay_start = getattr(cfg, "tto_decay_start", 15)
        lr_p = getattr(cfg, "tto_lr_p", 0.003)
        lr_q = getattr(cfg, "tto_lr_q", 0.003)
        lr_final = getattr(cfg, "tto_lr_final", 0.0001)
        tto_initialize_from_previous_step_factor = 10 # NOTE: DEBUG
        tto_initialize_from_previous_lr_factor = 0.1
        tto_fg_mask_th = 0.1
        sgd_flag = False
        render_func = render_test_tto_ours_v3 # NOTE: the render tto is same to v2, no change
    elif dataset_mode == "iphone_ours_v3":
        (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            _,
            gt_testing_cxcy_ratio_list,
        ) = load_iphone_gt_poses(data_root, getattr(cfg, "t_subsample", 1))
        gt_cam_idx_list = [cam_idx for cam_idx in range(len(gt_testing_cam_T_wi_list))] # NOTE: placeholder
        try:
            all_train_camera_T_wi = copy.deepcopy(gt_training_cam_T_wi)
            all_test_camera_T_wi_list = copy.deepcopy(gt_testing_cam_T_wi_list)
            test_camera_pick_mask_list = []

            gt_training_cam_T_wi = gt_training_cam_T_wi[left_bound:right_bound+1, ...]
            for test_i in range(len(gt_testing_cam_T_wi_list)):
                '''
                size example: iphone apple
                total sequence: [0, 474]
                gt_testing_cam_T_wi_list[test_i]: 0 for 212x4x4, 1 for 320x4x4
                gt_testing_tids_list[test_i]: 0 for list of len=212 (non-uniform list [8, ..., 395]), 1 for list of len=320 (non-uniform list [0, 1, ..., 474])
                '''
                # NOTE: assum all of these are sorted!
                start_value = left_bound
                end_value = right_bound
                none_flag = False
                while not (start_value in gt_testing_tids_list[test_i]):
                    start_value += 1
                    if start_value > gt_testing_tids_list[test_i][-1]:
                        none_flag = True
                        break
                while not (end_value in gt_testing_tids_list[test_i]):
                    end_value -= 1
                    if end_value < gt_testing_tids_list[test_i][0]:
                        none_flag = True
                        break
                if none_flag is False:
                    strat_pos = gt_testing_tids_list[test_i].index(start_value)
                    end_pos = gt_testing_tids_list[test_i].index(end_value)
                    test_camera_pick_mask_list.append(torch.zeros(gt_testing_cam_T_wi_list[test_i].shape[0]))
                    test_camera_pick_mask_list[test_i][strat_pos:end_pos+1] = 1.
                    gt_testing_cam_T_wi_list[test_i] = gt_testing_cam_T_wi_list[test_i][strat_pos:end_pos+1, ...]
                    gt_testing_tids_list[test_i] = gt_testing_tids_list[test_i][strat_pos:end_pos+1]
                    gt_testing_tids_list[test_i] = [each_tid - left_bound for each_tid in gt_testing_tids_list[test_i]]
                    gt_testing_fns_list[test_i] = gt_testing_fns_list[test_i][strat_pos:end_pos+1]
                    gt_testing_fov_list[test_i] = gt_testing_fov_list[test_i] # NOTE: no change
                    gt_testing_cxcy_ratio_list[test_i] = gt_testing_cxcy_ratio_list[test_i] # NOTE: no change
                else:
                    test_camera_pick_mask_list.append(None)
                    gt_testing_cam_T_wi_list[test_i] = None
                    gt_testing_tids_list[test_i] = None
                    gt_testing_fns_list[test_i] = None
                    gt_testing_fov_list[test_i] = None
                    gt_testing_cxcy_ratio_list[test_i] = None
        except:
            import pdb; pdb.set_trace()
        print(left_bound, right_bound, gt_testing_fns_list)
        
        gt_dir = osp.join(data_root, "test_images")
        # * cfg
        tto_steps = getattr(cfg, "tto_steps", 30)
        decay_start = getattr(cfg, "tto_decay_start", 15)
        lr_p = getattr(cfg, "tto_lr_p", 0.003)
        lr_q = getattr(cfg, "tto_lr_q", 0.003)
        lr_final = getattr(cfg, "tto_lr_final", 0.0001)
        sgd_flag = False
        tto_initialize_from_previous_step_factor = 10
        tto_initialize_from_previous_lr_factor = 0.1
        tto_fg_mask_th = 0.1
        render_func = render_test_tto_ours_v3
    else:
        raise ValueError(
            f"Unknown dataset mode: {dataset_mode}, shouldn't call test funcs"
        )
    # id the image size
    sample_fn = [
        f for f in os.listdir(gt_dir) if f.endswith(".png") or f.endswith(".jpg")
    ][0]
    sample = imageio.imread(osp.join(gt_dir, sample_fn))
    H, W = sample.shape[:2]

    ######################################################################
    ######################################################################

    eval_prefix = "tto_" if tto_flag else ""
    all_frames = []
    if not skip_test_gen:
        for test_i in range(len(gt_testing_cam_T_wi_list)):
            if gt_testing_tids_list[test_i] is None: 
                all_frames.append(None)
                continue

            testing_fov = gt_testing_fov_list[test_i]
            testing_focal = 1.0 / np.tan(np.deg2rad(testing_fov) / 2.0)

            if tto_flag:
                if dataset_mode in ['nvidia_ours_v3', 'nerfds_ours_v3']:
                    save_dir = osp.join(rendering_dir, f"{eval_prefix}cam{str(gt_cam_idx_list[test_i]).zfill(3)}_test")
                    save_pose_fn = osp.join(rendering_dir, f"{eval_prefix}cam{str(gt_cam_idx_list[test_i]).zfill(3)}_test_pose_{test_i}")
                elif dataset_mode == 'iphone_ours_v3':
                    save_dir = osp.join(rendering_dir, f"{eval_prefix}test")
                    save_pose_fn = osp.join(rendering_dir, f"tto_test_pose_{test_i}")
                else:
                    raise ValueError(
                        f"Unknown dataset mode: {dataset_mode}, shouldn't call test funcs"
                    )
                frames = render_func(
                    gt_rgb_dir=gt_dir,
                    tto_steps=tto_steps,
                    decay_start=decay_start,
                    lr_p=lr_p,
                    lr_q=lr_q,
                    lr_final=lr_final,
                    use_sgd=sgd_flag,
                    #
                    H=H,
                    W=W,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    train_camera_T_wi=gt_training_cam_T_wi,
                    test_camera_T_wi=gt_testing_cam_T_wi_list[test_i],
                    test_camera_tid=gt_testing_tids_list[test_i],
                    save_dir=save_dir,
                    save_pose_fn=save_pose_fn,
                    fn_list=gt_testing_fns_list[test_i],
                    focal=testing_focal,
                    cxcy_ratio=gt_testing_cxcy_ratio_list[test_i],
                    #
                    initialize_from_previous_camera=True,
                    initialize_from_previous_step_factor=tto_initialize_from_previous_step_factor,
                    initialize_from_previous_lr_factor=tto_initialize_from_previous_lr_factor,
                    fg_mask_th=tto_fg_mask_th,
                    #
                    all_cams_list=layer_cams_list, 
                    all_train_camera_T_wi=all_train_camera_T_wi, 
                    all_test_camera_T_wi=all_test_camera_T_wi_list[test_i], 
                    test_camera_pick_mask=test_camera_pick_mask_list[test_i], 
                    fachain_d_model_list=fachain_d_model_list, 
                )
                all_frames.append(frames)
    return all_frames

def test_main_ours_v3_post_stats(
    cfg,
    data_root,
    device,
    tto_flag,
    eval_also_dyncheck_non_masked=False,
    skip_test_gen=False,
    left_bound=-1, 
    right_bound=-1, 
    layer_cams_list=[], 
    rendering_dir=None, 
    all_frames=None, 
):
    # get cfg
    if data_root.endswith("/"):
        data_root = data_root[:-1]
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
        OmegaConf.set_readonly(cfg, True)

    dataset_mode = getattr(cfg, "mode", "iphone")
    # max_sph_order = getattr(cfg, "max_sph_order", 1)
    logging.info(f"Dataset mode: {dataset_mode}")

    eval_prefix = "tto_" if tto_flag else ""
    # * Test
    if dataset_mode == "nvidia_ours_v3":
        if data_root.endswith("/"):
            data_root = data_root[:-1]
        gt_dir = osp.join(data_root, 'gt_images')
        mask_dir = osp.join(data_root, 'sam_results', 'masks')
        images_dir = osp.join(data_root, "images")
        images_name_list = sorted(os.listdir(images_dir))
        N_images = len(images_name_list)
        gt_cam_idx_list = getattr(cfg, "gt_cam_idx_list")
        for gt_cam_idx in gt_cam_idx_list:
            override_eval = getattr(cfg, "override_eval", True)
            if override_eval or (not osp.exists(osp.join(rendering_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test_report"))):
                eval_nvidia_ours_dir(
                    gt_dir=gt_dir,
                    pred_dir=osp.join(rendering_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test"),
                    report_dir=osp.join(rendering_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test_report"),
                    fixed_view_id=gt_cam_idx,
                    N_images=N_images,
                    left_bound=left_bound, 
                    right_bound=right_bound, 
                )
            override_eval_mask = getattr(cfg, "override_eval_mask", True)
            if osp.exists(mask_dir) and (override_eval_mask or (not osp.exists(osp.join(rendering_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test_report_masked")))):
                eval_nvidia_mask_ours_dir(
                    gt_dir=gt_dir,
                    mask_dir=mask_dir, 
                    pred_dir=osp.join(rendering_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test"),
                    report_dir=osp.join(rendering_dir, f"{eval_prefix}cam{str(gt_cam_idx).zfill(3)}_test_report_masked"),
                    fixed_view_id=gt_cam_idx,
                    N_images=N_images,
                    left_bound=left_bound, 
                    right_bound=right_bound, 
                )
    elif dataset_mode == "iphone_ours_v3":
        gt_dir = osp.join(data_root, "test_images")
        eval_dycheck_ours(
            save_dir=rendering_dir,
            gt_rgb_dir=gt_dir,
            gt_mask_dir=osp.join(data_root, "test_covisible"),
            pred_dir=osp.join(rendering_dir, f"{eval_prefix}test"),
            save_prefix=eval_prefix,
            strict_eval_all_gt_flag=True,  # ! only support full len now!!
            eval_non_masked=eval_also_dyncheck_non_masked,
        )

    logging.info(f"Finished, saved to {rendering_dir}")
    return

def get_leaf_from_bounds(leaves_list, src_t, dst_t):
    src_idx, dst_idx = -1, -1
    leaf_idx = 0
    for each_leaf in leaves_list:
        if each_leaf.left_bound <= src_t <= each_leaf.right_bound:
            if src_idx == -1:
                src_idx = leaf_idx
            else:
                print(f"seems error in leaves_list: src_t:{str(src_t)}, leaf_idx: {str(leaf_idx)}, left-right:{str(each_leaf.left_bound)}-{str(each_leaf.right_bound)}")
                exit(-1)
        if each_leaf.left_bound <= dst_t <= each_leaf.right_bound:
            if dst_idx == -1:
                dst_idx = leaf_idx
            else:
                print(f"seems error in leaves_list: dst_t:{str(dst_t)}, leaf_idx: {str(leaf_idx)}, left-right:{str(each_leaf.left_bound)}-{str(each_leaf.right_bound)}")
                exit(-1)
        leaf_idx += 1
    assert src_idx != -1 and dst_idx != -1
    return src_idx, dst_idx

def trans_pts_to_cam_cross_interval(cams_list, dst_idx, dst_t, pts_w): # NOTE: src_t, dst_t has been minus left_bound
    assert pts_w.shape[-1] == 3  # and pts_w.ndim == 2
    # R, t = self.Rt_cw(dst_t)
    R, t = cams_list[dst_idx].Rt_cw(dst_t)
    original_shape = pts_w.shape
    pts_c = torch.einsum("ij,nj->ni", R, pts_w.reshape(-1, 3)) + t
    return pts_c.reshape(*original_shape)

def project_cross_interval(xyz, cams, th=1e-5):
    # assert xyz.ndim == 2
    assert xyz.shape[-1] == 3
    xy = xyz[..., :2]
    z = xyz[..., 2:]
    z_close_mask = abs(z) < th
    if z_close_mask.any():
        # logging.warning(
        #     f"Projection may create singularity with a point too close to the camera, detected [{z_close_mask.sum()}] points, clamp it"
        # )
        z_close_mask = z_close_mask.float()
        z = (
            z * (1 - z_close_mask) + (1.0 * th) * z_close_mask
        )  # ! always clamp to positive
        assert not (abs(z) < th).any()
    rel_f = torch.as_tensor(cams.rel_focal).to(xyz)
    cxcy = torch.as_tensor(cams.cxcy_ratio).to(xyz) * 2.0 - 1.0
    uv = (xy * rel_f[None] / z) + cxcy[None, :]
    return uv  # [-1,1]


def test_fps(saved_dir, rounds=1, device=torch.device("cuda:0")):
    cams = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(saved_dir, "photometric_cam.pth"), map_location=torch.device('cpu'))
    ).to(device)
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_s_model_{GS_BACKEND.lower()}.pth"), 
            map_location=torch.device('cpu'), 
        ),
        device=device,
    )
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_d_model_{GS_BACKEND.lower()}.pth"), 
            map_location=torch.device('cpu'), 
        ),
        device=device,
    )

    cams.to(device)
    cams.eval()
    d_model.to(device)
    d_model.eval()
    s_model.to(device)
    s_model.eval()

    d_model.set_inference_mode()

    sample_t = [0, cams.T // 2, cams.T - 1]

    s_gs5 = s_model()
    H, W = cams.default_H, cams.default_W
    K = cams.K(H, W)

    viz = []
    for t in sample_t:
        d_gs5 = d_model(t)
        rd = render([s_gs5, d_gs5], H, W, K, T_cw=cams.T_cw(t))
        rd_sample = rd["rgb"].permute(1, 2, 0).cpu().detach().numpy()
        viz.append(rd_sample)
    viz = np.concatenate(viz, 1)
    imageio.imsave(osp.join(saved_dir, "fps_eval_samples.jpg"), viz)

    cnt = cams.T * rounds
    with torch.no_grad():
        start_t = time.time()
        for t in tqdm(range(cnt)):
            t = t % d_model.T
            d_gs5 = d_model(t)
            rd = render([s_gs5, d_gs5], H, W, K, T_cw=cams.T_cw(t))
        end_t = time.time()
    duration = end_t - start_t
    fps = cnt / duration
    logging.info(f"FPS: {fps} tested in rounds {rounds}, rendered {cnt} frames")
    with open(osp.join(saved_dir, "fps_eval.txt"), "w") as fp:
        fp.write(f"FPS: {fps : .10f}\n")
    return


if __name__ == "__main__":
    pass