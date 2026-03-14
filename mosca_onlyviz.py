import numpy as np
import torch
import imageio
import sys, os, os.path as osp
import logging
import kornia
from omegaconf import OmegaConf

from lib_prior.prior_loading import Saved2D

from lib_render.render_helper import GS_BACKEND

from lib_moca.camera import MonocularCameras

from lib_mosca.mosca import MoSca
from lib_mosca.dynamic_solver import get_dynamic_curves
from lib_mosca.dynamic_solver import geometry_scf_init
from lib_mosca.photo_recon_utils import OptimCFG, GSControlCFG
from lib_mosca.mosca import MoSca
from lib_mosca.photo_recon import DynReconstructionSolver
from lib_mosca.static_gs import StaticGaussian
from lib_mosca.misc import seed_everything
from lib_mosca.dynamic_solver_utils import (
    round_int_coordinates,
    query_image_buffer_by_pix_int_coord,
)

from mosca_viz import viz_main, viz_list_of_colored_points_in_cam_frame
from mosca_evaluate_ours import test_fps, test_main_ours_v3, test_main_ours_v3_post_stats
from lite_moca_reconstruct import static_reconstruct

from recon_utils import (
    SEED,
    seed_everything,
    setup_recon_ws,
    setup_recon_ours,
    resume_recon_ours, 
    auto_get_depth_dir_tap_mode,
    update_s2d_track_identification,
    viz_mosca_curves_before_optim,
    set_epi_mask_to_s2d_for_bg_render,
)
from lib_mosca.photo_recon_utils import DynamicSegTreeBFS
from recon_utils import parse_interval_from_dir, find_root, find_leaves, imgs_to_video, find_all_renders, find_all_nodes_and_intervals

from multiprocessing import Pool
from functools import partial
from lib_mosca.photo_recon_utils import get_available_gpus
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import copy
import signal
from threading import Event, Lock

from lib_mosca.photo_recon_utils import split_cams
from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
from lib_render.render_helper import render
from PIL import Image

def save_single_image(image_tensor, save_dir, filename, format='png', quality=95):
    """
    保存单个图像
    """
    # 确保是CPU tensor
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()
    
    # 转换为numpy数组
    image_np = image_tensor.numpy()
    
    # 0-1范围，转换为0-255
    image_np = (image_np.clip(0.0, 1.0) * 255).astype(np.uint8)
    
    # 创建保存路径
    if format.lower() == 'npy':
        save_path = os.path.join(save_dir, f"{filename}.npy")
        np.save(save_path, image_np)
    elif format.lower() == 'pt':
        save_path = os.path.join(save_dir, f"{filename}.pt")
        torch.save(image_tensor, save_path)
    else:
        # 图像格式
        save_path = os.path.join(save_dir, f"{filename}.{format}")
        
        # 转换为PIL图像并保存
        pil_image = Image.fromarray(image_np)
        if format.lower() == 'jpg':
            pil_image.save(save_path, 'JPEG', quality=quality)
        else:
            pil_image.save(save_path)
    
    return save_path

def save_composite_results(rendering_dir, pred_rgb_composited, pred_rgb_list, save_individual=True, format='png', quality=100):
    """
    保存合成图像和原始图像列表到指定目录
    
    Args:
        rendering_dir: 保存目录路径
        pred_rgb_composited: 合成图像 (H, W, 3) tensor
        pred_rgb_list: 原始图像列表，每个元素是(H, W, 3) tensor
        transparent_ratio_list: 透明度列表
        save_individual: 是否保存单独的图像
        format: 保存格式 ('png', 'jpg', 'npy', 'pt')
        quality: JPEG质量 (0-100)
    """
    
    # 创建保存目录
    os.makedirs(rendering_dir, exist_ok=True)
    print(f"保存结果到目录: {rendering_dir}")
    
    # 确保图像是tensor格式
    if isinstance(pred_rgb_composited, np.ndarray):
        pred_rgb_composited = torch.from_numpy(pred_rgb_composited)
    
    # 保存合成图像
    composite_save_path = save_single_image(
        pred_rgb_composited, rendering_dir, "composite_image", format, quality
    )
    
    # 保存单独的图像
    individual_paths = []
    if save_individual:
        for i, img in enumerate(pred_rgb_list):
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            
            img_path = save_single_image(
                img, rendering_dir, f"image_{i:03d}", format, quality
            )
            individual_paths.append(img_path)


def create_composite_image(pred_rgb_list, viz_fullscene_idx=0, transparent_ratio_list=None):
    """
    创建合成图像：以全景图为基础，叠加透明前景
    
    Args:
        pred_rgb_list: 图像列表，每个元素是形状为(H, W, 3)的tensor
        viz_fullscene_idx: 全景图的索引（默认为0）
        transparent_ratio_list: 透明度列表，长度应与pred_rgb_list一致
                               如果为None，则使用默认透明度
    
    Returns:
        composite_image: 合成后的图像 (H, W, 3)
        composite_tensor: 合成后的tensor
    """
    
    # 验证输入
    if len(pred_rgb_list) == 0:
        raise ValueError("pred_rgb_list不能为空")
    
    if transparent_ratio_list is None:
        # 默认透明度：从0.3到0.8线性增加
        transparent_ratio_list = [0.3 + 0.5 * (i / max(1, len(pred_rgb_list)-1)) 
                                 for i in range(len(pred_rgb_list))]
    
    if len(transparent_ratio_list) != len(pred_rgb_list):
        raise ValueError("transparent_ratio_list长度必须与pred_rgb_list一致")
    
    # 确保所有图像尺寸一致
    base_shape = pred_rgb_list[viz_fullscene_idx].shape
    for i, img in enumerate(pred_rgb_list):
        if img.shape != base_shape:
            raise ValueError(f"图像{i}的尺寸{img.shape}与基础图像尺寸{base_shape}不一致")
    
    # 初始化合成图像（使用全景图）
    pred_rgb_sum = pred_rgb_list[viz_fullscene_idx].clone()
    
    print(f"使用索引{viz_fullscene_idx}的全景图作为基础图像")
    print(f"图像尺寸: {base_shape}")
    
    # 叠加后续的前景图像
    for idx in range(viz_fullscene_idx + 1, len(pred_rgb_list)):
        current_img = pred_rgb_list[idx].clamp(0.0, 1.0)
        alpha = transparent_ratio_list[idx]
        
        # 创建前景图像的透明度mask
        # 假设白色背景的RGB值为(1.0, 1.0, 1.0)（如果图像值在0-1范围内）
        # 或者(255, 255, 255)（如果图像值在0-255范围内）
        
        # 图像值在0-1范围内
        white_threshold = 0.95
        background_mask = (current_img[:, :, 0] > white_threshold) & \
                            (current_img[:, :, 1] > white_threshold) & \
                            (current_img[:, :, 2] > white_threshold)
        
        # 创建前景mask（非白色区域）
        foreground_mask = ~background_mask
        
        # 将mask转换为float tensor
        foreground_mask_float = foreground_mask.float()
        
        # 应用透明度
        foreground_alpha = foreground_mask_float * alpha

        # 正确的方式：对每个通道分别处理
        for c in range(3):  # 对RGB三个通道分别处理
            # 获取当前通道的数据
            base_channel = pred_rgb_sum[:, :, c]
            current_channel = current_img[:, :, c]
            alpha_channel = foreground_alpha
            
            # 只在前景区域应用混合
            blended_channel = base_channel.clone()
            blended_channel[foreground_mask] = (1 - alpha_channel[foreground_mask]) * base_channel[foreground_mask] + \
                                            alpha_channel[foreground_mask] * current_channel[foreground_mask]
            
            pred_rgb_sum[:, :, c] = blended_channel
                
        print(f"叠加索引{idx}的前景图像，透明度: {alpha:.2f}, 前景像素比例: {foreground_mask_float.mean():.3f}")
    
    return pred_rgb_sum

def adjust_camera_distance(T_cw, scale_factor=2.0):
    """
    通过缩放相机到原点的距离来调整相机位置
    
    Args:
        T_cw: 4x4相机到世界坐标系的变换矩阵
        scale_factor: 距离缩放因子 (>1表示拉远，<1表示拉近)
    
    Returns:
        T_cw_new: 修改后的变换矩阵
    """
    if isinstance(T_cw, torch.Tensor):
        T_cw_np = T_cw.cpu().numpy()
    else:
        T_cw_np = T_cw.copy()
    
    # 相机在世界坐标系中的位置
    R_cw = T_cw_np[:3, :3]
    t_cw = T_cw_np[:3, 3]
    
    # 相机位置 = -R^T * t
    camera_position = -R_cw.T @ t_cw
    
    # 计算相机到原点的方向
    to_origin_direction = -camera_position  # 从相机指向原点
    
    # 归一化方向向量
    if np.linalg.norm(to_origin_direction) > 0:
        to_origin_direction = to_origin_direction / np.linalg.norm(to_origin_direction)
    
    # 计算新的相机位置（沿着到原点的方向移动）
    new_camera_position = camera_position + to_origin_direction * (scale_factor - 1.0) * np.linalg.norm(camera_position)
    
    # 计算新的平移向量: t = -R * camera_position
    t_cw_new = -R_cw @ new_camera_position
    
    # 构建新的变换矩阵
    T_cw_new = np.eye(4)
    T_cw_new[:3, :3] = R_cw
    T_cw_new[:3, 3] = t_cw_new
    
    return torch.from_numpy(T_cw_new).to(T_cw.device)
    
@torch.no_grad()
def ours_viz(
    cfg,
    all_nodes_and_intervals_list, 
    saved_dir_list, 
    data_root,
    device,
    tto_flag,
    eval_also_dyncheck_non_masked=False,
    skip_test_gen=False,
    bound_list=[],  
    node_depth=-1, 
    layer_cams_list=[], 
    rendering_dir=None, 
    use_fachain=False, 
    segtree=[], 
    viz_time_ratios=[0.0, 0.5, 1.0], 
):
    '''
    20251118: only support fix cam
    '''
    full_start_time = all_nodes_and_intervals_list[0]["left_t"]
    full_end_time = all_nodes_and_intervals_list[-1]["right_t"]
    viz_time_list = []
    for viz_time_ratio in viz_time_ratios:
        viz_time = full_start_time + (full_end_time - full_start_time) * viz_time_ratio
        viz_time_list.append(int(viz_time))
    
    # viz_fullscene_idx = viz_time_list[len(viz_time_list) // 2]
    viz_fullscene_idx = 0
    transparent_ratio_list = []
    pred_rgb_list = []
    full_pred_rgb_list = []
    fixed_cam = None
    for viz_time_idx in range(len(viz_time_list)):
        torch.cuda.empty_cache()
        if viz_fullscene_idx == viz_time_idx:
            viz_fullscene_flag = True
            transparent_ratio = 1.0
        else:
            viz_fullscene_flag = False
            transparent_ratio = max(0, 0.5 - abs(viz_fullscene_idx - viz_time_idx) * 0.2)
        transparent_ratio_list.append(transparent_ratio)
        viz_time = viz_time_list[viz_time_idx]
        node_idx = None
        for each_node_interval_idx in range(0, len(all_nodes_and_intervals_list)):
            each_node_interval = all_nodes_and_intervals_list[each_node_interval_idx]
            if each_node_interval["left_t"] > viz_time or viz_time > each_node_interval["right_t"]: continue
            node_idx = each_node_interval_idx
            break
        assert node_idx is not None
        saved_dir = saved_dir_list[node_idx]
        left_bound, right_bound = bound_list[node_idx]
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
        
        if fixed_cam is None and viz_time_idx == 0:
            fixed_cam = copy.deepcopy(cams).to(device)
            fixed_cam_view_idx = viz_time - left_bound

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
        
        fixed_cam.to(device)
        fixed_cam.eval()
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
        working_t = viz_time - left_bound
        H, W = fixed_cam.default_H, fixed_cam.default_W
        cam_K = fixed_cam.default_K
        T_cw = fixed_cam.T_cw(fixed_cam_view_idx)
        T_cw = adjust_camera_distance(T_cw, scale_factor=1.0)
        T_cw = T_cw.float()
        print(cam_K)
        print(T_cw)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            if viz_fullscene_flag:
                gs5 = [s_model(), d_model(working_t)]
            else:
                gs5 = [d_model(working_t)]
        if len(fachain_d_model_list) == 0:
            render_dict = render(gs5, H, W, cam_K, T_cw=T_cw, bg_color=[1.0, 1.0, 1.0])
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
            pred_rgb_list.append(pred_rgb)
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
            render_dict = render(gs5, H, W, cam_K, T_cw=T_cw, bg_color=[1.0, 1.0, 1.0])
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
            pred_rgb_list.append(pred_rgb)
        if not viz_fullscene_flag:
            fullgs5 = [s_model()] + gs5
            render_dict = render(fullgs5, H, W, cam_K, T_cw=T_cw, bg_color=[1.0, 1.0, 1.0])
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
            full_pred_rgb_list.append(pred_rgb)
    pred_rgb_composited = create_composite_image(pred_rgb_list, viz_fullscene_idx=viz_fullscene_idx, transparent_ratio_list=transparent_ratio_list)
    final_pred_rgb_list = pred_rgb_list + full_pred_rgb_list
    save_composite_results(rendering_dir, pred_rgb_composited, final_pred_rgb_list, save_individual=True, format='png')

def move_camera_backward(T_cw, distance):
    """
    将相机沿着lookat反方向移动指定距离
    
    Args:
        T_cw: 4x4相机位姿矩阵（世界到相机）
        distance: 移动距离
    
    Returns:
        T_cw_new: 移动后的4x4位姿矩阵
    """
    # 方法1：直接修改平移部分
    R = T_cw.detach().cpu().numpy()[:3, :3]
    t = T_cw.detach().cpu().numpy()[:3, 3]
    
    # 相机lookat方向（世界坐标系中）
    lookat_dir = R[:, 2]
    
    # 沿着反方向移动
    new_t = t - lookat_dir * distance
    
    T_cw_new = np.eye(4)
    T_cw_new[:3, :3] = R
    T_cw_new[:3, 3] = new_t

    T_cw_new = torch.from_numpy(T_cw_new).float()
    
    return T_cw_new

import torchvision
def ours_make_grid(pred_rgb_list, rows, cols):
    images = pred_rgb_list

    # 确保所有图像维度一致 [C, H, W]
    images_tensor = torch.stack(images).permute(0, 3, 1, 2)  # [rows*cols, C, H, W]

    # 创建grid，nrow指定每行显示多少张图像（即时间序列的长度）
    grid = torchvision.utils.make_grid(
        images_tensor, 
        nrow=cols,           # 每行显示cols张图像
        padding=2,           # 图像间的间距
        normalize=False,     # 是否归一化到[0,1]
        scale_each=False     # 是否单独缩放每张图像
    )

    return grid.permute(1, 2, 0) # [H, W, C]

@torch.no_grad()
def ours_viz_spatiotemporal(
    cfg,
    all_nodes_and_intervals_list, 
    saved_dir_list, 
    data_root,
    device,
    tto_flag,
    eval_also_dyncheck_non_masked=False,
    skip_test_gen=False,
    bound_list=[],  
    node_depth=-1, 
    layer_cams_list=[], 
    rendering_dir=None, 
    use_fachain=False, 
    segtree=[], 
    viz_time_ratios=[0.0, 0.5, 1.0], 
):
    '''
    20251118: cam-time variations
    '''
    full_start_time = all_nodes_and_intervals_list[0]["left_t"]
    full_end_time = all_nodes_and_intervals_list[-1]["right_t"]
    viz_time_list = []
    for viz_time_ratio in viz_time_ratios:
        viz_time = full_start_time + (full_end_time - full_start_time) * viz_time_ratio
        viz_time_list.append(int(viz_time))

    viz_cam_list = viz_time_list

    pred_rgb_list = []
    zoomout_pred_rgb_list = []
    fixed_cam = None
    for viz_cam_idx in range(len(viz_cam_list)):
        viz_cam = viz_cam_list[viz_cam_idx]
        node_idx = None
        for each_node_interval_idx in range(0, len(all_nodes_and_intervals_list)):
            each_node_interval = all_nodes_and_intervals_list[each_node_interval_idx]
            if each_node_interval["left_t"] > viz_cam or viz_cam > each_node_interval["right_t"]: continue
            node_idx = each_node_interval_idx
            break
        assert node_idx is not None
        saved_dir = saved_dir_list[node_idx]
        left_bound, right_bound = bound_list[node_idx]
        fixed_cam = MonocularCameras.load_from_ckpt(
            torch.load(osp.join(saved_dir, "photometric_cam.pth"), map_location=torch.device('cpu'))
        ).to(device)
        fixed_cam_view_idx = viz_cam - left_bound
        H, W = fixed_cam.default_H, fixed_cam.default_W
        cam_K = fixed_cam.default_K
        T_cw = fixed_cam.T_cw(fixed_cam_view_idx)
        zoomout_cam_distance = getattr(cfg, "zoomout_cam_distance", 1.0)
        T_cw_zoomout = move_camera_backward(T_cw, distance=zoomout_cam_distance).to(device)
        for viz_time_idx in range(len(viz_time_list)):
            torch.cuda.empty_cache()
            viz_time = viz_time_list[viz_time_idx]

            node_idx = None
            for each_node_interval_idx in range(0, len(all_nodes_and_intervals_list)):
                each_node_interval = all_nodes_and_intervals_list[each_node_interval_idx]
                if each_node_interval["left_t"] > viz_time or viz_time > each_node_interval["right_t"]: continue
                node_idx = each_node_interval_idx
                break
            assert node_idx is not None
            saved_dir = saved_dir_list[node_idx]
            left_bound, right_bound = bound_list[node_idx]
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
            
            if viz_cam_idx == 0 and viz_time_idx == 0:
                fixed_cam = copy.deepcopy(cams).to(device)

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
            
            fixed_cam.to(device)
            fixed_cam.eval()
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
            working_t = viz_time - left_bound
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                gs5 = [s_model(), d_model(working_t)]
            if len(fachain_d_model_list) == 0:
                render_dict = render(gs5, H, W, cam_K, T_cw=T_cw, bg_color=[1.0, 1.0, 1.0])
                pred_rgb = render_dict["rgb"].permute(1, 2, 0)
                pred_rgb_list.append(pred_rgb)
                render_dict_zoomout = render(gs5, H, W, cam_K, T_cw=T_cw_zoomout, bg_color=[1.0, 1.0, 1.0])
                pred_rgb_zoomout = render_dict_zoomout["rgb"].permute(1, 2, 0)
                zoomout_pred_rgb_list.append(pred_rgb_zoomout)
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
                render_dict = render(gs5, H, W, cam_K, T_cw=T_cw, bg_color=[1.0, 1.0, 1.0])
                pred_rgb = render_dict["rgb"].permute(1, 2, 0)
                pred_rgb_list.append(pred_rgb)
                render_dict_zoomout = render(gs5, H, W, cam_K, T_cw=T_cw_zoomout, bg_color=[1.0, 1.0, 1.0])
                pred_rgb_zoomout = render_dict_zoomout["rgb"].permute(1, 2, 0)
                zoomout_pred_rgb_list.append(pred_rgb_zoomout)

            pred_rgb_save_name = f"cam-{str(viz_cam).zfill(4)}-time-{str(viz_time).zfill(4)}"
            pred_rgb_rendering_dir = os.path.join(rendering_dir, "pred_rgb")
            os.makedirs(pred_rgb_rendering_dir, exist_ok=True)
            pred_rgb_save_path = save_single_image(pred_rgb, pred_rgb_rendering_dir, pred_rgb_save_name, 'png', 100)
            pred_rgb_save_name_zoomout = f"cam-{str(viz_cam).zfill(4)}-time-{str(viz_time).zfill(4)}"
            zoomout_pred_rgb_rendering_dir = os.path.join(rendering_dir, "pred_rgb_zoomout")
            os.makedirs(zoomout_pred_rgb_rendering_dir, exist_ok=True)
            pred_rgb_save_path_zoomout = save_single_image(pred_rgb_zoomout, zoomout_pred_rgb_rendering_dir, pred_rgb_save_name_zoomout, 'png', 100)
    pred_rgb_grid = ours_make_grid(pred_rgb_list, len(viz_cam_list), len(viz_time_list))
    pred_rgb_grid_save_path = save_single_image(pred_rgb_grid, rendering_dir, "pred_rgb_grid", 'png', 100)
    zoomout_pred_rgb_grid = ours_make_grid(zoomout_pred_rgb_list, len(viz_cam_list), len(viz_time_list))
    zoomout_pred_rgb_grid_save_path = save_single_image(zoomout_pred_rgb_grid, rendering_dir, "pred_rgb_zoomout_grid", 'png', 100)
    
def process_layer(process_height, logdir, args, cfg, datamode, gpu_id):
    '''
    need to get:
        all involved nodes list, len = N
        the corresponding time interval of all nodes (list), len = N
    '''
    torch.cuda.set_device(gpu_id)
    node_device = torch.device(f"cuda:{gpu_id}")
    # 设置当前进程使用的GPU
    print(f"gpu-id:{gpu_id} initializing.\n")

    use_fachain = getattr(cfg, "use_fachain", True)
    print(f"gpu-id:{gpu_id}. use fachain: {use_fachain}.\n")
    
    all_cams_list = []

    dir_list = sorted(os.listdir(logdir))
    all_nodename_list = []
    for dir_name in dir_list:
        interval = parse_interval_from_dir(dir_name)
        if interval:
            all_nodename_list.append(dir_name)
    root_name = find_root(all_nodename_list)
    interval = parse_interval_from_dir(root_name)
    assert interval
    start_t, end_t = interval
    segtree = ["None" for i in range(1000)]
    leaves_list = find_leaves(logdir, all_nodename_list, segtree, 1) # NOTE: construct segtree
    all_nodes_and_intervals_list = find_all_nodes_and_intervals(logdir, segtree, 1, process_height, post_cut=getattr(cfg, "min_interval", 12))
    assert all_nodes_and_intervals_list and len(all_nodes_and_intervals_list) > 0

    for idx in range(0, len(all_nodes_and_intervals_list)):
        node_name = all_nodes_and_intervals_list[idx]["node_name"]
        interval = parse_interval_from_dir(node_name)
        assert interval
        left_bound, right_bound = interval
        left_t, right_t = all_nodes_and_intervals_list[idx]["left_t"], all_nodes_and_intervals_list[idx]["right_t"]
        node_cam_dir = os.path.join(logdir, node_name, "photometric_cam.pth")
        if left_bound == left_t and right_bound == right_t:
            node_cam = MonocularCameras.load_from_ckpt(
                torch.load(node_cam_dir, map_location=torch.device('cpu'))
            ).to(node_device)
        else:
            if left_bound == left_t:
                node_cam = split_cams(node_cam_dir, right_t+1, "left", device=node_device)
            elif right_bound == right_t:
                node_cam = split_cams(node_cam_dir, left_t, "right", device=node_device)
            else:
                print(f"gpuid: {gpu_id}. bound: [{left_bound}, {right_bound}]. t: [{left_t}, {right_t}]. no such case.")
                exit(-1)
        all_cams_list.append(node_cam)
    
    # assert check
    for idx in range(0, len(all_nodes_and_intervals_list)):
        node_cam = all_cams_list[idx]
        node_name = all_nodes_and_intervals_list[idx]["node_name"]
        interval = parse_interval_from_dir(node_name)
        assert interval
        left_bound, right_bound = interval
        left_t, right_t = all_nodes_and_intervals_list[idx]["left_t"], all_nodes_and_intervals_list[idx]["right_t"]
        # start assert
        assert node_cam.T == right_t - left_t + 1
        assert left_bound <= left_t and right_t <= right_bound
        if idx != len(all_nodes_and_intervals_list) - 1:
            assert right_t + 1 == all_nodes_and_intervals_list[idx+1]["left_t"]

    rendering_dir = os.path.join(logdir, f"visualization_results_max_height_{str(process_height)}")
    override_viz = getattr(cfg, "override_viz", False)
    if override_viz and os.path.exists(rendering_dir):
        os.system(f"rm -r {rendering_dir}")
    os.makedirs(rendering_dir, exist_ok=True)
    leaf_dir_list = []
    bound_list = []
    for idx in range(0, len(all_nodes_and_intervals_list)):
        node_name = all_nodes_and_intervals_list[idx]["node_name"]
        left_t, right_t = all_nodes_and_intervals_list[idx]["left_t"], all_nodes_and_intervals_list[idx]["right_t"]
        leaf_dir = os.path.join(logdir, node_name)
        leaf_dir_list.append(leaf_dir)
        bound_list.append([left_t, right_t])
    print(f"gpu-id:{gpu_id}: evaluate {rendering_dir}.\n")
    viz_time_ratios = getattr(cfg, "viz_time_ratios", [0.0, 0.5, 1.0])
    # ours_viz(
    ours_viz_spatiotemporal(
        cfg,
        all_nodes_and_intervals_list=all_nodes_and_intervals_list, 
        saved_dir_list=leaf_dir_list,
        data_root=args.ws,
        device=torch.device(f"cuda:{gpu_id}"),
        tto_flag=True,
        eval_also_dyncheck_non_masked=False,
        skip_test_gen=False,
        bound_list=bound_list,
        node_depth=process_height, 
        layer_cams_list=all_cams_list, 
        rendering_dir=rendering_dir, 
        use_fachain=use_fachain, 
        segtree=segtree, 
        viz_time_ratios=viz_time_ratios, 
    )

    print(f"gpu-id:{gpu_id}: done for evaluate layer {process_height}.\n")

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

    logdir = resume_recon_ours(args.model_path)

    dir_list = sorted(os.listdir(logdir))
    nodename_list = []
    for dir_name in dir_list:
        interval = parse_interval_from_dir(dir_name)
        if interval:
            nodename_list.append(dir_name)
    root_name = find_root(nodename_list)
    interval = parse_interval_from_dir(root_name)
    assert interval
    start_t, end_t = interval

    tree_height = getattr(cfg, "stack_depth_max", 3)
    eval_tree_height_start = tree_height

    datamode = getattr(cfg, "mode", "iphone")

    # TODO: height parallel
    gpu_list = get_available_gpus()
    gpu_list = [gpu_id for gpu_id in range(len(gpu_list))]
    for max_tree_height in reversed(range(eval_tree_height_start, tree_height+1, len(gpu_list))):
        try:
            h_start = max_tree_height
            h_end = max_tree_height + len(gpu_list)
            h_end = min(tree_height+1, h_end)
            mx_wks = h_end - h_start
            with ThreadPoolExecutor(max_workers=mx_wks) as executor:
                futures = []
                for process_height in range(h_start, h_end):
                    gpu_id = gpu_list[process_height % len(gpu_list)]
                    futures.append(
                        executor.submit(
                            process_layer, 
                            process_height, logdir, args, cfg, datamode, gpu_id
                        )
                    )
            for future in futures:
                result = future.result()
        except Exception as e:
            torch.cuda.empty_cache()
            raise e
