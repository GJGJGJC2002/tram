"""
OAR 窗口级 Debug 可视化（基于 viser）

加载 pipeline 缓存的 OAR 前中间结果（PipelineData），对指定窗口复刻：
  1. 原始 pose + transl（对齐前）
  2. pp_static_joint 滑步消除后
  3. Depth Contact IK 深度穿透修正后
三组 SMPL mesh + 深度图点云，在 viser 中交互式可视化。

不影响推理流程，所有处理独立完成。

用法:
    python lib/scripts/viser_oar_window_debug.py \
        --result-dir results/promptbase_video_depthrefine_emdb2 \
        --sequence 56_outdoor_stairs_up_down \
        --center-frame 220 \
        --port 8080
"""

import os
import sys
import gc
import time
import gzip
import pickle
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GVHMR_ROOT = os.path.join(PROJECT_ROOT, 'thirdparty', 'GVHMR')
if GVHMR_ROOT not in sys.path:
    sys.path.insert(0, GVHMR_ROOT)

PROMPTHMR_ROOT = os.path.join(PROJECT_ROOT, 'thirdparty', 'PromptHMR')


# ========== 颜色 ==========
COLOR_ORIGINAL = (200, 200, 200)   # 灰色：原始
COLOR_SKATING  = (80, 200, 130)    # 绿色：滑步消除后
COLOR_CONTACT  = (80, 130, 230)    # 蓝色：深度穿透修正后
COLOR_POINTCLOUD = (180, 160, 140) # 点云色


# ========== SMPL 模型加载 ==========

_smplx_model = None
_smplx2smpl = None
_smpl_faces = None


def _ensure_smplx_model(device='cuda'):
    global _smplx_model, _smplx2smpl, _smpl_faces
    if _smplx_model is not None:
        return

    old_cwd = os.getcwd()
    os.chdir(GVHMR_ROOT)
    try:
        from hmr4d.utils.smplx_utils import make_smplx
        print("[Model] Loading SMPL-X (supermotion) + smplx2smpl...")
        _smplx_model = make_smplx("supermotion").to(device).eval()
        _smplx2smpl = torch.load(
            os.path.join(GVHMR_ROOT, "hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        ).to(device)
        _smpl_faces = make_smplx("smpl").faces
        print("[Model] Loaded.")
    finally:
        os.chdir(old_cwd)


def compute_world_vertices(global_orient_w, body_pose_aa, betas, global_trans, device='cuda'):
    """计算世界坐标系 SMPL vertices。返回 (F, V, 3) CPU tensor。"""
    _ensure_smplx_model(device)
    with torch.no_grad():
        out = _smplx_model(
            global_orient=global_orient_w.to(device),
            body_pose=body_pose_aa.to(device),
            betas=betas.to(device),
            transl=global_trans.to(device),
        )
        verts = torch.stack([_smplx2smpl @ v for v in out.vertices])
    return verts.cpu()


# ========== EnDecoder (FK) ==========

_endecoder = None


def _ensure_endecoder(device='cuda'):
    global _endecoder
    if _endecoder is not None:
        return _endecoder

    gvhmr_abs = os.path.abspath(GVHMR_ROOT)
    if gvhmr_abs not in sys.path:
        sys.path.insert(0, gvhmr_abs)

    # 清除可能由 PromptHMR 版本占据的 hmr4d 模块缓存
    hmr4d_modules = [k for k in sys.modules if k.startswith('hmr4d')]
    for mod_name in hmr4d_modules:
        del sys.modules[mod_name]

    # 确保 gvhmr_root 在 sys.path 最前面
    if sys.path[0] != gvhmr_abs:
        if gvhmr_abs in sys.path:
            sys.path.remove(gvhmr_abs)
        sys.path.insert(0, gvhmr_abs)

    old_cwd = os.getcwd()
    os.chdir(gvhmr_abs)
    try:
        from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
        _endecoder = EnDecoder().to(device)
    finally:
        os.chdir(old_cwd)
    return _endecoder


# ========== pp_static_joint ==========

def apply_pp_static_joint(global_orient_w, body_pose_aa, betas, global_trans,
                          static_conf_logits, center_idx, device='cuda'):
    """复刻 pp_static_joint 滑步修正（只改 transl）。"""
    from hmr4d.model.gvhmr.utils.postprocess import pp_static_joint

    endecoder = _ensure_endecoder(device)
    outputs = {
        "pred_smpl_params_global": {
            "global_orient": global_orient_w.unsqueeze(0).to(device),
            "body_pose": body_pose_aa.unsqueeze(0).to(device),
            "betas": betas.unsqueeze(0).to(device),
            "transl": global_trans.unsqueeze(0).to(device),
        },
        "static_conf_logits": (
            static_conf_logits.unsqueeze(0).to(device)
            if static_conf_logits is not None
            else torch.zeros(1, global_orient_w.shape[0], 6, device=device)
        ),
    }
    post_w_transl = pp_static_joint(outputs, endecoder)  # (1, F, 3)
    result = post_w_transl[0].cpu()

    # 以 center_idx 为锚点
    anchor_offset = global_trans[center_idx] - result[center_idx]
    result = result + anchor_offset
    return result


# ========== Depth Contact IK ==========

def gmof(x, sigma=100):
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def apply_depth_contact_ik(
    global_orient_w, body_pose_aa, betas, win_transl,
    static_conf_logits, depth_map, R_cw, t_cw, K,
    center, start, end, device='cuda',
    opt_steps=30, lr=0.003,
    loss_contact_w=5.0, loss_contact_sigma=0.05,
    loss_bp_reg_w=0.1, loss_smooth_w=0.5,
):
    """复刻 _apply_depth_contact_ik。

    Returns:
        (refined_body_pose, scale_factor): scale_factor = median(z_scene / z_smpl)
    """
    endecoder = _ensure_endecoder(device)

    win_len = end - start
    center_local = center - start
    pen_eps = 0.03

    CONTACT_JOINTS = {0: 7, 1: 10, 2: 8, 3: 11}
    LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
    leg_bp_indices = []
    for j in LEG_JOINTS:
        leg_bp_indices.extend([(j-1)*3, (j-1)*3+1, (j-1)*3+2])

    if static_conf_logits is not None:
        static_conf = torch.sigmoid(static_conf_logits[start:end].to(device))
    else:
        static_conf = torch.zeros(win_len, 6, device=device)

    win_orient = global_orient_w[start:end].to(device)
    win_betas = betas[start:end].to(device)
    win_bp = body_pose_aa.to(device)
    win_tr = win_transl.to(device)

    R_center = R_cw[center].to(device)
    t_center = t_cw[center].to(device)
    K_dev = K.to(device)
    K_inv = torch.inverse(K_dev)
    depth_dev = depth_map.to(device)
    dH, dW = depth_dev.shape

    # Phase 1: scale factor
    with torch.no_grad():
        joints_world_init = endecoder.fk_v2(
            body_pose=win_bp.unsqueeze(0),
            betas=win_betas.unsqueeze(0),
            global_orient=win_orient.unsqueeze(0),
            transl=win_tr.unsqueeze(0),
        )[0]

    j_center_world = joints_world_init[center_local]
    j_center_cam = (R_center @ j_center_world.T).T + t_center
    pj_center = (K_dev @ j_center_cam.T).T
    pj_center_2d = pj_center[:, :2] / (pj_center[:, 2:3] + 1e-6)

    z_smpl_list, z_scene_list = [], []
    for jid in range(22):
        z_s = j_center_cam[jid, 2].item()
        if z_s <= 0.01:
            continue
        u_px = int(round(pj_center_2d[jid, 0].item()))
        v_px = int(round(pj_center_2d[jid, 1].item()))
        if u_px < 0 or u_px >= dW or v_px < 0 or v_px >= dH:
            continue
        z_d = depth_dev[v_px, u_px].item()
        if z_d < 0.1 or z_d > 100.0:
            continue
        z_smpl_list.append(z_s)
        z_scene_list.append(z_d)

    if len(z_smpl_list) < 3:
        print(f"[ContactIK] Window {center}: too few joints for scale, skipping")
        return body_pose_aa, 1.0

    scale_factor = float(np.median(np.array(z_scene_list) / (np.array(z_smpl_list) + 1e-8)))
    print(f"[ContactIK] Window {center}: scale_factor={scale_factor:.4f}")

    # Phase 2: find contact targets
    contact_targets = {}
    n_targets = 0

    for fi in range(win_len):
        for conf_idx, jid in CONTACT_JOINTS.items():
            conf_val = static_conf[fi, conf_idx].item()
            if conf_val < 0.5:
                continue
            j_world = joints_world_init[fi, jid]
            j_cam = R_center @ j_world + t_center
            z_smpl_aligned = j_cam[2].item() * scale_factor

            pj = K_dev @ j_cam
            u = pj[0].item() / (pj[2].item() + 1e-6)
            v = pj[1].item() / (pj[2].item() + 1e-6)
            ui, vi = int(round(u)), int(round(v))
            if ui < 0 or ui >= dW or vi < 0 or vi >= dH:
                continue
            z_scene = depth_dev[vi, ui].item()
            if z_scene < 0.1 or z_scene > 100.0:
                continue
            penetration = z_smpl_aligned - z_scene - pen_eps
            if penetration <= 0:
                continue

            z_target_smpl_scale = z_scene / scale_factor
            pixel_homo = torch.tensor([u, v, 1.0], device=device)
            p_target_cam = z_target_smpl_scale * (K_inv @ pixel_homo)
            p_target_world = R_center.T @ (p_target_cam - t_center)
            contact_targets.setdefault(fi, {})[jid] = p_target_world.detach()
            n_targets += 1

    if n_targets == 0:
        print(f"[ContactIK] Window {center}: no penetrating contacts, skipping")
        return body_pose_aa, scale_factor

    print(f"[ContactIK] Window {center}: {n_targets} targets, optimizing {opt_steps} steps")

    # Phase 3: Adam optimize
    bp_orig = win_bp.clone().detach()
    delta_bp = torch.zeros(win_len, len(leg_bp_indices), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_bp], lr=lr)

    for step in range(opt_steps):
        optimizer.zero_grad()
        bp_mod = bp_orig.clone()
        bp_mod[:, leg_bp_indices] = bp_orig[:, leg_bp_indices] + delta_bp

        with torch.enable_grad():
            joints_world_opt = endecoder.fk_v2(
                body_pose=bp_mod.unsqueeze(0),
                betas=win_betas.unsqueeze(0),
                global_orient=win_orient.unsqueeze(0),
                transl=win_tr.unsqueeze(0),
            )[0]

        loss_contact = torch.tensor(0.0, device=device)
        cnt = 0
        for fi, targets in contact_targets.items():
            for jid, target_pos in targets.items():
                diff = joints_world_opt[fi, jid] - target_pos
                loss_contact = loss_contact + gmof(diff, sigma=loss_contact_sigma).sum()
                cnt += 1
        if cnt > 0:
            loss_contact = loss_contact / cnt

        loss_reg = delta_bp.pow(2).mean()
        loss_smooth = torch.tensor(0.0, device=device)
        if win_len >= 2:
            loss_smooth = (delta_bp[1:] - delta_bp[:-1]).pow(2).mean()

        total_loss = loss_contact_w * loss_contact + loss_bp_reg_w * loss_reg + loss_smooth_w * loss_smooth
        total_loss.backward()
        optimizer.step()

        if step == 0 or step == opt_steps - 1:
            print(f"  [step={step}] contact={loss_contact.item():.6f} reg={loss_reg.item():.6f} "
                  f"smooth={loss_smooth.item():.6f} total={total_loss.item():.6f}")

    with torch.no_grad():
        refined_bp = bp_orig.clone()
        refined_bp[:, leg_bp_indices] = bp_orig[:, leg_bp_indices] + delta_bp.detach()

    return refined_bp.cpu(), scale_factor


# ========== Metric3D depth inference ==========

def infer_depth(image_path, intrinsics_np, metric3d_model, model_version='metric3d_vit_small'):
    """推理单帧深度图，返回 (H, W) CPU float tensor。"""
    cam_dir = os.path.join(PROMPTHMR_ROOT, 'pipeline', 'camera')
    if cam_dir not in sys.path:
        sys.path.insert(0, cam_dir)
    from depth_utils import prep_metric3d, post_metric3d

    fx, fy = intrinsics_np[0, 0], intrinsics_np[1, 1]
    cx, cy = intrinsics_np[0, 2], intrinsics_np[1, 2]
    calib = [fx, fy, cx, cy]

    img = np.array(Image.open(image_path).convert('RGB'))
    rgb_prep, intrinsic_prep, pad_info, rgb_origin = prep_metric3d(img, calib, model_version)
    rgb_batch = rgb_prep.cuda().half()

    with torch.inference_mode():
        pred_depth, confidence, _ = metric3d_model.inference({'input': rgb_batch})

    depth = post_metric3d(pred_depth, confidence if confidence is not None else None,
                          pad_info, rgb_origin, intrinsic_prep)
    depth_cpu = depth.cpu().squeeze()
    del rgb_batch, pred_depth, confidence
    torch.cuda.empty_cache()
    return depth_cpu


def load_metric3d(model_version='metric3d_vit_small'):
    """加载 Metric3D 模型（通过 hubconf pretrain=True 自动下载权重）。"""
    import copy
    import types

    metric3d_root = os.path.join(PROMPTHMR_ROOT, 'pipeline', 'yvanyin_metric3d_main')
    prompthmr_abs = os.path.abspath(PROMPTHMR_ROOT)
    for p in [metric3d_root, prompthmr_abs]:
        if p not in sys.path:
            sys.path.insert(0, p)
    cam_dir = os.path.join(prompthmr_abs, 'pipeline', 'camera')
    if cam_dir not in sys.path:
        sys.path.insert(0, cam_dir)

    old_cwd = os.getcwd()
    os.chdir(metric3d_root)

    _deepcopy_dispatch = copy._deepcopy_dispatch
    _had_module = types.ModuleType in _deepcopy_dispatch
    _old_module_copier = _deepcopy_dispatch.get(types.ModuleType)
    _deepcopy_dispatch[types.ModuleType] = lambda x, memo: x

    try:
        from hubconf import metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2
        model_fn = {
            'metric3d_vit_small': metric3d_vit_small,
            'metric3d_vit_large': metric3d_vit_large,
            'metric3d_vit_giant2': metric3d_vit_giant2,
        }[model_version]
        model = model_fn(pretrain=True)
        model = model.cuda().half().eval()
        print(f"[Metric3D] Loaded {model_version}")
    finally:
        if _had_module:
            _deepcopy_dispatch[types.ModuleType] = _old_module_copier
        else:
            _deepcopy_dispatch.pop(types.ModuleType, None)
        os.chdir(old_cwd)
    return model


# ========== 深度图 → 点云 ==========

def depth_to_pointcloud(depth_map, K, R_cw_center, t_cw_center,
                        max_depth=20.0, subsample=4, rgb_image=None,
                        body_mask=None):
    """将深度图转换为世界坐标系点云。

    Args:
        depth_map: (H, W) tensor
        K: (3, 3) intrinsics
        R_cw_center: (3, 3) 中心帧 w2c rotation
        t_cw_center: (3,) 中心帧 w2c translation
        max_depth: 最大深度截断
        subsample: 采样步长（加速）
        rgb_image: (H, W, 3) np.ndarray uint8，若提供则用图片真实颜色
        body_mask: (H, W) np.ndarray bool/uint8，True 表示人体区域（将被排除）

    Returns:
        points_world: (N, 3) np.ndarray
        colors: (N, 3) np.ndarray (0-255 uint8)
    """
    H, W = depth_map.shape
    depth_np = depth_map.numpy() if isinstance(depth_map, torch.Tensor) else depth_map

    # 生成像素网格
    vs, us = np.mgrid[0:H:subsample, 0:W:subsample]
    us = us.flatten().astype(np.float32)
    vs = vs.flatten().astype(np.float32)
    ds = depth_np[::subsample, ::subsample].flatten()

    # 过滤无效深度
    valid = (ds > 0.1) & (ds < max_depth)

    # 人体 mask 过滤
    if body_mask is not None:
        mask_sub = body_mask[::subsample, ::subsample].flatten()
        valid = valid & (mask_sub == 0)

    us, vs, ds = us[valid], vs[valid], ds[valid]

    # 反投影到相机坐标系
    K_np = K.numpy() if isinstance(K, torch.Tensor) else K
    fx, fy, cx, cy = K_np[0, 0], K_np[1, 1], K_np[0, 2], K_np[1, 2]
    x_cam = (us - cx) * ds / fx
    y_cam = (vs - cy) * ds / fy
    z_cam = ds
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # 相机 → 世界：p_world = R_cw^T @ (p_cam - t_cw)
    R_np = R_cw_center.numpy() if isinstance(R_cw_center, torch.Tensor) else R_cw_center
    t_np = t_cw_center.numpy() if isinstance(t_cw_center, torch.Tensor) else t_cw_center
    pts_world = (pts_cam - t_np[None, :]) @ R_np  # R_cw^T = R_cw.T, 行向量乘

    # 颜色：优先用 RGB 图片采样，否则用深度伪彩
    if rgb_image is not None:
        img_h, img_w = rgb_image.shape[:2]
        ui = np.clip(np.round(us).astype(int), 0, img_w - 1)
        vi = np.clip(np.round(vs).astype(int), 0, img_h - 1)
        colors = rgb_image[vi, ui].astype(np.uint8)  # (N, 3)
    else:
        d_norm = np.clip((ds - ds.min()) / (ds.max() - ds.min() + 1e-6), 0, 1)
        colors = np.zeros((len(ds), 3), dtype=np.uint8)
        colors[:, 0] = (d_norm * 200 + 55).astype(np.uint8)
        colors[:, 1] = ((1 - d_norm) * 100 + 100).astype(np.uint8)
        colors[:, 2] = ((1 - d_norm) * 200 + 55).astype(np.uint8)

    return pts_world.astype(np.float32), colors


def scale_vertices_to_depth(vertices, scale_factor, R_cw_center, t_cw_center):
    """以相机原点为锚点，缩放 SMPL vertices 到深度图尺度。

    在相机坐标系下做缩放：
        v_cam = R_cw @ v_world + t_cw
        v_cam_scaled = v_cam * scale_factor
        v_world_scaled = R_cw^T @ (v_cam_scaled - t_cw)

    Args:
        vertices: (F, V, 3) tensor, 世界坐标系
        scale_factor: float, z_scene / z_smpl
        R_cw_center: (3, 3) 中心帧 w2c rotation
        t_cw_center: (3,) 中心帧 w2c translation

    Returns:
        vertices_scaled: (F, V, 3) tensor
    """
    R = R_cw_center.float()
    t = t_cw_center.float()
    F_n, V, _ = vertices.shape

    verts_flat = vertices.reshape(-1, 3)  # (F*V, 3)
    # world → cam
    v_cam = (R @ verts_flat.T).T + t.unsqueeze(0)  # (F*V, 3)
    # scale in cam space
    v_cam_scaled = v_cam * scale_factor
    # cam → world
    v_world_scaled = (R.T @ (v_cam_scaled - t.unsqueeze(0)).T).T  # (F*V, 3)

    return v_world_scaled.reshape(F_n, V, 3)


def load_pipeline_masks(result_dir, sequence):
    """从 pipeline 缓存加载 SAM 分割的人体 mask。

    优先查找:
      1. {result_dir}/{sequence}/masks.pt
      2. {result_dir}/intermediate/{sequence}/iter0_human_segmentation_masks.pt
      3. {result_dir}/intermediate/{sequence}/iter0_human_segmentation_data.pkl (PipelineData.masks)

    Returns:
        masks: torch.Tensor [N, H, W] uint8，0=背景，1=人体
    """
    # 尝试 1: 最终结果 masks.pt
    p1 = os.path.join(result_dir, sequence, 'masks.pt')
    if os.path.exists(p1):
        print(f"[Mask] Loading from {p1}")
        masks = torch.load(p1, map_location='cpu')
        print(f"  Loaded: {masks.shape}")
        return masks

    # 尝试 2: 中间结果单独保存的 masks.pt
    p2 = os.path.join(result_dir, 'intermediate', sequence, 'iter0_human_segmentation_masks.pt')
    if os.path.exists(p2):
        print(f"[Mask] Loading from {p2}")
        masks = torch.load(p2, map_location='cpu')
        print(f"  Loaded: {masks.shape}")
        return masks

    # 尝试 3: 从 PipelineData 缓存加载
    p3 = os.path.join(result_dir, 'intermediate', sequence, 'iter0_human_segmentation_data.pkl')
    if os.path.exists(p3):
        print(f"[Mask] Loading from PipelineData: {p3}")
        try:
            with gzip.open(p3, 'rb') as f:
                raw = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(p3, 'rb') as f:
                raw = pickle.load(f)
        masks_np = raw.get('masks')
        if masks_np is not None:
            masks = torch.from_numpy(masks_np) if isinstance(masks_np, np.ndarray) else masks_np
            print(f"  Loaded: {masks.shape}")
            return masks

    print("[Mask] WARNING: No pipeline mask cache found, falling back to no mask")
    return None


# ========== 加载 PipelineData ==========

def load_pipeline_data(pkl_path):
    """加载 pipeline 缓存的 PipelineData。"""
    print(f"Loading PipelineData from {pkl_path} ...")
    try:
        with gzip.open(pkl_path, 'rb') as f:
            raw = pickle.load(f)
    except (gzip.BadGzipFile, OSError):
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)

    from lib.pipeline.core.data import PipelineData
    data = PipelineData.load(pkl_path)
    print(f"  Loaded: {data.sequence_name}, {data.get_num_frames()} frames")
    return data


def _to_tensor(x):
    if x is None:
        return None
    return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()


# ========== Viser 场景 ==========

def build_scene(server, stages_data, faces, pointcloud=None):
    """构建 viser 场景。

    Args:
        stages_data: {stage_name: {'vertices': (F,V,3), 'color': (r,g,b),
                       optional 'vertices_scaled': (F,V,3)}}
        faces: SMPL faces
        pointcloud: optional (points, colors) tuple
    """
    import viser
    import viser.transforms as vtf

    faces_int = faces.astype(np.int32)

    first = list(stages_data.values())[0]
    num_frames = first['vertices'].shape[0]
    has_scaled = any('vertices_scaled' in d for d in stages_data.values())

    # 点云
    if pointcloud is not None:
        pts, colors = pointcloud
        server.scene.add_point_cloud(
            "/pointcloud",
            points=pts,
            colors=colors,
            point_size=0.008,
            point_shape="rounded",
        )

    # 地面
    all_verts = np.concatenate(
        [d['vertices'].numpy().reshape(-1, 3) for d in stages_data.values()],
        axis=0,
    )
    y_min = all_verts[:, 1].min()
    xz = all_verts[:, [0, 2]]
    cx = (xz.min(0)[0] + xz.max(0)[0]) / 2
    cz = (xz.min(0)[1] + xz.max(0)[1]) / 2
    scale = max(xz.max(0)[0] - xz.min(0)[0], xz.max(0)[1] - xz.min(0)[1]) * 1.5
    scale = max(scale, 4.0)

    from lib.vis.tools import checkerboard_geometry
    gv, gf, gc, _ = checkerboard_geometry(
        length=scale, c1=cx, c2=cz, up="y",
        color0=[0.85, 0.9, 0.9], color1=[0.65, 0.7, 0.7],
        tile_width=0.5,
    )
    gv[:, 1] = y_min
    server.scene.add_mesh_simple(
        "/ground/solid", vertices=gv.astype(np.float32), faces=gf.astype(np.int32),
        flat_shading=True, wireframe=False, color=(210, 218, 218), side="double",
    )
    server.scene.add_mesh_simple(
        "/ground/wire", vertices=gv.astype(np.float32), faces=gf.astype(np.int32),
        wireframe=True, color=(140, 150, 150), side="double",
    )

    # Frame nodes
    server.scene.add_frame("/frames", show_axes=False)
    frame_nodes = []
    stage_mesh_groups = {name: [] for name in stages_data}
    stage_mesh_scaled_groups = {name: [] for name in stages_data} if has_scaled else {}

    from tqdm import tqdm
    print(f"Building scene: {num_frames} frames, {len(stages_data)} stages...")
    for fi in tqdm(range(num_frames), desc="Building viser scene"):
        frame_node = server.scene.add_frame(f"/frames/t{fi}", show_axes=False)
        frame_nodes.append(frame_node)

        for stage_name, sdata in stages_data.items():
            verts_fi = sdata['vertices'][fi].numpy().astype(np.float32)
            mesh = server.scene.add_mesh_simple(
                f"/frames/t{fi}/{stage_name}/mesh",
                vertices=verts_fi,
                faces=faces_int,
                flat_shading=False, wireframe=False,
                color=sdata['color'],
            )
            stage_mesh_groups[stage_name].append(mesh)

            # scaled mesh（默认隐藏）
            if 'vertices_scaled' in sdata:
                verts_scaled_fi = sdata['vertices_scaled'][fi].numpy().astype(np.float32)
                mesh_s = server.scene.add_mesh_simple(
                    f"/frames/t{fi}/{stage_name}/mesh_scaled",
                    vertices=verts_scaled_fi,
                    faces=faces_int,
                    flat_shading=False, wireframe=False,
                    color=sdata['color'],
                )
                mesh_s.visible = False
                stage_mesh_scaled_groups[stage_name].append(mesh_s)

    # 初始化：只显示第一帧
    for i, fn in enumerate(frame_nodes):
        fn.visible = (i == 0)

    return frame_nodes, stage_mesh_groups, stage_mesh_scaled_groups


def setup_gui(server, stages_data, frame_nodes, stage_mesh_groups, num_frames,
              stage_mesh_scaled_groups=None, scale_factor=1.0):
    """设置 GUI 控件。"""

    gui_timestep = server.gui.add_slider(
        "Timestep", min=0, max=num_frames - 1, step=1, initial_value=0, disabled=True,
    )
    gui_playing = server.gui.add_checkbox("Playing", True)
    gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=10)
    gui_next = server.gui.add_button("Next Frame", disabled=True)
    gui_prev = server.gui.add_button("Prev Frame", disabled=True)

    server.gui.add_markdown("---")
    server.gui.add_markdown("**Stages**")
    stage_toggles = {}
    for stage_name, sdata in stages_data.items():
        color = sdata['color']
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        toggle = server.gui.add_checkbox(f"Show {stage_name} ({color_hex})", True)
        stage_toggles[stage_name] = toggle

    server.gui.add_markdown("---")
    gui_show_pointcloud = server.gui.add_checkbox("Show PointCloud", True)

    has_scaled = stage_mesh_scaled_groups and len(stage_mesh_scaled_groups) > 0
    gui_scale_body = None
    if has_scaled:
        gui_scale_body = server.gui.add_checkbox(
            f"Scale Body (x{scale_factor:.3f})", False,
        )

    gui_show_all = server.gui.add_checkbox("Show All Frames", False)

    prev_timestep = [0]
    show_all_state = [False]

    @gui_timestep.on_update
    def _(_):
        if show_all_state[0]:
            return
        cur = gui_timestep.value
        prev = prev_timestep[0]
        if cur != prev:
            with server.atomic():
                frame_nodes[cur].visible = True
                frame_nodes[prev].visible = False
            prev_timestep[0] = cur
            server.flush()

    @gui_next.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def _(_):
        gui_timestep.disabled = gui_playing.value
        gui_next.disabled = gui_playing.value
        gui_prev.disabled = gui_playing.value

    for stage_name, toggle in stage_toggles.items():
        meshes = stage_mesh_groups[stage_name]
        scaled_meshes = stage_mesh_scaled_groups.get(stage_name, []) if has_scaled else []
        def _make_cb(meshes, scaled_meshes, toggle):
            def cb(_):
                scaled_on = gui_scale_body.value if gui_scale_body is not None else False
                for m in meshes:
                    m.visible = toggle.value and not scaled_on
                for m in scaled_meshes:
                    m.visible = toggle.value and scaled_on
            return cb
        toggle.on_update(_make_cb(meshes, scaled_meshes, toggle))

    if gui_scale_body is not None:
        @gui_scale_body.on_update
        def _(_):
            scaled_on = gui_scale_body.value
            with server.atomic():
                for stage_name in stages_data:
                    show = stage_toggles[stage_name].value
                    for m in stage_mesh_groups[stage_name]:
                        m.visible = show and not scaled_on
                    for m in stage_mesh_scaled_groups.get(stage_name, []):
                        m.visible = show and scaled_on
            server.flush()

    @gui_show_all.on_update
    def _(_):
        show_all = gui_show_all.value
        show_all_state[0] = show_all
        with server.atomic():
            if show_all:
                for fn in frame_nodes:
                    fn.visible = True
                gui_playing.value = False
            else:
                cur = gui_timestep.value
                for i, fn in enumerate(frame_nodes):
                    fn.visible = (i == cur)
                prev_timestep[0] = cur
        server.flush()

    return gui_playing, gui_timestep, gui_fps


# ========== 主入口 ==========

@dataclass
class Config:
    result_dir: str = "results/promptbase_video_depthrefine_emdb2"
    """pipeline 结果目录"""

    sequence: str = "56_outdoor_stairs_up_down"
    """序列名"""

    center_frame: int = 220
    """要可视化的窗口中心帧"""

    window_size: int = 21
    """窗口大小"""

    cache_stage: str = "world_transform"
    """使用的 pipeline 缓存阶段（OAR 前的最后一个阶段）"""

    depth_model: str = "metric3d_vit_small"
    """Metric3D 模型版本"""

    pc_subsample: int = 4
    """点云下采样步长"""

    port: int = 8080
    """viser 服务端口"""

    device: str = "cuda"
    """GPU 设备"""


def main(cfg: Config):
    import viser

    device = cfg.device

    # 1. 加载 PipelineData 缓存
    pkl_path = os.path.join(
        cfg.result_dir, 'intermediate', cfg.sequence,
        f'iter0_{cfg.cache_stage}_data.pkl'
    )
    if not os.path.exists(pkl_path):
        print(f"ERROR: Cache not found: {pkl_path}")
        print("Available stages:")
        cache_dir = os.path.join(cfg.result_dir, 'intermediate', cfg.sequence)
        if os.path.exists(cache_dir):
            for f in sorted(os.listdir(cache_dir)):
                if f.endswith('_data.pkl'):
                    print(f"  {f}")
        return

    data = load_pipeline_data(pkl_path)

    # 2. 提取参数
    sp = data.smpl_params
    cam = data.camera_params

    global_orient_w = _to_tensor(sp.global_orient_w)
    body_pose_aa = _to_tensor(sp.body_pose_aa)
    betas = _to_tensor(sp.betas)
    global_trans = _to_tensor(sp.global_trans)
    static_conf_logits = _to_tensor(sp.static_conf_logits)
    F_total = global_orient_w.shape[0]

    # 相机参数
    wt = data.metadata.get('world_transform', {})
    R_wc = _to_tensor(wt.get('R_wc', cam.world_R if cam.world_R is not None else cam.R))
    T_wc = _to_tensor(wt.get('T_wc', cam.world_T if cam.world_T is not None else cam.T))
    R_cw = R_wc.transpose(-1, -2)
    t_cw = -(R_cw @ T_wc.unsqueeze(-1)).squeeze(-1)

    # K
    K_gt = _to_tensor(cam.intrinsics)
    hpe_K = data.metadata.get('hpe_K', None)
    if hpe_K is not None:
        K = _to_tensor(hpe_K)
    elif K_gt is not None:
        K = K_gt
    else:
        img0 = np.array(Image.open(data.image_paths[0]))
        H, W = img0.shape[:2]
        f_est = (H**2 + W**2) ** 0.5
        K = torch.tensor([[f_est, 0, W/2.], [0, f_est, H/2.], [0, 0, 1]], dtype=torch.float)
    K_for_depth = K_gt if K_gt is not None else K

    # 3. 窗口参数
    half_win = cfg.window_size // 2
    center = cfg.center_frame
    start = max(0, center - half_win)
    end = min(F_total, center + half_win + 1)
    center_local = center - start
    print(f"\n{'='*60}")
    print(f"Sequence: {cfg.sequence}")
    print(f"Window: center={center}, range=[{start}, {end}), len={end-start}")
    print(f"Total frames: {F_total}")
    print(f"{'='*60}\n")

    # ========== Stage 1: 原始 ==========
    print("[Stage 1] Original pose + transl")
    win_orient = global_orient_w[start:end]
    win_bp_orig = body_pose_aa[start:end].clone()
    win_betas = betas[start:end]
    win_transl_orig = global_trans[start:end].clone()

    verts_original = compute_world_vertices(win_orient, win_bp_orig, win_betas, win_transl_orig, device)
    print(f"  Vertices: {verts_original.shape}")

    # ========== Stage 2: pp_static_joint 滑步消除 ==========
    print("[Stage 2] pp_static_joint skating removal")
    win_transl_skating = apply_pp_static_joint(
        win_orient, win_bp_orig, win_betas, win_transl_orig,
        static_conf_logits[start:end] if static_conf_logits is not None else None,
        center_idx=center_local, device=device,
    )
    verts_skating = compute_world_vertices(win_orient, win_bp_orig, win_betas, win_transl_skating, device)
    print(f"  Vertices: {verts_skating.shape}")
    print(f"  Transl delta: mean={( win_transl_skating - win_transl_orig).abs().mean():.4f}")

    # ========== Stage 3: Depth Contact IK ==========
    print("[Stage 3] Depth inference + Contact IK")

    # 深度推理
    intrinsics_np = K_for_depth.numpy() if isinstance(K_for_depth, torch.Tensor) else K_for_depth
    print(f"  Loading Metric3D ({cfg.depth_model})...")
    metric3d_model = load_metric3d(cfg.depth_model)

    center_image_path = data.image_paths[center]
    print(f"  Inferring depth for frame {center}: {center_image_path}")
    depth_map = infer_depth(center_image_path, intrinsics_np, metric3d_model, cfg.depth_model)
    print(f"  Depth map: {depth_map.shape}, range=[{depth_map.min():.2f}, {depth_map.max():.2f}]")

    # 保存深度图到 figures/
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    depth_fig_dir = os.path.join(PROJECT_ROOT, 'figures', 'depth_maps')
    os.makedirs(depth_fig_dir, exist_ok=True)
    depth_np_save = depth_map.numpy() if isinstance(depth_map, torch.Tensor) else depth_map

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # 左：原始 RGB
    center_img = np.array(Image.open(center_image_path).convert('RGB'))
    axes[0].imshow(center_img)
    axes[0].set_title(f'{cfg.sequence} frame {center} (RGB)', fontsize=11)
    axes[0].axis('off')
    # 右：深度图（viridis colormap）
    im = axes[1].imshow(depth_np_save, cmap='viridis')
    axes[1].set_title(f'Metric3D depth [{depth_np_save.min():.2f}, {depth_np_save.max():.2f}]m', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='depth (m)')
    plt.tight_layout()

    depth_fig_path = os.path.join(depth_fig_dir, f'{cfg.sequence}_frame{center:05d}_depth.png')
    fig.savefig(depth_fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Depth figure saved: {depth_fig_path}")

    # 释放 Metric3D
    del metric3d_model
    gc.collect()
    torch.cuda.empty_cache()

    # Contact IK
    win_bp_contact, scale_factor = apply_depth_contact_ik(
        global_orient_w=global_orient_w,
        body_pose_aa=win_bp_orig,
        betas=betas,
        win_transl=win_transl_skating,
        static_conf_logits=static_conf_logits,
        depth_map=depth_map,
        R_cw=R_cw, t_cw=t_cw, K=K,
        center=center, start=start, end=end,
        device=device,
    )
    verts_contact = compute_world_vertices(win_orient, win_bp_contact, win_betas, win_transl_skating, device)
    print(f"  Vertices: {verts_contact.shape}")
    print(f"  BodyPose delta: mean={(win_bp_contact - win_bp_orig).abs().mean():.4f}")
    print(f"  Scale factor (z_scene/z_smpl): {scale_factor:.4f}")

    # ========== 人体 mask + 点云 ==========
    print("[PointCloud] Loading pipeline SAM mask...")
    center_rgb = np.array(Image.open(center_image_path).convert('RGB'))
    img_h, img_w = center_rgb.shape[:2]

    # 从 pipeline 缓存加载 SAM 分割 mask（MASK-SLAM 使用的同一份）
    pipeline_masks = load_pipeline_masks(cfg.result_dir, cfg.sequence)
    body_msk = None
    if pipeline_masks is not None:
        center_mask = pipeline_masks[center].numpy() if isinstance(pipeline_masks, torch.Tensor) else pipeline_masks[center]
        # mask 可能分辨率与图像不同，需要 resize
        if center_mask.shape[0] != img_h or center_mask.shape[1] != img_w:
            import cv2
            center_mask = cv2.resize(
                center_mask.astype(np.uint8), (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            )
        body_msk = (center_mask > 0).astype(np.uint8)
        print(f"  Body mask (SAM): {body_msk.sum()} / {body_msk.size} pixels masked "
              f"({body_msk.sum() / body_msk.size * 100:.1f}%)")
        del pipeline_masks  # 释放大量内存
    else:
        print("  WARNING: No mask available, pointcloud will include human region")

    # 生成原始尺度点云
    print("[PointCloud] Converting depth to world pointcloud (body excluded)...")
    pts_world, pts_colors = depth_to_pointcloud(
        depth_map, K_for_depth, R_cw[center], t_cw[center],
        max_depth=20.0, subsample=cfg.pc_subsample,
        rgb_image=center_rgb,
        body_mask=body_msk,
    )
    print(f"  Points: {pts_world.shape[0]}")
    scale_factor = 0.77
    # 以相机原点为锚点，缩放 SMPL vertices 到深度图尺度
    print(f"[Scale] Scaling SMPL vertices by {scale_factor:.4f} (camera-anchored)...")
    verts_original_scaled = scale_vertices_to_depth(
        verts_original, scale_factor, R_cw[center], t_cw[center])
    verts_skating_scaled = scale_vertices_to_depth(
        verts_skating, scale_factor, R_cw[center], t_cw[center])
    verts_contact_scaled = scale_vertices_to_depth(
        verts_contact, scale_factor, R_cw[center], t_cw[center])

    # ========== Viser ==========
    print("\n[Viser] Starting server...")

    stages_data = {
        'Original': {
            'vertices': verts_original,
            'vertices_scaled': verts_original_scaled,
            'color': COLOR_ORIGINAL,
        },
        'SkatingRemoved': {
            'vertices': verts_skating,
            'vertices_scaled': verts_skating_scaled,
            'color': COLOR_SKATING,
        },
        'ContactIK': {
            'vertices': verts_contact,
            'vertices_scaled': verts_contact_scaled,
            'color': COLOR_CONTACT,
        },
    }

    faces = _smpl_faces
    server = viser.ViserServer(port=cfg.port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    frame_nodes, stage_mesh_groups, stage_mesh_scaled_groups = build_scene(
        server, stages_data, faces,
        pointcloud=(pts_world, pts_colors),
    )

    gui_playing, gui_timestep, gui_fps = setup_gui(
        server, stages_data, frame_nodes, stage_mesh_groups, end - start,
        stage_mesh_scaled_groups=stage_mesh_scaled_groups,
        scale_factor=scale_factor,
    )

    print(f"\n{'='*60}")
    print(f"Viser server running at: http://localhost:{cfg.port}")
    print(f"Window: center={center}, frames=[{start}, {end})")
    print(f"Stages: Original(gray), SkatingRemoved(green), ContactIK(blue)")
    print(f"PointCloud: {pts_world.shape[0]} points (RGB, body excluded)")
    print(f"Scale factor (z_scene/z_smpl): {scale_factor:.4f}")
    print(f"{'='*60}")
    print(f"Open http://localhost:{cfg.port} in your browser")
    print(f"Press Ctrl+C to stop\n")

    try:
        while True:
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % (end - start)
            time.sleep(1.0 / gui_fps.value)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    import tyro
    cfg = tyro.cli(Config)
    main(cfg)
