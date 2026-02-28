"""DepthSceneRefineComponent - 光学锚定 + 场景感知的滑动窗口精修组件 (OAR)

基于 Metric3D 深度图 + 2D 重投影约束，以滑动窗口方式优化世界坐标系下的
global_orient 和 body_pose。

核心流程:
1. 对均匀采样帧调用 Metric3D 生成度量深度图
2. SMPL-X forward 预计算世界坐标系关节点
3. 滑动窗口遍历序列，每个窗口中心帧做 Adam 优化:
   - 重投影 loss: proj(joints) vs 2D 关键点
   - 深度穿透 loss: SMPL 关节 z vs 场景深度
   - 平滑/接触/地面约束
4. 高斯衰减传播修正量到窗口内其他帧
5. 归一化合并所有窗口
6. IK 修正 body_pose
"""

import sys
import os
import gc
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import logging
from PIL import Image

from lib.pipeline.core.component import Component
from lib.pipeline.core.data import PipelineData


def gmof(x, sigma=100):
    """Geman-McClure robust error function."""
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


class DepthSceneRefineComponent(Component):
    """OAR: Optical-Anchored Refinement with Scene Depth Constraints.

    在 WorldTransform 之后插入，精修 global_orient_w / body_pose_aa / global_trans。
    """

    COMPONENT_TYPE = "depth_scene_refine"

    DEFAULT_CONFIG = {
        # Metric3D
        'metric3d_model': 'metric3d_vit_small',
        'metric3d_root': 'thirdparty/PromptHMR/pipeline/yvanyin_metric3d_main',
        'prompthmr_root': 'thirdparty/PromptHMR',
        'metric3d_batch_size': 32,
        # 滑动窗口
        'window_size': 21,
        'stride': 5,
        'sigma': 5.0,
        # 优化
        'opt_steps': 50,
        'opt_lr': 0.01,
        # 损失权重
        'loss_reproj_w': 10.0,
        'loss_depth_w': 1.0,
        'loss_smooth_vel_w': 0.01,
        'loss_smooth_acc_w': 0.01,
        'loss_contact_vel_w': 100.0,
        'loss_contact_height_w': 1.0,
        'loss_floor_w': 5.0,
        'loss_reg_w': 0.01,
        'reproj_sigma': 50,
        'depth_sigma': 1.0,
        # 功能开关
        'enable_depth_constraint': True,
        'enable_ik': True,
        'enable_orient_refine': True,
        'opt_contact': True,
        # 图像模型 3D body pose 纠正
        'enable_img_pose_refine': True,        # 是否用图像模型单帧 SMPL 纠正四肢 body_pose
        'img_pose_limb_joints': [1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21],  # FK-22 中的四肢关节
        'img_pose_opt_steps': 30,               # body pose 优化迭代次数
        'img_pose_opt_lr': 0.005,               # body pose 优化学习率
        'loss_img3d_w': 5.0,                    # 3D 关节 loss 权重
        'loss_img3d_sigma': 0.1,                # gmof sigma（米）
        'loss_bodypose_reg_w': 0.1,             # body pose 正则化权重
        # EnDecoder
        'gvhmr_root': 'thirdparty/GVHMR',
        # 深度帧采样
        'depth_frame_stride': 5,
        # 杂项
        'save_depth_maps': False,
        'fps': 30,
        'device': 'cuda',
        # OAR 诊断可视化
        'vis_oar_diagnostics': False,
        'vis_oar_frame_stride': 5,
    }

    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)
        self.endecoder = None
        self.metric3d_model = None

    def validate_input(self, data: PipelineData) -> bool:
        if data.smpl_params is None:
            self.logger.warning("No smpl_params found")
            return False
        sp = data.smpl_params
        if sp.global_orient_w is None or sp.body_pose_aa is None or sp.global_trans is None:
            self.logger.warning("smpl_params missing world-space fields")
            return False
        if data.camera_params is None:
            self.logger.warning("No camera_params found")
            return False
        if not data.image_paths:
            self.logger.warning("No image_paths found")
            return False
        return True

    def setup(self):
        self._setup_endecoder()
        self._is_setup = True
        self.logger.info("DepthSceneRefine (OAR) component initialized")

    def _setup_endecoder(self):
        """Load GVHMR EnDecoder for FK / IK."""
        gvhmr_abs = os.path.abspath(self.config['gvhmr_root'])
        if gvhmr_abs not in sys.path:
            sys.path.insert(0, gvhmr_abs)

        hmr4d_modules = [k for k in sys.modules if k.startswith('hmr4d')]
        for mod_name in hmr4d_modules:
            del sys.modules[mod_name]
        if sys.path[0] != gvhmr_abs:
            if gvhmr_abs in sys.path:
                sys.path.remove(gvhmr_abs)
            sys.path.insert(0, gvhmr_abs)

        old_cwd = os.getcwd()
        os.chdir(gvhmr_abs)
        try:
            from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
            self.endecoder = EnDecoder().to(self.device)
        finally:
            os.chdir(old_cwd)

    # ------------------------------------------------------------------
    # Metric3D depth inference
    # ------------------------------------------------------------------
    def _load_metric3d(self):
        """Load Metric3D model on demand."""
        import copy
        import types

        metric3d_abs = os.path.abspath(self.config['metric3d_root'])
        prompthmr_abs = os.path.abspath(self.config['prompthmr_root'])
        for p in [metric3d_abs, prompthmr_abs]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # Also add the camera sub-dir for depth_utils
        cam_dir = os.path.join(prompthmr_abs, 'pipeline', 'camera')
        if cam_dir not in sys.path:
            sys.path.insert(0, cam_dir)

        old_cwd = os.getcwd()
        os.chdir(metric3d_abs)

        # Register a custom deepcopy dispatcher for ModuleType so that
        # mmcv Config.fromfile -> copy.deepcopy does not choke on numpy/etc
        # modules that appear in Metric3D config .py files.
        _deepcopy_dispatch = copy._deepcopy_dispatch
        _had_module = types.ModuleType in _deepcopy_dispatch
        _old_module_copier = _deepcopy_dispatch.get(types.ModuleType)

        def _copy_module(x, memo):
            return x  # modules are singletons

        _deepcopy_dispatch[types.ModuleType] = _copy_module
        try:
            from hubconf import metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2
            model_fn = {
                'metric3d_vit_small': metric3d_vit_small,
                'metric3d_vit_large': metric3d_vit_large,
                'metric3d_vit_giant2': metric3d_vit_giant2,
            }[self.config['metric3d_model']]
            model = model_fn(pretrain=True)
            model = model.cuda().half().eval()
            self.metric3d_model = model
            self.logger.info(f"Loaded Metric3D ({self.config['metric3d_model']})")
        finally:
            if _had_module:
                _deepcopy_dispatch[types.ModuleType] = _old_module_copier
            else:
                _deepcopy_dispatch.pop(types.ModuleType, None)
            os.chdir(old_cwd)

    def _unload_metric3d(self):
        if self.metric3d_model is not None:
            del self.metric3d_model
            self.metric3d_model = None
            gc.collect()
            torch.cuda.empty_cache()

    def _infer_depth_maps(
        self,
        image_paths: List[str],
        frame_indices: np.ndarray,
        intrinsics: np.ndarray,
        save_dir: Optional[str] = None,
    ) -> Dict[int, torch.Tensor]:
        """Run Metric3D on selected frames. Returns {frame_idx: depth_tensor_cpu}."""
        prompthmr_abs = os.path.abspath(self.config['prompthmr_root'])
        cam_dir = os.path.join(prompthmr_abs, 'pipeline', 'camera')
        if cam_dir not in sys.path:
            sys.path.insert(0, cam_dir)
        from depth_utils import prep_metric3d, post_metric3d

        # Free GPU memory left over from previous pipeline stages before
        # loading the (potentially large) Metric3D model.
        gc.collect()
        torch.cuda.empty_cache()

        self._load_metric3d()

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        calib = [fx, fy, cx, cy]
        model_version = self.config['metric3d_model']

        # Get prep params from first image
        first_img = np.array(Image.open(image_paths[frame_indices[0]]).convert('RGB'))
        _, intrinsic_prep, pad_info, rgb_origin = prep_metric3d(first_img, calib, model_version)

        depth_maps = {}
        batch_size = self.config.get('metric3d_batch_size', 4)

        # Process in batches – use try/except to auto-reduce batch size on OOM
        while batch_size >= 1:
            try:
                for batch_start in range(0, len(frame_indices), batch_size):
                    batch_idxs = frame_indices[batch_start:batch_start + batch_size]
                    batch_tensors = []
                    for idx in batch_idxs:
                        img = np.array(Image.open(image_paths[idx]).convert('RGB'))
                        rgb_prep, _, _, _ = prep_metric3d(img, calib, model_version)
                        batch_tensors.append(rgb_prep)

                    rgb_batch = torch.cat(batch_tensors, dim=0).cuda().half()

                    with torch.inference_mode():
                        pred_depth, confidence, _ = self.metric3d_model.inference({'input': rgb_batch})

                    # Post-process each frame
                    for i, idx in enumerate(batch_idxs):
                        depth_i = post_metric3d(
                            pred_depth[i:i+1], confidence[i:i+1] if confidence is not None else None,
                            pad_info, rgb_origin, intrinsic_prep
                        )
                        depth_maps[int(idx)] = depth_i.cpu().squeeze()

                        if save_dir is not None:
                            os.makedirs(save_dir, exist_ok=True)
                            np.save(os.path.join(save_dir, f'depth_{int(idx):05d}.npy'),
                                    depth_i.cpu().squeeze().numpy())

                    # Free intermediates every batch
                    del rgb_batch, pred_depth, confidence, batch_tensors
                    torch.cuda.empty_cache()

                break  # success
            except torch.cuda.OutOfMemoryError:
                # Clear partial results from this failed attempt for frames
                # that haven't been stored yet, then retry with smaller batch
                old_bs = batch_size
                batch_size = max(1, batch_size // 2)
                self.logger.warning(
                    f"Metric3D OOM with batch_size={old_bs}, retrying with batch_size={batch_size}"
                )
                gc.collect()
                torch.cuda.empty_cache()
                if old_bs == 1:
                    raise  # truly cannot fit even a single image

        self._unload_metric3d()
        self.logger.info(f"Generated {len(depth_maps)} depth maps for frames: {frame_indices.tolist()[:5]}...")
        return depth_maps

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_w2c(R_wc: torch.Tensor, T_wc: torch.Tensor):
        """Convert c2w to w2c: R_cw = R_wc^T, t_cw = -R_cw @ T_wc."""
        R_cw = R_wc.transpose(-1, -2)
        t_cw = -(R_cw @ T_wc.unsqueeze(-1)).squeeze(-1)
        return R_cw, t_cw

    @staticmethod
    def _project_joints(j3d_world, R_cw, t_cw, K):
        """World joints -> 2D pixel coords. All (F, J, 3) or (F, 3, 3)."""
        j_cam = torch.einsum('fij,fkj->fki', R_cw, j3d_world) + t_cw[:, None, :]
        pj = torch.einsum('ij,fkj->fki', K, j_cam)
        pj_2d = pj[..., :2] / (pj[..., 2:3] + 1e-6)
        return pj_2d, j_cam

    # ------------------------------------------------------------------
    # Core optimisation
    # ------------------------------------------------------------------
    def _sliding_window_optimize(
        self,
        joints_world: torch.Tensor,   # (F, J, 3)
        R_cw: torch.Tensor,            # (F, 3, 3)
        t_cw: torch.Tensor,            # (F, 3)
        K: torch.Tensor,               # (3, 3)
        kp2d: Optional[torch.Tensor],  # (F, 17, 3) COCO-17 x,y,conf  or None
        depth_maps: Dict[int, torch.Tensor],
        contact_conf: Optional[torch.Tensor],  # (F, J_contact)
        bbox_height: Optional[torch.Tensor],   # (F,)
        masks: Optional[torch.Tensor],         # (F, H, W) human masks
        img_h: int,
        img_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run sliding-window optimization. Returns (delta_transl, delta_orient_aa)."""
        device = self.device
        F_total = joints_world.shape[0]
        window_size = self.config['window_size']
        stride = self.config['stride']
        sigma = self.config['sigma']
        half_win = window_size // 2
        fps = self.config['fps']

        # Accumulators (on CPU to save GPU memory)
        acc_delta_t = torch.zeros(F_total, 3)
        acc_delta_orient = torch.zeros(F_total, 3)  # axis-angle delta
        acc_weights = torch.zeros(F_total)

        # Pre-compute gaussian weights for window positions
        offsets = torch.arange(-half_win, half_win + 1).float()
        gauss_w = torch.exp(-offsets ** 2 / (2 * sigma ** 2))
        gauss_w = gauss_w / gauss_w.max()

        # Contact joint ids (SMPL-22 FK: L_Foot=7, L_ToeBase=10, R_Foot=8, R_ToeBase=11)
        contact_joint_ids = [7, 10, 8, 11]
        # Number of joints actually available from FK
        n_joints = joints_world.shape[1]  # typically 22 from fk_v2
        # Also filter contact ids to valid range
        contact_joint_ids = [j for j in contact_joint_ids if j < n_joints]

        # FK-22 → COCO-17 joint pair mapping for reprojection loss
        # FK_idx: SMPL joint → COCO_idx: ViTPose detection
        _FK_TO_COCO_PAIRS = [
            # (fk_idx, coco_idx)
            (1, 11),   # L_Hip → L_hip
            (2, 12),   # R_Hip → R_hip
            (4, 13),   # L_Knee → L_knee
            (5, 14),   # R_Knee → R_knee
            (7, 15),   # L_Ankle → L_ankle
            (8, 16),   # R_Ankle → R_ankle
            (16, 5),   # L_Shoulder → L_shoulder (NOT 13=L_Collar which is too medial)
            (17, 6),   # R_Shoulder → R_shoulder (NOT 14=R_Collar which is too medial)
            (18, 7),   # L_Elbow → L_elbow
            (19, 8),   # R_Elbow → R_elbow
            (20, 9),   # L_Wrist → L_wrist
            (21, 10),  # R_Wrist → R_wrist
        ]
        # Filter pairs to valid FK joint range
        fk_coco_pairs = [(fk, coco) for fk, coco in _FK_TO_COCO_PAIRS if fk < n_joints]
        fk_pair_idxs = [p[0] for p in fk_coco_pairs]
        coco_pair_idxs = [p[1] for p in fk_coco_pairs]

        K_dev = K.to(device)

        # --- 动态地面高度估计 ---
        # 用脚部关节（L_Foot=7, R_Foot=8, L_ToeBase=10, R_ToeBase=11）的最低 Y 值估算地面
        foot_y = joints_world[:, contact_joint_ids, 1]  # (F, 4)
        floor_y = foot_y.min().item()
        self.logger.info(
            f"[OAR Ground Check] Joint Y stats: "
            f"all_min={joints_world[:, :, 1].min():.3f}, all_max={joints_world[:, :, 1].max():.3f}, "
            f"all_mean={joints_world[:, :, 1].mean():.3f} | "
            f"foot_min={foot_y.min():.3f}, foot_max={foot_y.max():.3f}, "
            f"foot_mean={foot_y.mean():.3f} | "
            f"estimated_floor_y={floor_y:.3f}"
        )

        num_windows = 0
        # Debug: 累计各项 loss 用于日志
        _dbg_loss_sums = {
            'reproj': 0.0, 'depth': 0.0, 'vel': 0.0, 'acc': 0.0,
            'contact_h': 0.0, 'floor': 0.0, 'reg': 0.0, 'total': 0.0,
        }
        _dbg_count = 0

        for center in range(0, F_total, stride):
            start = max(0, center - half_win)
            end = min(F_total, center + half_win + 1)
            win_len = end - start
            center_local = center - start

            # Get data for this window on device
            j_win = joints_world[start:end].to(device)  # (W, J, 3)
            R_win = R_cw[start:end].to(device)
            t_win = t_cw[start:end].to(device)

            # Optimization variables for center frame
            from lib.utils.rotation_conversions import (
                axis_angle_to_matrix,
                matrix_to_axis_angle,
            )

            delta_t = torch.zeros(3, device=device, requires_grad=True)

            if self.config['enable_orient_refine']:
                delta_orient_aa = torch.zeros(3, device=device, requires_grad=True)
                opt_params = [delta_t, delta_orient_aa]
            else:
                delta_orient_aa = torch.zeros(3, device=device)
                opt_params = [delta_t]

            optimizer = torch.optim.Adam(opt_params, lr=self.config['opt_lr'])

            for step in range(self.config['opt_steps']):
                optimizer.zero_grad()

                # Apply delta to center frame joints
                delta_R = axis_angle_to_matrix(delta_orient_aa.unsqueeze(0))  # (1, 3, 3)
                j_center_mod = (delta_R @ j_win[center_local].unsqueeze(-1)).squeeze(-1) + delta_t  # (J, 3)

                # Project center frame
                j_cam_center = (R_win[center_local] @ j_center_mod.unsqueeze(-1)).squeeze(-1) + t_win[center_local]  # (J, 3)
                pj = (K_dev @ j_cam_center.unsqueeze(-1)).squeeze(-1)
                pj_2d = pj[:, :2] / (pj[:, 2:3] + 1e-6)

                total_loss = torch.tensor(0.0, device=device)
                # Per-step loss tracking for debug
                _step_losses = {}

                # --- Loss 1: 2D reprojection (FK joints vs COCO-17 ViTPose) ---
                if kp2d is not None and len(fk_coco_pairs) > 0:
                    kp_center = kp2d[center].to(device)  # (17, 3)
                    gt_2d = kp_center[coco_pair_idxs, :2]  # (N_pairs, 2)
                    conf = kp_center[coco_pair_idxs, 2]    # (N_pairs,)
                    proj_2d = pj_2d[fk_pair_idxs]          # (N_pairs, 2)

                    reproj_err = gmof(proj_2d - gt_2d, sigma=self.config['reproj_sigma'])
                    # 使用固定归一化常数代替 bbox_height，避免 reproj loss 被过度压缩
                    reproj_err = reproj_err / 1000.0
                    conf_mask = conf > 0.5
                    loss_reproj = (conf_mask.unsqueeze(-1) * reproj_err).mean()
                    total_loss = total_loss + self.config['loss_reproj_w'] * loss_reproj
                    _step_losses['reproj'] = loss_reproj.item()

                # --- Loss 2: Depth penetration ---
                if self.config['enable_depth_constraint'] and depth_maps:
                    actual_frame = center
                    # Find nearest depth frame
                    depth_frame = min(depth_maps.keys(), key=lambda k: abs(k - actual_frame))
                    if abs(depth_frame - actual_frame) <= self.config['depth_frame_stride'] * 2:
                        depth_map = depth_maps[depth_frame].to(device)
                        dH, dW = depth_map.shape

                        # Sample depth at projected joint locations
                        u = pj_2d[:, 0].long().clamp(0, dW - 1)
                        v = pj_2d[:, 1].long().clamp(0, dH - 1)
                        z_smpl = j_cam_center[:, 2]
                        z_scene = depth_map[v, u]

                        # Penetration: SMPL joint is behind scene surface
                        valid_depth = (z_scene > 0.1) & (z_scene < 100.0) & (z_smpl > 0)
                        penetration = F.relu(z_smpl - z_scene - 0.05)  # 5cm margin
                        loss_depth = gmof(penetration[valid_depth], sigma=self.config['depth_sigma']).mean() if valid_depth.any() else torch.tensor(0.0, device=device)
                        total_loss = total_loss + self.config['loss_depth_w'] * loss_depth
                        _step_losses['depth'] = loss_depth.item()

                # --- Loss 3: Smoothness (on delta-modified transl trajectory) ---
                # We use the original joints + propagated delta for smoothness
                # For the window, create a modified trajectory
                t_weights = gauss_w[start - center + half_win:end - center + half_win].to(device)
                delta_t_propagated = t_weights.unsqueeze(-1) * delta_t.unsqueeze(0)  # (W, 3)
                j_mod_transl = j_win[:, 0, :] + delta_t_propagated  # root joint trajectory (W, 3)

                if win_len >= 3:
                    vel = (j_mod_transl[1:] - j_mod_transl[:-1]) * fps
                    loss_vel = vel.pow(2).mean()
                    acc = (j_mod_transl[2:] + j_mod_transl[:-2] - 2 * j_mod_transl[1:-1]) * fps
                    loss_acc = acc.norm(dim=-1).mean()
                    total_loss = total_loss + self.config['loss_smooth_vel_w'] * loss_vel
                    total_loss = total_loss + self.config['loss_smooth_acc_w'] * loss_acc
                    _step_losses['vel'] = loss_vel.item()
                    _step_losses['acc'] = loss_acc.item()

                # --- Loss 4: Contact constraints ---
                if self.config['opt_contact'] and contact_conf is not None:
                    cc = torch.sigmoid(contact_conf[center].to(device))  # (J_contact,)
                    contact_j = j_center_mod[contact_joint_ids[:len(cc)]]  # (4, 3)

                    # Contact height: 使用动态地面高度代替硬编码 0.08
                    floor_diff = torch.abs(contact_j[:, 1] - (floor_y + 0.08))
                    loss_contact_h = (floor_diff * cc[:len(contact_j)]).mean()
                    total_loss = total_loss + self.config['loss_contact_height_w'] * loss_contact_h
                    _step_losses['contact_h'] = loss_contact_h.item()

                # --- Loss 5: No joints below floor (使用动态地面高度) ---
                loss_floor = F.relu(floor_y - j_center_mod[:, 1]).mean()
                total_loss = total_loss + self.config['loss_floor_w'] * loss_floor
                _step_losses['floor'] = loss_floor.item()

                # --- Loss 6: Regularization ---
                loss_reg = delta_t.norm() + delta_orient_aa.norm()
                total_loss = total_loss + self.config['loss_reg_w'] * loss_reg
                _step_losses['reg'] = loss_reg.item()
                _step_losses['total'] = total_loss.item()

                # Debug 日志：每个窗口的第一步和最后一步
                if step == 0 or step == self.config['opt_steps'] - 1:
                    self.logger.debug(
                        f"[OAR Win center={center} step={step}] "
                        f"reproj={_step_losses.get('reproj', 0):.6f} "
                        f"depth={_step_losses.get('depth', 0):.6f} "
                        f"vel={_step_losses.get('vel', 0):.6f} "
                        f"acc={_step_losses.get('acc', 0):.6f} "
                        f"contact_h={_step_losses.get('contact_h', 0):.6f} "
                        f"floor={_step_losses.get('floor', 0):.6f} "
                        f"reg={_step_losses.get('reg', 0):.6f} "
                        f"total={_step_losses.get('total', 0):.6f} "
                        f"delta_t={delta_t.detach().cpu().tolist()}"
                    )

                total_loss.backward()
                optimizer.step()

            # 累计最后一步 loss 用于汇总日志
            for k in _dbg_loss_sums:
                _dbg_loss_sums[k] += _step_losses.get(k, 0.0)
            _dbg_count += 1

            # Propagate delta to window
            with torch.no_grad():
                w = gauss_w[start - center + half_win:end - center + half_win]
                for j in range(start, end):
                    local_j = j - start
                    wt = w[local_j].item()
                    acc_delta_t[j] += wt * delta_t.detach().cpu()
                    acc_delta_orient[j] += wt * delta_orient_aa.detach().cpu()
                    acc_weights[j] += wt

            num_windows += 1

        # 汇总 loss 日志
        if _dbg_count > 0:
            avg_str = " | ".join(
                f"{k}={v / _dbg_count:.6f}" for k, v in _dbg_loss_sums.items()
            )
            self.logger.info(f"[OAR Loss Avg over {_dbg_count} windows (last step)] {avg_str}")

        self.logger.info(f"Processed {num_windows} windows")

        # Normalize by accumulated weights
        valid = acc_weights > 1e-6
        acc_delta_t[valid] /= acc_weights[valid].unsqueeze(-1)
        acc_delta_orient[valid] /= acc_weights[valid].unsqueeze(-1)

        return acc_delta_t, acc_delta_orient

    # ------------------------------------------------------------------
    # IK correction
    # ------------------------------------------------------------------
    def _apply_ik_correction(
        self,
        global_orient_w: torch.Tensor,  # (F, 3) aa
        body_pose_aa: torch.Tensor,      # (F, 63)
        betas: torch.Tensor,             # (F, 10)
        global_trans: torch.Tensor,      # (F, 3)
        static_conf_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run GVHMR process_ik to correct body_pose after root modification."""
        gvhmr_abs = os.path.abspath(self.config['gvhmr_root'])
        if gvhmr_abs not in sys.path:
            sys.path.insert(0, gvhmr_abs)

        from hmr4d.model.gvhmr.utils.postprocess import process_ik

        outputs = {
            "pred_smpl_params_global": {
                "global_orient": global_orient_w.unsqueeze(0).to(self.device),
                "body_pose": body_pose_aa.unsqueeze(0).to(self.device),
                "betas": betas.unsqueeze(0).to(self.device),
                "transl": global_trans.unsqueeze(0).to(self.device),
            },
            "static_conf_logits": (
                static_conf_logits.unsqueeze(0).to(self.device)
                if static_conf_logits is not None
                else torch.zeros(1, global_orient_w.shape[0], 6, device=self.device)
            ),
        }

        corrected_body_pose = process_ik(outputs, self.endecoder)  # (1, F, 63)
        return corrected_body_pose[0].cpu()

    # ------------------------------------------------------------------
    # 图像模型 3D body pose 纠正
    # ------------------------------------------------------------------
    def _prepare_img_model_joints(
        self,
        phmr_img_smpl: Dict[str, torch.Tensor],
        betas_video: torch.Tensor,  # (F, 10) 视频头的 betas
    ) -> torch.Tensor:
        """从图像模型的 SMPL rotmat 中提取 body-local 关节位置。

        在 body-local space（无 global_orient、无 transl）中做 FK，
        完全消除 global_orient 不一致的影响。
        统一使用视频头的 betas 以确保骨骼长度一致。

        Returns:
            img_j3d_local: (F, 22, 3) body-local root-relative joints
        """
        from lib.utils.rotation_conversions import matrix_to_axis_angle

        rotmat = phmr_img_smpl['rotmat']  # (N, 22, 3, 3)

        # 提取 body_pose 部分（joint 1-21），转为 axis-angle
        body_rotmat = rotmat[:, 1:, :, :]          # (N, 21, 3, 3)
        body_aa = matrix_to_axis_angle(body_rotmat) # (N, 21, 3)
        body_aa_flat = body_aa.reshape(-1, 63)      # (N, 63)

        # Body-local FK: 不传 global_orient 和 transl，消除全局朝向差异
        with torch.no_grad():
            joints_local = self.endecoder.fk_v2(
                body_pose=body_aa_flat.unsqueeze(0).to(self.device),
                betas=betas_video.unsqueeze(0).to(self.device),
                global_orient=None,
                transl=None,
            )[0]  # (N, 22, 3)

        # Root-relative
        img_j3d_local = joints_local - joints_local[:, 0:1, :]
        return img_j3d_local

    def _refine_body_pose_with_img_model(
        self,
        body_pose_aa: torch.Tensor,       # (F, 63) axis-angle
        betas: torch.Tensor,              # (F, 10)
        img_j3d_local: torch.Tensor,      # (F, 22, 3) body-local root-relative
    ) -> torch.Tensor:
        """通过图像模型的单帧 body-local 3D 关节约束来优化 body_pose。

        核心思想：
        - 在 body-local space（无 global_orient / transl）做 FK 和比较
        - 完全消除 global_orient 不一致的影响
        - 优化四肢关节的 body_pose_aa delta，使 FK 的 body-local joints 匹配图像模型

        Returns:
            refined_body_pose: (F, 63) 修正后的 body_pose
        """
        device = self.device
        F_total = body_pose_aa.shape[0]
        opt_steps = self.config['img_pose_opt_steps']
        lr = self.config['img_pose_opt_lr']
        limb_joints = self.config['img_pose_limb_joints']

        # body_pose 索引映射: FK joint i (i>=1) → body_pose[(i-1)*3 : i*3]
        limb_bp_indices = []
        for j in limb_joints:
            if j == 0 or j > 21:
                continue
            limb_bp_indices.extend([(j-1)*3, (j-1)*3+1, (j-1)*3+2])

        self.logger.info(
            f"[ImgPoseRefine] Optimizing {len(limb_joints)} limb joints "
            f"({len(limb_bp_indices)} params) for {F_total} frames, "
            f"{opt_steps} steps, lr={lr}"
        )

        delta_bp = torch.zeros(F_total, len(limb_bp_indices), device=device, requires_grad=True)

        bp_orig = body_pose_aa.to(device)
        beta = betas.to(device)
        img_target = img_j3d_local.to(device)  # (F, 22, 3)

        optimizer = torch.optim.Adam([delta_bp], lr=lr)

        # 要约束的关节索引（FK-22 空间）
        fk_limb_idxs = [j for j in limb_joints if j < 22]

        for step in range(opt_steps):
            optimizer.zero_grad()

            bp_mod = bp_orig.clone()
            bp_mod[:, limb_bp_indices] = bp_orig[:, limb_bp_indices] + delta_bp

            # Body-local FK: 不传 global_orient / transl
            with torch.enable_grad():
                joints_local = self.endecoder.fk_v2(
                    body_pose=bp_mod.unsqueeze(0),
                    betas=beta.unsqueeze(0),
                    global_orient=None,
                    transl=None,
                )[0]  # (F, 22, 3)

            # Root-relative body-local
            fk_root_rel = joints_local - joints_local[:, 0:1, :]

            fk_limbs = fk_root_rel[:, fk_limb_idxs]
            img_limbs = img_target[:, fk_limb_idxs]

            loss_3d = gmof(
                fk_limbs - img_limbs,
                sigma=self.config['loss_img3d_sigma']
            ).mean()

            loss_reg = delta_bp.pow(2).mean()

            total_loss = (
                self.config['loss_img3d_w'] * loss_3d
                + self.config['loss_bodypose_reg_w'] * loss_reg
            )

            total_loss.backward() #这是用反向传播的？不是用逆运动学吗
            optimizer.step()

            if step == 0 or step == opt_steps - 1:
                self.logger.debug(
                    f"[ImgPoseRefine step={step}] "
                    f"loss_3d={loss_3d.item():.6f}, "
                    f"loss_reg={loss_reg.item():.6f}, "
                    f"total={total_loss.item():.6f}, "
                    f"delta_bp_norm={delta_bp.detach().norm().item():.6f}"
                )

        with torch.no_grad():
            refined_bp_dev = bp_orig.clone()
            refined_bp_dev[:, limb_bp_indices] = bp_orig[:, limb_bp_indices] + delta_bp.detach()
            refined_bp = refined_bp_dev.cpu()

        delta_norm = delta_bp.detach().abs().mean().item()
        self.logger.info(
            f"[ImgPoseRefine] Done. "
            f"Final loss_3d={loss_3d.item():.6f}, "
            f"delta_bp_mean={delta_norm:.6f}"
        )

        return refined_bp

    # ------------------------------------------------------------------
    # OAR 诊断可视化
    # ------------------------------------------------------------------
    def _compute_per_frame_projections(
        self,
        joints_world: torch.Tensor,   # (F, J, 3)
        R_cw: torch.Tensor,            # (F, 3, 3)
        t_cw: torch.Tensor,            # (F, 3)
        K: torch.Tensor,               # (3, 3)
    ) -> torch.Tensor:
        """投影所有帧的世界关节到 2D，返回 (F, J, 2)。"""
        pj_2d, _ = self._project_joints(joints_world, R_cw, t_cw, K)
        return pj_2d  # (F, J, 2)

    def _vis_depth_maps(
        self,
        depth_maps: Dict[int, torch.Tensor],
        image_paths: List[str],
        joints_cam: torch.Tensor,       # (F, J, 3) camera-space joints
        proj_2d: torch.Tensor,           # (F, J, 2) projected 2D joints
        output_dir: str,
        kp2d: torch.Tensor = None,      # (F, 17, 3) ViTPose COCO-17 x,y,conf (optional, for debug)
    ):
        """可视化深度图：彩色深度图叠加到原图 + 关节深度标注。

        对每个有深度图的帧，生成左右拼接图：
          左：原图 + 关节投影点（颜色编码关节深度）
          右：彩色深度图 + 关节投影点（标注 SMPL 关节深度 vs 场景深度）
        """
        import cv2
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        vis_dir = os.path.join(output_dir, 'oar_depth')
        os.makedirs(vis_dir, exist_ok=True)

        # 深度 colormap
        all_depths = torch.cat([d.flatten() for d in depth_maps.values()])
        valid_depths = all_depths[(all_depths > 0.1) & (all_depths < 100.0)]
        if len(valid_depths) == 0:
            self.logger.warning("No valid depth values for visualization")
            return
        # quantile() has a limit on tensor size (~2^24 elements).
        # Subsample if too large to avoid "input tensor is too large" error.
        if valid_depths.numel() > 10_000_000:
            indices = torch.randperm(valid_depths.numel())[:10_000_000]
            sampled = valid_depths[indices]
        else:
            sampled = valid_depths
        d_min, d_max = valid_depths.min().item(), sampled.quantile(0.95).item()

        for fidx, depth_t in depth_maps.items():
            if fidx >= len(image_paths):
                continue
            img = cv2.imread(image_paths[fidx])
            if img is None:
                continue
            H, W = img.shape[:2]
            depth_np = depth_t.numpy()
            dH, dW = depth_np.shape

            # 生成彩色深度图
            depth_norm = np.clip((depth_np - d_min) / (d_max - d_min + 1e-6), 0, 1)
            depth_color = (cm.get_cmap('plasma')(depth_norm)[:, :, :3] * 255).astype(np.uint8)
            depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

            # 将深度图 resize 到图像尺寸
            if (dH, dW) != (H, W):
                depth_color = cv2.resize(depth_color, (W, H), interpolation=cv2.INTER_LINEAR)
                depth_np_resized = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                depth_np_resized = depth_np

            # 叠加深度到原图（半透明）
            depth_overlay = cv2.addWeighted(img, 0.5, depth_color, 0.5, 0)

            # 在两张图上标注关节
            n_joints = proj_2d.shape[1]
            for j in range(n_joints):
                u, v = int(proj_2d[fidx, j, 0].item()), int(proj_2d[fidx, j, 1].item())
                if 0 <= u < W and 0 <= v < H:
                    z_smpl = joints_cam[fidx, j, 2].item()
                    z_scene = depth_np_resized[min(v, H-1), min(u, W-1)]

                    # 颜色：穿透（红）、正常（绿）、无效深度（灰）
                    if z_scene < 0.1 or z_scene > 100.0:
                        color = (128, 128, 128)
                    elif z_smpl > z_scene + 0.05:
                        color = (0, 0, 255)  # 穿透 - 红
                    else:
                        color = (0, 255, 0)  # 正常 - 绿

                    cv2.circle(depth_overlay, (u, v), 4, color, -1)
                    cv2.circle(depth_color, (u, v), 4, color, -1)

                    # 标注深度差值（仅对有效深度）
                    if 0.1 < z_scene < 100.0:
                        diff = z_smpl - z_scene
                        label = f"{diff:+.2f}"
                        cv2.putText(depth_color, label, (u + 6, v - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # [Debug] 画 ViTPose 2D 检测点（青色）
            if kp2d is not None and fidx < kp2d.shape[0]:
                kp = kp2d[fidx]  # (17, 3): x, y, conf
                COCO_SKELETON = [
                    (0, 1), (0, 2), (1, 3), (2, 4),
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16),
                ]
                # 画 skeleton 连线（青色）
                for i, j_idx in COCO_SKELETON:
                    if kp[i, 2] > 0.3 and kp[j_idx, 2] > 0.3:
                        pt1 = (int(kp[i, 0].item()), int(kp[i, 1].item()))
                        pt2 = (int(kp[j_idx, 0].item()), int(kp[j_idx, 1].item()))
                        cv2.line(depth_overlay, pt1, pt2, (255, 255, 0), 1)
                        cv2.line(depth_color, pt1, pt2, (255, 255, 0), 1)
                # 画关键点（青色空心圆）
                for ki in range(kp.shape[0]):
                    if kp[ki, 2] > 0.3:
                        pt = (int(kp[ki, 0].item()), int(kp[ki, 1].item()))
                        cv2.circle(depth_overlay, pt, 5, (255, 255, 0), 1)
                        cv2.circle(depth_color, pt, 5, (255, 255, 0), 1)

            # 拼接
            combined = np.concatenate([depth_overlay, depth_color], axis=1)

            # 标题
            cv2.putText(combined, f"Frame {fidx} | Depth Overlay", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Depth Map + Joint Z diff", (W + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out_path = os.path.join(vis_dir, f'{fidx:05d}.jpg')
            cv2.imwrite(out_path, combined)

        self.logger.info(f"Saved depth visualizations ({len(depth_maps)} frames) to {vis_dir}")

    def _vis_keypoints_2d(
        self,
        kp2d: torch.Tensor,             # (F, 17, 3) COCO-17 x,y,conf
        proj_2d: torch.Tensor,           # (F, J, 2)  FK-22 投影
        image_paths: List[str],
        output_dir: str,
        frame_stride: int = 5,
        tag: str = 'pre',
    ):
        """可视化 2D 关键点：ViTPose 检测点（绿色）+ FK 投影关节（蓝色）。

        COCO-17 skeleton 连线 + 各关键点置信度标注。
        """
        import cv2

        vis_dir = os.path.join(output_dir, f'oar_keypoints_{tag}')
        os.makedirs(vis_dir, exist_ok=True)

        # COCO-17 skeleton
        COCO_SKELETON = [
            (0, 1), (0, 2), (1, 3), (2, 4),         # head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # upper body
            (5, 11), (6, 12), (11, 12),                # torso
            (11, 13), (13, 15), (12, 14), (14, 16),    # lower body
        ]
        # COCO-17 joint names
        COCO_NAMES = [
            'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
            'L_sho', 'R_sho', 'L_elb', 'R_elb', 'L_wri', 'R_wri',
            'L_hip', 'R_hip', 'L_kne', 'R_kne', 'L_ank', 'R_ank',
        ]

        # FK→COCO mapping for drawing correspondence lines
        _FK_TO_COCO = {
            1: 11, 2: 12, 4: 13, 5: 14, 7: 15, 8: 16,
            16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10,
        }

        F_total = kp2d.shape[0]
        vis_frames = list(range(0, F_total, frame_stride))

        for fidx in vis_frames:
            if fidx >= len(image_paths):
                continue
            img = cv2.imread(image_paths[fidx])
            if img is None:
                continue
            H, W = img.shape[:2]
            canvas = img.copy()

            # 画 COCO skeleton (绿色)
            kp = kp2d[fidx]  # (17, 3)
            for i, j in COCO_SKELETON:
                if kp[i, 2] > 0.3 and kp[j, 2] > 0.3:
                    pt1 = (int(kp[i, 0]), int(kp[i, 1]))
                    pt2 = (int(kp[j, 0]), int(kp[j, 1]))
                    cv2.line(canvas, pt1, pt2, (0, 200, 0), 2)

            # 画 COCO-17 关键点（绿色圆圈 + 置信度）
            for ki in range(17):
                x, y, c = kp[ki, 0].item(), kp[ki, 1].item(), kp[ki, 2].item()
                if c > 0.3 and 0 <= x < W and 0 <= y < H:
                    cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(canvas, f"{c:.1f}", (int(x) + 6, int(y) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # 画 FK-22 投影关节（蓝色小圆）
            n_joints = proj_2d.shape[1]
            for ji in range(n_joints):
                u, v = int(proj_2d[fidx, ji, 0].item()), int(proj_2d[fidx, ji, 1].item())
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(canvas, (u, v), 3, (255, 100, 0), -1)  # 蓝色

            # 画 FK→COCO 对应线（黄色虚线）
            for fk_idx, coco_idx in _FK_TO_COCO.items():
                if fk_idx >= n_joints:
                    continue
                c = kp[coco_idx, 2].item()
                if c <= 0.3:
                    continue
                u_fk = int(proj_2d[fidx, fk_idx, 0].item())
                v_fk = int(proj_2d[fidx, fk_idx, 1].item())
                u_coco = int(kp[coco_idx, 0].item())
                v_coco = int(kp[coco_idx, 1].item())
                if 0 <= u_fk < W and 0 <= v_fk < H and 0 <= u_coco < W and 0 <= v_coco < H:
                    cv2.line(canvas, (u_fk, v_fk), (u_coco, v_coco), (0, 255, 255), 1)

            # 图例
            cv2.putText(canvas, f"Frame {fidx} | 2D KP ({tag})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, "Green=ViTPose  Blue=FK_proj  Yellow=correspondence",
                        (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            out_path = os.path.join(vis_dir, f'{fidx:05d}.jpg')
            cv2.imwrite(out_path, canvas)

        self.logger.info(f"Saved 2D keypoint vis ({tag}, {len(vis_frames)} frames) to {vis_dir}")

    def _vis_reprojection_error(
        self,
        kp2d: torch.Tensor,             # (F, 17, 3) COCO-17 x,y,conf
        proj_2d_pre: torch.Tensor,       # (F, J, 2) FK 投影 (OAR 前)
        proj_2d_post: torch.Tensor,      # (F, J, 2) FK 投影 (OAR 后)
        image_paths: List[str],
        output_dir: str,
        frame_stride: int = 5,
    ):
        """可视化重投影误差：OAR 前后对比。

        生成左右拼接图：
          左：OAR 前的重投影误差（红线 = FK投影→ViTPose）
          右：OAR 后的重投影误差
        底部叠加数值统计。
        """
        import cv2

        vis_dir = os.path.join(output_dir, 'oar_reproj_error')
        os.makedirs(vis_dir, exist_ok=True)

        _FK_TO_COCO = {
            1: 11, 2: 12, 4: 13, 5: 14, 7: 15, 8: 16,
            16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10,
        }

        F_total = kp2d.shape[0]
        n_joints = proj_2d_pre.shape[1]
        vis_frames = list(range(0, F_total, frame_stride))

        # 全局统计
        all_err_pre, all_err_post = [], []

        for fidx in vis_frames:
            if fidx >= len(image_paths):
                continue
            img = cv2.imread(image_paths[fidx])
            if img is None:
                continue
            H, W = img.shape[:2]
            canvas_pre = img.copy()
            canvas_post = img.copy()

            kp = kp2d[fidx]  # (17, 3)
            frame_err_pre, frame_err_post = [], []

            for fk_idx, coco_idx in _FK_TO_COCO.items():
                if fk_idx >= n_joints:
                    continue
                c = kp[coco_idx, 2].item()
                if c <= 0.5:
                    continue

                gt_x, gt_y = int(kp[coco_idx, 0].item()), int(kp[coco_idx, 1].item())

                # OAR 前
                px_pre = int(proj_2d_pre[fidx, fk_idx, 0].item())
                py_pre = int(proj_2d_pre[fidx, fk_idx, 1].item())
                err_pre = np.sqrt((px_pre - gt_x)**2 + (py_pre - gt_y)**2)
                frame_err_pre.append(err_pre)

                # OAR 后
                px_post = int(proj_2d_post[fidx, fk_idx, 0].item())
                py_post = int(proj_2d_post[fidx, fk_idx, 1].item())
                err_post = np.sqrt((px_post - gt_x)**2 + (py_post - gt_y)**2)
                frame_err_post.append(err_post)

                # 误差颜色映射 (绿→黄→红)
                def err_color(e, max_e=50.0):
                    ratio = min(e / max_e, 1.0)
                    r = int(255 * ratio)
                    g = int(255 * (1 - ratio))
                    return (0, g, r)  # BGR

                # 画 pre
                if 0 <= px_pre < W and 0 <= py_pre < H:
                    cv2.circle(canvas_pre, (gt_x, gt_y), 5, (0, 255, 0), -1)       # GT - 绿
                    cv2.circle(canvas_pre, (px_pre, py_pre), 4, (0, 0, 255), -1)    # FK - 红
                    cv2.line(canvas_pre, (gt_x, gt_y), (px_pre, py_pre),
                             err_color(err_pre), 2)
                    cv2.putText(canvas_pre, f"{err_pre:.0f}", (px_pre + 5, py_pre - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, err_color(err_pre), 1)

                # 画 post
                if 0 <= px_post < W and 0 <= py_post < H:
                    cv2.circle(canvas_post, (gt_x, gt_y), 5, (0, 255, 0), -1)
                    cv2.circle(canvas_post, (px_post, py_post), 4, (0, 0, 255), -1)
                    cv2.line(canvas_post, (gt_x, gt_y), (px_post, py_post),
                             err_color(err_post), 2)
                    cv2.putText(canvas_post, f"{err_post:.0f}", (px_post + 5, py_post - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, err_color(err_post), 1)

            # 帧内统计
            mean_pre = np.mean(frame_err_pre) if frame_err_pre else 0
            mean_post = np.mean(frame_err_post) if frame_err_post else 0
            all_err_pre.extend(frame_err_pre)
            all_err_post.extend(frame_err_post)

            # 标题
            cv2.putText(canvas_pre, f"Frame {fidx} PRE-OAR | mean={mean_pre:.1f}px", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas_post, f"Frame {fidx} POST-OAR | mean={mean_post:.1f}px", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 图例
            cv2.putText(canvas_pre, "Green=ViTPose  Red=FK_proj",
                        (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(canvas_post, "Green=ViTPose  Red=FK_proj",
                        (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            combined = np.concatenate([canvas_pre, canvas_post], axis=1)
            out_path = os.path.join(vis_dir, f'{fidx:05d}.jpg')
            cv2.imwrite(out_path, combined)
            print("saving:", out_path) 
        # 全局统计日志
        if all_err_pre:
            self.logger.info(
                f"Reproj error (px): PRE mean={np.mean(all_err_pre):.1f}, "
                f"median={np.median(all_err_pre):.1f}, max={np.max(all_err_pre):.1f} | "
                f"POST mean={np.mean(all_err_post):.1f}, "
                f"median={np.median(all_err_post):.1f}, max={np.max(all_err_post):.1f}"
            )

        self.logger.info(f"Saved reproj error vis ({len(vis_frames)} frames) to {vis_dir}")

    # ------------------------------------------------------------------
    # Visualisation hook helper
    # ------------------------------------------------------------------
    def _save_before_after_vis(self, data: PipelineData, pre_params: dict, output_dir: str):
        """Save side-by-side visualisation: before (left) vs after (right) OAR.

        Uses pytorch3d Renderer (same as adjacent_smpl_renderer gvhmr_world mode)
        to avoid EGL/pyrender dependency. Generates world-space vertices then
        projects to camera space via T_w2c before rendering.
        """
        import sys
        import cv2

        gvhmr_abs = os.path.abspath(self.config['gvhmr_root'])
        if gvhmr_abs not in sys.path:
            sys.path.insert(0, gvhmr_abs)

        old_cwd = os.getcwd()
        os.chdir(gvhmr_abs)
        try:
            from hmr4d.utils.smplx_utils import make_smplx
            from hmr4d.utils.geo.hmr_global import get_T_w2c_from_wcparams
            smplx_model = make_smplx("supermotion").to(self.device)
            smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(self.device)
            faces_smpl = make_smplx("smpl").faces
        except Exception as e:
            self.logger.warning(f"Cannot load SMPLX for vis: {e}")
            os.chdir(old_cwd)
            return
        os.chdir(old_cwd)

        try:
            from lib.vis.renderer import Renderer as Pytorch3dRenderer
        except ImportError:
            self.logger.warning("Cannot import pytorch3d Renderer, skipping vis")
            return

        vis_dir = os.path.join(output_dir, 'oar_comparison')
        os.makedirs(vis_dir, exist_ok=True)

        sp = data.smpl_params
        cam = data.camera_params
        device = torch.device(self.device)

        def _to_t(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            return x.float() if x is not None else None

        # Params
        go_w_after = _to_t(sp.global_orient_w).to(device)
        go_w_before = _to_t(pre_params['global_orient_w']).to(device)
        bp_after = _to_t(sp.body_pose_aa).to(device)
        bp_before = _to_t(pre_params['body_pose_aa']).to(device)
        betas = _to_t(sp.betas).to(device)
        global_trans_after = _to_t(sp.global_trans).to(device)
        global_trans_before = _to_t(pre_params['global_trans']).to(device)

        # Compute T_w2c from GVHMR global+incam params
        go_c = _to_t(sp.global_orient_c).to(device)
        trans_c = _to_t(sp.trans).to(device) if sp.trans is not None else None
        skeleton_offset = _to_t(sp.skeleton_offset).to(device) if sp.skeleton_offset is not None else torch.zeros(3, device=device)

        if trans_c is None or go_c is None:
            self.logger.warning("No incam params for T_w2c computation, skipping vis")
            return

        # 优先使用 HPE estimate_K（与推理一致），否则自动计算
        hpe_K = data.metadata.get('hpe_K', None)
        if hpe_K is not None:
            K = _to_t(hpe_K)
        else:
            # 旧缓存没有 hpe_K，用 estimate_K 方式计算
            img0 = cv2.imread(data.image_paths[0])
            if img0 is None:
                self.logger.warning("Cannot read first image for K estimation")
                return
            H, W = img0.shape[:2]
            f_est = (H**2 + W**2) ** 0.5
            K = torch.tensor([[f_est, 0, W/2.], [0, f_est, H/2.], [0, 0, 1]], dtype=torch.float)

        # Use pre-OAR world params to compute T_w2c (incam params unchanged by OAR)
        T_w2c = get_T_w2c_from_wcparams(
            go_w_before, global_trans_before, go_c, trans_c, skeleton_offset
        )  # (F, 4, 4)
        R_w2c = T_w2c[:, :3, :3]  # (F, 3, 3)
        t_w2c = T_w2c[:, :3, 3]   # (F, 3)

        # Generate world-space vertices for before & after
        with torch.no_grad():
            out_before = smplx_model(
                global_orient=go_w_before,
                body_pose=bp_before[:, :63],
                betas=betas,
                transl=global_trans_before,
            )
            verts_world_before = torch.stack(
                [torch.matmul(smplx2smpl, v) for v in out_before.vertices]
            )  # (F, V, 3)

            out_after = smplx_model(
                global_orient=go_w_after,
                body_pose=bp_after[:, :63],
                betas=betas,
                transl=global_trans_after,
            )
            verts_world_after = torch.stack(
                [torch.matmul(smplx2smpl, v) for v in out_after.vertices]
            )  # (F, V, 3)

        # Setup pytorch3d renderer
        img0 = cv2.imread(data.image_paths[0])
        H, W = img0.shape[:2]
        focal_length = float(K[0, 0])
        renderer = Pytorch3dRenderer(W, H, focal_length, device, faces_smpl)

        # Sample frames for vis
        F_total = go_w_after.shape[0]
        vis_frames = list(range(0, F_total, max(1, F_total // 30)))[:30]

        for fi in vis_frames:
            try:
                img = cv2.imread(data.image_paths[fi])

                # world -> cam_i
                R_i = R_w2c[fi]  # (3, 3)
                t_i = t_w2c[fi]  # (3,)

                verts_cam_b = torch.einsum('ij,vj->vi', R_i, verts_world_before[fi]) + t_i
                verts_cam_a = torch.einsum('ij,vj->vi', R_i, verts_world_after[fi]) + t_i

                img_before = renderer.render_mesh(verts_cam_b, img.copy(), colors=[0.5, 0.8, 0.5])
                img_after = renderer.render_mesh(verts_cam_a, img.copy(), colors=[0.8, 0.5, 0.5])

                # Side by side (BGR, cv2 format)
                combined = np.concatenate([img_before, img_after], axis=1)
                out_path = os.path.join(vis_dir, f'{fi:05d}.jpg')
                cv2.imwrite(out_path, combined)

            except Exception as e:
                self.logger.warning(f"Vis failed for frame {fi}: {e}")
                continue

        # Cleanup
        del smplx_model, smplx2smpl, verts_world_before, verts_world_after
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info(f"Saved OAR comparison visualisation to {vis_dir}")

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------
    def execute(self, data: PipelineData) -> PipelineData:
        sp = data.smpl_params
        cam = data.camera_params

        def _to_tensor(x):
            if x is None:
                return None
            return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()

        # Save pre-OAR params for visualisation
        pre_params = {
            'global_orient_w': (
                sp.global_orient_w.clone() if isinstance(sp.global_orient_w, torch.Tensor)
                else sp.global_orient_w.copy()
            ),
            'body_pose_aa': (
                sp.body_pose_aa.clone() if isinstance(sp.body_pose_aa, torch.Tensor)
                else sp.body_pose_aa.copy()
            ),
            'global_trans': (
                sp.global_trans.clone() if isinstance(sp.global_trans, torch.Tensor)
                else sp.global_trans.copy()
            ),
        }

        global_orient_w = _to_tensor(sp.global_orient_w)   # (F, 3) aa
        body_pose_aa = _to_tensor(sp.body_pose_aa)          # (F, 63)
        betas = _to_tensor(sp.betas)                         # (F, 10)
        global_trans = _to_tensor(sp.global_trans)           # (F, 3)
        static_conf_logits = _to_tensor(sp.static_conf_logits)

        F_total = global_orient_w.shape[0]
        self.logger.info(f"OAR: {F_total} frames, window={self.config['window_size']}, stride={self.config['stride']}")

        # --- 1. Get camera w2c ---
        wt = data.metadata.get('world_transform', {})
        R_wc = _to_tensor(wt.get('R_wc', cam.world_R if cam.world_R is not None else cam.R))
        T_wc = _to_tensor(wt.get('T_wc', cam.world_T if cam.world_T is not None else cam.T))
        R_cw, t_cw = self._build_w2c(R_wc, T_wc)

        # 优先使用 HPE 推理时用的 K（estimate_K），而非 GT K。
        # 因为 global_orient_w / global_trans 是基于 estimate_K 推理的 incam 参数转换而来，
        # 投影时必须用相同的 K 才能正确对齐到图像。
        K_gt = _to_tensor(cam.intrinsics)  # GT K（可能来自 SLAM/annotations）
        hpe_K = data.metadata.get('hpe_K', None)
        if hpe_K is not None:
            K = _to_tensor(hpe_K)
            if K_gt is not None:
                self.logger.info(
                    f"Using HPE estimate_K (fx={float(K[0,0]):.1f}) instead of "
                    f"GT K (fx={float(K_gt[0,0]):.1f}) for projection consistency"
                )
            else:
                self.logger.info(f"Using HPE estimate_K: fx={float(K[0,0]):.1f}")
        else:
            # hpe_K 不在 metadata 中（可能是旧缓存），用 estimate_K 方式重新计算
            img0 = np.array(Image.open(data.image_paths[0]))
            H, W = img0.shape[:2]
            f_est = (H**2 + W**2) ** 0.5  # 与 PromptHMR estimate_K 一致
            K = torch.tensor([[f_est, 0, W/2.], [0, f_est, H/2.], [0, 0, 1]], dtype=torch.float)
            if K_gt is not None:
                self.logger.info(
                    f"No HPE estimate_K in metadata, computed estimate_K "
                    f"(fx={f_est:.1f}) instead of GT K (fx={float(K_gt[0,0]):.1f})"
                )
            else:
                self.logger.info(f"Using computed estimate_K: fx={f_est:.1f}")

        # 深度推理仍使用 GT K（Metric3D 需要真实内参来恢复度量深度）
        K_for_depth = K_gt if K_gt is not None else K

        # --- 2. Generate depth maps for sampled frames ---
        depth_stride = self.config['depth_frame_stride']
        depth_frame_indices = np.arange(0, F_total, depth_stride)
        depth_maps = {}
        if self.config['enable_depth_constraint']:
            intrinsics_np = K_for_depth.numpy() if isinstance(K_for_depth, torch.Tensor) else K_for_depth
            save_dir = None
            if self.config['save_depth_maps']:
                output_dir = data.metadata.get('output_dir', 'results/oar_debug')
                save_dir = os.path.join(output_dir, data.sequence_name, 'depth_maps')
            depth_maps = self._infer_depth_maps(
                data.image_paths, depth_frame_indices, intrinsics_np, save_dir
            )

        # --- 3. FK: compute world joints --- #这是啥？
        from lib.utils.rotation_conversions import axis_angle_to_matrix
        with torch.no_grad():
            joints_world = self.endecoder.fk_v2( #世界坐标系下的人体joints
                body_pose=body_pose_aa.unsqueeze(0).to(self.device),
                betas=betas.unsqueeze(0).to(self.device),
                global_orient=global_orient_w.unsqueeze(0).to(self.device),
                transl=global_trans.unsqueeze(0).to(self.device),
            )  # (1, F, 22, 3)
            joints_world = joints_world[0].cpu()  # (F, 22, 3)

        # --- 4. Prepare 2D keypoints ---
        kp2d = None
        bbox_height = None
        # Try to find ViTPose COCO-17 keypoints from HPE output
        if 'vitpose_kp2d' in data.metadata:
            kp2d = _to_tensor(data.metadata['vitpose_kp2d'])  # (F, 17, 3)
            self.logger.info(f"Using ViTPose COCO-17 keypoints: {kp2d.shape}")
        # Otherwise, project joints as pseudo-2D keypoints (self-consistency, weak signal)
        if kp2d is None:
            self.logger.info("No external 2D keypoints, using projected joints as pseudo-kp2d (weak self-consistency)")
            pj2d, _ = self._project_joints(joints_world[:, :22], R_cw, t_cw, K)
            # Create pseudo kp2d in COCO-17 format using FK→COCO mapping
            kp2d = torch.zeros(F_total, 17, 3)
            # FK→COCO: 1→11, 2→12, 4→13, 5→14, 7→15, 8→16, 13→5, 14→6, 18→7, 19→8, 20→9, 21→10
            _fk2coco = {1:11, 2:12, 4:13, 5:14, 7:15, 8:16, 13:5, 14:6, 18:7, 19:8, 20:9, 21:10}
            for fk_idx, coco_idx in _fk2coco.items():
                kp2d[:, coco_idx, :2] = pj2d[:, fk_idx]
                kp2d[:, coco_idx, 2] = 1.0  # confidence

        # Compute bbox height from joint spread
        if bbox_height is None:
            pj_for_bbox, _ = self._project_joints(joints_world[:, :22], R_cw, t_cw, K)
            bbox_height = pj_for_bbox[:, :, 1].max(dim=1).values - pj_for_bbox[:, :, 1].min(dim=1).values
            bbox_height = bbox_height.clamp(min=50.0)

        # Get image size
        img0 = np.array(Image.open(data.image_paths[0]))
        img_h, img_w = img0.shape[:2]

        # --- 4b. Pre-OAR 2D projections (for diagnostic vis) --- #将世界坐标系转换到相机坐标系下，使用CLIFF的K
        proj_2d_pre = self._compute_per_frame_projections(joints_world, R_cw, t_cw, K)
        # Camera-space joints for depth vis
        _, joints_cam_pre = self._project_joints(joints_world, R_cw, t_cw, K)
        
        # --- 5. Sliding window optimization --- #这个逻辑是一定会开启的，通过修改transl来修改pose?
        delta_transl, delta_orient = self._sliding_window_optimize( #输出的是delta_transl和delta_orient，但是其实我就想改pose
            joints_world=joints_world,
            R_cw=R_cw,
            t_cw=t_cw,
            K=K,
            kp2d=kp2d,
            depth_maps=depth_maps,
            contact_conf=static_conf_logits,
            bbox_height=bbox_height,
            masks=data.masks,
            img_h=img_h,
            img_w=img_w,
        )

        self.logger.info(
            f"OAR deltas - transl: mean={delta_transl.abs().mean():.4f}, max={delta_transl.abs().max():.4f}; "
            f"orient: mean={delta_orient.abs().mean():.6f}, max={delta_orient.abs().max():.6f}"
        )

        # --- 6. Apply corrections ---
        from lib.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle


        # Apply orient correction
        if self.config['enable_orient_refine']:
            orient_mat_orig = axis_angle_to_matrix(global_orient_w)  # (F, 3, 3)
            delta_orient_mat = axis_angle_to_matrix(delta_orient)     # (F, 3, 3)
            orient_mat_new = delta_orient_mat @ orient_mat_orig
            global_orient_w_new = matrix_to_axis_angle(orient_mat_new)
        else:
            global_orient_w_new = global_orient_w

        # Apply transl correction
        global_trans_new = global_trans + delta_transl

        # --- 6b. Reprojection 劣化回滚保障 ---
        # 比较 PRE 和 POST 的 reproj error，如果恶化超过阈值则回滚
        _rollback_threshold = 1.5  # POST > 1.5x PRE 则回滚
        _rolled_back = False
        if kp2d is not None and self.config['enable_rollback_check']:
            _fk2coco_rb = {1:11, 2:12, 4:13, 5:14, 7:15, 8:16, 13:5, 14:6, 18:7, 19:8, 20:9, 21:10}
            _fk_idxs_rb = [k for k in _fk2coco_rb.keys()]
            _coco_idxs_rb = [v for v in _fk2coco_rb.values()]

            with torch.no_grad():
                # POST-OAR FK joints
                joints_world_post_check = self.endecoder.fk_v2(
                    body_pose=body_pose_aa.unsqueeze(0).to(self.device),
                    betas=betas.unsqueeze(0).to(self.device),
                    global_orient=global_orient_w_new.unsqueeze(0).to(self.device),
                    transl=global_trans_new.unsqueeze(0).to(self.device),
                )[0].cpu()

                proj_post_check = self._compute_per_frame_projections(
                    joints_world_post_check, R_cw, t_cw, K
                )  # (F, J, 2)

                # 计算 PRE/POST 平均 reproj error（仅高置信度关节）
                conf_all = kp2d[:, _coco_idxs_rb, 2]  # (F, N_pairs)
                mask_all = conf_all > 0.5

                gt_kp = kp2d[:, _coco_idxs_rb, :2]  # (F, N_pairs, 2)

                pre_diff = (proj_2d_pre[:, _fk_idxs_rb] - gt_kp).norm(dim=-1)  # (F, N_pairs)
                post_diff = (proj_post_check[:, _fk_idxs_rb] - gt_kp).norm(dim=-1)  # (F, N_pairs)

                pre_err_mean = pre_diff[mask_all].mean().item() if mask_all.any() else 0.0
                post_err_mean = post_diff[mask_all].mean().item() if mask_all.any() else 0.0

                self.logger.info(
                    f"[OAR Rollback Check] PRE reproj={pre_err_mean:.1f}px, "
                    f"POST reproj={post_err_mean:.1f}px, "
                    f"ratio={post_err_mean / max(pre_err_mean, 1e-6):.2f}"
                )

                if pre_err_mean > 0 and post_err_mean > pre_err_mean * _rollback_threshold:
                    self.logger.warning(
                        f"[OAR ROLLBACK] POST reproj ({post_err_mean:.1f}px) > "
                        f"{_rollback_threshold}x PRE ({pre_err_mean:.1f}px), "
                        f"reverting OAR corrections!"
                    )
                    global_orient_w_new = global_orient_w
                    global_trans_new = global_trans
                    _rolled_back = True

        #self.logger.inf(f"new pose:{body_pose_new} pre_pose:{body_pose_aa}")
        # --- 7. Image model 3D body pose refinement (四肢纠正) ---
        phmr_img_smpl = data.metadata.get('phmr_img_smpl', None)
        if (self.config['enable_img_pose_refine']
                and phmr_img_smpl is not None
                and not _rolled_back):
            self.logger.info("Running image-model 3D body pose refinement (body-local space)...")
            img_j3d_local = self._prepare_img_model_joints(
                phmr_img_smpl, betas,
            )
            body_pose_aa = self._refine_body_pose_with_img_model(
                body_pose_aa, betas, img_j3d_local,
            )
        elif self.config['enable_img_pose_refine'] and phmr_img_smpl is None:
            self.logger.info(
                "Image-model body pose refinement enabled but no phmr_img_smpl in metadata, skipping"
            )

        # --- 8. Optional IK correction ---
        if self.config['enable_ik'] and not _rolled_back:
            self.logger.info("Running IK correction after OAR...")
            body_pose_new = self._apply_ik_correction(
                global_orient_w_new, body_pose_aa, betas, global_trans_new,
                static_conf_logits,
            )
        else:
            body_pose_new = body_pose_aa
            if _rolled_back:
                self.logger.info("Skipping IK correction due to OAR rollback")

        #展示变化量：
        self.logger.info(f"OAR deltas - transl: mean={delta_transl.abs().mean():.4f}, max={delta_transl.abs().max():.4f}; "
            f"orient: mean={delta_orient.abs().mean():.6f}, max={delta_orient.abs().max():.6f}")
        dif_pose = body_pose_new - body_pose_aa
        self.logger.info(f"new pose:{body_pose_new} pre_pose:{body_pose_aa} difference:{dif_pose.abs().mean()}")
        
        
        # --- 9. Update data ---
        data.smpl_params.global_orient_w = global_orient_w_new
        data.smpl_params.global_trans = global_trans_new
        data.smpl_params.transl_w_raw = global_trans_new.clone()
        data.smpl_params.body_pose_aa = body_pose_new

        # Store metadata
        data.metadata['oar_refine'] = {
            'window_size': self.config['window_size'],
            'stride': self.config['stride'],
            'opt_steps': self.config['opt_steps'],
            'delta_transl_mean': delta_transl.abs().mean().item(),
            'delta_orient_mean': delta_orient.abs().mean().item(),
            'num_depth_frames': len(depth_maps),
            'enable_depth_constraint': self.config['enable_depth_constraint'],
            'enable_ik': self.config['enable_ik'],
            'enable_orient_refine': self.config['enable_orient_refine'],
            'enable_img_pose_refine': self.config['enable_img_pose_refine'],
            'img_pose_refined': (
                self.config['enable_img_pose_refine']
                and phmr_img_smpl is not None
                and not _rolled_back
            ),
            'rolled_back': _rolled_back,
        }

        # --- 10. Visualisation ---
        base_output_dir = data.metadata.get('output_dir',
                                        os.path.join('results', data.sequence_name))
        vis_output_dir = os.path.join(base_output_dir, data.sequence_name)
        self._save_before_after_vis(data, pre_params, vis_output_dir)

        # --- 10b. OAR 诊断可视化 ---
        if self.config.get('vis_oar_diagnostics', False):
            self.logger.info("Generating OAR diagnostic visualizations...")
            vis_stride = self.config.get('vis_oar_frame_stride', 5)

            # Post-OAR FK joints in world space
            with torch.no_grad():
                joints_world_post = self.endecoder.fk_v2(
                    body_pose=body_pose_new.unsqueeze(0).to(self.device),
                    betas=betas.unsqueeze(0).to(self.device),
                    global_orient=global_orient_w_new.unsqueeze(0).to(self.device),
                    transl=global_trans_new.unsqueeze(0).to(self.device),
                )
                joints_world_post = joints_world_post[0].cpu()

            proj_2d_post = self._compute_per_frame_projections( #差别来源于world的trans的调整
                joints_world_post, R_cw, t_cw, K
            )
            _, joints_cam_post = self._project_joints(
                joints_world_post, R_cw, t_cw, K
            )

            try:
                # 1. 深度图可视化
                if depth_maps:
                    self._vis_depth_maps(
                        depth_maps, data.image_paths,
                        joints_cam_pre, proj_2d_pre,
                        vis_output_dir,
                        kp2d=kp2d,
                    )

                # 2. 2D 关键点可视化 (OAR 前后)
                if kp2d is not None:
                    self._vis_keypoints_2d(
                        kp2d, proj_2d_pre, data.image_paths,
                        vis_output_dir, frame_stride=vis_stride, tag='pre',
                    )
                    self._vis_keypoints_2d(
                        kp2d, proj_2d_post, data.image_paths,
                        vis_output_dir, frame_stride=vis_stride, tag='post',
                    )

                # 3. 重投影误差可视化 (前后对比)
                if kp2d is not None: #这里报告了有差别
                    self._vis_reprojection_error(
                        kp2d, proj_2d_pre, proj_2d_post,
                        data.image_paths, vis_output_dir,
                        frame_stride=vis_stride,
                    )
            except Exception as e:
                self.logger.warning(f"OAR diagnostic visualization failed: {e}")

        self.logger.info("OAR refinement complete")
        return data

    def cleanup(self):
        self._unload_metric3d()
        self.endecoder = None
        super().cleanup()
