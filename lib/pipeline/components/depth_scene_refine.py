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
        'loss_reproj_w': 0.5,
        'loss_depth_w': 10.0,
        'loss_smooth_vel_w': 0.25,
        'loss_smooth_acc_w': 0.25,
        'loss_contact_vel_w': 1000.0,
        'loss_contact_height_w': 10.0,
        'loss_floor_w': 100.0,
        'loss_reg_w': 0.1,
        'reproj_sigma': 100,
        'depth_sigma': 0.5,
        # 功能开关
        'enable_depth_constraint': True,
        'enable_ik': True,
        'enable_orient_refine': True,
        'opt_contact': True,
        # EnDecoder
        'gvhmr_root': 'thirdparty/GVHMR',
        # 深度帧采样
        'depth_frame_stride': 5,
        # 杂项
        'save_depth_maps': False,
        'fps': 30,
        'device': 'cuda',
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
            (1, 11),   # leftUpLeg → L_hip
            (2, 12),   # rightUpLeg → R_hip
            (4, 13),   # leftLeg → L_knee
            (5, 14),   # rightLeg → R_knee
            (7, 15),   # leftFoot → L_ankle
            (8, 16),   # rightFoot → R_ankle
            (13, 5),   # leftShoulder → L_shoulder
            (14, 6),   # rightShoulder → R_shoulder
            (18, 7),   # leftForeArm → L_elbow
            (19, 8),   # rightForeArm → R_elbow
            (20, 9),   # leftHand → L_wrist
            (21, 10),  # rightHand → R_wrist
        ]
        # Filter pairs to valid FK joint range
        fk_coco_pairs = [(fk, coco) for fk, coco in _FK_TO_COCO_PAIRS if fk < n_joints]
        fk_pair_idxs = [p[0] for p in fk_coco_pairs]
        coco_pair_idxs = [p[1] for p in fk_coco_pairs]

        K_dev = K.to(device)

        num_windows = 0
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

                # --- Loss 1: 2D reprojection (FK joints vs COCO-17 ViTPose) ---
                if kp2d is not None and len(fk_coco_pairs) > 0:
                    kp_center = kp2d[center].to(device)  # (17, 3)
                    gt_2d = kp_center[coco_pair_idxs, :2]  # (N_pairs, 2)
                    conf = kp_center[coco_pair_idxs, 2]    # (N_pairs,)
                    proj_2d = pj_2d[fk_pair_idxs]          # (N_pairs, 2)

                    reproj_err = gmof(proj_2d - gt_2d, sigma=self.config['reproj_sigma'])
                    bh = bbox_height[center].to(device).clamp(min=1e-6) if bbox_height is not None else torch.tensor(500.0, device=device)
                    reproj_err = reproj_err / bh
                    conf_mask = conf > 0.5
                    loss_reproj = (conf_mask.unsqueeze(-1) * reproj_err).mean()
                    total_loss = total_loss + self.config['loss_reproj_w'] * loss_reproj

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

                # --- Loss 4: Contact constraints ---
                if self.config['opt_contact'] and contact_conf is not None:
                    cc = torch.sigmoid(contact_conf[center].to(device))  # (J_contact,)
                    contact_j = j_center_mod[contact_joint_ids[:len(cc)]]  # (4, 3)

                    # Contact height: feet should be near floor (y ~= 0.08)
                    floor_diff = torch.abs(contact_j[:, 1] - 0.08)
                    loss_contact_h = (floor_diff * cc[:len(contact_j)]).mean()
                    total_loss = total_loss + self.config['loss_contact_height_w'] * loss_contact_h

                # --- Loss 5: No joints below floor ---
                loss_floor = F.relu(-j_center_mod[:, 1]).mean()
                total_loss = total_loss + self.config['loss_floor_w'] * loss_floor

                # --- Loss 6: Regularization ---
                loss_reg = delta_t.norm() + delta_orient_aa.norm()
                total_loss = total_loss + self.config['loss_reg_w'] * loss_reg

                total_loss.backward()
                optimizer.step()

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

        K = _to_t(cam.intrinsics) if cam.intrinsics is not None else None
        if K is None:
            self.logger.warning("No intrinsics for visualisation")
            return

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

        K = _to_tensor(cam.intrinsics)
        if K is None:
            self.logger.warning("No intrinsics, using estimated K")
            img0 = np.array(Image.open(data.image_paths[0]))
            H, W = img0.shape[:2]
            f_est = max(H, W) * 1.0
            K = torch.tensor([[f_est, 0, W/2], [0, f_est, H/2], [0, 0, 1]], dtype=torch.float)

        # --- 2. Generate depth maps for sampled frames ---
        depth_stride = self.config['depth_frame_stride']
        depth_frame_indices = np.arange(0, F_total, depth_stride)
        depth_maps = {}
        if self.config['enable_depth_constraint']:
            intrinsics_np = K.numpy() if isinstance(K, torch.Tensor) else K
            save_dir = None
            if self.config['save_depth_maps']:
                output_dir = data.metadata.get('output_dir', 'results/oar_debug')
                save_dir = os.path.join(output_dir, data.sequence_name, 'depth_maps')
            depth_maps = self._infer_depth_maps(
                data.image_paths, depth_frame_indices, intrinsics_np, save_dir
            )

        # --- 3. FK: compute world joints ---
        from lib.utils.rotation_conversions import axis_angle_to_matrix
        with torch.no_grad():
            joints_world = self.endecoder.fk_v2(
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

        # --- 5. Sliding window optimization ---
        delta_transl, delta_orient = self._sliding_window_optimize(
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

        # --- 7. Optional IK correction ---
        if self.config['enable_ik']:
            self.logger.info("Running IK correction after OAR...")
            body_pose_new = self._apply_ik_correction(
                global_orient_w_new, body_pose_aa, betas, global_trans_new,
                static_conf_logits,
            )
        else:
            body_pose_new = body_pose_aa

        # --- 8. Update data ---
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
        }

        # --- 9. Visualisation ---
        base_output_dir = data.metadata.get('output_dir',
                                        os.path.join('results', data.sequence_name))
        vis_output_dir = os.path.join(base_output_dir, data.sequence_name)
        self._save_before_after_vis(data, pre_params, vis_output_dir)

        self.logger.info("OAR refinement complete")
        return data

    def cleanup(self):
        self._unload_metric3d()
        self.endecoder = None
        super().cleanup()
