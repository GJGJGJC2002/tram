"""DepthSceneRefineComponent - 光学锚定 + 场景感知的滑动窗口精修组件 (OAR)

基于 Metric3D 深度图 + 2D 重投影约束，以滑动窗口方式精修
global_trans / body_pose。

核心流程:
1. 加载 Metric3D 模型（窗口循环期间保持常驻）
2. FK 预计算世界坐标系关节点
3. 滑动窗口遍历序列，每个窗口：
   a. 对中心帧按需推理单张深度图
   b. Adam 优化中心帧 delta_t:
      - 重投影 loss: proj(joints) vs 2D 关键点
      - 深度 loss: 相机坐标系 z_smpl vs z_scene（深度 scale 对齐）
      - 平滑 + 正则化
   c. 高斯衰减传播 delta_t 到窗口邻居帧
   d. 窗口内滑步修正（ik: 旋转矩阵加权平均 / pp_static_joint: transl 修正）
4. 卸载 Metric3D
5. 全局高斯合并写回 data 供 eval
6. 最终重投影可视化
"""

import sys
import os
import gc
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
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
        'loss_reg_w': 0.01,
        'reproj_sigma': 50,
        'depth_sigma': 1.0,
        # 功能开关
        'enable_depth_constraint': True,
        'enable_ik': True,
        # 窗口内滑步修正模式: 'ik' / 'pp_static_joint' / 'none'
        #   ik:              旋转矩阵空间加权平均修正 body_pose
        #   pp_static_joint: 只修正 transl（静态关节位移修正平移轨迹）
        #   none:            不做窗口内滑步修正
        'window_skating_mode': 'ik',
        'enable_orient_refine': True,
        # 图像模型 3D body pose 纠正
        'enable_img_pose_refine': True,        # 是否用图像模型单帧 SMPL 纠正四肢 body_pose
        'img_pose_limb_joints': [1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21],  # FK-22 中的四肢关节
        'img_pose_opt_steps': 30,               # body pose 优化迭代次数
        'img_pose_opt_lr': 0.005,               # body pose 优化学习率
        'loss_img3d_w': 5.0,                    # 3D 关节 loss 权重
        'loss_img3d_sigma': 0.1,                # gmof sigma（米）
        'loss_bodypose_reg_w': 0.1,             # body pose 正则化权重
        # Depth Contact IK
        'enable_depth_contact_ik': True,        # 是否用深度图穿透信号做 Contact IK
        'contact_ik_steps': 30,                 # Contact IK 优化迭代次数
        'contact_ik_lr': 0.003,                 # Contact IK 学习率
        'loss_contact_w': 5.0,                  # 接触关节位置 loss 权重
        'loss_contact_sigma': 0.05,             # gmof sigma（米）
        'loss_contact_bp_reg_w': 0.1,           # body pose 正则化权重
        'loss_contact_smooth_w': 0.5,           # 时序平滑权重
        # EnDecoder
        'gvhmr_root': 'thirdparty/GVHMR',
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

    def _infer_single_depth(
        self,
        image_path: str,
        intrinsics: np.ndarray,
    ) -> torch.Tensor:
        """对单帧推理 Metric3D 深度图（假设模型已加载）。

        Args:
            image_path: 图片路径
            intrinsics: (3, 3) 相机内参

        Returns:
            depth: (H, W) CPU float tensor（米制深度）
        """
        prompthmr_abs = os.path.abspath(self.config['prompthmr_root'])
        cam_dir = os.path.join(prompthmr_abs, 'pipeline', 'camera')
        if cam_dir not in sys.path:
            sys.path.insert(0, cam_dir)
        from depth_utils import prep_metric3d, post_metric3d

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        calib = [fx, fy, cx, cy]
        model_version = self.config['metric3d_model']

        img = np.array(Image.open(image_path).convert('RGB'))
        rgb_prep, intrinsic_prep, pad_info, rgb_origin = prep_metric3d(img, calib, model_version)
        rgb_batch = rgb_prep.cuda().half()

        with torch.inference_mode():
            pred_depth, confidence, _ = self.metric3d_model.inference({'input': rgb_batch})

        depth = post_metric3d(
            pred_depth, confidence if confidence is not None else None,
            pad_info, rgb_origin, intrinsic_prep
        )
        depth_cpu = depth.cpu().squeeze()

        del rgb_batch, pred_depth, confidence
        torch.cuda.empty_cache()

        return depth_cpu

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
    # Window-level SMPL rendering (5 frames per window)
    # ------------------------------------------------------------------
    def _load_smplx_for_render(self):
        """一次性加载 SMPL-X 模型和渲染器组件，返回 (smplx_model, smplx2smpl, faces_smpl)。"""
        import sys
        gvhmr_abs = os.path.abspath(self.config['gvhmr_root'])
        if gvhmr_abs not in sys.path:
            sys.path.insert(0, gvhmr_abs)

        old_cwd = os.getcwd()
        os.chdir(gvhmr_abs)
        try:
            from hmr4d.utils.smplx_utils import make_smplx
            smplx_model = make_smplx("supermotion").to(self.device)
            smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(self.device)
            faces_smpl = make_smplx("smpl").faces
        finally:
            os.chdir(old_cwd)
        return smplx_model, smplx2smpl, faces_smpl

    def _render_window_smpl(
        self,
        center: int,
        global_orient_w: torch.Tensor,   # (F, 3) aa, full sequence
        body_pose_aa: torch.Tensor,       # (win_len, 63) window-local body pose
        betas: torch.Tensor,              # (F, 10) full sequence
        win_transl: torch.Tensor,         # (win_len, 3) window-local transl
        win_start: int,
        win_end: int,
        R_cw: torch.Tensor,               # (F, 3, 3)
        t_cw: torch.Tensor,               # (F, 3)
        K: torch.Tensor,                   # (3, 3)
        image_paths: List[str],
        output_dir: str,
        tag: str = 'pre',
        render_offsets: List[int] = None,
        smplx_model=None,
        smplx2smpl=None,
        faces_smpl=None,
        renderer=None,
    ):
        """渲染窗口内指定偏移帧的 SMPL mesh 到中心帧视角，保存 debug 图。

        对中心帧，渲染 render_offsets 指定偏移的帧（默认 -10,-5,0,+5,+10），
        所有 SMPL 用 w2c 投影到中心帧的相机坐标系下。

        Args:
            body_pose_aa: (win_len, 63) 窗口级 body pose（可能被 IK 修正过）
            global_orient_w: (F, 3) 全局序列的 orient（用全局索引访问）
            betas: (F, 10) 全局序列的 shape（用全局索引访问）
            win_transl: (win_len, 3) 窗口内独立 transl
            smplx_model, smplx2smpl, faces_smpl: 外部传入避免重复加载
            renderer: 外部传入的 Pytorch3dRenderer 实例
        """
        import cv2

        if render_offsets is None:
            render_offsets = [-10, -5, 0, 5, 10]

        vis_dir = os.path.join(output_dir, f'oar_window_{tag}')
        os.makedirs(vis_dir, exist_ok=True)

        device = torch.device(self.device)

        # 中心帧的 w2c
        R_w2c_center = R_cw[center].to(device)
        t_w2c_center = t_cw[center].to(device)

        img_bg = cv2.imread(image_paths[center])

        # 颜色表：不同偏移用不同颜色（从远过去到远未来：蓝→绿→红）
        offset_colors = {
            -10: [0.2, 0.4, 0.9],
            -5:  [0.3, 0.7, 0.9],
            0:   [0.5, 0.9, 0.5],
            5:   [0.9, 0.7, 0.3],
            10:  [0.9, 0.4, 0.2],
        }
        default_color = [0.7, 0.7, 0.7]

        canvas = img_bg.copy()

        for offset in render_offsets:
            global_idx = center + offset
            if global_idx < win_start or global_idx >= win_end:
                continue
            if global_idx < 0 or global_idx >= len(image_paths):
                continue

            local_idx = global_idx - win_start

            with torch.no_grad():
                out = smplx_model(
                    global_orient=global_orient_w[global_idx:global_idx+1].to(device),
                    body_pose=body_pose_aa[local_idx:local_idx+1].to(device),
                    betas=betas[global_idx:global_idx+1].to(device),
                    transl=win_transl[local_idx:local_idx+1].to(device),
                )
                verts_world = torch.matmul(smplx2smpl, out.vertices[0])

            verts_cam = torch.einsum('ij,vj->vi', R_w2c_center, verts_world) + t_w2c_center

            color = offset_colors.get(offset, default_color)
            try:
                canvas = renderer.render_mesh(verts_cam, canvas, colors=color)
            except Exception as e:
                self.logger.warning(f"Render failed for offset {offset}: {e}")
                continue

        # 标题和图例
        title = f"Frame {center} | Window [{win_start}:{win_end}] | {tag.upper()}"
        cv2.putText(canvas, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_legend = 50
        for offset in render_offsets:
            color = offset_colors.get(offset, default_color)
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            label = f"offset={offset:+d}"
            if offset == 0:
                label += " (center)"
            cv2.putText(canvas, label, (10, y_legend),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)
            y_legend += 18

        out_path = os.path.join(vis_dir, f'{center:05d}.jpg')
        cv2.imwrite(out_path, canvas)

    # ------------------------------------------------------------------
    # IK correction (旋转矩阵空间加权平均)
    # ------------------------------------------------------------------
    def _apply_ik_correction(
        self,
        global_orient_w: torch.Tensor,  # (F, 3) aa
        body_pose_aa: torch.Tensor,      # (F, 63)
        betas: torch.Tensor,             # (F, 10)
        global_trans: torch.Tensor,      # (F, 3)
        static_conf_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """旋转矩阵空间加权平均 IK：对静态关节的 local rotation 做帧间 SLERP。

        与原 process_ik（位置空间 rollout merge + CCD IK）不同，
        直接在旋转矩阵空间做加权平均修改 body_pose：
        1. FK 得到 local rotation matrix (B, L, 22, 3, 3)
        2. 对静态关节（脚踝、脚趾、手腕）沿运动链的 local rotation，
           用 static_conf 在相邻帧之间做加权平均
        3. 转回 axis-angle 作为修正后的 body_pose

        Returns:
            corrected body_pose: (F, 63)
        """
        from lib.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

        device = self.device
        F_total = global_orient_w.shape[0]

        # 构造 static_conf (F, 6) → sigmoid
        if static_conf_logits is not None:
            static_conf = torch.sigmoid(static_conf_logits.to(device))  # (F, 6)
        else:
            static_conf = torch.zeros(F_total, 6, device=device)

        # FK 获取 local rotation matrices
        with torch.no_grad():
            _, local_mat, _ = self.endecoder.fk_v2(
                body_pose=body_pose_aa.unsqueeze(0).to(device),
                betas=betas.unsqueeze(0).to(device),
                global_orient=global_orient_w.unsqueeze(0).to(device),
                transl=global_trans.unsqueeze(0).to(device),
                get_intermediate=True,
            )
            # local_mat: (1, F, 22, 4, 4), local_rotmat: (1, F, 22, 3, 3)

        local_rotmat = local_mat[0, :, :, :3, :3].clone()  # (F, 22, 3, 3)

        # 静态关节 IDs 和对应的运动链关节
        # joint_ids = [7, 10, 8, 11, 20, 21]
        # 对应 static_conf 列: 0=L_Ankle(7), 1=L_Foot(10), 2=R_Ankle(8),
        #                       3=R_Foot(11), 4=L_Wrist(20), 5=R_Wrist(21)
        # 修正运动链上的关节（不修 root=0）:
        chain_map = {
            0: [1, 4, 7, 10],      # L_Ankle: L_Hip(1)→L_Knee(4)→L_Ankle(7)→L_Foot(10)
            1: [1, 4, 7, 10],      # L_Foot: 同上
            2: [2, 5, 8, 11],      # R_Ankle: R_Hip(2)→R_Knee(5)→R_Ankle(8)→R_Foot(11)
            3: [2, 5, 8, 11],      # R_Foot: 同上
            4: [13, 16, 18, 20],   # L_Wrist: L_Collar(13)→L_Shoulder(16)→L_Elbow(18)→L_Wrist(20)
            5: [14, 17, 19, 21],   # R_Wrist: R_Collar(14)→R_Shoulder(17)→R_Elbow(19)→R_Wrist(21)
        }

        # 旋转矩阵空间 rollout merge:
        # 对每帧 i，如果 static_conf[i-1, j] 高，则链上关节的 local_rotmat
        # 趋近前一帧（已平均）的值
        # R_new[i] = (1 - c) * R_orig[i] + c * R_prev[i-1]
        # 在旋转矩阵空间做加权平均后 SVD 投影回 SO(3)
        merged_rotmat = local_rotmat.clone()  # (F, 22, 3, 3)

        for i in range(1, F_total):
            for conf_idx, chain_joints in chain_map.items():
                c = static_conf[min(i - 1, static_conf.shape[0] - 1), conf_idx].item()
                if c < 0.1:
                    continue
                for jid in chain_joints:
                    if jid >= merged_rotmat.shape[1]:
                        continue
                    R_prev = merged_rotmat[i - 1, jid]  # (3, 3)
                    R_curr = local_rotmat[i, jid]         # (3, 3)
                    # 旋转矩阵加权平均: R_avg = (1-c)*R_curr + c*R_prev，然后 SVD 投影到 SO(3)
                    R_avg = (1.0 - c) * R_curr + c * R_prev
                    U, _, Vh = torch.linalg.svd(R_avg)
                    R_proj = U @ Vh
                    # 确保 det > 0 (proper rotation)
                    if torch.det(R_proj) < 0:
                        U[:, -1] *= -1
                        R_proj = U @ Vh
                    merged_rotmat[i, jid] = R_proj

        # 转回 axis-angle: body_pose = joints 1-21 的 local rotation
        body_rotmat = merged_rotmat[:, 1:, :, :]  # (F, 21, 3, 3)
        body_aa = matrix_to_axis_angle(body_rotmat)  # (F, 21, 3)
        corrected_body_pose = body_aa.reshape(F_total, 63).cpu()

        self.logger.debug(
            f"[RotMat IK] Corrected body_pose for {F_total} frames, "
            f"delta_norm={((corrected_body_pose - body_pose_aa.cpu()).norm(dim=-1)).mean():.4f}"
        )

        return corrected_body_pose

    def _apply_pp_static_joint(
        self,
        global_orient_w: torch.Tensor,  # (F, 3) aa
        body_pose_aa: torch.Tensor,      # (F, 63)
        betas: torch.Tensor,             # (F, 10)
        global_trans: torch.Tensor,      # (F, 3)
        static_conf_logits: Optional[torch.Tensor],
        center_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Run GVHMR pp_static_joint to correct transl (not body_pose).

        Args:
            center_idx: 窗口中心帧的局部索引。若提供，则以中心帧为锚点对齐，
                        即保证中心帧 transl 不变，只修正其他帧的相对位移。
                        若为 None，则保持 pp_static_joint 原始行为（第一帧对齐）。

        Returns:
            corrected transl: (F, 3)
        """
        gvhmr_abs = os.path.abspath(self.config['gvhmr_root'])
        if gvhmr_abs not in sys.path:
            sys.path.insert(0, gvhmr_abs)

        from hmr4d.model.gvhmr.utils.postprocess import pp_static_joint

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
        post_w_transl = pp_static_joint(outputs, self.endecoder)  # (1, F, 3)
        result = post_w_transl[0].cpu()

        if center_idx is not None:
            # pp_static_joint 以第一帧为锚点 cumsum，并做了地面对齐 (y -= min_y)。
            # 我们希望以中心帧为锚点：计算中心帧的偏移量，整体平移回去。
            anchor_offset = global_trans[center_idx] - result[center_idx]
            result = result + anchor_offset
        else:
            # 全局调用：只还原 y 轴（地面对齐由下游处理）
            result[:, 1] = global_trans[:, 1]

        return result

    # ------------------------------------------------------------------
    # Depth Contact 检测 + Debug 可视化
    # ------------------------------------------------------------------
    def _vis_depth_contact_debug(
        self,
        center: int,
        start: int,
        end: int,
        global_orient_w: torch.Tensor,   # (F_total, 3) aa
        body_pose_aa: torch.Tensor,      # (win_len, 63) 窗口 body_pose
        betas: torch.Tensor,             # (F_total, 10)
        win_transl: torch.Tensor,        # (win_len, 3) 窗口 transl
        static_conf_logits: Optional[torch.Tensor],  # (F_total, 6)
        depth_map: torch.Tensor,         # (H, W) 中心帧深度图
        R_cw: torch.Tensor,              # (F_total, 3, 3) w2c rotation
        t_cw: torch.Tensor,              # (F_total, 3) w2c translation
        K: torch.Tensor,                 # (3, 3) intrinsics
        image_path: str,                 # 中心帧图片路径
        output_dir: str,
    ):
        """检测窗口内穿透场景的静态接触关节，生成 Contact IK 信号并可视化。

        不依赖 R_cw/t_cw 的绝对精度——仅用于投影到像素。
        通过窗口内所有关节的 z_scene/z_smpl 中位数做局部 scale 对齐，
        然后在对齐后的深度空间检测脚部穿透。

        流程：
        1. FK → 世界关节 → 中心帧相机坐标 → 像素 (u,v) + z_smpl
        2. 用中心帧所有 22 个关节计算 scale_factor = median(z_scene / z_smpl)
        3. z_smpl_aligned = z_smpl * scale_factor
        4. 对静态脚部关节：若 z_smpl_aligned > z_scene + eps → 穿透
        5. 可视化 + 返回 contact_info_list 作为 IK 信号

        颜色编码：
        - 蓝色小点：用于 scale 对齐的中心帧所有关节
        - 绿色圆圈：静态脚部、对齐后未穿透
        - 红色圆圈 + 箭头：对齐后仍穿透，需要 Contact IK
        """
        import cv2

        vis_dir = os.path.join(output_dir, 'oar_depth_contact_debug')
        os.makedirs(vis_dir, exist_ok=True)

        device = self.device
        win_len = end - start
        center_local = center - start

        # static_conf 列索引 → FK-22 关节 ID
        CONTACT_JOINTS = {
            0: (7,  'L_Ankle'),
            1: (10, 'L_Foot'),
            2: (8,  'R_Ankle'),
            3: (11, 'R_Foot'),
        }
        PENETRATION_EPS = 0.03  # 3cm 穿透阈值（对齐后用更小阈值）

        # 构造 static_conf (win_len, 6)
        if static_conf_logits is not None:
            win_static = static_conf_logits[start:end].to(device)
            static_conf = torch.sigmoid(win_static)  # (win_len, 6)
        else:
            static_conf = torch.zeros(win_len, 6, device=device)

        # FK 计算窗口内关节的世界坐标系位置
        win_orient = global_orient_w[start:end].to(device)
        win_betas = betas[start:end].to(device)
        win_bp = body_pose_aa.to(device)
        win_tr = win_transl.to(device)

        with torch.no_grad():
            joints_world = self.endecoder.fk_v2(
                body_pose=win_bp.unsqueeze(0),
                betas=win_betas.unsqueeze(0),
                global_orient=win_orient.unsqueeze(0),
                transl=win_tr.unsqueeze(0),
            )[0]  # (win_len, 22, 3)

        R_center = R_cw[center].to(device)  # (3, 3)
        t_center = t_cw[center].to(device)  # (3,)
        K_dev = K.to(device)
        depth_dev = depth_map.to(device)
        dH, dW = depth_dev.shape

        img = cv2.imread(image_path)
        if img is None:
            self.logger.warning(f"Cannot read image: {image_path}")
            return
        H, W = img.shape[:2]
        canvas = img.copy()

        # ================================================================
        # Phase 1: 用中心帧所有 22 个关节计算局部 scale_factor
        # ================================================================
        z_smpl_all = []
        z_scene_all = []
        center_joint_pixels = []  # 用于可视化 scale 对齐的采样点

        j_center_world = joints_world[center_local]  # (22, 3)
        j_center_cam = (R_center @ j_center_world.T).T + t_center  # (22, 3)
        pj_center = (K_dev @ j_center_cam.T).T  # (22, 3)
        pj_center_2d = pj_center[:, :2] / (pj_center[:, 2:3] + 1e-6)  # (22, 2)

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
            z_smpl_all.append(z_s)
            z_scene_all.append(z_d)
            center_joint_pixels.append((u_px, v_px, jid, z_s, z_d))

        if len(z_smpl_all) < 3:
            self.logger.warning(f"Window {center}: too few valid joints for scale alignment ({len(z_smpl_all)})")
            return

        z_smpl_arr = np.array(z_smpl_all)
        z_scene_arr = np.array(z_scene_all)
        ratios = z_scene_arr / (z_smpl_arr + 1e-8)
        scale_factor = float(np.median(ratios))

        self.logger.debug(
            f"[DepthContact] Window {center}: scale_factor={scale_factor:.4f} "
            f"(median of {len(ratios)} joints, range=[{ratios.min():.3f}, {ratios.max():.3f}])"
        )

        # 在画布上画 scale 对齐采样点（蓝色小点）
        for u_px, v_px, jid, zs, zd in center_joint_pixels:
            if 0 <= u_px < W and 0 <= v_px < H:
                cv2.circle(canvas, (u_px, v_px), 2, (255, 150, 0), -1)

        # ================================================================
        # Phase 2: 对齐后检测脚部静态关节穿透
        # ================================================================
        n_static = 0
        n_penetrate = 0
        contact_info_list = []

        for fi in range(win_len):
            frame_global = start + fi
            for conf_idx, (jid, jname) in CONTACT_JOINTS.items():
                conf_val = static_conf[fi, conf_idx].item()
                if conf_val < 0.5:
                    continue

                n_static += 1

                j_world = joints_world[fi, jid]
                j_cam = R_center @ j_world + t_center
                z_smpl_raw = j_cam[2].item()
                z_smpl_aligned = z_smpl_raw * scale_factor

                pj = K_dev @ j_cam
                u = pj[0].item() / (pj[2].item() + 1e-6)
                v = pj[1].item() / (pj[2].item() + 1e-6)
                ui, vi = int(round(u)), int(round(v))

                if ui < 0 or ui >= dW or vi < 0 or vi >= dH:
                    continue
                if ui < 0 or ui >= W or vi < 0 or vi >= H:
                    continue

                z_scene = depth_dev[vi, ui].item()
                if z_scene < 0.1 or z_scene > 100.0:
                    continue

                penetration = z_smpl_aligned - z_scene - PENETRATION_EPS
                is_penetrating = penetration > 0

                if is_penetrating:
                    n_penetrate += 1

                contact_info_list.append({
                    'frame': frame_global,
                    'fi': fi,
                    'jid': jid,
                    'jname': jname,
                    'conf': conf_val,
                    'u': ui, 'v': vi,
                    'z_smpl_raw': z_smpl_raw,
                    'z_smpl_aligned': z_smpl_aligned,
                    'z_scene': z_scene,
                    'penetration': penetration if is_penetrating else 0.0,
                    'is_penetrating': is_penetrating,
                })

        # ---- 绘制接触点 ----
        for info in contact_info_list:
            u, v = info['u'], info['v']
            if info['is_penetrating']:
                radius = max(5, min(14, int(info['penetration'] * 200)))
                cv2.circle(canvas, (u, v), radius, (0, 0, 255), 2)
                arrow_len = max(12, min(50, int(info['penetration'] * 400)))
                cv2.arrowedLine(canvas, (u, v), (u, v - arrow_len),
                                (0, 0, 255), 2, tipLength=0.3)
                label = f"{info['jname']}@f{info['frame']} pen={info['penetration']:.3f}m"
                cv2.putText(canvas, label, (u + 5, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1)
            else:
                cv2.circle(canvas, (u, v), 3, (0, 200, 0), -1)
                label = f"{info['jname']}@f{info['frame']}"
                cv2.putText(canvas, label, (u + 3, v - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, (0, 200, 0), 1)

        # ---- 图例和统计 ----
        cv2.putText(canvas,
                    f"Window center={center} | scale={scale_factor:.3f} | "
                    f"Static: {n_static} | Penetrating: {n_penetrate}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(canvas,
                    "Blue=scale_ref  Green=static(OK)  Red+arrow=penetrating(contact IK)",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        if contact_info_list:
            pen_items = [c for c in contact_info_list if c['is_penetrating']]
            if pen_items:
                y_offset = 50
                cv2.putText(canvas, f"Penetration Summary (scale={scale_factor:.3f}):",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 150, 255), 1)
                y_offset += 18
                for item in pen_items[:15]:
                    txt = (f"  f{item['frame']} {item['jname']}: "
                           f"z_raw={item['z_smpl_raw']:.2f} z_align={item['z_smpl_aligned']:.2f} "
                           f"z_scene={item['z_scene']:.2f} pen={item['penetration']:.3f}m")
                    cv2.putText(canvas, txt, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 150, 255), 1)
                    y_offset += 14
                if len(pen_items) > 15:
                    cv2.putText(canvas, f"  ... and {len(pen_items) - 15} more",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 150, 255), 1)

        out_path = os.path.join(vis_dir, f'{center:05d}.jpg')
        cv2.imwrite(out_path, canvas)

        if n_penetrate > 0:
            self.logger.debug(
                f"[DepthContact] Window {center}: scale={scale_factor:.3f}, "
                f"{n_static} static, {n_penetrate} penetrating"
            )

        return contact_info_list

    # ------------------------------------------------------------------
    # Depth Contact IK: 基于深度图穿透信号优化 body_pose
    # ------------------------------------------------------------------
    def _apply_depth_contact_ik(
        self,
        global_orient_w: torch.Tensor,   # (F_total, 3) aa
        body_pose_aa: torch.Tensor,      # (win_len, 63) 窗口 body_pose
        betas: torch.Tensor,             # (F_total, 10)
        win_transl: torch.Tensor,        # (win_len, 3) 窗口 transl
        static_conf_logits: Optional[torch.Tensor],  # (F_total, 6)
        depth_map: torch.Tensor,         # (H, W) 中心帧深度图
        R_cw: torch.Tensor,              # (F_total, 3, 3)
        t_cw: torch.Tensor,              # (F_total, 3)
        K: torch.Tensor,                 # (3, 3)
        center: int,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """基于深度图的 Contact IK：优化窗口内 body_pose 使穿透接触关节对齐场景表面。

        流程：
        1. 用中心帧所有 22 个关节计算局部 scale_factor = median(z_scene / z_smpl)
        2. 对窗口内每帧的静态脚部关节，计算 scale-aligned 后的目标 3D 位置
           （从深度图反投影得到相机坐标系下目标点，再转回世界坐标系）
        3. Adam 优化 delta_body_pose（只修改腿部运动链），使 FK 后关节到达目标位置
        4. 同时加入时序平滑正则 + body_pose 正则

        只修改腿部运动链：L_Hip(1)→L_Knee(4)→L_Ankle(7)→L_Foot(10)
                          R_Hip(2)→R_Knee(5)→R_Ankle(8)→R_Foot(11)

        Returns:
            refined body_pose: (win_len, 63)
        """
        device = self.device
        win_len = end - start
        center_local = center - start

        # --- 配置 ---
        opt_steps = self.config.get('contact_ik_steps', 30)
        lr = self.config.get('contact_ik_lr', 0.003)
        loss_contact_w = self.config.get('loss_contact_w', 5.0)
        loss_contact_sigma = self.config.get('loss_contact_sigma', 0.05)
        loss_bp_reg_w = self.config.get('loss_contact_bp_reg_w', 0.1)
        loss_smooth_w = self.config.get('loss_contact_smooth_w', 0.5)
        pen_eps = 0.03  # 3cm 穿透阈值

        # --- static_conf ---
        CONTACT_JOINTS = {
            0: 7,   # L_Ankle
            1: 10,  # L_Foot
            2: 8,   # R_Ankle
            3: 11,  # R_Foot
        }
        # 腿部运动链关节 (body_pose 索引)
        LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
        leg_bp_indices = []
        for j in LEG_JOINTS:
            leg_bp_indices.extend([(j-1)*3, (j-1)*3+1, (j-1)*3+2])

        if static_conf_logits is not None:
            static_conf = torch.sigmoid(static_conf_logits[start:end].to(device))
        else:
            static_conf = torch.zeros(win_len, 6, device=device)

        # --- FK 计算当前关节位置 ---
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

        # --- Phase 1: 计算局部 scale_factor ---
        with torch.no_grad():
            joints_world_init = self.endecoder.fk_v2(
                body_pose=win_bp.unsqueeze(0),
                betas=win_betas.unsqueeze(0),
                global_orient=win_orient.unsqueeze(0),
                transl=win_tr.unsqueeze(0),
            )[0]  # (win_len, 22, 3)

        j_center_world = joints_world_init[center_local]  # (22, 3)
        j_center_cam = (R_center @ j_center_world.T).T + t_center  # (22, 3)
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
            self.logger.debug(f"[ContactIK] Window {center}: too few joints for scale, skipping")
            return body_pose_aa

        scale_factor = float(np.median(np.array(z_scene_list) / (np.array(z_smpl_list) + 1e-8)))

        # --- Phase 2: 为每帧每个穿透的静态脚部关节计算目标世界坐标 ---
        # contact_targets[fi][jid] = target_world_pos (3,)
        contact_targets = {}  # {fi: {jid: (3,) tensor}}
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
                    continue  # 没穿透，不需要修正

                # 用深度图反投影得到目标相机坐标系 3D 位置
                # p_target_cam = z_scene * K^{-1} @ [u, v, 1]^T
                # 但注意 z_scene 是 metric depth，而我们的相机坐标系有 scale 偏差
                # 所以目标深度应该是 z_scene / scale_factor（转回 SMPL 的 scale 空间）
                z_target_smpl_scale = z_scene / scale_factor
                pixel_homo = torch.tensor([u, v, 1.0], device=device)
                p_target_cam = z_target_smpl_scale * (K_inv @ pixel_homo)  # (3,)

                # 转回世界坐标系：p_world = R_center^T @ (p_cam - t_center)
                p_target_world = R_center.T @ (p_target_cam - t_center)

                contact_targets.setdefault(fi, {})[jid] = p_target_world.detach()
                n_targets += 1

        if n_targets == 0:
            self.logger.debug(f"[ContactIK] Window {center}: no penetrating contacts, skipping")
            return body_pose_aa

        self.logger.debug(
            f"[ContactIK] Window {center}: {n_targets} targets, "
            f"scale={scale_factor:.3f}, optimizing {opt_steps} steps"
        )

        # --- Phase 3: Adam 优化 delta_body_pose ---
        bp_orig = win_bp.clone().detach()
        delta_bp = torch.zeros(win_len, len(leg_bp_indices), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([delta_bp], lr=lr)

        for step in range(opt_steps):
            optimizer.zero_grad()

            bp_mod = bp_orig.clone()
            bp_mod[:, leg_bp_indices] = bp_orig[:, leg_bp_indices] + delta_bp

            # 全局 FK（带 global_orient 和 transl）
            with torch.enable_grad():
                joints_world_opt = self.endecoder.fk_v2(
                    body_pose=bp_mod.unsqueeze(0),
                    betas=win_betas.unsqueeze(0),
                    global_orient=win_orient.unsqueeze(0),
                    transl=win_tr.unsqueeze(0),
                )[0]  # (win_len, 22, 3)

            # Loss 1: 接触关节到目标位置的距离
            loss_contact = torch.tensor(0.0, device=device)
            cnt = 0
            for fi, targets in contact_targets.items():
                for jid, target_pos in targets.items():
                    diff = joints_world_opt[fi, jid] - target_pos
                    loss_contact = loss_contact + gmof(diff, sigma=loss_contact_sigma).sum()
                    cnt += 1
            if cnt > 0:
                loss_contact = loss_contact / cnt

            # Loss 2: body_pose 正则化（不偏离原始太多）
            loss_reg = delta_bp.pow(2).mean()

            # Loss 3: 时序平滑（相邻帧的 delta 应该相近）
            loss_smooth = torch.tensor(0.0, device=device)
            if win_len >= 2:
                delta_diff = delta_bp[1:] - delta_bp[:-1]
                loss_smooth = delta_diff.pow(2).mean()

            total_loss = (
                loss_contact_w * loss_contact
                + loss_bp_reg_w * loss_reg
                + loss_smooth_w * loss_smooth
            )

            total_loss.backward()
            optimizer.step()

            if step == 0 or step == opt_steps - 1:
                self.logger.debug(
                    f"[ContactIK step={step}] "
                    f"contact={loss_contact.item():.6f}, "
                    f"reg={loss_reg.item():.6f}, "
                    f"smooth={loss_smooth.item():.6f}, "
                    f"total={total_loss.item():.6f}"
                )

        # --- Phase 4: 应用修正 ---
        with torch.no_grad():
            refined_bp = bp_orig.clone()
            refined_bp[:, leg_bp_indices] = bp_orig[:, leg_bp_indices] + delta_bp.detach()

        delta_norm = delta_bp.detach().abs().mean().item()
        self.logger.debug(
            f"[ContactIK] Window {center}: done, "
            f"delta_bp_mean={delta_norm:.6f}, {n_targets} targets"
        )

        return refined_bp.cpu()

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
            #print("saving:", out_path) 
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
    # Window-level optimization (single window)
    # ------------------------------------------------------------------
    def _optimize_single_window(
        self,
        center: int,
        start: int,
        end: int,
        joints_world: torch.Tensor,  # (F, J, 3)
        R_cw: torch.Tensor,
        t_cw: torch.Tensor,
        K: torch.Tensor,
        kp2d: Optional[torch.Tensor],
        depth_map: Optional[torch.Tensor],  # (H, W) 中心帧的单张深度图，或 None
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """对单个窗口做 depth + 2D KP refine，返回中心帧的 delta_t (3,)。

        深度约束直接在相机坐标系下比较：
        - SMPL 关节 w2c 投影得到 z_smpl
        - 深度图在关节投影像素位置采样得到 z_scene
        - 只修正 transl（本质是深度 scale 对齐）
        """
        device = self.device
        fps = self.config['fps']
        win_len = end - start
        center_local = center - start

        j_win = joints_world[start:end].to(device)
        R_win = R_cw[start:end].to(device)
        t_win = t_cw[start:end].to(device)

        from lib.utils.rotation_conversions import axis_angle_to_matrix

        delta_t = torch.zeros(3, device=device, requires_grad=True)

        if self.config['enable_orient_refine']:
            delta_orient_aa = torch.zeros(3, device=device, requires_grad=True)
            opt_params = [delta_t, delta_orient_aa]
        else:
            delta_orient_aa = torch.zeros(3, device=device)
            opt_params = [delta_t]

        optimizer = torch.optim.Adam(opt_params, lr=self.config['opt_lr'])

        # FK→COCO mapping
        n_joints = joints_world.shape[1]
        _FK_TO_COCO_PAIRS = [
            (1, 11), (2, 12), (4, 13), (5, 14), (7, 15), (8, 16),
            (16, 5), (17, 6), (18, 7), (19, 8), (20, 9), (21, 10),
        ]
        fk_coco_pairs = [(fk, coco) for fk, coco in _FK_TO_COCO_PAIRS if fk < n_joints]
        fk_pair_idxs = [p[0] for p in fk_coco_pairs]
        coco_pair_idxs = [p[1] for p in fk_coco_pairs]

        # Gaussian weights for smoothness
        half_win = self.config['window_size'] // 2
        sigma = self.config['sigma']
        offsets = torch.arange(-half_win, half_win + 1).float()
        gauss_w = torch.exp(-offsets ** 2 / (2 * sigma ** 2))
        gauss_w = gauss_w / gauss_w.max()

        K_dev = K.to(device)

        # 预处理深度图
        depth_dev = None
        if depth_map is not None and self.config['enable_depth_constraint']:
            depth_dev = depth_map.to(device)

        for step in range(self.config['opt_steps']):
            optimizer.zero_grad()

            delta_R = axis_angle_to_matrix(delta_orient_aa.unsqueeze(0))
            j_center_mod = (delta_R @ j_win[center_local].unsqueeze(-1)).squeeze(-1) + delta_t

            j_cam_center = (R_win[center_local] @ j_center_mod.unsqueeze(-1)).squeeze(-1) + t_win[center_local]
            pj = (K_dev @ j_cam_center.unsqueeze(-1)).squeeze(-1)
            pj_2d = pj[:, :2] / (pj[:, 2:3] + 1e-6)

            total_loss = torch.tensor(0.0, device=device)

            # Loss 1: 2D reprojection
            if kp2d is not None and len(fk_coco_pairs) > 0:
                kp_center = kp2d[center].to(device)
                gt_2d = kp_center[coco_pair_idxs, :2]
                conf = kp_center[coco_pair_idxs, 2]
                proj_2d = pj_2d[fk_pair_idxs]
                reproj_err = gmof(proj_2d - gt_2d, sigma=self.config['reproj_sigma'])
                reproj_err = reproj_err / 1000.0
                conf_mask = conf > 0.5
                loss_reproj = (conf_mask.unsqueeze(-1) * reproj_err).mean()
                total_loss = total_loss + self.config['loss_reproj_w'] * loss_reproj

            # Loss 2: Depth alignment (相机坐标系 z_smpl vs z_scene)
            if depth_dev is not None:
                dH, dW = depth_dev.shape
                u = pj_2d[:, 0].long().clamp(0, dW - 1)
                v = pj_2d[:, 1].long().clamp(0, dH - 1)
                z_smpl = j_cam_center[:, 2]
                z_scene = depth_dev[v, u]
                valid_depth = (z_scene > 0.1) & (z_scene < 100.0) & (z_smpl > 0)
                penetration = F.relu(z_smpl - z_scene - 0.05)
                if valid_depth.any():
                    loss_depth = gmof(penetration[valid_depth], sigma=self.config['depth_sigma']).mean()
                else:
                    loss_depth = torch.tensor(0.0, device=device)
                total_loss = total_loss + self.config['loss_depth_w'] * loss_depth

            # Loss 3: Smoothness
            t_weights = gauss_w[start - center + half_win:end - center + half_win].to(device)
            delta_t_propagated = t_weights.unsqueeze(-1) * delta_t.unsqueeze(0)
            j_mod_transl = j_win[:, 0, :] + delta_t_propagated
            if win_len >= 3:
                vel = (j_mod_transl[1:] - j_mod_transl[:-1]) * fps
                loss_vel = vel.pow(2).mean()
                acc = (j_mod_transl[2:] + j_mod_transl[:-2] - 2 * j_mod_transl[1:-1]) * fps
                loss_acc = acc.norm(dim=-1).mean()
                total_loss = total_loss + self.config['loss_smooth_vel_w'] * loss_vel
                total_loss = total_loss + self.config['loss_smooth_acc_w'] * loss_acc

            # Loss 4: Regularization
            loss_reg = delta_t.norm() + delta_orient_aa.norm()
            total_loss = total_loss + self.config['loss_reg_w'] * loss_reg

            total_loss.backward()
            optimizer.step()

        return delta_t.detach().cpu()

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------
    def execute(self, data: PipelineData) -> PipelineData:
        """窗口级独立 Pipeline：每个窗口独立创建 transl 副本，
        渲染 pre → depth+2DKP refine → IK → 渲染 post。
        同时仍将全局修正（高斯合并）写回 data.smpl_params 供 eval 使用。
        """
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
        window_size = self.config['window_size']
        stride = self.config['stride']
        sigma = self.config['sigma']
        half_win = window_size // 2
        self.logger.info(f"OAR: {F_total} frames, window={window_size}, stride={stride}")

        # --- 1. Get camera w2c ---
        wt = data.metadata.get('world_transform', {})
        R_wc = _to_tensor(wt.get('R_wc', cam.world_R if cam.world_R is not None else cam.R))
        T_wc = _to_tensor(wt.get('T_wc', cam.world_T if cam.world_T is not None else cam.T))
        R_cw, t_cw = self._build_w2c(R_wc, T_wc)

        # K selection (HPE K for projection, GT K for depth)
        K_gt = _to_tensor(cam.intrinsics)
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
            img0 = np.array(Image.open(data.image_paths[0]))
            H, W = img0.shape[:2]
            f_est = (H**2 + W**2) ** 0.5
            K = torch.tensor([[f_est, 0, W/2.], [0, f_est, H/2.], [0, 0, 1]], dtype=torch.float)
            if K_gt is not None:
                self.logger.info(
                    f"No HPE estimate_K in metadata, computed estimate_K "
                    f"(fx={f_est:.1f}) instead of GT K (fx={float(K_gt[0,0]):.1f})"
                )
            else:
                self.logger.info(f"Using computed estimate_K: fx={f_est:.1f}")
        K_for_depth = K_gt if K_gt is not None else K

        # --- 2. 加载 Metric3D（窗口循环内按需推理，循环后卸载） ---
        intrinsics_np = K_for_depth.numpy() if isinstance(K_for_depth, torch.Tensor) else K_for_depth
        if self.config['enable_depth_constraint']:
            gc.collect()
            torch.cuda.empty_cache()
            self._load_metric3d()
            self.logger.info("Metric3D loaded (will infer per-window center frame on demand)")

        # --- 3. FK: compute world joints --- #这是前向传播吗？
        from lib.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
        with torch.no_grad():
            joints_world = self.endecoder.fk_v2(
                body_pose=body_pose_aa.unsqueeze(0).to(self.device),
                betas=betas.unsqueeze(0).to(self.device),
                global_orient=global_orient_w.unsqueeze(0).to(self.device),
                transl=global_trans.unsqueeze(0).to(self.device),
            )
            joints_world = joints_world[0].cpu()  # (F, 22, 3)

        # --- 4. Prepare 2D keypoints ---
        kp2d = None
        if 'vitpose_kp2d' in data.metadata:
            kp2d = _to_tensor(data.metadata['vitpose_kp2d'])
            self.logger.info(f"Using ViTPose COCO-17 keypoints: {kp2d.shape}")
        if kp2d is None:
            self.logger.info("No external 2D keypoints, using projected joints as pseudo-kp2d")
            pj2d, _ = self._project_joints(joints_world[:, :22], R_cw, t_cw, K)
            kp2d = torch.zeros(F_total, 17, 3)
            _fk2coco = {1:11, 2:12, 4:13, 5:14, 7:15, 8:16, 13:5, 14:6, 18:7, 19:8, 20:9, 21:10}
            for fk_idx, coco_idx in _fk2coco.items():
                kp2d[:, coco_idx, :2] = pj2d[:, fk_idx]
                kp2d[:, coco_idx, 2] = 1.0

        img0 = np.array(Image.open(data.image_paths[0]))
        img_h, img_w = img0.shape[:2]

        # --- 5. Load SMPL-X model for rendering (一次性加载) ---
        import cv2
        smplx_model, smplx2smpl, faces_smpl = None, None, None
        p3d_renderer = None
        vis_output_dir = os.path.join(
            data.metadata.get('output_dir', os.path.join('results', data.sequence_name)),
            data.sequence_name,
        )
        try:
            smplx_model, smplx2smpl, faces_smpl = self._load_smplx_for_render()
            from lib.vis.renderer import Renderer as Pytorch3dRenderer
            focal_length = float(K[0, 0])
            p3d_renderer = Pytorch3dRenderer(img_w, img_h, focal_length, self.device, faces_smpl)
            self.logger.info("Loaded SMPL-X + Renderer for window visualization")
        except Exception as e:
            self.logger.warning(f"Cannot load SMPL-X/Renderer for window vis: {e}")

        # --- 6. Sliding-window pipeline: per-window independent ---
        # 渲染偏移量
        render_offsets = self.config.get('render_offsets', [-10, -5, 0, 5, 10])

        # gauss_w 在窗口内优化中仍有使用
        gauss_offsets = torch.arange(-half_win, half_win + 1).float()
        gauss_w = torch.exp(-gauss_offsets ** 2 / (2 * sigma ** 2))
        gauss_w = gauss_w / gauss_w.max()

        # 存储每个窗口独立结果
        window_results = []

        num_windows = 0
        for center in range(0, F_total, stride):
            start = max(0, center - half_win)
            end = min(F_total, center + half_win + 1)
            win_len = end - start

            # ---- Step A: 创建窗口独立 transl 副本 ----
            win_transl = global_trans[start:end].clone()  # (win_len, 3) 独立副本
            win_body_pose = body_pose_aa[start:end].clone()  # (win_len, 63) 独立副本

            # ---- Step B: 渲染 PRE-refine 的 5 帧 SMPL (debug) ----
            if smplx_model is not None and p3d_renderer is not None:
                try:
                    self._render_window_smpl(
                        center=center,
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,  # 用窗口的 body_pose（此时还是原始的）
                        betas=betas,
                        win_transl=win_transl,
                        win_start=start,
                        win_end=end,
                        R_cw=R_cw, t_cw=t_cw, K=K,
                        image_paths=data.image_paths,
                        output_dir=vis_output_dir,
                        tag='pre',
                        render_offsets=render_offsets,
                        smplx_model=smplx_model,
                        smplx2smpl=smplx2smpl,
                        faces_smpl=faces_smpl,
                        renderer=p3d_renderer,
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} pre-render failed: {e}")

            # ---- Step B2: PRE-render 后先做 pp_static_joint 纠正滑步 ----
            win_orient = global_orient_w[start:end]
            win_betas = betas[start:end]
            win_static = static_conf_logits[start:end] if static_conf_logits is not None else None
            center_local = center - start
            win_transl = self._apply_pp_static_joint(
                win_orient, win_body_pose, win_betas, win_transl, win_static,
                center_idx=center_local,
            )

            # ---- Step C: 中心帧深度图推理（按需，相机坐标系） ----
            center_depth_map = None
            if self.config['enable_depth_constraint'] and self.metric3d_model is not None:
                try:
                    center_depth_map = self._infer_single_depth(
                        data.image_paths[center], intrinsics_np,
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} depth inference failed: {e}")

            # ---- Step C2: Depth Contact Debug 可视化 ----
            if center_depth_map is not None:
                try:
                    self._vis_depth_contact_debug(
                        center=center,
                        start=start,
                        end=end,
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,
                        betas=betas,
                        win_transl=win_transl,
                        static_conf_logits=static_conf_logits,
                        depth_map=center_depth_map,
                        R_cw=R_cw, t_cw=t_cw, K=K,
                        image_path=data.image_paths[center],
                        output_dir=vis_output_dir,
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} depth contact debug failed: {e}")

            # ---- Step E: 渲染 POST-refine 的 5 帧 SMPL (debug) ----
            if smplx_model is not None and p3d_renderer is not None:
                try:
                    self._render_window_smpl( #会在这里进行w2c变换
                        center=center,
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,
                        betas=betas,
                        win_transl=win_transl,
                        win_start=start,
                        win_end=end,
                        R_cw=R_cw, t_cw=t_cw, K=K,
                        image_paths=data.image_paths,
                        output_dir=vis_output_dir,
                        tag='post_pre_dep',
                        render_offsets=render_offsets,
                        smplx_model=smplx_model,
                        smplx2smpl=smplx2smpl,
                        faces_smpl=faces_smpl,
                        renderer=p3d_renderer,
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} post-render failed: {e}")

            # ---- Step C3: Depth Contact IK (基于深度穿透信号优化 body_pose) ----
            if center_depth_map is not None and self.config.get('enable_depth_contact_ik', True):
                try:
                    win_body_pose = self._apply_depth_contact_ik(
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,
                        betas=betas,
                        win_transl=win_transl,
                        static_conf_logits=static_conf_logits,
                        depth_map=center_depth_map,
                        R_cw=R_cw, t_cw=t_cw, K=K,
                        center=center,
                        start=start,
                        end=end,
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} depth contact IK failed: {e}")

            # ---- Step E: 渲染 POST-refine 的 5 帧 SMPL (debug) ----
            if smplx_model is not None and p3d_renderer is not None:
                try:
                    self._render_window_smpl( #会在这里进行w2c变换
                        center=center,
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,
                        betas=betas,
                        win_transl=win_transl,
                        win_start=start,
                        win_end=end,
                        R_cw=R_cw, t_cw=t_cw, K=K,
                        image_paths=data.image_paths,
                        output_dir=vis_output_dir,
                        tag='post',
                        render_offsets=render_offsets,
                        smplx_model=smplx_model,
                        smplx2smpl=smplx2smpl,
                        faces_smpl=faces_smpl,
                        renderer=p3d_renderer,
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} post-render failed: {e}")

            # ---- Step F: 保存窗口结果 ----
            window_results.append({
                'center': center,
                'start': start,
                'end': end,
                'transl': win_transl.clone(),        # (win_len, 3) 独立副本
                'body_pose': win_body_pose.clone(),   # (win_len, 63)
            })

            num_windows += 1
            if num_windows % 10 == 0:
                self.logger.info(f"Processed {num_windows} windows (center={center})")

        self.logger.info(f"Processed {num_windows} windows total")

        # --- 7. 释放 Metric3D + 渲染模型 ---
        self._unload_metric3d()
        if smplx_model is not None:
            del smplx_model, smplx2smpl, p3d_renderer
            torch.cuda.empty_cache()

        # --- 8. 全局修正（高斯加权合并 window_results）写回 data 供 eval ---
        # 用 window_results 中每个窗口的 transl 和 body_pose 进行高斯距离加权合并
        acc_transl = torch.zeros(F_total, 3)
        acc_body_pose = torch.zeros(F_total, 63)
        acc_t_weights = torch.zeros(F_total)
        acc_bp_weights = torch.zeros(F_total)

        for wr in window_results:
            w_center, w_start, w_end = wr['center'], wr['start'], wr['end']
            w_transl = wr['transl']       # (win_len, 3)
            w_bp = wr['body_pose']         # (win_len, 63)
            w_len = w_end - w_start

            # 高斯权重：基于帧到中心帧的距离
            frame_indices = torch.arange(w_start, w_end).float()
            offsets = frame_indices - float(w_center)
            w_gauss = torch.exp(-offsets ** 2 / (2 * sigma ** 2))
            w_gauss = w_gauss / w_gauss.max()  # 中心帧权重=1

            acc_transl[w_start:w_end] += w_gauss.unsqueeze(-1) * w_transl
            acc_t_weights[w_start:w_end] += w_gauss

            acc_body_pose[w_start:w_end] += w_gauss.unsqueeze(-1) * w_bp
            acc_bp_weights[w_start:w_end] += w_gauss

        # 归一化 transl
        t_valid = acc_t_weights > 1e-6
        global_trans_new = global_trans.clone()
        global_trans_new[t_valid] = acc_transl[t_valid] / acc_t_weights[t_valid].unsqueeze(-1)

        # 归一化 body_pose
        bp_valid = acc_bp_weights > 1e-6
        body_pose_new = body_pose_aa.clone()
        body_pose_new[bp_valid] = acc_body_pose[bp_valid] / acc_bp_weights[bp_valid].unsqueeze(-1)

        self.logger.info(
            f"OAR global merge - transl delta: mean={( global_trans_new - global_trans).abs().mean():.4f}, "
            f"max={(global_trans_new - global_trans).abs().max():.4f}, "
            f"body_pose delta: mean={(body_pose_new - body_pose_aa).abs().mean():.4f}, "
            f"max={(body_pose_new - body_pose_aa).abs().max():.4f}"
        )

        # --- 9. Image model 3D body pose refinement (保留但可关闭) ---
        phmr_img_smpl = data.metadata.get('phmr_img_smpl', None)
        if (self.config['enable_img_pose_refine']
                and phmr_img_smpl is not None):
            self.logger.info("Running image-model 3D body pose refinement (body-local space)...")
            img_j3d_local = self._prepare_img_model_joints(phmr_img_smpl, betas)
            body_pose_new = self._refine_body_pose_with_img_model(
                body_pose_new, betas, img_j3d_local,
            )
        elif self.config['enable_img_pose_refine'] and phmr_img_smpl is None:
            self.logger.info("Image-model body pose refinement enabled but no phmr_img_smpl, skipping")

        # --- 11. Update data ---
        data.smpl_params.global_orient_w = global_orient_w
        data.smpl_params.global_trans = global_trans_new
        data.smpl_params.transl_w_raw = global_trans_new.clone()
        data.smpl_params.body_pose_aa = body_pose_new

        # Store window results in metadata for downstream (adjacent_render, SLAM)
        data.metadata['oar_window_results'] = window_results

        data.metadata['oar_refine'] = {
            'window_size': window_size,
            'stride': stride,
            'opt_steps': self.config['opt_steps'],
            'delta_transl_mean': (global_trans_new - global_trans).abs().mean().item(),
            'num_windows': num_windows,
            'enable_depth_constraint': self.config['enable_depth_constraint'],
            'enable_ik': self.config['enable_ik'],
            'enable_orient_refine': self.config['enable_orient_refine'],
            'enable_img_pose_refine': self.config['enable_img_pose_refine'],
            'img_pose_refined': (
                self.config['enable_img_pose_refine']
                and phmr_img_smpl is not None
            ),
        }

        # --- 12. Visualisation (before/after comparison) ---
        self._save_before_after_vis(data, pre_params, vis_output_dir)

        # --- 12b. 最终重投影可视化（所有窗口+IK处理完的最终结果） ---
        self.logger.info("Generating final reprojection visualization...")
        try:
            with torch.no_grad():
                joints_world_final = self.endecoder.fk_v2(
                    body_pose=body_pose_new.unsqueeze(0).to(self.device),
                    betas=betas.unsqueeze(0).to(self.device),
                    global_orient=global_orient_w.unsqueeze(0).to(self.device),
                    transl=global_trans_new.unsqueeze(0).to(self.device),
                )[0].cpu()

            proj_2d_pre = self._compute_per_frame_projections(joints_world, R_cw, t_cw, K)
            proj_2d_final = self._compute_per_frame_projections(joints_world_final, R_cw, t_cw, K)

            vis_stride = self.config.get('vis_oar_frame_stride', 5)
            if kp2d is not None:
                self._vis_reprojection_error(
                    kp2d, proj_2d_pre, proj_2d_final,
                    data.image_paths, vis_output_dir,
                    frame_stride=vis_stride,
                )
                self._vis_keypoints_2d(
                    kp2d, proj_2d_final, data.image_paths,
                    vis_output_dir, frame_stride=vis_stride, tag='final',
                )
        except Exception as e:
            self.logger.warning(f"Final reprojection visualization failed: {e}")

        # --- 12c. OAR 诊断可视化 ---
        if self.config.get('vis_oar_diagnostics', False):
            self.logger.info("Generating OAR diagnostic visualizations...")
            vis_stride = self.config.get('vis_oar_frame_stride', 5)

            proj_2d_pre = self._compute_per_frame_projections(joints_world, R_cw, t_cw, K)
            _, joints_cam_pre = self._project_joints(joints_world, R_cw, t_cw, K)

            with torch.no_grad():
                joints_world_post = self.endecoder.fk_v2(
                    body_pose=body_pose_new.unsqueeze(0).to(self.device),
                    betas=betas.unsqueeze(0).to(self.device),
                    global_orient=global_orient_w.unsqueeze(0).to(self.device),
                    transl=global_trans_new.unsqueeze(0).to(self.device),
                )
                joints_world_post = joints_world_post[0].cpu()

            proj_2d_post = self._compute_per_frame_projections(
                joints_world_post, R_cw, t_cw, K
            )

            try:
                if kp2d is not None:
                    self._vis_keypoints_2d(
                        kp2d, proj_2d_pre, data.image_paths,
                        vis_output_dir, frame_stride=vis_stride, tag='pre',
                    )
                    self._vis_keypoints_2d(
                        kp2d, proj_2d_post, data.image_paths,
                        vis_output_dir, frame_stride=vis_stride, tag='post',
                    )
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
