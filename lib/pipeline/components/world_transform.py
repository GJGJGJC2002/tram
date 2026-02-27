"""WorldTransformComponent - 用 SLAM 相机参数将相机坐标系 SMPL 转换到世界坐标系

当 PromptHMR 以 static_cam=True 运行时，其视频头输出的世界坐标不可靠
（因为没有输入真实的相机运动）。此组件用 PromptHMR 的相机坐标系输出
(global_orient_c, trans) + SLAM 组件得到的相机外参 (R_wc, T_wc)
重新计算准确的世界坐标系 SMPL 参数。

参考: PromptHMR/pipeline/world.py 中的 transform_smpl_params 和 world_hps_estimation
"""

import os
from typing import Dict, Any
import numpy as np
import torch

from lib.pipeline.core.component import Component
from lib.pipeline.core.data import PipelineData


class WorldTransformComponent(Component):
    """
    世界坐标系变换组件

    从 PromptHMR 的相机坐标系 SMPL 参数 + SLAM 相机外参，
    计算世界坐标系下的 global_orient 和 transl。

    Config:
        use_smooth_cam: 是否对相机参数做 One Euro 平滑 (默认 True)
        use_spec_calib: 是否使用 SPEC 重力校准 (默认 False, align_to_world 已处理)
        smplx_model_path: SMPLX 模型路径 (用于计算 T-pose pelvis offset)
        device: 计算设备
    """

    COMPONENT_TYPE = "world_transform"

    DEFAULT_CONFIG = {
        'use_smooth_cam': True,
        'smplx_model_path': None,
        'device': 'cuda',
    }

    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)

        self.use_smooth_cam = self.config['use_smooth_cam']
        self.smplx_model_path = self.config.get('smplx_model_path')
        self.smplx_model = None

    def validate_input(self, data: PipelineData) -> bool:
        """验证输入"""
        if data.smpl_params is None:
            self.logger.warning("No smpl_params found")
            return False

        sp = data.smpl_params
        if sp.global_orient_c is None or sp.trans is None:
            self.logger.warning(
                "smpl_params missing camera-space fields (global_orient_c, trans). "
                "Ensure PromptHMR backend was used."
            )
            return False

        if data.camera_params is None:
            self.logger.warning("No camera_params found (need SLAM output)")
            return False

        # 需要 c2w 相机参数
        has_cam = (
            (data.camera_params.world_R is not None and data.camera_params.world_T is not None)
            or (data.camera_params.R is not None and data.camera_params.T is not None)
        )
        if not has_cam:
            self.logger.warning("camera_params has no R/T (need SLAM c2w output)")
            return False

        return True

    def setup(self):
        """初始化 SMPLX 模型（用于计算 T-pose pelvis offset）"""
        if self.smplx_model_path and os.path.exists(self.smplx_model_path):
            from smplx import SMPLX
            self.smplx_model = SMPLX(
                self.smplx_model_path,
                use_pca=False,
                flat_hand_mean=True,
                num_betas=10,
            )
            self.logger.info(f"Loaded SMPLX model from {self.smplx_model_path}")
        else:
            self.logger.info(
                "No SMPLX model path provided, will use skeleton_offset from HPE output"
            )

        self._is_setup = True
        self.logger.info("WorldTransform component initialized")

    def cleanup(self):
        """释放资源"""
        self.smplx_model = None
        super().cleanup()

    @staticmethod
    def transform_smpl_params(root_orient, transl, R_wc, t_wc, smpl_t_pose_pelvis):
        """
        将相机坐标系的 SMPL 参数转换到世界坐标系

        参考 PromptHMR/pipeline/world.py 的 transform_smpl_params

        Args:
            root_orient: [F, 3, 3] 相机坐标系的 root rotation matrix
            transl: [F, 3] 相机坐标系的 translation
            R_wc: [F, 3, 3] camera-to-world rotation
            t_wc: [F, 3] camera-to-world translation
            smpl_t_pose_pelvis: [3] T-pose 时 pelvis joint 位置

        Returns:
            root_orient_w: [F, 3, 3] 世界坐标系 root rotation
            transl_w: [F, 3] 世界坐标系 translation
        """
        assert smpl_t_pose_pelvis.shape == (3,)
        offset = smpl_t_pose_pelvis[None, :, None]  # (1, 3, 1)
        transl = transl.unsqueeze(-1).float()       # (F, 3, 1)
        t_wc = t_wc.unsqueeze(-1)                   # (F, 3, 1)

        transl_w = R_wc @ (offset + transl) + t_wc - offset
        root_orient_w = R_wc @ root_orient

        transl_w = transl_w.squeeze(-1)  # (F, 3)
        return root_orient_w, transl_w

    def execute(self, data: PipelineData) -> PipelineData:
        """执行世界坐标系变换"""
        sp = data.smpl_params

        # --- 获取相机坐标系 SMPL 参数 ---
        def _to_tensor(x):
            if x is None:
                return None
            return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()

        global_orient_c = _to_tensor(sp.global_orient_c)  # (F, 3) axis-angle
        transl_c = _to_tensor(sp.trans)  # (F, 3) camera-space translation
        betas = _to_tensor(sp.betas)  # (F, 10)
        body_pose_aa = _to_tensor(sp.body_pose_aa)  # (F, 63)

        F_frames = global_orient_c.shape[0]

        # --- 获取 c2w 相机参数 (R_wc, T_wc) ---
        cam = data.camera_params
        if cam.world_R is not None and cam.world_T is not None:
            R_wc = _to_tensor(cam.world_R)  # (N, 3, 3)
            T_wc = _to_tensor(cam.world_T)  # (N, 3)
            coord_source = "world_R/world_T (gravity-aligned)"
        else:
            R_wc = _to_tensor(cam.R)
            T_wc = _to_tensor(cam.T)
            coord_source = "R/T (raw SLAM)"

        self.logger.info(f"Using camera params from {coord_source}")
        self.logger.info(f"SMPL frames: {F_frames}, Camera frames: {R_wc.shape[0]}")

        # 确保帧数一致
        assert R_wc.shape[0] == F_frames, (
            f"Camera frames ({R_wc.shape[0]}) != SMPL frames ({F_frames})"
        )

        # --- 可选：平滑相机参数 ---
        if self.use_smooth_cam:
            try:
                import sys
                # PromptHMR 的 one_euro_filter 可能在 thirdparty/PromptHMR 或外部 PromptHMR
                prompthmr_paths = [
                    os.path.abspath('thirdparty/PromptHMR'),
                    os.path.abspath('../PromptHMR'),
                ]
                for p in prompthmr_paths:
                    if p not in sys.path and os.path.isdir(p):
                        sys.path.insert(0, p)
                from prompt_hmr.utils.one_euro_filter import smooth_one_euro
                min_cutoff = 0.001
                beta = 0.1
                T_wc = torch.from_numpy(
                    smooth_one_euro(T_wc.numpy(), min_cutoff, beta)
                ).float()
                R_wc = torch.from_numpy(
                    smooth_one_euro(R_wc.numpy(), min_cutoff, beta, is_rot=True)
                ).float()
                self.logger.info("Applied One Euro filter to camera params")
            except (ImportError, Exception) as e:
                self.logger.warning(
                    f"Cannot import smooth_one_euro ({e}), skipping camera smoothing"
                )

        # --- 计算 T-pose pelvis offset ---
        smpl_t_pose_pelvis = self._get_pelvis_offset(betas, sp.skeleton_offset)
        self.logger.info(f"T-pose pelvis offset: {smpl_t_pose_pelvis}")

        # --- 将 global_orient_c 从 axis-angle 转为 rotation matrix ---
        from lib.utils.rotation_conversions import (
            axis_angle_to_matrix,
            matrix_to_axis_angle,
        )

        root_orient_c_mat = axis_angle_to_matrix(global_orient_c)  # (F, 3, 3)

        # --- 执行坐标变换 ---
        root_orient_w_mat, transl_w = self.transform_smpl_params(
            root_orient_c_mat, transl_c, R_wc, T_wc, smpl_t_pose_pelvis
        )

        # 转回 axis-angle
        global_orient_w = matrix_to_axis_angle(root_orient_w_mat)  # (F, 3)

        # --- 坐标系后处理 ---
        # TRAM 的 align_to_world (est_gravity.py) 已经将 world_R/world_T 转到 y-up 坐标系
        # （通过 R_wg = [[1,0,0],[0,-1,0],[0,0,-1]]）。
        # 如果使用 world_R/world_T，不需要额外翻转。
        # 如果使用原始 R/T（未经 align_to_world），需要翻转。
        using_aligned = (cam.world_R is not None and cam.world_T is not None)

        if not using_aligned:
            # 原始 DROID-SLAM 输出是 z-forward, y-down，需要翻转到 y-up
            R_flip = torch.tensor(
                [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float
            )
            root_orient_w_mat = R_flip[None] @ root_orient_w_mat
            transl_w = torch.einsum('ij,bj->bi', R_flip, transl_w)
            global_orient_w = matrix_to_axis_angle(root_orient_w_mat)
            R_wc = torch.einsum('ij,bjk->bik', R_flip, R_wc)
            T_wc = torch.einsum('ij,bj->bi', R_flip, T_wc)
            self.logger.info("Applied R_flip (raw SLAM -> y-up)")

        # 地面对齐：skating_removal 中的 pp_static_joint 会自动做 ground_y 校正
        # 这里不做额外的地面对齐，保持与 SLAM 尺度一致

        self.logger.info(
            f"World transform complete. "
            f"transl_w range: x=[{transl_w[:, 0].min():.3f}, {transl_w[:, 0].max():.3f}], "
            f"y=[{transl_w[:, 1].min():.3f}, {transl_w[:, 1].max():.3f}], "
            f"z=[{transl_w[:, 2].min():.3f}, {transl_w[:, 2].max():.3f}]"
        )

        # --- 更新 data ---
        data.smpl_params.global_orient_w = global_orient_w
        data.smpl_params.global_trans = transl_w.clone()
        data.smpl_params.transl_w_raw = transl_w.clone()  # 供 skating_removal 使用

        # 保存世界坐标系相机参数到 metadata（供后续渲染/评估使用）
        data.metadata['world_transform'] = {
            'coord_source': coord_source,
            'use_smooth_cam': self.use_smooth_cam,
            'R_wc': R_wc,
            'T_wc': T_wc,
            'pelvis_offset': smpl_t_pose_pelvis,
            'using_aligned_cam': using_aligned,
        }

        self.logger.info("WorldTransform execution complete")
        return data

    def _get_pelvis_offset(self, betas, skeleton_offset):
        """计算 T-pose pelvis offset"""
        if skeleton_offset is not None:
            # 使用 HPE 输出的 skeleton_offset
            offset = skeleton_offset
            if isinstance(offset, np.ndarray):
                offset = torch.from_numpy(offset).float()
            elif isinstance(offset, torch.Tensor):
                offset = offset.float()
            return offset

        if self.smplx_model is not None:
            # 使用 SMPLX 模型计算
            mean_shape = betas.mean(dim=0, keepdim=True)
            pelvis = self.smplx_model(
                global_orient=torch.zeros(1, 3),
                body_pose=torch.zeros(1, 21 * 3),
                betas=mean_shape,
            ).joints[0, 0]
            return pelvis.detach()

        # fallback: 零偏移
        self.logger.warning(
            "No skeleton_offset or SMPLX model available, using zero pelvis offset"
        )
        return torch.zeros(3)
