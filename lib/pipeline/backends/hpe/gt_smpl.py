"""GTSmplBackend - 从真值加载 SMPL 参数"""

from typing import Dict, Any
import numpy as np
import torch

from lib.pipeline.core.component import Backend


class GTSmplBackend(Backend):
    """
    GT SMPL 后端 - 从真值加载 SMPL 参数

    直接从 annotations 中读取 GT SMPL 参数。

    Config:
        convert_to_rotmat: 是否将 axis-angle 转换为 rotation matrix (默认 True)
    """

    DEFAULT_CONFIG = {
        'convert_to_rotmat': True,
    }

    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)

        self.convert_to_rotmat = self.config['convert_to_rotmat']

    def setup(self):
        """初始化：无需加载模型"""
        self._is_setup = True
        self.logger.info("GT SMPL backend initialized")

    def estimate_smpl(
        self,
        annotations: Dict[str, Any],
        sampled_indices: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        从真值估计 SMPL 参数

        Args:
            annotations: 包含 SMPL 数据的 annotations 字典
            sampled_indices: 可选的帧采样索引

        Returns:
            SMPL 参数字典，包含：
            - pred_pose: [N, 24, 3] axis-angle 格式（相机坐标系）
            - pred_shape: [N, 10] betas
            - pred_trans: [N, 3] 平移（相机坐标系）
            - pred_rotmat: [N, 24, 3, 3] rotation matrix（相机坐标系）
            - pred_cam: [N, 3] 相机参数 [scale, tx, ty]
        """
        from lib.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

        ann = annotations
        ext = ann['camera']['extrinsics']  # [N, 3, 4] 或 [N, 4, 4]
        intrinsics = ann['camera']['intrinsics']  # [3, 3]

        # 应用帧采样（如果提供）
        if sampled_indices is not None:
            self.logger.info(
                f"Applying frame sampling to GT SMPL: "
                f"{len(ann['smpl']['poses_body'])} -> {len(sampled_indices)} frames"
            )
        else:
            sampled_indices = np.arange(len(ann['smpl']['poses_body']))

        # 读取 SMPL 参数（世界坐标系）
        poses_body = ann["smpl"]["poses_body"][sampled_indices]  # [N, 23, 3]
        poses_root = ann["smpl"]["poses_root"][sampled_indices]  # [N, 3]
        betas = np.repeat(
            ann["smpl"]["betas"].reshape((1, -1)),
            repeats=len(sampled_indices),
            axis=0
        )  # [N, 10]
        trans_world = ann["smpl"]["trans"][sampled_indices]  # [N, 3]
        ext = ext[sampled_indices]

        # 将 root orientation 转换到相机坐标系
        poses_root_cam_matrix = (
            torch.from_numpy(ext[:, :3, :3]).float() @
            axis_angle_to_matrix(torch.from_numpy(poses_root).float())
        )  # [N, 3, 3]
        poses_root_cam = matrix_to_axis_angle(poses_root_cam_matrix).numpy()  # [N, 3]

        # 将 trans 从世界坐标系转换到相机坐标系
        trans_cam = np.einsum(
            'nij,nj->ni',
            ext[:, :3, :3],
            trans_world
        ) + ext[:, :3, 3]

        self.logger.info(
            f"Loaded GT SMPL: {len(trans_cam)} frames, "
            f"converted to camera coordinate system"
        )

        # 转换为 rotation matrix 格式（相机坐标系）
        if self.convert_to_rotmat:
            # poses_root_cam: [N, 3] -> [N, 1, 3, 3]
            root_rotmat = axis_angle_to_matrix(
                torch.from_numpy(poses_root_cam).float()
            ).numpy()  # [N, 3, 3]
            root_rotmat = root_rotmat[:, None, :, :]  # [N, 1, 3, 3]

            # poses_body: [N, 23, 3] -> [N, 23, 3, 3]
            body_rotmat = axis_angle_to_matrix(
                torch.from_numpy(poses_body).reshape(-1, 3).float()
            ).numpy()  # [N*23, 3, 3]
            body_rotmat = body_rotmat.reshape(-1, 23, 3, 3)  # [N, 23, 3, 3]

            # 合并: [N, 24, 3, 3]
            pred_rotmat = np.concatenate([root_rotmat, body_rotmat], axis=1)
            pred_pose = None
        else:
            # 保持 axis-angle 格式（相机坐标系）
            poses_root_expanded = poses_root_cam[:, None, :]  # [N, 1, 3]
            pred_pose = np.concatenate([poses_root_expanded, poses_body], axis=1)
            pred_rotmat = None

        # 计算 pred_cam（使用 GT 的内参）
        N = len(trans_cam)
        focal_length_gt = intrinsics[0, 0]
        img_focal_gt = intrinsics[0, 0]
        pred_cam_scale = focal_length_gt / img_focal_gt  # 通常为 1.0

        pred_cam = np.zeros((N, 3))
        pred_cam[:, 0] = pred_cam_scale

        self.logger.info(
            f"GT camera: focal={focal_length_gt:.2f}, "
            f"pred_cam scale={pred_cam_scale:.2f}"
        )

        # 返回与 VIMOBackend 相同格式的字典
        return {
            'pred_pose': pred_pose,  # axis-angle（相机坐标系）
            'pred_shape': torch.from_numpy(betas).float(),
            'pred_trans': torch.from_numpy(trans_cam).float(),
            'pred_rotmat': torch.from_numpy(pred_rotmat).float() if pred_rotmat is not None else None,
            'pred_cam': torch.from_numpy(pred_cam).float(),
        }

    def cleanup(self):
        """清理资源"""
        super().cleanup()
