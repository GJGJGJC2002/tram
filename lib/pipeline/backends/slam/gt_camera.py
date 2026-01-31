"""GTCameraBackend - 从真值加载相机参数"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch

from lib.pipeline.core.component import Backend


class GTCameraBackend(Backend):
    """
    GT Camera 后端 - 从真值加载相机参数

    直接从 annotations 中读取 GT 相机内外参。

    Config:
        无需额外配置
    """

    DEFAULT_CONFIG = {}

    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)

    def setup(self):
        """初始化：无需加载模型"""
        self._is_setup = True
        self.logger.info("GT Camera backend initialized")

    def estimate_camera(
        self,
        image_folder: str,
        masks: Optional[torch.Tensor] = None,
        intrinsics: Optional[list] = None,
        is_static: bool = False,
        annotations: Dict[str, Any] = None,
        sampled_indices: np.ndarray = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从真值估计相机运动

        Args:
            image_folder: 图像文件夹路径（未使用，保持接口兼容）
            masks: 未使用（保持接口兼容）
            intrinsics: 未使用（保持接口兼容）
            is_static: 未使用（保持接口兼容）
            annotations: 包含相机数据的 annotations 字典
            sampled_indices: 可选的帧采样索引

        Returns:
            cam_R: 相机旋转矩阵 [N, 3, 3]
            cam_T: 相机平移向量 [N, 3]
        """
        if annotations is None:
            raise ValueError("annotations is required for GT Camera backend")

        ext = annotations['camera']['extrinsics']  # [N, 3, 4] 或 [N, 4, 4]

        # 应用帧采样（如果提供）
        if sampled_indices is not None:
            self.logger.info(
                f"Applying frame sampling to GT camera: "
                f"{len(ext)} -> {len(sampled_indices)} frames"
            )
            ext = ext[sampled_indices]

        # 提取旋转和平移
        # EMDB 的 extrinsics 格式是 [R_wc | t_wc]，即世界到相机变换
        cam_R = ext[:, :3, :3]  # [N, 3, 3]
        cam_T = ext[:, :3, 3]   # [N, 3]

        self.logger.info(
            f"Loaded GT camera parameters: {len(cam_R)} frames"
        )

        return torch.from_numpy(cam_R).float(), torch.from_numpy(cam_T).float()

    def align_to_world_frame(
        self,
        first_image_path: str,
        cam_R: torch.Tensor,
        cam_T: torch.Tensor,
        annotations: Dict[str, Any] = None,
        sampled_indices: np.ndarray = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        将相机轨迹对齐到世界坐标系

        对于 GT 数据，extrinsics 已经是世界到相机变换（w2c）。
        我们需要计算相机到世界变换（c2w）作为 world_R 和 world_T。

        Args:
            first_image_path: 第一帧图像路径（未使用，保持接口兼容）
            cam_R: 相机旋转矩阵 [N, 3, 3]（w2c）
            cam_T: 相机平移向量 [N, 3]（w2c）
            annotations: 包含相机数据的 annotations 字典
            sampled_indices: 可选的帧采样索引

        Returns:
            world_R: 世界坐标系下的旋转（c2w）[N, 3, 3]
            world_T: 世界坐标系下的平移（c2w）[N, 3]
            spec_focal: 估计的焦距（从 GT 内参获取）
        """
        if annotations is None:
            raise ValueError("annotations is required for GT Camera backend")

        # 从 GT 内参获取焦距
        intrinsics = annotations['camera']['intrinsics']  # [3, 3]
        spec_focal = float(intrinsics[0, 0])

        # cam_R 和 cam_T 是 w2c 变换：X_cam = R_wc @ X_world + t_wc
        # 我们需要 c2w 变换：X_world = R_cw @ X_cam + t_cw
        # 其中：R_cw = R_wc^T, t_cw = -R_cw @ t_wc

        R_wc = cam_R.numpy() if isinstance(cam_R, torch.Tensor) else cam_R
        t_wc = cam_T.numpy() if isinstance(cam_T, torch.Tensor) else cam_T

        # c2w 的旋转是 w2c 的转置
        world_R = np.transpose(R_wc, (0, 2, 1))  # R_cw = R_wc^T

        # c2w 的平移：t_cw = -R_cw @ t_wc
        world_T = -np.einsum('nij,nj->ni', world_R, t_wc)

        self.logger.info(
            f"Aligned GT camera to world frame: "
            f"focal={spec_focal:.2f}"
        )

        return (
            torch.from_numpy(world_R).float(),
            torch.from_numpy(world_T).float(),
            spec_focal
        )

    def cleanup(self):
        """清理资源"""
        super().cleanup()
