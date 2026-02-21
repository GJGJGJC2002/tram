"""DroidSLAM Tiaozhen Backend - Interpolation support for downsampled images

This backend runs DROID-SLAM on downsampled images (from adjacent_smpl_renderer)
and interpolates the camera parameters back to the original frame count.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from glob import glob
import os
from scipy.spatial.transform import Rotation as RotLib

from lib.pipeline.core.component import Backend


class DroidTiaozhenBackend(Backend):
    """
    DROID-SLAM 调整后端 - 支持插值

    对相邻帧渲染后的下采样图像运行 DROID-SLAM，
    并将结果插值回原始帧数。

    Config:
        image_dir: 处理后的图像目录（从 adjacent_smpl_renderer 输出）
        interpolation_method: 插值方法 ('linear', 'cubic')
        align_to_world: 是否对齐到世界坐标系
    """

    DEFAULT_CONFIG = {
        'image_dir': None,  # 从 adjacent_smpl_renderer 的 output_dir 读取
        'interpolation_method': 'linear',
        'align_to_world': True,
    }

    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)

        self.interpolation_method = self.config['interpolation_method']
        self.align_to_world = self.config['align_to_world']

    def setup(self):
        """初始化"""
        self._is_setup = True
        self.logger.info("DroidSLAM Tiaozhen backend initialized")

    def estimate_camera(
        self,
        image_folder: str = None,
        masks: Optional[torch.Tensor] = None,
        intrinsics: Optional[list] = None,
        annotations: Dict[str, Any] = None,
        adjacent_render_info: Dict[str, Any] = None,
        is_static: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        估计相机运动（带插值）

        Args:
            image_folder: 原始图像文件夹（未使用，保留接口兼容性）
            masks: 人体 mask，按 rendered_indices 下采样后的，与渲染图像帧一一对应
            intrinsics: 相机内参 [fx, fy, cx, cy]
            annotations: 标注数据（未使用）
            adjacent_render_info: adjacent_smpl_renderer 的渲染信息
                - total_frames: 原始总帧数
                - pre_dis: 采样间隔
                - rendered_indices: 渲染的帧索引
                - last_frame_idx: 最后一帧索引（如果不整除）
            is_static: 是否为静态相机

        Returns:
            cam_R: 相机旋转矩阵 [total_frames, 3, 3]（插值后）
            cam_T: 相机平移向量 [total_frames, 3]（插值后）
        """
        self.ensure_setup()

        if adjacent_render_info is None:
            raise ValueError("adjacent_render_info is required for interpolation")

        # 获取处理后的图像目录
        image_dir = self.config.get('image_dir')
        if image_dir is None:
            # 从 adjacent_render_info 中获取输出目录
            image_dir = adjacent_render_info.get('output_dir')
        if image_dir is None:
            raise ValueError("image_dir must be specified in config or adjacent_render_info")

        self.logger.info(f"Running DROID-SLAM on processed images from {image_dir}")

        # 获取渲染信息
        total_frames = adjacent_render_info['total_frames']
        pre_dis = adjacent_render_info['pre_dis']
        rendered_indices = adjacent_render_info['rendered_indices']
        last_frame_idx = adjacent_render_info.get('last_frame_idx')

        # 实际处理的帧数（可能比 rendered_indices 多 1，如果包含最后一帧）
        num_processed = adjacent_render_info.get('num_processed_frames', len(rendered_indices))
        self.logger.info(
            f"Original frames: {total_frames}, "
            f"Processed frames: {num_processed}, "
            f"Pre_dis: {pre_dis}"
        )

        # 运行 DROID-SLAM（在处理后的图像上）
        from lib.camera import run_metric_slam

        # 检查图像目录
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")

        cam_R, cam_T = run_metric_slam(
            image_dir,
            masks=masks,  # 使用当前帧人体 mask，辅助 SLAM 前后端
            calib=intrinsics,
            is_static=is_static
        )

        self.logger.info(f"DROID-SLAM output: R={cam_R.shape}, T={cam_T.shape}")

        # 检查输出帧数是否匹配
        if len(cam_R) != num_processed:
            self.logger.warning(
                f"DROID-SLAM output {len(cam_R)} frames, "
                f"expected {num_processed} frames. "
                f"Using actual output frame count."
            )
            # 使用 DROID-SLAM 实际输出的帧数
            num_processed = len(cam_R)

        # 对齐到世界坐标系
        if self.align_to_world:
            self.logger.info("Aligning to world frame...")
            # 获取第一张处理后的图像用于重力估计
            processed_images = sorted(glob(os.path.join(image_dir, "*.jpg")))
            if len(processed_images) == 0:
                processed_images = sorted(glob(os.path.join(image_dir, "*.png")))
            if len(processed_images) == 0:
                raise ValueError(f"No images found in {image_dir}")

            from lib.camera import align_cam_to_world
            cam_R, cam_T, spec_f = align_cam_to_world(
                processed_images[0], cam_R, cam_T
            )
            self.logger.info(f"Aligned to world frame, spec_focal: {spec_f:.2f}")

        # 插值回原始帧数
        cam_R_full, cam_T_full = self._interpolate_camera_params(
            cam_R, cam_T, rendered_indices, total_frames, last_frame_idx
        )

        self.logger.info(f"Interpolated to {len(cam_R_full)} frames")

        return cam_R_full, cam_T_full

    def _interpolate_camera_params(
        self,
        cam_R: torch.Tensor,
        cam_T: torch.Tensor,
        rendered_indices: list,
        total_frames: int,
        last_frame_idx: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        插值相机参数到原始帧数

        对于旋转矩阵，使用 SLERP 插值
        对于平移向量，使用线性插值

        Args:
            cam_R: 采样帧的旋转矩阵 [num_processed, 3, 3]
            cam_T: 采样帧的平移向量 [num_processed, 3]
            rendered_indices: 采样帧的索引列表
            total_frames: 目标总帧数
            last_frame_idx: 最后一帧的索引（如果有单独保存）

        Returns:
            cam_R_full: 插值后的旋转矩阵 [total_frames, 3, 3]
            cam_T_full: 插值后的平移向量 [total_frames, 3]
        """
        num_processed = len(cam_R)
        num_rendered = len(rendered_indices)
        has_last_frame = (last_frame_idx is not None and num_processed > num_rendered)

        # 初始化输出
        device = cam_R.device if isinstance(cam_R, torch.Tensor) else torch.device('cpu')
        dtype = cam_R.dtype if isinstance(cam_R, torch.Tensor) else torch.float32

        cam_R_full = torch.zeros((total_frames, 3, 3), dtype=dtype, device=device)
        cam_T_full = torch.zeros((total_frames, 3), dtype=dtype, device=device)

        # 转换为 numpy（如果是 tensor）
        if isinstance(cam_R, torch.Tensor):
            cam_R_np = cam_R.cpu().numpy()
            cam_T_np = cam_T.cpu().numpy()
        else:
            cam_R_np = cam_R
            cam_T_np = cam_T

        # 对每个采样间隔进行插值
        # 如果有最后一帧，只插值前 num_rendered - 1 个间隔
        num_intervals = num_rendered - 1 if has_last_frame else len(rendered_indices) - 1

        for i in range(num_intervals):
            idx_start = rendered_indices[i]
            idx_end = rendered_indices[i + 1]

            # 起始和结束帧的相机参数
            R_start = cam_R_np[i]
            R_end = cam_R_np[i + 1]
            T_start = cam_T_np[i]
            T_end = cam_T_np[i + 1]

            # 插值帧数
            num_interp = idx_end - idx_start + 1

            # 插值
            for j in range(num_interp):
                idx = idx_start + j
                if idx >= total_frames:
                    break

                # 插值系数 [0, 1]
                t = j / (idx_end - idx_start) if idx_end > idx_start else 0

                # 旋转矩阵 SLERP 插值
                R_interp = self._slerp_rotation(R_start, R_end, t)

                # 平移向量线性插值
                T_interp = (1 - t) * T_start + t * T_end

                cam_R_full[idx] = torch.from_numpy(R_interp).to(device=device, dtype=dtype)
                cam_T_full[idx] = torch.from_numpy(T_interp).to(device=device, dtype=dtype)

        # 处理最后一帧（如果存在）
        last_rendered_idx = rendered_indices[-1]

        if has_last_frame:
            # 有额外的最后一帧，从最后一帧的 SLAM 结果中获取相机参数
            # 最后一帧的相机参数在 cam_R_np[-1], cam_T_np[-1]
            R_last = cam_R_np[-1]
            T_last = cam_T_np[-1]

            # 插值从最后一个渲染帧到最后一帧
            if last_frame_idx > last_rendered_idx:
                # 渲染帧数（不包括最后一帧）
                num_interp = last_frame_idx - last_rendered_idx

                for j in range(1, num_interp + 1):
                    idx = last_rendered_idx + j
                    if idx >= total_frames:
                        break

                    # 插值系数 [0, 1]
                    t = j / num_interp

                    # 使用最后一个渲染帧和最后一帧进行插值
                    R_start = cam_R_np[num_rendered - 1]
                    R_interp = self._slerp_rotation(R_start, R_last, t)
                    T_start = cam_T_np[num_rendered - 1]
                    T_interp = (1 - t) * T_start + t * T_last

                    cam_R_full[idx] = torch.from_numpy(R_interp).to(device=device, dtype=dtype)
                    cam_T_full[idx] = torch.from_numpy(T_interp).to(device=device, dtype=dtype)

        elif last_rendered_idx < total_frames - 1:
            # 没有最后一帧，重复最后一个相机的参数
            # 转换为 tensor 后再赋值
            if isinstance(cam_R_np[-1], np.ndarray):
                cam_R_full[last_rendered_idx + 1:] = torch.from_numpy(cam_R_np[-1]).to(device=device, dtype=dtype)
                cam_T_full[last_rendered_idx + 1:] = torch.from_numpy(cam_T_np[-1]).to(device=device, dtype=dtype)
            else:
                cam_R_full[last_rendered_idx + 1:] = cam_R_np[-1].to(device=device, dtype=dtype)
                cam_T_full[last_rendered_idx + 1:] = cam_T_np[-1].to(device=device, dtype=dtype)

        return cam_R_full, cam_T_full

    def _slerp_rotation(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        旋转矩阵的 SLERP 插值

        Args:
            R1: 起始旋转矩阵 [3, 3]
            R2: 结束旋转矩阵 [3, 3]
            t: 插值系数 [0, 1]

        Returns:
            R_interp: 插值后的旋转矩阵 [3, 3]
        """
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp

        # 转换为 Rotation 对象
        rot1 = R.from_matrix(R1)
        rot2 = R.from_matrix(R2)

        # 创建插值关键帧
        key_rots = R.from_matrix([R1, R2])
        key_times = [0, 1]

        # SLERP 插值
        slerp = Slerp(key_times, key_rots)
        rot_interp = slerp([t])[0]

        return rot_interp.as_matrix()

    def align_to_world_frame(
        self,
        first_image_path: str,
        cam_R: torch.Tensor,
        cam_T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        将相机轨迹对齐到世界坐标系

        （已经在 estimate_camera 中调用，这里保留接口兼容性）

        Args:
            first_image_path: 第一帧图像路径
            cam_R: 相机旋转矩阵 [N, 3, 3]
            cam_T: 相机平移向量 [N, 3]

        Returns:
            world_R: 世界坐标系下的旋转 [N, 3, 3]
            world_T: 世界坐标系下的平移 [N, 3]
            spec_focal: 估计的焦距
        """
        from lib.camera import align_cam_to_world

        world_R, world_T, spec_f = align_cam_to_world(
            first_image_path, cam_R, cam_T
        )

        return world_R, world_T, spec_f

    def align_scale_to_reference(
        self,
        cam_R: torch.Tensor,
        cam_T: torch.Tensor,
        ref_R: torch.Tensor,
        ref_T: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 tiaozhen 轨迹的尺度对齐到参考轨迹（第一次 SLAM 的结果）

        使用 Sim3 对齐求解最优尺度因子 s，然后只应用尺度校正到 cam_T 上，
        保留 tiaozhen 改善过的轨迹形状。

        Args:
            cam_R: tiaozhen 的旋转 [N, 3, 3]
            cam_T: tiaozhen 的平移 [N, 3]
            ref_R: 参考（第一次 SLAM）的旋转 [N, 3, 3]
            ref_T: 参考（第一次 SLAM）的平移 [N, 3]

        Returns:
            cam_R: 旋转不变 [N, 3, 3]
            cam_T_scaled: 尺度校正后的平移 [N, 3]
        """
        # 转换为 numpy
        if isinstance(cam_T, torch.Tensor):
            cam_T_np = cam_T.cpu().numpy()
        else:
            cam_T_np = np.array(cam_T)

        if isinstance(ref_T, torch.Tensor):
            ref_T_np = ref_T.cpu().numpy()
        else:
            ref_T_np = np.array(ref_T)

        # 使用 evo 的 Umeyama 对齐来估计 Sim3 变换（包含尺度）
        # 这里我们需要：给定 tiaozhen 轨迹和参考轨迹，求 s, R, t 使得
        # ref_T ≈ s * R @ cam_T + t
        # 然后只取 s 来缩放 cam_T
        try:
            from evo.core import trajectory, sync
            from evo.core.trajectory import PoseTrajectory3D

            n = min(len(cam_T_np), len(ref_T_np))
            cam_T_np = cam_T_np[:n]
            ref_T_np = ref_T_np[:n]

            # 使用 Umeyama 算法估计 Sim3
            # umeyama: src -> tgt, 求 s, R, t 使得 tgt = s*R*src + t
            s, R_align, t_align = self._umeyama_alignment(cam_T_np, ref_T_np)

            self.logger.info(f"Scale alignment: s={s:.4f}")

            # 只应用尺度因子到 cam_T
            if isinstance(cam_T, torch.Tensor):
                cam_T_scaled = cam_T * s
            else:
                cam_T_scaled = torch.from_numpy(cam_T_np * s).float()

            return cam_R, cam_T_scaled

        except Exception as e:
            self.logger.warning(f"Scale alignment failed: {e}, returning original trajectory")
            return cam_R, cam_T

    @staticmethod
    def _umeyama_alignment(src: np.ndarray, tgt: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Umeyama 算法：求解 Sim3 对齐 tgt = s * R @ src + t

        Args:
            src: 源点集 [N, 3]
            tgt: 目标点集 [N, 3]

        Returns:
            s: 尺度因子
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
        """
        assert src.shape == tgt.shape
        n, dim = src.shape

        # 去均值
        src_mean = src.mean(axis=0)
        tgt_mean = tgt.mean(axis=0)
        src_centered = src - src_mean
        tgt_centered = tgt - tgt_mean

        # 方差
        src_var = np.sum(src_centered ** 2) / n

        # 协方差矩阵
        H = (tgt_centered.T @ src_centered) / n

        # SVD
        U, D, Vt = np.linalg.svd(H)

        # 处理反射
        S = np.eye(dim)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[dim - 1, dim - 1] = -1

        # 旋转
        R = U @ S @ Vt

        # 尺度
        s = np.trace(np.diag(D) @ S) / src_var

        # 平移
        t = tgt_mean - s * R @ src_mean

        return s, R, t

    def cleanup(self):
        """清理资源"""
        super().cleanup()
