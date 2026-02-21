"""SLAMComponent - 相机运动估计组件"""

import logging
import os
from typing import Dict, Any, Optional
import numpy as np
import torch
from glob import glob

from lib.pipeline.core.component import BackendComponent
from lib.pipeline.core.data import PipelineData, CameraParams
from lib.pipeline.backends.slam import DroidSLAMBackend, GTCameraBackend, DroidTiaozhenBackend, DroidWarmstartBackend

logger = logging.getLogger(__name__)


class SLAMComponent(BackendComponent):
    """
    相机运动估计组件
    
    使用 SLAM 方法估计相机运动轨迹。
    
    Config:
        backend: SLAM 后端类型 ('droid', 'dpvo', ...)
        use_masks: 是否使用人体 mask 遮挡动态物体
        align_to_world: 是否将轨迹对齐到世界坐标系
    """
    
    COMPONENT_TYPE = "slam"
    
    BACKENDS = {
        'droid': DroidSLAMBackend,
        'gt': GTCameraBackend,
        'droid_tiaozhen': DroidTiaozhenBackend,
        'droid_warmstart': DroidWarmstartBackend,
    }
    
    DEFAULT_BACKEND = 'droid'
    
    DEFAULT_CONFIG = {
        'use_masks': True,
        'align_to_world': True,
    }
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)
        
        self.use_masks = self.config['use_masks']
        self.align_to_world = self.config['align_to_world']
    
    def validate_input(self, data: PipelineData) -> bool:
        """验证输入：需要图像路径或图像，以及可选的 masks"""
        has_images = data.images is not None or len(data.image_paths) > 0
        if self.use_masks:
            has_masks = data.masks is not None
            return has_images and has_masks
        return has_images
    
    def execute(self, data: PipelineData) -> PipelineData:
        """
        执行相机运动估计

        结果存储在 data.camera_params 中。
        """
        # 根据后端类型调用不同的方法
        if self.backend_type == 'gt':
            return self._execute_gt(data)
        elif self.backend_type == 'droid_tiaozhen':
            return self._execute_tiaozhen(data)
        elif self.backend_type == 'droid_warmstart':
            return self._execute_warmstart(data)
        else:
            return self._execute_droid(data)

    def _execute_gt(self, data: PipelineData) -> PipelineData:
        """使用 GT 后端执行"""
        if data.annotations is None:
            raise ValueError("GT backend requires annotations")

        # 检查是否有帧采样信息
        sampling_info = data.metadata.get('frame_sampling')
        if sampling_info:
            sampled_indices = np.array(sampling_info['sampled_indices'])
        else:
            sampled_indices = None

        self.logger.info("Loading GT camera parameters...")

        # 调用 GT 后端估计相机（返回 w2c 变换）
        cam_R_w2c, cam_T_w2c = self.backend.estimate_camera(
            image_folder=None,
            masks=None,
            intrinsics=None,
            annotations=data.annotations,
            sampled_indices=sampled_indices
        )

        # 获取 GT 内参
        intrinsics = data.annotations['camera']['intrinsics']

        # 对齐到世界坐标系（将 w2c 转换为 c2w）
        if self.align_to_world:
            self.logger.info("Aligning GT camera trajectory to world frame...")
            cam_R, cam_T, spec_f = self.backend.align_to_world_frame(
                first_image_path=None,
                cam_R=cam_R_w2c,
                cam_T=cam_T_w2c,
                annotations=data.annotations,
                sampled_indices=sampled_indices
            )
            data.metadata['spec_focal'] = spec_f
        else:
            # 如果不对齐，仍然使用 w2c（但这会导致可视化不正确）
            logger.warning("GT camera not aligned to world frame - visualization will be incorrect")
            cam_R, cam_T = cam_R_w2c, cam_T_w2c
            spec_f = float(intrinsics[0, 0])

        # 创建相机参数（使用 c2w 变换，与 Droid-SLAM 保持一致）
        camera_params = CameraParams(
            R=cam_R,
            T=cam_T,
            intrinsics=intrinsics,
            focal_length=float(intrinsics[0, 0]),
            principal_point=intrinsics[:2, 2],
        )

        data.camera_params = camera_params

        # 统计
        num_frames = len(cam_T)
        trajectory_length = self._compute_trajectory_length(cam_T)
        data.metadata['slam_stats'] = {
            'num_frames': num_frames,
            'trajectory_length': trajectory_length,
            'backend': 'gt',
        }

        self.logger.info(
            f"GT camera loaded: {num_frames} frames, "
            f"trajectory length: {trajectory_length:.2f}m"
        )

        return data

    def _execute_droid(self, data: PipelineData) -> PipelineData:
        """使用 DroidSLAM 后端执行"""
        # 确定图像文件夹
        if data.image_paths:
            # 从图像路径推断文件夹
            image_folder = '/'.join(data.image_paths[0].split('/')[:-1])
        else:
            raise ValueError("SLAM requires image file paths")

        # 获取相机内参
        intrinsics = self._get_intrinsics(data)

        # 准备 masks
        masks = data.masks if self.use_masks else None

        self.logger.info(f"Running SLAM on {image_folder}...")
        self.logger.info(f"Use masks: {self.use_masks}, Align to world: {self.align_to_world}")

        # 检查是否需要返回关键帧信息（供下游 keyframe 模式使用）
        return_keyframe_info = self.config.get('return_keyframe_info', False)

        # 估计相机运动
        result = self.backend.estimate_camera(
            image_folder,
            masks=masks,
            intrinsics=intrinsics,
            return_keyframe_info=return_keyframe_info
        )

        if return_keyframe_info:
            cam_R, cam_T, keyframe_info = result
            if keyframe_info is not None:
                data.metadata['slam_keyframe_info'] = keyframe_info
                self.logger.info(
                    f"Stored keyframe info: {len(keyframe_info['keyframe_tstamps'])} keyframes"
                )
        else:
            cam_R, cam_T = result

        # 创建相机参数
        camera_params = CameraParams(
            R=cam_R,
            T=cam_T,
            intrinsics=data.annotations.get('camera', {}).get('intrinsics') if data.annotations else None,
            focal_length=intrinsics[0] if intrinsics else None,
            principal_point=np.array(intrinsics[2:4]) if intrinsics else None,
        )

        # 对齐到世界坐标系
        if self.align_to_world:
            self.logger.info("Aligning camera trajectory to world frame...")
            world_R, world_T, spec_f = self.backend.align_to_world_frame(
                data.image_paths[0],
                cam_R,
                cam_T
            )
            camera_params.world_R = world_R
            camera_params.world_T = world_T
            data.metadata['spec_focal'] = spec_f

        data.camera_params = camera_params

        # 统计
        num_frames = len(cam_T)
        trajectory_length = self._compute_trajectory_length(cam_T)
        data.metadata['slam_stats'] = {
            'num_frames': num_frames,
            'trajectory_length': trajectory_length,
            'backend': 'droid',
        }

        self.logger.info(f"SLAM completed: {num_frames} frames, trajectory length: {trajectory_length:.2f}m")

        return data
    
    def _get_intrinsics(self, data: PipelineData) -> Optional[list]:
        """从数据中获取相机内参"""
        if data.annotations and 'camera' in data.annotations:
            intr = data.annotations['camera'].get('intrinsics')
            if intr is not None:
                # 从 3x3 内参矩阵提取 [fx, fy, cx, cy]
                return [intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]]
        return None
    
    def _compute_trajectory_length(self, cam_T: torch.Tensor) -> float:
        """计算轨迹长度"""
        if isinstance(cam_T, torch.Tensor):
            cam_T = cam_T.numpy()

        if len(cam_T) < 2:
            return 0.0

        diffs = np.diff(cam_T, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.sum(distances))

    def _execute_warmstart(self, data: PipelineData) -> PipelineData:
        """使用 DroidWarmstart 后端执行（强制关键帧 + warm start，无插值）"""
        # 获取相邻帧渲染信息
        adjacent_render_info = data.metadata.get('adjacent_render_info')
        if adjacent_render_info is None:
            raise ValueError(
                "droid_warmstart backend requires adjacent_render_info in metadata. "
                "Ensure adjacent_smpl_renderer component runs before this component."
            )

        # 获取关键帧信息（来自第一次 SLAM）
        keyframe_info = data.metadata.get('slam_keyframe_info')
        if keyframe_info is None:
            raise ValueError(
                "droid_warmstart backend requires slam_keyframe_info in metadata. "
                "Ensure the first SLAM component has return_keyframe_info: true."
            )

        # 保存第一次 SLAM 的结果用于尺度对齐
        first_slam_cam_R = None
        first_slam_cam_T = None
        if data.camera_params is not None and data.camera_params.T is not None:
            first_slam_cam_R = data.camera_params.R
            first_slam_cam_T = data.camera_params.T
            self.logger.info("Saved first SLAM results for scale alignment")

        # 获取渲染图像目录
        image_dir = self.config.get('image_dir')
        if image_dir is None:
            image_dir = adjacent_render_info.get('output_dir')
        if image_dir is None:
            raise ValueError("image_dir must be specified in config or adjacent_render_info")

        # 获取相机内参
        intrinsics = self._get_intrinsics(data)

        # 准备 forced keyframes 和 warm start 数据
        forced_keyframes = keyframe_info['keyframe_tstamps']
        initial_poses_se3 = None
        initial_disps = None

        # 是否注入 warmstart poses（默认关闭，用于调试对比）
        use_warmstart_poses = self.config.get('use_warmstart_poses', False)
        if use_warmstart_poses:
            if 'keyframe_poses_se3' in keyframe_info:
                tstamps = keyframe_info['keyframe_tstamps']
                poses = keyframe_info['keyframe_poses_se3']
                initial_poses_se3 = {int(t): poses[i] for i, t in enumerate(tstamps)}

            if 'keyframe_disps' in keyframe_info:
                tstamps = keyframe_info['keyframe_tstamps']
                disps = keyframe_info['keyframe_disps']
                initial_disps = {int(t): disps[i] for i, t in enumerate(tstamps)}
            self.logger.info("  Warm start poses: ENABLED")
        else:
            self.logger.info("  Warm start poses: DISABLED (using Identity/None like original SLAM)")

        # 准备 masks（用于尺度估计时排除人体区域）
        masks = data.masks if self.use_masks else None

        # 获取原始图像目录（用于 ZoeDepth 深度估计，避免在渲染后图像上推理）
        original_image_dir = None
        if hasattr(data, 'image_paths') and data.image_paths:
            original_image_dir = os.path.dirname(data.image_paths[0])
            self.logger.info(f"  Original image dir for ZoeDepth: {original_image_dir}")

        self.logger.info(f"Running Warmstart DROID-SLAM...")
        self.logger.info(f"  Image directory: {image_dir}")
        self.logger.info(f"  Forced keyframes: {len(forced_keyframes)}")
        self.logger.info(f"  Total frames: {adjacent_render_info['total_frames']}")
        self.logger.info(f"  Use masks: {self.use_masks}")

        # 是否在关键帧上使用渲染图像（带 SMPL mesh）
        use_rendered_for_keyframes = self.config.get('use_rendered_for_keyframes', False)
        self.logger.info(f"  Use rendered for keyframes: {use_rendered_for_keyframes}")

        # 运行 warmstart SLAM
        cam_R, cam_T = self.backend.estimate_camera(
            image_dir=image_dir,
            forced_keyframes=forced_keyframes,
            initial_poses_se3=initial_poses_se3,
            initial_disps=initial_disps,
            masks=masks,
            intrinsics=intrinsics,
            original_image_dir=original_image_dir,
            use_rendered_for_keyframes=use_rendered_for_keyframes,
        )

        # 尺度对齐
        enable_scale_alignment = self.config.get('enable_scale_alignment', True)
        if first_slam_cam_T is not None and enable_scale_alignment:
            self.logger.info("Aligning warmstart trajectory scale to first SLAM...")
            cam_R, cam_T = self.backend.align_scale_to_reference(
                cam_R, cam_T, first_slam_cam_R, first_slam_cam_T
            )
            self.logger.info("Scale alignment completed")
            data.metadata['scale_aligned_to_first_slam'] = True
        else:
            if not enable_scale_alignment:
                self.logger.info("Scale alignment disabled by configuration")
            data.metadata['scale_aligned_to_first_slam'] = False

        # 获取内参
        if data.annotations and 'camera' in data.annotations:
            gt_intrinsics = data.annotations['camera']['intrinsics']
            focal_length = float(gt_intrinsics[0, 0])
            principal_point = gt_intrinsics[:2, 2]
        elif intrinsics:
            focal_length = intrinsics[0]
            principal_point = np.array(intrinsics[2:4])
        else:
            self.logger.warning("No camera intrinsics available, using defaults")
            focal_length = 1000.0
            principal_point = np.array([540, 960])

        # 创建相机参数
        camera_params = CameraParams(
            R=cam_R,
            T=cam_T,
            intrinsics=data.annotations.get('camera', {}).get('intrinsics') if data.annotations else None,
            focal_length=focal_length,
            principal_point=principal_point,
        )

        if self.align_to_world:
            data.metadata['spec_focal_warmstart'] = focal_length

        data.camera_params = camera_params

        # 统计
        num_frames = len(cam_T)
        trajectory_length = self._compute_trajectory_length(cam_T)
        data.metadata['slam_stats_warmstart'] = {
            'num_frames': num_frames,
            'trajectory_length': trajectory_length,
            'backend': 'droid_warmstart',
            'num_forced_keyframes': len(forced_keyframes),
            'original_total_frames': adjacent_render_info['total_frames'],
        }

        # 记录 debug 信息路径
        debug_path = os.path.join(image_dir, '..', 'droid_debug', 'droid_frontend_debug.json')
        if os.path.exists(debug_path):
            data.metadata['droid_debug_path'] = os.path.abspath(debug_path)

        self.logger.info(
            f"Warmstart SLAM completed: {num_frames} frames "
            f"(forced {len(forced_keyframes)} keyframes), "
            f"trajectory length: {trajectory_length:.2f}m"
        )

        return data

    def _execute_tiaozhen(self, data: PipelineData) -> PipelineData:
        """使用 DroidTiaozhen 后端执行（带插值）"""
        # 获取相邻帧渲染信息
        adjacent_render_info = data.metadata.get('adjacent_render_info')
        if adjacent_render_info is None:
            raise ValueError(
                "droid_tiaozhen backend requires adjacent_render_info in metadata. "
                "Ensure adjacent_smpl_renderer component runs before this component."
            )

        # === 保存第一次 SLAM 的结果，用于后续尺度对齐 ===
        first_slam_cam_R = None
        first_slam_cam_T = None
        if data.camera_params is not None and data.camera_params.T is not None:
            first_slam_cam_R = data.camera_params.R
            first_slam_cam_T = data.camera_params.T
            self.logger.info("Saved first SLAM results for scale alignment")
        else:
            self.logger.warning("No first SLAM results available for scale alignment")

        # 获取处理后的图像目录
        image_dir = self.config.get('image_dir')
        if image_dir is None:
            # 优先从 adjacent_render_info 中获取
            image_dir = adjacent_render_info.get('output_dir')
        if image_dir is None:
            # 最后回退：根据 output_dir 和 sequence_name 拼接
            sequence_name = data.metadata.get('sequence_name', 'sequence')
            output_dir_base = self.config.get('output_dir', 'results/adjacent_smpl')
            image_dir = os.path.join(output_dir_base, sequence_name)
            self.logger.info(f"Using fallback image directory: {image_dir}")

        # 获取相机内参
        intrinsics = self._get_intrinsics(data)

        self.logger.info(f"Running DROID-SLAM Tiaozhen on processed images...")
        self.logger.info(f"  Image directory: {image_dir}")
        self.logger.info(f"  Original frames: {adjacent_render_info['total_frames']}")
        self.logger.info(f"  Processed frames: {len(adjacent_render_info['rendered_indices'])}")
        self.logger.info(f"  Pre_dis: {adjacent_render_info['pre_dis']}")

        # 准备 masks：按 rendered_indices 下采样，使其与渲染后的图像帧一一对应
        tiaozhen_masks = None
        if self.use_masks and data.masks is not None:
            rendered_indices = adjacent_render_info['rendered_indices']
            last_frame_idx = adjacent_render_info.get('last_frame_idx')
            # 收集渲染帧对应的 mask 索引
            mask_indices = list(rendered_indices)
            if last_frame_idx is not None and last_frame_idx not in rendered_indices:
                mask_indices.append(last_frame_idx)
            # 按索引提取对应帧的 masks
            mask_indices = [i for i in mask_indices if i < len(data.masks)]
            tiaozhen_masks = data.masks[mask_indices]
            self.logger.info(f"  Using masks for tiaozhen SLAM: {len(tiaozhen_masks)} masks (from {len(data.masks)} original)")
        else:
            self.logger.info(f"  Not using masks for tiaozhen SLAM (use_masks={self.use_masks}, masks available={data.masks is not None})")

        # 估计相机运动（带插值）
        cam_R, cam_T = self.backend.estimate_camera(
            image_folder=None,  # 不使用，从 image_dir config 读取
            masks=tiaozhen_masks,
            intrinsics=intrinsics,
            adjacent_render_info=adjacent_render_info,
        )

        # === 尺度对齐：将 tiaozhen 的尺度对齐到第一次 SLAM ===
        # BUG FIX: 添加配置选项来控制是否进行尺度对齐
        # 如果进行了尺度对齐，ate 和 ate_s 会非常接近，因为尺度已经被校正
        enable_scale_alignment = self.config.get('enable_scale_alignment', True)
        
        if first_slam_cam_T is not None and enable_scale_alignment:
            self.logger.info("Aligning tiaozhen trajectory scale to first SLAM...")
            cam_R, cam_T = self.backend.align_scale_to_reference(
                cam_R, cam_T, first_slam_cam_R, first_slam_cam_T
            )
            self.logger.info("Scale alignment completed")
            data.metadata['scale_aligned_to_first_slam'] = True
        else:
            if not enable_scale_alignment:
                self.logger.info("Scale alignment disabled by configuration")
            data.metadata['scale_aligned_to_first_slam'] = False

        # 获取 GT 内参（如果有）
        if data.annotations and 'camera' in data.annotations:
            gt_intrinsics = data.annotations['camera']['intrinsics']
            focal_length = float(gt_intrinsics[0, 0])
            principal_point = gt_intrinsics[:2, 2]
        elif intrinsics:
            focal_length = intrinsics[0]
            principal_point = np.array(intrinsics[2:4])
        else:
            self.logger.warning("No camera intrinsics available, using defaults")
            focal_length = 1000.0
            principal_point = np.array([540, 960])

        # 创建相机参数
        camera_params = CameraParams(
            R=cam_R,
            T=cam_T,
            intrinsics=data.annotations.get('camera', {}).get('intrinsics') if data.annotations else None,
            focal_length=focal_length,
            principal_point=principal_point,
        )

        # 对齐到世界坐标系（已经在 backend 中完成）
        # 这里只需要保存额外的 metadata
        if self.align_to_world:
            data.metadata['spec_focal_tiaozhen'] = focal_length

        # 注意：覆盖之前的 camera_params（如果需要保留之前的，可以使用不同的字段名）
        data.camera_params = camera_params

        # 统计
        num_frames = len(cam_T)
        trajectory_length = self._compute_trajectory_length(cam_T)
        data.metadata['slam_stats_tiaozhen'] = {
            'num_frames': num_frames,
            'trajectory_length': trajectory_length,
            'backend': 'droid_tiaozhen',
            'pre_dis': adjacent_render_info['pre_dis'],
            'original_total_frames': adjacent_render_info['total_frames'],
        }

        # 记录 droid debug 信息路径（由 run_slam 自动保存）
        debug_path = os.path.join(image_dir, '..', 'droid_debug', 'droid_frontend_debug.json')
        if os.path.exists(debug_path):
            data.metadata['droid_debug_path'] = os.path.abspath(debug_path)

        self.logger.info(
            f"DROID-SLAM Tiaozhen completed: {num_frames} frames (interpolated from "
            f"{len(adjacent_render_info['rendered_indices'])} processed frames), "
            f"trajectory length: {trajectory_length:.2f}m"
        )

        return data


