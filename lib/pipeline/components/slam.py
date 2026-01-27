"""SLAMComponent - 相机运动估计组件"""

from typing import Dict, Any, Optional
import numpy as np
import torch
from glob import glob

from lib.pipeline.core.component import BackendComponent
from lib.pipeline.core.data import PipelineData, CameraParams
from lib.pipeline.backends.slam import DroidSLAMBackend


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
        
        # 估计相机运动
        cam_R, cam_T = self.backend.estimate_camera(
            image_folder,
            masks=masks,
            intrinsics=intrinsics
        )
        
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


