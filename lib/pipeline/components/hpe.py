"""HPEComponent - 人体姿态估计组件"""

from typing import Dict, Any
import numpy as np
import torch

from lib.pipeline.core.component import BackendComponent
from lib.pipeline.core.data import PipelineData, SMPLParams
from lib.pipeline.backends.hpe import VIMOBackend


class HPEComponent(BackendComponent):
    """
    人体姿态估计组件
    
    从图像序列中估计 SMPL 人体模型参数。
    
    Config:
        backend: HPE 后端类型 ('vimo', 'hmr', ...)
        mode: 估计模式 ('accurate', 'efficient')
        use_mean_shape: 是否使用平均形状参数
    """
    
    COMPONENT_TYPE = "hpe"
    
    BACKENDS = {
        'vimo': VIMOBackend,
    }
    
    DEFAULT_BACKEND = 'vimo'
    
    DEFAULT_CONFIG = {
        'mode': 'accurate',
        'use_mean_shape': True,
        'frame_stride': 1,      # 跳帧步长，1=不跳帧，20=每20帧取一次
        'max_frames': None,     # 最大处理帧数，None=全部
    }
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)
        
        self.mode = self.config['mode']
        self.use_mean_shape = self.config['use_mean_shape']
    
    def validate_input(self, data: PipelineData) -> bool:
        """验证输入：需要图像路径和边界框"""
        has_images = len(data.image_paths) > 0
        has_boxes = data.bboxes is not None
        return has_images and has_boxes

    def _sample_frames(self, data: PipelineData):
        """
        对输入数据进行帧采样

        根据 frame_stride 和 max_frames 对数据采样。
        会修改 data.image_paths, data.bboxes, data.camera_params, data.valid_frames_mask

        Returns:
            sampled_indices: 采样的帧索引
        """
        frame_stride = self.config.get('frame_stride', 1)
        max_frames = self.config.get('max_frames')

        if frame_stride <= 1 and max_frames is None:
            # 不需要采样
            return np.arange(len(data.image_paths))

        num_frames = len(data.image_paths)

        # 生成采样索引
        if frame_stride > 1:
            indices = np.arange(0, num_frames, frame_stride)
        else:
            indices = np.arange(num_frames)

        # 应用 max_frames 限制
        if max_frames is not None and len(indices) > max_frames:
            indices = indices[:max_frames]

        self.logger.info(
            f"Frame sampling: {num_frames} -> {len(indices)} frames "
            f"(stride={frame_stride}, max_frames={max_frames})"
        )

        # 采样 image_paths
        data.image_paths = [data.image_paths[i] for i in indices]

        # 采样 bboxes
        if data.bboxes is not None:
            data.bboxes = data.bboxes[indices]

        # 采样 camera_params
        if data.camera_params is not None:
            from lib.pipeline.core.data import CameraParams
            data.camera_params = CameraParams(
                R=data.camera_params.R[indices] if data.camera_params.R is not None else None,
                T=data.camera_params.T[indices] if data.camera_params.T is not None else None,
                focal_length=data.camera_params.focal_length,
                principal_point=data.camera_params.principal_point,
            )

        # 采样 valid_frames_mask
        if data.valid_frames_mask is not None:
            data.valid_frames_mask = data.valid_frames_mask[indices]

        # 记录采样信息到 metadata
        data.metadata['frame_sampling'] = {
            'original_num_frames': num_frames,
            'sampled_num_frames': len(indices),
            'sampled_indices': indices.tolist(),
            'frame_stride': frame_stride,
            'max_frames': max_frames,
        }

        return indices
    
    def execute(self, data: PipelineData) -> PipelineData:
        """
        执行人体姿态估计

        结果存储在 data.smpl_params 中。
        """
        # 帧采样（如果配置了跳帧）
        self._sample_frames(data)

        # 获取相机参数
        img_focal, img_center = self._get_camera_params(data)

        self.logger.info(f"Estimating SMPL parameters ({self.mode} mode)...")
        self.logger.info(f"Focal length: {img_focal:.2f}, Center: {img_center}")

        # 估计 SMPL
        smpl_results = self.backend.estimate_smpl(
            image_paths=data.image_paths,
            bboxes=data.bboxes,
            img_focal=img_focal,
            img_center=img_center,
            mode=self.mode
        )

        # 后处理
        pred_shape = smpl_results['pred_shape']
        if self.use_mean_shape:
            mean_shape = pred_shape.mean(dim=0, keepdim=True)
            pred_shape = mean_shape.repeat(len(pred_shape), 1)

        # 创建 SMPL 参数对象
        data.smpl_params = SMPLParams(
            poses=smpl_results['pred_pose'],
            betas=pred_shape,
            trans=smpl_results['pred_trans'],
            rotmat=smpl_results['pred_rotmat'],
            pred_cam=smpl_results['pred_cam'],
        )

        # 记录元数据
        num_frames = len(smpl_results['pred_pose'])
        data.metadata['hpe_stats'] = {
            'num_frames': num_frames,
            'mode': self.mode,
            'use_mean_shape': self.use_mean_shape,
            'img_focal': img_focal,
        }

        self.logger.info(f"SMPL estimation completed for {num_frames} frames")

        return data
    
    def _get_camera_params(self, data: PipelineData):
        """获取相机参数"""
        # 优先从标注中获取
        if data.annotations and 'camera' in data.annotations:
            intr = data.annotations['camera'].get('intrinsics')
            if intr is not None:
                img_focal = (intr[0, 0] + intr[1, 1]) / 2.0
                img_center = intr[:2, 2]
                return img_focal, img_center
        
        # 从 camera_params 获取
        if data.camera_params:
            if data.camera_params.focal_length:
                img_focal = data.camera_params.focal_length
            else:
                img_focal = 1000.0  # 默认值
            
            if data.camera_params.principal_point is not None:
                img_center = data.camera_params.principal_point
            else:
                # 假设图像中心
                img_size = data.get_image_size()
                if img_size:
                    img_center = np.array([img_size[1] / 2, img_size[0] / 2])
                else:
                    img_center = np.array([540, 960])  # 默认 1080p
            
            return img_focal, img_center
        
        # 使用默认值
        self.logger.warning("Using default camera parameters")
        return 1000.0, np.array([540, 960])


