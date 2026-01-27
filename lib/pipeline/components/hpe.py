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
    
    def execute(self, data: PipelineData) -> PipelineData:
        """
        执行人体姿态估计
        
        结果存储在 data.smpl_params 中。
        """
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


