"""
Pipeline 模块 - 可组合的模块化 Pipeline 架构

提供灵活的 Pipeline 系统，支持：
- 模块化组件设计
- 多后端支持
- 配置驱动的 Pipeline 构建
- Hook 机制用于调试和可视化
- 迭代优化支持

Example:
    # 从配置文件构建 Pipeline
    from lib.pipeline import PipelineBuilder
    
    pipeline = PipelineBuilder.from_config('configs/pipelines/emdb_basic.yaml')
    pipeline.setup()
    
    result = pipeline.execute(data)
    
    # 或者使用预设的 Pipeline
    pipeline = PipelineBuilder.create_evaluation_pipeline()

Available Components:
    - DetectionComponent: 人体检测
    - SegmentationComponent: 图像分割
    - SLAMComponent: 相机运动估计
    - HPEComponent: 人体姿态估计
    - EvaluationComponent: 评估指标计算

Available Backends:
    - Detection: vitdet
    - Segmentation: sam
    - SLAM: droid
    - HPE: vimo
"""

from .core.data import PipelineData, CameraParams, SMPLParams
from .core.component import Component, BackendComponent, Backend
from .core.pipeline import Pipeline, IterativePipeline

from .components import (
    DetectionComponent,
    SegmentationComponent,
    SLAMComponent,
    HPEComponent,
    EvaluationComponent,
)

from .builder import PipelineBuilder

from . import hooks


__all__ = [
    # Core
    'PipelineData',
    'CameraParams',
    'SMPLParams',
    'Component',
    'BackendComponent',
    'Backend',
    'Pipeline',
    'IterativePipeline',
    
    # Components
    'DetectionComponent',
    'SegmentationComponent',
    'SLAMComponent',
    'HPEComponent',
    'EvaluationComponent',
    
    # Builder
    'PipelineBuilder',
    
    # Hooks
    'hooks',
]
