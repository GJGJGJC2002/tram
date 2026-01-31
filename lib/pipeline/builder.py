"""PipelineBuilder - 从配置文件构建 Pipeline"""

from typing import Dict, Any, Optional, Type, List
import os
import yaml
import logging

from .core.pipeline import Pipeline, IterativePipeline
from .core.component import Component
from .components import (
    DetectionComponent,
    SegmentationComponent,
    SLAMComponent,
    HPEComponent,
    EvaluationComponent,
    AdjacentSMPLRenderer,
)


logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Pipeline 构建器
    
    从 YAML 配置文件或字典配置构建 Pipeline 实例。
    
    Example:
        # 从配置文件构建
        pipeline = PipelineBuilder.from_config('configs/pipelines/basic.yaml')
        
        # 从字典构建
        config = {
            'name': 'my_pipeline',
            'components': [
                {'type': 'detection', 'backend': 'vitdet'},
                {'type': 'slam', 'backend': 'droid'},
            ]
        }
        pipeline = PipelineBuilder.from_dict(config)
    """
    
    # 组件类型注册表
    COMPONENT_REGISTRY: Dict[str, Type[Component]] = {
        'detection': DetectionComponent,
        'segmentation': SegmentationComponent,
        'slam': SLAMComponent,
        'hpe': HPEComponent,
        'evaluation': EvaluationComponent,
        'adjacent_smpl_renderer': AdjacentSMPLRenderer,
    }
    
    @classmethod
    def register_component(cls, type_name: str, component_cls: Type[Component]):
        """
        注册新的组件类型
        
        Args:
            type_name: 组件类型名称
            component_cls: 组件类
        """
        cls.COMPONENT_REGISTRY[type_name] = component_cls
        logger.info(f"Registered component type: {type_name}")
    
    @classmethod
    def list_component_types(cls) -> List[str]:
        """列出所有已注册的组件类型"""
        return list(cls.COMPONENT_REGISTRY.keys())
    
    @classmethod
    def from_config(cls, config_path: str) -> Pipeline:
        """
        从 YAML 配置文件构建 Pipeline
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            Pipeline 实例
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls.from_dict(config)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Pipeline:
        """
        从字典配置构建 Pipeline
        
        Args:
            config: 配置字典
        
        Returns:
            Pipeline 实例
        """
        # 获取 Pipeline 类型
        pipeline_mode = config.get('mode', 'sequential')
        pipeline_name = config.get('name', 'pipeline')
        
        # 创建 Pipeline 实例
        if pipeline_mode == 'iterative':
            pipeline = IterativePipeline(name=pipeline_name, config=config)
        else:
            pipeline = Pipeline(name=pipeline_name, config=config)
        
        # 添加组件
        components_config = config.get('components', [])
        for i, comp_config in enumerate(components_config):
            component = cls._create_component(comp_config, i)
            pipeline.add_component(component)
        
        # 添加 hooks
        hooks_config = config.get('hooks', [])
        for hook_config in hooks_config:
            cls._add_hook(pipeline, hook_config)
        
        logger.info(f"Built Pipeline '{pipeline_name}' with {len(components_config)} components")
        
        return pipeline
    
    @classmethod
    def _create_component(cls, comp_config: Dict[str, Any], index: int) -> Component:
        """创建组件实例"""
        comp_type = comp_config.get('type')
        if comp_type is None:
            raise ValueError(f"Component config missing 'type' field: {comp_config}")
        
        if comp_type not in cls.COMPONENT_REGISTRY:
            raise ValueError(
                f"Unknown component type: {comp_type}. "
                f"Available types: {cls.list_component_types()}"
            )
        
        # 获取组件名称
        comp_name = comp_config.get('name', f"{comp_type}_{index}")
        
        # 创建组件
        component_cls = cls.COMPONENT_REGISTRY[comp_type]
        component = component_cls(name=comp_name, config=comp_config)
        
        return component
    
    @classmethod
    def _add_hook(cls, pipeline: Pipeline, hook_config: Dict[str, Any]):
        """添加 hook 到 pipeline"""
        stage = hook_config.get('stage')
        function_path = hook_config.get('function')
        enabled = hook_config.get('enabled', True)
        params = hook_config.get('params', {})

        if not stage or not function_path:
            logger.warning(f"Invalid hook config: {hook_config}")
            return

        if not enabled:
            logger.info(f"Hook {function_path} is disabled, skipping")
            return

        try:
            # 动态导入 hook 函数
            hook_fn = cls._import_function(function_path)

            # 如果有参数，创建偏函数
            if params:
                import functools
                hook_fn = functools.partial(hook_fn, **params)

            pipeline.add_hook(stage, hook_fn)
            logger.info(f"Added hook: {function_path} at stage '{stage}'")
        except Exception as e:
            logger.warning(f"Failed to add hook {function_path}: {e}")
    
    @staticmethod
    def _import_function(function_path: str):
        """动态导入函数"""
        parts = function_path.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid function path: {function_path}")
        
        module_path, function_name = parts
        
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, function_name)
    
    @classmethod
    def create_basic_pipeline(
        cls,
        name: str = "basic_pipeline",
        detection_backend: str = "vitdet",
        slam_backend: str = "droid",
        hpe_backend: str = "vimo",
        device: str = "cuda",
        output_dir: str = "results"
    ) -> Pipeline:
        """
        创建基础的两阶段 Pipeline
        
        相机估计 -> 人体估计
        
        Args:
            name: Pipeline 名称
            detection_backend: 检测后端
            slam_backend: SLAM 后端
            hpe_backend: HPE 后端
            device: 计算设备
            output_dir: 输出目录
        """
        config = {
            'name': name,
            'device': device,
            'output_dir': output_dir,
            'components': [
                {
                    'type': 'detection',
                    'name': 'detection',
                    'backend': detection_backend,
                    'device': device,
                },
                {
                    'type': 'segmentation',
                    'name': 'segmentation',
                    'backend': 'sam',
                    'device': device,
                },
                {
                    'type': 'slam',
                    'name': 'slam',
                    'backend': slam_backend,
                    'device': device,
                },
                {
                    'type': 'hpe',
                    'name': 'hpe',
                    'backend': hpe_backend,
                    'device': device,
                },
            ]
        }
        
        return cls.from_dict(config)
    
    @classmethod
    def create_evaluation_pipeline(
        cls,
        name: str = "eval_pipeline",
        device: str = "cuda",
        output_dir: str = "results",
        metrics: List[str] = None
    ) -> Pipeline:
        """
        创建包含评估的完整 Pipeline
        
        检测 -> 分割 -> SLAM -> HPE -> 评估
        """
        config = {
            'name': name,
            'device': device,
            'output_dir': output_dir,
            'components': [
                {
                    'type': 'detection',
                    'name': 'detection',
                    'backend': 'vitdet',
                    'device': device,
                },
                {
                    'type': 'segmentation',
                    'name': 'segmentation',
                    'backend': 'sam',
                    'device': device,
                },
                {
                    'type': 'slam',
                    'name': 'slam',
                    'backend': 'droid',
                    'device': device,
                },
                {
                    'type': 'hpe',
                    'name': 'hpe',
                    'backend': 'vimo',
                    'device': device,
                },
                {
                    'type': 'evaluation',
                    'name': 'evaluation',
                    'metrics': metrics or EvaluationComponent.ALL_METRICS,
                },
            ]
        }
        
        return cls.from_dict(config)


