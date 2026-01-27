"""Pipeline - Pipeline 执行引擎"""

from typing import Dict, Any, Optional, List, Callable, Union
import logging
import time
from datetime import datetime
import os

from .data import PipelineData
from .component import Component


# Hook 类型定义
HookFunction = Callable[[PipelineData], None]


class Pipeline:
    """
    Pipeline 执行引擎
    
    负责组件的编排、执行和 Hook 管理。
    
    Attributes:
        name: Pipeline 名称
        config: 配置字典
        components: 组件列表
        hooks: Hook 函数字典
    
    Example:
        pipeline = Pipeline("my_pipeline")
        pipeline.add_component(DetectionComponent("detect", config))
        pipeline.add_component(SLAMComponent("slam", config))
        
        pipeline.add_hook("after_detect", visualize_detection)
        
        pipeline.setup()
        result = pipeline.execute(data)
        pipeline.cleanup()
    """
    
    def __init__(self, name: str = "pipeline", config: Dict[str, Any] = None):
        """
        初始化 Pipeline
        
        Args:
            name: Pipeline 名称
            config: 配置字典
        """
        self.name = name
        self.config = config or {}
        self.components: List[Component] = []
        self.hooks: Dict[str, List[HookFunction]] = {}
        
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # 状态
        self._is_setup = False
        self._execution_count = 0
        
        # 配置
        self.device = self.config.get('device', 'cuda')
        self.output_dir = self.config.get('output_dir', 'results')
        self.save_intermediate = self.config.get('save_intermediate', False)
    
    def add_component(self, component: Component) -> 'Pipeline':
        """
        添加组件到 Pipeline
        
        Args:
            component: 组件实例
        
        Returns:
            self（支持链式调用）
        """
        self.components.append(component)
        self.logger.debug(f"Added component: {component.name}")
        return self
    
    def add_components(self, components: List[Component]) -> 'Pipeline':
        """批量添加组件"""
        for component in components:
            self.add_component(component)
        return self
    
    def remove_component(self, name: str) -> bool:
        """移除指定名称的组件"""
        for i, component in enumerate(self.components):
            if component.name == name:
                self.components.pop(i)
                self.logger.debug(f"Removed component: {name}")
                return True
        return False
    
    def get_component(self, name: str) -> Optional[Component]:
        """获取指定名称的组件"""
        for component in self.components:
            if component.name == name:
                return component
        return None
    
    # === Hook 管理 ===
    
    def add_hook(self, stage: str, hook_fn: HookFunction) -> 'Pipeline':
        """
        添加 Hook 函数
        
        Hook 会在指定阶段被调用。阶段名称格式：
        - "before_{component_name}": 组件执行前
        - "after_{component_name}": 组件执行后
        - "on_error": 发生错误时
        - "on_complete": Pipeline 执行完成时
        
        Args:
            stage: 阶段名称
            hook_fn: Hook 函数，接受 PipelineData 作为参数
        
        Returns:
            self
        """
        if stage not in self.hooks:
            self.hooks[stage] = []
        self.hooks[stage].append(hook_fn)
        self.logger.debug(f"Added hook at stage '{stage}': {hook_fn.__name__}")
        return self
    
    def remove_hooks(self, stage: str):
        """移除指定阶段的所有 Hook"""
        if stage in self.hooks:
            del self.hooks[stage]
    
    def _run_hooks(self, stage: str, data: PipelineData):
        """执行指定阶段的所有 Hook"""
        hooks = self.hooks.get(stage, [])
        for hook_fn in hooks:
            try:
                hook_fn(data)
            except Exception as e:
                self.logger.warning(f"Hook '{hook_fn.__name__}' at stage '{stage}' failed: {e}")
                # Hook 失败不影响主流程
    
    # === 生命周期管理 ===
    
    def setup(self):
        """初始化所有组件"""
        self.logger.info(f"Setting up Pipeline '{self.name}' with {len(self.components)} components...")
        
        start_time = time.time()
        
        for component in self.components:
            component.ensure_setup()
        
        self._is_setup = True
        elapsed = time.time() - start_time
        self.logger.info(f"Pipeline setup completed in {elapsed:.2f}s")
    
    def cleanup(self):
        """清理所有组件资源"""
        self.logger.info("Cleaning up Pipeline...")
        
        for component in self.components:
            try:
                component.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up {component.name}: {e}")
        
        self._is_setup = False
    
    # === 执行 ===
    
    def execute(self, data: PipelineData) -> PipelineData:
        """
        执行 Pipeline
        
        按顺序执行所有组件，在每个组件前后调用相应的 Hook。
        
        Args:
            data: 输入数据容器
        
        Returns:
            处理后的数据容器
        """
        if not self._is_setup:
            self.setup()
        
        self._execution_count += 1
        self.logger.info(f"Executing Pipeline '{self.name}' (run #{self._execution_count})")
        
        start_time = time.time()
        data.metadata['pipeline_name'] = self.name
        data.metadata['execution_start'] = datetime.now().isoformat()
        
        try:
            for i, component in enumerate(self.components):
                # 检查是否应该停止
                if data.should_stop:
                    self.logger.info(f"Pipeline stopped early at component {component.name}")
                    break
                
                # 执行前 Hook
                self._run_hooks(f"before_{component.name}", data)
                
                # 执行组件
                self.logger.info(f"[{i+1}/{len(self.components)}] Running {component.name}...")
                data = component(data)
                
                # 执行后 Hook
                self._run_hooks(f"after_{component.name}", data)
                
                # 保存中间结果
                if self.save_intermediate:
                    self._save_intermediate(data, component.name)
            
            # 完成 Hook
            self._run_hooks("on_complete", data)
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            data.error_message = str(e)
            self._run_hooks("on_error", data)
            raise
        
        finally:
            elapsed = time.time() - start_time
            data.metadata['execution_time'] = elapsed
            data.metadata['execution_end'] = datetime.now().isoformat()
            self.logger.info(f"Pipeline execution completed in {elapsed:.2f}s")
        
        return data
    
    def _save_intermediate(self, data: PipelineData, stage_name: str):
        """保存中间结果"""
        intermediate_dir = os.path.join(
            self.output_dir, 
            'intermediate', 
            data.sequence_name or 'unnamed'
        )
        os.makedirs(intermediate_dir, exist_ok=True)
        
        prefix = f"iter{data.iteration}_{stage_name}"
        data.save_results(intermediate_dir, prefix)
    
    def __call__(self, data: PipelineData) -> PipelineData:
        """允许直接调用 Pipeline"""
        return self.execute(data)
    
    # === 信息查询 ===
    
    def get_component_names(self) -> List[str]:
        """获取所有组件名称"""
        return [c.name for c in self.components]
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行统计摘要"""
        return {
            'pipeline_name': self.name,
            'num_components': len(self.components),
            'total_executions': self._execution_count,
            'component_times': {
                c.name: {
                    'avg': c.get_avg_execution_time(),
                    'total': c.get_total_execution_time()
                }
                for c in self.components
            }
        }
    
    def __repr__(self) -> str:
        return (
            f"Pipeline(name='{self.name}', "
            f"components={self.get_component_names()})"
        )


class IterativePipeline(Pipeline):
    """
    支持迭代优化的 Pipeline
    
    可以多次执行 Pipeline，直到收敛或达到最大迭代次数。
    适用于需要迭代优化的场景，如相机-人体联合优化。
    
    Attributes:
        max_iterations: 最大迭代次数
        convergence_threshold: 收敛阈值
        convergence_metric: 用于判断收敛的指标名称
    """
    
    def __init__(self, name: str = "iterative_pipeline", config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        self.max_iterations = self.config.get('max_iterations', 3)
        self.convergence_threshold = self.config.get('convergence_threshold', 1e-4)
        self.convergence_metric = self.config.get('convergence_metric', 'reprojection_error')
    
    def execute(self, data: PipelineData) -> PipelineData:
        """
        迭代执行 Pipeline
        
        Args:
            data: 输入数据容器
        
        Returns:
            处理后的数据容器
        """
        if not self._is_setup:
            self.setup()
        
        self._execution_count += 1
        self.logger.info(
            f"Executing IterativePipeline '{self.name}' "
            f"(max_iterations={self.max_iterations})"
        )
        
        prev_metric = float('inf')
        
        for iteration in range(self.max_iterations):
            data.iteration = iteration
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            self.logger.info(f"{'='*60}")
            
            # 执行一次完整的 Pipeline
            data = super().execute(data)
            
            # 检查是否应该停止
            if data.should_stop:
                self.logger.info("Pipeline stopped by component")
                break
            
            # 检查收敛
            current_metric = data.metrics.get(self.convergence_metric, float('inf'))
            metric_change = abs(prev_metric - current_metric)
            
            self.logger.info(
                f"Convergence metric '{self.convergence_metric}': "
                f"{current_metric:.6f} (change: {metric_change:.6f})"
            )
            
            if metric_change < self.convergence_threshold:
                self.logger.info(
                    f"Converged! Change ({metric_change:.6f}) < "
                    f"threshold ({self.convergence_threshold})"
                )
                break
            
            prev_metric = current_metric
            
            # 迭代间 Hook
            self._run_hooks("on_iteration_end", data)
        
        data.metadata['final_iteration'] = data.iteration + 1
        return data


