# Pipeline 架构使用指南

## 概述

新的 Pipeline 架构提供了高度模块化和可配置的方式来运行 EMDB 评估。

## 目录结构

```
lib/pipeline/
├── core/                    # 核心模块
│   ├── data.py             # PipelineData 数据容器
│   ├── component.py        # Component 基类
│   └── pipeline.py         # Pipeline 执行引擎
├── components/             # 功能组件
│   ├── detection.py        # 人体检测
│   ├── segmentation.py     # 图像分割
│   ├── slam.py             # 相机估计
│   ├── hpe.py              # 人体姿态估计
│   └── evaluation.py       # 评估指标
├── backends/               # 算法后端
│   ├── detection/          # VitDet
│   ├── segmentation/       # SAM
│   ├── slam/               # DroidSLAM
│   └── hpe/                # VIMO
├── builder.py              # Pipeline 构建器
└── hooks.py                # 预定义 hooks

configs/pipelines/
├── emdb_basic.yaml         # 基础 Pipeline
├── emdb_efficient.yaml     # 快速模式
└── emdb_iterative.yaml     # 迭代优化
```

## 快速开始

### 1. 使用预设脚本运行

```bash
# 激活环境
conda activate alignHMR

# 运行基础 Pipeline
bash scripts/run_pipeline.sh
```

### 2. 使用命令行参数

```bash
# 基础用法：处理单个序列
python scripts/run_pipeline.py \
    --config configs/pipelines/emdb_basic.yaml \
    --seq 09_outdoor_walk \
    --person P0 \
    --split 2

# 处理整个 split
python scripts/run_pipeline.py \
    --config configs/pipelines/emdb_basic.yaml \
    --split 2

# 使用预设 Pipeline
python scripts/run_pipeline.py \
    --preset basic \
    --seq 09_outdoor_walk \
    --person P0

# 启用调试和可视化
python scripts/run_pipeline.py \
    --preset evaluation \
    --split 2 \
    --debug \
    --visualize \
    --save_intermediate
```

### 3. 使用不同的配置

```bash
# 快速模式（efficient SMPL 估计）
python scripts/run_pipeline.py \
    --config configs/pipelines/emdb_efficient.yaml \
    --split 2

# 迭代优化模式
python scripts/run_pipeline.py \
    --config configs/pipelines/emdb_iterative.yaml \
    --seq 09_outdoor_walk \
    --person P0
```

## 命令行参数

### Pipeline 配置
- `--config`: YAML 配置文件路径
- `--preset`: 使用预设配置 (basic, efficient, iterative, evaluation)

### 数据集参数
- `--seq`: 指定序列名称 (如 09_outdoor_walk)
- `--person`: 指定人物 (如 P0)
- `--split`: EMDB split 编号 (1, 2, 或 3)
- `--dataset_path`: EMDB 数据集路径

### 输出参数
- `--output_dir`: 输出目录
- `--save_intermediate`: 保存中间结果

### Pipeline 参数
- `--device`: 计算设备 (cuda 或 cpu)
- `--hpe_mode`: HPE 模式 (accurate 或 efficient)

### 调试选项
- `--debug`: 启用调试 hooks
- `--visualize`: 启用可视化 hooks

## 配置文件说明

### 基础配置 (emdb_basic.yaml)

```yaml
name: "EMDB Basic Pipeline"
mode: "sequential"
device: "cuda"
output_dir: "results/emdb"

components:
  - type: detection
    backend: vitdet
    threshold: 0.5
    
  - type: segmentation
    backend: sam
    
  - type: slam
    backend: droid
    use_masks: true
    
  - type: hpe
    backend: vimo
    mode: accurate
    
  - type: evaluation
    metrics:
      - pa_mpjpe
      - mpjpe
      - w_mpjpe
      - ate
```

### 迭代优化配置 (emdb_iterative.yaml)

```yaml
name: "EMDB Iterative Pipeline"
mode: "iterative"
max_iterations: 3
convergence_threshold: 0.001
convergence_metric: "reprojection_error"

components:
  # ... 同基础配置 ...
```

## 自定义 Pipeline

### 方法 1: 修改 YAML 配置

创建新的配置文件：

```yaml
name: "My Custom Pipeline"
mode: "sequential"
device: "cuda"

components:
  - type: detection
    name: my_detector
    backend: vitdet
    threshold: 0.7  # 自定义阈值
    
  - type: slam
    name: my_slam
    backend: droid
    use_masks: false  # 不使用 mask
```

### 方法 2: 使用 Python API

```python
from lib.pipeline import PipelineBuilder, PipelineData

# 方式 1: 从配置文件
pipeline = PipelineBuilder.from_config('my_config.yaml')

# 方式 2: 使用预设
pipeline = PipelineBuilder.create_evaluation_pipeline(
    name="my_pipeline",
    device="cuda",
    output_dir="results/my_exp"
)

# 方式 3: 手动构建
from lib.pipeline.components import DetectionComponent, SLAMComponent

pipeline = Pipeline(name="custom")
pipeline.add_component(DetectionComponent("det", config={...}))
pipeline.add_component(SLAMComponent("slam", config={...}))

# 执行
pipeline.setup()
result = pipeline.execute(data)
pipeline.cleanup()
```

## 添加 Hooks

Hooks 允许在 Pipeline 执行的特定阶段插入自定义逻辑。

### 预定义 Hooks

```python
from lib.pipeline import hooks

# 添加调试 hooks
for stage, hook_list in hooks.DEBUG_HOOKS.items():
    for hook_fn in hook_list:
        pipeline.add_hook(stage, hook_fn)

# 添加可视化 hooks
for stage, hook_list in hooks.VISUALIZATION_HOOKS.items():
    for hook_fn in hook_list:
        pipeline.add_hook(stage, hook_fn)

# 添加性能监控 hooks
for stage, hook_list in hooks.MONITORING_HOOKS.items():
    for hook_fn in hook_list:
        pipeline.add_hook(stage, hook_fn)
```

### 自定义 Hooks

```python
def my_hook(data: PipelineData):
    """自定义 hook 函数"""
    print(f"Processing {data.sequence_name}")
    print(f"Current stage: {data.current_stage}")
    # 执行自定义操作...

# 添加到 Pipeline
pipeline.add_hook("after_detection", my_hook)
pipeline.add_hook("on_complete", my_hook)
```

### Hook 阶段

可用的 hook 阶段：
- `before_{component_name}`: 组件执行前
- `after_{component_name}`: 组件执行后
- `on_error`: 发生错误时
- `on_complete`: Pipeline 完成时
- `on_iteration_end`: 迭代结束时（仅 IterativePipeline）

## 扩展组件

### 添加新的后端

```python
from lib.pipeline.core.component import Backend

class MyCustomBackend(Backend):
    def setup(self):
        # 初始化模型
        self.model = load_my_model()
        self._is_setup = True
    
    def my_method(self, inputs):
        # 实现具体功能
        return self.model(inputs)

# 注册后端
from lib.pipeline.components import DetectionComponent
DetectionComponent.register_backend('my_backend', MyCustomBackend)
```

### 添加新的组件类型

```python
from lib.pipeline.core.component import Component
from lib.pipeline.core.data import PipelineData

class MyComponent(Component):
    COMPONENT_TYPE = "my_component"
    
    def setup(self):
        # 初始化
        pass
    
    def validate_input(self, data: PipelineData) -> bool:
        # 验证输入
        return True
    
    def execute(self, data: PipelineData) -> PipelineData:
        # 执行功能
        # ... 处理 data ...
        return data

# 注册组件
from lib.pipeline import PipelineBuilder
PipelineBuilder.register_component('my_component', MyComponent)
```

## 输出结果

Pipeline 执行后会生成以下文件：

```
results/pipeline/
├── pipeline.log                    # 日志文件
├── evaluation_results.xlsx         # 所有序列的评估结果
├── summary.xlsx                    # 汇总统计
└── {sequence_name}/
    ├── camera.npz                  # 相机参数
    ├── smpl.npz                    # SMPL 参数
    └── metrics.json                # 评估指标
```

如果启用 `--save_intermediate`：

```
results/pipeline/
└── intermediate/
    └── {sequence_name}/
        ├── iter0_human_detection/
        ├── iter0_camera_estimation/
        └── iter0_smpl_estimation/
```

## 评估指标

Pipeline 计算以下指标：

### 局部运动指标
- **PA-MPJPE**: Procrustes-Aligned Mean Per Joint Position Error
- **MPJPE**: Mean Per Joint Position Error
- **PVE**: Per Vertex Error
- **Accel**: Acceleration Error

### 全局运动指标
- **W-MPJPE**: World-aligned MPJPE (分块对齐)
- **WA-MPJPE**: World-aligned MPJPE (全局对齐)
- **RTE**: Root Trajectory Error
- **ERVE**: Ego-centric Root Velocity Error

### 相机运动指标
- **ATE**: Absolute Trajectory Error (with scale)
- **ATE_S**: Absolute Trajectory Error (without scale)

## 故障排除

### 常见问题

1. **ImportError: cannot import name 'SMPL'**
   - 确保已正确安装所有依赖
   - 检查 `lib/models/smpl.py` 是否存在

2. **CUDA out of memory**
   - 使用 `--device cpu`
   - 或减少 batch_size（修改配置文件）

3. **No sequences found**
   - 检查 `--dataset_path` 是否正确
   - 确认序列属于指定的 split

4. **版本不兼容**
   - NumPy: 使用 1.23.5
   - PyTorch: 使用兼容版本

### 测试安装

```bash
# 测试模块导入
python scripts/test_pipeline_import.py

# 测试单个序列（快速）
python scripts/run_pipeline.py \
    --preset basic \
    --seq 09_outdoor_walk \
    --person P0 \
    --split 2
```

## 性能优化

### 使用 Efficient 模式

```bash
python scripts/run_pipeline.py \
    --config configs/pipelines/emdb_efficient.yaml \
    --split 2
```

### 跳过某些组件

修改配置文件，移除不需要的组件。

### 并行处理多个序列

```bash
# 使用 GNU parallel
parallel -j 4 python scripts/run_pipeline.py \
    --preset basic --seq {} --person P0 \
    ::: 09_outdoor_walk 10_indoor_sitting 11_outdoor_run
```

## 更多信息

- 查看 `lib/pipeline/` 源代码了解实现细节
- 参考 `configs/pipelines/` 了解配置选项
- 查看 `lib/pipeline/hooks.py` 了解可用的 hooks

## 联系和支持

如有问题，请查看项目 README 或提交 issue。

