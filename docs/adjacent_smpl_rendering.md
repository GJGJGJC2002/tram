# Adjacent SMPL Rendering Component

## 功能说明

这个组件复现了`adjTram`的功能，用于渲染相邻帧的SMPL mesh。对每一帧，它会在当前帧的相机视角下渲染：
- 当前帧的SMPL mesh
- 前x帧的SMPL mesh
- 后x帧的SMPL mesh

## 主要用途

1. **可视化时间一致性**：查看SMPL参数在不同帧之间的连续性
2. **调试坐标变换**：验证world2cam变换是否正确
3. **对比GT和预测**：同时渲染GT和预测结果，便于比较

## 使用方法

### 1. 基本使用

```bash
# 使用默认参数（pre_dis=20, person=P0, seq=00_mvs_a）
./scripts/run_adjacent_render_pipeline.sh

# 指定序列和人物
./scripts/run_adjacent_render_pipeline.sh --seq "00_mvs_a" --person "P0"

# 自定义帧间隔
./scripts/run_adjacent_render_pipeline.sh --seq "00_mvs_a" --person "P0" --pre_dis 10

# 指定GPU
./scripts/run_adjacent_render_pipeline.sh --seq "00_mvs_a" --person "P0" --gpu 1

# 组合使用
./scripts/run_adjacent_render_pipeline.sh \
    --seq "09_outdoor_walk" \
    --person "P1" \
    --pre_dis 15 \
    --gpu 0
```

**注意**：
- `--seq` 参数使用序列名（不含person前缀），如 "00_mvs_a"
- `--person` 参数指定人物文件夹，如 "P0"
- 完整路径会自动组合为 `P0/00_mvs_a`

### 2. 修改配置文件

编辑 `configs/pipelines/emdb_adjacent_render.yaml`：

```yaml
# 5. Adjacent Frame SMPL Rendering
- type: adjacent_smpl_renderer
  name: adjacent_frame_rendering
  pre_dis: 20  # 调整这个参数改变帧间隔
  device: "cuda"  # 或 "cpu"
  output_dir: "results/emdb_adjacent_render"
  save_gt: true  # 是否同时渲染GT
  render_gt_only: false  # 如果只想看GT，设为true
```

### 3. 在Python中使用

```python
from lib.pipeline.builder import PipelineBuilder

# 构建pipeline
pipeline = PipelineBuilder.from_config('configs/pipelines/emdb_adjacent_render.yaml')

# 运行
data = {
    'sequence_path': 'datasets/EMDB/P0/00_mvs_a',
    'sequence_name': 'P0_00_mvs_a',
}

result = pipeline.run(data)
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pre_dis` | int | 20 | 相邻帧间隔（前后各x帧） |
| `device` | str | 'cpu' | 计算设备（'cpu'或'cuda'） |
| `output_dir` | str | 'results/adjacent_smpl' | 输出目录 |
| `save_gt` | bool | True | 是否同时渲染GT SMPL |
| `render_gt_only` | bool | False | 是否只渲染GT（用于GT可视化） |

## 输出格式

对每一帧，输出的图像包含：

```
┌─────────────┬─────────────┬─────────────┐
│  pre_{x}    │  current    │  next_{x}   │  ← GT (如果save_gt=True)
├─────────────┼─────────────┼─────────────┤
│  pre_{x}    │  current    │  next_{x}   │  ← TRAM预测
└─────────────┴─────────────┴─────────────┘
```

- 上半部分：GT的渲染结果（如果`save_gt=True`）
- 下半部分：TRAM预测的渲染结果（如果`render_gt_only=False`）
- 每部分从左到右：前x帧 | 当前帧 | 后x帧

## 关键实现

### 1. 坐标变换

```python
# 从世界坐标系转换到相机坐标系
local_vertices = np.einsum("ij, nj->ni", w2c[:, :3, :3], vertices_world) + w2c[:, :3, 3]
```

### 2. 相邻帧渲染

对于每一帧i，渲染三帧的mesh：
- 帧i-pre_dis：使用帧i的相机参数，渲染帧i-pre_dis的SMPL mesh
- 帧i：使用帧i的相机参数，渲染帧i的SMPL mesh
- 帧i+pre_dis：使用帧i的相机参数，渲染帧i+pre_dis的SMPL mesh

这样可以清楚地看到SMPL mesh在不同时间步的变化。

### 3. 双渲染器

- **TRAM渲染器**：使用TRAM估计的内参
- **GT渲染器**：使用GT的内参

确保渲染结果使用正确的相机内参，避免因内参不匹配导致的偏移。

## 示例输出

```
results/emdb_adjacent_render/
├── P0_00_mvs_a/
│   ├── frame_000001.jpg  # [pre_20 | current | next_20]
│   ├── frame_000002.jpg
│   └── ...
└── ...
```

## 依赖

- pytorch3d：用于3D渲染
- opencv：用于图像读取和保存
- smplx：用于SMPL模型
- tqdm：用于进度条显示

## 故障排除

### 1. 如果渲染结果有偏移

检查：
- GT的内参是否正确保存到metadata中
- world2cam变换是否正确应用
- SMPL vertices是否在正确的坐标系中

### 2. 如果渲染速度慢

尝试：
- 使用`device: "cuda"`加速
- 减少`pre_dis`值
- 使用更小的图像尺寸

### 3. 如果内存不足

- 设置`save_gt: false`只渲染TRAM结果
- 使用CPU而不是GPU
- 减少batch size或处理更短的序列
