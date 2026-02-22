#!/bin/bash

# Adjacent Frame SMPL Rendering (GT + Keyframe) - 单序列运行
# 使用第一次 SLAM 关键帧信息指导渲染和第二次 warm start SLAM
# 使用示例：bash scripts/run_adjacent_render_pipeline_gt_keyframe.sh

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 设置参数
CONFIG="configs/pipelines/emdb_adjacent_render_gt_keyframe.yaml"
SEQ="09_outdoor_walk"
PERSON="P0"
SPLIT=2
DATASET_PATH="datasets/EMDB"
GPU_ID=1

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "使用 GPU: $GPU_ID"
echo "使用配置: $CONFIG"
echo "处理序列: $PERSON/$SEQ"

# 运行 Pipeline
python lib/scripts/run_pipeline.py \
    --config $CONFIG \
    --seq $SEQ \
    --person $PERSON \
    --split $SPLIT \
    --dataset_path $DATASET_PATH \
    --device cuda \
    --debug \
    --visualize

echo "渲染完成"
