#!/bin/bash

# Adjacent Frame SMPL Rendering 运行脚本
# 使用示例：bash scripts/run_adjacent_render_pipeline.sh

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 设置参数
CONFIG="configs/pipelines/gvhmr_base.yaml"
SEQ="24_outdoor_long_walk"
PERSON="P2"
SPLIT=2
DATASET_PATH="datasets/EMDB"
GPU_ID=0


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
