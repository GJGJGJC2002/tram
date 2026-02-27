#!/bin/bash

# GVHMR Base + Warmstart Pipeline
# 基于 gvhmr_base，第 6 步替换为 droid_warmstart
# 使用示例：bash scripts/run_adjacent_render_pipeline_gvbase_warmstart.sh

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 设置参数
CONFIG="configs/pipelines/gvhmr_base_warmstart.yaml"
SEQ="48_outdoor_walk_downhill"
PERSON="P6"
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
