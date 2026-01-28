#!/bin/bash

# Pipeline 运行脚本
# 使用示例：bash scripts/run_pipeline.sh

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 设置参数
CONFIG="configs/pipelines/emdb_basic.yaml"
SEQ="09_outdoor_walk"
PERSON="P0"
SPLIT=2
DATASET_PATH="/home/gejunchen/Work/2026-1/Datasets/EMDB"
OUTPUT_DIR="results/pipeline"
GPU_ID=3

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
    --output_dir $OUTPUT_DIR \
    --device cuda \
    --debug \
    --visualize

echo "Pipeline 执行完成！"
