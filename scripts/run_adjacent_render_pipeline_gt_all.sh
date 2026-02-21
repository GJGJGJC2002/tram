#!/bin/bash

# Adjacent Frame SMPL Rendering (GT) - 在所有 EMDB2 序列上运行评估
# 使用示例：bash scripts/run_adjacent_render_pipeline_gt_all.sh

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 设置参数
CONFIG="configs/pipelines/emdb_adjacent_render_gt.yaml"
SPLIT=2
DATASET_PATH="datasets/EMDB"
GPU_ID=3

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "使用 GPU: $GPU_ID"
echo "使用配置: $CONFIG"
echo "处理所有 EMDB${SPLIT} 序列"

# 运行 Pipeline（不指定 --seq 和 --person，处理整个 split）
python lib/scripts/run_pipeline.py \
    --config $CONFIG \
    --split $SPLIT \
    --dataset_path $DATASET_PATH \
    --device cuda \
    --debug \
    --visualize

echo "所有序列评估完成"
