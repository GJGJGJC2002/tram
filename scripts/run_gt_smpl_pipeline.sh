#!/bin/bash

# EMDB GT SMPL Pipeline 运行脚本
# 使用真值SMPL参数进行可视化和评估
#
# 使用示例：
#   bash scripts/run_gt_smpl_pipeline.sh

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 设置参数
CONFIG="configs/pipelines/emdb_gt_smpl.yaml"
SEQ="09_outdoor_walk"
PERSON="P0"
SPLIT=2
DATASET_PATH="datasets/EMDB"
GPU_ID=3

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "使用 GPU: $GPU_ID"
echo "使用配置: $CONFIG"
echo "处理序列: $PERSON/$SEQ"
echo "Pipeline模式: 使用真值SMPL参数"

# 运行 Pipeline
python lib/scripts/run_pipeline.py \
    --config $CONFIG \
    --seq $SEQ \
    --person $PERSON \
    --split $SPLIT \
    --dataset_path $DATASET_PATH \
    --device cuda \
    --visualize

echo "GT SMPL Pipeline 执行完成！"
