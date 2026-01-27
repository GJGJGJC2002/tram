#!/bin/bash

# EMDB 统一评估脚本 - 使用示例

# 激活 conda 环境
echo "激活 conda 环境: alignHMR"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 配置
DATASET_PATH="/home/gejunchen/Work/2026-1/Datasets/EMDB"
OUTPUT_DIR="results/emdb_unified"
SPLIT=2
export CUDA_VISIBLE_DEVICES=2
echo "=========================================="
echo "EMDB 统一评估脚本 - 示例运行"
echo "=========================================="
echo ""

# 示例 1: 处理单个视频（完整流程）
echo "示例 1: 处理单个视频"
echo "------------------------------------------"
python scripts/emdb_eval/eval_emdb_unified.py \
    --seq 09_outdoor_walk \
    --person P0 \
    --split $SPLIT \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR

echo ""
echo "完成！结果保存在: $OUTPUT_DIR"
echo ""

# 示例 2: 只评估已有结果（快速）
# 取消注释以下代码来运行
# echo "示例 2: 只评估已有结果"
# echo "------------------------------------------"
# python scripts/emdb_eval/eval_emdb_unified.py \
#     --seq 01_mvs_b \
#     --person P0 \
#     --split $SPLIT \
#     --dataset_path $DATASET_PATH \
#     --output_dir $OUTPUT_DIR \
#     --eval_only

# 示例 3: 批量处理多个视频（使用快速模式）
# 取消注释以下代码来运行
# echo "示例 3: 批量处理 P0 的所有视频"
# echo "------------------------------------------"
# python scripts/emdb_eval/eval_emdb_unified.py \
#     --person P0 \
#     --split $SPLIT \
#     --dataset_path $DATASET_PATH \
#     --output_dir $OUTPUT_DIR \
#     --efficient

