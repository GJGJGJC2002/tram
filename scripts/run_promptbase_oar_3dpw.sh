#!/bin/bash

# PromptHMR + OAR Pipeline (3DPW Test Set)
# 使用示例：
#   bash scripts/run_promptbase_oar_3dpw.sh                  # 处理所有测试序列所有人
#   SEQ=downtown_arguing_00 bash scripts/run_promptbase_oar_3dpw.sh   # 单个序列
#   SEQ=downtown_arguing_00 PERSON=0 bash scripts/run_promptbase_oar_3dpw.sh  # 单人

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 设置参数（可通过环境变量覆盖）
CONFIG="configs/pipelines/promptbase_video_depthrefine_3dpw.yaml"
SEQ="${SEQ:-}"
PERSON="${PERSON:-}"
SPLIT="${SPLIT:-test}"
DATASET_PATH="${DATASET_PATH:-datasets/3DPW}"
GPU_ID="${GPU_ID:-0}"

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "使用 GPU: $GPU_ID"
echo "使用配置: $CONFIG"
echo "3DPW Split: $SPLIT"

# 构建运行命令
CMD="python lib/scripts/run_pipeline.py \
    --config $CONFIG \
    --dataset 3dpw \
    --split $SPLIT \
    --dataset_path $DATASET_PATH \
    --device cuda \
    --debug \
    --visualize"

# 如果指定了序列，则添加参数
if [ -n "$SEQ" ]; then
    CMD="$CMD --seq $SEQ"
    echo "处理序列: $SEQ"
fi

# 如果指定了人物索引，则添加参数
if [ -n "$PERSON" ]; then
    CMD="$CMD --person $PERSON"
    echo "处理人物: $PERSON"
fi

# 运行 Pipeline
eval $CMD

echo "PromptHMR 3DPW 评估完成"
