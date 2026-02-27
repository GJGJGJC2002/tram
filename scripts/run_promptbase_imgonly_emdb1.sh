#!/bin/bash

# PromptHMR Image-Only Pipeline (EMDB1 - InCam Evaluation)
# 仅使用 PromptHMR 图像模型逐帧推理，不经过视频头
# 与 PromptHMR 官方评估 (eval_phmr.py) 完全对齐
# 使用示例：bash scripts/run_promptbase_imgonly_emdb1.sh

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 设置参数
CONFIG="configs/pipelines/promptbase_imgonly_emdb1.yaml"
SEQ="14_outdoor_climb"
PERSON="P1"
SPLIT=1
DATASET_PATH="datasets/EMDB"
GPU_ID=1

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "使用 GPU: $GPU_ID"
echo "使用配置: $CONFIG"
echo "EMDB Split: $SPLIT (InCam Evaluation - Image Only)"

# 构建运行命令
CMD="python lib/scripts/run_pipeline.py \
    --config $CONFIG \
    --split $SPLIT \
    --dataset_path $DATASET_PATH \
    --device cuda \
    --debug \
    --visualize"

# 如果指定了序列和人物，则添加参数
if [ -n "$SEQ" ]; then
    CMD="$CMD --seq $SEQ"
    echo "处理序列: $SEQ"
fi

if [ -n "$PERSON" ]; then
    CMD="$CMD --person $PERSON"
    echo "处理人物: $PERSON"
fi

# 运行 Pipeline
eval $CMD

echo "PromptHMR Image-Only EMDB1 评估完成"
