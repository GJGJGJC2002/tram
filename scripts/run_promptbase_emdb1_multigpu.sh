#!/bin/bash

# PromptHMR Base Pipeline (EMDB1 - InCam Evaluation) - 多 GPU 并行
# EMDB1 仅评估 incam 指标（pa_mpjpe, mpjpe, pve, accel）
# 将序列均分到多张 GPU 上同时运行
# 使用示例：bash scripts/run_promptbase_emdb1_multigpu.sh
# 自定义 GPU：GPU_IDS="0,1,2" bash scripts/run_promptbase_emdb1_multigpu.sh

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 设置参数
CONFIG="configs/pipelines/promptbase_emdb1.yaml"
SPLIT=1
DATASET_PATH="datasets/EMDB"

# GPU 列表（可通过环境变量覆盖）
GPU_IDS="${GPU_IDS:-1,2,3}"

# 解析 GPU 列表
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

echo "============================================"
echo "多 GPU 并行运行 PromptHMR Base Pipeline (EMDB1)"
echo "GPU 列表: ${GPU_IDS} (共 ${NUM_GPUS} 张)"
echo "配置文件: ${CONFIG}"
echo "EMDB Split: ${SPLIT} (InCam Evaluation)"
echo "============================================"

# 存放各进程 PID
PIDS=()

# 为每张 GPU 启动一个 shard
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU=${GPUS[$i]}
    echo ""
    echo "[Shard ${i}/${NUM_GPUS}] 使用 GPU ${GPU} 启动..."

    CUDA_VISIBLE_DEVICES=$GPU python lib/scripts/run_pipeline.py \
        --config $CONFIG \
        --split $SPLIT \
        --dataset_path $DATASET_PATH \
        --device cuda \
        --debug \
        --visualize \
        --shard_id $i \
        --num_shards $NUM_GPUS \
        2>&1 | sed "s/^/[GPU${GPU}] /" &

    PIDS+=($!)
done

echo ""
echo "所有 ${NUM_GPUS} 个进程已启动，PID: ${PIDS[*]}"
echo "等待所有进程完成..."

# 等待所有进程并收集退出码
FAILED=0
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[Shard ${i}] GPU ${GPUS[$i]} 进程失败，退出码: ${EXIT_CODE}"
        FAILED=$((FAILED + 1))
    else
        echo "[Shard ${i}] GPU ${GPUS[$i]} 进程完成"
    fi
done

echo ""
echo "============================================"
if [ $FAILED -eq 0 ]; then
    echo "所有 ${NUM_GPUS} 个分片全部成功完成！"
else
    echo "警告：${FAILED}/${NUM_GPUS} 个分片失败"
fi
echo "============================================"
