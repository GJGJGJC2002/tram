#!/bin/bash
# ================================================================
# 交互式多方法 SMPL 运动序列对比可视化（基于 viser）
#
# 在浏览器中实时对比不同 HMR 方法的全局运动。
# 启动后访问 http://localhost:8080 即可交互。
#
# 用法:
#   bash scripts/run_viser_compare.sh                    # 默认参数
#   SEQ=55_outdoor_walk bash scripts/run_viser_compare.sh # 指定序列
#   SHOW_GT=1 bash scripts/run_viser_compare.sh           # 显示 GT
# ================================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# ---------- 可配置参数 ----------
SEQ="${SEQ:-09_outdoor_walk}"
PORT="${PORT:-8080}"
SUBSAMPLE="${SUBSAMPLE:-10}"
SHOW_GT="${SHOW_GT:-1}"
DATASET_ROOT="${DATASET_ROOT:-datasets/EMDB}"
SPREAD_GAP="${SPREAD_GAP:-1.0}"
NO_SPREAD="${NO_SPREAD:-0}"
START_FRAME="${START_FRAME:-50}"
END_FRAME="${END_FRAME:-150}"

# 方法列表（格式: Name:result_dir，空格分隔）
METHODS="${METHODS:-Prompt:results/promptbase_video_warmstart_emdb2 Ours:results/gvhmr_base_warmstart}"
# ------------------------------------

echo "=========================================="
echo "Viser 交互式多方法对比可视化"
echo "=========================================="
echo "序列: $SEQ"
echo "方法: $METHODS"
echo "端口: $PORT"
echo "帧采样: $SUBSAMPLE"
echo "帧范围: [$START_FRAME, $END_FRAME]"
echo "显示GT: $SHOW_GT"
echo "方法间距: $SPREAD_GAP m"
echo "=========================================="

CMD="python lib/scripts/viser_compare_methods.py \
    --sequence $SEQ \
    --methods $METHODS \
    --subsample $SUBSAMPLE \
    --start-frame $START_FRAME \
    --end-frame $END_FRAME \
    --spread-gap $SPREAD_GAP \
    --port $PORT"

if [ "$SHOW_GT" = "1" ]; then
    CMD="$CMD --show-gt --dataset-root $DATASET_ROOT"
fi

if [ "$NO_SPREAD" = "1" ]; then
    CMD="$CMD --no-spread"
fi

eval $CMD
