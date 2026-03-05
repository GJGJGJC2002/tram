#!/bin/bash

# OAR 窗口级 Debug 可视化（viser）
# 对指定窗口可视化：原始 → 滑步消除 → 深度穿透修正 三组 mesh + 深度点云
#
# 用法:
#   bash scripts/run_viser_oar_window_debug.sh
#   SEQ=09_outdoor_walk CENTER=225 bash scripts/run_viser_oar_window_debug.sh
#   PORT=9090 bash scripts/run_viser_oar_window_debug.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# 可通过环境变量覆盖
RESULT_DIR="${RESULT_DIR:-results/promptbase_video_depthrefine_emdb2}"
SEQ="${SEQ:-09_outdoor_walk}"
CENTER="${CENTER:-100}"
WINDOW="${WINDOW:-21}"
PORT="${PORT:-8080}"
DEVICE="${DEVICE:-cuda}"
DEPTH_MODEL="${DEPTH_MODEL:-metric3d_vit_small}"
CACHE_STAGE="${CACHE_STAGE:-world_transform}"
PC_SUBSAMPLE="${PC_SUBSAMPLE:-4}"

echo "============================================"
echo "OAR Window Debug Visualization (viser)"
echo "Result dir:  ${RESULT_DIR}"
echo "Sequence:    ${SEQ}"
echo "Center frame: ${CENTER}"
echo "Window size: ${WINDOW}"
echo "Depth model: ${DEPTH_MODEL}"
echo "Cache stage: ${CACHE_STAGE}"
echo "Port:        ${PORT}"
echo "============================================"

python lib/scripts/viser_oar_window_debug.py \
    --result-dir "${RESULT_DIR}" \
    --sequence "${SEQ}" \
    --center-frame "${CENTER}" \
    --window-size "${WINDOW}" \
    --cache-stage "${CACHE_STAGE}" \
    --depth-model "${DEPTH_MODEL}" \
    --pc-subsample "${PC_SUBSAMPLE}" \
    --port "${PORT}" \
    --device "${DEVICE}"
