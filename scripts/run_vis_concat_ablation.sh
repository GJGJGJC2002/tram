#!/bin/bash
# ================================================================
# 消融实验可视化 —— 横向拼接 OAR debug 图片
#
# 将每个序列中的 Input 原始图、OAR 各阶段 debug 图横向拼接，
# 方便对比消融效果。输出目录与源 debug 文件夹并列。
#
# 用法:
#   bash scripts/run_concat_ablation_vis.sh                         # 默认参数
#   SEQ="09_outdoor_walk 20_outdoor_walk" bash scripts/run_concat_ablation_vis.sh  # 指定序列
#   TARGET_HEIGHT=360 bash scripts/run_concat_ablation_vis.sh       # 指定缩放高度
# ================================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# ========== 可通过环境变量覆盖的参数 ==========

# 结果根目录
RESULT_DIR="${RESULT_DIR:-results/promptbase_video_depthrefine_emdb2}"

# EMDB 数据集根目录（用于读取 Input 原始图片）
DATASET_ROOT="${DATASET_ROOT:-datasets/EMDB}"

# 要拼接的 debug 子文件夹（从左到右排列，不含 Input）
SOURCE_DIRS="${SOURCE_DIRS:-oar_depth_contact_debug oar_window_pre oar_window_post_pre_dep oar_window_post oar_reproj_error}"

# 各列标签（包含 Input，数量 = source_dirs + 1）
LABELS="${LABELS:-Input Depth+Contact Pre-OAR Post-PreDep Post-OAR Reproj-Error}"

# 输出子目录名称（在每个 seq_dir 下创建，与 debug 文件夹并列）
OUTPUT_DIR="${OUTPUT_DIR:-oar_ablation_concat}"

# 指定序列（空格分隔），留空则处理所有序列
SEQ="${SEQ:-}"

# 统一缩放的目标高度（像素），留空则保持原始高度
TARGET_HEIGHT="${TARGET_HEIGHT:-480}"

# 图片间分割线宽度（像素），0 表示无分割线
SEPARATOR_WIDTH="${SEPARATOR_WIDTH:-2}"

# 是否隐藏标签栏（1=隐藏, 0=显示）
NO_LABELS="${NO_LABELS:-0}"

# 标签栏高度（像素）
LABEL_BAR_HEIGHT="${LABEL_BAR_HEIGHT:-36}"

# 帧采样步长（仅处理帧号为该值倍数的帧），留空则处理所有帧
FRAME_STRIDE="${FRAME_STRIDE:-}"

# 并行进程数
NUM_WORKERS="${NUM_WORKERS:-8}"

# ========== 打印配置 ==========
echo "=========================================="
echo "消融实验可视化 —— 横向拼接"
echo "=========================================="
echo "Result dir:    ${RESULT_DIR}"
echo "Dataset root:  ${DATASET_ROOT}"
echo "Source dirs:   ${SOURCE_DIRS}"
echo "Labels:        ${LABELS}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Sequences:     ${SEQ:-ALL}"
echo "Target height: ${TARGET_HEIGHT:-original}"
echo "Separator:     ${SEPARATOR_WIDTH}px"
echo "No labels:     ${NO_LABELS}"
echo "Label bar:     ${LABEL_BAR_HEIGHT}px"
echo "Frame stride:  ${FRAME_STRIDE:-all}"
echo "Workers:       ${NUM_WORKERS}"
echo "=========================================="

# ========== 构建命令 ==========
CMD="python lib/scripts/concat_ablation_vis.py \
    --result-dir ${RESULT_DIR} \
    --dataset-root ${DATASET_ROOT} \
    --source-dirs ${SOURCE_DIRS} \
    --labels ${LABELS} \
    --output-dir ${OUTPUT_DIR} \
    --separator-width ${SEPARATOR_WIDTH} \
    --label-bar-height ${LABEL_BAR_HEIGHT} \
    --num-workers ${NUM_WORKERS}"

# 可选参数
if [ -n "${TARGET_HEIGHT}" ]; then
    CMD="${CMD} --target-height ${TARGET_HEIGHT}"
fi

if [ -n "${SEQ}" ]; then
    CMD="${CMD} --sequences ${SEQ}"
fi

if [ -n "${FRAME_STRIDE}" ]; then
    CMD="${CMD} --frame-stride ${FRAME_STRIDE}"
fi

if [ "${NO_LABELS}" = "1" ]; then
    CMD="${CMD} --no-labels"
fi

# ========== 执行 ==========
echo ""
echo "Running: ${CMD}"
echo ""
eval ${CMD}
