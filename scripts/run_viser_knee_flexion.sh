#!/bin/bash
# ================================================================
# WHAM Figure 4 风格可视化：左膝屈曲角度对比 + Root-Aligned 下半身 Mesh
#
# 生成包含三行的静态图像：
#   1. 关键帧原始图像序列（标注 t1, t2）
#   2. 左膝屈曲角度曲线（GT vs 各方法）
#   3. t1, t2 时刻各方法 root-aligned 下半身 SMPL mesh 侧视图
#
# 每个方法格式为 "Label:path"，path 可以是绝对路径或相对路径
#
# 用法:
#   bash scripts/run_viser_knee_flexion.sh                         # 默认参数
#   SEQ=55_outdoor_walk bash scripts/run_viser_knee_flexion.sh     # 指定序列
#   T1=50 T2=200 bash scripts/run_viser_knee_flexion.sh            # 指定关键帧
# ================================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# ---------- 可配置参数 ----------
SEQ="${SEQ:-29_outdoor_stairs_up}"
T1="${T1:-200}"
T2="${T2:-400}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:-500}"
NUM_IMAGE_FRAMES="${NUM_IMAGE_FRAMES:-5}"
DATASET_PATH="${DATASET_PATH:-datasets/EMDB}"
OUTPUT="${OUTPUT:-figures/knee_flexion_${SEQ}.png}"
DEVICE="${DEVICE:-cuda}"

# 方法列表（格式: "Label:result_dir"，每个方法可指定独立的绝对/相对路径）
METHODS=(
    "${M1:-Ours:results/gvhmr_base_warmstart}"
    "${M2:-PromptHMR:results/promptbase_video_warmstart_emdb2}"
    "${M3:-GVHMR:/mnt/storage/users/jcge_data/Work/2026-1/Projects/tram/results/gvhmr_base}"
    "${M4:-TRAM:/mnt/storage/users/jcge_data/Work/2026-1/Projects/tram/results/emdb_basic}"
)

# 过滤空项
METHODS_FILTERED=()
for m in "${METHODS[@]}"; do
    [[ -n "$m" ]] && METHODS_FILTERED+=("$m")
done
# ------------------------------------

echo "=========================================="
echo "左膝屈曲角度对比可视化 (WHAM Fig.4 style)"
echo "=========================================="
echo "序列:       $SEQ"
echo "方法:"
for m in "${METHODS_FILTERED[@]}"; do
    echo "            $m"
done
echo "关键帧:     t1=$T1, t2=$T2"
echo "帧范围:     [$FRAME_START, $FRAME_END]"
echo "图像帧数:   $NUM_IMAGE_FRAMES"
echo "输出:       $OUTPUT"
echo "=========================================="

python lib/scripts/visualize_knee_flexion.py \
    --seq $SEQ \
    --methods "${METHODS_FILTERED[@]}" \
    --t1 $T1 \
    --t2 $T2 \
    --frame_range $FRAME_START $FRAME_END \
    --num_image_frames $NUM_IMAGE_FRAMES \
    --dataset_path $DATASET_PATH \
    --output $OUTPUT \
    --device $DEVICE
