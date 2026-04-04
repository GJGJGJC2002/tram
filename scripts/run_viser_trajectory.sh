#!/bin/bash
# ================================================================
# 全局轨迹 2D/3D 对比可视化
#
# 上下两行：3D 轨迹 + 2D 鸟瞰 (XZ)，对齐到 GT
#
# 用法:
#   bash scripts/run_viser_trajectory.sh                    # 所有序列
#   SEQ=09_outdoor_walk bash scripts/run_viser_trajectory.sh # 单个序列
# ================================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# ---------- 可配置参数 ----------
SEQ="${SEQ:-}"
DATASET_PATH="${DATASET_PATH:-datasets/EMDB}"
OUTPUT_DIR="${OUTPUT_DIR:-figures/trajectory}"

# 方法列表（格式: "Label:result_dir"）
METHODS=(
    "${M1:-Ours:results/gvhmr_base_warmstart}"
    "${M2:-PromptHMR:results/promptbase_video_depthrefine_emdb2_woMask}"
    "${M3:-}"
    "${M4:-}"
)

# 过滤空项
METHODS_FILTERED=()
for m in "${METHODS[@]}"; do
    [[ -n "$m" ]] && METHODS_FILTERED+=("$m")
done

# ---------- 构建序列列表 ----------
if [[ -n "$SEQ" ]]; then
    # 单个序列
    SEQS=("$SEQ")
else
    # 自动发现所有序列：取各方法结果目录下都存在 smpl.npz 的序列
    # 先收集第一个方法目录下的所有序列
    FIRST_DIR=""
    for m in "${METHODS_FILTERED[@]}"; do
        FIRST_DIR="${m#*:}"
        break
    done
    if [[ -z "$FIRST_DIR" ]]; then
        echo "ERROR: 没有指定任何方法" >&2
        exit 1
    fi

    SEQS=()
    for seq_dir in "$FIRST_DIR"/*/; do
        [[ ! -d "$seq_dir" ]] && continue
        seq_name="$(basename "$seq_dir")"
        # 检查所有方法都有该序列的结果
        all_exist=true
        for m in "${METHODS_FILTERED[@]}"; do
            mdir="${m#*:}"
            if [[ ! -f "$mdir/$seq_name/smpl.npz" ]]; then
                all_exist=false
                break
            fi
        done
        if $all_exist; then
            SEQS+=("$seq_name")
        fi
    done

    # 排序
    IFS=$'\n' SEQS=($(sort <<<"${SEQS[*]}")); unset IFS
fi

# ---------- 运行 ----------
echo "=========================================="
echo "全局轨迹对比可视化"
echo "=========================================="
echo "方法:"
for m in "${METHODS_FILTERED[@]}"; do
    echo "  $m"
done
echo "序列数: ${#SEQS[@]}"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

TOTAL=${#SEQS[@]}
COUNT=0
FAIL=0
for seq in "${SEQS[@]}"; do
    COUNT=$((COUNT + 1))
    OUTPUT="$OUTPUT_DIR/${seq}.svg"
    echo ""
    echo "[$COUNT/$TOTAL] $seq -> $OUTPUT"

    python lib/scripts/visualize_trajectory.py \
        --seq "$seq" \
        --methods "${METHODS_FILTERED[@]}" \
        --dataset_path "$DATASET_PATH" \
        --output "$OUTPUT"

    if [[ $? -ne 0 ]]; then
        echo "  FAILED: $seq"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=========================================="
echo "完成: $((TOTAL - FAIL))/$TOTAL 成功"
[[ $FAIL -gt 0 ]] && echo "失败: $FAIL 个序列"
echo "=========================================="
