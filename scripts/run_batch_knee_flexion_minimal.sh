#!/bin/bash
# ================================================================
# 批量生成 EMDB 所有序列的纯净模式膝关节角度曲线图
#
# 功能：
#   1. 自动扫描 results 目录下所有有效序列
#   2. 为每个序列生成 minimal 模式的膝关节角度曲线
#   3. 自动确定帧范围（FRAME_END=-1 使用全序列）
#   4. 智能设置 T1/T2（分别位于 1/3 和 2/3 位置）
#   5. 统一输出到 figures/knee_flexion_minimal/ 目录
#
# 用法:
#   bash scripts/run_batch_knee_flexion_minimal.sh                        # 使用默认配置
#   RESULT_DIR=results/other bash scripts/run_batch_knee_flexion_minimal.sh
#   DRY_RUN=1 bash scripts/run_batch_knee_flexion_minimal.sh              # 只打印不执行
#   PARALLEL=1 bash scripts/run_batch_knee_flexion_minimal.sh             # 后台并行（慎用）
# ================================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# ---------- 可配置参数 ----------
RESULT_DIR="${RESULT_DIR:-results/gvhmr_base_warmstart}"
DATASET_PATH="${DATASET_PATH:-datasets/EMDB}"
OUTPUT_DIR="${OUTPUT_DIR:-figures/knee_flexion_minimal}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"      # 1=只打印不执行
PARALLEL="${PARALLEL:-0}"    # 1=后台并行（GPU 内存足够时使用）
MAX_PARALLEL="${MAX_PARALLEL:-2}"  # 最大并行数

# 方法列表（与单个脚本保持一致）
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

# Python 脚本：快速读取 annots.pkl 获取帧数
read -r -d '' PY_GET_NFRAMES << 'EOF'
import sys, pickle
try:
    with open(sys.argv[1], 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    n_frames = data.get('n_frames', len(data.get('poses', [])))
    print(n_frames)
except:
    print(0)
EOF

echo "=========================================="
echo "批量生成 EMDB 膝关节角度曲线（纯净模式）"
echo "=========================================="
echo "结果目录:   $RESULT_DIR"
echo "输出目录:   $OUTPUT_DIR"
if [ "$PARALLEL" = "1" ]; then
    echo "并行模式:   启用 (max=$MAX_PARALLEL)"
fi
echo "方法:"
for m in "${METHODS_FILTERED[@]}"; do
    echo "            $m"
done
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 扫描所有序列
SEQUENCES=($(ls -1 "$RESULT_DIR" | grep -E "^[0-9]+_" | sort -n))
TOTAL=${#SEQUENCES[@]}

if [ $TOTAL -eq 0 ]; then
    echo "ERROR: No sequences found in $RESULT_DIR"
    exit 1
fi

echo "找到 $TOTAL 个序列"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# 并行任务控制
PIDS=()

process_sequence() {
    local SEQ=$1
    local IDX=$2
    
    echo "[$IDX/$TOTAL] 处理序列: $SEQ"
    
    # 检查结果是否存在
    if [ ! -f "$RESULT_DIR/$SEQ/smpl.npz" ] && [ ! -f "$RESULT_DIR/$SEQ/smpl.pkl" ]; then
        echo "  ⚠ 跳过: 缺少 smpl 结果"
        return 2
    fi
    
    # 使用 Python 脚本查找 GT 数据（沿用 visualize_knee_flexion.py 的逻辑）
    local PY_FIND_SEQ="
import sys, os
from glob import glob
dataset_path, seq_name = sys.argv[1], sys.argv[2]
for person_dir in sorted(glob(os.path.join(dataset_path, 'P*'))):
    seq_dir = os.path.join(person_dir, seq_name)
    if os.path.isdir(seq_dir):
        person = os.path.basename(person_dir)
        ann_file = os.path.join(seq_dir, f'{person}_{seq_name}_data.pkl')
        if os.path.exists(ann_file):
            print(ann_file)
            sys.exit(0)
"
    
    local ANN_FILE=$(python -c "$PY_FIND_SEQ" "$DATASET_PATH" "$SEQ" 2>/dev/null)
    if [ -z "$ANN_FILE" ]; then
        echo "  ⚠ 跳过: 未找到 GT 数据"
        return 2
    fi
    
    # 读取总帧数
    local N_FRAMES=$(python -c "$PY_GET_NFRAMES" "$ANN_FILE" 2>/dev/null)
    if [ -z "$N_FRAMES" ] || [ "$N_FRAMES" -le 0 ]; then
        echo "  ⚠ 跳过: 无法读取帧数"
        return 2
    fi
    
    # 智能设置 T1, T2（1/3 和 2/3 位置）
    local T1=$((N_FRAMES / 3))
    local T2=$((N_FRAMES * 2 / 3))
    
    local OUTPUT="$OUTPUT_DIR/${SEQ}_minimal.png"
    
    # 构建命令
    local CMD="MINIMAL=1 FRAME_START=0 FRAME_END=-1 SEQ=$SEQ T1=$T1 T2=$T2 OUTPUT=$OUTPUT DEVICE=$DEVICE bash scripts/run_viser_knee_flexion.sh"
    
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] T1=$T1 T2=$T2 N=$N_FRAMES"
        return 0
    else
        # 执行
        local LOG_FILE="$OUTPUT_DIR/${SEQ}_minimal.log"
        if eval "$CMD" > "$LOG_FILE" 2>&1; then
            echo "  ✓ 成功: $OUTPUT (T1=$T1, T2=$T2, N=$N_FRAMES)"
            rm -f "$LOG_FILE"  # 成功则删除日志
            return 0
        else
            echo "  ✗ 失败 (详见 $LOG_FILE)"
            return 1
        fi
    fi
}

# 顺序或并行处理
for i in "${!SEQUENCES[@]}"; do
    SEQ="${SEQUENCES[$i]}"
    IDX=$((i + 1))
    
    if [ "$PARALLEL" = "1" ]; then
        # 后台并行
        process_sequence "$SEQ" "$IDX" &
        PIDS+=($!)
        
        # 控制并行数
        if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
            wait ${PIDS[0]}
            RET=$?
            [ $RET -eq 0 ] && ((SUCCESS_COUNT++)) || [ $RET -eq 1 ] && ((FAIL_COUNT++)) || ((SKIP_COUNT++))
            PIDS=("${PIDS[@]:1}")
        fi
    else
        # 顺序执行
        process_sequence "$SEQ" "$IDX"
        RET=$?
        [ $RET -eq 0 ] && ((SUCCESS_COUNT++)) || [ $RET -eq 1 ] && ((FAIL_COUNT++)) || ((SKIP_COUNT++))
        echo ""
    fi
done

# 等待剩余并行任务
if [ "$PARALLEL" = "1" ]; then
    for PID in "${PIDS[@]}"; do
        wait $PID
        RET=$?
        [ $RET -eq 0 ] && ((SUCCESS_COUNT++)) || [ $RET -eq 1 ] && ((FAIL_COUNT++)) || ((SKIP_COUNT++))
    done
fi

echo ""
echo "=========================================="
echo "批量处理完成"
echo "=========================================="
echo "总序列数:   $TOTAL"
echo "成功:       $SUCCESS_COUNT"
echo "失败:       $FAIL_COUNT"
echo "跳过:       $SKIP_COUNT"
echo "输出目录:   $OUTPUT_DIR"
echo "=========================================="
