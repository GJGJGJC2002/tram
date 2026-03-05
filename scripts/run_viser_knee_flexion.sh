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
#   FRAME_END=-1 bash scripts/run_viser_knee_flexion.sh            # 使用数据最后一帧
#   MINIMAL=1 bash scripts/run_viser_knee_flexion.sh               # 纯净模式（只显示曲线）
#   BATCH=1 bash scripts/run_viser_knee_flexion.sh                 # 批量处理所有EMDB2序列
# ================================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate alignHMR

# ---------- 可配置参数 ----------
BATCH="${BATCH:-0}"  # 0=单个序列, 1=批量处理所有序列
SEQ="${SEQ:-29_outdoor_stairs_up}"
T1="${T1:-823}"
T2="${T2:-858}"
FRAME_START="${FRAME_START:-800}"
FRAME_END="${FRAME_END:-900}"
NUM_IMAGE_FRAMES="${NUM_IMAGE_FRAMES:-5}"
DATASET_PATH="${DATASET_PATH:-datasets/EMDB}"
OUTPUT="${OUTPUT:-figures/knee_flexion/knee_flexion_${SEQ}.svg}"
OUTPUT_DIR="${OUTPUT_DIR:-figures/knee_flexion}"
DEVICE="${DEVICE:-cuda}"
MINIMAL="${MINIMAL:-0}"  # 0=完整模式, 1=纯净模式
RESULT_DIR="${RESULT_DIR:-results/gvhmr_base_warmstart}"  # 用于批量模式扫描序列

# 方法列表（格式: "Label:result_dir"，每个方法可指定独立的绝对/相对路径）
METHODS=(
    "${M1:-Ours:results/gvhmr_base_warmstart}"
    "${M2:-PromptHMR:results/promptbase_video_warmstart_emdb2}"
    # "${M3:-GVHMR:/mnt/storage/users/jcge_data/Work/2026-1/Projects/tram/results/gvhmr_base}"
    "${M4:-TRAM:/mnt/storage/users/jcge_data/Work/2026-1/Projects/tram/results/emdb_basic}"
)

# 过滤空项
METHODS_FILTERED=()
for m in "${METHODS[@]}"; do
    [[ -n "$m" ]] && METHODS_FILTERED+=("$m")
done
# ------------------------------------

# Python 脚本：读取帧数和查找GT文件
read -r -d '' PY_UTILS << 'EOF'
import sys, pickle, os
from glob import glob

if sys.argv[1] == "get_nframes":
    # 读取帧数
    try:
        with open(sys.argv[2], 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        n_frames = data.get('n_frames', len(data.get('poses', [])))
        print(n_frames)
    except:
        print(0)
elif sys.argv[1] == "find_seq":
    # 查找序列GT文件
    dataset_path, seq_name = sys.argv[2], sys.argv[3]
    for person_dir in sorted(glob(os.path.join(dataset_path, 'P*'))):
        seq_dir = os.path.join(person_dir, seq_name)
        if os.path.isdir(seq_dir):
            person = os.path.basename(person_dir)
            ann_file = os.path.join(seq_dir, f'{person}_{seq_name}_data.pkl')
            if os.path.exists(ann_file):
                print(ann_file)
                sys.exit(0)
EOF

# 单个序列处理函数
process_single_sequence() {
    local SEQ=$1
    local T1=$2
    local T2=$3
    local OUTPUT=$4
    local SHOW_HEADER=${5:-1}
    
    if [ "$SHOW_HEADER" = "1" ]; then
        echo "=========================================="
        echo "左膝屈曲角度对比可视化 (WHAM Fig.4 style)"
        echo "=========================================="
        echo "序列:       $SEQ"
        if [ "$MINIMAL" = "1" ]; then
            echo "模式:       纯净模式（只显示曲线）"
        else
            echo "模式:       完整模式（图像+曲线+mesh）"
        fi
        echo "方法:"
        for m in "${METHODS_FILTERED[@]}"; do
            echo "            $m"
        done
        echo "关键帧:     t1=$T1, t2=$T2"
        echo "帧范围:     [$FRAME_START, $FRAME_END]"
        if [ "$MINIMAL" != "1" ]; then
            echo "图像帧数:   $NUM_IMAGE_FRAMES"
        fi
        echo "输出:       $OUTPUT"
        echo "=========================================="
    fi
    
    # 构建 Python 命令
    local CMD="python lib/scripts/visualize_knee_flexion.py \
        --seq $SEQ \
        --methods ${METHODS_FILTERED[@]} \
        --t1 $T1 \
        --t2 $T2 \
        --frame_range $FRAME_START $FRAME_END \
        --dataset_path $DATASET_PATH \
        --output $OUTPUT \
        --device $DEVICE"
    
    # 完整模式需要额外参数
    if [ "$MINIMAL" != "1" ]; then
        CMD="$CMD --num_image_frames $NUM_IMAGE_FRAMES"
    else
        CMD="$CMD --minimal"
    fi
    
    eval $CMD
    return $?
}

# ================================================================
# 主逻辑：单个序列 vs 批量处理
# ================================================================

if [ "$BATCH" = "1" ]; then
    # ============================================================
    # 批量模式
    # ============================================================
    echo "=========================================="
    echo "批量生成 EMDB2 膝关节角度曲线"
    echo "=========================================="
    echo "结果目录:   $RESULT_DIR"
    echo "输出目录:   $OUTPUT_DIR"
    if [ "$MINIMAL" = "1" ]; then
        echo "模式:       纯净模式（只显示曲线）"
    else
        echo "模式:       完整模式（图像+曲线+mesh）"
    fi
    echo "方法:"
    for m in "${METHODS_FILTERED[@]}"; do
        echo "            $m"
    done
    echo "=========================================="
    echo ""
    
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
    
    for i in "${!SEQUENCES[@]}"; do
        SEQ="${SEQUENCES[$i]}"
        IDX=$((i + 1))
        
        echo "[$IDX/$TOTAL] 处理序列: $SEQ"
        
        # 检查结果是否存在
        if [ ! -f "$RESULT_DIR/$SEQ/smpl.npz" ] && [ ! -f "$RESULT_DIR/$SEQ/smpl.pkl" ]; then
            echo "  ⚠ 跳过: 缺少 smpl 结果"
            ((SKIP_COUNT++))
            echo ""
            continue
        fi
        
        # 查找GT文件
        ANN_FILE=$(python -c "$PY_UTILS" "find_seq" "$DATASET_PATH" "$SEQ" 2>/dev/null)
        if [ -z "$ANN_FILE" ]; then
            echo "  ⚠ 跳过: 未找到 GT 数据"
            ((SKIP_COUNT++))
            echo ""
            continue
        fi
        
        # 读取总帧数
        N_FRAMES=$(python -c "$PY_UTILS" "get_nframes" "$ANN_FILE" 2>/dev/null)
        if [ -z "$N_FRAMES" ] || [ "$N_FRAMES" -le 0 ]; then
            echo "  ⚠ 跳过: 无法读取帧数"
            ((SKIP_COUNT++))
            echo ""
            continue
        fi
        
        # 智能设置 T1, T2（1/3 和 2/3 位置）
        T1_AUTO=$((N_FRAMES / 3))
        T2_AUTO=$((N_FRAMES * 2 / 3))
        
        # 输出文件
        if [ "$MINIMAL" = "1" ]; then
            OUTPUT_FILE="$OUTPUT_DIR/${SEQ}_minimal.png"
        else
            OUTPUT_FILE="$OUTPUT_DIR/knee_flexion_${SEQ}.svg"
        fi
        
        # 执行
        LOG_FILE="$OUTPUT_DIR/${SEQ}.log"
        if process_single_sequence "$SEQ" "$T1_AUTO" "$T2_AUTO" "$OUTPUT_FILE" 0 > "$LOG_FILE" 2>&1; then
            echo "  ✓ 成功: $OUTPUT_FILE (T1=$T1_AUTO, T2=$T2_AUTO, N=$N_FRAMES)"
            rm -f "$LOG_FILE"
            ((SUCCESS_COUNT++))
        else
            echo "  ✗ 失败 (详见 $LOG_FILE)"
            ((FAIL_COUNT++))
        fi
        
        echo ""
    done
    
    echo "=========================================="
    echo "批量处理完成"
    echo "=========================================="
    echo "总序列数:   $TOTAL"
    echo "成功:       $SUCCESS_COUNT"
    echo "失败:       $FAIL_COUNT"
    echo "跳过:       $SKIP_COUNT"
    echo "输出目录:   $OUTPUT_DIR"
    echo "=========================================="
    
else
    # ============================================================
    # 单个序列模式
    # ============================================================
    process_single_sequence "$SEQ" "$T1" "$T2" "$OUTPUT" 1
fi
