#!/bin/bash

# 收集评估结果 - 快捷脚本
# 用于多 GPU 运行后重新汇总所有序列的评估结果
#
# 使用方法：
#   bash scripts/collect_results.sh results/emdb_adjacent_render_tram
#   bash scripts/collect_results.sh results/emdb_basic


OUTPUT_DIR=results/promptbase_video_depthrefine_emdb1

echo "从 ${OUTPUT_DIR} 收集评估结果..."
python lib/utils/collect_evaluation_results.py --output_dir ${OUTPUT_DIR}
