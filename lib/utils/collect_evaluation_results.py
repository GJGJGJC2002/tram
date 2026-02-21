#!/usr/bin/env python3
"""
收集评估结果脚本

从指定输出目录的所有序列子目录中收集 metrics.json，
汇总生成 evaluation_results.xlsx 和 summary.xlsx

使用方法：
    python scripts/collect_evaluation_results.py --output_dir results/emdb_adjacent_render_tram
    python scripts/collect_evaluation_results.py --output_dir results/emdb_adjacent_render_tram --output evaluation_results_new.xlsx
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


def collect_metrics_from_directory(output_dir: str) -> list:
    """
    从输出目录收集所有序列的 metrics.json
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        包含所有序列指标的列表，每个元素为 {'seq': seq_name, 'metric1': value1, ...}
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    all_results = []
    found_seqs = []
    missing_seqs = []
    
    # 遍历所有子目录
    for seq_dir in sorted(output_path.iterdir()):
        if not seq_dir.is_dir():
            continue
        
        # 跳过特殊目录
        if seq_dir.name in ['intermediate', 'visualization']:
            continue
        
        seq_name = seq_dir.name
        metrics_file = seq_dir / 'metrics.json'
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # 添加序列名称
                result = {'seq': seq_name}
                result.update(metrics)
                all_results.append(result)
                found_seqs.append(seq_name)
                
            except Exception as e:
                print(f"⚠️  警告：无法读取 {metrics_file}: {e}")
                missing_seqs.append(seq_name)
        else:
            missing_seqs.append(seq_name)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"收集到 {len(all_results)} 个序列的评估结果")
    print(f"{'='*60}")
    
    if found_seqs:
        print(f"\n✅ 成功收集的序列 ({len(found_seqs)}):")
        for seq in found_seqs:
            print(f"   - {seq}")
    
    if missing_seqs:
        print(f"\n⚠️  缺失 metrics.json 的序列 ({len(missing_seqs)}):")
        for seq in missing_seqs:
            print(f"   - {seq}")
    
    return all_results


def compute_summary_statistics(all_results: list) -> dict:
    """
    计算汇总统计信息（平均值）
    
    Args:
        all_results: 所有序列的结果列表
        
    Returns:
        包含平均值的字典
    """
    if not all_results:
        return {}
    
    # 收集所有指标的值
    metrics_values = defaultdict(list)
    for result in all_results:
        for key, value in result.items():
            if key != 'seq' and isinstance(value, (int, float)):
                metrics_values[key].append(value)
    
    # 计算平均值
    summary = {}
    for metric, values in metrics_values.items():
        summary[metric] = np.mean(values)
    
    return summary


def save_results(all_results: list, output_dir: str, 
                 results_filename: str = "evaluation_results.xlsx",
                 summary_filename: str = "summary.xlsx"):
    """
    保存结果到 Excel 文件
    
    Args:
        all_results: 所有序列的结果列表
        output_dir: 输出目录
        results_filename: 评估结果文件名
        summary_filename: 汇总文件名
    """
    output_path = Path(output_dir)
    
    if not all_results:
        print("\n❌ 没有找到任何评估结果，无法生成文件")
        return
    
    # 保存详细结果
    df_results = pd.DataFrame(all_results)
    results_file = output_path / results_filename
    df_results.to_excel(results_file, index=False)
    print(f"\n✅ 详细结果已保存到: {results_file}")
    
    # 计算并保存汇总
    summary = compute_summary_statistics(all_results)
    if summary:
        df_summary = pd.DataFrame([summary])
        summary_file = output_path / summary_filename
        df_summary.to_excel(summary_file, index=False)
        print(f"✅ 汇总结果已保存到: {summary_file}")
        
        # 打印汇总统计
        print(f"\n{'='*60}")
        print("平均指标:")
        print(f"{'='*60}")
        for metric, value in sorted(summary.items()):
            print(f"  {metric:20s}: {value:.4f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='收集多 GPU 运行后的评估结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认收集并覆盖现有文件
  python scripts/collect_evaluation_results.py --output_dir results/emdb_adjacent_render_tram
  
  # 指定输出文件名
  python scripts/collect_evaluation_results.py --output_dir results/emdb_adjacent_render_tram \\
      --output evaluation_results_collected.xlsx
        """
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='Pipeline 输出目录路径'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='evaluation_results.xlsx',
        help='评估结果输出文件名（默认：evaluation_results.xlsx）'
    )
    parser.add_argument(
        '--summary', 
        type=str, 
        default='summary.xlsx',
        help='汇总结果输出文件名（默认：summary.xlsx）'
    )
    
    args = parser.parse_args()
    
    print(f"\n开始收集评估结果...")
    print(f"输出目录: {args.output_dir}")
    
    try:
        # 收集结果
        all_results = collect_metrics_from_directory(args.output_dir)
        
        # 保存结果
        save_results(
            all_results, 
            args.output_dir,
            results_filename=args.output,
            summary_filename=args.summary
        )
        
        print("\n✅ 收集完成！")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
