#!/usr/bin/env python3
"""
Pipeline 运行脚本

使用新的模块化 Pipeline 架构运行 EMDB 评估。

Usage:
    # 使用配置文件运行
    python scripts/run_pipeline.py --config configs/pipelines/emdb_basic.yaml --seq 09_outdoor_walk --person P0

    # 使用预设的 Pipeline 运行
    python scripts/run_pipeline.py --preset basic --seq 09_outdoor_walk --person P0 --split 2

    # 处理整个 split
    python scripts/run_pipeline.py --preset basic --split 2
"""

import sys
import os
# 文件位于 lib/scripts/，需要将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import argparse
import logging
import pickle as pkl
from glob import glob
from datetime import datetime
from collections import defaultdict


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    # ANSI 颜色代码
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'BLUE': '\033[34m',
        'CYAN': '\033[36m',
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'RED': '\033[31m',
        'MAGENTA': '\033[35m',
    }

    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        # 只对终端输出添加颜色
        if hasattr(record, 'levelname'):
            levelname = record.levelname
            if levelname == 'INFO':
                level_color = self.COLORS['GREEN']
            elif levelname == 'WARNING':
                level_color = self.COLORS['YELLOW']
            elif levelname == 'ERROR':
                level_color = self.COLORS['RED']
            elif levelname == 'DEBUG':
                level_color = self.COLORS['CYAN']
            else:
                level_color = self.COLORS['RESET']
            record.levelname = f"{level_color}{levelname}{self.COLORS['RESET']}"

        # 格式化消息
        message = super().format(record)

        # 为文件名和行号添加颜色和加粗
        # 匹配格式: [filename.py:lineno]
        import re
        pattern = r'\[([^\]]+\.py:\d+)\]'
        replacement = f"[{self.COLORS['BOLD']}{self.COLORS['BLUE']}\\1{self.COLORS['RESET']}]"
        message = re.sub(pattern, replacement, message)

        return message


class PlainFormatter(logging.Formatter):
    """文件日志格式化器（无颜色）"""
    pass

import numpy as np
import pandas as pd
import torch

from lib.pipeline import (
    PipelineBuilder,
    PipelineData,
    hooks,
)


def setup_logging(output_dir: str):
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'pipeline.log')

    # 创建根 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除已有的 handlers

    # 日志格式
    log_format = '%(asctime)s - %(name)s [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s'

    # 文件 handler（无颜色）
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(PlainFormatter(log_format))
    logger.addHandler(file_handler)

    # 终端 handler（带颜色）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(log_format))
    logger.addHandler(console_handler)

    return logging.getLogger(__name__)


def load_emdb_sequences(dataset_path: str, split: int, seq_name: str = None, person: str = None):
    """
    加载 EMDB 数据集序列
    
    Args:
        dataset_path: EMDB 数据集根目录
        split: split 编号 (1, 2, 或 3)
        seq_name: 指定序列名称
        person: 指定人物编号
    
    Returns:
        list: 符合条件的序列路径列表
    """
    roots = []
    
    if person is not None:
        person_ids = [int(person[1:])] if person.startswith('P') else [int(person)]
    else:
        person_ids = range(10)
    
    for p in person_ids:
        folder = f'{dataset_path}/P{p}'
        if not os.path.exists(folder):
            continue
        root = sorted(glob(f'{folder}/*'))
        roots.extend(root)
    
    emdb = []
    for root in roots:
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        if not os.path.exists(annfile):
            continue
        
        ann = pkl.load(open(annfile, 'rb'))
        if ann.get(f'emdb{split}', False):
            if seq_name is None or root.split('/')[-1] == seq_name:
                emdb.append(root)
    
    return emdb


def load_sequence_data(root: str) -> PipelineData:
    """
    加载序列数据到 PipelineData
    
    Args:
        root: 序列根目录
    
    Returns:
        PipelineData 实例
    """
    seq_name = root.split('/')[-1]
    
    # 加载图像路径
    img_folder = f'{root}/images'
    image_paths = sorted(glob(f'{img_folder}/*.jpg'))
    
    if len(image_paths) == 0:
        raise ValueError(f'No images found in {img_folder}')
    
    # 加载标注
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    annotations = pkl.load(open(annfile, 'rb'))
    
    # 创建 PipelineData
    data = PipelineData(
        sequence_name=seq_name,
        sequence_path=root,
        image_paths=image_paths,
        annotations=annotations,
        valid_frames_mask=annotations.get('good_frames_mask'),
        bboxes=annotations['bboxes']['bboxes'],  # GT bboxes
    )
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Run EMDB evaluation pipeline')
    
    # Pipeline 配置
    parser.add_argument('--config', type=str, default=None,
                       help='Path to pipeline config file (YAML)')

    
    # 数据集参数
    parser.add_argument('--seq', type=str, default=None,
                       help='Specific sequence to process')
    parser.add_argument('--person', type=str, default=None,
                       help='Specific person folder')
    parser.add_argument('--split', type=int, default=2,
                       help='EMDB split number')
    parser.add_argument('--dataset_path', type=str, 
                       default='/home/gejunchen/Work/2026-1/Datasets/EMDB',
                       help='Path to EMDB dataset')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results/pipeline',
                       help='Output directory')
    
    # Pipeline 参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--hpe_mode', type=str, default='accurate',
                       choices=['accurate', 'efficient'],
                       help='HPE estimation mode')
    
    # 调试选项
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug hooks')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization hooks')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.output_dir)
    logger.info('='*80)
    logger.info('EMDB Pipeline Runner')
    logger.info('='*80)
    logger.info(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Arguments: {vars(args)}')
    
    # 创建 Pipeline
    if args.config:
        logger.info(f"Loading pipeline from config: {args.config}")
        pipeline = PipelineBuilder.from_config(args.config)
    else:
        logger.info(f"Creating Evaluation pipline")
        pipeline = PipelineBuilder.create_evaluation_pipeline(
            name="emdb_eval",
            device=args.device,
            output_dir=args.output_dir
        )
    
    # 配置更新
    pipeline.config['output_dir'] = args.output_dir
    
    # 添加调试 hooks
    if args.debug:
        for stage, hook_list in hooks.DEBUG_HOOKS.items():
            for hook_fn in hook_list:
                pipeline.add_hook(stage, hook_fn)
    
    # 添加可视化 hooks
    if args.visualize:
        for stage, hook_list in hooks.VISUALIZATION_HOOKS.items():
            for hook_fn in hook_list:
                pipeline.add_hook(stage, hook_fn)
    
    # 加载序列
    logger.info('Loading EMDB sequences...')
    sequences = load_emdb_sequences(args.dataset_path, args.split, args.seq, args.person)
    
    if len(sequences) == 0:
        logger.error('No sequences found matching the criteria!')
        return
    
    logger.info(f'Found {len(sequences)} sequence(s) to process:')
    for seq_path in sequences:
        logger.info(f'  - {seq_path}')
    
    # 初始化 Pipeline
    logger.info('Setting up pipeline...')
    pipeline.setup()
    
    # 处理每个序列
    all_results = []
    
    for i, root in enumerate(sequences):
        seq_name = root.split('/')[-1]
        logger.info('')
        logger.info('='*80)
        logger.info(f'Processing sequence {i+1}/{len(sequences)}: {seq_name}')
        logger.info('='*80)
        
        try:
            # 加载数据
            data = load_sequence_data(root)
            data.metadata['output_dir'] = args.output_dir
            
            # 执行 Pipeline
            result = pipeline.execute(data)
            
            # 保存结果
            seq_output_dir = os.path.join(args.output_dir, seq_name)
            result.save_results(seq_output_dir)
            
            # 收集指标
            if result.metrics:
                metrics = {'seq': seq_name}
                metrics.update(result.metrics)
                all_results.append(metrics)
                
                logger.info(f'Results for {seq_name}:')
                for k, v in result.metrics.items():
                    logger.info(f'  {k}: {v:.4f}')
            
            # 清理 GPU 内存
            if args.device == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f'Error processing {seq_name}: {e}')
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 汇总结果
    logger.info('')
    logger.info('='*80)
    logger.info('Summary')
    logger.info('='*80)
    
    if len(all_results) > 0:
        # 计算平均值
        metrics_sum = defaultdict(list)
        for result in all_results:
            for k, v in result.items():
                if k != 'seq':
                    metrics_sum[k].append(v)
        
        avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items()}
        
        logger.info('Average metrics:')
        for k, v in avg_metrics.items():
            logger.info(f'  {k}: {v:.4f}')
        
        # 保存到 Excel
        df = pd.DataFrame(all_results)
        excel_file = os.path.join(args.output_dir, 'evaluation_results.xlsx')
        df.to_excel(excel_file, index=False)
        logger.info(f'Results saved to {excel_file}')
        
        # 保存汇总
        summary_df = pd.DataFrame([avg_metrics])
        summary_file = os.path.join(args.output_dir, 'summary.xlsx')
        summary_df.to_excel(summary_file, index=False)
        logger.info(f'Summary saved to {summary_file}')
    else:
        logger.error('No sequences were successfully processed!')
    
    # 清理
    pipeline.cleanup()
    
    logger.info('='*80)
    logger.info(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('Pipeline execution complete!')


if __name__ == '__main__':
    main()


