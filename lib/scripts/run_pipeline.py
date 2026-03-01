#!/usr/bin/env python3
"""
Pipeline 运行脚本

使用新的模块化 Pipeline 架构运行 EMDB / 3DPW 评估。

Usage:
    # EMDB: 使用配置文件运行
    python scripts/run_pipeline.py --config configs/pipelines/emdb_basic.yaml --seq 09_outdoor_walk --person P0

    # EMDB: 处理整个 split
    python scripts/run_pipeline.py --preset basic --split 2

    # 3DPW: 运行测试集
    python scripts/run_pipeline.py --config configs/pipelines/promptbase_video_depthrefine_3dpw.yaml \
        --dataset 3dpw --dataset_path datasets/3DPW --split test

    # 3DPW: 运行单个序列的某个人
    python scripts/run_pipeline.py --config ... --dataset 3dpw --dataset_path datasets/3DPW \
        --split test --seq downtown_arguing_00 --person 0
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


# ===================== 3DPW 数据加载 =====================

def load_3dpw_sequences(dataset_path: str, split: str = 'test',
                        seq_name: str = None, person: str = None):
    """
    加载 3DPW 数据集序列（按人拆分）

    Args:
        dataset_path: 3DPW 数据集根目录（含 sequenceFiles/, imageFiles/）
        split: split 名称 ('train', 'validation', 'test')
        seq_name: 指定序列名称（如 'downtown_arguing_00'）
        person: 指定人物索引（如 '0' 或 '1'）

    Returns:
        list[dict]: 每个元素包含 {pkl_path, seq_name, person_idx, full_name}
    """
    seq_dir = os.path.join(dataset_path, 'sequenceFiles', split)
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f'3DPW sequence directory not found: {seq_dir}')

    pkl_files = sorted(glob(os.path.join(seq_dir, '*.pkl')))
    if seq_name:
        pkl_files = [p for p in pkl_files
                     if os.path.basename(p).replace('.pkl', '') == seq_name]

    sequences = []
    for pkl_path in pkl_files:
        sname = os.path.basename(pkl_path).replace('.pkl', '')
        with open(pkl_path, 'rb') as f:
            data = pkl.load(f, encoding='latin1')
        num_persons = len(data['poses'])

        for pidx in range(num_persons):
            if person is not None and int(person) != pidx:
                continue
            sequences.append({
                'pkl_path': pkl_path,
                'seq_name': sname,
                'person_idx': pidx,
                'full_name': f'{sname}_person{pidx}',
            })

    return sequences


def _gender_str(g: str) -> str:
    """将 3DPW 性别标识映射为 SMPL 模型名称"""
    mapping = {'m': 'male', 'f': 'female', 'male': 'male', 'female': 'female'}
    return mapping.get(g, 'neutral')


def _bbox_from_poses2d(poses2d: np.ndarray, img_h: int = 1080, img_w: int = 1920,
                       padding: float = 1.2) -> np.ndarray:
    """
    从 3DPW poses2d (COCO 18) 推导 bbox，作为无检测器时的 fallback。

    Args:
        poses2d: (N, 3, 18) — 每帧 18 个关键点，3 = (x, y, conf)
        img_h, img_w: 图像尺寸（用于 clamp）
        padding: 边界框外扩比例

    Returns:
        bboxes: (N, 1, 5) — (x1, y1, x2, y2, score)
    """
    N = poses2d.shape[0]
    bboxes = np.zeros((N, 1, 5), dtype=np.float32)

    for i in range(N):
        kp = poses2d[i]  # (3, 18)
        conf = kp[2]
        valid = conf > 0.1
        if valid.sum() < 2:
            # 关键点太少，使用全图 bbox
            bboxes[i, 0] = [0, 0, img_w, img_h, 0.1]
            continue
        xs = kp[0, valid]
        ys = kp[1, valid]
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * padding, (y2 - y1) * padding
        x1 = max(0, cx - w / 2)
        y1 = max(0, cy - h / 2)
        x2 = min(img_w, cx + w / 2)
        y2 = min(img_h, cy + h / 2)
        bboxes[i, 0] = [x1, y1, x2, y2, float(conf.mean())]

    return bboxes


def load_3dpw_sequence_data(seq_info: dict, dataset_path: str) -> PipelineData:
    """
    加载 3DPW 单人序列数据到 PipelineData

    将 3DPW pkl 格式转换为与 EMDB 评估组件兼容的 annotations 字典。

    Args:
        seq_info: load_3dpw_sequences 返回的单条记录
        dataset_path: 3DPW 数据集根目录

    Returns:
        PipelineData 实例
    """
    with open(seq_info['pkl_path'], 'rb') as f:
        raw = pkl.load(f, encoding='latin1')

    pidx = seq_info['person_idx']
    sname = seq_info['seq_name']
    full_name = seq_info['full_name']

    # --- SMPL 参数 ---
    poses = raw['poses'][pidx]           # (N, 72) axis-angle
    trans = raw['trans'][pidx]           # (N, 3)
    betas_full = raw['betas'][pidx]     # (300,) — 只取前 10
    gender = _gender_str(raw['genders'][pidx])
    n_frames = poses.shape[0]

    poses_root = poses[:, :3]            # (N, 3)
    poses_body = poses[:, 3:]            # (N, 69)
    betas = betas_full[:10]              # (10,)

    # --- 相机参数 ---
    cam_poses = raw['cam_poses']         # (N, 4, 4) world2cam extrinsics
    cam_intrinsics = raw['cam_intrinsics']  # (3, 3)

    # cam_poses 是 world-to-cam 外参矩阵
    extrinsics = cam_poses.astype(np.float32)  # (N, 4, 4)

    # --- 有效帧 ---
    campose_valid = raw['campose_valid'][pidx]  # (N,)
    good_frames_mask = campose_valid.astype(bool)

    # --- 图像路径 ---
    # 注意: img_frame_ids 是原始视频帧编号（可能不连续，如 0,2,4,...），
    # 但 imageFiles/ 中的图片是按顺序编号的 (image_00000.jpg, image_00001.jpg, ...)
    img_dir = os.path.join(dataset_path, 'imageFiles', sname)
    image_paths = [os.path.join(img_dir, f'image_{idx:05d}.jpg') for idx in range(n_frames)]
    
    # 过滤掉末尾不存在的图像帧（部分序列图片数 < 标注帧数）
    while image_paths and not os.path.exists(image_paths[-1]):
        image_paths.pop()
    
    actual_frames = len(image_paths)
    if actual_frames < n_frames:
        # 截断所有标注到实际存在的帧数
        poses_root = poses_root[:actual_frames]
        poses_body = poses_body[:actual_frames]
        trans = trans[:actual_frames]
        extrinsics = extrinsics[:actual_frames]
        good_frames_mask = good_frames_mask[:actual_frames]
        if 'poses2d' in raw and raw['poses2d'][pidx] is not None:
            raw['poses2d'][pidx] = raw['poses2d'][pidx][:actual_frames]
        n_frames = actual_frames

    # --- 构建 EMDB 兼容的 annotations ---
    annotations = {
        'smpl': {
            'poses_root': poses_root,
            'poses_body': poses_body,
            'betas': betas,
            'trans': trans,
        },
        'camera': {
            'extrinsics': extrinsics,
            'intrinsics': cam_intrinsics,
        },
        'gender': gender,
        'n_frames': n_frames,
        'good_frames_mask': good_frames_mask,
    }

    # --- Bbox (fallback from poses2d，detection 组件会覆盖) ---
    bboxes = None
    if 'poses2d' in raw and raw['poses2d'][pidx] is not None:
        poses2d = raw['poses2d'][pidx]  # (N, 3, 18)
        bboxes = _bbox_from_poses2d(poses2d)

    # --- PipelineData ---
    data = PipelineData(
        sequence_name=full_name,
        sequence_path=img_dir,
        image_paths=image_paths,
        annotations=annotations,
        valid_frames_mask=good_frames_mask,
        bboxes=bboxes,
    )
    data.metadata['dataset_type'] = '3dpw'

    return data


# ===================== EMDB 数据加载 =====================

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
    parser = argparse.ArgumentParser(description='Run EMDB / 3DPW evaluation pipeline')
    
    # Pipeline 配置
    parser.add_argument('--config', type=str, default=None,
                       help='Path to pipeline config file (YAML)')

    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='emdb',
                       choices=['emdb', '3dpw'],
                       help='Dataset type (emdb or 3dpw)')
    parser.add_argument('--seq', type=str, default=None,
                       help='Specific sequence to process')
    parser.add_argument('--person', type=str, default=None,
                       help='Specific person (EMDB: P0~P9, 3DPW: 0/1)')
    parser.add_argument('--split', type=str, default='2',
                       help='Dataset split (EMDB: 1/2/3, 3DPW: train/validation/test)')
    parser.add_argument('--dataset_path', type=str, 
                       default='/home/gejunchen/Work/2026-1/Datasets/EMDB',
                       help='Path to dataset root')
    
    # Pipeline 参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--hpe_mode', type=str, default='accurate',
                       choices=['accurate', 'efficient'],
                       help='HPE estimation mode')
    
    # 多 GPU 分片参数
    parser.add_argument('--shard_id', type=int, default=None,
                       help='Shard index for multi-GPU parallel (0-based)')
    parser.add_argument('--num_shards', type=int, default=None,
                       help='Total number of shards for multi-GPU parallel')
    
    # 调试选项
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug hooks')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization hooks')
    
    args = parser.parse_args()

    # 如果提供了配置文件，先加载配置
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        output_dir = config.get('output_dir')
    else:
        # 使用默认输出目录
        output_dir = 'outputs/pipeline'

    # 设置日志
    logger = setup_logging(output_dir)
    logger.info('='*80)
    logger.info(f'Pipeline Runner (dataset={args.dataset})')
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
            output_dir=output_dir
        )
    
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
    
    # 加载序列（根据数据集类型分发）
    is_3dpw = (args.dataset == '3dpw')

    if is_3dpw:
        logger.info(f'Loading 3DPW sequences (split={args.split})...')
        seq_infos = load_3dpw_sequences(
            args.dataset_path, args.split, args.seq, args.person)
        sequences = seq_infos  # list[dict]
    else:
        logger.info(f'Loading EMDB sequences (split={args.split})...')
        sequences = load_emdb_sequences(
            args.dataset_path, int(args.split), args.seq, args.person)
    
    if len(sequences) == 0:
        logger.error('No sequences found matching the criteria!')
        return
    
    # 多 GPU 分片：只处理当前分片对应的序列子集
    if args.shard_id is not None and args.num_shards is not None:
        total = len(sequences)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, total)
        sequences = sequences[start:end]
        logger.info(f'Shard {args.shard_id}/{args.num_shards}: processing sequences [{start}:{end}] ({len(sequences)} seqs)')
    
    logger.info(f'Found {len(sequences)} sequence(s) to process:')
    for item in sequences:
        if is_3dpw:
            logger.info(f'  - {item["full_name"]}')
        else:
            logger.info(f'  - {item}')
    
    # 初始化 Pipeline
    logger.info('Setting up pipeline...')
    pipeline.setup()
    
    # 处理每个序列
    all_results = []
    
    for i, item in enumerate(sequences):
        if is_3dpw:
            seq_name = item['full_name']
        else:
            seq_name = item.split('/')[-1]

        logger.info('')
        logger.info('='*80)
        logger.info(f'Processing sequence {i+1}/{len(sequences)}: {seq_name}')
        logger.info('='*80)
        
        try:
            # 加载数据
            if is_3dpw:
                data = load_3dpw_sequence_data(item, args.dataset_path)
            else:
                data = load_sequence_data(item)
            data.metadata['output_dir'] = output_dir

            # 执行 Pipeline
            result = pipeline.execute(data)

            # 保存结果
            seq_output_dir = os.path.join(output_dir, seq_name)
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
        excel_file = os.path.join(output_dir, 'evaluation_results.xlsx')
        df.to_excel(excel_file, index=False)
        logger.info(f'Results saved to {excel_file}')

        # 保存汇总
        summary_df = pd.DataFrame([avg_metrics])
        summary_file = os.path.join(output_dir, 'summary.xlsx')
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


