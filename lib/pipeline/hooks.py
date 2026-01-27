"""预定义的 Pipeline Hooks"""

import os
import time
import logging
import json
from typing import Optional
import numpy as np
import torch

from .core.data import PipelineData


logger = logging.getLogger(__name__)


# === 日志和监控 Hooks ===

def log_stage_info(data: PipelineData):
    """记录阶段信息"""
    logger.info(f"Stage: {data.current_stage}, Iteration: {data.iteration}")
    if data.metrics:
        for k, v in data.metrics.items():
            logger.info(f"  {k}: {v:.4f}")


def log_iteration_metrics(data: PipelineData):
    """记录迭代指标"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Iteration {data.iteration + 1} completed")
    logger.info(f"{'='*50}")
    if data.metrics:
        for k, v in data.metrics.items():
            logger.info(f"  {k}: {v:.4f}")


def performance_monitor(data: PipelineData):
    """性能监控"""
    current_time = time.time()
    
    # 计算阶段耗时
    if '_last_hook_time' in data.metadata:
        elapsed = current_time - data.metadata['_last_hook_time']
        logger.info(f"Stage {data.current_stage} completed in {elapsed:.2f}s")
    
    data.metadata['_last_hook_time'] = current_time
    
    # GPU 内存监控
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")


def memory_cleanup(data: PipelineData):
    """清理 GPU 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")


# === 数据保存 Hooks ===

def save_intermediate_results(data: PipelineData):
    """保存中间结果"""
    output_dir = data.metadata.get('output_dir', 'results/intermediate')
    stage = data.current_stage
    iteration = data.iteration
    
    save_dir = os.path.join(output_dir, f'iter{iteration}', stage)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存相机参数
    if data.camera_params:
        np.savez(
            os.path.join(save_dir, 'camera.npz'),
            **data.camera_params.to_dict()
        )
    
    # 保存 SMPL 参数
    if data.smpl_params:
        np.savez(
            os.path.join(save_dir, 'smpl.npz'),
            **data.smpl_params.to_dict()
        )
    
    # 保存指标
    if data.metrics:
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(data.metrics, f, indent=2)
    
    logger.info(f"Intermediate results saved to {save_dir}")


def save_masks(data: PipelineData):
    """保存分割 masks"""
    if data.masks is None:
        return
    
    output_dir = data.metadata.get('output_dir', 'results')
    masks_dir = os.path.join(output_dir, 'masks', data.sequence_name or 'unnamed')
    os.makedirs(masks_dir, exist_ok=True)
    
    torch.save(data.masks, os.path.join(masks_dir, 'masks.pt'))
    logger.info(f"Masks saved to {masks_dir}")


# === 验证 Hooks ===

def validate_camera_params(data: PipelineData):
    """验证相机参数"""
    if data.camera_params is None:
        return
    
    cam = data.camera_params
    
    # 检查旋转矩阵
    if cam.R is not None:
        R = cam.R
        if isinstance(R, torch.Tensor):
            R = R.numpy()
        
        # 检查是否正交
        for i in range(min(5, len(R))):  # 只检查前5帧
            RtR = R[i] @ R[i].T
            if not np.allclose(RtR, np.eye(3), atol=1e-5):
                logger.warning(f"Frame {i}: Rotation matrix not orthogonal")
    
    # 检查平移范围
    if cam.T is not None:
        T = cam.T
        if isinstance(T, torch.Tensor):
            T = T.numpy()
        
        max_dist = np.linalg.norm(T, axis=1).max()
        if max_dist > 100:
            logger.warning(f"Unusually large camera translation: {max_dist:.2f}m")


def validate_smpl_params(data: PipelineData):
    """验证 SMPL 参数"""
    if data.smpl_params is None:
        return
    
    smpl = data.smpl_params
    
    # 检查形状参数范围
    if smpl.betas is not None:
        betas = smpl.betas
        if isinstance(betas, torch.Tensor):
            betas = betas.numpy()
        
        if np.abs(betas).max() > 10:
            logger.warning(f"Unusual shape parameters: max |beta| = {np.abs(betas).max():.2f}")


# === 可视化 Hooks ===

def visualize_detection(data: PipelineData):
    """可视化检测结果（保存到文件）"""
    try:
        import cv2
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Visualization requires cv2 and matplotlib")
        return
    
    if data.images is None or data.bboxes is None:
        return
    
    output_dir = data.metadata.get('output_dir', 'results')
    vis_dir = os.path.join(output_dir, 'visualization', 'detection')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 可视化前几帧
    for i in range(min(5, len(data.images))):
        img = data.images[i].copy()
        boxes = data.bboxes[i]
        
        for box in boxes:
            if box[4] > 0:  # 有效检测
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{box[4]:.2f}', (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(os.path.join(vis_dir, f'frame_{i:04d}.jpg'), img)
    
    logger.info(f"Detection visualization saved to {vis_dir}")


def visualize_camera_trajectory(data: PipelineData):
    """可视化相机轨迹"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("Visualization requires matplotlib")
        return
    
    if data.camera_params is None or data.camera_params.T is None:
        return
    
    output_dir = data.metadata.get('output_dir', 'results')
    vis_dir = os.path.join(output_dir, 'visualization', 'camera')
    os.makedirs(vis_dir, exist_ok=True)
    
    T = data.camera_params.T
    if isinstance(T, torch.Tensor):
        T = T.numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(T[:, 0], T[:, 1], T[:, 2], 'b-', linewidth=1)
    ax.scatter(T[0, 0], T[0, 1], T[0, 2], c='g', s=100, marker='o', label='Start')
    ax.scatter(T[-1, 0], T[-1, 1], T[-1, 2], c='r', s=100, marker='x', label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory')
    ax.legend()
    
    plt.savefig(os.path.join(vis_dir, 'camera_trajectory.png'), dpi=150)
    plt.close()
    
    logger.info(f"Camera trajectory visualization saved to {vis_dir}")


# === 早停 Hooks ===

def early_stopping_on_error(data: PipelineData):
    """当误差过大时早停"""
    threshold = data.metadata.get('early_stop_threshold', 100)
    metric_name = data.metadata.get('early_stop_metric', 'mpjpe')
    
    if metric_name in data.metrics:
        value = data.metrics[metric_name]
        if value > threshold:
            logger.warning(f"{metric_name} ({value:.2f}) exceeds threshold ({threshold}), stopping early")
            data.should_stop = True


def early_stopping_on_convergence(data: PipelineData):
    """当收敛时早停"""
    threshold = data.metadata.get('convergence_threshold', 1e-4)
    metric_name = data.metadata.get('convergence_metric', 'reprojection_error')
    
    prev_value = data.metadata.get('_prev_convergence_value')
    
    if metric_name in data.metrics:
        current_value = data.metrics[metric_name]
        
        if prev_value is not None:
            change = abs(prev_value - current_value)
            if change < threshold:
                logger.info(f"Converged: {metric_name} change ({change:.6f}) < threshold ({threshold})")
                data.should_stop = True
        
        data.metadata['_prev_convergence_value'] = current_value


# === Hook 集合 ===

# 标准调试 hooks
DEBUG_HOOKS = {
    'after_detection': [log_stage_info, validate_camera_params],
    'after_segmentation': [log_stage_info],
    'after_slam': [log_stage_info, validate_camera_params],
    'after_hpe': [log_stage_info, validate_smpl_params],
    'after_evaluation': [log_stage_info],
    'on_complete': [performance_monitor],
}

# 可视化 hooks
VISUALIZATION_HOOKS = {
    'after_detection': [visualize_detection],
    'after_slam': [visualize_camera_trajectory],
}

# 监控 hooks
MONITORING_HOOKS = {
    'before_detection': [performance_monitor],
    'after_detection': [performance_monitor, memory_cleanup],
    'after_segmentation': [performance_monitor, memory_cleanup],
    'after_slam': [performance_monitor, memory_cleanup],
    'after_hpe': [performance_monitor, memory_cleanup],
    'after_evaluation': [performance_monitor],
}


