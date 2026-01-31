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
    # logger.info(f"Stage: {data.current_stage}, Iteration: {data.iteration}")
    # if data.metrics:
    #     for k, v in data.metrics.items():
    #         logger.info(f"  {k}: {v:.4f}")


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


def visualize_body_trajectory(data: PipelineData):
    """可视化人体全局轨迹"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("Visualization requires matplotlib")
        return

    if data.smpl_params is None:
        logger.warning("No SMPL params available")
        return

    # 优先使用 global_trans（世界坐标系），如果没有则使用 trans（相机坐标系）
    if data.smpl_params.global_trans is not None:
        trans = data.smpl_params.global_trans
        coord_system = "world"
    elif data.smpl_params.trans is not None:
        trans = data.smpl_params.trans
        coord_system = "camera"
        logger.warning("Only camera-coordinate trans available, trajectory may not be globally accurate")
    else:
        logger.warning("No body trajectory data available")
        return

    output_dir = data.metadata.get('output_dir', 'results')
    vis_dir = os.path.join(output_dir, 'visualization', 'body')
    os.makedirs(vis_dir, exist_ok=True)

    if isinstance(trans, torch.Tensor):
        trans = trans.numpy()

    # trans shape: [N, 3] or [N, M, 3] (M = number of people)
    if len(trans.shape) == 3:
        # Multiple people, visualize the first one
        logger.info(f"Multiple people detected ({trans.shape[1]}), visualizing first person")
        trans = trans[:, 0, :]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(trans[:, 0], trans[:, 1], trans[:, 2], 'b-', linewidth=1)
    ax.scatter(trans[0, 0], trans[0, 1], trans[0, 2], c='g', s=100, marker='o', label='Start')
    ax.scatter(trans[-1, 0], trans[-1, 1], trans[-1, 2], c='r', s=100, marker='x', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Body Trajectory ({coord_system} coordinate system)')
    ax.legend()

    plt.savefig(os.path.join(vis_dir, 'body_trajectory.png'), dpi=150)
    plt.close()

    logger.info(f"Body trajectory visualization saved to {vis_dir} ({coord_system} coords)")


def visualize_smpl_mesh(data: PipelineData, interval: int = None):
    """
    每隔 N 帧将估计的 SMPL mesh 投影到图片上并保存

    Args:
        data: Pipeline 数据容器
        interval: 可视化间隔（帧数），如果为 None 则从 metadata 读取
    """
    # 从 metadata 获取间隔，如果没有则使用默认值 100
    if interval is None:
        interval = data.metadata.get('smpl_vis_interval', 100)
    try:
        import cv2
        import numpy as np
        import torch
        from glob import glob
    except ImportError:
        logger.warning("Visualization requires cv2, numpy, torch, glob")
        return

    # 检查必要数据
    if data.smpl_params is None:
        logger.info("No SMPL parameters to visualize")
        return

    if len(data.image_paths) == 0:
        logger.info("No image paths available")
        return

    # 获取输出目录
    output_dir = data.metadata.get('output_dir', 'results')
    vis_dir = os.path.join(output_dir, 'visualization', 'smpl_mesh')
    os.makedirs(vis_dir, exist_ok=True)

    # 延迟导入（避免启动时加载）
    try:
        from lib.models.smpl import SMPL
        from lib.vis.renderer import Renderer
    except ImportError:
        logger.warning("SMPL or Renderer not available")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 获取 SMPL 模型和 faces
    smpl_model = SMPL().to(device)
    faces = smpl_model.faces

    # 获取图像尺寸
    img_sample = cv2.imread(data.image_paths[0])
    img_height, img_width = img_sample.shape[:2]

    # 获取相机参数
    img_focal = 1000.0  # 默认值
    img_center = None

    # 尝试从 camera_params 获取
    if data.camera_params and data.camera_params.focal_length:
        img_focal = data.camera_params.focal_length
        if data.camera_params.principal_point is not None:
            img_center = data.camera_params.principal_point

    if img_center is None:
        img_center = np.array([img_width / 2, img_height / 2])

    # 创建 renderer
    renderer = Renderer(img_width, img_height, img_focal, device, faces=faces,
                       bin_size=-1, max_faces_per_bin=30000)

    # 获取 SMPL 参数
    smpl = data.smpl_params

    # 使用 rotmat 而不是 poses（rotmat 已经是正确的 rotation matrix 格式）
    if smpl.rotmat is None:
        logger.warning("SMPL rotmat is None, cannot visualize. Skipping visualization.")
        return

    poses = smpl.rotmat  # [N, 24, 3, 3] - 已经是 rotation matrix 格式
    betas = smpl.betas  # [N, 10]
    trans = smpl.trans  # [N, 3]

    # 转换为 tensor
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses).float()
    if isinstance(betas, np.ndarray):
        betas = torch.from_numpy(betas).float()
    if isinstance(trans, np.ndarray):
        trans = torch.from_numpy(trans).float()

    poses = poses.to(device)
    betas = betas.to(device)
    trans = trans.to(device)

    # 输出调试信息
    # logger.info(f"[DEBUG] Original SMPL params shapes:")
    # logger.info(f"  - rotmat: {poses.shape}")
    # logger.info(f"  - betas: {betas.shape}")
    # logger.info(f"  - trans: {trans.shape}")

    # logger.info(f"[DEBUG] Rotmat tensor info:")
    # logger.info(f"  - Shape: {poses.shape}")
    # logger.info(f"  - Dimensions: {poses.dim()}")
    # logger.info(f"  - Device: {poses.device}")
    # logger.info(f"  - Dtype: {poses.dtype}")
    # logger.info(f"  - Min value: {poses.min().item():.4f}")
    # logger.info(f"  - Max value: {poses.max().item():.4f}")

    # rotmat 应该已经是 [N, 24, 3, 3] 格式，不需要转换
    if poses.dim() != 4 or poses.shape[1:] != (24, 3, 3):
        logger.warning(f"Unexpected rotmat shape: {poses.shape}, expected [N, 24, 3, 3]")
        return

    logger.info(f"[DEBUG] Rotmat format is correct, no conversion needed!")

    # 每隔 interval 帧可视化一次
    num_frames = len(data.image_paths)
    for frame_idx in range(0, num_frames, interval):
        try:
            # 加载图像
            img = cv2.imread(data.image_paths[frame_idx])
            if img is None:
                continue

            # 获取当前帧的 SMPL 参数
            pose = poses[frame_idx:frame_idx+1]  # [1, 24, 3, 3]
            beta = betas[frame_idx:frame_idx+1] if betas.dim() == 2 else betas  # [1, 10]

            # 处理 trans - 确保形状是 [1, 3] (SMPLx 期望 [batch_size, 3])
            tran = trans[frame_idx]  # [1, 3]
            if tran.dim() == 2 and tran.shape[0] == 1:  # [1, 3]
                pass  # 保持 [1, 3]
            elif tran.dim() == 1:  # [3]
                tran = tran.unsqueeze(0)  # -> [1, 3]
            else:
                logger.warning(f"Unexpected trans shape: {tran.shape}")
                tran = tran.view(1, -1)[:, :3]  # 强制变成 [1, 3]


            # 分离 global_orient 和 body_pose
            global_orient = pose[:, [0]]  # [1, 1, 3, 3]
            body_pose = pose[:, 1:]  # [1, 23, 3, 3]

            # logger.info(f"[DEBUG] Frame {frame_idx} SMPL input shapes:")
            # logger.info(f"  - global_orient: {global_orient.shape}")
            # logger.info(f"  - body_pose: {body_pose.shape}")
            # logger.info(f"  - betas: {beta.shape}")
            # logger.info(f"  - transl: {tran.shape} (should be [1, 3])")

            # 推理 SMPL
            with torch.no_grad():
                smpl_output = smpl_model(
                    body_pose=body_pose,
                    global_orient=global_orient,
                    betas=beta,
                    transl=tran,  # 应该是 [3]
                    pose2rot=False,
                    default_smpl=True  # 使用标准 SMPL 输出，避免额外的关节计算
                )

            # logger.info(f"[DEBUG] SMPL output shapes:")
            # logger.info(f"  - vertices: {smpl_output.vertices.shape}")
            # logger.info(f"  - joints: {smpl_output.joints.shape if hasattr(smpl_output, 'joints') else 'N/A'}")

            vertices = smpl_output.vertices[0]  # [6890, 3] - 在 GPU 上

            # 调试输出（第一帧和中间帧）
            if frame_idx == 0 or frame_idx == 100:
                logger.info(f"[DEBUG VISUALIZE] Frame {frame_idx}:")
                logger.info(f"  trans: {tran.cpu().numpy() if hasattr(tran, 'cpu') else tran}")
                logger.info(f"  vertices range: [{vertices.min():.2f}, {vertices.max():.2f}]")
                logger.info(f"  vertices mean: {vertices.mean(dim=0).cpu().numpy()}")
                logger.info(f"  vertices std: {vertices.std(dim=0).cpu().numpy()}")
                logger.info(f"  img_focal: {img_focal:.2f}")
                logger.info(f"  img_center: {img_center}")
                logger.info(f"  img_size: ({img_width}, {img_height})")

            # 渲染（vertices 保持在 GPU 上，与 renderer 的设备一致）
            rendered_img = renderer.render_mesh(
                vertices,  # GPU tensor
                img.copy(),
                colors=[0.5, 0.8, 0.5]  # 绿色
            )

            # 保存
            output_path = os.path.join(vis_dir, f'frame_{frame_idx:04d}.jpg')
            cv2.imwrite(output_path, rendered_img)

        except Exception as e:
            logger.warning(f"Failed to visualize frame {frame_idx}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            continue

    logger.info(f"SMPL mesh visualization saved to {vis_dir} (every {interval} frames)")


def visualize_gt_smpl_mesh(data: PipelineData, interval: int = None):
    """
    可视化GT SMPL mesh（使用世界坐标系的trans）

    专门用于GTSmplLoaderComponent加载的数据，使用metadata中保存的trans_world。
    """
    # 从 metadata 获取间隔，如果没有则使用默认值 100
    if interval is None:
        interval = data.metadata.get('smpl_vis_interval', 100)
    try:
        import cv2
        import numpy as np
        import torch
        from glob import glob
    except ImportError:
        logger.warning("Visualization requires cv2, numpy, torch, glob")
        return

    # 检查必要数据
    if data.smpl_params is None:
        logger.info("No SMPL parameters to visualize")
        return

    # 检查是否有trans_world（GT SMPL特有的）
    if 'trans_world' not in data.metadata:
        logger.info("No trans_world in metadata, using standard visualization")
        return visualize_smpl_mesh(data, interval)

    if len(data.image_paths) == 0:
        logger.info("No image paths available")
        return

    # 获取输出目录
    output_dir = data.metadata.get('output_dir', 'results')
    vis_dir = os.path.join(output_dir, 'visualization', 'smpl_mesh_gt')
    os.makedirs(vis_dir, exist_ok=True)

    # 延迟导入
    try:
        from lib.models.smpl import SMPL
        from lib.vis.renderer import Renderer
    except ImportError:
        logger.warning("SMPL or Renderer not available")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 获取 SMPL 模型和 faces
    smpl_model = SMPL().to(device)
    faces = smpl_model.faces

    # 获取图像尺寸
    img_sample = cv2.imread(data.image_paths[0])
    img_height, img_width = img_sample.shape[:2]

    # 获取相机参数
    img_focal = 1000.0
    img_center = None

    if data.annotations and 'camera' in data.annotations:
        intr = data.annotations['camera'].get('intrinsics')
        if intr is not None:
            img_focal = (intr[0, 0] + intr[1, 1]) / 2.0
            img_center = intr[:2, 2]

    if data.camera_params and data.camera_params.focal_length:
        img_focal = data.camera_params.focal_length
        if data.camera_params.principal_point is not None:
            img_center = data.camera_params.principal_point

    if img_center is None:
        img_center = np.array([img_width / 2, img_height / 2])

    # 创建 renderer
    renderer = Renderer(img_width, img_height, img_focal, device, faces=faces,
                       bin_size=-1, max_faces_per_bin=30000)

    # 获取 SMPL 参数
    smpl = data.smpl_params
    trans_world = data.metadata['trans_world']  # 世界坐标系的trans
    poses_root_world = data.metadata.get('poses_root_world')  # 世界坐标系的root orientation

    # 使用 rotmat
    if smpl.rotmat is None:
        logger.warning("SMPL rotmat is None, cannot visualize. Skipping visualization.")
        return

    poses = smpl.rotmat  # 相机坐标系 [N, 24, 3, 3]
    betas = smpl.betas  # [N, 10]

    # 转换为 tensor
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses).float()
    if isinstance(betas, np.ndarray):
        betas = torch.from_numpy(betas).float()
    if isinstance(trans_world, np.ndarray):
        trans_world = torch.from_numpy(trans_world).float()

    poses = poses.to(device)
    betas = betas.to(device)
    trans_world = trans_world.to(device)

    # 将poses_root_world转换为rotation matrix
    from lib.utils.rotation_conversions import axis_angle_to_matrix
    if poses_root_world is not None:
        if isinstance(poses_root_world, np.ndarray):
            poses_root_world = torch.from_numpy(poses_root_world).float()
        poses_root_world = poses_root_world.to(device)
        root_rotmat_world = axis_angle_to_matrix(poses_root_world)  # [N, 3, 3]
        root_rotmat_world = root_rotmat_world[:, None, :, :]  # [N, 1, 3, 3]

        # 替换poses中的root orientation为世界坐标系的
        body_rotmat = poses[:, 1:]  # [N, 23, 3, 3]
        poses_world = torch.cat([root_rotmat_world, body_rotmat], dim=1)  # [N, 24, 3, 3]
    else:
        poses_world = poses

    # 每隔 interval 帧可视化一次
    num_frames = len(data.image_paths)
    for frame_idx in range(0, num_frames, interval):
        try:
            # 加载图像
            img = cv2.imread(data.image_paths[frame_idx])
            if img is None:
                continue

            # 获取当前帧的 SMPL 参数
            pose = poses_world[frame_idx:frame_idx+1]  # [1, 24, 3, 3]
            beta = betas[frame_idx:frame_idx+1] if betas.dim() == 2 else betas  # [1, 10]
            tran = trans_world[frame_idx]  # [3]

            if tran.dim() == 1:
                tran = tran.unsqueeze(0)  # -> [1, 3]

            # 分离 global_orient 和 body_pose
            global_orient = pose[:, [0]]  # [1, 1, 3, 3]
            body_pose = pose[:, 1:]  # [1, 23, 3, 3]

            # 推理 SMPL（使用世界坐标系的trans）
            with torch.no_grad():
                smpl_output = smpl_model(
                    body_pose=body_pose,
                    global_orient=global_orient,
                    betas=beta,
                    transl=tran,
                    pose2rot=False,
                    default_smpl=True
                )

            vertices = smpl_output.vertices[0]  # [6890, 3]

            # 渲染
            rendered_img = renderer.render_mesh(
                vertices,
                img.copy(),
                colors=[0.5, 0.8, 0.5]
            )

            # 保存
            output_path = os.path.join(vis_dir, f'frame_{frame_idx:04d}.jpg')
            cv2.imwrite(output_path, rendered_img)

        except Exception as e:
            logger.warning(f"Failed to visualize frame {frame_idx}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            continue

    logger.info(f"GT SMPL mesh visualization saved to {vis_dir} (every {interval} frames, world coordinate system)")


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
    'after_hpe': [visualize_smpl_mesh],
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


