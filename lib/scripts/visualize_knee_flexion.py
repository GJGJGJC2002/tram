#!/usr/bin/env python3
"""
WHAM Figure 4 风格可视化：左膝屈曲角度曲线 + Root-Aligned 下半身 SMPL Mesh

功能：
  1. 上方：选定帧的原始图像序列（标注 t1, t2）
  2. 中间：左膝屈曲角度 (Left Knee Flexion) 随时间变化曲线，对比 GT 和多种方法
  3. 下方：在 t1, t2 时刻，各方法 root-aligned 后的下半身 SMPL mesh 侧视图

支持数据集：EMDB2

Usage:
    python lib/scripts/visualize_knee_flexion.py \
        --seq 29_outdoor_stairs_up \
        --methods gvhmr_base_warmstart promptbase_video_warmstart_emdb2 \
        --labels GVHMR PromptHMR \
        --t1 100 --t2 200 \
        --frame_range 0 400 \
        --output figures/knee_flexion_comparison.png
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from glob import glob

from lib.models.smpl import SMPL

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GVHMR_ROOT = os.path.join(PROJECT_ROOT, 'thirdparty', 'GVHMR')
if GVHMR_ROOT not in sys.path:
    sys.path.insert(0, GVHMR_ROOT)

# PyTorch3D
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    look_at_view_transform,
)


# ============================================================
# J_regressor 加载（用于坐标系对齐）
# ============================================================
_J_regressor = None

def _get_J_regressor():
    global _J_regressor
    if _J_regressor is None:
        path = os.path.join(GVHMR_ROOT, "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
        _J_regressor = torch.load(path)
    return _J_regressor


# ============================================================
# 坐标系对齐（移植自 viser_compare_methods.py）
# ============================================================
def ensure_y_up(verts):
    """
    检测并修正 Y 轴方向：如果头部 Y < 脚部 Y，做 180° X 轴翻转。
    SMPL joints: 15=head, 7/8=ankle(L/R)

    Args:
        verts: (F, V, 3) tensor
    Returns:
        verts: (F, V, 3) tensor, 确保 Y-up
    """
    J_regressor = _get_J_regressor()
    n_check = min(5, verts.shape[0])
    verts_check = verts[:n_check].to(J_regressor.device)
    joints_check = torch.einsum('jv,fvi->fji', J_regressor, verts_check)

    head_y = joints_check[:, 15, 1].mean().item()
    ankle_y = (joints_check[:, 7, 1].mean().item() + joints_check[:, 8, 1].mean().item()) / 2

    if head_y < ankle_y:
        print(f"  [Align] Y-down detected (head_y={head_y:.3f} < ankle_y={ankle_y:.3f}), flipping")
        verts = verts.clone()
        verts[:, :, 1] = -verts[:, :, 1]
        verts[:, :, 2] = -verts[:, :, 2]
    else:
        print(f"  [Align] Y-up OK (head_y={head_y:.3f} > ankle_y={ankle_y:.3f})")

    return verts


def move_to_start_point_face_z(verts):
    """
    将 vertices 归一化：XZ 原点、站在地面、面朝 Z 轴。

    Args:
        verts: (F, V, 3) tensor
    Returns:
        verts: (F, V, 3) aligned tensor
    """
    from hmr4d.utils.geo_transform import compute_T_ayfz2ay, apply_T_on_points
    from einops import einsum as einops_einsum

    J_regressor = _get_J_regressor()
    device = J_regressor.device
    verts = verts.clone().to(device)

    # 位置归一化
    offset = einops_einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # pelvis
    offset[1] = verts[:, :, 1].min()
    verts = verts - offset

    # 面朝 Z 轴旋转
    T_ay2ayfz = compute_T_ayfz2ay(
        einops_einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"),
        inverse=True
    )
    verts = apply_T_on_points(verts, T_ay2ayfz)

    return verts.cpu()


# ============================================================
# 颜色配置
# ============================================================
METHOD_COLORS_HEX = {
    'GT': '#888888',
    'palette': [
        '#4CAF50',   # green
        '#2196F3',   # blue
        '#FF9800',   # orange
        '#E91E63',   # pink
        '#9C27B0',   # purple
        '#00BCD4',   # cyan
    ]
}

# 下半身 mesh 渲染颜色 (RGB 0-1)
MESH_COLORS_RGB = [
    (0.45, 0.75, 0.45),   # green
    (0.40, 0.60, 0.90),   # blue
    (0.90, 0.65, 0.30),   # orange
    (0.85, 0.35, 0.55),   # pink
    (0.60, 0.30, 0.70),   # purple
    (0.20, 0.75, 0.80),   # cyan
]
GT_MESH_COLOR = (0.70, 0.70, 0.70)


# ============================================================
# SMPL 下半身 faces 提取
# ============================================================
def get_lower_body_faces(smpl_model):
    """
    基于 SMPL skinning weights 提取下半身面片

    下半身关节: pelvis(0), left_hip(1), right_hip(2), left_knee(4),
                right_knee(5), left_ankle(7), right_ankle(8),
                left_foot(10), right_foot(11)
    """
    lbs_weights = smpl_model.lbs_weights.cpu().numpy()  # (6890, 24)
    lower_body_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_weight = lbs_weights[:, lower_body_joints].sum(axis=1)
    lower_body_mask = lower_body_weight > 0.5  # (6890,)

    faces = smpl_model.faces.astype(np.int64)
    face_mask = (lower_body_mask[faces[:, 0]] &
                 lower_body_mask[faces[:, 1]] &
                 lower_body_mask[faces[:, 2]])
    lower_faces = faces[face_mask]

    return lower_faces, lower_body_mask


# ============================================================
# 膝关节屈曲角度提取
# ============================================================
def extract_knee_flexion(body_pose, joint_name='left_knee'):
    """
    从 body_pose axis-angle 中提取膝关节屈曲角度 (degrees)

    SMPL body joints (不含 root):
      0=left_hip, 1=right_hip, 2=spine, 3=left_knee, 4=right_knee, ...

    Args:
        body_pose: (N, 63) or (N, 69) axis-angle
        joint_name: 'left_knee' or 'right_knee'

    Returns:
        angles: (N,) in degrees
    """
    joint_idx = 3 if joint_name == 'left_knee' else 4
    offset = joint_idx * 3
    aa = body_pose[:, offset:offset + 3]
    angles = np.degrees(np.linalg.norm(aa, axis=1))
    return angles


# ============================================================
# EMDB2 数据加载
# ============================================================
def find_emdb_sequence(dataset_path, seq_name):
    for person_dir in sorted(glob(os.path.join(dataset_path, 'P*'))):
        seq_dir = os.path.join(person_dir, seq_name)
        if os.path.isdir(seq_dir):
            person = os.path.basename(person_dir)
            ann_file = os.path.join(seq_dir, f'{person}_{seq_name}_data.pkl')
            if os.path.exists(ann_file):
                return seq_dir, ann_file
    raise FileNotFoundError(f'Sequence {seq_name} not found in {dataset_path}')


def load_gt_data(ann_file):
    with open(ann_file, 'rb') as f:
        ann = pickle.load(f)
    return {
        'poses_root': ann['smpl']['poses_root'],
        'poses_body': ann['smpl']['poses_body'],
        'betas': ann['smpl']['betas'],
        'trans': ann['smpl']['trans'],
        'gender': ann['gender'],
        'n_frames': ann['n_frames'],
    }


def load_pred_data(result_dir, seq_name):
    smpl_file = os.path.join(result_dir, seq_name, 'smpl.npz')
    if not os.path.exists(smpl_file):
        raise FileNotFoundError(f'Prediction not found: {smpl_file}')
    data = np.load(smpl_file)

    if 'global_orient_c' in data:
        # GVHMR / PromptHMR 格式
        return {
            'poses_root': data['global_orient_c'],
            'poses_body': data['body_pose_aa'],
            'betas': data['betas'],
            'trans': data['trans'],
            'format': 'gvhmr',
        }
    elif 'rotmat' in data:
        # TRAM 格式: rotmat (N, 24, 3, 3) -> axis-angle
        from pytorch3d.transforms import matrix_to_axis_angle
        rotmat = torch.from_numpy(data['rotmat']).float()  # (N, 24, 3, 3)
        aa = matrix_to_axis_angle(rotmat)  # (N, 24, 3)
        poses_root = aa[:, 0, :].numpy()   # (N, 3)
        poses_body = aa[:, 1:, :].reshape(-1, 69).numpy()  # (N, 69)
        return {
            'poses_root': poses_root,
            'poses_body': poses_body,
            'betas': data['betas'],
            'trans': data['trans'],
            'format': 'tram',
        }
    else:
        raise ValueError(f"Unknown smpl.npz format, keys: {list(data.keys())}")


# ============================================================
# PyTorch3D 下半身 Mesh 渲染
# ============================================================
def render_lower_body_pytorch3d(vertices, lower_faces, color_rgb,
                                device='cuda', img_size=(300, 400)):
    """
    用 PyTorch3D 渲染 root-aligned 下半身 mesh 侧视图（白色背景）

    Args:
        vertices: (V, 3) numpy array — root-aligned full SMPL vertices
        lower_faces: (F, 3) numpy array — lower body faces
        color_rgb: tuple (r, g, b) 0-1
        device: cuda device
        img_size: (width, height)

    Returns:
        image: (H, W, 3) uint8
    """
    w, h = img_size

    verts_t = torch.from_numpy(vertices).float().unsqueeze(0).to(device)
    faces_t = torch.from_numpy(lower_faces.copy()).long().unsqueeze(0).to(device)

    # 逐顶点颜色
    colors_t = torch.tensor(color_rgb, dtype=torch.float32, device=device)
    colors_t = colors_t.view(1, 1, 3).expand(1, verts_t.shape[1], 3)
    textures = TexturesVertex(verts_features=colors_t)

    mesh = Meshes(verts=verts_t, faces=faces_t, textures=textures)

    # 计算下半身中心和范围
    lb_vert_ids = np.unique(lower_faces.flatten())
    lb_verts_np = vertices[lb_vert_ids]
    lb_center = lb_verts_np.mean(0)
    extent = (lb_verts_np.max(0) - lb_verts_np.min(0)).max()

    # 侧视图相机：从左前方看（azim=-70°）
    dist = extent * 3.0
    R, T = look_at_view_transform(
        dist=dist, elev=0, azim=-70,
        at=((lb_center[0], lb_center[1], lb_center[2]),),
        up=((0, 1, 0),),
    )

    cameras = PerspectiveCameras(
        device=device,
        R=R.to(device),
        T=T.to(device),
        focal_length=torch.tensor([[max(w, h) * 1.5]], device=device),
        principal_point=torch.tensor([[w / 2, h / 2]], device=device),
        image_size=torch.tensor([[h, w]], device=device),
        in_ndc=False,
    )

    # lights = PointLights(
    #     device=device,
    #     location=cameras.get_camera_center(),
    #     ambient_color=[[1.0, 1.0, 1.0]],
    #     diffuse_color=[[0.0, 0.0, 0.0]],
    #     specular_color=[[0.0, 0.0, 0.0]],
    # )

    lights = PointLights(
        device=device,
        location=cameras.get_camera_center(),
        ambient_color=[[0.2, 0.2, 0.2]],
        diffuse_color=[[0.6, 0.6, 0.6]],
    )

    raster_settings = RasterizationSettings(
        image_size=(h, w),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
        )
    )

    # 渲染
    materials = Materials(device=device, shininess=5)
    result = renderer(mesh, materials=materials)
    img = result[0, ..., :3].cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)

    # 将背景替换为白色
    alpha = result[0, ..., 3].cpu().numpy()
    mask = alpha < 0.5
    img[mask] = 255

    return img


def compute_sequence_vertices(smpl_model, poses_root, poses_body, betas, trans):
    """
    计算整个序列的 SMPL 顶点 (F, V, 3)，用于后续对齐。
    """
    N = poses_root.shape[0]
    tt = lambda x: torch.from_numpy(x).float()

    bp = poses_body.copy()
    if bp.shape[1] == 63:
        bp = np.concatenate([bp, np.zeros((N, 6), dtype=np.float32)], axis=1)

    if betas.ndim == 1:
        betas_full = np.tile(betas, (N, 1))
    else:
        betas_full = betas

    # 分批计算以节省显存
    batch_size = 64
    all_verts = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        output = smpl_model(
            body_pose=tt(bp[start:end]),
            global_orient=tt(poses_root[start:end]),
            betas=tt(betas_full[start:end]),
            transl=tt(trans[start:end]),
            pose2rot=True,
            default_smpl=True,
        )
        all_verts.append(output.vertices.detach().cpu())

    return torch.cat(all_verts, dim=0)  # (F, V, 3)


def compute_face_z_transform(verts_frame):
    """
    计算单帧 face-z 对齐的变换参数（offset + rotation），不实际变换。
    用于以 GT 朝向为基准，统一应用到所有方法。

    Args:
        verts_frame: (V, 3) tensor
    Returns:
        offset: (3,) tensor — pelvis offset (Y=地面最低点)
        T_ay2ayfz: (1, 4, 4) tensor — face-z 旋转变换矩阵
    """
    from hmr4d.utils.geo_transform import compute_T_ayfz2ay
    from einops import einsum as einops_einsum

    J_regressor = _get_J_regressor()
    device = J_regressor.device

    verts = verts_frame.clone().unsqueeze(0).to(device)

    joints = einops_einsum(J_regressor, verts[0], "j v, v i -> j i")
    offset = joints[0].clone()
    offset[1] = verts[0, :, 1].min()

    # 先减去 offset 再算旋转
    verts_centered = verts - offset
    joints_for_rot = einops_einsum(J_regressor, verts_centered, "j v, l v i -> l j i")
    T_ay2ayfz = compute_T_ayfz2ay(joints_for_rot, inverse=True)

    return offset, T_ay2ayfz


def apply_face_z_transform(verts_frame, offset, T_ay2ayfz):
    """
    用给定的变换参数对齐单帧 vertices。

    Args:
        verts_frame: (V, 3) tensor
        offset: (3,) tensor — from compute_face_z_transform
        T_ay2ayfz: (1, 4, 4) tensor — from compute_face_z_transform
    Returns:
        verts: (V, 3) numpy array
    """
    from hmr4d.utils.geo_transform import apply_T_on_points

    device = offset.device
    verts = verts_frame.clone().unsqueeze(0).to(device)
    verts = verts - offset
    verts = apply_T_on_points(verts, T_ay2ayfz)
    return verts[0].cpu().numpy()


# ============================================================
# 主绘图函数
# ============================================================
def create_figure(seq_name, seq_dir, gt_data, pred_data_list, method_labels,
                  t1, t2, frame_range, output_path, num_image_frames=5,
                  smpl_model=None, device='cuda'):

    n_methods = len(method_labels)
    fstart, fend = frame_range

    lower_faces, lower_body_mask = get_lower_body_faces(smpl_model)

    # --- 膝关节角度 ---
    gt_angles = extract_knee_flexion(gt_data['poses_body'], 'left_knee')
    pred_angles_list = [extract_knee_flexion(p['poses_body'], 'left_knee')
                        for p in pred_data_list]

    # --- 方法颜色 ---
    palette = METHOD_COLORS_HEX['palette']
    method_colors = [palette[i % len(palette)] for i in range(n_methods)]

    # --- 选择图像帧 ---
    frame_indices = np.linspace(fstart, fend - 1, num_image_frames).astype(int)
    idx_t1 = np.argmin(np.abs(frame_indices - t1))
    idx_t2 = np.argmin(np.abs(frame_indices - t2))
    if idx_t1 == idx_t2:
        idx_t2 = min(idx_t1 + 1, num_image_frames - 1)
    frame_indices[idx_t1] = t1
    frame_indices[idx_t2] = t2

    # --- 加载图像 ---
    img_dir = os.path.join(seq_dir, 'images')
    images = []
    for fi in frame_indices:
        img_path = os.path.join(img_dir, f'{fi:05d}.jpg')
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        images.append(img)

    # --- 计算全序列 vertices 并做 Y-up 修正 ---
    print("Computing GT vertices...")
    gt_verts_seq = compute_sequence_vertices(
        smpl_model, gt_data['poses_root'], gt_data['poses_body'],
        gt_data['betas'], gt_data['trans'])
    gt_verts_seq = ensure_y_up(gt_verts_seq)

    pred_verts_seqs = []
    for i, pred in enumerate(pred_data_list):
        print(f"Computing {method_labels[i]} vertices...")
        pred_verts = compute_sequence_vertices(
            smpl_model, pred['poses_root'], pred['poses_body'],
            pred['betas'], pred['trans'])
        pred_verts = ensure_y_up(pred_verts)
        pred_verts_seqs.append(pred_verts)

    # --- 渲染下半身 mesh（各方法独立 face-z 对齐） ---
    def render_all_at_frame(frame_idx):
        renders = []
        # GT
        gt_offset, gt_T = compute_face_z_transform(gt_verts_seq[frame_idx])
        verts = apply_face_z_transform(gt_verts_seq[frame_idx], gt_offset, gt_T)
        renders.append(render_lower_body_pytorch3d(
            verts, lower_faces, GT_MESH_COLOR, device=device))

        # 各方法（各自独立 face-z 对齐）
        for i, pred_verts in enumerate(pred_verts_seqs):
            pred_offset, pred_T = compute_face_z_transform(pred_verts[frame_idx])
            verts = apply_face_z_transform(pred_verts[frame_idx], pred_offset, pred_T)
            color = MESH_COLORS_RGB[i % len(MESH_COLORS_RGB)]
            renders.append(render_lower_body_pytorch3d(
                verts, lower_faces, color, device=device))
        return renders

    print(f"Rendering lower body meshes at t1={t1}...")
    renders_t1 = render_all_at_frame(t1)
    print(f"Rendering lower body meshes at t2={t2}...")
    renders_t2 = render_all_at_frame(t2)

    # ============================================================
    # 绘图
    # ============================================================
    n_mesh_cols = 1 + n_methods  # GT + methods
    fig_width = max(num_image_frames * 2.5, n_mesh_cols * 2 * 2.0)
    fig_height = 12

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1.5, 1.2], hspace=0.15)

    # ---- Row 1: 图像序列 ----
    gs_images = gridspec.GridSpecFromSubplotSpec(
        1, num_image_frames, subplot_spec=gs[0], wspace=0.05)
    for j in range(num_image_frames):
        ax = fig.add_subplot(gs_images[0, j])
        ax.imshow(images[j])
        ax.axis('off')

        fi = frame_indices[j]
        # if fi == t1:
        #     ax.set_title(r'$t_1$', fontsize=16, fontweight='bold')
        # elif fi == t2:
        #     ax.set_title(r'$t_2$', fontsize=16, fontweight='bold')

    # ---- Row 2: 膝关节角度曲线 ----
    ax_curve = fig.add_subplot(gs[1])
    frames = np.arange(fstart, fend)

    ax_curve.plot(frames, gt_angles[fstart:fend],
                  color=METHOD_COLORS_HEX['GT'], linewidth=2.5,
                  label='GT', alpha=0.8)

    for i, (angles, label) in enumerate(zip(pred_angles_list, method_labels)):
        ax_curve.plot(frames, angles[fstart:fend],
                      color=method_colors[i], linewidth=2.0,
                      label=label, alpha=0.9)

    ax_curve.axvline(x=t1, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax_curve.axvline(x=t2, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ymin, ymax = ax_curve.get_ylim()
    # ax_curve.text(t1, ymin - (ymax - ymin) * 0.08, r'$t_1$',
    #               ha='center', fontsize=13, fontweight='bold')
    # ax_curve.text(t2, ymin - (ymax - ymin) * 0.08, r'$t_2$',
    #               ha='center', fontsize=13, fontweight='bold')

    ax_curve.set_ylabel('Left Knee Flexion (°)', fontsize=14)
    ax_curve.set_xlim(fstart, fend)
    ax_curve.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['right'].set_visible(False)

    # ---- Row 3: 下半身 mesh ----
    total_mesh_cols = n_mesh_cols * 2 + 1  # +1 for gap between t1 and t2
    gs_mesh = gridspec.GridSpecFromSubplotSpec(
        1, total_mesh_cols, subplot_spec=gs[2], wspace=0.001)

    all_labels = ['GT'] + method_labels
    all_colors_hex = [METHOD_COLORS_HEX['GT']] + method_colors

    for k, renders in enumerate([renders_t1, renders_t2]):
        col_offset = k * (n_mesh_cols + 1)
        for j, render_img in enumerate(renders):
            ax = fig.add_subplot(gs_mesh[0, col_offset + j])
            ax.imshow(render_img)
            ax.axis('off')
            ax.set_title(all_labels[j], fontsize=14,
                         color=all_colors_hex[j], fontweight='bold')

    plt.suptitle(f'Left Knee Flexion Comparison — {seq_name}',
                 fontsize=16, fontweight='bold', y=0.98)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_kwargs = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    if not output_path.lower().endswith('.svg'):
        save_kwargs['dpi'] = 150
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    print(f"Figure saved to: {output_path}")


def create_minimal_figure(seq_name, gt_data, pred_data_list, method_labels,
                         t1, t2, frame_range, output_path):
    """纯净模式：只绘制角度曲线，宽度随帧数自适应。"""
    
    n_methods = len(method_labels)
    fstart, fend = frame_range
    n_frames = fend - fstart
    
    # 膝关节角度
    gt_angles = extract_knee_flexion(gt_data['poses_body'], 'left_knee')
    pred_angles_list = [extract_knee_flexion(p['poses_body'], 'left_knee')
                        for p in pred_data_list]
    
    # 方法颜色
    palette = METHOD_COLORS_HEX['palette']
    method_colors = [palette[i % len(palette)] for i in range(n_methods)]
    
    # 根据帧数自适应宽度：每100帧约4英寸，最小8英寸
    fig_width = max(8.0, n_frames / 100 * 4.0)
    fig_height = 5.0
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    frames = np.arange(fstart, fend)
    
    # 绘制 GT
    ax.plot(frames, gt_angles[fstart:fend],
            color=METHOD_COLORS_HEX['GT'], linewidth=2.5,
            label='GT', alpha=0.8)
    
    # 绘制各方法
    for i, (angles, label) in enumerate(zip(pred_angles_list, method_labels)):
        ax.plot(frames, angles[fstart:fend],
                color=method_colors[i], linewidth=2.0,
                label=label, alpha=0.9)
    
    # t1, t2 标记线
    ax.axvline(x=t1, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvline(x=t2, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # 标签
    ax.set_xlabel('Frame', fontsize=13)
    ax.set_ylabel('Left Knee Flexion (°)', fontsize=13)
    ax.set_xlim(fstart, fend)
    ax.set_title(f'{seq_name} — Left Knee Flexion',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_kwargs = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    if not output_path.lower().endswith('.svg'):
        save_kwargs['dpi'] = 150
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    print(f"Minimal figure saved to: {output_path}")



# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='WHAM Figure4 风格可视化: 左膝屈曲角度对比')

    parser.add_argument('--seq', type=str, required=True,
                        help='EMDB2 序列名 (e.g. 29_outdoor_stairs_up)')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['GVHMR:results/gvhmr_base_warmstart',
                                 'PromptHMR:results/promptbase_video_warmstart_emdb2'],
                        help='方法列表，格式: Label:result_dir (result_dir 可为绝对路径或相对路径)')
    parser.add_argument('--t1', type=int, required=True,
                        help='第一个关键时刻帧号')
    parser.add_argument('--t2', type=int, required=True,
                        help='第二个关键时刻帧号')
    parser.add_argument('--frame_range', type=int, nargs=2, default=None,
                        help='曲线帧范围 [start end]')
    parser.add_argument('--num_image_frames', type=int, default=5,
                        help='上方图像帧数量')
    parser.add_argument('--dataset_path', type=str, default='datasets/EMDB')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--minimal', action='store_true',
                        help='纯净模式：只绘制角度曲线，宽度自适应帧数')

    args = parser.parse_args()

    # 解析 methods: "Label:path" 格式
    method_labels = []
    method_dirs = []
    for spec in args.methods:
        if ':' in spec:
            label, path = spec.split(':', 1)
        else:
            label = os.path.basename(spec)
            path = spec
        method_labels.append(label)
        method_dirs.append(path)

    if args.output is None:
        args.output = f'figures/knee_flexion_{args.seq}.svg'

    print(f"Loading GT data for {args.seq}...")
    seq_dir, ann_file = find_emdb_sequence(args.dataset_path, args.seq)
    gt_data = load_gt_data(ann_file)

    pred_data_list = []
    for label, result_dir in zip(method_labels, method_dirs):
        print(f"Loading predictions from {label} ({result_dir})...")
        pred = load_pred_data(result_dir, args.seq)
        pred_data_list.append(pred)

    n_frames = gt_data['n_frames']
    if args.frame_range is None:
        args.frame_range = [0, n_frames]
    else:
        # 支持 FRAME_END=-1 表示使用数据最后一帧
        if args.frame_range[1] == -1:
            args.frame_range[1] = n_frames
    args.frame_range[1] = min(args.frame_range[1], n_frames)

    assert args.frame_range[0] <= args.t1 < args.frame_range[1]
    assert args.frame_range[0] <= args.t2 < args.frame_range[1]

    if args.minimal:
        # 纯净模式：只需要角度数据，不需要 SMPL 模型和渲染
        create_minimal_figure(
            seq_name=args.seq,
            gt_data=gt_data,
            pred_data_list=pred_data_list,
            method_labels=method_labels,
            t1=args.t1,
            t2=args.t2,
            frame_range=args.frame_range,
            output_path=args.output,
        )
    else:
        # 完整模式：需要 SMPL 模型和渲染
        print("Loading SMPL model...")
        smpl_model = SMPL(gender=gt_data['gender'])

        create_figure(
            seq_name=args.seq,
            seq_dir=seq_dir,
            gt_data=gt_data,
            pred_data_list=pred_data_list,
            method_labels=method_labels,
            t1=args.t1,
            t2=args.t2,
            frame_range=args.frame_range,
            output_path=args.output,
            num_image_frames=args.num_image_frames,
            smpl_model=smpl_model,
            device=args.device,
        )


if __name__ == '__main__':
    main()
