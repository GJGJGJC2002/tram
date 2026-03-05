#!/usr/bin/env python3
"""
人体全局轨迹 2D/3D 对比可视化

上下两行排列：
  上：3D pelvis 轨迹对比
  下：2D pelvis 轨迹对比（XY 平面）

GT 从 EMDB pkl 的 SMPL 参数计算全局 pelvis 位置。
预测从 smpl.npz 的世界坐标系参数计算全局 pelvis 位置。
使用前 10% 帧做 Umeyama Sim(3) 对齐，保证起点一致且后续漂移可见。

Usage:
    python lib/scripts/visualize_body_trajectory.py \
        --seq 29_outdoor_stairs_up \
        --methods "Ours:results/gvhmr_base_warmstart" \
                  "PromptHMR:results/promptbase_video_warmstart_emdb2" \
        --output figures/body_trajectory/29_outdoor_stairs_up.svg
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
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgba
from glob import glob

from lib.models.smpl import SMPL


# ============================================================
# 颜色
# ============================================================
GT_COLOR = '#888888'
METHOD_PALETTE = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']


# ============================================================
# Umeyama 对齐
# ============================================================
def umeyama_alignment(x, y, with_scale=True):
    """
    求 s, R, t 使得 y ≈ s*R*x + t
    x: (N, 3) pred,  y: (N, 3) GT
    返回 s, R, t
    """
    assert x.shape == y.shape
    n, d = x.shape

    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    x0 = x - mx
    y0 = y - my

    sx = np.sqrt(np.sum(x0 ** 2) / n)

    W = y0.T @ x0 / n
    U, D, Vt = np.linalg.svd(W)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[d - 1, d - 1] = -1

    R = U @ S @ Vt
    if with_scale:
        s = np.trace(np.diag(D) @ S) / (sx ** 2)
    else:
        s = 1.0
    t = my - s * R @ mx

    return s, R, t


def align_first_percent(pred_traj, gt_traj, percent=0.1, with_scale=True):
    """
    用前 percent 比例的帧做 Umeyama 对齐，将变换应用到全部帧。
    起点自然对齐，后续漂移暴露。
    """
    n = min(len(pred_traj), len(gt_traj))
    n_align = max(2, int(n * percent))

    s, R, t = umeyama_alignment(
        pred_traj[:n_align], gt_traj[:n_align], with_scale=with_scale
    )
    # 应用到全部帧
    aligned = s * (R @ pred_traj[:n].T).T + t
    return aligned


# ============================================================
# 数据加载
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


def _smpl_pelvis(smpl_model, poses_root, poses_body, betas, trans, batch_size=128):
    """批量 SMPL forward，返回 pelvis 轨迹 (N, 3)"""
    tt = lambda x: torch.from_numpy(x).float()
    N = poses_root.shape[0]
    if betas.ndim == 1:
        betas = np.tile(betas, (N, 1))

    bp = poses_body.copy()
    if bp.shape[-1] == 63:
        bp = np.concatenate([bp, np.zeros((N, 6), dtype=np.float32)], axis=1)

    all_pelvis = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        out = smpl_model(
            body_pose=tt(bp[s:e]),
            global_orient=tt(poses_root[s:e]),
            betas=tt(betas[s:e]),
            transl=tt(trans[s:e]),
            pose2rot=True, default_smpl=True,
        )
        all_pelvis.append(out.joints[:, 0].detach().numpy())  # pelvis = joint 0
    return np.concatenate(all_pelvis, axis=0)  # (N, 3)


def load_gt_body_traj(ann_file):
    """从 EMDB pkl 加载 GT pelvis 轨迹"""
    with open(ann_file, 'rb') as f:
        ann = pickle.load(f)

    smpl_model = SMPL(gender=ann['gender'])
    return _smpl_pelvis(
        smpl_model,
        ann['smpl']['poses_root'],
        ann['smpl']['poses_body'],
        ann['smpl']['betas'],
        ann['smpl']['trans'],
    )


def load_pred_body_traj(result_dir, seq_name):
    """从 smpl.npz 加载预测 pelvis 轨迹，用 camera.npz 的 c2w 变换到世界坐标系。

    步骤：
      1. 用相机坐标系下的 SMPL 参数 (global_orient_c + trans) 计算 pelvis_cam
      2. 读取 camera.npz 的 R (c2w 旋转), T (c2w 平移)
      3. pelvis_world = R @ pelvis_cam + T
    """
    smpl_file = os.path.join(result_dir, seq_name, 'smpl.npz')
    cam_file = os.path.join(result_dir, seq_name, 'camera.npz')
    if not os.path.exists(smpl_file):
        raise FileNotFoundError(f'Not found: {smpl_file}')
    if not os.path.exists(cam_file):
        raise FileNotFoundError(f'Not found: {cam_file}')

    data = np.load(smpl_file)
    cam_data = np.load(cam_file)

    # --- 相机坐标系 SMPL 参数 ---
    if 'global_orient_c' in data:
        poses_root = data['global_orient_c']
        poses_body = data['body_pose_aa']
    elif 'rotmat' in data:
        from pytorch3d.transforms import matrix_to_axis_angle
        rotmat = torch.from_numpy(data['rotmat']).float()
        aa = matrix_to_axis_angle(rotmat)
        poses_root = aa[:, 0, :].numpy()
        poses_body = aa[:, 1:, :].reshape(-1, 69).numpy()
    else:
        raise ValueError(f"Unknown smpl format, keys: {list(data.keys())}")

    trans_cam = data['trans']  # (N, 3) 相机坐标系平移

    # --- 计算相机坐标系下的 pelvis ---
    smpl_model = SMPL()
    pelvis_cam = _smpl_pelvis(smpl_model, poses_root, poses_body, data['betas'], trans_cam)
    # pelvis_cam: (N, 3)

    # --- c2w 变换 ---
    R_c2w = cam_data['R']  # (N, 3, 3)
    T_c2w = cam_data['T']  # (N, 3)

    N = min(len(pelvis_cam), len(R_c2w))
    pelvis_cam = pelvis_cam[:N]
    R_c2w = R_c2w[:N]
    T_c2w = T_c2w[:N]

    # pelvis_world = R_c2w @ pelvis_cam + T_c2w
    pelvis_world = np.einsum('bij,bj->bi', R_c2w, pelvis_cam) + T_c2w
    print(f"  -> camera-space pelvis + c2w transform ({N} frames)")
    return pelvis_world


# ============================================================
# 绘图
# ============================================================
def create_figure(seq_name, gt_traj, pred_trajs, method_labels, output_path,
                  frame_range=None, align_percent=0.1):
    n_methods = len(method_labels)
    colors = [METHOD_PALETTE[i % len(METHOD_PALETTE)] for i in range(n_methods)]

    N = gt_traj.shape[0]
    if frame_range is not None:
        fstart, fend = frame_range
        fend = min(fend, N)
    else:
        fstart, fend = 0, N

    gt = gt_traj[fstart:fend]

    # 前 align_percent 帧 Umeyama 对齐，变换应用到全部帧
    aligned_preds = []
    for pred in pred_trajs:
        p = pred[fstart:fend]
        aligned = align_first_percent(p, gt, percent=align_percent, with_scale=True)
        aligned_preds.append(aligned)

    min_len = min(len(gt), min(len(a) for a in aligned_preds))
    gt = gt[:min_len]
    aligned_preds = [a[:min_len] for a in aligned_preds]

    # 以 GT 起点为原点
    origin = gt[0].copy()
    gt = gt - origin
    aligned_preds = [a - origin for a in aligned_preds]

    # 散点采样
    length = len(gt)
    step = max(1, length // 150)
    alpha_arr = np.linspace(0.3, 0.95, len(gt[::step]))

    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(7, 10),
                             gridspec_kw={'height_ratios': [1.2, 1]})

    # ---- 上：3D 轨迹 ----
    ax3d = fig.add_subplot(2, 1, 1, projection='3d')
    axes[0].remove()

    gt_rgba = np.array([to_rgba(GT_COLOR, a) for a in alpha_arr])
    ax3d.scatter(gt[::step, 0], gt[::step, 1], gt[::step, 2],
                 s=8, c=gt_rgba, edgecolors='none', label='GT')
    for i, (traj, label) in enumerate(zip(aligned_preds, method_labels)):
        a = np.linspace(0.3, 0.95, len(traj[::step, 0]))
        rgba = np.array([to_rgba(colors[i], ai) for ai in a])
        ax3d.scatter(traj[::step, 0], traj[::step, 1], traj[::step, 2],
                     s=8, c=rgba, edgecolors='none', label=label)

    ax3d.scatter(*gt[0], c='k', s=60, marker='o', zorder=10)
    ax3d.scatter(*gt[-1], c='k', s=60, marker='x', zorder=10)

    ax3d.set_xlabel('X (m)', fontsize=10)
    ax3d.set_ylabel('Y (m)', fontsize=10)
    ax3d.set_zlabel('Z (m)', fontsize=10)
    ax3d.set_title('3D Body Trajectory', fontsize=13, fontweight='bold')
    ax3d.tick_params(labelsize=8)

    # ---- 下：2D (XY 平面) ----
    ax2d = axes[1]

    ax2d.scatter(gt[::step, 0], gt[::step, 1],
                 s=12, c=GT_COLOR, alpha=alpha_arr, edgecolors='none', label='GT')
    for i, (traj, label) in enumerate(zip(aligned_preds, method_labels)):
        a = np.linspace(0.3, 0.95, len(traj[::step, 0]))
        ax2d.scatter(traj[::step, 0], traj[::step, 1],
                     s=12, c=colors[i], alpha=a, edgecolors='none', label=label)

    ax2d.scatter(gt[0, 0], gt[0, 1], c='k', s=60, marker='o', zorder=10)
    ax2d.scatter(gt[-1, 0], gt[-1, 1], c='k', s=60, marker='x', zorder=10)

    ax2d.set_xlabel('X (m)', fontsize=11)
    ax2d.set_ylabel('Y (m)', fontsize=11)
    ax2d.set_title('2D Body Trajectory (XY Plane)', fontsize=13, fontweight='bold')
    ax2d.set_aspect('equal', adjustable='datalim')
    ax2d.grid(linewidth=0.4, linestyle='--', alpha=0.6)
    ax2d.tick_params(axis='both', labelsize=9)
    ax2d.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2d.yaxis.set_major_locator(ticker.MultipleLocator(2))

    plt.suptitle(f'Body Trajectory — {seq_name}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_kwargs = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    if not output_path.lower().endswith('.svg'):
        save_kwargs['dpi'] = 200
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    print(f"Figure saved to: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='人体全局轨迹 2D/3D 对比可视化')
    parser.add_argument('--seq', type=str, required=True)
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['Ours:results/gvhmr_base_warmstart',
                                 'PromptHMR:results/promptbase_video_warmstart_emdb2'],
                        help='格式: Label:result_dir')
    parser.add_argument('--align_percent', type=float, default=0.1,
                        help='用于对齐的帧比例 (默认 0.1 = 前10%%)')
    parser.add_argument('--frame_range', type=int, nargs=2, default=None)
    parser.add_argument('--dataset_path', type=str, default='datasets/EMDB')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if args.output is None:
        args.output = f'figures/body_trajectory/{args.seq}.svg'

    method_labels, method_dirs = [], []
    for spec in args.methods:
        if ':' in spec:
            label, path = spec.split(':', 1)
        else:
            label = os.path.basename(spec)
            path = spec
        method_labels.append(label)
        method_dirs.append(path)

    print(f"Loading GT body trajectory for {args.seq}...")
    _, ann_file = find_emdb_sequence(args.dataset_path, args.seq)
    gt_traj = load_gt_body_traj(ann_file)

    pred_trajs = []
    for label, result_dir in zip(method_labels, method_dirs):
        print(f"Loading {label} from {result_dir}...")
        pred_trajs.append(load_pred_body_traj(result_dir, args.seq))

    print(f"Creating figure (align first {args.align_percent*100:.0f}% frames)...")
    create_figure(
        seq_name=args.seq,
        gt_traj=gt_traj,
        pred_trajs=pred_trajs,
        method_labels=method_labels,
        output_path=args.output,
        frame_range=args.frame_range,
        align_percent=args.align_percent,
    )


if __name__ == '__main__':
    main()
