#!/usr/bin/env python3
"""
相机轨迹 2D/3D 对比可视化

上下两行排列：
  上：3D 轨迹对比（XYZ）
  下：2D 轨迹对比（XY 平面）

GT 从 EMDB pkl 的 extrinsics (w2c) 转为 c2w 位置。
预测从 camera.npz 的 T (c2w 位置) 读取。
使用与 ATE 评估相同的 Sim(3) Umeyama 对齐（evo 库）。

Usage:
    python lib/scripts/visualize_trajectory.py \
        --seq 29_outdoor_stairs_up \
        --methods "Ours:results/gvhmr_base_warmstart" \
                  "PromptHMR:results/promptbase_video_warmstart_emdb2" \
        --output figures/trajectory/29_outdoor_stairs_up.svg
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgba
from glob import glob


# ============================================================
# 颜色
# ============================================================
GT_COLOR = '#888888'
METHOD_PALETTE = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']


# ============================================================
# Sim(3) Umeyama 对齐 (与 evo 库 align=True, correct_scale=True 等价)
# ============================================================
def umeyama_alignment(x, y, with_scale=True):
    """
    Umeyama alignment: 求 s, R, t 使得 y ≈ s*R*x + t
    x: (N, 3) 待对齐轨迹 (pred)
    y: (N, 3) 参考轨迹 (GT)
    返回对齐后的 x_aligned (N, 3)
    """
    assert x.shape == y.shape
    n, d = x.shape

    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    x0 = x - mx
    y0 = y - my

    sx = np.sqrt(np.sum(x0 ** 2) / n)
    sy = np.sqrt(np.sum(y0 ** 2) / n)

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

    x_aligned = s * (R @ x.T).T + t
    return x_aligned


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


def load_gt_camera_traj(ann_file):
    """从 EMDB pkl 加载 GT 相机轨迹 (c2w 位置)"""
    with open(ann_file, 'rb') as f:
        ann = pickle.load(f)

    ext = ann['camera']['extrinsics']  # (N, 4, 4) or (N, 3, 4), w2c
    R_w2c = ext[:, :3, :3]            # (N, 3, 3)
    t_w2c = ext[:, :3, 3]             # (N, 3)
    # c2w position: t_c2w = -R_w2c^T @ t_w2c
    R_c2w = R_w2c.transpose(0, 2, 1)  # (N, 3, 3)
    t_c2w = np.einsum('bij,bj->bi', R_c2w, -t_w2c)  # (N, 3)
    return t_c2w


def load_pred_camera_traj(result_dir, seq_name):
    """从 camera.npz 加载预测相机轨迹 (c2w 位置)"""
    cam_file = os.path.join(result_dir, seq_name, 'camera.npz')
    if not os.path.exists(cam_file):
        raise FileNotFoundError(f'Camera file not found: {cam_file}')
    data = np.load(cam_file)
    t_c2w = data['T']  # (N, 3), c2w position
    return t_c2w


# ============================================================
# 绘图
# ============================================================
def create_figure(seq_name, gt_traj, pred_trajs, method_labels, output_path,
                  frame_range=None):
    n_methods = len(method_labels)
    colors = [METHOD_PALETTE[i % len(METHOD_PALETTE)] for i in range(n_methods)]

    N = gt_traj.shape[0]
    if frame_range is not None:
        fstart, fend = frame_range
        fend = min(fend, N)
    else:
        fstart, fend = 0, N

    gt = gt_traj[fstart:fend]

    # Sim(3) 对齐 (与 ATE 评估一致: align=True, correct_scale=True)
    # 然后平移使起点对齐到 GT 起点
    aligned_preds = []
    for pred in pred_trajs:
        p = pred[fstart:fend]
        min_len = min(len(gt), len(p))
        p_aligned = umeyama_alignment(p[:min_len], gt[:min_len], with_scale=True)
        aligned_preds.append(p_aligned)

    min_len = min(len(gt), min(len(a) for a in aligned_preds))
    gt = gt[:min_len]
    aligned_preds = [a[:min_len] for a in aligned_preds]

    # 平移：以 GT 起点为原点
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
    ax3d.set_title('3D Camera Trajectory', fontsize=13, fontweight='bold')
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
    ax2d.set_title('2D Camera Trajectory (XY Plane)', fontsize=13, fontweight='bold')
    ax2d.set_aspect('equal', adjustable='datalim')
    ax2d.grid(linewidth=0.4, linestyle='--', alpha=0.6)
    ax2d.tick_params(axis='both', labelsize=9)
    ax2d.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2d.yaxis.set_major_locator(ticker.MultipleLocator(2))

    plt.suptitle(f'Camera Trajectory — {seq_name}', fontsize=14, fontweight='bold', y=0.98)
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
    parser = argparse.ArgumentParser(description='相机轨迹 2D/3D 对比可视化')
    parser.add_argument('--seq', type=str, required=True,
                        help='EMDB2 序列名')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['Ours:results/gvhmr_base_warmstart',
                                 'PromptHMR:results/promptbase_video_warmstart_emdb2'],
                        help='方法列表，格式: Label:result_dir')
    parser.add_argument('--frame_range', type=int, nargs=2, default=None,
                        help='帧范围 [start end]')
    parser.add_argument('--dataset_path', type=str, default='datasets/EMDB')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if args.output is None:
        args.output = f'figures/trajectory/{args.seq}.svg'

    # 解析 methods
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

    print(f"Loading GT camera trajectory for {args.seq}...")
    _, ann_file = find_emdb_sequence(args.dataset_path, args.seq)
    gt_traj = load_gt_camera_traj(ann_file)

    pred_trajs = []
    for label, result_dir in zip(method_labels, method_dirs):
        print(f"Loading {label} camera from {result_dir}...")
        pred_t = load_pred_camera_traj(result_dir, args.seq)
        pred_trajs.append(pred_t)

    print("Creating figure...")
    create_figure(
        seq_name=args.seq,
        gt_traj=gt_traj,
        pred_trajs=pred_trajs,
        method_labels=method_labels,
        output_path=args.output,
        frame_range=args.frame_range,
    )


if __name__ == '__main__':
    main()
