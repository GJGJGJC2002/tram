"""
从 pipeline 结果文件夹直接读取 smpl 和 camera 结果，
按官方 run_eval.py 的跨序列帧级加权方式评估。

支持三种模型格式:
  - TRAM/VIMO: smpl.npz 含 rotmat (24,3,3), betas, trans
  - PromptHMR: smpl.npz 含 global_orient_c (3), body_pose_aa (63), betas, trans
  - GVHMR:     同 PromptHMR 格式

用法:
    cd /home/gejunchen/Work/2026-1/Projects/tram
    python scripts/eval_from_results.py \
        --results_dir results/emdb_basic \
        --dataset_path datasets/EMDB \
        --split 2
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob
from tqdm import tqdm
from collections import defaultdict

from lib.models.smpl import SMPL
from lib.utils.eval_utils import *
from lib.utils.rotation_conversions import *
from lib.vis.traj import traj_filter
from lib.camera.slam_utils import eval_slam


def load_emdb_sequences(dataset_path, split):
    """加载 EMDB 数据集序列，返回 {seq_name: root_path} 字典"""
    roots = []
    for p in range(10):
        folder = f'{dataset_path}/P{p}'
        if not os.path.exists(folder):
            continue
        root = sorted(glob(f'{folder}/*'))
        roots.extend(root)

    emdb = {}
    for root in roots:
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        if not os.path.exists(annfile):
            continue
        ann = pkl.load(open(annfile, 'rb'))
        if ann.get(f'emdb{split}', False):
            seq = root.split('/')[-1]
            emdb[seq] = root
    return emdb


def detect_format(pred_smpl):
    """检测 smpl.npz 的格式: 'tram' (有 rotmat) 或 'smplx' (有 global_orient_c)"""
    if 'rotmat' in pred_smpl:
        return 'tram'
    elif 'global_orient_c' in pred_smpl:
        return 'smplx'
    else:
        raise ValueError(f"无法识别 smpl.npz 格式, 键: {list(pred_smpl.keys())}")


def load_smplx_models(gvhmr_root):
    """加载 SMPL-X supermotion 模型和转换矩阵"""
    gvhmr_abs = os.path.abspath(gvhmr_root)
    if gvhmr_abs not in sys.path:
        sys.path.insert(0, gvhmr_abs)

    from hmr4d.utils.smplx_utils import make_smplx

    smplx_model = make_smplx("supermotion")
    smplx_model.eval()
    smplx2smpl = torch.load(
        os.path.join(gvhmr_abs, "hmr4d/utils/body_model/smplx2smpl_sparse.pt"),
        weights_only=True
    )
    J_regressor = torch.load(
        os.path.join(gvhmr_abs, "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt"),
        weights_only=True
    )
    print(f"已加载 SMPL-X supermotion 模型 + smplx2smpl + J_regressor")
    return smplx_model, smplx2smpl, J_regressor


def compute_pred_tram(pred_smpl, pred_cam_data, smpls):
    """TRAM/VIMO 格式: rotmat + camera c2w"""
    pred_rotmat = torch.tensor(pred_smpl['rotmat']).float()
    pred_shape = torch.tensor(pred_smpl['betas']).float()
    pred_trans = torch.tensor(pred_smpl['trans']).float()

    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.repeat(len(pred_shape), 1)

    pred = smpls['neutral'](body_pose=pred_rotmat[:, 1:],
                            global_orient=pred_rotmat[:, [0]],
                            betas=pred_shape,
                            transl=pred_trans.squeeze(),
                            pose2rot=False,
                            default_smpl=True)
    pred_vert = pred.vertices
    pred_j3d = pred.joints[:, :24]
    root_rotmat = pred_rotmat[:, 0]  # (N, 3, 3)

    pred_camr = torch.tensor(pred_cam_data['R']).float()
    pred_camt = torch.tensor(pred_cam_data['T']).float()

    pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:, None]
    pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:, None]
    pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, root_rotmat)

    pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)

    return pred_j3d, pred_vert, pred_j3d_w, pred_vert_w, pred_ori_w, pred_camr, pred_camt


def compute_pred_smplx(pred_smpl, pred_cam_data, smplx_model, smplx2smpl, J_regressor):
    """PromptHMR/GVHMR 格式: global_orient_c + body_pose_aa, 用 SMPL-X forward + camera c2w"""
    global_orient_c = torch.tensor(pred_smpl['global_orient_c']).float()  # (N, 3)
    body_pose_aa = torch.tensor(pred_smpl['body_pose_aa']).float()       # (N, 63)
    pred_shape = torch.tensor(pred_smpl['betas']).float()                # (N, 10)
    pred_trans = torch.tensor(pred_smpl['trans']).float()                # (N, 3)

    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.repeat(len(pred_shape), 1)

    root_rotmat = axis_angle_to_matrix(global_orient_c)  # (N, 3, 3)

    # SMPL-X forward (camera coordinate system)
    with torch.no_grad():
        smplx_out = smplx_model(
            global_orient=global_orient_c,
            body_pose=body_pose_aa,
            betas=pred_shape,
            transl=pred_trans.squeeze(),
        )
        # SMPL-X vertices (10475) -> SMPL vertices (6890)
        pred_vert = torch.stack(
            [torch.matmul(smplx2smpl, v) for v in smplx_out.vertices]
        )
        # SMPL vertices -> 24 joints
        pred_j3d = torch.matmul(J_regressor, pred_vert)

    # camera c2w
    pred_camr = torch.tensor(pred_cam_data['R']).float()
    pred_camt = torch.tensor(pred_cam_data['T']).float()

    pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:, None]
    pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:, None]
    pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, root_rotmat)

    pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)

    return pred_j3d, pred_vert, pred_j3d_w, pred_vert_w, pred_ori_w, pred_camr, pred_camt


def main():
    parser = argparse.ArgumentParser(description='跨序列帧级加权评估（与官方 run_eval.py 一致）')
    # /mnt/storage/users/jcge_data/Work/2026-1/Projects/tram/results/emdb_basic
    parser.add_argument('--results_dir', type=str, default='/mnt/storage/users/jcge_data/Work/2026-1/Projects/tram/results/emdb_basic',
                        help='Pipeline 输出的结果目录')
    parser.add_argument('--dataset_path', type=str, default='datasets/EMDB',
                        help='EMDB 数据集路径')
    parser.add_argument('--split', type=int, default=2, help='EMDB split (1 or 2)')
    parser.add_argument('--gvhmr_root', type=str, default='thirdparty/GVHMR',
                        help='GVHMR 项目根目录（PromptHMR/GVHMR 格式需要）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 Excel 路径（默认: {results_dir}/cross_seq_eval.xlsx）')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.results_dir, 'cross_seq_eval.xlsx')

    # 加载 EMDB 序列
    emdb_seqs = load_emdb_sequences(args.dataset_path, args.split)
    print(f"EMDB split {args.split}: 共 {len(emdb_seqs)} 个序列")

    # 找出 results_dir 中有结果的序列
    result_subdirs = [d for d in os.listdir(args.results_dir)
                      if os.path.isdir(os.path.join(args.results_dir, d))
                      and os.path.exists(os.path.join(args.results_dir, d, 'smpl.npz'))
                      and os.path.exists(os.path.join(args.results_dir, d, 'camera.npz'))]

    # 取交集
    valid_seqs = sorted([s for s in result_subdirs if s in emdb_seqs])
    print(f"结果目录中找到 {len(result_subdirs)} 个序列，与 EMDB split {args.split} 交集: {len(valid_seqs)} 个")

    if len(valid_seqs) == 0:
        print("错误: 没有可评估的序列！")
        print(f"  结果目录序列: {sorted(result_subdirs)}")
        print(f"  EMDB 序列: {sorted(emdb_seqs.keys())}")
        return

    for s in valid_seqs:
        print(f"  - {s}")

    # 检测格式（用第一个序列）
    first_smpl = dict(np.load(os.path.join(args.results_dir, valid_seqs[0], 'smpl.npz')))
    fmt = detect_format(first_smpl)
    print(f"\n检测到结果格式: {fmt}")

    # SMPL 模型（GT 和 TRAM 格式需要）
    smpls = {g: SMPL(gender=g) for g in ['neutral', 'male', 'female']}

    # SMPL-X 模型（PromptHMR/GVHMR 格式需要）
    smplx_model, smplx2smpl, J_regressor_smplx = None, None, None
    if fmt == 'smplx':
        smplx_model, smplx2smpl, J_regressor_smplx = load_smplx_models(args.gvhmr_root)

    # 跨序列帧级累加器（与官方一致）
    accumulator = defaultdict(list)
    per_seq_results = []
    m2mm = 1e3

    # 相机评估结果
    cam_results = {}

    for seq in tqdm(valid_seqs, desc="Evaluating"):
        root = emdb_seqs[seq]

        # === 加载 GT ===
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        ann = pkl.load(open(annfile, 'rb'))

        ext = ann['camera']['extrinsics']
        intr = ann['camera']['intrinsics']
        valid = ann['good_frames_mask']
        gender = ann['gender']
        poses_body = ann["smpl"]["poses_body"]
        poses_root = ann["smpl"]["poses_root"]
        betas = np.repeat(ann["smpl"]["betas"].reshape((1, -1)), repeats=ann["n_frames"], axis=0)
        trans = ann["smpl"]["trans"]

        tt = lambda x: torch.from_numpy(x).float()

        # GT 世界坐标系
        gt = smpls[gender](body_pose=tt(poses_body), global_orient=tt(poses_root),
                           betas=tt(betas), transl=tt(trans),
                           pose2rot=True, default_smpl=True)
        gt_vert = gt.vertices
        gt_ori = axis_angle_to_matrix(tt(poses_root))

        # GT 相机坐标系
        poses_root_cam = matrix_to_axis_angle(tt(ext[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root)))
        gt_cam = smpls[gender](body_pose=tt(poses_body), global_orient=poses_root_cam,
                               betas=tt(betas), pose2rot=True, default_smpl=True)
        gt_vert_cam = gt_cam.vertices

        # GT joints: SMPLX 格式需要用同一个 J_regressor（与 GVHMR 官方评估一致）
        if fmt == 'smplx':
            gt_j3d = torch.matmul(J_regressor_smplx, gt_vert)
            gt_j3d_cam = torch.matmul(J_regressor_smplx, gt_vert_cam)
        else:
            gt_j3d = gt.joints[:, :24]
            gt_j3d_cam = gt_cam.joints[:, :24]

        # === 加载 Pipeline 预测结果 ===
        pred_smpl = dict(np.load(os.path.join(args.results_dir, seq, 'smpl.npz')))
        pred_cam_data = dict(np.load(os.path.join(args.results_dir, seq, 'camera.npz')))

        # 根据格式计算预测的 joints/vertices/world
        if fmt == 'tram':
            pred_j3d, pred_vert, pred_j3d_w, pred_vert_w, pred_ori_w, pred_camr, pred_camt = \
                compute_pred_tram(pred_smpl, pred_cam_data, smpls)
        else:  # smplx (PromptHMR / GVHMR)
            pred_j3d, pred_vert, pred_j3d_w, pred_vert_w, pred_ori_w, pred_camr, pred_camt = \
                compute_pred_smplx(pred_smpl, pred_cam_data, smplx_model, smplx2smpl, J_regressor_smplx)

        # === 应用 valid mask ===
        gt_j3d = gt_j3d[valid]
        gt_ori = gt_ori[valid]
        pred_j3d_w = pred_j3d_w[valid]
        pred_ori_w = pred_ori_w[valid]

        gt_j3d_cam = gt_j3d_cam[valid]
        gt_vert_cam = gt_vert_cam[valid]
        pred_j3d_local = pred_j3d[valid]
        pred_vert_local = pred_vert[valid]

        # === 局部运动评估 ===
        pred_j3d_local, gt_j3d_cam, pred_vert_local, gt_vert_cam = batch_align_by_pelvis(
            [pred_j3d_local, gt_j3d_cam, pred_vert_local, gt_vert_cam], pelvis_idxs=[1, 2]
        )
        S1_hat = batch_compute_similarity_transform_torch(pred_j3d_local, gt_j3d_cam)
        pa_mpjpe = torch.sqrt(((S1_hat - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        mpjpe = torch.sqrt(((pred_j3d_local - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        pve = torch.sqrt(((pred_vert_local - gt_vert_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm

        accel = compute_error_accel(joints_pred=pred_j3d_local.cpu(), joints_gt=gt_j3d_cam.cpu())[1:-1]
        accel = accel * (30 ** 2)

        accumulator['pa_mpjpe'].append(pa_mpjpe)
        accumulator['mpjpe'].append(mpjpe)
        accumulator['pve'].append(pve)
        accumulator['accel'].append(accel)

        # === 全局运动评估（分块） ===
        chunk_length = 100
        w_mpjpe, wa_mpjpe = [], []
        n_valid = valid.sum()
        for start in range(0, n_valid - chunk_length, chunk_length):
            end = start + chunk_length
            if start + 2 * chunk_length > n_valid:
                end = n_valid - 1

            target_j3d = gt_j3d[start:end].clone().cpu()
            pred_j3d_chunk = pred_j3d_w[start:end].clone().cpu()

            w_j3d = first_align_joints(target_j3d, pred_j3d_chunk)
            wa_j3d = global_align_joints(target_j3d, pred_j3d_chunk)

            w_jpe = compute_jpe(target_j3d, w_j3d)
            wa_jpe = compute_jpe(target_j3d, wa_j3d)
            w_mpjpe.append(w_jpe)
            wa_mpjpe.append(wa_jpe)

        if w_mpjpe:
            w_mpjpe = np.concatenate(w_mpjpe) * m2mm
            wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm
        else:
            w_mpjpe = np.array([])
            wa_mpjpe = np.array([])

        # === 全局轨迹评估 ===
        pred_j3d_align = first_align_joints(gt_j3d, pred_j3d_w)
        rte_align_all = compute_rte(gt_j3d[:, 0], pred_j3d_w[:, 0]) * 1e2

        # ERVE
        erve = computer_erve(gt_ori, gt_j3d, pred_ori_w, pred_j3d_w) * m2mm

        accumulator['wa_mpjpe'].append(wa_mpjpe)
        accumulator['w_mpjpe'].append(w_mpjpe)
        accumulator['rte'].append(rte_align_all)
        accumulator['erve'].append(erve)

        # === 相机运动评估 ===
        cam_r = ext[:, :3, :3].transpose(0, 2, 1)
        cam_t = np.einsum('bij, bj->bi', cam_r, -ext[:, :3, -1])
        cam_q = matrix_to_quaternion(torch.from_numpy(cam_r)).numpy()

        pred_camq = matrix_to_quaternion(pred_camr)
        pred_traj = torch.concat([pred_camt, pred_camq], dim=-1).numpy()

        stats_slam, _, _ = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=True)
        stats_metric, traj_ref, traj_est = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=False)

        cam_results[seq] = {
            'ate': stats_slam['mean'],
            'ate_s': stats_metric['mean'],
        }

        # 单序列结果（帧级 mean）
        seq_result = {
            'seq': seq,
            'n_frames': int(n_valid),
            'pa_mpjpe': float(pa_mpjpe.mean()),
            'mpjpe': float(mpjpe.mean()),
            'pve': float(pve.mean()),
            'accel': float(accel.mean()),
            'w_mpjpe': float(w_mpjpe.mean()) if len(w_mpjpe) > 0 else float('nan'),
            'wa_mpjpe': float(wa_mpjpe.mean()) if len(wa_mpjpe) > 0 else float('nan'),
            'rte': float(rte_align_all.mean()),
            'erve': float(erve.mean()),
            'ate': stats_slam['mean'],
            'ate_s': stats_metric['mean'],
        }
        per_seq_results.append(seq_result)

    # === 跨序列帧级加权聚合（与官方 run_eval.py 完全一致） ===
    print("\n" + "=" * 80)
    print("跨序列帧级加权评估结果（与官方 run_eval.py 一致的聚合方式）")
    print("=" * 80)

    cross_seq_metrics = {}
    for k, v in accumulator.items():
        arr = np.concatenate(v)
        cross_seq_metrics[k] = float(arr.mean())

    # ATE/ATE-s 是序列级平均（与官方一致）
    cross_seq_metrics['ate'] = float(np.mean([r['ate'] for r in cam_results.values()]))
    cross_seq_metrics['ate_s'] = float(np.mean([r['ate_s'] for r in cam_results.values()]))

    for k, v in cross_seq_metrics.items():
        print(f"  {k}: {v:.4f}")

    # === 序列级加权（Pipeline 方式，作为对比） ===
    print("\n" + "=" * 80)
    print("序列级加权评估结果（Pipeline 聚合方式，作为对比）")
    print("=" * 80)

    seq_level_metrics = {}
    metric_keys = ['pa_mpjpe', 'mpjpe', 'pve', 'accel', 'w_mpjpe', 'wa_mpjpe', 'rte', 'erve', 'ate', 'ate_s']
    for k in metric_keys:
        vals = [r[k] for r in per_seq_results if not np.isnan(r[k])]
        seq_level_metrics[k] = float(np.mean(vals))
        print(f"  {k}: {seq_level_metrics[k]:.4f}")

    # === 保存结果 ===
    # 1. 每序列结果
    df_per_seq = pd.DataFrame(per_seq_results)

    # 2. 跨序列汇总
    summary_rows = [
        {'aggregation': 'cross_seq_frame_weighted (official)', **cross_seq_metrics},
        {'aggregation': 'per_seq_mean (pipeline)', **seq_level_metrics},
    ]
    df_summary = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_per_seq.to_excel(writer, sheet_name='Per Sequence', index=False)

    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
