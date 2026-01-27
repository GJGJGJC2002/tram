import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import logging
from datetime import datetime

# Model imports
from torch.amp import autocast
from segment_anything import SamPredictor, sam_model_registry
from detectron2.config import LazyConfig
from torch.utils.data import default_collate

# Local imports
from lib.camera import run_metric_slam, align_cam_to_world
from lib.pipeline.tools import arrange_boxes
from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
from lib.models import get_hmr_vimo
from lib.datasets.image_dataset import ImageDataset
from lib.utils.eval_utils import *
from lib.utils.rotation_conversions import *
from lib.vis.traj import *
from lib.camera.slam_utils import eval_slam


def setup_logging(output_dir):
    """设置日志系统"""
    log_file = os.path.join(output_dir, 'processing.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_emdb_sequences(dataset_path, split, seq_name=None, person=None):
    """
    加载 EMDB 数据集序列
    
    Args:
        dataset_path: EMDB 数据集根目录
        split: split 编号 (1, 2, 或 3)
        seq_name: 指定序列名称，如 '01_mvs_b'，为 None 则加载所有
        person: 指定人物编号，如 'P0'，为 None 则加载所有
    
    Returns:
        list: 符合条件的序列路径列表
    """
    roots = []
    
    if person is not None:
        # 只处理指定人物
        person_ids = [int(person[1:])] if person.startswith('P') else [int(person)]
    else:
        # 处理所有人物
        person_ids = range(10)
    
    for p in person_ids:
        folder = f'{dataset_path}/P{p}'
        if not os.path.exists(folder):
            continue
        root = sorted(glob(f'{folder}/*'))
        roots.extend(root)
    
    # 根据 split 筛选
    emdb = []
    for root in roots:
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        if not os.path.exists(annfile):
            continue
        
        ann = pkl.load(open(annfile, 'rb'))
        if ann.get(f'emdb{split}', False):
            # 如果指定了序列名，只添加匹配的
            if seq_name is None or root.split('/')[-1] == seq_name:
                emdb.append(root)
    
    return emdb


def initialize_models(device='cuda'):
    """
    初始化所有需要的模型
    
    Returns:
        dict: 包含所有模型的字典
    """
    models = {}
    
    # ViTDet for detection
    cfg_path = 'data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    models['detector'] = DefaultPredictor_Lazy(detectron2_cfg)
    
    # SAM for segmentation
    sam = sam_model_registry["vit_h"](checkpoint="data/pretrain/sam_vit_h_4b8939.pth")
    _ = sam.to(device)
    models['sam_predictor'] = SamPredictor(sam)
    
    # HMR-VIMO for human pose estimation
    models['hmr_vimo'] = get_hmr_vimo(checkpoint='data/pretrain/vimo_checkpoint.pth.tar').to(device)
    
    return models


def estimate_camera(root, models, output_dir, logger, device='cuda'):
    """
    估计相机运动
    
    Args:
        root: 视频序列根目录
        models: 模型字典
        output_dir: 输出目录
        logger: 日志记录器
        device: 设备
    
    Returns:
        dict: 相机参数字典，如果失败则返回 None
    """
    seq = root.split('/')[-1]
    savefile = f'{output_dir}/camera/{seq}.npz'
    
    try:
        logger.info(f'Estimating camera motion for {seq}...')
        
        img_folder = f'{root}/images'
        imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
        
        if len(imgfiles) == 0:
            logger.error(f'No images found in {img_folder}')
            return None
        
        # Detection and segmentation
        masks_ = []
        detector = models['detector']
        predictor = models['sam_predictor']
        
        for t, imgpath in enumerate(tqdm(imgfiles, desc='Detection & Segmentation')):
            img_cv2 = cv2.imread(imgpath)
            
            # Detection
            with torch.no_grad():
                with autocast('cuda'):
                    det_out = detector(img_cv2)
                    det_instances = det_out['instances']
                    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
                    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                    confs = det_instances.scores[valid_idx].cpu().numpy()
                    
                    boxes = np.hstack([boxes, confs[:, None]])
                    boxes = arrange_boxes(boxes, mode='size', min_size=100)
            
            # SAM segmentation
            if len(boxes) > 0:
                with autocast('cuda'):
                    predictor.set_image(img_cv2, image_format='BGR')
                    bb = torch.tensor(boxes[:, :4]).cuda()
                    bb = predictor.transform.apply_boxes_torch(bb, img_cv2.shape[:2])
                    masks, scores, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=bb,
                        multimask_output=False
                    )
                    scores = scores.cpu()
                    masks = masks.cpu().squeeze(1)
                    mask = masks.sum(dim=0)
            else:
                mask = torch.zeros((img_cv2.shape[0], img_cv2.shape[1]))
            
            masks_.append(mask.byte())
        
        masks = torch.stack(masks_)
        
        # Load camera intrinsics
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        ann = pkl.load(open(annfile, 'rb'))
        intr = ann['camera']['intrinsics']
        
        cam_int = [intr[0,0], intr[1,1], intr[0,2], intr[1,2]]
        
        # Run metric SLAM
        logger.info('Running metric SLAM...')
        cam_R, cam_T = run_metric_slam(img_folder, masks=masks, calib=cam_int)
        wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)
        
        camera = {
            'pred_cam_R': cam_R.numpy(), 
            'pred_cam_T': cam_T.numpy(), 
            'world_cam_R': wd_cam_R.numpy(), 
            'world_cam_T': wd_cam_T.numpy(),
            'img_focal': cam_int[0], 
            'img_center': cam_int[2:], 
            'spec_focal': spec_f
        }
        
        # Save results
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        np.savez(savefile, **camera)
        logger.info(f'Camera results saved to {savefile}')
        
        return camera
        
    except Exception as e:
        logger.error(f'Error in camera estimation for {seq}: {str(e)}')
        import traceback
        logger.error(traceback.format_exc())
        return None


def estimate_smpl(root, models, output_dir, logger, efficient=False, device='cuda'):
    """
    估计 SMPL 参数
    
    Args:
        root: 视频序列根目录
        models: 模型字典
        output_dir: 输出目录
        logger: 日志记录器
        efficient: 是否使用快速模式
        device: 设备
    
    Returns:
        dict: SMPL 参数字典，如果失败则返回 None
    """
    seq = root.split('/')[-1]
    savefile = f'{output_dir}/smpl/{seq}.npz'
    
    try:
        logger.info(f'Estimating SMPL for {seq}...')
        
        imgfiles = sorted(glob(f'{root}/images/*.jpg'))
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        ann = pkl.load(open(annfile, 'rb'))
        
        ext = ann['camera']['extrinsics']
        intr = ann['camera']['intrinsics']
        ann_boxes = ann['bboxes']['bboxes']
        img_focal = (intr[0,0] +  intr[1,1]) / 2.
        img_center = intr[:2, 2]
        
        # Create dataset
        db = ImageDataset(imgfiles, ann_boxes, img_focal=img_focal, 
                         img_center=img_center, normalization=True)
        dataloader = torch.utils.data.DataLoader(db, batch_size=64, shuffle=False, num_workers=12)
        
        model = models['hmr_vimo']
        
        # Results
        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []
        
        if efficient:
            logger.info('Using efficient mode (non-overlapping sliding window)')
            for batch in tqdm(dataloader, desc='SMPL Estimation'):
                batch = {k: v.to(device) for k, v in batch.items() if type(v)==torch.Tensor}
                
                # Last batch
                n = len(batch['img'])
                if n < 64:
                    for k in batch:
                        batch[k] = torch.cat([previous_batch[k][n-64:], batch[k]], dim=0)
                
                with torch.no_grad():
                    out, _ = model(batch)
                
                # Last batch
                if n < 64:
                    for k in out:
                        out[k] = out[k][64-n:]
                
                pred_cam.append(out['pred_cam'].cpu())
                pred_pose.append(out['pred_pose'].cpu())
                pred_shape.append(out['pred_shape'].cpu())
                pred_rotmat.append(out['pred_rotmat'].cpu())
                pred_trans.append(out['trans_full'].cpu())
                previous_batch = batch
        else:
            logger.info('Using overlapping sliding window mode (more accurate)')
            items = []
            for i in tqdm(range(len(db)), desc='SMPL Estimation'):
                item = db[i]
                items.append(item)
                
                if len(items) < 16:
                    continue
                elif len(items) == 16:
                    batch = default_collate(items)
                else:
                    items.pop(0)
                    batch = default_collate(items)
                
                with torch.no_grad():
                    batch = {k: v.to(device) for k, v in batch.items() if type(v)==torch.Tensor}
                    out, _ = model.forward(batch)
                
                if i == 15:
                    out = {k:v[:9] for k,v in out.items()}
                elif i == len(db) - 1:
                    out = {k:v[8:] for k,v in out.items()}
                else:
                    out = {k:v[[8]] for k,v in out.items()}
                
                pred_cam.append(out['pred_cam'].cpu())
                pred_pose.append(out['pred_pose'].cpu())
                pred_shape.append(out['pred_shape'].cpu())
                pred_rotmat.append(out['pred_rotmat'].cpu())
                pred_trans.append(out['trans_full'].cpu())
        
        results = {
            'pred_cam': torch.cat(pred_cam),
            'pred_pose': torch.cat(pred_pose),
            'pred_shape': torch.cat(pred_shape),
            'pred_rotmat': torch.cat(pred_rotmat),
            'pred_trans': torch.cat(pred_trans),
            'img_focal': img_focal,
            'img_center': img_center
        }
        
        # Save results
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        np.savez(savefile, **results)
        logger.info(f'SMPL results saved to {savefile}')
        
        return results
        
    except Exception as e:
        logger.error(f'Error in SMPL estimation for {seq}: {str(e)}')
        import traceback
        logger.error(traceback.format_exc())
        return None


def evaluate_sequence(root, output_dir, logger):
    """
    评估单个序列
    
    Args:
        root: 视频序列根目录
        output_dir: 输出目录
        logger: 日志记录器
    
    Returns:
        dict: 评估结果字典
    """
    seq = root.split('/')[-1]
    
    try:
        logger.info(f'Evaluating {seq}...')
        
        # Load GT annotations
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        ann = pkl.load(open(annfile, 'rb'))
        
        ext = ann['camera']['extrinsics']
        intr = ann['camera']['intrinsics']
        img_focal = (intr[0,0] +  intr[1,1]) / 2.
        img_center = intr[:2, 2]
        
        valid = ann['good_frames_mask']
        gender = ann['gender']
        poses_body = ann["smpl"]["poses_body"]
        poses_root = ann["smpl"]["poses_root"]
        betas = np.repeat(ann["smpl"]["betas"].reshape((1, -1)), repeats=ann["n_frames"], axis=0)
        trans = ann["smpl"]["trans"]
        
        # Load SMPL models
        smpls = {g:SMPL(gender=g) for g in ['neutral', 'male', 'female']}
        
        tt = lambda x: torch.from_numpy(x).float()
        gt = smpls[gender](body_pose=tt(poses_body), global_orient=tt(poses_root), betas=tt(betas), transl=tt(trans),
                        pose2rot=True, default_smpl=True)
        gt_vert = gt.vertices
        gt_j3d = gt.joints[:,:24] 
        gt_ori = axis_angle_to_matrix(tt(poses_root))
        
        # Groundtruth local motion
        poses_root_cam = matrix_to_axis_angle(tt(ext[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root)))
        gt_cam = smpls[gender](body_pose=tt(poses_body), global_orient=poses_root_cam, betas=tt(betas),
                               pose2rot=True, default_smpl=True)
        gt_vert_cam = gt_cam.vertices
        gt_j3d_cam = gt_cam.joints[:,:24]
        
        # Load predictions
        pred_cam = dict(np.load(f'{output_dir}/camera/{seq}.npz'))
        pred_smpl = dict(np.load(f'{output_dir}/smpl/{seq}.npz'))
        
        pred_rotmat = torch.tensor(pred_smpl['pred_rotmat'])
        pred_shape = torch.tensor(pred_smpl['pred_shape'])
        pred_trans = torch.tensor(pred_smpl['pred_trans'])
        
        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)
        
        pred = smpls['neutral'](body_pose=pred_rotmat[:,1:], 
                                global_orient=pred_rotmat[:,[0]], 
                                betas=pred_shape, 
                                transl=pred_trans.squeeze(),
                                pose2rot=False, 
                                default_smpl=True)
        pred_vert = pred.vertices
        pred_j3d = pred.joints[:, :24]
        
        pred_camt = torch.tensor(pred_cam['pred_cam_T']) 
        pred_camr = torch.tensor(pred_cam['pred_cam_R'])
       
        pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:,None]
        pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:,None]
        pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, pred_rotmat[:,0])
        pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)
        
        # Valid mask
        gt_j3d = gt_j3d[valid]
        gt_ori = gt_ori[valid]
        pred_j3d_w  = pred_j3d_w[valid]
        pred_ori_w = pred_ori_w[valid]
        
        gt_j3d_cam = gt_j3d_cam[valid]
        gt_vert_cam = gt_vert_cam[valid]
        pred_j3d = pred_j3d[valid]
        pred_vert = pred_vert[valid]
        
        # Evaluation on the local motion
        m2mm = 1e3
        pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam = batch_align_by_pelvis(
            [pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam], pelvis_idxs=[1,2]
        )
        S1_hat = batch_compute_similarity_transform_torch(pred_j3d, gt_j3d_cam)
        pa_mpjpe = torch.sqrt(((S1_hat - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        mpjpe = torch.sqrt(((pred_j3d - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        pve = torch.sqrt(((pred_vert - gt_vert_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        
        accel = compute_error_accel(joints_pred=pred_j3d.cpu(), joints_gt=gt_j3d_cam.cpu())[1:-1]
        accel = accel * (30 ** 2)
        
        # Evaluation on the global motion
        chunk_length = 100
        w_mpjpe, wa_mpjpe = [], []
        for start in range(0, valid.sum() - chunk_length, chunk_length):
            end = start + chunk_length
            if start + 2 * chunk_length > valid.sum(): 
                end = valid.sum() - 1
            
            target_j3d = gt_j3d[start:end].clone().cpu()
            pred_j3d_chunk = pred_j3d_w[start:end].clone().cpu()
            
            w_j3d = first_align_joints(target_j3d, pred_j3d_chunk)
            wa_j3d = global_align_joints(target_j3d, pred_j3d_chunk)
            
            w_jpe = compute_jpe(target_j3d, w_j3d)
            wa_jpe = compute_jpe(target_j3d, wa_j3d)
            w_mpjpe.append(w_jpe)
            wa_mpjpe.append(wa_jpe)
        
        w_mpjpe = np.concatenate(w_mpjpe) * m2mm
        wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm
        
        # Evaluation on the entire global motion
        pred_j3d_align = first_align_joints(gt_j3d, pred_j3d_w)
        rte_align_first= compute_jpe(gt_j3d[:,[0]], pred_j3d_align[:,[0]])
        rte_align_all = compute_rte(gt_j3d[:,0], pred_j3d_w[:,0]) * 1e2 
        
        # ERVE: Ego-centric root velocity error
        erve = computer_erve(gt_ori, gt_j3d, pred_ori_w, pred_j3d_w) * m2mm
        
        # Camera motion evaluation
        cam_r = ext[:,:3,:3].transpose(0,2,1)
        cam_t = np.einsum('bij, bj->bi', cam_r, -ext[:, :3, -1])
        cam_q = matrix_to_quaternion(torch.from_numpy(cam_r)).numpy()
        
        pred_camq = matrix_to_quaternion(pred_camr)
        pred_traj = torch.concat([pred_camt, pred_camq], dim=-1).numpy()
        
        stats_slam, _, _ = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=True)
        stats_metric, traj_ref, traj_est = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=False)
        
        results = {
            'seq': seq,
            'pa_mpjpe': pa_mpjpe.mean(),
            'mpjpe': mpjpe.mean(),
            'pve': pve.mean(),
            'accel': accel.mean(),
            'wa_mpjpe': wa_mpjpe.mean(),
            'w_mpjpe': w_mpjpe.mean(),
            'rte': rte_align_all.mean(),
            'erve': erve.mean(),
            'ate': stats_slam['mean'],
            'ate_s': stats_metric['mean']
        }
        
        logger.info(f'Evaluation results for {seq}:')
        for k, v in results.items():
            if k != 'seq':
                logger.info(f'  {k}: {v:.4f}')
        
        return results
        
    except Exception as e:
        logger.error(f'Error in evaluation for {seq}: {str(e)}')
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(description='Unified EMDB evaluation pipeline')
    
    # Dataset parameters
    parser.add_argument('--seq', type=str, default=None,
                       help='Sequence name (e.g., 01_mvs_b). If None, process all sequences in split')
    parser.add_argument('--person', type=str, default=None,
                       help='Person ID (e.g., P0). If None, process all persons')
    parser.add_argument('--split', type=int, default=2,
                       help='EMDB split number (1, 2, or 3)')
    parser.add_argument('--dataset_path', type=str, 
                       default='/home/gejunchen/Work/2026-1/Datasets/EMDB',
                       help='Path to EMDB dataset')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results/emdb',
                       help='Output directory')
    
    # Processing options
    parser.add_argument('--efficient', action='store_true',
                       help='Use efficient mode for SMPL estimation (faster but less accurate)')
    parser.add_argument('--skip_camera', action='store_true',
                       help='Skip camera estimation (use existing results)')
    parser.add_argument('--skip_smpl', action='store_true',
                       help='Skip SMPL estimation (use existing results)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation (requires existing camera and SMPL results)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/camera', exist_ok=True)
    os.makedirs(f'{args.output_dir}/smpl', exist_ok=True)
    
    logger = setup_logging(args.output_dir)
    logger.info('='*80)
    logger.info('EMDB Unified Evaluation Pipeline')
    logger.info('='*80)
    logger.info(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Arguments: {vars(args)}')
    
    # Load sequences
    logger.info('Loading EMDB sequences...')
    sequences = load_emdb_sequences(args.dataset_path, args.split, args.seq, args.person)
    
    if len(sequences) == 0:
        logger.error('No sequences found matching the criteria!')
        return
    
    logger.info(f'Found {len(sequences)} sequence(s) to process:')
    for seq_path in sequences:
        logger.info(f'  - {seq_path}')
    
    # Initialize models if needed
    models = None
    if not args.eval_only:
        logger.info('Initializing models...')
        models = initialize_models(args.device)
        logger.info('Models initialized successfully')
    
    # Process each sequence
    all_results = []
    for i, root in enumerate(sequences):
        seq = root.split('/')[-1]
        logger.info('')
        logger.info('='*80)
        logger.info(f'Processing sequence {i+1}/{len(sequences)}: {seq}')
        logger.info('='*80)
        
        # Stage 1: Camera estimation
        if not args.skip_camera and not args.eval_only:
            camera_result = estimate_camera(root, models, args.output_dir, logger, args.device)
            if camera_result is None:
                logger.warning(f'Skipping {seq} due to camera estimation failure')
                continue
            # Clear GPU cache
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Stage 2: SMPL estimation
        if not args.skip_smpl and not args.eval_only:
            smpl_result = estimate_smpl(root, models, args.output_dir, logger, 
                                       args.efficient, args.device)
            if smpl_result is None:
                logger.warning(f'Skipping {seq} due to SMPL estimation failure')
                continue
            # Clear GPU cache
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Stage 3: Evaluation
        eval_result = evaluate_sequence(root, args.output_dir, logger)
        if eval_result is not None:
            all_results.append(eval_result)
    
    # Summary
    logger.info('')
    logger.info('='*80)
    logger.info('Evaluation Summary')
    logger.info('='*80)
    
    if len(all_results) > 0:
        # Calculate average metrics
        metrics = defaultdict(list)
        for result in all_results:
            for k, v in result.items():
                if k != 'seq':
                    metrics[k].append(v)
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        
        logger.info('Average metrics across all sequences:')
        for k, v in avg_metrics.items():
            logger.info(f'  {k}: {v:.4f}')
        
        # Save to Excel
        df = pd.DataFrame(all_results)
        excel_file = f'{args.output_dir}/evaluation_results.xlsx'
        df.to_excel(excel_file, index=False)
        logger.info(f'Results saved to {excel_file}')
        
        # Save summary
        summary = pd.DataFrame([avg_metrics])
        summary_file = f'{args.output_dir}/evaluation_summary.xlsx'
        summary.to_excel(summary_file, index=False)
        logger.info(f'Summary saved to {summary_file}')
    else:
        logger.error('No sequences were successfully processed!')
    
    logger.info('='*80)
    logger.info(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('Processing complete!')


if __name__ == '__main__':
    main()

