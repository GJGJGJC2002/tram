"""EvaluationComponent - 评估组件"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch

from lib.pipeline.core.component import Component
from lib.pipeline.core.data import PipelineData, CameraParams, SMPLParams


class EvaluationComponent(Component):
    """
    评估组件
    
    计算各种评估指标：
    - 局部运动: PA-MPJPE, MPJPE, PVE, Accel
    - 全局运动: W-MPJPE, WA-MPJPE, RTE, ERVE
    - 相机运动: ATE
    
    Config:
        metrics: 要计算的指标列表
        chunk_length: 全局运动评估的分块长度
        fps: 视频帧率（用于加速度计算）
    """
    
    COMPONENT_TYPE = "evaluation"
    
    ALL_METRICS = [
        'pa_mpjpe', 'mpjpe', 'pve', 'accel',  # 局部运动
        'w_mpjpe', 'wa_mpjpe', 'rte', 'erve',  # 全局运动
        'ate', 'ate_s'  # 相机运动
    ]
    
    DEFAULT_CONFIG = {
        'metrics': ALL_METRICS,
        'chunk_length': 100,
        'fps': 30,
    }
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)
        
        self.metrics_to_compute = self.config['metrics']
        self.chunk_length = self.config['chunk_length']
        self.fps = self.config['fps']
        
        # SMPL 模型（延迟加载）
        self._smpls = None
    
    def setup(self):
        """初始化 SMPL 模型"""
        from lib.models.smpl import SMPL
        
        self._smpls = {
            gender: SMPL(gender=gender) 
            for gender in ['neutral', 'male', 'female']
        }
        self._is_setup = True
        self.logger.info("Evaluation component initialized")
    
    def validate_input(self, data: PipelineData) -> bool:
        """验证输入：需要预测结果和 GT 标注"""
        has_predictions = data.camera_params is not None and data.smpl_params is not None
        has_gt = data.annotations is not None
        return has_predictions and has_gt
    
    def execute(self, data: PipelineData) -> PipelineData:
        """执行评估"""
        self.logger.info(f"Computing metrics: {self.metrics_to_compute}")
        
        # 加载 GT 数据
        gt_data = self._load_gt_data(data)
        
        # 计算预测的 SMPL 输出
        pred_data = self._compute_pred_smpl(data)
        
        # 应用 valid mask
        valid_mask = data.valid_frames_mask if data.valid_frames_mask is not None else gt_data.get('valid_mask')
        if valid_mask is not None:
            gt_data, pred_data = self._apply_valid_mask(gt_data, pred_data, valid_mask)
        
        metrics = {}
        m2mm = 1e3  # 米转毫米
        
        # === 局部运动评估 ===
        if any(m in self.metrics_to_compute for m in ['pa_mpjpe', 'mpjpe', 'pve']):
            local_metrics = self._evaluate_local_motion(gt_data, pred_data, m2mm)
            metrics.update(local_metrics)
        
        if 'accel' in self.metrics_to_compute:
            accel = self._compute_acceleration_error(
                pred_data['pred_j3d'], 
                gt_data['gt_j3d_cam']
            ) * (self.fps ** 2)
            metrics['accel'] = float(accel.mean())
        
        # === 全局运动评估 ===
        if any(m in self.metrics_to_compute for m in ['w_mpjpe', 'wa_mpjpe']):
            global_metrics = self._evaluate_global_motion(
                gt_data['gt_j3d'], 
                pred_data['pred_j3d_w'],
                m2mm
            )
            metrics.update(global_metrics)
        
        if 'rte' in self.metrics_to_compute:
            rte = self._compute_rte(
                gt_data['gt_j3d'][:, 0], 
                pred_data['pred_j3d_w'][:, 0]
            ) * 1e2  # 米转厘米
            metrics['rte'] = float(rte.mean())
        
        if 'erve' in self.metrics_to_compute:
            erve = self._compute_erve(
                gt_data['gt_ori'],
                gt_data['gt_j3d'],
                pred_data['pred_ori_w'],
                pred_data['pred_j3d_w']
            ) * m2mm
            metrics['erve'] = float(erve.mean())
        
        # === 相机运动评估 ===
        if 'ate' in self.metrics_to_compute or 'ate_s' in self.metrics_to_compute:
            cam_metrics = self._evaluate_camera_motion(data, gt_data)
            metrics.update(cam_metrics)
        
        # 保存结果
        data.metrics = metrics
        
        # self.logger.info("Evaluation results:")
        # for k, v in metrics.items():
        #     self.logger.info(f"  {k}: {v:.4f}")
        
        return data
    
    def _load_gt_data(self, data: PipelineData) -> Dict[str, Any]:
        """加载 GT 数据"""
        from lib.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

        ann = data.annotations

        # 检查是否有帧采样信息
        sampling_info = data.metadata.get('frame_sampling')
        if sampling_info:
            sampled_indices = np.array(sampling_info['sampled_indices'])
            self.logger.info(
                f"Applying frame sampling to GT: {len(ann['smpl']['poses_body'])} -> {len(sampled_indices)} frames"
            )
        else:
            sampled_indices = None

        gender = ann['gender']
        poses_body = ann["smpl"]["poses_body"]
        poses_root = ann["smpl"]["poses_root"]
        betas = np.repeat(
            ann["smpl"]["betas"].reshape((1, -1)),
            repeats=ann["n_frames"],
            axis=0
        )
        trans = ann["smpl"]["trans"]
        ext = ann['camera']['extrinsics']

        # 应用采样
        if sampled_indices is not None:
            poses_body = poses_body[sampled_indices]
            poses_root = poses_root[sampled_indices]
            betas = betas[sampled_indices]
            trans = trans[sampled_indices]
            ext = ext[sampled_indices]

        tt = lambda x: torch.from_numpy(x).float()

        # 世界坐标系下的 GT
        gt = self._smpls[gender](
            body_pose=tt(poses_body),
            global_orient=tt(poses_root),
            betas=tt(betas),
            transl=tt(trans),
            pose2rot=True,
            default_smpl=True
        )

        # 相机坐标系下的 GT
        poses_root_cam = matrix_to_axis_angle(
            tt(ext[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root))
        )
        gt_cam = self._smpls[gender](
            body_pose=tt(poses_body),
            global_orient=poses_root_cam,
            betas=tt(betas),
            pose2rot=True,
            default_smpl=True
        )

        valid_mask = ann.get('good_frames_mask')
        if sampled_indices is not None and valid_mask is not None:
            valid_mask = valid_mask[sampled_indices]

        return {
            'gender': gender,
            'gt_j3d': gt.joints[:, :24],
            'gt_vert': gt.vertices,
            'gt_ori': axis_angle_to_matrix(tt(poses_root)),
            'gt_j3d_cam': gt_cam.joints[:, :24],
            'gt_vert_cam': gt_cam.vertices,
            'ext': ext,
            'valid_mask': valid_mask,
        }
    
    def _compute_pred_smpl(self, data: PipelineData) -> Dict[str, Any]:
        """计算预测的 SMPL 输出"""
        from lib.vis.traj import traj_filter
        
        smpl = data.smpl_params
        cam = data.camera_params
        
        pred_rotmat = torch.tensor(smpl.rotmat) if not isinstance(smpl.rotmat, torch.Tensor) else smpl.rotmat
        pred_shape = torch.tensor(smpl.betas) if not isinstance(smpl.betas, torch.Tensor) else smpl.betas
        pred_trans = torch.tensor(smpl.trans) if not isinstance(smpl.trans, torch.Tensor) else smpl.trans
        
        # SMPL forward
        pred = self._smpls['neutral'](
            body_pose=pred_rotmat[:, 1:], 
            global_orient=pred_rotmat[:, [0]], 
            betas=pred_shape, 
            transl=pred_trans.squeeze(),
            pose2rot=False, 
            default_smpl=True
        )
        
        pred_vert = pred.vertices
        pred_j3d = pred.joints[:, :24]
        
        # 转换到世界坐标系
        pred_camr = torch.tensor(cam.R) if not isinstance(cam.R, torch.Tensor) else cam.R
        pred_camt = torch.tensor(cam.T) if not isinstance(cam.T, torch.Tensor) else cam.T
        
        pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:, None]
        pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:, None]
        pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, pred_rotmat[:, 0])
        
        # 轨迹滤波
        pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)
        
        return {
            'pred_j3d': pred_j3d,
            'pred_vert': pred_vert,
            'pred_j3d_w': pred_j3d_w,
            'pred_vert_w': pred_vert_w,
            'pred_ori_w': pred_ori_w,
            'pred_camr': pred_camr,
            'pred_camt': pred_camt,
        }
    
    def _apply_valid_mask(self, gt_data, pred_data, valid_mask):
        """应用有效帧 mask"""
        for key in ['gt_j3d', 'gt_vert', 'gt_ori', 'gt_j3d_cam', 'gt_vert_cam']:
            if key in gt_data and gt_data[key] is not None:
                gt_data[key] = gt_data[key][valid_mask]
        
        for key in ['pred_j3d', 'pred_vert', 'pred_j3d_w', 'pred_vert_w', 'pred_ori_w']:
            if key in pred_data and pred_data[key] is not None:
                pred_data[key] = pred_data[key][valid_mask]
        
        return gt_data, pred_data
    
    def _evaluate_local_motion(self, gt_data, pred_data, m2mm):
        """评估局部运动"""
        from lib.utils.eval_utils import (
            batch_align_by_pelvis, 
            batch_compute_similarity_transform_torch
        )
        
        pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam = batch_align_by_pelvis(
            [pred_data['pred_j3d'], gt_data['gt_j3d_cam'], 
             pred_data['pred_vert'], gt_data['gt_vert_cam']], 
            pelvis_idxs=[1, 2]
        )
        
        metrics = {}
        
        # PA-MPJPE
        S1_hat = batch_compute_similarity_transform_torch(pred_j3d, gt_j3d_cam)
        pa_mpjpe = torch.sqrt(((S1_hat - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        metrics['pa_mpjpe'] = float(pa_mpjpe.mean())
        
        # MPJPE
        mpjpe = torch.sqrt(((pred_j3d - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        metrics['mpjpe'] = float(mpjpe.mean())
        
        # PVE
        pve = torch.sqrt(((pred_vert - gt_vert_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        metrics['pve'] = float(pve.mean())
        
        return metrics
    
    def _evaluate_global_motion(self, gt_j3d, pred_j3d_w, m2mm):
        """评估全局运动"""
        from lib.utils.eval_utils import first_align_joints, global_align_joints, compute_jpe
        
        chunk_length = self.chunk_length
        w_mpjpe, wa_mpjpe = [], []
        
        num_frames = len(gt_j3d)
        
        for start in range(0, num_frames - chunk_length, chunk_length):
            end = start + chunk_length
            if start + 2 * chunk_length > num_frames:
                end = num_frames - 1
            
            target_j3d = gt_j3d[start:end].clone().cpu()
            pred_j3d = pred_j3d_w[start:end].clone().cpu()
            
            w_j3d = first_align_joints(target_j3d, pred_j3d)
            wa_j3d = global_align_joints(target_j3d, pred_j3d)
            
            w_jpe = compute_jpe(target_j3d, w_j3d)
            wa_jpe = compute_jpe(target_j3d, wa_j3d)
            
            w_mpjpe.append(w_jpe)
            wa_mpjpe.append(wa_jpe)
        
        if w_mpjpe:
            w_mpjpe = np.concatenate(w_mpjpe) * m2mm
            wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm
            return {
                'w_mpjpe': float(w_mpjpe.mean()),
                'wa_mpjpe': float(wa_mpjpe.mean())
            }
        
        return {'w_mpjpe': 0.0, 'wa_mpjpe': 0.0}
    
    def _compute_acceleration_error(self, pred_j3d, gt_j3d):
        """计算加速度误差"""
        from lib.utils.eval_utils import compute_error_accel
        accel = compute_error_accel(
            joints_pred=pred_j3d.cpu(), 
            joints_gt=gt_j3d.cpu()
        )[1:-1]
        return accel
    
    def _compute_rte(self, gt_root, pred_root):
        """计算根轨迹误差"""
        from lib.utils.eval_utils import compute_rte
        return compute_rte(gt_root, pred_root)
    
    def _compute_erve(self, gt_ori, gt_j3d, pred_ori, pred_j3d):
        """计算自我中心根速度误差"""
        from lib.utils.eval_utils import computer_erve
        return computer_erve(gt_ori, gt_j3d, pred_ori, pred_j3d)
    
    def _evaluate_camera_motion(self, data: PipelineData, gt_data):
        """评估相机运动"""
        from lib.utils.rotation_conversions import matrix_to_quaternion
        from lib.camera.slam_utils import eval_slam
        
        ext = gt_data['ext']
        cam_r = ext[:, :3, :3].transpose(0, 2, 1)
        cam_t = np.einsum('bij, bj->bi', cam_r, -ext[:, :3, -1])
        cam_q = matrix_to_quaternion(torch.from_numpy(cam_r)).numpy()
        
        pred_camr = data.camera_params.R
        pred_camt = data.camera_params.T
        
        if isinstance(pred_camr, torch.Tensor):
            pred_camr = pred_camr
        else:
            pred_camr = torch.from_numpy(pred_camr)
        
        if isinstance(pred_camt, torch.Tensor):
            pred_camt = pred_camt
        else:
            pred_camt = torch.from_numpy(pred_camt)
        
        pred_camq = matrix_to_quaternion(pred_camr)
        pred_traj = torch.concat([pred_camt, pred_camq], dim=-1).numpy()
        
        metrics = {}
        
        # ATE with scale correction
        stats_slam, _, _ = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=True)
        metrics['ate'] = stats_slam['mean']
        
        # ATE without scale correction
        stats_metric, _, _ = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=False)
        metrics['ate_s'] = stats_metric['mean']
        
        return metrics

