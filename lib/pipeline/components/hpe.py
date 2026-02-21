"""HPEComponent - 人体姿态估计组件"""

from typing import Dict, Any
import numpy as np
import torch

from lib.pipeline.core.component import BackendComponent
from lib.pipeline.core.data import PipelineData, SMPLParams
from lib.pipeline.backends.hpe import VIMOBackend, GTSmplBackend


class HPEComponent(BackendComponent):
    """
    人体姿态估计组件
    
    从图像序列中估计 SMPL 人体模型参数。
    
    Config:
        backend: HPE 后端类型 ('vimo', 'hmr', ...)
        mode: 估计模式 ('accurate', 'efficient')
        use_mean_shape: 是否使用平均形状参数
    """
    
    COMPONENT_TYPE = "hpe"
    
    BACKENDS = {
        'vimo': VIMOBackend,
        'gt': GTSmplBackend,
    }
    
    DEFAULT_BACKEND = 'vimo'
    
    DEFAULT_CONFIG = {
        'mode': 'accurate',
        'use_mean_shape': True,
        'frame_stride': 1,      # 跳帧步长，1=不跳帧，20=每20帧取一次
        'max_frames': None,     # 最大处理帧数，None=全部
    }
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)
        
        self.mode = self.config['mode']
        self.use_mean_shape = self.config['use_mean_shape']
    
    def validate_input(self, data: PipelineData) -> bool:
        """验证输入：需要图像路径和边界框"""
        has_images = len(data.image_paths) > 0
        has_boxes = data.bboxes is not None
        return has_images and has_boxes

    def _sample_frames(self, data: PipelineData):
        """
        对输入数据进行帧采样

        根据 frame_stride 和 max_frames 对数据采样。
        会修改 data.image_paths, data.bboxes, data.camera_params, data.valid_frames_mask

        Returns:
            sampled_indices: 采样的帧索引
        """
        frame_stride = self.config.get('frame_stride', 1)
        max_frames = self.config.get('max_frames')

        if frame_stride <= 1 and max_frames is None:
            # 不需要采样
            return np.arange(len(data.image_paths))

        num_frames = len(data.image_paths)

        # 生成采样索引
        if frame_stride > 1:
            indices = np.arange(0, num_frames, frame_stride)
        else:
            indices = np.arange(num_frames)

        # 应用 max_frames 限制
        if max_frames is not None and len(indices) > max_frames:
            indices = indices[:max_frames]

        self.logger.info(
            f"Frame sampling: {num_frames} -> {len(indices)} frames "
            f"(stride={frame_stride}, max_frames={max_frames})"
        )

        # 采样 image_paths
        data.image_paths = [data.image_paths[i] for i in indices]

        # 采样 bboxes
        if data.bboxes is not None:
            data.bboxes = data.bboxes[indices]

        # 采样 camera_params
        if data.camera_params is not None:
            from lib.pipeline.core.data import CameraParams
            data.camera_params = CameraParams(
                R=data.camera_params.R[indices] if data.camera_params.R is not None else None,
                T=data.camera_params.T[indices] if data.camera_params.T is not None else None,
                focal_length=data.camera_params.focal_length,
                principal_point=data.camera_params.principal_point,
                intrinsics=data.camera_params.intrinsics,
                world_R=data.camera_params.world_R[indices] if data.camera_params.world_R is not None else None,
                world_T=data.camera_params.world_T[indices] if data.camera_params.world_T is not None else None,
            )

        # 采样 valid_frames_mask
        if data.valid_frames_mask is not None:
            data.valid_frames_mask = data.valid_frames_mask[indices]

        # 记录采样信息到 metadata
        data.metadata['frame_sampling'] = {
            'original_num_frames': num_frames,
            'sampled_num_frames': len(indices),
            'sampled_indices': indices.tolist(),
            'frame_stride': frame_stride,
            'max_frames': max_frames,
        }

        return indices
    
    def execute(self, data: PipelineData) -> PipelineData:
        """
        执行人体姿态估计

        结果存储在 data.smpl_params 中。
        """
        # 根据后端类型调用不同的方法
        if self.backend_type == 'gt':
            return self._execute_gt(data)
        else:
            return self._execute_vimo(data)

    def _execute_gt(self, data: PipelineData) -> PipelineData:
        """使用 GT 后端执行"""
        if data.annotations is None:
            raise ValueError("GT backend requires annotations")

        # 检查是否有帧采样信息
        sampling_info = data.metadata.get('frame_sampling')
        if sampling_info:
            sampled_indices = np.array(sampling_info['sampled_indices'])
        else:
            sampled_indices = None

        self.logger.info("Loading GT SMPL parameters...")

        # 调用 GT 后端
        smpl_results = self.backend.estimate_smpl(
            annotations=data.annotations,
            sampled_indices=sampled_indices
        )

        # 后处理
        pred_shape = smpl_results['pred_shape']
        if self.use_mean_shape:
            mean_shape = pred_shape.mean(dim=0, keepdim=True)
            pred_shape = mean_shape.repeat(len(pred_shape), 1)

        # 从 annotations 中读取世界坐标系的 trans（global_trans）
        if sampled_indices is not None:
            global_trans = data.annotations['smpl']['trans'][sampled_indices]
        else:
            global_trans = data.annotations['smpl']['trans']

        # 创建 SMPL 参数对象
        data.smpl_params = SMPLParams(
            poses=smpl_results['pred_pose'],
            betas=pred_shape,
            trans=smpl_results['pred_trans'],  # 相机坐标系
            global_trans=torch.from_numpy(global_trans).float(),  # 世界坐标系
            rotmat=smpl_results['pred_rotmat'],
            pred_cam=smpl_results['pred_cam'],
        )

        # 记录元数据
        num_frames = len(smpl_results['pred_trans'])
        data.metadata['hpe_stats'] = {
            'num_frames': num_frames,
            'backend': 'gt',
            'use_mean_shape': self.use_mean_shape,
        }

        self.logger.info(f"GT SMPL loaded for {num_frames} frames")

        return data

    def _execute_vimo(self, data: PipelineData) -> PipelineData:
        """使用 VIMO 后端执行"""
        # 帧采样（如果配置了跳帧）
        self._sample_frames(data)

        # 获取相机参数
        img_focal, img_center = self._get_camera_params(data)

        self.logger.info(f"Estimating SMPL parameters ({self.mode} mode)...")
        self.logger.info(f"Focal length: {img_focal:.2f}, Center: {img_center}")

        # 估计 SMPL
        smpl_results = self.backend.estimate_smpl(
            image_paths=data.image_paths,
            bboxes=data.bboxes,
            img_focal=img_focal,
            img_center=img_center,
            mode=self.mode
        )

        # 后处理
        pred_shape = smpl_results['pred_shape']
        if self.use_mean_shape:
            mean_shape = pred_shape.mean(dim=0, keepdim=True)
            pred_shape = mean_shape.repeat(len(pred_shape), 1)

        # 计算 global_trans（从相机坐标系转换到世界坐标系）
        trans_cam = smpl_results['pred_trans']  # 相机坐标系
        global_trans = None
        R_cw = None
        t_cw = None
        coord_source = None

        # 如果有相机外参（c2w 变换），则转换到世界坐标系
        # 优先使用 world_R/world_T（对齐到世界坐标系后的值）
        if data.camera_params is not None:
            # 使用 world_R/world_T（如果存在），否则回退到 R/T
            if data.camera_params.world_R is not None and data.camera_params.world_T is not None:
                R_cw = data.camera_params.world_R  # [N, 3, 3] 相机到世界的旋转（已对齐）
                t_cw = data.camera_params.world_T  # [N, 3] 相机到世界的平移（已对齐）
                coord_source = "world_R/world_T"
            elif data.camera_params.R is not None and data.camera_params.T is not None:
                R_cw = data.camera_params.R  # [N, 3, 3]
                t_cw = data.camera_params.T  # [N, 3]
                coord_source = "R/T"

        if R_cw is not None and t_cw is not None:
            self.logger.info(f"Using camera params from {coord_source} for global_trans calculation")

            # 转换到世界坐标系: p_world = R_cw @ p_cam + t_cw
            if isinstance(R_cw, np.ndarray):
                R_cw = torch.from_numpy(R_cw).float()
            if isinstance(t_cw, np.ndarray):
                t_cw = torch.from_numpy(t_cw).float()
            if isinstance(trans_cam, np.ndarray):
                trans_cam = torch.from_numpy(trans_cam).float()

            # 确保形状一致
            self.logger.info(f"Tensor shapes before einsum: R_cw={R_cw.shape}, trans_cam={trans_cam.shape}, t_cw={t_cw.shape}")

            # 处理 trans_cam 的额外维度（可能是 [N, 1, 3] 或 [N, M, 3]）
            if len(trans_cam.shape) == 3:
                if trans_cam.shape[1] == 1:
                    # [N, 1, 3] -> [N, 3]
                    trans_cam = trans_cam.squeeze(1)
                    self.logger.info(f"Squeezed trans_cam from [N, 1, 3] to {trans_cam.shape}")
                elif trans_cam.shape[2] == 3:
                    # [N, M, 3] -> [N, 3]，取第一个或平均
                    # 这里我们取第一个，假设 M 维度是冗余的
                    trans_cam = trans_cam[:, 0, :]
                    self.logger.info(f"Reduced trans_cam from [N, M, 3] to {trans_cam.shape} (took first)")

            # 处理可能的额外维度
            if len(R_cw.shape) == 4 and R_cw.shape[-2:] == (3, 3):
                # R_cw 是 [N, M, 3, 3]，需要 reshape
                N, M = R_cw.shape[:2]
                R_cw = R_cw.reshape(-1, 3, 3)  # [N*M, 3, 3]
                if trans_cam.shape[0] == N:
                    # trans_cam 是 [N, 3]，需要扩展到 [N*M, 3]
                    trans_cam = trans_cam.unsqueeze(1).expand(-1, M, -1).reshape(-1, 3)
                if t_cw.shape[0] == N:
                    # t_cw 是 [N, 3]，需要扩展到 [N*M, 3]
                    t_cw = t_cw.unsqueeze(1).expand(-1, M, -1).reshape(-1, 3)
                self.logger.info(f"Reshaped tensors: R_cw={R_cw.shape}, trans_cam={trans_cam.shape}, t_cw={t_cw.shape}")

            if len(R_cw.shape) == 3 and R_cw.shape[0] == trans_cam.shape[0]:
                global_trans = torch.einsum('nij,nj->ni', R_cw, trans_cam) + t_cw
                self.logger.info("Converted body trajectory from camera to world coordinate system")
            else:
                self.logger.warning(
                    f"Cannot compute global_trans: shape mismatch "
                    f"(R_cw: {R_cw.shape}, trans_cam: {trans_cam.shape})"
                )
        else:
            self.logger.warning(
                "No camera extrinsics available, using camera-coordinate trans. "
                "Body trajectory will not be globally accurate."
            )

        # 创建 SMPL 参数对象
        data.smpl_params = SMPLParams(
            poses=smpl_results['pred_pose'],
            betas=pred_shape,
            trans=trans_cam,  # 相机坐标系
            global_trans=global_trans,  # 世界坐标系（如果有相机外参）
            rotmat=smpl_results['pred_rotmat'],
            pred_cam=smpl_results['pred_cam'],
        )

        # 记录元数据
        num_frames = len(smpl_results['pred_pose'])
        data.metadata['hpe_stats'] = {
            'num_frames': num_frames,
            'backend': 'vimo',
            'mode': self.mode,
            'use_mean_shape': self.use_mean_shape,
            'img_focal': img_focal,
        }

        self.logger.info(f"SMPL estimation completed for {num_frames} frames")

        return data
    
    def _get_camera_params(self, data: PipelineData):
        """获取相机参数"""
        
        # 从 camera_params 获取
        if data.camera_params:
            if data.camera_params.focal_length:
                img_focal = data.camera_params.focal_length
            else:
                img_focal = 1000.0  # 默认值
            
            if data.camera_params.principal_point is not None:
                img_center = data.camera_params.principal_point
            else:
                # 假设图像中心
                img_size = data.get_image_size()
                if img_size:
                    img_center = np.array([img_size[1] / 2, img_size[0] / 2])
                else:
                    img_center = np.array([540, 960])  # 默认 1080p
            
            return img_focal, img_center
        
        # 使用默认值
        self.logger.warning("Using default camera parameters")
        return 1000.0, np.array([540, 960])


