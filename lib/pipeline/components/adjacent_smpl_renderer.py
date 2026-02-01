"""
AdjacentSMPLRenderer - 渲染相邻帧SMPL mesh的组件

这个组件复现adjTram的功能，对每一帧渲染当前帧、前x帧、后x帧的SMPL mesh，
并将它们在当前帧的相机视角下拼接显示。

设计原则：
- 统一使用data.smpl_params（无论是TRAM还是GT）
- GT SMPL Loader负责将GT数据转换成和VIMO一样的格式
- 本组件不需要区分数据来源
"""

import os
import cv2
import numpy as np
import torch
from typing import Dict, Any
from tqdm import tqdm

from lib.pipeline.core.component import Component
from lib.pipeline.core.data import PipelineData
from lib.models.smpl import SMPL


class AdjacentSMPLRenderer(Component):
    """
    渲染相邻帧SMPL mesh的组件

    对于每一帧，渲染当前帧、前x帧、后x帧的SMPL mesh，
    在当前帧的相机视角下显示，用于可视化和调试。

    Config:
        pre_dis: 相邻帧间隔（默认20）
        device: 计算设备（默认'cpu'）
        output_dir: 输出目录（默认'results/adjacent_smpl'）
    """

    COMPONENT_TYPE = "adjacent_smpl_renderer"

    DEFAULT_CONFIG = {
        'pre_dis': 20,  # 相邻帧间隔
        'device': 'cpu',
        'output_dir': 'results/adjacent_smpl',
    }

    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)

        self.pre_dis = self.config['pre_dis']
        self.device = self.config['device']
        self.output_dir = self.config['output_dir']

        # SMPL模型和faces
        self.smpl_model = None
        self.smpl_faces = None

    def validate_input(self, data: PipelineData) -> bool:
        """验证输入"""
        # 需要SMPL参数
        has_smpl = data.smpl_params is not None
        if not has_smpl:
            self.logger.warning("No SMPL params found in data")
            return False

        # 需要图像路径
        has_images = hasattr(data, 'image_paths') or data.metadata.get('image_paths')
        if not has_images:
            self.logger.warning("No image paths found in data")
            return False

        # 需要相机参数（优先从 data.camera_params，其次从 annotations）
        if data.camera_params is None:
            self.logger.warning("No camera_params found in data")
            return False

        return True

    def setup(self):
        """初始化SMPL模型"""
        self.smpl_model = SMPL(
            gender='neutral',
            model_type='smpl',
        ).to(self.device)

        self.smpl_faces = self.smpl_model.faces
        self.logger.info(f"SMPL faces shape: {self.smpl_faces.shape}")

        self._is_setup = True
        self.logger.info("Adjacent SMPL Renderer component initialized")

    def execute(self, data: PipelineData) -> PipelineData:
        """执行渲染"""
        from lib.vis.renderer import Renderer

        # 获取图像路径
        if hasattr(data, 'image_paths'):
            img_paths = data.image_paths
        else:
            img_paths = data.metadata.get('image_paths', [])

        N = len(img_paths)
        first_img = cv2.imread(img_paths[0])
        H, W = first_img.shape[:2]

        # 获取相机内参（从 data.camera_params）
        if data.camera_params.intrinsics is not None:
            intrinsics = data.camera_params.intrinsics
        elif data.annotations and 'camera' in data.annotations and 'intrinsics' in data.annotations['camera']:
            intrinsics = data.annotations['camera']['intrinsics']
        else:
            self.logger.error("No camera intrinsics found in camera_params or annotations")
            return data

        focal_length = float(intrinsics[0, 0])

        # 获取相机外参（从 data.camera_params 计算 w2c）
        # data.camera_params.R 和 T 是 c2w 变换，需要转换成 w2c
        if data.camera_params.R is not None and data.camera_params.T is not None:
            R_cw = data.camera_params.R  # [N, 3, 3] 相机到世界
            t_cw = data.camera_params.T  # [N, 3] 相机到世界的平移

            # 转换为 numpy（如果是 tensor）
            if isinstance(R_cw, torch.Tensor):
                R_cw = R_cw.cpu().numpy()
            if isinstance(t_cw, torch.Tensor):
                t_cw = t_cw.cpu().numpy()

            # 计算 w2c 变换
            # w2c 的旋转: R_wc = R_cw^T
            # w2c 的平移: t_wc = -R_wc @ t_cw
            R_wc = np.transpose(R_cw, (0, 2, 1))  # [N, 3, 3]
            t_wc = -np.einsum('nij,nj->ni', R_wc, t_cw)  # [N, 3]

            # 构造 w2c 矩阵 [N, 3, 4]
            w2c = np.zeros((N, 3, 4))
            w2c[:, :3, :3] = R_wc
            w2c[:, :3, 3] = t_wc

            self.logger.info(f"Computed w2c matrices from camera_params: {w2c.shape}")
        elif data.annotations and 'camera' in data.annotations and 'extrinsics' in data.annotations['camera']:
            # 回退到 annotations
            w2c = data.annotations['camera']['extrinsics']
            self.logger.info("Using camera extrinsics from annotations")
        else:
            self.logger.error("No camera extrinsics found")
            return data

        # 获取SMPL参数（统一格式，无论是TRAM还是GT）
        smpl_params = data.smpl_params
        if smpl_params.rotmat is None:
            self.logger.error("SMPL rotmat is required for rendering")
            return data

        rotmat = smpl_params.rotmat  # [N, 24, 3, 3]
        betas = smpl_params.betas  # [N, 10]

        # 优先使用 global_trans（世界坐标系），如果没有则使用 trans（相机坐标系）
        if smpl_params.global_trans is not None:
            trans = smpl_params.global_trans
            coord_system = "world"
            self.logger.info("Using global_trans (world coordinate system)")

            # 如果使用世界坐标系的 trans，需要将 rotmat 的 root orientation 也转换到世界坐标系
            # GT 后端的 rotmat[:, 0] 是相机坐标系的，需要转换回来
            if data.camera_params.R is not None:
                R_cw = data.camera_params.R  # [N, 3, 3] 相机到世界
                if isinstance(R_cw, torch.Tensor):
                    R_cw = R_cw.cpu().numpy()

                # 将 root orientation 从相机坐标系转换到世界坐标系
                # R_world_root = R_cw @ R_cam_root
                if isinstance(rotmat, np.ndarray):
                    rotmat_cam_root = rotmat[:, 0]  # [N, 3, 3]
                    rotmat_world_root = np.einsum('nij,njk->nik', R_cw, rotmat_cam_root)
                    rotmat[:, 0] = rotmat_world_root
                else:
                    rotmat_cam_root = rotmat[:, 0].cpu().numpy()
                    rotmat_world_root = np.einsum('nij,njk->nik', R_cw, rotmat_cam_root)
                    rotmat[:, 0] = torch.from_numpy(rotmat_world_root).to(rotmat.device)

                self.logger.info("Converted root orientation from camera to world coordinate system")
        elif smpl_params.trans is not None:
            trans = smpl_params.trans
            coord_system = "camera"
            self.logger.warning(
                "Using camera-coordinate trans. Rendered mesh may not be globally accurate. "
                "Consider using a pipeline that provides global_trans."
            )
        else:
            self.logger.error("Neither trans nor global_trans available in smpl_params")
            return data

        # 确保trans是正确的形状 [N, 3]
        if trans.ndim == 3 and trans.shape[1] == 1:
            trans = trans.squeeze(1)  # [N, 1, 3] -> [N, 3]
            self.logger.info(f"Squeezed trans to {trans.shape}")

        self.logger.info(f"SMPL params: rotmat={rotmat.shape}, betas={betas.shape}, trans={trans.shape}")

        # 生成SMPL vertices
        vertices = self._generate_vertices(rotmat, betas, trans)
        self.logger.info(f"Generated vertices: {vertices.shape} ({coord_system} coords)")

        # 创建渲染器
        render = Renderer(W, H, focal_length, self.device, self.smpl_faces)

        # 创建输出目录
        output_dir = os.path.join(self.output_dir, data.metadata.get('sequence_name', 'sequence'))
        os.makedirs(output_dir, exist_ok=True)

        # 计算渲染的帧索引（每隔 pre_dis 帧）
        rendered_indices = list(range(0, N, self.pre_dis))

        # 如果总帧数不能整除 pre_dis，需要保存最后一帧用于后续 SLAM
        # 最后一帧不会被渲染 mesh，但会参与 SLAM 过程（利用背景信息估计相机位姿）
        last_frame_idx = N - 1
        last_frame_saved = False
        if last_frame_idx not in rendered_indices:
            self.logger.info(f"Last frame {last_frame_idx} will be saved (not rendered) for SLAM")
            # 保存最后一帧的原始图像（不渲染 mesh）
            # 保持原始文件名，这样 DROID-SLAM 会读取它参与 SLAM
            last_img = cv2.imread(img_paths[last_frame_idx])
            last_img_name = os.path.basename(img_paths[last_frame_idx])
            last_output_path = os.path.join(output_dir, last_img_name)
            cv2.imwrite(last_output_path, last_img)
            last_frame_saved = True
            self.logger.info(f"Saved last frame to {last_output_path} for SLAM (will be included in DROID-SLAM processing)")

        # 渲染每隔 pre_dis 帧的图像
        self.logger.info(f"Rendering {len(rendered_indices)} frames (every {self.pre_dis}th frame)...")

        for i in tqdm(rendered_indices, desc="Rendering adjacent frames"):
            img = cv2.imread(img_paths[i])
            final_img = img.copy()
            img_name = os.path.basename(img_paths[i])

            # 如果是世界坐标系，需要将 vertices 转换到当前帧的相机坐标系
            if coord_system == "world":
                # 获取当前帧的 w2c 变换矩阵
                w2c_i = w2c[i]  # [3, 4]
                R_w2c = w2c_i[:3, :3]  # [3, 3]
                t_w2c = w2c_i[:3, 3]   # [3]

                # 转换为 tensor
                if isinstance(R_w2c, np.ndarray):
                    R_w2c = torch.from_numpy(R_w2c).float().to(self.device)
                if isinstance(t_w2c, np.ndarray):
                    t_w2c = torch.from_numpy(t_w2c).float().to(self.device)

                # 渲染前x帧（转换到当前帧相机坐标系）
                if i - self.pre_dis >= 0:
                    verts_pre_world = torch.tensor(vertices[i - self.pre_dis]).to(self.device)  # [6890, 3]
                    verts_pre_cam = torch.einsum('ij,vj->vi', R_w2c, verts_pre_world) + t_w2c
                    final_img = render.render_mesh(verts_pre_cam, final_img)

                # 渲染当前帧（转换到当前帧相机坐标系）
                verts_world = torch.tensor(vertices[i]).to(self.device)  # [6890, 3]
                verts_cam = torch.einsum('ij,vj->vi', R_w2c, verts_world) + t_w2c
                final_img = render.render_mesh(verts_cam, final_img)

                # 渲染后x帧（转换到当前帧相机坐标系）
                if i + self.pre_dis < N:
                    verts_next_world = torch.tensor(vertices[i + self.pre_dis]).to(self.device)  # [6890, 3]
                    verts_next_cam = torch.einsum('ij,vj->vi', R_w2c, verts_next_world) + t_w2c
                    final_img = render.render_mesh(verts_next_cam, final_img)
            else:
                # 相机坐标系，直接渲染
                if i - self.pre_dis >= 0:
                    verts_pre = torch.tensor(vertices[i - self.pre_dis]).to(self.device)
                    final_img = render.render_mesh(verts_pre, final_img)

                verts = torch.tensor(vertices[i]).to(self.device)
                final_img = render.render_mesh(verts, final_img)

                if i + self.pre_dis < N:
                    verts_next = torch.tensor(vertices[i + self.pre_dis]).to(self.device)
                    final_img = render.render_mesh(verts_next, final_img)

            # 保存图像
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, final_img)

        self.logger.info(f"Rendered images saved to {output_dir}")
        self.logger.info(f"  - Each frame shows: [pre_{self.pre_dis} | current | next_{self.pre_dis}]")
        self.logger.info(f"  - Total frames rendered: {len(rendered_indices)} out of {N} total frames")
        if last_frame_saved:
            self.logger.info(f"  - Last frame {last_frame_idx} saved without mesh rendering (will be used by DROID-SLAM)")

        # 保存渲染的帧索引到 metadata，供后续插值使用
        # 注意：如果保存了最后一帧，DROID-SLAM 会读取 len(rendered_indices) + 1 帧
        # 最后一帧的相机参数将用于插值时填充末尾帧
        data.metadata['adjacent_render_info'] = {
            'total_frames': N,
            'pre_dis': self.pre_dis,
            'rendered_indices': rendered_indices,
            'last_frame_idx': last_frame_idx if last_frame_saved else None,
            'num_processed_frames': len(rendered_indices) + (1 if last_frame_saved else 0),
        }

        return data

    def _generate_vertices(self, rotmat, betas, trans):
        """从SMPL参数生成vertices

        支持输入为 numpy.ndarray 或 torch.Tensor
        """
        N = rotmat.shape[0]

        with torch.no_grad():
            # 转换为 tensor（如果需要）
            if isinstance(rotmat, np.ndarray):
                rotmat_tensor = torch.from_numpy(rotmat).float().to(self.device)
            else:
                rotmat_tensor = rotmat.float().to(self.device)

            if isinstance(betas, np.ndarray):
                betas_tensor = torch.from_numpy(betas).float().to(self.device)
            else:
                betas_tensor = betas.float().to(self.device)

            if isinstance(trans, np.ndarray):
                trans_tensor = torch.from_numpy(trans).float().to(self.device)
            else:
                trans_tensor = trans.float().to(self.device)

            global_orient = rotmat_tensor[:, [0]]
            body_pose = rotmat_tensor[:, 1:]

            smpl_output = self.smpl_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas_tensor,
                transl=trans_tensor,
                pose2rot=False,
            )

            vertices = smpl_output.vertices

        return vertices.cpu().numpy()
