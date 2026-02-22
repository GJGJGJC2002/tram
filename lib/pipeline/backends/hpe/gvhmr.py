"""GVHMRBackend - GVHMR 人体运动估计后端

GVHMR 输出两套 SMPL 参数：
  - smpl_params_global: 世界坐标系(ay)下的 global_orient, body_pose, betas, transl
  - smpl_params_incam: 相机坐标系下的 global_orient, body_pose, betas, transl

以及 static_conf_logits 用于后续滑步消除。

该后端只执行 GVHMR 推理（不做后处理），后处理（滑步消除、IK）由独立的
SkatingRemovalComponent 完成。
"""

import sys
import os
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import torch

from lib.pipeline.core.component import Backend


class GVHMRBackend(Backend):
    """
    GVHMR 后端 - 全局人体运动估计

    使用 GVHMR 模型估计世界坐标系和相机坐标系下的 SMPL 参数。

    Config:
        gvhmr_root: GVHMR 项目根目录
        static_cam: 是否假设静态相机（跳过 VO）
        f_mm: 相机焦距 (mm)，None 则使用默认估计
        vo_method: 视觉里程计方法 ('sift' 或 'dpvo')
        vo_scale: VO 图像缩放比例
        vo_step: VO 帧步长
    """

    DEFAULT_CONFIG = {
        'gvhmr_root': 'thirdparty/GVHMR',
        'static_cam': False,
        'f_mm': None,
        'vo_method': 'sift',
        'vo_scale': 0.5,
        'vo_step': 8,
    }

    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)

        self.gvhmr_root = self.config['gvhmr_root']
        self.static_cam = self.config['static_cam']
        self.f_mm = self.config['f_mm']
        self.vo_method = self.config['vo_method']
        self.vo_scale = self.config['vo_scale']
        self.vo_step = self.config['vo_step']

        self.model = None
        self.smplx_model = None
        self.smplx2smpl = None

    def setup(self):
        """初始化 GVHMR 模型"""
        gvhmr_abs = os.path.abspath(self.gvhmr_root)
        if gvhmr_abs not in sys.path:
            sys.path.insert(0, gvhmr_abs)

        from hmr4d.configs import register_store_gvhmr
        from hydra import initialize_config_module, compose
        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
        import hydra as hydra_lib

        # 注册配置
        register_store_gvhmr()

        # 使用 hydra compose 获取默认配置
        overrides = [
            "video_name=__placeholder__",
            f"static_cam={self.static_cam}",
            "verbose=False",
        ]
        if self.f_mm is not None:
            overrides.append(f"f_mm={self.f_mm}")

        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            cfg = compose(config_name="demo", overrides=overrides)

        # 实例化模型
        self.model = hydra_lib.utils.instantiate(cfg.model, _recursive_=False)

        # ckpt_path 在 GVHMR 配置中是相对路径，需要基于 GVHMR 根目录转为绝对路径
        ckpt_path = cfg.ckpt_path
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(gvhmr_abs, ckpt_path)
        self.model.load_pretrained_model(ckpt_path)
        self.model = self.model.eval().cuda()

        # 加载 smplx 模型（用于 get_skeleton 计算 offset）
        from hmr4d.utils.smplx_utils import make_smplx
        self.smplx_model = make_smplx("supermotion").cuda()
        self.smplx2smpl = torch.load(
            os.path.join(gvhmr_abs, "hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        ).cuda()

        self._is_setup = True
        self.logger.info("GVHMR backend initialized")

    def estimate_smpl(
        self,
        image_paths: List[str],
        bboxes: np.ndarray,
        img_focal: float = None,
        img_center: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        运行 GVHMR 推理

        Args:
            image_paths: 图像路径列表
            bboxes: 边界框 [N, 4] 或 [N, K, 5]
            img_focal: 图像焦距（可选，None 则由 GVHMR 自行估计）
            img_center: 图像主点（可选）

        Returns:
            包含 GVHMR 全部输出的字典
        """
        self.ensure_setup()

        gvhmr_abs = os.path.abspath(self.gvhmr_root)
        sys.path.insert(0, gvhmr_abs)

        # 在切换工作目录前，将图像路径转为绝对路径
        image_paths = [os.path.abspath(p) for p in image_paths]

        # GVHMR 的预处理工具使用相对路径查找模型，需要临时切换工作目录
        original_cwd = os.getcwd()
        os.chdir(gvhmr_abs)

        try:
            import cv2
            from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
            from hmr4d.utils.geo.hmr_cam import (
                get_bbx_xys_from_xyxy,
                estimate_K,
                convert_K_to_K4,
                create_camera_sensor,
            )
            from hmr4d.utils.geo_transform import compute_cam_angvel
            from hmr4d.utils.net_utils import detach_to_cpu
            from pytorch3d.transforms import quaternion_to_matrix

            N = len(image_paths)
            first_img = cv2.imread(image_paths[0])
            height, width = first_img.shape[:2]
            self.logger.info(f"GVHMR processing {N} frames, image size: {width}x{height}")

            # --- 1. 准备 bboxes ---
            if bboxes.ndim == 3:
                bboxes_xyxy = bboxes[:, 0, :4]
            elif bboxes.ndim == 2 and bboxes.shape[1] > 4:
                bboxes_xyxy = bboxes[:, :4]
            else:
                bboxes_xyxy = bboxes

            bbx_xyxy = torch.from_numpy(bboxes_xyxy).float()
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

            # --- 2. VitPose 关键点检测 ---
            self.logger.info("Running VitPose...")
            vitpose_extractor = VitPoseExtractor()
            vitpose = vitpose_extractor.extract_from_images(image_paths, bbx_xys)
            del vitpose_extractor
            torch.cuda.empty_cache()

            # --- 3. ViT 特征提取 ---
            self.logger.info("Running ViT feature extraction...")
            extractor = Extractor()
            vit_features = extractor.extract_image_features(image_paths, bbx_xys)
            del extractor
            torch.cuda.empty_cache()

            # --- 4. 视觉里程计（相机旋转估计） ---
            if self.static_cam:
                R_w2c = torch.eye(3).repeat(N, 1, 1)
            else:
                self.logger.info(f"Running Visual Odometry ({self.vo_method})...")
                simple_vo = SimpleVO(
                    image_paths,
                    scale=self.vo_scale,
                    step=self.vo_step,
                    method=self.vo_method,
                    f_mm=self.f_mm,
                    width=width,
                    height=height,
                )
                vo_results = simple_vo.compute()
                R_w2c = torch.from_numpy(vo_results[:, :3, :3])

            # --- 5. 构建相机内参 ---
            if self.f_mm is not None:
                K_fullimg = create_camera_sensor(width, height, self.f_mm)[2].repeat(N, 1, 1)
            else:
                K_fullimg = estimate_K(width, height).repeat(N, 1, 1)

            # --- 6. 组装数据 ---
            data = {
                "length": torch.tensor(N),
                "bbx_xys": bbx_xys,
                "kp2d": vitpose,
                "K_fullimg": K_fullimg,
                "cam_angvel": compute_cam_angvel(R_w2c),
                "f_imgseq": vit_features,
            }

            # --- 7. GVHMR 推理 ---
            self.logger.info("Running GVHMR inference...")
            pred = self.model.predict(data, static_cam=self.static_cam)
            pred = detach_to_cpu(pred)

            # --- 8. 计算 skeleton offset ---
            betas_global = pred["smpl_params_global"]["betas"]  # (F, 10)
            skeleton_offset = self.smplx_model.get_skeleton(
                betas_global[0:1].cuda()
            )[0, 0].cpu()  # (3,) root joint offset

            self.logger.info(
                f"GVHMR inference complete. "
                f"global orient shape: {pred['smpl_params_global']['global_orient'].shape}, "
                f"skeleton offset: {skeleton_offset}"
            )

            return {
                'smpl_params_global': pred['smpl_params_global'],
                'smpl_params_incam': pred['smpl_params_incam'],
                'static_conf_logits': pred['net_outputs']['static_conf_logits'],
                'K_fullimg': pred['K_fullimg'],
                'skeleton_offset': skeleton_offset,
            }
        finally:
            # 恢复工作目录
            os.chdir(original_cwd)

    def cleanup(self):
        """清理资源"""
        self.model = None
        self.smplx_model = None
        self.smplx2smpl = None
        super().cleanup()
