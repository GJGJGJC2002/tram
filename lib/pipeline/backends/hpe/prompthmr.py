"""PromptHMRBackend - PromptHMR 人体运动估计后端

PromptHMR 使用自己的图像模型提取特征（2通道: smpl_token + loc_token），
然后通过 PromptHMR 视频头（GVHMR 改进版）进行时序推理。

输出格式与 GVHMR 完全一致：
  - smpl_params_global: 世界坐标系(ay)下的 global_orient, body_pose, betas, transl
  - smpl_params_incam: 相机坐标系下的 global_orient, body_pose, betas, transl
  - static_conf_logits 用于后续滑步消除

该后端只执行推理（不做后处理），后处理由独立的 SkatingRemovalComponent 完成。

子步骤缓存：
  VitPose 关键点检测和 PromptHMR 图像模型特征提取是最耗时的两个子步骤。
  当 substep_cache_dir 被设置时，这两步的结果会被缓存到磁盘，
  下次处理同一序列（相同图像 + bbox）时直接加载，跳过推理。
"""

import sys
import os
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import torch

from lib.pipeline.core.component import Backend


class PromptHMRBackend(Backend):
    """
    PromptHMR 后端 - 基于 PromptHMR 图像特征的全局人体运动估计

    使用 PromptHMR 图像模型提取 2 通道特征，再通过 PromptHMR 视频头
    估计世界坐标系和相机坐标系下的 SMPL 参数。

    Config:
        prompthmr_root: PromptHMR 项目根目录
        gvhmr_root: GVHMR 项目根目录（用于共享工具函数）
        static_cam: 是否假设静态相机（跳过 VO）
        f_mm: 相机焦距 (mm)，None 则使用默认估计
        phmr_ckpt: PromptHMR 图像模型 checkpoint 路径
        phmr_vid_cfg: PromptHMR 视频头配置文件路径
        phmr_vid_ckpt: PromptHMR 视频头 checkpoint 路径
        mask_prompt: 是否使用 mask prompt（默认 True）
        img_size: PromptHMR 输入图像尺寸（默认 896）
    """

    DEFAULT_CONFIG = {
        'prompthmr_root': '../PromptHMR',
        'gvhmr_root': 'thirdparty/GVHMR',
        'static_cam': True,
        'f_mm': None,
        'phmr_ckpt': 'data/pretrain/phmr/checkpoint.ckpt',
        'phmr_vid_cfg': 'data/pretrain/phmr_vid/prhmr_release_002.yaml',
        'phmr_vid_ckpt': 'data/pretrain/phmr_vid/prhmr_release_002.ckpt',
        'mask_prompt': True,
        'img_size': 896,
        # 子步骤缓存配置
        'substep_cache_dir': None,       # 缓存目录，None 则不启用子步骤缓存
        'substep_cache_use': True,       # 是否读取已有缓存
        'substep_cache_overwrite': False, # 是否覆盖已有缓存
    }

    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)

        self.prompthmr_root = self.config['prompthmr_root']
        self.gvhmr_root = self.config['gvhmr_root']
        self.static_cam = self.config['static_cam']
        self.f_mm = self.config['f_mm']
        self.phmr_ckpt = self.config['phmr_ckpt']
        self.phmr_vid_cfg = self.config['phmr_vid_cfg']
        self.phmr_vid_ckpt = self.config['phmr_vid_ckpt']
        self.mask_prompt = self.config['mask_prompt']
        self.img_size = self.config['img_size']

        # 子步骤缓存
        self.substep_cache_dir = self.config['substep_cache_dir']
        self.substep_cache_use = self.config['substep_cache_use']
        self.substep_cache_overwrite = self.config['substep_cache_overwrite']

        self.phmr_model = None
        self.vid_head = None
        self.smplx_model = None

    def setup(self):
        """初始化 PromptHMR 图像模型和视频头"""
        prompthmr_abs = os.path.abspath(self.prompthmr_root)
        gvhmr_abs = os.path.abspath(self.gvhmr_root)

        # 添加路径
        # 注意：phmr_gvhmr_path 必须在 gvhmr_abs 之前，
        # 以确保 hmr4d 包从 PromptHMR 的版本加载（包含 NetworkEncoderRoPE 直接构造），
        # 而不是 thirdparty/GVHMR 的旧版（使用 hydra.instantiate，会因缺少 _target_ 而失败）
        phmr_gvhmr_path = os.path.join(prompthmr_abs, 'pipeline', 'gvhmr')
        # 按优先级设置 sys.path：phmr_gvhmr_path > prompthmr_abs > gvhmr_abs
        for p in [gvhmr_abs, prompthmr_abs, phmr_gvhmr_path]:
            if p in sys.path:
                sys.path.remove(p)
        sys.path.insert(0, gvhmr_abs)
        sys.path.insert(0, prompthmr_abs)
        sys.path.insert(0, phmr_gvhmr_path)

        original_cwd = os.getcwd()
        os.chdir(prompthmr_abs)

        try:
            # 1. 加载 PromptHMR 图像模型
            from prompt_hmr import load_model as load_phmr
            self.logger.info(f"Loading PromptHMR image model from {self.phmr_ckpt}...")
            self.phmr_model = load_phmr(self.phmr_ckpt)
            self.logger.info("PromptHMR image model loaded")

            # 2. 加载 PromptHMR 视频头
            from omegaconf import OmegaConf
            import importlib.util

            # 直接加载 gvhmr_pl_demo 模块，绕过 pipeline/__init__.py（避免触发 droid_slam 导入）
            # 清除可能已缓存的 hmr4d 模块（如被 thirdparty/GVHMR 版本占据），
            # 确保从 PromptHMR 版本重新加载
            hmr4d_modules = [k for k in sys.modules if k.startswith('hmr4d')]
            for mod_name in hmr4d_modules:
                del sys.modules[mod_name]

            demo_pl_path = os.path.join(prompthmr_abs, 'pipeline', 'gvhmr', 'hmr4d', 'model', 'gvhmr', 'gvhmr_pl_demo.py')
            spec = importlib.util.spec_from_file_location("gvhmr_pl_demo", demo_pl_path)
            demo_pl_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(demo_pl_module)
            DemoPL = demo_pl_module.DemoPL

            self.logger.info(f"Loading PromptHMR video head from {self.phmr_vid_ckpt}...")
            phmr_vid_cfg = OmegaConf.load(self.phmr_vid_cfg)
            self.vid_head = DemoPL(
                pipeline=phmr_vid_cfg.model.pipeline,
                smplx_path='data/body_models/smplx/SMPLX_NEUTRAL.npz',
            )
            self.vid_head = self.vid_head.eval().cuda()
            self.vid_head.load_pretrained_model(self.phmr_vid_ckpt)
            self.logger.info("PromptHMR video head loaded")
        finally:
            os.chdir(original_cwd)

        # 3. 加载 smplx 模型（用于 get_skeleton 计算 offset）
        os.chdir(gvhmr_abs)
        try:
            from hmr4d.utils.smplx_utils import make_smplx
            self.smplx_model = make_smplx("supermotion").cuda()
        finally:
            os.chdir(original_cwd)

        self._is_setup = True
        self.logger.info("PromptHMR backend initialized")

    # === 子步骤缓存 ===

    def _compute_cache_key(self, image_paths: List[str], bboxes: np.ndarray) -> str:
        """根据图像路径列表和 bbox 计算缓存 key（MD5 hash）"""
        h = hashlib.md5()
        # 用图像文件名（不含目录）+ 帧数 作为主要标识
        h.update(str(len(image_paths)).encode())
        for p in image_paths:
            h.update(os.path.basename(p).encode())
        # bbox 内容也参与 hash，防止同一视频不同检测框混淆
        h.update(bboxes.tobytes())
        return h.hexdigest()[:16]

    def _get_substep_cache_path(self, cache_key: str, substep_name: str) -> Optional[str]:
        """获取子步骤缓存文件路径，如果 cache_dir 未设置则返回 None"""
        if self.substep_cache_dir is None:
            return None
        cache_dir = os.path.join(self.substep_cache_dir, 'substep_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{cache_key}_{substep_name}.pt")

    def _load_substep_cache(self, cache_key: str, substep_name: str) -> Optional[torch.Tensor]:
        """尝试加载子步骤缓存，返回 tensor 或 None"""
        if not self.substep_cache_use or self.substep_cache_overwrite:
            return None
        cache_path = self._get_substep_cache_path(cache_key, substep_name)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        try:
            data = torch.load(cache_path, map_location='cpu', weights_only=True)
            self.logger.info(f"Loaded {substep_name} cache: {cache_path} (shape: {data.shape})")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load {substep_name} cache: {e}")
            return None

    def _save_substep_cache(self, cache_key: str, substep_name: str, data: torch.Tensor):
        """保存子步骤缓存到磁盘"""
        cache_path = self._get_substep_cache_path(cache_key, substep_name)
        if cache_path is None:
            return
        try:
            torch.save(data, cache_path)
            self.logger.info(f"Saved {substep_name} cache: {cache_path} (shape: {data.shape})")
        except Exception as e:
            self.logger.warning(f"Failed to save {substep_name} cache: {e}")

    def _run_image_model(self, image_paths, bboxes_xyxy, vitpose_kp2d, K_fullimg):
        """
        运行 PromptHMR 图像模型，提取每帧的 2 通道特征 (smpl_token, loc_token)

        Args:
            image_paths: 图像路径列表
            bboxes_xyxy: (N, 4) xyxy 格式边界框
            vitpose_kp2d: (N, 17/25, 3) VitPose 关键点
            K_fullimg: (N, 3, 3) 相机内参

        Returns:
            features: (N, 2, 1024) PromptHMR 图像特征
        """
        import cv2
        from PIL import Image, ImageOps
        from torchvision.transforms import Normalize, ToTensor, Compose

        normalization = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

        IMG_SIZE = self.img_size
        N = len(image_paths)
        all_features = []

        self.logger.info(f"Running PromptHMR image model on {N} frames...")

        from torch.utils.data import Dataset, DataLoader

        class SingleFrameDataset(Dataset):
            def __init__(ds_self, image_paths, bboxes_xyxy, vitpose_kp2d, K_fullimg, img_size):
                ds_self.image_paths = image_paths
                ds_self.bboxes_xyxy = bboxes_xyxy
                ds_self.vitpose_kp2d = vitpose_kp2d
                ds_self.K_fullimg = K_fullimg
                ds_self.img_size = img_size
                ds_self.normalization = normalization

            def __len__(ds_self):
                return len(ds_self.image_paths)

            def __getitem__(ds_self, idx):
                img = cv2.imread(ds_self.image_paths[idx])
                if img is None:
                    raise FileNotFoundError(f"Failed to load image: {ds_self.image_paths[idx]}")
                img = img[..., ::-1].copy()  # BGR -> RGB

                bbox = ds_self.bboxes_xyxy[idx]  # (4,)
                boxes = torch.from_numpy(bbox).float().unsqueeze(0)  # (1, 4)
                boxes = torch.cat([boxes, torch.ones(1, 1)], dim=-1)  # (1, 5)

                # 关键点 (取前 25 个 joints，如果有)
                kpt = ds_self.vitpose_kp2d[idx]  # (J, 3)
                if isinstance(kpt, np.ndarray):
                    kpt = torch.from_numpy(kpt).float()
                kpt = kpt.unsqueeze(0)  # (1, J, 3)

                cam_int = ds_self.K_fullimg[idx]  # (3, 3)
                if isinstance(cam_int, np.ndarray):
                    cam_int = torch.from_numpy(cam_int).float()
                cam_int = cam_int.unsqueeze(0)  # (1, 3, 3)

                # pad image to IMG_SIZE
                size = np.array([img.shape[1], img.shape[0]])
                scale = ds_self.img_size / max(size)
                offset = (ds_self.img_size - scale * size) / 2

                img_pil = Image.fromarray(img)
                img_pil = ImageOps.contain(img_pil, (ds_self.img_size, ds_self.img_size))
                img_pil = ImageOps.pad(img_pil, size=(ds_self.img_size, ds_self.img_size))
                img_np = np.array(img_pil)

                # 调整 cam_int
                cam_int = cam_int.mean(dim=0, keepdim=True)
                cam_int[:, :2] *= scale
                cam_int[:, :2, -1] += torch.tensor(offset).float()

                # 调整 boxes
                boxes[:, :4] *= scale
                boxes[:, :2] += torch.tensor(offset).float()
                boxes[:, 2:4] += torch.tensor(offset).float()

                # 调整 keypoints
                kpt[:, :, :2] *= scale
                kpt[:, :, :2] += torch.tensor(offset).float()

                image_tensor = ds_self.normalization(img_np)

                item = {
                    'image': image_tensor,
                    'image_cv': torch.tensor(img_np),
                    'boxes': boxes,
                    'kpts': kpt,
                    'cam_int': cam_int,
                    'masks': None,
                }
                return item

        dataset = SingleFrameDataset(image_paths, bboxes_xyxy, vitpose_kp2d, K_fullimg, IMG_SIZE)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                                collate_fn=lambda x: x)

        from tqdm import tqdm
        from torch.amp import autocast

        for batch in tqdm(dataloader, desc="PromptHMR image model"):
            with torch.no_grad(), autocast('cuda'):
                output = self.phmr_model(batch, mask_prompt=self.mask_prompt)

            for bid in range(len(batch)):
                # output[bid]['features'] 形状: (num_persons, 2, 1024)
                # 由于每帧只有 1 个人, 取 [0]
                feat = output[bid]['features'][0]  # (2, 1024)
                all_features.append(feat.cpu())

        features = torch.stack(all_features)  # (N, 2, 1024)
        self.logger.info(f"PromptHMR image features extracted: {features.shape}")
        return features

    def estimate_smpl(
        self,
        image_paths: List[str],
        bboxes: np.ndarray,
        img_focal: float = None,
        img_center: np.ndarray = None,
        K_fullimg_override: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        运行 PromptHMR 推理

        Args:
            image_paths: 图像路径列表
            bboxes: 边界框 [N, 4] 或 [N, K, 5]
            img_focal: 图像焦距（可选）
            img_center: 图像主点（可选）
            K_fullimg_override: 完整的 3x3 相机内参矩阵（可选）

        Returns:
            包含 PromptHMR 全部输出的字典（格式与 GVHMR 一致）
        """
        self.ensure_setup()

        prompthmr_abs = os.path.abspath(self.prompthmr_root)
        gvhmr_abs = os.path.abspath(self.gvhmr_root)
        phmr_gvhmr_path = os.path.join(prompthmr_abs, 'pipeline', 'gvhmr')

        # 确保路径正确：phmr_gvhmr_path 必须在 gvhmr_abs 之前，
        # 以确保 hmr4d 包从 PromptHMR 版本加载（其 __init__.py 有 try/except 保护），
        # 而不是 GVHMR 版本（会无条件导入 relpose.SimpleVO 导致失败）
        for p in [gvhmr_abs, prompthmr_abs, phmr_gvhmr_path]:
            if p in sys.path:
                sys.path.remove(p)
        # 按优先级插入：phmr_gvhmr_path > prompthmr_abs > gvhmr_abs
        sys.path.insert(0, gvhmr_abs)
        sys.path.insert(0, prompthmr_abs)
        sys.path.insert(0, phmr_gvhmr_path)

        # 将图像路径转为绝对路径
        image_paths = [os.path.abspath(p) for p in image_paths]

        original_cwd = os.getcwd()

        try:
            import cv2
            import scipy.signal as signal
            from scipy.ndimage import gaussian_filter1d
            from torch.amp import autocast

            N = len(image_paths)
            first_img = cv2.imread(image_paths[0])
            height, width = first_img.shape[:2]
            self.logger.info(f"PromptHMR processing {N} frames, image size: {width}x{height}")

            # --- 1. 准备 bboxes ---
            if bboxes.ndim == 3:
                bboxes_xyxy = bboxes[:, 0, :4]
            elif bboxes.ndim == 2 and bboxes.shape[1] > 4:
                bboxes_xyxy = bboxes[:, :4]
            else:
                bboxes_xyxy = bboxes

            # 计算缓存 key（基于图像路径 + bbox 内容）
            cache_key = self._compute_cache_key(image_paths, bboxes_xyxy)

            # 切换到 GVHMR 目录（VitPose 等工具需要）
            os.chdir(gvhmr_abs)

            # 确保 hmr4d.utils.preproc 从 PromptHMR 版本加载（有 try/except 保护），
            # 而非 GVHMR 版本（会无条件 import relpose.SimpleVO 导致 ModuleNotFoundError）。
            # 如果 preproc 包已从 GVHMR 版本缓存，先清除再重新导入。
            _preproc_mod = sys.modules.get('hmr4d.utils.preproc')
            if _preproc_mod is not None:
                _mod_file = getattr(_preproc_mod, '__file__', '') or ''
                if 'GVHMR' in _mod_file or phmr_gvhmr_path not in _mod_file:
                    # 来自 GVHMR 版本，清除缓存
                    _to_remove = [k for k in sys.modules if k.startswith('hmr4d.utils.preproc')]
                    for k in _to_remove:
                        del sys.modules[k]

            from hmr4d.utils.preproc.vitpose import VitPoseExtractor
            from hmr4d.utils.preproc.vitfeat_extractor import get_batch
            from hmr4d.utils.geo.hmr_cam import (
                get_bbx_xys_from_xyxy,
                estimate_K,
                create_camera_sensor,
                normalize_kp2d,
            )
            from hmr4d.utils.geo_transform import compute_cam_angvel
            from hmr4d.utils.net_utils import detach_to_cpu

            bbx_xyxy = torch.from_numpy(bboxes_xyxy).float()
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

            # --- 2-3. VitPose 关键点检测（带缓存） ---
            from tqdm import tqdm
            vitpose = self._load_substep_cache(cache_key, 'vitpose')
            if vitpose is not None:
                self.logger.info("VitPose loaded from cache, skipping inference")
            else:
                self.logger.info(f"Loading {N} images for VitPose...")
                imgs_list = []
                for p in tqdm(image_paths, desc="Loading images", leave=False):
                    img = cv2.imread(p)
                    if img is None:
                        raise FileNotFoundError(f"Failed to load image: {p}")
                    imgs_list.append(img[..., ::-1])  # BGR → RGB
                imgs_np = np.stack(imgs_list)
                del imgs_list

                self.logger.info("Running VitPose...")
                vitpose_extractor = VitPoseExtractor(tqdm_leave=False)
                vitpose_imgs, vitpose_bbx_xys = get_batch(imgs_np, bbx_xys, img_ds=1.0, path_type="np")
                vitpose = vitpose_extractor.extract(vitpose_imgs, vitpose_bbx_xys)
                del vitpose_extractor, vitpose_imgs, vitpose_bbx_xys, imgs_np
                torch.cuda.empty_cache()
                self.logger.info(f"VitPose complete: {vitpose.shape}")

                # 保存 VitPose 缓存
                self._save_substep_cache(cache_key, 'vitpose', vitpose)

            # --- 4. 构建相机内参 ---
            self.logger.info("Estimating camera intrinsics...")
            if K_fullimg_override is not None:
                K_np = K_fullimg_override
                if isinstance(K_np, np.ndarray):
                    K_np = torch.from_numpy(K_np).float()
                K_fullimg = K_np.unsqueeze(0).repeat(N, 1, 1) if K_np.ndim == 2 else K_np[:1].repeat(N, 1, 1)
            elif img_focal is not None:
                from hmr4d.utils.geo.hmr_cam import convert_f_to_K
                K_fullimg = convert_f_to_K(img_focal, width, height).repeat(N, 1, 1)
            elif self.f_mm is not None:
                K_fullimg = create_camera_sensor(width, height, self.f_mm)[2].repeat(N, 1, 1)
            else:
                K_fullimg = estimate_K(width, height).repeat(N, 1, 1)

            # --- 5. 运行 PromptHMR 图像模型提取特征（带缓存） ---
            phmr_features = self._load_substep_cache(cache_key, 'phmr_features')
            if phmr_features is not None:
                self.logger.info("PromptHMR image features loaded from cache, skipping inference")
            else:
                # 先将视频头卸载到 CPU 以腾出显存给图像模型（DINOv2 backbone 很大）
                self.vid_head.cpu()
                self.smplx_model.cpu()
                torch.cuda.empty_cache()
                # 确保图像模型在 GPU 上（处理上一个序列后可能被卸到了 CPU）
                self.phmr_model.cuda()

                os.chdir(prompthmr_abs)
                phmr_features = self._run_image_model(
                    image_paths, bboxes_xyxy, vitpose.numpy(), K_fullimg.numpy()
                )  # (N, 2, 1024)
                torch.cuda.empty_cache()

                # 保存 Image Model 缓存
                self._save_substep_cache(cache_key, 'phmr_features', phmr_features)

                # 图像特征提取完毕，将图像模型卸载到 CPU
                self.phmr_model.cpu()
                torch.cuda.empty_cache()

            # 确保视频头和 smplx 在 GPU 上
            self.vid_head.cuda()
            self.smplx_model.cuda()

            # --- 6. 视觉里程计（相机旋转估计） ---
            os.chdir(gvhmr_abs)
            if self.static_cam:
                self.logger.info("Using static camera assumption (skipping VO)")
                R_w2c = torch.eye(3).repeat(N, 1, 1)
            else:
                self.logger.info("Running Visual Odometry (sift)...")
                # relpose 模块仅存在于 GVHMR 的 hmr4d 中。由于 hmr4d 已被
                # PromptHMR 版本加载（namespace package 的 __path__ 仅指向
                # PromptHMR 目录），必须临时清除整个 hmr4d 包链的缓存，
                # 让 Python 从 GVHMR 路径重新发现所有子包。
                _orig_path = sys.path.copy()
                _orig_modules = {k: v for k, v in sys.modules.items()
                                 if k == 'hmr4d' or k.startswith('hmr4d.')}
                # 清除所有 hmr4d 模块缓存
                for k in list(_orig_modules.keys()):
                    del sys.modules[k]
                # 把 phmr_gvhmr_path 暂时移除，让 gvhmr_abs 优先
                if phmr_gvhmr_path in sys.path:
                    sys.path.remove(phmr_gvhmr_path)
                if prompthmr_abs in sys.path:
                    sys.path.remove(prompthmr_abs)
                if gvhmr_abs not in sys.path:
                    sys.path.insert(0, gvhmr_abs)

                from hmr4d.utils.preproc.relpose.utils import focal_length_from_mm
                from hmr4d.utils.preproc.relpose.matcher_wrapper import Matcher
                from hmr4d.utils.preproc.relpose.solver_two_view import (
                    TwoPairSolver, CameraParams as VOCameraParams, interpolate_missing_frames,
                )

                # 保存 relpose 相关模块（后面需要使用）
                _relpose_modules = {k: v for k, v in sys.modules.items()
                                    if 'relpose' in k}
                # 恢复 sys.path 和 sys.modules（还原 PromptHMR 版本的 hmr4d）
                sys.path[:] = _orig_path
                # 先清除 GVHMR 版本的 hmr4d 缓存
                for k in [k for k in sys.modules if k == 'hmr4d' or k.startswith('hmr4d.')]:
                    del sys.modules[k]
                # 恢复 PromptHMR 版本的 hmr4d 缓存
                sys.modules.update(_orig_modules)
                # 补回 relpose 模块，确保后续使用不出问题
                sys.modules.update(_relpose_modules)

                vo_scale = 0.5
                vo_step = 8
                f_mm = 24 if self.f_mm is None else self.f_mm

                frames = np.stack([
                    cv2.resize(cv2.imread(p), (0, 0), fx=vo_scale, fy=vo_scale)
                    for p in image_paths
                ])

                sample_idxs = np.arange(0, N, vo_step)
                if sample_idxs[-1] != N - 1:
                    sample_idxs = np.concatenate([sample_idxs, [N - 1]])
                sampled_frames = frames[sample_idxs]
                _, H_vo, W_vo, _ = sampled_frames.shape

                matcher = Matcher('sift')
                camera_params = VOCameraParams(W_vo, H_vo, focal_length=focal_length_from_mm(W_vo, H_vo, f_mm))
                solver = TwoPairSolver(camera_params, solver="cv2")

                T_w2c_list = [np.eye(4)]
                prev_frame = sampled_frames[0]
                for fi in tqdm(range(1, len(sampled_frames)), desc="SimpleVO"):
                    curr_frame = sampled_frames[fi]
                    pts0, pts1 = matcher.match_np(prev_frame, curr_frame)
                    T_delta = solver.solve(pts0, pts1)
                    T_w2c_list.append(T_delta @ T_w2c_list[-1])
                    prev_frame = curr_frame

                vo_results = interpolate_missing_frames(T_w2c_list, sample_idxs)
                R_w2c = torch.from_numpy(vo_results[:, :3, :3])
                del frames, sampled_frames

            # --- 7. 平滑 bbox 并准备视频头输入 ---
            self.logger.info("Preparing video head input...")
            bbx_xys_np = bbx_xys.numpy()
            smoothed = np.array([signal.medfilt(param, 11) for param in bbx_xys_np.T]).T
            bbx_xys_smooth = np.array([gaussian_filter1d(traj, 3) for traj in smoothed.T]).T
            bbx_xys_smooth = torch.from_numpy(bbx_xys_smooth).float()

            cam_angvel = compute_cam_angvel(R_w2c)
            vitpose_norm = normalize_kp2d(vitpose, bbx_xys_smooth).float()

            batch = {
                "length": torch.tensor([N]),
                "obs": vitpose_norm[None],
                "bbx_xys": bbx_xys_smooth[None],
                "K_fullimg": K_fullimg[None],
                "cam_angvel": cam_angvel[None],
                "f_imgseq": phmr_features[None],  # (1, N, 2, 1024)
            }
            batch = {k: v.cuda() for k, v in batch.items()}

            # --- 8. 视频头推理（两次 forward，参考 phmr_vid.py） ---
            self.logger.info("Running PromptHMR video head inference...")
            os.chdir(prompthmr_abs)

            # 第一次：带关键点（用于 translation）
            with torch.no_grad(), autocast('cuda'):
                output_w_kpts = self.vid_head.pipeline.forward(
                    batch, train=False, postproc=False, static_cam=self.static_cam
                )

            # 第二次：不带关键点（用于 pose/shape）
            batch_no_kpts = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch_no_kpts['obs'] = torch.zeros_like(batch['obs'])
            with torch.no_grad(), autocast('cuda'):
                output_no_kpts = self.vid_head.pipeline.forward(
                    batch_no_kpts, train=False, postproc=False, static_cam=self.static_cam
                )

            self.logger.info("PromptHMR video head inference complete")

            # --- 9. 组装输出（格式与 GVHMR 一致） ---
            # 使用不带关键点的 pose/shape, 带关键点的 translation
            pred = {
                "smpl_params_global": {
                    "body_pose": output_no_kpts["pred_smpl_params_global"]["body_pose"][0].cpu(),
                    "betas": output_no_kpts["pred_smpl_params_global"]["betas"][0].cpu(),
                    "global_orient": output_no_kpts["pred_smpl_params_global"]["global_orient"][0].cpu(),
                    "transl": output_w_kpts["pred_smpl_params_global"]["transl"][0].cpu(),
                },
                "smpl_params_incam": {
                    "body_pose": output_no_kpts["pred_smpl_params_incam"]["body_pose"][0].cpu(),
                    "betas": output_no_kpts["pred_smpl_params_incam"]["betas"][0].cpu(),
                    "global_orient": output_no_kpts["pred_smpl_params_incam"]["global_orient"][0].cpu(),
                    "transl": output_w_kpts["pred_smpl_params_incam"]["transl"][0].cpu(),
                },
            }

            static_conf_logits = output_no_kpts["model_output"]["static_conf_logits"][0].cpu()

            # --- 10. 计算 skeleton offset ---
            os.chdir(gvhmr_abs)
            betas_global = pred["smpl_params_global"]["betas"]  # (F, 10)
            skeleton_offset = self.smplx_model.get_skeleton(
                betas_global[0:1].cuda()
            )[0, 0].cpu()

            self.logger.info(
                f"PromptHMR inference complete. "
                f"global orient shape: {pred['smpl_params_global']['global_orient'].shape}, "
                f"skeleton offset: {skeleton_offset}"
            )

            return {
                'smpl_params_global': pred['smpl_params_global'],
                'smpl_params_incam': pred['smpl_params_incam'],
                'static_conf_logits': static_conf_logits,
                'K_fullimg': K_fullimg,
                'skeleton_offset': skeleton_offset,
                'vitpose_kp2d': vitpose,  # (N, 17, 3) COCO-17 keypoints from ViTPose
            }
        finally:
            os.chdir(original_cwd)

    def cleanup(self):
        """清理资源"""
        self.phmr_model = None
        self.vid_head = None
        self.smplx_model = None
        super().cleanup()
