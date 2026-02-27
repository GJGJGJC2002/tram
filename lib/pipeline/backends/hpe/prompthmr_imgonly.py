"""PromptHMRImgOnlyBackend - PromptHMR 图像模型后端（不经过视频头）

仅使用 PromptHMR 的图像模型逐帧推理，与 PromptHMR 官方评估 (evaluator.py) 完全一致：
  - Pred: SMPL-X forward → smplx2smpl (dense) → SMPL vertices
  - 输出 SMPL vertices、SMPL-X rotmat/betas/transl

不使用 video head，不使用 GVHMR，不做时序推理。
适用于与 PromptHMR 官方单帧评估结果直接对比。
"""

import sys
import os
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import torch

from lib.pipeline.core.component import Backend


class PromptHMRImgOnlyBackend(Backend):
    """
    PromptHMR 图像模型后端（仅单帧推理，不经过视频头）

    Config:
        prompthmr_root: PromptHMR 项目根目录
        phmr_ckpt: PromptHMR 图像模型 checkpoint 路径
        mask_prompt: 是否使用 mask prompt（默认 True）
        img_size: PromptHMR 输入图像尺寸（默认 896）
    """

    DEFAULT_CONFIG = {
        'prompthmr_root': '../PromptHMR',
        'phmr_ckpt': 'data/pretrain/phmr/checkpoint.ckpt',
        'mask_prompt': True,
        'img_size': 896,
        # 子步骤缓存配置
        'substep_cache_dir': None,
        'substep_cache_use': True,
        'substep_cache_overwrite': False,
    }

    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)

        self.prompthmr_root = self.config['prompthmr_root']
        self.phmr_ckpt = self.config['phmr_ckpt']
        self.mask_prompt = self.config['mask_prompt']
        self.img_size = self.config['img_size']

        # 子步骤缓存
        self.substep_cache_dir = self.config['substep_cache_dir']
        self.substep_cache_use = self.config['substep_cache_use']
        self.substep_cache_overwrite = self.config['substep_cache_overwrite']

        self.phmr_model = None

    def setup(self):
        """初始化 PromptHMR 图像模型"""
        prompthmr_abs = os.path.abspath(self.prompthmr_root)

        # 添加 PromptHMR 到 sys.path
        if prompthmr_abs not in sys.path:
            sys.path.insert(0, prompthmr_abs)

        original_cwd = os.getcwd()
        os.chdir(prompthmr_abs)

        try:
            from prompt_hmr import load_model as load_phmr
            self.logger.info(f"Loading PromptHMR image model from {self.phmr_ckpt}...")
            self.phmr_model = load_phmr(self.phmr_ckpt)
            self.logger.info("PromptHMR image model loaded")
        finally:
            os.chdir(original_cwd)

        self._is_setup = True
        self.logger.info("PromptHMR image-only backend initialized")

    # === 子步骤缓存 ===

    def _compute_cache_key(self, image_paths: List[str], bboxes: np.ndarray) -> str:
        h = hashlib.md5()
        h.update(str(len(image_paths)).encode())
        for p in image_paths:
            h.update(os.path.basename(p).encode())
        h.update(bboxes.tobytes())
        return h.hexdigest()[:16]

    def _get_substep_cache_path(self, cache_key: str, substep_name: str) -> Optional[str]:
        if self.substep_cache_dir is None:
            return None
        cache_dir = os.path.join(self.substep_cache_dir, 'substep_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{cache_key}_{substep_name}.pt")

    def _load_substep_cache(self, cache_key: str, substep_name: str) -> Optional[Any]:
        if not self.substep_cache_use or self.substep_cache_overwrite:
            return None
        cache_path = self._get_substep_cache_path(cache_key, substep_name)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        try:
            data = torch.load(cache_path, map_location='cpu', weights_only=False)
            self.logger.info(f"Loaded {substep_name} cache: {cache_path}")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load {substep_name} cache: {e}")
            return None

    def _save_substep_cache(self, cache_key: str, substep_name: str, data: Any):
        cache_path = self._get_substep_cache_path(cache_key, substep_name)
        if cache_path is None:
            return
        try:
            torch.save(data, cache_path)
            self.logger.info(f"Saved {substep_name} cache: {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save {substep_name} cache: {e}")

    def estimate_smpl(
        self,
        image_paths: List[str],
        bboxes: np.ndarray,
        img_focal: float = None,
        img_center: np.ndarray = None,
        K_fullimg_override: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        运行 PromptHMR 图像模型逐帧推理

        与 PromptHMR 官方评估完全一致：
        - 模型输出 SMPL-X vertices → smplx2smpl → SMPL vertices
        - 用 smpl.J_regressor[:24] 回归 joints

        Returns:
            包含逐帧 SMPL 预测结果的字典
        """
        self.ensure_setup()

        prompthmr_abs = os.path.abspath(self.prompthmr_root)
        if prompthmr_abs not in sys.path:
            sys.path.insert(0, prompthmr_abs)

        image_paths = [os.path.abspath(p) for p in image_paths]
        original_cwd = os.getcwd()

        try:
            import cv2
            from PIL import Image, ImageOps
            from torchvision.transforms import Normalize, ToTensor, Compose
            from torch.utils.data import Dataset, DataLoader
            from torch.amp import autocast
            from tqdm import tqdm

            N = len(image_paths)
            first_img = cv2.imread(image_paths[0])
            height, width = first_img.shape[:2]
            self.logger.info(f"PromptHMR img-only processing {N} frames, image size: {width}x{height}")

            # --- 1. 准备 bboxes ---
            if bboxes.ndim == 3:
                bboxes_xyxy = bboxes[:, 0, :4]
            elif bboxes.ndim == 2 and bboxes.shape[1] > 4:
                bboxes_xyxy = bboxes[:, :4]
            else:
                bboxes_xyxy = bboxes

            cache_key = self._compute_cache_key(image_paths, bboxes_xyxy)

            # --- 2. 构建相机内参（用于 PromptHMR 输入） ---
            if K_fullimg_override is not None:
                K_np = K_fullimg_override
                if isinstance(K_np, torch.Tensor):
                    K_np = K_np.numpy()
                if K_np.ndim == 2:
                    K_fullimg = np.tile(K_np[None], (N, 1, 1))
                else:
                    K_fullimg = np.tile(K_np[:1], (N, 1, 1))
            elif img_focal is not None:
                K_fullimg = np.zeros((N, 3, 3), dtype=np.float32)
                K_fullimg[:, 0, 0] = img_focal
                K_fullimg[:, 1, 1] = img_focal
                K_fullimg[:, 0, 2] = width / 2.0
                K_fullimg[:, 1, 2] = height / 2.0
                K_fullimg[:, 2, 2] = 1.0
            else:
                K_fullimg = np.zeros((N, 3, 3), dtype=np.float32)
                focal_est = (width * width + height * height) ** 0.5
                K_fullimg[:, 0, 0] = focal_est
                K_fullimg[:, 1, 1] = focal_est
                K_fullimg[:, 0, 2] = width / 2.0
                K_fullimg[:, 1, 2] = height / 2.0
                K_fullimg[:, 2, 2] = 1.0

            # --- 3. 尝试加载缓存 ---
            cached = self._load_substep_cache(cache_key, 'phmr_imgonly')
            if cached is not None:
                self.logger.info("PromptHMR img-only results loaded from cache")
                return cached

            # --- 4. 逐帧推理 ---
            os.chdir(prompthmr_abs)
            self.phmr_model.cuda()

            normalization = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])

            IMG_SIZE = self.img_size

            class SingleFrameDataset(Dataset):
                def __init__(ds_self, image_paths, bboxes_xyxy, K_fullimg, img_size):
                    ds_self.image_paths = image_paths
                    ds_self.bboxes_xyxy = bboxes_xyxy
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

                    bbox = ds_self.bboxes_xyxy[idx]
                    boxes = torch.from_numpy(bbox).float().unsqueeze(0)
                    boxes = torch.cat([boxes, torch.ones(1, 1)], dim=-1)  # (1, 5)

                    cam_int = ds_self.K_fullimg[idx]
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

                    image_tensor = ds_self.normalization(img_np)

                    item = {
                        'image': image_tensor,
                        'image_cv': torch.tensor(img_np),
                        'boxes': boxes,
                        'kpts': None,
                        'cam_int': cam_int,
                        'masks': None,
                    }
                    return item

            dataset = SingleFrameDataset(image_paths, bboxes_xyxy, K_fullimg, IMG_SIZE)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                                    collate_fn=lambda x: x)

            all_vertices = []      # SMPL-X vertices (10475, 3)
            all_smpl_verts = []    # SMPL vertices (6890, 3) via smplx2smpl
            all_rotmat = []        # (22, 3, 3)
            all_betas = []         # (10,)
            all_transl = []        # (3,)

            self.logger.info(f"Running PromptHMR image model on {N} frames (no video head)...")

            for batch in tqdm(dataloader, desc="PromptHMR img-only"):
                with torch.no_grad(), autocast('cuda'):
                    output = self.phmr_model(batch, mask_prompt=self.mask_prompt)

                for bid in range(len(batch)):
                    out = output[bid]
                    # 取第一个人的结果
                    all_vertices.append(out['vertices'][0].cpu())        # (10475, 3)
                    all_smpl_verts.append(out['smpl_vertices'][0].cpu()) # (6890, 3)
                    all_rotmat.append(out['rotmat'][0].cpu())            # (22, 3, 3)
                    all_betas.append(out['betas'][0].cpu())              # (10,)
                    all_transl.append(out['transl'][0].cpu())            # (3,)

            all_vertices = torch.stack(all_vertices)      # (N, 10475, 3)
            all_smpl_verts = torch.stack(all_smpl_verts)  # (N, 6890, 3)
            all_rotmat = torch.stack(all_rotmat)          # (N, 22, 3, 3)
            all_betas = torch.stack(all_betas)            # (N, 10)
            all_transl = torch.stack(all_transl)          # (N, 3)

            # 获取 smpl.J_regressor[:24] 用于 joints 回归（与官方一致）
            j_regressor = self.phmr_model.smpl.J_regressor[:24].cpu()  # (24, 6890)
            all_smpl_j3d = torch.matmul(j_regressor, all_smpl_verts)   # (N, 24, 3)

            self.logger.info(
                f"PromptHMR img-only complete. "
                f"smpl_verts: {all_smpl_verts.shape}, "
                f"smpl_j3d: {all_smpl_j3d.shape}, "
                f"rotmat: {all_rotmat.shape}"
            )

            K_fullimg_tensor = torch.from_numpy(K_fullimg).float()

            result = {
                'smplx_vertices': all_vertices,     # (N, 10475, 3)
                'smpl_vertices': all_smpl_verts,     # (N, 6890, 3)
                'smpl_j3d': all_smpl_j3d,            # (N, 24, 3)
                'j_regressor': j_regressor,          # (24, 6890) - smpl.J_regressor[:24]
                'rotmat': all_rotmat,                # (N, 22, 3, 3)
                'betas': all_betas,                  # (N, 10)
                'transl': all_transl,                # (N, 3)
                'K_fullimg': K_fullimg_tensor,       # (N, 3, 3)
            }

            # 保存缓存
            self._save_substep_cache(cache_key, 'phmr_imgonly', result)

            return result
        finally:
            os.chdir(original_cwd)

    def cleanup(self):
        self.phmr_model = None
        super().cleanup()
