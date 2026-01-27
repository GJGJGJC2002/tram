"""VIMO Backend - Video-based Human Motion estimation"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
from torch.utils.data import default_collate
from tqdm import tqdm

from lib.pipeline.core.component import Backend


class VIMOBackend(Backend):
    """
    VIMO 后端 - 视频人体运动估计
    
    使用滑动窗口的方式估计 SMPL 参数。
    """
    
    DEFAULT_CONFIG = {
        'checkpoint': 'data/pretrain/vimo_checkpoint.pth.tar',
        'batch_size': 64,
        'num_workers': 12,
        'window_size': 16,
    }
    
    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)
        
        self.model = None
        self.batch_size = self.config['batch_size']
        self.num_workers = self.config['num_workers']
        self.window_size = self.config['window_size']
    
    def setup(self):
        """初始化 VIMO 模型"""
        from lib.models import get_hmr_vimo
        
        self.model = get_hmr_vimo(checkpoint=self.config['checkpoint'])
        self.model.to(self.device)
        self.model.eval()
        
        self._is_setup = True
        self.logger.info("VIMO backend initialized")
    
    def estimate_smpl(
        self,
        image_paths: List[str],
        bboxes: np.ndarray,
        img_focal: float,
        img_center: np.ndarray,
        mode: str = 'accurate'
    ) -> Dict[str, torch.Tensor]:
        """
        估计 SMPL 参数
        
        Args:
            image_paths: 图像路径列表
            bboxes: 边界框 [N, 4] 或 [N, K, 5]
            img_focal: 图像焦距
            img_center: 图像中心点 [2]
            mode: 'accurate' (重叠滑动窗口) 或 'efficient' (非重叠窗口)
        
        Returns:
            SMPL 参数字典
        """
        self.ensure_setup()
        
        from lib.datasets.image_dataset import ImageDataset
        
        # 处理 bboxes 格式
        if bboxes.ndim == 3:
            # [N, K, 5] -> 取第一个检测
            bboxes = bboxes[:, 0, :4]
        elif bboxes.ndim == 2 and bboxes.shape[1] > 4:
            bboxes = bboxes[:, :4]
        
        # 创建数据集
        db = ImageDataset(
            image_paths, 
            bboxes, 
            img_focal=img_focal,
            img_center=img_center, 
            normalization=True
        )
        
        if mode == 'efficient':
            return self._estimate_efficient(db)
        else:
            return self._estimate_accurate(db)
    
    def _estimate_efficient(self, db) -> Dict[str, torch.Tensor]:
        """非重叠滑动窗口模式（更快但精度略低）"""
        dataloader = torch.utils.data.DataLoader(
            db, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        results = {
            'pred_cam': [],
            'pred_pose': [],
            'pred_shape': [],
            'pred_rotmat': [],
            'pred_trans': []
        }
        
        previous_batch = None
        
        for batch in tqdm(dataloader, desc='SMPL Estimation (efficient)'):
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # 处理最后一个不完整的 batch
            n = len(batch['img'])
            if n < self.batch_size and previous_batch is not None:
                for k in batch:
                    batch[k] = torch.cat([previous_batch[k][n-self.batch_size:], batch[k]], dim=0)
            
            with torch.no_grad():
                out, _ = self.model(batch)
            
            # 处理最后一个 batch 的输出
            if n < self.batch_size and previous_batch is not None:
                for k in out:
                    out[k] = out[k][self.batch_size-n:]
            
            results['pred_cam'].append(out['pred_cam'].cpu())
            results['pred_pose'].append(out['pred_pose'].cpu())
            results['pred_shape'].append(out['pred_shape'].cpu())
            results['pred_rotmat'].append(out['pred_rotmat'].cpu())
            results['pred_trans'].append(out['trans_full'].cpu())
            
            previous_batch = batch
        
        # Concatenate
        for k in results:
            results[k] = torch.cat(results[k])
        
        return results
    
    def _estimate_accurate(self, db) -> Dict[str, torch.Tensor]:
        """重叠滑动窗口模式（更准确）"""
        results = {
            'pred_cam': [],
            'pred_pose': [],
            'pred_shape': [],
            'pred_rotmat': [],
            'pred_trans': []
        }
        
        items = []
        window_size = self.window_size
        center_idx = window_size // 2  # 8 for window_size=16
        
        for i in tqdm(range(len(db)), desc='SMPL Estimation (accurate)'):
            item = db[i]
            items.append(item)
            
            if len(items) < window_size:
                continue
            elif len(items) == window_size:
                batch = default_collate(items)
            else:
                items.pop(0)
                batch = default_collate(items)
            
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                out, _ = self.model.forward(batch)
            
            # 确定输出哪些帧
            if i == window_size - 1:
                # 第一个完整窗口：输出前半部分 + 中心帧
                out = {k: v[:center_idx+1] for k, v in out.items()}
            elif i == len(db) - 1:
                # 最后一个窗口：输出中心帧到末尾
                out = {k: v[center_idx:] for k, v in out.items()}
            else:
                # 中间窗口：只输出中心帧
                out = {k: v[[center_idx]] for k, v in out.items()}
            
            results['pred_cam'].append(out['pred_cam'].cpu())
            results['pred_pose'].append(out['pred_pose'].cpu())
            results['pred_shape'].append(out['pred_shape'].cpu())
            results['pred_rotmat'].append(out['pred_rotmat'].cpu())
            results['pred_trans'].append(out['trans_full'].cpu())
        
        # Concatenate
        for k in results:
            results[k] = torch.cat(results[k])
        
        return results
    
    def cleanup(self):
        """清理资源"""
        self.model = None
        super().cleanup()


