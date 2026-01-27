"""VitDet Backend - ViT-based Detection"""

from typing import Dict, Any, Tuple, List
import numpy as np
import torch

from lib.pipeline.core.component import Backend


class VitDetBackend(Backend):
    """
    ViTDet 检测后端
    
    使用 Detectron2 的 ViTDet 模型进行人体检测。
    """
    
    DEFAULT_CONFIG = {
        'config_path': 'data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py',
        'checkpoint_url': 'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl',
        'threshold': 0.5,
        'person_class_id': 0,  # COCO 中人的类别 ID
    }
    
    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)
        
        self.threshold = self.config['threshold']
        self.person_class_id = self.config['person_class_id']
        self.detector = None
    
    def setup(self):
        """初始化 ViTDet 模型"""
        from detectron2.config import LazyConfig
        from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
        
        cfg_path = self.config['config_path']
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = self.config['checkpoint_url']
        
        # 设置检测阈值
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        self._is_setup = True
        self.logger.info("VitDet backend initialized")
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测单张图像中的人体
        
        Args:
            image: BGR 格式的图像 [H, W, 3]
        
        Returns:
            boxes: 检测框 [K, 4] (x1, y1, x2, y2)
            scores: 置信度分数 [K]
        """
        self.ensure_setup()
        
        with torch.no_grad():
            det_out = self.detector(image)
            det_instances = det_out['instances']
            
            # 筛选人类检测结果
            valid_idx = (
                (det_instances.pred_classes == self.person_class_id) & 
                (det_instances.scores > self.threshold)
            )
            
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            scores = det_instances.scores[valid_idx].cpu().numpy()
        
        return boxes, scores
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        批量检测
        
        Args:
            images: 图像列表
        
        Returns:
            检测结果列表 [(boxes, scores), ...]
        """
        results = []
        for image in images:
            boxes, scores = self.detect(image)
            results.append((boxes, scores))
        return results
    
    def cleanup(self):
        """清理资源"""
        self.detector = None
        super().cleanup()


