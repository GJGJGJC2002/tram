"""DetectionComponent - 人体检测组件"""

from typing import Dict, Any, List
import numpy as np
import cv2
from tqdm import tqdm

from lib.pipeline.core.component import BackendComponent
from lib.pipeline.core.data import PipelineData
from lib.pipeline.backends.detection import VitDetBackend


class DetectionComponent(BackendComponent):
    """
    人体检测组件
    
    支持多种检测后端，从图像中检测人体边界框。
    
    Config:
        backend: 检测后端类型 ('vitdet', 'yolo', ...)
        threshold: 检测置信度阈值
        min_size: 最小边界框尺寸
        max_detections: 每帧最大检测数量
        arrange_mode: 边界框排序模式 ('size', 'score', 'none')
    """
    
    COMPONENT_TYPE = "detection"
    
    BACKENDS = {
        'vitdet': VitDetBackend,
    }
    
    DEFAULT_BACKEND = 'vitdet'
    
    DEFAULT_CONFIG = {
        'threshold': 0.5,
        'min_size': 100,
        'max_detections': 10,
        'arrange_mode': 'size',
    }
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)
        
        self.min_size = self.config['min_size']
        self.max_detections = self.config['max_detections']
        self.arrange_mode = self.config['arrange_mode']
    
    def validate_input(self, data: PipelineData) -> bool:
        """验证输入：需要图像或图像路径"""
        return data.images is not None or len(data.image_paths) > 0
    
    def execute(self, data: PipelineData) -> PipelineData:
        """
        执行检测
        
        从图像中检测人体，结果存储在 data.bboxes 中。
        """
        # 获取图像
        if data.images is not None:
            images = data.images
        else:
            images = self._load_images(data.image_paths)
            data.images = images
        
        num_frames = len(images)
        all_boxes = []
        
        self.logger.info(f"Detecting humans in {num_frames} frames...")
        
        for img in tqdm(images, desc='Detection'):
            boxes, scores = self.backend.detect(img)
            
            # 组合 boxes 和 scores
            if len(boxes) > 0:
                boxes_with_scores = np.hstack([boxes, scores[:, None]])
                boxes_with_scores = self._arrange_boxes(boxes_with_scores)
            else:
                boxes_with_scores = np.zeros((0, 5))
            
            all_boxes.append(boxes_with_scores)
        
        # 转换为统一格式
        # 找到最大检测数量，以便 padding
        max_det = max(len(b) for b in all_boxes) if all_boxes else 0
        max_det = min(max_det, self.max_detections)
        
        if max_det > 0:
            padded_boxes = np.zeros((num_frames, max_det, 5))
            for i, boxes in enumerate(all_boxes):
                n = min(len(boxes), max_det)
                if n > 0:
                    padded_boxes[i, :n] = boxes[:n]
            data.bboxes = padded_boxes
        else:
            data.bboxes = np.zeros((num_frames, 1, 5))
        
        # 记录统计信息
        total_detections = sum(len(b) for b in all_boxes)
        data.metadata['detection_stats'] = {
            'total_frames': num_frames,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / num_frames if num_frames > 0 else 0,
        }
        
        self.logger.info(f"Detected {total_detections} humans in {num_frames} frames")
        
        return data
    
    def _load_images(self, image_paths: List[str]) -> np.ndarray:
        """加载图像"""
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            images.append(img)
        return np.array(images)
    
    def _arrange_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """
        排序和筛选边界框
        
        Args:
            boxes: [K, 5] (x1, y1, x2, y2, score)
        
        Returns:
            排序和筛选后的边界框
        """
        if len(boxes) == 0:
            return boxes
        
        # 过滤小框
        if self.min_size > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            sizes = np.minimum(widths, heights)
            valid_mask = sizes >= self.min_size
            boxes = boxes[valid_mask]
        
        if len(boxes) == 0:
            return boxes
        
        # 排序
        if self.arrange_mode == 'size':
            # 按面积从大到小排序
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            sorted_idx = np.argsort(-areas)
            boxes = boxes[sorted_idx]
        elif self.arrange_mode == 'score':
            # 按置信度从高到低排序
            sorted_idx = np.argsort(-boxes[:, 4])
            boxes = boxes[sorted_idx]
        
        # 限制数量
        if self.max_detections > 0:
            boxes = boxes[:self.max_detections]
        
        return boxes


