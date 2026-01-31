"""Pipeline Components 层 - 各种功能组件"""

from .detection import DetectionComponent
from .segmentation import SegmentationComponent
from .slam import SLAMComponent
from .hpe import HPEComponent
from .evaluation import EvaluationComponent
from .adjacent_smpl_renderer import AdjacentSMPLRenderer

__all__ = [
    'DetectionComponent',
    'SegmentationComponent',
    'SLAMComponent',
    'HPEComponent',
    'EvaluationComponent',
    'AdjacentSMPLRenderer',
]


