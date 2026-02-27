"""Pipeline Components 层 - 各种功能组件"""

from .detection import DetectionComponent
from .segmentation import SegmentationComponent
from .slam import SLAMComponent
from .hpe import HPEComponent
from .evaluation import EvaluationComponent
from .adjacent_smpl_renderer import AdjacentSMPLRenderer
from .skating_removal import SkatingRemovalComponent
from .world_transform import WorldTransformComponent
from .depth_scene_refine import DepthSceneRefineComponent

__all__ = [
    'DetectionComponent',
    'SegmentationComponent',
    'SLAMComponent',
    'HPEComponent',
    'EvaluationComponent',
    'AdjacentSMPLRenderer',
    'SkatingRemovalComponent',
    'WorldTransformComponent',
    'DepthSceneRefineComponent',
]


