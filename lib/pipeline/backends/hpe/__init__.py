"""HPE (Human Pose Estimation) Backends"""

from .vimo import VIMOBackend
from .gt_smpl import GTSmplBackend
from .gvhmr import GVHMRBackend
from .prompthmr import PromptHMRBackend
from .prompthmr_imgonly import PromptHMRImgOnlyBackend

__all__ = ['VIMOBackend', 'GTSmplBackend', 'GVHMRBackend', 'PromptHMRBackend', 'PromptHMRImgOnlyBackend']


