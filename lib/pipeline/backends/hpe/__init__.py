"""HPE (Human Pose Estimation) Backends"""

from .vimo import VIMOBackend
from .gt_smpl import GTSmplBackend

__all__ = ['VIMOBackend', 'GTSmplBackend']


