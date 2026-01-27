"""Pipeline Backends 层 - 各种算法后端实现"""

from .detection import VitDetBackend
from .segmentation import SAMBackend
from .slam import DroidSLAMBackend
from .hpe import VIMOBackend

__all__ = [
    'VitDetBackend',
    'SAMBackend',
    'DroidSLAMBackend',
    'VIMOBackend',
]


