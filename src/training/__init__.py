"""
Training components for multi-task diabetic retinopathy learning.
"""

from .losses import (
    FocalLoss, 
    DiceLoss, 
    CombinedSegmentationLoss, 
    QuadraticWeightedKappa,
    MultiTaskLoss
)

__all__ = [
    'FocalLoss',
    'DiceLoss', 
    'CombinedSegmentationLoss',
    'QuadraticWeightedKappa',
    'MultiTaskLoss'
]
