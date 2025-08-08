"""
Metrics utilities for diabetic retinopathy model evaluation.
Provides common evaluation metrics for classification and segmentation tasks.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    cohen_kappa_score, balanced_accuracy_score
)
import logging

logger = logging.getLogger(__name__)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice coefficient for segmentation.
    
    Args:
        pred: Predicted segmentation masks [B, H, W] or [B, C, H, W]
        target: Ground truth masks [B, H, W] or [B, C, H, W]
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient tensor
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    pred = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    pred = (pred > 0.5).float()
    target = target.float()
    
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Calculate Dice score for segmentation (alias for dice_coefficient).
    
    Args:
        pred: Predicted segmentation masks [B, H, W] or [B, C, H, W]
        target: Ground truth masks [B, H, W] or [B, C, H, W]
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice score as float
    """
    return dice_coefficient(pred, target, smooth).item()


def calculate_iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Calculate IoU score for segmentation (alias for iou_score).
    
    Args:
        pred: Predicted segmentation masks
        target: Ground truth masks
        smooth: Smoothing factor
        
    Returns:
        IoU score as float
    """
    return iou_score(pred, target, smooth).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) for segmentation.
    
    Args:
        pred: Predicted segmentation masks
        target: Ground truth masks
        smooth: Smoothing factor
        
    Returns:
        IoU score tensor
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    pred = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    pred = (pred > 0.5).float()
    target = target.float()
    
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def sensitivity_specificity(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate sensitivity (recall) and specificity for segmentation.
    
    Args:
        pred: Predicted segmentation masks
        target: Ground truth masks
        
    Returns:
        Tuple of (sensitivity, specificity)
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    pred = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    pred = (pred > 0.5).float()
    target = target.float()
    
    # True/False Positives/Negatives
    tp = (pred * target).sum(dim=(1, 2))
    tn = ((1 - pred) * (1 - target)).sum(dim=(1, 2))
    fp = (pred * (1 - target)).sum(dim=(1, 2))
    fn = ((1 - pred) * target).sum(dim=(1, 2))
    
    # Sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn + 1e-6)
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp + 1e-6)
    
    return sensitivity.mean(), specificity.mean()


def classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_proba: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC calculation)
        class_names: Names of classes
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary of classification metrics
    """
    # Convert tensors to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_proba is not None and isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Cohen's Kappa
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # AUC if probabilities provided
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                # Multi-class AUC
                metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
            else:
                # Binary AUC
                if y_proba.ndim == 2:
                    y_proba = y_proba[:, 1]  # Use positive class probabilities
                metrics['auc'] = roc_auc_score(y_true, y_proba)
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            metrics['auc'] = 0.0
    
    return metrics


def segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive segmentation metrics.
    
    Args:
        pred: Predicted segmentation masks
        target: Ground truth masks
        threshold: Threshold for binary predictions
        
    Returns:
        Dictionary of segmentation metrics
    """
    metrics = {}
    
    # Dice coefficient
    metrics['dice'] = dice_coefficient(pred, target).item()
    
    # IoU score
    metrics['iou'] = iou_score(pred, target).item()
    
    # Sensitivity and Specificity
    sensitivity, specificity = sensitivity_specificity(pred, target)
    metrics['sensitivity'] = sensitivity.item()
    metrics['specificity'] = specificity.item()
    
    # Pixel accuracy
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()
    
    correct_pixels = (pred_binary == target_binary).float().sum()
    total_pixels = target_binary.numel()
    metrics['pixel_accuracy'] = (correct_pixels / total_pixels).item()
    
    return metrics


def multitask_metrics(
    cls_pred: torch.Tensor,
    cls_target: torch.Tensor,
    seg_pred: Optional[torch.Tensor] = None,
    seg_target: Optional[torch.Tensor] = None,
    cls_proba: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate metrics for multi-task models.
    
    Args:
        cls_pred: Classification predictions
        cls_target: Classification targets
        seg_pred: Segmentation predictions (optional)
        seg_target: Segmentation targets (optional)
        cls_proba: Classification probabilities (optional)
        
    Returns:
        Dictionary of multi-task metrics
    """
    metrics = {}
    
    # Classification metrics
    cls_metrics = classification_metrics(cls_target, cls_pred, cls_proba)
    for key, value in cls_metrics.items():
        metrics[f'cls_{key}'] = value
    
    # Segmentation metrics if available
    if seg_pred is not None and seg_target is not None:
        seg_metrics = segmentation_metrics(seg_pred, seg_target)
        for key, value in seg_metrics.items():
            metrics[f'seg_{key}'] = value
    
    # Combined score
    cls_score = cls_metrics.get('f1_score', 0.0)
    seg_score = seg_metrics.get('dice', 0.0) if seg_pred is not None else 0.0
    
    if seg_pred is not None:
        metrics['combined_score'] = (cls_score + seg_score) / 2.0
    else:
        metrics['combined_score'] = cls_score
    
    return metrics


def diabetic_retinopathy_metrics(
    severity_pred: torch.Tensor,
    severity_target: torch.Tensor,
    severity_proba: Optional[torch.Tensor] = None,
    lesion_pred: Optional[torch.Tensor] = None,
    lesion_target: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate metrics specific to diabetic retinopathy evaluation.
    
    Args:
        severity_pred: DR severity predictions (0-4)
        severity_target: DR severity targets (0-4)
        severity_proba: DR severity probabilities
        lesion_pred: Lesion segmentation predictions
        lesion_target: Lesion segmentation targets
        
    Returns:
        Dictionary of DR-specific metrics
    """
    metrics = {}
    
    # Severity classification metrics
    severity_metrics = classification_metrics(
        severity_target, 
        severity_pred, 
        severity_proba,
        class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
        average='weighted'
    )
    
    for key, value in severity_metrics.items():
        metrics[f'severity_{key}'] = value
    
    # Binary DR detection (No DR vs Any DR)
    binary_target = (severity_target > 0).float()
    binary_pred = (severity_pred > 0).float()
    
    if severity_proba is not None:
        # Sum probabilities for any DR (classes 1-4)
        binary_proba = severity_proba[:, 1:].sum(dim=1) if severity_proba.dim() == 2 else None
    else:
        binary_proba = None
    
    binary_metrics = classification_metrics(
        binary_target, 
        binary_pred, 
        binary_proba,
        average='binary'
    )
    
    for key, value in binary_metrics.items():
        metrics[f'binary_dr_{key}'] = value
    
    # Referable DR detection (Moderate, Severe, Proliferative)
    referable_target = (severity_target >= 2).float()
    referable_pred = (severity_pred >= 2).float()
    
    if severity_proba is not None:
        referable_proba = severity_proba[:, 2:].sum(dim=1) if severity_proba.dim() == 2 else None
    else:
        referable_proba = None
    
    referable_metrics = classification_metrics(
        referable_target,
        referable_pred,
        referable_proba,
        average='binary'
    )
    
    for key, value in referable_metrics.items():
        metrics[f'referable_dr_{key}'] = value
    
    # Lesion segmentation metrics if available
    if lesion_pred is not None and lesion_target is not None:
        lesion_metrics = segmentation_metrics(lesion_pred, lesion_target)
        for key, value in lesion_metrics.items():
            metrics[f'lesion_{key}'] = value
    
    # Overall clinical score
    severity_score = severity_metrics.get('kappa', 0.0)  # Quadratic weighted kappa preferred
    binary_score = binary_metrics.get('f1_score', 0.0)
    referable_score = referable_metrics.get('f1_score', 0.0)
    
    metrics['clinical_score'] = (severity_score + binary_score + referable_score) / 3.0
    
    return metrics


class MetricsTracker:
    """Track metrics during training and validation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics_dict: Dict[str, float], batch_size: int = 1):
        """Update metrics with new batch results."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value * batch_size
            self.counts[key] += batch_size
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        avg_metrics = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                avg_metrics[key] = self.metrics[key] / self.counts[key]
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def get_summary(self) -> str:
        """Get formatted summary of metrics."""
        avg_metrics = self.compute()
        
        summary_lines = []
        for key, value in avg_metrics.items():
            if isinstance(value, float):
                summary_lines.append(f"{key}: {value:.4f}")
            else:
                summary_lines.append(f"{key}: {value}")
        
        return " | ".join(summary_lines)


def compute_class_weights(labels: Union[np.ndarray, torch.Tensor], num_classes: int) -> torch.Tensor:
    """
    Compute class weights for balanced training.
    
    Args:
        labels: Class labels
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Count class occurrences
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Compute weights (inverse frequency)
    total_samples = len(labels)
    weights = total_samples / (num_classes * class_counts + 1e-6)
    
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Example usage
    print("Metrics utilities for diabetic retinopathy evaluation")
    
    # Test classification metrics
    y_true = np.array([0, 1, 2, 3, 4, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 3, 3, 1, 1, 0])
    
    cls_metrics = classification_metrics(y_true, y_pred)
    print("Classification metrics:", cls_metrics)
    
    # Test segmentation metrics
    pred_seg = torch.randn(2, 1, 64, 64)
    target_seg = torch.randint(0, 2, (2, 64, 64)).float()
    
    seg_metrics = segmentation_metrics(pred_seg, target_seg)
    print("Segmentation metrics:", seg_metrics)
