"""
Phase 6: Advanced Metrics Calculator
Comprehensive metrics calculation for classification and segmentation tasks.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    kappa: float
    auc_roc: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: Dict
    confidence_stats: Dict[str, float]


@dataclass
class SegmentationMetrics:
    """Container for segmentation metrics"""
    dice_coefficient: float
    iou_score: float
    pixel_accuracy: float
    precision: float
    recall: float
    per_lesion_dice: Dict[str, float]
    per_lesion_iou: Dict[str, float]


@dataclass
class EvaluationResults:
    """Complete evaluation results"""
    classification: ClassificationMetrics
    segmentation: SegmentationMetrics
    combined_score: float
    dataset_name: str
    model_name: str
    timestamp: str


class MetricsCalculator:
    """Advanced metrics calculator for multi-task DR models"""
    
    def __init__(self, 
                 num_classes: int = 5,
                 class_names: Optional[List[str]] = None,
                 lesion_types: Optional[List[str]] = None):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of DR severity classes
            class_names: Names of DR severity classes
            lesion_types: Names of lesion types for segmentation
        """
        self.num_classes = num_classes
        self.class_names = class_names or [
            'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'
        ]
        self.lesion_types = lesion_types or [
            'microaneurysms', 'hemorrhages', 'hard_exudates', 'soft_exudates'
        ]
        
    def calculate_classification_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_prob: np.ndarray) -> ClassificationMetrics:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            ClassificationMetrics object
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Quadratic weighted kappa for ordinal classification
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        # ROC-AUC (one-vs-rest for multiclass)
        try:
            if self.num_classes > 2:
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                auc_roc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
            else:
                auc_roc = roc_auc_score(y_true, y_prob[:, 1])
        except Exception as e:
            logger.warning(f"Failed to calculate ROC-AUC: {e}")
            auc_roc = 0.0
        
        # Per-class accuracy
        per_class_acc = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = y_true == i
            if np.any(class_mask):
                class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
                per_class_acc[class_name] = float(class_acc)
            else:
                per_class_acc[class_name] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Confidence statistics
        confidence_stats = self._calculate_confidence_stats(y_prob)
        
        return ClassificationMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            kappa=float(kappa),
            auc_roc=float(auc_roc),
            per_class_accuracy=per_class_acc,
            confusion_matrix=cm,
            classification_report=report,
            confidence_stats=confidence_stats
        )
    
    def calculate_segmentation_metrics(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray) -> SegmentationMetrics:
        """
        Calculate comprehensive segmentation metrics
        
        Args:
            y_true: Ground truth masks [B, C, H, W] or [B, H, W]
            y_pred: Predicted masks [B, C, H, W] or [B, H, W]
            
        Returns:
            SegmentationMetrics object
        """
        # Ensure correct dimensions
        if y_true.ndim == 3:
            y_true = np.expand_dims(y_true, axis=1)
        if y_pred.ndim == 3:
            y_pred = np.expand_dims(y_pred, axis=1)
        
        # Convert to binary predictions if needed
        if y_pred.dtype != bool:
            y_pred = y_pred > 0.5
        if y_true.dtype != bool:
            y_true = y_true > 0.5
        
        # Calculate overall metrics
        dice_coeff = self._calculate_dice_coefficient(y_true, y_pred)
        iou_score = self._calculate_iou(y_true, y_pred)
        pixel_acc = self._calculate_pixel_accuracy(y_true, y_pred)
        precision = self._calculate_segmentation_precision(y_true, y_pred)
        recall = self._calculate_segmentation_recall(y_true, y_pred)
        
        # Per-lesion metrics
        per_lesion_dice = {}
        per_lesion_iou = {}
        
        for i, lesion_type in enumerate(self.lesion_types):
            if i < y_true.shape[1]:  # Check if this channel exists
                lesion_true = y_true[:, i]
                lesion_pred = y_pred[:, i]
                
                per_lesion_dice[lesion_type] = float(
                    self._calculate_dice_coefficient(lesion_true, lesion_pred)
                )
                per_lesion_iou[lesion_type] = float(
                    self._calculate_iou(lesion_true, lesion_pred)
                )
        
        return SegmentationMetrics(
            dice_coefficient=float(dice_coeff),
            iou_score=float(iou_score),
            pixel_accuracy=float(pixel_acc),
            precision=float(precision),
            recall=float(recall),
            per_lesion_dice=per_lesion_dice,
            per_lesion_iou=per_lesion_iou
        )
    
    def calculate_bootstrap_confidence_intervals(self,
                                               y_true: np.ndarray,
                                               y_pred: np.ndarray,
                                               y_prob: Optional[np.ndarray] = None,
                                               n_bootstrap: int = 1000,
                                               confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary of metric confidence intervals
        """
        n_samples = len(y_true)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Storage for bootstrap samples
        bootstrap_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'kappa': []
        }
        
        if y_prob is not None:
            bootstrap_metrics['auc_roc'] = []
        
        # Bootstrap sampling
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics for this bootstrap sample
            try:
                bootstrap_metrics['accuracy'].append(
                    accuracy_score(y_true_boot, y_pred_boot)
                )
                bootstrap_metrics['precision'].append(
                    precision_score(y_true_boot, y_pred_boot, average='macro', zero_division=0)
                )
                bootstrap_metrics['recall'].append(
                    recall_score(y_true_boot, y_pred_boot, average='macro', zero_division=0)
                )
                bootstrap_metrics['f1_score'].append(
                    f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0)
                )
                bootstrap_metrics['kappa'].append(
                    cohen_kappa_score(y_true_boot, y_pred_boot, weights='quadratic')
                )
                
                if y_prob is not None:
                    y_prob_boot = y_prob[indices]
                    if self.num_classes > 2:
                        y_true_bin = label_binarize(y_true_boot, classes=range(self.num_classes))
                        auc = roc_auc_score(y_true_bin, y_prob_boot, average='macro', multi_class='ovr')
                    else:
                        auc = roc_auc_score(y_true_boot, y_prob_boot[:, 1])
                    bootstrap_metrics['auc_roc'].append(auc)
                    
            except Exception as e:
                logger.warning(f"Bootstrap sample failed: {e}")
                continue
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric_name, values in bootstrap_metrics.items():
            if values:
                lower = np.percentile(values, lower_percentile)
                upper = np.percentile(values, upper_percentile)
                confidence_intervals[metric_name] = (float(lower), float(upper))
            else:
                confidence_intervals[metric_name] = (0.0, 0.0)
        
        return confidence_intervals
    
    def _calculate_confidence_stats(self, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate prediction confidence statistics"""
        # Max probability as confidence
        confidences = np.max(y_prob, axis=1)
        
        return {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'median_confidence': float(np.median(confidences)),
            'low_confidence_ratio': float(np.mean(confidences < 0.5)),
            'high_confidence_ratio': float(np.mean(confidences > 0.9))
        }
    
    def _calculate_dice_coefficient(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Dice coefficient"""
        intersection = np.logical_and(y_true, y_pred).sum()
        total = y_true.sum() + y_pred.sum()
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * intersection / total
    
    def _calculate_iou(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Intersection over Union"""
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def _calculate_pixel_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate pixel-wise accuracy"""
        return np.mean(y_true == y_pred)
    
    def _calculate_segmentation_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate segmentation precision"""
        true_positives = np.logical_and(y_true, y_pred).sum()
        predicted_positives = y_pred.sum()
        
        if predicted_positives == 0:
            return 1.0 if true_positives == 0 else 0.0
        
        return true_positives / predicted_positives
    
    def _calculate_segmentation_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate segmentation recall (sensitivity)"""
        true_positives = np.logical_and(y_true, y_pred).sum()
        actual_positives = y_true.sum()
        
        if actual_positives == 0:
            return 1.0 if true_positives == 0 else 0.0
        
        return true_positives / actual_positives
    
    def export_metrics_to_csv(self, 
                             results: List[EvaluationResults], 
                             output_path: str) -> None:
        """Export evaluation results to CSV format"""
        import pandas as pd
        
        # Prepare data for CSV
        data = []
        for result in results:
            row = {
                'model_name': result.model_name,
                'dataset': result.dataset_name,
                'timestamp': result.timestamp,
                'combined_score': result.combined_score,
                
                # Classification metrics
                'cls_accuracy': result.classification.accuracy,
                'cls_precision': result.classification.precision,
                'cls_recall': result.classification.recall,
                'cls_f1_score': result.classification.f1_score,
                'cls_kappa': result.classification.kappa,
                'cls_auc_roc': result.classification.auc_roc,
                
                # Segmentation metrics
                'seg_dice': result.segmentation.dice_coefficient,
                'seg_iou': result.segmentation.iou_score,
                'seg_pixel_acc': result.segmentation.pixel_accuracy,
                'seg_precision': result.segmentation.precision,
                'seg_recall': result.segmentation.recall
            }
            
            # Add per-class accuracies
            for class_name, acc in result.classification.per_class_accuracy.items():
                row[f'cls_acc_{class_name.lower().replace(" ", "_")}'] = acc
            
            # Add per-lesion metrics
            for lesion, dice in result.segmentation.per_lesion_dice.items():
                row[f'seg_dice_{lesion}'] = dice
            for lesion, iou in result.segmentation.per_lesion_iou.items():
                row[f'seg_iou_{lesion}'] = iou
            
            data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Metrics exported to {output_path}")
