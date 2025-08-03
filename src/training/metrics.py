"""
Phase 4: Advanced Metrics and Evaluation System
Comprehensive evaluation metrics for diabetic retinopathy multi-task learning
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torchmetrics
from torchmetrics import Metric, Accuracy, Precision, Recall, F1Score, AUROC
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
try:
    from torchmetrics.functional import dice
except ImportError:
    from torchmetrics.functional.segmentation import dice
from sklearn.metrics import cohen_kappa_score
import logging


class QuadraticWeightedKappa(Metric):
    """Quadratic Weighted Kappa for ordinal classification (DR grading)"""
    
    def __init__(self, num_classes: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # State variables
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets"""
        if preds.dim() > 1:
            preds = torch.argmax(preds, dim=1)
        
        self.predictions.append(preds.cpu())
        self.targets.append(target.cpu())
    
    def compute(self) -> torch.Tensor:
        """Compute Quadratic Weighted Kappa"""
        if len(self.predictions) == 0:
            return torch.tensor(0.0)
        
        predictions = torch.cat(self.predictions).numpy()
        targets = torch.cat(self.targets).numpy()
        
        # Compute quadratic weighted kappa
        kappa = cohen_kappa_score(targets, predictions, weights='quadratic')
        return torch.tensor(kappa, dtype=torch.float32)


class DiceCoefficient(Metric):
    """Dice coefficient for segmentation evaluation"""
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.smooth = smooth
        
        self.add_state("dice_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets"""
        # Apply threshold to predictions
        preds_binary = (torch.sigmoid(preds) > self.threshold).float()
        
        # Flatten tensors
        preds_flat = preds_binary.view(-1)
        target_flat = target.view(-1)
        
        # Compute dice coefficient
        intersection = (preds_flat * target_flat).sum()
        union = preds_flat.sum() + target_flat.sum()
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        self.dice_sum += dice_score
        self.num_samples += 1
    
    def compute(self) -> torch.Tensor:
        """Compute average dice coefficient"""
        return self.dice_sum / self.num_samples


class IoUScore(Metric):
    """Intersection over Union (IoU) for segmentation evaluation"""
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.smooth = smooth
        
        self.add_state("iou_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets"""
        # Apply threshold to predictions
        preds_binary = (torch.sigmoid(preds) > self.threshold).float()
        
        # Flatten tensors
        preds_flat = preds_binary.view(-1)
        target_flat = target.view(-1)
        
        # Compute IoU
        intersection = (preds_flat * target_flat).sum()
        union = preds_flat.sum() + target_flat.sum() - intersection
        
        iou_score = (intersection + self.smooth) / (union + self.smooth)
        
        self.iou_sum += iou_score
        self.num_samples += 1
    
    def compute(self) -> torch.Tensor:
        """Compute average IoU score"""
        return self.iou_sum / self.num_samples


class PixelAccuracy(Metric):
    """Pixel-wise accuracy for segmentation"""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        
        self.add_state("correct_pixels", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets"""
        # Apply threshold to predictions
        preds_binary = (torch.sigmoid(preds) > self.threshold).float()
        
        # Count correct pixels
        correct = (preds_binary == target).sum()
        total = target.numel()
        
        self.correct_pixels += correct
        self.total_pixels += total
    
    def compute(self) -> torch.Tensor:
        """Compute pixel accuracy"""
        return self.correct_pixels.float() / self.total_pixels


class Sensitivity(Metric):
    """Sensitivity (Recall) for binary segmentation"""
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.smooth = smooth
        
        self.add_state("true_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets"""
        # Apply threshold to predictions
        preds_binary = (torch.sigmoid(preds) > self.threshold).float()
        
        # Flatten tensors
        preds_flat = preds_binary.view(-1)
        target_flat = target.view(-1)
        
        # Compute TP and FN
        self.true_positives += (preds_flat * target_flat).sum()
        self.false_negatives += ((1 - preds_flat) * target_flat).sum()
    
    def compute(self) -> torch.Tensor:
        """Compute sensitivity"""
        return (self.true_positives + self.smooth) / (self.true_positives + self.false_negatives + self.smooth)


class Specificity(Metric):
    """Specificity for binary segmentation"""
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.smooth = smooth
        
        self.add_state("true_negatives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets"""
        # Apply threshold to predictions
        preds_binary = (torch.sigmoid(preds) > self.threshold).float()
        
        # Flatten tensors
        preds_flat = preds_binary.view(-1)
        target_flat = target.view(-1)
        
        # Compute TN and FP
        self.true_negatives += ((1 - preds_flat) * (1 - target_flat)).sum()
        self.false_positives += (preds_flat * (1 - target_flat)).sum()
    
    def compute(self) -> torch.Tensor:
        """Compute specificity"""
        return (self.true_negatives + self.smooth) / (self.true_negatives + self.false_positives + self.smooth)


@dataclass
class MetricResults:
    """Container for metric results"""
    classification: Dict[str, float]
    segmentation: Dict[str, float]
    combined: Dict[str, float]
    
    def __str__(self) -> str:
        """String representation of results"""
        lines = ["=== Metric Results ==="]
        
        lines.append("\nEVALUATION Classification Metrics:")
        for name, value in self.classification.items():
            lines.append(f"  {name:.<25} {value:.4f}")
        
        lines.append("\nTARGET Segmentation Metrics:")
        for name, value in self.segmentation.items():
            lines.append(f"  {name:.<25} {value:.4f}")
        
        lines.append("\nCombined Combined Metrics:")
        for name, value in self.combined.items():
            lines.append(f"  {name:.<25} {value:.4f}")
        
        return "\n".join(lines)


class AdvancedMetricsCollector:
    """Advanced metrics collection and evaluation system"""
    
    def __init__(self, num_classes: int = 5, device: str = "cuda"):
        self.num_classes = num_classes
        self.device = device
        
        # Classification metrics
        self.cls_metrics = {
            "accuracy": MulticlassAccuracy(num_classes=num_classes).to(device),
            "precision": MulticlassPrecision(num_classes=num_classes, average="macro").to(device),
            "recall": MulticlassRecall(num_classes=num_classes, average="macro").to(device),
            "f1_score": MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
            "kappa": QuadraticWeightedKappa(num_classes=num_classes).to(device),
            "auc_roc": AUROC(task="multiclass", num_classes=num_classes).to(device)
        }
        
        # Segmentation metrics
        self.seg_metrics = {
            "dice": DiceCoefficient().to(device),
            "iou": IoUScore().to(device),
            "pixel_accuracy": PixelAccuracy().to(device),
            "sensitivity": Sensitivity().to(device),
            "specificity": Specificity().to(device)
        }
        
        # Combined metrics tracking
        self.history = {
            "classification": {},
            "segmentation": {},
            "combined": {}
        }
        
        # Initialize history
        for metric_name in self.cls_metrics.keys():
            self.history["classification"][metric_name] = []
        for metric_name in self.seg_metrics.keys():
            self.history["segmentation"][metric_name] = []
        
        self.history["combined"]["combined_score"] = []
        self.history["combined"]["weighted_f1_dice"] = []
        self.history["combined"]["clinical_score"] = []
    
    def reset(self):
        """Reset all metrics"""
        for metric in self.cls_metrics.values():
            metric.reset()
        for metric in self.seg_metrics.values():
            metric.reset()
    
    def update_classification(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update classification metrics"""
        for metric in self.cls_metrics.values():
            metric.update(preds, targets)
    
    def update_segmentation(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update segmentation metrics"""
        for metric in self.seg_metrics.values():
            metric.update(preds, targets)
    
    def update(self, 
               cls_preds: torch.Tensor, 
               cls_targets: torch.Tensor,
               seg_preds: Optional[torch.Tensor] = None,
               seg_targets: Optional[torch.Tensor] = None):
        """Update all metrics"""
        self.update_classification(cls_preds, cls_targets)
        
        if seg_preds is not None and seg_targets is not None:
            self.update_segmentation(seg_preds, seg_targets)
    
    def compute(self) -> MetricResults:
        """Compute all metrics and return results"""
        # Compute classification metrics
        cls_results = {}
        for name, metric in self.cls_metrics.items():
            try:
                value = metric.compute().item()
                cls_results[name] = value
            except Exception as e:
                logging.warning(f"Failed to compute {name}: {e}")
                cls_results[name] = 0.0
        
        # Compute segmentation metrics
        seg_results = {}
        for name, metric in self.seg_metrics.items():
            try:
                value = metric.compute().item()
                seg_results[name] = value
            except Exception as e:
                logging.warning(f"Failed to compute {name}: {e}")
                seg_results[name] = 0.0
        
        # Compute combined metrics
        combined_results = self._compute_combined_metrics(cls_results, seg_results)
        
        # Update history
        for name, value in cls_results.items():
            self.history["classification"][name].append(value)
        for name, value in seg_results.items():
            self.history["segmentation"][name].append(value)
        for name, value in combined_results.items():
            self.history["combined"][name].append(value)
        
        return MetricResults(
            classification=cls_results,
            segmentation=seg_results,
            combined=combined_results
        )
    
    def _compute_combined_metrics(self, cls_results: Dict[str, float], seg_results: Dict[str, float]) -> Dict[str, float]:
        """Compute combined metrics"""
        combined = {}
        
        # Combined score (for early stopping and best model selection)
        cls_score = cls_results.get("kappa", cls_results.get("accuracy", 0.0))
        seg_score = seg_results.get("dice", 0.0)
        combined["combined_score"] = cls_score + seg_score
        
        # Weighted F1-Dice score
        f1_score = cls_results.get("f1_score", 0.0)
        dice_score = seg_results.get("dice", 0.0)
        combined["weighted_f1_dice"] = 0.6 * f1_score + 0.4 * dice_score
        
        # Clinical relevance score (emphasize kappa and sensitivity)
        kappa = cls_results.get("kappa", 0.0)
        sensitivity = seg_results.get("sensitivity", 0.0)
        combined["clinical_score"] = 0.7 * kappa + 0.3 * sensitivity
        
        return combined
    
    def get_best_epoch(self, metric_name: str = "combined_score") -> int:
        """Get epoch with best performance for given metric"""
        if metric_name in self.history["classification"]:
            values = self.history["classification"][metric_name]
        elif metric_name in self.history["segmentation"]:
            values = self.history["segmentation"][metric_name]
        elif metric_name in self.history["combined"]:
            values = self.history["combined"][metric_name]
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        if not values:
            return 0
        
        # Ensure all values are numeric
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                numeric_values.append(0.0)
        
        if not numeric_values:
            return 0
        
        return int(np.argmax(numeric_values))
    
    def get_metric_trend(self, metric_name: str, window: int = 5) -> str:
        """Get trend for a metric over recent epochs"""
        if metric_name in self.history["classification"]:
            values = self.history["classification"][metric_name]
        elif metric_name in self.history["segmentation"]:
            values = self.history["segmentation"][metric_name]
        elif metric_name in self.history["combined"]:
            values = self.history["combined"][metric_name]
        else:
            return "unknown"
        
        if len(values) < window:
            return "insufficient_data"
        
        # Ensure all values are numeric
        numeric_values = []
        for v in values[-window:]:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                numeric_values.append(0.0)
        
        if not numeric_values:
            return "insufficient_data"
        
        try:
            trend = np.polyfit(range(len(numeric_values)), numeric_values, 1)[0]
            
            if trend > 0.001:
                return "improving"
            elif trend < -0.001:
                return "declining"
            else:
                return "stable"
        except Exception:
            return "unknown"
    
    def should_stop_early(self, 
                          patience: int, 
                          monitor_metric: str = "combined_score",
                          min_delta: float = 1e-4) -> bool:
        """Check if training should stop early"""
        if monitor_metric in self.history["classification"]:
            values = self.history["classification"][monitor_metric]
        elif monitor_metric in self.history["segmentation"]:
            values = self.history["segmentation"][monitor_metric]
        elif monitor_metric in self.history["combined"]:
            values = self.history["combined"][monitor_metric]
        else:
            return False
        
        if len(values) < patience + 1:
            return False
        
        # Ensure all values are numeric (convert strings to 0.0)
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                numeric_values.append(0.0)
        
        if not numeric_values:
            return False
            
        best_value = max(numeric_values)
        recent_best = max(numeric_values[-patience:])
        
        return (best_value - recent_best) > min_delta
    
    def generate_report(self) -> str:
        """Generate comprehensive metrics report"""
        if not self.history["combined"]["combined_score"]:
            return "No metrics computed yet."
        
        lines = ["=" * 60]
        lines.append("EVALUATION DIABETIC RETINOPATHY TRAINING METRICS REPORT")
        lines.append("=" * 60)
        
        # Current epoch metrics
        current_cls = {k: v[-1] for k, v in self.history["classification"].items() if v}
        current_seg = {k: v[-1] for k, v in self.history["segmentation"].items() if v}
        current_combined = {k: v[-1] for k, v in self.history["combined"].items() if v}
        
        current_results = MetricResults(
            classification=current_cls,
            segmentation=current_seg,
            combined=current_combined
        )
        
        lines.append(str(current_results))
        
        # Best performance
        lines.append(f"\n🏆 Best Performance:")
        best_epoch = self.get_best_epoch("combined_score")
        lines.append(f"  Best epoch: {best_epoch}")
        lines.append(f"  Best combined score: {max(self.history['combined']['combined_score']):.4f}")
        
        if self.history["classification"]["kappa"]:
            best_kappa = max(self.history["classification"]["kappa"])
            lines.append(f"  Best kappa: {best_kappa:.4f}")
        
        if self.history["segmentation"]["dice"]:
            best_dice = max(self.history["segmentation"]["dice"])
            lines.append(f"  Best dice: {best_dice:.4f}")
        
        # Trends
        lines.append(f"\n📈 Recent Trends:")
        key_metrics = ["kappa", "dice", "combined_score"]
        for metric in key_metrics:
            if metric in self.history["classification"]:
                trend = self.get_metric_trend(metric)
                lines.append(f"  {metric}: {trend}")
            elif metric in self.history["segmentation"]:
                trend = self.get_metric_trend(metric)
                lines.append(f"  {metric}: {trend}")
            elif metric in self.history["combined"]:
                trend = self.get_metric_trend(metric)
                lines.append(f"  {metric}: {trend}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Test metrics collector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics_collector = AdvancedMetricsCollector(num_classes=5, device=device)
    
    # Dummy predictions and targets
    batch_size = 8
    cls_preds = torch.randn(batch_size, 5).to(device)
    cls_targets = torch.randint(0, 5, (batch_size,)).to(device)
    seg_preds = torch.randn(batch_size, 1, 224, 224).to(device)
    seg_targets = torch.randint(0, 2, (batch_size, 1, 224, 224)).float().to(device)
    
    # Update metrics
    metrics_collector.update(cls_preds, cls_targets, seg_preds, seg_targets)
    
    # Compute results
    results = metrics_collector.compute()
    print(results)
    
    # Generate report
    print(metrics_collector.generate_report())
