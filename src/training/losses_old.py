"""
Loss Functions for Multi-Task Diabetic Retinopathy Learning
Implements classification and segmentation losses with combined multi-task objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Optional, Union, Tuple
from sklearn.metrics import cohen_kappa_score
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    Focuses learning on hard examples by down-weighting easy ones.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class imbalance (None, float, or tensor)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # Base cross entropy loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        
        logger.info(f"FocalLoss initialized: alpha={alpha}, gamma={gamma}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for focal loss.
        
        Args:
            inputs: Predictions of shape (N, C) where C = number of classes
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = self.ce_loss(inputs, targets)
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Measures overlap between predicted and ground truth masks.
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        reduction: str = 'mean',
        include_background: bool = False
    ):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
            include_background: Whether to include background class in loss
        """
        super().__init__()
        
        self.smooth = smooth
        self.reduction = reduction
        self.include_background = include_background
        
        logger.info(f"DiceLoss initialized: smooth={smooth}, include_background={include_background}")
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for dice loss.
        
        Args:
            inputs: Predictions of shape (N, C, H, W)
            targets: Ground truth masks of shape (N, C, H, W) or (N, H, W)
            class_weights: Optional class weights
            
        Returns:
            Dice loss value
        """
        # Handle different target shapes
        if len(targets.shape) == 3:  # (N, H, W)
            targets = targets.unsqueeze(1)  # (N, 1, H, W)
        
        # Apply softmax/sigmoid to inputs
        if inputs.shape[1] > 1:  # Multi-class
            inputs = F.softmax(inputs, dim=1)
        else:  # Binary
            inputs = torch.sigmoid(inputs)
        
        # Flatten spatial dimensions
        inputs_flat = inputs.view(inputs.shape[0], inputs.shape[1], -1)  # (N, C, H*W)
        targets_flat = targets.view(targets.shape[0], targets.shape[1], -1)  # (N, C, H*W)
        
        # Compute dice for each class
        dice_scores = []
        num_classes = inputs.shape[1]
        
        start_idx = 0 if self.include_background else 1
        
        for class_idx in range(start_idx, num_classes):
            # Get class predictions and targets
            pred_class = inputs_flat[:, class_idx, :]  # (N, H*W)
            target_class = targets_flat[:, class_idx, :]  # (N, H*W)
            
            # Compute intersection and union
            intersection = (pred_class * target_class).sum(dim=1)  # (N,)
            pred_sum = pred_class.sum(dim=1)  # (N,)
            target_sum = target_class.sum(dim=1)  # (N,)
            
            # Compute dice coefficient
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            dice_scores.append(dice)
        
        # Stack dice scores
        if dice_scores:
            dice_tensor = torch.stack(dice_scores, dim=1)  # (N, num_classes)
            
            # Apply class weights if provided
            if class_weights is not None:
                if len(class_weights) != dice_tensor.shape[1]:
                    raise ValueError(f"Class weights length {len(class_weights)} doesn't match number of classes {dice_tensor.shape[1]}")
                dice_tensor = dice_tensor * class_weights.unsqueeze(0)
            
            # Compute loss (1 - dice)
            dice_loss = 1.0 - dice_tensor
            
            # Apply reduction
            if self.reduction == 'mean':
                return dice_loss.mean()
            elif self.reduction == 'sum':
                return dice_loss.sum()
            else:
                return dice_loss
        else:
            # No valid classes
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation: Dice + Cross Entropy + Focal
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.3,
        focal_weight: float = 0.2,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize combined segmentation loss.
        
        Args:
            dice_weight: Weight for dice loss
            ce_weight: Weight for cross entropy loss
            focal_weight: Weight for focal loss
            focal_gamma: Gamma parameter for focal loss
            class_weights: Class weights for imbalanced data
        """
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        
        # Initialize individual losses
        self.dice_loss = DiceLoss(smooth=1e-5)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        
        logger.info(f"CombinedSegmentationLoss: dice={dice_weight}, ce={ce_weight}, focal={focal_weight}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for combined loss.
        
        Args:
            inputs: Predictions of shape (N, C, H, W)
            targets: Ground truth of shape (N, H, W) or (N, C, H, W)
            
        Returns:
            Dictionary with individual and total losses
        """
        # Check if binary segmentation (single channel output)
        if inputs.shape[1] == 1:
            # Binary segmentation - handle autocast properly
            # Ensure targets are float and have correct shape
            if len(targets.shape) == 3:  # (N, H, W)
                targets_binary = targets.float().unsqueeze(1)  # (N, 1, H, W)
            else:  # (N, C, H, W)
                targets_binary = targets.float()
            
            # Compute losses for binary case
            dice = self.dice_loss(inputs, targets_binary)
            
            # Use BCEWithLogitsLoss by getting logits from sigmoid outputs
            # More stable than inverse sigmoid
            with torch.cuda.amp.autocast(enabled=False):
                # Convert sigmoid outputs back to logits for BCEWithLogitsLoss
                epsilon = 1e-7
                inputs_clamped = torch.clamp(inputs.float(), epsilon, 1 - epsilon)
                inputs_logits = torch.log(inputs_clamped / (1 - inputs_clamped))
                
                bce_with_logits_loss = nn.BCEWithLogitsLoss()
                ce = bce_with_logits_loss(inputs_logits, targets_binary.float())
            
            # Skip focal loss for binary segmentation to avoid CUDA errors
            focal = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
        else:
            # Multi-class segmentation - original logic
            # Handle target shape for cross entropy and focal loss
            if len(targets.shape) == 4 and targets.shape[1] > 1:  # One-hot encoded
                ce_targets = targets.argmax(dim=1)  # Convert to class indices
            elif len(targets.shape) == 4 and targets.shape[1] == 1:  # Single channel
                ce_targets = targets.squeeze(1).long()
            else:  # Already class indices
                ce_targets = targets.long()
            
            # Compute individual losses
            dice = self.dice_loss(inputs, targets)
            ce = self.ce_loss(inputs, ce_targets)
            focal = self.focal_loss(inputs, ce_targets)
        
        # Combine losses
        total_loss = (self.dice_weight * dice + 
                     self.ce_weight * ce + 
                     self.focal_weight * focal)
        
        return {
            'total': total_loss,
            'dice': dice,
            'cross_entropy': ce,
            'focal': focal
        }

class QuadraticWeightedKappa(nn.Module):
    """
    Quadratic Weighted Kappa loss for ordinal classification (DR grading).
    Penalizes predictions that are far from the true class.
    """
    
    def __init__(self, num_classes: int = 5, reduction: str = 'mean'):
        """
        Initialize quadratic weighted kappa loss.
        
        Args:
            num_classes: Number of classes (DR grades 0-4)
            reduction: Reduction method
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.reduction = reduction
        
        # Create weight matrix for quadratic kappa
        self.register_buffer('weight_matrix', self._get_quadratic_weights(num_classes))
        
        logger.info(f"QuadraticWeightedKappa initialized for {num_classes} classes")
    
    def _get_quadratic_weights(self, num_classes: int) -> torch.Tensor:
        """Generate quadratic weight matrix."""
        weights = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                weights[i, j] = (i - j) ** 2 / (num_classes - 1) ** 2
        return weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quadratic weighted kappa loss.
        
        Args:
            inputs: Predictions of shape (N, C)
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Kappa-based loss value
        """
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Create confusion matrix
        batch_size = inputs.shape[0]
        confusion = torch.zeros(self.num_classes, self.num_classes, device=inputs.device)
        
        for i in range(batch_size):
            pred_class = probs[i].argmax().item()
            true_class = targets[i].item()
            confusion[int(true_class), int(pred_class)] += 1
        
        # Normalize confusion matrix
        confusion = confusion / batch_size
        
        # Compute expected confusion matrix
        marginal_true = confusion.sum(dim=1)
        marginal_pred = confusion.sum(dim=0)
        expected = torch.outer(marginal_true, marginal_pred)
        
        # Compute quadratic weighted kappa
        numerator = (confusion * self.weight_matrix).sum()
        denominator = (expected * self.weight_matrix).sum()
        
        kappa = 1 - numerator / (denominator + 1e-7)
        
        # Return negative kappa as loss (we want to maximize kappa)
        return -kappa

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning: classification + segmentation.
    Handles dynamic loss weighting and scheduling.
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        segmentation_weight: float = 0.5,
        use_focal_cls: bool = True,
        use_kappa_cls: bool = False,
        use_combined_seg: bool = True,
        focal_gamma: float = 2.0,
        class_weights_cls: Optional[torch.Tensor] = None,
        class_weights_seg: Optional[torch.Tensor] = None,
        adaptive_weighting: bool = False
    ):
        """
        Initialize multi-task loss.
        
        Args:
            classification_weight: Weight for classification loss
            segmentation_weight: Weight for segmentation loss
            use_focal_cls: Use focal loss for classification
            use_kappa_cls: Use quadratic kappa loss for classification
            use_combined_seg: Use combined loss for segmentation
            focal_gamma: Gamma parameter for focal loss
            class_weights_cls: Class weights for classification
            class_weights_seg: Class weights for segmentation
            adaptive_weighting: Use adaptive loss weighting
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.segmentation_weight = segmentation_weight
        self.adaptive_weighting = adaptive_weighting
        
        # Classification losses
        if use_focal_cls:
            self.cls_loss = FocalLoss(alpha=class_weights_cls, gamma=focal_gamma)
        else:
            self.cls_loss = nn.CrossEntropyLoss(weight=class_weights_cls)
        
        # Additional kappa loss for ordinal classification
        if use_kappa_cls:
            self.kappa_loss = QuadraticWeightedKappa(num_classes=5)
        else:
            self.kappa_loss = None
        
        # Segmentation losses
        if use_combined_seg:
            self.seg_loss = CombinedSegmentationLoss(class_weights=class_weights_seg)
        else:
            self.seg_loss = DiceLoss()
        
        # Adaptive weighting parameters
        if adaptive_weighting:
            self.log_vars = nn.Parameter(torch.zeros(2))  # Learnable loss weights
        
        logger.info(f"MultiTaskLoss: cls_weight={classification_weight}, seg_weight={segmentation_weight}")
    
    def forward(
        self,
        cls_pred: torch.Tensor,
        seg_pred: torch.Tensor,
        cls_target: torch.Tensor,
        seg_target: torch.Tensor,
        epoch: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task loss.
        
        Args:
            cls_pred: Classification predictions (N, num_classes)
            seg_pred: Segmentation predictions (N, C, H, W)
            cls_target: Classification targets (N,)
            seg_target: Segmentation targets (N, H, W) or (N, C, H, W)
            epoch: Current epoch for scheduling (optional)
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Classification loss
        cls_loss = self.cls_loss(cls_pred, cls_target)
        losses['classification'] = cls_loss
        
        # Add kappa loss if enabled
        if self.kappa_loss is not None:
            kappa_loss = self.kappa_loss(cls_pred, cls_target)
            losses['kappa'] = kappa_loss
            cls_loss = cls_loss + 0.1 * kappa_loss  # Small weight for kappa
        
        # Segmentation loss
        if isinstance(self.seg_loss, CombinedSegmentationLoss):
            seg_losses = self.seg_loss(seg_pred, seg_target)
            seg_loss = seg_losses['total']
            losses.update({f'seg_{k}': v for k, v in seg_losses.items()})
        else:
            seg_loss = self.seg_loss(seg_pred, seg_target)
            losses['segmentation'] = seg_loss
        
        # Compute weights
        if self.adaptive_weighting:
            # Homoscedastic uncertainty weighting
            cls_weight = torch.exp(-self.log_vars[0])
            seg_weight = torch.exp(-self.log_vars[1])
            
            total_loss = (cls_weight * cls_loss + self.log_vars[0] +
                         seg_weight * seg_loss + self.log_vars[1])
            
            losses['cls_weight'] = cls_weight
            losses['seg_weight'] = seg_weight
        else:
            # Fixed weighting with optional scheduling
            cls_weight = self.classification_weight
            seg_weight = self.segmentation_weight
            
            # Progressive training: start with classification only
            if epoch is not None:
                # Ensure epoch is an integer
                try:
                    epoch_int = int(epoch)
                except (ValueError, TypeError):
                    epoch_int = 0
                    
                if epoch_int < 5:
                    seg_weight = 0.0
                elif epoch_int < 15:
                    seg_weight = self.segmentation_weight * (epoch_int - 4) / 10
            
            total_loss = cls_weight * cls_loss + seg_weight * seg_loss
        
        losses['total'] = total_loss
        losses['cls_weighted'] = cls_weight * cls_loss
        losses['seg_weighted'] = seg_weight * seg_loss
        
        return losses
    
    def update_weights(self, cls_weight: float, seg_weight: float):
        """Update loss weights during training."""
        self.classification_weight = cls_weight
        self.segmentation_weight = seg_weight

def test_losses():
    """Test all loss implementations."""
    logger.info("Testing loss functions...")
    
    batch_size = 4
    num_classes_cls = 5
    num_classes_seg = 1
    height, width = 64, 64
    
    # Create test data
    cls_pred = torch.randn(batch_size, num_classes_cls)
    cls_target = torch.randint(0, num_classes_cls, (batch_size,))
    
    seg_pred = torch.randn(batch_size, num_classes_seg, height, width)
    seg_target = torch.randint(0, 2, (batch_size, height, width)).float()
    
    # Test Focal Loss
    logger.info("Testing FocalLoss...")
    focal_loss = FocalLoss(gamma=2.0)
    focal_result = focal_loss(cls_pred, cls_target)
    logger.info(f"Focal loss: {focal_result.item():.4f}")
    
    # Test Dice Loss
    logger.info("Testing DiceLoss...")
    dice_loss = DiceLoss()
    dice_result = dice_loss(torch.sigmoid(seg_pred), seg_target.unsqueeze(1))
    logger.info(f"Dice loss: {dice_result.item():.4f}")
    
    # Test Combined Segmentation Loss
    logger.info("Testing CombinedSegmentationLoss...")
    combined_seg_loss = CombinedSegmentationLoss()
    combined_seg_result = combined_seg_loss(seg_pred, seg_target)
    logger.info(f"Combined seg losses: {combined_seg_result}")
    
    # Test Multi-Task Loss
    logger.info("Testing MultiTaskLoss...")
    multi_task_loss = MultiTaskLoss(
        classification_weight=1.0,
        segmentation_weight=0.5,
        adaptive_weighting=False
    )
    
    multi_result = multi_task_loss(cls_pred, seg_pred, cls_target, seg_target, epoch=10)
    logger.info(f"Multi-task losses: {multi_result}")
    
    # Test with adaptive weighting
    logger.info("Testing adaptive weighting...")
    adaptive_loss = MultiTaskLoss(adaptive_weighting=True)
    adaptive_result = adaptive_loss(cls_pred, seg_pred, cls_target, seg_target)
    logger.info(f"Adaptive losses: {adaptive_result}")
    
    logger.info("All loss tests completed successfully!")

if __name__ == "__main__":
    # Run tests
    test_losses()
