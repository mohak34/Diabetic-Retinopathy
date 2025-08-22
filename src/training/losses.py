"""
Phase 4: Robust Loss Functions
Error-free loss implementations with proper type handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class RobustFocalLoss(nn.Module):
    """
    Robust Focal Loss with proper type handling and error checking.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)  # Ensure gamma is float
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with robust type handling"""
        # Ensure inputs and targets are on the same device
        if inputs.device != targets.device:
            targets = targets.to(inputs.device)
        
        # Ensure targets are long type for classification
        targets = targets.long()
        
        # Compute cross entropy loss
        ce_loss = self.ce_loss(inputs, targets)
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply focal loss formula
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (int, float)):
                alpha_t = float(self.alpha)
                focal_loss = alpha_t * focal_loss
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets)
                focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class RobustDiceLoss(nn.Module):
    """
    Robust Dice Loss with proper type handling.
    """
    
    def __init__(self, smooth: float = 1e-5, reduction: str = 'mean'):
        super().__init__()
        self.smooth = float(smooth)
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with robust type handling"""
        # Ensure same device
        if inputs.device != targets.device:
            targets = targets.to(inputs.device)
        
        # Handle binary segmentation case
        if inputs.shape[1] == 1:
            # Apply sigmoid for binary case
            inputs = torch.sigmoid(inputs)
            
            # Ensure targets are float and have correct shape
            if len(targets.shape) == 3:  # (N, H, W)
                targets = targets.float().unsqueeze(1)  # (N, 1, H, W)
            else:
                targets = targets.float()
        else:
            # Apply softmax for multi-class
            inputs = F.softmax(inputs, dim=1)
            
            # Handle target shape
            if len(targets.shape) == 3:  # (N, H, W) - class indices
                # Convert to one-hot
                targets_onehot = F.one_hot(targets.long(), num_classes=inputs.shape[1])
                targets = targets_onehot.permute(0, 3, 1, 2).float()
            else:
                targets = targets.float()
        
        # Handle size mismatch by resizing targets to match inputs
        if inputs.shape[-2:] != targets.shape[-2:]:
            targets = F.interpolate(
                targets, 
                size=inputs.shape[-2:], 
                mode='nearest'
            )
        
        # Flatten spatial dimensions
        inputs_flat = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets_flat = targets.view(targets.shape[0], targets.shape[1], -1)
        
        # Compute dice for each class
        intersection = (inputs_flat * targets_flat).sum(dim=2)
        inputs_sum = inputs_flat.sum(dim=2)
        targets_sum = targets_flat.sum(dim=2)
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (inputs_sum + targets_sum + self.smooth)
        
        # Dice loss (1 - dice)
        dice_loss = 1.0 - dice
        
        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class RobustMultiTaskLoss(nn.Module):
    """
    Robust Multi-Task Loss with proper epoch type handling and progressive training.
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        segmentation_weight: float = 0.5,
        use_focal_cls: bool = True,
        use_kappa_cls: bool = False,
        focal_gamma: float = 2.0,
        kappa_weight: float = 0.1
    ):
        super().__init__()
        
        self.classification_weight = float(classification_weight)
        self.segmentation_weight = float(segmentation_weight)
        self.kappa_weight = float(kappa_weight)
        
        # Classification loss
        if use_focal_cls:
            self.cls_loss = RobustFocalLoss(gamma=focal_gamma)
        else:
            self.cls_loss = nn.CrossEntropyLoss()
        
        # Segmentation loss
        self.seg_loss = RobustDiceLoss()
        
        # Optional kappa loss for ordinal classification
        self.use_kappa = use_kappa_cls
        if use_kappa_cls:
            self.kappa_loss = nn.CrossEntropyLoss()  # Simplified for stability
        
        logger.info(f"MultiTaskLoss initialized: cls_weight={classification_weight}, seg_weight={segmentation_weight}")
    
    def get_progressive_weights(self, epoch: Union[int, str, float, None]) -> Tuple[float, float]:
        """
        Get progressive training weights based on epoch with robust type handling.
        """
        # Handle None epoch
        if epoch is None:
            return self.classification_weight, self.segmentation_weight
        
        # Convert epoch to integer safely
        try:
            if isinstance(epoch, str):
                epoch_int = int(float(epoch))  # Handle string numbers
            elif isinstance(epoch, float):
                epoch_int = int(epoch)
            else:
                epoch_int = int(epoch)
        except (ValueError, TypeError):
            logger.warning(f"Invalid epoch type/value: {epoch} ({type(epoch)}), using 0")
            epoch_int = 0
        
        # Ensure non-negative
        epoch_int = max(0, epoch_int)
        
        # Progressive training strategy
        cls_weight = self.classification_weight
        
        if epoch_int < 5:
            # Phase 1: Classification only
            seg_weight = 0.0
        elif epoch_int < 15:
            # Phase 2: Gradual segmentation introduction
            progress = float(epoch_int - 4) / 10.0  # 0.0 to 1.0
            seg_weight = self.segmentation_weight * progress
        else:
            # Phase 3: Full multi-task training
            seg_weight = self.segmentation_weight
        
        return cls_weight, seg_weight
    
    def forward(
        self,
        cls_pred: torch.Tensor,
        seg_pred: torch.Tensor,
        cls_target: torch.Tensor,
        seg_target: torch.Tensor,
        epoch: Optional[Union[int, str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with robust type handling and progressive training.
        """
        losses = {}
        
        # Ensure all tensors are on the same device
        device = cls_pred.device
        if cls_target.device != device:
            cls_target = cls_target.to(device)
        if seg_pred.device != device:
            seg_pred = seg_pred.to(device)
        if seg_target.device != device:
            seg_target = seg_target.to(device)
        
        # Handle size mismatch between prediction and target
        if seg_pred.shape[-2:] != seg_target.shape[-2:]:
            seg_target = F.interpolate(
                seg_target.float(), 
                size=seg_pred.shape[-2:], 
                mode='nearest'
            )
        
        # Classification loss
        cls_loss = self.cls_loss(cls_pred, cls_target)
        losses['classification'] = cls_loss
        
        # Add kappa loss if enabled
        if self.use_kappa:
            kappa_loss = self.kappa_loss(cls_pred, cls_target)
            losses['kappa'] = kappa_loss
            cls_loss = cls_loss + self.kappa_weight * kappa_loss
        
        # Segmentation loss
        seg_loss = self.seg_loss(seg_pred, seg_target)
        losses['segmentation'] = seg_loss
        
        # Get progressive weights
        cls_weight, seg_weight = self.get_progressive_weights(epoch)
        
        # Store weights for logging
        losses['cls_weight'] = torch.tensor(cls_weight, device=device)
        losses['seg_weight'] = torch.tensor(seg_weight, device=device)
        
        # Compute weighted losses
        cls_weighted = cls_weight * cls_loss
        seg_weighted = seg_weight * seg_loss
        
        losses['cls_weighted'] = cls_weighted
        losses['seg_weighted'] = seg_weighted
        losses['total'] = cls_weighted + seg_weighted
        
        return losses
    
    def update_weights(self, cls_weight: float, seg_weight: float):
        """Update loss weights during training"""
        self.classification_weight = float(cls_weight)
        self.segmentation_weight = float(seg_weight)


def test_robust_losses():
    """Test all robust loss implementations"""
    logger.info("Testing robust loss functions...")
    
    # Test parameters
    batch_size = 4
    num_classes_cls = 5
    height, width = 64, 64
    
    # Create test data
    cls_pred = torch.randn(batch_size, num_classes_cls)
    cls_target = torch.randint(0, num_classes_cls, (batch_size,))
    
    seg_pred = torch.randn(batch_size, 1, height, width)  # Binary segmentation
    seg_target = torch.randint(0, 2, (batch_size, height, width)).float()
    
    # Test Focal Loss
    logger.info("Testing RobustFocalLoss...")
    focal_loss = RobustFocalLoss(gamma=2.0)
    focal_result = focal_loss(cls_pred, cls_target)
    logger.info(f"Focal loss: {focal_result.item():.4f}")
    
    # Test Dice Loss
    logger.info("Testing RobustDiceLoss...")
    dice_loss = RobustDiceLoss()
    dice_result = dice_loss(seg_pred, seg_target)
    logger.info(f"Dice loss: {dice_result.item():.4f}")
    
    # Test Multi-Task Loss with various epoch types
    logger.info("Testing RobustMultiTaskLoss...")
    multi_task_loss = RobustMultiTaskLoss()
    
    # Test with different epoch types
    test_epochs = [0, 5, 10, 15, "5", "10.5", None, -1]
    
    for epoch in test_epochs:
        result = multi_task_loss(cls_pred, seg_pred, cls_target, seg_target, epoch=epoch)
        logger.info(f"Epoch {epoch} ({type(epoch)}): total_loss={result['total'].item():.4f}, "
                   f"cls_weight={result['cls_weight'].item():.3f}, "
                   f"seg_weight={result['seg_weight'].item():.3f}")
    
    logger.info("✅ All robust loss tests completed successfully!")


if __name__ == "__main__":
    test_robust_losses()
