
"""
Basic loss functions for multi-task diabetic retinopathy model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicDiceLoss(nn.Module):
    """Basic Dice Loss for segmentation tasks"""
    
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Calculate Dice Loss
        
        Args:
            predictions: (B, C, H, W) - predicted segmentation masks
            targets: (B, C, H, W) or (B, H, W) - ground truth masks
        """
        # Handle different target formats
        if targets.dim() == 3:
            # Convert to one-hot if targets are class indices
            num_classes = predictions.size(1)
            targets = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten tensors
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)
        
        # Apply sigmoid to predictions
        predictions = torch.sigmoid(predictions)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum(dim=2)
        union = predictions.sum(dim=2) + targets.sum(dim=2)
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - average dice as loss
        return 1.0 - dice.mean()

class BasicFocalLoss(nn.Module):
    """Basic Focal Loss for classification with class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Calculate Focal Loss
        
        Args:
            predictions: (B, C) - predicted class logits
            targets: (B,) - ground truth class indices
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BasicCombinedLoss(nn.Module):
    """Basic combined loss for multi-task learning"""
    
    def __init__(self, classification_weight=1.0, segmentation_weight=1.0):
        super().__init__()
        self.classification_weight = classification_weight
        self.segmentation_weight = segmentation_weight
        
        # Loss functions
        self.focal_loss = BasicFocalLoss(alpha=1.0, gamma=2.0)
        self.dice_loss = BasicDiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        """
        Calculate combined multi-task loss
        
        Args:
            predictions: dict with 'classification' and 'segmentation' outputs
            targets: dict with 'classification' and 'segmentation' targets
        """
        total_loss = 0.0
        loss_components = {}
        
        # Classification loss
        if 'classification' in predictions and 'classification' in targets:
            cls_pred = predictions['classification']
            cls_target = targets['classification']
            
            # Use focal loss for classification
            cls_loss = self.focal_loss(cls_pred, cls_target)
            total_loss += self.classification_weight * cls_loss
            loss_components['classification_loss'] = cls_loss.item()
        
        # Segmentation loss
        if 'segmentation' in predictions and 'segmentation' in targets:
            seg_pred = predictions['segmentation']
            seg_target = targets['segmentation']
            
            # Use dice loss for segmentation
            seg_loss = self.dice_loss(seg_pred, seg_target)
            total_loss += self.segmentation_weight * seg_loss
            loss_components['segmentation_loss'] = seg_loss.item()
        
        loss_components['total_loss'] = total_loss.item()
        return total_loss, loss_components
    
    def update_weights(self, classification_weight, segmentation_weight):
        """Update loss weights for progressive training"""
        self.classification_weight = classification_weight
        self.segmentation_weight = segmentation_weight

def create_loss_function(loss_type='combined', **kwargs):
    """Create loss function"""
    if loss_type == 'dice':
        return BasicDiceLoss(**kwargs)
    elif loss_type == 'focal':
        return BasicFocalLoss(**kwargs)
    elif loss_type == 'combined':
        return BasicCombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
