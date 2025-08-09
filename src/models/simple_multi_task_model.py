"""
Simple Multi-Task Model with NO tensor creation issues.
This is a completely safe implementation that avoids all possible tensor creation errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union

from .backbone import EfficientNetBackbone
from .heads import ClassificationHead, SegmentationHead

# Setup logging
logger = logging.getLogger(__name__)

class SimpleMultiTaskModel(nn.Module):
    """
    Simple multi-task model that completely avoids tensor creation issues.
    This model is designed to be bulletproof for hyperparameter optimization.
    """
    
    def __init__(
        self,
        num_classes_cls: int = 5,
        num_classes_seg: int = 1,
        backbone_name: str = 'tf_efficientnet_b0_ns',
        pretrained: bool = True,
        cls_dropout: float = 0.3,
        seg_dropout: float = 0.1,
        cls_hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_classes_cls = num_classes_cls
        self.num_classes_seg = num_classes_seg
        self.backbone_name = backbone_name
        
        logger.info(f"Creating SIMPLE multi-task model with backbone: {backbone_name}")
        
        # Initialize backbone with error handling
        try:
            self.backbone = EfficientNetBackbone(
                model_name=backbone_name,
                pretrained=pretrained,
                features_only=True
            )
            backbone_features = self.backbone.feature_info[-1]['num_chs']
            logger.info(f"Backbone created successfully with {backbone_features} features")
        except Exception as e:
            logger.error(f"Failed to create backbone {backbone_name}: {e}")
            # Use ResNet50 as ultimate fallback
            try:
                logger.warning("Falling back to ResNet50")
                self.backbone = EfficientNetBackbone(
                    model_name='resnet50',
                    pretrained=True,
                    features_only=True
                )
                backbone_features = self.backbone.feature_info[-1]['num_chs']
                self.backbone_name = 'resnet50'
                logger.info(f"Fallback backbone created with {backbone_features} features")
            except Exception as fallback_error:
                logger.error(f"Even fallback failed: {fallback_error}")
                raise
        
        # Initialize classification head
        self.classification_head = ClassificationHead(
            in_features=backbone_features,
            num_classes=num_classes_cls,
            dropout_rate=cls_dropout,
            hidden_dim=cls_hidden_dim
        )
        
        # Initialize SIMPLE segmentation head (no skip connections, no advanced features)
        self.segmentation_head = SegmentationHead(
            in_features=backbone_features,
            num_classes=num_classes_seg,
            decoder_channels=[256, 128, 64, 32],  # Fixed simple channels
            use_skip_connections=False,  # NO skip connections
            dropout_rate=seg_dropout
        )
        
        logger.info(f"Simple multi-task model created successfully:")
        logger.info(f"  Backbone: {self.backbone_name}")
        logger.info(f"  Classification: {num_classes_cls} classes")
        logger.info(f"  Segmentation: {num_classes_seg} classes")
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_features: Whether to return features dictionary
            
        Returns:
            If return_features=False: Tuple of (classification_logits, segmentation_masks)
            If return_features=True: Dictionary with detailed outputs
        """
        # Extract features from backbone
        features = self.backbone(x)
        final_features = features[-1]  # Use only the final features
        
        # Classification branch
        cls_logits = self.classification_head(final_features)
        
        # Segmentation branch
        seg_masks = self.segmentation_head(final_features)
        
        if return_features:
            return {
                'classification': cls_logits,
                'segmentation': seg_masks,
                'features': final_features,
                'backbone_features': features
            }
        else:
            return cls_logits, seg_masks
    
    def predict_classification(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict classification only."""
        features = self.backbone(x)
        final_features = features[-1]
        cls_logits = self.classification_head(final_features)
        
        return {
            'logits': cls_logits,
            'probabilities': torch.softmax(cls_logits, dim=1),
            'predictions': torch.argmax(cls_logits, dim=1)
        }
    
    def predict_segmentation(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict segmentation only."""
        features = self.backbone(x)
        final_features = features[-1]
        seg_masks = self.segmentation_head(final_features)
        
        return {
            'masks': seg_masks,
            'probabilities': torch.sigmoid(seg_masks),
            'predictions': (torch.sigmoid(seg_masks) > 0.5).float()
        }
    
    def predict_both(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict both classification and segmentation."""
        features = self.backbone(x)
        final_features = features[-1]
        
        cls_logits = self.classification_head(final_features)
        seg_masks = self.segmentation_head(final_features)
        
        return {
            'classification': {
                'logits': cls_logits,
                'probabilities': torch.softmax(cls_logits, dim=1),
                'predictions': torch.argmax(cls_logits, dim=1)
            },
            'segmentation': {
                'masks': seg_masks,
                'probabilities': torch.sigmoid(seg_masks),
                'predictions': (torch.sigmoid(seg_masks) > 0.5).float()
            }
        }


def create_simple_multi_task_model(config_dict: Dict) -> SimpleMultiTaskModel:
    """
    Factory function to create a simple multi-task model.
    This is completely safe and won't cause tensor creation errors.
    """
    logger.info("Creating simple multi-task model (safe version)")
    
    model_config = {
        'num_classes_cls': config_dict.get('num_classes_cls', 5),
        'num_classes_seg': config_dict.get('num_classes_seg', 1),
        'backbone_name': config_dict.get('backbone_name', 'tf_efficientnet_b0_ns'),
        'pretrained': config_dict.get('pretrained', True),
        'cls_dropout': config_dict.get('cls_dropout', 0.3),
        'seg_dropout': config_dict.get('seg_dropout', 0.1),
        'cls_hidden_dim': config_dict.get('cls_hidden_dim', 256)
    }
    
    return SimpleMultiTaskModel(**model_config)
