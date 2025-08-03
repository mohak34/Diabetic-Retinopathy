"""
EfficientNetV2-S Backbone Implementation for Multi-Task Diabetic Retinopathy Project
Uses pre-trained EfficientNetV2-S as feature extractor for both classification and segmentation tasks.
"""

import torch
import torch.nn as nn
import timm
import logging
from typing import Dict, List, Optional, Tuple
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientNetBackbone(nn.Module):
    """
    EfficientNetV2-S backbone for feature extraction.
    Extracts features from multiple stages for potential skip connections.
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnetv2_s',
        pretrained: bool = True,
        features_only: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0
    ):
        """
        Initialize EfficientNetV2-S backbone.
        
        Args:
            model_name: Name of the EfficientNet model
            pretrained: Whether to load pre-trained ImageNet weights
            features_only: Whether to return features only (no classification head)
            drop_rate: Dropout rate for the model
            drop_path_rate: Drop path rate for stochastic depth
        """
        super().__init__()
        
        self.model_name = model_name
        
        try:
            # Create EfficientNetV2-S with pre-trained weights
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=features_only,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                out_indices=(1, 2, 3, 4)  # Extract features from multiple stages
            )
            
            # Get feature information
            self.feature_info = self.backbone.feature_info
            self.num_features = self.feature_info[-1]['num_chs']  # Final feature channels
            
            logger.info(f"Successfully loaded {model_name} backbone")
            logger.info(f"Feature dimensions: {[info['num_chs'] for info in self.feature_info]}")
            logger.info(f"Feature reduction ratios: {[info['reduction'] for info in self.feature_info]}")
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
        
        # Freeze early layers option
        self._frozen = False
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of feature tensors from different stages
        """
        features = self.backbone(x)
        return features
    
    def get_final_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get only the final (highest-level) features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Final feature tensor
        """
        features = self.forward(x)
        return features[-1]  # Return the final feature map
    
    def freeze_early_layers(self, freeze_stages: int = 2):
        """
        Freeze early layers for fine-tuning.
        
        Args:
            freeze_stages: Number of early stages to freeze
        """
        if freeze_stages <= 0:
            return
            
        # Get the backbone's named parameters
        stage_params = []
        current_stage = 0
        
        for name, param in self.backbone.named_parameters():
            # EfficientNet stage naming convention
            if 'blocks.' in name:
                # Extract stage number from parameter name
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'blocks' and i + 1 < len(parts):
                        block_idx = int(parts[i + 1])
                        # Map block indices to stages (approximate)
                        if block_idx < 2:
                            stage = 0
                        elif block_idx < 4:
                            stage = 1
                        elif block_idx < 6:
                            stage = 2
                        else:
                            stage = 3
                        
                        if stage < freeze_stages:
                            param.requires_grad = False
                            stage_params.append(name)
                        break
        
        self._frozen = True
        logger.info(f"Frozen {freeze_stages} early stages ({len(stage_params)} parameters)")
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._frozen = False
        logger.info("Unfrozen all backbone parameters")
    
    def get_feature_info(self) -> List[Dict]:
        """Get information about feature maps at each stage."""
        return [
            {
                'stage': i,
                'channels': info['num_chs'],
                'reduction': info['reduction'],
                'size_at_512': 512 // info['reduction']
            }
            for i, info in enumerate(self.feature_info)
        ]
    
    def estimate_memory_usage(self, batch_size: int = 4, input_size: int = 512) -> Dict[str, float]:
        """
        Estimate memory usage for given input configuration.
        
        Args:
            batch_size: Batch size
            input_size: Input image size (assumes square)
            
        Returns:
            Dictionary with memory estimates in MB
        """
        # Calculate approximate memory for each feature stage
        memory_estimates = {}
        
        # Input tensor
        input_memory = batch_size * 3 * input_size * input_size * 4 / 1e6  # 4 bytes per float32
        memory_estimates['input'] = input_memory
        
        # Feature map memory
        total_feature_memory = 0
        for i, info in enumerate(self.feature_info):
            feature_size = input_size // info['reduction']
            feature_memory = batch_size * info['num_chs'] * feature_size * feature_size * 4 / 1e6
            memory_estimates[f'stage_{i}'] = feature_memory
            total_feature_memory += feature_memory
        
        memory_estimates['total_features'] = total_feature_memory
        memory_estimates['total_estimated'] = input_memory + total_feature_memory
        
        return memory_estimates

class BackboneWithAdaptivePool(nn.Module):
    """
    Backbone with adaptive pooling for flexible output sizes.
    Useful when input images have different aspect ratios.
    """
    
    def __init__(
        self,
        backbone: EfficientNetBackbone,
        output_size: Tuple[int, int] = (7, 7),
        pool_type: str = 'adaptive_avg'
    ):
        """
        Initialize backbone with adaptive pooling.
        
        Args:
            backbone: EfficientNetBackbone instance
            output_size: Target output size for pooling
            pool_type: Type of pooling ('adaptive_avg', 'adaptive_max')
        """
        super().__init__()
        
        self.backbone = backbone
        self.output_size = output_size
        
        if pool_type == 'adaptive_avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'adaptive_max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive pooling."""
        features = self.backbone.get_final_features(x)
        pooled = self.pool(features)
        return pooled

def create_efficientnet_backbone(
    model_name: str = 'efficientnetv2_s',
    pretrained: bool = True,
    freeze_stages: int = 0
) -> EfficientNetBackbone:
    """
    Factory function to create EfficientNet backbone.
    
    Args:
        model_name: Name of the EfficientNet model
        pretrained: Whether to load pre-trained weights
        freeze_stages: Number of early stages to freeze
        
    Returns:
        Configured EfficientNetBackbone instance
    """
    backbone = EfficientNetBackbone(
        model_name=model_name,
        pretrained=pretrained
    )
    
    if freeze_stages > 0:
        backbone.freeze_early_layers(freeze_stages)
    
    return backbone

def test_backbone_functionality():
    """Test the backbone implementation."""
    logger.info("Testing EfficientNetV2-S backbone...")
    
    # Create backbone
    backbone = create_efficientnet_backbone()
    
    # Test input
    batch_size = 2
    input_size = 512
    test_input = torch.randn(batch_size, 3, input_size, input_size)
    
    # Test forward pass
    with torch.no_grad():
        features = backbone(test_input)
        final_features = backbone.get_final_features(test_input)
    
    # Print results
    logger.info(f"Input shape: {test_input.shape}")
    logger.info(f"Number of feature stages: {len(features)}")
    
    for i, feature in enumerate(features):
        logger.info(f"Stage {i} features shape: {feature.shape}")
    
    logger.info(f"Final features shape: {final_features.shape}")
    
    # Test memory estimation
    memory_est = backbone.estimate_memory_usage(batch_size=4, input_size=512)
    logger.info(f"Estimated memory usage: {memory_est}")
    
    # Test feature info
    feature_info = backbone.get_feature_info()
    logger.info(f"Feature info: {feature_info}")
    
    logger.info("Backbone test completed successfully!")
    
    return backbone

if __name__ == "__main__":
    # Run tests
    test_backbone_functionality()
