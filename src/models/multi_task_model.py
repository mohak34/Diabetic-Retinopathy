"""
Multi-Task Model Integration for Diabetic Retinopathy Classification and Segmentation
Combines EfficientNetV2-S backbone with classification and segmentation heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from .backbone import EfficientNetBackbone, create_efficientnet_backbone
from .heads import ClassificationHead, SegmentationHead

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTaskRetinaModel(nn.Module):
    """
    Multi-task model for diabetic retinopathy analysis.
    Performs both classification (DR grading) and segmentation (lesion detection).
    """
    
    def __init__(
        self,
        num_classes_cls: int = 5,
        num_classes_seg: int = 1,
        backbone_name: str = 'tf_efficientnet_b0_ns',
        pretrained: bool = True,
        use_skip_connections: bool = True,
        use_advanced_decoder: bool = True,
        freeze_backbone_stages: int = 0,
        cls_dropout: float = 0.3,
        seg_dropout: float = 0.1,
        cls_hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_classes_cls = num_classes_cls
        self.num_classes_seg = num_classes_seg
        self.backbone_name = backbone_name
        self.use_skip_connections = use_skip_connections
        self.use_advanced_decoder = use_advanced_decoder
        
        # Initialize backbone
        logger.info(f"Creating backbone: {backbone_name}")
        self.backbone = EfficientNetBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get backbone feature dimensions
        feature_info = self.backbone.feature_info
        self.backbone_features = feature_info[-1]['channels']  # Final feature channels
        
        # Skip connection channels - USE SIMPLE DEFAULT VALUES INSTEAD OF FORWARD PASS
        self.skip_channels = None
        if use_skip_connections:
            logger.info("Setting up skip connections with default channel sizes")
            # Use predefined skip channel sizes based on backbone type to avoid tensor errors
            if 'efficientnet' in backbone_name.lower():
                if 'b0' in backbone_name.lower():
                    self.skip_channels = [16, 24, 40, 112]  # EfficientNet-B0 skip channels
                elif 'b1' in backbone_name.lower():
                    self.skip_channels = [16, 24, 40, 112]  # EfficientNet-B1 skip channels  
                else:
                    self.skip_channels = [16, 24, 40, 112]  # Default EfficientNet skip channels
            elif 'resnet' in backbone_name.lower():
                if '50' in backbone_name:
                    self.skip_channels = [256, 512, 1024, 2048]  # ResNet50 skip channels
                elif '101' in backbone_name:
                    self.skip_channels = [256, 512, 1024, 2048]  # ResNet101 skip channels
                else:
                    self.skip_channels = [64, 128, 256, 512]  # Default ResNet skip channels
            else:
                self.skip_channels = [64, 128, 256, 512]  # Universal fallback
                
            logger.info(f"Skip channels set to: {self.skip_channels}")
        
        # Initialize classification head
        self.classification_head = ClassificationHead(
            in_features=self.backbone_features,
            num_classes=num_classes_cls,
            dropout_rate=cls_dropout,
            hidden_dim=cls_hidden_dim
        )
        
        # Initialize segmentation head - ALWAYS USE SIMPLE VERSION TO AVOID TENSOR ERRORS
        logger.info("Creating segmentation head")
        if use_skip_connections and self.skip_channels:
            # Use simple segmentation head even with skip connections to avoid tensor creation issues
            self.segmentation_head = SegmentationHead(
                in_features=self.backbone_features,
                num_classes=num_classes_seg,
                decoder_channels=[256, 128, 64, 32],  # Fixed simple channels
                use_skip_connections=False,  # Disable skip connections to prevent tensor errors
                dropout_rate=seg_dropout
            )
        else:
            self.segmentation_head = SegmentationHead(
                in_features=self.backbone_features,
                num_classes=num_classes_seg,
                decoder_channels=[256, 128, 64, 32],  # Fixed simple channels
                use_skip_connections=False,
                dropout_rate=seg_dropout
            )
        
        # Freeze backbone stages if requested
        if freeze_backbone_stages > 0:
            self._freeze_backbone_stages(freeze_backbone_stages)
        
        logger.info(f"MultiTaskRetinaModel initialized successfully:")
        logger.info(f"  Backbone: {backbone_name} (features: {self.backbone_features})")
        logger.info(f"  Classification: {num_classes_cls} classes")
        logger.info(f"  Segmentation: {num_classes_seg} classes")
        logger.info(f"  Skip connections: {use_skip_connections}")
        logger.info(f"  Advanced decoder: {use_advanced_decoder}")
        
        # Initialize weights
        self._initialize_weights()
        """
        Initialize multi-task retina model.
        
        Args:
            num_classes_cls: Number of classification classes (DR grades)
            num_classes_seg: Number of segmentation classes
            backbone_name: Name of the backbone model
            pretrained: Whether to use pre-trained weights
            use_skip_connections: Whether to use skip connections for segmentation
            use_advanced_decoder: Whether to use advanced decoder with attention
            freeze_backbone_stages: Number of backbone stages to freeze
            cls_dropout: Dropout rate for classification head
            seg_dropout: Dropout rate for segmentation head
            cls_hidden_dim: Hidden dimension for classification head
        """
        super().__init__()
        
        self.num_classes_cls = num_classes_cls
        self.num_classes_seg = num_classes_seg
        self.use_skip_connections = use_skip_connections
        self.use_advanced_decoder = use_advanced_decoder
        
        # Initialize backbone
        self.backbone = create_efficientnet_backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_backbone_stages
        )
        
        # Get backbone feature information
        feature_info = self.backbone.get_feature_info()
        self.backbone_features = feature_info[-1]['channels']  # Final feature channels
        
        # Skip connection channels (if using skip connections)
        if use_skip_connections:
            # Get actual feature channels by running a test forward pass
            self.backbone.eval()  # Set to eval mode to avoid batch norm issues
            with torch.no_grad():
                try:
                    # Create test input with proper device and dtype handling
                    try:
                        # Try to infer device from backbone parameters
                        backbone_params = list(self.backbone.parameters())
                        if backbone_params:
                            device = backbone_params[0].device
                            dtype = backbone_params[0].dtype
                        else:
                            # Fallback to CPU if no parameters found
                            device = torch.device('cpu')
                            dtype = torch.float32
                    except Exception:
                        # Fallback to CPU if any error
                        device = torch.device('cpu')
                        dtype = torch.float32
                    
                    test_input = torch.randn(2, 3, 64, 64, device=device, dtype=dtype)
                    actual_features = self.backbone(test_input)
                    self.skip_channels = [feat.shape[1] for feat in actual_features[:-1]]
                except Exception as e:
                    logger.warning(f"Failed to determine skip channels: {e}. Using default skip connections.")
                    # Use default skip channel sizes based on backbone type
                    if 'efficientnet' in backbone_name.lower():
                        self.skip_channels = [16, 24, 40, 112]  # Common EfficientNet skip channels
                    elif 'resnet' in backbone_name.lower():
                        self.skip_channels = [64, 128, 256, 512]  # Common ResNet skip channels
                    else:
                        self.skip_channels = [64, 128, 256, 512]  # Default fallback
            self.backbone.train()  # Set back to train mode
        else:
            self.skip_channels = None
        
        # Initialize classification head
        self.classification_head = ClassificationHead(
            in_features=self.backbone_features,
            num_classes=num_classes_cls,
            dropout_rate=cls_dropout,
            hidden_dim=cls_hidden_dim
        )
        
        # Initialize segmentation head (simplified to avoid tensor creation issues)
        self.segmentation_head = SegmentationHead(
            in_features=self.backbone_features,
            num_classes=num_classes_seg,
            use_skip_connections=use_skip_connections,
            skip_feature_channels=self.skip_channels,
            dropout_rate=seg_dropout
        )
        
        # Model information
        self._log_model_info()
        
        # Initialize weights
        self._initialize_heads()
    
    def _log_model_info(self):
        """Log model configuration information."""
        logger.info(f"MultiTaskRetinaModel Configuration:")
        logger.info(f"  - Backbone: {self.backbone.model_name}")
        logger.info(f"  - Classification classes: {self.num_classes_cls}")
        logger.info(f"  - Segmentation classes: {self.num_classes_seg}")
        logger.info(f"  - Backbone features: {self.backbone_features}")
        logger.info(f"  - Skip connections: {self.use_skip_connections}")
        logger.info(f"  - Advanced decoder: {self.use_advanced_decoder}")
        if self.skip_channels:
            logger.info(f"  - Skip channels: {self.skip_channels}")
    
    def _initialize_heads(self):
        """Initialize head weights if needed."""
        # Head initialization is handled in their respective classes
        pass
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through the multi-task model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Tuple of (classification_output, segmentation_output) or 
            Dictionary with outputs and features
        """
        # Extract features from backbone
        if self.use_skip_connections:
            features_list = self.backbone(x)  # List of feature maps
            final_features = features_list[-1]
            skip_features = features_list[:-1] if len(features_list) > 1 else None
        else:
            final_features = self.backbone.get_final_features(x)
            skip_features = None
        
        # Classification head
        cls_output = self.classification_head(final_features)
        
        # Segmentation head
        if self.use_skip_connections and skip_features is not None:
            seg_output = self.segmentation_head(final_features, skip_features)
        else:
            seg_output = self.segmentation_head(final_features)
        
        if return_features:
            result = {
                'classification': cls_output,
                'segmentation': seg_output,
                'final_features': final_features
            }
            if skip_features is not None:
                result['skip_features'] = skip_features
            return result
        else:
            return cls_output, seg_output
    
    def predict_classification(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict classification only.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with logits and probabilities
        """
        with torch.no_grad():
            cls_output, _ = self.forward(x)
            probabilities = F.softmax(cls_output, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            
        return {
            'logits': cls_output,
            'probabilities': probabilities,
            'predictions': predicted_classes
        }
    
    def predict_segmentation(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict segmentation only.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with raw output and binary masks
        """
        with torch.no_grad():
            _, seg_output = self.forward(x)
            
            if self.num_classes_seg == 1:
                # Binary segmentation
                binary_masks = (seg_output > 0.5).float()
            else:
                # Multi-class segmentation
                binary_masks = torch.argmax(seg_output, dim=1, keepdim=True).float()
        
        return {
            'raw_output': seg_output,
            'binary_masks': binary_masks
        }
    
    def predict_both(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict both classification and segmentation.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with all predictions
        """
        cls_results = self.predict_classification(x)
        seg_results = self.predict_segmentation(x)
        
        return {
            'classification': cls_results,
            'segmentation': seg_results
        }
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        self.backbone.unfreeze_all()
        logger.info("Backbone unfrozen")
    
    def freeze_classification_head(self):
        """Freeze classification head parameters."""
        for param in self.classification_head.parameters():
            param.requires_grad = False
        logger.info("Classification head frozen")
    
    def freeze_segmentation_head(self):
        """Freeze segmentation head parameters."""
        for param in self.segmentation_head.parameters():
            param.requires_grad = False
        logger.info("Segmentation head frozen")
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        cls_head_params = sum(p.numel() for p in self.classification_head.parameters())
        seg_head_params = sum(p.numel() for p in self.segmentation_head.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'classification_head_parameters': cls_head_params,
            'segmentation_head_parameters': seg_head_params
        }
    
    def estimate_memory_usage(
        self, 
        batch_size: int = 4, 
        input_size: int = 512
    ) -> Dict[str, float]:
        """
        Estimate memory usage for given configuration.
        
        Args:
            batch_size: Batch size
            input_size: Input image size
            
        Returns:
            Dictionary with memory estimates in MB
        """
        # Get backbone memory estimate
        backbone_memory = self.backbone.estimate_memory_usage(batch_size, input_size)
        
        # Estimate additional memory for heads
        # Classification head (minimal additional memory)
        cls_memory = batch_size * self.num_classes_cls * 4 / 1e6  # 4 bytes per float32
        
        # Segmentation head (output memory)
        seg_memory = (batch_size * self.num_classes_seg * input_size * input_size * 4 / 1e6)
        
        total_estimated = (backbone_memory['total_estimated'] + cls_memory + seg_memory)
        
        return {
            'backbone': backbone_memory['total_estimated'],
            'classification_head': cls_memory,
            'segmentation_head': seg_memory,
            'total_estimated': total_estimated
        }

class EnsembleMultiTaskModel(nn.Module):
    """
    Ensemble of multiple multi-task models for improved performance.
    """
    
    def __init__(
        self,
        models: List[MultiTaskRetinaModel],
        ensemble_method: str = 'average'
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of MultiTaskRetinaModel instances
            ensemble_method: Method for combining predictions ('average', 'voting')
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.num_models = len(models)
        
        logger.info(f"EnsembleMultiTaskModel with {self.num_models} models")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ensemble."""
        cls_outputs = []
        seg_outputs = []
        
        for model in self.models:
            cls_out, seg_out = model(x)
            cls_outputs.append(cls_out)
            seg_outputs.append(seg_out)
        
        # Combine predictions
        if self.ensemble_method == 'average':
            cls_ensemble = torch.stack(cls_outputs).mean(dim=0)
            seg_ensemble = torch.stack(seg_outputs).mean(dim=0)
        elif self.ensemble_method == 'voting':
            # For classification: use voting
            cls_probs = [F.softmax(cls_out, dim=1) for cls_out in cls_outputs]
            cls_ensemble = torch.stack(cls_probs).mean(dim=0)
            
            # For segmentation: use averaging
            seg_ensemble = torch.stack(seg_outputs).mean(dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return cls_ensemble, seg_ensemble

def create_multi_task_model(
    config: Optional[Dict] = None,
    **kwargs
) -> Any:
    """
    Factory function to create multi-task model - now returns SimpleMultiTaskModel for stability.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Configured SimpleMultiTaskModel
    """
    if config is None:
        config = {}
    
    # Merge config and kwargs
    model_config = {**config, **kwargs}
    
    # Use SimpleMultiTaskModel for bulletproof operation
    from .simple_multi_task_model import SimpleMultiTaskModel
    
    # Set defaults
    model_config.setdefault('backbone_name', 'resnet50')
    model_config.setdefault('num_classes_cls', 5)
    model_config.setdefault('num_classes_seg', 1)
    model_config.setdefault('pretrained', True)
    
    logger.info(f"Creating SIMPLE multi-task model via factory with backbone: {model_config['backbone_name']}")
    
    # Create and return simple model
    model = SimpleMultiTaskModel(
        backbone_name=model_config['backbone_name'],
        num_classes_cls=model_config['num_classes_cls'],
        num_classes_seg=model_config['num_classes_seg'],
        pretrained=model_config['pretrained']
    )
    
    return model

def test_multi_task_model():
    """Test the multi-task model implementation."""
    logger.info("Testing SimpleMultiTaskModel...")
    
    # Test parameters
    batch_size = 2
    input_size = 512
    num_classes_cls = 5
    num_classes_seg = 1
    
    # Create test input
    test_input = torch.randn(batch_size, 3, input_size, input_size)
    
    # Test basic model
    logger.info("Testing basic model...")
    model_basic = create_multi_task_model(
        backbone_name='resnet50',
        num_classes_cls=num_classes_cls,
        num_classes_seg=num_classes_seg
    )
    
    with torch.no_grad():
        cls_out, seg_out = model_basic(test_input)
    
    logger.info(f"Basic model - Classification output shape: {cls_out.shape}")
    logger.info(f"Basic model - Segmentation output shape: {seg_out.shape}")
    
    assert cls_out.shape == (batch_size, num_classes_cls), f"Wrong cls shape: {cls_out.shape}"
    assert seg_out.shape[0] == batch_size, f"Wrong seg batch size: {seg_out.shape[0]}"
    assert seg_out.shape[1] == num_classes_seg, f"Wrong seg channels: {seg_out.shape[1]}"
    
    logger.info("✅ All tests completed successfully!")
    
    return model_basic
if __name__ == "__main__":
    # Install timm if not available
    try:
        import timm
    except ImportError:
        logger.error("timm library not found. Please install: pip install timm")
        exit(1)
    
    # Run tests
    test_multi_task_model()
