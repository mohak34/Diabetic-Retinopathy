
"""
Basic backbone implementation for diabetic retinopathy model
"""
import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

class BasicEfficientNetBackbone(nn.Module):
    """Basic EfficientNet backbone implementation"""
    
    def __init__(self, model_name="tf_efficientnet_b0_ns", pretrained=True, num_classes=0):
        super().__init__()
        self.model_name = model_name
        
        if TIMM_AVAILABLE:
            # Use timm if available
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=num_classes
            )
            self.features_dim = self.backbone.num_features
        else:
            # Fallback: Create a simple CNN backbone
            self.backbone = self._create_simple_cnn()
            self.features_dim = 1280
    
    def _create_simple_cnn(self):
        """Create a simple CNN backbone as fallback"""
        return nn.Sequential(
            # Input: 3 x 512 x 512
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 32 x 256 x 256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 1280, kernel_size=3, stride=2, padding=1),  # 1280 x 8 x 8
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((7, 7))  # 1280 x 7 x 7
        )
    
    def forward(self, x):
        """Forward pass through backbone"""
        if TIMM_AVAILABLE:
            return self.backbone.forward_features(x)
        else:
            return self.backbone(x)
    
    def get_features_dim(self):
        """Get features dimension"""
        return self.features_dim

def create_backbone(model_name="tf_efficientnet_b0_ns", pretrained=True):
    """Create backbone model"""
    return BasicEfficientNetBackbone(model_name=model_name, pretrained=pretrained)
