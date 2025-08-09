
"""
Basic multi-task model for diabetic retinopathy detection
"""
import torch
import torch.nn as nn

class BasicMultiTaskModel(nn.Module):
    """Basic multi-task model combining classification and segmentation"""
    
    def __init__(self, backbone, classification_head, segmentation_head):
        super().__init__()
        self.backbone = backbone
        self.classification_head = classification_head
        self.segmentation_head = segmentation_head
    
    def forward(self, x):
        """Forward pass through multi-task model"""
        # Extract features using backbone
        features = self.backbone(x)
        
        # Get predictions from both heads
        classification_output = self.classification_head(features)
        segmentation_output = self.segmentation_head(features)
        
        return {
            'classification': classification_output,
            'segmentation': segmentation_output,
            'features': features
        }
    
    def get_total_parameters(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_parameter_breakdown(self):
        """Get parameter breakdown by component"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        cls_params = sum(p.numel() for p in self.classification_head.parameters())
        seg_params = sum(p.numel() for p in self.segmentation_head.parameters())
        
        return {
            'backbone': backbone_params,
            'classification_head': cls_params,
            'segmentation_head': seg_params,
            'total': backbone_params + cls_params + seg_params
        }

def create_basic_multi_task_model(backbone_name="tf_efficientnet_b0_ns", 
                                 num_classes=5, 
                                 num_segmentation_classes=4, 
                                 pretrained=True):
    """Create basic multi-task model"""
    # Import components (absolute imports to work without package context)
    from src.models.basic_backbone import create_backbone
    from src.models.basic_heads import create_classification_head, create_segmentation_head
    
    # Create components
    backbone = create_backbone(model_name=backbone_name, pretrained=pretrained)
    features_dim = backbone.get_features_dim()
    
    classification_head = create_classification_head(
        features_dim=features_dim, 
        num_classes=num_classes
    )
    
    segmentation_head = create_segmentation_head(
        features_dim=features_dim, 
        num_classes=num_segmentation_classes
    )
    
    # Create multi-task model
    model = BasicMultiTaskModel(
        backbone=backbone,
        classification_head=classification_head,
        segmentation_head=segmentation_head
    )
    
    return model
