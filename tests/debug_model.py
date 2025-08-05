#!/usr/bin/env python3
"""
Debug script to identify the exact channel mismatch issue
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.multi_task_model import create_multi_task_model

def debug_model_channels():
    """Debug the channel flow in the model"""
    print("=== DEBUGGING MODEL CHANNELS ===")
    
    # Create model with same config as training
    model = create_multi_task_model(
        backbone_name="tf_efficientnetv2_b0",
        num_classes_cls=5,
        num_classes_seg=1,
        pretrained=False,  # Skip pretrained to avoid download
        use_skip_connections=True,
        use_advanced_decoder=False  # Use basic decoder first
    )
    
    print(f"Backbone features: {model.backbone_features}")
    print(f"Skip channels: {model.skip_channels}")
    
    # Create test input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 512, 512)
    
    print(f"Input shape: {test_input.shape}")
    
    try:
        # Try forward pass
        cls_out, seg_out = model(test_input)
        print(f" SUCCESS!")
        print(f"Classification output: {cls_out.shape}")
        print(f"Segmentation output: {seg_out.shape}")
    except Exception as e:
        print(f" ERROR: {e}")
        
        # Debug step by step
        print("\n=== STEP BY STEP DEBUG ===")
        
        # Get backbone features
        try:
            features_list = model.backbone(test_input)
            print(f"Features list length: {len(features_list)}")
            for i, feat in enumerate(features_list):
                print(f"  Feature {i}: {feat.shape}")
                
            final_features = features_list[-1]
            skip_features = features_list[:-1]
            print(f"Final features: {final_features.shape}")
            print(f"Skip features: {[f.shape for f in skip_features]}")
            
            # Try classification head
            cls_out = model.classification_head(final_features)
            print(f" Classification head works: {cls_out.shape}")
            
            # Try segmentation head
            print("\n=== DEBUGGING SEGMENTATION HEAD ===")
            seg_head = model.segmentation_head
            print(f"Segmentation head type: {type(seg_head)}")
            print(f"In features: {seg_head.in_features}")
            print(f"Use skip connections: {seg_head.use_skip_connections}")
            if hasattr(seg_head, 'skip_channels'):
                print(f"Skip channels: {seg_head.skip_channels}")
            
            # Try segmentation head forward
            seg_out = seg_head(final_features, skip_features)
            print(f" Segmentation head works: {seg_out.shape}")
            
        except Exception as e2:
            print(f" Detailed error: {e2}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_model_channels()
