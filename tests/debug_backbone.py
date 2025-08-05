#!/usr/bin/env python3
"""
Debug backbone features to understand the mismatch
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.backbone import create_efficientnet_backbone

def debug_backbone():
    """Debug backbone feature info vs actual features"""
    print("=== DEBUGGING BACKBONE FEATURES ===")
    
    # Create backbone
    backbone = create_efficientnet_backbone(
        model_name="tf_efficientnetv2_b0",
        pretrained=False
    )
    
    # Get feature info
    feature_info = backbone.get_feature_info()
    print(f"Feature info length: {len(feature_info)}")
    for i, info in enumerate(feature_info):
        print(f"  Stage {i}: {info}")
    
    # Test forward pass
    test_input = torch.randn(2, 3, 512, 512)
    features = backbone(test_input)
    
    print(f"\nActual features length: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Feature {i}: {feat.shape}")
    
    # Compare
    print(f"\nFeature info suggests: {[info['channels'] for info in feature_info]} channels")
    print(f"Actual features have: {[f.shape[1] for f in features]} channels")

if __name__ == "__main__":
    debug_backbone()
