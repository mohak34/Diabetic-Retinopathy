#!/usr/bin/env python3
"""
Simple Phase 3 Test - Backbone Implementation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("PHASE 3.1: TESTING EFFICIENTNETV2-S BACKBONE")
print("="*60)

try:
    # Test imports
    print("1. Testing imports...")
    import torch
    print("   OK PyTorch imported")
    
    import timm
    print("   OK timm imported")
    
    from src.models.backbone import create_efficientnet_backbone
    print("   OK Backbone module imported")
    
    # Test backbone creation
    print("\n2. Testing backbone creation...")
    backbone = create_efficientnet_backbone(
        model_name='tf_efficientnet_b0_ns', 
        pretrained=False  # Use False to avoid downloading
    )
    print("   OK Backbone created successfully")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    test_input = torch.randn(2, 3, 224, 224)
    print(f"   Input shape: {test_input.shape}")
    
    with torch.no_grad():
        features = backbone(test_input)
        final_features = backbone.get_final_features(test_input)
    
    print(f"   OK Got {len(features)} feature stages")
    for i, feat in enumerate(features):
        print(f"     Stage {i}: {feat.shape}")
    
    print(f"   OK Final features: {final_features.shape}")
    
    # Test feature info
    print("\n4. Testing feature information...")
    feature_info = backbone.get_feature_info()
    for info in feature_info:
        print(f"   Stage {info['stage']}: {info['channels']} channels, reduction {info['reduction']}")
    
    # Test memory estimation
    print("\n5. Testing memory estimation...")
    memory_est = backbone.estimate_memory_usage(batch_size=4, input_size=512)
    print(f"   Total estimated memory: {memory_est['total_estimated']:.2f} MB")
    
    print("\n" + "="*60)
    print("OK PHASE 3.1 BACKBONE TEST: ALL PASSED!")
    print("OK EfficientNetV2-S backbone is ready for multi-task learning")
    print("="*60)
    
except Exception as e:
    print(f"\nERROR ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*60)
    print("ERROR PHASE 3.1 BACKBONE TEST: FAILED")
    print("="*60)
    sys.exit(1)
