#!/usr/bin/env python3
"""
Simple Phase 3 Test - Multi-Task Heads
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("PHASE 3.2: TESTING MULTI-TASK HEADS")
print("="*60)

try:
    # Test imports
    print("1. Testing imports...")
    import torch
    print("   OK PyTorch imported")
    
    from src.models.heads import ClassificationHead, SegmentationHead
    print("   OK Heads module imported")
    
    # Test parameters
    batch_size = 2
    backbone_features = 1280  # EfficientNetV2-S features
    feature_size = 16  # Typical feature map size
    
    # Create test features
    test_features = torch.randn(batch_size, backbone_features, feature_size, feature_size)
    print(f"   Test features shape: {test_features.shape}")
    
    # Test Classification Head
    print("\n2. Testing Classification Head...")
    cls_head = ClassificationHead(
        in_features=backbone_features,
        num_classes=5,  # DR grades 0-4
        dropout_rate=0.3
    )
    print("   OK Classification head created")
    
    with torch.no_grad():
        cls_output = cls_head(test_features)
        cls_probs = cls_head.get_probabilities(test_features)
    
    print(f"   OK Classification output: {cls_output.shape}")
    print(f"   OK Classification probabilities: {cls_probs.shape}")
    print(f"   OK Probability sum check: {cls_probs.sum(dim=1).mean().item():.4f}")
    
    # Test Segmentation Head
    print("\n3. Testing Segmentation Head...")
    seg_head = SegmentationHead(
        in_features=backbone_features,
        num_classes=1,  # Binary segmentation
        decoder_channels=[512, 256, 128, 64]
    )
    print("   OK Segmentation head created")
    
    with torch.no_grad():
        seg_output = seg_head(test_features)
    
    print(f"   OK Segmentation output: {seg_output.shape}")
    print(f"   OK Output range: [{seg_output.min().item():.4f}, {seg_output.max().item():.4f}]")
    
    # Validate outputs
    assert cls_output.shape == (batch_size, 5), f"Wrong cls shape: {cls_output.shape}"
    assert seg_output.shape[0] == batch_size, f"Wrong seg batch: {seg_output.shape[0]}"
    assert seg_output.shape[1] == 1, f"Wrong seg channels: {seg_output.shape[1]}"
    
    print("\n" + "="*60)
    print("OK PHASE 3.2 HEADS TEST: ALL PASSED!")
    print("OK Classification and Segmentation heads are working correctly")
    print("="*60)
    
except Exception as e:
    print(f"\nERROR ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*60)
    print("ERROR PHASE 3.2 HEADS TEST: FAILED")
    print("="*60)
    sys.exit(1)
