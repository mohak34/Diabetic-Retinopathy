#!/usr/bin/env python3
"""
Simple Phase 3 Test - Integrated Multi-Task Model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("PHASE 3.4: TESTING INTEGRATED MULTI-TASK MODEL")
print("="*60)

try:
    # Test imports
    print("1. Testing imports...")
    import torch
    print("   OK PyTorch imported")
    
    from src.models.multi_task_model import create_multi_task_model
    print("   OK Multi-task model imported")
    
    # Test parameters
    batch_size = 2
    input_size = 224  # Smaller size for faster testing
    
    # Create test input
    test_input = torch.randn(batch_size, 3, input_size, input_size)
    print(f"   Test input shape: {test_input.shape}")
    
    # Test basic model creation
    print("\n2. Testing model creation...")
    model = create_multi_task_model(
        num_classes_cls=5,
        num_classes_seg=1,
        backbone_name='efficientnetv2_s',
        pretrained=False,  # Faster for testing
        use_skip_connections=False
    )
    print("   OK Multi-task model created")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    with torch.no_grad():
        cls_output, seg_output = model(test_input)
    
    print(f"   OK Classification output: {cls_output.shape}")
    print(f"   OK Segmentation output: {seg_output.shape}")
    
    # Validate outputs
    assert cls_output.shape == (batch_size, 5), f"Wrong cls shape: {cls_output.shape}"
    assert seg_output.shape[0] == batch_size, f"Wrong seg batch: {seg_output.shape[0]}"
    
    # Test prediction methods
    print("\n4. Testing prediction methods...")
    cls_pred = model.predict_classification(test_input)
    seg_pred = model.predict_segmentation(test_input)
    
    print(f"   OK Classification predictions: {list(cls_pred.keys())}")
    print(f"   OK Segmentation predictions: {list(seg_pred.keys())}")
    
    # Test model info
    print("\n5. Testing model analysis...")
    model_size = model.get_model_size()
    memory_usage = model.estimate_memory_usage(batch_size=4, input_size=512)
    
    print(f"   Total parameters: {model_size['total_parameters']:,}")
    print(f"   Trainable parameters: {model_size['trainable_parameters']:,}")
    print(f"   Estimated memory (4x512): {memory_usage['total_estimated']:.2f} MB")
    
    # Check RTX 3080 constraint
    memory_gb = memory_usage['total_estimated'] / 1000
    if memory_gb <= 8.0:
        print(f"   OK Memory constraint satisfied: {memory_gb:.2f} GB ≤ 8.0 GB")
    else:
        print(f"   WARNING  Memory might exceed RTX 3080: {memory_gb:.2f} GB")
    
    print("\n" + "="*60)
    print("OK PHASE 3.4 INTEGRATED MODEL TEST: ALL PASSED!")
    print("OK Multi-task model is ready for training")
    print("="*60)
    
except Exception as e:
    print(f"\nERROR ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*60)
    print("ERROR PHASE 3.4 INTEGRATED MODEL TEST: FAILED")
    print("="*60)
    sys.exit(1)
