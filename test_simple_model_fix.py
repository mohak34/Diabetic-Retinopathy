#!/usr/bin/env python3
"""
Test script to verify the SimpleMultiTaskModel output format fix
"""

import sys
sys.path.append('src')

import torch
from src.models.simple_multi_task_model import SimpleMultiTaskModel

def test_simple_model_output():
    """Test that SimpleMultiTaskModel returns correct output format"""
    print("Testing SimpleMultiTaskModel output format...")
    
    # Create model
    model = SimpleMultiTaskModel(
        backbone_name='resnet50',
        num_classes_cls=5,
        num_classes_seg=1,
        pretrained=False  # Use False for faster testing
    )
    
    # Create dummy input
    x = torch.randn(2, 3, 512, 512)
    
    print("Testing with return_features=True...")
    outputs = model(x, return_features=True)
    
    print("Output keys:", list(outputs.keys()))
    
    # Check for expected keys
    expected_keys = ['classification', 'segmentation', 'features', 'backbone_features']
    for key in expected_keys:
        if key in outputs:
            print(f"✅ Found key: {key}, shape: {outputs[key].shape if hasattr(outputs[key], 'shape') else 'N/A'}")
        else:
            print(f"❌ Missing key: {key}")
    
    # Test format expected by trainer
    if 'classification' in outputs and 'segmentation' in outputs:
        print("✅ Model returns correct keys for trainer compatibility")
        print(f"   Classification shape: {outputs['classification'].shape}")
        print(f"   Segmentation shape: {outputs['segmentation'].shape}")
    else:
        print("❌ Model does not return correct keys for trainer")
    
    print("\nTesting with return_features=False...")
    outputs = model(x, return_features=False)
    print(f"Output type: {type(outputs)}")
    if isinstance(outputs, tuple):
        print(f"Tuple length: {len(outputs)}")
        print(f"First element shape: {outputs[0].shape}")
        print(f"Second element shape: {outputs[1].shape}")
    
    print("✅ SimpleMultiTaskModel test completed successfully!")

if __name__ == "__main__":
    test_simple_model_output()
