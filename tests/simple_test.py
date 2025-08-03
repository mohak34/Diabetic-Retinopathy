"""
Simple Training Test - Quick Verification
"""

import sys
import os

# Add src to path properly
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"OK PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"ERROR PyTorch import failed: {e}")
        return False
    
    try:
        import timm
        print(f"OK timm: {timm.__version__}")
    except ImportError as e:
        print(f"ERROR timm import failed: {e}")
        return False
    
    try:
        from models.multi_task_model import create_multi_task_model
        print("OK Multi-task model import successful")
    except ImportError as e:
        print(f"ERROR Model import failed: {e}")
        return False
    
    try:
        from training.losses import MultiTaskLoss
        print("OK Loss functions import successful")
    except ImportError as e:
        print(f"ERROR Loss import failed: {e}")
        return False
    
    return True

def test_device():
    """Test device availability"""
    import torch
    
    print("\nTesting device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"OK Device: {device}")
    
    if torch.cuda.is_available():
        print(f"OK GPU: {torch.cuda.get_device_name()}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"OK GPU Memory: {memory_gb:.1f} GB")
    
    return device

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        from models.multi_task_model import create_multi_task_model
        
        model = create_multi_task_model(
            backbone_name="efficientnet_v2_s",
            num_classes_cls=5,
            pretrained=True
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"OK Model created: {total_params:,} parameters")
        
        return model
    except Exception as e:
        print(f"ERROR Model creation failed: {e}")
        return None

def main():
    print("Test SIMPLE TRAINING TEST")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\nERROR Import test failed")
        return
    
    # Test device
    device = test_device()
    
    # Test model
    model = test_model_creation()
    
    if model is not None:
        print("\nSUCCESS: ALL BASIC TESTS PASSED!")
        print("OK Ready to start training")
    else:
        print("\nERROR Model test failed")

if __name__ == "__main__":
    main()
