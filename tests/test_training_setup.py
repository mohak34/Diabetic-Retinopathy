"""
Quick Test Script - Verify Training Setup
Run this before starting actual training to make sure everything works
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.multi_task_model import create_multi_task_model
from src.training.losses import MultiTaskLoss
from src.training.trainer import TrainingConfig, MultiTaskTrainer


def test_model_creation():
    """Test model creation and basic functionality"""
    print("Testing Testing model creation...")
    
    # Create model
    model = create_multi_task_model(
        backbone_name="efficientnet_v2_s",
        num_classes_cls=5,
        pretrained=True,
        use_skip_connections=True
    )
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 512).to(device)
    
    with torch.no_grad():
        cls_logits, seg_preds = model(input_tensor)
    
    print(f"   OK Model created successfully")
    print(f"   OK Input shape: {input_tensor.shape}")
    print(f"   OK Classification output: {cls_logits.shape}")
    print(f"   OK Segmentation output: {seg_preds.shape}")
    
    # Memory estimation
    if device.type == 'cuda':
        memory_mb = model.estimate_memory_usage(batch_size)
        print(f"   OK Estimated memory usage: {memory_mb:.1f} MB")
    
    return model, device


def test_loss_functions():
    """Test loss function computation"""
    print("\nTesting Testing loss functions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy predictions and targets
    batch_size = 4
    cls_logits = torch.randn(batch_size, 5).to(device)
    seg_preds = torch.randn(batch_size, 1, 512, 512).to(device)
    cls_labels = torch.randint(0, 5, (batch_size,)).to(device)
    seg_masks = torch.randint(0, 2, (batch_size, 1, 512, 512)).float().to(device)
    
    # Test loss function
    loss_fn = MultiTaskLoss(
        classification_weight=1.0,
        segmentation_weight=0.5,
        focal_gamma=2.0,
        dice_smooth=1e-5
    )
    
    total_loss, cls_loss, seg_loss = loss_fn(cls_logits, seg_preds, cls_labels, seg_masks)
    
    print(f"   OK Total loss: {total_loss.item():.4f}")
    print(f"   OK Classification loss: {cls_loss.item():.4f}")
    print(f"   OK Segmentation loss: {seg_loss.item():.4f}")
    
    return loss_fn


def test_data_loading():
    """Test data loading with dummy data"""
    print("\nTesting Testing data loading...")
    
    # Create dummy dataset
    batch_size = 4
    num_samples = 20
    
    images = torch.randn(num_samples, 3, 512, 512)
    cls_labels = torch.randint(0, 5, (num_samples,))
    seg_masks = torch.randint(0, 2, (num_samples, 1, 512, 512)).float()
    
    dataset = TensorDataset(images, cls_labels, seg_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Test iteration
    for batch_idx, (batch_images, batch_cls, batch_seg) in enumerate(dataloader):
        print(f"   OK Batch {batch_idx}: Images {batch_images.shape}, Labels {batch_cls.shape}, Masks {batch_seg.shape}")
        if batch_idx >= 2:  # Just test a few batches
            break
    
    return dataloader


def test_trainer_setup():
    """Test trainer initialization"""
    print("\nTesting Testing trainer setup...")
    
    config = TrainingConfig(
        batch_size=2,
        num_epochs=3,
        classification_only_epochs=1,
        segmentation_warmup_epochs=1,
        log_every_n_steps=1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MultiTaskTrainer(config=config, device=device)
    
    print(f"   OK Trainer initialized on {device}")
    print(f"   OK Config: batch_size={config.batch_size}, epochs={config.num_epochs}")
    
    return trainer, config


def test_training_step():
    """Test a single training step"""
    print("\nTesting Testing training step...")
    
    # Setup
    model, device = test_model_creation()
    loss_fn = test_loss_functions()
    dataloader = test_data_loading()
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Get one batch
    images, cls_labels, seg_masks = next(iter(dataloader))
    images = images.to(device)
    cls_labels = cls_labels.to(device)
    seg_masks = seg_masks.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    cls_logits, seg_preds = model(images)
    total_loss, cls_loss, seg_loss = loss_fn(cls_logits, seg_preds, cls_labels, seg_masks)
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    print(f"   OK Forward pass completed")
    print(f"   OK Loss computed: {total_loss.item():.4f}")
    print(f"   OK Backward pass completed")
    print(f"   OK Optimizer step completed")


def test_system_requirements():
    """Test system requirements"""
    print("\nTesting Testing system requirements...")
    
    # Check PyTorch
    print(f"   OK PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"   OK CUDA available: {torch.cuda.get_device_name()}")
        print(f"   OK CUDA version: {torch.version.cuda}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   OK GPU memory: {memory_gb:.1f} GB")
        
        if memory_gb < 6:
            print("   WARNING  Warning: Less than 6GB GPU memory. Consider reducing batch size.")
        else:
            print("   OK Sufficient GPU memory for training")
    else:
        print("   WARNING  CUDA not available - training will be slow on CPU")
    
    # Check if timm is installed
    try:
        import timm
        print(f"   OK timm library available: {timm.__version__}")
    except ImportError:
        print("   ERROR timm library not found - run: uv add timm")
        return False
    
    # Check if other dependencies are available
    required_modules = ['matplotlib', 'numpy', 'tqdm', 'torchmetrics']
    for module in required_modules:
        try:
            __import__(module)
            print(f"   OK {module} available")
        except ImportError:
            print(f"   ERROR {module} not found - install required dependencies")
            return False
    
    return True


def main():
    """Run all tests"""
    print("Test DIABETIC RETINOPATHY MODEL - TRAINING SETUP TEST")
    print("=" * 60)
    
    try:
        # Test system requirements first
        if not test_system_requirements():
            print("\nERROR System requirements not met. Please install missing dependencies.")
            return
        
        # Test individual components
        test_model_creation()
        test_loss_functions()
        test_data_loading()
        test_trainer_setup()
        test_training_step()
        
        print("\n" + "=" * 60)
        print("SUCCESS: ALL TESTS PASSED!")
        print("OK Your setup is ready for training")
        print("\n📝 Next steps:")
        print("   1. Replace dummy dataset with your actual data")
        print("   2. Run: python train_model.py")
        print("   3. Monitor with: python monitor_training.py")
        print("   4. Check TensorBoard: tensorboard --logdir logs")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
