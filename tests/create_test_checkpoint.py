"""
Create a dummy checkpoint for testing Phase 6 evaluation
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level from tests/ to project root
sys.path.append(str(project_root))

from src.models.multi_task_model import MultiTaskRetinaModel

def create_dummy_checkpoint():
    """Create a dummy trained model checkpoint for testing"""
    
    # Create model
    model = MultiTaskRetinaModel(
        num_classes_cls=5,
        num_classes_seg=4,
        backbone_name='tf_efficientnet_b0_ns',
        pretrained=False  # Use False for testing to avoid download
    )
    
    # Create dummy checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 50,
        'train_loss': 0.25,
        'val_loss': 0.28,
        'val_accuracy': 0.82,
        'val_dice': 0.75,
        'optimizer_state_dict': {},  # Empty optimizer state
        'config': {
            'num_classes': 5,
            'num_segmentation_classes': 4,
            'backbone': 'tf_efficientnet_b0_ns'
        }
    }
    
    # Save checkpoint
    checkpoint_dir = project_root / 'experiments/phase6_test_model/checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / 'best_model.pth'
    torch.save(checkpoint, checkpoint_path)
    
    print(f"Dummy checkpoint created at: {checkpoint_path}")
    return str(checkpoint_path)

if __name__ == "__main__":
    checkpoint_path = create_dummy_checkpoint()
    print(f"Test checkpoint available at: {checkpoint_path}")
