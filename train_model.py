"""
Quick Start Training Script for Multi-Task Diabetic Retinopathy Model
Run this script to start training with real-time monitoring
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path

# Import your custom modules
from src.training.trainer import MultiTaskTrainer, TrainingConfig
from src.models.multi_task_model import create_multi_task_model


# Quick dataset class for demo (replace with your actual dataset)
class DummyDRDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing - replace with your actual dataset"""
    
    def __init__(self, num_samples=100, image_size=512):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Dummy retinal image (3, 512, 512)
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Dummy DR grade (0-4)
        cls_label = torch.randint(0, 5, (1,)).long().squeeze()
        
        # Dummy segmentation mask (1, 512, 512) - binary mask
        seg_mask = torch.randint(0, 2, (1, self.image_size, self.image_size)).float()
        
        return image, cls_label, seg_mask


def setup_data_loaders(batch_size=4, num_workers=2):
    """Setup data loaders - replace with your actual data loading logic"""
    
    # Create dummy datasets (replace with your actual datasets)
    train_dataset = DummyDRDataset(num_samples=200, image_size=512)
    val_dataset = DummyDRDataset(num_samples=50, image_size=512)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Training configuration
    config = TrainingConfig(
        # Model configuration
        model_name="efficientnet_v2_s",
        num_classes=5,
        pretrained=True,
        use_skip_connections=True,
        use_advanced_decoder=False,  # Set to True for better segmentation
        
        # Training parameters
        batch_size=4,  # Adjust based on your GPU memory
        learning_rate=1e-4,
        num_epochs=50,  # Start with fewer epochs for testing
        
        # Progressive training
        classification_only_epochs=5,
        segmentation_warmup_epochs=10,
        
        # Loss weights
        segmentation_weight_end=0.8,
        
        # Optimization
        gradient_accumulation_steps=4,  # Effective batch size: 16
        use_mixed_precision=True,
        
        # Validation and saving
        val_every_n_epochs=1,
        patience=15,
        save_best_model=True,
        
        # Logging
        log_every_n_steps=5,
        save_tensorboard=True,
        save_plots=True
    )
    
    # Create trainer
    trainer = MultiTaskTrainer(config=config, device=device, logger=logger)
    
    # Setup data loaders
    logger.info("Setting up data loaders...")
    train_loader, val_loader = setup_data_loaders(
        batch_size=config.batch_size,
        num_workers=2
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Start training
    try:
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from_checkpoint=None  # Set path to resume training
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation score: {trainer.best_val_score:.4f}")
        
        # Print final metrics
        if history['val_cls_acc']:
            logger.info(f"Final validation accuracy: {history['val_cls_acc'][-1]:.3f}")
        if history['val_dice']:
            logger.info(f"Final validation dice: {history['val_dice'][-1]:.3f}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint on interruption
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
