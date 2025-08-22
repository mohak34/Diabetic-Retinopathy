#!/usr/bin/env python3
"""
Simple, Fast Training Test Script
Tests actual training with a single configuration - should complete in 5-10 minutes
"""

import sys
import time
import logging
import torch
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('SimpleTrainingTest')

def test_single_config_fast():
    """Test a single hyperparameter configuration with minimal epochs"""
    
    logger.info("🧪 Testing ACTUAL training with single configuration (FAST)")
    logger.info("⏱️ Expected time: 5-10 minutes")
    
    start_time = time.time()
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        # Import required modules
        from src.training.config import Phase4Config
        from src.training.trainer import RobustPhase4Trainer
        from src.data.dataloaders import DataLoaderFactory
        from src.models.simple_multi_task_model import create_simple_multi_task_model
        
        logger.info("✅ All modules imported successfully")
        
        # Create minimal configuration for fast testing
        config = Phase4Config()
        config.experiment_name = f"test_fast_{int(time.time())}"
        
        # Set minimal epochs for speed
        config.progressive.phase1_epochs = 1  # Just 1 epoch per phase
        config.progressive.phase2_epochs = 1
        config.progressive.phase3_epochs = 1
        
        # Small batch size to go faster
        config.hardware.batch_size = 4
        config.hardware.num_workers = 2
        
        # Optimize for speed
        config.model.backbone_name = "resnet18"  # Smaller model
        config.early_stopping.patience = 1
        
        logger.info(f"📋 Config: {config.progressive.phase1_epochs + config.progressive.phase2_epochs + config.progressive.phase3_epochs} total epochs")
        logger.info(f"🔧 Batch size: {config.hardware.batch_size}")
        logger.info(f"🏗️ Model: {config.model.backbone_name}")
        
        # Create data loaders
        logger.info("📊 Creating data loaders...")
        factory = DataLoaderFactory()
        
        loaders = factory.create_multitask_loaders(
            processed_data_dir="dataset/processed",
            splits_dir="dataset/splits", 
            batch_size=config.hardware.batch_size,
            num_workers=config.hardware.num_workers,
            image_size=224  # Smaller image size for speed
        )
        
        train_loader = loaders['train']
        val_loader = loaders['val']
        
        logger.info(f"✅ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # Create simple model
        logger.info("🏗️ Creating model...")
        model_config = {
            'backbone_name': config.model.backbone_name,
            'num_classes_cls': config.model.num_classes,
            'num_classes_seg': config.model.segmentation_classes,
            'pretrained': True
        }
        
        model = create_simple_multi_task_model(model_config)
        
        # Move to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        logger.info(f"✅ Model created and moved to {device}")
        
        # Create trainer
        logger.info("🚀 Creating trainer...")
        trainer = RobustPhase4Trainer(model=model, config=config)
        logger.info("✅ Trainer created")
        
        # Create save directory
        save_dir = Path("experiments") / config.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Run actual training
        logger.info("🏃‍♂️ Starting ACTUAL training...")
        training_start = time.time()
        
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=str(save_dir)
        )
        
        training_time = time.time() - training_start
        logger.info(f"⏱️ Training completed in {training_time/60:.1f} minutes")
        
        # Check results
        if results and 'best_metrics' in results:
            metrics = results['best_metrics']
            logger.info("🎉 ACTUAL TRAINING SUCCESSFUL!")
            logger.info(f"📊 Results:")
            logger.info(f"   ✅ Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"   ✅ Dice Score: {metrics.get('dice_score', 0):.4f}")
            logger.info(f"   ✅ Combined Score: {metrics.get('combined_score', 0):.4f}")
            logger.info(f"   ✅ F1 Score: {metrics.get('f1_score', 0):.4f}")
            
            total_time = time.time() - start_time
            logger.info(f"🕐 Total test time: {total_time/60:.1f} minutes")
            
            return True, metrics
        else:
            logger.error("❌ Training completed but no metrics returned")
            return False, {}
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def main():
    """Main test function"""
    
    logger.info("=" * 60)
    logger.info("SIMPLE FAST TRAINING TEST")
    logger.info("=" * 60)
    logger.info("🎯 Goal: Verify ACTUAL training works (not simulation)")
    logger.info("⏱️ Expected: 5-10 minutes total")
    logger.info("🔥 Hardware: GPU training if available")
    logger.info("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("⚠️ CUDA not available - using CPU (will be slower)")
    
    # Run test
    success, metrics = test_single_config_fast()
    
    # Final results
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    
    if success:
        logger.info("🎉 SUCCESS: ACTUAL training is working!")
        logger.info("✅ The hyperparameter optimization system can now use real training")
        logger.info("✅ No more simulation - real neural network training confirmed")
        
        # Performance check
        accuracy = metrics.get('accuracy', 0)
        dice = metrics.get('dice_score', 0)
        
        if accuracy > 0.7 and dice > 0.5:
            logger.info("🏆 Performance looks reasonable for minimal training")
        else:
            logger.warning("⚠️ Performance is low (expected with minimal epochs)")
        
        logger.info("\n🚀 Ready to run hyperparameter optimization!")
        return 0
    else:
        logger.error("❌ FAILED: Issues with actual training implementation")
        logger.error("🔧 Need to fix training pipeline before hyperparameter optimization")
        return 1

if __name__ == "__main__":
    sys.exit(main())
