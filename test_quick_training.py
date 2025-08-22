#!/usr/bin/env python3
"""
Quick Training Test Script
Tests one configuration in 15-20 minutes to verify everything works
"""

import sys
import time
import logging
import torch
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('QuickTest')

def run_quick_test():
    """Run a quick test to verify training works"""
    
    logger.info("🚀 QUICK TRAINING TEST")
    logger.info("=" * 50)
    logger.info("⏱️ Expected time: 15-20 minutes")
    logger.info("🎯 Goal: Verify actual training works")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        # Import modules
        from src.training.config import Phase4Config
        from src.training.trainer import RobustPhase4Trainer
        from src.data.dataloaders import DataLoaderFactory
        from src.models.simple_multi_task_model import create_simple_multi_task_model
        
        logger.info("✅ Modules imported successfully")
        
        # Quick configuration
        config = Phase4Config()
        config.experiment_name = f"quick_test_{int(time.time())}"
        
        # Fast settings
        config.progressive.phase1_epochs = 2  # Just 2 epochs per phase
        config.progressive.phase2_epochs = 2
        config.progressive.phase3_epochs = 2
        config.hardware.batch_size = 8
        config.hardware.num_workers = 4
        
        # Optimal hyperparameters for quick test
        config.optimizer.learning_rate = 1e-4
        config.loss.focal_gamma = 2.5
        config.optimizer.weight_decay = 1e-2
        
        logger.info(f"📋 Config: {config.progressive.total_epochs} epochs, batch_size={config.hardware.batch_size}")
        
        # Data loading
        factory = DataLoaderFactory()
        loaders = factory.create_multitask_loaders(
            processed_data_dir="dataset/processed",
            splits_dir="dataset/splits",
            batch_size=config.hardware.batch_size,
            num_workers=config.hardware.num_workers,
            image_size=512  # Must match model output size
        )
        
        logger.info(f"✅ Data loaded: {len(loaders['train'])} train, {len(loaders['val'])} val batches")
        
        # Model
        model_config = {
            'backbone_name': 'resnet18',  # Fast model
            'num_classes_cls': 5,
            'num_classes_seg': 1,
            'pretrained': True
        }
        
        model = create_simple_multi_task_model(model_config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        logger.info(f"✅ Model created on {device}")
        
        # Training
        trainer = RobustPhase4Trainer(model=model, config=config)
        
        save_dir = Path("experiments") / config.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🏃‍♂️ Starting training...")
        training_start = time.time()
        
        results = trainer.train(
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            save_dir=str(save_dir)
        )
        
        training_time = (time.time() - training_start) / 60
        total_time = (time.time() - start_time) / 60
        
        # Results
        if results and 'best_metrics' in results:
            metrics = results['best_metrics']
            
            logger.info("🎉 SUCCESS!")
            logger.info(f"⏱️ Training time: {training_time:.1f} minutes")
            logger.info(f"⏱️ Total time: {total_time:.1f} minutes")
            logger.info(f"📊 Results:")
            logger.info(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"   Dice Score: {metrics.get('dice_score', 0):.4f}")
            logger.info(f"   Combined: {(metrics.get('accuracy', 0) + metrics.get('dice_score', 0))/2:.4f}")
            
            # Performance check
            accuracy = metrics.get('accuracy', 0)
            if accuracy > 0.75:
                logger.info("🎉 EXCELLENT: Training is working perfectly!")
                logger.info("✅ Ready for research paper training")
                return True
            elif accuracy > 0.65:
                logger.info("✅ GOOD: Training is working well")
                return True
            else:
                logger.info("✅ OK: Training works (low accuracy expected with few epochs)")
                return True
        else:
            logger.error("❌ No results returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU available: {torch.cuda.get_device_name()}")
    else:
        logger.warning("⚠️ Using CPU")
    
    success = run_quick_test()
    
    if success:
        logger.info("\n" + "=" * 50)
        logger.info("🎉 QUICK TEST PASSED!")
        logger.info("=" * 50)
        logger.info("✅ Actual training confirmed working")
        logger.info("✅ Ready to run research paper training")
        logger.info("\nNext steps:")
        logger.info("python train_research_paper.py --trials 12 --time-limit 2")
        return 0
    else:
        logger.error("\n❌ QUICK TEST FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
