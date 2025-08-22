#!/usr/bin/env python3
"""
Fixed Diabetic Retinopathy Training Script
This script provides a simple, reliable way to train the model with actual training (no simulation)
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_logging(experiment_name: str, log_level: str = "INFO"):
    """Setup logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/train_{experiment_name}_{timestamp}.log"
    
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('FixedTraining')

def run_actual_training(experiment_name: str = None, 
                       epochs: int = 15,
                       batch_size: int = 16,
                       learning_rate: float = 1e-4,
                       log_level: str = "INFO"):
    """Run actual model training with specified parameters"""
    
    if experiment_name is None:
        experiment_name = f"fixed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger = setup_logging(experiment_name, log_level)
    
    logger.info("🚀 DIABETIC RETINOPATHY TRAINING - FIXED VERSION")
    logger.info("="*70)
    logger.info(f"🧪 Experiment: {experiment_name}")
    logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"⚙️ Parameters:")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch Size: {batch_size}")
    logger.info(f"   Learning Rate: {learning_rate}")
    logger.info("="*70)
    
    try:
        # Import training components
        from src.training.config import Phase4Config
        from src.training.trainer import RobustPhase4Trainer
        from src.models.simple_multi_task_model import create_simple_multi_task_model
        from src.data.dataloaders import DataLoaderFactory
        
        logger.info("✅ All modules imported successfully")
        
        # Create configuration
        config = Phase4Config()
        config.experiment_name = experiment_name
        config.experiment_dir = f"experiments/{experiment_name}"
        
        # Update configuration with parameters
        config.hardware.batch_size = batch_size
        config.optimizer.learning_rate = learning_rate
        config.progressive.phase1_epochs = max(1, epochs // 3)
        config.progressive.phase2_epochs = max(1, epochs // 3)
        config.progressive.phase3_epochs = max(1, epochs // 3)
        
        logger.info("✅ Configuration created")
        
        # Create model
        model_config = {
            'backbone_name': 'tf_efficientnet_b0_ns',
            'num_classes_cls': 5,
            'num_classes_seg': 1,
            'pretrained': True
        }
        
        model = create_simple_multi_task_model(model_config)
        logger.info("✅ Model created successfully")
        
        # Create data loaders
        factory = DataLoaderFactory()
        loaders = factory.create_multitask_loaders(
            processed_data_dir="dataset/processed",
            splits_dir="dataset/splits",
            batch_size=batch_size,
            num_workers=4,
            image_size=512
        )
        
        train_loader = loaders['train']
        val_loader = loaders['val']
        
        logger.info(f"✅ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # Create trainer
        trainer = RobustPhase4Trainer(model=model, config=config)
        logger.info("✅ Trainer created successfully")
        
        # Create experiment directory
        experiment_dir = Path(config.experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Start training
        logger.info("🚀 Starting ACTUAL training...")
        logger.info(f"💾 Results will be saved to: {experiment_dir}")
        
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=str(experiment_dir)
        )
        
        # Extract and report results
        if training_results and 'best_metrics' in training_results:
            metrics = training_results['best_metrics']
            
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*50)
            logger.info("📊 FINAL RESULTS:")
            logger.info(f"   🎯 Classification Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"   🎯 Segmentation Dice Score: {metrics.get('dice_score', 0):.4f}")
            logger.info(f"   🎯 Combined Score: {metrics.get('combined_score', 0):.4f}")
            logger.info(f"   📊 Sensitivity: {metrics.get('sensitivity', 0):.4f}")
            logger.info(f"   📊 Specificity: {metrics.get('specificity', 0):.4f}")
            logger.info(f"   📊 F1 Score: {metrics.get('f1_score', 0):.4f}")
            logger.info(f"   📊 AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
            logger.info("="*50)
            logger.info(f"💾 Model saved to: {experiment_dir}")
            
            # Performance analysis
            accuracy = metrics.get('accuracy', 0)
            if accuracy >= 0.90:
                logger.info("🏆 EXCELLENT: Accuracy ≥90%")
            elif accuracy >= 0.85:
                logger.info("✅ GOOD: Accuracy ≥85%")
            elif accuracy >= 0.80:
                logger.info("⚠️ ACCEPTABLE: Accuracy ≥80%")
            else:
                logger.info("❌ NEEDS IMPROVEMENT: Accuracy <80%")
            
            return {
                'status': 'success',
                'experiment_name': experiment_name,
                'metrics': metrics,
                'experiment_dir': str(experiment_dir),
                'training_results': training_results
            }
        else:
            logger.error("❌ Training completed but no metrics returned")
            return {
                'status': 'error',
                'error': 'No metrics returned from training',
                'experiment_name': experiment_name
            }
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        return {
            'status': 'error',
            'error': str(e),
            'experiment_name': experiment_name
        }

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(
        description='Fixed Diabetic Retinopathy Training Script'
    )
    
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate for optimizer')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    try:
        results = run_actual_training(
            experiment_name=args.experiment_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            log_level=args.log_level
        )
        
        if results['status'] == 'success':
            print(f"\n🎉 Training completed successfully!")
            print(f"📄 Experiment: {results['experiment_name']}")
            print(f"💾 Results: {results['experiment_dir']}")
            
            metrics = results['metrics']
            accuracy = metrics.get('accuracy', 0)
            dice = metrics.get('dice_score', 0)
            print(f"🎯 Accuracy: {accuracy:.4f}")
            print(f"🎯 Dice Score: {dice:.4f}")
            print(f"🎯 Combined: {(accuracy + dice) / 2:.4f}")
            
            return 0
        else:
            print(f"❌ Training failed: {results['error']}")
            return 1
            
    except Exception as e:
        print(f"❌ Script execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
