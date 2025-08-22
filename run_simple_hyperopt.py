#!/usr/bin/env python3
"""
Simple Hyperparameter Optimization Script
Tests 3 research-backed configurations quickly (15-30 minutes total)
"""

import sys
import time
import logging
import torch
import json
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('SimpleHyperopt')

def test_hyperparameter_config(config_name: str, hyperparams: dict, trial_num: int):
    """Test a single hyperparameter configuration"""
    
    logger.info(f"\n� TRIAL {trial_num}: {config_name}")
    logger.info(f"📋 Parameters: {hyperparams}")
    
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
        
        # Create configuration
        config = Phase4Config()
        config.experiment_name = f"hyperopt_{config_name}_{int(time.time())}"
        
        # Apply hyperparameters
        config.optimizer.learning_rate = hyperparams['learning_rate']
        config.optimizer.weight_decay = hyperparams['weight_decay']
        config.hardware.batch_size = hyperparams['batch_size']
        config.loss.focal_gamma = hyperparams['focal_gamma']
        config.model.cls_dropout = hyperparams['classification_dropout']
        
        # Set minimal epochs for speed (total: 6 epochs)
        config.progressive.phase1_epochs = 2
        config.progressive.phase2_epochs = 2
        config.progressive.phase3_epochs = 2
        
        # Small image size for speed
        image_size = 224
        
        logger.info(f"⚙️ LR: {config.optimizer.learning_rate}, Batch: {config.hardware.batch_size}")
        logger.info(f"📐 Epochs: {config.progressive.total_epochs}, Image size: {image_size}")
        
        # Create data loaders
        factory = DataLoaderFactory()
        
        loaders = factory.create_multitask_loaders(
            processed_data_dir="dataset/processed",
            splits_dir="dataset/splits", 
            batch_size=config.hardware.batch_size,
            num_workers=2,
            image_size=image_size
        )
        
        train_loader = loaders['train']
        val_loader = loaders['val']
        
        logger.info(f"✅ Data: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Create model
        model_config = {
            'backbone_name': 'resnet18',  # Fast model for testing
            'num_classes_cls': config.model.num_classes,
            'num_classes_seg': config.model.segmentation_classes,
            'pretrained': True
        }
        
        model = create_simple_multi_task_model(model_config)
        
        # Move to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # Create trainer
        trainer = RobustPhase4Trainer(model=model, config=config)
        
        # Create save directory
        save_dir = Path("experiments") / config.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Run training
        logger.info("🏃‍♂️ Training...")
        training_start = time.time()
        
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=str(save_dir)
        )
        
        training_time = time.time() - training_start
        total_time = time.time() - start_time
        
        # Extract metrics
        if results and 'best_metrics' in results:
            metrics = results['best_metrics']
            
            trial_result = {
                'trial': trial_num,
                'config_name': config_name,
                'hyperparams': hyperparams,
                'metrics': {
                    'accuracy': metrics.get('accuracy', 0),
                    'dice_score': metrics.get('dice_score', 0),
                    'combined_score': metrics.get('combined_score', 0),
                    'f1_score': metrics.get('f1_score', 0)
                },
                'training_time_minutes': training_time / 60,
                'total_time_minutes': total_time / 60,
                'status': 'completed'
            }
            
            logger.info(f"✅ COMPLETED in {total_time/60:.1f} min")
            logger.info(f"📊 Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"📊 Dice: {metrics.get('dice_score', 0):.4f}")
            logger.info(f"📊 Combined: {metrics.get('combined_score', 0):.4f}")
            
            return trial_result
        else:
            logger.error("❌ No metrics returned")
            return {
                'trial': trial_num,
                'config_name': config_name,
                'hyperparams': hyperparams,
                'status': 'failed',
                'error': 'No metrics returned'
            }
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Trial failed after {total_time/60:.1f} min: {e}")
        return {
            'trial': trial_num,
            'config_name': config_name,
            'hyperparams': hyperparams,
            'status': 'failed',
            'error': str(e),
            'total_time_minutes': total_time / 60
        }

def main():
    """Main hyperparameter optimization function"""
    
    logger.info("=" * 60)
    logger.info("SIMPLE HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 60)
    logger.info("🎯 Testing 3 research-backed configurations")
    logger.info("⏱️ Expected time: 15-30 minutes total")
    logger.info("🔥 Each trial: ~5-10 minutes")
    logger.info("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    else:
        logger.warning("⚠️ Using CPU (will be slower)")
    
    # Define 3 research-backed configurations
    configurations = [
        {
            'name': 'conservative_stable',
            'params': {
                'learning_rate': 1e-4,
                'focal_gamma': 2.0,
                'weight_decay': 1e-2,
                'batch_size': 8,
                'classification_dropout': 0.3
            }
        },
        {
            'name': 'balanced_optimal',
            'params': {
                'learning_rate': 5e-4,
                'focal_gamma': 2.5,
                'weight_decay': 5e-3,
                'batch_size': 16,
                'classification_dropout': 0.4
            }
        },
        {
            'name': 'aggressive_performance',
            'params': {
                'learning_rate': 1e-3,
                'focal_gamma': 3.0,
                'weight_decay': 1e-3,
                'batch_size': 4,
                'classification_dropout': 0.5
            }
        }
    ]
    
    logger.info(f"� Testing {len(configurations)} configurations:")
    for i, config in enumerate(configurations, 1):
        logger.info(f"  {i}. {config['name']}")
    
    # Run experiments
    results = []
    start_time = time.time()
    
    for i, config in enumerate(configurations, 1):
        result = test_hyperparameter_config(
            config_name=config['name'],
            hyperparams=config['params'],
            trial_num=i
        )
        results.append(result)
        
        # Show progress
        completed = sum(1 for r in results if r.get('status') == 'completed')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        elapsed = (time.time() - start_time) / 60
        
        logger.info(f"\n📈 Progress: {completed} completed, {failed} failed, {elapsed:.1f} min elapsed")
    
    # Final analysis
    total_time = (time.time() - start_time) / 60
    successful_results = [r for r in results if r.get('status') == 'completed']
    
    logger.info("\n" + "=" * 60)
    logger.info("HYPERPARAMETER OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"⏱️ Total time: {total_time:.1f} minutes")
    logger.info(f"✅ Successful: {len(successful_results)}/{len(configurations)}")
    
    if successful_results:
        # Find best configuration
        best_result = max(successful_results, 
                         key=lambda x: x['metrics']['combined_score'])
        
        logger.info(f"\n🏆 BEST CONFIGURATION:")
        logger.info(f"  Name: {best_result['config_name']}")
        logger.info(f"  Combined Score: {best_result['metrics']['combined_score']:.4f}")
        logger.info(f"  Accuracy: {best_result['metrics']['accuracy']:.4f}")
        logger.info(f"  Dice Score: {best_result['metrics']['dice_score']:.4f}")
        logger.info(f"  Training Time: {best_result['training_time_minutes']:.1f} min")
        
        logger.info(f"\n📋 Best Parameters:")
        for param, value in best_result['hyperparams'].items():
            logger.info(f"  {param}: {value}")
        
        # Show all results
        logger.info(f"\n📊 All Results:")
        for result in sorted(successful_results, 
                           key=lambda x: x['metrics']['combined_score'], 
                           reverse=True):
            score = result['metrics']['combined_score']
            acc = result['metrics']['accuracy']
            dice = result['metrics']['dice_score']
            time_min = result['training_time_minutes']
            logger.info(f"  {result['config_name']}: {score:.4f} (acc:{acc:.3f}, dice:{dice:.3f}, {time_min:.1f}min)")
        
        # Save results
        results_file = Path("results") / f"hyperopt_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'experiment_type': 'simple_hyperopt',
                'total_time_minutes': total_time,
                'configurations_tested': len(configurations),
                'successful_runs': len(successful_results),
                'best_config': best_result,
                'all_results': results
            }, f, indent=2)
        
        logger.info(f"\n� Results saved to: {results_file}")
        
        # Performance assessment
        best_acc = best_result['metrics']['accuracy']
        if best_acc > 0.8:
            logger.info("🎉 EXCELLENT: Hyperparameter optimization is working great!")
        elif best_acc > 0.7:
            logger.info("✅ GOOD: Hyperparameter optimization is working well")
        else:
            logger.warning("⚠️ OKAY: Results are reasonable for minimal training")
        
        logger.info("\n🚀 Ready for full hyperparameter optimization!")
        return 0
    else:
        logger.error("❌ All configurations failed - check training implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
