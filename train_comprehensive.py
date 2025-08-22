#!/usr/bin/env python3
"""
Comprehensive Fixed Training Script for Diabetic Retinopathy Detection
This script provides multiple training modes with proper error handling and actual training
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_logging(experiment_name: str, log_level: str = "INFO"):
    """Setup comprehensive logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/comprehensive_{experiment_name}_{timestamp}.log"
    
    Path("logs").mkdir(exist_ok=True)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('ComprehensiveTraining')

class FixedTrainer:
    """Fixed trainer that handles all configuration issues properly"""
    
    def __init__(self, experiment_name: str, logger: logging.Logger):
        self.experiment_name = experiment_name
        self.logger = logger
        self.experiment_dir = Path(f"experiments/{experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def create_working_config(self, 
                             batch_size: int = 16,
                             learning_rate: float = 1e-4,
                             epochs: int = 15) -> Any:
        """Create a working configuration that avoids all known issues"""
        
        try:
            from src.training.config import Phase4Config
            
            config = Phase4Config()
            
            # Set basic parameters
            config.experiment_name = self.experiment_name
            config.experiment_dir = str(self.experiment_dir)
            
            # Hardware configuration
            config.hardware.batch_size = batch_size
            config.hardware.num_workers = 4
            config.hardware.mixed_precision = True
            config.hardware.gradient_accumulation_steps = max(1, 32 // batch_size)
            
            # Optimizer configuration  
            config.optimizer.learning_rate = learning_rate
            config.optimizer.weight_decay = 1e-4
            
            # Progressive training - distribute epochs
            phase_epochs = max(1, epochs // 3)
            config.progressive.phase1_epochs = phase_epochs
            config.progressive.phase2_epochs = phase_epochs
            config.progressive.phase3_epochs = epochs - (2 * phase_epochs)
            
            # Loss configuration
            config.loss.focal_gamma = 2.0
            config.loss.dice_smooth = 1e-6
            
            # Model configuration
            config.model.backbone_name = "tf_efficientnet_b0_ns"
            config.model.num_classes = 5
            config.model.pretrained = True
            
            # Early stopping
            config.early_stopping.enabled = True
            config.early_stopping.patience = max(3, epochs // 5)
            
            self.logger.info(f"✅ Configuration created: {epochs} total epochs")
            self.logger.info(f"   Phase 1: {config.progressive.phase1_epochs} epochs")
            self.logger.info(f"   Phase 2: {config.progressive.phase2_epochs} epochs") 
            self.logger.info(f"   Phase 3: {config.progressive.phase3_epochs} epochs")
            self.logger.info(f"   Total: {config.progressive.total_epochs} epochs")
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Config creation failed: {e}")
            raise
    
    def create_model(self, backbone_name: str = "tf_efficientnet_b0_ns") -> Any:
        """Create model with proper error handling"""
        
        try:
            from src.models.simple_multi_task_model import create_simple_multi_task_model
            
            model_config = {
                'backbone_name': backbone_name,
                'num_classes_cls': 5,
                'num_classes_seg': 1,
                'pretrained': True
            }
            
            self.logger.info(f"🏗️ Creating model: {backbone_name}")
            model = create_simple_multi_task_model(model_config)
            
            # Move to device
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            
            self.logger.info(f"✅ Model created and moved to {device}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Model creation failed: {e}")
            raise
    
    def create_data_loaders(self, batch_size: int = 16) -> Dict[str, Any]:
        """Create data loaders with proper error handling"""
        
        try:
            from src.data.dataloaders import DataLoaderFactory
            
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
            
            self.logger.info(f"✅ Data loaders created:")
            self.logger.info(f"   Train: {len(train_loader)} batches ({len(train_loader.dataset)} images)")
            self.logger.info(f"   Val: {len(val_loader)} batches ({len(val_loader.dataset)} images)")
            
            return {'train': train_loader, 'val': val_loader}
            
        except Exception as e:
            self.logger.error(f"❌ Data loader creation failed: {e}")
            raise
    
    def run_training(self, 
                    batch_size: int = 16,
                    learning_rate: float = 1e-4,
                    epochs: int = 15,
                    backbone_name: str = "tf_efficientnet_b0_ns") -> Dict[str, Any]:
        """Run complete training with proper error handling"""
        
        self.logger.info("🚀 STARTING COMPREHENSIVE DIABETIC RETINOPATHY TRAINING")
        self.logger.info("="*80)
        self.logger.info(f"🧪 Experiment: {self.experiment_name}")
        self.logger.info(f"⚙️ Batch Size: {batch_size}")
        self.logger.info(f"⚙️ Learning Rate: {learning_rate}")
        self.logger.info(f"⚙️ Epochs: {epochs}")
        self.logger.info(f"⚙️ Backbone: {backbone_name}")
        self.logger.info("="*80)
        
        try:
            # Step 1: Create configuration
            self.logger.info("📋 Step 1: Creating configuration...")
            config = self.create_working_config(batch_size, learning_rate, epochs)
            
            # Step 2: Create model
            self.logger.info("🏗️ Step 2: Creating model...")
            model = self.create_model(backbone_name)
            
            # Step 3: Create data loaders
            self.logger.info("📁 Step 3: Creating data loaders...")
            loaders = self.create_data_loaders(batch_size)
            
            # Step 4: Create trainer
            self.logger.info("🎯 Step 4: Creating trainer...")
            from src.training.trainer import RobustPhase4Trainer
            trainer = RobustPhase4Trainer(model=model, config=config)
            self.logger.info("✅ Trainer created successfully")
            
            # Step 5: Run actual training
            self.logger.info("🚀 Step 5: Starting ACTUAL training...")
            self.logger.info(f"💾 Results will be saved to: {self.experiment_dir}")
            
            training_results = trainer.train(
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                save_dir=str(self.experiment_dir)
            )
            
            # Step 6: Process results
            self.logger.info("📊 Step 6: Processing results...")
            return self.process_results(training_results)
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            self.logger.error(f"❌ Error type: {type(e).__name__}")
            return {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__,
                'experiment_name': self.experiment_name
            }
    
    def process_results(self, training_results: Any) -> Dict[str, Any]:
        """Process and save training results"""
        
        if training_results and 'best_metrics' in training_results:
            metrics = training_results['best_metrics']
            
            self.logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            self.logger.info("📊 FINAL RESULTS:")
            self.logger.info(f"   🎯 Classification Accuracy: {metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"   🎯 Segmentation Dice Score: {metrics.get('dice_score', 0):.4f}")
            self.logger.info(f"   🎯 Combined Score: {metrics.get('combined_score', 0):.4f}")
            
            if 'sensitivity' in metrics:
                self.logger.info(f"   📊 Sensitivity: {metrics.get('sensitivity', 0):.4f}")
            if 'specificity' in metrics:
                self.logger.info(f"   📊 Specificity: {metrics.get('specificity', 0):.4f}")
            if 'f1_score' in metrics:
                self.logger.info(f"   📊 F1 Score: {metrics.get('f1_score', 0):.4f}")
            if 'auc_roc' in metrics:
                self.logger.info(f"   📊 AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
            
            self.logger.info("="*60)
            self.logger.info(f"💾 Model saved to: {self.experiment_dir}")
            
            # Performance analysis
            accuracy = metrics.get('accuracy', 0)
            if accuracy >= 0.90:
                performance_level = "🏆 EXCELLENT"
            elif accuracy >= 0.85:
                performance_level = "✅ GOOD"
            elif accuracy >= 0.80:
                performance_level = "⚠️ ACCEPTABLE"
            else:
                performance_level = "❌ NEEDS IMPROVEMENT"
            
            self.logger.info(f"📈 Performance Level: {performance_level} (Accuracy: {accuracy:.1%})")
            
            # Save results summary
            results_summary = {
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'metrics': metrics,
                'performance_level': performance_level,
                'experiment_dir': str(self.experiment_dir)
            }
            
            results_file = self.experiment_dir / "training_summary.json"
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            return results_summary
        else:
            self.logger.error("❌ Training completed but no metrics returned")
            return {
                'status': 'error',
                'error': 'No metrics returned from training',
                'experiment_name': self.experiment_name
            }

def run_quick_test():
    """Run a quick test to verify everything works"""
    
    experiment_name = f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(experiment_name, "INFO")
    
    logger.info("🧪 RUNNING QUICK SYSTEM TEST")
    logger.info("="*50)
    
    trainer = FixedTrainer(experiment_name, logger)
    
    # Run with minimal parameters for quick test
    results = trainer.run_training(
        batch_size=8,      # Small batch for speed
        learning_rate=1e-4,
        epochs=3,          # Just 3 epochs for testing
        backbone_name="tf_efficientnet_b0_ns"
    )
    
    return results

def run_full_training(experiment_name: Optional[str] = None,
                     batch_size: int = 16,
                     learning_rate: float = 1e-4,
                     epochs: int = 15,
                     backbone_name: str = "tf_efficientnet_b0_ns"):
    """Run full training with specified parameters"""
    
    if experiment_name is None:
        experiment_name = f"full_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger = setup_logging(experiment_name, "INFO")
    trainer = FixedTrainer(experiment_name, logger)
    
    results = trainer.run_training(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        backbone_name=backbone_name
    )
    
    return results

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Fixed Training Script for Diabetic Retinopathy Detection'
    )
    
    parser.add_argument('--mode', type=str, choices=['test', 'train'], default='test',
                       help='Mode: test (quick verification) or train (full training)')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate for optimizer')
    parser.add_argument('--backbone', type=str, default='tf_efficientnet_b0_ns',
                       choices=['tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns', 'resnet50'],
                       help='Model backbone architecture')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'test':
            print("🧪 Running quick system test...")
            results = run_quick_test()
        else:
            print("🚀 Running full training...")
            results = run_full_training(
                experiment_name=args.experiment_name,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                backbone_name=args.backbone
            )
        
        if results['status'] == 'success':
            print(f"\n🎉 {args.mode.title()} completed successfully!")
            print(f"📄 Experiment: {results['experiment_name']}")
            print(f"💾 Results: {results['experiment_dir']}")
            
            metrics = results['metrics']
            accuracy = metrics.get('accuracy', 0)
            dice = metrics.get('dice_score', 0)
            combined = metrics.get('combined_score', 0)
            
            print(f"🎯 Final Results:")
            print(f"   Accuracy: {accuracy:.4f} ({accuracy:.1%})")
            print(f"   Dice Score: {dice:.4f}")
            print(f"   Combined Score: {combined:.4f}")
            print(f"📈 {results.get('performance_level', 'Unknown performance level')}")
            
            return 0
        else:
            print(f"❌ {args.mode.title()} failed: {results['error']}")
            return 1
            
    except Exception as e:
        print(f"❌ Script execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
