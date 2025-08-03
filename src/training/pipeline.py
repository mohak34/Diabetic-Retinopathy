"""
Phase 4: Complete Training Pipeline Integration
Comprehensive training system that brings together all Phase 4 components
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
from torch.utils.data import DataLoader, Dataset
import yaml


class DummyDataset(Dataset):
    """Dummy dataset for smoke testing"""
    def __init__(self, size: int = 1000, num_classes: int = 5):
        self.size = size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Return dummy data matching expected format
        image = torch.randn(3, 512, 512)  # RGB image
        cls_label = torch.randint(0, self.num_classes, (1,)).long()
        seg_mask = torch.randint(0, 2, (512, 512)).float()  # Binary mask (H, W) with float values
        
        return image, cls_label.squeeze(), seg_mask


# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Local imports
from .config import AdvancedTrainingConfig, setup_reproducibility, create_experiment_directory
from .phase4_trainer import Phase4Trainer
from .hyperparameter_optimizer import (
    HyperparameterOptimizer, 
    HyperparameterSpace, 
    OptimizationConfig
)
from .metrics import AdvancedMetricsCollector
from ..models.multi_task_model import create_multi_task_model


class Phase4TrainingPipeline:
    """
    Phase 4: Complete Training Pipeline
    
    Orchestrates the entire training process with:
    - Advanced configuration management
    - Hyperparameter optimization
    - Progressive multi-task training
    - Comprehensive evaluation
    - Automated experiment tracking
    """
    
    def __init__(self, config_path: Optional[str] = None, logger: Optional[logging.Logger] = None, smoke_test: bool = False):
        self.config_path = config_path
        self.logger = logger or self._setup_logger()
        self.smoke_test = smoke_test
        
        # Load configuration
        if config_path:
            self.config = self._load_config_from_file(config_path)
        else:
            self.config = AdvancedTrainingConfig()
        
        # Initialize components
        self.trainer: Optional[Phase4Trainer] = None
        self.optimizer: Optional[HyperparameterOptimizer] = None
        
        self.logger.info("Phase 4 Training Pipeline initialized")
        self.logger.info(f"Configuration: {self.config.experiment_name}")
        if smoke_test:
            self.logger.info("Smoke test mode enabled")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging for the pipeline"""
        logger = logging.getLogger('Phase4Pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler with colored output
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_config_from_file(self, config_path: str) -> AdvancedTrainingConfig:
        """Load configuration from YAML file"""
        self.logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config from dictionary
        config = AdvancedTrainingConfig.from_dict(config_dict)
        
        self.logger.info("Configuration loaded successfully")
        return config
    
    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Setup data loaders for training, validation, and testing
        
        Note: This is a placeholder implementation. In a real project, you would:
        1. Load your diabetic retinopathy dataset
        2. Apply data transformations and augmentations
        3. Create train/val/test splits
        4. Return actual DataLoader objects
        """
        self.logger.info("Setting up data loaders...")
        
        # Placeholder implementation - replace with actual data loading
        # Create datasets
        train_dataset = DummyDataset(size=800, num_classes=self.config.model.num_classes)
        val_dataset = DummyDataset(size=200, num_classes=self.config.model.num_classes)
        test_dataset = DummyDataset(size=100, num_classes=self.config.model.num_classes)
        
        # Use single-threaded loading for smoke test to avoid multiprocessing issues
        num_workers = 0 if self.smoke_test else self.config.hardware.num_workers
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.hardware.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.hardware.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.hardware.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.logger.info(f"Data loaders created:")
        self.logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
        self.logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
        self.logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
        
        return train_loader, val_loader, test_loader
    
    def run_hyperparameter_optimization(self, 
                                       train_loader: DataLoader, 
                                       val_loader: DataLoader,
                                       param_space: Optional[HyperparameterSpace] = None,
                                       opt_config: Optional[OptimizationConfig] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        self.logger.info("=" * 80)
        self.logger.info("SEARCH STARTING HYPERPARAMETER OPTIMIZATION")
        self.logger.info("=" * 80)
        
        # Use default parameter space if not provided
        if param_space is None:
            param_space = HyperparameterSpace()
        
        # Use default optimization config if not provided
        if opt_config is None:
            opt_config = OptimizationConfig(
                study_name=f"{self.config.experiment_name}_optimization",
                search_method="optuna",
                n_trials=20,
                optimization_direction="maximize",
                primary_metric="val_combined_score"
            )
        
        # Create optimizer
        self.optimizer = HyperparameterOptimizer(
            base_config=self.config,
            param_space=param_space,
            optimization_config=opt_config,
            logger=self.logger
        )
        
        # Run optimization
        results = self.optimizer.run_optimization()
        
        # Update config with best parameters
        if results.get('best_params'):
            self._update_config_with_best_params(results['best_params'])
        
        return results
    
    def _update_config_with_best_params(self, best_params: Dict[str, Any]):
        """Update configuration with best hyperparameters"""
        self.logger.info("Updating configuration with best hyperparameters...")
        
        # Update model configuration
        if 'backbone_name' in best_params:
            self.config.model.backbone_name = best_params['backbone_name']
        
        # Update optimizer configuration
        if 'lr' in best_params:
            self.config.optimizer.lr = best_params['lr']
        if 'weight_decay' in best_params:
            self.config.optimizer.weight_decay = best_params['weight_decay']
        
        # Update batch size
        if 'batch_size' in best_params:
            self.config.hardware.batch_size = best_params['batch_size']
        
        # Update loss function parameters
        if 'focal_gamma' in best_params:
            self.config.focal_gamma = best_params['focal_gamma']
        if 'dice_smooth' in best_params:
            self.config.dice_smooth = best_params['dice_smooth']
        
        # Update training phases
        if 'phase1_epochs' in best_params:
            self.config.phase1.epochs = best_params['phase1_epochs']
        if 'phase2_epochs' in best_params:
            self.config.phase2.epochs = best_params['phase2_epochs']
        if 'phase3_epochs' in best_params:
            self.config.phase3.epochs = best_params['phase3_epochs']
        
        # Update total epochs
        self.config.total_epochs = (self.config.phase1.epochs + 
                                   self.config.phase2.epochs + 
                                   self.config.phase3.epochs)
        
        # Update segmentation weight progression
        if 'seg_weight_max' in best_params:
            self.config.segmentation_weight_max = best_params['seg_weight_max']
        if 'seg_weight_warmup' in best_params:
            self.config.segmentation_weight_warmup_epochs = best_params['seg_weight_warmup']
        
        # Update experiment name to reflect optimization
        self.config.experiment_name = f"{self.config.experiment_name}_optimized"
        
        self.logger.info("Configuration updated with optimized hyperparameters")
    
    def run_training(self, 
                    train_loader: DataLoader, 
                    val_loader: DataLoader,
                    resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Run the main training process"""
        
        self.logger.info("=" * 80)
        self.logger.info("START STARTING PHASE 4 TRAINING")
        self.logger.info("=" * 80)
        
        # Create trainer
        self.trainer = Phase4Trainer(
            config=self.config,
            logger=self.logger
        )
        
        # Run training
        training_results = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        self.logger.info("Training completed successfully!")
        return training_results
    
    def evaluate_model(self, 
                      test_loader: DataLoader,
                      checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION STARTING MODEL EVALUATION")
        self.logger.info("=" * 80)
        
        # Load model if trainer not available
        if self.trainer is None:
            self.trainer = Phase4Trainer(
                config=self.config,
                logger=self.logger
            )
            self.trainer.setup_model()
            
            # Load checkpoint if provided
            if checkpoint_path:
                self.trainer.load_checkpoint(checkpoint_path)
        
        # Set model to evaluation mode
        self.trainer.model.eval()
        
        # Initialize metrics collector
        metrics_collector = AdvancedMetricsCollector(
            num_classes=self.config.model.num_classes,
            device=str(self.trainer.device)
        )
        
        # Evaluation loop
        total_samples = 0
        
        self.logger.info("Running evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                # Handle different batch formats
                if len(batch_data) == 3:
                    images, cls_labels, seg_masks = batch_data
                else:
                    images, cls_labels = batch_data
                    seg_masks = None
                
                # Move to device
                images = images.to(self.trainer.device, non_blocking=True)
                cls_labels = cls_labels.to(self.trainer.device, non_blocking=True)
                if seg_masks is not None:
                    seg_masks = seg_masks.to(self.trainer.device, non_blocking=True)
                
                # Forward pass
                cls_logits, seg_preds = self.trainer.model(images)
                
                # Update metrics
                metrics_collector.update_classification(cls_logits, cls_labels)
                if seg_masks is not None:
                    metrics_collector.update_segmentation(seg_preds, seg_masks)
                
                total_samples += images.size(0)
                
                # Progress logging
                if batch_idx % 50 == 0:
                    self.logger.info(f"Evaluated {total_samples} samples...")
        
        # Compute final metrics
        final_metrics = metrics_collector.compute()
        
        # Generate comprehensive report
        evaluation_report = metrics_collector.generate_report()
        
        # Save evaluation results
        self._save_evaluation_results(final_metrics, evaluation_report)
        
        self.logger.info("Model evaluation completed!")
        self.logger.info(f"Evaluated {total_samples} samples")
        self.logger.info(f"\n{evaluation_report}")
        
        return {
            'metrics': final_metrics,
            'report': evaluation_report,
            'total_samples': total_samples
        }
    
    def _save_evaluation_results(self, metrics: Any, report: str):
        """Save evaluation results to files"""
        eval_dir = Path(self.config.experiment_dir) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = eval_dir / "evaluation_metrics.json"
        try:
            import json
            with open(metrics_file, 'w') as f:
                json.dump(metrics.__dict__, f, indent=2, default=str)
        except:
            self.logger.warning("Could not save metrics to JSON")
        
        # Save report
        report_file = eval_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Evaluation results saved to {eval_dir}")
    
    def run_complete_pipeline(self, 
                             enable_hyperparameter_optimization: bool = True,
                             enable_evaluation: bool = True,
                             resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete Phase 4 training pipeline"""
        
        self.logger.info("=" * 100)
        self.logger.info("TARGET PHASE 4: COMPLETE TRAINING PIPELINE STARTED")
        self.logger.info("=" * 100)
        self.logger.info(f"Experiment: {self.config.experiment_name}")
        self.logger.info(f"Configuration file: {self.config_path or 'Default'}")
        self.logger.info(f"Hyperparameter optimization: {enable_hyperparameter_optimization}")
        self.logger.info(f"Model evaluation: {enable_evaluation}")
        self.logger.info("=" * 100)
        
        pipeline_results = {}
        
        try:
            # 1. Setup data loaders
            train_loader, val_loader, test_loader = self.setup_data_loaders()
            
            # 2. Hyperparameter optimization (optional)
            if enable_hyperparameter_optimization:
                optimization_results = self.run_hyperparameter_optimization(
                    train_loader, val_loader
                )
                pipeline_results['optimization'] = optimization_results
            
            # 3. Main training
            training_results = self.run_training(
                train_loader, val_loader, resume_from_checkpoint
            )
            pipeline_results['training'] = training_results
            
            # 4. Model evaluation (optional)
            if enable_evaluation:
                # Use best model checkpoint
                best_checkpoint = None
                if self.trainer:
                    best_checkpoint = str(Path(self.config.checkpoints.save_dir) / "best_model.pth")
                
                evaluation_results = self.evaluate_model(test_loader, best_checkpoint)
                pipeline_results['evaluation'] = evaluation_results
            
            # 5. Generate final summary
            summary = self._generate_pipeline_summary(pipeline_results)
            pipeline_results['summary'] = summary
            
            self.logger.info("=" * 100)
            self.logger.info("SUCCESS PHASE 4 PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 100)
            self.logger.info(f"\n{summary}")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _generate_pipeline_summary(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive pipeline summary"""
        summary_lines = [
            "=" * 80,
            "SUMMARY PHASE 4 PIPELINE SUMMARY",
            "=" * 80
        ]
        
        # Experiment info
        summary_lines.extend([
            f"Experiment: {self.config.experiment_name}",
            f"Model: {self.config.model.backbone_name}",
            f"Total epochs: {self.config.total_epochs}",
            ""
        ])
        
        # Hyperparameter optimization results
        if 'optimization' in results:
            opt_results = results['optimization']
            summary_lines.extend([
                "SEARCH Hyperparameter Optimization:",
                f"  Method: {getattr(self.optimizer.opt_config, 'search_method', 'Unknown') if self.optimizer else 'Unknown'}",
                f"  Trials completed: {opt_results.get('n_trials', 0)}",
                f"  Best score: {opt_results.get('best_score', 0.0):.4f}",
                ""
            ])
        
        # Training results
        if 'training' in results:
            train_results = results['training']
            summary_lines.extend([
                "START Training Results:",
                f"  Total epochs: {train_results.get('total_epochs', 0)}",
                f"  Training time: {train_results.get('total_time_hours', 0.0):.2f} hours",
                f"  Best validation score: {train_results.get('best_metric_value', 0.0):.4f}",
                ""
            ])
        
        # Evaluation results
        if 'evaluation' in results:
            eval_results = results['evaluation']
            metrics = eval_results.get('metrics', {})
            
            # Extract key metrics
            cls_metrics = getattr(metrics, 'classification', {})
            seg_metrics = getattr(metrics, 'segmentation', {})
            combined_metrics = getattr(metrics, 'combined', {})
            
            summary_lines.extend([
                "EVALUATION Final Evaluation:",
                f"  Samples evaluated: {eval_results.get('total_samples', 0)}",
                f"  Classification accuracy: {cls_metrics.get('accuracy', 0.0):.3f}",
                f"  Classification kappa: {cls_metrics.get('kappa', 0.0):.3f}",
                f"  Segmentation Dice: {seg_metrics.get('dice', 0.0):.3f}",
                f"  Segmentation IoU: {seg_metrics.get('iou', 0.0):.3f}",
                f"  Combined score: {combined_metrics.get('combined_score', 0.0):.3f}",
                ""
            ])
        
        # Configuration summary
        summary_lines.extend([
            "Configuration  Configuration:",
            f"  Progressive training: {self.config.phase1.epochs}+{self.config.phase2.epochs}+{self.config.phase3.epochs} epochs",
            f"  Learning rate: {self.config.optimizer.lr:.2e}",
            f"  Batch size: {self.config.hardware.batch_size}",
            f"  Mixed precision: {self.config.mixed_precision}",
            ""
        ])
        
        # Paths
        summary_lines.extend([
            "📁 Output Paths:",
            f"  Experiment directory: {self.config.experiment_dir}",
            f"  Checkpoints: {self.config.checkpoints.save_dir}",
            f"  Logs: {self.config.logging.tensorboard_log_dir}",
            ""
        ])
        
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)
    
    def save_config(self, save_path: Optional[str] = None):
        """Save current configuration to file"""
        if save_path is None:
            save_path = Path(self.config.experiment_dir) / "final_config.yaml"
        
        config_dict = self.config.to_dict()
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, indent=2, default_flow_style=False)
        
        self.logger.info(f"Configuration saved to: {save_path}")


def main():
    """Main entry point for Phase 4 training pipeline"""
    parser = argparse.ArgumentParser(description="Phase 4: Diabetic Retinopathy Training Pipeline")
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--experiment-name', type=str, help='Override experiment name')
    
    # Pipeline options
    parser.add_argument('--skip-optimization', action='store_true', 
                       help='Skip hyperparameter optimization')
    parser.add_argument('--skip-evaluation', action='store_true', 
                       help='Skip final evaluation')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # Hardware options
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], 
                       default='auto', help='Device to use for training')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    
    # Optimization options
    parser.add_argument('--optimization-method', type=str, 
                       choices=['optuna', 'ray', 'grid'], default='optuna',
                       help='Hyperparameter optimization method')
    parser.add_argument('--optimization-trials', type=int, default=20,
                       help='Number of optimization trials')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create pipeline
        pipeline = Phase4TrainingPipeline(config_path=args.config)
        
        # Override configuration from command line arguments
        if args.experiment_name:
            pipeline.config.experiment_name = args.experiment_name
        
        if args.device != 'auto':
            pipeline.config.hardware.device = args.device
        
        if args.batch_size:
            pipeline.config.hardware.batch_size = args.batch_size
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            enable_hyperparameter_optimization=not args.skip_optimization,
            enable_evaluation=not args.skip_evaluation,
            resume_from_checkpoint=args.resume
        )
        
        # Save final configuration
        pipeline.save_config()
        
        print("\nSUCCESS Phase 4 training pipeline completed successfully!")
        print(f"Results saved to: {pipeline.config.experiment_dir}")
        
    except Exception as e:
        print(f"\nERROR Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
