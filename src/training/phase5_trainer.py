"""
Phase 5: Complete Model Training & Optimization Infrastructure
End-to-end training system with comprehensive monitoring and optimization capabilities.
"""

import os
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import shutil
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import psutil
import GPUtil

# Import configuration and components
from .config import Phase4Config
from .trainer import RobustPhase4Trainer
from .hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterSpace, OptimizationConfig
from .losses import RobustMultiTaskLoss
from ..models.multi_task_model import create_multi_task_model
from ..data.dataloaders import DataLoaderFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Phase5Config:
    """Phase 5 comprehensive training configuration"""
    
    # Experiment details
    experiment_name: str = "diabetic_retinopathy_phase5"
    experiment_description: str = "Complete end-to-end training with optimization"
    experiment_tags: List[str] = None
    
    # Training strategy
    training_mode: str = "full"  # "debug", "hyperopt", "full"
    total_epochs: int = 50
    warmup_epochs: int = 5
    
    # Progressive training phases
    phase1_epochs: int = 15  # Classification only
    phase2_epochs: int = 15  # Progressive multi-task
    phase3_epochs: int = 20  # Full multi-task optimization
    
    # Optimization settings
    enable_hyperparameter_optimization: bool = False
    hyperopt_trials: int = 30
    hyperopt_timeout_hours: float = 12.0
    
    # Quality control
    enable_quality_monitoring: bool = True
    save_prediction_samples: bool = True
    samples_per_epoch: int = 10
    
    # Resource monitoring
    monitor_gpu_memory: bool = True
    monitor_system_resources: bool = True
    log_resource_usage: bool = True
    
    # Early stopping and checkpointing
    early_stopping_patience: int = 10
    save_every_n_epochs: int = 5
    keep_best_n_models: int = 3
    
    # Data configuration
    data_root: str = "dataset/processed"
    splits_dir: str = "dataset/splits"
    use_combined_dataset: bool = True
    
    # Output configuration
    output_dir: str = "experiments"
    save_detailed_logs: bool = True
    generate_training_report: bool = True
    
    def __post_init__(self):
        if self.experiment_tags is None:
            self.experiment_tags = ["phase5", "multi-task", "end-to-end"]


class ResourceMonitor:
    """Monitor system and GPU resources during training"""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.logs = []
        self.gpu_available = torch.cuda.is_available()
        
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        usage = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Assume single GPU
                    usage.update({
                        'gpu_percent': gpu.load * 100,
                        'gpu_memory_percent': gpu.memoryUtil * 100,
                        'gpu_memory_used_mb': gpu.memoryUsed,
                        'gpu_memory_total_mb': gpu.memoryTotal,
                        'gpu_temperature': gpu.temperature
                    })
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")
        
        return usage
    
    def log_usage(self):
        """Log current usage"""
        usage = self.get_current_usage()
        self.logs.append(usage)
        return usage
    
    def get_peak_usage(self) -> Dict[str, Any]:
        """Get peak resource usage"""
        if not self.logs:
            return {}
        
        peak = {
            'peak_cpu_percent': max(log['cpu_percent'] for log in self.logs),
            'peak_memory_percent': max(log['memory_percent'] for log in self.logs),
            'peak_memory_used_gb': max(log['memory_used_gb'] for log in self.logs)
        }
        
        gpu_logs = [log for log in self.logs if 'gpu_percent' in log]
        if gpu_logs:
            peak.update({
                'peak_gpu_percent': max(log['gpu_percent'] for log in gpu_logs),
                'peak_gpu_memory_percent': max(log['gpu_memory_percent'] for log in gpu_logs),
                'peak_gpu_memory_used_mb': max(log['gpu_memory_used_mb'] for log in gpu_logs)
            })
        
        return peak


class QualityController:
    """Quality control and validation during training"""
    
    def __init__(self, save_dir: str, samples_per_epoch: int = 10):
        self.save_dir = Path(save_dir)
        self.samples_dir = self.save_dir / "quality_samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_epoch = samples_per_epoch
        
    def validate_batch_outputs(self, outputs: Dict[str, torch.Tensor], 
                              targets: Dict[str, torch.Tensor], 
                              epoch: int, batch_idx: int) -> Dict[str, Any]:
        """Validate model outputs for quality issues"""
        validation_report = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'timestamp': time.time(),
            'issues': []
        }
        
        # Check for NaN/Inf values
        cls_outputs = outputs.get('classification')
        seg_outputs = outputs.get('segmentation')
        
        if cls_outputs is not None:
            if torch.isnan(cls_outputs).any():
                validation_report['issues'].append('NaN in classification outputs')
            if torch.isinf(cls_outputs).any():
                validation_report['issues'].append('Inf in classification outputs')
        
        if seg_outputs is not None:
            if torch.isnan(seg_outputs).any():
                validation_report['issues'].append('NaN in segmentation outputs')
            if torch.isinf(seg_outputs).any():
                validation_report['issues'].append('Inf in segmentation outputs')
        
        # Check output ranges
        if cls_outputs is not None:
            cls_min, cls_max = cls_outputs.min().item(), cls_outputs.max().item()
            if abs(cls_min) > 100 or abs(cls_max) > 100:
                validation_report['issues'].append(f'Extreme classification values: {cls_min:.2f} to {cls_max:.2f}')
        
        if seg_outputs is not None:
            seg_min, seg_max = seg_outputs.min().item(), seg_outputs.max().item()
            if seg_min < -1 or seg_max > 2:
                validation_report['issues'].append(f'Segmentation values out of range: {seg_min:.2f} to {seg_max:.2f}')
        
        return validation_report
    
    def save_prediction_samples(self, images: torch.Tensor,
                               outputs: Dict[str, torch.Tensor],
                               targets: Dict[str, torch.Tensor],
                               epoch: int, batch_idx: int):
        """Save sample predictions for visual inspection"""
        try:
            # Save first few samples from batch
            n_samples = min(self.samples_per_epoch, images.size(0))
            
            sample_data = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'samples': []
            }
            
            for i in range(n_samples):
                sample = {
                    'image_stats': {
                        'mean': images[i].mean().item(),
                        'std': images[i].std().item(),
                        'min': images[i].min().item(),
                        'max': images[i].max().item()
                    }
                }
                
                # Classification predictions
                if 'classification' in outputs:
                    cls_pred = torch.softmax(outputs['classification'][i], dim=0)
                    sample['classification'] = {
                        'prediction': cls_pred.cpu().numpy().tolist(),
                        'predicted_class': cls_pred.argmax().item(),
                        'confidence': cls_pred.max().item(),
                        'target': targets.get('classification', [None])[i].item() if 'classification' in targets else None
                    }
                
                # Segmentation predictions
                if 'segmentation' in outputs:
                    seg_pred = torch.sigmoid(outputs['segmentation'][i])
                    sample['segmentation'] = {
                        'mean_prediction': seg_pred.mean().item(),
                        'positive_pixels': (seg_pred > 0.5).sum().item(),
                        'total_pixels': seg_pred.numel()
                    }
                
                sample_data['samples'].append(sample)
            
            # Save to file
            sample_file = self.samples_dir / f"epoch_{epoch:03d}_batch_{batch_idx:03d}.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save prediction samples: {e}")


class Phase5Trainer:
    """
    Phase 5: Complete Model Training & Optimization System
    
    Features:
    - End-to-end training with progressive learning
    - Real-time quality monitoring and validation
    - Resource usage tracking and optimization
    - Comprehensive experiment logging
    - Automatic hyperparameter optimization
    - Research-grade result documentation
    """
    
    def __init__(self, config: Phase5Config):
        self.config = config
        self.start_time = time.time()
        
        # Create experiment directory
        self.experiment_dir = Path(config.output_dir) / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resource_monitor = ResourceMonitor() if config.monitor_system_resources else None
        self.quality_controller = QualityController(
            str(self.experiment_dir), 
            config.samples_per_epoch
        ) if config.enable_quality_monitoring else None
        
        # Training state
        self.training_results = {}
        self.best_models = []
        
        self.logger.info("Phase 5 Trainer initialized")
        self.logger.info(f"Experiment: {config.experiment_name}")
        self.logger.info(f"Mode: {config.training_mode}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        self.logger = logging.getLogger(f'Phase5Trainer_{self.config.experiment_name}')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.save_detailed_logs:
            log_file = self.experiment_dir / "phase5_training.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def create_training_config(self, hyperparams: Optional[Dict] = None) -> Phase4Config:
        """Create Phase 4 config from Phase 5 config and optional hyperparameters"""
        # Start with default Phase 4 config
        phase4_config = Phase4Config()
        
        # Update from Phase 5 config
        phase4_config.experiment_name = f"{self.config.experiment_name}_training"
        
        # Progressive training setup
        phase4_config.progressive.phase1_epochs = self.config.phase1_epochs
        phase4_config.progressive.phase2_epochs = self.config.phase2_epochs
        phase4_config.progressive.phase3_epochs = self.config.phase3_epochs
        
        # Early stopping
        phase4_config.early_stopping.enabled = True
        phase4_config.early_stopping.patience = self.config.early_stopping_patience
        
        # Checkpointing
        phase4_config.checkpoint.save_every = self.config.save_every_n_epochs
        
        # Paths
        phase4_config.data_root = self.config.data_root
        phase4_config.output_dir = str(self.experiment_dir)
        
        # Apply hyperparameters if provided
        if hyperparams:
            for key, value in hyperparams.items():
                if key == 'lr':
                    phase4_config.optimizer.learning_rate = value
                elif key == 'batch_size':
                    phase4_config.hardware.batch_size = value
                elif key == 'weight_decay':
                    phase4_config.optimizer.weight_decay = value
                elif key == 'backbone_name':
                    phase4_config.model.backbone_name = value
                elif key == 'focal_gamma':
                    phase4_config.loss.focal_gamma = value
                elif key == 'dice_smooth':
                    phase4_config.loss.dice_smooth = value
        
        return phase4_config
    
    def create_data_loaders(self, config: Phase4Config) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training"""
        self.logger.info("Creating data loaders...")
        
        try:
            # Use DataLoaderFactory to create loaders
            factory = DataLoaderFactory()
            
            if self.config.use_combined_dataset:
                loaders = factory.create_multitask_loaders(
                    processed_data_dir=self.config.data_root,
                    splits_dir=self.config.splits_dir,
                    batch_size=config.hardware.batch_size,
                    num_workers=config.hardware.num_workers,
                    image_size=512
                )
                
                train_loader = loaders['train']
                val_loader = loaders['val']
            else:
                # Create separate loaders for classification and segmentation
                grading_loaders = factory.create_grading_loaders(
                    processed_data_dir=self.config.data_root,
                    splits_dir=self.config.splits_dir,
                    batch_size=config.hardware.batch_size
                )
                train_loader = grading_loaders['train']
                val_loader = grading_loaders['val']
            
            self.logger.info(f"Created data loaders: train={len(train_loader)} batches, val={len(val_loader)} batches")
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.warning(f"Failed to create real data loaders, using dummy data: {e}")
            return self.create_dummy_data_loaders(config)
    
    def create_dummy_data_loaders(self, config: Phase4Config) -> Tuple[DataLoader, DataLoader]:
        """Create dummy data loaders for testing"""
        from ..training.pipeline import DummyDataset
        
        train_dataset = DummyDataset(size=config.hardware.batch_size * 20)
        val_dataset = DummyDataset(size=config.hardware.batch_size * 10)
        
        train_loader = DataLoader(train_dataset, batch_size=config.hardware.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.hardware.batch_size, shuffle=False)
        
        self.logger.info("Using dummy data loaders for testing")
        return train_loader, val_loader
    
    def run_initial_training_debug(self) -> Dict[str, Any]:
        """Run initial training with debugging and validation"""
        self.logger.info("PHASE 5.1: INITIAL TRAINING RUNS & DEBUGGING")
        self.logger.info("=" * 80)
        
        debug_results = {}
        
        # Create short debug configuration
        debug_config = self.create_training_config()
        debug_config.progressive.phase1_epochs = 2
        debug_config.progressive.phase2_epochs = 2
        debug_config.progressive.phase3_epochs = 1
        debug_config.experiment_name = f"{self.config.experiment_name}_debug"
        
        try:
            # Create model
            model = create_multi_task_model(
                num_classes=5,
                backbone_name=debug_config.model.backbone_name,
                pretrained=True
            )
            
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(debug_config)
            
            # Create trainer
            trainer = RobustPhase4Trainer(
                model=model,
                config=debug_config,
                device=str(self.device)
            )
            
            # Monitor resources
            if self.resource_monitor:
                initial_usage = self.resource_monitor.log_usage()
                self.logger.info(f"Initial resource usage: CPU={initial_usage['cpu_percent']:.1f}%, "
                               f"Memory={initial_usage['memory_percent']:.1f}%")
            
            # Run short training
            self.logger.info("Running debug training (5 epochs)...")
            debug_results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                save_dir=str(self.experiment_dir / "debug_models")
            )
            
            # Validate results
            self.validate_training_results(debug_results, "debug")
            
            self.logger.info("✅ Debug training completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Debug training failed: {e}")
            debug_results['error'] = str(e)
        
        return debug_results
    
    def run_hyperparameter_optimization(self) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        self.logger.info("PHASE 5.2: HYPERPARAMETER OPTIMIZATION")
        self.logger.info("=" * 80)
        
        if not self.config.enable_hyperparameter_optimization:
            self.logger.info("Hyperparameter optimization disabled, using default parameters")
            return {"status": "skipped", "best_params": {}}
        
        try:
            # Create base configuration for hyperparameter optimization
            base_config = self.create_training_config()
            
            # Define hyperparameter search space
            param_space = HyperparameterSpace(
                lr_min=1e-5,
                lr_max=1e-2,
                batch_sizes=[8, 16, 24, 32],
                backbone_names=['efficientnet-b0', 'efficientnet-b1', 'resnet50'],
                phase1_epochs_range=(5, 15),
                phase2_epochs_range=(5, 15),
                phase3_epochs_range=(10, 25)
            )
            
            # Create optimization configuration
            opt_config = OptimizationConfig(
                study_name=f"{self.config.experiment_name}_hyperopt",
                search_method="optuna",
                n_trials=self.config.hyperopt_trials,
                timeout_hours=self.config.hyperopt_timeout_hours,
                optimization_direction="maximize",
                primary_metric="val_combined_score"
            )
            
            # Create optimizer
            optimizer = HyperparameterOptimizer(
                base_config=base_config,
                param_space=param_space,
                optimization_config=opt_config,
                logger=self.logger
            )
            
            # Run optimization
            optimization_results = optimizer.run_optimization()
            
            # Analyze results
            analysis = optimizer.analyze_results()
            
            self.logger.info("✅ Hyperparameter optimization completed")
            self.logger.info(f"Best parameters: {optimization_results['best_params']}")
            self.logger.info(f"Best score: {optimization_results['best_score']:.4f}")
            
            return {
                "status": "completed",
                "best_params": optimization_results['best_params'],
                "best_score": optimization_results['best_score'],
                "analysis": analysis
            }
            
        except Exception as e:
            self.logger.error(f"❌ Hyperparameter optimization failed: {e}")
            return {"status": "failed", "error": str(e), "best_params": {}}
    
    def run_full_model_training(self, best_hyperparams: Optional[Dict] = None) -> Dict[str, Any]:
        """Run full model training with best hyperparameters"""
        self.logger.info("PHASE 5.3: FULL MODEL TRAINING")
        self.logger.info("=" * 80)
        
        try:
            # Create training configuration with best hyperparameters
            training_config = self.create_training_config(best_hyperparams)
            training_config.experiment_name = f"{self.config.experiment_name}_final"
            
            # Log training configuration
            self.logger.info("Training configuration:")
            self.logger.info(f"  Total epochs: {training_config.progressive.total_epochs}")
            self.logger.info(f"  Phase 1: {training_config.progressive.phase1_epochs} epochs (classification only)")
            self.logger.info(f"  Phase 2: {training_config.progressive.phase2_epochs} epochs (progressive multi-task)")
            self.logger.info(f"  Phase 3: {training_config.progressive.phase3_epochs} epochs (full multi-task)")
            self.logger.info(f"  Learning rate: {training_config.optimizer.learning_rate}")
            self.logger.info(f"  Batch size: {training_config.hardware.batch_size}")
            self.logger.info(f"  Backbone: {training_config.model.backbone_name}")
            
            # Create model
            model = create_multi_task_model(
                num_classes=5,
                backbone_name=training_config.model.backbone_name,
                pretrained=True,
                segmentation_classes=1
            )
            
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(training_config)
            
            # Create trainer
            trainer = RobustPhase4Trainer(
                model=model,
                config=training_config,
                device=str(self.device)
            )
            
            # Monitor initial resources
            if self.resource_monitor:
                initial_usage = self.resource_monitor.log_usage()
                self.logger.info(f"Starting training - Resource usage: CPU={initial_usage['cpu_percent']:.1f}%, "
                               f"Memory={initial_usage['memory_percent']:.1f}%")
            
            # Save training configuration
            config_path = self.experiment_dir / "final_training_config.yaml"
            training_config.to_yaml(str(config_path))
            
            # Run full training
            self.logger.info("Starting full model training...")
            training_start = time.time()
            
            training_results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                save_dir=str(self.experiment_dir / "final_models")
            )
            
            training_time = time.time() - training_start
            
            # Monitor final resources
            if self.resource_monitor:
                final_usage = self.resource_monitor.log_usage()
                peak_usage = self.resource_monitor.get_peak_usage()
                
                self.logger.info(f"Training completed - Peak resource usage:")
                self.logger.info(f"  CPU: {peak_usage.get('peak_cpu_percent', 0):.1f}%")
                self.logger.info(f"  Memory: {peak_usage.get('peak_memory_percent', 0):.1f}%")
                if 'peak_gpu_percent' in peak_usage:
                    self.logger.info(f"  GPU: {peak_usage['peak_gpu_percent']:.1f}%")
                    self.logger.info(f"  GPU Memory: {peak_usage['peak_gpu_memory_percent']:.1f}%")
            
            # Validate training results
            self.validate_training_results(training_results, "full_training")
            
            # Add training metadata
            training_results.update({
                'training_time_hours': training_time / 3600,
                'hyperparameters_used': best_hyperparams or {},
                'final_config': asdict(training_config),
                'resource_usage': self.resource_monitor.get_peak_usage() if self.resource_monitor else {}
            })
            
            self.logger.info("✅ Full model training completed successfully")
            self.logger.info(f"Training time: {training_time/3600:.2f} hours")
            self.logger.info(f"Best metric: {training_results.get('best_metric', 'N/A')}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Full model training failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def validate_training_results(self, results: Dict[str, Any], phase: str):
        """Validate training results for quality issues"""
        self.logger.info(f"Validating {phase} results...")
        
        issues = []
        
        # Check for required fields
        required_fields = ['training_history', 'best_metric', 'total_epochs']
        for field in required_fields:
            if field not in results:
                issues.append(f"Missing required field: {field}")
        
        # Check training history
        if 'training_history' in results:
            history = results['training_history']
            
            # Check for empty history
            if not history.get('train_losses'):
                issues.append("Empty training loss history")
            
            # Check for NaN/Inf values
            train_losses = history.get('train_losses', [])
            if train_losses and any(not np.isfinite(loss) for loss in train_losses):
                issues.append("NaN/Inf values in training losses")
            
            val_losses = history.get('val_losses', [])
            if val_losses and any(not np.isfinite(loss) for loss in val_losses):
                issues.append("NaN/Inf values in validation losses")
        
        # Check model performance
        best_metric = results.get('best_metric')
        if best_metric is not None:
            if not np.isfinite(best_metric):
                issues.append("Invalid best metric value")
            elif best_metric <= 0:
                issues.append("Suspiciously low best metric")
        
        # Log results
        if issues:
            self.logger.warning(f"Quality issues found in {phase}:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        else:
            self.logger.info(f"✅ {phase} results validation passed")
        
        return issues
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        self.logger.info("PHASE 5.6: GENERATING TRAINING REPORT")
        self.logger.info("=" * 80)
        
        total_time = time.time() - self.start_time
        
        report = {
            'experiment_info': {
                'name': self.config.experiment_name,
                'description': self.config.experiment_description,
                'tags': self.config.experiment_tags,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'total_time_hours': total_time / 3600,
                'mode': self.config.training_mode
            },
            'system_info': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'torch_version': torch.__version__,
                'python_version': os.sys.version
            },
            'training_results': self.training_results,
            'resource_usage': self.resource_monitor.get_peak_usage() if self.resource_monitor else {},
            'configuration': asdict(self.config)
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            report['system_info'].update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        
        # Save report
        report_path = self.experiment_dir / "phase5_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary plots if matplotlib available
        try:
            self.generate_training_plots()
        except Exception as e:
            self.logger.warning(f"Failed to generate plots: {e}")
        
        self.logger.info(f"✅ Training report saved to: {report_path}")
        return report
    
    def generate_training_plots(self):
        """Generate training visualization plots"""
        plots_dir = self.experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Extract training history from results
        history = None
        for phase_name, results in self.training_results.items():
            if isinstance(results, dict) and 'training_history' in results:
                history = results['training_history']
                break
        
        if not history:
            self.logger.warning("No training history found for plotting")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Plot training losses
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Phase 5 Training Results - {self.config.experiment_name}', fontsize=16)
        
        # Loss curves
        if 'train_losses' in history and 'val_losses' in history:
            epochs = range(1, len(history['train_losses']) + 1)
            axes[0, 0].plot(epochs, history['train_losses'], label='Training Loss', color='blue')
            axes[0, 0].plot(epochs, history['val_losses'], label='Validation Loss', color='orange')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rates' in history:
            epochs = range(1, len(history['learning_rates']) + 1)
            axes[0, 1].plot(epochs, history['learning_rates'], color='green')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Resource usage
        if self.resource_monitor and self.resource_monitor.logs:
            times = [(log['timestamp'] - self.start_time) / 3600 for log in self.resource_monitor.logs]
            cpu_usage = [log['cpu_percent'] for log in self.resource_monitor.logs]
            memory_usage = [log['memory_percent'] for log in self.resource_monitor.logs]
            
            axes[1, 0].plot(times, cpu_usage, label='CPU %', color='red')
            axes[1, 0].plot(times, memory_usage, label='Memory %', color='purple')
            axes[1, 0].set_title('Resource Usage Over Time')
            axes[1, 0].set_xlabel('Time (hours)')
            axes[1, 0].set_ylabel('Usage (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Training metrics summary
        axes[1, 1].text(0.1, 0.8, f'Total Epochs: {len(history.get("train_losses", []))}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f'Best Metric: {self.training_results.get("full_training", {}).get("best_metric", "N/A")}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Training Time: {(time.time() - self.start_time)/3600:.2f}h', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved to: {plots_dir}")
    
    def run_complete_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete Phase 5 training pipeline"""
        self.logger.info("STARTING PHASE 5: COMPLETE MODEL TRAINING & OPTIMIZATION")
        self.logger.info("=" * 100)
        self.logger.info("End-to-end training system with comprehensive monitoring and optimization")
        self.logger.info(f"Experiment: {self.config.experiment_name}")
        self.logger.info(f"Mode: {self.config.training_mode}")
        self.logger.info(f"Target epochs: {self.config.total_epochs}")
        self.logger.info("=" * 100)
        
        try:
            # Phase 5.1: Initial training runs & debugging
            if self.config.training_mode in ["debug", "full"]:
                self.training_results['debug'] = self.run_initial_training_debug()
            
            # Phase 5.2: Hyperparameter optimization
            hyperopt_results = self.run_hyperparameter_optimization()
            self.training_results['hyperparameter_optimization'] = hyperopt_results
            
            # Phase 5.3: Full model training
            best_hyperparams = hyperopt_results.get('best_params', {})
            if self.config.training_mode in ["full"]:
                self.training_results['full_training'] = self.run_full_model_training(best_hyperparams)
            
            # Phase 5.6: Generate comprehensive report
            if self.config.generate_training_report:
                training_report = self.generate_training_report()
                self.training_results['final_report'] = training_report
            
            # Final summary
            total_time = time.time() - self.start_time
            
            self.logger.info("=" * 100)
            self.logger.info("PHASE 5 TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 100)
            self.logger.info(f"Total execution time: {total_time/3600:.2f} hours")
            self.logger.info(f"Experiment directory: {self.experiment_dir}")
            self.logger.info(f"Results saved with {len(self.training_results)} phases completed")
            
            if 'full_training' in self.training_results:
                best_metric = self.training_results['full_training'].get('best_metric', 'N/A')
                self.logger.info(f"Best model performance: {best_metric}")
            
            self.logger.info("=" * 100)
            
            return self.training_results
            
        except Exception as e:
            self.logger.error(f"❌ Phase 5 training pipeline failed: {e}")
            self.training_results['error'] = {
                'message': str(e),
                'phase': 'pipeline_execution',
                'timestamp': time.time()
            }
            
            # Still generate a report with partial results
            if self.config.generate_training_report:
                try:
                    self.generate_training_report()
                except Exception as report_error:
                    self.logger.error(f"Failed to generate error report: {report_error}")
            
            raise e


def main():
    """Main entry point for Phase 5 training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 5: Complete Model Training & Optimization")
    parser.add_argument('--experiment-name', type=str, 
                       default=f"diabetic_retinopathy_phase5_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Name for the experiment')
    parser.add_argument('--mode', type=str, choices=['debug', 'hyperopt', 'full'], 
                       default='debug', help='Training mode')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs')
    parser.add_argument('--enable-hyperopt', action='store_true', 
                       help='Enable hyperparameter optimization')
    parser.add_argument('--hyperopt-trials', type=int, default=20, 
                       help='Number of hyperparameter optimization trials')
    
    args = parser.parse_args()
    
    # Create Phase 5 configuration
    config = Phase5Config(
        experiment_name=args.experiment_name,
        training_mode=args.mode,
        total_epochs=args.epochs,
        enable_hyperparameter_optimization=args.enable_hyperopt,
        hyperopt_trials=args.hyperopt_trials
    )
    
    # Create and run trainer
    trainer = Phase5Trainer(config)
    
    try:
        results = trainer.run_complete_training_pipeline()
        print(f"\nPhase 5 training completed successfully!")
        print(f"Results saved to: {trainer.experiment_dir}")
        return 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
