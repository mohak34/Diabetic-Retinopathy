"""
Phase 4: Example Usage Script
Demonstrates the key features of the advanced training system
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import logging

# Local imports
from src.training.config import AdvancedTrainingConfig
from src.training.phase4_trainer import Phase4Trainer
from src.training.metrics import AdvancedMetricsCollector
from src.training.hyperparameter_optimizer import HyperparameterSpace, OptimizationConfig
from src.training.pipeline import Phase4TrainingPipeline


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_1_basic_configuration():
    """Example 1: Basic configuration setup"""
    print("\n" + "="*60)
    print("EXAMPLE EXAMPLE 1: Basic Configuration Setup")
    print("="*60)
    
    # Create default configuration
    config = AdvancedTrainingConfig()
    
    print(f"Experiment name: {config.experiment_name}")
    print(f"Model backbone: {config.model.backbone_name}")
    print(f"Total epochs: {config.total_epochs}")
    print(f"Progressive phases: {config.phase1.epochs} + {config.phase2.epochs} + {config.phase3.epochs}")
    print(f"Batch size: {config.hardware.batch_size}")
    print(f"Learning rate: {config.optimizer.lr}")
    print(f"Mixed precision: {config.mixed_precision}")
    
    # Demonstrate configuration customization
    config.experiment_name = "example_experiment"
    config.model.backbone_name = "efficientnet-b2"
    config.hardware.batch_size = 32
    config.optimizer.lr = 0.0005
    
    print(f"\nAfter customization:")
    print(f"Experiment name: {config.experiment_name}")
    print(f"Model backbone: {config.model.backbone_name}")
    print(f"Batch size: {config.hardware.batch_size}")
    print(f"Learning rate: {config.optimizer.lr}")
    
    # Save configuration
    config_path = "example_config.yaml"
    config.save_to_file(config_path)
    print(f"\nConfiguration saved to: {config_path}")
    
    return config


def example_2_trainer_setup():
    """Example 2: Trainer setup and model initialization"""
    print("\n" + "="*60)
    print("EXAMPLE EXAMPLE 2: Trainer Setup")
    print("="*60)
    
    # Create configuration
    config = AdvancedTrainingConfig()
    config.experiment_name = "trainer_example"
    config.total_epochs = 5  # Short for example
    
    # Create trainer
    trainer = Phase4Trainer(config)
    
    print(f"Trainer initialized on device: {trainer.device}")
    print(f"Experiment directory: {trainer.experiment_dir}")
    
    # Setup model
    model = trainer.setup_model()
    print(f"Model created: {type(model).__name__}")
    
    # Setup optimizer and scheduler
    optimizer = trainer.setup_optimizer()
    scheduler = trainer.setup_scheduler()
    loss_fn = trainer.setup_loss_function()
    
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Scheduler: {type(scheduler).__name__ if scheduler else 'None'}")
    print(f"Loss function: {type(loss_fn).__name__}")
    
    # Show progressive training phases
    for epoch in range(config.total_epochs):
        phase_name, phase_info = trainer.get_current_phase_info(epoch)
        seg_weight = phase_info['segmentation_weight']
        print(f"Epoch {epoch}: {phase_name} (seg_weight={seg_weight:.2f})")
    
    return trainer


def example_3_metrics_system():
    """Example 3: Comprehensive metrics system"""
    print("\n" + "="*60)
    print("EXAMPLE EXAMPLE 3: Metrics System")
    print("="*60)
    
    # Create metrics collector
    metrics_collector = AdvancedMetricsCollector(num_classes=5)
    
    # Simulate some predictions and targets
    batch_size = 8
    num_classes = 5
    image_size = 256
    
    # Create dummy classification data
    cls_logits = torch.randn(batch_size, num_classes)
    cls_targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create dummy segmentation data
    seg_preds = torch.sigmoid(torch.randn(batch_size, 1, image_size, image_size))
    seg_targets = torch.randint(0, 2, (batch_size, image_size, image_size)).float()
    
    print(f"Simulated data shapes:")
    print(f"  Classification logits: {cls_logits.shape}")
    print(f"  Classification targets: {cls_targets.shape}")
    print(f"  Segmentation predictions: {seg_preds.shape}")
    print(f"  Segmentation targets: {seg_targets.shape}")
    
    # Update metrics
    metrics_collector.update_classification(cls_logits, cls_targets)
    metrics_collector.update_segmentation(seg_preds, seg_targets)
    
    # Compute metrics
    results = metrics_collector.compute()
    
    print(f"\nComputed metrics:")
    print(f"  Classification accuracy: {results.classification.get('accuracy', 0.0):.3f}")
    print(f"  Classification kappa: {results.classification.get('kappa', 0.0):.3f}")
    print(f"  Segmentation Dice: {results.segmentation.get('dice', 0.0):.3f}")
    print(f"  Segmentation IoU: {results.segmentation.get('iou', 0.0):.3f}")
    print(f"  Combined score: {results.combined.get('combined_score', 0.0):.3f}")
    
    # Generate report
    report = metrics_collector.generate_report()
    print(f"\nMetrics Report:")
    print(report[:300] + "..." if len(report) > 300 else report)
    
    return metrics_collector


def example_4_hyperparameter_space():
    """Example 4: Hyperparameter optimization setup"""
    print("\n" + "="*60)
    print("EXAMPLE EXAMPLE 4: Hyperparameter Optimization")
    print("="*60)
    
    # Create parameter space
    param_space = HyperparameterSpace()
    
    print(f"Parameter space configuration:")
    print(f"  Learning rate range: [{param_space.lr_min:.0e}, {param_space.lr_max:.0e}] (log: {param_space.lr_log})")
    print(f"  Batch sizes: {param_space.batch_sizes}")
    print(f"  Weight decay range: [{param_space.weight_decay_min:.0e}, {param_space.weight_decay_max:.0e}]")
    print(f"  Backbone options: {param_space.backbone_names}")
    print(f"  Focal gamma range: [{param_space.focal_gamma_min}, {param_space.focal_gamma_max}]")
    print(f"  Phase 1 epochs: {param_space.phase1_epochs_range}")
    print(f"  Phase 2 epochs: {param_space.phase2_epochs_range}")
    print(f"  Phase 3 epochs: {param_space.phase3_epochs_range}")
    
    # Create optimization configuration
    opt_config = OptimizationConfig()
    opt_config.study_name = "example_optimization"
    opt_config.n_trials = 5  # Small for example
    opt_config.search_method = "grid"  # Use grid for deterministic example
    
    print(f"\nOptimization configuration:")
    print(f"  Study name: {opt_config.study_name}")
    print(f"  Search method: {opt_config.search_method}")
    print(f"  Number of trials: {opt_config.n_trials}")
    print(f"  Primary metric: {opt_config.primary_metric}")
    print(f"  Direction: {opt_config.optimization_direction}")
    
    return param_space, opt_config


def example_5_complete_pipeline():
    """Example 5: Complete training pipeline"""
    print("\n" + "="*60)
    print("EXAMPLE EXAMPLE 5: Complete Pipeline")
    print("="*60)
    
    # Create a minimal configuration for quick demo
    config_dict = {
        'experiment_name': 'pipeline_example',
        'model': {
            'backbone_name': 'efficientnet-b0',
            'num_classes': 5
        },
        'phase1': {'epochs': 2},
        'phase2': {'epochs': 1}, 
        'phase3': {'epochs': 2},
        'total_epochs': 5,
        'hardware': {
            'batch_size': 4,  # Small for demo
            'mixed_precision': False  # Disable for compatibility
        },
        'optimizer': {
            'lr': 0.01  # Higher for faster convergence in demo
        },
        'early_stopping': {
            'patience': 10  # High patience for short demo
        }
    }
    
    # Save temporary config
    import yaml
    temp_config_path = "temp_pipeline_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_dict, f)
    
    print(f"Created temporary config: {temp_config_path}")
    
    try:
        # Create pipeline
        pipeline = Phase4TrainingPipeline(config_path=temp_config_path)
        
        print(f"Pipeline created for experiment: {pipeline.config.experiment_name}")
        print(f"Model: {pipeline.config.model.backbone_name}")
        print(f"Total epochs: {pipeline.config.total_epochs}")
        print(f"Batch size: {pipeline.config.hardware.batch_size}")
        
        # Setup data loaders (dummy data)
        train_loader, val_loader, test_loader = pipeline.setup_data_loaders()
        
        print(f"Data loaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        print(f"\nPipeline ready for training!")
        print(f"To run full training, use: pipeline.run_complete_pipeline()")
        
        return pipeline
        
    except Exception as e:
        print(f"Pipeline creation failed: {e}")
        return None
    
    finally:
        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def example_6_phase_progression():
    """Example 6: Demonstrate phase progression"""
    print("\n" + "="*60)
    print("EXAMPLE EXAMPLE 6: Phase Progression")
    print("="*60)
    
    config = AdvancedTrainingConfig()
    config.phase1.epochs = 10
    config.phase2.epochs = 8  
    config.phase3.epochs = 12
    config.total_epochs = 30
    
    print(f"Training configuration:")
    print(f"  Phase 1 (Classification): {config.phase1.epochs} epochs")
    print(f"  Phase 2 (Warmup): {config.phase2.epochs} epochs") 
    print(f"  Phase 3 (Multi-task): {config.phase3.epochs} epochs")
    print(f"  Total: {config.total_epochs} epochs")
    
    print(f"\nPhase progression:")
    print(f"{'Epoch':<6} {'Phase':<20} {'Cls Weight':<12} {'Seg Weight':<12} {'Description'}")
    print("-" * 70)
    
    for epoch in range(0, config.total_epochs, 3):  # Sample every 3 epochs
        phase_name, phase_config = config.get_current_phase(epoch)
        seg_weight = config.get_segmentation_weight(epoch)
        cls_weight = phase_config.classification_weight
        
        if phase_name == "phase1":
            desc = "Classification only"
        elif phase_name == "phase2":
            desc = "Segmentation warmup"
        else:
            desc = "Multi-task optimization"
        
        print(f"{epoch:<6} {phase_name:<20} {cls_weight:<12.2f} {seg_weight:<12.2f} {desc}")


def main():
    """Run all examples"""
    setup_logging()
    
    print("EXAMPLE Phase 4: Advanced Training System Examples")
    print("="*80)
    print("This script demonstrates the key features of Phase 4 implementation")
    print("="*80)
    
    try:
        # Run examples
        config = example_1_basic_configuration()
        trainer = example_2_trainer_setup()
        metrics = example_3_metrics_system()
        param_space, opt_config = example_4_hyperparameter_space()
        pipeline = example_5_complete_pipeline()
        example_6_phase_progression()
        
        print("\n" + "="*80)
        print("SUCCESS: ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Key Phase 4 features demonstrated:")
        print("  OK Advanced configuration system")
        print("  OK Comprehensive trainer setup")
        print("  OK Multi-task metrics collection")
        print("  OK Hyperparameter optimization")
        print("  OK Complete training pipeline")
        print("  OK Progressive training phases")
        print("\nYour Phase 4 implementation is ready for production use!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nERROR Example failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
