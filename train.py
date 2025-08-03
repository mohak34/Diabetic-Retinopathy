"""
Phase 4: Diabetic Retinopathy Training Script
Complete training pipeline with all advanced features
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Imports
import torch
import torch.multiprocessing as mp

from src.training.pipeline import Phase4TrainingPipeline
from src.training.config import AdvancedTrainingConfig
from src.training.hyperparameter_optimizer import HyperparameterSpace, OptimizationConfig


def setup_logging(log_level: str = "INFO"):
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def check_environment():
    """Check if the environment is properly set up"""
    logger = logging.getLogger(__name__)
    
    # Check PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_gb = gpu_props.total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_props.name} ({memory_gb:.1f}GB)")
    else:
        logger.warning("CUDA not available - using CPU")
    
    # Check memory
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory_gb < 8:
            logger.warning(f"GPU memory ({memory_gb:.1f}GB) may be insufficient for training")
            logger.warning("Consider reducing batch size or enabling mixed precision")
    
    return True


def create_sample_config():
    """Create a sample configuration file"""
    config_path = Path("configs/phase4_config.yaml")
    
    if config_path.exists():
        print(f"Configuration file already exists: {config_path}")
        return str(config_path)
    
    # Create configs directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use the existing config file we created
    print(f"Using configuration file: {config_path}")
    return str(config_path)


def run_training_pipeline(args):
    """Run the complete training pipeline"""
    logger = logging.getLogger(__name__)
    
    try:
        # Check environment
        check_environment()
        
        # Create or use configuration
        if args.config:
            config_path = args.config
        else:
            config_path = create_sample_config()
        
        logger.info(f"Using configuration: {config_path}")
        
        # Create pipeline
        pipeline = Phase4TrainingPipeline(config_path=config_path, smoke_test=args.smoke_test)
        
        # Override configuration from command line
        if args.experiment_name:
            pipeline.config.experiment_name = args.experiment_name
        
        if args.epochs:
            # Distribute epochs across phases proportionally
            total_original = pipeline.config.total_epochs
            scale_factor = args.epochs / total_original
            
            pipeline.config.phase1.epochs = max(1, int(pipeline.config.phase1.epochs * scale_factor))
            pipeline.config.phase2.epochs = max(1, int(pipeline.config.phase2.epochs * scale_factor))
            pipeline.config.phase3.epochs = max(1, int(pipeline.config.phase3.epochs * scale_factor))
            pipeline.config.total_epochs = args.epochs
        
        if args.batch_size:
            pipeline.config.hardware.batch_size = args.batch_size
        
        if args.lr:
            pipeline.config.optimizer.lr = args.lr
        
        if args.device:
            pipeline.config.hardware.device = args.device
        
        # Setup hyperparameter optimization if requested
        hyperopt_config = None
        if args.optimize_hyperparams:
            param_space = HyperparameterSpace()
            
            # Customize parameter space based on arguments
            if args.optimization_trials:
                optimization_trials = args.optimization_trials
            else:
                optimization_trials = 20
            
            hyperopt_config = OptimizationConfig(
                study_name=f"{pipeline.config.experiment_name}_optimization",
                search_method=args.optimization_method,
                n_trials=optimization_trials,
                optimization_direction="maximize",
                primary_metric="val_combined_score",
                max_epochs_per_trial=min(30, pipeline.config.total_epochs),
                n_parallel_trials=1  # Sequential for stability
            )
        
        # Run complete pipeline
        logger.info("Starting Phase 4 training pipeline...")
        
        results = pipeline.run_complete_pipeline(
            enable_hyperparameter_optimization=args.optimize_hyperparams,
            enable_evaluation=not args.skip_evaluation,
            resume_from_checkpoint=args.resume
        )
        
        # Print final results
        print("\n" + "="*80)
        print("SUCCESS: TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        if 'training' in results:
            training_results = results['training']
            print(f"Best validation score: {training_results.get('best_metric_value', 0.0):.4f}")
            print(f"Total training time: {training_results.get('total_time_hours', 0.0):.2f} hours")
            print(f"Total epochs: {training_results.get('total_epochs', 0)}")
        
        if 'evaluation' in results:
            eval_results = results['evaluation']
            print(f"Final evaluation samples: {eval_results.get('total_samples', 0)}")
        
        print(f"Results saved to: {pipeline.config.experiment_dir}")
        print("="*80)
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Phase 4: Advanced Diabetic Retinopathy Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default configuration
  python train.py
  
  # Training with custom configuration
  python train.py --config configs/my_config.yaml
  
  # Quick training run (reduced epochs)
  python train.py --epochs 20 --batch-size 8
  
  # Training with hyperparameter optimization
  python train.py --optimize-hyperparams --optimization-trials 30
  
  # Resume training from checkpoint
  python train.py --resume checkpoints/latest_checkpoint.pth
  
  # Training with custom experiment name
  python train.py --experiment-name "my_experiment_v2"
        """
    )
    
    # Configuration
    parser.add_argument('--config', type=str, 
                       help='Path to YAML configuration file')
    parser.add_argument('--experiment-name', type=str,
                       help='Custom experiment name')
    
    # Training parameters
    parser.add_argument('--epochs', type=int,
                       help='Total number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Training batch size')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    # Pipeline options
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip final model evaluation')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint path')
    
    # Hyperparameter optimization
    parser.add_argument('--optimize-hyperparams', action='store_true',
                       help='Run hyperparameter optimization before training')
    parser.add_argument('--optimization-method', type=str, 
                       choices=['optuna', 'grid'], default='optuna',
                       help='Hyperparameter optimization method')
    parser.add_argument('--optimization-trials', type=int, default=20,
                       help='Number of optimization trials')
    
    # Logging
    parser.add_argument('--log-level', type=str, 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    # Development options
    parser.add_argument('--smoke-test', action='store_true',
                       help='Run quick smoke test (1 epoch)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling (development)')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        setup_logging('WARNING')
    else:
        setup_logging(args.log_level)
    
    # Smoke test configuration
    if args.smoke_test:
        args.epochs = 2
        args.batch_size = 4
        args.optimize_hyperparams = False
        print("Testing: Running smoke test (quick validation)")
    
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Print banner
    print("\n" + "="*80)
    print("PHASE 4: PHASE 4: DIABETIC RETINOPATHY TRAINING PIPELINE")
    print("="*80)
    print("Advanced multi-task training with progressive learning strategy")
    print("Features: GPU optimization, mixed precision, comprehensive metrics")
    print("="*80)
    
    # Run pipeline
    try:
        results = run_training_pipeline(args)
        
        if results:
            print("\nSUCCESS Pipeline completed successfully!")
            return 0
        else:
            print("\nERROR Pipeline was interrupted")
            return 1
            
    except Exception as e:
        print(f"\n💥 Pipeline failed: {str(e)}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
