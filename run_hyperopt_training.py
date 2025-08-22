#!/usr/bin/env python3
"""
Hyperparameter Optimization Training Runner
Enables actual training with dynamic hyperparameters instead of simulation
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('HyperoptTraining')

def load_base_config(config_path: str = "configs/phase4_config.yaml") -> Dict:
    """Load base configuration file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def update_config_with_hyperparams(base_config: Dict, hyperparams: Dict) -> Dict:
    """Update base configuration with hyperparameters"""
    config = base_config.copy()
    
    # Update optimizer parameters
    if 'learning_rate' in hyperparams:
        if 'optimizer' not in config:
            config['optimizer'] = {}
        config['optimizer']['learning_rate'] = hyperparams['learning_rate']
    
    if 'weight_decay' in hyperparams:
        if 'optimizer' not in config:
            config['optimizer'] = {}
        config['optimizer']['weight_decay'] = hyperparams['weight_decay']
    
    # Update loss parameters
    if 'focal_gamma' in hyperparams:
        if 'loss' not in config:
            config['loss'] = {}
        config['loss']['focal_gamma'] = hyperparams['focal_gamma']
    
    # Update model parameters
    if 'classification_dropout' in hyperparams:
        if 'model' not in config:
            config['model'] = {}
        config['model']['cls_dropout'] = hyperparams['classification_dropout']
    
    # Update training parameters
    if 'batch_size' in hyperparams:
        if 'hardware' not in config:
            config['hardware'] = {}
        config['hardware']['batch_size'] = hyperparams['batch_size']
    
    if 'segmentation_weight_final' in hyperparams:
        if 'phase3' not in config:
            config['phase3'] = {}
        config['phase3']['segmentation_weight'] = hyperparams['segmentation_weight_final']
    
    return config

def run_training_with_hyperparams(hyperparams: Dict, mode: str = "quick", 
                                config_path: str = "configs/phase4_config.yaml",
                                log_level: str = "INFO") -> Dict:
    """Run training with specified hyperparameters"""
    
    logger = setup_logging(log_level)
    
    try:
        # Load base configuration
        logger.info(f"📋 Loading base config from: {config_path}")
        base_config = load_base_config(config_path)
        
        # Update with hyperparameters
        logger.info(f"🔧 Applying hyperparameters: {hyperparams}")
        updated_config = update_config_with_hyperparams(base_config, hyperparams)
        
        # Create temporary config file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_config_path = Path(f"temp_hyperopt_config_{timestamp}.yaml")
        
        logger.info(f"💾 Saving temporary config: {temp_config_path}")
        with open(temp_config_path, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False)
        
        # Import and run focused training
        logger.info(f"🚀 Starting training with mode: {mode}")
        
        # Modify environment to use our temporary config
        original_config = os.environ.get('TRAINING_CONFIG', None)
        os.environ['TRAINING_CONFIG'] = str(temp_config_path)
        
        try:
            from run_focused_training import run_focused_training
            
            # Run actual training
            results = run_focused_training(mode=mode, log_level=log_level)
            
            # Ensure we have proper results structure
            if results and isinstance(results, dict):
                # Add hyperparameter tracking
                results['hyperparameters_used'] = hyperparams
                results['config_file_used'] = str(temp_config_path)
                results['actual_training'] = True
                
                logger.info("✅ Training completed successfully")
                return results
            else:
                logger.warning("⚠️ Training returned incomplete results")
                return {'error': 'Training returned incomplete results'}
                
        finally:
            # Restore original environment
            if original_config:
                os.environ['TRAINING_CONFIG'] = original_config
            elif 'TRAINING_CONFIG' in os.environ:
                del os.environ['TRAINING_CONFIG']
            
            # Clean up temporary config
            if temp_config_path.exists():
                temp_config_path.unlink()
                logger.info(f"🧹 Cleaned up temporary config: {temp_config_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return {'error': str(e), 'training_failed': True}

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Run training with hyperparameters')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--classification_dropout', type=float, default=0.3, help='Classification dropout')
    parser.add_argument('--segmentation_weight_final', type=float, default=0.8, help='Final segmentation weight')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'], help='Training mode')
    parser.add_argument('--config', type=str, default='configs/phase4_config.yaml', help='Base config file')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Prepare hyperparameters
    hyperparams = {
        'learning_rate': args.learning_rate,
        'focal_gamma': args.focal_gamma,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'classification_dropout': args.classification_dropout,
        'segmentation_weight_final': args.segmentation_weight_final
    }
    
    # Run training
    results = run_training_with_hyperparams(
        hyperparams=hyperparams,
        mode=args.mode,
        config_path=args.config,
        log_level=args.log_level
    )
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        sys.exit(1)
    else:
        training_results = results.get('training_results', {})
        if 'best_metrics' in training_results:
            metrics = training_results['best_metrics']
            print(f"✅ Training completed successfully")
            print(f"📊 Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"📊 Dice Score: {metrics.get('dice_score', 0):.4f}")
            print(f"📊 Combined Score: {metrics.get('combined_score', 0):.4f}")
        else:
            print("⚠️ Results structure incomplete")

if __name__ == "__main__":
    main()
