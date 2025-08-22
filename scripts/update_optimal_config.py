#!/usr/bin/env python3
"""
Configuration Updater for Optimal Hyperparameters
Updates training configuration files with optimal parameters from hyperparameter optimization.
"""

import json
import yaml
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def load_optimization_results(results_file: str) -> Dict:
    """Load optimization results"""
    with open(results_file, 'r') as f:
        return json.load(f)

def load_config(config_file: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def backup_config(config_file: str) -> str:
    """Create backup of current configuration"""
    config_path = Path(config_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.parent / f"{config_path.stem}_backup_{timestamp}{config_path.suffix}"
    
    shutil.copy2(config_file, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return str(backup_path)

def update_config_with_optimal_params(config: Dict, optimal_params: Dict) -> Dict:
    """Update configuration with optimal parameters"""
    updated_config = config.copy()
    
    # Update optimizer parameters
    if 'learning_rate' in optimal_params:
        updated_config['optimizer']['lr'] = optimal_params['learning_rate']
    
    if 'weight_decay' in optimal_params:
        updated_config['optimizer']['weight_decay'] = optimal_params['weight_decay']
    
    # Update loss parameters
    if 'focal_gamma' in optimal_params:
        updated_config['loss']['focal_gamma'] = optimal_params['focal_gamma']
    
    if 'dice_smooth' in optimal_params:
        updated_config['loss']['dice_smooth'] = optimal_params['dice_smooth']
    
    # Update model parameters
    if 'classification_dropout' in optimal_params:
        updated_config['model']['dropout_rate'] = optimal_params['classification_dropout']
    
    # Update training parameters
    if 'segmentation_weight_final' in optimal_params:
        updated_config['phase3']['segmentation_weight'] = optimal_params['segmentation_weight_final']
    
    # Update hardware parameters
    if 'batch_size' in optimal_params:
        updated_config['hardware']['batch_size'] = optimal_params['batch_size']
    
    # Update scheduler parameters
    if 'cosine_t_max' in optimal_params:
        updated_config['scheduler']['T_max'] = optimal_params['cosine_t_max']
    
    # Add optimization metadata
    updated_config['optimization_info'] = {
        'source': 'hyperparameter_optimization',
        'updated_at': datetime.now().isoformat(),
        'optimal_parameters': optimal_params
    }
    
    return updated_config

def save_config(config: Dict, config_file: str):
    """Save updated configuration to file"""
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def create_optimal_config_summary(optimal_params: Dict, performance: Dict) -> str:
    """Create a summary of the optimal configuration"""
    summary = f"""# Optimal Hyperparameter Configuration Summary

## Performance Improvements
- **Classification Accuracy**: {performance.get('accuracy', 0):.4f}
- **Segmentation Dice Score**: {performance.get('dice_score', 0):.4f}
- **Combined Score**: {performance.get('combined_score', 0):.4f}
- **Sensitivity**: {performance.get('sensitivity', 0):.4f}
- **Specificity**: {performance.get('specificity', 0):.4f}

## Optimal Parameters

### Training Parameters
- **Learning Rate**: {optimal_params.get('learning_rate', 'N/A')}
- **Weight Decay**: {optimal_params.get('weight_decay', 'N/A')}
- **Batch Size**: {optimal_params.get('batch_size', 'N/A')}

### Loss Function Parameters
- **Focal Gamma**: {optimal_params.get('focal_gamma', 'N/A')}
- **Dice Smoothing**: {optimal_params.get('dice_smooth', 'N/A')}

### Model Parameters
- **Classification Dropout**: {optimal_params.get('classification_dropout', 'N/A')}
- **Segmentation Dropout**: {optimal_params.get('segmentation_dropout', 'N/A')}

### Multi-task Parameters
- **Final Segmentation Weight**: {optimal_params.get('segmentation_weight_final', 'N/A')}

## Implementation

The configuration has been automatically updated in `configs/phase4_config.yaml`.
You can now run training with these optimal parameters:

```bash
uv run run_focused_training.py --mode full
```

## Research Validation

These parameters were selected based on systematic hyperparameter optimization
and align with research evidence from medical imaging and multi-task learning literature.

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return summary

def main():
    """Main function for configuration update"""
    parser = argparse.ArgumentParser(
        description='Update configuration with optimal hyperparameters'
    )
    parser.add_argument('results_file', 
                       help='Path to hyperparameter optimization results JSON file')
    parser.add_argument('--config-file', 
                       default='configs/phase4_config.yaml',
                       help='Configuration file to update')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Create backup of current configuration')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show changes without updating files')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.results_file).exists():
        print(f"❌ Results file not found: {args.results_file}")
        return 1
    
    if not Path(args.config_file).exists():
        print(f"❌ Config file not found: {args.config_file}")
        return 1
    
    print("🔧 CONFIGURATION UPDATER")
    print("="*50)
    print(f"📄 Results file: {args.results_file}")
    print(f"⚙️ Config file: {args.config_file}")
    
    # Load optimization results
    results = load_optimization_results(args.results_file)
    
    if not results.get('best_config'):
        print("❌ No best configuration found in results file")
        return 1
    
    best_config = results['best_config']
    optimal_params = best_config['parameters']
    performance = best_config['metrics']
    
    print(f"\n🏆 Best Configuration: {best_config['config_name']}")
    print(f"📊 Combined Score: {performance['combined_score']:.4f}")
    print(f"🎯 Accuracy: {performance['accuracy']:.4f}")
    print(f"🎯 Dice Score: {performance['dice_score']:.4f}")
    
    # Load current configuration
    current_config = load_config(args.config_file)
    
    # Update configuration
    updated_config = update_config_with_optimal_params(current_config, optimal_params)
    
    # Show changes
    print(f"\n📋 Parameter Updates:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value}")
    
    if args.dry_run:
        print(f"\n⚠️ DRY RUN MODE - No files will be modified")
        return 0
    
    # Create backup
    if args.backup:
        backup_file = backup_config(args.config_file)
    
    # Save updated configuration
    save_config(updated_config, args.config_file)
    print(f"✅ Configuration updated: {args.config_file}")
    
    # Create summary
    summary = create_optimal_config_summary(optimal_params, performance)
    summary_file = Path(args.config_file).parent / "optimal_config_summary.md"
    
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"📋 Summary created: {summary_file}")
    
    # Show next steps
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Review the updated configuration in {args.config_file}")
    print(f"2. Run training with optimal parameters:")
    print(f"   uv run run_focused_training.py --mode full")
    print(f"3. Compare results with baseline performance")
    print(f"4. Document improvements for research paper")
    
    if args.backup:
        print(f"\n💾 Original configuration backed up to: {backup_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())
