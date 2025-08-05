"""
Phase 5: Experimental Design and Management System
Research-grade experiment planning and execution for diabetic retinopathy detection.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import yaml
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ExperimentGroup:
    """Define a group of related experiments"""
    
    name: str
    description: str
    base_config: Dict[str, Any]
    parameter_variations: Dict[str, List[Any]]
    priority: int = 1  # 1=high, 2=medium, 3=low
    expected_runtime_hours: float = 2.0
    tags: List[str] = field(default_factory=list)
    
    def generate_experiment_configs(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations from parameter variations"""
        experiments = []
        
        # Get all parameter combinations
        param_names = list(self.parameter_variations.keys())
        param_values = list(self.parameter_variations.values())
        
        for combination in itertools.product(*param_values):
            config = self.base_config.copy()
            
            # Apply parameter variations
            for param_name, param_value in zip(param_names, combination):
                # Handle nested parameters (e.g., "optimizer.learning_rate")
                if '.' in param_name:
                    keys = param_name.split('.')
                    target = config
                    for key in keys[:-1]:
                        if key not in target:
                            target[key] = {}
                        target = target[key]
                    target[keys[-1]] = param_value
                else:
                    config[param_name] = param_value
            
            # Add experiment metadata
            config['experiment_group'] = self.name
            config['parameter_combination'] = dict(zip(param_names, combination))
            
            experiments.append(config)
        
        return experiments


@dataclass
class ExperimentPlan:
    """Complete experimental design plan"""
    
    name: str
    description: str
    experiment_groups: List[ExperimentGroup]
    total_budget_hours: float = 48.0
    parallel_experiments: int = 1
    priority_scheduling: bool = True
    save_all_checkpoints: bool = False
    
    def get_total_experiments(self) -> int:
        """Get total number of experiments"""
        return sum(len(group.generate_experiment_configs()) for group in self.experiment_groups)
    
    def estimate_total_runtime(self) -> float:
        """Estimate total runtime for all experiments"""
        total_time = 0
        for group in self.experiment_groups:
            n_experiments = len(group.generate_experiment_configs())
            total_time += n_experiments * group.expected_runtime_hours
        
        if self.parallel_experiments > 1:
            total_time /= self.parallel_experiments
        
        return total_time
    
    def generate_execution_schedule(self) -> List[Dict[str, Any]]:
        """Generate optimized execution schedule"""
        all_experiments = []
        
        # Collect all experiments with metadata
        for group in self.experiment_groups:
            configs = group.generate_experiment_configs()
            for i, config in enumerate(configs):
                experiment = {
                    'group_name': group.name,
                    'experiment_id': f"{group.name}_exp_{i:03d}",
                    'config': config,
                    'priority': group.priority,
                    'estimated_hours': group.expected_runtime_hours,
                    'tags': group.tags
                }
                all_experiments.append(experiment)
        
        # Sort by priority if enabled
        if self.priority_scheduling:
            all_experiments.sort(key=lambda x: (x['priority'], x['estimated_hours']))
        
        return all_experiments


class ExperimentDesigner:
    """Design and manage diabetic retinopathy experiments"""
    
    def __init__(self, output_dir: str = "experiments/phase5_experimental_design"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('ExperimentDesigner')
        
    def create_baseline_experiments(self) -> ExperimentGroup:
        """Create baseline experiments with standard configurations"""
        
        base_config = {
            'experiment_description': 'Baseline experiments with standard configurations',
            'training_mode': 'full',
            'phase1_epochs': 15,
            'phase2_epochs': 15,
            'phase3_epochs': 20,
            'enable_hyperparameter_optimization': False,
            'model': {
                'backbone_name': 'efficientnetv2_s',
                'num_classes': 5,
                'pretrained': True
            },
            'optimizer': {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            },
            'hardware': {
                'batch_size': 16,
                'mixed_precision': True
            }
        }
        
        # Parameter variations for baseline
        parameter_variations = {
            'optimizer.learning_rate': [5e-4, 1e-3, 2e-3],
            'hardware.batch_size': [8, 16, 24],
            'model.backbone_name': ['efficientnetv2_s', 'resnet50']
        }
        
        return ExperimentGroup(
            name="baseline_experiments",
            description="Baseline experiments with standard configurations and key parameter sweeps",
            base_config=base_config,
            parameter_variations=parameter_variations,
            priority=1,
            expected_runtime_hours=3.0,
            tags=["baseline", "parameter_sweep"]
        )
    
    def create_progressive_training_experiments(self) -> ExperimentGroup:
        """Create experiments to optimize progressive training strategy"""
        
        base_config = {
            'experiment_description': 'Progressive training strategy optimization',
            'training_mode': 'full',
            'enable_hyperparameter_optimization': False,
            'model': {
                'backbone_name': 'efficientnetv2_s',
                'num_classes': 5,
                'pretrained': True
            },
            'optimizer': {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            },
            'hardware': {
                'batch_size': 16,
                'mixed_precision': True
            }
        }
        
        # Progressive training variations
        parameter_variations = {
            'phase1_epochs': [10, 15, 20],
            'phase2_epochs': [10, 15, 20],
            'phase3_epochs': [15, 20, 25],
            'segmentation_weight_max': [0.5, 0.8, 1.0],
            'segmentation_weight_warmup_epochs': [5, 10, 15]
        }
        
        return ExperimentGroup(
            name="progressive_training",
            description="Optimize progressive training phases and segmentation weight scheduling",
            base_config=base_config,
            parameter_variations=parameter_variations,
            priority=2,
            expected_runtime_hours=3.5,
            tags=["progressive", "multi_task", "scheduling"]
        )
    
    def create_architecture_experiments(self) -> ExperimentGroup:
        """Create experiments to test different model architectures"""
        
        base_config = {
            'experiment_description': 'Architecture comparison experiments',
            'training_mode': 'full',
            'phase1_epochs': 15,
            'phase2_epochs': 15,
            'phase3_epochs': 20,
            'enable_hyperparameter_optimization': False,
            'optimizer': {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            },
            'hardware': {
                'batch_size': 16,
                'mixed_precision': True
            }
        }
        
        # Architecture variations
        parameter_variations = {
            'model.backbone_name': [
                'efficientnetv2_s', 
                'efficientnetv2_b0',
                'efficientnetv2_b1',
                'resnet50',
                'resnet101'
            ],
            'model.use_skip_connections': [True, False],
            'model.use_advanced_decoder': [True, False],
            'model.cls_dropout': [0.2, 0.3, 0.4],
            'model.seg_dropout': [0.1, 0.2, 0.3]
        }
        
        return ExperimentGroup(
            name="architecture_comparison",
            description="Compare different backbone architectures and decoder configurations",
            base_config=base_config,
            parameter_variations=parameter_variations,
            priority=2,
            expected_runtime_hours=4.0,
            tags=["architecture", "backbone", "decoder"]
        )
    
    def create_loss_function_experiments(self) -> ExperimentGroup:
        """Create experiments to optimize loss functions"""
        
        base_config = {
            'experiment_description': 'Loss function optimization experiments',
            'training_mode': 'full',
            'phase1_epochs': 15,
            'phase2_epochs': 15,
            'phase3_epochs': 20,
            'enable_hyperparameter_optimization': False,
            'model': {
                'backbone_name': 'efficientnetv2_s',
                'num_classes': 5,
                'pretrained': True
            },
            'optimizer': {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            },
            'hardware': {
                'batch_size': 16,
                'mixed_precision': True
            }
        }
        
        # Loss function variations
        parameter_variations = {
            'loss.focal_gamma': [0.0, 1.0, 2.0, 3.0],
            'loss.dice_smooth': [1e-6, 1e-3, 1e-1],
            'loss.classification_weight': [0.7, 1.0, 1.3],
            'loss.segmentation_weight': [0.7, 1.0, 1.3],
            'loss.use_kappa_cls': [True, False],
            'loss.kappa_weight': [0.1, 0.3, 0.5]
        }
        
        return ExperimentGroup(
            name="loss_optimization",
            description="Optimize loss function components and weighting strategies",
            base_config=base_config,
            parameter_variations=parameter_variations,
            priority=2,
            expected_runtime_hours=3.0,
            tags=["loss_function", "weighting", "optimization"]
        )
    
    def create_data_augmentation_experiments(self) -> ExperimentGroup:
        """Create experiments to test data augmentation strategies"""
        
        base_config = {
            'experiment_description': 'Data augmentation strategy experiments',
            'training_mode': 'full',
            'phase1_epochs': 15,
            'phase2_epochs': 15,
            'phase3_epochs': 20,
            'enable_hyperparameter_optimization': False,
            'model': {
                'backbone_name': 'efficientnetv2_s',
                'num_classes': 5,
                'pretrained': True
            },
            'optimizer': {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            },
            'hardware': {
                'batch_size': 16,
                'mixed_precision': True
            }
        }
        
        # Data augmentation variations
        parameter_variations = {
            'augmentation_strength': ['light', 'medium', 'heavy'],
            'use_cutmix': [True, False],
            'use_mixup': [True, False],
            'rotation_range': [15, 30, 45],
            'brightness_contrast_limit': [0.1, 0.2, 0.3],
            'gaussian_noise_var': [0.01, 0.02, 0.05]
        }
        
        return ExperimentGroup(
            name="augmentation_experiments",
            description="Test different data augmentation strategies and intensities",
            base_config=base_config,
            parameter_variations=parameter_variations,
            priority=3,
            expected_runtime_hours=3.0,
            tags=["augmentation", "data", "robustness"]
        )
    
    def create_hyperparameter_optimization_experiments(self) -> ExperimentGroup:
        """Create experiments with hyperparameter optimization enabled"""
        
        base_config = {
            'experiment_description': 'Hyperparameter optimization experiments',
            'training_mode': 'full',
            'enable_hyperparameter_optimization': True,
            'hyperopt_trials': 50,
            'hyperopt_timeout_hours': 8.0,
            'phase1_epochs': 10,  # Shorter for faster optimization
            'phase2_epochs': 10,
            'phase3_epochs': 15
        }
        
        # Optimization strategy variations
        parameter_variations = {
            'hyperopt_search_method': ['optuna', 'grid'],
            'hyperopt_trials': [30, 50, 100],
            'hyperopt_pruning': [True, False],
            'optimization_objective': ['val_combined_score', 'val_classification_acc', 'val_segmentation_dice']
        }
        
        return ExperimentGroup(
            name="hyperparameter_optimization",
            description="Compare different hyperparameter optimization strategies",
            base_config=base_config,
            parameter_variations=parameter_variations,
            priority=3,
            expected_runtime_hours=10.0,
            tags=["hyperopt", "optimization", "search"]
        )
    
    def create_comprehensive_experiment_plan(self) -> ExperimentPlan:
        """Create a comprehensive experimental design plan"""
        
        experiment_groups = [
            self.create_baseline_experiments(),
            self.create_progressive_training_experiments(),
            self.create_architecture_experiments(),
            self.create_loss_function_experiments(),
            self.create_data_augmentation_experiments(),
            self.create_hyperparameter_optimization_experiments()
        ]
        
        return ExperimentPlan(
            name="diabetic_retinopathy_comprehensive_study",
            description="Comprehensive experimental study for diabetic retinopathy detection optimization",
            experiment_groups=experiment_groups,
            total_budget_hours=72.0,
            parallel_experiments=1,
            priority_scheduling=True,
            save_all_checkpoints=False
        )
    
    def create_quick_validation_plan(self) -> ExperimentPlan:
        """Create a quick validation plan for testing"""
        
        base_config = {
            'experiment_description': 'Quick validation experiments',
            'training_mode': 'debug',
            'phase1_epochs': 2,
            'phase2_epochs': 2,
            'phase3_epochs': 1,
            'enable_hyperparameter_optimization': False,
            'model': {
                'backbone_name': 'efficientnetv2_s',
                'num_classes': 5,
                'pretrained': True
            },
            'optimizer': {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            },
            'hardware': {
                'batch_size': 8,
                'mixed_precision': True
            }
        }
        
        # Minimal parameter variations
        parameter_variations = {
            'optimizer.learning_rate': [1e-3, 2e-3],
            'hardware.batch_size': [8, 16]
        }
        
        quick_group = ExperimentGroup(
            name="quick_validation",
            description="Quick validation experiments for testing pipeline",
            base_config=base_config,
            parameter_variations=parameter_variations,
            priority=1,
            expected_runtime_hours=0.5,
            tags=["validation", "quick", "test"]
        )
        
        return ExperimentPlan(
            name="quick_validation_study",
            description="Quick validation study for pipeline testing",
            experiment_groups=[quick_group],
            total_budget_hours=2.0,
            parallel_experiments=1,
            priority_scheduling=True,
            save_all_checkpoints=False
        )
    
    def save_experiment_plan(self, plan: ExperimentPlan, filename: str = None) -> str:
        """Save experiment plan to file"""
        if filename is None:
            filename = f"{plan.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        plan_path = self.output_dir / filename
        
        # Convert to serializable format
        plan_dict = {
            'plan_info': {
                'name': plan.name,
                'description': plan.description,
                'total_budget_hours': plan.total_budget_hours,
                'parallel_experiments': plan.parallel_experiments,
                'priority_scheduling': plan.priority_scheduling,
                'save_all_checkpoints': plan.save_all_checkpoints,
                'created_at': datetime.now().isoformat()
            },
            'statistics': {
                'total_experiment_groups': len(plan.experiment_groups),
                'total_experiments': plan.get_total_experiments(),
                'estimated_runtime_hours': plan.estimate_total_runtime()
            },
            'experiment_groups': []
        }
        
        for group in plan.experiment_groups:
            group_dict = {
                'name': group.name,
                'description': group.description,
                'priority': group.priority,
                'expected_runtime_hours': group.expected_runtime_hours,
                'tags': group.tags,
                'base_config': group.base_config,
                'parameter_variations': group.parameter_variations,
                'n_experiments': len(group.generate_experiment_configs())
            }
            plan_dict['experiment_groups'].append(group_dict)
        
        # Save to YAML
        with open(plan_path, 'w') as f:
            yaml.dump(plan_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Experiment plan saved to: {plan_path}")
        return str(plan_path)
    
    def generate_execution_scripts(self, plan: ExperimentPlan) -> List[str]:
        """Generate execution scripts for the experiment plan"""
        scripts_dir = self.output_dir / "execution_scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        execution_schedule = plan.generate_execution_schedule()
        script_paths = []
        
        for i, experiment in enumerate(execution_schedule):
            script_name = f"run_{experiment['experiment_id']}.py"
            script_path = scripts_dir / script_name
            
            # Generate script content
            script_content = self._generate_experiment_script(experiment)
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_path, 0o755)
            script_paths.append(str(script_path))
        
        # Generate master execution script
        master_script = self._generate_master_script(execution_schedule, scripts_dir)
        master_path = scripts_dir / "run_all_experiments.py"
        with open(master_path, 'w') as f:
            f.write(master_script)
        os.chmod(master_path, 0o755)
        script_paths.append(str(master_path))
        
        self.logger.info(f"Generated {len(script_paths)} execution scripts in: {scripts_dir}")
        return script_paths
    
    def _generate_experiment_script(self, experiment: Dict[str, Any]) -> str:
        """Generate individual experiment execution script"""
        script_template = '''#!/usr/bin/env python3
"""
Auto-generated experiment script for: {experiment_id}
Group: {group_name}
Priority: {priority}
Estimated runtime: {estimated_hours:.1f} hours
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from training.phase5_trainer import Phase5Trainer, Phase5Config

def main():
    """Run experiment: {experiment_id}"""
    
    # Experiment configuration
    config = Phase5Config(
        experiment_name="{experiment_id}",
        experiment_description="{description}",
        experiment_tags={tags}
    )
    
    # Apply experiment-specific parameters
    experiment_config = {experiment_config}
    
    # Update config with experiment parameters
    for key, value in experiment_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create trainer and run
    trainer = Phase5Trainer(config)
    
    try:
        results = trainer.run_complete_training_pipeline()
        print(f"✅ Experiment {experiment_id} completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Experiment {experiment_id} failed: {{e}}")
        return 1

if __name__ == "__main__":
    exit(main())
'''
        
        return script_template.format(
            experiment_id=experiment['experiment_id'],
            group_name=experiment['group_name'],
            priority=experiment['priority'],
            estimated_hours=experiment['estimated_hours'],
            description=experiment['config'].get('experiment_description', ''),
            tags=experiment['tags'],
            experiment_config=experiment['config']
        )
    
    def _generate_master_script(self, execution_schedule: List[Dict], scripts_dir: Path) -> str:
        """Generate master execution script"""
        script_template = '''#!/usr/bin/env python3
"""
Master execution script for experimental plan
Generated on: {timestamp}
Total experiments: {total_experiments}
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def run_experiment(script_path):
    """Run a single experiment script"""
    print(f"Starting experiment: {{script_path.stem}}")
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, timeout=3600*12)  # 12 hour timeout
        
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {{script_path.stem}} completed successfully in {{runtime/3600:.2f}}h")
            return True
        else:
            print(f"❌ {{script_path.stem}} failed after {{runtime/3600:.2f}}h")
            print(f"Error: {{result.stderr}}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {{script_path.stem}} timed out after 12 hours")
        return False
    except Exception as e:
        print(f"💥 {{script_path.stem}} crashed: {{e}}")
        return False

def main():
    """Run all experiments in sequence"""
    script_dir = Path(__file__).parent
    
    experiments = [
{experiment_list}
    ]
    
    print(f"Starting experimental plan with {{len(experiments)}} experiments")
    print("=" * 80)
    
    completed = 0
    failed = 0
    total_start_time = time.time()
    
    for script_name in experiments:
        script_path = script_dir / script_name
        
        if script_path.exists():
            success = run_experiment(script_path)
            if success:
                completed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  Script not found: {{script_path}}")
            failed += 1
        
        print("-" * 40)
    
    total_runtime = time.time() - total_start_time
    
    print("=" * 80)
    print("EXPERIMENTAL PLAN COMPLETED")
    print(f"Total runtime: {{total_runtime/3600:.2f}} hours")
    print(f"Completed: {{completed}}")
    print(f"Failed: {{failed}}")
    print(f"Success rate: {{completed/(completed+failed)*100:.1f}}%")
    print("=" * 80)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())
'''
        
        experiment_list = '\n'.join([
            f'        "run_{exp["experiment_id"]}.py",'
            for exp in execution_schedule
        ])
        
        return script_template.format(
            timestamp=datetime.now().isoformat(),
            total_experiments=len(execution_schedule),
            experiment_list=experiment_list
        )
    
    def generate_analysis_report(self, plan: ExperimentPlan) -> str:
        """Generate analysis report for the experiment plan"""
        report_path = self.output_dir / f"{plan.name}_analysis_report.md"
        
        # Calculate statistics
        total_experiments = plan.get_total_experiments()
        estimated_runtime = plan.estimate_total_runtime()
        
        # Group statistics
        group_stats = []
        for group in plan.experiment_groups:
            n_experiments = len(group.generate_experiment_configs())
            group_stats.append({
                'name': group.name,
                'n_experiments': n_experiments,
                'estimated_hours': n_experiments * group.expected_runtime_hours,
                'priority': group.priority,
                'tags': group.tags
            })
        
        # Generate report content
        report_content = f"""# Experimental Design Analysis Report

## Plan Overview
- **Name**: {plan.name}
- **Description**: {plan.description}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- **Total Experiments**: {total_experiments:,}
- **Estimated Runtime**: {estimated_runtime:.1f} hours ({estimated_runtime/24:.1f} days)
- **Budget**: {plan.total_budget_hours:.1f} hours
- **Budget Utilization**: {(estimated_runtime/plan.total_budget_hours)*100:.1f}%
- **Parallel Experiments**: {plan.parallel_experiments}

## Experiment Groups

"""
        
        for i, (group, stats) in enumerate(zip(plan.experiment_groups, group_stats)):
            report_content += f"""### {i+1}. {group.name}
- **Description**: {group.description}
- **Experiments**: {stats['n_experiments']:,}
- **Estimated Time**: {stats['estimated_hours']:.1f} hours
- **Priority**: {stats['priority']} ({'High' if stats['priority']==1 else 'Medium' if stats['priority']==2 else 'Low'})
- **Tags**: {', '.join(stats['tags'])}

**Parameter Variations**:
"""
            for param, values in group.parameter_variations.items():
                report_content += f"- `{param}`: {values}\n"
            
            report_content += "\n"
        
        # Resource requirements
        report_content += f"""## Resource Requirements

### Computational Resources
- **Estimated GPU Hours**: {estimated_runtime:.1f}
- **Estimated Cost** (approx $0.50/hour): ${estimated_runtime * 0.50:.2f}
- **Storage Requirements**: ~{total_experiments * 2:.1f} GB (2GB per experiment)

### Timeline
- **Sequential Execution**: {estimated_runtime:.1f} hours ({estimated_runtime/24:.1f} days)
- **With {plan.parallel_experiments} parallel**: {estimated_runtime/plan.parallel_experiments:.1f} hours ({estimated_runtime/plan.parallel_experiments/24:.1f} days)

## Execution Strategy

### Priority Order
1. **High Priority** ({len([g for g in group_stats if g['priority']==1])} groups): Core baseline and architecture experiments
2. **Medium Priority** ({len([g for g in group_stats if g['priority']==2])} groups): Advanced optimization experiments  
3. **Low Priority** ({len([g for g in group_stats if g['priority']==3])} groups): Exploratory and validation experiments

### Recommendations
- Start with high-priority baseline experiments to establish performance benchmarks
- Run architecture comparisons early to inform subsequent experiments
- Save hyperparameter optimization experiments for last (highest computational cost)
- Monitor resource usage and adjust batch sizes if needed
- Enable checkpointing for experiments > 4 hours

## Quality Assurance
- All experiments include validation monitoring
- Resource usage tracking enabled
- Automatic quality control checks
- Comprehensive logging and reporting
- Reproducible random seeds

## Success Metrics
- **Primary**: Validation accuracy for diabetic retinopathy grading
- **Secondary**: Segmentation Dice score for lesion detection
- **Efficiency**: Training time and resource utilization
- **Robustness**: Consistency across parameter variations

---
*Generated by Phase 5 Experimental Design System*
"""
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Analysis report saved to: {report_path}")
        return str(report_path)


def main():
    """Main function for experimental design"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 5: Experimental Design System")
    parser.add_argument('--plan-type', type=str, 
                       choices=['comprehensive', 'quick', 'baseline', 'architecture'], 
                       default='quick',
                       help='Type of experimental plan to create')
    parser.add_argument('--output-dir', type=str, 
                       default='experiments/phase5_experimental_design',
                       help='Output directory for experimental design')
    parser.add_argument('--generate-scripts', action='store_true',
                       help='Generate execution scripts')
    
    args = parser.parse_args()
    
    # Create designer
    designer = ExperimentDesigner(args.output_dir)
    
    # Create experiment plan
    if args.plan_type == 'comprehensive':
        plan = designer.create_comprehensive_experiment_plan()
    elif args.plan_type == 'quick':
        plan = designer.create_quick_validation_plan()
    elif args.plan_type == 'baseline':
        plan = ExperimentPlan(
            name="baseline_study",
            description="Baseline experiments only",
            experiment_groups=[designer.create_baseline_experiments()],
            total_budget_hours=24.0
        )
    elif args.plan_type == 'architecture':
        plan = ExperimentPlan(
            name="architecture_study", 
            description="Architecture comparison study",
            experiment_groups=[designer.create_architecture_experiments()],
            total_budget_hours=48.0
        )
    
    # Save plan
    plan_path = designer.save_experiment_plan(plan)
    print(f"Experiment plan saved to: {plan_path}")
    
    # Generate analysis report
    report_path = designer.generate_analysis_report(plan)
    print(f"Analysis report saved to: {report_path}")
    
    # Generate execution scripts if requested
    if args.generate_scripts:
        script_paths = designer.generate_execution_scripts(plan)
        print(f"Generated {len(script_paths)} execution scripts")
        print(f"Run with: python {script_paths[-1]}")  # Master script
    
    # Print summary
    print(f"\nExperimental Design Summary:")
    print(f"  Plan: {plan.name}")
    print(f"  Total experiments: {plan.get_total_experiments()}")
    print(f"  Estimated runtime: {plan.estimate_total_runtime():.1f} hours")
    print(f"  Budget: {plan.total_budget_hours:.1f} hours")


if __name__ == "__main__":
    main()
