"""
Phase 4: Hyperparameter Search and Optimization System
Advanced hyperparameter tuning with Optuna, Ray Tune, and grid search
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader

# Hyperparameter optimization libraries
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Local imports
from .config import Phase4Config, OptimizerConfig, SchedulerConfig
from .trainer import RobustPhase4Trainer


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space"""
    
    # Learning rate
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    lr_log: bool = True
    
    # Batch size
    batch_sizes: List[int] = None
    
    # Optimizer parameters
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-2
    weight_decay_log: bool = True
    
    # Model architecture
    backbone_names: List[str] = None
    use_skip_connections: bool = True
    use_advanced_decoder: bool = True
    
    # Loss function
    focal_gamma_min: float = 0.0
    focal_gamma_max: float = 3.0
    dice_smooth_min: float = 1e-6
    dice_smooth_max: float = 1.0
    
    # Training strategy
    phase1_epochs_range: Tuple[int, int] = (5, 20)
    phase2_epochs_range: Tuple[int, int] = (5, 15)
    phase3_epochs_range: Tuple[int, int] = (10, 30)
    
    # Segmentation weight progression
    seg_weight_max_range: Tuple[float, float] = (0.3, 1.0)
    seg_weight_warmup_range: Tuple[int, int] = (5, 15)
    
    # Regularization
    dropout_range: Tuple[float, float] = (0.1, 0.5)
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [8, 16, 24, 32]
        
        if self.backbone_names is None:
            self.backbone_names = [
                'efficientnet-b0',
                'efficientnet-b1', 
                'efficientnet-b2',
                'resnet50',
                'resnet101'
            ]


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    
    # Study configuration
    study_name: str = "diabetic_retinopathy_optimization"
    storage_url: Optional[str] = None  # For distributed optimization
    
    # Search strategy
    search_method: str = "optuna"  # 'optuna', 'ray', 'grid'
    n_trials: int = 50
    timeout_hours: Optional[float] = None
    
    # Pruning and early stopping
    enable_pruning: bool = True
    pruning_patience: int = 5
    min_trials_before_pruning: int = 10
    
    # Parallel execution
    n_parallel_trials: int = 1
    
    # Validation strategy
    validation_split: float = 0.2
    cross_validation_folds: Optional[int] = None
    
    # Objective function
    optimization_direction: str = "maximize"  # 'maximize' or 'minimize'
    primary_metric: str = "val_combined_score"
    early_stopping_epochs: int = 10
    
    # Resource constraints
    max_epochs_per_trial: int = 50
    max_time_per_trial_hours: float = 6.0
    
    # Results
    save_all_trials: bool = True
    save_best_n_models: int = 5


class HyperparameterOptimizer:
    """
    Phase 4: Advanced Hyperparameter Optimization System
    
    Features:
    - Multiple optimization backends (Optuna, Ray Tune, Grid Search)
    - Intelligent pruning of poor trials
    - Parallel trial execution
    - Cross-validation support
    - Automated resource management
    - Comprehensive results analysis
    """
    
    def __init__(self,
                 base_config: Phase4Config,
                 param_space: HyperparameterSpace,
                 optimization_config: OptimizationConfig,
                 logger: Optional[logging.Logger] = None):
        
        self.base_config = base_config
        self.param_space = param_space
        self.opt_config = optimization_config
        self.logger = logger or self._setup_logger()
        
        # Create optimization directory
        self.optimization_dir = Path(base_config.experiment_dir) / "hyperparameter_optimization"
        self.optimization_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.trials_results = []
        self.best_trials = []
        
        # Resource tracking
        self.total_trials_run = 0
        self.start_time = None
        
        self.logger.info("Hyperparameter Optimizer initialized")
        self.logger.info(f"Search method: {optimization_config.search_method}")
        self.logger.info(f"Number of trials: {optimization_config.n_trials}")
        self.logger.info(f"Optimization directory: {self.optimization_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for optimization"""
        logger = logging.getLogger('HyperparameterOptimizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = self.optimization_dir / "optimization.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial (Optuna-style)"""
        params = {}
        
        # Learning rate
        if self.param_space.lr_log:
            params['lr'] = trial.suggest_float('lr', self.param_space.lr_min, self.param_space.lr_max, log=True)
        else:
            params['lr'] = trial.suggest_float('lr', self.param_space.lr_min, self.param_space.lr_max)
        
        # Batch size
        params['batch_size'] = trial.suggest_categorical('batch_size', self.param_space.batch_sizes)
        
        # Weight decay
        if self.param_space.weight_decay_log:
            params['weight_decay'] = trial.suggest_float('weight_decay', 
                                                        self.param_space.weight_decay_min, 
                                                        self.param_space.weight_decay_max, 
                                                        log=True)
        else:
            params['weight_decay'] = trial.suggest_float('weight_decay', 
                                                        self.param_space.weight_decay_min, 
                                                        self.param_space.weight_decay_max)
        
        # Model architecture
        params['backbone_name'] = trial.suggest_categorical('backbone_name', self.param_space.backbone_names)
        
        # Loss function parameters
        params['focal_gamma'] = trial.suggest_float('focal_gamma', 
                                                   self.param_space.focal_gamma_min, 
                                                   self.param_space.focal_gamma_max)
        params['dice_smooth'] = trial.suggest_float('dice_smooth', 
                                                   self.param_space.dice_smooth_min, 
                                                   self.param_space.dice_smooth_max)
        
        # Training strategy
        params['phase1_epochs'] = trial.suggest_int('phase1_epochs', 
                                                   self.param_space.phase1_epochs_range[0], 
                                                   self.param_space.phase1_epochs_range[1])
        params['phase2_epochs'] = trial.suggest_int('phase2_epochs', 
                                                   self.param_space.phase2_epochs_range[0], 
                                                   self.param_space.phase2_epochs_range[1])
        params['phase3_epochs'] = trial.suggest_int('phase3_epochs', 
                                                   self.param_space.phase3_epochs_range[0], 
                                                   self.param_space.phase3_epochs_range[1])
        
        # Segmentation weight progression
        params['seg_weight_max'] = trial.suggest_float('seg_weight_max', 
                                                      self.param_space.seg_weight_max_range[0], 
                                                      self.param_space.seg_weight_max_range[1])
        params['seg_weight_warmup'] = trial.suggest_int('seg_weight_warmup', 
                                                       self.param_space.seg_weight_warmup_range[0], 
                                                       self.param_space.seg_weight_warmup_range[1])
        
        return params
    
    def create_config_from_params(self, params: Dict[str, Any], trial_id: int) -> Phase4Config:
        """Create training configuration from hyperparameters"""
        # Copy base config
        config = Phase4Config()
        
        # Update experiment name
        config.experiment_name = f"{self.base_config.experiment_name}_trial_{trial_id:03d}"
        config.experiment_dir = self.optimization_dir / f"trial_{trial_id:03d}"
        
        # Update model configuration
        config.model.backbone_name = params['backbone_name']
        config.model.use_skip_connections = self.param_space.use_skip_connections
        config.model.use_advanced_decoder = self.param_space.use_advanced_decoder
        
        # Update optimizer configuration
        config.optimizer.learning_rate = params['lr']
        config.optimizer.weight_decay = params['weight_decay']
        
        # Update batch size
        config.hardware.batch_size = params['batch_size']
        
        # Update loss function parameters
        config.loss.focal_gamma = params['focal_gamma']
        config.loss.dice_smooth = params['dice_smooth']
        
        # Update training phases
        config.progressive.phase1_epochs = params['phase1_epochs']
        config.progressive.phase2_epochs = params['phase2_epochs']
        config.progressive.phase3_epochs = params['phase3_epochs']
        
        # Update segmentation weight progression
        config.segmentation_weight_max = params['seg_weight_max']
        config.segmentation_weight_warmup_epochs = params['seg_weight_warmup']
        
        # Set early stopping for trials
        config.early_stopping.patience = self.opt_config.early_stopping_epochs
        
        # Limit maximum epochs for efficiency
        if config.progressive.total_epochs > self.opt_config.max_epochs_per_trial:
            scale_factor = self.opt_config.max_epochs_per_trial / config.progressive.total_epochs
            config.progressive.phase1_epochs = max(1, int(config.progressive.phase1_epochs * scale_factor))
            config.progressive.phase2_epochs = max(1, int(config.progressive.phase2_epochs * scale_factor))
            config.progressive.phase3_epochs = max(1, int(config.progressive.phase3_epochs * scale_factor))
        
        return config
    
    def objective_function(self, trial) -> float:
        """Objective function for a single trial"""
        trial_start_time = time.time()
        
        try:
            # Generate hyperparameters
            params = self.suggest_hyperparameters(trial)
            
            # Create configuration
            config = self.create_config_from_params(params, trial.number)
            
            self.logger.info(f"Starting trial {trial.number} with params: {params}")
            
            # Create trainer
            trainer = RobustPhase4Trainer(config, logger=self.logger)
            
            # Note: In a real implementation, you would need to provide train_loader and val_loader
            # For this example, we'll simulate the training process
            
            # Simulate training (replace with actual training)
            best_score = self._simulate_training(trainer, config, trial)
            
            # Track trial time
            trial_time = time.time() - trial_start_time
            
            # Save trial results
            trial_result = {
                'trial_id': trial.number,
                'params': params,
                'score': best_score,
                'trial_time_minutes': trial_time / 60,
                'config': asdict(config)
            }
            
            self.trials_results.append(trial_result)
            self._save_trial_result(trial_result)
            
            self.logger.info(f"Trial {trial.number} completed: score={best_score:.4f}, time={trial_time/60:.1f}min")
            
            return best_score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            return -float('inf') if self.opt_config.optimization_direction == "maximize" else float('inf')
    
    def _simulate_training(self, trainer: RobustPhase4Trainer, config: Phase4Config, trial) -> float:
        """
        Simulate training for demonstration purposes
        In real implementation, replace with actual training loop
        """
        # This is a placeholder - replace with actual training
        # For demonstration, we'll return a random score based on parameters
        
        import random
        
        # Simulate training progress with some randomness
        base_score = 0.75
        
        # Bonus for better architectures
        if 'efficientnet' in config.model.backbone_name:
            base_score += 0.05
        
        # Penalty for very high learning rates
        if config.optimizer.learning_rate > 0.01:
            base_score -= 0.1
        
        # Bonus for reasonable batch sizes
        if 16 <= config.hardware.batch_size <= 32:
            base_score += 0.02
        
        # Add some randomness
        score = base_score + random.uniform(-0.1, 0.1)
        
        # Simulate pruning by reporting intermediate values
        if OPTUNA_AVAILABLE and hasattr(trial, 'report'):
            for epoch in range(min(10, config.progressive.total_epochs)):
                intermediate_score = score * (epoch + 1) / config.progressive.total_epochs
                trial.report(intermediate_score, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        return max(0.0, min(1.0, score))
    
    def _save_trial_result(self, trial_result: Dict[str, Any]):
        """Save individual trial result"""
        trial_file = self.optimization_dir / f"trial_{trial_result['trial_id']:03d}.json"
        with open(trial_file, 'w') as f:
            json.dump(trial_result, f, indent=2, default=str)
    
    def optimize_with_optuna(self) -> Dict[str, Any]:
        """Run hyperparameter optimization with Optuna"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for this optimization method")
        
        self.logger.info("Starting Optuna optimization...")
        
        # Create study
        study = optuna.create_study(
            study_name=self.opt_config.study_name,
            direction=self.opt_config.optimization_direction,
            storage=self.opt_config.storage_url,
            sampler=TPESampler(),
            pruner=MedianPruner(
                n_startup_trials=self.opt_config.min_trials_before_pruning,
                n_warmup_steps=self.opt_config.pruning_patience
            ) if self.opt_config.enable_pruning else None
        )
        
        # Set timeout
        timeout = None
        if self.opt_config.timeout_hours:
            timeout = self.opt_config.timeout_hours * 3600
        
        # Run optimization
        study.optimize(
            self.objective_function,
            n_trials=self.opt_config.n_trials,
            timeout=timeout,
            n_jobs=self.opt_config.n_parallel_trials
        )
        
        # Get results
        best_trial = study.best_trial
        
        results = {
            'best_params': best_trial.params,
            'best_score': best_trial.value,
            'n_trials': len(study.trials),
            'optimization_history': [(t.number, t.value) for t in study.trials if t.value is not None]
        }
        
        self.logger.info(f"Optuna optimization completed!")
        self.logger.info(f"Best score: {best_trial.value:.4f}")
        self.logger.info(f"Best params: {best_trial.params}")
        
        return results
    
    def optimize_with_ray(self) -> Dict[str, Any]:
        """Run hyperparameter optimization with Ray Tune"""
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune is required for this optimization method")
        
        self.logger.info("Starting Ray Tune optimization...")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
        
        # Define search space for Ray Tune
        search_space = self._create_ray_search_space()
        
        # Configure scheduler and search algorithm
        scheduler = ASHAScheduler(
            metric=self.opt_config.primary_metric,
            mode=self.opt_config.optimization_direction,
            max_t=self.opt_config.max_epochs_per_trial,
            grace_period=self.opt_config.early_stopping_epochs
        )
        
        search_alg = OptunaSearch() if OPTUNA_AVAILABLE else None
        
        # Run optimization
        analysis = tune.run(
            self._ray_trainable_function,
            config=search_space,
            num_samples=self.opt_config.n_trials,
            scheduler=scheduler,
            search_alg=search_alg,
            local_dir=str(self.optimization_dir),
            name="ray_optimization"
        )
        
        # Get results
        best_trial = analysis.best_trial
        
        results = {
            'best_params': best_trial.config,
            'best_score': best_trial.last_result[self.opt_config.primary_metric],
            'n_trials': len(analysis.trials),
            'optimization_history': [(i, trial.last_result.get(self.opt_config.primary_metric, 0)) 
                                   for i, trial in enumerate(analysis.trials)]
        }
        
        self.logger.info(f"Ray Tune optimization completed!")
        self.logger.info(f"Best score: {results['best_score']:.4f}")
        self.logger.info(f"Best params: {results['best_params']}")
        
        return results
    
    def _create_ray_search_space(self) -> Dict[str, Any]:
        """Create search space for Ray Tune"""
        search_space = {}
        
        if self.param_space.lr_log:
            search_space['lr'] = tune.loguniform(self.param_space.lr_min, self.param_space.lr_max)
        else:
            search_space['lr'] = tune.uniform(self.param_space.lr_min, self.param_space.lr_max)
        
        search_space['batch_size'] = tune.choice(self.param_space.batch_sizes)
        search_space['backbone_name'] = tune.choice(self.param_space.backbone_names)
        
        if self.param_space.weight_decay_log:
            search_space['weight_decay'] = tune.loguniform(self.param_space.weight_decay_min, 
                                                          self.param_space.weight_decay_max)
        else:
            search_space['weight_decay'] = tune.uniform(self.param_space.weight_decay_min, 
                                                       self.param_space.weight_decay_max)
        
        search_space['focal_gamma'] = tune.uniform(self.param_space.focal_gamma_min, 
                                                  self.param_space.focal_gamma_max)
        search_space['dice_smooth'] = tune.uniform(self.param_space.dice_smooth_min, 
                                                  self.param_space.dice_smooth_max)
        
        search_space['phase1_epochs'] = tune.randint(self.param_space.phase1_epochs_range[0], 
                                                    self.param_space.phase1_epochs_range[1] + 1)
        search_space['phase2_epochs'] = tune.randint(self.param_space.phase2_epochs_range[0], 
                                                    self.param_space.phase2_epochs_range[1] + 1)
        search_space['phase3_epochs'] = tune.randint(self.param_space.phase3_epochs_range[0], 
                                                    self.param_space.phase3_epochs_range[1] + 1)
        
        search_space['seg_weight_max'] = tune.uniform(self.param_space.seg_weight_max_range[0], 
                                                     self.param_space.seg_weight_max_range[1])
        search_space['seg_weight_warmup'] = tune.randint(self.param_space.seg_weight_warmup_range[0], 
                                                        self.param_space.seg_weight_warmup_range[1] + 1)
        
        return search_space
    
    def _ray_trainable_function(self, config: Dict[str, Any]):
        """Trainable function for Ray Tune"""
        # This would be implemented similarly to objective_function
        # but adapted for Ray Tune's callback system
        pass
    
    def optimize_with_grid_search(self) -> Dict[str, Any]:
        """Run hyperparameter optimization with grid search"""
        self.logger.info("Starting grid search optimization...")
        
        # Generate all parameter combinations
        param_combinations = self._generate_grid_combinations()
        
        self.logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        best_score = -float('inf') if self.opt_config.optimization_direction == "maximize" else float('inf')
        best_params = None
        
        for i, params in enumerate(param_combinations[:self.opt_config.n_trials]):
            try:
                # Create configuration
                config = self.create_config_from_params(params, i)
                
                self.logger.info(f"Grid search trial {i+1}/{min(len(param_combinations), self.opt_config.n_trials)}")
                
                # Create trainer and simulate training
                trainer = RobustPhase4Trainer(config, logger=self.logger)
                score = self._simulate_training(trainer, config, None)
                
                # Check if this is the best score
                is_better = (score > best_score if self.opt_config.optimization_direction == "maximize" 
                           else score < best_score)
                
                if is_better:
                    best_score = score
                    best_params = params.copy()
                
                # Save trial result
                trial_result = {
                    'trial_id': i,
                    'params': params,
                    'score': score,
                    'config': asdict(config)
                }
                self.trials_results.append(trial_result)
                self._save_trial_result(trial_result)
                
            except Exception as e:
                self.logger.error(f"Grid search trial {i} failed: {str(e)}")
                continue
        
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.trials_results),
            'optimization_history': [(i, result['score']) for i, result in enumerate(self.trials_results)]
        }
        
        self.logger.info(f"Grid search optimization completed!")
        self.logger.info(f"Best score: {best_score:.4f}")
        self.logger.info(f"Best params: {best_params}")
        
        return results
    
    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search"""
        from itertools import product
        
        # Define discrete values for each parameter
        lr_values = [1e-4, 5e-4, 1e-3, 5e-3]
        batch_size_values = self.param_space.batch_sizes
        weight_decay_values = [1e-5, 1e-4, 1e-3]
        backbone_values = self.param_space.backbone_names[:3]  # Limit for efficiency
        focal_gamma_values = [0.0, 1.0, 2.0]
        dice_smooth_values = [1e-6, 1e-3, 1e-1]
        
        phase1_epoch_values = [10, 15]
        phase2_epoch_values = [10, 15]
        phase3_epoch_values = [20, 30]
        
        seg_weight_max_values = [0.5, 0.8, 1.0]
        seg_weight_warmup_values = [5, 10]
        
        # Generate all combinations
        combinations = []
        for combo in product(
            lr_values, batch_size_values, weight_decay_values, backbone_values,
            focal_gamma_values, dice_smooth_values, phase1_epoch_values,
            phase2_epoch_values, phase3_epoch_values, seg_weight_max_values,
            seg_weight_warmup_values
        ):
            params = {
                'lr': combo[0],
                'batch_size': combo[1],
                'weight_decay': combo[2],
                'backbone_name': combo[3],
                'focal_gamma': combo[4],
                'dice_smooth': combo[5],
                'phase1_epochs': combo[6],
                'phase2_epochs': combo[7],
                'phase3_epochs': combo[8],
                'seg_weight_max': combo[9],
                'seg_weight_warmup': combo[10]
            }
            combinations.append(params)
        
        return combinations
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run hyperparameter optimization with the specified method"""
        self.start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("SEARCH HYPERPARAMETER OPTIMIZATION STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Method: {self.opt_config.search_method}")
        self.logger.info(f"Number of trials: {self.opt_config.n_trials}")
        self.logger.info(f"Primary metric: {self.opt_config.primary_metric}")
        self.logger.info(f"Optimization direction: {self.opt_config.optimization_direction}")
        self.logger.info("=" * 80)
        
        # Run optimization based on method
        if self.opt_config.search_method.lower() == "optuna":
            results = self.optimize_with_optuna()
        elif self.opt_config.search_method.lower() == "ray":
            results = self.optimize_with_ray()
        elif self.opt_config.search_method.lower() == "grid":
            results = self.optimize_with_grid_search()
        else:
            raise ValueError(f"Unsupported search method: {self.opt_config.search_method}")
        
        # Save comprehensive results
        self._save_optimization_results(results)
        
        total_time = time.time() - self.start_time
        self.logger.info("=" * 80)
        self.logger.info("TARGET HYPERPARAMETER OPTIMIZATION COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Total time: {total_time/3600:.2f} hours")
        self.logger.info(f"Trials completed: {results['n_trials']}")
        self.logger.info(f"Best score: {results['best_score']:.4f}")
        self.logger.info("=" * 80)
        
        return results
    
    def _save_optimization_results(self, results: Dict[str, Any]):
        """Save comprehensive optimization results"""
        # Save main results
        results_file = self.optimization_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save all trial results
        if self.opt_config.save_all_trials:
            all_trials_file = self.optimization_dir / "all_trials.json"
            with open(all_trials_file, 'w') as f:
                json.dump(self.trials_results, f, indent=2, default=str)
        
        # Save best configurations
        if self.trials_results:
            sorted_trials = sorted(
                self.trials_results, 
                key=lambda x: x['score'], 
                reverse=(self.opt_config.optimization_direction == "maximize")
            )
            
            best_trials = sorted_trials[:self.opt_config.save_best_n_models]
            best_trials_file = self.optimization_dir / "best_trials.json"
            with open(best_trials_file, 'w') as f:
                json.dump(best_trials, f, indent=2, default=str)
        
        self.logger.info(f"Optimization results saved to {self.optimization_dir}")
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze optimization results"""
        if not self.trials_results:
            self.logger.warning("No trial results available for analysis")
            return {}
        
        # Performance analysis
        scores = [trial['score'] for trial in self.trials_results]
        
        analysis = {
            'performance_stats': {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'best_score': np.max(scores),
                'worst_score': np.min(scores),
                'median_score': np.median(scores)
            },
            'parameter_importance': self._analyze_parameter_importance(),
            'convergence_analysis': self._analyze_convergence(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save analysis
        analysis_file = self.optimization_dir / "optimization_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze which parameters have the most impact on performance"""
        if len(self.trials_results) < 10:
            return {}
        
        # Simple correlation analysis
        param_importance = {}
        scores = [trial['score'] for trial in self.trials_results]
        
        # Get numerical parameters
        numerical_params = ['lr', 'weight_decay', 'focal_gamma', 'dice_smooth', 
                           'seg_weight_max', 'phase1_epochs', 'phase2_epochs', 'phase3_epochs']
        
        for param in numerical_params:
            param_values = []
            for trial in self.trials_results:
                if param in trial['params']:
                    param_values.append(trial['params'][param])
                else:
                    param_values.append(0)
            
            if len(param_values) == len(scores) and len(set(param_values)) > 1:
                correlation = np.corrcoef(param_values, scores)[0, 1]
                if not np.isnan(correlation):
                    param_importance[param] = abs(correlation)
        
        return param_importance
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence"""
        if len(self.trials_results) < 5:
            return {}
        
        scores = [trial['score'] for trial in self.trials_results]
        
        # Calculate running best
        running_best = []
        current_best = -float('inf') if self.opt_config.optimization_direction == "maximize" else float('inf')
        
        for score in scores:
            if self.opt_config.optimization_direction == "maximize":
                current_best = max(current_best, score)
            else:
                current_best = min(current_best, score)
            running_best.append(current_best)
        
        # Analyze improvement rate
        improvements = []
        for i in range(1, len(running_best)):
            if running_best[i] != running_best[i-1]:
                improvements.append(i)
        
        return {
            'running_best': running_best,
            'improvement_trials': improvements,
            'convergence_rate': len(improvements) / len(scores) if scores else 0,
            'plateaued': len(improvements) == 0 and len(scores) > 10
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if len(self.trials_results) < 10:
            recommendations.append("Increase number of trials for better parameter exploration")
        
        # Analyze parameter ranges
        param_importance = self._analyze_parameter_importance()
        if param_importance:
            most_important = max(param_importance.items(), key=lambda x: x[1])
            recommendations.append(f"Parameter '{most_important[0]}' shows highest correlation with performance")
        
        # Analyze convergence
        convergence = self._analyze_convergence()
        if convergence.get('plateaued', False):
            recommendations.append("Optimization may have plateaued - consider expanding search space")
        
        if convergence.get('convergence_rate', 0) < 0.1:
            recommendations.append("Low improvement rate - consider focusing search around best parameters")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Create base configuration
    base_config = Phase4Config()
    
    # Define parameter space
    param_space = HyperparameterSpace()
    
    # Create optimization configuration
    opt_config = OptimizationConfig(
        search_method="optuna",
        n_trials=20,
        optimization_direction="maximize",
        primary_metric="val_combined_score"
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(base_config, param_space, opt_config)
    
    print("Hyperparameter Optimizer initialized!")
    print(f"Search method: {opt_config.search_method}")
    print(f"Number of trials: {opt_config.n_trials}")
    print(f"Primary metric: {opt_config.primary_metric}")
