"""
Phase 4: Advanced Training Configuration System
Config-driven training with YAML-based hyperparameter management
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import torch


@dataclass
class ProgressiveTrainingConfig:
    """Progressive training phase configuration"""
    epochs: int
    classification_weight: float
    segmentation_weight: float
    segmentation_weight_start: Optional[float] = None
    segmentation_weight_end: Optional[float] = None
    description: str = ""


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    name: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    name: str = "CosineAnnealingLR"
    T_max: int = 30
    eta_min: float = 1e-6
    step_size: int = 30
    gamma: float = 0.1
    patience: int = 10
    factor: float = 0.5
    threshold: float = 1e-4


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration"""
    patience: int = 5
    monitor: str = "val_combined_score"
    mode: str = "max"  # "max" or "min"
    min_delta: float = 1e-4


@dataclass
class HardwareConfig:
    """Hardware configuration"""
    device: str = "cuda"
    batch_size: int = 4
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    tensorboard_log_dir: str = "results/logs"
    log_every_n_steps: int = 50
    save_plots: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "dr-multitask"
    wandb_entity: Optional[str] = None


@dataclass
class CheckpointConfig:
    """Checkpoint configuration"""
    save_dir: str = "results/models"
    save_every_n_epochs: int = 5
    save_best: bool = True
    monitor: str = "val_combined_score"
    keep_top_k: int = 3


@dataclass
class ModelConfig:
    """Model configuration"""
    backbone_name: str = "efficientnetv2_s"
    num_classes: int = 5
    pretrained: bool = True
    use_skip_connections: bool = True
    use_advanced_decoder: bool = False
    freeze_early_layers: bool = False
    dropout: float = 0.3


@dataclass
class AdvancedTrainingConfig:
    """Advanced training configuration for Phase 4"""
    
    # Experiment tracking
    experiment_name: str = field(default_factory=lambda: f"dr_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Progressive training phases
    phase1: ProgressiveTrainingConfig = field(default_factory=lambda: ProgressiveTrainingConfig(
        epochs=10, classification_weight=1.0, segmentation_weight=0.0,
        description="Classification warmup - learn DR grading first"
    ))
    phase2: ProgressiveTrainingConfig = field(default_factory=lambda: ProgressiveTrainingConfig(
        epochs=20, classification_weight=1.0, segmentation_weight=0.5,
        segmentation_weight_start=0.0, segmentation_weight_end=0.5,
        description="Gradual multi-task introduction"
    ))
    phase3: ProgressiveTrainingConfig = field(default_factory=lambda: ProgressiveTrainingConfig(
        epochs=20, classification_weight=1.0, segmentation_weight=0.8,
        description="Full multi-task optimization"
    ))
    
    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Training parameters
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    val_every_n_epochs: int = 1
    
    # Early stopping
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    
    # Hardware
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # Logging and monitoring
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Loss function parameters
    focal_gamma: float = 2.0
    dice_smooth: float = 1e-5
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    
    # Data pipeline
    image_size: int = 512
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Set total epochs
        self.total_epochs = self.phase1.epochs + self.phase2.epochs + self.phase3.epochs
        
        # Update scheduler T_max if using CosineAnnealingLR
        if self.scheduler.name == "CosineAnnealingLR":
            self.scheduler.T_max = self.total_epochs
        
        # Create experiment directory
        self.experiment_dir = Path("experiments") / self.experiment_name
        
        # Update paths to be experiment-specific
        self.logging.tensorboard_log_dir = str(self.experiment_dir / "logs")
        self.checkpoints.save_dir = str(self.experiment_dir / "checkpoints")
        
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation"""
        return self.hardware.batch_size * self.gradient_accumulation_steps
    
    def get_current_phase(self, epoch: int) -> tuple[str, ProgressiveTrainingConfig]:
        """Get current training phase based on epoch"""
        if epoch < self.phase1.epochs:
            return "phase1", self.phase1
        elif epoch < self.phase1.epochs + self.phase2.epochs:
            return "phase2", self.phase2
        else:
            return "phase3", self.phase3
    
    def get_segmentation_weight(self, epoch: int) -> float:
        """Calculate segmentation weight for current epoch"""
        phase_name, phase = self.get_current_phase(epoch)
        
        if phase_name == "phase1":
            return phase.segmentation_weight
        elif phase_name == "phase2":
            # Linear interpolation for phase 2
            phase_start_epoch = self.phase1.epochs
            phase_progress = (epoch - phase_start_epoch) / self.phase2.epochs
            start_weight = phase.segmentation_weight_start or 0.0
            end_weight = phase.segmentation_weight_end or phase.segmentation_weight
            return start_weight + phase_progress * (end_weight - start_weight)
        else:
            return phase.segmentation_weight
    
    def save_config(self, save_path: Optional[str] = None) -> str:
        """Save configuration to YAML file"""
        if save_path is None:
            save_path = self.experiment_dir / "config.yaml"
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save
        config_dict = asdict(self)
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        return str(save_path)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AdvancedTrainingConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested dataclasses
        if 'phase1' in config_dict:
            config_dict['phase1'] = ProgressiveTrainingConfig(**config_dict['phase1'])
        if 'phase2' in config_dict:
            config_dict['phase2'] = ProgressiveTrainingConfig(**config_dict['phase2'])
        if 'phase3' in config_dict:
            config_dict['phase3'] = ProgressiveTrainingConfig(**config_dict['phase3'])
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'optimizer' in config_dict:
            config_dict['optimizer'] = OptimizerConfig(**config_dict['optimizer'])
        if 'scheduler' in config_dict:
            config_dict['scheduler'] = SchedulerConfig(**config_dict['scheduler'])
        if 'early_stopping' in config_dict:
            config_dict['early_stopping'] = EarlyStoppingConfig(**config_dict['early_stopping'])
        if 'hardware' in config_dict:
            config_dict['hardware'] = HardwareConfig(**config_dict['hardware'])
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        if 'checkpoints' in config_dict:
            config_dict['checkpoints'] = CheckpointConfig(**config_dict['checkpoints'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AdvancedTrainingConfig':
        """Create configuration from dictionary"""
        # Process nested configurations
        if 'phase1' in config_dict:
            phase1_dict = config_dict['phase1'].copy()
            # Remove unknown fields
            phase1_dict.pop('name', None)
            phase1_dict.pop('freeze_segmentation_head', None)
            config_dict['phase1'] = ProgressiveTrainingConfig(**phase1_dict)
        if 'phase2' in config_dict:
            phase2_dict = config_dict['phase2'].copy()
            # Remove unknown fields and set segmentation_weight
            phase2_dict.pop('name', None)
            phase2_dict.pop('freeze_segmentation_head', None)
            if 'segmentation_weight' not in phase2_dict:
                phase2_dict['segmentation_weight'] = phase2_dict.get('segmentation_weight_end', 0.5)
            config_dict['phase2'] = ProgressiveTrainingConfig(**phase2_dict)
        if 'phase3' in config_dict:
            phase3_dict = config_dict['phase3'].copy()
            # Remove unknown fields
            phase3_dict.pop('name', None)
            phase3_dict.pop('enable_advanced_augmentation', None)
            config_dict['phase3'] = ProgressiveTrainingConfig(**phase3_dict)
        if 'model' in config_dict:
            model_dict = config_dict['model'].copy()
            # Map fields from YAML to ModelConfig
            if 'backbone_name' not in model_dict and 'backbone_name' in model_dict:
                model_dict['backbone_name'] = model_dict.get('backbone_name', 'efficientnetv2_s')
            # Remove unknown fields
            model_dict.pop('use_advanced_decoder', None)
            model_dict.pop('freeze_early_layers', None)
            model_dict.pop('dropout_rate', None)  # Should be 'dropout'
            if 'dropout_rate' in config_dict['model']:
                model_dict['dropout'] = config_dict['model']['dropout_rate']
            config_dict['model'] = ModelConfig(**model_dict)
        if 'optimizer' in config_dict:
            optimizer_dict = config_dict['optimizer'].copy()
            # Remove unknown fields
            optimizer_dict.pop('amsgrad', None)
            # Ensure proper type conversion for numeric values
            if 'eps' in optimizer_dict and isinstance(optimizer_dict['eps'], str):
                optimizer_dict['eps'] = float(optimizer_dict['eps'])
            if 'lr' in optimizer_dict and isinstance(optimizer_dict['lr'], str):
                optimizer_dict['lr'] = float(optimizer_dict['lr'])
            if 'weight_decay' in optimizer_dict and isinstance(optimizer_dict['weight_decay'], str):
                optimizer_dict['weight_decay'] = float(optimizer_dict['weight_decay'])
            config_dict['optimizer'] = OptimizerConfig(**optimizer_dict)
        if 'scheduler' in config_dict:
            config_dict['scheduler'] = SchedulerConfig(**config_dict['scheduler'])
        if 'early_stopping' in config_dict:
            config_dict['early_stopping'] = EarlyStoppingConfig(**config_dict['early_stopping'])
        if 'hardware' in config_dict:
            hardware_dict = config_dict['hardware'].copy()
            # Map fields correctly
            if 'mixed_precision' in hardware_dict:
                hardware_dict.pop('mixed_precision')  # This goes to main config
            if 'gradient_accumulation_steps' in hardware_dict:
                hardware_dict.pop('gradient_accumulation_steps')  # This goes to main config
            if 'max_grad_norm' in hardware_dict:
                hardware_dict.pop('max_grad_norm')  # This goes to main config
            config_dict['hardware'] = HardwareConfig(**hardware_dict)
        if 'logging' in config_dict:
            logging_dict = config_dict['logging'].copy()
            # Remove unknown fields
            logging_dict.pop('level', None)
            logging_dict.pop('save_plots', None)
            logging_dict.pop('plot_every_n_epochs', None)
            logging_dict.pop('log_every_n_steps', None)
            logging_dict.pop('wandb_tags', None)
            # Map fields correctly
            if 'wandb_enabled' in logging_dict:
                logging_dict['wandb_enabled'] = logging_dict.get('wandb_enabled', False)
            if 'wandb_project' in logging_dict:
                logging_dict['wandb_project'] = logging_dict.get('wandb_project', 'dr-multitask')
            config_dict['logging'] = LoggingConfig(**logging_dict)
        if 'checkpoints' in config_dict:
            checkpoints_dict = config_dict['checkpoints'].copy()
            # Remove unknown fields
            checkpoints_dict.pop('mode', None)
            checkpoints_dict.pop('save_optimizer', None)
            checkpoints_dict.pop('save_scheduler', None)
            config_dict['checkpoints'] = CheckpointConfig(**checkpoints_dict)
        
        # Handle top-level fields that don't belong to nested configs
        filtered_dict = {}
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        
        for key, value in config_dict.items():
            if key in valid_fields:
                filtered_dict[key] = value
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_base_config(cls, base_config_path: str = "configs/base_config.yaml") -> 'AdvancedTrainingConfig':
        """Create from existing base configuration"""
        if not Path(base_config_path).exists():
            logging.warning(f"Base config {base_config_path} not found, using defaults")
            return cls()
        
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Extract relevant sections
        training_config = base_config.get('training', {})
        model_config = base_config.get('model', {})
        hardware_config = base_config.get('hardware', {})
        logging_config = base_config.get('logging', {})
        
        # Map base config to new structure
        config_kwargs = {}
        
        # Progressive training
        if 'progressive' in training_config:
            prog_config = training_config['progressive']
            if 'phase1' in prog_config:
                config_kwargs['phase1'] = ProgressiveTrainingConfig(**prog_config['phase1'])
            if 'phase2' in prog_config:
                config_kwargs['phase2'] = ProgressiveTrainingConfig(**prog_config['phase2'])
            if 'phase3' in prog_config:
                config_kwargs['phase3'] = ProgressiveTrainingConfig(**prog_config['phase3'])
        
        # Model config
        if model_config:
            backbone_config = model_config.get('backbone', {})
            cls_head_config = model_config.get('classification_head', {})
            
            config_kwargs['model'] = ModelConfig(
                backbone_name=backbone_config.get('name', 'efficientnetv2_s'),
                num_classes=cls_head_config.get('num_classes', 5),
                pretrained=backbone_config.get('pretrained', True),
                dropout=cls_head_config.get('dropout', 0.3)
            )
        
        # Optimizer config
        if 'optimizer' in training_config:
            opt_config = training_config['optimizer']
            config_kwargs['optimizer'] = OptimizerConfig(**opt_config)
        
        # Scheduler config
        if 'scheduler' in training_config:
            sched_config = training_config['scheduler']
            config_kwargs['scheduler'] = SchedulerConfig(**sched_config)
        
        # Hardware config
        if hardware_config:
            config_kwargs['hardware'] = HardwareConfig(**hardware_config)
        
        # Other training parameters
        config_kwargs.update({
            'mixed_precision': training_config.get('mixed_precision', True),
            'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 4),
            'max_grad_norm': training_config.get('max_grad_norm', 1.0)
        })
        
        # Early stopping
        if 'early_stopping' in training_config:
            config_kwargs['early_stopping'] = EarlyStoppingConfig(**training_config['early_stopping'])
        
        # Logging and checkpoints
        if logging_config:
            tb_config = logging_config.get('tensorboard', {})
            wandb_config = logging_config.get('wandb', {})
            cp_config = logging_config.get('checkpoints', {})
            
            config_kwargs['logging'] = LoggingConfig(
                tensorboard_log_dir=tb_config.get('log_dir', 'results/logs'),
                log_every_n_steps=tb_config.get('log_every_n_steps', 50),
                wandb_enabled=wandb_config.get('enabled', False),
                wandb_project=wandb_config.get('project', 'dr-multitask'),
                wandb_entity=wandb_config.get('entity')
            )
            
            config_kwargs['checkpoints'] = CheckpointConfig(
                save_dir=cp_config.get('save_dir', 'results/models'),
                save_every_n_epochs=cp_config.get('save_every_n_epochs', 5),
                save_best=cp_config.get('save_best', True),
                monitor=cp_config.get('monitor', 'val_combined_score')
            )
        
        return cls(**config_kwargs)


def setup_reproducibility(config: AdvancedTrainingConfig):
    """Setup reproducibility settings"""
    import random
    import numpy as np
    
    # Set seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Set deterministic behavior
    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = config.benchmark


def create_experiment_directory(config: AdvancedTrainingConfig) -> Path:
    """Create experiment directory structure"""
    exp_dir = Path(config.experiment_dir)
    
    # Create subdirectories
    subdirs = [
        "checkpoints",
        "logs",
        "plots",
        "configs",
        "results"
    ]
    
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save_config(exp_dir / "configs" / "training_config.yaml")
    
    return exp_dir


def load_training_config(config_path: Optional[str] = None) -> AdvancedTrainingConfig:
    """Load training configuration"""
    if config_path and Path(config_path).exists():
        return AdvancedTrainingConfig.from_yaml(config_path)
    elif Path("configs/base_config.yaml").exists():
        return AdvancedTrainingConfig.from_base_config()
    else:
        logging.warning("No config file found, using default configuration")
        return AdvancedTrainingConfig()


# Example usage and testing
if __name__ == "__main__":
    # Create default config
    config = AdvancedTrainingConfig()
    
    print("=== Phase 4 Training Configuration ===")
    print(f"Experiment: {config.experiment_name}")
    print(f"Total epochs: {config.total_epochs}")
    print(f"Effective batch size: {config.effective_batch_size}")
    print(f"Device: {config.hardware.device}")
    print(f"Mixed precision: {config.mixed_precision}")
    
    print("\n=== Progressive Training Phases ===")
    print(f"Phase 1: {config.phase1.epochs} epochs - {config.phase1.description}")
    print(f"Phase 2: {config.phase2.epochs} epochs - {config.phase2.description}")
    print(f"Phase 3: {config.phase3.epochs} epochs - {config.phase3.description}")
    
    # Test segmentation weight progression
    print("\n=== Segmentation Weight Progression ===")
    for epoch in [0, 5, 15, 25, 35, 45]:
        weight = config.get_segmentation_weight(epoch)
        phase_name, _ = config.get_current_phase(epoch)
        print(f"Epoch {epoch:2d} ({phase_name}): seg_weight = {weight:.3f}")
    
    # Save config
    config_path = config.save_config()
    print(f"\nConfig saved to: {config_path}")
    
    # Test loading
    loaded_config = AdvancedTrainingConfig.from_yaml(config_path)
    print(f"Config loaded successfully: {loaded_config.experiment_name}")
