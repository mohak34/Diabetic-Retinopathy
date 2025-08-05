"""
Phase 4: Robust Training Configuration
Error-free configuration system with proper type handling and validation.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Hardware and performance configuration"""
    device: str = "auto"  # auto, cuda, cpu
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    max_memory_usage: float = 0.8  # Fraction of GPU memory to use


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8

    # Alias for backward compatibility
    @property
    def lr(self) -> float:
        return self.learning_rate
    
    @lr.setter
    def lr(self, value: float):
        self.learning_rate = value


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    backbone_name: str = "resnet50"
    use_skip_connections: bool = True
    use_advanced_decoder: bool = True
    num_classes: int = 5
    segmentation_classes: int = 1


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    name: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    max_epochs: int = 50
    min_lr: float = 1e-6
    patience: int = 10  # For plateau scheduler


@dataclass
class LossConfig:
    """Loss function configuration"""
    classification_weight: float = 1.0
    segmentation_weight: float = 0.5
    use_focal_cls: bool = True
    use_kappa_cls: bool = True
    use_combined_seg: bool = True
    focal_gamma: float = 2.0
    kappa_weight: float = 0.1
    dice_smooth: float = 1e-6


@dataclass
class ProgressiveTrainingConfig:
    """Progressive training phase configuration"""
    phase1_epochs: int = 15  # Classification only
    phase2_epochs: int = 10  # Gradual segmentation introduction
    phase3_epochs: int = 25  # Full multi-task training
    
    @property
    def total_epochs(self) -> int:
        return self.phase1_epochs + self.phase2_epochs + self.phase3_epochs


@dataclass
class CheckpointConfig:
    """Checkpointing and saving configuration"""
    save_every: int = 5  # Save checkpoint every N epochs
    keep_best: bool = True
    keep_last: int = 3  # Keep last N checkpoints
    monitor_metric: str = "combined_score"  # combined_score, kappa, dice
    mode: str = "max"  # max or min


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration"""
    enabled: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    monitor_metric: str = "combined_score"
    mode: str = "max"


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_every: int = 10  # Log every N batches
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "diabetic-retinopathy"
    save_images: bool = True
    max_images: int = 8


@dataclass
class Phase4Config:
    """Complete Phase 4 training configuration"""
    # Basic settings
    experiment_name: str = "phase4_training"
    seed: int = 42
    
    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    progressive: ProgressiveTrainingConfig = field(default_factory=ProgressiveTrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Additional hyperparameter optimization attributes
    segmentation_weight_max: float = 1.0
    segmentation_weight_warmup_epochs: int = 10
    
    # Experiment directory (computed)
    experiment_dir: Optional[str] = None
    
    # Data paths
    data_root: str = "dataset/processed"
    train_split: str = "dataset/splits/train.json"
    val_split: str = "dataset/splits/val.json"
    test_split: str = "dataset/splits/test.json"
    
    # Output paths
    output_dir: str = "experiments"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    def __post_init__(self):
        """Post-initialization validation"""
        # Ensure scheduler max_epochs matches progressive training
        self.scheduler.max_epochs = self.progressive.total_epochs
        
        # Validate paths exist
        if not os.path.exists(self.data_root):
            logger.warning(f"Data root path does not exist: {self.data_root}")
    
    # Backward compatibility properties
    @property
    def focal_gamma(self) -> float:
        return self.loss.focal_gamma
    
    @focal_gamma.setter
    def focal_gamma(self, value: float):
        self.loss.focal_gamma = value
    
    @property
    def dice_smooth(self) -> float:
        return self.loss.dice_smooth
    
    @dice_smooth.setter
    def dice_smooth(self, value: float):
        self.loss.dice_smooth = value
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Phase4Config':
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert nested dictionaries to dataclass instances
            config = cls()
            
            # Update fields if they exist in the YAML
            for key, value in config_dict.items():
                if hasattr(config, key):
                    if key == 'model' and isinstance(value, dict):
                        config.model = ModelConfig(**value)
                    elif key == 'hardware' and isinstance(value, dict):
                        config.hardware = HardwareConfig(**value)
                    elif key == 'optimizer' and isinstance(value, dict):
                        config.optimizer = OptimizerConfig(**value)
                    elif key == 'scheduler' and isinstance(value, dict):
                        config.scheduler = SchedulerConfig(**value)
                    elif key == 'loss' and isinstance(value, dict):
                        config.loss = LossConfig(**value)
                    elif key == 'progressive' and isinstance(value, dict):
                        config.progressive = ProgressiveTrainingConfig(**value)
                    elif key == 'checkpoint' and isinstance(value, dict):
                        config.checkpoint = CheckpointConfig(**value)
                    elif key == 'early_stopping' and isinstance(value, dict):
                        config.early_stopping = EarlyStoppingConfig(**value)
                    elif key == 'logging' and isinstance(value, dict):
                        config.logging = LoggingConfig(**value)
                    else:
                        setattr(config, key, value)
            
            config.__post_init__()
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML file"""
        try:
            # Convert dataclass to dictionary
            config_dict = {
                'experiment_name': self.experiment_name,
                'seed': self.seed,
                'model': {
                    'backbone_name': self.model.backbone_name,
                    'use_skip_connections': self.model.use_skip_connections,
                    'use_advanced_decoder': self.model.use_advanced_decoder,
                    'num_classes': self.model.num_classes,
                    'segmentation_classes': self.model.segmentation_classes
                },
                'hardware': {
                    'device': self.hardware.device,
                    'batch_size': self.hardware.batch_size,
                    'num_workers': self.hardware.num_workers,
                    'pin_memory': self.hardware.pin_memory,
                    'mixed_precision': self.hardware.mixed_precision,
                    'gradient_accumulation_steps': self.hardware.gradient_accumulation_steps,
                    'max_memory_usage': self.hardware.max_memory_usage
                },
                'optimizer': {
                    'name': self.optimizer.name,
                    'learning_rate': self.optimizer.learning_rate,
                    'weight_decay': self.optimizer.weight_decay,
                    'betas': self.optimizer.betas,
                    'eps': self.optimizer.eps
                },
                'scheduler': {
                    'name': self.scheduler.name,
                    'warmup_epochs': self.scheduler.warmup_epochs,
                    'max_epochs': self.scheduler.max_epochs,
                    'min_lr': self.scheduler.min_lr,
                    'patience': self.scheduler.patience
                },
                'loss': {
                    'classification_weight': self.loss.classification_weight,
                    'segmentation_weight': self.loss.segmentation_weight,
                    'use_focal_cls': self.loss.use_focal_cls,
                    'use_kappa_cls': self.loss.use_kappa_cls,
                    'use_combined_seg': self.loss.use_combined_seg,
                    'focal_gamma': self.loss.focal_gamma,
                    'kappa_weight': self.loss.kappa_weight,
                    'dice_smooth': self.loss.dice_smooth
                },
                'progressive': {
                    'phase1_epochs': self.progressive.phase1_epochs,
                    'phase2_epochs': self.progressive.phase2_epochs,
                    'phase3_epochs': self.progressive.phase3_epochs
                },
                'checkpoint': {
                    'save_every': self.checkpoint.save_every,
                    'keep_best': self.checkpoint.keep_best,
                    'keep_last': self.checkpoint.keep_last,
                    'monitor_metric': self.checkpoint.monitor_metric,
                    'mode': self.checkpoint.mode
                },
                'early_stopping': {
                    'enabled': self.early_stopping.enabled,
                    'patience': self.early_stopping.patience,
                    'min_delta': self.early_stopping.min_delta,
                    'monitor_metric': self.early_stopping.monitor_metric,
                    'mode': self.early_stopping.mode
                },
                'logging': {
                    'log_every': self.logging.log_every,
                    'use_tensorboard': self.logging.use_tensorboard,
                    'use_wandb': self.logging.use_wandb,
                    'wandb_project': self.logging.wandb_project,
                    'save_images': self.logging.save_images,
                    'max_images': self.logging.max_images
                },
                'segmentation_weight_max': self.segmentation_weight_max,
                'segmentation_weight_warmup_epochs': self.segmentation_weight_warmup_epochs,
                'data_root': self.data_root,
                'train_split': self.train_split,
                'val_split': self.val_split,
                'test_split': self.test_split,
                'output_dir': self.output_dir,
                'model_dir': self.model_dir,
                'log_dir': self.log_dir
            }
            
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if there's a directory component
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {output_path}: {e}")
    
    def create_smoke_test_config(self) -> 'Phase4Config':
        """Create a minimal configuration for smoke testing"""
        config = Phase4Config()
        
        # Minimal epochs for quick testing
        config.progressive.phase1_epochs = 1
        config.progressive.phase2_epochs = 1 
        config.progressive.phase3_epochs = 0
        
        # Smaller batch size and less frequent logging
        config.hardware.batch_size = 2
        config.hardware.gradient_accumulation_steps = 1
        config.logging.log_every = 1
        config.checkpoint.save_every = 1
        
        # Disable some features for faster testing
        config.early_stopping.enabled = False
        config.logging.use_wandb = False
        config.logging.save_images = False
        
        # Quick optimizer settings
        config.optimizer.learning_rate = 1e-3
        config.scheduler.warmup_epochs = 0
        
        config.experiment_name = "smoke_test"
        
        return config


def setup_reproducibility(seed: int = 42):
    """Setup reproducible training environment"""
    import random
    import numpy as np
    import torch
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Reproducibility setup complete with seed: {seed}")


def get_device(device_str: str = "auto") -> str:
    """Get the appropriate device for training"""
    import torch
    
    if device_str == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
    else:
        device = device_str
        logger.info(f"Using specified device: {device}")
    
    return device


if __name__ == "__main__":
    # Test configuration
    config = Phase4Config()
    
    # Save default config
    config.to_yaml("configs/phase4_default.yaml")
    
    # Create smoke test config
    smoke_config = config.create_smoke_test_config()
    smoke_config.to_yaml("configs/phase4_smoke_test.yaml")
    
    print("✅ Phase 4 configuration system ready!")
