"""
Phase 4: Advanced Multi-Task Training Infrastructure
GPU-optimized, config-driven training system with comprehensive monitoring
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Local imports
from .config import AdvancedTrainingConfig, setup_reproducibility, create_experiment_directory
from .metrics import AdvancedMetricsCollector, MetricResults
from ..models.multi_task_model import MultiTaskRetinaModel, create_multi_task_model
from ..training.losses import MultiTaskLoss


class Phase4Trainer:
    """
    Phase 4: Advanced Multi-Task Training Infrastructure
    
    Features:
    - GPU-optimized training with mixed precision
    - Progressive 3-phase training strategy
    - Comprehensive metrics collection
    - Config-driven hyperparameter management
    - Robust checkpointing and resuming
    - Real-time monitoring with TensorBoard
    - Early stopping with multiple criteria
    - Memory-efficient batch processing
    - Gradient accumulation for large effective batch sizes
    """
    
    def __init__(self, 
                 config: AdvancedTrainingConfig,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.config = config
        self.device = device or torch.device(config.hardware.device if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.logger = logger or self._setup_logger()
        
        # Setup reproducibility
        setup_reproducibility(config)
        
        # Create experiment directory
        self.experiment_dir = create_experiment_directory(config)
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        
        # Initialize training components
        self.model: Optional[MultiTaskRetinaModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.loss_fn: Optional[MultiTaskLoss] = None
        self.scaler: Optional[GradScaler] = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric_value = -float('inf')
        self.patience_counter = 0
        self.current_phase = "phase1"
        
        # Metrics collection
        self.metrics_collector = AdvancedMetricsCollector(
            num_classes=config.model.num_classes,
            device=str(self.device)
        )
        
        # Training history
        self.training_history = {
            'epoch': [],
            'phase': [],
            'train_loss': [],
            'val_loss': [],
            'train_cls_loss': [],
            'val_cls_loss': [],
            'train_seg_loss': [],
            'val_seg_loss': [],
            'segmentation_weight': [],
            'learning_rate': [],
            'gpu_memory_mb': [],
            'epoch_time_minutes': []
        }
        
        # Setup TensorBoard
        if config.logging.save_plots:
            self.writer = SummaryWriter(
                log_dir=config.logging.tensorboard_log_dir,
                comment=f"_{config.experiment_name}"
            )
            self.logger.info(f"TensorBoard logs: {config.logging.tensorboard_log_dir}")
        else:
            self.writer = None
        
        # Setup W&B if enabled
        self.wandb_run = None
        if config.logging.wandb_enabled:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.logging.wandb_project,
                    entity=config.logging.wandb_entity,
                    name=config.experiment_name,
                    config=config.__dict__
                )
                self.logger.info("W&B logging enabled")
            except ImportError:
                self.logger.warning("W&B not installed, skipping W&B logging")
        
        self.logger.info(f"Phase 4 Trainer initialized on {self.device}")
        self.logger.info(f"Configuration: {config.experiment_name}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('Phase4Trainer')
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
            log_file = self.config.experiment_dir / "training.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def setup_model(self) -> MultiTaskRetinaModel:
        """Setup multi-task model with GPU optimization"""
        self.logger.info("Setting up multi-task model...")
        
        # Create model
        self.model = create_multi_task_model(
            backbone_name=self.config.model.backbone_name,
            num_classes_cls=self.config.model.num_classes,
            pretrained=self.config.model.pretrained,
            use_skip_connections=self.config.model.use_skip_connections,
            use_advanced_decoder=self.config.model.use_advanced_decoder
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Model analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {self.config.model.backbone_name}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Memory estimation
        if self.device.type == 'cuda':
            try:
                memory_mb = self.model.estimate_memory_usage(self.config.hardware.batch_size)
                self.logger.info(f"Estimated memory usage: {memory_mb:.1f} MB")
                
                # Check if memory usage is within limits
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                memory_gb = memory_mb / 1000
                if memory_gb > gpu_memory_gb * 0.8:
                    self.logger.warning(f"Memory usage ({memory_gb:.1f}GB) may exceed GPU capacity ({gpu_memory_gb:.1f}GB)")
                    self.logger.warning("Consider reducing batch size or enabling mixed precision")
                
            except Exception as e:
                self.logger.warning(f"Could not estimate memory usage: {e}")
        
        # Freeze early layers if specified
        if self.config.model.freeze_early_layers:
            self._freeze_early_layers()
        
        return self.model
    
    def _freeze_early_layers(self):
        """Freeze early layers of the backbone"""
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'model'):
            # Freeze first few layers of EfficientNet
            for i, (name, param) in enumerate(self.model.backbone.model.named_parameters()):
                if i < 10:  # Freeze first 10 layers
                    param.requires_grad = False
                    
            frozen_params = sum(1 for p in self.model.parameters() if not p.requires_grad)
            self.logger.info(f"Frozen {frozen_params} parameters in early layers")
    
    def setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with config-driven parameters"""
        self.logger.info(f"Setting up optimizer: {self.config.optimizer.name}")
        
        if self.config.optimizer.name.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps
            )
        elif self.config.optimizer.name.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps
            )
        elif self.config.optimizer.name.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer.name}")
        
        self.logger.info(f"Optimizer configured: lr={self.config.optimizer.lr}, weight_decay={self.config.optimizer.weight_decay}")
        return self.optimizer
    
    def setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        self.logger.info(f"Setting up scheduler: {self.config.scheduler.name}")
        
        if self.config.scheduler.name.lower() == "cosineannealinglr":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler.T_max,
                eta_min=self.config.scheduler.eta_min
            )
        elif self.config.scheduler.name.lower() == "steplr":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler.step_size,
                gamma=self.config.scheduler.gamma
            )
        elif self.config.scheduler.name.lower() == "reducelronplateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.scheduler.factor,
                patience=self.config.scheduler.patience,
                threshold=self.config.scheduler.threshold,
                verbose=True
            )
        elif self.config.scheduler.name.lower() == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler.name}")
        
        return self.scheduler
    
    def setup_loss_function(self) -> MultiTaskLoss:
        """Setup multi-task loss function"""
        self.loss_fn = MultiTaskLoss(
            classification_weight=1.0,  # Will be updated per phase
            segmentation_weight=0.0,    # Will be updated per epoch
            focal_gamma=self.config.focal_gamma
        )
        
        self.logger.info("Multi-task loss function configured")
        return self.loss_fn
    
    def setup_mixed_precision(self):
        """Setup mixed precision training"""
        if self.config.mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            if self.config.mixed_precision:
                self.logger.warning("Mixed precision requested but CUDA not available")
    
    def get_current_phase_info(self, epoch: int) -> Tuple[str, Dict[str, Any]]:
        """Get current training phase information"""
        phase_name, phase_config = self.config.get_current_phase(epoch)
        seg_weight = self.config.get_segmentation_weight(epoch)
        
        return phase_name, {
            'config': phase_config,
            'segmentation_weight': seg_weight,
            'classification_weight': phase_config.classification_weight
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with progressive training strategy"""
        self.model.train()
        
        # Get current phase information
        phase_name, phase_info = self.get_current_phase_info(epoch)
        seg_weight = phase_info['segmentation_weight']
        cls_weight = phase_info['classification_weight']
        
        # Update loss function weights
        self.loss_fn.classification_weight = cls_weight
        self.loss_fn.segmentation_weight = seg_weight
        
        # Reset metrics
        self.metrics_collector.reset()
        
        # Training metrics
        total_loss = 0.0
        cls_loss_total = 0.0
        seg_loss_total = 0.0
        num_batches = len(train_loader)
        
        # Progress bar
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{self.config.total_epochs} [{phase_name}]",
            leave=False
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Handle different batch formats
            if len(batch_data) == 3:
                images, cls_labels, seg_masks = batch_data
            else:
                images, cls_labels = batch_data
                seg_masks = None
            
            # Move to device
            images = images.to(self.device, non_blocking=True)
            cls_labels = cls_labels.to(self.device, non_blocking=True)
            if seg_masks is not None:
                seg_masks = seg_masks.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    cls_logits, seg_preds = self.model(images)
                    
                    # Handle case where no segmentation masks available
                    if seg_masks is not None:
                        loss_dict = self.loss_fn(cls_logits, seg_preds, cls_labels, seg_masks, epoch)
                        loss = loss_dict['total']
                        cls_loss = loss_dict['cls_weighted']
                        seg_loss = loss_dict['seg_weighted']
                    else:
                        # Classification only
                        cls_loss = self.loss_fn.classification_loss(cls_logits, cls_labels)
                        seg_loss = torch.tensor(0.0, device=self.device)
                        loss = cls_loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                cls_logits, seg_preds = self.model(images)
                
                if seg_masks is not None:
                    loss_dict = self.loss_fn(cls_logits, seg_preds, cls_labels, seg_masks, epoch)
                    loss = loss_dict['total']
                    cls_loss = loss_dict['cls_weighted']
                    seg_loss = loss_dict['seg_weighted']
                else:
                    cls_loss = self.loss_fn.classification_loss(cls_logits, cls_labels)
                    seg_loss = torch.tensor(0.0, device=self.device)
                    loss = cls_loss
                
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation and optimization step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step with scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            with torch.no_grad():
                self.metrics_collector.update_classification(cls_logits, cls_labels)
                if seg_masks is not None and seg_weight > 0:
                    self.metrics_collector.update_segmentation(seg_preds, seg_masks)
            
            # Accumulate losses
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            cls_loss_total += cls_loss.item()
            seg_loss_total += seg_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'ClsLoss': f'{cls_loss.item():.4f}',
                'SegLoss': f'{seg_loss.item():.4f}',
                'SegW': f'{seg_weight:.2f}',
                'Phase': phase_name
            })
            
            # Log to TensorBoard every N steps
            if (self.writer and self.global_step % self.config.logging.log_every_n_steps == 0):
                self._log_training_step(loss.item(), cls_loss.item(), seg_loss.item(), seg_weight)
        
        # Compute epoch metrics
        metrics_results = self.metrics_collector.compute()
        
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'cls_loss': cls_loss_total / num_batches,
            'seg_loss': seg_loss_total / num_batches,
            'cls_accuracy': metrics_results.classification.get('accuracy', 0.0),
            'cls_kappa': metrics_results.classification.get('kappa', 0.0),
            'seg_dice': metrics_results.segmentation.get('dice', 0.0),
            'seg_iou': metrics_results.segmentation.get('iou', 0.0),
            'segmentation_weight': seg_weight,
            'phase': phase_name
        }
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        # Reset metrics
        self.metrics_collector.reset()
        
        total_loss = 0.0
        cls_loss_total = 0.0
        seg_loss_total = 0.0
        num_batches = len(val_loader)
        
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_data in progress_bar:
                # Handle different batch formats
                if len(batch_data) == 3:
                    images, cls_labels, seg_masks = batch_data
                else:
                    images, cls_labels = batch_data
                    seg_masks = None
                
                # Move to device
                images = images.to(self.device, non_blocking=True)
                cls_labels = cls_labels.to(self.device, non_blocking=True)
                if seg_masks is not None:
                    seg_masks = seg_masks.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        cls_logits, seg_preds = self.model(images)
                        if seg_masks is not None:
                            loss_dict = self.loss_fn(cls_logits, seg_preds, cls_labels, seg_masks, epoch)
                            loss = loss_dict['total']
                            cls_loss = loss_dict['cls_weighted']
                            seg_loss = loss_dict['seg_weighted']
                        else:
                            cls_loss = self.loss_fn.classification_loss(cls_logits, cls_labels)
                            seg_loss = torch.tensor(0.0, device=self.device)
                            loss = cls_loss
                else:
                    cls_logits, seg_preds = self.model(images)
                    if seg_masks is not None:
                        loss_dict = self.loss_fn(cls_logits, seg_preds, cls_labels, seg_masks, epoch)
                        loss = loss_dict['total']
                        cls_loss = loss_dict['cls_weighted']
                        seg_loss = loss_dict['seg_weighted']
                    else:
                        cls_loss = self.loss_fn.classification_loss(cls_logits, cls_labels)
                        seg_loss = torch.tensor(0.0, device=self.device)
                        loss = cls_loss
                
                # Update metrics
                self.metrics_collector.update_classification(cls_logits, cls_labels)
                if seg_masks is not None and self.loss_fn.segmentation_weight > 0:
                    self.metrics_collector.update_segmentation(seg_preds, seg_masks)
                
                # Accumulate losses
                total_loss += loss.item()
                cls_loss_total += cls_loss.item()
                seg_loss_total += seg_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'ClsLoss': f'{cls_loss.item():.4f}',
                    'SegLoss': f'{seg_loss.item():.4f}'
                })
        
        # Compute epoch metrics
        metrics_results = self.metrics_collector.compute()
        
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'cls_loss': cls_loss_total / num_batches,
            'seg_loss': seg_loss_total / num_batches,
            'cls_accuracy': metrics_results.classification.get('accuracy', 0.0),
            'cls_kappa': metrics_results.classification.get('kappa', 0.0),
            'seg_dice': metrics_results.segmentation.get('dice', 0.0),
            'seg_iou': metrics_results.segmentation.get('iou', 0.0),
            'combined_score': metrics_results.combined.get('combined_score', 0.0)
        }
        
        return epoch_metrics
    
    def _log_training_step(self, loss: float, cls_loss: float, seg_loss: float, seg_weight: float):
        """Log training step to TensorBoard"""
        if self.writer:
            self.writer.add_scalar('Train/StepLoss', loss, self.global_step)
            self.writer.add_scalar('Train/StepClassificationLoss', cls_loss, self.global_step)
            self.writer.add_scalar('Train/StepSegmentationLoss', seg_loss, self.global_step)
            self.writer.add_scalar('Train/SegmentationWeight', seg_weight, self.global_step)
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Optimization/LearningRate', current_lr, self.global_step)
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1e6
                self.writer.add_scalar('System/GPUMemoryMB', memory_mb, self.global_step)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save comprehensive checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'metrics_history': self.metrics_collector.history,
            'current_metrics': metrics,
            'best_metric_value': self.best_metric_value,
            'patience_counter': self.patience_counter,
            'current_phase': self.current_phase
        }
        
        # Save latest checkpoint
        checkpoint_dir = Path(self.config.checkpoints.save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        latest_path = checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        if epoch % self.config.checkpoints.save_every_n_epochs == 0:
            epoch_path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved (epoch {epoch}): {self.config.checkpoints.monitor}={metrics.get(self.config.checkpoints.monitor, 0.0):.4f}")
        
        # Clean up old checkpoints (keep only top K)
        self._cleanup_old_checkpoints(checkpoint_dir)
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        """Clean up old epoch checkpoints, keeping only the most recent ones"""
        if self.config.checkpoints.keep_top_k <= 0:
            return
        
        # Find all epoch checkpoints
        epoch_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if len(epoch_checkpoints) > self.config.checkpoints.keep_top_k:
            # Sort by modification time (newest first)
            epoch_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old checkpoints
            for old_checkpoint in epoch_checkpoints[self.config.checkpoints.keep_top_k:]:
                old_checkpoint.unlink()
                self.logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and resume training"""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint['global_step']
        self.best_metric_value = checkpoint['best_metric_value']
        self.patience_counter = checkpoint['patience_counter']
        self.current_phase = checkpoint['current_phase']
        
        # Load history
        self.training_history = checkpoint['training_history']
        if 'metrics_history' in checkpoint:
            self.metrics_collector.history = checkpoint['metrics_history']
        
        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"Resumed from epoch {checkpoint['epoch']}, best metric: {self.best_metric_value:.4f}")
        
        return start_epoch
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop with Phase 4 infrastructure"""
        
        # Setup all training components
        if self.model is None:
            self.setup_model()
        if self.optimizer is None:
            self.setup_optimizer()
        if self.scheduler is None:
            self.setup_scheduler()
        if self.loss_fn is None:
            self.setup_loss_function()
        
        self.setup_mixed_precision()
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
            start_epoch = self.load_checkpoint(resume_from_checkpoint)
        
        # Log training start
        self.logger.info("=" * 80)
        self.logger.info("START PHASE 4: ADVANCED MULTI-TASK TRAINING STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment: {self.config.experiment_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.config.mixed_precision}")
        self.logger.info(f"Total epochs: {self.config.total_epochs}")
        self.logger.info(f"Effective batch size: {self.config.effective_batch_size}")
        self.logger.info(f"Progressive training phases:")
        self.logger.info(f"  Phase 1: Classification only ({self.config.phase1.epochs} epochs)")
        self.logger.info(f"  Phase 2: Segmentation warmup ({self.config.phase2.epochs} epochs)")
        self.logger.info(f"  Phase 3: Multi-task optimization ({self.config.phase3.epochs} epochs)")
        self.logger.info("=" * 80)
        
        training_start_time = time.time()
        
        # Training loop
        for epoch in range(start_epoch, self.config.total_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Get current phase
            phase_name, phase_info = self.get_current_phase_info(epoch)
            self.current_phase = phase_name
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = {}
            if epoch % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        monitor_value = val_metrics.get(self.config.checkpoints.monitor, 0.0)
                        self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update training history
            self._update_training_history(epoch, train_metrics, val_metrics, current_lr, epoch_time)
            
            # Check for best model
            is_best = False
            if val_metrics:
                monitor_value = val_metrics.get(self.config.checkpoints.monitor, -float('inf'))
                if monitor_value > self.best_metric_value:
                    self.best_metric_value = monitor_value
                    self.patience_counter = 0
                    is_best = True
                else:
                    self.patience_counter += 1
            
            # Save checkpoint
            if self.config.checkpoints.save_best or epoch % self.config.checkpoints.save_every_n_epochs == 0:
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(epoch, all_metrics, is_best)
            
            # Logging
            self._log_epoch_results(epoch, train_metrics, val_metrics, current_lr, epoch_time, phase_name)
            
            # TensorBoard logging
            if self.writer:
                self._log_epoch_to_tensorboard(epoch, train_metrics, val_metrics, current_lr)
            
            # W&B logging
            if self.wandb_run:
                self._log_epoch_to_wandb(epoch, train_metrics, val_metrics, current_lr)
            
            # Early stopping check
            if self._should_stop_early():
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs (patience: {self.config.early_stopping.patience})")
                break
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Training completed
        total_time = time.time() - training_start_time
        self.logger.info("=" * 80)
        self.logger.info("SUCCESS TRAINING COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Total training time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best validation score: {self.best_metric_value:.4f}")
        
        # Generate final report
        final_report = self.metrics_collector.generate_report()
        self.logger.info(f"\n{final_report}")
        
        # Save final plots
        if self.config.logging.save_plots:
            self._save_final_plots()
        
        # Close logging
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
        
        return {
            'training_history': self.training_history,
            'metrics_history': self.metrics_collector.history,
            'best_metric_value': self.best_metric_value,
            'total_epochs': epoch + 1,
            'total_time_hours': total_time / 3600
        }
    
    def _update_training_history(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float, epoch_time: float):
        """Update training history"""
        self.training_history['epoch'].append(epoch)
        self.training_history['phase'].append(train_metrics.get('phase', 'unknown'))
        self.training_history['train_loss'].append(train_metrics.get('loss', 0.0))
        self.training_history['val_loss'].append(val_metrics.get('loss', 0.0))
        self.training_history['train_cls_loss'].append(train_metrics.get('cls_loss', 0.0))
        self.training_history['val_cls_loss'].append(val_metrics.get('cls_loss', 0.0))
        self.training_history['train_seg_loss'].append(train_metrics.get('seg_loss', 0.0))
        self.training_history['val_seg_loss'].append(val_metrics.get('seg_loss', 0.0))
        self.training_history['segmentation_weight'].append(train_metrics.get('segmentation_weight', 0.0))
        self.training_history['learning_rate'].append(lr)
        self.training_history['epoch_time_minutes'].append(epoch_time / 60)
        
        # GPU memory
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1e6
            self.training_history['gpu_memory_mb'].append(memory_mb)
        else:
            self.training_history['gpu_memory_mb'].append(0.0)
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float, epoch_time: float, phase: str):
        """Log epoch results"""
        # Training metrics
        self.logger.info(
            f"Epoch {epoch+1:3d}/{self.config.total_epochs} [{phase}] - "
            f"Time: {epoch_time/60:.1f}min - "
            f"LR: {lr:.2e}"
        )
        self.logger.info(
            f"  Train - Loss: {train_metrics.get('loss', 0.0):.4f} | "
            f"Cls: {train_metrics.get('cls_loss', 0.0):.4f} | "
            f"Seg: {train_metrics.get('seg_loss', 0.0):.4f} | "
            f"Acc: {train_metrics.get('cls_accuracy', 0.0):.3f} | "
            f"Dice: {train_metrics.get('seg_dice', 0.0):.3f} | "
            f"SegW: {train_metrics.get('segmentation_weight', 0.0):.2f}"
        )
        
        # Validation metrics
        if val_metrics:
            self.logger.info(
                f"  Val   - Loss: {val_metrics.get('loss', 0.0):.4f} | "
                f"Cls: {val_metrics.get('cls_loss', 0.0):.4f} | "
                f"Seg: {val_metrics.get('seg_loss', 0.0):.4f} | "
                f"Acc: {val_metrics.get('cls_accuracy', 0.0):.3f} | "
                f"Dice: {val_metrics.get('seg_dice', 0.0):.3f} | "
                f"Combined: {val_metrics.get('combined_score', 0.0):.3f}"
            )
        
        # GPU memory
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1e6
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"  GPU Memory: {memory_mb:.0f}MB / {memory_gb:.1f}GB")
    
    def _log_epoch_to_tensorboard(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """Log epoch results to TensorBoard"""
        if self.writer:
            # Log training metrics
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            # Log validation metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Log learning rate
            self.writer.add_scalar('Optimization/LearningRate', lr, epoch)
            
            # Log GPU memory usage
            if 'gpu_memory_mb' in self.training_history and self.training_history['gpu_memory_mb']:
                self.writer.add_scalar('System/GPUMemoryMB', self.training_history['gpu_memory_mb'][-1], epoch)
    
    def _log_epoch_to_wandb(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """Log epoch metrics to W&B"""
        if not self.wandb_run:
            return
        
        log_data = {'epoch': epoch, 'learning_rate': lr}
        
        # Add training metrics
        for metric_name, value in train_metrics.items():
            if isinstance(value, (int, float)):
                log_data[f'train_{metric_name}'] = value
        
        # Add validation metrics
        for metric_name, value in val_metrics.items():
            if isinstance(value, (int, float)):
                log_data[f'val_{metric_name}'] = value
        
        self.wandb_run.log(log_data)
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early"""
        return self.patience_counter >= self.config.early_stopping.patience
    
    def _save_final_plots(self):
        """Save final training plots"""
        try:
            plots_dir = self.experiment_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Create comprehensive training plots
            self._plot_training_curves(plots_dir / "training_curves.png")
            self._plot_phase_progression(plots_dir / "phase_progression.png")
            self._plot_metrics_comparison(plots_dir / "metrics_comparison.png")
            
        except Exception as e:
            self.logger.warning(f"Failed to save plots: {e}")
    
    def _plot_training_curves(self, save_path: Path):
        """Plot comprehensive training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = self.training_history['epoch']
        
        # Loss curves
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Classification metrics
        if 'accuracy' in self.metrics_collector.history['classification']:
            train_acc = self.metrics_collector.history['classification']['accuracy']
            axes[0, 1].plot(epochs[:len(train_acc)], train_acc, 'b-', label='Accuracy', linewidth=2)
        
        if 'kappa' in self.metrics_collector.history['classification']:
            train_kappa = self.metrics_collector.history['classification']['kappa']
            axes[0, 1].plot(epochs[:len(train_kappa)], train_kappa, 'g-', label='Kappa', linewidth=2)
        
        axes[0, 1].set_title('Classification Metrics')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Segmentation metrics
        if 'dice' in self.metrics_collector.history['segmentation']:
            train_dice = self.metrics_collector.history['segmentation']['dice']
            axes[0, 2].plot(epochs[:len(train_dice)], train_dice, 'b-', label='Dice', linewidth=2)
        
        if 'iou' in self.metrics_collector.history['segmentation']:
            train_iou = self.metrics_collector.history['segmentation']['iou']
            axes[0, 2].plot(epochs[:len(train_iou)], train_iou, 'g-', label='IoU', linewidth=2)
        
        axes[0, 2].set_title('Segmentation Metrics')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.training_history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Segmentation weight progression
        axes[1, 1].plot(epochs, self.training_history['segmentation_weight'], 'm-', linewidth=2)
        axes[1, 1].set_title('Segmentation Weight Progression')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].grid(True, alpha=0.3)
        
        # GPU memory usage
        axes[1, 2].plot(epochs, self.training_history['gpu_memory_mb'], 'c-', linewidth=2)
        axes[1, 2].set_title('GPU Memory Usage')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Memory (MB)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_phase_progression(self, save_path: Path):
        """Plot training phase progression"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        epochs = self.training_history['epoch']
        phases = self.training_history['phase']
        
        # Create phase timeline
        phase_colors = {'phase1': 'red', 'phase2': 'orange', 'phase3': 'green'}
        
        for i, (epoch, phase) in enumerate(zip(epochs, phases)):
            color = phase_colors.get(phase, 'gray')
            ax.scatter(epoch, 1, c=color, s=100, alpha=0.7)
        
        # Add phase boundaries
        if self.config.phase1.epochs > 0:
            ax.axvline(x=self.config.phase1.epochs, color='red', linestyle='--', alpha=0.5, label='Phase 1 → 2')
        
        if self.config.phase1.epochs + self.config.phase2.epochs > 0:
            ax.axvline(x=self.config.phase1.epochs + self.config.phase2.epochs, 
                      color='orange', linestyle='--', alpha=0.5, label='Phase 2 → 3')
        
        ax.set_xlabel('Epoch')
        ax.set_title('Training Phase Progression')
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add phase descriptions
        ax.text(self.config.phase1.epochs / 2, 1.2, 'Phase 1:\nClassification Only', 
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        phase2_center = self.config.phase1.epochs + self.config.phase2.epochs / 2
        ax.text(phase2_center, 1.2, 'Phase 2:\nSegmentation Warmup', 
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
        
        phase3_center = self.config.phase1.epochs + self.config.phase2.epochs + self.config.phase3.epochs / 2
        ax.text(phase3_center, 1.2, 'Phase 3:\nMulti-task Optimization', 
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, save_path: Path):
        """Plot metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combined score evolution
        if 'combined_score' in self.metrics_collector.history['combined']:
            combined_scores = self.metrics_collector.history['combined']['combined_score']
            epochs = list(range(len(combined_scores)))
            
            axes[0, 0].plot(epochs, combined_scores, 'b-', linewidth=2)
            axes[0, 0].set_title('Combined Score Evolution')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Classification vs Segmentation performance
        if ('kappa' in self.metrics_collector.history['classification'] and 
            'dice' in self.metrics_collector.history['segmentation']):
            
            kappa_scores = self.metrics_collector.history['classification']['kappa']
            dice_scores = self.metrics_collector.history['segmentation']['dice']
            min_len = min(len(kappa_scores), len(dice_scores))
            
            epochs = list(range(min_len))
            axes[0, 1].plot(epochs, kappa_scores[:min_len], 'r-', label='Kappa (Classification)', linewidth=2)
            axes[0, 1].plot(epochs, dice_scores[:min_len], 'b-', label='Dice (Segmentation)', linewidth=2)
            axes[0, 1].set_title('Classification vs Segmentation')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Loss components
        epochs = self.training_history['epoch']
        axes[1, 0].plot(epochs, self.training_history['train_cls_loss'], 'r-', label='Classification', linewidth=2)
        axes[1, 0].plot(epochs, self.training_history['train_seg_loss'], 'b-', label='Segmentation', linewidth=2)
        axes[1, 0].set_title('Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training efficiency
        axes[1, 1].plot(epochs, self.training_history['epoch_time_minutes'], 'g-', linewidth=2)
        axes[1, 1].set_title('Training Efficiency')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time per Epoch (minutes)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = AdvancedTrainingConfig()
    
    # Create trainer
    trainer = Phase4Trainer(config)
    
    print("Phase 4 Advanced Trainer initialized!")
    print(f"Experiment: {config.experiment_name}")
    print(f"Total epochs: {config.total_epochs}")
    print(f"Progressive phases: {config.phase1.epochs} + {config.phase2.epochs} + {config.phase3.epochs}")
    print(f"Effective batch size: {config.effective_batch_size}")
