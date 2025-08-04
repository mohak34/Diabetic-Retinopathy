"""
Phase 4: Robust Training Infrastructure
GPU-optimized trainer with mixed precision and progressive learning.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from .phase4_config import Phase4Config
from .phase4_losses import RobustMultiTaskLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility with robust metric tracking"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'max',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        self.is_better = self._get_is_better_func()
    
    def _get_is_better_func(self):
        if self.mode == 'max':
            return lambda current, best: current > best + self.min_delta
        else:
            return lambda current, best: current < best - self.min_delta
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best model weights")
        
        return self.early_stop


class MetricsTracker:
    """Simple metrics tracking utility"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics_dict: Dict[str, float], batch_size: int = 1):
        for name, value in metrics_dict.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += float(value) * batch_size
            self.counts[name] += batch_size
    
    def compute(self) -> Dict[str, float]:
        return {
            name: self.metrics[name] / max(self.counts[name], 1)
            for name in self.metrics
        }


class RobustPhase4Trainer:
    """
    Robust Phase 4 Trainer with GPU optimization and progressive training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Phase4Config,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup loss function
        self.loss_fn = RobustMultiTaskLoss(
            classification_weight=config.loss.classification_weight,
            segmentation_weight=config.loss.segmentation_weight,
            use_focal_cls=config.loss.use_focal_cls,
            use_kappa_cls=config.loss.use_kappa_cls,
            focal_gamma=config.loss.focal_gamma,
            kappa_weight=config.loss.kappa_weight
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.hardware.mixed_precision else None
        
        # Setup early stopping
        self.early_stopping = None
        if config.early_stopping.enabled:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping.patience,
                min_delta=config.early_stopping.min_delta,
                mode=config.early_stopping.mode
            )
        
        # Setup logging
        self.writer = None
        if config.logging.use_tensorboard:
            log_dir = Path(config.output_dir) / config.experiment_name / config.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('-inf') if config.checkpoint.mode == 'max' else float('inf')
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        logger.info(f"RobustPhase4Trainer initialized on {self.device}")
        logger.info(f"Mixed precision: {config.hardware.mixed_precision}")
        logger.info(f"Gradient accumulation steps: {config.hardware.gradient_accumulation_steps}")
    
    def _create_optimizer(self):
        """Create optimizer based on configuration"""
        if self.config.optimizer.name.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps
            )
        elif self.config.optimizer.name.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer.name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on configuration"""
        if self.config.scheduler.name.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler.max_epochs,
                eta_min=self.config.scheduler.min_lr
            )
        elif self.config.scheduler.name.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self.config.scheduler.name.lower() == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.config.scheduler.patience,
                factor=0.1,
                min_lr=self.config.scheduler.min_lr
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with progressive learning strategy"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start_time = time.time()
        
        for batch_idx, (images, cls_labels, seg_masks) in enumerate(train_loader):
            # Move data to device
            images = images.to(self.device, non_blocking=True)
            cls_labels = cls_labels.to(self.device, non_blocking=True)
            seg_masks = seg_masks.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    cls_logits = outputs['classification']
                    seg_logits = outputs['segmentation']
                    
                    # Compute loss with current epoch for progressive training
                    loss_dict = self.loss_fn(
                        cls_logits, seg_logits, cls_labels, seg_masks, epoch=self.epoch
                    )
                    loss = loss_dict['total']
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.hardware.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.hardware.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                # Without mixed precision
                outputs = self.model(images)
                cls_logits = outputs['classification']
                seg_logits = outputs['segmentation']
                
                # Compute loss
                loss_dict = self.loss_fn(
                    cls_logits, seg_logits, cls_labels, seg_masks, epoch=self.epoch
                )
                loss = loss_dict['total']
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.hardware.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.hardware.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            # Update metrics
            batch_metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
            self.train_metrics.update(batch_metrics, images.size(0))
            
            # Log training progress
            if batch_idx % self.config.logging.log_every == 0:
                logger.info(
                    f"Epoch {self.epoch+1}/{self.config.progressive.total_epochs} "
                    f"[{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
                
                # TensorBoard logging
                if self.writer is not None:
                    for name, value in batch_metrics.items():
                        self.writer.add_scalar(f'Train/{name}', value, self.global_step)
                    
                    self.writer.add_scalar('Train/LearningRate', 
                                         self.optimizer.param_groups[0]['lr'], self.global_step)
        
        # Epoch timing
        epoch_time = time.time() - epoch_start_time
        
        # Compute average metrics
        avg_metrics = self.train_metrics.compute()
        avg_metrics['epoch_time'] = epoch_time
        
        return avg_metrics
    
    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        for images, cls_labels, seg_masks in val_loader:
            # Move data to device
            images = images.to(self.device, non_blocking=True)
            cls_labels = cls_labels.to(self.device, non_blocking=True)
            seg_masks = seg_masks.to(self.device, non_blocking=True)
            
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    cls_logits = outputs['classification']
                    seg_logits = outputs['segmentation']
                    
                    loss_dict = self.loss_fn(
                        cls_logits, seg_logits, cls_labels, seg_masks, epoch=self.epoch
                    )
            else:
                outputs = self.model(images)
                cls_logits = outputs['classification']
                seg_logits = outputs['segmentation']
                
                loss_dict = self.loss_fn(
                    cls_logits, seg_logits, cls_labels, seg_masks, epoch=self.epoch
                )
            
            # Update metrics
            batch_metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
            self.val_metrics.update(batch_metrics, images.size(0))
        
        # Compute average metrics
        avg_metrics = self.val_metrics.compute()
        
        # TensorBoard logging
        if self.writer is not None:
            for name, value in avg_metrics.items():
                self.writer.add_scalar(f'Val/{name}', value, self.epoch)
        
        return avg_metrics
    
    def save_checkpoint(self, checkpoint_path: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = str(Path(checkpoint_path).parent / 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"✅ Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        logger.info(f"✅ Checkpoint loaded from {checkpoint_path}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str
    ) -> Dict[str, Any]:
        """Complete training loop with progressive learning"""
        logger.info("🚀 Starting Phase 4 Training")
        logger.info(f"Total epochs: {self.config.progressive.total_epochs}")
        logger.info(f"Progressive phases: {self.config.progressive.phase1_epochs} + "
                   f"{self.config.progressive.phase2_epochs} + "
                   f"{self.config.progressive.phase3_epochs}")
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.config.progressive.total_epochs):
            self.epoch = epoch
            
            # Determine training phase
            if epoch < self.config.progressive.phase1_epochs:
                phase = "Phase 1 (Classification Only)"
            elif epoch < self.config.progressive.phase1_epochs + self.config.progressive.phase2_epochs:
                phase = "Phase 2 (Progressive Multi-task)"
            else:
                phase = "Phase 3 (Full Multi-task)"
            
            logger.info(f"\n📍 Epoch {epoch+1}/{self.config.progressive.total_epochs} - {phase}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            training_history['train_losses'].append(train_metrics['total'])
            training_history['train_metrics'].append(train_metrics)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            training_history['val_losses'].append(val_metrics['total'])
            training_history['val_metrics'].append(val_metrics)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            training_history['learning_rates'].append(current_lr)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total'])
                else:
                    self.scheduler.step()
            
            # Combined score for checkpointing (simple average)
            combined_score = -(val_metrics['total'])  # Negative because we want to minimize loss
            
            # Check for best model
            is_best = False
            if self.config.checkpoint.mode == 'max':
                if combined_score > self.best_metric:
                    self.best_metric = combined_score
                    is_best = True
            else:
                if combined_score < self.best_metric:
                    self.best_metric = combined_score
                    is_best = True
            
            # Log epoch results
            logger.info(f"📊 Train Loss: {train_metrics['total']:.4f} | "
                       f"Val Loss: {val_metrics['total']:.4f} | "
                       f"LR: {current_lr:.2e} | "
                       f"Time: {train_metrics['epoch_time']:.1f}s")
            
            if is_best:
                logger.info("⭐ New best model!")
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint.save_every == 0 or is_best:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                self.save_checkpoint(checkpoint_path, is_best=is_best)
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(combined_score, self.model):
                    logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Training complete
        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time/3600:.2f} hours")
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return {
            'training_history': training_history,
            'best_metric': self.best_metric,
            'total_epochs': epoch + 1,
            'total_time': total_time
        }
