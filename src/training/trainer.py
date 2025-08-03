"""
Comprehensive Multi-Task Model Trainer with Real-Time Monitoring
Diabetic Retinopathy Classification + Lesion Segmentation
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torchmetrics
from torchmetrics import Accuracy, Dice, JaccardIndex
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.multi_task_model import MultiTaskRetinaModel, create_multi_task_model
from ..training.losses import MultiTaskLoss


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    # Model Config
    model_name: str = "efficientnet_v2_s"
    num_classes: int = 5
    pretrained: bool = True
    use_skip_connections: bool = True
    use_advanced_decoder: bool = False
    
    # Training Config
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    
    # Progressive Training
    classification_only_epochs: int = 10  # Train classification first
    segmentation_warmup_epochs: int = 20  # Gradually introduce segmentation
    
    # Loss Weights
    classification_weight: float = 1.0
    segmentation_weight_start: float = 0.0
    segmentation_weight_end: float = 0.8
    focal_gamma: float = 2.0
    dice_smooth: float = 1e-5
    
    # Optimization
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Mixed Precision
    use_mixed_precision: bool = True
    
    # Validation
    val_every_n_epochs: int = 1
    save_best_model: bool = True
    patience: int = 15  # Early stopping patience
    
    # Logging
    log_every_n_steps: int = 10
    save_tensorboard: bool = True
    save_plots: bool = True
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"


class MultiTaskTrainer:
    """
    Comprehensive trainer for multi-task diabetic retinopathy model
    with real-time monitoring and progressive training strategy
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.logger = logger or self._setup_logger()
        
        # Setup directories
        self._setup_directories()
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_cls_acc': [], 'val_cls_acc': [],
            'train_dice': [], 'val_dice': [],
            'learning_rate': []
        }
        
        # Metrics
        self.metrics = self._setup_metrics()
        
        # TensorBoard
        if config.save_tensorboard:
            self.writer = SummaryWriter(log_dir=self.config.log_dir)
        else:
            self.writer = None
            
        self.logger.info(f"Trainer initialized on device: {self.device}")
        self.logger.info(f"Training config: {asdict(config)}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('MultiTaskTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.config.checkpoint_dir, self.config.log_dir, self.config.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _setup_metrics(self) -> Dict[str, torchmetrics.Metric]:
        """Setup torchmetrics for evaluation"""
        metrics = {
            'cls_accuracy': Accuracy(task='multiclass', num_classes=self.config.num_classes).to(self.device),
            'seg_dice': Dice(num_classes=1, average='macro').to(self.device),
            'seg_iou': JaccardIndex(task='binary', num_classes=1).to(self.device),
        }
        return metrics
    
    def setup_model(self) -> MultiTaskRetinaModel:
        """Initialize and setup the multi-task model"""
        self.logger.info("Setting up multi-task model...")
        
        self.model = create_multi_task_model(
            backbone_name=self.config.model_name,
            num_classes_cls=self.config.num_classes,
            pretrained=self.config.pretrained,
            use_skip_connections=self.config.use_skip_connections,
            use_advanced_decoder=self.config.use_advanced_decoder
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {self.config.model_name}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Memory estimation
        if self.device.type == 'cuda':
            memory_mb = self.model.estimate_memory_usage(self.config.batch_size)
            self.logger.info(f"Estimated memory usage: {memory_mb:.1f} MB")
        
        return self.model
    
    def setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        if self.config.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        self.logger.info(f"Optimizer: {self.config.optimizer}")
        return self.optimizer
    
    def setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        if self.config.scheduler == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Scheduler: {self.config.scheduler}")
        return self.scheduler
    
    def setup_loss_function(self) -> MultiTaskLoss:
        """Setup multi-task loss function"""
        self.loss_fn = MultiTaskLoss(
            classification_weight=self.config.classification_weight,
            segmentation_weight=self.config.segmentation_weight_start,
            focal_gamma=self.config.focal_gamma,
            dice_smooth=self.config.dice_smooth
        )
        
        self.logger.info("Multi-task loss function configured")
        return self.loss_fn
    
    def setup_mixed_precision(self):
        """Setup mixed precision training"""
        if self.config.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
    
    def get_current_segmentation_weight(self, epoch: int) -> float:
        """Calculate current segmentation weight for progressive training"""
        if epoch < self.config.classification_only_epochs:
            return 0.0
        elif epoch < self.config.classification_only_epochs + self.config.segmentation_warmup_epochs:
            # Linear ramp-up
            progress = (epoch - self.config.classification_only_epochs) / self.config.segmentation_warmup_epochs
            return self.config.segmentation_weight_start + progress * (
                self.config.segmentation_weight_end - self.config.segmentation_weight_start
            )
        else:
            return self.config.segmentation_weight_end
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Update segmentation weight for progressive training
        seg_weight = self.get_current_segmentation_weight(epoch)
        self.loss_fn.segmentation_weight = seg_weight
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        total_loss = 0.0
        cls_loss_total = 0.0
        seg_loss_total = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, (images, cls_labels, seg_masks) in enumerate(progress_bar):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            seg_masks = seg_masks.to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    cls_logits, seg_preds = self.model(images)
                    loss, cls_loss, seg_loss = self.loss_fn(
                        cls_logits, seg_preds, cls_labels, seg_masks
                    )
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                cls_logits, seg_preds = self.model(images)
                loss, cls_loss, seg_loss = self.loss_fn(
                    cls_logits, seg_preds, cls_labels, seg_masks
                )
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            with torch.no_grad():
                self.metrics['cls_accuracy'](cls_logits, cls_labels)
                if seg_weight > 0:
                    seg_preds_binary = (torch.sigmoid(seg_preds) > 0.5).float()
                    self.metrics['seg_dice'](seg_preds_binary, seg_masks)
                    self.metrics['seg_iou'](seg_preds_binary, seg_masks)
            
            # Accumulate losses
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            cls_loss_total += cls_loss.item()
            seg_loss_total += seg_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'ClsLoss': f'{cls_loss.item():.4f}',
                'SegLoss': f'{seg_loss.item():.4f}',
                'SegWeight': f'{seg_weight:.2f}'
            })
            
            # Log to TensorBoard
            if self.writer and (batch_idx + 1) % self.config.log_every_n_steps == 0:
                step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/SegmentationWeight', seg_weight, step)
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'cls_loss': cls_loss_total / num_batches,
            'seg_loss': seg_loss_total / num_batches,
            'cls_accuracy': self.metrics['cls_accuracy'].compute().item(),
            'seg_dice': self.metrics['seg_dice'].compute().item() if seg_weight > 0 else 0.0,
            'seg_iou': self.metrics['seg_iou'].compute().item() if seg_weight > 0 else 0.0,
            'seg_weight': seg_weight
        }
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        total_loss = 0.0
        cls_loss_total = 0.0
        seg_loss_total = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            
            for images, cls_labels, seg_masks in progress_bar:
                images = images.to(self.device)
                cls_labels = cls_labels.to(self.device)
                seg_masks = seg_masks.to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        cls_logits, seg_preds = self.model(images)
                        loss, cls_loss, seg_loss = self.loss_fn(
                            cls_logits, seg_preds, cls_labels, seg_masks
                        )
                else:
                    cls_logits, seg_preds = self.model(images)
                    loss, cls_loss, seg_loss = self.loss_fn(
                        cls_logits, seg_preds, cls_labels, seg_masks
                    )
                
                # Update metrics
                self.metrics['cls_accuracy'](cls_logits, cls_labels)
                if self.loss_fn.segmentation_weight > 0:
                    seg_preds_binary = (torch.sigmoid(seg_preds) > 0.5).float()
                    self.metrics['seg_dice'](seg_preds_binary, seg_masks)
                    self.metrics['seg_iou'](seg_preds_binary, seg_masks)
                
                # Accumulate losses
                total_loss += loss.item()
                cls_loss_total += cls_loss.item()
                seg_loss_total += seg_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'ClsAcc': f'{self.metrics["cls_accuracy"].compute().item():.3f}',
                    'Dice': f'{self.metrics["seg_dice"].compute().item():.3f}' if self.loss_fn.segmentation_weight > 0 else 'N/A'
                })
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'cls_loss': cls_loss_total / num_batches,
            'seg_loss': seg_loss_total / num_batches,
            'cls_accuracy': self.metrics['cls_accuracy'].compute().item(),
            'seg_dice': self.metrics['seg_dice'].compute().item() if self.loss_fn.segmentation_weight > 0 else 0.0,
            'seg_iou': self.metrics['seg_iou'].compute().item() if self.loss_fn.segmentation_weight > 0 else 0.0,
        }
        
        return epoch_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'config': asdict(self.config),
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with score: {self.best_val_score:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_score = checkpoint['best_val_score']
        self.training_history = checkpoint['training_history']
        
        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return start_epoch
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress"""
        if not self.training_history['train_loss']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Classification accuracy
        axes[0, 1].plot(self.training_history['train_cls_acc'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(self.training_history['val_cls_acc'], label='Val Accuracy', color='red')
        axes[0, 1].set_title('Classification Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Segmentation Dice
        axes[1, 0].plot(self.training_history['train_dice'], label='Train Dice', color='blue')
        axes[1, 0].plot(self.training_history['val_dice'], label='Val Dice', color='red')
        axes[1, 0].set_title('Segmentation Dice Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.training_history['learning_rate'], color='green')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Main training loop"""
        # Setup training components
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
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint)
        
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.config.use_mixed_precision}")
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            if epoch % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate_epoch(val_loader, epoch)
            else:
                val_metrics = {}
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        # Use combined score for plateau scheduler
                        combined_score = val_metrics['cls_accuracy'] + val_metrics['seg_dice']
                        self.scheduler.step(combined_score)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_cls_acc'].append(train_metrics['cls_accuracy'])
            self.training_history['train_dice'].append(train_metrics['seg_dice'])
            self.training_history['learning_rate'].append(current_lr)
            
            if val_metrics:
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['val_cls_acc'].append(val_metrics['cls_accuracy'])
                self.training_history['val_dice'].append(val_metrics['seg_dice'])
            
            # Calculate combined validation score
            if val_metrics:
                val_score = val_metrics['cls_accuracy'] + val_metrics['seg_dice']
                is_best = val_score > self.best_val_score
                if is_best:
                    self.best_val_score = val_score
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            else:
                is_best = False
            
            # Save checkpoint
            if self.config.save_best_model:
                self.save_checkpoint(epoch, is_best)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Time: {epoch_time:.2f}s - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Train Acc: {train_metrics['cls_accuracy']:.3f} - "
                f"Train Dice: {train_metrics['seg_dice']:.3f} - "
                f"SegWeight: {train_metrics['seg_weight']:.2f} - "
                f"LR: {current_lr:.2e}"
            )
            
            if val_metrics:
                self.logger.info(
                    f"Val Loss: {val_metrics['loss']:.4f} - "
                    f"Val Acc: {val_metrics['cls_accuracy']:.3f} - "
                    f"Val Dice: {val_metrics['seg_dice']:.3f}"
                )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/ClassificationAccuracy', train_metrics['cls_accuracy'], epoch)
                self.writer.add_scalar('Train/SegmentationDice', train_metrics['seg_dice'], epoch)
                self.writer.add_scalar('Train/SegmentationWeight', train_metrics['seg_weight'], epoch)
                self.writer.add_scalar('Optimization/LearningRate', current_lr, epoch)
                
                if val_metrics:
                    self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('Val/ClassificationAccuracy', val_metrics['cls_accuracy'], epoch)
                    self.writer.add_scalar('Val/SegmentationDice', val_metrics['seg_dice'], epoch)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Plot progress periodically
            if self.config.save_plots and (epoch + 1) % 10 == 0:
                plot_path = Path(self.config.results_dir) / f'training_progress_epoch_{epoch+1}.png'
                self.plot_training_progress(str(plot_path))
        
        # Training completed
        total_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Best validation score: {self.best_val_score:.4f}")
        
        # Final plot
        if self.config.save_plots:
            final_plot_path = Path(self.config.results_dir) / 'final_training_progress.png'
            self.plot_training_progress(str(final_plot_path))
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return self.training_history
