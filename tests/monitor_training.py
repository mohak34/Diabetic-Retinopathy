"""
Real-Time Training Monitor Dashboard
Monitor your diabetic retinopathy model training progress
"""

import os
import time
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import subprocess
import psutil


class TrainingMonitor:
    """Real-time training progress monitor"""
    
    def __init__(self, checkpoint_dir="checkpoints", log_dir="logs", results_dir="results"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        
        # Initialize plotting
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Diabetic Retinopathy Model Training Monitor', fontsize=16)
        
        # Training history
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rate': [],
            'segmentation_weight': []
        }
        
        # Setup plots
        self.setup_plots()
        
    def setup_plots(self):
        """Setup matplotlib plots"""
        # Loss plot
        self.loss_line_train, = self.axes[0, 0].plot([], [], 'b-', label='Train', linewidth=2)
        self.loss_line_val, = self.axes[0, 0].plot([], [], 'r-', label='Validation', linewidth=2)
        self.axes[0, 0].set_title('Training & Validation Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        self.acc_line_train, = self.axes[0, 1].plot([], [], 'b-', label='Train', linewidth=2)
        self.acc_line_val, = self.axes[0, 1].plot([], [], 'r-', label='Validation', linewidth=2)
        self.axes[0, 1].set_title('Classification Accuracy')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].set_ylim(0, 1)
        
        # Dice score plot
        self.dice_line_train, = self.axes[1, 0].plot([], [], 'b-', label='Train', linewidth=2)
        self.dice_line_val, = self.axes[1, 0].plot([], [], 'r-', label='Validation', linewidth=2)
        self.axes[1, 0].set_title('Segmentation Dice Score')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Dice Score')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].set_ylim(0, 1)
        
        # Learning rate and segmentation weight
        self.lr_line, = self.axes[1, 1].plot([], [], 'g-', label='Learning Rate', linewidth=2)
        self.axes[1, 1].set_title('Learning Rate & Segmentation Weight')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Learning Rate', color='g')
        self.axes[1, 1].tick_params(axis='y', labelcolor='g')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        # Secondary y-axis for segmentation weight
        self.ax2 = self.axes[1, 1].twinx()
        self.seg_weight_line, = self.ax2.plot([], [], 'm-', label='Seg Weight', linewidth=2)
        self.ax2.set_ylabel('Segmentation Weight', color='m')
        self.ax2.tick_params(axis='y', labelcolor='m')
        self.ax2.set_ylim(0, 1)
        
        plt.tight_layout()
    
    def load_latest_checkpoint(self):
        """Load training history from latest checkpoint"""
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'training_history' in checkpoint:
                    history = checkpoint['training_history']
                    
                    # Update our history
                    epochs = list(range(len(history.get('train_loss', []))))
                    self.history['epochs'] = epochs
                    self.history['train_loss'] = history.get('train_loss', [])
                    self.history['val_loss'] = history.get('val_loss', [])
                    self.history['train_acc'] = history.get('train_cls_acc', [])
                    self.history['val_acc'] = history.get('val_cls_acc', [])
                    self.history['train_dice'] = history.get('train_dice', [])
                    self.history['val_dice'] = history.get('val_dice', [])
                    self.history['learning_rate'] = history.get('learning_rate', [])
                    
                    # Calculate segmentation weight progression
                    self.history['segmentation_weight'] = self.calculate_seg_weights(epochs)
                    
                    return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        
        return False
    
    def calculate_seg_weights(self, epochs):
        """Calculate segmentation weight for each epoch"""
        weights = []
        for epoch in epochs:
            if epoch < 5:  # classification_only_epochs
                weight = 0.0
            elif epoch < 15:  # classification_only + warmup
                progress = (epoch - 5) / 10
                weight = progress * 0.8
            else:
                weight = 0.8
            weights.append(weight)
        return weights
    
    def update_plots(self):
        """Update all plots with latest data"""
        if not self.history['epochs']:
            return
        
        epochs = self.history['epochs']
        
        # Update loss plot
        if self.history['train_loss']:
            self.loss_line_train.set_data(epochs, self.history['train_loss'])
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()
        
        if self.history['val_loss']:
            val_epochs = epochs[:len(self.history['val_loss'])]
            self.loss_line_val.set_data(val_epochs, self.history['val_loss'])
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()
        
        # Update accuracy plot
        if self.history['train_acc']:
            self.acc_line_train.set_data(epochs, self.history['train_acc'])
        
        if self.history['val_acc']:
            val_epochs = epochs[:len(self.history['val_acc'])]
            self.acc_line_val.set_data(val_epochs, self.history['val_acc'])
        
        # Update dice plot
        if self.history['train_dice']:
            self.dice_line_train.set_data(epochs, self.history['train_dice'])
        
        if self.history['val_dice']:
            val_epochs = epochs[:len(self.history['val_dice'])]
            self.dice_line_val.set_data(val_epochs, self.history['val_dice'])
        
        # Update learning rate and segmentation weight
        if self.history['learning_rate']:
            self.lr_line.set_data(epochs, self.history['learning_rate'])
            self.axes[1, 1].relim()
            self.axes[1, 1].autoscale_view()
        
        if self.history['segmentation_weight']:
            self.seg_weight_line.set_data(epochs, self.history['segmentation_weight'])
    
    def get_system_info(self):
        """Get current system information"""
        info = {}
        
        # GPU info
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name()
            info['gpu_memory_used'] = torch.cuda.memory_allocated() / 1e9
            info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 'N/A'
        else:
            info['gpu_name'] = 'No GPU available'
            info['gpu_memory_used'] = 0
            info['gpu_memory_total'] = 0
            info['gpu_utilization'] = 'N/A'
        
        # CPU and RAM
        info['cpu_percent'] = psutil.cpu_percent()
        info['ram_percent'] = psutil.virtual_memory().percent
        info['ram_available'] = psutil.virtual_memory().available / 1e9
        
        return info
    
    def print_status(self):
        """Print current training status"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("DIABETIC RETINOPATHY MODEL - DIABETIC RETINOPATHY MODEL - TRAINING MONITOR")
        print("=" * 80)
        print(f"⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System info
        info = self.get_system_info()
        print("SYSTEM STATUS: SYSTEM STATUS:")
        print(f"   GPU: {info['gpu_name']}")
        print(f"   GPU Memory: {info['gpu_memory_used']:.1f}GB / {info['gpu_memory_total']:.1f}GB")
        print(f"   GPU Utilization: {info['gpu_utilization']}")
        print(f"   CPU Usage: {info['cpu_percent']:.1f}%")
        print(f"   RAM Usage: {info['ram_percent']:.1f}% ({info['ram_available']:.1f}GB available)")
        print()
        
        # Training status
        if self.history['epochs']:
            current_epoch = len(self.history['epochs'])
            print("TRAINING STATUS: TRAINING STATUS:")
            print(f"   Current Epoch: {current_epoch}")
            
            if self.history['train_loss']:
                print(f"   Latest Train Loss: {self.history['train_loss'][-1]:.4f}")
            if self.history['val_loss']:
                print(f"   Latest Val Loss: {self.history['val_loss'][-1]:.4f}")
            if self.history['train_acc']:
                print(f"   Latest Train Accuracy: {self.history['train_acc'][-1]:.3f}")
            if self.history['val_acc']:
                print(f"   Latest Val Accuracy: {self.history['val_acc'][-1]:.3f}")
            if self.history['train_dice']:
                print(f"   Latest Train Dice: {self.history['train_dice'][-1]:.3f}")
            if self.history['val_dice']:
                print(f"   Latest Val Dice: {self.history['val_dice'][-1]:.3f}")
            if self.history['learning_rate']:
                print(f"   Current Learning Rate: {self.history['learning_rate'][-1]:.2e}")
            if self.history['segmentation_weight']:
                print(f"   Segmentation Weight: {self.history['segmentation_weight'][-1]:.2f}")
        else:
            print("TRAINING STATUS: TRAINING STATUS: Waiting for training to start...")
        
        print()
        print("PROGRESSIVE TRAINING PHASES: PROGRESSIVE TRAINING PHASES:")
        current_epoch = len(self.history['epochs'])
        if current_epoch < 5:
            print("   Phase 1: Classification Only COMPLETE (Current)")
        elif current_epoch < 15:
            print("   Phase 1: Classification Only COMPLETE")
            print("   Phase 2: Segmentation Warmup CURRENT (Current)")
        else:
            print("   Phase 1: Classification Only COMPLETE")
            print("   Phase 2: Segmentation Warmup COMPLETE")
            print("   Phase 3: Multi-Task Training CURRENT (Current)")
        
        print()
        print(" CONTROLS:")
        print("   Ctrl+C: Stop monitoring")
        print("   Close plot window: Exit")
        print("=" * 80)
    
    def animate(self, frame):
        """Animation function for matplotlib"""
        self.load_latest_checkpoint()
        self.update_plots()
        self.print_status()
        return []
    
    def start_monitoring(self, update_interval=5):
        """Start real-time monitoring"""
        print("Starting training monitor...")
        print(f"Monitoring directories:")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {self.log_dir}")
        print(f"  Results: {self.results_dir}")
        print()
        
        # Animation
        ani = animation.FuncAnimation(
            self.fig, 
            self.animate, 
            interval=update_interval * 1000,
            blit=False,
            cache_frame_data=False
        )
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Monitor diabetic retinopathy model training')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', default='logs', help='Log directory')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--update-interval', type=int, default=5, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = TrainingMonitor(
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        results_dir=args.results_dir
    )
    
    try:
        monitor.start_monitoring(update_interval=args.update_interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")


if __name__ == "__main__":
    main()
