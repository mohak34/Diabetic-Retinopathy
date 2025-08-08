
"""
Basic monitoring system for training
"""
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class BasicTrainingMonitor:
    """Basic training monitor"""
    
    def __init__(self, experiment_dir="experiments/default", log_interval=10):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_dice': [],
            'learning_rates': [],
            'epochs': [],
            'timestamps': []
        }
    
    def log_metrics(self, epoch, metrics):
        """Log training metrics"""
        timestamp = datetime.now().isoformat()
        
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['timestamps'].append(timestamp)
        
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Save metrics to file
        metrics_file = self.experiment_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def plot_training_curves(self):
        """Generate training curve plots"""
        if not self.metrics_history['epochs']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = self.metrics_history['epochs']
        
        # Loss curves
        if self.metrics_history['train_loss']:
            axes[0, 0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss')
        if self.metrics_history['val_loss']:
            axes[0, 0].plot(epochs, self.metrics_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curve
        if self.metrics_history['val_accuracy']:
            axes[0, 1].plot(epochs, self.metrics_history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Dice score curve
        if self.metrics_history['val_dice']:
            axes[1, 0].plot(epochs, self.metrics_history['val_dice'], label='Validation Dice')
        axes[1, 0].set_title('Validation Dice Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curve
        if self.metrics_history['learning_rates']:
            axes[1, 1].plot(epochs, self.metrics_history['learning_rates'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.experiment_dir / "training_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def get_summary_stats(self):
        """Get training summary statistics"""
        if not self.metrics_history['epochs']:
            return {}
        
        summary = {
            'total_epochs': len(self.metrics_history['epochs']),
            'final_train_loss': self.metrics_history['train_loss'][-1] if self.metrics_history['train_loss'] else None,
            'final_val_loss': self.metrics_history['val_loss'][-1] if self.metrics_history['val_loss'] else None,
            'best_val_accuracy': max(self.metrics_history['val_accuracy']) if self.metrics_history['val_accuracy'] else None,
            'best_val_dice': max(self.metrics_history['val_dice']) if self.metrics_history['val_dice'] else None,
            'experiment_dir': str(self.experiment_dir)
        }
        
        return summary

def create_monitor(experiment_dir="experiments/default"):
    """Create training monitor"""
    return BasicTrainingMonitor(experiment_dir=experiment_dir)
