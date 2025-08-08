"""
Visualization utilities for diabetic retinopathy models.
Provides functions for visualizing predictions, training progress, and model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import torch
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def visualize_predictions(
    images: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    targets: Optional[Dict[str, torch.Tensor]] = None,
    save_path: Optional[str] = None,
    max_samples: int = 8,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Visualize model predictions alongside ground truth.
    
    Args:
        images: Input images tensor [B, C, H, W]
        predictions: Dictionary with prediction tensors
        targets: Dictionary with target tensors (optional)
        save_path: Path to save the visualization
        max_samples: Maximum number of samples to show
        figsize: Figure size
    """
    try:
        batch_size = min(images.size(0), max_samples)
        
        # Determine number of columns based on available data
        n_cols = 2  # image + prediction
        if targets is not None:
            n_cols += 1  # add ground truth
        if 'segmentation' in predictions:
            n_cols += 1  # add segmentation
        
        fig, axes = plt.subplots(batch_size, n_cols, figsize=figsize)
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            col_idx = 0
            
            # Original image
            img = images[i].cpu()
            if img.shape[0] == 3:  # RGB
                img = img.permute(1, 2, 0)
            img = torch.clamp(img, 0, 1)
            
            axes[i, col_idx].imshow(img)
            axes[i, col_idx].set_title(f'Input Image {i+1}')
            axes[i, col_idx].axis('off')
            col_idx += 1
            
            # Classification prediction
            if 'classification' in predictions:
                cls_pred = predictions['classification'][i].cpu()
                if cls_pred.dim() == 1:  # logits
                    cls_probs = torch.softmax(cls_pred, dim=0)
                    pred_class = cls_probs.argmax().item()
                    confidence = cls_probs.max().item()
                    
                    # Create bar plot
                    axes[i, col_idx].bar(range(len(cls_probs)), cls_probs.numpy())
                    axes[i, col_idx].set_title(f'Prediction: Class {pred_class} ({confidence:.2f})')
                    axes[i, col_idx].set_xlabel('Class')
                    axes[i, col_idx].set_ylabel('Probability')
                col_idx += 1
            
            # Ground truth
            if targets is not None and 'classification' in targets:
                target_class = targets['classification'][i].item()
                axes[i, col_idx].text(0.5, 0.5, f'Ground Truth:\nClass {target_class}', 
                                     ha='center', va='center', fontsize=12,
                                     transform=axes[i, col_idx].transAxes)
                axes[i, col_idx].set_title('Ground Truth')
                axes[i, col_idx].axis('off')
                col_idx += 1
            
            # Segmentation prediction
            if 'segmentation' in predictions:
                seg_pred = predictions['segmentation'][i].cpu()
                if seg_pred.dim() == 3:  # Remove channel dim if present
                    seg_pred = seg_pred.squeeze(0)
                
                seg_pred = torch.sigmoid(seg_pred)
                axes[i, col_idx].imshow(seg_pred, cmap='hot', alpha=0.7)
                axes[i, col_idx].set_title('Segmentation Prediction')
                axes[i, col_idx].axis('off')
                col_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
        figsize: Figure size
    """
    try:
        n_plots = 0
        plots_info = []
        
        # Determine what to plot
        if 'train_losses' in history and 'val_losses' in history:
            plots_info.append(('Loss', 'train_losses', 'val_losses'))
            n_plots += 1
        
        if 'train_accuracies' in history and 'val_accuracies' in history:
            plots_info.append(('Accuracy', 'train_accuracies', 'val_accuracies'))
            n_plots += 1
        
        if 'learning_rates' in history:
            plots_info.append(('Learning Rate', 'learning_rates', None))
            n_plots += 1
        
        if n_plots == 0:
            logger.warning("No recognized metrics found in history")
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        for i, (title, train_key, val_key) in enumerate(plots_info):
            epochs = range(1, len(history[train_key]) + 1)
            
            axes[i].plot(epochs, history[train_key], label=f'Train {title}', marker='o')
            
            if val_key and val_key in history:
                axes[i].plot(epochs, history[val_key], label=f'Val {title}', marker='s')
            
            axes[i].set_title(title)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            if title == 'Learning Rate':
                axes[i].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to plot training history: {e}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the plot
        figsize: Figure size
    """
    try:
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_names: Names of classes
        save_path: Path to save the plot
        figsize: Figure size
    """
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        n_classes = y_proba.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=figsize)
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to plot ROC curves: {e}")


def visualize_segmentation_overlay(
    image: torch.Tensor,
    prediction: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    alpha: float = 0.5
) -> None:
    """
    Visualize segmentation as overlay on original image.
    
    Args:
        image: Original image tensor [C, H, W]
        prediction: Predicted segmentation mask [H, W] or [1, H, W]
        target: Ground truth mask (optional)
        save_path: Path to save the visualization
        alpha: Transparency of overlay
    """
    try:
        # Convert tensors to numpy
        img = image.cpu()
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        img = torch.clamp(img, 0, 1).numpy()
        
        pred = prediction.cpu()
        if pred.dim() == 3:
            pred = pred.squeeze(0)
        pred = torch.sigmoid(pred).numpy()
        
        n_cols = 2 if target is None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction overlay
        axes[1].imshow(img)
        axes[1].imshow(pred, cmap='hot', alpha=alpha)
        axes[1].set_title('Prediction Overlay')
        axes[1].axis('off')
        
        # Ground truth overlay (if available)
        if target is not None:
            target_np = target.cpu()
            if target_np.dim() == 3:
                target_np = target_np.squeeze(0)
            target_np = target_np.numpy()
            
            axes[2].imshow(img)
            axes[2].imshow(target_np, cmap='hot', alpha=alpha)
            axes[2].set_title('Ground Truth Overlay')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Segmentation overlay saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create segmentation overlay: {e}")


def create_prediction_grid(
    images: List[torch.Tensor],
    predictions: List[Dict[str, torch.Tensor]],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    grid_size: Optional[Tuple[int, int]] = None
) -> None:
    """
    Create a grid of prediction visualizations.
    
    Args:
        images: List of image tensors
        predictions: List of prediction dictionaries
        titles: Optional titles for each image
        save_path: Path to save the grid
        grid_size: (rows, cols) for the grid layout
    """
    try:
        n_samples = len(images)
        
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n_samples)))
            rows = int(np.ceil(n_samples / cols))
        else:
            rows, cols = grid_size
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for i in range(min(n_samples, len(axes))):
            img = images[i].cpu()
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = torch.clamp(img, 0, 1)
            
            axes[i].imshow(img)
            
            # Add prediction info as title
            title = titles[i] if titles else f'Sample {i+1}'
            if 'classification' in predictions[i]:
                cls_pred = predictions[i]['classification']
                if cls_pred.dim() == 1:
                    pred_class = torch.softmax(cls_pred, dim=0).argmax().item()
                    confidence = torch.softmax(cls_pred, dim=0).max().item()
                    title += f'\nPred: Class {pred_class} ({confidence:.2f})'
            
            axes[i].set_title(title)
            axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction grid saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create prediction grid: {e}")


def setup_visualization_style():
    """Setup matplotlib style for consistent visualizations."""
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


# Set up visualization style on import
setup_visualization_style()


if __name__ == "__main__":
    print("Visualization utilities for diabetic retinopathy models")
    
    # Example usage
    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_predictions = {
        'classification': torch.randn(4, 5),
        'segmentation': torch.randn(4, 1, 224, 224)
    }
    dummy_targets = {
        'classification': torch.randint(0, 5, (4,))
    }
    
    print("Creating example visualization...")
    visualize_predictions(dummy_images, dummy_predictions, dummy_targets)
