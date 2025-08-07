"""
Phase 6: Explainability & Visualization Analyzer
Handles Step 6.3: Explainability & Visualization including t-SNE, mask overlays, and interpretability analysis.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2
from PIL import Image
import matplotlib.patches as patches

from .comprehensive_evaluator import ComprehensiveEvaluator

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ExplainabilityAnalyzer:
    """
    Explainability and visualization analyzer for diabetic retinopathy models.
    Handles Step 6.3: Explainability & Visualization
    """
    
    def __init__(self,
                 comprehensive_evaluator: ComprehensiveEvaluator,
                 output_dir: str = 'results/phase6_explainability'):
        """
        Initialize explainability analyzer
        
        Args:
            comprehensive_evaluator: Main evaluator instance
            output_dir: Directory to save explainability results
        """
        self.evaluator = comprehensive_evaluator
        self.device = comprehensive_evaluator.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'tsne_plots').mkdir(exist_ok=True)
        (self.output_dir / 'mask_overlays').mkdir(exist_ok=True)
        (self.output_dir / 'feature_maps').mkdir(exist_ok=True)
        (self.output_dir / 'cluster_analysis').mkdir(exist_ok=True)
        
        self.class_names = [
            'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'
        ]
        
        self.lesion_types = [
            'microaneurysms', 'hemorrhages', 'hard_exudates', 'soft_exudates'
        ]
        
        logger.info(f"ExplainabilityAnalyzer initialized")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def run_comprehensive_explainability_analysis(self,
                                                 checkpoint_path: str,
                                                 data_loaders: Dict[str, DataLoader],
                                                 model_config: Optional[Dict] = None,
                                                 n_samples: int = 1000) -> Dict[str, any]:
        """
        Run comprehensive explainability analysis
        
        Args:
            checkpoint_path: Path to model checkpoint
            data_loaders: Dictionary of dataset_name -> DataLoader
            model_config: Model configuration
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary containing all explainability results
        """
        logger.info("Starting comprehensive explainability analysis")
        
        # Load model
        model = self.evaluator._load_model_from_checkpoint(checkpoint_path, model_config)
        model.eval()
        
        # Extract features and predictions
        feature_data = self._extract_features_and_predictions(
            model, data_loaders, n_samples
        )
        
        # Perform t-SNE visualization
        tsne_results = self._perform_tsne_analysis(feature_data)
        
        # Create segmentation mask overlays
        overlay_results = self._create_segmentation_overlays(
            model, data_loaders, n_samples=50
        )
        
        # Perform cluster analysis
        cluster_results = self._perform_cluster_analysis(feature_data)
        
        # Generate interpretability insights
        insights = self._generate_interpretability_insights(
            feature_data, tsne_results, cluster_results
        )
        
        # Create comprehensive visualizations
        self._create_comprehensive_visualizations(
            feature_data, tsne_results, cluster_results
        )
        
        # Generate explainability report
        report_path = self._generate_explainability_report(
            tsne_results, overlay_results, cluster_results, insights
        )
        
        return {
            'feature_data': feature_data,
            'tsne_results': tsne_results,
            'overlay_results': overlay_results,
            'cluster_results': cluster_results,
            'insights': insights,
            'report_path': report_path
        }
    
    def _extract_features_and_predictions(self,
                                        model: nn.Module,
                                        data_loaders: Dict[str, DataLoader],
                                        n_samples: int) -> Dict[str, any]:
        """Extract intermediate features and predictions from the model"""
        
        logger.info("Extracting features and predictions")
        
        # Hook to capture intermediate features
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output.detach()
            return hook
        
        # Register hooks for different layers
        hooks = []
        
        # Hook the backbone (assume EfficientNet-like structure)
        if hasattr(model, 'backbone'):
            backbone = model.backbone
            if hasattr(backbone, 'features'):
                # EfficientNet features
                hooks.append(backbone.features[-1].register_forward_hook(hook_fn('backbone_features')))
        
        # Hook classification head
        if hasattr(model, 'classification_head'):
            hooks.append(model.classification_head.register_forward_hook(hook_fn('cls_features')))
        
        # Storage for extracted data
        all_features = []
        all_images = []
        all_cls_labels = []
        all_cls_preds = []
        all_cls_probs = []
        all_seg_masks = []
        all_seg_preds = []
        all_dataset_labels = []
        
        sample_count = 0
        
        model.eval()
        with torch.no_grad():
            for dataset_name, data_loader in data_loaders.items():
                logger.info(f"Processing {dataset_name} dataset")
                
                for batch_idx, batch in enumerate(data_loader):
                    if sample_count >= n_samples:
                        break
                    
                    # Extract batch data - handle both tuple and dict formats
                    if isinstance(batch, (list, tuple)):
                        # Handle tuple format: (images, labels) or (images, labels, masks)
                        images = batch[0].to(self.device)
                        cls_targets = batch[1] if len(batch) > 1 else None
                        seg_targets = batch[2] if len(batch) > 2 else None
                    elif isinstance(batch, dict):
                        # Handle dictionary format
                        images = batch['image'].to(self.device)
                        cls_targets = batch.get('grade', batch.get('label'))
                        seg_targets = batch.get('mask', batch.get('segmentation'))
                    else:
                        logger.error(f"Unknown batch format: {type(batch)}")
                        continue
                    
                    # Forward pass
                    cls_outputs, seg_outputs = model(images)
                    
                    # Get predictions
                    cls_probs = torch.softmax(cls_outputs, dim=1)
                    cls_preds = torch.argmax(cls_probs, dim=1)
                    seg_probs = torch.sigmoid(seg_outputs)
                    seg_preds = (seg_probs > 0.5).float()
                    
                    # Extract features from the last hooked layer
                    if 'backbone_features' in features:
                        batch_features = features['backbone_features']
                        # Global average pooling if needed
                        if batch_features.dim() > 2:
                            batch_features = F.adaptive_avg_pool2d(batch_features, 1)
                            batch_features = batch_features.view(batch_features.size(0), -1)
                    elif 'cls_features' in features:
                        batch_features = features['cls_features']
                    else:
                        # Fallback: use pre-classification layer
                        batch_features = cls_outputs
                    
                    # Store data
                    batch_size = images.size(0)
                    remaining_samples = min(batch_size, n_samples - sample_count)
                    
                    all_features.append(batch_features[:remaining_samples].cpu().numpy())
                    all_images.append(images[:remaining_samples].cpu().numpy())
                    all_cls_preds.append(cls_preds[:remaining_samples].cpu().numpy())
                    all_cls_probs.append(cls_probs[:remaining_samples].cpu().numpy())
                    all_seg_preds.append(seg_preds[:remaining_samples].cpu().numpy())
                    
                    if cls_targets is not None:
                        all_cls_labels.append(cls_targets[:remaining_samples].cpu().numpy())
                    else:
                        all_cls_labels.append(np.full(remaining_samples, -1))
                    
                    if seg_targets is not None:
                        all_seg_masks.append(seg_targets[:remaining_samples].cpu().numpy())
                    else:
                        all_seg_masks.append(np.zeros((remaining_samples, len(self.lesion_types), 
                                                     seg_preds.size(2), seg_preds.size(3))))
                    
                    all_dataset_labels.extend([dataset_name] * remaining_samples)
                    
                    sample_count += remaining_samples
                    
                    if sample_count >= n_samples:
                        break
                
                if sample_count >= n_samples:
                    break
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate all data
        feature_data = {
            'features': np.vstack(all_features),
            'images': np.vstack(all_images),
            'cls_labels': np.concatenate(all_cls_labels),
            'cls_preds': np.concatenate(all_cls_preds),
            'cls_probs': np.vstack(all_cls_probs),
            'seg_masks': np.vstack(all_seg_masks),
            'seg_preds': np.vstack(all_seg_preds),
            'dataset_labels': all_dataset_labels
        }
        
        logger.info(f"Extracted features from {sample_count} samples")
        return feature_data
    
    def _perform_tsne_analysis(self, feature_data: Dict[str, any]) -> Dict[str, any]:
        """Perform t-SNE analysis on extracted features"""
        
        logger.info("Performing t-SNE analysis")
        
        features = feature_data['features']
        cls_labels = feature_data['cls_labels']
        cls_preds = feature_data['cls_preds']
        dataset_labels = feature_data['dataset_labels']
        
        # Apply PCA first for dimensionality reduction (if features are high-dimensional)
        if features.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            features_pca = pca.fit_transform(features)
            logger.info(f"PCA reduced features from {features.shape[1]} to {features_pca.shape[1]} dimensions")
        else:
            features_pca = features
        
        # Perform t-SNE
        # Adjust perplexity based on sample size (must be less than n_samples)
        n_samples = features_pca.shape[0]
        perplexity = min(30, max(5, n_samples - 1))  # Ensure perplexity is valid
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        tsne_features = tsne.fit_transform(features_pca)
        
        # Create visualizations
        self._plot_tsne_by_grade(tsne_features, cls_labels, 'true_grades')
        self._plot_tsne_by_grade(tsne_features, cls_preds, 'predicted_grades')
        self._plot_tsne_by_dataset(tsne_features, dataset_labels)
        self._plot_tsne_comparison(tsne_features, cls_labels, cls_preds)
        
        # Analyze t-SNE results
        tsne_analysis = self._analyze_tsne_clusters(
            tsne_features, cls_labels, cls_preds, dataset_labels
        )
        
        return {
            'tsne_coordinates': tsne_features,
            'analysis': tsne_analysis,
            'pca_explained_variance': getattr(pca, 'explained_variance_ratio_', None) if features.shape[1] > 50 else None
        }
    
    def _plot_tsne_by_grade(self, tsne_features: np.ndarray, labels: np.ndarray, 
                           title_suffix: str) -> None:
        """Plot t-SNE colored by DR grade"""
        
        plt.figure(figsize=(12, 10))
        
        # Define colors for each grade
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        # Plot each grade
        for grade in range(5):
            mask = labels == grade
            if np.any(mask):
                plt.scatter(tsne_features[mask, 0], tsne_features[mask, 1],
                           c=colors[grade], label=self.class_names[grade],
                           alpha=0.6, s=20)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f't-SNE Visualization by DR Grade ({title_suffix})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / 'tsne_plots' / f'tsne_by_{title_suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tsne_by_dataset(self, tsne_features: np.ndarray, dataset_labels: List[str]) -> None:
        """Plot t-SNE colored by dataset"""
        
        plt.figure(figsize=(12, 10))
        
        unique_datasets = list(set(dataset_labels))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_datasets)))
        
        for i, dataset in enumerate(unique_datasets):
            mask = np.array(dataset_labels) == dataset
            plt.scatter(tsne_features[mask, 0], tsne_features[mask, 1],
                       c=[colors[i]], label=dataset, alpha=0.6, s=20)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization by Dataset')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / 'tsne_plots' / 'tsne_by_dataset.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tsne_comparison(self, tsne_features: np.ndarray, 
                             true_labels: np.ndarray, pred_labels: np.ndarray) -> None:
        """Plot t-SNE with correct/incorrect predictions highlighted"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left plot: Correct vs incorrect predictions
        correct_mask = true_labels == pred_labels
        incorrect_mask = ~correct_mask
        
        ax1.scatter(tsne_features[correct_mask, 0], tsne_features[correct_mask, 1],
                   c='green', label='Correct', alpha=0.6, s=20)
        ax1.scatter(tsne_features[incorrect_mask, 0], tsne_features[incorrect_mask, 1],
                   c='red', label='Incorrect', alpha=0.6, s=20)
        
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.set_title('t-SNE: Correct vs Incorrect Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Error patterns by grade
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for grade in range(5):
            true_grade_mask = true_labels == grade
            correct_grade_mask = true_grade_mask & correct_mask
            incorrect_grade_mask = true_grade_mask & incorrect_mask
            
            if np.any(correct_grade_mask):
                ax2.scatter(tsne_features[correct_grade_mask, 0], 
                           tsne_features[correct_grade_mask, 1],
                           c=colors[grade], alpha=0.8, s=20, marker='o',
                           label=f'{self.class_names[grade]} (Correct)')
            
            if np.any(incorrect_grade_mask):
                ax2.scatter(tsne_features[incorrect_grade_mask, 0], 
                           tsne_features[incorrect_grade_mask, 1],
                           c=colors[grade], alpha=0.8, s=20, marker='x',
                           label=f'{self.class_names[grade]} (Error)')
        
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.set_title('t-SNE: Error Patterns by Grade')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'tsne_plots' / 'tsne_prediction_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_tsne_clusters(self, tsne_features: np.ndarray, 
                              true_labels: np.ndarray, pred_labels: np.ndarray,
                              dataset_labels: List[str]) -> Dict[str, any]:
        """Analyze t-SNE clustering patterns"""
        
        # Calculate cluster purity for each grade
        grade_purities = {}
        for grade in range(5):
            grade_mask = true_labels == grade
            if np.any(grade_mask):
                grade_features = tsne_features[grade_mask]
                
                # Calculate intra-cluster distance
                center = np.mean(grade_features, axis=0)
                distances = np.linalg.norm(grade_features - center, axis=1)
                
                grade_purities[self.class_names[grade]] = {
                    'mean_distance_to_center': float(np.mean(distances)),
                    'std_distance_to_center': float(np.std(distances)),
                    'sample_count': int(np.sum(grade_mask))
                }
        
        # Calculate separation between grades
        grade_separations = {}
        for i in range(5):
            for j in range(i + 1, 5):
                mask_i = true_labels == i
                mask_j = true_labels == j
                
                if np.any(mask_i) and np.any(mask_j):
                    center_i = np.mean(tsne_features[mask_i], axis=0)
                    center_j = np.mean(tsne_features[mask_j], axis=0)
                    separation = np.linalg.norm(center_i - center_j)
                    
                    pair_name = f"{self.class_names[i]} vs {self.class_names[j]}"
                    grade_separations[pair_name] = float(separation)
        
        # Analyze dataset clustering
        dataset_analysis = {}
        unique_datasets = list(set(dataset_labels))
        
        for dataset in unique_datasets:
            dataset_mask = np.array(dataset_labels) == dataset
            if np.any(dataset_mask):
                dataset_features = tsne_features[dataset_mask]
                center = np.mean(dataset_features, axis=0)
                distances = np.linalg.norm(dataset_features - center, axis=1)
                
                dataset_analysis[dataset] = {
                    'mean_distance_to_center': float(np.mean(distances)),
                    'std_distance_to_center': float(np.std(distances)),
                    'sample_count': int(np.sum(dataset_mask))
                }
        
        return {
            'grade_purities': grade_purities,
            'grade_separations': grade_separations,
            'dataset_analysis': dataset_analysis
        }
    
    def _create_segmentation_overlays(self, 
                                    model: nn.Module,
                                    data_loaders: Dict[str, DataLoader],
                                    n_samples: int = 50) -> Dict[str, any]:
        """Create segmentation mask overlays for visualization"""
        
        logger.info("Creating segmentation mask overlays")
        
        overlay_results = {}
        sample_count = 0
        
        model.eval()
        with torch.no_grad():
            for dataset_name, data_loader in data_loaders.items():
                logger.info(f"Creating overlays for {dataset_name}")
                
                dataset_overlays = []
                
                for batch_idx, batch in enumerate(data_loader):
                    if sample_count >= n_samples:
                        break
                    
                    # Extract batch data - handle both tuple and dict formats
                    if isinstance(batch, (list, tuple)):
                        # Handle tuple format: (images, labels) or (images, labels, masks)
                        images = batch[0].to(self.device)
                        cls_targets = batch[1] if len(batch) > 1 else None
                        seg_targets = batch[2] if len(batch) > 2 else None
                    elif isinstance(batch, dict):
                        # Handle dictionary format
                        images = batch['image'].to(self.device)
                        seg_targets = batch.get('mask', batch.get('segmentation'))
                        cls_targets = batch.get('grade', batch.get('label'))
                    else:
                        logger.error(f"Unknown batch format: {type(batch)}")
                        continue
                    
                    # Forward pass
                    cls_outputs, seg_outputs = model(images)
                    
                    # Get predictions
                    cls_preds = torch.argmax(torch.softmax(cls_outputs, dim=1), dim=1)
                    seg_probs = torch.sigmoid(seg_outputs)
                    seg_preds = (seg_probs > 0.5).float()
                    
                    # Create overlays for each image in batch
                    batch_size = min(images.size(0), n_samples - sample_count)
                    
                    for i in range(batch_size):
                        # Convert image to numpy
                        image = images[i].cpu().numpy().transpose(1, 2, 0)
                        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0,1]
                        
                        # Get true and predicted grades
                        true_grade = cls_targets[i].item() if cls_targets is not None else -1
                        pred_grade = cls_preds[i].item()
                        
                        # Create overlay
                        overlay_info = {
                            'sample_id': sample_count,
                            'dataset': dataset_name,
                            'true_grade': true_grade,
                            'predicted_grade': pred_grade,
                            'grade_correct': true_grade == pred_grade if true_grade >= 0 else None
                        }
                        
                        # Create and save overlay image
                        overlay_path = self._create_single_overlay(
                            image, 
                            seg_targets[i].cpu().numpy() if seg_targets is not None else None,
                            seg_preds[i].cpu().numpy(),
                            overlay_info
                        )
                        
                        overlay_info['overlay_path'] = overlay_path
                        dataset_overlays.append(overlay_info)
                        
                        sample_count += 1
                        
                        if sample_count >= n_samples:
                            break
                    
                    if sample_count >= n_samples:
                        break
                
                overlay_results[dataset_name] = dataset_overlays
                
                if sample_count >= n_samples:
                    break
        
        # Create summary visualization
        self._create_overlay_summary(overlay_results)
        
        return overlay_results
    
    def _create_single_overlay(self, 
                             image: np.ndarray,
                             true_masks: Optional[np.ndarray],
                             pred_masks: np.ndarray,
                             overlay_info: Dict) -> str:
        """Create a single segmentation overlay visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # True grade vs predicted grade info
        true_grade_name = self.class_names[overlay_info['true_grade']] if overlay_info['true_grade'] >= 0 else 'Unknown'
        pred_grade_name = self.class_names[overlay_info['predicted_grade']]
        correct_symbol = "✓" if overlay_info['grade_correct'] else "✗" if overlay_info['grade_correct'] is not None else "?"
        
        axes[0, 1].text(0.5, 0.5, f"True: {true_grade_name}\nPred: {pred_grade_name}\n{correct_symbol}",
                       ha='center', va='center', fontsize=12, transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Classification Result')
        axes[0, 1].axis('off')
        
        # Predicted masks overlay
        pred_overlay = image.copy()
        colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5), (1, 1, 0, 0.5)]  # RGBA
        
        for lesion_idx in range(min(len(self.lesion_types), pred_masks.shape[0])):
            mask = pred_masks[lesion_idx] > 0.5
            if np.any(mask):
                # Create colored overlay
                color_overlay = np.zeros((*mask.shape, 4))
                color_overlay[mask] = colors[lesion_idx]
                
                # Blend with image
                pred_overlay = self._blend_overlay(pred_overlay, color_overlay[:, :, :3], 0.3)
        
        axes[0, 2].imshow(pred_overlay)
        axes[0, 2].set_title('Predicted Lesions')
        axes[0, 2].axis('off')
        
        # Individual lesion predictions
        for lesion_idx in range(min(len(self.lesion_types), 3)):
            if lesion_idx < pred_masks.shape[0]:
                mask = pred_masks[lesion_idx]
                axes[1, lesion_idx].imshow(mask, cmap='hot', alpha=0.8)
                axes[1, lesion_idx].imshow(image, alpha=0.5)
                axes[1, lesion_idx].set_title(f'{self.lesion_types[lesion_idx]}')
                axes[1, lesion_idx].axis('off')
            else:
                axes[1, lesion_idx].axis('off')
        
        plt.tight_layout()
        
        # Save overlay
        sample_id = overlay_info['sample_id']
        dataset = overlay_info['dataset']
        filename = f"overlay_{dataset}_{sample_id:04d}.png"
        output_path = self.output_dir / 'mask_overlays' / filename
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _blend_overlay(self, base_image: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
        """Blend overlay with base image"""
        return base_image * (1 - alpha) + overlay * alpha
    
    def _create_overlay_summary(self, overlay_results: Dict[str, List[Dict]]) -> None:
        """Create summary visualization of overlays"""
        
        # Create grid of example overlays
        n_datasets = len(overlay_results)
        examples_per_dataset = 4
        
        fig, axes = plt.subplots(n_datasets, examples_per_dataset, figsize=(20, 5 * n_datasets))
        
        if n_datasets == 1:
            axes = axes.reshape(1, -1)
        
        for i, (dataset_name, overlays) in enumerate(overlay_results.items()):
            for j in range(examples_per_dataset):
                if j < len(overlays):
                    # Load and display overlay
                    overlay_path = overlays[j]['overlay_path']
                    overlay_img = plt.imread(overlay_path)
                    
                    axes[i, j].imshow(overlay_img)
                    axes[i, j].set_title(f"{dataset_name}\nGrade: {overlays[j]['predicted_grade']}")
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        summary_path = self.output_dir / 'mask_overlays' / 'overlay_summary.png'
        plt.savefig(summary_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def _perform_cluster_analysis(self, feature_data: Dict[str, any]) -> Dict[str, any]:
        """Perform unsupervised clustering analysis on features"""
        
        logger.info("Performing cluster analysis")
        
        features = feature_data['features']
        cls_labels = feature_data['cls_labels']
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(50, features.shape[1]), random_state=42)
        features_pca = pca.fit_transform(features)
        
        # Try different numbers of clusters
        cluster_results = {}
        silhouette_scores = []
        
        for n_clusters in range(2, min(11, len(np.unique(cls_labels[cls_labels >= 0])) + 3)):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_pca)
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            sil_score = silhouette_score(features_pca, cluster_labels)
            silhouette_scores.append(sil_score)
            
            # Analyze cluster composition
            cluster_composition = self._analyze_cluster_composition(cluster_labels, cls_labels)
            
            cluster_results[n_clusters] = {
                'cluster_labels': cluster_labels,
                'silhouette_score': float(sil_score),
                'cluster_composition': cluster_composition
            }
        
        # Find optimal number of clusters
        optimal_k = np.argmax(silhouette_scores) + 2
        
        # Create cluster visualizations
        self._create_cluster_visualizations(
            features_pca, cluster_results, optimal_k, cls_labels
        )
        
        return {
            'cluster_results': cluster_results,
            'optimal_k': optimal_k,
            'silhouette_scores': silhouette_scores,
            'pca_components': features_pca
        }
    
    def _analyze_cluster_composition(self, cluster_labels: np.ndarray, 
                                   cls_labels: np.ndarray) -> Dict[str, any]:
        """Analyze the composition of clusters by DR grade"""
        
        composition = {}
        n_clusters = len(np.unique(cluster_labels))
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_grades = cls_labels[cluster_mask]
            
            # Remove unknown labels
            valid_grades = cluster_grades[cluster_grades >= 0]
            
            if len(valid_grades) > 0:
                grade_counts = np.bincount(valid_grades, minlength=5)
                grade_proportions = grade_counts / len(valid_grades)
                
                # Find dominant grade
                dominant_grade = np.argmax(grade_counts)
                dominant_proportion = grade_proportions[dominant_grade]
                
                composition[f'cluster_{cluster_id}'] = {
                    'size': int(np.sum(cluster_mask)),
                    'grade_counts': grade_counts.tolist(),
                    'grade_proportions': grade_proportions.tolist(),
                    'dominant_grade': int(dominant_grade),
                    'dominant_grade_name': self.class_names[dominant_grade],
                    'dominant_proportion': float(dominant_proportion),
                    'purity': float(dominant_proportion)  # How pure is this cluster
                }
        
        return composition
    
    def _create_cluster_visualizations(self, features_pca: np.ndarray,
                                     cluster_results: Dict,
                                     optimal_k: int,
                                     cls_labels: np.ndarray) -> None:
        """Create cluster analysis visualizations"""
        
        # Silhouette score plot
        plt.figure(figsize=(10, 6))
        k_values = list(cluster_results.keys())
        sil_scores = [cluster_results[k]['silhouette_score'] for k in k_values]
        
        plt.plot(k_values, sil_scores, 'bo-')
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Cluster Analysis: Silhouette Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / 'cluster_analysis' / 'silhouette_scores.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cluster visualization with optimal k
        optimal_clusters = cluster_results[optimal_k]['cluster_labels']
        
        # 2D PCA visualization
        plt.figure(figsize=(15, 5))
        
        # Plot by clusters
        plt.subplot(1, 3, 1)
        colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
        for cluster_id in range(optimal_k):
            mask = optimal_clusters == cluster_id
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1],
                       c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                       alpha=0.6, s=20)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Clusters (Unsupervised)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot by true grades
        plt.subplot(1, 3, 2)
        grade_colors = ['blue', 'green', 'orange', 'red', 'purple']
        for grade in range(5):
            mask = cls_labels == grade
            if np.any(mask):
                plt.scatter(features_pca[mask, 0], features_pca[mask, 1],
                           c=grade_colors[grade], label=self.class_names[grade],
                           alpha=0.6, s=20)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('True DR Grades')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cluster purity heatmap
        plt.subplot(1, 3, 3)
        composition = cluster_results[optimal_k]['cluster_composition']
        
        # Create purity matrix
        purity_matrix = np.zeros((optimal_k, 5))
        for cluster_info in composition.values():
            cluster_id = int(cluster_info['dominant_grade'])  # This is wrong, let me fix
            
        # Fix the purity matrix creation
        for cluster_name, cluster_info in composition.items():
            cluster_id = int(cluster_name.split('_')[1])
            proportions = cluster_info['grade_proportions']
            purity_matrix[cluster_id, :] = proportions
        
        sns.heatmap(purity_matrix, 
                   xticklabels=self.class_names,
                   yticklabels=[f'Cluster {i}' for i in range(optimal_k)],
                   annot=True, fmt='.2f', cmap='Blues')
        plt.title('Cluster Composition by Grade')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_analysis' / 'cluster_visualization.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comprehensive_visualizations(self, feature_data: Dict,
                                           tsne_results: Dict,
                                           cluster_results: Dict) -> None:
        """Create comprehensive visualization dashboard"""
        
        logger.info("Creating comprehensive visualizations")
        
        # Create a dashboard-style visualization
        fig = plt.figure(figsize=(20, 15))
        
        # t-SNE by grade
        ax1 = plt.subplot(2, 3, 1)
        tsne_coords = tsne_results['tsne_coordinates']
        cls_labels = feature_data['cls_labels']
        
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for grade in range(5):
            mask = cls_labels == grade
            if np.any(mask):
                plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                           c=colors[grade], label=self.class_names[grade],
                           alpha=0.6, s=10)
        plt.title('t-SNE by DR Grade')
        plt.legend()
        
        # t-SNE by dataset
        ax2 = plt.subplot(2, 3, 2)
        dataset_labels = feature_data['dataset_labels']
        unique_datasets = list(set(dataset_labels))
        dataset_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_datasets)))
        
        for i, dataset in enumerate(unique_datasets):
            mask = np.array(dataset_labels) == dataset
            plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                       c=[dataset_colors[i]], label=dataset, alpha=0.6, s=10)
        plt.title('t-SNE by Dataset')
        plt.legend()
        
        # Cluster analysis
        ax3 = plt.subplot(2, 3, 3)
        optimal_k = cluster_results['optimal_k']
        cluster_labels = cluster_results['cluster_results'][optimal_k]['cluster_labels']
        
        cluster_colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
        for cluster_id in range(optimal_k):
            mask = cluster_labels == cluster_id
            plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                       c=[cluster_colors[cluster_id]], label=f'Cluster {cluster_id}',
                       alpha=0.6, s=10)
        plt.title(f'Clusters (k={optimal_k})')
        plt.legend()
        
        # Performance metrics by grade
        ax4 = plt.subplot(2, 3, 4)
        cls_preds = feature_data['cls_preds']
        accuracies = []
        
        for grade in range(5):
            mask = cls_labels == grade
            if np.any(mask):
                accuracy = np.mean(cls_labels[mask] == cls_preds[mask])
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        bars = plt.bar(self.class_names, accuracies)
        plt.title('Accuracy by DR Grade')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Color bars based on performance
        for bar, acc in zip(bars, accuracies):
            if acc >= 0.8:
                bar.set_color('green')
            elif acc >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Confusion analysis in t-SNE space
        ax5 = plt.subplot(2, 3, 5)
        correct_mask = cls_labels == cls_preds
        
        plt.scatter(tsne_coords[correct_mask, 0], tsne_coords[correct_mask, 1],
                   c='green', alpha=0.6, s=10, label='Correct')
        plt.scatter(tsne_coords[~correct_mask, 0], tsne_coords[~correct_mask, 1],
                   c='red', alpha=0.6, s=10, label='Incorrect')
        plt.title('Prediction Accuracy in t-SNE Space')
        plt.legend()
        
        # Feature importance (if available)
        ax6 = plt.subplot(2, 3, 6)
        if tsne_results.get('pca_explained_variance') is not None:
            explained_var = tsne_results['pca_explained_variance'][:10]  # Top 10 components
            plt.bar(range(len(explained_var)), explained_var)
            plt.title('PCA Explained Variance (Top 10)')
            plt.xlabel('Component')
            plt.ylabel('Explained Variance Ratio')
        else:
            plt.text(0.5, 0.5, 'PCA not applied\n(Low-dimensional features)',
                    ha='center', va='center', transform=ax6.transAxes)
            plt.title('Feature Dimensionality')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_visualization_dashboard.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_interpretability_insights(self, feature_data: Dict,
                                          tsne_results: Dict,
                                          cluster_results: Dict) -> Dict[str, any]:
        """Generate interpretability insights from analysis"""
        
        insights = {}
        
        # Grade separation analysis
        tsne_analysis = tsne_results['analysis']
        grade_separations = tsne_analysis['grade_separations']
        
        # Find most/least separable grade pairs
        separations_list = [(pair, dist) for pair, dist in grade_separations.items()]
        separations_list.sort(key=lambda x: x[1])
        
        insights['grade_separation'] = {
            'most_separable': separations_list[-1],
            'least_separable': separations_list[0],
            'average_separation': float(np.mean(list(grade_separations.values())))
        }
        
        # Cluster analysis insights
        optimal_k = cluster_results['optimal_k']
        optimal_clusters = cluster_results['cluster_results'][optimal_k]
        
        # Find purest clusters
        compositions = optimal_clusters['cluster_composition']
        purities = [(name, info['purity']) for name, info in compositions.items()]
        purities.sort(key=lambda x: x[1], reverse=True)
        
        insights['clustering'] = {
            'optimal_k': optimal_k,
            'purest_cluster': purities[0] if purities else None,
            'least_pure_cluster': purities[-1] if purities else None,
            'average_purity': float(np.mean([info['purity'] for info in compositions.values()]))
        }
        
        # Model bias analysis
        cls_labels = feature_data['cls_labels']
        cls_preds = feature_data['cls_preds']
        
        grade_biases = {}
        for grade in range(5):
            true_mask = cls_labels == grade
            if np.any(true_mask):
                pred_distribution = np.bincount(cls_preds[true_mask], minlength=5)
                pred_distribution = pred_distribution / np.sum(pred_distribution)
                
                # Calculate bias towards other grades
                bias_scores = {}
                for other_grade in range(5):
                    if other_grade != grade:
                        bias_scores[self.class_names[other_grade]] = float(pred_distribution[other_grade])
                
                grade_biases[self.class_names[grade]] = bias_scores
        
        insights['model_bias'] = grade_biases
        
        # Feature space analysis
        features = feature_data['features']
        feature_analysis = {
            'dimensionality': features.shape[1],
            'feature_range': {
                'min': float(np.min(features)),
                'max': float(np.max(features)),
                'mean': float(np.mean(features)),
                'std': float(np.std(features))
            }
        }
        
        insights['feature_space'] = feature_analysis
        
        return insights
    
    def _generate_explainability_report(self, tsne_results: Dict,
                                      overlay_results: Dict,
                                      cluster_results: Dict,
                                      insights: Dict) -> str:
        """Generate comprehensive explainability report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'explainability_report_{timestamp}.html'
        
        html_content = self._generate_explainability_html(
            tsne_results, overlay_results, cluster_results, insights
        )
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Explainability report saved to: {report_path}")
        return str(report_path)
    
    def _generate_explainability_html(self, tsne_results: Dict,
                                    overlay_results: Dict,
                                    cluster_results: Dict,
                                    insights: Dict) -> str:
        """Generate HTML explainability report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phase 6: Explainability & Visualization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .insight {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }}
                .metric {{ margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Phase 6: Explainability & Visualization Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Analysis Type:</strong> t-SNE, Segmentation Overlays, Cluster Analysis</p>
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
                
                <div class="insight">
                    <h3>Grade Separability</h3>
                    <div class="metric">Most Separable: {insights['grade_separation']['most_separable'][0]} 
                        (Distance: {insights['grade_separation']['most_separable'][1]:.3f})</div>
                    <div class="metric">Least Separable: {insights['grade_separation']['least_separable'][0]} 
                        (Distance: {insights['grade_separation']['least_separable'][1]:.3f})</div>
                    <div class="metric">Average Separation: {insights['grade_separation']['average_separation']:.3f}</div>
                </div>
                
                <div class="insight">
                    <h3>Clustering Analysis</h3>
                    <div class="metric">Optimal Number of Clusters: {insights['clustering']['optimal_k']}</div>
                    <div class="metric">Average Cluster Purity: {insights['clustering']['average_purity']:.3f}</div>
        """
        
        if insights['clustering']['purest_cluster']:
            html += f"""
                    <div class="metric">Purest Cluster: {insights['clustering']['purest_cluster'][0]} 
                        (Purity: {insights['clustering']['purest_cluster'][1]:.3f})</div>
            """
        
        html += """
                </div>
            </div>
            
            <div class="section">
                <h2>t-SNE Analysis Results</h2>
        """
        
        # Add t-SNE analysis details
        tsne_analysis = tsne_results['analysis']
        
        html += """
                <h3>Grade Purity in Feature Space</h3>
                <table>
                    <tr>
                        <th>Grade</th>
                        <th>Sample Count</th>
                        <th>Mean Distance to Center</th>
                        <th>Std Distance to Center</th>
                    </tr>
        """
        
        for grade_name, purity_info in tsne_analysis['grade_purities'].items():
            html += f"""
                    <tr>
                        <td>{grade_name}</td>
                        <td>{purity_info['sample_count']}</td>
                        <td>{purity_info['mean_distance_to_center']:.3f}</td>
                        <td>{purity_info['std_distance_to_center']:.3f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Segmentation Overlay Analysis</h2>
        """
        
        # Add overlay statistics
        total_overlays = sum(len(overlays) for overlays in overlay_results.values())
        html += f"<div class='metric'>Total Overlay Visualizations Created: {total_overlays}</div>"
        
        for dataset_name, overlays in overlay_results.items():
            correct_predictions = sum(1 for o in overlays if o.get('grade_correct') == True)
            total_predictions = sum(1 for o in overlays if o.get('grade_correct') is not None)
            
            html += f"""
                <div class="metric">{dataset_name}: {len(overlays)} overlays created</div>
            """
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                html += f"""
                    <div class="metric">{dataset_name} Classification Accuracy: {accuracy:.3f} 
                        ({correct_predictions}/{total_predictions})</div>
                """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Model Bias Analysis</h2>
        """
        
        # Add bias analysis
        for true_grade, bias_scores in insights['model_bias'].items():
            html += f"""
                <h3>Bias for {true_grade}</h3>
                <table>
                    <tr>
                        <th>Confused with Grade</th>
                        <th>Confusion Rate</th>
                    </tr>
            """
            
            # Sort by confusion rate
            sorted_biases = sorted(bias_scores.items(), key=lambda x: x[1], reverse=True)
            
            for confused_grade, rate in sorted_biases[:3]:  # Top 3 confusions
                html += f"""
                    <tr>
                        <td>{confused_grade}</td>
                        <td>{rate:.3f}</td>
                    </tr>
                """
            
            html += "</table>"
        
        html += """
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <div class="insight">
        """
        
        # Generate recommendations based on insights
        recommendations = self._generate_explainability_recommendations(insights)
        
        for recommendation in recommendations:
            html += f"<div class='metric'>• {recommendation}</div>"
        
        html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_explainability_recommendations(self, insights: Dict) -> List[str]:
        """Generate recommendations based on explainability analysis"""
        
        recommendations = []
        
        # Grade separation recommendations
        least_sep = insights['grade_separation']['least_separable']
        if least_sep[1] < 2.0:  # Arbitrary threshold
            recommendations.append(
                f"Consider improving feature discrimination between {least_sep[0]} "
                f"(separation distance: {least_sep[1]:.3f})"
            )
        
        # Clustering recommendations
        avg_purity = insights['clustering']['average_purity']
        if avg_purity < 0.7:
            recommendations.append(
                f"Low cluster purity ({avg_purity:.3f}) suggests mixed feature representations. "
                "Consider additional training or different architecture."
            )
        
        # Model bias recommendations
        high_bias_grades = []
        for grade, biases in insights['model_bias'].items():
            max_bias = max(biases.values()) if biases else 0
            if max_bias > 0.3:  # 30% confusion rate
                high_bias_grades.append(grade)
        
        if high_bias_grades:
            recommendations.append(
                f"High confusion rates detected for: {', '.join(high_bias_grades)}. "
                "Consider class balancing or targeted data augmentation."
            )
        
        # Feature space recommendations
        feature_analysis = insights['feature_space']
        if feature_analysis['dimensionality'] > 1000:
            recommendations.append(
                "High-dimensional feature space detected. Consider dimensionality reduction "
                "or feature selection for improved interpretability."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Feature representations appear well-separated and interpretable.")
            recommendations.append("Model shows good clustering properties aligned with clinical grades.")
        
        return recommendations
