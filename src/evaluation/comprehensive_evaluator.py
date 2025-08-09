"""
Phase 6: Comprehensive Model Evaluator
Main evaluation orchestrator for multi-task diabetic retinopathy models.
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
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from .metrics_calculator import MetricsCalculator, EvaluationResults, ClassificationMetrics, SegmentationMetrics
from ..models.multi_task_model import MultiTaskRetinaModel
from ..data.datasets import GradingRetinaDataset, SegmentationRetinaDataset, MultiTaskRetinaDataset

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for multi-task diabetic retinopathy models.
    Handles Step 6.1: Comprehensive Performance Evaluation
    """
    
    def __init__(self,
                 num_classes: int = 5,
                 device: str = 'cuda',
                 output_dir: str = 'results/phase6_evaluation'):
        """
        Initialize comprehensive evaluator
        
        Args:
            num_classes: Number of DR severity classes
            device: Device for evaluation ('cuda' or 'cpu')
            output_dir: Directory to save evaluation results
        """
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(num_classes=num_classes)
        
        # Class names for DR grading
        self.class_names = [
            'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'
        ]
        
        # Lesion types for segmentation
        self.lesion_types = [
            'microaneurysms', 'hemorrhages', 'hard_exudates', 'soft_exudates'
        ]
        
        logger.info(f"ComprehensiveEvaluator initialized on {self.device}")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def evaluate_model_checkpoint(self,
                                checkpoint_path: str,
                                validation_loaders: Dict[str, DataLoader],
                                model_config: Optional[Dict] = None) -> Dict[str, EvaluationResults]:
        """
        Evaluate a model checkpoint on multiple validation datasets
        
        Args:
            checkpoint_path: Path to model checkpoint
            validation_loaders: Dictionary of dataset_name -> DataLoader
            model_config: Model configuration (if needed for loading)
            
        Returns:
            Dictionary of dataset_name -> EvaluationResults
        """
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Load model
        model = self._load_model_from_checkpoint(checkpoint_path, model_config)
        model.eval()
        
        # Evaluate on all datasets
        all_results = {}
        
        for dataset_name, data_loader in validation_loaders.items():
            logger.info(f"Evaluating on {dataset_name} dataset")
            
            try:
                results = self._evaluate_single_dataset(
                    model, data_loader, dataset_name, checkpoint_path
                )
                all_results[dataset_name] = results
                
                logger.info(f"Completed evaluation on {dataset_name}")
                logger.info(f"  Classification Accuracy: {results.classification.accuracy:.4f}")
                logger.info(f"  Segmentation Dice: {results.segmentation.dice_coefficient:.4f}")
                logger.info(f"  Combined Score: {results.combined_score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate on {dataset_name}: {e}")
                continue
        
        # Save comprehensive results
        self._save_evaluation_results(all_results, checkpoint_path)
        
        return all_results
    
    def evaluate_multiple_checkpoints(self,
                                    checkpoint_paths: List[str],
                                    validation_loaders: Dict[str, DataLoader],
                                    model_config: Optional[Dict] = None) -> Dict[str, Dict[str, EvaluationResults]]:
        """
        Evaluate multiple model checkpoints
        
        Args:
            checkpoint_paths: List of checkpoint paths
            validation_loaders: Dictionary of dataset_name -> DataLoader
            model_config: Model configuration
            
        Returns:
            Dictionary of checkpoint_name -> {dataset_name -> EvaluationResults}
        """
        all_checkpoint_results = {}
        
        for checkpoint_path in checkpoint_paths:
            checkpoint_name = Path(checkpoint_path).stem
            logger.info(f"Evaluating checkpoint: {checkpoint_name}")
            
            try:
                results = self.evaluate_model_checkpoint(
                    checkpoint_path, validation_loaders, model_config
                )
                all_checkpoint_results[checkpoint_name] = results
                
            except Exception as e:
                logger.error(f"Failed to evaluate checkpoint {checkpoint_name}: {e}")
                continue
        
        # Generate comparative analysis
        self._generate_comparative_analysis(all_checkpoint_results)
        
        return all_checkpoint_results
    
    def _evaluate_single_dataset(self,
                               model: nn.Module,
                               data_loader: DataLoader,
                               dataset_name: str,
                               model_path: str) -> EvaluationResults:
        """Evaluate model on a single dataset"""
        
        # Storage for predictions and targets
        all_cls_preds = []
        all_cls_targets = []
        all_cls_probs = []
        
        all_seg_preds = []
        all_seg_targets = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Extract batch data - handle both tuple and dict formats
                if isinstance(batch, (list, tuple)):
                    # Handle tuple format: (images, labels) or (images, labels, masks)
                    images = batch[0].to(self.device)
                    cls_targets = batch[1].to(self.device) if len(batch) > 1 else None
                    seg_targets = batch[2].to(self.device) if len(batch) > 2 else None
                elif isinstance(batch, dict):
                    # Handle dictionary format
                    images = batch['image'].to(self.device)
                    cls_targets = batch.get('grade', batch.get('label'))
                    seg_targets = batch.get('mask', batch.get('segmentation'))
                    
                    if cls_targets is not None:
                        cls_targets = cls_targets.to(self.device)
                    if seg_targets is not None:
                        seg_targets = seg_targets.to(self.device)
                else:
                    logger.error(f"Unknown batch format: {type(batch)}")
                    continue
                
                # Forward pass
                cls_outputs, seg_outputs = model(images)
                
                # Process classification outputs
                if cls_targets is not None:
                    cls_probs = torch.softmax(cls_outputs, dim=1)
                    cls_preds = torch.argmax(cls_probs, dim=1)
                    
                    all_cls_targets.extend(cls_targets.cpu().numpy())
                    all_cls_preds.extend(cls_preds.cpu().numpy())
                    all_cls_probs.extend(cls_probs.cpu().numpy())
                
                # Process segmentation outputs
                if seg_targets is not None:
                    seg_probs = torch.sigmoid(seg_outputs)
                    seg_preds = (seg_probs > 0.5).float()
                    
                    all_seg_targets.extend(seg_targets.cpu().numpy())
                    all_seg_preds.extend(seg_preds.cpu().numpy())
                
                # Progress logging
                if batch_idx % 50 == 0:
                    logger.info(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Convert to numpy arrays
        all_cls_targets = np.array(all_cls_targets) if all_cls_targets else np.array([])
        all_cls_preds = np.array(all_cls_preds) if all_cls_preds else np.array([])
        all_cls_probs = np.array(all_cls_probs) if all_cls_probs else np.array([])
        
        all_seg_targets = np.array(all_seg_targets) if all_seg_targets else np.array([])
        all_seg_preds = np.array(all_seg_preds) if all_seg_preds else np.array([])
        
        # Calculate metrics
        if len(all_cls_targets) > 0:
            cls_metrics = self.metrics_calculator.calculate_classification_metrics(
                all_cls_targets, all_cls_preds, all_cls_probs
            )
        else:
            cls_metrics = self._create_empty_classification_metrics()
        
        if len(all_seg_targets) > 0:
            seg_metrics = self.metrics_calculator.calculate_segmentation_metrics(
                all_seg_targets, all_seg_preds
            )
        else:
            seg_metrics = self._create_empty_segmentation_metrics()
        
        # Calculate combined score
        combined_score = self._calculate_combined_score(cls_metrics, seg_metrics)
        
        # Create evaluation results
        results = EvaluationResults(
            classification=cls_metrics,
            segmentation=seg_metrics,
            combined_score=combined_score,
            dataset_name=dataset_name,
            model_name=Path(model_path).stem,
            timestamp=datetime.now().isoformat()
        )
        
        return results
    
    def _load_model_from_checkpoint(self,
                                  checkpoint_path: str,
                                  model_config: Optional[Dict] = None) -> nn.Module:
        """Load model from checkpoint"""
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Create model with default configuration
        if model_config:
            # Use provided config
            model = MultiTaskRetinaModel(
                num_classes_cls=model_config.get('num_classes', self.num_classes),
                num_classes_seg=model_config.get('num_segmentation_classes', len(self.lesion_types)),
                backbone_name=model_config.get('backbone', 'tf_efficientnet_b0_ns'),
                pretrained=False  # Don't load pretrained weights when loading from checkpoint
            )
        else:
            # Use default configuration
            model = MultiTaskRetinaModel(
                num_classes_cls=self.num_classes,
                num_classes_seg=len(self.lesion_types),
                backbone_name='tf_efficientnet_b0_ns',
                pretrained=False
            )
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(f"Failed to load with strict=True, trying strict=False: {e}")
            model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        
        return model
    
    def _calculate_combined_score(self,
                                cls_metrics: ClassificationMetrics,
                                seg_metrics: SegmentationMetrics,
                                cls_weight: float = 0.6,
                                seg_weight: float = 0.4) -> float:
        """Calculate combined performance score"""
        
        # Use quadratic weighted kappa for classification (more appropriate for ordinal data)
        cls_score = cls_metrics.kappa if cls_metrics.kappa > 0 else 0.0
        
        # Use Dice coefficient for segmentation
        seg_score = seg_metrics.dice_coefficient
        
        # Weighted combination
        combined = (cls_weight * cls_score) + (seg_weight * seg_score)
        
        return float(combined)
    
    def _create_empty_classification_metrics(self) -> ClassificationMetrics:
        """Create empty classification metrics for cases with no classification data"""
        return ClassificationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            kappa=0.0,
            auc_roc=0.0,
            per_class_accuracy={name: 0.0 for name in self.class_names},
            confusion_matrix=np.zeros((self.num_classes, self.num_classes)),
            classification_report={},
            confidence_stats={}
        )
    
    def _create_empty_segmentation_metrics(self) -> SegmentationMetrics:
        """Create empty segmentation metrics for cases with no segmentation data"""
        return SegmentationMetrics(
            dice_coefficient=0.0,
            iou_score=0.0,
            pixel_accuracy=0.0,
            precision=0.0,
            recall=0.0,
            per_lesion_dice={lesion: 0.0 for lesion in self.lesion_types},
            per_lesion_iou={lesion: 0.0 for lesion in self.lesion_types}
        )
    
    def _save_evaluation_results(self,
                               results: Dict[str, EvaluationResults],
                               checkpoint_path: str) -> None:
        """Save evaluation results to files"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = Path(checkpoint_path).stem
        
        # Create output directory for this evaluation
        eval_dir = self.output_dir / f"eval_{checkpoint_name}_{timestamp}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        json_results = {}
        for dataset_name, result in results.items():
            json_results[dataset_name] = {
                'model_name': result.model_name,
                'dataset_name': result.dataset_name,
                'timestamp': result.timestamp,
                'combined_score': result.combined_score,
                'classification': {
                    'accuracy': result.classification.accuracy,
                    'precision': result.classification.precision,
                    'recall': result.classification.recall,
                    'f1_score': result.classification.f1_score,
                    'kappa': result.classification.kappa,
                    'auc_roc': result.classification.auc_roc,
                    'per_class_accuracy': result.classification.per_class_accuracy,
                    'confusion_matrix': result.classification.confusion_matrix.tolist(),
                    'confidence_stats': result.classification.confidence_stats
                },
                'segmentation': {
                    'dice_coefficient': result.segmentation.dice_coefficient,
                    'iou_score': result.segmentation.iou_score,
                    'pixel_accuracy': result.segmentation.pixel_accuracy,
                    'precision': result.segmentation.precision,
                    'recall': result.segmentation.recall,
                    'per_lesion_dice': result.segmentation.per_lesion_dice,
                    'per_lesion_iou': result.segmentation.per_lesion_iou
                }
            }
        
        json_path = eval_dir / 'detailed_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save summary table as CSV
        summary_data = []
        for dataset_name, result in results.items():
            summary_data.append({
                'dataset': dataset_name,
                'model': result.model_name,
                'combined_score': result.combined_score,
                'cls_accuracy': result.classification.accuracy,
                'cls_kappa': result.classification.kappa,
                'cls_auc_roc': result.classification.auc_roc,
                'seg_dice': result.segmentation.dice_coefficient,
                'seg_iou': result.segmentation.iou_score,
                'timestamp': result.timestamp
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = eval_dir / 'summary_metrics.csv'
        summary_df.to_csv(csv_path, index=False)
        
        logger.info(f"Evaluation results saved to: {eval_dir}")
    
    def _generate_comparative_analysis(self,
                                     all_results: Dict[str, Dict[str, EvaluationResults]]) -> None:
        """Generate comparative analysis across multiple checkpoints"""
        
        # Prepare comparative data
        comparative_data = []
        
        for checkpoint_name, checkpoint_results in all_results.items():
            for dataset_name, result in checkpoint_results.items():
                comparative_data.append({
                    'checkpoint': checkpoint_name,
                    'dataset': dataset_name,
                    'combined_score': result.combined_score,
                    'cls_accuracy': result.classification.accuracy,
                    'cls_kappa': result.classification.kappa,
                    'cls_auc_roc': result.classification.auc_roc,
                    'seg_dice': result.segmentation.dice_coefficient,
                    'seg_iou': result.segmentation.iou_score,
                    'timestamp': result.timestamp
                })
        
        # Create comparative DataFrame
        df = pd.DataFrame(comparative_data)
        
        # Save comparative analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparative_path = self.output_dir / f'comparative_analysis_{timestamp}.csv'
        df.to_csv(comparative_path, index=False)
        
        # Calculate summary statistics
        summary_stats = df.groupby(['dataset']).agg({
            'combined_score': ['mean', 'std', 'min', 'max'],
            'cls_accuracy': ['mean', 'std', 'min', 'max'],
            'cls_kappa': ['mean', 'std', 'min', 'max'],
            'seg_dice': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Save summary statistics
        summary_path = self.output_dir / f'summary_statistics_{timestamp}.csv'
        summary_stats.to_csv(summary_path)
        
        logger.info(f"Comparative analysis saved to: {comparative_path}")
        logger.info(f"Summary statistics saved to: {summary_path}")
    
    def generate_evaluation_report(self,
                                 results: Dict[str, EvaluationResults],
                                 output_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report"""
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f'evaluation_report_{timestamp}.html'
        
        # Generate HTML report
        html_content = self._generate_html_report(results)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to: {output_path}")
        return str(output_path)
    
    def _generate_html_report(self, results: Dict[str, EvaluationResults]) -> str:
        """Generate HTML evaluation report"""
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phase 6: Comprehensive Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .dataset { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                .metric { margin: 5px 0; }
                .good { color: green; font-weight: bold; }
                .moderate { color: orange; font-weight: bold; }
                .poor { color: red; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Phase 6: Comprehensive Model Evaluation Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Datasets Evaluated:</strong> {num_datasets}</p>
            </div>
        """.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            num_datasets=len(results)
        )
        
        # Add summary table
        html += """
            <div class="section">
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Dataset</th>
                        <th>Combined Score</th>
                        <th>Classification Accuracy</th>
                        <th>Classification Kappa</th>
                        <th>Segmentation Dice</th>
                        <th>Segmentation IoU</th>
                    </tr>
        """
        
        for dataset_name, result in results.items():
            html += f"""
                    <tr>
                        <td>{dataset_name}</td>
                        <td class="{self._get_score_class(result.combined_score)}">{result.combined_score:.4f}</td>
                        <td class="{self._get_score_class(result.classification.accuracy)}">{result.classification.accuracy:.4f}</td>
                        <td class="{self._get_score_class(result.classification.kappa)}">{result.classification.kappa:.4f}</td>
                        <td class="{self._get_score_class(result.segmentation.dice_coefficient)}">{result.segmentation.dice_coefficient:.4f}</td>
                        <td class="{self._get_score_class(result.segmentation.iou_score)}">{result.segmentation.iou_score:.4f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        # Add detailed results for each dataset
        for dataset_name, result in results.items():
            html += f"""
            <div class="dataset">
                <h3>{dataset_name} Dataset Results</h3>
                
                <h4>Classification Metrics</h4>
                <div class="metric">Accuracy: {result.classification.accuracy:.4f}</div>
                <div class="metric">Precision: {result.classification.precision:.4f}</div>
                <div class="metric">Recall: {result.classification.recall:.4f}</div>
                <div class="metric">F1-Score: {result.classification.f1_score:.4f}</div>
                <div class="metric">Quadratic Kappa: {result.classification.kappa:.4f}</div>
                <div class="metric">ROC-AUC: {result.classification.auc_roc:.4f}</div>
                
                <h4>Per-Class Accuracy</h4>
            """
            
            for class_name, acc in result.classification.per_class_accuracy.items():
                html += f'<div class="metric">{class_name}: {acc:.4f}</div>'
            
            html += f"""
                <h4>Segmentation Metrics</h4>
                <div class="metric">Dice Coefficient: {result.segmentation.dice_coefficient:.4f}</div>
                <div class="metric">IoU Score: {result.segmentation.iou_score:.4f}</div>
                <div class="metric">Pixel Accuracy: {result.segmentation.pixel_accuracy:.4f}</div>
                <div class="metric">Precision: {result.segmentation.precision:.4f}</div>
                <div class="metric">Recall: {result.segmentation.recall:.4f}</div>
                
                <h4>Per-Lesion Dice Scores</h4>
            """
            
            for lesion, dice in result.segmentation.per_lesion_dice.items():
                html += f'<div class="metric">{lesion}: {dice:.4f}</div>'
            
            html += """
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score value"""
        if score >= 0.8:
            return 'good'
        elif score >= 0.6:
            return 'moderate'
        else:
            return 'poor'
