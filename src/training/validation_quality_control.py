"""
Phase 5: Validation and Quality Control System
Comprehensive validation framework for ensuring model quality and reliability.
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import tempfile
import shutil
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
try:
    from ..models.multi_task_model import MultiTaskRetinaModel
    from ..data.datasets import APTOSDataset, IDRiDDataset
    from ..data.dataloaders import create_dataloaders
    from ..utils.metrics import calculate_dice_score, calculate_iou_score
    from ..utils.visualization import visualize_predictions
except ImportError as e:
    logging.warning(f"Import warning: {e}")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation procedures"""
    
    # Cross-validation settings
    cv_folds: int = 5
    stratify_by_grade: bool = True
    
    # Test-time augmentation
    use_tta: bool = True
    tta_transforms: int = 8
    
    # Validation thresholds
    min_accuracy: float = 0.7
    min_dice_score: float = 0.5
    min_stability_score: float = 0.8
    
    # Quality checks
    check_gradient_flow: bool = True
    check_feature_distribution: bool = True
    check_prediction_confidence: bool = True
    
    # Output settings
    save_predictions: bool = True
    save_visualizations: bool = True
    generate_detailed_report: bool = True


@dataclass
class ValidationResults:
    """Results from model validation"""
    
    # Overall validation info
    validation_id: str
    model_path: str
    dataset_name: str
    validation_timestamp: float
    
    # Performance metrics
    overall_accuracy: float
    class_accuracies: Dict[str, float]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    
    # Segmentation metrics (if applicable)
    dice_scores: Optional[Dict[str, float]] = None
    iou_scores: Optional[Dict[str, float]] = None
    
    # Cross-validation results
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Quality assessments
    stability_score: Optional[float] = None
    confidence_analysis: Optional[Dict[str, float]] = None
    gradient_flow_check: Optional[bool] = None
    
    # Additional analysis
    error_analysis: Optional[Dict[str, Any]] = None
    robustness_analysis: Optional[Dict[str, Any]] = None
    
    # Validation status
    passed_validation: bool = False
    validation_notes: List[str] = None


class BaseValidator(ABC):
    """Base class for model validators"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def validate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                device: torch.device) -> ValidationResults:
        """Perform validation"""
        pass
    
    @abstractmethod
    def generate_report(self, results: ValidationResults, output_dir: str) -> str:
        """Generate validation report"""
        pass


class ClassificationValidator(BaseValidator):
    """Validator for classification models"""
    
    def __init__(self, config: ValidationConfig, num_classes: int = 5):
        super().__init__(config)
        self.num_classes = num_classes
        self.class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    def validate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                device: torch.device) -> ValidationResults:
        """Perform comprehensive classification validation"""
        
        validation_id = f"classification_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting classification validation: {validation_id}")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        prediction_confidences = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['image'].to(device)
                targets = batch['grade'].to(device)
                
                # Forward pass
                if hasattr(model, 'classify'):
                    outputs = model.classify(images)
                else:
                    outputs = model(images)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # Calculate prediction confidence (max probability)
                confidence = torch.max(probabilities, dim=1)[0]
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                prediction_confidences.extend(confidence.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    self.logger.info(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        prediction_confidences = np.array(prediction_confidences)
        
        # Calculate basic metrics
        overall_accuracy = accuracy_score(all_targets, all_predictions)
        class_accuracies = {}
        
        for i, class_name in enumerate(self.class_names):
            class_mask = all_targets == i
            if np.any(class_mask):
                class_acc = accuracy_score(all_targets[class_mask], all_predictions[class_mask])
                class_accuracies[class_name] = float(class_acc)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Classification report
        report = classification_report(all_targets, all_predictions, 
                                     target_names=self.class_names, output_dict=True)
        
        # Confidence analysis
        confidence_analysis = {
            'mean_confidence': float(np.mean(prediction_confidences)),
            'std_confidence': float(np.std(prediction_confidences)),
            'low_confidence_ratio': float(np.mean(prediction_confidences < 0.5)),
            'high_confidence_ratio': float(np.mean(prediction_confidences > 0.9))
        }
        
        # Error analysis
        error_analysis = self._perform_error_analysis(
            all_targets, all_predictions, all_probabilities, prediction_confidences
        )
        
        # Cross-validation if requested
        cv_scores = None
        cv_mean = None
        cv_std = None
        
        if self.config.cv_folds > 1:
            self.logger.info("Performing cross-validation...")
            cv_scores, cv_mean, cv_std = self._perform_cross_validation(
                model, data_loader, device
            )
        
        # Test-time augmentation
        if self.config.use_tta:
            self.logger.info("Performing test-time augmentation validation...")
            tta_accuracy = self._perform_tta_validation(model, data_loader, device)
            confidence_analysis['tta_accuracy'] = tta_accuracy
        
        # Create validation results
        results = ValidationResults(
            validation_id=validation_id,
            model_path="",  # To be filled by caller
            dataset_name="classification_dataset",
            validation_timestamp=datetime.now().timestamp(),
            overall_accuracy=float(overall_accuracy),
            class_accuracies=class_accuracies,
            confusion_matrix=cm.tolist(),
            classification_report=report,
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            confidence_analysis=confidence_analysis,
            error_analysis=error_analysis,
            validation_notes=[]
        )
        
        # Check validation criteria
        results.passed_validation = self._check_validation_criteria(results)
        
        return results
    
    def _perform_error_analysis(self, targets: np.ndarray, predictions: np.ndarray,
                               probabilities: np.ndarray, confidences: np.ndarray) -> Dict[str, Any]:
        """Perform detailed error analysis"""
        
        # Find misclassified samples
        misclassified = targets != predictions
        error_indices = np.where(misclassified)[0]
        
        error_analysis = {
            'total_errors': int(np.sum(misclassified)),
            'error_rate': float(np.mean(misclassified)),
            'error_by_class': {},
            'confusion_patterns': {},
            'confidence_of_errors': {}
        }
        
        # Error analysis by class
        for i, class_name in enumerate(self.class_names):
            class_mask = targets == i
            if np.any(class_mask):
                class_errors = misclassified[class_mask]
                error_analysis['error_by_class'][class_name] = {
                    'error_count': int(np.sum(class_errors)),
                    'error_rate': float(np.mean(class_errors)),
                    'total_samples': int(np.sum(class_mask))
                }
        
        # Confusion patterns (most common misclassifications)
        if error_indices.size > 0:
            confusion_pairs = list(zip(targets[error_indices], predictions[error_indices]))
            from collections import Counter
            common_confusions = Counter(confusion_pairs).most_common(5)
            
            for (true_class, pred_class), count in common_confusions:
                pattern_name = f"{self.class_names[true_class]} -> {self.class_names[pred_class]}"
                error_analysis['confusion_patterns'][pattern_name] = {
                    'count': count,
                    'percentage': float(count / len(error_indices) * 100)
                }
        
        # Confidence analysis of errors
        if error_indices.size > 0:
            error_confidences = confidences[error_indices]
            correct_confidences = confidences[~misclassified]
            
            error_analysis['confidence_of_errors'] = {
                'mean_error_confidence': float(np.mean(error_confidences)),
                'mean_correct_confidence': float(np.mean(correct_confidences)),
                'high_confidence_errors': int(np.sum(error_confidences > 0.8)),
                'low_confidence_correct': int(np.sum(correct_confidences < 0.5))
            }
        
        return error_analysis
    
    def _perform_cross_validation(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                 device: torch.device) -> Tuple[List[float], float, float]:
        """Perform cross-validation"""
        
        # Extract all data from dataloader
        all_data = []
        all_targets = []
        
        for batch in data_loader:
            for i in range(batch['image'].size(0)):
                all_data.append(batch['image'][i])
                all_targets.append(batch['grade'][i].item())
        
        all_targets = np.array(all_targets)
        
        # Setup cross-validation
        if self.config.stratify_by_grade:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(all_data, all_targets)):
            self.logger.info(f"Cross-validation fold {fold + 1}/{self.config.cv_folds}")
            
            # Create validation subset
            val_data = [all_data[i] for i in val_idx]
            val_targets = all_targets[val_idx]
            
            # Create temporary dataloader for this fold
            val_dataset = torch.utils.data.TensorDataset(
                torch.stack(val_data),
                torch.tensor(val_targets, dtype=torch.long)
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=32, shuffle=False
            )
            
            # Evaluate on this fold
            fold_accuracy = self._evaluate_fold(model, val_loader, device)
            cv_scores.append(fold_accuracy)
        
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        
        return cv_scores, cv_mean, cv_std
    
    def _evaluate_fold(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                      device: torch.device) -> float:
        """Evaluate model on a single fold"""
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)
                
                if hasattr(model, 'classify'):
                    outputs = model.classify(images)
                else:
                    outputs = model(images)
                
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        return correct / total
    
    def _perform_tta_validation(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                               device: torch.device) -> float:
        """Perform test-time augmentation validation"""
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(device)
                targets = batch['grade'].to(device)
                
                # Apply multiple augmentations and average predictions
                tta_predictions = []
                
                for _ in range(self.config.tta_transforms):
                    # Apply random augmentation
                    aug_images = self._apply_tta_augmentation(images)
                    
                    if hasattr(model, 'classify'):
                        outputs = model.classify(aug_images)
                    else:
                        outputs = model(aug_images)
                    
                    probabilities = torch.softmax(outputs, dim=1)
                    tta_predictions.append(probabilities)
                
                # Average predictions across augmentations
                avg_predictions = torch.mean(torch.stack(tta_predictions), dim=0)
                final_predictions = torch.argmax(avg_predictions, dim=1)
                
                all_predictions.extend(final_predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return accuracy_score(all_targets, all_predictions)
    
    def _apply_tta_augmentation(self, images: torch.Tensor) -> torch.Tensor:
        """Apply test-time augmentation"""
        # Simple TTA: random horizontal flip
        if torch.rand(1) > 0.5:
            images = torch.flip(images, dims=[3])  # Horizontal flip
        
        return images
    
    def _check_validation_criteria(self, results: ValidationResults) -> bool:
        """Check if model passes validation criteria"""
        
        criteria_passed = []
        notes = []
        
        # Check overall accuracy
        if results.overall_accuracy >= self.config.min_accuracy:
            criteria_passed.append(True)
            notes.append(f"✓ Overall accuracy ({results.overall_accuracy:.3f}) meets minimum requirement")
        else:
            criteria_passed.append(False)
            notes.append(f"✗ Overall accuracy ({results.overall_accuracy:.3f}) below minimum ({self.config.min_accuracy})")
        
        # Check cross-validation stability
        if results.cv_std is not None:
            stability_score = 1.0 - (results.cv_std / results.cv_mean) if results.cv_mean > 0 else 0.0
            results.stability_score = stability_score
            
            if stability_score >= self.config.min_stability_score:
                criteria_passed.append(True)
                notes.append(f"✓ Model stability ({stability_score:.3f}) is acceptable")
            else:
                criteria_passed.append(False)
                notes.append(f"✗ Model stability ({stability_score:.3f}) below threshold ({self.config.min_stability_score})")
        
        # Check confidence distribution
        if results.confidence_analysis:
            low_conf_ratio = results.confidence_analysis['low_confidence_ratio']
            if low_conf_ratio < 0.3:  # Less than 30% low confidence predictions
                criteria_passed.append(True)
                notes.append(f"✓ Low confidence predictions ({low_conf_ratio:.1%}) within acceptable range")
            else:
                criteria_passed.append(False)
                notes.append(f"✗ Too many low confidence predictions ({low_conf_ratio:.1%})")
        
        results.validation_notes = notes
        return all(criteria_passed)
    
    def generate_report(self, results: ValidationResults, output_dir: str) -> str:
        """Generate comprehensive validation report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"classification_validation_report_{results.validation_id}.html"
        
        # Generate HTML report
        html_content = self._generate_html_report(results)
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        # Save detailed results as JSON
        json_file = output_path / f"validation_results_{results.validation_id}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)
        
        # Generate visualizations
        if self.config.save_visualizations:
            self._generate_visualizations(results, output_path)
        
        self.logger.info(f"Validation report saved to: {report_file}")
        return str(report_file)
    
    def _generate_html_report(self, results: ValidationResults) -> str:
        """Generate HTML validation report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Classification Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .notes {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #ccc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Classification Validation Report</h1>
                <p><strong>Validation ID:</strong> {results.validation_id}</p>
                <p><strong>Timestamp:</strong> {datetime.fromtimestamp(results.validation_timestamp)}</p>
                <p><strong>Status:</strong> 
                    <span class="{'pass' if results.passed_validation else 'fail'}">
                        {'PASSED' if results.passed_validation else 'FAILED'}
                    </span>
                </p>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="metric">
                    <strong>Overall Accuracy:</strong> {results.overall_accuracy:.4f}
                </div>
        """
        
        # Add class accuracies
        if results.class_accuracies:
            html += "<h3>Class-wise Accuracies</h3><ul>"
            for class_name, accuracy in results.class_accuracies.items():
                html += f"<li><strong>{class_name}:</strong> {accuracy:.4f}</li>"
            html += "</ul>"
        
        # Add cross-validation results
        if results.cv_mean is not None:
            html += f"""
            <div class="section">
                <h2>Cross-Validation Results</h2>
                <div class="metric"><strong>Mean CV Score:</strong> {results.cv_mean:.4f}</div>
                <div class="metric"><strong>CV Standard Deviation:</strong> {results.cv_std:.4f}</div>
                <div class="metric"><strong>Stability Score:</strong> {results.stability_score:.4f}</div>
            </div>
            """
        
        # Add confidence analysis
        if results.confidence_analysis:
            conf = results.confidence_analysis
            html += f"""
            <div class="section">
                <h2>Prediction Confidence Analysis</h2>
                <div class="metric"><strong>Mean Confidence:</strong> {conf['mean_confidence']:.4f}</div>
                <div class="metric"><strong>Low Confidence Ratio:</strong> {conf['low_confidence_ratio']:.1%}</div>
                <div class="metric"><strong>High Confidence Ratio:</strong> {conf['high_confidence_ratio']:.1%}</div>
            </div>
            """
        
        # Add error analysis
        if results.error_analysis:
            error = results.error_analysis
            html += f"""
            <div class="section">
                <h2>Error Analysis</h2>
                <div class="metric"><strong>Total Errors:</strong> {error['total_errors']}</div>
                <div class="metric"><strong>Error Rate:</strong> {error['error_rate']:.1%}</div>
            </div>
            """
        
        # Add validation notes
        if results.validation_notes:
            html += '<div class="section"><h2>Validation Notes</h2><div class="notes">'
            for note in results.validation_notes:
                html += f"<p>{note}</p>"
            html += "</div></div>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_visualizations(self, results: ValidationResults, output_dir: Path):
        """Generate validation visualizations"""
        
        # Confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(results.confusion_matrix, annot=True, fmt='d', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Class accuracies bar plot
        if results.class_accuracies:
            plt.figure(figsize=(12, 6))
            classes = list(results.class_accuracies.keys())
            accuracies = list(results.class_accuracies.values())
            
            bars = plt.bar(classes, accuracies)
            plt.ylim(0, 1)
            plt.ylabel('Accuracy')
            plt.title('Class-wise Accuracy')
            plt.xticks(rotation=45, ha='right')
            
            # Color bars based on performance
            for bar, acc in zip(bars, accuracies):
                if acc >= 0.8:
                    bar.set_color('green')
                elif acc >= 0.6:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'class_accuracies.png', dpi=300, bbox_inches='tight')
            plt.close()


class SegmentationValidator(BaseValidator):
    """Validator for segmentation models"""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.lesion_types = ['Hard Exudates', 'Soft Exudates', 'Hemorrhages', 'Microaneurysms']
    
    def validate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                device: torch.device) -> ValidationResults:
        """Perform comprehensive segmentation validation"""
        
        validation_id = f"segmentation_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting segmentation validation: {validation_id}")
        
        model.eval()
        all_dice_scores = {lesion: [] for lesion in self.lesion_types}
        all_iou_scores = {lesion: [] for lesion in self.lesion_types}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)  # Assuming multi-channel masks
                
                # Forward pass
                if hasattr(model, 'segment'):
                    outputs = model.segment(images)
                else:
                    outputs = model(images)
                
                # Calculate metrics for each lesion type
                for i, lesion in enumerate(self.lesion_types):
                    pred_mask = torch.sigmoid(outputs[:, i]) > 0.5
                    true_mask = masks[:, i] > 0.5
                    
                    # Calculate Dice score
                    dice = calculate_dice_score(pred_mask, true_mask)
                    all_dice_scores[lesion].append(dice.item())
                    
                    # Calculate IoU score
                    iou = calculate_iou_score(pred_mask, true_mask)
                    all_iou_scores[lesion].append(iou.item())
                
                if batch_idx % 50 == 0:
                    self.logger.info(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Calculate average scores
        dice_scores = {lesion: float(np.mean(scores)) for lesion, scores in all_dice_scores.items()}
        iou_scores = {lesion: float(np.mean(scores)) for lesion, scores in all_iou_scores.items()}
        
        # Overall metrics
        overall_dice = float(np.mean(list(dice_scores.values())))
        
        # Create validation results
        results = ValidationResults(
            validation_id=validation_id,
            model_path="",  # To be filled by caller
            dataset_name="segmentation_dataset",
            validation_timestamp=datetime.now().timestamp(),
            overall_accuracy=overall_dice,  # Use Dice as primary metric
            class_accuracies=dice_scores,
            confusion_matrix=[],  # Not applicable for segmentation
            classification_report={},  # Not applicable for segmentation
            dice_scores=dice_scores,
            iou_scores=iou_scores,
            validation_notes=[]
        )
        
        # Check validation criteria
        results.passed_validation = self._check_segmentation_criteria(results)
        
        return results
    
    def _check_segmentation_criteria(self, results: ValidationResults) -> bool:
        """Check if segmentation model passes validation criteria"""
        
        criteria_passed = []
        notes = []
        
        # Check overall Dice score
        if results.overall_accuracy >= self.config.min_dice_score:
            criteria_passed.append(True)
            notes.append(f"✓ Overall Dice score ({results.overall_accuracy:.3f}) meets minimum requirement")
        else:
            criteria_passed.append(False)
            notes.append(f"✗ Overall Dice score ({results.overall_accuracy:.3f}) below minimum ({self.config.min_dice_score})")
        
        # Check individual lesion performance
        if results.dice_scores:
            poor_performing_lesions = []
            for lesion, score in results.dice_scores.items():
                if score < self.config.min_dice_score * 0.8:  # 80% of minimum threshold
                    poor_performing_lesions.append(f"{lesion} ({score:.3f})")
            
            if poor_performing_lesions:
                criteria_passed.append(False)
                notes.append(f"✗ Poor performance on lesion types: {', '.join(poor_performing_lesions)}")
            else:
                criteria_passed.append(True)
                notes.append("✓ All lesion types meet performance requirements")
        
        results.validation_notes = notes
        return all(criteria_passed)
    
    def generate_report(self, results: ValidationResults, output_dir: str) -> str:
        """Generate segmentation validation report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"segmentation_validation_report_{results.validation_id}.html"
        
        # Generate HTML report
        html_content = self._generate_segmentation_html_report(results)
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        # Save detailed results as JSON
        json_file = output_path / f"validation_results_{results.validation_id}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)
        
        # Generate visualizations
        if self.config.save_visualizations:
            self._generate_segmentation_visualizations(results, output_path)
        
        self.logger.info(f"Segmentation validation report saved to: {report_file}")
        return str(report_file)
    
    def _generate_segmentation_html_report(self, results: ValidationResults) -> str:
        """Generate HTML segmentation validation report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Segmentation Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .notes {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #ccc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Segmentation Validation Report</h1>
                <p><strong>Validation ID:</strong> {results.validation_id}</p>
                <p><strong>Timestamp:</strong> {datetime.fromtimestamp(results.validation_timestamp)}</p>
                <p><strong>Status:</strong> 
                    <span class="{'pass' if results.passed_validation else 'fail'}">
                        {'PASSED' if results.passed_validation else 'FAILED'}
                    </span>
                </p>
            </div>
            
            <div class="section">
                <h2>Segmentation Metrics</h2>
                <div class="metric">
                    <strong>Overall Dice Score:</strong> {results.overall_accuracy:.4f}
                </div>
        """
        
        # Add Dice scores by lesion type
        if results.dice_scores:
            html += "<h3>Dice Scores by Lesion Type</h3><ul>"
            for lesion, score in results.dice_scores.items():
                html += f"<li><strong>{lesion}:</strong> {score:.4f}</li>"
            html += "</ul>"
        
        # Add IoU scores by lesion type
        if results.iou_scores:
            html += "<h3>IoU Scores by Lesion Type</h3><ul>"
            for lesion, score in results.iou_scores.items():
                html += f"<li><strong>{lesion}:</strong> {score:.4f}</li>"
            html += "</ul>"
        
        # Add validation notes
        if results.validation_notes:
            html += '<div class="section"><h2>Validation Notes</h2><div class="notes">'
            for note in results.validation_notes:
                html += f"<p>{note}</p>"
            html += "</div></div>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_segmentation_visualizations(self, results: ValidationResults, output_dir: Path):
        """Generate segmentation validation visualizations"""
        
        # Dice scores bar plot
        if results.dice_scores:
            plt.figure(figsize=(12, 6))
            lesions = list(results.dice_scores.keys())
            scores = list(results.dice_scores.values())
            
            bars = plt.bar(lesions, scores)
            plt.ylim(0, 1)
            plt.ylabel('Dice Score')
            plt.title('Dice Scores by Lesion Type')
            plt.xticks(rotation=45, ha='right')
            
            # Color bars based on performance
            for bar, score in zip(bars, scores):
                if score >= 0.7:
                    bar.set_color('green')
                elif score >= 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'dice_scores.png', dpi=300, bbox_inches='tight')
            plt.close()


class MultiTaskValidator(BaseValidator):
    """Validator for multi-task models (classification + segmentation)"""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.classification_validator = ClassificationValidator(config)
        self.segmentation_validator = SegmentationValidator(config)
    
    def validate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                device: torch.device) -> ValidationResults:
        """Perform comprehensive multi-task validation"""
        
        validation_id = f"multitask_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting multi-task validation: {validation_id}")
        
        # Validate classification component
        cls_results = self.classification_validator.validate(model, data_loader, device)
        
        # Validate segmentation component
        seg_results = self.segmentation_validator.validate(model, data_loader, device)
        
        # Combine results
        combined_results = ValidationResults(
            validation_id=validation_id,
            model_path="",
            dataset_name="multitask_dataset",
            validation_timestamp=datetime.now().timestamp(),
            overall_accuracy=cls_results.overall_accuracy,
            class_accuracies=cls_results.class_accuracies,
            confusion_matrix=cls_results.confusion_matrix,
            classification_report=cls_results.classification_report,
            dice_scores=seg_results.dice_scores,
            iou_scores=seg_results.iou_scores,
            cv_scores=cls_results.cv_scores,
            cv_mean=cls_results.cv_mean,
            cv_std=cls_results.cv_std,
            stability_score=cls_results.stability_score,
            confidence_analysis=cls_results.confidence_analysis,
            error_analysis=cls_results.error_analysis,
            validation_notes=[]
        )
        
        # Check combined validation criteria
        combined_results.passed_validation = self._check_multitask_criteria(
            combined_results, cls_results, seg_results
        )
        
        return combined_results
    
    def _check_multitask_criteria(self, combined_results: ValidationResults,
                                 cls_results: ValidationResults,
                                 seg_results: ValidationResults) -> bool:
        """Check if multi-task model passes validation criteria"""
        
        notes = []
        
        # Check if both tasks pass individual criteria
        cls_passed = cls_results.passed_validation
        seg_passed = seg_results.passed_validation
        
        if cls_passed:
            notes.append("✓ Classification task passes validation criteria")
        else:
            notes.append("✗ Classification task fails validation criteria")
            notes.extend(cls_results.validation_notes)
        
        if seg_passed:
            notes.append("✓ Segmentation task passes validation criteria")
        else:
            notes.append("✗ Segmentation task fails validation criteria")
            notes.extend(seg_results.validation_notes)
        
        # Additional multi-task specific checks
        # Balance between tasks (neither should be significantly worse)
        cls_score = cls_results.overall_accuracy
        seg_score = seg_results.overall_accuracy
        
        if abs(cls_score - seg_score) > 0.3:
            notes.append(f"⚠ Large performance gap between tasks (Cls: {cls_score:.3f}, Seg: {seg_score:.3f})")
        
        combined_results.validation_notes = notes
        return cls_passed and seg_passed
    
    def generate_report(self, results: ValidationResults, output_dir: str) -> str:
        """Generate multi-task validation report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"multitask_validation_report_{results.validation_id}.html"
        
        # Generate combined HTML report
        html_content = self._generate_multitask_html_report(results)
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        # Save detailed results as JSON
        json_file = output_path / f"validation_results_{results.validation_id}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)
        
        # Generate visualizations
        if self.config.save_visualizations:
            self.classification_validator._generate_visualizations(results, output_path)
            self.segmentation_validator._generate_segmentation_visualizations(results, output_path)
        
        self.logger.info(f"Multi-task validation report saved to: {report_file}")
        return str(report_file)
    
    def _generate_multitask_html_report(self, results: ValidationResults) -> str:
        """Generate HTML multi-task validation report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Task Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .notes {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #ccc; }}
                .task-section {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Multi-Task Validation Report</h1>
                <p><strong>Validation ID:</strong> {results.validation_id}</p>
                <p><strong>Timestamp:</strong> {datetime.fromtimestamp(results.validation_timestamp)}</p>
                <p><strong>Status:</strong> 
                    <span class="{'pass' if results.passed_validation else 'fail'}">
                        {'PASSED' if results.passed_validation else 'FAILED'}
                    </span>
                </p>
            </div>
            
            <div class="task-section">
                <h2>Classification Task Performance</h2>
                <div class="metric">
                    <strong>Overall Accuracy:</strong> {results.overall_accuracy:.4f}
                </div>
        """
        
        # Add class accuracies
        if results.class_accuracies:
            html += "<h3>Class-wise Accuracies</h3><ul>"
            for class_name, accuracy in results.class_accuracies.items():
                html += f"<li><strong>{class_name}:</strong> {accuracy:.4f}</li>"
            html += "</ul>"
        
        html += '</div><div class="task-section"><h2>Segmentation Task Performance</h2>'
        
        # Add segmentation metrics
        if results.dice_scores:
            html += f'<div class="metric"><strong>Overall Dice Score:</strong> {np.mean(list(results.dice_scores.values())):.4f}</div>'
            html += "<h3>Dice Scores by Lesion Type</h3><ul>"
            for lesion, score in results.dice_scores.items():
                html += f"<li><strong>{lesion}:</strong> {score:.4f}</li>"
            html += "</ul>"
        
        html += "</div>"
        
        # Add validation notes
        if results.validation_notes:
            html += '<div class="section"><h2>Validation Summary</h2><div class="notes">'
            for note in results.validation_notes:
                if "✓" in note:
                    html += f'<p class="pass">{note}</p>'
                elif "✗" in note:
                    html += f'<p class="fail">{note}</p>'
                elif "⚠" in note:
                    html += f'<p class="warning">{note}</p>'
                else:
                    html += f"<p>{note}</p>"
            html += "</div></div>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html


class QualityControlSystem:
    """Comprehensive quality control system for model validation"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger('QualityControlSystem')
        
        # Initialize validators
        self.classification_validator = ClassificationValidator(config)
        self.segmentation_validator = SegmentationValidator(config)
        self.multitask_validator = MultiTaskValidator(config)
    
    def run_full_validation(self, model_path: str, test_data_path: str,
                           model_type: str = 'multitask', device: str = 'cuda') -> Dict[str, Any]:
        """Run comprehensive validation pipeline"""
        
        self.logger.info(f"Starting full validation for model: {model_path}")
        
        # Load model
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model = self._load_model(model_path, device)
        
        # Load test data
        test_loader = self._load_test_data(test_data_path, model_type)
        
        # Select appropriate validator
        if model_type == 'classification':
            validator = self.classification_validator
        elif model_type == 'segmentation':
            validator = self.segmentation_validator
        else:
            validator = self.multitask_validator
        
        # Run validation
        results = validator.validate(model, test_loader, device)
        results.model_path = model_path
        
        # Generate report
        output_dir = Path(model_path).parent / "validation_results"
        report_path = validator.generate_report(results, str(output_dir))
        
        # Create summary
        validation_summary = {
            'validation_id': results.validation_id,
            'model_path': model_path,
            'model_type': model_type,
            'passed_validation': results.passed_validation,
            'overall_score': results.overall_accuracy,
            'report_path': report_path,
            'timestamp': results.validation_timestamp,
            'summary_metrics': self._extract_summary_metrics(results)
        }
        
        self.logger.info(f"Validation completed. Status: {'PASSED' if results.passed_validation else 'FAILED'}")
        return validation_summary
    
    def _load_model(self, model_path: str, device: torch.device) -> nn.Module:
        """Load model from checkpoint"""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Initialize model (this would need to be adapted based on your model structure)
            model = MultiTaskRetinaModel()  # Adjust parameters as needed
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _load_test_data(self, test_data_path: str, model_type: str) -> torch.utils.data.DataLoader:
        """Load test data"""
        try:
            # This would need to be adapted based on your data structure
            # For now, creating a simple loader
            
            if model_type == 'classification':
                dataset = APTOSDataset(test_data_path, split='test')
            elif model_type == 'segmentation':
                dataset = IDRiDDataset(test_data_path, split='test', task='segmentation')
            else:
                # Multi-task dataset
                dataset = IDRiDDataset(test_data_path, split='test', task='both')
            
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=16, shuffle=False, num_workers=4
            )
            
            return loader
            
        except Exception as e:
            self.logger.error(f"Failed to load test data from {test_data_path}: {e}")
            raise
    
    def _extract_summary_metrics(self, results: ValidationResults) -> Dict[str, float]:
        """Extract key metrics for summary"""
        summary = {
            'overall_accuracy': results.overall_accuracy
        }
        
        if results.cv_mean is not None:
            summary['cv_mean'] = results.cv_mean
            summary['cv_std'] = results.cv_std
            summary['stability_score'] = results.stability_score or 0.0
        
        if results.dice_scores:
            summary['mean_dice_score'] = float(np.mean(list(results.dice_scores.values())))
        
        if results.confidence_analysis:
            summary['mean_confidence'] = results.confidence_analysis['mean_confidence']
        
        return summary


def main():
    """Main function for validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 5: Model Validation and Quality Control")
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--model-type', type=str, 
                       choices=['classification', 'segmentation', 'multitask'],
                       default='multitask', help='Type of model to validate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for validation')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--use-tta', action='store_true',
                       help='Use test-time augmentation')
    
    args = parser.parse_args()
    
    # Create validation configuration
    config = ValidationConfig(
        cv_folds=args.cv_folds,
        use_tta=args.use_tta,
        save_visualizations=True,
        generate_detailed_report=True
    )
    
    # Run validation
    qc_system = QualityControlSystem(config)
    
    try:
        results = qc_system.run_full_validation(
            model_path=args.model_path,
            test_data_path=args.test_data,
            model_type=args.model_type,
            device=args.device
        )
        
        print("Validation completed successfully!")
        print(f"Status: {'PASSED' if results['passed_validation'] else 'FAILED'}")
        print(f"Overall Score: {results['overall_score']:.4f}")
        print(f"Report saved to: {results['report_path']}")
        
    except Exception as e:
        print(f"Validation failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
