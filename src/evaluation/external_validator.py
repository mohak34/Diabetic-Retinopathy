"""
Phase 6: External Validation & Generalization Testing
Handles evaluation on external datasets (like Messidor-2) and generalization analysis.
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
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics_calculator import MetricsCalculator, EvaluationResults
from .comprehensive_evaluator import ComprehensiveEvaluator

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ExternalValidator:
    """
    External validation and generalization testing for diabetic retinopathy models.
    Handles Step 6.2: External Validation & Generalization Testing
    """
    
    def __init__(self,
                 comprehensive_evaluator: ComprehensiveEvaluator,
                 output_dir: str = 'results/phase6_external_validation'):
        """
        Initialize external validator
        
        Args:
            comprehensive_evaluator: Main evaluator instance
            output_dir: Directory to save external validation results
        """
        self.evaluator = comprehensive_evaluator
        self.device = comprehensive_evaluator.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExternalValidator initialized")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def run_external_validation(self,
                              checkpoint_path: str,
                              internal_results: Dict[str, EvaluationResults],
                              external_loaders: Dict[str, DataLoader],
                              model_config: Optional[Dict] = None) -> Dict[str, any]:
        """
        Run complete external validation analysis
        
        Args:
            checkpoint_path: Path to model checkpoint
            internal_results: Results from internal validation datasets
            external_loaders: Dictionary of external dataset name -> DataLoader
            model_config: Model configuration
            
        Returns:
            Dictionary containing external validation results and analysis
        """
        logger.info("Starting external validation analysis")
        
        # Evaluate on external datasets
        external_results = self.evaluator.evaluate_model_checkpoint(
            checkpoint_path, external_loaders, model_config
        )
        
        # Calculate generalization gaps
        generalization_analysis = self._calculate_generalization_gaps(
            internal_results, external_results
        )
        
        # Perform failure analysis
        failure_analysis = self._perform_failure_analysis(
            checkpoint_path, external_loaders, model_config
        )
        
        # Generate external validation report
        report_path = self._generate_external_validation_report(
            internal_results, external_results, generalization_analysis, failure_analysis
        )
        
        # Create visualizations
        self._create_generalization_visualizations(
            internal_results, external_results, generalization_analysis
        )
        
        return {
            'external_results': external_results,
            'generalization_analysis': generalization_analysis,
            'failure_analysis': failure_analysis,
            'report_path': report_path
        }
    
    def _calculate_generalization_gaps(self,
                                     internal_results: Dict[str, EvaluationResults],
                                     external_results: Dict[str, EvaluationResults]) -> Dict[str, any]:
        """Calculate generalization gaps between internal and external validation"""
        
        logger.info("Calculating generalization gaps")
        
        # Calculate average internal performance
        internal_metrics = {
            'accuracy': np.mean([r.classification.accuracy for r in internal_results.values()]),
            'kappa': np.mean([r.classification.kappa for r in internal_results.values()]),
            'auc_roc': np.mean([r.classification.auc_roc for r in internal_results.values()]),
            'dice': np.mean([r.segmentation.dice_coefficient for r in internal_results.values()]),
            'iou': np.mean([r.segmentation.iou_score for r in internal_results.values()]),
            'combined_score': np.mean([r.combined_score for r in internal_results.values()])
        }
        
        # Calculate external performance
        external_metrics = {}
        generalization_gaps = {}
        
        for ext_dataset, ext_result in external_results.items():
            external_metrics[ext_dataset] = {
                'accuracy': ext_result.classification.accuracy,
                'kappa': ext_result.classification.kappa,
                'auc_roc': ext_result.classification.auc_roc,
                'dice': ext_result.segmentation.dice_coefficient,
                'iou': ext_result.segmentation.iou_score,
                'combined_score': ext_result.combined_score
            }
            
            # Calculate gaps
            gaps = {}
            for metric_name in internal_metrics.keys():
                gap = internal_metrics[metric_name] - external_metrics[ext_dataset][metric_name]
                gaps[metric_name] = {
                    'absolute_gap': float(gap),
                    'relative_gap': float(gap / internal_metrics[metric_name]) if internal_metrics[metric_name] > 0 else 0.0,
                    'internal_score': float(internal_metrics[metric_name]),
                    'external_score': float(external_metrics[ext_dataset][metric_name])
                }
            
            generalization_gaps[ext_dataset] = gaps
        
        # Calculate overall generalization metrics
        overall_analysis = self._calculate_overall_generalization_metrics(
            internal_metrics, external_metrics, generalization_gaps
        )
        
        return {
            'internal_avg_metrics': internal_metrics,
            'external_metrics': external_metrics,
            'generalization_gaps': generalization_gaps,
            'overall_analysis': overall_analysis
        }
    
    def _calculate_overall_generalization_metrics(self,
                                                internal_metrics: Dict,
                                                external_metrics: Dict,
                                                gaps: Dict) -> Dict[str, any]:
        """Calculate overall generalization performance metrics"""
        
        # Average generalization gap across all external datasets
        all_gaps = {}
        for metric_name in internal_metrics.keys():
            absolute_gaps = [gaps[dataset][metric_name]['absolute_gap'] 
                           for dataset in gaps.keys()]
            relative_gaps = [gaps[dataset][metric_name]['relative_gap'] 
                           for dataset in gaps.keys()]
            
            all_gaps[metric_name] = {
                'mean_absolute_gap': float(np.mean(absolute_gaps)),
                'std_absolute_gap': float(np.std(absolute_gaps)),
                'mean_relative_gap': float(np.mean(relative_gaps)),
                'std_relative_gap': float(np.std(relative_gaps)),
                'max_absolute_gap': float(np.max(absolute_gaps)),
                'min_absolute_gap': float(np.min(absolute_gaps))
            }
        
        # Generalization score (inverse of mean relative gap)
        mean_relative_gaps = [all_gaps[metric]['mean_relative_gap'] 
                            for metric in ['accuracy', 'kappa', 'dice']]
        generalization_score = 1.0 - np.mean(np.abs(mean_relative_gaps))
        
        # Risk assessment
        risk_level = self._assess_generalization_risk(all_gaps)
        
        return {
            'metric_gaps': all_gaps,
            'generalization_score': float(generalization_score),
            'risk_assessment': risk_level,
            'num_external_datasets': len(external_metrics)
        }
    
    def _assess_generalization_risk(self, gaps: Dict) -> Dict[str, any]:
        """Assess generalization risk based on performance gaps"""
        
        # Define risk thresholds
        high_risk_threshold = 0.15  # 15% relative gap
        moderate_risk_threshold = 0.08  # 8% relative gap
        
        risk_indicators = {}
        
        for metric_name, gap_stats in gaps.items():
            mean_rel_gap = abs(gap_stats['mean_relative_gap'])
            max_abs_gap = abs(gap_stats['max_absolute_gap'])
            
            if mean_rel_gap > high_risk_threshold:
                risk_level = 'HIGH'
            elif mean_rel_gap > moderate_risk_threshold:
                risk_level = 'MODERATE'
            else:
                risk_level = 'LOW'
            
            risk_indicators[metric_name] = {
                'risk_level': risk_level,
                'mean_relative_gap': mean_rel_gap,
                'max_absolute_gap': max_abs_gap,
                'concerning': mean_rel_gap > high_risk_threshold
            }
        
        # Overall risk assessment
        high_risk_metrics = sum(1 for r in risk_indicators.values() if r['risk_level'] == 'HIGH')
        moderate_risk_metrics = sum(1 for r in risk_indicators.values() if r['risk_level'] == 'MODERATE')
        
        if high_risk_metrics >= 2:
            overall_risk = 'HIGH'
        elif high_risk_metrics >= 1 or moderate_risk_metrics >= 3:
            overall_risk = 'MODERATE'
        else:
            overall_risk = 'LOW'
        
        return {
            'overall_risk': overall_risk,
            'metric_risks': risk_indicators,
            'high_risk_count': high_risk_metrics,
            'moderate_risk_count': moderate_risk_metrics,
            'recommendations': self._generate_risk_recommendations(overall_risk, risk_indicators)
        }
    
    def _generate_risk_recommendations(self, 
                                     overall_risk: str, 
                                     metric_risks: Dict) -> List[str]:
        """Generate recommendations based on risk assessment"""
        
        recommendations = []
        
        if overall_risk == 'HIGH':
            recommendations.append("🚨 HIGH RISK: Significant generalization gaps detected")
            recommendations.append("• Consider collecting more diverse training data")
            recommendations.append("• Implement domain adaptation techniques")
            recommendations.append("• Increase regularization during training")
            recommendations.append("• Evaluate on additional external datasets")
        
        elif overall_risk == 'MODERATE':
            recommendations.append("⚠️  MODERATE RISK: Some generalization concerns")
            recommendations.append("• Monitor performance on additional datasets")
            recommendations.append("• Consider data augmentation strategies")
            recommendations.append("• Validate with clinical experts")
        
        else:
            recommendations.append("✅ LOW RISK: Good generalization performance")
            recommendations.append("• Continue monitoring with new datasets")
            recommendations.append("• Model appears suitable for deployment")
        
        # Metric-specific recommendations
        concerning_metrics = [name for name, risk in metric_risks.items() 
                            if risk['concerning']]
        
        if concerning_metrics:
            recommendations.append(f"• Pay special attention to: {', '.join(concerning_metrics)}")
        
        return recommendations
    
    def _perform_failure_analysis(self,
                                checkpoint_path: str,
                                external_loaders: Dict[str, DataLoader],
                                model_config: Optional[Dict] = None) -> Dict[str, any]:
        """Perform detailed failure analysis on external datasets"""
        
        logger.info("Performing failure analysis")
        
        # Load model
        model = self.evaluator._load_model_from_checkpoint(checkpoint_path, model_config)
        model.eval()
        
        failure_analysis = {}
        
        for dataset_name, data_loader in external_loaders.items():
            logger.info(f"Analyzing failures on {dataset_name}")
            
            # Collect predictions and analyze failures
            failures = self._analyze_dataset_failures(model, data_loader, dataset_name)
            failure_analysis[dataset_name] = failures
        
        return failure_analysis
    
    def _analyze_dataset_failures(self,
                                model: nn.Module,
                                data_loader: DataLoader,
                                dataset_name: str) -> Dict[str, any]:
        """Analyze failures for a specific dataset"""
        
        misclassified_samples = []
        poor_segmentation_samples = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['image'].to(self.device)
                
                # Get targets if available
                cls_targets = batch.get('grade', batch.get('label'))
                seg_targets = batch.get('mask', batch.get('segmentation'))
                
                if cls_targets is not None:
                    cls_targets = cls_targets.to(self.device)
                if seg_targets is not None:
                    seg_targets = seg_targets.to(self.device)
                
                # Forward pass
                cls_outputs, seg_outputs = model(images)
                
                # Analyze classification failures
                if cls_targets is not None:
                    cls_probs = torch.softmax(cls_outputs, dim=1)
                    cls_preds = torch.argmax(cls_probs, dim=1)
                    confidences = torch.max(cls_probs, dim=1)[0]
                    
                    # Find misclassified samples
                    misclassified = cls_preds != cls_targets
                    
                    for i in range(images.size(0)):
                        if misclassified[i]:
                            misclassified_samples.append({
                                'sample_id': sample_count + i,
                                'true_grade': int(cls_targets[i].item()),
                                'predicted_grade': int(cls_preds[i].item()),
                                'confidence': float(confidences[i].item()),
                                'batch_idx': batch_idx,
                                'sample_idx': i
                            })
                
                # Analyze segmentation failures
                if seg_targets is not None:
                    seg_probs = torch.sigmoid(seg_outputs)
                    seg_preds = (seg_probs > 0.5).float()
                    
                    # Calculate per-sample Dice scores
                    for i in range(images.size(0)):
                        sample_dice = self._calculate_sample_dice(
                            seg_targets[i].cpu().numpy(),
                            seg_preds[i].cpu().numpy()
                        )
                        
                        # If Dice is very low, consider it a failure
                        if sample_dice < 0.3:
                            poor_segmentation_samples.append({
                                'sample_id': sample_count + i,
                                'dice_score': float(sample_dice),
                                'batch_idx': batch_idx,
                                'sample_idx': i
                            })
                
                sample_count += images.size(0)
                
                # Limit analysis to avoid excessive computation
                if batch_idx >= 100:  # Analyze first 100 batches
                    break
        
        # Analyze failure patterns
        failure_patterns = self._analyze_failure_patterns(
            misclassified_samples, poor_segmentation_samples
        )
        
        return {
            'misclassified_samples': misclassified_samples[:50],  # Keep top 50
            'poor_segmentation_samples': poor_segmentation_samples[:50],
            'failure_patterns': failure_patterns,
            'total_samples_analyzed': sample_count
        }
    
    def _calculate_sample_dice(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Dice score for a single sample"""
        intersection = np.logical_and(y_true, y_pred).sum()
        total = y_true.sum() + y_pred.sum()
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * intersection / total
    
    def _analyze_failure_patterns(self,
                                misclassified: List[Dict],
                                poor_segmentation: List[Dict]) -> Dict[str, any]:
        """Analyze patterns in model failures"""
        
        patterns = {}
        
        # Classification failure patterns
        if misclassified:
            # Grade confusion patterns
            confusion_pairs = [(s['true_grade'], s['predicted_grade']) 
                             for s in misclassified]
            
            from collections import Counter
            common_confusions = Counter(confusion_pairs).most_common(5)
            
            # Confidence analysis of misclassified samples
            confidences = [s['confidence'] for s in misclassified]
            
            patterns['classification'] = {
                'total_misclassified': len(misclassified),
                'common_confusions': [
                    {'true_grade': pair[0], 'pred_grade': pair[1], 'count': count}
                    for (pair, count) in common_confusions
                ],
                'confidence_stats': {
                    'mean': float(np.mean(confidences)),
                    'std': float(np.std(confidences)),
                    'median': float(np.median(confidences)),
                    'high_confidence_errors': sum(1 for c in confidences if c > 0.8)
                }
            }
        
        # Segmentation failure patterns
        if poor_segmentation:
            dice_scores = [s['dice_score'] for s in poor_segmentation]
            
            patterns['segmentation'] = {
                'total_poor_segmentation': len(poor_segmentation),
                'dice_stats': {
                    'mean': float(np.mean(dice_scores)),
                    'std': float(np.std(dice_scores)),
                    'median': float(np.median(dice_scores)),
                    'min': float(np.min(dice_scores))
                }
            }
        
        return patterns
    
    def _create_generalization_visualizations(self,
                                            internal_results: Dict[str, EvaluationResults],
                                            external_results: Dict[str, EvaluationResults],
                                            generalization_analysis: Dict) -> None:
        """Create visualizations for generalization analysis"""
        
        logger.info("Creating generalization visualizations")
        
        # Create visualization directory
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Performance comparison plot
        self._plot_performance_comparison(
            internal_results, external_results, viz_dir
        )
        
        # 2. Generalization gap heatmap
        self._plot_generalization_gaps(
            generalization_analysis['generalization_gaps'], viz_dir
        )
        
        # 3. Risk assessment visualization
        self._plot_risk_assessment(
            generalization_analysis['overall_analysis']['risk_assessment'], viz_dir
        )
    
    def _plot_performance_comparison(self,
                                   internal_results: Dict[str, EvaluationResults],
                                   external_results: Dict[str, EvaluationResults],
                                   output_dir: Path) -> None:
        """Plot performance comparison between internal and external datasets"""
        
        # Prepare data
        metrics = ['accuracy', 'kappa', 'auc_roc', 'dice', 'iou']
        
        internal_scores = {
            'accuracy': [r.classification.accuracy for r in internal_results.values()],
            'kappa': [r.classification.kappa for r in internal_results.values()],
            'auc_roc': [r.classification.auc_roc for r in internal_results.values()],
            'dice': [r.segmentation.dice_coefficient for r in internal_results.values()],
            'iou': [r.segmentation.iou_score for r in internal_results.values()]
        }
        
        external_scores = {}
        for dataset_name, result in external_results.items():
            external_scores[dataset_name] = {
                'accuracy': result.classification.accuracy,
                'kappa': result.classification.kappa,
                'auc_roc': result.classification.auc_roc,
                'dice': result.segmentation.dice_coefficient,
                'iou': result.segmentation.iou_score
            }
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Internal performance (box plot)
            internal_data = internal_scores[metric]
            ax.boxplot([internal_data], positions=[1], labels=['Internal'], 
                      patch_artist=True, boxprops=dict(facecolor='lightblue'))
            
            # External performance (scatter plot)
            external_data = [external_scores[dataset][metric] 
                           for dataset in external_scores.keys()]
            external_labels = list(external_scores.keys())
            
            for j, (label, score) in enumerate(zip(external_labels, external_data)):
                ax.scatter(2 + j * 0.1, score, color='red', s=100, alpha=0.7, label=label)
            
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add legend for external datasets
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Remove extra subplot
        axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_generalization_gaps(self, gaps: Dict, output_dir: Path) -> None:
        """Plot generalization gaps as heatmap"""
        
        # Prepare data for heatmap
        datasets = list(gaps.keys())
        metrics = ['accuracy', 'kappa', 'auc_roc', 'dice', 'iou']
        
        gap_matrix = np.zeros((len(datasets), len(metrics)))
        
        for i, dataset in enumerate(datasets):
            for j, metric in enumerate(metrics):
                gap_matrix[i, j] = gaps[dataset][metric]['relative_gap']
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(gap_matrix, 
                   xticklabels=metrics,
                   yticklabels=datasets,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0,
                   cbar_kws={'label': 'Relative Performance Gap'})
        
        plt.title('Generalization Gaps by Dataset and Metric')
        plt.xlabel('Metrics')
        plt.ylabel('External Datasets')
        plt.tight_layout()
        plt.savefig(output_dir / 'generalization_gaps_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_assessment(self, risk_assessment: Dict, output_dir: Path) -> None:
        """Plot risk assessment visualization"""
        
        metric_risks = risk_assessment['metric_risks']
        
        # Prepare data
        metrics = list(metric_risks.keys())
        risk_levels = [metric_risks[m]['risk_level'] for m in metrics]
        relative_gaps = [metric_risks[m]['mean_relative_gap'] for m in metrics]
        
        # Create risk level plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk level bar chart
        colors = {'HIGH': 'red', 'MODERATE': 'orange', 'LOW': 'green'}
        bar_colors = [colors[risk] for risk in risk_levels]
        
        ax1.bar(metrics, relative_gaps, color=bar_colors, alpha=0.7)
        ax1.set_title('Risk Assessment by Metric')
        ax1.set_ylabel('Mean Relative Gap')
        ax1.set_xlabel('Metrics')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add risk level thresholds
        ax1.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='High Risk Threshold')
        ax1.axhline(y=0.08, color='orange', linestyle='--', alpha=0.5, label='Moderate Risk Threshold')
        ax1.legend()
        
        # Overall risk pie chart
        risk_counts = {
            'HIGH': sum(1 for r in risk_levels if r == 'HIGH'),
            'MODERATE': sum(1 for r in risk_levels if r == 'MODERATE'),
            'LOW': sum(1 for r in risk_levels if r == 'LOW')
        }
        
        risk_counts = {k: v for k, v in risk_counts.items() if v > 0}
        
        ax2.pie(risk_counts.values(), 
               labels=risk_counts.keys(),
               colors=[colors[k] for k in risk_counts.keys()],
               autopct='%1.0f%%',
               startangle=90)
        ax2.set_title('Overall Risk Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_assessment.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_external_validation_report(self,
                                           internal_results: Dict[str, EvaluationResults],
                                           external_results: Dict[str, EvaluationResults],
                                           generalization_analysis: Dict,
                                           failure_analysis: Dict) -> str:
        """Generate comprehensive external validation report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'external_validation_report_{timestamp}.html'
        
        html_content = self._generate_external_validation_html(
            internal_results, external_results, generalization_analysis, failure_analysis
        )
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"External validation report saved to: {report_path}")
        return str(report_path)
    
    def _generate_external_validation_html(self,
                                         internal_results: Dict,
                                         external_results: Dict,
                                         generalization_analysis: Dict,
                                         failure_analysis: Dict) -> str:
        """Generate HTML report for external validation"""
        
        overall_analysis = generalization_analysis['overall_analysis']
        risk_assessment = overall_analysis['risk_assessment']
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phase 6: External Validation & Generalization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .risk-high {{ color: red; font-weight: bold; }}
                .risk-moderate {{ color: orange; font-weight: bold; }}
                .risk-low {{ color: green; font-weight: bold; }}
                .metric {{ margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .recommendation {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Phase 6: External Validation & Generalization Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Generalization Score:</strong> {overall_analysis['generalization_score']:.4f}</p>
                <p><strong>Overall Risk Level:</strong> 
                    <span class="risk-{risk_assessment['overall_risk'].lower()}">{risk_assessment['overall_risk']}</span>
                </p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Dataset Type</th>
                        <th>Accuracy</th>
                        <th>Kappa</th>
                        <th>AUC-ROC</th>
                        <th>Dice</th>
                        <th>IoU</th>
                    </tr>
                    <tr>
                        <td>Internal (Average)</td>
                        <td>{generalization_analysis['internal_avg_metrics']['accuracy']:.4f}</td>
                        <td>{generalization_analysis['internal_avg_metrics']['kappa']:.4f}</td>
                        <td>{generalization_analysis['internal_avg_metrics']['auc_roc']:.4f}</td>
                        <td>{generalization_analysis['internal_avg_metrics']['dice']:.4f}</td>
                        <td>{generalization_analysis['internal_avg_metrics']['iou']:.4f}</td>
                    </tr>
        """
        
        # Add external dataset results
        for dataset_name, metrics in generalization_analysis['external_metrics'].items():
            html += f"""
                    <tr>
                        <td>{dataset_name}</td>
                        <td>{metrics['accuracy']:.4f}</td>
                        <td>{metrics['kappa']:.4f}</td>
                        <td>{metrics['auc_roc']:.4f}</td>
                        <td>{metrics['dice']:.4f}</td>
                        <td>{metrics['iou']:.4f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Generalization Gap Analysis</h2>
        """
        
        # Add generalization gaps
        for dataset_name, gaps in generalization_analysis['generalization_gaps'].items():
            html += f"""
                <h3>{dataset_name} Generalization Gaps</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Internal Score</th>
                        <th>External Score</th>
                        <th>Absolute Gap</th>
                        <th>Relative Gap</th>
                    </tr>
            """
            
            for metric_name, gap_data in gaps.items():
                html += f"""
                    <tr>
                        <td>{metric_name.upper()}</td>
                        <td>{gap_data['internal_score']:.4f}</td>
                        <td>{gap_data['external_score']:.4f}</td>
                        <td>{gap_data['absolute_gap']:.4f}</td>
                        <td>{gap_data['relative_gap']:.1%}</td>
                    </tr>
                """
            
            html += "</table>"
        
        html += """
            </div>
            
            <div class="section">
                <h2>Risk Assessment</h2>
        """
        
        # Add risk assessment details
        for metric_name, risk_data in risk_assessment['metric_risks'].items():
            risk_class = f"risk-{risk_data['risk_level'].lower()}"
            html += f"""
                <div class="metric">
                    <strong>{metric_name.upper()}:</strong> 
                    <span class="{risk_class}">{risk_data['risk_level']} RISK</span>
                    (Gap: {risk_data['mean_relative_gap']:.1%})
                </div>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
        """
        
        for recommendation in risk_assessment['recommendations']:
            html += f'<div class="recommendation">{recommendation}</div>'
        
        html += """
            </div>
            
            <div class="section">
                <h2>Failure Analysis Summary</h2>
        """
        
        # Add failure analysis summary
        for dataset_name, failure_data in failure_analysis.items():
            html += f"""
                <h3>{dataset_name} Failure Analysis</h3>
                <div class="metric">Total Samples Analyzed: {failure_data['total_samples_analyzed']}</div>
            """
            
            if 'classification' in failure_data['failure_patterns']:
                cls_patterns = failure_data['failure_patterns']['classification']
                html += f"""
                    <div class="metric">Misclassified Samples: {cls_patterns['total_misclassified']}</div>
                    <div class="metric">High Confidence Errors: {cls_patterns['confidence_stats']['high_confidence_errors']}</div>
                """
            
            if 'segmentation' in failure_data['failure_patterns']:
                seg_patterns = failure_data['failure_patterns']['segmentation']
                html += f"""
                    <div class="metric">Poor Segmentation Samples: {seg_patterns['total_poor_segmentation']}</div>
                    <div class="metric">Mean Dice Score (failures): {seg_patterns['dice_stats']['mean']:.4f}</div>
                """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
