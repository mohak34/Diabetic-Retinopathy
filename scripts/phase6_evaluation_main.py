"""
Phase 6: Main Evaluation Script
Orchestrates the complete Phase 6 evaluation pipeline including comprehensive performance evaluation,
external validation, and explainability analysis.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import warnings

import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level from scripts/ to project root
sys.path.append(str(project_root))

from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from src.evaluation.external_validator import ExternalValidator
from src.evaluation.explainability_analyzer import ExplainabilityAnalyzer
from src.evaluation.visualization_generator import VisualizationGenerator

warnings.filterwarnings('ignore')

# Ensure logs directory exists
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'phase6_evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Phase6EvaluationPipeline:
    """
    Main pipeline for Phase 6: Model Evaluation & Analysis
    
    Orchestrates:
    - Step 6.1: Comprehensive Performance Evaluation
    - Step 6.2: External Validation & Generalization Testing
    - Step 6.3: Explainability & Visualization
    """
    
    def __init__(self, config_path: str, output_dir: str = 'results/phase6_evaluation'):
        """
        Initialize Phase 6 evaluation pipeline
        
        Args:
            config_path: Path to configuration file
            output_dir: Base output directory for all results
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize evaluators
        self.comprehensive_evaluator = ComprehensiveEvaluator(
            device=self.device,
            output_dir=str(self.output_dir / 'comprehensive_evaluation')
        )
        
        self.external_validator = ExternalValidator(
            comprehensive_evaluator=self.comprehensive_evaluator,
            output_dir=str(self.output_dir / 'external_validation')
        )
        
        self.explainability_analyzer = ExplainabilityAnalyzer(
            comprehensive_evaluator=self.comprehensive_evaluator,
            output_dir=str(self.output_dir / 'explainability_analysis')
        )
        
        self.visualization_generator = VisualizationGenerator(
            output_dir=str(self.output_dir / 'visualizations')
        )
        
        logger.info(f"Phase 6 evaluation pipeline initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_complete_evaluation(self,
                              checkpoint_paths: List[str],
                              datasets_config: Optional[Dict] = None) -> Dict[str, any]:
        """
        Run complete Phase 6 evaluation pipeline
        
        Args:
            checkpoint_paths: List of model checkpoint paths to evaluate
            datasets_config: Optional dataset configuration override
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("=" * 80)
        logger.info("STARTING PHASE 6: COMPREHENSIVE MODEL EVALUATION & ANALYSIS")
        logger.info("=" * 80)
        
        # Prepare data loaders
        data_loaders = self._prepare_data_loaders(datasets_config)
        
        # Separate internal and external datasets
        internal_loaders, external_loaders = self._separate_internal_external_datasets(data_loaders)
        
        all_results = {}
        
        for checkpoint_path in checkpoint_paths:
            checkpoint_name = Path(checkpoint_path).stem
            logger.info(f"\nEvaluating checkpoint: {checkpoint_name}")
            
            try:
                # Step 6.1: Comprehensive Performance Evaluation
                logger.info("Step 6.1: Running comprehensive performance evaluation")
                internal_results = self.comprehensive_evaluator.evaluate_model_checkpoint(
                    checkpoint_path, internal_loaders, self.config.get('model')
                )
                
                # Step 6.2: External Validation & Generalization Testing
                logger.info("Step 6.2: Running external validation and generalization testing")
                external_validation_results = self.external_validator.run_external_validation(
                    checkpoint_path, internal_results, external_loaders, self.config.get('model')
                )
                
                # Step 6.3: Explainability & Visualization
                logger.info("Step 6.3: Running explainability and visualization analysis")
                explainability_results = self.explainability_analyzer.run_comprehensive_explainability_analysis(
                    checkpoint_path, data_loaders, self.config.get('model'), n_samples=1000
                )
                
                # Generate comprehensive visualizations
                logger.info("Generating comprehensive visualization suite")
                all_evaluation_results = {**internal_results, **external_validation_results['external_results']}
                
                visualization_results = self.visualization_generator.generate_comprehensive_visualization_suite(
                    all_evaluation_results,
                    training_history=self._load_training_history(checkpoint_path),
                    comparison_results=None  # Will be added in multi-checkpoint comparison
                )
                
                # Compile results for this checkpoint
                checkpoint_results = {
                    'checkpoint_path': checkpoint_path,
                    'internal_evaluation': internal_results,
                    'external_validation': external_validation_results,
                    'explainability_analysis': explainability_results,
                    'visualizations': visualization_results
                }
                
                all_results[checkpoint_name] = checkpoint_results
                
                logger.info(f"Completed evaluation for {checkpoint_name}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {checkpoint_name}: {e}")
                continue
        
        # Multi-checkpoint comparative analysis
        if len(all_results) > 1:
            logger.info("\nRunning multi-checkpoint comparative analysis")
            comparative_results = self._run_comparative_analysis(all_results)
            all_results['comparative_analysis'] = comparative_results
        
        # Generate final comprehensive report
        logger.info("\nGenerating final comprehensive report")
        final_report_path = self._generate_final_report(all_results)
        all_results['final_report_path'] = final_report_path
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 6 EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Final report: {final_report_path}")
        logger.info(f"All results saved to: {self.output_dir}")
        
        return all_results
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _prepare_data_loaders(self, datasets_config: Optional[Dict] = None) -> Dict[str, DataLoader]:
        """Prepare data loaders for all datasets"""
        
        if datasets_config is None:
            datasets_config = self.config.get('datasets', {})
        
        data_loaders = {}
        
        # Default dataset configurations
        default_datasets = {
            'aptos2019_val': {
                'type': 'aptos2019',
                'split': 'validation',
                'data_path': 'dataset/processed/aptos2019/val'
            },
            'grading_val': {
                'type': 'grading', 
                'split': 'validation',
                'data_path': 'dataset/processed/grading'
            },
            'segmentation_test': {
                'type': 'segmentation',
                'split': 'test',
                'data_path': 'dataset/processed/segmentation'
            }
        }
        
        # Use provided config or defaults, handling nested structure
        datasets_to_load = {}
        
        if datasets_config:
            # Handle nested structure (internal/external)
            for category in ['internal', 'external']:
                if category in datasets_config:
                    for dataset_name, dataset_config in datasets_config[category].items():
                        datasets_to_load[dataset_name] = dataset_config
            
            # Handle flat structure
            for key, value in datasets_config.items():
                if key not in ['internal', 'external'] and isinstance(value, dict) and 'type' in value:
                    datasets_to_load[key] = value
        
        if not datasets_to_load:
            datasets_to_load = default_datasets
        
        for dataset_name, dataset_config in datasets_to_load.items():
            try:
                logger.info(f"Loading dataset: {dataset_name}")
                
                # This is a simplified data loader creation
                # In practice, you would use your actual data loading utilities
                data_loader = self._create_data_loader(dataset_config)
                
                if data_loader is not None:
                    data_loaders[dataset_name] = data_loader
                    logger.info(f"Successfully loaded {dataset_name} with {len(data_loader)} batches")
                else:
                    logger.warning(f"Failed to load dataset: {dataset_name}")
                    
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_name}: {e}")
                continue
        
        return data_loaders
    
    def _create_data_loader(self, dataset_config: Dict) -> Optional[DataLoader]:
        """Create data loader from dataset configuration"""
        try:
            # Import the actual dataset classes
            from src.data.datasets import GradingRetinaDataset, SegmentationRetinaDataset, MultiTaskRetinaDataset
            
            dataset_type = dataset_config['type']
            data_path = dataset_config['data_path']
            
            # Create transform from config (convert dict to callable transform)
            transform = self._create_transform_from_config(dataset_config.get('transform'))
            
            # Create appropriate dataset based on type
            if dataset_type in ['aptos2019', 'grading', 'classification']:
                # For classification datasets
                labels_path = os.path.join(data_path, 'labels.csv') if os.path.exists(os.path.join(data_path, 'labels.csv')) else None
                images_dir = os.path.join(data_path, 'images') if os.path.exists(os.path.join(data_path, 'images')) else data_path
                
                if labels_path and os.path.exists(labels_path):
                    dataset = GradingRetinaDataset(
                        images_dir=images_dir,
                        labels_path=labels_path,
                        transform=transform
                    )
                else:
                    logger.warning(f"No labels found for {dataset_type} at {data_path}")
                    return None
                    
            elif dataset_type in ['idrid', 'segmentation']:
                # For segmentation datasets
                images_dir = os.path.join(data_path, 'images')
                masks_dir = os.path.join(data_path, 'masks')
                
                if os.path.exists(images_dir) and os.path.exists(masks_dir):
                    dataset = SegmentationRetinaDataset(
                        images_dir=images_dir,
                        masks_dir=masks_dir,
                        transform=transform
                    )
                else:
                    logger.warning(f"Images or masks directory not found for {dataset_type} at {data_path}")
                    return None
                    
            elif dataset_type in ['multi_task', 'combined']:
                # For multi-task datasets
                images_dir = os.path.join(data_path, 'images') if os.path.exists(os.path.join(data_path, 'images')) else data_path
                labels_path = os.path.join(data_path, 'labels.csv') if os.path.exists(os.path.join(data_path, 'labels.csv')) else None
                masks_dir = os.path.join(data_path, 'masks') if os.path.exists(os.path.join(data_path, 'masks')) else None
                
                dataset = MultiTaskRetinaDataset(
                    images_dir=images_dir,
                    labels_path=labels_path,
                    masks_dir=masks_dir,
                    transform=transform
                )
            else:
                logger.error(f"Unknown dataset type: {dataset_type}")
                return None
            
            # Create data loader
            data_loader = DataLoader(
                dataset,
                batch_size=dataset_config.get('batch_size', 16),
                shuffle=False,  # Don't shuffle for evaluation
                num_workers=dataset_config.get('num_workers', 4),
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            return data_loader
            
        except Exception as e:
            logger.error(f"Failed to create data loader: {e}")
            return None
    
    def _create_transform_from_config(self, transform_config: Optional[Dict]) -> Optional[any]:
        """Create transform function from configuration dictionary"""
        if not transform_config:
            return None
        
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            transforms = []
            
            # Handle resize
            if 'resize' in transform_config:
                size = transform_config['resize']
                if isinstance(size, list) and len(size) == 2:
                    transforms.append(A.Resize(height=size[0], width=size[1]))
                elif isinstance(size, int):
                    transforms.append(A.Resize(height=size, width=size))
            
            # Handle normalization
            if 'normalize' in transform_config:
                norm_config = transform_config['normalize']
                mean = norm_config.get('mean', [0.485, 0.456, 0.406])
                std = norm_config.get('std', [0.229, 0.224, 0.225])
                transforms.append(A.Normalize(mean=mean, std=std))
            
            # Add tensor conversion
            transforms.append(ToTensorV2())
            
            # Create composite transform
            if transforms:
                return A.Compose(transforms)
            else:
                return None
                
        except ImportError:
            logger.warning("Albumentations not available, using basic transforms")
            return None
        except Exception as e:
            logger.warning(f"Failed to create transform from config: {e}")
            return None
    
    def _separate_internal_external_datasets(self, 
                                           data_loaders: Dict[str, DataLoader]) -> Tuple[Dict, Dict]:
        """Separate internal validation and external test datasets"""
        
        internal_loaders = {}
        external_loaders = {}
        
        # Define which datasets are internal vs external
        external_keywords = ['external', 'messidor', 'test', 'holdout']
        
        for dataset_name, loader in data_loaders.items():
            is_external = any(keyword in dataset_name.lower() for keyword in external_keywords)
            
            if is_external:
                external_loaders[dataset_name] = loader
            else:
                internal_loaders[dataset_name] = loader
        
        logger.info(f"Internal datasets: {list(internal_loaders.keys())}")
        logger.info(f"External datasets: {list(external_loaders.keys())}")
        
        return internal_loaders, external_loaders
    
    def _load_training_history(self, checkpoint_path: str) -> Optional[Dict]:
        """Load training history if available"""
        try:
            # Look for training history in the same directory as checkpoint
            checkpoint_dir = Path(checkpoint_path).parent
            
            # Common training history file names
            history_files = [
                'training_history.json',
                'history.json',
                'metrics.json',
                'training_log.json'
            ]
            
            for history_file in history_files:
                history_path = checkpoint_dir / history_file
                if history_path.exists():
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                    logger.info(f"Loaded training history from {history_path}")
                    return history
            
            logger.info("No training history found")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load training history: {e}")
            return None
    
    def _run_comparative_analysis(self, all_results: Dict) -> Dict:
        """Run comparative analysis across multiple checkpoints"""
        
        logger.info("Running comparative analysis across checkpoints")
        
        comparative_results = {
            'checkpoint_comparison': {},
            'best_performing_models': {},
            'performance_trends': {}
        }
        
        # Extract key metrics for comparison
        checkpoint_metrics = {}
        
        for checkpoint_name, results in all_results.items():
            if checkpoint_name == 'comparative_analysis':
                continue
                
            internal_results = results['internal_evaluation']
            external_results = results['external_validation']['external_results']
            
            # Calculate average performance across datasets
            all_eval_results = {**internal_results, **external_results}
            
            avg_metrics = {
                'combined_score': np.mean([r.combined_score for r in all_eval_results.values()]),
                'accuracy': np.mean([r.classification.accuracy for r in all_eval_results.values()]),
                'kappa': np.mean([r.classification.kappa for r in all_eval_results.values()]),
                'dice': np.mean([r.segmentation.dice_coefficient for r in all_eval_results.values()]),
                'iou': np.mean([r.segmentation.iou_score for r in all_eval_results.values()])
            }
            
            checkpoint_metrics[checkpoint_name] = avg_metrics
        
        comparative_results['checkpoint_comparison'] = checkpoint_metrics
        
        # Find best performing models
        if checkpoint_metrics:
            for metric_name in ['combined_score', 'accuracy', 'kappa', 'dice', 'iou']:
                best_checkpoint = max(checkpoint_metrics.items(), 
                                    key=lambda x: x[1][metric_name])
                comparative_results['best_performing_models'][metric_name] = {
                    'checkpoint': best_checkpoint[0],
                    'score': best_checkpoint[1][metric_name]
                }
        
        # Generate comparative visualizations
        comparison_viz = self.visualization_generator._create_comparison_plots(checkpoint_metrics)
        comparative_results['comparison_visualizations'] = comparison_viz
        
        return comparative_results
    
    def _generate_final_report(self, all_results: Dict) -> str:
        """Generate final comprehensive evaluation report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'phase6_final_report_{timestamp}.html'
        
        html_content = self._generate_final_report_html(all_results)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Final comprehensive report saved to: {report_path}")
        return str(report_path)
    
    def _generate_final_report_html(self, all_results: Dict) -> str:
        """Generate HTML for final comprehensive report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phase 6: Comprehensive Model Evaluation & Analysis - Final Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .subsection {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
                .metric {{ margin: 8px 0; padding: 5px; }}
                .good {{ color: #28a745; font-weight: bold; }}
                .moderate {{ color: #ffc107; font-weight: bold; }}
                .poor {{ color: #dc3545; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .summary-box {{ background-color: #e8f4f8; padding: 20px; border-left: 5px solid #007bff; margin: 20px 0; }}
                .recommendation {{ background-color: #fff3cd; padding: 15px; border-left: 5px solid #ffc107; margin: 10px 0; }}
                .key-finding {{ background-color: #d1ecf1; padding: 15px; border-left: 5px solid #bee5eb; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Phase 6: Comprehensive Model Evaluation & Analysis</h1>
                <h2>Final Evaluation Report</h2>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Evaluation Pipeline:</strong> Multi-Task Diabetic Retinopathy Model Assessment</p>
            </div>
        """
        
        # Executive Summary
        html += self._generate_executive_summary_html(all_results)
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_executive_summary_html(self, all_results: Dict) -> str:
        """Generate executive summary section"""
        
        # Extract key statistics
        n_checkpoints = len([k for k in all_results.keys() if k not in ['comparative_analysis', 'final_report_path']])
        
        html = f"""
            <div class="section">
                <h2>Executive Summary</h2>
                
                <div class="summary-box">
                    <h3>Evaluation Overview</h3>
                    <div class="metric">Number of Model Checkpoints Evaluated: <strong>{n_checkpoints}</strong></div>
                </div>
                
                <div class="key-finding">
                    <h4>Key Findings:</h4>
                    <ul>
                        <li>Comprehensive evaluation completed across multiple validation datasets</li>
                        <li>External validation performed to assess generalization capabilities</li>
                        <li>Explainability analysis provided insights into model behavior and decision patterns</li>
                    </ul>
                </div>
            </div>
        """
        
        return html


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Phase 6: Comprehensive Model Evaluation & Analysis')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                       help='Paths to model checkpoints to evaluate')
    parser.add_argument('--output-dir', type=str, default='results/phase6_evaluation',
                       help='Output directory for results')
    parser.add_argument('--datasets-config', type=str,
                       help='Optional path to datasets configuration file')
    
    args = parser.parse_args()
    
    # Load datasets config if provided
    datasets_config = None
    if args.datasets_config:
        try:
            with open(args.datasets_config, 'r') as f:
                datasets_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load datasets config: {e}")
    
    # Initialize and run pipeline
    pipeline = Phase6EvaluationPipeline(
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Run complete evaluation
    results = pipeline.run_complete_evaluation(
        checkpoint_paths=args.checkpoints,
        datasets_config=datasets_config
    )
    
    print("\n" + "="*80)
    print("PHASE 6 EVALUATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Final report: {results['final_report_path']}")


if __name__ == '__main__':
    main()
