"""
Phase 5: Advanced Monitoring and Analysis System
Real-time monitoring, quality control, and comprehensive analysis for training experiments.
"""

import os
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
import queue
from collections import defaultdict, deque

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific point in time"""
    
    timestamp: float
    epoch: int
    step: int
    phase: str
    
    # Loss metrics
    train_loss: float
    val_loss: float
    cls_loss: float
    seg_loss: float
    
    # Performance metrics
    cls_accuracy: Optional[float] = None
    cls_f1_score: Optional[float] = None
    seg_dice_score: Optional[float] = None
    seg_iou_score: Optional[float] = None
    
    # Learning metrics
    learning_rate: float = 0.0
    gradient_norm: Optional[float] = None
    
    # Resource metrics
    gpu_memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    
    # Training dynamics
    batch_time: Optional[float] = None
    data_loading_time: Optional[float] = None


@dataclass
class ExperimentStatus:
    """Current status of a training experiment"""
    
    experiment_id: str
    start_time: float
    current_epoch: int
    total_epochs: int
    current_phase: str
    status: str  # 'running', 'completed', 'failed', 'paused'
    
    best_metric: float
    best_epoch: int
    current_metric: float
    
    estimated_time_remaining: Optional[float] = None
    last_update: Optional[float] = None
    error_message: Optional[str] = None


class RealTimeMonitor:
    """Real-time monitoring system for training experiments"""
    
    def __init__(self, experiment_dir: str, update_interval: float = 30.0):
        self.experiment_dir = Path(experiment_dir)
        self.update_interval = update_interval
        
        # Monitoring state
        self.active_experiments = {}
        self.metrics_history = defaultdict(list)
        self.alert_thresholds = self._setup_alert_thresholds()
        
        # Threading for background monitoring
        self.monitoring_queue = queue.Queue()
        self.stop_monitoring = threading.Event()
        self.monitor_thread = None
        
        # Setup logging
        self.logger = logging.getLogger('RealTimeMonitor')
        
        # Create monitoring directory
        self.monitoring_dir = self.experiment_dir / "monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup alert thresholds for various metrics"""
        return {
            'loss': {
                'explosion': 100.0,  # Loss > 100
                'stagnation_epochs': 20,  # No improvement for 20 epochs
                'nan_detection': True
            },
            'gradient': {
                'explosion': 10.0,  # Gradient norm > 10
                'vanishing': 1e-6   # Gradient norm < 1e-6
            },
            'memory': {
                'gpu_critical': 95.0,  # GPU memory > 95%
                'cpu_critical': 90.0,  # CPU > 90%
                'system_critical': 95.0  # System memory > 95%
            },
            'performance': {
                'accuracy_drop': 0.1,  # Accuracy drop > 10%
                'dice_threshold': 0.3   # Dice score < 0.3
            }
        }
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("Real-time monitoring started")
    
    def stop_monitoring_thread(self):
        """Stop background monitoring thread"""
        self.stop_monitoring.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        while not self.stop_monitoring.is_set():
            try:
                # Check for new metric updates
                self._process_metric_updates()
                
                # Update experiment statuses
                self._update_experiment_statuses()
                
                # Check for alerts
                self._check_alerts()
                
                # Save monitoring state
                self._save_monitoring_state()
                
                # Wait for next update
                self.stop_monitoring.wait(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Brief pause before retrying
    
    def add_experiment(self, experiment_id: str, total_epochs: int):
        """Add a new experiment to monitoring"""
        status = ExperimentStatus(
            experiment_id=experiment_id,
            start_time=time.time(),
            current_epoch=0,
            total_epochs=total_epochs,
            current_phase="initialization",
            status="running",
            best_metric=float('-inf'),
            best_epoch=0,
            current_metric=0.0,
            last_update=time.time()
        )
        
        self.active_experiments[experiment_id] = status
        self.logger.info(f"Added experiment to monitoring: {experiment_id}")
    
    def update_metrics(self, experiment_id: str, metrics: TrainingMetrics):
        """Update metrics for an experiment"""
        self.monitoring_queue.put(('metrics', experiment_id, metrics))
    
    def update_status(self, experiment_id: str, status_update: Dict[str, Any]):
        """Update experiment status"""
        self.monitoring_queue.put(('status', experiment_id, status_update))
    
    def _process_metric_updates(self):
        """Process queued metric updates"""
        while not self.monitoring_queue.empty():
            try:
                update_type, experiment_id, data = self.monitoring_queue.get_nowait()
                
                if update_type == 'metrics':
                    self._update_experiment_metrics(experiment_id, data)
                elif update_type == 'status':
                    self._update_experiment_status(experiment_id, data)
                    
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing metric update: {e}")
    
    def _update_experiment_metrics(self, experiment_id: str, metrics: TrainingMetrics):
        """Update metrics for an experiment"""
        # Store metrics history
        self.metrics_history[experiment_id].append(metrics)
        
        # Limit history size to prevent memory issues
        if len(self.metrics_history[experiment_id]) > 10000:
            self.metrics_history[experiment_id] = self.metrics_history[experiment_id][-5000:]
        
        # Update experiment status
        if experiment_id in self.active_experiments:
            status = self.active_experiments[experiment_id]
            status.current_epoch = metrics.epoch
            status.current_phase = metrics.phase
            status.current_metric = metrics.val_loss  # Use validation loss as primary metric
            status.last_update = metrics.timestamp
            
            # Update best metric (assuming lower is better for loss)
            if metrics.val_loss < status.best_metric or status.best_metric == float('-inf'):
                status.best_metric = metrics.val_loss
                status.best_epoch = metrics.epoch
            
            # Estimate time remaining
            if status.current_epoch > 0:
                elapsed_time = metrics.timestamp - status.start_time
                avg_time_per_epoch = elapsed_time / status.current_epoch
                remaining_epochs = status.total_epochs - status.current_epoch
                status.estimated_time_remaining = remaining_epochs * avg_time_per_epoch
    
    def _update_experiment_status(self, experiment_id: str, status_update: Dict[str, Any]):
        """Update experiment status"""
        if experiment_id in self.active_experiments:
            status = self.active_experiments[experiment_id]
            
            for key, value in status_update.items():
                if hasattr(status, key):
                    setattr(status, key, value)
            
            status.last_update = time.time()
    
    def _update_experiment_statuses(self):
        """Update statuses for all experiments"""
        current_time = time.time()
        
        for experiment_id, status in self.active_experiments.items():
            # Check for stale experiments (no updates for 5 minutes)
            if status.last_update and (current_time - status.last_update) > 300:
                if status.status == "running":
                    status.status = "stalled"
                    self.logger.warning(f"Experiment {experiment_id} appears stalled")
    
    def _check_alerts(self):
        """Check for alert conditions across all experiments"""
        for experiment_id, metrics_list in self.metrics_history.items():
            if not metrics_list:
                continue
            
            latest_metrics = metrics_list[-1]
            self._check_experiment_alerts(experiment_id, latest_metrics, metrics_list)
    
    def _check_experiment_alerts(self, experiment_id: str, latest_metrics: TrainingMetrics, 
                                history: List[TrainingMetrics]):
        """Check alerts for a specific experiment"""
        alerts = []
        
        # Loss explosion
        if latest_metrics.train_loss > self.alert_thresholds['loss']['explosion']:
            alerts.append(f"Loss explosion detected: {latest_metrics.train_loss:.2f}")
        
        # NaN detection
        if np.isnan(latest_metrics.train_loss) or np.isnan(latest_metrics.val_loss):
            alerts.append("NaN values detected in loss")
        
        # Gradient problems
        if latest_metrics.gradient_norm:
            if latest_metrics.gradient_norm > self.alert_thresholds['gradient']['explosion']:
                alerts.append(f"Gradient explosion: {latest_metrics.gradient_norm:.2e}")
            elif latest_metrics.gradient_norm < self.alert_thresholds['gradient']['vanishing']:
                alerts.append(f"Vanishing gradients: {latest_metrics.gradient_norm:.2e}")
        
        # Memory issues
        if latest_metrics.gpu_memory_mb and latest_metrics.gpu_memory_mb > self.alert_thresholds['memory']['gpu_critical']:
            alerts.append(f"Critical GPU memory usage: {latest_metrics.gpu_memory_mb:.1f}%")
        
        # Performance degradation
        if len(history) > 10:
            recent_val_losses = [m.val_loss for m in history[-10:]]
            if len(recent_val_losses) == 10 and all(np.diff(recent_val_losses) > 0):
                alerts.append("Validation loss increasing for 10 consecutive steps")
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"ALERT [{experiment_id}]: {alert}")
    
    def _save_monitoring_state(self):
        """Save current monitoring state to disk"""
        try:
            # Save experiment statuses
            status_data = {
                exp_id: asdict(status) 
                for exp_id, status in self.active_experiments.items()
            }
            
            status_file = self.monitoring_dir / "experiment_statuses.json"
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2, default=str)
            
            # Save recent metrics (last 100 per experiment)
            recent_metrics = {}
            for exp_id, metrics_list in self.metrics_history.items():
                if metrics_list:
                    recent_metrics[exp_id] = [
                        asdict(m) for m in metrics_list[-100:]
                    ]
            
            metrics_file = self.monitoring_dir / "recent_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(recent_metrics, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save monitoring state: {e}")
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get summary for a specific experiment"""
        if experiment_id not in self.active_experiments:
            return {"error": "Experiment not found"}
        
        status = self.active_experiments[experiment_id]
        metrics_list = self.metrics_history.get(experiment_id, [])
        
        summary = {
            'status': asdict(status),
            'metrics_count': len(metrics_list),
            'runtime_hours': (time.time() - status.start_time) / 3600
        }
        
        if metrics_list:
            latest_metrics = metrics_list[-1]
            summary['latest_metrics'] = asdict(latest_metrics)
            
            # Performance trends
            if len(metrics_list) > 10:
                recent_losses = [m.val_loss for m in metrics_list[-10:]]
                summary['val_loss_trend'] = 'improving' if recent_losses[-1] < recent_losses[0] else 'worsening'
        
        return summary
    
    def get_all_experiments_summary(self) -> Dict[str, Any]:
        """Get summary for all experiments"""
        summary = {
            'total_experiments': len(self.active_experiments),
            'active_experiments': sum(1 for s in self.active_experiments.values() if s.status == 'running'),
            'completed_experiments': sum(1 for s in self.active_experiments.values() if s.status == 'completed'),
            'failed_experiments': sum(1 for s in self.active_experiments.values() if s.status == 'failed'),
            'experiments': {}
        }
        
        for exp_id in self.active_experiments:
            summary['experiments'][exp_id] = self.get_experiment_summary(exp_id)
        
        return summary


class AdvancedAnalyzer:
    """Advanced analysis system for training results"""
    
    def __init__(self, experiments_dir: str):
        self.experiments_dir = Path(experiments_dir)
        self.analysis_dir = self.experiments_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('AdvancedAnalyzer')
    
    def analyze_experiment_results(self, experiment_dirs: List[str]) -> Dict[str, Any]:
        """Analyze results from multiple experiments"""
        self.logger.info(f"Analyzing {len(experiment_dirs)} experiments")
        
        # Load all experiment data
        experiments_data = []
        for exp_dir in experiment_dirs:
            try:
                exp_data = self._load_experiment_data(exp_dir)
                if exp_data:
                    experiments_data.append(exp_data)
            except Exception as e:
                self.logger.warning(f"Failed to load experiment {exp_dir}: {e}")
        
        if not experiments_data:
            self.logger.error("No valid experiment data found")
            return {}
        
        # Perform comprehensive analysis
        analysis_results = {
            'summary': self._generate_summary_statistics(experiments_data),
            'performance_analysis': self._analyze_performance_trends(experiments_data),
            'parameter_analysis': self._analyze_parameter_effects(experiments_data),
            'resource_analysis': self._analyze_resource_usage(experiments_data),
            'convergence_analysis': self._analyze_convergence_patterns(experiments_data),
            'recommendations': self._generate_recommendations(experiments_data)
        }
        
        # Save analysis results
        self._save_analysis_results(analysis_results)
        
        return analysis_results
    
    def _load_experiment_data(self, exp_dir: str) -> Optional[Dict[str, Any]]:
        """Load data from a single experiment"""
        exp_path = Path(exp_dir)
        
        # Load training results
        results_file = exp_path / "phase5_training_report.json"
        if not results_file.exists():
            results_file = exp_path / "training_results.yaml"
        
        if not results_file.exists():
            self.logger.warning(f"No results file found in {exp_dir}")
            return None
        
        try:
            with open(results_file, 'r') as f:
                if results_file.suffix == '.json':
                    data = json.load(f)
                else:
                    import yaml
                    data = yaml.safe_load(f)
            
            # Extract key information
            experiment_data = {
                'experiment_id': exp_path.name,
                'experiment_dir': str(exp_path),
                'results': data,
                'config': self._extract_config(data),
                'metrics': self._extract_metrics(data),
                'final_performance': self._extract_final_performance(data)
            }
            
            return experiment_data
            
        except Exception as e:
            self.logger.error(f"Failed to load experiment data from {exp_dir}: {e}")
            return None
    
    def _extract_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration from experiment data"""
        config = {}
        
        # Look for configuration in various locations
        if 'configuration' in data:
            config = data['configuration']
        elif 'config' in data:
            config = data['config']
        elif 'final_config' in data.get('training_results', {}):
            config = data['training_results']['final_config']
        
        return config
    
    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, List]:
        """Extract training metrics from experiment data"""
        metrics = {}
        
        # Look for training history
        history = None
        if 'training_results' in data:
            if 'full_training' in data['training_results']:
                history = data['training_results']['full_training'].get('training_history')
            elif 'training_history' in data['training_results']:
                history = data['training_results']['training_history']
        
        if history:
            metrics = {
                'train_losses': history.get('train_losses', []),
                'val_losses': history.get('val_losses', []),
                'learning_rates': history.get('learning_rates', []),
                'epochs': list(range(1, len(history.get('train_losses', [])) + 1))
            }
        
        return metrics
    
    def _extract_final_performance(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract final performance metrics"""
        performance = {}
        
        # Look for best metric
        if 'training_results' in data:
            if 'full_training' in data['training_results']:
                training_data = data['training_results']['full_training']
                performance['best_metric'] = training_data.get('best_metric')
                performance['total_epochs'] = training_data.get('total_epochs')
                performance['training_time_hours'] = training_data.get('training_time_hours')
        
        return performance
    
    def _generate_summary_statistics(self, experiments_data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics across experiments"""
        summary = {
            'total_experiments': len(experiments_data),
            'completed_experiments': 0,
            'failed_experiments': 0,
            'performance_statistics': {},
            'runtime_statistics': {}
        }
        
        # Extract performance metrics
        best_metrics = []
        training_times = []
        total_epochs = []
        
        for exp_data in experiments_data:
            perf = exp_data.get('final_performance', {})
            
            if perf.get('best_metric') is not None:
                best_metrics.append(perf['best_metric'])
                summary['completed_experiments'] += 1
            else:
                summary['failed_experiments'] += 1
            
            if perf.get('training_time_hours'):
                training_times.append(perf['training_time_hours'])
            
            if perf.get('total_epochs'):
                total_epochs.append(perf['total_epochs'])
        
        # Calculate statistics
        if best_metrics:
            summary['performance_statistics'] = {
                'mean_best_metric': float(np.mean(best_metrics)),
                'std_best_metric': float(np.std(best_metrics)),
                'min_best_metric': float(np.min(best_metrics)),
                'max_best_metric': float(np.max(best_metrics)),
                'median_best_metric': float(np.median(best_metrics))
            }
        
        if training_times:
            summary['runtime_statistics'] = {
                'mean_training_time': float(np.mean(training_times)),
                'total_training_time': float(np.sum(training_times)),
                'min_training_time': float(np.min(training_times)),
                'max_training_time': float(np.max(training_times))
            }
        
        return summary
    
    def _analyze_performance_trends(self, experiments_data: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trends across experiments"""
        trends = {
            'convergence_patterns': {},
            'learning_curve_analysis': {},
            'best_performing_configs': []
        }
        
        # Analyze convergence for each experiment
        convergence_data = []
        for exp_data in experiments_data:
            metrics = exp_data.get('metrics', {})
            val_losses = metrics.get('val_losses', [])
            
            if len(val_losses) > 10:
                # Find convergence point (where improvement < 1% for 5 epochs)
                convergence_epoch = self._find_convergence_point(val_losses)
                convergence_data.append({
                    'experiment_id': exp_data['experiment_id'],
                    'convergence_epoch': convergence_epoch,
                    'final_loss': val_losses[-1] if val_losses else None,
                    'best_loss': min(val_losses) if val_losses else None
                })
        
        if convergence_data:
            conv_epochs = [d['convergence_epoch'] for d in convergence_data if d['convergence_epoch']]
            if conv_epochs:
                trends['convergence_patterns'] = {
                    'mean_convergence_epoch': float(np.mean(conv_epochs)),
                    'std_convergence_epoch': float(np.std(conv_epochs)),
                    'early_convergers': len([e for e in conv_epochs if e < 20]),
                    'late_convergers': len([e for e in conv_epochs if e > 40])
                }
        
        # Find best performing configurations
        performance_data = []
        for exp_data in experiments_data:
            perf = exp_data.get('final_performance', {})
            config = exp_data.get('config', {})
            
            if perf.get('best_metric') is not None:
                performance_data.append({
                    'experiment_id': exp_data['experiment_id'],
                    'best_metric': perf['best_metric'],
                    'config': config
                })
        
        # Sort by performance and take top 5
        performance_data.sort(key=lambda x: x['best_metric'])
        trends['best_performing_configs'] = performance_data[:5]
        
        return trends
    
    def _find_convergence_point(self, losses: List[float], 
                               improvement_threshold: float = 0.01,
                               patience: int = 5) -> Optional[int]:
        """Find the epoch where training converged"""
        if len(losses) < patience + 1:
            return None
        
        best_loss = float('inf')
        no_improvement_count = 0
        
        for i, loss in enumerate(losses):
            if loss < best_loss * (1 - improvement_threshold):
                best_loss = loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
                if no_improvement_count >= patience:
                    return i - patience + 1
        
        return None
    
    def _analyze_parameter_effects(self, experiments_data: List[Dict]) -> Dict[str, Any]:
        """Analyze effects of different parameters on performance"""
        parameter_effects = {}
        
        # Extract parameter-performance pairs
        param_performance = defaultdict(list)
        
        for exp_data in experiments_data:
            config = exp_data.get('config', {})
            perf = exp_data.get('final_performance', {})
            
            if perf.get('best_metric') is not None:
                # Extract key parameters
                params = self._extract_key_parameters(config)
                
                for param_name, param_value in params.items():
                    param_performance[param_name].append({
                        'value': param_value,
                        'performance': perf['best_metric']
                    })
        
        # Analyze each parameter
        for param_name, data_points in param_performance.items():
            if len(data_points) > 1:
                parameter_effects[param_name] = self._analyze_parameter_effect(data_points)
        
        return parameter_effects
    
    def _extract_key_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key parameters from configuration"""
        params = {}
        
        # Common parameters to analyze
        param_paths = [
            'optimizer.learning_rate',
            'hardware.batch_size',
            'model.backbone_name',
            'loss.focal_gamma',
            'phase1_epochs',
            'phase2_epochs',
            'phase3_epochs'
        ]
        
        for param_path in param_paths:
            value = self._get_nested_value(config, param_path)
            if value is not None:
                params[param_path] = value
        
        return params
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def _analyze_parameter_effect(self, data_points: List[Dict]) -> Dict[str, Any]:
        """Analyze the effect of a parameter on performance"""
        values = [dp['value'] for dp in data_points]
        performances = [dp['performance'] for dp in data_points]
        
        analysis = {
            'data_points': len(data_points),
            'unique_values': len(set(values)),
            'value_range': [min(values), max(values)] if isinstance(values[0], (int, float)) else None
        }
        
        # If parameter is categorical, analyze by category
        if not isinstance(values[0], (int, float)):
            category_performance = defaultdict(list)
            for dp in data_points:
                category_performance[dp['value']].append(dp['performance'])
            
            analysis['categorical_analysis'] = {}
            for category, perfs in category_performance.items():
                analysis['categorical_analysis'][str(category)] = {
                    'count': len(perfs),
                    'mean_performance': float(np.mean(perfs)),
                    'std_performance': float(np.std(perfs)) if len(perfs) > 1 else 0.0
                }
        
        # If parameter is numerical, calculate correlation
        else:
            if len(set(values)) > 1:  # Need variation to calculate correlation
                correlation, p_value = stats.pearsonr(values, performances)
                analysis['correlation'] = {
                    'pearson_r': float(correlation),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        return analysis
    
    def _analyze_resource_usage(self, experiments_data: List[Dict]) -> Dict[str, Any]:
        """Analyze resource usage patterns"""
        resource_analysis = {}
        
        # Extract resource usage data
        gpu_usage = []
        memory_usage = []
        training_times = []
        
        for exp_data in experiments_data:
            results = exp_data.get('results', {})
            
            # Look for resource usage in various locations
            if 'resource_usage' in results:
                usage = results['resource_usage']
                if 'peak_gpu_memory_percent' in usage:
                    gpu_usage.append(usage['peak_gpu_memory_percent'])
                if 'peak_memory_percent' in usage:
                    memory_usage.append(usage['peak_memory_percent'])
            
            perf = exp_data.get('final_performance', {})
            if 'training_time_hours' in perf:
                training_times.append(perf['training_time_hours'])
        
        # Analyze resource statistics
        if gpu_usage:
            resource_analysis['gpu_memory'] = {
                'mean_peak_usage': float(np.mean(gpu_usage)),
                'max_peak_usage': float(np.max(gpu_usage)),
                'std_peak_usage': float(np.std(gpu_usage))
            }
        
        if memory_usage:
            resource_analysis['system_memory'] = {
                'mean_peak_usage': float(np.mean(memory_usage)),
                'max_peak_usage': float(np.max(memory_usage)),
                'std_peak_usage': float(np.std(memory_usage))
            }
        
        if training_times:
            resource_analysis['training_efficiency'] = {
                'mean_time_per_experiment': float(np.mean(training_times)),
                'total_compute_hours': float(np.sum(training_times)),
                'time_variation': float(np.std(training_times))
            }
        
        return resource_analysis
    
    def _analyze_convergence_patterns(self, experiments_data: List[Dict]) -> Dict[str, Any]:
        """Analyze convergence patterns across experiments"""
        convergence_analysis = {}
        
        # Analyze learning curves
        all_curves = []
        for exp_data in experiments_data:
            metrics = exp_data.get('metrics', {})
            val_losses = metrics.get('val_losses', [])
            
            if len(val_losses) > 10:
                # Normalize curve length and values
                normalized_curve = self._normalize_learning_curve(val_losses)
                all_curves.append(normalized_curve)
        
        if all_curves:
            # Calculate average learning curve
            min_length = min(len(curve) for curve in all_curves)
            truncated_curves = [curve[:min_length] for curve in all_curves]
            avg_curve = np.mean(truncated_curves, axis=0)
            std_curve = np.std(truncated_curves, axis=0)
            
            convergence_analysis['average_learning_curve'] = {
                'mean_curve': avg_curve.tolist(),
                'std_curve': std_curve.tolist(),
                'n_curves': len(all_curves)
            }
            
            # Identify convergence characteristics
            convergence_analysis['convergence_characteristics'] = {
                'fast_convergers': len([c for c in all_curves if self._is_fast_converger(c)]),
                'slow_convergers': len([c for c in all_curves if self._is_slow_converger(c)]),
                'oscillatory_patterns': len([c for c in all_curves if self._has_oscillatory_pattern(c)])
            }
        
        return convergence_analysis
    
    def _normalize_learning_curve(self, losses: List[float]) -> List[float]:
        """Normalize learning curve for comparison"""
        if not losses or len(losses) < 2:
            return []
        
        # Normalize by initial value
        initial_loss = losses[0]
        if initial_loss == 0:
            return losses
        
        return [(loss / initial_loss) for loss in losses]
    
    def _is_fast_converger(self, curve: List[float]) -> bool:
        """Check if learning curve shows fast convergence"""
        if len(curve) < 10:
            return False
        
        # Check if 80% of improvement happens in first 25% of training
        initial_loss = curve[0]
        final_loss = curve[-1]
        total_improvement = initial_loss - final_loss
        
        quarter_point = len(curve) // 4
        quarter_loss = curve[quarter_point]
        quarter_improvement = initial_loss - quarter_loss
        
        return quarter_improvement > 0.8 * total_improvement
    
    def _is_slow_converger(self, curve: List[float]) -> bool:
        """Check if learning curve shows slow convergence"""
        if len(curve) < 20:
            return False
        
        # Check if improvement is still significant in last quarter
        three_quarter_point = (3 * len(curve)) // 4
        final_quarter = curve[three_quarter_point:]
        
        # Check for continued improvement
        if len(final_quarter) > 5:
            recent_improvement = final_quarter[0] - final_quarter[-1]
            total_improvement = curve[0] - curve[-1]
            return recent_improvement > 0.1 * total_improvement
        
        return False
    
    def _has_oscillatory_pattern(self, curve: List[float]) -> bool:
        """Check if learning curve has oscillatory pattern"""
        if len(curve) < 10:
            return False
        
        # Count direction changes
        direction_changes = 0
        for i in range(1, len(curve) - 1):
            if ((curve[i] > curve[i-1] and curve[i] > curve[i+1]) or
                (curve[i] < curve[i-1] and curve[i] < curve[i+1])):
                direction_changes += 1
        
        # Oscillatory if many direction changes relative to length
        return direction_changes > len(curve) * 0.3
    
    def _generate_recommendations(self, experiments_data: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Analyze performance distribution
        performances = []
        for exp_data in experiments_data:
            perf = exp_data.get('final_performance', {})
            if perf.get('best_metric') is not None:
                performances.append(perf['best_metric'])
        
        if performances:
            mean_perf = np.mean(performances)
            std_perf = np.std(performances)
            
            if std_perf > mean_perf * 0.2:
                recommendations.append(
                    "High performance variability detected. Consider standardizing training procedures."
                )
            
            if mean_perf > 1.0:  # Assuming loss metric where lower is better
                recommendations.append(
                    "Average performance could be improved. Consider hyperparameter optimization."
                )
        
        # Check for training efficiency
        training_times = []
        for exp_data in experiments_data:
            perf = exp_data.get('final_performance', {})
            if 'training_time_hours' in perf:
                training_times.append(perf['training_time_hours'])
        
        if training_times:
            mean_time = np.mean(training_times)
            if mean_time > 10:
                recommendations.append(
                    "Long training times detected. Consider reducing model complexity or using mixed precision."
                )
        
        # Add general recommendations
        recommendations.extend([
            "Monitor gradient norms to detect optimization issues early.",
            "Use early stopping to prevent overfitting and save compute time.",
            "Consider ensemble methods for best performing configurations.",
            "Implement cross-validation for more robust performance estimates."
        ])
        
        return recommendations
    
    def _save_analysis_results(self, analysis_results: Dict[str, Any]):
        """Save analysis results to file"""
        results_file = self.analysis_dir / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        self.logger.info(f"Analysis results saved to: {results_file}")
    
    def generate_analysis_plots(self, experiments_data: List[Dict]) -> str:
        """Generate comprehensive analysis plots"""
        plots_dir = self.analysis_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Create performance comparison plot
        self._create_performance_comparison_plot(experiments_data, plots_dir)
        
        # Create learning curves plot
        self._create_learning_curves_plot(experiments_data, plots_dir)
        
        # Create parameter effects plot
        self._create_parameter_effects_plot(experiments_data, plots_dir)
        
        # Create resource usage plot
        self._create_resource_usage_plot(experiments_data, plots_dir)
        
        self.logger.info(f"Analysis plots saved to: {plots_dir}")
        return str(plots_dir)
    
    def _create_performance_comparison_plot(self, experiments_data: List[Dict], plots_dir: Path):
        """Create performance comparison plot"""
        try:
            # Extract performance data
            exp_names = []
            performances = []
            
            for exp_data in experiments_data:
                perf = exp_data.get('final_performance', {})
                if perf.get('best_metric') is not None:
                    exp_names.append(exp_data['experiment_id'])
                    performances.append(perf['best_metric'])
            
            if not performances:
                return
            
            # Create plot
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(performances)), performances)
            plt.xlabel('Experiment')
            plt.ylabel('Best Metric (Lower is Better)')
            plt.title('Performance Comparison Across Experiments')
            plt.xticks(range(len(exp_names)), exp_names, rotation=45, ha='right')
            
            # Color bars by performance (green for best, red for worst)
            norm_performances = np.array(performances)
            norm_performances = (norm_performances - norm_performances.min()) / (norm_performances.max() - norm_performances.min())
            
            for bar, norm_perf in zip(bars, norm_performances):
                bar.set_color(plt.cm.RdYlGn_r(norm_perf))
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create performance comparison plot: {e}")
    
    def _create_learning_curves_plot(self, experiments_data: List[Dict], plots_dir: Path):
        """Create learning curves plot"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot individual learning curves
            subplot_idx = 1
            n_experiments = min(len(experiments_data), 9)  # Limit to 9 subplots
            n_cols = 3
            n_rows = (n_experiments + n_cols - 1) // n_cols
            
            for i, exp_data in enumerate(experiments_data[:n_experiments]):
                plt.subplot(n_rows, n_cols, subplot_idx)
                
                metrics = exp_data.get('metrics', {})
                train_losses = metrics.get('train_losses', [])
                val_losses = metrics.get('val_losses', [])
                epochs = metrics.get('epochs', [])
                
                if train_losses and val_losses:
                    plt.plot(epochs, train_losses, label='Train Loss', alpha=0.8)
                    plt.plot(epochs, val_losses, label='Val Loss', alpha=0.8)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title(f'{exp_data["experiment_id"][:20]}...')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                subplot_idx += 1
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create learning curves plot: {e}")
    
    def _create_parameter_effects_plot(self, experiments_data: List[Dict], plots_dir: Path):
        """Create parameter effects plot"""
        try:
            # Extract parameter-performance relationships
            param_data = defaultdict(lambda: {'values': [], 'performances': []})
            
            for exp_data in experiments_data:
                config = exp_data.get('config', {})
                perf = exp_data.get('final_performance', {})
                
                if perf.get('best_metric') is not None:
                    # Extract key numerical parameters
                    if 'optimizer' in config and 'learning_rate' in config['optimizer']:
                        param_data['learning_rate']['values'].append(config['optimizer']['learning_rate'])
                        param_data['learning_rate']['performances'].append(perf['best_metric'])
                    
                    if 'hardware' in config and 'batch_size' in config['hardware']:
                        param_data['batch_size']['values'].append(config['hardware']['batch_size'])
                        param_data['batch_size']['performances'].append(perf['best_metric'])
            
            # Create scatter plots
            n_params = len(param_data)
            if n_params > 0:
                fig, axes = plt.subplots(1, min(n_params, 3), figsize=(15, 5))
                if n_params == 1:
                    axes = [axes]
                
                for i, (param_name, data) in enumerate(list(param_data.items())[:3]):
                    ax = axes[i] if n_params > 1 else axes[0]
                    
                    ax.scatter(data['values'], data['performances'], alpha=0.7)
                    ax.set_xlabel(param_name.replace('_', ' ').title())
                    ax.set_ylabel('Best Metric')
                    ax.set_title(f'Effect of {param_name.replace("_", " ").title()}')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'parameter_effects.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create parameter effects plot: {e}")
    
    def _create_resource_usage_plot(self, experiments_data: List[Dict], plots_dir: Path):
        """Create resource usage plot"""
        try:
            # Extract resource usage data
            gpu_usage = []
            memory_usage = []
            training_times = []
            exp_names = []
            
            for exp_data in experiments_data:
                results = exp_data.get('results', {})
                perf = exp_data.get('final_performance', {})
                
                if 'resource_usage' in results:
                    usage = results['resource_usage']
                    exp_names.append(exp_data['experiment_id'])
                    gpu_usage.append(usage.get('peak_gpu_memory_percent', 0))
                    memory_usage.append(usage.get('peak_memory_percent', 0))
                    training_times.append(perf.get('training_time_hours', 0))
            
            if exp_names:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # GPU memory usage
                axes[0, 0].bar(range(len(gpu_usage)), gpu_usage)
                axes[0, 0].set_title('Peak GPU Memory Usage (%)')
                axes[0, 0].set_ylabel('GPU Memory %')
                axes[0, 0].tick_params(axis='x', labelbottom=False)
                
                # System memory usage
                axes[0, 1].bar(range(len(memory_usage)), memory_usage)
                axes[0, 1].set_title('Peak System Memory Usage (%)')
                axes[0, 1].set_ylabel('System Memory %')
                axes[0, 1].tick_params(axis='x', labelbottom=False)
                
                # Training time
                axes[1, 0].bar(range(len(training_times)), training_times)
                axes[1, 0].set_title('Training Time (Hours)')
                axes[1, 0].set_ylabel('Hours')
                axes[1, 0].set_xticks(range(len(exp_names)))
                axes[1, 0].set_xticklabels(exp_names, rotation=45, ha='right')
                
                # Resource efficiency (performance vs time)
                if len(training_times) > 1:
                    performances = []
                    for exp_data in experiments_data[:len(training_times)]:
                        perf = exp_data.get('final_performance', {})
                        if perf.get('best_metric') is not None:
                            performances.append(perf['best_metric'])
                    
                    if len(performances) == len(training_times):
                        axes[1, 1].scatter(training_times, performances)
                        axes[1, 1].set_xlabel('Training Time (Hours)')
                        axes[1, 1].set_ylabel('Best Metric')
                        axes[1, 1].set_title('Performance vs Training Time')
                        axes[1, 1].grid(True, alpha=0.3)
                    else:
                        axes[1, 1].axis('off')
                else:
                    axes[1, 1].axis('off')
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'resource_usage.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create resource usage plot: {e}")


def main():
    """Main function for monitoring and analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 5: Advanced Monitoring and Analysis")
    parser.add_argument('--mode', type=str, choices=['monitor', 'analyze'], 
                       default='analyze', help='Operation mode')
    parser.add_argument('--experiments-dir', type=str, 
                       default='experiments', help='Experiments directory')
    parser.add_argument('--experiment-ids', type=str, nargs='+',
                       help='Specific experiment IDs to analyze')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        # Run analysis
        analyzer = AdvancedAnalyzer(args.experiments_dir)
        
        # Find experiment directories
        exp_dirs = []
        if args.experiment_ids:
            for exp_id in args.experiment_ids:
                exp_dir = Path(args.experiments_dir) / exp_id
                if exp_dir.exists():
                    exp_dirs.append(str(exp_dir))
        else:
            # Analyze all experiments
            experiments_path = Path(args.experiments_dir)
            for exp_dir in experiments_path.iterdir():
                if exp_dir.is_dir() and exp_dir.name.startswith('diabetic_retinopathy'):
                    exp_dirs.append(str(exp_dir))
        
        if exp_dirs:
            print(f"Analyzing {len(exp_dirs)} experiments...")
            results = analyzer.analyze_experiment_results(exp_dirs)
            
            # Generate plots
            analyzer.generate_analysis_plots([
                analyzer._load_experiment_data(exp_dir) 
                for exp_dir in exp_dirs
            ])
            
            print("Analysis completed!")
            print(f"Results saved to: {analyzer.analysis_dir}")
        else:
            print("No experiment directories found")
    
    elif args.mode == 'monitor':
        # Start monitoring (example usage)
        monitor = RealTimeMonitor(args.experiments_dir)
        monitor.start_monitoring()
        
        print("Real-time monitoring started...")
        print("Press Ctrl+C to stop")
        
        try:
            import time
            while True:
                time.sleep(10)
                summary = monitor.get_all_experiments_summary()
                print(f"Active experiments: {summary['active_experiments']}")
        except KeyboardInterrupt:
            monitor.stop_monitoring_thread()
            print("Monitoring stopped")


if __name__ == "__main__":
    main()
