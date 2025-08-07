#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
Coordinates and runs all 6 pipeline phases for comprehensive diabetic retinopathy model training.

This main pipeline:
- Orchestrates all 6 phases from data processing to evaluation
- Provides options to run individual phases or complete workflow
- Manages dependencies between phases
- Generates comprehensive project reports
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/main_pipeline_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('MainPipeline')

def check_system_requirements():
    """Check system requirements for running the complete pipeline"""
    logger = logging.getLogger('MainPipeline')
    
    logger.info("Checking system requirements...")
    
    requirements_met = True
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        requirements_met = False
    else:
        logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check critical dependencies
    critical_packages = ['torch', 'torchvision', 'numpy', 'pandas', 'pillow', 'opencv-python']
    
    for package in critical_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✅ {package} available")
        except ImportError:
            logger.error(f"❌ {package} not found")
            requirements_met = False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.warning("⚠️ CUDA not available, training will be slower")
    except ImportError:
        pass
    
    # Check disk space (estimate for full pipeline)
    try:
        statvfs = os.statvfs('.')
        available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        if available_gb < 50:  # Minimum 50GB recommended
            logger.warning(f"⚠️ Low disk space: {available_gb:.1f}GB available (50GB+ recommended)")
        else:
            logger.info(f"✅ Disk space: {available_gb:.1f}GB available")
    except:
        logger.warning("⚠️ Could not check disk space")
    
    return requirements_met

def import_pipeline_modules():
    """Import all pipeline modules"""
    logger = logging.getLogger('MainPipeline')
    
    pipeline_modules = {}
    
    try:
        from run_pipeline1_data_processing import run_pipeline1_complete
        pipeline_modules['pipeline1'] = run_pipeline1_complete
        logger.info("✅ Pipeline 1 (Data Processing) module loaded")
    except ImportError as e:
        logger.error(f"❌ Pipeline 1 module not found: {e}")
        return None
    
    try:
        from run_pipeline2_data_pipeline import run_pipeline2_complete
        pipeline_modules['pipeline2'] = run_pipeline2_complete
        logger.info("✅ Pipeline 2 (Data Pipeline) module loaded")
    except ImportError as e:
        logger.error(f"❌ Pipeline 2 module not found: {e}")
        return None
    
    try:
        from run_pipeline3_model_architecture import run_pipeline3_complete
        pipeline_modules['pipeline3'] = run_pipeline3_complete
        logger.info("✅ Pipeline 3 (Model Architecture) module loaded")
    except ImportError as e:
        logger.error(f"❌ Pipeline 3 module not found: {e}")
        return None
    
    try:
        from run_pipeline4_training_infrastructure import run_pipeline4_complete
        pipeline_modules['pipeline4'] = run_pipeline4_complete
        logger.info("✅ Pipeline 4 (Training Infrastructure) module loaded")
    except ImportError as e:
        logger.error(f"❌ Pipeline 4 module not found: {e}")
        return None
    
    try:
        from run_pipeline5_training_optimization import run_pipeline5_complete
        pipeline_modules['pipeline5'] = run_pipeline5_complete
        logger.info("✅ Pipeline 5 (Training & Optimization) module loaded")
    except ImportError as e:
        logger.error(f"❌ Pipeline 5 module not found: {e}")
        return None
    
    try:
        from run_pipeline6_evaluation_analysis import run_pipeline6_complete
        pipeline_modules['pipeline6'] = run_pipeline6_complete
        logger.info("✅ Pipeline 6 (Evaluation & Analysis) module loaded")
    except ImportError as e:
        logger.error(f"❌ Pipeline 6 module not found: {e}")
        return None
    
    return pipeline_modules

def run_single_pipeline(pipeline_name, pipeline_modules, experiment_name=None, mode="full"):
    """Run a single pipeline"""
    logger = logging.getLogger('MainPipeline')
    
    logger.info("="*80)
    logger.info(f"🚀 RUNNING {pipeline_name.upper()}")
    logger.info("="*80)
    
    start_time = datetime.now()
    
    try:
        if pipeline_name == 'pipeline1':
            # Data Processing
            results = pipeline_modules[pipeline_name](mode=mode, log_level="INFO")
        elif pipeline_name == 'pipeline2':
            # Data Pipeline
            results = pipeline_modules[pipeline_name](mode=mode, log_level="INFO")
        elif pipeline_name == 'pipeline3':
            # Model Architecture
            results = pipeline_modules[pipeline_name](mode=mode, log_level="INFO")
        elif pipeline_name == 'pipeline4':
            # Training Infrastructure
            results = pipeline_modules[pipeline_name](
                experiment_name=experiment_name,
                mode=mode,
                log_level="INFO"
            )
        elif pipeline_name == 'pipeline5':
            # Training & Optimization
            results = pipeline_modules[pipeline_name](
                experiment_name=experiment_name,
                mode=mode,
                log_level="INFO"
            )
        elif pipeline_name == 'pipeline6':
            # Evaluation & Analysis
            results = pipeline_modules[pipeline_name](log_level="INFO")
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if results.get('status') == 'completed':
            logger.info(f"✅ {pipeline_name.upper()} completed successfully in {duration}")
            return results
        else:
            logger.error(f"❌ {pipeline_name.upper()} failed: {results.get('error', 'Unknown error')}")
            return results
            
    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.error(f"❌ {pipeline_name.upper()} failed after {duration}: {e}")
        return {'status': 'failed', 'error': str(e), 'duration': str(duration)}

def run_complete_pipeline(experiment_name=None, mode="full", skip_phases=None):
    """Run the complete 6-phase pipeline"""
    logger = setup_logging()
    
    logger.info("🌟 DIABETIC RETINOPATHY DETECTION - COMPLETE PIPELINE")
    logger.info("="*80)
    logger.info("Starting comprehensive model training workflow from data to deployment")
    logger.info("="*80)
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("❌ System requirements not met. Please address the issues above.")
        return {'status': 'failed', 'error': 'System requirements not met'}
    
    # Import pipeline modules
    pipeline_modules = import_pipeline_modules()
    if pipeline_modules is None:
        logger.error("❌ Failed to load pipeline modules.")
        return {'status': 'failed', 'error': 'Failed to load pipeline modules'}
    
    # Initialize tracking
    if experiment_name is None:
        experiment_name = f"complete_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    skip_phases = skip_phases or []
    
    pipeline_execution = {
        'experiment_name': experiment_name,
        'mode': mode,
        'start_time': datetime.now().isoformat(),
        'phases': {},
        'overall_status': 'running',
        'phases_completed': [],
        'phases_failed': [],
        'phases_skipped': skip_phases
    }
    
    logger.info(f"Experiment Name: {experiment_name}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Phases to Skip: {skip_phases if skip_phases else 'None'}")
    
    # Define pipeline phases
    phases = [
        ('pipeline1', 'Data Processing', 'Downloads and processes raw datasets'),
        ('pipeline2', 'Data Pipeline', 'Sets up data augmentation and loaders'),
        ('pipeline3', 'Model Architecture', 'Builds and tests model architecture'),
        ('pipeline4', 'Training Infrastructure', 'Sets up training infrastructure'),
        ('pipeline5', 'Training & Optimization', 'Trains model with hyperparameter optimization'),
        ('pipeline6', 'Evaluation & Analysis', 'Comprehensive evaluation and deployment preparation')
    ]
    
    try:
        # Run each phase
        for phase_key, phase_name, phase_description in phases:
            if phase_key in skip_phases:
                logger.info(f"⏭️ Skipping {phase_name}")
                pipeline_execution['phases'][phase_key] = {
                    'status': 'skipped',
                    'name': phase_name,
                    'description': phase_description
                }
                continue
            
            logger.info(f"\n🔄 PHASE: {phase_name}")
            logger.info(f"📝 Description: {phase_description}")
            
            # Run the phase
            phase_results = run_single_pipeline(
                phase_key, 
                pipeline_modules, 
                experiment_name=experiment_name,
                mode=mode
            )
            
            # Record results
            pipeline_execution['phases'][phase_key] = {
                'name': phase_name,
                'description': phase_description,
                'results': phase_results,
                'status': phase_results.get('status', 'unknown')
            }
            
            if phase_results.get('status') == 'completed':
                pipeline_execution['phases_completed'].append(phase_key)
                logger.info(f"✅ Phase {phase_name} completed successfully")
                
                # Show key metrics if available
                show_phase_metrics(phase_key, phase_results, logger)
                
            else:
                pipeline_execution['phases_failed'].append(phase_key)
                logger.error(f"❌ Phase {phase_name} failed")
                
                # For critical phases, decide whether to continue
                if phase_key in ['pipeline1', 'pipeline2']:  # Critical early phases
                    logger.error("❌ Critical phase failed. Stopping pipeline.")
                    pipeline_execution['overall_status'] = 'failed'
                    break
                else:
                    logger.warning("⚠️ Phase failed but continuing with next phase...")
            
            # Add delay between phases for system stability
            if phase_key != 'pipeline6':  # No delay after last phase
                logger.info("⏳ Pausing before next phase...")
                time.sleep(5)
        
        # Determine overall status
        if len(pipeline_execution['phases_failed']) == 0:
            pipeline_execution['overall_status'] = 'completed'
        elif len(pipeline_execution['phases_completed']) >= 4:  # Most phases completed
            pipeline_execution['overall_status'] = 'partial_success'
        else:
            pipeline_execution['overall_status'] = 'failed'
        
        pipeline_execution['end_time'] = datetime.now().isoformat()
        
        # Generate final report
        generate_final_report(pipeline_execution, logger)
        
        return pipeline_execution
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        pipeline_execution['overall_status'] = 'failed'
        pipeline_execution['error'] = str(e)
        pipeline_execution['end_time'] = datetime.now().isoformat()
        return pipeline_execution

def show_phase_metrics(phase_key, phase_results, logger):
    """Show key metrics for each completed phase"""
    
    if phase_key == 'pipeline1':
        # Data processing metrics
        results = phase_results.get('data_processing_summary', {})
        if results:
            logger.info(f"  📊 Datasets processed: {len(results.get('datasets_processed', []))}")
            logger.info(f"  📁 Total images: {results.get('total_images_processed', 'N/A')}")
    
    elif phase_key == 'pipeline2':
        # Data pipeline metrics
        results = phase_results.get('data_pipeline_summary', {})
        if results:
            logger.info(f"  🔄 Augmentations created: {results.get('augmentations_count', 'N/A')}")
            logger.info(f"  📦 Datasets created: {results.get('datasets_created', 'N/A')}")
    
    elif phase_key == 'pipeline3':
        # Model architecture metrics
        results = phase_results.get('model_architecture_summary', {})
        if results:
            logger.info(f"  🏗️ Backbone: {results.get('backbone_type', 'N/A')}")
            logger.info(f"  📐 Parameters: {results.get('total_parameters', 'N/A')}")
    
    elif phase_key == 'pipeline4':
        # Training infrastructure metrics
        results = phase_results.get('training_infrastructure_summary', {})
        if results:
            logger.info(f"  ⚙️ Training phases: {results.get('training_phases', 'N/A')}")
            logger.info(f"  📈 Loss functions: {results.get('loss_functions_count', 'N/A')}")
    
    elif phase_key == 'pipeline5':
        # Training results
        final_training = phase_results.get('final_model_training', {})
        if final_training and final_training.get('final_metrics'):
            metrics = final_training['final_metrics']
            logger.info(f"  🎯 Final Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
            logger.info(f"  🎯 Final Dice Score: {metrics.get('dice_score', 'N/A'):.3f}")
            logger.info(f"  🎯 Combined Score: {metrics.get('combined_score', 'N/A'):.3f}")
    
    elif phase_key == 'pipeline6':
        # Evaluation results
        clinical_report = phase_results.get('clinical_report', {})
        if clinical_report:
            logger.info(f"  📋 Clinical Grade: {clinical_report.get('overall_clinical_grade', 'N/A')}")
            logger.info(f"  🚀 Deployment Status: {clinical_report.get('deployment_recommendation', 'N/A')}")

def generate_final_report(pipeline_execution, logger):
    """Generate comprehensive final report"""
    logger.info("\n" + "="*80)
    logger.info("📋 FINAL PIPELINE REPORT")
    logger.info("="*80)
    
    # Overall status
    status = pipeline_execution['overall_status']
    if status == 'completed':
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    elif status == 'partial_success':
        logger.info("⚠️ PIPELINE PARTIALLY COMPLETED")
    else:
        logger.info("❌ PIPELINE FAILED")
    
    # Execution summary
    start_time = datetime.fromisoformat(pipeline_execution['start_time'])
    end_time = datetime.fromisoformat(pipeline_execution['end_time'])
    duration = end_time - start_time
    
    logger.info(f"\n📊 EXECUTION SUMMARY:")
    logger.info(f"  Experiment: {pipeline_execution['experiment_name']}")
    logger.info(f"  Duration: {duration}")
    logger.info(f"  Phases Completed: {len(pipeline_execution['phases_completed'])}/6")
    logger.info(f"  Phases Failed: {len(pipeline_execution['phases_failed'])}")
    logger.info(f"  Phases Skipped: {len(pipeline_execution['phases_skipped'])}")
    
    # Phase status
    logger.info(f"\n🔍 PHASE STATUS:")
    for phase_key, phase_data in pipeline_execution['phases'].items():
        phase_name = phase_data['name']
        phase_status = phase_data['status']
        
        if phase_status == 'completed':
            logger.info(f"  ✅ {phase_name}")
        elif phase_status == 'failed':
            logger.info(f"  ❌ {phase_name}")
        elif phase_status == 'skipped':
            logger.info(f"  ⏭️ {phase_name}")
        else:
            logger.info(f"  ❓ {phase_name} ({phase_status})")
    
    # Final model performance (if available)
    pipeline5_results = pipeline_execution['phases'].get('pipeline5', {}).get('results', {})
    if pipeline5_results.get('final_model_training'):
        final_metrics = pipeline5_results['final_model_training'].get('final_metrics', {})
        if final_metrics:
            logger.info(f"\n🎯 FINAL MODEL PERFORMANCE:")
            logger.info(f"  Classification Accuracy: {final_metrics.get('accuracy', 'N/A'):.3f}")
            logger.info(f"  Segmentation Dice Score: {final_metrics.get('dice_score', 'N/A'):.3f}")
            logger.info(f"  Combined Score: {final_metrics.get('combined_score', 'N/A'):.3f}")
    
    # Clinical assessment (if available)
    pipeline6_results = pipeline_execution['phases'].get('pipeline6', {}).get('results', {})
    if pipeline6_results.get('clinical_report'):
        clinical_report = pipeline6_results['clinical_report']
        logger.info(f"\n🏥 CLINICAL ASSESSMENT:")
        logger.info(f"  Clinical Grade: {clinical_report.get('overall_clinical_grade', 'N/A')}")
        logger.info(f"  Deployment Readiness: {clinical_report.get('deployment_recommendation', 'N/A')}")
    
    # Deployment package (if available)
    if pipeline6_results.get('deployment_package'):
        deployment = pipeline6_results['deployment_package']
        logger.info(f"\n📦 DEPLOYMENT PACKAGE:")
        logger.info(f"  Location: {deployment.get('package_location', 'N/A')}")
        logger.info(f"  Size: {deployment.get('package_size_mb', 'N/A'):.1f} MB")
    
    # Save final report
    results_dir = Path("results/complete_pipeline")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f"complete_pipeline_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(pipeline_execution, f, indent=2, default=str)
    
    logger.info(f"\n📄 Complete report saved to: {report_file}")
    
    # Next steps
    if status == 'completed':
        logger.info(f"\n🚀 NEXT STEPS:")
        logger.info(f"  1. Review deployment package in results/")
        logger.info(f"  2. Consider clinical validation studies")
        logger.info(f"  3. Prepare for regulatory submission")
        logger.info(f"  4. Set up production deployment")
    elif status == 'partial_success':
        logger.info(f"\n🔧 RECOMMENDED ACTIONS:")
        logger.info(f"  1. Review failed phases and error logs")
        logger.info(f"  2. Re-run specific phases if needed")
        logger.info(f"  3. Check system resources and dependencies")
    else:
        logger.info(f"\n🛠️ TROUBLESHOOTING:")
        logger.info(f"  1. Check error logs in logs/ directory")
        logger.info(f"  2. Verify system requirements")
        logger.info(f"  3. Check data availability and permissions")
        logger.info(f"  4. Consider running phases individually")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Main Pipeline Orchestrator for Diabetic Retinopathy Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --mode full

  # Run quick test mode
  python main.py --mode quick

  # Run specific phase only
  python main.py --phase pipeline5 --experiment-name my_training

  # Run pipeline skipping data processing
  python main.py --skip-phases pipeline1

  # Run pipeline with custom experiment name
  python main.py --experiment-name production_model_v1
        """
    )
    
    parser.add_argument('--experiment-name', type=str,
                       help='Name for this experiment (auto-generated if not provided)')
    parser.add_argument('--mode', type=str, choices=['full', 'quick'],
                       default='full', help='Pipeline execution mode')
    parser.add_argument('--phase', type=str, 
                       choices=['pipeline1', 'pipeline2', 'pipeline3', 'pipeline4', 'pipeline5', 'pipeline6'],
                       help='Run only a specific phase')
    parser.add_argument('--skip-phases', type=str, nargs='+',
                       choices=['pipeline1', 'pipeline2', 'pipeline3', 'pipeline4', 'pipeline5', 'pipeline6'],
                       help='Phases to skip in complete pipeline')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # If specific phase requested, run only that phase
    if args.phase:
        logger = setup_logging(args.log_level)
        
        # Import pipeline modules
        pipeline_modules = import_pipeline_modules()
        if pipeline_modules is None:
            print("❌ Failed to load pipeline modules.")
            sys.exit(1)
        
        # Run specific phase
        results = run_single_pipeline(
            args.phase, 
            pipeline_modules, 
            experiment_name=args.experiment_name,
            mode=args.mode
        )
        
        if results.get('status') == 'completed':
            print(f"\n🎉 {args.phase.upper()} completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ {args.phase.upper()} failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
    
    # Run complete pipeline
    else:
        results = run_complete_pipeline(
            experiment_name=args.experiment_name,
            mode=args.mode,
            skip_phases=args.skip_phases
        )
        
        if results['overall_status'] == 'completed':
            print("\n🎉 Complete pipeline finished successfully!")
            print("Check the deployment package in results/ directory")
            sys.exit(0)
        elif results['overall_status'] == 'partial_success':
            print("\n⚠️ Pipeline partially completed. Check logs for details.")
            sys.exit(2)
        else:
            print(f"\n❌ Pipeline failed: {results.get('error', 'Multiple phase failures')}")
            print("Check logs for detailed error information.")
            sys.exit(1)

if __name__ == "__main__":
    main()
