#!/usr/bin/env python3
"""
Simple Phase 5 Training Monitor
Monitor training progress without running the full pipeline
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse
import time

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('Phase5Monitor')

def find_recent_experiments(experiments_dir: Path, max_age_hours: int = 24) -> List[Path]:
    """Find recent experiment directories"""
    if not experiments_dir.exists():
        return []
    
    recent_experiments = []
    cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
    
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and 'phase5' in exp_dir.name:
            # Check if directory was created recently
            if exp_dir.stat().st_ctime > cutoff_time:
                recent_experiments.append(exp_dir)
    
    # Sort by creation time (newest first)
    recent_experiments.sort(key=lambda p: p.stat().st_ctime, reverse=True)
    return recent_experiments

def check_experiment_status(exp_dir: Path) -> Dict:
    """Check the status of a single experiment"""
    status = {
        'experiment_id': exp_dir.name,
        'path': str(exp_dir),
        'status': 'unknown',
        'created': datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat(),
        'last_modified': datetime.fromtimestamp(exp_dir.stat().st_mtime).isoformat(),
        'files': [],
        'pipeline_results': None,
        'summary': None
    }
    
    # Check for key files
    key_files = [
        'phase5_pipeline.log',
        'phase5_pipeline_results.json',
        'PHASE5_SUMMARY_REPORT.md',
        'experimental_plan.json'
    ]
    
    for filename in key_files:
        file_path = exp_dir / filename
        if file_path.exists():
            status['files'].append({
                'name': filename,
                'size': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    # Load pipeline results if available
    results_file = exp_dir / 'phase5_pipeline_results.json'
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                pipeline_results = json.load(f)
            status['pipeline_results'] = pipeline_results
            status['status'] = pipeline_results.get('status', 'unknown')
        except Exception as e:
            status['pipeline_results'] = {'error': str(e)}
    
    # Check for log file to determine if running
    log_file = exp_dir / 'phase5_pipeline.log'
    if log_file.exists():
        # Check if log was recently modified (within last 5 minutes)
        log_age = time.time() - log_file.stat().st_mtime
        if log_age < 300:  # 5 minutes
            status['status'] = 'running'
    
    return status

def monitor_experiments(experiments_dir: Path = None, watch: bool = False, interval: int = 30):
    """Monitor Phase 5 experiments"""
    logger = setup_logging()
    
    if experiments_dir is None:
        experiments_dir = Path("experiments")
    
    logger.info(f"Monitoring experiments in: {experiments_dir}")
    
    while True:
        try:
            # Find recent experiments
            recent_experiments = find_recent_experiments(experiments_dir)
            
            if not recent_experiments:
                logger.info("No recent Phase 5 experiments found")
                if not watch:
                    break
                time.sleep(interval)
                continue
            
            # Check status of each experiment
            print(f"\n{'='*80}")
            print(f"PHASE 5 EXPERIMENT MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            for exp_dir in recent_experiments:
                status = check_experiment_status(exp_dir)
                
                print(f"\nExperiment: {status['experiment_id']}")
                print(f"Status: {status['status'].upper()}")
                print(f"Created: {status['created']}")
                print(f"Last Modified: {status['last_modified']}")
                
                # Show pipeline results summary
                if status['pipeline_results']:
                    results = status['pipeline_results']
                    print(f"Phases Completed: {', '.join(results.get('phases_completed', []))}")
                    
                    experiments_run = results.get('experiments_run', [])
                    if experiments_run:
                        successful = len([e for e in experiments_run if e.get('status') != 'failed'])
                        print(f"Experiments: {successful}/{len(experiments_run)} successful")
                    
                    if results.get('error'):
                        print(f"Error: {results['error']}")
                
                # Show files
                if status['files']:
                    print("Files:")
                    for file_info in status['files']:
                        size_kb = file_info['size'] // 1024
                        print(f"  - {file_info['name']}: {size_kb}KB")
                
                print(f"Path: {status['path']}")
            
            if not watch:
                break
            
            print(f"\n{'='*80}")
            print(f"Refreshing in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
            if not watch:
                break
            time.sleep(interval)

def show_experiment_details(exp_dir: Path):
    """Show detailed information about a specific experiment"""
    logger = setup_logging()
    
    if not exp_dir.exists():
        logger.error(f"Experiment directory not found: {exp_dir}")
        return
    
    status = check_experiment_status(exp_dir)
    
    print(f"\n{'='*80}")
    print(f"PHASE 5 EXPERIMENT DETAILS")
    print(f"{'='*80}")
    print(f"Experiment ID: {status['experiment_id']}")
    print(f"Status: {status['status'].upper()}")
    print(f"Created: {status['created']}")
    print(f"Last Modified: {status['last_modified']}")
    print(f"Path: {status['path']}")
    
    # Show detailed pipeline results
    if status['pipeline_results']:
        results = status['pipeline_results']
        print(f"\nPipeline Results:")
        print(f"  Pipeline ID: {results.get('pipeline_id', 'N/A')}")
        print(f"  Start Time: {results.get('start_time', 'N/A')}")
        print(f"  End Time: {results.get('end_time', 'N/A')}")
        print(f"  Status: {results.get('status', 'N/A')}")
        
        phases = results.get('phases_completed', [])
        print(f"  Phases Completed ({len(phases)}):")
        for phase in phases:
            print(f"    ✅ {phase.replace('_', ' ').title()}")
        
        experiments_run = results.get('experiments_run', [])
        if experiments_run:
            successful = len([e for e in experiments_run if e.get('status') != 'failed'])
            failed = len(experiments_run) - successful
            print(f"  Experiments: {len(experiments_run)} total, {successful} successful, {failed} failed")
        
        validation_results = results.get('validation_results', {})
        if validation_results:
            passed = len([v for v in validation_results.values() if v.get('passed_validation', False)])
            print(f"  Validation: {passed}/{len(validation_results)} models passed")
        
        if results.get('error'):
            print(f"  Error: {results['error']}")
    
    # Show log file tail
    log_file = exp_dir / 'phase5_pipeline.log'
    if log_file.exists():
        print(f"\nRecent Log Entries (last 10 lines):")
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"  Error reading log: {e}")
    
    # Show summary report if available
    summary_file = exp_dir / 'PHASE5_SUMMARY_REPORT.md'
    if summary_file.exists():
        print(f"\nSummary Report: {summary_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Phase 5 Training Monitor")
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                       help='Experiments directory path')
    parser.add_argument('--watch', action='store_true',
                       help='Continuously monitor experiments')
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds (for watch mode)')
    parser.add_argument('--experiment', type=str,
                       help='Show details for specific experiment')
    parser.add_argument('--max-age-hours', type=int, default=24,
                       help='Maximum age of experiments to show (hours)')
    
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    
    if args.experiment:
        # Show details for specific experiment
        exp_dir = experiments_dir / args.experiment
        show_experiment_details(exp_dir)
    else:
        # Monitor experiments
        monitor_experiments(
            experiments_dir=experiments_dir,
            watch=args.watch,
            interval=args.interval
        )

if __name__ == "__main__":
    main()
