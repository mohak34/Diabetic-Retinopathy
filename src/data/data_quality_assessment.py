"""
Phase 2.3: Comprehensive Data Quality Assessment & Validation
Ensures data integrity, class balance, and reproducibility before model training.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import cv2
from PIL import Image
import warnings
from typing import Dict, List, Tuple, Any
import hashlib

class DataQualityAssessment:
    """Comprehensive data quality assessment and validation."""
    
    def __init__(self, project_root: Path = None):
        """Initialize assessment with project paths."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = project_root
        self.dataset_dir = project_root / "dataset"
        self.processed_dir = self.dataset_dir / "processed"
        self.splits_dir = self.dataset_dir / "splits"
        self.results_dir = project_root / "results"
        self.notebooks_dir = project_root / "notebooks"
        
        # Ensure results directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        
        self.quality_report = {}
        
    def load_all_splits(self) -> Dict[str, Dict]:
        """Load all split files and return organized split data."""
        print("Loading all split files...")
        
        splits_data = {}
        
        # Load APTOS splits
        aptos_file = self.splits_dir / "aptos2019_splits.json"
        if aptos_file.exists():
            with open(aptos_file, 'r') as f:
                splits_data['aptos2019'] = json.load(f)
        else:
            print(f"Warning: {aptos_file} not found")
            
        # Load IDRiD grading splits
        grading_file = self.splits_dir / "idrid_grading_splits.json"
        if grading_file.exists():
            with open(grading_file, 'r') as f:
                splits_data['idrid_grading'] = json.load(f)
        else:
            print(f"Warning: {grading_file} not found")
            
        # Load IDRiD segmentation splits
        seg_file = self.splits_dir / "idrid_segmentation_splits.json"
        if seg_file.exists():
            with open(seg_file, 'r') as f:
                splits_data['idrid_segmentation'] = json.load(f)
        else:
            print(f"Warning: {seg_file} not found")
            
        return splits_data
    
    def verify_class_balance(self, splits_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Verify class balance across all splits."""
        print("\nVerifying class balance across splits...")
        
        balance_report = {}
        
        for dataset_name, split_info in splits_data.items():
            print(f"\nAnalyzing {dataset_name}...")
            
            if 'label_distribution' in split_info:
                # Use pre-calculated distributions
                train_dist = split_info['label_distribution']['train']
                val_dist = split_info['label_distribution']['val']
                
                balance_report[dataset_name] = {
                    'train_distribution': train_dist,
                    'val_distribution': val_dist,
                    'train_total': sum(train_dist.values()),
                    'val_total': sum(val_dist.values())
                }
                
                # Calculate balance ratios
                total_dist = {}
                for label in set(list(train_dist.keys()) + list(val_dist.keys())):
                    total_dist[label] = train_dist.get(label, 0) + val_dist.get(label, 0)
                
                balance_report[dataset_name]['total_distribution'] = total_dist
                
                print(f"  Train samples: {sum(train_dist.values())}")
                print(f"  Val samples: {sum(val_dist.values())}")
                print(f"  Class distribution (train): {train_dist}")
                print(f"  Class distribution (val): {val_dist}")
                
            elif 'splits' in split_info:
                # Segmentation data - count pairs
                train_count = len(split_info['splits']['train'])
                val_count = len(split_info['splits']['val'])
                
                balance_report[dataset_name] = {
                    'train_count': train_count,
                    'val_count': val_count,
                    'total_count': train_count + val_count,
                    'train_ratio': train_count / (train_count + val_count),
                    'val_ratio': val_count / (train_count + val_count)
                }
                
                print(f"  Train pairs: {train_count}")
                print(f"  Val pairs: {val_count}")
                print(f"  Train ratio: {balance_report[dataset_name]['train_ratio']:.3f}")
                
        return balance_report
    
    def check_data_integrity(self, splits_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Run comprehensive data integrity checks."""
        print("\nRunning data integrity checks...")
        
        integrity_report = {
            'missing_files': [],
            'corrupted_images': [],
            'mismatched_pairs': [],
            'file_size_anomalies': [],
            'readable_checks': {}
        }
        
        for dataset_name, split_info in splits_data.items():
            print(f"\nChecking integrity for {dataset_name}...")
            
            dataset_issues = {
                'missing_files': [],
                'corrupted_images': [],
                'unreadable_images': []
            }
            
            if dataset_name == 'aptos2019':
                images_dir = self.processed_dir / "aptos2019" / "images"
                image_ids = split_info['splits']['train'] + split_info['splits']['val']
                
                for img_id in image_ids:
                    img_path = images_dir / f"{img_id}.jpg"
                    if not img_path.exists():
                        dataset_issues['missing_files'].append(str(img_path))
                    else:
                        # Check if image is readable
                        try:
                            img = cv2.imread(str(img_path))
                            if img is None:
                                dataset_issues['corrupted_images'].append(str(img_path))
                        except Exception as e:
                            dataset_issues['unreadable_images'].append(str(img_path))
                            
            elif dataset_name == 'idrid_grading':
                images_dir = self.processed_dir / "grading" / "images"
                image_ids = split_info['splits']['train'] + split_info['splits']['val']
                
                for img_id in image_ids:
                    img_path = images_dir / f"{img_id}.jpg"
                    if not img_path.exists():
                        dataset_issues['missing_files'].append(str(img_path))
                    else:
                        try:
                            img = cv2.imread(str(img_path))
                            if img is None:
                                dataset_issues['corrupted_images'].append(str(img_path))
                        except Exception:
                            dataset_issues['unreadable_images'].append(str(img_path))
                            
            elif dataset_name == 'idrid_segmentation':
                images_dir = self.processed_dir / "segmentation" / "images"
                masks_dir = self.processed_dir / "segmentation" / "masks"
                pair_ids = split_info['splits']['train'] + split_info['splits']['val']
                
                for pair_id in pair_ids:
                    img_path = images_dir / f"{pair_id}.jpg"
                    mask_path = masks_dir / f"{pair_id}.tif"
                    
                    if not img_path.exists():
                        dataset_issues['missing_files'].append(str(img_path))
                    if not mask_path.exists():
                        dataset_issues['missing_files'].append(str(mask_path))
                        
                    # Check readability
                    if img_path.exists():
                        try:
                            img = cv2.imread(str(img_path))
                            if img is None:
                                dataset_issues['corrupted_images'].append(str(img_path))
                        except Exception:
                            dataset_issues['unreadable_images'].append(str(img_path))
                            
                    if mask_path.exists():
                        try:
                            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                            if mask is None:
                                dataset_issues['corrupted_images'].append(str(mask_path))
                        except Exception:
                            dataset_issues['unreadable_images'].append(str(mask_path))
            
            integrity_report['readable_checks'][dataset_name] = dataset_issues
            
            # Summary for this dataset
            total_issues = (len(dataset_issues['missing_files']) + 
                          len(dataset_issues['corrupted_images']) + 
                          len(dataset_issues['unreadable_images']))
            
            print(f"  Missing files: {len(dataset_issues['missing_files'])}")
            print(f"  Corrupted images: {len(dataset_issues['corrupted_images'])}")
            print(f"  Unreadable images: {len(dataset_issues['unreadable_images'])}")
            print(f"  Total issues: {total_issues}")
            
        return integrity_report
    
    def detect_data_leakage(self, splits_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Detect potential data leakage between train/val splits and across datasets."""
        print("\nDetecting potential data leakage...")
        
        leakage_report = {
            'within_dataset_leakage': {},
            'cross_dataset_overlaps': {},
            'potential_issues': []
        }
        
        # Check within-dataset leakage (train/val overlap)
        for dataset_name, split_info in splits_data.items():
            if 'splits' in split_info:
                train_ids = set(split_info['splits']['train'])
                val_ids = set(split_info['splits']['val'])
                
                overlap = train_ids.intersection(val_ids)
                leakage_report['within_dataset_leakage'][dataset_name] = {
                    'train_count': len(train_ids),
                    'val_count': len(val_ids),
                    'overlap_count': len(overlap),
                    'overlapping_ids': list(overlap)
                }
                
                if overlap:
                    issue = f"{dataset_name}: {len(overlap)} samples overlap between train/val"
                    leakage_report['potential_issues'].append(issue)
                    print(f"  WARNING: {issue}")
                else:
                    print(f"  {dataset_name}: No train/val overlap detected")
        
        # Check cross-dataset overlaps (same image in multiple datasets)
        all_images = {}
        for dataset_name, split_info in splits_data.items():
            if 'splits' in split_info:
                all_ids = split_info['splits']['train'] + split_info['splits']['val']
                for img_id in all_ids:
                    if img_id not in all_images:
                        all_images[img_id] = []
                    all_images[img_id].append(dataset_name)
        
        cross_overlaps = {img_id: datasets for img_id, datasets in all_images.items() 
                         if len(datasets) > 1}
        
        leakage_report['cross_dataset_overlaps'] = cross_overlaps
        
        if cross_overlaps:
            print(f"  Found {len(cross_overlaps)} images appearing in multiple datasets")
            for img_id, datasets in list(cross_overlaps.items())[:5]:  # Show first 5
                print(f"    {img_id}: {datasets}")
        else:
            print("  No cross-dataset overlaps detected")
            
        return leakage_report
    
    def create_immutable_split_documentation(self, splits_data: Dict[str, Dict]) -> None:
        """Create immutable CSV files documenting exact splits for reproducibility."""
        print("\nCreating immutable split documentation...")
        
        splits_export_dir = self.results_dir / "immutable_splits"
        splits_export_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, split_info in splits_data.items():
            if 'splits' in split_info:
                # Create train split CSV
                train_df = pd.DataFrame({
                    'image_id': split_info['splits']['train'],
                    'split': 'train',
                    'dataset': dataset_name
                })
                
                # Create val split CSV
                val_df = pd.DataFrame({
                    'image_id': split_info['splits']['val'],
                    'split': 'val',
                    'dataset': dataset_name
                })
                
                # Combine and save
                combined_df = pd.concat([train_df, val_df], ignore_index=True)
                
                # Add labels if available
                if 'label_distribution' in split_info:
                    # Create label mapping from distribution
                    label_map = {}
                    for split_type in ['train', 'val']:
                        for label, count in split_info['label_distribution'][split_type].items():
                            # This is simplified - in practice, you'd need actual label mapping
                            pass
                
                csv_file = splits_export_dir / f"{dataset_name}_splits.csv"
                combined_df.to_csv(csv_file, index=False)
                
                print(f"  Exported {dataset_name} splits to {csv_file}")
                
                # Create SHA256 hash for integrity verification
                with open(csv_file, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                hash_file = splits_export_dir / f"{dataset_name}_splits.sha256"
                with open(hash_file, 'w') as f:
                    f.write(f"{file_hash}  {csv_file.name}\n")
    
    def generate_quality_visualizations(self, splits_data: Dict[str, Dict], 
                                      balance_report: Dict[str, Any]) -> None:
        """Generate comprehensive quality assessment visualizations."""
        print("\nGenerating quality assessment visualizations...")
        
        viz_dir = self.results_dir / "quality_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for publication-quality plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Class distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Class Distribution Analysis Across Datasets', fontsize=16, fontweight='bold')
        
        # APTOS distribution
        if 'aptos2019' in balance_report:
            ax = axes[0, 0]
            data = balance_report['aptos2019']
            
            labels = list(data['total_distribution'].keys())
            values = list(data['total_distribution'].values())
            
            bars = ax.bar(labels, values, alpha=0.8)
            ax.set_title('APTOS 2019 - DR Grade Distribution')
            ax.set_xlabel('DR Grade')
            ax.set_ylabel('Number of Images')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       str(value), ha='center', va='bottom')
        
        # IDRiD Grading distribution
        if 'idrid_grading' in balance_report:
            ax = axes[0, 1]
            data = balance_report['idrid_grading']
            
            labels = list(data['total_distribution'].keys())
            values = list(data['total_distribution'].values())
            
            bars = ax.bar(labels, values, alpha=0.8, color='orange')
            ax.set_title('IDRiD Grading - DR Grade Distribution')
            ax.set_xlabel('DR Grade')
            ax.set_ylabel('Number of Images')
            
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       str(value), ha='center', va='bottom')
        
        # Split ratios
        ax = axes[1, 0]
        datasets = []
        train_ratios = []
        val_ratios = []
        
        for dataset_name, data in balance_report.items():
            if 'train_ratio' in data:
                datasets.append(dataset_name)
                train_ratios.append(data['train_ratio'])
                val_ratios.append(data['val_ratio'])
            elif 'train_total' in data and 'val_total' in data:
                total = data['train_total'] + data['val_total']
                datasets.append(dataset_name)
                train_ratios.append(data['train_total'] / total)
                val_ratios.append(data['val_total'] / total)
        
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, train_ratios, width, label='Train', alpha=0.8)
        ax.bar(x + width/2, val_ratios, width, label='Validation', alpha=0.8)
        
        ax.set_title('Train/Validation Split Ratios')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Dataset sizes comparison
        ax = axes[1, 1]
        datasets = []
        sizes = []
        
        for dataset_name, data in balance_report.items():
            if 'total_count' in data:
                datasets.append(dataset_name)
                sizes.append(data['total_count'])
            elif 'train_total' in data and 'val_total' in data:
                datasets.append(dataset_name)
                sizes.append(data['train_total'] + data['val_total'])
        
        bars = ax.bar(datasets, sizes, alpha=0.8, color='green')
        ax.set_title('Dataset Sizes')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Total Images')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   str(size), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "class_distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved class distribution analysis to {viz_dir}")
    
    def run_complete_assessment(self) -> Dict[str, Any]:
        """Run the complete data quality assessment pipeline."""
        print("="*60)
        print("PHASE 2.3: COMPREHENSIVE DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Step 1: Load all splits
        splits_data = self.load_all_splits()
        
        # Step 2: Verify class balance
        balance_report = self.verify_class_balance(splits_data)
        
        # Step 3: Check data integrity
        integrity_report = self.check_data_integrity(splits_data)
        
        # Step 4: Detect data leakage
        leakage_report = self.detect_data_leakage(splits_data)
        
        # Step 5: Create immutable documentation
        self.create_immutable_split_documentation(splits_data)
        
        # Step 6: Generate visualizations
        self.generate_quality_visualizations(splits_data, balance_report)
        
        # Compile final report
        final_report = {
            'assessment_timestamp': pd.Timestamp.now().isoformat(),
            'datasets_analyzed': list(splits_data.keys()),
            'class_balance': balance_report,
            'data_integrity': integrity_report,
            'leakage_detection': leakage_report,
            'total_images': sum([
                sum(data.get('total_distribution', {}).values()) if 'total_distribution' in data
                else data.get('total_count', 0)
                for data in balance_report.values()
            ])
        }
        
        # Save comprehensive report
        report_file = self.results_dir / "data_quality_assessment_report.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\nQuality assessment complete!")
        print(f"Full report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("QUALITY ASSESSMENT SUMMARY")
        print("="*60)
        
        for dataset_name in splits_data.keys():
            print(f"\n{dataset_name.upper()}:")
            if dataset_name in balance_report:
                data = balance_report[dataset_name]
                if 'total_count' in data:
                    print(f"  Total samples: {data['total_count']}")
                    print(f"  Train/Val ratio: {data['train_ratio']:.3f}/{data['val_ratio']:.3f}")
                elif 'train_total' in data:
                    total = data['train_total'] + data['val_total']
                    print(f"  Total samples: {total}")
                    print(f"  Train: {data['train_total']}, Val: {data['val_total']}")
            
            if dataset_name in integrity_report['readable_checks']:
                issues = integrity_report['readable_checks'][dataset_name]
                total_issues = sum(len(v) for v in issues.values())
                print(f"  Data integrity issues: {total_issues}")
        
        leakage_issues = len(leakage_report['potential_issues'])
        print(f"\nData leakage issues: {leakage_issues}")
        
        if leakage_issues == 0 and all(
            sum(len(v) for v in issues.values()) == 0 
            for issues in integrity_report['readable_checks'].values()
        ):
            print("\nSTATUS: All quality checks PASSED - Data ready for training!")
        else:
            print("\nSTATUS: Quality issues detected - Review required before training")
        
        return final_report

def main():
    """Run the complete data quality assessment."""
    assessor = DataQualityAssessment()
    report = assessor.run_complete_assessment()
    return report

if __name__ == "__main__":
    main()
