"""
Data Organization Script for Diabetic Retinopathy Multi-Task Project
Organizes datasets according to the processing plan and validates data integrity.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from collections import defaultdict
import json

# Add project root to Python path for consistent imports
import sys
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

class DataOrganizer:
    """Organizes and validates diabetic retinopathy datasets"""
    
    def __init__(self, base_data_path: str):
        self.base_path = Path(base_data_path)
        self.processed_path = self.base_path / "processed"
        self.results = {}
        
    def organize_datasets(self) -> Dict:
        """Main function to organize all datasets"""
        print("="*80)
        print("DIABETIC RETINOPATHY DATASET ORGANIZATION")
        print("="*80)
        
        # Create processed directories
        self._create_processed_structure()
        
        # Process each dataset
        self.results['segmentation'] = self._process_segmentation_data()
        self.results['grading'] = self._process_grading_data()
        self.results['localization'] = self._process_localization_data()
        self.results['aptos2019'] = self._process_aptos_data()
        
        # Generate summary report
        self._generate_summary_report()
        
        return self.results
    
    def _create_processed_structure(self):
        """Create processed data directory structure"""
        subdirs = [
            "segmentation/images",
            "segmentation/masks",
            "segmentation/train",
            "segmentation/val",
            "grading/images", 
            "grading/train",
            "grading/val",
            "localization/images",
            "localization/boxes",
            "localization/generated_masks",
            "aptos2019/images",
            "aptos2019/train", 
            "aptos2019/val",
            "combined/train",
            "combined/val"
        ]
        
        for subdir in subdirs:
            (self.processed_path / subdir).mkdir(parents=True, exist_ok=True)
        
        print(f"OK Created processed data structure at: {self.processed_path}")
    
    def _process_segmentation_data(self) -> Dict:
        """Process A. Segmentation dataset"""
        print("\n Processing A. Segmentation Dataset...")
        
        seg_path = self.base_path / "A. Segmentation"
        images_path = seg_path / "1. Original Images"
        masks_path = seg_path / "2. All Segmentation Groundtruths"
        
        if not seg_path.exists():
            print("ERROR A. Segmentation folder not found!")
            return {'status': 'failed', 'reason': 'folder_not_found'}
        
        # Analyze dataset structure
        result = {
            'status': 'success',
            'images': self._analyze_images_folder(images_path),
            'masks': self._analyze_masks_folder(masks_path),
            'pairs': []
        }
        
        # Find image-mask pairs
        if images_path.exists() and masks_path.exists():
            result['pairs'] = self._find_image_mask_pairs(images_path, masks_path)
        
        print(f"OK Found {len(result.get('pairs', []))} image-mask pairs")
        print(f"OK Total images: {result['images']['count']}")
        print(f"OK Total masks: {result['masks']['count']}")
        
        return result
    
    def _process_grading_data(self) -> Dict:
        """Process B. Disease Grading dataset"""
        print("\nProcessing Processing B. Disease Grading Dataset...")
        
        grading_path = self.base_path / "B. Disease Grading"
        images_path = grading_path / "1. Original Images"
        labels_path = grading_path / "2. Groundtruths"
        
        if not grading_path.exists():
            print("ERROR B. Disease Grading folder not found!")
            return {'status': 'failed', 'reason': 'folder_not_found'}
        
        result = {
            'status': 'success',
            'images': self._analyze_images_folder(images_path),
            'labels': self._analyze_labels_folder(labels_path),
            'grade_distribution': {}
        }
        
        # Analyze grade distribution if labels exist
        if labels_path.exists():
            result['grade_distribution'] = self._analyze_grade_distribution(labels_path)
        
        print(f"OK Total images: {result['images']['count']}")
        print(f"OK Grade distribution: {result['grade_distribution']}")
        
        return result
    
    def _process_localization_data(self) -> Dict:
        """Process C. Localization dataset"""
        print("\n Processing C. Localization Dataset...")
        
        loc_path = self.base_path / "C. Localization"
        images_path = loc_path / "1. Original Images"
        boxes_path = loc_path / "2. Groundtruths"
        
        if not loc_path.exists():
            print("ERROR C. Localization folder not found!")
            return {'status': 'failed', 'reason': 'folder_not_found'}
        
        result = {
            'status': 'success',
            'images': self._analyze_images_folder(images_path),
            'annotations': self._analyze_boxes_folder(boxes_path),
            'bbox_pairs': []
        }
        
        # Find image-annotation pairs
        if images_path.exists() and boxes_path.exists():
            result['bbox_pairs'] = self._find_image_bbox_pairs(images_path, boxes_path)
        
        print(f"OK Found {len(result.get('bbox_pairs', []))} image-annotation pairs")
        print(f"OK Total images: {result['images']['count']}")
        
        return result
    
    def _process_aptos_data(self) -> Dict:
        """Process APTOS 2019 dataset"""
        print("\nProcessing Processing APTOS 2019 Dataset...")
        
        aptos_path = self.base_path / "aptos2019-blindness-detection"
        train_images_path = aptos_path / "train_images"
        test_images_path = aptos_path / "test_images"
        train_csv = aptos_path / "train.csv"
        
        if not aptos_path.exists():
            print("ERROR APTOS 2019 folder not found!")
            return {'status': 'failed', 'reason': 'folder_not_found'}
        
        result = {
            'status': 'success',
            'train_images': self._analyze_images_folder(train_images_path),
            'test_images': self._analyze_images_folder(test_images_path),
            'train_labels': {},
            'grade_distribution': {}
        }
        
        # Analyze train labels
        if train_csv.exists():
            df = pd.read_csv(train_csv)
            result['train_labels'] = {
                'count': len(df),
                'columns': list(df.columns),
                'sample': df.head().to_dict('records') if len(df) > 0 else []
            }
            
            # Grade distribution
            if 'diagnosis' in df.columns:
                grade_counts = df['diagnosis'].value_counts().to_dict()
                result['grade_distribution'] = grade_counts
        
        print(f"OK Train images: {result['train_images']['count']}")
        print(f"OK Test images: {result['test_images']['count']}")
        print(f"OK Train labels: {result['train_labels'].get('count', 0)}")
        print(f"OK Grade distribution: {result['grade_distribution']}")
        
        return result
    
    def _analyze_images_folder(self, folder_path: Path) -> Dict:
        """Analyze images in a folder"""
        if not folder_path.exists():
            return {'count': 0, 'extensions': {}, 'subfolders': []}
        
        extensions = defaultdict(int)
        subfolders = []
        total_count = 0
        
        image_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        for root, dirs, files in os.walk(folder_path):
            # Track subfolders
            if root != str(folder_path):
                rel_path = Path(root).relative_to(folder_path)
                subfolders.append(str(rel_path))
            
            # Count images
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in image_exts:
                    extensions[ext] += 1
                    total_count += 1
        
        return {
            'count': total_count,
            'extensions': dict(extensions),
            'subfolders': subfolders
        }
    
    def _analyze_masks_folder(self, folder_path: Path) -> Dict:
        """Analyze mask files"""
        return self._analyze_images_folder(folder_path)
    
    def _analyze_labels_folder(self, folder_path: Path) -> Dict:
        """Analyze label files"""
        if not folder_path.exists():
            return {'count': 0, 'files': [], 'formats': {}}
        
        formats = defaultdict(int)
        files = []
        
        for root, dirs, file_list in os.walk(folder_path):
            for file in file_list:
                ext = Path(file).suffix.lower()
                formats[ext] += 1
                files.append(file)
        
        return {
            'count': len(files),
            'files': files[:10],  # First 10 files as sample
            'formats': dict(formats)
        }
    
    def _analyze_boxes_folder(self, folder_path: Path) -> Dict:
        """Analyze bounding box annotation files"""
        return self._analyze_labels_folder(folder_path)
    
    def _find_image_mask_pairs(self, images_path: Path, masks_path: Path) -> List[Tuple[str, str]]:
        """Find matching image-mask pairs for IDRiD dataset"""
        pairs = []
        
        # Get all image files
        image_files = {}
        for root, dirs, files in os.walk(images_path):
            for file in files:
                if Path(file).suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}:
                    # Extract IDRiD image ID (e.g., IDRiD_73 from IDRiD_73.jpg)
                    base_name = Path(file).stem
                    full_path = Path(root) / file
                    image_files[base_name] = str(full_path)
        
        # Get all mask files and group by image ID
        mask_groups = defaultdict(list)
        for root, dirs, files in os.walk(masks_path):
            for file in files:
                if Path(file).suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}:
                    # Extract image ID from mask filename (e.g., IDRiD_73 from IDRiD_73_HE.tif)
                    stem = Path(file).stem
                    if '_' in stem:
                        # Split by underscore and take first parts (IDRiD_73 from IDRiD_73_HE)
                        parts = stem.split('_')
                        if len(parts) >= 2:
                            image_id = '_'.join(parts[:2])  # IDRiD_73
                            full_path = Path(root) / file
                            mask_groups[image_id].append(str(full_path))
        
        # Create pairs - for each image, create pairs with all its masks
        for image_id, image_path in image_files.items():
            if image_id in mask_groups:
                for mask_path in mask_groups[image_id]:
                    pairs.append((image_path, mask_path))
        
        return pairs
    
    def _find_image_bbox_pairs(self, images_path: Path, boxes_path: Path) -> List[Tuple[str, str]]:
        """Find matching image-bbox annotation pairs for IDRiD dataset"""
        pairs = []
        
        # Get all image files
        image_files = {}
        for root, dirs, files in os.walk(images_path):
            for file in files:
                if Path(file).suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}:
                    # Extract IDRiD image ID (e.g., IDRiD_73 from IDRiD_73.jpg)
                    base_name = Path(file).stem
                    full_path = Path(root) / file
                    image_files[base_name] = str(full_path)
        
        # Get all annotation files - look for various formats
        annot_files = {}
        for root, dirs, files in os.walk(boxes_path):
            for file in files:
                # Look for various annotation file formats
                if Path(file).suffix.lower() in {'.txt', '.xml', '.json', '.csv'}:
                    stem = Path(file).stem
                    
                    # Try to extract image ID from annotation filename
                    if '_' in stem:
                        # For files like IDRiD_73_annotations.txt
                        parts = stem.split('_')
                        if len(parts) >= 2:
                            image_id = '_'.join(parts[:2])  # IDRiD_73
                            full_path = Path(root) / file
                            if image_id not in annot_files:
                                annot_files[image_id] = []
                            annot_files[image_id].append(str(full_path))
                    else:
                        # For files that might match exactly
                        full_path = Path(root) / file
                        annot_files[stem] = [str(full_path)]
        
        # Create pairs - each image with its annotation files
        for image_id, image_path in image_files.items():
            if image_id in annot_files:
                for annot_path in annot_files[image_id]:
                    pairs.append((image_path, annot_path))
            else:
                # Try to find by partial matching
                for annot_id, annot_paths in annot_files.items():
                    if image_id in annot_id or annot_id in image_id:
                        for annot_path in annot_paths:
                            pairs.append((image_path, annot_path))
                        break
        
        return pairs
    
    def _analyze_grade_distribution(self, labels_path: Path) -> Dict:
        """Analyze DR grade distribution from label files"""
        distribution = defaultdict(int)
        
        # Try to find CSV files
        csv_files = list(labels_path.glob("*.csv"))
        if csv_files:
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'diagnosis' in df.columns:
                        for grade in df['diagnosis']:
                            distribution[int(grade)] += 1
                    elif 'grade' in df.columns:
                        for grade in df['grade']:
                            distribution[int(grade)] += 1
                except Exception as e:
                    print(f"Warning: Could not read {csv_file}: {e}")
        
        return dict(distribution)
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("DATASET ORGANIZATION SUMMARY")
        print("="*80)
        
        total_images = 0
        total_masks = 0
        total_pairs = 0
        
        for dataset_name, data in self.results.items():
            if data.get('status') == 'success':
                print(f"\nOK {dataset_name.upper()}:")
                
                if dataset_name == 'segmentation':
                    img_count = data.get('images', {}).get('count', 0)
                    mask_count = data.get('masks', {}).get('count', 0)
                    pair_count = len(data.get('pairs', []))
                    print(f"   Images: {img_count:,}")
                    print(f"   Masks: {mask_count:,}")
                    print(f"   Valid pairs: {pair_count:,}")
                    total_images += img_count
                    total_masks += mask_count
                    total_pairs += pair_count
                
                elif dataset_name == 'grading':
                    img_count = data.get('images', {}).get('count', 0)
                    print(f"   Images: {img_count:,}")
                    print(f"   Grade distribution: {data.get('grade_distribution', {})}")
                    total_images += img_count
                
                elif dataset_name == 'localization':
                    img_count = data.get('images', {}).get('count', 0)
                    pair_count = len(data.get('bbox_pairs', []))
                    print(f"   Images: {img_count:,}")
                    print(f"   Annotation pairs: {pair_count:,}")
                    total_images += img_count
                
                elif dataset_name == 'aptos2019':
                    train_count = data.get('train_images', {}).get('count', 0)
                    test_count = data.get('test_images', {}).get('count', 0)
                    print(f"   Train images: {train_count:,}")
                    print(f"   Test images: {test_count:,}")
                    print(f"   Grade distribution: {data.get('grade_distribution', {})}")
                    total_images += train_count + test_count
            else:
                print(f"\nERROR {dataset_name.upper()}: {data.get('reason', 'failed')}")
        
        print(f"\nProcessing TOTALS:")
        print(f"   Total images: {total_images:,}")
        print(f"   Total masks: {total_masks:,}")
        print(f"   Valid image-mask pairs: {total_pairs:,}")
        
        # Save results to JSON
        results_file = self.processed_path / "organization_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults Results saved to: {results_file}")

def main():
    """Main function to run data organization"""
    # Use relative path from project root
    project_root = Path(__file__).parent.parent.parent
    base_path = str(project_root / "dataset")
    
    organizer = DataOrganizer(base_path)
    results = organizer.organize_datasets()
    
    return results

if __name__ == "__main__":
    main()
