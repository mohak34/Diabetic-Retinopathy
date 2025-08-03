"""
Comprehensive Data Preprocessing Pipeline for DR Multi-Task Project
Processes all datasets with identical preprocessing steps and validates data integrity.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our preprocessing modules
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import RetinaPreprocessor, load_image, save_image
from src.utils.config import get_preprocessing_config


class DataPreprocessingPipeline:
    """Comprehensive preprocessing pipeline for all DR datasets"""
    
    def __init__(self, base_data_path: str, num_workers: int = 4):
        self.base_path = Path(base_data_path)
        self.processed_path = self.base_path / "processed"
        self.num_workers = num_workers
        
        # Load preprocessing configuration
        try:
            config = get_preprocessing_config()
            self.preprocessor = RetinaPreprocessor(
                clahe_clip_limit=config.get('clahe', {}).get('clip_limit', 2.0),
                clahe_tile_grid_size=tuple(config.get('clahe', {}).get('tile_grid_size', [8, 8])),
                ben_graham_sigma=config.get('ben_graham', {}).get('sigma', 10.0),
                resize_to=256,
                crop_to=config.get('image_size', [224, 224])[0]
            )
        except Exception as e:
            print(f"Warning: Could not load config ({e}). Using defaults.")
            self.preprocessor = RetinaPreprocessor(
                clahe_clip_limit=2.0,
                clahe_tile_grid_size=(8, 8),
                ben_graham_sigma=10.0,
                resize_to=256,
                crop_to=224
            )
        
        self.results = {}
    
    def process_all_datasets(self) -> Dict:
        """Process all datasets with identical preprocessing"""
        print("="*80)
        print("COMPREHENSIVE DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        # Load organization results
        org_results_file = self.processed_path / "organization_results.json"
        if not org_results_file.exists():
            raise FileNotFoundError("Please run data_organizer.py first!")
        
        with open(org_results_file, 'r') as f:
            self.org_results = json.load(f)
        
        # Process each dataset
        self.results['segmentation'] = self._process_segmentation_dataset()
        self.results['grading'] = self._process_grading_dataset()
        self.results['localization'] = self._process_localization_dataset()
        self.results['aptos2019'] = self._process_aptos_dataset()
        
        # Generate masks from bounding boxes
        self.results['generated_masks'] = self._generate_masks_from_boxes()
        
        # Create training splits
        self.results['splits'] = self._create_training_splits()
        
        # Validate processed data
        self.results['validation'] = self._validate_processed_data()
        
        # Save results
        self._save_processing_results()
        
        return self.results
    
    def _process_segmentation_dataset(self) -> Dict:
        """Process segmentation dataset (A. Segmentation)"""
        print("\nProcessing Processing Segmentation Dataset...")
        
        seg_data = self.org_results.get('segmentation', {})
        if seg_data.get('status') != 'success':
            print("ERROR Segmentation data not available")
            return {'status': 'failed', 'reason': 'data_not_available'}
        
        pairs = seg_data.get('pairs', [])
        if not pairs:
            print("ERROR No image-mask pairs found")
            return {'status': 'failed', 'reason': 'no_pairs'}
        
        print(f"Processing {len(pairs)} image-mask pairs...")
        
        # Create output directories
        img_output_dir = self.processed_path / "segmentation" / "images"
        mask_output_dir = self.processed_path / "segmentation" / "masks"
        img_output_dir.mkdir(parents=True, exist_ok=True)
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_pairs = []
        failed_pairs = []
        
        # Process pairs
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i, (img_path, mask_path) in enumerate(pairs):
                future = executor.submit(
                    self._process_segmentation_pair,
                    img_path, mask_path, i, img_output_dir, mask_output_dir
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                if result['success']:
                    processed_pairs.append(result)
                else:
                    failed_pairs.append(result)
        
        # Create metadata file
        metadata = {
            'total_pairs': len(pairs),
            'processed_pairs': len(processed_pairs),
            'failed_pairs': len(failed_pairs),
            'pairs': processed_pairs
        }
        
        metadata_file = self.processed_path / "segmentation" / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"OK Processed {len(processed_pairs)}/{len(pairs)} pairs successfully")
        if failed_pairs:
            print(f"WARNING  {len(failed_pairs)} pairs failed processing")
        
        return {
            'status': 'success',
            'processed_count': len(processed_pairs),
            'failed_count': len(failed_pairs),
            'output_dir': str(img_output_dir.parent)
        }
    
    def _process_segmentation_pair(self, img_path: str, mask_path: str, 
                                 index: int, img_output_dir: Path, 
                                 mask_output_dir: Path) -> Dict:
        """Process a single image-mask pair"""
        try:
            # Load image
            image = load_image(img_path)
            if image is None:
                return {'success': False, 'reason': 'failed_to_load_image', 'index': index}
            
            # Load mask - handle .tif files properly
            try:
                if str(mask_path).lower().endswith('.tif') or str(mask_path).lower().endswith('.tiff'):
                    # Use cv2 for TIFF files
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        # Try with different flags
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                else:
                    mask = load_image(mask_path)
                
                if mask is None:
                    return {'success': False, 'reason': 'failed_to_load_mask', 'index': index}
                    
            except Exception as e:
                return {'success': False, 'reason': f'mask_load_error: {str(e)}', 'index': index}
            
            # Get original dimensions
            original_shape = image.shape
            
            # Preprocess image
            processed_image = self.preprocessor.preprocess(image)
            
            # Process mask (same spatial transformations, no normalization)
            if len(mask.shape) == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask.copy()
            
            # Ensure mask is in the right data type
            if mask_gray.dtype != np.uint8:
                mask_gray = mask_gray.astype(np.uint8)
            
            # Apply same spatial transformations to mask
            # 1. Crop black borders (using same method but adapted for masks)
            # For masks, we need to be more careful with border detection
            h, w = mask_gray.shape
            if h > 50 and w > 50:  # Only crop if image is large enough
                # Find non-zero regions
                coords = cv2.findNonZero(mask_gray)
                if coords is not None:
                    x, y, w_crop, h_crop = cv2.boundingRect(coords)
                    # Add some padding
                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w_crop = min(w - x, w_crop + 2*padding)
                    h_crop = min(h - y, h_crop + 2*padding)
                    mask_cropped = mask_gray[y:y+h_crop, x:x+w_crop]
                else:
                    mask_cropped = mask_gray
            else:
                mask_cropped = mask_gray
            
            # 2. Resize and crop (no CLAHE or normalization for masks)
            mask_resized = cv2.resize(mask_cropped, (self.preprocessor.resize_to, self.preprocessor.resize_to), 
                                    interpolation=cv2.INTER_NEAREST)  # Use nearest neighbor for masks
            
            # Center crop
            h, w = mask_resized.shape
            crop_size = self.preprocessor.crop_to
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
            mask_processed = mask_resized[start_h:start_h+crop_size, start_w:start_w+crop_size]
            
            # Binarize mask (threshold at any non-zero value)
            _, mask_binary = cv2.threshold(mask_processed, 0, 255, cv2.THRESH_BINARY)
            
            # Save processed files
            img_filename = f"seg_{index:04d}.png"
            mask_filename = f"seg_{index:04d}_mask.png"
            
            img_output_path = img_output_dir / img_filename
            mask_output_path = mask_output_dir / mask_filename
            
            save_image(processed_image, str(img_output_path))
            cv2.imwrite(str(mask_output_path), mask_binary)
            
            return {
                'success': True,
                'index': index,
                'original_image': img_path,
                'original_mask': mask_path,
                'processed_image': str(img_output_path),
                'processed_mask': str(mask_output_path),
                'original_shape': original_shape,
                'processed_shape': processed_image.shape
            }
            
        except Exception as e:
            return {'success': False, 'reason': str(e), 'index': index}
    
    def _process_grading_dataset(self) -> Dict:
        """Process grading dataset (B. Disease Grading)"""
        print("\nProcessing Processing Disease Grading Dataset...")
        
        grading_data = self.org_results.get('grading', {})
        if grading_data.get('status') != 'success':
            print("ERROR Grading data not available")
            return {'status': 'failed', 'reason': 'data_not_available'}
        
        # Find all images and labels
        grading_path = self.base_path / "B. Disease Grading"
        images_path = grading_path / "1. Original Images"
        labels_path = grading_path / "2. Groundtruths"
        
        # Get all image files
        image_files = []
        for root, dirs, files in os.walk(images_path):
            for file in files:
                if Path(file).suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}:
                    image_files.append(str(Path(root) / file))
        
        print(f"Processing {len(image_files)} images...")
        
        # Create output directory
        output_dir = self.processed_path / "grading" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_images = []
        failed_images = []
        
        # Process images
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i, img_path in enumerate(image_files):
                future = executor.submit(
                    self._process_grading_image,
                    img_path, i, output_dir
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                if result['success']:
                    processed_images.append(result)
                else:
                    failed_images.append(result)
        
        # Process labels
        labels_data = self._process_grading_labels(labels_path, processed_images)
        
        # Create metadata
        metadata = {
            'total_images': len(image_files),
            'processed_images': len(processed_images),
            'failed_images': len(failed_images),
            'labels': labels_data,
            'images': processed_images
        }
        
        metadata_file = self.processed_path / "grading" / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"OK Processed {len(processed_images)}/{len(image_files)} images successfully")
        
        return {
            'status': 'success',
            'processed_count': len(processed_images),
            'failed_count': len(failed_images),
            'labels_data': labels_data,
            'output_dir': str(output_dir.parent)
        }
    
    def _process_grading_image(self, img_path: str, index: int, output_dir: Path) -> Dict:
        """Process a single grading image"""
        try:
            # Load and preprocess image
            image = load_image(img_path)
            if image is None:
                return {'success': False, 'reason': 'failed_to_load', 'index': index}
            
            original_shape = image.shape
            processed_image = self.preprocessor.preprocess(image)
            
            # Save processed image
            img_filename = f"grade_{index:04d}.png"
            output_path = output_dir / img_filename
            save_image(processed_image, str(output_path))
            
            # Extract original filename for label matching
            original_filename = Path(img_path).name
            
            return {
                'success': True,
                'index': index,
                'original_image': img_path,
                'original_filename': original_filename,
                'processed_image': str(output_path),
                'original_shape': original_shape,
                'processed_shape': processed_image.shape
            }
            
        except Exception as e:
            return {'success': False, 'reason': str(e), 'index': index}
    
    def _process_grading_labels(self, labels_path: Path, processed_images: List) -> Dict:
        """Process grading labels"""
        labels_data = {'files': [], 'labels': {}, 'distribution': {}}
        
        if not labels_path.exists():
            return labels_data
        
        # Find CSV files
        csv_files = list(labels_path.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                labels_data['files'].append(str(csv_file))
                
                # Process labels based on columns
                if 'Image name' in df.columns and 'Retinopathy grade' in df.columns:
                    for _, row in df.iterrows():
                        img_name = row['Image name']
                        grade = int(row['Retinopathy grade'])
                        labels_data['labels'][img_name] = grade
                        
                        # Update distribution
                        if grade not in labels_data['distribution']:
                            labels_data['distribution'][grade] = 0
                        labels_data['distribution'][grade] += 1
                
                elif 'id_code' in df.columns and 'diagnosis' in df.columns:
                    for _, row in df.iterrows():
                        img_name = row['id_code']
                        grade = int(row['diagnosis'])
                        labels_data['labels'][img_name] = grade
                        
                        # Update distribution
                        if grade not in labels_data['distribution']:
                            labels_data['distribution'][grade] = 0
                        labels_data['distribution'][grade] += 1
                
            except Exception as e:
                print(f"Warning: Could not process {csv_file}: {e}")
        
        return labels_data
    
    def _process_localization_dataset(self) -> Dict:
        """Process localization dataset (C. Localization)"""
        print("\n📍 Processing Localization Dataset...")
        
        loc_data = self.org_results.get('localization', {})
        if loc_data.get('status') != 'success':
            print("ERROR Localization data not available")
            return {'status': 'failed', 'reason': 'data_not_available'}
        
        bbox_pairs = loc_data.get('bbox_pairs', [])
        if not bbox_pairs:
            print("ERROR No image-annotation pairs found")
            return {'status': 'failed', 'reason': 'no_pairs'}
        
        print(f"Processing {len(bbox_pairs)} image-annotation pairs...")
        
        # Create output directories
        output_dir = self.processed_path / "localization"
        img_output_dir = output_dir / "images"
        boxes_output_dir = output_dir / "boxes"
        
        img_output_dir.mkdir(parents=True, exist_ok=True)
        boxes_output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_pairs = []
        failed_pairs = []
        
        # Process pairs
        for i, (img_path, bbox_path) in enumerate(tqdm(bbox_pairs, desc="Processing")):
            result = self._process_localization_pair(
                img_path, bbox_path, i, img_output_dir, boxes_output_dir
            )
            if result['success']:
                processed_pairs.append(result)
            else:
                failed_pairs.append(result)
        
        # Create metadata
        metadata = {
            'total_pairs': len(bbox_pairs),
            'processed_pairs': len(processed_pairs),
            'failed_pairs': len(failed_pairs),
            'pairs': processed_pairs
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"OK Processed {len(processed_pairs)}/{len(bbox_pairs)} pairs successfully")
        
        return {
            'status': 'success',
            'processed_count': len(processed_pairs),
            'failed_count': len(failed_pairs),
            'output_dir': str(output_dir)
        }
    
    def _process_localization_pair(self, img_path: str, bbox_path: str, 
                                 index: int, img_output_dir: Path, 
                                 boxes_output_dir: Path) -> Dict:
        """Process a single image-bbox pair"""
        try:
            # Load and preprocess image
            image = load_image(img_path)
            if image is None:
                return {'success': False, 'reason': 'failed_to_load_image', 'index': index}
            
            original_shape = image.shape
            processed_image = self.preprocessor.preprocess(image)
            
            # Save processed image
            img_filename = f"loc_{index:04d}.png"
            img_output_path = img_output_dir / img_filename
            save_image(processed_image, str(img_output_path))
            
            # Copy annotation file
            bbox_filename = f"loc_{index:04d}_boxes.txt"
            bbox_output_path = boxes_output_dir / bbox_filename
            shutil.copy2(bbox_path, bbox_output_path)
            
            return {
                'success': True,
                'index': index,
                'original_image': img_path,
                'original_bbox': bbox_path,
                'processed_image': str(img_output_path),
                'processed_bbox': str(bbox_output_path),
                'original_shape': original_shape,
                'processed_shape': processed_image.shape
            }
            
        except Exception as e:
            return {'success': False, 'reason': str(e), 'index': index}
    
    def _process_aptos_dataset(self) -> Dict:
        """Process APTOS 2019 dataset"""
        print("\nProcessing Processing APTOS 2019 Dataset...")
        
        aptos_data = self.org_results.get('aptos2019', {})
        if aptos_data.get('status') != 'success':
            print("ERROR APTOS data not available")
            return {'status': 'failed', 'reason': 'data_not_available'}
        
        # Process train images
        aptos_path = self.base_path / "aptos2019-blindness-detection"
        train_images_path = aptos_path / "train_images"
        train_csv_path = aptos_path / "train.csv"
        
        # Get all train images
        train_image_files = list(train_images_path.glob("*.png"))
        print(f"Processing {len(train_image_files)} training images...")
        
        # Create output directory
        output_dir = self.processed_path / "aptos2019" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_images = []
        failed_images = []
        
        # Process images
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i, img_path in enumerate(train_image_files):
                future = executor.submit(
                    self._process_aptos_image,
                    str(img_path), i, output_dir
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                if result['success']:
                    processed_images.append(result)
                else:
                    failed_images.append(result)
        
        # Process labels
        labels_data = {}
        if train_csv_path.exists():
            df = pd.read_csv(train_csv_path)
            labels_data = {
                'csv_file': str(train_csv_path),
                'labels': df.to_dict('records'),
                'distribution': df['diagnosis'].value_counts().to_dict()
            }
            
            # Copy CSV to processed directory
            processed_csv = self.processed_path / "aptos2019" / "train_labels.csv"
            df.to_csv(processed_csv, index=False)
        
        # Create metadata
        metadata = {
            'total_images': len(train_image_files),
            'processed_images': len(processed_images),
            'failed_images': len(failed_images),
            'labels': labels_data,
            'images': processed_images
        }
        
        metadata_file = self.processed_path / "aptos2019" / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"OK Processed {len(processed_images)}/{len(train_image_files)} images successfully")
        
        return {
            'status': 'success',
            'processed_count': len(processed_images),
            'failed_count': len(failed_images),
            'labels_data': labels_data,
            'output_dir': str(output_dir.parent)
        }
    
    def _process_aptos_image(self, img_path: str, index: int, output_dir: Path) -> Dict:
        """Process a single APTOS image"""
        try:
            # Load and preprocess image
            image = load_image(img_path)
            if image is None:
                return {'success': False, 'reason': 'failed_to_load', 'index': index}
            
            original_shape = image.shape
            processed_image = self.preprocessor.preprocess(image)
            
            # Save processed image
            original_filename = Path(img_path).stem
            img_filename = f"aptos_{index:04d}_{original_filename}.png"
            output_path = output_dir / img_filename
            save_image(processed_image, str(output_path))
            
            return {
                'success': True,
                'index': index,
                'original_image': img_path,
                'original_filename': original_filename,
                'processed_image': str(output_path),
                'original_shape': original_shape,
                'processed_shape': processed_image.shape
            }
            
        except Exception as e:
            return {'success': False, 'reason': str(e), 'index': index}
    
    def _generate_masks_from_boxes(self) -> Dict:
        """Generate segmentation masks from bounding boxes"""
        print("\nChecking Generating Masks from Bounding Boxes...")
        
        loc_data = self.results.get('localization', {})
        if loc_data.get('status') != 'success':
            print("ERROR No localization data to process")
            return {'status': 'failed', 'reason': 'no_localization_data'}
        
        # This is a placeholder - actual implementation depends on annotation format
        # For now, we'll create a simple framework
        
        output_dir = self.processed_path / "localization" / "generated_masks"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("WARNING  Mask generation from bounding boxes needs annotation format analysis")
        print("   This will be implemented after examining the annotation files")
        
        return {
            'status': 'pending',
            'reason': 'needs_annotation_format_analysis',
            'output_dir': str(output_dir)
        }
    
    def _create_training_splits(self) -> Dict:
        """Create training/validation splits for all datasets"""
        print("\nCreating  Creating Training Splits...")
        
        splits = {}
        
        # Segmentation splits
        seg_metadata_file = self.processed_path / "segmentation" / "metadata.json"
        if seg_metadata_file.exists():
            splits['segmentation'] = self._create_segmentation_splits()
        
        # Grading splits  
        grade_metadata_file = self.processed_path / "grading" / "metadata.json"
        if grade_metadata_file.exists():
            splits['grading'] = self._create_grading_splits()
        
        # APTOS splits
        aptos_metadata_file = self.processed_path / "aptos2019" / "metadata.json"
        if aptos_metadata_file.exists():
            splits['aptos2019'] = self._create_aptos_splits()
        
        return splits
    
    def _create_segmentation_splits(self) -> Dict:
        """Create segmentation train/val splits"""
        # Load metadata
        with open(self.processed_path / "segmentation" / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        pairs = metadata['pairs']
        np.random.seed(42)
        np.random.shuffle(pairs)
        
        # 80/20 split
        split_idx = int(0.8 * len(pairs))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        
        return {
            'train_count': len(train_pairs),
            'val_count': len(val_pairs),
            'train_pairs': train_pairs,
            'val_pairs': val_pairs
        }
    
    def _create_grading_splits(self) -> Dict:
        """Create grading train/val splits with stratification"""
        # Load metadata
        with open(self.processed_path / "grading" / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Simple random split for now - stratification would need label matching
        images = metadata['images']
        np.random.seed(42)
        np.random.shuffle(images)
        
        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        return {
            'train_count': len(train_images),
            'val_count': len(val_images),
            'train_images': train_images,
            'val_images': val_images
        }
    
    def _create_aptos_splits(self) -> Dict:
        """Create APTOS train/val splits with stratification"""
        # Load metadata and labels
        with open(self.processed_path / "aptos2019" / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load labels CSV
        labels_csv = self.processed_path / "aptos2019" / "train_labels.csv"
        if labels_csv.exists():
            df = pd.read_csv(labels_csv)
            
            # Stratified split by diagnosis
            from sklearn.model_selection import train_test_split
            
            train_df, val_df = train_test_split(
                df, test_size=0.2, random_state=42, 
                stratify=df['diagnosis']
            )
            
            return {
                'train_count': len(train_df),
                'val_count': len(val_df),
                'train_labels': train_df.to_dict('records'),
                'val_labels': val_df.to_dict('records'),
                'train_distribution': train_df['diagnosis'].value_counts().to_dict(),
                'val_distribution': val_df['diagnosis'].value_counts().to_dict()
            }
        
        # Fallback to simple split
        images = metadata['images']
        np.random.seed(42)
        np.random.shuffle(images)
        
        split_idx = int(0.8 * len(images))
        return {
            'train_count': split_idx,
            'val_count': len(images) - split_idx
        }
    
    def _validate_processed_data(self) -> Dict:
        """Validate processed data integrity"""
        print("\nOK Validating Processed Data...")
        
        validation = {
            'segmentation': self._validate_segmentation_data(),
            'grading': self._validate_grading_data(),
            'aptos2019': self._validate_aptos_data()
        }
        
        return validation
    
    def _validate_segmentation_data(self) -> Dict:
        """Validate segmentation data"""
        seg_dir = self.processed_path / "segmentation"
        if not seg_dir.exists():
            return {'status': 'not_found'}
        
        img_dir = seg_dir / "images"
        mask_dir = seg_dir / "masks"
        
        img_count = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        mask_count = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0
        
        return {
            'status': 'success',
            'image_count': img_count,
            'mask_count': mask_count,
            'pairs_match': img_count == mask_count
        }
    
    def _validate_grading_data(self) -> Dict:
        """Validate grading data"""
        grade_dir = self.processed_path / "grading"
        if not grade_dir.exists():
            return {'status': 'not_found'}
        
        img_dir = grade_dir / "images"
        img_count = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        
        return {
            'status': 'success',
            'image_count': img_count
        }
    
    def _validate_aptos_data(self) -> Dict:
        """Validate APTOS data"""
        aptos_dir = self.processed_path / "aptos2019"
        if not aptos_dir.exists():
            return {'status': 'not_found'}
        
        img_dir = aptos_dir / "images"
        img_count = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        
        return {
            'status': 'success',
            'image_count': img_count
        }
    
    def _save_processing_results(self):
        """Save processing results"""
        results_file = self.processed_path / "preprocessing_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults Processing results saved to: {results_file}")


def main():
    """Main function to run preprocessing pipeline"""
    # Use relative path from project root
    project_root = Path(__file__).parent.parent.parent
    base_path = str(project_root / "dataset")
    
    pipeline = DataPreprocessingPipeline(base_path, num_workers=4)
    results = pipeline.process_all_datasets()
    
    return results


if __name__ == "__main__":
    main()
