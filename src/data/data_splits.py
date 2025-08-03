"""
Data Splitting Utilities for Diabetic Retinopathy Multi-Task Learning
Creates stratified train/validation splits with proper class balance.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import defaultdict, Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSplitter:
    """
    Creates stratified train/validation splits for diabetic retinopathy datasets.
    Ensures balanced representation across all DR severity classes and lesion types.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.splits = {}
    
    def create_classification_splits(
        self,
        data_dir: str,
        labels_path: str,
        val_size: float = 0.2,
        test_size: float = 0.1,
        min_samples_per_class: int = 1
    ) -> Dict[str, List[str]]:
        """
        Create stratified splits for classification datasets.
        
        Args:
            data_dir: Directory containing images
            labels_path: Path to CSV file with labels
            val_size: Fraction of data for validation
            test_size: Fraction of data for testing (0 to skip test split)
            min_samples_per_class: Minimum samples required per class
            
        Returns:
            Dictionary with 'train', 'val', and optionally 'test' image IDs
        """
        # Load labels
        labels_df = pd.read_csv(labels_path)
        
        # Auto-detect columns
        if 'id_code' in labels_df.columns and 'diagnosis' in labels_df.columns:
            # APTOS format
            image_col, label_col = 'id_code', 'diagnosis'
        elif 'Image name' in labels_df.columns and 'Retinopathy grade' in labels_df.columns:
            # IDRiD format
            image_col, label_col = 'Image name', 'Retinopathy grade'
        else:
            # Auto-detect: first column as image, last as label
            image_col, label_col = labels_df.columns[0], labels_df.columns[-1]
            logger.warning(f"Auto-detected columns: {image_col} (images), {label_col} (labels)")
        
        # Extract image IDs and labels
        image_ids = []
        labels = []
        
        for _, row in labels_df.iterrows():
            img_id = str(row[image_col])
            if img_id.endswith('.jpg') or img_id.endswith('.png'):
                img_id = Path(img_id).stem
            
            # Check if image exists
            img_path = self._find_image_file(data_dir, img_id)
            if img_path:
                image_ids.append(img_id)
                labels.append(int(row[label_col]))
        
        # Check class distribution
        label_counts = Counter(labels)
        logger.info(f"Label distribution: {dict(label_counts)}")
        
        # Filter out classes with too few samples
        valid_indices = []
        for i, label in enumerate(labels):
            if label_counts[label] >= min_samples_per_class:
                valid_indices.append(i)
        
        image_ids = [image_ids[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        logger.info(f"After filtering: {len(image_ids)} samples, {len(set(labels))} classes")
        
        # Create splits
        if test_size > 0:
            # Three-way split
            X_temp, X_test, y_temp, y_test = train_test_split(
                image_ids, labels,
                test_size=test_size,
                stratify=labels,
                random_state=self.random_state
            )
            
            # Adjust val_size for remaining data
            adjusted_val_size = val_size / (1 - test_size)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=adjusted_val_size,
                stratify=y_temp,
                random_state=self.random_state
            )
            
            splits = {
                'train': X_train,
                'val': X_val,
                'test': X_test
            }
            
            # Log split distributions
            self._log_split_distribution('train', y_train)
            self._log_split_distribution('val', y_val)
            self._log_split_distribution('test', y_test)
            
        else:
            # Two-way split
            X_train, X_val, y_train, y_val = train_test_split(
                image_ids, labels,
                test_size=val_size,
                stratify=labels,
                random_state=self.random_state
            )
            
            splits = {
                'train': X_train,
                'val': X_val
            }
            
            # Log split distributions
            self._log_split_distribution('train', y_train)
            self._log_split_distribution('val', y_val)
        
        return splits
    
    def create_segmentation_splits(
        self,
        image_mask_pairs: List[Tuple[str, str]],
        val_size: float = 0.2,
        test_size: float = 0.1,
        stratify_by_mask_presence: bool = True
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Create splits for segmentation datasets.
        
        Args:
            image_mask_pairs: List of (image_path, mask_path) tuples
            val_size: Fraction of data for validation
            test_size: Fraction of data for testing (0 to skip test split)
            stratify_by_mask_presence: Whether to stratify by mask presence/absence
            
        Returns:
            Dictionary with 'train', 'val', and optionally 'test' pairs
        """
        if stratify_by_mask_presence:
            # Create stratification labels based on mask content
            stratify_labels = []
            
            for img_path, mask_path in image_mask_pairs:
                # Load mask and check if it contains lesions
                try:
                    import cv2
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        has_lesions = np.any(mask > 127)  # Check if mask has positive pixels
                        stratify_labels.append(1 if has_lesions else 0)
                    else:
                        stratify_labels.append(0)  # Assume no lesions if can't load
                except Exception:
                    stratify_labels.append(0)  # Default to no lesions
            
            logger.info(f"Segmentation mask distribution: "
                       f"with lesions: {sum(stratify_labels)}, "
                       f"without lesions: {len(stratify_labels) - sum(stratify_labels)}")
            
            # Create stratified splits
            if test_size > 0:
                pairs_temp, pairs_test, labels_temp, labels_test = train_test_split(
                    image_mask_pairs, stratify_labels,
                    test_size=test_size,
                    stratify=stratify_labels,
                    random_state=self.random_state
                )
                
                adjusted_val_size = val_size / (1 - test_size)
                
                pairs_train, pairs_val, labels_train, labels_val = train_test_split(
                    pairs_temp, labels_temp,
                    test_size=adjusted_val_size,
                    stratify=labels_temp,
                    random_state=self.random_state
                )
                
                splits = {
                    'train': pairs_train,
                    'val': pairs_val,
                    'test': pairs_test
                }
                
            else:
                pairs_train, pairs_val, labels_train, labels_val = train_test_split(
                    image_mask_pairs, stratify_labels,
                    test_size=val_size,
                    stratify=stratify_labels,
                    random_state=self.random_state
                )
                
                splits = {
                    'train': pairs_train,
                    'val': pairs_val
                }
        
        else:
            # Random split without stratification
            np.random.seed(self.random_state)
            indices = np.random.permutation(len(image_mask_pairs))
            
            if test_size > 0:
                test_size_idx = int(len(indices) * test_size)
                val_size_idx = int(len(indices) * val_size)
                
                test_indices = indices[:test_size_idx]
                val_indices = indices[test_size_idx:test_size_idx + val_size_idx]
                train_indices = indices[test_size_idx + val_size_idx:]
                
                splits = {
                    'train': [image_mask_pairs[i] for i in train_indices],
                    'val': [image_mask_pairs[i] for i in val_indices],
                    'test': [image_mask_pairs[i] for i in test_indices]
                }
            else:
                val_size_idx = int(len(indices) * val_size)
                
                val_indices = indices[:val_size_idx]
                train_indices = indices[val_size_idx:]
                
                splits = {
                    'train': [image_mask_pairs[i] for i in train_indices],
                    'val': [image_mask_pairs[i] for i in val_indices]
                }
        
        # Log split sizes
        for split_name, split_data in splits.items():
            logger.info(f"{split_name}: {len(split_data)} pairs")
        
        return splits
    
    def create_kfold_splits(
        self,
        data_dir: str,
        labels_path: str,
        n_folds: int = 5
    ) -> Dict[int, Dict[str, List[str]]]:
        """
        Create K-fold cross-validation splits for classification.
        
        Args:
            data_dir: Directory containing images
            labels_path: Path to CSV file with labels
            n_folds: Number of folds
            
        Returns:
            Dictionary mapping fold numbers to train/val splits
        """
        # Load data
        labels_df = pd.read_csv(labels_path)
        
        # Auto-detect columns
        if 'id_code' in labels_df.columns and 'diagnosis' in labels_df.columns:
            image_col, label_col = 'id_code', 'diagnosis'
        elif 'Image name' in labels_df.columns and 'Retinopathy grade' in labels_df.columns:
            image_col, label_col = 'Image name', 'Retinopathy grade'
        else:
            image_col, label_col = labels_df.columns[0], labels_df.columns[-1]
        
        # Extract image IDs and labels
        image_ids = []
        labels = []
        
        for _, row in labels_df.iterrows():
            img_id = str(row[image_col])
            if img_id.endswith('.jpg') or img_id.endswith('.png'):
                img_id = Path(img_id).stem
            
            img_path = self._find_image_file(data_dir, img_id)
            if img_path:
                image_ids.append(img_id)
                labels.append(int(row[label_col]))
        
        # Create K-fold splits
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        kfold_splits = {}
        for fold, (train_idx, val_idx) in enumerate(skf.split(image_ids, labels)):
            train_ids = [image_ids[i] for i in train_idx]
            val_ids = [image_ids[i] for i in val_idx]
            
            kfold_splits[fold] = {
                'train': train_ids,
                'val': val_ids
            }
            
            # Log fold distribution
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            
            logger.info(f"Fold {fold}: train={len(train_ids)}, val={len(val_ids)}")
            self._log_split_distribution(f'fold_{fold}_train', train_labels)
            self._log_split_distribution(f'fold_{fold}_val', val_labels)
        
        return kfold_splits
    
    def save_splits(self, splits: Dict, save_path: str, dataset_name: str):
        """
        Save splits to JSON file for reproducibility.
        
        Args:
            splits: Dictionary containing splits
            save_path: Path to save JSON file
            dataset_name: Name of the dataset
        """
        # Convert numpy arrays and Path objects to serializable format
        serializable_splits = {}
        for split_name, split_data in splits.items():
            if isinstance(split_data, dict):
                # K-fold splits
                serializable_splits[split_name] = {}
                for fold, fold_data in split_data.items():
                    serializable_splits[split_name][fold] = {
                        'train': [str(x) for x in fold_data['train']],
                        'val': [str(x) for x in fold_data['val']]
                    }
            elif isinstance(split_data, list) and len(split_data) > 0:
                if isinstance(split_data[0], tuple):
                    # Segmentation pairs
                    serializable_splits[split_name] = [
                        {'image': str(img), 'mask': str(mask)} 
                        for img, mask in split_data
                    ]
                else:
                    # Classification image IDs
                    serializable_splits[split_name] = [str(x) for x in split_data]
        
        # Add metadata
        split_info = {
            'dataset_name': dataset_name,
            'random_state': self.random_state,
            'splits': serializable_splits,
            'split_sizes': {
                split_name: len(split_data) if not isinstance(split_data, dict) else 
                           {fold: len(fold_data['train']) + len(fold_data['val']) 
                            for fold, fold_data in split_data.items()}
                for split_name, split_data in splits.items()
            }
        }
        
        # Save to file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Splits saved to: {save_path}")
    
    def load_splits(self, splits_path: str) -> Dict:
        """
        Load splits from JSON file.
        
        Args:
            splits_path: Path to JSON file containing splits
            
        Returns:
            Dictionary containing splits
        """
        with open(splits_path, 'r') as f:
            split_info = json.load(f)
        
        logger.info(f"Loaded splits for dataset: {split_info['dataset_name']}")
        return split_info['splits']
    
    def _find_image_file(self, data_dir: str, image_id: str) -> Optional[str]:
        """Find image file with given ID in directory."""
        image_exts = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        
        for ext in image_exts:
            img_path = os.path.join(data_dir, f"{image_id}{ext}")
            if os.path.exists(img_path):
                return img_path
        
        return None
    
    def _log_split_distribution(self, split_name: str, labels: List[int]):
        """Log the class distribution for a split."""
        label_counts = Counter(labels)
        total = len(labels)
        
        distribution = {label: f"{count} ({count/total*100:.1f}%)" 
                       for label, count in sorted(label_counts.items())}
        
        logger.info(f"{split_name} distribution: {distribution}")

def create_all_splits(
    processed_data_dir: str,
    splits_dir: str,
    val_size: float = 0.2,
    test_size: float = 0.0,
    random_state: int = 42
):
    """
    Create splits for all datasets in the project.
    
    Args:
        processed_data_dir: Path to processed data directory
        splits_dir: Directory to save split files
        val_size: Validation split size
        test_size: Test split size (0 to skip)
        random_state: Random seed
    """
    processed_path = Path(processed_data_dir)
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)
    
    splitter = DataSplitter(random_state=random_state)
    
    # 1. APTOS 2019 classification splits
    print("\nCreating Creating APTOS 2019 splits...")
    aptos_images_dir = processed_path / "aptos2019" / "images"
    # Try processed labels first, then original
    aptos_labels_path = processed_path / "aptos2019" / "train_labels.csv"
    if not aptos_labels_path.exists():
        aptos_labels_path = processed_path.parent / "aptos2019-blindness-detection" / "train.csv"
    
    if aptos_images_dir.exists() and aptos_labels_path.exists():
        aptos_splits = splitter.create_classification_splits(
            str(aptos_images_dir),
            str(aptos_labels_path),
            val_size=val_size,
            test_size=test_size
        )
        
        splitter.save_splits(
            aptos_splits,
            str(splits_path / "aptos2019_splits.json"),
            "APTOS 2019"
        )
        print(f"OK APTOS 2019 splits saved")
    else:
        print(f"ERROR APTOS 2019 data not found")
    
    # 2. IDRiD Disease Grading splits
    print("\nCreating Creating IDRiD Disease Grading splits...")
    grading_images_dir = processed_path / "grading" / "images"
    
    # Try processed labels first, then original
    grading_labels_path = processed_path / "grading" / "labels.csv"
    if not grading_labels_path.exists():
        # Look for IDRiD grading labels in B. Disease Grading folder
        grading_base = processed_path.parent / "B. Disease Grading" / "2. Groundtruths"
        grading_csv_files = list(grading_base.glob("*.csv")) if grading_base.exists() else []
        if grading_csv_files:
            grading_labels_path = grading_csv_files[0]
        else:
            grading_labels_path = None
    
    if grading_images_dir.exists() and grading_labels_path and grading_labels_path.exists():
        grading_splits = splitter.create_classification_splits(
            str(grading_images_dir),
            str(grading_labels_path),
            val_size=val_size,
            test_size=test_size
        )
        
        splitter.save_splits(
            grading_splits,
            str(splits_path / "idrid_grading_splits.json"),
            "IDRiD Disease Grading"
        )
        print(f"OK IDRiD Disease Grading splits saved")
    else:
        print(f"ERROR IDRiD Disease Grading data not found")
    
    # 3. IDRiD Segmentation splits
    print("\nCreating Creating IDRiD Segmentation splits...")
    
    # Load segmentation pairs from organization results
    org_results_path = processed_path / "organization_results.json"
    if org_results_path.exists():
        with open(org_results_path, 'r') as f:
            org_results = json.load(f)
        
        if 'segmentation' in org_results and org_results['segmentation'].get('status') == 'success':
            # Convert string pairs back to tuples
            pairs = org_results['segmentation']['pairs']
            image_mask_pairs = [(pair[0], pair[1]) for pair in pairs]
            
            segmentation_splits = splitter.create_segmentation_splits(
                image_mask_pairs,
                val_size=val_size,
                test_size=test_size,
                stratify_by_mask_presence=True
            )
            
            splitter.save_splits(
                segmentation_splits,
                str(splits_path / "idrid_segmentation_splits.json"),
                "IDRiD Segmentation"
            )
            print(f"OK IDRiD Segmentation splits saved")
        else:
            print(f"ERROR IDRiD Segmentation data not found in organization results")
    else:
        print(f"ERROR Organization results not found")
    
    print(f"\nAll splits saved to: {splits_path}")

if __name__ == "__main__":
    # Example usage - use relative paths from project root
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    processed_data_dir = str(project_root / "dataset" / "processed")
    splits_dir = str(project_root / "dataset" / "splits")
    
    create_all_splits(
        processed_data_dir=processed_data_dir,
        splits_dir=splits_dir,
        val_size=0.2,
        test_size=0.0,  # No test split for now
        random_state=42
    )
