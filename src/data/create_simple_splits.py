"""
Simple Data Splits Creation for Processed Data
Creates train/validation splits based on the processed data structure.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_aptos_splits(processed_dir: str, splits_dir: str, val_size: float = 0.2, random_state: int = 42):
    """Create splits for APTOS dataset using processed data."""
    
    processed_path = Path(processed_dir)
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)
    
    # Load APTOS metadata
    aptos_metadata_path = processed_path / "aptos2019" / "metadata.json"
    aptos_images_dir = processed_path / "aptos2019" / "images"
    
    if not aptos_metadata_path.exists() or not aptos_images_dir.exists():
        print("APTOS processed data not found")
        return
    
    print("Creating APTOS 2019 splits...")
    
    with open(aptos_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get image files and their corresponding labels
    image_files = [f for f in os.listdir(aptos_images_dir) if f.endswith('.png')]
    image_files.sort()
    
    # Create mapping based on order in metadata
    labels_list = metadata['labels']['labels']
    
    if len(image_files) != len(labels_list):
        print(f"⚠️ Mismatch: {len(image_files)} images vs {len(labels_list)} labels")
    
    # Create image_id to label mapping
    image_labels = {}
    for i, img_file in enumerate(image_files):
        if i < len(labels_list):
            img_id = Path(img_file).stem  # Remove .png extension
            diagnosis = labels_list[i]['diagnosis']
            image_labels[img_id] = diagnosis
    
    # Prepare data for splitting
    image_ids = list(image_labels.keys())
    labels = list(image_labels.values())
    
    print(f"Total images: {len(image_ids)}")
    print(f"Label distribution: {dict(Counter(labels))}")
    
    # Create stratified split
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        image_ids, labels,
        test_size=val_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Create splits dictionary
    splits = {
        'train': train_ids,
        'val': val_ids
    }
    
    # Save splits
    split_info = {
        'dataset_name': 'APTOS 2019',
        'random_state': random_state,
        'splits': splits,
        'split_sizes': {
            'train': len(train_ids),
            'val': len(val_ids)
        },
        'label_distribution': {
            'train': dict(Counter(train_labels)),
            'val': dict(Counter(val_labels))
        }
    }
    
    splits_file = splits_path / "aptos2019_splits.json"
    with open(splits_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"APTOS splits saved: train={len(train_ids)}, val={len(val_ids)}")
    print(f"Train distribution: {dict(Counter(train_labels))}")
    print(f"Val distribution: {dict(Counter(val_labels))}")

def create_grading_splits(processed_dir: str, splits_dir: str, val_size: float = 0.2, random_state: int = 42):
    """Create splits for IDRiD grading dataset."""
    
    processed_path = Path(processed_dir)
    splits_path = Path(splits_dir)
    
    # Check for grading data
    grading_images_dir = processed_path / "grading" / "images"
    grading_metadata_path = processed_path / "grading" / "metadata.json"
    
    if not grading_images_dir.exists():
        print("IDRiD grading images not found")
        return
    
    print("Creating IDRiD Disease Grading splits...")
    
    # Get image files
    image_files = [f for f in os.listdir(grading_images_dir) if f.endswith('.png')]
    image_files.sort()
    
    # Try to load metadata or find labels
    image_labels = {}
    
    if grading_metadata_path.exists():
        # Use metadata if available
        with open(grading_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get labels dictionary
        labels_dict = metadata['labels']['labels']
        
        # Get processing results from 'images' key
        if 'images' in metadata and isinstance(metadata['images'], list):
            for result in metadata['images']:
                if result.get('success', False):
                    processed_filename = Path(result['processed_image']).name
                    original_filename = result['original_filename']
                    
                    if processed_filename in image_files:
                        # Extract IDRiD ID from original filename
                        original_id = Path(original_filename).stem  # e.g., "IDRiD_043"
                        
                        if original_id in labels_dict:
                            img_id = Path(processed_filename).stem  # e.g., "grade_0005"
                            image_labels[img_id] = labels_dict[original_id]
    
    else:
        # Try to load from preprocessing results
        preprocessing_results_path = processed_path / "preprocessing_results.json"
        if preprocessing_results_path.exists():
            with open(preprocessing_results_path, 'r') as f:
                results = json.load(f)
            
            if 'grading' in results and 'labels_data' in results['grading']:
                labels_dict = results['grading']['labels_data'].get('labels', {})
                
                for img_file in image_files:
                    img_id = Path(img_file).stem
                    # Try to extract original ID from processed filename
                    if 'idrid_' in img_id.lower():
                        # Extract IDRiD ID
                        parts = img_id.split('_')
                        if len(parts) >= 2:
                            original_id = '_'.join(parts[-2:])  # e.g., "IDRiD_001"
                            if original_id in labels_dict:
                                image_labels[img_id] = labels_dict[original_id]
    
    if not image_labels:
        print("Could not find grading labels")
        return
    
    # Prepare data for splitting
    image_ids = list(image_labels.keys())
    labels = list(image_labels.values())
    
    print(f"Total images: {len(image_ids)}")
    print(f"Label distribution: {dict(Counter(labels))}")
    
    # Create stratified split
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        image_ids, labels,
        test_size=val_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Create splits dictionary
    splits = {
        'train': train_ids,
        'val': val_ids
    }
    
    # Save splits
    split_info = {
        'dataset_name': 'IDRiD Disease Grading',
        'random_state': random_state,
        'splits': splits,
        'split_sizes': {
            'train': len(train_ids),
            'val': len(val_ids)
        },
        'label_distribution': {
            'train': dict(Counter(train_labels)),
            'val': dict(Counter(val_labels))
        }
    }
    
    splits_file = splits_path / "idrid_grading_splits.json"
    with open(splits_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"IDRiD grading splits saved: train={len(train_ids)}, val={len(val_ids)}")

def create_segmentation_splits(processed_dir: str, splits_dir: str, val_size: float = 0.2, random_state: int = 42):
    """Create splits for segmentation dataset."""
    
    processed_path = Path(processed_dir)
    splits_path = Path(splits_dir)
    
    # Check for processed segmentation data
    seg_images_dir = processed_path / "segmentation" / "images"
    seg_masks_dir = processed_path / "segmentation" / "masks"
    
    if not seg_images_dir.exists() or not seg_masks_dir.exists():
        print("Processed segmentation data not found")
        return
    
    print("Creating IDRiD Segmentation splits...")
    
    # Get image files
    image_files = [f for f in os.listdir(seg_images_dir) if f.endswith('.png')]
    image_files.sort()
    
    # Create image-mask pairs based on naming convention
    processed_pairs = []
    
    for img_file in image_files:
        img_path = seg_images_dir / img_file
        
        # Find corresponding mask file
        img_stem = Path(img_file).stem  # e.g., "seg_0000"
        mask_file = f"{img_stem}_mask.png"  # e.g., "seg_0000_mask.png"
        mask_path = seg_masks_dir / mask_file
        
        if mask_path.exists():
            processed_pairs.append((str(img_path), str(mask_path)))
        else:
            print(f"⚠️ Missing mask for {img_file}")
    
    if not processed_pairs:
        print("Could not find processed segmentation pairs")
        return
    
    print(f"Found {len(processed_pairs)} segmentation pairs")
    
    # Create random split (stratification is complex for segmentation)
    np.random.seed(random_state)
    indices = np.random.permutation(len(processed_pairs))
    
    val_size_idx = int(len(indices) * val_size)
    val_indices = indices[:val_size_idx]
    train_indices = indices[val_size_idx:]
    
    train_pairs = [processed_pairs[i] for i in train_indices]
    val_pairs = [processed_pairs[i] for i in val_indices]
    
    # Create splits dictionary
    splits = {
        'train': [{'image': img, 'mask': mask} for img, mask in train_pairs],
        'val': [{'image': img, 'mask': mask} for img, mask in val_pairs]
    }
    
    # Save splits
    split_info = {
        'dataset_name': 'IDRiD Segmentation',
        'random_state': random_state,
        'splits': splits,
        'split_sizes': {
            'train': len(train_pairs),
            'val': len(val_pairs)
        }
    }
    
    splits_file = splits_path / "idrid_segmentation_splits.json"
    with open(splits_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Segmentation splits saved: train={len(train_pairs)}, val={len(val_pairs)}")

def main():
    """Create all splits."""
    # Use relative paths from project root
    project_root = Path(__file__).parent.parent.parent
    processed_dir = str(project_root / "dataset" / "processed")
    splits_dir = str(project_root / "dataset" / "splits")
    
    print("Creating Data Splits for Processed Data")
    print("=" * 50)
    
    # Create splits directory
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    
    # Create all splits
    create_aptos_splits(processed_dir, splits_dir, val_size=0.2, random_state=42)
    create_grading_splits(processed_dir, splits_dir, val_size=0.2, random_state=42)
    create_segmentation_splits(processed_dir, splits_dir, val_size=0.2, random_state=42)
    
    print(f"\nAll splits created in: {splits_dir}")

if __name__ == "__main__":
    main()
