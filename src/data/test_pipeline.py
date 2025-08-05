"""
Visual Sanity Check and Testing for Data Pipeline
Validates augmentations, dataset functionality, and DataLoader performance.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

def get_project_paths():
    """Get standardized project paths."""
    root = Path(__file__).parent.parent.parent
    return {
        'processed_dir': str(root / "dataset" / "processed"),
        'splits_dir': str(root / "dataset" / "splits"), 
        'viz_dir': str(root / "visualizations")
    }

from src.data.datasets import GradingRetinaDataset, SegmentationRetinaDataset, MultiTaskRetinaDataset
from src.data.transforms import (
    get_transforms, denormalize_tensor, visualize_augmentations,
    IMAGENET_MEAN, IMAGENET_STD
)
from src.data.dataloaders import DataLoaderFactory, create_all_dataloaders
from src.data.data_splits import create_all_splits

def test_transforms_pipeline():
    """Test different transform pipelines."""
    print("\nTesting Testing Transform Pipelines...")
    
    # Test all transform types
    transform_types = [
        ('classification', 'train'),
        ('classification', 'val'),
        ('segmentation', 'train'),
        ('segmentation', 'val')
    ]
    
    for task, split in transform_types:
        try:
            transform = get_transforms(task, split, image_size=512)
            print(f"OK {task} {split} transforms: OK")
            
            # Test with dummy data
            dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            if task == 'segmentation':
                dummy_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
                result = transform(image=dummy_image, mask=dummy_mask)
                print(f"   - Image shape: {result['image'].shape}")
                print(f"   - Mask shape: {result['mask'].shape}")
            else:
                result = transform(image=dummy_image)
                print(f"   - Image shape: {result['image'].shape}")
                
        except Exception as e:
            print(f"ERROR {task} {split} transforms: {e}")

def test_dataset_loading():
    """Test dataset loading and basic functionality."""
    print("\nTesting Dataset Loading...")
    
    paths = get_project_paths()
    processed_dir = paths['processed_dir']
    
    # Test APTOS dataset
    aptos_images_dir = Path(processed_dir) / "aptos2019" / "images"
    aptos_labels_path = Path(processed_dir).parent / "aptos2019-blindness-detection" / "train.csv"
    
    if aptos_images_dir.exists() and aptos_labels_path.exists():
        try:
            transform = get_transforms('classification', 'train', 512)
            dataset = GradingRetinaDataset(
                str(aptos_images_dir),
                str(aptos_labels_path),
                transform=transform
            )
            
            print(f"OK APTOS dataset: {len(dataset)} samples")
            
            # Test loading a sample
            if len(dataset) > 0:
                image, label = dataset[0]
                print(f"   - Sample shape: {image.shape}, label: {label}")
                
        except Exception as e:
            print(f"ERROR APTOS dataset: {e}")
    else:
        print("SKIP APTOS dataset: Data not found")
    
    # Test segmentation dataset
    splits_dir = Path(processed_dir).parent / "data" / "splits"
    seg_splits_path = splits_dir / "idrid_segmentation_splits.json"
    
    if seg_splits_path.exists():
        try:
            with open(seg_splits_path, 'r') as f:
                split_info = json.load(f)
            
            if 'train' in split_info['splits']:
                splits = split_info['splits']['train']
                
                # Convert to proper format
                if isinstance(splits[0], dict):
                    pairs = [(item['image'], item['mask']) for item in splits[:5]]  # Test first 5
                else:
                    pairs = splits[:5]
                
                transform = get_transforms('segmentation', 'train', 512)
                dataset = SegmentationRetinaDataset(
                    "",  # Not used
                    "",  # Not used
                    transform=transform,
                    image_mask_pairs=pairs
                )
                
                print(f"OK Segmentation dataset: {len(dataset)} pairs")
                
                # Test loading a sample
                if len(dataset) > 0:
                    image, mask = dataset[0]
                    print(f"   - Image shape: {image.shape}, mask shape: {mask.shape}")
                    print(f"   - Mask unique values: {torch.unique(mask)}")
                    
        except Exception as e:
            print(f"ERROR Segmentation dataset: {e}")
    else:
        print("SKIP Segmentation dataset: Splits not found")

def visualize_sample_augmentations(save_dir: str = "visualizations"):
    """Visualize augmentation effects on sample data."""
    print("\nVisualizing Sample Augmentations...")
    
    os.makedirs(save_dir, exist_ok=True)
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    
    # 1. Classification augmentations (APTOS)
    aptos_images_dir = processed_dir / "aptos2019" / "images"
    aptos_labels_path = processed_dir.parent / "aptos2019-blindness-detection" / "train.csv"
    
    if aptos_images_dir.exists() and aptos_labels_path.exists():
        try:
            # Create dataset with training transforms
            train_transform = get_transforms('classification', 'train', 512)
            val_transform = get_transforms('classification', 'val', 512)
            
            train_dataset = GradingRetinaDataset(
                str(aptos_images_dir),
                str(aptos_labels_path),
                transform=train_transform
            )
            
            val_dataset = GradingRetinaDataset(
                str(aptos_images_dir),
                str(aptos_labels_path),
                transform=val_transform
            )
            
            # Visualize training augmentations
            if len(train_dataset) > 0:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                
                # Show same image with different augmentations
                for i in range(4):
                    # Training (augmented)
                    image_train, label_train = train_dataset[0]  # Same image, different augmentations
                    img_train_vis = denormalize_tensor(image_train.numpy())
                    img_train_vis = np.transpose(img_train_vis, (1, 2, 0))
                    img_train_vis = np.clip(img_train_vis, 0, 1)
                    
                    axes[0, i].imshow(img_train_vis)
                    axes[0, i].set_title(f'Train Aug {i+1}\nLabel: {label_train}')
                    axes[0, i].axis('off')
                    
                    # Validation (minimal augmentation)
                    image_val, label_val = val_dataset[0]
                    img_val_vis = denormalize_tensor(image_val.numpy())
                    img_val_vis = np.transpose(img_val_vis, (1, 2, 0))
                    img_val_vis = np.clip(img_val_vis, 0, 1)
                    
                    axes[1, i].imshow(img_val_vis)
                    axes[1, i].set_title(f'Val Aug {i+1}\nLabel: {label_val}')
                    axes[1, i].axis('off')
                
                plt.suptitle('Classification Augmentations: Training vs Validation')
                plt.tight_layout()
                plt.savefig(f"{save_dir}/classification_augmentations.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"OK Classification augmentations saved to {save_dir}")
                
        except Exception as e:
            print(f"ERROR Classification visualization: {e}")
    
    # 2. Segmentation augmentations
    splits_dir = Path(processed_dir).parent / "data" / "splits"
    seg_splits_path = splits_dir / "idrid_segmentation_splits.json"
    
    if seg_splits_path.exists():
        try:
            with open(seg_splits_path, 'r') as f:
                split_info = json.load(f)
            
            if 'train' in split_info['splits']:
                splits = split_info['splits']['train']
                
                # Convert to proper format
                if isinstance(splits[0], dict):
                    pairs = [(item['image'], item['mask']) for item in splits[:2]]
                else:
                    pairs = splits[:2]
                
                train_transform = get_transforms('segmentation', 'train', 512)
                val_transform = get_transforms('segmentation', 'val', 512)
                
                train_dataset = SegmentationRetinaDataset(
                    "", "", transform=train_transform, image_mask_pairs=pairs
                )
                val_dataset = SegmentationRetinaDataset(
                    "", "", transform=val_transform, image_mask_pairs=pairs
                )
                
                if len(train_dataset) > 0:
                    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                    
                    for i in range(4):
                        # Training augmented
                        image_train, mask_train = train_dataset[0]
                        
                        img_vis = denormalize_tensor(image_train.numpy())
                        img_vis = np.transpose(img_vis, (1, 2, 0))
                        img_vis = np.clip(img_vis, 0, 1)
                        
                        mask_vis = mask_train.numpy() if isinstance(mask_train, torch.Tensor) else mask_train
                        
                        # Show image
                        axes[0, i].imshow(img_vis)
                        axes[0, i].set_title(f'Train Image {i+1}')
                        axes[0, i].axis('off')
                        
                        # Show mask
                        axes[1, i].imshow(mask_vis, cmap='gray')
                        axes[1, i].set_title(f'Train Mask {i+1}')
                        axes[1, i].axis('off')
                        
                        # Show overlay
                        overlay = img_vis.copy()
                        if len(mask_vis.shape) == 2:
                            mask_colored = np.zeros_like(overlay)
                            mask_colored[:, :, 0] = mask_vis  # Red channel
                            overlay = 0.7 * overlay + 0.3 * mask_colored
                        
                        axes[2, i].imshow(overlay)
                        axes[2, i].set_title(f'Overlay {i+1}')
                        axes[2, i].axis('off')
                    
                    plt.suptitle('Segmentation Augmentations: Image-Mask Alignment')
                    plt.tight_layout()
                    plt.savefig(f"{save_dir}/segmentation_augmentations.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"OK Segmentation augmentations saved to {save_dir}")
                    
        except Exception as e:
            print(f"ERROR Segmentation visualization: {e}")

def test_dataloader_performance():
    """Test DataLoader performance and memory usage."""
    print("\nTesting DataLoader Performance...")
    
    try:
        # Create data splits first
        processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        splits_dir = Path(__file__).parent.parent.parent / "data" / "splits"
        
        # Create splits if they don't exist
        if not Path(splits_dir).exists():
            print("Creating data splits...")
            create_all_splits(processed_dir, splits_dir, val_size=0.2, random_state=42)
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            print(f"\nTesting Testing batch_size={batch_size}")
            
            try:
                # Create factory
                factory = DataLoaderFactory(
                    batch_size=batch_size,
                    num_workers=2,
                    pin_memory=True
                )
                
                # Test APTOS DataLoader
                aptos_images_dir = Path(processed_dir) / "aptos2019" / "images"
                aptos_labels_path = Path(processed_dir).parent / "aptos2019-blindness-detection" / "train.csv"
                aptos_splits_path = Path(splits_dir) / "aptos2019_splits.json"
                
                if all(p.exists() for p in [aptos_images_dir, aptos_labels_path, aptos_splits_path]):
                    dataloaders = factory.create_classification_dataloaders(
                        str(aptos_images_dir),
                        str(aptos_labels_path),
                        str(aptos_splits_path),
                        image_size=512
                    )
                    
                    # Time a few batches
                    if 'train' in dataloaders:
                        dataloader = dataloaders['train']
                        
                        start_time = time.time()
                        for i, (images, labels) in enumerate(dataloader):
                            if i >= 5:  # Test first 5 batches
                                break
                            
                            # Simulate some processing
                            _ = images.shape, labels.shape
                        
                        elapsed = time.time() - start_time
                        avg_time = elapsed / min(5, len(dataloader))
                        
                        print(f"   - APTOS train: {avg_time:.3f}s/batch")
                        print(f"   - Batch shape: {images.shape}")
                        print(f"   - Labels shape: {labels.shape}")
                
            except Exception as e:
                print(f"   ERROR Batch size {batch_size}: {e}")
                
    except Exception as e:
        print(f"ERROR DataLoader performance test: {e}")

def check_class_distribution():
    """Check class distribution in datasets and splits."""
    print("\nChecking Class Distribution...")
    
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    splits_dir = Path(__file__).parent.parent.parent / "data" / "splits"
    
    # Check APTOS distribution
    aptos_labels_path = processed_dir.parent / "aptos2019-blindness-detection" / "train.csv"
    aptos_splits_path = splits_dir / "aptos2019_splits.json"
    
    if aptos_labels_path.exists() and aptos_splits_path.exists():
        try:
            import pandas as pd
            from collections import Counter
            
            # Load labels
            df = pd.read_csv(aptos_labels_path)
            overall_dist = Counter(df['diagnosis'])
            
            # Load splits
            with open(aptos_splits_path, 'r') as f:
                split_info = json.load(f)
            
            print("Distribution APTOS 2019 Class Distribution:")
            print(f"   Overall: {dict(overall_dist)}")
            
            # Check split distributions
            for split_name in ['train', 'val']:
                if split_name in split_info['splits']:
                    split_ids = split_info['splits'][split_name]
                    split_df = df[df['id_code'].isin(split_ids)]
                    split_dist = Counter(split_df['diagnosis'])
                    print(f"   {split_name}: {dict(split_dist)}")
                    
        except Exception as e:
            print(f"ERROR APTOS distribution check: {e}")

def check_mask_alignment():
    """Check image-mask alignment in segmentation data."""
    print("\nChecking Image-Mask Alignment...")
    
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    splits_dir = processed_dir.parent / "data" / "splits"
    seg_splits_path = splits_dir / "idrid_segmentation_splits.json"
    
    if seg_splits_path.exists():
        try:
            with open(seg_splits_path, 'r') as f:
                split_info = json.load(f)
            
            if 'train' in split_info['splits']:
                splits = split_info['splits']['train']
                
                # Test first few pairs
                test_pairs = splits[:3] if len(splits) >= 3 else splits
                
                # Create dataset without transforms first
                if isinstance(test_pairs[0], dict):
                    pairs = [(item['image'], item['mask']) for item in test_pairs]
                else:
                    pairs = test_pairs
                
                dataset = SegmentationRetinaDataset(
                    "", "", transform=None, image_mask_pairs=pairs
                )
                
                alignment_issues = 0
                
                for i in range(len(dataset)):
                    try:
                        image, mask = dataset[i]
                        
                        # Check shapes match
                        if image.shape[:2] != mask.shape:
                            print(f"   WARNING Size mismatch in pair {i}: image {image.shape[:2]} vs mask {mask.shape}")
                            alignment_issues += 1
                        
                        # Check if mask has content
                        if isinstance(mask, np.ndarray):
                            mask_content = np.sum(mask > 127)
                        else:
                            mask_content = torch.sum(mask > 0.5).item()
                        
                        print(f"   Pair {i}: image {image.shape[:2]}, mask pixels: {mask_content}")
                        
                    except Exception as e:
                        print(f"   ERROR Error in pair {i}: {e}")
                        alignment_issues += 1
                
                if alignment_issues == 0:
                    print("OK All masks properly aligned")
                else:
                    print(f"WARNING Found {alignment_issues} alignment issues")
                    
        except Exception as e:
            print(f"ERROR Mask alignment check: {e}")

def run_full_sanity_check():
    """Run complete sanity check pipeline."""
    print("COMPREHENSIVE DATA PIPELINE SANITY CHECK")
    print("=" * 60)
    
    # Create visualization directory
    viz_dir = Path(__file__).parent.parent.parent / "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Run all checks
    test_transforms_pipeline()
    test_dataset_loading()
    visualize_sample_augmentations(viz_dir)
    test_dataloader_performance()
    check_class_distribution()
    check_mask_alignment()
    
    print("\nOK Sanity check complete!")
    print(f" Visualizations saved to: {viz_dir}")

if __name__ == "__main__":
    run_full_sanity_check()
