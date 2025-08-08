"""
Project transforms for diabetic retinopathy datasets using Albumentations.
Provides: get_training_transforms, get_validation_transforms, get_segmentation_transforms
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def _norm():
    # Standard ImageNet normalization
    return A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def _cutout_or_dropout(
    *,
    image_size: int | None = None,
    num_holes: int = 4,
    max_size_frac: float | None = 0.08,
    max_size_px: int | None = None,
    p: float = 0.2,
):
    """
    Version-compatible regularization: prefer A.Cutout when available,
    otherwise fall back to A.CoarseDropout with roughly equivalent parameters.
    """
    if hasattr(A, "Cutout"):
        # Compute max sizes
        if max_size_px is None:
            if image_size is None or max_size_frac is None:
                raise ValueError("Provide image_size and max_size_frac or set max_size_px")
            max_size_px = int(max_size_frac * image_size)
        return A.Cutout(num_holes=num_holes, max_h_size=max_size_px, max_w_size=max_size_px, fill_value=0, p=p)
    # Fallback: version-agnostic custom cutout using Lambda
    if max_size_px is None:
        if image_size is None or max_size_frac is None:
            raise ValueError("Provide image_size and max_size_frac or set max_size_px")
        max_size_px = int(max_size_frac * image_size)

    def _apply_cutout_np(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        for _ in range(num_holes):
            ch = np.random.randint(1, max(2, max_size_px + 1))
            cw = np.random.randint(1, max(2, max_size_px + 1))
            top = np.random.randint(0, max(1, h - ch + 1))
            left = np.random.randint(0, max(1, w - cw + 1))
            img[top:top + ch, left:left + cw] = 0
        return img

    return A.Lambda(image=_apply_cutout_np, p=p)


def get_training_transforms(image_size: int = 512):
    """Augmentations for training classification/multitask images."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.RandomCrop(height=image_size, width=image_size, p=0.2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        # Replace deprecated/ warned transform with recommended Affine
        A.Affine(
            rotate=(-20, 20),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            p=0.5,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
    # Version-compatible regularization op
    _cutout_or_dropout(image_size=image_size, num_holes=4, max_size_frac=0.08, p=0.2),
        _norm(),
        ToTensorV2(),
    ])


def get_validation_transforms(image_size: int = 512):
    """Light, deterministic preprocessing for validation/test."""
    return A.Compose([
        A.Resize(image_size, image_size),
        _norm(),
        ToTensorV2(),
    ])


def get_segmentation_transforms(image_size: int = 512):
    """Segmentation transforms; keep geometry consistent for image+mask."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        _norm(),
        ToTensorV2(),
    ])
"""
Data Augmentation Pipelines for Diabetic Retinopathy Multi-Task Learning
Uses albumentations for efficient and robust augmentation strategies.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Callable, Tuple, Optional

# ImageNet normalization values (commonly used for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms_classification(image_size: int = 512) -> Callable:
    """
    Get training augmentation pipeline for classification tasks (APTOS, Disease Grading).
    
    Args:
        image_size: Target image size for resizing
        
    Returns:
        Albumentations compose transform pipeline
    """
    return A.Compose([
        # Resize to target size first
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
        
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),  # Medical images can be rotated
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.1, 
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.6
        ),
        
        # Optical distortions (subtle for medical images)
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
        
        # Color augmentations (careful with medical images)
        A.RandomBrightnessContrast(
            brightness_limit=0.15, 
            contrast_limit=0.15, 
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.4
        ),
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.4
        ),
        
        # Noise and blur (simulate real-world conditions)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # CLAHE enhancement (good for retinal images)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        
        # Cutout/Dropout for regularization
    _cutout_or_dropout(image_size=image_size, num_holes=8, max_size_px=32, p=0.3),
        
        # Normalization and tensor conversion
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transforms_classification(image_size: int = 512) -> Callable:
    """
    Get validation transforms for classification tasks (minimal augmentation).
    
    Args:
        image_size: Target image size for resizing
        
    Returns:
        Albumentations compose transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # Always apply CLAHE
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_train_transforms_segmentation(image_size: int = 512) -> Callable:
    """
    Get training augmentation pipeline for segmentation tasks (IDRiD/Segmentation).
    Important: All transforms must be applied identically to both image and mask!
    
    Args:
        image_size: Target image size for resizing
        
    Returns:
        Albumentations compose transform pipeline
    """
    return A.Compose([
        # Resize both image and mask
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
        
        # Geometric transformations (applied to both image and mask)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1, 
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.6
        ),
        
        # Elastic transform (subtle for medical segmentation)
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            alpha_affine=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.3
        ),
        
        # Grid distortion (very subtle)
        A.GridDistortion(
            num_steps=3,
            distort_limit=0.05,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.2
        ),
        
        # Color augmentations (only applied to image, not mask)
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=10,
            val_shift_limit=5,
            p=0.3
        ),
        
        # Noise (only applied to image)
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.3), p=1.0),
        ], p=0.2),
        
        # CLAHE enhancement (only for image)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        
        # Normalization (only for image) and tensor conversion
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transforms_segmentation(image_size: int = 512) -> Callable:
    """
    Get validation transforms for segmentation tasks (minimal augmentation).
    
    Args:
        image_size: Target image size for resizing
        
    Returns:
        Albumentations compose transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # Always apply CLAHE
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_test_time_augmentation(image_size: int = 512, n_augments: int = 4) -> list:
    """
    Get multiple augmentation pipelines for test-time augmentation (TTA).
    
    Args:
        image_size: Target image size for resizing
        n_augments: Number of different augmentation pipelines
        
    Returns:
        List of augmentation pipelines
    """
    augmentations = []
    
    # Original (no augmentation)
    augmentations.append(A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]))
    
    # Horizontal flip
    if n_augments > 1:
        augmentations.append(A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]))
    
    # Vertical flip
    if n_augments > 2:
        augmentations.append(A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.VerticalFlip(p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]))
    
    # Both flips
    if n_augments > 3:
        augmentations.append(A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]))
    
    # Additional rotations if more augments requested
    for i in range(4, n_augments):
        angle = 90 * (i - 3)  # 90, 180, 270 degree rotations
        augmentations.append(A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.Rotate(limit=[angle, angle], border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]))
    
    return augmentations

def denormalize_tensor(tensor: np.ndarray, mean: list = IMAGENET_MEAN, std: list = IMAGENET_STD) -> np.ndarray:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor (C, H, W)
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Denormalized tensor
    """
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    return tensor * std + mean

def visualize_augmentations(dataset, num_samples: int = 4, save_path: Optional[str] = None):
    """
    Visualize augmentation effects on sample images.
    
    Args:
        dataset: PyTorch dataset with augmentations
        num_samples: Number of samples to visualize
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import torch
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Get sample
        if hasattr(dataset, 'get_item_with_id'):
            sample = dataset.get_item_with_id(i)
        else:
            sample = dataset[i]
        
        if isinstance(sample, tuple):
            image, target = sample
        else:
            image = sample['image']
            target = sample.get('mask', sample.get('label', None))
        
        # Denormalize image for visualization
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
        else:
            image_np = image
            
        image_vis = denormalize_tensor(image_np)
        image_vis = np.transpose(image_vis, (1, 2, 0))
        image_vis = np.clip(image_vis, 0, 1)
        
        # Plot image
        axes[0, i].imshow(image_vis)
        axes[0, i].set_title(f'Sample {i+1}')
        axes[0, i].axis('off')
        
        # Plot target (mask or label)
        if target is not None:
            if isinstance(target, torch.Tensor) and len(target.shape) > 0:
                if len(target.shape) == 3:  # Mask
                    mask_vis = target.numpy() if isinstance(target, torch.Tensor) else target
                    axes[1, i].imshow(mask_vis.squeeze(), cmap='gray')
                    axes[1, i].set_title(f'Mask {i+1}')
                else:  # Label
                    axes[1, i].text(0.5, 0.5, f'Label: {target}', 
                                   transform=axes[1, i].transAxes, 
                                   ha='center', va='center', fontsize=12)
                    axes[1, i].set_title(f'Label: {target}')
            else:
                axes[1, i].text(0.5, 0.5, f'Label: {target}', 
                               transform=axes[1, i].transAxes, 
                               ha='center', va='center', fontsize=12)
                axes[1, i].set_title(f'Label: {target}')
        
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

# Transform configurations for different tasks
TRANSFORM_CONFIGS = {
    'classification_train': {
        'image_size': 512,
        'transform_func': get_train_transforms_classification
    },
    'classification_val': {
        'image_size': 512,
        'transform_func': get_val_transforms_classification
    },
    'segmentation_train': {
        'image_size': 512,
        'transform_func': get_train_transforms_segmentation
    },
    'segmentation_val': {
        'image_size': 512,
        'transform_func': get_val_transforms_segmentation
    }
}

def get_transforms(task: str, split: str, image_size: int = 512) -> Callable:
    """
    Get transforms for a specific task and split.
    
    Args:
        task: 'classification' or 'segmentation'
        split: 'train' or 'val'
        image_size: Target image size
        
    Returns:
        Appropriate transform pipeline
    """
    config_key = f"{task}_{split}"
    if config_key not in TRANSFORM_CONFIGS:
        raise ValueError(f"Unknown task/split combination: {config_key}")
    
    transform_func = TRANSFORM_CONFIGS[config_key]['transform_func']
    return transform_func(image_size=image_size)
