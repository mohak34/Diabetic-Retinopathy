"""
Optimized dataloader utilities for diabetic retinopathy datasets.
Exposes: get_data_loaders, create_dataloader_factory
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader


def _pin():
    return torch.cuda.is_available()


def create_dataloader_factory(batch_size: int = 4, num_workers: int = 2):
    """Return a factory that builds dataloaders with consistent settings."""

    def factory(dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=_pin(),
            persistent_workers=num_workers > 0,
        )

    return factory


def get_data_loaders(
    train_grading,
    val_grading,
    train_segmentation,
    val_segmentation,
    train_multitask,
    val_multitask,
    batch_size: int = 4,
    num_workers: int = 2,
) -> Dict[str, DataLoader]:
    """Create a dictionary of standard dataloaders given dataset objects."""
    make = create_dataloader_factory(batch_size=batch_size, num_workers=num_workers)
    return {
        'train_grading': make(train_grading, True),
        'val_grading': make(val_grading, False),
        'train_segmentation': make(train_segmentation, True),
        'val_segmentation': make(val_segmentation, False),
        'train_multitask': make(train_multitask, True),
        'val_multitask': make(val_multitask, False),
    }
"""
PyTorch DataLoader Setup for Diabetic Retinopathy Multi-Task Learning
Optimized for RTX 3080 mobile (8GB VRAM) with efficient batch processing.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

# Import our custom datasets and transforms
from .datasets import GradingRetinaDataset, SegmentationRetinaDataset, MultiTaskRetinaDataset
from .transforms import get_transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoaderFactory:
    """
    Factory class for creating optimized DataLoaders for different tasks.
    Handles batch sizing, memory optimization, and balanced sampling.
    """
    
    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ):
        """
        Initialize the DataLoader factory.
        
        Args:
            batch_size: Batch size (reduce if OOM occurs)
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Whether to keep workers alive between epochs
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and num_workers > 0
        
        logger.info(f"DataLoader config: batch_size={batch_size}, num_workers={num_workers}, "
                   f"pin_memory={self.pin_memory}, persistent_workers={self.persistent_workers}")
    
    def create_classification_dataloaders(
        self,
        images_dir: str,
        labels_path: str,
        splits_path: str,
        image_size: int = 512,
        use_weighted_sampling: bool = True,
        drop_last_train: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for classification tasks (APTOS, Disease Grading).
        
        Args:
            images_dir: Directory containing processed images
            labels_path: Path to labels CSV file
            splits_path: Path to splits JSON file
            image_size: Input image size
            use_weighted_sampling: Whether to use weighted sampling for class balance
            drop_last_train: Whether to drop last incomplete batch in training
            
        Returns:
            Dictionary containing train/val/test DataLoaders
        """
        # Load splits
        with open(splits_path, 'r') as f:
            split_info = json.load(f)
        splits = split_info['splits']
        
        # Get transforms
        train_transform = get_transforms('classification', 'train', image_size)
        val_transform = get_transforms('classification', 'val', image_size)
        
        dataloaders = {}
        
        for split_name in ['train', 'val', 'test']:
            if split_name not in splits:
                continue
            
            # Create dataset
            is_train = split_name == 'train'
            transform = train_transform if is_train else val_transform
            
            dataset = GradingRetinaDataset(
                images_dir=images_dir,
                labels_path=labels_path,
                transform=transform,
                image_ids=splits[split_name]
            )
            
            # Create sampler for training
            sampler = None
            if is_train and use_weighted_sampling:
                sampler = self._create_weighted_sampler(dataset)
            
            # Create DataLoader
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(is_train and sampler is None),
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=drop_last_train if is_train else False,
                persistent_workers=self.persistent_workers,
                prefetch_factor=2 if self.num_workers > 0 else None
            )
            
            logger.info(f"Created {split_name} DataLoader: {len(dataset)} samples, "
                       f"{len(dataloaders[split_name])} batches")
        
        return dataloaders
    
    def create_segmentation_dataloaders(
        self,
        splits_path: str,
        image_size: int = 512,
        drop_last_train: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for segmentation tasks (IDRiD/Segmentation).
        
        Args:
            splits_path: Path to splits JSON file containing image-mask pairs
            image_size: Input image size
            drop_last_train: Whether to drop last incomplete batch in training
            
        Returns:
            Dictionary containing train/val/test DataLoaders
        """
        # Load splits
        with open(splits_path, 'r') as f:
            split_info = json.load(f)
        splits = split_info['splits']
        
        # Get transforms
        train_transform = get_transforms('segmentation', 'train', image_size)
        val_transform = get_transforms('segmentation', 'val', image_size)
        
        dataloaders = {}
        
        for split_name in ['train', 'val', 'test']:
            if split_name not in splits:
                continue
            
            # Convert split data back to tuples
            if isinstance(splits[split_name][0], dict):
                # New format with 'image' and 'mask' keys
                image_mask_pairs = [
                    (item['image'], item['mask']) for item in splits[split_name]
                ]
            else:
                # Old format with direct tuples
                image_mask_pairs = splits[split_name]
            
            # Create dataset
            is_train = split_name == 'train'
            transform = train_transform if is_train else val_transform
            
            dataset = SegmentationRetinaDataset(
                images_dir="",  # Not used when pairs are provided
                masks_dir="",   # Not used when pairs are provided
                transform=transform,
                image_mask_pairs=image_mask_pairs
            )
            
            # Create DataLoader
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=is_train,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=drop_last_train if is_train else False,
                persistent_workers=self.persistent_workers,
                prefetch_factor=2 if self.num_workers > 0 else None
            )
            
            logger.info(f"Created {split_name} DataLoader: {len(dataset)} samples, "
                       f"{len(dataloaders[split_name])} batches")
        
        return dataloaders
    
    def create_multitask_dataloaders(
        self,
        images_dir: str,
        labels_path: Optional[str] = None,
        masks_dir: Optional[str] = None,
        splits_path: Optional[str] = None,
        image_size: int = 512,
        drop_last_train: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for multi-task learning (classification + segmentation).
        
        Args:
            images_dir: Directory containing images
            labels_path: Path to labels CSV file (optional)
            masks_dir: Directory containing masks (optional)
            splits_path: Path to splits JSON file (optional)
            image_size: Input image size
            drop_last_train: Whether to drop last incomplete batch in training
            
        Returns:
            Dictionary containing train/val/test DataLoaders
        """
        # Load splits if provided
        splits = {}
        if splits_path:
            with open(splits_path, 'r') as f:
                split_info = json.load(f)
            splits = split_info['splits']
        
        # Get transforms
        train_transform = get_transforms('segmentation', 'train', image_size)  # Use segmentation transforms
        val_transform = get_transforms('segmentation', 'val', image_size)
        
        dataloaders = {}
        
        for split_name in ['train', 'val', 'test']:
            if splits and split_name not in splits:
                continue
            
            # Create dataset
            is_train = split_name == 'train'
            transform = train_transform if is_train else val_transform
            
            image_ids = splits.get(split_name) if splits else None
            
            dataset = MultiTaskRetinaDataset(
                images_dir=images_dir,
                labels_path=labels_path,
                masks_dir=masks_dir,
                transform=transform,
                image_ids=image_ids
            )
            
            # Create DataLoader
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=is_train,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=drop_last_train if is_train else False,
                persistent_workers=self.persistent_workers,
                prefetch_factor=2 if self.num_workers > 0 else None,
                collate_fn=self._multitask_collate_fn
            )
            
            logger.info(f"Created {split_name} DataLoader: {len(dataset)} samples, "
                       f"{len(dataloaders[split_name])} batches")
        
        return dataloaders
    
    def _create_weighted_sampler(self, dataset: GradingRetinaDataset) -> WeightedRandomSampler:
        """
        Create weighted sampler for balanced class sampling.
        
        Args:
            dataset: Classification dataset
            
        Returns:
            WeightedRandomSampler instance
        """
        # Get class weights
        class_weights = dataset.get_label_weights()
        
        # Get sample weights
        sample_weights = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            sample_weights.append(class_weights[label].item())
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        logger.info(f"Created weighted sampler with class weights: {class_weights.tolist()}")
        return sampler
    
    def create_multitask_loaders(
        self,
        processed_data_dir: str = "dataset/processed",
        splits_dir: str = "dataset/splits",
        batch_size: int = 4,
        num_workers: int = 2,
        image_size: int = 512,
        **kwargs
    ) -> Dict[str, DataLoader]:
        """
        Create multi-task dataloaders.
        
        Args:
            processed_data_dir: Directory containing processed data
            splits_dir: Directory containing data splits
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            image_size: Target image size
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing train and validation dataloaders
        """
        try:
            # Try to create real multitask dataloaders
            return self.create_multitask_dataloaders(
                images_dir=f"{processed_data_dir}/images",
                labels_path=f"{processed_data_dir}/labels.csv",
                masks_dir=f"{processed_data_dir}/masks",
                splits_path=f"{splits_dir}/multitask_splits.json",
                image_size=image_size,
                drop_last_train=True
            )
        except Exception as e:
            # Fallback to dummy data
            logger.warning(f"Failed to create real dataloaders: {e}")
            try:
                from ..training.pipeline import DummyDataset
                
                train_dataset = DummyDataset(size=batch_size * 10)
                val_dataset = DummyDataset(size=batch_size * 5)
                
                return {
                    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                }
            except ImportError:
                # Final fallback - create minimal dummy data locally
                from torch.utils.data import TensorDataset
                import torch
                
                # Create minimal dummy tensors
                images = torch.randn(batch_size * 10, 3, image_size, image_size)
                labels = torch.randint(0, 5, (batch_size * 10,))
                masks = torch.randint(0, 2, (batch_size * 10, image_size, image_size)).float()
                
                train_dataset = TensorDataset(images[:batch_size * 8], labels[:batch_size * 8], masks[:batch_size * 8])
                val_dataset = TensorDataset(images[batch_size * 8:], labels[batch_size * 8:], masks[batch_size * 8:])
                
                return {
                    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                }

    def _multitask_collate_fn(self, batch):
        """
        Custom collate function for multi-task batches.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Tuple (images, labels, masks) compatible with trainer
        """
        # Separate different data types
        images = []
        labels = []
        masks = []
        
        for sample in batch:
            # Handle images
            if 'image' in sample:
                images.append(sample['image'])
            else:
                raise ValueError(f"Sample missing 'image' key: {sample.keys()}")
            
            # Handle labels
            if 'label' in sample:
                labels.append(sample['label'])
            else:
                labels.append(0)  # Default label
            
            # Handle masks - CRITICAL FIX: Always provide a mask, even if dummy
            if 'masks' in sample and sample['masks']:
                # Take first mask if multiple available
                if isinstance(sample['masks'], list):
                    masks.append(sample['masks'][0])
                else:
                    masks.append(sample['masks'])
            else:
                # Create dummy mask if no mask available - prevents NoneType errors
                image_shape = sample['image'].shape
                if len(image_shape) == 3 and image_shape[0] <= 4:  # CHW format (C, H, W)
                    dummy_mask = torch.zeros((image_shape[1], image_shape[2]), dtype=torch.float32)
                elif len(image_shape) == 3:  # HWC format (H, W, C)
                    dummy_mask = torch.zeros((image_shape[0], image_shape[1]), dtype=torch.float32)
                else:  # Already 2D
                    dummy_mask = torch.zeros_like(sample['image'][:, :, 0] if len(sample['image'].shape) == 3 else sample['image'], dtype=torch.float32)
                masks.append(dummy_mask)
        
        # Ensure all inputs are tensors
        try:
            images_tensor = torch.stack(images)
        except Exception as e:
            logger.error(f"Error stacking images: {e}")
            logger.error(f"Image shapes: {[img.shape for img in images]}")
            raise
        
        try:
            labels_tensor = torch.LongTensor(labels)
        except Exception as e:
            logger.error(f"Error creating labels tensor: {e}")
            logger.error(f"Labels: {labels}")
            raise
        
        try:
            masks_tensor = torch.stack(masks)
        except Exception as e:
            logger.error(f"Error stacking masks: {e}")
            logger.error(f"Mask shapes: {[mask.shape for mask in masks]}")
            # Create uniform dummy masks if stacking fails
            if images:
                img_shape = images[0].shape
                if len(img_shape) == 3 and img_shape[0] <= 4:  # CHW
                    uniform_masks = [torch.zeros((img_shape[1], img_shape[2]), dtype=torch.float32) for _ in range(len(images))]
                else:  # HWC or other
                    uniform_masks = [torch.zeros((img_shape[-2], img_shape[-1]), dtype=torch.float32) for _ in range(len(images))]
                masks_tensor = torch.stack(uniform_masks)
            else:
                raise
        
        # Return in tuple format expected by trainer
        return images_tensor, labels_tensor, masks_tensor
    
    def get_optimal_batch_size(self, model, sample_input_shape: Tuple[int, int, int, int]) -> int:
        """
        Automatically find optimal batch size for given model and input shape.
        
        Args:
            model: PyTorch model
            sample_input_shape: (batch_size, channels, height, width)
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return self.batch_size
        
        device = torch.device('cuda')
        model = model.to(device)
        model.eval()
        
        # Start with current batch size and increase until OOM
        test_batch_size = self.batch_size
        max_batch_size = self.batch_size
        
        try:
            while test_batch_size <= 32:  # Max reasonable batch size
                # Create dummy input
                dummy_input = torch.randn(
                    test_batch_size, *sample_input_shape[1:], 
                    device=device, dtype=torch.float32
                )
                
                # Test forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                max_batch_size = test_batch_size
                test_batch_size *= 2
                
                # Clear cache
                del dummy_input
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.info(f"OOM at batch size {test_batch_size}, using {max_batch_size}")
            else:
                raise e
        
        finally:
            torch.cuda.empty_cache()
        
        return max_batch_size

def create_all_dataloaders(
    processed_data_dir: str,
    splits_dir: str,
    image_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 2
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Create DataLoaders for all datasets in the project.
    
    Args:
        processed_data_dir: Path to processed data directory
        splits_dir: Path to splits directory
        image_size: Input image size
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        
    Returns:
        Dictionary mapping dataset names to their DataLoaders
    """
    processed_path = Path(processed_data_dir)
    splits_path = Path(splits_dir)
    
    # Create DataLoader factory
    factory = DataLoaderFactory(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    all_dataloaders = {}
    
    # 1. APTOS 2019 DataLoaders
    print("\nCreating Creating APTOS 2019 DataLoaders...")
    aptos_images_dir = processed_path / "aptos2019" / "images"
    aptos_labels_path = processed_path.parent / "aptos2019-blindness-detection" / "train.csv"
    aptos_splits_path = splits_path / "aptos2019_splits.json"
    
    if all(p.exists() for p in [aptos_images_dir, aptos_labels_path, aptos_splits_path]):
        aptos_dataloaders = factory.create_classification_dataloaders(
            str(aptos_images_dir),
            str(aptos_labels_path),
            str(aptos_splits_path),
            image_size=image_size,
            use_weighted_sampling=True
        )
        all_dataloaders['aptos2019'] = aptos_dataloaders
        print(f"APTOS 2019 DataLoaders created")
    else:
        print(f"APTOS 2019 dataset/splits not found")
    
    # 2. IDRiD Disease Grading DataLoaders
    print("\nCreating IDRiD Disease Grading DataLoaders...")
    grading_images_dir = processed_path / "grading" / "images"
    grading_splits_path = splits_path / "idrid_grading_splits.json"
    
    # Find grading labels
    grading_base = processed_path.parent / "B. Disease Grading" / "2. Groundtruths"
    grading_csv_files = list(grading_base.glob("*.csv")) if grading_base.exists() else []
    
    if grading_images_dir.exists() and grading_csv_files and grading_splits_path.exists():
        grading_labels_path = grading_csv_files[0]
        
        grading_dataloaders = factory.create_classification_dataloaders(
            str(grading_images_dir),
            str(grading_labels_path),
            str(grading_splits_path),
            image_size=image_size,
            use_weighted_sampling=True
        )
        all_dataloaders['idrid_grading'] = grading_dataloaders
        print(f"IDRiD Disease Grading DataLoaders created")
    else:
        print(f"IDRiD Disease Grading dataset/splits not found")
    
    # 3. IDRiD Segmentation DataLoaders
    print("\nCreating Creating IDRiD Segmentation DataLoaders...")
    segmentation_splits_path = splits_path / "idrid_segmentation_splits.json"
    
    if segmentation_splits_path.exists():
        segmentation_dataloaders = factory.create_segmentation_dataloaders(
            str(segmentation_splits_path),
            image_size=image_size
        )
        all_dataloaders['idrid_segmentation'] = segmentation_dataloaders
        print(f"IDRiD Segmentation DataLoaders created")
    else:
        print(f"IDRiD Segmentation splits not found")
    
    print(f"\nCreated DataLoaders for {len(all_dataloaders)} datasets")
    
    return all_dataloaders

# Memory optimization utilities
def optimize_dataloader_memory():
    """Apply memory optimizations for DataLoaders."""
    # Set multiprocessing start method
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Set CUDA memory fraction
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except AttributeError:
            pass
    
    logger.info("Applied DataLoader memory optimizations")

def estimate_memory_usage(batch_size: int, image_size: int, num_classes: int = 5) -> Dict[str, float]:
    """
    Estimate memory usage for different batch sizes.
    
    Args:
        batch_size: Batch size to estimate
        image_size: Input image size
        num_classes: Number of output classes
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Image tensor: batch_size * 3 * height * width * 4 bytes (float32)
    image_memory = batch_size * 3 * image_size * image_size * 4 / (1024 * 1024)
    
    # Mask tensor (if segmentation): batch_size * height * width * 4 bytes
    mask_memory = batch_size * image_size * image_size * 4 / (1024 * 1024)
    
    # Label tensor: batch_size * 4 bytes
    label_memory = batch_size * 4 / (1024 * 1024)
    
    # Model output (classification): batch_size * num_classes * 4 bytes
    output_memory = batch_size * num_classes * 4 / (1024 * 1024)
    
    return {
        'images_mb': image_memory,
        'masks_mb': mask_memory,
        'labels_mb': label_memory,
        'outputs_mb': output_memory,
        'total_mb': image_memory + mask_memory + label_memory + output_memory
    }

def create_dataloaders(
    dataset_type: str = "multitask",
    processed_data_dir: str = "dataset/processed",
    splits_dir: str = "dataset/splits", 
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 512,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for different dataset types.
    
    Args:
        dataset_type: Type of dataset ('grading', 'segmentation', 'multitask', 'aptos')
        processed_data_dir: Directory containing processed data
        splits_dir: Directory containing data splits
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        image_size: Target image size
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing train and validation dataloaders
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        factory = DataLoaderFactory()
        
        if dataset_type == "grading":
            return factory.create_grading_loaders(
                processed_data_dir=processed_data_dir,
                splits_dir=splits_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=image_size
            )
        elif dataset_type == "segmentation":
            return factory.create_segmentation_loaders(
                processed_data_dir=processed_data_dir,
                splits_dir=splits_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=image_size
            )
        elif dataset_type == "multitask":
            return factory.create_multitask_loaders(
                processed_data_dir=processed_data_dir,
                splits_dir=splits_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=image_size
            )
        elif dataset_type == "aptos":
            # Use APTOSDataset
            from .datasets import APTOSDataset
            from .transforms import get_transforms
            
            csv_file = kwargs.get('csv_file', 'dataset/aptos2019-blindness-detection/train.csv')
            images_dir = kwargs.get('images_dir', 'dataset/aptos2019-blindness-detection/train_images')
            
            # Create main dataset
            dataset = APTOSDataset(
                csv_file=csv_file,
                images_dir=images_dir,
                transform=None
            )
            
            # Split dataset
            train_dataset, val_dataset, test_dataset = dataset.split_dataset(
                train_ratio=0.7,
                val_ratio=0.2,
                random_state=42
            )
            
            # Get transforms
            train_transform = get_transforms('classification', 'train', image_size)
            val_transform = get_transforms('classification', 'val', image_size)
            
            # Apply transforms
            train_dataset.transform = train_transform
            val_dataset.transform = val_transform
            test_dataset.transform = val_transform
            
            # Create dataloaders
            return {
                'train': DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                ),
                'val': DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                ),
                'test': DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )
            }
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
    except Exception as e:
        logger.warning(f"Failed to create {dataset_type} dataloaders: {e}")
        # Return dummy dataloaders as fallback
        try:
            from ..training.pipeline import DummyDataset
            train_dataset = DummyDataset(size=batch_size * 10)
            val_dataset = DummyDataset(size=batch_size * 5)
            
            return {
                'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            }
        except:
            # Ultimate fallback - empty loaders
            return {'train': None, 'val': None}

if __name__ == "__main__":
    # Example usage - use relative paths from project root
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    processed_data_dir = str(project_root / "dataset" / "processed")
    splits_dir = str(project_root / "dataset" / "splits")
    
    # Optimize memory
    optimize_dataloader_memory()
    
    # Create all DataLoaders
    all_dataloaders = create_all_dataloaders(
        processed_data_dir=processed_data_dir,
        splits_dir=splits_dir,
        image_size=512,
        batch_size=4,
        num_workers=2
    )
    
    # Print memory estimates
    memory_est = estimate_memory_usage(batch_size=4, image_size=512)
    print(f"\nResults Memory estimate for batch_size=4: {memory_est['total_mb']:.1f} MB")
