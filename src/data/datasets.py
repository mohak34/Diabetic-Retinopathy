"""
Custom PyTorch Dataset Classes for Diabetic Retinopathy Multi-Task Learning
Supports classification (APTOS, Disease Grading) and segmentation (IDRiD) tasks.
"""

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union
import json
import logging
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APTOSDataset(Dataset):
    """APTOS 2019 Blindness Detection Dataset for Phase 5 training."""
    
    def __init__(
        self,
        csv_file: str,
        images_dir: str,
        transform: Optional[Callable] = None,
        mode: str = 'train'
    ):
        """
        Initialize APTOS dataset.
        
        Args:
            csv_file: Path to CSV file with id_code,diagnosis columns
            images_dir: Directory containing image files
            transform: Optional transform to apply to images
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.csv_file = csv_file
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.mode = mode
        
        # Load and validate CSV
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        self.data = pd.read_csv(csv_file)
        
        # Validate required columns
        required_cols = ['id_code', 'diagnosis']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # Filter out any NaN values
        self.data = self.data.dropna(subset=required_cols)
        
        logger.info(f"Loaded APTOS dataset with {len(self.data)} samples from {csv_file}")
        logger.info(f"Class distribution: {self.data['diagnosis'].value_counts().to_dict()}")
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image info
        row = self.data.iloc[idx]
        image_id = row['id_code']
        diagnosis = int(row['diagnosis'])
        
        # Load image
        image_path = self.images_dir / f"{image_id}.png"
        if not image_path.exists():
            # Try .jpg extension
            image_path = self.images_dir / f"{image_id}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_id}.png or {image_id}.jpg")
        
        # Load and convert image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Create a dummy image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            try:
                if hasattr(self.transform, '__call__'):
                    # Albumentations transform
                    transformed = self.transform(image=image)
                    image = transformed['image']
                else:
                    # Torchvision transforms
                    image = self.transform(image)
            except Exception as e:
                logger.warning(f"Transform failed for image {image_id}: {e}")
                # Convert to tensor manually if transform fails
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'image': image,
            'diagnosis': torch.tensor(diagnosis, dtype=torch.long),
            'id_code': image_id
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        class_counts = self.data['diagnosis'].value_counts().sort_index()
        total_samples = len(self.data)
        
        # Calculate inverse frequency weights
        weights = total_samples / (len(class_counts) * class_counts.values)
        return torch.tensor(weights, dtype=torch.float32)
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                     random_state: int = 42) -> Tuple['APTOSDataset', 'APTOSDataset', 'APTOSDataset']:
        """Split dataset into train/val/test sets."""
        # Stratified split to maintain class distribution
        train_data, temp_data = train_test_split(
            self.data, 
            test_size=(1 - train_ratio),
            stratify=self.data['diagnosis'],
            random_state=random_state
        )
        
        # Calculate validation ratio from remaining data
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_data['diagnosis'],
            random_state=random_state
        )
        
        # Create temporary CSV files for each split
        temp_dir = Path(self.csv_file).parent / 'temp_splits'
        temp_dir.mkdir(exist_ok=True)
        
        train_csv = temp_dir / 'train_split.csv'
        val_csv = temp_dir / 'val_split.csv'
        test_csv = temp_dir / 'test_split.csv'
        
        train_data.to_csv(train_csv, index=False)
        val_data.to_csv(val_csv, index=False)
        test_data.to_csv(test_csv, index=False)
        
        # Create dataset instances
        train_dataset = APTOSDataset(str(train_csv), str(self.images_dir), self.transform, 'train')
        val_dataset = APTOSDataset(str(val_csv), str(self.images_dir), self.transform, 'val')
        test_dataset = APTOSDataset(str(test_csv), str(self.images_dir), self.transform, 'test')
        
        return train_dataset, val_dataset, test_dataset


class IDRiDDataset(Dataset):
    """IDRiD Dataset for diabetic retinopathy detection and segmentation."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        task: str = 'grading',  # 'grading', 'segmentation', 'localization'
        transform: Optional[Callable] = None,
        image_size: int = 512
    ):
        """
        Initialize IDRiD dataset.
        
        Args:
            root_dir: Root directory containing IDRiD data
            split: Dataset split ('train', 'test')
            task: Task type ('grading', 'segmentation', 'localization')
            transform: Optional transform to apply to images
            image_size: Target image size for resizing
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.task = task
        self.transform = transform
        self.image_size = image_size
        
        # Setup paths based on task
        if task == 'grading':
            self.images_dir = self.root_dir / 'B. Disease Grading' / f'{split.capitalize()}ing Set'
            self.labels_file = self.images_dir / f'{split.capitalize()}ing Set.csv'
        elif task == 'segmentation':
            self.images_dir = self.root_dir / 'A. Segmentation' / f'{split.capitalize()}ing Set' / 'Images'
            self.masks_dir = self.root_dir / 'A. Segmentation' / f'{split.capitalize()}ing Set' / 'Masks'
        elif task == 'localization':
            self.images_dir = self.root_dir / 'C. Localization' / f'{split.capitalize()}ing Set' / 'Images'
            self.labels_dir = self.root_dir / 'C. Localization' / f'{split.capitalize()}ing Set' / 'Groundtruths'
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Load data based on task
        self.data = self._load_data()
        
        logger.info(f"Loaded IDRiD {task} dataset: {split} split with {len(self.data)} samples")
    
    def _load_data(self) -> List[Dict]:
        """Load data based on task type."""
        data = []
        
        if self.task == 'grading':
            # Load grading labels
            if self.labels_file.exists():
                import pandas as pd
                df = pd.read_csv(self.labels_file)
                for _, row in df.iterrows():
                    image_name = row['Image name']
                    data.append({
                        'image_name': image_name,
                        'retinopathy_grade': row.get('Retinopathy grade', 0),
                        'risk_of_macular_edema': row.get('Risk of macular edema', 0)
                    })
            else:
                # Fallback: use all images in directory
                for img_path in self.images_dir.glob('*.jpg'):
                    data.append({
                        'image_name': img_path.name,
                        'retinopathy_grade': 0,  # Default grade
                        'risk_of_macular_edema': 0
                    })
        
        elif self.task == 'segmentation':
            # Load segmentation data
            if self.images_dir.exists():
                for img_path in self.images_dir.glob('*.jpg'):
                    # Look for corresponding masks
                    mask_patterns = [
                        f"IDRiD_{img_path.stem.split('_')[1]}_EX.tif",  # Exudates
                        f"IDRiD_{img_path.stem.split('_')[1]}_HE.tif",  # Hemorrhages
                        f"IDRiD_{img_path.stem.split('_')[1]}_MA.tif",  # Microaneurysms
                        f"IDRiD_{img_path.stem.split('_')[1]}_SE.tif",  # Soft Exudates
                    ]
                    
                    masks = {}
                    for pattern in mask_patterns:
                        mask_path = self.masks_dir / pattern
                        if mask_path.exists():
                            lesion_type = pattern.split('_')[-1].split('.')[0]
                            masks[lesion_type] = str(mask_path)
                    
                    data.append({
                        'image_name': img_path.name,
                        'image_path': str(img_path),
                        'masks': masks
                    })
        
        elif self.task == 'localization':
            # Load localization data
            if self.images_dir.exists():
                for img_path in self.images_dir.glob('*.jpg'):
                    data.append({
                        'image_name': img_path.name,
                        'image_path': str(img_path)
                    })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]
        
        if self.task == 'grading':
            return self._get_grading_sample(sample)
        elif self.task == 'segmentation':
            return self._get_segmentation_sample(sample)
        elif self.task == 'localization':
            return self._get_localization_sample(sample)
    
    def _get_grading_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Get grading sample."""
        # Load image
        image_path = self.images_dir / sample['image_name']
        if not image_path.exists():
            # Try different extensions
            for ext in ['.jpg', '.png', '.tiff']:
                alt_path = self.images_dir / (sample['image_name'].replace('.jpg', ext))
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Apply transforms
        if self.transform:
            try:
                if hasattr(self.transform, '__call__'):
                    transformed = self.transform(image=image)
                    image = transformed['image']
                else:
                    image = self.transform(image)
            except Exception as e:
                logger.warning(f"Transform failed for {sample['image_name']}: {e}")
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'image': image,
            'retinopathy_grade': torch.tensor(sample['retinopathy_grade'], dtype=torch.long),
            'risk_of_macular_edema': torch.tensor(sample['risk_of_macular_edema'], dtype=torch.long),
            'image_name': sample['image_name']
        }
    
    def _get_segmentation_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Get segmentation sample."""
        # Load image
        image_path = sample['image_path']
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Load masks
        masks = {}
        for lesion_type, mask_path in sample['masks'].items():
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (self.image_size, self.image_size))
                    mask = (mask > 127).astype(np.uint8)  # Binarize
                    masks[lesion_type] = torch.from_numpy(mask).float()
                else:
                    masks[lesion_type] = torch.zeros((self.image_size, self.image_size), dtype=torch.float)
            except Exception as e:
                logger.warning(f"Error loading mask {mask_path}: {e}")
                masks[lesion_type] = torch.zeros((self.image_size, self.image_size), dtype=torch.float)
        
        # Combine masks into a single tensor
        combined_mask = torch.zeros((self.image_size, self.image_size), dtype=torch.float)
        for mask in masks.values():
            combined_mask = torch.maximum(combined_mask, mask)
        
        # Apply transforms
        if self.transform:
            try:
                if hasattr(self.transform, '__call__'):
                    transformed = self.transform(image=image, mask=combined_mask.numpy())
                    image = transformed['image']
                    combined_mask = torch.from_numpy(transformed['mask']).float()
                else:
                    image = self.transform(image)
            except Exception as e:
                logger.warning(f"Transform failed for {sample['image_name']}: {e}")
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'image': image,
            'mask': combined_mask.unsqueeze(0),  # Add channel dimension
            'individual_masks': masks,
            'image_name': sample['image_name']
        }
    
    def _get_localization_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Get localization sample."""
        # Load image
        image_path = sample['image_path']
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Apply transforms
        if self.transform:
            try:
                if hasattr(self.transform, '__call__'):
                    transformed = self.transform(image=image)
                    image = transformed['image']
                else:
                    image = self.transform(image)
            except Exception as e:
                logger.warning(f"Transform failed for {sample['image_name']}: {e}")
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'image': image,
            'image_name': sample['image_name']
        }


class GradingRetinaDataset(Dataset):
    """
    PyTorch Dataset for diabetic retinopathy grading tasks (APTOS and Disease Grading).
    Handles multi-class classification with DR severity levels (0-4).
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_path: Optional[str] = None,
        labels_dict: Optional[Dict] = None,
        transform: Optional[Callable] = None,
        image_ids: Optional[List[str]] = None,
        grade_mapping: Optional[Dict] = None
    ):
        """
        Initialize the grading dataset.
        
        Args:
            images_dir: Directory containing images
            labels_path: Path to CSV file with labels (optional if labels_dict provided)
            labels_dict: Dictionary mapping image IDs to labels (optional if labels_path provided)
            transform: Albumentations transform pipeline
            image_ids: Specific image IDs to include (for train/val splits)
            grade_mapping: Optional mapping to remap grade values
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.grade_mapping = grade_mapping or {}
        
        # Load labels
        if labels_dict is not None:
            self.labels = labels_dict
        elif labels_path is not None:
            self.labels = self._load_labels_from_csv(labels_path)
        else:
            raise ValueError("Either labels_path or labels_dict must be provided")
        
        # Get image files
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        if image_ids is not None:
            # Use specific image IDs (for train/val splits)
            self.image_files = [f for f in os.listdir(self.images_dir) 
                               if self._get_image_id(f) in image_ids]
        else:
            # Use all images in directory that have labels
            self.image_files = [f for f in os.listdir(self.images_dir) 
                               if Path(f).suffix.lower() in self.image_extensions
                               and self._get_image_id(f) in self.labels]
        
        self.image_files.sort()
        
        # Validate dataset
        self._validate_dataset()
        
        logger.info(f"Initialized GradingRetinaDataset with {len(self.image_files)} images")
        self._log_grade_distribution()
    
    def _load_labels_from_csv(self, csv_path: str) -> Dict[str, int]:
        """Load labels from CSV file."""
        df = pd.read_csv(csv_path)
        
        # Handle different CSV formats
        if 'id_code' in df.columns and 'diagnosis' in df.columns:
            # APTOS format
            labels = dict(zip(df['id_code'], df['diagnosis']))
        elif 'Image name' in df.columns and 'Retinopathy grade' in df.columns:
            # IDRiD format
            labels = {}
            for _, row in df.iterrows():
                img_id = Path(row['Image name']).stem
                grade = int(row['Retinopathy grade'])
                labels[img_id] = grade
        else:
            # Try to auto-detect columns
            img_col = df.columns[0]  # Assume first column is image identifier
            label_col = df.columns[-1]  # Assume last column is label
            
            logger.warning(f"Auto-detected columns: {img_col} (images), {label_col} (labels)")
            
            labels = {}
            for _, row in df.iterrows():
                img_id = str(row[img_col])
                if img_id.endswith('.jpg') or img_id.endswith('.png'):
                    img_id = Path(img_id).stem
                labels[img_id] = int(row[label_col])
        
        return labels
    
    def _get_image_id(self, filename: str) -> str:
        """Extract image ID from filename."""
        return Path(filename).stem
    
    def _validate_dataset(self):
        """Validate that all images have corresponding labels."""
        missing_labels = []
        for img_file in self.image_files:
            img_id = self._get_image_id(img_file)
            if img_id not in self.labels:
                missing_labels.append(img_id)
        
        if missing_labels:
            logger.warning(f"Found {len(missing_labels)} images without labels: {missing_labels[:5]}...")
            # Remove images without labels
            self.image_files = [f for f in self.image_files 
                               if self._get_image_id(f) not in missing_labels]
    
    def _log_grade_distribution(self):
        """Log the distribution of grades in the dataset."""
        grades = [self.labels[self._get_image_id(f)] for f in self.image_files]
        grade_counts = {}
        for grade in range(5):  # DR grades 0-4
            count = grades.count(grade)
            grade_counts[grade] = count
        
        logger.info(f"Grade distribution: {grade_counts}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        # Get image path and load
        img_file = self.image_files[idx]
        img_path = self.images_dir / img_file
        
        # Load image
        image = self._load_image(str(img_path))
        
        # Get label
        img_id = self._get_image_id(img_file)
        label = self.labels[img_id]
        
        # Apply grade mapping if provided
        if self.grade_mapping:
            label = self.grade_mapping.get(label, label)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image from path."""
        try:
            # Try with cv2 first (handles most formats including TIFF)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            
            # Fallback to PIL
            image = Image.open(img_path).convert('RGB')
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def get_label_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        grades = [self.labels[self._get_image_id(f)] for f in self.image_files]
        grade_counts = [grades.count(i) for i in range(5)]
        
        total_samples = len(grades)
        weights = [total_samples / (len(grade_counts) * count) if count > 0 else 0 
                  for count in grade_counts]
        
        return torch.FloatTensor(weights)

class SegmentationRetinaDataset(Dataset):
    """
    PyTorch Dataset for diabetic retinopathy segmentation tasks (IDRiD/Segmentation).
    Handles image-mask pairs for lesion segmentation.
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        image_mask_pairs: Optional[List[Tuple[str, str]]] = None,
        mask_threshold: float = 127.0
    ):
        """
        Initialize the segmentation dataset.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            transform: Albumentations transform pipeline
            image_mask_pairs: Pre-computed image-mask pairs
            mask_threshold: Threshold for binarizing masks
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.mask_threshold = mask_threshold
        
        # Get image-mask pairs
        if image_mask_pairs is not None:
            self.pairs = image_mask_pairs
        else:
            self.pairs = self._find_image_mask_pairs()
        
        # Validate pairs
        self._validate_pairs()
        
        logger.info(f"Initialized SegmentationRetinaDataset with {len(self.pairs)} pairs")
    
    def _find_image_mask_pairs(self) -> List[Tuple[str, str]]:
        """Find matching image-mask pairs."""
        pairs = []
        
        # Get all image files
        image_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = {}
        
        for img_file in os.listdir(self.images_dir):
            if Path(img_file).suffix.lower() in image_exts:
                img_id = Path(img_file).stem
                image_files[img_id] = img_file
        
        # Find corresponding masks
        for mask_file in os.listdir(self.masks_dir):
            if Path(mask_file).suffix.lower() in image_exts:
                mask_stem = Path(mask_file).stem
                
                # Try to match with image
                # For IDRiD: mask might be "IDRiD_73_HE.tif", image "IDRiD_73.jpg"
                if '_' in mask_stem:
                    parts = mask_stem.split('_')
                    if len(parts) >= 2:
                        img_id = '_'.join(parts[:2])  # "IDRiD_73"
                        if img_id in image_files:
                            img_path = self.images_dir / image_files[img_id]
                            mask_path = self.masks_dir / mask_file
                            pairs.append((str(img_path), str(mask_path)))
        
        return pairs
    
    def _validate_pairs(self):
        """Validate that all pairs exist and are readable."""
        valid_pairs = []
        
        for img_path, mask_path in self.pairs:
            if os.path.exists(img_path) and os.path.exists(mask_path):
                valid_pairs.append((img_path, mask_path))
            else:
                logger.warning(f"Missing file in pair: {img_path}, {mask_path}")
        
        self.pairs = valid_pairs
        logger.info(f"Validated {len(self.pairs)} pairs")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        img_path, mask_path = self.pairs[idx]
        
        # Load image and mask
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)
        
        # Apply transforms (both image and mask)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure mask is binary and correct shape
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).float()
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
        else:
            mask = (mask > self.mask_threshold).astype(np.float32)
            mask = torch.from_numpy(mask)
        
        return image, mask
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image from path."""
        try:
            # Try with cv2 first
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            
            # Fallback to PIL
            image = Image.open(img_path).convert('RGB')
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load mask from path."""
        try:
            # Handle TIFF masks
            if mask_path.lower().endswith('.tif') or mask_path.lower().endswith('.tiff'):
                # Use cv2 for TIFF files
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask
            
            # Try PIL for other formats
            mask = Image.open(mask_path).convert('L')
            return np.array(mask)
            
        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {e}")
            return np.zeros((512, 512), dtype=np.uint8)

class MultiTaskRetinaDataset(Dataset):
    """
    PyTorch Dataset for multi-task learning (both classification and segmentation).
    Returns dictionaries with image, label, and mask (when available).
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_path: Optional[str] = None,
        masks_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_ids: Optional[List[str]] = None
    ):
        """
        Initialize the multi-task dataset.
        
        Args:
            images_dir: Directory containing images
            labels_path: Path to CSV file with classification labels
            masks_dir: Directory containing segmentation masks (optional)
            transform: Albumentations transform pipeline
            image_ids: Specific image IDs to include
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.transform = transform
        
        # Load labels
        self.labels = {}
        if labels_path:
            self.labels = self._load_labels_from_csv(labels_path)
        
        # Get image files
        image_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        if image_ids is not None:
            self.image_files = [f for f in os.listdir(self.images_dir) 
                               if Path(f).suffix.lower() in image_exts
                               and Path(f).stem in image_ids]
        else:
            self.image_files = [f for f in os.listdir(self.images_dir) 
                               if Path(f).suffix.lower() in image_exts]
        
        self.image_files.sort()
        
        # Find masks for each image
        self.image_masks = {}
        if self.masks_dir and self.masks_dir.exists():
            self._find_masks()
        
        logger.info(f"Initialized MultiTaskRetinaDataset with {len(self.image_files)} images")
    
    def _load_labels_from_csv(self, csv_path: str) -> Dict[str, int]:
        """Load labels from CSV file."""
        df = pd.read_csv(csv_path)
        
        if 'id_code' in df.columns and 'diagnosis' in df.columns:
            return dict(zip(df['id_code'], df['diagnosis']))
        elif 'Image name' in df.columns and 'Retinopathy grade' in df.columns:
            labels = {}
            for _, row in df.iterrows():
                img_id = Path(row['Image name']).stem
                labels[img_id] = int(row['Retinopathy grade'])
            return labels
        else:
            # Auto-detect
            img_col, label_col = df.columns[0], df.columns[-1]
            labels = {}
            for _, row in df.iterrows():
                img_id = Path(str(row[img_col])).stem
                labels[img_id] = int(row[label_col])
            return labels
    
    def _find_masks(self):
        """Find masks corresponding to each image."""
        mask_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        for img_file in self.image_files:
            img_id = Path(img_file).stem
            self.image_masks[img_id] = []
            
            # Look for masks with matching ID
            for mask_file in os.listdir(self.masks_dir):
                if Path(mask_file).suffix.lower() in mask_exts:
                    mask_stem = Path(mask_file).stem
                    if mask_stem.startswith(img_id):
                        mask_path = self.masks_dir / mask_file
                        self.image_masks[img_id].append(str(mask_path))
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, List]]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with 'image', 'label' (if available), 'masks' (if available)
        """
        img_file = self.image_files[idx]
        img_path = self.images_dir / img_file
        img_id = Path(img_file).stem
        
        # Load image
        image = self._load_image(str(img_path))
        
        # Prepare sample dictionary
        sample = {'image': image}
        
        # Add label if available
        if img_id in self.labels:
            sample['label'] = self.labels[img_id]
        
        # Add masks if available
        if img_id in self.image_masks and self.image_masks[img_id]:
            masks = []
            for mask_path in self.image_masks[img_id]:
                mask = self._load_mask(mask_path)
                masks.append(mask)
            sample['masks'] = masks
        
        # Apply transforms
        if self.transform:
            if 'masks' in sample:
                # Apply transform to image and first mask
                augmented = self.transform(image=sample['image'], mask=sample['masks'][0])
                sample['image'] = augmented['image']
                sample['masks'][0] = augmented['mask']
                
                # Apply same transform to additional masks
                for i in range(1, len(sample['masks'])):
                    augmented_mask = self.transform(image=sample['image'], mask=sample['masks'][i])
                    sample['masks'][i] = augmented_mask['mask']
            else:
                augmented = self.transform(image=sample['image'])
                sample['image'] = augmented['image']
        
        return sample
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image from path."""
        try:
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            
            image = Image.open(img_path).convert('RGB')
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load mask from path."""
        try:
            if mask_path.lower().endswith('.tif') or mask_path.lower().endswith('.tiff'):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask
            
            mask = Image.open(mask_path).convert('L')
            return np.array(mask)
            
        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {e}")
            return np.zeros((512, 512), dtype=np.uint8)
