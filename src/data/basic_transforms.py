
"""
Basic data transforms for diabetic retinopathy images
"""
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

def get_basic_training_transforms(image_size=512):
    """Basic training transforms"""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_basic_validation_transforms(image_size=512):
    """Basic validation transforms"""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_basic_segmentation_transforms(image_size=512):
    """Basic segmentation transforms (same as validation for simplicity)"""
    return get_basic_validation_transforms(image_size)
