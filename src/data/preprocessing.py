"""
Image preprocessing module for diabetic retinopathy images.

This module implements the preprocessing pipeline including:
- Black border cropping
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Ben Graham illumination normalization
- Resize and center crop operations
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RetinaPreprocessor:
    """
    Retinal image preprocessor implementing the complete preprocessing pipeline.
    """
    
    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        ben_graham_sigma: float = 10.0,
        resize_to: int = 256,
        crop_to: int = 224,
        border_threshold: int = 10
    ):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            clahe_clip_limit: Clip limit for CLAHE
            clahe_tile_grid_size: Tile grid size for CLAHE
            ben_graham_sigma: Sigma for Gaussian blur in Ben Graham normalization
            resize_to: Size to resize image to before cropping
            crop_to: Final crop size for model input
            border_threshold: Threshold for black border detection
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.ben_graham_sigma = ben_graham_sigma
        self.resize_to = resize_to
        self.crop_to = crop_to
        self.border_threshold = border_threshold
        
        # Initialize CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )
    
    def crop_black_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Crop black borders from retinal image.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            
        Returns:
            Cropped image with black borders removed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create mask for valid retina region
        mask = gray > self.border_threshold
        
        # Find coordinates of non-zero pixels
        coords = np.column_stack(np.where(mask))
        
        if len(coords) == 0:
            logger.warning("No valid pixels found, returning original image")
            return image
        
        # Get bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Crop the image
        cropped = image[y_min:y_max+1, x_min:x_max+1]
        
        return cropped
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            
        Returns:
            Image with CLAHE applied
        """
        # Convert RGB to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split LAB channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l_channel)
        
        # Merge channels back
        lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])
        
        # Convert back to RGB
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return rgb_clahe
    
    def ben_graham_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Ben Graham illumination normalization.
        
        The formula is: normalized = 4 * original - 4 * blur + 128
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            
        Returns:
            Illumination normalized image
        """
        # Convert to float for calculations
        image_float = image.astype(np.float32)
        
        # Apply Gaussian blur
        kernel_size = int(6 * self.ben_graham_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(
            image_float, 
            (kernel_size, kernel_size), 
            self.ben_graham_sigma
        )
        
        # Apply Ben Graham formula: 4*original - 4*blur + 128
        normalized = 4 * image_float - 4 * blurred + 128
        
        # Clip to valid range [0, 255]
        normalized = np.clip(normalized, 0, 255)
        
        return normalized.astype(np.uint8)
    
    def resize_and_crop(self, image: np.ndarray, random_crop: bool = False) -> np.ndarray:
        """
        Resize image and apply center or random crop.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            random_crop: If True, apply random crop; if False, center crop
            
        Returns:
            Resized and cropped image
        """
        # Resize to target size
        resized = cv2.resize(image, (self.resize_to, self.resize_to), interpolation=cv2.INTER_AREA)
        
        # Calculate crop coordinates
        crop_margin = (self.resize_to - self.crop_to) // 2
        
        if random_crop and crop_margin > 0:
            # Random crop
            max_offset = self.resize_to - self.crop_to
            x_offset = np.random.randint(0, max_offset + 1)
            y_offset = np.random.randint(0, max_offset + 1)
        else:
            # Center crop
            x_offset = crop_margin
            y_offset = crop_margin
        
        # Apply crop
        cropped = resized[
            y_offset:y_offset + self.crop_to,
            x_offset:x_offset + self.crop_to
        ]
        
        return cropped
    
    def preprocess(self, image: np.ndarray, random_crop: bool = False) -> np.ndarray:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            random_crop: If True, apply random crop; if False, center crop
            
        Returns:
            Fully preprocessed image ready for model input
        """
        # Step 1: Crop black borders
        cropped = self.crop_black_borders(image)
        
        # Step 2: Apply CLAHE
        clahe_applied = self.apply_clahe(cropped)
        
        # Step 3: Ben Graham normalization
        normalized = self.ben_graham_normalization(clahe_applied)
        
        # Step 4: Resize and crop
        final = self.resize_and_crop(normalized, random_crop=random_crop)
        
        return final
    
    def preprocess_batch(self, images: list, random_crop: bool = False) -> list:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of RGB images as numpy arrays
            random_crop: If True, apply random crop; if False, center crop
            
        Returns:
            List of preprocessed images
        """
        return [self.preprocess(img, random_crop=random_crop) for img in images]


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path and convert to RGB.
    
    Args:
        image_path: Path to image file
        
    Returns:
        RGB image as numpy array or None if loading fails
    """
    try:
        # Load image using OpenCV (loads as BGR)
        image_bgr = cv2.imread(image_path)
        
        if image_bgr is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        return image_rgb
    
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save RGB image to file.
    
    Args:
        image: RGB image as numpy array
        output_path: Path to save image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save image
        success = cv2.imwrite(output_path, image_bgr)
        
        if not success:
            logger.error(f"Failed to save image: {output_path}")
        
        return success
    
    except Exception as e:
        logger.error(f"Error saving image {output_path}: {str(e)}")
        return False


# Example usage and testing functions
if __name__ == "__main__":
    # Initialize preprocessor with default parameters
    preprocessor = RetinaPreprocessor()
    
    # Example usage (requires actual image file)
    # image = load_image("path/to/retinal/image.jpg")
    # if image is not None:
    #     preprocessed = preprocessor.preprocess(image)
    #     save_image(preprocessed, "path/to/output/preprocessed.jpg")
    
    print("RetinaPreprocessor initialized successfully!")
    print(f"Configuration:")
    print(f"  CLAHE clip limit: {preprocessor.clahe_clip_limit}")
    print(f"  CLAHE tile grid size: {preprocessor.clahe_tile_grid_size}")
    print(f"  Ben Graham sigma: {preprocessor.ben_graham_sigma}")
    print(f"  Resize to: {preprocessor.resize_to}")
    print(f"  Crop to: {preprocessor.crop_to}")
    print(f"  Border threshold: {preprocessor.border_threshold}")
