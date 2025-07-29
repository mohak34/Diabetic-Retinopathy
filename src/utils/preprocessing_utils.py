"""
Utility functions for image preprocessing testing and validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os


def create_test_image(size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Create a synthetic retinal image for testing preprocessing functions.
    
    Args:
        size: Image size as (height, width)
        
    Returns:
        Synthetic RGB retinal image
    """
    height, width = size
    
    # Create base image with circular retina
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Create circular mask for retina
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    
    # Initialize image with black borders
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill retina area with brownish background
    image[mask] = [120, 80, 60]  # Brownish fundus color
    
    # Add some vessels (darker lines)
    for angle in np.linspace(0, 2*np.pi, 8):
        x_line = center_x + np.cos(angle) * np.linspace(0, radius*0.8, 100)
        y_line = center_y + np.sin(angle) * np.linspace(0, radius*0.8, 100)
        
        for i in range(len(x_line)):
            x_coord = int(x_line[i])
            y_coord = int(y_line[i])
            if 0 <= x_coord < width and 0 <= y_coord < height:
                # Create vessel with varying thickness
                thickness = max(1, int(3 * (1 - i/len(x_line))))
                cv2_available = True
                try:
                    import cv2
                    cv2.circle(image, (x_coord, y_coord), thickness, (60, 40, 30), -1)
                except ImportError:
                    # Fallback without cv2
                    for dx in range(-thickness, thickness+1):
                        for dy in range(-thickness, thickness+1):
                            if dx*dx + dy*dy <= thickness*thickness:
                                px, py = x_coord + dx, y_coord + dy
                                if 0 <= px < width and 0 <= py < height:
                                    image[py, px] = [60, 40, 30]
    
    # Add some bright spots (potential lesions)
    np.random.seed(42)
    for _ in range(10):
        spot_x = np.random.randint(center_x - radius//2, center_x + radius//2)
        spot_y = np.random.randint(center_y - radius//2, center_y + radius//2)
        
        if mask[spot_y, spot_x]:
            spot_size = np.random.randint(2, 6)
            image[spot_y-spot_size:spot_y+spot_size, spot_x-spot_size:spot_x+spot_size] = [200, 150, 100]
    
    return image


def visualize_preprocessing_steps(
    original_image: np.ndarray,
    preprocessor,
    save_path: str = None
) -> None:
    """
    Visualize each step of the preprocessing pipeline.
    
    Args:
        original_image: Original RGB image
        preprocessor: RetinaPreprocessor instance
        save_path: Optional path to save the visualization
    """
    # Apply each preprocessing step
    cropped = preprocessor.crop_black_borders(original_image)
    clahe_applied = preprocessor.apply_clahe(cropped)
    normalized = preprocessor.ben_graham_normalization(clahe_applied)
    final = preprocessor.resize_and_crop(normalized)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Retinal Image Preprocessing Pipeline', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Cropped image
    axes[0, 1].imshow(cropped)
    axes[0, 1].set_title('After Border Cropping')
    axes[0, 1].axis('off')
    
    # CLAHE applied
    axes[0, 2].imshow(clahe_applied)
    axes[0, 2].set_title('After CLAHE')
    axes[0, 2].axis('off')
    
    # Ben Graham normalized
    axes[1, 0].imshow(normalized)
    axes[1, 0].set_title('After Ben Graham Normalization')
    axes[1, 0].axis('off')
    
    # Final processed
    axes[1, 1].imshow(final)
    axes[1, 1].set_title(f'Final ({final.shape[0]}x{final.shape[1]})')
    axes[1, 1].axis('off')
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def compare_before_after(
    original_images: List[np.ndarray],
    processed_images: List[np.ndarray],
    titles: List[str] = None,
    save_path: str = None
) -> None:
    """
    Compare original and processed images side by side.
    
    Args:
        original_images: List of original images
        processed_images: List of processed images
        titles: Optional list of titles for each image pair
        save_path: Optional path to save the comparison
    """
    n_images = len(original_images)
    fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_images):
        # Original image
        axes[0, i].imshow(original_images[i])
        title = f'Original {i+1}' if titles is None else f'Original {titles[i]}'
        axes[0, i].set_title(title)
        axes[0, i].axis('off')
        
        # Processed image
        axes[1, i].imshow(processed_images[i])
        title = f'Processed {i+1}' if titles is None else f'Processed {titles[i]}'
        axes[1, i].set_title(title)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()


def analyze_image_statistics(image: np.ndarray, title: str = "Image") -> dict:
    """
    Analyze and print image statistics.
    
    Args:
        image: RGB image as numpy array
        title: Title for the analysis
        
    Returns:
        Dictionary containing image statistics
    """
    stats = {
        'shape': image.shape,
        'dtype': image.dtype,
        'min': image.min(),
        'max': image.max(),
        'mean': image.mean(),
        'std': image.std()
    }
    
    print(f"\n{title} Statistics:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Data type: {stats['dtype']}")
    print(f"  Min value: {stats['min']}")
    print(f"  Max value: {stats['max']}")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std deviation: {stats['std']:.2f}")
    
    # Channel-wise statistics for RGB images
    if len(image.shape) == 3 and image.shape[2] == 3:
        channel_names = ['Red', 'Green', 'Blue']
        for i, channel_name in enumerate(channel_names):
            channel_mean = image[:, :, i].mean()
            channel_std = image[:, :, i].std()
            print(f"  {channel_name} channel - Mean: {channel_mean:.2f}, Std: {channel_std:.2f}")
    
    return stats


def test_preprocessing_robustness(preprocessor, n_test_images: int = 10) -> bool:
    """
    Test preprocessing robustness with various synthetic images.
    
    Args:
        preprocessor: RetinaPreprocessor instance
        n_test_images: Number of test images to generate
        
    Returns:
        True if all tests pass, False otherwise
    """
    print(f"Testing preprocessing robustness with {n_test_images} synthetic images...")
    
    all_passed = True
    
    for i in range(n_test_images):
        # Generate test image with random size
        size = (
            np.random.randint(300, 800),
            np.random.randint(300, 800)
        )
        
        test_image = create_test_image(size)
        
        try:
            # Test complete preprocessing pipeline
            processed = preprocessor.preprocess(test_image)
            
            # Validate output
            expected_shape = (preprocessor.crop_to, preprocessor.crop_to, 3)
            if processed.shape != expected_shape:
                print(f"  FAIL Test {i+1}: Wrong output shape {processed.shape}, expected {expected_shape}")
                all_passed = False
            else:
                print(f"  PASS Test {i+1}: Passed (input: {size}, output: {processed.shape})")
        
        except Exception as e:
            print(f"  FAIL Test {i+1}: Exception occurred - {str(e)}")
            all_passed = False
    
    return all_passed


# Unit test functions
def test_border_cropping():
    """Test black border cropping functionality."""
    print("Testing black border cropping...")
    
    # Create test image with black borders
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    # Add retina content in center
    test_image[50:150, 50:150] = [100, 80, 60]
    
    from src.data.preprocessing import RetinaPreprocessor
    preprocessor = RetinaPreprocessor()
    
    cropped = preprocessor.crop_black_borders(test_image)
    
    # Should crop to approximately 100x100
    expected_size = 100
    tolerance = 5
    
    if abs(cropped.shape[0] - expected_size) <= tolerance and abs(cropped.shape[1] - expected_size) <= tolerance:
        print("  PASS Border cropping test passed")
        return True
    else:
        print(f"  FAIL Border cropping test failed: output shape {cropped.shape}")
        return False


def test_clahe():
    """Test CLAHE functionality."""
    print("Testing CLAHE...")
    
    # Create low contrast test image
    test_image = np.full((100, 100, 3), 128, dtype=np.uint8)
    test_image[25:75, 25:75] = 100  # Slightly darker center
    
    from src.data.preprocessing import RetinaPreprocessor
    preprocessor = RetinaPreprocessor()
    
    clahe_result = preprocessor.apply_clahe(test_image)
    
    # CLAHE should increase contrast
    original_std = test_image.std()
    clahe_std = clahe_result.std()
    
    if clahe_std >= original_std:
        print("  PASS CLAHE test passed (contrast enhanced)")
        return True
    else:
        print(f"  FAIL CLAHE test failed: std decreased from {original_std:.2f} to {clahe_std:.2f}")
        return False


def test_ben_graham():
    """Test Ben Graham normalization."""
    print("Testing Ben Graham normalization...")
    
    # Create test image with illumination gradient
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            # Create gradient from 50 to 200
            intensity = int(50 + 150 * i / 100)
            test_image[i, j] = [intensity, intensity, intensity]
    
    from src.data.preprocessing import RetinaPreprocessor
    preprocessor = RetinaPreprocessor()
    
    normalized = preprocessor.ben_graham_normalization(test_image)
    
    # Check if normalization reduced the gradient effect
    original_gradient = test_image[-1, -1, 0] - test_image[0, 0, 0]
    normalized_gradient = abs(normalized[-1, -1, 0] - normalized[0, 0, 0])
    
    if normalized_gradient < original_gradient:
        print("  PASS Ben Graham normalization test passed (gradient reduced)")
        return True
    else:
        print(f"  FAIL Ben Graham test failed: gradient not reduced sufficiently")
        return False


def run_all_tests():
    """Run all unit tests."""
    print("Running all preprocessing unit tests...")
    print("=" * 50)
    
    tests = [
        test_border_cropping,
        test_clahe,
        test_ben_graham
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("SUCCESS: All tests passed!")
        return True
    else:
        print("FAILURE: Some tests failed!")
        return False
