"""
Create minimal test datasets for Phase 6 evaluation
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import json

def create_test_datasets():
    """Create minimal test datasets for Phase 6 evaluation"""
    
    # Get project root (parent of tests directory)
    project_root = Path(__file__).parent.parent
    
    # Create test dataset directories
    test_data_dir = project_root / 'dataset/test_phase6'
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create APTOS-style validation dataset
    aptos_dir = test_data_dir / 'aptos2019_val'
    aptos_images_dir = aptos_dir / 'images'
    aptos_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy images and labels
    image_ids = []
    labels = []
    
    for i in range(10):  # Create 10 dummy images
        # Create a random image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image_pil = Image.fromarray(image)
        
        image_id = f'test_image_{i:03d}'
        image_path = aptos_images_dir / f'{image_id}.jpg'
        image_pil.save(image_path)
        
        image_ids.append(image_id)
        labels.append(np.random.randint(0, 5))  # Random DR grade 0-4
    
    # Create labels CSV
    labels_df = pd.DataFrame({
        'id_code': image_ids,
        'diagnosis': labels
    })
    labels_path = aptos_dir / 'labels.csv'
    labels_df.to_csv(labels_path, index=False)
    
    print(f"Created APTOS test dataset at: {aptos_dir}")
    print(f"  - {len(image_ids)} images")
    print(f"  - Labels: {labels_path}")
    
    # Create segmentation test dataset
    seg_dir = test_data_dir / 'segmentation_test'
    seg_images_dir = seg_dir / 'images'
    seg_masks_dir = seg_dir / 'masks'
    seg_images_dir.mkdir(parents=True, exist_ok=True)
    seg_masks_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(5):  # Create 5 dummy image-mask pairs
        # Create a random image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image_pil = Image.fromarray(image)
        
        # Create a random mask (binary)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
        mask_pil = Image.fromarray(mask, mode='L')
        
        image_id = f'seg_test_{i:03d}'
        image_path = seg_images_dir / f'{image_id}.jpg'
        mask_path = seg_masks_dir / f'{image_id}_HE.png'  # IDRiD style naming
        
        image_pil.save(image_path)
        mask_pil.save(mask_path)
    
    print(f"Created segmentation test dataset at: {seg_dir}")
    print(f"  - 5 image-mask pairs")
    
    # Return updated config for test datasets
    test_config = {
        'internal': {
            'aptos2019_val': {
                'type': 'aptos2019',
                'split': 'validation',
                'data_path': str(aptos_dir),
                'batch_size': 4,
                'num_workers': 2
            }
        },
        'external': {
            'segmentation_test': {
                'type': 'segmentation',
                'split': 'test',
                'data_path': str(seg_dir),
                'batch_size': 4,
                'num_workers': 2
            }
        }
    }
    
    # Save test config
    config_path = test_data_dir / 'test_datasets_config.json'
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"Test dataset configuration saved to: {config_path}")
    
    return test_config

if __name__ == "__main__":
    config = create_test_datasets()
    print("\nTest datasets created successfully!")
    print("You can now run Phase 6 evaluation with test data.")
