#!/usr/bin/env python3
"""
Script to count images in each folder of the Diabetic Retinopathy dataset
"""

import os
from pathlib import Path
from collections import defaultdict

def count_images_in_directory(directory_path):
    """Count image files in a directory (including subdirectories)"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'}
    count = 0
    
    if not os.path.exists(directory_path):
        return 0
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                count += 1
    
    return count

def count_images_detailed(base_path):
    """Detailed count of images in each subfolder"""
    base_path = Path(base_path)
    results = {}
    
    # Define the main folders to analyze
    main_folders = [
        "A. Segmentation",
        "B. Disease Grading", 
        "C. Localization",
        "aptos2019-blindness-detection"
    ]
    
    total_images = 0
    
    print("="*80)
    print("DIABETIC RETINOPATHY DATASET - IMAGE COUNT ANALYSIS")
    print("="*80)
    
    for folder in main_folders:
        folder_path = base_path / folder
        if folder_path.exists():
            print(f"\n{folder}/")
            print("-" * 60)
            
            folder_total = 0
            
            # Get all subdirectories and count images
            for subdir in sorted(folder_path.iterdir()):
                if subdir.is_dir():
                    count = count_images_in_directory(subdir)
                    folder_total += count
                    
                    # Further breakdown for nested directories
                    if count > 0:
                        print(f"  {subdir.name}/")
                        
                        # Check for further subdirectories
                        sub_subdirs = [d for d in subdir.iterdir() if d.is_dir()]
                        if sub_subdirs:
                            for sub_subdir in sorted(sub_subdirs):
                                sub_count = count_images_in_directory(sub_subdir)
                                if sub_count > 0:
                                    print(f"    {sub_subdir.name}/: {sub_count:,} images")
                        else:
                            print(f"    Total: {count:,} images")
            
            # Count images directly in the main folder (not in subdirs)
            direct_count = len([f for f in folder_path.iterdir() 
                              if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'}])
            
            if direct_count > 0:
                folder_total += direct_count
                print(f"  Direct files: {direct_count:,} images")
            
            print(f"  FOLDER TOTAL: {folder_total:,} images")
            total_images += folder_total
            results[folder] = folder_total
        else:
            print(f"\n{folder}/ - NOT FOUND")
            results[folder] = 0
    
    # Check other directories
    other_dirs = ["raw", "processed", "external"]
    print(f"\nOther Directories/")
    print("-" * 60)
    
    for dirname in other_dirs:
        dir_path = base_path / dirname
        if dir_path.exists():
            count = count_images_in_directory(dir_path)
            print(f"   {dirname}/: {count:,} images")
            total_images += count
            results[dirname] = count
        else:
            print(f"   {dirname}/: 0 images (empty/not found)")
            results[dirname] = 0
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for folder, count in results.items():
        print(f"{folder:35s}: {count:>8,} images")
    
    print("-" * 80)
    print(f"{'TOTAL DATASET':35s}: {total_images:>8,} images")
    print("="*80)
    
    return results, total_images

if __name__ == "__main__":
    # Use relative path from project root
    project_root = Path(__file__).parent
    data_path = project_root / "dataset"
    results, total = count_images_detailed(str(data_path))
