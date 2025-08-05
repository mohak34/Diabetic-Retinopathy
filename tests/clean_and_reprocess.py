#!/usr/bin/env python3
"""
Clean and re-run preprocessing with fixes for segmentation data
"""

import shutil
from pathlib import Path

def clean_processed_data():
    """Remove all processed data and start fresh"""
    # Use relative path from project root
    base_path = Path(__file__).parent
    
    directories_to_clean = [
        base_path / "dataset" / "processed",
        base_path / "dataset" / "splits", 
        base_path / "dataset" / "metadata",
        base_path / "results",
        base_path / "visualizations"
    ]
    
    for directory in directories_to_clean:
        if directory.exists():
            print(f"Cleaning: {directory.relative_to(base_path)}")
            shutil.rmtree(directory)
        else:
            print(f"Directory not found (skipping): {directory.relative_to(base_path)}")
    
    # Recreate base directories
    for directory in directories_to_clean:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory.relative_to(base_path)}")
    
    print("Cleanup complete!")
    print("\nTo reprocess the data, run these commands:")
    print("   uv run python src/data/data_organizer.py")
    print("   (or)")  
    print("   uv run python src/data/data_preprocessing_pipeline.py")
    print("   (or)")
    print("   uv run python src/utils/validate_processing.py")

if __name__ == "__main__":
    clean_processed_data()
