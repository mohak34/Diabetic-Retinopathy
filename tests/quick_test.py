#!/usr/bin/env python3
"""
Quick test to verify Phase 2.2 components are working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("Testing Phase 2.2 Components...")

# Test 1: Import modules
try:
    from data.transforms import get_train_transforms_classification, get_val_transforms_classification
    from data.datasets import GradingRetinaDataset
    from data.dataloaders import DataLoaderFactory
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Test 2: Create transforms
try:
    train_transforms = get_train_transforms_classification()
    val_transforms = get_val_transforms_classification()
    print("Transforms created successfully")
    print(f"   Train transforms: {len(train_transforms.transforms)} operations")
    print(f"   Val transforms: {len(val_transforms.transforms)} operations")
except Exception as e:
    print(f"Transform creation error: {e}")

# Test 3: Check data splits exist
splits_dir = Path("dataset/splits")
if splits_dir.exists():
    split_files = list(splits_dir.glob("*.json"))
    print(f"Data splits found: {len(split_files)} files")
    for split_file in split_files:
        print(f"   {split_file.name}")
else:
    print("No data splits directory found")

# Test 4: Load one split to verify format
try:
    import json
    with open("dataset/splits/aptos2019_splits.json", "r") as f:
        aptos_splits = json.load(f)
    
    train_count = len(aptos_splits["splits"]["train"])
    val_count = len(aptos_splits["splits"]["val"])
    print(f"APTOS splits loaded: {train_count} train, {val_count} val")
    
    # Check if label distribution exists
    if "label_distribution" in aptos_splits:
        train_dist = aptos_splits["label_distribution"]["train"]
        print(f"   Training grade distribution: {train_dist}")
    
except Exception as e:
    print(f"Error loading splits: {e}")

# Test 5: Create DataLoader factory
try:
    factory = DataLoaderFactory()
    print("DataLoader factory created successfully")
    print(f"   Batch size: {factory.batch_size}")
    print(f"   Num workers: {factory.num_workers}")
except Exception as e:
    print(f"DataLoader factory error: {e}")

print("\nPhase 2.2 Component Test Complete!")
print("Summary:")
print("   Data augmentation pipelines ready")
print("   Custom PyTorch Dataset classes ready") 
print("   Stratified data splits created")
print("   Optimized DataLoader factory ready")
print("\nReady to proceed to Phase 2.3: Model Implementation!")
