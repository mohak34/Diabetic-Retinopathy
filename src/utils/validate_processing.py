"""
Quick validation script to check data processing results
"""

import json
from pathlib import Path

def validate_processing_results():
    """Validate that data processing completed successfully"""
    
    # Use relative path from project root
    project_root = Path(__file__).parent.parent.parent
    base_path = project_root / "dataset"
    processed_path = base_path / "processed"
    
    print("VALIDATING DATA PROCESSING RESULTS")
    print("="*50)
    
    # Check organization results
    org_results_file = processed_path / "organization_results.json"
    if org_results_file.exists():
        with open(org_results_file, 'r') as f:
            org_results = json.load(f)
        
        print("Data organization complete!")
        
        total_images = 0
        for dataset_name, data in org_results.items():
            if data.get('status') == 'success':
                if dataset_name == 'segmentation':
                    img_count = data.get('images', {}).get('count', 0)
                    pair_count = len(data.get('pairs', []))
                    print(f"   Segmentation: {img_count} images, {pair_count} pairs")
                    total_images += img_count
                elif dataset_name == 'grading':
                    img_count = data.get('images', {}).get('count', 0)
                    print(f"   Disease Grading: {img_count} images")
                    total_images += img_count
                elif dataset_name == 'localization':
                    img_count = data.get('images', {}).get('count', 0)
                    pair_count = len(data.get('bbox_pairs', []))
                    print(f"   Localization: {img_count} images, {pair_count} annotation pairs")
                    total_images += img_count
                elif dataset_name == 'aptos2019':
                    train_count = data.get('train_images', {}).get('count', 0)
                    test_count = data.get('test_images', {}).get('count', 0)
                    print(f"   APTOS 2019: {train_count} train + {test_count} test images")
                    total_images += train_count + test_count
        
        print(f"   TOTAL: {total_images:,} images")
        
    else:
        print("Organization results not found")
        return False
    
    # Check preprocessing results
    preprocessing_results_file = processed_path / "preprocessing_results.json"
    if preprocessing_results_file.exists():
        with open(preprocessing_results_file, 'r') as f:
            preprocessing_results = json.load(f)
        
        print("\nData preprocessing complete!")
        
        for dataset_name, result in preprocessing_results.items():
            if isinstance(result, dict) and result.get('status') == 'success':
                processed_count = result.get('processed_count', 0)
                failed_count = result.get('failed_count', 0)
                total = processed_count + failed_count
                success_rate = (processed_count / total * 100) if total > 0 else 0
                
                print(f"   {dataset_name.upper()}: {processed_count:,}/{total:,} ({success_rate:.1f}%)")
    else:
        print("\nPreprocessing results not found - may still be running")
    
    # Check processed directories
    print(f"\nPROCESSED DATA DIRECTORIES:")
    for subdir in ['segmentation', 'grading', 'localization', 'aptos2019']:
        subdir_path = processed_path / subdir
        if subdir_path.exists():
            # Count files in subdirectories
            total_files = 0
            for item in subdir_path.rglob('*'):
                if item.is_file():
                    total_files += 1
            print(f"   {subdir}: {total_files} files")
        else:
            print(f"   {subdir}: not found")
    
    print(f"\nProcessed data location: {processed_path}")
    
    return True

if __name__ == "__main__":
    validate_processing_results()
