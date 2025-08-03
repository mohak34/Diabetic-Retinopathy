#!/usr/bin/env python3
"""
Simple test script for Phase 2.3 data quality assessment.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("PHASE 2.3 DATA QUALITY ASSESSMENT TEST")
print("="*60)

try:
    from src.data.data_quality_assessment import DataQualityAssessment
    print("Successfully imported DataQualityAssessment")
    
    # Initialize assessment
    assessor = DataQualityAssessment()
    print("Successfully initialized assessor")
    
    # Test loading splits
    splits = assessor.load_all_splits()
    print(f"Successfully loaded {len(splits)} datasets:")
    
    for dataset_name, data in splits.items():
        total_images = data.get('total_images', 0)
        splits_info = data.get('splits', {})
        print(f"  - {dataset_name}: {total_images} images, {len(splits_info)} splits")
    
    # Test class balance verification
    print("\n" + "="*40)
    print("Testing class balance verification...")
    balance_results = assessor.verify_class_balance(splits)
    print(f"Class balance check completed for {len(balance_results)} datasets")
    
    # Test data integrity check on a sample
    print("\n" + "="*40)
    print("Testing data integrity check...")
    integrity_results = assessor.check_data_integrity()
    print(f"Data integrity check completed for {len(integrity_results)} datasets")
    
    # Test leakage detection
    print("\n" + "="*40)
    print("Testing leakage detection...")
    leakage_results = assessor.detect_data_leakage(splits)
    print(f"Leakage detection completed for {len(leakage_results)} datasets")
    
    # Test documentation creation
    print("\n" + "="*40)
    print("Testing documentation creation...")
    doc_results = assessor.create_immutable_split_documentation(splits)
    print(f"Documentation creation completed for {len(doc_results)} datasets")
    
    print("\n" + "="*60)
    print("PHASE 2.3 QUALITY ASSESSMENT: ALL TESTS PASSED!")
    print("="*60)
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
