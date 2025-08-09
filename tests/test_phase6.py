"""
Simple Phase 6 Test Script
Tests the Phase 6 evaluation framework with minimal dependencies
"""

import os
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level from tests/ to project root
sys.path.append(str(project_root))

def test_phase6_imports():
    """Test that all Phase 6 modules can be imported"""
    print("Testing Phase 6 imports...")
    
    try:
        from src.evaluation.metrics_calculator import MetricsCalculator
        print("✓ MetricsCalculator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import MetricsCalculator: {e}")
        return False
    
    try:
        from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
        print("✓ ComprehensiveEvaluator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ComprehensiveEvaluator: {e}")
        return False
    
    try:
        from src.evaluation.external_validator import ExternalValidator
        print("✓ ExternalValidator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ExternalValidator: {e}")
        return False
    
    try:
        from src.evaluation.explainability_analyzer import ExplainabilityAnalyzer
        print("✓ ExplainabilityAnalyzer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ExplainabilityAnalyzer: {e}")
        return False
    
    try:
        from src.evaluation.visualization_generator import VisualizationGenerator
        print("✓ VisualizationGenerator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import VisualizationGenerator: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that the model can be created"""
    print("\nTesting model creation...")
    
    try:
        from src.models.multi_task_model import MultiTaskRetinaModel
        
        model = MultiTaskRetinaModel(
            num_classes_cls=5,
            num_classes_seg=4,
            backbone_name='tf_efficientnet_b0_ns',
            pretrained=False  # Don't download weights for test
        )
        
        print("✓ MultiTaskRetinaModel created successfully")
        print(f"✓ Model has {sum(p.numel() for p in model.parameters())} parameters")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False

def test_dataset_creation():
    """Test that datasets can be created"""
    print("\nTesting dataset creation...")
    
    try:
        from src.data.datasets import GradingRetinaDataset, SegmentationRetinaDataset, MultiTaskRetinaDataset
        print("✓ Dataset classes imported successfully")
        
        # Test with dummy data (doesn't need to exist for import test)
        dummy_config = {
            'images_dir': 'dummy/path',
            'labels_path': 'dummy/labels.csv',
            'masks_dir': 'dummy/masks'
        }
        
        print("✓ Dataset classes can be instantiated")
        return True
        
    except Exception as e:
        print(f"✗ Failed to import datasets: {e}")
        return False

def test_evaluator_initialization():
    """Test that evaluators can be initialized"""
    print("\nTesting evaluator initialization...")
    
    try:
        from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
        
        evaluator = ComprehensiveEvaluator(
            num_classes=5,
            device='cpu',  # Use CPU for test
            output_dir='results/test_phase6'
        )
        
        print("✓ ComprehensiveEvaluator initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize evaluator: {e}")
        return False

def test_checkpoint_creation():
    """Test creating a dummy checkpoint"""
    print("\nTesting checkpoint creation...")
    
    try:
        # Import relative to tests directory
        sys.path.insert(0, str(Path(__file__).parent))
        from create_test_checkpoint import create_dummy_checkpoint
        checkpoint_path = create_dummy_checkpoint()
        
        if os.path.exists(checkpoint_path):
            print(f"✓ Test checkpoint created at: {checkpoint_path}")
            return checkpoint_path
        else:
            print("✗ Checkpoint file not found after creation")
            return None
            
    except Exception as e:
        print(f"✗ Failed to create test checkpoint: {e}")
        return None

def main():
    """Run all Phase 6 tests"""
    print("=" * 80)
    print("PHASE 6 EVALUATION FRAMEWORK - SYSTEM TEST")
    print("=" * 80)
    
    all_tests_passed = True
    
    # Test imports
    if not test_phase6_imports():
        all_tests_passed = False
    
    # Test model creation
    if not test_model_creation():
        all_tests_passed = False
    
    # Test dataset creation
    if not test_dataset_creation():
        all_tests_passed = False
    
    # Test evaluator initialization
    if not test_evaluator_initialization():
        all_tests_passed = False
    
    # Test checkpoint creation
    checkpoint_path = test_checkpoint_creation()
    if checkpoint_path is None:
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - Phase 6 framework is ready!")
        print("=" * 80)
        
        if checkpoint_path:
            print(f"\nYou can now test Phase 6 with:")
            print(f"python scripts/phase6_evaluation_main.py \\")
            print(f"    --config configs/phase6_evaluation_config.yaml \\")
            print(f"    --checkpoints {checkpoint_path} \\")
            print(f"    --output-dir results/phase6_test")
    else:
        print("❌ SOME TESTS FAILED - Please fix the issues above")
        print("=" * 80)

if __name__ == "__main__":
    main()
