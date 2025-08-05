"""
Simple test to verify UV integration with Phase 4
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all Phase 4 components can be imported"""
    print("Testing Testing Phase 4 imports with UV...")
    
    try:
        # Add src to path for imports
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))
        
        # Test core imports
        print("  OK Testing configuration system...")
        from src.training.config import AdvancedTrainingConfig
        
        print("  OK Testing metrics system...")
        from src.training.metrics import AdvancedMetricsCollector
        
        print("  OK Testing trainer...")
        from src.training.trainer import Phase4Trainer
        
        print("  OK Testing pipeline...")
        from src.training.pipeline import Phase4Pipeline
        
        print("  OK Testing hyperparameter optimization...")
        from src.training.hyperparameter_optimizer import HyperparameterOptimizer
        
        print("OK All Phase 4 components imported successfully!")
        return True
        
    except ImportError as e:
        print(f"ERROR Import failed: {e}")
        return False

def test_configuration():
    """Test configuration creation"""
    print("\nConfiguration Testing Phase 4 configuration...")
    
    try:
        from src.training.config import AdvancedTrainingConfig
        
        config = AdvancedTrainingConfig()
        print(f"  OK Created config: {config.experiment_name}")
        print(f"  OK Model: {config.model.backbone_name}")
        print(f"  OK Total epochs: {config.total_epochs}")
        print(f"  OK Progressive phases: {config.phase1.epochs} + {config.phase2.epochs} + {config.phase3.epochs}")
        
        return True
        
    except Exception as e:
        print(f"ERROR Configuration test failed: {e}")
        return False

def test_dependencies():
    """Test that key dependencies are available"""
    print("\nDependencies Testing dependencies...")
    
    required_packages = [
        "torch", "torchvision", "numpy", "matplotlib", 
        "sklearn", "tqdm", "yaml", "tensorboard"
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            if pkg == "sklearn":
                import sklearn
            elif pkg == "yaml":
                import yaml
            else:
                __import__(pkg)
            print(f"  OK {pkg}")
        except ImportError:
            print(f"  ERROR {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\nWARNING  Missing packages: {missing}")
        print("Run: uv sync --extra phase4")
        return False
    else:
        print("OK All dependencies available!")
        return True

def main():
    """Main test function"""
    print("Test Phase 4: UV Integration Test")
    print("=" * 50)
    
    # Test results
    results = []
    
    # Run tests
    results.append(test_dependencies())
    results.append(test_imports()) 
    results.append(test_configuration())
    
    # Summary
    print("\n" + "=" * 50)
    if all(results):
        print("SUCCESS: ALL TESTS PASSED!")
        print("Phase 4 is properly configured with UV!")
        print("\nNext steps:")
        print("  uv run python train.py --smoke-test")
        print("  uv run python examples/phase4_examples.py")
        print("  uv run python train.py --help")
    else:
        print("ERROR Some tests failed")
        print("Try: uv sync --extra phase4")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
