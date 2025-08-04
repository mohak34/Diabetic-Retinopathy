#!/usr/bin/env python3
"""
Test script for the robust Phase 4 implementation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_phase4_losses():
    """Test the robust loss functions"""
    print("🧪 Testing Phase 4 losses...")
    
    try:
        from src.training.phase4_losses import test_robust_losses
        test_robust_losses()
        print("✅ Phase 4 losses test passed!")
        return True
    except Exception as e:
        print(f"❌ Phase 4 losses test failed: {e}")
        return False

def test_phase4_config():
    """Test the configuration system"""
    print("🧪 Testing Phase 4 configuration...")
    
    try:
        from src.training.phase4_config import Phase4Config
        
        # Test default config
        config = Phase4Config()
        print(f"  - Default config created: {config.experiment_name}")
        
        # Test smoke test config
        smoke_config = config.create_smoke_test_config()
        print(f"  - Smoke test config: {smoke_config.progressive.total_epochs} total epochs")
        
        # Test YAML save/load
        config.to_yaml("test_config.yaml")
        loaded_config = Phase4Config.from_yaml("test_config.yaml")
        print(f"  - YAML save/load: {loaded_config.experiment_name}")
        
        print("✅ Phase 4 configuration test passed!")
        return True
    except Exception as e:
        print(f"❌ Phase 4 configuration test failed: {e}")
        return False

def test_smoke_test():
    """Test the smoke test functionality"""
    print("🧪 Testing smoke test...")
    
    try:
        from src.training.phase4_pipeline import Phase4Pipeline
        
        # Create pipeline in smoke test mode
        pipeline = Phase4Pipeline(smoke_test=True)
        
        # Run smoke test
        success = pipeline.run_smoke_test()
        
        if success:
            print("✅ Smoke test passed!")
            return True
        else:
            print("❌ Smoke test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Smoke test failed with exception: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Robust Phase 4 Implementation")
    print("=" * 50)
    
    tests = [
        ("Configuration System", test_phase4_config),
        ("Loss Functions", test_phase4_losses),
        ("Smoke Test", test_smoke_test),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 {name}")
        success = test_func()
        results.append((name, success))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Phase 4 is ready to use.")
        return 0
    else:
        print("\n💥 Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
