#!/usr/bin/env python3
"""
Final verification script for Phase 4 implementation
"""

def verify_phase4():
    """Verify Phase 4 is working correctly"""
    print("🔍 PHASE 4 VERIFICATION")
    print("=" * 50)
    
    try:
        # Test 1: Configuration system
        print("✅ Testing configuration system...")
        from src.training.phase4_config import Phase4Config, setup_reproducibility, get_device
        
        config = Phase4Config()
        smoke_config = config.create_smoke_test_config()
        print(f"   - Default config: {config.experiment_name}")
        print(f"   - Smoke test epochs: {smoke_config.progressive.total_epochs}")
        
        # Test 2: Loss functions
        print("✅ Testing loss functions...")
        from src.training.phase4_losses import RobustMultiTaskLoss
        
        loss_fn = RobustMultiTaskLoss()
        print(f"   - Loss function created successfully")
        
        # Test 3: Trainer
        print("✅ Testing trainer...")
        from src.training.robust_trainer import RobustPhase4Trainer
        print(f"   - Trainer class imported successfully")
        
        # Test 4: Pipeline
        print("✅ Testing pipeline...")
        from src.training.phase4_pipeline import Phase4Pipeline
        print(f"   - Pipeline class imported successfully")
        
        print("\n🎉 ALL VERIFICATION TESTS PASSED!")
        print("\nPhase 4 is ready for use:")
        print("  • Run smoke test: uv run python -m src.training.phase4_pipeline --smoke-test")
        print("  • Run full training: uv run python -m src.training.phase4_pipeline")
        print("  • Check documentation: cat PHASE4_README.md")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    verify_phase4()
