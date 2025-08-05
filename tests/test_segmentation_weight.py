#!/usr/bin/env python3
"""
Test the segmentation weight calculation specifically
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.training.config import AdvancedTrainingConfig
    
    print("Testing segmentation weight calculation...")
    config = AdvancedTrainingConfig.from_yaml("configs/base_config.yaml")
    
    print(f"Phase 1 epochs: {config.phase1.epochs} (type: {type(config.phase1.epochs)})")
    print(f"Phase 2 epochs: {config.phase2.epochs} (type: {type(config.phase2.epochs)})")
    print(f"Phase 3 epochs: {config.phase3.epochs} (type: {type(config.phase3.epochs)})")
    
    # Test various epoch values
    test_epochs = [0, 1, 5, 10, 15, 16, 20, 25, 30, 40, 50]
    
    print("\n=== Segmentation Weight Test ===")
    for epoch in test_epochs:
        try:
            weight = config.get_segmentation_weight(epoch)
            phase_name, _ = config.get_current_phase(epoch)
            print(f"Epoch {epoch:2d}: {phase_name} -> weight = {weight:.3f}")
        except Exception as e:
            print(f"Epoch {epoch:2d}: ERROR - {e}")
    
    print("\n=== Testing with problematic values ===")
    problematic_values = ["0", 0.0, "1.0", 1.5, None]
    for val in problematic_values:
        try:
            weight = config.get_segmentation_weight(val)
            print(f"Value {val} (type {type(val)}): weight = {weight:.3f}")
        except Exception as e:
            print(f"Value {val} (type {type(val)}): ERROR - {e}")
    
    print("\n All segmentation weight tests completed!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
