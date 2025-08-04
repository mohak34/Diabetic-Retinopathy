#!/usr/bin/env python3
"""
Debug script to isolate the configuration error
"""

import traceback
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.training.config import AdvancedTrainingConfig
    
    print("Loading configuration...")
    config = AdvancedTrainingConfig.from_yaml("configs/phase4_config.yaml")
    
    print("Testing segmentation weight calculation...")
    
    # Test various epochs
    for epoch in range(5):
        print(f"\n=== Testing epoch {epoch} ===")
        try:
            phase_name, phase = config.get_current_phase(epoch)
            print(f"Phase: {phase_name}")
            print(f"Phase object: {phase}")
            print(f"Phase segmentation_weight: {phase.segmentation_weight} (type: {type(phase.segmentation_weight)})")
            
            if hasattr(phase, 'segmentation_weight_start'):
                print(f"Phase segmentation_weight_start: {phase.segmentation_weight_start} (type: {type(phase.segmentation_weight_start)})")
            if hasattr(phase, 'segmentation_weight_end'):
                print(f"Phase segmentation_weight_end: {phase.segmentation_weight_end} (type: {type(phase.segmentation_weight_end)})")
            
            weight = config.get_segmentation_weight(epoch)
            print(f"Calculated weight: {weight}")
            
        except Exception as e:
            print(f"ERROR at epoch {epoch}: {e}")
            print(f"Error type: {type(e)}")
            traceback.print_exc()
            break

except Exception as e:
    print(f"Configuration loading failed: {e}")
    traceback.print_exc()
