#!/usr/bin/env python3
"""Debug configuration loading issue"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.config import AdvancedTrainingConfig

def main():
    print("=== DEBUG CONFIGURATION LOADING ===")
    
    # Load the configuration being used
    config = AdvancedTrainingConfig.from_yaml('configs/phase4_config.yaml')
    
    print("Original configuration:")
    print(f"Phase1 epochs: {config.phase1.epochs} (type: {type(config.phase1.epochs)})")
    print(f"Phase2 epochs: {config.phase2.epochs} (type: {type(config.phase2.epochs)})")
    print(f"Phase2 segmentation_weight_start: {config.phase2.segmentation_weight_start} (type: {type(config.phase2.segmentation_weight_start)})")
    print(f"Phase2 segmentation_weight_end: {config.phase2.segmentation_weight_end} (type: {type(config.phase2.segmentation_weight_end)})")
    print(f"Phase2 segmentation_weight: {config.phase2.segmentation_weight} (type: {type(config.phase2.segmentation_weight)})")
    
    # Modify for smoke test like train.py does
    original_total = config.total_epochs
    new_total = 2
    scale_factor = new_total / original_total
    
    print(f"\nApplying smoke test modifications (scale_factor: {scale_factor}):")
    config.phase1.epochs = max(1, int(config.phase1.epochs * scale_factor))
    config.phase2.epochs = max(1, int(config.phase2.epochs * scale_factor))
    config.phase3.epochs = max(1, int(config.phase3.epochs * scale_factor))
    config.total_epochs = new_total
    
    print("After modification:")
    print(f"Phase1 epochs: {config.phase1.epochs} (type: {type(config.phase1.epochs)})")
    print(f"Phase2 epochs: {config.phase2.epochs} (type: {type(config.phase2.epochs)})")
    print(f"Phase2 segmentation_weight_start: {config.phase2.segmentation_weight_start} (type: {type(config.phase2.segmentation_weight_start)})")
    print(f"Phase2 segmentation_weight_end: {config.phase2.segmentation_weight_end} (type: {type(config.phase2.segmentation_weight_end)})")
    print(f"Phase2 segmentation_weight: {config.phase2.segmentation_weight} (type: {type(config.phase2.segmentation_weight)})")
    
    # Test the method that's failing
    print("\n=== TESTING get_segmentation_weight ===")
    for epoch in range(config.total_epochs):
        try:
            weight = config.get_segmentation_weight(epoch)
            print(f"Epoch {epoch}: segmentation_weight = {weight} (type: {type(weight)})")
        except Exception as e:
            print(f"Epoch {epoch}: ERROR - {e}")
            print(f"Error type: {type(e)}")
            
            # Debug the actual issue
            phase_name, phase = config.get_current_phase(epoch)
            print(f"  Phase: {phase_name}")
            if phase_name == "phase2":
                print(f"  segmentation_weight_start: {phase.segmentation_weight_start} (type: {type(phase.segmentation_weight_start)})")
                print(f"  segmentation_weight_end: {phase.segmentation_weight_end} (type: {type(phase.segmentation_weight_end)})")
                print(f"  segmentation_weight: {phase.segmentation_weight} (type: {type(phase.segmentation_weight)})")
                
                # Try the calculation manually
                try:
                    phase_start_epoch = config.phase1.epochs
                    phase_progress = (epoch - phase_start_epoch) / config.phase2.epochs
                    print(f"  phase_start_epoch: {phase_start_epoch}")
                    print(f"  phase_progress: {phase_progress}")
                    
                    start_weight = float(phase.segmentation_weight_start or 0.0)
                    end_weight = float(phase.segmentation_weight_end or phase.segmentation_weight)
                    print(f"  start_weight: {start_weight} (type: {type(start_weight)})")
                    print(f"  end_weight: {end_weight} (type: {type(end_weight)})")
                    
                    result = start_weight + phase_progress * (end_weight - start_weight)
                    print(f"  calculated result: {result}")
                except Exception as inner_e:
                    print(f"  Inner calculation error: {inner_e}")
            break

if __name__ == "__main__":
    main()
