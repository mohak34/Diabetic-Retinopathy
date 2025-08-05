#!/usr/bin/env python3
"""
Debug script to test exact training initialization sequence
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("1. Loading configuration...")
    from src.training.config import AdvancedTrainingConfig
    config = AdvancedTrainingConfig.from_yaml("configs/base_config.yaml")
    print(" Configuration loaded")
    
    print("\n2. Testing phase calculation for epoch 0...")
    phase_name, phase_config = config.get_current_phase(0)
    print(f"Phase: {phase_name}")
    print(f"Phase config: {phase_config}")
    print(f"Type of phase config: {type(phase_config)}")
    
    print("\n3. Testing segmentation weight calculation for epoch 0...")
    seg_weight = config.get_segmentation_weight(0)
    print(f"Segmentation weight: {seg_weight}")
    
    print("\n4. Testing epoch 1 (phase transition)...")
    seg_weight_1 = config.get_segmentation_weight(1)
    print(f"Epoch 1 segmentation weight: {seg_weight_1}")
    
    print("\n5. Testing Phase4Trainer initialization...")
    from src.training.trainer import Phase4Trainer
    
    trainer = Phase4Trainer(config=config)
    print(" Phase4Trainer initialized")
    
    print("\n6. Testing phase info calculation...")
    phase_name, phase_info = trainer.get_current_phase_info(0)
    print(f"Phase info: {phase_info}")
    print(f"Segmentation weight from trainer: {phase_info['segmentation_weight']}")
    print(f"Type of segmentation weight: {type(phase_info['segmentation_weight'])}")
    
    print("\n7. Testing trainer setup...")
    trainer.setup_model()
    trainer.setup_optimizer()
    trainer.setup_loss_function()
    print(" Trainer setup complete")
    
    print("\n All tests passed!")
    
except Exception as e:
    print(f"\n Error: {e}")
    print(f"Error type: {type(e)}")
    traceback.print_exc()
