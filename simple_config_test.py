#!/usr/bin/env python3
"""
Simple configuration test
"""

import yaml
from pathlib import Path

# Load the YAML directly
config_path = "configs/phase4_config.yaml"
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)

print("=== Testing YAML Loading ===")
print(f"Phase2 segmentation_weight_start: {config_dict['phase2']['segmentation_weight_start']} (type: {type(config_dict['phase2']['segmentation_weight_start'])})")
print(f"Phase2 segmentation_weight_end: {config_dict['phase2']['segmentation_weight_end']} (type: {type(config_dict['phase2']['segmentation_weight_end'])})")

print(f"Optimizer eps: {config_dict['optimizer']['eps']} (type: {type(config_dict['optimizer']['eps'])})")

# Test type conversion
try:
    start_weight = float(config_dict['phase2']['segmentation_weight_start'])
    end_weight = float(config_dict['phase2']['segmentation_weight_end'])
    eps_val = float(config_dict['optimizer']['eps'])
    
    print(f"Converted start_weight: {start_weight}")
    print(f"Converted end_weight: {end_weight}")
    print(f"Converted eps: {eps_val}")
    
    # Test arithmetic
    result = end_weight - start_weight
    print(f"Arithmetic test: {end_weight} - {start_weight} = {result}")
    
except Exception as e:
    print(f"Conversion error: {e}")
    import traceback
    traceback.print_exc()
