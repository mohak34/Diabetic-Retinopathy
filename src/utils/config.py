"""
Configuration and utility functions for loading project configurations.
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, loads from configs/base_config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Get project root directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        config_path = project_root / "configs" / "base_config.yaml"
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}


def get_data_config() -> Dict[str, Any]:
    """Get data-specific configuration."""
    config = load_config()
    return config.get('data', {})


def get_model_config() -> Dict[str, Any]:
    """Get model-specific configuration."""
    config = load_config()
    return config.get('model', {})


def get_training_config() -> Dict[str, Any]:
    """Get training-specific configuration."""
    config = load_config()
    return config.get('training', {})


def get_preprocessing_config() -> Dict[str, Any]:
    """Get preprocessing-specific configuration."""
    data_config = get_data_config()
    return data_config.get('preprocessing', {})
