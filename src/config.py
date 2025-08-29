import yaml
import logging
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fire_forecasting.log')
        ]
    )
    return logging.getLogger(__name__)

def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # TensorFlow seed setting
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seeds
    set_seeds(config.get('project', {}).get('seed', 42))
    
    return config

def get_project_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get project-level configuration"""
    return config.get('project', {})

def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get data configuration"""
    return config.get('data', {})

def get_labeling_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get labeling configuration"""
    return config.get('labeling', {})

def get_features_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get features configuration"""
    return config.get('features', {})

def get_train_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get training configuration"""
    return config.get('train', {})
