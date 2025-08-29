import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name} completed in {elapsed:.2f} seconds")

def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(data: Any, filepath: str) -> None:
    """Save data as JSON"""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Any:
    """Load data from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """Save DataFrame as CSV"""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    df.to_csv(filepath, index=index)

def load_csv(filepath: str) -> pd.DataFrame:
    """Load DataFrame from CSV"""
    return pd.read_csv(filepath)

def save_pickle(obj: Any, filepath: str) -> None:
    """Save object as pickle"""
    import pickle
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    """Load object from pickle"""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def format_number(num: float, decimals: int = 3) -> str:
    """Format number with specified decimal places"""
    if abs(num) < 0.001:
        return f"{num:.2e}"
    return f"{num:.{decimals}f}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if denominator is zero"""
    if denominator == 0:
        return default
    return numerator / denominator

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate DataFrame has required columns"""
    missing = set(required_columns) - set(df.columns)
    if missing:
        print(f"Missing required columns: {missing}")
        return False
    return True

def get_memory_usage(df: pd.DataFrame) -> str:
    """Get memory usage of DataFrame in human readable format"""
    memory_bytes = df.memory_usage(deep=True).sum()
    for unit in ['B', 'KB', 'MB', 'GB']:
        if memory_bytes < 1024.0:
            return f"{memory_bytes:.2f} {unit}"
        memory_bytes /= 1024.0
    return f"{memory_bytes:.2f} TB"
