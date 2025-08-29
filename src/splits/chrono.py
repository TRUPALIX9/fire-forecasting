import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings

def create_chronological_split(df: pd.DataFrame, train_frac: float = 0.7,
                              val_frac: float = 0.15, test_frac: float = 0.15) -> Dict[str, pd.DataFrame]:
    """
    Create chronological train/validation/test split
    
    Args:
        df: DataFrame with datetime index or date column
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
        test_frac: Fraction of data for testing
    
    Returns:
        Dictionary with train, val, test DataFrames
    """
    print("Creating chronological split...")
    
    # Ensure we have datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have datetime index or 'date' column")
    
    # Sort by date
    df = df.sort_index()
    
    # Get unique dates
    unique_dates = df.index.date.unique()
    unique_dates = sorted(unique_dates)
    
    total_dates = len(unique_dates)
    print(f"  Total unique dates: {total_dates}")
    
    # Calculate split indices
    train_end_idx = int(total_dates * train_frac)
    val_end_idx = int(total_dates * (train_frac + val_frac))
    
    # Get split dates
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    print(f"  Train dates: {len(train_dates)} ({train_dates[0]} to {train_dates[-1]})")
    print(f"  Val dates: {len(val_dates)} ({val_dates[0]} to {val_dates[-1]})")
    print(f"  Test dates: {len(test_dates)} ({test_dates[0]} to {test_dates[-1]})")
    
    # Create splits
    train_df = df[df.index.date.isin(train_dates)].copy()
    val_df = df[df.index.date.isin(val_dates)].copy()
    test_df = df[df.index.date.isin(test_dates)].copy()
    
    # Reset index to get date as column
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    test_df = test_df.reset_index()
    
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    # Print split statistics
    for split_name, split_df in splits.items():
        total_records = len(split_df)
        positive_records = split_df['fire_tomorrow'].sum() if 'fire_tomorrow' in split_df.columns else 0
        positive_rate = positive_records / total_records if total_records > 0 else 0
        
        print(f"  {split_name.capitalize()}: {total_records} records, "
              f"{positive_records} positives ({positive_rate:.3f})")
    
    return splits

def fit_scaler_on_train(train_df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    """
    Fit StandardScaler on training data
    
    Args:
        train_df: Training DataFrame
        feature_cols: List of feature column names
    
    Returns:
        Fitted StandardScaler
    """
    print("Fitting StandardScaler on training data...")
    
    # Check if feature columns exist
    missing_cols = [col for col in feature_cols if col not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    print(f"  Scaler fitted on {len(feature_cols)} features")
    print(f"  Feature means: {scaler.mean_[:5]}...")  # Show first 5
    print(f"  Feature scales: {scaler.scale_[:5]}...")
    
    return scaler

def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, feature_cols: List[str]) -> pd.DataFrame:
    """
    Apply fitted scaler to DataFrame
    
    Args:
        df: DataFrame to scale
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
    
    Returns:
        DataFrame with scaled features
    """
    df_scaled = df.copy()
    
    # Scale features
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    
    return df_scaled

def compute_class_weights(train_df: pd.DataFrame, target_col: str = 'fire_tomorrow') -> Dict[int, float]:
    """
    Compute class weights for imbalanced classification
    
    Args:
        train_df: Training DataFrame
        target_col: Target column name
    
    Returns:
        Dictionary mapping class labels to weights
    """
    print("Computing class weights...")
    
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Get class labels and counts
    classes = train_df[target_col].unique()
    class_counts = train_df[target_col].value_counts()
    
    print(f"  Class distribution:")
    for class_label in sorted(classes):
        count = class_counts[class_label]
        print(f"    Class {class_label}: {count} samples")
    
    # Compute balanced class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=train_df[target_col]
    )
    
    # Create weight dictionary
    weight_dict = {class_label: weight for class_label, weight in zip(classes, class_weights)}
    
    print(f"  Class weights: {weight_dict}")
    
    return weight_dict

def prepare_data_for_training(splits: Dict[str, pd.DataFrame], feature_cols: List[str],
                             target_col: str = 'fire_tomorrow') -> Dict[str, np.ndarray]:
    """
    Prepare data arrays for training
    
    Args:
        splits: Dictionary with train/val/test DataFrames
        feature_cols: List of feature column names
        target_col: Target column name
    
    Returns:
        Dictionary with X and y arrays for each split
    """
    print("Preparing data arrays for training...")
    
    prepared_data = {}
    
    for split_name, split_df in splits.items():
        print(f"  Preparing {split_name} data...")
        
        # Check if required columns exist
        missing_cols = [col for col in feature_cols + [target_col] if col not in split_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {split_name}: {missing_cols}")
        
        # Extract features and target
        X = split_df[feature_cols].values
        y = split_df[target_col].values
        
        # Check for NaN values
        if np.isnan(X).any():
            print(f"    Warning: Found NaN values in {split_name} features")
            X = np.nan_to_num(X, nan=0.0)
        
        if np.isnan(y).any():
            print(f"    Warning: Found NaN values in {split_name} targets")
            y = np.nan_to_num(y, nan=0.0)
        
        prepared_data[split_name] = {
            'X': X,
            'y': y,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_positives': int(y.sum()),
            'positive_rate': float(y.mean())
        }
        
        print(f"    {split_name}: {X.shape[0]} samples, {X.shape[1]} features, "
              f"{int(y.sum())} positives ({y.mean():.3f})")
    
    return prepared_data

def create_chronological_pipeline(df: pd.DataFrame, feature_cols: List[str],
                                target_col: str = 'fire_tomorrow',
                                train_frac: float = 0.7, val_frac: float = 0.15,
                                test_frac: float = 0.15) -> Dict:
    """
    Complete chronological splitting pipeline
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        train_frac: Training fraction
        val_frac: Validation fraction
        test_frac: Test fraction
    
    Returns:
        Dictionary with splits, scaler, and prepared data
    """
    print("Starting chronological splitting pipeline...")
    
    # Create chronological splits
    splits = create_chronological_split(df, train_frac, val_frac, test_frac)
    
    # Fit scaler on training data
    scaler = fit_scaler_on_train(splits['train'], feature_cols)
    
    # Apply scaler to all splits
    for split_name in splits:
        splits[split_name] = apply_scaler(splits[split_name], scaler, feature_cols)
    
    # Compute class weights
    class_weights = compute_class_weights(splits['train'], target_col)
    
    # Prepare data arrays
    prepared_data = prepare_data_for_training(splits, feature_cols, target_col)
    
    # Create pipeline result
    pipeline_result = {
        'splits': splits,
        'scaler': scaler,
        'class_weights': class_weights,
        'prepared_data': prepared_data,
        'feature_cols': feature_cols,
        'target_col': target_col
    }
    
    print("Chronological splitting pipeline complete")
    
    return pipeline_result

def validate_splits(splits: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate that splits are properly separated
    
    Args:
        splits: Dictionary with train/val/test DataFrames
    
    Returns:
        True if valid, raises error if not
    """
    print("Validating splits...")
    
    # Check that all splits exist
    required_splits = ['train', 'val', 'test']
    for split_name in required_splits:
        if split_name not in splits:
            raise ValueError(f"Missing required split: {split_name}")
    
    # Check for overlap in dates
    train_dates = set(splits['train']['date'].dt.date)
    val_dates = set(splits['val']['date'].dt.date)
    test_dates = set(splits['test']['date'].dt.date)
    
    # Check for overlaps
    train_val_overlap = train_dates.intersection(val_dates)
    train_test_overlap = train_dates.intersection(test_dates)
    val_test_overlap = val_dates.intersection(test_dates)
    
    if train_val_overlap:
        raise ValueError(f"Train/val overlap: {len(train_val_overlap)} dates")
    
    if train_test_overlap:
        raise ValueError(f"Train/test overlap: {len(train_test_overlap)} dates")
    
    if val_test_overlap:
        raise ValueError(f"Val/test overlap: {len(val_test_overlap)} dates")
    
    # Check chronological ordering
    train_max_date = splits['train']['date'].max()
    val_min_date = splits['val']['date'].min()
    val_max_date = splits['val']['date'].max()
    test_min_date = splits['test']['date'].min()
    
    if val_min_date <= train_max_date:
        raise ValueError("Validation set starts before training set ends")
    
    if test_min_date <= val_max_date:
        raise ValueError("Test set starts before validation set ends")
    
    print("  Split validation passed")
    return True
