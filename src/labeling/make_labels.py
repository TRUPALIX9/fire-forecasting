import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

from ..features.fwi import calculate_light_fwi_proxy, get_fwi_threshold
from ..data_sources.firms import create_fire_labels_for_sites
from ..data_sources.raws import get_site_coordinates

def create_firms_proximity_labels(weather_df: pd.DataFrame, firms_df: pd.DataFrame,
                                buffer_km: float = 15) -> pd.DataFrame:
    """
    Create fire labels based on FIRMS proximity (next-day fire detection)
    
    Args:
        weather_df: DataFrame with weather data (site, date, ...)
        firms_df: DataFrame with FIRMS fire data
        buffer_km: Buffer radius for fire detection
    
    Returns:
        DataFrame with site, date, and fire_tomorrow columns
    """
    print("Creating FIRMS proximity labels...")
    
    # Get unique sites and date range
    sites = weather_df['site'].unique()
    start_date = weather_df['date'].min()
    end_date = weather_df['date'].max()
    
    # Create labels using FIRMS data
    labels_df = create_fire_labels_for_sites(sites, start_date, end_date, firms_df, buffer_km)
    
    return labels_df

def create_fwi_threshold_labels(weather_df: pd.DataFrame, 
                               threshold_quantile: float = 0.85) -> pd.DataFrame:
    """
    Create fire labels based on FWI threshold
    
    Args:
        weather_df: DataFrame with weather data
        threshold_quantile: Quantile threshold for FWI (0.85 = 85th percentile)
    
    Returns:
        DataFrame with site, date, and fire_tomorrow columns
    """
    print("Creating FWI threshold labels...")
    
    # Calculate FWI proxy for each site
    sites = weather_df['site'].unique()
    
    all_labels = []
    
    for site in sites:
        print(f"  Processing {site}...")
        
        # Get weather data for this site
        site_weather = weather_df[weather_df['site'] == site].copy()
        
        if site_weather.empty:
            continue
        
        # Calculate FWI proxy
        try:
            site_weather['FWI_PROXY'] = calculate_light_fwi_proxy(site_weather)
        except Exception as e:
            print(f"    Error calculating FWI for {site}: {e}")
            continue
        
        # Get threshold for this site
        threshold = get_fwi_threshold(site_weather['FWI_PROXY'], threshold_quantile)
        
        # Create labels
        site_weather['fire_tomorrow'] = (site_weather['FWI_PROXY'] >= threshold).astype(int)
        
        # Add to results
        labels = site_weather[['site', 'date', 'fire_tomorrow']].copy()
        all_labels.append(labels)
    
    if not all_labels:
        raise ValueError("No FWI labels could be created")
    
    # Combine all labels
    labels_df = pd.concat(all_labels, ignore_index=True)
    
    # Calculate summary statistics
    total_records = len(labels_df)
    total_positives = labels_df['fire_tomorrow'].sum()
    positive_rate = total_positives / total_records
    
    print(f"  FWI threshold labeling complete:")
    print(f"    Total records: {total_records}")
    print(f"    Positive rate: {positive_rate:.3f} ({total_positives} fires)")
    print(f"    Threshold quantile: {threshold_quantile}")
    
    return labels_df

def merge_features_and_labels(weather_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather features with fire labels
    
    Args:
        weather_df: DataFrame with weather features
        labels_df: DataFrame with fire labels
    
    Returns:
        Merged DataFrame
    """
    print("Merging features and labels...")
    
    # Ensure both DataFrames have the same structure
    if 'date' in weather_df.columns:
        weather_df = weather_df.set_index(['site', 'date'])
    else:
        weather_df = weather_df.set_index(['site', 'date'])
    
    if 'date' in labels_df.columns:
        labels_df = labels_df.set_index(['site', 'date'])
    else:
        labels_df = labels_df.set_index(['site', 'date'])
    
    # Merge on site and date
    merged_df = weather_df.merge(labels_df, left_index=True, right_index=True, how='inner')
    
    # Reset index to get site and date as columns
    merged_df = merged_df.reset_index()
    
    # Ensure chronological ordering
    merged_df = merged_df.sort_values(['site', 'date'])
    
    print(f"  Merge complete: {len(merged_df)} records")
    
    return merged_df

def create_labels_pipeline(weather_df: pd.DataFrame, firms_df: pd.DataFrame,
                          config: Dict) -> pd.DataFrame:
    """
    Main pipeline for creating fire labels
    
    Args:
        weather_df: DataFrame with weather data
        firms_df: DataFrame with FIRMS fire data
        config: Configuration dictionary
    
    Returns:
        DataFrame with features and labels merged
    """
    print("Starting label creation pipeline...")
    
    # Get labeling configuration
    label_method = config.get('labeling', {}).get('label_method', 'firms')
    buffer_km = config.get('labeling', {}).get('buffer_km', 15)
    fwi_threshold_quantile = config.get('labeling', {}).get('fwi_threshold_quantile', 0.85)
    
    # Create labels based on method
    if label_method == 'firms':
        print(f"Using FIRMS proximity method (buffer: {buffer_km}km)")
        labels_df = create_firms_proximity_labels(weather_df, firms_df, buffer_km)
    elif label_method == 'fwi':
        print(f"Using FWI threshold method (quantile: {fwi_threshold_quantile})")
        labels_df = create_fwi_threshold_labels(weather_df, fwi_threshold_quantile)
    else:
        raise ValueError(f"Unknown labeling method: {label_method}")
    
    # Merge features and labels
    merged_df = merge_features_and_labels(weather_df, labels_df)
    
    # Validate merged data
    validate_labeled_data(merged_df)
    
    print("Label creation pipeline complete")
    
    return merged_df

def validate_labeled_data(df: pd.DataFrame) -> bool:
    """
    Validate the labeled dataset
    
    Args:
        df: DataFrame with features and labels
    
    Returns:
        True if valid, raises error if not
    """
    print("Validating labeled data...")
    
    # Check required columns
    required_cols = ['site', 'date', 'fire_tomorrow']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values in critical columns
    critical_cols = ['site', 'date', 'fire_tomorrow']
    for col in critical_cols:
        if df[col].isna().any():
            raise ValueError(f"Found NaN values in {col}")
    
    # Check label distribution
    label_counts = df['fire_tomorrow'].value_counts()
    total_records = len(df)
    
    print(f"  Label distribution:")
    print(f"    Total records: {total_records}")
    print(f"    No fire (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total_records:.3f})")
    print(f"    Fire (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total_records:.3f})")
    
    # Check for reasonable positive rate (should be low for wildfires)
    positive_rate = label_counts.get(1, 0) / total_records
    if positive_rate > 0.1:  # More than 10% would be unusual
        print(f"  Warning: High positive rate ({positive_rate:.3f}) - check data quality")
    
    # Check chronological ordering
    sites = df['site'].unique()
    for site in sites:
        site_data = df[df['site'] == site].sort_values('date')
        if not site_data['date'].is_monotonic_increasing:
            raise ValueError(f"Dates not in chronological order for site: {site}")
    
    print("  Data validation passed")
    return True

def get_labeling_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for the labeled dataset
    
    Args:
        df: DataFrame with features and labels
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'total_sites': df['site'].nunique(),
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d')
        },
        'label_distribution': df['fire_tomorrow'].value_counts().to_dict(),
        'positive_rate': df['fire_tomorrow'].mean(),
        'records_per_site': df.groupby('site').size().to_dict(),
        'site_positive_rates': df.groupby('site')['fire_tomorrow'].mean().to_dict()
    }
    
    return summary

def create_ablation_labels(weather_df: pd.DataFrame, firms_df: pd.DataFrame,
                          config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Create labels using multiple methods for ablation study
    
    Args:
        weather_df: DataFrame with weather data
        firms_df: DataFrame with FIRMS fire data
        config: Configuration dictionary
    
    Returns:
        Dictionary with different labeling methods
    """
    print("Creating ablation labels...")
    
    ablation_results = {}
    
    # FIRMS proximity labels
    print("  Creating FIRMS proximity labels...")
    firms_labels = create_firms_proximity_labels(weather_df, firms_df, buffer_km=15)
    ablation_results['firms'] = firms_labels
    
    # FWI threshold labels
    print("  Creating FWI threshold labels...")
    fwi_labels = create_fwi_threshold_labels(weather_df, threshold_quantile=0.85)
    ablation_results['fwi'] = fwi_labels
    
    # Merge both with features
    print("  Merging features with labels...")
    for method, labels in ablation_results.items():
        merged = merge_features_and_labels(weather_df, labels)
        ablation_results[method] = merged
    
    print("Ablation labels complete")
    
    return ablation_results
