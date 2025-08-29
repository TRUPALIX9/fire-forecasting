import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import warnings

def add_lag_features(df: pd.DataFrame, lag_days: int = 7, 
                    feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add lag features for specified columns
    
    Args:
        df: DataFrame with datetime index
        lag_days: Number of days to lag
        feature_cols: Columns to create lags for (default: weather columns)
    
    Returns:
        DataFrame with lag features added
    """
    if feature_cols is None:
        feature_cols = ['TMAX', 'TMIN', 'RH', 'WIND_SPD', 'WIND_DIR', 'PRCP']
    
    df = df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have datetime index or 'date' column")
    
    # Sort by date
    df = df.sort_index()
    
    # Create lag features
    for col in feature_cols:
        if col in df.columns:
            for lag in range(1, lag_days + 1):
                lag_col = f"{col}_lag{lag}"
                df[lag_col] = df[col].shift(lag)
    
    return df

def add_rolling_features(df: pd.DataFrame, window: int = 7,
                        feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add rolling window features (mean, std, min, max)
    
    Args:
        df: DataFrame with datetime index
        window: Rolling window size in days
        feature_cols: Columns to create rolling features for
    
    Returns:
        DataFrame with rolling features added
    """
    if feature_cols is None:
        feature_cols = ['TMAX', 'TMIN', 'RH', 'WIND_SPD', 'PRCP']
    
    df = df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have datetime index or 'date' column")
    
    # Sort by date
    df = df.sort_index()
    
    # Create rolling features
    for col in feature_cols:
        if col in df.columns:
            # Rolling mean
            df[f"{col}_roll{window}_mean"] = df[col].rolling(window=window, min_periods=1).mean()
            
            # Rolling std
            df[f"{col}_roll{window}_std"] = df[col].rolling(window=window, min_periods=1).std()
            
            # Rolling min
            df[f"{col}_roll{window}_min"] = df[col].rolling(window=window, min_periods=1).min()
            
            # Rolling max
            df[f"{col}_roll{window}_max"] = df[col].rolling(window=window, min_periods=1).max()
            
            # Rolling sum (for precipitation)
            if col == 'PRCP':
                df[f"{col}_roll{window}_sum"] = df[col].rolling(window=window, min_periods=1).sum()
    
    return df

def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seasonal features (day of year, sin/cos transformations)
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame with seasonal features added
    """
    df = df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have datetime index or 'date' column")
    
    # Day of year (1-366)
    df['day_of_year'] = df.index.dayofyear
    
    # Sin and cos transformations for day of year
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 366)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 366)
    
    # Month (1-12)
    df['month'] = df.index.month
    
    # Sin and cos transformations for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Season (1-4: Winter, Spring, Summer, Fall)
    df['season'] = pd.cut(df.index.month, bins=[0, 3, 6, 9, 12], labels=[1, 2, 3, 4])
    
    return df

def add_neighbor_features(df: pd.DataFrame, site_coords: Dict[str, Tuple[float, float]],
                         neighbor_radius_km: float = 60, k_neighbors: int = 1) -> pd.DataFrame:
    """
    Add neighbor site features using inverse distance weighting
    
    Args:
        df: DataFrame with site and date columns
        site_coords: Dictionary mapping site names to (lat, lon) tuples
        neighbor_radius_km: Maximum distance to consider neighbors
        k_neighbors: Number of nearest neighbors to use
    
    Returns:
        DataFrame with neighbor features added
    """
    df = df.copy()
    
    # Ensure we have site and date columns
    if 'site' not in df.columns:
        raise ValueError("DataFrame must have 'site' column")
    
    # Create site coordinates DataFrame
    sites_df = pd.DataFrame([
        {'site': site, 'lat': lat, 'lon': lon}
        for site, (lat, lon) in site_coords.items()
    ])
    
    # Calculate distances between all sites
    distances = {}
    for site1 in site_coords:
        for site2 in site_coords:
            if site1 != site2:
                lat1, lon1 = site_coords[site1]
                lat2, lon2 = site_coords[site2]
                
                # Haversine distance
                R = 6371  # Earth radius in km
                lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
                    np.radians, [lat1, lon1, lat2, lon2]
                )
                dlat = lat2_rad - lat1_rad
                dlon = lon2_rad - lon1_rad
                a = (np.sin(dlat/2)**2 + 
                     np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
                c = 2 * np.arcsin(np.sqrt(a))
                distance = R * c
                
                distances[(site1, site2)] = distance
    
    # For each site, find k nearest neighbors within radius
    neighbor_features = {}
    
    for site in site_coords:
        # Get distances to other sites
        site_distances = [
            (other_site, dist) for (site1, other_site), dist in distances.items()
            if site1 == site and dist <= neighbor_radius_km
        ]
        
        # Sort by distance and take k nearest
        site_distances.sort(key=lambda x: x[1])
        nearest_neighbors = site_distances[:k_neighbors]
        
        if nearest_neighbors:
            # Calculate inverse distance weights
            total_weight = sum(1/dist for _, dist in nearest_neighbors)
            weights = [(other_site, 1/dist/total_weight) for other_site, dist in nearest_neighbors]
            
            neighbor_features[site] = weights
        else:
            neighbor_features[site] = []
    
    # Add neighbor lag1 features
    weather_cols = ['TMAX', 'TMIN', 'RH', 'WIND_SPD', 'PRCP']
    
    for site in site_coords:
        if neighbor_features[site]:
            for col in weather_cols:
                if col in df.columns:
                    # Create weighted average of neighbor lag1 values
                    neighbor_col = f"{col}_neighbor_lag1"
                    
                    def get_neighbor_value(row):
                        if row['site'] == site:
                            # Get neighbor values for the previous day
                            neighbor_values = []
                            neighbor_weights = []
                            
                            for neighbor_site, weight in neighbor_features[site]:
                                # Find neighbor's value for the previous day
                                neighbor_row = df[
                                    (df['site'] == neighbor_site) & 
                                    (df.index == row.name - pd.Timedelta(days=1))
                                ]
                                
                                if not neighbor_row.empty:
                                    neighbor_values.append(neighbor_row[col].iloc[0])
                                    neighbor_weights.append(weight)
                            
                            if neighbor_values:
                                # Weighted average
                                return np.average(neighbor_values, weights=neighbor_weights)
                            else:
                                return np.nan
                        else:
                            return np.nan
                    
                    df[neighbor_col] = df.apply(get_neighbor_value, axis=1)
    
    return df

def add_static_features(df: pd.DataFrame, site_coords: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Add static site features (latitude, longitude, elevation proxy)
    
    Args:
        df: DataFrame with site column
        site_coords: Dictionary mapping site names to (lat, lon) tuples
    
    Returns:
        DataFrame with static features added
    """
    df = df.copy()
    
    # Create site coordinates DataFrame
    sites_df = pd.DataFrame([
        {'site': site, 'lat': lat, 'lon': lon}
        for site, (lat, lon) in site_coords.items()
    ])
    
    # Merge coordinates
    df = df.merge(sites_df, on='site', how='left')
    
    # Add elevation proxy (rough approximation based on latitude)
    # This is a simplified approach - in practice you'd use actual elevation data
    df['elevation_proxy'] = (df['lat'] - 33.5) * 1000  # Rough meters above sea level
    
    # Add distance from coast (rough approximation)
    # Assuming coast is roughly at longitude -118.5
    df['distance_from_coast_km'] = abs(df['lon'] - (-118.5)) * 111  # Rough km conversion
    
    return df

def engineer_all_features(df: pd.DataFrame, config: Dict, 
                         site_coords: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Apply all feature engineering steps
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        site_coords: Site coordinates dictionary
    
    Returns:
        DataFrame with all features engineered
    """
    print("Engineering features...")
    
    # Get configuration
    lag_days = config.get('features', {}).get('lag_days', 7)
    roll_window = config.get('features', {}).get('roll_window', 7)
    use_neighbors = config.get('features', {}).get('use_neighbors', True)
    neighbor_radius_km = config.get('features', {}).get('neighbor_radius_km', 60)
    k_neighbors = config.get('features', {}).get('k_neighbors', 1)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have datetime index or 'date' column")
    
    # Sort by date
    df = df.sort_index()
    
    # Add lag features
    print(f"  Adding lag features (1-{lag_days} days)...")
    df = add_lag_features(df, lag_days=lag_days)
    
    # Add rolling features
    print(f"  Adding rolling features (window={roll_window})...")
    df = add_rolling_features(df, window=roll_window)
    
    # Add seasonal features
    print("  Adding seasonal features...")
    df = add_seasonal_features(df)
    
    # Add static features
    print("  Adding static features...")
    df = add_static_features(df, site_coords)
    
    # Add neighbor features if enabled
    if use_neighbors and k_neighbors > 0:
        print(f"  Adding neighbor features (k={k_neighbors}, radius={neighbor_radius_km}km)...")
        df = add_neighbor_features(df, site_coords, neighbor_radius_km, k_neighbors)
    
    # Drop rows with NaN values (from lag features)
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    
    print(f"  Feature engineering complete: {initial_rows} -> {final_rows} rows")
    print(f"  Final feature count: {len(df.columns)}")
    
    return df

def get_feature_columns(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Get list of feature columns (exclude target and metadata columns)
    
    Args:
        df: DataFrame
        exclude_cols: Additional columns to exclude
    
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Standard columns to exclude
    standard_exclude = ['site', 'date', 'target', 'fire_tomorrow', 'label']
    
    # Combine exclusions
    all_exclude = list(set(standard_exclude + exclude_cols))
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in all_exclude]
    
    return feature_cols
