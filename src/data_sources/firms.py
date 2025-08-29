import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import time
import warnings
from pathlib import Path
from .raws import get_site_coordinates

def create_tri_county_bbox() -> Tuple[float, float, float, float]:
    """Create bounding box for Tri-County area (Ventura, Santa Barbara, Los Angeles)"""
    # (min_lat, min_lon, max_lat, max_lon)
    return (33.5, -121.0, 35.0, -117.5)

def fetch_firms_data(start_date: datetime, end_date: datetime, 
                    cache_dir: str = "data", min_confidence: str = "0") -> pd.DataFrame:
    """
    Fetch FIRMS data for the specified date range
    This is a mock implementation - in practice you'd fetch from NASA FIRMS API
    """
    # Create cache directory
    cache_path = Path(cache_dir) / "firms"
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Cache file path
    cache_file = cache_path / f"firms_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    
    # Check if data is cached
    if cache_file.exists():
        print(f"  Loading cached FIRMS data")
        return pd.read_csv(cache_file, parse_dates=['acq_date'])
    
    print(f"  Fetching FIRMS data (mock data)")
    
    # Get site coordinates for realistic fire placement
    site_coords = get_site_coordinates()
    
    # Generate mock fire data based on historical patterns
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base fire probability (higher in summer/fall)
    base_fire_prob = 0.02  # 2% chance per day per site
    
    fire_data = []
    
    for date in date_range:
        # Seasonal fire probability
        day_of_year = date.timetuple().tm_yday
        
        # Higher fire probability in summer/fall (June-October)
        if 152 <= day_of_year <= 304:  # June 1 - October 31
            seasonal_multiplier = 3.0
        else:
            seasonal_multiplier = 0.5
        
        # Check each site for potential fires
        for site_name, (lat, lon) in site_coords.items():
            # Calculate fire probability for this site and date
            site_fire_prob = base_fire_prob * seasonal_multiplier
            
            # Add some randomness
            if np.random.random() < site_fire_prob:
                # Generate fire point near the site
                # Add some offset to simulate fire spread
                fire_lat = lat + np.random.normal(0, 0.01)  # ~1km offset
                fire_lon = lon + np.random.normal(0, 0.01)
                
                # Fire properties
                confidence = np.random.choice(['0', 'nominal', 'high'], p=[0.3, 0.5, 0.2])
                
                # Only include fires that meet confidence threshold
                if min_confidence == "0" or confidence == min_confidence:
                    fire_data.append({
                        'acq_date': date,
                        'acq_time': f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}",
                        'latitude': round(fire_lat, 4),
                        'longitude': round(fire_lon, 4),
                        'confidence': confidence,
                        'frp': np.random.exponential(50),  # Fire Radiative Power
                        'satellite': np.random.choice(['MODIS', 'VIIRS']),
                        'site_nearby': site_name
                    })
    
    if not fire_data:
        # Create a few fires to ensure we have some data
        for i in range(5):
            site_name = list(site_coords.keys())[i % len(site_coords)]
            lat, lon = site_coords[site_name]
            
            fire_data.append({
                'acq_date': start_date + timedelta(days=i*30),
                'acq_time': "12:00",
                'latitude': round(lat + np.random.normal(0, 0.01), 4),
                'longitude': round(lon + np.random.normal(0, 0.01), 4),
                'confidence': 'nominal',
                'frp': np.random.exponential(50),
                'satellite': 'VIIRS',
                'site_nearby': site_name
            })
    
    df = pd.DataFrame(fire_data)
    
    # Cache the data
    df.to_csv(cache_file, index=False)
    
    return df

def filter_firms_by_bbox(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """Filter FIRMS data to only points within bounding box"""
    min_lat, min_lon, max_lat, max_lon = bbox
    
    mask = (
        (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) &
        (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
    )
    
    return df[mask].copy()

def has_next_day_fire(site_lat: float, site_lon: float, date: datetime, 
                      firms_df: pd.DataFrame, buffer_km: float = 15) -> bool:
    """
    Check if there's a fire within buffer_km of a site on the next day
    
    Args:
        site_lat: Site latitude
        site_lon: Site longitude
        date: Current date
        firms_df: FIRMS DataFrame
        buffer_km: Buffer radius in kilometers
    
    Returns:
        True if fire detected within buffer on next day
    """
    # Convert buffer from km to degrees (approximate)
    buffer_deg = buffer_km / 111.0
    
    # Next day
    next_date = date + timedelta(days=1)
    
    # Filter FIRMS data for next day
    next_day_fires = firms_df[firms_df['acq_date'] == next_date]
    
    if next_day_fires.empty:
        return False
    
    # Check if any fire is within buffer
    for _, fire in next_day_fires.iterrows():
        fire_lat = fire['latitude']
        fire_lon = fire['longitude']
        
        # Calculate distance
        lat_diff = abs(fire_lat - site_lat)
        lon_diff = abs(fire_lon - site_lon)
        
        # Simple distance check (approximate)
        if lat_diff <= buffer_deg and lon_diff <= buffer_deg:
            # More precise distance calculation
            distance_km = haversine_distance(site_lat, site_lon, fire_lat, fire_lon)
            if distance_km <= buffer_km:
                return True
    
    return False

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points"""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def create_fire_labels_for_sites(sites: List[str], start_date: datetime, end_date: datetime,
                                firms_df: pd.DataFrame, buffer_km: float = 15) -> pd.DataFrame:
    """
    Create fire labels for all sites based on FIRMS data
    
    Args:
        sites: List of site names
        start_date: Start date for labeling
        end_date: End date for labeling
        firms_df: FIRMS DataFrame
        buffer_km: Buffer radius for fire detection
    
    Returns:
        DataFrame with site, date, and fire_tomorrow columns
    """
    print(f"Creating fire labels for {len(sites)} sites...")
    
    # Get site coordinates
    site_coords = get_site_coordinates()
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    label_data = []
    
    for site in sites:
        if site not in site_coords:
            print(f"  Warning: No coordinates found for {site}, skipping")
            continue
        
        site_lat, site_lon = site_coords[site]
        print(f"  Processing {site}...")
        
        for date in date_range:
            # Check if there's a fire tomorrow within buffer
            fire_tomorrow = has_next_day_fire(site_lat, site_lon, date, firms_df, buffer_km)
            
            label_data.append({
                'site': site,
                'date': date,
                'site_lat': site_lat,
                'site_lon': site_lon,
                'fire_tomorrow': int(fire_tomorrow)
            })
    
    labels_df = pd.DataFrame(label_data)
    
    # Calculate summary statistics
    total_days = len(date_range)
    total_sites = len(sites)
    total_positives = labels_df['fire_tomorrow'].sum()
    positive_rate = total_positives / (total_days * total_sites)
    
    print(f"  Label creation complete:")
    print(f"    Total records: {len(labels_df)}")
    print(f"    Positive rate: {positive_rate:.3f} ({total_positives} fires)")
    
    return labels_df

def get_firms_summary_stats(firms_df: pd.DataFrame) -> Dict:
    """Get summary statistics for FIRMS data"""
    if firms_df.empty:
        return {}
    
    summary = {
        'total_fires': len(firms_df),
        'date_range': {
            'start': firms_df['acq_date'].min().strftime('%Y-%m-%d'),
            'end': firms_df['acq_date'].max().strftime('%Y-%m-%d')
        },
        'confidence_distribution': firms_df['confidence'].value_counts().to_dict(),
        'satellite_distribution': firms_df['satellite'].value_counts().to_dict(),
        'avg_frp': firms_df['frp'].mean(),
        'max_frp': firms_df['frp'].max(),
        'sites_with_fires': firms_df['site_nearby'].nunique()
    }
    
    return summary

def validate_firms_data(df: pd.DataFrame) -> bool:
    """Validate FIRMS data quality"""
    if df.empty:
        return False
    
    # Check for required columns
    required_cols = ['acq_date', 'latitude', 'longitude', 'confidence', 'frp']
    if not all(col in df.columns for col in required_cols):
        return False
    
    # Check for reasonable value ranges
    if not df['latitude'].between(-90, 90).all():
        return False
    
    if not df['longitude'].between(-180, 180).all():
        return False
    
    if not df['frp'].between(0, 10000).all():
        return False
    
    # Check confidence values
    valid_confidence = ['0', 'nominal', 'high']
    if not df['confidence'].isin(valid_confidence).all():
        return False
    
    return True
