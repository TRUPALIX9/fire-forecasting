import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import time
import warnings
from pathlib import Path

# Site coordinates for Tri-County area (approximate)
SITE_COORDINATES = {
    # Santa Barbara County
    "Montecito": (34.4344, -119.6343),
    "Hope Ranch": (34.4567, -119.7234),
    "San Roque / La Cumbre (foothills)": (34.4567, -119.7234),
    "Santa Barbara - The Mesa": (34.4233, -119.7023),
    
    # Ventura County
    "Thousand Oaks (WUI canyons)": (34.1706, -118.8376),
    "Newbury Park (WUI canyons)": (34.1794, -118.9181),
    "Moorpark (expanded VHFHSZ)": (34.2856, -118.8770),
    "Simi Valley (canyon edges)": (34.2694, -118.7815),
    "Ojai Valley": (34.4481, -119.2429),
    "Ventura Hillsides (Clearpoint/Ondulando/Skyline)": (34.2744, -119.2295),
    "Fillmore (north/inland)": (34.3992, -118.9181),
    "Santa Paula (north/inland)": (34.3544, -119.0592),
    
    # Los Angeles County
    "Malibu (Santa Monica Mountains)": (34.0259, -118.7798),
    "Santa Monica Mountains Interior": (34.1000, -118.7000),
    "Glendale / Pasadena / Altadena / La Cañada Foothills": (34.1425, -118.2551),
    "Bel Air / Brentwood / Hollywood Hills / Pacific Palisades / Woodland Hills": (34.1000, -118.4500),
    "Chatsworth / Porter Ranch (Santa Susana)": (34.2572, -118.6012),
    "Eagle Rock / Highland Park / El Sereno (Eastside WUI)": (34.1333, -118.2000),
    "Palos Verdes Peninsula": (33.7833, -118.3833),
    "Antelope Valley fringe (Lancaster/Palmdale WUI)": (34.6867, -118.1542),
}

# RAWS station catalog (simplified - in practice you'd fetch this from NOAA)
RAWS_STATIONS = {
    "Montecito": "MONTECITO",
    "Hope Ranch": "HOPE RANCH",
    "San Roque / La Cumbre (foothills)": "SAN ROQUE",
    "Santa Barbara - The Mesa": "SANTA BARBARA",
    "Thousand Oaks (WUI canyons)": "THOUSAND OAKS",
    "Newbury Park (WUI canyons)": "NEWBURY PARK",
    "Moorpark (expanded VHFHSZ)": "MOORPARK",
    "Simi Valley (canyon edges)": "SIMI VALLEY",
    "Ojai Valley": "OJAI",
    "Ventura Hillsides (Clearpoint/Ondulando/Skyline)": "VENTURA",
    "Fillmore (north/inland)": "FILLMORE",
    "Santa Paula (north/inland)": "SANTA PAULA",
    "Malibu (Santa Monica Mountains)": "MALIBU",
    "Santa Monica Mountains Interior": "SANTA MONICA MTNS",
    "Glendale / Pasadena / Altadena / La Cañada Foothills": "GLENDALE",
    "Bel Air / Brentwood / Hollywood Hills / Pacific Palisades / Woodland Hills": "HOLLYWOOD HILLS",
    "Chatsworth / Porter Ranch (Santa Susana)": "CHATSWORTH",
    "Eagle Rock / Highland Park / El Sereno (Eastside WUI)": "EAGLE ROCK",
    "Palos Verdes Peninsula": "PALOS VERDES",
    "Antelope Valley fringe (Lancaster/Palmdale WUI)": "LANCASTER",
}

def get_nearest_raws_station(site_lat: float, site_lon: float, 
                            station_catalog: Optional[Dict] = None) -> str:
    """
    Find the nearest RAWS station to a given site
    This is a simplified version - in practice you'd use actual station coordinates
    """
    if station_catalog is None:
        # Use default station mapping
        return "SANTA BARBARA"  # Default fallback
    
    # For now, return a reasonable default based on region
    if site_lat > 34.5:  # Northern region
        return "LANCASTER"
    elif site_lat > 34.0:  # Central region
        return "SANTA BARBARA"
    else:  # Southern region
        return "MALIBU"

def fetch_raws_data_daily(station_id: str, start_date: datetime, 
                          end_date: datetime, cache_dir: str = "data") -> pd.DataFrame:
    """
    Fetch daily RAWS data for a station
    This is a mock implementation - in practice you'd fetch from NOAA/RAWS API
    """
    # Create cache directory
    cache_path = Path(cache_dir) / "raws"
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Cache file path
    cache_file = cache_path / f"{station_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    
    # Check if data is cached
    if cache_file.exists():
        print(f"  Loading cached RAWS data for {station_id}")
        df_cached = pd.read_csv(cache_file, parse_dates=['date'])
        df_cached['date'] = pd.to_datetime(df_cached['date']).dt.normalize()
        return df_cached
    
    print(f"  Fetching RAWS data for {station_id} (mock data)")
    
    # Generate mock daily weather data
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base weather patterns (simplified)
    base_temp = 20  # Base temperature in Celsius
    base_rh = 60    # Base relative humidity
    
    data = []
    for date in date_range:
        # Seasonal temperature variation
        day_of_year = date.timetuple().tm_yday
        seasonal_temp = base_temp + 10 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        
        # Add some randomness
        temp_max = seasonal_temp + np.random.normal(5, 3)
        temp_min = seasonal_temp + np.random.normal(-5, 3)
        
        # Relative humidity (inverse relationship with temperature)
        rh = max(20, min(100, base_rh - (temp_max - base_temp) * 2 + np.random.normal(0, 10)))
        
        # Wind speed (higher in afternoon)
        wind_speed = np.random.exponential(10) + np.random.normal(5, 2)
        wind_speed = min(wind_speed, 50)  # Cap at 50 km/h
        
        # Wind direction (prevailing westerly)
        wind_dir = np.random.normal(270, 45)  # West is 270 degrees
        wind_dir = wind_dir % 360
        
        # Precipitation (mostly in winter)
        if date.month in [12, 1, 2, 3]:  # Winter months
            precip_prob = 0.3
        else:
            precip_prob = 0.05
        
        if np.random.random() < precip_prob:
            precip = np.random.exponential(5)  # mm
        else:
            precip = 0
        
        data.append({
            'date': date,
            'TMAX': round(temp_max, 1),
            'TMIN': round(temp_min, 1),
            'RH': round(rh, 1),
            'WIND_SPD': round(wind_speed, 1),
            'WIND_DIR': round(wind_dir, 1),
            'PRCP': round(precip, 1)
        })
    
    df = pd.DataFrame(data)
    # Ensure proper datetime dtype normalized to day
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    # Cache the data
    df.to_csv(cache_file, index=False)
    
    return df

def standardize_raws_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize RAWS data columns and units
    """
    df = df.copy()
    
    # Ensure required columns exist
    required_cols = ['date', 'TMAX', 'TMIN', 'RH', 'WIND_SPD', 'WIND_DIR', 'PRCP']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert temperature to Celsius if needed
    if df['TMAX'].max() > 100:  # Likely Fahrenheit
        df['TMAX'] = (df['TMAX'] - 32) * 5/9
        df['TMIN'] = (df['TMIN'] - 32) * 5/9
    
    # Ensure wind speed is in km/h
    if df['WIND_SPD'].max() < 50:  # Likely m/s
        df['WIND_SPD'] = df['WIND_SPD'] * 3.6
    
    # Ensure precipitation is in mm
    if df['PRCP'].max() < 10:  # Likely inches
        df['PRCP'] = df['PRCP'] * 25.4
    
    # Round values
    df['TMAX'] = df['TMAX'].round(1)
    df['TMIN'] = df['TMIN'].round(1)
    df['RH'] = df['RH'].round(1)
    df['WIND_SPD'] = df['WIND_SPD'].round(1)
    df['WIND_DIR'] = df['WIND_DIR'].round(1)
    df['PRCP'] = df['PRCP'].round(1)
    
    return df

def impute_missing_values(df: pd.DataFrame, max_gap_days: int = 3) -> pd.DataFrame:
    """
    Impute missing values in RAWS data
    """
    df = df.copy()
    
    # Set date as index
    df = df.set_index('date')
    
    # Sort by date
    df = df.sort_index()
    
    # Check for gaps
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    missing_dates = date_range.difference(df.index)
    
    if len(missing_dates) > 0:
        print(f"  Found {len(missing_dates)} missing dates")
        
        # Add missing dates with NaN values
        df = df.reindex(date_range)
        
        # Forward fill for small gaps
        df = df.fillna(method='ffill', limit=max_gap_days)
        
        # Backward fill for remaining gaps
        df = df.fillna(method='bfill', limit=max_gap_days)
        
        # Linear interpolation for remaining gaps
        df = df.interpolate(method='linear', limit_direction='both', limit=max_gap_days)
        
        # Drop rows that still have NaN values (large gaps)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        if initial_rows != final_rows:
            print(f"  Dropped {initial_rows - final_rows} rows with large gaps")
    
    return df

def fetch_all_sites_raws_data(sites: List[str], start_date: datetime, 
                              end_date: datetime, cache_dir: str = "data") -> pd.DataFrame:
    """
    Fetch RAWS data for all sites
    """
    print(f"Fetching RAWS data for {len(sites)} sites...")
    
    all_data = []
    
    for site in sites:
        print(f"  Processing {site}...")
        
        # Get site coordinates
        if site not in SITE_COORDINATES:
            print(f"    Warning: No coordinates found for {site}, skipping")
            continue
        
        site_lat, site_lon = SITE_COORDINATES[site]
        
        # Get nearest station
        station_id = get_nearest_raws_station(site_lat, site_lon)
        
        try:
            # Fetch data
            site_data = fetch_raws_data_daily(station_id, start_date, end_date, cache_dir)
            
            # Add site identifier
            site_data['site'] = site
            site_data['site_lat'] = site_lat
            site_data['site_lon'] = site_lon
            
            all_data.append(site_data)
            
            print(f"    Successfully fetched {len(site_data)} records")
            
        except Exception as e:
            print(f"    Error fetching data for {site}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No RAWS data could be fetched for any sites")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Standardize data
    print("Standardizing RAWS data...")
    combined_df = standardize_raws_data(combined_df)
    
    # Impute missing values
    print("Imputing missing values...")
    combined_df = impute_missing_values(combined_df)
    
    # Materialize 'date' column from index for downstream merges
    if isinstance(combined_df.index, pd.DatetimeIndex):
        combined_df = combined_df.reset_index().rename(columns={'index': 'date'})
        combined_df['date'] = pd.to_datetime(combined_df['date']).dt.normalize()
    
    print(f"RAWS data fetch complete: {len(combined_df)} total records")
    
    return combined_df

def get_site_coordinates() -> Dict[str, Tuple[float, float]]:
    """Get site coordinates dictionary"""
    return SITE_COORDINATES.copy()

def validate_raws_data(df: pd.DataFrame) -> bool:
    """
    Validate RAWS data quality
    """
    if df.empty:
        return False
    
    # Check for required columns
    required_cols = ['site', 'date', 'TMAX', 'TMIN', 'RH', 'WIND_SPD', 'WIND_DIR', 'PRCP']
    if not all(col in df.columns for col in required_cols):
        return False
    
    # Check for reasonable value ranges
    if not (df['TMAX'].between(-50, 60).all() and df['TMIN'].between(-50, 60).all()):
        return False
    
    if not df['RH'].between(0, 100).all():
        return False
    
    if not df['WIND_SPD'].between(0, 200).all():
        return False
    
    if not df['WIND_DIR'].between(0, 360).all():
        return False
    
    if not df['PRCP'].between(0, 1000).all():
        return False
    
    return True
