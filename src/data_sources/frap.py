import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import json

def create_tri_county_bbox() -> Tuple[float, float, float, float]:
    """Create bounding box for Tri-County area (Ventura, Santa Barbara, Los Angeles)"""
    # (min_lat, min_lon, max_lat, max_lon)
    return (33.5, -121.0, 35.0, -117.5)

def fetch_frap_data(start_year: int = 2017, end_year: int = 2024, 
                   cache_dir: str = "data") -> gpd.GeoDataFrame:
    """
    Fetch FRAP data for the specified year range
    This is a mock implementation - in practice you'd fetch from CAL FIRE FRAP API
    """
    # Create cache directory
    cache_path = Path(cache_dir) / "frap"
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Cache file path
    cache_file = cache_path / f"frap_{start_year}_{end_year}.geojson"
    
    # Check if data is cached
    if cache_file.exists():
        print(f"  Loading cached FRAP data")
        return gpd.read_file(cache_file)
    
    print(f"  Fetching FRAP data (mock data)")
    
    # Get Tri-County bounding box
    bbox = create_tri_county_bbox()
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # Generate mock fire perimeters based on historical patterns
    fire_perimeters = []
    
    # Major historical fires in Tri-County area
    major_fires = [
        {
            'name': 'Thomas Fire',
            'year': 2017,
            'start_date': '2017-12-04',
            'end_date': '2018-01-12',
            'acres': 281893,
            'county': 'Ventura',
            'center_lat': 34.4,
            'center_lon': -119.2
        },
        {
            'name': 'Woolsey Fire',
            'year': 2018,
            'start_date': '2018-11-08',
            'end_date': '2018-11-22',
            'acres': 96949,
            'county': 'Los Angeles',
            'center_lat': 34.1,
            'center_lon': -118.8
        },
        {
            'name': 'Cave Fire',
            'year': 2019,
            'start_date': '2019-11-25',
            'end_date': '2019-12-02',
            'acres': 3231,
            'county': 'Santa Barbara',
            'center_lat': 34.5,
            'center_lon': -119.7
        },
        {
            'name': 'Bobcat Fire',
            'year': 2020,
            'start_date': '2020-09-06',
            'end_date': '2020-10-18',
            'acres': 115574,
            'county': 'Los Angeles',
            'center_lat': 34.2,
            'center_lon': -118.0
        },
        {
            'name': 'Alisal Fire',
            'year': 2021,
            'start_date': '2021-10-11',
            'end_date': '2021-10-20',
            'acres': 16802,
            'county': 'Santa Barbara',
            'center_lat': 34.5,
            'center_lon': -120.1
        }
    ]
    
    # Generate additional smaller fires
    for year in range(start_year, end_year + 1):
        # Number of fires per year (varies)
        if year in [2017, 2018, 2020]:  # High fire years
            num_fires = np.random.randint(8, 15)
        else:
            num_fires = np.random.randint(3, 8)
        
        for i in range(num_fires):
            # Random location within Tri-County area
            fire_lat = np.random.uniform(min_lat, max_lat)
            fire_lon = np.random.uniform(min_lon, max_lon)
            
            # Fire size (acres) - log-normal distribution
            fire_size_acres = np.random.lognormal(7, 1.5)  # Most fires small, some large
            fire_size_acres = min(fire_size_acres, 50000)  # Cap at 50k acres
            
            # Fire duration (days)
            if fire_size_acres > 10000:
                duration_days = np.random.randint(7, 30)
            else:
                duration_days = np.random.randint(1, 7)
            
            # Start date (mostly summer/fall)
            if year == 2017:
                start_month = np.random.choice([6, 7, 8, 9, 10, 11, 12])
            else:
                start_month = np.random.choice([6, 7, 8, 9, 10, 11])
            
            start_day = np.random.randint(1, 28)
            start_date = datetime(year, start_month, start_day)
            end_date = start_date + timedelta(days=duration_days)
            
            # Create fire name
            fire_name = f"Mock Fire {year}-{i+1:03d}"
            
            # Determine county based on location
            if fire_lat > 34.3:
                county = "Santa Barbara"
            elif fire_lat > 34.0:
                county = "Ventura"
            else:
                county = "Los Angeles"
            
            # Create simple polygon (circle approximation)
            # Convert acres to approximate radius in degrees
            radius_deg = np.sqrt(fire_size_acres / 1000) * 0.01  # Rough conversion
            
            # Create polygon points
            angles = np.linspace(0, 2*np.pi, 16)
            polygon_points = []
            for angle in angles:
                lat = fire_lat + radius_deg * np.cos(angle)
                lon = fire_lon + radius_deg * np.sin(angle)
                polygon_points.append([lon, lat])  # GeoJSON uses [lon, lat]
            
            # Close polygon
            polygon_points.append(polygon_points[0])
            
            # Create polygon
            polygon = Polygon(polygon_points)
            
            fire_perimeters.append({
                'geometry': polygon,
                'fire_name': fire_name,
                'year': year,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'acres': round(fire_size_acres),
                'county': county,
                'center_lat': fire_lat,
                'center_lon': fire_lon,
                'duration_days': duration_days
            })
    
    # Add major historical fires
    for fire in major_fires:
        if start_year <= fire['year'] <= end_year:
            # Create larger polygon for major fires
            radius_deg = np.sqrt(fire['acres'] / 1000) * 0.01
            
            # Create polygon points
            angles = np.linspace(0, 2*np.pi, 32)  # More points for larger fires
            polygon_points = []
            for angle in angles:
                lat = fire['center_lat'] + radius_deg * np.cos(angle)
                lon = fire['center_lon'] + radius_deg * np.sin(angle)
                polygon_points.append([lon, lat])
            
            # Close polygon
            polygon_points.append(polygon_points[0])
            
            # Create polygon
            polygon = Polygon(polygon_points)
            
            fire_perimeters.append({
                'geometry': polygon,
                'fire_name': fire['name'],
                'year': fire['year'],
                'start_date': fire['start_date'],
                'end_date': fire['end_date'],
                'acres': fire['acres'],
                'county': fire['county'],
                'center_lat': fire['center_lat'],
                'center_lon': fire['center_lon'],
                'duration_days': (datetime.strptime(fire['end_date'], '%Y-%m-%d') - 
                                datetime.strptime(fire['start_date'], '%Y-%m-%d')).days
            })
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(fire_perimeters, crs='EPSG:4326')
    
    # Filter to Tri-County area
    bbox_polygon = Polygon([
        [min_lon, min_lat], [max_lon, min_lat], 
        [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]
    ])
    
    gdf = gdf[gdf.geometry.intersects(bbox_polygon)]
    
    # Simplify geometries to reduce file size
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001)
    
    # Cache the data
    gdf.to_file(cache_file, driver='GeoJSON')
    
    print(f"  Generated {len(gdf)} fire perimeters")
    
    return gdf

def filter_frap_by_year(gdf: gpd.GeoDataFrame, start_year: int, end_year: int) -> gpd.GeoDataFrame:
    """Filter FRAP data by year range"""
    mask = (gdf['year'] >= start_year) & (gdf['year'] <= end_year)
    return gdf[mask].copy()

def filter_frap_by_county(gdf: gpd.GeoDataFrame, counties: List[str]) -> gpd.GeoDataFrame:
    """Filter FRAP data by county"""
    mask = gdf['county'].isin(counties)
    return gdf[mask].copy()

def get_frap_summary_stats(gdf: gpd.GeoDataFrame) -> Dict:
    """Get summary statistics for FRAP data"""
    if gdf.empty:
        return {}
    
    summary = {
        'total_fires': len(gdf),
        'year_range': {
            'start': int(gdf['year'].min()),
            'end': int(gdf['year'].max())
        },
        'total_acres_burned': int(gdf['acres'].sum()),
        'county_distribution': gdf['county'].value_counts().to_dict(),
        'year_distribution': gdf['year'].value_counts().sort_index().to_dict(),
        'avg_fire_size_acres': round(gdf['acres'].mean()),
        'max_fire_size_acres': int(gdf['acres'].max()),
        'avg_duration_days': round(gdf['duration_days'].mean(), 1)
    }
    
    return summary

def simplify_frap_geometries(gdf: gpd.GeoDataFrame, tolerance: float = 0.001) -> gpd.GeoDataFrame:
    """Simplify FRAP geometries to reduce file size"""
    gdf_simplified = gdf.copy()
    gdf_simplified['geometry'] = gdf_simplified['geometry'].simplify(tolerance=tolerance)
    return gdf_simplified

def merge_overlapping_fires(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge overlapping fire perimeters"""
    # Group by year and merge overlapping geometries
    merged_fires = []
    
    for year in gdf['year'].unique():
        year_fires = gdf[gdf['year'] == year].copy()
        
        if len(year_fires) > 1:
            # Merge overlapping geometries
            merged_geom = unary_union(year_fires.geometry)
            
            # Create merged fire record
            merged_fire = {
                'geometry': merged_geom,
                'fire_name': f"Merged {year}",
                'year': year,
                'start_date': year_fires['start_date'].min(),
                'end_date': year_fires['end_date'].max(),
                'acres': int(year_fires['acres'].sum()),
                'county': 'Multiple',
                'center_lat': year_fires['center_lat'].mean(),
                'center_lon': year_fires['center_lon'].mean(),
                'duration_days': int(year_fires['duration_days'].max())
            }
            
            merged_fires.append(merged_fire)
        else:
            merged_fires.append(year_fires.iloc[0].to_dict())
    
    return gpd.GeoDataFrame(merged_fires, crs='EPSG:4326')

def export_frap_to_geojson(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """Export FRAP data to GeoJSON format"""
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to GeoJSON
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"FRAP data exported to {output_path}")

def validate_frap_data(gdf: gpd.GeoDataFrame) -> bool:
    """Validate FRAP data quality"""
    if gdf.empty:
        return False
    
    # Check for required columns
    required_cols = ['geometry', 'fire_name', 'year', 'acres', 'county']
    if not all(col in gdf.columns for col in required_cols):
        return False
    
    # Check for valid geometries
    if not gdf.geometry.is_valid.all():
        return False
    
    # Check for reasonable value ranges
    if not gdf['year'].between(1900, 2030).all():
        return False
    
    if not gdf['acres'].between(1, 1000000).all():
        return False
    
    return True
