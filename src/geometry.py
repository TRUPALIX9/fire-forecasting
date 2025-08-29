import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import geopandas as gpd

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def haversine_distance_vectorized(lats1: np.ndarray, lons1: np.ndarray, 
                                 lat2: float, lon2: float) -> np.ndarray:
    """Vectorized haversine distance calculation"""
    # Convert to radians
    lats1_rad = np.radians(lats1)
    lons1_rad = np.radians(lons1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lats1_rad
    dlon = lon2_rad - lons1_rad
    a = np.sin(dlat/2)**2 + np.cos(lats1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def find_nearest_station(site_lat: float, site_lon: float, 
                        station_lats: np.ndarray, station_lons: np.ndarray,
                        station_ids: List[str]) -> Tuple[str, float]:
    """Find nearest weather station to a site"""
    distances = haversine_distance_vectorized(station_lats, station_lons, site_lat, site_lon)
    min_idx = np.argmin(distances)
    return station_ids[min_idx], distances[min_idx]

def create_buffer_around_point(lat: float, lon: float, radius_km: float) -> Polygon:
    """Create a circular buffer around a point"""
    point = Point(lon, lat)
    # Convert km to degrees (approximate)
    radius_deg = radius_km / 111.0
    return point.buffer(radius_deg)

def point_in_buffer(point_lat: float, point_lon: float, 
                   buffer_lat: float, buffer_lon: float, 
                   buffer_radius_km: float) -> bool:
    """Check if a point is within a buffer around another point"""
    distance = haversine_distance(point_lat, point_lon, buffer_lat, buffer_lon)
    return distance <= buffer_radius_km

def create_tri_county_bbox() -> Tuple[float, float, float, float]:
    """Create bounding box for Tri-County area (Ventura, Santa Barbara, Los Angeles)"""
    # Approximate bounding box coordinates
    # (min_lat, min_lon, max_lat, max_lon)
    return (33.5, -121.0, 35.0, -117.5)

def filter_points_in_bbox(points_df: pd.DataFrame, 
                         bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """Filter points to only those within bounding box"""
    min_lat, min_lon, max_lat, max_lon = bbox
    
    mask = (
        (points_df['lat'] >= min_lat) & (points_df['lat'] <= max_lat) &
        (points_df['lon'] >= min_lon) & (points_df['lon'] <= max_lon)
    )
    
    return points_df[mask].copy()

def simplify_geometries(gdf: gpd.GeoDataFrame, tolerance: float = 0.001) -> gpd.GeoDataFrame:
    """Simplify geometries to reduce file size"""
    gdf_simplified = gdf.copy()
    gdf_simplified['geometry'] = gdf_simplified['geometry'].simplify(tolerance)
    return gdf_simplified

def merge_overlapping_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge overlapping polygons to reduce redundancy"""
    # Dissolve overlapping polygons
    dissolved = gdf.dissolve(by=None)
    
    # If multiple polygons remain, merge them
    if len(dissolved) > 1:
        merged_geom = unary_union(dissolved.geometry)
        dissolved = gpd.GeoDataFrame(geometry=[merged_geom])
    
    return dissolved

def calculate_centroid_distance(lat1: float, lon1: float, 
                              lat2: float, lon2: float) -> float:
    """Calculate distance between two centroids"""
    return haversine_distance(lat1, lon1, lat2, lon2)

def create_neighbor_weights(distances: np.ndarray, k: int = 1) -> np.ndarray:
    """Create inverse distance weights for k nearest neighbors"""
    if k == 0:
        return np.zeros_like(distances)
    
    # Sort distances and get k nearest
    sorted_indices = np.argsort(distances)
    k_nearest_indices = sorted_indices[:k]
    
    # Create weights (inverse distance)
    weights = np.zeros_like(distances)
    for idx in k_nearest_indices:
        if distances[idx] > 0:
            weights[idx] = 1.0 / distances[idx]
    
    # Normalize weights
    if weights.sum() > 0:
        weights = weights / weights.sum()
    
    return weights
