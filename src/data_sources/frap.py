import os
import geopandas as gpd

def build_frap_geojson(
    shp_path: str = "data/frap/FirePerimeters.shp",
    out_path: str = "artifacts/geo/frap_fire_perimeters.geojson",
    simplify: float = 0.0005
) -> str:
    """
    Convert FRAP shapefile to GeoJSON (WGS84) with optional geometry simplification.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    if simplify and simplify > 0:
        gdf["geometry"] = gdf["geometry"].simplify(simplify, preserve_topology=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path
