#!/usr/bin/env python3
"""
FRAP Data Fetcher
Fetches real California Historical Fire Perimeters from ArcGIS REST API
Filters by Tri-County bbox and year range, simplifies geometries for web display
"""

import os
import json
import requests
import geopandas as gpd
import yaml

CONFIG_PATH = "config.yaml"
ARCGIS_URL = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "California_Historic_Fire_Perimeters/FeatureServer/0/query"
)
RAW_DIR = "data/frap"
ART_DIR = "artifacts/geo"
RAW_BASENAME = "historical_perimeters_tri_county_{ymin}_{ymax}.geojson"
OUT_SIMPLIFIED = "frap_fire_perimeters.geojson"

DEFAULTS = {
    "bbox": [-119.828, 33.422, -117.274, 34.931],
    "year_min": 2019,
    "year_max": 2024,
    "simplify_tolerance": 0.0005,
}

def load_frap_cfg():
    """Load FRAP configuration from config.yaml with fallback to defaults"""
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    frap = (cfg or {}).get("frap", {}) or {}
    return {
        "bbox": frap.get("bbox", DEFAULTS["bbox"]),
        "year_min": int(frap.get("year_min", DEFAULTS["year_min"])),
        "year_max": int(frap.get("year_max", DEFAULTS["year_max"])),
        "simplify_tolerance": float(frap.get("simplify_tolerance", DEFAULTS["simplify_tolerance"])),
    }

def fetch_geojson(bbox, year_min, year_max):
    """Fetch GeoJSON from ArcGIS REST API with bbox and year filters"""
    where = f"YEAR_>={year_min} AND YEAR_<={year_max}"
    params = {
        "where": where,
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": 4326,
        "f": "geojson",
        # Note: MaxRecordCount is server-limited; envelope+year cut keeps results small.
    }
    
    print(f"[FRAP] Fetching data from ArcGIS REST API...")
    print(f"[FRAP] Bbox: {bbox}")
    print(f"[FRAP] Year range: {year_min}-{year_max}")
    
    r = requests.get(ARCGIS_URL, params=params, timeout=180)
    r.raise_for_status()
    return r.json()

def main():
    """Main function to fetch, process, and save FRAP data"""
    try:
        # Load configuration
        cfg = load_frap_cfg()
        bbox = cfg["bbox"]
        ymin, ymax = cfg["year_min"], cfg["year_max"]
        tol = cfg["simplify_tolerance"]

        print(f"[FRAP] Configuration loaded:")
        print(f"[FRAP]   Bbox: {bbox}")
        print(f"[FRAP]   Year range: {ymin}-{ymax}")
        print(f"[FRAP]   Simplify tolerance: {tol}")

        # Create directories
        os.makedirs(RAW_DIR, exist_ok=True)
        os.makedirs(ART_DIR, exist_ok=True)

        # Fetch data from ArcGIS
        gj = fetch_geojson(bbox, ymin, ymax)
        feats = gj.get("features", [])
        
        if not feats:
            print(f"[FRAP] Warning: No features returned from API")
            print(f"[FRAP] This might indicate an issue with the query parameters")

        # Save raw subset
        raw_path = os.path.join(RAW_DIR, RAW_BASENAME.format(ymin=ymin, ymax=ymax))
        with open(raw_path, "w") as f:
            json.dump(gj, f)
        print(f"[FRAP] Wrote raw subset → {raw_path}  (features: {len(feats)})")

        # Simplify for web map
        if feats:
            print(f"[FRAP] Processing {len(feats)} features...")
            gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
            
            if "geometry" in gdf.columns and not gdf.empty:
                print(f"[FRAP] Simplifying geometries with tolerance {tol}...")
                gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
                
                # Add some basic statistics
                if "YEAR_" in gdf.columns:
                    years = gdf["YEAR_"].dropna()
                    if not years.empty:
                        print(f"[FRAP] Year range in data: {years.min()}-{years.max()}")
                
                if "ACRES" in gdf.columns:
                    acres = gdf["ACRES"].dropna()
                    if not acres.empty:
                        print(f"[FRAP] Fire size range: {acres.min():.1f} - {acres.max():.1f} acres")
                
            out_path = os.path.join(ART_DIR, OUT_SIMPLIFIED)
            gdf.to_file(out_path, driver="GeoJSON")
            print(f"[FRAP] Wrote simplified perimeters → {out_path}  (rows: {len(gdf)})")
            
            # Show file sizes
            raw_size = os.path.getsize(raw_path) / 1024  # KB
            simplified_size = os.path.getsize(out_path) / 1024  # KB
            compression = ((raw_size - simplified_size) / raw_size) * 100 if raw_size > 0 else 0
            print(f"[FRAP] File sizes: Raw: {raw_size:.1f}KB, Simplified: {simplified_size:.1f}KB")
            print(f"[FRAP] Compression: {compression:.1f}%")
            
        else:
            # Create an empty FeatureCollection so the API returns valid GeoJSON
            out_path = os.path.join(ART_DIR, OUT_SIMPLIFIED)
            empty = {"type": "FeatureCollection", "features": []}
            with open(out_path, "w") as f:
                json.dump(empty, f)
            print(f"[FRAP] No features; wrote empty GeoJSON → {out_path}")

        print(f"[FRAP] FRAP data fetch completed successfully!")
        
    except requests.exceptions.RequestException as e:
        print(f"[FRAP] Error fetching data from ArcGIS API: {e}")
        print(f"[FRAP] Please check your internet connection and try again")
        raise
    except Exception as e:
        print(f"[FRAP] Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
