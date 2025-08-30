import os
import json
from typing import Tuple, Optional
from datetime import datetime
import pandas as pd
import requests
import glob

# VIIRS WFS endpoint (NASA FIRMS). We request GeoJSON via WFS.
# Note: Specific layer names can vary by region; "fires_viirs" is commonly used.
FIRMS_WFS_URL = "https://firms.modaps.eosdis.nasa.gov/wfs/viirs"

def load_real_firms_data(
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    min_confidence: str = "n",  # "n" for nominal, "l" for low, "h" for high
    firms_data_dir: str = "data/firms"
) -> pd.DataFrame:
    """
    Load and consolidate real NASA FIRMS data from DL_FIRE_* subdirectories.
    Returns a standardized DataFrame with required columns for the pipeline.
    """
    # Find all FIRMS CSV files in subdirectories
    pattern = os.path.join(firms_data_dir, "DL_FIRE_*", "*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No FIRMS CSV files found in {firms_data_dir}/DL_FIRE_*")
    
    print(f"[FIRMS] Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    all_data = []
    for csv_file in csv_files:
        try:
            # Read CSV with proper date parsing
            df = pd.read_csv(csv_file, parse_dates=["acq_date"], low_memory=False)
            
            # Filter by confidence if specified
            if min_confidence != "n":
                confidence_map = {"l": "l", "n": "n", "h": "h"}
                if min_confidence in confidence_map:
                    df = df[df["confidence"] == min_confidence]
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df["acq_date"] >= start_dt) & (df["acq_date"] <= end_dt)]
            
            # Add source identifier
            df["source"] = "VIIRS"
            
            # Standardize column names and add required columns
            if "date" not in df.columns:
                df["date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
            
            # Ensure required columns exist
            required_cols = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "date", "source"]
            for col in required_cols:
                if col not in df.columns:
                    if col == "acq_time" and "acq_time" not in df.columns:
                        df["acq_time"] = "0000"  # Default time if missing
                    elif col == "confidence" and "confidence" not in df.columns:
                        df["confidence"] = 50  # Default confidence if missing
            
            all_data.append(df)
            print(f"[FIRMS] Loaded {len(df):,} records from {os.path.basename(csv_file)}")
            
        except Exception as e:
            print(f"[FIRMS] Error reading {csv_file}: {e}")
            continue
    
    if not all_data:
        raise RuntimeError("No valid FIRMS data could be loaded from any CSV files")
    
    # Concatenate all data
    consolidated_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates and sort by date
    consolidated_df = consolidated_df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    consolidated_df = consolidated_df.sort_values("date")
    
    # Drop rows with missing critical data
    consolidated_df = consolidated_df.dropna(subset=["latitude", "longitude", "date"])
    
    print(f"[FIRMS] Consolidated {len(consolidated_df):,} total FIRMS records")
    return consolidated_df

def fetch_firms_wfs(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    min_confidence: int = 0,
    layer: str = "fires_viirs",
    timeout: int = 60
) -> pd.DataFrame:
    """
    Fetch FIRMS detections from WFS as GeoJSON for the given bbox and date range.
    Returns a DataFrame with at least: latitude, longitude, acq_date, acq_time, confidence, date.
    """
    map_key = os.getenv("FIRMS_MAP_KEY")
    if not map_key:
        raise RuntimeError("FIRMS_MAP_KEY is not set; export it in your environment or .env file")
    
    params = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "1.1.0",
        "typename": layer,
        "outputFormat": "application/json",
        # bbox order: minx, miny, maxx, maxy (lon/lat)
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        # ISO date range
        "time": f"{start_date}/{end_date}",
        "map_key": map_key,
    }
    
    r = requests.get(FIRMS_WFS_URL, params=params, timeout=timeout)
    r.raise_for_status()
    gj = r.json()
    feats = gj.get("features", [])
    
    rows = []
    for f in feats:
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates", [None, None])
        lon = props.get("longitude", coords[0] if len(coords) > 0 else None)
        lat = props.get("latitude", coords[1] if len(coords) > 1 else None)
        acq_date = props.get("acq_date")
        acq_time = props.get("acq_time")
        conf = props.get("confidence", 0) or 0
        if conf is None:
            conf = 0
        
        if lat is None or lon is None:
            continue
        
        if conf >= min_confidence:
            rows.append({
                "latitude": lat,
                "longitude": lon,
                "acq_date": acq_date,
                "acq_time": acq_time,
                "confidence": conf,
                "source": "VIIRS"
            })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"])
    
    return df

def load_firms_data(
    prefer_real: bool = True,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_confidence: int = 0,
    layer: str = "fires_viirs",
    firms_data_dir: str = "data/firms"
) -> pd.DataFrame:
    """
    Try real FIRMS data first (when prefer_real=True). If it fails or returns empty,
    fall back to WFS fetch. Returns a standardized dataframe with a 'date' column.
    """
    if prefer_real:
        try:
            df = load_real_firms_data(
                start_date=start_date or "2019-01-01",
                end_date=end_date or "2024-12-31",
                min_confidence="n",  # Use nominal confidence for real data
                firms_data_dir=firms_data_dir
            )
            if not df.empty:
                print(f"[FIRMS] Successfully loaded {len(df):,} real FIRMS records")
                return df
        except Exception as e:
            print(f"[FIRMS] Real data load failed: {e}. Falling back to WFS.")
    
    # Fallback to WFS if real data failed or prefer_real=False
    if bbox and start_date and end_date:
        try:
            df = fetch_firms_wfs(bbox, start_date, end_date, min_confidence=min_confidence, layer=layer)
            if not df.empty:
                print(f"[FIRMS] Successfully fetched {len(df):,} records via WFS")
                return df
        except Exception as e:
            print(f"[FIRMS] WFS fetch failed: {e}")
    
    # If all else fails, raise an error
    raise RuntimeError("Could not load FIRMS data from any source (real files or WFS)")
