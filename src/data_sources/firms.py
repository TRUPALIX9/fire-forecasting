import os
import glob
from typing import Tuple, Optional
from datetime import datetime
import pandas as pd

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

def load_firms_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_confidence: str = "n",
    firms_data_dir: str = "data/firms"
) -> pd.DataFrame:
    """
    Load real NASA FIRMS data from DL_FIRE_* subdirectories.
    Returns a standardized dataframe with a 'date' column.
    """
    try:
        df = load_real_firms_data(
            start_date=start_date or "2019-01-01",
            end_date=end_date or "2024-12-31",
            min_confidence=min_confidence,
            firms_data_dir=firms_data_dir
        )
        print(f"[FIRMS] Successfully loaded {len(df):,} real FIRMS records")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load FIRMS data: {e}. Please ensure DL_FIRE_* CSV files exist in {firms_data_dir}/")
