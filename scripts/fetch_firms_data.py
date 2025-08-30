#!/usr/bin/env python3
import os
from src.data_sources.firms import load_real_firms_data

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

def main():
    os.makedirs("data/firms", exist_ok=True)
    print("Loading real NASA FIRMS data...")
    
    try:
        # Load real FIRMS data from DL_FIRE_* subdirectories
        df = load_real_firms_data(
            start_date=START_DATE,
            end_date=END_DATE,
            min_confidence="n",  # nominal confidence
            firms_data_dir="data/firms"
        )
        
        print(f"Successfully loaded {len(df):,} real FIRMS records")
        
        # Save consolidated data for pipeline use
        consolidated_path = "data/firms/firms_consolidated.csv"
        df.to_csv(consolidated_path, index=False)
        print(f"Wrote consolidated data to {consolidated_path}")
        
        # Also save as WFS format for pipeline compatibility
        wfs_path = "data/firms/firms_wfs.csv"
        df.to_csv(wfs_path, index=False)
        print(f"Wrote WFS-compatible data to {wfs_path}")
        
        # Show data summary
        print(f"\nData Summary:")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Total records: {len(df):,}")
        print(f"Unique dates: {df['date'].nunique()}")
        print(f"Latitude range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        print(f"Longitude range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
        
    except Exception as e:
        print(f"Error loading real FIRMS data: {e}")
        print("Please ensure you have extracted the DL_FIRE_* zip files to data/firms/")
        print("The pipeline now requires real FIRMS data and cannot fall back to WFS.")

if __name__ == "__main__":
    main()
