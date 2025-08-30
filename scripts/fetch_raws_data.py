#!/usr/bin/env python3
"""
Fetch RAWS (Remote Automated Weather Station) data for monitoring sites.
This script fetches weather data from the RAWS network for the Tri-County area.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import requests
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_sources.raws import SITE_COORDINATES, fetch_raws_data_daily

def main():
    """Main function to fetch RAWS data for all sites"""
    print("🌤️  Fetching RAWS weather data for monitoring sites...")
    
    # Create data directory
    data_dir = Path("data/raws")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define date range (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📍 Sites: {len(SITE_COORDINATES)} monitoring locations")
    
    all_data = []
    
    for site_name, coords in SITE_COORDINATES.items():
        print(f"  📡 Fetching data for {site_name}...")
        
        try:
            # Fetch data for this site
            site_data = fetch_raws_data_daily(
                site_name=site_name,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if not site_data.empty:
                # Add site identifier
                site_data['site'] = site_name
                site_data['lat'] = coords['lat']
                site_data['lon'] = coords['lon']
                
                all_data.append(site_data)
                print(f"    ✅ {len(site_data)} records for {site_name}")
            else:
                print(f"    ⚠️  No data available for {site_name}")
                
        except Exception as e:
            print(f"    ❌ Error fetching data for {site_name}: {e}")
    
    if all_data:
        # Combine all site data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by date and site
        combined_data = combined_data.sort_values(['date', 'site'])
        
        # Save combined data
        output_path = data_dir / "raws_combined_data.csv"
        combined_data.to_csv(output_path, index=False)
        print(f"\n💾 Saved combined RAWS data: {output_path}")
        print(f"📊 Total records: {len(combined_data):,}")
        print(f"🏘️  Sites with data: {combined_data['site'].nunique()}")
        
        # Save individual site files
        for site_name in combined_data['site'].unique():
            site_data = combined_data[combined_data['site'] == site_name]
            site_file = data_dir / f"{site_name.replace(' ', '_')}_20170101_20241231.csv"
            site_data.to_csv(site_file, index=False)
            print(f"  📁 {site_name}: {site_file}")
        
        # Save summary statistics
        summary = {
            'total_records': len(combined_data),
            'sites_count': combined_data['site'].nunique(),
            'date_range': {
                'start': combined_data['date'].min().strftime('%Y-%m-%d'),
                'end': combined_data['date'].max().strftime('%Y-%d-%d')
            },
            'sites': list(combined_data['site'].unique()),
            'columns': list(combined_data.columns),
            'fetch_timestamp': datetime.now().isoformat()
        }
        
        summary_path = data_dir / "raws_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📋 Summary saved: {summary_path}")
        
    else:
        print("\n❌ No RAWS data was successfully fetched")
        print("💡 Check your internet connection and RAWS API access")
    
    print("\n✨ RAWS data fetch complete!")

if __name__ == "__main__":
    main()
