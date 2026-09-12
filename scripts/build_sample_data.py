#!/usr/bin/env python3
"""Build the small sample dataset bundled with the dashboard prototype.

Reads data/trihourly_weather.csv (the committed trihourly weather CSV) and
writes frontend/lib/sample-forecast.json with:

- weather: the 3-hourly rows for 15-17 Aug 2023, copied from the CSV
  (the first row is kept where the CSV repeats a timestamp);
- sites: eight fictional monitoring sites in the Tri-County area, each with
  an illustrative 72 h series of fire-risk probabilities.

The risk series are hand-authored sample values for the UI, not the output of
a trained model. There is no model service in this repository.

Usage: python3 scripts/build_sample_data.py
"""
import csv
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "trihourly_weather.csv")
OUT_PATH = os.path.join(ROOT, "frontend", "lib", "sample-forecast.json")

WINDOW_START = "2023-08-15 00:00:00"
WINDOW_END = "2023-08-17 21:00:00"
THRESHOLD = 0.5

# Shape of the sample risk curve (24 steps of 3 h), peaking at step 15.
BASE = [0.41, 0.30, 0.22, 0.20, 0.27, 0.43, 0.66, 0.74,
        0.69, 0.48, 0.33, 0.31, 0.38, 0.55, 0.76, 0.82,
        0.72, 0.52, 0.36, 0.34, 0.40, 0.51, 0.68, 0.71]
BASE_PEAK = max(BASE)

# Fictional sites: (id, name, county, lat, lon, peak risk, peak shift in steps)
SITES = [
    ("matilija-canyon", "Matilija Canyon", "Ventura", 34.49, -119.30, 0.82, 0),
    ("gibraltar-road", "Gibraltar Road", "Santa Barbara", 34.47, -119.68, 0.71, 0),
    ("santa-paula-ridge", "Santa Paula Ridge", "Ventura", 34.39, -119.07, 0.64, -1),
    ("castaic-north", "Castaic North", "Los Angeles", 34.53, -118.61, 0.46, 1),
    ("topanga-overlook", "Topanga Overlook", "Los Angeles", 34.10, -118.60, 0.38, 0),
    ("simi-hills", "Simi Hills", "Ventura", 34.25, -118.73, 0.29, -1),
    ("big-tujunga", "Big Tujunga", "Los Angeles", 34.31, -118.20, 0.22, 1),
    ("lake-hughes", "Lake Hughes", "Los Angeles", 34.68, -118.44, 0.14, 0),
]

WEATHER_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
    "soil_moisture_0_to_7cm",
]


def load_weather():
    rows = {}
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            dt = r["datetime"]
            if not (WINDOW_START <= dt <= WINDOW_END) or dt in rows:
                continue
            if dt[14:16] != "00" or int(dt[11:13]) % 3:
                continue
            rows[dt] = r
    out = []
    for dt in sorted(rows):
        r = rows[dt]
        item = {"time": dt.replace(" ", "T")[:16]}
        for k in WEATHER_FIELDS:
            item[k] = round(float(r[k]), 3 if k.startswith("soil") else 1)
        out.append(item)
    return out


def site_series(peak, shift, k):
    n = len(BASE)
    scale = peak / BASE_PEAK
    series = []
    for i in range(n):
        j = min(max(i - shift, 0), n - 1)
        v = BASE[j] * scale + 0.015 * math.sin(1.7 * i + k)
        series.append(v)
    top = max(range(n), key=lambda i: series[i])
    series = [min(max(v, 0.02), peak - 0.01) for v in series]
    series[top] = peak
    return [round(v, 2) for v in series]


def main():
    weather = load_weather()
    if len(weather) != len(BASE):
        raise SystemExit(f"expected {len(BASE)} weather rows, found {len(weather)}")
    sites = []
    for k, (sid, name, county, lat, lon, peak, shift) in enumerate(SITES):
        series = BASE if sid == "matilija-canyon" else site_series(peak, shift, k)
        sites.append({
            "id": sid,
            "name": name,
            "county": county,
            "lat": lat,
            "lon": lon,
            "risk": series,
        })
    data = {
        "description": (
            "Sample data for the Fire Forecasting dashboard prototype. Weather rows "
            "come from data/trihourly_weather.csv; sites and risk values are "
            "fictional, illustrative samples, not model output."
        ),
        "source": "data/trihourly_weather.csv",
        "stepHours": 3,
        "threshold": THRESHOLD,
        "region": {
            "name": "Tri-County area",
            "counties": ["Santa Barbara", "Ventura", "Los Angeles"],
            "bbox": [-119.828, 33.422, -117.274, 34.931],
        },
        "weather": weather,
        "sites": sites,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}: {len(weather)} weather rows, {len(sites)} sites")


if __name__ == "__main__":
    main()
