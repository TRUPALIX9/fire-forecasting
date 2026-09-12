# Trihourly weather dataset

The repository ships one weather file: [`data/trihourly_weather.csv`](../data/trihourly_weather.csv).
The dashboard does not read it at runtime. `scripts/build_sample_data.py` copies a small
window of it (15-17 Aug 2023) into `frontend/lib/sample-forecast.json`, which the UI bundles.

Earlier versions of this summary described three files (hourly, bihourly and trihourly,
82,775 rows in total). Only the trihourly file is in the repository; the other two are not.

## Source

Processed weather and fire data from the
[fire-prediction repository](https://github.com/gauravsurtani/fire-prediction)
(its final datasets folder), cleaned and formatted for machine learning. The file was
committed as `CLEANED_Trihourly_Weather_Dataset (1).csv` and renamed to
`data/trihourly_weather.csv`.

## Facts (checked against the file)

| | |
|---|---|
| Rows | 18,069 |
| Columns | 15 |
| First / last timestamp | 2020-01-01 06:00 / 2024-01-01 06:00 |
| Rows per year | 2020: 8,599 · 2021: 3,096 · 2022: 3,202 · 2023: 3,169 · 2024: 3 |
| `forest_fire = Y` | 6,380 rows (35.3%) |
| `Severity` range | 0 to 42.6 |

## Columns

| Column | Meaning |
|---|---|
| `datetime`, `date`, `time` | Timestamp, date and time of the observation |
| `Severity` | Fire severity score |
| `temperature_2m` | Air temperature at 2 m (°C) |
| `relative_humidity_2m` | Relative humidity at 2 m (%) |
| `precipitation` | Precipitation (mm) |
| `surface_pressure` | Surface pressure (hPa) |
| `cloud_cover` | Cloud cover (%) |
| `wind_speed_10m` | Wind speed at 10 m |
| `soil_temperature_0_to_7cm`, `soil_temperature_7_to_28cm` | Soil temperature (°C) |
| `soil_moisture_0_to_7cm`, `soil_moisture_7_to_28cm` | Soil moisture (fraction) |
| `forest_fire` | Fire indicator (`N` / `Y`) |

## Known data-quality caveats

- There is no location column, and timestamps repeat: 12,408 distinct timestamps, and
  5,661 rows repeat a timestamp that already appears (for example `2020-08-20 09:00:00`
  appears 849 times). Some rows marked `Y` carry values that differ sharply from the
  row with the same timestamp (at `2023-08-15 21:00:00`, `surface_pressure` is 918.98
  in the `Y` row and 934.72 in the `N` row).
- Line 6822 (`2020-08-26 09:00:00`) is empty apart from the timestamp and a 0.0 value.
- 57 rows have a `time` value that disagrees with the time in `datetime`
  (the first row is `2020-01-01 06:00:00` with `time` `07:00:00`).
- Coverage is uneven: 2020 has almost three times as many rows as later years.

`build_sample_data.py` works around the duplicates by keeping the first row for each
3-hourly timestamp in its window.
