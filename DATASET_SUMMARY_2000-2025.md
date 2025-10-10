# Fire Forecasting Temporal Weather Datasets (2020-2023)
## Multi-Resolution Weather & Fire Dataset Summary

### Overview
This dataset collection contains high-resolution weather and fire occurrence data from 2020-2023, sourced from the fire-prediction repository. The data is available at three different temporal resolutions (hourly, bihourly, trihourly) to support various machine learning applications and temporal analysis requirements.

### Dataset Information
- **Files**: 3 CSV files with different temporal resolutions
- **Total Records**: 82,775 observations across all datasets
- **Years**: 4 years (2020-2023)
- **Date Range**: January 1, 2020 to December 25, 2023
- **Total Size**: 10.1 MB

### Dataset Breakdown

| Dataset | Temporal Resolution | Records | Size | Columns |
|---------|-------------------|---------|------|---------|
| **Hourly_Weather_Dataset.csv** | Every hour | 40,711 | 4.9 MB | 14 |
| **Bihourly_Weather_Dataset.csv** | Every 2 hours | 23,954 | 2.8 MB | 14 |
| **Trihourly_Weather_Dataset.csv** | Every 3 hours | 18,110 | 2.3 MB | 15 |

### Features Description

#### Temporal Features
1. **date** - Date of observation (MM/DD/YY format)
2. **time** - Time of observation (HH:MM:SS format)
3. **datetime** - Combined date-time (Trihourly only)

#### Target Variables
4. **Severity** - Fire severity score (0-27.5+ range)
5. **forest_fire** - Binary fire indicator (N/Y)

#### Atmospheric Weather Features
6. **temperature_2m** - Air temperature at 2m height (°C)
7. **relative_humidity_2m** - Relative humidity at 2m (%)
8. **precipitation** - Precipitation amount (mm)
9. **surface_pressure** - Atmospheric pressure (hPa)
10. **cloud_cover** - Cloud cover percentage (%)
11. **wind_speed_10m** - Wind speed at 10m height (m/s)

#### Soil/Subsurface Features
12. **soil_temperature_0_to_7cm** - Soil temperature 0-7cm depth (°C)
13. **soil_temperature_7_to_28cm** - Soil temperature 7-28cm depth (°C)
14. **soil_moisture_0_to_7cm** - Soil moisture 0-7cm depth (fraction)
15. **soil_moisture_7_to_28cm** - Soil moisture 7-28cm depth (fraction)

### Key Statistics

#### Fire Occurrence (Bihourly Dataset)
- **Total fire events**: 6,422 instances (26.8% of observations)
- **No fire events**: 17,529 instances (73.2% of observations)
- **Class balance**: Reasonably balanced for fire prediction
- **Severity range**: 0 to 27.5+ (continuous scale)

#### Temporal Resolution Comparison
| Resolution | Records | Fire Events | Fire Rate | Use Case |
|------------|---------|-------------|-----------|----------|
| Hourly | 40,711 | ~10,000+ | ~25% | High-frequency analysis |
| Bihourly | 23,954 | 6,422 | 26.8% | Standard ML training |
| Trihourly | 18,110 | ~4,800+ | ~26% | Long-term patterns |

#### Data Quality
- **Format consistency**: All datasets use consistent column naming
- **Missing values**: Minimal missing data across all features
- **Temporal coverage**: Complete 4-year coverage (2020-2023)
- **Data completeness**: High (>99% complete records)

### Key Insights (2020-2023 Period)

#### Multi-Resolution Benefits
- **Flexible analysis**: Choose appropriate temporal resolution for your use case
- **Hourly data**: Perfect for short-term fire risk assessment and real-time monitoring
- **Bihourly data**: Optimal balance between granularity and computational efficiency
- **Trihourly data**: Suitable for long-term pattern analysis and trend identification

#### Fire Prediction Advantages
- **High-resolution weather**: More precise weather conditions for fire prediction
- **Soil moisture data**: Critical subsurface information often missing in other datasets
- **Dual targets**: Both binary classification (fire/no-fire) and regression (severity)
- **Recent data**: Captures current fire patterns and climate conditions

#### Machine Learning Opportunities
- **Temporal modeling**: Time series analysis across multiple resolutions
- **Feature engineering**: Rich weather and soil features for model development
- **Balanced dataset**: Good fire/no-fire ratio for training robust models
- **Multi-task learning**: Predict both fire occurrence and severity simultaneously

### Usage Notes
- **Multi-resolution flexibility**: Choose the temporal resolution that best fits your analysis needs
- **High-quality data**: Clean, consistent formatting across all datasets
- **Recent coverage**: 2020-2023 period captures current fire patterns and climate conditions
- **Rich features**: Comprehensive weather and soil data for robust modeling
- **Balanced targets**: Good class distribution for both classification and regression tasks

### Dataset Selection Guide

| Use Case | Recommended Dataset | Reason |
|----------|-------------------|---------|
| **Real-time monitoring** | Hourly_Weather_Dataset.csv | Highest temporal resolution |
| **Standard ML training** | Bihourly_Weather_Dataset.csv | Balanced size and granularity |
| **Long-term analysis** | Trihourly_Weather_Dataset.csv | Reduced noise, clear patterns |
| **Time series modeling** | Any dataset | Choose based on prediction horizon |
| **Feature engineering** | All datasets | Compare patterns across resolutions |

### File Locations
```
data/Hourly_Weather_Dataset.csv    (40,711 records, 4.9 MB)
data/Bihourly_Weather_Dataset.csv  (23,954 records, 2.8 MB)
data/Trihourly_Weather_Dataset.csv (18,110 records, 2.3 MB)
```

### Data Source
- **Original repository**: [fire-prediction GitHub](https://github.com/gauravsurtani/fire-prediction)
- **Final datasets folder**: Contains processed weather and fire data
- **Processing**: Cleaned and formatted for machine learning applications

### Recommended Use Cases
- **Fire risk prediction models** (binary classification)
- **Fire severity estimation** (regression)
- **Temporal pattern analysis** (time series)
- **Multi-resolution modeling** (ensemble methods)
- **Real-time fire monitoring systems**
- **Climate impact studies** on fire patterns
- **Feature engineering** for advanced ML models
