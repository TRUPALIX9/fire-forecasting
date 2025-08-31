# California Weather & Fire Dataset (2000-2025)
## Consolidated Dataset Summary

### Overview
This dataset contains daily weather and fire occurrence data for California from 2000 to 2025, consolidated from the original 1984-2025 dataset. This period focuses on the most recent and relevant fire patterns for modern machine learning applications.

### Dataset Information
- **File**: `CA_Weather_Fire_Dataset_2000-2025.csv`
- **Records**: 9,144 daily observations
- **Years**: 26 years (2000-2025)
- **Date Range**: January 1, 2000 to January 12, 2025
- **File Size**: 924,107 bytes (902.4 KB)

### Data Reduction Summary
- **Records removed**: 5,844 (39.0% reduction from original)
- **Years removed**: 16 years (1984-1999)
- **Fire start days removed**: 1,512
- **Size reduction**: 39.1%

### Columns Description
1. **DATE** - Date of observation (YYYY-MM-DD)
2. **PRECIPITATION** - Daily precipitation in inches
3. **MAX_TEMP** - Maximum temperature in Fahrenheit
4. **MIN_TEMP** - Minimum temperature in Fahrenheit
5. **AVG_WIND_SPEED** - Average wind speed in mph
6. **FIRE_START_DAY** - Boolean indicating if a fire started on this day
7. **YEAR** - Year of observation
8. **TEMP_RANGE** - Temperature range (MAX_TEMP - MIN_TEMP)
9. **WIND_TEMP_RATIO** - Ratio of wind speed to temperature range
10. **MONTH** - Month number (1-12)
11. **SEASON** - Season (Winter, Spring, Summer, Fall)
12. **LAGGED_PRECIPITATION** - Previous day's precipitation
13. **LAGGED_AVG_WIND_SPEED** - Previous day's average wind speed
14. **DAY_OF_YEAR** - Day of the year (1-366)

### Key Statistics

#### Fire Occurrence
- **Total fire start days**: 3,459 (37.8% of all days)
- **Average fire start days per year**: 133.0
- **Most fire-prone years**:
  - 2017: 197 fire start days
  - 2020: 188 fire start days
  - 2018: 186 fire start days
  - 2021: 173 fire start days
  - 2012: 172 fire start days

#### Seasonal Analysis
| Season | Fire Start Days | Avg Precipitation | Avg Max Temp | Avg Wind Speed |
|--------|----------------|-------------------|--------------|----------------|
| Summer | 1,760          | 0.00"            | 74.3°F       | 7.8 mph        |
| Fall   | 888            | 0.01"            | 73.9°F       | 6.6 mph        |
| Spring | 605            | 0.03"            | 67.7°F       | 8.1 mph        |
| Winter | 206            | 0.08"            | 65.9°F       | 6.3 mph        |

#### Temperature Statistics
- **Average max temperature**: 70.4°F
- **Average min temperature**: 56.7°F
- **Average temperature range**: 13.8°F
- **Maximum temperature recorded**: 105.0°F
- **Minimum temperature recorded**: 35.0°F

#### Data Quality
- **Missing wind speed values**: 9 records
- **Missing wind-temperature ratio**: 9 records
- **Data completeness**: 99.9%

### Key Insights (2000-2025 Period)

#### Modern Fire Patterns
- **Higher fire frequency**: 37.8% of days had fires (vs 33.1% in 1984-2025)
- **Recent peak years**: 2017-2021 saw the highest fire activity
- **Climate change impact**: More extreme fire seasons in recent years

#### Seasonal Trends
- **Summer dominance**: 51% of all fires occur in summer
- **Extended fire season**: Significant fall fire activity (26% of fires)
- **Winter protection**: Only 6% of fires in winter months

#### Weather Patterns
- **Drier conditions**: Lower average precipitation (0.03" vs 0.04" in full dataset)
- **Higher temperatures**: Slightly elevated average temperatures
- **Wind patterns**: Consistent wind speeds across seasons

### Usage Notes
- **Modern relevance**: Focuses on current fire patterns and climate conditions
- **ML optimization**: Reduced dataset size improves training efficiency
- **Recent trends**: Captures the impact of climate change on fire patterns
- **Data quality**: High completeness with minimal missing values
- **Temporal features**: All lagged and seasonal features preserved

### Comparison with Other Periods

| Metric | 1984-2025 | 1990-2025 | 2000-2025 |
|--------|-----------|-----------|-----------|
| Records | 14,988 | 12,796 | 9,144 |
| Years | 42 | 36 | 26 |
| Fire Days | 4,971 | 4,377 | 3,459 |
| Fire Rate | 33.1% | 34.2% | 37.8% |
| File Size | 1.48 MB | 1.25 MB | 0.90 MB |

### File Location
```
data/CA_Weather_Fire_Dataset_2000-2025.csv
```

### Related Files
- Original dataset: `data/CA_Weather_Fire_Dataset_1984-2025.csv`
- 1990-2025 dataset: `data/CA_Weather_Fire_Dataset_1990-2025.csv`
- 1990 summary: `data/DATASET_SUMMARY_1990-2025.md`
- This summary: `data/DATASET_SUMMARY_2000-2025.md`

### Recommended Use Cases
- **Modern fire prediction models**
- **Climate change impact studies**
- **Recent fire pattern analysis**
- **Efficient ML training with reduced data size**
- **Real-time fire risk assessment systems**
