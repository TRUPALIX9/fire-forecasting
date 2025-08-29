import numpy as np
import pandas as pd
from typing import Optional

def calculate_dew_point(temp_c: float, rh: float) -> float:
    """Calculate dew point temperature from temperature and relative humidity"""
    # Magnus formula
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

def calculate_vapor_pressure_deficit(temp_c: float, rh: float) -> float:
    """Calculate vapor pressure deficit"""
    # Saturation vapor pressure (Tetens formula)
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    
    # Actual vapor pressure
    ea = es * (rh / 100.0)
    
    # Vapor pressure deficit
    vpd = es - ea
    return vpd

def calculate_fine_fuel_moisture_code(temp_c: float, rh: float, 
                                    wind_kmh: float, precip_mm: float,
                                    prev_ffmc: Optional[float] = None) -> float:
    """
    Calculate Fine Fuel Moisture Code (FFMC)
    Simplified version based on temperature, humidity, wind, and precipitation
    """
    if prev_ffmc is None:
        prev_ffmc = 85.0  # Default starting value
    
    # Calculate moisture content from temperature and humidity
    if rh > 50:
        # Wet conditions
        moisture = 0.1 * (100 - rh) * (temp_c / 20)
    else:
        # Dry conditions
        moisture = 0.1 * (100 - rh) * (temp_c / 30)
    
    # Wind effect
    wind_factor = 0.05 * wind_kmh
    
    # Precipitation effect
    if precip_mm > 0:
        rain_factor = 0.1 * precip_mm
        moisture = max(0, moisture - rain_factor)
    
    # Calculate FFMC
    ffmc = prev_ffmc - moisture + wind_factor
    
    # Clamp to valid range
    ffmc = np.clip(ffmc, 0, 101.3)
    
    return ffmc

def calculate_duff_moisture_code(temp_c: float, rh: float, 
                                precip_mm: float, prev_dmc: Optional[float] = None) -> float:
    """
    Calculate Duff Moisture Code (DMC)
    Simplified version based on temperature, humidity, and precipitation
    """
    if prev_dmc is None:
        prev_dmc = 6.0  # Default starting value
    
    # Temperature effect
    temp_factor = 0.1 * max(0, temp_c - 10)
    
    # Humidity effect
    if rh > 50:
        humidity_factor = 0.05 * (100 - rh)
    else:
        humidity_factor = 0.1 * (100 - rh)
    
    # Precipitation effect
    if precip_mm > 0:
        rain_factor = 0.2 * precip_mm
        humidity_factor = max(0, humidity_factor - rain_factor)
    
    # Calculate DMC
    dmc = prev_dmc + temp_factor + humidity_factor
    
    # Clamp to valid range
    dmc = np.clip(dmc, 0, 1000)
    
    return dmc

def calculate_drought_code(temp_c: float, precip_mm: float, 
                          prev_dc: Optional[float] = None) -> float:
    """
    Calculate Drought Code (DC)
    Simplified version based on temperature and precipitation
    """
    if prev_dc is None:
        prev_dc = 15.0  # Default starting value
    
    # Temperature effect
    temp_factor = 0.05 * max(0, temp_c - 10)
    
    # Precipitation effect
    if precip_mm > 0:
        rain_factor = 0.1 * precip_mm
        temp_factor = max(0, temp_factor - rain_factor)
    
    # Calculate DC
    dc = prev_dc + temp_factor
    
    # Clamp to valid range
    dc = np.clip(dc, 0, 1000)
    
    return dc

def calculate_fire_weather_index(ffmc: float, dmc: float, dc: float) -> float:
    """
    Calculate Fire Weather Index (FWI)
    Based on FFMC, DMC, and DC values
    """
    # Duff Moisture Code effect
    dmc_factor = 0.03229 * dmc
    
    # Drought Code effect
    dc_factor = 0.0279 * dc
    
    # Initial Spread Index
    isi = 0.208 * np.exp(0.05039 * ffmc) * (1 + dmc_factor)
    
    # Buildup Index
    bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    
    # Fire Weather Index
    if bui <= 80:
        fwi = 0.1 * isi * (0.626 * bui**0.809 + 2)
    else:
        fwi = 0.1 * isi * (1000 / (25 + 108.64 * np.exp(-0.023 * bui)))
    
    return fwi

def calculate_light_fwi_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Calculate a light FWI proxy for fire risk assessment
    Uses simplified calculations to avoid heavy computational overhead
    """
    # Ensure required columns exist
    required_cols = ['TMAX', 'TMIN', 'RH', 'WIND_SPD', 'PRCP']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    # Calculate daily average temperature
    df = df.copy()
    df['TEMP_AVG'] = (df['TMAX'] + df['TMIN']) / 2
    
    # Calculate dew point
    df['DEW_POINT'] = calculate_dew_point(df['TEMP_AVG'], df['RH'])
    
    # Calculate vapor pressure deficit
    df['VPD'] = calculate_vapor_pressure_deficit(df['TEMP_AVG'], df['RH'])
    
    # Calculate simplified moisture index
    df['MOISTURE_INDEX'] = np.where(
        df['RH'] > 50,
        0.1 * (100 - df['RH']) * (df['TEMP_AVG'] / 20),
        0.1 * (100 - df['RH']) * (df['TEMP_AVG'] / 30)
    )
    
    # Calculate wind factor
    df['WIND_FACTOR'] = 0.05 * df['WIND_SPD']
    
    # Calculate precipitation factor
    df['PRECIP_FACTOR'] = np.where(df['PRCP'] > 0, 0.1 * df['PRCP'], 0)
    
    # Calculate FWI proxy
    df['FWI_PROXY'] = (
        0.3 * df['TEMP_AVG'] / 30 +  # Temperature component
        0.3 * (100 - df['RH']) / 100 +  # Humidity component
        0.2 * df['WIND_SPD'] / 50 +  # Wind component
        0.2 * (1 - np.exp(-df['PRECIP_FACTOR']))  # Precipitation component
    )
    
    # Normalize to 0-100 scale
    df['FWI_PROXY'] = np.clip(df['FWI_PROXY'] * 100, 0, 100)
    
    return df['FWI_PROXY']

def get_fwi_threshold(fwi_values: pd.Series, quantile: float = 0.85) -> float:
    """Get FWI threshold based on quantile of values"""
    return fwi_values.quantile(quantile)

def classify_fire_risk(fwi_value: float, threshold: float) -> str:
    """Classify fire risk based on FWI value and threshold"""
    if fwi_value >= threshold:
        return "High"
    elif fwi_value >= threshold * 0.7:
        return "Moderate"
    else:
        return "Low"
