"""
Utilitaires pour le traitement des données météo Open-Meteo

Fonctions pour:
- Requêtes API Open-Meteo
- Calcul de variables dérivées (HDD, CDD)
- Pondération multi-zones
- Qualité des données météo

Auteur: Energy Forecast Project
Date: 2026-08-28
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def fetch_openmeteo_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7,
    variables: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Récupère les prévisions météo Open-Meteo pour une zone.
    
    Args:
        latitude: Latitude de la zone
        longitude: Longitude de la zone
        forecast_days: Nombre de jours de prévision (1-16)
        variables: Variables à récupérer (None = toutes)
    
    Returns:
        DataFrame avec colonnes: datetime, temperature_2m, humidity, etc.
    """
    if variables is None:
        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "windspeed_10m",
            "cloudcover"
        ]
    
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "forecast_days": min(forecast_days, 16),  # Max 16 jours
        "timezone": "America/Toronto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Parser la réponse
        df = pd.DataFrame({
            'datetime': pd.to_datetime(data['hourly']['time']),
            **{var: data['hourly'][var] for var in variables}
        })
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur API Open-Meteo: {e}")
        return pd.DataFrame()


def calculate_hdd(temperature: float, base_temp: float = 18.0) -> float:
    """
    Calcule les Heating Degree Days (HDD).
    HDD mesure le besoin en chauffage.
    
    Args:
        temperature: Température en °C
        base_temp: Température de référence (18°C par défaut)
    
    Returns:
        HDD (0 si temp >= base_temp)
    """
    return max(base_temp - temperature, 0)


def calculate_cdd(temperature: float, base_temp: float = 18.0) -> float:
    """
    Calcule les Cooling Degree Days (CDD).
    CDD mesure le besoin en climatisation.
    
    Args:
        temperature: Température en °C
        base_temp: Température de référence (18°C par défaut)
    
    Returns:
        CDD (0 si temp <= base_temp)
    """
    return max(temperature - base_temp, 0)


def calculate_weighted_weather(
    zone_forecasts: Dict[str, pd.DataFrame],
    zone_weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Calcule les variables météo pondérées à travers plusieurs zones.
    
    Args:
        zone_forecasts: Dict {zone_name: DataFrame avec prévisions}
        zone_weights: Dict {zone_name: weight}
    
    Returns:
        DataFrame avec variables pondérées
    """
    if not zone_forecasts:
        return pd.DataFrame()
    
    # Premier DataFrame comme base pour les timestamps
    first_zone = list(zone_forecasts.keys())[0]
    result = zone_forecasts[first_zone][['datetime']].copy()
    
    # Variables numériques à pondérer
    numeric_cols = zone_forecasts[first_zone].select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        weighted_values = np.zeros(len(result))
        
        for zone_name, df in zone_forecasts.items():
            if zone_name in zone_weights and col in df.columns:
                weight = zone_weights[zone_name]
                weighted_values += df[col].values * weight
        
        result[f'weighted_{col}'] = weighted_values
    
    return result


def add_derived_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les variables météo dérivées (HDD, CDD, etc.).
    
    Args:
        df: DataFrame avec au minimum une colonne 'temperature_2m'
    
    Returns:
        DataFrame enrichi avec HDD, CDD, etc.
    """
    df = df.copy()
    
    if 'temperature_2m' in df.columns:
        df['hdd'] = df['temperature_2m'].apply(calculate_hdd)
        df['cdd'] = df['temperature_2m'].apply(calculate_cdd)
    
    if 'weighted_temperature_2m' in df.columns:
        df['weighted_hdd'] = df['weighted_temperature_2m'].apply(calculate_hdd)
        df['weighted_cdd'] = df['weighted_temperature_2m'].apply(calculate_cdd)
    
    # Indice de confort thermique simple
    if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
        df['heat_index'] = df['temperature_2m'] + (df['relative_humidity_2m'] / 100) * 2
    
    return df


def check_weather_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Vérifie la qualité des données météo.
    
    Args:
        df: DataFrame avec données météo
    
    Returns:
        Dict avec statistiques de qualité
    """
    quality = {
        'total_rows': len(df),
        'missing_values': {},
        'outliers': {},
        'valid': True
    }
    
    # Vérifier valeurs manquantes
    for col in df.columns:
        if col != 'datetime':
            missing = df[col].isna().sum()
            if missing > 0:
                quality['missing_values'][col] = missing
                quality['valid'] = False
    
    # Vérifier outliers (valeurs aberrantes)
    checks = {
        'temperature_2m': (-50, 50),  # °C
        'relative_humidity_2m': (0, 100),  # %
        'windspeed_10m': (0, 150)  # km/h
    }
    
    for col, (min_val, max_val) in checks.items():
        if col in df.columns:
            outliers = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if outliers > 0:
                quality['outliers'][col] = outliers
                quality['valid'] = False
    
    return quality


if __name__ == "__main__":
    # Test rapide
    print("\n☁️  TEST WEATHER UTILS\n")
    print("="*70)
    
    # Test HDD/CDD
    temps = [-10, 0, 10, 18, 25, 35]
    print("\nHDD/CDD pour différentes températures:")
    for t in temps:
        hdd = calculate_hdd(t)
        cdd = calculate_cdd(t)
        print(f"  {t:3d}°C  →  HDD: {hdd:5.1f}  |  CDD: {cdd:5.1f}")
    
    print("\n" + "="*70)
    print("✅ Tests OK")
