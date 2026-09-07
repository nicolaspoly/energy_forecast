"""
08b - Construction des features de prédiction pour le modèle 7 jours.

Version complète et directe (sans wrapper exec) adaptée pour le modèle 7j.
Télécharge les prévisions météo pour 7 jours, construit l'historique de demande
IESO, génère toutes les features et les sauvegarde dans Unity Catalog avec
le suffixe _7j pour éviter de conflits avec le modèle 24h.
"""

import sys
import time
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import yaml
from pyspark.sql import SparkSession

# Résolution centralisée du projet / de la configuration.
#
# Le script peut être exécuté directement depuis le fichier courant ou via
# `exec(open(...).read())` depuis un notebook d'inférence. Dans ce second cas,
# `__file__` n'est pas toujours disponible ; on garde donc un fallback explicite
# vers le workspace courant.
DEFAULT_PROJECT_ROOT = Path(
    "/Workspace/Users/n.jouglet23@gmail.com/"
    "energy_forecast"
)
PROJECT_ROOT_CANDIDATES = []

if "__file__" in globals():
    PROJECT_ROOT_CANDIDATES.append(
        Path(__file__).resolve().parents[2]
    )

PROJECT_ROOT_CANDIDATES.append(DEFAULT_PROJECT_ROOT)

PROJECT_ROOT = next(
    (
        candidate
        for candidate in PROJECT_ROOT_CANDIDATES
        if (candidate / "config" / "config.yaml").exists()
    ),
    DEFAULT_PROJECT_ROOT,
)
CONFIG_DIRECTORY = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIRECTORY / "config.yaml"

if str(CONFIG_DIRECTORY) not in sys.path:
    sys.path.append(str(CONFIG_DIRECTORY))

from zones_config import (
    WEATHER_ZONES,
    validate_weights,
)

with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    config = yaml.safe_load(_f)

CATALOG = config['catalog']['name']
SCHEMA = config['catalog']['schema']

# ============================================================
# CONFIGURATION SPÉCIFIQUE AU MODÈLE 7J
# ============================================================

MODEL_HORIZON_KEY = "horizon_7j"

# Le modèle 7j utilise plus de lags que le modèle 24h
LAG_HOURS = [1, 2, 3, 6, 24, 48, 72, 144, 168, 336]
ROLLING_WINDOWS = [3, 6, 12, 24, 48, 72, 168]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "America/Toronto"
FORECAST_HOURS = 168  # 7 jours

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "surface_pressure",
]


# ============================================================
# VALIDATION DE LA CONFIGURATION DES ZONES
# ============================================================

if not WEATHER_ZONES:
    raise ValueError(
        "WEATHER_ZONES est vide dans zones_config.py."
    )

print("=" * 80)
print("ZONES MÉTÉOROLOGIQUES CONFIGURÉES")
print("=" * 80)

for zone_key, zone_config in WEATHER_ZONES.items():
    print(
        f"{zone_key:15s} | "
        f"ville={zone_config.city:20s} | "
        f"lat={zone_config.latitude:8.4f} | "
        f"lon={zone_config.longitude:9.4f} | "
        f"poids={zone_config.weight:.4f}"
    )

total_weight = sum(
    zone.weight
    for zone in WEATHER_ZONES.values()
)

print(f"\nNombre de zones météo : {len(WEATHER_ZONES)}")
print(f"Somme des poids météo : {total_weight:.6f}")

if not validate_weights():
    raise ValueError(
        "La configuration des poids météo dans zones_config.py "
        "n'est pas valide."
    )

if not np.isclose(total_weight, 1.0, atol=1e-6):
    print(
        "ATTENTION : la somme des poids météorologiques "
        f"est égale à {total_weight:.6f}, et non à 1."
    )

WEATHER_ZONE_NAME_TO_WEIGHT = {
    zone_config.name.capitalize(): zone_config.weight
    for zone_config in WEATHER_ZONES.values()
}
WEIGHTED_ONTARIO_ZONE_NAME = "Ontario"
WEIGHTED_ONTARIO_CITY = "Ontario (weighted)"
WEIGHTED_ONTARIO_LATITUDE = sum(
    zone_config.latitude * zone_config.weight
    for zone_config in WEATHER_ZONES.values()
)
WEIGHTED_ONTARIO_LONGITUDE = sum(
    zone_config.longitude * zone_config.weight
    for zone_config in WEATHER_ZONES.values()
)

print(
    f"Zone synthétique inférence : {WEIGHTED_ONTARIO_ZONE_NAME} "
    "(moyenne pondérée des zones météo)"
)


def build_weighted_ontario_rows(
    source_df,
    group_columns,
    value_columns,
    passthrough_columns=None,
    extra_values=None,
):
    weighted_source = source_df.loc[
        source_df["zone"].astype(str).isin(
            WEATHER_ZONE_NAME_TO_WEIGHT
        )
    ].copy()

    if weighted_source.empty:
        return pd.DataFrame()

    weighted_source["_weather_zone_weight"] = (
        weighted_source["zone"]
        .astype(str)
        .map(WEATHER_ZONE_NAME_TO_WEIGHT)
        .astype(float)
    )

    ontario_rows = []

    for group_key, group_df in weighted_source.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = dict(
            zip(
                group_columns,
                group_key,
            )
        )

        if passthrough_columns:
            first_row = group_df.iloc[0]
            for column in passthrough_columns:
                row[column] = first_row[column]

        for column in value_columns:
            valid_values = group_df.loc[
                group_df[column].notna(),
                [column, "_weather_zone_weight"],
            ]

            if valid_values.empty:
                row[column] = np.nan
            else:
                row[column] = np.average(
                    valid_values[column],
                    weights=valid_values[
                        "_weather_zone_weight"
                    ],
                )

        row["zone"] = WEIGHTED_ONTARIO_ZONE_NAME

        if extra_values:
            row.update(extra_values)

        ontario_rows.append(row)

    return pd.DataFrame(ontario_rows)


def download_zone_weather_combined(
    zone_key,
    zone_config,
    prediction_start,
    prediction_end,
    timezone="America/Toronto",
    max_retries=3,
):
    """
    Télécharge les données météo pour une fenêtre de prédiction.
    Combine Archive API (heures passées) + Forecast API (heures futures).

    Parameters
    ----------
    zone_key : str
        Clé de la zone dans WEATHER_ZONES.

    zone_config : WeatherZone
        Configuration contenant latitude, longitude, ville et poids.

    prediction_start : pd.Timestamp
        Début de la fenêtre (ancre 08h).

    prediction_end : pd.Timestamp
        Fin de la fenêtre (ancre 08h + 168h pour 7j).

    timezone : str
        Fuseau horaire utilisé par Open-Meteo.

    max_retries : int
        Nombre maximal de tentatives HTTP.

    Returns
    -------
    pandas.DataFrame
        Données météo horaires (observations passées + prévisions futures).
    """

    # Déterminer maintenant (heure courante arrondie)
    now = (
        pd.Timestamp.now(tz=timezone)
        .floor("h")
        .tz_localize(None)
    )
    
    # Initialiser les listes pour combiner les données
    historical_df = None
    forecast_df = None
    
    # ============================================================
    # PARTIE 1 : HEURES PASSÉES (prediction_start → now)
    # ============================================================
    
    if prediction_start < now:
        # On a besoin de données historiques pour les heures passées
        archive_url = "https://archive-api.open-meteo.com/v1/archive"
        
        archive_params = {
            "latitude": zone_config.latitude,
            "longitude": zone_config.longitude,
            "start_date": prediction_start.strftime("%Y-%m-%d"),
            "end_date": now.strftime("%Y-%m-%d"),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": timezone,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(
                    archive_url,
                    params=archive_params,
                    timeout=60,
                )
                response.raise_for_status()
                
                payload = response.json()
                
                if "hourly" not in payload:
                    raise ValueError(
                        f"Archive API pour {zone_key} ne contient pas 'hourly'."
                    )
                
                hourly = payload["hourly"]
                
                if "time" not in hourly:
                    raise ValueError(
                        f"Archive API pour {zone_key} ne contient pas les timestamps."
                    )
                
                row_count = len(hourly["time"])
                
                def get_hourly_values(variable):
                    return hourly.get(variable, [np.nan] * row_count)
                
                historical_df = pd.DataFrame({
                    "target_datetime": pd.to_datetime(
                        hourly["time"],
                        errors="coerce",
                    ),
                    "weather_target_temperature_2m": pd.to_numeric(
                        get_hourly_values("temperature_2m"),
                        errors="coerce",
                    ),
                    "weather_target_relative_humidity_2m": pd.to_numeric(
                        get_hourly_values("relative_humidity_2m"),
                        errors="coerce",
                    ),
                    "weather_target_dew_point_2m": pd.to_numeric(
                        get_hourly_values("dew_point_2m"),
                        errors="coerce",
                    ),
                    "weather_target_precipitation": pd.to_numeric(
                        get_hourly_values("precipitation"),
                        errors="coerce",
                    ),
                    "weather_target_wind_speed_10m": pd.to_numeric(
                        get_hourly_values("wind_speed_10m"),
                        errors="coerce",
                    ),
                    "weather_target_wind_gusts_10m": pd.to_numeric(
                        get_hourly_values("wind_gusts_10m"),
                        errors="coerce",
                    ),
                    "weather_target_cloud_cover": pd.to_numeric(
                        get_hourly_values("cloud_cover"),
                        errors="coerce",
                    ),
                    "weather_target_surface_pressure": pd.to_numeric(
                        get_hourly_values("surface_pressure"),
                        errors="coerce",
                    ),
                })
                
                historical_df = historical_df.dropna(
                    subset=["target_datetime"]
                ).copy()
                
                # Filtrer pour la fenêtre [prediction_start, now[
                historical_df = historical_df[
                    (historical_df["target_datetime"] >= prediction_start) &
                    (historical_df["target_datetime"] < now)
                ].copy()
                
                print(
                    f"  Archive: {len(historical_df)} heures passées "
                    f"({historical_df['target_datetime'].min()} → {historical_df['target_datetime'].max()})"
                )
                
                break
                
            except (
                requests.RequestException,
                ValueError,
                KeyError,
            ) as error:
                last_error = error
                
                print(
                    f"  Archive tentative {attempt}/{max_retries} échouée: {error}"
                )
                
                if attempt < max_retries:
                    wait_seconds = 2 ** (attempt - 1)
                    time.sleep(wait_seconds)
        
        if historical_df is None:
            raise RuntimeError(
                f"Impossible de récupérer les données Archive pour "
                f"{zone_key} après {max_retries} tentatives."
            ) from last_error
    
    # ============================================================
    # PARTIE 2 : HEURES FUTURES (now → prediction_end)
    # ============================================================
    
    if now < prediction_end:
        # On a besoin de prévisions pour les heures futures
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        
        forecast_params = {
            "latitude": zone_config.latitude,
            "longitude": zone_config.longitude,
            "hourly": ",".join(HOURLY_VARIABLES),
            "forecast_hours": 168,
            "timezone": timezone,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(
                    forecast_url,
                    params=forecast_params,
                    timeout=60,
                )
                response.raise_for_status()
                
                payload = response.json()
                
                if "hourly" not in payload:
                    raise ValueError(
                        f"Forecast API pour {zone_key} ne contient pas 'hourly'."
                    )
                
                hourly = payload["hourly"]
                
                if "time" not in hourly:
                    raise ValueError(
                        f"Forecast API pour {zone_key} ne contient pas les timestamps."
                    )
                
                row_count = len(hourly["time"])
                
                def get_hourly_values(variable):
                    return hourly.get(variable, [np.nan] * row_count)
                
                forecast_df = pd.DataFrame({
                    "target_datetime": pd.to_datetime(
                        hourly["time"],
                        errors="coerce",
                    ),
                    "weather_target_temperature_2m": pd.to_numeric(
                        get_hourly_values("temperature_2m"),
                        errors="coerce",
                    ),
                    "weather_target_relative_humidity_2m": pd.to_numeric(
                        get_hourly_values("relative_humidity_2m"),
                        errors="coerce",
                    ),
                    "weather_target_dew_point_2m": pd.to_numeric(
                        get_hourly_values("dew_point_2m"),
                        errors="coerce",
                    ),
                    "weather_target_precipitation": pd.to_numeric(
                        get_hourly_values("precipitation"),
                        errors="coerce",
                    ),
                    "weather_target_wind_speed_10m": pd.to_numeric(
                        get_hourly_values("wind_speed_10m"),
                        errors="coerce",
                    ),
                    "weather_target_wind_gusts_10m": pd.to_numeric(
                        get_hourly_values("wind_gusts_10m"),
                        errors="coerce",
                    ),
                    "weather_target_cloud_cover": pd.to_numeric(
                        get_hourly_values("cloud_cover"),
                        errors="coerce",
                    ),
                    "weather_target_surface_pressure": pd.to_numeric(
                        get_hourly_values("surface_pressure"),
                        errors="coerce",
                    ),
                })
                
                forecast_df = forecast_df.dropna(
                    subset=["target_datetime"]
                ).copy()
                
                # Filtrer pour la fenêtre [now, prediction_end]
                forecast_df = forecast_df[
                    (forecast_df["target_datetime"] >= now) &
                    (forecast_df["target_datetime"] <= prediction_end)
                ].copy()
                
                print(
                    f"  Forecast: {len(forecast_df)} heures futures "
                    f"({forecast_df['target_datetime'].min()} → {forecast_df['target_datetime'].max()})"
                )
                
                break
                
            except (
                requests.RequestException,
                ValueError,
                KeyError,
            ) as error:
                last_error = error
                
                print(
                    f"  Forecast tentative {attempt}/{max_retries} échouée: {error}"
                )
                
                if attempt < max_retries:
                    wait_seconds = 2 ** (attempt - 1)
                    time.sleep(wait_seconds)
        
        if forecast_df is None:
            raise RuntimeError(
                f"Impossible de récupérer les prévisions Forecast pour "
                f"{zone_key} après {max_retries} tentatives."
            ) from last_error
    
    # ============================================================
    # PARTIE 3 : COMBINER HISTORIQUE + FORECAST
    # ============================================================
    
    combined_parts = []
    if historical_df is not None and not historical_df.empty:
        combined_parts.append(historical_df)
    if forecast_df is not None and not forecast_df.empty:
        combined_parts.append(forecast_df)
    
    if not combined_parts:
        raise RuntimeError(
            f"Aucune donnée météo (historique ou forecast) pour {zone_key}."
        )
    
    combined = pd.concat(combined_parts, ignore_index=True)
    combined = combined.sort_values("target_datetime").reset_index(drop=True)
    
    # Vérifier qu'on couvre bien la fenêtre de prédiction
    min_hours_required = int((prediction_end - prediction_start) / pd.Timedelta(hours=1))
    if len(combined) < min_hours_required:
        raise ValueError(
            f"{zone_key}: seulement {len(combined)} heures disponibles, "
            f"minimum {min_hours_required} requis pour la fenêtre de prédiction."
        )
    
    # Informations de la zone
    combined["zone"] = zone_config.name.capitalize()
    combined["weather_zone_key"] = zone_key
    combined["weather_city"] = zone_config.city
    combined["weather_latitude"] = zone_config.latitude
    combined["weather_longitude"] = zone_config.longitude
    combined["weather_weight"] = zone_config.weight
    
    # Moment d'émission (maintenant)
    combined["issue_datetime"] = now
    
    combined["forecast_horizon_hours"] = (
        (
            combined["target_datetime"]
            - combined["issue_datetime"]
        )
        / pd.Timedelta(hours=1)
    ).astype("int16")
    
    # Variables thermiques dérivées
    temperature = combined["weather_target_temperature_2m"]
    
    combined["weather_target_hdd18"] = (
        18.0 - temperature
    ).clip(lower=0)
    
    combined["weather_target_cdd18"] = (
        temperature - 18.0
    ).clip(lower=0)
    
    combined["weather_target_temperature_squared"] = (
        temperature ** 2
    )
    
    combined["weather_target_has_precipitation"] = (
        combined["weather_target_precipitation"] > 0
    ).astype("int8")
    
    # Calendrier de l'heure cible
    target_datetime = combined["target_datetime"]
    
    combined["target_year"] = (
        target_datetime.dt.year.astype("int16")
    )
    combined["target_month"] = (
        target_datetime.dt.month.astype("int8")
    )
    combined["target_day"] = (
        target_datetime.dt.day.astype("int8")
    )
    combined["target_hour"] = (
        target_datetime.dt.hour.astype("int8")
    )
    combined["target_day_of_week"] = (
        target_datetime.dt.dayofweek.astype("int8")
    )
    combined["target_day_of_year"] = (
        target_datetime.dt.dayofyear.astype("int16")
    )
    combined["target_is_weekend"] = (
        combined["target_day_of_week"] >= 5
    ).astype("int8")
    
    # Encodages cycliques
    combined["target_hour_sin"] = np.sin(
        2.0
        * np.pi
        * combined["target_hour"]
        / 24.0
    )
    combined["target_hour_cos"] = np.cos(
        2.0
        * np.pi
        * combined["target_hour"]
        / 24.0
    )
    
    combined["target_day_of_week_sin"] = np.sin(
        2.0
        * np.pi
        * combined["target_day_of_week"]
        / 7.0
    )
    combined["target_day_of_week_cos"] = np.cos(
        2.0
        * np.pi
        * combined["target_day_of_week"]
        / 7.0
    )
    
    combined["target_day_of_year_sin"] = np.sin(
        2.0
        * np.pi
        * combined["target_day_of_year"]
        / 365.25
    )
    combined["target_day_of_year_cos"] = np.cos(
        2.0
        * np.pi
        * combined["target_day_of_year"]
        / 365.25
    )
    
    combined["target_month_sin"] = np.sin(
        2.0
        * np.pi
        * combined["target_month"]
        / 12.0
    )
    combined["target_month_cos"] = np.cos(
        2.0
        * np.pi
        * combined["target_month"]
        / 12.0
    )
    
    return combined


# ============================================================
# LECTURE MODÈLE MLFLOW POUR CONNAÎTRE LES FEATURES ATTENDUES
# ============================================================

print("\n" + "=" * 80)
print("LECTURE MÉTADONNÉES DU MODÈLE 7J DEPUIS MLFLOW")
print("=" * 80)

import mlflow

mlflow.set_tracking_uri("databricks")

MLFLOW_EXPERIMENT = config["models"][MODEL_HORIZON_KEY]["mlflow"]["experiment_name"]

print(f"Expérience MLflow : {MLFLOW_EXPERIMENT}")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)

if experiment is None:
    raise ValueError(
        f"L'expérience MLflow '{MLFLOW_EXPERIMENT}' "
        "n'existe pas."
    )

experiment_id = experiment.experiment_id

runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="tags.`mlflow.runName` LIKE 'lgbm_zonal_direct_multi_horizon_168h%'",
    order_by=["start_time DESC"],
    max_results=1,
)

if runs.empty:
    raise RuntimeError(
        f"Aucun run MLflow 'lgbm_zonal_direct_multi_horizon_168h%' trouvé dans "
        f"l'expérience {MLFLOW_EXPERIMENT}."
    )

run_id = runs.iloc[0]["run_id"]
model_uri = f"runs:/{run_id}/model"

print(f"Run ID : {run_id}")
print(f"Model URI : {model_uri}")

model_info = mlflow.models.get_model_info(model_uri)

signature = model_info.signature

if signature is None or signature.inputs is None:
    raise ValueError(
        f"Le modèle {model_uri} n'a pas de signature d'entrée."
    )

MODEL_FEATURES = signature.inputs.input_names()

print(f"\nFeatures attendues par le modèle : {len(MODEL_FEATURES)}")

category_features = [
    feature
    for feature in MODEL_FEATURES
    if feature == "zone"
]

if category_features:
    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        if hasattr(loaded_model, "feature_names_in_"):
            feature_names = list(loaded_model.feature_names_in_)
        else:
            feature_names = MODEL_FEATURES
        
        zone_index = (
            feature_names.index("zone")
            if "zone" in feature_names
            else None
        )
        
        if zone_index is not None:
            if hasattr(loaded_model, "_le"):
                encoder = loaded_model._le
                if hasattr(encoder, "classes_"):
                    MODEL_ZONE_CATEGORIES = list(
                        encoder.classes_
                    )
                else:
                    MODEL_ZONE_CATEGORIES = []
            else:
                MODEL_ZONE_CATEGORIES = []
        else:
            MODEL_ZONE_CATEGORIES = []
    except (ImportError, ModuleNotFoundError) as e:
        print(
            f"ATTENTION : Impossible de charger le modèle ({e}). "
            "Utilisation des zones de configuration comme fallback."
        )
        # Fallback: utiliser les zones de la configuration
        MODEL_ZONE_CATEGORIES = sorted([
            zone_config.name.capitalize()
            for zone_config in WEATHER_ZONES.values()
        ] + [WEIGHTED_ONTARIO_ZONE_NAME])
else:
    MODEL_ZONE_CATEGORIES = []

if MODEL_ZONE_CATEGORIES:
    print(f"\nZones catégorielles du modèle : {MODEL_ZONE_CATEGORIES}")
else:
    print(
        "\nATTENTION : Impossible de récupérer les zones "
        "catégorielles du modèle."
    )


# ============================================================
# TÉLÉCHARGEMENT MÉTÉO PAR ZONE (FORECAST 7 JOURS)
# ============================================================

# Calculer la DERNIÈRE ancre 08h passée (identique au modèle 24h)
current_time = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)

forecast_anchor = current_time.replace(
    hour=8, minute=0, second=0, microsecond=0
)

if current_time.hour < 8:
    forecast_anchor -= pd.Timedelta(days=1)

# Pour le modèle 7j : prédire pour 7 jours complets (168 heures) à partir de l'ancre
prediction_start_7j = forecast_anchor
prediction_end_7j = forecast_anchor + pd.Timedelta(hours=168)

print("\n" + "=" * 80)
print("TÉLÉCHARGEMENT MÉTÉO PAR ZONE (FORECAST 7 JOURS)")
print("=" * 80)
print(f"Fenêtre de prédiction 7j : {prediction_start_7j} → {prediction_end_7j}")
print(f"Ancre de référence (dernière 08h passée) : {forecast_anchor}")

weather_dfs = []

for zone_key, zone_config in WEATHER_ZONES.items():
    print(f"\n{zone_key} ({zone_config.city}):")
    
    zone_df = download_zone_weather_combined(
        zone_key=zone_key,
        zone_config=zone_config,
        prediction_start=prediction_start_7j,
        prediction_end=prediction_end_7j,
        timezone=TIMEZONE,
        max_retries=3,
    )
    
    weather_dfs.append(zone_df)
    
    print(
        f"  ✅ {len(zone_df)} heures "
        f"({zone_df['target_datetime'].min()} → {zone_df['target_datetime'].max()})"
    )

weather_forecast_7j = pd.concat(
    weather_dfs,
    ignore_index=True,
    sort=False,
)

print("\n" + "=" * 80)
print("MÉTÉO TOTALE TÉLÉCHARGÉE")
print("=" * 80)
print(f"Nombre de lignes : {len(weather_forecast_7j):,}")
print(f"Zones : {sorted(weather_forecast_7j['zone'].unique())}")
print(
    f"Période : "
    f"{weather_forecast_7j['target_datetime'].min()} "
    f"à {weather_forecast_7j['target_datetime'].max()}"
)

# Ajouter la zone Ontario synthétique
if (
    WEIGHTED_ONTARIO_ZONE_NAME
    not in weather_forecast_7j["zone"].astype(str).unique()
):
    ontario_weather = build_weighted_ontario_rows(
        source_df=weather_forecast_7j,
        group_columns=["target_datetime"],
        value_columns=[
            "weather_target_temperature_2m",
            "weather_target_relative_humidity_2m",
            "weather_target_dew_point_2m",
            "weather_target_precipitation",
            "weather_target_wind_speed_10m",
            "weather_target_wind_gusts_10m",
            "weather_target_cloud_cover",
            "weather_target_surface_pressure",
            "weather_target_hdd18",
            "weather_target_cdd18",
            "weather_target_temperature_squared",
            "weather_target_has_precipitation",
        ],
        passthrough_columns=[
            "target_year",
            "target_month",
            "target_day",
            "target_hour",
            "target_day_of_week",
            "target_day_of_year",
            "target_is_weekend",
            "target_hour_sin",
            "target_hour_cos",
            "target_day_of_week_sin",
            "target_day_of_week_cos",
            "target_day_of_year_sin",
            "target_day_of_year_cos",
            "target_month_sin",
            "target_month_cos",
            "issue_datetime",
            "forecast_horizon_hours",
        ],
        extra_values={
            "weather_zone_key": "ontario_weighted",
            "weather_city": WEIGHTED_ONTARIO_CITY,
            "weather_latitude": WEIGHTED_ONTARIO_LATITUDE,
            "weather_longitude": WEIGHTED_ONTARIO_LONGITUDE,
            "weather_weight": 1.0,
        },
    )
    
    weather_forecast_7j = pd.concat(
        [weather_forecast_7j, ontario_weather],
        ignore_index=True,
        sort=False,
    )
    
    print(f"\n✅ Zone Ontario synthétique ajoutée ({len(ontario_weather)} lignes)")

weather_forecast_7j = (
    weather_forecast_7j
    .sort_values(
        ["zone", "target_datetime"]
    )
    .reset_index(drop=True)
)


# ============================================================
# IMPORT DE LA DEMANDE IESO TEMPS RÉEL PAR ZONE
# ============================================================

from io import StringIO

IESO_REALTIME_DEMAND_URL = (
    "https://reports-public.ieso.ca/public/"
    "RealtimeDemandZonal/PUB_RealtimeDemandZonal.csv"
)

# Le fichier temps réel de l'année contient suffisamment
# d'historique pour les lags jusqu'à 336 heures.
DEMAND_HISTORY_HOURS = 340
DEMAND_SAFETY_MARGIN_HOURS = 48

demand_start = (
    forecast_anchor
    - pd.Timedelta(
        hours=(
            DEMAND_HISTORY_HOURS
            + DEMAND_SAFETY_MARGIN_HOURS
        )
    )
)

print("\n" + "=" * 80)
print("IMPORT DE LA DEMANDE IESO TEMPS RÉEL PAR ZONE")
print("=" * 80)
print(f"URL : {IESO_REALTIME_DEMAND_URL}")
print(f"Historique demandé depuis : {demand_start}")


# ------------------------------------------------------------
# 1. Téléchargement avec nouvelles tentatives
# ------------------------------------------------------------

def download_ieso_realtime_zonal_demand(
    url,
    max_retries=3,
    timeout=120,
):
    """
    Télécharge et prépare le rapport IESO de demande zonale
    en temps réel à intervalles de cinq minutes.

    Returns
    -------
    raw_5min : pandas.DataFrame
        Données au format large, un enregistrement par intervalle.

    report_metadata : dict
        Métadonnées extraites de l'en-tête du rapport.
    """

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            # Éviter de récupérer une copie mise en cache.
            request_url = (
                f"{url}?download_timestamp="
                f"{int(pd.Timestamp.now().timestamp())}"
            )

            response = requests.get(
                request_url,
                timeout=timeout,
                headers={
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                    "User-Agent": (
                        "OntarioDemandForecast/1.0"
                    ),
                },
            )

            response.raise_for_status()

            if not response.text.strip():
                raise ValueError(
                    "Le fichier IESO téléchargé est vide."
                )

            # Les trois premières lignes contiennent :
            # 1. le nom du rapport
            # 2. la date de création
            # 3. l'année couverte
            header_lines = response.text.splitlines()[:3]

            report_metadata = {
                "report_name": (
                    header_lines[0].strip()
                    if len(header_lines) >= 1
                    else None
                ),
                "created_at_raw": (
                    header_lines[1].strip()
                    if len(header_lines) >= 2
                    else None
                ),
                "report_year_raw": (
                    header_lines[2].strip()
                    if len(header_lines) >= 3
                    else None
                ),
                "downloaded_at": (
                    pd.Timestamp.now(
                        tz="America/Toronto"
                    )
                ),
            }

            # La quatrième ligne est l'en-tête CSV.
            raw_5min = pd.read_csv(
                StringIO(response.text),
                skiprows=3,
                skipinitialspace=True,
            )

            # Nettoyer les noms de colonnes.
            raw_5min.columns = [
                str(column).strip()
                for column in raw_5min.columns
            ]

            required_columns = {
                "Date",
                "Hour",
                "Interval",
                "Ontario Demand",
                "NORTHWEST",
                "NORTHEAST",
                "OTTAWA",
                "EAST",
                "TORONTO",
                "ESSA",
                "BRUCE",
                "SOUTHWEST",
                "NIAGARA",
                "WEST",
            }

            missing_columns = (
                required_columns
                - set(raw_5min.columns)
            )

            if missing_columns:
                raise ValueError(
                    "Colonnes IESO manquantes : "
                    f"{sorted(missing_columns)}"
                )

            print(
                f"Rapport : "
                f"{report_metadata['report_name']}"
            )
            print(
                f"Création : "
                f"{report_metadata['created_at_raw']}"
            )
            print(
                f"Lignes téléchargées : "
                f"{len(raw_5min):,}"
            )

            return raw_5min, report_metadata

        except (
            requests.RequestException,
            ValueError,
            pd.errors.ParserError,
        ) as error:
            last_error = error

            print(
                f"Tentative {attempt}/{max_retries} "
                f"échouée : {error}"
            )

            if attempt < max_retries:
                wait_seconds = 2 ** (attempt - 1)
                print(
                    f"Nouvelle tentative dans "
                    f"{wait_seconds} seconde(s)..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        "Impossible de télécharger le rapport IESO "
        f"après {max_retries} tentatives."
    ) from last_error


ieso_realtime_raw, ieso_report_metadata = (
    download_ieso_realtime_zonal_demand(
        url=IESO_REALTIME_DEMAND_URL,
        max_retries=3,
        timeout=120,
    )
)


# ------------------------------------------------------------
# 2. Préparation des timestamps de cinq minutes
# ------------------------------------------------------------

ieso_realtime_raw["Date"] = pd.to_datetime(
    ieso_realtime_raw["Date"],
    errors="coerce",
)

ieso_realtime_raw["Hour"] = pd.to_numeric(
    ieso_realtime_raw["Hour"],
    errors="coerce",
)

ieso_realtime_raw["Interval"] = pd.to_numeric(
    ieso_realtime_raw["Interval"],
    errors="coerce",
)

ieso_realtime_raw = ieso_realtime_raw.dropna(
    subset=["Date", "Hour", "Interval"]
).copy()

ieso_realtime_raw["Hour"] = (
    ieso_realtime_raw["Hour"].astype("int16")
)

ieso_realtime_raw["Interval"] = (
    ieso_realtime_raw["Interval"].astype("int8")
)

# Validations du format IESO.
invalid_hours = ~ieso_realtime_raw["Hour"].between(
    1,
    24,
)

invalid_intervals = ~ieso_realtime_raw[
    "Interval"
].between(
    1,
    12,
)

if invalid_hours.any():
    raise ValueError(
        "Le fichier IESO contient des heures "
        "hors de l'intervalle 1 à 24."
    )

if invalid_intervals.any():
    raise ValueError(
        "Le fichier IESO contient des intervalles "
        "hors de l'intervalle 1 à 12."
    )

# timestamp_hour représente le début de l'heure IESO.
#
# Hour = 1  -> 00:00
# Hour = 2  -> 01:00
# Hour = 24 -> 23:00
ieso_realtime_raw["timestamp_hour"] = (
    ieso_realtime_raw["Date"]
    + pd.to_timedelta(
        ieso_realtime_raw["Hour"] - 1,
        unit="h",
    )
)

# timestamp_5min représente la fin de l'intervalle.
#
# Interval = 1  -> HH:05
# Interval = 12 -> heure suivante à HH+1:00
ieso_realtime_raw["timestamp_5min"] = (
    ieso_realtime_raw["timestamp_hour"]
    + pd.to_timedelta(
        ieso_realtime_raw["Interval"] * 5,
        unit="m",
    )
)


# ------------------------------------------------------------
# 3. Conversion des demandes en valeurs numériques
# ------------------------------------------------------------

IESO_ZONE_COLUMNS = [
    "NORTHWEST",
    "NORTHEAST",
    "OTTAWA",
    "EAST",
    "TORONTO",
    "ESSA",
    "BRUCE",
    "SOUTHWEST",
    "NIAGARA",
    "WEST",
]

IESO_OTHER_DEMAND_COLUMNS = [
    "Ontario Demand",
    "Zones Total",
    "DIFF",
]

numeric_columns = (
    IESO_ZONE_COLUMNS
    + IESO_OTHER_DEMAND_COLUMNS
)

for column in numeric_columns:
    if column in ieso_realtime_raw.columns:
        ieso_realtime_raw[column] = pd.to_numeric(
            ieso_realtime_raw[column],
            errors="coerce",
        )


# ------------------------------------------------------------
# 4. Détection de la dernière observation publiée
# ------------------------------------------------------------

valid_zone_measurement = (
    ieso_realtime_raw[IESO_ZONE_COLUMNS]
    .notna()
    .any(axis=1)
)

ieso_realtime_raw = ieso_realtime_raw.loc[
    valid_zone_measurement
].copy()

if ieso_realtime_raw.empty:
    raise RuntimeError(
        "Le rapport IESO ne contient aucune mesure "
        "zonale valide."
    )

latest_5min_timestamp = (
    ieso_realtime_raw["timestamp_5min"].max()
)

current_local_time = (
    pd.Timestamp.now(tz=TIMEZONE)
    .tz_localize(None)
)

publication_delay = (
    current_local_time - latest_5min_timestamp
)

publication_delay_minutes = (
    publication_delay
    / pd.Timedelta(minutes=1)
)

print("\nDernière donnée IESO disponible")
print("-" * 50)
print(f"Timestamp : {latest_5min_timestamp}")
print(
    f"Retard apparent : "
    f"{publication_delay_minutes:.1f} minute(s)"
)

# Ne pas bloquer immédiatement, mais avertir.
MAX_ACCEPTABLE_DELAY_MINUTES = 180

if (
    publication_delay_minutes
    > MAX_ACCEPTABLE_DELAY_MINUTES
):
    print(
        "ATTENTION : le rapport IESO semble en retard "
        f"de plus de {MAX_ACCEPTABLE_DELAY_MINUTES} minutes."
    )


# ------------------------------------------------------------
# 5. Format long à cinq minutes
# ------------------------------------------------------------

demand_5min_zones = ieso_realtime_raw.melt(
    id_vars=[
        "timestamp_hour",
        "timestamp_5min",
        "Date",
        "Hour",
        "Interval",
    ],
    value_vars=IESO_ZONE_COLUMNS,
    var_name="zone",
    value_name="demand_mw",
)

demand_5min_zones["demand_mw"] = pd.to_numeric(
    demand_5min_zones["demand_mw"],
    errors="coerce",
)

demand_5min_zones = (
    demand_5min_zones
    .dropna(
        subset=[
            "timestamp_5min",
            "zone",
            "demand_mw",
        ]
    )
    .sort_values(
        ["zone", "timestamp_5min"]
    )
    .reset_index(drop=True)
)

# Adapter les zones IESO au format du modèle MLflow.
#
# NORTHWEST -> Northwest
# TORONTO   -> Toronto
# SOUTHWEST -> Southwest
IESO_TO_MODEL_ZONE = {
    "NORTHWEST": "Northwest",
    "NORTHEAST": "Northeast",
    "OTTAWA": "Ottawa",
    "EAST": "East",
    "TORONTO": "Toronto",
    "ESSA": "Essa",
    "BRUCE": "Bruce",
    "SOUTHWEST": "Southwest",
    "NIAGARA": "Niagara",
    "WEST": "West",
}

demand_5min_zones["zone"] = (
    demand_5min_zones["zone"]
    .map(IESO_TO_MODEL_ZONE)
)

if demand_5min_zones["zone"].isna().any():
    raise ValueError(
        "Certaines zones IESO n'ont pas de correspondance "
        "dans IESO_TO_MODEL_ZONE."
    )

demand_5min_ontario = (
    ieso_realtime_raw[
        [
            "timestamp_hour",
            "timestamp_5min",
            "Date",
            "Hour",
            "Interval",
            "Ontario Demand",
        ]
    ]
    .rename(
        columns={
            "Ontario Demand": "demand_mw",
        }
    )
    .assign(zone=WEIGHTED_ONTARIO_ZONE_NAME)
)

demand_5min_ontario["demand_mw"] = pd.to_numeric(
    demand_5min_ontario["demand_mw"],
    errors="coerce",
)

demand_5min_ontario = demand_5min_ontario.dropna(
    subset=[
        "timestamp_5min",
        "zone",
        "demand_mw",
    ]
).reset_index(drop=True)

demand_5min_long = (
    pd.concat(
        [
            demand_5min_zones,
            demand_5min_ontario,
        ],
        ignore_index=True,
        sort=False,
    )
    .sort_values(
        ["zone", "timestamp_5min"]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 6. Vue réellement la plus récente par zone
# ------------------------------------------------------------

realtime_demand_by_zone = (
    demand_5min_long
    .sort_values("timestamp_5min")
    .groupby(
        "zone",
        as_index=False,
        observed=True,
    )
    .tail(1)
    .sort_values("zone")
    .reset_index(drop=True)
)

realtime_demand_by_zone = (
    realtime_demand_by_zone[
        [
            "timestamp_5min",
            "zone",
            "demand_mw",
        ]
    ]
)

realtime_demand_by_zone["data_age_minutes"] = (
    (
        current_local_time
        - realtime_demand_by_zone["timestamp_5min"]
    )
    / pd.Timedelta(minutes=1)
)

print("\n" + "=" * 80)
print("DEMANDE LA PLUS RÉCENTE PAR ZONE")
print("=" * 80)

print(realtime_demand_by_zone.to_string())


# ------------------------------------------------------------
# 7. Agrégation horaire pour les features du modèle
# ------------------------------------------------------------

# On agrège les 12 intervalles de cinq minutes par moyenne.
#
# Une heure est considérée complète uniquement si elle possède
# les 12 intervalles. Cela évite d'utiliser une heure partielle
# dans les rolling windows et les lags.
demand_hourly = (
    demand_5min_long
    .groupby(
        ["zone", "timestamp_hour"],
        as_index=False,
        observed=True,
    )
    .agg(
        demand_mw=("demand_mw", "mean"),
        interval_count=("Interval", "nunique"),
        demand_min_mw=("demand_mw", "min"),
        demand_max_mw=("demand_mw", "max"),
        last_interval=("Interval", "max"),
    )
)

complete_hour_mask = (
    demand_hourly["interval_count"] == 12
)

incomplete_hour_count = (
    ~complete_hour_mask
).sum()

if incomplete_hour_count:
    print(
        f"\nHeures incomplètes exclues : "
        f"{incomplete_hour_count}"
    )

demand_hourly = (
    demand_hourly.loc[complete_hour_mask]
    .rename(
        columns={
            "timestamp_hour": "datetime",
        }
    )
    .sort_values(["zone", "datetime"])
    .reset_index(drop=True)
)

# Dernière heure complète réellement disponible.
latest_complete_hour = demand_hourly["datetime"].max()

# forecast_anchor est déjà calculé plus haut (avant téléchargement météo)
ref_time = forecast_anchor

print("\nAgrégation horaire")
print("-" * 50)
print(f"Dernière heure complète IESO : {latest_complete_hour}")
print(f"Heure de référence retenue   : {ref_time}")


# ------------------------------------------------------------
# 8. Limitation à l'historique nécessaire
# ------------------------------------------------------------

history_start = (
    ref_time
    - pd.Timedelta(
        hours=(
            DEMAND_HISTORY_HOURS
            + DEMAND_SAFETY_MARGIN_HOURS
        )
    )
)

demand_history = (
    demand_hourly.loc[
        (
            demand_hourly["datetime"]
            >= history_start
        )
        & (
            demand_hourly["datetime"]
            <= ref_time
        ),
        [
            "datetime",
            "zone",
            "demand_mw",
            "interval_count",
            "demand_min_mw",
            "demand_max_mw",
        ],
    ]
    .sort_values(["zone", "datetime"])
    .reset_index(drop=True)
)

if demand_history.empty:
    raise RuntimeError(
        "Aucune donnée horaire IESO disponible "
        "dans la période demandée."
    )

print("\nHistorique IESO préparé")
print("-" * 50)
print(f"Nombre de lignes : {len(demand_history):,}")
print(
    f"Période : "
    f"{demand_history['datetime'].min()} "
    f"à {demand_history['datetime'].max()}"
)
print(
    f"Zones : "
    f"{sorted(demand_history['zone'].unique())}"
)


# ------------------------------------------------------------
# 9. Validation par rapport aux zones MLflow
# ------------------------------------------------------------

ieso_zones = set(
    demand_history["zone"].astype(str)
)

model_zones = set(
    str(zone)
    for zone in MODEL_ZONE_CATEGORIES
)

zones_missing_from_ieso = sorted(
    model_zones - ieso_zones
)

zones_unknown_to_model = sorted(
    ieso_zones - model_zones
)

if zones_missing_from_ieso:
    print(
        "\nATTENTION : zones du modèle sans demande IESO :"
    )
    for zone in zones_missing_from_ieso:
        print(f"  - {zone}")

if zones_unknown_to_model:
    print(
        "\nZones IESO non utilisées par le modèle :"
    )
    for zone in zones_unknown_to_model:
        print(f"  - {zone}")


# ------------------------------------------------------------
# 10. Validation de profondeur historique
# ------------------------------------------------------------

history_depth_by_zone = (
    demand_history
    .groupby(
        "zone",
        observed=True,
    )
    .agg(
        row_count=("datetime", "size"),
        first_datetime=("datetime", "min"),
        last_datetime=("datetime", "max"),
        missing_demand=("demand_mw", lambda x: x.isna().sum()),
    )
    .reset_index()
)

history_depth_by_zone["available_hours"] = (
    (
        history_depth_by_zone["last_datetime"]
        - history_depth_by_zone["first_datetime"]
    )
    / pd.Timedelta(hours=1)
    + 1
)

print("\nProfondeur historique par zone")
print(history_depth_by_zone.to_string())

required_history_hours = max(LAG_HOURS)

insufficient_history = history_depth_by_zone.loc[
    history_depth_by_zone["available_hours"]
    < required_history_hours
]

if not insufficient_history.empty:
    print(
        "ATTENTION : historique inférieur à "
        f"{required_history_hours} heures pour certaines zones."
    )
    print(insufficient_history.to_string())


# ============================================================
# CONSTRUCTION DES FEATURES DE DEMANDE
# ============================================================

print("\n" + "=" * 80)
print("CONSTRUCTION DES FEATURES DE DEMANDE")
print("=" * 80)

demand_features_list = []

for zone_name, zone_data in demand_history.groupby(
    "zone",
    observed=True,
):
    zone_data = (
        zone_data
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    full_range = pd.date_range(
        start=zone_data["datetime"].min(),
        end=ref_time,
        freq="h",
    )

    zone_series = (
        zone_data
        .set_index("datetime")["demand_mw"]
        .reindex(full_range)
        .astype(float)
    )

    missing_hours = int(
        zone_series.isna().sum()
    )

    if missing_hours:
        print(
            f"ATTENTION {zone_name}: "
            f"{missing_hours} heure(s) manquante(s)."
        )

    zone_series = zone_series.interpolate(
        method="time",
        limit=2,
        limit_area="inside",
    )

    features = {
        "zone": zone_name,
        "demand_realtime_mw": (
            realtime_demand_by_zone.loc[
                realtime_demand_by_zone["zone"]
                == zone_name,
                "demand_mw",
            ].iloc[0]
            if (
                realtime_demand_by_zone["zone"]
                == zone_name
            ).any()
            else np.nan
        ),
        "demand_reference_datetime": ref_time,
    }

    d_ref = zone_series.get(
        ref_time,
        np.nan,
    )

    # Lags stricts
    for lag in LAG_HOURS:
        lag_time = (
            ref_time
            - pd.Timedelta(hours=lag)
        )

        features[f"demand_lag_{lag}h"] = (
            zone_series.get(
                lag_time,
                np.nan,
            )
        )

    # Fenêtres glissantes terminées à ref_time
    for window in ROLLING_WINDOWS:
        window_start = (
            ref_time
            - pd.Timedelta(
                hours=window - 1
            )
        )

        window_data = zone_series.loc[
            window_start:ref_time
        ]

        features[
            f"demand_rolling_min_{window}h"
        ] = window_data.min()

        features[
            f"demand_rolling_max_{window}h"
        ] = window_data.max()

        features[
            f"demand_rolling_mean_{window}h"
        ] = window_data.mean()

        features[
            f"demand_rolling_std_{window}h"
        ] = window_data.std()

    # Variations
    d_1 = zone_series.get(
        ref_time - pd.Timedelta(hours=1),
        np.nan,
    )

    d_24 = zone_series.get(
        ref_time - pd.Timedelta(hours=24),
        np.nan,
    )

    d_168 = zone_series.get(
        ref_time - pd.Timedelta(hours=168),
        np.nan,
    )

    features["demand_change_1h"] = (
        d_ref - d_1
    )

    features["demand_change_24h"] = (
        d_ref - d_24
    )

    features["demand_change_168h"] = (
        d_ref - d_168
    )

    if (
        pd.notna(d_ref)
        and pd.notna(d_1)
        and abs(d_1) > 1e-6
    ):
        features["demand_pct_change_1h"] = (
            (d_ref - d_1)
            / d_1
            * 100.0
        )
    else:
        features["demand_pct_change_1h"] = np.nan

    if (
        pd.notna(d_ref)
        and pd.notna(d_24)
        and abs(d_24) > 1e-6
    ):
        features["demand_pct_change_24h"] = (
            (d_ref - d_24)
            / d_24
            * 100.0
        )
    else:
        features["demand_pct_change_24h"] = np.nan

    if (
        pd.notna(d_24)
        and pd.notna(d_168)
        and abs(d_168) > 1e-6
    ):
        features["demand_ratio_24h_168h"] = d_24 / d_168
    else:
        features["demand_ratio_24h_168h"] = np.nan

    # Écarts aux moyennes glissantes
    rolling_mean_24h = features.get(
        "demand_rolling_mean_24h",
        np.nan,
    )

    rolling_mean_168h = features.get(
        "demand_rolling_mean_168h",
        np.nan,
    )

    features["demand_vs_rolling_mean_24h"] = (
        d_ref - rolling_mean_24h
        if (
            pd.notna(d_ref)
            and pd.notna(rolling_mean_24h)
        )
        else np.nan
    )

    features["demand_vs_rolling_mean_168h"] = (
        d_ref - rolling_mean_168h
        if (
            pd.notna(d_ref)
            and pd.notna(rolling_mean_168h)
        )
        else np.nan
    )

    demand_features_list.append(features)


demand_features_all_df = pd.DataFrame(
    demand_features_list
)

model_demand_features = [
    feature
    for feature in MODEL_FEATURES
    if feature.startswith("demand_")
]

missing_demand_feature_columns = [
    feature
    for feature in model_demand_features
    if feature not in demand_features_all_df.columns
]

if missing_demand_feature_columns:
    raise ValueError(
        "Features de demande attendues mais non calculées : "
        f"{missing_demand_feature_columns}"
    )

demand_features_df = (
    demand_features_all_df[
        ["zone"] + model_demand_features
    ]
    .copy()
)

print(
    f"\nFeatures de demande : "
    f"{len(demand_features_df)} zones, "
    f"{len(model_demand_features)} features"
)

print(demand_features_df.to_string())


# ============================================================
# IMPORT MÉTÉO HISTORIQUE (Open-Meteo Archive API)
# ============================================================

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

issue_datetime = weather_forecast_7j["issue_datetime"].iloc[0]

weather_hist_start = issue_datetime - pd.Timedelta(hours=DEMAND_HISTORY_HOURS + 48)

print("\n" + "=" * 80)
print("IMPORT MÉTÉO HISTORIQUE (Open-Meteo Archive)")
print("=" * 80)
print(f"Periode : {weather_hist_start} -> {issue_datetime}")

weather_hist_dfs = []

for zone_key, zone_cfg in WEATHER_ZONES.items():
    params = {
        "latitude": zone_cfg.latitude,
        "longitude": zone_cfg.longitude,
        "start_date": weather_hist_start.strftime("%Y-%m-%d"),
        "end_date": issue_datetime.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": TIMEZONE,
    }

    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        hourly = payload.get("hourly", {})

        hist_df = pd.DataFrame({
            "datetime": pd.to_datetime(hourly.get("time", []), errors="coerce"),
            "temperature_c": pd.to_numeric(hourly.get("temperature_2m", []), errors="coerce"),
            "relative_humidity_pct": pd.to_numeric(hourly.get("relative_humidity_2m", []), errors="coerce"),
            "precipitation_mm": pd.to_numeric(hourly.get("precipitation", []), errors="coerce"),
        })

        hist_df = hist_df.dropna(subset=["datetime"]).copy()
        hist_df["zone"] = zone_cfg.name.capitalize()
        hist_df = hist_df[
            (hist_df["datetime"] >= weather_hist_start)
            & (hist_df["datetime"] <= issue_datetime)
        ]

        weather_hist_dfs.append(hist_df)
        print(f"  {zone_key} : {len(hist_df):,} lignes")

    except Exception as e:
        print(f"  Erreur {zone_key} : {e}")

if not weather_hist_dfs:
    raise RuntimeError("Aucune donnee meteo historique telechargee.")

weather_history = pd.concat(
    weather_hist_dfs,
    ignore_index=True,
)

if (
    WEIGHTED_ONTARIO_ZONE_NAME
    not in weather_history["zone"].astype(str).unique()
):
    weather_history_ontario = build_weighted_ontario_rows(
        source_df=weather_history,
        group_columns=["datetime"],
        value_columns=[
            "temperature_c",
            "relative_humidity_pct",
            "precipitation_mm",
        ],
    )

    weather_history = pd.concat(
        [weather_history, weather_history_ontario],
        ignore_index=True,
        sort=False,
    )
    print(
        "Zone Ontario synthétique ajoutée à la météo historique."
    )

weather_history = weather_history.sort_values(
    ["zone", "datetime"]
).reset_index(drop=True)

print(f"\nMeteo historique : {len(weather_history):,} lignes")


# ============================================================
# CONSTRUCTION DES FEATURES MÉTÉO HISTORIQUES
# ============================================================

print("\n" + "=" * 80)
print("CONSTRUCTION DES FEATURES MÉTÉO HISTORIQUES")
print("=" * 80)

weather_hist_features_list = []

for zone_name, zone_data in weather_history.groupby("zone"):
    zone_data = zone_data.sort_values("datetime").reset_index(drop=True)

    full_range = pd.date_range(
        start=zone_data["datetime"].min(),
        end=ref_time,
        freq="h",
    )

    temp_series = zone_data.set_index("datetime")["temperature_c"].reindex(full_range)
    humidity_series = zone_data.set_index("datetime")["relative_humidity_pct"].reindex(full_range)
    precip_series = zone_data.set_index("datetime")["precipitation_mm"].reindex(full_range)

    features = {"zone": zone_name}

    # Temperature actuelle
    features["temperature_c"] = temp_series.get(ref_time, np.nan)

    # Lags de temperature
    for lag in [24, 168]:
        lag_time = ref_time - pd.Timedelta(hours=lag - 1)
        features[f"temperature_lag_{lag}h"] = temp_series.get(lag_time, np.nan)

    # Rolling temperature
    for window in [24, 72, 168]:
        window_data = temp_series.loc[:ref_time].tail(window)
        features[f"temperature_rolling_min_{window}h"] = window_data.min()
        features[f"temperature_rolling_max_{window}h"] = window_data.max()
        features[f"temperature_rolling_mean_{window}h"] = window_data.mean()
        features[f"temperature_rolling_std_{window}h"] = window_data.std()

    # Rolling humidite
    for window in [24, 48, 72, 168]:
        humidity_window = humidity_series.loc[:ref_time].tail(window)
        features[f"humidity_rolling_mean_{window}h"] = humidity_window.mean()

    # Rolling precipitation
    for window in [72, 168]:
        precip_window = precip_series.loc[:ref_time].tail(window)
        features[f"precipitation_rolling_sum_{window}h"] = precip_window.sum()

    weather_hist_features_list.append(features)

weather_hist_features_df = pd.DataFrame(weather_hist_features_list)

model_temp_features = [
    f for f in MODEL_FEATURES
    if f.startswith("temperature_")
    or f.startswith("humidity_")
    or f.startswith("precipitation_")
]
weather_hist_features_df = weather_hist_features_df[["zone"] + model_temp_features]

print(f"\nFeatures meteo historiques : {len(weather_hist_features_df)} zones, {len(model_temp_features)} features")
print(weather_hist_features_df.to_string())


# ============================================================
# FUSION DES FEATURES DANS WEATHER_FORECAST_7J
# ============================================================

print("\n" + "=" * 80)
print("FUSION DES FEATURES")
print("=" * 80)

# Convertir zone en string pour la fusion
weather_forecast_7j["zone"] = weather_forecast_7j["zone"].astype(str)

weather_forecast_7j = weather_forecast_7j.merge(
    demand_features_df, on="zone", how="left",
)

weather_forecast_7j = weather_forecast_7j.merge(
    weather_hist_features_df, on="zone", how="left",
)

# Restaurer le type category pour zone avec les memes categories
weather_forecast_7j["zone"] = pd.Categorical(
    weather_forecast_7j["zone"],
    categories=MODEL_ZONE_CATEGORIES,
)

print(f"\nFeatures fusionnées : {len(weather_forecast_7j):,} lignes, {len(weather_forecast_7j.columns)} colonnes")


# ============================================================
# VÉRIFICATION DE COUVERTURE DES FEATURES
# ============================================================

produced_columns = set(weather_forecast_7j.columns)
model_feature_set = set(MODEL_FEATURES)

already_produced = sorted(model_feature_set & produced_columns)
missing_features = sorted(model_feature_set - produced_columns)

print("\n" + "=" * 80)
print("COUVERTURE DES FEATURES DU MODÈLE 7J")
print("=" * 80)
print(f"Features produites ici    : {len(already_produced)}/{len(MODEL_FEATURES)}")
print(f"Features manquantes       : {len(missing_features)}")

if missing_features:
    print("\nFeatures manquantes :")
    for feat in missing_features:
        print(f"  - {feat}")
else:
    print("\nToutes les features du modele sont presentes !")

# Apercu final
print("\n" + "=" * 80)
print("APERCU DES FEATURES DE PREDICTION 7J")
print("=" * 80)
print(f"Nombre de lignes : {len(weather_forecast_7j):,}")
print(f"Nombre de colonnes : {len(weather_forecast_7j.columns)}")
print(f"Zones : {sorted(weather_forecast_7j['zone'].dropna().astype(str).unique())}")
print(f"Période : {weather_forecast_7j['target_datetime'].min()} à {weather_forecast_7j['target_datetime'].max()}")

available_model_features = [
    f for f in MODEL_FEATURES if f in weather_forecast_7j.columns
]

print(f"\nFeatures du modele disponibles : {len(available_model_features)}/{len(MODEL_FEATURES)}")

print(
    weather_forecast_7j[
        ["zone", "target_datetime"] + available_model_features[:10]
    ].head(20).to_string()
)


# ============================================================
# SAUVEGARDE DES FEATURES DANS UNITY CATALOG (TABLES _7J)
# ============================================================

print("\n" + "=" * 80)
print("SAUVEGARDE DES FEATURES DANS UNITY CATALOG")
print("=" * 80)

spark = SparkSession.builder.getOrCreate()

# Tables de sortie avec suffixe _7j pour le modèle 7 jours
FEATURE_DEMAND_TABLE = f"{CATALOG}.{SCHEMA}.feature_demand_history_7j"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.feature_weather_forecast_7j"
METADATA_TABLE = f"{CATALOG}.{SCHEMA}.feature_metadata_7j"

# 1. Historique de demande brut (pour prédiction directe 7j)
print(f"\n[1/3] Sauvegarde de l'historique de demande dans {FEATURE_DEMAND_TABLE}...")
demand_history_spark = spark.createDataFrame(demand_history)
demand_history_spark.write.format("delta").mode("overwrite").saveAsTable(
    FEATURE_DEMAND_TABLE
)
print(f"  ✅ {len(demand_history):,} lignes écrites (340h d'historique)")

# 2. Features complètes (météo + demande + historique météo)
print(f"\n[2/3] Sauvegarde des features complètes dans {FEATURE_TABLE}...")
weather_forecast_export = weather_forecast_7j.copy()
weather_forecast_export["zone"] = weather_forecast_export["zone"].astype(str)
weather_forecast_spark = spark.createDataFrame(weather_forecast_export)
weather_forecast_spark.write.format("delta").mode("overwrite").saveAsTable(
    FEATURE_TABLE
)
print(f"  ✅ {len(weather_forecast_7j):,} lignes écrites")
print(f"  ✅ {len(weather_forecast_7j.columns)} colonnes (toutes les features assemblées)")

# 3. Métadonnées de référence
print(f"\n[3/3] Sauvegarde des métadonnées de référence...")
metadata_df = pd.DataFrame([{
    "ref_time": ref_time,
    "prediction_start": prediction_start_7j,
    "prediction_end": prediction_end_7j,
    "created_at": pd.Timestamp.now(),
}])
metadata_spark = spark.createDataFrame(metadata_df)
metadata_spark.write.format("delta").mode("overwrite").saveAsTable(
    METADATA_TABLE
)
print(f"  ✅ Métadonnées écrites dans {METADATA_TABLE}")

print("\n" + "=" * 80)
print("✅ FEATURES 7J SAUVEGARDÉES")
print("=" * 80)
print(f"Tables créées :")
print(f"  1. {FEATURE_DEMAND_TABLE}")
print(f"     → Historique brut IESO (340h) pour prédiction directe 7j")
print(f"     → {len(demand_history):,} lignes")
print(f"  2. {FEATURE_TABLE}")
print(f"     → Toutes les features assemblées (météo 7j + demande + historique météo)")
print(f"     → {len(weather_forecast_7j):,} lignes x {len(weather_forecast_7j.columns)} colonnes")
print(f"  3. {METADATA_TABLE}")
print(f"     → Timestamps de référence (ref_time, prediction_start, prediction_end)")
print(f"\nPériode de prédiction : {prediction_start_7j} → {prediction_end_7j} (168 heures)")
print("\nVous pouvez maintenant exécuter 09b_batch_prediction_7j.py indépendamment.")