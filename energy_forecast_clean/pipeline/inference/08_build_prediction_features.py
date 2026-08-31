import sys
import time
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import yaml

# Résolution centralisée du projet / de la configuration.
#
# Le script peut être exécuté directement depuis le fichier courant ou via
# `exec(open(...).read())` depuis un notebook d'inférence. Dans ce second cas,
# `__file__` n'est pas toujours disponible ; on garde donc un fallback explicite
# vers le workspace courant.
DEFAULT_PROJECT_ROOT = Path(
    "/Workspace/Users/n.jouglet23@gmail.com/"
    "energy_forecast_clean"
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

# ============================================================
# CONFIGURATION
# ============================================================

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


def download_zone_weather(
    zone_key,
    zone_config,
    forecast_hours=24,
    timezone="America/Toronto",
    max_retries=3,
):
    """
    Télécharge les prévisions météo horaires d'une zone.

    Parameters
    ----------
    zone_key : str
        Clé de la zone dans WEATHER_ZONES.

    zone_config : WeatherZone
        Configuration contenant latitude, longitude, ville et poids.

    forecast_hours : int
        Nombre d'heures futures à conserver.

    timezone : str
        Fuseau horaire utilisé par Open-Meteo.

    max_retries : int
        Nombre maximal de tentatives HTTP.

    Returns
    -------
    pandas.DataFrame
        Prévisions horaires de la zone.
    """

    params = {
        "latitude": zone_config.latitude,
        "longitude": zone_config.longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_days": 8,
        "timezone": timezone,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                OPEN_METEO_URL,
                params=params,
                timeout=60,
            )
            response.raise_for_status()

            payload = response.json()

            if "hourly" not in payload:
                raise ValueError(
                    f"La réponse pour {zone_key} ne contient "
                    "pas la section 'hourly'."
                )

            hourly = payload["hourly"]

            if "time" not in hourly:
                raise ValueError(
                    f"La réponse pour {zone_key} ne contient "
                    "pas les timestamps horaires."
                )

            row_count = len(hourly["time"])

            def get_hourly_values(variable):
                """
                Retourne les valeurs de la variable ou une liste de NaN
                si la variable est absente.
                """
                return hourly.get(
                    variable,
                    [np.nan] * row_count,
                )

            forecast = pd.DataFrame({
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

            forecast = forecast.dropna(
                subset=["target_datetime"]
            ).copy()

            # Première heure complète à prédire.
            #
            # Exemple :
            # exécution à 15 h 25 -> première heure = 16 h 00
            first_forecast_hour = (
                pd.Timestamp.now(tz=timezone)
                .ceil("h")
                .tz_localize(None)
            )

            forecast_end = (
                first_forecast_hour
                + pd.Timedelta(hours=forecast_hours)
            )

            forecast = (
                forecast.loc[
                    (
                        forecast["target_datetime"]
                        >= first_forecast_hour
                    )
                    & (
                        forecast["target_datetime"]
                        < forecast_end
                    )
                ]
                .sort_values("target_datetime")
                .head(forecast_hours)
                .reset_index(drop=True)
            )

            if len(forecast) != forecast_hours:
                raise ValueError(
                    f"{zone_key}: {len(forecast)} heures "
                    f"récupérées au lieu de {forecast_hours}."
                )

            # Informations de la zone
            forecast["zone"] = zone_config.name.capitalize()
            forecast["weather_zone_key"] = zone_key
            forecast["weather_city"] = zone_config.city
            forecast["weather_latitude"] = zone_config.latitude
            forecast["weather_longitude"] = zone_config.longitude
            forecast["weather_weight"] = zone_config.weight

            # Moment d'émission de la prévision
            issue_datetime = (
                pd.Timestamp.now(tz=timezone)
                .floor("h")
                .tz_localize(None)
            )

            forecast["issue_datetime"] = issue_datetime

            forecast["forecast_horizon_hours"] = (
                (
                    forecast["target_datetime"]
                    - forecast["issue_datetime"]
                )
                / pd.Timedelta(hours=1)
            ).astype("int16")

            # Variables thermiques dérivées
            temperature = (
                forecast["weather_target_temperature_2m"]
            )

            forecast["weather_target_hdd18"] = (
                18.0 - temperature
            ).clip(lower=0)

            forecast["weather_target_cdd18"] = (
                temperature - 18.0
            ).clip(lower=0)

            forecast["weather_target_temperature_squared"] = (
                temperature ** 2
            )

            forecast["weather_target_has_precipitation"] = (
                forecast["weather_target_precipitation"] > 0
            ).astype("int8")

            # Calendrier de l'heure cible
            target_datetime = forecast["target_datetime"]

            forecast["target_year"] = (
                target_datetime.dt.year.astype("int16")
            )
            forecast["target_month"] = (
                target_datetime.dt.month.astype("int8")
            )
            forecast["target_day"] = (
                target_datetime.dt.day.astype("int8")
            )
            forecast["target_hour"] = (
                target_datetime.dt.hour.astype("int8")
            )
            forecast["target_day_of_week"] = (
                target_datetime.dt.dayofweek.astype("int8")
            )
            forecast["target_day_of_year"] = (
                target_datetime.dt.dayofyear.astype("int16")
            )
            forecast["target_is_weekend"] = (
                forecast["target_day_of_week"] >= 5
            ).astype("int8")

            # Encodages cycliques
            forecast["target_hour_sin"] = np.sin(
                2.0
                * np.pi
                * forecast["target_hour"]
                / 24.0
            )

            forecast["target_hour_cos"] = np.cos(
                2.0
                * np.pi
                * forecast["target_hour"]
                / 24.0
            )

            forecast["target_day_of_week_sin"] = np.sin(
                2.0
                * np.pi
                * forecast["target_day_of_week"]
                / 7.0
            )

            forecast["target_day_of_week_cos"] = np.cos(
                2.0
                * np.pi
                * forecast["target_day_of_week"]
                / 7.0
            )

            return forecast

        except (
            requests.RequestException,
            ValueError,
            KeyError,
        ) as error:
            last_error = error

            print(
                f"Tentative {attempt}/{max_retries} "
                f"échouée pour {zone_key}: {error}"
            )

            if attempt < max_retries:
                wait_seconds = 2 ** (attempt - 1)
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Impossible de récupérer la météo pour "
        f"{zone_key} après {max_retries} tentatives."
    ) from last_error
# ============================================================
# IMPORT DE TOUTES LES ZONES
# ============================================================

forecast_dfs = {}
download_errors = {}

print("\n" + "=" * 80)
print("TÉLÉCHARGEMENT DES PRÉVISIONS MÉTÉOROLOGIQUES")
print("=" * 80)

for zone_key, zone_config in WEATHER_ZONES.items():
    print(
        f"\nTéléchargement : {zone_key} "
        f"({zone_config.city})"
    )

    try:
        zone_forecast = download_zone_weather(
            zone_key=zone_key,
            zone_config=zone_config,
            forecast_hours=FORECAST_HOURS,
            timezone=TIMEZONE,
        )

        forecast_dfs[zone_key] = zone_forecast

        print(
            f"OK : {len(zone_forecast)} heures | "
            f"{zone_forecast['target_datetime'].min()} à "
            f"{zone_forecast['target_datetime'].max()}"
        )

    except Exception as error:
        download_errors[zone_key] = str(error)
        print(f"ERREUR : {error}")


if not forecast_dfs:
    raise RuntimeError(
        "Aucune zone météo n'a pu être téléchargée."
    )


weather_forecast_24h = pd.concat(
    forecast_dfs.values(),
    ignore_index=True,
)

if (
    WEIGHTED_ONTARIO_ZONE_NAME
    not in weather_forecast_24h["zone"].astype(str).unique()
):
    forecast_ontario = build_weighted_ontario_rows(
        source_df=weather_forecast_24h,
        group_columns=[
            "issue_datetime",
            "target_datetime",
        ],
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
            "forecast_horizon_hours",
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
        ],
        extra_values={
            "weather_zone_key": "ontario_weighted",
            "weather_city": WEIGHTED_ONTARIO_CITY,
            "weather_latitude": WEIGHTED_ONTARIO_LATITUDE,
            "weather_longitude": WEIGHTED_ONTARIO_LONGITUDE,
            "weather_weight": 1.0,
        },
    )

    weather_forecast_24h = pd.concat(
        [weather_forecast_24h, forecast_ontario],
        ignore_index=True,
        sort=False,
    )
    print(
        "Zone Ontario synthétique ajoutée aux prévisions météo."
    )

weather_forecast_24h = (
    weather_forecast_24h
    .sort_values(["target_datetime", "zone"])
    .reset_index(drop=True)
)

print("\nTéléchargement terminé.")
print(f"Zones réussies : {len(forecast_dfs)}")
print(f"Zones en erreur : {len(download_errors)}")
print(f"Nombre de lignes : {len(weather_forecast_24h)}")

print(
    weather_forecast_24h[
        [
            "issue_datetime",
            "target_datetime",
            "forecast_horizon_hours",
            "zone",
            "weather_city",
            "weather_weight",
            "weather_target_temperature_2m",
            "weather_target_dew_point_2m",
            "weather_target_relative_humidity_2m",
            "weather_target_precipitation",
            "weather_target_wind_speed_10m",
            "weather_target_hdd18",
            "weather_target_cdd18",
        ]
    ]
    .sort_values(["zone", "target_datetime"])
    .head(20)
    .to_string()
)


# ============================================================
# IMPORT DES FEATURES DU MODÈLE DEPUIS MLFLOW
# ============================================================

import json
import mlflow

# Clé du modèle cible : 'horizon_24h' ou 'horizon_7j'.
# Peut être surdéfinie avant exec() par le notebook appelant.
MODEL_HORIZON_KEY = globals().get("MODEL_HORIZON_KEY", "horizon_24h")
MLFLOW_EXPERIMENT = config["models"][MODEL_HORIZON_KEY]["mlflow"]["experiment_name"]

mlflow.set_experiment(MLFLOW_EXPERIMENT)

mlflow_runs = mlflow.search_runs(
    experiment_names=[MLFLOW_EXPERIMENT],
    order_by=["start_time DESC"],
    max_results=1,
)

if mlflow_runs.empty:
    raise RuntimeError(
        "Aucun run MLflow trouvé pour l'expérience "
        f"{MLFLOW_EXPERIMENT}."
    )

MLFLOW_RUN_ID = mlflow_runs.iloc[0]["run_id"]

print(f"\nRun MLflow sélectionné : {MLFLOW_RUN_ID}")

# Téléchargement des artefacts du run.
selected_features_path = (
    mlflow.artifacts.download_artifacts(
        run_id=MLFLOW_RUN_ID,
        artifact_path="analysis/selected_features.json",
    )
)

with open(
    selected_features_path,
    "r",
    encoding="utf-8",
) as f:
    MODEL_FEATURES = json.load(f)

metadata_path = (
    mlflow.artifacts.download_artifacts(
        run_id=MLFLOW_RUN_ID,
        artifact_path="analysis/preprocessing_metadata.json",
    )
)

with open(
    metadata_path,
    "r",
    encoding="utf-8",
) as f:
    MODEL_METADATA = json.load(f)

MODEL_TARGET_COLUMN = MODEL_METADATA["target_column"]
MODEL_ZONE_CATEGORIES = MODEL_METADATA["zone_categories"]

print(
    f"Modèle entraîné avec {len(MODEL_FEATURES)} features."
)
print(f"Colonne cible : {MODEL_TARGET_COLUMN}")
print(f"Catégories de zone : {MODEL_ZONE_CATEGORIES}")

print("\nFeatures attendues par le modèle :")
for i, feat in enumerate(MODEL_FEATURES, 1):
    print(f"{i:3d}. {feat}")


# ============================================================
# ALIGNEMENT DES FEATURES MÉTÉO AVEC LE MODÈLE
# ============================================================

# Le modèle a été entraîné sur des colonnes dont les noms diffèrent
# de celles produites par Open-Meteo. On renomme et on crée les
# features dérivées attendues.

weather_forecast_24h = weather_forecast_24h.rename(
    columns={
        "weather_target_temperature_2m": (
            "weather_target_temperature_c"
        ),
        "weather_target_relative_humidity_2m": (
            "weather_target_relative_humidity_pct"
        ),
        "weather_target_dew_point_2m": (
            "weather_target_dew_point_c"
        ),
        "weather_target_wind_speed_10m": (
            "weather_target_wind_speed_kmh"
        ),
        "weather_target_temperature_squared": (
            "weather_target_temperature_sq"
        ),
    }
)

# Features météo dérivées attendues par le modèle.
temperature_c = weather_forecast_24h[
    "weather_target_temperature_c"
]
humidity_pct = weather_forecast_24h[
    "weather_target_relative_humidity_pct"
]

# Température du thermomètre mouillé approximative.
weather_forecast_24h[
    "weather_target_wet_bulb_approx_c"
] = (
    temperature_c
    * np.arctan(0.151977 * np.sqrt(
        humidity_pct + 8.313659
    ))
    + np.arctan(temperature_c + humidity_pct)
    - np.arctan(humidity_pct - 1.676331)
    + 0.00391838 * humidity_pct ** 1.5
    * np.arctan(0.023101 * humidity_pct)
    - 4.686035
)

# Interaction température x humidité.
weather_forecast_24h[
    "weather_target_temperature_x_humidity"
] = temperature_c * humidity_pct

# Écart température - point de rosée.
weather_forecast_24h[
    "weather_target_temp_dew_spread"
] = (
    temperature_c
    - weather_forecast_24h["weather_target_dew_point_c"]
)

# HDD avec base 15.5 °C.
weather_forecast_24h[
    "weather_target_hdd15_5"
] = (15.5 - temperature_c).clip(lower=0)

# Interactions HDD18 x encodages cycliques.
weather_forecast_24h[
    "weather_target_hdd18_x_hour_sin"
] = (
    weather_forecast_24h["weather_target_hdd18"]
    * weather_forecast_24h["target_hour_sin"]
)

weather_forecast_24h[
    "weather_target_hdd18_x_hour_cos"
] = (
    weather_forecast_24h["weather_target_hdd18"]
    * weather_forecast_24h["target_hour_cos"]
)

weather_forecast_24h[
    "weather_target_hdd18_x_weekend"
] = (
    weather_forecast_24h["weather_target_hdd18"]
    * weather_forecast_24h["target_is_weekend"]
)

weather_forecast_24h[
    "weather_target_temperature_x_hour_cos"
] = (
    temperature_c
    * weather_forecast_24h["target_hour_cos"]
)

# Encodages cycliques du jour de l'année.
weather_forecast_24h[
    "target_day_of_year_sin"
] = np.sin(
    2.0
    * np.pi
    * weather_forecast_24h["target_day_of_year"]
    / 365.25
)

weather_forecast_24h[
    "target_day_of_year_cos"
] = np.cos(
    2.0
    * np.pi
    * weather_forecast_24h["target_day_of_year"]
    / 365.25
)

# Heure d'émission (issue_datetime).
issue_hour = weather_forecast_24h[
    "issue_datetime"
].dt.hour.astype("int8")

weather_forecast_24h["issue_hour_sin"] = np.sin(
    2.0 * np.pi * issue_hour / 24.0
)

weather_forecast_24h["issue_hour_cos"] = np.cos(
    2.0 * np.pi * issue_hour / 24.0
)

# Jour férié (placeholder -- a remplacer par un vrai calendrier).
weather_forecast_24h["is_holiday"] = 0

# Conversion de zone en type category avec les memes
# catégories que le modèle.
weather_forecast_24h["zone"] = pd.Categorical(
    weather_forecast_24h["zone"],
    categories=MODEL_ZONE_CATEGORIES,
)


# ============================================================
# IMPORT DE LA DEMANDE IESO TEMPS REEL PAR ZONE
# ============================================================

from io import StringIO
import requests
import numpy as np
import pandas as pd


IESO_REALTIME_DEMAND_URL = (
    "https://reports-public.ieso.ca/public/"
    "RealtimeDemandZonal/PUB_RealtimeDemandZonal.csv"
)

# Le fichier temps réel de l'année contient suffisamment
# d'historique pour les lags jusqu'à 336 heures.
DEMAND_HISTORY_HOURS = 340
DEMAND_SAFETY_MARGIN_HOURS = 48

demand_start = (
    weather_forecast_24h["issue_datetime"].min()
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

# ref_time doit être basé sur les données réellement disponibles,
# et non simplement sur issue_datetime - 1 heure.
ref_time = min(
    (
        weather_forecast_24h["issue_datetime"].min()
        - pd.Timedelta(hours=1)
    ),
    latest_complete_hour,
)

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

required_history_hours = max(
    [1, 24, 48, 72, 144, 168, 336]
)

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

LAG_HOURS = globals().get("LAG_HOURS", [
    1,
    24,
    48,
    72,
    144,
    168,
    336,
])

ROLLING_WINDOWS = globals().get("ROLLING_WINDOWS", [
    3,
    12,
    24,
    48,
    72,
    168,
])

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

    # Interpolation limitée aux petits trous.
    #
    # Ne pas interpoler de longues périodes absentes.
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

    # Valeur à la dernière heure complète.
    d_ref = zone_series.get(
        ref_time,
        np.nan,
    )

    # Lags stricts.
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

    # Fenêtres glissantes terminées à ref_time.
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

    # Variations.
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

    # Écarts aux moyennes glissantes.
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

# Ne conserver que les features utilisées par le modèle.
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
# IMPORT METEO HISTORIQUE (Open-Meteo Archive API)
# ============================================================

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# issue_datetime est defini dans download_zone_weather (scope local).
# On le recupere depuis le DataFrame pour le scope module.
issue_datetime = weather_forecast_24h["issue_datetime"].iloc[0]

weather_hist_start = issue_datetime - pd.Timedelta(hours=DEMAND_HISTORY_HOURS + 48)

print("\n" + "=" * 80)
print("IMPORT METEO HISTORIQUE (Open-Meteo Archive)")
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
# CONSTRUCTION DES FEATURES METEO HISTORIQUES
# ============================================================

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

    # Temperature actuelle.
    features["temperature_c"] = temp_series.get(ref_time, np.nan)

    # Lags de temperature.
    for lag in [24, 168]:
        lag_time = ref_time - pd.Timedelta(hours=lag - 1)
        features[f"temperature_lag_{lag}h"] = temp_series.get(lag_time, np.nan)

    # Rolling temperature.
    for window in [24, 72, 168]:
        window_data = temp_series.loc[:ref_time].tail(window)
        features[f"temperature_rolling_min_{window}h"] = window_data.min()
        features[f"temperature_rolling_max_{window}h"] = window_data.max()
        features[f"temperature_rolling_mean_{window}h"] = window_data.mean()

    # Rolling humidite.
    humidity_168 = humidity_series.loc[:ref_time].tail(168)
    features["humidity_rolling_mean_168h"] = humidity_168.mean()

    # Rolling precipitation.
    precip_168 = precip_series.loc[:ref_time].tail(168)
    features["precipitation_rolling_sum_168h"] = precip_168.sum()

    weather_hist_features_list.append(features)

weather_hist_features_df = pd.DataFrame(weather_hist_features_list)

# Ne conserver que les features meteo historiques du modele.
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
# FUSION DES FEATURES DANS WEATHER_FORECAST_24H
# ============================================================

# Les features de demande et meteo historique sont identiques
# pour toutes les heures de prediction d'une meme zone.
# On fusionne sur la colonne 'zone'.

# Convertir zone en string pour la fusion (elle est category).
weather_forecast_24h["zone"] = weather_forecast_24h["zone"].astype(str)

weather_forecast_24h = weather_forecast_24h.merge(
    demand_features_df, on="zone", how="left",
)

weather_forecast_24h = weather_forecast_24h.merge(
    weather_hist_features_df, on="zone", how="left",
)

# Restaurer le type category pour zone avec les memes categories.
weather_forecast_24h["zone"] = pd.Categorical(
    weather_forecast_24h["zone"],
    categories=MODEL_ZONE_CATEGORIES,
)


# ============================================================
# VERIFICATION DE COUVERTURE DES FEATURES
# ============================================================

produced_columns = set(weather_forecast_24h.columns)
model_feature_set = set(MODEL_FEATURES)

already_produced = sorted(model_feature_set & produced_columns)
missing_features = sorted(model_feature_set - produced_columns)

print("\n" + "=" * 80)
print("COUVERTURE DES FEATURES DU MODELE")
print("=" * 80)
print(f"Features produites ici    : {len(already_produced)}/{len(MODEL_FEATURES)}")
print(f"Features manquantes       : {len(missing_features)}")

if missing_features:
    print("\nFeatures manquantes :")
    for feat in missing_features:
        print(f"  - {feat}")
else:
    print("\nToutes les features du modele sont presentes !")

# Apercu final.
print("\n" + "=" * 80)
print("APERCU DES FEATURES DE PREDICTION")
print("=" * 80)
print(f"Nombre de lignes : {len(weather_forecast_24h):,}")
print(f"Nombre de colonnes : {len(weather_forecast_24h.columns)}")
print(f"Zones : {sorted(weather_forecast_24h['zone'].dropna().astype(str).unique())}")

# Selectionner uniquement les features du modele pour l'apercu.
available_model_features = [
    f for f in MODEL_FEATURES if f in weather_forecast_24h.columns
]

print(f"\nFeatures du modele disponibles : {len(available_model_features)}/{len(MODEL_FEATURES)}")

print(
    weather_forecast_24h[
        ["zone", "target_datetime"] + available_model_features
    ].head(20).to_string()
)


