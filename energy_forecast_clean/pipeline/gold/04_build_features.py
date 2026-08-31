
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gold Layer - Features pour Machine Learning

Objectif
--------
Créer les features nécessaires pour prévoir la demande électrique de chaque
zone IESO exactement 24 heures à l'avance.

Grain du dataset
----------------
Une ligne représente :
    zone + issue_datetime

La cible est :
    demande électrique à issue_datetime + 24 heures

Les variables météo utilisées pour le modèle à horizon 24 h sont alignées
sur target_datetime, c'est-à-dire la date et l'heure réellement prédites.

Source
------
workspace.energy_forecast.demand_weather_silver

Destination
-----------
workspace.energy_forecast.ml_features_gold

Hypothèses
----------
1. La table Silver contient une observation par zone et par heure.
2. La colonne datetime est un timestamp horaire.
3. Pour l'entraînement historique, la météo future est obtenue avec lead(24).
4. En production, les colonnes weather_target_* devront être alimentées avec
   les prévisions Open-Meteo valides pour target_datetime.
5. La table contient au minimum les colonnes suivantes :
   - zone
   - datetime
   - demand_mw
   - temperature_c
   - dew_point_c
   - relative_humidity_pct
   - wind_speed_kmh
   - precipitation_mm
"""

import yaml
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

CONFIG_PATH = (
    "/Workspace/Users/n.jouglet23@gmail.com/"
    "energy_forecast_clean/config/config.yaml"
)

FORECAST_HORIZON_HOURS = 24
BASE_TEMPERATURE_C = 18.0
SECONDARY_BASE_TEMPERATURE_C = 15.5

REQUIRED_COLUMNS = [
    "zone",
    "datetime",
    "demand_mw",
    "temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
]

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

CATALOG = config["catalog"]["name"]
SCHEMA = config["catalog"]["schema"]

INPUT_TABLE = f"{CATALOG}.{SCHEMA}.demand_weather_silver"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.ml_features_gold_24h"

spark = SparkSession.builder.getOrCreate()

# Important pour les timestamps et les jours fériés ontariens.
spark.conf.set("spark.sql.session.timeZone", "America/Toronto")

print("=" * 80)
print("GOLD LAYER - Feature Engineering pour ML")
print(f"Source      : {INPUT_TABLE}")
print(f"Destination : {OUTPUT_TABLE}")
print(f"Horizon     : {FORECAST_HORIZON_HOURS} heures")
print("=" * 80)


# =============================================================================
# 1. CHARGEMENT ET VALIDATION DE LA TABLE SILVER
# =============================================================================

print("\n1. Chargement de la table Silver")

df_silver = spark.table(INPUT_TABLE)

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df_silver.columns
]

if missing_columns:
    raise ValueError(
        "Colonnes obligatoires absentes de la table Silver : "
        + ", ".join(missing_columns)
    )

df = (
    df_silver
    .select(
        "*",
        F.col("datetime").cast("timestamp").alias("_validated_datetime")
    )
    .drop("datetime")
    .withColumnRenamed("_validated_datetime", "datetime")
)

# Conversion explicite des variables numériques.
numeric_columns = [
    "demand_mw",
    "temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
]

df = df.withColumns({
    column: F.col(column).cast(DoubleType())
    for column in numeric_columns
})

# Suppression des observations sans clé ou sans demande.
df = df.filter(
    F.col("zone").isNotNull()
    & F.col("datetime").isNotNull()
    & F.col("demand_mw").isNotNull()
)

count_loaded = df.count()
print(f"   Lignes chargées : {count_loaded:,}")

if count_loaded == 0:
    raise ValueError("La table Silver ne contient aucune ligne exploitable.")


# =============================================================================
# 2. CONTRÔLES DE QUALITÉ
# =============================================================================

print("\n2. Contrôles de qualité")

duplicate_count = (
    df.groupBy("zone", "datetime")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

print(f"   Doublons zone-datetime : {duplicate_count:,}")

if duplicate_count > 0:
    print(
        "   Attention : les doublons sont agrégés avant le calcul des features."
    )

    aggregation_expressions = []

    for column in df.columns:
        if column not in ["zone", "datetime"]:
            aggregation_expressions.append(
                F.first(F.col(column), ignorenulls=True).alias(column)
            )

    df = (
        df.groupBy("zone", "datetime")
        .agg(*aggregation_expressions)
    )

# Fenêtre utilisée pour contrôler la continuité.
continuity_window = Window.partitionBy("zone").orderBy("datetime")

df = df.withColumn(
    "_previous_datetime",
    F.lag("datetime", 1).over(continuity_window)
)

df = df.withColumn(
    "_hours_since_previous",
    (
        F.col("datetime").cast("long")
        - F.col("_previous_datetime").cast("long")
    ) / F.lit(3600.0)
)

gap_count = (
    df.filter(
        F.col("_previous_datetime").isNotNull()
        & (F.col("_hours_since_previous") != 1.0)
    )
    .count()
)

print(f"   Intervalles non horaires détectés : {gap_count:,}")

if gap_count > 0:
    print(
        "   Attention : les lags Spark sont basés sur les lignes précédentes. "
        "Les lignes entourant les trous seront invalidées par les contrôles "
        "de continuité ajoutés plus bas."
    )

df = df.drop("_previous_datetime", "_hours_since_previous")


# =============================================================================
# 3. VARIABLES TEMPORELLES DE L'INSTANT D'ORIGINE
# =============================================================================

print("\n3. Variables temporelles de l'instant d'origine")

df = (
    df
    .withColumn("issue_datetime", F.col("datetime"))
    .withColumn(
        "target_datetime",
        F.expr(
            f"datetime + INTERVAL {FORECAST_HORIZON_HOURS} HOURS"
        )
    )
    .withColumn("issue_date", F.to_date("issue_datetime"))
    .withColumn("issue_year", F.year("issue_datetime"))
    .withColumn("issue_month", F.month("issue_datetime"))
    .withColumn("issue_day", F.dayofmonth("issue_datetime"))
    .withColumn("issue_hour", F.hour("issue_datetime"))
    .withColumn(
        "issue_day_of_week",
        F.pmod(F.dayofweek("issue_datetime") + F.lit(5), F.lit(7))
    )
    .withColumn(
        "issue_day_of_year",
        F.dayofyear("issue_datetime")
    )
    .withColumn(
        "issue_is_weekend",
        F.when(F.col("issue_day_of_week") >= 5, 1).otherwise(0)
    )
)

# Cycles relatifs à l'instant où la prédiction est produite.
df = (
    df
    .withColumn(
        "issue_hour_sin",
        F.sin(
            F.lit(2.0 * np.pi)
            * F.col("issue_hour")
            / F.lit(24.0)
        )
    )
    .withColumn(
        "issue_hour_cos",
        F.cos(
            F.lit(2.0 * np.pi)
            * F.col("issue_hour")
            / F.lit(24.0)
        )
    )
)


# =============================================================================
# 4. VARIABLES TEMPORELLES DE LA DATE CIBLE
# =============================================================================

print("\n4. Variables temporelles de l'heure à prévoir")

df = (
    df
    .withColumn("target_date", F.to_date("target_datetime"))
    .withColumn("target_year", F.year("target_datetime"))
    .withColumn("target_month", F.month("target_datetime"))
    .withColumn("target_day", F.dayofmonth("target_datetime"))
    .withColumn("target_hour", F.hour("target_datetime"))
    .withColumn(
        "target_day_of_week",
        F.pmod(F.dayofweek("target_datetime") + F.lit(5), F.lit(7))
    )
    .withColumn(
        "target_day_of_year",
        F.dayofyear("target_datetime")
    )
    .withColumn(
        "target_is_weekend",
        F.when(F.col("target_day_of_week") >= 5, 1).otherwise(0)
    )
)

# Variables cycliques alignées sur l'heure réellement prédite.
df = (
    df
    .withColumn(
        "target_hour_sin",
        F.sin(
            F.lit(2.0 * np.pi)
            * F.col("target_hour")
            / F.lit(24.0)
        )
    )
    .withColumn(
        "target_hour_cos",
        F.cos(
            F.lit(2.0 * np.pi)
            * F.col("target_hour")
            / F.lit(24.0)
        )
    )
    .withColumn(
        "target_day_of_week_sin",
        F.sin(
            F.lit(2.0 * np.pi)
            * F.col("target_day_of_week")
            / F.lit(7.0)
        )
    )
    .withColumn(
        "target_day_of_week_cos",
        F.cos(
            F.lit(2.0 * np.pi)
            * F.col("target_day_of_week")
            / F.lit(7.0)
        )
    )
    .withColumn(
        "target_month_sin",
        F.sin(
            F.lit(2.0 * np.pi)
            * (F.col("target_month") - F.lit(1))
            / F.lit(12.0)
        )
    )
    .withColumn(
        "target_month_cos",
        F.cos(
            F.lit(2.0 * np.pi)
            * (F.col("target_month") - F.lit(1))
            / F.lit(12.0)
        )
    )
    .withColumn(
        "target_day_of_year_sin",
        F.sin(
            F.lit(2.0 * np.pi)
            * (F.col("target_day_of_year") - F.lit(1))
            / F.lit(365.25)
        )
    )
    .withColumn(
        "target_day_of_year_cos",
        F.cos(
            F.lit(2.0 * np.pi)
            * (F.col("target_day_of_year") - F.lit(1))
            / F.lit(365.25)
        )
    )
)

# Périodes de la journée cible.
df = (
    df
    .withColumn(
        "target_time_morning",
        F.when(
            (F.col("target_hour") >= 6)
            & (F.col("target_hour") < 12),
            1
        ).otherwise(0)
    )
    .withColumn(
        "target_time_afternoon",
        F.when(
            (F.col("target_hour") >= 12)
            & (F.col("target_hour") < 18),
            1
        ).otherwise(0)
    )
    .withColumn(
        "target_time_evening",
        F.when(
            (F.col("target_hour") >= 18)
            & (F.col("target_hour") < 22),
            1
        ).otherwise(0)
    )
    .withColumn(
        "target_time_night",
        F.when(
            (F.col("target_hour") < 6)
            | (F.col("target_hour") >= 22),
            1
        ).otherwise(0)
    )
)


# =============================================================================
# 5. FENÊTRES SPARK
# =============================================================================

window_zone = Window.partitionBy("zone").orderBy("datetime")


# =============================================================================
# 6. LAGS DE DEMANDE
# =============================================================================

print("\n5. Lags de demande")

demand_lags = [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    144,
    168,
    336,
]

# Créer toutes les colonnes de lag en une seule passe
lag_cols = {}
for lag_hours in demand_lags:
    lag_cols[f"_demand_lag_raw_{lag_hours}h"] = F.lag("demand_mw", lag_hours).over(window_zone)
    lag_cols[f"_datetime_lag_{lag_hours}h"] = F.lag("datetime", lag_hours).over(window_zone)

df = df.withColumns(lag_cols)

# Continuité exacte des principaux lags.
# Si la série contient un trou, le lag est invalidé.
validated_lag_cols = {}
for lag_hours in demand_lags:
    validated_lag_cols[f"demand_lag_{lag_hours}h"] = F.when(
        (
            F.col("datetime").cast("long")
            - F.col(f"_datetime_lag_{lag_hours}h").cast("long")
        ) == lag_hours * 3600,
        F.col(f"_demand_lag_raw_{lag_hours}h")
    ).otherwise(F.lit(None).cast("double"))

df = df.withColumns(validated_lag_cols)

for lag_hours in demand_lags:
    print(f"   Créé : demand_lag_{lag_hours}h")

# Supprimer les colonnes temporaires
temp_cols_to_drop = (
    [f"_demand_lag_raw_{lag_hours}h" for lag_hours in demand_lags]
    + [f"_datetime_lag_{lag_hours}h" for lag_hours in demand_lags]
)
df = df.drop(*temp_cols_to_drop)


# =============================================================================
# 7. VARIATIONS ET RATIOS DE DEMANDE
# =============================================================================

print("\n6. Variations et ratios de demande")

df = (
    df
    .withColumn(
        "demand_change_1h",
        F.col("demand_mw") - F.col("demand_lag_1h")
    )
    .withColumn(
        "demand_change_24h",
        F.col("demand_mw") - F.col("demand_lag_24h")
    )
    .withColumn(
        "demand_change_168h",
        F.col("demand_mw") - F.col("demand_lag_168h")
    )
    .withColumn(
        "demand_pct_change_1h",
        F.when(
            F.abs(F.col("demand_lag_1h")) > 0,
            (
                F.col("demand_mw") - F.col("demand_lag_1h")
            ) / F.col("demand_lag_1h")
        )
    )
    .withColumn(
        "demand_pct_change_24h",
        F.when(
            F.abs(F.col("demand_lag_24h")) > 0,
            (
                F.col("demand_mw") - F.col("demand_lag_24h")
            ) / F.col("demand_lag_24h")
        )
    )
    .withColumn(
        "demand_ratio_24h_168h",
        F.when(
            F.abs(F.col("demand_lag_168h")) > 0,
            F.col("demand_lag_24h") / F.col("demand_lag_168h")
        )
    )
)


# =============================================================================
# 8. ROLLING FEATURES DE DEMANDE
# =============================================================================

print("\n7. Moyennes mobiles de demande")

rolling_demand_windows = [3, 6, 12, 24, 48, 72, 168]

# Créer un dictionnaire avec toutes les colonnes rolling demand
rolling_demand_cols = {}
for window_size in rolling_demand_windows:
    rolling_window = (
        Window.partitionBy("zone")
        .orderBy("datetime")
        .rowsBetween(-window_size, -1)
    )
    rolling_demand_cols[f"demand_rolling_mean_{window_size}h"] = F.avg("demand_mw").over(rolling_window)
    rolling_demand_cols[f"demand_rolling_min_{window_size}h"] = F.min("demand_mw").over(rolling_window)
    rolling_demand_cols[f"demand_rolling_max_{window_size}h"] = F.max("demand_mw").over(rolling_window)

df = df.withColumns(rolling_demand_cols)

for window_size in rolling_demand_windows:
    print(f"   Créé : rolling demand {window_size} h")

df = df.withColumns({
    f"demand_rolling_std_{window_size}h": F.stddev_samp("demand_mw").over(
        Window.partitionBy("zone")
        .orderBy("datetime")
        .rowsBetween(-window_size, -1)
    )
    for window_size in [6, 24, 72, 168]
})

# Position de la demande actuelle par rapport à sa moyenne récente.
df = (
    df
    .withColumn(
        "demand_vs_rolling_mean_24h",
        F.col("demand_mw")
        - F.col("demand_rolling_mean_24h")
    )
    .withColumn(
        "demand_vs_rolling_mean_168h",
        F.col("demand_mw")
        - F.col("demand_rolling_mean_168h")
    )
)


# =============================================================================
# 9. FEATURES MÉTÉO HISTORIQUES
# =============================================================================

print("\n8. Features météo historiques")

# Lags météo seulement disponibles au moment de l'émission.
weather_lags = [1, 3, 6, 12, 24, 48, 72, 168]

weather_lag_cols = {}
for lag_hours in weather_lags:
    weather_lag_cols[f"temperature_lag_{lag_hours}h"] = F.lag("temperature_c", lag_hours).over(window_zone)
    weather_lag_cols[f"humidity_lag_{lag_hours}h"] = F.lag("relative_humidity_pct", lag_hours).over(window_zone)

df = df.withColumns(weather_lag_cols)

# Tendances de température.
df = (
    df
    .withColumn(
        "temperature_change_1h",
        F.col("temperature_c") - F.col("temperature_lag_1h")
    )
    .withColumn(
        "temperature_change_6h",
        F.col("temperature_c") - F.col("temperature_lag_6h")
    )
    .withColumn(
        "temperature_change_24h",
        F.col("temperature_c") - F.col("temperature_lag_24h")
    )
)

# Moyennes météo historiques.
weather_rolling_cols = {}
for window_size in [3, 6, 12, 24, 48, 72, 168]:
    weather_window = (
        Window.partitionBy("zone")
        .orderBy("datetime")
        .rowsBetween(-window_size + 1, 0)
    )
    weather_rolling_cols[f"temperature_rolling_mean_{window_size}h"] = F.avg("temperature_c").over(weather_window)
    weather_rolling_cols[f"humidity_rolling_mean_{window_size}h"] = F.avg("relative_humidity_pct").over(weather_window)

df = df.withColumns(weather_rolling_cols)

weather_extended_cols = {}
for window_size in [24, 72, 168]:
    weather_window = (
        Window.partitionBy("zone")
        .orderBy("datetime")
        .rowsBetween(-window_size + 1, 0)
    )
    weather_extended_cols[f"temperature_rolling_min_{window_size}h"] = F.min("temperature_c").over(weather_window)
    weather_extended_cols[f"temperature_rolling_max_{window_size}h"] = F.max("temperature_c").over(weather_window)
    weather_extended_cols[f"temperature_rolling_std_{window_size}h"] = F.stddev_samp("temperature_c").over(weather_window)
    weather_extended_cols[f"precipitation_rolling_sum_{window_size}h"] = F.sum(
        F.coalesce(F.col("precipitation_mm"), F.lit(0.0))
    ).over(weather_window)

df = df.withColumns(weather_extended_cols)


# =============================================================================
# 10. MÉTÉO ALIGNÉE SUR LA DATE CIBLE
# =============================================================================

print("\n9. Météo alignée sur target_datetime")

# Pendant l'entraînement, lead(24) agit comme proxy de la météo connue pour
# la date cible. Idéalement, il faut remplacer cela par des forecasts météo
# archivés qui auraient réellement été disponibles à issue_datetime.
future_weather_mapping = {
    "temperature_c": "weather_target_temperature_c",
    "dew_point_c": "weather_target_dew_point_c",
    "relative_humidity_pct": "weather_target_relative_humidity_pct",
    "wind_speed_kmh": "weather_target_wind_speed_kmh",
    "precipitation_mm": "weather_target_precipitation_mm",
}

df = df.withColumns({
    target_column: F.lead(source_column, FORECAST_HORIZON_HOURS).over(window_zone)
    for source_column, target_column in future_weather_mapping.items()
})

# Timestamp réellement associé à la ligne obtenue par lead(24).
df = df.withColumn(
    "_lead_24_datetime",
    F.lead(
        "datetime",
        FORECAST_HORIZON_HOURS
    ).over(window_zone)
)

# Si 24 lignes ne correspondent pas exactement à 24 heures, la météo cible
# et la cible sont invalidées.
valid_target_alignment = (
    F.col("_lead_24_datetime").isNotNull()
    & (
        (
            F.col("_lead_24_datetime").cast("long")
            - F.col("datetime").cast("long")
        )
        == FORECAST_HORIZON_HOURS * 3600
    )
)

df = df.withColumns({
    target_column: F.when(
        valid_target_alignment,
        F.col(target_column)
    ).otherwise(F.lit(None).cast("double"))
    for target_column in future_weather_mapping.values()
})


# =============================================================================
# 11. FEATURES MÉTÉO DÉRIVÉES POUR LA DATE CIBLE
# =============================================================================

print("\n10. Features météo dérivées")

target_temperature = F.col("weather_target_temperature_c")
target_dew_point = F.col("weather_target_dew_point_c")
target_humidity = F.col("weather_target_relative_humidity_pct")
target_wind = F.col("weather_target_wind_speed_kmh")
target_precipitation = F.col("weather_target_precipitation_mm")

df = (
    df
    # Non-linéarité de la relation température-demande.
    .withColumn(
        "weather_target_temperature_sq",
        F.pow(target_temperature, 2)
    )
    .withColumn(
        "weather_target_temperature_cube",
        F.pow(target_temperature, 3)
    )

    # Degrés de chauffage et de climatisation.
    .withColumn(
        "weather_target_hdd18",
        F.greatest(
            F.lit(BASE_TEMPERATURE_C) - target_temperature,
            F.lit(0.0)
        )
    )
    .withColumn(
        "weather_target_cdd18",
        F.greatest(
            target_temperature - F.lit(BASE_TEMPERATURE_C),
            F.lit(0.0)
        )
    )
    .withColumn(
        "weather_target_hdd15_5",
        F.greatest(
            F.lit(SECONDARY_BASE_TEMPERATURE_C) - target_temperature,
            F.lit(0.0)
        )
    )
    .withColumn(
        "weather_target_cdd15_5",
        F.greatest(
            target_temperature - F.lit(SECONDARY_BASE_TEMPERATURE_C),
            F.lit(0.0)
        )
    )

    # Écart température-point de rosée.
    .withColumn(
        "weather_target_temp_dew_spread",
        target_temperature - target_dew_point
    )

    # Approximation simple de la température humide.
    .withColumn(
        "weather_target_wet_bulb_approx_c",
        (
            target_temperature
            * F.atan(
                F.lit(0.151977)
                * F.sqrt(target_humidity + F.lit(8.313659))
            )
            + F.atan(target_temperature + target_humidity)
            - F.atan(target_humidity - F.lit(1.676331))
            + F.lit(0.00391838)
            * F.pow(target_humidity, F.lit(1.5))
            * F.atan(
                F.lit(0.023101) * target_humidity
            )
            - F.lit(4.686035)
        )
    )

    # Indicateurs simples de conditions météo.
    .withColumn(
        "weather_target_is_precipitation",
        F.when(
            F.coalesce(target_precipitation, F.lit(0.0)) > 0.0,
            1
        ).otherwise(0)
    )
    .withColumn(
        "weather_target_is_high_humidity",
        F.when(target_humidity >= 80.0, 1).otherwise(0)
    )
    .withColumn(
        "weather_target_is_strong_wind",
        F.when(target_wind >= 30.0, 1).otherwise(0)
    )
)


# =============================================================================
# 12. INTERACTIONS MÉTÉO-CALENDRIER
# =============================================================================

print("\n11. Interactions météo-calendrier")

df = (
    df
    .withColumn(
        "weather_target_temperature_x_hour_sin",
        F.col("weather_target_temperature_c")
        * F.col("target_hour_sin")
    )
    .withColumn(
        "weather_target_temperature_x_hour_cos",
        F.col("weather_target_temperature_c")
        * F.col("target_hour_cos")
    )
    .withColumn(
        "weather_target_hdd18_x_hour_sin",
        F.col("weather_target_hdd18")
        * F.col("target_hour_sin")
    )
    .withColumn(
        "weather_target_hdd18_x_hour_cos",
        F.col("weather_target_hdd18")
        * F.col("target_hour_cos")
    )
    .withColumn(
        "weather_target_cdd18_x_hour_sin",
        F.col("weather_target_cdd18")
        * F.col("target_hour_sin")
    )
    .withColumn(
        "weather_target_cdd18_x_hour_cos",
        F.col("weather_target_cdd18")
        * F.col("target_hour_cos")
    )
    .withColumn(
        "weather_target_hdd18_x_weekend",
        F.col("weather_target_hdd18")
        * F.col("target_is_weekend")
    )
    .withColumn(
        "weather_target_cdd18_x_weekend",
        F.col("weather_target_cdd18")
        * F.col("target_is_weekend")
    )
    .withColumn(
        "weather_target_temperature_x_humidity",
        F.col("weather_target_temperature_c")
        * F.col("weather_target_relative_humidity_pct")
    )
)


# =============================================================================
# 13. TABLE DES JOURS FÉRIÉS ONTARIENS
# =============================================================================

print("\n12. Création des variables de jours fériés")

year_bounds = (
    df.agg(
        F.min("target_year").alias("min_year"),
        F.max("target_year").alias("max_year")
    )
    .collect()[0]
)

minimum_year = int(year_bounds["min_year"])
maximum_year = int(year_bounds["max_year"])

# Générer un calendrier comprenant des marges pour les jours avant/après.
calendar_start = f"{minimum_year - 1}-01-01"
calendar_end = f"{maximum_year + 1}-12-31"

calendar_df = (
    spark.range(1)
    .select(
        F.explode(
            F.sequence(
                F.to_date(F.lit(calendar_start)),
                F.to_date(F.lit(calendar_end)),
                F.expr("INTERVAL 1 DAY")
            )
        ).alias("calendar_date")
    )
    .withColumn("calendar_year", F.year("calendar_date"))
    .withColumn("calendar_month", F.month("calendar_date"))
    .withColumn("calendar_day", F.dayofmonth("calendar_date"))
    .withColumn(
        "calendar_day_of_week",
        F.pmod(F.dayofweek("calendar_date") + F.lit(5), F.lit(7))
    )
)

# Fonctions Spark pour identifier les fêtes mobiles.
# day_of_week : lundi=0, mardi=1, ..., dimanche=6.

calendar_df = (
    calendar_df
    .withColumn(
        "is_new_year",
        (
            (F.col("calendar_month") == 1)
            & (F.col("calendar_day") == 1)
        ).cast("int")
    )
    .withColumn(
        "is_family_day",
        (
            (F.col("calendar_month") == 2)
            & (F.col("calendar_day_of_week") == 0)
            & (F.col("calendar_day").between(15, 21))
        ).cast("int")
    )
    .withColumn(
        "is_victoria_day",
        (
            (F.col("calendar_month") == 5)
            & (F.col("calendar_day_of_week") == 0)
            & (F.col("calendar_day").between(18, 24))
        ).cast("int")
    )
    .withColumn(
        "is_canada_day",
        (
            (F.col("calendar_month") == 7)
            & (F.col("calendar_day") == 1)
        ).cast("int")
    )
    .withColumn(
        "is_civic_holiday",
        (
            (F.col("calendar_month") == 8)
            & (F.col("calendar_day_of_week") == 0)
            & (F.col("calendar_day").between(1, 7))
        ).cast("int")
    )
    .withColumn(
        "is_labour_day",
        (
            (F.col("calendar_month") == 9)
            & (F.col("calendar_day_of_week") == 0)
            & (F.col("calendar_day").between(1, 7))
        ).cast("int")
    )
    .withColumn(
        "is_thanksgiving",
        (
            (F.col("calendar_month") == 10)
            & (F.col("calendar_day_of_week") == 0)
            & (F.col("calendar_day").between(8, 14))
        ).cast("int")
    )
    .withColumn(
        "is_christmas",
        (
            (F.col("calendar_month") == 12)
            & (F.col("calendar_day") == 25)
        ).cast("int")
    )
    .withColumn(
        "is_boxing_day",
        (
            (F.col("calendar_month") == 12)
            & (F.col("calendar_day") == 26)
        ).cast("int")
    )
)

# Calcul de Pâques avec l'algorithme de Meeus dans des expressions Spark.
year_col = F.col("calendar_year")

a = F.pmod(year_col, F.lit(19))
b = F.floor(year_col / F.lit(100))
c = F.pmod(year_col, F.lit(100))
d = F.floor(b / F.lit(4))
e = F.pmod(b, F.lit(4))
f = F.floor((b + F.lit(8)) / F.lit(25))
g = F.floor((b - f + F.lit(1)) / F.lit(3))
h = F.pmod(
    F.lit(19) * a + b - d - g + F.lit(15),
    F.lit(30)
)
i = F.floor(c / F.lit(4))
k = F.pmod(c, F.lit(4))
l = F.pmod(
    F.lit(32)
    + F.lit(2) * e
    + F.lit(2) * i
    - h
    - k,
    F.lit(7)
)
m = F.floor(
    (
        a
        + F.lit(11) * h
        + F.lit(22) * l
    ) / F.lit(451)
)

easter_month = F.floor(
    (
        h
        + l
        - F.lit(7) * m
        + F.lit(114)
    ) / F.lit(31)
)

easter_day = (
    F.pmod(
        h
        + l
        - F.lit(7) * m
        + F.lit(114),
        F.lit(31)
    )
    + F.lit(1)
)

calendar_df = calendar_df.withColumn(
    "_easter_date",
    F.make_date(
        F.col("calendar_year"),
        easter_month.cast("int"),
        easter_day.cast("int")
    )
)

calendar_df = (
    calendar_df
    .withColumn(
        "is_good_friday",
        (
            F.col("calendar_date")
            == F.date_sub(F.col("_easter_date"), 2)
        ).cast("int")
    )
    .withColumn(
        "is_easter_monday",
        (
            F.col("calendar_date")
            == F.date_add(F.col("_easter_date"), 1)
        ).cast("int")
    )
)

holiday_columns = [
    "is_new_year",
    "is_family_day",
    "is_good_friday",
    "is_easter_monday",
    "is_victoria_day",
    "is_canada_day",
    "is_civic_holiday",
    "is_labour_day",
    "is_thanksgiving",
    "is_christmas",
    "is_boxing_day",
]

calendar_df = calendar_df.withColumn(
    "is_holiday",
    F.greatest(*[F.col(column) for column in holiday_columns])
)

# Récupérer le caractère férié du lendemain et de la veille.
calendar_window = Window.orderBy("calendar_date")

calendar_df = (
    calendar_df
    .withColumn(
        "is_day_before_holiday",
        F.coalesce(
            F.lead("is_holiday", 1).over(calendar_window),
            F.lit(0)
        )
    )
    .withColumn(
        "is_day_after_holiday",
        F.coalesce(
            F.lag("is_holiday", 1).over(calendar_window),
            F.lit(0)
        )
    )
    .select(
        F.col("calendar_date").alias("target_date"),
        *holiday_columns,
        "is_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
    )
)

# Jointure des jours fériés sur la date cible.
df = df.join(
    F.broadcast(calendar_df),
    on="target_date",
    how="left"
)

calendar_output_columns = (
    holiday_columns
    + [
        "is_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
    ]
)

df = df.withColumns({
    column: F.coalesce(F.col(column), F.lit(0)).cast("int")
    for column in calendar_output_columns
})


# =============================================================================
# 14. INTERACTIONS AVEC LES JOURS FÉRIÉS
# =============================================================================

print("\n13. Interactions avec les jours fériés")

df = (
    df
    .withColumn(
        "weather_target_hdd18_x_holiday",
        F.col("weather_target_hdd18")
        * F.col("is_holiday")
    )
    .withColumn(
        "weather_target_cdd18_x_holiday",
        F.col("weather_target_cdd18")
        * F.col("is_holiday")
    )
    .withColumn(
        "holiday_x_target_hour_sin",
        F.col("is_holiday")
        * F.col("target_hour_sin")
    )
    .withColumn(
        "holiday_x_target_hour_cos",
        F.col("is_holiday")
        * F.col("target_hour_cos")
    )
)


# =============================================================================
# 15. CRÉATION DE LA CIBLE
# =============================================================================

print("\n14. Création de la cible à 24 heures")

df = df.withColumn(
    "target_demand_mw",
    F.lead(
        "demand_mw",
        FORECAST_HORIZON_HOURS
    ).over(window_zone)
)

df = df.withColumn(
    "target_demand_mw",
    F.when(
        valid_target_alignment,
        F.col("target_demand_mw")
    ).otherwise(F.lit(None).cast("double"))
)

# Horizon explicite, utile si plusieurs horizons sont ajoutés plus tard.
df = df.withColumn(
    "forecast_horizon_hours",
    F.lit(FORECAST_HORIZON_HOURS).cast("int")
)

df = df.drop("_lead_24_datetime")


# =============================================================================
# 16. CONTRÔLES DE VALIDITÉ POUR L'ENTRAÎNEMENT
# =============================================================================

print("\n15. Contrôles de validité des lignes")

# Les lags 1h, 24h et 168h sont considérés comme essentiels.
essential_feature_columns = [
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_mean_24h",
    "weather_target_temperature_c",
    "weather_target_relative_humidity_pct",
]

# Créer la condition de validité des features en une seule expression
from functools import reduce
from operator import and_

valid_features_condition = reduce(
    and_,
    [F.col(column).isNotNull() for column in essential_feature_columns],
    F.lit(True)
)

df = (
    df
    .withColumn(
        "has_valid_features",
        valid_features_condition.cast("int")
    )
    .withColumn(
        "has_valid_target",
        F.col("target_demand_mw").isNotNull().cast("int")
    )
    .withColumn(
        "is_training_row",
        (
            valid_features_condition
            & F.col("target_demand_mw").isNotNull()
        ).cast("int")
    )
)


# =============================================================================
# 17. SÉLECTION DES COLONNES FINALES
# =============================================================================

print("\n16. Sélection des colonnes finales")

metadata_columns = [
    "zone",
    "issue_datetime",
    "target_datetime",
    "forecast_horizon_hours",
    "issue_date",
    "target_date",
    "issue_year",
    "issue_month",
    "issue_day",
    "issue_hour",
    "target_year",
    "target_month",
    "target_day",
    "target_hour",
]

temporal_feature_columns = [
    "issue_day_of_week",
    "issue_day_of_year",
    "issue_is_weekend",
    "issue_hour_sin",
    "issue_hour_cos",
    "target_day_of_week",
    "target_day_of_year",
    "target_is_weekend",
    "target_hour_sin",
    "target_hour_cos",
    "target_day_of_week_sin",
    "target_day_of_week_cos",
    "target_month_sin",
    "target_month_cos",
    "target_day_of_year_sin",
    "target_day_of_year_cos",
    "target_time_morning",
    "target_time_afternoon",
    "target_time_evening",
    "target_time_night",
]

current_weather_columns = [
    "temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
]

historical_weather_feature_columns = (
    [
        f"temperature_lag_{lag_hours}h"
        for lag_hours in weather_lags
    ]
    + [
        f"humidity_lag_{lag_hours}h"
        for lag_hours in weather_lags
    ]
    + [
        "temperature_change_1h",
        "temperature_change_6h",
        "temperature_change_24h",
    ]
    + [
        f"temperature_rolling_mean_{window_size}h"
        for window_size in [3, 6, 12, 24, 48, 72, 168]
    ]
    + [
        f"humidity_rolling_mean_{window_size}h"
        for window_size in [3, 6, 12, 24, 48, 72, 168]
    ]
    + [
        f"temperature_rolling_min_{window_size}h"
        for window_size in [24, 72, 168]
    ]
    + [
        f"temperature_rolling_max_{window_size}h"
        for window_size in [24, 72, 168]
    ]
    + [
        f"temperature_rolling_std_{window_size}h"
        for window_size in [24, 72, 168]
    ]
    + [
        f"precipitation_rolling_sum_{window_size}h"
        for window_size in [24, 72, 168]
    ]
)

target_weather_feature_columns = [
    "weather_target_temperature_c",
    "weather_target_dew_point_c",
    "weather_target_relative_humidity_pct",
    "weather_target_wind_speed_kmh",
    "weather_target_precipitation_mm",
    "weather_target_temperature_sq",
    "weather_target_temperature_cube",
    "weather_target_hdd18",
    "weather_target_cdd18",
    "weather_target_hdd15_5",
    "weather_target_cdd15_5",
    "weather_target_temp_dew_spread",
    "weather_target_wet_bulb_approx_c",
    "weather_target_is_precipitation",
    "weather_target_is_high_humidity",
    "weather_target_is_strong_wind",
]

demand_lag_feature_columns = [
    f"demand_lag_{lag_hours}h"
    for lag_hours in demand_lags
]

demand_change_feature_columns = [
    "demand_change_1h",
    "demand_change_24h",
    "demand_change_168h",
    "demand_pct_change_1h",
    "demand_pct_change_24h",
    "demand_ratio_24h_168h",
]

demand_rolling_feature_columns = (
    [
        f"demand_rolling_mean_{window_size}h"
        for window_size in rolling_demand_windows
    ]
    + [
        f"demand_rolling_min_{window_size}h"
        for window_size in rolling_demand_windows
    ]
    + [
        f"demand_rolling_max_{window_size}h"
        for window_size in rolling_demand_windows
    ]
    + [
        f"demand_rolling_std_{window_size}h"
        for window_size in [6, 24, 72, 168]
    ]
    + [
        "demand_vs_rolling_mean_24h",
        "demand_vs_rolling_mean_168h",
    ]
)

interaction_feature_columns = [
    "weather_target_temperature_x_hour_sin",
    "weather_target_temperature_x_hour_cos",
    "weather_target_hdd18_x_hour_sin",
    "weather_target_hdd18_x_hour_cos",
    "weather_target_cdd18_x_hour_sin",
    "weather_target_cdd18_x_hour_cos",
    "weather_target_hdd18_x_weekend",
    "weather_target_cdd18_x_weekend",
    "weather_target_temperature_x_humidity",
    "weather_target_hdd18_x_holiday",
    "weather_target_cdd18_x_holiday",
    "holiday_x_target_hour_sin",
    "holiday_x_target_hour_cos",
]

calendar_feature_columns = [
    *holiday_columns,
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
]

technical_columns = [
    "has_valid_features",
    "has_valid_target",
    "is_training_row",
]

reference_columns = [
    "demand_mw",
]

target_columns = [
    "target_demand_mw",
]

feature_columns = (
    temporal_feature_columns
    + current_weather_columns
    + historical_weather_feature_columns
    + target_weather_feature_columns
    + demand_lag_feature_columns
    + demand_change_feature_columns
    + demand_rolling_feature_columns
    + interaction_feature_columns
    + calendar_feature_columns
)

final_columns = (
    metadata_columns
    + feature_columns
    + technical_columns
    + reference_columns
    + target_columns
)

# Vérification programmatique des colonnes.
missing_final_columns = [
    column
    for column in final_columns
    if column not in df.columns
]

if missing_final_columns:
    raise ValueError(
        "Colonnes finales manquantes : "
        + ", ".join(missing_final_columns)
    )

# Retirer les doublons éventuels dans la liste de colonnes tout en conservant
# l'ordre.
final_columns = list(dict.fromkeys(final_columns))

df_gold_final = df.select(*final_columns)


# =============================================================================
# 18. ÉCRITURE DE LA TABLE DELTA
# =============================================================================

print("\n17. Écriture de la table Gold")

(
    df_gold_final.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.columnMapping.mode", "name")
    .partitionBy("target_year", "target_month", "zone")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"   Table créée : {OUTPUT_TABLE}")


# =============================================================================
# 19. PROPRIÉTÉS ET COMMENTAIRES DE TABLE
# =============================================================================

spark.sql(
    f"""
    COMMENT ON TABLE {OUTPUT_TABLE} IS
    'Features ML par zone pour la prévision de demande électrique à 24 heures (modèle horizon_24h)'
    """
)


# =============================================================================
# 20. VALIDATION FINALE
# =============================================================================

print("\n18. Validation finale")

df_check = spark.table(OUTPUT_TABLE)

stats = (
    df_check
    .agg(
        F.count("*").alias("total_rows"),
        F.countDistinct("zone").alias("zones"),
        F.min("issue_datetime").alias("min_issue_datetime"),
        F.max("issue_datetime").alias("max_issue_datetime"),
        F.min("target_datetime").alias("min_target_datetime"),
        F.max("target_datetime").alias("max_target_datetime"),
        F.sum("has_valid_features").alias("valid_feature_rows"),
        F.sum("has_valid_target").alias("valid_target_rows"),
        F.sum("is_training_row").alias("valid_training_rows"),
    )
    .collect()[0]
)

print(f"   Total lignes              : {stats['total_rows']:,}")
print(f"   Nombre de zones           : {stats['zones']:,}")
print(
    "   Période d'émission        : "
    f"{stats['min_issue_datetime']} -> {stats['max_issue_datetime']}"
)
print(
    "   Période cible             : "
    f"{stats['min_target_datetime']} -> {stats['max_target_datetime']}"
)
print(
    "   Lignes avec features      : "
    f"{stats['valid_feature_rows']:,}"
)
print(
    "   Lignes avec cible         : "
    f"{stats['valid_target_rows']:,}"
)
print(
    "   Lignes valides training   : "
    f"{stats['valid_training_rows']:,}"
)

print(f"   Nombre total de features  : {len(feature_columns):,}")


# =============================================================================
# 21. VALIDATION PAR ZONE
# =============================================================================

print("\n19. Résumé par zone")

(
    df_check
    .groupBy("zone")
    .agg(
        F.count("*").alias("total_observations"),
        F.sum("is_training_row").alias("training_observations"),
        F.round(F.avg("demand_mw"), 2).alias("average_current_demand_mw"),
        F.round(
            F.avg("target_demand_mw"),
            2
        ).alias("average_target_demand_mw"),
        F.round(
            F.avg("weather_target_temperature_c"),
            2
        ).alias("average_target_temperature_c"),
        F.round(
            100.0
            * F.avg(
                F.when(
                    F.col("weather_target_temperature_c").isNull(),
                    1.0
                ).otherwise(0.0)
            ),
            3
        ).alias("missing_target_weather_pct"),
    )
    .orderBy("zone")
    .show(50, truncate=False)
)


# =============================================================================
# 22. VALIDATION DE LA CIBLE
# =============================================================================

print("\n20. Vérification de l'alignement de la cible")

alignment_check = (
    df_check
    .filter(F.col("target_demand_mw").isNotNull())
    .select(
        "zone",
        "issue_datetime",
        "target_datetime",
        "demand_mw",
        "target_demand_mw",
        "weather_target_temperature_c",
        "demand_lag_24h",
        "demand_lag_168h",
    )
    .orderBy("zone", "issue_datetime")
)

alignment_check.show(20, truncate=False)


# =============================================================================
# 23. CRÉATION OPTIONNELLE D'UNE VUE D'ENTRAÎNEMENT
# =============================================================================

TRAINING_VIEW = f"{CATALOG}.{SCHEMA}.ml_features_training_gold"

spark.sql(
    f"""
    CREATE OR REPLACE VIEW {TRAINING_VIEW} AS
    SELECT *
    FROM {OUTPUT_TABLE}
    WHERE is_training_row = 1
    """
)

print(f"\nVue d'entraînement créée : {TRAINING_VIEW}")


# =============================================================================
# 24. TERMINÉ
# =============================================================================

print("\n" + "=" * 80)
print("Gold Layer terminée avec succès")
print(f"Table complète       : {OUTPUT_TABLE}")
print(f"Vue pour entraînement: {TRAINING_VIEW}")
print(f"Features disponibles : {len(feature_columns)}")
print("=" * 80)