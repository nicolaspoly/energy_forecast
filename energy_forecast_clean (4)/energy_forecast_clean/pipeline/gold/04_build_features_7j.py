#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gold Layer — Construction des features multi-horizon pour le modèle 7 jours

Résumé
------
Ce script construit la table Gold dédiée au modèle direct multi-horizon de
prévision de la demande électrique en Ontario. Il produit une ligne par
combinaison `(zone, issue_datetime, forecast_horizon_hours)` pour tous les
horizons de H+1 à H+168.

Rôle dans le pipeline
---------------------
Ce fichier alimente la table Gold spécifique au modèle 7 jours :
`workspace.energy_forecast.ml_features_gold_7j`.

Cette table ne doit pas être confondue avec la Gold du modèle 24h
(`workspace.energy_forecast.ml_features_gold_24h`). Les deux tables ont des
contrats de données différents :

* la Gold 24h contient uniquement l’horizon H+24 ;
* la Gold 7j contient l’ensemble des horizons H+1 à H+168.

Source et destination
---------------------
Source :
    `workspace.energy_forecast.demand_weather_silver`

Destination :
    `workspace.energy_forecast.ml_features_gold_7j`

Grain métier
------------
Une ligne représente :
    `zone + issue_datetime + forecast_horizon_hours`

La cible associée est :
    la demande réelle observée à `target_datetime`, où
    `target_datetime = issue_datetime + forecast_horizon_hours heures`

Contenu fonctionnel
-------------------
Le script réalise les étapes suivantes :

1. Charge et valide la table Silver.
2. Nettoie les doublons `(zone, datetime)` avant calcul des features.
3. Calcule les variables temporelles à l’instant d’émission.
4. Construit les lags, variations et statistiques glissantes de demande.
5. Construit les lags et statistiques glissantes météo historiques.
6. Étend chaque instant d’émission sur 168 horizons.
7. Calcule `target_datetime` pour chaque horizon.
8. Ajoute les variables calendaires et cycliques de la date cible.
9. Joint la Silver sur `(zone, target_datetime)` pour récupérer :
   * la cible `target_demand_mw` ;
   * la météo observée à l’horizon cible, utilisée comme proxy
     d’entraînement.
10. Calcule des features dérivées météo et des interactions avec le temps
    et les jours fériés.
11. Marque les lignes techniquement valides pour l’entraînement.
12. Écrit la table Delta finale partitionnée.

Hypothèses importantes
----------------------
* Le fuseau horaire Spark est forcé à `America/Toronto` pour rester cohérent
  avec la logique métier Ontario.
* Les colonnes météo jointes à `target_datetime` correspondent à de la météo
  observée et servent uniquement de proxy d’entraînement. Pour une inférence
  réellement causale en production, elles devront être remplacées par des
  prévisions météo archivées disponibles à `issue_datetime`.
* Certaines lignes proches du début ou de la fin de l’historique peuvent ne
  pas disposer de tous les lags ou de la cible. Elles sont conservées mais
  marquées via `has_valid_features`, `has_valid_target` et `is_training_row`.

Colonnes techniques de qualité
------------------------------
Le script expose trois indicateurs utiles en aval :

* `has_valid_features` : les features essentielles sont présentes ;
* `has_valid_target` : la cible est disponible ;
* `is_training_row` : la ligne est exploitable pour l’entraînement.

Partitionnement et volumétrie
-----------------------------
La table finale est écrite en Delta avec partitionnement par :
    `target_year`, `target_month`, `zone`

Ce partitionnement est adapté aux lectures par période cible et par zone,
ainsi qu’aux contrôles de couverture multi-horizon.
"""

import yaml
import numpy as np
from functools import reduce
from operator import and_

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

MIN_HORIZON = 1
MAX_HORIZON = 168
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
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.ml_features_gold_7j"

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "America/Toronto")

print("=" * 80)
print("GOLD LAYER 7J — Feature Engineering multi-horizon pour ML")
print(f"Source      : {INPUT_TABLE}")
print(f"Destination : {OUTPUT_TABLE}")
print(f"Horizons    : H+{MIN_HORIZON} à H+{MAX_HORIZON}")
print("=" * 80)


# =============================================================================
# 1. CHARGEMENT ET VALIDATION DE LA TABLE SILVER
# =============================================================================

print("\n1. Chargement de la table Silver")

df_silver_raw = spark.table(INPUT_TABLE)

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df_silver_raw.columns
]

if missing_columns:
    raise ValueError(
        "Colonnes obligatoires absentes de la table Silver : "
        + ", ".join(missing_columns)
    )

df = (
    df_silver_raw
    .select(
        "*",
        F.col("datetime").cast("timestamp").alias("_validated_datetime")
    )
    .drop("datetime")
    .withColumnRenamed("_validated_datetime", "datetime")
)

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
    print("   Attention : les doublons sont agrégés avant le calcul des features.")
    aggregation_expressions = []
    for column in df.columns:
        if column not in ["zone", "datetime"]:
            aggregation_expressions.append(
                F.first(F.col(column), ignorenulls=True).alias(column)
            )
    df = df.groupBy("zone", "datetime").agg(*aggregation_expressions)

# Conserver une référence propre pour la jointure cible (avant ajout des features issue).
df_silver_clean = df

# Fenêtre pour les lags / rolling au niveau issue_datetime.
window_zone = Window.partitionBy("zone").orderBy("datetime")


# =============================================================================
# 3. VARIABLES TEMPORELLES DE L’INSTANT D’ÉMISSION
# =============================================================================

print("\n3. Variables temporelles de l’instant d’émission")

df = (
    df
    .withColumn("issue_datetime", F.col("datetime"))
    .withColumn("issue_date", F.to_date("datetime"))
    .withColumn("issue_year", F.year("datetime"))
    .withColumn("issue_month", F.month("datetime"))
    .withColumn("issue_day", F.dayofmonth("datetime"))
    .withColumn("issue_hour", F.hour("datetime"))
    .withColumn(
        "issue_day_of_week",
        F.pmod(F.dayofweek("datetime") + F.lit(5), F.lit(7))
    )
    .withColumn("issue_day_of_year", F.dayofyear("datetime"))
    .withColumn(
        "issue_is_weekend",
        F.when(F.col("issue_day_of_week") >= 5, 1).otherwise(0)
    )
    .withColumn(
        "issue_hour_sin",
        F.sin(F.lit(2.0 * np.pi) * F.col("issue_hour") / F.lit(24.0))
    )
    .withColumn(
        "issue_hour_cos",
        F.cos(F.lit(2.0 * np.pi) * F.col("issue_hour") / F.lit(24.0))
    )
)


# =============================================================================
# 4. LAGS DE DEMANDE
# =============================================================================

print("\n4. Lags de demande")

demand_lags = [1, 2, 3, 6, 12, 24, 48, 72, 144, 168, 336]

lag_cols = {}
for lag_hours in demand_lags:
    lag_cols[f"_demand_lag_raw_{lag_hours}h"] = F.lag("demand_mw", lag_hours).over(window_zone)
    lag_cols[f"_datetime_lag_{lag_hours}h"] = F.lag("datetime", lag_hours).over(window_zone)

df = df.withColumns(lag_cols)

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

temp_cols = (
    [f"_demand_lag_raw_{h}h" for h in demand_lags]
    + [f"_datetime_lag_{h}h" for h in demand_lags]
)
df = df.drop(*temp_cols)

for lag_hours in demand_lags:
    print(f"   Créé : demand_lag_{lag_hours}h")


# =============================================================================
# 5. VARIATIONS ET RATIOS DE DEMANDE
# =============================================================================

print("\n5. Variations et ratios de demande")

df = (
    df
    .withColumn("demand_change_1h", F.col("demand_mw") - F.col("demand_lag_1h"))
    .withColumn("demand_change_24h", F.col("demand_mw") - F.col("demand_lag_24h"))
    .withColumn("demand_change_168h", F.col("demand_mw") - F.col("demand_lag_168h"))
    .withColumn(
        "demand_pct_change_1h",
        F.when(
            F.abs(F.col("demand_lag_1h")) > 0,
            (F.col("demand_mw") - F.col("demand_lag_1h")) / F.col("demand_lag_1h")
        )
    )
    .withColumn(
        "demand_pct_change_24h",
        F.when(
            F.abs(F.col("demand_lag_24h")) > 0,
            (F.col("demand_mw") - F.col("demand_lag_24h")) / F.col("demand_lag_24h")
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
# 6. ROLLING FEATURES DE DEMANDE
# =============================================================================

print("\n6. Moyennes mobiles de demande")

rolling_demand_windows = [3, 6, 12, 24, 48, 72, 168]

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

df = df.withColumns({
    f"demand_rolling_std_{w}h": F.stddev_samp("demand_mw").over(
        Window.partitionBy("zone").orderBy("datetime").rowsBetween(-w, -1)
    )
    for w in [6, 24, 72, 168]
})

df = (
    df
    .withColumn(
        "demand_vs_rolling_mean_24h",
        F.col("demand_mw") - F.col("demand_rolling_mean_24h")
    )
    .withColumn(
        "demand_vs_rolling_mean_168h",
        F.col("demand_mw") - F.col("demand_rolling_mean_168h")
    )
)

for window_size in rolling_demand_windows:
    print(f"   Créé : rolling demand {window_size} h")


# =============================================================================
# 7. FEATURES MÉTÉO HISTORIQUES
# =============================================================================

print("\n7. Features météo historiques")

weather_lags = [1, 3, 6, 12, 24, 48, 72, 168]

weather_lag_cols = {}
for lag_hours in weather_lags:
    weather_lag_cols[f"temperature_lag_{lag_hours}h"] = F.lag("temperature_c", lag_hours).over(window_zone)
    weather_lag_cols[f"humidity_lag_{lag_hours}h"] = F.lag("relative_humidity_pct", lag_hours).over(window_zone)

df = df.withColumns(weather_lag_cols)

df = (
    df
    .withColumn("temperature_change_1h", F.col("temperature_c") - F.col("temperature_lag_1h"))
    .withColumn("temperature_change_6h", F.col("temperature_c") - F.col("temperature_lag_6h"))
    .withColumn("temperature_change_24h", F.col("temperature_c") - F.col("temperature_lag_24h"))
)

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

for lag_hours in weather_lags:
    print(f"   Créé : lags météo {lag_hours}h")

df_issue = df


# =============================================================================
# 8. EXPANSION MULTI-HORIZON (H+1 à H+168)
# Spark broadcast le côté horizon (168 lignes) automatiquement.
# =============================================================================

print("\n8. Expansion multi-horizon")

horizons = spark.range(MIN_HORIZON, MAX_HORIZON + 1).select(
    F.col("id").cast("int").alias("forecast_horizon_hours")
)

df_expanded = df_issue.crossJoin(horizons)

df_expanded = df_expanded.withColumn(
    "target_datetime",
    (
        F.col("issue_datetime").cast("long")
        + F.col("forecast_horizon_hours") * F.lit(3600)
    ).cast("timestamp")
)

print(f"   Nombre d’horizons : {MIN_HORIZON} à {MAX_HORIZON}")


# =============================================================================
# 9. VARIABLES TEMPORELLES DE LA DATE CIBLE
# =============================================================================

print("\n9. Variables temporelles de l’heure à prévoir")

df_expanded = (
    df_expanded
    .withColumn("target_date", F.to_date("target_datetime"))
    .withColumn("target_year", F.year("target_datetime"))
    .withColumn("target_month", F.month("target_datetime"))
    .withColumn("target_day", F.dayofmonth("target_datetime"))
    .withColumn("target_hour", F.hour("target_datetime"))
    .withColumn(
        "target_day_of_week",
        F.pmod(F.dayofweek("target_datetime") + F.lit(5), F.lit(7))
    )
    .withColumn("target_day_of_year", F.dayofyear("target_datetime"))
    .withColumn(
        "target_is_weekend",
        F.when(F.col("target_day_of_week") >= 5, 1).otherwise(0)
    )
    .withColumn(
        "target_hour_sin",
        F.sin(F.lit(2.0 * np.pi) * F.col("target_hour") / F.lit(24.0))
    )
    .withColumn(
        "target_hour_cos",
        F.cos(F.lit(2.0 * np.pi) * F.col("target_hour") / F.lit(24.0))
    )
    .withColumn(
        "target_day_of_week_sin",
        F.sin(F.lit(2.0 * np.pi) * F.col("target_day_of_week") / F.lit(7.0))
    )
    .withColumn(
        "target_day_of_week_cos",
        F.cos(F.lit(2.0 * np.pi) * F.col("target_day_of_week") / F.lit(7.0))
    )
    .withColumn(
        "target_month_sin",
        F.sin(F.lit(2.0 * np.pi) * (F.col("target_month") - F.lit(1)) / F.lit(12.0))
    )
    .withColumn(
        "target_month_cos",
        F.cos(F.lit(2.0 * np.pi) * (F.col("target_month") - F.lit(1)) / F.lit(12.0))
    )
    .withColumn(
        "target_day_of_year_sin",
        F.sin(F.lit(2.0 * np.pi) * (F.col("target_day_of_year") - F.lit(1)) / F.lit(365.25))
    )
    .withColumn(
        "target_day_of_year_cos",
        F.cos(F.lit(2.0 * np.pi) * (F.col("target_day_of_year") - F.lit(1)) / F.lit(365.25))
    )
    .withColumn(
        "target_time_morning",
        F.when((F.col("target_hour") >= 6) & (F.col("target_hour") < 12), 1).otherwise(0)
    )
    .withColumn(
        "target_time_afternoon",
        F.when((F.col("target_hour") >= 12) & (F.col("target_hour") < 18), 1).otherwise(0)
    )
    .withColumn(
        "target_time_evening",
        F.when((F.col("target_hour") >= 18) & (F.col("target_hour") < 22), 1).otherwise(0)
    )
    .withColumn(
        "target_time_night",
        F.when((F.col("target_hour") < 6) | (F.col("target_hour") >= 22), 1).otherwise(0)
    )
)


# =============================================================================
# 10. FEATURES SPÉCIFIQUES À L’HORIZON
# =============================================================================

print("\n10. Features d’horizon (7 jours)")

df_expanded = (
    df_expanded
    # Jour dans la semaine de prévision (1 = day-ahead, ..., 7 = seven-days-ahead).
    .withColumn(
        "forecast_day",
        ((F.col("forecast_horizon_hours") - 1) / F.lit(24)).cast("int") + 1
    )
    # Heure dans le jour de prévision (1–24).
    .withColumn(
        "forecast_hour_in_day",
        F.pmod(F.col("forecast_horizon_hours") - 1, F.lit(24)) + 1
    )
    # Cycles dans la journée (périodicité 24 h).
    .withColumn(
        "forecast_horizon_sin_24h",
        F.sin(F.lit(2.0 * np.pi) * F.col("forecast_horizon_hours") / F.lit(24.0))
    )
    .withColumn(
        "forecast_horizon_cos_24h",
        F.cos(F.lit(2.0 * np.pi) * F.col("forecast_horizon_hours") / F.lit(24.0))
    )
    # Cycles sur la semaine (périodicité 168 h).
    .withColumn(
        "forecast_horizon_sin_168h",
        F.sin(F.lit(2.0 * np.pi) * F.col("forecast_horizon_hours") / F.lit(168.0))
    )
    .withColumn(
        "forecast_horizon_cos_168h",
        F.cos(F.lit(2.0 * np.pi) * F.col("forecast_horizon_hours") / F.lit(168.0))
    )
)


# =============================================================================
# 11. JOINTURE SILVER POUR LA DEMANDE ET LA MÉTÉO CIBLES
# La météo observée à target_datetime est utilisée comme proxy d’entraînement.
# En production, remplacer par des prévisions météo archivées.
# =============================================================================

print("\n11. Jointure Silver pour la demande et la météo cibles")

df_silver_target = (
    df_silver_clean
    .select(
        "zone",
        F.col("datetime").alias("target_datetime"),
        F.col("demand_mw").alias("target_demand_mw"),
        F.col("temperature_c").alias("weather_target_temperature_c"),
        F.col("dew_point_c").alias("weather_target_dew_point_c"),
        F.col("relative_humidity_pct").alias("weather_target_relative_humidity_pct"),
        F.col("wind_speed_kmh").alias("weather_target_wind_speed_kmh"),
        F.col("precipitation_mm").alias("weather_target_precipitation_mm"),
    )
)

df_expanded = df_expanded.join(
    df_silver_target,
    on=["zone", "target_datetime"],
    how="left"
)


# =============================================================================
# 12. FEATURES MÉTÉO DÉRIVÉES POUR LA DATE CIBLE
# =============================================================================

print("\n12. Features météo dérivées pour la date cible")

target_temperature = F.col("weather_target_temperature_c")
target_dew_point = F.col("weather_target_dew_point_c")
target_humidity = F.col("weather_target_relative_humidity_pct")
target_wind = F.col("weather_target_wind_speed_kmh")
target_precipitation = F.col("weather_target_precipitation_mm")

df_expanded = (
    df_expanded
    .withColumn("weather_target_temperature_sq", F.pow(target_temperature, 2))
    .withColumn("weather_target_temperature_cube", F.pow(target_temperature, 3))
    .withColumn(
        "weather_target_hdd18",
        F.greatest(F.lit(BASE_TEMPERATURE_C) - target_temperature, F.lit(0.0))
    )
    .withColumn(
        "weather_target_cdd18",
        F.greatest(target_temperature - F.lit(BASE_TEMPERATURE_C), F.lit(0.0))
    )
    .withColumn(
        "weather_target_hdd15_5",
        F.greatest(F.lit(SECONDARY_BASE_TEMPERATURE_C) - target_temperature, F.lit(0.0))
    )
    .withColumn(
        "weather_target_cdd15_5",
        F.greatest(target_temperature - F.lit(SECONDARY_BASE_TEMPERATURE_C), F.lit(0.0))
    )
    .withColumn(
        "weather_target_temp_dew_spread",
        target_temperature - target_dew_point
    )
    .withColumn(
        "weather_target_wet_bulb_approx_c",
        (
            target_temperature
            * F.atan(F.lit(0.151977) * F.sqrt(target_humidity + F.lit(8.313659)))
            + F.atan(target_temperature + target_humidity)
            - F.atan(target_humidity - F.lit(1.676331))
            + F.lit(0.00391838)
            * F.pow(target_humidity, F.lit(1.5))
            * F.atan(F.lit(0.023101) * target_humidity)
            - F.lit(4.686035)
        )
    )
    .withColumn(
        "weather_target_is_precipitation",
        F.when(F.coalesce(target_precipitation, F.lit(0.0)) > 0.0, 1).otherwise(0)
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
# 13. INTERACTIONS MÉTÉO-CALENDRIER
# =============================================================================

print("\n13. Interactions météo-calendrier")

df_expanded = (
    df_expanded
    .withColumn(
        "weather_target_temperature_x_hour_sin",
        F.col("weather_target_temperature_c") * F.col("target_hour_sin")
    )
    .withColumn(
        "weather_target_temperature_x_hour_cos",
        F.col("weather_target_temperature_c") * F.col("target_hour_cos")
    )
    .withColumn(
        "weather_target_hdd18_x_hour_sin",
        F.col("weather_target_hdd18") * F.col("target_hour_sin")
    )
    .withColumn(
        "weather_target_hdd18_x_hour_cos",
        F.col("weather_target_hdd18") * F.col("target_hour_cos")
    )
    .withColumn(
        "weather_target_cdd18_x_hour_sin",
        F.col("weather_target_cdd18") * F.col("target_hour_sin")
    )
    .withColumn(
        "weather_target_cdd18_x_hour_cos",
        F.col("weather_target_cdd18") * F.col("target_hour_cos")
    )
    .withColumn(
        "weather_target_hdd18_x_weekend",
        F.col("weather_target_hdd18") * F.col("target_is_weekend")
    )
    .withColumn(
        "weather_target_cdd18_x_weekend",
        F.col("weather_target_cdd18") * F.col("target_is_weekend")
    )
    .withColumn(
        "weather_target_temperature_x_humidity",
        F.col("weather_target_temperature_c") * F.col("weather_target_relative_humidity_pct")
    )
)


# =============================================================================
# 14. TABLE DES JOURS FÉRIÉS ONTARIENS
# =============================================================================

print("\n14. Création des variables de jours fériés")

year_bounds = (
    df_expanded.agg(
        F.min("target_year").alias("min_year"),
        F.max("target_year").alias("max_year")
    )
    .collect()[0]
)

minimum_year = int(year_bounds["min_year"])
maximum_year = int(year_bounds["max_year"])

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

calendar_df = (
    calendar_df
    .withColumn("is_new_year", ((F.col("calendar_month") == 1) & (F.col("calendar_day") == 1)).cast("int"))
    .withColumn("is_family_day", ((F.col("calendar_month") == 2) & (F.col("calendar_day_of_week") == 0) & (F.col("calendar_day").between(15, 21))).cast("int"))
    .withColumn("is_victoria_day", ((F.col("calendar_month") == 5) & (F.col("calendar_day_of_week") == 0) & (F.col("calendar_day").between(18, 24))).cast("int"))
    .withColumn("is_canada_day", ((F.col("calendar_month") == 7) & (F.col("calendar_day") == 1)).cast("int"))
    .withColumn("is_civic_holiday", ((F.col("calendar_month") == 8) & (F.col("calendar_day_of_week") == 0) & (F.col("calendar_day").between(1, 7))).cast("int"))
    .withColumn("is_labour_day", ((F.col("calendar_month") == 9) & (F.col("calendar_day_of_week") == 0) & (F.col("calendar_day").between(1, 7))).cast("int"))
    .withColumn("is_thanksgiving", ((F.col("calendar_month") == 10) & (F.col("calendar_day_of_week") == 0) & (F.col("calendar_day").between(8, 14))).cast("int"))
    .withColumn("is_christmas", ((F.col("calendar_month") == 12) & (F.col("calendar_day") == 25)).cast("int"))
    .withColumn("is_boxing_day", ((F.col("calendar_month") == 12) & (F.col("calendar_day") == 26)).cast("int"))
)

# Calcul de Pâques avec l’algorithme de Meeus (native Spark).
year_col = F.col("calendar_year")
a = F.pmod(year_col, F.lit(19))
b = F.floor(year_col / F.lit(100))
c = F.pmod(year_col, F.lit(100))
d = F.floor(b / F.lit(4))
e = F.pmod(b, F.lit(4))
f = F.floor((b + F.lit(8)) / F.lit(25))
g = F.floor((b - f + F.lit(1)) / F.lit(3))
h = F.pmod(F.lit(19) * a + b - d - g + F.lit(15), F.lit(30))
i = F.floor(c / F.lit(4))
k = F.pmod(c, F.lit(4))
l = F.pmod(F.lit(32) + F.lit(2) * e + F.lit(2) * i - h - k, F.lit(7))
m = F.floor((a + F.lit(11) * h + F.lit(22) * l) / F.lit(451))
easter_month = F.floor((h + l - F.lit(7) * m + F.lit(114)) / F.lit(31))
easter_day = F.pmod(h + l - F.lit(7) * m + F.lit(114), F.lit(31)) + F.lit(1)

calendar_df = calendar_df.withColumn(
    "_easter_date",
    F.make_date(F.col("calendar_year"), easter_month.cast("int"), easter_day.cast("int"))
)

calendar_df = (
    calendar_df
    .withColumn("is_good_friday", (F.col("calendar_date") == F.date_sub(F.col("_easter_date"), 2)).cast("int"))
    .withColumn("is_easter_monday", (F.col("calendar_date") == F.date_add(F.col("_easter_date"), 1)).cast("int"))
)

holiday_columns = [
    "is_new_year", "is_family_day", "is_good_friday", "is_easter_monday",
    "is_victoria_day", "is_canada_day", "is_civic_holiday", "is_labour_day",
    "is_thanksgiving", "is_christmas", "is_boxing_day",
]

calendar_df = calendar_df.withColumn(
    "is_holiday",
    F.greatest(*[F.col(c) for c in holiday_columns])
)

calendar_window = Window.orderBy("calendar_date")
calendar_df = (
    calendar_df
    .withColumn(
        "is_day_before_holiday",
        F.coalesce(F.lead("is_holiday", 1).over(calendar_window), F.lit(0))
    )
    .withColumn(
        "is_day_after_holiday",
        F.coalesce(F.lag("is_holiday", 1).over(calendar_window), F.lit(0))
    )
    .select(
        F.col("calendar_date").alias("target_date"),
        *holiday_columns,
        "is_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
    )
)

df_expanded = df_expanded.join(
    F.broadcast(calendar_df),
    on="target_date",
    how="left"
)

calendar_output_columns = (
    holiday_columns
    + ["is_holiday", "is_day_before_holiday", "is_day_after_holiday"]
)

df_expanded = df_expanded.withColumns({
    column: F.coalesce(F.col(column), F.lit(0)).cast("int")
    for column in calendar_output_columns
})


# =============================================================================
# 15. INTERACTIONS AVEC LES JOURS FÉRIÉS
# =============================================================================

print("\n15. Interactions avec les jours fériés")

df_expanded = (
    df_expanded
    .withColumn(
        "weather_target_hdd18_x_holiday",
        F.col("weather_target_hdd18") * F.col("is_holiday")
    )
    .withColumn(
        "weather_target_cdd18_x_holiday",
        F.col("weather_target_cdd18") * F.col("is_holiday")
    )
    .withColumn(
        "holiday_x_target_hour_sin",
        F.col("is_holiday") * F.col("target_hour_sin")
    )
    .withColumn(
        "holiday_x_target_hour_cos",
        F.col("is_holiday") * F.col("target_hour_cos")
    )
)


# =============================================================================
# 16. CONTRÔLES DE VALIDITÉ
# =============================================================================

print("\n16. Contrôles de validité des lignes")

# Les lags 1h, 24h et 168h sont considérés comme essentiels.
# La météo cible n’est pas exigée dans les features essentielles car elle
# peut être absente aux derniers horizons d’un historique incomplet.
essential_feature_columns = [
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_mean_24h",
]

valid_features_condition = reduce(
    and_,
    [F.col(column).isNotNull() for column in essential_feature_columns],
    F.lit(True)
)

df_expanded = (
    df_expanded
    .withColumn("has_valid_features", valid_features_condition.cast("int"))
    .withColumn("has_valid_target", F.col("target_demand_mw").isNotNull().cast("int"))
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

print("\n17. Sélection des colonnes finales")

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

horizon_feature_columns = [
    "forecast_day",
    "forecast_hour_in_day",
    "forecast_horizon_sin_24h",
    "forecast_horizon_cos_24h",
    "forecast_horizon_sin_168h",
    "forecast_horizon_cos_168h",
]

current_weather_columns = [
    "temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
]

historical_weather_feature_columns = (
    [f"temperature_lag_{h}h" for h in weather_lags]
    + [f"humidity_lag_{h}h" for h in weather_lags]
    + ["temperature_change_1h", "temperature_change_6h", "temperature_change_24h"]
    + [f"temperature_rolling_mean_{w}h" for w in [3, 6, 12, 24, 48, 72, 168]]
    + [f"humidity_rolling_mean_{w}h" for w in [3, 6, 12, 24, 48, 72, 168]]
    + [f"temperature_rolling_min_{w}h" for w in [24, 72, 168]]
    + [f"temperature_rolling_max_{w}h" for w in [24, 72, 168]]
    + [f"temperature_rolling_std_{w}h" for w in [24, 72, 168]]
    + [f"precipitation_rolling_sum_{w}h" for w in [24, 72, 168]]
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

demand_lag_feature_columns = [f"demand_lag_{h}h" for h in demand_lags]

demand_change_feature_columns = [
    "demand_change_1h",
    "demand_change_24h",
    "demand_change_168h",
    "demand_pct_change_1h",
    "demand_pct_change_24h",
    "demand_ratio_24h_168h",
]

demand_rolling_feature_columns = (
    [f"demand_rolling_mean_{w}h" for w in rolling_demand_windows]
    + [f"demand_rolling_min_{w}h" for w in rolling_demand_windows]
    + [f"demand_rolling_max_{w}h" for w in rolling_demand_windows]
    + [f"demand_rolling_std_{w}h" for w in [6, 24, 72, 168]]
    + ["demand_vs_rolling_mean_24h", "demand_vs_rolling_mean_168h"]
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

technical_columns = ["has_valid_features", "has_valid_target", "is_training_row"]
reference_columns = ["demand_mw"]
target_columns = ["target_demand_mw"]

feature_columns = (
    temporal_feature_columns
    + horizon_feature_columns
    + current_weather_columns
    + historical_weather_feature_columns
    + target_weather_feature_columns
    + demand_lag_feature_columns
    + demand_change_feature_columns
    + demand_rolling_feature_columns
    + interaction_feature_columns
    + calendar_feature_columns
)

final_columns = list(dict.fromkeys(
    metadata_columns
    + feature_columns
    + technical_columns
    + reference_columns
    + target_columns
))

missing_final_columns = [
    column
    for column in final_columns
    if column not in df_expanded.columns
]

if missing_final_columns:
    raise ValueError(
        "Colonnes finales manquantes : "
        + ", ".join(missing_final_columns)
    )

df_gold_final = df_expanded.select(*final_columns)


# =============================================================================
# 18. ÉCRITURE DE LA TABLE DELTA
# =============================================================================

print("\n18. Écriture de la table Gold 7j")

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
# 19. COMMENTAIRES DE TABLE
# =============================================================================

spark.sql(
    f"""
    COMMENT ON TABLE {OUTPUT_TABLE} IS
    'Features ML multi-horizon (H+1 à H+168) par zone pour la prévision
    de demande électrique à 7 jours (modèle horizon_7j)'
    """
)


# =============================================================================
# 20. VALIDATION FINALE
# =============================================================================

print("\n19. Validation finale")

df_check = spark.table(OUTPUT_TABLE)

stats = (
    df_check
    .agg(
        F.count("*").alias("total_rows"),
        F.countDistinct("zone").alias("zones"),
        F.countDistinct("forecast_horizon_hours").alias("horizons"),
        F.min("forecast_horizon_hours").alias("min_horizon"),
        F.max("forecast_horizon_hours").alias("max_horizon"),
        F.min("issue_datetime").alias("min_issue_datetime"),
        F.max("issue_datetime").alias("max_issue_datetime"),
        F.sum("has_valid_features").alias("valid_feature_rows"),
        F.sum("has_valid_target").alias("valid_target_rows"),
        F.sum("is_training_row").alias("valid_training_rows"),
    )
    .collect()[0]
)

print(f"   Total lignes          : {stats['total_rows']:,}")
print(f"   Zones                 : {stats['zones']:,}")
print(f"   Horizons distincts    : {stats['horizons']:,} ({stats['min_horizon']} à {stats['max_horizon']})")
print(f"   Période d’émission    : {stats['min_issue_datetime']} -> {stats['max_issue_datetime']}")
print(f"   Lignes avec features  : {stats['valid_feature_rows']:,}")
print(f"   Lignes avec cible     : {stats['valid_target_rows']:,}")
print(f"   Lignes valides train  : {stats['valid_training_rows']:,}")
print(f"   Features totales      : {len(feature_columns):,}")


# =============================================================================
# 21. RÉSUMÉ DE COUVERTURE PAR HORIZON
# =============================================================================

print("\n20. Couverture par horizon (5 premiers et 5 derniers)")

coverage = (
    df_check
    .filter(F.col("is_training_row") == 1)
    .groupBy("forecast_horizon_hours")
    .agg(
        F.count("*").alias("training_rows"),
        F.round(F.avg("target_demand_mw"), 2).alias("avg_target_demand_mw"),
    )
    .orderBy("forecast_horizon_hours")
)

print("  Horizons H+1 à H+5 :")
coverage.filter(F.col("forecast_horizon_hours") <= 5).show(truncate=False)

print("  Horizons H+164 à H+168 :")
coverage.filter(F.col("forecast_horizon_hours") >= 164).show(truncate=False)
