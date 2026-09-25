# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Entraînement LightGBM multi-horizon H+1 à H+168
# MAGIC
# MAGIC Ce notebook entraîne un modèle direct, non récursif, pour prévoir
# MAGIC les 168 prochaines heures de demande électrique par zone.
# MAGIC
# MAGIC Meilleures pratiques implémentées :
# MAGIC - **Modèle direct** (forecast_horizon_hours comme feature, pas de récursion)
# MAGIC - **Walk-forward backtesting** (5 folds de 30 jours)
# MAGIC - **Quantile forecasting** P10/P50/P90
# MAGIC - **Réconciliation hiérarchique** bottom-up (zones → Ontario total)
# MAGIC
# MAGIC Structure attendue dans la table Gold :
# MAGIC
# MAGIC - issue_datetime
# MAGIC - target_datetime
# MAGIC - forecast_horizon_hours
# MAGIC - zone
# MAGIC - target_demand_mw
# MAGIC - features de demande disponibles à issue_datetime
# MAGIC - features météo correspondant à target_datetime

# COMMAND ----------

# MAGIC %pip install lightgbm optuna -q

# COMMAND ----------

# Décommenter seulement si Databricks demande un redémarrage.
# dbutils.library.restartPython()

# COMMAND ----------

import os
import json
import shutil
import warnings

import yaml
import mlflow
import mlflow.sklearn
import lightgbm as lgb
import optuna

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

print(f"LightGBM : {lgb.__version__}")
print(f"MLflow   : {mlflow.__version__}")
print(f"Optuna   : {optuna.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# DBTITLE 1,Configuration
# Chemin du projet : surchargeable via ENERGY_FORECAST_PROJECT_ROOT.
PROJECT_ROOT = os.environ.get(
    "ENERGY_FORECAST_PROJECT_ROOT",
    "/Workspace/Users/n.jouglet23@gmail.com/energy_forecast",
)
CONFIG_PATH = f"{PROJECT_ROOT}/config/config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

CATALOG = config["catalog"]["name"]
SCHEMA = config["catalog"]["schema"]

GOLD_TABLE = f"{CATALOG}.{SCHEMA}.ml_features_gold_7j"

# NOTE (nettoyage 2026-08-29): utilisait auparavant `config["model"]`, partagé
# à l'identique avec le modèle 24h (06_train_model_24h.py) -> les deux
# entraînements écrivaient dans la même expérience MLflow. On utilise
# maintenant la config dédiée `models.horizon_7j`.
MODEL_CONFIG = config["models"]["horizon_7j"]
MLFLOW_EXPERIMENT = MODEL_CONFIG["mlflow"]["experiment_name"]
REGISTERED_MODEL_NAME = MODEL_CONFIG["mlflow"]["registry_model_name"]

mlflow.set_experiment(MLFLOW_EXPERIMENT)

RANDOM_STATE = 42

ISSUE_DATETIME_COLUMN = "issue_datetime"
TARGET_DATETIME_COLUMN = "target_datetime"
HORIZON_COLUMN = "forecast_horizon_hours"
TARGET_COLUMN = "target_demand_mw"
ZONE_COLUMN = "zone"

MIN_FORECAST_HORIZON = 1
MAX_FORECAST_HORIZON = 168
FORECAST_DAYS = 7

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

MAX_MISSING_RATE = 0.30
NEAR_CONSTANT_THRESHOLD = 0.999

FEATURE_COUNTS_TO_TEST = [
    60,
]

WAPE_TOLERANCE_PERCENTAGE_POINT = 0.05

# Quantile forecasting : P10 (optimiste), P50 (médiane), P90 (pessimiste).
QUANTILES = [0.1, 0.5, 0.9]

# Walk-forward backtesting : 5 folds de 30 jours avec fenêtre expansive.
N_WALK_FORWARD_FOLDS = 5
WALK_FORWARD_TEST_DAYS = 30

# Réconciliation hiérarchique : bottom-up (somme des prévisions zonales = total Ontario).
RECONCILIATION_METHOD = "bottom_up"

ARTIFACT_DIRECTORY = (
    "/tmp/lightgbm_energy_forecast_168h"
)

print("=" * 80)
print("ENTRAÎNEMENT LIGHTGBM MULTI-HORIZON")
print("=" * 80)
print(f"Table Gold        : {GOLD_TABLE}")
print(f"Expérience MLflow : {MLFLOW_EXPERIMENT}")
print(
    f"Horizons          : "
    f"H+{MIN_FORECAST_HORIZON} à "
    f"H+{MAX_FORECAST_HORIZON}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chargement de la table Gold

# COMMAND ----------

print("Chargement de la table Gold...")

spark_df = spark.table(GOLD_TABLE)

spark_row_count = spark_df.count()

print(f"Lignes Spark   : {spark_row_count:,}")
print(f"Colonnes Spark : {len(spark_df.columns):,}")

REQUIRED_COLUMNS = {
    ISSUE_DATETIME_COLUMN,
    TARGET_DATETIME_COLUMN,
    HORIZON_COLUMN,
    TARGET_COLUMN,
    ZONE_COLUMN,
}

missing_required_columns = sorted(
    REQUIRED_COLUMNS - set(spark_df.columns)
)

if missing_required_columns:
    raise ValueError(
        "La table Gold n'est pas compatible avec "
        "l'entraînement multi-horizon.\n"
        "Colonnes manquantes :\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_required_columns
        )
    )

spark_training_df = spark_df.filter(
    spark_df[TARGET_COLUMN].isNotNull()
    & spark_df[ISSUE_DATETIME_COLUMN].isNotNull()
    & spark_df[TARGET_DATETIME_COLUMN].isNotNull()
    & spark_df[ZONE_COLUMN].isNotNull()
    & spark_df[HORIZON_COLUMN].between(
        MIN_FORECAST_HORIZON,
        MAX_FORECAST_HORIZON,
    )
)

if "is_training_row" in spark_training_df.columns:
    spark_training_df = spark_training_df.filter(
        spark_training_df["is_training_row"] == 1
    )

training_row_count = spark_training_df.count()

if training_row_count == 0:
    raise ValueError(
        "Aucune ligne valide pour l'entraînement."
    )

print(
    f"Lignes admissibles : {training_row_count:,}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Conversion Pandas et contrôles

# COMMAND ----------

# DBTITLE 1,Sous-échantillonnage : 1 émission tous les 3 jours par zone
# Un modèle 7 jours émet une prévision par jour (standard opérationnel).
# Garder toutes les émissions horaires (~24/jour/zone) produirait ~105M lignes
# impossibles à tenir en mémoire pandas sur compute serverless CPU.
#
# Stratégie : 1 émission tous les 3 jours par zone (rotation sur tous les
# jours de semaine, couverture saisonnière conservée) => ~1.5M lignes.
#   - 1/jour  = 4.4M lignes  -> OOM sur serverless CPU
#   - 1/3j    = 1.5M lignes  -> ~3.7 GB de pic -> OK
from pyspark.sql import functions as F

_SAMPLING_STRIDE_DAYS = 5  # garder 1 jour sur 5 (~880K lignes, niveau 24h)

# Trouver la première issue_datetime de chaque (zone, issue_date),
# puis filtrer sur les jours sélectionnés selon le numéro de jour absolu.
_first_issue_per_day = (
    spark_training_df
    .groupBy(ZONE_COLUMN, "issue_date")
    .agg(F.min(ISSUE_DATETIME_COLUMN).alias(ISSUE_DATETIME_COLUMN))
    .withColumn(
        "_day_offset",
        F.datediff(
            F.col("issue_date"),
            F.lit("2020-01-01"),
        ).cast("int"),
    )
    .filter(F.col("_day_offset") % _SAMPLING_STRIDE_DAYS == 0)
    .select(ZONE_COLUMN, ISSUE_DATETIME_COLUMN)
)

spark_training_df = spark_training_df.join(
    _first_issue_per_day,
    on=[ZONE_COLUMN, ISSUE_DATETIME_COLUMN],
    how="inner",
)

downsampled_count = spark_training_df.count()
print(
    f"Sous-échantillonnage "
    f"(1 émission/{_SAMPLING_STRIDE_DAYS}j/zone, 168 horizons) : "
    f"{downsampled_count:,} lignes "
    f"(réduction {105_107_772 / downsampled_count:.1f}×)"
)

# COMMAND ----------

# DBTITLE 1,Cell 10
print("Conversion Spark vers Pandas...")

pd_data = spark_training_df.toPandas()

pd_data[ISSUE_DATETIME_COLUMN] = pd.to_datetime(
    pd_data[ISSUE_DATETIME_COLUMN],
    errors="coerce",
)

pd_data[TARGET_DATETIME_COLUMN] = pd.to_datetime(
    pd_data[TARGET_DATETIME_COLUMN],
    errors="coerce",
)

pd_data[HORIZON_COLUMN] = pd.to_numeric(
    pd_data[HORIZON_COLUMN],
    errors="coerce",
)

pd_data[TARGET_COLUMN] = pd.to_numeric(
    pd_data[TARGET_COLUMN],
    errors="coerce",
)

pd_data = pd_data.dropna(
    subset=[
        ISSUE_DATETIME_COLUMN,
        TARGET_DATETIME_COLUMN,
        HORIZON_COLUMN,
        TARGET_COLUMN,
        ZONE_COLUMN,
    ]
).copy()

pd_data[HORIZON_COLUMN] = (
    pd_data[HORIZON_COLUMN].astype("int16")
)

pd_data = pd_data[
    pd_data[HORIZON_COLUMN].between(
        MIN_FORECAST_HORIZON,
        MAX_FORECAST_HORIZON,
    )
].copy()

# Vérifier que target_datetime correspond réellement à l'horizon.
calculated_horizon = (
    (
        pd_data[TARGET_DATETIME_COLUMN]
        - pd_data[ISSUE_DATETIME_COLUMN]
    )
    / pd.Timedelta(hours=1)
)

invalid_horizon_mask = (
    ~np.isclose(
        calculated_horizon,
        pd_data[HORIZON_COLUMN],
        atol=1e-6,
    )
)

if invalid_horizon_mask.any():
    invalid_horizon_rows = pd_data.loc[
        invalid_horizon_mask,
        [
            ZONE_COLUMN,
            ISSUE_DATETIME_COLUMN,
            TARGET_DATETIME_COLUMN,
            HORIZON_COLUMN,
        ],
    ].copy()

    invalid_horizon_rows[
        "calculated_horizon"
    ] = calculated_horizon.loc[
        invalid_horizon_mask
    ]

    display(invalid_horizon_rows.head(100))

    print(
        f"AVERTISSEMENT : {invalid_horizon_mask.sum():,} lignes "
        "avec forecast_horizon_hours incompatible avec les dates "
        "(probablement DST) — supprimées."
    )
    pd_data = pd_data.loc[~invalid_horizon_mask].copy()

# Contrôle des doublons.
KEY_COLUMNS = [
    ISSUE_DATETIME_COLUMN,
    TARGET_DATETIME_COLUMN,
    HORIZON_COLUMN,
    ZONE_COLUMN,
]

duplicate_mask = pd_data.duplicated(
    subset=KEY_COLUMNS,
    keep=False,
)

if duplicate_mask.any():
    display(
        pd_data.loc[
            duplicate_mask,
            KEY_COLUMNS,
        ].head(100)
    )

    raise ValueError(
        "Des doublons ont été détectés pour la clé "
        "issue_datetime/target_datetime/horizon/zone."
    )

pd_data = pd_data.sort_values(
    [
        ISSUE_DATETIME_COLUMN,
        ZONE_COLUMN,
        HORIZON_COLUMN,
    ]
).reset_index(drop=True)

print(f"Dataset Pandas : {len(pd_data):,} lignes")
print(
    f"Issue datetime : "
    f"{pd_data[ISSUE_DATETIME_COLUMN].min()} à "
    f"{pd_data[ISSUE_DATETIME_COLUMN].max()}"
)
print(
    f"Target datetime: "
    f"{pd_data[TARGET_DATETIME_COLUMN].min()} à "
    f"{pd_data[TARGET_DATETIME_COLUMN].max()}"
)
print(
    f"Horizons       : "
    f"{pd_data[HORIZON_COLUMN].min()} à "
    f"{pd_data[HORIZON_COLUMN].max()}"
)
print(
    f"Zones          : "
    f"{sorted(pd_data[ZONE_COLUMN].unique())}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Contrôle de la couverture H+1 à H+168

# COMMAND ----------

coverage = (
    pd_data
    .groupby(
        [
            ISSUE_DATETIME_COLUMN,
            ZONE_COLUMN,
        ],
        observed=True,
    )
    .agg(
        horizon_count=(
            HORIZON_COLUMN,
            "nunique",
        ),
        minimum_horizon=(
            HORIZON_COLUMN,
            "min",
        ),
        maximum_horizon=(
            HORIZON_COLUMN,
            "max",
        ),
    )
    .reset_index()
)

complete_coverage_mask = (
    coverage["horizon_count"].eq(168)
    & coverage["minimum_horizon"].eq(1)
    & coverage["maximum_horizon"].eq(168)
)

coverage_rate = float(
    complete_coverage_mask.mean()
)

print(
    f"Couverture complète H+1 à H+168 : "
    f"{coverage_rate:.2%}"
)

if not complete_coverage_mask.all():
    incomplete_coverage = coverage.loc[
        ~complete_coverage_mask
    ]

    print(
        f"Couples émission/zone incomplets : "
        f"{len(incomplete_coverage):,}"
    )

    display(incomplete_coverage.head(100))

# Conserver uniquement les émissions complètes.
complete_keys = coverage.loc[
    complete_coverage_mask,
    [
        ISSUE_DATETIME_COLUMN,
        ZONE_COLUMN,
    ],
]

pd_data = pd_data.merge(
    complete_keys,
    on=[
        ISSUE_DATETIME_COLUMN,
        ZONE_COLUMN,
    ],
    how="inner",
    validate="many_to_one",
)

if pd_data.empty:
    raise ValueError(
        "Aucun couple issue_datetime/zone ne contient "
        "les 168 horizons complets."
    )

print(
    f"Lignes après filtrage des émissions complètes : "
    f"{len(pd_data):,}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Définition des features candidates

# COMMAND ----------

# Exclure uniquement : la cible, les dates brutes (inutiles en epoch),
# les colonnes techniques, et demand_mw (même grain que la cible).
EXCLUDED_COLUMNS = {
    "target_demand_mw",
    "target_24h_ahead",
    "datetime",
    "issue_datetime",
    "target_datetime",
    "issue_date",
    "target_date",
    "date",
    "processed_time",
    "has_valid_features",
    "has_valid_target",
    "is_training_row",
    "demand_mw",
}

candidate_features = [
    column
    for column in pd_data.columns
    if column not in EXCLUDED_COLUMNS
    and column != TARGET_COLUMN
]

# Features obligatoires conservées dans tous les cas (utilisées par la
# sélection de features en aval).
MANDATORY_FEATURES = [
    ZONE_COLUMN,
    HORIZON_COLUMN,
    "forecast_day",
    "forecast_hour_in_day",
    "forecast_horizon_sqrt",
    "forecast_horizon_log1p",
    "forecast_horizon_sin_24h",
    "forecast_horizon_cos_24h",
    "forecast_horizon_sin_168h",
    "forecast_horizon_cos_168h",
    "target_hour_sin",
    "target_hour_cos",
    "target_day_of_week_sin",
    "target_day_of_week_cos",
    "target_day_of_year_sin",
    "target_day_of_year_cos",
    "target_is_weekend",
]

for feature in MANDATORY_FEATURES:
    if feature in pd_data.columns and feature not in candidate_features:
        candidate_features.append(feature)

# La table Gold est déjà nettoyée (EDA validée) : pas de vérification
# de fuites ni de présélection technique.
selected_candidate_features = candidate_features.copy()

print(f"Features candidates : {len(selected_candidate_features)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Préparation des types

# COMMAND ----------

model_data = pd_data[
    [
        ISSUE_DATETIME_COLUMN,
        TARGET_DATETIME_COLUMN,
        TARGET_COLUMN,
    ]
    + selected_candidate_features
].copy()

zone_categories = sorted(
    model_data[ZONE_COLUMN]
    .astype(str)
    .dropna()
    .unique()
    .tolist()
)

model_data[ZONE_COLUMN] = pd.Categorical(
    model_data[ZONE_COLUMN].astype(str),
    categories=zone_categories,
)

boolean_features = [
    column
    for column in selected_candidate_features
    if pd.api.types.is_bool_dtype(
        model_data[column]
    )
]

for column in boolean_features:
    model_data[column] = (
        model_data[column].astype("int8")
    )

numeric_features = [
    column
    for column in selected_candidate_features
    if column != ZONE_COLUMN
]

for column in numeric_features:
    model_data[column] = pd.to_numeric(
        model_data[column],
        errors="coerce",
    )

model_data[TARGET_COLUMN] = pd.to_numeric(
    model_data[TARGET_COLUMN],
    errors="coerce",
)

model_data = model_data.dropna(
    subset=[
        ISSUE_DATETIME_COLUMN,
        TARGET_DATETIME_COLUMN,
        HORIZON_COLUMN,
        TARGET_COLUMN,
        ZONE_COLUMN,
    ]
).copy()

model_data = model_data.sort_values(
    [
        ISSUE_DATETIME_COLUMN,
        ZONE_COLUMN,
        HORIZON_COLUMN,
    ]
).reset_index(drop=True)

print(
    f"Lignes après préparation : "
    f"{len(model_data):,}"
)
print(
    f"Features candidates       : "
    f"{len(selected_candidate_features)}"
)

# COMMAND ----------

# DBTITLE 1,Walk-forward header
# MAGIC %md
# MAGIC ## 9. Walk-forward backtesting
# MAGIC
# MAGIC Au lieu d'un seul split train/validation/test, on définit **5 folds de 30 jours**
# MAGIC avec une fenêtre d'entraînement expansive. Le dernier fold sert de test final,
# MAGIC et les folds précédents permettent d'évaluer la robustesse du modèle dans le temps.

# COMMAND ----------

# DBTITLE 1,Walk-forward split
# --- Walk-forward : définition des folds ---
unique_issue_datetimes = np.sort(
    model_data[
        ISSUE_DATETIME_COLUMN
    ].dropna().unique()
)

if len(unique_issue_datetimes) < 100:
    raise ValueError(
        "Moins de 100 dates d'émission pour le walk-forward."
    )

# 5 folds de 30 jours, fenêtre expansive, depuis la fin des données.
max_datetime = pd.Timestamp(unique_issue_datetimes[-1])
fold_boundaries = []
for i in range(N_WALK_FORWARD_FOLDS):
    fold_end = max_datetime - pd.Timedelta(days=WALK_FORWARD_TEST_DAYS * i)
    fold_start = fold_end - pd.Timedelta(days=WALK_FORWARD_TEST_DAYS)
    fold_boundaries.append((fold_start, fold_end))
fold_boundaries.reverse()

print("Folds walk-forward :")
for i, (start, end) in enumerate(fold_boundaries, 1):
    train_count = (model_data[ISSUE_DATETIME_COLUMN] < start).sum()
    test_count = ((model_data[ISSUE_DATETIME_COLUMN] >= start) & (model_data[ISSUE_DATETIME_COLUMN] < end)).sum()
    print(f"  Fold {i}: test {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} | train: {train_count:,} | test: {test_count:,}")

# --- Split final : dernier fold = test, avant-dernier = validation ---
last_fold_start, last_fold_end = fold_boundaries[-1]
second_last_start, _ = fold_boundaries[-2]

validation_start = pd.Timestamp(second_last_start)
test_start = pd.Timestamp(last_fold_start)

# Purge des cibles qui débordent dans la période suivante.
train_latest_target = validation_start - pd.Timedelta(hours=1)
validation_latest_target = test_start - pd.Timedelta(hours=1)

train_mask = (
    (model_data[ISSUE_DATETIME_COLUMN] < validation_start)
    & (model_data[TARGET_DATETIME_COLUMN] <= train_latest_target)
)

validation_mask = (
    (model_data[ISSUE_DATETIME_COLUMN] >= validation_start)
    & (model_data[ISSUE_DATETIME_COLUMN] < test_start)
    & (model_data[TARGET_DATETIME_COLUMN] <= validation_latest_target)
)

test_mask = (
    (model_data[ISSUE_DATETIME_COLUMN] >= test_start)
    & (model_data[ISSUE_DATETIME_COLUMN] < last_fold_end)
)

train_data = model_data.loc[train_mask].copy()
validation_data = model_data.loc[validation_mask].copy()
test_data = model_data.loc[test_mask].copy()

if min(
    len(train_data),
    len(validation_data),
    len(test_data),
) == 0:
    raise ValueError(
        "Au moins un jeu de données est vide "
        "après la purge temporelle."
    )

if not (
    train_data[TARGET_DATETIME_COLUMN].max()
    < validation_data[ISSUE_DATETIME_COLUMN].min()
):
    raise AssertionError(
        "Fuite temporelle entre train et validation."
    )

if not (
    validation_data[TARGET_DATETIME_COLUMN].max()
    < test_data[ISSUE_DATETIME_COLUMN].min()
):
    raise AssertionError(
        "Fuite temporelle entre validation et test."
    )

print("=" * 80)
print("SPLIT FINAL (walk-forward purgé)")
print("=" * 80)

print(
    f"Train      : {len(train_data):,} lignes | "
    f"issue {train_data[ISSUE_DATETIME_COLUMN].min()} "
    f"à {train_data[ISSUE_DATETIME_COLUMN].max()} | "
    f"target max "
    f"{train_data[TARGET_DATETIME_COLUMN].max()}"
)

print(
    f"Validation : {len(validation_data):,} lignes | "
    f"issue {validation_data[ISSUE_DATETIME_COLUMN].min()} "
    f"à {validation_data[ISSUE_DATETIME_COLUMN].max()} | "
    f"target max "
    f"{validation_data[TARGET_DATETIME_COLUMN].max()}"
)

print(
    f"Test       : {len(test_data):,} lignes | "
    f"issue {test_data[ISSUE_DATETIME_COLUMN].min()} "
    f"à {test_data[ISSUE_DATETIME_COLUMN].max()} | "
    f"target max "
    f"{test_data[TARGET_DATETIME_COLUMN].max()}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Fonctions utilitaires

# COMMAND ----------

def calculate_metrics(y_true, y_pred):
    """Calcule les métriques de régression."""

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    valid_mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "smape": np.nan,
            "wape": np.nan,
            "bias": np.nan,
        }

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = (
        r2_score(y_true, y_pred)
        if len(y_true) >= 2
        else np.nan
    )

    non_zero_mask = (
        np.abs(y_true) > 1e-6
    )

    if non_zero_mask.any():
        mape = (
            np.mean(
                np.abs(
                    (
                        y_true[non_zero_mask]
                        - y_pred[non_zero_mask]
                    )
                    / y_true[non_zero_mask]
                )
            )
            * 100.0
        )
    else:
        mape = np.nan

    smape_denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    smape_mask = (
        smape_denominator > 1e-6
    )

    if smape_mask.any():
        smape = (
            np.mean(
                2.0
                * np.abs(
                    y_true[smape_mask]
                    - y_pred[smape_mask]
                )
                / smape_denominator[
                    smape_mask
                ]
            )
            * 100.0
        )
    else:
        smape = np.nan

    target_sum = np.abs(y_true).sum()

    if target_sum > 1e-6:
        wape = (
            np.abs(
                y_true - y_pred
            ).sum()
            / target_sum
            * 100.0
        )
    else:
        wape = np.nan

    bias = np.mean(
        y_pred - y_true
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "smape": float(smape),
        "wape": float(wape),
        "bias": float(bias),
    }


def print_metrics(name, metrics):
    """Affiche les métriques."""

    print(f"\n{name}")
    print("-" * 50)
    print(f"MAE   : {metrics['mae']:.3f} MW")
    print(f"RMSE  : {metrics['rmse']:.3f} MW")
    print(f"R²    : {metrics['r2']:.5f}")
    print(f"MAPE  : {metrics['mape']:.3f} %")
    print(f"sMAPE : {metrics['smape']:.3f} %")
    print(f"WAPE  : {metrics['wape']:.3f} %")
    print(f"Biais : {metrics['bias']:.3f} MW")


def train_lightgbm(
    train_frame,
    validation_frame,
    features,
    target_column,
    seed=42,
):
    """Entraîne un LightGBM avec early stopping."""

    X_train = train_frame[
        features
    ].copy()

    y_train = train_frame[
        target_column
    ].copy()

    X_validation = validation_frame[
        features
    ].copy()

    y_validation = validation_frame[
        target_column
    ].copy()

    categorical_features = [
        feature
        for feature in features
        if str(
            X_train[feature].dtype
        ) == "category"
    ]

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=1.00,
        random_state=seed,
        n_jobs=-1,
        importance_type="gain",
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        eval_metric="mae",
        categorical_feature=(
            categorical_features
        ),
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=30,
                verbose=False,
            ),
            lgb.log_evaluation(
                period=0
            ),
        ],
    )

    return model


def get_feature_importance(
    model,
    features,
):
    """Retourne les importances LightGBM."""

    booster = model.booster_

    importance = pd.DataFrame({
        "feature": features,
        "gain": booster.feature_importance(
            importance_type="gain"
        ),
        "split": booster.feature_importance(
            importance_type="split"
        ),
    })

    total_gain = importance["gain"].sum()

    if total_gain > 0:
        importance["gain_pct"] = (
            importance["gain"]
            / total_gain
            * 100.0
        )
    else:
        importance["gain_pct"] = 0.0

    importance = importance.sort_values(
        "gain",
        ascending=False,
    ).reset_index(drop=True)

    importance[
        "cumulative_gain_pct"
    ] = importance["gain_pct"].cumsum()

    return importance


def calculate_grouped_metrics(
    source_data,
    predictions,
    group_columns,
):
    """Calcule les métriques par groupes."""

    result = source_data[
        group_columns
        + [TARGET_COLUMN]
    ].copy()

    result["prediction_mw"] = predictions

    rows = []

    grouped = result.groupby(
        group_columns,
        observed=True,
    )

    for group_key, group_data in grouped:
        if not isinstance(
            group_key,
            tuple,
        ):
            group_key = (group_key,)

        row = {
            column: (
                str(value)
                if column == ZONE_COLUMN
                else int(value)
                if column in {
                    "forecast_day",
                    HORIZON_COLUMN,
                }
                else value
            )
            for column, value in zip(
                group_columns,
                group_key,
            )
        }

        metrics = calculate_metrics(
            group_data[TARGET_COLUMN],
            group_data["prediction_mw"],
        )

        row.update({
            "observations": len(group_data),
            **metrics,
        })

        rows.append(row)

    return pd.DataFrame(rows)


def save_dataframe_csv(dataframe, path):
    """Sauvegarde un DataFrame en CSV."""

    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def prepare_mlflow_example(frame):
    """
    Prépare un exemple d'entrée sérialisable.
    """

    output = frame.copy()

    for column in output.columns:
        if str(
            output[column].dtype
        ) == "category":
            output[column] = (
                output[column].astype(str)
            )

    return output

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Modèle baseline

# COMMAND ----------

baseline_features = (
    selected_candidate_features.copy()
)

print(
    f"Entraînement baseline avec "
    f"{len(baseline_features)} features..."
)

baseline_model = train_lightgbm(
    train_frame=train_data,
    validation_frame=validation_data,
    features=baseline_features,
    target_column=TARGET_COLUMN,
    seed=RANDOM_STATE,
)

baseline_validation_predictions = (
    baseline_model.predict(
        validation_data[
            baseline_features
        ],
        num_iteration=(
            baseline_model.best_iteration_
        ),
    )
)

baseline_validation_metrics = (
    calculate_metrics(
        validation_data[TARGET_COLUMN],
        baseline_validation_predictions,
    )
)

print_metrics(
    "BASELINE VALIDATION",
    baseline_validation_metrics,
)

baseline_importance = (
    get_feature_importance(
        baseline_model,
        baseline_features,
    )
)

display(
    baseline_importance.head(50)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Sélection des features

# COMMAND ----------

ranked_features = baseline_importance.loc[
    baseline_importance["gain"] > 0,
    "feature",
].tolist()

if not ranked_features:
    raise ValueError(
        "Toutes les features ont une importance nulle."
    )

feature_counts = sorted(
    set(
        [
            count
            for count in FEATURE_COUNTS_TO_TEST
            if count <= len(ranked_features)
        ]
        + [len(ranked_features)]
    )
)


def get_selected_features(
    feature_count,
):
    """
    Sélectionne les meilleures features en conservant
    toutes les variables obligatoires.
    """

    selected = ranked_features[
        :feature_count
    ].copy()

    for feature in MANDATORY_FEATURES:
        if (
            feature in baseline_features
            and feature not in selected
        ):
            selected.append(feature)

    return list(
        dict.fromkeys(selected)
    )


selection_results = []
selection_models = {}

print(
    f"Configurations testées : "
    f"{feature_counts}"
)

for selection_key in feature_counts:
    current_features = (
        get_selected_features(
            selection_key
        )
    )

    current_model = train_lightgbm(
        train_frame=train_data,
        validation_frame=validation_data,
        features=current_features,
        target_column=TARGET_COLUMN,
        seed=RANDOM_STATE,
    )

    current_predictions = (
        current_model.predict(
            validation_data[
                current_features
            ],
            num_iteration=(
                current_model.best_iteration_
            ),
        )
    )

    current_metrics = calculate_metrics(
        validation_data[TARGET_COLUMN],
        current_predictions,
    )

    selection_results.append({
        "selection_key": selection_key,
        "requested_feature_count": (
            selection_key
        ),
        "actual_feature_count": len(
            current_features
        ),
        "best_iteration": int(
            current_model.best_iteration_
        ),
        **current_metrics,
    })

    selection_models[selection_key] = {
        "model": current_model,
        "features": current_features,
    }

    print(
        f"{len(current_features):3d} features | "
        f"WAPE={current_metrics['wape']:.3f}% | "
        f"MAPE={current_metrics['mape']:.3f}% | "
        f"MAE={current_metrics['mae']:.3f} MW"
    )

selection_results_df = (
    pd.DataFrame(selection_results)
    .sort_values(
        [
            "actual_feature_count",
            "wape",
        ]
    )
    .reset_index(drop=True)
)

display(selection_results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Choix de la configuration finale

# COMMAND ----------

best_validation_wape = (
    selection_results_df["wape"].min()
)

acceptable_results = (
    selection_results_df[
        selection_results_df["wape"]
        <= (
            best_validation_wape
            + WAPE_TOLERANCE_PERCENTAGE_POINT
        )
    ]
    .copy()
)

chosen_result = (
    acceptable_results
    .sort_values(
        [
            "actual_feature_count",
            "wape",
            "mae",
        ]
    )
    .iloc[0]
)

CHOSEN_SELECTION_KEY = int(
    chosen_result["selection_key"]
)

selected_features = selection_models[
    CHOSEN_SELECTION_KEY
]["features"]

selected_validation_model = (
    selection_models[
        CHOSEN_SELECTION_KEY
    ]["model"]
)

BEST_ITERATION = int(
    selected_validation_model.best_iteration_
)

if HORIZON_COLUMN not in selected_features:
    raise ValueError(
        "forecast_horizon_hours doit rester dans "
        "les features du modèle."
    )

print("=" * 80)
print("CONFIGURATION SÉLECTIONNÉE")
print("=" * 80)
print(
    f"Meilleur WAPE validation : "
    f"{best_validation_wape:.3f} %"
)
print(
    f"WAPE retenu              : "
    f"{chosen_result['wape']:.3f} %"
)
print(
    f"Features sélectionnées   : "
    f"{len(selected_features)}"
)
print(
    f"Nombre d'arbres          : "
    f"{BEST_ITERATION}"
)

for index, feature in enumerate(
    selected_features,
    start=1,
):
    print(f"{index:3d}. {feature}")

# COMMAND ----------

# DBTITLE 1,Optuna header
# MAGIC %md
# MAGIC ## 13b. Optimisation des hyperparamètres (Optuna)
# MAGIC
# MAGIC Recherche bayésienne sur les hyperparamètres LightGBM, en utilisant les features
# MAGIC sélectionnées à l'étape précédente. Optimisation sur le WAPE de validation.
# MAGIC
# MAGIC Espace de recherche :
# MAGIC - `learning_rate`, `num_leaves`, `min_child_samples`
# MAGIC - `subsample`, `colsample_bytree`
# MAGIC - `reg_alpha`, `reg_lambda`, `max_depth`
# MAGIC
# MAGIC Early stopping (50 rounds) pour trouver automatiquement le nombre optimal d'arbres.

# COMMAND ----------

# DBTITLE 1,Optuna tuning
# --- Optimisation Optuna des hyperparamètres ---
optuna.logging.set_verbosity(optuna.logging.WARNING)

cat_cols_tune = [
    c for c in selected_features
    if str(train_data[c].dtype) == "category"
]

X_tune_train = train_data[selected_features].copy()
y_tune_train = train_data[TARGET_COLUMN].copy()
X_tune_val = validation_data[selected_features].copy()
y_tune_val = validation_data[TARGET_COLUMN].copy()


def optuna_objective(trial):
    """Minimise le WAPE sur le jeu de validation."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        **params,
        subsample_freq=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_tune_train,
        y_tune_train,
        eval_set=[(X_tune_val, y_tune_val)],
        categorical_feature=cat_cols_tune,
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    val_pred = model.predict(X_tune_val)
    val_metrics = calculate_metrics(y_tune_val, val_pred)

    trial.set_user_attr("best_iteration", int(model.best_iteration_))

    return val_metrics["wape"]


N_OPTUNA_TRIALS = 20

print(f"Optimisation Optuna : {N_OPTUNA_TRIALS} trials")
print(f"Features : {len(selected_features)}")
print(f"Train : {len(train_data):,} | Validation : {len(validation_data):,}")
print("=" * 70)

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)
study.optimize(
    optuna_objective,
    n_trials=N_OPTUNA_TRIALS,
    show_progress_bar=True,
)

tuned_params = study.best_params
tuned_n_estimators = study.best_trial.user_attrs["best_iteration"]

print("\n" + "=" * 70)
print("MEILLEURS HYPERPARAMÈTRES (Optuna)")
print("=" * 70)
print(f"WAPE validation : {study.best_value:.3f}%")
print(f"N arbres        : {tuned_n_estimators}")
for key, value in tuned_params.items():
    print(f"  {key:20s} : {value}")

# Comparaison avec les paramètres par défaut
default_params = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.10,
    "reg_lambda": 1.00,
}
print("\nComparaison paramètres par défaut → optimisés :")
for key in default_params:
    old = default_params[key]
    new = tuned_params.get(key, old)
    print(f"  {key:20s} : {old} → {new}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Réentraînement final

# COMMAND ----------

train_validation_data = (
    pd.concat(
        [
            train_data,
            validation_data,
        ],
        axis=0,
    )
    .sort_values(
        [
            ISSUE_DATETIME_COLUMN,
            ZONE_COLUMN,
            HORIZON_COLUMN,
        ]
    )
    .reset_index(drop=True)
)

X_train_validation = (
    train_validation_data[
        selected_features
    ].copy()
)

y_train_validation = (
    train_validation_data[
        TARGET_COLUMN
    ].copy()
)

X_test = test_data[
    selected_features
].copy()

y_test = test_data[
    TARGET_COLUMN
].copy()

categorical_features_final = [
    feature
    for feature in selected_features
    if str(
        X_train_validation[
            feature
        ].dtype
    ) == "category"
]

final_params = {
    "objective": "regression_l1",
    "n_estimators": tuned_n_estimators,
    **tuned_params,
    "subsample_freq": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "importance_type": "gain",
    "verbosity": -1,
}

print(
    f"Réentraînement avec "
    f"{len(selected_features)} features et "
    f"{BEST_ITERATION} arbres..."
)

final_model = lgb.LGBMRegressor(
    **final_params
)

final_model.fit(
    X_train_validation,
    y_train_validation,
    categorical_feature=(
        categorical_features_final
    ),
    callbacks=[
        lgb.log_evaluation(period=0)
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Évaluation finale

# COMMAND ----------

train_validation_predictions = (
    final_model.predict(
        X_train_validation
    )
)

test_predictions = (
    final_model.predict(
        X_test
    )
)

train_validation_metrics = (
    calculate_metrics(
        y_train_validation,
        train_validation_predictions,
    )
)

test_metrics = calculate_metrics(
    y_test,
    test_predictions,
)

print_metrics(
    "TRAIN + VALIDATION",
    train_validation_metrics,
)

print_metrics(
    "TEST FINAL",
    test_metrics,
)

test_predictions_df = test_data[
    [
        ISSUE_DATETIME_COLUMN,
        TARGET_DATETIME_COLUMN,
        HORIZON_COLUMN,
        "forecast_day",
        ZONE_COLUMN,
        TARGET_COLUMN,
    ]
].copy()

test_predictions_df["prediction_mw"] = (
    test_predictions
)

test_predictions_df["error_mw"] = (
    test_predictions_df["prediction_mw"]
    - test_predictions_df[TARGET_COLUMN]
)

test_predictions_df[
    "absolute_error_mw"
] = np.abs(
    test_predictions_df["error_mw"]
)

test_predictions_df[
    "absolute_percentage_error"
] = np.where(
    np.abs(
        test_predictions_df[TARGET_COLUMN]
    ) > 1e-6,
    (
        test_predictions_df[
            "absolute_error_mw"
        ]
        / np.abs(
            test_predictions_df[
                TARGET_COLUMN
            ]
        )
        * 100.0
    ),
    np.nan,
)

display(
    test_predictions_df.head(100)
)

# COMMAND ----------

# DBTITLE 1,Quantile forecasting header
# MAGIC %md
# MAGIC ## Quantile forecasting P10/P50/P90
# MAGIC
# MAGIC Entraînement de 3 modèles LightGBM avec `objective="quantile"` pour produire
# MAGIC des intervalles de prévision sur les 168 horizons :
# MAGIC - **P10** : scénario optimiste (10e percentile)
# MAGIC - **P50** : médiane (50e percentile)
# MAGIC - **P90** : scénario pessimiste (90e percentile)

# COMMAND ----------

# DBTITLE 1,Quantile forecasting training
# --- Entraînement des modèles quantile ---
quantile_models = {}
quantile_predictions_test = {}

cat_cols_07 = [
    c for c in selected_features
    if str(X_train_validation[c].dtype) == "category"
]

for q in QUANTILES:
    print(f"\nEntraînement modèle quantile P{int(q*100)}...")
    
    quantile_model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=tuned_n_estimators,
        **tuned_params,
        subsample_freq=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    
    quantile_model.fit(
        X_train_validation,
        y_train_validation,
        categorical_feature=cat_cols_07,
    )
    
    quantile_models[q] = quantile_model
    quantile_predictions_test[q] = quantile_model.predict(X_test)
    print(f"  P{int(q*100)} : {len(quantile_predictions_test[q])} prédictions générées")

# --- Évaluation : pinball loss ---
def pinball_loss(y_true, y_pred, quantile):
    """Calcule la pinball loss pour un quantile donné."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))

print("\n" + "=" * 70)
print("ÉVALUATION QUANTILE FORECASTING")
print("=" * 70)

for q in QUANTILES:
    pl = pinball_loss(y_test, quantile_predictions_test[q], q)
    print(f"  P{int(q*100):2d} pinball loss : {pl:.3f} MW")

y_test_arr = np.asarray(y_test, dtype=float)
coverage = np.mean(
    (y_test_arr >= quantile_predictions_test[0.1])
    & (y_test_arr <= quantile_predictions_test[0.9])
) * 100
print(f"\n  Couverture P10-P90 : {coverage:.1f}% (cible : 80%)")

interval_width = np.mean(quantile_predictions_test[0.9] - quantile_predictions_test[0.1])
print(f"  Largeur moyenne P10-P90 : {interval_width:.1f} MW")

# Ajouter les colonnes quantile au DataFrame de prédictions
test_predictions_df["p10_mw"] = quantile_predictions_test[0.1]
test_predictions_df["p50_mw"] = quantile_predictions_test[0.5]
test_predictions_df["p90_mw"] = quantile_predictions_test[0.9]

print("\n✅ Quantile forecasting ajouté au DataFrame de prédictions")

# COMMAND ----------

# DBTITLE 1,Walk-forward evaluation header
# MAGIC %md
# MAGIC ## Walk-forward backtesting (allégé)
# MAGIC
# MAGIC Évaluation du modèle sur les 5 folds. Pour chaque fold :
# MAGIC 1. entraîner un **point forecast** sur toutes les données émises avant le fold ;
# MAGIC 2. prédire sur le fold et calculer les métriques.
# MAGIC
# MAGIC Les **quantiles P10/P50/P90** sont entraînés uniquement sur le **dernier fold** (test set)
# MAGIC pour réduire le temps de calcul (8 modèles au lieu de 20).

# COMMAND ----------

# DBTITLE 1,Walk-forward evaluation
# --- Walk-forward backtesting (allégé) ---
# Folds 1-5 : point forecast uniquement
# Dernier fold (test) : point forecast + 3 quantiles
wf_results = []
wf_predictions_all = []

print("=" * 70)
print("WALK-FORWARD BACKTESTING (allégé)")
print("=" * 70)

for fold_idx, (fold_start, fold_end) in enumerate(fold_boundaries, 1):
    fold_start_ts = pd.Timestamp(fold_start)
    fold_end_ts = pd.Timestamp(fold_end)

    wf_train = model_data[model_data[ISSUE_DATETIME_COLUMN] < fold_start_ts]
    wf_test = model_data[
        (model_data[ISSUE_DATETIME_COLUMN] >= fold_start_ts)
        & (model_data[ISSUE_DATETIME_COLUMN] < fold_end_ts)
    ]

    if len(wf_train) == 0 or len(wf_test) == 0:
        print(f"  Fold {fold_idx}: ignoré (train={len(wf_train)}, test={len(wf_test)})")
        continue

    X_wf_train = wf_train[selected_features].copy()
    y_wf_train = wf_train[TARGET_COLUMN].copy()
    X_wf_test = wf_test[selected_features].copy()
    y_wf_test = wf_test[TARGET_COLUMN].copy()

    cat_cols_wf = [c for c in selected_features if str(X_wf_train[c].dtype) == "category"]

    # Modèle point forecast
    wf_model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=tuned_n_estimators,
        **tuned_params,
        subsample_freq=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    wf_model.fit(X_wf_train, y_wf_train, categorical_feature=cat_cols_wf)
    wf_pred = wf_model.predict(X_wf_test)
    wf_metrics = calculate_metrics(y_wf_test, wf_pred)

    wf_row = {
        "fold": fold_idx,
        "test_start": fold_start_ts.strftime("%Y-%m-%d"),
        "test_end": fold_end_ts.strftime("%Y-%m-%d"),
        "train_size": len(wf_train),
        "test_size": len(wf_test),
        **wf_metrics,
    }

    wf_pred_df = wf_test[
        [ISSUE_DATETIME_COLUMN, TARGET_DATETIME_COLUMN, HORIZON_COLUMN, ZONE_COLUMN, TARGET_COLUMN]
    ].copy()
    wf_pred_df["prediction_mw"] = wf_pred
    wf_pred_df["fold"] = fold_idx

    # Quantiles uniquement sur le dernier fold (test set)
    is_last_fold = fold_idx == len(fold_boundaries)
    if is_last_fold:
        wf_quantile_preds = {}
        for q in QUANTILES:
            wf_q_model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=tuned_n_estimators,
                **tuned_params,
                subsample_freq=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )
            wf_q_model.fit(X_wf_train, y_wf_train, categorical_feature=cat_cols_wf)
            wf_quantile_preds[q] = wf_q_model.predict(X_wf_test)

        wf_pl = {q: pinball_loss(y_wf_test, wf_quantile_preds[q], q) for q in QUANTILES}
        y_wf_arr = np.asarray(y_wf_test, dtype=float)
        wf_coverage = np.mean(
            (y_wf_arr >= wf_quantile_preds[0.1])
            & (y_wf_arr <= wf_quantile_preds[0.9])
        ) * 100

        wf_row["pinball_p10"] = wf_pl[0.1]
        wf_row["pinball_p50"] = wf_pl[0.5]
        wf_row["pinball_p90"] = wf_pl[0.9]
        wf_row["coverage_p10_p90"] = wf_coverage

        wf_pred_df["p10_mw"] = wf_quantile_preds[0.1]
        wf_pred_df["p50_mw"] = wf_quantile_preds[0.5]
        wf_pred_df["p90_mw"] = wf_quantile_preds[0.9]

        print(
            f"  Fold {fold_idx} (TEST): WAPE={wf_metrics['wape']:.3f}% "
            f"| MAE={wf_metrics['mae']:.1f} MW | Couverture={wf_coverage:.1f}%"
        )
    else:
        print(f"  Fold {fold_idx}         : WAPE={wf_metrics['wape']:.3f}% | MAE={wf_metrics['mae']:.1f} MW")

    wf_results.append(wf_row)
    wf_predictions_all.append(wf_pred_df)

wf_results_df = pd.DataFrame(wf_results)
wf_predictions_all_df = pd.concat(wf_predictions_all, ignore_index=True)

print("\n" + "=" * 70)
print("MÉTRIQUES AGRÉGÉES WALK-FORWARD")
print("=" * 70)
for metric in ["wape", "mae", "rmse", "mape"]:
    values = wf_results_df[metric]
    print(f"  {metric.upper():6s} : {values.mean():.3f} ± {values.std():.3f}")

# Pinball et coverage uniquement sur le dernier fold
if "pinball_p10" in wf_results_df.columns:
    last = wf_results_df[wf_results_df["pinball_p10"].notna()]
    if len(last) > 0:
        for q in QUANTILES:
            val = last[f"pinball_p{int(q*100)}"].iloc[0]
            print(f"  PL P{int(q*100):2d}  : {val:.3f} MW (dernier fold)")
        print(f"  COUVERT : {last['coverage_p10_p90'].iloc[0]:.1f}% (dernier fold)")

display(wf_results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Métriques détaillées

# COMMAND ----------

metrics_by_zone = calculate_grouped_metrics(
    source_data=test_data,
    predictions=test_predictions,
    group_columns=[
        ZONE_COLUMN,
    ],
).sort_values("wape")

metrics_by_forecast_day = (
    calculate_grouped_metrics(
        source_data=test_data,
        predictions=test_predictions,
        group_columns=[
            "forecast_day",
        ],
    )
    .sort_values("forecast_day")
)

metrics_by_zone_and_day = (
    calculate_grouped_metrics(
        source_data=test_data,
        predictions=test_predictions,
        group_columns=[
            ZONE_COLUMN,
            "forecast_day",
        ],
    )
    .sort_values(
        [
            ZONE_COLUMN,
            "forecast_day",
        ]
    )
)

metrics_by_horizon = (
    calculate_grouped_metrics(
        source_data=test_data,
        predictions=test_predictions,
        group_columns=[
            HORIZON_COLUMN,
        ],
    )
    .sort_values(HORIZON_COLUMN)
)

print("Métriques par zone")
display(metrics_by_zone)

print("Métriques par jour de prévision")
display(metrics_by_forecast_day)

print("Métriques par zone et par jour")
display(metrics_by_zone_and_day)

print("Métriques par horizon")
display(metrics_by_horizon)

# COMMAND ----------

# DBTITLE 1,Reconciliation header
# MAGIC %md
# MAGIC ## Réconciliation hiérarchique zone ↔ Ontario total
# MAGIC
# MAGIC **Méthode bottom-up** : la somme des prévisions zonales constitue le total Ontario.
# MAGIC Garantit la cohérence hiérarchique par construction (Σ zones = total Ontario).

# COMMAND ----------

# DBTITLE 1,Reconciliation bottom-up
# --- Réconciliation bottom-up : somme des prévisions zonales = total Ontario ---

# Identifier les zones individuelles (exclure "Ontario" qui est le total).
all_zones = sorted(test_predictions_df[ZONE_COLUMN].dropna().unique())
individual_zones = [z for z in all_zones if z != "Ontario"]
print(f"Zones individuelles : {individual_zones}")
print(f"Zone 'Ontario' présente : {'Ontario' in all_zones}")

# Bottom-up : somme des prévisions des zones individuelles.
zonal_pred = test_predictions_df[
    test_predictions_df[ZONE_COLUMN].isin(individual_zones)
]

ontario_bottomup = (
    zonal_pred
    .groupby(
        [ISSUE_DATETIME_COLUMN, TARGET_DATETIME_COLUMN],
        as_index=False,
    )
    .agg(
        bottomup_actual_mw=(TARGET_COLUMN, "sum"),
        bottomup_predicted_mw=("prediction_mw", "sum"),
        bottomup_p10_mw=("p10_mw", "sum"),
        bottomup_p50_mw=("p50_mw", "sum"),
        bottomup_p90_mw=("p90_mw", "sum"),
    )
    .sort_values([ISSUE_DATETIME_COLUMN, TARGET_DATETIME_COLUMN])
)

# Ontario direct : prévisions de la zone "Ontario" (si présente).
if "Ontario" in all_zones:
    ontario_direct = (
        test_predictions_df[test_predictions_df[ZONE_COLUMN] == "Ontario"]
        .groupby([ISSUE_DATETIME_COLUMN, TARGET_DATETIME_COLUMN], as_index=False)
        .agg(
            direct_actual_mw=(TARGET_COLUMN, "first"),
            direct_predicted_mw=("prediction_mw", "first"),
        )
        .sort_values([ISSUE_DATETIME_COLUMN, TARGET_DATETIME_COLUMN])
    )
    
    comparison = ontario_bottomup.merge(
        ontario_direct,
        on=[ISSUE_DATETIME_COLUMN, TARGET_DATETIME_COLUMN],
        how="inner",
    )
    
    coherence_gap = np.mean(np.abs(
        comparison["bottomup_actual_mw"] - comparison["direct_actual_mw"]
    ))
    print(f"\nÉcart moyen actual bottom-up vs Ontario direct : {coherence_gap:.1f} MW")
    
    ontario_metrics = calculate_metrics(
        comparison["bottomup_actual_mw"],
        comparison["bottomup_predicted_mw"],
    )
    ontario_direct_metrics = calculate_metrics(
        comparison["direct_actual_mw"],
        comparison["direct_predicted_mw"],
    )
else:
    ontario_metrics = calculate_metrics(
        ontario_bottomup["bottomup_actual_mw"],
        ontario_bottomup["bottomup_predicted_mw"],
    )

print("\n" + "=" * 70)
print("RÉCONCILIATION BOTTOM-UP : TOTAL ONTARIO")
print("=" * 70)
print_metrics("Total Ontario (bottom-up)", ontario_metrics)

if "Ontario" in all_zones:
    print_metrics("Total Ontario (modèle direct)", ontario_direct_metrics)
    print(f"\n→ Le bottom-up (somme des {len(individual_zones)} zones) peut différer")
    print(f"  du modèle direct (Ontario prédit séparément).")
    print(f"  La réconciliation assure la cohérence hiérarchique.")

# Couverture P10-P90 au niveau Ontario
ontario_coverage = np.mean(
    (ontario_bottomup["bottomup_actual_mw"] >= ontario_bottomup["bottomup_p10_mw"])
    & (ontario_bottomup["bottomup_actual_mw"] <= ontario_bottomup["bottomup_p90_mw"])
) * 100
print(f"\nCouverture P10-P90 (Ontario bottom-up) : {ontario_coverage:.1f}%")

# Visualisation : dernière prévision hebdomadaire avec intervalles
latest_test_issue = ontario_bottomup[ISSUE_DATETIME_COLUMN].max()
latest_week_ontario = ontario_bottomup[
    ontario_bottomup[ISSUE_DATETIME_COLUMN] == latest_test_issue
].sort_values(TARGET_DATETIME_COLUMN)

fig, ax = plt.subplots(figsize=(16, 7))

ax.fill_between(
    latest_week_ontario[TARGET_DATETIME_COLUMN],
    latest_week_ontario["bottomup_p10_mw"],
    latest_week_ontario["bottomup_p90_mw"],
    alpha=0.25,
    color="orange",
    label="Intervalle P10-P90",
)
ax.plot(
    latest_week_ontario[TARGET_DATETIME_COLUMN],
    latest_week_ontario["bottomup_actual_mw"],
    label="Demande réelle Ontario",
)
ax.plot(
    latest_week_ontario[TARGET_DATETIME_COLUMN],
    latest_week_ontario["bottomup_predicted_mw"],
    label="Prédiction (bottom-up)",
)

ax.set_xlabel("Date et heure cible")
ax.set_ylabel("Demande totale Ontario (MW)")
ax.set_title("Réconciliation bottom-up : total Ontario avec intervalles P10/P90")
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Comparaison avec les baselines

# COMMAND ----------

baseline_rows = []

for baseline_feature in [
    "demand_lag_24h",
    "demand_lag_168h",
]:
    if baseline_feature not in test_data.columns:
        continue

    baseline_predictions = pd.to_numeric(
        test_data[baseline_feature],
        errors="coerce",
    ).to_numpy()

    baseline_metrics = calculate_metrics(
        test_data[TARGET_COLUMN],
        baseline_predictions,
    )

    baseline_rows.append({
        "model": baseline_feature,
        **baseline_metrics,
    })

baseline_rows.append({
    "model": "lightgbm_direct_168h",
    **test_metrics,
})

baseline_comparison = (
    pd.DataFrame(baseline_rows)
    .sort_values("wape")
    .reset_index(drop=True)
)

display(baseline_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Importance des features

# COMMAND ----------

final_importance = (
    get_feature_importance(
        final_model,
        selected_features,
    )
)

display(final_importance)

top_final_importance = (
    final_importance
    .head(30)
    .sort_values(
        "gain",
        ascending=True,
    )
)

fig, ax = plt.subplots(
    figsize=(12, 10)
)

ax.barh(
    top_final_importance["feature"],
    top_final_importance["gain"],
)

ax.set_xlabel("Importance par gain")
ax.set_ylabel("Feature")
ax.set_title(
    "Top 30 des features du modèle H+1 à H+168"
)

plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Erreur selon l'horizon

# COMMAND ----------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(16, 10),
    sharex=True,
)

axes[0].plot(
    metrics_by_horizon[
        HORIZON_COLUMN
    ],
    metrics_by_horizon["mae"],
)

axes[0].set_ylabel("MAE (MW)")
axes[0].set_title(
    "Erreur selon l'horizon"
)
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    metrics_by_horizon[
        HORIZON_COLUMN
    ],
    metrics_by_horizon["bias"],
)

axes[1].axhline(
    0,
    color="black",
    linestyle="--",
)

axes[1].set_xlabel(
    "Horizon de prévision (heures)"
)
axes[1].set_ylabel("Biais moyen (MW)")
axes[1].grid(True, alpha=0.3)

for axis in axes:
    for boundary in range(
        24,
        168,
        24,
    ):
        axis.axvline(
            boundary,
            color="grey",
            linestyle=":",
            alpha=0.6,
        )

plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Visualisation de la dernière prévision hebdomadaire du test

# COMMAND ----------

latest_test_issue = (
    test_predictions_df[
        ISSUE_DATETIME_COLUMN
    ].max()
)

latest_week = test_predictions_df[
    test_predictions_df[
        ISSUE_DATETIME_COLUMN
    ] == latest_test_issue
].copy()

latest_week_global = (
    latest_week
    .groupby(
        TARGET_DATETIME_COLUMN,
        as_index=False,
    )
    .agg(
        actual_demand_mw=(
            TARGET_COLUMN,
            "sum",
        ),
        predicted_demand_mw=(
            "prediction_mw",
            "sum",
        ),
    )
    .sort_values(
        TARGET_DATETIME_COLUMN
    )
)

fig, ax = plt.subplots(
    figsize=(16, 7)
)

ax.plot(
    latest_week_global[
        TARGET_DATETIME_COLUMN
    ],
    latest_week_global[
        "actual_demand_mw"
    ],
    label="Demande réelle",
)

ax.plot(
    latest_week_global[
        TARGET_DATETIME_COLUMN
    ],
    latest_week_global[
        "predicted_demand_mw"
    ],
    label="Demande prédite",
)

ax.set_xlabel("Date et heure cible")
ax.set_ylabel("Demande totale (MW)")
ax.set_title(
    "Dernière prévision hebdomadaire du jeu de test"
)
ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. Création des artefacts

# COMMAND ----------

# DBTITLE 1,MLflow artifacts
if os.path.exists(
    ARTIFACT_DIRECTORY
):
    shutil.rmtree(
        ARTIFACT_DIRECTORY
    )

os.makedirs(
    ARTIFACT_DIRECTORY,
    exist_ok=True,
)

artifact_dataframes = {
    "baseline_feature_importance.csv": (
        baseline_importance
    ),
    "final_feature_importance.csv": (
        final_importance
    ),
    "feature_selection_results.csv": (
        selection_results_df
    ),
    "test_metrics_by_zone.csv": (
        metrics_by_zone
    ),
    "test_metrics_by_forecast_day.csv": (
        metrics_by_forecast_day
    ),
    "test_metrics_by_zone_and_day.csv": (
        metrics_by_zone_and_day
    ),
    "test_metrics_by_horizon.csv": (
        metrics_by_horizon
    ),
    "baseline_comparison.csv": (
        baseline_comparison
    ),
    "test_predictions.csv": (
        test_predictions_df
    ),
    "walk_forward_results.csv": (
        wf_results_df
    ),
    "walk_forward_predictions.csv": (
        wf_predictions_all_df
    ),
}

for filename, dataframe in (
    artifact_dataframes.items()
):
    save_dataframe_csv(
        dataframe,
        os.path.join(
            ARTIFACT_DIRECTORY,
            filename,
        ),
    )

selected_features_path = os.path.join(
    ARTIFACT_DIRECTORY,
    "selected_features.json",
)

with open(
    selected_features_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        selected_features,
        file,
        ensure_ascii=False,
        indent=2,
    )

zone_categories_path = os.path.join(
    ARTIFACT_DIRECTORY,
    "zone_categories.json",
)

with open(
    zone_categories_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        zone_categories,
        file,
        ensure_ascii=False,
        indent=2,
    )

preprocessing_metadata = {
    "model_type": (
        "lightgbm_direct_multi_horizon"
    ),
    "prediction_strategy": (
        "direct_non_recursive"
    ),
    "target_column": TARGET_COLUMN,
    "zone_column": ZONE_COLUMN,
    "issue_datetime_column": (
        ISSUE_DATETIME_COLUMN
    ),
    "target_datetime_column": (
        TARGET_DATETIME_COLUMN
    ),
    "horizon_column": HORIZON_COLUMN,
    "minimum_forecast_horizon": (
        MIN_FORECAST_HORIZON
    ),
    "maximum_forecast_horizon": (
        MAX_FORECAST_HORIZON
    ),
    "forecast_days": FORECAST_DAYS,
    "categorical_features": (
        categorical_features_final
    ),
    "selected_features": selected_features,
    "mandatory_features": MANDATORY_FEATURES,
    "zone_categories": zone_categories,
    "best_iteration": BEST_ITERATION,
    "train_last_issue_datetime": str(
        train_data[
            ISSUE_DATETIME_COLUMN
        ].max()
    ),
    "train_last_target_datetime": str(
        train_data[
            TARGET_DATETIME_COLUMN
        ].max()
    ),
    "validation_first_issue_datetime": str(
        validation_data[
            ISSUE_DATETIME_COLUMN
        ].min()
    ),
    "validation_last_target_datetime": str(
        validation_data[
            TARGET_DATETIME_COLUMN
        ].max()
    ),
    "test_first_issue_datetime": str(
        test_data[
            ISSUE_DATETIME_COLUMN
        ].min()
    ),
}

metadata_path = os.path.join(
    ARTIFACT_DIRECTORY,
    "preprocessing_metadata.json",
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        preprocessing_metadata,
        file,
        ensure_ascii=False,
        indent=2,
    )

print(
    f"Artefacts créés dans : "
    f"{ARTIFACT_DIRECTORY}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. Enregistrement dans MLflow

# COMMAND ----------

# DBTITLE 1,MLflow tags
with mlflow.start_run(
    run_name=(
        "lgbm_direct_multi_horizon_quantile_168h"
    )
) as run:

    mlflow.log_params({
        "model_type": "LightGBM",
        "prediction_strategy": (
            "direct_non_recursive"
        ),
        "objective": (
            final_params["objective"]
        ),
        "learning_rate": (
            final_params["learning_rate"]
        ),
        "num_leaves": (
            final_params["num_leaves"]
        ),
        "min_child_samples": (
            final_params[
                "min_child_samples"
            ]
        ),
        "subsample": (
            final_params["subsample"]
        ),
        "colsample_bytree": (
            final_params[
                "colsample_bytree"
            ]
        ),
        "reg_alpha": (
            final_params["reg_alpha"]
        ),
        "reg_lambda": (
            final_params["reg_lambda"]
        ),
        "random_state": RANDOM_STATE,
        "best_iteration": BEST_ITERATION,
        "initial_feature_count": len(
            candidate_features
        ),
        "clean_feature_count": len(
            baseline_features
        ),
        "selected_feature_count": len(
            selected_features
        ),
        "minimum_forecast_horizon": (
            MIN_FORECAST_HORIZON
        ),
        "maximum_forecast_horizon": (
            MAX_FORECAST_HORIZON
        ),
        "forecast_days": FORECAST_DAYS,
        "target_column": TARGET_COLUMN,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": (
            VALIDATION_RATIO
        ),
        "test_ratio": TEST_RATIO,
        "n_walk_forward_folds": N_WALK_FORWARD_FOLDS,
        "walk_forward_test_days": WALK_FORWARD_TEST_DAYS,
        "quantiles": str(QUANTILES),
        "reconciliation_method": RECONCILIATION_METHOD,
    })

    mlflow.log_metrics({
        "train_validation_mae": (
            train_validation_metrics["mae"]
        ),
        "train_validation_rmse": (
            train_validation_metrics["rmse"]
        ),
        "train_validation_r2": (
            train_validation_metrics["r2"]
        ),
        "train_validation_mape": (
            train_validation_metrics["mape"]
        ),
        "train_validation_smape": (
            train_validation_metrics["smape"]
        ),
        "train_validation_wape": (
            train_validation_metrics["wape"]
        ),
        "train_validation_bias": (
            train_validation_metrics["bias"]
        ),
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mape": test_metrics["mape"],
        "test_smape": test_metrics["smape"],
        "test_wape": test_metrics["wape"],
        "test_bias": test_metrics["bias"],
        "baseline_validation_wape": (
            baseline_validation_metrics["wape"]
        ),
        "selected_validation_wape": float(
            chosen_result["wape"]
        ),
        "test_pinball_p10": pinball_loss(y_test, quantile_predictions_test[0.1], 0.1),
        "test_pinball_p50": pinball_loss(y_test, quantile_predictions_test[0.5], 0.5),
        "test_pinball_p90": pinball_loss(y_test, quantile_predictions_test[0.9], 0.9),
        "test_quantile_coverage_p10_p90": coverage,
        "wf_mean_wape": wf_results_df["wape"].mean(),
        "wf_std_wape": wf_results_df["wape"].std(),
        "wf_mean_mae": wf_results_df["mae"].mean(),
        "wf_mean_coverage": wf_results_df["coverage_p10_p90"].mean(),
    })

    for _, row in (
        metrics_by_forecast_day.iterrows()
    ):
        forecast_day = int(
            row["forecast_day"]
        )

        mlflow.log_metrics({
            f"test_day_{forecast_day}_mae": float(
                row["mae"]
            ),
            f"test_day_{forecast_day}_rmse": float(
                row["rmse"]
            ),
            f"test_day_{forecast_day}_mape": float(
                row["mape"]
            ),
            f"test_day_{forecast_day}_wape": float(
                row["wape"]
            ),
            f"test_day_{forecast_day}_bias": float(
                row["bias"]
            ),
        })

    mlflow.set_tags({
        "project": "energy_forecast",
        "forecast_type": (
            "zonal_multi_horizon"
        ),
        "forecast_horizon": (
            "1h_to_168h"
        ),
        "forecast_days": "7",
        "prediction_strategy": (
            "direct_non_recursive"
        ),
        "data_table": GOLD_TABLE,
        "split_strategy": (
            "walk_forward_backtesting"
        ),
        "feature_selection": (
            "lightgbm_gain_validation_wape"
        ),
        "quantile_forecasting": "P10_P50_P90",
        "reconciliation": "bottom_up",
    })

    mlflow.log_artifacts(
        ARTIFACT_DIRECTORY,
        artifact_path="analysis",
    )

    input_example_native = (
        X_train_validation
        .head(5)
        .copy()
    )

    input_example_mlflow = (
        prepare_mlflow_example(
            input_example_native
        )
    )

    signature_predictions = (
        final_model.predict(
            input_example_native
        )
    )

    signature = infer_signature(
        input_example_mlflow,
        signature_predictions,
    )

    # NOTE (nettoyage 2026-08-29): ajout de `registered_model_name`, absent
    # auparavant — sans cela, ce modèle n'était jamais inscrit au Model
    # Registry sous un nom stable, seulement loggé dans son run MLflow.
    model_info = mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example_mlflow,
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    run_id = run.info.run_id
    model_uri = model_info.model_uri

print("\n" + "=" * 80)
print("MODÈLE ENREGISTRÉ DANS MLFLOW")
print("=" * 80)
print(f"Run ID    : {run_id}")
print(f"Model URI : {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 23. Test de rechargement du modèle

# COMMAND ----------

reloaded_model = mlflow.sklearn.load_model(
    model_uri
)

reload_test_sample = (
    X_test.head(10).copy()
)

original_sample_predictions = (
    final_model.predict(
        reload_test_sample
    )
)

reloaded_sample_predictions = (
    reloaded_model.predict(
        reload_test_sample
    )
)

if not np.allclose(
    original_sample_predictions,
    reloaded_sample_predictions,
    rtol=1e-8,
    atol=1e-8,
):
    raise AssertionError(
        "Les prédictions du modèle rechargé "
        "diffèrent du modèle initial."
    )

print(
    "Test de rechargement MLflow réussi."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 24. Résumé final

# COMMAND ----------

# DBTITLE 1,Final summary
print("=" * 80)
print("RÉSUMÉ FINAL")
print("=" * 80)

print(f"Target                  : {TARGET_COLUMN}")
print(
    f"Horizons                : "
    f"H+{MIN_FORECAST_HORIZON} à "
    f"H+{MAX_FORECAST_HORIZON}"
)
print(
    f"Stratégie               : "
    f"directe, non récursive"
)
print(
    f"Features avant nettoyage: "
    f"{len(candidate_features)}"
)
print(
    f"Features après nettoyage: "
    f"{len(baseline_features)}"
)
print(
    f"Features sélectionnées  : "
    f"{len(selected_features)}"
)
print(
    f"Nombre optimal d'arbres : "
    f"{BEST_ITERATION}"
)
print(
    f"MAPE test               : "
    f"{test_metrics['mape']:.3f} %"
)
print(
    f"WAPE test               : "
    f"{test_metrics['wape']:.3f} %"
)
print(
    f"MAE test                : "
    f"{test_metrics['mae']:.3f} MW"
)
print(
    f"RMSE test               : "
    f"{test_metrics['rmse']:.3f} MW"
)
print(
    f"Biais test              : "
    f"{test_metrics['bias']:.3f} MW"
)
print(
    f"R² test                 : "
    f"{test_metrics['r2']:.5f}"
)
print(f"MLflow Run ID           : {run_id}")
print(f"Model URI               : {model_uri}")
print(f"Walk-forward             : {N_WALK_FORWARD_FOLDS} folds × {WALK_FORWARD_TEST_DAYS}j")
print(f"  WAPE moyen             : {wf_results_df['wape'].mean():.3f}% ± {wf_results_df['wape'].std():.3f}")
print(f"  MAE moyen              : {wf_results_df['mae'].mean():.1f} MW")
print(f"  Couverture P10-P90     : {wf_results_df['coverage_p10_p90'].mean():.1f}%")
print(f"Quantile forecasting     : P10/P50/P90")
print(f"  Couverture test         : {coverage:.1f}%")
print(f"  Largeur intervalle     : {interval_width:.1f} MW")
print(f"Réconciliation           : bottom-up (zones → Ontario)")

print("\nPerformances par jour :")

for _, row in (
    metrics_by_forecast_day.iterrows()
):
    print(
        f"J+{int(row['forecast_day'])} | "
        f"MAE={row['mae']:.2f} MW | "
        f"WAPE={row['wape']:.3f}% | "
        f"Biais={row['bias']:.2f} MW"
    )

print("\nTop 20 des features :")

for index, row in (
    final_importance
    .head(20)
    .iterrows()
):
    print(
        f"{index + 1:3d}. "
        f"{row['feature']:50s} "
        f"gain={row['gain_pct']:7.3f}%"
    )

print(
    "\nEntraînement multi-horizon "
    "terminé avec succès."
)