# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Entraînement et sélection de features LightGBM
# MAGIC
# MAGIC Ce notebook :
# MAGIC
# MAGIC 1. charge les features Gold ;
# MAGIC 2. effectue un nettoyage technique des features ;
# MAGIC 3. réalise un split strictement temporel ;
# MAGIC 4. entraîne un LightGBM de référence ;
# MAGIC 5. classe les features par gain ;
# MAGIC 6. compare plusieurs nombres de features ;
# MAGIC 7. entraîne le modèle final ;
# MAGIC 8. enregistre le modèle et les artefacts dans MLflow.

# COMMAND ----------

# MAGIC %pip install lightgbm -q

# COMMAND ----------

# Redémarrer Python si Databricks le demande après l'installation.
# dbutils.library.restartPython()

# COMMAND ----------

import os
import json
import yaml
import warnings

import mlflow
import mlflow.sklearn

import lightgbm as lgb
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

print(f"LightGBM version : {lgb.__version__}")
print(f"MLflow version   : {mlflow.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# DBTITLE 1,Configuration
CONFIG_PATH = "/Workspace/Users/n.jouglet23@gmail.com/energy_forecast_clean/config/config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

CATALOG = config["catalog"]["name"]
SCHEMA = config["catalog"]["schema"]

# NOTE: ml_features_gold_24h n'existe pas encore.
# On utilise ml_features_gold (multi-horizon) et on filtre sur forecast_horizon_hours = 24.
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.ml_features_gold"

# NOTE (nettoyage 2026-08-29): utilisait auparavant `config["model"]`, partagé
# à l'identique avec le modèle 7 jours (07_train_model_7j.py) -> les deux
# entraînements écrivaient dans la même expérience MLflow, rendant impossible
# de savoir quel modèle charger ensuite. On utilise maintenant la config
# dédiée `models.horizon_24h`.
MODEL_CONFIG = config["models"]["horizon_24h"]
MLFLOW_EXPERIMENT = MODEL_CONFIG["mlflow"]["experiment_name"]
REGISTERED_MODEL_NAME = MODEL_CONFIG["mlflow"]["registry_model_name"]
mlflow.set_experiment(MLFLOW_EXPERIMENT)

RANDOM_STATE = 42

# Proportions temporelles.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Présélection technique.
MAX_MISSING_RATE = 0.30
NEAR_CONSTANT_THRESHOLD = 0.999

# Nombres de features à tester.
FEATURE_COUNTS_TO_TEST = [
    20,
    40,
    60,
    80,
    100,
    120,
]

# Tolérance par rapport au meilleur MAPE.
# Exemple : un modèle situé à moins de 0,05 point de pourcentage du meilleur
# peut être choisi s'il utilise moins de features.
MAPE_TOLERANCE_PERCENTAGE_POINT = 0.05

# Colonnes connues représentant la cible.
POSSIBLE_TARGET_COLUMNS = [
    "target_demand_mw",
    "target_24h_ahead",
]

print("=" * 80)
print("ENTRAÎNEMENT LIGHTGBM")
print(f"Table Gold       : {GOLD_TABLE}")
print(f"Expérience MLflow: {MLFLOW_EXPERIMENT}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chargement des données

# COMMAND ----------

print("Chargement de la table Gold...")

spark_df = spark.table(GOLD_TABLE)

# Filtrer sur l'horizon 24h si la colonne forecast_horizon_hours existe.
if "forecast_horizon_hours" in spark_df.columns:
    print("Filtrage sur forecast_horizon_hours = 24...")
    spark_df = spark_df.filter(spark_df["forecast_horizon_hours"] == 24)

print(f"Nombre de lignes Spark   : {spark_df.count():,}")
print(f"Nombre de colonnes Spark : {len(spark_df.columns):,}")

available_target_columns = [
    column
    for column in POSSIBLE_TARGET_COLUMNS
    if column in spark_df.columns
]

if not available_target_columns:
    raise ValueError(
        "Aucune colonne cible trouvée. "
        f"Colonnes attendues : {POSSIBLE_TARGET_COLUMNS}"
    )

# Priorité à target_demand_mw si elle existe.
if "target_demand_mw" in available_target_columns:
    TARGET_COLUMN = "target_demand_mw"
else:
    TARGET_COLUMN = "target_24h_ahead"

print(f"Colonne cible sélectionnée : {TARGET_COLUMN}")

if "zone" not in spark_df.columns:
    raise ValueError(
        "La colonne 'zone' est absente de la table Gold."
    )

# Déterminer la colonne temporelle principale.
if "issue_datetime" in spark_df.columns:
    DATETIME_COLUMN = "issue_datetime"
elif "datetime" in spark_df.columns:
    DATETIME_COLUMN = "datetime"
else:
    raise ValueError(
        "Aucune colonne temporelle trouvée. "
        "Une colonne 'issue_datetime' ou 'datetime' est requise."
    )

print(f"Colonne temporelle sélectionnée : {DATETIME_COLUMN}")

# Limiter immédiatement les lignes inutilisables pour la cible.
spark_training_df = spark_df.filter(
    spark_df[TARGET_COLUMN].isNotNull()
)

# Si la nouvelle table Gold contient is_training_row, l'utiliser.
if "is_training_row" in spark_training_df.columns:
    spark_training_df = spark_training_df.filter(
        spark_training_df["is_training_row"] == 1
    )

training_row_count = spark_training_df.count()

if training_row_count == 0:
    raise ValueError(
        "Aucune ligne valide pour l'entraînement."
    )

print(f"Lignes candidates pour training : {training_row_count:,}")

# Conversion Pandas.
# Pour un projet Databricks Free, cette approche convient si les données
# tiennent en mémoire sur le driver.
print("Conversion en Pandas...")

pd_data = spark_training_df.toPandas()

pd_data[DATETIME_COLUMN] = pd.to_datetime(
    pd_data[DATETIME_COLUMN],
    errors="coerce"
)

pd_data = pd_data[
    pd_data[DATETIME_COLUMN].notna()
    & pd_data[TARGET_COLUMN].notna()
].copy()

pd_data = pd_data.sort_values(
    [DATETIME_COLUMN, "zone"]
).reset_index(drop=True)

print(f"Dataset Pandas : {len(pd_data):,} lignes")
print(f"Période        : {pd_data[DATETIME_COLUMN].min()}")
print(f"              à {pd_data[DATETIME_COLUMN].max()}")
print(f"Zones          : {sorted(pd_data['zone'].dropna().unique())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Définition des features candidates
# MAGIC
# MAGIC Les colonnes techniques, les identifiants temporels bruts et toutes les
# MAGIC cibles possibles sont exclus. Les variables cycliques restent incluses.

# COMMAND ----------

EXCLUDED_COLUMNS = {
    # Cibles.
    "target_demand_mw",
    "target_24h_ahead",

    # Dates et identifiants bruts.
    "datetime",
    "issue_datetime",
    "target_datetime",
    "issue_date",
    "target_date",
    "date",
    "processed_time",

    # Colonnes techniques.
    "has_valid_features",
    "has_valid_target",
    "is_training_row",
    "forecast_horizon_hours",  # Déjà utilisé pour filtrer, pas une feature

    # Demande courante laissée en référence.
    # Retirer cette colonne de l'ensemble uniquement si elle est certainement
    # disponible au moment de chaque prédiction.
    "demand_mw",
}

# Colonnes temporelles brutes potentiellement redondantes.
# Les versions sin/cos et les indicateurs calendaires sont conservés.
RAW_CALENDAR_COLUMNS = {
    "year",
    "month",
    "day",
    "hour",
    "day_of_year",
    "day_of_week",
    "issue_year",
    "issue_month",
    "issue_day",
    "issue_hour",
    "issue_day_of_year",
    "issue_day_of_week",
    "target_year",
    "target_month",
    "target_day",
    "target_hour",
    "target_day_of_year",
    "target_day_of_week",
}

EXCLUDED_COLUMNS.update(RAW_CALENDAR_COLUMNS)

# Zone est conservée comme variable catégorielle.
candidate_features = [
    column
    for column in pd_data.columns
    if column not in EXCLUDED_COLUMNS
    and column != TARGET_COLUMN
]

if "zone" not in candidate_features:
    candidate_features.append("zone")

print(f"Features candidates initiales : {len(candidate_features)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Protection contre les fuites de données

# COMMAND ----------

# Noms de colonnes suspects.
# Certaines weather_target_* sont légitimes si elles proviennent de véritables
# prévisions météo disponibles au moment de l'émission.
#
# Si elles proviennent de la météo réellement observée 24 heures plus tard,
# elles créent un avantage irréaliste lors de l'évaluation.
LEAKAGE_PATTERNS = [
    "lead_demand",
    "future_demand",
    "actual_future",
    "target_24h_ahead",
    "target_demand_mw",
]

suspected_leakage_features = [
    column
    for column in candidate_features
    if any(
        pattern.lower() in column.lower()
        for pattern in LEAKAGE_PATTERNS
    )
]

if suspected_leakage_features:
    print("Features supprimées pour risque de fuite :")
    for feature in suspected_leakage_features:
        print(f"  - {feature}")

candidate_features = [
    column
    for column in candidate_features
    if column not in suspected_leakage_features
]

weather_target_features = [
    column
    for column in candidate_features
    if column.startswith("weather_target_")
]

if weather_target_features:
    print(
        "\nAttention : des features weather_target_* sont présentes."
    )
    print(
        "Elles doivent représenter la prévision météo disponible à "
        "issue_datetime pour target_datetime."
    )
    print(
        "Si elles correspondent à la météo réellement observée dans le futur, "
        "le score de test sera optimiste."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Présélection technique
# MAGIC
# MAGIC Cette étape supprime uniquement :
# MAGIC
# MAGIC - les features entièrement manquantes ;
# MAGIC - les features ayant plus de 30 % de valeurs manquantes ;
# MAGIC - les features constantes ;
# MAGIC - les features quasi constantes ;
# MAGIC - les colonnes texte non prises en charge, sauf `zone`.

# COMMAND ----------

feature_quality_rows = []

for feature in candidate_features:
    series = pd_data[feature]

    missing_rate = float(series.isna().mean())
    unique_count = int(series.nunique(dropna=True))

    if len(series.dropna()) > 0:
        dominant_rate = float(
            series.value_counts(
                normalize=True,
                dropna=True
            ).iloc[0]
        )
    else:
        dominant_rate = 1.0

    feature_quality_rows.append({
        "feature": feature,
        "dtype": str(series.dtype),
        "missing_rate": missing_rate,
        "unique_count": unique_count,
        "dominant_rate": dominant_rate,
    })

feature_quality = pd.DataFrame(feature_quality_rows)

features_all_missing = feature_quality.loc[
    feature_quality["missing_rate"] >= 1.0,
    "feature",
].tolist()

features_high_missing = feature_quality.loc[
    (
        feature_quality["missing_rate"] > MAX_MISSING_RATE
    )
    & (
        feature_quality["missing_rate"] < 1.0
    ),
    "feature",
].tolist()

features_constant = feature_quality.loc[
    feature_quality["unique_count"] <= 1,
    "feature",
].tolist()

features_near_constant = feature_quality.loc[
    (
        feature_quality["dominant_rate"] >= NEAR_CONSTANT_THRESHOLD
    )
    & (
        feature_quality["unique_count"] > 1
    ),
    "feature",
].tolist()

object_features = [
    column
    for column in candidate_features
    if (
        pd.api.types.is_object_dtype(pd_data[column])
        or pd.api.types.is_string_dtype(pd_data[column])
    )
    and column != "zone"
]

technical_features_to_remove = set(
    features_all_missing
    + features_high_missing
    + features_constant
    + features_near_constant
    + object_features
)

selected_candidate_features = [
    feature
    for feature in candidate_features
    if feature not in technical_features_to_remove
]

print(f"Features entièrement manquantes : {len(features_all_missing)}")
print(f"Features trop manquantes         : {len(features_high_missing)}")
print(f"Features constantes              : {len(features_constant)}")
print(f"Features quasi constantes        : {len(features_near_constant)}")
print(f"Autres colonnes texte retirées   : {len(object_features)}")
print(
    f"Features après nettoyage         : "
    f"{len(selected_candidate_features)}"
)

if "zone" not in selected_candidate_features:
    selected_candidate_features.append("zone")

if len(selected_candidate_features) == 0:
    raise ValueError(
        "Aucune feature restante après la présélection technique."
    )

display(
    feature_quality.sort_values(
        ["missing_rate", "dominant_rate"],
        ascending=[False, False],
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Préparation des types
# MAGIC
# MAGIC LightGBM peut utiliser directement une colonne Pandas de type `category`.
# MAGIC Cela évite d'imposer un ordre numérique arbitraire aux zones.

# COMMAND ----------

model_data = pd_data[
    [DATETIME_COLUMN, TARGET_COLUMN]
    + selected_candidate_features
].copy()

# Zone catégorielle native.
model_data["zone"] = (
    model_data["zone"]
    .astype("string")
    .fillna("UNKNOWN")
    .astype("category")
)

# Conversion des booléens en entiers.
boolean_features = [
    column
    for column in selected_candidate_features
    if pd.api.types.is_bool_dtype(model_data[column])
]

for column in boolean_features:
    model_data[column] = model_data[column].astype("int8")

# Conversion des autres features en numérique.
numeric_features = [
    column
    for column in selected_candidate_features
    if column != "zone"
]

for column in numeric_features:
    model_data[column] = pd.to_numeric(
        model_data[column],
        errors="coerce"
    )

model_data[TARGET_COLUMN] = pd.to_numeric(
    model_data[TARGET_COLUMN],
    errors="coerce"
)

model_data = model_data[
    model_data[TARGET_COLUMN].notna()
].copy()

model_data = model_data.sort_values(
    [DATETIME_COLUMN, "zone"]
).reset_index(drop=True)

print(f"Lignes après préparation : {len(model_data):,}")
print(f"Features finales candidates : {len(selected_candidate_features)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Split temporel strict

# COMMAND ----------

unique_datetimes = np.sort(
    model_data[DATETIME_COLUMN].dropna().unique()
)

if len(unique_datetimes) < 100:
    raise ValueError(
        "Pas assez de timestamps pour réaliser un split temporel robuste."
    )

train_end_index = int(
    len(unique_datetimes) * TRAIN_RATIO
)

validation_end_index = int(
    len(unique_datetimes)
    * (TRAIN_RATIO + VALIDATION_RATIO)
)

train_end_datetime = unique_datetimes[train_end_index - 1]
validation_start_datetime = unique_datetimes[train_end_index]
validation_end_datetime = unique_datetimes[validation_end_index - 1]
test_start_datetime = unique_datetimes[validation_end_index]

train_mask = (
    model_data[DATETIME_COLUMN] <= train_end_datetime
)

validation_mask = (
    (model_data[DATETIME_COLUMN] >= validation_start_datetime)
    & (model_data[DATETIME_COLUMN] <= validation_end_datetime)
)

test_mask = (
    model_data[DATETIME_COLUMN] >= test_start_datetime
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
        "Au moins un jeu temporel est vide."
    )

print("=" * 70)
print("SPLIT TEMPOREL")
print("=" * 70)

print(
    f"Train      : {len(train_data):,} lignes | "
    f"{train_data[DATETIME_COLUMN].min()} à "
    f"{train_data[DATETIME_COLUMN].max()}"
)

print(
    f"Validation : {len(validation_data):,} lignes | "
    f"{validation_data[DATETIME_COLUMN].min()} à "
    f"{validation_data[DATETIME_COLUMN].max()}"
)

print(
    f"Test       : {len(test_data):,} lignes | "
    f"{test_data[DATETIME_COLUMN].min()} à "
    f"{test_data[DATETIME_COLUMN].max()}"
)

# Vérification explicite de l'absence de chevauchement.
assert (
    train_data[DATETIME_COLUMN].max()
    < validation_data[DATETIME_COLUMN].min()
)

assert (
    validation_data[DATETIME_COLUMN].max()
    < test_data[DATETIME_COLUMN].min()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Fonctions utilitaires

# COMMAND ----------

def calculate_metrics(y_true, y_pred):
    """
    Calcule les métriques globales de régression.

    WAPE est généralement plus stable que MAPE lorsque certaines zones ont
    une faible demande.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

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
        }

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    r2 = r2_score(y_true, y_pred)

    non_zero_mask = np.abs(y_true) > 1e-6

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

    smape_mask = smape_denominator > 1e-6

    if smape_mask.any():
        smape = (
            np.mean(
                2.0
                * np.abs(
                    y_true[smape_mask]
                    - y_pred[smape_mask]
                )
                / smape_denominator[smape_mask]
            )
            * 100.0
        )
    else:
        smape = np.nan

    absolute_target_sum = np.sum(
        np.abs(y_true)
    )

    if absolute_target_sum > 1e-6:
        wape = (
            np.sum(np.abs(y_true - y_pred))
            / absolute_target_sum
            * 100.0
        )
    else:
        wape = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "smape": float(smape),
        "wape": float(wape),
    }


def print_metrics(name, metrics):
    """Affiche les métriques proprement."""

    print(f"\n{name}")
    print("-" * 50)
    print(f"MAE   : {metrics['mae']:.3f} MW")
    print(f"RMSE  : {metrics['rmse']:.3f} MW")
    print(f"R²    : {metrics['r2']:.5f}")
    print(f"MAPE  : {metrics['mape']:.3f} %")
    print(f"sMAPE : {metrics['smape']:.3f} %")
    print(f"WAPE  : {metrics['wape']:.3f} %")


def calculate_metrics_by_zone(
    source_data,
    predictions,
    target_column,
):
    """Calcule les métriques séparément pour chaque zone."""

    result = source_data[
        ["zone", DATETIME_COLUMN, target_column]
    ].copy()

    result["prediction_mw"] = predictions
    result["error_mw"] = (
        result["prediction_mw"]
        - result[target_column]
    )
    result["absolute_error_mw"] = np.abs(
        result["error_mw"]
    )

    rows = []

    for zone, zone_data in result.groupby(
        "zone",
        observed=True,
    ):
        metrics = calculate_metrics(
            zone_data[target_column],
            zone_data["prediction_mw"],
        )

        rows.append({
            "zone": str(zone),
            "observations": len(zone_data),
            **metrics,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("wape")
        .reset_index(drop=True)
    )


def train_lightgbm(
    train_frame,
    validation_frame,
    features,
    target_column,
    seed=42,
):
    """Entraîne un modèle LightGBM avec early stopping."""

    X_train_local = train_frame[features].copy()
    y_train_local = train_frame[target_column].copy()

    X_validation_local = validation_frame[features].copy()
    y_validation_local = validation_frame[target_column].copy()

    categorical_features = [
        feature
        for feature in features
        if str(X_train_local[feature].dtype) == "category"
    ]

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=3000,
        learning_rate=0.03,
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
        X_train_local,
        y_train_local,
        eval_set=[
            (
                X_validation_local,
                y_validation_local,
            )
        ],
        eval_metric="mae",
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    return model


def get_feature_importance(model, features):
    """Retourne les importances gain et split."""

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

    importance["cumulative_gain_pct"] = (
        importance["gain_pct"].cumsum()
    )

    return importance


def save_dataframe_csv(dataframe, path):
    """Sauvegarde un DataFrame en CSV."""

    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    return path


def convert_categories_to_string_for_signature(frame):
    """
    MLflow peut rencontrer des difficultés avec certaines catégories Pandas.
    Cette fonction prépare uniquement l'exemple et la signature.
    """

    output = frame.copy()

    for column in output.columns:
        if str(output[column].dtype) == "category":
            output[column] = output[column].astype(str)

    return output

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Modèle baseline avec toutes les features nettoyées

# COMMAND ----------

baseline_features = selected_candidate_features.copy()

print(f"Entraînement baseline avec {len(baseline_features)} features...")

baseline_model = train_lightgbm(
    train_frame=train_data,
    validation_frame=validation_data,
    features=baseline_features,
    target_column=TARGET_COLUMN,
    seed=RANDOM_STATE,
)

baseline_validation_predictions = baseline_model.predict(
    validation_data[baseline_features],
    num_iteration=baseline_model.best_iteration_,
)

baseline_validation_metrics = calculate_metrics(
    validation_data[TARGET_COLUMN],
    baseline_validation_predictions,
)

print_metrics(
    "BASELINE VALIDATION",
    baseline_validation_metrics,
)

print(
    f"\nMeilleure itération : "
    f"{baseline_model.best_iteration_}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Importance des features du modèle baseline

# COMMAND ----------

baseline_importance = get_feature_importance(
    baseline_model,
    baseline_features,
)

display(baseline_importance.head(50))

fig, ax = plt.subplots(figsize=(12, 10))

top_importance = (
    baseline_importance
    .head(30)
    .sort_values("gain", ascending=True)
)

ax.barh(
    top_importance["feature"],
    top_importance["gain"],
)

ax.set_xlabel("Importance par gain")
ax.set_ylabel("Feature")
ax.set_title(
    "Top 30 des features LightGBM par gain"
)

plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Comparaison de plusieurs nombres de features

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

# Toujours conserver zone pour comparer les zones.
def get_top_features_with_zone(feature_count):
    top_features = ranked_features[:feature_count]

    if (
        "zone" in baseline_features
        and "zone" not in top_features
    ):
        if len(top_features) >= feature_count:
            top_features = top_features[:-1]

        top_features.append("zone")

    return list(dict.fromkeys(top_features))


selection_results = []
selection_models = {}

print(
    f"Configurations testées : {feature_counts}"
)

for feature_count in feature_counts:
    current_features = get_top_features_with_zone(
        feature_count
    )

    current_model = train_lightgbm(
        train_frame=train_data,
        validation_frame=validation_data,
        features=current_features,
        target_column=TARGET_COLUMN,
        seed=RANDOM_STATE,
    )

    validation_predictions = current_model.predict(
        validation_data[current_features],
        num_iteration=current_model.best_iteration_,
    )

    current_metrics = calculate_metrics(
        validation_data[TARGET_COLUMN],
        validation_predictions,
    )

    selection_results.append({
        "requested_feature_count": feature_count,
        "actual_feature_count": len(current_features),
        "best_iteration": current_model.best_iteration_,
        **current_metrics,
    })

    selection_models[len(current_features)] = {
        "model": current_model,
        "features": current_features,
    }

    print(
        f"{len(current_features):3d} features | "
        f"MAPE={current_metrics['mape']:.3f}% | "
        f"WAPE={current_metrics['wape']:.3f}% | "
        f"MAE={current_metrics['mae']:.3f} MW"
    )

selection_results_df = (
    pd.DataFrame(selection_results)
    .sort_values("actual_feature_count")
    .reset_index(drop=True)
)

display(selection_results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Choix du nombre final de features
# MAGIC
# MAGIC On sélectionne le plus petit ensemble dont le MAPE est situé à moins
# MAGIC de la tolérance configurée par rapport au meilleur MAPE.

# COMMAND ----------

best_validation_mape = (
    selection_results_df["mape"].min()
)

acceptable_results = selection_results_df[
    selection_results_df["mape"]
    <= (
        best_validation_mape
        + MAPE_TOLERANCE_PERCENTAGE_POINT
    )
].copy()

chosen_result = (
    acceptable_results
    .sort_values(
        [
            "actual_feature_count",
            "mape",
        ]
    )
    .iloc[0]
)

CHOSEN_FEATURE_COUNT = int(
    chosen_result["actual_feature_count"]
)

selected_features = selection_models[
    CHOSEN_FEATURE_COUNT
]["features"]

print("=" * 70)
print("SÉLECTION DES FEATURES")
print("=" * 70)
print(
    f"Meilleur MAPE validation : "
    f"{best_validation_mape:.3f} %"
)
print(
    f"Tolérance                : "
    f"{MAPE_TOLERANCE_PERCENTAGE_POINT:.3f} point"
)
print(
    f"Nombre choisi            : "
    f"{CHOSEN_FEATURE_COUNT}"
)
print("\nFeatures sélectionnées :")

for index, feature in enumerate(
    selected_features,
    start=1,
):
    print(f"{index:3d}. {feature}")

# COMMAND ----------

# Visualisation performance contre complexité.

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    selection_results_df["actual_feature_count"],
    selection_results_df["mape"],
    marker="o",
)

ax.axhline(
    best_validation_mape,
    linestyle="--",
    label="Meilleur MAPE",
)

ax.scatter(
    [CHOSEN_FEATURE_COUNT],
    [chosen_result["mape"]],
    s=120,
    label="Configuration choisie",
)

ax.set_xlabel("Nombre de features")
ax.set_ylabel("MAPE validation (%)")
ax.set_title(
    "Performance selon le nombre de features"
)
ax.legend()

plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Réentraînement final sur train + validation
# MAGIC
# MAGIC Le meilleur nombre d'itérations a été déterminé avec la validation.
# MAGIC Le modèle final est ensuite réentraîné sur train + validation.
# MAGIC Le test reste totalement isolé jusqu'à l'évaluation finale.

# COMMAND ----------

selected_validation_model = selection_models[
    CHOSEN_FEATURE_COUNT
]["model"]

BEST_ITERATION = int(
    selected_validation_model.best_iteration_
)

train_validation_data = pd.concat(
    [train_data, validation_data],
    axis=0,
).sort_values(
    [DATETIME_COLUMN, "zone"]
).reset_index(drop=True)

X_train_validation = train_validation_data[
    selected_features
].copy()

y_train_validation = train_validation_data[
    TARGET_COLUMN
].copy()

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
        X_train_validation[feature].dtype
    ) == "category"
]

final_params = {
    "objective": "regression_l1",
    "n_estimators": BEST_ITERATION,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.10,
    "reg_lambda": 1.00,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "importance_type": "gain",
    "verbosity": -1,
}

print(
    f"Réentraînement final avec "
    f"{len(selected_features)} features et "
    f"{BEST_ITERATION} arbres..."
)

final_model = lgb.LGBMRegressor(
    **final_params
)

final_model.fit(
    X_train_validation,
    y_train_validation,
    categorical_feature=categorical_features_final,
    callbacks=[
        lgb.log_evaluation(period=0)
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Évaluation finale sur le test

# COMMAND ----------

train_validation_predictions = final_model.predict(
    X_train_validation
)

test_predictions = final_model.predict(
    X_test
)

train_validation_metrics = calculate_metrics(
    y_train_validation,
    train_validation_predictions,
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

# Résultats détaillés.
test_predictions_df = test_data[
    [
        DATETIME_COLUMN,
        "zone",
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

test_predictions_df["absolute_error_mw"] = np.abs(
    test_predictions_df["error_mw"]
)

test_predictions_df["absolute_percentage_error"] = np.where(
    np.abs(test_predictions_df[TARGET_COLUMN]) > 1e-6,
    (
        test_predictions_df["absolute_error_mw"]
        / np.abs(
            test_predictions_df[TARGET_COLUMN]
        )
        * 100.0
    ),
    np.nan,
)

display(test_predictions_df.head(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Métriques par zone

# COMMAND ----------

metrics_by_zone = calculate_metrics_by_zone(
    source_data=test_data,
    predictions=test_predictions,
    target_column=TARGET_COLUMN,
)

display(metrics_by_zone)

print("\nRésumé par zone :")

for _, row in metrics_by_zone.iterrows():
    print(
        f"{row['zone']:15s} | "
        f"MAE={row['mae']:9.2f} MW | "
        f"MAPE={row['mape']:7.3f}% | "
        f"WAPE={row['wape']:7.3f}% | "
        f"R²={row['r2']:8.4f}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Importance finale

# COMMAND ----------

final_importance = get_feature_importance(
    final_model,
    selected_features,
)

display(final_importance)

fig, ax = plt.subplots(figsize=(12, 10))

top_final_importance = (
    final_importance
    .head(30)
    .sort_values("gain", ascending=True)
)

ax.barh(
    top_final_importance["feature"],
    top_final_importance["gain"],
)

ax.set_xlabel("Importance par gain")
ax.set_ylabel("Feature")
ax.set_title(
    "Top 30 des features du modèle final"
)

plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Visualisation réel contre prédit

# COMMAND ----------

# Agrégation de toutes les zones pour une vue globale.
global_test_plot = (
    test_predictions_df
    .groupby(DATETIME_COLUMN, as_index=False)
    .agg(
        actual_demand_mw=(TARGET_COLUMN, "sum"),
        predicted_demand_mw=("prediction_mw", "sum"),
    )
    .sort_values(DATETIME_COLUMN)
)

# Limiter le graphique aux 14 derniers jours pour rester lisible.
maximum_plot_datetime = global_test_plot[
    DATETIME_COLUMN
].max()

minimum_plot_datetime = (
    maximum_plot_datetime
    - pd.Timedelta(days=14)
)

global_test_plot_last_days = global_test_plot[
    global_test_plot[DATETIME_COLUMN]
    >= minimum_plot_datetime
]

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(
    global_test_plot_last_days[DATETIME_COLUMN],
    global_test_plot_last_days["actual_demand_mw"],
    label="Demande réelle",
)

ax.plot(
    global_test_plot_last_days[DATETIME_COLUMN],
    global_test_plot_last_days["predicted_demand_mw"],
    label="Demande prédite",
)

ax.set_xlabel("Date et heure")
ax.set_ylabel("Demande totale (MW)")
ax.set_title(
    "Demande réelle contre demande prédite, 14 derniers jours du test"
)
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Enregistrement MLflow

# COMMAND ----------

ARTIFACT_DIRECTORY = "/tmp/lightgbm_energy_forecast"
os.makedirs(ARTIFACT_DIRECTORY, exist_ok=True)

# Artefacts CSV.
feature_quality_path = os.path.join(
    ARTIFACT_DIRECTORY,
    "feature_quality.csv",
)

baseline_importance_path = os.path.join(
    ARTIFACT_DIRECTORY,
    "baseline_feature_importance.csv",
)

final_importance_path = os.path.join(
    ARTIFACT_DIRECTORY,
    "final_feature_importance.csv",
)

selection_results_path = os.path.join(
    ARTIFACT_DIRECTORY,
    "feature_selection_results.csv",
)

metrics_by_zone_path = os.path.join(
    ARTIFACT_DIRECTORY,
    "test_metrics_by_zone.csv",
)

test_predictions_path = os.path.join(
    ARTIFACT_DIRECTORY,
    "test_predictions.csv",
)

save_dataframe_csv(
    feature_quality,
    feature_quality_path,
)

save_dataframe_csv(
    baseline_importance,
    baseline_importance_path,
)

save_dataframe_csv(
    final_importance,
    final_importance_path,
)

save_dataframe_csv(
    selection_results_df,
    selection_results_path,
)

save_dataframe_csv(
    metrics_by_zone,
    metrics_by_zone_path,
)

save_dataframe_csv(
    test_predictions_df,
    test_predictions_path,
)

# Liste des features en JSON.
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

# Catégories de zones.
zone_categories = (
    model_data["zone"]
    .cat.categories
    .astype(str)
    .tolist()
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

# Métadonnées de préparation.
preprocessing_metadata = {
    "target_column": TARGET_COLUMN,
    "datetime_column": DATETIME_COLUMN,
    "categorical_features": categorical_features_final,
    "selected_features": selected_features,
    "zone_categories": zone_categories,
    "max_missing_rate": MAX_MISSING_RATE,
    "near_constant_threshold": NEAR_CONSTANT_THRESHOLD,
    "train_end_datetime": str(
        train_data[DATETIME_COLUMN].max()
    ),
    "validation_end_datetime": str(
        validation_data[DATETIME_COLUMN].max()
    ),
    "test_start_datetime": str(
        test_data[DATETIME_COLUMN].min()
    ),
    "best_iteration": BEST_ITERATION,
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

# COMMAND ----------

# MLflow peut journaliser le modèle LightGBM via l'interface sklearn.
# L'exemple d'entrée conserve les mêmes colonnes que le modèle.

with mlflow.start_run(
    run_name="lgbm_zonal_forecast_feature_selection"
) as run:

    # Paramètres fondamentaux.
    mlflow.log_params({
        "model_type": "LightGBM",
        "objective": final_params["objective"],
        "learning_rate": final_params["learning_rate"],
        "num_leaves": final_params["num_leaves"],
        "min_child_samples": final_params["min_child_samples"],
        "subsample": final_params["subsample"],
        "colsample_bytree": final_params["colsample_bytree"],
        "reg_alpha": final_params["reg_alpha"],
        "reg_lambda": final_params["reg_lambda"],
        "random_state": RANDOM_STATE,
        "best_iteration": BEST_ITERATION,
        "initial_feature_count": len(candidate_features),
        "clean_feature_count": len(baseline_features),
        "selected_feature_count": len(selected_features),
        "max_missing_rate": MAX_MISSING_RATE,
        "target_column": TARGET_COLUMN,
        "datetime_column": DATETIME_COLUMN,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
    })

    # Métriques.
    metrics_to_log = {
        "train_validation_mae": train_validation_metrics["mae"],
        "train_validation_rmse": train_validation_metrics["rmse"],
        "train_validation_r2": train_validation_metrics["r2"],
        "train_validation_mape": train_validation_metrics["mape"],
        "train_validation_smape": train_validation_metrics["smape"],
        "train_validation_wape": train_validation_metrics["wape"],

        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mape": test_metrics["mape"],
        "test_smape": test_metrics["smape"],
        "test_wape": test_metrics["wape"],

        "baseline_validation_mape": (
            baseline_validation_metrics["mape"]
        ),
        "selected_validation_mape": float(
            chosen_result["mape"]
        ),
    }

    mlflow.log_metrics(metrics_to_log)

    # Tags permettant de retrouver le modèle.
    mlflow.set_tags({
        "project": "energy_forecast",
        "forecast_type": "zonal",
        "forecast_horizon": "24h",
        "data_table": GOLD_TABLE,
        "split_strategy": "strict_temporal_train_validation_test",
        "feature_selection": "lightgbm_gain_validation_mape",
    })

    # Artefacts.
    mlflow.log_artifacts(
        ARTIFACT_DIRECTORY,
        artifact_path="analysis",
    )

    # Signature.
    input_example = X_train_validation.head(5).copy()

    # Pour l'inférence, la colonne zone doit garder un type compatible.
    signature_input = convert_categories_to_string_for_signature(
        input_example
    )

    signature_predictions = final_model.predict(
        input_example
    )

    signature = infer_signature(
        signature_input,
        signature_predictions,
    )

    # NOTE (nettoyage 2026-08-29): `registered_model_name` manquait ici, donc
    # aucun modèle n'était jamais réellement inscrit au Model Registry — le
    # `client.get_latest_versions(MODEL_NAME, stages=["Production"])` de
    # 09_batch_prediction.py ne pouvait donc jamais réussir et retombait
    # systématiquement sur "dernier run". On l'ajoute pour permettre le
    # vrai flux Registry -> stage "Production".
    model_info = mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        signature=signature,
        input_example=signature_input,
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    run_id = run.info.run_id

    print("\n" + "=" * 80)
    print("MODÈLE ENREGISTRÉ DANS MLFLOW")
    print("=" * 80)
    print(f"Run ID       : {run_id}")
    print(f"Model URI    : {model_info.model_uri}")
    print(f"Test MAE     : {test_metrics['mae']:.3f} MW")
    print(f"Test RMSE    : {test_metrics['rmse']:.3f} MW")
    print(f"Test MAPE    : {test_metrics['mape']:.3f} %")
    print(f"Test sMAPE   : {test_metrics['smape']:.3f} %")
    print(f"Test WAPE    : {test_metrics['wape']:.3f} %")
    print(f"Test R²      : {test_metrics['r2']:.5f}")
    print(
        f"Features     : "
        f"{len(selected_features)} sur "
        f"{len(baseline_features)}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Validation finale

# COMMAND ----------

print("=" * 80)
print("RÉSUMÉ FINAL")
print("=" * 80)

print(f"Target                    : {TARGET_COLUMN}")
print(f"Features avant nettoyage  : {len(candidate_features)}")
print(f"Features après nettoyage  : {len(baseline_features)}")
print(f"Features sélectionnées    : {len(selected_features)}")
print(f"Nombre optimal d'arbres   : {BEST_ITERATION}")
print(f"MAPE test                 : {test_metrics['mape']:.3f} %")
print(f"WAPE test                 : {test_metrics['wape']:.3f} %")
print(f"MAE test                  : {test_metrics['mae']:.3f} MW")
print(f"RMSE test                 : {test_metrics['rmse']:.3f} MW")
print(f"R² test                   : {test_metrics['r2']:.5f}")
print(f"MLflow Run ID             : {run_id}")
print(f"Model URI                 : {model_info.model_uri}")

print("\nTop 20 des features finales :")

for index, row in final_importance.head(20).iterrows():
    print(
        f"{index + 1:3d}. "
        f"{row['feature']:50s} "
        f"gain={row['gain_pct']:7.3f}%"
    )

print("\nEntraînement terminé avec succès.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Étape suivante :** adapter `09_batch_prediction.py` pour :
# MAGIC
# MAGIC - charger `selected_features.json` ;
# MAGIC - créer exactement les mêmes colonnes ;
# MAGIC - appliquer les mêmes catégories à `zone` ;
# MAGIC - charger le modèle à partir de son URI MLflow ;
# MAGIC - écrire les prédictions dans une table Delta.