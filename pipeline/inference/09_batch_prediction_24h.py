#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# 09 - Batch Predictions 24h (Récursif, Itératif)

Prédictions itératives sur **24 heures** avec le modèle MLflow "Production" (ou,
à défaut, le dernier run de l'expérience). Chaque heure prédite alimente les
features de demande (lags, rolling) de l'heure suivante.

**Approche récursive :**
- Le modèle 24h (entraîné par `06_train_model_24hv2`) est utilisé de façon
  itérative pour générer H+1, H+2, ..., H+24.
- Chaque prédiction devient une "observation" pour calculer les features de
  l'heure suivante (demand_lag_1h, rolling windows, etc.).

**Pour des prévisions à 7 jours (H+168) :**
Utilisez le notebook `09b_batch_prediction_7j_direct` qui utilise le modèle
7 jours avec une approche directe (non-récursive) basée sur
`forecast_horizon_hours`.

**Next:**
- `10_model_evaluation.py` (Monitoring)
- `09b_batch_prediction_7j_direct` pour des prévisions à 7 jours (H+168)
"""

import subprocess
subprocess.run(["pip", "install", "lightgbm", "-q"], check=True)

import os
import yaml
import json
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

# ============================================================
# CONFIGURATION
# ============================================================

# Chemin du projet : surchargeable via ENERGY_FORECAST_PROJECT_ROOT.
PROJECT_ROOT = os.environ.get(
    "ENERGY_FORECAST_PROJECT_ROOT",
    "/Workspace/Users/n.jouglet23@gmail.com/energy_forecast",
)

with open(f'{PROJECT_ROOT}/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

CATALOG = config['catalog']['name']
SCHEMA = config['catalog']['schema']
# Table de prédictions 24h (récursif)
FORECAST_TABLE = f"{CATALOG}.{SCHEMA}.load_forecast_24h"
SHAP_TABLE = f"{CATALOG}.{SCHEMA}.load_shap_24h"
# Modèle utilisé pour la boucle récursive (voir avertissement ci-dessus).
# Changer pour "horizon_7j" uniquement si le modèle long-terme a été entraîné
# avec les mêmes features récursives (lags courts) que le modèle 24h — ce qui
# n'est PAS le cas du modèle produit par 07_train_model_7j.py aujourd'hui.
PREDICTION_MODEL_KEY = "horizon_24h"
MLFLOW_EXPERIMENT = config['models'][PREDICTION_MODEL_KEY]['mlflow']['experiment_name']
MODEL_NAME = config['models'][PREDICTION_MODEL_KEY]['mlflow']['registry_model_name']

print("=" * 80)
print("09 - BATCH PREDICTIONS 24H (Récursif, Itératif H+1 à H+24)")
print("=" * 80)
print(f"Modèle       : {MODEL_NAME}")
print(f"Expérience   : {MLFLOW_EXPERIMENT}")
print(f"Table sortie : {FORECAST_TABLE}")

# ============================================================
# 1. CHARGEMENT DES FEATURES DEPUIS UNITY CATALOG
# ============================================================

print("\n[1/5] Chargement des features depuis Unity Catalog...")

# Tables de features
FEATURE_DEMAND_TABLE = f"{CATALOG}.{SCHEMA}.feature_demand_history"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.feature_weather_forecast"
METADATA_TABLE = f"{CATALOG}.{SCHEMA}.feature_metadata"

# Charger les métadonnées de référence
print("  Chargement des métadonnées...")

# Vérifier que les tables de features existent
try:
    spark.sql(f"DESCRIBE TABLE {METADATA_TABLE}").collect()
    table_exists = True
except Exception:
    table_exists = False

if not table_exists:
    raise RuntimeError(
        f"\n{'='*80}\n"
        f"ERREUR: La table {METADATA_TABLE} n'existe pas.\n"
        f"{'='*80}\n\n"
        f"Vous devez d'abord exécuter 08_build_prediction_features.py pour:\n"
        f"  1. Télécharger les données IESO et météo (340h+ d'historique)\n"
        f"  2. Calculer toutes les features (lags, rolling windows, etc.)\n"
        f"  3. Sauvegarder dans les tables Unity Catalog\n\n"
        f"Tables requises:\n"
        f"  - {FEATURE_DEMAND_TABLE} (historique brut pour itération)\n"
        f"  - {FEATURE_TABLE} (toutes les features assemblées)\n"
        f"  - {METADATA_TABLE} (timestamps de référence)\n\n"
        f"Commande: Exécutez 08_build_prediction_features.py\n"
        f"{'='*80}"
    )

metadata_df = spark.table(METADATA_TABLE).toPandas()
if metadata_df.empty:
    raise ValueError(
        f"Aucune métadonnée trouvée dans {METADATA_TABLE}. "
        "Exécutez d'abord 08_build_prediction_features.py."
    )

ref_time = pd.to_datetime(metadata_df['ref_time'].iloc[0])
prediction_start = pd.to_datetime(metadata_df['prediction_start'].iloc[0])
prediction_end = pd.to_datetime(metadata_df['prediction_end'].iloc[0])

print(f"    ref_time: {ref_time}")
print(f"    Fenêtre: {prediction_start} -> {prediction_end}")

# Charger l'historique de demande brut (pour la prédiction itérative)
print(f"  Chargement de {FEATURE_DEMAND_TABLE}...")
demand_history = spark.table(FEATURE_DEMAND_TABLE).toPandas()
print(f"    {len(demand_history):,} lignes (historique brut IESO)")

# Charger les features complètes (météo + demande + historique météo)
print(f"  Chargement de {FEATURE_TABLE}...")
weather_forecast_24h = spark.table(FEATURE_TABLE).toPandas()
print(f"    {len(weather_forecast_24h):,} lignes x {len(weather_forecast_24h.columns)} colonnes")

# Charger MODEL_FEATURES et MODEL_ZONE_CATEGORIES depuis le modèle
print("  Chargement des artefacts du modèle...")
mlflow.set_experiment(MLFLOW_EXPERIMENT)
client = mlflow.tracking.MlflowClient()

try:
    latest_version = client.get_latest_versions(
        MODEL_NAME, stages=["Production"]
    )[0]
    run_id = latest_version.run_id
except Exception:
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(
            f"Aucun run trouvé dans {MLFLOW_EXPERIMENT}. "
            "Entraînez d'abord le modèle 24h."
        )
    run_id = runs.iloc[0]['run_id']

try:
    artifact_path = client.download_artifacts(run_id, "analysis")
    with open(f"{artifact_path}/selected_features.json", 'r') as f:
        MODEL_FEATURES = json.load(f)
    with open(f"{artifact_path}/zone_categories.json", 'r') as f:
        MODEL_ZONE_CATEGORIES = json.load(f)
    print(f"    Features: {len(MODEL_FEATURES)}")
    print(f"    Zones: {MODEL_ZONE_CATEGORIES}")
except Exception as e:
    raise ValueError(
        f"Impossible de charger les artefacts du modèle: {e}. "
        "Vérifiez que le modèle a été entraîné avec les bons artefacts."
    )

# ============================================================
# 2. CHARGEMENT DU MODÈLE LightGBM
# ============================================================

print("\n[2/5] Chargement du modèle...")

forecast_start = prediction_start
forecast_end = prediction_end

try:
    latest_version = client.get_latest_versions(
        MODEL_NAME, stages=["Production"]
    )[0]
    model_uri = f"models:/{MODEL_NAME}/Production"
    print(
        f"  Modèle Production: version {latest_version.version}"
    )
except Exception:
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs.iloc[0]['run_id']
    model_uri = f"runs:/{run_id}/model"
    print(
        f"  Pas de modèle en Production, dernier run: {run_id}"
    )

model = mlflow.sklearn.load_model(model_uri)
print(f"  Modèle chargé: {type(model).__name__}")

# ============================================================
# 3. PRÉDICTION ITÉRATIVE SUR 24 HEURES
# ============================================================

print("\n[3/5] Prédiction itérative sur 24h...")

LAG_HOURS = [1, 24, 48, 72, 144, 168, 336]
ROLLING_WINDOWS = [3, 6, 12, 24, 48, 72, 168]  # 6h ajouté pour demand_rolling_min_6h

predictions_all = []
shap_all = []

available_zones = sorted(demand_history['zone'].unique())

for zone_name in available_zones:
    # --- Initialisation de la série de demande ---
    # Historique IESO + prédictions au fur et à mesure.
    zone_hist = (
        demand_history[demand_history['zone'] == zone_name]
        .sort_values('datetime')
        .drop_duplicates(subset=['datetime'], keep='last')
    )

    full_range = pd.date_range(
        start=zone_hist['datetime'].min(),
        end=ref_time,
        freq='h',
    )

    demand_series = (
        zone_hist
        .set_index('datetime')['demand_mw']
        .astype(float)
        .reindex(full_range)
    )
    demand_series = demand_series.interpolate(
        method='time', limit=2, limit_area='inside',
    )

    # --- Prévisions météo pour cette zone (limitées à 24h) ---
    zone_weather = (
    weather_forecast_24h
    [weather_forecast_24h['zone'].astype(str) == zone_name]
    .sort_values('target_datetime')
    .reset_index(drop=True)
)

    for _, row in zone_weather.iterrows():
        target_dt = row['target_datetime']
        current_ref = target_dt - pd.Timedelta(hours=1)

        # --- Features de demande calculées à current_ref ---
        d_ref = demand_series.get(current_ref, np.nan)

        feat = {}

        # Lags stricts.
        for lag in LAG_HOURS:
            feat[f'demand_lag_{lag}h'] = demand_series.get(
                current_ref - pd.Timedelta(hours=lag), np.nan,
            )

        # Fenêtres glissantes terminées à current_ref.
        for window in ROLLING_WINDOWS:
            window_start = current_ref - pd.Timedelta(
                hours=window - 1,
            )
            window_data = demand_series.loc[window_start:current_ref]

            feat[f'demand_rolling_min_{window}h'] = window_data.min()
            feat[f'demand_rolling_max_{window}h'] = window_data.max()
            feat[f'demand_rolling_mean_{window}h'] = window_data.mean()
            feat[f'demand_rolling_std_{window}h'] = window_data.std()

        # Variations.
        d_1 = demand_series.get(
            current_ref - pd.Timedelta(hours=1), np.nan,
        )
        d_24 = demand_series.get(
            current_ref - pd.Timedelta(hours=24), np.nan,
        )
        d_168 = demand_series.get(
            current_ref - pd.Timedelta(hours=168), np.nan,
        )

        feat['demand_change_1h'] = d_ref - d_1
        feat['demand_change_24h'] = d_ref - d_24
        feat['demand_change_168h'] = d_ref - d_168

        if (
            pd.notna(d_ref)
            and pd.notna(d_1)
            and abs(d_1) > 1e-6
        ):
            feat['demand_pct_change_1h'] = (
                (d_ref - d_1) / d_1 * 100.0
            )
        else:
            feat['demand_pct_change_1h'] = np.nan

        # demand_pct_change_24h
        if (
            pd.notna(d_ref)
            and pd.notna(d_24)
            and abs(d_24) > 1e-6
        ):
            feat['demand_pct_change_24h'] = (
                (d_ref - d_24) / d_24 * 100.0
            )
        else:
            feat['demand_pct_change_24h'] = np.nan

        # demand_ratio_24h_168h
        if (
            pd.notna(d_24)
            and pd.notna(d_168)
            and abs(d_168) > 1e-6
        ):
            feat['demand_ratio_24h_168h'] = d_24 / d_168
        else:
            feat['demand_ratio_24h_168h'] = np.nan

        feat['demand_vs_rolling_mean_24h'] = (
            d_ref - feat.get('demand_rolling_mean_24h', np.nan)
        )
        feat['demand_vs_rolling_mean_168h'] = (
            d_ref - feat.get('demand_rolling_mean_168h', np.nan)
        )

        # --- Assemblage du vecteur de features ---
        # Demand features: valeurs itératives.
        # Autres features (météo, calendrier): depuis la ligne de prévision.
        feature_row = {}
        for col in MODEL_FEATURES:
            if col in feat:
                feature_row[col] = feat[col]
            elif col == 'zone':
                feature_row[col] = zone_name
            elif col in row.index:
                feature_row[col] = row[col]
            else:
                feature_row[col] = np.nan

        X = pd.DataFrame([feature_row])[MODEL_FEATURES]
        X['zone'] = pd.Categorical(
            X['zone'], categories=MODEL_ZONE_CATEGORIES,
        )



        # --- Prédiction ---
        y_pred = model.predict(X)[0]

        predictions_all.append({
            'zone': zone_name,
            'target_datetime': target_dt,
            'predicted_demand_mw': float(y_pred),
        })

        # --- SHAP values (LightGBM natif) ---
        try:
            shap_contrib = model.predict(X, pred_contrib=True)[0]
            base_value = float(shap_contrib[-1])
            for feat_name, shap_val in zip(
                MODEL_FEATURES, shap_contrib[:-1],
            ):
                shap_all.append({
                    'zone': zone_name,
                    'target_datetime': target_dt,
                    'feature_name': feat_name,
                    'shap_value': float(shap_val),
                    'base_value': base_value,
                })
        except (TypeError, AttributeError):
            print(
                f"  SHAP non disponible pour "
                f"{type(model).__name__}"
            )

        # --- Injection de la prédiction dans la série ---
        # La prédiction devient la "demande réelle" pour les itérations
        # suivantes (lag_1h, rolling windows, etc.).
        demand_series[target_dt] = y_pred

    print(
        f"  {zone_name}: {len(zone_weather)} prédictions (24h)"
    )

predictions_df = pd.DataFrame(predictions_all)
predictions_df['prediction_time'] = datetime.now()
predictions_df['target_date'] = pd.to_datetime(predictions_df['target_datetime']).dt.date

shap_df = pd.DataFrame(shap_all)
shap_df['prediction_time'] = datetime.now()
shap_df['target_date'] = pd.to_datetime(shap_df['target_datetime']).dt.date

print(f"\nTotal: {len(predictions_df)} prédictions")
print(
    f"Zones: {sorted(predictions_df['zone'].unique())}"
)
print(
    f"Période: {predictions_df['target_datetime'].min()}"
    f" -> {predictions_df['target_datetime'].max()}"
)

print("\nAperçu (premières prédictions):")
print(predictions_df.head(20).to_string())

# ============================================================
# 4. ÉCRITURE DES RÉSULTATS
# ============================================================

print(f"\n[4/5] Écriture dans {FORECAST_TABLE}...")

spark_df = spark.createDataFrame(predictions_df)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FORECAST_TABLE} (
        zone STRING,
        target_datetime TIMESTAMP,
        predicted_demand_mw DOUBLE,
        prediction_time TIMESTAMP,
        target_date DATE
    )
    USING DELTA
    PARTITIONED BY (target_date)
""")

spark_df.write.format("delta").mode("overwrite").saveAsTable(
    FORECAST_TABLE,
)

print(f"Prédictions écrites dans {FORECAST_TABLE}")

# --- Écriture de la table SHAP ---
if not shap_df.empty:
    print(f"\nÉcriture des SHAP values dans {SHAP_TABLE}...")

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {SHAP_TABLE} (
            zone STRING,
            target_datetime TIMESTAMP,
            feature_name STRING,
            shap_value DOUBLE,
            base_value DOUBLE,
            prediction_time TIMESTAMP,
            target_date DATE
        )
        USING DELTA
        PARTITIONED BY (target_date)
    """)

    shap_spark_df = spark.createDataFrame(shap_df)
    shap_spark_df.write.format("delta").mode("overwrite").saveAsTable(
        SHAP_TABLE,
    )

    print(f"SHAP values écrites dans {SHAP_TABLE}")
    print(
        f"  {len(shap_df):,} lignes "
        f"({len(predictions_df)} prédictions x "
        f"{len(MODEL_FEATURES)} features)"
    )

    # Top 10 features par importance SHAP moyenne.
    spark.sql(f"""
        SELECT
            feature_name,
            ROUND(AVG(ABS(shap_value)), 4) as avg_abs_shap,
            ROUND(AVG(shap_value), 4) as avg_shap
        FROM {SHAP_TABLE}
        GROUP BY feature_name
        ORDER BY avg_abs_shap DESC
        LIMIT 10
    """).show()
else:
    print("\nSHAP values non disponibles pour ce modèle.")

# Stats par zone.
print("\n[5/5] Statistiques finales...")
spark.sql(f"""
    SELECT
        zone,
        COUNT(*) as total,
        ROUND(AVG(predicted_demand_mw), 1) as avg_pred,
        ROUND(MIN(predicted_demand_mw), 1) as min_pred,
        ROUND(MAX(predicted_demand_mw), 1) as max_pred,
        MIN(target_datetime) as min_date,
        MAX(target_datetime) as max_date
    FROM {FORECAST_TABLE}
    GROUP BY zone
    ORDER BY zone
""").show()

print("\n" + "=" * 80)
print("✅ PRÉDICTIONS 24H TERMINÉES")
print("=" * 80)
print(f"Table de sortie : {FORECAST_TABLE}")
print(f"Total prédictions : {len(predictions_df):,}")
print(f"Période : {forecast_start} -> {forecast_end}")
