# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,En-tête : Prédictions directes multi-horizon 7j
# MAGIC %md
# MAGIC # 09b - Batch Predictions Multi-Horizon 7j (Direct, Non-Récursif)
# MAGIC
# MAGIC Ce notebook fait des prédictions **DIRECTES** pour H+1 à H+168 en utilisant le
# MAGIC modèle 7 jours entraîné avec `forecast_horizon_hours` comme feature.
# MAGIC
# MAGIC **Différence clé avec 09_batch_prediction :**
# MAGIC - **Approche directe** : Le modèle 7j prédit directement chaque horizon H+1 à
# MAGIC   H+168 en une seule passe par zone, sans récursion.
# MAGIC - **forecast_horizon_hours** est une feature du modèle, pas une itération.
# MAGIC - **Pas de mise à jour des lags** : Les features de demande (lags, rolling) sont
# MAGIC   calculées UNE fois à `ref_time` et réutilisées pour tous les horizons.
# MAGIC - Chaque zone génère 168 prédictions simultanées (une par heure future).
# MAGIC
# MAGIC **Architecture :**
# MAGIC 1. Charger les features de base (demande historique + météo)
# MAGIC 2. Charger le modèle 7j depuis MLflow
# MAGIC 3. Pour chaque zone : créer un DataFrame de 168 lignes (H+1 à H+168)
# MAGIC 4. Prédire en une seule fois avec `model.predict(X_168_hours)`
# MAGIC 5. Écrire dans `load_forecast_7j` (table distincte du modèle 24h)

# COMMAND ----------

# DBTITLE 1,Installation de LightGBM
import subprocess
subprocess.run(["pip", "install", "lightgbm", "-q"], check=True)

# COMMAND ----------

# DBTITLE 1,Configuration et imports
import os
import yaml
import json
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql import functions as F

# ============================================================
# CONFIGURATION
# ============================================================

# Chemin du projet : surchargeable via ENERGY_FORECAST_PROJECT_ROOT.
PROJECT_ROOT = os.environ.get(
    "ENERGY_FORECAST_PROJECT_ROOT",
    "/Workspace/Users/n.jouglet23@gmail.com/energy_forecast_clean",
)

with open(f'{PROJECT_ROOT}/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

CATALOG = config['catalog']['name']
SCHEMA = config['catalog']['schema']

# Tables de sortie spécifiques au modèle 7j (distinctes du modèle 24h)
FORECAST_TABLE = f"{CATALOG}.{SCHEMA}.load_forecast_7j"
SHAP_TABLE = f"{CATALOG}.{SCHEMA}.load_shap_7j"

# Modèle 7 jours multi-horizon direct
PREDICTION_MODEL_KEY = "horizon_7j"
MLFLOW_EXPERIMENT = config['models'][PREDICTION_MODEL_KEY]['mlflow']['experiment_name']
MODEL_NAME = config['models'][PREDICTION_MODEL_KEY]['mlflow']['registry_model_name']

print("=" * 80)
print("09b - BATCH PREDICTIONS 7J (Direct, Non-Récursif, H+1 à H+168)")
print("=" * 80)
print(f"Modèle       : {MODEL_NAME}")
print(f"Expérience   : {MLFLOW_EXPERIMENT}")
print(f"Table sortie : {FORECAST_TABLE}")

# COMMAND ----------

# DBTITLE 1,1. Construction des features de base
print("\n[1/5] Construction des features de base...")

# Exécution de 08_build_prediction_features.py pour obtenir :
# - demand_history : historique de demande par zone
# - weather_forecast_24h : prévisions météo sur 168h
# - ref_time : timestamp de référence pour les prédictions
exec(
    open(
        f'{PROJECT_ROOT}/pipeline/inference/08b_build_prediction_features_7j.py'
    ).read()
)

print(f"  ref_time : {ref_time}")
print(f"  Zones disponibles : {sorted(demand_history['zone'].unique())}")
print(f"  Prévisions météo : {len(weather_forecast_24h)} lignes")

# COMMAND ----------

# DBTITLE 1,2. Chargement du modèle 7j depuis MLflow
print("\n[2/5] Chargement du modèle 7j...")

mlflow.set_experiment(MLFLOW_EXPERIMENT)
client = mlflow.tracking.MlflowClient()

# Essayer d'abord Production, sinon le dernier run
try:
    latest_version = client.get_latest_versions(
        MODEL_NAME, stages=["Production"]
    )[0]
    model_uri = f"models:/{MODEL_NAME}/Production"
    run_id = latest_version.run_id
    print(f"  Modèle Production : version {latest_version.version}")
except Exception:
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(
            f"Aucun run trouvé dans l'expérience {MLFLOW_EXPERIMENT}. "
            "Entraînez d'abord le modèle 7j avec 07_train_model_7j."
        )
    run_id = runs.iloc[0]['run_id']
    model_uri = f"runs:/{run_id}/model"
    print(f"  Pas de modèle en Production, dernier run : {run_id}")

# Chargement du modèle
model = mlflow.sklearn.load_model(model_uri)
print(f"  Modèle chargé : {type(model).__name__}")

# Récupération des artefacts (features et catégories de zones)
try:
    artifact_path = client.download_artifacts(run_id, "analysis")
    
    with open(f"{artifact_path}/selected_features.json", 'r') as f:
        MODEL_FEATURES = json.load(f)
    
    with open(f"{artifact_path}/zone_categories.json", 'r') as f:
        MODEL_ZONE_CATEGORIES = json.load(f)
    
    print(f"  Features : {len(MODEL_FEATURES)}")
    print(f"  Zones catégorielles : {MODEL_ZONE_CATEGORIES}")
except Exception as e:
    raise ValueError(
        f"Impossible de charger les artefacts du modèle : {e}. "
        "Vérifiez que le modèle 7j a bien été entraîné avec 07_train_model_7j."
    )

# COMMAND ----------

# DBTITLE 1,3. Calcul des features de demande à ref_time
print("\n[3/5] Calcul des features de demande à ref_time...")

# Définition des features de demande (lags et rolling windows)
LAG_HOURS = [1, 24, 48, 72, 144, 168, 336]
ROLLING_WINDOWS = [3, 12, 24, 48, 72, 168]

# Calculer les features de demande pour chaque zone UNE FOIS à ref_time
# Ces features seront réutilisées pour tous les horizons H+1 à H+168
zone_demand_features = {}

available_zones = sorted(demand_history['zone'].unique())

for zone_name in available_zones:
    # Historique de demande pour cette zone
    zone_hist = (
        demand_history[demand_history['zone'] == zone_name]
        .sort_values('datetime')
        .drop_duplicates(subset=['datetime'], keep='last')
    )

    # Créer une série temporelle continue jusqu'à ref_time
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

    # Calculer les features de demande à ref_time
    feat = {}
    d_ref = demand_series.get(ref_time, np.nan)

    # Lags stricts
    for lag in LAG_HOURS:
        feat[f'demand_lag_{lag}h'] = demand_series.get(
            ref_time - pd.Timedelta(hours=lag), np.nan,
        )

    # Fenêtres glissantes terminées à ref_time
    for window in ROLLING_WINDOWS:
        window_start = ref_time - pd.Timedelta(hours=window - 1)
        window_data = demand_series.loc[window_start:ref_time]

        feat[f'demand_rolling_min_{window}h'] = window_data.min()
        feat[f'demand_rolling_max_{window}h'] = window_data.max()
        feat[f'demand_rolling_mean_{window}h'] = window_data.mean()
        feat[f'demand_rolling_std_{window}h'] = window_data.std()

    # Variations
    d_1 = demand_series.get(ref_time - pd.Timedelta(hours=1), np.nan)
    d_24 = demand_series.get(ref_time - pd.Timedelta(hours=24), np.nan)
    d_168 = demand_series.get(ref_time - pd.Timedelta(hours=168), np.nan)

    feat['demand_change_1h'] = d_ref - d_1
    feat['demand_change_24h'] = d_ref - d_24
    feat['demand_change_168h'] = d_ref - d_168

    if pd.notna(d_ref) and pd.notna(d_1) and abs(d_1) > 1e-6:
        feat['demand_pct_change_1h'] = (d_ref - d_1) / d_1 * 100.0
    else:
        feat['demand_pct_change_1h'] = np.nan

    feat['demand_vs_rolling_mean_24h'] = (
        d_ref - feat.get('demand_rolling_mean_24h', np.nan)
    )
    feat['demand_vs_rolling_mean_168h'] = (
        d_ref - feat.get('demand_rolling_mean_168h', np.nan)
    )

    zone_demand_features[zone_name] = feat

print(f"  Features de demande calculées pour {len(zone_demand_features)} zones")

# COMMAND ----------

# DBTITLE 1,4. Prédiction DIRECTE multi-horizon (H+1 à H+168)
print("\n[4/5] Prédiction DIRECTE multi-horizon (H+1 à H+168)...")

predictions_all = []
shap_all = []

for zone_name in available_zones:
    # Récupérer les features de demande pré-calculées pour cette zone
    zone_demand_feat = zone_demand_features[zone_name]

    # Prévisions météo pour cette zone
    zone_weather = (
        weather_forecast_24h
        [weather_forecast_24h['zone'].astype(str) == zone_name]
        .sort_values('target_datetime')
        .reset_index(drop=True)
    )

    if zone_weather.empty:
        print(f"  ⚠️  {zone_name} : pas de prévisions météo disponibles")
        continue

    # Créer un DataFrame avec 168 lignes (une par horizon H+1 à H+168)
    rows_168h = []

    for _, weather_row in zone_weather.iterrows():
        target_dt = weather_row['target_datetime']
        forecast_horizon = int(
            (target_dt - ref_time).total_seconds() / 3600
        )

        if forecast_horizon < 1 or forecast_horizon > 168:
            continue

        # Calculer forecast_day et forecast_hour_in_day
        forecast_day = (forecast_horizon - 1) // 24 + 1
        forecast_hour_in_day = (forecast_horizon - 1) % 24 + 1

        # Assembler la ligne de features
        row = {
            'target_datetime': target_dt,
            'forecast_horizon_hours': forecast_horizon,
            'forecast_day': forecast_day,
            'forecast_hour_in_day': forecast_hour_in_day,
            'zone': zone_name,
        }

        # Ajouter les features de demande (identiques pour tous les horizons)
        row.update(zone_demand_feat)

        # Ajouter les features météo et calendrier
        for col in weather_row.index:
            if col not in ['target_datetime', 'zone']:
                row[col] = weather_row[col]

        rows_168h.append(row)

    if not rows_168h:
        print(f"  ⚠️  {zone_name} : aucun horizon valide (H+1 à H+168)")
        continue

    # Créer le DataFrame de prédiction (jusqu'à 168 lignes)
    pred_df = pd.DataFrame(rows_168h)

    # S'assurer que toutes les features du modèle sont présentes
    for feat in MODEL_FEATURES:
        if feat not in pred_df.columns:
            pred_df[feat] = np.nan

    # Ordonner les colonnes selon MODEL_FEATURES
    X = pred_df[MODEL_FEATURES].copy()

    # Convertir la zone en catégorielle
    X['zone'] = pd.Categorical(
        X['zone'], categories=MODEL_ZONE_CATEGORIES,
    )

    # PRÉDICTION DIRECTE : une seule fois pour tous les horizons
    y_pred = model.predict(X)

    # Stocker les prédictions
    for i, pred_value in enumerate(y_pred):
        predictions_all.append({
            'zone': zone_name,
            'target_datetime': pred_df.iloc[i]['target_datetime'],
            'forecast_horizon_hours': int(
                pred_df.iloc[i]['forecast_horizon_hours']
            ),
            'predicted_demand_mw': float(pred_value),
        })

    # SHAP values (si supporté par le modèle LightGBM)
    try:
        shap_contrib = model.predict(X, pred_contrib=True)
        base_value = float(shap_contrib[0, -1])

        for i in range(len(shap_contrib)):
            for feat_idx, feat_name in enumerate(MODEL_FEATURES):
                shap_all.append({
                    'zone': zone_name,
                    'target_datetime': pred_df.iloc[i]['target_datetime'],
                    'feature_name': feat_name,
                    'shap_value': float(shap_contrib[i, feat_idx]),
                    'base_value': base_value,
                })
    except (TypeError, AttributeError):
        pass  # SHAP non disponible pour ce type de modèle

    print(f"  {zone_name} : {len(y_pred)} prédictions (H+1 à H+{len(y_pred)})")

if not predictions_all:
    raise ValueError(
        "Aucune prédiction générée. Vérifiez les données météo et la configuration."
    )

predictions_df = pd.DataFrame(predictions_all)
predictions_df['prediction_time'] = datetime.now()
predictions_df['target_date'] = pd.to_datetime(
    predictions_df['target_datetime']
).dt.date

shap_df = pd.DataFrame(shap_all)
if not shap_df.empty:
    shap_df['prediction_time'] = datetime.now()
    shap_df['target_date'] = pd.to_datetime(
        shap_df['target_datetime']
    ).dt.date

print(f"\nTotal : {len(predictions_df)} prédictions")
print(f"Zones : {sorted(predictions_df['zone'].unique())}")
print(
    f"Période : {predictions_df['target_datetime'].min()} "
    f"-> {predictions_df['target_datetime'].max()}"
)
print(
    f"Horizons : H+{predictions_df['forecast_horizon_hours'].min()} "
    f"à H+{predictions_df['forecast_horizon_hours'].max()}"
)

# COMMAND ----------

# DBTITLE 1,5. Écriture des résultats dans Unity Catalog
print(f"\n[5/5] Écriture dans {FORECAST_TABLE}...")

# Conversion en Spark DataFrame avec cast explicite des types
spark_df = spark.createDataFrame(predictions_df)
spark_df = spark_df.withColumn(
    "forecast_horizon_hours",
    F.col("forecast_horizon_hours").cast("int")
)

# Création de la table si elle n'existe pas
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FORECAST_TABLE} (
        zone STRING,
        target_datetime TIMESTAMP,
        forecast_horizon_hours INT,
        predicted_demand_mw DOUBLE,
        prediction_time TIMESTAMP,
        target_date DATE
    )
    USING DELTA
    PARTITIONED BY (target_date)
""")

# Écriture en mode overwrite
spark_df.write.format("delta").mode("overwrite").saveAsTable(
    FORECAST_TABLE,
)

print(f"✓ Prédictions écrites dans {FORECAST_TABLE}")

# Écriture de la table SHAP si disponible
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

    print(f"✓ SHAP values écrites dans {SHAP_TABLE}")
    print(
        f"  {len(shap_df):,} lignes "
        f"({len(predictions_df)} prédictions x "
        f"{len(MODEL_FEATURES)} features)"
    )
else:
    print("\n⚠️  SHAP values non disponibles pour ce modèle.")

# COMMAND ----------

# DBTITLE 1,Statistiques et aperçu des résultats
print("\n" + "=" * 80)
print("STATISTIQUES DES PRÉDICTIONS 7J (DIRECTES)")
print("=" * 80)

# Aperçu des premières prédictions
print("\nAperçu (20 premières lignes) :")
print(
    predictions_df
    .sort_values(['zone', 'forecast_horizon_hours'])
    .head(20)
    .to_string(index=False)
)

# Stats par zone
print("\nStatistiques par zone :")
spark.sql(f"""
    SELECT
        zone,
        COUNT(*) as total_predictions,
        ROUND(AVG(predicted_demand_mw), 1) as avg_demand_mw,
        ROUND(MIN(predicted_demand_mw), 1) as min_demand_mw,
        ROUND(MAX(predicted_demand_mw), 1) as max_demand_mw,
        MIN(forecast_horizon_hours) as min_horizon,
        MAX(forecast_horizon_hours) as max_horizon,
        MIN(target_datetime) as first_target,
        MAX(target_datetime) as last_target
    FROM {FORECAST_TABLE}
    GROUP BY zone
    ORDER BY zone
""").show(truncate=False)

# Stats par jour de prévision (forecast_day 1 à 7)
print("\nStatistiques par jour de prévision :")
spark.sql(f"""
    SELECT
        CAST((forecast_horizon_hours - 1) / 24 + 1 AS INT) as forecast_day,
        COUNT(*) as total_predictions,
        ROUND(AVG(predicted_demand_mw), 1) as avg_demand_mw,
        ROUND(STDDEV(predicted_demand_mw), 1) as std_demand_mw,
        MIN(forecast_horizon_hours) as min_horizon,
        MAX(forecast_horizon_hours) as max_horizon
    FROM {FORECAST_TABLE}
    GROUP BY forecast_day
    ORDER BY forecast_day
""").show()

# Top 10 features SHAP (si disponible)
if not shap_df.empty:
    print("\nTop 10 features par importance SHAP moyenne :")
    spark.sql(f"""
        SELECT
            feature_name,
            ROUND(AVG(ABS(shap_value)), 4) as avg_abs_shap,
            ROUND(AVG(shap_value), 4) as avg_shap
        FROM {SHAP_TABLE}
        GROUP BY feature_name
        ORDER BY avg_abs_shap DESC
        LIMIT 10
    """).show(truncate=False)

print("\n" + "=" * 80)
print("✓ Prédictions 7j DIRECTES terminées avec succès")
print("=" * 80)