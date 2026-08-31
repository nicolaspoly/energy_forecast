# Databricks notebook source
# MAGIC %md
# MAGIC # 10 - Model Evaluation & Monitoring
# MAGIC
# MAGIC Évalue la performance des prédictions batch en continu, en comparant
# MAGIC `load_forecast_gold` (prédictions) aux valeurs réelles observées ensuite
# MAGIC dans `load_zonal_bronze`.
# MAGIC
# MAGIC **Corrections (nettoyage 2026-08-29):**
# MAGIC - `config['catalog']['tables']['gold']['demand_forecast']` n'existait pas
# MAGIC   dans `config.yaml` (KeyError garanti) → remplacé par `load_forecast`.
# MAGIC - Le script précédent lisait une seule table supposée contenir à la fois
# MAGIC   `load_mw` (réel) et `predicted_load_mw` (prédit) sur une série globale.
# MAGIC   Ce n'est plus le schéma réel : `load_forecast_gold` ne contient que les
# MAGIC   prédictions par zone (`zone`, `target_datetime`, `predicted_demand_mw`).
# MAGIC   Il faut donc une jointure explicite avec `load_zonal_bronze` (réel par
# MAGIC   zone) pour pouvoir calculer une erreur.

# COMMAND ----------

# DBTITLE 1,Configuration
import yaml
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

with open('/Workspace/Users/nicolasjouglet@laposte.net/energy_forecast/energy_forecast_clean/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

CATALOG = config['catalog']['name']
SCHEMA = config['catalog']['schema']

FORECAST_TABLE = f"{CATALOG}.{SCHEMA}.{config['catalog']['tables']['gold']['load_forecast']}"
ACTUAL_TABLE = f"{CATALOG}.{SCHEMA}.{config['catalog']['tables']['bronze']['load_zonal']}"
MONITORING_TABLE = f"{CATALOG}.{SCHEMA}.{config['catalog']['tables']['gold']['model_performance']}"

print("📊 Évaluation performance modèle")
print(f"Prédictions : {FORECAST_TABLE}")
print(f"Réel        : {ACTUAL_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jointure prédictions ↔ réel

# COMMAND ----------

# On ne peut évaluer que les prédictions dont l'échéance (target_datetime)
# est déjà passée, c'est-à-dire pour lesquelles la donnée réelle a été ingérée.
df = spark.sql(f"""
    SELECT
        f.zone,
        f.target_datetime,
        f.prediction_time,
        f.predicted_demand_mw,
        a.demand_mw AS actual_demand_mw,
        CAST(
            (unix_timestamp(f.target_datetime) - unix_timestamp(f.prediction_time)) / 3600
            AS INT
        ) AS horizon_hours
    FROM {FORECAST_TABLE} f
    INNER JOIN {ACTUAL_TABLE} a
        ON f.zone = a.zone AND f.target_datetime = a.datetime
    WHERE f.target_datetime <= current_timestamp()
""").toPandas()

print(f"Prédictions évaluables (réel disponible) : {len(df):,}")

if df.empty:
    dbutils.notebook.exit("Aucune prédiction évaluable pour le moment (pas encore de réel disponible).")

print(f"Période: {df['target_datetime'].min()} à {df['target_datetime'].max()}")
print(f"Zones: {sorted(df['zone'].unique())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Métriques globales

# COMMAND ----------

mae = mean_absolute_error(df['actual_demand_mw'], df['predicted_demand_mw'])
rmse = np.sqrt(mean_squared_error(df['actual_demand_mw'], df['predicted_demand_mw']))
mape = mean_absolute_percentage_error(df['actual_demand_mw'], df['predicted_demand_mw']) * 100

print("\n🎯 MÉTRIQUES GLOBALES (toutes zones, tous horizons)")
print("=" * 70)
print(f"MAE:  {mae:,.1f} MW")
print(f"RMSE: {rmse:,.1f} MW")
print(f"MAPE: {mape:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Métriques par horizon (cibles: <2% @24h, <2.5% @48h, <3.5% @168h)

# COMMAND ----------

# Regroupement en buckets d'horizon alignés sur les cibles du README/config.yaml.
def horizon_bucket(h):
    if h <= 24:
        return "0-24h"
    elif h <= 48:
        return "24-48h"
    elif h <= 72:
        return "48-72h"
    else:
        return "72-168h"


df['horizon_bucket'] = df['horizon_hours'].apply(horizon_bucket)

horizon_metrics = (
    df.groupby('horizon_bucket')
    .apply(lambda g: pd.Series({
        'n': len(g),
        'mae': mean_absolute_error(g['actual_demand_mw'], g['predicted_demand_mw']),
        'rmse': np.sqrt(mean_squared_error(g['actual_demand_mw'], g['predicted_demand_mw'])),
        'mape': mean_absolute_percentage_error(g['actual_demand_mw'], g['predicted_demand_mw']) * 100,
    }))
    .reindex(["0-24h", "24-48h", "48-72h", "72-168h"])
)

print("\n⏱️ PERFORMANCE PAR HORIZON")
print(horizon_metrics.round(2))

targets = config.get('monitoring', {}).get('targets_mape', {})
print(f"\nCibles MAPE (config.yaml): {targets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Métriques par zone

# COMMAND ----------

zone_metrics = (
    df.groupby('zone')
    .apply(lambda g: pd.Series({
        'n': len(g),
        'mae': mean_absolute_error(g['actual_demand_mw'], g['predicted_demand_mw']),
        'mape': mean_absolute_percentage_error(g['actual_demand_mw'], g['predicted_demand_mw']) * 100,
    }))
    .sort_values('mape', ascending=False)
)

print("\n🗺️ PERFORMANCE PAR ZONE")
print(zone_metrics.round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualisation (zone Ontario ou première zone disponible, 7 derniers jours)

# COMMAND ----------

plot_zone = 'Ontario' if 'Ontario' in df['zone'].unique() else sorted(df['zone'].unique())[0]
df_plot = df[df['zone'] == plot_zone].sort_values('target_datetime').tail(7 * 24)

plt.figure(figsize=(14, 6))
plt.plot(df_plot['target_datetime'], df_plot['actual_demand_mw'], label='Réel', linewidth=2)
plt.plot(df_plot['target_datetime'], df_plot['predicted_demand_mw'], label='Prédit', linewidth=2, linestyle='--')
plt.xlabel('Date')
plt.ylabel('Charge (MW)')
plt.title(f'Prédictions vs Réalité — zone {plot_zone} (7 derniers jours)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ Visualisation générée")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Détection de drift

# COMMAND ----------

print("\n🚨 DÉTECTION DRIFT")
print("=" * 70)

df['date'] = pd.to_datetime(df['target_datetime']).dt.date
recent_dates = sorted(df['date'].unique())[-7:]

df_recent = df[df['date'].isin(recent_dates)]
df_hist = df[~df['date'].isin(recent_dates)]

if len(df_hist) > 0:
    mape_recent = mean_absolute_percentage_error(df_recent['actual_demand_mw'], df_recent['predicted_demand_mw']) * 100
    mape_hist = mean_absolute_percentage_error(df_hist['actual_demand_mw'], df_hist['predicted_demand_mw']) * 100
    drift = mape_recent - mape_hist

    print(f"MAPE historique: {mape_hist:.2f}%")
    print(f"MAPE récent (7j): {mape_recent:.2f}%")
    print(f"Drift: {drift:+.2f}%")

    alert_threshold = config.get('monitoring', {}).get('alerts', {}).get('mape_threshold', 5.0)
    if mape_recent > alert_threshold or abs(drift) > 1.0:
        print(f"\n⚠️  ALERTE: dégradation détectée (MAPE récent > {alert_threshold}% ou drift > 1pt)")
        print("Action recommandée: ré-entraîner le(s) modèle(s)")
    else:
        print("\n✅ Performance stable")
else:
    print("⚠️  Pas assez de données historiques pour comparer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Écriture des métriques

# COMMAND ----------

metrics_df = pd.DataFrame([{
    'evaluation_time': pd.Timestamp.now(),
    'mae': float(mae),
    'rmse': float(rmse),
    'mape': float(mape),
    'n_predictions': int(len(df)),
}])

spark_metrics = spark.createDataFrame(metrics_df)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {MONITORING_TABLE} (
        evaluation_time TIMESTAMP,
        mae DOUBLE,
        rmse DOUBLE,
        mape DOUBLE,
        n_predictions INT
    )
    USING DELTA
""")

spark_metrics.write.format("delta").mode("append").saveAsTable(MONITORING_TABLE)

print(f"\n✅ Métriques enregistrées dans {MONITORING_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Pipeline complet
# MAGIC
# MAGIC - Bronze: Ingestion IESO (global + zonal) + météo historique
# MAGIC - Silver: Jointure/nettoyage demande + météo
# MAGIC - Gold: Feature engineering + sélection de features
# MAGIC - Modeling: Entraînement 24h + 7 jours (2 modèles MLflow distincts)
# MAGIC - Inference: Construction des features de prédiction + batch prediction
# MAGIC - Monitoring: Évaluation continue par zone et par horizon