# Architecture Projet - Prévision Demande Électrique Ontario

**Version:** 1.1.0
**Date:** 2026-08-29 (mise à jour lors du nettoyage — voir `CHANGELOG_CLEANUP.md`)
**Auteur:** Energy Forecast Team

---

## 📋 Vue d'ensemble

Système de prévision de la demande électrique de l'Ontario, **par zone**, sur
deux horizons distincts (24h et 168h), avec **deux modèles LightGBM séparés**
et leur propre expérience/registre MLflow :
- **Données de demande :** IESO (globale + zonale)
- **Météo d'entraînement :** Weather.gc.ca (observations historiques par station)
- **Météo d'inférence :** Open-Meteo (prévisions futures — Weather.gc.ca n'en fournit pas)
- **Variables calendaires** (jours fériés Ontario, saisons, cycles)
- **Architecture Medallion** (Bronze → Silver → Gold)
- **LightGBM** + **MLflow** (tracking + Model Registry)

---

## 🏗️ Architecture Medallion

```
IESO API                Weather.gc.ca                Open-Meteo
(Demand,                (climate-hourly,              (forecast + archive,
 DemandZonal)            historique)                   inférence uniquement)
      |                        |                              |
      v                        v                              |
+------------------+   +------------------+                   |
| BRONZE           |   | BRONZE           |                   |
| load_actual_     |   | weather_bronze   |                   |
| bronze,          |   +------------------+                   |
| load_zonal_bronze|            |                              |
+------------------+            |                              |
      |                         |                              |
      +-----------+-------------+                              |
                  |                                             |
                  v                                             |
        +--------------------+                                 |
        | SILVER             |                                 |
        | demand_weather_    |                                 |
        | silver             |                                 |
        +--------------------+                                 |
                  |                                             |
                  v                                             |
        +--------------------+                                 |
        | GOLD               |                                 |
        | ml_features_gold   |                                 |
        | - Lags (1-336h)    |                                 |
        | - Rolling windows  |                                 |
        | - HDD/CDD          |                                 |
        | - Cycliques        |                                 |
        | - Holidays         |                                 |
        +--------------------+                                 |
              |          |                                      |
              v          v                                      |
    +--------------+ +------------------+                       |
    | Modèle 24h   | | Modèle 7j (168h) |                       |
    | LightGBM     | | LightGBM direct  |                       |
    | + MLflow     | | + MLflow         |                       |
    +--------------+ +------------------+                       |
              |                                                  |
              v  (scoring récursif — voir README,                |
              |   seul le modèle 24h est câblé ici)   <----------+
    +--------------------+
    | load_forecast_gold |
    | load_shap_gold     |
    +--------------------+
              |
              v
    +---------------------+
    | model_performance_  |
    | gold (monitoring)   |
    +---------------------+
              |
              v
    +-------------------+
    | DASHBOARD          |
    | Databricks Lakeview|
    +-------------------+
```

---

## 🗂️ Structure du Projet

Voir le README pour l'arborescence complète et à jour (`pipeline/{bronze,
silver, gold, modeling, inference, monitoring}`, `config/`, `utils/`, `data/`,
`dashboards/`).

---

## 📊 Tables Unity Catalog

### Bronze Layer (données brutes)

**`load_actual_bronze`** — demande globale Ontario
```sql
CREATE TABLE workspace.energy_forecast.load_actual_bronze (
  datetime TIMESTAMP,
  market_demand DOUBLE,
  ontario_demand DOUBLE,
  ingestion_time TIMESTAMP
)
USING DELTA
PARTITIONED BY (date(datetime));
```

**`load_zonal_bronze`** — demande par zone
```sql
CREATE TABLE workspace.energy_forecast.load_zonal_bronze (
  datetime TIMESTAMP,
  zone STRING,
  demand_mw DOUBLE,
  ingestion_time TIMESTAMP
)
USING DELTA
PARTITIONED BY (date(datetime), zone);
```

**`weather_bronze`** — météo historique Weather.gc.ca, par zone/station

### Silver Layer

**`demand_weather_silver`** — jointure demande zonale + météo, nettoyée
(déduplication, contrôle de plausibilité, harmonisation timezone UTC)

### Gold Layer

**`ml_features_gold`** — table de features utilisée à la fois pour
l'entraînement (avec `forecast_horizon_hours` et `target_demand_mw`) et pour
le scoring.

**`load_forecast_gold`**
```sql
CREATE TABLE workspace.energy_forecast.load_forecast_gold (
  zone STRING,
  target_datetime TIMESTAMP,
  predicted_demand_mw DOUBLE,
  prediction_time TIMESTAMP,
  target_date DATE
)
USING DELTA
PARTITIONED BY (target_date);
```

**`load_shap_gold`** — contributions SHAP par feature pour chaque prédiction.

**`model_performance_gold`**
```sql
CREATE TABLE workspace.energy_forecast.model_performance_gold (
  evaluation_time TIMESTAMP,
  mae DOUBLE,
  rmse DOUBLE,
  mape DOUBLE,
  n_predictions INT
)
USING DELTA;
```

---

## 🤖 Modélisation

### Modèle 24h (`pipeline/modeling/06_train_model_24h.py`)

LightGBM entraîné sur `ml_features_gold`, tag `forecast_horizon: "24h"`,
sélection de features par gain + MAPE de validation, split temporel strict
(train / validation / test). Enregistré dans MLflow sous
`ontario_demand_lightgbm_24h`.

### Modèle 7 jours (`pipeline/modeling/07_train_model_7j.py`)

LightGBM **multi-horizon direct** : `forecast_horizon_hours` (1 à 168) est une
feature du modèle (avec transformations sqrt/log1p/sin/cos), donc un seul
modèle prédit n'importe quel horizon dans [1h, 168h] en une seule inférence
(pas de récursion). Enregistré sous `ontario_demand_lightgbm_7j`.

**Hyperparamètres communs (`config.yaml -> models.common.hyperparameters`) :**
```yaml
objective: regression
metric: rmse
num_leaves: 31
learning_rate: 0.05
feature_fraction: 0.9
bagging_fraction: 0.8
bagging_freq: 5
max_depth: -1
min_data_in_leaf: 20
lambda_l1: 0.1
lambda_l2: 0.1
```

### Scoring batch (`pipeline/inference/09_batch_prediction.py`)

Approche **récursive** : charge un seul modèle (24h par défaut), prédit heure
par heure jusqu'à l'horizon disponible dans les prévisions météo, réinjecte
chaque prédiction comme lag pour l'heure suivante. Voir l'avertissement dans le
README concernant le modèle 7j, qui n'est pas encore branché à un script de
scoring dédié.

---

## 🔄 Orchestration (Jobs Databricks)

Voir `config.yaml -> jobs` pour la définition exacte (tâches, cron). Résumé :

| Job | Fréquence | Tâches |
|---|---|---|
| `ontario_demand_ingestion` | Toutes les 6h | `01_ingest_ieso_demand`, `02_ingest_weather_historical` |
| `ontario_demand_processing` | 30 min après ingestion | `03_join_demand_weather`, `04_build_features`, `05_feature_selection` |
| `ontario_demand_training` | Hebdomadaire (dim. 03h) | `06_train_model_24h`, `07_train_model_7j` |
| `ontario_demand_prediction` | 01h, 07h, 13h, 19h | `08_build_prediction_features`, `09_batch_prediction` |
| `ontario_demand_evaluation` | Quotidien à 02h | `10_model_evaluation` |

---

## 📈 Dashboard

Deux exports Lakeview sont fournis dans `dashboards/` :
- `energy_forecast_dashboard.lvdash.json`
- `dashboard_draft_20260828.lvdash.json` (brouillon plus récent, à comparer/fusionner)

Pages suggérées : vue exécutive (KPI, pic attendu, MAPE), prévisions futures
(24h / 7j avec intervalles), météo multi-zones, performance modèle (MAE/RMSE/MAPE
par horizon, feature importance / SHAP), diagnostic qualité des données.

---

## 🎯 Objectifs de Performance

| Horizon | MAPE Cible |
|---------|------------|
| 24h     | < 2%       |
| 48h     | < 2.5%     |
| 168h (7j) | < 3.5%   |

---

## 🔐 Sécurité & Gouvernance

- **Unity Catalog** pour la gouvernance des données
- **Secrets** Databricks pour les clés/API si besoin (IESO et Open-Meteo sont
  publiques et ne nécessitent pas de clé aujourd'hui)
- **RBAC** sur tables et notebooks
- **Model Registry MLflow** avec deux modèles nommés séparément (`_24h`, `_7j`)

---

## 📚 Références

- [IESO Public Reports](https://reports-public.ieso.ca/public/)
- [Weather.gc.ca API](https://api.weather.gc.ca/)
- [Open-Meteo API](https://open-meteo.com/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [MLflow Tracking](https://mlflow.org/)
