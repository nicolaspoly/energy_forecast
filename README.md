# 🔌 Prévision Demande Électrique Ontario

**Projet de prévision de la demande électrique de l'Ontario, par zone, sur deux horizons : 24h et 7 jours (168h).**

> 🧹 **Nettoyage 2026-08-29** puis **2026-09-01** : ce projet a été réorganisé,
> dédupliqué, puis les 3 points laissés ouverts par le premier passage ont été
> traités (voir `CHANGELOG_CLEANUP.md`). Le README ci-dessous décrit l'état
> **réel** du code après ce second nettoyage — table par table, script par
> script.

---

## 🎯 Objectif

Entraîner et déployer **2 modèles LightGBM séparés**, chacun avec son propre
pipeline Gold et son propre scoring :

| Modèle | Entraînement | Gold | Scoring batch | Registre MLflow |
|---|---|---|---|---|
| **24h** | `06_train_model_24h.py` | `ml_features_gold_24h` | `09_batch_prediction.py` (récursif) | `ontario_demand_lightgbm_24h` |
| **7 jours (168h)** | `07_train_model_7j.py` | `ml_features_gold_7j` | `09b_batch_prediction_7j_direct.py` (direct) | `ontario_demand_lightgbm_7j` |

en combinant :
- 📊 Demande historique IESO (globale + par zone)
- 🌤️ Météo historique Weather.gc.ca (entraînement) + prévisions Open-Meteo (inférence)
- 📅 Variables calendaires (jours fériés Ontario, saisons, cycles)
- 🤖 LightGBM avec tracking MLflow (Unity Catalog + Model Registry)

**Cibles MAPE :** < 2% @ 24h, < 2.5% @ 48h, < 3.5% @ 168h (voir `config.yaml -> monitoring.targets_mape`)

---

## 🔀 Deux modèles, deux pipelines de scoring

Les deux modèles ne partagent **ni** table Gold **ni** script de scoring — ils
ont des schémas de features différents et ont chacun leur chaîne dédiée :

**Modèle 24h — récursif** : un seul modèle, ré-injecté heure par heure jusqu'à
168h (chaque prédiction sert de "lag" pour l'heure suivante).
```
04_build_features.py -> ml_features_gold_24h -> 06_train_model_24h.py
08_build_prediction_features.py -> 09_batch_prediction.py -> load_forecast_gold / load_shap_gold
```

**Modèle 7j — direct** : `forecast_horizon_hours` (1 à 168) est une feature
d'entrée ; le modèle prédit chaque horizon en une seule passe, sans boucle.
```
04_build_features_7j.py -> ml_features_gold_7j -> 07_train_model_7j.py
08b_build_prediction_features_7j.py -> 09b_batch_prediction_7j_direct.py -> load_forecast_7j / load_shap_7j
```

`08b` et `09b` sont des wrappers minces : ils surchargent `MODEL_HORIZON_KEY`
puis exécutent respectivement `08_build_prediction_features.py` et
`08b_build_prediction_features_7j.py` via `exec()`, pour ne pas dupliquer la
logique de récupération météo/demande. Détail des colonnes et de la
comparaison des deux modèles dans `10_model_evaluation.py`.

---

## 🏗️ Architecture

### Medallion Bronze → Silver → Gold

```
IESO (demande)              Weather.gc.ca (météo historique)      Open-Meteo (prévisions)
      |                              |                                    |
      v                              v                                    v
+------------------+       +----------------------+          (utilisé en inférence
| BRONZE           |       | BRONZE               |           uniquement, pas de
| load_actual_bronze|      | weather_bronze        |           table Bronze dédiée)
| load_zonal_bronze |      +----------------------+
+------------------+                 |
      |                              |
      +--------------+---------------+
                     |
                     v
           +--------------------+
           | SILVER             |
           | demand_weather_    |
           | silver             |
           +--------------------+
                     |
        +------------+------------+
        v                         v
+----------------------+  +----------------------+
| GOLD 24h              |  | GOLD 7j               |
| ml_features_gold_24h  |  | ml_features_gold_7j   |
| (+ vue training)       |  | (multi-horizon)       |
+----------------------+  +----------------------+
        |                         |
        v                         v
+----------------+       +----------------------+
| Modèle 24h     |       | Modèle 7j (168h)     |
| (LightGBM,     |       | (LightGBM, direct)   |
|  récursif)     |       +----------------------+
+----------------+                 |
        |                          v
        v                +----------------------+
+--------------------+   | load_forecast_7j     |
| load_forecast_gold |   | load_shap_7j          |
| load_shap_gold     |   +----------------------+
+--------------------+             |
        |                          |
        +------------+-------------+
                     v
           +---------------------+      +--------------------+
           | model_performance_  | <--- | DASHBOARD (Lakeview)|
           | gold (monitoring)   |      +--------------------+
           +---------------------+
```

### Stack Technique
- **Plateforme :** Databricks (Unity Catalog, MLflow)
- **Compute :** Serverless
- **Modèle :** LightGBM — 2 modèles distincts (24h, 7j)
- **Orchestration :** Jobs Databricks (voir `config.yaml -> jobs` : 5 jobs —
  ingestion, processing, training, prediction_24h, prediction_7j, evaluation)
- **Dashboard :** Databricks Lakeview (`dashboards/`)

---

## 📂 Structure du Projet

```
energy_forecast/
├── pipeline/
│   ├── bronze/
│   │   ├── 01_ingest_ieso_demand.py         # IESO demande globale + zonale
│   │   └── 02_ingest_weather_historical.py  # Weather.gc.ca (historique, entraînement)
│   ├── silver/
│   │   └── 03_join_demand_weather.py        # Jointure + nettoyage
│   ├── gold/
│   │   ├── 04_build_features.py             # Feature engineering 24h -> ml_features_gold_24h
│   │   ├── 04_build_features_7j.py          # Feature engineering 7j (multi-horizon) -> ml_features_gold_7j
│   │   └── 05_feature_selection.py          # Sélection de features 24h (-> selected_features.yaml)
│   ├── modeling/
│   │   ├── 06_train_model_24h.py            # Modèle 24h
│   │   └── 07_train_model_7j.py             # Modèle 7 jours (multi-horizon direct)
│   ├── inference/
│   │   ├── 08_build_prediction_features.py     # Features météo (Open-Meteo) + demande, modèle 24h
│   │   ├── 08b_build_prediction_features_7j.py # Wrapper : idem, paramétré pour le modèle 7j
│   │   ├── 09_batch_prediction.py              # Scoring batch récursif (24h)
│   │   └── 09b_batch_prediction_7j_direct.py   # Scoring batch direct (7j, sans récursion)
│   └── monitoring/
│       └── 10_model_evaluation.py           # Évaluation continue (par zone, par horizon, par jour/heure)
├── config/
│   ├── config.yaml                # Configuration globale (tables, modèles, jobs)
│   ├── zones_config.py            # Zones météo Ontario (poids, coordonnées)
│   └── selected_features.yaml     # Sortie de 05_feature_selection.py (modèle 24h)
├── utils/
│   ├── holidays_ontario.py
│   ├── weather_utils.py           # Utilitaires Open-Meteo
│   └── feature_utils.py
├── data/
│   ├── archive/                   # Export statique 2020-présent (backfill de secours)
│   └── exports/                   # Exports ponctuels (ex: dashboard prototyping)
├── dashboards/                    # Exports Lakeview (.lvdash.json)
├── mlruns/                        # Artefacts MLflow locaux
├── docs/architecture.md           # Détails des tables Unity Catalog
└── CHANGELOG_CLEANUP.md           # Détail des deux passages de nettoyage
```

---

## 🚀 Quick Start

### 1. Configuration

Éditer `config/config.yaml` : catalog UC, tables, modèles, jobs.

Tous les notebooks résolvent le chemin du projet via la variable
d'environnement **`ENERGY_FORECAST_PROJECT_ROOT`**, avec comme valeur par
défaut `/Workspace/Users/n.jouglet23@gmail.com/energy_forecast_clean` (le
workspace d'origine). Pour déployer ailleurs, définir cette variable
d'environnement sur le cluster / job Databricks — aucun fichier à modifier.

### 2. Créer les Tables Unity Catalog

Les notebooks créent leurs tables via `CREATE TABLE IF NOT EXISTS` — pas de
script DDL séparé à exécuter au préalable. Voir `docs/architecture.md` pour le
détail des schémas.

### 3. Exécuter le Pipeline (dans l'ordre)

```
pipeline/bronze/01_ingest_ieso_demand.py         # BACKFILL=True pour le 1er chargement
pipeline/bronze/02_ingest_weather_historical.py
pipeline/silver/03_join_demand_weather.py
pipeline/gold/04_build_features.py               # -> ml_features_gold_24h
pipeline/gold/04_build_features_7j.py            # -> ml_features_gold_7j
pipeline/gold/05_feature_selection.py            # modèle 24h uniquement
pipeline/modeling/06_train_model_24h.py
pipeline/modeling/07_train_model_7j.py
pipeline/inference/09_batch_prediction.py             # inclut 08_build_prediction_features.py
pipeline/inference/09b_batch_prediction_7j_direct.py  # inclut 08b puis 08 (via exec en cascade)
pipeline/monitoring/10_model_evaluation.py
```

### 4. Consulter le Dashboard

Importer un des fichiers `dashboards/*.lvdash.json` dans Databricks Lakeview.

---

## 🔧 Limites connues

- **`selected_features.yaml`** ne contient que 28 features globales
  (température, lags, calendrier) sans détail par zone, et ne concerne que le
  modèle 24h (`05_feature_selection.py` lit `ml_features_gold_24h`). Le modèle
  7j fait sa propre sélection de features, intégrée à `07_train_model_7j.py`.
  À régénérer via `05_feature_selection.py` si `04_build_features.py` change.
- **Chemin du projet configurable mais avec une seule valeur par défaut** :
  la bascule se fait via `ENERGY_FORECAST_PROJECT_ROOT` (voir "Quick Start"),
  ce qui suffit pour changer de workspace, mais reste un chemin absolu codé en
  dur comme repli — une migration vers Databricks Repos/Asset Bundles (import
  relatif au notebook) resterait plus propre à terme.
- **Deux jobs de prédiction distincts** (`prediction_24h`, `prediction_7j`,
  voir `config.yaml -> jobs`) plutôt qu'un seul : c'est voulu (schémas de
  features différents) mais ça veut dire deux plannings à maintenir en
  cohérence si l'un des deux change de fréquence.

---

## 📊 Tables Unity Catalog (noms réels)

Voir `config.yaml -> catalog.tables` et `docs/architecture.md` pour le détail.

| Couche | Table | Contenu |
|---|---|---|
| Bronze | `load_actual_bronze` | Demande Ontario globale |
| Bronze | `load_zonal_bronze` | Demande par zone |
| Bronze | `weather_bronze` | Météo historique (Weather.gc.ca) |
| Silver | `demand_weather_silver` | Demande + météo jointes, nettoyées |
| Gold | `ml_features_gold_24h` | Features + target, modèle 24h |
| Gold | `ml_features_training_gold` | Vue sur `ml_features_gold_24h` filtrée aux lignes valides |
| Gold | `ml_features_gold_7j` | Features multi-horizon (H+1 à H+168), modèle 7j |
| Gold | `load_forecast_gold` | Prédictions batch 24h (récursif) |
| Gold | `load_shap_gold` | Contributions SHAP, prédictions 24h |
| Gold | `load_forecast_7j` | Prédictions batch 7j (direct, tous horizons) |
| Gold | `load_shap_7j` | Contributions SHAP, prédictions 7j |
| Gold | `model_performance_gold` | Métriques de monitoring (les deux modèles) |

---

## 🌍 Zones Météo Ontario

| Zone       | Poids | Ville Référence      |
|------------|-------|----------------------|
| Toronto    | 25%   | Toronto              |
| Ottawa     | 15%   | Ottawa               |
| West       | 15%   | Kitchener-Waterloo   |
| Southwest  | 10%   | London / Windsor     |
| Niagara    | 10%   | Niagara Falls        |
| East       | 8%    | Kingston / Brockville|
| Northeast  | 7%    | Sudbury / Timmins    |
| Northwest  | 5%    | Thunder Bay / Sioux Lookout |
| Bruce      | 3%    | Bruce Peninsula      |
| Essa       | 2%    | Barrie               |

---

## 📚 Documentation

- [Architecture détaillée](docs/architecture.md)
- [Changelog des nettoyages (2026-08-29 et 2026-09-01)](CHANGELOG_CLEANUP.md)
- [IESO Public Reports](https://reports-public.ieso.ca/public/)
- [Weather.gc.ca API](https://api.weather.gc.ca/)
- [Open-Meteo API](https://open-meteo.com/)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)

---

## 📧 Contact

**Energy Forecast Team**
Version: 1.2.0
Date: 2026-09-01
