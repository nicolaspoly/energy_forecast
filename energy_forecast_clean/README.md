# 🔌 Prévision Demande Électrique Ontario

**Projet de prévision de la demande électrique de l'Ontario, par zone, sur deux horizons : 24h et 7 jours (168h).**

> 🧹 **Nettoyage 2026-08-29** : ce projet a été réorganisé et dédupliqué (voir
> `CHANGELOG_CLEANUP.md`). Le README ci-dessous décrit l'état **réel** du code
> après nettoyage, pas un état visé — en particulier les noms de table et les
> deux modèles MLflow distincts.

---

## 🎯 Objectif

Entraîner et déployer **2 modèles LightGBM séparés** :

| Modèle | Fichier d'entraînement | Registre MLflow | Usage |
|---|---|---|---|
| **24h** | `pipeline/modeling/06_train_model_24h.py` | `ontario_demand_lightgbm_24h` | Prévision court terme |
| **7 jours (168h)** | `pipeline/modeling/07_train_model_7j.py` | `ontario_demand_lightgbm_7j` | Prévision multi-horizon (H+1 à H+168), `forecast_horizon_hours` en feature |

en combinant :
- 📊 Demande historique IESO (globale + par zone)
- 🌤️ Météo historique Weather.gc.ca (entraînement) + prévisions Open-Meteo (inférence)
- 📅 Variables calendaires (jours fériés Ontario, saisons, cycles)
- 🤖 LightGBM avec tracking MLflow (Unity Catalog + Model Registry)

**Cibles MAPE :** < 2% @ 24h, < 2.5% @ 48h, < 3.5% @ 168h (voir `config.yaml -> monitoring.targets_mape`)

---

## ⚠️ Point ouvert important : le lien modèle 7j ↔ batch prediction

Le seul script de scoring batch existant, `pipeline/inference/09_batch_prediction.py`,
fonctionne de façon **récursive** : il charge **un seul** modèle et prédit heure par
heure jusqu'à 168h, en réinjectant chaque prédiction comme "lag" pour l'heure
suivante. Aujourd'hui il pointe par défaut sur le modèle **24h**
(`PREDICTION_MODEL_KEY = "horizon_24h"` dans le script).

Le modèle **7 jours** (`07_train_model_7j.py`) est entraîné différemment : il
prend `forecast_horizon_hours` directement en feature et prédit chaque horizon
en une seule passe (pas de récursion). **Aucun script de scoring batch ne
l'utilise pour l'instant.** Si tu veux que ce modèle serve réellement les
prévisions à 48h-168h, il faut écrire un notebook de scoring dédié qui appelle
`model.predict(...)` directement avec `forecast_horizon_hours` renseigné, sans
boucle récursive. Ce n'était pas dans le périmètre de ce nettoyage (uniquement
structurel) — à faire dans une prochaine étape si besoin.

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
                     v
           +--------------------+
           | GOLD               |
           | ml_features_gold   |
           +--------------------+
                     |
        +------------+------------+
        v                         v
+----------------+       +----------------------+
| Modèle 24h     |       | Modèle 7j (168h)     |
| (LightGBM)     |       | (LightGBM, direct)   |
+----------------+       +----------------------+
        |
        v (récursif, voir section ci-dessus)
+--------------------+
| load_forecast_gold |
| load_shap_gold     |
+--------------------+
        |
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
- **Orchestration :** Jobs Databricks (voir `config.yaml -> jobs`)
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
│   │   ├── 04_build_features.py             # Feature engineering complet
│   │   └── 05_feature_selection.py          # Sélection de features (-> selected_features.yaml)
│   ├── modeling/
│   │   ├── 06_train_model_24h.py            # Modèle 24h
│   │   └── 07_train_model_7j.py             # Modèle 7 jours (multi-horizon direct)
│   ├── inference/
│   │   ├── 08_build_prediction_features.py  # Features météo (Open-Meteo) + demande pour scoring
│   │   └── 09_batch_prediction.py           # Scoring batch récursif (voir avertissement plus haut)
│   └── monitoring/
│       └── 10_model_evaluation.py           # Évaluation continue (par zone, par horizon)
├── config/
│   ├── config.yaml                # Configuration globale (tables, modèles, jobs)
│   ├── zones_config.py            # Zones météo Ontario (poids, coordonnées)
│   └── selected_features.yaml     # Sortie de 05_feature_selection.py
├── utils/
│   ├── holidays_ontario.py
│   ├── weather_utils.py           # Utilitaires Open-Meteo
│   └── feature_utils.py
├── data/
│   ├── archive/                   # Export statique 2020-présent (backfill de secours)
│   ├── exports/                   # Exports ponctuels (ex: dashboard prototyping)
│   └── misc_unused/                # Données non reliées au pipeline actuel — à confirmer/supprimer
├── dashboards/                    # Exports Lakeview (.lvdash.json)
├── mlruns/                        # Artefacts MLflow locaux
├── docs/architecture.md           # Détails des tables Unity Catalog
└── CHANGELOG_CLEANUP.md           # Détail du nettoyage du 2026-08-29
```

---

## 🚀 Quick Start

### 1. Configuration

Éditer `config/config.yaml` : catalog UC, tables, modèles, jobs. Adapter aussi
le chemin `/Workspace/Users/n.jouglet23@gmail.com/energy_forecast_clean/...` codé en
dur dans chaque notebook (voir "Limites connues" plus bas).

### 2. Créer les Tables Unity Catalog

Les notebooks créent leurs tables via `CREATE TABLE IF NOT EXISTS` — pas de
script DDL séparé à exécuter au préalable. Voir `docs/architecture.md` pour le
détail des schémas.

### 3. Exécuter le Pipeline (dans l'ordre)

```
pipeline/bronze/01_ingest_ieso_demand.py         # BACKFILL=True pour le 1er chargement
pipeline/bronze/02_ingest_weather_historical.py
pipeline/silver/03_join_demand_weather.py
pipeline/gold/04_build_features.py
pipeline/gold/05_feature_selection.py
pipeline/modeling/06_train_model_24h.py
pipeline/modeling/07_train_model_7j.py
pipeline/inference/08_build_prediction_features.py
pipeline/inference/09_batch_prediction.py
pipeline/monitoring/10_model_evaluation.py
```

### 4. Consulter le Dashboard

Importer un des fichiers `dashboards/*.lvdash.json` dans Databricks Lakeview.

---

## 🔧 Limites connues (à traiter séparément de ce nettoyage)

- **Chemins codés en dur :** tous les notebooks chargent la config via un
  chemin absolu `/Workspace/Users/n.jouglet23@gmail.com/energy_forecast_clean/...`.
  Ça marche tant que le projet reste dans ce workspace précis, mais ce n'est
  pas portable. Avec Databricks Repos/Asset Bundles, préférer un chemin
  relatif au notebook (`dbutils.notebook.entry_point...` ou variable de job).
- **Un seul chemin de scoring batch** pour deux modèles entraînés (voir
  section dédiée plus haut).
- **`selected_features.yaml`** ne contient que 28 features globales
  (température, lags, calendrier) sans détail par zone — à vérifier s'il est
  à jour par rapport à la table zonale actuelle `ml_features_gold`, ou à
  régénérer via `05_feature_selection.py`.
- **`data/misc_unused/ieso_t24_files_downloaded.csv`** : index de fichiers
  IESO "Adequacy3" non lié au pipeline demande/météo actuel — conservé au cas
  où mais probablement obsolète.

---

## 📊 Tables Unity Catalog (noms réels)

Voir `config.yaml -> catalog.tables` et `docs/architecture.md` pour le détail.

| Couche | Table | Contenu |
|---|---|---|
| Bronze | `load_actual_bronze` | Demande Ontario globale |
| Bronze | `load_zonal_bronze` | Demande par zone |
| Bronze | `weather_bronze` | Météo historique (Weather.gc.ca) |
| Silver | `demand_weather_silver` | Demande + météo jointes, nettoyées |
| Gold | `ml_features_gold` | Table de features (entraînement + scoring) |
| Gold | `load_forecast_gold` | Prédictions batch (zone, horizon, valeur) |
| Gold | `load_shap_gold` | Contributions SHAP des prédictions |
| Gold | `model_performance_gold` | Métriques de monitoring |

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
- [Changelog du nettoyage 2026-08-29](CHANGELOG_CLEANUP.md)
- [IESO Public Reports](https://reports-public.ieso.ca/public/)
- [Weather.gc.ca API](https://api.weather.gc.ca/)
- [Open-Meteo API](https://open-meteo.com/)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)

---

## 📧 Contact

**Energy Forecast Team**
Version: 1.1.0
Date: 2026-08-29
