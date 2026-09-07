# 🏗️ Architecture Médaillon - Ontario Energy Forecast

## Vue d'ensemble

Ce projet suit l'architecture **Medallion** (Bronze → Silver → Gold) dans Unity Catalog Databricks.

**Catalog Unity Catalog** : `workspace`  
**Schema** : `energy_forecast`  
**Chemin complet** : `workspace.energy_forecast.*`

---

## 📊 Flux de données

```
┌─────────────────────────────────────────────────────────────┐
│                      SOURCES EXTERNES                        │
│  • IESO API (demande électrique)                           │
│  • Weather.gc.ca (météo historique)                         │
│  • Open-Meteo (prévisions météo)                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    🥉 COUCHE BRONZE                          │
│              Données brutes, non transformées                │
│                                                               │
│  workspace.energy_forecast.load_actual_bronze                │
│    ├─ Demande globale Ontario (market_demand, ontario_demand)│
│    ├─ Source: IESO API /Demand                              │
│    └─ Fréquence: horaire                                     │
│                                                               │
│  workspace.energy_forecast.load_zonal_bronze                 │
│    ├─ Demande par zone (East, West, Toronto, etc.)          │
│    ├─ Source: IESO API /DemandZonal                         │
│    └─ Zones: 10 zones IESO                                   │
│                                                               │
│  workspace.energy_forecast.weather_bronze                    │
│    ├─ Observations météo historiques                         │
│    ├─ Source: Weather.gc.ca (climate-hourly)                │
│    └─ Variables: temp, humidity, dewpoint, windspeed, etc.  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    🥈 COUCHE SILVER                          │
│         Données nettoyées, jointures, qualité               │
│                                                               │
│  workspace.energy_forecast.demand_weather_silver             │
│    ├─ Jointure demande + météo par zone                     │
│    ├─ Nettoyage: valeurs nulles, outliers                   │
│    ├─ Enrichissement: météo agrégée par poids régionaux     │
│    └─ Granularité: 1 ligne = 1 heure × 1 zone               │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐  ┌──────────────────────────────┐
│    🥇 COUCHE GOLD 24H     │  │    🥇 COUCHE GOLD 7J         │
│   Features + Modèle 24h   │  │  Features + Modèle 7 jours   │
│                            │  │                               │
│  ml_features_gold_24h      │  │  ml_features_gold_7j          │
│  ├─ Features engineering  │  │  ├─ Multi-horizon (H+1→H+168)│
│  ├─ Lags: 1h, 24h, 168h  │  │  ├─ forecast_horizon_hours   │
│  ├─ Rolling: 3h, 24h...   │  │  ├─ Lags adaptés 7j          │
│  └─ Temporal encoding     │  │  └─ Direct prediction        │
│                            │  │                               │
│  ml_features_training_gold │  │                               │
│  └─ Vue filtrée (24h)     │  │                               │
│                            │  │                               │
│  load_forecast_gold        │  │  load_forecast_7j             │
│  └─ Prédictions 24h       │  │  └─ Prédictions 168h         │
│                            │  │                               │
│  load_shap_gold            │  │  load_shap_7j                 │
│  └─ Explainability 24h    │  │  └─ Explainability 7j        │
└───────────────────────────┘  └──────────────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              🎯 COUCHE MONITORING & ANALYTICS                │
│                                                               │
│  workspace.energy_forecast.model_performance_gold            │
│    ├─ Métriques: MAE, RMSE, MAPE par zone et horizon        │
│    ├─ Comparaison 24h vs 7j vs baseline                     │
│    └─ Alertes: MAPE > 5%, données manquantes                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Tables Unity Catalog - Détails

### 🥉 Bronze Layer

| Table | Chemin complet | Description | Source | Fréquence |
|-------|---------------|-------------|--------|-----------|
| `load_actual_bronze` | `workspace.energy_forecast.load_actual_bronze` | Demande globale Ontario | IESO API `/Demand` | Horaire |
| `load_zonal_bronze` | `workspace.energy_forecast.load_zonal_bronze` | Demande par zone (10 zones) | IESO API `/DemandZonal` | Horaire |
| `weather_bronze` | `workspace.energy_forecast.weather_bronze` | Météo historique (entraînement) | Weather.gc.ca | Historique |

### 🥈 Silver Layer

| Table | Chemin complet | Description | Colonnes clés |
|-------|---------------|-------------|---------------|
| `demand_weather_silver` | `workspace.energy_forecast.demand_weather_silver` | Demande + météo nettoyées | `datetime`, `zone`, `ontario_demand`, `temperature`, `humidity`, `windspeed` |

### 🥇 Gold Layer - Modèle 24h

| Table | Chemin complet | Description | Usage |
|-------|---------------|-------------|-------|
| `ml_features_gold_24h` | `workspace.energy_forecast.ml_features_gold_24h` | Features + target modèle 24h | Entraînement |
| `ml_features_training_gold` | `workspace.energy_forecast.ml_features_training_gold` | Vue filtrée (lignes valides) | Entraînement |
| `load_forecast_gold` | `workspace.energy_forecast.load_forecast_gold` | Prédictions batch 24h (récursif) | Inférence |
| `load_shap_gold` | `workspace.energy_forecast.load_shap_gold` | Valeurs SHAP 24h | Explainability |

### 🥇 Gold Layer - Modèle 7 jours

| Table | Chemin complet | Description | Usage |
|-------|---------------|-------------|-------|
| `ml_features_gold_7j` | `workspace.energy_forecast.ml_features_gold_7j` | Features multi-horizon (H+1 à H+168) | Entraînement |
| `load_forecast_7j` | `workspace.energy_forecast.load_forecast_7j` | Prédictions batch 7j (direct) | Inférence |
| `load_shap_7j` | `workspace.energy_forecast.load_shap_7j` | Valeurs SHAP 7j | Explainability |

### 🎯 Monitoring

| Table | Chemin complet | Description |
|-------|---------------|-------------|
| `model_performance_gold` | `workspace.energy_forecast.model_performance_gold` | Métriques de performance (MAE, RMSE, MAPE) |

---

## 🔄 Pipeline de transformation

### Bronze → Silver

**Script** : `pipeline/silver/03_join_demand_weather.py`

**Transformations** :
- Jointure `load_zonal_bronze` + `weather_bronze` sur `datetime` et `zone`
- Nettoyage des valeurs nulles et outliers
- Agrégation météo pondérée par zone (poids population/consommation)
- Validation de cohérence temporelle

### Silver → Gold (24h)

**Scripts** :
- `pipeline/gold/04_build_features.py` → `ml_features_gold_24h`
- `pipeline/gold/05_feature_selection.py` → `selected_features.yaml`

**Feature Engineering** :
- **Lags** : 1h, 24h, 48h, 72h, 144h, 168h, 336h
- **Rolling windows** : mean, std, min, max sur 3h, 12h, 24h, 48h, 72h, 168h
- **Temporal** : hour, day_of_week, month, is_weekend, is_holiday (Ontario)
- **Météo** : HDD/CDD (base 18°C), interactions température × saison
- **Cyclical encoding** : sin/cos pour hour, day_of_week, month

### Silver → Gold (7j)

**Script** : `pipeline/gold/04_build_features_7j.py` → `ml_features_gold_7j`

**Différences clés** :
- Feature additionnelle : `forecast_horizon_hours` (1 à 168)
- Lags adaptés pour prédiction directe multi-horizon
- Pas de récursion : chaque horizon prédit en une passe

---

## 📦 Conventions de nommage

### Tables

- **Bronze** : `{entity}_bronze` (ex: `load_actual_bronze`)
- **Silver** : `{entity}_silver` (ex: `demand_weather_silver`)
- **Gold** : `{entity}_gold_{variant}` (ex: `ml_features_gold_24h`, `load_forecast_gold`)

### Colonnes

- **Timestamps** : `datetime` (UTC), `date`, `hour`
- **Identifiants** : `zone` (East, West, Toronto, ...), `forecast_horizon_hours`
- **Métriques** : `ontario_demand`, `market_demand`, `temperature`, `humidity`
- **Features** : `{metric}_lag_{hours}h`, `{metric}_rolling_{window}h_{func}`

---

## 🔐 Gestion des accès

Toutes les tables sont dans **Unity Catalog** avec gestion des permissions intégrée.

**Catalog** : `workspace`  
**Schema** : `energy_forecast`

Pour créer/lire les tables, les notebooks utilisent :
```python
catalog_name = "workspace"
schema_name = "energy_forecast"
table_name = "demand_weather_silver"
full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
```

---

## 📊 Volumétrie estimée

| Layer | Tables | Lignes (approx.) | Taille |
|-------|--------|------------------|--------|
| Bronze | 3 | ~3M (5 ans × 10 zones × horaire) | ~500 MB |
| Silver | 1 | ~500K | ~100 MB |
| Gold 24h | 4 | ~500K (features) + ~50K (forecasts) | ~200 MB |
| Gold 7j | 3 | ~500K (features) + ~50K (forecasts) | ~200 MB |
| **Total** | **11** | **~5M lignes** | **~1 GB** |

---

## 🚀 Accès rapide

**Explorer les tables** :
- [workspace.energy_forecast](https://dbc-23fb8b12-3fa4.cloud.databricks.com/explore/data/workspace/energy_forecast)

**Requête exemple** :
```sql
SELECT 
  datetime,
  zone,
  ontario_demand,
  temperature,
  humidity
FROM workspace.energy_forecast.demand_weather_silver
WHERE datetime >= CURRENT_DATE() - INTERVAL 7 DAYS
ORDER BY datetime DESC, zone
LIMIT 100;
```

---

## 📚 Références

- Config global : [`config/config.yaml`](../config/config.yaml)
- Architecture détaillée : [`docs/architecture.md`](./architecture.md)
- README principal : [`README.md`](../README.md)
