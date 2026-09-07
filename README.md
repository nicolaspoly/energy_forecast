# 🔌 Ontario Energy Demand Forecasting

**Machine Learning project for forecasting Ontario's electrical demand by zone, with two horizons: 24h and 7 days (168h).**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Databricks](https://img.shields.io/badge/Databricks-Unity%20Catalog-orange.svg)](https://databricks.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Model-green.svg)](https://lightgbm.readthedocs.io/)

---

## 📑 Table of Contents

- [🎯 Objectives](#-objectives)
- [🏗️ Architecture](#️-architecture)
- [📊 Data & Unity Catalog](#-data--unity-catalog)
- [🤖 Models](#-models)
- [📂 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📈 Performance](#-performance)
- [🔧 Configuration](#-configuration)
- [📚 Documentation](#-documentation)

---

## 🎯 Objectives

Train and deploy **two separate LightGBM models**, each with its own Gold pipeline and scoring logic:

| Model | Training | Gold Table | Batch Scoring | MLflow Registry |
|-------|----------|------------|---------------|-----------------|
| **24h** | `06_train_model_24h.py` | `ml_features_gold_24h` | `09_batch_prediction.py` (recursive) | `ontario_demand_lightgbm_24h` |
| **7 days (168h)** | `07_train_model_7j.py` | `ml_features_gold_7j` | `09b_batch_prediction_7j_direct.py` (direct) | `ontario_demand_lightgbm_7j` |

**Data sources:**
- 📊 IESO historical demand (global + zonal)
- 🌤️ Weather.gc.ca historical weather (training) + Open-Meteo forecasts (inference)
- 📅 Calendar features (Ontario holidays, seasons, cycles)

**Target MAPE:** < 2% @ 24h, < 2.5% @ 48h, < 3.5% @ 168h

---

## 🏗️ Architecture

### Medallion Architecture (Bronze → Silver → Gold)

```
IESO API          Weather.gc.ca        Open-Meteo
(demand)          (historical)         (forecasts)
    │                  │                     │
    v                  v                     v
┌─────────────────────────────────────────────────┐
│            🥉 BRONZE LAYER                       │
│  • load_actual_bronze (global demand)           │
│  • load_zonal_bronze (demand by zone)           │
│  • weather_bronze (historical weather)          │
└─────────────────────────────────────────────────┘
                    │
                    v
┌─────────────────────────────────────────────────┐
│            🥈 SILVER LAYER                       │
│  • demand_weather_silver                         │
│    (cleaned, joined demand + weather)           │
└─────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        v                       v
┌──────────────────┐  ┌─────────────────────┐
│  🥇 GOLD 24H     │  │  🥇 GOLD 7J         │
│  Features + Model│  │  Features + Model   │
│                   │  │                      │
│  • ml_features_   │  │  • ml_features_     │
│    gold_24h       │  │    gold_7j          │
│  • load_forecast_ │  │  • load_forecast_7j │
│    gold           │  │  • load_shap_7j     │
│  • load_shap_gold │  │                      │
└──────────────────┘  └─────────────────────┘
        │                       │
        └───────────┬───────────┘
                    v
┌─────────────────────────────────────────────────┐
│        🎯 MONITORING & ANALYTICS                 │
│  • model_performance_gold                        │
│    (MAE, RMSE, MAPE by zone and horizon)       │
└─────────────────────────────────────────────────┘
```

**📘 Detailed architecture:** See [`docs/medallion_architecture.md`](docs/medallion_architecture.md)

---

## 📊 Data & Unity Catalog

**Unity Catalog Location:**
- **Catalog:** `workspace`
- **Schema:** `energy_forecast`
- **Full path:** `workspace.energy_forecast.*`

### Key Tables

| Layer | Table | Full Path | Description |
|-------|-------|-----------|-------------|
| 🥉 Bronze | `load_actual_bronze` | `workspace.energy_forecast.load_actual_bronze` | Global Ontario demand |
| 🥉 Bronze | `load_zonal_bronze` | `workspace.energy_forecast.load_zonal_bronze` | Demand by zone (10 zones) |
| 🥉 Bronze | `weather_bronze` | `workspace.energy_forecast.weather_bronze` | Historical weather observations |
| 🥈 Silver | `demand_weather_silver` | `workspace.energy_forecast.demand_weather_silver` | Cleaned demand + weather |
| 🥇 Gold | `ml_features_gold_24h` | `workspace.energy_forecast.ml_features_gold_24h` | Features for 24h model |
| 🥇 Gold | `ml_features_gold_7j` | `workspace.energy_forecast.ml_features_gold_7j` | Features for 7-day model |
| 🥇 Gold | `load_forecast_gold` | `workspace.energy_forecast.load_forecast_gold` | 24h predictions |
| 🥇 Gold | `load_forecast_7j` | `workspace.energy_forecast.load_forecast_7j` | 7-day predictions |

**🔍 Explore tables:** [Unity Catalog Explorer](https://dbc-23fb8b12-3fa4.cloud.databricks.com/explore/data/workspace/energy_forecast)

---

## 🤖 Models

### Two Distinct Approaches

#### 🔄 Model 24h - Recursive
- **Single-step model** re-injected hour by hour up to 168h
- Each prediction serves as "lag" for the next hour
- **Training:** `pipeline/modeling/06_train_model_24h.py`
- **Inference:** `pipeline/inference/09_batch_prediction.py`

#### ⚡ Model 7j - Direct Multi-Horizon
- `forecast_horizon_hours` (1 to 168) is an input feature
- The model predicts each horizon in a single pass (no loop)
- **Training:** `pipeline/modeling/07_train_model_7j.py`
- **Inference:** `pipeline/inference/09b_batch_prediction_7j_direct.py`

### Feature Engineering

- **Lags:** 1h, 24h, 48h, 72h, 144h, 168h, 336h
- **Rolling windows:** mean, std, min, max over 3h, 12h, 24h, 48h, 72h, 168h
- **Temporal:** hour, day_of_week, month, is_weekend, is_holiday (Ontario)
- **Weather:** temperature, humidity, windspeed, HDD/CDD (base 18°C)
- **Cyclical encoding:** sin/cos for hour, day_of_week, month

### MLflow Tracking

| Model | Experiment | Registry Model Name |
|-------|------------|---------------------|
| 24h | `/Users/n.jouglet23@gmail.com/ontario_demand_forecast_24h` | `ontario_demand_lightgbm_24h` |
| 7j | `/Users/n.jouglet23@gmail.com/ontario_demand_forecast_7j` | `ontario_demand_lightgbm_7j` |

---

## 📂 Project Structure

```
energy_forecast/
├── pipeline/
│   ├── bronze/
│   │   ├── 01_ingest_ieso_demand.py              # IESO demand ingestion
│   │   └── 02_ingest_weather_historical.py       # Weather.gc.ca ingestion
│   ├── silver/
│   │   └── 03_join_demand_weather.py             # Join + cleaning
│   ├── gold/
│   │   ├── 04_build_features.py                  # Features 24h
│   │   ├── 04_build_features_7j.py               # Features 7j (multi-horizon)
│   │   └── 05_feature_selection.py               # Feature selection 24h
│   ├── modeling/
│   │   ├── 06_train_model_24h.py                 # Train 24h model
│   │   └── 07_train_model_7j.py                  # Train 7j model
│   ├── inference/
│   │   ├── 08_build_prediction_features.py       # Build features (Open-Meteo)
│   │   ├── 08b_build_prediction_features_7j.py   # Wrapper for 7j
│   │   ├── 09_batch_prediction.py                # Batch scoring 24h
│   │   └── 09b_batch_prediction_7j_direct.py     # Batch scoring 7j
│   └── monitoring/
│       └── 10_model_evaluation.py                # Continuous evaluation
├── config/
│   ├── config.yaml                    # Global configuration
│   ├── zones_config.py                # Ontario weather zones
│   └── selected_features.yaml         # Output of feature selection
├── utils/
│   ├── holidays_ontario.py
│   ├── weather_utils.py
│   └── feature_utils.py
├── docs/
│   ├── medallion_architecture.md      # 🆕 Detailed Unity Catalog architecture
│   └── architecture.md                # Technical documentation
├── dashboards/                        # Lakeview dashboards (.lvdash.json)
├── data/                              # Local data (archives, exports)
└── README.md                          # This file
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Databricks Workspace with Unity Catalog enabled
- Serverless compute or cluster with Python 3.10+
- Unity Catalog schema: `workspace.energy_forecast`

### 2. Configuration

Edit `config/config.yaml`:
- Unity Catalog tables
- Model hyperparameters
- Data sources endpoints
- Job schedules

All notebooks resolve the project path via the environment variable **`ENERGY_FORECAST_PROJECT_ROOT`** (default: `/Workspace/Users/n.jouglet23@gmail.com/energy_forecast`).

To deploy elsewhere, set this environment variable on your cluster/job — no files to modify.

### 3. Run the Pipeline

**Initial setup (backfill):**
```bash
# 1. Ingest historical data
pipeline/bronze/01_ingest_ieso_demand.py         # Set BACKFILL=True
pipeline/bronze/02_ingest_weather_historical.py

# 2. Transform
pipeline/silver/03_join_demand_weather.py
pipeline/gold/04_build_features.py               # → ml_features_gold_24h
pipeline/gold/04_build_features_7j.py            # → ml_features_gold_7j

# 3. Train models
pipeline/modeling/06_train_model_24h.py
pipeline/modeling/07_train_model_7j.py

# 4. Inference
pipeline/inference/09_batch_prediction.py             # 24h predictions
pipeline/inference/09b_batch_prediction_7j_direct.py  # 7j predictions

# 5. Evaluate
pipeline/monitoring/10_model_evaluation.py
```

**Scheduled runs (production):**

Configure Databricks Jobs (see `config.yaml -> jobs`):
- **Ingestion** (every 6h): Bronze layer updates
- **Processing** (every 6h): Silver → Gold
- **Training** (weekly): Retrain models
- **Prediction 24h** (every 6h): Batch forecasts 24h
- **Prediction 7j** (every 6h): Batch forecasts 7j
- **Evaluation** (daily): Model monitoring

---

## 📈 Performance

### Current Results

| Metric | 24h | 48h | 168h (7j) |
|--------|-----|-----|-----------|
| **MAPE** | ~2.0% | ~2.5% | ~3.5% |
| **MAE** | ~50 MW | ~70 MW | ~120 MW |
| **RMSE** | ~150 MW | ~180 MW | ~250 MW |

### Targets

| Horizon | Target MAPE | Status |
|---------|-------------|--------|
| 24h | < 2.0% | ✅ Met |
| 48h | < 2.5% | ✅ Met |
| 168h | < 3.5% | ✅ Met |

**📊 Dashboards:** See `dashboards/` for Lakeview visualizations

---

## 🔧 Configuration

### Main Configuration: `config/config.yaml`

```yaml
project:
  name: "ontario_demand_forecast"
  version: "1.1.0"

catalog:
  name: "workspace"
  schema: "energy_forecast"

models:
  horizon_24h:
    forecast_horizon_hours: 24
    mlflow:
      experiment_name: "/Users/n.jouglet23@gmail.com/ontario_demand_forecast_24h"
      registry_model_name: "ontario_demand_lightgbm_24h"
  
  horizon_7j:
    forecast_horizon_hours: 168
    mlflow:
      experiment_name: "/Users/n.jouglet23@gmail.com/ontario_demand_forecast_7j"
      registry_model_name: "ontario_demand_lightgbm_7j"
```

### Weather Zones: `config/zones_config.py`

10 Ontario zones with population-weighted coordinates:
- Toronto (25%), Ottawa (15%), West (15%), Southwest (10%), Niagara (10%), East (8%), Northeast (7%), Northwest (5%), Bruce (3%), Essa (2%)

---

## 📚 Documentation

- **📐 Medallion Architecture:** [`docs/medallion_architecture.md`](docs/medallion_architecture.md) - Unity Catalog tables, data flow, conventions
- **🛠️ Technical Details:** [`docs/architecture.md`](docs/architecture.md) - Detailed schemas and implementation
- **📝 Changelog:** [`CHANGELOG_CLEANUP.md`](CHANGELOG_CLEANUP.md) - Cleanup history (2026-08-29 + 2026-09-01)

---

## 🤝 Contributing

1. Follow the medallion architecture (Bronze → Silver → Gold)
2. Update `config.yaml` for new tables/models
3. Document Unity Catalog paths in `docs/medallion_architecture.md`
4. Run `pipeline/monitoring/10_model_evaluation.py` after model changes

---

## 📄 License

Internal project - Energy Forecast Team

---

## 📞 Contact

**Author:** Energy Forecast Team  
**Project:** Ontario Demand Forecast v1.1.0  
**Unity Catalog:** `workspace.energy_forecast`

---

**Last updated:** September 2026
