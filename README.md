# 🔌 Ontario Energy Demand Forecasting

**End-to-end ML pipeline on Databricks that forecasts Ontario's electricity demand by zone, at two horizons (24h and 7 days), and serves the results through live dashboards.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Databricks](https://img.shields.io/badge/Databricks-Unity%20Catalog-orange.svg)](https://databricks.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Model-green.svg)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue.svg)](https://mlflow.org/)

Built solo to practice production-style MLOps: a medallion architecture in Unity Catalog, two forecasting strategies for two horizons, SHAP-based explainability, scheduled batch inference and monitoring — all orchestrated as Databricks Jobs.

---

## 🎯 What it does

Ingests hourly Ontario electricity demand and weather data, engineers time-series features, and trains two LightGBM models:

| Model | Strategy | Training | Inference |
|-------|----------|----------|-----------|
| **24h** | Recursive — re-injects its own prediction as the next lag | `06_train_model_24h.py` | `09_batch_prediction_24h.py` |
| **7 days (168h)** | Direct multi-horizon — `forecast_horizon_hours` is a model input, no loop | `07_train_model_7j.py` | `09b_batch_prediction_7j.py` |

**Data sources:** IESO (demand + generation/capacity), Weather.gc.ca (historical, training) & Open-Meteo (forecasts, inference), Ontario calendar/holidays.

Both models log to MLflow, register in Unity Catalog, and run on a schedule via Databricks Jobs. Predictions and SHAP values feed three Lakeview dashboards below.

---

## 📊 Dashboards

**7-day forecast**, with top SHAP drivers and demand by zone:
![7-day forecast dashboard](docs/images/dashboard_forecast_7j.png)

**24h forecast**, refreshed hourly:
![24h forecast dashboard](docs/images/dashboard_forecast_24h.png)

**Generation & capacity**, by zone/fuel type and utilization rate:
![Production dashboard](docs/images/dashboard_production.png)

Dashboard source files: [`dashboards/`](dashboards/) (`.lvdash.json`, importable into Databricks Lakeview).

---

## 🏗️ Architecture

```
IESO API · Weather.gc.ca · Open-Meteo
                │
                v
🥉 BRONZE  (energy_forecast_bronze)   raw demand, zonal demand, weather
                │
                v
🥈 SILVER  (energy_forecast_silver)   demand + weather, cleaned & joined
                │
        ┌───────┴───────┐
        v               v
🥇 GOLD — 24h        🥇 GOLD — 7j        (both: energy_forecast_gold)
features · forecast · SHAP    features · forecast · SHAP
        └───────┬───────┘
                v
🎯 MONITORING   model_performance — MAE/RMSE/MAPE by zone & horizon
```

Unity Catalog uses one schema per medallion layer rather than a flat schema, to keep permissions and lineage aligned with Bronze/Silver/Gold. Full table-by-table detail: [`docs/medallion_architecture.md`](docs/medallion_architecture.md).

**Feature engineering:** lags (1h–336h), rolling stats (3h–168h windows), calendar/cyclical encoding, weather + degree-days. SHAP values are persisted per prediction and power the "top drivers" charts above.

---

## 📂 Project structure

```
pipeline/
├── bronze/     01_ingest_ieso_demand_new.py, 01_import_gen.py, 02_ingest_weather_historical.py
├── silver/     03_join_demand_weather.py
├── gold/       04_build_features(_7j).py, 05_feature_selection.py
├── modeling/   06_train_model_24h.py, 07_train_model_7j.py
├── inference/  08_build_prediction_features(_7j).py, 09_batch_prediction_24h.py, 09b_batch_prediction_7j.py
└── monitoring/ 10_model_evaluation.py

config/    config.yaml, zones_config.py, selected_features.yaml
utils/     holidays_ontario.py, weather_utils.py, feature_utils.py
docs/      medallion_architecture.md, architecture.md, images/
dashboards/  Lakeview .lvdash.json files
```


---

## 🚀 Quick start

Requires a Databricks workspace with Unity Catalog and Python 3.10+. The project path is set via env var `ENERGY_FORECAST_PROJECT_ROOT`; everything else (tables, hyperparameters, job schedules) lives in `config/config.yaml`.

```bash
pipeline/bronze/01_ingest_ieso_demand_new.py      # ingest (+ 02_ingest_weather_historical.py)
pipeline/silver/03_join_demand_weather.py         # clean & join
pipeline/gold/04_build_features(_7j).py           # features
pipeline/modeling/06_train_model_24h.py           # train (+ 07_train_model_7j.py)
pipeline/inference/09_batch_prediction_24h.py     # score (+ 09b_..._7j.py)
pipeline/monitoring/10_model_evaluation.py        # evaluate
```

In production this runs as scheduled Databricks Jobs: ingestion/transform every 6h, weekly retraining, scoring every 6h, daily evaluation.

---

## 📚 Documentation

- [`docs/medallion_architecture.md`](docs/medallion_architecture.md) — Unity Catalog tables & data flow
- [`docs/architecture.md`](docs/architecture.md) — technical implementation notes
- [`CHANGELOG_CLEANUP.md`](CHANGELOG_CLEANUP.md) — cleanup history
