#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gold Layer - 02 Feature Selection

Objectif: Analyser et sélectionner les meilleures features pour le ML
- Corrélation avec la target
- Feature importance (LightGBM)
- Analyse de collinéarité
- Recommandations de features à garder

Source: workspace.energy_forecast.ml_features_gold_24h
Output: Rapport + config YAML

Ce script est dédié au modèle 24h : le modèle 7 jours a sa propre table Gold
(`ml_features_gold_7j`, voir `pipeline/gold/04_build_features_7j.py`) et sa
propre sélection de features intégrée à `07_train_model_7j.py`.
"""

# Install required packages
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Chemin du projet : surchargeable via ENERGY_FORECAST_PROJECT_ROOT.
PROJECT_ROOT = os.environ.get(
    "ENERGY_FORECAST_PROJECT_ROOT",
    "/Workspace/Users/n.jouglet23@gmail.com/energy_forecast_clean",
)

# Configuration
with open(f'{PROJECT_ROOT}/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

CATALOG = config['catalog']['name']
SCHEMA = config['catalog']['schema']

# NOTE (nettoyage complémentaire): pointait auparavant vers `ml_features_gold`,
# une table jamais créée par aucun script (04_build_features.py écrit en
# réalité dans `ml_features_gold_24h`) — ce qui aurait fait échouer ce script
# à l'exécution. Corrigé pour pointer vers la vraie table de sortie du 24h.
INPUT_TABLE = f"{CATALOG}.{SCHEMA}.ml_features_gold_24h"

print(f"🎯 FEATURE SELECTION pour ML")
print(f"Source: {INPUT_TABLE}\n")

# ============================================================================
# 1. LOAD DATA & PREPARE
# ============================================================================

print("📥 Chargement des données...")

spark = SparkSession.builder.getOrCreate()
df_spark = spark.table(INPUT_TABLE)

# Filtrer lignes avec target valide
df_spark_valid = df_spark.filter(F.col("target_demand_mw").isNotNull())

# Échantillon pour analyse rapide
sample_fraction = 0.2
print(f"  Échantillonnage: {sample_fraction*100:.0f}% pour analyse rapide")
df_sample = df_spark_valid.sample(withReplacement=False, fraction=sample_fraction, seed=42)

print(f"  ✅ {df_sample.count():,} lignes chargées\n")

df_pandas = df_sample.toPandas()

# ============================================================================
# 2. DÉFINIR GROUPES DE FEATURES
# ============================================================================

print("🔍 Classification des features...\n")

feature_groups = {
    'temporal_cyclic': [
        'issue_hour_sin', 'issue_hour_cos',
        'target_day_of_week_sin', 'target_day_of_week_cos',
        'target_month_sin', 'target_month_cos',
        'target_day_of_year_sin', 'target_day_of_year_cos'
    ],
    'temporal_categorical': [
        'issue_hour', 'target_day_of_week', 'target_day_of_year', 'target_is_weekend',
        'target_time_morning', 'target_time_afternoon', 'target_time_evening', 'target_time_night'
    ],
    'weather': [
        'temperature_c', 'dew_point_c', 'relative_humidity_pct',
        'wind_speed_kmh', 'precipitation_mm',
        'weather_target_hdd18', 'weather_target_cdd18'
    ],
    'lag': [
        'demand_lag_1h', 'demand_lag_2h', 'demand_lag_3h', 'demand_lag_6h',
        'demand_lag_12h', 'demand_lag_24h', 'demand_lag_48h', 'demand_lag_168h'
    ],
    'rolling': [
        'demand_rolling_mean_6h', 'demand_rolling_mean_12h',
        'demand_rolling_mean_24h', 'demand_rolling_mean_168h',
        'demand_rolling_std_24h', 'demand_rolling_std_168h'
    ],
    'calendar': [
        'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday'
    ]
}

all_features = []
for group in feature_groups.values():
    all_features.extend(group)

print(f"Total features: {len(all_features)}")
for group_name, features in feature_groups.items():
    print(f"  - {group_name}: {len(features)} features")

# ============================================================================
# 3. CORRÉLATION AVEC TARGET
# ============================================================================

print("\n📊 1. CORRÉLATION AVEC TARGET\n")

df_clean = df_pandas[all_features + ['target_demand_mw']].dropna()
correlations = df_clean[all_features].corrwith(df_clean['target_demand_mw']).abs().sort_values(ascending=False)

print("Top 15 features les plus corrélées:")
for i, (feature, corr) in enumerate(correlations.head(15).items(), 1):
    group = next((g for g, feats in feature_groups.items() if feature in feats), 'other')
    print(f"  {i:2d}. {feature:30s} {corr:.4f}  [{group}]")

CORR_THRESHOLD = 0.05
low_corr = correlations[correlations < CORR_THRESHOLD]
print(f"\n⚠️  Features faiblement corrélées (< {CORR_THRESHOLD}): {len(low_corr)}")

# ============================================================================
# 4. FEATURE IMPORTANCE (LIGHTGBM)
# ============================================================================

print("\n📊 2. FEATURE IMPORTANCE (LightGBM)\n")

X = df_clean[all_features]
y = df_clean['target_demand_mw']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Training: {len(X_train):,} | Test: {len(X_test):,}")

print("\n  🔧 Entraînement LightGBM...")

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
)

print(f"  ✅ Modèle entraîné ({model.best_iteration} iterations)")

importances = model.feature_importance(importance_type='gain')
feature_importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n  Top 15 features les plus importantes:")
for i, row in feature_importance_df.head(15).iterrows():
    group = next((g for g, feats in feature_groups.items() if row['feature'] in feats), 'other')
    pct = (row['importance'] / importances.sum()) * 100
    print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:8.0f} ({pct:5.2f}%)  [{group}]")

# ============================================================================
# 5. COLLINÉARITÉ
# ============================================================================

print("\n📊 3. COLLINÉARITÉ\n")

corr_matrix = df_clean[all_features].corr().abs()

COLLIN_THRESHOLD = 0.9
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > COLLIN_THRESHOLD:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

if high_corr_pairs:
    print(f"  ⚠️  {len(high_corr_pairs)} paires fortement corrélées (> {COLLIN_THRESHOLD}):")
    for feat1, feat2, corr in high_corr_pairs[:10]:
        print(f"    - {feat1:30s} <-> {feat2:30s}  {corr:.4f}")
else:
    print(f"  ✅ Aucune collinéarité forte (> {COLLIN_THRESHOLD})")

# ============================================================================
# 6. RECOMMANDATIONS
# ============================================================================

print("\n" + "="*80)
print("🎯 RECOMMANDATIONS")
print("="*80)

feature_scores = pd.DataFrame({
    'feature': all_features,
    'correlation': correlations.reindex(all_features).fillna(0),
    'importance': feature_importance_df.set_index('feature')['importance'].reindex(all_features).fillna(0)
})

feature_scores['corr_rank'] = feature_scores['correlation'].rank(ascending=False)
feature_scores['imp_rank'] = feature_scores['importance'].rank(ascending=False)
feature_scores['avg_rank'] = (feature_scores['corr_rank'] + feature_scores['imp_rank']) / 2
feature_scores = feature_scores.sort_values('avg_rank')

print("\n📋 TOP 20 FEATURES (score combiné):\n")
for i, row in feature_scores.head(20).iterrows():
    group = next((g for g, feats in feature_groups.items() if row['feature'] in feats), 'other')
    print(f"  {int(row['avg_rank']):2d}. {row['feature']:30s} [corr={row['correlation']:.3f}, imp={row['importance']:6.0f}]  [{group}]")

# Sélection par groupe
print("\n📊 SÉLECTION PAR GROUPE:\n")

selected_features = []

for group_name, features in feature_groups.items():
    group_scores = feature_scores[feature_scores['feature'].isin(features)].sort_values('avg_rank')
    
    if group_name == 'lag':
        n_keep = 5
    elif group_name == 'rolling':
        n_keep = 4
    elif group_name == 'temporal_cyclic':
        n_keep = 6
    elif group_name == 'weather':
        n_keep = len(features)
    else:
        n_keep = max(2, len(features) // 2)
    
    top_features = group_scores.head(n_keep)['feature'].tolist()
    selected_features.extend(top_features)
    
    print(f"  {group_name:20s}: {len(top_features):2d}/{len(features):2d} features")
    for feat in top_features:
        print(f"    ✓ {feat}")

print(f"\n✅ TOTAL: {len(selected_features)}/{len(all_features)} features")
print(f"   Réduction: {len(all_features) - len(selected_features)} features ({(1 - len(selected_features)/len(all_features))*100:.1f}%)")

# ============================================================================
# 7. SAUVEGARDE
# ============================================================================

print("\n💾 Sauvegarde...")

selection_config = {
    'selected_features': selected_features,
    'selection_date': pd.Timestamp.now().isoformat(),
    'total_features': len(all_features),
    'selected_count': len(selected_features),
    'criteria': {
        'correlation_threshold': CORR_THRESHOLD,
        'collinearity_threshold': COLLIN_THRESHOLD
    }
}

output_path = f'{PROJECT_ROOT}/config/selected_features.yaml'

with open(output_path, 'w') as f:
    yaml.dump(selection_config, f, default_flow_style=False)

print(f"  ✅ Sauvegardé: {output_path}")

print("\n" + "="*80)
print("✅ FEATURE SELECTION TERMINÉE")
print("="*80)
print("\n💡 PROCHAINES ÉTAPES:")
print("  1. Revoir la sélection et ajuster si nécessaire")
print(f"  2. Utiliser {output_path} dans le training")
print("  3. Comparer performance avec/sans sélection\n")
