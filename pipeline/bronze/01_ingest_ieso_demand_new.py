#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01 - Ingestion IESO Demand (Bronze Layer)
=========================================

Récupère les données de demande électrique (globale + zonale) depuis l'API
publique IESO et les charge dans les tables Bronze.

Source unique (optimisé):
- https://reports-public.ieso.ca/public/DemandZonal/PUB_DemandZonal_{year}.csv
  (contient à la fois les zones géographiques ET les totaux Ontario/Market)

Destinations:
- {catalog}.{schema}.load_actual_bronze (demande globale : market_demand, ontario_demand)
- {catalog}.{schema}.load_zonal_bronze (demande par zone : East, West, Toronto, Ottawa, etc.)

Fréquence: Toutes les heures (cron job)

Note historique: Pour le chargement initial (2020 → aujourd'hui), régler
BACKFILL=true via la variable d'environnement. En fonctionnement normal (cron),
laisser BACKFILL=false (ou non défini) pour ne récupérer qu'une fenêtre glissante
récente (moins de données à retélécharger).

Historique local: un export statique 2020-présent est disponible dans
data/archive/ieso_zonal_demand_2020_present.csv et peut servir de source de secours
pour le premier chargement si l'API ne sert pas facilement plusieurs années d'historique.

Optimisation v3 (2026-08-30):
- Une seule source (DemandZonal) au lieu de deux requêtes API redondantes
- Extraction intelligente : Ontario Demand + Market Demand → table globale
- Zones géographiques réelles → table zonale
- Exclusion explicite des colonnes agrégées (`Zone Total`, `Diff`) du melt
- Gestion robuste des heures ambiguës (`ambiguous=False`, `nonexistent='shift_forward'`)

Next: 02_ingest_weather_historical.py
"""

import os
import requests
import pandas as pd
import yaml
from io import StringIO
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# Colonnes de la source zonale IESO à exclure car ce ne sont pas de vraies zones
# géographiques mais des agrégats/doublons (voir en-tête du CSV IESO DemandZonal).
EXCLUDED_ZONE_COLUMNS = {"zone total", "diff", "market demand", "market_demand"}


def load_config() -> dict:
    """Charge la configuration du projet depuis config.yaml."""
    project_root = os.environ.get(
        "ENERGY_FORECAST_PROJECT_ROOT",
        "/Workspace/Users/n.jouglet23@gmail.com/energy_forecast_clean",
    )
    config_path = f"{project_root}/config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    catalog = config["catalog"]["name"]
    schema = config["catalog"]["schema"]
    table_global = f"{catalog}.{schema}.{config['catalog']['tables']['bronze']['load_actual']}"
    table_zonal = f"{catalog}.{schema}.{config['catalog']['tables']['bronze']['load_zonal']}"

    return {
        "project_root": project_root,
        "catalog": catalog,
        "schema": schema,
        "table_global": table_global,
        "table_zonal": table_zonal,
    }


def fetch_ieso_csv(data_type: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Télécharge et consolide les CSV IESO annuels pour un type de donnée donné.

    Args:
        data_type: "Demand" (global) ou "DemandZonal" (par zone)
        start_date: Date de début
        end_date: Date de fin

    Returns:
        DataFrame pandas consolidé, filtré sur [start_date, end_date]
    """
    years = range(start_date.year, end_date.year + 1)
    all_dfs = []

    print(f"🔄 Fetching {data_type}...")

    for year in years:
        url = f"https://reports-public.ieso.ca/public/{data_type}/PUB_{data_type}_{year}.csv"
        print(f"  {year}...", end=" ")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # L'API IESO retourne 3 lignes de métadonnées avant le header réel.
            df = pd.read_csv(StringIO(response.text), skiprows=3)

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
            df = df.dropna(subset=["Date", "Hour"])
            df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

            all_dfs.append(df)
            print(f"✅ {len(df):,}")

        except Exception as e:
            print(f"⚠️ Erreur: {e}")
            continue

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True).sort_values(["Date", "Hour"]).reset_index(drop=True)
        print(f"  Total: {len(result):,} lignes\n")
        return result

    print("  ❌ Aucune donnée\n")
    return pd.DataFrame()


def create_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la colonne `datetime` (Date + Hour-1, l'heure IESO va de 1 à 24) et
    convertit de America/Toronto (heure IESO) vers UTC naïf.

    `ambiguous=False` choisit explicitement la 1re occurrence lors du repli
    d'automne (fall back) plutôt que de laisser pandas deviner (`infer` peut
    lever une exception si le motif horaire n'est pas clairement DST->non-DST).
    `nonexistent='shift_forward'` gère l'heure inexistante au passage à l'heure
    d'été (spring forward).
    """
    if df.empty:
        return df

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"] - 1, unit="h")
    df["datetime"] = (
        df["datetime"]
        .dt.tz_localize("America/Toronto", ambiguous=False, nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    return df


def merge_to_table(spark, spark_df, table_name: str, merge_keys: list, update_cols: list):
    """Fonction générique de MERGE (upsert) dans une table Unity Catalog."""
    temp_view = f"temp_{table_name.split('.')[-1]}"
    spark_df.createOrReplaceTempView(temp_view)

    on_clause = " AND ".join([f"target.{k} = source.{k}" for k in merge_keys])
    set_clause = ", ".join([f"target.{col} = source.{col}" for col in update_cols])
    all_cols = spark_df.columns
    insert_cols = ", ".join(all_cols)
    insert_values = ", ".join([f"source.{col}" for col in all_cols])

    merge_query = f"""
    MERGE INTO {table_name} AS target
    USING {temp_view} AS source
    ON {on_clause}
    WHEN MATCHED THEN
        UPDATE SET {set_clause}
    WHEN NOT MATCHED THEN
        INSERT ({insert_cols})
        VALUES ({insert_values})
    """

    print(f"🔄 MERGE into {table_name}...")
    spark.sql(merge_query)
    print("✅ MERGE completed")


def ingest_global_and_zonal(spark, df_zonal: pd.DataFrame, cfg: dict) -> tuple:
    """
    Extrait les données globales et zonales du DataFrame IESO et les MERGE
    dans les tables Bronze.

    Returns:
        (count_global, count_zonal, n_zones)
    """
    table_global = cfg["table_global"]
    table_zonal = cfg["table_zonal"]
    catalog = cfg["catalog"]
    schema = cfg["schema"]

    # --- ÉTAPE 1 : Extraction des données GLOBALES (Ontario + Market Demand) ---
    print("\n🌍 Extraction des données globales...")

    df_global = pd.DataFrame({
        "datetime": df_zonal["datetime"],
        "market_demand": pd.to_numeric(df_zonal.get("Market Demand", None), errors="coerce"),
        "ontario_demand": pd.to_numeric(df_zonal.get("Ontario Demand", None), errors="coerce"),
        "ingestion_time": datetime.now(),
    }).dropna(subset=["datetime"])

    print(f"📊 Global : {len(df_global):,} lignes extraites")
    print(f"Période: {df_global['datetime'].min()} → {df_global['datetime'].max()}")

    spark_df_global = spark.createDataFrame(df_global)
    spark_df_global_deduped = spark_df_global.dropDuplicates(["datetime"])

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_global} (
            datetime TIMESTAMP,
            market_demand DOUBLE,
            ontario_demand DOUBLE,
            ingestion_time TIMESTAMP
        )
        USING DELTA
        PARTITIONED BY (date(datetime))
        COMMENT 'Bronze - IESO demand (global) - extrait du fichier DemandZonal'
    """)

    merge_to_table(
        spark,
        spark_df_global_deduped,
        table_global,
        merge_keys=["datetime"],
        update_cols=["market_demand", "ontario_demand", "ingestion_time"],
    )

    count_global = spark.table(table_global).count()
    print(f"✅ Table globale : {count_global:,} lignes totales\n")

    # --- ÉTAPE 2 : Extraction des données ZONALES (vraies zones géographiques) ---
    print("🗺️ Extraction des zones géographiques...")

    technical_columns = {"Date", "Hour", "datetime", "SourceYear"}
    all_columns = df_zonal.columns.tolist()
    candidate_zone_cols = [c for c in all_columns if c not in technical_columns]

    zone_cols = [c for c in candidate_zone_cols if c.strip().lower() not in EXCLUDED_ZONE_COLUMNS]
    excluded_found = [c for c in all_columns if c not in zone_cols and c not in ["datetime"]]

    print(f"📍 Zones géographiques retenues: {zone_cols}")
    print(f"🚫 Colonnes exclues (agrégats/non-zones): {excluded_found}")

    df_zonal_long = df_zonal.melt(
        id_vars=["datetime"],
        value_vars=zone_cols,
        var_name="zone",
        value_name="demand_mw",
    )

    df_zonal_long["demand_mw"] = pd.to_numeric(df_zonal_long["demand_mw"], errors="coerce")
    df_zonal_long["ingestion_time"] = datetime.now()
    df_zonal_long = df_zonal_long.dropna(subset=["datetime", "zone", "demand_mw"])

    # Renommer "Ontario Demand" en "Ontario" pour cohérence de nommage.
    df_zonal_long["zone"] = df_zonal_long["zone"].str.strip().replace("Ontario Demand", "Ontario")

    print(f"📊 Zonal : {len(df_zonal_long):,} lignes transformées")
    print(f"Zones: {sorted(df_zonal_long['zone'].unique())}\n")

    spark_df_zonal = spark.createDataFrame(df_zonal_long)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_zonal} (
            datetime TIMESTAMP,
            zone STRING,
            demand_mw DOUBLE,
            ingestion_time TIMESTAMP
        )
        USING DELTA
        PARTITIONED BY (zone)
        COMMENT 'Bronze - IESO zonal demand (raw)'
    """)

    spark_df_zonal_deduped = spark_df_zonal.dropDuplicates(["datetime", "zone"])

    merge_to_table(
        spark,
        spark_df_zonal_deduped,
        table_zonal,
        merge_keys=["datetime", "zone"],
        update_cols=["demand_mw", "ingestion_time"],
    )

    count_zonal = spark.table(table_zonal).count()
    print(f"✅ Table zonale : {count_zonal:,} lignes totales\n")

    return count_global, count_zonal, len(zone_cols)


def validate(spark, cfg: dict):
    """Vérifie les tables Bronze après ingestion."""
    table_global = cfg["table_global"]
    table_zonal = cfg["table_zonal"]

    print("🔍 VALIDATION FINALE")
    print("=" * 70)

    # Vérification table globale
    try:
        stats_global = spark.table(table_global).agg(
            F.min("datetime").alias("min"),
            F.max("datetime").alias("max"),
            F.count("*").alias("count"),
        ).collect()[0]
        print(f"🌍 Global : {stats_global['count']:,} lignes")
        print(f"   Période: {stats_global['min']} → {stats_global['max']}")
    except Exception as e:
        print(f"⚠️ Table globale non disponible: {e}")

    # Vérification table zonale
    try:
        stats_zonal = spark.table(table_zonal).agg(
            F.min("datetime").alias("min"),
            F.max("datetime").alias("max"),
            F.count("*").alias("count"),
            F.countDistinct("zone").alias("zones"),
        ).collect()[0]
        print(f"\n🗺️ Zonal  : {stats_zonal['count']:,} lignes ({stats_zonal['zones']} zones)")
        print(f"   Période: {stats_zonal['min']} → {stats_zonal['max']}")

        null_count = spark.table(table_zonal).filter(F.col("demand_mw").isNull()).count()
        if null_count > 0:
            print(f"   ⚠️ Valeurs nulles: {null_count}")
    except Exception as e:
        print(f"⚠️ Table zonale non disponible: {e}")

    print("\n" + "=" * 70)
    print("✅ Ingestion Bronze IESO terminée avec succès")
    print("=" * 70)


def main():
    """
    Point d'entrée principal du script d'ingestion IESO.

    Variables d'environnement:
        ENERGY_FORECAST_PROJECT_ROOT : chemin racine du projet (défaut: chemin absolu)
        BACKFILL                     : "true" pour chargement initial complet depuis 2020
        INCREMENTAL_WINDOW_DAYS      : fenêtre glissante en jours pour le mode cron (défaut: 7)
    """
    # BACKFILL=true  -> charge tout l'historique depuis 2020 (à utiliser une seule fois)
    # BACKFILL=false -> ne charge qu'une fenêtre glissante récente (mode incrémental / cron)
    backfill = os.environ.get("BACKFILL", "false").lower() == "true"
    incremental_window_days = int(os.environ.get("INCREMENTAL_WINDOW_DAYS", "7"))

    if backfill:
        start_date = datetime(2020, 1, 1)
    else:
        start_date = datetime.now() - timedelta(days=incremental_window_days)
    end_date = datetime.now()

    cfg = load_config()
    spark = SparkSession.builder.getOrCreate()

    print(f"📊 Ingestion IESO Demand → {cfg['table_global']} / {cfg['table_zonal']}")
    print(
        f"📅 Période: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')} "
        f"({'backfill complet' if backfill else 'fenêtre incrémentale'})\n"
    )

    # --- Fetch ---
    df_zonal = fetch_ieso_csv("DemandZonal", start_date, end_date)

    if df_zonal.empty:
        # Fallback archive locale
        archive_candidates = [
            f"{cfg['project_root']}/data/archive/ieso_zonal_demand_2020_present.parquet",
            f"{cfg['project_root']}/data/archive/ieso_zonal_demand_2020_present.csv",
        ]
        for archive_path in archive_candidates:
            try:
                if archive_path.endswith(".parquet"):
                    df_zonal = pd.read_parquet(archive_path)
                else:
                    df_zonal = pd.read_csv(archive_path)

                print(f"📂 Fallback archive locale utilisée: {archive_path}")
                df_zonal["Date"] = pd.to_datetime(df_zonal["Date"], errors="coerce")
                df_zonal["Hour"] = pd.to_numeric(df_zonal["Hour"], errors="coerce")
                df_zonal = df_zonal.dropna(subset=["Date", "Hour"])
                df_zonal = df_zonal[(df_zonal["Date"] >= start_date) & (df_zonal["Date"] <= end_date)]
                break
            except FileNotFoundError:
                continue

    if df_zonal is not None and not df_zonal.empty:
        df_zonal = create_datetime_column(df_zonal)
        count_global, count_zonal, n_zones = ingest_global_and_zonal(spark, df_zonal, cfg)

        print("=" * 70)
        print("✅ OPTIMISATION RÉUSSIE : 1 source → 2 tables")
        print(f"   • Global : {count_global:,} lignes (market + ontario demand)")
        print(f"   • Zonal  : {count_zonal:,} lignes ({n_zones} zones géographiques)")
        print(f"   • API calls économisés : 50% (1 au lieu de 2 par année)")
        print("=" * 70)
    else:
        print("⚠️ Aucune donnée disponible (ni API IESO, ni archive locale)\n")

    validate(spark, cfg)


if __name__ == "__main__":
    main()
