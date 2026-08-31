#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02 - Ingestion Weather.gc.ca (Bronze Layer)
Code adapté de notebook:4400736406054312:Import_meteo_zone

Source: Weather.gc.ca Climate Hourly API
Destination: workspace.energy_forecast.weather_bronze
"""

import os
import requests
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from time import sleep
from pyspark.sql import SparkSession
from pyspark.sql import functions as F, Window

# Configuration
with open('/Workspace/Users/n.jouglet23@gmail.com/energy_forecast_clean/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

CATALOG = config['catalog']['name']
SCHEMA = config['catalog']['schema']
TABLE_NAME = "weather_bronze"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

API_URL = "https://api.weather.gc.ca/collections/climate-hourly/items"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Période d'ingestion (modifier pour incrémental)
START_DATE = "2020-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

print(f"☁️ Ingestion Weather.gc.ca → {FULL_TABLE_NAME}")
print(f"Période: {START_DATE} à {END_DATE}\n")

# ============================================================================
# VÉRIFICATION IMPORTATION RÉCENTE (< 4h)
# ============================================================================

REIMPORT_THRESHOLD_HOURS = 4

def should_skip_import(table_name: str) -> bool:
    """
    Vérifie si une importation récente (< 4h) existe déjà.
    Retourne True si on doit skip l'importation.
    """
    try:
        spark = SparkSession.builder.getOrCreate()
        
        # Vérifier si la table existe
        if not spark.catalog.tableExists(table_name):
            print(f"ℹ️  Table {table_name} n'existe pas encore → importation nécessaire")
            return False
        
        # Récupérer la dernière modification de la table
        table_history = spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT 1").collect()
        
        if not table_history:
            print(f"ℹ️  Pas d'historique pour {table_name} → importation nécessaire")
            return False
        
        last_modified = table_history[0]['timestamp']
        hours_since_last_import = (datetime.now() - last_modified.replace(tzinfo=None)).total_seconds() / 3600
        
        if hours_since_last_import < REIMPORT_THRESHOLD_HOURS:
            print(f"\n✅ Importation récente détectée pour {table_name}")
            print(f"   Dernière importation : {last_modified.strftime('%Y-%m-%d %H:%M:%S')} ({hours_since_last_import:.1f}h)")
            print(f"   Seuil de ré-importation : {REIMPORT_THRESHOLD_HOURS}h")
            print(f"   ⏭️  SKIP : Utilisation des données existantes\n")
            return True
        else:
            print(f"ℹ️  Dernière importation : {last_modified.strftime('%Y-%m-%d %H:%M:%S')} ({hours_since_last_import:.1f}h)")
            print(f"   → Importation nécessaire (> {REIMPORT_THRESHOLD_HOURS}h)\n")
            return False
            
    except Exception as e:
        print(f"⚠️  Erreur lors de la vérification : {e}")
        print("   → Importation par défaut\n")
        return False

# Vérifier si on doit skip
if should_skip_import(FULL_TABLE_NAME):
    print("🎯 Table à jour → Fin du script")
else:
    print("🔄 Lancement de l'importation météo...\n")

    # ============================================================================
    # ZONE MAPPING
    # ============================================================================

    ZONE_STATIONS = {
        "Northwest": {"station_name": "SIOUX LOOKOUT AIRPORT", "climate_id": "6037800"},
        "Northeast": {"station_name": "TIMMINS CLIMATE", "climate_id": "6078282"},
        "Ottawa": {"station_name": "OTTAWA CDA RCS", "climate_id": "6105978"},
        "East": {"station_name": "BROCKVILLE CLIMATE", "climate_id": "6100970"},
        "Toronto": {"station_name": "TORONTO CITY CENTRE", "climate_id": "6158359"},
        "Essa": {"station_name": "BARRIE-ORO", "climate_id": "6117700"},
        "Bruce": {"station_name": "TOBERMORY RCS", "climate_id": "6128330"},
        "Southwest": {"station_name": "WINDSOR A", "climate_id": "6139530"},
        "Niagara": {"station_name": "VINELAND STATION RCS", "climate_id": "6139148"},
        "West": {"station_name": "KITCHENER/WATERLOO", "climate_id": "6144239"}
    }

    ZONE_COORDS = {
        zone.capitalize(): {
            "latitude": values["latitude"],
            "longitude": values["longitude"],
        }
        for zone, values in config.get("weather_zones", {}).items()
    }

    # ============================================================================
    # ARCHIVE LOCALE (fallback si l'API est inaccessible depuis ce compute)
    # ============================================================================
    # Déposer un fichier parquet dans data/archive/weather_gcca_historical.parquet
    # pour court-circuiter l'API. Format attendu : colonnes zone, datetime,
    # datetime_local, temperature_c, dew_point_c, relative_humidity_pct,
    # wind_speed_kmh, precipitation_mm.
    #
    # Pour créer l'archive depuis un cluster avec accès internet :
    #   weather_final.to_parquet("/tmp/weather_gcca_historical.parquet")
    #   puis déposer dans data/archive/ via dbutils.fs.cp ou l'UI Databricks.

    ARCHIVE_BASE = (
        "/Workspace/Users/n.jouglet23@gmail.com"
        "/energy_forecast_clean/data/archive"
    )
    WEATHER_ARCHIVE_PATH = f"{ARCHIVE_BASE}/weather_gcca_historical.parquet"
    USE_ARCHIVE = os.path.exists(WEATHER_ARCHIVE_PATH)

    if USE_ARCHIVE:
        print(f"📂 Archive locale détectée → {WEATHER_ARCHIVE_PATH}")
        print("   (téléchargement Weather.gc.ca ignoré)\n")
    else:
        print("🌐 Aucune archive locale — téléchargement via Weather.gc.ca\n")

    # ============================================================================
    # DOWNLOAD FUNCTIONS
    # ============================================================================

    session = requests.Session()

    def geojson_to_df(payload):
        """Convertit GeoJSON Weather.gc.ca en DataFrame"""
        rows = []
        for feature in payload.get("features", []):
            row = feature["properties"].copy()
            rows.append(row)
        return pd.DataFrame(rows)


    def is_name_resolution_error(exc):
        message = str(exc)
        return (
            "Failed to resolve" in message
            or "Temporary failure in name resolution" in message
        )


    def download_zone_weather_gcca(zone, climate_id):
        """Télécharge les données météo Weather.gc.ca pour une zone."""
        print(f"Téléchargement {zone} via Weather.gc.ca")

        months = pd.period_range(START_DATE, END_DATE, freq="M")
        dfs = []

        for month in months:
            start = month.start_time.strftime("%Y-%m-%d")
            end = month.end_time.strftime("%Y-%m-%d")

            params = {
                "f": "json",
                "CLIMATE_IDENTIFIER": climate_id,
                "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
                "limit": 1000,
            }

            next_url = API_URL
            next_params = params
            pages = []

            while next_url:
                try:
                    r = session.get(next_url, params=next_params, timeout=120)
                    r.raise_for_status()
                except requests.exceptions.RequestException as exc:
                    if is_name_resolution_error(exc):
                        raise RuntimeError("WEATHER_GCCA_DNS") from exc
                    raise

                payload = r.json()

                df_page = geojson_to_df(payload)
                if not df_page.empty:
                    pages.append(df_page)

                next_url = None
                for link in payload.get("links", []):
                    if link.get("rel") == "next":
                        next_url = link["href"]
                        next_params = None
                        break

            if pages:
                month_df = pd.concat(pages, ignore_index=True)
                month_df["ZONE"] = zone
                dfs.append(month_df)

            sleep(0.2)

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


    def request_openmeteo_with_retry(params, max_attempts=6):
        """Appelle Open-Meteo avec backoff pour éviter les 429."""
        wait_seconds = 5

        for attempt in range(1, max_attempts + 1):
            response = session.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=120)

            if response.status_code != 429:
                response.raise_for_status()
                return response

            retry_after = response.headers.get("Retry-After")
            wait_seconds = int(retry_after) if retry_after else wait_seconds
            print(
                f"  ⏳ Open-Meteo rate limit (tentative {attempt}/{max_attempts}), "
                f"pause {wait_seconds}s..."
            )
            sleep(wait_seconds)
            wait_seconds = min(wait_seconds * 2, 60)

        response.raise_for_status()
        return response


    def download_zone_weather_openmeteo(zone, latitude, longitude):
        """Fallback archive Open-Meteo quand Weather.gc.ca est inaccessible."""
        print(f"Téléchargement {zone} via Open-Meteo Archive")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": "temperature_2m,dewpoint_2m,relativehumidity_2m,windspeed_10m,precipitation",
            "timezone": "UTC",
        }

        response = request_openmeteo_with_retry(params)
        payload = response.json()

        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        df = pd.DataFrame({
            "UTC_DATE": times,
            "TEMP": hourly.get("temperature_2m"),
            "DEW_POINT_TEMP": hourly.get("dewpoint_2m"),
            "RELATIVE_HUMIDITY": hourly.get("relativehumidity_2m"),
            "WIND_SPEED": hourly.get("windspeed_10m"),
            "PRECIP_AMOUNT": hourly.get("precipitation"),
        })
        df["ZONE"] = zone
        sleep(1)

        return df

    # ============================================================================
    # CHARGEMENT DES DONNÉES (archive locale ou API Weather.gc.ca)
    # ============================================================================

    if USE_ARCHIVE:
        # --- Chargement depuis l'archive locale ---
        print(f"📂 Chargement depuis {WEATHER_ARCHIVE_PATH}...")
        weather_final = pd.read_parquet(WEATHER_ARCHIVE_PATH)

        # Normaliser le timestamp (supprimer timezone si présente)
        weather_final["datetime"] = pd.to_datetime(weather_final["datetime"], errors="coerce")
        if weather_final["datetime"].dt.tz is not None:
            weather_final["datetime"] = weather_final["datetime"].dt.tz_localize(None)

        # Filtrer sur la période configurée
        weather_final = weather_final[
            (weather_final["datetime"] >= pd.Timestamp(START_DATE))
            & (weather_final["datetime"] <= pd.Timestamp(END_DATE))
        ].copy()
        weather_final["ingestion_time"] = datetime.now()

        print(f"  ✓ {len(weather_final):,} observations")
        print(f"  Zones : {sorted(weather_final['zone'].unique())}")
        print(f"  Période : {weather_final['datetime'].min()} → {weather_final['datetime'].max()}")

    else:
        # --- Téléchargement via Weather.gc.ca, avec fallback Open-Meteo si DNS AWS bloque ---
        weather_by_zone = {}

        try:
            for zone, station in ZONE_STATIONS.items():
                df = download_zone_weather_gcca(zone, station["climate_id"])

                if not df.empty:
                    weather_by_zone[zone] = df
                    print(f"  {zone}: {len(df):,} observations")
        except RuntimeError as exc:
            if str(exc) != "WEATHER_GCCA_DNS":
                raise

            print("\n⚠️ Weather.gc.ca inaccessible depuis ce compute AWS (DNS).")
            print("   Bascule automatique vers Open-Meteo Archive.\n")
            weather_by_zone = {}

            for zone, coords in ZONE_COORDS.items():
                df = download_zone_weather_openmeteo(
                    zone,
                    coords["latitude"],
                    coords["longitude"],
                )

                if not df.empty:
                    weather_by_zone[zone] = df
                    print(f"  {zone}: {len(df):,} observations")

    if weather_by_zone:
        print(f"\n📊 Données chargées: {len(weather_by_zone)} zones")
    else:
        print("\n⚠️ Aucune donnée téléchargée")
        raise SystemExit("Arrêt: aucune donnée")

    # --- Normalisation ---
        weather_normalized = {}

        print("\n" + "="*80)
        print("NORMALISATION DES DONNÉES PAR ZONE")
        print("="*80)

        for zone, df_raw in weather_by_zone.items():
            print(f"\nTraitement de {zone}...")

            df = df_raw.copy()

            df["TIMESTAMP_UTC"] = pd.to_datetime(df["UTC_DATE"], utc=True, errors="coerce")
            df["TIMESTAMP_LOCAL"] = df["TIMESTAMP_UTC"].dt.tz_convert("America/Toronto")

            df["TEMPERATURE_C"] = pd.to_numeric(df["TEMP"], errors="coerce")
            df["DEW_POINT_C"] = pd.to_numeric(df["DEW_POINT_TEMP"], errors="coerce")
            df["RELATIVE_HUMIDITY_PCT"] = pd.to_numeric(df["RELATIVE_HUMIDITY"], errors="coerce")
            df["WIND_SPEED_KMH"] = pd.to_numeric(df["WIND_SPEED"], errors="coerce")
            df["PRECIPITATION_MM"] = pd.to_numeric(df["PRECIP_AMOUNT"], errors="coerce")

            df = df.dropna(subset=["TIMESTAMP_UTC"]).copy()
            duplicates_before = len(df)
            df = df.drop_duplicates(subset=["TIMESTAMP_UTC"], keep="first")
            duplicates_removed = duplicates_before - len(df)
            df = df.sort_values("TIMESTAMP_UTC").reset_index(drop=True)

            weather_normalized[zone] = df

            print(f"  ✓ {len(df):,} observations")
            if duplicates_removed > 0:
                print(f"  ⚠️ {duplicates_removed:,} doublons supprimés")
            print(f"  Période: {df['TIMESTAMP_UTC'].min()} à {df['TIMESTAMP_UTC'].max()}")
            print(f"  Température moyenne: {df['TEMPERATURE_C'].mean():.1f}°C")

    print("\n" + "="*80)
    total_obs = sum(len(df) for df in weather_normalized.values())
    print(f"✅ {len(weather_normalized)} zones normalisées - {total_obs:,} observations")

    # --- Consolidation ---
    weather_final = pd.concat(weather_normalized.values(), ignore_index=True)

    colonnes_finales = [
        "ZONE", "TIMESTAMP_UTC", "TIMESTAMP_LOCAL",
        "TEMPERATURE_C", "DEW_POINT_C", "RELATIVE_HUMIDITY_PCT",
        "WIND_SPEED_KMH", "PRECIPITATION_MM"
    ]
    weather_final = weather_final[colonnes_finales]
    weather_final["INGESTION_TIME"] = datetime.now()

    weather_final = weather_final.rename(columns={
        "ZONE": "zone",
        "TIMESTAMP_UTC": "datetime",
        "TIMESTAMP_LOCAL": "datetime_local",
        "TEMPERATURE_C": "temperature_c",
        "DEW_POINT_C": "dew_point_c",
        "RELATIVE_HUMIDITY_PCT": "relative_humidity_pct",
        "WIND_SPEED_KMH": "wind_speed_kmh",
        "PRECIPITATION_MM": "precipitation_mm",
        "INGESTION_TIME": "ingestion_time"
    })

    weather_final["datetime"] = weather_final["datetime"].dt.tz_localize(None)
    weather_final["datetime_local"] = weather_final["datetime_local"].dt.tz_localize(None)

    ontario_weather = (
        weather_final.groupby("datetime", as_index=False)
        .agg({
            "temperature_c": "mean",
            "dew_point_c": "mean",
            "relative_humidity_pct": "mean",
            "wind_speed_kmh": "mean",
            "precipitation_mm": "mean",
            "datetime_local": "first",
        })
    )
    ontario_weather["zone"] = "Ontario"
    ontario_weather["ingestion_time"] = datetime.now()
    weather_final = pd.concat([weather_final, ontario_weather], ignore_index=True)

    print(f"\n📊 DataFrame final: {len(weather_final):,} observations")
    print(f"Zones: {sorted(weather_final['zone'].unique())}")
    print(f"Période: {weather_final['datetime'].min()} à {weather_final['datetime'].max()}")

    # Conversion Spark et écriture
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    weather_spark = spark.createDataFrame(weather_final)

    print(f"\n✍️ Écriture dans {FULL_TABLE_NAME}...")

    weather_spark.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .option("delta.columnMapping.mode", "name") \
        .partitionBy("zone") \
        .saveAsTable(FULL_TABLE_NAME)

    print(f"✅ Table enregistrée: {FULL_TABLE_NAME}")

    # Vérification
    count = spark.table(FULL_TABLE_NAME).count()
    print(f"\n📊 Total rows: {count:,}")

    print("\n🔍 Statistiques par zone:")
    spark.sql(f"""
        SELECT 
            zone,
            COUNT(*) as observations,
            MIN(datetime) as first_obs,
            MAX(datetime) as last_obs,
            ROUND(AVG(temperature_c), 1) as avg_temp_c
        FROM {FULL_TABLE_NAME}
        GROUP BY zone
        ORDER BY zone
    """).show()

    print("\n✅ Ingestion météo terminée")
