


import os
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

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

# Tables sources
DEMAND_TABLE = f"{CATALOG}.{SCHEMA}.load_zonal_bronze"
WEATHER_TABLE = f"{CATALOG}.{SCHEMA}.weather_bronze"




# Table destination
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.demand_weather_silver"

print(f"🔗 JOIN DEMAND + WEATHER → {OUTPUT_TABLE}")
print(f"Sources:")
print(f"  - {DEMAND_TABLE}")
print(f"  - {WEATHER_TABLE}")

# ============================================================================
# LOAD DATA FROM BRONZE
# ============================================================================

spark = SparkSession.builder.getOrCreate()

print("\n📥 Chargement des données Bronze...")

df_demand = spark.table(DEMAND_TABLE)
df_weather = spark.table(WEATHER_TABLE)

# DEBUG: Afficher les zones AVANT filtrage
print("\n🔍 Zones dans demand AVANT filtrage:")
df_demand.select("zone").distinct().orderBy("zone").show(50, False)

# Exclure uniquement les zones calculées: Diff, Zone Total, Total, Market
# Regex case-insensitive: zone.?total matche "Zone Total", "zone_total", "ZONE TOTAL", etc.
df_demand = df_demand.filter(
    ~F.col("zone").rlike("(?i)^(diff|total|zone.?total|market)$")
)
df_weather = df_weather.filter(
    ~F.col("zone").rlike("(?i)^(diff|total|zone.?total|market)$")
)

# DEBUG: Afficher les zones APRÈS filtrage
print("\n✅ Zones dans demand APRÈS filtrage:")
df_demand.select("zone").distinct().orderBy("zone").show(50, False)

print(f"  ✅ Demand: {df_demand.count():,} lignes")
print(f"  ✅ Weather: {df_weather.count():,} lignes")

# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

print("\n🔍 Contrôles qualité...")

# Vérifier les zones
demand_zones = df_demand.select("zone").distinct().collect()
weather_zones = df_weather.select("zone").distinct().collect()

demand_zones_set = {row.zone for row in demand_zones}
weather_zones_set = {row.zone for row in weather_zones}

print(f"  Zones demand: {sorted(demand_zones_set)}")
print(f"  Zones météo: {sorted(weather_zones_set)}")

common_zones = demand_zones_set.intersection(weather_zones_set)
print(f"  ✅ Zones communes: {sorted(common_zones)}")

if len(common_zones) == 0:
    raise ValueError("❌ Aucune zone commune entre demand et météo !")

# Vérifier les périodes
demand_period = df_demand.agg(
    F.min("datetime").alias("min_date"),
    F.max("datetime").alias("max_date")
).collect()[0]

weather_period = df_weather.agg(
    F.min("datetime").alias("min_date"),
    F.max("datetime").alias("max_date")
).collect()[0]

print(f"\n  Période demand: {demand_period.min_date} → {demand_period.max_date}")
print(f"  Période météo: {weather_period.min_date} → {weather_period.max_date}")

# ============================================================================
# JOIN DEMAND + WEATHER
# ============================================================================

print("\n🔗 Jointure demand + météo sur (datetime, zone)...")

# Calculer les features dérivées météo en Silver (pas dans Bronze)
print("  🔧 Calcul HDD18/CDD18 depuis temperature_c...")
df_weather_transformed = df_weather \
    .withColumn("hdd18", F.greatest(F.lit(18) - F.col("temperature_c"), F.lit(0))) \
    .withColumn("cdd18", F.greatest(F.col("temperature_c") - F.lit(18), F.lit(0)))

print("  ✅ HDD18 et CDD18 calculés")

# Renommer les colonnes météo pour éviter les conflits
weather_cols_renamed = df_weather_transformed.select(
    F.col("zone").alias("weather_zone"),
    F.col("datetime").alias("weather_datetime"),
    "datetime_local",
    "temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
    "hdd18",
    "cdd18"
)

# Join
df_joined = df_demand.join(
    weather_cols_renamed,
    (df_demand.datetime == weather_cols_renamed.weather_datetime) & 
    (df_demand.zone == weather_cols_renamed.weather_zone),
    "left"
)

# Supprimer les colonnes temporaires de join
df_joined = df_joined.drop("weather_zone", "weather_datetime")

print(f"  ✅ Jointure effectuée: {df_joined.count():,} lignes")

# Vérifier les nulls dans les données météo (après join)
null_weather = df_joined.filter(F.col("temperature_c").isNull()).count()
if null_weather > 0:
    print(f"  ⚠️ {null_weather:,} lignes sans données météo (zones ou périodes manquantes)")
else:
    print(f"  ✅ Toutes les lignes ont des données météo")

# ============================================================================
# ADD DERIVED FEATURES
# ============================================================================

print("\n🔧 Ajout de features dérivées...")

# Features temporelles de base (depuis datetime)
# Note: day_of_week → 0=Monday, 6=Sunday
df_silver = df_joined.withColumn("year", F.year("datetime")) \
    .withColumn("month", F.month("datetime")) \
    .withColumn("day", F.dayofmonth("datetime")) \
    .withColumn("hour", F.hour("datetime")) \
    .withColumn("day_of_week", F.dayofweek("datetime") - 1) \
    .withColumn("is_weekend", F.when(F.col("day_of_week") >= 5, 1).otherwise(0))

# Calculer le load factor (charge relative)
# Normalisation par zone pour avoir une charge relative
window_zone = Window.partitionBy("zone")
df_silver = df_silver.withColumn(
    "demand_normalized",
    (F.col("demand_mw") - F.avg("demand_mw").over(window_zone)) / F.stddev("demand_mw").over(window_zone)
)

print(f"  ✅ Features dérivées ajoutées")

# ============================================================================
# WRITE TO SILVER TABLE
# ============================================================================

print(f"\n✍️ Écriture dans {OUTPUT_TABLE}...")

# Ordre des colonnes final
final_cols = [
    # Keys
    "zone", "datetime", "datetime_local",
    # Temporal features
    "year", "month", "day", "hour", "day_of_week", "is_weekend",
    # Demand
    "demand_mw", "demand_normalized",
    # Weather
    "temperature_c", "dew_point_c", "relative_humidity_pct",
    "wind_speed_kmh", "precipitation_mm", "hdd18", "cdd18",
    # Metadata
    "ingestion_time"
]

df_silver_final = df_silver.select(*final_cols)

# Write
df_silver_final.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.columnMapping.mode", "name") \
    .partitionBy("year", "month", "zone") \
    .saveAsTable(OUTPUT_TABLE)

print(f"✅ Table Silver créée: {OUTPUT_TABLE}")

# ============================================================================
# VALIDATION & STATS
# ============================================================================

print("\n📊 Validation finale...")

df_check = spark.table(OUTPUT_TABLE)
count = df_check.count()

stats = df_check.agg(
    F.countDistinct("zone").alias("zones"),
    F.min("datetime").alias("min_date"),
    F.max("datetime").alias("max_date"),
    F.avg("demand_mw").alias("avg_demand"),
    F.avg("temperature_c").alias("avg_temp")
).collect()[0]

print(f"  Total lignes: {count:,}")
print(f"  Zones: {stats.zones}")
print(f"  Période: {stats.min_date} → {stats.max_date}")
print(f"  Demande moyenne: {stats.avg_demand:,.1f} MW")
print(f"  Température moyenne: {stats.avg_temp:.1f}°C")

print("\n🔍 Aperçu par zone:")
spark.sql(f"""
    SELECT 
        zone,
        COUNT(*) as observations,
        ROUND(AVG(demand_mw), 1) as avg_demand_mw,
        ROUND(AVG(temperature_c), 1) as avg_temp_c,
        MIN(datetime) as first_obs,
        MAX(datetime) as last_obs
    FROM {OUTPUT_TABLE}
    GROUP BY zone
    ORDER BY zone
""").show(20, False)

print("\n✅ Silver Layer - Join Demand + Weather terminé !")
