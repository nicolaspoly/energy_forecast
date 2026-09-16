# Changelog — Nettoyage du 2026-08-29

Ce nettoyage était **structurel** (dédoublonnage, organisation, cohérence
config ↔ code). Il ne modifie pas la logique de modélisation elle-même
(features, hyperparamètres, stratégie de split), sauf les deux corrections de
bugs bloquants listées plus bas.

## 1. Fichiers supprimés (doublons / obsolètes)

- **`_backup_20260828/`** (dossier entier) — snapshot identique ou antérieur
  aux fichiers déjà présents ailleurs dans le projet (vérifié par hash MD5).
- **`energy_forecast/`** (sous-dossier imbriqué du même nom que le projet) —
  doublon exact de `_backup_20260828/`.
- **Scripts ad-hoc à la racine** : `01_ieso_data_ingestion.py`,
  `02_demand_forecasting.py`, `Import_meteo_zone.py`, `Predict_demand_zonal.py`,
  `import_ieso_data.py`, `holidays_ontario.py` (racine). Ce sont des notebooks
  d'exploration antérieurs à la structure `Pipeline/` actuelle (ils référencent
  des variables de session interactive comme `key_files_data`, non
  ré-exécutables tels quels). La version canonique de `holidays_ontario.py`
  est conservée dans `utils/`.
- **Fichiers vides** : `Pipeline/Train/gold/New File *.py` (x2),
  `Pipeline/Prod/New File *.py` (0 octet chacun).
- **`Pipeline/Train/bronze/01_ingest_ieso_demand_v2.py`** — fusionné dans le
  nouveau `pipeline/bronze/01_ingest_ieso_demand.py` (voir section 3).

## 2. Fichiers dédupliqués / réorganisés (données)

- `ieso_zonal_demand_2020_present.{csv,parquet}` existaient en double (racine
  + `data/archive/`, contenu identique). Une seule copie conservée sous
  `data/archive/`.
- `load_forecast_predictions.csv` (racine, export manuel non référencé par le
  code) → déplacé vers `data/exports/`.
- `data_ieso/ieso_t24_files_downloaded.csv` (index de fichiers IESO
  "Adequacy3", sans lien avec le pipeline demande/météo actuel, non référencé
  par aucun script) → déplacé vers `data/misc_unused/` avec un avertissement
  dans le README. À confirmer si encore utile, sinon supprimable.
- Les deux dashboards Lakeview → renommés sans espaces sous `dashboards/`.

## 3. Renumérotation et fusion du pipeline

Tous les scripts de pipeline sont renumérotés séquentiellement (01 → 10) sous
`pipeline/{bronze,silver,gold,modeling,inference,monitoring}/` :

| Ancien chemin | Nouveau chemin |
|---|---|
| `Pipeline/Train/bronze/01_ingest_ieso_demand.py` + `01_ingest_ieso_demand_v2.py` | `pipeline/bronze/01_ingest_ieso_demand.py` (fusion) |
| `Pipeline/Train/bronze/02_weather_gcca.py` | `pipeline/bronze/02_ingest_weather_historical.py` |
| `Pipeline/Train/silver/01_join_demand_weather.py` | `pipeline/silver/03_join_demand_weather.py` |
| `Pipeline/Train/gold/01_features_ml.py` | `pipeline/gold/04_build_features.py` |
| `Pipeline/Train/gold/02_feature_selection.py` | `pipeline/gold/05_feature_selection.py` |
| `Pipeline/modeling/06_train_lightgbm.py` | `pipeline/modeling/06_train_model_24h.py` |
| `Pipeline/modeling/07_train_long_terme.py` | `pipeline/modeling/07_train_model_7j.py` |
| `Pipeline/Prod/05_build_prediction_features.py` | `pipeline/inference/08_build_prediction_features.py` |
| `Pipeline/modeling/07_batch_prediction.py` | `pipeline/inference/09_batch_prediction.py` |
| `Pipeline/monitoring/08_model_evaluation.py` | `pipeline/monitoring/10_model_evaluation.py` |

**Fusion bronze (`01_ingest_ieso_demand.py`)** : les deux versions
coexistantes divergeaient sur deux points ; le fichier fusionné prend le
meilleur des deux :
- Ingestion **globale + zonale** (comme la v2), plutôt que zonale seule (v1).
- **Exclusion des colonnes agrégées** `Zone Total` / `Diff` / `Market Demand`
  de la liste des "zones" (comme la v1) — la v2 les aurait ingérées à tort
  comme si c'étaient de vraies zones géographiques.
- Gestion DST explicite `ambiguous=False` (comme la v1) plutôt que
  `ambiguous='infer'` (v2), qui peut lever une exception selon le motif
  horaire rencontré.
- Ajout d'un mode `BACKFILL` (bool) pour distinguer chargement initial complet
  (2020 → présent) et fenêtre incrémentale (usage cron normal) — les deux
  anciennes versions avaient chacune une seule des deux stratégies en dur.

## 4. Bugs corrigés (bloquants, hors périmètre "juste renommer")

1. **`config.yaml` désynchronisé du code réel** : les noms de table déclarés
   (`forecast_features_gold`, `weather_forecast_bronze`, etc.) ne
   correspondaient à aucune table réellement utilisée par les scripts
   (`ml_features_gold`, `weather_bronze`, etc.). Corrigé : `config.yaml`
   reflète maintenant exactement les noms utilisés dans `pipeline/*.py`.
2. **`10_model_evaluation.py`** (ex-`08_model_evaluation.py`) référençait
   `config['catalog']['tables']['gold']['demand_forecast']`, une clé qui
   n'existe pas → `KeyError` garanti à l'exécution. Corrigé pour utiliser
   `load_forecast`. Le script a aussi été réécrit pour joindre
   `load_forecast_gold` (qui ne contient QUE les prédictions, par zone) avec
   `load_zonal_bronze` (le réel), au lieu de supposer à tort une table unique
   contenant déjà `load_mw` et `predicted_load_mw` côte à côte.
3. **Modèles 24h et 7j partageaient la même expérience/registre MLflow**
   (`config['model']['mlflow']...`, une seule entrée). Corrigé en scindant
   `config.yaml -> models` en `horizon_24h` et `horizon_7j`, chacun avec son
   `experiment_name` et son `registry_model_name` propre.
4. **`registered_model_name` absent** des deux appels `mlflow.sklearn.log_model(...)`
   dans les scripts d'entraînement : les modèles étaient loggés dans leur run
   mais jamais réellement inscrits au Model Registry. Sans ça,
   `client.get_latest_versions(MODEL_NAME, stages=["Production"])` dans
   `09_batch_prediction.py` ne pouvait jamais réussir. Ajouté dans les deux
   scripts.
5. **`08_build_prediction_features.py`** (ex-`05_build_prediction_features.py`)
   ne chargeait pas `config.yaml` du tout et avait un nom d'expérience MLflow
   codé en dur, désynchronisé du reste. Corrigé pour charger la config et
   utiliser `models.horizon_24h.mlflow.experiment_name`.
6. Chemin de `exec(open(...))` dans `09_batch_prediction.py` mis à jour pour
   pointer vers le nouveau chemin de `08_build_prediction_features.py`.

## 5. Non traité volontairement (hors périmètre de ce nettoyage)

- **Le lien modèle 7j ↔ scoring batch** : `09_batch_prediction.py` ne sait
  scorer que de façon récursive avec un seul modèle. Le modèle 7j (multi-horizon
  direct) n'est appelé par aucun script de scoring aujourd'hui. Voir README.
  Nécessite une décision de conception, pas juste un renommage.
- Le contenu des scripts de feature engineering / entraînement (1700-2700
  lignes chacun) n'a pas été audité ligne à ligne — seule leur intégration
  (noms de table, config, MLflow) a été vérifiée et corrigée.
- Pertinence de `data/misc_unused/` et de `selected_features.yaml` (28
  features globales, potentiellement obsolète face à la table zonale
  actuelle) — signalé dans le README, à trancher avec toi.

---

# Changelog — Nettoyage du 2026-09-01

Suite du nettoyage du 2026-08-29 : traitement des 3 points laissés ouverts,
plus deux bugs supplémentaires découverts en creusant le premier.

## 1. Chemins codés en dur → configurables

Tous les notebooks lisaient `config.yaml` (et, pour certains, des fichiers
d'archive) via le chemin absolu
`/Workspace/Users/n.jouglet23@gmail.com/energy_forecast_clean/...` codé en
dur. Remplacé partout par une variable `PROJECT_ROOT`, résolue via
`os.environ.get("ENERGY_FORECAST_PROJECT_ROOT", "<même chemin par défaut>")` —
donc rien ne change en l'état, mais le projet devient déployable ailleurs sans
toucher au code, juste en définissant cette variable d'environnement sur le
cluster/job cible. Fichiers concernés : `01_ingest_ieso_demand.py`,
`02_ingest_weather_historical.py`, `03_join_demand_weather.py`,
`04_build_features.py`, `04_build_features_7j.py`, `05_feature_selection.py`,
`06_train_model_24h.py`, `07_train_model_7j.py`,
`08b_build_prediction_features_7j.py`, `09_batch_prediction.py`,
`09b_batch_prediction_7j_direct.py`, `10_model_evaluation.py`.

## 2. Le lien modèle 7j ↔ scoring batch — en fait déjà écrit, juste non câblé

Le README du 2026-08-29 affirmait qu'aucun script de scoring n'existait pour
le modèle 7j. C'était **inexact** : le zip contenait déjà
`pipeline/gold/04_build_features_7j.py`,
`pipeline/inference/08b_build_prediction_features_7j.py` et
`pipeline/inference/09b_batch_prediction_7j_direct.py` — un pipeline de
scoring direct complet et fonctionnel pour le modèle 7j, avec ses propres
tables (`ml_features_gold_7j`, `load_forecast_7j`, `load_shap_7j`). Ces
fichiers avaient simplement été oubliés par le premier nettoyage : absents du
README, de `docs/architecture.md` et de `config.yaml -> jobs`.

Traité :
- `config.yaml -> catalog.tables.gold` complété avec les tables du modèle 7j.
- `config.yaml -> jobs` : job `prediction` scindé en `prediction_24h` et
  `prediction_7j`, plus `04_build_features_7j` ajouté au job `processing`.
- README et `docs/architecture.md` réécrits pour documenter les deux
  pipelines de scoring en parallèle (24h récursif / 7j direct) au lieu de
  décrire le 7j comme "non branché".

## 3. `data/misc_unused/ieso_t24_files_downloaded.csv` — supprimé

Le nettoyage précédent l'avait déjà identifié comme "probablement obsolète,
sans lien avec le pipeline actuel". Confirmé et supprimé (le dossier
`data/misc_unused/` n'existe plus).

## 4. Bugs supplémentaires trouvés en câblant le point 2

En intégrant `04_build_features_7j.py` dans `config.yaml`, deux bugs latents
sont apparus :

1. **`ml_features_gold` (générique) n'existe pas.** `04_build_features.py`
   écrit en réalité dans `ml_features_gold_24h` (son propre docstring disait
   `ml_features_gold`, désynchronisé du code — corrigé aussi). Mais
   `05_feature_selection.py` et `06_train_model_24h.py` lisaient encore
   `ml_features_gold` : ça aurait levé une erreur "table not found" à la
   première exécution après le nettoyage du 29 août. Corrigé : les deux
   pointent maintenant vers `ml_features_gold_24h`.
2. **`10_model_evaluation.py` évaluait le modèle 7j sur des données
   dégénérées.** Le script chargeait une seule table (`ml_features_gold`,
   donc déjà cassé par le bug ci-dessus) et "fabriquait" les features
   d'horizon manquantes pour le modèle 7j à partir de
   `forecast_horizon_hours`, une colonne **toujours égale à 24** dans la
   table 24h — donc sans variance, ce qui aurait produit un MAPE 7j basé sur
   un seul horizon dupliqué plutôt qu'une vraie évaluation multi-horizon. Le
   script comparait aussi les prédictions à `demand_mw` (demande au moment de
   l'émission, une feature) au lieu de `target_demand_mw` (la vraie cible).
   Réécrit pour charger `ml_features_gold_24h` et `ml_features_gold_7j`
   séparément, comparer chaque modèle à `target_demand_mw`, et ajouter une
   vraie ventilation MAPE par horizon (H+24/H+48/H+168) pour le modèle 7j.

## 5. Toujours non traité (au-delà du périmètre convenu)

- Migration vers Databricks Repos/Asset Bundles pour un import réellement
  relatif (au lieu d'un chemin par défaut + variable d'environnement).
- Audit ligne à ligne du contenu des scripts de feature engineering /
  entraînement (1700-2700 lignes chacun) au-delà des points ci-dessus.
- `selected_features.yaml` n'a pas été régénéré (toujours 28 features
  globales) — à faire en relançant `05_feature_selection.py` si besoin.
