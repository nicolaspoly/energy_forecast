"""
08b - Construction des features de prédiction pour le modèle 7 jours.

Wrapper mince autour de 08_build_prediction_features.py : surdéfinit
MODEL_HORIZON_KEY='horizon_7j' puis exécute le script original via exec().

Cela garantit que MODEL_FEATURES, MODEL_ZONE_CATEGORIES et
MLFLOW_EXPERIMENT proviennent du modèle 7j (horizon_7j) et non du
modèle 24h (horizon_24h).
"""

MODEL_HORIZON_KEY = "horizon_7j"

# Le modèle 7j utilise des lags et rolling windows supplémentaires
# par rapport au modèle 24h.
LAG_HOURS = [1, 2, 3, 6, 24, 48, 72, 144, 168, 336]
ROLLING_WINDOWS = [3, 6, 12, 24, 48, 72, 168]

exec(
    open(
        '/Workspace/Users/n.jouglet23@gmail.com/'
        'energy_forecast_clean/pipeline/inference/'
        '08_build_prediction_features.py'
    ).read()
)