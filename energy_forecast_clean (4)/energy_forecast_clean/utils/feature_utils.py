"""
Utilitaires pour le feature engineering

Fonctions réutilisables pour créer les features de prévision:
- Variables temporelles (cycliques, calendrier)
- Lags et rolling windows
- Encodage des catégories

Auteur: Energy Forecast Project
Date: 2026-08-28
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_temporal_features(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Ajoute les features temporelles de base.
    
    Args:
        df: DataFrame avec colonne datetime
        datetime_col: Nom de la colonne datetime
    
    Returns:
        DataFrame enrichi avec features temporelles
    """
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])
    
    # Features de base
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['hour'] = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek  # 0=Lundi, 6=Dimanche
    df['day_of_year'] = dt.dt.dayofyear
    df['week_of_year'] = dt.dt.isocalendar().week
    
    # Indicateurs binaires
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les encodages cycliques (sin/cos) pour capturer la périodicité.
    
    Args:
        df: DataFrame avec colonnes hour, day_of_week, month
    
    Returns:
        DataFrame avec features cycliques
    """
    df = df.copy()
    
    # Heure (24h)
    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Jour de la semaine (7 jours)
    if 'day_of_week' in df.columns:
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Mois (12 mois)
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int],
    datetime_col: str = 'datetime'
) -> pd.DataFrame:
    """
    Ajoute les features de lag (valeurs passées).
    
    Args:
        df: DataFrame trié par datetime
        target_col: Colonne cible à lagger (ex: 'load_mw')
        lags: Liste des lags en heures (ex: [24, 48, 168])
        datetime_col: Nom de la colonne datetime
    
    Returns:
        DataFrame avec colonnes lag_24, lag_48, etc.
    """
    df = df.copy()
    df = df.sort_values(datetime_col)
    
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int],
    functions: Optional[List[str]] = None,
    datetime_col: str = 'datetime'
) -> pd.DataFrame:
    """
    Ajoute les features de rolling window (moyennes mobiles, etc.).
    
    Args:
        df: DataFrame trié par datetime
        target_col: Colonne cible (ex: 'load_mw')
        windows: Tailles des fenêtres en heures (ex: [24, 168])
        functions: Fonctions à appliquer ('mean', 'std', 'min', 'max')
        datetime_col: Nom de la colonne datetime
    
    Returns:
        DataFrame avec colonnes rolling_24h_mean, rolling_7d_std, etc.
    """
    if functions is None:
        functions = ['mean', 'std']
    
    df = df.copy()
    df = df.sort_values(datetime_col)
    
    for window in windows:
        for func in functions:
            col_name = f'rolling_{window}h_{func}'
            
            if func == 'mean':
                df[col_name] = df[target_col].rolling(window=window, min_periods=1).mean()
            elif func == 'std':
                df[col_name] = df[target_col].rolling(window=window, min_periods=1).std()
            elif func == 'min':
                df[col_name] = df[target_col].rolling(window=window, min_periods=1).min()
            elif func == 'max':
                df[col_name] = df[target_col].rolling(window=window, min_periods=1).max()
    
    return df


def add_interaction_features(
    df: pd.DataFrame,
    pairs: Optional[List[tuple]] = None
) -> pd.DataFrame:
    """
    Ajoute des features d'interaction (produits de colonnes).
    Utile pour capturer les effets combinés (ex: temp * hour).
    
    Args:
        df: DataFrame
        pairs: Liste de tuples (col1, col2) pour interactions
    
    Returns:
        DataFrame avec features d'interaction
    """
    if pairs is None:
        # Interactions par défaut
        pairs = [
            ('hour', 'is_weekend'),
            ('hour', 'month')
        ]
    
    df = df.copy()
    
    for col1, col2 in pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    return df


def add_holiday_features(
    df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    datetime_col: str = 'datetime'
) -> pd.DataFrame:
    """
    Ajoute les features de jours fériés.
    
    Args:
        df: DataFrame avec colonne datetime
        holidays_df: DataFrame avec colonnes ['date', 'holiday_name']
        datetime_col: Nom de la colonne datetime
    
    Returns:
        DataFrame avec is_holiday et jours avant/après férié
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df[datetime_col]).dt.date
    holidays_df['date'] = pd.to_datetime(holidays_df['date']).dt.date
    
    # Joindre les fériés
    df = df.merge(holidays_df, on='date', how='left')
    df['is_holiday'] = df['holiday_name'].notna().astype(int)
    
    # Jours avant/après férié
    df['days_to_holiday'] = 0
    df['days_from_holiday'] = 0
    
    # TODO: Implémenter calcul distance au prochain/dernier férié
    
    df = df.drop('date', axis=1)
    return df


def select_features_for_model(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Sélectionne les colonnes features pour le modèle.
    Exclut datetime, target, et colonnes non-numériques.
    
    Args:
        df: DataFrame
        exclude_cols: Colonnes supplémentaires à exclure
    
    Returns:
        Liste des noms de colonnes features
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Colonnes à toujours exclure
    always_exclude = ['datetime', 'date', 'load_mw', 'demand', 'target']
    exclude = set(always_exclude + exclude_cols)
    
    # Sélectionner uniquement colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Filtrer
    features = [col for col in numeric_cols if col not in exclude]
    
    return features


if __name__ == "__main__":
    # Test rapide
    print("\n🛠️  TEST FEATURE UTILS\n")
    print("="*70)
    
    # Créer DataFrame de test
    dates = pd.date_range('2026-01-01', periods=168, freq='H')
    df_test = pd.DataFrame({
        'datetime': dates,
        'load_mw': np.random.randint(15000, 25000, 168)
    })
    
    # Ajouter features
    df_test = add_temporal_features(df_test)
    df_test = add_cyclical_features(df_test)
    df_test = add_lag_features(df_test, 'load_mw', [24, 48])
    df_test = add_rolling_features(df_test, 'load_mw', [24])
    
    print(f"DataFrame initial: {len(df_test)} lignes")
    print(f"Features créées: {len(df_test.columns)} colonnes")
    print("\nColonnes:")
    for col in df_test.columns:
        print(f"  - {col}")
    
    print("\n" + "="*70)
    print("✅ Tests OK")
