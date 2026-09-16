"""
Gestion des jours fériés et du calendrier pour l'Ontario, Canada

Ce module fournit des fonctions pour identifier:
- Les jours fériés provinciaux et fédéraux
- Les périodes de vacances scolaires
- Les événements spéciaux affectant la consommation électrique

Auteur: Energy Forecast Project
Date: 2026-08-27
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union, List


# ==================================================================================
# JOURS FÉRIÉS ONTARIO/CANADA
# ==================================================================================

def get_ontario_holidays(year: int) -> dict:
    """
    Retourne les jours fériés de l'Ontario pour une année donnée.
    
    Jours fériés inclus:
    - Fédéraux: Jour de l'An, Vendredi Saint, Fête de la Reine, Fête du Canada,
                Fête du Travail, Action de grâces, Jour du Souvenir, Noël, Boxing Day
    - Provinciaux: Jour de la famille (Ontario)
    
    Args:
        year: Année pour laquelle calculer les jours fériés
        
    Returns:
        Dictionnaire {nom_jour_férié: date}
    """
    holidays = {}
    
    # --- DATES FIXES ---
    holidays['new_year'] = pd.Timestamp(f'{year}-01-01')
    holidays['canada_day'] = pd.Timestamp(f'{year}-07-01')
    holidays['remembrance_day'] = pd.Timestamp(f'{year}-11-11')
    holidays['christmas'] = pd.Timestamp(f'{year}-12-25')
    holidays['boxing_day'] = pd.Timestamp(f'{year}-12-26')
    
    # --- DATES MOBILES (basées sur Pâques) ---
    easter = calculate_easter(year)
    holidays['good_friday'] = easter - timedelta(days=2)
    holidays['easter_monday'] = easter + timedelta(days=1)
    
    # --- DATES MOBILES (nième jour de la semaine) ---
    # Jour de la Famille (Ontario) - 3e lundi de février
    holidays['family_day'] = get_nth_weekday(year, 2, 0, 3)  # 3e lundi de février
    
    # Victoria Day - Lundi précédant le 25 mai
    may_24 = pd.Timestamp(f'{year}-05-24')
    holidays['victoria_day'] = may_24 - timedelta(days=(may_24.weekday() + 1) % 7)
    
    # Fête du Travail - 1er lundi de septembre
    holidays['labour_day'] = get_nth_weekday(year, 9, 0, 1)
    
    # Action de grâces - 2e lundi d'octobre
    holidays['thanksgiving'] = get_nth_weekday(year, 10, 0, 2)
    
    return holidays


def calculate_easter(year: int) -> pd.Timestamp:
    """
    Calcule la date de Pâques pour une année donnée (algorithme de Meeus).
    
    Args:
        year: Année
        
    Returns:
        Date de Pâques
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    
    return pd.Timestamp(f'{year}-{month:02d}-{day:02d}')


def get_nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """
    Retourne le nième jour de la semaine d'un mois.
    
    Args:
        year: Année
        month: Mois (1-12)
        weekday: Jour de la semaine (0=lundi, 6=dimanche)
        n: Occurrence (1 = premier, 2 = deuxième, etc.)
        
    Returns:
        Date correspondante
    """
    first_day = pd.Timestamp(f'{year}-{month:02d}-01')
    first_weekday = first_day.weekday()
    
    # Calculer le décalage pour atteindre le jour de la semaine désiré
    offset = (weekday - first_weekday) % 7
    
    # Ajouter (n-1) semaines
    target_date = first_day + timedelta(days=offset + (n - 1) * 7)
    
    return target_date


# ==================================================================================
# VACANCES SCOLAIRES ONTARIO
# ==================================================================================

def get_school_breaks(year: int) -> List[tuple]:
    """
    Retourne les périodes de vacances scolaires en Ontario.
    
    Périodes typiques:
    - Vacances d'hiver (2 semaines autour de Noël)
    - Relâche de mars (1 semaine mi-mars)
    - Vacances d'été (début juillet à fin août)
    
    Args:
        year: Année
        
    Returns:
        Liste de tuples (date_début, date_fin, nom_période)
    """
    breaks = []
    
    # Vacances d'hiver (du 23 décembre au 6 janvier)
    breaks.append((
        pd.Timestamp(f'{year-1}-12-23'),
        pd.Timestamp(f'{year}-01-06'),
        'winter_break'
    ))
    
    # Relâche de mars (2e semaine complète de mars - typiquement)
    march_break_start = get_nth_weekday(year, 3, 0, 2)  # 2e lundi de mars
    march_break_end = march_break_start + timedelta(days=6)
    breaks.append((march_break_start, march_break_end, 'march_break'))
    
    # Vacances d'été (1er juillet au 31 août)
    breaks.append((
        pd.Timestamp(f'{year}-07-01'),
        pd.Timestamp(f'{year}-08-31'),
        'summer_break'
    ))
    
    # Vacances d'hiver de l'année suivante (pour couvrir décembre de l'année courante)
    breaks.append((
        pd.Timestamp(f'{year}-12-23'),
        pd.Timestamp(f'{year+1}-01-06'),
        'winter_break'
    ))
    
    return breaks


# ==================================================================================
# ÉVÉNEMENTS SPÉCIAUX
# ==================================================================================

def get_special_events(year: int) -> dict:
    """
    Retourne les événements spéciaux pouvant affecter la demande électrique.
    
    Args:
        year: Année
        
    Returns:
        Dictionnaire {nom_événement: date}
    """
    events = {}
    
    # Super Bowl - Premier dimanche de février (approximatif)
    events['super_bowl'] = get_nth_weekday(year, 2, 6, 1)  # 1er dimanche de février
    
    # Note: D'autres événements peuvent être ajoutés selon les besoins
    # (élections, événements sportifs majeurs, etc.)
    
    return events


# ==================================================================================
# FONCTIONS D'AJOUT DE FEATURES
# ==================================================================================

def add_calendar_features(df: pd.DataFrame, date_col: str = 'datetime') -> pd.DataFrame:
    """
    Ajoute des features de calendrier à un DataFrame.
    
    Features ajoutées:
    - is_holiday: 1 si jour férié, 0 sinon
    - holiday_name: Nom du jour férié (si applicable)
    - is_school_break: 1 si période de vacances scolaires, 0 sinon
    - school_break_name: Nom de la période de vacances (si applicable)
    - days_to_next_holiday: Nombre de jours jusqu'au prochain jour férié
    - days_since_last_holiday: Nombre de jours depuis le dernier jour férié
    - is_holiday_eve: 1 si veille de jour férié, 0 sinon
    - is_holiday_aftermath: 1 si lendemain de jour férié, 0 sinon
    
    Args:
        df: DataFrame contenant une colonne de dates
        date_col: Nom de la colonne contenant les dates
        
    Returns:
        DataFrame avec les nouvelles features
    """
    df = df.copy()
    
    # Assurer que la colonne date est en format datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Initialiser les colonnes
    df['is_holiday'] = 0
    df['holiday_name'] = None
    df['is_school_break'] = 0
    df['school_break_name'] = None
    df['is_special_event'] = 0
    df['event_name'] = None
    
    # Extraire les années uniques
    years = df[date_col].dt.year.unique()
    
    # Créer un dictionnaire de tous les jours fériés sur la période
    all_holidays = {}
    for year in years:
        year_holidays = get_ontario_holidays(year)
        for name, date in year_holidays.items():
            all_holidays[date.date()] = name
    
    # Marquer les jours fériés
    df['date_only'] = df[date_col].dt.date
    df['is_holiday'] = df['date_only'].map(lambda x: 1 if x in all_holidays else 0)
    df['holiday_name'] = df['date_only'].map(lambda x: all_holidays.get(x, None))
    
    # Marquer les périodes de vacances scolaires
    for year in years:
        school_breaks = get_school_breaks(year)
        for start, end, break_name in school_breaks:
            mask = (df[date_col].dt.date >= start.date()) & (df[date_col].dt.date <= end.date())
            df.loc[mask, 'is_school_break'] = 1
            df.loc[mask, 'school_break_name'] = break_name
    
    # Marquer les événements spéciaux
    for year in years:
        events = get_special_events(year)
        for name, date in events.items():
            mask = df[date_col].dt.date == date.date()
            df.loc[mask, 'is_special_event'] = 1
            df.loc[mask, 'event_name'] = name
    
    # Veille et lendemain de jour férié
    df['is_holiday_eve'] = df['is_holiday'].shift(-1).fillna(0).astype(int)
    df['is_holiday_aftermath'] = df['is_holiday'].shift(1).fillna(0).astype(int)
    
    # Distance au prochain/précédent jour férié
    holiday_dates = sorted(all_holidays.keys())
    
    def days_to_next_holiday(date):
        date_only = date.date() if isinstance(date, pd.Timestamp) else date
        future_holidays = [h for h in holiday_dates if h > date_only]
        if future_holidays:
            return (future_holidays[0] - date_only).days
        return 365  # Valeur par défaut si aucun jour férié à venir
    
    def days_since_last_holiday(date):
        date_only = date.date() if isinstance(date, pd.Timestamp) else date
        past_holidays = [h for h in holiday_dates if h < date_only]
        if past_holidays:
            return (date_only - past_holidays[-1]).days
        return 365  # Valeur par défaut si aucun jour férié passé
    
    df['days_to_next_holiday'] = df[date_col].apply(days_to_next_holiday)
    df['days_since_last_holiday'] = df[date_col].apply(days_since_last_holiday)
    
    # Feature combinée: proximité d'un jour férié (dans les 3 jours avant/après)
    df['near_holiday'] = ((df['days_to_next_holiday'] <= 3) | (df['days_since_last_holiday'] <= 3)).astype(int)
    
    # Nettoyer les colonnes temporaires
    df = df.drop(columns=['date_only'])
    
    return df


def print_holiday_summary(year: int):
    """
    Affiche un résumé des jours fériés et vacances pour une année.
    
    Args:
        year: Année à afficher
    """
    print(f"\n{'='*70}")
    print(f"📅 CALENDRIER DES JOURS FÉRIÉS ET VACANCES - ONTARIO {year}")
    print(f"{'='*70}\n")
    
    # Jours fériés
    holidays = get_ontario_holidays(year)
    print("🎉 JOURS FÉRIÉS:\n")
    for name, date in sorted(holidays.items(), key=lambda x: x[1]):
        day_name = date.day_name()
        print(f"  • {date.strftime('%Y-%m-%d')} ({day_name:<10s}): {name.replace('_', ' ').title()}")
    
    # Vacances scolaires
    breaks = get_school_breaks(year)
    print("\n📚 VACANCES SCOLAIRES:\n")
    for start, end, name in breaks:
        if start.year == year:  # Éviter les doublons pour les vacances d'hiver
            duration = (end - start).days + 1
            print(f"  • {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} ({duration} jours): {name.replace('_', ' ').title()}")
    
    # Événements spéciaux
    events = get_special_events(year)
    print("\n⚡ ÉVÉNEMENTS SPÉCIAUX:\n")
    for name, date in sorted(events.items(), key=lambda x: x[1]):
        day_name = date.day_name()
        print(f"  • {date.strftime('%Y-%m-%d')} ({day_name:<10s}): {name.replace('_', ' ').title()}")
    
    print(f"\n{'='*70}\n")


# ==================================================================================
# TESTS ET DÉMONSTRATION
# ==================================================================================

if __name__ == "__main__":
    # Démonstration du module
    print("\n🔍 TEST DU MODULE holidays_ontario.py\n")
    
    # Afficher le calendrier 2024 et 2025
    for year in [2024, 2025, 2026]:
        print_holiday_summary(year)
    
    # Test sur un petit DataFrame
    print("\n📊 TEST D'AJOUT DE FEATURES:\n")
    test_df = pd.DataFrame({
        'datetime': pd.date_range('2024-12-20', '2025-01-10', freq='D')
    })
    
    test_df = add_calendar_features(test_df)
    
    print("Exemple de features ajoutées:")
    print(test_df[['datetime', 'is_holiday', 'holiday_name', 'is_school_break', 
                   'is_holiday_eve', 'days_to_next_holiday']].head(15))
    
    print("\n✅ Tests terminés!")
