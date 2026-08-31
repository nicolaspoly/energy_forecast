"""
Configuration des zones météo Ontario pour pondération régionale

Ce module définit les zones météo de l'Ontario avec leurs coordonnées géographiques
et leurs poids basés sur la population et la consommation électrique.

Utilisé pour calculer les températures et conditions météo pondérées à l'échelle
provinciale plutôt que de se fier uniquement à Toronto.

Auteur: Energy Forecast Project
Date: 2026-08-28
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class WeatherZone:
    """Représente une zone météo avec ses coordonnées et son poids."""
    name: str
    latitude: float
    longitude: float
    weight: float  # Pondération basée sur population/consommation
    city: str = ""  # Ville principale de référence


# ==================================================================================
# ZONES MÉTÉO ONTARIO
# ==================================================================================
# Poids basés sur:
# - Population régionale (Statistique Canada 2021)
# - Consommation électrique historique IESO par zone
# ==================================================================================

WEATHER_ZONES: Dict[str, WeatherZone] = {
    "toronto": WeatherZone(
        name="toronto",
        latitude=43.70,
        longitude=-79.42,
        weight=0.25,
        city="Toronto"
    ),
    
    "ottawa": WeatherZone(
        name="ottawa",
        latitude=45.42,
        longitude=-75.69,
        weight=0.15,
        city="Ottawa"
    ),
    
    "west": WeatherZone(
        name="west",
        latitude=43.45,
        longitude=-80.49,
        weight=0.15,
        city="Kitchener-Waterloo"
    ),
    
    "southwest": WeatherZone(
        name="southwest",
        latitude=42.98,
        longitude=-81.23,
        weight=0.10,
        city="London"
    ),
    
    "east": WeatherZone(
        name="east",
        latitude=44.23,
        longitude=-76.49,
        weight=0.08,
        city="Kingston"
    ),
    
    "niagara": WeatherZone(
        name="niagara",
        latitude=43.09,
        longitude=-79.07,
        weight=0.10,
        city="Niagara Falls"
    ),
    
    "northeast": WeatherZone(
        name="northeast",
        latitude=46.49,
        longitude=-80.99,
        weight=0.07,
        city="Sudbury"
    ),
    
    "northwest": WeatherZone(
        name="northwest",
        latitude=48.38,
        longitude=-89.25,
        weight=0.05,
        city="Thunder Bay"
    ),
    
    "bruce": WeatherZone(
        name="bruce",
        latitude=44.33,
        longitude=-81.60,
        weight=0.03,
        city="Bruce Peninsula"
    ),
    
    "essa": WeatherZone(
        name="essa",
        latitude=44.27,
        longitude=-79.78,
        weight=0.02,
        city="Barrie"
    )
}


def get_zone(zone_name: str) -> WeatherZone:
    """
    Retourne les informations d'une zone météo.
    
    Args:
        zone_name: Nom de la zone (ex: 'toronto', 'ottawa')
    
    Returns:
        WeatherZone object
    
    Raises:
        KeyError: Si la zone n'existe pas
    """
    return WEATHER_ZONES[zone_name.lower()]


def get_all_zones() -> Dict[str, WeatherZone]:
    """Retourne toutes les zones météo configurées."""
    return WEATHER_ZONES


def validate_weights() -> bool:
    """
    Vérifie que la somme des poids = 1.0 (ou très proche).
    
    Returns:
        True si la somme des poids est valide
    """
    total = sum(zone.weight for zone in WEATHER_ZONES.values())
    return abs(total - 1.0) < 0.01


def get_weighted_coords() -> tuple[float, float]:
    """
    Calcule les coordonnées géographiques pondérées de l'Ontario.
    Utile pour obtenir une météo "moyenne" provinciale.
    
    Returns:
        (latitude, longitude) pondérées
    """
    weighted_lat = sum(z.latitude * z.weight for z in WEATHER_ZONES.values())
    weighted_lon = sum(z.longitude * z.weight for z in WEATHER_ZONES.values())
    return weighted_lat, weighted_lon


if __name__ == "__main__":
    # Tests rapides
    print("\n🗺️  ZONES MÉTÉO ONTARIO\n")
    print("="*70)
    
    for name, zone in WEATHER_ZONES.items():
        print(f"{zone.city:20} | Lat: {zone.latitude:6.2f} | Lon: {zone.longitude:7.2f} | Poids: {zone.weight:.2%}")
    
    print("\n" + "="*70)
    print(f"Somme des poids: {sum(z.weight for z in WEATHER_ZONES.values()):.4f}")
    print(f"Validation: {'✅ OK' if validate_weights() else '❌ ERREUR'}")
    
    lat, lon = get_weighted_coords()
    print(f"\nCentre pondéré Ontario: Lat {lat:.2f}, Lon {lon:.2f}")
