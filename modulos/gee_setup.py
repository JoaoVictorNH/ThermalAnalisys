# ==============================================================================
# gee_setup.py — Google Earth Engine Authentication and Initialization
# ==============================================================================

import ee
import math


def authenticate_gee(project_id="ee-joaovictornh01"):
    """
    Authenticates and initializes Google Earth Engine.

    Args:
        project_id (str): GEE project ID.
    """
    try:
        ee.Initialize(project=project_id)
        print("✓ Earth Engine already authenticated")
    except Exception:
        print("Authenticating Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("✓ Earth Engine authenticated successfully")


def mount_drive():
    """Mounts Google Drive in Colab."""
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted")


def get_utm_crs(lon, lat):
    """
    Determines the appropriate UTM CRS for a given coordinate.

    Args:
        lon (float): Longitude in decimal degrees.
        lat (float): Latitude in decimal degrees.

    Returns:
        str: EPSG string of the UTM CRS (e.g., 'EPSG:32724').
    """
    zone_number = int((lon + 180) / 6) + 1

    # Exceptions for Norway and Svalbard
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone_number = 32
    elif 72 <= lat < 84:
        if 0 <= lon < 9:
            zone_number = 31
        elif 9 <= lon < 21:
            zone_number = 33
        elif 21 <= lon < 33:
            zone_number = 35
        elif 33 <= lon < 42:
            zone_number = 37

    if lat >= 0:
        epsg_code = 32600 + zone_number  # Northern
    else:
        epsg_code = 32700 + zone_number  # Southern

    return f'EPSG:{epsg_code}'