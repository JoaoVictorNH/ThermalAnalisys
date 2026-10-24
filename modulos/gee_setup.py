# ==============================================================================
# gee_setup.py — Google Earth Engine Authentication and Initialization
# ==============================================================================

import ee


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


