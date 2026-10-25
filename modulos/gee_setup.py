# ==============================================================================
# gee_setup.py — Google Earth Engine Authentication and Initialization
# ==============================================================================

import ee


def authenticate_gee(project_id=None):
    """
    Authenticates and initializes Google Earth Engine.

    Args:
        project_id (str): ID of YOUR Earth Engine project (e.g. 'ee-yourname'),
            shown at the top left of code.earthengine.google.com.
    """
    if not project_id or not str(project_id).strip():
        raise ValueError("Enter your Earth Engine project ID (cell 1.3), e.g. 'ee-yourname'. "
                         "Sign up at https://earthengine.google.com/signup/ if you do not have one.")
    project_id = str(project_id).strip()
    try:
        ee.Initialize(project=project_id)
        print("✓ Earth Engine already authenticated")
    except Exception:
        print("Authenticating Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("✓ Earth Engine authenticated successfully")


