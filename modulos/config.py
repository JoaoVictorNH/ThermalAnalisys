# ==============================================================================
# config.py — Global Settings and Constants
# ==============================================================================
# Centralizes color palettes, scales, CRS, and default parameters used across
# the project. Importing this module ensures consistency between modules.
# ==============================================================================

# ═══════════════════════════════════════════════════════════════════════════
# COLOR PALETTES
# ═══════════════════════════════════════════════════════════════════════════

LST_PALETTE = [
    '040274', '0502a3', '0502ce', '0502e6', '0602ff', '235cb1', '307ef3',
    '269db1', '30c8e2', '32d3ef', '3ae237', '86e26f', 'b5e22e', 'd6e21f',
    'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d', 'ff0000',
    'de0101', 'c21301', 'a71001', '911003'
]

UHI_PALETTE = ['2166ac', '67a9cf', 'd1e5f0', 'f7f7f7', 'fddbc7', 'ef8a62', 'b2182b']


NDVI_PALETTE = ['red', 'yellow', 'green']

CHANGE_PALETTE = ['blue', 'white', 'red']

# ═══════════════════════════════════════════════════════════════════════════
# SCALES AND PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SCALE = 30           # Default Landsat scale (meters)
SENTINEL2_SCALE = 10         # Sentinel-2 scale (meters)
DEFAULT_CRS = 'EPSG:4326'   # WGS84

# ═══════════════════════════════════════════════════════════════════════════
# CHANGE DETECTION MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

CD_TILE_SIZE = 128            # Tile size for inference
CD_TOPOLOGY = [64, 128, 256, 512]  # U-Net topology
CD_S1_BANDS = [0, 1]         # Sentinel-1 bands (VV, VH)
CD_S2_BANDS = [2, 1, 0, 3]   # Sentinel-2 bands (B4, B3, B2, B8)

# ═══════════════════════════════════════════════════════════════════════════
# SATELLITE BANDS
# ═══════════════════════════════════════════════════════════════════════════

LANDSAT_OPTICAL_BANDS = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
LANDSAT_THERMAL_BANDS = ['ST_B10']
LANDSAT_QA_BAND = 'QA_PIXEL'

SENTINEL2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
SENTINEL2_RGB_BANDS = ['B4', 'B3', 'B2']

# ═══════════════════════════════════════════════════════════════════════════
# UHI THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

UHI_THRESHOLDS = {
    'very_cold': -1.0,
    'cold': -0.5,
    'comfortable': 0.5,
    'warm': 1.0,
    'hot': float('inf')
}

# ═══════════════════════════════════════════════════════════════════════════
# UHI INTENSITY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

UHI_INTENSITY_LABELS = {
    1: 'LTZ — Low Temperature (< -1σ)',
    2: 'SLTZ — Sub-Low Temperature (-1σ to -0.5σ)',
    3: 'MTZ — Medium Temperature (-0.5σ to 0.5σ)',
    4: 'SHTZ — Sub-High Temperature (0.5σ to 1σ)',
    5: 'HTZ — High Temperature (> 1σ)'
}

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

UHI_INTENSITY_COLORS = {
    1: '#2166ac',
    2: '#67a9cf',
    3: '#f7f7f7',
    4: '#ef8a62',
    5: '#b2182b'
}