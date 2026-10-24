# ==============================================================================
# spectral_analysis.py — Spectral Indices and Threshold-Based Change Detection
# ==============================================================================
# Functions for computing spectral indices (Landsat and Sentinel-2),
# visualization, correlation, and threshold-based change detection.
# ==============================================================================

import ee
import geemap
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Optional imports for dynamic indices ---
try:
    import spyndex
    SPYNDEX_AVAILABLE = True
except ImportError:
    SPYNDEX_AVAILABLE = False
    print("⚠️  spyndex not installed. Run: pip install spyndex")

# --- Cross-module imports ---
from modulos.lst_analysis import calculate_uhi


# ==============================================================================
# DYNAMIC SPECTRAL INDEX FUNCTIONS (via spyndex)
# ==============================================================================

# Common bands available in Landsat 8/9 AND Sentinel-2
_COMMON_BANDS = {'B', 'G', 'R', 'N', 'S1', 'S2'}

# Band mapping per sensor for spyndex
_BAND_MAPPING = {
    'landsat': {
        'B': 'SR_B2', 'G': 'SR_B3', 'R': 'SR_B4',
        'N': 'SR_B5', 'S1': 'SR_B6', 'S2': 'SR_B7',
        'T1': 'ST_B10',
    },
    'sentinel2': {
        'B': 'B2', 'G': 'B3', 'R': 'B4',
        'N': 'B8', 'S1': 'B11', 'S2': 'B12',
    }
}

# Common constant parameters in index formulas
_DEFAULT_PARAMS = {
    'L': 0.5,      # SAVI soil correction factor
    'g': 2.5,      # EVI gain
    'C1': 6.0,     # EVI C1
    'C2': 7.5,     # EVI C2
    'gamma': 1.0,  # ARVI gamma
    'alpha': 0.1,  # generic alpha
    'sla': 1.0,    # soil line a
    'slb': 0.0,    # soil line b
    'nexp': 2.0,   # GDVI exponent
    'cexp': 1.16,  # OCVI exponent
    'omega': 2.0,  # generic omega
    'k': 0.0,      # generic k
    'fdelta': 0.581, # f delta for FCVI
    'lambdaN': 0.8545, # NIR wavelength
    'lambdaR': 0.6545, # Red wavelength
    'lambdaG': 0.555, # Green wavelength
}

# Mapping of generic band names → names by satellite
# Allows raw bands (NIR, SWIR1, etc.) to be used
# as predictors in MLR, correlation, and McHarg.
GENERIC_BAND_NAMES = {
    'blue':  {'landsat': 'SR_B2', 'sentinel2': 'B2'},
    'green': {'landsat': 'SR_B3', 'sentinel2': 'B3'},
    'red':   {'landsat': 'SR_B4', 'sentinel2': 'B4'},
    'nir':   {'landsat': 'SR_B5', 'sentinel2': 'B8'},
    'swir1': {'landsat': 'SR_B6', 'sentinel2': 'B11'},
    'swir2': {'landsat': 'SR_B7', 'sentinel2': 'B12'},
}

def resolve_predictors_from_image(image, predictor_names, sensor='landsat', verbose=True):
    """
    Separates generic band names from spectral indices and returns both.

    For band names (NIR, SWIR1, etc.), extracts directly from the image.
    For spectral indices, tries hardcoded first, then eemont/spyndex.

    Args:
        image (ee.Image): Satellite image (Landsat or Sentinel-2).
        predictor_names (list[str]): List of names (indices + bands).
        sensor (str): 'landsat' or 'sentinel2'.
        verbose (bool): If True, prints messages.

    Returns:
        dict: {name_lower: ee.Image} for each resolved predictor.
    """
    resolved = {}

    # Separate bands from indices
    bands_to_extract = []
    indices_to_calc = []
    for name in predictor_names:
        if name.lower() in GENERIC_BAND_NAMES:
            bands_to_extract.append(name.lower())
        else:
            indices_to_calc.append(name)

    # Extract generic bands
    if bands_to_extract:
        if verbose:
            print(f"   🔗 Generic bands detected: {[b.upper() for b in bands_to_extract]}")
        for bname in bands_to_extract:
            sat_band = GENERIC_BAND_NAMES[bname][sensor]
            try:
                resolved[bname.upper()] = image.select(sat_band).rename(bname.upper())
                if verbose:
                    print(f"     ✓ {bname.upper()} → {sat_band}")
            except Exception as e:
                if verbose:
                    print(f"     ❌ {bname.upper()} → error: {e}")

    # Calculate spectral indices
    if indices_to_calc:
        if sensor == 'landsat':
            hardcoded = calculate_spectral_indices_landsat(image)
        else:
            hardcoded = calculate_spectral_indices_sentinel2(image)

        # Separate hardcoded from those needing dynamic fallback
        missing = []
        for idx in indices_to_calc:
            idx_lower = idx.lower()
            if idx_lower in hardcoded:
                resolved[idx_lower] = hardcoded[idx_lower]
            else:
                missing.append(idx)

        # Batch dynamic fallback (single call for all)
        if missing:
            try:
                dynamic = calculate_spectral_indices_dynamic(
                    image, missing, sensor=sensor, verbose=verbose
                )
                resolved.update(dynamic)
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Failed to compute dynamic indices: {e}")

    return resolved


def list_available_indices(platforms=None, application_domain=None, bands_available=None, verbose=True):
    """
    Lists spectral indices from the Awesome Spectral Indices catalog filtered
    by platform, domain, and available bands.

    Args:
        platforms (list, optional): Required platforms. E.g.: ['Sentinel-2', 'Landsat-OLI']
            If None, filters for Landsat-OLI + Sentinel-2 compatibility.
        application_domain (str or list, optional): Application domain(s).
            E.g.: 'vegetation', 'water', 'urban', 'soil', 'burn', 'snow', 'radar'
            If None, returns all domains.
        bands_available (set, optional): Set of available bands (spyndex notation).
            If None, uses _COMMON_BANDS (B, G, R, N, S1, S2).
        verbose (bool): If True, prints the catalog to the terminal.

    Returns:
        pd.DataFrame: Table with short_name, long_name, domain, formula, reference
    """
    if not SPYNDEX_AVAILABLE:
        print("❌ spyndex is not installed. Run: pip install spyndex")
        return _list_hardcoded_indices()

    if platforms is None:
        platforms = ['Sentinel-2', 'Landsat-OLI']

    if bands_available is None:
        bands_available = _COMMON_BANDS

    indices_data = []

    for idx_name, idx_obj in spyndex.indices.items():
        try:
            idx_platforms = getattr(idx_obj, 'platforms', [])
            idx_domain = getattr(idx_obj, 'application_domain', '')
            idx_bands = getattr(idx_obj, 'bands', [])
            idx_formula = getattr(idx_obj, 'formula', '')
            idx_long = getattr(idx_obj, 'long_name', '')

            # Platform filter: index must support ALL required platforms
            if platforms:
                if not all(p in idx_platforms for p in platforms):
                    continue

            # Domain filter
            if application_domain:
                if isinstance(application_domain, str):
                    if idx_domain != application_domain:
                        continue
                elif isinstance(application_domain, list):
                    if idx_domain not in application_domain:
                        continue

            # Band filter: index must use only available bands
            required_bands = set(idx_bands) - set(_DEFAULT_PARAMS.keys())
            if not required_bands.issubset(bands_available):
                continue

            idx_reference = getattr(idx_obj, 'reference', '')

            indices_data.append({
                'short_name': idx_name,
                'long_name': idx_long,
                'domain': idx_domain,
                'formula': idx_formula,
                'reference': idx_reference,
            })
        except Exception:
            continue

    df = pd.DataFrame(indices_data)
    if not df.empty:
        df = df.sort_values(['domain', 'short_name']).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*80}")
        print(f"AVAILABLE SPECTRAL INDEX CATALOG")
        print(f"{'='*80}")
        print(f"  Total compatible indices (Landsat + Sentinel-2): {len(df)}")
        if not df.empty:
            domain_counts = df['domain'].value_counts()
            for domain, count in domain_counts.items():
                print(f"    • {domain}: {count} indices")
        print(f"{'='*80}\n")

    return df


def _list_hardcoded_indices():
    """Fallback when spyndex is not available."""
    data = [
        ('NDVI', 'Normalized Difference Vegetation Index', 'vegetation', '(N-R)/(N+R)', 'Rouse et al. (1974)'),
        ('EVI', 'Enhanced Vegetation Index', 'vegetation', '2.5*(N-R)/(N+6*R-7.5*B+1)', 'Huete et al. (2002)'),
        ('SAVI', 'Soil Adjusted Vegetation Index', 'vegetation', '((N-R)/(N+R+L))*(1+L)', 'Huete (1988)'),
        ('NDWI', 'Normalized Difference Water Index', 'water', '(G-N)/(G+N)', 'McFeeters (1996)'),
        ('MNDWI', 'Modified NDWI', 'water', '(G-S1)/(G+S1)', 'Xu (2006)'),
        ('NDBI', 'Normalized Difference Built-up Index', 'urban', '(S1-N)/(S1+N)', 'Zha et al. (2003)'),
        ('UI', 'Urban Index', 'urban', '(S1+R-N)/(S1+R+N)', 'Kawamura et al. (1996)'),
        ('BSI', 'Bare Soil Index', 'soil', '((S1+R)-(N+B))/((S1+R)+(N+B))', 'Rikimaru et al. (2002)'),
        ('NDTI', 'Normalized Difference Tillage Index', 'soil', '(S1-S2)/(S1+S2)', 'van Deventer et al. (1997)'),
        ('DBSI', 'Dry Bare Soil Index', 'soil', '((S1-G)/(S1+G))-NDVI', 'Rasul et al. (2018)'),
        ('NBAI', 'Normalized Built-up Area Index', 'urban', '(S2-S1/G)/(S2+S1/G)', 'Waqar et al. (2012)'),
        ('NBI', 'New Built-up Index', 'urban', '(R*S2)/N', 'Jieli et al. (2010)'),
        # --- Surface Albedo ---
        ('ALBEDO', 'Broadband Surface Albedo — Liang (2001) / Bonafoni & Sekertekin (2020)',
         'other', '0.356·B + 0.130·R + 0.373·N + 0.085·S1 + 0.072·S2 − 0.0018  [Landsat]',
         'Liang (2001) / Bonafoni & Sekertekin (2020)'),
    ]
    df = pd.DataFrame(data, columns=['short_name', 'long_name', 'domain', 'formula', 'reference'])
    print("⚠️  Using hardcoded index list (spyndex not available)")
    return df


def calculate_spectral_indices_dynamic(image, indices_list, sensor='landsat', verbose=True):
    """
    Dynamically computes spectral indices using spyndex/ee.Image.expression.

    Tries eemont first (image.spectralIndices), and if it fails,
    computes manually via ee.Image.expression using spyndex formulas.

    Args:
        image (ee.Image): Image with optical bands
        indices_list (list): List of index names (e.g., ['NDVI', 'NDBI', 'SAVI'])
        sensor (str): 'landsat' or 'sentinel2'

    Returns:
        dict: {index_name_lower: ee.Image} for each computed index
    """
    band_map = _BAND_MAPPING.get(sensor, _BAND_MAPPING['landsat'])
    results = {}

    if verbose:
        print(f"\n{'='*80}")
        print(f"COMPUTING DYNAMIC SPECTRAL INDICES ({sensor.upper()})")
        print(f"{'='*80}")
        print(f"  Requested indices: {', '.join(indices_list)}")

    # Build case-insensitive map of spyndex catalog (used by eemont and fallback)
    spyndex_name_map = {}
    try:
        if SPYNDEX_AVAILABLE:
            for key in spyndex.indices:
                spyndex_name_map[key.upper()] = key
    except Exception:
        pass

    # Resolve names before passing to eemont (e.g.: AWEINSH → AWEIsh)
    resolved_indices = []
    original_to_resolved = {}
    for idx in indices_list:
        resolved = spyndex_name_map.get(idx.upper(), idx)
        resolved_indices.append(resolved)
        original_to_resolved[idx] = resolved
        if resolved != idx:
            if verbose: print(f"  ℹ️  '{idx}' → '{resolved}' (catalog name corrected)")

    # Try via eemont first
    eemont_success = False
    try:
        import eemont
        # eemont expects the image to have standard band names
        # Try to process all at once (with resolved names)
        result_image = image.spectralIndices(resolved_indices)

        # Get actual band names (eemont may name them differently)
        try:
            actual_bands = result_image.bandNames().getInfo()
        except Exception:
            actual_bands = []

        # Build case-insensitive map of actual names
        band_name_map = {b.upper(): b for b in actual_bands}

        for idx_name in indices_list:
            idx_upper = idx_name.upper()
            # Look up the actual band name (case-insensitive)
            real_band_name = band_name_map.get(idx_upper)
            if real_band_name is None:
                # Try partial match (e.g.: AWEInsh vs AWEINSH)
                for actual in actual_bands:
                    if actual.upper() == idx_upper:
                        real_band_name = actual
                        break
            if real_band_name:
                try:
                    band = result_image.select(real_band_name)
                    results[idx_name.lower()] = band.rename(idx_upper)
                    if verbose: print(f"  ✓ {idx_name} computed via eemont (band: {real_band_name})")
                except Exception as e:
                    if verbose: print(f"  ⚠️  {idx_name}: band '{real_band_name}' failed - {e}")
            else:
                if verbose: print(f"  ⚠️  {idx_name}: not found in eemont output (bands: {actual_bands})")
        if results:
            eemont_success = True
    except Exception as e:
        if verbose:
            print(f"  ⚠️  eemont failed: {e}")
            print(f"  Using manual computation via spyndex...")

    # Fallback: compute manually using spyndex formulas
    remaining = [idx for idx in indices_list if idx.lower() not in results]
    if remaining:
        # Prepare band parameters as ee.Image
        params = {}
        for spyndex_band, image_band in band_map.items():
            try:
                params[spyndex_band] = image.select(image_band)
            except Exception:
                pass

        # Add constants
        for param_name, param_val in _DEFAULT_PARAMS.items():
            params[param_name] = param_val

        # spyndex_name_map already built above (before eemont)

        for idx_name in remaining:
            try:
                # Resolve name case-insensitively in spyndex catalog
                resolved_name = spyndex_name_map.get(idx_name.upper(), idx_name)

                if SPYNDEX_AVAILABLE and resolved_name in spyndex.indices:
                    idx_obj = spyndex.indices[resolved_name]
                    formula = idx_obj.formula
                    required_bands = idx_obj.bands

                    # Check available bands
                    expr_params = {}
                    all_available = True
                    for b in required_bands:
                        if b in params:
                            expr_params[b] = params[b]
                        else:
                            all_available = False
                            break

                    if all_available:
                        result = image.expression(formula, expr_params).rename(idx_name.upper())
                        results[idx_name.lower()] = result
                        if verbose: print(f"  ✓ {idx_name} computed via spyndex expression (catalog: {resolved_name})")
                    else:
                        if verbose: print(f"  ⚠️  {idx_name}: insufficient bands ({required_bands})")
                else:
                    # Fallback to hardcoded indices
                    result = _calculate_hardcoded_index(image, idx_name, sensor)
                    if result is not None:
                        results[idx_name.lower()] = result
                        if verbose: print(f"  ✓ {idx_name} computed via manual formula")
                    else:
                        if verbose: print(f"  ⚠️  {idx_name}: not available in catalog")
            except Exception as e:
                if verbose: print(f"  ❌ {idx_name}: computation error - {e}")

    if verbose:
        print(f"\n  📊 Total indices computed: {len(results)}/{len(indices_list)}")
        print(f"{'='*80}\n")

    return results


def _calculate_hardcoded_index(image, idx_name, sensor='landsat'):
    """Computes indices using manual formulas (fallback)."""
    band_map = _BAND_MAPPING.get(sensor, _BAND_MAPPING['landsat'])

    try:
        R = image.select(band_map['R'])
        G = image.select(band_map['G'])
        B = image.select(band_map['B'])
        N = image.select(band_map['N'])
        S1 = image.select(band_map['S1'])
        S2 = image.select(band_map['S2'])
    except Exception:
        return None

    L = 0.5
    idx = idx_name.upper()

    formulas = {
        'NDVI': lambda: ((N.subtract(R)).divide(N.add(R))).rename('NDVI'),
        'EVI': lambda: (N.subtract(R)).multiply(2.5).divide(
            N.add(R.multiply(6)).subtract(B.multiply(7.5)).add(1)).rename('EVI'),
        'SAVI': lambda: ((N.subtract(R)).divide(N.add(R).add(L))).multiply(1 + L).rename('SAVI'),
        'NDWI': lambda: ((G.subtract(N)).divide(G.add(N))).rename('NDWI'),
        'MNDWI': lambda: ((G.subtract(S1)).divide(G.add(S1))).rename('MNDWI'),
        'NDBI': lambda: ((S1.subtract(N)).divide(S1.add(N))).rename('NDBI'),
        'UI': lambda: ((S1.add(R).subtract(N)).divide(S1.add(R).add(N))).rename('UI'),
        'BSI': lambda: ((S1.add(R)).subtract(N.add(B))).divide(
            (S1.add(R)).add(N.add(B))).rename('BSI'),
        'NDTI': lambda: ((S1.subtract(S2)).divide(S1.add(S2))).rename('NDTI'),
        'DBSI': lambda: (((S1.subtract(G)).divide(S1.add(G))).subtract(
            (N.subtract(R)).divide(N.add(R)))).rename('DBSI'),
        'NBAI': lambda: ((S2.subtract(S1.divide(G))).divide(
            S2.add(S1.divide(G)))).rename('NBAI'),
        'NBI': lambda: ((R.multiply(S2)).divide(N)).rename('NBI'),
        'IBI': lambda: _calc_ibi(N, R, S1, G, L),
        # Surface albedo — Liang (2001), for Landsat SR
        'ALBEDO': lambda: _calc_albedo_liang_landsat(B, R, N, S1, S2),
    }

    if idx in formulas:
        return formulas[idx]()
    return None


def _calc_ibi(N, R, S1, G, L=0.5):
    """Computes IBI (Index-Based Built-Up Index)."""
    ndbi = (S1.subtract(N)).divide(S1.add(N))
    savi = ((N.subtract(R)).divide(N.add(R).add(L))).multiply(1 + L)
    mndwi = (G.subtract(S1)).divide(G.add(S1))
    ndbi_p = ndbi.add(1)
    savi_p = savi.add(1)
    mndwi_p = mndwi.add(1)
    ibi = (ndbi_p.subtract((savi_p.add(mndwi_p)).divide(2))).divide(
        ndbi_p.add((savi_p.add(mndwi_p)).divide(2)))
    return ibi.rename('IBI')


def _calc_albedo_liang_landsat(B, R, N, S1, S2):
    """
    Broadband surface albedo for Landsat 8/9 OLI — Liang (2001).

    Narrow-to-broadband conversion formula validated for surface reflectance
    (SR), compatible with Landsat C02 Level-2 (T1_L2).
    Applicable to urban and general terrestrial environments.

    Reference:
        Liang, S. (2001). Narrowband to broadband conversions of land
        surface albedo I: Algorithms. Remote Sensing of Environment,
        76(2), 213–238. https://doi.org/10.1016/S0034-4257(00)00205-4

        OLI adaptation: Smith, R. B. (2010) & NASA HLS review.

    Args:
        B, R, N, S1, S2 (ee.Image): Blue, Red, NIR, SWIR1, SWIR2 bands
            in surface reflectance (range 0–1).

    Returns:
        ee.Image: Albedo image (approx. 0–1), band named 'ALBEDO'.
    """
    return (B.multiply(0.356)
             .add(R.multiply(0.130))
             .add(N.multiply(0.373))
             .add(S1.multiply(0.085))
             .add(S2.multiply(0.072))
             .subtract(0.0018)
             ).rename('ALBEDO')


def _calc_albedo_bonafoni_s2(B, G, R, N, S1, S2):
    """
    Broadband surface albedo for Sentinel-2 — Bonafoni & Sekertekin (2020).

    Coefficients derived specifically for Sentinel-2 surface reflectance
    (10 m), validated in urban environments via field measurements.
    Compatible with S2_SR_HARMONIZED (Level-2A).

    Reference:
        Bonafoni, S., & Sekertekin, A. (2020). Albedo Retrieval from
        Sentinel-2 by New Narrow-to-Broadband Conversion Coefficients.
        IEEE Geoscience and Remote Sensing Letters, 17(9), 1618–1622.
        https://doi.org/10.1109/LGRS.2020.2971650

    Args:
        B, G, R, N, S1, S2 (ee.Image): B2, B3, B4, B8, B11, B12 bands
            in surface reflectance (range 0–1).

    Returns:
        ee.Image: Albedo image (approx. 0–1), band named 'ALBEDO'.
    """
    return (B.multiply(0.2266)
             .add(G.multiply(0.1236))
             .add(R.multiply(0.1573))
             .add(N.multiply(0.3417))
             .add(S1.multiply(0.1170))
             .add(S2.multiply(0.0338))
             ).rename('ALBEDO')




def _calc_fvc_emissivity(ndvi):
    """
    Computes FVC (Fractional Vegetation Cover) and surface emissivity via
    the NDVI Thresholds Method.

    Reference:
        Sobrino, J.A., Jiménez-Muñoz, J.C., & Paolini, L. (2004). Land surface
        temperature retrieval from LANDSAT TM 5. Remote Sensing of Environment,
        90(4), 434-440. https://doi.org/10.1016/j.rse.2004.02.003

    Constants used:
        NDVI_SOIL = 0.2  — lower threshold (bare soil)
        NDVI_VEG  = 0.5  — upper threshold (dense vegetation)
        E_SOIL    = 0.960 — soil emissivity (TIR band 10 µm)
        E_VEG     = 0.985 — full vegetation emissivity
        D_EPS     = 0.0038 — cavity effect (surface roughness)

    Args:
        ndvi (ee.Image): Pre-computed NDVI image with named band.

    Returns:
        tuple(ee.Image, ee.Image): (FVC, Emissivity), both renamed.
    """
    NDVI_SOLO = 0.2
    NDVI_VEG  = 0.5
    E_SOLO    = 0.960
    E_VEG     = 0.985
    D_EPS     = 0.0038

    # FVC: quadratic vegetation cover fraction — clamped to [0, 1]
    fvc = (ndvi.subtract(NDVI_SOLO)
               .divide(NDVI_VEG - NDVI_SOLO)
               .clamp(0, 1)
               .pow(2)
               .rename('FVC'))

    # Mixed emissivity: vegetation + soil + cavity effect
    # ε = ε_veg·FVC + ε_soil·(1-FVC) + ΔC·FVC·(1-FVC)
    one_minus_fvc = fvc.multiply(-1).add(1)
    e_mista = (fvc.multiply(E_VEG)
                  .add(one_minus_fvc.multiply(E_SOLO))
                  .add(fvc.multiply(one_minus_fvc).multiply(D_EPS)))

    # Apply thresholds: bare soil | mixed | full vegetation
    emissivity = (ee.Image(E_SOLO)
                  .where(ndvi.gte(NDVI_VEG), ee.Image(E_VEG))
                  .where(ndvi.gte(NDVI_SOLO).And(ndvi.lt(NDVI_VEG)), e_mista)
                  .rename('Emissivity'))

    return fvc, emissivity


def calculate_spectral_indices_landsat(image):
    """
    Computes spectral indices for a Landsat 8/9 image.

    Args:
        image: Processed Landsat image with optical bands

    Returns:
        dict: Dictionary with computed indices (ndvi, ndwi, ndbi, ui)
    """

    # Redefine bands based on reference

    R = image.select('SR_B4')   # Red
    G = image.select('SR_B3')   # Green
    B = image.select('SR_B2')   # Blue
    N = image.select('SR_B5')   # NIR
    SWIR1 = image.select('SR_B6')  # SWIR1
    SWIR2 = image.select('SR_B7')  # SWIR2
    T1 = image.select('ST_B10')  # Temperature
    L = 0.5 # soil correction factor

    # VEGETATION ---------------------------------------------------------------

    # NDVI (Normalized Difference Vegetation Index)
    ndvi = ((N - R)/(N + R)).rename('NDVI')

    # EVI (Enhanced Vegetation Index)
    evi = (2.5*(N-R)/(N+(6*R)-(7.5*B)+1)).rename('EVI')

    # SAVI (Soil-Adjusted Vegetation Index)
    savi = (((N - R) / (N + R + L)) * (1 + L)).rename('SAVI')

    # WATER -------------------------------------------------------------------

    # NDWI (Normalized Difference Water Index)
    ndwi = ((G - N) / (G + N)).rename('NDWI')

    # MNDWI (Modified Normalized Difference Water Index)
    mndwi = ((G - SWIR1) / (G + SWIR1)).rename('MNDWI')

    # SOIL --------------------------------------------------------------------
    bsi = ((SWIR1 + R) - (N + B))/((SWIR1 + R) + (N + B)).rename('BSI')
    ndti = ((SWIR1 - SWIR2)/(SWIR1 + SWIR2)).rename('NDTI')
    dbsi = (((SWIR1 - G)/(SWIR1 + G)) - ((N - R)/(N + R))).rename('DBSI')

    # URBAN -------------------------------------------------------------------
    ndbi = ((SWIR1 - N) / (SWIR1 + N)).rename('NDBI')
    nbai = ((SWIR2 - SWIR1/G)/(SWIR2 + SWIR1/G)).rename('NBAI')
    nbi = ((R * SWIR2) / (N)).rename('NBI')
    # UI (Urban Index)
    ui = ((SWIR1 + R - N) / (SWIR1 + R + N)).rename('UI')


    # MULTI-BAND COMBINATIONS -------------------------------------------------

    # IBI (Index-Based Built-Up Index)
    NDBI_, SAVI_, MNDWI_ = ndbi.add(1), savi.add(1), mndwi.add(1) # Addition required for IBI
    Ibi = ((NDBI_ - ((SAVI_ + MNDWI_)/2)) / (NDBI_ + ((SAVI_ + MNDWI_)/2))).rename('IBI')

    # Combination proposed by Rouibah & Belabbas (2020)  Doi: 10.4995/raet.2020.13787
    urban_arid = (ndti + bsi + ndvi).rename('urban_arid')

    # SURFACE ALBEDO (narrowband to broadband) --------------------------------
    # Liang (2001), adapted for Landsat 8/9 OLI — surface reflectance (SR)
    # Validated for terrestrial and urban environments.
    # Ref.: Liang (2001), Remote Sens. Environ. 76(2):213-238.
    albedo = _calc_albedo_liang_landsat(B, R, N, SWIR1, SWIR2)

    # FVC and surface emissivity (Sobrino et al., 2004)
    fvc, emissivity = _calc_fvc_emissivity(ndvi)

    # Uncomment to use indices for correlation testing
    # and checking collinearity — high-collinearity ones were commented out
    return {
        'ndvi': ndvi,
        'evi': evi,
        'savi': savi,
        'ndwi': ndwi,
        'mndwi': mndwi,
        'bsi': bsi,
        'ndti': ndti,
        'dbsi': dbsi,
        'ndbi': ndbi,
        'nbai': nbai,
        'nbi': nbi,
        'ibi': Ibi,
        'urban_arid': urban_arid,
        'ui': ui,
        'albedo': albedo,
        'fvc': fvc,
        'emissivity': emissivity,
    }



def calculate_spectral_indices_sentinel2(image):
    """
    Computes spectral indices for a Sentinel-2 image.

    Args:
        image: Dictionary with adjusted Sentinel-2 bands

    Returns:
        dict: Dictionary with computed indices (ndvi, ndwi, ndbi, ui)
    """
    # Create combined image from dictionary
    s2_img = ee.Image.cat([
        image['B2'].rename('B2'),
        image['B3'].rename('B3'),
        image['B4'].rename('B4'),
        image['B8'].rename('B8'),
        image['B11'].rename('B11'),
        image['B12'].rename('B12')
    ])

    # Redefine bands based on reference

    R = image.select('B4')  # Red
    G = image.select('B3')  # Green
    B = image.select('B2')  # Blue
    N = image.select('B8')  # NIR
    SWIR1 = image.select('B11') # SWIR1
    SWIR2 = image.select('B12') # SWIR2
    L = 0.5 # soil correction factor

    # VEGETATION ---------------------------------------------------------------

    # NDVI (Normalized Difference Vegetation Index)
    ndvi = ((N - R)/(N + R)).rename('NDVI')

    # EVI (Enhanced Vegetation Index)
    evi = (2.5*(N-R)/(N+(6*R)-(7.5*B)+1)).rename('EVI')

    # SAVI (Soil-Adjusted Vegetation Index)
    savi = (((N - R) / (N + R + L)) * (1 + L)).rename('SAVI')

    # WATER -------------------------------------------------------------------

    # NDWI (Normalized Difference Water Index)
    ndwi = ((G - N) / (G + N)).rename('NDWI')

    # MNDWI (Modified Normalized Difference Water Index)
    mndwi = ((G - SWIR1) / (G + SWIR1)).rename('MNDWI')

    # SOLO --------------------------------------------------------------------
    bsi = ((SWIR1 + R) - (N + B))/((SWIR1 + R) + (N + B)).rename('BSI')
    ndti = ((SWIR1 - SWIR2)/(SWIR1 + SWIR2)).rename('NDTI')
    dbsi = (((SWIR1 - G)/(SWIR1 + G)) - ((N - R)/(N + R))).rename('DBSI')

    # AMBIENTE URBANO ---------------------------------------------------------
    ndbi = ((SWIR1 - N) / (SWIR1 + N)).rename('NDBI')
    nbai = ((SWIR2 - SWIR1/G)/(SWIR2 + SWIR1/G)).rename('NBAI')
    nbi = ((R * SWIR2) / (N)).rename('NBI')
    # UI (Urban Index)
    ui = ((SWIR1 + R - N) / (SWIR1 + R + N)).rename('UI')


    # MULTI-BAND COMBINATIONS -------------------------------------------------

    # IBI (Index-Based Built-Up Index)
    NDBI_, SAVI_, MNDWI_ = ndbi.add(1), savi.add(1), mndwi.add(1) # Addition required for IBI
    Ibi = ((NDBI_ - ((SAVI_ + MNDWI_)/2)) / (NDBI_ + ((SAVI_ + MNDWI_)/2))).rename('IBI')

    # Combination proposed by: Rouibah & Belabbas (2020)  Doi: 10.4995/raet.2020.13787
    urban_arid = (ndti + bsi + ndvi).rename('urban_arid')

    # SURFACE ALBEDO (narrowband to broadband) ---------------------------------
    # Bonafoni & Sekertekin (2020) — coefficients for Sentinel-2 SR (Level-2A)
    # Validated specifically for urban environments, 10 m resolution.
    # Ref.: Bonafoni & Sekertekin (2020), IEEE GRSL 17(9):1618-1622.
    albedo = _calc_albedo_bonafoni_s2(B, G, R, N, SWIR1, SWIR2)

    # FVC and surface emissivity (Sobrino et al., 2004)
    fvc, emissivity = _calc_fvc_emissivity(ndvi)

    return {
        'ndvi': ndvi,
        'evi': evi,
        'savi': savi,
        'ndwi': ndwi,
        'mndwi': mndwi,
        'bsi': bsi,
        'ndti': ndti,
        'dbsi': dbsi,
        'ndbi': ndbi,
        'nbai': nbai,
        'nbi': nbi,
        'ibi': Ibi,
        'urban_arid': urban_arid,
        'ui': ui,
        'albedo': albedo,
        'fvc': fvc,
        'emissivity': emissivity,
    }




def calculate_spectral_unmixing(sentinel2_image, roi, period_name='T',
                                 unmix_scale=30, verbose=True):
    """
    Linear Spectral Unmixing (LSU) of a Sentinel-2 image, producing a
    per-class fraction map.

    The method assumes each pixel is a linear mixture of pure spectra
    (endmembers). Endmembers are extracted automatically from the image
    using spectral-index thresholds calibrated for semi-arid environments
    (Sobral-CE). Unmixing is solved with non-negativity and unit-sum
    constraints (sumToOne=True, nonNegative=True).

    Output classes:
        - veg_densa      : Dense vegetation (NDVI > 0.50) — riparian forest, parks
        - veg_rasteira   : Sparse vegetation (0.15 < NDVI ≤ 0.50) — caatinga, pasture
        - solo_exposto   : Bare soil (NDVI < 0.15, BSI > 0)
        - agua           : Water bodies (NDWI > 0.15)
        - area_construida: Built-up area (NDBI > 0.05, NDVI < 0.15)

    Ref.:
        Adams, J.B., Smith, M.O., & Johnson, P.E. (1986). Spectral mixture
        modeling: A new analysis of rock and soil types at the Viking Lander 1
        site. Journal of Geophysical Research, 91(B8), 8098-8112.

    Args:
        sentinel2_image (ee.Image or dict): Sentinel-2 bands adjusted by
            bandpass. Must contain B2, B3, B4, B8, B11, B12.
        roi (ee.Geometry): Region of interest for endmember extraction.
        period_name (str): Period label (for logs). Default: 'T'.
        unmix_scale (int): Scale (m) for endmember computation. Default: 30.
        verbose (bool): Print endmember extraction log.

    Returns:
        tuple(ee.Image, dict):
            - fractions (ee.Image): Image with 5 fraction bands [0, 1],
              named after the classes above.
            - endmember_info (dict): Extracted endmember spectra
              {class: [B2, B3, B4, B8, B11, B12]}.

    Raises:
        RuntimeError: If no pure pixels are found for any class.
    """
    BANDS        = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    CLASS_NAMES  = ['veg_densa', 'veg_rasteira', 'solo_exposto', 'agua', 'area_construida']

    if verbose:
        print(f"\n{'='*70}")
        print(f"SPECTRAL UNMIXING — {period_name}")
        print(f"{'='*70}")

    # ── 1. Build ee.Image ─────────────────────────────────────────────────────
    if isinstance(sentinel2_image, dict):
        img = ee.Image.cat([sentinel2_image[b].rename(b)
                            for b in BANDS if b in sentinel2_image])
    else:
        img = sentinel2_image.select(BANDS)

    # ── 2. Indices for endmember extraction ──────────────────────────────────
    R  = img.select('B4')
    G  = img.select('B3')
    B  = img.select('B2')
    N  = img.select('B8')
    S1 = img.select('B11')
    S2 = img.select('B12')

    ndvi = N.subtract(R).divide(N.add(R))
    ndwi = G.subtract(N).divide(G.add(N))   # Gao (1996) — NDWI
    ndbi = S1.subtract(N).divide(S1.add(N))
    bsi  = (S1.add(R).subtract(N.add(B))).divide(S1.add(R).add(N).add(B))

    # ── 3. Pure-pixel masks ───────────────────────────────────────────────────
    # Thresholds calibrated for semi-arid environment (Sobral-CE). Adjust if needed.
    masks = {
        'veg_densa':       ndvi.gt(0.50),
        'veg_rasteira':    ndvi.gt(0.15).And(ndvi.lte(0.50)).And(ndwi.lt(0.0)),
        'solo_exposto':    ndvi.lt(0.15).And(bsi.gt(0.0)).And(ndwi.lt(-0.15)),
        'agua':            ndwi.gt(0.15),
        'area_construida': ndbi.gt(0.05).And(ndvi.lt(0.15)).And(ndwi.lt(-0.05)),
    }

    # ── 4. Extract mean endmember spectra (client-side) ──────────────────────
    endmembers      = []
    endmember_info  = {}
    missing_classes = []

    if verbose:
        print("\nEndmember extraction (mean spectrum per class):")

    for cls in CLASS_NAMES:
        stats = (img.updateMask(masks[cls])
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=roi,
                        scale=unmix_scale,
                        maxPixels=1e10
                    ))
        em_list = ee.List([stats.get(b) for b in BANDS]).getInfo()

        if None in em_list:
            missing_classes.append(cls)
            em_list = [v if v is not None else 0.0 for v in em_list]
            if verbose:
                print(f"   ⚠️  {cls}: insufficient pure pixels — check thresholds")
        else:
            if verbose:
                vals = ', '.join([f'{v:.4f}' for v in em_list])
                print(f"   ✓  {cls}: [{vals}]")

        endmembers.append(em_list)
        endmember_info[cls] = em_list

    if missing_classes:
        print(f"\n⚠️  Classes without adequate endmember: {missing_classes}")
        print("   Tip: use a smaller unmix_scale or adjust index thresholds.")

    # ── 5. Constrained linear unmixing ────────────────────────────────────────
    if verbose:
        print("\nApplying linear spectral unmixing (sumToOne, nonNegative)...")

    fractions = (img.unmix(
                    endmembers=endmembers,
                    sumToOne=True,
                    nonNegative=True
                 )
                 .rename(CLASS_NAMES))

    if verbose:
        print("✓ Unmixing complete.")
        print("  Output bands:", CLASS_NAMES)

    return fractions, endmember_info


def visualize_spectral_indices(s2_median_image, roi, period_name, indices_to_visualize='all', Map=None, scale=15):
    """
    Creates an interactive map to visualize Sentinel-2 spectral indices.

    Args:
        s2_median_image (ee.Image): Median Sentinel-2 image for the period (already scaled).
        roi (ee.Geometry): Region of interest.
        period_name (str): Period label (e.g., 'T1', 'T2') for layer naming.
        indices_to_visualize (str or list): 'all' to visualize all indices or a list
                                             of index names (e.g., ['ndvi', 'ndbi']).
        Map (geemap.Map, optional): Existing map object to add layers to.
                                    If None, a new map will be created.

    Returns:
        geemap.Map: Interactive map with spectral index layers.
    """
    print("\n" + "=" * 80)
    print(f"VISUALIZING SENTINEL-2 SPECTRAL INDICES ON MAP ({period_name})")
    print("=" * 80)

    # Create a map centered on the ROI if none is provided
    if Map is None:
        Map = geemap.Map(height=900)
        Map.centerObject(roi, scale)
        Map.addLayer(roi, {'color': 'yellow'}, 'ROI', False)

    # Always add the current period RGB
    s2_vis_params_rgb = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 0.3,
        'gamma': 1.3
    }
    Map.addLayer(
        s2_median_image.clip(roi),
        s2_vis_params_rgb,
        f'Sentinel-2 RGB {period_name}',
        True
    )


    # Build a dict with the correct band keys (B2, B3, B4, B8, B11, B12)
    s2_bands_dict = {
        'B2': s2_median_image.select('B2'),
        'B3': s2_median_image.select('B3'),
        'B4': s2_median_image.select('B4'),
        'B8': s2_median_image.select('B8'),
        'B11': s2_median_image.select('B11'),
        'B12': s2_median_image.select('B12')
    }

    # Create an ee.Image from the band dictionary
    s2_image_for_indices = ee.Image.cat(list(s2_bands_dict.values()))
    # Rename bands to match names expected by calculate_spectral_indices_sentinel2
    s2_image_for_indices = s2_image_for_indices.rename(list(s2_bands_dict.keys()))

    # Calculate spectral indices
    sentinel2_indices = calculate_spectral_indices_sentinel2(s2_image_for_indices)

    # Filter indices based on user selection
    if indices_to_visualize == 'all':
        indices_to_display = sentinel2_indices
    elif isinstance(indices_to_visualize, list):
        indices_lower = [i.lower() for i in indices_to_visualize]
        indices_to_display = {
            idx_name: idx_image for idx_name, idx_image in sentinel2_indices.items()
            if idx_name.lower() in indices_lower
        }

        # Dynamic fallback: compute missing indices via eemont/spyndex
        found = [k.lower() for k in indices_to_display.keys()]
        missing = [idx for idx in indices_to_visualize if idx.lower() not in found]
        if missing:
            print(f"   ℹ️  Indices not found in hardcoded set: {missing}")
            print(f"   ℹ️  Computing via eemont/spyndex...")
            try:
                # Remove zero-fill from unmask before dynamic computation.
                # Indices like VIBI have denominator (NDVI+NDBI) = 0 when NIR=0,
                # which occurs in pixels filled via unmask(0).
                _s2_for_dynamic = s2_image_for_indices.updateMask(
                    s2_image_for_indices.select('B8').gt(0)
                )
                dynamic_indices = calculate_spectral_indices_dynamic(
                    _s2_for_dynamic, missing, sensor='sentinel2'
                )
                indices_to_display.update(dynamic_indices)
                print(f"   ✓ Dynamic indices computed: {list(dynamic_indices.keys())}")
            except Exception as e:
                print(f"   ⚠️  Failed to compute dynamic indices: {e}")
    else:
        print("⚠️  Invalid format for 'indices_to_visualize'. Visualizing all indices.")
        indices_to_display = sentinel2_indices

    # Palettes by index type (independent of numeric range)
    _PALETTES = {
        'vegetation': ['red', 'yellow', 'green'],   # ndvi, evi, savi
        'water':      ['00FFFF', '0000FF'],          # ndwi, mndwi
        'urban':      ['blue', 'white', 'red'],      # ndbi, ui, ibi, bsi, dbsi, ndti, nbai, nbi, urban_arid
        'albedo':     ['1a1a1a', '5a5a5a', 'aaaaaa', 'ffffff'],
        'default':    ['purple', 'white', 'orange'],
    }

    def _palette_for(name):
        n = name.lower()
        if n in ['ndvi', 'evi', 'savi']:
            return _PALETTES['vegetation']
        if n in ['ndwi', 'mndwi']:
            return _PALETTES['water']
        if n in ['ndbi', 'ui', 'ibi', 'bsi', 'dbsi', 'ndti', 'nbai', 'nbi', 'urban_arid']:
            return _PALETTES['urban']
        if n == 'albedo':
            return _PALETTES['albedo']
        return _PALETTES['default']

    def _data_range(image, geometry, reducer_scale):
        """Returns (vmin, vmax) via p2–p98 percentile of the image within the geometry."""
        try:
            band_name = image.bandNames().getInfo()[0]
            stats = image.rename('val').reduceRegion(
                reducer=ee.Reducer.percentile([2, 98]),
                geometry=geometry,
                scale=reducer_scale,
                maxPixels=1e9,
                bestEffort=True,
            ).getInfo()
            vmin = stats.get('val_p2')
            vmax = stats.get('val_p98')
            if vmin is None or vmax is None:
                return None, None
            vmin, vmax = float(vmin), float(vmax)
            if vmin == vmax:
                # Nearly constant image: create minimum range for visualization
                delta = max(abs(vmin) * 0.1, 0.5)
                vmin -= delta
                vmax += delta
            return vmin, vmax
        except Exception as exc:
            print(f"      ⚠️  Failed to compute range for {band_name}: {exc}")
            return None, None

    if not indices_to_display:
        print("⚠️  No index selected or found for visualization!")
    else:
        # Add each selected index as a layer to the map
        print("\n🗺️  Adding index layers to map...")
        for idx_name, idx_image in indices_to_display.items():
            vmin, vmax = _data_range(idx_image.clip(roi), roi, scale)

            if vmin is None:
                # Fallback: no data, skip layer
                print(f"   ⚠️  Could not determine range for '{idx_name}' — layer skipped")
                continue

            vis_params = {
                'min': vmin,
                'max': vmax,
                'palette': _palette_for(idx_name),
            }

            try:
                Map.addLayer(
                    idx_image.clip(roi),
                    vis_params,
                    f'{period_name} - {idx_name.upper()} (S2)',
                    False  # Layers are disabled by default
                )
                print(f"   ✓ Layer '{period_name} - {idx_name.upper()} (S2)' added  "
                      f"(min={vmin:.4f}, max={vmax:.4f})")
            except Exception as e:
                print(f"   ⚠️  Error adding layer '{period_name} - {idx_name.upper()} (S2)': {e}")

    print(f"\n✓ Index visualization for {period_name} created")

    return Map



def _apply_iqr_filter(df, multiplier=1.5):
    """Remove rows where any numeric variable is an outlier by the IQR criterion.

    For each numeric column computes Q1, Q3, and IQR = Q3 - Q1. Flags as an
    outlier any value outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR].
    Removes rows where ANY column is an outlier (conservative intersection).

    Args:
        df (pd.DataFrame): DataFrame with numeric columns.
        multiplier (float): IQR factor. Default 1.5 (Tukey criterion).

    Returns:
        pd.DataFrame: Filtered DataFrame (index reset).
    """
    mask = pd.Series(True, index=df.index)
    for col in df.select_dtypes(include='number').columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue  # constant variable — do not filter
        mask &= (df[col] >= q1 - multiplier * iqr) & (df[col] <= q3 + multiplier * iqr)
    return df[mask].reset_index(drop=True)


def analyze_spectral_indices_correlation(
    lst_30m,
    landsat_image,
    geometry,
    n_samples=1000,
    indices_to_analyze='all',
    use_dynamic=True,
    custom_indices_list=None,
    period_name='Period',
    plot_figures=True,
    include_domain=False,
    top_k=None,
    verbose=True,
    corr_method='pearson',
    presampled_df=None,
    iqr_multiplier=1.5,
    presampled_already_filtered=False,
):
    """
    Analyzes the correlation between spectral indices and LST 30m / UHI.
    Updated version with support for dynamic indices via spyndex/eemont.

    Args:
        lst_30m (ee.Image): LST image at 30m.
        landsat_image (ee.Image): Processed Landsat image.
        geometry: Region of interest geometry.
        n_samples (int): Number of sample points for correlation.
        indices_to_analyze (str or list): 'all' for all indices or a specific list.
            E.g.: ['NDVI', 'NDBI', 'SAVI', 'NDWI', 'EVI', 'BSI', 'UI']
        use_dynamic (bool): If True, uses calculate_spectral_indices_dynamic (spyndex).
            If False, uses calculate_spectral_indices_landsat (manual).
        custom_indices_list (list, optional): Custom index list.
            If provided, overrides indices_to_analyze.
        period_name (str): Period label (e.g., 'T1', 'T2') for titles.
        corr_method (str): Correlation method: 'pearson' (linear, default) or
            'spearman' (rank-based, robust to outliers and monotonic non-linear
            relationships). Must be the same in 4.C1 and 4.D1 for valid comparison.
        presampled_df (pd.DataFrame, optional): Already-sampled DataFrame (e.g.,
            results_sensitivity_t1['data']). When provided, ignores lst_30m,
            landsat_image, geometry, and n_samples — reuses pixels from a prior
            call, guaranteeing identical sample population. Useful for 4.D1 to
            compare individual indices against the sensitivity analysis (4.C1).

    Returns:
        dict: Correlation summary, data, and MLR recommendations.
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"CORRELATION ANALYSIS — SPECTRAL INDICES vs LST ({period_name})")
        print("=" * 80)

    # Load catalog once and reuse for domain_map (avoids a double call)
    _catalog_cache = None

    if presampled_df is not None:
        # Reuse DataFrame from a previous call (e.g., results_sensitivity_t1['data']),
        # ensuring 4.D1 operates on the same pixel population as 4.C1.
        if verbose:
            print(f"\n♻️  Using pre-sampled DataFrame ({len(presampled_df)} points) — no new GEE call.")
        df = presampled_df.copy()
        _non_idx = {'LST_30m', 'system:index', '.geo'}
        _df_idx_cols = [c for c in df.columns if c not in _non_idx]
        if custom_indices_list is not None:
            indices_list = [i.upper() for i in custom_indices_list if i.upper() in _df_idx_cols]
        elif isinstance(indices_to_analyze, list):
            indices_list = [i.upper() for i in indices_to_analyze if i.upper() in _df_idx_cols]
        else:
            indices_list = _df_idx_cols
        if not indices_list:
            if verbose: print("⚠️  No requested index found in the pre-sampled DataFrame!")
            return None
        if verbose: print(f"   Analyzing {len(indices_list)} index/indices: {indices_list}")
    else:
        # Define list of indices to compute
        if custom_indices_list is not None:
            target_indices = [idx.upper() for idx in custom_indices_list]
        elif isinstance(indices_to_analyze, list):
            target_indices = [idx.upper() for idx in indices_to_analyze]
        elif indices_to_analyze == 'all':
            try:
                _catalog_cache = list_available_indices(platforms=None, verbose=verbose)
                target_indices = _catalog_cache['short_name'].str.upper().tolist()
            except Exception as e:
                if verbose: print(f"⚠️  Error loading full catalog: {e}. Using fallback.")
                target_indices = ['NDVI', 'SAVI', 'NDWI', 'NDBI', 'DBSI', 'NBAI', 'EVI', 'BSI', 'UI', 'MNDWI']
        else:
            target_indices = ['NDVI', 'SAVI', 'NDWI', 'NDBI', 'DBSI', 'NBAI', 'EVI', 'BSI', 'UI', 'MNDWI']

        # Compute spectral indices (with support for generic bands)
        if verbose: print("\n🔢 Computing spectral indices...")
        indices = resolve_predictors_from_image(
            landsat_image, target_indices, sensor='landsat', verbose=verbose
        )

        # Check which predictors are available
        available_indices = list(indices.keys())
        if verbose: print(f"   Available indices: {[i.upper() for i in available_indices]}")

        if indices_to_analyze == 'all':
            indices_list = available_indices
        elif isinstance(indices_to_analyze, list) or custom_indices_list:
            indices_list = [idx for idx in available_indices]
        else:
            indices_list = available_indices

        if not indices_list:
            if verbose: print("⚠️  No index found for analysis!")
            return None

        if verbose: print(f"   Analyzing {len(indices_list)} predictors: {[i.upper() for i in indices_list]}")

        # === PREPARE COMBINED IMAGE ===
        combined = lst_30m.rename('LST_30m')
        for idx in indices_list:
            combined = combined.addBands(indices[idx].rename(idx.upper()))

        # === SAMPLE POINTS ===
        if verbose: print(f"\n📊 Sampling {n_samples} points...")
        sample = combined.sample(
            region=geometry,
            scale=30,
            numPixels=n_samples,
            seed=42,
            geometries=False,
            # tileScale: splits tiles into smaller blocks — reduces GEE memory usage
            # and often speeds up images with many bands (100+)
            tileScale=4,
            # dropNulls: avoids resampling masked pixels, reducing GEE retries
            dropNulls=True,
        )

        # Convert to DataFrame
        sample_list = sample.getInfo()['features']
        if not sample_list:
            if verbose: print("⚠️  No sample points obtained!")
            return None

        data = [feature['properties'] for feature in sample_list]
        df = pd.DataFrame(data)
        if verbose: print(f"   ✓ {len(df)} sample points obtained")

    _corr_method = corr_method if corr_method in ('pearson', 'spearman', 'pearson_iqr') else 'pearson'

    # Derive method variables once — used in prints and plots
    _pearson_iqr    = _corr_method == 'pearson_iqr'
    _is_spearman    = _corr_method == 'spearman'
    _corr_method_pd = 'pearson' if _pearson_iqr else _corr_method  # method accepted by pandas
    _r_sym  = 'ρ' if _is_spearman else ('r_IQR' if _pearson_iqr else 'r')
    _r2_sym = 'ρ²' if _is_spearman else ('R²_IQR' if _pearson_iqr else 'R²')

    # === COMPUTE CORRELATIONS with LST (vectorized) ===
    if verbose: print("\n📈 Computing correlations with LST (vectorized)...")
    domain_map = {}
    if include_domain:
        try:
            # Reuse already-loaded catalog (avoids a second spyndex call)
            cat = _catalog_cache if _catalog_cache is not None else list_available_indices(platforms=None, verbose=False)
            domain_map = dict(zip(cat['short_name'].str.upper(), cat['domain']))
        except Exception:
            pass

    # Select available columns for vectorized correlation
    idx_cols_available = [idx.upper() for idx in indices_list if idx.upper() in df.columns]
    target_cols = ['LST_30m'] + idx_cols_available
    target_cols = [c for c in target_cols if c in df.columns]

    # IQR filter (only for pearson_iqr): removes outliers before computing correlation.
    # Applied to the FULL df (all numeric columns), not just target_cols,
    # ensuring 4.C and 4.D use exactly the same rows when df is the same.
    _df_filtered_full = df  # default: sem filtragem
    if _pearson_iqr:
        if not presampled_already_filtered:
            _df_filtered_full = _apply_iqr_filter(df, multiplier=iqr_multiplier)
            _n_iqr_removed = len(df) - len(_df_filtered_full)
            if verbose:
                _pct_iqr = 100 * _n_iqr_removed / max(len(df), 1)
                print(f"   🔎 IQR filter ({iqr_multiplier}×IQR): {_n_iqr_removed} outliers removed "
                      f"({_pct_iqr:.1f}% of {len(df)} pixels) → {len(_df_filtered_full)} retained")
        else:
            _n_iqr_removed = 0
            if verbose:
                print(f"   ♻️  Data already pre-filtered by IQR ({iqr_multiplier}×IQR) — filter not reapplied.")
        df_for_corr = _df_filtered_full[target_cols]
    else:
        df_for_corr = df[target_cols]
        _n_iqr_removed = 0

    # Single .corr() call — much faster than N np.corrcoef loops
    corr_full = df_for_corr.corr(method=_corr_method_pd, min_periods=10)

    # Extract correlations with LST
    correlations = []
    if 'LST_30m' in corr_full.columns:
        for idx_upper in idx_cols_available:
            if idx_upper in corr_full.index:
                r = corr_full.loc[idx_upper, 'LST_30m']
                if pd.notna(r):
                    r2 = r ** 2
                    corr_dict = {
                        'indice': idx_upper, 'r': r, 'r2': r2,
                        'tipo': 'positive' if r > 0 else 'negative',
                        'forca': 'strong' if abs(r) > 0.7 else 'moderate' if abs(r) > 0.5 else 'weak'
                    }
                    if include_domain:
                        corr_dict['dominio'] = domain_map.get(idx_upper, 'other')
                    correlations.append(corr_dict)

    df_corr = pd.DataFrame(correlations).sort_values('r2', ascending=False) if correlations else pd.DataFrame()

    # === DISPLAY RESULTS ===
    # Apply top_k limit
    df_to_print = df_corr.head(top_k) if top_k is not None else df_corr

    if verbose or (top_k is not None and top_k <= 50):
        if _is_spearman:
            _method_label = 'Spearman — ρ (monotonic, rank-based)'
        elif _pearson_iqr:
            _method_label = f'Pearson w/ IQR filter — r_IQR ({_n_iqr_removed} outliers removed)'
        else:
            _method_label = 'Pearson — r (linear)'
        print("\n" + "=" * 80)
        print(f"📊 CORRELATION RESULTS ({period_name})")
        print(f"   method: {_method_label}")
        print("=" * 80)

        _col_r  = f'{_r_sym} (LST)'
        _col_r2 = f'{_r2_sym} (LST)'
        if include_domain:
            print(f"\n{'Index':<12} {'Domain':<13} {_col_r:<10} {_col_r2:<10} {'Correlation':<12} {'Strength':<10}")
            print("-" * 90)
        else:
            print(f"\n{'Index':<12} {_col_r:<10} {_col_r2:<10} {'Correlation':<12} {'Strength':<10}")
            print("-" * 80)

        for _, row in df_to_print.iterrows():
            if include_domain:
                print(f"{row['indice']:<12} {row['dominio']:<13} {row['r']:<+10.4f} {row['r2']:<10.4f} {row['tipo']:<12} {row['forca']:<10}")
            else:
                print(f"{row['indice']:<12} {row['r']:<+10.4f} {row['r2']:<10.4f} {row['tipo']:<12} {row['forca']:<10}")

        if top_k is not None and len(df_corr) > top_k:
            print(f"... and {len(df_corr) - top_k} more indices hidden. Showing top {top_k}.")
    else:
        print(f"\n📊 {len(df_corr)} correlations computed ({period_name}). Top 5 by {_r2_sym}:")
        for _, row in df_corr.head(5).iterrows():
            print(f"   {row['indice']:<10} {_r2_sym}={row['r2']:.4f}  ({row['tipo']})")
        print(f"   Use rank_predictors_by_importance() for the full ranking.")

    if plot_figures:
        # === FULL CORRELATION MATRIX HEATMAP ===
        print("\n🌡️  Generating correlation matrix heatmap...")
        # Reuse already-computed corr_full — avoids a second .corr() call
        cols_available = [c for c in target_cols if c in corr_full.columns]
        corr_matrix = corr_full.loc[cols_available, cols_available]

        if not corr_matrix.empty:
            hm_width = max(8.0, len(cols_available)*0.9)
            hm_height = max(6.0, len(cols_available)*0.7)
            fig_hm, ax_hm = plt.subplots(figsize=(hm_width, hm_height))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt='.2f', linewidths=0.5,
                        center=0, vmin=-1, vmax=1, ax=ax_hm,
                        annot_kws={'size': 9})
            if _is_spearman:
                _hm_method = 'Spearman (ρ)'
            elif _pearson_iqr:
                _hm_method = f'Pearson w/ IQR (r_IQR) — {_n_iqr_removed} outliers removed'
            else:
                _hm_method = 'Pearson (r)'
            ax_hm.set_title(f'Correlation Matrix {_hm_method} — {period_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
        # === SCATTER PLOTS (LST) ===
        print("\n📊 Generating scatter plots (Indices vs LST)...")
        n_indices = len(indices_list)
        n_cols = min(3, n_indices)
        n_rows = int(np.ceil(n_indices / n_cols))

        # Pre-compute: lookup dict and LST mask without NaN (avoids N filters in loop)
        corr_lookup = df_corr.set_index('indice')[['r', 'r2']].to_dict('index') if not df_corr.empty else {}
        # For pearson_iqr: scatter shows only the data used in the correlation (no outliers)
        _df_scatter = df_for_corr if _pearson_iqr else df
        df_lst_valid = _df_scatter.dropna(subset=['LST_30m'])

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))
        if n_indices == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        for i, idx in enumerate(indices_list):
            idx_upper = idx.upper()
            if idx_upper in _df_scatter.columns and idx_upper in corr_lookup:
                df_clean = df_lst_valid[[idx_upper, 'LST_30m']].dropna()
                if len(df_clean) > 10:
                    ax = axes[i]
                    ax.scatter(df_clean[idx_upper], df_clean['LST_30m'],
                              alpha=0.4, s=15, c='steelblue', edgecolors='none')
                    # Trend line:
                    #   Spearman → OLS on ranks interpolated back to original scale
                    #   Pearson / Pearson IQR → OLS on values (IQR already filtered outliers)
                    if _is_spearman:
                        x_r = df_clean[idx_upper].rank()
                        y_r = df_clean['LST_30m'].rank()
                        z = np.polyfit(x_r, y_r, 1)
                        p_rank = np.poly1d(z)
                        x_rank_line = np.linspace(x_r.min(), x_r.max(), 100)
                        y_rank_line = p_rank(x_rank_line)
                        x_sorted = df_clean[idx_upper].sort_values().values
                        y_sorted = df_clean['LST_30m'].sort_values().values
                        x_orig = np.interp(x_rank_line, np.linspace(1, len(x_sorted), len(x_sorted)), x_sorted)
                        y_orig = np.interp(y_rank_line, np.linspace(1, len(y_sorted), len(y_sorted)), y_sorted)
                        ax.plot(x_orig, y_orig, "r--", linewidth=2, alpha=0.8, label='trend (ranks)')
                    else:
                        z = np.polyfit(df_clean[idx_upper], df_clean['LST_30m'], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(df_clean[idx_upper].min(), df_clean[idx_upper].max(), 100)
                        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)

                    r  = corr_lookup[idx_upper]['r']
                    r2 = corr_lookup[idx_upper]['r2']
                    ax.set_xlabel(idx_upper, fontsize=10, fontweight='bold')
                    ax.set_ylabel('LST 30m (°C)', fontsize=10)
                    ax.set_title(f'{idx_upper} vs LST\n{_r2_sym} = {r2:.4f}', fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    # Main annotation: coefficient with correct symbol
                    _annot = f'{_r_sym} = {r:.3f}'
                    if _pearson_iqr:
                        _annot += f'\n(n={len(df_clean)}, IQR)'
                    ax.text(0.05, 0.95, _annot, transform=ax.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
        for i in range(n_indices, len(axes)):
            fig.delaxes(axes[i])
        plt.suptitle(f'Correlation: Spectral Indices vs LST ({period_name})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    return {
        'correlations': df_corr,
        'data': df,
        'data_filtered': _df_filtered_full,   # df with IQR applied (=df if not pearson_iqr)
        'correlation_matrix': corr_full if not corr_full.empty else None,
        'period': period_name,
        'corr_method': _corr_method,
        'iqr_multiplier': iqr_multiplier,
    }


def analyze_collinearity(
    df_indices_and_lst,
    correlation_threshold=0.7,
    indices_to_analyze=None,
    period_name='Period'
):
    """
    Analyzes collinearity among spectral indices and LST.

    Args:
        df_indices_and_lst (pd.DataFrame): DataFrame with LST column
                                          (named 'LST_30m') and spectral indices.
        correlation_threshold (float): Absolute threshold to identify high correlation.
                                       Default: 0.7
        indices_to_analyze (list, optional): List of index names (strings) to
                                            analyze for collinearity with each other
                                            and with LST. If None, analyzes all
                                            indices present in the DataFrame (except
                                            LST_30m).
        period_name (str): Period label (e.g., 'T1', 'T2') for titles.

    Returns:
        tuple: (pd.DataFrame of the correlation matrix, list of highly collinear indices)
    """
    print("\n" + "=" * 80)
    print(f"COLLINEARITY ANALYSIS — SPECTRAL INDICES ({period_name})")
    print("=" * 80)

    # Select columns for analysis
    if indices_to_analyze is None:
        # Analyze all indices present in the DataFrame, plus LST_30m
        cols_to_analyze = [col for col in df_indices_and_lst.columns if col != 'LST_30m']
        if 'LST_30m' in df_indices_and_lst.columns:
             cols_to_analyze.append('LST_30m')
    else:
        # Analyze only the specified indices, plus LST_30m if present
        cols_to_analyze = [idx.upper() for idx in indices_to_analyze if idx.upper() in df_indices_and_lst.columns]
        if 'LST_30m' in df_indices_and_lst.columns:
             cols_to_analyze.append('LST_30m')
        else:
             print("⚠️  Column 'LST_30m' not found in DataFrame. Analyzing collinearity among indices only.")


    df_subset = df_indices_and_lst[cols_to_analyze].dropna()

    if df_subset.empty:
        print("\n⚠️  Empty DataFrame after selection and NaN removal!")
        return None, []

    # Compute the correlation matrix
    correlation_matrix = df_subset.corr()

    print("\n📊 Correlation Matrix:")
    display(correlation_matrix)

    # Visualize correlation matrix as a heatmap
    print("\n🌡️  Correlation Matrix Heatmap:")
    hm_width = max(6.0, len(cols_to_analyze)*0.8)
    hm_height = max(5.0, len(cols_to_analyze)*0.7)
    plt.figure(figsize=(hm_width, hm_height))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation Matrix ({period_name})', fontsize=14, fontweight='bold')
    plt.show()

    # Identify pairs with high correlation (> threshold or < -threshold, excluding diagonal)
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []

    print("\n" + "=" * 80)
    print(f"🔎 Highly correlated pairs (abs > {correlation_threshold:.2f}):")
    print("=" * 80)

    found_high_corr = False
    for i in range(len(upper.columns)):
        for j in range(i+1, len(upper.columns)):
            if abs(upper.iloc[i, j]) > correlation_threshold:
                high_corr_pairs.append(
                    (upper.columns[i], upper.index[j], upper.iloc[i, j])
                )
                print(f"   - {upper.columns[i]} and {upper.index[j]}: {upper.iloc[i, j]:.2f}")
                found_high_corr = True

    if not found_high_corr:
         print(f"\n✅ No high correlation (abs > {correlation_threshold:.2f}) found among the selected indices.")

    print("\n💡 Considerations for regression models (e.g., MLR):")
    print("   - Highly correlated indices provide similar information.")
    print("   - Using both in a regression model may cause multicollinearity issues.")
    print("   - It is advisable to keep only one index from each highly correlated pair.")
    print("   - The choice of which index to retain depends on its theoretical relevance to the dependent variable (LST) and its individual correlation with it.")
    if 'LST_30m' in cols_to_analyze:
         print("   - Prioritize indices with high correlation with LST_30m and low correlation with other predictors.")


    # Return the correlation matrix and the list of highly correlated pairs
    return correlation_matrix, high_corr_pairs


def rank_predictors_by_importance(
    correlation_result,
    collinearity_threshold=0.7,
    top_k=None,
    period_name='Period',
    only_selected=False,
    seed_indices=None,
    exclude_indices=None,
):
    """
    Ranks predictors by R² with LST and selects the best non-collinear subset
    using greedy forward selection.

    Algorithm:
    1. Sort all predictors by R² (descending).
    2. Select the first one (highest R²).
    3. For each subsequent: add it if |r| < threshold with ALL already-selected.
    4. Flag excluded predictors and reason (which already-selected predictor conflicts).

    Args:
        correlation_result (dict): Output of analyze_spectral_indices_correlation.
        collinearity_threshold (float): |r| threshold for collinearity exclusion.
        top_k (int, optional): Show only the top K in the R² table.
        period_name (str): Period label for titles.
        only_selected (bool): If True, print only selected-status indices (hide
            those excluded for collinearity). Default False.
        seed_indices (list, optional): List of indices (e.g., ['NDVI', 'NDBI']) to
            pre-select. When provided, those indices are forced into the selection
            first; complementary non-collinear indices are then drawn from the full
            pool. When None (default), all available predictors are used.
        exclude_indices (list, optional): List of indices to discard from the
            collinearity analysis (e.g., ['MNDWI', 'EVI']). Removed from the pool
            before greedy selection — appear in the table as "manually excluded".
            Useful when the user already knows redundant or irrelevant indices.
            Default None (no manual exclusion).

    Returns:
        pd.DataFrame: Table of selected (non-collinear) predictors.
    """
    import pandas as pd

    if correlation_result is None:
        print("⚠️  No correlation results to rank.")
        return None

    df_corr = correlation_result.get('correlations')
    _corr_method = correlation_result.get('corr_method', 'pearson')
    _iqr_multiplier = correlation_result.get('iqr_multiplier', 1.5)

    # Use IQR-filtered data when available — ensures consistency with 4.D
    df_data = correlation_result.get('data_filtered')
    if df_data is None:
        df_data = correlation_result.get('data')

    if df_corr is None or df_corr.empty or df_data is None:
        print("⚠️  Insufficient data for collinearity analysis.")
        return None

    _is_spearman    = _corr_method == 'spearman'
    _pearson_iqr    = _corr_method == 'pearson_iqr'
    _corr_method_pd = 'pearson' if _pearson_iqr else _corr_method
    _r_sym  = 'ρ' if _is_spearman else ('r_IQR' if _pearson_iqr else 'r')
    _r2_sym = 'ρ²' if _is_spearman else ('R²_IQR' if _pearson_iqr else 'R²')

    # Available predictors (exclude LST and UHI)
    exclude_cols = {'LST_30m', 'UHI', 'system:index', '.geo'}
    pred_cols = [c for c in df_data.columns if c not in exclude_cols and c in df_corr['indice'].values]

    # Resolve seed_indices: pre-select and search for complements in the full pool
    seed_upper = []
    seeds_forced = []
    if seed_indices is not None:
        seed_upper = [s.upper() for s in seed_indices]
        seeds_forced = [s for s in seed_upper if s in pred_cols and s in df_data.columns]
        if not seeds_forced:
            print(f"⚠️  None of the seed_indices {seed_indices} found in available predictors.")
            return None
        print(f"   🌱 seed_indices: {seeds_forced} pre-selected. "
              f"Searching for non-collinear complements in the full pool...")

    if len(pred_cols) < 2:
        print("⚠️  Fewer than 2 predictors available, collinearity does not apply.")
        return df_corr

    # Sort by descending R² — always uses the full pool
    df_sorted = df_corr.sort_values('r2', ascending=False)

    # Compute pairwise correlation matrix among predictors
    # Uses min_periods=30 to ensure statistical significance
    available = [c for c in df_sorted['indice'].values if c in pred_cols and c in df_data.columns]
    corr_matrix = df_data[available].corr(method=_corr_method_pd, min_periods=30)

    # === MANUAL EXCLUSIONS (exclude_indices) ===
    # Pre-marks user-specified indices as excluded before greedy selection.
    exclude_upper = [e.upper() for e in exclude_indices] if exclude_indices else []
    manually_excluded_set = set(e for e in exclude_upper if e in available)
    if manually_excluded_set:
        print(f"   Manually excluded: {sorted(manually_excluded_set)}")

    # === GREEDY SELECTION ===
    # If seed_indices provided: pre-populate with seeds (without checking collinearity among them)
    selected = list(seeds_forced)  # [] quando seed_indices is None
    excluded = {}                  # {preditor: (nome_conflito, r_conflito)}

    for _, row in df_sorted.iterrows():
        idx = row['indice']
        if idx not in corr_matrix.columns:
            continue
        # Skip seeds: already pre-selected
        if idx in seed_upper:
            continue
        # Skip manual exclusions: do not participate in selection
        if idx in manually_excluded_set:
            continue

        # Check collinearity against all already-selected predictors
        conflict = None
        for sel in selected:
            if sel in corr_matrix.columns:
                r_val = abs(corr_matrix.loc[idx, sel])
                if r_val > collinearity_threshold:
                    conflict = (sel, r_val)
                    break

        if conflict is None:
            selected.append(idx)
        else:
            excluded[idx] = conflict

    # === FINAL VALIDATION: remove residual collinear predictors ===
    # Recomputes correlation only among selected predictors (more precise).
    # Seeds are never removed in validation (forced by the user).
    changed = True
    removed_in_validation = []
    while changed:
        changed = False
        if len(selected) < 2:
            break
        sel_cols = [s for s in selected if s in df_data.columns]
        sel_corr_val = df_data[sel_cols].corr(method=_corr_method_pd, min_periods=30)
        for i in range(len(sel_cols)):
            for j in range(i+1, len(sel_cols)):
                # Skip pair where both are seeds
                if sel_cols[i] in seed_upper and sel_cols[j] in seed_upper:
                    continue
                rv = sel_corr_val.iloc[i, j]
                if abs(rv) > collinearity_threshold:
                    # Seeds are never removed — drop the non-seed with lower R²
                    i_is_seed = sel_cols[i] in seed_upper
                    j_is_seed = sel_cols[j] in seed_upper
                    if j_is_seed:
                        to_remove, kept = sel_cols[i], sel_cols[j]
                    elif i_is_seed:
                        to_remove, kept = sel_cols[j], sel_cols[i]
                    else:
                        r2_i = df_corr[df_corr['indice']==sel_cols[i]]['r2'].values[0]
                        r2_j = df_corr[df_corr['indice']==sel_cols[j]]['r2'].values[0]
                        to_remove = sel_cols[j] if r2_j < r2_i else sel_cols[i]
                        kept = sel_cols[i] if to_remove == sel_cols[j] else sel_cols[j]
                    selected.remove(to_remove)
                    excluded[to_remove] = (kept, abs(rv))
                    removed_in_validation.append((to_remove, kept, abs(rv)))
                    changed = True
                    break
            if changed:
                break

    # === PRINT RESULTS ===
    print("\n" + "=" * 80)
    print(f"📊 PREDICTOR SELECTION — GREEDY FORWARD SELECTION ({period_name})")
    print("=" * 80)
    if _is_spearman:
        _method_label_rank = 'Spearman — ρ'
    elif _pearson_iqr:
        _method_label_rank = f'Pearson w/ IQR filter ({_iqr_multiplier}×IQR) — r_IQR'
    else:
        _method_label_rank = 'Pearson — r'
    print(f"\n   Collinearity threshold: |{_r_sym}| > {collinearity_threshold:.2f}  ({_method_label_rank})")
    print(f"   Predictors analyzed: {len(available)} | Selected: {len(selected)}\n")

    # Selection table
    _col_r_hdr  = f'{_r_sym} (LST)'
    _col_r2_hdr = _r2_sym
    header = f"{'Rank':<5} {'Predictor':<16} {_col_r_hdr:<10} {_col_r2_hdr:<10} {'Domain':<13} {'Status'}"
    print(header)
    print("-" * len(header))

    rank = 1
    shown = 0
    max_show = top_k if top_k else len(selected)  # quando only_selected, limitar pelos selecionados
    for _, row in df_sorted.iterrows():
        idx = row['indice']
        # Quando only_selected: only_selected conta apenas selecionados contra max_show
        if shown >= max_show:
            break

        r_val = row['r']
        r2_val = row['r2']
        dominio = row.get('dominio', '')

        if idx in selected:
            marker = "🌱✅" if idx in seed_upper else "✅ selected"
            print(f"{rank:<5} {idx:<16} {r_val:<+10.4f} {r2_val:<10.4f} {dominio:<13} {marker}")
            rank += 1
            shown += 1  # conta apenas selecionados
        elif idx in manually_excluded_set and not only_selected:
            print(f"{'--':<5} {idx:<16} {r_val:<+10.4f} {r2_val:<10.4f} {dominio:<13} 🚫 manually excluded")
            shown += 1
        elif idx in excluded and not only_selected:
            exc = excluded.get(idx)
            if exc is not None:
                conflict_name, conflict_r = exc
                print(f"{'--':<5} {idx:<16} {r_val:<+10.4f} {r2_val:<10.4f} {dominio:<13} ❌ {_r_sym}={conflict_r:.2f} com {conflict_name}")
                shown += 1
        # when only_selected and idx excluded: does not count, not printed → loop continues

    remaining_selected = len(selected) - (rank - 1)
    if only_selected:
        if remaining_selected > 0:
            print(f"      ... and {remaining_selected} more selected index/indices hidden (top_k={max_show})")
        if len(excluded) > 0:
            print(f"      ({len(excluded)} index/indices excluded for collinearity — only_selected=True)")
    else:
        remaining = len(df_sorted) - shown
        if remaining > 0:
            print(f"      ... and {remaining} more indices hidden")

    # Summary of selected predictors
    selected_data = []
    for sel in selected:
        row = df_corr[df_corr['indice'] == sel]
        if not row.empty:
            selected_data.append({
                'Preditor': sel,
                _r_sym: row['r'].values[0],
                _r2_sym: row['r2'].values[0],
            })
    df_selected = pd.DataFrame(selected_data)

    # Show collinearity among selected predictors (using pairwise correlation)
    if len(selected) >= 2:
        sel_cols_final = [s for s in selected if s in df_data.columns]
        sel_corr = df_data[sel_cols_final].corr(method=_corr_method_pd, min_periods=30)
        print(f"\n🔎 Collinearity among the {len(selected)} selected predictors (should be < {collinearity_threshold:.2f}):")

        # Show removals made during final validation
        if removed_in_validation:
            print(f"   ℹ️  {len(removed_in_validation)} predictor(s) removed in final validation:")
            for rem_name, kept_name, rem_r in removed_in_validation:
                print(f"      ❌ {rem_name} removed ({_r_sym}={rem_r:.3f} with {kept_name})")

        any_high = False
        for i in range(len(sel_corr.columns)):
            for j in range(i+1, len(sel_corr.columns)):
                rv = sel_corr.iloc[i, j]
                if abs(rv) > collinearity_threshold * 0.8:  # alerta precoce a 80% do threshold
                    marker = "⚠️" if abs(rv) > collinearity_threshold else "🔶"
                    print(f"   {marker} {sel_corr.columns[i]} ↔ {sel_corr.columns[j]}: {_r_sym}={rv:.3f}")
                    any_high = True
        if not any_high:
            print(f"   ✅ All pairs below {collinearity_threshold:.2f}")

    return df_selected


def calculate_spectral_indices_from_raster(
    raster_path,
    indices=['NDVI', 'NDBI', 'NDWI'],
    band_mapping=None,
    nodata_value=None
):
    """
    Computes spectral indices from a multispectral raster image.
    Flexible version that supports any spectral index.

    Args:
        raster_path (str): Path to the raster file with spectral bands.
        indices (list): List of indices to compute (strings).
        band_mapping (dict, optional): Custom band mapping.
            Example: {'B2': 1, 'B3': 2, 'B4': 3, 'B8': 4, 'B11': 5, 'B12': 6}
        nodata_value (float, optional): Custom nodata value.

    Returns:
        dict: Dictionary with computed index arrays and metadata.

    Default Sentinel-2 band layout:
        Band 1: Blue (B2)
        Band 2: Green (B3)
        Band 3: Red (B4)
        Band 4: NIR (B8)
        Band 5: SWIR1 (B11)
        Band 6: SWIR2 (B12)

    Available indices:
        Vegetation: NDVI, EVI, SAVI
        Water: NDWI, MNDWI
        Soil: BSI, NDTI, DBSI
        Urban: NDBI, NBAI, NBI, UI, IBI
        Combinations: URBAN_ARID
    """
    print("\n" + "="*80)
    print("COMPUTING SPECTRAL INDICES")
    print("="*80)

    # Default Sentinel-2 band mapping
    if band_mapping is None:
        band_mapping = {
            'B2': 1,   # Blue
            'B3': 2,   # Green
            'B4': 3,   # Red
            'B8': 4,   # NIR
            'B11': 5,  # SWIR1
            'B12': 6   # SWIR2
        }

    results = {}

    with rasterio.open(raster_path) as src:
        print(f"\n📂 Reading bands from: {raster_path}")
        print(f"   Dimensions: {src.width} x {src.height}")
        print(f"   CRS: {src.crs}")
        print(f"   Available bands: {src.count}")

        # Read required bands
        bands = {}
        for band_name, band_idx in band_mapping.items():
            if band_idx <= src.count:
                bands[band_name] = src.read(band_idx).astype(float)
            else:
                print(f"⚠️  Warning: Band {band_name} (index {band_idx}) not available")

        # Aliases for easier computation
        B = bands.get('B2', None)      # Blue
        G = bands.get('B3', None)      # Green
        R = bands.get('B4', None)      # Red
        N = bands.get('B8', None)      # NIR
        SWIR1 = bands.get('B11', None) # SWIR1
        SWIR2 = bands.get('B12', None) # SWIR2

        # Define nodata value
        nodata = nodata_value if nodata_value is not None else (src.nodata if src.nodata is not None else 0)

        # Create valid-data mask
        valid_mask = np.ones_like(R, dtype=bool) if R is not None else np.ones((src.height, src.width), dtype=bool)
        if R is not None:
            valid_mask &= (R != nodata)
        if N is not None:
            valid_mask &= (N != nodata)
        if SWIR1 is not None:
            valid_mask &= (SWIR1 != nodata)

        # Metadata
        results['metadata'] = {
            'shape': (src.height, src.width),
            'crs': src.crs,
            'transform': src.transform,
            'nodata': nodata,
            'bands_available': list(bands.keys())
        }

        # Soil correction factor (SAVI)
        L = 0.5

        print("\n📊 Computing indices:")

        # VEGETATION INDICES
        if 'NDVI' in indices and R is not None and N is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = np.where(valid_mask, (N - R) / (N + R + 1e-10), np.nan)
            results['NDVI'] = ndvi
            print(f"   ✓ NDVI calculado - Range: [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")

        if 'EVI' in indices and B is not None and R is not None and N is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                evi = np.where(valid_mask,
                              2.5 * (N - R) / (N + 6*R - 7.5*B + 1 + 1e-10),
                              np.nan)
            results['EVI'] = evi
            print(f"   ✓ EVI calculado - Range: [{np.nanmin(evi):.3f}, {np.nanmax(evi):.3f}]")

        if 'SAVI' in indices and R is not None and N is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                savi = np.where(valid_mask,
                               ((N - R) / (N + R + L + 1e-10)) * (1 + L),
                               np.nan)
            results['SAVI'] = savi
            print(f"   ✓ SAVI calculado - Range: [{np.nanmin(savi):.3f}, {np.nanmax(savi):.3f}]")

        # WATER INDICES
        if 'NDWI' in indices and G is not None and N is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ndwi = np.where(valid_mask, (G - N) / (G + N + 1e-10), np.nan)
            results['NDWI'] = ndwi
            print(f"   ✓ NDWI calculado - Range: [{np.nanmin(ndwi):.3f}, {np.nanmax(ndwi):.3f}]")

        if 'MNDWI' in indices and G is not None and SWIR1 is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                mndwi = np.where(valid_mask, (G - SWIR1) / (G + SWIR1 + 1e-10), np.nan)
            results['MNDWI'] = mndwi
            print(f"   ✓ MNDWI calculado - Range: [{np.nanmin(mndwi):.3f}, {np.nanmax(mndwi):.3f}]")

        # SOIL INDICES
        if 'BSI' in indices and B is not None and R is not None and N is not None and SWIR1 is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                bsi = np.where(valid_mask,
                              ((SWIR1 + R) - (N + B)) / ((SWIR1 + R) + (N + B) + 1e-10),
                              np.nan)
            results['BSI'] = bsi
            print(f"   ✓ BSI calculado - Range: [{np.nanmin(bsi):.3f}, {np.nanmax(bsi):.3f}]")

        if 'NDTI' in indices and SWIR1 is not None and SWIR2 is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ndti = np.where(valid_mask,
                               (SWIR1 - SWIR2) / (SWIR1 + SWIR2 + 1e-10),
                               np.nan)
            results['NDTI'] = ndti
            print(f"   ✓ NDTI calculado - Range: [{np.nanmin(ndti):.3f}, {np.nanmax(ndti):.3f}]")

        if 'DBSI' in indices and G is not None and R is not None and N is not None and SWIR1 is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                dbsi = np.where(valid_mask,
                               ((SWIR1 - G) / (SWIR1 + G + 1e-10)) - ((N - R) / (N + R + 1e-10)),
                               np.nan)
            results['DBSI'] = dbsi
            print(f"   ✓ DBSI calculado - Range: [{np.nanmin(dbsi):.3f}, {np.nanmax(dbsi):.3f}]")

        # URBAN INDICES
        if 'NDBI' in indices and N is not None and SWIR1 is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ndbi = np.where(valid_mask, (SWIR1 - N) / (SWIR1 + N + 1e-10), np.nan)
            results['NDBI'] = ndbi
            print(f"   ✓ NDBI calculado - Range: [{np.nanmin(ndbi):.3f}, {np.nanmax(ndbi):.3f}]")

        if 'NBAI' in indices and G is not None and SWIR1 is not None and SWIR2 is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                nbai = np.where(valid_mask,
                               (SWIR2 - (SWIR1 / (G + 1e-10))) / (SWIR2 + (SWIR1 / (G + 1e-10)) + 1e-10),
                               np.nan)
            results['NBAI'] = nbai
            print(f"   ✓ NBAI calculado - Range: [{np.nanmin(nbai):.3f}, {np.nanmax(nbai):.3f}]")

        if 'NBI' in indices and R is not None and N is not None and SWIR2 is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                nbi = np.where(valid_mask, (R * SWIR2) / (N + 1e-10), np.nan)
            results['NBI'] = nbi
            print(f"   ✓ NBI calculado - Range: [{np.nanmin(nbi):.3f}, {np.nanmax(nbi):.3f}]")

        if 'UI' in indices and R is not None and N is not None and SWIR1 is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ui = np.where(valid_mask,
                             (SWIR1 + R - N) / (SWIR1 + R + N + 1e-10),
                             np.nan)
            results['UI'] = ui
            print(f"   ✓ UI calculado - Range: [{np.nanmin(ui):.3f}, {np.nanmax(ui):.3f}]")

        # COMPOSITE INDICES
        if 'IBI' in indices:
            ndbi_calc = results.get('NDBI')
            savi_calc = results.get('SAVI')
            mndwi_calc = results.get('MNDWI')

            if ndbi_calc is None and N is not None and SWIR1 is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndbi_calc = np.where(valid_mask, (SWIR1 - N) / (SWIR1 + N + 1e-10), np.nan)

            if savi_calc is None and R is not None and N is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    savi_calc = np.where(valid_mask,
                                        ((N - R) / (N + R + L + 1e-10)) * (1 + L),
                                        np.nan)

            if mndwi_calc is None and G is not None and SWIR1 is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    mndwi_calc = np.where(valid_mask, (G - SWIR1) / (G + SWIR1 + 1e-10), np.nan)

            if ndbi_calc is not None and savi_calc is not None and mndwi_calc is not None:
                ndbi_norm = ndbi_calc + 1
                savi_norm = savi_calc + 1
                mndwi_norm = mndwi_calc + 1

                with np.errstate(divide='ignore', invalid='ignore'):
                    ibi = np.where(valid_mask,
                                  (ndbi_norm - ((savi_norm + mndwi_norm) / 2)) /
                                  (ndbi_norm + ((savi_norm + mndwi_norm) / 2) + 1e-10),
                                  np.nan)
                results['IBI'] = ibi
                print(f"   ✓ IBI calculado - Range: [{np.nanmin(ibi):.3f}, {np.nanmax(ibi):.3f}]")

        if 'URBAN_ARID' in indices:
            ndti_calc = results.get('NDTI')
            bsi_calc = results.get('BSI')
            ndvi_calc = results.get('NDVI')

            if ndti_calc is None and SWIR1 is not None and SWIR2 is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndti_calc = np.where(valid_mask,
                                        (SWIR1 - SWIR2) / (SWIR1 + SWIR2 + 1e-10),
                                        np.nan)

            if bsi_calc is None and B is not None and R is not None and N is not None and SWIR1 is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    bsi_calc = np.where(valid_mask,
                                       ((SWIR1 + R) - (N + B)) / ((SWIR1 + R) + (N + B) + 1e-10),
                                       np.nan)

            if ndvi_calc is None and R is not None and N is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi_calc = np.where(valid_mask, (N - R) / (N + R + 1e-10), np.nan)

            if ndti_calc is not None and bsi_calc is not None and ndvi_calc is not None:
                urban_arid = ndti_calc + bsi_calc + ndvi_calc
                results['URBAN_ARID'] = urban_arid
                print(f"   ✓ URBAN_ARID calculado - Range: [{np.nanmin(urban_arid):.3f}, {np.nanmax(urban_arid):.3f}]")

    print("\n✓ Spectral indices computed successfully")
    print("="*80)

    return results



def create_threshold_mask(index_array, min_threshold=-np.inf, max_threshold=np.inf, nodata=None):
    """
    Creates a binary mask based on spectral index thresholds.

    Args:
        index_array (np.array): Array with spectral index values.
        min_threshold (float): Minimum threshold value (default: -inf, no lower bound).
        max_threshold (float): Maximum threshold value (default: +inf, no upper bound).
        nodata (float): Nodata value to ignore.

    Returns:
        np.array: Binary mask (True where index is within thresholds).

    Examples:
        # Healthy vegetation: NDVI between 0.5 and 1.0
        mask_healthy_veg = create_threshold_mask(ndvi, min_threshold=0.5, max_threshold=1.0)

        # Dense built-up: NDBI > 0.3
        mask_dense_urban = create_threshold_mask(ndbi, min_threshold=0.3)

        # Sparse vegetation: NDVI between 0.2 and 0.5
        mask_sparse_veg = create_threshold_mask(ndvi, min_threshold=0.2, max_threshold=0.5)
    """
    if nodata is not None:
        valid_mask = (index_array != nodata) & ~np.isnan(index_array)
    else:
        valid_mask = ~np.isnan(index_array)

    threshold_mask = valid_mask & (index_array >= min_threshold) & (index_array <= max_threshold)

    n_pixels = np.sum(threshold_mask)
    total_valid = np.sum(valid_mask)
    percent = (n_pixels / total_valid * 100) if total_valid > 0 else 0

    print(f"✓ Mask created: {n_pixels:,} pixels ({percent:.2f}% of valid area)")
    print(f"  Applied threshold: [{min_threshold:.3f}, {max_threshold:.3f}]")

    return threshold_mask



def detect_threshold_based_changes(mask_t1, mask_t2, class_name='Elemento'):
    """
    Detects changes between bitemporal masks.

    Args:
        mask_t1 (np.array): Boolean mask for period T1.
        mask_t2 (np.array): Boolean mask for period T2.
        class_name (str): Class name for the report.

    Returns:
        dict: Dictionary with change masks and statistics.

    Detected changes:
        - Gain: pixels that went from False (T1) to True (T2)
        - Loss: pixels that went from True (T1) to False (T2)
        - Stable: pixels that remained True in both periods
        - No change: pixels that remained False in both periods
    """
    print("\n" + "="*80)
    print(f"CHANGE DETECTION: {class_name}")
    print("="*80)

    # Detect changes
    mask_gain = (~mask_t1) & mask_t2
    mask_loss = mask_t1 & (~mask_t2)
    mask_stable = mask_t1 & mask_t2
    mask_no_change = (~mask_t1) & (~mask_t2)
    mask_any_change = mask_gain | mask_loss

    # Statistics
    n_gain = np.sum(mask_gain)
    n_loss = np.sum(mask_loss)
    n_stable = np.sum(mask_stable)
    n_no_change = np.sum(mask_no_change)
    n_total = mask_t1.size
    n_any_change = n_gain + n_loss

    n_t1 = np.sum(mask_t1)
    n_t2 = np.sum(mask_t2)

    print(f"\n📊 CHANGE STATISTICS — {class_name}:")
    print(f"   Area in T1: {n_t1:,} pixels ({n_t1/n_total*100:.2f}%)")
    print(f"   Area in T2: {n_t2:,} pixels ({n_t2/n_total*100:.2f}%)")
    print(f"\n   🟢 Gain (T1=No, T2=Yes): {n_gain:,} pixels ({n_gain/n_total*100:.2f}%)")
    print(f"   🔴 Loss (T1=Yes, T2=No): {n_loss:,} pixels ({n_loss/n_total*100:.2f}%)")
    print(f"   🟡 Stable (present in both): {n_stable:,} pixels ({n_stable/n_total*100:.2f}%)")
    print(f"   ⚪ No change (absent in both): {n_no_change:,} pixels ({n_no_change/n_total*100:.2f}%)")
    print(f"\n   📈 Total changes: {n_any_change:,} pixels ({n_any_change/n_total*100:.2f}%)")

    net_change = n_gain - n_loss
    if net_change > 0:
        print(f"   💡 Net balance: +{net_change:,} pixels (GAIN of {class_name})")
    elif net_change < 0:
        print(f"   💡 Net balance: {net_change:,} pixels (LOSS of {class_name})")
    else:
        print(f"   💡 Net balance: {net_change:,} pixels (EQUILIBRIUM)")

    print("="*80)

    return {
        'mask_gain': mask_gain,
        'mask_loss': mask_loss,
        'mask_stable': mask_stable,
        'mask_no_change': mask_no_change,
        'mask_any_change': mask_any_change,
        'stats': {
            'class_name': class_name,
            'n_t1': int(n_t1),
            'n_t2': int(n_t2),
            'n_gain': int(n_gain),
            'n_loss': int(n_loss),
            'n_stable': int(n_stable),
            'n_any_change': int(n_any_change),
            'percent_t1': float(n_t1/n_total*100),
            'percent_t2': float(n_t2/n_total*100),
            'percent_gain': float(n_gain/n_total*100),
            'percent_loss': float(n_loss/n_total*100),
            'percent_change': float(n_any_change/n_total*100),
            'net_change': int(net_change)
        }
    }




def export_change_maps(
    change_results,
    reference_raster_path,
    output_dir,
    class_name='Elemento',
    export_gain=True,
    export_loss=False,
    export_stable=False,
    export_no_change=False
):
    """
    Exports change maps as GeoTIFF files.

    Args:
        change_results (dict): Output of detect_threshold_based_changes().
        reference_raster_path (str): Path to reference raster (to read CRS and transform).
        output_dir (str): Directory to save the files.
        class_name (str): Class name (used in file names).
        export_gain (bool): Export gain map.
        export_loss (bool): Export loss map.
        export_stable (bool): Export stable-area map.
        export_no_change (bool): Export no-change-area map.

    Returns:
        dict: Dictionary with paths of exported files.
    """
    print("\n" + "="*80)
    print(f"EXPORTING CHANGE MAPS: {class_name}")
    print("="*80)

    # Load metadata from reference raster
    with rasterio.open(reference_raster_path) as src:
        meta = src.meta.copy()

    # Update metadata for binary maps
    meta.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 0,
        'compress': 'lzw'
    })

    exported_files = {}
    class_slug = class_name.replace(' ', '_').lower()

    # Export gain
    if export_gain:
        mask_gain = change_results['mask_gain']
        mask_gain_uint8 = np.where(mask_gain, 255, 0).astype(np.uint8)
        gain_path = f"{output_dir}mask_gain_{class_slug}.tif"

        with rasterio.open(gain_path, 'w', **meta) as dst:
            dst.write(mask_gain_uint8, 1)

        print(f"  ✓ Gain exported: {gain_path}")
        print(f"    Gain pixels: {np.sum(mask_gain):,}")
        exported_files['gain'] = gain_path

    # Export loss
    if export_loss:
        mask_loss = change_results['mask_loss']
        mask_loss_uint8 = np.where(mask_loss, 255, 0).astype(np.uint8)
        loss_path = f"{output_dir}mask_loss_{class_slug}.tif"

        with rasterio.open(loss_path, 'w', **meta) as dst:
            dst.write(mask_loss_uint8, 1)

        print(f"  ✓ Loss exported: {loss_path}")
        print(f"    Loss pixels: {np.sum(mask_loss):,}")
        exported_files['loss'] = loss_path

    # Export stable
    if export_stable:
        mask_stable = change_results['mask_stable']
        mask_stable_uint8 = np.where(mask_stable, 255, 0).astype(np.uint8)
        stable_path = f"{output_dir}mask_stable_{class_slug}.tif"

        with rasterio.open(stable_path, 'w', **meta) as dst:
            dst.write(mask_stable_uint8, 1)

        print(f"  ✓ Stable exported: {stable_path}")
        print(f"    Stable pixels: {np.sum(mask_stable):,}")
        exported_files['stable'] = stable_path

    # Export no-change
    if export_no_change:
        mask_no_change = change_results['mask_no_change']
        mask_no_change_uint8 = np.where(mask_no_change, 255, 0).astype(np.uint8)
        no_change_path = f"{output_dir}mask_no_change_{class_slug}.tif"

        with rasterio.open(no_change_path, 'w', **meta) as dst:
            dst.write(mask_no_change_uint8, 1)

        print(f"  ✓ No-change exported: {no_change_path}")
        print(f"    No-change pixels: {np.sum(mask_no_change):,}")
        exported_files['no_change'] = no_change_path

    print("\n✓ Export complete")
    print("="*80)

    return exported_files



def visualize_threshold_based_changes(
    index_t1,
    index_t2,
    change_results,
    class_name='Elemento',
    index_name='Index',
    cmap_index='RdYlGn',
    save_path=None,
):
    """
    Visualizes changes detected by custom thresholds.

    Args:
        index_t1 (np.array): Index for period T1.
        index_t2 (np.array): Index for period T2.
        change_results (dict): Output of detect_threshold_based_changes().
        class_name (str): Name of the analyzed class.
        index_name (str): Index name (NDVI, NDBI, etc.).
        cmap_index (str): Colormap for the index.
        save_path (str): Path to save the figure (optional).
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    mask_gain = change_results['mask_gain']
    mask_loss = change_results['mask_loss']
    stats = change_results['stats']

    # Create combined change map
    change_map = np.zeros_like(mask_gain, dtype=int)
    change_map[mask_gain] = 1  # Gain
    change_map[mask_loss] = 2  # Loss
    change_map[change_results['mask_stable']] = 3  # Stable

    # 1. Index T1
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(index_t1, cmap=cmap_index, vmin=-1, vmax=1)
    ax1.set_title(f'{index_name} - T1', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Index T2
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(index_t2, cmap=cmap_index, vmin=-1, vmax=1)
    ax2.set_title(f'{index_name} - T2', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Index difference
    ax3 = fig.add_subplot(gs[0, 2])
    diff = index_t2 - index_t1
    im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax3.set_title(f'Δ {index_name} (T2 - T1)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Index change', rotation=270, labelpad=20)

    # 4. Change map
    ax4 = fig.add_subplot(gs[1, :])
    colors = ['white', '#00ff00', '#ff0000', '#ffff00']
    cmap_change = plt.matplotlib.colors.ListedColormap(colors)
    im4 = ax4.imshow(change_map, cmap=cmap_change, vmin=0, vmax=3)
    ax4.set_title(f'Change Map: {class_name}', fontsize=16, fontweight='bold')
    ax4.axis('off')

    # Legend
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label=f'No change ({stats["percent_change"]:.1f}%)'),
        Patch(facecolor='#00ff00', label=f'Gain ({stats["percent_gain"]:.1f}%)'),
        Patch(facecolor='#ff0000', label=f'Loss ({stats["percent_loss"]:.1f}%)'),
        Patch(facecolor='#ffff00', label=f'Stable ({stats["n_stable"]/change_map.size*100:.1f}%)')
    ]
    ax4.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=12, frameon=True)

    # Overall title
    fig.suptitle(f'Change Analysis: {class_name}\n' +
                f'Net balance: {stats["net_change"]:+,} pixels | ' +
                f'T1: {stats["n_t1"]:,} pixels → T2: {stats["n_t2"]:,} pixels',
                fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")

    plt.tight_layout()
    plt.show()



def run_spectral_change_detection(
    sentinel_t1_path,
    sentinel_t2_path,
    index_name='NDVI',
    threshold_min=0.5,
    threshold_max=1.0,
    class_name=None,
    output_dir=None,
    band_mapping=None,
    export_gain=True,
    export_loss=False,
    export_stable=False,
    export_no_change=False,
    visualize=True
):
    """
    Complete pipeline for spectral-index-based change detection.

    Args:
        sentinel_t1_path (str): Path to Sentinel-2 T1 image.
        sentinel_t2_path (str): Path to Sentinel-2 T2 image.
        index_name (str): Index to compute.
        threshold_min (float): Minimum threshold for the spectral mask.
        threshold_max (float): Maximum threshold for the spectral mask.
        class_name (str, optional): Descriptive name for the analyzed class.
        output_dir (str, optional): Directory to save results.
        band_mapping (dict, optional): Custom band mapping.
        export_gain (bool): Export gain map.
        export_loss (bool): Export loss map.
        export_stable (bool): Export stable-area map.
        export_no_change (bool): Export no-change-area map.
        visualize (bool): Generate visualization.

    Returns:
        dict: Full analysis results.

    Example:
        results = run_spectral_change_detection(
            sentinel_t1_path='sentinel2_t1.tif',
            sentinel_t2_path='sentinel2_t2.tif',
            index_name='NDVI',
            threshold_min=0.3,
            threshold_max=1.0,
            class_name='Vegetation',
            output_dir='/output/',
            export_gain=True,
            export_loss=True,
            export_stable=False,
            export_no_change=False,
            visualize=True
        )
    """
    if class_name is None:
        class_name = f'{index_name} [{threshold_min:.2f}, {threshold_max:.2f}]'

    print("\n" + "="*80)
    print("SPECTRAL CHANGE DETECTION PIPELINE")
    print("="*80)
    print(f"Index: {index_name}")
    print(f"Class: {class_name}")
    print(f"Thresholds: [{threshold_min}, {threshold_max}]")

    # Step 1: Compute indices
    print("\n[1/5] Computing spectral indices...")
    indices_t1 = calculate_spectral_indices_from_raster(
        raster_path=sentinel_t1_path,
        indices=[index_name],
        band_mapping=band_mapping
    )

    indices_t2 = calculate_spectral_indices_from_raster(
        raster_path=sentinel_t2_path,
        indices=[index_name],
        band_mapping=band_mapping
    )

    # Step 2: Create masks
    print(f"\n[2/5] Creating masks for {class_name}...")
    print("\nPeriod T1:")
    mask_t1 = create_threshold_mask(
        index_array=indices_t1[index_name],
        min_threshold=threshold_min,
        max_threshold=threshold_max,
        nodata=indices_t1['metadata']['nodata']
    )

    print("\nPeriod T2:")
    mask_t2 = create_threshold_mask(
        index_array=indices_t2[index_name],
        min_threshold=threshold_min,
        max_threshold=threshold_max,
        nodata=indices_t2['metadata']['nodata']
    )

    # Step 3: Detect changes
    print(f"\n[3/5] Detecting changes in {class_name}...")
    changes = detect_threshold_based_changes(
        mask_t1=mask_t1,
        mask_t2=mask_t2,
        class_name=class_name
    )

    # Step 4: Export maps
    exported_files = {}
    if output_dir and (export_gain or export_loss or export_stable or export_no_change):
        print(f"\n[4/5] Exporting change maps...")
        exported_files = export_change_maps(
            change_results=changes,
            reference_raster_path=sentinel_t1_path,
            output_dir=output_dir,
            class_name=class_name,
            export_gain=export_gain,
            export_loss=export_loss,
            export_stable=export_stable,
            export_no_change=export_no_change
        )
    else:
        print(f"\n[4/5] Skipping export (output_dir not specified or no option selected)")

    # Step 5: Visualize
    if visualize:
        print(f"\n[5/5] Generating visualization...")
        save_path_viz = f'{output_dir}{class_name.replace(" ", "_")}_changes.png' if output_dir else None
        visualize_threshold_based_changes(
            index_t1=indices_t1[index_name],
            index_t2=indices_t2[index_name],
            change_results=changes,
            class_name=class_name,
            index_name=index_name,
            save_path=None #save_path_viz
        )
    else:
        print(f"\n[5/5] Visualization disabled")

    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)

    return {
        'indices_t1': indices_t1,
        'indices_t2': indices_t2,
        'masks': {
            'mask_t1': mask_t1,
            'mask_t2': mask_t2
        },
        'changes': changes,
        'exported_files': exported_files
    }

