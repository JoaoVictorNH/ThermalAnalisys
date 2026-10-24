# ==============================================================================
# spectral_analysis.py — Spectral Indices
# ==============================================================================
# Spectral indices for Landsat 8/9 and Sentinel-2: built-in set plus
# the Awesome Spectral Indices catalog (spyndex / eemont).
# ==============================================================================

import ee
import pandas as pd

# --- Optional imports for dynamic indices ---
try:
    import spyndex
    SPYNDEX_AVAILABLE = True
except ImportError:
    SPYNDEX_AVAILABLE = False
    print("⚠️  spyndex not installed. Run: pip install spyndex")


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
        import eemont  # noqa: F401  (registers ee.Image.spectralIndices)
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


