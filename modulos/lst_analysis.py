# ==============================================================================
# lst_analysis.py — Land Surface Temperature Analysis (LST, UHI)
# ==============================================================================
# UHI calculation functions, intensity classification,
# statistics, and LST processing for a single period.
# ==============================================================================

import ee
import numpy as np

# --- Module imports ---
from modulos.image_processing import perform_bandpass_adjustment, process_landsat_median, process_sentinel2_median




def calculate_uhi(lst_image, geometry, scale=30):
    """
    Calculates the normalized Urban Heat Island (UHI) index.
    """
    # Calculate mean and standard deviation in a single server call
    mean_std = lst_image.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=geometry,
        scale=scale,
        maxPixels=1e9
    )
    band_name = lst_image.bandNames().get(0)
    lst_mean = mean_std.get(ee.String(band_name).cat('_mean'))
    lst_std = mean_std.get(ee.String(band_name).cat('_stdDev'))

    # Convert to ee.Number and add protection
    lst_mean = ee.Number(lst_mean)
    lst_std = ee.Number(lst_std).max(1e-9)

    # Calculate normalized UHI
    uhi = lst_image.subtract(lst_mean).divide(lst_std).rename('UHI')

    stats = {
        'lst_mean': lst_mean,
        'lst_std': lst_std
    }

    return uhi, stats



def classify_uhi_intensity(uhi_normalized):
    """
    Classifies Urban Heat Island (UHI) intensity according to the
    methodology of Deng et al. (2023, Ecological Indicators).

    Classification based on mean and standard deviation method:
    (μ = LST mean, σ = LST standard deviation)
        5 - High Temperature Zone (HTZ):       Ts > μ + σ
        4 - Sub-High Temperature Zone (SHTZ):  μ + 0.5σ < Ts ≤ μ + σ
        3 - Medium Temperature Zone (MTZ):     μ - 0.5σ ≤ Ts ≤ μ + 0.5σ
        2 - Sub-Low Temperature Zone (SLTZ):   μ - σ ≤ Ts < μ - 0.5σ
        1 - Low Temperature Zone (LTZ):        Ts < μ - σ

    Args:
        uhi_normalized (ee.Image): Normalized UHI (z-score)

    Returns:
        ee.Image: UHI intensity classification (1-5)
    """
    # Apply classification based on z-score intervals
    # Since UHI is already normalized (z-score), μ=0 and σ=1
    uhi_class = (
        uhi_normalized
        .where(uhi_normalized.lt(-1), 1)                    # LTZ: < -1σ
        .where(uhi_normalized.gte(-1).And(uhi_normalized.lt(-0.5)), 2)  # SLTZ: -1σ to -0.5σ
        .where(uhi_normalized.gte(-0.5).And(uhi_normalized.lte(0.5)), 3)  # MTZ: -0.5σ to 0.5σ
        .where(uhi_normalized.gt(0.5).And(uhi_normalized.lte(1)), 4)    # SHTZ: 0.5σ to 1σ
        .where(uhi_normalized.gt(1), 5)                     # HTZ: > 1σ
    )

    return uhi_class.rename('uhi_intensity')


def calculate_uhi_statistics(uhi_image, geometry, scale=30):
    """
    Calculates detailed UHI index statistics.

    Args:
        uhi_image (ee.Image): Image with UHI values
        geometry (ee.Geometry): Region of interest geometry
        scale (int): Computation scale in meters (default: 30)

    Returns:
        dict: Dictionary with complete statistics
    """
    combined = uhi_image.rename('UHI')

    # Calculate statistics
    stats = combined.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.minMax(),
            sharedInputs=True
        ).combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ).combine(
            reducer2=ee.Reducer.percentile([25, 50, 75]),
            sharedInputs=True
        ),
        geometry=geometry,
        scale=scale,
        maxPixels=1e9
    )

    # Get stats with default values to prevent errors if keys are missing
    stats_info = stats.getInfo()

    # Create a new dictionary with default values
    safe_stats = {}
    for key in stats_info:
        safe_stats[key] = stats_info[key]

    # Ensure expected UHI keys are present, even if they are missing in the original stats
    expected_keys = [
        'UHI_mean', 'UHI_min', 'UHI_max', 'UHI_stdDev', 'UHI_p25', 'UHI_p50', 'UHI_p75'
    ]
    for key in expected_keys:
        if key not in safe_stats:
            safe_stats[key] = None # Or a suitable default like 0 or np.nan

    return safe_stats



def classify_uhi_areas(uhi_image, geometry, scale=30):
    """
    Classifies areas by UHI intensity according to Deng et al. (2023) and computes area in km².

    Classification based on mean and standard deviation method:
        5 - High Temperature Zone (HTZ):       Ts > μ + σ (UHI > 1)
        4 - Sub-High Temperature Zone (SHTZ):  μ + 0.5σ < Ts ≤ μ + σ (0.5 < UHI ≤ 1)
        3 - Medium Temperature Zone (MTZ):     μ - 0.5σ ≤ Ts ≤ μ + 0.5σ (-0.5 ≤ UHI ≤ 0.5)
        2 - Sub-Low Temperature Zone (SLTZ):   μ - σ ≤ Ts < μ - 0.5σ (-1 ≤ UHI < -0.5)
        1 - Low Temperature Zone (LTZ):        Ts < μ - σ (UHI < -1)

    Args:
        uhi_image: Normalized UHI image (z-score)
        geometry: Region geometry
        scale: Scale in meters

    Returns:
        dict: Areas in km² for each category
    """
    # Create masks for each category following Deng et al. (2023) and rename them
    ltz = uhi_image.lt(-1).rename('LTZ')                                          # Class 1: LTZ (UHI < -1)
    sltz = uhi_image.gte(-1).And(uhi_image.lt(-0.5)).rename('SLTZ')               # Class 2: SLTZ (-1 ≤ UHI < -0.5)
    mtz = uhi_image.gte(-0.5).And(uhi_image.lte(0.5)).rename('MTZ')               # Class 3: MTZ (-0.5 ≤ UHI ≤ 0.5)
    shtz = uhi_image.gt(0.5).And(uhi_image.lte(1)).rename('SHTZ')                 # Class 4: SHTZ (0.5 < UHI ≤ 1)
    htz = uhi_image.gt(1).rename('HTZ')                                           # Class 5: HTZ (UHI > 1)

    # Combine all masks into a single multi-band image
    combined_masks = ee.Image.cat([ltz, sltz, mtz, shtz, htz])
    pixel_area = ee.Image.pixelArea()

    # Reduce all areas simultaneously (single request)
    areas_result = combined_masks.multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e12
    ).getInfo()

    # Extract and convert to km²
    def get_area(key):
        val = areas_result.get(key)
        return float(val) / 1e6 if val is not None else 0.0

    area_ltz = get_area('LTZ')
    area_sltz = get_area('SLTZ')
    area_mtz = get_area('MTZ')
    area_shtz = get_area('SHTZ')
    area_htz = get_area('HTZ')

    return {
        'LTZ': area_ltz,           # Low temperature
        'SLTZ': area_sltz,         # Sub-low temperature
        'MTZ': area_mtz,           # Medium temperature
        'SHTZ': area_shtz,         # Sub-high temperature
        'HTZ': area_htz,           # High temperature
        'total': area_ltz + area_sltz + area_mtz + area_shtz + area_htz,
    }


def perform_linear_regression_mlr(lst, roi, scale=30, **spectral_indices):
    """
    Performs multiple linear regression between LST and spectral indices.
    Based on the MLR method from cd_thermalanalysis_v1_2.py

    Args:
        lst: LST image (ee.Image)
        roi: Region of interest (ee.Geometry)
        scale: Processing scale (default: 30m)
        **spectral_indices: Spectral indices as keyword arguments
                           (e.g., ndvi=ndvi_image, ndbi=ndbi_image)

    Returns:
        dict: Regression coefficients {'intercept': value, 'ndvi': value, ...}
    """
    # Create list of input bands starting with constant band
    input_bands = [ee.Image(1).rename('constant')]
    band_names = ['constant']

    # Add spectral index bands provided
    for name, img in spectral_indices.items():
        input_bands.append(img.rename(name))
        band_names.append(name)

    # Add LST band as dependent variable
    input_bands.append(lst.rename('lst'))
    band_names.append('lst')

    # Stack bands into a single image
    bands = ee.Image.cat(input_bands).select(band_names)

    # Define number of independent variables (constant + spectral indices)
    num_x = len(spectral_indices) + 1  # +1 for the constant band

    # Run multiple linear regression
    regression = bands.reduceRegion(
        reducer=ee.Reducer.linearRegression(numX=num_x, numY=1),
        geometry=roi,
        scale=scale,
        maxPixels=1e9,
        tileScale=16
    )

    try:
        # Extract coefficients
        coef_list = ee.Array(regression.get('coefficients')).toList()

        coefficients = {'intercept': ee.Number(ee.List(coef_list.get(0)).get(0)).getInfo()}

        # Extract coefficients for spectral indices
        for i, name in enumerate(spectral_indices.keys()):
            coefficients[name] = ee.Number(ee.List(coef_list.get(i + 1)).get(0)).getInfo()

        return coefficients
    except Exception as e:
        error_msg = str(e)
        if "values' is required and may not be null" in error_msg or "null" in error_msg.lower():
            raise ValueError(
                f"MLR regression error: Null coefficients found. This usually occurs "
                f"due to perfect multicollinearity between the chosen indices (independent variables) "
                f"(e.g., using NDVI, MNDWI, and DBSI together). Check the index combinations."
            ) from e
        raise e



def calculate_modeled_lst_mlr(coefficients, **spectral_indices):
    """
    Calculates modeled LST using MLR regression coefficients.

    Args:
        coefficients: Dictionary with regression coefficients
        **spectral_indices: Spectral index images as keyword arguments

    Returns:
        ee.Image: Modeled LST
    """
    modeled_lst = ee.Image(coefficients['intercept'])

    for name, img in spectral_indices.items():
        modeled_lst = modeled_lst.add(ee.Image(coefficients[name]).multiply(img))

    return modeled_lst



def calculate_10m_lst(config, s2_median, s2_projection, l8_optical_bands,
                      lst_celsius, scale_agg=250, mlr_list_ind=['ndvi', 'ndbi'],
                      methods=['RLS'], water_mask=None,
                      rf_n_trees=50, gb_n_trees=50, num_samples=5000):
    """
    Calculates 10m LST using multiple methods.
    Full implementation as per the original JavaScript code.

    Uses:
    - process_sentinel2_median()
    - process_landsat_median()
    - Keeps standard Landsat band names (SR_B2, SR_B3, etc.)

    Args:
        config: Configuration dictionary
        s2_median: Sentinel-2 median image
        s2_projection: Sentinel-2 projection
        l8_optical_bands: Landsat 8 optical bands
        lst_celsius: LST in Celsius (30m)
        scale_agg: Aggregation scale for validation (default: 250m)
        mlr_list_ind: List of indices for MLR
        methods: List of methods to apply

    Returns:
        dict: Results with downscaled LST and accuracy metrics
    """
    from modulos.downscaling import (calculate_accuracy_metrics, method_mlr, method_ols,
                                     method_rls, method_random_forest, method_gradient_boosting)
    print("\n" + "=" * 80)
    print("STARTING PROCESSING")
    print("=" * 80)

    adjusted_s2 = perform_bandpass_adjustment(
        s2_median, l8_optical_bands, config['geometry']
    )

    # Apply selected methods
    results = {
        'config': config,
        's2_median': s2_median,
        'l8_median': l8_optical_bands,
        'lst_30m': lst_celsius,
        'adjusted_s2': adjusted_s2,
        's2_projection': s2_projection,
        'methods': {}
    }

    # Variable to store agg_LST (used by multiple methods)
    agg_LST = lst_celsius.reproject(crs=s2_projection, scale=scale_agg)

    if 'RLS' in methods:
        lst_rls = method_rls(l8_optical_bands, adjusted_s2, lst_celsius, agg_LST, config['geometry'], s2_projection)
        accuracy_rls = calculate_accuracy_metrics(lst_rls, agg_LST, scale_agg, config['geometry'], 'RLS')
        results['methods']['RLS'] = {
            'lst': lst_rls,
            'accuracy': accuracy_rls
        }
        print(f"   ✓ RLS completed")

    if 'OLS' in methods:
        lst_ols, agg_LST_ols = method_ols(l8_optical_bands, adjusted_s2, lst_celsius, agg_LST, config['geometry'], s2_projection)
        accuracy_ols = calculate_accuracy_metrics(lst_ols, agg_LST_ols, scale_agg, config['geometry'], 'OLS')
        results['methods']['OLS'] = {
            'lst': lst_ols,
            'accuracy': accuracy_ols
        }
        print(f"   ✓ OLS completed")

    if 'MLR' in methods:
        lst_mlr = method_mlr(
            l8_optical_bands, adjusted_s2, lst_celsius, config['geometry'],
            s2_projection, indices_to_use=mlr_list_ind, water_mask=water_mask
        )
        accuracy_mlr = calculate_accuracy_metrics(lst_mlr, agg_LST, scale_agg, config['geometry'], 'MLR')
        results['methods']['MLR'] = {
            'lst': lst_mlr,
            'accuracy': accuracy_mlr
        }
        print(f"   ✓ MLR completed (indices: {', '.join(mlr_list_ind).upper()})")

    if 'RF' in methods:
        lst_rf = method_random_forest(
            l8_optical_bands, adjusted_s2, lst_celsius, config['geometry'],
            s2_projection, indices_to_use=mlr_list_ind, water_mask=water_mask,
            n_trees=rf_n_trees, num_samples=num_samples
        )
        accuracy_rf = calculate_accuracy_metrics(lst_rf, agg_LST, scale_agg, config['geometry'], 'RF')
        results['methods']['RF'] = {
            'lst': lst_rf,
            'accuracy': accuracy_rf
        }
        print(f"   ✓ Random Forest completed (indices: {', '.join(mlr_list_ind).upper()}, trees: {rf_n_trees})")

    if 'GB' in methods:
        lst_gb = method_gradient_boosting(
            l8_optical_bands, adjusted_s2, lst_celsius, config['geometry'],
            s2_projection, indices_to_use=mlr_list_ind, water_mask=water_mask,
            n_trees=gb_n_trees, num_samples=num_samples
        )
        accuracy_gb = calculate_accuracy_metrics(lst_gb, agg_LST, scale_agg, config['geometry'], 'GB')
        results['methods']['GB'] = {
            'lst': lst_gb,
            'accuracy': accuracy_gb
        }
        print(f"   ✓ Gradient Boosting completed (indices: {', '.join(mlr_list_ind).upper()}, trees: {gb_n_trees})")

    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    # Print accuracy metrics
    print("\n📊 ACCURACY METRICS (Downscaling):")
    print("=" * 80)
    print(f"{'Method':<12} {'RMSE (°C)':>12} {'MAE (°C)':>12} {'MD/Bias (°C)':>14}")
    print("-" * 52)

    for method_name, method_data in results['methods'].items():
        try:
            rmse = method_data['accuracy']['rmse'].getInfo()
            mae = method_data['accuracy']['mae'].getInfo()
            md_bias = method_data['accuracy']['md_bias'].getInfo()

            # Bias indicator
            bias_indicator = "↑" if md_bias > 0 else "↓" if md_bias < 0 else "="

            print(f"{method_name:<12} {rmse:>12.3f} {mae:>12.3f} {md_bias:>+13.3f} {bias_indicator}")

        except Exception as e:
            print(f"{method_name:<12} Calculation error - {str(e)}")

    print("-" * 52)
    print("MD/Bias: (+) overestimates | (-) underestimates temperature")
    print("=" * 80)

    return results