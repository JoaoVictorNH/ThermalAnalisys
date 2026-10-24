# ==============================================================================
# downscaling.py — LST Downscaling Methods
# ==============================================================================
# Implements OLS, RLS, MLR methods for downscaling
# LST from 30m to 10m, along with accuracy metrics.
# ==============================================================================

import ee
import numpy as np

# --- Module imports ---
from modulos.spectral_analysis import (
    calculate_spectral_indices_landsat,
    calculate_spectral_indices_sentinel2,
    calculate_spectral_indices_dynamic,
    resolve_predictors_from_image
)




def method_ols(l8_median, adjusted_s2, lst_celsius, agg_LST, geometry, S2proj):
    """
    OLS (Ordinary Least Squares) method — as per the original JavaScript code.
    Uses standard Landsat band names (SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7).
    """
    print("\n📊 Applying OLS method...")

    # Create constant image
    constant = ee.Image(1)

    # Concatenate bands for regression (using standard Landsat names)
    img_regress = ee.Image.cat(
        constant,
        l8_median.select('SR_B4'),  # Red
        l8_median.select('SR_B3'),  # Green
        l8_median.select('SR_B2'),  # Blue
        l8_median.select('SR_B5'),  # NIR
        l8_median.select('SR_B6'),  # SWIR1
        l8_median.select('SR_B7'),  # SWIR2
        agg_LST,
        lst_celsius
    )

    # Calculate OLS coefficients
    linear_regression = img_regress.reduceRegion(
        reducer=ee.Reducer.linearRegression(numX=8, numY=1),
        geometry=geometry,
        scale=30,
        tileScale=16,
        maxPixels=1e10
    )

    # Extract coefficients
    coef_list = ee.Array(linear_regression.get('coefficients')).toList()

    b4 = ee.List(coef_list.get(1)).get(0)  # Red
    b3 = ee.List(coef_list.get(2)).get(0)  # Green
    b2 = ee.List(coef_list.get(3)).get(0)  # Blue
    b5 = ee.List(coef_list.get(4)).get(0)  # NIR
    b6 = ee.List(coef_list.get(5)).get(0)  # SWIR1
    b7 = ee.List(coef_list.get(6)).get(0)  # SWIR2
    b1 = ee.List(coef_list.get(7)).get(0)  # aggLST
    b0 = ee.List(coef_list.get(0)).get(0)  # intercept

    # Calculate S2-LST OLS
    s2_lst = (adjusted_s2['B4'].multiply(ee.Number(b4))
             .add(adjusted_s2['B3'].multiply(ee.Number(b3)))
             .add(adjusted_s2['B2'].multiply(ee.Number(b2)))
             .add(adjusted_s2['B8'].multiply(ee.Number(b5)))
             .add(adjusted_s2['B11'].multiply(ee.Number(b6)))
             .add(adjusted_s2['B12'].multiply(ee.Number(b7)))
             .add(lst_celsius.multiply(ee.Number(b1)))
             .add(ee.Number(b0))
             .reproject(S2proj, None, 10))

    return s2_lst, agg_LST



def method_rls(l8_median, adjusted_s2, lst_celsius, agg_LST, geometry, S2proj):
    """
    RLS (Robust Least Squares) method — as per the original JavaScript code.
    Uses standard Landsat band names.
    """
    print("\n📊 Applying RLS method...")

    constant = ee.Image(1)

    img_regress = ee.Image.cat(
        constant,
        l8_median.select('SR_B4'),
        l8_median.select('SR_B3'),
        l8_median.select('SR_B2'),
        l8_median.select('SR_B5'),
        l8_median.select('SR_B6'),
        l8_median.select('SR_B7'),
        agg_LST,
        lst_celsius
    )

    # Calculate RLS coefficients
    linear_regression_rls = img_regress.reduceRegion(
        reducer=ee.Reducer.robustLinearRegression(numX=8, numY=1),
        geometry=geometry,
        scale=30,
        tileScale=16,
        maxPixels=1e10
    )

    # Extract coefficients
    coef_list_rls = ee.Array(linear_regression_rls.get('coefficients')).toList()

    b4rls = ee.List(coef_list_rls.get(1)).get(0)
    b3rls = ee.List(coef_list_rls.get(2)).get(0)
    b2rls = ee.List(coef_list_rls.get(3)).get(0)
    b5rls = ee.List(coef_list_rls.get(4)).get(0)
    b6rls = ee.List(coef_list_rls.get(5)).get(0)
    b7rls = ee.List(coef_list_rls.get(6)).get(0)
    b1rls = ee.List(coef_list_rls.get(7)).get(0)
    b0rls = ee.List(coef_list_rls.get(0)).get(0)

    # Calculate S2-LST RLS
    s2_lst_rls = (adjusted_s2['B4'].multiply(ee.Number(b4rls))
                 .add(adjusted_s2['B3'].multiply(ee.Number(b3rls)))
                 .add(adjusted_s2['B2'].multiply(ee.Number(b2rls)))
                 .add(adjusted_s2['B8'].multiply(ee.Number(b5rls)))
                 .add(adjusted_s2['B11'].multiply(ee.Number(b6rls)))
                 .add(adjusted_s2['B12'].multiply(ee.Number(b7rls)))
                 .add(lst_celsius.multiply(ee.Number(b1rls)))
                 .add(ee.Number(b0rls))
                 .reproject(S2proj, None, 10))

    return s2_lst_rls




def method_mlr(l8_median, adjusted_s2, lst_celsius, geometry, S2proj,
               indices_to_use=['ndvi', 'ndbi'], water_mask=None):
    """
    MLR (Multiple Linear Regression) method for LST downscaling.

    Accepts both spectral indices (NDVI, NDBI, ALBEDO, ...) and
    generic band names (NIR, SWIR1, SWIR2, RED, GREEN, BLUE).
    Band names are automatically mapped to the correct bands of each
    satellite via resolve_predictors_from_image.
    """
    from modulos.lst_analysis import calculate_modeled_lst_mlr, perform_linear_regression_mlr

    label = ', '.join([n.upper() for n in indices_to_use])
    print(f"📊 Applying MLR method with predictors: {label}...")
    if water_mask is not None:
        print("   • Water mask active — water pixels excluded from training")

    # Prepare adjusted Sentinel-2 image
    if isinstance(adjusted_s2, dict):
        bands_list = []
        for band_name in ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']:
            if band_name in adjusted_s2:
                bands_list.append(adjusted_s2[band_name].rename(band_name))
        adjusted_s2_image = ee.Image.cat(bands_list)
    else:
        adjusted_s2_image = adjusted_s2

    # Resolve predictors (indices + bands) for both satellites
    print("   • Resolving Landsat predictors (30m)...")
    landsat_selected = resolve_predictors_from_image(
        l8_median, indices_to_use, sensor='landsat', verbose=True
    )

    print("   • Resolving Sentinel-2 predictors (10m)...")
    sentinel2_selected = resolve_predictors_from_image(
        adjusted_s2_image, indices_to_use, sensor='sentinel2', verbose=True
    )

    print(f"   • Final Landsat predictors: {list(landsat_selected.keys())}")
    print(f"   • Final Sentinel-2 predictors: {list(sentinel2_selected.keys())}")

    if not landsat_selected:
        print(f"   ⚠️  WARNING: No Landsat predictors available for {indices_to_use}!")
    if not sentinel2_selected:
        print(f"   ⚠️  WARNING: No Sentinel-2 predictors available for {indices_to_use}!")

    # ── Water mask: exclude water pixels from training ──────────────────────
    # Regression is fitted only on land pixels to prevent distinct
    # water physics from contaminating the model coefficients.
    if water_mask is not None:
        land_mask_30m = water_mask.reproject(crs=S2proj, scale=30).Not()
        landsat_selected_train = {k: v.updateMask(land_mask_30m)
                                  for k, v in landsat_selected.items()}
        lst_for_train = lst_celsius.updateMask(land_mask_30m)
    else:
        landsat_selected_train = landsat_selected
        lst_for_train = lst_celsius

    # Run linear regression
    print("   • Running MLR linear regression...")
    coefficients = perform_linear_regression_mlr(
        lst_for_train,
        geometry,
        scale=30,
        **landsat_selected_train
    )

    print(f"   • MLR coefficients: {coefficients}")

    # Calculate modeled LST at 30m
    lst_model_30m = calculate_modeled_lst_mlr(
        coefficients,
        **landsat_selected
    )

    # Calculate residuals at 30m (actual LST - modeled LST)
    residuals_30m = lst_celsius.subtract(lst_model_30m)

    # ── Safeguard 1: clamp residuals ──────────────────────────────────────────
    # Limits residuals to ±15°C before smoothing. Edge pixels, residual cloud
    # shadow, or water can produce extreme residuals that propagate through the
    # Gaussian convolution and contaminate neighboring healthy pixels.
    residuals_30m = residuals_30m.clamp(-15, 15)

    # Smooth residuals with Gaussian filter
    print("   • Smoothing residuals...")
    gaussian_kernel = ee.Kernel.gaussian(radius=1.5, units='pixels')
    residuals_smoothed = residuals_30m.resample('bicubic').convolve(gaussian_kernel)

    # Calculate downscaled LST at 10m
    print("   • Applying downscaling to 10m...")
    lst_downscaled_10m = calculate_modeled_lst_mlr(
        coefficients,
        **sentinel2_selected
    )

    # Add smoothed residuals
    s2_lst_mlr = lst_downscaled_10m.add(residuals_smoothed)

    # ── Safeguard 2: mask pixels outside physical range ────────────────────
    # Pixels extrapolated by the MLR model become NoData (masked) and are
    # automatically excluded from any reduceRegion/statistics.
    # Using updateMask instead of clamp() avoids the artifact of minimum = 0°C
    # that occurs when clamp converts negative values to the lower bound.
    # Range: 0°C (minimum plausible tropical) to 80°C (extreme dry asphalt)
    valid_mask = s2_lst_mlr.gte(0).And(s2_lst_mlr.lte(80))
    s2_lst_mlr = s2_lst_mlr.updateMask(valid_mask)

    # ── Fill water pixels with bilinear resample of Landsat LST ──────────
    # MLR regression is not applied over water. The coarse LST (30m) is
    # resampled to 10m via bilinear interpolation — physically justifiable
    # because water bodies have spatially smooth temperature (no high-frequency
    # variability that would justify regression-based downscaling).
    if water_mask is not None:
        water_mask_10m = water_mask.reproject(crs=S2proj, scale=10).gt(0)
        lst_water_10m = lst_celsius.resample('bilinear').reproject(crs=S2proj, scale=10)
        s2_lst_mlr = s2_lst_mlr.where(water_mask_10m, lst_water_10m)
        print("   • Water pixels filled with bilinear resample of Landsat LST")

    print("   ✓ MLR method completed")

    return s2_lst_mlr




def _build_predictor_image(predictors_dict):
    """Stacks a predictor dictionary into an ee.Image with band names equal to the keys."""
    bands = [v.rename(k) for k, v in predictors_dict.items()]
    return ee.Image.cat(bands)


def method_random_forest(l8_median, adjusted_s2, lst_celsius, geometry, S2proj,
                         indices_to_use=['ndvi', 'ndbi'], water_mask=None,
                         n_trees=50, num_samples=5000):
    """
    Random Forest (RF) method for LST downscaling with residual reintroduction.

    The model is trained on Landsat predictors (30 m) and applied to Sentinel-2
    predictors (10 m) with the same band names. Prediction residuals at 30 m are
    smoothed and added to the 10 m prediction (same logic as MLR).

    Args:
        l8_median: ee.Image — Landsat 8 median with optical bands
        adjusted_s2: ee.Image or dict — bandpass-adjusted Sentinel-2
        lst_celsius: ee.Image — LST in Celsius at 30 m resolution
        geometry: ee.Geometry — ROI
        S2proj: ee.Projection — Sentinel-2 projection (EPSG:32724)
        indices_to_use: list — spectral predictor indices
        water_mask: ee.Image or None — water mask to exclude from training
        n_trees: int — number of Random Forest trees (default: 50)
        num_samples: int — number of pixels sampled for training (default: 5000)
    """
    label = ', '.join([n.upper() for n in indices_to_use])
    print(f"📊 Applying Random Forest method with predictors: {label} ({n_trees} trees)...")
    if water_mask is not None:
        print("   • Water mask active — water pixels excluded from training")

    # Prepare adjusted Sentinel-2 image
    if isinstance(adjusted_s2, dict):
        bands_list = []
        for band_name in ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']:
            if band_name in adjusted_s2:
                bands_list.append(adjusted_s2[band_name].rename(band_name))
        adjusted_s2_image = ee.Image.cat(bands_list)
    else:
        adjusted_s2_image = adjusted_s2

    # Resolve predictors for both satellites (identical band names)
    print("   • Resolving Landsat predictors (30m)...")
    landsat_selected = resolve_predictors_from_image(
        l8_median, indices_to_use, sensor='landsat', verbose=True
    )
    print("   • Resolving Sentinel-2 predictors (10m)...")
    sentinel2_selected = resolve_predictors_from_image(
        adjusted_s2_image, indices_to_use, sensor='sentinel2', verbose=True
    )

    # Water mask
    if water_mask is not None:
        land_mask_30m = water_mask.reproject(crs=S2proj, scale=30).Not()
        landsat_selected_train = {k: v.updateMask(land_mask_30m)
                                  for k, v in landsat_selected.items()}
        lst_for_train = lst_celsius.updateMask(land_mask_30m)
    else:
        landsat_selected_train = landsat_selected
        lst_for_train = lst_celsius

    predictor_names = list(landsat_selected.keys())

    # Build training image (predictors + LST target)
    train_pred_image = _build_predictor_image(landsat_selected_train)
    training_image = train_pred_image.addBands(lst_for_train.rename('LST'))

    # Sample training pixels
    print(f"   • Sampling {num_samples} training pixels (30m scale)...")
    training_data = training_image.sample(
        region=geometry,
        scale=30,
        numPixels=num_samples,
        seed=42,
        geometries=False
    )

    # Train Random Forest in regression mode
    print(f"   • Training Random Forest ({n_trees} trees)...")
    classifier = (ee.Classifier.smileRandomForest(numberOfTrees=n_trees)
                  .setOutputMode('REGRESSION')
                  .train(features=training_data,
                         classProperty='LST',
                         inputProperties=predictor_names))

    # Prediction at 30m → residuals
    l8_pred_image = _build_predictor_image(landsat_selected)
    lst_model_30m = l8_pred_image.classify(classifier).rename('LST')
    residuals_30m = lst_celsius.subtract(lst_model_30m).clamp(-15, 15)

    # Smooth residuals
    print("   • Smoothing residuals...")
    gaussian_kernel = ee.Kernel.gaussian(radius=1.5, units='pixels')
    residuals_smoothed = residuals_30m.resample('bicubic').convolve(gaussian_kernel)

    # Prediction at 10m (Sentinel-2) + residual reintroduction
    print("   • Applying downscaling to 10m...")
    s2_pred_image = _build_predictor_image(sentinel2_selected)
    lst_model_10m = s2_pred_image.classify(classifier).rename('LST')
    s2_lst_rf = lst_model_10m.add(residuals_smoothed)

    # Safeguard: mask pixels outside physical range
    valid_mask = s2_lst_rf.gte(0).And(s2_lst_rf.lte(80))
    s2_lst_rf = s2_lst_rf.updateMask(valid_mask)

    # Fill water pixels with bilinear resample
    if water_mask is not None:
        water_mask_10m = water_mask.reproject(crs=S2proj, scale=10).gt(0)
        lst_water_10m = lst_celsius.resample('bilinear').reproject(crs=S2proj, scale=10)
        s2_lst_rf = s2_lst_rf.where(water_mask_10m, lst_water_10m)
        print("   • Water pixels filled with bilinear resample of Landsat LST")

    print("   ✓ Random Forest method completed")
    return s2_lst_rf


def method_gradient_boosting(l8_median, adjusted_s2, lst_celsius, geometry, S2proj,
                              indices_to_use=['ndvi', 'ndbi'], water_mask=None,
                              n_trees=50, num_samples=5000):
    """
    Gradient Boosting (GB) method for LST downscaling with residual reintroduction.

    Uses ee.Classifier.smileGradientTreeBoost in REGRESSION mode. The training,
    prediction, and residual reintroduction logic is identical to the RF method.

    Args:
        l8_median: ee.Image — Landsat 8 median with optical bands
        adjusted_s2: ee.Image or dict — bandpass-adjusted Sentinel-2
        lst_celsius: ee.Image — LST in Celsius at 30 m resolution
        geometry: ee.Geometry — ROI
        S2proj: ee.Projection — Sentinel-2 projection (EPSG:32724)
        indices_to_use: list — spectral predictor indices
        water_mask: ee.Image or None — water mask to exclude from training
        n_trees: int — number of trees (boosting iterations) (default: 50)
        num_samples: int — number of pixels sampled for training (default: 5000)
    """
    label = ', '.join([n.upper() for n in indices_to_use])
    print(f"📊 Applying Gradient Boosting method with predictors: {label} ({n_trees} trees)...")
    if water_mask is not None:
        print("   • Water mask active — water pixels excluded from training")

    # Prepare adjusted Sentinel-2 image
    if isinstance(adjusted_s2, dict):
        bands_list = []
        for band_name in ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']:
            if band_name in adjusted_s2:
                bands_list.append(adjusted_s2[band_name].rename(band_name))
        adjusted_s2_image = ee.Image.cat(bands_list)
    else:
        adjusted_s2_image = adjusted_s2

    # Resolve predictors for both satellites (identical band names)
    print("   • Resolving Landsat predictors (30m)...")
    landsat_selected = resolve_predictors_from_image(
        l8_median, indices_to_use, sensor='landsat', verbose=True
    )
    print("   • Resolving Sentinel-2 predictors (10m)...")
    sentinel2_selected = resolve_predictors_from_image(
        adjusted_s2_image, indices_to_use, sensor='sentinel2', verbose=True
    )

    # Water mask
    if water_mask is not None:
        land_mask_30m = water_mask.reproject(crs=S2proj, scale=30).Not()
        landsat_selected_train = {k: v.updateMask(land_mask_30m)
                                  for k, v in landsat_selected.items()}
        lst_for_train = lst_celsius.updateMask(land_mask_30m)
    else:
        landsat_selected_train = landsat_selected
        lst_for_train = lst_celsius

    predictor_names = list(landsat_selected.keys())

    # Build training image (predictors + LST target)
    train_pred_image = _build_predictor_image(landsat_selected_train)
    training_image = train_pred_image.addBands(lst_for_train.rename('LST'))

    # Sample training pixels
    print(f"   • Sampling {num_samples} training pixels (30m scale)...")
    training_data = training_image.sample(
        region=geometry,
        scale=30,
        numPixels=num_samples,
        seed=42,
        geometries=False
    )

    # Train Gradient Boosting in regression mode
    print(f"   • Training Gradient Boosting ({n_trees} trees)...")
    classifier = (ee.Classifier.smileGradientTreeBoost(numberOfTrees=n_trees)
                  .setOutputMode('REGRESSION')
                  .train(features=training_data,
                         classProperty='LST',
                         inputProperties=predictor_names))

    # Prediction at 30m → residuals
    l8_pred_image = _build_predictor_image(landsat_selected)
    lst_model_30m = l8_pred_image.classify(classifier).rename('LST')
    residuals_30m = lst_celsius.subtract(lst_model_30m).clamp(-15, 15)

    # Smooth residuals
    print("   • Smoothing residuals...")
    gaussian_kernel = ee.Kernel.gaussian(radius=1.5, units='pixels')
    residuals_smoothed = residuals_30m.resample('bicubic').convolve(gaussian_kernel)

    # Prediction at 10m (Sentinel-2) + residual reintroduction
    print("   • Applying downscaling to 10m...")
    s2_pred_image = _build_predictor_image(sentinel2_selected)
    lst_model_10m = s2_pred_image.classify(classifier).rename('LST')
    s2_lst_gb = lst_model_10m.add(residuals_smoothed)

    # Safeguard: mask pixels outside physical range
    valid_mask = s2_lst_gb.gte(0).And(s2_lst_gb.lte(80))
    s2_lst_gb = s2_lst_gb.updateMask(valid_mask)

    # Fill water pixels with bilinear resample
    if water_mask is not None:
        water_mask_10m = water_mask.reproject(crs=S2proj, scale=10).gt(0)
        lst_water_10m = lst_celsius.resample('bilinear').reproject(crs=S2proj, scale=10)
        s2_lst_gb = s2_lst_gb.where(water_mask_10m, lst_water_10m)
        print("   • Water pixels filled with bilinear resample of Landsat LST")

    print("   ✓ Gradient Boosting method completed")
    return s2_lst_gb


def calculate_accuracy_metrics(lst_10m, agg_LST, scale_agg, geometry, method_name='RSL'):
    """
    Calculates accuracy metrics: RMSE, MAE, and MD (Bias).

    Metrics:
        - RMSE: Root Mean Square Error
        - MAE: Mean Absolute Error
        - MD/Bias: Mean Deviation (bias — signed mean difference)
    """
    # Aggregate LST 10m for comparison
    agg_S2LST = lst_10m.reproject(agg_LST.projection(), None, scale_agg)

    # Difference: predicted - observed
    difference = agg_S2LST.subtract(agg_LST)

    # ==================== MAE (Mean Absolute Error) ====================
    # MAE = mean(|predicted - observed|)
    abs_difference = difference.abs()

    mae_dict = abs_difference.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=scale_agg,
        tileScale=16,
        maxPixels=1e10
    )

    # ==================== MD / Bias (Mean Deviation) ====================
    # MD = mean(predicted - observed)
    # Positive = overestimation, Negative = underestimation
    md_dict = difference.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=scale_agg,
        tileScale=16,
        maxPixels=1e10
    )

    # ==================== RMSE (Root Mean Square Error) ====================
    # RMSE = sqrt(mean((predicted - observed)²))
    squared_difference = difference.pow(2)

    mse_dict = squared_difference.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=scale_agg,
        tileScale=16,
        maxPixels=1e10
    )

    # Extract correct band based on method
    if method_name == 'MLR':
        band_name = 'constant'
    elif method_name in ('RF', 'GB'):
        band_name = 'LST'
    else:
        band_name = 'B4'  # OLS/RLS

    # Calculate RMSE from MSE
    rmse = ee.Number(mse_dict.get(band_name)).sqrt()
    mae = ee.Number(mae_dict.get(band_name))
    md_bias = ee.Number(md_dict.get(band_name))

    return {
        'mae': mae,
        'md_bias': md_bias,
        'rmse': rmse,
        'band_name': band_name,
        'aggregated': agg_S2LST
    }