# ==============================================================================
# zonal_analysis.py — Zonal Analysis with Change Detection Masks
# ==============================================================================
# KML import, GeoTIFF loading, thresholding of the U-Net change map,
# numpy→EE mask conversion, zonal analysis (new buildings vs. control)
# and stratification by UHI intensity zones (Deng et al., 2023).
# ==============================================================================

import ee
import geemap
import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch



def kml_to_ee_fixed(kml_path, return_geometry=False):
    """
    Converts a KML file to an Earth Engine FeatureCollection or Geometry.

    Automatically removes Z coordinates (altitude) that cause errors in Earth Engine.

    Parameters:
    -----------
    kml_path : str
        Full path to the KML file
    return_geometry : bool, optional (default=False)
        If True, returns ee.Geometry (dissolve of all features)
        If False, returns ee.FeatureCollection

    Returns:
    --------
    ee.FeatureCollection or ee.Geometry
        Earth Engine object ready for use

    Example:
    --------
    >>> # Return as FeatureCollection
    >>> roi_fc = kml_to_ee_fixed("/path/to/file.kml")
    >>>
    >>> # Return as Geometry
    >>> roi = kml_to_ee_fixed("/path/to/file.kml", return_geometry=True)
    """

    def remove_z_coordinate(coords):
        """Recursively removes the third coordinate (Z) from coordinate arrays."""
        if isinstance(coords[0], (int, float)):
            # Individual coordinate [x, y, z]
            return coords[:2]
        else:
            # List of coordinates
            return [remove_z_coordinate(coord) for coord in coords]

    try:
        # Step 1: Convert KML to GeoJSON
        temp_geojson = "/tmp/temp_kml.geojson"
        geemap.kml_to_geojson(kml_path, temp_geojson)

        # Step 2: Read GeoJSON and remove Z coordinate
        with open(temp_geojson, 'r') as f:
            geojson_data = json.load(f)

        # Apply Z removal to all features
        for feature in geojson_data['features']:
            if 'geometry' in feature and feature['geometry'] is not None:
                coords = feature['geometry']['coordinates']
                feature['geometry']['coordinates'] = remove_z_coordinate(coords)

        # Step 3: Save corrected GeoJSON
        corrected_geojson = "/tmp/corrected_kml.geojson"
        with open(corrected_geojson, 'w') as f:
            json.dump(geojson_data, f)

        # Step 4: Convert to Earth Engine
        roi_fc = geemap.geojson_to_ee(corrected_geojson)

        # Import information
        num_features = roi_fc.size().getInfo()
        print(f"✓ KML imported successfully!")
        print(f"  Number of features: {num_features}")

        if return_geometry:
            roi = roi_fc.geometry()
            return roi
        else:
            print(f"  Type: FeatureCollection")
            return roi_fc

    except Exception as e:
        print(f"✗ Error importing KML: {str(e)}")
        raise


def load_geotiff(file_path):
    """Loads a GeoTIFF file and returns the array and metadata."""
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first band
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        bounds = src.bounds
        shape = data.shape
    return data, transform, crs, nodata, bounds, shape


def apply_threshold(prediction_data, threshold, nodata_value=None):
    """
    Applies a threshold to the prediction raster to create a binary mask.

    Args:
        prediction_data: numpy.array - Prediction data
        threshold: float - Threshold value
        nodata_value: Nodata value

    Returns:
        numpy.array: Binary mask (1=change, 0=no change, NaN=nodata)
    """
    # Create mask as float to support NaN
    binary_mask = np.zeros(prediction_data.shape, dtype=np.float32)

    # Apply threshold
    if nodata_value is not None:
        valid_mask = (prediction_data != nodata_value) & ~np.isnan(prediction_data)
    else:
        valid_mask = ~np.isnan(prediction_data)

    # Classify: >= threshold = 1 (change), < threshold = 0 (no change)
    binary_mask[valid_mask & (prediction_data >= threshold)] = 1.0
    binary_mask[valid_mask & (prediction_data < threshold)] = 0.0
    binary_mask[~valid_mask] = np.nan

    # Binary mask statistics
    total_valid = np.sum(valid_mask)
    pixels_change = np.sum(binary_mask == 1)
    pixels_no_change = np.sum(binary_mask == 0)
    percent_change = (pixels_change / total_valid * 100) if total_valid > 0 else 0

    print(f"\n✓ Threshold applied: {threshold:.2f}")
    print(f"   • Change pixels (≥{threshold:.2f}): {pixels_change:,} ({percent_change:.2f}%)")
    print(f"   • No-change pixels (<{threshold:.2f}): {pixels_no_change:,} ({100-percent_change:.2f}%)")

    return binary_mask


def visualize_binary_mask(binary_mask, prediction_data, threshold):
    """
    Visualizes the resulting binary mask.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original prediction
    im1 = axes[0].imshow(prediction_data, cmap='gray', vmin=np.nanmin(prediction_data), vmax=np.nanmax(prediction_data))
    axes[0].set_title('Original Prediction\n(Continuous Values)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Change areas overlay
    overlay = np.ones((*prediction_data.shape, 3))
    valid_mask = ~np.isnan(binary_mask)
    overlay[~valid_mask] = [1, 1, 1]
    overlay[valid_mask & (binary_mask == 0)] = [0.8, 0.8, 0.8]
    overlay[valid_mask & (binary_mask == 1)] = [0.9, 0.2, 0.2]

    axes[1].imshow(overlay)
    axes[1].set_title(f'Change Areas\n(Threshold = {threshold:.0f}  |  Red = Change, Gray = No Change)',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')

    legend_elements = [
        Patch(facecolor=[0.9, 0.2, 0.2], label=f'Change (≥{threshold:.0f})'),
        Patch(facecolor=[0.8, 0.8, 0.8], label=f'No Change (<{threshold:.0f})')
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 6.X ZONAL ANALYSIS WITH BUILDING CHANGE MASK (ROBUST VERSION)
# ==============================================================================

def apply_building_change_mask_to_results(
    results_dict,
    binary_change_mask,
    roi,
    scale=10,
    output_dir=None,
    roi_name='study_area',
    mask_name='building_change',
    mask_ee=None,
    verbose=False,
    show_violin=False,
):
    """
    Applies binary change mask to the integrated analysis results.
    ROBUST VERSION: Uses direct sampling instead of spatial conversion.
    UPDATED: Includes UHI intensity classification analysis.
    """
    _print = print if verbose else lambda *a, **k: None

    _print("\n" + "="*80)
    _print(f"ZONAL ANALYSIS WITH MASK: {mask_name.upper()}")
    _print("="*80)
    _print(f"📍 Area: {roi_name}")
    _print(f"📊 Index: {results_dict['index_name']}")
    _print(f"📏 Scale: {scale}m")

    # ==============================================================================
    # 0. VALIDATE MASK AND CHECK INTENSITY CLASSIFICATIONS
    # ==============================================================================

    _print("\n[0/4] Validating mask...")

    if isinstance(binary_change_mask, tuple):
        binary_change_mask = binary_change_mask[0]

    if not isinstance(binary_change_mask, np.ndarray):
        raise TypeError(f"❌ Expected numpy.ndarray, received: {type(binary_change_mask)}")

    n_change = np.sum(binary_change_mask == 1)
    n_no_change = np.sum(binary_change_mask == 0)

    _print(f"   ✓ Shape: {binary_change_mask.shape}")
    _print(f"   ✓ New construction pixels: {n_change:,} ({n_change/binary_change_mask.size*100:.2f}%)")
    _print(f"   ✓ Other pixels: {n_no_change:,}")

    if n_change == 0:
        raise ValueError("❌ No change pixels found!")

    # Check if UHI intensity classification is available
    has_uhi_intensity = ('uhi_intensity_t1' in results_dict['images'] and
                         'uhi_intensity_t2' in results_dict['images'] and
                         'delta_uhi_intensity' in results_dict['images'])

    if has_uhi_intensity:
        _print(f"   ✓ UHI intensity classification available")


    # ==============================================================================
    # 1. PREPARE BANDS + MASK FOR NATIVE GEE SAMPLING
    # ==============================================================================

    _print("\n[1/4] Preparing sampling via GEE...")

    # Get or convert mask_ee
    if mask_ee is None:
        _print("   → Converting numpy mask → ee.Image...")
        mask_ee_local = _numpy_mask_to_ee_image(binary_change_mask, roi, band_name='mask_class').toInt()
    else:
        _print("   ✓ Using pre-computed mask_ee")
        mask_ee_local = mask_ee.rename('mask_class').toInt()

    # Extrair imagens
    delta_index = results_dict['images']['delta_index']
    delta_lst   = results_dict['images']['delta_lst']
    delta_uhi   = results_dict['images']['delta_uhi']
    _imgs       = results_dict['images']
    _index_t1   = _imgs.get('index_t1')
    _index_t2   = _imgs.get('index_t2')
    _lst_t1     = _imgs.get('lst_t1')
    _lst_t2     = _imgs.get('lst_t2')
    _has_abs    = all(v is not None for v in [_index_t1, _index_t2, _lst_t1, _lst_t2])

    # Combinar deltas + máscara em uma única imagem
    combined_bands = [
        delta_index.rename('delta_index'),
        delta_lst.rename('delta_lst'),
        delta_uhi.rename('delta_uhi'),
        mask_ee_local,
    ]

    if _has_abs:
        combined_bands.extend([
            _index_t1.rename('index_t1'),
            _index_t2.rename('index_t2'),
            _lst_t1.rename('lst_t1'),
            _lst_t2.rename('lst_t2'),
        ])

    if has_uhi_intensity:
        combined_bands.extend([
            results_dict['images']['delta_uhi_intensity'].rename('delta_uhi_intensity'),
            results_dict['images']['uhi_intensity_t1'].rename('uhi_intensity_t1'),
            results_dict['images']['uhi_intensity_t2'].rename('uhi_intensity_t2')
        ])

    combined = ee.Image.cat(combined_bands).clip(roi)

    # ==============================================================================
    # 2-3. NATIVE GEE STRATIFIED SAMPLING
    # ==============================================================================

    _print("\n[2/4] Sampling pixels (stratified by mask)...")

    try:
        samples = combined.stratifiedSample(
            numPoints=2000,
            classBand='mask_class',
            region=roi,
            scale=scale,
            geometries=False
        )

        all_data = samples.getInfo()['features']

        # Separar por classe
        data_change = [f for f in all_data if f['properties'].get('mask_class', 0) == 1]
        data_no_change = [f for f in all_data if f['properties'].get('mask_class', 0) == 0]

        _print(f"   ✓ {len(data_change)} change samples")
        _print(f"   ✓ {len(data_no_change)} control samples")

        def _extract(features, key):
            return [f['properties'][key] for f in features
                    if f['properties'].get(key) is not None]

        # Extrair arrays — mudança
        delta_index_change    = _extract(data_change, 'delta_index')
        delta_lst_change      = _extract(data_change, 'delta_lst')
        delta_uhi_change      = _extract(data_change, 'delta_uhi')
        index_t1_change       = _extract(data_change, 'index_t1') if _has_abs else []
        index_t2_change       = _extract(data_change, 'index_t2') if _has_abs else []
        lst_t1_change         = _extract(data_change, 'lst_t1')   if _has_abs else []
        lst_t2_change         = _extract(data_change, 'lst_t2')   if _has_abs else []

        # Extrair arrays — controle
        delta_index_no_change = _extract(data_no_change, 'delta_index')
        delta_lst_no_change   = _extract(data_no_change, 'delta_lst')
        delta_uhi_no_change   = _extract(data_no_change, 'delta_uhi')
        index_t1_no_change    = _extract(data_no_change, 'index_t1') if _has_abs else []
        index_t2_no_change    = _extract(data_no_change, 'index_t2') if _has_abs else []
        lst_t1_no_change      = _extract(data_no_change, 'lst_t1')   if _has_abs else []
        lst_t2_no_change      = _extract(data_no_change, 'lst_t2')   if _has_abs else []

        # Intensidade UHI
        if has_uhi_intensity:
            delta_uhi_intensity_change = [f['properties']['delta_uhi_intensity'] for f in data_change
                                          if f['properties'].get('delta_uhi_intensity') is not None]
            uhi_intensity_t1_change = [f['properties']['uhi_intensity_t1'] for f in data_change
                                       if f['properties'].get('uhi_intensity_t1') is not None]
            uhi_intensity_t2_change = [f['properties']['uhi_intensity_t2'] for f in data_change
                                       if f['properties'].get('uhi_intensity_t2') is not None]

            delta_uhi_intensity_no_change = [f['properties']['delta_uhi_intensity'] for f in data_no_change
                                             if f['properties'].get('delta_uhi_intensity') is not None]
            uhi_intensity_t1_no_change = [f['properties']['uhi_intensity_t1'] for f in data_no_change
                                          if f['properties'].get('uhi_intensity_t1') is not None]
            uhi_intensity_t2_no_change = [f['properties']['uhi_intensity_t2'] for f in data_no_change
                                          if f['properties'].get('uhi_intensity_t2') is not None]
        else:
            delta_uhi_intensity_change = uhi_intensity_t1_change = uhi_intensity_t2_change = []
            delta_uhi_intensity_no_change = uhi_intensity_t1_no_change = uhi_intensity_t2_no_change = []

        _print("\n[3/4] Samples extracted successfully.")

    except Exception as e:
        print(f"   ⚠️  Sampling error: {str(e)[:80]}")
        delta_index_change = delta_lst_change = delta_uhi_change = []
        delta_index_no_change = delta_lst_no_change = delta_uhi_no_change = []
        index_t1_change = index_t2_change = lst_t1_change = lst_t2_change = []
        index_t1_no_change = index_t2_no_change = lst_t1_no_change = lst_t2_no_change = []
        delta_uhi_intensity_change = uhi_intensity_t1_change = uhi_intensity_t2_change = []
        delta_uhi_intensity_no_change = uhi_intensity_t1_no_change = uhi_intensity_t2_no_change = []

    # ==============================================================================
    # 4. COMPUTE STATISTICS AND VISUALIZE
    # ==============================================================================

    _print("\n[4/4] Computing statistics...")

    # Convert to numpy arrays
    delta_index_change = np.array(delta_index_change)
    delta_lst_change = np.array(delta_lst_change)
    delta_uhi_change = np.array(delta_uhi_change)

    delta_uhi_intensity_change = np.array(delta_uhi_intensity_change)
    uhi_intensity_t1_change = np.array(uhi_intensity_t1_change)
    uhi_intensity_t2_change = np.array(uhi_intensity_t2_change)

    delta_index_no_change = np.array(delta_index_no_change)
    delta_lst_no_change = np.array(delta_lst_no_change)
    delta_uhi_no_change = np.array(delta_uhi_no_change)

    delta_uhi_intensity_no_change = np.array(delta_uhi_intensity_no_change)
    uhi_intensity_t1_no_change = np.array(uhi_intensity_t1_no_change)
    uhi_intensity_t2_no_change = np.array(uhi_intensity_t2_no_change)

    # Convert abs arrays to numpy before filter (they're still lists when _has_abs is True)
    index_t1_change    = np.array(index_t1_change)
    index_t2_change    = np.array(index_t2_change)
    lst_t1_change      = np.array(lst_t1_change)
    lst_t2_change      = np.array(lst_t2_change)
    index_t1_no_change = np.array(index_t1_no_change)
    index_t2_no_change = np.array(index_t2_no_change)
    lst_t1_no_change   = np.array(lst_t1_no_change)
    lst_t2_no_change   = np.array(lst_t2_no_change)

    # ── 3×IQR outlier removal on Δ index — mirrors cell 8.1 ──────────────────
    # Applied independently per group; UHI intensity arrays (class labels) are excluded.
    def _apply_iqr3(group_label, di, *linked):
        if len(di) <= 10:
            return (di,) + linked
        q1, q3 = np.percentile(di, [25, 75])
        iqr = q3 - q1
        if iqr <= 1e-9:
            return (di,) + linked
        lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        keep = (di >= lo) & (di <= hi)
        n_rem = int((~keep).sum())
        if n_rem:
            print(f"   🔬 Outliers removed 3×IQR Δ{results_dict['index_name']} [{group_label}]: "
                  f"{n_rem} ({n_rem / len(di) * 100:.1f}%)")
        return (di[keep],) + tuple(a[keep] if len(a) == len(di) else a for a in linked)

    (delta_index_change, delta_lst_change, delta_uhi_change,
     index_t1_change, index_t2_change,
     lst_t1_change, lst_t2_change) = _apply_iqr3(
        'change',
        delta_index_change, delta_lst_change, delta_uhi_change,
        index_t1_change, index_t2_change, lst_t1_change, lst_t2_change,
    )

    (delta_index_no_change, delta_lst_no_change, delta_uhi_no_change,
     index_t1_no_change, index_t2_no_change,
     lst_t1_no_change, lst_t2_no_change) = _apply_iqr3(
        'no_change',
        delta_index_no_change, delta_lst_no_change, delta_uhi_no_change,
        index_t1_no_change, index_t2_no_change, lst_t1_no_change, lst_t2_no_change,
    )

    # Estatísticas
    def calc_stats(arr, name):
        if len(arr) == 0:
            return {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'count': 0}
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'count': len(arr)
        }

    stats_change = {
        'delta_index': calc_stats(delta_index_change, 'index'),
        'delta_lst': calc_stats(delta_lst_change, 'lst'),
        'delta_uhi': calc_stats(delta_uhi_change, 'uhi')
    }

    stats_no_change = {
        'delta_index': calc_stats(delta_index_no_change, 'index'),
        'delta_lst': calc_stats(delta_lst_no_change, 'lst'),
        'delta_uhi': calc_stats(delta_uhi_no_change, 'uhi')
    }

    # Add UHI intensity statistics if available
    if has_uhi_intensity and len(delta_uhi_intensity_change) > 0:
        stats_change['delta_uhi_intensity'] = calc_stats(delta_uhi_intensity_change, 'uhi_intensity')
        stats_change['uhi_intensity_t1'] = calc_stats(uhi_intensity_t1_change, 'uhi_intensity_t1')
        stats_change['uhi_intensity_t2'] = calc_stats(uhi_intensity_t2_change, 'uhi_intensity_t2')

        stats_no_change['delta_uhi_intensity'] = calc_stats(delta_uhi_intensity_no_change, 'uhi_intensity')
        stats_no_change['uhi_intensity_t1'] = calc_stats(uhi_intensity_t1_no_change, 'uhi_intensity_t1')
        stats_no_change['uhi_intensity_t2'] = calc_stats(uhi_intensity_t2_no_change, 'uhi_intensity_t2')


    _print(f"\n   📊 New constructions:")
    _print(f"      • Δ LST: {stats_change['delta_lst']['mean']:+.2f}°C ({stats_change['delta_lst']['count']} px)")
    _print(f"      • Δ UHI: {stats_change['delta_uhi']['mean']:+.4f} ({stats_change['delta_uhi']['count']} px)")

    if has_uhi_intensity and len(delta_uhi_intensity_change) > 0:
        _print(f"      • Δ UHI Intensity: {stats_change['delta_uhi_intensity']['mean']:+.2f} classes ({stats_change['delta_uhi_intensity']['count']} px)")
        _print(f"      • UHI Intensity T1: {stats_change['uhi_intensity_t1']['mean']:.2f} ({stats_change['uhi_intensity_t1']['count']} px)")
        _print(f"      • UHI Intensity T2: {stats_change['uhi_intensity_t2']['mean']:.2f} ({stats_change['uhi_intensity_t2']['count']} px)")

    _print(f"\n   📊 Others:")
    _print(f"      • Δ LST: {stats_no_change['delta_lst']['mean']:+.2f}°C ({stats_no_change['delta_lst']['count']} px)")
    _print(f"      • Δ UHI: {stats_no_change['delta_uhi']['mean']:+.4f} ({stats_no_change['delta_uhi']['count']} px)")

    if has_uhi_intensity and len(delta_uhi_intensity_no_change) > 0:
        _print(f"      • Δ UHI Intensity: {stats_no_change['delta_uhi_intensity']['mean']:+.2f} classes ({stats_no_change['delta_uhi_intensity']['count']} px)")
        _print(f"      • UHI Intensity T1: {stats_no_change['uhi_intensity_t1']['mean']:.2f} ({stats_no_change['uhi_intensity_t1']['count']} px)")
        _print(f"      • UHI Intensity T2: {stats_no_change['uhi_intensity_t2']['mean']:.2f} ({stats_no_change['uhi_intensity_t2']['count']} px)")

    # Differences
    diff_lst = stats_change['delta_lst']['mean'] - stats_no_change['delta_lst']['mean']
    diff_uhi = stats_change['delta_uhi']['mean'] - stats_no_change['delta_uhi']['mean']

    _print(f"\n   🎯 DIFFERENCES (new construction mean - other mean):")
    _print(f"      • Δ LST: {diff_lst:+.2f}°C")
    _print(f"      • Δ UHI: {diff_uhi:+.4f}")

    if has_uhi_intensity and len(delta_uhi_intensity_change) > 0 and len(delta_uhi_intensity_no_change) > 0:
        diff_uhi_intensity = stats_change['delta_uhi_intensity']['mean'] - stats_no_change['delta_uhi_intensity']['mean']
        _print(f"      • Δ UHI Intensity: {diff_uhi_intensity:+.2f} classes")

    # ── Correlações ───────────────────────────────────────────────────────────
    from scipy.stats import pearsonr, spearmanr, mannwhitneyu

    corr_il = r2_il = p_il = np.nan
    corr_iu = r2_iu = p_iu = np.nan
    rho_il  = p_rho_il = np.nan
    rho_iu  = p_rho_iu = np.nan

    if len(delta_index_change) > 30:
        corr_il, p_il = pearsonr(delta_index_change, delta_lst_change)
        corr_iu, p_iu = pearsonr(delta_index_change, delta_uhi_change)
        r2_il = corr_il ** 2
        r2_iu = corr_iu ** 2

        rho_il, p_rho_il = spearmanr(delta_index_change, delta_lst_change)
        rho_iu, p_rho_iu = spearmanr(delta_index_change, delta_uhi_change)

        _print(f"\n   📈 CORRELATIONS (new buildings):")
        _print(f"      • Δ Index × Δ LST — Pearson r={corr_il:+.4f}  R²={r2_il:.3f}  (p={p_il:.4f})")
        _print(f"                        Spearman ρ={rho_il:+.4f}               (p={p_rho_il:.4f})")
        _print(f"      • Δ Index × Δ UHI — Pearson r={corr_iu:+.4f}  R²={r2_iu:.3f}  (p={p_iu:.4f})")
        _print(f"                        Spearman ρ={rho_iu:+.4f}               (p={p_rho_iu:.4f})")

        if has_uhi_intensity and len(delta_uhi_intensity_change) > 30:
            corr_i_uhi_intensity, p_i_uhi_intensity = pearsonr(delta_index_change, delta_uhi_intensity_change)
            r2_i_uhi_intensity = corr_i_uhi_intensity ** 2
            rho_i_uhi_intensity, p_rho_i_uhi_intensity = spearmanr(delta_index_change, delta_uhi_intensity_change)
            _print(f"      • Δ Index × Δ UHI Intensity — Pearson r={corr_i_uhi_intensity:+.4f}  R²={r2_i_uhi_intensity:.3f}  (p={p_i_uhi_intensity:.4f})")
            _print(f"                                   Spearman ρ={rho_i_uhi_intensity:+.4f}               (p={p_rho_i_uhi_intensity:.4f})")

    # ── Hypothesis test (Mann-Whitney U) + effect size (Cohen's d) ────
    p_mw_lst = p_mw_uhi = np.nan
    cohens_d_lst = cohens_d_uhi = np.nan

    # Helper functions defined here so they are always in scope,
    # even when sample size is too small to run the statistical test.
    def _sig_stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'ns'

    def _effect_label(d):
        ad = abs(d)
        if ad < 0.2:  return 'negligible'
        if ad < 0.5:  return 'small'
        if ad < 0.8:  return 'medium'
        return 'large'

    if len(delta_lst_change) > 10 and len(delta_lst_no_change) > 10:
        # Mann-Whitney U — two-sided (does not assume normality)
        _, p_mw_lst = mannwhitneyu(delta_lst_change, delta_lst_no_change, alternative='two-sided')
        _, p_mw_uhi = mannwhitneyu(delta_uhi_change, delta_uhi_no_change, alternative='two-sided')

        # Cohen's d (mean difference / pooled standard deviation)
        def _cohens_d(a, b):
            pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
            return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0

        cohens_d_lst = _cohens_d(delta_lst_change, delta_lst_no_change)
        cohens_d_uhi = _cohens_d(delta_uhi_change, delta_uhi_no_change)

        _print(f"\n   📐 HYPOTHESIS TEST (Mann-Whitney U, two-sided):")
        _print(f"      • Δ LST:  p={p_mw_lst:.4f} {_sig_stars(p_mw_lst)}  |  d={cohens_d_lst:+.3f} ({_effect_label(cohens_d_lst)})")
        _print(f"      • Δ UHI:  p={p_mw_uhi:.4f} {_sig_stars(p_mw_uhi)}  |  d={cohens_d_uhi:+.3f} ({_effect_label(cohens_d_uhi)})")

    # ── Automatic interpretation ───────────────────────────────────────────────
    _sig_lst = (not np.isnan(p_mw_lst)) and (p_mw_lst < 0.05)
    _dir = "LESS" if diff_lst < 0 else "MORE"
    _hipotese = (
        "supports the central hypothesis" if diff_lst > 0 and _sig_lst
        else "contradicts the central hypothesis" if diff_lst < 0 and _sig_lst
        else "does not allow a definitive conclusion (non-significant difference)"
    )
    _provavel_causa = (
        " Probable cause: replacement of bare/agricultural soil with pre-existing high LST."
        if diff_lst < 0 else ""
    )

    interpretation = (
        f"Areas with new construction warmed {_dir} than control areas "
        f"(Δ = {diff_lst:+.2f} °C). "
        f"The difference is {'statistically significant' if _sig_lst else 'not significant'} "
        f"(Mann-Whitney U, p={p_mw_lst:.4f}), {_effect_label(cohens_d_lst)} effect "
        f"(d={cohens_d_lst:+.3f}). "
        f"Result {_hipotese}.{_provavel_causa}"
    )
    # Interpretation available in zonal_results['interpretation'] — displayed in notebook via HTML

    # ── Visualization (violin plots + scatter) ─────────────────────────────────
    n_plots = 3  # Base: Δ LST, Δ UHI, Scatter
    if has_uhi_intensity and len(delta_uhi_intensity_change) > 10:
        n_plots += 1

    if show_violin and len(delta_lst_change) > 10 and len(delta_lst_no_change) > 10:

        COLORS = {'change': '#E05252', 'no_change': '#5285E0'}

        def _violin_ax(ax, data_c, data_nc, ylabel, title, sig_p):
            """Draws violins for both groups with mean + significance annotation."""
            parts_c  = ax.violinplot([data_c],  positions=[1], showmedians=True, showextrema=False)
            parts_nc = ax.violinplot([data_nc], positions=[2], showmedians=True, showextrema=False)

            for pc in parts_c['bodies']:
                pc.set_facecolor(COLORS['change'])
                pc.set_alpha(0.6)
            parts_c['cmedians'].set_color(COLORS['change'])

            for pc in parts_nc['bodies']:
                pc.set_facecolor(COLORS['no_change'])
                pc.set_alpha(0.6)
            parts_nc['cmedians'].set_color(COLORS['no_change'])

            # Mean as point
            ax.scatter([1], [np.mean(data_c)],  color=COLORS['change'],    zorder=5, s=60, marker='D', label='Mean')
            ax.scatter([2], [np.mean(data_nc)], color=COLORS['no_change'], zorder=5, s=60, marker='D')

            # Reference line y=0
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

            # Significance annotation
            stars = _sig_stars(sig_p)
            y_max = max(np.percentile(data_c, 95), np.percentile(data_nc, 95))
            y_ann = y_max * 1.08
            ax.plot([1, 1, 2, 2], [y_max, y_ann, y_ann, y_max], color='black', linewidth=0.8)
            ax.text(1.5, y_ann * 1.01, stars, ha='center', va='bottom', fontsize=11,
                    color='black' if stars != 'ns' else 'gray')

            ax.set_xticks([1, 2])
            ax.set_xticklabels(['With New\nBuildings', 'Without New\nBuildings'], fontsize=9)
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(title, fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.25, axis='y')

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        fig.suptitle(
            f'Zonal Analysis — {results_dict["index_name"].upper()} × Thermal ({roi_name})\n'
            f'n_change={len(delta_lst_change):,}  |  n_control={len(delta_lst_no_change):,}  '
            f'|  change_pixels={n_change:,} ({n_change/binary_change_mask.size*100:.1f}%)',
            fontsize=10, y=1.02
        )

        plot_idx = 0

        # Violin Δ LST
        _violin_ax(
            axes[plot_idx],
            delta_lst_change, delta_lst_no_change,
            'Δ LST (°C)', f'Temperature Change\n(T2 − T1)',
            p_mw_lst
        )
        # Add numerical summary to subplot
        axes[plot_idx].text(
            0.02, 0.98,
            f"With:    {np.mean(delta_lst_change):+.2f} ± {np.std(delta_lst_change):.2f} °C\n"
            f"Without: {np.mean(delta_lst_no_change):+.2f} ± {np.std(delta_lst_no_change):.2f} °C\n"
            f"Δ diff: {diff_lst:+.2f} °C  |  d={cohens_d_lst:+.2f}",
            transform=axes[plot_idx].transAxes,
            va='top', ha='left', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
        )
        plot_idx += 1

        # Violin Δ UHI
        _violin_ax(
            axes[plot_idx],
            delta_uhi_change, delta_uhi_no_change,
            'Δ UHI (z-score)', 'Urban Heat Island Change\n(T2 − T1)',
            p_mw_uhi
        )
        axes[plot_idx].text(
            0.02, 0.98,
            f"With:    {np.mean(delta_uhi_change):+.4f}\n"
            f"Without: {np.mean(delta_uhi_no_change):+.4f}\n"
            f"Δ diff: {diff_uhi:+.4f}  |  d={cohens_d_uhi:+.2f}",
            transform=axes[plot_idx].transAxes,
            va='top', ha='left', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
        )
        plot_idx += 1

        # Scatter Δ Index × Δ LST (change zone)
        if len(delta_index_change) > 10:
            ax_sc = axes[plot_idx]
            ax_sc.scatter(delta_index_change, delta_lst_change,
                          alpha=0.35, s=10, color=COLORS['change'], label='With change')
            # Trend line
            if not np.isnan(r2_il):
                m, b = np.polyfit(delta_index_change, delta_lst_change, 1)
                x_line = np.linspace(delta_index_change.min(), delta_index_change.max(), 100)
                ax_sc.plot(x_line, m * x_line + b, color='darkred', linewidth=1.5,
                           label=f'Trend (R²={r2_il:.2f})')
            ax_sc.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax_sc.axvline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax_sc.set_xlabel(f'Δ {results_dict["index_name"].upper()}', fontweight='bold')
            ax_sc.set_ylabel('Δ LST (°C)', fontweight='bold')
            ax_sc.set_title(f'Correlation in Change Areas\n(R²={r2_il:.2f})', fontweight='bold', fontsize=10)
            ax_sc.grid(True, alpha=0.25)
            ax_sc.legend(fontsize=8)
            plot_idx += 1

        # Violin Δ UHI Intensity (if available)
        if has_uhi_intensity and len(delta_uhi_intensity_change) > 10:
            _, p_mw_uhi_int = mannwhitneyu(
                delta_uhi_intensity_change, delta_uhi_intensity_no_change, alternative='two-sided'
            )
            _violin_ax(
                axes[plot_idx],
                delta_uhi_intensity_change, delta_uhi_intensity_no_change,
                'Δ UHI Class', 'UHI Intensity\nClass Change',
                p_mw_uhi_int
            )
            plot_idx += 1

        plt.tight_layout()

        if output_dir:
            plt.savefig(
                f'{output_dir}zonal_{mask_name}_{results_dict["index_name"]}_{roi_name}.png',
                dpi=300, bbox_inches='tight'
            )

        plt.show()

    print("\n" + "="*80)
    print("✅ ZONAL ANALYSIS COMPLETE")
    print("="*80)

    # ── Prepare return value ───────────────────────────────────────────────────
    result = {
        'mask_name': mask_name,
        'index_name': results_dict['index_name'],
        'n_change_pixels': int(n_change),
        'n_total_pixels': int(binary_change_mask.size),
        'stats_change': stats_change,
        'stats_no_change': stats_no_change,
        'samples_change': {
            'delta_index': delta_index_change,
            'delta_lst':   delta_lst_change,
            'delta_uhi':   delta_uhi_change,
            'index_t1':    index_t1_change,
            'index_t2':    index_t2_change,
            'lst_t1':      lst_t1_change,
            'lst_t2':      lst_t2_change,
        },
        'samples_no_change': {
            'delta_index': delta_index_no_change,
            'delta_lst':   delta_lst_no_change,
            'delta_uhi':   delta_uhi_no_change,
            'index_t1':    index_t1_no_change,
            'index_t2':    index_t2_no_change,
            'lst_t1':      lst_t1_no_change,
            'lst_t2':      lst_t2_no_change,
        },
        'correlations': {
            'index_lst': {
                'pearson_r': corr_il, 'r2': r2_il, 'pearson_p': p_il,
                'spearman_rho': rho_il, 'spearman_p': p_rho_il
            },
            'index_uhi': {
                'pearson_r': corr_iu, 'r2': r2_iu, 'pearson_p': p_iu,
                'spearman_rho': rho_iu, 'spearman_p': p_rho_iu
            }
        },
        'differences': {
            'delta_lst': diff_lst,
            'delta_uhi': diff_uhi
        },
        'statistics': {
            'p_mannwhitney_lst': float(p_mw_lst) if not np.isnan(p_mw_lst) else None,
            'p_mannwhitney_uhi': float(p_mw_uhi) if not np.isnan(p_mw_uhi) else None,
            'cohens_d_lst': float(cohens_d_lst) if not np.isnan(cohens_d_lst) else None,
            'cohens_d_uhi': float(cohens_d_uhi) if not np.isnan(cohens_d_uhi) else None,
        },
        'interpretation': interpretation,
    }

    # UHI intensity data if available
    if has_uhi_intensity:
        result['samples_change']['delta_uhi_intensity'] = delta_uhi_intensity_change
        result['samples_change']['uhi_intensity_t1'] = uhi_intensity_t1_change
        result['samples_change']['uhi_intensity_t2'] = uhi_intensity_t2_change

        result['samples_no_change']['delta_uhi_intensity'] = delta_uhi_intensity_no_change
        result['samples_no_change']['uhi_intensity_t1'] = uhi_intensity_t1_no_change
        result['samples_no_change']['uhi_intensity_t2'] = uhi_intensity_t2_no_change

        if len(delta_uhi_intensity_change) > 0 and len(delta_uhi_intensity_no_change) > 0:
            result['differences']['delta_uhi_intensity'] = diff_uhi_intensity

    return result


# ==============================================================================
# 6.Y INTERACTIVE ZONAL ANALYSIS MAP (CORRECTED)
# ==============================================================================

def create_interactive_zonal_map(
    results_dict,
    zonal_results,
    binary_change_mask,
    roi,
    sentinel_t1_image,
    sentinel_t2_image,
    amostras=3000,
    zoom=15,
    roi_name='study_area',
    mask_ee=None,
    lst_t1_img=None,
    lst_t2_img=None,
    lst_vis=None,
    lst_t1_label='T1',
    lst_t2_label='T2',
    percentile_threshold=97,
    verbose=False,
    show_CDI=True,
):
    """
    Creates an interactive map showing the change mask and filtered deltas.

    Args:
        mask_ee (ee.Image, optional): Mask already converted to ee.Image.
            If provided, skips numpy→GEE conversion. Create via cell 8.3.
    """
    _print = print if verbose else lambda *a, **k: None

    _print("\n" + "="*80)
    _print("CREATING INTERACTIVE MAP - ZONAL ANALYSIS")
    _print("="*80)
    _print(f"📍 Area: {roi_name}")
    _print(f"🎭 Mask: {zonal_results['mask_name']}")
    _print(f"📊 Index: {zonal_results['index_name']}")

    # Check classification availability
    has_uhi_intensity = ('uhi_intensity_t1' in results_dict['images'] and
                         'uhi_intensity_t2' in results_dict['images'] and
                         'delta_uhi_intensity' in results_dict['images'])

    if has_uhi_intensity:
        _print(f"   ✓ UHI intensity classification available")

    # ==============================================================================
    # 1. PREPARE MASK AS GEE RASTER
    # ==============================================================================

    if mask_ee is not None:
        _print("\n[1/4] Using pre-computed ee.Image mask (mask_ee)...")
        mask_viz = mask_ee.rename('mask')
    else:
        _print("\n[1/4] Converting mask to GEE raster...")
        mask_viz = _numpy_mask_to_ee_image(binary_change_mask, roi, band_name='mask')

    # ==============================================================================
    # 2. PREPARE MASKED DELTAS
    # ==============================================================================

    _print("\n[2/4] Preparing masked deltas...")

    # Extract images
    delta_index = results_dict['images']['delta_index']
    delta_lst = results_dict['images']['delta_lst']
    delta_uhi = results_dict['images']['delta_uhi']
    _index_t1_img = results_dict['images'].get('index_t1')
    _index_t2_img = results_dict['images'].get('index_t2')

    # Create binary mask from raster
    mask_binary = mask_viz.gt(0)

    # Apply mask to deltas and absolute indices
    delta_index_masked = delta_index.updateMask(mask_binary)
    delta_lst_masked = delta_lst.updateMask(mask_binary)
    delta_uhi_masked = delta_uhi.updateMask(mask_binary)
    _index_t1_masked = _index_t1_img.updateMask(mask_binary) if _index_t1_img is not None else None
    _index_t2_masked = _index_t2_img.updateMask(mask_binary) if _index_t2_img is not None else None

    _print(f"   ✓ Basic deltas masked")

    # Prepare intensity classifications if available
    if has_uhi_intensity:
        uhi_intensity_t1 = results_dict['images']['uhi_intensity_t1']
        uhi_intensity_t2 = results_dict['images']['uhi_intensity_t2']
        delta_uhi_intensity = results_dict['images']['delta_uhi_intensity']

        uhi_intensity_t1_masked = uhi_intensity_t1.updateMask(mask_binary)
        uhi_intensity_t2_masked = uhi_intensity_t2.updateMask(mask_binary)
        delta_uhi_intensity_masked = delta_uhi_intensity.updateMask(mask_binary)

        _print(f"   ✓ UHI intensity classifications masked")

    # ==============================================================================
    # 3. PREPARE VISUALIZATION PARAMETERS
    # ==============================================================================

    _print("\n[3/4] Preparing visualization parameters...")

    # Get statistics for palettes
    stats_delta_index = results_dict['statistics']['delta_index']
    stats_delta_lst = results_dict['statistics']['delta_lst']
    stats_delta_uhi = results_dict['statistics']['delta_uhi']

    # Compute symmetric ranges
    delta_index_max = max(
        abs(stats_delta_index.get('delta_index_min', -0.5)),
        abs(stats_delta_index.get('delta_index_max', 0.5))
    )

    delta_lst_max = max(
        abs(stats_delta_lst.get('delta_lst_min', -5)),
        abs(stats_delta_lst.get('delta_lst_max', 5))
    )

    delta_uhi_max = max(
        abs(stats_delta_uhi.get('delta_uhi_min', -2)),
        abs(stats_delta_uhi.get('delta_uhi_max', 2))
    )

    # ── Palettes (aligned with 8.1/8.2 — distinct from UHI blue→white→red) ────
    _div_rdb       = ['b2182b', 'ef8a62', 'fddbc7', 'f7f7f7', 'd1e5f0', '67a9cf', '2166ac']
    _VEGE_INDICES  = {'NDVI', 'EVI', 'SAVI', 'LAI', 'BNIRV', 'NIRV', 'EVI2', 'MSAVI'}
    _URBAN_INDICES = {'NDBI', 'IBI', 'EMBI', 'BUI', 'MNDBI', 'NBI', 'VIBI'}
    _idx_name_up   = zonal_results['index_name'].upper()

    # Absolute index — BrBG / PRGn / RdYlGn (same as 8.2)
    _PAL_VEGE_ABS  = ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']
    _PAL_URBAN_ABS = ['543005', 'bf812d', 'f6e8c3', 'f5f5f5', '80cdc1', '35978f', '003c30']
    _PAL_OTHER_ABS = ['40004b', '9970ab', 'c2a5cf', 'f7f7f7', 'a6dba0', '5aae61', '00441b']
    if _idx_name_up in _VEGE_INDICES:
        _idx_pal = _PAL_VEGE_ABS
    elif _idx_name_up in _URBAN_INDICES:
        _idx_pal = _PAL_URBAN_ABS
    else:
        _idx_pal = _PAL_OTHER_ABS

    # Delta index — RdYlGn for veg, PuGn for others (distinct from UHI)
    if _idx_name_up in _VEGE_INDICES:
        _delta_idx_pal5 = ['d73027', 'f46d43', 'ffffbf', 'a6d96a', '1a9850']
    else:
        _delta_idx_pal5 = ['7b3294', 'c2a5cf', 'f7f7f7', 'a6dba0', '1b7837']

    # Δ LST — classic blue→red (cooling→warming)
    _delta_lst_pal5 = ['2166ac', 'd1e5f0', 'f7f7f7', 'fddbc7', 'b2182b']

    # CDI — red→white→green (same as visualization.py)
    _cdi_pal5 = ['b2182b', 'fddbc7', 'f7f7f7', 'a6d96a', '1a9850']

    # Range for absolute index: global min/max across T1 and T2 for consistent stretch
    import numpy as _np_zi
    _sc  = zonal_results.get('samples_change', {})
    _snc = zonal_results.get('samples_no_change', {})
    _idx_t1_s = list(_sc.get('index_t1', [])) + list(_snc.get('index_t1', []))
    _idx_t2_s = list(_sc.get('index_t2', [])) + list(_snc.get('index_t2', []))
    if len(_idx_t1_s) > 10 and len(_idx_t2_s) > 10:
        _idx_min = float(min(_np_zi.percentile(_idx_t1_s, 2),  _np_zi.percentile(_idx_t2_s, 2)))
        _idx_max = float(max(_np_zi.percentile(_idx_t1_s, 98), _np_zi.percentile(_idx_t2_s, 98)))
    elif len(_idx_t1_s) > 10:
        _idx_min = float(_np_zi.percentile(_idx_t1_s, 2))
        _idx_max = float(_np_zi.percentile(_idx_t1_s, 98))
    else:
        _idx_min, _idx_max = -1.0, 1.0
    vis_index_abs = {'min': _idx_min, 'max': _idx_max, 'palette': _idx_pal}

    # Visualization parameters
    vis_delta_index = {
        'min': -delta_index_max,
        'max': delta_index_max,
        'palette': _delta_idx_pal5,
    }

    vis_delta_lst = {
        'min': -delta_lst_max,
        'max': delta_lst_max,
        'palette': list(reversed(_div_rdb)),
    }

    vis_delta_uhi = {
        'min': -delta_uhi_max,
        'max': delta_uhi_max,
        'palette': list(reversed(_div_rdb)),
    }

    # CDI range from sample-derived deltas (percentile-based, same approach as 8.1)
    _sc_idx_t1 = _np_zi.array(list(_sc.get('index_t1', [])))
    _sc_idx_t2 = _np_zi.array(list(_sc.get('index_t2', [])))
    _sc_lst_t1 = _np_zi.array(list(_sc.get('lst_t1', [])))
    _sc_lst_t2 = _np_zi.array(list(_sc.get('lst_t2', [])))
    _have_di = len(_sc_idx_t1) > 10 and len(_sc_idx_t1) == len(_sc_idx_t2)
    _have_dl = len(_sc_lst_t1) > 10 and len(_sc_lst_t1) == len(_sc_lst_t2)
    if _have_di and _have_dl and len(_sc_idx_t1) == len(_sc_lst_t1):
        _cdi_s = (_sc_idx_t2 - _sc_idx_t1) * (_sc_lst_t2 - _sc_lst_t1)
        _cdi_max = max(float(_np_zi.percentile(_np_zi.abs(_cdi_s), percentile_threshold)), 0.05)
    else:
        # fallback: product of individual percentile ranges
        _dmax_fb = float(_np_zi.percentile(_np_zi.abs(_sc_idx_t2 - _sc_idx_t1), percentile_threshold)) \
                   if _have_di else delta_index_max
        _lmax_fb = float(_np_zi.percentile(_np_zi.abs(_sc_lst_t2 - _sc_lst_t1), percentile_threshold)) \
                   if _have_dl else delta_lst_max
        _cdi_max = max(_dmax_fb * _lmax_fb, 0.05)
    vis_cdi = {'min': -_cdi_max, 'max': _cdi_max, 'palette': _cdi_pal5}

    # Absolute LST palette (fallback used when lst_vis is not provided)
    _lst_pal_abs = ['040274', '0602ff', '307ef3', '30c8e2', '3ff38f',
                    'fff705', 'ffb613', 'ff6e08', 'ff0000', 'a71001']
    _lst_vis_eff = lst_vis if lst_vis is not None else {
        'min': 20, 'max': 60, 'palette': _lst_pal_abs
    }

    # Mask with higher contrast
    vis_mask = {
        'min': 0,
        'max': 1,
        'palette': ['000000', 'FFFF00']  # Bright yellow
    }

    # Intensity parameters
    vis_uhi_intensity = {
        'min': 1,
        'max': 5,
        'palette': ['313695', '74add1', 'ffffbf', 'fdae61', 'a50026']
    }

    vis_delta_uhi_intensity = {
        'min': -2,
        'max': 2,
        'palette': 'RdBu_r'
    }

    _print(f"   ✓ Parameters configured")

    # ==============================================================================
    # 4. CREATE INTERACTIVE MAP
    # ==============================================================================

    _print("\n[4/4] Building interactive map...")

    Map = geemap.Map(basemap="SATELLITE", height=1200)
    Map.layers[0].opacity = 0.3

    # Center map
    center = roi.centroid(maxError=1).getInfo()['coordinates']
    Map.setCenter(center[0], center[1], zoom)

    # ===== GROUP 1: BASE IMAGES =====
    Map.addLayer(
        sentinel_t1_image.select(['B4', 'B3', 'B2']).clip(roi),
        {'min': 0, 'max': 0.3, 'gamma': 1.3},
        '🛰️ Sentinel-2 RGB T1',
        True
    )

    Map.addLayer(
        sentinel_t2_image.select(['B4', 'B3', 'B2']).clip(roi),
        {'min': 0, 'max': 0.3, 'gamma': 1.3},
        '🛰️ Sentinel-2 RGB T2',
        True
    )

    # ===== GROUP 2: ROI AND MASK =====
    Map.addLayer(roi, {'color': 'black'}, '🔲 ROI', True, 0.95)

    Map.addLayer(
        mask_viz,
        vis_mask,
        f'🎭 Mask: {zonal_results["mask_name"]}',
        True,
        0.4
    )

    # ===== GROUP 3: ABSOLUTE SPECTRAL INDEX T1 / T2 (masked by change area) =====
    if _index_t1_masked is not None:
        Map.addLayer(
            _index_t1_masked.clip(roi),
            vis_index_abs,
            f'🌿 {_idx_name_up} T1 — Change',
            False
        )
    if _index_t2_masked is not None:
        Map.addLayer(
            _index_t2_masked.clip(roi),
            vis_index_abs,
            f'🌿 {_idx_name_up} T2 — Change',
            False
        )

    # ===== GROUP 3.5: ABSOLUTE LST T1 / T2 (masked by change area) =====
    if lst_t1_img is not None:
        Map.addLayer(
            lst_t1_img.updateMask(mask_binary).clip(roi),
            _lst_vis_eff,
            f'🌡️ LST T1 (°C) [{lst_t1_label}] — Change',
            False
        )
    if lst_t2_img is not None:
        Map.addLayer(
            lst_t2_img.updateMask(mask_binary).clip(roi),
            _lst_vis_eff,
            f'🌡️ LST T2 (°C) [{lst_t2_label}] — Change',
            False
        )

    # ===== GROUP 4: MASKED DELTAS =====
    Map.addLayer(
        delta_index_masked.clip(roi),
        vis_delta_index,
        f'🔍 Δ {zonal_results["index_name"].upper()} (Change Only)',
        False
    )

    Map.addLayer(
        delta_lst_masked.clip(roi),
        vis_delta_lst,
        '🔥 Δ LST (Change Only)',
        False
    )

    if show_CDI:
        cdi_masked = delta_lst_masked.multiply(delta_index_masked).rename('CDI')
        Map.addLayer(
            cdi_masked.clip(roi),
            vis_cdi,
            f'📐 CDI (Change Only)',
            False
        )

    Map.addLayer(
        delta_uhi_masked.clip(roi),
        vis_delta_uhi,
        '🌆 Δ UHI (Change Only)',
        True
    )

    # ===== GROUPS 5-8: CLASSIFICATIONS =====
    if has_uhi_intensity:
        Map.addLayer(
            delta_uhi_intensity.clip(roi),
            vis_delta_uhi_intensity,
            '📈 Δ UHI Classif. (Full)',
            False
        )

        Map.addLayer(
            delta_uhi_intensity_masked.clip(roi),
            vis_delta_uhi_intensity,
            '📈 Δ UHI Classif. (Change Only)',
            False
        )

    # ===== LEGENDS =====

    def _mpl_colorbar_html(palette_hex, bounds, abbrevs, ivals, title):
        """Discrete matplotlib colorbar as base64 PNG for map embedding (same as 8.1)."""
        import matplotlib.colors as _mc
        import matplotlib.pyplot as _plt
        import io, base64
        colors = ['#' + c if not c.startswith('#') else c for c in palette_hex]
        cmap = _mc.ListedColormap(colors)
        norm = _mc.BoundaryNorm(bounds, ncolors=len(colors))
        ticks = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(colors))]
        tick_lbs = [f'{a}\n{v}' for a, v in zip(abbrevs, ivals)]
        fig, ax = _plt.subplots(figsize=(8.5, 0.90))
        fig.patch.set_alpha(0.0)
        sm = _plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, cax=ax, orientation='horizontal', ticks=ticks)
        cb.set_ticklabels(tick_lbs)
        cb.ax.tick_params(labelsize=11, length=0, pad=3)
        cb.set_label(title, fontsize=14, labelpad=5, fontweight='bold')
        cb.outline.set_linewidth(0.5)
        ax.set_facecolor('none')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', transparent=True)
        _plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:430px;">'

    # ── bottomleft: Change Mask + LST colorbar + Index colorbar ──────────────
    Map.add_legend(
        title='Change Mask',
        legend_dict={'Other': '000000', 'New Buildings': 'FFFF00'},
        position='bottomleft'
    )

    try:
        Map.add_colorbar(
            _lst_vis_eff,
            label='Land Surface Temperature — LST (°C)',
            orientation='horizontal',
            transparent_bg=True,
            position='bottomleft',
        )
    except Exception:
        pass

    try:
        Map.add_colorbar(
            vis_index_abs,
            label=f'{_idx_name_up} (shared scale: T1 ∪ T2 range)',
            orientation='horizontal',
            transparent_bg=True,
            position='bottomleft',
        )
    except Exception:
        pass

    # ── bottomright: Δ index, Δ LST, CDI discrete colorbars (same as 8.1) ───
    # Symmetric proportional bounds for 5 classes
    _b_idx = [round(-delta_index_max + i * 0.4 * delta_index_max, 6) for i in range(6)]
    _b_lst = [round(-delta_lst_max   + i * 0.4 * delta_lst_max,   4) for i in range(6)]
    if show_CDI:
        _b_cdi = [round(-_cdi_max + i * 0.4 * _cdi_max, 3) for i in range(6)]

    _abbr_di = ['SL', 'ML', 'Stb', 'MG', 'SG'] if _idx_name_up in _VEGE_INDICES \
               else ['SD', 'MD', 'Stb', 'MI', 'SI']
    _ivals_di = [
        f'< {_b_idx[1]:.3f}',
        f'{_b_idx[1]:.3f} to {_b_idx[2]:.3f}',
        f'{_b_idx[2]:.3f} to {_b_idx[3]:.3f}',
        f'{_b_idx[3]:.3f} to {_b_idx[4]:.3f}',
        f'> {_b_idx[4]:.3f}',
    ]
    try:
        Map.add_html(
            _mpl_colorbar_html(_delta_idx_pal5, _b_idx, _abbr_di, _ivals_di,
                               f'Δ {_idx_name_up} (T2 − T1)'),
            position='bottomright'
        )
    except Exception:
        pass

    _abbr_dlst  = ['SC',  'MC',  'Stb', 'MW',  'SW']
    _ivals_dlst = [
        f'< {_b_lst[1]:.1f} °C',
        f'{_b_lst[1]:.1f} to {_b_lst[2]:.1f} °C',
        f'{_b_lst[2]:.1f} to {_b_lst[3]:.1f} °C',
        f'{_b_lst[3]:.1f} to {_b_lst[4]:.1f} °C',
        f'> {_b_lst[4]:.1f} °C',
    ]
    try:
        Map.add_html(
            _mpl_colorbar_html(_delta_lst_pal5, _b_lst, _abbr_dlst, _ivals_dlst,
                               'Δ LST °C (T2 − T1)'),
            position='bottomright'
        )
    except Exception:
        pass

    if show_CDI:
        _abbr_cdi  = ['SN+',  'SNm',  'NA',  'SPm',  'SP+']
        _ivals_cdi = [
            f'< {_b_cdi[1]:.3f}',
            f'{_b_cdi[1]:.3f} to {_b_cdi[2]:.3f}',
            f'{_b_cdi[2]:.3f} to {_b_cdi[3]:.3f}',
            f'{_b_cdi[3]:.3f} to {_b_cdi[4]:.3f}',
            f'> {_b_cdi[4]:.3f}',
        ]
        try:
            Map.add_html(
                _mpl_colorbar_html(_cdi_pal5, _b_cdi, _abbr_cdi, _ivals_cdi,
                                   f'CDI = Δ LST × Δ {_idx_name_up}'),
                position='bottomright'
            )
        except Exception:
            pass

    Map.add_legend(
        title='UHI Intensity Classes (Deng et al., 2023)',
        legend_dict={
            '1 - LTZ (Low Temp.)':  '313695',
            '2 - SLTZ (Sub-Low)':   '74add1',
            '3 - MTZ (Medium)':     'ffffbf',
            '4 - SHTZ (Sub-High)':  'fdae61',
            '5 - HTZ (High Temp.)': 'a50026',
        },
        position='bottomleft'
    )

    Map.add_html(
        """<div style="background:white;border:1px solid #aaa;border-radius:4px;
                       padding:4px 8px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.3);
                       font-family:sans-serif;line-height:1.1;">
             <div style="font-size:11px;font-weight:bold;color:#222;margin-bottom:1px;">N</div>
             <svg width="20" height="40" viewBox="0 0 20 40">
               <polygon points="10,1 2,20 10,16 18,20" fill="#222"/>
               <polygon points="10,39 2,20 10,24 18,20" fill="#aaa" stroke="#555" stroke-width="0.5"/>
             </svg>
           </div>""",
        position='topright'
    )

    Map.add_inspector()

    _print(f"\n✅ MAP CREATED SUCCESSFULLY")
    _print(f"   • Mask rendered as continuous raster via vectorization")

    return Map


# extract_and_plot_transect is imported from modulos.visualization (see top of file)
# -- do not redefine it here, it would silently diverge from that copy.

# ==============================================================================
# 6.Z GENERIC SPECTRAL INDEX CHANGE ANALYSIS
# ==============================================================================

# ==============================================================================
# 7. McHARG METHODOLOGY — WEIGHTED LAYER OVERLAY
# ==============================================================================
# Implementation of Ian McHarg's method (1969, Design with Nature) for
# building a composite index by weighted overlay of thematic layers,
# with support for continuous and discrete data.
# ==============================================================================


def _numpy_mask_to_ee_image(binary_mask, roi, band_name='mask', transform=None, crs=None):
    """
    Converts a numpy binary mask to a continuous ee.Image raster in GEE.

    Uses vectorization (rasterio.features.shapes) to convert contiguous
    pixel regions into polygons, which are then rasterized in GEE via
    ee.Image.paint() — producing complete raster coverage.

    Args:
        binary_mask (numpy.ndarray): Binary mask (0/1).
        roi (ee.Geometry): Region of interest.
        band_name (str): Name of the resulting band.
        transform (rasterio.Affine, optional): Real geospatial transform from GeoTIFF
            (returned by load_geotiff). When provided with `crs`, ensures correct
            mask alignment with the ROI.
        crs (rasterio.crs.CRS, optional): GeoTIFF CRS. Polygons are reprojected
            to WGS84 (EPSG:4326) before sending to GEE.

    Returns:
        ee.Image: Binary raster image in GEE (0 and 1), clipped to ROI.
    """
    from rasterio.features import shapes as rasterio_shapes
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_geom as _transform_geom

    if transform is not None and crs is not None:
        # Use real GeoTIFF transform and CRS for correct positioning
        _affine = transform
        _src_crs = crs
        try:
            _need_reproject = crs.to_epsg() != 4326
        except Exception:
            _need_reproject = True  # reproject as safety fallback if EPSG cannot be determined
    else:
        # Fallback: maps pixels uniformly to ROI bounds in WGS84
        roi_coords = roi.bounds().getInfo()['coordinates'][0]
        west  = min(c[0] for c in roi_coords)
        south = min(c[1] for c in roi_coords)
        east  = max(c[0] for c in roi_coords)
        north = max(c[1] for c in roi_coords)
        _affine = from_bounds(west, south, east, north,
                              binary_mask.shape[1], binary_mask.shape[0])
        _src_crs = 'EPSG:4326'
        _need_reproject = False

    # Prepare mask: replace NaN with 0, convert to int16
    mask_clean = binary_mask.copy()
    mask_clean[np.isnan(mask_clean)] = 0
    mask_clean = mask_clean.astype(np.int16)

    # Vectorize: convert contiguous pixel regions to GeoJSON polygons
    print(f"   → Vectorizing mask ({mask_clean.shape}) to polygons...")
    change_polys = []
    for geom, val in rasterio_shapes(mask_clean, transform=_affine):
        if val == 1:
            if _need_reproject:
                geom = _transform_geom(_src_crs, 'EPSG:4326', geom)
            change_polys.append(ee.Feature(ee.Geometry(geom)))

    if not change_polys:
        print(f"   ⚠️ No change pixels found in mask!")
        return ee.Image.constant(0).toFloat().rename(band_name).clip(roi)

    print(f"   ✓ {len(change_polys)} polygons vectorized")

    # Limit number of features if necessary
    max_features = 50000
    if len(change_polys) > max_features:
        print(f"   ⚠️ Reducing from {len(change_polys)} to {max_features} polygons...")
        step = len(change_polys) // max_features
        change_polys = change_polys[::step]

    mask_fc = ee.FeatureCollection(change_polys)

    # Rasterize: paint 1 inside polygons, 0 outside
    mask_image = (
        ee.Image.constant(0)
        .toFloat()
        .paint(featureCollection=mask_fc, color=1)
        .clip(roi)
        .rename(band_name)
    )

    print(f"   ✓ Mask converted to ee.Image raster (band: '{band_name}')")
    return mask_image


# ==============================================================================
# 8. ZONAL ANALYSIS BY UHI CLASSES
# ==============================================================================
# Analysis of morphological composition (spectral indices) and thermal (LST)
# in each UHI intensity zone (Deng et al., 2023).
# ==============================================================================

# UHI zone names (1-5)
_UHI_ZONE_NAMES = {
    1: 'LTZ\n(Low)',
    2: 'SLTZ\n(Sub-Low)',
    3: 'MTZ\n(Medium)',
    4: 'SHTZ\n(Sub-High)',
    5: 'HTZ\n(High)',
}

_UHI_ZONE_COLORS = {
    1: '#2166ac',
    2: '#67a9cf',
    3: '#f7f7f7',
    4: '#ef8a62',
    5: '#b2182b',
}


def _create_uhi_classification(uhi_zscore_image):
    """
    Creates UHI classification image (1-5) from z-score.
    Deng et al. (2023):
      1-LTZ: UHI < -1 | 2-SLTZ: -1<=UHI<-0.5 | 3-MTZ: -0.5<=UHI<=0.5
      4-SHTZ: 0.5<UHI<=1 | 5-HTZ: UHI > 1
    """
    return (
        ee.Image(1)
        .where(uhi_zscore_image.gte(-1).And(uhi_zscore_image.lt(-0.5)), 2)
        .where(uhi_zscore_image.gte(-0.5).And(uhi_zscore_image.lte(0.5)), 3)
        .where(uhi_zscore_image.gt(0.5).And(uhi_zscore_image.lte(1)), 4)
        .where(uhi_zscore_image.gt(1), 5)
    ).rename('uhi_zone').toInt()


def analyze_uhi_zones(uhi_image, lst_image, indices_dict, roi,
                      scale=10, n_samples=5000):
    """
    Analyzes morphological and thermal composition per UHI zone.

    Classifies the UHI image (z-score) into 5 zones, samples LST and spectral
    indices via stratifiedSample, and returns data for boxplots.

    Args:
        uhi_image (ee.Image): UHI z-score image.
        lst_image (ee.Image): LST image in °C.
        indices_dict (dict): {index_name: ee.Image} for each index.
        roi (ee.Geometry): Region of interest.
        scale (int): Sampling scale in meters.
        n_samples (int): Total number of samples (distributed across zones).

    Returns:
        dict: {
            'zone_data': {zone_id: {band_name: [values]}},
            'zone_names': {zone_id: label},
            'band_names': [band_name, ...],
            'n_total': int,
            'uhi_classified': ee.Image
        }
    """
    print('\n' + '=' * 80)
    print('ZONAL ANALYSIS BY UHI CLASSES (Deng et al., 2023)')
    print('=' * 80)

    # 1. Classify UHI
    uhi_classified = _create_uhi_classification(uhi_image)

    # 2. Build multi-band image
    bands = [uhi_classified, lst_image.rename('LST')]
    band_names = ['LST']

    for idx_name, idx_img in indices_dict.items():
        band_name = idx_name.upper()
        bands.append(idx_img.rename(band_name))
        band_names.append(band_name)

    multi = ee.Image.cat(bands).clip(roi)

    print(f'   Bands for sampling: uhi_zone, LST, {", ".join(band_names[1:])}')
    print(f'   Requested samples: {n_samples}')

    # 3. Stratified sampling
    print('   Sampling via stratifiedSample...')
    samples_per_zone = max(n_samples // 5, 100)

    samples = multi.stratifiedSample(
        numPoints=samples_per_zone,
        classBand='uhi_zone',
        region=roi,
        scale=scale,
        geometries=False,
        seed=42
    )

    data_raw = samples.getInfo()
    features = data_raw.get('features', [])
    print(f'   ✓ {len(features)} samples retrieved')

    if not features:
        print('   ❌ No samples retrieved!')
        return None

    # 4. Organize data by zone
    zone_data = {z: {bn: [] for bn in band_names} for z in range(1, 6)}

    for f in features:
        props = f['properties']
        zone = int(props.get('uhi_zone', 0))
        if zone < 1 or zone > 5:
            continue
        for bn in band_names:
            val = props.get(bn)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                zone_data[zone][bn].append(val)

    # Converter para arrays
    for z in range(1, 6):
        for bn in band_names:
            zone_data[z][bn] = np.array(zone_data[z][bn])

    # 5. Print statistics
    print(f'\n{"─" * 80}')
    print(f'{"Zone":<15}', end='')
    for bn in band_names:
        print(f'{bn:>12}', end='')
    print(f'{"N":>8}')
    print(f'{"─" * 80}')

    for z in range(1, 6):
        zone_label = _UHI_ZONE_NAMES[z].replace('\n', ' ')
        n = len(zone_data[z]['LST'])
        print(f'{zone_label:<15}', end='')
        for bn in band_names:
            arr = zone_data[z][bn]
            if len(arr) > 0:
                print(f'{np.mean(arr):>12.4f}', end='')
            else:
                print(f'{"N/A":>12}', end='')
        print(f'{n:>8}')

    print(f'{"─" * 80}')

    return {
        'zone_data': zone_data,
        'zone_names': _UHI_ZONE_NAMES,
        'zone_colors': _UHI_ZONE_COLORS,
        'band_names': band_names,
        'n_total': len(features),
        'uhi_classified': uhi_classified,
    }


def compute_global_band_limits(result_t1, result_t2):
    """
    Compute unified Y-axis limits for each band from two analyze_uhi_zones results.

    Merges the sampled distributions of T1 and T2 across all zones per band and
    returns the global (min, max) for each band, padded by 10 % of the range.
    Passing these limits to plot_uhi_zones_boxplots guarantees identical Y-axis
    scales whether the cell is run for T1 or T2, enabling direct visual comparison
    across separate execution sessions.

    Args:
        result_t1 (dict): Output of analyze_uhi_zones() for period T1.
        result_t2 (dict): Output of analyze_uhi_zones() for period T2.

    Returns:
        dict: {band_name: (ymin, ymax)} for every band present in both results.
    """
    limits = {}
    bands = result_t1['band_names']

    for bn in bands:
        whisker_lo_vals = []
        whisker_hi_vals = []

        for period_result in (result_t1, result_t2):
            for z in range(1, 6):
                arr = np.asarray(period_result['zone_data'][z].get(bn, []))
                arr = arr[np.isfinite(arr)]
                if len(arr) <= 1:
                    continue
                # Per-zone whisker: last data point within each zone's own IQR fence
                # (mirrors exactly what matplotlib boxplot draws with showfliers=False)
                q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
                iqr    = q3 - q1
                fence_lo, fence_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                in_fence = arr[(arr >= fence_lo) & (arr <= fence_hi)]
                if len(in_fence) == 0:
                    continue
                whisker_lo_vals.append(in_fence.min())
                whisker_hi_vals.append(in_fence.max())

        if not whisker_lo_vals:
            continue

        vmin = min(whisker_lo_vals)
        vmax = max(whisker_hi_vals)
        pad  = max((vmax - vmin) * 0.12, 0.01)
        limits[bn] = (vmin - pad, vmax + pad)

    return limits


def plot_uhi_zones_boxplots(analysis_result, periodo='T2', band_limits=None):
    """
    Generate boxplots of LST and spectral indices by UHI intensity zone.

    One panel per variable, UHI zones on the X-axis. A diamond marker shows
    the mean; whiskers follow the 1.5 × IQR rule (no fliers plotted).

    Y-axis scale modes
    ------------------
    - T1 / T2 with band_limits provided : fixed limits from compute_global_band_limits(),
      guaranteeing identical scales across separate execution sessions so that
      T1 and T2 plots are directly comparable.
    - T1 / T2 without band_limits        : auto-scale from current period data
      (whisker envelope for LST, matplotlib default for indices).
    - delta                              : symmetric scale around 0 (± max_abs)
      computed from the current delta distribution, regardless of band_limits.
      This keeps zero centred and gives equal visual weight to gains and losses.

    Args:
        analysis_result (dict): Output of analyze_uhi_zones().
        periodo (str): 'T1', 'T2' or 'delta' (used in titles).
        band_limits (dict | None): {band_name: (ymin, ymax)} from
            compute_global_band_limits(). When supplied for T1/T2, these limits
            are applied directly. Ignored for delta.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    zone_data   = analysis_result['zone_data']
    band_names  = analysis_result['band_names']
    zone_names  = analysis_result['zone_names']
    zone_colors = analysis_result['zone_colors']

    n_bands   = len(band_names)
    n_cols    = min(3, n_bands)
    n_rows_bp = (n_bands + n_cols - 1) // n_cols

    periodo_label = periodo.upper() if periodo != 'delta' else 'Δ (T2-T1)'
    is_delta      = periodo == 'delta'

    fig_w = max(18, 6 * n_cols)
    fig_h = 5 * n_rows_bp

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        n_rows_bp, n_cols, figure=fig,
        hspace=0.28, wspace=0.30,
    )

    axes_bp = []
    for idx in range(n_rows_bp * n_cols):
        r, c = divmod(idx, n_cols)
        axes_bp.append(fig.add_subplot(gs[r, c]))

    _ZONE_SHORT = {1: 'LTZ', 2: 'SLTZ', 3: 'MTZ', 4: 'SHTZ', 5: 'HTZ'}

    for i, bn in enumerate(band_names):
        ax = axes_bp[i]

        data_list, tick_labels, colors = [], [], []
        for z in range(1, 6):
            arr = zone_data[z][bn]
            data_list.append(arr if len(arr) > 0 else np.array([0.0]))
            tick_labels.append(_ZONE_SHORT[z])
            colors.append(zone_colors[z])

        bp = ax.boxplot(
            data_list, labels=tick_labels, patch_artist=True,
            widths=0.6, showfliers=False,
            medianprops=dict(color='black', linewidth=2),
        )
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # ── Y-axis limits ──────────────────────────────────────────────────
        # Use actual rendered whisker extents to guarantee nothing is clipped
        whisker_vals = [v for w in bp['whiskers'] for v in w.get_ydata()]
        actual_lo = min(whisker_vals) if whisker_vals else 0.0
        actual_hi = max(whisker_vals) if whisker_vals else 1.0

        if is_delta:
            # Symmetric around 0 based on real whisker extent
            max_abs = max(abs(actual_lo), abs(actual_hi))
            pad     = max(max_abs * 0.15, 0.01)
            ax.set_ylim(-(max_abs + pad), max_abs + pad)
        elif band_limits and bn in band_limits:
            # Fixed global limits (already 10%-padded by compute_global_band_limits).
            # Do NOT widen per-period — that would break cross-session comparability.
            ax.set_ylim(band_limits[bn])
        else:
            min_pad = 1.0 if bn == 'LST' else 0.01
            pad = max((actual_hi - actual_lo) * 0.15, min_pad)
            ax.set_ylim(actual_lo - pad, actual_hi + pad)

        y_lo, y_hi = ax.get_ylim()
        if y_lo < 0 < y_hi:
            ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

        fmt = ticker.FormatStrFormatter('%.1f') if bn == 'LST' else ticker.FormatStrFormatter('%.2f')
        ax.yaxis.set_major_formatter(fmt)

        ylabel = ('LST (°C)' if bn == 'LST' and not is_delta
                  else 'Δ LST (°C)' if bn == 'LST'
                  else bn if not is_delta
                  else f'Δ {bn}')
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
        ax.set_title(f'{ylabel} by UHI Zone ({periodo_label})',
                     fontweight='bold', fontsize=12)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        for j, arr in enumerate(data_list):
            if len(arr) > 1 or arr[0] != 0:
                ax.scatter(j + 1, np.mean(arr), color='black',
                           marker='D', s=40, zorder=5)

        # Annotate scale source in bottom-right corner
        if not is_delta and band_limits and bn in band_limits:
            ax.annotate('scale: T1+T2', xy=(1, 0), xycoords='axes fraction',
                        fontsize=7, color='gray', ha='right', va='bottom',
                        xytext=(-4, 3), textcoords='offset points')

    for i in range(n_bands, len(axes_bp)):
        axes_bp[i].set_visible(False)

    fig.suptitle(f'Composition by UHI Zone — {periodo_label}',
                 fontsize=15, fontweight='bold', y=1.01)

    legend_patches = [
        mpatches.Patch(facecolor=zone_colors[z], alpha=0.75,
                       label=zone_names[z].replace('\n', ' '))
        for z in range(1, 6)
    ]
    legend_patches.append(
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
                   markersize=8, label='Mean')
    )
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    return fig


def plot_uhi_spectral_profile(analysis_result, periodo='T2'):
    """
    Generate a normalized spectral-profile line chart showing the characteristic
    signature of each UHI intensity class across LST and spectral indices.

    Each band is min-max normalized to [0, 1] across the 5 zones independently,
    making LST (°C) and dimensionless indices visually comparable while preserving
    the relative ordering of zones within each variable (Xiong et al., 2012).

    Args:
        analysis_result (dict): Output of analyze_uhi_zones().
        periodo (str): 'T1', 'T2' or 'delta' (used in titles).

    Returns:
        matplotlib.figure.Figure, or None if fewer than 2 bands are available.
    """
    import matplotlib.pyplot as plt

    zone_data   = analysis_result['zone_data']
    band_names  = analysis_result['band_names']
    zone_names  = analysis_result['zone_names']
    zone_colors = analysis_result['zone_colors']

    if len(band_names) < 2:
        return None

    _ZONE_SHORT   = {1: 'LTZ', 2: 'SLTZ', 3: 'MTZ', 4: 'SHTZ', 5: 'HTZ'}
    periodo_label = periodo.upper() if periodo != 'delta' else 'Δ (T2-T1)'

    # Median per zone per band
    medians = {}
    for bn in band_names:
        medians[bn] = np.array([
            float(np.median(zone_data[z][bn])) if len(zone_data[z][bn]) > 1
            else np.nan
            for z in range(1, 6)
        ])

    # Min-max normalize each band across the 5 zones
    normed = {}
    for bn in band_names:
        v = medians[bn]
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        normed[bn] = (v - vmin) / (vmax - vmin) if vmax > vmin else np.full(5, 0.5)

    # MTZ (#f7f7f7) is near-white — override only for this plot
    _colors = dict(zone_colors)
    _colors[3] = '#888888'

    fig_w = max(10, 2.5 * len(band_names))
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    x_pos = np.arange(len(band_names))
    for z in range(1, 6):
        y = [normed[bn][z - 1] for bn in band_names]
        ax.plot(x_pos, y, 'o-',
                color=_colors[z], linewidth=2.2, markersize=8,
                label=f'{_ZONE_SHORT[z]} — {zone_names[z].replace(chr(10), " ")}',
                zorder=z)

    xticklabels = ['Δ ' + bn if periodo == 'delta' else bn for bn in band_names]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xticklabels, fontsize=11)
    ax.set_ylim(-0.08, 1.08)
    ax.set_ylabel('Normalized median\n(0 = min zone  ·  1 = max zone)', fontsize=10)
    ax.set_title(
        f'Spectral Profile per UHI Zone — {periodo_label}\n'
        r'(Deng et al., 2023)',
        fontweight='bold', fontsize=12,
    )
    ax.grid(True, alpha=0.25)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhspan(0.65, 1.08, alpha=0.04, color='red')
    ax.axhspan(-0.08, 0.35, alpha=0.04, color='blue')

    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.14),
        ncol=3, fontsize=9,
        title='UHI Intensity Zone', title_fontsize=9,
        framealpha=0.9, edgecolor='#cccccc',
    )

    fig.suptitle(f'Spectral Profile by UHI Zone — {periodo_label}',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


def create_uhi_zones_map(uhi_classified, roi, sentinel_image, zoom=15,
                         periodo='T2', sentinel_image_t1=None,
                         indices_dict=None):
    """
    Create an interactive map with UHI zone classification.

    Layers included by mode:
      - T1 / T2 : Sentinel-2 RGB, UHI Classes, absolute spectral index
      - delta   : RGB T1 + T2, Delta UHI Classes, delta spectral index

    Args:
        uhi_classified (ee.Image): Classified image (1-5).
        roi (ee.Geometry): Region of interest.
        sentinel_image (ee.Image): Sentinel-2 image (T2 or T2 in delta mode).
        zoom (int): Zoom level.
        periodo (str): 'T1', 'T2' or 'delta'.
        sentinel_image_t1 (ee.Image, optional): Sentinel-2 T1 (delta mode).
        indices_dict (dict, optional): {idx_name: ee.Image} — absolute index (T1/T2)
            or delta index (delta).

    Returns:
        geemap.Map
    """
    Map = geemap.Map(basemap="SATELLITE", height=800)
    Map.layers[0].opacity = 0.3
    center = roi.centroid(maxError=1).getInfo()['coordinates']
    Map.setCenter(center[0], center[1], zoom)

    _URBAN_INDX  = {'NDBI', 'IBI', 'EMBI', 'BUI', 'MNDBI', 'NBI', 'VIBI'}
    _VEGE_INDX   = {'NDVI', 'EVI', 'SAVI', 'LAI', 'NIRV', 'BNIRV'}

    # Palettes distinct from UHI (blue→white→red) — used for index colorbars
    # Vegetation absolute: RdYlGn (red=low, green=high — conventional, distinct from UHI)
    _PAL_VEGE_ABS  = ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']
    # Urban absolute: BrBG (brown→teal — distinct from UHI blue→red)
    _PAL_URBAN_ABS = ['543005', 'bf812d', 'f6e8c3', 'f5f5f5', '80cdc1', '35978f', '003c30']
    # Other absolute: PRGn (purple→green — clearly distinct from UHI)
    _PAL_OTHER_ABS = ['40004b', '9970ab', 'c2a5cf', 'f7f7f7', 'a6dba0', '5aae61', '00441b']
    # Delta (all types): PuOr (purple→white→orange — distinct from UHI blue→white→red)
    _PAL_DELTA     = ['7b3294', 'c2a5cf', 'e7d4e8', 'f7f7f7', 'd9f0d3', 'a6dba0', '1b7837']

    # ── RGB ──────────────────────────────────────────────────────────────────
    if periodo == 'delta' and sentinel_image_t1 is not None:
        Map.addLayer(
            sentinel_image_t1.select(['B4', 'B3', 'B2']).clip(roi),
            {'min': 0, 'max': 0.3, 'gamma': 1.3},
            '🛰️ Sentinel-2 RGB T1', False
        )
        Map.addLayer(
            sentinel_image.select(['B4', 'B3', 'B2']).clip(roi),
            {'min': 0, 'max': 0.3, 'gamma': 1.3},
            '🛰️ Sentinel-2 RGB T2', True
        )
    else:
        Map.addLayer(
            sentinel_image.select(['B4', 'B3', 'B2']).clip(roi),
            {'min': 0, 'max': 0.3, 'gamma': 1.3},
            f'🛰️ Sentinel-2 RGB {periodo.upper()}', True
        )

    Map.addLayer(roi, {'color': 'black'}, 'ROI', True, 0.5)

    # ── Spectral indices (T1/T2: absolute; delta: Δ) ──────────────────────────
    _idx_colorbars = []  # [(colorbar_label, vis_params)]
    if indices_dict:
        for _idx_name, _idx_img in indices_dict.items():
            if _idx_img is None:
                continue
            try:
                _stats = _idx_img.reduceRegion(
                    reducer=ee.Reducer.percentile([2, 98]),
                    geometry=roi, scale=30, maxPixels=1e9, bestEffort=True
                ).getInfo() or {}
                _vals = [v for v in _stats.values() if v is not None]
            except Exception:
                _vals = []

            _iup = _idx_name.upper()
            if periodo == 'delta':
                _dmax = max(abs(min(_vals)), abs(max(_vals))) if _vals else 0.3
                _dmax = max(_dmax, 0.05)
                _vis = {'min': -_dmax, 'max': _dmax, 'palette': _PAL_DELTA}
                _layer_label = f'📊 Δ {_iup} (T2−T1)'
                _cb_label    = f'Δ {_iup} (T2 − T1)'
            else:
                _imin = min(_vals) if _vals else -1
                _imax = max(_vals) if _vals else 1
                if _iup in _VEGE_INDX:
                    _pal = _PAL_VEGE_ABS
                elif _iup in _URBAN_INDX:
                    _pal = _PAL_URBAN_ABS
                else:
                    _pal = _PAL_OTHER_ABS
                _vis = {'min': _imin, 'max': _imax, 'palette': _pal}
                _layer_label = f'📊 {_iup} ({periodo.upper()})'
                _cb_label    = f'{_iup} ({periodo.upper()})'

            Map.addLayer(_idx_img.clip(roi), _vis, _layer_label, False, 1)
            _idx_colorbars.append((_cb_label, _vis))

    # ── UHI Classes ──────────────────────────────────────────────────────────
    periodo_label = 'Δ UHI' if periodo == 'delta' else periodo.upper()
    Map.addLayer(
        uhi_classified.clip(roi),
        {'min': 1, 'max': 5, 'palette': ['2166ac', '67a9cf', 'f7f7f7', 'ef8a62', 'b2182b']},
        f'🏙️ UHI Classes ({periodo_label})', True, 0.5
    )

    # ── Legends ───────────────────────────────────────────────────────────────
    legend = {
        '1 - LTZ (Low Temp.)':    '2166ac',
        '2 - SLTZ (Sub-Low)':     '67a9cf',
        '3 - MTZ (Medium)':       'f7f7f7',
        '4 - SHTZ (Sub-High)':    'ef8a62',
        '5 - HTZ (High Temp.)':   'b2182b',
    }
    Map.add_legend(title=f'UHI Classes — {periodo_label}',
                  legend_dict=legend, position='bottomright')

    # Continuous colorbars for each spectral index — bottomleft (same style as cell 8.1)
    for _cb_label, _vis in _idx_colorbars:
        try:
            Map.add_colorbar(
                _vis,
                label=_cb_label,
                orientation='horizontal',
                transparent_bg=True,
                position='bottomleft',
            )
        except Exception:
            pass

    Map.add_html(
        """<div style="background:white;border:1px solid #aaa;border-radius:4px;
                       padding:4px 8px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.3);
                       font-family:sans-serif;line-height:1.1;">
             <div style="font-size:11px;font-weight:bold;color:#222;margin-bottom:1px;">N</div>
             <svg width="20" height="40" viewBox="0 0 20 40">
               <polygon points="10,1 2,20 10,16 18,20" fill="#222"/>
               <polygon points="10,39 2,20 10,24 18,20" fill="#aaa" stroke="#555" stroke-width="0.5"/>
             </svg>
           </div>""",
        position='topright'
    )

    Map.add_inspector()

    return Map


