# ==============================================================================
# visualization.py — Visualization Functions and Interactive Maps
# ==============================================================================
# Bitemporal LST/UHI processing, thermal change statistics, integrated
# spectral-index × LST × UHI analysis, plots and interactive maps.
# ==============================================================================

import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt

# --- Inter-module imports ---
from modulos.image_processing import (
    generate_sentinel1_composite_from_s2_date,
    get_landsat_collection, get_selected_landsat_image,
    get_selected_sentinel2_image, get_sentinel2_collection,
    process_landsat_image, process_landsat_median,
    process_sentinel2_image, process_sentinel2_median
)
from modulos.lst_analysis import (
    calculate_uhi,
    calculate_uhi_statistics, classify_uhi_areas,
    classify_uhi_intensity
)
from modulos.spectral_analysis import (
    calculate_spectral_indices_landsat,
    calculate_spectral_indices_sentinel2
)


def _process_period(config, s2_collection, label):
    """
    Loads and processes Landsat (LST 30 m + SR bands) and Sentinel-2 for one period.

    Uses the specific image IDs when both are present in the config; otherwise
    falls back to the median of the filtered collections.

    Returns:
        dict: s2 (ee.Image, 6 bands), s2_projection, ls_optical (ee.Image),
              lst_celsius (ee.Image, band ST_B10), indices_landsat, indices_sentinel2
    """
    print("\n" + "=" * 80)
    print(f"PROCESSING PERIOD {label}")
    print("=" * 80)

    if config.get('landsat_id') and config.get('sentinel2_id'):
        print(f"\n🎯 Using specific images for {label}")
        print(f"   Landsat ID: {config['landsat_id']}")
        print(f"   Sentinel-2 ID: {config['sentinel2_id']}")

        ls_image = get_selected_landsat_image(
            config['landsat_id'], config['geometry'], config['landsat_type']
        )
        s2_image = get_selected_sentinel2_image(
            config['sentinel2_id'], config['geometry'], s2_collection=s2_collection
        )
        s2, s2_proj = process_sentinel2_image(s2_image)
        ls_optical, lst_celsius, _ = process_landsat_image(ls_image, s2_proj)
    else:
        print(f"\n📊 Using collection median for {label}")
        s2, s2_proj = process_sentinel2_median(get_sentinel2_collection(config))
        ls_optical, lst_celsius, _ = process_landsat_median(get_landsat_collection(config), s2_proj)

    print(f"\n📊 Calculating Landsat spectral indices {label}...")
    indices_landsat = calculate_spectral_indices_landsat(ls_optical)
    print(f"📊 Calculating Sentinel-2 spectral indices {label}...")
    indices_sentinel2 = calculate_spectral_indices_sentinel2(s2)

    return {
        's2': s2,
        's2_projection': s2_proj,
        'ls_optical': ls_optical,
        'lst_celsius': lst_celsius,
        'indices_landsat': indices_landsat,
        'indices_sentinel2': indices_sentinel2,
    }


def calculate_bitemporal_lst(config_t1, config_t2,
                             s2_collection_t1='COPERNICUS/S2_SR_HARMONIZED',
                             s2_collection_t2='COPERNICUS/S2_SR_HARMONIZED'):
    """
    Bitemporal processing: Landsat LST at native 30 m, Sentinel-2 (10 m),
    Landsat and Sentinel-2 spectral indices, and Sentinel-1 SAR composites.

    Args:
        config_t1 (dict): Configuration for period T1 (create_tenst_config).
        config_t2 (dict): Configuration for period T2.
        s2_collection_t1 (str): Sentinel-2 collection used to load the T1 image ID.
        s2_collection_t2 (str): Sentinel-2 collection used to load the T2 image ID.

    Returns:
        dict: {'t1': {...}, 't2': {...}, 's2_median_t1/t2', 's1_t1/t2',
               'lst_30m_t1/t2', 'indices_landsat': {'t1','t2'},
               'indices_sentinel2': {'t1','t2'}}
    """
    print("\n" + "=" * 80)
    print("BITEMPORAL PROCESSING - LST 30m + SPECTRAL INDICES")
    print("=" * 80)

    p1 = _process_period(config_t1, s2_collection_t1, 'T1')
    p2 = _process_period(config_t2, s2_collection_t2, 'T2')

    # ── Sentinel-1 composites (input of the change detection model) ─────────
    print("\n" + "=" * 80)
    print("PROCESSING SENTINEL-1 SAR")
    print("=" * 80)

    s1 = {}
    for key, cfg in (('t1', config_t1), ('t2', config_t2)):
        s1[key] = None
        if cfg.get('sentinel2_id'):
            print(f"\n📡 Generating Sentinel-1 composite {key.upper()}...")
            try:
                s1[key] = generate_sentinel1_composite_from_s2_date(
                    cfg['geometry'], cfg['sentinel2_id'], polarizations=['VV', 'VH']
                )
                print(f"   ✓ Sentinel-1 {key.upper()} processed")
            except Exception as e:
                print(f"   ⚠️ Error processing Sentinel-1 {key.upper()}: {e}")

    def _period_result(cfg, p):
        return {
            'config': cfg,
            'lst_30m': p['lst_celsius'],
            'lst_celsius': p['lst_celsius'],
            's2_projection': p['s2_projection'],
            'sentinel2_image': p['s2'],
            'landsat_image': p['ls_optical'],
            'ls_optical': p['ls_optical'],
        }

    results = {
        't1': _period_result(config_t1, p1),
        't2': _period_result(config_t2, p2),
        's2_median_t1': p1['s2'],
        's2_median_t2': p2['s2'],
        's1_t1': s1['t1'],
        's1_t2': s1['t2'],
        'lst_30m_t1': p1['lst_celsius'],
        'lst_30m_t2': p2['lst_celsius'],
        'indices_landsat': {'t1': p1['indices_landsat'], 't2': p2['indices_landsat']},
        'indices_sentinel2': {'t1': p1['indices_sentinel2'], 't2': p2['indices_sentinel2']},
    }

    print("\n✅ Bitemporal processing complete!")
    return results


def calculate_thermal_change_statistics(results, geometry):
    """
    Calculate thermal change statistics between two periods.

    Args:
        results (dict): Bitemporal processing results
        geometry: Region of interest geometry

    Returns:
        dict: {'LST_30m': {...}, 'UHI_30m': {..., 'images', 'intensity_classification'}}
    """
    print("\n" + "=" * 80)
    print("THERMAL CHANGE STATISTICS")
    print("=" * 80)

    stats = {}

    # ====================================================================
    # PART 1: LST ANALYSIS (LAND SURFACE TEMPERATURE)
    # ====================================================================

    # LST 30m (Landsat)
    print("\n🌡️  LAND SURFACE TEMPERATURE - LST 30m (Landsat)")
    print("-" * 80)

    stats_30m_t1 = results['lst_30m_t1'].reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), '', True).combine(
            ee.Reducer.stdDev(), '', True
        ),
        geometry=geometry,
        scale=30,
        maxPixels=1e12
    ).getInfo()

    stats_30m_t2 = results['lst_30m_t2'].reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), '', True).combine(
            ee.Reducer.stdDev(), '', True
        ),
        geometry=geometry,
        scale=30,
        maxPixels=1e12
    ).getInfo()

    delta_30m = {
        'min': stats_30m_t2['ST_B10_min'] - stats_30m_t1['ST_B10_min'],
        'mean': stats_30m_t2['ST_B10_mean'] - stats_30m_t1['ST_B10_mean'],
        'max': stats_30m_t2['ST_B10_max'] - stats_30m_t1['ST_B10_max'],
        'stdDev': stats_30m_t2['ST_B10_stdDev'] - stats_30m_t1['ST_B10_stdDev']
    }

    print(f"   T1 - Min: {stats_30m_t1['ST_B10_min']:.2f}°C | "
          f"Mean: {stats_30m_t1['ST_B10_mean']:.2f}°C | "
          f"Max: {stats_30m_t1['ST_B10_max']:.2f}°C | "
          f"StdDev: {stats_30m_t1['ST_B10_stdDev']:.2f}°C")
    print(f"   T2 - Min: {stats_30m_t2['ST_B10_min']:.2f}°C | "
          f"Mean: {stats_30m_t2['ST_B10_mean']:.2f}°C | "
          f"Max: {stats_30m_t2['ST_B10_max']:.2f}°C | "
          f"StdDev: {stats_30m_t2['ST_B10_stdDev']:.2f}°C")
    print(f"   Δ  - Min: {delta_30m['min']:+.2f}°C | "
          f"Mean: {delta_30m['mean']:+.2f}°C | "
          f"Max: {delta_30m['max']:+.2f}°C | "
          f"StdDev: {delta_30m['stdDev']:+.2f}°C")

    stats['LST_30m'] = {
        't1': stats_30m_t1,
        't2': stats_30m_t2,
        'delta': delta_30m
    }

    # ====================================================================
    # PART 2: URBAN HEAT ISLAND ANALYSIS (NORMALIZED UHI)
    # ====================================================================

    print("\n" + "=" * 80)
    print("🔥 URBAN HEAT ISLAND ANALYSIS (NORMALIZED UHI)")
    print("=" * 80)
    print("\nNote: Normalized UHI expresses deviations relative to the regional mean")
    print("      UHI = 0: temperature equals mean | UHI > 0: warmer | UHI < 0: cooler")

    # UHI para LST 30m
    print("\n📊 UHI 30m (Landsat)")
    print("-" * 80)

    # T1 - 30m
    uhi_30m_t1, lst_stats_t1 = calculate_uhi(results['lst_30m_t1'], geometry=geometry, scale=30)
    uhi_stats_30m_t1_dict = calculate_uhi_statistics(uhi_30m_t1, geometry=geometry, scale=30)

    # Classify UHI intensity T1
    uhi_intensity_30m_t1 = classify_uhi_intensity(uhi_30m_t1)

    # Extract LST statistics for conversion (single server round trip)
    lst_mean_t1, lst_std_t1 = ee.List([lst_stats_t1['lst_mean'], lst_stats_t1['lst_std']]).getInfo()

    # T2 - 30m
    uhi_30m_t2, lst_stats_t2 = calculate_uhi(results['lst_30m_t2'], geometry=geometry, scale=30)
    uhi_stats_30m_t2_dict = calculate_uhi_statistics(uhi_30m_t2, geometry=geometry, scale=30)

    # Classify UHI intensity T2
    uhi_intensity_30m_t2 = classify_uhi_intensity(uhi_30m_t2)

    # Extract LST statistics for conversion (single server round trip)
    lst_mean_t2, lst_std_t2 = ee.List([lst_stats_t2['lst_mean'], lst_stats_t2['lst_std']]).getInfo()

    # Calculate equivalent temperatures for extreme UHI values
    uhi_min_t1 = uhi_stats_30m_t1_dict.get('UHI_min', -3)
    uhi_max_t1 = uhi_stats_30m_t1_dict.get('UHI_max', 3)
    temp_min_t1 = lst_mean_t1 + (uhi_min_t1 * lst_std_t1)
    temp_max_t1 = lst_mean_t1 + (uhi_max_t1 * lst_std_t1)

    uhi_min_t2 = uhi_stats_30m_t2_dict.get('UHI_min', -3)
    uhi_max_t2 = uhi_stats_30m_t2_dict.get('UHI_max', 3)
    temp_min_t2 = lst_mean_t2 + (uhi_min_t2 * lst_std_t2)
    temp_max_t2 = lst_mean_t2 + (uhi_max_t2 * lst_std_t2)

    # Print T1 results
    print(f"\n   T1:")
    print(f"      LST Base     : Mean = {lst_mean_t1:.2f}°C | StdDev = {lst_std_t1:.2f}°C")
    print(f"      UHI (z-score): Mean = {uhi_stats_30m_t1_dict.get('UHI_mean', 0):.3f} | "
          f"StdDev = {uhi_stats_30m_t1_dict.get('UHI_stdDev', 0):.3f}")
    print(f"      UHI Range: Min = {uhi_min_t1:.2f} ({temp_min_t1:.1f}°C) | "
          f"Max = {uhi_max_t1:.2f} ({temp_max_t1:.1f}°C)")

    # Print T2 results
    print(f"\n   T2:")
    print(f"      LST Base     : Mean = {lst_mean_t2:.2f}°C | StdDev = {lst_std_t2:.2f}°C")
    print(f"      UHI (z-score): Mean = {uhi_stats_30m_t2_dict.get('UHI_mean', 0):.3f} | "
          f"StdDev = {uhi_stats_30m_t2_dict.get('UHI_stdDev', 0):.3f}")
    print(f"      UHI Range: Min = {uhi_min_t2:.2f} ({temp_min_t2:.1f}°C) | "
          f"Max = {uhi_max_t2:.2f} ({temp_max_t2:.1f}°C)")

    # Change analysis
    delta_uhi_max = uhi_max_t2 - uhi_max_t1
    delta_temp_hotspot = temp_max_t2 - temp_max_t1

    print(f"\n   Change T1→T2:")
    print(f"      Δ UHI Max    : {delta_uhi_max:+.2f} (from {uhi_max_t1:.2f} to {uhi_max_t2:.2f})")
    print(f"      Δ Temp Hotspot: {delta_temp_hotspot:+.2f}°C (from {temp_max_t1:.1f}°C to {temp_max_t2:.1f}°C)")

    # Classify areas T1
    print(f"\n   Intensity Classification (T1) - Deng et al. (2023):")
    area_categories_t1 = classify_uhi_areas(uhi_30m_t1, geometry, scale=30)
    print(f"      1. LTZ  - Low Temperature (UHI < -1):          {area_categories_t1['LTZ']:.2f} km²")
    print(f"      2. SLTZ - Sub-Low Temp (-1 ≤ UHI < -0.5):      {area_categories_t1['SLTZ']:.2f} km²")
    print(f"      3. MTZ  - Medium Temp (-0.5 ≤ UHI ≤ 0.5):      {area_categories_t1['MTZ']:.2f} km²")
    print(f"      4. SHTZ - Sub-High Temp (0.5 < UHI ≤ 1):       {area_categories_t1['SHTZ']:.2f} km²")
    print(f"      5. HTZ  - High Temperature (UHI > 1):          {area_categories_t1['HTZ']:.2f} km²")
    print(f"      Total: {area_categories_t1['total']:.2f} km²")

    # Classify areas T2
    print(f"\n   Intensity Classification (T2) - Deng et al. (2023):")
    area_categories_t2 = classify_uhi_areas(uhi_30m_t2, geometry, scale=30)
    print(f"      1. LTZ  - Low Temperature (UHI < -1):          {area_categories_t2['LTZ']:.2f} km²")
    print(f"      2. SLTZ - Sub-Low Temp (-1 ≤ UHI < -0.5):      {area_categories_t2['SLTZ']:.2f} km²")
    print(f"      3. MTZ  - Medium Temp (-0.5 ≤ UHI ≤ 0.5):      {area_categories_t2['MTZ']:.2f} km²")
    print(f"      4. SHTZ - Sub-High Temp (0.5 < UHI ≤ 1):       {area_categories_t2['SHTZ']:.2f} km²")
    print(f"      5. HTZ  - High Temperature (UHI > 1):          {area_categories_t2['HTZ']:.2f} km²")
    print(f"      Total: {area_categories_t2['total']:.2f} km²")

    # Calculate area changes
    print(f"\n   Area Change T1→T2:")
    print(f"      Δ LTZ  : {area_categories_t2['LTZ'] - area_categories_t1['LTZ']:+.2f} km²")
    print(f"      Δ SLTZ : {area_categories_t2['SLTZ'] - area_categories_t1['SLTZ']:+.2f} km²")
    print(f"      Δ MTZ  : {area_categories_t2['MTZ'] - area_categories_t1['MTZ']:+.2f} km²")
    print(f"      Δ SHTZ : {area_categories_t2['SHTZ'] - area_categories_t1['SHTZ']:+.2f} km²")
    print(f"      Δ HTZ  : {area_categories_t2['HTZ'] - area_categories_t1['HTZ']:+.2f} km²")

    # Store UHI data
    stats['UHI_30m'] = {
        't1': {
            'uhi_stats': uhi_stats_30m_t1_dict,
            'lst_mean': lst_mean_t1,
            'lst_std': lst_std_t1,
            'temp_range': {'min': temp_min_t1, 'max': temp_max_t1}
        },
        't2': {
            'uhi_stats': uhi_stats_30m_t2_dict,
            'lst_mean': lst_mean_t2,
            'lst_std': lst_std_t2,
            'temp_range': {'min': temp_min_t2, 'max': temp_max_t2}
        },
        'delta': {
            'uhi_max': delta_uhi_max,
            'temp_hotspot': delta_temp_hotspot
        },
        'images': {
            't1': uhi_30m_t1,
            't2': uhi_30m_t2
        },
        'intensity_classification': {
            't1': uhi_intensity_30m_t1,
            't2': uhi_intensity_30m_t2
        },
        'area_classification_t1': area_categories_t1,  # NOVO
        'area_classification_t2': area_categories_t2
    }

    return stats


def visualize_bitemporal_results(results, geometry, zoom=15,
                                 lst_min=20, lst_max=60,
                                 uhi_min=-4, uhi_max=4):
    """
    Create a comparative visualization of bitemporal results.

    Args:
        results (dict): Bitemporal processing results
        geometry: Region of interest geometry
        zoom (int): Map zoom level
        lst_min (float): Minimum LST value for visualization
        lst_max (float): Maximum LST value for visualization
        uhi_min (float): Minimum UHI value for visualization
        uhi_max (float): Maximum UHI value for visualization

    Returns:
        geemap.Map: Interactive map with layers
    """
    print("\n🗺️  Creating bitemporal visualization...")

    # Create map
    Map = geemap.Map(basemap="SATELLITE", height=900)
    Map.layers[0].opacity = 0.3
    Map.centerObject(geometry, zoom)

    # LST color palette (ref. Cell 9.1)
    lst_palette = [
        '040274', '0602ff', '307ef3', '30c8e2', '3ff38f',
        '86e26f', 'fff705', 'ffb613', 'ff6e08', 'ff0000', 'a71001'
    ]

    # Sentinel-2 RGB
    Map.addLayer(
        results['s2_median_t1'].select(['B4', 'B3', 'B2']).clip(geometry),
        {'min': 0, 'max': 0.3, 'gamma': 1.3},
        'Sentinel-2 RGB T1',
        True
    )

    Map.addLayer(
        results['s2_median_t2'].select(['B4', 'B3', 'B2']).clip(geometry),
        {'min': 0, 'max': 0.3, 'gamma': 1.3},
        'Sentinel-2 RGB T2',
        True
    )

    # LST 30m
    Map.addLayer(
        results['lst_30m_t1'].clip(geometry),
        {'min': lst_min, 'max': lst_max, 'palette': lst_palette},
        'LST 30m T1 (Landsat)',
        False
    )

    Map.addLayer(
        results['lst_30m_t2'].clip(geometry),
        {'min': lst_min, 'max': lst_max, 'palette': lst_palette},
        'LST 30m T2 (Landsat)',
        False
    )

    # === URBAN HEAT ISLAND (UHI) VISUALIZATION ===
    print("   Adding UHI layers...")

    # UHI color palette (divergent)
    uhi_palette = ['2166ac', '67a9cf', 'd1e5f0', 'f7f7f7', 'fddbc7', 'ef8a62', 'b2182b']

    # Calculate UHI for LST 30m
    uhi_30m_t1, _ = calculate_uhi(results['lst_30m_t1'], geometry, scale=30)
    uhi_30m_t2, _ = calculate_uhi(results['lst_30m_t2'], geometry, scale=30)

    # Add UHI 30m layers
    Map.addLayer(
        uhi_30m_t1.clip(geometry),
        {'min': uhi_min, 'max': uhi_max, 'palette': uhi_palette},
        'UHI 30m T1 (Normalized)',
        False
    )

    Map.addLayer(
        uhi_30m_t2.clip(geometry),
        {'min': uhi_min, 'max': uhi_max, 'palette': uhi_palette},
        'UHI 30m T2 (Normalized)',
        False
    )

    # === UHI INTENSITY CLASSIFICATION VISUALIZATION (Deng et al. 2023) ===
    print("   Adding UHI Intensity Classification layers...")

    # UHI classification color palette (5 classes)
    intensity_uhi_palette = ['2166ac', '67a9cf', 'f7f7f7', 'ef8a62', 'b2182b']

    # Classify UHI intensity 30m
    uhi_intensity_30m_t1 = classify_uhi_intensity(uhi_30m_t1)
    uhi_intensity_30m_t2 = classify_uhi_intensity(uhi_30m_t2)

    # Add UHI 30m classification layers
    Map.addLayer(
        uhi_intensity_30m_t1.clip(geometry),
        {'min': 1, 'max': 5, 'palette': intensity_uhi_palette, 'opacity': 0.4},
        'Classif. UHI 30m T1',
        False
    )

    Map.addLayer(
        uhi_intensity_30m_t2.clip(geometry),
        {'min': 1, 'max': 5, 'palette': intensity_uhi_palette, 'opacity': 0.4},
        'Classif. UHI 30m T2',
        True
    )

    # === ADD LEGENDS ===
    print("   Adding legends...")

    # LST legend
    Map.add_colorbar(
        {'min': lst_min,
          'max': lst_max,
          'palette': lst_palette},
        label='Land Surface Temperature (°C)',
        orientation='horizontal',
        transparent_bg=True
    )

    # UHI legend (continuous z-score)
    uhi_legend_dict = {
        f'{uhi_min} (Anomalies below mean)': '#2166ac',
        '0 (Mean)': '#f7f7f7',
        f'{uhi_max} (Anomalies above mean)': '#b2182b'
    }
    Map.add_legend(
        title='UHI (z-score)',
        legend_dict=uhi_legend_dict,
        position='bottomleft'
    )

    # UHI Classification legend (Deng et al. 2023)
    uhi_class_legend = {
        '1 - LTZ (Low Temp)':  intensity_uhi_palette[0],
        '2 - SLTZ (Sub-Low)':  intensity_uhi_palette[1],
        '3 - MTZ (Medium)':    intensity_uhi_palette[2],
        '4 - SHTZ (Sub-High)': intensity_uhi_palette[3],
        '5 - HTZ (High Temp)': intensity_uhi_palette[4]
    }
    Map.add_legend(
        title='UHI Classification (Deng et al. 2023)',
        legend_dict=uhi_class_legend,
        position='bottomleft'
    )

    print("✓ Visualization created")

    return Map


def analyze_spectral_index_thermal_uhi_integration_gee(
    indices_t1_dict,
    indices_t2_dict,
    lst_t1_image,
    lst_t2_image,
    uhi_t1_image,
    uhi_t2_image,
    roi,
    uhi_intensity_t1_image=None,
    uhi_intensity_t2_image=None,
    sentinel_t1_image=None,
    sentinel_t2_image=None,
    index_name='NDVI',
    index_min_threshold=None,
    index_max_threshold=None,
    scale=30,
    num_samples=3000,
    roi_name='study_area'
,
    verbose=False
):
    """
    Integrated analysis of spectral index, temperature and UHI changes.
    Efficient sampling limited to the masked area when thresholds are applied.

    Args:
        indices_t1_dict: Dictionary with T1 spectral indices (ee.Image)
        indices_t2_dict: Dictionary with T2 spectral indices (ee.Image)
        lst_t1_image: ee.Image with LST for period T1 (°C)
        lst_t2_image: ee.Image with LST for period T2 (°C)
        uhi_t1_image: ee.Image with UHI for period T1 (z-scores)
        uhi_t2_image: ee.Image with UHI for period T2 (z-scores)
        roi: ee.Geometry of the region of interest
        uhi_intensity_t1_image: ee.Image with UHI intensity classification T1 (1-5)
        uhi_intensity_t2_image: ee.Image with UHI intensity classification T2 (1-5)
        sentinel_t1_image: ee.Image RGB Sentinel-2 T1
        sentinel_t2_image: ee.Image RGB Sentinel-2 T2
        index_name: Name of the spectral index to analyze
        index_min_threshold: Minimum threshold to filter the index (None = no filter)
        index_max_threshold: Maximum threshold to filter the index (None = no filter)
        scale: Scale in meters for analysis
        num_samples: Number of samples for correlation calculation (default: 3000)
        roi_name: Study area name

    Returns:
        dict: Complete integrated analysis results
    """
    _print = print if verbose else lambda *a, **k: None

    _print("\n" + "="*80)
    _print("INTEGRATED ANALYSIS: SPECTRAL INDEX × TEMPERATURE × UHI (GEE)")
    _print("="*80)
    _print(f"📊 Spectral Index: {index_name}")
    _print(f"📍 Study Area: {roi_name}")
    _print(f"📏 Analysis Scale: {scale}m")
    _print(f"🔢 Requested Samples: {num_samples}")

    # Show thresholds if applied
    if index_min_threshold is not None or index_max_threshold is not None:
        _print(f"\n🎯 APPLIED THRESHOLDS:")
        if index_min_threshold is not None:
            _print(f"   • Minimum: {index_name} ≥ {index_min_threshold}")
        if index_max_threshold is not None:
            _print(f"   • Maximum: {index_name} ≤ {index_max_threshold}")

    # ==============================================================================
    # 1. DATA EXTRACTION AND VALIDATION
    # ==============================================================================

    _print("\n[1/6] Extracting spectral index data from GEE...")

    if index_name not in indices_t1_dict:
        raise ValueError(f"❌ Index '{index_name}' not found in T1. Available: {list(indices_t1_dict.keys())}")
    if index_name not in indices_t2_dict:
        raise ValueError(f"❌ Index '{index_name}' not found in T2. Available: {list(indices_t2_dict.keys())}")

    index_t1_image = indices_t1_dict[index_name]
    index_t2_image = indices_t2_dict[index_name]

    _print(f"   ✓ {index_name} T1: ee.Image")
    _print(f"   ✓ {index_name} T2: ee.Image")
    _print(f"   ✓ LST, UHI: ee.Image")

    # Check intensity classifications
    has_uhi_intensity = (uhi_intensity_t1_image is not None and
                         uhi_intensity_t2_image is not None)

    if has_uhi_intensity:
        _print(f"   ✓ UHI Intensity Classification: ee.Image")
    if sentinel_t1_image and sentinel_t2_image:
        _print(f"   ✓ Sentinel-2 RGB: ee.Image")

    # ==============================================================================
    # 2. APPLY THRESHOLDS AND CREATE EFFICIENT SAMPLING REGION
    # ==============================================================================

    _print("\n[2/6] Applying thresholds to spectral index...")

    threshold_applied = False
    combined_threshold_mask = None
    sampling_region = roi

    if index_min_threshold is not None or index_max_threshold is not None:
        # Create threshold-based mask
        threshold_mask_t1 = ee.Image(1)
        threshold_mask_t2 = ee.Image(1)

        if index_min_threshold is not None:
            threshold_mask_t1 = threshold_mask_t1.And(index_t1_image.gte(index_min_threshold))
            threshold_mask_t2 = threshold_mask_t2.And(index_t2_image.gte(index_min_threshold))
            _print(f"   ✓ Applied minimum threshold: {index_name} ≥ {index_min_threshold}")

        if index_max_threshold is not None:
            threshold_mask_t1 = threshold_mask_t1.And(index_t1_image.lte(index_max_threshold))
            threshold_mask_t2 = threshold_mask_t2.And(index_t2_image.lte(index_max_threshold))
            _print(f"   ✓ Applied maximum threshold: {index_name} ≤ {index_max_threshold}")

        # Combine T1 and T2 masks
        combined_threshold_mask = threshold_mask_t1.And(threshold_mask_t2)

        _print(f"   🎯 Creating optimized sampling region...")

        try:
            class_image = combined_threshold_mask.rename('class')
            threshold_applied = True
            _print(f"   ✓ Sampling region configured (only pixels within thresholds)")

        except Exception as e:
            _print(f"   ⚠️  Error configuring region: {str(e)[:100]}")
            _print(f"   ℹ️  Using default sampling method")
            threshold_applied = False

        # Apply mask to images
        index_t1_image_filtered = index_t1_image.updateMask(combined_threshold_mask)
        index_t2_image_filtered = index_t2_image.updateMask(combined_threshold_mask)
        lst_t1_image_filtered = lst_t1_image.updateMask(combined_threshold_mask)
        lst_t2_image_filtered = lst_t2_image.updateMask(combined_threshold_mask)
        uhi_t1_image_filtered = uhi_t1_image.updateMask(combined_threshold_mask)
        uhi_t2_image_filtered = uhi_t2_image.updateMask(combined_threshold_mask)

        if has_uhi_intensity:
            uhi_intensity_t1_image_filtered = uhi_intensity_t1_image.updateMask(combined_threshold_mask)
            uhi_intensity_t2_image_filtered = uhi_intensity_t2_image.updateMask(combined_threshold_mask)
        else:
            uhi_intensity_t1_image_filtered = None
            uhi_intensity_t2_image_filtered = None

        # Calculate filtered area statistics
        try:
            area_stats = combined_threshold_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=roi,
                scale=scale,
                maxPixels=1e9
            )

            total_area_stats = ee.Image.pixelArea().reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=roi,
                scale=scale,
                maxPixels=1e9
            )

            filtered_area = area_stats.getInfo().get('area', 0)
            total_area = total_area_stats.getInfo().get('area', 1)
            area_percentage = (filtered_area / total_area) * 100 if total_area > 0 else 0

            _print(f"\n   📊 FILTERED AREA:")
            _print(f"      • Total area: {total_area/1e6:.2f} km²")
            _print(f"      • Filtered area: {filtered_area/1e6:.2f} km²")
            _print(f"      • Percentage: {area_percentage:.1f}%")

            if area_percentage < 1.0:
                _print(f"\n      ⚠️  Filtered area is very small ({area_percentage:.1f}%)")
                _print(f"         • Sampling will be focused on this area only")
                _print(f"         • Consider increasing num_samples if needed")

        except Exception as e:
            _print(f"   ⚠️  Could not calculate filtered area: {str(e)[:50]}")

        # Use filtered images
        index_t1_analysis = index_t1_image_filtered
        index_t2_analysis = index_t2_image_filtered
        lst_t1_analysis = lst_t1_image_filtered
        lst_t2_analysis = lst_t2_image_filtered
        uhi_t1_analysis = uhi_t1_image_filtered
        uhi_t2_analysis = uhi_t2_image_filtered
        uhi_intensity_t1_analysis = uhi_intensity_t1_image_filtered
        uhi_intensity_t2_analysis = uhi_intensity_t2_image_filtered

    else:
        # No thresholds - use complete images
        _print(f"   ℹ️  No threshold applied - analyzing the full index")

        index_t1_analysis = index_t1_image
        index_t2_analysis = index_t2_image
        lst_t1_analysis = lst_t1_image
        lst_t2_analysis = lst_t2_image
        uhi_t1_analysis = uhi_t1_image
        uhi_t2_analysis = uhi_t2_image
        uhi_intensity_t1_analysis = uhi_intensity_t1_image
        uhi_intensity_t2_analysis = uhi_intensity_t2_image

        threshold_applied = False

    # ==============================================================================
    # 3. CALCULAR DELTAS
    # ==============================================================================

    _print("\n[3/6] Calculating changes in GEE...")

    delta_index = index_t2_analysis.subtract(index_t1_analysis).rename('delta_index')
    delta_lst = lst_t2_analysis.subtract(lst_t1_analysis).rename('delta_lst')
    delta_uhi = uhi_t2_analysis.subtract(uhi_t1_analysis).rename('delta_uhi')

    _print(f"   ✓ Δ {index_name}, Δ LST, Δ UHI calculados")

    if has_uhi_intensity and uhi_intensity_t1_analysis and uhi_intensity_t2_analysis:
        delta_uhi_intensity = uhi_intensity_t2_analysis.subtract(uhi_intensity_t1_analysis).rename('delta_uhi_intensity')
        _print(f"   ✓ Δ UHI Intensity Classification calculated")
    else:
        delta_uhi_intensity = None

    # Combine all bands
    combined_image = ee.Image.cat([
        delta_index,
        delta_lst,
        delta_uhi,
        index_t1_analysis.rename('index_t1'),
        index_t2_analysis.rename('index_t2'),
        lst_t1_analysis.rename('lst_t1'),
        lst_t2_analysis.rename('lst_t2'),
    ])

    # ==============================================================================
    # 4. CALCULATE GENERAL STATISTICS
    # ==============================================================================

    _print("\n[4/6] Calculating general statistics...")

    stats_delta_index = delta_index.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev().combine(
                reducer2=ee.Reducer.minMax().combine(
                    reducer2=ee.Reducer.percentile([5, 25, 50, 75, 95]),
                    sharedInputs=True
                ),
                sharedInputs=True
            ),
            sharedInputs=True
        ),
        geometry=roi,
        scale=scale,
        maxPixels=1e9
    ).getInfo()

    _print(f"\n   📊 Δ {index_name} Statistics:")
    _print(f"      • Mean: {stats_delta_index.get('delta_index_mean', 0):+.4f}")
    _print(f"      • Median: {stats_delta_index.get('delta_index_p50', 0):+.4f}")
    _print(f"      • StdDev: {stats_delta_index.get('delta_index_stdDev', 0):.4f}")

    stats_delta_lst = delta_lst.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev().combine(
                reducer2=ee.Reducer.minMax(),
                sharedInputs=True
            ),
            sharedInputs=True
        ),
        geometry=roi,
        scale=scale,
        maxPixels=1e9
    ).getInfo()

    _print(f"\n   📊 Δ LST Statistics:")
    _print(f"      • Mean: {stats_delta_lst.get('delta_lst_mean', 0):+.2f}°C")

    stats_delta_uhi = delta_uhi.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ),
        geometry=roi,
        scale=scale,
        maxPixels=1e9
    ).getInfo()

    _print(f"   📊 Δ UHI: {stats_delta_uhi.get('delta_uhi_mean', 0):+.4f}")

    stats_delta_uhi_intensity = {}

    if delta_uhi_intensity is not None:
        stats_delta_uhi_intensity = delta_uhi_intensity.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9
        ).getInfo()
        _print(f"   📊 Δ UHI Intensity: {stats_delta_uhi_intensity.get('delta_uhi_intensity_mean', 0):+.2f} classes")

    # ==============================================================================
    # 5. OPTIMIZED SAMPLING (MASKED AREA ONLY WHEN THRESHOLDS APPLIED)
    # ==============================================================================

    _print(f"\n[5/6] Optimized sampling for correlations...")

    if threshold_applied:
        _print(f"   🎯 DIRECTED sampling: only pixels within thresholds")

        # OPTIMIZED STRATEGY: stratifiedSample with class = 1
        # This ensures ALL samples come from the masked area
        try:
            # Create class image for stratified sampling
            class_for_sampling = combined_threshold_mask.rename('class')

            # Combine with the data we want to sample
            sampling_image = combined_image.addBands(class_for_sampling)

            # stratifiedSample: sample only where class = 1
            sample_fc = sampling_image.stratifiedSample(
                numPoints=num_samples,
                classBand='class',
                region=roi,
                scale=scale,
                seed=42,
                classValues=[1],  # ONLY class 1 (area within thresholds)
                classPoints=[num_samples],  # All samples from class 1
                dropNulls=True,
                geometries=False
            )

            _print(f"   ✓ Stratified sampling configured (class=1 only)")

        except Exception as e:
            _print(f"   ⚠️  Stratified sampling failed: {str(e)[:100]}")
            _print(f"   🔄 Using fallback method: sample + mask")

            # Fallback: standard sample (less efficient, but works)
            sample_fc = combined_image.sample(
                region=roi,
                scale=scale,
                numPixels=num_samples * 3,  # Request 3x more to compensate for filtering
                seed=42,
                geometries=False
            )
    else:
        _print(f"   📍 Standard sampling: entire ROI")

        # No thresholds: standard sampling
        sample_fc = combined_image.sample(
            region=roi,
            scale=scale,
            numPixels=num_samples,
            seed=42,
            geometries=False
        )

    # Check how many samples were obtained
    sample_size = sample_fc.size().getInfo()
    _print(f"   ℹ️  Points sampled: {sample_size}")

    # Warning always visible: ROI saturation or insufficient sampling
    if sample_size < num_samples:
        if sample_size < num_samples * 0.3:
            print(f"   ⚠️  WARNING: only {sample_size} of {num_samples} requested samples ({sample_size/num_samples*100:.0f}%).")
            print(f"      Possible causes: small ROI, restrictive cloud masks, or different Landsat paths.")
            print(f"      Increasing n_samples above {sample_size} will have no effect — the ceiling is the number of valid pixels in the ROI.")
        else:
            print(f"   ℹ️  {sample_size} valid pixels returned from {num_samples} requested "
                  f"({sample_size/num_samples*100:.0f}%) — ROI ceiling reached.")
            print(f"      Increasing n_samples above {sample_size} will not change results.")

    # Limit download to requested (absolute safety cap: 10,000)
    max_download = min(sample_size, num_samples, 10000)
    sample_fc_limited = sample_fc.limit(max_download)
    samples_list = sample_fc_limited.toList(max_download)

    # Extraction function
    def get_properties(feature):
        feature = ee.Feature(feature)
        return [
            feature.get('delta_index'),
            feature.get('delta_lst'),
            feature.get('delta_uhi'),
            feature.get('index_t1'),
            feature.get('index_t2'),
            feature.get('lst_t1'),
            feature.get('lst_t2'),
        ]

    # Sample download
    _print(f"   ⬇️  Downloading {max_download} samples...")

    try:
        samples_values = samples_list.map(get_properties).getInfo()

        delta_index_sample = np.array([v[0] for v in samples_values if v[0] is not None], dtype=float)
        delta_lst_sample   = np.array([v[1] for v in samples_values if v[1] is not None], dtype=float)
        delta_uhi_sample   = np.array([v[2] for v in samples_values if v[2] is not None], dtype=float)
        index_t1_sample    = np.array([v[3] for v in samples_values if len(v) > 3 and v[3] is not None], dtype=float)
        index_t2_sample    = np.array([v[4] for v in samples_values if len(v) > 4 and v[4] is not None], dtype=float)
        lst_t1_sample      = np.array([v[5] for v in samples_values if len(v) > 5 and v[5] is not None], dtype=float)
        lst_t2_sample      = np.array([v[6] for v in samples_values if len(v) > 6 and v[6] is not None], dtype=float)

    except Exception as e:
        _print(f"   ⚠️  Direct download failed, using batches...")

        batch_size = 500
        delta_index_sample = []
        delta_lst_sample   = []
        delta_uhi_sample   = []
        index_t1_sample    = []
        index_t2_sample    = []
        lst_t1_sample      = []
        lst_t2_sample      = []

        num_batches = int(np.ceil(max_download / batch_size))

        for i in range(num_batches):
            try:
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, max_download)

                batch_list = samples_list.slice(start_idx, end_idx)
                batch_values = batch_list.map(get_properties).getInfo()

                delta_index_sample.extend([v[0] for v in batch_values if v[0] is not None])
                delta_lst_sample.extend([v[1] for v in batch_values if v[1] is not None])
                delta_uhi_sample.extend([v[2] for v in batch_values if v[2] is not None])
                index_t1_sample.extend([v[3] for v in batch_values if len(v) > 3 and v[3] is not None])
                index_t2_sample.extend([v[4] for v in batch_values if len(v) > 4 and v[4] is not None])
                lst_t1_sample.extend([v[5] for v in batch_values if len(v) > 5 and v[5] is not None])
                lst_t2_sample.extend([v[6] for v in batch_values if len(v) > 6 and v[6] is not None])

            except Exception as batch_error:
                _print(f"      ⚠️  Batch {i+1} error")
                continue

        delta_index_sample = np.array(delta_index_sample, dtype=float)
        delta_lst_sample   = np.array(delta_lst_sample, dtype=float)
        delta_uhi_sample   = np.array(delta_uhi_sample, dtype=float)
        index_t1_sample    = np.array(index_t1_sample, dtype=float)
        index_t2_sample    = np.array(index_t2_sample, dtype=float)
        lst_t1_sample      = np.array(lst_t1_sample, dtype=float)
        lst_t2_sample      = np.array(lst_t2_sample, dtype=float)

    # Synchronize lengths (delta arrays define the base length)
    min_length = min(len(delta_index_sample), len(delta_lst_sample),
                     len(delta_uhi_sample))
    delta_index_sample = delta_index_sample[:min_length]
    delta_lst_sample   = delta_lst_sample[:min_length]
    delta_uhi_sample   = delta_uhi_sample[:min_length]
    # Absolute T1/T2: truncate to same length if available
    if len(index_t1_sample) >= min_length:
        index_t1_sample = index_t1_sample[:min_length]
        index_t2_sample = index_t2_sample[:min_length]
        lst_t1_sample   = lst_t1_sample[:min_length]
        lst_t2_sample   = lst_t2_sample[:min_length]
    else:
        index_t1_sample = index_t2_sample = lst_t1_sample = lst_t2_sample = np.array([])

    # Remove NaN and Inf
    valid_mask = (np.isfinite(delta_index_sample) &
                  np.isfinite(delta_lst_sample) &
                  np.isfinite(delta_uhi_sample))

    delta_index_sample = delta_index_sample[valid_mask]
    delta_lst_sample   = delta_lst_sample[valid_mask]
    delta_uhi_sample   = delta_uhi_sample[valid_mask]
    _has_abs = len(index_t1_sample) == min_length
    if _has_abs:
        index_t1_sample = index_t1_sample[valid_mask]
        index_t2_sample = index_t2_sample[valid_mask]
        lst_t1_sample   = lst_t1_sample[valid_mask]
        lst_t2_sample   = lst_t2_sample[valid_mask]

    # Remove outliers via IQR × 3 on Δ index.
    # Indices like DBSI have near-zero denominators in some pixels that produce
    # large but finite values (not caught by isfinite) which distort statistics
    # and correlations. This step is applied to all arrays consistently.
    _iqr_outliers_removed = 0
    _iqr_range = None
    if len(delta_index_sample) > 10:
        _q1, _q3 = np.percentile(delta_index_sample, [25, 75])
        _iqr = _q3 - _q1
        if _iqr > 1e-9:
            _lo = _q1 - 3.0 * _iqr
            _hi = _q3 + 3.0 * _iqr
            _keep = (delta_index_sample >= _lo) & (delta_index_sample <= _hi)
            _n_before = len(delta_index_sample)
            delta_index_sample = delta_index_sample[_keep]
            delta_lst_sample   = delta_lst_sample[_keep]
            delta_uhi_sample   = delta_uhi_sample[_keep]
            if _has_abs:
                index_t1_sample = index_t1_sample[_keep]
                index_t2_sample = index_t2_sample[_keep]
                lst_t1_sample   = lst_t1_sample[_keep]
                lst_t2_sample   = lst_t2_sample[_keep]
            _iqr_outliers_removed = _n_before - len(delta_index_sample)
            _iqr_range = (_lo, _hi)
            if _iqr_outliers_removed:
                print(f"   🔬 Outliers removed (3×IQR Δ{index_name}): "
                      f"{_iqr_outliers_removed} points ({_iqr_outliers_removed / _n_before * 100:.1f}%)")
                print(f"      Δ{index_name} range after filter: [{_lo:.4f}, {_hi:.4f}]")

    n_valid_samples = len(delta_index_sample)
    _print(f"   ✓ Valid samples: {n_valid_samples}")

    if threshold_applied:
        efficiency = (n_valid_samples / sample_size * 100) if sample_size > 0 else 0
        _print(f"   ✓ Sampling efficiency: {efficiency:.1f}% (all samples from filtered area)")

    # ==============================================================================
    # 6. CALCULATE CORRELATIONS
    # ==============================================================================

    from scipy.stats import pearsonr, spearmanr

    if n_valid_samples >= 30:
        try:
            corr_index_lst, p_index_lst = pearsonr(delta_index_sample, delta_lst_sample)
            corr_index_uhi, p_index_uhi = pearsonr(delta_index_sample, delta_uhi_sample)

            corr_lst_uhi, p_lst_uhi = pearsonr(delta_lst_sample, delta_uhi_sample)

            corr_index_lst_spearman, p_index_lst_spearman = spearmanr(delta_index_sample, delta_lst_sample)
            corr_index_uhi_spearman, p_index_uhi_spearman = spearmanr(delta_index_sample, delta_uhi_sample)

            corr_lst_uhi_spearman, p_lst_uhi_spearman = spearmanr(delta_lst_sample, delta_uhi_sample)

            _print(f"\n   📊 CORRELATIONS (Pearson, n={n_valid_samples}):")
            _print(f"      • Δ {index_name} × Δ LST: r = {corr_index_lst:+.4f} (p = {p_index_lst:.6f})")
            _print(f"      • Δ {index_name} × Δ UHI: r = {corr_index_uhi:+.4f} (p = {p_index_uhi:.6f})")

        except Exception as corr_error:
            _print(f"   ⚠️  Correlation error: {str(corr_error)[:100]}")
            corr_index_lst = corr_index_uhi = np.nan
            corr_lst_uhi = np.nan
            p_index_lst = p_index_uhi = np.nan
            p_lst_uhi = np.nan
            corr_index_lst_spearman = corr_index_uhi_spearman = np.nan
            corr_lst_uhi_spearman = np.nan
            p_index_lst_spearman = p_index_uhi_spearman = np.nan
            p_lst_uhi_spearman = np.nan
    else:
        _print(f"   ⚠️  Insufficient samples: {n_valid_samples} < 30")
        _print(f"\n   💡 Suggestions:")
        _print(f"      • Increase scale: {scale}m → {scale*2}m or {scale*3}m")
        _print(f"      • Increase num_samples: {num_samples} → {num_samples*2} or {num_samples*3}")
        if threshold_applied and index_min_threshold is not None:
            _print(f"      • Relax threshold: {index_min_threshold} → {index_min_threshold*0.8:.2f}")

        corr_index_lst = corr_index_uhi = np.nan
        corr_lst_uhi = np.nan
        p_index_lst = p_index_uhi = np.nan
        p_lst_uhi = np.nan
        corr_index_lst_spearman = corr_index_uhi_spearman = np.nan
        corr_lst_uhi_spearman = np.nan
        p_index_lst_spearman = p_index_uhi_spearman = np.nan
        p_lst_uhi_spearman = np.nan

    # ==============================================================================
    # 7. RETORNAR RESULTADOS
    # ==============================================================================

    _print("\n[6/6] Preparing results...")

    results = {
        'index_name': index_name,
        'scale': scale,
        'num_samples_requested': num_samples,
        'num_samples_obtained': n_valid_samples,
        'roi': roi,
        'roi_name': roi_name,
        'threshold_applied': threshold_applied,
        'index_min_threshold': index_min_threshold,
        'index_max_threshold': index_max_threshold,
        'threshold_mask': combined_threshold_mask,
        'sentinel_t1_image': sentinel_t1_image,
        'sentinel_t2_image': sentinel_t2_image,
        'images': {
            'index_t1': index_t1_image,
            'index_t2': index_t2_image,
            'index_t1_filtered': index_t1_analysis,
            'index_t2_filtered': index_t2_analysis,
            'lst_t1': lst_t1_analysis,
            'lst_t2': lst_t2_analysis,
            'delta_index': delta_index,
            'delta_lst': delta_lst,
            'delta_uhi': delta_uhi
        },
        'statistics': {
            'delta_index': stats_delta_index,
            'delta_lst': stats_delta_lst,
            'delta_uhi': stats_delta_uhi
        },
        'correlations': {
            'index_lst': {
                'pearson_r': corr_index_lst,
                'pearson_p': p_index_lst,
                'spearman_rho': corr_index_lst_spearman,
                'spearman_p': p_index_lst_spearman
            },
            'index_uhi': {
                'pearson_r': corr_index_uhi,
                'pearson_p': p_index_uhi,
                'spearman_rho': corr_index_uhi_spearman,
                'spearman_p': p_index_uhi_spearman
            },
            'lst_uhi': {
                'pearson_r': corr_lst_uhi,
                'pearson_p': p_lst_uhi,
                'spearman_rho': corr_lst_uhi_spearman,
                'spearman_p': p_lst_uhi_spearman
            }
        },
        'samples': {
            'delta_index': delta_index_sample,
            'delta_lst': delta_lst_sample,
            'delta_uhi': delta_uhi_sample,
            'index_t1': index_t1_sample,
            'index_t2': index_t2_sample,
            'lst_t1': lst_t1_sample,
            'lst_t2': lst_t2_sample,
            'n_samples': n_valid_samples,
            'iqr_outliers_removed': _iqr_outliers_removed,
            'iqr_range': _iqr_range,
        }
    }

    if delta_uhi_intensity is not None:
        results['images']['uhi_intensity_t1'] = uhi_intensity_t1_image
        results['images']['uhi_intensity_t2'] = uhi_intensity_t2_image
        results['images']['delta_uhi_intensity'] = delta_uhi_intensity
        results['statistics']['delta_uhi_intensity'] = stats_delta_uhi_intensity


    _print("\n" + "="*80)
    _print("✅ INTEGRATED ANALYSIS COMPLETE")
    _print("="*80)

    if threshold_applied:
        _print(f"\n💡 OPTIMIZED sampling active:")
        _print(f"   • All {n_valid_samples} samples came from the filtered area")
        _print(f"   • Thresholds: {index_min_threshold} ≤ {index_name} ≤ {index_max_threshold}")

    return results


def create_interactive_analysis_map(results_bitemporal, roi, results_integration, zoom=13,
                                    lst_t1=None, lst_t2=None, lst_vis=None,
                                    lst_t1_label='T1', lst_t2_label='T2',
                                    percentile_threshold=97, show_CDI=True):
    """
    Creates an interactive geemap map with layers for the analyzed index.

    Layers included:
      - RGB Sentinel-2 T1 and T2 (corrected visParams: min=0, max=0.3, gamma=1.3)
      - Index T1 and T2 (absolute scale)
      - Δ Index  (diverging: loss/decrease = red, gain/increase = blue)
      - Δ LST    (diverging: warming = red, cooling = blue)
      - Δ UHI    (diverging)
    Legends:
      - add_colorbar for Δ Index and absolute LST
      - discrete add_legend for Δ LST and Δ index (qualitative categories)

    Args:
        results_bitemporal : dict from calculate_bitemporal_lst (contains s2_median_t1/t2)
        roi                : ee.Geometry
        results_integration: dict from analyze_spectral_index_thermal_uhi_integration_gee
        zoom               : initial zoom level

    Returns:
        geemap.Map with layers and legends
    """
    import geemap
    import numpy as _np

    idx_name = results_integration.get('index_name', 'index').upper()
    imgs     = results_integration.get('images', {})

    m = geemap.Map(basemap="SATELLITE", zoom=zoom, height='850px')
    m.layers[0].opacity = 0.3
    m.centerObject(roi, zoom)

    # ── Palettes ──────────────────────────────────────────────────────────────
    # Absolute LST: cool blue → hot red (module default)
    _lst_pal  = ['040274', '0602ff', '307ef3', '30c8e2', '3ff38f',
                 'fff705', 'ffb613', 'ff6e08', 'ff0000', 'a71001']

    _VEGE_INDICES  = {'NDVI', 'EVI', 'SAVI', 'LAI', 'BNIRV', 'NIRV', 'EVI2', 'MSAVI'}
    _URBAN_INDICES = {'NDBI', 'IBI', 'EMBI', 'BUI', 'MNDBI', 'NBI', 'VIBI'}
    _WATER_INDICES = {'MNDWI', 'NDWI', 'DSWI4', 'AWEInsh', 'AWEIsh', 'AWEI'}

    # Index absolute palettes — distinct from UHI (blue→white→red)
    # Vegetation: RdYlGn — conventional and distinct from UHI
    _PAL_VEGE_ABS  = ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']
    # Urban: BrBG (brown→teal)
    _PAL_URBAN_ABS = ['543005', 'bf812d', 'f6e8c3', 'f5f5f5', '80cdc1', '35978f', '003c30']
    # Other (water, soil, etc.): PRGn (purple→green)
    _PAL_OTHER_ABS = ['40004b', '9970ab', 'c2a5cf', 'f7f7f7', 'a6dba0', '5aae61', '00441b']
    # Delta — PuGn diverging (purple→white→green, distinct from UHI blue→white→red)
    _PAL_DELTA     = ['7b3294', 'c2a5cf', 'e7d4e8', 'f7f7f7', 'd9f0d3', 'a6dba0', '1b7837']

    if idx_name in _VEGE_INDICES:
        _idx_pal = _PAL_VEGE_ABS
    elif idx_name in _URBAN_INDICES:
        _idx_pal = _PAL_URBAN_ABS
    else:
        _idx_pal = _PAL_OTHER_ABS

    # 5-color palettes for Δ index discrete colorbar
    if idx_name in _VEGE_INDICES:
        _delta_pal5 = ['d73027', 'f46d43', 'ffffbf', 'a6d96a', '1a9850']  # RdYlGn: loss→gain
    else:  # urban, water, soil, other — PuGn 5-step
        _delta_pal5 = ['7b3294', 'c2a5cf', 'f7f7f7', 'a6dba0', '1b7837']

    _delta_lst_pal5 = ['2166ac', 'd1e5f0', 'f7f7f7', 'fddbc7', 'b2182b']  # cooling→warming

    # ── Dynamic ranges from samples ───────────────────────────────────────────
    _s_t1 = results_integration.get('samples', {}).get('index_t1', _np.array([]))
    _s_t2 = results_integration.get('samples', {}).get('index_t2', _np.array([]))
    if len(_s_t1) > 10 and len(_s_t2) > 10:
        # Global range: min/max across T1 and T2 so both layers share the same stretch
        _idx_min = float(min(_np.percentile(_s_t1, 2),  _np.percentile(_s_t2, 2)))
        _idx_max = float(max(_np.percentile(_s_t1, 98), _np.percentile(_s_t2, 98)))
    elif len(_s_t1) > 10:
        _idx_min = float(_np.percentile(_s_t1, 2))
        _idx_max = float(_np.percentile(_s_t1, 98))
    else:
        _idx_min, _idx_max = -1.0, 1.0

    _delta_idx_s = results_integration.get('samples', {}).get('delta_index', _np.array([]))
    _dmax = float(_np.percentile(_np.abs(_delta_idx_s), percentile_threshold)) if len(_delta_idx_s) > 10 else 0.3
    _dmax = max(_dmax, 0.05)  # evitar range degenerado
    _delta_lst_s = results_integration.get('samples', {}).get('delta_lst', _np.array([]))
    _lst_dmax = float(_np.percentile(_np.abs(_delta_lst_s), percentile_threshold)) if len(_delta_lst_s) > 10 else 5.0
    _lst_dmax = max(_lst_dmax, 0.5)
    _lst_t1_s = results_integration.get('samples', {}).get('lst_t1', _np.array([]))
    _lst_t2_s = results_integration.get('samples', {}).get('lst_t2', _np.array([]))
    if len(_lst_t1_s) > 10:
        _lst_min = float(_np.percentile(_np.concatenate([_lst_t1_s, _lst_t2_s]), 2))
        _lst_max = float(_np.percentile(_np.concatenate([_lst_t1_s, _lst_t2_s]), 98))
    else:
        _lst_min, _lst_max = 20.0, 60.0

    vis_idx  = {'min': _idx_min,  'max': _idx_max,  'palette': _idx_pal}
    vis_didx = {'min': -_dmax,    'max': _dmax,     'palette': _delta_pal5}
    vis_dlst = {'min': -_lst_dmax,'max': _lst_dmax, 'palette': _delta_lst_pal5}
    vis_lst  = {'min': _lst_min,  'max': _lst_max,  'palette': _lst_pal}
    _div_rdb = ['b2182b', 'ef8a62', 'fddbc7', 'f7f7f7', 'd1e5f0', '67a9cf', '2166ac']
    vis_duhi = {'min': -1.5, 'max': 1.5, 'palette': list(reversed(_div_rdb))}

    # ── RGB Sentinel-2 ────────────────────────────────────────────────────────
    # results_bitemporal uses keys s2_median_t1 / s2_median_t2 (already scaled 0-1)
    _vis_rgb = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3, 'gamma': 1.3}
    try:
        _s2_t1 = results_bitemporal.get('s2_median_t1')
        if _s2_t1 is not None:
            m.addLayer(_s2_t1.select(['B4', 'B3', 'B2']).clip(roi), _vis_rgb,
                       'RGB S2 T1', True)
    except Exception:
        pass
    try:
        _s2_t2 = results_bitemporal.get('s2_median_t2')
        if _s2_t2 is not None:
            m.addLayer(_s2_t2.select(['B4', 'B3', 'B2']).clip(roi), _vis_rgb,
                       'RGB S2 T2', True)
    except Exception:
        pass

    # ── Index T1 / T2 ────────────────────────────────────────────────────────
    if 'index_t1' in imgs:
        m.addLayer(imgs['index_t1'].clip(roi), vis_idx, f'{idx_name} T1', False)
    if 'index_t2' in imgs:
        m.addLayer(imgs['index_t2'].clip(roi), vis_idx, f'{idx_name} T2', False)

    # ── LST T1 / T2 (absolute) ───────────────────────────────────────────────
    _lv = lst_vis if lst_vis is not None else vis_lst
    if lst_t1 is not None:
        m.addLayer(lst_t1.clip(roi), _lv, f'🌡️ LST T1 (°C) [{lst_t1_label}]', False)
    if lst_t2 is not None:
        m.addLayer(lst_t2.clip(roi), _lv, f'🌡️ LST T2 (°C) [{lst_t2_label}]', False)

    # ── Deltas ────────────────────────────────────────────────────────────────
    if 'delta_index' in imgs:
        m.addLayer(imgs['delta_index'].clip(roi), vis_didx,
                   f'Δ {idx_name} (T2−T1)', False)
    if 'delta_lst' in imgs:
        m.addLayer(imgs['delta_lst'].clip(roi), vis_dlst,
                   'Δ LST °C (T2−T1)', False)

    # ── CDI: Change Detection Index (Silva & Torres, 2021) ───────────────────
    if show_CDI and 'delta_lst' in imgs and 'delta_index' in imgs:
        cdi_img = imgs['delta_lst'].multiply(imgs['delta_index']).rename('CDI')
        _cdi_s = (_delta_lst_s * _delta_idx_s) if (len(_delta_lst_s) and len(_delta_idx_s)
                                                   and len(_delta_lst_s) == len(_delta_idx_s)) else _np.array([])
        if len(_cdi_s) > 10:
            _cdi_max = float(_np.percentile(_np.abs(_cdi_s), percentile_threshold))
        else:
            _cdi_max = _dmax * _lst_dmax
        _cdi_max = max(_cdi_max, 0.05)
        _cdi_pal5 = ['b2182b', 'fddbc7', 'f7f7f7', 'a6d96a', '1a9850']
        vis_cdi = {'min': -_cdi_max, 'max': _cdi_max, 'palette': _cdi_pal5}
        m.addLayer(cdi_img.clip(roi), vis_cdi,
                   f'CDI (Δ LST × Δ {idx_name})', True)

    # ── Delta UHI z-score ─────────────────────────────────────────────────────
    # if 'delta_uhi' in imgs:
    #     m.addLayer(imgs['delta_uhi'].clip(roi), vis_duhi,
    #                'Δ UHI z-score (T2−T1)', False)

    # ── ROI boundary ──────────────────────────────────────────────────────────
    empty = ee.Image().byte()
    roi_outline = empty.paint(
        featureCollection=ee.FeatureCollection([ee.Feature(roi)]),
        color=2, width=2
    )
    m.addLayer(roi_outline, {'palette': ['black']}, '📍 Study Area (ROI)', True)

    # ── Legends ───────────────────────────────────────────────────────────────

    def _mpl_colorbar_html(palette_hex, bounds, abbrevs, ivals, title):
        """Generates a discrete matplotlib colorbar as base64 PNG to embed in the map."""
        import matplotlib.colors as _mc
        import io, base64
        colors = ['#' + c if not c.startswith('#') else c for c in palette_hex]
        cmap = _mc.ListedColormap(colors)
        norm = _mc.BoundaryNorm(bounds, ncolors=len(colors))
        ticks = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(colors))]
        tick_lbs = [f'{a}\n{v}' for a, v in zip(abbrevs, ivals)]
        fig, ax = plt.subplots(figsize=(8.5, 0.90))
        fig.patch.set_alpha(0.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, cax=ax, orientation='horizontal', ticks=ticks)
        cb.set_ticklabels(tick_lbs)
        cb.ax.tick_params(labelsize=11, length=0, pad=3)
        cb.set_label(title, fontsize=14, labelpad=5, fontweight='bold')
        cb.outline.set_linewidth(0.5)
        ax.set_facecolor('none')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', transparent=True)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:430px;">'

    # symmetric proportional bounds for 5 classes: [-dmax, -0.6d, -0.2d, +0.2d, +0.6d, +dmax]
    _b = [round(-_dmax + i * 0.4 * _dmax, 6) for i in range(6)]  # 6 bordas, 5 classes

    # ── 2. Colorbar — LST absoluta ────────────────────────────────────────────
    try:
        m.add_colorbar(
            _lv,
            label='Land Surface Temperature — LST (°C)',
            orientation='horizontal',
            transparent_bg=True,
            position='bottomleft',
        )
    except Exception:
        pass

    # ── 2b. Colorbar — Index (T1 scale, shared by T1 and T2 layers) ──────────
    try:
        m.add_colorbar(
            vis_idx,
            label=f'{idx_name} (shared scale: T1 ∪ T2 range)',
            orientation='horizontal',
            transparent_bg=True,
            position='bottomleft',
        )
    except Exception:
        pass

    # ── 3. Discrete colorbar — Δ index ───────────────────────────────────────
    if idx_name in _VEGE_INDICES:
        _abbr_di   = ['SL',  'ML',   'Stb', 'MG',   'SG']
        _ivals_di  = [f'< {_b[1]:.3f}',
                      f'{_b[1]:.3f} to {_b[2]:.3f}',
                      f'{_b[2]:.3f} to {_b[3]:.3f}',
                      f'{_b[3]:.3f} to {_b[4]:.3f}',
                      f'> {_b[4]:.3f}']
    elif idx_name in _URBAN_INDICES:
        _abbr_di   = ['SD',  'MD',   'Stb', 'MI',   'SI']
        _ivals_di  = [f'< {_b[1]:.3f}',
                      f'{_b[1]:.3f} to {_b[2]:.3f}',
                      f'{_b[2]:.3f} to {_b[3]:.3f}',
                      f'{_b[3]:.3f} to {_b[4]:.3f}',
                      f'> {_b[4]:.3f}']
    else:  # water, soil, other
        _abbr_di   = ['SD',  'MD',   'Stb', 'MI',   'SI']
        _ivals_di  = [f'< {_b[1]:.3f}',
                      f'{_b[1]:.3f} to {_b[2]:.3f}',
                      f'{_b[2]:.3f} to {_b[3]:.3f}',
                      f'{_b[3]:.3f} to {_b[4]:.3f}',
                      f'> {_b[4]:.3f}']
    try:
        _html_di = _mpl_colorbar_html(
            _delta_pal5, _b, _abbr_di, _ivals_di, f'Δ {idx_name} (T2 − T1)')
        m.add_html(_html_di, position='bottomright')
    except Exception:
        pass
    # ── 4. Colorbar discreta — Δ LST ─────────────────────────────────────────
    _bl = [round(-_lst_dmax + i * 0.4 * _lst_dmax, 4) for i in range(6)]
    _abbr_dlst  = ['SC',   'MC',  'Stb', 'MW',   'SW']
    _ivals_dlst = [f'< {_bl[1]:.1f} °C',
                   f'{_bl[1]:.1f} to {_bl[2]:.1f} °C',
                   f'{_bl[2]:.1f} to {_bl[3]:.1f} °C',
                   f'{_bl[3]:.1f} to {_bl[4]:.1f} °C',
                   f'> {_bl[4]:.1f} °C']
    try:
        _html_dlst = _mpl_colorbar_html(
            _delta_lst_pal5, _bl, _abbr_dlst, _ivals_dlst, 'Δ LST °C (T2 − T1)')
        m.add_html(_html_dlst, position='bottomright')
    except Exception:
        pass
    # ── 5. Discrete colorbar — CDI (Δ LST × Δ Index) ─────────────────────────
    if show_CDI and 'delta_lst' in imgs and 'delta_index' in imgs:
        _bc = [round(-_cdi_max + i * 0.4 * _cdi_max, 3) for i in range(6)]
        _abbr_cdi  = ['SN+',  'SNm',   'NA',  'SPm',   'SP+']
        _ivals_cdi = [f'< {_bc[1]:.3f}',
                      f'{_bc[1]:.3f} a {_bc[2]:.3f}',
                      f'{_bc[2]:.3f} a {_bc[3]:.3f}',
                      f'{_bc[3]:.3f} a {_bc[4]:.3f}',
                      f'> {_bc[4]:.3f}']
        try:
            _html_cdi = _mpl_colorbar_html(
                _cdi_pal5, _bc, _abbr_cdi, _ivals_cdi,
                f'CDI = Δ LST × Δ {idx_name}')
            m.add_html(_html_cdi, position='bottomright')
        except Exception:
            pass

    m.add_html(
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

    return m


# ==============================================================================
# VISUALIZATION FUNCTION: CORRELATION PLOTS
# ==============================================================================

def create_correlation_plots(
    results_dict,
    output_dir=None,
    figsize=(14, 22),
    verbose=False,
    percentile_threshold=97
):
    """
    Integrated analysis plots for planners:
    - Row 0: T1 vs T2 distributions (a) index  (b) LST
    - Row 1: delta distributions (c) Δ index  (d) Δ LST with statistical lines
    - Row 2: scatter Δ index × Δ LST (full width) with 95% PI and metrics box
    - Row 3: interpretation panel full width
    """
    _print = print if verbose else lambda *a, **k: None

    index_name   = results_dict['index_name']
    roi_name     = results_dict['roi_name']
    samples      = results_dict['samples']
    correlations = results_dict['correlations']

    delta_index_sample = samples['delta_index']
    delta_lst_sample   = samples['delta_lst']
    index_t1_sample    = samples.get('index_t1', np.array([]))
    index_t2_sample    = samples.get('index_t2', np.array([]))
    lst_t1_sample      = samples.get('lst_t1', np.array([]))
    lst_t2_sample      = samples.get('lst_t2', np.array([]))
    _iqr_out   = samples.get('iqr_outliers_removed', 0)
    _iqr_range = samples.get('iqr_range', None)

    n = len(delta_index_sample)
    has_abs = len(index_t1_sample) > 10

    if n < 30:
        _print(f"   Insufficient samples ({n} < 30). Skipping visualization.")
        return None

    _print(f"   Creating plots for {index_name} (n={n})...")

    # ── Correlations ─────────────────────────────────────────────────────────
    corr_r  = correlations['index_lst']['pearson_r']
    corr_p  = correlations['index_lst']['pearson_p']
    r2      = corr_r ** 2

    # ── Semantic interpretation ───────────────────────────────────────────────
    abs_r = abs(corr_r)
    if abs_r >= 0.70:
        strength = 'strong'
    elif abs_r >= 0.40:
        strength = 'moderate'
    elif abs_r >= 0.20:
        strength = 'weak'
    else:
        strength = 'very weak'

    direction = 'negative' if corr_r < 0 else 'positive'
    _idx_up = index_name.upper()
    if _idx_up in ('NDVI', 'EVI', 'SAVI', 'LAI', 'BNIRV', 'NIRV'):
        _meaning = (
            f'{"Loss" if np.mean(delta_index_sample) < 0 else "Gain"} of vegetation cover '
            f'is associated with {"increase" if corr_r < 0 else "reduction"} in temperature.'
        )
    elif _idx_up in ('NDBI', 'IBI', 'EMBI', 'BUI', 'MNDBI'):
        _meaning = (
            f'{"Expansion" if np.mean(delta_index_sample) > 0 else "Reduction"} of impervious surfaces '
            f'is associated with {"increase" if corr_r > 0 else "reduction"} in temperature.'
        )
    elif _idx_up in ('MNDWI', 'NDWI', 'DSWI4', 'AWEInsh', 'AWEIsh'):
        _meaning = (
            f'Variation in water bodies / moisture '
            f'{"reduces" if corr_r < 0 else "increases"} land surface temperature.'
        )
    else:
        _dir_en = 'increase' if corr_r > 0 else 'reduce'
        _meaning = f'An increase in the index tends to {_dir_en} temperature.'

    # ── FIGURA ────────────────────────────────────────────────────────────────
    import matplotlib.gridspec as gridspec
    from scipy import stats as _sst

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           height_ratios=[1.0, 1.0, 1.6, 1.1],
                           hspace=0.25, wspace=0.35)

    _ALPHA = 0.65
    _BINS  = 40

    # ── Clip helpers (robust display range, outliers excluded from bins) ──────
    def _clip_range(arr, plo=1, phi=99):
        lo, hi = np.percentile(arr, plo), np.percentile(arr, phi)
        margin = max((hi - lo) * 0.05, 1e-9)
        return lo - margin, hi + margin

    # ── (a) [0,0] T1 vs T2 Distribution — INDEX ──────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    if has_abs:
        _idx_all   = np.concatenate([index_t1_sample, index_t2_sample])
        _idx_lo, _idx_hi = _clip_range(_idx_all)
        _t1_clip   = index_t1_sample[(index_t1_sample >= _idx_lo) & (index_t1_sample <= _idx_hi)]
        _t2_clip   = index_t2_sample[(index_t2_sample >= _idx_lo) & (index_t2_sample <= _idx_hi)]
        ax_a.hist(_t1_clip, bins=_BINS, color='steelblue', alpha=_ALPHA,
                  edgecolor='none', label=f'T1 (med={np.median(index_t1_sample):.3f})', density=True)
        ax_a.hist(_t2_clip, bins=_BINS, color='crimson', alpha=_ALPHA,
                  edgecolor='none', label=f'T2 (med={np.median(index_t2_sample):.3f})', density=True)
        ax_a.axvline(np.median(index_t1_sample), color='steelblue', lw=1.5, ls='--')
        ax_a.axvline(np.median(index_t2_sample), color='crimson',   lw=1.5, ls='--')
        ax_a.set_xlabel(index_name, fontsize=10)
        ax_a.set_ylabel('Density', fontsize=10)
        # Build title: show IQR removal info when outliers were removed upstream
        if _iqr_out > 0 and _iqr_range is not None:
            _a_title = (f'(a)  {index_name} — T1 vs T2\n'
                        f'({_iqr_out} outlier(s) removed, 3×IQR on Δ: '
                        f'[{_iqr_range[0]:.3f}, {_iqr_range[1]:.3f}])')
        else:
            _a_title = f'(a)  {index_name} — T1 vs T2'
        ax_a.set_title(_a_title, fontsize=10, fontweight='bold')
        ax_a.legend(fontsize=8)
        ax_a.set_xlim(_idx_lo, _idx_hi)
    else:
        ax_a.set_visible(False)
    ax_a.grid(True, alpha=0.3)

    # ── (b) [0,1] T1 vs T2 Distribution — LST ────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if has_abs:
        _lst_all   = np.concatenate([lst_t1_sample, lst_t2_sample])
        _lst_lo, _lst_hi = _clip_range(_lst_all)
        _lt1_clip  = lst_t1_sample[(lst_t1_sample >= _lst_lo) & (lst_t1_sample <= _lst_hi)]
        _lt2_clip  = lst_t2_sample[(lst_t2_sample >= _lst_lo) & (lst_t2_sample <= _lst_hi)]
        ax_b.hist(_lt1_clip, bins=_BINS, color='steelblue', alpha=_ALPHA,
                  edgecolor='none', label=f'T1 (med={np.median(lst_t1_sample):.1f}°C)', density=True)
        ax_b.hist(_lt2_clip, bins=_BINS, color='crimson', alpha=_ALPHA,
                  edgecolor='none', label=f'T2 (med={np.median(lst_t2_sample):.1f}°C)', density=True)
        ax_b.axvline(np.median(lst_t1_sample), color='steelblue', lw=1.5, ls='--')
        ax_b.axvline(np.median(lst_t2_sample), color='crimson',   lw=1.5, ls='--')
        ax_b.set_xlabel('LST (°C)', fontsize=10)
        ax_b.set_ylabel('Density', fontsize=10)
        ax_b.set_title('(b)  LST — T1 vs T2', fontsize=11, fontweight='bold')
        ax_b.legend(fontsize=8)
        ax_b.set_xlim(_lst_lo, _lst_hi)
    else:
        ax_b.set_visible(False)
    ax_b.grid(True, alpha=0.3)

    # ── (c) [1,0] Δ Index Distribution ───────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    _mu_di   = np.mean(delta_index_sample)
    _sd_di   = np.std(delta_index_sample, ddof=1)
    _pct_di  = np.percentile(np.abs(delta_index_sample), percentile_threshold)
    _di_lo, _di_hi = _clip_range(delta_index_sample)
    _di_clip = delta_index_sample[(delta_index_sample >= _di_lo) & (delta_index_sample <= _di_hi)]
    _n_out_c = n - len(_di_clip)
    _lbl_c   = f'Frequency (n={n}' + (f', {_n_out_c} outlier(s) excl.)' if _n_out_c > 0 else ')')
    ax_c.hist(_di_clip, bins=_BINS, color='teal', alpha=_ALPHA,
              edgecolor='none', label=_lbl_c)
    ax_c.axvline(_mu_di,          color='black', lw=1.5, ls='-',
                 label=f'Mean ({_mu_di:+.3f})')
    ax_c.axvline(_mu_di - _sd_di, color='black', lw=1.0, ls='--',
                 label=f'Mean ± sample SD ({_sd_di:.3f})  [dispersion ref.]')
    ax_c.axvline(_mu_di + _sd_di, color='black', lw=1.0, ls='--')
    ax_c.axvline( _pct_di, color='red', lw=1.5, ls='-.',
                 label=f'±P{percentile_threshold}(|Δ|) = {_pct_di:.3f}  [limiar colormap]')
    ax_c.axvline(-_pct_di, color='red', lw=1.5, ls='-.')
    ax_c.set_xlabel(f'Δ {index_name}', fontsize=10)
    ax_c.set_ylabel('Pixels (30 m × 30 m)', fontsize=10)
    ax_c.set_title(f'(c)  Δ {index_name} Distribution  (n = {n})',
                   fontsize=11, fontweight='bold')
    ax_c.legend(fontsize=7.5)
    ax_c.set_xlim(_di_lo, _di_hi)
    ax_c.grid(True, alpha=0.3)

    # ── (d) [1,1] Δ LST Distribution ─────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    _mu_dl   = np.mean(delta_lst_sample)
    _sd_dl   = np.std(delta_lst_sample, ddof=1)
    _pct_dl  = np.percentile(np.abs(delta_lst_sample), percentile_threshold)
    _dl_lo, _dl_hi = _clip_range(delta_lst_sample)
    _dl_clip = delta_lst_sample[(delta_lst_sample >= _dl_lo) & (delta_lst_sample <= _dl_hi)]
    ax_d.hist(_dl_clip, bins=_BINS, color='salmon', alpha=_ALPHA,
              edgecolor='none', label=f'Frequency (n={n}, no filtering)')
    ax_d.axvline(_mu_dl,          color='black', lw=1.5, ls='-',
                 label=f'Mean ({_mu_dl:+.2f}°C)')
    ax_d.axvline(_mu_dl - _sd_dl, color='black', lw=1.0, ls='--',
                 label=f'Mean ± sample SD ({_sd_dl:.2f}°C)  [dispersion ref.]')
    ax_d.axvline(_mu_dl + _sd_dl, color='black', lw=1.0, ls='--')
    ax_d.axvline( _pct_dl, color='red', lw=1.5, ls='-.',
                 label=f'±P{percentile_threshold}(|Δ LST|) = {_pct_dl:.2f}°C  [limiar colormap]')
    ax_d.axvline(-_pct_dl, color='red', lw=1.5, ls='-.')
    ax_d.set_xlabel('Δ LST (°C)', fontsize=10)
    ax_d.set_ylabel('Pixels (30 m × 30 m)', fontsize=10)
    ax_d.set_title(f'(d)  Δ LST (°C) Distribution  (n = {n})',
                   fontsize=11, fontweight='bold')
    ax_d.legend(fontsize=7.5)
    ax_d.set_xlim(_dl_lo, _dl_hi)
    ax_d.grid(True, alpha=0.3)

    # ── Scatter [2,:] Δ Index × Δ LST with 95% PI ────────────────────────────
    ax_sc = fig.add_subplot(gs[2, :])
    ax_sc.scatter(delta_index_sample, delta_lst_sample,
                  color='black', alpha=0.25, s=6, label=f'Δ {index_name} vs Δ LST')
    ax_sc.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax_sc.axvline(0, color='gray', ls='--', lw=0.8, alpha=0.6)

    # Linear regression + constant-width prediction band (parallel)
    _sl, _ic, _rv, _pv, _se = _sst.linregress(delta_index_sample, delta_lst_sample)
    _resid  = delta_lst_sample - (_sl * delta_index_sample + _ic)
    _rmse   = np.sqrt(np.mean(_resid ** 2))
    _s_res  = np.sqrt(np.sum(_resid ** 2) / (n - 2))
    _t95    = _sst.t.ppf(0.975, df=n - 2)
    # x_fit and axis limits clipped to ±3 SD to avoid distortion from outliers
    _xm     = np.mean(delta_index_sample)
    _xsd    = np.std(delta_index_sample)
    _xlim_lo = max(delta_index_sample.min(), _xm - 3 * _xsd)
    _xlim_hi = min(delta_index_sample.max(), _xm + 3 * _xsd)
    _x_fit  = np.linspace(_xlim_lo, _xlim_hi, 300)
    _y_fit  = _sl * _x_fit + _ic
    # Parallel band: prediction interval with constant width ≈ ±t × s_res
    _band   = _t95 * _s_res
    _ci_u   = _y_fit + _band
    _ci_l   = _y_fit - _band

    # Y-axis limits: ±3 SD of Δ LST (robust against thermal outliers)
    _ym     = np.mean(delta_lst_sample)
    _ysd    = np.std(delta_lst_sample)
    _ylim_lo = max(delta_lst_sample.min(), _ym - 3 * _ysd)
    _ylim_hi = min(delta_lst_sample.max(), _ym + 3 * _ysd)

    # Coefficient CIs for the text box
    _xss    = np.sum((delta_index_sample - _xm) ** 2)
    _ci_sl  = _t95 * _se
    _se_ic  = _s_res * np.sqrt(1 / n + _xm ** 2 / _xss)
    _ci_ic  = _t95 * _se_ic

    ax_sc.plot(_x_fit, _y_fit, color='steelblue', lw=2,
               label=f'OLS Regression: Δ {index_name} vs Δ LST')
    ax_sc.fill_between(_x_fit, _ci_l, _ci_u, color='steelblue', alpha=0.18,
                       label='95% PI (±t·sᵣₑₛ, approx. constant width)')
    ax_sc.plot(_x_fit, _ci_u, color='steelblue', lw=1, ls='--', alpha=0.7)
    ax_sc.plot(_x_fit, _ci_l, color='steelblue', lw=1, ls='--', alpha=0.7)
    ax_sc.set_xlim(_xlim_lo, _xlim_hi)
    ax_sc.set_ylim(_ylim_lo, _ylim_hi)

    _box_txt = (
        f'OLS Regression: f(x) = p1·x + p2\n'
        f'95% CI of coefficients (t-Student, df={n-2})\n'
        f'p1 = {_sl:.3f}  ({_sl-_ci_sl:.3f} to {_sl+_ci_sl:.3f})\n'
        f'p2 = {_ic:.3f}  ({_ic-_ci_ic:.3f} to {_ic+_ci_ic:.3f})\n'
        f'R²   = {r2:.3f}\n'
        f'RMSE = {_rmse:.3f}'
    )
    ax_sc.text(0.02, 0.97, _box_txt, transform=ax_sc.transAxes,
               fontsize=8.5, va='top', ha='left', family='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='gray', alpha=0.85))

    ax_sc.set_xlabel(f'Δ {index_name}', fontsize=11)
    ax_sc.set_ylabel('Δ LST (°C)', fontsize=11)
    _sc_iqr_note = (f'  |  {_iqr_out} outlier(s) removed (3×IQR)'
                    if _iqr_out > 0 else '')
    ax_sc.set_title(
        f'Δ {index_name} × Δ LST  |  r = {corr_r:+.3f}   R² = {r2:.3f}   '
        f'[{strength} {direction}]  |  n = {n}{_sc_iqr_note}',
        fontsize=11, fontweight='bold'
    )
    ax_sc.legend(fontsize=9, loc='lower right')
    ax_sc.grid(True, alpha=0.3)

    # ── Panel [3,:] Interpretation ────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[3, :])
    ax_txt.axis('off')

    spearman_rho = correlations['index_lst'].get('spearman_rho', float('nan'))
    spearman_p   = correlations['index_lst'].get('spearman_p',   float('nan'))
    scale_label  = results_dict.get('scale', 30)

    def _sig(p):
        if np.isnan(p):  return '—'
        if p < 0.001:    return 'p < 0.001 ***'
        if p < 0.01:     return f'p = {p:.4f} **'
        if p < 0.05:     return f'p = {p:.4f} *'
        return f'p = {p:.4f} ns'

    _bg = {'strong': '#fff3cd', 'moderate': '#d1ecf1',
           'weak': '#f0f4f8', 'very weak': '#f0f4f8'}.get(strength, '#f0f4f8')
    ax_txt.set_facecolor(_bg)
    ax_txt.patch.set_visible(True)

    col_l, col_r = 0.01, 0.51
    y = 0.97

    def _line(x, yy, txt, fs=8.5, fw='normal', color='black'):
        ax_txt.text(x, yy, txt, transform=ax_txt.transAxes,
                    fontsize=fs, fontweight=fw, va='top', ha='left',
                    color=color, family='monospace' if fw == 'normal' else 'sans-serif')

    _line(col_l, y,
          f'ANALYSIS READING  —  {index_name} × Land Surface Temperature  |  {roi_name}',
          fs=10, fw='bold')
    y -= 0.10
    ax_txt.plot([0.01, 0.99], [y + 0.03, y + 0.03],
                transform=ax_txt.transAxes, color='gray', lw=0.8, alpha=0.6)

    # Left column — period data
    _line(col_l, y, '① PERIOD DATA  (median of sampled pixels)', fs=9, fw='bold')
    y -= 0.11
    if has_abs:
        med_idx_t1 = np.median(index_t1_sample)
        med_idx_t2 = np.median(index_t2_sample)
        med_lst_t1 = np.median(lst_t1_sample)
        med_lst_t2 = np.median(lst_t2_sample)
        _line(col_l, y, f'  {index_name:<6}  T1: {med_idx_t1:+.4f}  →  T2: {med_idx_t2:+.4f}   Δ = {med_idx_t2 - med_idx_t1:+.4f}')
        y -= 0.09
        _line(col_l, y, f'  LST (°C) T1: {med_lst_t1:.2f}      →  T2: {med_lst_t2:.2f}      Δ = {med_lst_t2 - med_lst_t1:+.2f} °C')
        y -= 0.09
    y -= 0.02

    _line(col_l, y, '② CORRELATION ANALYSIS', fs=9, fw='bold')
    y -= 0.11
    _line(col_l, y, f'  Pearson r  :  r = {corr_r:+.4f}   R² = {r2:.4f}   {_sig(corr_p)}')
    y -= 0.09
    if not np.isnan(spearman_rho):
        _line(col_l, y, f'  Spearman ρ :  ρ = {spearman_rho:+.4f}   {_sig(spearman_p)}')
    else:
        _line(col_l, y, f'  Spearman ρ :  not available')
    y -= 0.09
    _line(col_l, y, f'  Strength: {strength} {direction}   |   n = {n} samples   |   scale {scale_label} m')

    # Right column — practical meaning
    y_r = 0.97 - 0.10
    _line(col_r, y_r, '③ PRACTICAL MEANING', fs=9, fw='bold')
    y_r -= 0.11
    words = _meaning.split()
    line_buf = '  '
    for w in words:
        if len(line_buf) + len(w) + 1 > 62:
            _line(col_r, y_r, line_buf)
            y_r -= 0.09
            line_buf = '  ' + w + ' '
        else:
            line_buf += w + ' '
    if line_buf.strip():
        _line(col_r, y_r, line_buf)

    # ── Overall title ─────────────────────────────────────────────────────────
    fig.suptitle(
        f'Integrated Analysis: {index_name} × Temperature × UHI  |  {roi_name}',
        fontsize=13, fontweight='bold', y=0.91
    )

    if output_dir:
        save_path = f'{output_dir}correlation_analysis_{index_name}_{roi_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        _print(f"   Figure saved: {save_path}")

    _print("   Plots created successfully.")
    return fig


# ==============================================================================
# SECTION 5.2 — LST 10m vs 30m Correlation + Interactive Split Map
# ==============================================================================

# ==============================================================================
# create_zonal_comparison_plots — Comparative Visualization 9.2
# ==============================================================================

def create_zonal_comparison_plots(zonal_results, roi_name=None, figsize=(16, 14)):
    """
    Generates a comparative figure between groups with and without new buildings.

    Layout (5 rows × 2 columns + full width):
      Row 0: (a) Δ LST With vs Without  |  (b) Δ Index With vs Without
      Row 1: (c) Index T1 With vs Without  |  (d) Index T2 With vs Without
      Row 2: (e) LST T1 With vs Without  |  (f) LST T2 With vs Without
      Row 3: Scatter Δ Index × Δ LST by group with regressions (full width)
      Row 4: Text panel with statistics (full width)

    Args:
        zonal_results (dict): Return value of apply_building_change_mask_to_results.
        roi_name (str): Study area name for the title.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats as _sst

    index_name = zonal_results['index_name'].upper()
    _roi = roi_name or zonal_results.get('roi_name', '')

    sc  = zonal_results['samples_change']
    snc = zonal_results['samples_no_change']

    delta_idx_chg  = sc['delta_index']
    delta_lst_chg  = sc['delta_lst']
    delta_idx_nchg = snc['delta_index']
    delta_lst_nchg = snc['delta_lst']

    if len(delta_lst_chg) < 10 or len(delta_lst_nchg) < 10:
        print('   ⚠️  Insufficient samples to generate comparative plot.')
        return None

    C_CHG  = '#E05252'
    C_NCHG = '#5285E0'
    _ALPHA = 0.55
    _BINS  = 40

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[1.0, 1.6, 1.1],
                            hspace=0.40, wspace=0.35)

    def _hist_ax(ax, data_chg, data_nchg, xlabel, title, ref_zero=True, fmt='.3f'):
        fmt_str = f'{{:+{fmt}}}'
        ax.hist(data_chg,  bins=_BINS, color=C_CHG,  alpha=_ALPHA, density=True,
                label=f'With new bldgs. (med={fmt_str.format(np.median(data_chg))})')
        ax.hist(data_nchg, bins=_BINS, color=C_NCHG, alpha=_ALPHA, density=True,
                label=f'Without new bldgs. (med={fmt_str.format(np.median(data_nchg))})')
        ax.axvline(np.median(data_chg),  color=C_CHG,  lw=1.5, ls='--')
        ax.axvline(np.median(data_nchg), color=C_NCHG, lw=1.5, ls='--')
        if ref_zero:
            ax.axvline(0, color='gray', lw=0.8, ls='-', alpha=0.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Linha 0: deltas ──────────────────────────────────────────────────────
    _hist_ax(fig.add_subplot(gs[0, 0]),
             delta_lst_chg, delta_lst_nchg,
             'Δ LST (°C)', '(a)  Δ LST — With vs Without New Buildings',
             fmt='.2f')

    _hist_ax(fig.add_subplot(gs[0, 1]),
             delta_idx_chg, delta_idx_nchg,
             f'Δ {index_name}', f'(b)  Δ {index_name} — With vs Without New Buildings')

    # ── Row 1: Scatter Δ Index × Δ LST by group ──────────────────────────────
    ax_sc = fig.add_subplot(gs[1, :])
    ax_sc.scatter(delta_idx_nchg, delta_lst_nchg,
                  color=C_NCHG, alpha=0.20, s=7, rasterized=True,
                  label='Without new buildings')
    ax_sc.scatter(delta_idx_chg, delta_lst_chg,
                  color=C_CHG, alpha=0.20, s=7, rasterized=True,
                  label='With new buildings')

    for d_idx, d_lst, color, lbl in [
        (delta_idx_nchg, delta_lst_nchg, C_NCHG, 'Without new bldgs.'),
        (delta_idx_chg,  delta_lst_chg,  C_CHG,  'With new bldgs.'),
    ]:
        sl, ic, rv, pv, _ = _sst.linregress(d_idx, d_lst)
        r2 = rv ** 2
        x_line = np.linspace(d_idx.min(), d_idx.max(), 100)
        ax_sc.plot(x_line, sl * x_line + ic, color=color, lw=2.0,
                   label=f'Regression {lbl}  (R²={r2:.3f},  y={sl:+.3f}x{ic:+.3f})')

    ax_sc.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax_sc.axvline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax_sc.set_xlabel(f'Δ {index_name}', fontsize=11)
    ax_sc.set_ylabel('Δ LST (°C)', fontsize=11)
    ax_sc.set_title(f'Δ {index_name} × Δ LST — With vs Without New Buildings',
                    fontsize=11, fontweight='bold')
    ax_sc.legend(fontsize=9, loc='lower right')
    ax_sc.grid(True, alpha=0.25)

    # ── Linha 2: Painel de texto ──────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis('off')
    ax_txt.set_facecolor('#f0f4f8')
    ax_txt.patch.set_visible(True)

    corr  = zonal_results.get('correlations', {}).get('index_lst', {})
    pr    = corr.get('pearson_r',    float('nan'))
    pp    = corr.get('pearson_p',    float('nan'))
    r2v   = corr.get('r2',           float('nan'))
    sr    = corr.get('spearman_rho', float('nan'))
    sp    = corr.get('spearman_p',   float('nan'))
    stat  = zonal_results.get('statistics', {})
    p_mw  = stat.get('p_mannwhitney_lst')
    d_coh = stat.get('cohens_d_lst')

    def _sig(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return 'n/d'
        if p < 0.001: return 'p < 0.001 ***'
        if p < 0.01:  return f'p = {p:.4f} **'
        if p < 0.05:  return f'p = {p:.4f} *'
        return f'p = {p:.4f} ns'

    def _effect(d):
        if d is None or (isinstance(d, float) and np.isnan(d)):
            return '—'
        ad = abs(d)
        label = ('negligible' if ad < 0.2 else
                 'small'      if ad < 0.5 else
                 'medium'     if ad < 0.8 else 'Large')
        return f'{d:+.3f} ({label})'

    def _fmt(v, spec='+.4f'):
        return '—' if (v is None or (isinstance(v, float) and np.isnan(v))) else format(v, spec)

    def _line(x, y_pos, txt, fs=9, fw='normal', color='black'):
        ax_txt.text(x, y_pos, txt, transform=ax_txt.transAxes,
                    fontsize=fs, fontweight=fw, va='top', ha='left', color=color,
                    family='monospace' if fw == 'normal' else 'sans-serif')

    col_l, col_r = 0.02, 0.52
    y = 0.97

    _line(col_l, y,
          f'STATISTICAL SUMMARY  —  {index_name} × Thermal Impact  |  {_roi}',
          fs=10, fw='bold')
    y -= 0.10
    ax_txt.plot([0.01, 0.99], [y + 0.03, y + 0.03],
                transform=ax_txt.transAxes, color='gray', lw=0.8, alpha=0.6)

    _line(col_l, y, '① GROUP COMPARISON  (Δ LST, °C)', fs=9, fw='bold')
    y -= 0.11
    _line(col_l, y, f'  With new bldgs.:    {np.mean(delta_lst_chg):+.3f} ± {np.std(delta_lst_chg):.3f} °C')
    y -= 0.09
    _line(col_l, y, f'  Without new bldgs.: {np.mean(delta_lst_nchg):+.3f} ± {np.std(delta_lst_nchg):.3f} °C')
    y -= 0.09
    _line(col_l, y, f'  Difference:         {np.mean(delta_lst_chg) - np.mean(delta_lst_nchg):+.3f} °C')
    y -= 0.09
    y -= 0.02
    _line(col_l, y, f'  Mann-Whitney U:     {_sig(p_mw)}')
    y -= 0.09
    _line(col_l, y, f"  Cohen's d:          {_effect(d_coh)}")

    y_r = 0.97 - 0.10
    _line(col_r, y_r, f'② CORRELATION  Δ {index_name} × Δ LST  (with new bldgs.)', fs=9, fw='bold')
    y_r -= 0.11
    _line(col_r, y_r, f'  Pearson r  :  r = {_fmt(pr)}   R² = {_fmt(r2v, ".4f")}   {_sig(pp)}')
    y_r -= 0.09
    _line(col_r, y_r, f'  Spearman ρ :  ρ = {_fmt(sr)}                  {_sig(sp)}')
    y_r -= 0.09
    n_corr = len(delta_idx_chg)
    _line(col_r, y_r, f'  n = {n_corr} samples (new buildings area)   |   scale {zonal_results.get("scale", 30)} m')

    fig.suptitle(
        f'Zonal Analysis — {index_name} × Thermal | With vs Without New Buildings'
        + (f' ({_roi})' if _roi else ''),
        fontsize=13, fontweight='bold', y=0.93
    )

    return fig


# ==============================================================================
# 5. PREVIEW VISUALIZATION (T1/T2 SELECTION)
# ==============================================================================

def visualize_selected_images(l8_id, s2_id, roi, zoom=12, Map=None):
    """
    Creates an interactive map for preview of selected images.

    Displays the Sentinel-2 RGB composite and the thermal band (ST_B10) of
    Landsat 8 for the region of interest. Used in selection cells 3.2 and 3.4.

    Args:
        l8_id (str): Landsat 8 image ID (e.g. 'LC08_218063_20190810').
        s2_id (str): Sentinel-2 image ID (e.g. '20190810T131251_20190810T1...').
        roi (ee.Geometry): Region of interest for centering and clipping.
        zoom (int): Initial zoom level. Default: 12.
        Map (geemap.Map, optional): Existing map. If None, creates a new one.

    Returns:
        geemap.Map: Configured interactive map.
    """
    if Map is None:
        Map = geemap.Map(height=800)
        center = roi.centroid(maxError=1).getInfo()['coordinates']
        Map.setCenter(center[0], center[1], zoom)
        Map.add_basemap('SATELLITE')

    # ---- Landsat 8 (daily mosaic → covers entire ROI) ----
    try:
        # Extract date from ID: 'LC08_218063_20190810' → '2019-08-10'
        _l8_date_raw = l8_id.split('_')[-1]
        _l8_date = f'{_l8_date_raw[:4]}-{_l8_date_raw[4:6]}-{_l8_date_raw[6:8]}'
        l8_col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterDate(_l8_date, ee.Date(_l8_date).advance(1, 'day'))
                  .filterBounds(roi))
        l8_mosaic = l8_col.mosaic()
        # LST in Celsius (ST_B10: scale=0.00341802, offset=149.0 K → −273.15 → °C)
        l8_lst = (l8_mosaic
                  .select('ST_B10')
                  .multiply(0.00341802).add(149.0)
                  .subtract(273.15)
                  .clip(roi))
        vis_lst_prev = {'min': 20, 'max': 60,
                        'palette': ['#313695','#4575b4','#74add1','#abd9e9',
                                    '#e0f3f8','#ffffbf','#fee090','#fdae61',
                                    '#f46d43','#d73027','#a50026']}
        
        l8_rgb = (l8_mosaic
                  .select(['SR_B4', 'SR_B3', 'SR_B2'])
                  .multiply(0.0000275).add(-0.2)
                  .clip(roi))
        vis_l8 = {'min': 0, 'max': 0.3, 'gamma': 1.3}
        Map.addLayer(l8_rgb, vis_l8, '🛰️ Landsat 8 (RGB)', True)
        
    except Exception as e:
        print(f"Error loading Landsat 8: {e}")

    # ---- Sentinel-2 (daily mosaic → covers entire ROI) ----
    try:
        # Extract date from ID: '20190810T131251_20190810T131257_T24MUB' → '2019-08-10'
        _s2_date_raw = s2_id[:8]
        _s2_date = f'{_s2_date_raw[:4]}-{_s2_date_raw[4:6]}-{_s2_date_raw[6:8]}'
        s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterDate(_s2_date, ee.Date(_s2_date).advance(1, 'day'))
                  .filterBounds(roi))
        s2_img = s2_col.mosaic().multiply(0.0001).clip(roi)
        vis_s2 = {'min': 0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2'], 'gamma': 1.3}
        Map.addLayer(s2_img, vis_s2, '🛰️ Sentinel-2 (RGB)', True)
    except Exception as e:
        print(f"Error loading Sentinel-2: {e}")

    # LST layer
    Map.addLayer(l8_lst, vis_lst_prev, '🌡️ LST — Temperature (°C)', True, 0.5)

    # ---- ROI ----
    empty = ee.Image().byte()
    roi_outline = empty.paint(
        featureCollection=ee.FeatureCollection([ee.Feature(roi)]),
        color=2, width=2
    )
    Map.addLayer(roi_outline, {'palette': ['yellow']}, '📍 Study Area (ROI)', True)

    # ---- LST legend ----
    try:
        Map.add_colorbar(
            vis_lst_prev,
            label='Land Surface Temperature — LST (°C)',
            orientation='horizontal',
            transparent_bg=True,
            position='bottomleft',
        )
    except Exception as e:
        print(f"Error adding LST legend: {e}")

    return Map
