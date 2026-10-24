# ==============================================================================
# export_utils.py — Export Functions for Google Drive
# ==============================================================================
# Export of bitemporal results (LST, UHI, indices, deltas, CDI, Sentinel-1/2
# inputs of the change detection model) to Google Drive.
# ==============================================================================

import ee

# --- Cross-module imports ---
from modulos.image_processing import generate_sentinel1_composite_from_s2_date


EXPORT_EXTENTS = {
    'roi': 'Analysis area (ROI)',
    'perimeter': 'Thermal reference (urban perimeter)',
    'both': 'Both',
}

# Products that only exist inside the ROI: Sentinel-2 / Sentinel-1 are loaded
# for the ROI only (input of the change-detection model), so they cannot be
# exported for the urban perimeter.
_ROI_ONLY_FLAGS = ('export_sentinel2', 'export_sentinel1', 'export_sentinel2_indices',
                   'export_delta_sentinel2_indices', 'export_cdi_sentinel2')


def _clip_tree(obj, geometry):
    """Copy of nested dicts with every ee.Image clipped to `geometry` (other values kept)."""
    if isinstance(obj, ee.Image):
        return obj.clip(geometry)
    if isinstance(obj, dict):
        return {k: _clip_tree(v, geometry) for k, v in obj.items()}
    return obj


def distinct_reference_geometry(results, roi):
    """
    The thermal reference area (urban perimeter) stored in the results, or None
    when the z-score was referenced to the ROI itself (no separate perimeter).
    """
    ref = results.get('reference_geometry')
    roi_cfg = results.get('t1', {}).get('config', {}).get('geometry')
    if ref is None or ref is roi or ref is roi_cfg:
        return None
    return ref


def _normalise_extent(export_extent):
    key = str(export_extent).strip().lower()
    for k, label in EXPORT_EXTENTS.items():
        if key in (k, label.lower()):
            return k
    if 'perimeter' in key or 'reference' in key:
        return 'perimeter'
    if 'both' in key or 'ambos' in key:
        return 'both'
    return 'roi'


def export_bitemporal_results(results, config, roi, stats_change=None, s1_t1=None, s1_t2=None,
                              export_extent='roi', reference_geometry=None, **flags):
    """
    Exports the bitemporal products for the analysis area, the urban perimeter
    (thermal reference of the SUHI z-score), or both.

    Landsat products (LST, SUHI z-score, intensity classes, Landsat bands and
    indices, their T2 − T1 deltas and the Landsat CDI) are computed over
    ROI ∪ perimeter (calculate_bitemporal_lst), so they can be exported for
    either extent. Sentinel-2 / Sentinel-1 products exist only inside the ROI
    and are skipped for the perimeter.

    Every raster is clipped to its extent polygon (pixels outside = no data).
    File names: ROI → `<product>_<site>_...` (as before);
                perimeter → `<product>_<site>_perimeter_...`.

    Args:
        results, config, roi, stats_change, s1_t1, s1_t2: as in `_export_products`.
        export_extent (str): 'roi', 'perimeter' or 'both' (or the labels in
            EXPORT_EXTENTS, as used by the notebook form).
        reference_geometry: perimeter geometry. None = the one stored in
            results['reference_geometry'] (cell 2.3).
        **flags: product switches of `_export_products` (export_lst_30m=True, ...).

    Returns:
        list: started ee.batch tasks (all extents).
    """
    extent = _normalise_extent(export_extent)
    ref = reference_geometry if reference_geometry is not None else distinct_reference_geometry(results, roi)

    jobs = []
    if extent in ('roi', 'both'):
        jobs.append(('roi', roi, config['site_name']))
    if extent in ('perimeter', 'both'):
        if ref is None:
            print("⚠️  No separate urban perimeter: the SUHI reference is the ROI itself "
                  "(cell 2.3 = 'Same as analysis area'). Perimeter export skipped.")
        else:
            jobs.append(('perimeter', ref, f"{config['site_name']}_perimeter"))

    tasks = []
    for key, geom, site in jobs:
        print("\n" + "#" * 80)
        print(f"# EXTENT: {EXPORT_EXTENTS[key]}  →  file names with '{site}'")
        print("#" * 80)
        fl = dict(flags)
        if key == 'perimeter':
            skipped = [f for f in _ROI_ONLY_FLAGS if fl.get(f, f in ('export_sentinel2', 'export_sentinel1'))]
            for f in _ROI_ONLY_FLAGS:
                fl[f] = False
            if skipped:
                print("ℹ️  Sentinel-1/2 products exist only inside the ROI — not exported for the "
                      "perimeter: " + ', '.join(s.replace('export_', '') for s in skipped))
        cfg = {**config, 'site_name': site}
        tasks += _export_products(
            _clip_tree(results, geom), cfg, geom,
            stats_change=_clip_tree(stats_change, geom) if stats_change else stats_change,
            s1_t1=s1_t1.clip(geom) if s1_t1 is not None else None,
            s1_t2=s1_t2.clip(geom) if s1_t2 is not None else None,
            **fl)
    return tasks


def _export_products(
    results,
    config,
    roi,
    stats_change=None,
    s1_t1=None,
    s1_t2=None,
    export_lst_30m=True,
    export_uhi_30m=True,
    export_uhi_intensity_30m=True,
    export_landsat_indices=False,
    export_sentinel2_indices=False,
    landsat_indices_list='all',
    sentinel2_indices_list='all',
    export_sentinel2=True,
    export_sentinel1=True,
    sentinel2_bands='all',
    export_landsat=False,
    landsat_bands='all',
    export_delta_landsat_indices=False,
    export_delta_sentinel2_indices=False,
    export_delta_uhi_30m=False,
    export_delta_uhi_intensity_30m=False,
    export_delta_lst_30m=False,
    export_cdi_landsat=False,
    export_cdi_sentinel2=False,
):
    """
    Exports selected products from the bitemporal thermal analysis to Google Drive.

    Args:
        results (dict): Bitemporal processing results.
        config (dict): Configuration with export parameters.
        roi: Region of interest geometry (ee.Geometry).
        stats_change (dict): Change statistics with UHI images (optional).
        s1_t1 (ee.Image, optional): Sentinel-1 T1 image.
        s1_t2 (ee.Image, optional): Sentinel-1 T2 image.
        export_lst_30m (bool): Export Landsat 30m LST.
        export_uhi_30m (bool): Export normalized 30m UHI.
        export_uhi_intensity_30m (bool): Export 30m UHI intensity classification.
        export_landsat_indices (bool): Export Landsat spectral indices.
        export_sentinel2_indices (bool): Export Sentinel-2 spectral indices.
        landsat_indices_list (str or list): 'all' for all, or list of index names.
        sentinel2_indices_list (str or list): 'all' for all, or list of index names.
        export_sentinel2 (bool): Export Sentinel-2 imagery.
        export_sentinel1 (bool): Export Sentinel-1 imagery.
        sentinel2_bands (str or list): 'all', 'rgb', or custom band list.
        export_landsat (bool): Export Landsat SR optical imagery (T1 and T2).
        landsat_bands (str or list): 'all', 'rgb', or custom band list for Landsat export.
        export_delta_landsat_indices (bool): Export T2-T1 delta for Landsat spectral indices (30m).
        export_delta_sentinel2_indices (bool): Export T2-T1 delta for Sentinel-2 spectral indices (10m).
        export_delta_uhi_30m (bool): Export T2-T1 delta for UHI 30m (z-score difference).
        export_delta_uhi_intensity_30m (bool): Export T2-T1 delta for UHI Intensity classification 30m.
        export_delta_lst_30m (bool): Export T2-T1 delta for LST 30m (degrees Celsius).
        export_cdi_landsat (bool): Export CDI = delta_LST_30m x delta_LandsatIndex per index (30m).
        export_cdi_sentinel2 (bool): Export CDI = delta_LST_30m x delta_S2Index per index (10m).

    Returns:
        list: List of started export tasks.
    """
    print("\n" + "=" * 80)
    print("EXPORTING SELECTED PRODUCTS TO GOOGLE DRIVE")
    print("=" * 80)

    tasks = []

    # Sentinel-1 is exported on exactly the same grid as Sentinel-2 (same CRS,
    # 10 m, same region): the change detection model stacks both rasters
    # pixel-by-pixel, so any CRS/scale mismatch would misalign or break them.
    crs = config['export_crs']
    scale = 10

    # Sentinel-1 polarizations
    s1_pols = config.get('s1_polarizations', ['VV', 'VH'])

    # Export summary
    print("\n📋 EXPORT SUMMARY:")
    print(f"  • LST 30m (Landsat): {'YES' if export_lst_30m else 'NO'}")
    print(f"  • UHI 30m (Normalized): {'YES' if export_uhi_30m else 'NO'}")
    print(f"  • UHI Intensity Classification 30m (Deng et al. 2023): {'YES' if export_uhi_intensity_30m else 'NO'}")
    print(f"  • Landsat Indices (30m): {'YES' if export_landsat_indices else 'NO'}")
    print(f"  • Landsat SR Imagery (30m): {'YES' if export_landsat else 'NO'}")
    print(f"  • Delta LST 30m T2-T1: {'YES' if export_delta_lst_30m else 'NO'}")
    print(f"  • Delta Landsat Indices T2-T1 (30m): {'YES' if export_delta_landsat_indices else 'NO'}")
    print(f"  • Delta Sentinel-2 Indices T2-T1 (10m): {'YES' if export_delta_sentinel2_indices else 'NO'}")
    print(f"  • Delta UHI 30m T2-T1: {'YES' if export_delta_uhi_30m else 'NO'}")
    print(f"  • Delta UHI Intensity 30m T2-T1: {'YES' if export_delta_uhi_intensity_30m else 'NO'}")
    print(f"  • CDI Landsat (dLST_30m x dIndex, 30m): {'YES' if export_cdi_landsat else 'NO'}")
    print(f"  • CDI Sentinel-2 (dLST_30m x dIndex, 10m): {'YES' if export_cdi_sentinel2 else 'NO'}")

    # ==================== EXPORT LST 30m (LANDSAT) ====================
    if export_lst_30m:
        print("\n" + "=" * 80)
        print("EXPORTING LST 30m (LANDSAT)")
        print("=" * 80)

        # T1
        try:
            print("\n📊 Exporting LST 30m T1...")
            task_lst30_t1 = ee.batch.Export.image.toDrive(
                image=results['lst_30m_t1'],
                description=f'LST_30m_{config["site_name"]}_T1',
                folder=config['export_folder'],
                fileNamePrefix=f'LST_30m_{config["site_name"]}_t1',
                region=roi,
                crs=config['export_crs'],
                scale=30,
                maxPixels=1e13,
                fileFormat='GeoTIFF',
                formatOptions={'cloudOptimized': True}
            )
            task_lst30_t1.start()
            tasks.append(task_lst30_t1)
            print(f"  ✓ LST 30m T1 started")
        except Exception as e:
            print(f"  ⚠️ Failed: {e}")

        # T2
        try:
            print("📊 Exporting LST 30m T2...")
            task_lst30_t2 = ee.batch.Export.image.toDrive(
                image=results['lst_30m_t2'],
                description=f'LST_30m_{config["site_name"]}_T2',
                folder=config['export_folder'],
                fileNamePrefix=f'LST_30m_{config["site_name"]}_t2',
                region=roi,
                crs=config['export_crs'],
                scale=30,
                maxPixels=1e13,
                fileFormat='GeoTIFF',
                formatOptions={'cloudOptimized': True}
            )
            task_lst30_t2.start()
            tasks.append(task_lst30_t2)
            print(f"  ✓ LST 30m T2 started")
        except Exception as e:
            print(f"  ⚠️ Failed: {e}")

    # ==================== EXPORT UHI 30m (NORMALIZED) ====================
    if export_uhi_30m and stats_change and 'UHI_30m' in stats_change:
        print("\n" + "=" * 80)
        print("EXPORTING UHI 30m (NORMALIZED)")
        print("=" * 80)
        print("  Note: UHI values are z-scores (standard deviations from the mean)")

        try:
            uhi_images = stats_change['UHI_30m']['images']

            # T1
            if 't1' in uhi_images:
                print("\n🔥 Exporting UHI 30m T1...")
                task_uhi30_t1 = ee.batch.Export.image.toDrive(
                    image=uhi_images['t1'],
                    description=f'UHI_30m_{config["site_name"]}_T1',
                    folder=config['export_folder'],
                    fileNamePrefix=f'UHI_30m_{config["site_name"]}_t1',
                    region=roi,
                    crs=config['export_crs'],
                    scale=30,
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_uhi30_t1.start()
                tasks.append(task_uhi30_t1)
                print(f"  ✓ UHI 30m T1 started")

            # T2
            if 't2' in uhi_images:
                print("🔥 Exporting UHI 30m T2...")
                task_uhi30_t2 = ee.batch.Export.image.toDrive(
                    image=uhi_images['t2'],
                    description=f'UHI_30m_{config["site_name"]}_T2',
                    folder=config['export_folder'],
                    fileNamePrefix=f'UHI_30m_{config["site_name"]}_t2',
                    region=roi,
                    crs=config['export_crs'],
                    scale=30,
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_uhi30_t2.start()
                tasks.append(task_uhi30_t2)
                print(f"  ✓ UHI 30m T2 started")

        except Exception as e:
            print(f"  ⚠️ Failed to export UHI 30m: {e}")
    elif export_uhi_30m:
        print("\n  ℹ️ UHI 30m not available (stats_change not provided or no UHI data)")

    # ==================== EXPORT UHI INTENSITY CLASSIFICATION 30m ====================
    if export_uhi_intensity_30m and stats_change and 'UHI_30m' in stats_change:
        print("\n" + "=" * 80)
        print("EXPORTING UHI INTENSITY CLASSIFICATION 30m (Deng et al. 2023)")
        print("=" * 80)
        print("  Note: Classes 1-LTZ, 2-SLTZ, 3-MTZ, 4-SHTZ, 5-HTZ")

        try:
            intensity_class = stats_change['UHI_30m']['intensity_classification']

            # T1
            if 't1' in intensity_class:
                print("\n🔥 Exporting UHI Intensity Classification 30m T1...")
                task_intensity30_t1 = ee.batch.Export.image.toDrive(
                    image=intensity_class['t1'],
                    description=f'UHI_Intensity_30m_{config["site_name"]}_T1',
                    folder=config['export_folder'],
                    fileNamePrefix=f'UHI_Intensity_30m_{config["site_name"]}_t1',
                    region=roi,
                    crs=config['export_crs'],
                    scale=30,
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_intensity30_t1.start()
                tasks.append(task_intensity30_t1)
                print(f"  ✓ UHI Intensity Classification 30m T1 started")

            # T2
            if 't2' in intensity_class:
                print("🔥 Exporting UHI Intensity Classification 30m T2...")
                task_intensity30_t2 = ee.batch.Export.image.toDrive(
                    image=intensity_class['t2'],
                    description=f'UHI_Intensity_30m_{config["site_name"]}_T2',
                    folder=config['export_folder'],
                    fileNamePrefix=f'UHI_Intensity_30m_{config["site_name"]}_t2',
                    region=roi,
                    crs=config['export_crs'],
                    scale=30,
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_intensity30_t2.start()
                tasks.append(task_intensity30_t2)
                print(f"  ✓ UHI Intensity Classification 30m T2 started")

        except Exception as e:
            print(f"  ⚠️ Failed to export UHI Intensity Classification 30m: {e}")
    elif export_uhi_intensity_30m:
        print("\n  ℹ️ UHI Intensity Classification 30m not available (stats_change not provided or no data)")

    # ==================== EXPORT LANDSAT INDICES ====================
    if export_landsat_indices and 'indices_landsat' in results:
        print("\n" + "=" * 80)
        print("EXPORTING LANDSAT SPECTRAL INDICES (30m)")
        print("=" * 80)

        # Extract years from config
        year_t1 = config.get('t1_start_date', 't1').split('-')[0] if 't1_start_date' in config else 't1'
        year_t2 = config.get('t2_start_date', 't2').split('-')[0] if 't2_start_date' in config else 't2'

        # Get all available indices
        indices_landsat_t1 = results['indices_landsat'].get('t1', {})
        indices_landsat_t2 = results['indices_landsat'].get('t2', {})

        # Determine which indices to export
        if landsat_indices_list == 'all':
            indices_to_export_t1 = indices_landsat_t1
            indices_to_export_t2 = indices_landsat_t2
            print(f"  Exporting all available Landsat indices")
        else:
            indices_to_export_t1 = {}
            indices_to_export_t2 = {}

            for idx_name in landsat_indices_list:
                idx_lower = idx_name.lower()
                if idx_lower in indices_landsat_t1:
                    indices_to_export_t1[idx_lower] = indices_landsat_t1[idx_lower]
                if idx_lower in indices_landsat_t2:
                    indices_to_export_t2[idx_lower] = indices_landsat_t2[idx_lower]

            print(f"  Exporting selected indices: {[idx.upper() for idx in landsat_indices_list]}")

        # Export T1 indices
        for index_name, index_image in indices_to_export_t1.items():
            try:
                print(f"\n📊 Exporting Landsat {index_name.upper()} T1...")
                task_name = f"Landsat_{index_name.upper()}_{config['site_name']}_T1_{year_t1}"
                task = ee.batch.Export.image.toDrive(
                    image=index_image.clip(roi),
                    description=task_name,
                    folder=config['export_folder'],
                    fileNamePrefix=task_name,
                    region=roi,
                    scale=30,
                    crs=config['export_crs'],
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task.start()
                tasks.append(task)
                print(f"  ✓ {index_name.upper()} T1 started")
            except Exception as e:
                print(f"  ⚠️ Failed to export {index_name.upper()} T1: {e}")

        # Export T2 indices
        for index_name, index_image in indices_to_export_t2.items():
            try:
                print(f"\n📊 Exporting Landsat {index_name.upper()} T2...")
                task_name = f"Landsat_{index_name.upper()}_{config['site_name']}_T2_{year_t2}"
                task = ee.batch.Export.image.toDrive(
                    image=index_image.clip(roi),
                    description=task_name,
                    folder=config['export_folder'],
                    fileNamePrefix=task_name,
                    region=roi,
                    scale=30,
                    crs=config['export_crs'],
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task.start()
                tasks.append(task)
                print(f"  ✓ {index_name.upper()} T2 started")
            except Exception as e:
                print(f"  ⚠️ Failed to export {index_name.upper()} T2: {e}")
    elif export_landsat_indices:
        print("\n  ℹ️ Landsat indices not available for export")

    # ==================== EXPORT SENTINEL-2 INDICES ====================
    if export_sentinel2_indices and 'indices_sentinel2' in results:
        print("\n" + "=" * 80)
        print("EXPORTING SENTINEL-2 SPECTRAL INDICES (10m)")
        print("=" * 80)

        # Extract years from config
        year_t1 = config.get('t1_start_date', 't1').split('-')[0] if 't1_start_date' in config else 't1'
        year_t2 = config.get('t2_start_date', 't2').split('-')[0] if 't2_start_date' in config else 't2'

        # Get all available indices
        indices_s2_t1 = results['indices_sentinel2'].get('t1', {})
        indices_s2_t2 = results['indices_sentinel2'].get('t2', {})

        # Determine which indices to export
        if sentinel2_indices_list == 'all':
            indices_to_export_t1 = indices_s2_t1
            indices_to_export_t2 = indices_s2_t2
            print(f"  Exporting all available Sentinel-2 indices")
        else:
            indices_to_export_t1 = {}
            indices_to_export_t2 = {}

            for idx_name in sentinel2_indices_list:
                idx_lower = idx_name.lower()
                if idx_lower in indices_s2_t1:
                    indices_to_export_t1[idx_lower] = indices_s2_t1[idx_lower]
                if idx_lower in indices_s2_t2:
                    indices_to_export_t2[idx_lower] = indices_s2_t2[idx_lower]

            print(f"  Exporting selected indices: {[idx.upper() for idx in sentinel2_indices_list]}")

        # Export T1 indices
        for index_name, index_image in indices_to_export_t1.items():
            try:
                print(f"\n📊 Exporting Sentinel-2 {index_name.upper()} T1...")
                task_name = f"Sentinel2_{index_name.upper()}_{config['site_name']}_T1_{year_t1}"
                task = ee.batch.Export.image.toDrive(
                    image=index_image.clip(roi),
                    description=task_name,
                    folder=config['export_folder'],
                    fileNamePrefix=task_name,
                    region=roi,
                    scale=10,
                    crs=config['export_crs'],
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task.start()
                tasks.append(task)
                print(f"  ✓ {index_name.upper()} T1 started")
            except Exception as e:
                print(f"  ⚠️ Failed to export {index_name.upper()} T1: {e}")

        # Export T2 indices
        for index_name, index_image in indices_to_export_t2.items():
            try:
                print(f"\n📊 Exporting Sentinel-2 {index_name.upper()} T2...")
                task_name = f"Sentinel2_{index_name.upper()}_{config['site_name']}_T2_{year_t2}"
                task = ee.batch.Export.image.toDrive(
                    image=index_image.clip(roi),
                    description=task_name,
                    folder=config['export_folder'],
                    fileNamePrefix=task_name,
                    region=roi,
                    scale=10,
                    crs=config['export_crs'],
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task.start()
                tasks.append(task)
                print(f"  ✓ {index_name.upper()} T2 started")
            except Exception as e:
                print(f"  ⚠️ Failed to export {index_name.upper()} T2: {e}")
    elif export_sentinel2_indices:
        print("\n  ℹ️ Sentinel-2 indices not available for export")

    # ==================== EXPORT LANDSAT SR IMAGERY ====================
    if export_landsat:
        print("\n" + "=" * 80)
        print("EXPORTING LANDSAT SR IMAGERY (30m)")
        print("=" * 80)

        l8_t1 = results.get('t1', {}).get('landsat_image')
        l8_t2 = results.get('t2', {}).get('landsat_image')

        # Determine bands to export
        if landsat_bands == 'rgb':
            bands_to_export_l8 = ['SR_B4', 'SR_B3', 'SR_B2']
            bands_label_l8 = 'RGB'
        elif landsat_bands == 'all':
            try:
                if l8_t1 is not None:
                    bands_to_export_l8 = l8_t1.bandNames().getInfo()
                    bands_label_l8 = 'All'
                else:
                    bands_to_export_l8 = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
                    bands_label_l8 = 'Default (T1 unavailable)'
                    print("  ⚠️ Landsat T1 not available. Using default bands.")
            except Exception as e:
                print(f"  ⚠️ Error getting Landsat T1 band names: {e}. Using default bands.")
                bands_to_export_l8 = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
                bands_label_l8 = 'Default (error getting band names)'
        else:
            bands_to_export_l8 = landsat_bands
            bands_label_l8 = 'Custom'

        print(f"  Bands ({bands_label_l8}): {bands_to_export_l8}")

        # T1
        if l8_t1 is not None:
            try:
                print(f"\n📊 Exporting Landsat SR T1...")
                task_l8_t1 = ee.batch.Export.image.toDrive(
                    image=l8_t1.select(bands_to_export_l8),
                    description=f'Landsat_{config["site_name"]}_T1',
                    folder=config['export_folder'],
                    fileNamePrefix=f'landsat_{config["site_name"]}_t1',
                    region=roi,
                    crs=config['export_crs'],
                    scale=30,
                    maxPixels=1e12,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_l8_t1.start()
                tasks.append(task_l8_t1)
                print(f"  ✓ Landsat SR T1 started")
            except Exception as e:
                print(f"  ⚠️ Failed: {e}")
        else:
            print("  ℹ️ Landsat SR T1 not available for export.")

        # T2
        if l8_t2 is not None:
            try:
                print(f"📊 Exporting Landsat SR T2...")
                task_l8_t2 = ee.batch.Export.image.toDrive(
                    image=l8_t2.select(bands_to_export_l8),
                    description=f'Landsat_{config["site_name"]}_T2',
                    folder=config['export_folder'],
                    fileNamePrefix=f'landsat_{config["site_name"]}_t2',
                    region=roi,
                    crs=config['export_crs'],
                    scale=30,
                    maxPixels=1e12,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_l8_t2.start()
                tasks.append(task_l8_t2)
                print(f"  ✓ Landsat SR T2 started")
            except Exception as e:
                print(f"  ⚠️ Failed: {e}")
        else:
            print("  ℹ️ Landsat SR T2 not available for export.")

    # ==================== EXPORT SENTINEL-2 ====================
    if export_sentinel2:
        print("\n" + "=" * 80)
        print("EXPORTING SENTINEL-2")
        print("=" * 80)

        # Determine which bands to export
        if sentinel2_bands == 'rgb':
            bands_to_export = ['B4', 'B3', 'B2']
            bands_label = 'RGB'
        elif sentinel2_bands == 'all':
            try:
                if 's2_median_t1' in results and results['s2_median_t1'] is not None:
                    bands_to_export = results['s2_median_t1'].bandNames().getInfo()
                    bands_label = 'All'
                else:
                    bands_to_export = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
                    bands_label = 'Default (T1 median unavailable)'
                    print("  ⚠️ S2 T1 median not available. Using default bands.")
            except Exception as e:
                print(f"  ⚠️ Error getting S2 T1 band names: {e}. Using default bands.")
                bands_to_export = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
                bands_label = 'Default (error getting band names)'
        else:
            bands_to_export = sentinel2_bands
            bands_label = 'Custom'

        print(f"  Bands ({bands_label}): {bands_to_export}")

        # T1
        if 's2_median_t1' in results and results['s2_median_t1'] is not None:
            try:
                print(f"\n📊 Exporting Sentinel-2 T1...")
                task_s2_t1 = ee.batch.Export.image.toDrive(
                    image=results['s2_median_t1'].select(bands_to_export),
                    description=f'Sentinel2_{config["site_name"]}_T1',
                    folder=config['export_folder'],
                    fileNamePrefix=f'sentinel2_{config["site_name"]}_t1',
                    region=roi,
                    crs=config['export_crs'],
                    scale=10,
                    maxPixels=1e12,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_s2_t1.start()
                tasks.append(task_s2_t1)
                print(f"  ✓ Sentinel-2 T1 started")
            except Exception as e:
                print(f"  ⚠️ Failed: {e}")
        else:
            print("  ℹ️ Sentinel-2 T1 (median) not available for export.")

        # T2
        if 's2_median_t2' in results and results['s2_median_t2'] is not None:
            try:
                print(f"📊 Exporting Sentinel-2 T2...")
                task_s2_t2 = ee.batch.Export.image.toDrive(
                    image=results['s2_median_t2'].select(bands_to_export),
                    description=f'Sentinel2_{config["site_name"]}_T2',
                    folder=config['export_folder'],
                    fileNamePrefix=f'sentinel2_{config["site_name"]}_t2',
                    region=roi,
                    crs=config['export_crs'],
                    scale=10,
                    maxPixels=1e12,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_s2_t2.start()
                tasks.append(task_s2_t2)
                print(f"  ✓ Sentinel-2 T2 started")
            except Exception as e:
                print(f"  ⚠️ Failed: {e}")
        else:
            print("  ℹ️ Sentinel-2 T2 (median) not available for export.")

    # ==================== EXPORT SENTINEL-1 ====================
    if export_sentinel1:
        print("\n" + "=" * 80)
        print("EXPORTING SENTINEL-1 SAR")
        print("=" * 80)

        try:
            s1_t1_img = s1_t1 if s1_t1 is not None else results.get('s1_t1')
            s1_t2_img = s1_t2 if s1_t2 is not None else results.get('s1_t2')

            # Auto-generate if needed
            if s1_t1_img is None and config.get('sentinel2_t1_id'):
                print("\n📡 Generating Sentinel-1 T1 composite (fallback)...")
                s1_t1_img = generate_sentinel1_composite_from_s2_date(
                    roi,
                    config['sentinel2_t1_id'],
                    polarizations=s1_pols
                )

            if s1_t2_img is None and config.get('sentinel2_t2_id'):
                print("\n📡 Generating Sentinel-1 T2 composite (fallback)...")
                s1_t2_img = generate_sentinel1_composite_from_s2_date(
                    roi,
                    config['sentinel2_t2_id'],
                    polarizations=s1_pols
                )

            # Export T1
            if s1_t1_img is not None:
                print("\n📡 Exporting Sentinel-1 T1...")
                task_s1_t1 = ee.batch.Export.image.toDrive(
                    image=s1_t1_img.select(s1_pols),
                    description=f'Sentinel1_{config["site_name"]}_T1',
                    folder=config['export_folder'],
                    fileNamePrefix=f'sentinel1_{config["site_name"]}_t1',
                    region=roi,
                    crs=crs,
                    scale=scale,
                    maxPixels=1e12,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_s1_t1.start()
                tasks.append(task_s1_t1)
                print("  ✓ Sentinel-1 T1 started")
            else:
                print("  ℹ️ Sentinel-1 T1 not available for export.")

            # Export T2
            if s1_t2_img is not None:
                print("\n📡 Exporting Sentinel-1 T2...")
                task_s1_t2 = ee.batch.Export.image.toDrive(
                    image=s1_t2_img.select(s1_pols),
                    description=f'Sentinel1_{config["site_name"]}_T2',
                    folder=config['export_folder'],
                    fileNamePrefix=f'sentinel1_{config["site_name"]}_t2',
                    region=roi,
                    crs=crs,
                    scale=scale,
                    maxPixels=1e12,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_s1_t2.start()
                tasks.append(task_s1_t2)
                print("  ✓ Sentinel-1 T2 started")
            else:
                print("  ℹ️ Sentinel-1 T2 not available for export.")

        except Exception as e:
            print(f"  ⚠️ Failed to process/export Sentinel-1: {e}")

    # ==================== EXPORT DELTA LST 30m (T2 - T1) ====================
    if export_delta_lst_30m and 'lst_30m_t1' in results and 'lst_30m_t2' in results:
        print("\n" + "=" * 80)
        print("EXPORTING DELTA LST 30m (T2 - T1)")
        print("=" * 80)
        print("  Note: Positive values = warming; units in degrees Celsius")

        try:
            delta_lst = results['lst_30m_t2'].subtract(results['lst_30m_t1']).rename('Delta_LST_30m')
            task_name = f"Delta_LST_30m_{config['site_name']}"
            task = ee.batch.Export.image.toDrive(
                image=delta_lst.clip(roi),
                description=task_name,
                folder=config['export_folder'],
                fileNamePrefix=task_name,
                region=roi,
                crs=config['export_crs'],
                scale=30,
                maxPixels=1e13,
                fileFormat='GeoTIFF',
                formatOptions={'cloudOptimized': True}
            )
            task.start()
            tasks.append(task)
            print("  ✓ Delta LST 30m started")
        except Exception as e:
            print(f"  ⚠️ Failed to export Delta LST 30m: {e}")
    elif export_delta_lst_30m:
        print("\n  ℹ️ LST 30m not available for delta computation")

    # ==================== EXPORT DELTA LANDSAT INDICES (T2 - T1) ====================
    if export_delta_landsat_indices and 'indices_landsat' in results:
        print("\n" + "=" * 80)
        print("EXPORTING DELTA LANDSAT SPECTRAL INDICES (T2 - T1) (30m)")
        print("=" * 80)

        year_t1 = config.get('t1_start_date', 't1').split('-')[0] if 't1_start_date' in config else 't1'
        year_t2 = config.get('t2_start_date', 't2').split('-')[0] if 't2_start_date' in config else 't2'

        indices_landsat_t1 = results['indices_landsat'].get('t1', {})
        indices_landsat_t2 = results['indices_landsat'].get('t2', {})

        if landsat_indices_list == 'all':
            indices_for_delta = list(set(indices_landsat_t1.keys()) & set(indices_landsat_t2.keys()))
            print("  Exporting delta for all available Landsat indices")
        else:
            indices_for_delta = [x.lower() for x in landsat_indices_list]
            print(f"  Exporting delta for: {[x.upper() for x in indices_for_delta]}")

        for idx_name in indices_for_delta:
            if idx_name in indices_landsat_t1 and idx_name in indices_landsat_t2:
                try:
                    print(f"\n📊 Exporting Delta Landsat {idx_name.upper()} (T2-T1)...")
                    delta_img = indices_landsat_t2[idx_name].subtract(indices_landsat_t1[idx_name]).rename(f'Delta_{idx_name.upper()}')
                    task_name = f"Delta_Landsat_{idx_name.upper()}_{config['site_name']}_{year_t1}_{year_t2}"
                    task = ee.batch.Export.image.toDrive(
                        image=delta_img.clip(roi),
                        description=task_name,
                        folder=config['export_folder'],
                        fileNamePrefix=task_name,
                        region=roi,
                        scale=30,
                        crs=config['export_crs'],
                        maxPixels=1e13,
                        fileFormat='GeoTIFF',
                        formatOptions={'cloudOptimized': True}
                    )
                    task.start()
                    tasks.append(task)
                    print(f"  ✓ Delta Landsat {idx_name.upper()} started")
                except Exception as e:
                    print(f"  ⚠️ Failed to export Delta {idx_name.upper()}: {e}")
            else:
                print(f"  ℹ️ {idx_name.upper()} not available for both periods")
    elif export_delta_landsat_indices:
        print("\n  ℹ️ Landsat indices not available for delta computation")

    # ==================== EXPORT DELTA SENTINEL-2 INDICES (T2 - T1) ====================
    if export_delta_sentinel2_indices and 'indices_sentinel2' in results:
        print("\n" + "=" * 80)
        print("EXPORTING DELTA SENTINEL-2 SPECTRAL INDICES (T2 - T1) (10m)")
        print("=" * 80)

        year_t1 = config.get('t1_start_date', 't1').split('-')[0] if 't1_start_date' in config else 't1'
        year_t2 = config.get('t2_start_date', 't2').split('-')[0] if 't2_start_date' in config else 't2'

        indices_s2_t1 = results['indices_sentinel2'].get('t1', {})
        indices_s2_t2 = results['indices_sentinel2'].get('t2', {})

        if sentinel2_indices_list == 'all':
            indices_for_delta = list(set(indices_s2_t1.keys()) & set(indices_s2_t2.keys()))
            print("  Exporting delta for all available Sentinel-2 indices")
        else:
            indices_for_delta = [x.lower() for x in sentinel2_indices_list]
            print(f"  Exporting delta for: {[x.upper() for x in indices_for_delta]}")

        for idx_name in indices_for_delta:
            if idx_name in indices_s2_t1 and idx_name in indices_s2_t2:
                try:
                    print(f"\n📊 Exporting Delta Sentinel-2 {idx_name.upper()} (T2-T1)...")
                    delta_img = indices_s2_t2[idx_name].subtract(indices_s2_t1[idx_name]).rename(f'Delta_{idx_name.upper()}')
                    task_name = f"Delta_S2_{idx_name.upper()}_{config['site_name']}_{year_t1}_{year_t2}"
                    task = ee.batch.Export.image.toDrive(
                        image=delta_img.clip(roi),
                        description=task_name,
                        folder=config['export_folder'],
                        fileNamePrefix=task_name,
                        region=roi,
                        scale=10,
                        crs=config['export_crs'],
                        maxPixels=1e13,
                        fileFormat='GeoTIFF',
                        formatOptions={'cloudOptimized': True}
                    )
                    task.start()
                    tasks.append(task)
                    print(f"  ✓ Delta Sentinel-2 {idx_name.upper()} started")
                except Exception as e:
                    print(f"  ⚠️ Failed to export Delta {idx_name.upper()}: {e}")
            else:
                print(f"  ℹ️ {idx_name.upper()} not available for both periods")
    elif export_delta_sentinel2_indices:
        print("\n  ℹ️ Sentinel-2 indices not available for delta computation")

    # ==================== EXPORT DELTA UHI 30m (T2 - T1) ====================
    if export_delta_uhi_30m and stats_change and 'UHI_30m' in stats_change:
        print("\n" + "=" * 80)
        print("EXPORTING DELTA UHI 30m (T2 - T1)")
        print("=" * 80)
        print("  Note: Delta of z-score UHI values (positive = UHI increased)")

        try:
            uhi_images = stats_change['UHI_30m']['images']
            if 't1' in uhi_images and 't2' in uhi_images:
                delta_uhi = uhi_images['t2'].subtract(uhi_images['t1']).rename('Delta_UHI_30m')
                task_name = f"Delta_UHI_30m_{config['site_name']}"
                task = ee.batch.Export.image.toDrive(
                    image=delta_uhi.clip(roi),
                    description=task_name,
                    folder=config['export_folder'],
                    fileNamePrefix=task_name,
                    region=roi,
                    crs=config['export_crs'],
                    scale=30,
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task.start()
                tasks.append(task)
                print("  ✓ Delta UHI 30m started")
            else:
                print("  ℹ️ UHI 30m images not available for both periods")
        except Exception as e:
            print(f"  ⚠️ Failed to export Delta UHI 30m: {e}")
    elif export_delta_uhi_30m:
        print("\n  ℹ️ UHI 30m not available for delta computation")

    # ==================== EXPORT DELTA UHI INTENSITY 30m (T2 - T1) ====================
    if export_delta_uhi_intensity_30m and stats_change and 'UHI_30m' in stats_change:
        print("\n" + "=" * 80)
        print("EXPORTING DELTA UHI INTENSITY CLASSIFICATION 30m (T2 - T1)")
        print("=" * 80)
        print("  Note: Delta of class values (1-LTZ to 5-HTZ); positive = warming shift")

        try:
            intensity_class = stats_change['UHI_30m']['intensity_classification']
            if 't1' in intensity_class and 't2' in intensity_class:
                delta_intensity = intensity_class['t2'].subtract(intensity_class['t1']).rename('Delta_UHI_Intensity_30m')
                task_name = f"Delta_UHI_Intensity_30m_{config['site_name']}"
                task = ee.batch.Export.image.toDrive(
                    image=delta_intensity.clip(roi),
                    description=task_name,
                    folder=config['export_folder'],
                    fileNamePrefix=task_name,
                    region=roi,
                    crs=config['export_crs'],
                    scale=30,
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task.start()
                tasks.append(task)
                print("  ✓ Delta UHI Intensity 30m started")
            else:
                print("  ℹ️ UHI Intensity 30m classification not available for both periods")
        except Exception as e:
            print(f"  ⚠️ Failed to export Delta UHI Intensity 30m: {e}")
    elif export_delta_uhi_intensity_30m:
        print("\n  ℹ️ UHI Intensity 30m not available for delta computation")

    # ==================== EXPORT CDI LANDSAT (dLST_30m x dIndex, 30m) ====================
    if export_cdi_landsat and 'indices_landsat' in results and 'lst_30m_t1' in results and 'lst_30m_t2' in results:
        print("\n" + "=" * 80)
        print("EXPORTING CDI LANDSAT (delta_LST_30m x delta_Index) (30m)")
        print("=" * 80)
        print("  Formula: CDI = (LST_T2 - LST_T1) x (Index_T2 - Index_T1)")
        print("  Reference: Silva & Torres (2021)")

        year_t1 = config.get('t1_start_date', 't1').split('-')[0] if 't1_start_date' in config else 't1'
        year_t2 = config.get('t2_start_date', 't2').split('-')[0] if 't2_start_date' in config else 't2'

        indices_landsat_t1 = results['indices_landsat'].get('t1', {})
        indices_landsat_t2 = results['indices_landsat'].get('t2', {})

        if landsat_indices_list == 'all':
            indices_for_cdi = list(set(indices_landsat_t1.keys()) & set(indices_landsat_t2.keys()))
        else:
            indices_for_cdi = [x.lower() for x in landsat_indices_list]

        try:
            delta_lst_30m = results['lst_30m_t2'].subtract(results['lst_30m_t1'])

            for idx_name in indices_for_cdi:
                if idx_name in indices_landsat_t1 and idx_name in indices_landsat_t2:
                    try:
                        print(f"\n📊 Exporting CDI Landsat LST_30m x {idx_name.upper()}...")
                        delta_index = indices_landsat_t2[idx_name].subtract(indices_landsat_t1[idx_name])
                        cdi_img = delta_lst_30m.multiply(delta_index).rename(f'CDI_{idx_name.upper()}')
                        task_name = f"CDI_Landsat_{idx_name.upper()}_{config['site_name']}_{year_t1}_{year_t2}"
                        task = ee.batch.Export.image.toDrive(
                            image=cdi_img.clip(roi),
                            description=task_name,
                            folder=config['export_folder'],
                            fileNamePrefix=task_name,
                            region=roi,
                            scale=30,
                            crs=config['export_crs'],
                            maxPixels=1e13,
                            fileFormat='GeoTIFF',
                            formatOptions={'cloudOptimized': True}
                        )
                        task.start()
                        tasks.append(task)
                        print(f"  ✓ CDI Landsat {idx_name.upper()} started")
                    except Exception as e:
                        print(f"  ⚠️ Failed to export CDI {idx_name.upper()}: {e}")
                else:
                    print(f"  ℹ️ {idx_name.upper()} not available for both periods")
        except Exception as e:
            print(f"  ⚠️ Failed to compute LST 30m delta for CDI: {e}")
    elif export_cdi_landsat:
        print("\n  ℹ️ Landsat indices or LST 30m not available for CDI computation")

    # ==================== EXPORT CDI SENTINEL-2 (dLST_30m x dS2Index, 10m) ====================
    if export_cdi_sentinel2 and 'indices_sentinel2' in results and 'lst_30m_t1' in results and 'lst_30m_t2' in results:
        print("\n" + "=" * 80)
        print("EXPORTING CDI SENTINEL-2 (delta_LST_30m x delta_S2Index) (10m)")
        print("=" * 80)
        print("  Formula: CDI = (LST_T2 - LST_T1) x (S2Index_T2 - S2Index_T1)")
        print("  Note: LST_30m resampled to 10m via bilinear interpolation")
        print("  Reference: Silva & Torres (2021)")

        year_t1 = config.get('t1_start_date', 't1').split('-')[0] if 't1_start_date' in config else 't1'
        year_t2 = config.get('t2_start_date', 't2').split('-')[0] if 't2_start_date' in config else 't2'

        indices_s2_t1 = results['indices_sentinel2'].get('t1', {})
        indices_s2_t2 = results['indices_sentinel2'].get('t2', {})

        if sentinel2_indices_list == 'all':
            indices_for_cdi = list(set(indices_s2_t1.keys()) & set(indices_s2_t2.keys()))
        else:
            indices_for_cdi = [x.lower() for x in sentinel2_indices_list]

        try:
            delta_lst_30m = results['lst_30m_t2'].subtract(results['lst_30m_t1'])
            delta_lst_10m = delta_lst_30m.resample('bilinear')

            for idx_name in indices_for_cdi:
                if idx_name in indices_s2_t1 and idx_name in indices_s2_t2:
                    try:
                        print(f"\n📊 Exporting CDI S2 LST_30m x {idx_name.upper()}...")
                        delta_index = indices_s2_t2[idx_name].subtract(indices_s2_t1[idx_name])
                        cdi_img = delta_lst_10m.multiply(delta_index).rename(f'CDI_{idx_name.upper()}')
                        task_name = f"CDI_S2_{idx_name.upper()}_{config['site_name']}_{year_t1}_{year_t2}"
                        task = ee.batch.Export.image.toDrive(
                            image=cdi_img.clip(roi),
                            description=task_name,
                            folder=config['export_folder'],
                            fileNamePrefix=task_name,
                            region=roi,
                            scale=10,
                            crs=config['export_crs'],
                            maxPixels=1e13,
                            fileFormat='GeoTIFF',
                            formatOptions={'cloudOptimized': True}
                        )
                        task.start()
                        tasks.append(task)
                        print(f"  ✓ CDI S2 {idx_name.upper()} started")
                    except Exception as e:
                        print(f"  ⚠️ Failed to export CDI S2 {idx_name.upper()}: {e}")
                else:
                    print(f"  ℹ️ {idx_name.upper()} not available for both periods")
        except Exception as e:
            print(f"  ⚠️ Failed to compute LST 30m delta for CDI S2: {e}")
    elif export_cdi_sentinel2:
        print("\n  ℹ️ Sentinel-2 indices or LST 30m not available for CDI computation")

    # ==================== FINAL SUMMARY ====================
    print("\n" + "=" * 80)
    print(f"✅ {len(tasks)} EXPORTS STARTED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📂 Location: Google Drive/{config['export_folder']}/")
    print(f"🔗 Monitor at: https://code.earthengine.google.com/tasks")

    # List exported products
    print("\n📦 Exported products:")
    product_count = {}
    for task in tasks:
        desc = task.config['description']
        product_type = desc.split('_')[0]
        product_count[product_type] = product_count.get(product_type, 0) + 1

    for product, count in sorted(product_count.items()):
        print(f"  • {product}: {count} file(s)")

    return tasks