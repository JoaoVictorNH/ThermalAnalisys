# ==============================================================================
# export_utils.py — Export Functions for Google Drive
# ==============================================================================
# Export of results (LST, UHI, indices) to Google Drive,
# for both single-period and bitemporal analyses.
# ==============================================================================

import ee

# --- Cross-module imports ---
from modulos.gee_setup import get_utm_crs
from modulos.image_processing import generate_sentinel1_composite_from_s2_date




def export_results(results):
    """
    Exports results to Google Drive.
    """
    print("\n📤 Preparing export...")

    # Build export identifier
    start_date = results['config']['start_date'].replace('-', '')
    export_id = f"{start_date}_{results['config']['landsat_type']}"

    # Export Landsat LST 30m
    task_l8 = ee.batch.Export.image.toDrive(
        image=results['lst_30m'],
        description=f'L8_LST_{export_id}',
        folder=results['config']['export_folder'],
        scale=30,
        region=results['config']['geometry'],
        fileFormat='GeoTIFF',
        maxPixels=1e10,
        formatOptions={'cloudOptimized': True}
    )

    print(f"✓ Task created: L8_LST_{export_id}")

    # Export results for each method
    for method_name, method_data in results['methods'].items():
        task = ee.batch.Export.image.toDrive(
            image=method_data['lst'],
            description=f'S2_{method_name}_LST_{export_id}',
            folder=results['config']['export_folder'],
            scale=10,
            region=results['config']['geometry'],
            fileFormat='GeoTIFF',
            maxPixels=1e10
        )




def export_s1_bitemporal_data(roi, s1_t1, s1_t2, config, polarizations=["VV", "VH"], export_scale=10):
    """
    Exports bitemporal Sentinel-1 data to Google Drive.
    """
    print("\n" + "=" * 80)
    print("EXPORTING SENTINEL-1 DATA TO GOOGLE DRIVE")
    print("=" * 80)

    try:
        coords = roi.centroid(0.1).coordinates().getInfo()
        crs = get_utm_crs(coords[0], coords[1])
    except NameError:
        print("Warning: get_utm_crs function not found. Using CRS from config.")
        crs = config.get('export_crs', 'EPSG:4326')

    print(f"Determined CRS: {crs}")
    print(f"Sentinel-1 polarizations: {polarizations}")

    tasks = []

    task_s1_t1 = ee.batch.Export.image.toDrive(
        image=s1_t1.select(polarizations),
        description=f'Sentinel1_{config["site_name"]}_T1',
        folder=config['export_folder'],
        fileNamePrefix=f'sentinel1_{config["site_name"]}_t1',
        region=roi,
        crs=crs,
        scale=export_scale,
        maxPixels=1e12,
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True}
    )
    task_s1_t1.start()
    tasks.append(task_s1_t1)
    print(f"✓ Sentinel-1 T1 export started")

    task_s1_t2 = ee.batch.Export.image.toDrive(
        image=s1_t2.select(polarizations),
        description=f'Sentinel1_{config["site_name"]}_T2',
        folder=config['export_folder'],
        fileNamePrefix=f'sentinel1_{config["site_name"]}_t2',
        region=roi,
        crs=crs,
        scale=export_scale,
        maxPixels=1e12,
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True}
    )
    task_s1_t2.start()
    tasks.append(task_s1_t2)
    print(f"✓ Sentinel-1 T2 export started")

    print("\n📁 Sentinel-1 exports started to Google Drive!")
    print(f"   Folder: {config['export_folder']}")

    return tasks



def export_bitemporal_results(
    results,
    config,
    roi,
    stats_change=None,
    methods=None,
    s1_t1=None,
    s1_t2=None,
    export_lst_30m=True,
    export_lst_10m=True,
    export_uhi_30m=True,
    export_uhi_10m=True,
    export_uhi_intensity_30m=True,
    export_uhi_intensity_10m=True,
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
    export_delta_uhi_10m=False,
    export_delta_uhi_intensity_10m=False,
    export_delta_lst_30m=False,
    export_delta_lst_10m=False,
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
        methods (list): List of LST 10m methods to export (None = all).
        s1_t1 (ee.Image, optional): Sentinel-1 T1 image.
        s1_t2 (ee.Image, optional): Sentinel-1 T2 image.
        export_lst_30m (bool): Export Landsat 30m LST.
        export_lst_10m (bool): Export downscaled 10m LST.
        export_uhi_30m (bool): Export normalized 30m UHI.
        export_uhi_10m (bool): Export normalized 10m UHI.
        export_uhi_intensity_30m (bool): Export 30m UHI intensity classification.
        export_uhi_intensity_10m (bool): Export 10m UHI intensity classification.
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
        export_delta_uhi_10m (bool): Export T2-T1 delta for UHI 10m per downscaling method.
        export_delta_uhi_intensity_10m (bool): Export T2-T1 delta for UHI Intensity 10m per method.
        export_delta_lst_30m (bool): Export T2-T1 delta for LST 30m (degrees Celsius).
        export_delta_lst_10m (bool): Export T2-T1 delta for LST 10m per downscaling method.
        export_cdi_landsat (bool): Export CDI = delta_LST_30m x delta_LandsatIndex per index (30m).
        export_cdi_sentinel2 (bool): Export CDI = delta_LST_30m x delta_S2Index per index (10m).

    Returns:
        list: List of started export tasks.
    """
    print("\n" + "=" * 80)
    print("EXPORTING SELECTED PRODUCTS TO GOOGLE DRIVE")
    print("=" * 80)

    tasks = []

    # Determine methods if not specified
    if methods is None and (export_lst_10m or export_uhi_10m):
        try:
            methods = list(results['t1']['methods'].keys())
        except Exception:
            methods = []
            export_lst_10m = False
            export_uhi_10m = False
            print("  ⚠️ Could not determine LST 10m methods")

    # Determine UTM CRS from geometry
    if not isinstance(roi, ee.Geometry):
        print("  ⚠️ ROI is not a valid geometry. Cannot determine UTM CRS.")
        crs = config.get('export_crs', 'EPSG:4326')
        print(f"  Using default CRS: {crs}")
    else:
        try:
            coords = roi.centroid(0.1).coordinates().getInfo()
            crs = get_utm_crs(coords[0], coords[1])
            print(f"Determined CRS: {crs}")
        except Exception as e:
            print(f"  ⚠️ Error determining UTM CRS: {e}")
            crs = config.get('export_crs', 'EPSG:4326')
            print(f"  Using default CRS: {crs}")

    # Export scale
    scale = config.get('export_scale', config.get('export_resolution', 10))

    # Sentinel-1 polarizations
    s1_pols = config.get('s1_polarizations', ['VV', 'VH'])

    # Export summary
    print("\n📋 EXPORT SUMMARY:")
    print(f"  • LST 30m (Landsat): {'YES' if export_lst_30m else 'NO'}")
    print(f"  • LST 10m (Downscaled): {'YES' if export_lst_10m else 'NO'}")
    if export_lst_10m and methods:
        print(f"    Methods: {', '.join(methods)}")
    print(f"  • UHI 30m (Normalized): {'YES' if export_uhi_30m else 'NO'}")
    print(f"  • UHI 10m (Normalized): {'YES' if export_uhi_10m else 'NO'}")
    if export_uhi_10m and methods:
        print(f"    Methods: {', '.join(methods)}")
    print(f"  • UHI Intensity Classification 30m (Deng et al. 2023): {'YES' if export_uhi_intensity_30m else 'NO'}")
    print(f"  • UHI Intensity Classification 10m (Deng et al. 2023): {'YES' if export_uhi_intensity_10m else 'NO'}")
    print(f"  • Landsat Indices (30m): {'YES' if export_landsat_indices else 'NO'}")
    print(f"  • Landsat SR Imagery (30m): {'YES' if export_landsat else 'NO'}")
    print(f"  • Delta LST 30m T2-T1: {'YES' if export_delta_lst_30m else 'NO'}")
    print(f"  • Delta LST 10m T2-T1: {'YES' if export_delta_lst_10m else 'NO'}")
    print(f"  • Delta Landsat Indices T2-T1 (30m): {'YES' if export_delta_landsat_indices else 'NO'}")
    print(f"  • Delta Sentinel-2 Indices T2-T1 (10m): {'YES' if export_delta_sentinel2_indices else 'NO'}")
    print(f"  • Delta UHI 30m T2-T1: {'YES' if export_delta_uhi_30m else 'NO'}")
    print(f"  • Delta UHI Intensity 30m T2-T1: {'YES' if export_delta_uhi_intensity_30m else 'NO'}")
    print(f"  • Delta UHI 10m T2-T1: {'YES' if export_delta_uhi_10m else 'NO'}")
    print(f"  • Delta UHI Intensity 10m T2-T1: {'YES' if export_delta_uhi_intensity_10m else 'NO'}")
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

    # ==================== EXPORT LST 10m (DOWNSCALED) ====================
    if export_lst_10m and methods:
        print("\n" + "=" * 80)
        print("EXPORTING LST 10m (DOWNSCALED)")
        print("=" * 80)

        for method in methods:
            # T1
            try:
                print(f"\n📊 Exporting {method} 10m T1...")
                lst_10m_t1 = results['t1']['methods'][method]['lst']
                task_t1 = ee.batch.Export.image.toDrive(
                    image=lst_10m_t1,
                    description=f'LST_10m_{method}_{config["site_name"]}_T1',
                    folder=config['export_folder'],
                    fileNamePrefix=f'LST_10m_{method}_{config["site_name"]}_t1',
                    region=roi,
                    crs=config['export_crs'],
                    scale=10,
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_t1.start()
                tasks.append(task_t1)
                print(f"  ✓ {method} 10m T1 started")
            except Exception as e:
                print(f"  ⚠️ Failed: {e}")

            # T2
            try:
                print(f"📊 Exporting {method} 10m T2...")
                lst_10m_t2 = results['t2']['methods'][method]['lst']
                task_t2 = ee.batch.Export.image.toDrive(
                    image=lst_10m_t2,
                    description=f'LST_10m_{method}_{config["site_name"]}_T2',
                    folder=config['export_folder'],
                    fileNamePrefix=f'LST_10m_{method}_{config["site_name"]}_t2',
                    region=roi,
                    crs=config['export_crs'],
                    scale=10,
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task_t2.start()
                tasks.append(task_t2)
                print(f"  ✓ {method} 10m T2 started")
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

    # ==================== EXPORT UHI 10m (NORMALIZED) ====================
    if export_uhi_10m and stats_change and methods:
        print("\n" + "=" * 80)
        print("EXPORTING UHI 10m (NORMALIZED)")
        print("=" * 80)
        print("  Note: UHI values are z-scores (standard deviations from the mean)")

        for method in methods:
            uhi_key = f'UHI_{method}_10m'

            if uhi_key not in stats_change:
                print(f"\n  ℹ️ UHI 10m {method} not available in stats_change")
                continue

            try:
                uhi_images = stats_change[uhi_key]['images']

                # T1
                if 't1' in uhi_images:
                    print(f"\n🔥 Exporting UHI 10m {method} T1...")
                    task_uhi10_t1 = ee.batch.Export.image.toDrive(
                        image=uhi_images['t1'],
                        description=f'UHI_10m_{method}_{config["site_name"]}_T1',
                        folder=config['export_folder'],
                        fileNamePrefix=f'UHI_10m_{method}_{config["site_name"]}_t1',
                        region=roi,
                        crs=config['export_crs'],
                        scale=10,
                        maxPixels=1e13,
                        fileFormat='GeoTIFF',
                        formatOptions={'cloudOptimized': True}
                    )
                    task_uhi10_t1.start()
                    tasks.append(task_uhi10_t1)
                    print(f"  ✓ UHI 10m {method} T1 started")

                # T2
                if 't2' in uhi_images:
                    print(f"🔥 Exporting UHI 10m {method} T2...")
                    task_uhi10_t2 = ee.batch.Export.image.toDrive(
                        image=uhi_images['t2'],
                        description=f'UHI_10m_{method}_{config["site_name"]}_T2',
                        folder=config['export_folder'],
                        fileNamePrefix=f'UHI_10m_{method}_{config["site_name"]}_t2',
                        region=roi,
                        crs=config['export_crs'],
                        scale=10,
                        maxPixels=1e13,
                        fileFormat='GeoTIFF',
                        formatOptions={'cloudOptimized': True}
                    )
                    task_uhi10_t2.start()
                    tasks.append(task_uhi10_t2)
                    print(f"  ✓ UHI 10m {method} T2 started")

            except Exception as e:
                print(f"  ⚠️ Failed to export UHI 10m {method}: {e}")
    elif export_uhi_10m:
        print("\n  ℹ️ UHI 10m not available (stats_change not provided or no UHI data)")

    # ==================== EXPORT UHI INTENSITY CLASSIFICATION 10m ====================
    if export_uhi_intensity_10m and stats_change and methods:
        print("\n" + "=" * 80)
        print("EXPORTING UHI INTENSITY CLASSIFICATION 10m (Deng et al. 2023)")
        print("=" * 80)
        print("  Note: Classes 1-LTZ, 2-SLTZ, 3-MTZ, 4-SHTZ, 5-HTZ")

        for method in methods:
            uhi_key = f'UHI_{method}_10m'

            if uhi_key not in stats_change:
                print(f"\n  ℹ️ UHI Intensity Classification 10m {method} not available in stats_change")
                continue

            try:
                intensity_class = stats_change[uhi_key]['intensity_classification']

                # T1
                if 't1' in intensity_class:
                    print(f"\n🔥 Exporting UHI Intensity Classification 10m {method} T1...")
                    task_intensity10_t1 = ee.batch.Export.image.toDrive(
                        image=intensity_class['t1'],
                        description=f'UHI_Intensity_10m_{method}_{config["site_name"]}_T1',
                        folder=config['export_folder'],
                        fileNamePrefix=f'UHI_Intensity_10m_{method}_{config["site_name"]}_t1',
                        region=roi,
                        crs=config['export_crs'],
                        scale=10,
                        maxPixels=1e13,
                        fileFormat='GeoTIFF',
                        formatOptions={'cloudOptimized': True}
                    )
                    task_intensity10_t1.start()
                    tasks.append(task_intensity10_t1)
                    print(f"  ✓ UHI Intensity Classification 10m {method} T1 started")

                # T2
                if 't2' in intensity_class:
                    print(f"🔥 Exporting UHI Intensity Classification 10m {method} T2...")
                    task_intensity10_t2 = ee.batch.Export.image.toDrive(
                        image=intensity_class['t2'],
                        description=f'UHI_Intensity_10m_{method}_{config["site_name"]}_T2',
                        folder=config['export_folder'],
                        fileNamePrefix=f'UHI_Intensity_10m_{method}_{config["site_name"]}_t2',
                        region=roi,
                        crs=config['export_crs'],
                        scale=10,
                        maxPixels=1e13,
                        fileFormat='GeoTIFF',
                        formatOptions={'cloudOptimized': True}
                    )
                    task_intensity10_t2.start()
                    tasks.append(task_intensity10_t2)
                    print(f"  ✓ UHI Intensity Classification 10m {method} T2 started")

            except Exception as e:
                print(f"  ⚠️ Failed to export UHI Intensity Classification 10m {method}: {e}")
    elif export_uhi_intensity_10m:
        print("\n  ℹ️ UHI Intensity Classification 10m not available (stats_change not provided or no data)")

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

    # ==================== EXPORT DELTA LST 10m (T2 - T1) ====================
    if export_delta_lst_10m and methods:
        print("\n" + "=" * 80)
        print("EXPORTING DELTA LST 10m (T2 - T1)")
        print("=" * 80)
        print("  Note: Positive values = warming; units in degrees Celsius")

        for method in methods:
            try:
                lst_10m_t1 = results['t1']['methods'][method]['lst']
                lst_10m_t2 = results['t2']['methods'][method]['lst']
                delta_lst = lst_10m_t2.subtract(lst_10m_t1).rename(f'Delta_LST_{method}_10m')
                task_name = f"Delta_LST_10m_{method}_{config['site_name']}"
                task = ee.batch.Export.image.toDrive(
                    image=delta_lst.clip(roi),
                    description=task_name,
                    folder=config['export_folder'],
                    fileNamePrefix=task_name,
                    region=roi,
                    crs=config['export_crs'],
                    scale=10,
                    maxPixels=1e13,
                    fileFormat='GeoTIFF',
                    formatOptions={'cloudOptimized': True}
                )
                task.start()
                tasks.append(task)
                print(f"  ✓ Delta LST 10m {method} started")
            except Exception as e:
                print(f"  ⚠️ Failed to export Delta LST 10m {method}: {e}")
    elif export_delta_lst_10m:
        print("\n  ℹ️ LST 10m methods not available for delta computation")

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

    # ==================== EXPORT DELTA UHI 10m (T2 - T1) ====================
    if export_delta_uhi_10m and stats_change and methods:
        print("\n" + "=" * 80)
        print("EXPORTING DELTA UHI 10m (T2 - T1)")
        print("=" * 80)
        print("  Note: Delta of z-score UHI values per downscaling method")

        for method in methods:
            uhi_key = f'UHI_{method}_10m'
            if uhi_key not in stats_change:
                print(f"\n  ℹ️ UHI 10m {method} not available in stats_change")
                continue
            try:
                uhi_images = stats_change[uhi_key]['images']
                if 't1' in uhi_images and 't2' in uhi_images:
                    delta_uhi = uhi_images['t2'].subtract(uhi_images['t1']).rename(f'Delta_UHI_{method}_10m')
                    task_name = f"Delta_UHI_10m_{method}_{config['site_name']}"
                    task = ee.batch.Export.image.toDrive(
                        image=delta_uhi.clip(roi),
                        description=task_name,
                        folder=config['export_folder'],
                        fileNamePrefix=task_name,
                        region=roi,
                        crs=config['export_crs'],
                        scale=10,
                        maxPixels=1e13,
                        fileFormat='GeoTIFF',
                        formatOptions={'cloudOptimized': True}
                    )
                    task.start()
                    tasks.append(task)
                    print(f"  ✓ Delta UHI 10m {method} started")
                else:
                    print(f"  ℹ️ UHI 10m {method} images not available for both periods")
            except Exception as e:
                print(f"  ⚠️ Failed to export Delta UHI 10m {method}: {e}")
    elif export_delta_uhi_10m:
        print("\n  ℹ️ UHI 10m not available for delta computation")

    # ==================== EXPORT DELTA UHI INTENSITY 10m (T2 - T1) ====================
    if export_delta_uhi_intensity_10m and stats_change and methods:
        print("\n" + "=" * 80)
        print("EXPORTING DELTA UHI INTENSITY CLASSIFICATION 10m (T2 - T1)")
        print("=" * 80)
        print("  Note: Delta of class values (1-LTZ to 5-HTZ) per method; positive = warming shift")

        for method in methods:
            uhi_key = f'UHI_{method}_10m'
            if uhi_key not in stats_change:
                print(f"\n  ℹ️ UHI Intensity 10m {method} not available in stats_change")
                continue
            try:
                intensity_class = stats_change[uhi_key]['intensity_classification']
                if 't1' in intensity_class and 't2' in intensity_class:
                    delta_intensity = intensity_class['t2'].subtract(intensity_class['t1']).rename(f'Delta_UHI_Intensity_{method}_10m')
                    task_name = f"Delta_UHI_Intensity_10m_{method}_{config['site_name']}"
                    task = ee.batch.Export.image.toDrive(
                        image=delta_intensity.clip(roi),
                        description=task_name,
                        folder=config['export_folder'],
                        fileNamePrefix=task_name,
                        region=roi,
                        crs=config['export_crs'],
                        scale=10,
                        maxPixels=1e13,
                        fileFormat='GeoTIFF',
                        formatOptions={'cloudOptimized': True}
                    )
                    task.start()
                    tasks.append(task)
                    print(f"  ✓ Delta UHI Intensity 10m {method} started")
                else:
                    print(f"  ℹ️ UHI Intensity 10m {method} not available for both periods")
            except Exception as e:
                print(f"  ⚠️ Failed to export Delta UHI Intensity 10m {method}: {e}")
    elif export_delta_uhi_intensity_10m:
        print("\n  ℹ️ UHI Intensity 10m not available for delta computation")

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