# ==============================================================================
# image_processing.py — Satellite Image Processing
# ==============================================================================
# Analysis configuration (bitemporal), collection processing for
# Landsat/Sentinel-2/Sentinel-1, cloud masking, image selection,
# bandpass adjustment, and preview visualization.
# ==============================================================================

import re
import ee
import geemap
import numpy as np

from modulos.gee_setup import get_utm_crs


# ==========================================================================
# 1. ANALYSIS CONFIGURATION
# ==========================================================================

def create_tenst_config(
    start_date,
    end_date,
    geometry,
    landsat_type='L8',
    cloud_cover=20,
    landsat_id=None,
    sentinel2_id=None,
    export_folder=None,
    export_crs='EPSG:32724',
):
    """
    Configures processing parameters for Landsat.

    Args:
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        geometry: Region of interest geometry.
        landsat_type (str): 'L8' or 'L9'.
        cloud_cover (int): Maximum cloud cover (%).
        landsat_id (str, optional): Specific Landsat image ID.
        sentinel2_id (str, optional): Specific Sentinel-2 image ID.
        export_folder (str): Google Drive export folder.
        export_crs (str): Coordinate system for export.

    Returns:
        dict: Configuration dictionary.
    """
    config = {
        "start_date": start_date,
        "end_date": end_date,
        "geometry": geometry,
        "landsat_type": landsat_type,
        "cloud_cover": cloud_cover,
        "landsat_id": landsat_id,
        "sentinel2_id": sentinel2_id,
        "export_folder": export_folder,
        "export_crs": export_crs,
    }

    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Search period: {start_date} to {end_date}")
    print(f"Satellite: Landsat {landsat_type}")
    print(f"Maximum cloud cover: {cloud_cover}%")
    if landsat_id:
        print(f"Selected Landsat ID: {landsat_id}")
    if sentinel2_id:
        print(f"Selected Sentinel-2 ID: {sentinel2_id}")
    print(f"Coordinate system: {export_crs}")
    print("=" * 80)

    return config


def create_bitemporal_config(
    t1_start_date,
    t1_end_date,
    t2_start_date,
    t2_end_date,
    geometry,
    landsat_type='L8',
    cloud_cover=20,
    landsat_t1_id=None,
    landsat_t2_id=None,
    sentinel2_t1_id=None,
    sentinel2_t2_id=None,
    export_folder=None,
    site_name='study_area',
    export_crs='EPSG:32724',
):
    """
    Configuration for bitemporal thermal analysis.

    Args:
        t1_start_date (str): T1 period start date (YYYY-MM-DD).
        t1_end_date (str): T1 period end date (YYYY-MM-DD).
        t2_start_date (str): T2 period start date (YYYY-MM-DD).
        t2_end_date (str): T2 period end date (YYYY-MM-DD).
        geometry: Region of interest geometry.
        landsat_type (str): 'L8' or 'L9'.
        cloud_cover (int): Maximum cloud cover (%).
        landsat_t1_id (str, optional): Specific Landsat T1 image ID.
        landsat_t2_id (str, optional): Specific Landsat T2 image ID.
        sentinel2_t1_id (str, optional): Specific Sentinel-2 T1 image ID.
        sentinel2_t2_id (str, optional): Specific Sentinel-2 T2 image ID.
        export_folder (str): Google Drive export folder.
        site_name (str): Study site name.
        export_crs (str): Coordinate system for export.

    Returns:
        dict: Configuration dictionary.
    """
    config = {
        "t1_start_date": t1_start_date,
        "t1_end_date": t1_end_date,
        "t2_start_date": t2_start_date,
        "t2_end_date": t2_end_date,
        "geometry": geometry,
        "landsat_type": landsat_type,
        "cloud_cover": cloud_cover,
        "landsat_t1_id": landsat_t1_id,
        "landsat_t2_id": landsat_t2_id,
        "sentinel2_t1_id": sentinel2_t1_id,
        "sentinel2_t2_id": sentinel2_t2_id,
        "export_folder": export_folder,
        "site_name": site_name,
        "export_crs": export_crs,
    }

    print("=" * 80)
    print("BITEMPORAL THERMAL ANALYSIS CONFIGURATION")
    print("=" * 80)
    print(f"T1 period: {t1_start_date} to {t1_end_date}")
    if landsat_t1_id:
        print(f"  Landsat T1 ID: {landsat_t1_id}")
    if sentinel2_t1_id:
        print(f"  Sentinel-2 T1 ID: {sentinel2_t1_id}")
    print(f"T2 period: {t2_start_date} to {t2_end_date}")
    if landsat_t2_id:
        print(f"  Landsat T2 ID: {landsat_t2_id}")
    if sentinel2_t2_id:
        print(f"  Sentinel-2 T2 ID: {sentinel2_t2_id}")
    print(f"Satellite: Landsat {landsat_type}")
    print(f"Maximum cloud cover: {cloud_cover}%")
    print(f"Site: {site_name}")
    print(f"Coordinate system: {export_crs}")
    print("=" * 80)

    return config


def create_configs_from_ids(
    roi,
    t1_start_date,
    t1_end_date,
    t2_start_date,
    t2_end_date,
    landsat_type,
    cloud_cover,
    landsat_t1_id,
    sentinel2_t1_id,
    landsat_t2_id,
    sentinel2_t2_id,
    site_name='study_area',
    export_crs='EPSG:32724',
    export_folder=None,
):
    """
    Creates configuration dictionaries for bitemporal analysis and individual
    periods based on selected image IDs.

    Args:
        roi: Region of interest.
        t1_start_date (str): T1 period start date (YYYY-MM-DD).
        t1_end_date (str): T1 period end date (YYYY-MM-DD).
        t2_start_date (str): T2 period start date (YYYY-MM-DD).
        t2_end_date (str): T2 period end date (YYYY-MM-DD).
        landsat_type (str): 'L8' or 'L9'.
        cloud_cover (int): Maximum cloud cover (%).
        landsat_t1_id (str): Specific Landsat T1 image ID.
        sentinel2_t1_id (str): Specific Sentinel-2 T1 image ID.
        landsat_t2_id (str): Specific Landsat T2 image ID.
        sentinel2_t2_id (str): Specific Sentinel-2 T2 image ID.
        site_name (str): Study site name.
        export_crs (str): Coordinate system for export.
        export_folder (str): Google Drive export folder.

    Returns:
        tuple: (config_bitemporal, config_t1, config_t2)
    """
    config_bitemporal = create_bitemporal_config(
        t1_start_date=t1_start_date,
        t1_end_date=t1_end_date,
        t2_start_date=t2_start_date,
        t2_end_date=t2_end_date,
        geometry=roi,
        landsat_type=landsat_type,
        cloud_cover=cloud_cover,
        landsat_t1_id=landsat_t1_id,
        sentinel2_t1_id=sentinel2_t1_id,
        landsat_t2_id=landsat_t2_id,
        sentinel2_t2_id=sentinel2_t2_id,
        site_name=site_name,
        export_crs=export_crs,
        export_folder=export_folder,
    )

    config_t1 = create_tenst_config(
        start_date=t1_start_date,
        end_date=t1_end_date,
        geometry=roi,
        landsat_type=landsat_type,
        cloud_cover=cloud_cover,
        landsat_id=landsat_t1_id,
        sentinel2_id=sentinel2_t1_id,
        export_crs=export_crs,
        export_folder=export_folder,
    )

    config_t2 = create_tenst_config(
        start_date=t2_start_date,
        end_date=t2_end_date,
        geometry=roi,
        landsat_type=landsat_type,
        cloud_cover=cloud_cover,
        landsat_id=landsat_t2_id,
        sentinel2_id=sentinel2_t2_id,
        export_crs=export_crs,
        export_folder=export_folder,
    )

    return config_bitemporal, config_t1, config_t2


# ==========================================================================
# 2. SCALE FACTORS AND CLOUD MASKING
# ==========================================================================

def apply_landsat_scale_factors(image):
    """
    Applies scale factors for Landsat Collection 2.

    Converts optical bands to surface reflectance and the thermal band
    (ST_B10) to temperature in degrees Celsius.

    Args:
        image (ee.Image): Landsat C02 T1_L2 image.

    Returns:
        ee.Image: Image with scale factors applied.
    """
    optical = (image.select('SR_B.')
               .multiply(0.0000275)
               .add(-0.2))

    thermal = (image.select('ST_B.*')
               .multiply(0.00341802)
               .add(149.0)
               .subtract(273.15))

    return image.addBands(optical, None, True).addBands(thermal, None, True)


def apply_sentinel2_scale_factors(image):
    """
    Applies scale factors for Sentinel-2 (Level-2A).

    Converts optical bands to surface reflectance (factor 0.0001).

    Args:
        image (ee.Image): Sentinel-2 Level-2A image.

    Returns:
        ee.Image: Image with scale factors applied.
    """
    optical = (image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
               .multiply(0.0001))

    return image.addBands(optical, None, True)


def maskL8clouds(image):
    """
    Applies cloud, shadow, fill, and water mask for Landsat 8/9 Collection 2.

    Uses the QA_PIXEL band from Landsat C02 Level-2.
    Masking water is essential for LST analysis: the USGS ST algorithm does not
    process water bodies (ASTER GED emissivity unavailable), so ST_B10 is null
    on those pixels while optical bands remain valid.
    Without this mask, water pixels create a nullity mismatch between RGB and LST.

    Masked bits:
        Bit 0: Fill         — pixels with no sensor data
        Bit 3: Cloud        — clouds
        Bit 4: Cloud Shadow — cloud shadows
        Bit 7: Water        — water (ST_B10 is null here by C2 design)

    Args:
        image (ee.Image): Landsat C02 T1_L2 image.

    Returns:
        ee.Image: Masked image.
    """
    qa = image.select('QA_PIXEL')

    fill_bit_mask         = 1 << 0  # Bit 0: Fill
    clouds_bit_mask       = 1 << 3  # Bit 3: Cloud
    cloud_shadow_bit_mask = 1 << 4  # Bit 4: Cloud Shadow

    mask = (qa.bitwiseAnd(fill_bit_mask).eq(0)
            .And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
            .And(qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0)))

    return image.updateMask(mask)


def maskS2clouds(image):
    """
    Cloud and cirrus mask for Sentinel-2 SR.

    Based on the QA60 band (bits 10 and 11).

    Args:
        image (ee.Image): Sentinel-2 image.

    Returns:
        ee.Image: Masked image.
    """
    qa = image.select('QA60')

    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
           qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask)


# ==========================================================================
# 3. IMAGE COLLECTION SEARCH
# ==========================================================================

def get_sentinel2_collection(config):
    """
    Retrieves Sentinel-2 image collection filtered by date, region, and cloud cover.

    Automatically selects between S2_HARMONIZED (TOA) and S2_SR_HARMONIZED
    (Surface Reflectance) based on the start date.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: (ee.ImageCollection, str) — Filtered collection and collection name.
    """
    start_date_obj = ee.Date(config['start_date'])
    cutoff_date = ee.Date('2017-03-28')

    if start_date_obj.millis().lt(cutoff_date.millis()).getInfo():
        collection_name = 'COPERNICUS/S2_HARMONIZED'
        print(f"Using collection: {collection_name} (TOA)")
    else:
        collection_name = 'COPERNICUS/S2_SR_HARMONIZED'
        print(f"Using collection: {collection_name} (Surface Reflectance)")

    collection = (ee.ImageCollection(collection_name)
                 .filterBounds(config['geometry'])
                 .filterDate(config['start_date'], config['end_date'])
                 .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', config['cloud_cover']))

    if collection_name == 'COPERNICUS/S2_SR_HARMONIZED':
        collection = collection.map(maskS2clouds).map(apply_sentinel2_scale_factors)
    else:
        collection = collection.map(maskS2clouds).map(apply_sentinel2_scale_factors)

    return collection


def get_landsat_collection(config):
    """
    Retrieves Landsat 8 or 9 image collection filtered by date, region, and cloud cover.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        ee.ImageCollection: Filtered Landsat collection.
    """
    collection_ids = {
        'L8': 'LANDSAT/LC08/C02/T1_L2',
        'L9': 'LANDSAT/LC09/C02/T1_L2'
    }

    collection = (ee.ImageCollection(collection_ids[config['landsat_type']])
                 .filterBounds(config['geometry'])
                 .filterDate(config['start_date'], config['end_date'])
                 .filterMetadata('CLOUD_COVER', 'less_than', config['cloud_cover'])
                 .map(maskL8clouds)
                 .map(apply_landsat_scale_factors))

    return collection


def get_sentinel1_collection(roi, start_date, end_date, polarizations=["VV", "VH"]):
    """
    Retrieves Sentinel-1 SAR image collection (GRD, IW mode).

    Args:
        roi: Region of interest geometry.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        polarizations: List of polarizations (default: ['VV', 'VH']).

    Returns:
        ee.ImageCollection: Filtered Sentinel-1 collection.
    """
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filterMetadata('instrumentMode', 'equals', 'IW')
                  .select(polarizations)
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

    return collection


# ==========================================================================
# 4. MEDIAN PROCESSING (COLLECTIONS)
# ==========================================================================

def process_sentinel2_median(sentinel2_collection):
    """
    Processes the Sentinel-2 collection median.

    Args:
        sentinel2_collection (ee.ImageCollection): Sentinel-2 collection.

    Returns:
        tuple: (s2_median, s2_projection)
    """
    print("\n🛰️  Processing Sentinel-2...")

    median = sentinel2_collection.median()
    s2_bands = median.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    s2_projection = median.select('B2').projection()

    print("   ✓ Sentinel-2 median computed")

    return s2_bands, s2_projection


def process_landsat_median(landsat_collection, s2_projection):
    """
    Processes the Landsat collection median and reprojects to the Sentinel-2 grid.

    Extracts LST directly from the ST_B10 band (already converted to °C by scale
    factors) and optical bands, reprojecting both to 30 m.

    Args:
        landsat_collection (ee.ImageCollection): Landsat collection.
        s2_projection (ee.Projection): Reference Sentinel-2 projection.

    Returns:
        tuple: (optical_bands, lst_celsius, ls_projection)
    """
    print("\n🌡️  Processing Landsat...")

    median = landsat_collection.median()
    landsat_projection = landsat_collection.first().projection()

    # LST directly from ST_B10 band (already in Celsius)
    lst_celsius = median.select('ST_B10')

    # Reproject LST
    lst_celsius = (lst_celsius
                   .setDefaultProjection(crs=landsat_projection)
                   .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
                   .reproject(s2_projection, scale=30))

    # Extract and reproject optical bands
    optical_bands = (median.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
                     .setDefaultProjection(crs=landsat_projection)
                     .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
                     .reproject(s2_projection, scale=30))

    ls_projection = median.select('SR_B2').projection()

    print("   ✓ LST and optical bands processed and reprojected")

    return optical_bands, lst_celsius, ls_projection


# ==========================================================================
# 5. SPECIFIC IMAGE SELECTION
# ==========================================================================

def get_selected_landsat_image(landsat_id, roi, landsat_type='L8'):
    """
    Retrieves and processes a Landsat image by ID, creating a mosaic of all scenes
    from the same day that intersect the ROI.

    The ID is used to determine the date and sensor (L8/L9). All available scenes
    from that day covering the ROI are mosaicked — this ensures full coverage when
    the ROI crosses two adjacent paths (e.g., 216/217).

    Automatically detects Landsat 8 or 9 from the ID prefix.

    Args:
        landsat_id (str): Landsat image ID (format: LC08_PPPRRR_YYYYMMDD).
        roi: Region of interest.
        landsat_type (str): 'L8' or 'L9' (fallback if ID does not identify).

    Returns:
        ee.Image: Landsat day mosaic, processed and clipped to ROI.
    """
    collection_ids = {
        'L8': 'LANDSAT/LC08/C02/T1_L2',
        'L9': 'LANDSAT/LC09/C02/T1_L2'
    }

    if landsat_id.startswith('LC08'):
        collection_id = collection_ids['L8']
    elif landsat_id.startswith('LC09'):
        collection_id = collection_ids['L9']
    else:
        collection_id = collection_ids[landsat_type]

    # Parse date from ID (format: LC08_PPPRRR_YYYYMMDD)
    date_str = landsat_id.split('_')[-1]  # 'YYYYMMDD'
    date = ee.Date.fromYMD(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    next_day = date.advance(1, 'day')

    # Native projection of the reference scene (lazy — no download cost).
    # Required because mosaic() does not inherit projection, and process_landsat_image
    # calls .projection() for the setDefaultProjection → reduceResolution chain.
    # Uses SR_B2 (optical 30m) as projection reference for ST_B10 as well.
    ref_projection = ee.Image(f"{collection_id}/{landsat_id}").select('SR_B2').projection()

    # Mosaic of all day scenes covering the ROI (includes adjacent paths)
    landsat_col = (ee.ImageCollection(collection_id)
                   .filterDate(date, next_day)
                   .filterBounds(roi)
                   .map(maskL8clouds)
                   .map(apply_landsat_scale_factors))

    # setDefaultProjection declares the native projection WITHOUT resampling pixels.
    # Unlike .reproject(), it does not force recalculation on the grid — avoids nulls in ST_B10.
    landsat_img = (landsat_col.mosaic()
                   .setDefaultProjection(crs=ref_projection)
                   .clip(roi))

    return landsat_img


def get_selected_sentinel2_image(sentinel2_id, roi, s2_collection='COPERNICUS/S2_HARMONIZED'):
    """
    Loads the Sentinel-2 mosaic from the same day as the provided ID and fills gaps in the ROI.

    Example ID: '20190810T131251_20190810T131246_T24MUB'

    Args:
        sentinel2_id (str): Sentinel-2 image ID.
        roi: Region of interest.
        s2_collection (str): Sentinel-2 collection name.

    Returns:
        ee.Image: Processed and clipped Sentinel-2 mosaic.
    """
    match = re.match(r"(\d{8})", sentinel2_id)
    if not match:
        raise ValueError(f"Invalid ID: {sentinel2_id}")
    date_str = match.group(1)
    date = ee.Date.fromYMD(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    next_day = date.advance(1, 'day')

    # Native projection of the reference tile (lazy — no download cost).
    # Required because mosaic() does not inherit projection; process_sentinel2_image
    # calls .projection() to get the target CRS for LST reproject().
    # Without this, s2_projection is undefined and LST reproject() produces nulls.
    ref_s2_projection = ee.Image(f"{s2_collection}/{sentinel2_id}").select('B2').projection()

    s2 = (ee.ImageCollection(s2_collection)
          .filterDate(date, next_day)
          .filterBounds(roi))

    def mask_s2_clouds(img):
        qa = img.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return img.updateMask(mask)

    try:
        s2 = s2.map(mask_s2_clouds)
    except Exception:
        pass

    s2_mosaic = s2.mosaic()
    s2_mosaic = apply_sentinel2_scale_factors(s2_mosaic)
    s2_mosaic = s2_mosaic.clamp(0, 1).unmask(0).float()
    # setDefaultProjection declares the native projection of the reference tile WITHOUT resampling.
    s2_mosaic = s2_mosaic.setDefaultProjection(crs=ref_s2_projection)
    s2_mosaic = s2_mosaic.clip(roi)

    return s2_mosaic


def generate_sentinel1_composite_from_s2_date(roi, s2_image_id, polarizations=["VV", "VH"]):
    """
    Generates a Sentinel-1 composite based on the Sentinel-2 image date.

    Searches for Sentinel-1 images within a ±30-day window around the Sentinel-2
    image date, selects the orbit with the most images, and produces a normalized
    mean composite.

    Args:
        roi: Region of interest geometry.
        s2_image_id: Reference Sentinel-2 image ID.
        polarizations: List of polarizations (default: ['VV', 'VH']).

    Returns:
        ee.Image: Processed Sentinel-1 composite, or None on error.
    """
    product_id = s2_image_id

    if product_id and len(product_id) >= 8:
        date_part = product_id.split('_')[0]
        if len(date_part) >= 8:
            ano = date_part[:4]
            mes = date_part[4:6]
            dia = date_part[6:8]
            date_str = f'{ano}-{mes}-{dia}'
            s2_date = ee.Date(date_str)
        else:
            print(f"Error: Could not extract date from PRODUCT_ID part: {date_part}")
            return None
    else:
        print(f"Error: Could not extract date from PRODUCT_ID: {product_id}")
        return None

    start_date = s2_date.advance(-30, 'day')
    end_date = s2_date.advance(30, 'day')

    print(f"\nProcessing Sentinel-1 SAR...")
    try:
        start_date_info = start_date.format('YYYY-MM-dd').getInfo()
        end_date_info = end_date.format('YYYY-MM-dd').getInfo()
        print(f"  Period: {start_date_info} to {end_date_info}")
    except ee.EEException as e:
        print(f"Error getting date info: {e}")
        return None

    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(roi)
          .filterDate(start_date, end_date)
          .filterMetadata('instrumentMode', 'equals', 'IW')
          .select(polarizations))

    for pol in polarizations:
        s1 = s1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', pol))

    s1 = s1.map(lambda img: img.updateMask(img.gte(-25)))

    asc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    desc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

    asc_count = asc.size().getInfo()
    desc_count = desc.size().getInfo()

    print(f"  Ascending images: {asc_count}")
    print(f"  Descending images: {desc_count}")

    s1 = asc if asc_count > desc_count else desc
    orbit_type = 'ASCENDING' if asc_count > desc_count else 'DESCENDING'
    print(f"  Using orbit: {orbit_type}")

    orbit_numbers = (s1.toList(s1.size())
                     .map(lambda img: ee.Image(img).get('relativeOrbitNumber_start'))
                     .distinct())

    orbit_list = orbit_numbers.getInfo()
    print(f"  Relative orbits: {orbit_list}")

    def process_orbit(orbit_num):
        orbit_col = s1.filter(ee.Filter.eq('relativeOrbitNumber_start', orbit_num))
        return orbit_col.mean()

    means = ee.ImageCollection([process_orbit(orbit) for orbit in orbit_list])

    mean_mosaic = (means.mosaic()
                   .unitScale(-25, 0)
                   .clamp(0, 1)
                   .unmask()
                   .float())

    band_names = mean_mosaic.bandNames().getInfo()
    new_band_names = []
    for name in band_names:
        if 'VV' in name:
            new_band_names.append('VV')
        elif 'VH' in name:
            new_band_names.append('VH')
        else:
            new_band_names.append(name)

    mean_mosaic = mean_mosaic.rename(new_band_names)

    print(f"  ✓ SAR composite created")

    return mean_mosaic


# ==========================================================================
# 6. SPECIFIC IMAGE PROCESSING
# ==========================================================================

def process_landsat_image(landsat_image, s2_projection):
    """
    Processes a specific Landsat image and reprojects to the Sentinel-2 grid.

    Extracts LST directly from the ST_B10 band (already in °C) and optical bands.

    Args:
        landsat_image (ee.Image): Landsat image.
        s2_projection (ee.Projection): Reference Sentinel-2 projection.

    Returns:
        tuple: (optical_bands, lst_celsius, ls_projection)
    """
    print("\n🌡️  Processing Landsat (specific image)...")

    landsat_projection = landsat_image.projection()

    # LST directly from ST_B10 band (already in Celsius)
    lst_celsius = landsat_image.select('ST_B10')

    # Reproject LST to S2 grid maintaining 30 m scale (no aggregation)
    lst_celsius = lst_celsius.reproject(crs=s2_projection.crs(), scale=30)

    # Extract and reproject optical bands to S2 grid (30 m)
    optical_bands = (landsat_image
                     .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
                     .reproject(crs=s2_projection.crs(), scale=30))

    ls_projection = landsat_image.select('SR_B2').projection()

    print("   ✓ LST and optical bands processed and reprojected")

    return optical_bands, lst_celsius, ls_projection


def process_sentinel2_image(sentinel2_image):
    """
    Processes a specific Sentinel-2 image, extracting bands and projection.

    Args:
        sentinel2_image (ee.Image): Sentinel-2 image.

    Returns:
        tuple: (s2_bands, s2_projection)
    """
    print("\n🛰️  Processing Sentinel-2 (specific image)...")

    s2_bands = sentinel2_image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    s2_projection = sentinel2_image.select('B2').projection()

    print("   ✓ Sentinel-2 image processed")

    return s2_bands, s2_projection


# ==========================================================================
# 7. IMAGE SEARCH AND VISUALIZATION
# ==========================================================================

def search_available_images(config, max_display=20):
    """
    Searches and lists available Landsat and Sentinel-2 images.

    Optimized version with minimum getInfo() calls.

    Args:
        config (dict): Configuration dictionary.
        max_display (int): Maximum number of images to display.

    Returns:
        dict: Information about found images.
    """
    print("\n" + "=" * 80)
    print(f"SEARCHING AVAILABLE IMAGES: {config['start_date']} to {config['end_date']}")
    print("=" * 80)
    print("ℹ️  Times converted to Brasília timezone (UTC-3)")

    def extract_info_landsat(img):
        """Extracts Landsat image information (server-side)."""
        img = ee.Image(img)
        time_utc = ee.Date(img.get('system:time_start'))
        time_brasilia = time_utc.advance(-3, 'hour')
        return ee.Dictionary({
            'id': img.id(),
            'date_utc': time_utc.format('YYYY-MM-dd'),
            'time_utc': time_utc.format('HH:mm:ss'),
            'date_brasilia': time_brasilia.format('YYYY-MM-dd'),
            'time_brasilia': time_brasilia.format('HH:mm:ss'),
            'datetime_brasilia': time_brasilia.format('YYYY-MM-dd HH:mm:ss'),
            'cloud_cover': img.get('CLOUD_COVER')
        })

    def extract_info_sentinel(img):
        """Extracts Sentinel-2 image information (server-side)."""
        img = ee.Image(img)
        time_utc = ee.Date(img.get('system:time_start'))
        time_brasilia = time_utc.advance(-3, 'hour')
        return ee.Dictionary({
            'id': img.id(),
            'date_utc': time_utc.format('YYYY-MM-dd'),
            'time_utc': time_utc.format('HH:mm:ss'),
            'date_brasilia': time_brasilia.format('YYYY-MM-dd'),
            'time_brasilia': time_brasilia.format('HH:mm:ss'),
            'datetime_brasilia': time_brasilia.format('YYYY-MM-dd HH:mm:ss'),
            'cloud_cover': img.get('CLOUDY_PIXEL_PERCENTAGE')
        })

    # ==================== LANDSAT ====================
    print(f"\n🛰️  Searching Landsat {config['landsat_type']}...")
    landsat_col = get_landsat_collection(config)

    landsat_result = ee.Dictionary({
        'count': landsat_col.size(),
        'images': landsat_col.toList(max_display).map(extract_info_landsat)
    }).getInfo()

    n_landsat = landsat_result['count']
    landsat_info = landsat_result['images']
    print(f"   Images found: {n_landsat}")

    if n_landsat > 0:
        print("\n   --- Landsat image IDs ---")
        for i, info in enumerate(landsat_info):
            print(f"   {i+1}. {info['id']}")
            print(f"      Date: {info['date_brasilia']} | Time (Brasília): {info['time_brasilia']} | Clouds: {info['cloud_cover']:.2f}%")
        if n_landsat > max_display:
            print(f"   ... and {n_landsat - max_display} more images")
    else:
        print(f"\n⚠️  WARNING: No Landsat {config['landsat_type']} images found!")

    # ==================== SENTINEL-2 ====================
    print(f"\n🛰️  Searching Sentinel-2...")
    s2_col = get_sentinel2_collection(config)

    s2_result = ee.Dictionary({
        'count': s2_col.size(),
        'images': s2_col.toList(max_display).map(extract_info_sentinel)
    }).getInfo()

    n_s2 = s2_result['count']
    s2_info = s2_result['images']
    print(f"   Images found: {n_s2}")

    if n_s2 > 0:
        print("\n   --- Sentinel-2 image IDs ---")
        for i, info in enumerate(s2_info):
            print(f"   {i+1}. {info['id']}")
            print(f"      Date: {info['date_brasilia']} | Time (Brasília): {info['time_brasilia']} | Clouds: {info['cloud_cover']:.2f}%")
        if n_s2 > max_display:
            print(f"   ... and {n_s2 - max_display} more images")
    else:
        print("\n⚠️  WARNING: No Sentinel-2 images found!")

    if n_s2 > 0 and n_landsat > 0:
        print("\n✓ Sufficient images found for processing")

    return {
        'landsat': landsat_info,
        'sentinel2': s2_info,
        'n_landsat': n_landsat,
        'n_sentinel2': n_s2
    }


def create_preview_map(roi,
                       landsat_id=None,
                       sentinel2_id=None,
                       s2_collection='COPERNICUS/S2_HARMONIZED/',
                       landsat_type='L8',
                       zoom=15,
                       lst_min=20,
                       lst_max=60):
    """
    Creates an interactive map to visualize specific images before selecting.

    Args:
        roi: Region of interest.
        landsat_id (str, optional): Landsat image ID.
        sentinel2_id (str, optional): Sentinel-2 image ID.
        s2_collection (str): Sentinel-2 collection to use.
        landsat_type (str): 'L8' or 'L9'.
        zoom (int): Zoom level.
        lst_min (float): Minimum temperature for visualization.
        lst_max (float): Maximum temperature for visualization.

    Returns:
        geemap.Map: Interactive map with added layers.
    """
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION MAP")
    print("=" * 80)

    Map = geemap.Map()
    Map.centerObject(roi, zoom=zoom)

    Map.addLayer(roi, {'color': 'yellow'}, 'ROI', True, 0.5)

    lst_palette = [
        '040274', '0502a3', '0502ce', '0602ff', '307ef3',
        '30c8e2', '32d3ef', 'fff705', 'ffd611', 'ffb613',
        'ff8b13', 'ff6e08', 'ff500d', 'ff0000', 'de0101',
        'c21301', 'a71001', '911003'
    ]

    # Add Landsat if provided
    if landsat_id:
        print(f"\n📷 Loading Landsat: {landsat_id}")
        try:
            landsat_img = get_selected_landsat_image(landsat_id, roi, landsat_type)

            # RGB
            Map.addLayer(
                landsat_img,
                {'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min': 0, 'max': 0.3, 'gamma': 1.4},
                f'Landsat RGB',
                True
            )

            # LST directly from ST_B10 band
            lst = landsat_img.select('ST_B10')

            Map.addLayer(
                lst,
                {'min': lst_min, 'max': lst_max, 'palette': lst_palette},
                'Landsat LST',
                False
            )

            print(f"   ✓ LST computed")

        except Exception as e:
            print(f"   ⚠️  Error loading Landsat: {str(e)}")

    # Add Sentinel-2 if provided
    if sentinel2_id:
        print(f"\n📷 Loading Sentinel-2: {sentinel2_id}")
        try:
            s2_img = get_selected_sentinel2_image(sentinel2_id, roi, s2_collection)

            Map.addLayer(
                s2_img,
                {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3, 'gamma': 1.3},
                'Sentinel-2 RGB',
                True
            )

            Map.addLayer(
                s2_img,
                {'bands': ['B8', 'B4', 'B3'], 'min': 0, 'max': 0.3, 'gamma': 1.3},
                'Sentinel-2 False Color',
                False
            )

            print("   ✓ Sentinel-2 loaded")

        except Exception as e:
            print(f"   ⚠️  Error loading Sentinel-2: {str(e)}")

    # Add Sentinel-1 if Sentinel-2 provided
    if sentinel2_id:
        print(f"\n📡 Loading Sentinel-1 based on S2: {sentinel2_id}")
        try:
            s1_composite = generate_sentinel1_composite_from_s2_date(roi, sentinel2_id)

            if s1_composite:
                Map.addLayer(
                    s1_composite.select('VV').clip(roi),
                    {'min': 0, 'max': 1, 'palette': ['black', 'white']},
                    'Sentinel-1 VV',
                    False
                )

                Map.addLayer(
                    s1_composite.select('VH').clip(roi),
                    {'min': 0, 'max': 1, 'palette': ['black', 'white']},
                    'Sentinel-1 VH',
                    False
                )

                s1_rgb = s1_composite.select(['VV', 'VH']).addBands(
                    s1_composite.select('VV').divide(s1_composite.select('VH')).rename('ratio')
                )
                Map.addLayer(
                    s1_rgb.clip(roi),
                    {'min': [0, 0, 0], 'max': [1, 1, 2], 'bands': ['VV', 'VH', 'ratio']},
                    'Sentinel-1 RGB',
                    False
                )

                print("   ✓ Sentinel-1 loaded")

        except Exception as e:
            print(f"   ⚠️  Error loading Sentinel-1: {str(e)}")

    if landsat_id:
        Map.add_colorbar(
            {'min': lst_min,
             'max': lst_max,
             'palette': lst_palette},
            label='Surface Temperature (°C)',
            orientation='horizontal',
            transparent_bg=True
        )

    print("\n✓ Map created successfully")

    return Map


# ==========================================================================
# 8. BANDPASS ADJUSTMENT
# ==========================================================================

def perform_bandpass_adjustment(s2_median, l8_median, geometry):
    """
    Performs bandpass adjustment between Sentinel-2 and Landsat 8.

    For each band pair (S2 → L8), computes a linear regression and applies
    the correction to the Sentinel-2 band, reprojecting to 10 m.

    Args:
        s2_median: Sentinel-2 median image.
        l8_median: Landsat median image (optical bands).
        geometry: Region geometry for regression.

    Returns:
        dict: Dictionary with adjusted Sentinel-2 bands.
    """
    print("\n🔧 Performing Bandpass Adjustment...")

    adjusted_bands = {}

    band_mapping = {
        'B4': 'SR_B4',   # Red
        'B3': 'SR_B3',   # Green
        'B2': 'SR_B2',   # Blue
        'B8': 'SR_B5',   # NIR
        'B11': 'SR_B6',  # SWIR1
        'B12': 'SR_B7'   # SWIR2
    }

    S2proj = s2_median.select('B2').projection()

    for s2_band, l8_band in band_mapping.items():
        combined = ee.Image.cat(
            s2_median.select(s2_band),
            l8_median.select(l8_band)
        )

        linear_fit = combined.reduceRegion(
            reducer=ee.Reducer.linearFit(),
            geometry=geometry,
            scale=30,
            tileScale=16,
            maxPixels=1e10
        )

        b0 = linear_fit.get('offset')
        b1 = linear_fit.get('scale')

        adjusted = (s2_median.select(s2_band)
                   .multiply(ee.Number(b1))
                   .add(ee.Number(b0))
                   .reproject(S2proj, None, 10))

        adjusted_bands[s2_band] = adjusted

        print(f"   ✓ Band {s2_band} adjusted")

    return adjusted_bands