# ==============================================================================
# lst_analysis.py — Land Surface Temperature Analysis (LST, UHI)
# ==============================================================================
# Normalized SUHI (z-score), intensity classification (Deng et al., 2023),
# UHI statistics, area per intensity class and antecedent rainfall (CHIRPS).
# ==============================================================================

import ee


def calculate_uhi(lst_image, geometry, scale=30, exclude_water=True):
    """
    Calculates the normalized SUHI index (z-score of LST).

        SUHI = (LST − μ) / σ

    μ and σ are taken over `geometry`, the thermal reference area. For an
    intra-urban SUHI (Martin et al., 2015) pass the urban perimeter here, not
    the small analysis ROI: a z-score referenced to the ROI itself has mean 0
    on every date, so a relative increase in one part forces a decrease in another.

    Args:
        lst_image (ee.Image): LST in °C (single band).
        geometry (ee.Geometry): Thermal reference area for μ and σ.
        scale (int): Scale in meters for the statistics.
        exclude_water (bool): Leave permanent water bodies (JRC Global Surface
            Water, occurrence ≥ 50 %) out of μ and σ, as Martin et al. (2015)
            exclude water from the urban reference. The returned image is not masked.

    Returns:
        tuple: (ee.Image 'UHI', {'lst_mean': ee.Number, 'lst_std': ee.Number})
    """
    lst_for_stats = lst_image
    if exclude_water:
        water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gte(50)
        lst_for_stats = lst_image.updateMask(water.unmask(0).Not())

    # Mean and standard deviation in a single server call
    mean_std = lst_for_stats.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=geometry,
        scale=scale,
        maxPixels=1e10,
        tileScale=4
    )
    band_name = lst_image.bandNames().get(0)
    lst_mean = ee.Number(mean_std.get(ee.String(band_name).cat('_mean')))
    lst_std = ee.Number(mean_std.get(ee.String(band_name).cat('_stdDev'))).max(1e-9)

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


def antecedent_rainfall(geometry, date_str, days=30, clim_years=(1991, 2020)):
    """
    Rainfall accumulated in the `days` before an acquisition date (CHIRPS daily,
    ~5.5 km), and the same window's climatological mean for comparison.

    Silva and Torres (2021) check the rainfall before each image because soil
    moisture changes the surface temperature, mostly on vegetated and bare soil.

    Args:
        geometry (ee.Geometry): Area averaged (ROI or urban perimeter).
        date_str (str): 'YYYY-MM-DD' acquisition date.
        days (int): Window length before the date.
        clim_years (tuple): First and last year of the climatology.

    Returns:
        dict: {'date', 'days', 'rain_mm', 'clim_mean_mm', 'anomaly_mm', 'anomaly_pct'}
    """
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').select('precipitation')
    end = ee.Date(date_str)

    def _window_total(end_date):
        end_date = ee.Date(end_date)
        img = chirps.filterDate(end_date.advance(-days, 'day'), end_date).sum()
        return ee.Number(img.reduceRegion(ee.Reducer.mean(), geometry, 5566,
                                          maxPixels=1e9, bestEffort=True).get('precipitation'))

    md = str(date_str)[5:10]
    month_day = ee.String('02-28' if md == '02-29' else md)  # leap day → Feb 28
    years = ee.List.sequence(clim_years[0], clim_years[1])
    clim = years.map(lambda y: _window_total(
        ee.Date.parse('YYYY-MM-dd', ee.Number(y).format('%d').cat('-').cat(month_day))))

    rain, clim_mean = ee.List([_window_total(end),
                               ee.List(clim).reduce(ee.Reducer.mean())]).getInfo()
    rain = float(rain or 0.0)
    clim_mean = float(clim_mean or 0.0)
    return {
        'date': date_str,
        'days': days,
        'rain_mm': rain,
        'clim_mean_mm': clim_mean,
        'anomaly_mm': rain - clim_mean,
        'anomaly_pct': (100.0 * (rain - clim_mean) / clim_mean) if clim_mean > 0 else float('nan'),
    }


def print_antecedent_rainfall(rows):
    """Prints the output of antecedent_rainfall for T1 and T2 ({'T1': dict, 'T2': dict})."""
    print('=' * 78)
    print('ANTECEDENT RAINFALL (CHIRPS daily) — context for comparing the two dates')
    print('=' * 78)
    print(f"{'':<5}{'date':<12}{'window':>8}{'rain (mm)':>12}{'climatology (mm)':>19}{'anomaly':>14}")
    for lbl, r in rows.items():
        anom = '—' if r['anomaly_pct'] != r['anomaly_pct'] else f"{r['anomaly_pct']:+.0f}%"
        print(f"{lbl:<5}{r['date']:<12}{r['days']:>6} d{r['rain_mm']:>12.1f}{r['clim_mean_mm']:>19.1f}"
              f"{r['anomaly_mm']:>+9.1f} ({anom})")
    print('Climatology: mean of the same calendar window, 1991–2020.')
    print('Very different antecedent rain between T1 and T2 means part of ΔLST reflects soil moisture.')
