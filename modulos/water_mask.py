# ==============================================================================
# water_mask.py — Manual Water Body Mask
# ==============================================================================
# Tools for interactive creation of water masks via manual polygon drawing
# on geemap maps. The mask excludes water bodies from MLR regression training
# and replaces those pixels with bilinear resample of the coarse LST.
#
# Typical usage (notebook):
#   Cell N  : map_water = create_combined_water_mask_map(s2_t1, s2_t2, roi)
#               display(map_water)       # user draws polygons
#   Cell N+1: water_mask_t1 = extract_water_mask(map_water, roi)
#               water_mask_t2 = water_mask_t1   # same mask for both periods
# ==============================================================================

import ee
import geemap


def create_combined_water_mask_map(s2_t1, s2_t2, roi,
                                   label_t1='T1', label_t2='T2', zoom=12):
    """
    Creates a combined geemap map with RGB and MNDWI for T1 and T2 to help
    the user draw polygons over water bodies.

    The map displays four toggleable layers:
      - RGB T1 (visible by default)
      - MNDWI T1 (blue = water)
      - RGB T2
      - MNDWI T2

    A single drawing session generates the mask applied to both periods,
    since permanent water bodies do not change position between T1 and T2.

    Args:
        s2_t1      : ee.Image — Sentinel-2 SR mosaic for T1 (requires B3, B4, B11)
        s2_t2      : ee.Image — Sentinel-2 SR mosaic for T2 (requires B3, B4, B11)
        roi        : ee.Geometry — region of interest
        label_t1   : str — label for period 1 (e.g., 'T1')
        label_t2   : str — label for period 2 (e.g., 'T2')
        zoom       : int — initial map zoom level

    Returns:
        geemap.Map — map ready for interactive drawing.
        Access .draw_features after drawing to retrieve the polygons.
    """
    def build_mosaic(img):
        date = ee.Date(img.get('system:time_start'))
        next_day = date.advance(1, 'day')
        col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(roi) \
                .filterDate(date, next_day)
        return col.mosaic().clip(roi)

    s2_t1_mosaic = build_mosaic(s2_t1)
    s2_t2_mosaic = build_mosaic(s2_t2)

    vis_rgb   = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.4}
    vis_mndwi = {'min': -0.3, 'max': 0.5, 'palette': ['D2691E', 'FFFACD', '0000FF']}

    mndwi_t1 = s2_t1_mosaic.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    mndwi_t2 = s2_t2_mosaic.normalizedDifference(['B3', 'B11']).rename('MNDWI')

    ndvi_t1 = s2_t1_mosaic.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndvi_t2 = s2_t2_mosaic.normalizedDifference(['B8', 'B4']).rename('NDVI')

    m = geemap.Map()
    m.centerObject(roi, zoom)

    # T1 visible by default; T2 off — user toggles as needed
    m.addLayer(s2_t1_mosaic, vis_rgb,   f'RGB {label_t1}',                       shown=True)
    m.addLayer(mndwi_t1,     vis_mndwi, f'MNDWI {label_t1} (brown→beige→blue)',  shown=False, opacity=0.65)
    m.addLayer(ndvi_t1,      {'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green']}, f'NDVI {label_t1}', shown=False, opacity=0.65)
    m.addLayer(s2_t2_mosaic, vis_rgb,   f'RGB {label_t2}',                       shown=False)
    m.addLayer(mndwi_t2,     vis_mndwi, f'MNDWI {label_t2} (brown→beige→blue)',  shown=False, opacity=0.65)
    m.addLayer(ndvi_t2,      {'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green']}, f'NDVI {label_t2}', shown=False, opacity=0.65)
    m.addLayer(roi, {'color': 'FFFF00'}, 'ROI', shown=True, opacity=0.3)

    print("╔══ Water Mask — Single Polygon Set for T1 and T2 ════════════════════╗")
    print("║  1. Toggle RGB/MNDWI layers to identify water bodies               ║")
    print("║     (use the layer panel in the upper-right corner of the map)     ║")
    print("║  2. Use the polygon tool (toolbar on the left side of the map)     ║")
    print("║  3. Draw over each water body visible in the ROI                   ║")
    print("║  4. You can draw multiple polygons — blue in MNDWI indicates water ║")
    print("║  5. The same mask will be applied to T1 and T2 automatically       ║")
    print("║  6. If there are no water bodies, run the next cell                ║")
    print("║     WITHOUT drawing — the pipeline will continue normally           ║")
    print("╚═════════════════════════════════════════════════════════════════════╝")

    return m


def extract_water_mask(map_obj, roi):
    """
    Extracts polygons drawn on the map and creates a binary water mask (ee.Image).

    The generated mask has value 1 inside polygons (water) and 0 outside (land).
    It is used in method_mlr() to:
      - Exclude water pixels from regression training
      - Fill water pixels with bilinear resample of the coarse LST

    Args:
        map_obj : geemap.Map — map where polygons were drawn
        roi     : ee.Geometry — region of interest (to clip the mask)

    Returns:
        ee.Image with band 'water_mask' (1=water, 0=land), or None if no
        polygons were drawn.
    """
    drawn = map_obj.draw_features  # list of GeoJSON feature dicts

    if not drawn:
        print("  No polygons drawn.")
        print("  → Downscaling will proceed without a water mask.")
        return None

    n = len(drawn)
    print(f"  {n} polygon(s) detected. Generating water mask...")

    geojson_fc = {"type": "FeatureCollection", "features": drawn}
    water_fc = ee.FeatureCollection(geojson_fc)

    water_mask = (
        water_fc
        .map(lambda f: f.set('water', 1))
        .reduceToImage(properties=['water'], reducer=ee.Reducer.first())
        .unmask(0)
        .gt(0)
        .clip(roi)
        .rename('water_mask')
    )

    print(f"  Water mask created successfully ({n} polygon(s)).")
    print(f"  → Water pixels will be excluded from MLR training (T1 and T2).")
    print(f"  → Water pixels will receive bilinear resample of Landsat LST (30m→10m).")
    return water_mask