# ==============================================================================
# validation_export.py — Validation package (geometries, metadata, tables)
# ==============================================================================
# Writes, next to the rasters exported by export_utils, everything needed to
# re-check the results later without re-running the pipeline:
#
#   areas/            analysis area (ROI) and urban perimeter as GeoJSON
#   run_metadata.json image IDs, dates, CRS, reference mode, areas, version
#   reference_stats.csv   μ, σ, °C class limits and valid share of the
#                         thermal reference, per date
#   extent_summary.csv    LST / SUHI / ΔLST / ΔSUHI statistics for the ROI
#                         AND for the urban perimeter (same water mask as μ, σ)
#   class_areas.csv       area of each SUHI class (Deng et al., 2023) per extent
#   pixel_samples_<extent>.csv  random 30 m pixels with all variables (lon/lat)
#   + the tables of the analysis cells that were run (rainfall, 8.1b, 9.1,
#     9.1b, 9.3, 6.6).
#
# Everything is plain CSV / JSON / GeoJSON (QGIS, R, Excel, pandas).
# ==============================================================================

import csv
import datetime as _dt
import json
import math
import os
import shutil

import ee
import numpy as np

from modulos.export_utils import distinct_reference_geometry

try:
    from modulos import __version__ as _VERSION
except Exception:  # pragma: no cover
    _VERSION = 'unknown'

CLASS_NAMES = {1: 'LTZ', 2: 'SLTZ', 3: 'MTZ', 4: 'SHTZ', 5: 'HTZ'}
_WATER = 'JRC/GSW1_4/GlobalSurfaceWater'


# ── Generic helpers ──────────────────────────────────────────────────────────

def _jsonable(obj):
    """Converts numpy / tuples / NaN to JSON types; drops Earth Engine objects."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, (ee.ComputedObject,)):
                continue
            out[str(k)] = _jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, ee.ComputedObject):
        return None
    return obj


def _flat(row, prefix=''):
    """Flattens nested dicts / pairs into CSV columns (a: {b: 1} → a_b; (lo, hi) → _lo, _hi)."""
    out = {}
    for k, v in row.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flat(v, key + '_'))
        elif isinstance(v, (list, tuple)) and len(v) == 2 and all(
                isinstance(x, (int, float, np.floating, np.integer)) or x is None for x in v):
            out[key + '_lo'], out[key + '_hi'] = v
        elif isinstance(v, (list, tuple, np.ndarray)):
            out[key] = json.dumps(_jsonable(v))
        else:
            out[key] = _jsonable(v)
    return out


def _write_csv(path, rows):
    rows = [_flat(r) for r in rows]
    if not rows:
        return None
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ('' if r.get(c) is None else r.get(c)) for c in cols})
    return path


def _write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_jsonable(obj), f, ensure_ascii=False, indent=2, default=str)
    return path


def _columns_csv(path, columns):
    """Writes a dict of equal-length 1-D arrays as a CSV table."""
    names = list(columns)
    n = len(next(iter(columns.values()))) if columns else 0
    rows = [{k: _jsonable(columns[k][i]) for k in names} for i in range(n)]
    return _write_csv(path, rows)


# ── Earth Engine layers shared by the summaries ──────────────────────────────

def _stack(results, stats_change):
    """LST, SUHI z-score, classes and deltas for T1/T2 as one multi-band image."""
    u = stats_change['UHI_30m']
    lst1, lst2 = results['lst_30m_t1'], results['lst_30m_t2']
    z1, z2 = u['images']['t1'], u['images']['t2']
    bands = [lst1.rename('LST_t1'), lst2.rename('LST_t2'), lst2.subtract(lst1).rename('dLST'),
             z1.rename('SUHI_t1'), z2.rename('SUHI_t2'), z2.subtract(z1).rename('dSUHI')]
    cls = u.get('intensity_classification', {})
    if cls.get('t1') is not None and cls.get('t2') is not None:
        bands += [cls['t1'].rename('class_t1'), cls['t2'].rename('class_t2')]
    return ee.Image.cat(bands).toFloat()


def _water_mask():
    """1 = land (kept), 0 = permanent water (JRC occurrence ≥ 50 %), as in calculate_uhi."""
    return ee.Image(_WATER).select('occurrence').gte(50).unmask(0).Not()


def _extents(results, roi, reference_geometry=None):
    ref = reference_geometry if reference_geometry is not None else distinct_reference_geometry(results, roi)
    ext = {'roi': roi}
    if ref is not None:
        ext['perimeter'] = ref
    return ext


# ── Tables computed on Earth Engine ──────────────────────────────────────────

VARIABLES = ('LST_t1', 'LST_t2', 'dLST', 'SUHI_t1', 'SUHI_t2', 'dSUHI')


def summarize_extents(results, stats_change, roi, reference_geometry=None, scale=30,
                      exclude_water=True):
    """
    Statistics of LST, SUHI and their T2 − T1 change for the ROI and for the
    urban perimeter (one server round trip).

    Water bodies are left out (as in the μ, σ of the z-score) when
    `exclude_water`; the perimeter mean of LST_t1 / LST_t2 then reproduces
    the reference μ of reference_stats.csv.

    Returns:
        (summary_rows, class_rows): lists of dicts for extent_summary.csv and
        class_areas.csv.
    """
    ext = _extents(results, roi, reference_geometry)
    img = _stack(results, stats_change)
    if exclude_water:
        img = img.updateMask(_water_mask())
    cont = img.select(list(VARIABLES))
    red = (ee.Reducer.count()
           .combine(ee.Reducer.mean(), '', True)
           .combine(ee.Reducer.median(), '', True)
           .combine(ee.Reducer.stdDev(), '', True)
           .combine(ee.Reducer.minMax(), '', True)
           .combine(ee.Reducer.percentile([5, 25, 75, 95]), '', True))
    has_cls = 'class_t1' in img.bandNames().getInfo()

    server = {}
    for key, geom in ext.items():
        d = {'stats': cont.reduceRegion(red, geom, scale, maxPixels=1e12, tileScale=4),
             'area_km2': geom.area(1).divide(1e6)}
        if has_cls:
            for t in ('t1', 't2'):
                d[f'classes_{t}'] = (ee.Image.pixelArea().divide(1e6)
                                     .addBands(img.select(f'class_{t}').round().toInt())
                                     .reduceRegion(ee.Reducer.sum().group(1, 'class'), geom, scale,
                                                   maxPixels=1e12, tileScale=4).get('groups'))
        server[key] = ee.Dictionary(d)
    info = ee.Dictionary(server).getInfo()

    summary, classes = [], []
    for key in ext:
        st, area = info[key]['stats'], info[key]['area_km2']
        for v in VARIABLES:
            n = st.get(f'{v}_count') or 0
            summary.append({
                'extent': key, 'variable': v,
                'unit': '°C' if 'LST' in v else 'z',
                'area_km2': area, 'n_pixels': n,
                'valid_area_km2': n * scale * scale / 1e6,
                'mean': st.get(f'{v}_mean'), 'median': st.get(f'{v}_median'),
                'std': st.get(f'{v}_stdDev'), 'min': st.get(f'{v}_min'), 'max': st.get(f'{v}_max'),
                'p05': st.get(f'{v}_p5'), 'p25': st.get(f'{v}_p25'),
                'p75': st.get(f'{v}_p75'), 'p95': st.get(f'{v}_p95'),
                'water_excluded': bool(exclude_water),
            })
        if has_cls:
            for t in ('t1', 't2'):
                groups = {int(g['class']): g['sum'] for g in (info[key].get(f'classes_{t}') or [])}
                tot = sum(v for c, v in groups.items() if c in CLASS_NAMES) or float('nan')
                for c, name in CLASS_NAMES.items():
                    a = groups.get(c, 0.0)
                    classes.append({'extent': key, 'date': t.upper(), 'class': c, 'class_name': name,
                                    'area_km2': a, 'share': a / tot if tot == tot else None})
    return summary, classes


def sample_extent_pixels(results, stats_change, roi, reference_geometry=None, mask_ee=None,
                         n_pixels=3000, scale=30, seed=42):
    """
    Random 30 m pixels of each extent with LST, SUHI, classes, deltas, the
    water flag, whether the pixel lies in the ROI / perimeter and (when
    `mask_ee` is given) the share of new buildings detected by the U-Net.

    Returns:
        dict {extent: list of row dicts} (lon/lat in EPSG:4326).
    """
    ext = _extents(results, roi, reference_geometry)
    img = _stack(results, stats_change).addBands(
        _water_mask().Not().rename('water')).addBands(
        ee.Image.constant(1).clip(roi).unmask(0).rename('in_roi'))
    if 'perimeter' in ext:
        img = img.addBands(ee.Image.constant(1).clip(ext['perimeter']).unmask(0).rename('in_perimeter'))
    if mask_ee is not None:
        # the U-Net mask is painted from polygons (cell 6.5) and has no default projection:
        # declare its 10 m grid first, then average it to the 30 m thermal grid
        crs = results['lst_30m_t1'].projection().crs()
        frac = (ee.Image(mask_ee).unmask(0).clip(roi).toFloat()
                .setDefaultProjection(crs=crs, scale=10)
                .reduceResolution(ee.Reducer.mean(), maxPixels=1024)
                .reproject(ee.Projection(crs).atScale(scale)))
        img = img.addBands(frac.unmask(0).rename('frac_new_building'))
    out = {}
    for key, geom in ext.items():
        n = int(min(n_pixels, 5000))
        fc = img.sample(region=geom, scale=scale, numPixels=n, seed=seed, geometries=True, tileScale=4)
        feats = fc.limit(5000).getInfo()['features']
        rows = []
        for f in feats:
            lon, lat = f['geometry']['coordinates']
            rows.append({'extent': key, 'lon': lon, 'lat': lat, **f['properties']})
        out[key] = rows
    return out


# ── Tables from the analysis cells (already in Python) ───────────────────────

def _grid_tables(res_grid):
    cells = {k: np.asarray(v) for k, v in res_grid['cells'].items()}
    m = res_grid.get('masks', {})
    n = len(next(iter(cells.values())))
    group = np.array(['other'] * n, dtype=object)
    for name in ('stable', 'changed', 'built'):
        if name in m:
            group[np.asarray(m[name], bool)] = name
    cols = {}
    xy = res_grid.get('cell_xy')
    if xy is not None:
        cols['x'], cols['y'] = np.asarray(xy['x']), np.asarray(xy['y'])
    cols.update(cells)
    cols['group'] = group
    summary = {k: v for k, v in res_grid.items() if k not in ('cells', 'masks', 'cell_xy')}
    if xy is not None:
        summary['cell_xy_crs'] = xy['crs']
    return cols, summary


def _joint_tables(res_joint):
    s = res_joint['samples']
    X = np.asarray(s['X'])
    cols = {'lon': np.asarray(s['lon']), 'lat': np.asarray(s['lat'])}
    for i, nm in enumerate(res_joint['index_names']):
        cols[f'd_{nm}'] = X[:, i]
    cols['response'] = np.asarray(s['y'])
    summary = {k: v for k, v in res_joint.items() if k != 'samples'}
    return cols, summary


def _zonal_summary(zonal_results):
    keep = ('differences', 'stats_change', 'stats_no_change', 'statistics', 'interpretation',
            'n_change_pixels', 'n_total_pixels')
    return {k: zonal_results[k] for k in keep if k in zonal_results}


def _benchmark_rows(res_cd):
    ci = res_cd.get('ci', {})
    rows = []
    for r in res_cd['rows']:
        lab = r['method']
        rows.append({**r, 'F1_ci': ci.get(f'F1|{lab}'), 'dF1_vs_unet_ci': ci.get(f'dF1|{lab}')})
    meta = {k: v for k, v in res_cd.items() if k not in ('rows', 'ci', 'maps')}
    return rows, meta


# ── Metadata and geometries ──────────────────────────────────────────────────

def _run_metadata(results, config, ext_info, extra=None):
    def _cfg(c):
        keep = ('start_date', 'end_date', 'landsat_type', 'cloud_cover', 'landsat_id', 'sentinel2_id')
        return {k: c.get(k) for k in keep}
    meta = {
        'framework_version': _VERSION,
        'created_utc': _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        'site_name': config.get('site_name'),
        'export_crs': config.get('export_crs'),
        'export_folder': config.get('export_folder'),
        'periods': {k: _cfg(results[k]['config']) for k in ('t1', 't2') if k in results},
        'dates_used': results.get('dates'),
        'thermal_reference': ('urban perimeter (separate geometry)' if 'perimeter' in ext_info
                              else 'analysis area itself (z-score mean = 0 in the ROI)'),
        'suhi_definition': 'z = (LST - mu_ref) / sigma_ref; mu, sigma over the thermal reference, '
                           'permanent water (JRC GSW occurrence >= 50 %) excluded; '
                           'classes after Deng et al. (2023)',
        'lst_source': 'Landsat C2 L2 ST_B10 (DN*0.00341802+149-273.15), 30 m',
        'thermal_composition': results.get('thermal', {'mode': 'selected_scenes'}),
        'extents': ext_info,
        'raster_naming': {'roi': '<product>_<site>_...',
                          'perimeter': '<product>_<site>_perimeter_...'},
    }
    if extra:
        meta['analysis_settings'] = extra
    return meta


def _write_geometries(folder, ext):
    info = ee.Dictionary({k: ee.Dictionary({'geometry': g, 'area_km2': g.area(1).divide(1e6),
                                            'perimeter_km': g.perimeter(1).divide(1e3),
                                            'bounds': g.bounds(1)})
                          for k, g in ext.items()}).getInfo()
    names = {'roi': 'analysis_area_roi', 'perimeter': 'urban_perimeter_thermal_reference'}
    out = {}
    for k, d in info.items():
        fc = {'type': 'FeatureCollection', 'features': [{
            'type': 'Feature', 'geometry': d['geometry'],
            'properties': {'extent': k, 'area_km2': d['area_km2'], 'perimeter_km': d['perimeter_km']}}]}
        path = os.path.join(folder, f'{names[k]}.geojson')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(fc, f)
        xs = [c[0] for c in d['bounds']['coordinates'][0]]
        ys = [c[1] for c in d['bounds']['coordinates'][0]]
        out[k] = {'file': f'areas/{names[k]}.geojson', 'area_km2': d['area_km2'],
                  'perimeter_km': d['perimeter_km'],
                  'bbox_lonlat': [min(xs), min(ys), max(xs), max(ys)]}
    return out


# ── Main entry point ─────────────────────────────────────────────────────────

def export_validation_package(output_dir, results, stats_change, config, roi,
                              reference_geometry=None, mask_ee=None,
                              rainfall=None, res_joint=None, zonal_results=None,
                              res_grid=None, res_dates=None, res_cd_benchmark=None,
                              n_pixel_samples=3000, include_pixel_samples=True,
                              exclude_water=True, analysis_settings=None, make_zip=True):
    """
    Writes the validation package of one run to `output_dir` (e.g. a Google
    Drive folder mounted in Colab). Only the inputs that are given are written;
    the ROI and urban-perimeter tables are always produced.

    Args:
        output_dir (str): destination folder (created if needed).
        results, stats_change, config, roi: outputs of cells 4.0 / 5.1.
        reference_geometry: urban perimeter (None = results['reference_geometry']).
        mask_ee (ee.Image, optional): U-Net change mask (cell 6.5) → share of new
            buildings in the pixel samples.
        rainfall (dict, optional): {'T1': antecedent_rainfall(...), 'T2': ...} (5.2).
        res_joint (dict, optional): analyze_joint_index_regression (8.1b).
        zonal_results (dict, optional): apply_building_change_mask_to_results (9.1).
        res_grid (dict, optional): analyze_new_buildings_grid (9.1b).
        res_dates (dict, optional): thermal_date_sensitivity (9.3).
        res_cd_benchmark (dict, optional): benchmark_change_detection (6.6).
        n_pixel_samples (int): random pixels per extent (≤ 5000).
        include_pixel_samples (bool): write pixel_samples_<extent>.csv.
        exclude_water (bool): same water mask as the z-score reference.
        analysis_settings (dict, optional): thresholds etc. stored in run_metadata.json.
        make_zip (bool): also write <output_dir>.zip.

    Returns:
        dict: {'folder', 'files', 'zip'}
    """
    os.makedirs(output_dir, exist_ok=True)
    area_dir = os.path.join(output_dir, 'areas')
    tab_dir = os.path.join(output_dir, 'tables')
    os.makedirs(area_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    files = []

    def _add(p):
        if p:
            files.append(os.path.relpath(p, output_dir))
            print(f'   ✓ {os.path.relpath(p, output_dir)}')

    print('=' * 78)
    print('VALIDATION PACKAGE — analysis area (ROI) and urban perimeter')
    print('=' * 78)
    ext = _extents(results, roi, reference_geometry)
    if 'perimeter' not in ext:
        print("⚠️  No separate urban perimeter (cell 2.3 = 'Same as analysis area'): "
              "only the ROI is written.")

    print('\n— Geometries')
    ext_info = _write_geometries(area_dir, ext)
    for k in ext_info:
        _add(os.path.join(output_dir, ext_info[k]['file']))

    print('\n— Thermal reference and extent statistics (Earth Engine)')
    u = stats_change['UHI_30m']
    ref_rows = []
    for t in ('t1', 't2'):
        th = u['thresholds_c'][t]
        ref_rows.append({
            'date': t.upper(),
            'landsat_date': (results.get('dates') or {}).get(t, {}).get('landsat'),
            'composite': (results.get('dates') or {}).get(t, {}).get('composite'),
            'reference': 'urban perimeter' if 'perimeter' in ext else 'roi',
            'mu_c': u[t]['lst_mean'], 'sigma_c': u[t]['lst_std'],
            'valid_reference_share': u.get('valid_reference_share', {}).get(t),
            'LTZ_below_c': th['LTZ_below'], 'SLTZ_c': th['SLTZ'], 'MTZ_c': th['MTZ'],
            'SHTZ_c': th['SHTZ'], 'HTZ_above_c': th['HTZ_above'],
        })
    _add(_write_csv(os.path.join(tab_dir, 'reference_stats.csv'), ref_rows))

    summary, classes = summarize_extents(results, stats_change, roi, ext.get('perimeter'),
                                         exclude_water=exclude_water)
    _add(_write_csv(os.path.join(tab_dir, 'extent_summary.csv'), summary))
    _add(_write_csv(os.path.join(tab_dir, 'class_areas.csv'), classes))

    if include_pixel_samples:
        print(f'\n— Pixel samples ({min(n_pixel_samples, 5000)} per extent)')
        try:
            smp = sample_extent_pixels(results, stats_change, roi, ext.get('perimeter'),
                                       mask_ee=mask_ee, n_pixels=n_pixel_samples)
            for k, rows in smp.items():
                _add(_write_csv(os.path.join(tab_dir, f'pixel_samples_{k}.csv'), rows))
        except Exception as e:
            print(f'   ⚠️ pixel samples failed: {e}')

    print('\n— Analysis tables')
    if rainfall:
        _add(_write_csv(os.path.join(tab_dir, 'rainfall_antecedent.csv'),
                        [{'date_label': k, **v} for k, v in rainfall.items()]))
    if res_cd_benchmark:
        rows, meta = _benchmark_rows(res_cd_benchmark)
        _add(_write_csv(os.path.join(tab_dir, 'cd_benchmark.csv'), rows))
        _add(_write_json(os.path.join(tab_dir, 'cd_benchmark_meta.json'), meta))
    if res_joint:
        cols, summ = _joint_tables(res_joint)
        _add(_columns_csv(os.path.join(tab_dir, 'joint_regression_samples.csv'), cols))
        _add(_write_json(os.path.join(tab_dir, 'joint_regression_summary.json'), summ))
    if zonal_results:
        _add(_write_json(os.path.join(tab_dir, 'zonal_built_vs_unbuilt.json'),
                         _zonal_summary(zonal_results)))
    if res_grid:
        cols, summ = _grid_tables(res_grid)
        _add(_columns_csv(os.path.join(tab_dir, 'grid_cells.csv'), cols))
        _add(_write_json(os.path.join(tab_dir, 'grid_summary.json'), summ))
    if res_dates and res_dates.get('rows'):
        _add(_write_csv(os.path.join(tab_dir, 'robustness_scenarios.csv'), res_dates['rows']))

    meta = _run_metadata(results, config, ext_info, analysis_settings)
    meta['files'] = sorted(files + ['run_metadata.json', 'README.txt'])
    _add(_write_json(os.path.join(output_dir, 'run_metadata.json'), meta))
    _add(_write_readme(os.path.join(output_dir, 'README.txt'), meta, 'perimeter' in ext))

    zip_path = None
    if make_zip:
        zip_path = shutil.make_archive(output_dir.rstrip('/'), 'zip', output_dir)
        print(f'\n📦 {zip_path}')
    print(f'\n✅ {len(files)} files in {output_dir}')
    return {'folder': output_dir, 'files': files, 'zip': zip_path}


def _write_readme(path, meta, has_perimeter):
    lines = [
        f"Validation package — {meta.get('site_name')} — framework v{meta.get('framework_version')}",
        f"Created {meta.get('created_utc')}",
        '',
        'Extents',
        '  roi        analysis area (focus zone) — areas/analysis_area_roi.geojson',
        ('  perimeter  urban perimeter = thermal reference of the SUHI z-score — '
         'areas/urban_perimeter_thermal_reference.geojson') if has_perimeter else
        '  (no separate perimeter: the z-score was referenced to the ROI)',
        '',
        'tables/',
        '  reference_stats.csv      mu, sigma (°C) of the reference, °C limits of each SUHI class, valid share',
        '  extent_summary.csv       LST_t1, LST_t2, dLST (°C) and SUHI_t1, SUHI_t2, dSUHI (z) per extent:',
        '                           n, mean, median, std, min, max, p05-p95 (water excluded as in mu, sigma)',
        '  class_areas.csv          km² and share of LTZ…HTZ per extent and date',
        '  pixel_samples_<ext>.csv  random 30 m pixels (lon/lat WGS84) with all variables, in_roi,',
        '                           in_perimeter, water and frac_new_building (U-Net) when available',
        '  rainfall_antecedent.csv  CHIRPS rain before each date (cell 5.2)',
        '  cd_benchmark*.{csv,json} U-Net vs index differencing (cell 6.6)',
        '  joint_regression_*       samples and commonality analysis (cell 8.1b)',
        '  zonal_built_vs_unbuilt   pixels with vs without new buildings (cell 9.1)',
        '  grid_cells.csv           grid cells (x, y in export CRS) with frac_new, dLST, dUHI, group (9.1b)',
        '  grid_summary.json        groups, contrasts, dose-response, model, bootstrap CI (9.1b)',
        '  robustness_scenarios.csv selected pair / anniversary pair / seasonal median (9.3)',
        '',
        'Rasters (Google Drive export folder, cell 10.2)',
        '  ROI:       <product>_<site>_t1 / _t2 / delta ...',
        '  perimeter: <product>_<site>_perimeter_t1 / _t2 / delta ... (Landsat products only)',
        '',
        'Consistency checks',
        '  * extent_summary (perimeter, LST_t1 / LST_t2, mean) ≈ reference_stats mu_c',
        '  * extent_summary (perimeter, SUHI_t1 / SUHI_t2, mean) ≈ 0 and std ≈ 1',
        '  * SUHI = (LST - mu_c) / sigma_c for every row of pixel_samples',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path
