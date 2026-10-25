# ==============================================================================
# thermal_composite.py — Seasonal median of the thermal data (Landsat only)
# ==============================================================================
# A single Landsat pair mixes land-cover change with the weather and season of
# each day (Silva and Torres, 2021). Here the LST and the Landsat bands of each
# period are replaced by the per-pixel median of all clear days of a seasonal
# window (as Campelo et al., 2026, aggregate many scenes per season), while the
# change detection (Sentinel-1/2 of the selected scenes) and the Sentinel-2
# indices are kept unchanged.
#
# The choice is defensible only if the two windows are comparable. The checks
# reported by `check_seasonal_windows` are:
#   1. same season      — windows on the same calendar days in both years and the
#                         mean day of year of the scenes within 15 days;
#   2. same length      — window lengths differ by ≤ 10 %;
#   3. change dates     — each window contains the Sentinel-2 date of the change
#                         mask, so the thermal data bracket the detected change;
#   4. similar sample   — ≥ 3 clear days per window and ratio between periods ≥ 0.5;
#   5. same cloud limit — the same maximum cloud cover in both windows;
#   6. valid days/pixel — after the cloud mask, every pixel of the analysis area
#                         has ≥ 2 valid days (≥ 3 recommended).
# ==============================================================================

import datetime as _dt

import ee

from modulos.image_processing import get_landsat_collection, process_landsat_median, landsat_label
from modulos.spectral_analysis import calculate_spectral_indices_landsat

MIN_DAYS = 3
MIN_RATIO = 0.5
MAX_SEASON_OFFSET_DAYS = 15
MAX_LENGTH_DIFF = 0.10
MIN_VALID_DAYS_PIXEL = 2


# ── Composition ──────────────────────────────────────────────────────────────

def _daily_mosaics(col):
    """One mosaic per acquisition day (adjacent rows/paths of the same day are merged),
    so that a day covered by two scenes is not counted twice in the median."""
    days = col.aggregate_array('system:time_start').map(
        lambda t: ee.Date(t).format('YYYY-MM-dd')).distinct().sort()

    def _mosaic(d):
        d = ee.Date.parse('YYYY-MM-dd', d)
        day = col.filterDate(d, d.advance(1, 'day'))
        return (day.mosaic().set('system:time_start', d.millis())
                .set('date', d.format('YYYY-MM-dd')))

    return ee.ImageCollection(days.map(_mosaic)), days


def seasonal_landsat(geometry, start, end, landsat_type, cloud_cover, s2_projection,
                     roi=None, reference_geometry=None):
    """
    Per-pixel median of the clear Landsat days between `start` and `end`.

    Returns:
        dict: lst (ee.Image, band ST_B10 in °C), ls_optical, indices,
              days (list 'YYYY-MM-DD'), product_ids, sensors, valid_days
              ({'roi': {...}, 'perimeter': {...}}: min / mean / share < 3 of the
              number of valid days per pixel).
    """
    cfg = {'geometry': geometry, 'start_date': start, 'end_date': end,
           'landsat_type': landsat_type, 'cloud_cover': cloud_cover}
    col = get_landsat_collection(cfg)
    daily, days = _daily_mosaics(col)
    if not days.size().getInfo():
        raise ValueError(f'No clear {landsat_label(landsat_type)} scene between {start} and {end} '
                         f'with < {cloud_cover}% clouds.')
    ref_proj = col.first().select('ST_B10').projection()

    count = daily.select('ST_B10').count().rename('n').setDefaultProjection(ref_proj)
    red = (ee.Reducer.min().combine(ee.Reducer.mean(), '', True))
    areas = {'roi': roi if roi is not None else geometry}
    if reference_geometry is not None:
        areas['perimeter'] = reference_geometry
    valid = {k: ee.Dictionary({
        'stats': count.reduceRegion(red, g, 30, maxPixels=1e10, tileScale=4),
        'share_lt3': count.lt(3).reduceRegion(ee.Reducer.mean(), g, 30, maxPixels=1e10, tileScale=4).get('n'),
    }) for k, g in areas.items()}
    info = ee.Dictionary({
        'days': days,
        'ids': col.aggregate_array('LANDSAT_PRODUCT_ID'),
        'clouds': col.aggregate_array('CLOUD_COVER'),
        'valid': ee.Dictionary(valid),
    }).getInfo()

    ls_opt, lst, _ = process_landsat_median(daily.map(lambda i: i.setDefaultProjection(ref_proj)), s2_projection)
    lst, ls_opt = lst.clip(geometry), ls_opt.clip(geometry)

    valid_days = {}
    for k, v in info['valid'].items():
        st = v.get('stats') or {}
        valid_days[k] = {'min': st.get('n_min'), 'mean': st.get('n_mean'), 'share_lt3': v.get('share_lt3')}
    sensors = sorted({('Landsat 9' if str(i).startswith('LC09') else 'Landsat 8') for i in info['ids']})
    return {'lst': lst, 'ls_optical': ls_opt, 'indices': calculate_spectral_indices_landsat(ls_opt),
            'days': info['days'], 'product_ids': info['ids'], 'scene_clouds': info['clouds'],
            'sensors': sensors, 'valid_days': valid_days}


# ── Checks ───────────────────────────────────────────────────────────────────

def _d(s):
    return _dt.date.fromisoformat(str(s)[:10])


def _doy_offset(a, b):
    """Smallest difference in day of year between two dates (wrap-around safe)."""
    da, db = _d(a).timetuple().tm_yday, _d(b).timetuple().tm_yday
    x = abs(da - db)
    return min(x, 365 - x)


def _mean_doy(days):
    import math
    ang = [2 * math.pi * _d(x).timetuple().tm_yday / 365.25 for x in days]
    m = math.atan2(sum(math.sin(a) for a in ang), sum(math.cos(a) for a in ang))
    return (m % (2 * math.pi)) * 365.25 / (2 * math.pi)


def check_seasonal_windows(windows, periods, change_dates, cloud_cover):
    """
    Comparability checks of the two seasonal windows (see the module header).

    Args:
        windows: {'t1': (start, end), 't2': (start, end)} ('YYYY-MM-DD').
        periods: {'t1': seasonal_landsat(...), 't2': ...}.
        change_dates: {'t1': 'YYYY-MM-DD', 't2': ...} Sentinel-2 dates of the change mask.
        cloud_cover: maximum cloud cover used in both windows.

    Returns:
        list of dicts {criterion, t1, t2, ok (True/False/None), rule}.
    """
    rows = []
    (s1, e1), (s2, e2) = windows['t1'], windows['t2']
    off = max(_doy_offset(s1, s2), _doy_offset(e1, e2))
    m1, m2 = _mean_doy(periods['t1']['days']), _mean_doy(periods['t2']['days'])
    phase = min(abs(m1 - m2), 365.25 - abs(m1 - m2))
    rows.append({'criterion': 'Same season (window start/end offset; mean day of the scenes)',
                 't1': f'{s1} → {e1}', 't2': f'{s2} → {e2}',
                 'value': f'window offset {off} d; scene phase offset {phase:.0f} d',
                 'ok': off <= MAX_SEASON_OFFSET_DAYS and phase <= MAX_SEASON_OFFSET_DAYS,
                 'rule': f'≤ {MAX_SEASON_OFFSET_DAYS} days'})
    l1 = (_d(e1) - _d(s1)).days + 1
    l2 = (_d(e2) - _d(s2)).days + 1
    rows.append({'criterion': 'Same window length', 't1': f'{l1} d', 't2': f'{l2} d',
                 'value': f'{abs(l1 - l2) / max(l1, l2):.0%}',
                 'ok': abs(l1 - l2) / max(l1, l2) <= MAX_LENGTH_DIFF, 'rule': f'difference ≤ {MAX_LENGTH_DIFF:.0%}'})
    inside = {k: (change_dates.get(k) is not None and _d(windows[k][0]) <= _d(change_dates[k]) <= _d(windows[k][1]))
              for k in ('t1', 't2')}
    rows.append({'criterion': 'Window contains the Sentinel-2 date of the change mask',
                 't1': f"{change_dates.get('t1')} {'✓' if inside['t1'] else '✗'}",
                 't2': f"{change_dates.get('t2')} {'✓' if inside['t2'] else '✗'}",
                 'value': '', 'ok': all(inside.values()), 'rule': 'both inside'})
    n1, n2 = len(periods['t1']['days']), len(periods['t2']['days'])
    ratio = min(n1, n2) / max(n1, n2) if max(n1, n2) else 0
    rows.append({'criterion': 'Clear days per window (similar sample)', 't1': n1, 't2': n2,
                 'value': f'ratio {ratio:.2f}', 'ok': min(n1, n2) >= MIN_DAYS and ratio >= MIN_RATIO,
                 'rule': f'≥ {MIN_DAYS} days each; ratio ≥ {MIN_RATIO}'})
    rows.append({'criterion': 'Same maximum cloud cover', 't1': f'{cloud_cover}%', 't2': f'{cloud_cover}%',
                 'value': '', 'ok': True, 'rule': 'identical'})
    v1 = periods['t1']['valid_days'].get('roi', {})
    v2 = periods['t2']['valid_days'].get('roi', {})

    def _fmt(v):
        if v.get('min') is None:
            return '—'
        return f"min {v['min']:.0f}, mean {v['mean']:.1f}, <3: {(v.get('share_lt3') or 0):.0%}"
    mins = [v.get('min') for v in (v1, v2)]
    rows.append({'criterion': 'Valid days per pixel in the analysis area (after cloud mask)',
                 't1': _fmt(v1), 't2': _fmt(v2), 'value': '',
                 'ok': None if None in mins else min(mins) >= MIN_VALID_DAYS_PIXEL,
                 'rule': f'min ≥ {MIN_VALID_DAYS_PIXEL} (≥ 3 recommended)'})
    rows.append({'criterion': 'Sensors (informative)', 't1': ' + '.join(periods['t1']['sensors']),
                 't2': ' + '.join(periods['t2']['sensors']), 'value': '', 'ok': None,
                 'rule': 'same Collection 2 Level 2 calibration'})
    return rows


def print_seasonal_checks(thermal):
    """Prints the scenes of each window and the comparability checks."""
    print('=' * 92)
    print('THERMAL DATA — seasonal median (Landsat only); change mask and Sentinel-2 from the selected scenes')
    print('=' * 92)
    for k in ('t1', 't2'):
        p = thermal['periods'][k]
        s, e = thermal['windows'][k]
        print(f"{k.upper()}  {s} → {e}  |  {len(p['days'])} clear days ({' + '.join(p['sensors'])}): "
              + ', '.join(x[5:] for x in p['days']))
    print('\nComparability of the two windows (report these with the results):')
    for r in thermal['checks']:
        mark = '—' if r['ok'] is None else ('✓' if r['ok'] else '⚠️')
        extra = f"  [{r['value']}]" if r['value'] else ''
        print(f"  {mark}  {r['criterion']}: T1 {r['t1']} | T2 {r['t2']}{extra}  (rule: {r['rule']})")
    bad = [r for r in thermal['checks'] if r['ok'] is False]
    if bad:
        print('\n⚠️  Some criteria failed: the two medians may not be comparable. Adjust the windows or the cloud limit,'
              ' or report the selected pair as main result and the median as sensitivity.')
    else:
        print('\n✓ The two windows are comparable by all criteria.')


# ── Pipeline entry point ─────────────────────────────────────────────────────

def apply_seasonal_thermal(results, t1_window, t2_window, landsat_type='L8+L9', cloud_cover=10):
    """
    Replaces the thermal data of `results` (calculate_bitemporal_lst) by the
    seasonal medians, keeping the Sentinel-1/2 data, the Sentinel-2 indices and
    therefore the change detection of the selected scenes.

    Replaced: lst_30m_t1/t2, t1/t2 lst_30m, lst_celsius, landsat_image, ls_optical,
    indices_landsat. Added: results['thermal'] with windows, days, checks.

    Returns:
        dict: new results (the input is not modified).
    """
    windows = {'t1': tuple(t1_window), 't2': tuple(t2_window)}
    new = dict(results)
    new['indices_landsat'] = dict(results.get('indices_landsat', {}))
    new['dates'] = {k: dict(v) for k, v in results.get('dates', {}).items()}
    periods = {}
    for k in ('t1', 't2'):
        cfg = results[k]['config']
        roi = cfg['geometry']
        ref = cfg.get('reference_geometry')
        geom = roi if ref is None or ref is roi else roi.union(ref, 1)
        p = seasonal_landsat(geom, windows[k][0], windows[k][1], landsat_type, cloud_cover,
                             results[k]['s2_projection'], roi=roi,
                             reference_geometry=None if ref is None or ref is roi else ref)
        periods[k] = p
        new[k] = {**results[k], 'lst_30m': p['lst'], 'lst_celsius': p['lst'],
                  'landsat_image': p['ls_optical'], 'ls_optical': p['ls_optical']}
        new[f'lst_30m_{k}'] = p['lst']
        new['indices_landsat'][k] = p['indices']
        new['dates'].setdefault(k, {})
        new['dates'][k].update({'thermal_window': list(windows[k]), 'thermal_days': p['days'],
                                'thermal_composite': True})
    change_dates = {k: results.get('dates', {}).get(k, {}).get('sentinel2') for k in ('t1', 't2')}
    new['thermal'] = {
        'mode': 'seasonal_median',
        'landsat_type': landsat_type,
        'cloud_cover': cloud_cover,
        'windows': windows,
        'periods': {k: {kk: vv for kk, vv in p.items() if kk not in ('lst', 'ls_optical', 'indices')}
                    for k, p in periods.items()},
        'change_detection_dates': change_dates,
        'checks': check_seasonal_windows(windows, periods, change_dates, cloud_cover),
    }
    return new
