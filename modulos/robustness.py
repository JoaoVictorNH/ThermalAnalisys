# ==============================================================================
# robustness.py — Sensitivity of the thermal results to the acquisition dates
# ==============================================================================
# A single pair of Landsat scenes mixes land-cover change with the weather and
# season of each day (Silva and Torres, 2021). This module recomputes the
# thermal results for alternative date choices — e.g. the anniversary pair and
# the median of all clear scenes of the same season (as in Campelo et al.,
# 2026, who aggregate many scenes per season) — keeping the change mask and the
# Sentinel-2 grid fixed, and reports whether the conclusions hold.
# ==============================================================================

import math

import ee
import numpy as np

from modulos.image_processing import get_selected_landsat_image, process_landsat_image
from modulos.lst_analysis import calculate_uhi
from modulos.spectral_analysis import calculate_spectral_indices_landsat
from modulos.stats_analysis import analyze_joint_index_regression, analyze_new_buildings_grid
from modulos.thermal_composite import seasonal_landsat


def landsat_thermal_period(roi, s2_projection, reference_geometry=None, landsat_id=None,
                           start=None, end=None, landsat_type='L8', cloud_cover=10):
    """
    LST (°C) and Landsat spectral indices for one period, from a single scene
    (landsat_id) or from the median of all scenes between start and end.

    Returns dict: lst, ls_optical, indices, n_scenes, dates (list of 'YYYY-MM-DD'), label.
    """
    geom = roi if reference_geometry is None else roi.union(reference_geometry, 1)
    if landsat_id:
        img = get_selected_landsat_image(landsat_id, geom, landsat_type)
        ls_opt, lst, _ = process_landsat_image(img, s2_projection)
        d = str(landsat_id).split('_')[-1]
        dates = [f'{d[:4]}-{d[4:6]}-{d[6:8]}']
        label = dates[0]
    else:
        # same seasonal median as the main pipeline (cell 5.1): one mosaic per clear day
        p = seasonal_landsat(geom, start, end, landsat_type, cloud_cover, s2_projection,
                             roi=roi, reference_geometry=reference_geometry)
        ls_opt, lst, dates = p['ls_optical'], p['lst'], p['days']
        label = f'median {start}…{end}'
    return {'lst': lst.rename('LST'), 'ls_optical': ls_opt,
            'indices': calculate_spectral_indices_landsat(ls_opt),
            'n_scenes': len(dates), 'dates': dates, 'label': label}


def _region_stats(img, geometry, scale=30):
    r = img.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.median(), sharedInputs=True),
                         geometry, scale, maxPixels=1e10, tileScale=4).getInfo()
    keys = list(r.keys())
    mean = next((r[k] for k in keys if k.endswith('_mean')), None)
    med = next((r[k] for k in keys if k.endswith('_median')), None)
    return (float('nan') if mean is None else float(mean),
            float('nan') if med is None else float(med))


def thermal_date_sensitivity(
    scenarios,
    roi,
    s2_projection,
    mask_ee,
    crs,
    reference_geometry=None,
    index_names=('SAVI', 'DBSI', 'NBAI'),
    controls_t1=None,
    change_indices=None,
    response_joint='delta_uhi',
    response_grid='dlst',
    landsat_type='L8',
    cloud_cover=10,
    num_samples=2000,
    n_boot=500,
    rainfall_days=30,
    verbose=True,
):
    """
    Recomputes the main thermal results for several choices of acquisition dates.

    scenarios: {label: {'t1': {...}, 't2': {...}}}, each period given either as
        {'landsat_id': 'LC08_218063_20190927'} or {'start': '2019-08-01', 'end': '2019-10-31'}.
    The change mask (mask_ee), the Sentinel-2 change indices and controls stay the same,
    so only the thermal side varies between scenarios.

    For each scenario it reports:
      - mean ΔLST in the analysis area and the shift of the reference mean (μ2 − μ1):
        their difference is the warming of the area relative to the city;
      - mean ΔSUHI in the analysis area;
      - joint R² of the Δ indices (and the unique/shared split) on ΔSUHI or ΔLST;
      - grid contrast built − stable with 95% CI, and the dose–response slope b1 with CI;
      - antecedent rainfall (CHIRPS) for single-date periods.
    """
    from modulos.lst_analysis import antecedent_rainfall

    ref = reference_geometry if reference_geometry is not None else roi
    rows, details = [], {}
    for label, sc in scenarios.items():
        if verbose:
            print(f'⏳ {label} …')
        try:
            p = {k: landsat_thermal_period(roi, s2_projection, reference_geometry,
                                           landsat_type=sc[k].get('landsat_type', landsat_type),
                                           cloud_cover=sc[k].get('cloud_cover', cloud_cover),
                                           landsat_id=sc[k].get('landsat_id'),
                                           start=sc[k].get('start'), end=sc[k].get('end'))
                 for k in ('t1', 't2')}
            uhi, st = {}, {}
            for k in ('t1', 't2'):
                uhi[k], s = calculate_uhi(p[k]['lst'], ref)
                st[k] = {kk: float(v.getInfo()) for kk, v in s.items()}
            dlst = p['t2']['lst'].subtract(p['t1']['lst']).rename('dLST')
            duhi = uhi['t2'].subtract(uhi['t1']).rename('dUHI')
            dlst_mean, dlst_med = _region_stats(dlst, roi)
            duhi_mean, _ = _region_stats(duhi, roi)
            ref_shift = st['t2']['lst_mean'] - st['t1']['lst_mean']

            joint = analyze_joint_index_regression(
                p['t1']['indices'], p['t2']['indices'], list(index_names),
                p['t1']['lst'], p['t2']['lst'], roi,
                uhi_t1_image=uhi['t1'], uhi_t2_image=uhi['t2'],
                response=response_joint, scale=30, num_samples=num_samples, n_boot=n_boot)
            grid = analyze_new_buildings_grid(
                mask_ee, p['t1']['lst'], p['t2']['lst'], roi, crs,
                controls_t1=controls_t1, change_indices=change_indices,
                uhi_t1_image=uhi['t1'], uhi_t2_image=uhi['t2'],
                response=response_grid, n_boot=n_boot)

            rain = {}
            for k in ('t1', 't2'):
                if p[k]['n_scenes'] == 1:
                    try:
                        rain[k] = antecedent_rainfall(ref, p[k]['dates'][0], days=rainfall_days)['rain_mm']
                    except Exception:
                        rain[k] = float('nan')
                else:
                    rain[k] = float('nan')

            bs = grid['contrasts']['built_vs_stable']
            row = {
                'scenario': label,
                't1': p['t1']['label'], 't2': p['t2']['label'],
                'n_scenes_t1': p['t1']['n_scenes'], 'n_scenes_t2': p['t2']['n_scenes'],
                'rain_t1_mm': rain['t1'], 'rain_t2_mm': rain['t2'],
                'ref_mean_t1': st['t1']['lst_mean'], 'ref_mean_t2': st['t2']['lst_mean'],
                'ref_shift': ref_shift,
                'dlst_area_mean': dlst_mean, 'dlst_area_median': dlst_med,
                'dlst_area_minus_ref': dlst_mean - ref_shift,
                'duhi_area_mean': duhi_mean,
                'joint_r2': joint['partition']['r2_joint'],
                'joint_r2_ci': joint['ci'].get('r2_joint', (math.nan, math.nan)),
                'shared': joint['partition']['shared'],
                'unique': dict(joint['partition']['unique']),
                'built_minus_stable': bs['diff'],
                'built_minus_stable_ci': grid['ci'].get('built_vs_stable', (math.nan, math.nan)),
                'cliffs_delta': bs['cliffs_delta'],
                'b1': grid['model']['coef']['frac_new'],
                'b1_ci': grid['ci'].get('b_frac_new', (math.nan, math.nan)),
                'n_built': grid['groups']['built']['n'], 'n_stable': grid['groups']['stable']['n'],
                'sufficient': (grid['sufficiency'] or {}).get('built_mean_outside'),
                'grid_unit': '°C' if response_grid == 'dlst' else 'z',
                'joint_response': joint['y_label'],
            }
            rows.append(row)
            details[label] = {'periods': p, 'joint': joint, 'grid': grid}
        except Exception as e:
            print(f'   ✗ {label}: {e}')
    return {'rows': rows, 'details': details, 'index_names': list(index_names)}


def print_thermal_sensitivity(res):
    """Summary table: do the conclusions hold across date choices?"""
    rows = res['rows']
    if not rows:
        print('No scenario could be computed.')
        return
    u = rows[0]['grid_unit']
    print('=' * 112)
    print('SENSITIVITY OF THE THERMAL RESULTS TO THE ACQUISITION DATES (change mask fixed)')
    print('=' * 112)
    for r in rows:
        print(f"\n▶ {r['scenario']}:  T1 = {r['t1']} ({r['n_scenes_t1']} scene(s))  |  "
              f"T2 = {r['t2']} ({r['n_scenes_t2']} scene(s))")
        if not (math.isnan(r['rain_t1_mm']) and math.isnan(r['rain_t2_mm'])):
            print(f"   antecedent rain (CHIRPS): T1 {r['rain_t1_mm']:.1f} mm · T2 {r['rain_t2_mm']:.1f} mm")
        print(f"   reference mean LST: {r['ref_mean_t1']:.2f} → {r['ref_mean_t2']:.2f} °C "
              f"(shift {r['ref_shift']:+.2f} °C)")
        print(f"   ΔLST analysis area: mean {r['dlst_area_mean']:+.2f} °C, median {r['dlst_area_median']:+.2f} °C"
              f"  →  relative to the city {r['dlst_area_minus_ref']:+.2f} °C  |  mean ΔSUHI {r['duhi_area_mean']:+.2f} z")
        lo, hi = r['joint_r2_ci']
        uq = ', '.join(f'{k} {v:.3f}' for k, v in r['unique'].items())
        print(f"   joint R² of Δ indices on {r['joint_response']}: {r['joint_r2']:.3f} [{lo:.3f}, {hi:.3f}]"
              f"  (shared {r['shared']:.3f}; unique: {uq})")
        lo, hi = r['built_minus_stable_ci']
        blo, bhi = r['b1_ci']
        print(f"   grid: built − stable {r['built_minus_stable']:+.2f} {u} [{lo:+.2f}, {hi:+.2f}], "
              f"Cliff's δ {r['cliffs_delta']:+.2f} (n built {r['n_built']}, stable {r['n_stable']}); "
              f"b1 {r['b1']:+.2f} [{blo:+.2f}, {bhi:+.2f}]; outside stable range: {r['sufficient']}")
    signs = {np.sign(r['built_minus_stable']) for r in rows if not math.isnan(r['built_minus_stable'])}
    excl0 = [r['built_minus_stable_ci'][1] < 0 or r['built_minus_stable_ci'][0] > 0 for r in rows]
    print('\n' + '-' * 112)
    print('Robust if the sign of built − stable is the same in every scenario and every interval excludes 0.')
    print(f"Same sign in all scenarios: {len(signs) == 1}  |  intervals excluding 0: {sum(excl0)}/{len(rows)}")


def plot_thermal_sensitivity(res, figsize=(13, 4.2)):
    """Forest plots of built − stable, b1 and joint R² per scenario."""
    import matplotlib.pyplot as plt
    rows = res['rows']
    labels = [r['scenario'] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    specs = ((axes[0], 'built_minus_stable', 'built_minus_stable_ci', f"built − stable ({rows[0]['grid_unit']})", 0),
             (axes[1], 'b1', 'b1_ci', f"b1: fully built cell ({rows[0]['grid_unit']})", 0),
             (axes[2], 'joint_r2', 'joint_r2_ci', 'joint R² of Δ indices', None))
    for ax, k, kci, title, ref in specs:
        for yi, r in zip(y, rows):
            lo, hi = r[kci]
            ax.plot([lo, hi], [yi, yi], color='#555555', lw=2)
            ax.plot(r[k], yi, 'o', color='#d73027', ms=7)
        if ref is not None:
            ax.axvline(ref, color='k', lw=0.8, ls='--')
        ax.set_title(title, fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    fig.suptitle('Do the results hold for other acquisition dates? (95% CI, spatial block bootstrap)', fontsize=11)
    fig.tight_layout()
    return fig
