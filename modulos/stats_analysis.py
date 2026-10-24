# ==============================================================================
# stats_analysis.py — Joint statistics for Axis 1 and Axis 2
# ==============================================================================
# Axis 1: correlation matrix of the Δ spectral indices and the thermal change,
#         partial correlations, joint (multiple) regression with the explained
#         variance split into the part unique to each index and the part they
#         share (commonality analysis), VIF, and spatial block bootstrap CIs.
# Axis 2: effect of new buildings estimated on a regular grid (~ native TIRS
#         resolution): cells with new buildings vs STABLE cells (no buildings and
#         small Δ indices) vs cells that CHANGED without buildings; whether the
#         effect exceeds the variation of stable land; dose–response on the
#         fraction of new built area; regression controlling for the initial
#         state (LST and SAVI in T1).
#
# Why: individual R² of correlated indices (e.g. SAVI and DBSI) overlap and
# cannot be summed; and pixel-level "with vs. without new buildings" tests
# compare against land that may also have changed, ignore the initial state,
# the 100 m thermal footprint and spatial dependence between neighbouring pixels.
# ==============================================================================

import math

import ee
import numpy as np


# ------------------------------------------------------------------------------
# Generic helpers (numpy only — testable without Earth Engine)
# ------------------------------------------------------------------------------

def _ols(y, X):
    """OLS with intercept. Returns (coefficients incl. intercept, R², adjusted R²)."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else float('nan')
    return coef, r2, r2_adj


def _r2(y, X):
    return _ols(y, X)[1]


def variance_partition(y, X, names):
    """
    Commonality analysis: splits the joint R² into the part UNIQUE to each
    predictor and the part SHARED among them.

        unique_i = R²(all) − R²(all except i)
        shared   = R²(all) − Σ unique_i

    Also returns each predictor's individual R² (what a simple correlation
    reports), their sum (which double-counts the shared part), VIF and
    standardized coefficients.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    k = X.shape[1]
    _, r2_full, r2_adj = _ols(y, X)

    individual, unique, vif = {}, {}, {}
    for i, nm in enumerate(names):
        individual[nm] = _r2(y, X[:, i])
        others = [j for j in range(k) if j != i]
        unique[nm] = r2_full - (_r2(y, X[:, others]) if others else 0.0)
        if others:
            r2_i = _r2(X[:, i], X[:, others])
            vif[nm] = 1.0 / (1.0 - r2_i) if r2_i < 1 else float('inf')
        else:
            vif[nm] = 1.0

    shared = r2_full - sum(unique.values())

    # Standardized coefficients (β): effect of 1 SD of each predictor, in SD of y
    sx = X.std(axis=0, ddof=1)
    sy = y.std(ddof=1)
    Xs = (X - X.mean(axis=0)) / np.where(sx > 0, sx, 1)
    coef_s, _, _ = _ols((y - y.mean()) / (sy if sy > 0 else 1), Xs)
    beta = {nm: float(coef_s[i + 1]) for i, nm in enumerate(names)}

    corr = np.corrcoef(X, rowvar=False) if k > 1 else np.array([[1.0]])

    return {
        'r2_joint': float(r2_full),
        'r2_adj': float(r2_adj),
        'r2_individual': individual,
        'r2_sum_individual': float(sum(individual.values())),
        'unique': unique,
        'shared': float(shared),
        'vif': vif,
        'beta_std': beta,
        'predictor_corr': corr,
        'n': int(len(y)),
    }


def correlation_report(X, y, names, y_name='y'):
    """
    Pearson and Spearman correlation matrices of [Δ indices, response], and the
    PARTIAL correlation of each index with the response controlling for the
    other indices (correlation of the residuals of both on the others).

    The matrix shows the redundancy between indices; the partial correlation
    shows what each index still says about the response once the others are known.
    """
    from scipy.stats import spearmanr

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    M = np.column_stack([X, y])
    labels = [f'Δ{n}' for n in names] + [y_name]
    pearson = np.corrcoef(M, rowvar=False)
    spearman = spearmanr(M).correlation if M.shape[1] > 2 else np.array(
        [[1.0, spearmanr(M[:, 0], M[:, 1]).correlation],
         [spearmanr(M[:, 0], M[:, 1]).correlation, 1.0]])

    partial = {}
    k = X.shape[1]
    for i, nm in enumerate(names):
        others = [j for j in range(k) if j != i]
        if not others:
            partial[nm] = float(pearson[0, -1])
            continue
        A = np.column_stack([np.ones(len(y)), X[:, others]])
        rx = X[:, i] - A @ np.linalg.lstsq(A, X[:, i], rcond=None)[0]
        ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        partial[nm] = float(np.corrcoef(rx, ry)[0, 1])

    return {'labels': labels, 'pearson': pearson, 'spearman': np.asarray(spearman),
            'partial_r': partial}


def _block_ids(x_m, y_m, block_size_m):
    """Assigns each point to a square spatial block of side block_size_m (meters)."""
    bx = np.floor((np.asarray(x_m) - np.min(x_m)) / block_size_m).astype(int)
    by = np.floor((np.asarray(y_m) - np.min(y_m)) / block_size_m).astype(int)
    return bx * 100000 + by


def block_bootstrap(stat_fn, n_points, block_ids, n_boot=1000, seed=42):
    """
    Spatial block bootstrap: resamples whole blocks (not single pixels), so the
    confidence interval respects the dependence between neighbouring pixels.

    stat_fn(idx) -> dict of floats, computed on the rows idx.
    Returns {key: (low_2.5%, high_97.5%)}.
    """
    rng = np.random.default_rng(seed)
    blocks = np.unique(block_ids)
    members = {b: np.where(block_ids == b)[0] for b in blocks}
    draws = {}
    for _ in range(int(n_boot)):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([members[b] for b in pick])
        if len(idx) < 10:
            continue
        try:
            res = stat_fn(idx)
        except Exception:
            continue
        for key, val in res.items():
            draws.setdefault(key, []).append(val)
    return {key: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for key, v in draws.items()}


def cliffs_delta(a, b):
    """
    Cliff's delta (non-parametric effect size, consistent with Mann–Whitney):
    P(a > b) − P(a < b). |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium,
    otherwise large (Romano et al., 2006).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    gt = lt = 0
    # chunked to keep memory bounded
    for chunk in np.array_split(a, max(1, len(a) // 2000 + 1)):
        diff = chunk[:, None] - b[None, :]
        gt += int((diff > 0).sum())
        lt += int((diff < 0).sum())
    return (gt - lt) / (len(a) * len(b))


def cliffs_label(d):
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return '—'
    ad = abs(d)
    return ('negligible' if ad < 0.147 else 'small' if ad < 0.33
            else 'medium' if ad < 0.474 else 'large')


def _lonlat_to_m(lon, lat):
    """Local equirectangular projection (meters) — enough for block assignment."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lat0 = np.nanmean(lat)
    x = np.radians(lon) * 6371000 * math.cos(math.radians(lat0))
    y = np.radians(lat) * 6371000
    return x, y


# ------------------------------------------------------------------------------
# AXIS 1 — joint regression of Δ indices on Δ LST
# ------------------------------------------------------------------------------

def analyze_joint_index_regression(
    indices_t1_dict,
    indices_t2_dict,
    index_names,
    lst_t1_image,
    lst_t2_image,
    roi,
    uhi_t1_image=None,
    uhi_t2_image=None,
    response='delta_lst',
    scale=30,
    num_samples=2000,
    block_size_m=300,
    n_boot=1000,
    iqr_k=3.0,
    seed=42,
):
    """
    Joint explanatory power of several Δ indices on the thermal change.

    All Δ indices and the response are sampled at the SAME points, so the
    individual R², the joint R² and the unique/shared parts are comparable.

    Args:
        indices_t1_dict / indices_t2_dict: {name_lower: ee.Image}.
        index_names (list[str]): e.g. ['SAVI', 'DBSI'].
        lst_t1_image / lst_t2_image: LST (°C).
        uhi_t1_image / uhi_t2_image: normalized LST (z), needed if response='delta_uhi'.
        response: 'delta_lst' (°C) or 'delta_uhi' (Δ z-score).
        scale: sampling scale (m).
        block_size_m: side of the spatial blocks for the bootstrap.
        iqr_k: rows with any Δ index outside Q1−k·IQR … Q3+k·IQR are dropped
            (same 3×IQR rule as cell 8.1). None disables it.

    Returns:
        dict with 'partition' (see variance_partition), 'ci' (bootstrap
        intervals), the sample arrays and metadata.
    """
    names = [n.upper() for n in index_names]
    bands = []
    for nm in names:
        key = nm.lower()
        if key not in indices_t1_dict or key not in indices_t2_dict:
            raise ValueError(f"Index '{nm}' not available in both periods.")
        bands.append(indices_t2_dict[key].subtract(indices_t1_dict[key]).rename(f'd_{nm}'))

    if response == 'delta_uhi':
        if uhi_t1_image is None or uhi_t2_image is None:
            raise ValueError("response='delta_uhi' requires uhi_t1_image and uhi_t2_image.")
        bands.append(uhi_t2_image.subtract(uhi_t1_image).rename('y'))
        y_label = 'Δ UHI (z)'
    else:
        bands.append(lst_t2_image.subtract(lst_t1_image).rename('y'))
        y_label = 'Δ LST (°C)'

    stack = ee.Image.cat(bands)
    fc = stack.sample(region=roi, scale=scale, numPixels=int(num_samples),
                      seed=seed, geometries=True)
    feats = fc.getInfo()['features']

    cols = [f'd_{nm}' for nm in names] + ['y']
    rows, lon, lat = [], [], []
    for f in feats:
        p = f['properties']
        if all(p.get(c) is not None for c in cols):
            rows.append([p[c] for c in cols])
            lon.append(f['geometry']['coordinates'][0])
            lat.append(f['geometry']['coordinates'][1])
    data = np.array(rows, dtype=float).reshape(-1, len(cols))
    lon, lat = np.array(lon), np.array(lat)

    ok = np.all(np.isfinite(data), axis=1)
    if iqr_k is not None:
        for j in range(len(names)):
            col = data[:, j]
            q1, q3 = np.nanpercentile(col[ok], [25, 75])
            iqr = q3 - q1
            if iqr > 1e-9:
                ok &= (col >= q1 - iqr_k * iqr) & (col <= q3 + iqr_k * iqr)
    data, lon, lat = data[ok], lon[ok], lat[ok]

    X, y = data[:, :-1], data[:, -1]
    if len(y) < 30:
        raise ValueError(f"Too few valid samples ({len(y)}) for a joint regression.")

    part = variance_partition(y, X, names)
    corr = correlation_report(X, y, names, y_name=('ΔUHI (z)' if response == 'delta_uhi' else 'ΔLST'))

    xm, ym = _lonlat_to_m(lon, lat)
    blocks = _block_ids(xm, ym, block_size_m)

    def _stat(idx):
        p = variance_partition(y[idx], X[idx], names)
        out = {'r2_joint': p['r2_joint'], 'shared': p['shared']}
        for nm in names:
            out[f'unique_{nm}'] = p['unique'][nm]
            out[f'r2_{nm}'] = p['r2_individual'][nm]
        return out

    ci = block_bootstrap(_stat, len(y), blocks, n_boot=n_boot, seed=seed)

    return {
        'index_names': names,
        'response': response,
        'y_label': y_label,
        'scale': scale,
        'block_size_m': block_size_m,
        'n_blocks': int(len(np.unique(blocks))),
        'partition': part,
        'correlation': corr,
        'ci': ci,
        'samples': {'X': X, 'y': y, 'lon': lon, 'lat': lat},
    }


def print_joint_regression_summary(res):
    """Plain-text summary of analyze_joint_index_regression."""
    p, ci = res['partition'], res['ci']
    names = res['index_names']

    def _ci(key):
        lo_hi = ci.get(key)
        return f"[{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]" if lo_hi else ''

    print('=' * 78)
    print(f"JOINT EXPLANATORY POWER — {' + '.join('Δ' + n for n in names)} → {res['y_label']}")
    print('=' * 78)
    print(f"n = {p['n']} samples at {res['scale']} m | {res['n_blocks']} spatial blocks "
          f"of {res['block_size_m']} m for the 95% CI")
    print()
    print(f"{'':<26}{'R²':>8}   95% CI (block bootstrap)")
    for nm in names:
        print(f"{'Individual Δ' + nm:<26}{p['r2_individual'][nm]:>8.3f}   {_ci('r2_' + nm)}")
    print(f"{'Sum of individual R²':<26}{p['r2_sum_individual']:>8.3f}   (double-counts the shared part)")
    print(f"{'Joint model':<26}{p['r2_joint']:>8.3f}   {_ci('r2_joint')}")
    print(f"{'Joint model (adjusted)':<26}{p['r2_adj']:>8.3f}")
    print()
    print('Partition of the joint R²:')
    for nm in names:
        print(f"   unique to Δ{nm:<12}{p['unique'][nm]:>8.3f}   {_ci('unique_' + nm)}")
    print(f"   shared{'':<13}{p['shared']:>8.3f}   {_ci('shared')}")
    print()
    print('Collinearity:')
    for nm in names:
        print(f"   VIF Δ{nm:<10}{p['vif'][nm]:>7.2f}    standardized β = {p['beta_std'][nm]:+.3f}")
    if len(names) == 2:
        print(f"   r(Δ{names[0]}, Δ{names[1]}) = {p['predictor_corr'][0, 1]:+.3f}")
    print('   (VIF > 5 indicates strong overlap between predictors)')
    c = res.get('correlation')
    if c:
        print()
        print(f"Correlation with {c['labels'][-1]}:        simple r    partial r (others held constant)")
        for i, nm in enumerate(names):
            print(f"   Δ{nm:<22}{c['pearson'][i, -1]:>+9.3f}    {c['partial_r'][nm]:>+9.3f}")
        print('   A partial r much smaller than the simple r means the index mostly repeats')
        print('   information already carried by the other indices.')


def plot_joint_regression(res, figsize=(9, 4.2)):
    """Horizontal stacked bar: unique parts + shared part = joint R²; dots = individual R²."""
    import matplotlib.pyplot as plt

    p = res['partition']
    names = res['index_names']
    colors = ['#4C78A8', '#B07AA1', '#59A14F', '#E15759', '#F28E2B']

    fig, ax = plt.subplots(figsize=figsize)
    left = 0.0
    for i, nm in enumerate(names):
        w = max(p['unique'][nm], 0)
        ax.barh(0, w, left=left, color=colors[i % len(colors)], label=f'Unique to Δ{nm}')
        ax.text(left + w / 2, 0, f"{w:.2f}", ha='center', va='center', color='white', fontsize=10)
        left += w
    sh = max(p['shared'], 0)
    ax.barh(0, sh, left=left, color='#BAB0AC', label='Shared (overlap)')
    ax.text(left + sh / 2, 0, f"{sh:.2f}", ha='center', va='center', fontsize=10)

    ax.axvline(p['r2_joint'], color='black', lw=1.2)
    ax.text(p['r2_joint'], 0.48, f"joint R² = {p['r2_joint']:.2f}", ha='center', fontsize=9)
    ax.axvline(p['r2_sum_individual'], color='#E15759', lw=1.2, ls='--')
    ax.text(p['r2_sum_individual'], -0.52, f"sum of individual R² = {p['r2_sum_individual']:.2f}",
            ha='center', fontsize=9, color='#E15759')

    ax.set_yticks([])
    ax.set_ylim(-0.75, 0.75)
    ax.set_xlim(0, max(1.0, p['r2_sum_individual'] * 1.1))
    ax.set_xlabel(f"Share of the variance of {res['y_label']} explained")
    ax.set_title('Joint explanatory power of the Δ indices (commonality analysis)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=len(names) + 1, fontsize=9,
              frameon=False)
    fig.tight_layout()
    return fig


def plot_correlation_matrix(res, figsize=(11, 4.4)):
    """Heatmaps of the Pearson and Spearman matrices of [Δ indices, response]."""
    import matplotlib.pyplot as plt

    c = res['correlation']
    labels = c['labels']
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, key, title in ((axes[0], 'pearson', 'Pearson r'), (axes[1], 'spearman', 'Spearman ρ')):
        m = np.asarray(c[key])
        im = ax.imshow(m, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{m[i, j]:+.2f}", ha='center', va='center', fontsize=9,
                        color='white' if abs(m[i, j]) > 0.6 else 'black')
        ax.set_title(title, fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(f"Correlation between the Δ indices and {labels[-1]} (n = {res['partition']['n']})",
                 fontsize=11, fontweight='bold')
    return fig


# ------------------------------------------------------------------------------
# AXIS 2 — grid analysis of new buildings with initial-state controls
# ------------------------------------------------------------------------------

def analyze_new_buildings_grid(
    mask_ee,
    lst_t1_image,
    lst_t2_image,
    roi,
    crs,
    controls_t1=None,
    change_indices=None,
    uhi_t1_image=None,
    uhi_t2_image=None,
    response='dlst',
    cell_size=90,
    min_coverage=0.75,
    built_threshold=0.5,
    none_threshold=0.05,
    stable_quantile=0.5,
    block_cells=3,
    n_boot=1000,
    seed=42,
):
    """
    Did the new buildings change the surface temperature? Estimated per grid cell.

    Each cell (default 90 m, close to the 100 m native TIRS footprint) gets:
      frac_new  — fraction of its area flagged as new building by the U-Net mask
      dlst      — mean Δ LST (T2 − T1);  duhi — mean Δ z (if UHI images given)
      lst_t1    — mean LST in T1 (initial thermal state)
      <ctrl>_t1 — mean of each control index in T1 (e.g. SAVI: initial cover)
      d_<idx>   — mean Δ of each change index (e.g. SAVI, DBSI)

    Three groups of cells (instead of "new buildings vs everything else"):
      built    — frac_new ≥ built_threshold
      changed  — frac_new ≤ none_threshold, but some |Δ index| above the
                 `stable_quantile` of the cells without buildings (e.g. cleared lots)
      stable   — frac_new ≤ none_threshold and every |Δ index| at or below it
    built − stable is the effect of the new buildings against land that did not
    change; changed − stable is the effect of the other changes (clearing,
    vegetation loss). A uniform shift between the dates cancels in both.

    "Was it enough to change the LST?" is answered by magnitude, not only p-values:
      - sufficiency: is the built group's mean outside the stable group's
        2.5–97.5 % range, and what share of built cells exceeds it?
      - dose–response: mean response per class of frac_new, and the regression
        response = b0 + b1·frac_new + b2·lst_t1 + b3·<ctrl>_t1 + …
        (b1 = change of a cell fully covered by new buildings, at the same
        initial state; a uniform shift between dates goes into b0).

    95 % CIs come from a spatial block bootstrap (blocks of block_cells × block_cells).

    Args:
        mask_ee (ee.Image): 0/1 new-building mask (cell 6.5).
        lst_t1_image / lst_t2_image (ee.Image): LST in °C.
        roi (ee.Geometry): analysis area.
        crs (str): projected CRS for the grid (e.g. config_bitemporal['export_crs']).
        controls_t1 (dict): {name: ee.Image} indices in T1 used as initial-state controls.
        change_indices (dict): {name: (img_t1, img_t2)} used to separate stable from
            changed cells. None = no separation (stable = all cells without buildings).
        uhi_t1_image / uhi_t2_image: z-score images; adds 'duhi' per cell.
        response (str): 'dlst' (°C) or 'duhi' (Δ z, needs the UHI images).
        cell_size (int): grid cell size (m).
        min_coverage (float): minimum share of the cell inside the ROI.
        stable_quantile (float): |Δ index| quantile (among cells without buildings)
            below which a cell counts as stable.

    Returns:
        dict with 'cells', 'groups', 'contrasts', 'sufficiency', 'dose_response',
        'model', 'naive', 'stratified', 'ci' and metadata.
    """
    from scipy.stats import mannwhitneyu

    controls_t1 = dict(controls_t1 or {})
    change_indices = dict(change_indices or {})
    ctrl_names = [c.upper() for c in controls_t1]
    chg_names = [c.upper() for c in change_indices]

    proj = ee.Projection(crs).atScale(cell_size)
    grid = roi.coveringGrid(proj)
    proj_m = ee.Projection(crs)  # metric coordinates for the spatial blocks

    def _cell_props(f):
        g = f.geometry()
        cov = g.intersection(roi, 1).area(1).divide(g.area(1))
        xy = g.centroid(1).transform(proj_m, 1).coordinates()
        return f.set({'coverage': cov, 'cx': xy.get(0), 'cy': xy.get(1)})

    cells = grid.map(_cell_props)

    bands = [
        mask_ee.unmask(0).clip(roi).rename('frac_new'),
        lst_t2_image.subtract(lst_t1_image).rename('dlst'),
        lst_t1_image.rename('lst_t1'),
    ]
    for nm, img in controls_t1.items():
        bands.append(img.rename(f'{nm.upper()}_t1'))
    for nm, (img1, img2) in change_indices.items():
        bands.append(img2.subtract(img1).rename(f'd_{nm.upper()}'))
    has_uhi = uhi_t1_image is not None and uhi_t2_image is not None
    if has_uhi:
        bands.append(uhi_t2_image.subtract(uhi_t1_image).rename('duhi'))
    if response == 'duhi' and not has_uhi:
        raise ValueError("response='duhi' requires uhi_t1_image and uhi_t2_image.")
    stack = ee.Image.cat(bands)

    reduced = stack.reduceRegions(collection=cells, reducer=ee.Reducer.mean(),
                                  scale=10, crs=crs, tileScale=4)
    feats = reduced.getInfo()['features']

    keys = (['frac_new', 'dlst', 'lst_t1'] + [f'{c}_t1' for c in ctrl_names]
            + [f'd_{c}' for c in chg_names] + (['duhi'] if has_uhi else []))
    rows = []
    for f in feats:
        pr = f['properties']
        if pr.get('coverage') is None or pr.get('coverage') < min_coverage:
            continue
        if all(pr.get(k) is not None for k in keys):
            rows.append([pr[k] for k in keys] + [pr['cx'], pr['cy']])
    if len(rows) < 20:
        raise ValueError(f"Only {len(rows)} valid cells — enlarge the ROI or reduce cell_size.")
    arr = np.array(rows, dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    cellv = {k: arr[:, i] for i, k in enumerate(keys)}
    cx, cy = arr[:, -2], arr[:, -1]
    y = cellv[response]
    y_label = 'Δ LST (°C)' if response == 'dlst' else 'Δ UHI (z)'

    # ── Groups ───────────────────────────────────────────────────────────────
    built = cellv['frac_new'] >= built_threshold
    none = cellv['frac_new'] <= none_threshold
    stable = none.copy()
    change_thr = {}
    for c in chg_names:
        d = np.abs(cellv[f'd_{c}'])
        thr = float(np.quantile(d[none], stable_quantile)) if none.any() else float('nan')
        change_thr[c] = thr
        stable &= d <= thr
    changed = none & ~stable

    def _summ(m):
        v = y[m]
        return {'n': int(m.sum()),
                'mean': float(np.mean(v)) if len(v) else float('nan'),
                'median': float(np.median(v)) if len(v) else float('nan'),
                'p2_5': float(np.percentile(v, 2.5)) if len(v) else float('nan'),
                'p97_5': float(np.percentile(v, 97.5)) if len(v) else float('nan')}

    groups = {'built': _summ(built), 'changed': _summ(changed), 'stable': _summ(stable)}

    def _contrast(a, b):
        out = {'diff': float(np.mean(y[a]) - np.mean(y[b])) if a.any() and b.any() else float('nan'),
               'cliffs_delta': cliffs_delta(y[a], y[b])}
        out['p_mannwhitney'] = (float(mannwhitneyu(y[a], y[b], alternative='two-sided')[1])
                                if a.sum() > 3 and b.sum() > 3 else float('nan'))
        return out

    contrasts = {'built_vs_stable': _contrast(built, stable),
                 'changed_vs_stable': _contrast(changed, stable) if chg_names else None,
                 'built_vs_changed': _contrast(built, changed) if chg_names else None}

    # ── Sufficiency: is the built group outside the natural variation of stable land? ─
    st = groups['stable']
    if built.any() and stable.sum() >= 10:
        sufficiency = {
            'stable_range': (st['p2_5'], st['p97_5']),
            'built_mean_outside': bool(groups['built']['mean'] > st['p97_5']
                                       or groups['built']['mean'] < st['p2_5']),
            'share_built_above': float(np.mean(y[built] > st['p97_5'])),
            'share_built_below': float(np.mean(y[built] < st['p2_5'])),
        }
    else:
        sufficiency = None

    # ── Dose–response by class of frac_new ──────────────────────────────────
    # Cells that changed WITHOUT buildings (e.g. cleared lots) are left out of the
    # dose–response and of the regression: otherwise the "0 % new" end mixes stable
    # land with cleared land and hides the effect. ΔSAVI is not used as a control,
    # because building itself removes vegetation (it would absorb part of the effect).
    keep = ~changed
    edges = [0.0, none_threshold, 0.25, 0.5, 0.75, 1.0 + 1e-9]
    labels = [f'≤{none_threshold:.0%}', f'{none_threshold:.0%}–25%', '25–50%', '50–75%', '≥75%']
    dose = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (cellv['frac_new'] > lo if lo > 0 else cellv['frac_new'] >= lo) & (cellv['frac_new'] <= hi)
        m &= keep
        v = y[m]
        if len(v) >= 2:
            half = 1.96 * np.std(v, ddof=1) / math.sqrt(len(v))
        else:
            half = float('nan')
        dose.append({'class': lab, 'n': int(m.sum()),
                     'mean': float(np.mean(v)) if len(v) else float('nan'), 'ci_half': half})

    # ── Regression with initial-state controls ──────────────────────────────
    pred_names = ['frac_new', 'lst_t1'] + [f'{c}_t1' for c in ctrl_names]
    X = np.column_stack([cellv[k] for k in pred_names])
    coef, r2, r2_adj = _ols(y[keep], X[keep])
    model = {
        'predictors': pred_names,
        'coef': {nm: float(coef[i + 1]) for i, nm in enumerate(pred_names)},
        'intercept': float(coef[0]),
        'r2': float(r2),
        'r2_adj': float(r2_adj),
        'n_cells': int(keep.sum()),
        'excluded_changed_cells': int(changed.sum()),
    }
    c_simple, r2_simple, _ = _ols(y[keep], cellv['frac_new'][keep])
    model['coef_frac_new_unadjusted'] = float(c_simple[1])
    model['r2_unadjusted'] = float(r2_simple)

    # Naive contrast (built vs every cell without buildings), kept for comparison
    naive = {'n_built': int(built.sum()), 'n_none': int(none.sum()),
             'mean_dlst_built': float(np.mean(y[built])) if built.any() else float('nan'),
             'mean_dlst_none': float(np.mean(y[none])) if none.any() else float('nan')}
    naive['diff'] = naive['mean_dlst_built'] - naive['mean_dlst_none']
    naive['cliffs_delta'] = cliffs_delta(y[built], y[none])
    naive['p_mannwhitney'] = (float(mannwhitneyu(y[built], y[none], alternative='two-sided')[1])
                              if built.sum() > 3 and none.sum() > 3 else float('nan'))

    # Within strata of the initial temperature (terciles of LST_T1): built vs stable
    q = np.nanpercentile(cellv['lst_t1'], [33.333, 66.667])
    strata = np.digitize(cellv['lst_t1'], q)
    strat_rows, w_sum, w_diff = [], 0, 0.0
    for s_, label in enumerate(['cooler third (T1)', 'middle third (T1)', 'hotter third (T1)']):
        m = strata == s_
        b, n_ = m & built, m & stable
        d = float(np.mean(y[b]) - np.mean(y[n_])) if b.any() and n_.any() else float('nan')
        strat_rows.append({'stratum': label, 'n_built': int(b.sum()), 'n_none': int(n_.sum()),
                           'mean_dlst_built': float(np.mean(y[b])) if b.any() else float('nan'),
                           'mean_dlst_none': float(np.mean(y[n_])) if n_.any() else float('nan'),
                           'diff': d})
        if not math.isnan(d):
            w = min(b.sum(), n_.sum())
            w_sum += w
            w_diff += w * d
    stratified = {'rows': strat_rows,
                  'weighted_diff': (w_diff / w_sum) if w_sum else float('nan'),
                  'lst_t1_cuts': [float(q[0]), float(q[1])]}

    blocks = _block_ids(cx, cy, block_cells * cell_size)

    def _stat(idx):
        kk = idx[keep[idx]]
        c, r2b, _ = _ols(y[kk], X[kk])
        out = {'b_frac_new': c[1], 'r2': r2b}
        yi, bi, si, ci_ = y[idx], built[idx], stable[idx], changed[idx]
        if bi.any() and si.any():
            out['built_vs_stable'] = float(np.mean(yi[bi]) - np.mean(yi[si]))
        if ci_.any() and si.any():
            out['changed_vs_stable'] = float(np.mean(yi[ci_]) - np.mean(yi[si]))
        if bi.any() and none[idx].any():
            out['naive_diff'] = float(np.mean(yi[bi]) - np.mean(yi[none[idx]]))
        return out

    ci = block_bootstrap(_stat, len(y), blocks, n_boot=n_boot, seed=seed)

    return {
        'response': response,
        'y_label': y_label,
        'cell_size': cell_size,
        'block_size_m': block_cells * cell_size,
        'n_blocks': int(len(np.unique(blocks))),
        'built_threshold': built_threshold,
        'none_threshold': none_threshold,
        'stable_quantile': stable_quantile,
        'change_thresholds': change_thr,
        'controls': ctrl_names,
        'change_indices': chg_names,
        'cells': cellv,
        # cell centroids in `crs` (metres), for exporting / re-checking the grid
        'cell_xy': {'x': cx, 'y': cy, 'crs': crs},
        'masks': {'built': built, 'changed': changed, 'stable': stable},
        'groups': groups,
        'contrasts': contrasts,
        'sufficiency': sufficiency,
        'dose_response': dose,
        'model': model,
        'naive': naive,
        'stratified': stratified,
        'ci': ci,
    }


def print_new_buildings_grid_summary(res):
    """Plain-text summary of analyze_new_buildings_grid."""
    m, g, ct, ci = res['model'], res['groups'], res['contrasts'], res['ci']
    u = '°C' if res['response'] == 'dlst' else 'z'

    def _ci(key, fmt='+.2f'):
        lo_hi = ci.get(key)
        return f"[{format(lo_hi[0], fmt)}, {format(lo_hi[1], fmt)}]" if lo_hi else ''

    def _p(p):
        return '—' if p != p else f"{p:.4f}"

    print('=' * 78)
    print(f"NEW BUILDINGS × {res['y_label']} — GRID OF {res['cell_size']} m "
          f"({len(res['cells']['frac_new'])} cells, {res['n_blocks']} blocks of {res['block_size_m']} m)")
    print('=' * 78)
    print('1) Three groups of cells:')
    print(f"   built   (≥ {res['built_threshold']:.0%} new buildings)        "
          f"mean {g['built']['mean']:+.2f} {u}  (n={g['built']['n']})")
    if res['change_indices']:
        thr = ', '.join(f"|Δ{k}| > {v:.3f}" for k, v in res['change_thresholds'].items())
        print(f"   changed (≤ {res['none_threshold']:.0%} new, {thr})  "
              f"mean {g['changed']['mean']:+.2f} {u}  (n={g['changed']['n']})")
    print(f"   stable  (≤ {res['none_threshold']:.0%} new"
          + (", small Δ indices)          " if res['change_indices'] else ")                          ")
          + f"mean {g['stable']['mean']:+.2f} {u}  (n={g['stable']['n']})")
    print()
    print('2) Contrasts (difference of means, 95% CI by spatial block bootstrap):')
    rows = [('built − stable', 'built_vs_stable')]
    if res['change_indices']:
        rows += [('changed − stable', 'changed_vs_stable'), ('built − changed', 'built_vs_changed')]
    for lbl, key in rows:
        c = ct.get(key)
        if not c:
            continue
        print(f"   {lbl:<18}{c['diff']:+.2f} {u}  {_ci(key)}   Cliff's δ = {c['cliffs_delta']:+.3f} "
              f"({cliffs_label(c['cliffs_delta'])}), Mann–Whitney p = {_p(c['p_mannwhitney'])}")
    print()
    print('3) Was the change enough to move the LST beyond the variation of stable land?')
    sf = res['sufficiency']
    if sf:
        lo, hi = sf['stable_range']
        print(f"   stable cells, 2.5–97.5 %: [{lo:+.2f}, {hi:+.2f}] {u}")
        print(f"   built mean {'OUTSIDE' if sf['built_mean_outside'] else 'inside'} that range; "
              f"{sf['share_built_above']:.0%} of built cells above it, "
              f"{sf['share_built_below']:.0%} below (≈ 2.5% each expected by chance)")
    else:
        print('   not enough built or stable cells')
    print()
    print('4) Dose–response (mean per class of new-building fraction):')
    for r in res['dose_response']:
        if r['n'] == 0:
            continue
        ci_txt = '' if r['ci_half'] != r['ci_half'] else f" ± {r['ci_half']:.2f}"
        print(f"   {r['class']:<10} n={r['n']:>4}   {r['mean']:+.2f}{ci_txt} {u}")
    print(f"   regression b1 (frac_new) = {m['coef']['frac_new']:+.2f} {u} per cell fully built, "
          f"controlling for {', '.join(m['predictors'][1:])}   95% CI {_ci('b_frac_new')}")
    if m.get('excluded_changed_cells'):
        print(f"   ({m['excluded_changed_cells']} cells that changed without buildings left out "
              f"of the dose–response and the regression)")
    print(f"   without controls: {m['coef_frac_new_unadjusted']:+.2f} {u}  "
          f"(R² {m['r2_unadjusted']:.3f} → {m['r2']:.3f} with controls)")
    print()
    print('5) built − stable within strata of the initial temperature (LST T1):')
    for r in res['stratified']['rows']:
        print(f"   {r['stratum']:<20} built {r['mean_dlst_built']:+.2f} (n={r['n_built']:>3})  "
              f"stable {r['mean_dlst_none']:+.2f} (n={r['n_none']:>3})  Δ = {r['diff']:+.2f}")
    print(f"   weighted mean: {res['stratified']['weighted_diff']:+.2f} {u}")
    print()
    print('Reading: the effect of the new buildings is built − stable. If it stays inside the')
    print('natural variation of stable land and the dose–response is flat, the new buildings')
    print('were not enough to change the surface temperature at the 100 m thermal resolution.')


def plot_new_buildings_grid(res, figsize=(16, 4.8)):
    """(a) three groups vs the stable range; (b) dose–response; (c) built − stable by LST_T1 stratum."""
    import matplotlib.pyplot as plt

    g, u = res['groups'], ('°C' if res['response'] == 'dlst' else 'z')
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # (a) groups
    ax = axes[0]
    order = [('stable', '#4C78A8'), ('changed', '#F28E2B'), ('built', '#E15759')]
    if not res['change_indices']:
        order = [('stable', '#4C78A8'), ('built', '#E15759')]
    data = [res['cells'][res['response']][res['masks'][k]] for k, _ in order]
    bp = ax.boxplot([d if len(d) else np.array([np.nan]) for d in data], patch_artist=True,
                    showfliers=False, widths=0.55)
    for patch, (_, col) in zip(bp['boxes'], order):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    sf = res['sufficiency']
    if sf:
        ax.axhspan(sf['stable_range'][0], sf['stable_range'][1], color='#4C78A8', alpha=0.08,
                   label='stable 2.5–97.5 %')
        ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{k}\n(n={g[k]['n']})" for k, _ in order], fontsize=9)
    ax.set_ylabel(res['y_label'])
    ax.set_title('(a) Cells by group', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # (b) dose–response
    ax = axes[1]
    dr = res['dose_response']
    x = np.arange(len(dr))
    means = [r['mean'] for r in dr]
    errs = [0 if r['ci_half'] != r['ci_half'] else r['ci_half'] for r in dr]
    ax.errorbar(x, means, yerr=errs, fmt='o-', color='#E15759', capsize=4, lw=1.6)
    for i, r in enumerate(dr):
        ax.annotate(f"n={r['n']}", (i, means[i]), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8)
    if g['stable']['n']:
        ax.axhline(g['stable']['mean'], color='#4C78A8', ls='--', lw=1, label='stable mean')
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r['class'] for r in dr], fontsize=9)
    ax.set_xlabel('Fraction of the cell with new buildings')
    ax.set_ylabel(res['y_label'])
    ax.set_title('(b) Dose–response (mean ± 95% CI)', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) built − stable by stratum of LST_T1
    ax = axes[2]
    rows = res['stratified']['rows']
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r['mean_dlst_built'] for r in rows], 0.36, color='#E15759', label='built')
    ax.bar(x + 0.18, [r['mean_dlst_none'] for r in rows], 0.36, color='#4C78A8', label='stable')
    ax.set_xticks(x)
    ax.set_xticklabels([r['stratum'].replace(' (T1)', '\n(LST T1)') for r in rows], fontsize=9)
    ax.set_ylabel(f"Mean {res['y_label']}")
    ax.set_title('(c) Same initial temperature', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(f"Did the new buildings change the surface temperature? (grid of {res['cell_size']} m, "
                 f"{u})", fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig
