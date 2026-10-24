# ==============================================================================
# cd_benchmark.py — Change detection benchmark: U-Net vs spectral index differencing
# ==============================================================================
# Compares the binary map of the Siamese U-Net with the usual low-cost
# alternative, the difference of a spectral index between dates (e.g. ΔNDBI, as
# in the CDI of Silva and Torres, 2021), against the same reference map.
#
# Metrics follow the change-detection convention used by Hafner et al. (2023):
# F1, IoU, precision and recall of the CHANGE class, plus Cohen's kappa and
# overall accuracy. Confidence intervals come from a spatial block bootstrap
# (whole blocks of pixels are resampled, so neighbouring pixels are not treated
# as independent).
# ==============================================================================

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from modulos.stats_analysis import block_bootstrap

# Band order of the Sentinel-2 GeoTIFFs exported by the pipeline (cell 6.0)
S2_BANDS = ('B2', 'B3', 'B4', 'B8', 'B11', 'B12')

# Sign: +1 = the index increases where land becomes built/bare; −1 = it decreases
INDEX_DIRECTION = {'NDBI': +1, 'NBAI': +1, 'DBSI': +1, 'BSI': +1, 'NDVI': -1}


# ── Spectral indices from the exported Sentinel-2 stack ──────────────────────

def s2_index(arr, name):
    """Spectral index from a H×W×6 Sentinel-2 array (bands B2, B3, B4, B8, B11, B12; reflectance 0–1)."""
    arr = arr.astype(np.float64)
    B, G, R, N, S1, S2 = (arr[..., i] for i in range(6))
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (N - R) / (N + R)
        out = {
            'NDBI': (S1 - N) / (S1 + N),                                  # Zha et al. (2003)
            'NBAI': (S2 - S1 / G) / (S2 + S1 / G),                        # Waqar et al. (2012)
            'DBSI': (S1 - G) / (S1 + G) - ndvi,                           # Rasul et al. (2018)
            'BSI': ((S1 + R) - (N + B)) / ((S1 + R) + (N + B)),
            'NDVI': ndvi,
        }[name.upper()]
    return out


def otsu_threshold(values, nbins=256):
    """Otsu (1979) threshold of a 1-D sample (NaNs ignored)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    hist, edges = np.histogram(v, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0, dtype=float), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0, dtype=float), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between))])


# ── Metrics ───────────────────────────────────────────────────────────────────

def binary_change_metrics(pred, ref):
    """
    Metrics of the change class for two 1-D 0/1 arrays of valid pixels.
    Returns TP, FP, TN, FN, precision, recall, F1, IoU, kappa, OA, change share.
    """
    pred = np.asarray(pred).astype(bool)
    ref = np.asarray(ref).astype(bool)
    tp = int(np.sum(pred & ref))
    fp = int(np.sum(pred & ~ref))
    fn = int(np.sum(~pred & ref))
    tn = int(np.sum(~pred & ~ref))
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float('nan')
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float('nan')
    oa = (tp + tn) / n if n else float('nan')
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else float('nan')
    kappa = (oa - pe) / (1 - pe) if n and pe != 1 else float('nan')
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'precision': prec, 'recall': rec,
            'F1': f1, 'IoU': iou, 'kappa': kappa, 'OA': oa,
            'change_share': (tp + fp) / n if n else float('nan')}


# ── Raster helpers ────────────────────────────────────────────────────────────

def _read(path, band=None):
    with rasterio.open(path) as ds:
        arr = ds.read() if band is None else ds.read(band)
        return arr, ds.transform, ds.crs, ds.nodata


def _clean(arr):
    arr = arr.astype(np.float32)
    arr[np.abs(arr) > 1e10] = np.nan          # GEE float32 nodata
    return arr


def align_reference(reference_path, like_transform, like_crs, like_shape, change_values=(1,)):
    """
    Reads a reference (ground-truth) raster and resamples it (nearest neighbour)
    onto the grid of the prediction. Returns float array: 1 change, 0 no change,
    NaN outside the reference.
    """
    with rasterio.open(reference_path) as ds:
        src = ds.read(1).astype(np.float32)
        nod = ds.nodata
        if nod is not None:
            src[src == nod] = np.nan
        dst = np.full(like_shape, np.nan, dtype=np.float32)
        reproject(source=src, destination=dst, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=like_transform, dst_crs=like_crs,
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    out = np.where(np.isnan(dst), np.nan, np.isin(dst, change_values).astype(np.float32))
    return out


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_change_detection(
    folder_imgs,
    roi_name,
    reference_path,
    unet_threshold=45,
    indices=('NDBI', 'NBAI', 'DBSI', 'BSI', 'NDVI'),
    change_values=(1,),
    block_px=32,
    n_boot=1000,
    seed=42,
):
    """
    Compares, on the same reference map, the U-Net change map and the change maps
    obtained by thresholding the difference of spectral indices (ΔIndex = T2 − T1)
    computed from the SAME Sentinel-2 images given to the U-Net.

    Two thresholds are used for each index, so that the comparison does not
    depend on one arbitrary choice:
      - 'Otsu'       : automatic threshold of the ΔIndex distribution (no reference needed,
                       which is how a municipality would use it);
      - 'same area'  : threshold that flags the same share of pixels as the U-Net, so both
                       methods are compared at equal detected area.
    The U-Net is also reported at the threshold with the best F1 ('U-Net best'),
    an optimistic upper bound because it uses the reference to choose the threshold.

    Args:
        folder_imgs (str): folder with sentinel2_{roi}_t1/t2.tif and pred_{roi}.tif (ends with '/').
        roi_name (str): name used in the file names.
        reference_path (str): GeoTIFF of the reference map (any grid; resampled by nearest
            neighbour to the prediction grid). Pixels equal to `change_values` = change.
        unet_threshold (float): U-Net probability threshold (0–100) used in cell 6.4.
        indices (tuple): spectral indices to test.
        block_px (int): side of the bootstrap blocks in pixels (32 px = 320 m at 10 m).
        n_boot (int): bootstrap resamples.

    Returns:
        dict with 'rows' (one per method), 'ci' (95% CI of F1 and of ΔF1 vs U-Net),
        'maps' (2-D arrays for plotting) and metadata.
    """
    prob, tf, crs, _ = _read(f'{folder_imgs}pred_{roi_name}.tif', band=1)
    prob = _clean(prob)
    s2_1 = _clean(_read(f'{folder_imgs}sentinel2_{roi_name}_t1.tif')[0]).transpose(1, 2, 0)
    s2_2 = _clean(_read(f'{folder_imgs}sentinel2_{roi_name}_t2.tif')[0]).transpose(1, 2, 0)
    if s2_1.shape[:2] != prob.shape or s2_2.shape[:2] != prob.shape:
        raise ValueError('Sentinel-2 and prediction rasters have different shapes; '
                         'run the export (6.0) and the inference (6.2) for the same ROI.')
    ref = align_reference(reference_path, tf, crs, prob.shape, change_values)

    valid = (np.isfinite(ref) & np.isfinite(prob)
             & np.all(np.isfinite(s2_1), axis=-1) & np.all(np.isfinite(s2_2), axis=-1)
             & (np.nansum(s2_1, axis=-1) > 0) & (np.nansum(s2_2, axis=-1) > 0))
    if valid.sum() < 100:
        raise ValueError(f'Only {int(valid.sum())} valid pixels overlap the reference map.')

    rr, cc = np.nonzero(valid)
    y_ref = ref[valid].astype(bool)
    p = prob[valid]
    preds = {}

    unet = p >= unet_threshold
    preds[f'U-Net (threshold {unet_threshold:g})'] = unet
    share = unet.mean()

    sweep = {t: binary_change_metrics(p >= t, y_ref)['F1'] for t in range(5, 100, 5)}
    t_best = max(sweep, key=lambda t: (-1 if np.isnan(sweep[t]) else sweep[t]))
    preds[f'U-Net best (threshold {t_best}, optimistic)'] = p >= t_best

    deltas = {}
    for name in indices:
        d = (s2_index(s2_2, name) - s2_index(s2_1, name)) * INDEX_DIRECTION[name.upper()]
        dv = d[valid]
        deltas[name] = d
        t_otsu = otsu_threshold(dv)
        preds[f'Δ{name} (Otsu)'] = dv >= t_otsu
        t_area = float(np.nanquantile(dv, 1 - share)) if 0 < share < 1 else float('nan')
        preds[f'Δ{name} (same area as U-Net)'] = dv >= t_area

    rows = []
    for label, pr in preds.items():
        m = binary_change_metrics(pr, y_ref)
        m['method'] = label
        rows.append(m)

    # Spatial block bootstrap of F1 and of the F1 difference against the U-Net
    blocks = (rr // block_px) * 100000 + (cc // block_px)
    unet_label = next(iter(preds))

    def _stat(idx):
        out = {}
        f_unet = binary_change_metrics(preds[unet_label][idx], y_ref[idx])['F1']
        for label, pr in preds.items():
            f = binary_change_metrics(pr[idx], y_ref[idx])['F1']
            out[f'F1|{label}'] = f
            if not label.startswith('U-Net'):
                out[f'dF1|{label}'] = f_unet - f
        return out

    ci = block_bootstrap(_stat, len(y_ref), blocks, n_boot=n_boot, seed=seed)

    maps = {'reference': np.where(valid, ref, np.nan), 'valid': valid,
            'unet': np.where(valid, prob >= unet_threshold, np.nan)}
    best_idx = max((r for r in rows if r['method'].startswith('Δ')),
                   key=lambda r: (-1 if np.isnan(r['F1']) else r['F1']))
    thr_map = np.full(prob.shape, np.nan)
    thr_map[valid] = preds[best_idx['method']]
    maps['best_index'] = thr_map
    maps['best_index_label'] = best_idx['method']

    return {'rows': rows, 'ci': ci, 'maps': maps, 'unet_label': unet_label,
            'unet_threshold': unet_threshold, 'best_threshold_sweep': sweep,
            'n_valid': int(valid.sum()), 'reference_change_share': float(y_ref.mean()),
            'block_px': block_px, 'n_blocks': int(len(np.unique(blocks))),
            'pixel_size_m': abs(tf.a), 'best_index': best_idx['method']}


def print_change_benchmark(res):
    """Table of the benchmark (metrics of the change class)."""
    print('=' * 104)
    print('CHANGE DETECTION BENCHMARK — U-Net vs spectral index differencing (same reference map)')
    print('=' * 104)
    print(f"valid pixels: {res['n_valid']:,} ({res['pixel_size_m']:.0f} m) | change in reference: "
          f"{res['reference_change_share']:.1%} | 95% CI: spatial block bootstrap, "
          f"{res['n_blocks']} blocks of {res['block_px']}×{res['block_px']} px")
    print(f"\n{'method':<44}{'F1':>7}{'95% CI':>17}{'IoU':>7}{'Prec.':>7}{'Rec.':>7}{'κ':>7}{'area':>7}")
    for r in res['rows']:
        lo, hi = res['ci'].get(f"F1|{r['method']}", (float('nan'), float('nan')))
        print(f"{r['method']:<44}{r['F1']:>7.3f}   [{lo:.3f}, {hi:.3f}]{r['IoU']:>7.3f}"
              f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['kappa']:>7.3f}{r['change_share']:>7.1%}")
    print(f"\nF1 difference, {res['unet_label']} minus each alternative (95% CI):")
    for r in res['rows']:
        key = f"dF1|{r['method']}"
        if key in res['ci']:
            lo, hi = res['ci'][key]
            d = res['rows'][0]['F1'] - r['F1']
            verdict = 'U-Net better' if lo > 0 else ('alternative better' if hi < 0 else 'not distinguishable')
            print(f"   vs {r['method']:<40}{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")
    print("\n'Otsu' needs no reference map (how a municipality would apply it); 'same area' compares the")
    print("methods at an equal detected area; 'U-Net best' uses the reference to pick the threshold (upper bound).")


def plot_change_benchmark(res, figsize=(15, 4.6)):
    """Reference, U-Net and best index-differencing map, with errors coloured (TP / FP / FN)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    ref = res['maps']['reference']
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(np.where(np.isnan(ref), np.nan, ref), cmap=ListedColormap(['#f0f0f0', '#222222']),
                   interpolation='nearest')
    axes[0].set_title('Reference map (change in black)')
    cmap = ListedColormap(['#f0f0f0', '#1a9850', '#d73027', '#4575b4'])
    for ax, key, title in ((axes[1], 'unet', res['unet_label']),
                           (axes[2], 'best_index', res['maps']['best_index_label'])):
        pr = res['maps'][key]
        cat = np.full(ref.shape, np.nan)
        ok = np.isfinite(ref) & np.isfinite(pr)
        r_, p_ = ref[ok].astype(bool), pr[ok].astype(bool)
        cat[ok] = np.select([r_ & p_, ~r_ & p_, r_ & ~p_], [1, 2, 3], 0)
        ax.imshow(cat, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
        f1 = next(r['F1'] for r in res['rows'] if r['method'] == title)
        ax.set_title(f'{title}\nF1 = {f1:.3f}', fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.legend(handles=[Patch(color='#1a9850', label='correct change (TP)'),
                        Patch(color='#d73027', label='false alarm (FP)'),
                        Patch(color='#4575b4', label='missed change (FN)')],
               loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig
