# ThermalAnalisys — v0.03

Open-source pipeline (Google Earth Engine + Colab) that integrates deep learning change detection of new buildings (Sentinel-1/2) with bitemporal LST and SUHI analysis (Landsat).

- `Pipeline_A_Zone.ipynb` — main notebook
- `modulos/` — Python modules
- `Validation/` — data, reference map and notebook to reproduce the accuracy of the change detection model and the benchmark against spectral index differencing (see `Validation/README.md`)
- `CHANGELOG.md` — versions

## Outputs for later validation

Each run can be exported for two extents: the **analysis area** (ROI, cell 2.1) and the **urban perimeter**
(thermal reference of the SUHI z-score, cell 2.3).

- **Rasters** (cell 10.2, `export_extent` = ROI / perimeter / both) — Landsat LST, SUHI z-score, intensity classes,
  Landsat bands/indices and their T2 − T1 deltas, clipped to each polygon. Perimeter files end in `_perimeter`;
  Sentinel-1/2 products exist only in the ROI.
- **Validation package** (cell 10.4, `modulos/validation_export.py`) — written to `<project>/Results/<site>/validation_<T1>_<T2>/`
  (plus a `.zip`):
  - `areas/` — ROI and perimeter polygons (GeoJSON, WGS84);
  - `run_metadata.json` — image IDs, dates, CRS, reference mode, thresholds, version;
  - `tables/reference_stats.csv` — μ, σ and °C class limits of the reference per date;
  - `tables/extent_summary.csv`, `class_areas.csv` — LST, SUHI, ΔLST, ΔSUHI statistics and class areas per extent;
  - `tables/pixel_samples_<extent>.csv` — random 30 m pixels with all variables (lon/lat);
  - tables of the analysis cells that were run (5.2, 6.6, 8.1b, 9.1, 9.1b, 9.3).
