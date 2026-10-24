# Changelog

## v0.03 — 2026-10-24

Outputs for later validation of the analysis area and of the urban perimeter.

- **Extent of the raster exports** (cell 10.2, `export_bitemporal_results(..., export_extent=)`): analysis area (ROI), urban perimeter (thermal reference) or both. Perimeter files end in `_perimeter`; Sentinel-1/2 products are ROI-only and skipped for the perimeter. Rasters are now clipped to their polygon (before, pixels of the perimeter inside the ROI bounding box could appear in ROI files).
- **Validation package** (cell 10.4, `modulos/validation_export.py`): geometries (GeoJSON), run metadata, reference μ/σ and class limits, per-extent statistics and class areas, pixel samples, and the tables of cells 5.2, 6.6, 8.1b, 9.1, 9.1b and 9.3 as CSV/JSON.
- `analyze_new_buildings_grid` also returns the cell centroids (`cell_xy`, export CRS).

## v0.02 — 2026-10-24

Evaluation of the framework and robustness of the thermal results.

- **Intra-urban thermal reference** (cell 2.3): the SUHI z-score can use the urban perimeter (water excluded, JRC GSW) instead of the analysis area (Martin et al., 2015).
- **Date context**: antecedent rainfall from CHIRPS (cell 5.2); reference μ/σ and class limits in °C (cell 5.1).
- **Axis 1 — collinearity** (cell 8.1b, `stats_analysis.analyze_joint_index_regression`): joint R², unique and shared variance (commonality analysis), VIF, partial correlation, spatial block bootstrap.
- **Axis 2 — grid analysis** (cell 9.1b, `analyze_new_buildings_grid`): 90 m cells split into built / changed / stable, sufficiency test, dose–response with initial-state controls, Cliff's δ, spatial block bootstrap.
- **Benchmark of the change detection** (cell 6.6, `modulos/cd_benchmark.py`): U-Net vs spectral index differencing (ΔNDBI, ΔNBAI, ΔDBSI, ΔBSI, ΔNDVI) on the same reference map.
- **Robustness to acquisition dates** (cell 9.3, `modulos/robustness.py`): selected pair, anniversary pair and seasonal median (Landsat 8 + 9).
- **`Validation/`**: data, reference map and notebook to reproduce the validation of the change detection model and the benchmark.

## v0.01 — 2026-10-24 (commit f817072)

Clean version validated in Colab: unused functions and the LST downscaling removed, pipeline errors fixed
(sampling alignment, cloud masking, Sentinel-1/2 grid, export CRS, Landsat 9 preview).

## Baseline (branch `baseline-original`)

Code as it was before the review.
