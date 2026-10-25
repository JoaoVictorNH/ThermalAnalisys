# Changelog

## Unreleased (after v0.11)

- **MIT License** (`LICENSE`);
- Download link of the pre-trained weights in the README and in cell 6.1.
- README rewritten: what the framework does, quick start in Colab, notebook sections, recommended settings (with the conditions for a seasonal median), outputs, validation results, repository structure, limitations, citation; workflow figure in `docs/`.
- Cell 1.3 / `authenticate_gee`: no default Earth Engine project; each user enters their own (clear error message when empty).
- Notebook: personal Colab execution metadata and widget state removed.

## v0.11 — 2026-10-25

- **Validation folder aligned with the code**: expected results, `results/benchmark_output.txt` and `results/benchmark_maps.png` now come from `cd_benchmark` as run in Colab (cell 6.6 / validation notebook): only the 10 m pixels inside the extent of the reference map are evaluated (77,562 pixels, 4,096 of change, 5.3 %). U-Net F1 0.555 [0.513, 0.582]; best index alternative ΔNBAI (same area) 0.341 [0.250, 0.406]; NBAI post-classification 0.281 on the same pixels. The previous values (81,792 pixels, F1 0.554) counted pixels outside the delineated area as no change.
- Fix (cell 10.4): pixel samples failed because the U-Net mask has no default projection; it is now declared on its 10 m grid before being averaged to 30 m.

## v0.03 — 2026-10-24

Outputs for later validation of the analysis area and of the urban perimeter.

- **Extent of the raster exports** (cell 10.2, `export_bitemporal_results(..., export_extent=)`): analysis area (ROI), urban perimeter (thermal reference) or both. Perimeter files end in `_perimeter`; Sentinel-1/2 products are ROI-only and skipped for the perimeter. Rasters are now clipped to their polygon (before, pixels of the perimeter inside the ROI bounding box could appear in ROI files).
- **Validation package** (cell 10.4, `modulos/validation_export.py`): geometries (GeoJSON), run metadata, reference μ/σ and class limits, per-extent statistics and class areas, pixel samples, and the tables of cells 5.2, 6.6, 8.1b, 9.1, 9.1b and 9.3 as CSV/JSON.
- `analyze_new_buildings_grid` also returns the cell centroids (`cell_xy`, export CRS).
- **Landsat 8 + 9** (cell 3.1, `landsat_type='L8+L9'`): search, selection and median of both sensors merged (more clear scenes for the seasonal median). Landsat 9 IDs (`LC09_…`) are recognised in 3.2 / 3.4.
- T2 now uses its own maximum cloud cover (cell 3.3); before, cell 4.0 applied the T1 value to both periods.
- **Thermal composition** (cell 5.1, `modulos/thermal_composite.py`): `Selected scenes` or `Seasonal median (Landsat only)` — per-pixel median of the clear days of a seasonal window (one mosaic per day), while the change detection and the Sentinel-2 indices stay on the selected scenes. Six comparability checks are printed and stored (same season, same length, window contains the Sentinel-2 date of the change mask, ≥ 3 clear days and ratio ≥ 0.5, same cloud limit, valid days per pixel). Cell 5.2 lists the antecedent rain of every clear day; cell 9.3 uses the same seasonal median; the validation package records the composition and the checks.
- Cells 3.2 / 3.4: the image IDs should stay filled (they feed the validated change detection); empty IDs turn all inputs into medians.
- Fix: with `L8+L9` the image search listed IDs with the merge prefix (`1_LC08_…`, `2_LC09_…`); the prefix is now removed in the listing and ignored when an ID is selected.

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
