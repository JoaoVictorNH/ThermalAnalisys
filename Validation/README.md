# Validation of the change detection model

This folder lets anyone reproduce the accuracy of the change detection component (paper, Table 8) and the
comparison with spectral index differencing, without running the whole pipeline.

## How to run

Open `Validation_Change_Detection.ipynb` in Google Colab and run the cells in order. Cell 1 reads the project
from Google Drive or clones this repository. Cells 2–5 only need the files in `data/`; cell 6 (optional) re-runs
the U-Net and needs the pre-trained weights (`mmcr_train100.pt`, Hafner, Ban and Nascetti, 2023).

## Data (`data/`)

| File | Content |
|---|---|
| `sentinel2_Sobral_Belchior_t1.tif`, `sentinel2_Sobral_Belchior_t2.tif` | Sentinel-2 L1C (`COPERNICUS/S2_HARMONIZED`), bands B2, B3, B4, B8, B11, B12 as reflectance, 10 m, EPSG:32724. T1: 2016-09-04 (`20160904T131242_20160904T131242_T24MUB`); T2: 2023-11-12 (`20231112T131239_20231112T131242_T24MUA`) |
| `sentinel1_Sobral_Belchior_t1.tif`, `sentinel1_Sobral_Belchior_t2.tif` | Sentinel-1 VV/VH composites (model input) |
| `pred_Sobral_Belchior.tif` | U-Net output; band 1 = change probability (0–100); pre-trained weights, no fine-tuning |
| `ground_truth.tif` | Reference map of new buildings between T1 and T2, delineated by the authors; band 1: 255 = new building, 0 = no change; ≈1.12 m, EPSG:32724 |
| `mask_gain_NBAI_postclassification.tif` | Built-up gain by post-classification (NBAI > −0.8 in T1 and T2), 255 = gain |

The reference map is resampled (nearest neighbour) to the 10 m grid of the prediction: 81,792 pixels, 4,096 of change (5.0 %).

## Expected results

| Method | F1 [95 % CI] | IoU | Precision | Recall | κ |
|---|---|---|---|---|---|
| U-Net, threshold 50 (model default) | 0.554 [0.514, 0.582] | 0.383 | 0.455 | 0.708 | 0.525 |
| U-Net, threshold 45 (best F1 on this area) | 0.554 [0.515, 0.580] | 0.383 | 0.440 | 0.747 | 0.524 |
| ΔNBAI, same detected area as the U-Net | 0.335 [0.249, 0.402] | 0.201 | 0.275 | 0.428 | 0.292 |
| NBAI post-classification (> −0.8) | 0.277 | 0.161 | 0.214 | 0.394 | 0.227 |
| ΔNBAI, Otsu threshold | 0.253 [0.179, 0.321] | 0.145 | 0.157 | 0.652 | 0.188 |
| ΔNDVI, Otsu threshold | 0.122 [0.079, 0.171] | 0.065 | 0.065 | 0.911 | 0.032 |
| ΔNDBI, Otsu threshold | 0.078 [0.050, 0.110] | 0.041 | 0.042 | 0.646 | −0.018 |

Metrics refer to the change class (Hafner et al., 2023; Corley et al., 2024). The 95 % intervals come from a spatial
block bootstrap (blocks of 32 × 32 pixels = 320 m). The full output is in `results/benchmark_output.txt` and the
maps of correct and wrong detections in `results/benchmark_maps.png`.

Note: the threshold 45 used by default in the pipeline (cell 6.4) was selected on this same area. The model default
(50) gives the same F1, so the reported accuracy does not depend on that choice.

## References

- Hafner, S.; Ban, Y.; Nascetti, A. (2023). Semi-supervised urban change detection using multi-modal Sentinel-1 SAR and Sentinel-2 MSI data. *Remote Sensing*, 15(21), 5135. https://doi.org/10.3390/rs15215135
- Corley, I.; Robinson, C.; Ortiz, A. (2024). A change detection reality check. arXiv:2402.06994.
- Silva, A. G. L.; Torres, M. C. A. (2021). *Building and Environment*, 192, 107634. https://doi.org/10.1016/j.buildenv.2021.107634
