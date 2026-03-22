# SARcoherenceDPM

SARcoherenceDPM is an open-source InSAR Damage Proxy Mapping (DPM) workflow for research and operational prototyping.  
It provides a modular Python package + CLI pipeline that supports **RNN**, **ViT**, and **temporal CCD** paths.

## Language Versions

- English project overview (this file): `README.md`
- Chinese project overview: `README_ZH.md`
- Chinese notebook operation guide: `docs/JUPYTER_NOTEBOOK_GUIDE_ZH.md`
- English notebook operation guide: `docs/JUPYTER_NOTEBOOK_GUIDE_EN.md`

---

## 1) Core Goals

The project standardizes the end-to-end post-event InSAR DPM workflow:

1. Build per-pixel temporal signals from interferometric products.
2. Learn no-disaster temporal behavior with sequence/matrix models.
3. Compare observed post-event signals against predictions.
4. Export geocoded products for GIS analysis.

---

## 2) Feature Matrix

| Capability | Description | CLI Step |
|---|---|---|
| Data cropping | Crop interferometric products and geolocation rasters | `crop` |
| Dataset building | Build temporal datasets from `.cor` or ISCE `stack_int` | `build_dataset` |
| INT auxiliaries | Generate `unfilt_fine.cor`, under-amplitude products, and `filt_fine.std` from cropped INT/COR files | `prepare_int_aux` |
| RNN training/inference | LSTM/GRU with optional z-score branch | `train_predict` |
| Score generation | `auto`, `direct`, `ndi`, `zscore` scoring modes | `score` |
| Geocoded outputs | Geocode + subset + GeoTIFF export | `output` |
| Post-output mask | Threshold masking on final TIFF products | `output --mask-enable` |
| Visualization | Matplotlib or MintPy-style visualization | `visualize` |
| ViT matrix path | ViT matrix dataset + train/predict | `vit_build_dataset`, `vit_train_predict` |
| Full pipelines | One-command orchestration | `full`, `vit_full`, `ccd_full` |
| Temporal CCD | Jung et al. (2016)-style temporal decorrelation CCD | `ccd_build_stack`, `ccd_run` |

---

## 3) Package Structure

```text
insar_pipeline/
├── app.py                # CLI parser and step dispatcher
├── pipeline.py           # High-level orchestration
├── preprocess.py         # Cropping and file discovery
├── dataset_builder.py    # Temporal dataset construction
├── modeling.py           # RNN modeling (LSTM/GRU)
├── vit_modeling.py       # ViT matrix modeling
├── scoring.py            # Score computation
├── output_products.py    # Geocoding/export + threshold masking
├── temporal_ccd.py       # Temporal decorrelation CCD
└── visualization.py      # Matplotlib/MintPy visualization helper
```

---

## 4) Main CLI Steps

```bash
python -m insar_pipeline.app --step <STEP> [args...]
```

Supported steps:

- `load_data`, `crop`, `prepare_int_aux`, `build_dataset`, `train_predict`, `score`, `output`, `visualize`
- `full`, `vit_build_dataset`, `vit_train_predict`, `vit_full`
- `ccd_build_stack`, `ccd_run`, `ccd_full`

---

## 5) Threshold Masking (New)

Threshold masking is applied to generated final TIFF products (and optional additional TIFF files):

```bash
python -m insar_pipeline.app --step output \
  --output-dir /data/.../cropped \
  --lat-file /data/.../cropped/lat_cropped.rdr \
  --lon-file /data/.../cropped/lon_cropped.rdr \
  --mask-enable \
  --mask-method quantile \
  --mask-quantile 0.70 \
  --mask-output-suffix q70mask
```

Three threshold methods are supported:

1. `manual`: `--mask-threshold-manual <value>`
2. `quantile`: `--mask-quantile` (default `0.70`)
3. `std`: threshold = `mean + n * std`, with `--mask-std-n` (default `2.0`)

You can include extra TIFF files via `--mask-input-tif file1.tif file2.tif`.

---

## 6) Paper Experiment Helpers

The repository now includes an INT-derived auxiliary preparation step for paper experiments:

```bash
python -m insar_pipeline.app --step prepare_int_aux \
  --cropped-dir /data/.../cropped \
  --aux-variance-looks 3.0
```

This step prepares:

- `*_unfilt_fine.cor`
- `*_underamp_unfilt_fine.cor`
- `*_underamp_unfilt_fine_circ.cor`
- `*_filt_fine.std`

You can then build source-specific datasets with:

```bash
python -m insar_pipeline.app --step build_dataset \
  --cropped-dir /data/.../cropped \
  --output-dir /data/.../cropped \
  --input-source cor \
  --observation-file underamp_unfilt_fine.cor \
  --dataset-name dataset_rnn_underamp_unfilt_fine_cor
```

For the requested Amatrice paper workflow, use the dedicated notebook:

- `docs/AMATRICE_PAPER_EXPERIMENT_GUIDE.ipynb`

---

## 7) Requirements

Typical runtime dependencies:

- `numpy`, `torch`, `matplotlib`
- `osgeo.gdal`
- MintPy tools (`geocode.py`, `subset.py`, `save_gdal.py`, `view.py`)

Run this project in a configured InSAR/MintPy environment for full functionality.
