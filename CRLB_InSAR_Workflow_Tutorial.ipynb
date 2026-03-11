# InSAR App CLI Usage Guide

This document describes how to use the unified application entrypoint `insar_pipeline/app.py`.

## 1. Overview
`app.py` encapsulates the complete workflow into a single command-line interface and supports:

- **End-to-end execution** (`--step full`)
- **Step-wise execution** (`--step load_data|crop|build_dataset|train_predict|score|output`)
- **Branch control** for input mode and coherence mode

## 2. Help Command
```bash
python -m insar_pipeline.app -h
```

This prints all parameters, default values, and available step options.

## 3. Core Step Modes

### 3.1 End-to-End
```bash
python -m insar_pipeline.app \
  --step full \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --geom-reference-dir /data6/WORKDIR/AmatriceSenDT22/merged/geom_reference \
  --next-date 20160821_20160902
```

### 3.2 Data Discovery / Loading Check
```bash
python -m insar_pipeline.app \
  --step load_data \
  --input-source stack_int \
  --stack-root /path/to/stack/root \
  --coherence-source computed_phsig
```

### 3.3 Crop
```bash
python -m insar_pipeline.app \
  --step crop \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --geom-reference-dir /data6/WORKDIR/AmatriceSenDT22/merged/geom_reference \
  --lat-min 42.625 --lat-max 42.635 --lon-min 13.28 --lon-max 13.30
```

### 3.4 Build Dataset
```bash
python -m insar_pipeline.app \
  --step build_dataset \
  --input-source stack_int \
  --stack-root /path/to/stack/root \
  --coherence-source computed_crlb \
  --win 5 --looks 25 --event-date 20160824
```

### 3.5 Train and Predict
```bash
python -m insar_pipeline.app \
  --step train_predict \
  --dataset-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/dataset \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
  --next-date 20160821_20160902 \
  --epochs 15
```

### 3.6 Score
```bash
python -m insar_pipeline.app \
  --step score \
  --dataset-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/dataset \
  --predict-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/predict
```

### 3.7 Output Products
```bash
python -m insar_pipeline.app \
  --step output \
  --predict-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/predict \
  --lat-file /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/lat_cropped.rdr \
  --lon-file /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/lon_cropped.rdr
```

## 4. Input Branches

### 4.1 Existing Coherence Branch
- `--input-source cor`
- Reads `.cor` coherence directly.

### 4.2 Stack Interferogram Branch
- `--input-source stack_int`
- Requires `--stack-root`.
- `--coherence-source` options:
  - `isce` (read ISCE-generated `.cor`)
  - `computed_phsig` (self-computed from `.int`)
  - `computed_crlb` (self-computed from `.int`)

## 5. Practical Notes
- `--use-linear-std` switches from default circular phase statistics to linear phase statistics.
- `--persist-computed-cor` writes computed coherence into `.cor` for inspection.
- If `--cropped-dir`/`--output-dir` are omitted, defaults are `<base-dir>/cropped`.
- If `--lat-file`/`--lon-file` are omitted for `output` step, defaults are `<cropped-dir>/lat_cropped.rdr` and `<cropped-dir>/lon_cropped.rdr`.

- For `computed_phsig` / `computed_crlb`, the pipeline now prefers cropped interferograms `<cropped-dir>/<datepair>_filt_fine.int` when available; it falls back to stack `.int` only if cropped files are missing.


## 6. Crop Target Files (Updated)
The crop step currently scans each interferogram pair directory for the following filenames:
- `filt_fine.cor`
- `fine.cor.full`
- `filt_fine.int`

For diagnostics scripts, `insar_pipeline.preprocess.collect_target_files(...)` is now a public helper.
