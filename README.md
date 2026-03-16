# SARcoherenceDPM: An Open-Source, Research-Oriented Workflow for Rapid InSAR-Based Damage Proxy Mapping

## Overview

**SARcoherenceDPM** is an open-source workflow designed to make post-disaster damage assessment more efficient, reproducible, and extensible using multi-temporal **InSAR** observations.

The repository formalizes a full **Damage Proxy Mapping (DPM)** pipeline into a modular codebase for research and operational experimentation, with emphasis on:

- rapid end-to-end execution from cropped interferometric inputs to geocoded damage products,
- interchangeable modeling paths (RNN and ViT) for temporal prediction under no-disaster assumptions,
- metric-aware scoring strategies for coherence and phase-derived indicators,
- and reusable CLI/programmatic interfaces that reduce notebook-only friction.

At a high level, the workflow:

- builds pixel-wise time series from interferometric products,
- learns temporal patterns with sequence or matrix-based predictors,
- estimates expected post-event behavior under a no-disaster baseline,
- compares predicted and observed post-event signals,
- and exports geocoded outputs for downstream mapping and interpretation.

Compared with earlier notebook-centric workflows, this repository provides a modular Python package, **`insar_pipeline`**, with a step-wise CLI and reusable modules covering the full process:

**crop → dataset → train/predict → score → output**

In addition to the RNN time-series path, the repo now supports a **ViT-based DPM extension** for coherence/phase-std temporal matrix modeling:

**crop → dataset → vit_build_dataset → vit_train_predict → score → output**



---

## Scientific and Methodological Context

The DPM logic implemented in this repository follows the same core idea as the earlier workflow, while introducing improved modularity and implementation flexibility.

### 1. Preprocessing and Interferometric Inputs

- Start from co-registered multi-temporal SAR data and interferometric products.
- Crop region-of-interest products together with geolocation rasters (`lat` / `lon`) to ensure downstream consistency.

### 2. Time-Series Construction

- Assemble chronological per-pixel sequences from interferogram-derived coherence or phase-STD-like signals.
- Preserve temporal ordering and date features required by the prediction model.

### 3. Prediction Baseline (LSTM)

- Train an LSTM-based predictor on pre-event temporal behavior.
- Produce expected post-event values under a **no-disaster baseline**.

### 4. Damage Score Computation

- Compare observed and predicted post-event values.
- Use normalized-difference-style scoring to highlight anomalous changes.

### 5. Geocoding and Export

- Convert prediction and score products into geocoded outputs.
- Subset to target bounding boxes and export final map products such as **GeoTIFF**.

---

## Input and Output Summary

### Input

| Stage | Input Data | Description |
|-------|------------|-------------|
| Interferogram Generation | Registered multi-temporal SAR images | Co-registered Sentinel-1 stack covering pre- and post-event periods |
| Time-Series Construction | Sequential interferograms | Pixel-wise interferogram-derived signals from adjacent date pairs |
| Prediction Model | Pre-event sequences + timestamps | Chronologically arranged values with temporal context |

### Output

| Stage | Output Data | Description |
|-------|-------------|-------------|
| Time-Series Prediction | Hypothetical post-event signal | Predicted no-disaster baseline |
| Damage Mapping | Continuous DPM score | Pixel-wise contrast between predicted and observed post-event values |
| Final Delivery | Geocoded subset products | GIS-ready outputs for interpretation and analysis |

---

## Current Repository Capabilities

The `insar_pipeline` package currently supports the following features.

### Flexible Data Ingestion

- **`input_source='cor'`**  
  Read coherence directly from `.cor` files.

- **`input_source='stack_int'`**  
  Read ISCE stack interferograms (`.int`) and derive coherence-like products.

### Multiple Coherence Paths for Stack Inputs

- **`coherence_source='isce'`**
- **`coherence_source='computed_phsig'`**
- **`coherence_source='computed_crlb'`**

### Step-Wise CLI Execution

Supported execution steps:

- `load_data`
- `crop`
- `build_dataset`
- `train_predict`
- `score`
- `output`
- `full`
- `vit_build_dataset`
- `vit_train_predict`
- `ccd_build_stack`
- `ccd_run`
- `ccd_full`

### End-to-End Orchestration

A programmatic helper is also provided in:

- `insar_pipeline/pipeline.py`

through the high-level function:

- `run_full_pipeline(...)`

### Output Chain

The pipeline generates score products, geocodes them with MintPy tools, subsets them to the target area, and exports final raster outputs.

---

## Package Structure

```text
insar_pipeline/
├── app.py                # CLI entry point and argument parsing
├── pipeline.py           # High-level full workflow orchestrator
├── preprocess.py         # Cropping and target file collection
├── dataset_builder.py    # Observation collection and dataset serialization
├── isce_stack.py         # Stack pair discovery and .int access helpers
├── coherence.py          # Coherence estimation and mapping utilities
├── io_utils.py           # Raster read/write and bbox/index helpers
├── modeling.py           # LSTM dataset/model/training/prediction
├── vit_modeling.py       # ViT temporal-matrix dataset/model/training/prediction
├── scoring.py            # Score computation
└── output_products.py    # Geocoded output generation
````

---

## CLI Usage

Show help information:

```bash
python -m insar_pipeline.app -h
```

### 1. Crop

```bash
python -m insar_pipeline.app --step crop \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --geom-reference-dir /data6/WORKDIR/AmatriceSenDT22/merged/geom_reference
```

Crop targets include `filt_fine.cor`, `fine.cor.full`, `filt_fine.int`, and `slc.full` (when present under `--base-dir`).

For SLC (`*.slc.full`) workflows:
- Ensure each `.slc.full` has been materialized from VRT first, e.g.
  `gdal_translate -of envi 20160821.slc.full.vrt 20160821.slc.full`
- Then set `--base-dir` to your SLC root (e.g. `/data6/WORKDIR/AmatriceSenDT22/merged/SLC`) and run `--step crop`.

### 2. Build Dataset (CRLB Example)

```bash
python -m insar_pipeline.app --step build_dataset \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
  --input-source stack_int \
  --stack-root /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --coherence-source computed_crlb
```

### 3. Train and Predict

```bash
python -m insar_pipeline.app --step train_predict \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped
```

You can externalize model/data hyperparameters via JSON:

```bash
python -m insar_pipeline.app --step train_predict \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
  --param-file configs/model_params.example.json
```

### 3b. Build ViT Temporal-Matrix Dataset

```bash
python -m insar_pipeline.app --step vit_build_dataset \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
  --timeseries-metric coherence \
  --vit-matrix-mode similarity
```

### 3c. ViT Train and Predict

```bash
python -m insar_pipeline.app --step vit_train_predict \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
  --timeseries-metric coherence \
  --vit-matrix-mode similarity \
  --vit-patch-size 2 \
  --vit-depth 4
```

### 4. Score

```bash
python -m insar_pipeline.app --step score \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped
```

### 5. Output (Geocode / Subset / Export)

```bash
python -m insar_pipeline.app --step output \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
  --lat-file /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/lat_cropped.rdr \
  --lon-file /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/lon_cropped.rdr
```

---

## Notebooks in This Repository

### `CRLB_InSAR_Workflow_Tutorial.ipynb`

English tutorial focused on the **CRLB** path, including visualizations for generated artifacts such as `.npy` outputs using `matplotlib`.

### `Part1_Input_Dataset_Construction.ipynb`

Reference notebook for input preparation and dataset construction.

### `Part2_Prediction_DPM_Generation.ipynb`

Reference notebook for prediction and DPM generation.

### `Part3_Output.ipynb`

Reference notebook for geocoding and final output generation.

### `docs/JUPYTER_NOTEBOOK_GUIDE_ZH.md`

中文分步指南，面向在 **Jupyter Notebook** 中执行完整流程（含三条路径：LSTM+zscore、GRU+时间信息+普通score、ViT时间矩阵+zscore）。

---

## Detailed Workflow Notes

This section connects the earlier notebook-style explanation with the current modular implementation.

### A. Data Preparation

* Multi-temporal SAR data are processed into interferometric products, typically through external InSAR stack toolchains such as **ISCE topsStack**.
* This repository then crops the relevant interferometric files and geolocation rasters to the target area of interest.
* Cropped `lat` / `lon` rasters are subsequently used during the geocoding stage.

### B. Time-Series Modeling

The dataset builder serializes intermediate artifacts such as:

* `data.npy`
* `data_std.npy`
* `geninue.npy`
* `geninue_std.npy`
* `dates.pkl`

The modeling module then trains an LSTM baseline and produces:

* `future_predictions.npy`

### C. Damage Proxy Scoring

The scoring module supports two score branches and writes:

#### 1) Normalized-index score (default, `use_zscore=False`)

Let `obs` be the observed post-event map and `pred` be the predicted post-event map.

- **phase_std mode** (`--timeseries-metric phase_std`):

  ```text
  score = (obs - pred) / (obs + pred + eps)
  ```

- **coherence mode** (`--timeseries-metric coherence`):

  ```text
  score = (pred - obs) / (obs + pred + eps)
  ```

  This is the sign-flipped form of the phase-std formula (numerator order swapped).

#### 2) Z-score branch (`use_zscore=True`)

When z-score mode is enabled, training/prediction switches to distribution prediction
(mean + standard deviation), and scoring becomes metric-aware:

```text
phase_std: zscore = (obs - pred_mean) / (pred_std + eps)
coherence: zscore = (pred_mean - obs) / (pred_std + eps)
```

In this branch:

- model training applies metric-specific transform before sequence scaling:
  - coherence: `logit(coherence)`
  - phase_std: `phase_std -> coherence -> logit(coherence)`
- prediction outputs `future_predictions.npy` (mean) and `future_prediction_std.npy` (std),
- method-specific prefixed artifacts are also written (e.g., `rnn_*_future_predictions.npy`, `vit_*_future_predictions.npy`) to avoid cross-method overwrites,
- scoring loads both files and writes metric-consistent z-score values.

Masking policy:

- if either input pixel is NaN, score is NaN,
- if either input pixel is 0, score is set to 0.

Artifacts:

* `score.npy` (or auto-prefixed `rnn_*_score.npy` / `vit_*_score.npy` by default in CLI runs)

Score modes (`--score-mode`) are:
- `direct`: direct difference. coherence uses `pred-obs`; phase_std uses `obs-pred`.
- `ndi`: normalized difference index `(signed diff)/(obs+pred+eps)`.
- `zscore`: `(signed diff)/pred_std`; requires `future_prediction_std.npy`.
- `auto` (default): uses method-specific score if available (e.g., CCD probability map), otherwise `zscore` when `--use-zscore`, else `ndi`.

Optional artifact in z-score branch:

* `future_prediction_std.npy`

---

## New CLI Options for Time-Series and Scoring

The CLI now supports configurable metric/model/time-feature/z-score behavior:

```bash
python -m insar_pipeline.app --step train_predict \
  --timeseries-metric phase_std \
  --ts-model lstm \
  --use-zscore
```

Available options:

- `--timeseries-metric {phase_std,coherence}`: choose sequence variable and score formula family.
- `--ts-model {lstm,gru}`: choose the RNN backbone.
- `--disable-timestamp`: disable date-derived time features from `dates.pkl`.
- `--use-zscore`: enable logit + distribution prediction + z-score scoring.
- `--sequence-length`: limit pre-event adjacent interferometric pairs used to build the time-series window.
- `--matrix-size`: define the number of pre-event acquisition dates used for matrix-pair range filtering.
- `--param-file`: JSON file to set dataset/RNN/ViT hyperparameters in one place (CLI values still override file values when explicitly set).
- `--score-mode {auto,direct,ndi,zscore}`: select score construction rule.

ViT extension options:

- `--vit-matrix-mode {similarity,outer,difference}`: build temporal matrix from per-pixel sequence.
- `--vit-patch-size`: patch size used by ViT patch embedding.
- `--vit-hidden-dim`, `--vit-depth`, `--vit-heads`: ViT backbone size.
- `--vit-diag-mask-ratio`, `--vit-diag-loss-weight`: masked-diagonal self-supervised controls.

### D. Geospatial Output Generation

The output module:

* writes intermediate coherence-like rasters,
* invokes geocoding and subsetting tools,
* and exports final map products.

> **Note**
> This stage depends on a properly configured **MintPy / ISCE** runtime environment.

---

## Typical Artifact Locations

Common output locations include:

* **Cropped files and geometry**:
  `<output_dir>/`

  Examples:

  * `lat_cropped.rdr`
  * `lon_cropped.rdr`

* **Dataset artifacts**:
  `<output_dir>/dataset/`

* **Prediction and score artifacts**:
  `<output_dir>/predict/`

---

## Environment Requirements

Common dependencies include:

* `numpy`
* `matplotlib`
* `torch`
* `GDAL` (`osgeo.gdal`)
* MintPy utilities:

  * `geocode.py`
  * `subset.py`
  * `save_gdal.py`
  * `mintpy.utils.writefile`
* ISCE-compatible inputs for stack-based workflows

> **Note**
> If these tools are unavailable, run the pipeline inside your configured **InSAR / MintPy** environment.

---

## Recommended High-Level Workflow

A typical end-to-end use pattern is:

1. Prepare interferometric products and geometry files.
2. Crop the region of interest.
3. Build the time-series dataset.
4. Train the LSTM predictor and generate post-event baseline predictions.
5. Compute the DPM score.
6. Geocode, subset, and export final GIS-ready products.

---

## Acknowledgement

If this repository is used in operational or publication-oriented workflows, please:

* add an explicit project license,
* acknowledge data sources appropriately,
* and include attribution for upstream toolchains such as **ISCE** and **MintPy**.

---

## Related Notes

This repository is intended to support **time-series-based InSAR damage assessment** workflows where post-event anomalies are interpreted relative to an expected no-disaster temporal baseline. It is particularly useful for studies that aim to move beyond simple two-date comparison and toward temporally informed post-event change interpretation.


### Temporal Decorrelation CCD (Jung et al., 2016)

For SLC-based temporal decorrelation CCD, run:

```bash
python -m insar_pipeline.app --step ccd_build_stack \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/SLC \
  --geom-reference-dir /data6/WORKDIR/AmatriceSenDT22/merged/geom_reference

python -m insar_pipeline.app --step ccd_run \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/SLC/cropped \
  --event-date 20160824 \
  --ccd-max-temporal-baseline 84 \
  --ccd-threshold 0.75
```

Outputs are written into `predict/` as:
- `ccd_temporal_probability.npy`
- `ccd_temporal_change.npy`

> For each `*.slc.full`, ensure ENVI physical file exists first (if only VRT exists):
> `gdal_translate -of envi 20160821.slc.full.vrt 20160821.slc.full`
