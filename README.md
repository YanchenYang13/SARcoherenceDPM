# Damage Proxy Mapping with Time-Series Prediction of InSAR Phase Standard Deviation

## Overview

This project generates **Damage Proxy Maps (DPMs)** from multi-temporal **InSAR** observations by modeling phase-derived time-series behavior and highlighting post-event anomalies.

At a high level, the workflow:

- builds pixel-wise time series from interferometric products,
- learns temporal patterns with an LSTM-based predictor,
- estimates expected post-event behavior under a no-disaster baseline,
- compares predicted and observed post-event signals,
- and exports geocoded outputs for downstream mapping and interpretation.

Compared with earlier notebook-centric workflows, this repository now provides a modular Python package, **`insar_pipeline`**, with a step-wise CLI and reusable modules covering the full process:

**crop → dataset → train/predict → score → output**

<img width="4663" height="4000" alt="2" src="https://github.com/user-attachments/assets/57fdef2f-83dd-42f2-a39d-9332afc91bd9" />


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

The scoring module computes a normalized contrast between observed and predicted post-event maps and writes:

* `score.npy`

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


