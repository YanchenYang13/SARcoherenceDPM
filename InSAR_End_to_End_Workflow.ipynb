{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": "# End-to-End InSAR Sentinel-1 + ISCE Workflow (Consolidated)\n\nThis notebook replaces the original three notebooks:\n1. Input preparation and cropping (former Part 1)\n2. LSTM-based forecasting and score generation (former Part 2)\n3. Geocoded output production (former Part 3)\n\nMethodological principle: **the notebook should remain lightweight** (documentation + orchestration), while all reusable computational logic is implemented in `insar_pipeline/*.py`."
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": "## 1) Path Configuration\nPlease adapt the following paths to your local computational environment."
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": "from pathlib import Path\n\nBASE_INTERFEROGRAM_DIR = Path(\"/data6/WORKDIR/AmatriceSenDT22/merged/interferograms\")\nGEOM_REFERENCE_DIR = Path(\"/data6/WORKDIR/AmatriceSenDT22/merged/geom_reference\")\nNEXT_DATE = \"20160821_20160902\""
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": "## 2) Stage-wise Execution (Recommended for Debugging and Analysis)"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": "from insar_pipeline import (\n    CropConfig, batch_crop_filt_fine_cor,\n    DatasetConfig, build_and_save_dataset,\n    TrainingConfig, run_training_and_prediction,\n    ScoreConfig, compute_and_save_score,\n    OutputConfig, generate_geocoded_outputs,\n)"
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": "## 2.1) Optional: StackSentinel input mode\nIf your input comes directly from ISCE stack pair folders, set `input_source='stack_int'` and choose whether to read ISCE `.cor` or compute coherence from `.int`."
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": "# Example configuration for stackSentinel input\nfrom pathlib import Path\nfrom insar_pipeline import DatasetConfig\n\nSTACK_DATASET_CONFIG = DatasetConfig(\n    cropped_dir=BASE_INTERFEROGRAM_DIR / 'cropped',   # not used in stack_int mode\n    output_dir=BASE_INTERFEROGRAM_DIR / 'cropped',\n    input_source='stack_int',\n    stack_root=BASE_INTERFEROGRAM_DIR,\n    coherence_source='computed_phsig',  # 'isce' | 'computed_phsig' | 'computed_crlb'\n    win=5,\n    looks=25,\n    std_thresh=1.0,\n    use_circular_std=True,\n    persist_computed_cor=False,\n)"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": "# Step A: Crop coherence products and geometry rasters\ncropped_outputs = batch_crop_filt_fine_cor(\n    CropConfig(\n        base_path=BASE_INTERFEROGRAM_DIR,\n        geom_reference_path=GEOM_REFERENCE_DIR,\n        output_base_path=BASE_INTERFEROGRAM_DIR / \"cropped\",\n    )\n)\nprint(f\"Number of cropped files: {len(cropped_outputs)}\")"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": "# Step B: Build training datasets (data.npy / data_std.npy / dates.pkl / geninue*.npy)\ndataset_dir = build_and_save_dataset(\n    DatasetConfig(\n        cropped_dir=BASE_INTERFEROGRAM_DIR / \"cropped\",\n        output_dir=BASE_INTERFEROGRAM_DIR / \"cropped\",\n    )\n)\nprint(\"Dataset directory:\", dataset_dir)"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": "# Step C: Train model and generate prediction\npredict_dir = run_training_and_prediction(\n    TrainingConfig(\n        dataset_dir=dataset_dir,\n        output_dir=BASE_INTERFEROGRAM_DIR / \"cropped\",\n        next_date=NEXT_DATE,\n        epochs=15,\n    )\n)\nprint(\"Prediction directory:\", predict_dir)"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": "# Step D: Compute normalized difference score (score.npy)\nscore_path = compute_and_save_score(\n    ScoreConfig(\n        dataset_dir=dataset_dir,\n        predict_dir=predict_dir,\n    )\n)\nprint(\"Score file:\", score_path)"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": "# Step E: Generate geocoded products and GeoTIFF outputs\noutput_files = generate_geocoded_outputs(\n    OutputConfig(\n        predict_dir=predict_dir,\n        lat_file=BASE_INTERFEROGRAM_DIR / \"cropped\" / \"lat_cropped.rdr\",\n        lon_file=BASE_INTERFEROGRAM_DIR / \"cropped\" / \"lon_cropped.rdr\",\n    )\n)\nprint(output_files)"
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": "## 3) One-Click Execution (Recommended After Validation)"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": "from insar_pipeline import run_full_pipeline\n\nresult = run_full_pipeline(\n    base_dir=BASE_INTERFEROGRAM_DIR,\n    geom_reference_dir=GEOM_REFERENCE_DIR,\n    next_date=NEXT_DATE,\n)\nresult"
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": "## 4) Suggested Extensions\n- Externalize hyperparameters (e.g., epochs, batch size, spatial bounds) into a YAML configuration file.\n- Standardize training logs in `predict/train.log` for experiment traceability.\n- For larger-scale data, introduce sampling/chunking strategies in `modeling.py`."
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.x"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}