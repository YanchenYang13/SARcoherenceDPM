# SARcoherenceDPM Full-Feature Jupyter Notebook Guide (English)

This guide is designed to test and demonstrate almost all implemented repository features in one notebook workflow.

## Coverage

The notebook commands below cover:

- `load_data`, `crop`, `build_dataset`, `train_predict`, `score`, `output`, `visualize`
- `full`, `vit_build_dataset`, `vit_train_predict`, `vit_full`
- `ccd_build_stack`, `ccd_run`, `ccd_full`

---

## 1) Initialization

### Cell 1: Path setup

```python
from pathlib import Path

BASE_DIR = Path('/data6/WORKDIR/AmatriceSenDT22/merged/interferograms')
GEOM_REF_DIR = Path('/data6/WORKDIR/AmatriceSenDT22/merged/geom_reference')
CROPPED_DIR = BASE_DIR / 'cropped'

SLC_BASE_DIR = Path('/data6/WORKDIR/AmatriceSenDT22/merged/SLC')
SLC_CROPPED_DIR = SLC_BASE_DIR / 'cropped'

EVENT_DATE = '20160824'
NEXT_DATE = '20160821_20160902'
```

### Cell 2: Command helper

```python
import subprocess

def run_cmd(cmd: str, check: bool = True):
    print('>>>', cmd)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        if check:
            raise RuntimeError(f'Command failed: {cmd}')
    return p
```

---

## 2) Stepwise RNN workflow

### Cell 3: `load_data`

```python
run_cmd(
    f"python -m insar_pipeline.app --step load_data "
    f"--base-dir {BASE_DIR} --output-dir {CROPPED_DIR} --event-date {EVENT_DATE}"
)
```

### Cell 4: `crop`

```python
run_cmd(
    f"python -m insar_pipeline.app --step crop "
    f"--base-dir {BASE_DIR} --geom-reference-dir {GEOM_REF_DIR} --output-dir {CROPPED_DIR}"
)
```

### Cell 5: `build_dataset` (cor)

```python
run_cmd(
    f"python -m insar_pipeline.app --step build_dataset "
    f"--base-dir {BASE_DIR} --output-dir {CROPPED_DIR} --event-date {EVENT_DATE} --input-source cor"
)
```

### Cell 6: `train_predict` (LSTM + zscore)

```python
run_cmd(
    f"python -m insar_pipeline.app --step train_predict "
    f"--output-dir {CROPPED_DIR} --event-date {EVENT_DATE} --next-date {NEXT_DATE} "
    f"--timeseries-metric coherence --ts-model lstm --use-zscore"
)
```

### Cell 7: `score`

```python
run_cmd(
    f"python -m insar_pipeline.app --step score "
    f"--output-dir {CROPPED_DIR} --score-mode auto --use-zscore"
)
```

### Cell 8: `output`

```python
run_cmd(
    f"python -m insar_pipeline.app --step output "
    f"--output-dir {CROPPED_DIR} "
    f"--lat-file {CROPPED_DIR / 'lat_cropped.rdr'} --lon-file {CROPPED_DIR / 'lon_cropped.rdr'}"
)
```

### Cell 9: `output` + threshold masks (3 methods)

```python
# Manual threshold
run_cmd(
    f"python -m insar_pipeline.app --step output "
    f"--output-dir {CROPPED_DIR} --lat-file {CROPPED_DIR / 'lat_cropped.rdr'} "
    f"--lon-file {CROPPED_DIR / 'lon_cropped.rdr'} "
    f"--mask-enable --mask-method manual --mask-threshold-manual 0.30 --mask-output-suffix manual_mask"
)

# Quantile threshold (default 0.70)
run_cmd(
    f"python -m insar_pipeline.app --step output "
    f"--output-dir {CROPPED_DIR} --lat-file {CROPPED_DIR / 'lat_cropped.rdr'} "
    f"--lon-file {CROPPED_DIR / 'lon_cropped.rdr'} "
    f"--mask-enable --mask-method quantile --mask-quantile 0.70 --mask-output-suffix q70_mask"
)

# Mean + n*std threshold (default n=2)
run_cmd(
    f"python -m insar_pipeline.app --step output "
    f"--output-dir {CROPPED_DIR} --lat-file {CROPPED_DIR / 'lat_cropped.rdr'} "
    f"--lon-file {CROPPED_DIR / 'lon_cropped.rdr'} "
    f"--mask-enable --mask-method std --mask-std-n 2.0 --mask-output-suffix std2_mask"
)
```

### Cell 10: `visualize`

```python
run_cmd(
    f"python -m insar_pipeline.app --step visualize "
    f"--output-dir {CROPPED_DIR} --visualize-mode matplotlib "
    f"--visualize-input {CROPPED_DIR / 'predict' / 'score.npy'} "
    f"--visualize-output {CROPPED_DIR / 'predict' / 'score_matplotlib.png'} --visualize-nodisplay"
)
```

---

## 3) ViT path

```python
run_cmd(
    f"python -m insar_pipeline.app --step vit_build_dataset "
    f"--output-dir {CROPPED_DIR} --event-date {EVENT_DATE} "
    f"--timeseries-metric coherence --vit-matrix-mode similarity"
)

run_cmd(
    f"python -m insar_pipeline.app --step vit_train_predict "
    f"--output-dir {CROPPED_DIR} --event-date {EVENT_DATE} --next-date {NEXT_DATE} "
    f"--timeseries-metric coherence --vit-matrix-mode similarity --vit-patch-size 2 --vit-depth 4 --use-zscore"
)
```

---

## 4) CCD path

```python
run_cmd(
    f"python -m insar_pipeline.app --step ccd_build_stack "
    f"--base-dir {SLC_BASE_DIR} --geom-reference-dir {GEOM_REF_DIR} --output-dir {SLC_CROPPED_DIR}"
)

run_cmd(
    f"python -m insar_pipeline.app --step ccd_run "
    f"--output-dir {SLC_CROPPED_DIR} --event-date {EVENT_DATE} "
    f"--ccd-max-temporal-baseline 84 --ccd-threshold 0.75"
)
```

---

## 5) One-command pipelines

```python
run_cmd(
    f"python -m insar_pipeline.app --step full "
    f"--base-dir {BASE_DIR} --geom-reference-dir {GEOM_REF_DIR} "
    f"--output-dir {CROPPED_DIR} --event-date {EVENT_DATE} --next-date {NEXT_DATE}"
)

run_cmd(
    f"python -m insar_pipeline.app --step vit_full "
    f"--base-dir {BASE_DIR} --geom-reference-dir {GEOM_REF_DIR} --output-dir {CROPPED_DIR} "
    f"--event-date {EVENT_DATE} --next-date {NEXT_DATE} --timeseries-metric coherence --vit-matrix-mode similarity"
)

run_cmd(
    f"python -m insar_pipeline.app --step ccd_full "
    f"--base-dir {SLC_BASE_DIR} --geom-reference-dir {GEOM_REF_DIR} "
    f"--output-dir {SLC_CROPPED_DIR} --event-date {EVENT_DATE}"
)
```
