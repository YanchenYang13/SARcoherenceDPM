# ISCE Stack Input Integration and Refactor Guide

## 1. Scope of This Upgrade
This upgrade extends the current modular pipeline with two additional capabilities for ISCE `stackSentinel.py` outputs:

1. **Directly read ISCE-generated `.int` / `.cor` products** from stack directory structures and use them in downstream modeling.
2. **Compute coherence without relying on ISCE-generated coherence**, by deriving coherence from interferometric phase statistics using either:
   - ISCE-like phase-sigma threshold mapping (`phsig`), or
   - CRLB-inspired phase-std-to-coherence mapping (`crlb`).

## 2. Newly Added Modules

### 2.1 `insar_pipeline/isce_stack.py`
- Parses stack-like pair folders (e.g., `YYYYMMDD_YYYYMMDD`).
- Discovers available `.int` and `.cor` products per pair.
- Provides `read_isce_int(...)` with XML-aware shape inference.

### 2.2 `insar_pipeline/coherence.py`
- Implements phase standard deviation estimators:
  - linear estimator,
  - circular-statistics estimator.
- Implements two coherence mappings:
  - `coh_isce_phsig_from_std(...)`,
  - `coh_crlb_from_std(...)`.
- Supports writing ISCE BIP-style `.cor` output via `write_isce_bip_cor(...)`.

## 3. Dataset Construction Enhancement
`DatasetConfig` now supports both traditional and stack-integrated flows:

- `input_source="cor"`: use existing `.cor` as before.
- `input_source="stack_int"`: discover stack pairs and build observations from stack products.

For stack-int mode, `coherence_source` controls coherence origin:

- `coherence_source="isce"`: read ISCE-generated `.cor`.
- `coherence_source="computed_phsig"`: compute coherence from `.int` with ISCE-like phase-sigma mapping.
- `coherence_source="computed_crlb"`: compute coherence from `.int` with CRLB mapping.

## 4. Recommended Usage

### 4.1 Read existing ISCE coherence directly
```python
from pathlib import Path
from insar_pipeline import DatasetConfig, build_and_save_dataset

cfg = DatasetConfig(
    cropped_dir=Path('/path/to/cropped_or_pairs'),
    output_dir=Path('/path/to/output'),
    input_source='cor',
)
dataset_dir = build_and_save_dataset(cfg)
```

### 4.2 Use stack directories + ISCE `.cor`
```python
from pathlib import Path
from insar_pipeline import DatasetConfig, build_and_save_dataset

cfg = DatasetConfig(
    cropped_dir=Path('/unused_in_stack_mode'),
    output_dir=Path('/path/to/output'),
    input_source='stack_int',
    stack_root=Path('/path/to/stack/root'),
    coherence_source='isce',
)
dataset_dir = build_and_save_dataset(cfg)
```

### 4.3 Use stack directories + self-computed coherence (`phsig`)
```python
from pathlib import Path
from insar_pipeline import DatasetConfig, build_and_save_dataset

cfg = DatasetConfig(
    cropped_dir=Path('/unused_in_stack_mode'),
    output_dir=Path('/path/to/output'),
    input_source='stack_int',
    stack_root=Path('/path/to/stack/root'),
    coherence_source='computed_phsig',
    win=5,
    looks=25,
    std_thresh=1.0,
    use_circular_std=True,
    persist_computed_cor=True,
)
dataset_dir = build_and_save_dataset(cfg)
```

### 4.4 Use stack directories + self-computed coherence (`crlb`)
```python
from pathlib import Path
from insar_pipeline import DatasetConfig, build_and_save_dataset

cfg = DatasetConfig(
    cropped_dir=Path('/unused_in_stack_mode'),
    output_dir=Path('/path/to/output'),
    input_source='stack_int',
    stack_root=Path('/path/to/stack/root'),
    coherence_source='computed_crlb',
    win=5,
    looks=25,
    use_circular_std=True,
)
dataset_dir = build_and_save_dataset(cfg)
```

## 5. Notes
- In `.cor` reading, coherence band is prioritized (band 2 if present).
- `scipy` is required only when coherence is computed from `.int`.
- Existing modeling/scoring/output modules remain compatible with the generated dataset products.
