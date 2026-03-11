"""Top-level package exports for SARcoherenceDPM.

This module intentionally avoids eager imports so that CLI entrypoints such as
`python -m insar_pipeline.app` can start even when optional runtime dependencies
(e.g., NumPy/SciPy/Torch) are not yet available, and without forcing unrelated
submodules to be parsed at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MAP = {
    "CropConfig": ("insar_pipeline.preprocess", "CropConfig"),
    "batch_crop_filt_fine_cor": ("insar_pipeline.preprocess", "batch_crop_filt_fine_cor"),
    "DatasetConfig": ("insar_pipeline.dataset_builder", "DatasetConfig"),
    "collect_pair_observations": ("insar_pipeline.dataset_builder", "collect_pair_observations"),
    "build_and_save_dataset": ("insar_pipeline.dataset_builder", "build_and_save_dataset"),
    "TrainingConfig": ("insar_pipeline.modeling", "TrainingConfig"),
    "run_training_and_prediction": ("insar_pipeline.modeling", "run_training_and_prediction"),
    "ScoreConfig": ("insar_pipeline.scoring", "ScoreConfig"),
    "compute_and_save_score": ("insar_pipeline.scoring", "compute_and_save_score"),
    "OutputConfig": ("insar_pipeline.output_products", "OutputConfig"),
    "generate_geocoded_outputs": ("insar_pipeline.output_products", "generate_geocoded_outputs"),
    "run_full_pipeline": ("insar_pipeline.pipeline", "run_full_pipeline"),
    "run_full_vit_pipeline": ("insar_pipeline.pipeline", "run_full_vit_pipeline"),
    "StackPairProduct": ("insar_pipeline.isce_stack", "StackPairProduct"),
    "discover_stack_pair_products": ("insar_pipeline.isce_stack", "discover_stack_pair_products"),
    "read_isce_int": ("insar_pipeline.isce_stack", "read_isce_int"),
    "estimate_coherence_from_int": ("insar_pipeline.coherence", "estimate_coherence_from_int"),
    "phase_std_linear": ("insar_pipeline.coherence", "phase_std_linear"),
    "phase_std_circular": ("insar_pipeline.coherence", "phase_std_circular"),
    "coh_isce_phsig_from_std": ("insar_pipeline.coherence", "coh_isce_phsig_from_std"),
    "coh_crlb_from_std": ("insar_pipeline.coherence", "coh_crlb_from_std"),
    "write_isce_bip_cor": ("insar_pipeline.coherence", "write_isce_bip_cor"),
    "ViTConfig": ("insar_pipeline.vit_modeling", "ViTConfig"),
    "ViTDatasetBuildConfig": ("insar_pipeline.vit_modeling", "ViTDatasetBuildConfig"),
    "build_and_save_vit_matrix_dataset": ("insar_pipeline.vit_modeling", "build_and_save_vit_matrix_dataset"),
    "run_vit_training_and_prediction": ("insar_pipeline.vit_modeling", "run_vit_training_and_prediction"),
}


__all__ = sorted(list(_EXPORT_MAP.keys()) + ["app_main"])


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'insar_pipeline' has no attribute '{name}'")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


def app_main() -> None:
    """Lazy CLI entrypoint wrapper to avoid runpy re-import warnings."""
    from .app import main

    main()
