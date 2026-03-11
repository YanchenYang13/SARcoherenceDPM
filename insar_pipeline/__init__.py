from .coherence import (
    coh_crlb_from_std,
    coh_isce_phsig_from_std,
    estimate_coherence_from_int,
    phase_std_circular,
    phase_std_linear,
    write_isce_bip_cor,
)
from .dataset_builder import DatasetConfig, build_and_save_dataset, collect_pair_observations
from .isce_stack import StackPairProduct, discover_stack_pair_products, read_isce_int
from .modeling import TrainingConfig, run_training_and_prediction
from .output_products import OutputConfig, generate_geocoded_outputs
from .pipeline import run_full_pipeline
from .preprocess import CropConfig, batch_crop_filt_fine_cor
from .scoring import ScoreConfig, compute_and_save_score


def app_main() -> None:
    """Lazy CLI entrypoint wrapper to avoid runpy re-import warnings."""
    from .app import main

    main()


__all__ = [
    "CropConfig",
    "batch_crop_filt_fine_cor",
    "DatasetConfig",
    "collect_pair_observations",
    "build_and_save_dataset",
    "TrainingConfig",
    "run_training_and_prediction",
    "ScoreConfig",
    "compute_and_save_score",
    "OutputConfig",
    "generate_geocoded_outputs",
    "run_full_pipeline",
    "StackPairProduct",
    "discover_stack_pair_products",
    "read_isce_int",
    "estimate_coherence_from_int",
    "phase_std_linear",
    "phase_std_circular",
    "coh_isce_phsig_from_std",
    "coh_crlb_from_std",
    "write_isce_bip_cor",
    "app_main",
]
