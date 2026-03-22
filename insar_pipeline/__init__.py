__version__ = "0.3.0"

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
from .int_auxiliary import (
    IntAuxiliaryConfig,
    coherence_to_phase_std,
    convert_filtered_coherence_to_std,
    generate_underamp_products,
    generate_unfiltered_phsig_coherence,
    prepare_int_auxiliary_products,
)
from .modeling import TrainingConfig, run_training_and_prediction
from .output_products import OutputConfig, generate_geocoded_outputs
from .pipeline import run_full_pipeline, run_full_vit_pipeline
from .preprocess import CropConfig, batch_crop_filt_fine_cor
from .scoring import ScoreConfig, compute_and_save_score
from .vit_modeling import (
    ViTConfig,
    ViTDatasetBuildConfig,
    build_and_save_vit_matrix_dataset,
    run_vit_training_and_prediction,
)
from .temporal_ccd import CCDBuildConfig, CCDConfig, build_slc_stack_from_cropped, run_temporal_ccd
from .visualization import VisualizationConfig, visualize_file


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
    "run_full_vit_pipeline",
    "StackPairProduct",
    "discover_stack_pair_products",
    "read_isce_int",
    "IntAuxiliaryConfig",
    "generate_unfiltered_phsig_coherence",
    "generate_underamp_products",
    "coherence_to_phase_std",
    "convert_filtered_coherence_to_std",
    "prepare_int_auxiliary_products",
    "estimate_coherence_from_int",
    "phase_std_linear",
    "phase_std_circular",
    "coh_isce_phsig_from_std",
    "coh_crlb_from_std",
    "write_isce_bip_cor",
    "app_main",
    "ViTConfig",
    "ViTDatasetBuildConfig",
    "build_and_save_vit_matrix_dataset",
    "run_vit_training_and_prediction",
    "CCDBuildConfig",
    "CCDConfig",
    "build_slc_stack_from_cropped",
    "run_temporal_ccd",
    "VisualizationConfig",
    "visualize_file",
    "__version__",
]
