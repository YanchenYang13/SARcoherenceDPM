from __future__ import annotations

from pathlib import Path

from .dataset_builder import DatasetConfig, build_and_save_dataset
from .modeling import TrainingConfig, run_training_and_prediction
from .output_products import OutputConfig, generate_geocoded_outputs
from .preprocess import CropConfig, batch_crop_filt_fine_cor
from .scoring import ScoreConfig, compute_and_save_score
from .vit_modeling import ViTConfig, ViTDatasetBuildConfig, build_and_save_vit_matrix_dataset, run_vit_training_and_prediction
from .vit_modeling import ViTConfig, run_vit_training_and_prediction


def run_full_pipeline(
    base_dir: Path,
    geom_reference_dir: Path,
    next_date: str = "20160821_20160902",
    metric: str = "phase_std",
    model_type: str = "lstm",
    use_timestamp: bool = True,
    use_zscore: bool = False,
    timeseries_length: int | None = None,
) -> dict[str, Path]:
    cropped_dir = base_dir / "cropped"

    batch_crop_filt_fine_cor(
        CropConfig(base_path=base_dir, geom_reference_path=geom_reference_dir, output_base_path=cropped_dir)
    )
    dataset_dir = build_and_save_dataset(
        DatasetConfig(cropped_dir=cropped_dir, output_dir=cropped_dir, timeseries_length=timeseries_length)
    )
    predict_dir = run_training_and_prediction(
        TrainingConfig(
            dataset_dir=dataset_dir,
            output_dir=cropped_dir,
            next_date=next_date,
            metric=metric,
            model_type=model_type,
            use_timestamp=use_timestamp,
            use_zscore=use_zscore,
        )
    )
    score_path = compute_and_save_score(
        ScoreConfig(dataset_dir=dataset_dir, predict_dir=predict_dir, metric=metric, use_zscore=use_zscore)
    )

    generate_geocoded_outputs(
        OutputConfig(
            predict_dir=predict_dir,
            lat_file=cropped_dir / "lat_cropped.rdr",
            lon_file=cropped_dir / "lon_cropped.rdr",
        )
    )

    return {
        "cropped_dir": cropped_dir,
        "dataset_dir": dataset_dir,
        "predict_dir": predict_dir,
        "score_path": score_path,
    }


def run_full_vit_pipeline(
    base_dir: Path,
    geom_reference_dir: Path,
    metric: str = "coherence",
    matrix_mode: str = "similarity",
    timeseries_length: int | None = None,
    vit_matrix_size: int | None = None,
) -> dict[str, Path]:
    cropped_dir = base_dir / "cropped"

    batch_crop_filt_fine_cor(
        CropConfig(base_path=base_dir, geom_reference_path=geom_reference_dir, output_base_path=cropped_dir)
    )
    dataset_dir = build_and_save_dataset(
        DatasetConfig(cropped_dir=cropped_dir, output_dir=cropped_dir, timeseries_length=timeseries_length)
    )
    vit_dataset_dir = build_and_save_vit_matrix_dataset(
        ViTDatasetBuildConfig(
            dataset_dir=dataset_dir,
            output_dir=cropped_dir,
            metric=metric,
            matrix_mode=matrix_mode,
            cropped_dir=cropped_dir,
            matrix_size=vit_matrix_size,
        )
    )
    predict_dir = run_vit_training_and_prediction(
        ViTConfig(dataset_dir=vit_dataset_dir, output_dir=cropped_dir, metric=metric, matrix_mode=matrix_mode)
    dataset_dir = build_and_save_dataset(DatasetConfig(cropped_dir=cropped_dir, output_dir=cropped_dir))
    predict_dir = run_vit_training_and_prediction(
        ViTConfig(dataset_dir=dataset_dir, output_dir=cropped_dir, metric=metric, matrix_mode=matrix_mode)
    )
    score_path = compute_and_save_score(ScoreConfig(dataset_dir=dataset_dir, predict_dir=predict_dir, metric=metric))

    generate_geocoded_outputs(
        OutputConfig(
            predict_dir=predict_dir,
            lat_file=cropped_dir / "lat_cropped.rdr",
            lon_file=cropped_dir / "lon_cropped.rdr",
        )
    )

    return {
        "cropped_dir": cropped_dir,
        "dataset_dir": dataset_dir,
        "predict_dir": predict_dir,
        "score_path": score_path,
    }
