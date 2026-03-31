from __future__ import annotations

from pathlib import Path

from .dataset_builder import DatasetConfig, build_and_save_dataset
from .modeling import TrainingConfig, run_training_and_prediction
from .output_products import OutputConfig, generate_geocoded_outputs
from .preprocess import CropConfig, batch_crop_filt_fine_cor
from .scoring import ScoreConfig, compute_and_save_score
from .vit_modeling import ViTConfig, ViTDatasetBuildConfig, build_and_save_vit_matrix_dataset, run_vit_training_and_prediction


def _rnn_prefix(metric: str, model_type: str, use_zscore: bool, use_timestamp: bool) -> str:
    ztag = "zscore" if use_zscore else "raw"
    ttag = "time" if use_timestamp else "notime"
    return f"rnn_{model_type}_{metric}_{ztag}_{ttag}"


def _vit_prefix(metric: str, matrix_mode: str, use_zscore: bool, patch: int, depth: int) -> str:
    ztag = "zscore" if use_zscore else "raw"
    return f"vit_{matrix_mode}_{metric}_{ztag}_p{patch}_d{depth}"


def run_full_pipeline(
    base_dir: Path,
    geom_reference_dir: Path,
    next_date: str = "20160821_20160902",
    metric: str = "phase_std",
    model_type: str = "stepwise_gru",
    use_timestamp: bool = True,
    use_zscore: bool = False,
    sequence_length: int | None = None,
    matrix_size: int | None = None,
    rnn_hidden_dim: int = 64,
    rnn_num_layers: int = 2,
    rnn_dropout: float = 0.1,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    max_grad_norm: float | None = None,
    scaler_type: str = "robust",
) -> dict[str, Path]:
    cropped_dir = base_dir / "cropped"

    batch_crop_filt_fine_cor(
        CropConfig(base_path=base_dir, geom_reference_path=geom_reference_dir, output_base_path=cropped_dir)
    )
    dataset_dir = build_and_save_dataset(
        DatasetConfig(
            cropped_dir=cropped_dir,
            output_dir=cropped_dir,
            sequence_length=sequence_length,
            matrix_size=matrix_size,
        )
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
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
            optimizer=optimizer,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            scaler_type=scaler_type,
            artifact_prefix=_rnn_prefix(metric, model_type, use_zscore, use_timestamp),
        )
    )
    score_path = compute_and_save_score(
        ScoreConfig(dataset_dir=dataset_dir, predict_dir=predict_dir, metric=metric, use_zscore=use_zscore, score_filename=f"{_rnn_prefix(metric, model_type, use_zscore, use_timestamp)}_score.npy", artifact_prefix=_rnn_prefix(metric, model_type, use_zscore, use_timestamp))
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
    sequence_length: int | None = None,
    matrix_size: int | None = None,
    use_zscore: bool = False,
    vit_patch_size: int = 2,
    vit_hidden_dim: int = 64,
    vit_depth: int = 4,
    vit_heads: int = 4,
    vit_diag_mask_ratio: float = 0.5,
    vit_diag_loss_weight: float = 0.3,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
) -> dict[str, Path]:
    cropped_dir = base_dir / "cropped"

    batch_crop_filt_fine_cor(
        CropConfig(base_path=base_dir, geom_reference_path=geom_reference_dir, output_base_path=cropped_dir)
    )
    dataset_dir = build_and_save_dataset(
        DatasetConfig(
            cropped_dir=cropped_dir,
            output_dir=cropped_dir,
            sequence_length=sequence_length,
            matrix_size=matrix_size,
        )
    )
    vit_dataset_dir = build_and_save_vit_matrix_dataset(
        ViTDatasetBuildConfig(dataset_dir=dataset_dir, output_dir=cropped_dir, metric=metric, matrix_mode=matrix_mode)
    )
    predict_dir = run_vit_training_and_prediction(
        ViTConfig(
            dataset_dir=vit_dataset_dir,
            output_dir=cropped_dir,
            metric=metric,
            matrix_mode=matrix_mode,
            patch_size=vit_patch_size,
            hidden_dim=vit_hidden_dim,
            depth=vit_depth,
            heads=vit_heads,
            diag_mask_ratio=vit_diag_mask_ratio,
            diag_loss_weight=vit_diag_loss_weight,
            use_zscore=use_zscore,
            optimizer=optimizer,
            weight_decay=weight_decay,
            artifact_prefix=_vit_prefix(metric, matrix_mode, use_zscore, vit_patch_size, vit_depth),
        )
    )
    score_path = compute_and_save_score(ScoreConfig(dataset_dir=vit_dataset_dir, predict_dir=predict_dir, metric=metric, use_zscore=use_zscore, score_filename=f"{_vit_prefix(metric, matrix_mode, use_zscore, vit_patch_size, vit_depth)}_score.npy", artifact_prefix=_vit_prefix(metric, matrix_mode, use_zscore, vit_patch_size, vit_depth)))

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
        "vit_dataset_dir": vit_dataset_dir,
        "predict_dir": predict_dir,
        "score_path": score_path,
    }
