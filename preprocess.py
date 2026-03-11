from __future__ import annotations

from pathlib import Path

from .dataset_builder import DatasetConfig, build_and_save_dataset
from .modeling import TrainingConfig, run_training_and_prediction
from .output_products import OutputConfig, generate_geocoded_outputs
from .preprocess import CropConfig, batch_crop_filt_fine_cor
from .scoring import ScoreConfig, compute_and_save_score


def run_full_pipeline(base_dir: Path, geom_reference_dir: Path, next_date: str = "20160821_20160902") -> dict[str, Path]:
    cropped_dir = base_dir / "cropped"

    batch_crop_filt_fine_cor(
        CropConfig(base_path=base_dir, geom_reference_path=geom_reference_dir, output_base_path=cropped_dir)
    )
    dataset_dir = build_and_save_dataset(DatasetConfig(cropped_dir=cropped_dir, output_dir=cropped_dir))
    predict_dir = run_training_and_prediction(
        TrainingConfig(dataset_dir=dataset_dir, output_dir=cropped_dir, next_date=next_date)
    )
    score_path = compute_and_save_score(ScoreConfig(dataset_dir=dataset_dir, predict_dir=predict_dir))

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
