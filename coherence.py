from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path


STEP_CHOICES = [
    "load_data",
    "crop",
    "build_dataset",
    "train_predict",
    "score",
    "output",
    "full",
]


def _as_datetime(date_str: str) -> dt.datetime:
    return dt.datetime.strptime(date_str, "%Y%m%d")


def _build_dataset_config(args: argparse.Namespace):
    from .dataset_builder import DatasetConfig

    return DatasetConfig(
        cropped_dir=args.cropped_dir,
        output_dir=args.output_dir,
        event_date=_as_datetime(args.event_date),
        input_source=args.input_source,
        stack_root=args.stack_root,
        coherence_source=args.coherence_source,
        win=args.win,
        looks=args.looks,
        std_thresh=args.std_thresh,
        use_circular_std=not args.use_linear_std,
        persist_computed_cor=args.persist_computed_cor,
    )


def run_step(args: argparse.Namespace) -> None:
    if args.step == "load_data":
        from .dataset_builder import collect_pair_observations

        cfg = _build_dataset_config(args)
        observations = collect_pair_observations(cfg)
        print(f"[load_data] observation count: {len(observations)}")
        if observations:
            print(f"[load_data] first: {observations[0][1]}, shape={observations[0][2].shape}")
            print(f"[load_data] last : {observations[-1][1]}, shape={observations[-1][2].shape}")
        return

    if args.step == "crop":
        from .preprocess import CropConfig, batch_crop_filt_fine_cor

        outputs = batch_crop_filt_fine_cor(
            CropConfig(
                base_path=args.base_dir,
                geom_reference_path=args.geom_reference_dir,
                output_base_path=args.cropped_dir,
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
            )
        )
        print(f"[crop] cropped file count: {len(outputs)}")
        return

    if args.step == "build_dataset":
        from .dataset_builder import build_and_save_dataset

        dataset_dir = build_and_save_dataset(_build_dataset_config(args))
        print(f"[build_dataset] dataset dir: {dataset_dir}")
        return

    if args.step == "train_predict":
        from .modeling import TrainingConfig, run_training_and_prediction

        dataset_dir = args.dataset_dir or (args.output_dir / "dataset")
        predict_dir = run_training_and_prediction(
            TrainingConfig(
                dataset_dir=dataset_dir,
                output_dir=args.output_dir,
                next_date=args.next_date,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                pred_batch_size=args.pred_batch_size,
                lr=args.lr,
            )
        )
        print(f"[train_predict] predict dir: {predict_dir}")
        return

    if args.step == "score":
        from .scoring import ScoreConfig, compute_and_save_score

        dataset_dir = args.dataset_dir or (args.output_dir / "dataset")
        predict_dir = args.predict_dir or (args.output_dir / "predict")
        score_path = compute_and_save_score(
            ScoreConfig(
                dataset_dir=dataset_dir,
                predict_dir=predict_dir,
                score_filename=args.score_filename,
                chunk_size=args.score_chunk_size,
            )
        )
        print(f"[score] score path: {score_path}")
        return

    if args.step == "output":
        from .output_products import OutputConfig, generate_geocoded_outputs

        predict_dir = args.predict_dir or (args.output_dir / "predict")
        output_files = generate_geocoded_outputs(
            OutputConfig(
                predict_dir=predict_dir,
                lat_file=args.lat_file,
                lon_file=args.lon_file,
                subset_params=args.subset_params,
            )
        )
        print(f"[output] generated files: {output_files}")
        return

    if args.step == "full":
        from .pipeline import run_full_pipeline

        result = run_full_pipeline(
            base_dir=args.base_dir,
            geom_reference_dir=args.geom_reference_dir,
            next_date=args.next_date,
        )
        print(f"[full] result: {result}")
        return

    raise ValueError(f"Unsupported step: {args.step}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="insar-app",
        description=(
            "Unified CLI for InSAR Sentinel-1 + ISCE workflow. "
            "Supports both end-to-end execution and step-wise execution "
            "(data loading, crop, dataset build, training/prediction, scoring, output)."
        ),
    )

    parser.add_argument("--step", choices=STEP_CHOICES, default="full", help="Pipeline step to run.")

    parser.add_argument("--base-dir", type=Path, default=Path("/data6/WORKDIR/AmatriceSenDT22/merged/interferograms"))
    parser.add_argument("--geom-reference-dir", type=Path, default=Path("/data6/WORKDIR/AmatriceSenDT22/merged/geom_reference"))
    parser.add_argument("--cropped-dir", type=Path, default=None, help="Defaults to <base-dir>/cropped")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <base-dir>/cropped")

    parser.add_argument("--dataset-dir", type=Path, default=None, help="Optional override for dataset directory")
    parser.add_argument("--predict-dir", type=Path, default=None, help="Optional override for predict directory")

    parser.add_argument("--event-date", default="20160824", help="Earthquake date in YYYYMMDD")
    parser.add_argument("--next-date", default="20160821_20160902")

    parser.add_argument("--input-source", choices=["cor", "stack_int"], default="cor")
    parser.add_argument("--stack-root", type=Path, default=None)
    parser.add_argument(
        "--coherence-source",
        choices=["isce", "computed_phsig", "computed_crlb"],
        default="isce",
    )
    parser.add_argument("--win", type=int, default=5)
    parser.add_argument("--looks", type=float, default=None)
    parser.add_argument("--std-thresh", type=float, default=1.0)
    parser.add_argument("--use-linear-std", action="store_true", help="Use linear phase std; default is circular std.")
    parser.add_argument("--persist-computed-cor", action="store_true", help="Persist computed coherence as .cor files.")

    parser.add_argument("--lat-min", type=float, default=42.625)
    parser.add_argument("--lat-max", type=float, default=42.635)
    parser.add_argument("--lon-min", type=float, default=13.28)
    parser.add_argument("--lon-max", type=float, default=13.30)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--pred-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--score-filename", default="score.npy")
    parser.add_argument("--score-chunk-size", type=int, default=512)

    parser.add_argument("--lat-file", type=Path, default=None)
    parser.add_argument("--lon-file", type=Path, default=None)
    parser.add_argument("--subset-params", default="-l 42.625 42.635 -L 13.28 13.30")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cropped_dir is None:
        args.cropped_dir = args.base_dir / "cropped"
    if args.output_dir is None:
        args.output_dir = args.base_dir / "cropped"

    if args.lat_file is None:
        args.lat_file = args.cropped_dir / "lat_cropped.rdr"
    if args.lon_file is None:
        args.lon_file = args.cropped_dir / "lon_cropped.rdr"

    run_step(args)


if __name__ == "__main__":
    main()
