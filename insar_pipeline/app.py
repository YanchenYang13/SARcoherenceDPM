from __future__ import annotations

import argparse
import json
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
    "vit_build_dataset",
    "vit_train_predict",
    "vit_full",
    "ccd_build_stack",
    "ccd_run",
    "ccd_full",
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
        sequence_length=args.sequence_length,
        matrix_size=args.matrix_size,
    )


def _load_param_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--param-file must point to a JSON object")
    return data


def _apply_param_overrides(args: argparse.Namespace, defaults: argparse.Namespace) -> None:
    if args.param_file is None:
        return

    params = _load_param_file(args.param_file)
    merged: dict = {}
    for section in ("global", "dataset", "rnn", "vit"):
        sec = params.get(section, {})
        if isinstance(sec, dict):
            merged.update(sec)

    for key, value in merged.items():
        arg_key = key.replace("-", "_")
        if not hasattr(args, arg_key):
            continue
        if getattr(args, arg_key) == getattr(defaults, arg_key):
            setattr(args, arg_key, value)


def _section(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n[STEP] {title}\n{line}")


def _kv(key: str, value) -> None:
    print(f"  - {key}: {value}")


def _artifact_prefix_for_rnn(args: argparse.Namespace) -> str:
    ztag = "zscore" if args.use_zscore else "raw"
    ttag = "time" if not args.disable_timestamp else "notime"
    return f"rnn_{args.ts_model}_{args.timeseries_metric}_{ztag}_{ttag}"


def _artifact_prefix_for_vit(args: argparse.Namespace) -> str:
    ztag = "zscore" if args.use_zscore else "raw"
    return f"vit_{args.vit_matrix_mode}_{args.timeseries_metric}_{ztag}_p{args.vit_patch_size}_d{args.vit_depth}"


def _default_score_filename(args: argparse.Namespace, dataset_dir: Path) -> str:
    if args.score_filename != "score.npy":
        return args.score_filename
    if "vit" in dataset_dir.name:
        return f"{_artifact_prefix_for_vit(args)}_score.npy"
    return f"{_artifact_prefix_for_rnn(args)}_score.npy"


def _show_predict_artifacts(predict_dir: Path, prefix: str, use_zscore: bool) -> None:
    names = [
        "future_predictions.npy",
        f"{prefix}_future_predictions.npy",
        "best_model.pth",
        "best_vit_model.pth",
        f"{prefix}_best_model.pth",
        f"{prefix}_best_vit_model.pth",
    ]
    if use_zscore:
        names += ["future_prediction_std.npy", f"{prefix}_future_prediction_std.npy"]
    print("  - artifacts:")
    for n in names:
        p = predict_dir / n
        if p.exists():
            print(f"    * {n}")


def run_step(args: argparse.Namespace) -> None:
    _section(args.step)
    if args.step == "load_data":
        from .dataset_builder import collect_pair_observations

        cfg = _build_dataset_config(args)
        observations = collect_pair_observations(cfg)
        _kv("observation_count", len(observations))
        if observations:
            _kv("first", f"{observations[0][1]} shape={observations[0][2].shape}")
            _kv("last", f"{observations[-1][1]} shape={observations[-1][2].shape}")
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
        _kv("cropped_file_count", len(outputs))
        return

    if args.step == "build_dataset":
        from .dataset_builder import build_and_save_dataset

        dataset_dir = build_and_save_dataset(_build_dataset_config(args))
        _kv("dataset_dir", dataset_dir)
        return

    if args.step == "train_predict":
        from .modeling import TrainingConfig, run_training_and_prediction

        dataset_dir = args.dataset_dir or (args.output_dir / "dataset_rnn")
        prefix = _artifact_prefix_for_rnn(args)
        _kv("dataset_dir", dataset_dir)
        _kv("artifact_prefix", prefix)
        predict_dir = run_training_and_prediction(
            TrainingConfig(
                dataset_dir=dataset_dir,
                output_dir=args.output_dir,
                next_date=args.next_date,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                pred_batch_size=args.pred_batch_size,
                lr=args.lr,
                metric=args.timeseries_metric,
                model_type=args.ts_model,
                use_timestamp=not args.disable_timestamp,
                use_zscore=args.use_zscore,
                hidden_dim=args.rnn_hidden_dim,
                num_layers=args.rnn_num_layers,
                dropout=args.rnn_dropout,
                optimizer=args.optimizer,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                artifact_prefix=prefix,
            )
        )
        _show_predict_artifacts(predict_dir, prefix, args.use_zscore)
        return

    if args.step == "score":
        from .scoring import ScoreConfig, compute_and_save_score

        dataset_dir = args.dataset_dir or (args.output_dir / "dataset_rnn")
        predict_dir = args.predict_dir or (args.output_dir / "predict")
        prefix = _artifact_prefix_for_vit(args) if "vit" in dataset_dir.name else _artifact_prefix_for_rnn(args)
        score_filename = _default_score_filename(args, dataset_dir)
        _kv("dataset_dir", dataset_dir)
        _kv("predict_dir", predict_dir)
        _kv("artifact_prefix", prefix)
        _kv("score_filename", score_filename)
        score_path = compute_and_save_score(
            ScoreConfig(
                dataset_dir=dataset_dir,
                predict_dir=predict_dir,
                score_filename=score_filename,
                chunk_size=args.score_chunk_size,
                metric=args.timeseries_metric,
                use_zscore=args.use_zscore,
                artifact_prefix=prefix,
            )
        )
        _kv("score_path", score_path)
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
        print("  - output files:")
        for f in output_files:
            print(f"    * {f}")
        return

    if args.step == "full":
        from .pipeline import run_full_pipeline

        result = run_full_pipeline(
            base_dir=args.base_dir,
            geom_reference_dir=args.geom_reference_dir,
            next_date=args.next_date,
            metric=args.timeseries_metric,
            model_type=args.ts_model,
            use_timestamp=not args.disable_timestamp,
            use_zscore=args.use_zscore,
            sequence_length=args.sequence_length,
            matrix_size=args.matrix_size,
            rnn_hidden_dim=args.rnn_hidden_dim,
            rnn_num_layers=args.rnn_num_layers,
            rnn_dropout=args.rnn_dropout,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
        )
        print("  - full pipeline result:")
        for k, v in result.items():
            _kv(k, v)
        return

    if args.step == "vit_build_dataset":
        from .vit_modeling import ViTDatasetBuildConfig, build_and_save_vit_matrix_dataset

        dataset_dir = args.dataset_dir or (args.output_dir / "dataset_rnn")
        vit_dataset_dir = build_and_save_vit_matrix_dataset(
            ViTDatasetBuildConfig(
                dataset_dir=dataset_dir,
                output_dir=args.output_dir,
                metric=args.timeseries_metric,
                matrix_mode=args.vit_matrix_mode,
            )
        )
        _kv("vit_dataset_dir", vit_dataset_dir)
        return

    if args.step == "vit_train_predict":
        from .vit_modeling import ViTConfig, run_vit_training_and_prediction

        dataset_dir = args.dataset_dir or (args.output_dir / "vit_dataset")
        prefix = _artifact_prefix_for_vit(args)
        _kv("dataset_dir", dataset_dir)
        _kv("artifact_prefix", prefix)
        predict_dir = run_vit_training_and_prediction(
            ViTConfig(
                dataset_dir=dataset_dir,
                output_dir=args.output_dir,
                metric=args.timeseries_metric,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                pred_batch_size=args.pred_batch_size,
                lr=args.lr,
                matrix_mode=args.vit_matrix_mode,
                patch_size=args.vit_patch_size,
                hidden_dim=args.vit_hidden_dim,
                depth=args.vit_depth,
                heads=args.vit_heads,
                diag_mask_ratio=args.vit_diag_mask_ratio,
                diag_loss_weight=args.vit_diag_loss_weight,
                use_zscore=args.use_zscore,
                optimizer=args.optimizer,
                weight_decay=args.weight_decay,
                artifact_prefix=prefix,
            )
        )
        _show_predict_artifacts(predict_dir, prefix, args.use_zscore)
        return

    if args.step == "vit_full":
        from .pipeline import run_full_vit_pipeline

        result = run_full_vit_pipeline(
            base_dir=args.base_dir,
            geom_reference_dir=args.geom_reference_dir,
            metric=args.timeseries_metric,
            matrix_mode=args.vit_matrix_mode,
            sequence_length=args.sequence_length,
            matrix_size=args.matrix_size,
            use_zscore=args.use_zscore,
            vit_patch_size=args.vit_patch_size,
            vit_hidden_dim=args.vit_hidden_dim,
            vit_depth=args.vit_depth,
            vit_heads=args.vit_heads,
            vit_diag_mask_ratio=args.vit_diag_mask_ratio,
            vit_diag_loss_weight=args.vit_diag_loss_weight,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )
        print("  - vit_full pipeline result:")
        for k, v in result.items():
            _kv(k, v)
        return


    if args.step == "ccd_build_stack":
        from .temporal_ccd import CCDBuildConfig, build_slc_stack_from_cropped

        cropped_dir = args.cropped_dir or (args.base_dir / "cropped")
        _kv("slc_cropped_dir", cropped_dir)
        ccd_dataset_dir = build_slc_stack_from_cropped(
            CCDBuildConfig(cropped_dir=cropped_dir, output_dir=args.output_dir)
        )
        _kv("ccd_dataset_dir", ccd_dataset_dir)
        _kv("stack_file", ccd_dataset_dir / "slc_stack.npy")
        return

    if args.step == "ccd_run":
        from .temporal_ccd import CCDConfig, run_temporal_ccd

        dataset_dir = args.dataset_dir or (args.output_dir / "ccd_dataset")
        _kv("ccd_dataset_dir", dataset_dir)
        prob_path, change_path = run_temporal_ccd(
            CCDConfig(
                dataset_dir=dataset_dir,
                output_dir=args.output_dir,
                event_date=args.event_date,
                max_temporal_baseline=args.ccd_max_temporal_baseline,
                coherence_window_size=args.ccd_coherence_window_size,
                envelope_bin_width=args.ccd_envelope_bin_width,
                ccd_threshold=args.ccd_threshold,
                kde_bandwidth=args.ccd_kde_bandwidth,
                downsample=args.ccd_downsample,
                artifact_prefix=args.ccd_artifact_prefix,
            )
        )
        _kv("probability_map", prob_path)
        _kv("change_map", change_path)
        return

    if args.step == "ccd_full":
        from .preprocess import CropConfig, batch_crop_filt_fine_cor
        from .temporal_ccd import CCDBuildConfig, CCDConfig, build_slc_stack_from_cropped, run_temporal_ccd

        batch_crop_filt_fine_cor(
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
        ccd_dataset_dir = build_slc_stack_from_cropped(
            CCDBuildConfig(cropped_dir=args.cropped_dir, output_dir=args.output_dir)
        )
        prob_path, change_path = run_temporal_ccd(
            CCDConfig(
                dataset_dir=ccd_dataset_dir,
                output_dir=args.output_dir,
                event_date=args.event_date,
                max_temporal_baseline=args.ccd_max_temporal_baseline,
                coherence_window_size=args.ccd_coherence_window_size,
                envelope_bin_width=args.ccd_envelope_bin_width,
                ccd_threshold=args.ccd_threshold,
                kde_bandwidth=args.ccd_kde_bandwidth,
                downsample=args.ccd_downsample,
                artifact_prefix=args.ccd_artifact_prefix,
            )
        )
        _kv("ccd_dataset_dir", ccd_dataset_dir)
        _kv("probability_map", prob_path)
        _kv("change_map", change_path)
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
    parser.add_argument("--param-file", type=Path, default=None, help="JSON parameter file for dataset/RNN/ViT hyperparameters.")
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
    parser.add_argument("--sequence-length", type=int, default=None, help="Number of nearest pre-event adjacent pairs to keep for time series.")
    parser.add_argument("--matrix-size", type=int, default=None, help="Number of pre-event acquisition dates used for matrix-pair filtering.")

    parser.add_argument("--lat-min", type=float, default=42.625)
    parser.add_argument("--lat-max", type=float, default=42.635)
    parser.add_argument("--lon-min", type=float, default=13.28)
    parser.add_argument("--lon-max", type=float, default=13.30)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--pred-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timeseries-metric", choices=["phase_std", "coherence"], default="phase_std")
    parser.add_argument("--ts-model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--rnn-hidden-dim", type=int, default=64)
    parser.add_argument("--rnn-num-layers", type=int, default=2)
    parser.add_argument("--rnn-dropout", type=float, default=0.1)
    parser.add_argument("--disable-timestamp", action="store_true", help="Disable dates.pkl time feature inputs.")
    parser.add_argument("--use-zscore", action="store_true", help="Enable logit+distribution prediction and zscore scoring.")

    parser.add_argument("--vit-matrix-mode", choices=["similarity", "outer", "difference"], default="similarity")
    parser.add_argument("--vit-patch-size", type=int, default=2)
    parser.add_argument("--vit-hidden-dim", type=int, default=64)
    parser.add_argument("--vit-depth", type=int, default=4)
    parser.add_argument("--vit-heads", type=int, default=4)
    parser.add_argument("--vit-diag-mask-ratio", type=float, default=0.5)
    parser.add_argument("--vit-diag-loss-weight", type=float, default=0.3)

    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=None)

    parser.add_argument("--score-filename", default="score.npy")
    parser.add_argument("--score-chunk-size", type=int, default=512)

    parser.add_argument("--lat-file", type=Path, default=None)
    parser.add_argument("--lon-file", type=Path, default=None)
    parser.add_argument("--subset-params", default="-l 42.625 42.635 -L 13.28 13.30")

    parser.add_argument("--ccd-max-temporal-baseline", type=int, default=84)
    parser.add_argument("--ccd-coherence-window-size", type=int, default=5)
    parser.add_argument("--ccd-envelope-bin-width", type=int, default=12)
    parser.add_argument("--ccd-kde-bandwidth", type=float, default=0.05)
    parser.add_argument("--ccd-threshold", type=float, default=0.75)
    parser.add_argument("--ccd-downsample", type=int, default=1)
    parser.add_argument("--ccd-artifact-prefix", default="ccd_temporal")

    return parser


def main() -> None:
    parser = build_parser()
    defaults = parser.parse_args([])
    args = parser.parse_args()
    _apply_param_overrides(args, defaults)

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
