from __future__ import annotations

"""Command-line entrypoint for the SARcoherenceDPM workflow.

The parser exposes step-oriented sub-flows so users can run either:
- isolated processing stages (crop/dataset/train/score/output/visualize), or
- high-level orchestrations (full, vit_full, ccd_full).

This file only wires configuration and control flow; core algorithms live in the
specialized modules under `insar_pipeline/`.
"""

import argparse
import json
import datetime as dt
from pathlib import Path

from . import __version__


STEP_CHOICES = [
    "load_data",
    "crop",
    "prepare_int_aux",
    "build_dataset",
    "train_predict",
    "score",
    "output",
    "visualize",
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
        observation_file=args.observation_file,
        dataset_name=args.dataset_name,
    )


def _load_param_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--param-file must point to a JSON object")
    return data


def _apply_param_overrides(args: argparse.Namespace, defaults: argparse.Namespace) -> dict[str, tuple[object, object]]:
    if args.param_file is None:
        return {}

    params = _load_param_file(args.param_file)
    merged: dict = {}
    for section in ("global", "dataset", "rnn", "vit"):
        sec = params.get(section, {})
        if isinstance(sec, dict):
            merged.update(sec)

    applied: dict[str, tuple[object, object]] = {}
    for key, value in merged.items():
        arg_key = key.replace("-", "_")
        if not hasattr(args, arg_key):
            continue
        if getattr(args, arg_key) == getattr(defaults, arg_key):
            old_value = getattr(args, arg_key)
            setattr(args, arg_key, value)
            applied[arg_key] = (old_value, value)
    return applied


def _section(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n[SARcoherenceDPM v{__version__}] [STEP] {title}\n{line}", flush=True)


def _kv(key: str, value) -> None:
    print(f"  - {key}: {value}", flush=True)


def _artifact_prefix_for_rnn(args: argparse.Namespace) -> str:
    ztag = "zscore" if args.use_zscore else "raw"
    ttag = "time" if not args.disable_timestamp else "notime"
    base = f"rnn_{args.ts_model}_{args.timeseries_metric}_{ztag}_{ttag}"
    if args.artifact_tag:
        return f"{base}_{args.artifact_tag}"
    return base


def _artifact_prefix_for_vit(args: argparse.Namespace) -> str:
    ztag = "zscore" if args.use_zscore else "raw"
    base = f"vit_{args.vit_matrix_mode}_{args.timeseries_metric}_{ztag}_p{args.vit_patch_size}_d{args.vit_depth}"
    if args.artifact_tag:
        return f"{base}_{args.artifact_tag}"
    return base


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

    if args.step == "prepare_int_aux":
        from .int_auxiliary import IntAuxiliaryConfig, prepare_int_auxiliary_products

        outputs = prepare_int_auxiliary_products(
            IntAuxiliaryConfig(
                cropped_dir=args.cropped_dir,
                corr_win=args.aux_corr_win,
                phsig_win=args.aux_phsig_win,
                variance_win=args.aux_variance_win,
                variance_looks=args.aux_variance_looks,
                variance_block_lines=args.aux_block_lines,
                output_var=args.aux_output_var,
            )
        )
        _kv("prepared_aux_products", len(outputs))
        for path in outputs:
            print(f"    * {path}")
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
        _kv("param_file", args.param_file)
        if getattr(args, "_param_overrides", None):
            _kv("param_overrides", {k: new for k, (_, new) in args._param_overrides.items()})
        _kv("model", args.ts_model)
        _kv("metric", args.timeseries_metric)
        _kv("timestamp", "enabled" if not args.disable_timestamp else "disabled")
        _kv("device_policy", "cuda-if-available-else-cpu")
        _kv("epochs", args.epochs)
        _kv("train_batch_size", args.train_batch_size)
        _kv("pred_batch_size", args.pred_batch_size)
        _kv("lr", args.lr)
        _kv("rnn_hidden_dim", args.rnn_hidden_dim)
        _kv("rnn_num_layers", args.rnn_num_layers)
        _kv("rnn_dropout", args.rnn_dropout)
        _kv("optimizer", args.optimizer)
        _kv("weight_decay", args.weight_decay)
        _kv("max_grad_norm", args.max_grad_norm)
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
                score_mode=args.score_mode,
            )
        )
        _kv("score_path", score_path)
        return


    if args.step == "visualize":
        from .visualization import VisualizationConfig, visualize_file

        if args.visualize_input is not None:
            input_file = args.visualize_input
        else:
            predict_dir = args.predict_dir or (args.output_dir / "predict")
            candidates = sorted(predict_dir.glob("*score.npy"))
            if not candidates:
                candidates = sorted(predict_dir.glob("*_probability.npy"))
            if not candidates:
                candidates = sorted(predict_dir.glob("*.npy"))
            if not candidates:
                raise FileNotFoundError(f"No visualization candidate found in {predict_dir}; pass --visualize-input")
            input_file = candidates[-1]

        _kv("visualize_input", input_file)
        _kv("visualize_mode", args.visualize_mode)
        out = visualize_file(
            VisualizationConfig(
                input_file=input_file,
                output_file=args.visualize_output,
                mode=args.visualize_mode,
                cmap=args.visualize_cmap,
                vmin=args.visualize_vmin,
                vmax=args.visualize_vmax,
                mintpy_dataset=args.visualize_dataset,
                nodisplay=args.visualize_nodisplay,
            )
        )
        if out is not None:
            _kv("visualize_output", out)
        return

    if args.step == "output":
        from .output_products import (
            OutputConfig,
            ThresholdMaskConfig,
            apply_threshold_mask_to_tif,
            generate_geocoded_outputs,
        )

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

        if args.mask_enable:
            # Threshold masking is an optional post-output enhancement applied on
            # final TIFF products. It does not affect model/score internals.
            mask_config = ThresholdMaskConfig(
                method=args.mask_method,
                manual_threshold=args.mask_threshold_manual,
                quantile=args.mask_quantile,
                std_n=args.mask_std_n,
                output_suffix=args.mask_output_suffix,
            )
            # By default, mask all newly generated output TIFF files.
            # Users may append extra external TIFF files via --mask-input-tif.
            mask_targets = list(output_files)
            if args.mask_input_tif:
                mask_targets.extend(args.mask_input_tif)

            print("  - masked files:")
            for tif in mask_targets:
                masked = apply_threshold_mask_to_tif(tif, mask_config)
                print(f"    * {masked}")
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
    parser.add_argument(
        "--observation-file",
        choices=[
            "filt_fine.cor",
            "fine.cor.full",
            "unfilt_fine.cor",
            "underamp_unfilt_fine.cor",
            "underamp_unfilt_fine_circ.cor",
            "filt_fine.std",
        ],
        default="filt_fine.cor",
        help="Cropped observation product used when input_source=cor.",
    )
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
    parser.add_argument("--dataset-name", default=None, help="Optional dataset output folder name.")
    parser.add_argument("--artifact-tag", default="", help="Optional suffix added to model/score artifact prefixes.")

    parser.add_argument("--aux-corr-win", type=int, default=5, help="ICU PHASESIGMA correlation window for prepare_int_aux.")
    parser.add_argument("--aux-phsig-win", type=int, default=5, help="ICU PHASESIGMA sigma window for prepare_int_aux.")
    parser.add_argument("--aux-variance-win", type=int, default=5, help="Variance window for under-amplitude products.")
    parser.add_argument("--aux-variance-looks", type=float, default=3.0, help="Looks factor used for Q/std conversion.")
    parser.add_argument("--aux-block-lines", type=int, default=512, help="Block line count for under-amplitude generation.")
    parser.add_argument("--aux-output-var", action="store_true", help="Also persist intermediate .var outputs for prepare_int_aux.")

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
    parser.add_argument("--score-mode", choices=["auto", "direct", "ndi", "zscore"], default="auto")
    parser.add_argument("--score-chunk-size", type=int, default=512)

    parser.add_argument("--lat-file", type=Path, default=None)
    parser.add_argument("--lon-file", type=Path, default=None)
    parser.add_argument("--subset-params", default="-l 42.625 42.635 -L 13.28 13.30")

    # Optional threshold-based post-processing for output TIFF maps.
    parser.add_argument("--mask-enable", action="store_true", help="Apply threshold mask on output tif(s)")
    parser.add_argument("--mask-input-tif", type=Path, nargs="*", default=None, help="Additional tif files to mask")
    parser.add_argument("--mask-method", choices=["manual", "quantile", "std"], default="quantile")
    parser.add_argument("--mask-threshold-manual", type=float, default=None)
    parser.add_argument("--mask-quantile", type=float, default=0.70)
    parser.add_argument("--mask-std-n", type=float, default=2.0)
    parser.add_argument("--mask-output-suffix", default="mask")

    parser.add_argument("--visualize-input", type=Path, default=None)
    parser.add_argument("--visualize-output", type=Path, default=None)
    parser.add_argument("--visualize-mode", choices=["auto", "mintpy", "matplotlib"], default="auto")
    parser.add_argument("--visualize-dataset", default=None, help="Optional dataset name for mintpy view")
    parser.add_argument("--visualize-cmap", default="turbo")
    parser.add_argument("--visualize-vmin", type=float, default=None)
    parser.add_argument("--visualize-vmax", type=float, default=None)
    parser.add_argument("--visualize-nodisplay", action="store_true")

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
    args._param_overrides = _apply_param_overrides(args, defaults)

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
