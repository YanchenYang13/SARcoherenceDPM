from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .modeling import to_zscore_training_space

# Small constant used to avoid division by zero in score normalisation.
_SCORE_EPS = 1e-8


@dataclass
class ScoreConfig:
    dataset_dir: Path
    predict_dir: Path
    score_filename: str = "score.npy"
    chunk_size: int = 512
    metric: Literal["phase_std", "coherence"] = "phase_std"
    use_zscore: bool = False
    artifact_prefix: str = ""
    score_mode: Literal["auto", "direct", "ndi", "zscore"] = "auto"


def calculate_difference(
    interferogram1: np.ndarray,
    interferogram2: np.ndarray,
    chunk_size: int = 1024,
    metric: str = "phase_std",
) -> np.ndarray:
    if interferogram1.shape != interferogram2.shape:
        raise ValueError("Both interferograms must have the same shape.")

    rows, cols = interferogram1.shape
    difference = np.zeros((rows, cols), dtype=np.float32)

    for i in range(0, rows, chunk_size):
        for j in range(0, cols, chunk_size):
            end_i = min(i + chunk_size, rows)
            end_j = min(j + chunk_size, cols)
            chunk1 = interferogram1[i:end_i, j:end_j]
            chunk2 = interferogram2[i:end_i, j:end_j]
            denominator = chunk1 + chunk2 + 1e-8
            valid_mask = denominator != 0
            diff_chunk = np.full_like(chunk1, np.nan, dtype=np.float32)
            if metric == "coherence":
                diff_chunk[valid_mask] = (chunk2[valid_mask] - chunk1[valid_mask]) / denominator[valid_mask]
            else:
                diff_chunk[valid_mask] = (chunk1[valid_mask] - chunk2[valid_mask]) / denominator[valid_mask]
            difference[i:end_i, j:end_j] = diff_chunk

    mask = np.isnan(interferogram1) | np.isnan(interferogram2) | (interferogram1 == 0) | (interferogram2 == 0)
    difference[mask] = np.nan
    return difference



def _resolve_observation_filename(dataset_dir: Path, metric: str) -> str:
    candidates = ["score_observation_std.npy", "geninue_std.npy"] if metric == "phase_std" else ["score_observation.npy", "geninue.npy"]
    for name in candidates:
        if (dataset_dir / name).exists():
            return name
    raise FileNotFoundError(f"No score observation file found in {dataset_dir}. Tried: {candidates}")

def compute_and_save_score(config: ScoreConfig) -> Path:
    # Method-specific score fallback (e.g., temporal CCD already outputs probability map)
    if config.score_mode == "auto" and config.artifact_prefix:
        ccd_prob = config.predict_dir / f"{config.artifact_prefix}_probability.npy"
        if ccd_prob.exists():
            score = np.load(ccd_prob).astype(np.float32)
            output_path = config.predict_dir / config.score_filename
            np.save(output_path, score)
            return output_path

    genuine_filename = _resolve_observation_filename(config.dataset_dir, config.metric)
    genuine_data = np.load(config.dataset_dir / genuine_filename)
    pred_candidates = []
    if config.artifact_prefix:
        pred_candidates.append(f"{config.artifact_prefix}_future_predictions.npy")
    pred_candidates.append("future_predictions.npy")

    pred_path = None
    for name in pred_candidates:
        p = config.predict_dir / name
        if p.exists():
            pred_path = p
            break
    if pred_path is None:
        raise FileNotFoundError(f"No prediction file found in {config.predict_dir}. Tried: {pred_candidates}")
    future_predictions = np.load(pred_path)

    if genuine_data.ndim == 3:
        genuine_data = np.squeeze(genuine_data, axis=-1)

    resolved_mode = config.score_mode
    if resolved_mode == "auto":
        resolved_mode = "zscore" if config.use_zscore else "ndi"

    if resolved_mode == "zscore":
        # ------------------------------------------------------------------
        # Preferred: compute z-score in logit-coherence (latent) space.
        #
        # The RNN/GRU model is trained with Gaussian NLL in logit-coherence
        # space, so the statistically correct z-score is:
        #
        #   z = (genuine_latent - pred_latent) / pred_latent_std
        #
        # where pred_latent_std is the model's distribution std (σ) in that
        # space.  This is distinct from phase_std data values.
        #
        # Using metric-space z-scores (dividing by delta-method-propagated
        # uncertainty) is problematic because the Jacobian d(metric)/d(z)
        # approaches zero near extreme coherence values, making the
        # propagated std ≈ 0 and the z-score blow up to ±hundreds or ±thousands.
        # ------------------------------------------------------------------
        latent_candidates = []
        latent_std_candidates = []
        if config.artifact_prefix:
            latent_candidates.append(f"{config.artifact_prefix}_future_predictions_latent.npy")
            latent_std_candidates.append(f"{config.artifact_prefix}_future_prediction_latent_std.npy")
        latent_candidates.append("future_predictions_latent.npy")
        latent_std_candidates.append("future_prediction_latent_std.npy")

        pred_latent_path = next(
            (config.predict_dir / n for n in latent_candidates if (config.predict_dir / n).exists()), None
        )
        pred_latent_std_path = next(
            (config.predict_dir / n for n in latent_std_candidates if (config.predict_dir / n).exists()), None
        )

        if pred_latent_path is not None and pred_latent_std_path is not None:
            # Latent-space z-score: numerically stable and statistically correct.
            pred_latent = np.load(pred_latent_path)
            # pred_latent_std is the model distribution σ in logit-coherence
            # space — NOT a phase_std value despite the similar name.
            pred_latent_std = np.load(pred_latent_std_path)

            # Transform genuine observations to logit-coherence space.
            # Both metrics map to the same latent space so the sign convention
            # below is uniform: higher logit-coh means more stable/coherent.
            # A positive z-score means the observation was MORE coherent than
            # predicted (less change); negative means LESS coherent (more change).
            genuine_latent = to_zscore_training_space(genuine_data, config.metric)
            score = (genuine_latent - pred_latent) / (pred_latent_std + _SCORE_EPS)
            score = score.astype(np.float32)
        else:
            # Fallback: metric-space z-score (may produce extreme values near
            # coherence extremes; present for backward compatibility only).
            std_candidates = []
            if config.artifact_prefix:
                std_candidates.append(f"{config.artifact_prefix}_future_prediction_std.npy")
            std_candidates.append("future_prediction_std.npy")
            pred_std_path = None
            for name in std_candidates:
                p = config.predict_dir / name
                if p.exists():
                    pred_std_path = p
                    break
            if pred_std_path is None:
                raise FileNotFoundError(
                    f"future_prediction_latent_std.npy (preferred) or future_prediction_std.npy "
                    f"is required when score_mode=zscore. Tried: {latent_std_candidates + std_candidates}"
                )
            future_pred_std = np.load(pred_std_path)
            if config.metric == "coherence":
                score = (future_predictions - genuine_data) / (future_pred_std + _SCORE_EPS)
            else:
                score = (genuine_data - future_predictions) / (future_pred_std + _SCORE_EPS)
            score = score.astype(np.float32)
    elif resolved_mode == "direct":
        if config.metric == "coherence":
            score = (future_predictions - genuine_data).astype(np.float32)
        else:
            score = (genuine_data - future_predictions).astype(np.float32)
    else:  # ndi
        score = calculate_difference(
            genuine_data,
            future_predictions,
            chunk_size=config.chunk_size,
            metric=config.metric,
        )

    score = np.where(np.isnan(genuine_data), np.nan, np.where(genuine_data == 0, 0, score))
    score = np.where(np.isnan(future_predictions), np.nan, np.where(future_predictions == 0, 0, score))

    output_path = config.predict_dir / config.score_filename
    np.save(output_path, score)
    return output_path

