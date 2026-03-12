from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class ScoreConfig:
    dataset_dir: Path
    predict_dir: Path
    score_filename: str = "score.npy"
    chunk_size: int = 512
    metric: Literal["phase_std", "coherence"] = "phase_std"
    use_zscore: bool = False


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
    genuine_filename = _resolve_observation_filename(config.dataset_dir, config.metric)
    genuine_data = np.load(config.dataset_dir / genuine_filename)
    future_predictions = np.load(config.predict_dir / "future_predictions.npy")

    if genuine_data.ndim == 3:
        genuine_data = np.squeeze(genuine_data, axis=-1)

    if config.use_zscore:
        pred_std_path = config.predict_dir / "future_prediction_std.npy"
        if not pred_std_path.exists():
            raise FileNotFoundError("future_prediction_std.npy is required when use_zscore=True")
        future_pred_std = np.load(pred_std_path)
        if config.metric == "coherence":
            zscore = (future_predictions - genuine_data) / (future_pred_std + 1e-8)
        else:
            zscore = (genuine_data - future_predictions) / (future_pred_std + 1e-8)
        score = zscore.astype(np.float32)
    else:
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
