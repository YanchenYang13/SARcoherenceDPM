from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ScoreConfig:
    dataset_dir: Path
    predict_dir: Path
    score_filename: str = "score.npy"
    chunk_size: int = 512


def calculate_difference(interferogram1: np.ndarray, interferogram2: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
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
            diff_chunk[valid_mask] = (chunk1[valid_mask] - chunk2[valid_mask]) / denominator[valid_mask]
            difference[i:end_i, j:end_j] = diff_chunk

    mask = np.isnan(interferogram1) | np.isnan(interferogram2) | (interferogram1 == 0) | (interferogram2 == 0)
    difference[mask] = np.nan
    return difference


def compute_and_save_score(config: ScoreConfig) -> Path:
    geninue_std = np.load(config.dataset_dir / "geninue_std.npy")
    future_predictions = np.load(config.predict_dir / "future_predictions.npy")

    if geninue_std.ndim == 3:
        geninue_std = np.squeeze(geninue_std, axis=-1)

    phase_score = calculate_difference(geninue_std, future_predictions, chunk_size=config.chunk_size)
    phase_score = np.where(np.isnan(geninue_std), np.nan, np.where(geninue_std == 0, 0, phase_score))
    phase_score = np.where(np.isnan(future_predictions), np.nan, np.where(future_predictions == 0, 0, phase_score))

    output_path = config.predict_dir / config.score_filename
    np.save(output_path, phase_score)
    return output_path
