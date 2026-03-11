from __future__ import annotations

import datetime as dt
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coherence import estimate_coherence_from_int, write_isce_bip_cor
from .io_utils import read_isce_cor
from .isce_stack import discover_stack_pair_products


@dataclass
class DatasetConfig:
    cropped_dir: Path
    output_dir: Path
    event_date: dt.datetime = dt.datetime(2016, 8, 24)
    data_mode: str = "coherence"
    input_source: str = "cor"  # cor | stack_int
    stack_root: Path | None = None
    coherence_source: str = "isce"  # isce | computed_phsig | computed_crlb
    win: int = 5
    looks: float | None = None
    std_thresh: float = 1.0
    use_circular_std: bool = True
    persist_computed_cor: bool = False
    timeseries_length: int | None = None


def _parse_pair_dates(date_pair: str) -> tuple[dt.datetime, dt.datetime]:
    start_str, end_str = date_pair.split("_")
    return dt.datetime.strptime(start_str, "%Y%m%d"), dt.datetime.strptime(end_str, "%Y%m%d")


def _date_to_dt(date_str: str) -> dt.datetime:
    return _parse_pair_dates(date_str)[0]


def _pair_sort_key(date_pair: str) -> tuple[dt.datetime, dt.datetime]:
    return _parse_pair_dates(date_pair)


def find_cor_files_sorted(cropped_dir: Path) -> list[tuple[dt.datetime, str, Path]]:
    file_infos = []
    for path in cropped_dir.rglob("*filt_fine.cor"):
        m = re.search(r"(\d{8}_\d{8})", path.name)
        if not m:
            continue
        date_str = m.group(1)
        file_infos.append((_date_to_dt(date_str), date_str, path))
    return sorted(file_infos, key=lambda x: _pair_sort_key(x[1]))


def _select_adjacent_pair_observations(
    observations: list[tuple[dt.datetime, str, np.ndarray]],
    event_date: dt.datetime,
    timeseries_length: int | None = None,
) -> tuple[list[tuple[dt.datetime, str, np.ndarray]], np.ndarray]:
    if not observations:
        raise RuntimeError("No observations found.")

    pair_to_obs = {date_pair: (start_dt, date_pair, arr) for start_dt, date_pair, arr in observations}

    acquisitions = sorted(
        {
            acq_dt
            for _, date_pair, _ in observations
            for acq_dt in _parse_pair_dates(date_pair)
        }
    )
    pre_event_acqs = [d for d in acquisitions if d < event_date]
    post_event_acqs = [d for d in acquisitions if d >= event_date]

    if len(pre_event_acqs) < 2:
        raise RuntimeError("Need at least two pre-event acquisitions to build adjacent-pair time series.")
    if not post_event_acqs:
        raise RuntimeError("Need at least one post-event acquisition for genuine target.")

    train_observations: list[tuple[dt.datetime, str, np.ndarray]] = []
    for i in range(len(pre_event_acqs) - 1):
        pair = f"{pre_event_acqs[i]:%Y%m%d}_{pre_event_acqs[i + 1]:%Y%m%d}"
        obs = pair_to_obs.get(pair)
        if obs is not None:
            train_observations.append(obs)

    if len(train_observations) < 2:
        raise RuntimeError(
            "Found fewer than two adjacent pre-event interferograms. "
            "Please ensure adjacent *_filt_fine.cor pairs exist after cropping."
        )

    if timeseries_length is not None:
        if timeseries_length < 2:
            raise ValueError("timeseries_length must be >= 2 when provided")
        if len(train_observations) < timeseries_length:
            raise RuntimeError(
                f"Requested timeseries_length={timeseries_length}, but only "
                f"{len(train_observations)} adjacent pre-event pairs are available."
            )
        train_observations = train_observations[-timeseries_length:]

    first_post = post_event_acqs[0]
    last_pre = pre_event_acqs[-1]
    genuine_pair = f"{last_pre:%Y%m%d}_{first_post:%Y%m%d}"
    genuine_obs = pair_to_obs.get(genuine_pair)
    if genuine_obs is None:
        # Fallback: pick earliest pair that crosses from pre-event to post-event.
        crossing = []
        for _, date_pair, arr in observations:
            s, e = _parse_pair_dates(date_pair)
            if s < event_date <= e:
                crossing.append(((s, e), arr))
        if not crossing:
            raise RuntimeError(
                "No crossing interferogram found for genuine target. "
                "Need at least one pair with start<event_date<=end."
            )
        crossing.sort(key=lambda x: x[0])
        genuine_data = crossing[0][1]
    else:
        genuine_data = genuine_obs[2]

    train_observations.sort(key=lambda x: _pair_sort_key(x[1]))
    return train_observations, genuine_data


def _find_cropped_int(cropped_dir: Path, date_pair: str) -> Path | None:
    candidates = [
        cropped_dir / f"{date_pair}_filt_fine.int",
        cropped_dir / f"{date_pair}.filt_fine.int",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_pair_observations(config: DatasetConfig) -> list[tuple[dt.datetime, str, np.ndarray]]:
    """Collect per-pair coherence observations.

    Returns a list of tuples: (start_datetime, date_pair_str, coherence_2d_array)
    """
    observations: list[tuple[dt.datetime, str, np.ndarray]] = []

    if config.input_source == "cor":
        for start_dt, date_str, path in find_cor_files_sorted(config.cropped_dir):
            coh = read_isce_cor(path)
            observations.append((start_dt, date_str, coh))
        return observations

    if config.input_source != "stack_int":
        raise ValueError("input_source must be 'cor' or 'stack_int'")

    if config.stack_root is None:
        raise ValueError("stack_root must be provided when input_source='stack_int'")

    products = discover_stack_pair_products(config.stack_root)
    if config.persist_computed_cor:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    for p in products:
        start_dt = _date_to_dt(p.date_pair)

        if config.coherence_source == "isce":
            if p.cor_path is None:
                continue
            coh = read_isce_cor(p.cor_path)
        else:
            int_path = _find_cropped_int(config.cropped_dir, p.date_pair)
            if int_path is None:
                int_path = p.int_path
            if int_path is None:
                continue

            method = "phsig" if config.coherence_source == "computed_phsig" else "crlb"
            amp, coh, _ = estimate_coherence_from_int(
                int_path,
                win=config.win,
                looks=config.looks,
                std_thresh=config.std_thresh,
                use_circular_std=config.use_circular_std,
                method=method,
            )
            if config.persist_computed_cor:
                out_cor = config.output_dir / f"{p.date_pair}_{method}.cor"
                write_isce_bip_cor(amp, coh, out_cor)

        observations.append((start_dt, p.date_pair, coh.astype(np.float32)))

    return sorted(observations, key=lambda x: _pair_sort_key(x[1]))


def build_insar_timeseries_from_observations(
    observations: list[tuple[dt.datetime, str, np.ndarray]],
) -> tuple[np.ndarray, list[str]]:
    if len(observations) < 2:
        raise RuntimeError("Need at least 2 observations to build timeseries and target")

    first = observations[0][2]
    h, w = first.shape
    t = len(observations)

    timeseries = np.zeros((h, w, t - 1), dtype=np.float32)
    dates: list[str] = []

    for i, (_, date_str, coh) in enumerate(observations[:-1]):
        if coh.shape != (h, w):
            raise ValueError(f"Shape mismatch for {date_str}: {coh.shape} vs {(h, w)}")
        timeseries[:, :, i] = coh
        dates.append(date_str)

    return timeseries, dates


def calculate_std_from_cor(cor: np.ndarray, chunk_size: int = 50) -> np.ndarray:
    rows, cols, bands = cor.shape
    result = np.full_like(cor, np.nan, dtype=np.float32)
    epsilon = 1e-8

    for i in range(0, rows, chunk_size):
        for j in range(0, cols, chunk_size):
            end_i = min(i + chunk_size, rows)
            end_j = min(j + chunk_size, cols)
            chunk = cor[i:end_i, j:end_j, :]
            denominator = chunk**2
            valid_mask = denominator > epsilon
            std_chunk = np.where(valid_mask, np.sqrt((1 - denominator) / (2 * denominator)), 0.0)
            result[i:end_i, j:end_j, :] = std_chunk

    result[np.isnan(cor)] = np.nan
    result[cor == 0] = 0.0
    return result


def save_dataset(output_subfolder: Path, timeseries: np.ndarray, dates: list[str], geninue_data: np.ndarray) -> None:
    output_subfolder.mkdir(parents=True, exist_ok=True)
    np.save(output_subfolder / "data.npy", timeseries)
    with open(output_subfolder / "dates.pkl", "wb") as f:
        pickle.dump(dates, f)

    np.save(output_subfolder / "geninue.npy", geninue_data)
    np.save(output_subfolder / "data_std.npy", calculate_std_from_cor(timeseries))

    if geninue_data.ndim == 2:
        geninue_data = np.expand_dims(geninue_data, axis=-1)
    np.save(output_subfolder / "geninue_std.npy", calculate_std_from_cor(geninue_data))


def build_and_save_dataset(config: DatasetConfig) -> Path:
    observations = collect_pair_observations(config)
    train_observations, geninue_data = _select_adjacent_pair_observations(
        observations, config.event_date, config.timeseries_length
    )

    timeseries, dates = build_insar_timeseries_from_observations(train_observations)

    output_subfolder = config.output_dir / "dataset"
    save_dataset(output_subfolder, timeseries, dates, geninue_data)
    return output_subfolder
