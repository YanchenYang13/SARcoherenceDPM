from __future__ import annotations

import datetime as dt
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal

from .coherence import estimate_coherence_from_int, write_isce_bip_cor
from .io_utils import read_isce_cor, read_isce_file
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
    sequence_length: int | None = None
    matrix_size: int | None = None
    observation_file: str = "filt_fine.cor"
    dataset_name: str | None = None
    save_legacy_aliases: bool = True


def _observation_metric(observation_file: str) -> str:
    return "phase_std" if observation_file.endswith(".std") else "coherence"


def _date_to_dt(date_str: str) -> dt.datetime:
    return dt.datetime.strptime(date_str.split("_")[0], "%Y%m%d")


def parse_date_pair(date_pair: str) -> tuple[dt.datetime, dt.datetime]:
    master_str, slave_str = date_pair.split("_")
    return dt.datetime.strptime(master_str, "%Y%m%d"), dt.datetime.strptime(slave_str, "%Y%m%d")


def _collect_sorted_unique_dates(
    observations: list[tuple[dt.datetime, str, np.ndarray]],
) -> list[dt.datetime]:
    """Collect all unique acquisition dates from observations and return them sorted."""
    unique_dates: set[dt.datetime] = set()
    for _, date_pair, _ in observations:
        master_dt, slave_dt = parse_date_pair(date_pair)
        unique_dates.add(master_dt)
        unique_dates.add(slave_dt)
    return sorted(unique_dates)


def filter_adjacent_pairs(
    observations: list[tuple[dt.datetime, str, np.ndarray]],
) -> list[tuple[dt.datetime, str, np.ndarray]]:
    """Keep only observations whose date pair consists of consecutively-ordered
    acquisition dates.

    The correct definition of 'adjacent': collect all unique acquisition dates
    from the full observation list, sort them chronologically, then only pairs
    whose two dates are consecutive in that sorted order are adjacent.
    This is data-driven and independent of any fixed time-interval threshold.
    """
    sorted_dates = _collect_sorted_unique_dates(observations)
    adjacent_set: set[tuple[dt.datetime, dt.datetime]] = {
        (sorted_dates[i], sorted_dates[i + 1])
        for i in range(len(sorted_dates) - 1)
    }
    return [
        obs
        for obs in observations
        if parse_date_pair(obs[1]) in adjacent_set
    ]


def select_adjacent_sequence_window(
    observations: list[tuple[dt.datetime, str, np.ndarray]],
    reference_date: dt.datetime,
    sequence_length: int | None,
) -> list[tuple[dt.datetime, str, np.ndarray]]:
    pre_event_adjacent = [
        obs for obs in filter_adjacent_pairs(observations) if parse_date_pair(obs[1])[1] < reference_date
    ]
    pre_event_adjacent.sort(key=lambda x: parse_date_pair(x[1])[1], reverse=True)
    if sequence_length is not None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        pre_event_adjacent = pre_event_adjacent[:sequence_length]
    return sorted(pre_event_adjacent, key=lambda x: parse_date_pair(x[1])[1])


def select_matrix_pair_window(
    observations: list[tuple[dt.datetime, str, np.ndarray]],
    reference_date: dt.datetime,
    matrix_size: int | None,
) -> tuple[list[str], list[tuple[dt.datetime, str, np.ndarray]]]:
    if matrix_size is None:
        return [], []
    if matrix_size < 1:
        raise ValueError("matrix_size must be >= 1")

    acquisition_dates = sorted(
        {
            d
            for _, date_pair, _ in observations
            for d in parse_date_pair(date_pair)
            if d < reference_date
        }
    )
    selected_dates = acquisition_dates[-matrix_size:]
    selected_set = set(selected_dates)
    matrix_observations = [
        obs
        for obs in observations
        if all(d in selected_set for d in parse_date_pair(obs[1]))
    ]
    selected_date_str = [d.strftime("%Y%m%d") for d in selected_dates]
    return selected_date_str, sorted(matrix_observations, key=lambda x: parse_date_pair(x[1]))


def _find_observation_files_sorted(cropped_dir: Path, observation_file: str) -> list[tuple[dt.datetime, str, Path]]:
    file_infos = []
    pattern = f"*{observation_file}"
    for path in cropped_dir.rglob(pattern):
        m = re.search(r"(\d{8}_\d{8})", path.name)
        if not m:
            continue
        date_str = m.group(1)
        file_infos.append((_date_to_dt(date_str), date_str, path))
    return sorted(file_infos, key=lambda x: x[0])


def find_cor_files_sorted(cropped_dir: Path) -> list[tuple[dt.datetime, str, Path]]:
    return _find_observation_files_sorted(cropped_dir, "filt_fine.cor")


def _read_full_coherence_band(path: Path) -> np.ndarray:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {path}")
    if ds.RasterCount < 2:
        raise ValueError(f"{path} does not expose a second band for coherence.")
    band = ds.GetRasterBand(2)
    data = band.ReadAsArray()
    if data is None:
        raise RuntimeError(f"Cannot read coherence band from {path}")
    return data.astype(np.float32)


def _read_observation_array(path: Path, observation_file: str) -> np.ndarray:
    if observation_file == "fine.cor.full":
        return _read_full_coherence_band(path)
    if observation_file.endswith(".cor"):
        return read_isce_cor(path)
    data = read_isce_file(path)
    if data.ndim == 3:
        data = data[:, :, 0]
    return data.astype(np.float32)


def _default_dataset_name(observation_file: str) -> str:
    sanitized = observation_file.replace(".", "_")
    return f"dataset_rnn_{sanitized}"



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
        for start_dt, date_str, path in _find_observation_files_sorted(config.cropped_dir, config.observation_file):
            coh = _read_observation_array(path, config.observation_file)
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

    return sorted(observations, key=lambda x: x[0])


def build_insar_timeseries_from_observations(
    observations: list[tuple[dt.datetime, str, np.ndarray]],
) -> tuple[np.ndarray, list[str]]:
    if len(observations) < 1:
        raise RuntimeError("Need at least 1 observation to build timeseries")

    first = observations[0][2]
    h, w = first.shape
    t = len(observations)

    timeseries = np.zeros((h, w, t), dtype=np.float32)
    dates: list[str] = []

    for i, (_, date_str, coh) in enumerate(observations):
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


def _phase_std_to_coherence_from_looks(phase_std: np.ndarray, looks: float) -> np.ndarray:
    denom = np.sqrt(1.0 + 2.0 * float(looks) * (phase_std.astype(np.float32) ** 2))
    coh = 1.0 / np.maximum(denom, 1e-8)
    coh = np.where(np.isnan(phase_std), np.nan, np.where(phase_std == 0, 0.0, coh))
    return coh.astype(np.float32)


def save_dataset(
    output_subfolder: Path,
    timeseries: np.ndarray,
    dates: list[str],
    geninue_data: np.ndarray,
    data_mode: str = "coherence",
    looks: float = 3.0,
    save_legacy_aliases: bool = True,
) -> None:
    output_subfolder.mkdir(parents=True, exist_ok=True)

    if data_mode == "phase_std":
        coherence_timeseries = _phase_std_to_coherence_from_looks(timeseries, looks)
        coherence_observation = _phase_std_to_coherence_from_looks(geninue_data, looks)

        np.save(output_subfolder / "rnn_data.npy", coherence_timeseries)
        np.save(output_subfolder / "rnn_data_std.npy", timeseries)
        np.save(output_subfolder / "score_observation.npy", coherence_observation)
        np.save(output_subfolder / "score_observation_std.npy", np.expand_dims(geninue_data, axis=-1) if geninue_data.ndim == 2 else geninue_data)
        if save_legacy_aliases:
            np.save(output_subfolder / "data.npy", coherence_timeseries)
            np.save(output_subfolder / "data_std.npy", timeseries)
            np.save(output_subfolder / "geninue.npy", coherence_observation)
            np.save(output_subfolder / "geninue_std.npy", np.expand_dims(geninue_data, axis=-1) if geninue_data.ndim == 2 else geninue_data)
    else:
        timeseries_std = calculate_std_from_cor(timeseries)
        # RNN canonical files
        np.save(output_subfolder / "rnn_data.npy", timeseries)
        np.save(output_subfolder / "rnn_data_std.npy", timeseries_std)

        # score-required observations
        np.save(output_subfolder / "score_observation.npy", geninue_data)

        if geninue_data.ndim == 2:
            geninue_data = np.expand_dims(geninue_data, axis=-1)

        gen_std = calculate_std_from_cor(geninue_data)
        np.save(output_subfolder / "score_observation_std.npy", gen_std)
        if save_legacy_aliases:
            # backward-compatible aliases
            np.save(output_subfolder / "data.npy", timeseries)
            np.save(output_subfolder / "data_std.npy", timeseries_std)
            np.save(output_subfolder / "geninue.npy", geninue_data)
            np.save(output_subfolder / "geninue_std.npy", gen_std)

    with open(output_subfolder / "dates.pkl", "wb") as f:
        pickle.dump(dates, f)


def build_and_save_dataset(config: DatasetConfig) -> Path:
    observations = collect_pair_observations(config)
    train_observations = select_adjacent_sequence_window(
        observations,
        reference_date=config.event_date,
        sequence_length=config.sequence_length,
    )

    matrix_date_window, matrix_observations = select_matrix_pair_window(
        observations,
        reference_date=config.event_date,
        matrix_size=config.matrix_size,
    )

    timeseries, dates = build_insar_timeseries_from_observations(train_observations)
    geninue_data = observations[-1][2]

    output_subfolder = config.output_dir / (config.dataset_name or _default_dataset_name(config.observation_file))
    save_dataset(
        output_subfolder,
        timeseries,
        dates,
        geninue_data,
        data_mode=_observation_metric(config.observation_file),
        looks=config.looks or 3.0,
        save_legacy_aliases=config.save_legacy_aliases,
    )
    if matrix_date_window:
        with open(output_subfolder / "matrix_dates.pkl", "wb") as f:
            pickle.dump(matrix_date_window, f)
        with open(output_subfolder / "matrix_pairs.pkl", "wb") as f:
            pickle.dump([obs[1] for obs in matrix_observations], f)
    return output_subfolder


def validate_datasets(output_dir: Path) -> None:
    """Validate and print a structured summary of all datasets found under output_dir.

    For each dataset directory containing rnn_data.npy and dates.pkl, prints:
    - dataset name
    - rnn_data shape (H, W, T)
    - number of time steps T
    - list of date pairs used
    - score observation shape
    - whether all expected files exist

    Also prints a cross-dataset consistency check: all datasets should have
    the same (H, W) spatial dimensions and the same time steps if they share
    the same source dates.
    """
    REQUIRED_FILES = ["rnn_data.npy", "dates.pkl", "score_observation.npy"]
    OPTIONAL_FILES = ["rnn_data_std.npy", "score_observation_std.npy"]

    dataset_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / "rnn_data.npy").exists() and (d / "dates.pkl").exists()
    ])

    if not dataset_dirs:
        print(f"[validate_dataset] No datasets found under {output_dir}")
        return

    print(f"\n{'=' * 72}")
    print(f"[validate_dataset] Found {len(dataset_dirs)} dataset(s) under {output_dir}")
    print(f"{'=' * 72}")

    summary_rows = []

    for ds_dir in dataset_dirs:
        print(f"\n  Dataset: {ds_dir.name}")
        print(f"  {'─' * 60}")

        # Check required files
        missing = [f for f in REQUIRED_FILES if not (ds_dir / f).exists()]
        optional_present = [f for f in OPTIONAL_FILES if (ds_dir / f).exists()]

        if missing:
            print(f"  ⚠️  Missing required files: {missing}")
        else:
            print(f"  ✅ All required files present")

        for f in optional_present:
            print(f"  ℹ️  Optional file present: {f}")

        # Load dates
        with open(ds_dir / "dates.pkl", "rb") as f:
            dates = pickle.load(f)
        print(f"  time_steps : {len(dates)}")
        for d in dates:
            print(f"    - {d}")

        # Load rnn_data shape
        rnn_data = np.load(ds_dir / "rnn_data.npy", mmap_mode="r")
        print(f"  rnn_data   : shape={rnn_data.shape}  dtype={rnn_data.dtype}")
        if rnn_data.ndim == 3:
            H, W, T = rnn_data.shape
        elif rnn_data.ndim == 2:
            H, W, T = rnn_data.shape[0], rnn_data.shape[1], 1
        else:
            print(f"  ⚠️  Unexpected rnn_data ndim={rnn_data.ndim}; skipping shape unpack")
            H, W, T = 0, 0, 0

        # Load score_observation shape
        score_obs_path = ds_dir / "score_observation.npy"
        if score_obs_path.exists():
            obs = np.load(score_obs_path, mmap_mode="r")
            print(f"  score_obs  : shape={obs.shape}  dtype={obs.dtype}")
        else:
            print(f"  ⚠️  score_observation.npy not found for {ds_dir.name}")

        summary_rows.append({
            "name": ds_dir.name,
            "shape": rnn_data.shape,
            "T": T,
            "H": H,
            "W": W,
            "missing": missing,
        })

    # Cross-dataset consistency check
    print(f"\n{'=' * 72}")
    print("[validate_dataset] Cross-dataset consistency check")
    print(f"{'=' * 72}")

    shapes_hw = set((r["H"], r["W"]) for r in summary_rows)
    shapes_t  = set(r["T"] for r in summary_rows)

    if len(shapes_hw) == 1:
        print(f"  ✅ Spatial dimensions consistent: {shapes_hw.pop()}")
    else:
        print(f"  ❌ Spatial dimension mismatch: {shapes_hw}")

    if len(shapes_t) == 1:
        print(f"  ✅ Time steps consistent across all datasets: T={shapes_t.pop()}")
    else:
        print(f"  ⚠️  Time steps differ across datasets:")
        for r in summary_rows:
            flag = "✅" if not r["missing"] else "❌"
            print(f"    {flag} {r['name']:50s}  T={r['T']}  shape={r['shape']}")

    any_missing = [r for r in summary_rows if r["missing"]]
    if any_missing:
        print(f"\n  ❌ {len(any_missing)} dataset(s) have missing files:")
        for r in any_missing:
            print(f"    - {r['name']}: missing {r['missing']}")
    else:
        print(f"\n  ✅ No missing files in any dataset")
