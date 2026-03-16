from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity

from .io_utils import read_isce_file


@dataclass
class CCDBuildConfig:
    cropped_dir: Path
    output_dir: Path


@dataclass
class CCDConfig:
    dataset_dir: Path
    output_dir: Path
    event_date: str = "20160824"
    max_temporal_baseline: int = 84
    coherence_window_size: int = 5
    envelope_bin_width: int = 12
    ccd_threshold: float = 0.75
    kde_bandwidth: float = 0.05
    downsample: int = 1
    artifact_prefix: str = "ccd_temporal"


class TemporalDecorrelationCCD:
    def __init__(
        self,
        slc_data: np.ndarray,
        date_strings: list[str],
        event_date: str,
        coherence_window_size: int = 5,
        envelope_bin_width: int = 12,
        kde_bandwidth: float = 0.05,
    ):
        self.slc_data = slc_data.astype(np.complex64)
        self.date_strings = date_strings
        self.dates = [datetime.strptime(d, "%Y%m%d") for d in date_strings]
        self.event_date = datetime.strptime(event_date, "%Y%m%d")
        self.window_size = coherence_window_size
        self.envelope_bin_width = envelope_bin_width
        self.kde_bandwidth = kde_bandwidth

        self.coherence_stack: list[np.ndarray] = []
        self.pair_info: list[dict] = []

    @staticmethod
    def _envelope_model(dt, mu, tau_g, tau_v):
        term_v = (1.0 / (1.0 + mu)) * np.exp(-dt / max(tau_v, 1.0))
        term_g = (mu / (1.0 + mu)) * np.exp(-dt / max(tau_g, 1.0))
        return np.clip(term_v + term_g, 0, 1)

    def compute_coherence_5x5(self, slc1: np.ndarray, slc2: np.ndarray) -> np.ndarray:
        interferogram = slc1 * np.conj(slc2)
        power1 = np.abs(slc1) ** 2
        power2 = np.abs(slc2) ** 2

        num_real = uniform_filter(interferogram.real, size=self.window_size, mode="constant")
        num_imag = uniform_filter(interferogram.imag, size=self.window_size, mode="constant")
        numerator = np.sqrt(num_real**2 + num_imag**2)

        p1_avg = uniform_filter(power1, size=self.window_size, mode="constant")
        p2_avg = uniform_filter(power2, size=self.window_size, mode="constant")
        denominator = np.sqrt(p1_avg * p2_avg)

        coherence = numerator / (denominator + 1e-10)
        return np.clip(coherence, 0, 1)

    def compute_all_coherences(self, max_temporal_baseline: int = 84) -> None:
        n_times = self.slc_data.shape[2]
        self.coherence_stack = []
        self.pair_info = []

        for i in range(n_times):
            for j in range(i + 1, n_times):
                dt_days = (self.dates[j] - self.dates[i]).days
                if dt_days > max_temporal_baseline:
                    continue

                slc1 = self.slc_data[:, :, i]
                slc2 = self.slc_data[:, :, j]
                if np.sum(np.abs(slc1) > 0) == 0 or np.sum(np.abs(slc2) > 0) == 0:
                    continue

                coh = self.compute_coherence_5x5(slc1, slc2)
                if self.dates[i] < self.event_date and self.dates[j] < self.event_date:
                    pair_type = "reference"
                elif self.dates[i] < self.event_date <= self.dates[j]:
                    pair_type = "event"
                else:
                    pair_type = "post"

                self.coherence_stack.append(coh)
                self.pair_info.append({"idx1": i, "idx2": j, "type": pair_type, "dt": dt_days})

    def extract_envelope_parameters(self, row: int, col: int) -> dict | None:
        ref_coh: list[float] = []
        ref_dt: list[int] = []
        for idx, info in enumerate(self.pair_info):
            if info["type"] != "reference":
                continue
            c = float(self.coherence_stack[idx][row, col])
            if np.isfinite(c) and 0.01 < c < 0.99:
                ref_coh.append(c)
                ref_dt.append(info["dt"])

        if len(ref_coh) < 3:
            return None

        ref_coh_arr = np.asarray(ref_coh)
        ref_dt_arr = np.asarray(ref_dt)
        bin_width = self.envelope_bin_width
        bins = np.arange(0, ref_dt_arr.max() + bin_width + 1, bin_width)

        env_t: list[float] = []
        env_c: list[float] = []
        for k in range(len(bins) - 1):
            mask = (ref_dt_arr >= bins[k]) & (ref_dt_arr < bins[k + 1])
            if np.any(mask):
                cbin = ref_coh_arr[mask]
                tbin = ref_dt_arr[mask]
                m = int(np.argmax(cbin))
                env_t.append(float(tbin[m]))
                env_c.append(float(cbin[m]))

        if len(env_t) < 3:
            return None

        try:
            p0 = [2.0, 500.0, 30.0]
            bounds = ([0.1, 50, 5], [20, 5000, 500])
            popt, _ = curve_fit(
                self._envelope_model,
                np.asarray(env_t),
                np.asarray(env_c),
                p0=p0,
                bounds=bounds,
                maxfev=10000,
            )
            mu, tau_g, tau_v = [float(v) for v in popt]
            if mu > 0 and tau_g > tau_v > 0:
                return {"mu": mu, "tau_g": tau_g, "tau_v": tau_v}
        except Exception:
            return None

        return None

    @staticmethod
    def _extract_gamma_rand_single(coh_obs: float, dt: float, mu: float, tau_g: float, tau_v: float):
        gamma_g_time = math.exp(-dt / tau_g)
        gamma_v_time = math.exp(-dt / tau_v)
        alpha_g = gamma_g_time / (gamma_g_time + gamma_v_time + 1e-15)

        if alpha_g > 0.9:
            den = math.exp(-dt / tau_g)
            if den > 1e-10:
                return float(np.clip(coh_obs / den, 0, 1)), "ground"
            return None, None

        if alpha_g > 0.5:
            num = coh_obs - (1.0 / (1.0 + mu)) * math.exp(-dt / tau_v)
            den = (mu / (1.0 + mu)) * math.exp(-dt / tau_g)
            if den > 1e-10:
                return float(np.clip(num / den, 0, 1)), "coupled_ground"
            return None, None

        num = coh_obs - (mu / (1.0 + mu)) * math.exp(-dt / tau_g)
        den = (1.0 / (1.0 + mu)) * math.exp(-dt / tau_v)
        if den > 1e-10:
            return float(np.clip(num / den, 0, 1)), "coupled_volume"
        return None, None

    @staticmethod
    def _estimate_cdf_kde(reference_samples: list[float], query_points: np.ndarray, bandwidth: float = 0.05) -> np.ndarray:
        if len(reference_samples) < 2:
            return np.full(len(query_points), 0.5, dtype=np.float32)

        samples = np.asarray(reference_samples, dtype=np.float32).reshape(-1, 1)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(samples)

        x_min = max(0.0, float(samples.min() - 3 * bandwidth))
        x_max = min(1.0, float(samples.max() + 3 * bandwidth))
        x_grid = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
        log_pdf = kde.score_samples(x_grid)
        pdf = np.exp(log_pdf)
        dx = x_grid[1, 0] - x_grid[0, 0]
        cdf_grid = np.cumsum(pdf) * dx
        cdf_grid = cdf_grid / max(cdf_grid[-1], 1e-12)
        return np.interp(query_points, x_grid.ravel(), cdf_grid).astype(np.float32)

    def process_pixel(self, row: int, col: int) -> float:
        params = self.extract_envelope_parameters(row, col)
        if params is None:
            return float("nan")

        mu, tau_g, tau_v = params["mu"], params["tau_g"], params["tau_v"]
        ref_ground, ref_volume, evt_ground, evt_volume = [], [], [], []

        for idx, info in enumerate(self.pair_info):
            c = float(self.coherence_stack[idx][row, col])
            dt = float(info["dt"])
            if not (np.isfinite(c) and 0.01 < c < 0.99):
                continue

            g, ptype = self._extract_gamma_rand_single(c, dt, mu, tau_g, tau_v)
            if g is None:
                continue
            is_ground = ptype in {"ground", "coupled_ground"}

            if info["type"] == "reference":
                (ref_ground if is_ground else ref_volume).append(g)
            elif info["type"] == "event":
                (evt_ground if is_ground else evt_volume).append(g)

        probs: list[float] = []
        if len(ref_ground) >= 3 and len(evt_ground) > 0:
            cdf = self._estimate_cdf_kde(ref_ground, np.asarray(evt_ground, dtype=np.float32), bandwidth=self.kde_bandwidth)
            probs.extend((1.0 - cdf).tolist())
        if len(ref_volume) >= 3 and len(evt_volume) > 0:
            cdf = self._estimate_cdf_kde(ref_volume, np.asarray(evt_volume, dtype=np.float32), bandwidth=self.kde_bandwidth)
            probs.extend((1.0 - cdf).tolist())

        if len(probs) == 0:
            return float("nan")
        return float(np.mean(probs))

    def generate_probability_map(self, downsample: int = 1) -> np.ndarray:
        rows, cols = self.slc_data.shape[:2]
        prob = np.full((rows, cols), np.nan, dtype=np.float32)
        for r in range(0, rows, downsample):
            for c in range(0, cols, downsample):
                prob[r, c] = self.process_pixel(r, c)
        return prob


def build_slc_stack_from_cropped(config: CCDBuildConfig) -> Path:
    files = sorted(config.cropped_dir.glob("*_slc.full"))
    if len(files) == 0:
        raise FileNotFoundError(f"No *_slc.full files found in {config.cropped_dir}")

    dates = [f.name.split("_")[0] for f in files]
    arrays = []
    h, w = None, None
    for f in files:
        arr = read_isce_file(f)
        band = arr[:, :, 0]
        if h is None:
            h, w = band.shape
        elif band.shape != (h, w):
            raise ValueError(f"Shape mismatch: {f.name} has {band.shape}, expected {(h, w)}")
        arrays.append(band.astype(np.complex64))

    stack = np.stack(arrays, axis=2)
    out_dir = config.output_dir / "ccd_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "slc_stack.npy", stack)
    with open(out_dir / "dates.pkl", "wb") as f:
        pickle.dump(dates, f)
    return out_dir


def run_temporal_ccd(config: CCDConfig) -> tuple[Path, Path]:
    with open(config.dataset_dir / "dates.pkl", "rb") as f:
        dates = pickle.load(f)
    slc_stack = np.load(config.dataset_dir / "slc_stack.npy").astype(np.complex64)

    ccd = TemporalDecorrelationCCD(
        slc_data=slc_stack,
        date_strings=dates,
        event_date=config.event_date,
        coherence_window_size=config.coherence_window_size,
        envelope_bin_width=config.envelope_bin_width,
        kde_bandwidth=config.kde_bandwidth,
    )
    ccd.compute_all_coherences(max_temporal_baseline=config.max_temporal_baseline)
    prob_map = ccd.generate_probability_map(downsample=config.downsample)
    change_map = np.where(np.isnan(prob_map), np.nan, (prob_map >= config.ccd_threshold).astype(np.float32))

    predict_dir = config.output_dir / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    prob_path = predict_dir / f"{config.artifact_prefix}_probability.npy"
    chg_path = predict_dir / f"{config.artifact_prefix}_change.npy"
    np.save(prob_path, prob_map)
    np.save(chg_path, change_map)
    return prob_path, chg_path
