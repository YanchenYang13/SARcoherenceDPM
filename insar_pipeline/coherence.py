from __future__ import annotations

from pathlib import Path

import numpy as np

from .isce_stack import read_isce_int


def _uniform_filter(arr: np.ndarray, size: int) -> np.ndarray:
    try:
        from scipy.ndimage import uniform_filter as _scipy_uniform_filter
    except ImportError as exc:
        raise ImportError("scipy is required for coherence estimation from .int files") from exc
    return _scipy_uniform_filter(arr, size=size, mode="nearest")


def phase_std_linear(phi: np.ndarray, win: int) -> np.ndarray:
    mean = _uniform_filter(phi, size=win)
    mean2 = _uniform_filter(phi * phi, size=win)
    var = np.maximum(mean2 - mean * mean, 0.0)
    return np.sqrt(var).astype(np.float32)


def phase_std_circular(phi: np.ndarray, win: int) -> np.ndarray:
    c = _uniform_filter(np.cos(phi), size=win)
    s = _uniform_filter(np.sin(phi), size=win)
    r = np.sqrt(c * c + s * s)
    r = np.clip(r, 1e-8, 1.0)
    sigma = np.sqrt(-2.0 * np.log(r))
    return sigma.astype(np.float32)


def coh_isce_phsig_from_std(sigma: np.ndarray, std_thresh: float) -> np.ndarray:
    t = float(std_thresh)
    coh = 1.0 - np.minimum(sigma, t) / t
    return np.clip(coh, 0.0, 1.0).astype(np.float32)


def coh_crlb_from_std(sigma: np.ndarray, looks: float) -> np.ndarray:
    lks = float(looks)
    gamma2 = 1.0 / (1.0 + 2.0 * lks * (sigma.astype(np.float64) ** 2))
    return np.sqrt(np.clip(gamma2, 0.0, 1.0)).astype(np.float32)


def write_isce_bip_cor(amp: np.ndarray, coh: np.ndarray, out_cor: str | Path) -> None:
    out_cor = Path(out_cor)
    out_cor.parent.mkdir(parents=True, exist_ok=True)
    h, w = amp.shape
    with open(out_cor, "wb") as f:
        row = np.empty((2 * w,), dtype=np.float32)
        for i in range(h):
            row[0::2] = amp[i, :]
            row[1::2] = coh[i, :]
            row.tofile(f)


def estimate_coherence_from_int(
    int_path: str | Path,
    win: int = 5,
    looks: float | None = None,
    std_thresh: float = 1.0,
    use_circular_std: bool = True,
    method: str = "phsig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ifg = read_isce_int(int_path)
    phi = np.angle(ifg)
    sigma = phase_std_circular(phi, win) if use_circular_std else phase_std_linear(phi, win)

    if looks is None:
        looks = float(win * win)

    coh_phsig = coh_isce_phsig_from_std(sigma, std_thresh=std_thresh)
    coh_crlb = coh_crlb_from_std(sigma, looks=looks)
    amp = np.sqrt(np.abs(ifg)).astype(np.float32)

    if method == "phsig":
        coh = coh_phsig
    elif method == "crlb":
        coh = coh_crlb
    else:
        raise ValueError("method must be 'phsig' or 'crlb'")

    return amp, coh, sigma
