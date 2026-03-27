"""Histogram matching for InSAR time series data.

Implements histogram matching for SAR coherence / phase-std time series,
inspired by the approach described in:
  Liu et al. (2024), "A New Method for the Identification of Earthquake-Damaged
  Buildings Using Sentinel-1 Multitemporal Coherence Optimized by Homogeneous
  SAR Pixels and Histogram Matching", IEEE JSTARS, 17, 7124–7143.
  DOI: 10.1109/JSTARS.2024.3377218

The core idea: before building the training time series, align the histogram of
every acquisition image to a common reference so that systematic distribution
shifts (caused by seasonal decorrelation, baseline variation, etc.) are removed.

Unlike coherence (values in [0, 1]), phase-std values are in [0, ∞), so this
module is designed to be value-range-agnostic: the histogram range is derived
from the data itself rather than hard-coded.
"""
from __future__ import annotations

import numpy as np


def _valid_mask(img: np.ndarray) -> np.ndarray:
    """Return boolean mask of valid (finite, non-zero) pixels."""
    return np.isfinite(img) & (img != 0)


def _global_value_range(
    timeseries: np.ndarray,
    percentile_lo: float = 0.5,
    percentile_hi: float = 99.5,
) -> tuple[float, float]:
    """Compute a robust value range from all valid pixels in the timeseries.

    Parameters
    ----------
    timeseries : np.ndarray, shape (H, W, T)
    percentile_lo, percentile_hi : clipping percentiles to ignore extreme outliers

    Returns
    -------
    (vmin, vmax) as floats
    """
    valid = timeseries[_valid_mask(timeseries)]
    if valid.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(valid, percentile_lo))
    vmax = float(np.percentile(valid, percentile_hi))
    if vmin >= vmax:
        vmin = float(valid.min())
        vmax = float(valid.max())
    if vmin >= vmax:
        # All valid pixels have the same value; use a symmetric range
        vmin = vmax - 0.5
        vmax = vmax + 0.5
    return vmin, vmax


def _compute_cdf(
    values: np.ndarray,
    n_bins: int,
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalised CDF over [vmin, vmax] with n_bins bins.

    Returns
    -------
    bin_centers : ndarray (n_bins,)
    cdf         : ndarray (n_bins,), values in [0, 1]
    """
    bin_edges = np.linspace(vmin, vmax, n_bins + 1)
    hist, _ = np.histogram(values, bins=bin_edges)
    cdf = np.cumsum(hist).astype(np.float64)
    total = cdf[-1]
    if total > 0:
        cdf /= total
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, cdf


def _match_single_to_ref_cdf(
    src: np.ndarray,
    ref_bin_centers: np.ndarray,
    ref_cdf: np.ndarray,
    src_bin_centers: np.ndarray,
    src_cdf: np.ndarray,
) -> np.ndarray:
    """Apply histogram mapping: src → reference distribution.

    For each source pixel value v:
      1. Look up its CDF value: p = CDF_src(v)
      2. Find the reference value v' such that CDF_ref(v') ≈ p
      3. Replace v → v'

    Valid pixels (non-NaN, non-zero) are remapped; others are preserved.
    """
    mask = _valid_mask(src)
    if not np.any(mask):
        return src.copy()

    # Build value → reference_value mapping via CDF interpolation.
    # mapping[i] = reference value whose CDF equals cdf_src[i]
    mapping = np.interp(src_cdf, ref_cdf, ref_bin_centers)

    result = src.copy()
    src_vals = src[mask]

    # Quantise each valid source pixel to a bin index using the left boundary.
    # searchsorted(..., side="left") returns the leftmost index i such that
    # src_bin_centers[i] >= value.  Values below the lowest bin center map to 0;
    # values above the highest bin center map to n (clipped to n-1).
    bin_idx = np.searchsorted(src_bin_centers, src_vals, side="left")
    bin_idx = np.clip(bin_idx, 0, len(src_bin_centers) - 1)
    result[mask] = mapping[bin_idx]
    return result


def histogram_match_timeseries(
    timeseries: np.ndarray,
    strategy: str = "median",
    n_bins: int = 256,
) -> np.ndarray:
    """Apply histogram matching across the time dimension of an InSAR timeseries.

    Aligns the histogram of every time step to a common reference distribution
    so that systematic acquisition-to-acquisition distribution shifts are removed.

    Parameters
    ----------
    timeseries : np.ndarray, shape (H, W, T)
        The input data cube.  May contain NaN and/or zero values (treated as
        no-data and preserved in the output).
    strategy : {"median", "first", "mean"}
        How to define the reference distribution:

        - ``"median"``: use the time step whose per-pixel median is closest to
          the global median across all time steps (the most "typical" acquisition).
        - ``"first"``: use the first time step (index 0) as the reference.
        - ``"mean"``: pool all valid pixels from all time steps into one
          combined distribution and use that as the reference.
    n_bins : int
        Number of histogram bins.  256 is sufficient for most SAR data.

    Returns
    -------
    matched : np.ndarray, shape (H, W, T), dtype float32
        Histogram-matched timeseries.  NaN and zero pixels are unchanged.
    """
    if timeseries.ndim != 3:
        raise ValueError(f"timeseries must be 3-D (H, W, T), got shape {timeseries.shape}")

    _H, _W, T = timeseries.shape
    if T <= 1:
        return timeseries.copy()

    if strategy not in {"median", "first", "mean"}:
        raise ValueError(f"strategy must be 'median', 'first', or 'mean'; got {strategy!r}")

    vmin, vmax = _global_value_range(timeseries)

    # ------------------------------------------------------------------
    # Build the reference CDF
    # ------------------------------------------------------------------
    if strategy == "first":
        ref_img = timeseries[:, :, 0]
        ref_valid = ref_img[_valid_mask(ref_img)]
        ref_bin_centers, ref_cdf = _compute_cdf(ref_valid, n_bins, vmin, vmax)

    elif strategy == "median":
        # Choose the time step whose median pixel value is closest to the
        # global median of all per-step medians.
        step_medians = []
        for t in range(T):
            img = timeseries[:, :, t]
            valid = img[_valid_mask(img)]
            step_medians.append(float(np.median(valid)) if valid.size > 0 else 0.0)
        global_median = float(np.median(step_medians))
        ref_t = int(np.argmin([abs(m - global_median) for m in step_medians]))
        ref_img = timeseries[:, :, ref_t]
        ref_valid = ref_img[_valid_mask(ref_img)]
        ref_bin_centers, ref_cdf = _compute_cdf(ref_valid, n_bins, vmin, vmax)

    else:  # "mean" — pool all valid pixels
        all_valid = timeseries[_valid_mask(timeseries)]
        ref_bin_centers, ref_cdf = _compute_cdf(all_valid, n_bins, vmin, vmax)

    # ------------------------------------------------------------------
    # Match each time step to the reference distribution
    # ------------------------------------------------------------------
    matched = timeseries.astype(np.float32, copy=True)
    for t in range(T):
        src = timeseries[:, :, t]
        src_valid = src[_valid_mask(src)]
        if src_valid.size == 0:
            continue
        src_bin_centers, src_cdf = _compute_cdf(src_valid, n_bins, vmin, vmax)
        matched[:, :, t] = _match_single_to_ref_cdf(
            src, ref_bin_centers, ref_cdf, src_bin_centers, src_cdf
        )

    return matched
