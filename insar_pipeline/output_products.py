from __future__ import annotations

"""Utilities for score geocoding/export and optional threshold-based TIFF masking.

This module has two responsibilities:
1) Convert score `.npy` maps into geocoded/subsetted final GeoTIFF products.
2) Optionally apply post-processing threshold masks on generated/selected TIFFs.

The masking logic is intentionally post-output, so it does not alter model training,
prediction, or score computation behavior.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from mintpy.utils.writefile import write_isce_file
from osgeo import gdal

from .io_utils import read_isce_file, write_array_to_isce


@dataclass
class OutputConfig:
    """Runtime configuration for geocoding and raster export."""
    predict_dir: Path
    lat_file: Path
    lon_file: Path
    subset_params: str = "-l 42.625 42.635 -L 13.28 13.30"


@dataclass
class ThresholdMaskConfig:
    """Configuration for threshold masking on TIFF files.

    method:
        - manual: keep pixels >= manual_threshold
        - quantile: threshold from quantile(valid_values)
        - std: threshold = mean(valid_values) + std_n * std(valid_values)
    """
    method: Literal["manual", "quantile", "std"] = "quantile"
    manual_threshold: float | None = None
    quantile: float = 0.70
    std_n: float = 2.0
    output_suffix: str = "mask"


def _build_base_name(score_file: Path) -> str:
    # score.npy -> score ; xxx_score.npy -> xxx
    name = score_file.name
    if name == "score.npy":
        return "score"
    if name.endswith("score.npy"):
        return name[: -len("score.npy")].rstrip("._-") or "score"
    return score_file.stem




def _convert_npy_files_to_float32(directory: Path) -> None:
    for npy_file in sorted(directory.glob("*.npy")):
        if not npy_file.is_file():
            continue
        data = np.load(npy_file)
        if np.issubdtype(data.dtype, np.floating) and data.dtype != np.float32:
            np.save(npy_file, data.astype(np.float32))



def _convert_rdr_file_to_float32(file_path: Path) -> None:
    data = read_isce_file(file_path)
    if np.issubdtype(data.dtype, np.floating) and data.dtype != np.float32:
        write_array_to_isce(data.astype(np.float32), file_path)


def _prepare_geocode_inputs(config: OutputConfig) -> None:
    _convert_npy_files_to_float32(config.predict_dir)
    _convert_rdr_file_to_float32(config.lat_file)
    _convert_rdr_file_to_float32(config.lon_file)


def generate_geocoded_outputs(config: OutputConfig) -> list[Path]:
    _prepare_geocode_inputs(config)

    outputs: list[Path] = []
    score_files = sorted([p for p in config.predict_dir.glob("*score.npy") if p.is_file()])

    for score_file in score_files:
        data = np.load(score_file).astype(np.float32)
        # keep geocoding stable: replace NaN/Inf in score map before writing
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        base_name = _build_base_name(score_file)

        cor_file = config.predict_dir / f"{base_name}.cor"
        write_isce_file(data=data, out_file=str(cor_file), file_type="isce_cor")

        geocode_cmd = (
            f"geocode.py {cor_file} --lat-file {config.lat_file} "
            f"--lon-file {config.lon_file} --outdir {config.predict_dir}"
        )
        subprocess.run(geocode_cmd, shell=True, check=True)

        geo_cor_file = config.predict_dir / f"geo_{base_name}.cor"
        subset_cor_file = config.predict_dir / f"{base_name}final.cor"
        subset_cmd = f"subset.py {geo_cor_file} {config.subset_params} --output {subset_cor_file}"
        subprocess.run(subset_cmd, shell=True, check=True)

        tif_file = config.predict_dir / f"{base_name}final.tif"
        gdal_cmd = f"save_gdal.py {subset_cor_file} --output {tif_file}"
        subprocess.run(gdal_cmd, shell=True, check=True)

        outputs.append(tif_file)

    return outputs


def _compute_threshold(valid_values: np.ndarray, config: ThresholdMaskConfig) -> tuple[float, float, float]:
    """Compute a scalar threshold from valid pixel values and mask settings.

    Returns `(threshold, mean, std)` to support consistent logging.
    """
    mean = float(np.mean(valid_values))
    std = float(np.std(valid_values))

    if config.method == "manual":
        if config.manual_threshold is None:
            raise ValueError("manual threshold mask requires --mask-threshold-manual")
        threshold = float(config.manual_threshold)
    elif config.method == "quantile":
        q = float(config.quantile)
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"quantile must be within [0, 1], got {q}")
        threshold = float(np.quantile(valid_values, q))
    else:
        threshold = mean + float(config.std_n) * std

    return threshold, mean, std


def apply_threshold_mask_to_tif(tif_path: Path, config: ThresholdMaskConfig) -> Path:
    """Apply threshold mask to one TIFF and write a new GeoTIFF.

    The output keeps source georeferencing/projection metadata and writes nodata for
    all pixels that are invalid or below the selected threshold.
    """
    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open tif: {tif_path}")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    if arr is None:
        raise RuntimeError(f"Cannot read raster band from: {tif_path}")

    nodata = band.GetNoDataValue()
    if nodata is None:
        nodata = 0.0

    arr = arr.astype(np.float32, copy=False)
    valid_mask = np.isfinite(arr) & (arr != nodata)
    valid_values = arr[valid_mask]
    if valid_values.size == 0:
        raise ValueError(f"No valid pixels found in tif: {tif_path}")

    threshold, mean, std = _compute_threshold(valid_values, config)
    print(
        f"[mask] {tif_path.name}: method={config.method}, "
        f"mean={mean:.6f}, std={std:.6f}, threshold={threshold:.6f}"
    )

    result = np.full_like(arr, fill_value=float(nodata), dtype=np.float32)
    keep_mask = valid_mask & (arr >= threshold)
    result[keep_mask] = arr[keep_mask]

    output_tif = tif_path.with_name(f"{tif_path.stem}_{config.output_suffix}.tif")
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(str(output_tif), ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
    if out_ds is None:
        raise RuntimeError(f"Cannot create output tif: {output_tif}")

    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(result)
    out_band.SetNoDataValue(float(nodata))
    out_band.FlushCache()
    out_ds.FlushCache()

    out_band = None
    out_ds = None
    band = None
    ds = None
    return output_tif
