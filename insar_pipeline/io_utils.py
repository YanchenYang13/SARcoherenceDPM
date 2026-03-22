from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from mintpy.utils.writefile import write_isce_file
from osgeo import gdal


def bbox_to_sar_indices(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    lat_data: np.ndarray,
    lon_data: np.ndarray,
) -> list[int]:
    """Convert geographic bbox to rounded SAR row/col indices."""
    data_map = (
        (lon_data >= lon_min)
        & (lon_data <= lon_max)
        & (lat_data >= lat_min)
        & (lat_data <= lat_max)
    )
    region_list = np.argwhere(data_map)
    if region_list.size == 0:
        raise ValueError("No SAR pixels found inside requested geographic bounding box.")
    return [
        10 * math.floor(region_list[:, 0].min() / 10),
        10 * math.ceil(region_list[:, 0].max() / 10),
        10 * math.floor(region_list[:, 1].min() / 10),
        10 * math.ceil(region_list[:, 1].max() / 10),
    ]


def _open_raster(file_path: str | Path):
    file_path = str(file_path)
    ds = gdal.Open(file_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Unable to open file: {file_path}")
    return ds


def read_isce_file(file_path: str | Path) -> np.ndarray:
    """Read ISCE/GDAL raster as (H, W, 1) array with native numeric type.

    Notes:
    - `.unw` uses band 2 for phase.
    - `.cor` prefers coherence band (band 2) if present, otherwise band 1.
    - `.full` preserves all bands if more than one band is present.
    - `.int` keeps complex type (e.g., CFloat32) if present.
    """
    ds = _open_raster(file_path)
    ext = Path(file_path).suffix

    if ext == ".full" and ds.RasterCount > 1:
        bands = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)]
        return np.stack(bands, axis=2)

    if ext == ".unw":
        band_index = 2
    elif ext == ".cor" and ds.RasterCount >= 2:
        band_index = 2
    else:
        band_index = 1

    band = ds.GetRasterBand(band_index)
    if band is None:
        raise RuntimeError(f"Unable to read raster band from: {file_path}")

    data = np.expand_dims(band.ReadAsArray(), 2)
    return data


def read_isce_cor(file_path: str | Path, return_amp: bool = False):
    """Read ISCE coherence file.

    If the .cor file is BIP with 2 bands, band 1 is amplitude proxy and band 2 is coherence.
    For single-band files, coherence is read from band 1 and amplitude is returned as ones.
    """
    ds = _open_raster(file_path)
    if ds.RasterCount >= 2:
        amp = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        coh = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    else:
        coh = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        amp = np.ones_like(coh, dtype=np.float32)

    if return_amp:
        return amp, coh
    return coh




def _numpy_to_gdal_dtype(arr: np.ndarray):
    dt = arr.dtype
    if dt == np.uint8:
        return gdal.GDT_Byte
    if dt == np.int16:
        return gdal.GDT_Int16
    if dt == np.uint16:
        return gdal.GDT_UInt16
    if dt == np.int32:
        return gdal.GDT_Int32
    if dt == np.uint32:
        return gdal.GDT_UInt32
    if dt == np.float32:
        return gdal.GDT_Float32
    if dt == np.float64:
        return gdal.GDT_Float64
    if dt == np.complex64:
        return gdal.GDT_CFloat32
    if dt == np.complex128:
        return gdal.GDT_CFloat64
    raise ValueError(f"Unsupported numpy dtype for GDAL write: {dt}")


def write_gdal_file(arr: np.ndarray, output_filepath: str | Path, data_type=None) -> None:
    """Write a 2D/3D array to ENVI format using GDAL while preserving dtype/bands."""
    output_filepath = str(output_filepath)
    if arr.ndim == 2:
        rows, cols = arr.shape
        bands = 1
    elif arr.ndim == 3:
        rows, cols, bands = arr.shape
    else:
        raise ValueError("Array must be 2D or 3D.")

    if data_type is None:
        data_type = _numpy_to_gdal_dtype(arr)

    driver = gdal.GetDriverByName("ENVI")
    dataset = driver.Create(output_filepath, cols, rows, bands, data_type)
    if dataset is None:
        raise RuntimeError(f"Could not create file: {output_filepath}")

    if arr.ndim == 2:
        dataset.GetRasterBand(1).WriteArray(arr)
    else:
        for i in range(bands):
            dataset.GetRasterBand(i + 1).WriteArray(arr[:, :, i])

    dataset.FlushCache()


def build_vrt_for_output(output_filepath: str | Path) -> Path:
    """Build companion .vrt file for GDAL-readable outputs."""
    output_filepath = str(output_filepath)
    ds = gdal.Open(output_filepath, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot reopen output for VRT build: {output_filepath}")

    vrt_path = Path(output_filepath + ".vrt")
    vrt_driver = gdal.GetDriverByName("VRT")
    vrt_ds = vrt_driver.CreateCopy(str(vrt_path), ds)
    vrt_ds = None
    ds = None
    return vrt_path


def write_array_to_isce(arr: np.ndarray, output_filepath: str | Path) -> None:
    """Write array to ISCE or ENVI file according to extension."""
    output_filepath = str(output_filepath)
    ext = Path(output_filepath).suffix
    type_map = {
        ".unw": "MOD_isce_unw",
        ".cor": "isce_cor",
        ".int": "isce_int",
        ".full": "envi",
    }
    if ext == ".rdr":
        # Keep geolocation rasters as single-band Float32 ENVI datasets.
        write_gdal_file(np.asarray(arr, dtype=np.float32), output_filepath, data_type=gdal.GDT_Float32)
        return

    if ext == ".std":
        # Phase-standard-deviation rasters are stored as single-band Float32 ENVI datasets.
        write_gdal_file(np.asarray(arr, dtype=np.float32), output_filepath, data_type=gdal.GDT_Float32)
        return

    if ext == ".full":
        # Preserve numeric type for .full products (e.g., CFloat32 SLC).
        write_gdal_file(np.asarray(arr), output_filepath)
        build_vrt_for_output(output_filepath)
        return
    if ext not in type_map:
        raise ValueError(f"Unsupported file extension: {ext}")

    if arr.ndim == 2:
        data = arr
    elif arr.ndim == 3:
        data = arr[:, :, 0]
    else:
        raise ValueError("Array must be 2D or 3D.")

    write_isce_file(data=data, out_file=output_filepath, file_type=type_map[ext])
