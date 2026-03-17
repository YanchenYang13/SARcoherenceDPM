from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from .io_utils import read_isce_file


@dataclass
class VisualizationConfig:
    input_file: Path
    output_file: Path | None = None
    mode: Literal["auto", "mintpy", "matplotlib"] = "auto"
    cmap: str = "turbo"
    vmin: float | None = None
    vmax: float | None = None
    mintpy_dataset: str | None = None
    nodisplay: bool = False


def _visualize_with_mintpy(config: VisualizationConfig) -> Path | None:
    from mintpy.cli import view

    args: list[str] = [str(config.input_file)]
    if config.mintpy_dataset:
        args.append(config.mintpy_dataset)
    if config.vmin is not None and config.vmax is not None:
        args += ["-v", str(config.vmin), str(config.vmax)]
    if config.output_file is not None:
        args += ["-o", str(config.output_file), "--save"]
    if config.nodisplay:
        args += ["--nodisplay"]

    view.main(args)
    return config.output_file


def _read_2d_array(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = np.squeeze(arr)
        return arr.astype(np.float32)
    if ext in {".tif", ".tiff"}:
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(f"Cannot open tif: {path}")
        return ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    arr = read_isce_file(path)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float32)


def _visualize_with_matplotlib(config: VisualizationConfig) -> Path | None:
    arr = _read_2d_array(config.input_file)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(arr, cmap=config.cmap, vmin=config.vmin, vmax=config.vmax)
    plt.colorbar(im)
    plt.title(config.input_file.name)
    plt.tight_layout()

    if config.output_file is not None:
        plt.savefig(config.output_file, dpi=200, bbox_inches="tight")

    if config.nodisplay:
        plt.close()
    else:
        plt.show()

    return config.output_file


def visualize_file(config: VisualizationConfig) -> Path | None:
    mode = config.mode
    if mode == "auto":
        if config.input_file.suffix.lower() in {".npy", ".tif", ".tiff", ".cor", ".rdr", ".full", ".int"}:
            mode = "matplotlib"
        else:
            mode = "mintpy"

    if mode == "mintpy":
        return _visualize_with_mintpy(config)
    return _visualize_with_matplotlib(config)
