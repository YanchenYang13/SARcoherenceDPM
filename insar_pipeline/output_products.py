from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mintpy.utils.writefile import write_isce_file

from .io_utils import read_isce_file, write_array_to_isce


@dataclass
class OutputConfig:
    predict_dir: Path
    lat_file: Path
    lon_file: Path
    subset_params: str = "-l 42.625 42.635 -L 13.28 13.30"


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
