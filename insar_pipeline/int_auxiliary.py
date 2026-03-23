from __future__ import annotations

"""Prepare additional coherence-like products derived from cropped INT files.

This module integrates two experiment-oriented products into the repository flow:

1. MintPy/ISCE ICU phase-sigma coherence computed from an unfiltered interferogram.
2. Under-amplitude quality maps derived from linear/circular phase variance.

It also provides a helper to convert filtered coherence to phase standard deviation
using the CRLB-style relationship requested for the paper experiments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .coherence import write_isce_bip_cor
from .io_utils import read_isce_cor, write_array_to_isce


@dataclass
class IntAuxiliaryConfig:
    """Configuration for cropped INT auxiliary product generation."""

    cropped_dir: Path
    corr_win: int = 5
    phsig_win: int = 5
    variance_win: int = 5
    variance_looks: float = 3.0
    variance_block_lines: int = 512
    output_var: bool = False


def _load_int_metadata(path: Path):
    import isceobj

    img = isceobj.createIntImage()
    img.load(str(path) + ".xml")
    width = img.getWidth()
    length = img.getLength()
    return img, length, width


def _open_int_memmap(path: Path, length: int, width: int) -> np.memmap:
    return np.memmap(path, dtype=np.complex64, mode="r", shape=(length, width))


def _create_float_output(path: Path, width: int, length: int):
    import isceobj

    out = isceobj.createImage()
    out.dataType = "FLOAT"
    out.bands = 1
    out.setFilename(str(path))
    out.setWidth(width)
    out.setAccessMode("write")
    out.createImage()
    mm = np.memmap(path, dtype=np.float32, mode="r+", shape=(length, width))
    return out, mm


def _finalize_output(out) -> None:
    out.renderHdr()
    out.finalizeImage()


def _box_mean_2d(arr: np.ndarray, win: int) -> np.ndarray:
    pad = win // 2
    arr_pad = np.pad(arr, ((pad, pad), (pad, pad)), mode="edge")
    integral = np.pad(arr_pad, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    integral = integral.cumsum(axis=0).cumsum(axis=1)

    h, w = arr.shape
    y0 = np.arange(0, h)
    y1 = y0 + win
    x0 = np.arange(0, w)
    x1 = x0 + win
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    return sums / float(win * win)


def _compute_linear_var_from_phase(phase: np.ndarray, win: int) -> np.ndarray:
    mean_phase = _box_mean_2d(phase, win)
    mean_phase2 = _box_mean_2d(phase * phase, win)
    var = np.maximum(mean_phase2 - mean_phase * mean_phase, 0.0)
    return var.astype(np.float32)


def _compute_circular_var_from_phase(phase: np.ndarray, win: int) -> np.ndarray:
    cos_phase = np.cos(phase).astype(np.float32)
    sin_phase = np.sin(phase).astype(np.float32)
    mean_cos = _box_mean_2d(cos_phase, win)
    mean_sin = _box_mean_2d(sin_phase, win)
    resultant = np.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)
    resultant = np.clip(resultant, 0.0, 1.0)
    circ_var = 1.0 - resultant
    return circ_var.astype(np.float32)


def _map_var_to_q(var_arr: np.ndarray, looks: float) -> np.ndarray:
    q = 1.0 / np.sqrt(1.0 + 2.0 * float(looks) * var_arr)
    return q.astype(np.float32)


def _date_pair_from_cropped_int(path: Path) -> str:
    name = path.name
    if name.endswith("_fine.int"):
        return name[: -len("_fine.int")]
    if name.endswith(".fine.int"):
        return name[: -len(".fine.int")]
    raise ValueError(f"Cannot infer date pair from cropped int filename: {path.name}")


def generate_unfiltered_phsig_coherence(
    int_path: Path,
    output_path: Path,
    corr_win: int = 5,
    phsig_win: int = 5,
) -> Path:
    """Compute ICU PHASESIGMA coherence from an unfiltered interferogram."""

    import isce  # noqa: F401  # required by the ISCE runtime
    import isceobj
    from mroipac.icu.Icu import Icu

    int_image = isceobj.createIntImage()
    int_image.load(str(int_path) + ".xml")
    int_image.setAccessMode("read")
    int_image.createImage()

    phsig_image = isceobj.createImage()
    phsig_image.dataType = "FLOAT"
    phsig_image.bands = 1
    phsig_image.setWidth(int_image.getWidth())
    phsig_image.setFilename(str(output_path))
    phsig_image.setAccessMode("write")
    phsig_image.createImage()

    icu_obj = Icu(name="icu_unfilt_phsig")
    icu_obj.configure()
    icu_obj.unwrappingFlag = False
    icu_obj.useAmplitudeFlag = False
    icu_obj.filteringFlag = False
    icu_obj.correlationType = "PHASESIGMA"
    icu_obj.correlationBoxSize = corr_win
    icu_obj.phaseSigmaBoxSize = phsig_win
    icu_obj.icu(intImage=int_image, phsigImage=phsig_image)

    phsig_image.renderHdr()
    int_image.finalizeImage()
    phsig_image.finalizeImage()
    return output_path


def generate_underamp_products(
    int_path: Path,
    q_out: Path,
    var_out: Path | None,
    mode: Literal["linear", "circular"],
    win: int = 5,
    looks: float = 3.0,
    block_lines: int = 512,
) -> tuple[Path, Path | None]:
    """Generate block-wise under-amplitude quality/variance products from INT."""

    if win < 1 or win % 2 == 0:
        raise ValueError("--win must be a positive odd integer.")
    if block_lines < 1:
        raise ValueError("--block-lines must be >= 1.")

    _, length, width = _load_int_metadata(int_path)
    int_mm = _open_int_memmap(int_path, length, width)

    q_out_img, q_out_mm = _create_float_output(q_out, width=width, length=length)
    var_out_img = None
    var_out_mm = None
    if var_out is not None:
        var_out_img, var_out_mm = _create_float_output(var_out, width=width, length=length)

    pad = win // 2
    for start in range(0, length, block_lines):
        end = min(start + block_lines, length)
        read_start = max(0, start - pad)
        read_end = min(length, end + pad)

        block_complex = np.array(int_mm[read_start:read_end, :], dtype=np.complex64)
        phase_block = np.angle(block_complex).astype(np.float32)
        if mode == "linear":
            var_block_full = _compute_linear_var_from_phase(phase_block, win)
        else:
            var_block_full = _compute_circular_var_from_phase(phase_block, win)

        core_start = start - read_start
        core_end = core_start + (end - start)
        var_core = var_block_full[core_start:core_end, :]

        if var_out_mm is not None:
            var_out_mm[start:end, :] = var_core
        q_out_mm[start:end, :] = _map_var_to_q(var_core, looks)

    q_out_mm.flush()
    _finalize_output(q_out_img)
    if var_out_mm is not None and var_out_img is not None:
        var_out_mm.flush()
        _finalize_output(var_out_img)

    return q_out, var_out


def coherence_to_phase_std(coh: np.ndarray, looks: float) -> np.ndarray:
    """Convert coherence to phase standard deviation using the CRLB-style formula."""

    epsilon = 1e-8
    denominator = 2.0 * float(looks) * np.maximum(coh.astype(np.float32) ** 2, epsilon)
    std = np.sqrt(np.maximum(1.0 - coh.astype(np.float32) ** 2, 0.0) / denominator)
    std = np.where(np.isnan(coh), np.nan, np.where(coh == 0, 0.0, std))
    return std.astype(np.float32)


def convert_filtered_coherence_to_std(cor_path: Path, out_path: Path, looks: float = 3.0) -> Path:
    """Convert filtered coherence to filtered phase standard deviation."""

    coh = read_isce_cor(cor_path)
    std = coherence_to_phase_std(coh, looks=looks)
    write_array_to_isce(std, out_path)
    return out_path


def prepare_int_auxiliary_products(config: IntAuxiliaryConfig) -> list[Path]:
    """Generate experiment products for all cropped unfiltered INT files."""

    config.cropped_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for int_path in sorted(config.cropped_dir.glob("*_fine.int")):
        date_pair = _date_pair_from_cropped_int(int_path)

        phsig_out = config.cropped_dir / f"{date_pair}_unfilt_fine.cor"
        if not phsig_out.exists():
            generate_unfiltered_phsig_coherence(
                int_path,
                phsig_out,
                corr_win=config.corr_win,
                phsig_win=config.phsig_win,
            )
        outputs.append(phsig_out)

        linear_q_out = config.cropped_dir / f"{date_pair}_underamp_unfilt_fine.cor"
        linear_var_out = config.cropped_dir / f"{date_pair}_underamp_unfilt_fine.var" if config.output_var else None
        if not linear_q_out.exists():
            generate_underamp_products(
                int_path,
                q_out=linear_q_out,
                var_out=linear_var_out,
                mode="linear",
                win=config.variance_win,
                looks=config.variance_looks,
                block_lines=config.variance_block_lines,
            )
        outputs.append(linear_q_out)
        if linear_var_out is not None:
            outputs.append(linear_var_out)

        circ_q_out = config.cropped_dir / f"{date_pair}_underamp_unfilt_fine_circ.cor"
        circ_var_out = config.cropped_dir / f"{date_pair}_underamp_unfilt_fine_circ.var" if config.output_var else None
        if not circ_q_out.exists():
            generate_underamp_products(
                int_path,
                q_out=circ_q_out,
                var_out=circ_var_out,
                mode="circular",
                win=config.variance_win,
                looks=config.variance_looks,
                block_lines=config.variance_block_lines,
            )
        outputs.append(circ_q_out)
        if circ_var_out is not None:
            outputs.append(circ_var_out)

    for cor_path in sorted(config.cropped_dir.glob("*_filt_fine.cor")):
        std_path = cor_path.with_name(cor_path.name.replace("filt_fine.cor", "filt_fine.std"))
        if not std_path.exists():
            convert_filtered_coherence_to_std(cor_path, std_path, looks=config.variance_looks)
        outputs.append(std_path)

    return outputs
