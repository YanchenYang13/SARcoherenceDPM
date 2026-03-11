from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StackPairProduct:
    date_pair: str
    pair_dir: Path
    int_path: Path | None
    cor_path: Path | None


def _parse_isce_xml(xml_path: Path) -> tuple[int | None, int | None]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def find_value(key_candidates: list[str]) -> int | None:
        for elem in root.iter():
            name_attr = (elem.attrib.get("name") or "").lower()
            tag = (elem.tag or "").lower()
            if any(k in name_attr for k in key_candidates) or any(k in tag for k in key_candidates):
                for ch in list(elem):
                    if (ch.tag or "").lower().endswith("value") and ch.text is not None:
                        txt = ch.text.strip()
                        if txt.isdigit():
                            return int(txt)
                if elem.text and elem.text.strip().isdigit():
                    return int(elem.text.strip())
        return None

    width = find_value(["width"])
    length = find_value(["length", "lines", "nlines", "numberoflines", "height"])
    return width, length


def read_isce_int(int_path: str | Path, width: int | None = None, length: int | None = None) -> np.ndarray:
    int_path = Path(int_path)
    if width is None or length is None:
        xml_path = int_path.with_suffix(int_path.suffix + ".xml")
        if xml_path.exists():
            w, l = _parse_isce_xml(xml_path)
            width = width or w
            length = length or l

    if width is None:
        raise ValueError(f"Unable to infer width from XML for {int_path}. Please pass width explicitly.")

    filesize = os.path.getsize(int_path)
    bytes_per_pixel = np.dtype(np.complex64).itemsize
    if length is None:
        if filesize % (width * bytes_per_pixel) != 0:
            raise ValueError("File size is not divisible by width * complex64 itemsize.")
        length = filesize // (width * bytes_per_pixel)

    data = np.memmap(int_path, dtype=np.complex64, mode="r", shape=(length, width))
    return np.asarray(data)


def discover_stack_pair_products(stack_root: str | Path) -> list[StackPairProduct]:
    stack_root = Path(stack_root)
    pair_dirs: list[Path] = []
    for p in stack_root.rglob("*"):
        if not p.is_dir():
            continue
        if re.search(r"\d{8}_\d{8}", p.name):
            pair_dirs.append(p)

    products: list[StackPairProduct] = []
    for d in sorted(set(pair_dirs)):
        date_match = re.search(r"(\d{8}_\d{8})", d.name)
        if not date_match:
            continue
        date_pair = date_match.group(1)

        int_candidates = sorted([p for p in d.glob("*.int") if p.is_file()])
        cor_candidates = sorted([p for p in d.glob("*.cor") if p.is_file()])

        int_path = int_candidates[0] if int_candidates else None
        cor_path = cor_candidates[0] if cor_candidates else None
        if int_path is None and cor_path is None:
            continue
        products.append(StackPairProduct(date_pair=date_pair, pair_dir=d, int_path=int_path, cor_path=cor_path))

    return sorted(products, key=lambda x: x.date_pair)
