"""DAT file parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import re

from app.utils.models import ParseSettings, ParsedSnapshot


@dataclass
class DetectionResult:
    format_name: str
    grid_dims: tuple[int, int, int] | None
    has_coords: bool


def _read_lines(path: Path, max_lines: int = 200) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            if line.strip() == "":
                continue
            lines.append(line.strip())
    return lines


def inspect_header(path: str) -> dict[str, object]:
    lines = _read_lines(Path(path), max_lines=30)
    var_line = next((l for l in lines if l.startswith("VARIABLES=")), None)
    zone_line = next((l for l in lines if l.startswith("ZONE")), None)
    variables = _parse_variables_line(var_line) if var_line else []
    grid_shape = _parse_zone_dims(zone_line) if zone_line else None
    return {"variables": variables, "grid_shape": grid_shape, "has_zone": zone_line is not None}


def _parse_variables_line(line: str) -> list[str]:
    return re.findall(r"\"([^\"]+)\"", line)


def _parse_zone_dims(line: str) -> tuple[int, int, int] | None:
    m_i = re.search(r"I=(\\d+)", line)
    m_j = re.search(r"J=(\\d+)", line)
    m_k = re.search(r"K=(\\d+)", line)
    if m_i and m_j and m_k:
        return int(m_i.group(1)), int(m_j.group(1)), int(m_k.group(1))
    if m_i and m_j:
        return int(m_i.group(1)), int(m_j.group(1)), 1
    return None


def _find_var_index(vars_list: list[str], target: str) -> int | None:
    t = target.strip().lower()
    for i, name in enumerate(vars_list):
        n = name.strip().lower()
        if n == t or n.startswith(t):
            return i
    return None


def _split_numeric(line: str, delimiter: str) -> list[float] | None:
    if delimiter == " " or delimiter == "\t":
        parts = line.split()
    else:
        parts = line.split(delimiter)
    try:
        return [float(p) for p in parts if p != ""]
    except ValueError:
        return None


def detect_format(path: str, settings: ParseSettings) -> DetectionResult:
    lines = _read_lines(Path(path))
    numeric_rows = [row for row in ( _split_numeric(l, settings.delimiter) for l in lines) if row]
    if not numeric_rows:
        return DetectionResult("unknown", None, settings.has_coords)

    first = numeric_rows[0]
    if len(first) == 3 and all(float(x).is_integer() for x in first):
        dims = tuple(int(x) for x in first)
        return DetectionResult("grid_dims", dims, settings.has_coords)

    return DetectionResult("columns", None, settings.has_coords)


def _load_numeric_data(path: Path, settings: ParseSettings) -> np.ndarray:
    data = np.loadtxt(
        path,
        delimiter=None if settings.delimiter in (" ", "\t") else settings.delimiter,
        skiprows=settings.header_rows,
        dtype=float,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _parse_columns(data: np.ndarray, settings: ParseSettings) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    coords = None
    if settings.has_coords:
        coord_cols = settings.coord_cols
        coords = data[:, coord_cols]
    if settings.data_cols is None:
        start = 3 if settings.has_coords else 0
        data_cols = tuple(range(start, data.shape[1]))
    else:
        data_cols = settings.data_cols
    values = data[:, data_cols]
    variables = [f"var{i+1}" for i in range(values.shape[1])]
    return values, coords, variables


def _parse_grid_flat(path: Path, settings: ParseSettings) -> tuple[np.ndarray, np.ndarray | None, list[str], tuple[int, int, int]]:
    lines = _read_lines(path)
    numeric = [row for row in (_split_numeric(l, settings.delimiter) for l in lines) if row]
    if not numeric:
        raise ValueError("No numeric content found")
    dims_row = numeric[0]
    if len(dims_row) < 3:
        raise ValueError("Grid dims not found")
    nx, ny, nz = (int(dims_row[0]), int(dims_row[1]), int(dims_row[2]))
    grid_shape = (nx, ny, nz)
    values = np.loadtxt(
        path,
        delimiter=None if settings.delimiter in (" ", "\t") else settings.delimiter,
        skiprows=settings.header_rows + 1,
        dtype=float,
    ).reshape(-1)
    n_points = nx * ny * nz
    if values.size % n_points != 0:
        raise ValueError("Flattened values size is not divisible by grid points")
    n_vars = values.size // n_points
    data = values.reshape(n_points, n_vars)
    variables = [f"var{i+1}" for i in range(n_vars)]
    coords = None
    if settings.has_coords:
        xs = np.arange(nx)
        ys = np.arange(ny)
        zs = np.arange(nz)
        grid = np.meshgrid(xs, ys, zs, indexing="ij")
        coords = np.stack([g.reshape(-1) for g in grid], axis=1)
    return data, coords, variables, grid_shape


def parse_dat_file(path: str, settings: ParseSettings) -> ParsedSnapshot:
    path_obj = Path(path)
    header_lines = _read_lines(path_obj, max_lines=20)
    var_line = next((l for l in header_lines if l.startswith("VARIABLES=")), None)
    zone_line = next((l for l in header_lines if l.startswith("ZONE")), None)
    if var_line is not None:
        variables_all = _parse_variables_line(var_line)
        ncols = len(variables_all)
        grid_shape = _parse_zone_dims(zone_line) if zone_line else None
        data_rows: list[list[float]] = []
        started = False
        with path_obj.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not started:
                    if line.startswith("ZONE"):
                        started = True
                    continue
                parts = line.strip().split()
                if len(parts) != ncols:
                    if data_rows:
                        break
                    continue
                try:
                    data_rows.append([float(x) for x in parts])
                except ValueError:
                    if data_rows:
                        break
        data_raw = np.asarray(data_rows, dtype=float)
        coords = data_raw[:, settings.coord_cols] if settings.has_coords else None
        if settings.data_vars:
            idx = [variables_all.index(v) for v in settings.data_vars if v in variables_all]
            data_cols = tuple(idx)
        elif settings.data_cols is not None:
            data_cols = settings.data_cols
        else:
            iu = _find_var_index(variables_all, "U")
            iv = _find_var_index(variables_all, "V")
            iw = _find_var_index(variables_all, "W")
            if iu is not None and iv is not None and iw is not None:
                data_cols = (iu, iv, iw)
            else:
                start = 3 if settings.has_coords else 0
                data_cols = tuple(range(start, data_raw.shape[1]))
        values = data_raw[:, data_cols]
        variables = [variables_all[i] for i in data_cols]
        return ParsedSnapshot(data=values, coords=coords, grid_shape=grid_shape, variables=variables)

    detection = detect_format(path, settings)
    if detection.format_name == "grid_dims":
        data, coords, variables, grid_shape = _parse_grid_flat(path_obj, settings)
        return ParsedSnapshot(data=data, coords=coords, grid_shape=grid_shape, variables=variables)

    data_raw = _load_numeric_data(path_obj, settings)
    values, coords, variables = _parse_columns(data_raw, settings)
    return ParsedSnapshot(data=values, coords=coords, grid_shape=None, variables=variables)


def preview_table(path: str, settings: ParseSettings, max_rows: int = 10) -> np.ndarray:
    data = _load_numeric_data(Path(path), settings)
    return data[:max_rows, :]
