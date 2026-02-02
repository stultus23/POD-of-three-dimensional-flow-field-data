"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

import numpy as np


@dataclass
class ParseSettings:
    delimiter: str = " "
    header_rows: int = 0
    has_coords: bool = True
    coord_cols: tuple[int, int, int] = (0, 1, 2)
    data_cols: tuple[int, ...] | None = None
    data_vars: tuple[str, ...] | None = None
    grid_dims: tuple[int, int, int] | None = None
    assume_row_major: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PodSettings:
    mean_subtract: bool = True
    normalize: bool = False
    n_modes: int | None = None
    method: str = "svd"  # svd or randomized
    dtype: str = "float32"
    random_state: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExportSettings:
    export_cloud: bool = True
    export_metrics: bool = True
    export_modes: bool = False
    export_coeffs: bool = False
    export_fft: bool = True
    export_eigs: bool = True
    sample_freq_hz: float | None = None
    export_recon: bool = False
    recon_modes: int = 5
    export_recon: bool = False
    recon_modes: int = 5
    export_csv: bool = True
    export_npy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedSnapshot:
    data: np.ndarray  # (n_points, n_vars)
    coords: np.ndarray | None
    grid_shape: tuple[int, int, int] | None
    variables: list[str]


@dataclass
class PodResult:
    modes: np.ndarray  # (ndofs, r)
    coeffs: np.ndarray  # (r, n_snap)
    singular_values: np.ndarray  # (r,)
    mean: np.ndarray  # (ndofs,)
    energy: np.ndarray  # (r,)
    cumulative_energy: np.ndarray  # (r,)
    snapshot_shape: tuple[int, int]
    coords: np.ndarray | None
    grid_shape: tuple[int, int, int] | None
    variables: list[str]


@dataclass
class ProjectState:
    file_paths: list[str] = field(default_factory=list)
    parse_settings: ParseSettings = field(default_factory=ParseSettings)
    pod_settings: PodSettings = field(default_factory=PodSettings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_paths": list(self.file_paths),
            "parse_settings": self.parse_settings.to_dict(),
            "pod_settings": self.pod_settings.to_dict(),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ProjectState":
        ps = ParseSettings(**payload.get("parse_settings", {}))
        pod = PodSettings(**payload.get("pod_settings", {}))
        return ProjectState(file_paths=payload.get("file_paths", []), parse_settings=ps, pod_settings=pod)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def ensure_1d(arr: Iterable[float]) -> np.ndarray:
    return np.asarray(arr).reshape(-1)
