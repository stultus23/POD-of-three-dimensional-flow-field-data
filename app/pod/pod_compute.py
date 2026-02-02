"""POD computation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.utils.models import PodResult, PodSettings, ParsedSnapshot


def build_snapshot_matrix(snapshots: list[ParsedSnapshot], dtype: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray | None, tuple[int, int, int] | None, list[str]]:
    data = [s.data.reshape(-1) for s in snapshots]
    X = np.stack(data, axis=1).astype(dtype, copy=False)
    mean = X.mean(axis=1)
    coords = snapshots[0].coords
    grid_shape = snapshots[0].grid_shape
    variables = snapshots[0].variables
    return X, mean, X.shape, coords, grid_shape, variables


def compute_pod(snapshots: list[ParsedSnapshot], settings: PodSettings) -> PodResult:
    X, mean, shape, coords, grid_shape, variables = build_snapshot_matrix(snapshots, settings.dtype)
    if settings.mean_subtract:
        X = X - mean[:, None]

    if settings.normalize:
        scale = np.linalg.norm(X, axis=0)
        scale[scale == 0] = 1.0
        X = X / scale

    n_modes = settings.n_modes
    if n_modes is None:
        n_modes = min(X.shape)
    n_modes = max(1, min(n_modes, min(X.shape)))

    if settings.method == "randomized":
        try:
            from sklearn.utils.extmath import randomized_svd

            U, S, VT = randomized_svd(X, n_components=n_modes, random_state=settings.random_state)
        except Exception:
            U, S, VT = np.linalg.svd(X, full_matrices=False)
            U, S, VT = U[:, :n_modes], S[:n_modes], VT[:n_modes, :]
    else:
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        U, S, VT = U[:, :n_modes], S[:n_modes], VT[:n_modes, :]

    coeffs = (S[:, None] * VT)
    energy = (S ** 2) / (S ** 2).sum()
    cumulative_energy = np.cumsum(energy)

    return PodResult(
        modes=U,
        coeffs=coeffs,
        singular_values=S,
        mean=mean,
        energy=energy,
        cumulative_energy=cumulative_energy,
        snapshot_shape=shape,
        coords=coords,
        grid_shape=grid_shape,
        variables=variables,
    )
