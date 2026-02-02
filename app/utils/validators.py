"""Validation helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app.utils.models import ParsedSnapshot


def validate_snapshots(snapshots: Sequence[ParsedSnapshot]) -> tuple[bool, str]:
    if not snapshots:
        return False, "No snapshots loaded."
    ref = snapshots[0].data.shape
    for i, snap in enumerate(snapshots[1:], start=2):
        if snap.data.shape != ref:
            return False, f"Snapshot {i} shape {snap.data.shape} does not match {ref}."
    return True, "OK"


def warn_large_matrix(ndofs: int, nsnaps: int, max_elements: int = 2_000_000_000) -> str | None:
    total = ndofs * nsnaps
    if total > max_elements:
        return f"Snapshot matrix is very large ({total} elements). Consider randomized SVD or fewer snapshots."
    return None
