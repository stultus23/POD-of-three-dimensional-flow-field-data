"""Project state save/load."""

from __future__ import annotations

import json
from pathlib import Path

from app.utils.models import ProjectState


def save_project(path: str, state: ProjectState) -> None:
    payload = state.to_dict()
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_project(path: str) -> ProjectState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectState.from_dict(payload)
