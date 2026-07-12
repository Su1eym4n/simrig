"""Path helpers for SimRig projects and external model checkouts."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIRS = ("runs", "reports", "artifacts", "envs", "configs")


def ensure_project_dirs(root: Path | str = ".") -> list[Path]:
    base = Path(root)
    paths = [base / name for name in PROJECT_DIRS]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def menagerie_candidates(extra: Path | str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if extra is not None:
        candidates.append(Path(extra).expanduser())
    env_path = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "mujoco_menagerie",
            cwd.parent / "mujoco_menagerie",
            Path.home() / "Desktop" / "mujoco_menagerie",
        ]
    )
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def find_menagerie(extra: Path | str | None = None) -> Path:
    for candidate in menagerie_candidates(extra):
        if (candidate / "README.md").is_file() and any(candidate.glob("*/scene*.xml")):
            return candidate
    searched = "\n".join(str(path) for path in menagerie_candidates(extra))
    raise FileNotFoundError(
        "Could not find a MuJoCo Menagerie checkout. Set MUJOCO_MENAGERIE_PATH "
        f"or pass --menagerie. Searched:\n{searched}"
    )

