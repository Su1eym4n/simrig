"""Deterministic 360-degree target sets shared by reach previews and evaluation."""

from __future__ import annotations

import numpy as np


TARGET_COUNT = 30
TARGET_RADIUS_MIN = 0.22
TARGET_RADIUS_MAX = 0.32
TARGET_HEIGHT_MIN = 0.08
TARGET_HEIGHT_MAX = 0.24


def seeded_targets(seed: int, *, count: int = TARGET_COUNT) -> tuple[np.ndarray, ...]:
    """Scatter one target per angular sector around the robot base."""
    rng = np.random.default_rng(seed)
    sector_width = 2.0 * np.pi / count
    angles = (np.arange(count) + rng.uniform(0.15, 0.85, count)) * sector_width
    radii = rng.uniform(TARGET_RADIUS_MIN, TARGET_RADIUS_MAX, count)
    heights = rng.uniform(TARGET_HEIGHT_MIN, TARGET_HEIGHT_MAX, count)
    return tuple(
        np.asarray((radius * np.cos(angle), radius * np.sin(angle), height), dtype=float)
        for angle, radius, height in zip(angles, radii, heights, strict=True)
    )


def target_for_case(config: dict, set_name: str, case_seed: int) -> np.ndarray:
    """Resolve a reproducible target while retaining an independent reset seed."""
    try:
        target_set = config["target_sets"][set_name]
    except KeyError as exc:
        raise ValueError(f"Unknown target set: {set_name}") from exc
    count = int(target_set["count"])
    offset = int(target_set["case_seed_start"])
    index = (int(case_seed) - offset) % count
    return seeded_targets(int(target_set["generator_seed"]), count=count)[index]
