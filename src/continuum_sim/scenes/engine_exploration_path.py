"""Config model for optional engine exploration paths."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


EXPLORATION_PATH_TYPES = ("polyline",)
EXPLORATION_PATH_FRAMES = ("world", "engine")


@dataclass(frozen=True)
class ExplorationPathConfig:
    """A displayable exploration path defined by ordered points in meters."""

    name: str
    type: str
    frame: str
    enabled: bool
    points_m: np.ndarray
    radius_m: float
    rgba: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("exploration path name must be non-empty.")
        if self.type not in EXPLORATION_PATH_TYPES:
            raise ValueError(
                f"exploration path type must be one of {EXPLORATION_PATH_TYPES}, "
                f"got {self.type!r}."
            )
        if self.frame not in EXPLORATION_PATH_FRAMES:
            raise ValueError(
                f"exploration path frame must be one of {EXPLORATION_PATH_FRAMES}, "
                f"got {self.frame!r}."
            )
        points = _points_array(self.points_m, f"{self.name}.points_m")
        if self.radius_m <= 0.0:
            raise ValueError(f"{self.name}.radius_m must be positive.")
        rgba = _rgba(self.rgba, f"{self.name}.rgba")
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "rgba", rgba)

    def with_points(self, points_m: object) -> "ExplorationPathConfig":
        """Return a validated copy with a new ordered point sequence."""

        return replace(self, points_m=_points_array(points_m, f"{self.name}.points_m"))


def load_exploration_paths(raw_values: object) -> list[ExplorationPathConfig]:
    """Parse optional exploration paths from an engine scene mapping."""

    if raw_values is None:
        return []
    if not isinstance(raw_values, list):
        raise ValueError("exploration_paths must be a list.")
    return [
        _load_exploration_path(index, raw_value)
        for index, raw_value in enumerate(raw_values)
    ]


def _load_exploration_path(index: int, raw_value: object) -> ExplorationPathConfig:
    if not isinstance(raw_value, dict):
        raise ValueError(f"exploration_paths[{index}] must be a mapping.")
    name = str(_required(raw_value, "name", index))
    return ExplorationPathConfig(
        name=name,
        type=str(raw_value.get("type", "polyline")),
        frame=str(raw_value.get("frame", "world")),
        enabled=bool(raw_value.get("enabled", True)),
        points_m=_points_array(
            _required(raw_value, "points_m", index),
            f"exploration_paths[{index}].points_m",
        ),
        radius_m=_positive_float(
            raw_value.get("radius_m", 0.008),
            f"exploration_paths[{index}].radius_m",
        ),
        rgba=_rgba(
            raw_value.get("rgba", (0.1, 1.0, 0.3, 0.8)),
            f"exploration_paths[{index}].rgba",
        ),
    )


def _points_array(raw_value: object, name: str) -> np.ndarray:
    points = np.asarray(raw_value, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"Expected {name} with shape (N, 3), got {points.shape}.")
    if points.shape[0] < 2:
        raise ValueError(f"{name} requires at least two points.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1.0e-12):
        raise ValueError(f"{name} contains a zero-length adjacent segment.")
    return points.copy()


def _positive_float(raw_value: object, name: str) -> float:
    value = float(raw_value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value.")
    return value


def _rgba(raw_value: object, name: str) -> tuple[float, float, float, float]:
    values = np.asarray(raw_value, dtype=float)
    if values.shape != (4,):
        raise ValueError(f"Expected {name} with shape (4,), got {values.shape}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} values must be finite and within [0, 1].")
    return tuple(float(value) for value in values)


def _required(values: dict, name: str, index: int) -> object:
    if name not in values:
        raise ValueError(f"exploration_paths[{index}] missing required field {name!r}.")
    return values[name]
