"""Engine surface patch scaffold for future cleaning-path generation.

This module provides simple local surface patches and sampling utilities. It
does not perform CAD mesh planning or connect to the MuJoCo runtime. The
surface frame is right-handed: `tangent_v = normal x tangent_u` after
orthogonalizing `tangent_u` against `normal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    position_vector as _position_vector,
    positive_float_value as _positive_float_value,
    required as _required,
    section as _section,
)


SURFACE_PATCH_TYPES: tuple[str, ...] = (
    "plane_patch",
    "sphere_patch",
    "annotated_mesh_patch",
)


@dataclass(frozen=True)
class SurfaceFrame:
    """Right-handed local frame attached to a sampled patch point."""

    center: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray


@dataclass(frozen=True)
class EngineSurfacePatchConfig:
    """Lightweight local patch description for cleaning-path generation."""

    name: str
    type: str
    center: np.ndarray | None = None
    patch_center: np.ndarray | None = None
    normal: np.ndarray | None = None
    tangent_u: np.ndarray | None = None
    size_u_m: float = 0.0
    size_v_m: float = 0.0
    radius_m: float | None = None
    sphere_center: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_surface_patch_config(data_or_path: dict[str, object] | str | Path) -> EngineSurfacePatchConfig:
    """Load a patch config from a mapping or task YAML path."""

    if isinstance(data_or_path, str | Path):
        raw = load_yaml(Path(data_or_path).resolve())
        task_raw = _section(raw, "task")
        patch_raw = _section(task_raw, "surface_patch")
    elif isinstance(data_or_path, dict):
        patch_raw = data_or_path
    else:
        raise ValueError("surface patch input must be a mapping or YAML path.")

    patch = EngineSurfacePatchConfig(
        name=str(_required(patch_raw, "name")),
        type=str(_required(patch_raw, "type")),
        center=_optional_position(patch_raw.get("center")),
        patch_center=_optional_position(patch_raw.get("patch_center")),
        normal=_optional_position(patch_raw.get("normal")),
        tangent_u=_optional_position(patch_raw.get("tangent_u")),
        size_u_m=_positive_float_value(_required(patch_raw, "size_u_m"), "surface_patch.size_u_m"),
        size_v_m=_positive_float_value(_required(patch_raw, "size_v_m"), "surface_patch.size_v_m"),
        radius_m=_optional_positive_float(patch_raw.get("radius_m"), "surface_patch.radius_m"),
        sphere_center=_optional_position(patch_raw.get("sphere_center")),
        metadata=_optional_metadata(patch_raw.get("metadata")),
    )
    validate_surface_patch_config(patch)
    return patch


def validate_surface_patch_config(patch: EngineSurfacePatchConfig) -> None:
    """Validate patch fields and frame-defining vectors."""

    if patch.type not in SURFACE_PATCH_TYPES:
        raise ValueError(f"Unknown patch type {patch.type!r}; expected one of {SURFACE_PATCH_TYPES}.")
    if patch.size_u_m <= 0.0:
        raise ValueError(f"surface patch size_u_m must be positive, got {patch.size_u_m}.")
    if patch.size_v_m <= 0.0:
        raise ValueError(f"surface patch size_v_m must be positive, got {patch.size_v_m}.")
    if patch.normal is None:
        raise ValueError("surface patch normal is required.")
    if patch.tangent_u is None:
        raise ValueError("surface patch tangent_u is required.")
    if patch.type == "plane_patch":
        if patch.center is None:
            raise ValueError("plane_patch requires center.")
    elif patch.type == "sphere_patch":
        if patch.patch_center is None:
            raise ValueError("sphere_patch requires patch_center.")
        if patch.radius_m is None:
            raise ValueError("sphere_patch requires radius_m.")
        if patch.sphere_center is None:
            raise ValueError("sphere_patch requires sphere_center.")
    elif patch.type == "annotated_mesh_patch":
        if patch.patch_center is None and patch.center is None:
            raise ValueError("annotated_mesh_patch requires patch_center or center.")

    _orthonormalize_frame(patch.normal, patch.tangent_u)


def surface_frame_from_patch(patch: EngineSurfacePatchConfig) -> SurfaceFrame:
    """Return the orthonormal surface frame for a patch center."""

    validate_surface_patch_config(patch)
    normal, tangent_u, tangent_v = _orthonormalize_frame(patch.normal, patch.tangent_u)
    center = patch.center if patch.center is not None else patch.patch_center
    if center is None:
        raise ValueError("surface patch is missing a center.")
    return SurfaceFrame(center=center.copy(), normal=normal, tangent_u=tangent_u, tangent_v=tangent_v)


def sample_surface_point(patch: EngineSurfacePatchConfig, u: float, v: float) -> np.ndarray:
    """Sample one point using normalized patch coordinates in `[-0.5, 0.5]`."""

    frame = surface_frame_from_patch(patch)
    offset = frame.tangent_u * (u * patch.size_u_m) + frame.tangent_v * (v * patch.size_v_m)
    if patch.type == "plane_patch":
        return frame.center + offset
    if patch.type == "sphere_patch":
        if patch.sphere_center is None or patch.radius_m is None:
            raise ValueError("sphere_patch requires sphere_center and radius_m.")
        candidate = frame.center + offset
        direction = candidate - patch.sphere_center
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            raise ValueError("sphere_patch sample collapsed onto sphere_center.")
        return patch.sphere_center + (direction / norm) * patch.radius_m
    if patch.type == "annotated_mesh_patch":
        return frame.center + offset
    raise ValueError(f"Unknown patch type {patch.type!r}.")


def sample_surface_grid(patch: EngineSurfacePatchConfig, num_u: int, num_v: int) -> np.ndarray:
    """Sample a `num_v x num_u x 3` grid across the patch."""

    if num_u <= 0:
        raise ValueError(f"num_u must be positive, got {num_u}.")
    if num_v <= 0:
        raise ValueError(f"num_v must be positive, got {num_v}.")
    u_values = np.linspace(-0.5, 0.5, num_u)
    v_values = np.linspace(-0.5, 0.5, num_v)
    grid = np.zeros((num_v, num_u, 3), dtype=float)
    for row_index, v in enumerate(v_values):
        for col_index, u in enumerate(u_values):
            grid[row_index, col_index] = sample_surface_point(patch, float(u), float(v))
    return grid


def _orthonormalize_frame(
    normal: np.ndarray | None,
    tangent_u: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if normal is None or tangent_u is None:
        raise ValueError("normal and tangent_u are required for a surface frame.")
    unit_normal = _normalize_vector(normal, "normal")
    raw_tangent = np.asarray(tangent_u, dtype=float)
    projected_tangent = raw_tangent - np.dot(raw_tangent, unit_normal) * unit_normal
    tangent_norm = float(np.linalg.norm(projected_tangent))
    if tangent_norm <= 1.0e-8:
        raise ValueError("tangent_u is parallel or nearly parallel to normal.")
    unit_tangent_u = projected_tangent / tangent_norm
    unit_tangent_v = np.cross(unit_normal, unit_tangent_u)
    return unit_normal, unit_tangent_u, _normalize_vector(unit_tangent_v, "tangent_v")


def _normalize_vector(vector: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected {name} with shape (3,), got {array.shape}.")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must have non-zero length.")
    return array / norm


def _optional_position(raw_value: object) -> np.ndarray | None:
    if raw_value is None:
        return None
    return _position_vector(raw_value, "surface_patch.vector")


def _optional_positive_float(raw_value: object, name: str) -> float | None:
    if raw_value is None:
        return None
    return _positive_float_value(raw_value, name)


def _optional_metadata(raw_value: object) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError("surface_patch.metadata must be a mapping.")
    return dict(raw_value)
