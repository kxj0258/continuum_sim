"""Primitive collision hint parsing for engine scene diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PRIMITIVE_COLLISION_TYPES = ("capsule", "cylinder", "sphere", "box")
PRIMITIVE_COLLISION_FRAMES = ("world", "engine", "mesh")


@dataclass(frozen=True)
class PrimitiveBBox:
    """Axis-aligned bbox for a primitive hint in its declared frame."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(self.maximum[index] - self.minimum[index] for index in range(3))  # type: ignore[return-value]

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((self.minimum[index] + self.maximum[index]) * 0.5 for index in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class PrimitiveCollisionGeomConfig:
    """Config for an optional primitive collision/nozzle hint."""

    name: str
    type: str
    enabled: bool
    frame: str
    position_m: np.ndarray | None = None
    quat_wxyz: np.ndarray | None = None
    fromto_m: np.ndarray | None = None
    radius_m: float | None = None
    length_m: float | None = None
    size_m: np.ndarray | None = None
    rgba: tuple[float, float, float, float] | None = None
    note: str | None = None


def load_primitive_collision_geoms(raw_values: object) -> list[PrimitiveCollisionGeomConfig]:
    """Parse optional primitive collision hints from a scene YAML section."""

    if raw_values is None:
        return []
    if not isinstance(raw_values, list):
        raise ValueError("primitive_collision_geoms must be a list.")
    return [_load_primitive_collision_geom(index, raw_value) for index, raw_value in enumerate(raw_values)]


def iter_primitive_collision_geoms(
    geoms: Iterable[PrimitiveCollisionGeomConfig],
) -> Iterable[PrimitiveCollisionGeomConfig]:
    """Iterate primitive collision hints."""

    yield from geoms


def validate_primitive_collision_geoms(geoms: Iterable[PrimitiveCollisionGeomConfig]) -> None:
    """Validate parsed primitive collision hints."""

    for geom in geoms:
        _validate_primitive_collision_geom(geom)


def primitive_geom_bbox(geom: PrimitiveCollisionGeomConfig) -> PrimitiveBBox:
    """Compute a conservative axis-aligned bbox for a primitive hint."""

    _validate_primitive_collision_geom(geom)
    if geom.type == "capsule":
        if geom.fromto_m is not None:
            points = geom.fromto_m.reshape(2, 3)
            radius = float(geom.radius_m)
            minimum = tuple(float(np.min(points[:, axis]) - radius) for axis in range(3))
            maximum = tuple(float(np.max(points[:, axis]) + radius) for axis in range(3))
            return PrimitiveBBox(minimum=minimum, maximum=maximum)
        return _centered_bbox(geom.position_m, (geom.radius_m, geom.radius_m, geom.length_m * 0.5 + geom.radius_m))
    if geom.type == "cylinder":
        return _centered_bbox(geom.position_m, (geom.radius_m, geom.radius_m, geom.length_m * 0.5))
    if geom.type == "sphere":
        return _centered_bbox(geom.position_m, (geom.radius_m, geom.radius_m, geom.radius_m))
    if geom.type == "box":
        return _centered_bbox(geom.position_m, tuple(float(value) * 0.5 for value in geom.size_m))
    raise ValueError(f"primitive_collision_geoms type must be one of {PRIMITIVE_COLLISION_TYPES}.")


def _load_primitive_collision_geom(index: int, raw_value: object) -> PrimitiveCollisionGeomConfig:
    if not isinstance(raw_value, dict):
        raise ValueError(f"primitive_collision_geoms[{index}] must be a mapping.")
    name = str(_required(raw_value, "name", index))
    geom_type = str(_required(raw_value, "type", index))
    enabled = bool(raw_value.get("enabled", False))
    frame = str(raw_value.get("frame", "world"))
    geom = PrimitiveCollisionGeomConfig(
        name=name,
        type=geom_type,
        enabled=enabled,
        frame=frame,
        position_m=_optional_array(raw_value.get("position_m"), (3,), f"primitive_collision_geoms[{index}].position_m"),
        quat_wxyz=_optional_array(raw_value.get("quat_wxyz"), (4,), f"primitive_collision_geoms[{index}].quat_wxyz"),
        fromto_m=_optional_array(raw_value.get("fromto_m"), (6,), f"primitive_collision_geoms[{index}].fromto_m"),
        radius_m=_optional_positive_float(raw_value.get("radius_m"), f"primitive_collision_geoms[{index}].radius_m"),
        length_m=_optional_positive_float(raw_value.get("length_m"), f"primitive_collision_geoms[{index}].length_m"),
        size_m=_optional_array(raw_value.get("size_m"), (3,), f"primitive_collision_geoms[{index}].size_m"),
        rgba=_optional_rgba(raw_value.get("rgba"), f"primitive_collision_geoms[{index}].rgba"),
        note=str(raw_value["note"]) if "note" in raw_value else None,
    )
    _validate_primitive_collision_geom(geom)
    return geom


def _validate_primitive_collision_geom(geom: PrimitiveCollisionGeomConfig) -> None:
    if not geom.name:
        raise ValueError("primitive_collision_geoms name must be non-empty.")
    if geom.type not in PRIMITIVE_COLLISION_TYPES:
        raise ValueError(
            f"primitive_collision_geoms type must be one of {PRIMITIVE_COLLISION_TYPES}, got {geom.type!r}."
        )
    if geom.frame not in PRIMITIVE_COLLISION_FRAMES:
        raise ValueError(
            f"primitive_collision_geoms frame must be one of {PRIMITIVE_COLLISION_FRAMES}, got {geom.frame!r}."
        )
    if geom.type == "capsule":
        has_fromto = geom.fromto_m is not None and geom.radius_m is not None
        has_pose = (
            geom.position_m is not None
            and geom.quat_wxyz is not None
            and geom.radius_m is not None
            and geom.length_m is not None
        )
        if not has_fromto and not has_pose:
            raise ValueError(
                "capsule primitive_collision_geoms require fromto_m + radius_m "
                "or position_m + quat_wxyz + radius_m + length_m."
            )
    elif geom.type == "cylinder":
        if geom.position_m is None or geom.radius_m is None or geom.length_m is None:
            raise ValueError("cylinder primitive_collision_geoms require position_m + radius_m + length_m.")
    elif geom.type == "sphere":
        if geom.position_m is None or geom.radius_m is None:
            raise ValueError("sphere primitive_collision_geoms require position_m + radius_m.")
    elif geom.type == "box":
        if geom.position_m is None or geom.size_m is None:
            raise ValueError("box primitive_collision_geoms require position_m + size_m.")
        if np.any(geom.size_m <= 0.0):
            raise ValueError("box primitive_collision_geoms size_m values must be positive.")


def _centered_bbox(center: np.ndarray, half_extents: tuple[float, float, float]) -> PrimitiveBBox:
    minimum = tuple(float(center[index] - half_extents[index]) for index in range(3))
    maximum = tuple(float(center[index] + half_extents[index]) for index in range(3))
    return PrimitiveBBox(minimum=minimum, maximum=maximum)


def _required(values: dict, name: str, index: int) -> object:
    if name not in values:
        raise ValueError(f"primitive_collision_geoms[{index}] missing required field {name!r}.")
    return values[name]


def _optional_array(raw_value: object, shape: tuple[int, ...], name: str) -> np.ndarray | None:
    if raw_value is None:
        return None
    array = np.asarray(raw_value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    return array


def _optional_positive_float(raw_value: object, name: str) -> float | None:
    if raw_value is None:
        return None
    value = float(raw_value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _optional_rgba(raw_value: object, name: str) -> tuple[float, float, float, float] | None:
    if raw_value is None:
        return None
    values = tuple(float(value) for value in raw_value)  # type: ignore[union-attr]
    if len(values) != 4:
        raise ValueError(f"{name} must contain four values.")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{name} values must be in [0, 1].")
    return values  # type: ignore[return-value]
