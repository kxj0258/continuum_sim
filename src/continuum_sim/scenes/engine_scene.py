"""Independent engine scene loader for future dual-arm cleaning tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import warnings

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    choice_value as _choice_value,
    optional_section as _optional_section,
    position_vector as _position_vector,
    positive_float_value as _positive_float_value,
    required as _required,
    resolve_path as _resolve_config_path,
    section as _section,
)


ENGINE_SCENE_TYPES = ("engine_cleaning",)
ENGINE_REGION_TYPES = (
    "circular_port",
    "roi_sphere",
    "roi_box",
    "surface_patch",
    "box",
)
_MISSING = object()


@dataclass(frozen=True)
class EnginePoseConfig:
    """Pose for the engine model in the world frame."""

    position_m: np.ndarray
    quat_wxyz: np.ndarray


@dataclass(frozen=True)
class EngineAssetConfig:
    """Mesh and collision asset paths for the engine model."""

    visual_mesh: Path
    collision_mesh: Path | None
    collision_geoms: Path | None


@dataclass(frozen=True)
class ResolvedEngineAssetPaths:
    """Resolved asset paths rooted at a chosen directory."""

    visual_mesh: Path
    collision_mesh: Path | None
    collision_geoms: Path | None


@dataclass(frozen=True)
class EngineRegionConfig:
    """Named logical region used by future engine cleaning tasks."""

    name: str
    type: str
    center_m: np.ndarray | None = None
    position_m: np.ndarray | None = None
    normal: np.ndarray | None = None
    radius_m: float | None = None
    size_m: np.ndarray | None = None
    extents_m: np.ndarray | None = None
    description: str = ""


@dataclass(frozen=True)
class EngineModelConfig:
    """Engine mesh, scale, and pose information."""

    assets: EngineAssetConfig
    scale: float
    pose: EnginePoseConfig


@dataclass(frozen=True)
class EngineSceneConfig:
    """Complete engine scene description."""

    path: Path
    name: str
    scene_type: str
    engine: EngineModelConfig
    regions: dict[str, EngineRegionConfig]


def load_engine_scene_config(path: str | Path) -> EngineSceneConfig:
    """Load and validate a standalone engine scene YAML."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    engine = _section(raw, "engine")
    assets = _section(engine, "assets")
    pose = _section(engine, "pose")
    regions_raw = _required(raw, "regions")
    if not isinstance(regions_raw, dict):
        raise ValueError("regions must be a mapping of region-name -> config.")

    config = EngineSceneConfig(
        path=config_path,
        name=str(raw.get("name", config_path.stem)),
        scene_type=_choice_value(
            _required(raw, "scene_type"),
            "scene_type",
            ENGINE_SCENE_TYPES,
        ),
        engine=EngineModelConfig(
            assets=EngineAssetConfig(
                visual_mesh=_resolve_config_path(
                    config_path,
                    _required(assets, "visual_mesh"),
                ),
                collision_mesh=_optional_resolved_path(
                    config_path,
                    assets.get("collision_mesh", _MISSING),
                ),
                collision_geoms=_optional_resolved_path(
                    config_path,
                    assets.get("collision_geoms", _MISSING),
                ),
            ),
            scale=_positive_float_value(_required(engine, "scale"), "engine.scale"),
            pose=EnginePoseConfig(
                position_m=_position_vector(_required(pose, "position_m"), "engine.pose.position_m"),
                quat_wxyz=_quaternion(_required(pose, "quat_wxyz"), "engine.pose.quat_wxyz"),
            ),
        ),
        regions={
            name: _load_engine_region_config(name, values)
            for name, values in regions_raw.items()
        },
    )
    return config


def validate_engine_scene_config(
    config: EngineSceneConfig,
    *,
    strict_assets: bool = False,
) -> None:
    """Validate logical fields and optional asset existence checks."""

    if config.scene_type not in ENGINE_SCENE_TYPES:
        raise ValueError(
            f"scene_type must be one of {ENGINE_SCENE_TYPES}, got {config.scene_type!r}."
        )
    if not config.regions:
        return
    for region_name, region in config.regions.items():
        if region.type not in ENGINE_REGION_TYPES:
            raise ValueError(
                f"Unknown region type {region.type!r} for region {region_name!r}."
            )
        _validate_region_geometry(region)

    resolved_assets = resolve_engine_asset_paths(config, config.path.parent)
    for asset_name, asset_path in (
        ("visual_mesh", resolved_assets.visual_mesh),
        ("collision_mesh", resolved_assets.collision_mesh),
        ("collision_geoms", resolved_assets.collision_geoms),
    ):
        if asset_path is None or asset_path.exists():
            continue
        message = (
            f"Engine asset {asset_name!r} does not exist: {asset_path}. "
            f"Set strict_assets=False to allow placeholders."
        )
        if strict_assets:
            raise FileNotFoundError(message)
        warnings.warn(message, UserWarning, stacklevel=2)


def iter_engine_regions(config: EngineSceneConfig) -> Iterator[tuple[str, EngineRegionConfig]]:
    """Iterate named engine regions in YAML insertion order."""

    yield from config.regions.items()


def resolve_engine_asset_paths(
    config: EngineSceneConfig,
    root_dir: str | Path,
) -> ResolvedEngineAssetPaths:
    """Resolve engine asset paths relative to a chosen root directory."""

    root = Path(root_dir).resolve()
    assets = config.engine.assets
    return ResolvedEngineAssetPaths(
        visual_mesh=_resolve_asset_path(assets.visual_mesh, root),
        collision_mesh=_resolve_optional_asset_path(assets.collision_mesh, root),
        collision_geoms=_resolve_optional_asset_path(assets.collision_geoms, root),
    )


def _load_engine_region_config(name: object, values: object) -> EngineRegionConfig:
    region_name = str(name)
    if not isinstance(values, dict):
        raise ValueError(f"Region {region_name!r} must be a mapping.")
    raw_type = str(_required(values, "type"))
    if raw_type not in ENGINE_REGION_TYPES:
        raise ValueError(
            f"Unknown region type {raw_type!r} for region {region_name!r}. "
            f"Expected one of {ENGINE_REGION_TYPES}."
        )
    region_type = raw_type
    return EngineRegionConfig(
        name=region_name,
        type=region_type,
        center_m=_optional_position(values.get("center_m", _MISSING), f"regions.{region_name}.center_m"),
        position_m=_optional_position(
            values.get("position_m", _MISSING),
            f"regions.{region_name}.position_m",
        ),
        normal=_optional_position(values.get("normal", _MISSING), f"regions.{region_name}.normal"),
        radius_m=_optional_positive_float(
            values.get("radius_m", _MISSING),
            f"regions.{region_name}.radius_m",
        ),
        size_m=_optional_position(values.get("size_m", _MISSING), f"regions.{region_name}.size_m"),
        extents_m=_optional_position(
            values.get("extents_m", _MISSING),
            f"regions.{region_name}.extents_m",
        ),
        description=str(values.get("description", "")),
    )


def _validate_region_geometry(region: EngineRegionConfig) -> None:
    if region.type == "circular_port":
        _required_optional_array(region.center_m, f"{region.name}.center_m")
        _required_optional_array(region.normal, f"{region.name}.normal")
        _required_optional_float(region.radius_m, f"{region.name}.radius_m")
        return
    if region.type in ("roi_sphere",):
        _required_optional_array(region.center_m, f"{region.name}.center_m")
        _required_optional_float(region.radius_m, f"{region.name}.radius_m")
        return
    if region.type in ("roi_box", "box"):
        _required_optional_array(region.center_m, f"{region.name}.center_m")
        size = _required_optional_array(region.size_m, f"{region.name}.size_m")
        if np.any(size <= 0.0):
            raise ValueError(f"{region.name}.size_m values must be positive.")
        return
    if region.type == "surface_patch":
        _required_optional_array(region.position_m, f"{region.name}.position_m")
        _required_optional_array(region.normal, f"{region.name}.normal")
        extents = _required_optional_array(region.extents_m, f"{region.name}.extents_m")
        if np.any(extents <= 0.0):
            raise ValueError(f"{region.name}.extents_m values must be positive.")
        return
    raise ValueError(f"Unknown region type {region.type!r}.")


def _resolve_asset_path(path: Path, root_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def _resolve_optional_asset_path(path: Path | None, root_dir: Path) -> Path | None:
    if path is None:
        return None
    return _resolve_asset_path(path, root_dir)


def _optional_resolved_path(config_path: Path, raw_value: object) -> Path | None:
    if raw_value is _MISSING:
        return None
    return _resolve_config_path(config_path, raw_value)


def _optional_position(raw_value: object, name: str) -> np.ndarray | None:
    if raw_value is _MISSING:
        return None
    return _position_vector(raw_value, name)


def _optional_positive_float(raw_value: object, name: str) -> float | None:
    if raw_value is _MISSING:
        return None
    return _positive_float_value(raw_value, name)


def _quaternion(raw_value: object, name: str) -> np.ndarray:
    quat = np.asarray(raw_value, dtype=float)
    if quat.shape != (4,):
        raise ValueError(f"Expected {name} with shape (4,), got {quat.shape}.")
    if np.linalg.norm(quat) <= 1.0e-12:
        raise ValueError(f"{name} must have non-zero length.")
    return quat


def _required_optional_array(value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required config field {name!r}.")
    return value


def _required_optional_float(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required config field {name!r}.")
    return float(value)
