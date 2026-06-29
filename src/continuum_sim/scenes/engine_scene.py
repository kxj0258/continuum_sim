"""Independent engine scene loader for future dual-arm cleaning tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import warnings

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    choice_value as _choice_value,
    geom_group as _geom_group,
    optional_section as _optional_section,
    position_vector as _position_vector,
    positive_float_value as _positive_float_value,
    rgba_tuple as _rgba_tuple,
    required as _required,
    resolve_path as _resolve_config_path,
    section as _section,
)
from continuum_sim.scenes.engine_exploration_path import (
    ExplorationPathConfig,
    load_exploration_paths,
)
from continuum_sim.scenes.primitive_collision import (
    PrimitiveCollisionGeomConfig,
    load_primitive_collision_geoms,
    validate_primitive_collision_geoms as _validate_primitive_collision_geoms,
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
    frame_offset_m: np.ndarray | None = None


@dataclass(frozen=True)
class EngineAssetConfig:
    """Mesh and collision asset paths for the engine model."""

    visual_mesh: Path
    collision_mesh: Path | None
    collision_geoms: Path | None
    collision_mesh_offset_m: np.ndarray | None = None


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
    frame: str = "world"
    center_m: np.ndarray | None = None
    position_m: np.ndarray | None = None
    normal: np.ndarray | None = None
    radius_m: float | None = None
    size_m: np.ndarray | None = None
    extents_m: np.ndarray | None = None
    preview_rgba: tuple[float, float, float, float] | None = None
    description: str = ""


@dataclass(frozen=True)
class ExplorationStartConfig:
    """Optional exploration start point and nominal insertion direction."""

    frame: str
    point_m: np.ndarray
    normal: np.ndarray
    point_rgba: tuple[float, float, float, float] | None = None
    point_radius_m: float | None = None
    normal_rgba: tuple[float, float, float, float] | None = None
    normal_length_m: float | None = None
    normal_radius_m: float | None = None
    group: int | None = None
    description: str = ""


@dataclass(frozen=True)
class PreviewVisualizationConfig:
    """Preview-only style settings loaded from the engine scene YAML."""

    visual_mesh_rgba: tuple[float, float, float, float] = (0.72, 0.76, 0.80, 0.45)
    collision_mesh_rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.15, 0.28)
    bbox_rgba: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.85)
    bbox_edge_radius_m: float | None = None
    engine_axis_length_m: float | None = None
    engine_axis_radius_m: float | None = None
    engine_x_rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    engine_y_rgba: tuple[float, float, float, float] = (0.0, 0.8, 0.0, 1.0)
    engine_z_rgba: tuple[float, float, float, float] = (0.1, 0.2, 1.0, 1.0)
    region_default_rgba_by_type: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "circular_port": (0.1, 0.6, 1.0, 0.6),
            "roi_sphere": (0.2, 1.0, 0.4, 0.35),
            "roi_box": (0.2, 1.0, 0.4, 0.35),
            "surface_patch": (1.0, 0.7, 0.1, 0.55),
            "box": (1.0, 0.1, 0.1, 0.35),
        }
    )
    exploration_start_point_rgba: tuple[float, float, float, float] = (1.0, 0.45, 0.1, 0.95)
    exploration_start_point_radius_m: float | None = None
    exploration_start_normal_rgba: tuple[float, float, float, float] = (1.0, 0.7, 0.1, 0.9)
    exploration_start_normal_length_m: float | None = None
    exploration_start_normal_radius_m: float | None = None
    exploration_start_group: int = 0
    exploration_path_start_marker_rgba: tuple[float, float, float, float] = (0.1, 0.7, 1.0, 0.95)
    exploration_path_end_marker_rgba: tuple[float, float, float, float] = (1.0, 0.85, 0.1, 0.95)
    exploration_path_marker_radius_m: float | None = None
    exploration_path_group: int = 0


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
    preview_visualization: PreviewVisualizationConfig = field(default_factory=PreviewVisualizationConfig)
    exploration_start: ExplorationStartConfig | None = None
    exploration_paths: tuple[ExplorationPathConfig, ...] = ()
    primitive_collision_geoms: tuple[PrimitiveCollisionGeomConfig, ...] = ()


def load_engine_scene_config(path: str | Path) -> EngineSceneConfig:
    """Load and validate a standalone engine scene YAML."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    engine = _section(raw, "engine")
    assets = _section(engine, "assets")
    pose = _section(engine, "pose")
    _reject_legacy_pose_fields(pose)
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
                collision_mesh_offset_m=_optional_position(
                    assets.get("collision_mesh_offset_m", _MISSING),
                    "engine.assets.collision_mesh_offset_m",
                ),
            ),
            scale=_positive_float_value(_required(engine, "scale"), "engine.scale"),
            pose=EnginePoseConfig(
                position_m=_position_vector(_required(pose, "position_m"), "engine.pose.position_m"),
                frame_offset_m=_optional_position(
                    pose.get("frame_offset_m", _MISSING),
                    "engine.pose.frame_offset_m",
                ),
                quat_wxyz=_quaternion(_required(pose, "quat_wxyz"), "engine.pose.quat_wxyz"),
            ),
        ),
        regions={
            name: _load_engine_region_config(name, values)
            for name, values in regions_raw.items()
        },
        preview_visualization=_load_preview_visualization_config(
            raw.get("preview_visualization", _MISSING)
        ),
        exploration_start=_load_exploration_start_config(raw.get("exploration_start", _MISSING)),
        exploration_paths=tuple(load_exploration_paths(raw.get("exploration_paths"))),
        primitive_collision_geoms=tuple(
            load_primitive_collision_geoms(raw.get("primitive_collision_geoms"))
        ),
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
    _validate_primitive_collision_geoms(config.primitive_collision_geoms)
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


def iter_primitive_collision_geoms(
    config: EngineSceneConfig,
) -> Iterator[PrimitiveCollisionGeomConfig]:
    """Iterate optional primitive collision hints in YAML insertion order."""

    yield from config.primitive_collision_geoms


def validate_primitive_collision_geoms(config: EngineSceneConfig) -> None:
    """Validate optional primitive collision hints loaded with an engine scene."""

    _validate_primitive_collision_geoms(config.primitive_collision_geoms)


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


def effective_engine_frame_position(config: EngineSceneConfig) -> np.ndarray:
    """Return the world translation used by objects defined in the engine-local frame."""

    position = np.asarray(config.engine.pose.position_m, dtype=float)
    offset = config.engine.pose.frame_offset_m
    if offset is None:
        return position.copy()
    return position + np.asarray(offset, dtype=float)


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
        frame=_choice_value(
            values.get("frame", "world"),
            f"regions.{region_name}.frame",
            ("world", "engine"),
        ),
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
        preview_rgba=_optional_rgba(
            _optional_section(values, "visualization").get("rgba", _MISSING),
            f"regions.{region_name}.visualization.rgba",
        ),
        description=str(values.get("description", "")),
    )


def _load_exploration_start_config(raw_value: object) -> ExplorationStartConfig | None:
    if raw_value is _MISSING:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("exploration_start must be a mapping.")
    visualization = _optional_section(raw_value, "visualization")
    return ExplorationStartConfig(
        frame=_choice_value(
            raw_value.get("frame", "world"),
            "exploration_start.frame",
            ("world", "engine"),
        ),
        point_m=_position_vector(_required(raw_value, "point_m"), "exploration_start.point_m"),
        normal=_position_vector(_required(raw_value, "normal"), "exploration_start.normal"),
        point_rgba=_optional_rgba(
            visualization.get("point_rgba", _MISSING),
            "exploration_start.visualization.point_rgba",
        ),
        point_radius_m=_optional_positive_float(
            visualization.get("point_radius_m", _MISSING),
            "exploration_start.visualization.point_radius_m",
        ),
        normal_rgba=_optional_rgba(
            visualization.get("normal_rgba", _MISSING),
            "exploration_start.visualization.normal_rgba",
        ),
        normal_length_m=_optional_positive_float(
            visualization.get("normal_length_m", _MISSING),
            "exploration_start.visualization.normal_length_m",
        ),
        normal_radius_m=_optional_positive_float(
            visualization.get("normal_radius_m", _MISSING),
            "exploration_start.visualization.normal_radius_m",
        ),
        group=_optional_group(
            visualization.get("group", _MISSING),
            "exploration_start.visualization.group",
        ),
        description=str(raw_value.get("description", "")),
    )


def _load_preview_visualization_config(raw_value: object) -> PreviewVisualizationConfig:
    if raw_value is _MISSING:
        return PreviewVisualizationConfig()
    if not isinstance(raw_value, dict):
        raise ValueError("preview_visualization must be a mapping.")
    regions = _optional_section(raw_value, "regions")
    default_rgba_by_type = PreviewVisualizationConfig().region_default_rgba_by_type.copy()
    for region_type in ENGINE_REGION_TYPES:
        if region_type in regions:
            default_rgba_by_type[region_type] = _rgba_tuple(
                regions[region_type],
                f"preview_visualization.regions.{region_type}",
            )
    return PreviewVisualizationConfig(
        visual_mesh_rgba=_rgba_tuple(
            raw_value.get("visual_mesh_rgba", (0.72, 0.76, 0.80, 0.45)),
            "preview_visualization.visual_mesh_rgba",
        ),
        collision_mesh_rgba=_rgba_tuple(
            raw_value.get("collision_mesh_rgba", (0.9, 0.2, 0.15, 0.28)),
            "preview_visualization.collision_mesh_rgba",
        ),
        bbox_rgba=_rgba_tuple(
            raw_value.get("bbox_rgba", (1.0, 1.0, 0.0, 0.85)),
            "preview_visualization.bbox_rgba",
        ),
        bbox_edge_radius_m=_optional_positive_float(
            raw_value.get("bbox_edge_radius_m", _MISSING),
            "preview_visualization.bbox_edge_radius_m",
        ),
        engine_axis_length_m=_optional_positive_float(
            raw_value.get("engine_axis_length_m", _MISSING),
            "preview_visualization.engine_axis_length_m",
        ),
        engine_axis_radius_m=_optional_positive_float(
            raw_value.get("engine_axis_radius_m", _MISSING),
            "preview_visualization.engine_axis_radius_m",
        ),
        engine_x_rgba=_rgba_tuple(
            raw_value.get("engine_x_rgba", (1.0, 0.0, 0.0, 1.0)),
            "preview_visualization.engine_x_rgba",
        ),
        engine_y_rgba=_rgba_tuple(
            raw_value.get("engine_y_rgba", (0.0, 0.8, 0.0, 1.0)),
            "preview_visualization.engine_y_rgba",
        ),
        engine_z_rgba=_rgba_tuple(
            raw_value.get("engine_z_rgba", (0.1, 0.2, 1.0, 1.0)),
            "preview_visualization.engine_z_rgba",
        ),
        region_default_rgba_by_type=default_rgba_by_type,
        exploration_start_point_rgba=_rgba_tuple(
            raw_value.get("exploration_start_point_rgba", (1.0, 0.45, 0.1, 0.95)),
            "preview_visualization.exploration_start_point_rgba",
        ),
        exploration_start_point_radius_m=_optional_positive_float(
            raw_value.get("exploration_start_point_radius_m", _MISSING),
            "preview_visualization.exploration_start_point_radius_m",
        ),
        exploration_start_normal_rgba=_rgba_tuple(
            raw_value.get("exploration_start_normal_rgba", (1.0, 0.7, 0.1, 0.9)),
            "preview_visualization.exploration_start_normal_rgba",
        ),
        exploration_start_normal_length_m=_optional_positive_float(
            raw_value.get("exploration_start_normal_length_m", _MISSING),
            "preview_visualization.exploration_start_normal_length_m",
        ),
        exploration_start_normal_radius_m=_optional_positive_float(
            raw_value.get("exploration_start_normal_radius_m", _MISSING),
            "preview_visualization.exploration_start_normal_radius_m",
        ),
        exploration_start_group=_optional_group(
            raw_value.get("exploration_start_group", _MISSING),
            "preview_visualization.exploration_start_group",
        )
        or 0,
        exploration_path_start_marker_rgba=_rgba_tuple(
            raw_value.get("exploration_path_start_marker_rgba", (0.1, 0.7, 1.0, 0.95)),
            "preview_visualization.exploration_path_start_marker_rgba",
        ),
        exploration_path_end_marker_rgba=_rgba_tuple(
            raw_value.get("exploration_path_end_marker_rgba", (1.0, 0.85, 0.1, 0.95)),
            "preview_visualization.exploration_path_end_marker_rgba",
        ),
        exploration_path_marker_radius_m=_optional_positive_float(
            raw_value.get("exploration_path_marker_radius_m", _MISSING),
            "preview_visualization.exploration_path_marker_radius_m",
        ),
        exploration_path_group=_optional_group(
            raw_value.get("exploration_path_group", _MISSING),
            "preview_visualization.exploration_path_group",
        )
        or 0,
    )


def _reject_legacy_pose_fields(pose: dict[object, object]) -> None:
    if "world_offset_m" in pose:
        raise ValueError(
            "engine.pose.world_offset_m is no longer supported; "
            "use engine.pose.frame_offset_m instead."
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


def _optional_rgba(raw_value: object, name: str) -> tuple[float, float, float, float] | None:
    if raw_value is _MISSING:
        return None
    return _rgba_tuple(raw_value, name)


def _optional_positive_float(raw_value: object, name: str) -> float | None:
    if raw_value is _MISSING:
        return None
    return _positive_float_value(raw_value, name)


def _optional_group(raw_value: object, name: str) -> int | None:
    if raw_value is _MISSING:
        return None
    return _geom_group(raw_value, name)


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
