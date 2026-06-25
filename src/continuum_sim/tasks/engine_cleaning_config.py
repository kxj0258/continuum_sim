"""Engine-cleaning task config scaffold for M5 path-generation work.

This loader only assembles scene, tool, patch, and raster-path configuration
for tests and future controller work. It does not connect to CLI or MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import required as _required, resolve_path as _resolve_path, section as _section
from continuum_sim.scenes.engine_surfaces import EngineSurfacePatchConfig, load_surface_patch_config
from continuum_sim.tasks.engine_surface_path import EngineSurfacePathConfig


@dataclass(frozen=True)
class EngineCleaningTaskConfig:
    """Standalone config for future executor cleaning path generation."""

    config_path: Path
    task_type: str
    scene_config_path: Path
    target_region: str
    surface_patch: EngineSurfacePatchConfig
    path: EngineSurfacePathConfig
    tool_config_path: Path | None = None
    executor_arm: str | None = None
    observer_arm: str | None = None


def load_engine_cleaning_task_config(path: str | Path) -> EngineCleaningTaskConfig:
    """Load and validate the standalone M5 engine path YAML."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    task_raw = _section(raw, "task")
    surface_patch_raw = _section(task_raw, "surface_patch")
    path_raw = _section(task_raw, "path")

    config = EngineCleaningTaskConfig(
        config_path=config_path,
        task_type=str(_required(task_raw, "type")),
        scene_config_path=_resolve_path(config_path, _required(task_raw, "scene_config_path")),
        target_region=str(_required(task_raw, "target_region")),
        tool_config_path=_optional_resolved_path(config_path, task_raw.get("tool_config_path")),
        executor_arm=_optional_string(task_raw.get("executor_arm")),
        observer_arm=_optional_string(task_raw.get("observer_arm")),
        surface_patch=load_surface_patch_config(surface_patch_raw),
        path=EngineSurfacePathConfig(
            patch_name=str(surface_patch_raw.get("name", "")),
            num_passes_u=int(_required(path_raw, "num_passes_u")),
            num_passes_v=int(_required(path_raw, "num_passes_v")),
            approach_distance_m=float(_required(path_raw, "approach_distance_m")),
            retreat_distance_m=float(_required(path_raw, "retreat_distance_m")),
            target_force_n=float(_required(path_raw, "target_force_n")),
            standoff_distance_m=float(_required(path_raw, "standoff_distance_m")),
            snake_pattern=bool(path_raw.get("snake_pattern", True)),
        ),
    )
    validate_engine_cleaning_task_config(config)
    return config


def validate_engine_cleaning_task_config(config: EngineCleaningTaskConfig) -> None:
    """Validate the minimal cross-reference fields for M5 configs."""

    if not config.task_type:
        raise ValueError("task.type must be non-empty.")
    if not config.target_region:
        raise ValueError("task.target_region must be non-empty.")
    if config.path.patch_name != config.surface_patch.name:
        raise ValueError(
            "task.path.patch_name must match task.surface_patch.name, got "
            f"{config.path.patch_name!r} and {config.surface_patch.name!r}."
        )


def _optional_resolved_path(config_path: Path, raw_value: object) -> Path | None:
    if raw_value is None:
        return None
    return _resolve_path(config_path, raw_value)


def _optional_string(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    return str(raw_value)
