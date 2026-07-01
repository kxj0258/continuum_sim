"""YAML model for scenario-driven single/dual system experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import resolve_path


BACKEND_TYPES = ("analytic", "mujoco")
TASK_TYPES = ("idle", "tracking", "navigation", "wiping")


@dataclass(frozen=True)
class ScenarioBackendConfig:
    type: str
    mujoco_config_path: Path | None = None
    source_xml_path: Path | None = None
    generated_xml_path: Path | None = None
    retain_arm: str | None = None


@dataclass(frozen=True)
class ScenarioSceneConfig:
    engine_config_path: Path | None = None
    structured_config_path: Path | None = None


@dataclass(frozen=True)
class ScenarioTaskConfig:
    type: str
    waypoints_world: np.ndarray
    waypoint_tolerance_m: float
    observer_roi_world: np.ndarray | None
    loop: bool
    min_clearance_m: float
    terminate_on_clearance_violation: bool
    surface_normal_world: np.ndarray
    target_contact_distance_m: float
    contact_tolerance_m: float


@dataclass(frozen=True)
class ScenarioRuntimeConfig:
    controller_dt_s: float
    n_substeps: int
    max_steps: int


@dataclass(frozen=True)
class ScenarioHookConfig:
    recorder: bool
    tendon_debug: bool
    tendon_debug_stride: int
    viewer: str
    keep_viewer_open: bool


@dataclass(frozen=True)
class ScenarioArtifactConfig:
    enabled: bool
    output_root: Path
    save_npz: bool
    save_plots: bool
    save_gif: bool
    save_model: bool
    video_fps: int
    video_stride: int | None


@dataclass(frozen=True)
class ScenarioConfig:
    path: Path
    name: str
    assembly_config_path: Path
    backend: ScenarioBackendConfig
    scene: ScenarioSceneConfig
    task: ScenarioTaskConfig
    runtime: ScenarioRuntimeConfig
    hooks: ScenarioHookConfig
    artifacts: ScenarioArtifactConfig


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    """Load one reproducible application scenario."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    values = _mapping(raw.get("scenario"), "scenario")
    backend_values = _mapping(values.get("backend"), "scenario.backend")
    backend_type = str(backend_values.get("type", "analytic"))
    if backend_type not in BACKEND_TYPES:
        raise ValueError(f"scenario.backend.type must be one of {BACKEND_TYPES}.")
    task_values = _mapping(values.get("task", {}), "scenario.task")
    task_type = str(task_values.get("type", "idle"))
    if task_type not in TASK_TYPES:
        raise ValueError(f"scenario.task.type must be one of {TASK_TYPES}.")
    waypoints = np.asarray(task_values.get("waypoints_world", []), dtype=float)
    if task_type == "idle":
        waypoints = np.zeros((0, 3), dtype=float)
    elif waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] == 0:
        raise ValueError("Non-idle scenario tasks require waypoints_world with shape (N, 3).")
    roi_raw = task_values.get("observer_roi_world")
    roi = None if roi_raw is None else np.asarray(roi_raw, dtype=float)
    if roi is not None and roi.shape != (3,):
        raise ValueError("scenario.task.observer_roi_world must have shape (3,).")
    surface_normal = np.asarray(
        task_values.get("surface_normal_world", [0.0, 0.0, 1.0]),
        dtype=float,
    )
    if surface_normal.shape != (3,) or np.linalg.norm(surface_normal) <= 0.0:
        raise ValueError("scenario.task.surface_normal_world must be a nonzero 3D vector.")
    surface_normal /= np.linalg.norm(surface_normal)
    runtime_values = _mapping(values.get("runtime", {}), "scenario.runtime")
    hook_values = _mapping(values.get("hooks", {}), "scenario.hooks")
    scene_values = _mapping(values.get("scene", {}), "scenario.scene")
    artifact_values = _mapping(values.get("artifacts", {}), "scenario.artifacts")
    viewer = str(hook_values.get("viewer", "none"))
    if viewer not in ("none", "matplotlib", "mujoco"):
        raise ValueError("scenario.hooks.viewer must be none, matplotlib, or mujoco.")
    return ScenarioConfig(
        path=config_path,
        name=str(values.get("name", config_path.stem)),
        assembly_config_path=resolve_path(
            config_path,
            _required(values, "assembly_config_path", "scenario"),
        ),
        backend=ScenarioBackendConfig(
            type=backend_type,
            mujoco_config_path=_optional_path(
                config_path,
                backend_values.get("mujoco_config_path"),
            ),
            source_xml_path=_optional_path(
                config_path,
                backend_values.get("source_xml_path"),
            ),
            generated_xml_path=_optional_path(
                config_path,
                backend_values.get("generated_xml_path"),
            ),
            retain_arm=(
                None
                if backend_values.get("retain_arm") is None
                else str(backend_values["retain_arm"])
            ),
        ),
        scene=ScenarioSceneConfig(
            engine_config_path=_optional_path(
                config_path,
                scene_values.get("engine_config_path"),
            ),
            structured_config_path=_optional_path(
                config_path,
                scene_values.get("structured_config_path"),
            ),
        ),
        task=ScenarioTaskConfig(
            type=task_type,
            waypoints_world=waypoints.copy(),
            waypoint_tolerance_m=float(task_values.get("waypoint_tolerance_m", 0.001)),
            observer_roi_world=None if roi is None else roi.copy(),
            loop=bool(task_values.get("loop", False)),
            min_clearance_m=float(task_values.get("min_clearance_m", 0.01)),
            terminate_on_clearance_violation=bool(
                task_values.get("terminate_on_clearance_violation", True)
            ),
            surface_normal_world=surface_normal,
            target_contact_distance_m=float(
                task_values.get("target_contact_distance_m", 0.0)
            ),
            contact_tolerance_m=float(task_values.get("contact_tolerance_m", 0.002)),
        ),
        runtime=ScenarioRuntimeConfig(
            controller_dt_s=float(runtime_values.get("controller_dt_s", 0.02)),
            n_substeps=int(runtime_values.get("n_substeps", 20)),
            max_steps=int(runtime_values.get("max_steps", 1000)),
        ),
        hooks=ScenarioHookConfig(
            recorder=bool(hook_values.get("recorder", True)),
            tendon_debug=bool(hook_values.get("tendon_debug", False)),
            tendon_debug_stride=int(hook_values.get("tendon_debug_stride", 1)),
            viewer=viewer,
            keep_viewer_open=bool(hook_values.get("keep_viewer_open", True)),
        ),
        artifacts=ScenarioArtifactConfig(
            enabled=bool(artifact_values.get("enabled", task_type != "idle")),
            output_root=resolve_path(
                config_path,
                artifact_values.get("output_root", "../../output/runs"),
            ),
            save_npz=bool(artifact_values.get("save_npz", True)),
            save_plots=bool(artifact_values.get("save_plots", True)),
            save_gif=bool(artifact_values.get("save_gif", True)),
            save_model=bool(artifact_values.get("save_model", True)),
            video_fps=int(artifact_values.get("video_fps", 20)),
            video_stride=(
                None
                if artifact_values.get("video_stride") is None
                else int(artifact_values["video_stride"])
            ),
        ),
    )


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _required(values: dict, name: str, section: str) -> object:
    if name not in values:
        raise ValueError(f"Missing required field {section}.{name}.")
    return values[name]


def _optional_path(config_path: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(config_path, value)
