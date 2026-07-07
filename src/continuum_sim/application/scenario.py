"""YAML model for scenario-driven single/dual system experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import resolve_path
from continuum_sim.control.waypoint_scheduler import WAYPOINT_ADVANCE_MODES
from continuum_sim.tasks.engine_cleaning_path import EngineCleaningPathSpec
from continuum_sim.tasks.engine_navigation import EngineNavigationSpec
from continuum_sim.tasks.navigation_mission import NavigationMissionSpec
from continuum_sim.tasks.trajectory_generation import TrajectorySpec
from continuum_sim.tasks.wiping_path import WipingPathSpec


BACKEND_TYPES = ("analytic", "mujoco")
TASK_TYPES = (
    "idle",
    "tracking",
    "navigation",
    "wiping",
    "engine_cleaning",
    "engine_navigation",
)
WIPING_CONTROL_TYPES = ("contact_distance", "hybrid_force_position", "dynamic_adaptive_impedance")
MUJOCO_FEEDBACK_MODES = ("pcc_command", "mujoco_actual")
VIDEO_MODES = ("replay", "live_mujoco")


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
class ScenarioTrackingControlConfig:
    """Scenario-native trajectory-tracking controller parameters."""

    approach_samples: int = 0
    executor_position_gain: float = 4.0
    observer_position_gain: float = 5.0
    feedforward_speed_mps: float = 0.0
    max_target_speed_mps: float | None = None
    executor_tracking_weight: float = 100.0
    observer_tracking_weight: float = 40.0
    executor_collision_avoidance_weight: float = 80.0
    base_regularization_weight: float = 1.0
    tendon_regularization_weight: float = 0.2
    rank_tolerance: float = 1.0e-9
    minimum_singular_value: float = 1.0e-5
    nominal_damping: float = 1.0e-4
    maximum_damping: float = 5.0e-2
    minimum_velocity_scale: float = 0.05
    decouple_arm_singularity: bool = False

    def __post_init__(self) -> None:
        if self.approach_samples == 1 or self.approach_samples < 0:
            raise ValueError("tracking_control.approach_samples must be 0 or at least 2.")
        positive = {
            "executor_position_gain": self.executor_position_gain,
            "observer_position_gain": self.observer_position_gain,
            "executor_tracking_weight": self.executor_tracking_weight,
            "observer_tracking_weight": self.observer_tracking_weight,
            "executor_collision_avoidance_weight": self.executor_collision_avoidance_weight,
            "base_regularization_weight": self.base_regularization_weight,
            "tendon_regularization_weight": self.tendon_regularization_weight,
            "rank_tolerance": self.rank_tolerance,
            "minimum_singular_value": self.minimum_singular_value,
            "nominal_damping": self.nominal_damping,
            "maximum_damping": self.maximum_damping,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"tracking_control.{name} must be positive and finite.")
        if not np.isfinite(self.feedforward_speed_mps) or self.feedforward_speed_mps < 0.0:
            raise ValueError(
                "tracking_control.feedforward_speed_mps must be non-negative and finite."
            )
        if (
            self.max_target_speed_mps is not None
            and (
                not np.isfinite(self.max_target_speed_mps)
                or self.max_target_speed_mps <= 0.0
            )
        ):
            raise ValueError(
                "tracking_control.max_target_speed_mps must be positive and finite."
            )
        if self.maximum_damping < self.nominal_damping:
            raise ValueError(
                "tracking_control.maximum_damping must be at least nominal_damping."
            )
        if not 0.0 < self.minimum_velocity_scale <= 1.0:
            raise ValueError(
                "tracking_control.minimum_velocity_scale must be in (0, 1]."
            )


@dataclass(frozen=True)
class ScenarioTaskConfig:
    type: str
    waypoints_world: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )
    waypoint_source: str = "waypoints_world"
    waypoint_tolerance_m: float = 0.001
    observer_roi_world: np.ndarray | None = None
    loop: bool = False
    target_advance_mode: str = "tolerance"
    advance_time_s: float | None = None
    advance_steps: int | None = None
    min_clearance_m: float = 0.01
    terminate_on_clearance_violation: bool = True
    surface_normal_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )
    target_contact_distance_m: float = 0.0
    contact_tolerance_m: float = 0.002
    trajectory: TrajectorySpec | None = None
    mission: NavigationMissionSpec | None = None
    wiping_path: WipingPathSpec | None = None
    engine_cleaning: EngineCleaningPathSpec | None = None
    waypoint_phases: tuple[str, ...] = ()
    target_force_n: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    wiping_control_type: str = "contact_distance"
    feedback_mode: str = "mujoco_actual"
    normal_force_gain: float = 0.0
    target_normal_force_n: float = 0.0
    force_proxy_stiffness_n_m: float = 600.0
    max_contact_force_n: float | None = None
    contact_loss_tolerance_steps: int = 20
    engine_navigation: EngineNavigationSpec | None = None
    tracking_control: ScenarioTrackingControlConfig = field(
        default_factory=ScenarioTrackingControlConfig
    )


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
    show_live_tendon_panel: bool = False
    live_tendon_panel_stride: int = 1
    show_live_force_panel: bool = False
    live_force_panel_stride: int = 1
    live_force_panel_history_points: int = 300
    show_live_diagnostics_panel: bool = False
    live_diagnostics_panel_stride: int = 5
    live_diagnostics_panel_history_points: int = 300


@dataclass(frozen=True)
class ScenarioArtifactConfig:
    enabled: bool
    output_root: Path
    save_npz: bool
    save_plots: bool
    save_gif: bool
    save_model: bool
    video_mode: str = "replay"
    video_fps: int = 20
    video_stride: int | None = None


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
    trajectory = (
        None
        if task_values.get("trajectory") is None
        else TrajectorySpec.from_mapping(
            _mapping(task_values["trajectory"], "scenario.task.trajectory"),
            base_path=config_path,
        )
    )
    mission = (
        None
        if task_values.get("mission") is None
        else NavigationMissionSpec.from_mapping(
            _mapping(task_values["mission"], "scenario.task.mission")
        )
    )
    wiping_path = (
        None
        if task_values.get("wiping_path") is None
        else WipingPathSpec.from_mapping(
            _mapping(task_values["wiping_path"], "scenario.task.wiping_path")
        )
    )
    engine_cleaning = (
        None
        if task_values.get("engine_cleaning") is None
        else EngineCleaningPathSpec.from_mapping(
            _mapping(task_values["engine_cleaning"], "scenario.task.engine_cleaning")
        )
    )
    engine_navigation = (
        None
        if task_values.get("engine_navigation") is None
        else EngineNavigationSpec.from_mapping(
            _mapping(
                task_values["engine_navigation"],
                "scenario.task.engine_navigation",
            )
        )
    )
    waypoint_source = _waypoint_source(task_values)
    waypoints = np.asarray(task_values.get("waypoints_world", []), dtype=float)
    if task_type == "idle":
        waypoints = np.zeros((0, 3), dtype=float)
    elif waypoint_source == "waypoints_world" and (
        waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] == 0
    ):
        raise ValueError("Non-idle scenario tasks require waypoints_world with shape (N, 3).")
    elif waypoint_source != "waypoints_world":
        waypoints = np.zeros((0, 3), dtype=float)
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
    video_mode = str(artifact_values.get("video_mode", "replay"))
    if video_mode not in VIDEO_MODES:
        raise ValueError(f"scenario.artifacts.video_mode must be one of {VIDEO_MODES}.")
    if (
        backend_type != "mujoco"
        and bool(artifact_values.get("save_gif", True))
        and video_mode == "live_mujoco"
    ):
        raise ValueError("scenario.artifacts.video_mode='live_mujoco' requires a MuJoCo backend.")
    target_advance_mode = str(task_values.get("target_advance_mode", "tolerance"))
    if target_advance_mode not in WAYPOINT_ADVANCE_MODES:
        raise ValueError(f"scenario.task.target_advance_mode must be one of {WAYPOINT_ADVANCE_MODES}.")
    wiping_control_type = str(task_values.get("wiping_control_type", "contact_distance"))
    if wiping_control_type not in WIPING_CONTROL_TYPES:
        raise ValueError(f"scenario.task.wiping_control_type must be one of {WIPING_CONTROL_TYPES}.")
    feedback_mode = str(task_values.get("feedback_mode", "mujoco_actual"))
    if feedback_mode not in MUJOCO_FEEDBACK_MODES:
        raise ValueError(f"scenario.task.feedback_mode must be one of {MUJOCO_FEEDBACK_MODES}.")
    tracking_control_values = _mapping(
        task_values.get("tracking_control", {}),
        "scenario.task.tracking_control",
    )
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
            waypoint_source=waypoint_source,
            waypoint_tolerance_m=float(task_values.get("waypoint_tolerance_m", 0.001)),
            observer_roi_world=None if roi is None else roi.copy(),
            loop=bool(task_values.get("loop", False)),
            target_advance_mode=target_advance_mode,
            advance_time_s=_optional_float(task_values.get("advance_time_s")),
            advance_steps=(
                None
                if task_values.get("advance_steps") is None
                else int(task_values["advance_steps"])
            ),
            min_clearance_m=float(task_values.get("min_clearance_m", 0.01)),
            terminate_on_clearance_violation=bool(
                task_values.get("terminate_on_clearance_violation", True)
            ),
            surface_normal_world=surface_normal,
            target_contact_distance_m=float(
                task_values.get("target_contact_distance_m", 0.0)
            ),
            contact_tolerance_m=float(task_values.get("contact_tolerance_m", 0.002)),
            trajectory=trajectory,
            mission=mission,
            wiping_path=wiping_path,
            engine_cleaning=engine_cleaning,
            engine_navigation=engine_navigation,
            waypoint_phases=tuple(str(value) for value in task_values.get("waypoint_phases", ())),
            target_force_n=np.asarray(task_values.get("target_force_n", []), dtype=float),
            wiping_control_type=wiping_control_type,
            feedback_mode=feedback_mode,
            normal_force_gain=float(task_values.get("normal_force_gain", 0.0)),
            target_normal_force_n=float(task_values.get("target_normal_force_n", 0.0)),
            force_proxy_stiffness_n_m=float(
                task_values.get("force_proxy_stiffness_n_m", 600.0)
            ),
            max_contact_force_n=_optional_float(task_values.get("max_contact_force_n")),
            contact_loss_tolerance_steps=int(
                task_values.get("contact_loss_tolerance_steps", 20)
            ),
            tracking_control=ScenarioTrackingControlConfig(
                approach_samples=int(
                    tracking_control_values.get("approach_samples", 0)
                ),
                executor_position_gain=float(
                    tracking_control_values.get("executor_position_gain", 4.0)
                ),
                observer_position_gain=float(
                    tracking_control_values.get("observer_position_gain", 5.0)
                ),
                feedforward_speed_mps=float(
                    tracking_control_values.get("feedforward_speed_mps", 0.0)
                ),
                max_target_speed_mps=_optional_float(
                    tracking_control_values.get("max_target_speed_mps")
                ),
                executor_tracking_weight=float(
                    tracking_control_values.get("executor_tracking_weight", 100.0)
                ),
                observer_tracking_weight=float(
                    tracking_control_values.get("observer_tracking_weight", 40.0)
                ),
                executor_collision_avoidance_weight=float(
                    tracking_control_values.get(
                        "executor_collision_avoidance_weight",
                        80.0,
                    )
                ),
                base_regularization_weight=float(
                    tracking_control_values.get("base_regularization_weight", 1.0)
                ),
                tendon_regularization_weight=float(
                    tracking_control_values.get("tendon_regularization_weight", 0.2)
                ),
                rank_tolerance=float(
                    tracking_control_values.get("rank_tolerance", 1.0e-9)
                ),
                minimum_singular_value=float(
                    tracking_control_values.get("minimum_singular_value", 1.0e-5)
                ),
                nominal_damping=float(
                    tracking_control_values.get("nominal_damping", 1.0e-4)
                ),
                maximum_damping=float(
                    tracking_control_values.get("maximum_damping", 5.0e-2)
                ),
                minimum_velocity_scale=float(
                    tracking_control_values.get("minimum_velocity_scale", 0.05)
                ),
                decouple_arm_singularity=bool(
                    tracking_control_values.get("decouple_arm_singularity", False)
                ),
            ),
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
            show_live_tendon_panel=bool(hook_values.get("show_live_tendon_panel", False)),
            live_tendon_panel_stride=int(hook_values.get("live_tendon_panel_stride", 1)),
            show_live_force_panel=bool(hook_values.get("show_live_force_panel", False)),
            live_force_panel_stride=int(hook_values.get("live_force_panel_stride", 1)),
            live_force_panel_history_points=int(
                hook_values.get("live_force_panel_history_points", 300)
            ),
            show_live_diagnostics_panel=bool(
                hook_values.get("show_live_diagnostics_panel", False)
            ),
            live_diagnostics_panel_stride=int(
                hook_values.get("live_diagnostics_panel_stride", 5)
            ),
            live_diagnostics_panel_history_points=int(
                hook_values.get("live_diagnostics_panel_history_points", 300)
            ),
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
            video_mode=video_mode,
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


def _waypoint_source(task_values: dict[str, Any]) -> str:
    sources = [
        name
        for name in (
            "waypoints_world",
            "trajectory",
            "mission",
            "wiping_path",
            "engine_cleaning",
            "engine_navigation",
        )
        if task_values.get(name) is not None
    ]
    if not sources:
        return "waypoints_world"
    if len(sources) > 1:
        raise ValueError(f"scenario.task defines multiple waypoint sources: {sources}.")
    return sources[0]


def _required(values: dict, name: str, section: str) -> object:
    if name not in values:
        raise ValueError(f"Missing required field {section}.{name}.")
    return values[name]


def _optional_path(config_path: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(config_path, value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
