"""YAML model for scenario-driven single/dual system experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import resolve_path
from continuum_sim.control.contact_triggered_admittance import ContactTriggeredAdmittanceConfig
from continuum_sim.control.engine_cleaning_types import EngineCleaningControllerGains
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
WIPING_CONTROL_TYPES = (
    "contact_distance",
    "hybrid_force_position",
    "dynamic_adaptive_impedance",
    "contact_triggered_admittance",
)
NAVIGATION_CONTROL_TYPES = ("whole_body", "navigation_cbf_qp")
MUJOCO_FEEDBACK_MODES = ("pcc_command", "mujoco_actual")
VIDEO_MODES = ("replay", "live_mujoco")
WIPING_FORCE_STRATEGY_TYPES = (
    "contact_distance",
    "kinematic_hybrid",
    "dynamic_adaptive_impedance",
    "contact_triggered_admittance",
)
TRACKING_MODES = ("waypoint", "time")
OBSERVER_CONTROL_MODES = ("tracking", "collision_avoidance", "disabled")
SINGULARITY_STRATEGIES = ("damping_scale", "svd_projection")


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
    tracking_mode: str = "waypoint"
    trajectory_duration_s: float | None = None
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
    singularity_strategy: str = "svd_projection"
    enforce_target_speed_limit: bool = False
    enforce_solver_velocity_limits: bool = False
    enforce_backend_tendon_limits: bool = False

    def __post_init__(self) -> None:
        if self.approach_samples == 1 or self.approach_samples < 0:
            raise ValueError("tracking_control.approach_samples must be 0 or at least 2.")
        if self.tracking_mode not in TRACKING_MODES:
            raise ValueError(
                f"tracking_control.tracking_mode must be one of {TRACKING_MODES}."
            )
        if self.singularity_strategy not in SINGULARITY_STRATEGIES:
            raise ValueError(
                "tracking_control.singularity_strategy must be one of "
                f"{SINGULARITY_STRATEGIES}."
            )
        if self.tracking_mode == "time" and (
            self.trajectory_duration_s is None
            or not np.isfinite(self.trajectory_duration_s)
            or self.trajectory_duration_s <= 0.0
        ):
            raise ValueError(
                "tracking_control.trajectory_duration_s must be positive for "
                "tracking_mode='time'."
            )
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
            self.enforce_target_speed_limit
            and
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
class ScenarioObserverControlConfig:
    """Task-level observer collision-avoidance policy."""

    minimum_distance_m: float = 0.010
    influence_distance_m: float = 0.050
    critical_distance_m: float = 0.008
    release_margin_m: float = 0.005
    avoidance_gain: float = 0.4
    max_avoidance_speed_mps: float | None = None

    def __post_init__(self) -> None:
        non_negative = {
            "minimum_distance_m": self.minimum_distance_m,
            "influence_distance_m": self.influence_distance_m,
            "critical_distance_m": self.critical_distance_m,
            "release_margin_m": self.release_margin_m,
        }
        for name, value in non_negative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"observer_control.{name} must be non-negative and finite.")
        if self.influence_distance_m <= self.minimum_distance_m:
            raise ValueError(
                "observer_control.influence_distance_m must exceed minimum_distance_m."
            )
        if self.critical_distance_m > self.minimum_distance_m:
            raise ValueError(
                "observer_control.critical_distance_m must not exceed minimum_distance_m."
            )
        if not np.isfinite(self.avoidance_gain) or self.avoidance_gain <= 0.0:
            raise ValueError("observer_control.avoidance_gain must be positive and finite.")
        if self.max_avoidance_speed_mps is not None and (
            not np.isfinite(self.max_avoidance_speed_mps)
            or self.max_avoidance_speed_mps <= 0.0
        ):
            raise ValueError(
                "observer_control.max_avoidance_speed_mps must be positive and finite."
            )


@dataclass(frozen=True)
class ScenarioForceStrategyConfig:
    type: str = "contact_distance"


@dataclass(frozen=True)
class ScenarioAdmittanceConfig:
    target_normal_force_n: float = 0.0
    contact_force_threshold_n: float = 0.1
    tangent_tolerance_m: float = 1.0e-3
    force_tolerance_n: float = 0.08
    stable_steps_required: int = 1
    max_steps_per_target: int = 100
    position_gain: float = 10.0
    kp_force: float = 0.5
    ki_force: float = 0.012
    admittance_mass: float = 1.0
    admittance_damping: float = 20.0
    admittance_stiffness: float = 5.0
    admittance_clip_m: float = 0.012
    force_deadband_n: float = 0.03
    force_filter_alpha: float = 0.1
    max_tangent_velocity_m_s: float = 0.012
    max_normal_velocity_m_s: float = 0.010
    enforce_velocity_limits: bool = False


@dataclass(frozen=True)
class ScenarioTaskConfig:
    type: str
    waypoints_world: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )
    waypoint_source: str = "waypoints_world"
    waypoint_tolerance_m: float = 0.001
    observer_roi_world: np.ndarray | None = None
    observer_control_mode: str = "collision_avoidance"
    observer_control: ScenarioObserverControlConfig = field(
        default_factory=ScenarioObserverControlConfig
    )
    loop: bool = False
    target_advance_mode: str = "tolerance"
    advance_time_s: float | None = None
    advance_steps: int | None = None
    min_clearance_m: float = 0.01
    terminate_on_clearance_violation: bool = True
    navigation_control_type: str = "whole_body"
    navigation_cbf_gain: float = 4.0
    navigation_cbf_influence_distance_m: float | None = None
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
    dynamics_config_path: Path | None = None
    feedback_mode: str = "mujoco_actual"
    normal_force_gain: float = 0.0
    target_normal_force_n: float = 0.0
    force_proxy_stiffness_n_m: float = 600.0
    max_contact_force_n: float | None = None
    contact_loss_tolerance_steps: int = 20
    force_strategy: ScenarioForceStrategyConfig = field(
        default_factory=ScenarioForceStrategyConfig
    )
    admittance: ScenarioAdmittanceConfig = field(
        default_factory=ScenarioAdmittanceConfig
    )
    engine_navigation: EngineNavigationSpec | None = None
    contact_admittance: ContactTriggeredAdmittanceConfig | None = None
    engine_cleaning_control: EngineCleaningControllerGains | None = None
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
    show_live_diagnostics_panel: bool = True
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
    low_level_control_path: Path | None = None


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    """Load one reproducible application scenario."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    values = _mapping(raw.get("scenario"), "scenario")
    low_level_control_path = _optional_path(
        config_path,
        values.get("low_level_control_path"),
    )
    low_level_control_values = _load_low_level_control_values(
        low_level_control_path
    )
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
    observer_control_mode = str(
        task_values.get("observer_control_mode", "collision_avoidance")
    )
    if observer_control_mode not in OBSERVER_CONTROL_MODES:
        raise ValueError(
            "scenario.task.observer_control_mode must be one of "
            f"{OBSERVER_CONTROL_MODES}."
        )
    navigation_control_type = str(task_values.get("navigation_control_type", "whole_body"))
    if navigation_control_type not in NAVIGATION_CONTROL_TYPES:
        raise ValueError(
            "scenario.task.navigation_control_type must be one of "
            f"{NAVIGATION_CONTROL_TYPES}."
        )
    wiping_control_type = str(task_values.get("wiping_control_type", "contact_distance"))
    if wiping_control_type not in WIPING_CONTROL_TYPES:
        raise ValueError(f"scenario.task.wiping_control_type must be one of {WIPING_CONTROL_TYPES}.")
    force_strategy = _load_force_strategy_config(task_values, wiping_control_type)
    feedback_mode = str(task_values.get("feedback_mode", "mujoco_actual"))
    if feedback_mode not in MUJOCO_FEEDBACK_MODES:
        raise ValueError(f"scenario.task.feedback_mode must be one of {MUJOCO_FEEDBACK_MODES}.")
    tracking_control = _load_tracking_control_config(
        task_values,
        low_level_control_values,
    )
    observer_control = _load_observer_control_config(task_values)
    contact_admittance = _load_contact_admittance_config(task_values)
    if wiping_control_type == "contact_triggered_admittance" and contact_admittance is None:
        contact_admittance = ContactTriggeredAdmittanceConfig(
            target_normal_force_n=float(task_values.get("target_normal_force_n", 0.0))
        )
    engine_cleaning_control = _load_engine_cleaning_control_config(task_values)
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
        low_level_control_path=low_level_control_path,
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
            observer_control_mode=observer_control_mode,
            observer_control=observer_control,
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
            navigation_control_type=navigation_control_type,
            navigation_cbf_gain=float(task_values.get("navigation_cbf_gain", 4.0)),
            navigation_cbf_influence_distance_m=_optional_float(
                task_values.get("navigation_cbf_influence_distance_m")
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
            dynamics_config_path=_optional_path(
                config_path,
                task_values.get("dynamics_config_path"),
            ),
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
            force_strategy=force_strategy,
            admittance=_load_admittance_config(task_values),
            tracking_control=tracking_control,
            contact_admittance=contact_admittance,
            engine_cleaning_control=engine_cleaning_control,
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
                hook_values.get("show_live_diagnostics_panel", task_type != "idle")
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


def _load_tracking_control_config(
    task_values: dict[str, Any],
    low_level_values: dict[str, Any] | None = None,
) -> ScenarioTrackingControlConfig:
    task_control_values = _mapping(
        task_values.get("tracking_control", {}),
        "scenario.task.tracking_control",
    )
    values = {
        **({} if low_level_values is None else low_level_values),
        **task_control_values,
    }
    return ScenarioTrackingControlConfig(
        approach_samples=int(values.get("approach_samples", 0)),
        tracking_mode=str(values.get("tracking_mode", "waypoint")),
        trajectory_duration_s=_optional_float(values.get("trajectory_duration_s")),
        executor_position_gain=float(
            values.get(
                "arm_position_gain",
                values.get("executor_position_gain", 4.0),
            )
        ),
        observer_position_gain=float(
            values.get(
                "arm_position_gain",
                values.get("observer_position_gain", 5.0),
            )
        ),
        feedforward_speed_mps=float(values.get("feedforward_speed_mps", 0.0)),
        max_target_speed_mps=_optional_float(values.get("max_target_speed_mps")),
        executor_tracking_weight=float(values.get("executor_tracking_weight", 100.0)),
        observer_tracking_weight=float(values.get("observer_tracking_weight", 40.0)),
        executor_collision_avoidance_weight=float(
            values.get("executor_collision_avoidance_weight", 80.0)
        ),
        base_regularization_weight=float(values.get("base_regularization_weight", 1.0)),
        tendon_regularization_weight=float(
            values.get("tendon_regularization_weight", 0.2)
        ),
        rank_tolerance=float(values.get("rank_tolerance", 1.0e-9)),
        minimum_singular_value=float(values.get("minimum_singular_value", 1.0e-5)),
        nominal_damping=float(values.get("nominal_damping", 1.0e-4)),
        maximum_damping=float(values.get("maximum_damping", 5.0e-2)),
        minimum_velocity_scale=float(values.get("minimum_velocity_scale", 0.05)),
        decouple_arm_singularity=bool(
            values.get("decouple_arm_singularity", False)
        ),
        singularity_strategy=str(values.get("singularity_strategy", "svd_projection")),
        enforce_target_speed_limit=bool(
            values.get("enforce_target_speed_limit", False)
        ),
        enforce_solver_velocity_limits=bool(
            values.get("enforce_solver_velocity_limits", False)
        ),
        enforce_backend_tendon_limits=bool(
            values.get("enforce_backend_tendon_limits", False)
        ),
    )


def _load_observer_control_config(
    task_values: dict[str, Any],
) -> ScenarioObserverControlConfig:
    values = _mapping(
        task_values.get("observer_control", {}),
        "scenario.task.observer_control",
    )
    return ScenarioObserverControlConfig(
        minimum_distance_m=float(values.get("minimum_distance_m", 0.010)),
        influence_distance_m=float(values.get("influence_distance_m", 0.050)),
        critical_distance_m=float(values.get("critical_distance_m", 0.008)),
        release_margin_m=float(values.get("release_margin_m", 0.005)),
        avoidance_gain=float(values.get("avoidance_gain", 0.4)),
        max_avoidance_speed_mps=_optional_float(
            values.get("max_avoidance_speed_mps")
        ),
    )


def _load_force_strategy_config(
    task_values: dict[str, Any],
    wiping_control_type: str,
) -> ScenarioForceStrategyConfig:
    values = task_values.get("force_strategy")
    if values is None:
        strategy_type = {
            "contact_distance": "contact_distance",
            "hybrid_force_position": "kinematic_hybrid",
            "dynamic_adaptive_impedance": "dynamic_adaptive_impedance",
            "contact_triggered_admittance": "contact_triggered_admittance",
        }[wiping_control_type]
    else:
        mapping = _mapping(values, "scenario.task.force_strategy")
        strategy_type = str(mapping.get("type", "contact_distance"))
    if strategy_type not in WIPING_FORCE_STRATEGY_TYPES:
        raise ValueError(
            "scenario.task.force_strategy.type must be one of "
            f"{WIPING_FORCE_STRATEGY_TYPES}."
        )
    return ScenarioForceStrategyConfig(type=strategy_type)


def _load_admittance_config(task_values: dict[str, Any]) -> ScenarioAdmittanceConfig:
    values = _mapping(
        task_values.get("admittance", {}),
        "scenario.task.admittance",
    )
    return ScenarioAdmittanceConfig(
        target_normal_force_n=float(
            values.get(
                "target_normal_force_n",
                task_values.get("target_normal_force_n", 0.0),
            )
        ),
        contact_force_threshold_n=float(values.get("contact_force_threshold_n", 0.1)),
        tangent_tolerance_m=float(values.get("tangent_tolerance_m", 1.0e-3)),
        force_tolerance_n=float(values.get("force_tolerance_n", 0.08)),
        stable_steps_required=int(values.get("stable_steps_required", 1)),
        max_steps_per_target=int(values.get("max_steps_per_target", 100)),
        position_gain=float(values.get("position_gain", 10.0)),
        kp_force=float(values.get("kp_force", 0.5)),
        ki_force=float(values.get("ki_force", 0.012)),
        admittance_mass=float(values.get("admittance_mass", 1.0)),
        admittance_damping=float(values.get("admittance_damping", 20.0)),
        admittance_stiffness=float(values.get("admittance_stiffness", 5.0)),
        admittance_clip_m=float(values.get("admittance_clip_m", 0.012)),
        force_deadband_n=float(values.get("force_deadband_n", 0.03)),
        force_filter_alpha=float(values.get("force_filter_alpha", 0.1)),
        max_tangent_velocity_m_s=float(values.get("max_tangent_velocity_m_s", 0.012)),
        max_normal_velocity_m_s=float(values.get("max_normal_velocity_m_s", 0.010)),
        enforce_velocity_limits=bool(values.get("enforce_velocity_limits", False)),
    )


def _load_low_level_control_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = load_yaml(path)
    return _mapping(
        raw.get("low_level_control"),
        "low_level_control",
    )


def _load_contact_admittance_config(
    task_values: dict[str, Any],
) -> ContactTriggeredAdmittanceConfig | None:
    values = task_values.get("contact_admittance")
    if values is None:
        return None
    mapping = _mapping(values, "scenario.task.contact_admittance")
    return ContactTriggeredAdmittanceConfig(
        target_normal_force_n=float(
            mapping.get(
                "target_normal_force_n",
                task_values.get("target_normal_force_n", 0.0),
            )
        ),
        contact_force_threshold_n=float(
            mapping.get("contact_force_threshold_n", 0.1)
        ),
        tangent_tolerance_m=float(mapping.get("tangent_tolerance_m", 1.0e-3)),
        force_tolerance_n=float(mapping.get("force_tolerance_n", 0.08)),
        stable_steps_required=int(mapping.get("stable_steps_required", 1)),
        max_steps_per_target=int(mapping.get("max_steps_per_target", 100)),
        position_gain=float(mapping.get("position_gain", 10.0)),
        kp_force=float(mapping.get("kp_force", 0.5)),
        ki_force=float(mapping.get("ki_force", 0.012)),
        admittance_mass=float(mapping.get("admittance_mass", 1.0)),
        admittance_damping=float(mapping.get("admittance_damping", 20.0)),
        admittance_stiffness=float(mapping.get("admittance_stiffness", 5.0)),
        admittance_clip_m=float(mapping.get("admittance_clip_m", 0.012)),
        force_deadband_n=float(mapping.get("force_deadband_n", 0.03)),
        force_filter_alpha=float(mapping.get("force_filter_alpha", 0.1)),
        max_tangent_velocity_m_s=float(
            mapping.get("max_tangent_velocity_m_s", 0.012)
        ),
        max_normal_velocity_m_s=float(
            mapping.get("max_normal_velocity_m_s", 0.010)
        ),
        enforce_velocity_limits=bool(mapping.get("enforce_velocity_limits", False)),
    )


def _load_engine_cleaning_control_config(
    task_values: dict[str, Any],
) -> EngineCleaningControllerGains | None:
    values = task_values.get("engine_cleaning_control")
    if values is None:
        return None
    mapping = _mapping(values, "scenario.task.engine_cleaning_control")
    return EngineCleaningControllerGains(
        tangential_position_gain=float(mapping.get("tangential_position_gain", 8.0)),
        normal_position_gain=float(mapping.get("normal_position_gain", 3.0)),
        normal_force_gain=float(
            mapping.get(
                "normal_force_gain",
                max(float(task_values.get("normal_force_gain", 0.0)), 0.001),
            )
        ),
        approach_position_gain=float(mapping.get("approach_position_gain", 5.0)),
        retreat_position_gain=float(mapping.get("retreat_position_gain", 5.0)),
        max_tcp_speed_mps=float(mapping.get("max_tcp_speed_mps", 0.03)),
        max_normal_speed_mps=float(mapping.get("max_normal_speed_mps", 0.01)),
        waypoint_tolerance_m=float(
            mapping.get(
                "waypoint_tolerance_m",
                task_values.get("waypoint_tolerance_m", 0.001),
            )
        ),
        max_contact_force_n=float(
            mapping.get(
                "max_contact_force_n",
                task_values.get("max_contact_force_n", 5.0),
            )
        ),
        force_deadband_n=float(mapping.get("force_deadband_n", 0.05)),
        min_clearance_m=float(
            mapping.get("min_clearance_m", task_values.get("min_clearance_m", 0.0))
        ),
    )


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
