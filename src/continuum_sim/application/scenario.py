"""YAML model for scenario-driven single/dual system experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continuum_sim.config import (
    MUJOCO_CAMERA_FOLLOW_MODES,
    MujocoViewerCameraConfig,
    load_yaml,
)
from continuum_sim.config_validation import (
    choice_value as _choice_value,
    float_tuple as _float_tuple,
    positive_float_value as _positive_float_value,
    resolve_path,
)
from continuum_sim.control.contact_triggered_admittance import (
    ContactTriggeredAdmittanceConfig,
)
from continuum_sim.control.tendon_rate_control import (
    BendingRateServoConfig,
    TENDON_INNER_LOOP_MODES,
)
from continuum_sim.control.task_intent import CONTACT_FORCE_FEEDBACK_MODES
from continuum_sim.control.priority_stack import PriorityStackConfig
from continuum_sim.control.waypoint_scheduler import WAYPOINT_ADVANCE_MODES
from continuum_sim.kinematics.pcc import (
    DEFAULT_PCC_KINEMATICS_MODE,
    PCC_KINEMATICS_MODES,
    PCCKinematicsMode,
)
from continuum_sim.tasks.engine_navigation import EngineNavigationSpec
from continuum_sim.tasks.navigation_mission import NavigationMissionSpec
from continuum_sim.tasks.trajectory_generation import TrajectorySpec
from continuum_sim.tasks.wiping_path import WipingPathSpec


BACKEND_TYPES = ("mujoco",)
ARM_MODES = ("dual", "single")
TASK_TYPES = (
    "idle",
    "tracking",
    "navigation",
    "wiping",
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
OBSERVER_CONTROL_MODES = (
    "tracking",
    "collision_avoidance",
    "visual_servo",
    "disabled",
)
SINGULARITY_STRATEGIES = ("damping_scale", "svd_projection")
EXECUTOR_ORIENTATION_TRACKING_MODES = ("disabled", "weighted", "nullspace")
EXECUTOR_SCENE_AVOIDANCE_MODES = ("disabled", "nullspace_after_pose")
OBSERVER_SCENE_AVOIDANCE_MODES = (
    "disabled",
    "nullspace_after_interarm_lookat",
)
WAYPOINT_ORIENTATION_SOURCES = (
    "none",
    "explicit",
    "explicit_directions",
    "insertion_direction",
    "nearest_clearance",
)


@dataclass(frozen=True)
class ScenarioBackendConfig:
    type: str
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE
    mujoco_config_path: Path | None = None
    source_xml_path: Path | None = None
    generated_xml_path: Path | None = None
    retain_arm: str | None = None
    mujoco_viewer_camera: MujocoViewerCameraConfig | None = None

    def __post_init__(self) -> None:
        if self.kinematics_mode not in PCC_KINEMATICS_MODES:
            raise ValueError(
                f"backend.kinematics_mode must be one of {PCC_KINEMATICS_MODES}."
            )


@dataclass(frozen=True)
class ScenarioSceneConfig:
    engine_config_path: Path | None = None
    structured_config_path: Path | None = None


@dataclass(frozen=True)
class ScenarioTendonInnerLoopConfig:
    """MuJoCo tracking-only tendon target servo configuration."""

    mode: str = "legacy"
    rate_filter_time_constant_s: float = 0.06
    feedforward_lead_time_s: float = 0.0
    rate_proportional_time_s: float = 0.0
    rate_integral_gain: float = 1.0
    anti_windup_gain: float = 1.0
    enforce_target_lead_limit: bool = True
    max_target_lead_m: float | None = None
    soft_force_limit_n: float | None = None
    hard_force_limit_n: float | None = None
    zero_command_mode: str = "hold"
    zero_rate_tolerance_mps: float = 1.0e-7

    def __post_init__(self) -> None:
        if self.mode not in TENDON_INNER_LOOP_MODES:
            raise ValueError(
                "tracking_control.tendon_inner_loop.mode must be one of "
                f"{TENDON_INNER_LOOP_MODES}."
            )
        BendingRateServoConfig(
            rate_filter_time_constant_s=self.rate_filter_time_constant_s,
            feedforward_lead_time_s=self.feedforward_lead_time_s,
            rate_proportional_time_s=self.rate_proportional_time_s,
            rate_integral_gain=self.rate_integral_gain,
            anti_windup_gain=self.anti_windup_gain,
            enforce_target_lead_limit=self.enforce_target_lead_limit,
            max_target_lead_m=self.max_target_lead_m,
            soft_force_limit_n=self.soft_force_limit_n,
            hard_force_limit_n=self.hard_force_limit_n,
            zero_command_mode=self.zero_command_mode,
            zero_rate_tolerance_mps=self.zero_rate_tolerance_mps,
        )


@dataclass(frozen=True)
class ScenarioOnlineReachabilityConfig:
    """Online waypoint reachability scoring and auto-advance configuration."""

    enabled: bool = True
    auto_advance_enabled: bool = True
    score_threshold: float = 0.3
    window_steps: int = 25
    min_steps_before_auto_advance: int = 50
    low_score_patience_steps: int = 25
    good_progress_mps: float = 0.001
    good_tendon_speed_ratio: float = 0.75
    good_alignment: float = 0.8
    bad_model_residual_mps: float = 0.005

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(
                "tracking_control.online_reachability.score_threshold "
                "must be in [0, 1]."
            )
        if self.window_steps <= 1:
            raise ValueError(
                "tracking_control.online_reachability.window_steps must be greater than 1."
            )
        if self.min_steps_before_auto_advance < 0:
            raise ValueError(
                "tracking_control.online_reachability.min_steps_before_auto_advance "
                "must be non-negative."
            )
        if self.low_score_patience_steps <= 0:
            raise ValueError(
                "tracking_control.online_reachability.low_score_patience_steps "
                "must be positive."
            )
        positive = {
            "good_progress_mps": self.good_progress_mps,
            "good_tendon_speed_ratio": self.good_tendon_speed_ratio,
            "good_alignment": self.good_alignment,
            "bad_model_residual_mps": self.bad_model_residual_mps,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"tracking_control.online_reachability.{name} must be "
                    "positive and finite."
                )


@dataclass(frozen=True)
class ScenarioTrackingControlConfig:
    """Scenario-native trajectory-tracking controller parameters."""

    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE
    approach_samples: int = 0
    tracking_mode: str = "waypoint"
    trajectory_duration_s: float | None = None
    max_steps_per_waypoint: int | None = None
    stage_mobile_base: bool = False
    base_position_gain: float = 1.5
    base_orientation_gain: float = 2.0
    base_position_tolerance_m: float = 0.005
    base_orientation_tolerance_rad: float = 0.035
    base_approach_standoff_m: float = 0.030
    base_approach_z_bias: float = 1.0
    intermediate_waypoints_per_waypoint: int = 0
    executor_position_gain: float = 4.0
    executor_orientation_gain: float = 2.0
    observer_position_gain: float = 5.0
    feedforward_gain: float = 1.0
    feedforward_speed_mps: float = 0.0
    max_target_speed_mps: float | None = None
    max_target_angular_speed_rad_s: float | None = None
    executor_tracking_weight: float = 100.0
    executor_orientation_tracking_weight: float = 20.0
    executor_orientation_tracking_mode: str = "nullspace"
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
    tendon_inner_loop: ScenarioTendonInnerLoopConfig = field(
        default_factory=ScenarioTendonInnerLoopConfig
    )
    online_reachability: ScenarioOnlineReachabilityConfig = field(
        default_factory=ScenarioOnlineReachabilityConfig
    )
    priority_stack: PriorityStackConfig = field(default_factory=PriorityStackConfig)

    def __post_init__(self) -> None:
        if self.kinematics_mode not in PCC_KINEMATICS_MODES:
            raise ValueError(
                "tracking_control.kinematics_mode must be one of "
                f"{PCC_KINEMATICS_MODES}."
            )
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
        if self.executor_orientation_tracking_mode not in EXECUTOR_ORIENTATION_TRACKING_MODES:
            raise ValueError(
                "tracking_control.tendon_command.executor_orientation_tracking_mode "
                f"must be one of {EXECUTOR_ORIENTATION_TRACKING_MODES}."
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
        if (
            self.max_steps_per_waypoint is not None
            and self.max_steps_per_waypoint <= 0
        ):
            raise ValueError(
                "tracking_control.max_steps_per_waypoint must be positive."
            )
        if self.intermediate_waypoints_per_waypoint < 0:
            raise ValueError(
                "tracking_control.intermediate_waypoints_per_waypoint must be non-negative."
            )
        positive = {
            "base_position_gain": self.base_position_gain,
            "base_orientation_gain": self.base_orientation_gain,
            "base_position_tolerance_m": self.base_position_tolerance_m,
            "base_orientation_tolerance_rad": self.base_orientation_tolerance_rad,
            "executor_position_gain": self.executor_position_gain,
            "executor_orientation_gain": self.executor_orientation_gain,
            "observer_position_gain": self.observer_position_gain,
            "executor_tracking_weight": self.executor_tracking_weight,
            "executor_orientation_tracking_weight": (
                self.executor_orientation_tracking_weight
            ),
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
        if (
            not np.isfinite(self.base_approach_standoff_m)
            or self.base_approach_standoff_m < 0.0
        ):
            raise ValueError(
                "tracking_control.base_approach_standoff_m must be non-negative and finite."
            )
        if not np.isfinite(self.base_approach_z_bias):
            raise ValueError(
                "tracking_control.base_approach_z_bias must be finite."
            )
        if not np.isfinite(self.feedforward_speed_mps) or self.feedforward_speed_mps < 0.0:
            raise ValueError(
                "tracking_control.feedforward_speed_mps must be non-negative and finite."
            )
        if not np.isfinite(self.feedforward_gain) or self.feedforward_gain < 0.0:
            raise ValueError(
                "tracking_control.feedforward_gain must be non-negative and finite."
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
        if self.max_target_angular_speed_rad_s is not None and (
            not np.isfinite(self.max_target_angular_speed_rad_s)
            or self.max_target_angular_speed_rad_s <= 0.0
        ):
            raise ValueError(
                "tracking_control.max_target_angular_speed_rad_s must be positive and finite."
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
    collision_pair_count: int = 1
    collision_pair_index_separation: int = 1
    look_at_executor_tip: bool = False
    look_at_gain: float = 1.0
    look_at_weight: float = 10.0
    look_at_distance_m: float = 0.010
    look_at_max_speed_mps: float | None = 0.005
    look_at_max_angular_speed_rad_s: float | None = None
    visual_servo_center_gain: float = 1.0
    visual_servo_depth_gain: float = 1.0
    visual_servo_depth_target_m: float = 0.08
    visual_servo_max_speed_mps: float | None = 0.010
    visual_servo_max_angular_speed_rad_s: float | None = 1.0

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
        if self.collision_pair_count <= 0:
            raise ValueError("observer_control.collision_pair_count must be positive.")
        if self.collision_pair_index_separation < 1:
            raise ValueError(
                "observer_control.collision_pair_index_separation must be at least 1."
            )
        positive = {
            "look_at_gain": self.look_at_gain,
            "look_at_weight": self.look_at_weight,
            "look_at_distance_m": self.look_at_distance_m,
            "visual_servo_center_gain": self.visual_servo_center_gain,
            "visual_servo_depth_gain": self.visual_servo_depth_gain,
            "visual_servo_depth_target_m": self.visual_servo_depth_target_m,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"observer_control.{name} must be positive and finite.")
        if self.max_avoidance_speed_mps is not None and (
            not np.isfinite(self.max_avoidance_speed_mps)
            or self.max_avoidance_speed_mps <= 0.0
        ):
            raise ValueError(
                "observer_control.max_avoidance_speed_mps must be positive and finite."
            )
        if self.look_at_max_speed_mps is not None and (
            not np.isfinite(self.look_at_max_speed_mps)
            or self.look_at_max_speed_mps <= 0.0
        ):
            raise ValueError(
                "observer_control.look_at_max_speed_mps must be positive and finite."
            )
        if self.look_at_max_angular_speed_rad_s is not None and (
            not np.isfinite(self.look_at_max_angular_speed_rad_s)
            or self.look_at_max_angular_speed_rad_s <= 0.0
        ):
            raise ValueError(
                "observer_control.look_at_max_angular_speed_rad_s must be positive and finite."
            )
        if self.visual_servo_max_speed_mps is not None and (
            not np.isfinite(self.visual_servo_max_speed_mps)
            or self.visual_servo_max_speed_mps <= 0.0
        ):
            raise ValueError(
                "observer_control.visual_servo.max_speed_mps must be positive and finite."
            )
        if self.visual_servo_max_angular_speed_rad_s is not None and (
            not np.isfinite(self.visual_servo_max_angular_speed_rad_s)
            or self.visual_servo_max_angular_speed_rad_s <= 0.0
        ):
            raise ValueError(
                "observer_control.visual_servo.max_angular_speed_rad_s must be positive and finite."
            )


@dataclass(frozen=True)
class ScenarioSceneAvoidanceConfig:
    """Task-level scene collision-avoidance policy."""

    enabled: bool = True
    executor_mode: str = "nullspace_after_pose"
    observer_mode: str = "nullspace_after_interarm_lookat"
    engine_min_clearance_m: float = 0.010
    engine_influence_distance_m: float = 0.025
    engine_avoidance_gain: float = 4.0

    def __post_init__(self) -> None:
        if self.executor_mode not in EXECUTOR_SCENE_AVOIDANCE_MODES:
            raise ValueError(
                "scene_avoidance.executor_mode must be one of "
                f"{EXECUTOR_SCENE_AVOIDANCE_MODES}."
            )
        if self.observer_mode not in OBSERVER_SCENE_AVOIDANCE_MODES:
            raise ValueError(
                "scene_avoidance.observer_mode must be one of "
                f"{OBSERVER_SCENE_AVOIDANCE_MODES}."
            )
        if (
            not np.isfinite(self.engine_min_clearance_m)
            or self.engine_min_clearance_m < 0.0
        ):
            raise ValueError(
                "scene_avoidance.engine_min_clearance_m must be non-negative and finite."
            )
        if (
            not np.isfinite(self.engine_influence_distance_m)
            or self.engine_influence_distance_m < 0.0
        ):
            raise ValueError(
                "scene_avoidance.engine_influence_distance_m must be non-negative and finite."
            )
        if self.engine_influence_distance_m < self.engine_min_clearance_m:
            raise ValueError(
                "scene_avoidance.engine_influence_distance_m must be at least "
                "engine_min_clearance_m."
            )
        if (
            not np.isfinite(self.engine_avoidance_gain)
            or self.engine_avoidance_gain <= 0.0
        ):
            raise ValueError(
                "scene_avoidance.engine_avoidance_gain must be positive and finite."
            )


@dataclass(frozen=True)
class ScenarioForceStrategyConfig:
    type: str = "contact_distance"


@dataclass(frozen=True)
class ScenarioTaskConfig:
    type: str
    waypoints_world: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )
    waypoint_source: str = "waypoints_world"
    waypoint_tolerance_m: float = 0.001
    pose_servo_enabled: bool = False
    waypoint_orientations_world_wxyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=float)
    )
    waypoint_directions_world: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )
    waypoint_orientation_source: str = "none"
    orientation_tolerance_rad: float = 0.08
    orientation_roll_reference_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )
    observer_roi_world: np.ndarray | None = None
    observer_control_mode: str = "collision_avoidance"
    observer_control: ScenarioObserverControlConfig = field(
        default_factory=ScenarioObserverControlConfig
    )
    scene_avoidance: ScenarioSceneAvoidanceConfig = field(
        default_factory=ScenarioSceneAvoidanceConfig
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
    force_feedback_mode: str = "proxy_distance"
    max_normal_velocity_m_s: float = 0.03
    force_control_weight: float = 20.0
    max_contact_force_n: float | None = None
    contact_loss_tolerance_steps: int = 20
    force_strategy: ScenarioForceStrategyConfig = field(
        default_factory=ScenarioForceStrategyConfig
    )
    contact_admittance: ContactTriggeredAdmittanceConfig | None = None
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
    show_live_diagnostics_panel: bool = True
    live_diagnostics_panel_stride: int = 5
    live_diagnostics_panel_history_points: int = 300
    show_observer_camera: bool = False
    observer_camera_stride: int = 1


@dataclass(frozen=True)
class ScenarioArtifactConfig:
    enabled: bool
    output_root: Path
    save_npz: bool
    save_plots: bool
    save_gif: bool
    save_mp4: bool
    save_model: bool
    save_mujoco_pcc_diagnostics: bool = True
    mujoco_pcc_diagnostics_stride: int = 1
    video_mode: str = "replay"
    video_fps: int = 20
    video_stride: int | None = None


@dataclass(frozen=True)
class ScenarioConfig:
    path: Path
    name: str
    arm_mode: str
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
    backend_values = _mapping(values.get("backend"), "scenario.backend")
    arm_mode = _scenario_arm_mode(values, backend_values)
    low_level_control_path = _optional_path(
        config_path,
        values.get("low_level_control_path"),
    )
    low_level_control_values = _load_low_level_control_values(
        low_level_control_path
    )
    backend_type = str(backend_values.get("type", "analytic"))
    if backend_type not in BACKEND_TYPES:
        raise ValueError(f"scenario.backend.type must be one of {BACKEND_TYPES}.")
    kinematics_mode = _kinematics_mode(
        backend_values.get("kinematics_mode", DEFAULT_PCC_KINEMATICS_MODE),
        "scenario.backend.kinematics_mode",
    )
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
    pose_servo_values = _mapping(
        task_values.get("pose_servo", {}),
        "scenario.task.pose_servo",
    )
    pose_servo_enabled_value = pose_servo_values.get("enabled")
    pose_servo_explicitly_disabled = pose_servo_enabled_value is False
    pose_servo_enabled = bool(pose_servo_enabled_value)
    waypoint_orientation_source = str(
        pose_servo_values.get("orientation_source", "none")
    )
    if waypoint_orientation_source not in WAYPOINT_ORIENTATION_SOURCES:
        raise ValueError(
            "scenario.task.pose_servo.orientation_source must be one of "
            f"{WAYPOINT_ORIENTATION_SOURCES}."
        )
    waypoint_orientations = _waypoint_orientations(
        task_values.get(
            "waypoint_orientations_world_wxyz",
            pose_servo_values.get("waypoint_orientations_world_wxyz", []),
        )
    )
    waypoint_directions = _waypoint_directions(
        task_values.get(
            "waypoint_directions_world",
            pose_servo_values.get("waypoint_directions_world", []),
        )
    )
    if waypoint_orientations.shape[0] > 0 and waypoint_source == "waypoints_world":
        if waypoint_orientations.shape[0] != waypoints.shape[0]:
            raise ValueError(
                "scenario.task.waypoint_orientations_world_wxyz must match "
                "waypoints_world count."
            )
        if not pose_servo_explicitly_disabled:
            waypoint_orientation_source = "explicit"
            pose_servo_enabled = True
    if waypoint_directions.shape[0] > 0 and waypoint_source == "waypoints_world":
        if waypoint_directions.shape[0] != waypoints.shape[0]:
            raise ValueError(
                "scenario.task.pose_servo.waypoint_directions_world must match "
                "waypoints_world count."
            )
        if waypoint_orientations.shape[0] > 0:
            raise ValueError(
                "Provide either waypoint_orientations_world_wxyz or "
                "waypoint_directions_world, not both."
            )
        if not pose_servo_explicitly_disabled:
            waypoint_orientation_source = "explicit_directions"
            pose_servo_enabled = True
    if pose_servo_explicitly_disabled:
        waypoint_orientation_source = "none"
    orientation_tolerance_rad = float(
        pose_servo_values.get("orientation_tolerance_rad", 0.08)
    )
    if (
        not np.isfinite(orientation_tolerance_rad)
        or orientation_tolerance_rad < 0.0
    ):
        raise ValueError(
            "scenario.task.pose_servo.orientation_tolerance_rad must be non-negative."
        )
    orientation_roll_reference = np.asarray(
        pose_servo_values.get("roll_reference_world", [0.0, 0.0, 1.0]),
        dtype=float,
    )
    if (
        orientation_roll_reference.shape != (3,)
        or not np.all(np.isfinite(orientation_roll_reference))
        or np.linalg.norm(orientation_roll_reference) <= 1.0e-12
    ):
        raise ValueError(
            "scenario.task.pose_servo.roll_reference_world must be a finite "
            "nonzero 3-vector."
        )
    orientation_roll_reference = (
        orientation_roll_reference / np.linalg.norm(orientation_roll_reference)
    )
    runtime_values = _mapping(values.get("runtime", {}), "scenario.runtime")
    hook_values = _mapping(values.get("hooks", {}), "scenario.hooks")
    scene_values = _mapping(values.get("scene", {}), "scenario.scene")
    artifact_values = _mapping(values.get("artifacts", {}), "scenario.artifacts")
    video_mode = str(artifact_values.get("video_mode", "replay"))
    if video_mode not in VIDEO_MODES:
        raise ValueError(f"scenario.artifacts.video_mode must be one of {VIDEO_MODES}.")
    save_gif = bool(artifact_values.get("save_gif", True))
    save_mp4 = bool(artifact_values.get("save_mp4", False))
    mujoco_pcc_diagnostics_stride = int(
        artifact_values.get("mujoco_pcc_diagnostics_stride", 1)
    )
    if mujoco_pcc_diagnostics_stride < 1:
        raise ValueError("scenario.artifacts.mujoco_pcc_diagnostics_stride must be >= 1.")
    if (
        backend_type != "mujoco"
        and (save_gif or save_mp4)
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
    force_feedback_mode = str(
        task_values.get("force_feedback_mode", "proxy_distance")
    )
    if force_feedback_mode not in CONTACT_FORCE_FEEDBACK_MODES:
        raise ValueError(
            "scenario.task.force_feedback_mode must be one of "
            f"{CONTACT_FORCE_FEEDBACK_MODES}."
        )
    tracking_control = _load_tracking_control_config(
        task_values,
        low_level_control_values,
        kinematics_mode=kinematics_mode,
    )
    observer_control = _load_observer_control_config(task_values)
    scene_avoidance = _load_scene_avoidance_config(task_values)
    contact_admittance = _load_contact_admittance_config(task_values)
    if (
        wiping_control_type == "contact_triggered_admittance"
        and contact_admittance is None
    ):
        contact_admittance = ContactTriggeredAdmittanceConfig(
            target_normal_force_n=float(task_values.get("target_normal_force_n", 0.0))
        )
    viewer = str(hook_values.get("viewer", "none"))
    if viewer not in ("none", "matplotlib", "mujoco"):
        raise ValueError("scenario.hooks.viewer must be none, matplotlib, or mujoco.")
    return ScenarioConfig(
        path=config_path,
        name=str(values.get("name", config_path.stem)),
        arm_mode=arm_mode,
        assembly_config_path=_arm_mode_assembly_config_path(
            config_path,
            values,
            task_values,
            arm_mode,
        ),
        low_level_control_path=low_level_control_path,
        backend=ScenarioBackendConfig(
            type=backend_type,
            kinematics_mode=kinematics_mode,
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
                _arm_mode_generated_xml_path(backend_values, arm_mode),
            ),
            retain_arm=_arm_mode_retain_arm(arm_mode),
            mujoco_viewer_camera=_load_mujoco_viewer_camera_override(
                backend_values
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
            pose_servo_enabled=pose_servo_enabled,
            waypoint_orientations_world_wxyz=waypoint_orientations,
            waypoint_directions_world=waypoint_directions,
            waypoint_orientation_source=waypoint_orientation_source,
            orientation_tolerance_rad=orientation_tolerance_rad,
            orientation_roll_reference_world=orientation_roll_reference,
            observer_roi_world=None if roi is None else roi.copy(),
            observer_control_mode=observer_control_mode,
            observer_control=observer_control,
            scene_avoidance=scene_avoidance,
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
            force_feedback_mode=force_feedback_mode,
            max_normal_velocity_m_s=float(
                task_values.get("max_normal_velocity_m_s", 0.03)
            ),
            force_control_weight=float(task_values.get("force_control_weight", 20.0)),
            max_contact_force_n=_optional_float(task_values.get("max_contact_force_n")),
            contact_loss_tolerance_steps=int(
                task_values.get("contact_loss_tolerance_steps", 20)
            ),
            force_strategy=force_strategy,
            contact_admittance=contact_admittance,
            tracking_control=tracking_control,
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
            show_observer_camera=bool(
                hook_values.get("show_observer_camera", False)
            ),
            observer_camera_stride=int(
                hook_values.get("observer_camera_stride", 1)
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
            save_gif=save_gif,
            save_mp4=save_mp4,
            save_model=bool(artifact_values.get("save_model", True)),
            save_mujoco_pcc_diagnostics=bool(
                artifact_values.get("save_mujoco_pcc_diagnostics", True)
            ),
            mujoco_pcc_diagnostics_stride=mujoco_pcc_diagnostics_stride,
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


def _kinematics_mode(value: object, name: str) -> PCCKinematicsMode:
    mode = str(value)
    if mode not in PCC_KINEMATICS_MODES:
        raise ValueError(f"{name} must be one of {PCC_KINEMATICS_MODES}.")
    return mode  # type: ignore[return-value]


def _load_mujoco_viewer_camera_override(
    backend_values: dict[str, Any],
) -> MujocoViewerCameraConfig | None:
    viewer_values = backend_values.get("viewer")
    if viewer_values is None:
        return None
    viewer = _mapping(viewer_values, "scenario.backend.viewer")
    camera_values = viewer.get("camera")
    if camera_values is None:
        return None
    camera = _mapping(camera_values, "scenario.backend.viewer.camera")
    prefix = "scenario.backend.viewer.camera"
    return MujocoViewerCameraConfig(
        lookat=_float_tuple(
            camera.get("lookat", (0.0, 0.0, 0.06)),
            f"{prefix}.lookat",
            length=3,
        ),  # type: ignore[arg-type]
        distance=_positive_float_value(
            camera.get("distance", 0.25),
            f"{prefix}.distance",
        ),
        azimuth=float(camera.get("azimuth", 135.0)),
        elevation=float(camera.get("elevation", -25.0)),
        follow=_choice_value(
            camera.get("follow", "none"),
            f"{prefix}.follow",
            MUJOCO_CAMERA_FOLLOW_MODES,
        ),
    )


def _waypoint_source(task_values: dict[str, Any]) -> str:
    sources = [
        name
        for name in (
            "waypoints_world",
            "trajectory",
            "mission",
            "wiping_path",
            "engine_navigation",
        )
        if task_values.get(name) is not None
    ]
    if not sources:
        return "waypoints_world"
    if len(sources) > 1:
        raise ValueError(f"scenario.task defines multiple waypoint sources: {sources}.")
    return sources[0]


def _scenario_arm_mode(
    values: dict[str, Any],
    backend_values: dict[str, Any],
) -> str:
    explicit = values.get("arm_mode", values.get("robot_variant"))
    if explicit is not None:
        arm_mode = str(explicit)
        if arm_mode not in ARM_MODES:
            raise ValueError(f"scenario.arm_mode must be one of {ARM_MODES}.")
        return arm_mode
    assembly_hint = str(values.get("assembly_config_path", "")).replace("\\", "/").lower()
    retain_hint = backend_values.get("retain_arm")
    if retain_hint is not None and str(retain_hint) == "executor":
        return "single"
    if "single_spatial" in assembly_hint or "/single_" in assembly_hint:
        return "single"
    return "dual"


def _arm_mode_assembly_config_path(
    config_path: Path,
    values: dict[str, Any],
    task_values: dict[str, Any],
    arm_mode: str,
) -> Path:
    raw = values.get("assembly_config_path")
    if raw is None:
        raw = _default_assembly_config_path(task_values, arm_mode)
    return resolve_path(config_path, _replace_arm_mode_token(str(raw), arm_mode))


def _default_assembly_config_path(
    task_values: dict[str, Any],
    arm_mode: str,
) -> str:
    task_type = str(task_values.get("type", "idle"))
    mobile = task_type in ("navigation", "engine_navigation")
    prefix = "dual" if arm_mode == "dual" else "single"
    suffix = "_mobile" if mobile else ""
    return f"../robots/assemblies/{prefix}_spatial{suffix}.yaml"


def _arm_mode_generated_xml_path(
    backend_values: dict[str, Any],
    arm_mode: str,
) -> object | None:
    raw = backend_values.get("generated_xml_path")
    if raw is None:
        return None
    return _replace_arm_mode_token(str(raw), arm_mode)


def _arm_mode_retain_arm(
    arm_mode: str,
) -> str | None:
    return "executor" if arm_mode == "single" else None


def _replace_arm_mode_token(value: str, arm_mode: str) -> str:
    if arm_mode == "single":
        return (
            value.replace("dual_spatial_mobile.yaml", "single_spatial_mobile.yaml")
            .replace("dual_spatial.yaml", "single_spatial.yaml")
            .replace("scenario_dual_", "scenario_single_")
            .replace("dual_", "single_")
        )
    return (
        value.replace("single_spatial_mobile.yaml", "dual_spatial_mobile.yaml")
        .replace("single_spatial.yaml", "dual_spatial.yaml")
        .replace("scenario_single_", "scenario_dual_")
        .replace("single_", "dual_")
    )


def _load_tracking_control_config(
    task_values: dict[str, Any],
    low_level_values: dict[str, Any] | None = None,
    *,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> ScenarioTrackingControlConfig:
    task_control_values = _mapping(
        task_values.get("tracking_control", {}),
        "scenario.task.tracking_control",
    )
    low_level_control_values = {} if low_level_values is None else low_level_values
    task_space_values = _mapping(
        low_level_control_values.get("task_space_servo", {}),
        "low_level_control.task_space_servo",
    )
    tendon_command_values = _mapping(
        low_level_control_values.get("tendon_command", {}),
        "low_level_control.tendon_command",
    )
    execution_values = _mapping(
        low_level_control_values.get("execution", {}),
        "low_level_control.execution",
    )
    low_level_inner_loop = _mapping(
        execution_values.get("tendon_inner_loop", {}),
        "low_level_control.execution.tendon_inner_loop",
    )
    task_inner_loop = _mapping(
        task_control_values.get("tendon_inner_loop", {}),
        "scenario.task.tracking_control.tendon_inner_loop",
    )
    low_level_online_reachability = _mapping(
        low_level_control_values.get("online_reachability", {}),
        "low_level_control.online_reachability",
    )
    task_online_reachability = _mapping(
        task_control_values.get("online_reachability", {}),
        "scenario.task.tracking_control.online_reachability",
    )
    online_reachability_values = {
        **low_level_online_reachability,
        **task_online_reachability,
    }
    inner_loop_values = {**low_level_inner_loop, **task_inner_loop}
    task_space_override = _mapping(
        task_control_values.get("task_space_servo", {}),
        "scenario.task.tracking_control.task_space_servo",
    )
    tendon_command_override = _mapping(
        task_control_values.get("tendon_command", {}),
        "scenario.task.tracking_control.tendon_command",
    )
    execution_override = _mapping(
        task_control_values.get("execution", {}),
        "scenario.task.tracking_control.execution",
    )
    task_space = {**task_space_values, **task_space_override}
    tendon_command = {**tendon_command_values, **tendon_command_override}
    execution = {**execution_values, **execution_override}
    priority_stack_values = {
        **_mapping(
            tendon_command_values.get("priority_stack", {}),
            "low_level_control.tendon_command.priority_stack",
        ),
        **_mapping(
            tendon_command_override.get("priority_stack", {}),
            "scenario.task.tracking_control.tendon_command.priority_stack",
        ),
        **_mapping(
            task_control_values.get("priority_stack", {}),
            "scenario.task.tracking_control.priority_stack",
        ),
    }
    return ScenarioTrackingControlConfig(
        kinematics_mode=kinematics_mode,
        approach_samples=int(task_control_values.get("approach_samples", 0)),
        tracking_mode=str(task_control_values.get("tracking_mode", "waypoint")),
        trajectory_duration_s=_optional_float(
            task_control_values.get("trajectory_duration_s")
        ),
        max_steps_per_waypoint=(
            None
            if task_control_values.get("max_steps_per_waypoint") is None
            else int(task_control_values["max_steps_per_waypoint"])
        ),
        stage_mobile_base=bool(task_control_values.get("stage_mobile_base", False)),
        base_position_gain=float(task_control_values.get("base_position_gain", 1.5)),
        base_orientation_gain=float(task_control_values.get("base_orientation_gain", 2.0)),
        base_position_tolerance_m=float(
            task_control_values.get("base_position_tolerance_m", 0.005)
        ),
        base_orientation_tolerance_rad=float(
            task_control_values.get("base_orientation_tolerance_rad", 0.035)
        ),
        base_approach_standoff_m=float(
            task_control_values.get("base_approach_standoff_m", 0.030)
        ),
        base_approach_z_bias=float(
            task_control_values.get("base_approach_z_bias", 1.0)
        ),
        intermediate_waypoints_per_waypoint=int(
            task_control_values.get("intermediate_waypoints_per_waypoint", 0)
        ),
        executor_position_gain=float(
            task_space.get("position_gain", task_space.get("executor_position_gain", 4.0))
        ),
        observer_position_gain=float(
            task_space.get("observer_position_gain", task_space.get("position_gain", 5.0))
        ),
        feedforward_gain=float(task_space.get("feedforward_gain", 1.0)),
        feedforward_speed_mps=float(task_space.get("feedforward_speed_mps", 0.0)),
        max_target_speed_mps=_optional_float(task_space.get("max_speed_mps")),
        max_target_angular_speed_rad_s=_optional_float(
            task_space.get("max_angular_speed_rad_s")
        ),
        executor_orientation_gain=float(
            task_space.get(
                "orientation_gain",
                task_space.get("executor_orientation_gain", 2.0),
            )
        ),
        executor_tracking_weight=float(
            tendon_command.get("executor_tracking_weight", 100.0)
        ),
        executor_orientation_tracking_weight=float(
            tendon_command.get("executor_orientation_tracking_weight", 20.0)
        ),
        executor_orientation_tracking_mode=str(
            tendon_command.get("executor_orientation_tracking_mode", "nullspace")
        ),
        observer_tracking_weight=float(
            tendon_command.get("observer_tracking_weight", 40.0)
        ),
        executor_collision_avoidance_weight=float(
            tendon_command.get("collision_avoidance_weight", 80.0)
        ),
        base_regularization_weight=float(
            tendon_command.get("base_regularization_weight", 1.0)
        ),
        tendon_regularization_weight=float(
            tendon_command.get("tendon_regularization_weight", 0.2)
        ),
        rank_tolerance=float(tendon_command.get("rank_tolerance", 1.0e-9)),
        minimum_singular_value=float(
            tendon_command.get("minimum_singular_value", 1.0e-5)
        ),
        nominal_damping=float(tendon_command.get("nominal_damping", 1.0e-4)),
        maximum_damping=float(tendon_command.get("maximum_damping", 5.0e-2)),
        minimum_velocity_scale=float(
            tendon_command.get("minimum_velocity_scale", 0.05)
        ),
        decouple_arm_singularity=bool(
            tendon_command.get("decouple_arm_singularity", False)
        ),
        singularity_strategy=str(
            tendon_command.get("singularity_strategy", "svd_projection")
        ),
        enforce_target_speed_limit=bool(
            task_space.get("enforce_speed_limit", False)
        ),
        enforce_solver_velocity_limits=bool(
            tendon_command.get("enforce_velocity_limits", False)
        ),
        enforce_backend_tendon_limits=bool(
            execution.get("enforce_tendon_limits", False)
        ),
        tendon_inner_loop=ScenarioTendonInnerLoopConfig(
            mode=str(inner_loop_values.get("mode", "legacy")),
            rate_filter_time_constant_s=float(
                inner_loop_values.get("rate_filter_time_constant_s", 0.06)
            ),
            feedforward_lead_time_s=float(
                inner_loop_values.get("feedforward_lead_time_s", 0.0)
            ),
            rate_proportional_time_s=float(
                inner_loop_values.get("rate_proportional_time_s", 0.0)
            ),
            rate_integral_gain=float(
                inner_loop_values.get("rate_integral_gain", 1.0)
            ),
            anti_windup_gain=float(inner_loop_values.get("anti_windup_gain", 1.0)),
            enforce_target_lead_limit=bool(
                inner_loop_values.get("enforce_target_lead_limit", True)
            ),
            max_target_lead_m=_optional_float(
                inner_loop_values.get("max_target_lead_m")
            ),
            soft_force_limit_n=_optional_float(
                inner_loop_values.get("soft_force_limit_n")
            ),
            hard_force_limit_n=_optional_float(
                inner_loop_values.get("hard_force_limit_n")
            ),
            zero_command_mode=str(
                inner_loop_values.get("zero_command_mode", "hold")
            ),
            zero_rate_tolerance_mps=float(
                inner_loop_values.get("zero_rate_tolerance_mps", 1.0e-7)
            ),
        ),
        online_reachability=ScenarioOnlineReachabilityConfig(
            enabled=bool(online_reachability_values.get("enabled", True)),
            auto_advance_enabled=bool(
                online_reachability_values.get("auto_advance_enabled", True)
            ),
            score_threshold=float(
                online_reachability_values.get("score_threshold", 0.3)
            ),
            window_steps=int(online_reachability_values.get("window_steps", 25)),
            min_steps_before_auto_advance=int(
                online_reachability_values.get(
                    "min_steps_before_auto_advance",
                    50,
                )
            ),
            low_score_patience_steps=int(
                online_reachability_values.get("low_score_patience_steps", 25)
            ),
            good_progress_mps=float(
                online_reachability_values.get("good_progress_mps", 0.001)
            ),
            good_tendon_speed_ratio=float(
                online_reachability_values.get("good_tendon_speed_ratio", 0.75)
            ),
            good_alignment=float(
                online_reachability_values.get("good_alignment", 0.8)
            ),
            bad_model_residual_mps=float(
                online_reachability_values.get("bad_model_residual_mps", 0.005)
            ),
        ),
        priority_stack=PriorityStackConfig.from_mapping(priority_stack_values),
    )


def _load_observer_control_config(
    task_values: dict[str, Any],
) -> ScenarioObserverControlConfig:
    values = _mapping(
        task_values.get("observer_control", {}),
        "scenario.task.observer_control",
    )
    visual_servo = _mapping(
        values.get("visual_servo", {}),
        "scenario.task.observer_control.visual_servo",
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
        collision_pair_count=int(values.get("collision_pair_count", 1)),
        collision_pair_index_separation=int(
            values.get("collision_pair_index_separation", 1)
        ),
        look_at_executor_tip=bool(values.get("look_at_executor_tip", False)),
        look_at_gain=float(values.get("look_at_gain", 1.0)),
        look_at_weight=float(values.get("look_at_weight", 10.0)),
        look_at_distance_m=float(values.get("look_at_distance_m", 0.010)),
        look_at_max_speed_mps=_optional_float(
            values.get("look_at_max_speed_mps", 0.005)
        ),
        look_at_max_angular_speed_rad_s=_optional_float(
            values.get("look_at_max_angular_speed_rad_s")
        ),
        visual_servo_center_gain=float(visual_servo.get("center_gain", 1.0)),
        visual_servo_depth_gain=float(visual_servo.get("depth_gain", 1.0)),
        visual_servo_depth_target_m=float(
            visual_servo.get("depth_target_m", 0.08)
        ),
        visual_servo_max_speed_mps=_optional_float(
            visual_servo.get("max_speed_mps", 0.010)
        ),
        visual_servo_max_angular_speed_rad_s=_optional_float(
            visual_servo.get("max_angular_speed_rad_s", 1.0)
        ),
    )


def _load_scene_avoidance_config(
    task_values: dict[str, Any],
) -> ScenarioSceneAvoidanceConfig:
    values = _mapping(
        task_values.get("scene_avoidance", {}),
        "scenario.task.scene_avoidance",
    )
    return ScenarioSceneAvoidanceConfig(
        enabled=bool(values.get("enabled", True)),
        executor_mode=str(values.get("executor_mode", "nullspace_after_pose")),
        observer_mode=str(
            values.get("observer_mode", "nullspace_after_interarm_lookat")
        ),
        engine_min_clearance_m=float(
            values.get("engine_min_clearance_m", 0.010)
        ),
        engine_influence_distance_m=float(
            values.get("engine_influence_distance_m", 0.025)
        ),
        engine_avoidance_gain=float(values.get("engine_avoidance_gain", 4.0)),
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


def _load_low_level_control_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = load_yaml(path)
    return _mapping(
        raw.get("low_level_control"),
        "low_level_control",
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


def _waypoint_orientations(value: object) -> np.ndarray:
    orientations = np.asarray(value, dtype=float)
    if orientations.size == 0:
        return np.zeros((0, 4), dtype=float)
    if orientations.ndim != 2 or orientations.shape[1] != 4:
        raise ValueError(
            "scenario.task.waypoint_orientations_world_wxyz must have shape (N, 4)."
        )
    norms = np.linalg.norm(orientations, axis=1)
    if np.any((~np.isfinite(norms)) | (norms <= 1.0e-12)):
        raise ValueError(
            "scenario.task.waypoint_orientations_world_wxyz rows must be finite "
            "nonzero quaternions."
        )
    return orientations / norms[:, None]


def _waypoint_directions(value: object) -> np.ndarray:
    directions = np.asarray(value, dtype=float)
    if directions.size == 0:
        return np.zeros((0, 3), dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(
            "scenario.task.pose_servo.waypoint_directions_world must have shape (N, 3)."
        )
    norms = np.linalg.norm(directions, axis=1)
    if np.any((~np.isfinite(norms)) | (norms <= 1.0e-12)):
        raise ValueError(
            "scenario.task.pose_servo.waypoint_directions_world rows must be finite "
            "nonzero 3-vectors."
        )
    return directions / norms[:, None]
