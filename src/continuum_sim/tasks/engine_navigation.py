"""Resolve staged dual-arm navigation targets from engine-scene annotations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.tasks.executor_reference import base_to_straight_executor_tcp_pose
from continuum_sim.scenes.engine_scene import (
    EngineRegionConfig,
    EngineSceneConfig,
    effective_engine_frame_position,
)


LOCAL_PATH_TYPES = (
    "transverse_square",
    "transverse_circle",
    "transverse_figure_eight",
    "transverse_ellipse",
    "transverse_line",
    "transverse_lissajous",
    "square",
    "circle",
    "figure-eight",
    "figure_eight",
    "ellipse",
    "line",
    "lissajous",
)
LOCAL_TRACKING_ADVANCE_MODES = ("tolerance", "time", "steps")


@dataclass(frozen=True)
class EngineNavigationLocalTrackingSpec:
    """Waypoint advancement policy for local executor paths."""

    advance_mode: str = "tolerance"
    advance_time_s: float | None = None
    advance_steps: int | None = None
    waypoint_tolerance_m: float | None = None
    rejoin_tolerance_m: float | None = None
    max_steps_per_waypoint: int | None = None
    transition_samples: int = 20
    executor_position_gain: float = 3.0
    max_target_speed_mps: float | None = None
    enforce_tendon_rate_limits: bool = False


@dataclass(frozen=True)
class EngineNavigationObserverControlSpec:
    """Observer tracking target and inter-arm safety policy."""

    position_gain: float = 3.0
    executor_offset_world_m: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.04, 0.02], dtype=float)
    )
    roi_blend: float = 0.25
    inter_arm_influence_distance_m: float = 0.018
    inter_arm_safe_distance_m: float = 0.014
    inter_arm_critical_distance_m: float = 0.009
    inter_arm_release_margin_m: float = 0.002
    inter_arm_avoidance_gain: float = 6.0
    inter_arm_max_avoidance_speed_mps: float | None = None
    centerline_samples_per_segment: int = 8
    observer_tracking_weight: float = 20.0
    observer_collision_weight: float = 250.0
    stop_all_on_critical_distance: bool = False


@dataclass(frozen=True)
class EngineNavigationLocalPathSpec:
    """One executor path event placed along the insertion route."""

    name: str
    at_fraction: float
    path_type: str
    radius_m: float
    samples: int
    axial_retraction_m: float
    radius_x_m: float | None = None
    radius_y_m: float | None = None
    length_m: float | None = None
    side_length_m: float | None = None
    lissajous_frequency_x: int = 2
    lissajous_frequency_y: int = 1
    lissajous_phase_deg: float = 90.0


@dataclass(frozen=True)
class EngineNavigationSpec:
    """Validated staged engine-navigation parameters."""

    entry_region: str
    insertion_path: str
    pre_entry_standoff_m: float = 0.05
    insertion_waypoint_spacing_m: float = 0.02
    base_position_tolerance_m: float = 0.005
    base_orientation_tolerance_rad: float = 0.035
    base_position_gain: float = 1.5
    base_orientation_gain: float = 2.0
    local_path_type: str = "transverse_square"
    local_path_radius_m: float = 0.01
    local_path_radius_x_m: float | None = None
    local_path_radius_y_m: float | None = None
    local_path_length_m: float | None = None
    local_path_side_length_m: float | None = None
    local_path_lissajous_frequency_x: int = 2
    local_path_lissajous_frequency_y: int = 1
    local_path_lissajous_phase_deg: float = 90.0
    local_path_samples: int = 40
    local_path_axial_retraction_m: float = 0.01
    local_path_name: str = "endpoint_square"
    intermediate_local_paths: tuple[EngineNavigationLocalPathSpec, ...] = ()
    local_tracking: EngineNavigationLocalTrackingSpec = field(
        default_factory=EngineNavigationLocalTrackingSpec
    )
    observer_control: EngineNavigationObserverControlSpec = field(
        default_factory=EngineNavigationObserverControlSpec
    )
    phase_timeout_steps: int = 5000

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "EngineNavigationSpec":
        """Load a spec from the ``task.engine_navigation`` mapping."""

        if not isinstance(values, dict):
            raise ValueError("task.engine_navigation must be a mapping.")
        local_path = values.get("local_path", {})
        if not isinstance(local_path, dict):
            raise ValueError("task.engine_navigation.local_path must be a mapping.")
        local_path_values = _merge_shape_mapping(
            local_path,
            "engine_navigation.local_path",
        )
        intermediate_local_paths = _load_intermediate_local_paths(
            values.get("intermediate_local_paths", ())
        )
        local_tracking = _load_local_tracking(values.get("local_tracking", {}))
        observer_control = _load_observer_control_overrides(
            values.get("observer_control_overrides", {})
        )
        spec = cls(
            entry_region=str(_required(values, "entry_region")),
            insertion_path=str(_required(values, "insertion_path")),
            pre_entry_standoff_m=float(values.get("pre_entry_standoff_m", 0.05)),
            insertion_waypoint_spacing_m=float(
                values.get("insertion_waypoint_spacing_m", 0.02)
            ),
            base_position_tolerance_m=float(
                values.get("base_position_tolerance_m", 0.005)
            ),
            base_orientation_tolerance_rad=float(
                values.get("base_orientation_tolerance_rad", 0.035)
            ),
            base_position_gain=float(values.get("base_position_gain", 1.5)),
            base_orientation_gain=float(values.get("base_orientation_gain", 2.0)),
            local_path_type=str(local_path_values.get("type", "transverse_square")),
            local_path_radius_m=float(local_path_values.get("radius_m", 0.01)),
            local_path_radius_x_m=_optional_float(
                local_path_values.get("radius_x_m")
            ),
            local_path_radius_y_m=_optional_float(
                local_path_values.get("radius_y_m")
            ),
            local_path_length_m=_optional_float(local_path_values.get("length_m")),
            local_path_side_length_m=_optional_float(
                local_path_values.get("side_length_m")
            ),
            local_path_lissajous_frequency_x=int(
                local_path_values.get("lissajous_frequency_x", 2)
            ),
            local_path_lissajous_frequency_y=int(
                local_path_values.get("lissajous_frequency_y", 1)
            ),
            local_path_lissajous_phase_deg=float(
                local_path_values.get("lissajous_phase_deg", 90.0)
            ),
            local_path_samples=int(local_path_values.get("samples", 40)),
            local_path_axial_retraction_m=float(
                local_path_values.get("axial_retraction_m", 0.01)
            ),
            local_path_name=str(local_path_values.get("name", "endpoint_square")),
            intermediate_local_paths=intermediate_local_paths,
            local_tracking=local_tracking,
            observer_control=observer_control,
            phase_timeout_steps=int(values.get("phase_timeout_steps", 5000)),
        )
        spec._validate()
        return spec

    def _validate(self) -> None:
        if not self.entry_region:
            raise ValueError("engine_navigation.entry_region must be non-empty.")
        if not self.insertion_path:
            raise ValueError("engine_navigation.insertion_path must be non-empty.")
        if not self.local_path_name:
            raise ValueError("engine_navigation.local_path.name must be non-empty.")
        positive = {
            "pre_entry_standoff_m": self.pre_entry_standoff_m,
            "insertion_waypoint_spacing_m": self.insertion_waypoint_spacing_m,
            "base_position_tolerance_m": self.base_position_tolerance_m,
            "base_orientation_tolerance_rad": self.base_orientation_tolerance_rad,
            "base_position_gain": self.base_position_gain,
            "base_orientation_gain": self.base_orientation_gain,
            "local_path_radius_m": self.local_path_radius_m,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"engine_navigation.{name} must be positive and finite.")
        if self.local_path_type not in LOCAL_PATH_TYPES:
            raise ValueError(
                f"engine_navigation.local_path.type must be one of {LOCAL_PATH_TYPES}."
            )
        if self.local_path_samples < 4:
            raise ValueError("engine_navigation.local_path.samples must be at least 4.")
        _validate_local_path_shape(
            "engine_navigation.local_path",
            path_type=self.local_path_type,
            radius_m=self.local_path_radius_m,
            radius_x_m=self.local_path_radius_x_m,
            radius_y_m=self.local_path_radius_y_m,
            length_m=self.local_path_length_m,
            side_length_m=self.local_path_side_length_m,
            lissajous_frequency_x=self.local_path_lissajous_frequency_x,
            lissajous_frequency_y=self.local_path_lissajous_frequency_y,
        )
        if (
            not np.isfinite(self.local_path_axial_retraction_m)
            or self.local_path_axial_retraction_m < 0.0
        ):
            raise ValueError(
                "engine_navigation.local_path.axial_retraction_m must be "
                "finite and non-negative."
            )
        fractions = [path.at_fraction for path in self.intermediate_local_paths]
        if fractions != sorted(fractions) or len(fractions) != len(set(fractions)):
            raise ValueError(
                "engine_navigation.intermediate_local_paths fractions must be "
                "unique and in ascending order."
            )
        names = [
            *(path.name for path in self.intermediate_local_paths),
            self.local_path_name,
        ]
        if len(names) != len(set(names)):
            raise ValueError(
                "engine_navigation local path names must be unique."
            )
        if self.phase_timeout_steps <= 0:
            raise ValueError("engine_navigation.phase_timeout_steps must be positive.")


@dataclass(frozen=True)
class EngineNavigationPlan:
    """Resolved world-frame staged-navigation targets."""

    pre_entry_tip_world: np.ndarray
    insertion_direction_world: np.ndarray
    insertion_tip_waypoints_world: np.ndarray
    pre_entry_base_pose: Pose6D
    insertion_base_poses: tuple[Pose6D, ...]
    executor_waypoints_world: np.ndarray
    observer_roi_world: np.ndarray
    local_path_plans: tuple["EngineNavigationLocalPathPlan", ...] = ()


@dataclass(frozen=True)
class EngineNavigationLocalPathPlan:
    """Resolved world-frame executor event tied to one insertion waypoint."""

    name: str
    path_type: str
    at_fraction: float
    insertion_index: int
    insertion_target_world: np.ndarray
    center_world: np.ndarray
    waypoints_world: np.ndarray
    is_terminal: bool
    transition_waypoints_world: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )


def resolve_engine_navigation_plan(
    spec: EngineNavigationSpec,
    scene: EngineSceneConfig,
    assembly: RobotAssemblyConfig,
) -> EngineNavigationPlan:
    """Resolve named engine annotations into base and arm targets."""

    if assembly.base.control_mode == "fixed":
        raise ValueError("engine_navigation requires a non-fixed mobile base.")
    executors = [arm for arm in assembly.enabled_arms if arm.role == "executor"]
    observers = [arm for arm in assembly.enabled_arms if arm.role == "observer"]
    if len(executors) != 1 or len(observers) > 1:
        raise ValueError(
            "engine_navigation requires exactly one enabled executor and at "
            "most one observer."
        )

    region = _named_region(scene, spec.entry_region)
    entry_point = _region_point_world(region, scene)
    path = _named_path_world(scene, spec.insertion_path)
    if np.linalg.norm(path[0] - entry_point) > 0.01:
        raise ValueError(
            f"Engine path {spec.insertion_path!r} does not start at region "
            f"{spec.entry_region!r} within 0.01 m."
        )
    insertion_waypoints = _resample_polyline(
        path,
        spec.insertion_waypoint_spacing_m,
    )
    insertion_direction = _unit(path[1] - path[0], "insertion path direction")
    pre_entry_tip = entry_point - spec.pre_entry_standoff_m * insertion_direction
    tip_orientation = _orientation_along_z(insertion_direction)
    base_to_straight_tcp = base_to_straight_executor_tcp_pose(assembly)

    pre_entry_tip_pose = Pose6D(
        position=pre_entry_tip,
        quat=tip_orientation.quat,
    )
    pre_entry_base_pose = pre_entry_tip_pose.compose(base_to_straight_tcp.inverse())
    insertion_base_poses = tuple(
        Pose6D(position=point, quat=tip_orientation.quat).compose(
            base_to_straight_tcp.inverse()
        )
        for point in insertion_waypoints
    )
    path_specs = (
        *spec.intermediate_local_paths,
        EngineNavigationLocalPathSpec(
            name=spec.local_path_name,
            at_fraction=1.0,
            path_type=spec.local_path_type,
            radius_m=spec.local_path_radius_m,
            samples=spec.local_path_samples,
            axial_retraction_m=spec.local_path_axial_retraction_m,
            radius_x_m=spec.local_path_radius_x_m,
            radius_y_m=spec.local_path_radius_y_m,
            length_m=spec.local_path_length_m,
            side_length_m=spec.local_path_side_length_m,
            lissajous_frequency_x=spec.local_path_lissajous_frequency_x,
            lissajous_frequency_y=spec.local_path_lissajous_frequency_y,
            lissajous_phase_deg=spec.local_path_lissajous_phase_deg,
        ),
    )
    local_path_plans = tuple(
        _resolve_local_path_plan(
            path_spec,
            insertion_waypoints,
            insertion_direction,
            tip_orientation,
            transition_samples=spec.local_tracking.transition_samples,
        )
        for path_spec in path_specs
    )
    insertion_indices = [path.insertion_index for path in local_path_plans]
    if len(insertion_indices) != len(set(insertion_indices)):
        raise ValueError(
            "Engine navigation local paths resolve to duplicate insertion "
            "waypoints; reduce insertion_waypoint_spacing_m or move fractions."
        )
    endpoint_path = local_path_plans[-1]
    return EngineNavigationPlan(
        pre_entry_tip_world=pre_entry_tip,
        insertion_direction_world=insertion_direction,
        insertion_tip_waypoints_world=insertion_waypoints,
        pre_entry_base_pose=pre_entry_base_pose,
        insertion_base_poses=insertion_base_poses,
        executor_waypoints_world=endpoint_path.waypoints_world.copy(),
        observer_roi_world=endpoint_path.center_world.copy(),
        local_path_plans=local_path_plans,
    )


def _load_local_tracking(raw_value: object) -> EngineNavigationLocalTrackingSpec:
    if not isinstance(raw_value, dict):
        raise ValueError("engine_navigation.local_tracking must be a mapping.")
    mode = str(raw_value.get("advance_mode", "tolerance"))
    if mode not in LOCAL_TRACKING_ADVANCE_MODES:
        raise ValueError(
            "engine_navigation.local_tracking.advance_mode must be one of "
            f"{LOCAL_TRACKING_ADVANCE_MODES}."
        )
    advance_time_s: float | None = None
    advance_steps: int | None = None
    waypoint_tolerance_m: float | None = None
    rejoin_tolerance_m: float | None = None
    max_steps_per_waypoint: int | None = None
    tolerance_raw = raw_value.get("waypoint_tolerance_m")
    if tolerance_raw is not None:
        waypoint_tolerance_m = float(tolerance_raw)
        if not np.isfinite(waypoint_tolerance_m) or waypoint_tolerance_m < 0.0:
            raise ValueError(
                "engine_navigation.local_tracking."
                "waypoint_tolerance_m must be finite and non-negative."
            )
    rejoin_tolerance_raw = raw_value.get("rejoin_tolerance_m")
    if rejoin_tolerance_raw is not None:
        rejoin_tolerance_m = float(rejoin_tolerance_raw)
        if not np.isfinite(rejoin_tolerance_m) or rejoin_tolerance_m < 0.0:
            raise ValueError(
                "engine_navigation.local_tracking."
                "rejoin_tolerance_m must be finite and non-negative."
            )
    if mode == "time":
        if raw_value.get("advance_time_s") is None:
            raise ValueError(
                "Time-based engine local tracking requires advance_time_s."
            )
        advance_time_s = float(raw_value["advance_time_s"])
        if not np.isfinite(advance_time_s) or advance_time_s <= 0.0:
            raise ValueError(
                "engine_navigation.local_tracking.advance_time_s must be "
                "positive and finite."
            )
    elif mode == "steps":
        if raw_value.get("advance_steps") is None:
            raise ValueError(
                "Step-based engine local tracking requires advance_steps."
            )
        advance_steps = int(raw_value["advance_steps"])
        if advance_steps <= 0:
            raise ValueError(
                "engine_navigation.local_tracking.advance_steps must be positive."
            )
    else:
        max_steps_raw = raw_value.get("max_steps_per_waypoint")
        if max_steps_raw is not None:
            max_steps_per_waypoint = int(max_steps_raw)
            if max_steps_per_waypoint <= 0:
                raise ValueError(
                    "engine_navigation.local_tracking."
                    "max_steps_per_waypoint must be positive."
                )
    transition_samples = int(raw_value.get("transition_samples", 20))
    if transition_samples < 2:
        raise ValueError(
            "engine_navigation.local_tracking.transition_samples must be "
            "at least 2."
        )
    executor_position_gain = float(raw_value.get("executor_position_gain", 3.0))
    if not np.isfinite(executor_position_gain) or executor_position_gain <= 0.0:
        raise ValueError(
            "engine_navigation.local_tracking.executor_position_gain must be "
            "positive and finite."
        )
    max_target_speed_mps = _optional_float(raw_value.get("max_target_speed_mps"))
    if max_target_speed_mps is not None and (
        not np.isfinite(max_target_speed_mps) or max_target_speed_mps <= 0.0
    ):
        raise ValueError(
            "engine_navigation.local_tracking.max_target_speed_mps must be "
            "positive and finite when provided."
        )
    return EngineNavigationLocalTrackingSpec(
        advance_mode=mode,
        advance_time_s=advance_time_s,
        advance_steps=advance_steps,
        waypoint_tolerance_m=waypoint_tolerance_m,
        rejoin_tolerance_m=rejoin_tolerance_m,
        max_steps_per_waypoint=max_steps_per_waypoint,
        transition_samples=transition_samples,
        executor_position_gain=executor_position_gain,
        max_target_speed_mps=max_target_speed_mps,
        enforce_tendon_rate_limits=bool(
            raw_value.get("enforce_tendon_rate_limits", False)
        ),
    )


def _load_observer_control_overrides(
    raw_value: object,
) -> EngineNavigationObserverControlSpec:
    if not isinstance(raw_value, dict):
        raise ValueError("engine_navigation.observer_control_overrides must be a mapping.")
    offset = np.asarray(
        raw_value.get("executor_offset_world_m", (0.0, -0.04, 0.02)),
        dtype=float,
    )
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError(
            "engine_navigation.observer_control_overrides.executor_offset_world_m must "
            "be a finite 3-vector."
        )
    stop_all = raw_value.get(
        "stop_all_on_critical_distance",
        False,
    )
    if not isinstance(stop_all, bool):
        raise ValueError(
            "engine_navigation.observer_control_overrides."
            "stop_all_on_critical_distance must be boolean."
        )
    spec = EngineNavigationObserverControlSpec(
        position_gain=float(raw_value.get("position_gain", 3.0)),
        executor_offset_world_m=offset.copy(),
        roi_blend=float(raw_value.get("roi_blend", 0.25)),
        inter_arm_influence_distance_m=float(
            raw_value.get("inter_arm_influence_distance_m", 0.018)
        ),
        inter_arm_safe_distance_m=float(
            raw_value.get("inter_arm_safe_distance_m", 0.014)
        ),
        inter_arm_critical_distance_m=float(
            raw_value.get(
                "inter_arm_critical_distance_m",
                raw_value.get("inter_arm_hard_stop_distance_m", 0.009),
            )
        ),
        inter_arm_release_margin_m=float(
            raw_value.get("inter_arm_release_margin_m", 0.002)
        ),
        inter_arm_avoidance_gain=float(
            raw_value.get("inter_arm_avoidance_gain", 6.0)
        ),
        inter_arm_max_avoidance_speed_mps=(
            None
            if raw_value.get("inter_arm_max_avoidance_speed_mps") is None
            else float(raw_value["inter_arm_max_avoidance_speed_mps"])
        ),
        centerline_samples_per_segment=int(
            raw_value.get("centerline_samples_per_segment", 8)
        ),
        observer_tracking_weight=float(
            raw_value.get("observer_tracking_weight", 20.0)
        ),
        observer_collision_weight=float(
            raw_value.get("observer_collision_weight", 250.0)
        ),
        stop_all_on_critical_distance=stop_all,
    )
    positive = {
        "position_gain": spec.position_gain,
        "inter_arm_influence_distance_m": spec.inter_arm_influence_distance_m,
        "inter_arm_safe_distance_m": spec.inter_arm_safe_distance_m,
        "inter_arm_critical_distance_m": spec.inter_arm_critical_distance_m,
        "inter_arm_avoidance_gain": spec.inter_arm_avoidance_gain,
        "observer_tracking_weight": spec.observer_tracking_weight,
        "observer_collision_weight": spec.observer_collision_weight,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"engine_navigation.observer_control_overrides.{name} must be "
                "positive and finite."
            )
    if spec.inter_arm_max_avoidance_speed_mps is not None and (
        not np.isfinite(spec.inter_arm_max_avoidance_speed_mps)
        or spec.inter_arm_max_avoidance_speed_mps <= 0.0
    ):
        raise ValueError(
            "engine_navigation.observer_control_overrides."
            "inter_arm_max_avoidance_speed_mps must be positive and finite "
            "when provided."
        )
    if (
        not np.isfinite(spec.inter_arm_release_margin_m)
        or spec.inter_arm_release_margin_m < 0.0
    ):
        raise ValueError(
            "engine_navigation.observer_control_overrides.inter_arm_release_margin_m "
            "must be finite and non-negative."
        )
    if not 0.0 <= spec.roi_blend <= 1.0:
        raise ValueError(
            "engine_navigation.observer_control_overrides.roi_blend must be in [0, 1]."
        )
    if spec.centerline_samples_per_segment <= 0:
        raise ValueError(
            "engine_navigation.observer_control_overrides."
            "centerline_samples_per_segment must be positive."
        )
    if not (
        spec.inter_arm_critical_distance_m
        < spec.inter_arm_safe_distance_m
        < spec.inter_arm_influence_distance_m
    ):
        raise ValueError(
            "Observer inter-arm distances must satisfy "
            "critical < safe < influence."
        )
    return spec


def _load_intermediate_local_paths(
    raw_value: object,
) -> tuple[EngineNavigationLocalPathSpec, ...]:
    if not isinstance(raw_value, list | tuple):
        raise ValueError(
            "task.engine_navigation.intermediate_local_paths must be a list."
        )
    result: list[EngineNavigationLocalPathSpec] = []
    for index, values in enumerate(raw_value):
        if not isinstance(values, dict):
            raise ValueError(
                "Each engine_navigation intermediate local path must be a mapping."
            )
        prefix = f"engine_navigation.intermediate_local_paths[{index}]"
        merged = _merge_shape_mapping(values, prefix)
        spec = EngineNavigationLocalPathSpec(
            name=str(_required(merged, "name")),
            at_fraction=float(_required(merged, "at_fraction")),
            path_type=str(_required(merged, "type")),
            radius_m=float(merged.get("radius_m", 0.01)),
            samples=int(merged.get("samples", 40)),
            axial_retraction_m=float(merged.get("axial_retraction_m", 0.01)),
            radius_x_m=_optional_float(merged.get("radius_x_m")),
            radius_y_m=_optional_float(merged.get("radius_y_m")),
            length_m=_optional_float(merged.get("length_m")),
            side_length_m=_optional_float(merged.get("side_length_m")),
            lissajous_frequency_x=int(merged.get("lissajous_frequency_x", 2)),
            lissajous_frequency_y=int(merged.get("lissajous_frequency_y", 1)),
            lissajous_phase_deg=float(merged.get("lissajous_phase_deg", 90.0)),
        )
        if not spec.name:
            raise ValueError(f"{prefix}.name must be non-empty.")
        if not np.isfinite(spec.at_fraction) or not 0.0 < spec.at_fraction < 1.0:
            raise ValueError(f"{prefix}.at_fraction must be between zero and one.")
        if spec.path_type not in LOCAL_PATH_TYPES:
            raise ValueError(f"{prefix}.type must be one of {LOCAL_PATH_TYPES}.")
        if not np.isfinite(spec.radius_m) or spec.radius_m <= 0.0:
            raise ValueError(f"{prefix}.radius_m must be positive and finite.")
        if spec.samples < 4:
            raise ValueError(f"{prefix}.samples must be at least 4.")
        _validate_local_path_shape(
            prefix,
            path_type=spec.path_type,
            radius_m=spec.radius_m,
            radius_x_m=spec.radius_x_m,
            radius_y_m=spec.radius_y_m,
            length_m=spec.length_m,
            side_length_m=spec.side_length_m,
            lissajous_frequency_x=spec.lissajous_frequency_x,
            lissajous_frequency_y=spec.lissajous_frequency_y,
        )
        if (
            not np.isfinite(spec.axial_retraction_m)
            or spec.axial_retraction_m < 0.0
        ):
            raise ValueError(
                f"{prefix}.axial_retraction_m must be finite and non-negative."
            )
        result.append(spec)
    return tuple(result)


def _resolve_local_path_plan(
    spec: EngineNavigationLocalPathSpec,
    insertion_waypoints: np.ndarray,
    insertion_direction: np.ndarray,
    frame: Pose6D,
    *,
    transition_samples: int,
) -> EngineNavigationLocalPathPlan:
    insertion_index = _fraction_waypoint_index(
        insertion_waypoints,
        spec.at_fraction,
    )
    insertion_target = insertion_waypoints[insertion_index].copy()
    center = insertion_target - spec.axial_retraction_m * insertion_direction
    local_waypoints = _transverse_local_path(
        path_type=spec.path_type,
        center=center,
        frame=frame,
        radius_m=spec.radius_m,
        radius_x_m=spec.radius_x_m,
        radius_y_m=spec.radius_y_m,
        length_m=spec.length_m,
        side_length_m=spec.side_length_m,
        lissajous_frequency_x=spec.lissajous_frequency_x,
        lissajous_frequency_y=spec.lissajous_frequency_y,
        lissajous_phase_deg=spec.lissajous_phase_deg,
        samples=spec.samples,
    )
    return EngineNavigationLocalPathPlan(
        name=spec.name,
        path_type=spec.path_type,
        at_fraction=spec.at_fraction,
        insertion_index=insertion_index,
        insertion_target_world=insertion_target,
        center_world=center.copy(),
        waypoints_world=local_waypoints,
        transition_waypoints_world=_smooth_local_path_transition(
            insertion_target=insertion_target,
            insertion_direction=insertion_direction,
            local_waypoints=local_waypoints,
            samples=transition_samples,
        ),
        is_terminal=bool(np.isclose(spec.at_fraction, 1.0)),
    )


def _fraction_waypoint_index(points: np.ndarray, fraction: float) -> int:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    target_distance = float(fraction) * float(cumulative[-1])
    return int(np.argmin(np.abs(cumulative - target_distance)))


def _smooth_local_path_transition(
    *,
    insertion_target: np.ndarray,
    insertion_direction: np.ndarray,
    local_waypoints: np.ndarray,
    samples: int,
) -> np.ndarray:
    """Join an insertion point to a local path with a cubic Hermite curve."""

    start = np.asarray(insertion_target, dtype=float)
    end = np.asarray(local_waypoints[0], dtype=float)
    chord_length = float(np.linalg.norm(end - start))
    start_tangent = -_unit(
        insertion_direction,
        "insertion direction",
    ) * chord_length
    end_tangent = _unit(
        local_waypoints[1] - local_waypoints[0],
        "local path start tangent",
    ) * chord_length
    parameter = np.linspace(0.0, 1.0, samples)
    t = parameter[:, None]
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return (
        h00 * start[None, :]
        + h10 * start_tangent[None, :]
        + h01 * end[None, :]
        + h11 * end_tangent[None, :]
    )


def _named_region(scene: EngineSceneConfig, name: str) -> EngineRegionConfig:
    try:
        return scene.regions[name]
    except KeyError as exc:
        raise ValueError(f"Unknown engine navigation region {name!r}.") from exc


def _region_point_world(
    region: EngineRegionConfig,
    scene: EngineSceneConfig,
) -> np.ndarray:
    point = region.center_m if region.center_m is not None else region.position_m
    if point is None:
        raise ValueError(
            f"Engine navigation region {region.name!r} has no center or position."
        )
    return _point_world(point, region.frame, scene)


def _named_path_world(scene: EngineSceneConfig, name: str) -> np.ndarray:
    matches = [
        path
        for path in scene.exploration_paths
        if path.enabled and path.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown enabled engine navigation path {name!r}.")
    path = matches[0]
    if path.frame == "world":
        return path.points_m.copy()
    frame = _engine_frame_pose(scene)
    return frame.transform_points(path.points_m)


def _point_world(
    point: np.ndarray,
    frame: str,
    scene: EngineSceneConfig,
) -> np.ndarray:
    values = np.asarray(point, dtype=float)
    if frame == "world":
        return values.copy()
    return _engine_frame_pose(scene).transform_point(values)


def _engine_frame_pose(scene: EngineSceneConfig) -> Pose6D:
    return Pose6D(
        position=effective_engine_frame_position(scene),
        quat=scene.engine.pose.quat_wxyz,
    )


def _resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    result = [values[0].copy()]
    for start, end in zip(values[:-1], values[1:], strict=True):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-12:
            raise ValueError("Engine navigation path contains duplicate adjacent points.")
        intervals = max(1, int(np.ceil(distance / spacing_m)))
        result.extend(
            start + (index / intervals) * delta
            for index in range(1, intervals + 1)
        )
    return np.asarray(result, dtype=float)


def _orientation_along_z(direction: np.ndarray) -> Pose6D:
    z_axis = _unit(direction, "insertion direction")
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(reference @ z_axis)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    x_axis = _unit(reference - float(reference @ z_axis) * z_axis, "frame x-axis")
    y_axis = _unit(np.cross(z_axis, x_axis), "frame y-axis")
    x_axis = _unit(np.cross(y_axis, z_axis), "frame x-axis")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    return Pose6D.from_matrix(transform)


def _transverse_local_path(
    *,
    path_type: str,
    center: np.ndarray,
    frame: Pose6D,
    radius_m: float,
    radius_x_m: float | None,
    radius_y_m: float | None,
    length_m: float | None,
    side_length_m: float | None,
    lissajous_frequency_x: int,
    lissajous_frequency_y: int,
    lissajous_phase_deg: float,
    samples: int,
) -> np.ndarray:
    normalized_type = _normalize_local_path_type(path_type)
    if normalized_type == "transverse_square":
        planar = _closed_square(
            _shape_scale(side_length_m, 2.0 * radius_m),
            samples,
        )
    elif normalized_type == "transverse_line":
        planar = _line(_shape_scale(length_m, 2.0 * radius_m), samples)
    else:
        angle = np.linspace(
            0.0,
            2.0 * np.pi,
            samples,
            endpoint=True,
        )
        if normalized_type == "transverse_circle":
            planar = radius_m * np.column_stack(
                (np.cos(angle), np.sin(angle))
            )
        elif normalized_type == "transverse_ellipse":
            planar = np.column_stack(
                (
                    _shape_scale(radius_x_m, radius_m) * np.cos(angle),
                    _shape_scale(radius_y_m, radius_m) * np.sin(angle),
                )
            )
        elif normalized_type == "transverse_figure_eight":
            planar = np.column_stack(
                (
                    _shape_scale(radius_x_m, radius_m) * np.sin(angle),
                    _shape_scale(radius_y_m, 0.5 * radius_m) * np.sin(2.0 * angle),
                )
            )
        elif normalized_type == "transverse_lissajous":
            planar = np.column_stack(
                (
                    _shape_scale(radius_x_m, radius_m)
                    * np.sin(
                        int(lissajous_frequency_x) * angle
                        + np.deg2rad(lissajous_phase_deg)
                    ),
                    _shape_scale(radius_y_m, radius_m)
                    * np.sin(int(lissajous_frequency_y) * angle),
                )
            )
        else:
            raise ValueError(f"Unsupported local path type {path_type!r}.")
    rotation = frame.as_matrix()[:3, :3]
    return (
        np.asarray(center, dtype=float)[None, :]
        + planar[:, :1] * rotation[:, 0][None, :]
        + planar[:, 1:] * rotation[:, 1][None, :]
    )


def _line(length_m: float, samples: int) -> np.ndarray:
    return np.column_stack(
        (np.linspace(-0.5 * length_m, 0.5 * length_m, samples), np.zeros(samples))
    )


def _closed_square(side_length_m: float, samples: int) -> np.ndarray:
    half_side = 0.5 * side_length_m
    perimeter = 8.0 * half_side
    distances = np.linspace(0.0, perimeter, samples, endpoint=True)
    planar = np.empty((samples, 2), dtype=float)
    for index, distance in enumerate(distances):
        if index == samples - 1:
            planar[index] = (-half_side, -half_side)
            continue
        side = int(distance // (2.0 * half_side))
        offset = distance - side * 2.0 * half_side
        if side == 0:
            planar[index] = (-half_side + offset, -half_side)
        elif side == 1:
            planar[index] = (half_side, -half_side + offset)
        elif side == 2:
            planar[index] = (half_side - offset, half_side)
        else:
            planar[index] = (-half_side, half_side - offset)
    return planar


def _merge_shape_mapping(values: dict[str, object], prefix: str) -> dict[str, object]:
    shape = values.get("shape", {})
    if shape is None:
        shape = {}
    if not isinstance(shape, dict):
        raise ValueError(f"{prefix}.shape must be a mapping.")
    return {**values, **shape}


def _normalize_local_path_type(path_type: str) -> str:
    aliases = {
        "circle": "transverse_circle",
        "square": "transverse_square",
        "figure-eight": "transverse_figure_eight",
        "figure_eight": "transverse_figure_eight",
        "ellipse": "transverse_ellipse",
        "line": "transverse_line",
        "lissajous": "transverse_lissajous",
    }
    return aliases.get(path_type, path_type)


def _validate_local_path_shape(
    prefix: str,
    *,
    path_type: str,
    radius_m: float,
    radius_x_m: float | None,
    radius_y_m: float | None,
    length_m: float | None,
    side_length_m: float | None,
    lissajous_frequency_x: int,
    lissajous_frequency_y: int,
) -> None:
    normalized_type = _normalize_local_path_type(path_type)
    for name, value in (
        ("radius_m", radius_m),
        ("radius_x_m", radius_x_m),
        ("radius_y_m", radius_y_m),
        ("length_m", length_m),
        ("side_length_m", side_length_m),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"{prefix}.{name} must be positive and finite.")
    if normalized_type != "transverse_line" and (
        not np.isfinite(radius_m) or radius_m <= 0.0
    ):
        raise ValueError(f"{prefix}.radius_m must be positive and finite.")
    if normalized_type == "transverse_line" and length_m is None and (
        not np.isfinite(radius_m) or radius_m <= 0.0
    ):
        raise ValueError(f"{prefix}.length_m or radius_m must be positive and finite.")
    if normalized_type == "transverse_lissajous" and (
        lissajous_frequency_x <= 0 or lissajous_frequency_y <= 0
    ):
        raise ValueError(f"{prefix}.lissajous frequencies must be positive.")


def _shape_scale(value: float | None, default: float) -> float:
    return float(default if value is None else value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _unit(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm


def _required(values: dict[str, object], name: str) -> object:
    if name not in values:
        raise ValueError(f"Missing required engine_navigation field {name!r}.")
    return values[name]
