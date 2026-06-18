"""YAML-backed configuration for trajectory tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    bool_field as _bool,
    bool_value as _bool_value,
    choice as _choice,
    choice_value as _choice_value,
    float_value as _float_value,
    motor_vector as _motor_vector,
    nonnegative_float as _nonnegative_float,
    nonnegative_float_value as _nonnegative_float_value,
    optional_section as _optional_section,
    position_vector as _position_vector_value,
    positive_float as _positive_float,
    positive_float_value as _positive_float_value,
    positive_int as _positive_int,
    positive_int_value as _positive_int_value,
    required as _required,
    resolve_path as _resolve_config_path,
    section as _section,
)
from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.kinematics.differential import tip_position_from_q
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.tasks.dmp_trajectory import DiscreteDMP, load_demonstration


TRAJECTORY_TYPES = (
    "circle",
    "figure-eight",
    "ellipse",
    "line",
    "square",
    "lissajous",
    "helix",
    "dmp",
)
CENTER_MODES = ("straight_tip_xy", "straight_tip", "explicit")
Z_MODES = ("straight_tip_minus_radius", "center", "explicit")
PLANE_MODES = ("xy", "xz", "yz")
VISUALIZATION_MODES = ("static", "animation")
MUJOCO_TARGET_ADVANCE_MODES = ("time", "tolerance")
MUJOCO_FEEDBACK_MODES = ("pcc_command", "mujoco_actual")
_MISSING = object()
_UNSET = object()


@dataclass(frozen=True)
class TrackingSimulationConfig:
    """Simulation timing, limits, and initial state."""

    dt: float
    max_steps: int
    stop_on_completion: bool
    position_limit_rad: float
    initial_motor_position_rad: np.ndarray


@dataclass(frozen=True)
class TrackingControllerConfig:
    """Differential-IK controller parameters."""

    damping: float
    position_gain: float
    max_motor_velocity_rad_s: float
    position_tolerance_m: float


@dataclass(frozen=True)
class TrackingTrajectoryConfig:
    """Analytic target trajectory parameters."""

    type: str
    samples: int
    radius_m: float
    center_mode: str
    z_mode: str
    plane: str
    yaw_deg: float
    center_xyz_m: np.ndarray | None
    offset_xyz_m: np.ndarray
    z_value_m: float | None
    radius_x_m: float | None
    radius_y_m: float | None
    length_m: float | None
    side_length_m: float | None
    turns: float | None
    pitch_m: float | None
    lissajous_frequency_x: int
    lissajous_frequency_y: int
    lissajous_phase_deg: float
    dmp_demo_path: Path | None
    dmp_start_xyz_m: np.ndarray | None
    dmp_goal_xyz_m: np.ndarray | None
    dmp_tau: float
    dmp_basis_count: int


@dataclass(frozen=True)
class TrackingVisualizationConfig:
    """Static/animation visualization parameters."""

    mode: str
    show: bool
    show_summary_after_animation: bool
    animation_interval_ms: int
    animation_stride: int
    animation_samples_per_segment: int


@dataclass(frozen=True)
class TrackingMujocoConfig:
    """MuJoCo trajectory tracking behavior not tied to physics parameters."""

    target_advance_mode: str
    feedback_mode: str
    show_live_tendon_panel: bool
    live_tendon_panel_stride: int
    hold_viewer_open_after_run: bool
    show_summary: bool


@dataclass(frozen=True)
class TrackingConfig:
    """Complete YAML-backed PCC trajectory-tracking configuration."""

    robot_config_path: Path
    simulation: TrackingSimulationConfig
    controller: TrackingControllerConfig
    trajectory: TrackingTrajectoryConfig
    visualization: TrackingVisualizationConfig


@dataclass(frozen=True)
class MujocoTrackingConfig:
    """YAML-backed MuJoCo trajectory-tracking configuration."""

    robot_config_path: Path
    simulation: TrackingSimulationConfig
    controller: TrackingControllerConfig
    trajectory: TrackingTrajectoryConfig
    mujoco: TrackingMujocoConfig
    visualization: TrackingVisualizationConfig


def load_tracking_config(path: str | Path) -> TrackingConfig:
    """Load and validate the PCC trajectory-tracking YAML config."""
    config_path = Path(path)
    raw = load_yaml(config_path)

    robot = _section(raw, "robot")
    simulation = _section(raw, "simulation")
    controller = _section(raw, "controller")
    trajectory = _section(raw, "trajectory")
    visualization = _section(raw, "visualization")
    animation = _section(visualization, "animation")

    robot_config_path = _resolve_config_path(config_path, _required(robot, "config_path"))
    motor_count = len(load_motor_params_from_yaml(robot_config_path))
    simulation_config = TrackingSimulationConfig(
        dt=_positive_float(simulation, "dt"),
        max_steps=_positive_int(simulation, "max_steps"),
        stop_on_completion=_bool(simulation, "stop_on_completion"),
        position_limit_rad=_positive_float(simulation, "position_limit_rad"),
        initial_motor_position_rad=_motor_vector(
            _required(simulation, "initial_motor_position_rad"),
            "initial_motor_position_rad",
            expected_size=motor_count,
        ),
    )
    controller_type = str(_required(controller, "type"))
    if controller_type != "differential_ik":
        raise ValueError(f"controller.type must be 'differential_ik', got {controller_type!r}.")
    controller_config = TrackingControllerConfig(
        damping=_nonnegative_float(controller, "damping"),
        position_gain=_nonnegative_float(controller, "position_gain"),
        max_motor_velocity_rad_s=_positive_float(controller, "max_motor_velocity_rad_s"),
        position_tolerance_m=_nonnegative_float(controller, "position_tolerance_m"),
    )

    trajectory_type = _choice(trajectory, "type", TRAJECTORY_TYPES)
    placement = _optional_section(trajectory, "placement")
    shape = _optional_section(trajectory, "shape")
    trajectory_config = TrackingTrajectoryConfig(
        type=trajectory_type,
        samples=_positive_int(trajectory, "samples"),
        radius_m=_nonnegative_float_value(
            _field_from_nested_or_top_level(trajectory, shape, "radius_m", default=0.0),
            "trajectory.radius_m",
        ),
        center_mode=_choice_value(
            _field_from_nested_or_top_level(
                trajectory,
                placement,
                "center_mode",
                default="straight_tip_xy",
            ),
            "trajectory.center_mode",
            CENTER_MODES,
        ),
        z_mode=_choice_value(
            _field_from_nested_or_top_level(
                trajectory,
                placement,
                "z_mode",
                default="straight_tip_minus_radius",
            ),
            "trajectory.z_mode",
            Z_MODES,
        ),
        plane=_choice_value(
            _field_from_nested_or_top_level(trajectory, placement, "plane", default="xy"),
            "trajectory.plane",
            PLANE_MODES,
        ),
        yaw_deg=_float_value(
            _field_from_nested_or_top_level(trajectory, placement, "yaw_deg", default=0.0),
            "trajectory.yaw_deg",
        ),
        center_xyz_m=_optional_position_vector_value(
            _field_from_nested_or_top_level(trajectory, placement, "center_xyz_m", default=_MISSING),
            "trajectory.center_xyz_m",
        ),
        offset_xyz_m=_position_vector_value(
            _field_from_nested_or_top_level(
                trajectory,
                placement,
                "offset_xyz_m",
                default=(0.0, 0.0, 0.0),
            ),
            "trajectory.offset_xyz_m",
        ),
        z_value_m=_optional_float_value(
            _field_from_nested_or_top_level(trajectory, placement, "z_value_m", default=_MISSING),
            "trajectory.z_value_m",
        ),
        radius_x_m=_optional_nonnegative_float_value(
            _field_from_nested_or_top_level(trajectory, shape, "radius_x_m", default=_MISSING),
            "trajectory.radius_x_m",
        ),
        radius_y_m=_optional_nonnegative_float_value(
            _field_from_nested_or_top_level(trajectory, shape, "radius_y_m", default=_MISSING),
            "trajectory.radius_y_m",
        ),
        length_m=_optional_nonnegative_float_value(
            _field_from_nested_or_top_level(trajectory, shape, "length_m", default=_MISSING),
            "trajectory.length_m",
        ),
        side_length_m=_optional_nonnegative_float_value(
            _field_from_nested_or_top_level(trajectory, shape, "side_length_m", default=_MISSING),
            "trajectory.side_length_m",
        ),
        turns=_optional_nonnegative_float_value(
            _field_from_nested_or_top_level(trajectory, shape, "turns", default=_MISSING),
            "trajectory.turns",
        ),
        pitch_m=_optional_nonnegative_float_value(
            _field_from_nested_or_top_level(trajectory, shape, "pitch_m", default=_MISSING),
            "trajectory.pitch_m",
        ),
        lissajous_frequency_x=_positive_int_value(
            _field_from_nested_or_top_level(
                trajectory,
                shape,
                "lissajous_frequency_x",
                default=3,
            ),
            "trajectory.lissajous_frequency_x",
        ),
        lissajous_frequency_y=_positive_int_value(
            _field_from_nested_or_top_level(
                trajectory,
                shape,
                "lissajous_frequency_y",
                default=2,
            ),
            "trajectory.lissajous_frequency_y",
        ),
        lissajous_phase_deg=_float_value(
            _field_from_nested_or_top_level(
                trajectory,
                shape,
                "lissajous_phase_deg",
                default=0.0,
            ),
            "trajectory.lissajous_phase_deg",
        ),
        dmp_demo_path=_optional_resolved_path(
            config_path,
            _field_from_nested_or_top_level(trajectory, shape, "demo_path", default=_MISSING),
        ),
        dmp_start_xyz_m=_optional_position_vector_value(
            _field_from_nested_or_top_level(trajectory, shape, "start_xyz_m", default=_MISSING),
            "trajectory.start_xyz_m",
        ),
        dmp_goal_xyz_m=_optional_position_vector_value(
            _field_from_nested_or_top_level(trajectory, shape, "goal_xyz_m", default=_MISSING),
            "trajectory.goal_xyz_m",
        ),
        dmp_tau=_positive_float_value(
            _field_from_nested_or_top_level(trajectory, shape, "tau", default=1.0),
            "trajectory.tau",
        ),
        dmp_basis_count=_positive_int_value(
            _field_from_nested_or_top_level(trajectory, shape, "basis_count", default=24),
            "trajectory.basis_count",
        ),
    )
    visualization_config = TrackingVisualizationConfig(
        mode=_choice(visualization, "mode", VISUALIZATION_MODES),
        show=_bool(visualization, "show"),
        show_summary_after_animation=_bool(visualization, "show_summary_after_animation"),
        animation_interval_ms=_positive_int(animation, "interval_ms"),
        animation_stride=_positive_int(animation, "stride"),
        animation_samples_per_segment=_positive_int(animation, "samples_per_segment"),
    )
    return TrackingConfig(
        robot_config_path=robot_config_path,
        simulation=simulation_config,
        controller=controller_config,
        trajectory=trajectory_config,
        visualization=visualization_config,
    )


def load_mujoco_tracking_config(path: str | Path) -> MujocoTrackingConfig:
    """Load and validate a MuJoCo trajectory-tracking YAML config."""
    config_path = Path(path)
    raw = load_yaml(config_path)
    base_config = load_tracking_config(config_path)
    mujoco = _optional_section(raw, "mujoco")
    mujoco_config = TrackingMujocoConfig(
        target_advance_mode=_choice_value(
            mujoco.get("target_advance_mode", "time"),
            "mujoco.target_advance_mode",
            MUJOCO_TARGET_ADVANCE_MODES,
        ),
        feedback_mode=_choice_value(
            mujoco.get("feedback_mode", "mujoco_actual"),
            "mujoco.feedback_mode",
            MUJOCO_FEEDBACK_MODES,
        ),
        show_live_tendon_panel=_bool_value(
            mujoco.get("show_live_tendon_panel", True),
            "mujoco.show_live_tendon_panel",
        ),
        live_tendon_panel_stride=_positive_int_value(
            mujoco.get("live_tendon_panel_stride", 1),
            "mujoco.live_tendon_panel_stride",
        ),
        hold_viewer_open_after_run=_bool_value(
            mujoco.get("hold_viewer_open_after_run", False),
            "mujoco.hold_viewer_open_after_run",
        ),
        show_summary=_bool_value(
            mujoco.get("show_summary", True),
            "mujoco.show_summary",
        ),
    )
    return MujocoTrackingConfig(
        robot_config_path=base_config.robot_config_path,
        simulation=base_config.simulation,
        controller=base_config.controller,
        trajectory=base_config.trajectory,
        mujoco=mujoco_config,
        visualization=base_config.visualization,
    )


def build_target_positions(
    config: TrackingConfig | MujocoTrackingConfig,
    params: ThreeSegmentRobotParams,
) -> np.ndarray:
    """Generate target tip positions from a loaded tracking config."""
    if config.trajectory.type == "dmp":
        return _dmp_trajectory(config.trajectory)

    straight_tip = tip_position_from_q(np.zeros(params.q_size, dtype=float), params)
    center = _resolve_trajectory_center(config.trajectory, straight_tip)
    in_plane_u, in_plane_v, plane_normal = _plane_basis(
        config.trajectory.plane,
        config.trajectory.yaw_deg,
    )

    if config.trajectory.type == "circle":
        local_points = _circle_trajectory(
            _required_positive(
                config.trajectory.radius_m,
                "trajectory.radius_m must be positive for circle trajectories.",
            ),
            config.trajectory.samples,
        )
        return _lift_planar_trajectory(local_points, center, in_plane_u, in_plane_v)
    if config.trajectory.type == "figure-eight":
        local_points = _figure_eight_trajectory(
            _radius_x(config.trajectory, default_scale=1.0),
            _radius_y(config.trajectory, default_scale=0.5),
            config.trajectory.samples,
        )
        return _lift_planar_trajectory(local_points, center, in_plane_u, in_plane_v)
    if config.trajectory.type == "ellipse":
        local_points = _ellipse_trajectory(
            _radius_x(config.trajectory, default_scale=1.0),
            _radius_y(config.trajectory, default_scale=1.0),
            config.trajectory.samples,
        )
        return _lift_planar_trajectory(local_points, center, in_plane_u, in_plane_v)
    if config.trajectory.type == "line":
        local_points = _line_trajectory(
            _line_length(config.trajectory),
            config.trajectory.samples,
        )
        return _lift_planar_trajectory(local_points, center, in_plane_u, in_plane_v)
    if config.trajectory.type == "square":
        local_points = _square_trajectory(
            _square_side_length(config.trajectory),
            config.trajectory.samples,
        )
        return _lift_planar_trajectory(local_points, center, in_plane_u, in_plane_v)
    if config.trajectory.type == "lissajous":
        local_points = _lissajous_trajectory(
            _radius_x(config.trajectory, default_scale=1.0),
            _radius_y(config.trajectory, default_scale=1.0),
            config.trajectory.lissajous_frequency_x,
            config.trajectory.lissajous_frequency_y,
            np.deg2rad(config.trajectory.lissajous_phase_deg),
            config.trajectory.samples,
        )
        return _lift_planar_trajectory(local_points, center, in_plane_u, in_plane_v)
    if config.trajectory.type == "helix":
        return _helix_trajectory(
            center=center,
            in_plane_u=in_plane_u,
            in_plane_v=in_plane_v,
            plane_normal=plane_normal,
            radius=_radius_x(config.trajectory, default_scale=1.0),
            pitch_m=_helix_pitch(config.trajectory),
            turns=_helix_turns(config.trajectory),
            samples=config.trajectory.samples,
        )
    raise ValueError(f"Unsupported trajectory type {config.trajectory.type!r}.")


def _dmp_trajectory(trajectory: TrackingTrajectoryConfig) -> np.ndarray:
    if trajectory.dmp_demo_path is None:
        raise ValueError("trajectory.demo_path is required when trajectory.type is 'dmp'.")
    time, demo = load_demonstration(trajectory.dmp_demo_path)
    start = demo[0] if trajectory.dmp_start_xyz_m is None else trajectory.dmp_start_xyz_m
    goal = demo[-1] if trajectory.dmp_goal_xyz_m is None else trajectory.dmp_goal_xyz_m
    dmp = DiscreteDMP(
        basis_count=trajectory.dmp_basis_count,
        samples=trajectory.samples,
    ).imitate(time, demo)
    return dmp.rollout(start, goal, tau=trajectory.dmp_tau).position


def _resolve_trajectory_center(
    trajectory: TrackingTrajectoryConfig,
    straight_tip: np.ndarray,
) -> np.ndarray:
    if trajectory.center_mode == "straight_tip_xy":
        center = straight_tip.copy()
    elif trajectory.center_mode == "straight_tip":
        center = straight_tip.copy()
    elif trajectory.center_mode == "explicit":
        if trajectory.center_xyz_m is None:
            raise ValueError(
                "trajectory.center_xyz_m must be provided when trajectory.center_mode is 'explicit'."
            )
        center = trajectory.center_xyz_m.copy()
    else:
        raise ValueError(f"Unsupported center_mode {trajectory.center_mode!r}.")

    if trajectory.center_mode == "straight_tip_xy":
        center[0] = straight_tip[0]
        center[1] = straight_tip[1]

    center[2] = _resolve_trajectory_z(trajectory, straight_tip, center)
    center = center + trajectory.offset_xyz_m
    return center


def _resolve_trajectory_z(
    trajectory: TrackingTrajectoryConfig,
    straight_tip: np.ndarray,
    center: np.ndarray,
) -> float:
    if trajectory.z_mode == "straight_tip_minus_radius":
        return float(straight_tip[2] - _trajectory_reference_scale(trajectory))
    if trajectory.z_mode == "center":
        return float(center[2])
    if trajectory.z_mode == "explicit":
        if trajectory.z_value_m is None:
            raise ValueError(
                "trajectory.z_value_m must be provided when trajectory.z_mode is 'explicit'."
            )
        return float(trajectory.z_value_m)
    raise ValueError(f"Unsupported z_mode {trajectory.z_mode!r}.")


def _plane_basis(plane: str, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if plane == "xy":
        base_u = np.array([1.0, 0.0, 0.0], dtype=float)
        base_v = np.array([0.0, 1.0, 0.0], dtype=float)
    elif plane == "xz":
        base_u = np.array([1.0, 0.0, 0.0], dtype=float)
        base_v = np.array([0.0, 0.0, 1.0], dtype=float)
    elif plane == "yz":
        base_u = np.array([0.0, 1.0, 0.0], dtype=float)
        base_v = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        raise ValueError(f"Unsupported plane {plane!r}.")

    yaw = np.deg2rad(yaw_deg)
    in_plane_u = np.cos(yaw) * base_u + np.sin(yaw) * base_v
    in_plane_v = -np.sin(yaw) * base_u + np.cos(yaw) * base_v
    plane_normal = np.cross(in_plane_u, in_plane_v)
    return in_plane_u, in_plane_v, plane_normal


def _lift_planar_trajectory(
    local_points: np.ndarray,
    center: np.ndarray,
    in_plane_u: np.ndarray,
    in_plane_v: np.ndarray,
) -> np.ndarray:
    local_points = np.asarray(local_points, dtype=float)
    if local_points.ndim != 2 or local_points.shape[1] != 2:
        raise ValueError(f"Expected planar local_points with shape (N, 2), got {local_points.shape}.")
    return center[None, :] + local_points[:, [0]] * in_plane_u[None, :] + local_points[:, [1]] * in_plane_v[None, :]


def _figure_eight_trajectory(
    radius_x: float,
    radius_y: float,
    samples: int,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, _dense_sample_count(samples), endpoint=False)
    local_points = np.zeros((angles.size, 2), dtype=float)
    local_points[:, 0] = radius_x * np.sin(angles)
    local_points[:, 1] = radius_y * np.sin(2.0 * angles)
    return _resample_curve(local_points, samples, closed=True)


def _circle_trajectory(radius: float, samples: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, _dense_sample_count(samples), endpoint=False)
    local_points = np.zeros((angles.size, 2), dtype=float)
    local_points[:, 0] = radius * np.cos(angles)
    local_points[:, 1] = radius * np.sin(angles)
    return _resample_curve(local_points, samples, closed=True)


def _ellipse_trajectory(radius_x: float, radius_y: float, samples: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, _dense_sample_count(samples), endpoint=False)
    local_points = np.zeros((angles.size, 2), dtype=float)
    local_points[:, 0] = radius_x * np.cos(angles)
    local_points[:, 1] = radius_y * np.sin(angles)
    return _resample_curve(local_points, samples, closed=True)


def _line_trajectory(length_m: float, samples: int) -> np.ndarray:
    dense_count = _dense_sample_count(samples)
    local_points = np.zeros((dense_count, 2), dtype=float)
    local_points[:, 0] = np.linspace(-0.5 * length_m, 0.5 * length_m, dense_count)
    return _resample_curve(local_points, samples, closed=False)


def _square_trajectory(side_length_m: float, samples: int) -> np.ndarray:
    half = 0.5 * side_length_m
    corners = np.array(
        [
            [half, -half],
            [half, half],
            [-half, half],
            [-half, -half],
        ],
        dtype=float,
    )
    return _resample_curve(corners, samples, closed=True)


def _lissajous_trajectory(
    radius_x: float,
    radius_y: float,
    frequency_x: int,
    frequency_y: int,
    phase_rad: float,
    samples: int,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, _dense_sample_count(samples), endpoint=False)
    local_points = np.zeros((angles.size, 2), dtype=float)
    local_points[:, 0] = radius_x * np.sin(frequency_x * angles + phase_rad)
    local_points[:, 1] = radius_y * np.sin(frequency_y * angles)
    return _resample_curve(local_points, samples, closed=True)


def _helix_trajectory(
    *,
    center: np.ndarray,
    in_plane_u: np.ndarray,
    in_plane_v: np.ndarray,
    plane_normal: np.ndarray,
    radius: float,
    pitch_m: float,
    turns: float,
    samples: int,
) -> np.ndarray:
    dense_count = _dense_sample_count(samples)
    angles = np.linspace(0.0, 2.0 * np.pi * turns, dense_count)
    axial = pitch_m * (angles / (2.0 * np.pi) - 0.5 * turns)
    points = np.zeros((dense_count, 3), dtype=float)
    points[:, :] = center[None, :]
    points += radius * np.cos(angles)[:, None] * in_plane_u[None, :]
    points += radius * np.sin(angles)[:, None] * in_plane_v[None, :]
    points += axial[:, None] * plane_normal[None, :]
    return _resample_curve(points, samples, closed=False)


def _resample_curve(points: np.ndarray, samples: int, *, closed: bool) -> np.ndarray:
    curve = np.asarray(points, dtype=float)
    if curve.ndim != 2 or curve.shape[0] < 2:
        raise ValueError(f"Expected points with shape (N, D) and N >= 2, got {curve.shape}.")

    if closed:
        curve = np.vstack([curve, curve[0]])

    deltas = np.diff(curve, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        raise ValueError("Trajectory length must be positive.")

    if closed:
        target_distances = np.linspace(0.0, total_length, samples, endpoint=False)
    else:
        target_distances = np.linspace(0.0, total_length, samples)

    resampled = np.zeros((samples, curve.shape[1]), dtype=float)
    for dim in range(curve.shape[1]):
        resampled[:, dim] = np.interp(target_distances, cumulative, curve[:, dim])
    return resampled


def _dense_sample_count(samples: int) -> int:
    return max(8 * samples, 256)


def _trajectory_reference_scale(trajectory: TrackingTrajectoryConfig) -> float:
    candidates = [
        trajectory.radius_m,
        trajectory.radius_x_m,
        trajectory.radius_y_m,
        None if trajectory.length_m is None else 0.5 * trajectory.length_m,
        None if trajectory.side_length_m is None else 0.5 * trajectory.side_length_m,
    ]
    positive_candidates = [float(value) for value in candidates if value is not None and value > 0.0]
    if not positive_candidates:
        raise ValueError(
            "A positive trajectory scale is required to resolve trajectory.z_mode = 'straight_tip_minus_radius'."
        )
    return max(positive_candidates)


def _radius_x(trajectory: TrackingTrajectoryConfig, *, default_scale: float) -> float:
    if trajectory.radius_x_m is not None and trajectory.radius_x_m > 0.0:
        return float(trajectory.radius_x_m)
    return _required_positive(
        default_scale * trajectory.radius_m,
        "trajectory.radius_m must be positive or trajectory.radius_x_m must be provided.",
    )


def _radius_y(trajectory: TrackingTrajectoryConfig, *, default_scale: float) -> float:
    if trajectory.radius_y_m is not None and trajectory.radius_y_m > 0.0:
        return float(trajectory.radius_y_m)
    return _required_positive(
        default_scale * trajectory.radius_m,
        "trajectory.radius_m must be positive or trajectory.radius_y_m must be provided.",
    )


def _line_length(trajectory: TrackingTrajectoryConfig) -> float:
    if trajectory.length_m is not None and trajectory.length_m > 0.0:
        return float(trajectory.length_m)
    return _required_positive(
        2.0 * trajectory.radius_m,
        "trajectory.length_m must be positive or trajectory.radius_m must be positive for line trajectories.",
    )


def _square_side_length(trajectory: TrackingTrajectoryConfig) -> float:
    if trajectory.side_length_m is not None and trajectory.side_length_m > 0.0:
        return float(trajectory.side_length_m)
    return _required_positive(
        2.0 * trajectory.radius_m,
        "trajectory.side_length_m must be positive or trajectory.radius_m must be positive for square trajectories.",
    )


def _helix_turns(trajectory: TrackingTrajectoryConfig) -> float:
    turns = 1.0 if trajectory.turns is None else float(trajectory.turns)
    return _required_positive(turns, "trajectory.turns must be positive for helix trajectories.")


def _helix_pitch(trajectory: TrackingTrajectoryConfig) -> float:
    if trajectory.pitch_m is not None and trajectory.pitch_m > 0.0:
        return float(trajectory.pitch_m)
    return _required_positive(
        trajectory.radius_m,
        "trajectory.pitch_m must be positive or trajectory.radius_m must be positive for helix trajectories.",
    )


def _required_positive(value: float, message: str) -> float:
    if value <= 0.0:
        raise ValueError(message)
    return float(value)


def _field_from_nested_or_top_level(
    top_level: dict[str, Any],
    nested: dict[str, Any],
    name: str,
    *,
    default: object = _UNSET,
) -> object:
    if name in nested:
        return nested[name]
    if name in top_level:
        return top_level[name]
    if default is not _UNSET:
        return default
    raise ValueError(f"Missing required config field {name!r}.")


def _optional_nonnegative_float_value(raw_value: object, name: str) -> float | None:
    if raw_value is _MISSING:
        return None
    return _nonnegative_float_value(raw_value, name)


def _optional_float_value(raw_value: object, name: str) -> float | None:
    if raw_value is _MISSING:
        return None
    return float(raw_value)


def _optional_position_vector_value(raw_value: object, name: str) -> np.ndarray | None:
    if raw_value is _MISSING:
        return None
    return _position_vector_value(raw_value, name)


def _optional_resolved_path(config_path: Path, raw_value: object) -> Path | None:
    if raw_value is _MISSING:
        return None
    return _resolve_config_path(config_path, raw_value)


