"""Scenario-native task-space trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
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


@dataclass(frozen=True)
class TrackingWaypointPlan:
    """Tracking waypoints plus provenance for optional approach samples."""

    waypoints_world: np.ndarray
    approach_mask: np.ndarray
    source_waypoint_index: np.ndarray


@dataclass(frozen=True)
class TrajectorySpec:
    """Configurable waypoint generator for scenario tracking tasks."""

    type: str
    samples: int
    radius_m: float = 0.025
    center_mode: str = "straight_tip_xy"
    z_mode: str = "straight_tip_minus_radius"
    plane: str = "xy"
    yaw_deg: float = 0.0
    offset_xyz_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    center_xyz_m: np.ndarray | None = None
    z_value_m: float | None = None
    radius_x_m: float | None = None
    radius_y_m: float | None = None
    length_m: float | None = None
    side_length_m: float | None = None
    lissajous_frequency_x: int = 2
    lissajous_frequency_y: int = 1
    lissajous_phase_deg: float = 90.0
    pitch_m: float | None = None
    turns: float | None = None
    dmp_demo_path: Path | None = None
    dmp_basis_count: int = 20
    dmp_tau: float = 1.0
    dmp_start_xyz_m: np.ndarray | None = None
    dmp_goal_xyz_m: np.ndarray | None = None

    @classmethod
    def from_mapping(
        cls,
        values: dict[str, Any],
        *,
        base_path: Path | None = None,
    ) -> "TrajectorySpec":
        placement = _mapping(values.get("placement", {}), "trajectory.placement")
        shape = _mapping(values.get("shape", {}), "trajectory.shape")
        dmp = _mapping(values.get("dmp", {}), "trajectory.dmp")
        merged = {**values, **placement, **shape, **dmp}
        trajectory_type = str(_required(merged, "type", "trajectory"))
        if trajectory_type not in TRAJECTORY_TYPES:
            raise ValueError(f"trajectory.type must be one of {TRAJECTORY_TYPES}.")
        samples = int(merged.get("samples", 100))
        if samples <= 0:
            raise ValueError("trajectory.samples must be positive.")
        demo_path = merged.get("demo_path", merged.get("dmp_demo_path"))
        return cls(
            type=trajectory_type,
            samples=samples,
            radius_m=float(merged.get("radius_m", 0.025)),
            center_mode=str(merged.get("center_mode", "straight_tip_xy")),
            z_mode=str(merged.get("z_mode", "straight_tip_minus_radius")),
            plane=str(merged.get("plane", "xy")),
            yaw_deg=float(merged.get("yaw_deg", 0.0)),
            offset_xyz_m=_vector3(
                merged.get("offset_xyz_m", (0.0, 0.0, 0.0)),
                "trajectory.offset_xyz_m",
            ),
            center_xyz_m=_optional_vector3(merged.get("center_xyz_m"), "trajectory.center_xyz_m"),
            z_value_m=None if merged.get("z_value_m") is None else float(merged["z_value_m"]),
            radius_x_m=_optional_float(merged.get("radius_x_m")),
            radius_y_m=_optional_float(merged.get("radius_y_m")),
            length_m=_optional_float(merged.get("length_m")),
            side_length_m=_optional_float(merged.get("side_length_m")),
            lissajous_frequency_x=int(merged.get("lissajous_frequency_x", 2)),
            lissajous_frequency_y=int(merged.get("lissajous_frequency_y", 1)),
            lissajous_phase_deg=float(merged.get("lissajous_phase_deg", 90.0)),
            pitch_m=_optional_float(merged.get("pitch_m")),
            turns=_optional_float(merged.get("turns")),
            dmp_demo_path=_optional_path(demo_path, base_path),
            dmp_basis_count=int(merged.get("basis_count", merged.get("dmp_basis_count", 20))),
            dmp_tau=float(merged.get("tau", merged.get("dmp_tau", 1.0))),
            dmp_start_xyz_m=_optional_vector3(
                merged.get("start_xyz_m", merged.get("dmp_start_xyz_m")),
                "trajectory.dmp.start_xyz_m",
            ),
            dmp_goal_xyz_m=_optional_vector3(
                merged.get("goal_xyz_m", merged.get("dmp_goal_xyz_m")),
                "trajectory.dmp.goal_xyz_m",
            ),
        )


def generate_trajectory_waypoints(
    spec: TrajectorySpec,
    assembly: RobotAssemblyConfig,
) -> np.ndarray:
    """Generate world-frame executor waypoints from a scenario trajectory spec."""

    straight_tip = _straight_executor_tip_world(assembly)
    if spec.type == "dmp":
        return _dmp_trajectory(spec)
    center = _resolve_center(spec, straight_tip)
    in_plane_u, in_plane_v, axial = _plane_basis(spec.plane, spec.yaw_deg)
    if spec.type == "circle":
        points = _circle(_required_positive(spec.radius_m, "trajectory.radius_m"), spec.samples)
        return _lift_planar(points, center, in_plane_u, in_plane_v)
    if spec.type == "figure-eight":
        points = _figure_eight(_radius_x(spec, 1.0), _radius_y(spec, 0.5), spec.samples)
        return _lift_planar(points, center, in_plane_u, in_plane_v)
    if spec.type == "ellipse":
        points = _ellipse(_radius_x(spec, 1.0), _radius_y(spec, 1.0), spec.samples)
        return _lift_planar(points, center, in_plane_u, in_plane_v)
    if spec.type == "line":
        points = _line(_line_length(spec), spec.samples)
        return _lift_planar(points, center, in_plane_u, in_plane_v)
    if spec.type == "square":
        points = _square(_square_side(spec), spec.samples)
        return _lift_planar(points, center, in_plane_u, in_plane_v)
    if spec.type == "lissajous":
        points = _lissajous(
            _radius_x(spec, 1.0),
            _radius_y(spec, 1.0),
            spec.lissajous_frequency_x,
            spec.lissajous_frequency_y,
            np.deg2rad(spec.lissajous_phase_deg),
            spec.samples,
        )
        return _lift_planar(points, center, in_plane_u, in_plane_v)
    if spec.type == "helix":
        return _helix(
            center,
            in_plane_u,
            in_plane_v,
            axial,
            _radius_x(spec, 1.0),
            _helix_pitch(spec),
            _helix_turns(spec),
            spec.samples,
        )
    raise ValueError(f"Unsupported trajectory type {spec.type!r}.")


def prepend_tracking_approach(
    waypoints_world: np.ndarray,
    assembly: RobotAssemblyConfig,
    *,
    samples: int,
) -> TrackingWaypointPlan:
    """Prepend a quintic straight-tip approach without duplicating the first path point."""

    waypoints = np.asarray(waypoints_world, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] == 0:
        raise ValueError("waypoints_world must have shape (N, 3) with N > 0.")
    if samples < 0 or samples == 1:
        raise ValueError("samples must be 0 or at least 2.")
    if samples == 0:
        return TrackingWaypointPlan(
            waypoints_world=waypoints.copy(),
            approach_mask=np.zeros(waypoints.shape[0], dtype=bool),
            source_waypoint_index=np.arange(waypoints.shape[0], dtype=int),
        )
    start = _straight_executor_tip_world(assembly)
    progress = np.linspace(0.0, 1.0, samples, endpoint=False)
    blend = progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)
    approach = start[None, :] + blend[:, None] * (waypoints[0] - start)[None, :]
    return TrackingWaypointPlan(
        waypoints_world=np.vstack((approach, waypoints)),
        approach_mask=np.concatenate(
            (np.ones(samples, dtype=bool), np.zeros(waypoints.shape[0], dtype=bool))
        ),
        source_waypoint_index=np.concatenate(
            (
                np.full(samples, -1, dtype=int),
                np.arange(waypoints.shape[0], dtype=int),
            )
        ),
    )


def _straight_executor_tip_world(assembly: RobotAssemblyConfig) -> np.ndarray:
    names = [arm.name for arm in assembly.enabled_arms if arm.role == "executor"]
    if len(names) != 1:
        raise ValueError("Trajectory generation requires exactly one enabled executor arm.")
    arm = assembly.arms[names[0]]
    local_tip = forward_kinematics(
        np.zeros(arm.spatial_arm.params.q_size, dtype=float),
        arm.spatial_arm.params,
    ).tip_pose[:3, 3]
    return assembly.base.initial_pose.compose(arm.mount_pose).transform_point(local_tip)


def _resolve_center(spec: TrajectorySpec, straight_tip: np.ndarray) -> np.ndarray:
    if spec.center_mode == "straight_tip_xy":
        center = straight_tip.copy()
    elif spec.center_mode == "straight_tip":
        center = straight_tip.copy()
    elif spec.center_mode == "explicit":
        if spec.center_xyz_m is None:
            raise ValueError("trajectory.center_xyz_m is required for explicit center_mode.")
        center = spec.center_xyz_m.copy()
    else:
        raise ValueError(f"Unsupported trajectory.center_mode {spec.center_mode!r}.")
    if spec.center_mode == "straight_tip_xy":
        center[2] = _resolve_z(spec, straight_tip, center)
    elif spec.z_mode != "center":
        center[2] = _resolve_z(spec, straight_tip, center)
    return center + spec.offset_xyz_m


def _resolve_z(spec: TrajectorySpec, straight_tip: np.ndarray, center: np.ndarray) -> float:
    if spec.z_mode == "straight_tip_minus_radius":
        return float(straight_tip[2] - _reference_scale(spec))
    if spec.z_mode == "center":
        return float(center[2])
    if spec.z_mode == "explicit":
        if spec.z_value_m is None:
            raise ValueError("trajectory.z_value_m is required for explicit z_mode.")
        return float(spec.z_value_m)
    raise ValueError(f"Unsupported trajectory.z_mode {spec.z_mode!r}.")


def _plane_basis(plane: str, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if plane == "xy":
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        axial = np.array([0.0, 0.0, 1.0])
    elif plane == "xz":
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        axial = np.array([0.0, 1.0, 0.0])
    elif plane == "yz":
        u = np.array([0.0, 1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        axial = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError("trajectory.plane must be one of xy, xz, yz.")
    yaw = np.deg2rad(yaw_deg)
    return (
        np.cos(yaw) * u + np.sin(yaw) * v,
        -np.sin(yaw) * u + np.cos(yaw) * v,
        axial,
    )


def _dmp_trajectory(spec: TrajectorySpec) -> np.ndarray:
    if spec.dmp_demo_path is None:
        raise ValueError("trajectory.dmp.demo_path is required for dmp trajectories.")
    time, demo = load_demonstration(spec.dmp_demo_path)
    start = demo[0] if spec.dmp_start_xyz_m is None else spec.dmp_start_xyz_m
    goal = demo[-1] if spec.dmp_goal_xyz_m is None else spec.dmp_goal_xyz_m
    dmp = DiscreteDMP(basis_count=spec.dmp_basis_count, samples=spec.samples).imitate(time, demo)
    return dmp.rollout(start, goal, tau=spec.dmp_tau).position


def _lift_planar(points: np.ndarray, center: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return center[None, :] + points[:, :1] * u[None, :] + points[:, 1:2] * v[None, :]


def _circle(radius: float, samples: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return np.column_stack((radius * np.cos(t), radius * np.sin(t)))


def _figure_eight(radius_x: float, radius_y: float, samples: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return np.column_stack((radius_x * np.sin(t), radius_y * np.sin(2.0 * t)))


def _ellipse(radius_x: float, radius_y: float, samples: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return np.column_stack((radius_x * np.cos(t), radius_y * np.sin(t)))


def _line(length_m: float, samples: int) -> np.ndarray:
    return np.column_stack((np.linspace(-0.5 * length_m, 0.5 * length_m, samples), np.zeros(samples)))


def _square(side_length_m: float, samples: int) -> np.ndarray:
    t = np.linspace(0.0, 4.0, samples, endpoint=False)
    half = 0.5 * side_length_m
    points = []
    for value in t:
        side = int(value)
        frac = value - side
        if side == 0:
            points.append([half, -half + 2.0 * half * frac])
        elif side == 1:
            points.append([half - 2.0 * half * frac, half])
        elif side == 2:
            points.append([-half, half - 2.0 * half * frac])
        else:
            points.append([-half + 2.0 * half * frac, -half])
    return np.asarray(points, dtype=float)


def _lissajous(
    radius_x: float,
    radius_y: float,
    freq_x: int,
    freq_y: int,
    phase: float,
    samples: int,
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return np.column_stack((
        radius_x * np.sin(freq_x * t + phase),
        radius_y * np.sin(freq_y * t),
    ))


def _helix(
    center: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    axial: np.ndarray,
    radius: float,
    pitch_m: float,
    turns: float,
    samples: int,
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi * turns, samples)
    planar = radius * np.cos(t)[:, None] * u[None, :] + radius * np.sin(t)[:, None] * v[None, :]
    z = np.linspace(-0.5 * pitch_m * turns, 0.5 * pitch_m * turns, samples)
    return center[None, :] + planar + z[:, None] * axial[None, :]


def _reference_scale(spec: TrajectorySpec) -> float:
    values = [
        spec.radius_m,
        spec.radius_x_m,
        spec.radius_y_m,
        None if spec.length_m is None else 0.5 * spec.length_m,
        None if spec.side_length_m is None else 0.5 * spec.side_length_m,
    ]
    positive = [float(value) for value in values if value is not None and value > 0.0]
    if not positive:
        raise ValueError("A positive trajectory scale is required.")
    return max(positive)


def _radius_x(spec: TrajectorySpec, default_scale: float) -> float:
    if spec.radius_x_m is not None and spec.radius_x_m > 0.0:
        return float(spec.radius_x_m)
    return _required_positive(default_scale * spec.radius_m, "trajectory.radius_m")


def _radius_y(spec: TrajectorySpec, default_scale: float) -> float:
    if spec.radius_y_m is not None and spec.radius_y_m > 0.0:
        return float(spec.radius_y_m)
    return _required_positive(default_scale * spec.radius_m, "trajectory.radius_m")


def _line_length(spec: TrajectorySpec) -> float:
    if spec.length_m is not None and spec.length_m > 0.0:
        return float(spec.length_m)
    return _required_positive(2.0 * spec.radius_m, "trajectory.length_m")


def _square_side(spec: TrajectorySpec) -> float:
    if spec.side_length_m is not None and spec.side_length_m > 0.0:
        return float(spec.side_length_m)
    return _required_positive(2.0 * spec.radius_m, "trajectory.side_length_m")


def _helix_pitch(spec: TrajectorySpec) -> float:
    if spec.pitch_m is not None and spec.pitch_m > 0.0:
        return float(spec.pitch_m)
    return _required_positive(spec.radius_m, "trajectory.pitch_m")


def _helix_turns(spec: TrajectorySpec) -> float:
    turns = 1.0 if spec.turns is None else float(spec.turns)
    return _required_positive(turns, "trajectory.turns")


def _required_positive(value: float, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _required(values: dict[str, Any], name: str, section: str) -> object:
    if name not in values:
        raise ValueError(f"Missing required field {section}.{name}.")
    return values[name]


def _vector3(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    return array.copy()


def _optional_vector3(value: object, name: str) -> np.ndarray | None:
    if value is None:
        return None
    return _vector3(value, name)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_path(value: object, base_path: Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute() or base_path is None:
        return path.resolve()
    return (base_path.parent / path).resolve()
