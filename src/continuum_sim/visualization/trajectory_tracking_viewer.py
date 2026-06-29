"""Plotting helpers for offline differential-IK trajectory tracking."""

from __future__ import annotations

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

from continuum_sim.control.differential_ik import TrackingResult
from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.mobile_base_context import MobileBaseArmContext
from continuum_sim.model.robot_params import ThreeSegmentRobotParams


def make_circle_trajectory(
    center: np.ndarray,
    radius: float,
    z: float,
    samples: int,
) -> np.ndarray:
    """Generate a small circular target trajectory in the xy plane."""
    center_array = np.asarray(center, dtype=float)
    if center_array.shape != (2,) and center_array.shape != (3,):
        raise ValueError(f"Expected center with shape (2,) or (3,), got {center_array.shape}.")
    if radius < 0.0:
        raise ValueError(f"radius must be non-negative, got {radius}.")
    if samples < 1:
        raise ValueError(f"samples must be positive, got {samples}.")

    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    trajectory = np.zeros((samples, 3), dtype=float)
    trajectory[:, 0] = center_array[0] + radius * np.cos(angles)
    trajectory[:, 1] = center_array[1] + radius * np.sin(angles)
    trajectory[:, 2] = float(z)
    return trajectory


def make_figure_eight_trajectory(
    center: np.ndarray,
    radius: float,
    z: float,
    samples: int,
) -> np.ndarray:
    """Generate a compact horizontal figure-eight trajectory."""
    center_array = np.asarray(center, dtype=float)
    if center_array.shape != (2,) and center_array.shape != (3,):
        raise ValueError(f"Expected center with shape (2,) or (3,), got {center_array.shape}.")
    if radius < 0.0:
        raise ValueError(f"radius must be non-negative, got {radius}.")
    if samples < 1:
        raise ValueError(f"samples must be positive, got {samples}.")

    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    trajectory = np.zeros((samples, 3), dtype=float)
    trajectory[:, 0] = center_array[0] + radius * np.sin(angles)
    trajectory[:, 1] = center_array[1] + 0.5 * radius * np.sin(2.0 * angles)
    trajectory[:, 2] = float(z)
    return trajectory


def plot_tracking_result(
    result: TrackingResult,
    params: ThreeSegmentRobotParams | None = None,
    *,
    arm_context: MobileBaseArmContext | None = None,
) -> plt.Figure:
    """Plot target/actual tip trajectory, error, motor velocity, and position."""
    _validate_tracking_result(result)
    context = arm_context or MobileBaseArmContext.identity()
    target_position_world = context.local_points_to_world(result.target_position)
    tip_position_world = context.local_points_to_world(result.tip_position)
    fig = plt.figure(figsize=(13.0, 9.0))
    trajectory_ax = fig.add_subplot(2, 2, 1, projection="3d")
    error_ax = fig.add_subplot(2, 2, 2)
    velocity_ax = fig.add_subplot(2, 2, 3)
    position_ax = fig.add_subplot(2, 2, 4)

    trajectory_ax.plot(
        target_position_world[:, 0],
        target_position_world[:, 1],
        target_position_world[:, 2],
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label="target",
    )
    trajectory_ax.plot(
        tip_position_world[:, 0],
        tip_position_world[:, 1],
        tip_position_world[:, 2],
        color="tab:blue",
        linewidth=1.8,
        label="tip",
    )
    trajectory_ax.scatter(
        target_position_world[0, 0],
        target_position_world[0, 1],
        target_position_world[0, 2],
        color="tab:orange",
        s=26,
        marker="o",
    )
    trajectory_ax.scatter(
        tip_position_world[-1, 0],
        tip_position_world[-1, 1],
        tip_position_world[-1, 2],
        color="black",
        s=36,
        marker="*",
    )
    _format_trajectory_axes(
        trajectory_ax,
        target_position_world,
        tip_position_world,
        params,
    )

    error_ax.plot(result.time, result.error_norm, color="tab:red", linewidth=1.6)
    error_ax.set_title("Tip position error")
    error_ax.set_xlabel("time [s]")
    error_ax.set_ylabel("error norm [m]")
    error_ax.grid(True, alpha=0.35)

    for index in range(result.motor_velocity.shape[1]):
        velocity_ax.plot(result.time, result.motor_velocity[:, index], linewidth=0.9)
    velocity_ax.set_title("Motor velocity")
    velocity_ax.set_xlabel("time [s]")
    velocity_ax.set_ylabel("rad/s")
    velocity_ax.grid(True, alpha=0.35)

    for index in range(result.motor_position.shape[1]):
        position_ax.plot(result.time, result.motor_position[:, index], linewidth=0.9)
    position_ax.set_title("Motor position")
    position_ax.set_xlabel("time [s]")
    position_ax.set_ylabel("rad")
    position_ax.grid(True, alpha=0.35)

    fig.tight_layout()
    return fig


def animate_tracking_result(
    result: TrackingResult,
    params: ThreeSegmentRobotParams,
    *,
    samples_per_segment: int = 30,
    interval_ms: int = 50,
    stride: int = 1,
    arm_context: MobileBaseArmContext | None = None,
) -> tuple[plt.Figure, matplotlib.animation.FuncAnimation]:
    """Animate the offline tracking process in a matplotlib 3D window."""
    _validate_tracking_result(result)
    if samples_per_segment < 2:
        raise ValueError(f"samples_per_segment must be at least 2, got {samples_per_segment}.")
    if interval_ms <= 0:
        raise ValueError(f"interval_ms must be positive, got {interval_ms}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")

    context = arm_context or MobileBaseArmContext.identity()
    frame_indices = np.arange(0, result.time.shape[0], stride, dtype=int)
    if frame_indices[-1] != result.time.shape[0] - 1:
        frame_indices = np.append(frame_indices, result.time.shape[0] - 1)

    centerline_cache: dict[int, np.ndarray] = {}

    def centerline_for_frame(frame: int) -> np.ndarray:
        if frame not in centerline_cache:
            fk = forward_kinematics(
                result.q_est[frame],
                params,
                samples_per_segment=samples_per_segment,
            )
            centerline_cache[frame] = context.local_points_to_world(fk.centerline)
        return centerline_cache[frame]

    target_position_world = context.local_points_to_world(result.target_position)
    tip_position_world = context.local_points_to_world(result.tip_position)
    axis_limits = _animation_axis_limits(
        target_position_world,
        tip_position_world,
        params,
        frame_indices,
        centerline_for_frame,
    )
    fig = plt.figure(figsize=(8.5, 7.0))
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.plot(
        target_position_world[:, 0],
        target_position_world[:, 1],
        target_position_world[:, 2],
        color="tab:orange",
        linestyle="--",
        linewidth=1.6,
        label="target",
    )
    actual_line, = axis.plot([], [], [], color="tab:blue", linewidth=1.8, label="tip path")
    robot_line, = axis.plot([], [], [], color="0.15", linewidth=2.6, label="robot")
    tip_point, = axis.plot(
        [],
        [],
        [],
        color="tab:blue",
        marker="o",
        markersize=6,
        linestyle="None",
        label="tip",
    )
    target_point, = axis.plot(
        [],
        [],
        [],
        color="tab:orange",
        marker="o",
        markersize=6,
        linestyle="None",
        label="current target",
    )
    time_text = axis.text2D(0.03, 0.96, "", transform=axis.transAxes)

    axis.set_xlim(axis_limits["x"])
    axis.set_ylim(axis_limits["y"])
    axis.set_zlim(axis_limits["z"])
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title("PCC trajectory tracking replay in world frame")
    axis.grid(True)
    axis.legend(loc="upper left")
    axis.view_init(elev=24, azim=-60)
    axis.set_box_aspect((1.0, 1.0, 1.0))

    def update(frame: int):
        centerline = centerline_for_frame(frame)
        actual = tip_position_world[: frame + 1]
        target = target_position_world[frame]
        tip = tip_position_world[frame]

        actual_line.set_data(actual[:, 0], actual[:, 1])
        actual_line.set_3d_properties(actual[:, 2])
        robot_line.set_data(centerline[:, 0], centerline[:, 1])
        robot_line.set_3d_properties(centerline[:, 2])
        tip_point.set_data([tip[0]], [tip[1]])
        tip_point.set_3d_properties([tip[2]])
        target_point.set_data([target[0]], [target[1]])
        target_point.set_3d_properties([target[2]])
        time_text.set_text(f"t={result.time[frame]:.2f} s, error={result.error_norm[frame]:.3e} m")
        return actual_line, robot_line, tip_point, target_point, time_text

    anim = matplotlib.animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    update(int(frame_indices[0]))
    fig.tight_layout()
    return fig, anim


def _format_trajectory_axes(
    axis,
    target_position: np.ndarray,
    tip_position: np.ndarray,
    params: ThreeSegmentRobotParams | None,
) -> None:
    all_points = np.vstack((target_position, tip_position))
    xy_extent = float(np.max(np.abs(all_points[:, :2]))) if all_points.size else 0.01
    z_max = float(np.max(all_points[:, 2])) if all_points.size else 0.01
    if params is not None:
        z_max = max(z_max, float(np.sum(params.segment_lengths)))
    limit = max(xy_extent * 1.25, 0.01)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(0.0, max(z_max * 1.1, 0.01))
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title("Target and actual tip trajectory in world frame")
    axis.legend(loc="upper left")
    axis.grid(True)
    axis.set_box_aspect((1.0, 1.0, 1.2))


def _animation_axis_limits(
    target_position: np.ndarray,
    tip_position: np.ndarray,
    params: ThreeSegmentRobotParams,
    frame_indices: np.ndarray,
    centerline_for_frame,
) -> dict[str, tuple[float, float]]:
    centerlines = [centerline_for_frame(int(frame)) for frame in frame_indices]
    all_points = np.vstack(
        (
            target_position,
            tip_position,
            *centerlines,
        )
    )
    total_length = float(np.sum(params.segment_lengths))
    if all_points.size == 0:
        all_points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, total_length]])
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    mins[2] = min(0.0, mins[2])
    maxs[2] = max(total_length, maxs[2])
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half_span = max(0.5 * span * 1.12, total_length * 0.55, 0.01)
    return {
        "x": (center[0] - half_span, center[0] + half_span),
        "y": (center[1] - half_span, center[1] + half_span),
        "z": (center[2] - half_span, center[2] + half_span),
    }


def _validate_tracking_result(result: TrackingResult) -> None:
    sample_count = result.time.shape[0]
    expected_shapes = {
        "target_position": (sample_count, 3),
        "tip_position": (sample_count, 3),
        "error_norm": (sample_count,),
        "motor_position": (sample_count, 9),
        "motor_velocity": (sample_count, 9),
        "q_est": (sample_count, 9),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(getattr(result, name), dtype=float)
        if value.shape != shape:
            raise ValueError(f"Expected {name} with shape {shape}, got {value.shape}.")
