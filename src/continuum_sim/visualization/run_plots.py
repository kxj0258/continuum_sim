"""Static plot exporters for saved CLI simulation runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_run_plots(result: object, output_dir: str | Path, *, task_name: str) -> list[Path]:
    """Save summary PNG plots for a rollout result."""

    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    if _has_positions(result):
        saved.append(_save_trajectory_plot(result, output_path / "trajectory.png", task_name))
    if hasattr(result, "error_norm"):
        saved.append(_save_series_plot(
            _time(result),
            np.asarray(getattr(result, "error_norm"), dtype=float),
            output_path / "error.png",
            title="Tracking Error",
            ylabel="error [m]",
        ))
    if hasattr(result, "motor_velocity"):
        saved.append(_save_multiseries_plot(
            _time(result),
            np.asarray(getattr(result, "motor_velocity"), dtype=float),
            output_path / "motor_velocity.png",
            title="Motor Velocity",
            ylabel="rad/s",
        ))
    if hasattr(result, "tendon_length"):
        tendon_length = np.asarray(getattr(result, "tendon_length"), dtype=float)
        if tendon_length.ndim == 2 and tendon_length.shape[1] > 0:
            saved.append(_save_multiseries_plot(
                _time(result),
                tendon_length,
                output_path / "tendon_length.png",
                title="Tendon Length",
                ylabel="m",
            ))
    if hasattr(result, "normal_force_n"):
        saved.append(_save_wiping_force_plot(result, output_path / "wiping_force.png"))
    plt.close("all")
    return saved


def _save_trajectory_plot(result: object, path: Path, task_name: str) -> Path:
    import matplotlib.pyplot as plt

    target = _position_array(getattr(result, "target_position"))
    tip = _result_tip_position(result)
    fig = plt.figure(figsize=(8.0, 6.5))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.plot(
        target[:, 0],
        target[:, 1],
        target[:, 2],
        color="tab:orange",
        linestyle="--",
        linewidth=1.7,
        label="target",
    )
    axis.plot(
        tip[:, 0],
        tip[:, 1],
        tip[:, 2],
        color="tab:blue",
        linewidth=1.7,
        label="tip",
    )
    axis.scatter(target[0, 0], target[0, 1], target[0, 2], color="tab:orange", s=22)
    axis.scatter(tip[-1, 0], tip[-1, 1], tip[-1, 2], color="black", marker="*", s=36)
    _set_equal_axes(axis, np.vstack((target, tip)))
    axis.set_title(f"{task_name} trajectory")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.legend(loc="upper left")
    axis.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_series_plot(
    time: np.ndarray,
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    ylabel: str,
) -> Path:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.plot(time, values, linewidth=1.6)
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_multiseries_plot(
    time: np.ndarray,
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    ylabel: str,
) -> Path:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    if values.ndim == 1:
        axis.plot(time, values, linewidth=1.2)
    elif values.ndim == 2:
        for index in range(values.shape[1]):
            axis.plot(time, values[:, index], linewidth=0.9, label=str(index))
        if values.shape[1] <= 12:
            axis.legend(loc="upper right", ncol=3, fontsize=8)
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_wiping_force_plot(result: object, path: Path) -> Path:
    import matplotlib.pyplot as plt

    time = _time(result)
    normal_force = np.asarray(getattr(result, "normal_force_n"), dtype=float)
    force_error = np.asarray(getattr(result, "force_error_n"), dtype=float)
    contact_proxy = np.asarray(getattr(result, "contact_proxy_m"), dtype=float)
    target_force = normal_force + force_error
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    axes[0].plot(time, normal_force, color="tab:blue", linewidth=1.6, label="normal force")
    axes[0].plot(time, target_force, color="tab:green", linestyle="--", linewidth=1.2, label="target force")
    axes[0].plot(time, force_error, color="tab:red", linewidth=1.0, label="force error")
    axes[0].set_title("Wiping Normal Force")
    axes[0].set_ylabel("force [N]")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="upper right")

    axes[1].plot(time, 1000.0 * contact_proxy, color="tab:purple", linewidth=1.4)
    axes[1].axhline(0.0, color="0.4", linestyle="--", linewidth=0.9)
    axes[1].set_title("Contact Distance / Penetration Proxy")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("proxy [mm]")
    axes[1].grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _has_positions(result: object) -> bool:
    return hasattr(result, "target_position") and (
        hasattr(result, "tip_position") or hasattr(result, "tip_pose")
    )


def _position_array(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected position array with shape (N, 3), got {array.shape}.")
    return array


def _result_tip_position(result: object) -> np.ndarray:
    if hasattr(result, "tip_position"):
        return _position_array(getattr(result, "tip_position"))
    tip_pose = np.asarray(getattr(result, "tip_pose"), dtype=float)
    if tip_pose.ndim != 3 or tip_pose.shape[1:] != (4, 4):
        raise ValueError(f"Expected tip_pose with shape (N, 4, 4), got {tip_pose.shape}.")
    return tip_pose[:, :3, 3]


def _time(result: object) -> np.ndarray:
    if hasattr(result, "time"):
        return np.asarray(getattr(result, "time"), dtype=float)
    sample_count = _result_tip_position(result).shape[0]
    return np.arange(sample_count, dtype=float)


def _set_equal_axes(axis, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = max(0.5 * span * 1.15, 0.01)
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
    axis.set_box_aspect((1.0, 1.0, 1.0))
