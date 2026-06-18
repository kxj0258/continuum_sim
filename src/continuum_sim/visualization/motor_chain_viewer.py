"""Interactive matplotlib viewer for the offline motor-to-PCC chain."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from continuum_sim.actuation.motor_mapping import (
    MotorParams,
    motor_position_to_tendon_delta,
    motor_velocity_to_tendon_velocity,
    tendon_delta_to_motor_position,
)
from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.physical_tendon import PhysicalTendonPath
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.model.tendon_coupling import (
    build_coupling_matrix,
    coupling_diagnostics,
    physical_tendon_delta_to_q,
    q_to_physical_tendon_delta,
)
from continuum_sim.visualization.axis_limits import (
    apply_axis_limits,
    default_robot_axis_limits,
)

Q_LABELS = ("kx1", "ky1", "eps1", "kx2", "ky2", "eps2", "kx3", "ky3", "eps3")
NAMED_MOTOR_STATES = (
    "zero",
    "motor_1_pull",
    "motor_4_pull",
    "motor_7_pull",
    "segment_1_bend",
    "segment_2_bend",
    "segment_3_bend",
    "three_segment",
)
SEGMENT_COLORS = ("tab:blue", "tab:orange", "tab:green")


@dataclass(frozen=True)
class MotorChainViewData:
    """Computed data for the offline motor-to-shape visualization."""

    motor_position: np.ndarray
    motor_velocity: np.ndarray
    tendon_delta: np.ndarray
    tendon_velocity: np.ndarray
    q_est: np.ndarray
    q_dot_est: np.ndarray
    tip_position: np.ndarray
    centerline: np.ndarray
    segment_centerlines: tuple[np.ndarray, ...]
    diagnostics: dict[str, float | int | bool]


def compute_motor_chain_view_data(
    motor_position: np.ndarray,
    motor_velocity: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    *,
    samples_per_segment: int = 40,
) -> MotorChainViewData:
    """Run the offline motor -> tendon -> q -> FK chain."""
    motor_position_array = _as_motor_vector(motor_position, "motor_position")
    motor_velocity_array = _as_motor_vector(motor_velocity, "motor_velocity")

    tendon_delta = motor_position_to_tendon_delta(motor_position_array, motor_params)
    tendon_velocity = motor_velocity_to_tendon_velocity(motor_velocity_array, motor_params)
    q_est = physical_tendon_delta_to_q(tendon_delta, params, physical_tendons)
    q_dot_est = physical_tendon_delta_to_q(tendon_velocity, params, physical_tendons)
    fk = forward_kinematics(q_est, params, samples_per_segment=samples_per_segment)
    C = build_coupling_matrix(params, physical_tendons)

    return MotorChainViewData(
        motor_position=motor_position_array.copy(),
        motor_velocity=motor_velocity_array.copy(),
        tendon_delta=tendon_delta,
        tendon_velocity=tendon_velocity,
        q_est=q_est,
        q_dot_est=q_dot_est,
        tip_position=fk.tip_pose[:3, 3].copy(),
        centerline=fk.centerline.copy(),
        segment_centerlines=tuple(points.copy() for points in fk.segment_centerlines),
        diagnostics=coupling_diagnostics(C),
    )


def named_motor_chain_state(
    name: str,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    *,
    position_limit_rad: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a named motor position/velocity state."""
    motor_position = np.zeros(9, dtype=float)
    motor_velocity = np.zeros(9, dtype=float)
    pull_value = min(0.5, abs(float(position_limit_rad)))

    if name == "zero":
        return motor_position, motor_velocity
    if name == "motor_1_pull":
        motor_position[0] = pull_value
        return motor_position, motor_velocity
    if name == "motor_4_pull":
        motor_position[3] = pull_value
        return motor_position, motor_velocity
    if name == "motor_7_pull":
        motor_position[6] = pull_value
        return motor_position, motor_velocity

    q_states = {
        "segment_1_bend": np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "segment_2_bend": np.array([0.0, 0.0, 0.0, 0.0, -8.0, 0.0, 0.0, 0.0, 0.0]),
        "segment_3_bend": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 6.0, 0.0]),
        "three_segment": np.array(
            [8.0, 0.0, 0.0, 0.0, -7.0, 0.0, -5.0, 6.0, 0.005]
        ),
    }
    try:
        q_cmd = q_states[name]
    except KeyError as exc:
        raise ValueError(f"Unknown named motor state {name!r}. Choose one of {NAMED_MOTOR_STATES}.") from exc

    tendon_delta = q_to_physical_tendon_delta(q_cmd, params, physical_tendons)
    motor_position = tendon_delta_to_motor_position(tendon_delta, motor_params)
    motor_position = _clip_motor_position(motor_position, position_limit_rad)
    return motor_position, motor_velocity


class MotorChainInteractiveViewer:
    """Matplotlib UI for offline motor-chain exploration."""

    def __init__(
        self,
        params: ThreeSegmentRobotParams,
        physical_tendons: tuple[PhysicalTendonPath, ...],
        motor_params: tuple[MotorParams, ...],
        position_limit_rad: float = 2.0,
        velocity_limit_rad_s: float = 1.0,
        dt: float = 0.02,
        *,
        samples_per_segment: int = 40,
    ) -> None:
        if position_limit_rad <= 0.0:
            raise ValueError("position_limit_rad must be positive.")
        if velocity_limit_rad_s <= 0.0:
            raise ValueError("velocity_limit_rad_s must be positive.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self.params = params
        self.physical_tendons = physical_tendons
        self.motor_params = motor_params
        self.position_limit_rad = float(position_limit_rad)
        self.velocity_limit_rad_s = float(velocity_limit_rad_s)
        self.dt = float(dt)
        self.samples_per_segment = int(samples_per_segment)
        self.axis_limits = default_robot_axis_limits(params)
        self.motor_position = np.zeros(9, dtype=float)
        self.motor_velocity = np.zeros(9, dtype=float)
        self._running = False
        self._updating_sliders = False

        self.fig = plt.figure(figsize=(15.5, 9.0))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("continuum_sim motor chain viewer")
        self.ax = self.fig.add_axes((0.05, 0.36, 0.48, 0.58), projection="3d")
        self.info_ax = self.fig.add_axes((0.57, 0.47, 0.40, 0.47))
        self.info_ax.axis("off")
        self._info_text = self.info_ax.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.2,
        )

        self.position_sliders = self._build_position_sliders()
        self.velocity_sliders = self._build_velocity_sliders()
        self.reset_button, self.run_button, self.step_button, self.zero_velocity_button = (
            self._build_buttons()
        )
        self.radio = self._build_preset_radio()
        self.timer = self.fig.canvas.new_timer(interval=max(1, int(self.dt * 1000.0)))
        self.timer.add_callback(self._on_timer)
        self._connect_controls()
        self.update_plot(redraw=False)

    def _build_position_sliders(self) -> list[Slider]:
        sliders = []
        y0 = 0.285
        dy = 0.027
        for index in range(9):
            axis = self.fig.add_axes((0.10, y0 - index * dy, 0.34, 0.018))
            slider = Slider(
                ax=axis,
                label=f"motor_{index + 1} position [rad]",
                valmin=-self.position_limit_rad,
                valmax=self.position_limit_rad,
                valinit=float(self.motor_position[index]),
                valfmt="% .3f",
            )
            sliders.append(slider)
        return sliders

    def _build_velocity_sliders(self) -> list[Slider]:
        sliders = []
        y0 = 0.285
        dy = 0.027
        for index in range(9):
            axis = self.fig.add_axes((0.59, y0 - index * dy, 0.34, 0.018))
            slider = Slider(
                ax=axis,
                label=f"motor_{index + 1} velocity [rad/s]",
                valmin=-self.velocity_limit_rad_s,
                valmax=self.velocity_limit_rad_s,
                valinit=float(self.motor_velocity[index]),
                valfmt="% .3f",
            )
            sliders.append(slider)
        return sliders

    def _build_buttons(self) -> tuple[Button, Button, Button, Button]:
        reset_axis = self.fig.add_axes((0.06, 0.310, 0.10, 0.035))
        run_axis = self.fig.add_axes((0.18, 0.310, 0.10, 0.035))
        step_axis = self.fig.add_axes((0.30, 0.310, 0.10, 0.035))
        zero_velocity_axis = self.fig.add_axes((0.42, 0.310, 0.12, 0.035))
        return (
            Button(reset_axis, "Reset"),
            Button(run_axis, "Run"),
            Button(step_axis, "Step"),
            Button(zero_velocity_axis, "Zero Vel"),
        )

    def _build_preset_radio(self) -> RadioButtons:
        axis = self.fig.add_axes((0.57, 0.325, 0.40, 0.12))
        return RadioButtons(axis, NAMED_MOTOR_STATES, active=0)

    def _connect_controls(self) -> None:
        for slider in self.position_sliders:
            slider.on_changed(self._on_position_slider_changed)
        for slider in self.velocity_sliders:
            slider.on_changed(self._on_velocity_slider_changed)
        self.reset_button.on_clicked(lambda _event: self.reset())
        self.run_button.on_clicked(lambda _event: self.toggle_run())
        self.step_button.on_clicked(lambda _event: self.step())
        self.zero_velocity_button.on_clicked(lambda _event: self.zero_velocity())
        self.radio.on_clicked(lambda name: self.apply_named_state(str(name)))

    def _on_position_slider_changed(self, _value: float) -> None:
        if self._updating_sliders:
            return
        self.motor_position = np.array([slider.val for slider in self.position_sliders], dtype=float)
        self.update_plot()

    def _on_velocity_slider_changed(self, _value: float) -> None:
        if self._updating_sliders:
            return
        self.motor_velocity = np.array([slider.val for slider in self.velocity_sliders], dtype=float)
        self.update_plot()

    def reset(self) -> MotorChainViewData:
        """Zero motor positions and velocities, then redraw."""
        self.pause()
        return self.set_motor_state(np.zeros(9, dtype=float), np.zeros(9, dtype=float))

    def zero_velocity(self) -> MotorChainViewData:
        """Zero velocities without changing motor positions."""
        self.motor_velocity = np.zeros(9, dtype=float)
        self._sync_velocity_sliders()
        return self.update_plot()

    def toggle_run(self) -> None:
        """Toggle timer-driven velocity integration."""
        if self._running:
            self.pause()
            return
        self._running = True
        self.run_button.label.set_text("Pause")
        self.timer.start()

    def pause(self) -> None:
        """Stop timer-driven velocity integration."""
        self._running = False
        self.timer.stop()
        self.run_button.label.set_text("Run")

    def step(self, redraw: bool = True) -> MotorChainViewData:
        """Integrate motor velocity for one dt step and redraw."""
        next_position = self.motor_position + self.motor_velocity * self.dt
        self.motor_position = _clip_motor_position(next_position, self.position_limit_rad)
        self._sync_position_sliders()
        return self.update_plot(redraw=redraw)

    def apply_named_state(self, name: str) -> MotorChainViewData:
        """Apply a named motor-chain state."""
        self.pause()
        motor_position, motor_velocity = named_motor_chain_state(
            name,
            self.params,
            self.physical_tendons,
            self.motor_params,
            position_limit_rad=self.position_limit_rad,
        )
        return self.set_motor_state(motor_position, motor_velocity)

    def set_motor_state(
        self,
        motor_position: np.ndarray,
        motor_velocity: np.ndarray,
    ) -> MotorChainViewData:
        """Set motor state, synchronize sliders, and redraw."""
        self.motor_position = _clip_motor_position(
            _as_motor_vector(motor_position, "motor_position"),
            self.position_limit_rad,
        )
        velocity = _as_motor_vector(motor_velocity, "motor_velocity")
        self.motor_velocity = np.clip(
            velocity,
            -self.velocity_limit_rad_s,
            self.velocity_limit_rad_s,
        )
        self._sync_position_sliders()
        self._sync_velocity_sliders()
        return self.update_plot()

    def update_plot(self, redraw: bool = True) -> MotorChainViewData:
        """Refresh the plot from current motor state and return view data."""
        view_data = compute_motor_chain_view_data(
            self.motor_position,
            self.motor_velocity,
            self.params,
            self.physical_tendons,
            self.motor_params,
            samples_per_segment=self.samples_per_segment,
        )
        self.ax.cla()
        self._draw_centerlines(view_data)
        self._format_axes(view_data)
        self._info_text.set_text(_format_info_text(view_data))
        if redraw:
            self.fig.canvas.draw_idle()
        return view_data

    def _draw_centerlines(self, view_data: MotorChainViewData) -> None:
        if view_data.centerline.size:
            self.ax.plot(
                view_data.centerline[:, 0],
                view_data.centerline[:, 1],
                view_data.centerline[:, 2],
                color="0.20",
                linewidth=1.0,
                alpha=0.7,
                label="centerline",
            )
        for index, (points, color) in enumerate(
            zip(view_data.segment_centerlines, SEGMENT_COLORS, strict=True),
            start=1,
        ):
            self.ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                linewidth=2.4,
                label=f"seg{index}",
            )
            end = points[-1]
            self.ax.scatter(end[0], end[1], end[2], color=color, s=28)

        base = view_data.centerline[0]
        tip = view_data.tip_position
        self.ax.scatter(base[0], base[1], base[2], color="0.15", s=28, marker="s", label="base")
        self.ax.scatter(tip[0], tip[1], tip[2], color="black", s=55, marker="*", label="tip")

    def _format_axes(self, view_data: MotorChainViewData) -> None:
        apply_axis_limits(self.ax, self.axis_limits)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")
        self.ax.set_title("Motor -> tendon -> PCC shape")
        self.ax.grid(True)
        self.ax.view_init(elev=24, azim=-60)
        self.ax.set_box_aspect((1.0, 1.0, 1.4))
        self.ax.legend(loc="upper left", fontsize=8)

    def _sync_position_sliders(self) -> None:
        self._updating_sliders = True
        try:
            for slider, value in zip(self.position_sliders, self.motor_position, strict=True):
                slider.set_val(float(value))
        finally:
            self._updating_sliders = False

    def _sync_velocity_sliders(self) -> None:
        self._updating_sliders = True
        try:
            for slider, value in zip(self.velocity_sliders, self.motor_velocity, strict=True):
                slider.set_val(float(value))
        finally:
            self._updating_sliders = False

    def _on_timer(self) -> bool:
        if self._running:
            self.step()
        return True

    def show(self) -> None:
        """Start the matplotlib event loop."""
        plt.show()

    def close(self) -> None:
        """Close the viewer figure."""
        self.pause()
        plt.close(self.fig)


def _as_motor_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (9,):
        raise ValueError(f"Expected {name} with shape (9,), got {array.shape}.")
    return array


def _clip_motor_position(motor_position: np.ndarray, position_limit_rad: float) -> np.ndarray:
    limit = abs(float(position_limit_rad))
    return np.clip(np.asarray(motor_position, dtype=float), -limit, limit)


def _format_info_text(view_data: MotorChainViewData) -> str:
    tip = view_data.tip_position
    diagnostics = view_data.diagnostics
    lines = [
        "motor_position [rad]",
        *_format_vector_rows(view_data.motor_position, scale=1.0),
        "",
        "motor_velocity [rad/s]",
        *_format_vector_rows(view_data.motor_velocity, scale=1.0),
        "",
        "tendon_delta [mm]",
        *_format_vector_rows(view_data.tendon_delta, scale=1000.0),
        "",
        "tendon_velocity [mm/s]",
        *_format_vector_rows(view_data.tendon_velocity, scale=1000.0),
        "",
        "q_est [kx ky eps] x3",
    ]
    lines.extend(_format_segment_rows(view_data.q_est))
    lines.append("")
    lines.append("q_dot_est")
    lines.extend(_format_segment_rows(view_data.q_dot_est))
    lines.extend(
        [
            "",
            "tip_position [m]",
            f"  x={tip[0]: .5f}  y={tip[1]: .5f}  z={tip[2]: .5f}",
            "",
            "coupling diagnostics",
            f"  rank={diagnostics['rank']}",
            f"  condition_number={diagnostics['condition_number']: .3e}",
            f"  is_full_rank={diagnostics['is_full_rank']}",
        ]
    )
    return "\n".join(lines)


def _format_vector_rows(values: np.ndarray, *, scale: float) -> list[str]:
    scaled = np.asarray(values, dtype=float) * scale
    rows = []
    for start in range(0, 9, 3):
        chunk = scaled[start : start + 3]
        rows.append(
            "  {}-{}: [{: .4f}, {: .4f}, {: .4f}]".format(
                start + 1,
                start + 3,
                *chunk,
            )
        )
    return rows


def _format_segment_rows(values: np.ndarray) -> list[str]:
    rows = []
    for index, segment_values in enumerate(np.asarray(values, dtype=float).reshape(3, 3), start=1):
        rows.append(
            "  seg{}: {: .4f}, {: .4f}, {: .5f}".format(
                index,
                segment_values[0],
                segment_values[1],
                segment_values[2],
            )
        )
    return rows
