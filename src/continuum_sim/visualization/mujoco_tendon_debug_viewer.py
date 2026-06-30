"""Interactive matplotlib controls for the MuJoCo tendon-position backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from continuum_sim.backends import BackendState
from continuum_sim.config import MujocoConfig
from continuum_sim.model import PhysicalTendonPath, ThreeSegmentRobotParams
from continuum_sim.model.tendon_coupling import physical_tendon_delta_to_q


TENDON_DEBUG_NAMED_COMMANDS = (
    "zero",
    "executor_tendon_1_pull",
    "observer_tendon_1_pull",
    "executor_segment_1_triplet",
    "observer_segment_1_triplet",
    "all_tendons_pull",
    "base_x_plus",
    "base_y_plus",
    "base_z_plus",
    "base_yaw_plus",
)

_NONINTERACTIVE_MATPLOTLIB_BACKENDS = frozenset(
    {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
        "module://matplotlib_inline.backend_inline",
    }
)


@dataclass(frozen=True)
class MujocoTendonDebugViewData:
    """Live command and state readback for the tendon debug UI."""

    time_s: float
    commanded_tendon_delta: np.ndarray
    actual_tendon_length: np.ndarray
    actuator_force: np.ndarray
    tip_position: np.ndarray
    q_est: np.ndarray
    tendon_error: np.ndarray


def compute_mujoco_tendon_debug_view_data(
    commanded_tendon_delta: np.ndarray,
    state: BackendState,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    *,
    tendon_indices: np.ndarray | None = None,
) -> MujocoTendonDebugViewData:
    """Assemble UI readbacks from a MuJoCo tendon backend state."""

    tendon_count = len(physical_tendons)
    command = _as_tendon_vector(
        commanded_tendon_delta,
        "commanded_tendon_delta",
        expected_size=tendon_count,
    )
    actual_tendon_length = _select_tendon_vector(
        state.tendon_length,
        "state.tendon_length",
        expected_size=tendon_count,
        tendon_indices=tendon_indices,
    )
    actuator_force = _select_tendon_vector(
        state.actuator_force,
        "state.actuator_force",
        expected_size=tendon_count,
        tendon_indices=tendon_indices,
    )
    if state.tip_pose.shape != (4, 4):
        raise ValueError(f"Expected state.tip_pose with shape (4, 4), got {state.tip_pose.shape}.")

    q_est = _estimate_q_for_tendon_view(actual_tendon_length, params, physical_tendons)
    tip_position = np.asarray(state.tip_pose[:3, 3], dtype=float).copy()
    tendon_error = command - actual_tendon_length
    return MujocoTendonDebugViewData(
        time_s=float(state.time),
        commanded_tendon_delta=command.copy(),
        actual_tendon_length=actual_tendon_length.copy(),
        actuator_force=actuator_force.copy(),
        tip_position=tip_position,
        q_est=q_est,
        tendon_error=tendon_error,
    )


def named_tendon_command(name: str, config: MujocoConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return named tendon and base commands using the existing YAML parameters."""

    count = config.tendon_model.count
    command = np.zeros(count, dtype=float)
    base_command = np.zeros((6,), dtype=float)
    single_pull = float(config.smoke_tests.single_tendon_delta_m)
    triplet_pull = float(config.smoke_tests.symmetric_tendon_delta_m)

    if name == "zero":
        return command, base_command
    if name == "executor_tendon_1_pull":
        command[0] = single_pull
    elif name == "observer_tendon_1_pull":
        observer_start = _observer_tendon_start(count)
        command[observer_start] = single_pull
    elif name == "executor_segment_1_triplet":
        _require_tendon_count(count, 3, name)
        command[0:3] = triplet_pull
    elif name == "observer_segment_1_triplet":
        observer_start = _observer_tendon_start(count)
        _require_tendon_count(count, observer_start + 3, name)
        command[observer_start : observer_start + 3] = triplet_pull
    elif name == "all_tendons_pull":
        command[:] = triplet_pull
    elif name == "base_x_plus":
        base_command[0] = 0.01
    elif name == "base_y_plus":
        base_command[1] = 0.01
    elif name == "base_z_plus":
        base_command[2] = 0.01
    elif name == "base_yaw_plus":
        base_command[5] = np.deg2rad(5.0)
    else:
        raise ValueError(
            f"Unknown named tendon command {name!r}. Choose one of {TENDON_DEBUG_NAMED_COMMANDS}."
        )
    return clip_tendon_command(command, config.actuators.tendon_position.ctrlrange_m), base_command


def _require_tendon_count(count: int, minimum: int, command_name: str) -> None:
    if count < minimum:
        raise ValueError(
            f"Named command {command_name!r} requires at least {minimum} tendons, got {count}."
        )


def _observer_tendon_start(count: int) -> int:
    if count % 2 != 0:
        raise ValueError(f"Dual-arm tendon count must be even, got {count}.")
    return count // 2


def clip_tendon_command(
    tendon_command: np.ndarray,
    ctrlrange_m: tuple[float, float],
) -> np.ndarray:
    """Clip tendon commands to the configured actuator control range."""

    lower, upper = ctrlrange_m
    return np.clip(np.asarray(tendon_command, dtype=float), lower, upper)


def _is_noninteractive_matplotlib_backend(backend_name: str) -> bool:
    """Return True when the active matplotlib backend cannot open a live GUI window."""

    return backend_name.strip().lower() in _NONINTERACTIVE_MATPLOTLIB_BACKENDS


class MujocoTendonMonitorPanel:
    """Read-only matplotlib panel for live tendon command/state monitoring."""

    def __init__(
        self,
        config: MujocoConfig,
        params: ThreeSegmentRobotParams,
        physical_tendons: tuple[PhysicalTendonPath, ...],
        *,
        title: str = "continuum_sim MuJoCo tendon monitor",
        tendon_indices: np.ndarray | None = None,
    ) -> None:
        if config.control_mode != "tendon_position":
            raise ValueError(
                "MujocoTendonMonitorPanel requires control_mode 'tendon_position'."
            )
        if tendon_indices is None and len(physical_tendons) != config.tendon_model.count:
            raise ValueError(
                "physical_tendons length must match tendon_model.count, got "
                f"{len(physical_tendons)} and {config.tendon_model.count}."
            )

        self.config = config
        self.params = params
        self.physical_tendons = physical_tendons
        self.tendon_indices = tendon_indices
        self.tendon_count = len(physical_tendons)
        self.command_limit = config.actuators.tendon_position.ctrlrange_m
        self.force_limit = config.actuators.tendon_position.forcerange_n

        self.fig = plt.figure(figsize=(15.5, 9.0))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(title)
        self.length_ax = self.fig.add_axes((0.06, 0.57, 0.40, 0.34))
        self.force_ax = self.fig.add_axes((0.54, 0.57, 0.40, 0.34))
        self.info_ax = self.fig.add_axes((0.54, 0.18, 0.40, 0.33))
        self.info_ax.axis("off")
        self._info_text = self.info_ax.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
        )

    def update_from_state(
        self,
        commanded_tendon_delta: np.ndarray,
        state: BackendState,
        *,
        redraw: bool = True,
    ) -> MujocoTendonDebugViewData:
        view_data = compute_mujoco_tendon_debug_view_data(
            commanded_tendon_delta,
            state,
            self.params,
            self.physical_tendons,
            tendon_indices=self.tendon_indices,
        )
        return self.update_from_view_data(view_data, redraw=redraw)

    def update_from_view_data(
        self,
        view_data: MujocoTendonDebugViewData,
        *,
        redraw: bool = True,
    ) -> MujocoTendonDebugViewData:
        self._draw_length_chart(view_data)
        self._draw_force_chart(view_data)
        self._info_text.set_text(_format_debug_info_text(view_data))
        if redraw:
            self.fig.canvas.draw_idle()
        return view_data

    def show(self, *, block: bool = True) -> None:
        """Show the monitor figure."""

        if _is_noninteractive_matplotlib_backend(plt.get_backend()):
            return
        plt.show(block=block)
        if not block:
            self.flush_events()

    def flush_events(self) -> None:
        """Process pending GUI events for the monitor figure."""

        if not plt.fignum_exists(self.fig.number):
            return
        canvas = getattr(self.fig, "canvas", None)
        if canvas is None:
            return
        flush = getattr(canvas, "flush_events", None)
        if callable(flush):
            flush()

    def close(self) -> None:
        """Close the monitor figure."""

        plt.close(self.fig)

    def _draw_length_chart(self, view_data: MujocoTendonDebugViewData) -> None:
        self.length_ax.cla()
        indices = np.arange(1, self.tendon_count + 1, dtype=float)
        self.length_ax.bar(
            indices - 0.18,
            1000.0 * view_data.commanded_tendon_delta,
            width=0.34,
            color="tab:orange",
            alpha=0.55,
            label="command",
        )
        self.length_ax.bar(
            indices + 0.18,
            1000.0 * view_data.actual_tendon_length,
            width=0.34,
            color="tab:blue",
            alpha=0.75,
            label="actual",
        )
        lower_mm = 1000.0 * min(
            self.command_limit[0],
            float(np.min(view_data.actual_tendon_length)),
            float(np.min(view_data.commanded_tendon_delta)),
        )
        upper_mm = 1000.0 * max(
            self.command_limit[1],
            float(np.max(view_data.actual_tendon_length)),
            float(np.max(view_data.commanded_tendon_delta)),
        )
        margin_mm = max(0.5, 0.1 * max(abs(lower_mm), abs(upper_mm), 1.0))
        self.length_ax.set_xlim(0.3, self.tendon_count + 0.7)
        self.length_ax.set_ylim(lower_mm - margin_mm, upper_mm + margin_mm)
        self.length_ax.set_xticks(
            indices,
            [str(index) for index in range(1, self.tendon_count + 1)],
        )
        self.length_ax.set_ylabel("tendon delta [mm]")
        self.length_ax.set_title("Commanded vs actual tendon length")
        self.length_ax.grid(True, axis="y", alpha=0.25)
        self.length_ax.legend(loc="upper right", fontsize=8)

    def _draw_force_chart(self, view_data: MujocoTendonDebugViewData) -> None:
        self.force_ax.cla()
        indices = np.arange(1, self.tendon_count + 1, dtype=float)
        self.force_ax.bar(
            indices,
            view_data.actuator_force,
            width=0.56,
            color="tab:red",
            alpha=0.75,
        )
        lower = min(0.0, float(np.min(view_data.actuator_force)))
        upper = max(float(np.max(view_data.actuator_force)), 0.0)
        reference = max(abs(lower), abs(upper), 1.0e-3)
        margin = max(0.01, 0.15 * reference)
        self.force_ax.set_xlim(0.3, self.tendon_count + 0.7)
        self.force_ax.set_ylim(lower - margin, upper + margin)
        self.force_ax.set_xticks(
            indices,
            [str(index) for index in range(1, self.tendon_count + 1)],
        )
        self.force_ax.set_ylabel("actuator force [N]")
        self.force_ax.set_title("Actual tendon pull force")
        self.force_ax.grid(True, axis="y", alpha=0.25)
        self.force_ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")


class MujocoTendonDebugViewer:
    """Matplotlib control panel for tendon-position MuJoCo experiments."""

    def __init__(
        self,
        backend,
        config: MujocoConfig,
        params: ThreeSegmentRobotParams,
        physical_tendons: tuple[PhysicalTendonPath, ...],
        *,
        control_dt: float = 0.02,
        state_update_callback: Callable[[BackendState], None] | None = None,
    ) -> None:
        if config.control_mode != "tendon_position":
            raise ValueError(
                "MujocoTendonDebugViewer requires control_mode 'tendon_position'."
            )
        if control_dt <= 0.0:
            raise ValueError(f"control_dt must be positive, got {control_dt}.")
        if len(physical_tendons) != config.tendon_model.count:
            raise ValueError(
                "physical_tendons length must match tendon_model.count, got "
                f"{len(physical_tendons)} and {config.tendon_model.count}."
            )

        self.backend = backend
        self.config = config
        self.params = params
        self.physical_tendons = physical_tendons
        self.control_dt = float(control_dt)
        self.n_substeps = max(1, round(self.control_dt / self.config.solver.timestep))
        self.command_limit = config.actuators.tendon_position.ctrlrange_m
        self.force_limit = config.actuators.tendon_position.forcerange_n
        self.state_update_callback = state_update_callback
        self.tendon_count = config.tendon_model.count
        self.base_command_rpy = np.zeros((6,), dtype=float)

        self.commanded_tendon_delta = np.zeros(self.tendon_count, dtype=float)
        self.state = self.backend.reset()
        self._running = False
        self._updating_controls = False

        self.fig = plt.figure(figsize=(15.5, 9.0))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("continuum_sim MuJoCo tendon debug viewer")
        self.length_ax = self.fig.add_axes((0.06, 0.57, 0.40, 0.34))
        self.force_ax = self.fig.add_axes((0.54, 0.57, 0.40, 0.34))
        self.info_ax = self.fig.add_axes((0.54, 0.415, 0.40, 0.11))
        self.info_ax.axis("off")
        self._info_text = self.info_ax.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
        )

        self.command_sliders = self._build_command_sliders()
        self.base_sliders = self._build_base_sliders()
        self.reset_button, self.run_button, self.step_button, self.zero_button = (
            self._build_buttons()
        )
        self.radio = self._build_preset_radio()
        self.timer = self.fig.canvas.new_timer(
            interval=max(1, int(self.control_dt * 1000.0))
        )
        self.timer.add_callback(self._on_timer)
        self._connect_controls()
        self._notify_state_updated()
        self.update_view(redraw=False)

    def _build_command_sliders(self) -> list[Slider]:
        sliders: list[Slider] = []
        lower, upper = self.command_limit
        y0 = 0.355
        dy = 0.020
        is_dual_arm = getattr(self.config.model, "type", None) == "dual_distributed_links"
        tendons_per_column = max(1, self.tendon_count // 2) if is_dual_arm else self.tendon_count
        for index in range(self.tendon_count):
            column = 0 if index < tendons_per_column else 1
            local_index = index if column == 0 else index - tendons_per_column
            x0 = 0.08 if column == 0 else 0.54
            axis = self.fig.add_axes((x0, y0 - local_index * dy, 0.36, 0.012))
            slider = Slider(
                ax=axis,
                label=_tendon_slider_label(index, tendons_per_column=tendons_per_column),
                valmin=lower,
                valmax=upper,
                valinit=float(self.commanded_tendon_delta[index]),
                valfmt="% .4f",
            )
            sliders.append(slider)
        return sliders

    def _build_base_sliders(self) -> list[Slider]:
        labels = ("base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw")
        limits = (
            (-0.10, 0.10),
            (-0.10, 0.10),
            (-0.10, 0.10),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
        )
        sliders: list[Slider] = []
        for index, (label, (lower, upper)) in enumerate(zip(labels, limits, strict=True)):
            axis = self.fig.add_axes((0.08 + index * 0.145, 0.055, 0.11, 0.018))
            slider = Slider(
                ax=axis,
                label=label,
                valmin=lower,
                valmax=upper,
                valinit=float(self.base_command_rpy[index]),
                valfmt="% .4f",
            )
            sliders.append(slider)
        return sliders

    def _build_buttons(self) -> tuple[Button, Button, Button, Button]:
        reset_axis = self.fig.add_axes((0.08, 0.405, 0.12, 0.035))
        run_axis = self.fig.add_axes((0.22, 0.405, 0.12, 0.035))
        step_axis = self.fig.add_axes((0.36, 0.405, 0.12, 0.035))
        zero_axis = self.fig.add_axes((0.50, 0.405, 0.14, 0.035))
        return (
            Button(reset_axis, "Reset Sim"),
            Button(run_axis, "Run"),
            Button(step_axis, "Step"),
            Button(zero_axis, "Zero Cmd"),
        )

    def _build_preset_radio(self) -> RadioButtons:
        axis = self.fig.add_axes((0.08, 0.455, 0.38, 0.095))
        return RadioButtons(axis, TENDON_DEBUG_NAMED_COMMANDS, active=0)

    def _connect_controls(self) -> None:
        for slider in self.command_sliders:
            slider.on_changed(self._on_command_slider_changed)
        for slider in self.base_sliders:
            slider.on_changed(self._on_base_slider_changed)
        self.reset_button.on_clicked(lambda _event: self.reset())
        self.run_button.on_clicked(lambda _event: self.toggle_run())
        self.step_button.on_clicked(lambda _event: self.step())
        self.zero_button.on_clicked(lambda _event: self.zero_command())
        self.radio.on_clicked(lambda name: self.apply_named_command(str(name)))

    def _on_command_slider_changed(self, _value: float) -> None:
        if self._updating_controls:
            return
        self.commanded_tendon_delta = clip_tendon_command(
            np.array([slider.val for slider in self.command_sliders], dtype=float),
            self.command_limit,
        )
        if self._running:
            self.update_view()
            return
        self.step()

    def _on_base_slider_changed(self, _value: float) -> None:
        if self._updating_controls:
            return
        self.base_command_rpy = np.array([slider.val for slider in self.base_sliders], dtype=float)
        if self._running:
            self.update_view()
            return
        self.step()

    def reset(self) -> MujocoTendonDebugViewData:
        """Reset the simulation and clear all tendon commands."""

        self.pause()
        self.state = self.backend.reset()
        self.commanded_tendon_delta = np.zeros(self.tendon_count, dtype=float)
        self.base_command_rpy = np.zeros((6,), dtype=float)
        self._sync_command_sliders()
        self._sync_base_sliders()
        self._notify_state_updated()
        return self.update_view()

    def zero_command(self) -> MujocoTendonDebugViewData:
        """Zero tendon commands without resetting the current MuJoCo state."""

        self.base_command_rpy = np.zeros((6,), dtype=float)
        self._sync_base_sliders()
        return self.set_command(np.zeros(self.tendon_count, dtype=float))

    def set_command(
        self,
        tendon_command: np.ndarray,
        *,
        simulate: bool = True,
    ) -> MujocoTendonDebugViewData:
        """Set tendon command, synchronize sliders, and optionally step MuJoCo."""

        self.commanded_tendon_delta = clip_tendon_command(tendon_command, self.command_limit)
        self._sync_command_sliders()
        if simulate:
            return self.step()
        return self.update_view()

    def apply_named_command(self, name: str) -> MujocoTendonDebugViewData:
        """Apply a named tendon command."""

        tendon_command, base_command = named_tendon_command(name, self.config)
        self.base_command_rpy = base_command
        self._sync_base_sliders()
        return self.set_command(tendon_command)

    def toggle_run(self) -> None:
        """Toggle timer-driven MuJoCo stepping under the current tendon command."""

        if self._running:
            self.pause()
            return
        self._running = True
        self.run_button.label.set_text("Pause")
        self.timer.start()

    def pause(self) -> None:
        """Pause timer-driven MuJoCo stepping."""

        self._running = False
        self.timer.stop()
        self.run_button.label.set_text("Run")

    def step(self, redraw: bool = True) -> MujocoTendonDebugViewData:
        """Advance MuJoCo by one control step under the current tendon command."""

        self.state = self.backend.step(
            self._combined_control(),
            n_substeps=self.n_substeps,
        )
        self._notify_state_updated()
        return self.update_view(redraw=redraw)

    def update_view(self, redraw: bool = True) -> MujocoTendonDebugViewData:
        """Refresh charts and text from the latest MuJoCo backend state."""

        view_data = compute_mujoco_tendon_debug_view_data(
            self.commanded_tendon_delta,
            self.state,
            self.params,
            self.physical_tendons,
        )
        self._draw_length_chart(view_data)
        self._draw_force_chart(view_data)
        self._info_text.set_text(_format_debug_info_text(view_data))
        if redraw:
            self.fig.canvas.draw_idle()
        return view_data

    def _draw_length_chart(self, view_data: MujocoTendonDebugViewData) -> None:
        self.length_ax.cla()
        indices = np.arange(1, self.tendon_count + 1, dtype=float)
        self.length_ax.bar(
            indices - 0.18,
            1000.0 * view_data.commanded_tendon_delta,
            width=0.34,
            color="tab:orange",
            alpha=0.55,
            label="command",
        )
        self.length_ax.bar(
            indices + 0.18,
            1000.0 * view_data.actual_tendon_length,
            width=0.34,
            color="tab:blue",
            alpha=0.75,
            label="actual",
        )
        lower_mm = 1000.0 * min(
            self.command_limit[0],
            float(np.min(view_data.actual_tendon_length)),
            float(np.min(view_data.commanded_tendon_delta)),
        )
        upper_mm = 1000.0 * max(
            self.command_limit[1],
            float(np.max(view_data.actual_tendon_length)),
            float(np.max(view_data.commanded_tendon_delta)),
        )
        margin_mm = max(0.5, 0.1 * max(abs(lower_mm), abs(upper_mm), 1.0))
        self.length_ax.set_xlim(0.3, self.tendon_count + 0.7)
        self.length_ax.set_ylim(lower_mm - margin_mm, upper_mm + margin_mm)
        self.length_ax.set_xticks(indices, [str(index) for index in range(1, self.tendon_count + 1)])
        self.length_ax.set_ylabel("tendon delta [mm]")
        self.length_ax.set_title("Commanded vs actual tendon length")
        self.length_ax.grid(True, axis="y", alpha=0.25)
        self.length_ax.legend(loc="upper right", fontsize=8)

    def _draw_force_chart(self, view_data: MujocoTendonDebugViewData) -> None:
        self.force_ax.cla()
        indices = np.arange(1, self.tendon_count + 1, dtype=float)
        self.force_ax.bar(
            indices,
            view_data.actuator_force,
            width=0.56,
            color="tab:red",
            alpha=0.75,
        )
        lower = min(0.0, float(np.min(view_data.actuator_force)))
        upper = max(float(np.max(view_data.actuator_force)), 0.0)
        reference = max(abs(lower), abs(upper), 1.0e-3)
        margin = max(0.01, 0.15 * reference)
        self.force_ax.set_xlim(0.3, self.tendon_count + 0.7)
        self.force_ax.set_ylim(lower - margin, upper + margin)
        self.force_ax.set_xticks(indices, [str(index) for index in range(1, self.tendon_count + 1)])
        self.force_ax.set_ylabel("actuator force [N]")
        self.force_ax.set_title("Actual tendon pull force")
        self.force_ax.grid(True, axis="y", alpha=0.25)
        self.force_ax.axhline(self.force_limit[0], color="0.35", linewidth=0.9, linestyle="--")
        self.force_ax.axhline(self.force_limit[1], color="0.35", linewidth=0.9, linestyle="--")

    def _sync_command_sliders(self) -> None:
        self._updating_controls = True
        try:
            for slider, value in zip(
                self.command_sliders,
                self.commanded_tendon_delta,
                strict=True,
            ):
                slider.set_val(float(value))
        finally:
            self._updating_controls = False

    def _sync_base_sliders(self) -> None:
        self._updating_controls = True
        try:
            for slider, value in zip(
                self.base_sliders,
                self.base_command_rpy,
                strict=True,
            ):
                slider.set_val(float(value))
        finally:
            self._updating_controls = False

    def _combined_control(self) -> np.ndarray:
        if self.config.mobile_base_config_path is None:
            return self.commanded_tendon_delta
        return np.concatenate((self.commanded_tendon_delta, self.base_command_rpy))

    def _notify_state_updated(self) -> None:
        if self.state_update_callback is not None:
            self.state_update_callback(self.state)

    def _on_timer(self) -> bool:
        if self._running:
            self.step()
        return True

    def show(self) -> None:
        """Start the matplotlib event loop."""

        plt.show()

    def close(self) -> None:
        """Close the viewer and release timer resources."""

        self.pause()
        plt.close(self.fig)


def _as_tendon_vector(
    values: np.ndarray,
    name: str,
    *,
    expected_size: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(
            f"Expected {name} with shape ({expected_size},), got {array.shape}."
        )
    return array


def _select_tendon_vector(
    values: np.ndarray,
    name: str,
    *,
    expected_size: int,
    tendon_indices: np.ndarray | None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if tendon_indices is not None and array.shape[0] > expected_size:
        indices = np.asarray(tendon_indices, dtype=int)
        if indices.shape != (expected_size,):
            raise ValueError(
                f"Expected tendon_indices with shape ({expected_size},), got {indices.shape}."
            )
        array = array[indices]
    return _as_tendon_vector(array, name, expected_size=expected_size)


def _format_debug_info_text(view_data: MujocoTendonDebugViewData) -> str:
    tip = view_data.tip_position
    lines = [
        f"time_s: {view_data.time_s: .4f}",
        "tip_position [m]",
        f"  x={tip[0]: .5f}  y={tip[1]: .5f}  z={tip[2]: .5f}",
        "",
        "tendon error [mm]",
        *_format_vector_rows(view_data.tendon_error, scale=1000.0),
        "",
        "actuator force [N]",
        *_format_vector_rows(view_data.actuator_force, scale=1.0),
        "",
        "q_est [kx ky eps] x3",
        *_format_q_est_rows(view_data.q_est),
    ]
    return "\n".join(lines)


def _format_vector_rows(values: np.ndarray, *, scale: float) -> list[str]:
    scaled = np.asarray(values, dtype=float) * scale
    rows = []
    for start in range(0, scaled.size, 3):
        chunk = scaled[start : start + 3]
        if chunk.size < 3:
            break
        rows.append(
            "  {}-{}: [{: .3f}, {: .3f}, {: .3f}]".format(
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


def _format_q_est_rows(values: np.ndarray) -> list[str]:
    q = np.asarray(values, dtype=float)
    if q.shape == (9,):
        return _format_segment_rows(q)
    if q.shape == (2, 9):
        rows: list[str] = []
        for arm_name, arm_q in zip(("executor", "observer"), q, strict=True):
            rows.append(f"  {arm_name}")
            rows.extend(f"  {row}" for row in _format_segment_rows(arm_q))
        return rows
    return _format_vector_rows(q.reshape(-1), scale=1.0)


def _estimate_q_for_tendon_view(
    tendon_delta: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
) -> np.ndarray:
    if len(physical_tendons) > 9 and len(physical_tendons) % 2 == 0 and tendon_delta.shape == (len(physical_tendons),):
        tendons_per_arm = len(physical_tendons) // 2
        return np.vstack(
            (
                physical_tendon_delta_to_q(
                    tendon_delta[:tendons_per_arm],
                    params,
                    _local_tendons(physical_tendons[:tendons_per_arm]),
                ),
                physical_tendon_delta_to_q(
                    tendon_delta[tendons_per_arm:],
                    params,
                    _local_tendons(physical_tendons[tendons_per_arm:]),
                ),
            )
        )
    return physical_tendon_delta_to_q(tendon_delta, params, physical_tendons)


def _local_tendons(
    tendons: tuple[PhysicalTendonPath, ...],
) -> tuple[PhysicalTendonPath, ...]:
    return tuple(
        PhysicalTendonPath(
            id=tendon.id,
            global_index=index,
            motor_index=index,
            anchor_segment_index=tendon.anchor_segment_index,
            angle_deg=tendon.angle_deg,
            radial_offset=tendon.radial_offset,
            path_segment_indices=tendon.path_segment_indices,
            hole_index=tendon.hole_index,
        )
        for index, tendon in enumerate(tendons)
    )


def _tendon_slider_label(index: int, *, tendons_per_column: int) -> str:
    if index < tendons_per_column:
        return f"exec_{index + 1:02d} [m]"
    return f"obs_{index - tendons_per_column + 1:02d} [m]"


__all__ = [
    "MujocoTendonDebugViewData",
    "MujocoTendonDebugViewer",
    "MujocoTendonMonitorPanel",
    "TENDON_DEBUG_NAMED_COMMANDS",
    "clip_tendon_command",
    "compute_mujoco_tendon_debug_view_data",
    "named_tendon_command",
]
