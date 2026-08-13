"""Interactive curvature and tendon controls for the MuJoCo system."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from threading import RLock
from time import perf_counter
import traceback

import numpy as np

from continuum_sim.backends.mujoco_system_backend import MujocoSystemBackend
from continuum_sim.model.base_pose import quaternion_wxyz_to_rotation_matrix
from continuum_sim.model.mount_frame import load_mobile_base_mount_config
from continuum_sim.runtime.concurrency import (
    LatestValueSlot,
    MonotonicRateRunner,
    TimeRateGate,
)
from continuum_sim.system.types import (
    ArmTendonRateCommand,
    RobotSystemCommand,
    RobotSystemState,
)
from continuum_sim.visualization.system_tendon_debug import (
    SystemStatusPanel,
    SystemTendonMonitorPanel,
)


CURVATURE_MIN_1_PER_M = -30.0
CURVATURE_MAX_1_PER_M = 30.0


class _ManualControlPanel:
    """Own only the widgets for one manual-control mode."""

    def __init__(self, *, title: str) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self.fig = plt.figure(figsize=(14.0, 8.0))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(title)

    def show(self, *, block: bool = True) -> None:
        self._plt.show(block=block)

    def is_open(self) -> bool:
        return bool(self._plt.fignum_exists(self.fig.number))

    def close(self) -> None:
        self._plt.close(self.fig)


def target_rates(
    target_m: np.ndarray,
    current_target_m: np.ndarray,
    max_rate_mps: np.ndarray,
    *,
    dt: float,
) -> np.ndarray:
    """Return rates that approach an absolute target in one step, with clipping."""

    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    target = np.asarray(target_m, dtype=float)
    current = np.asarray(current_target_m, dtype=float)
    limit = np.asarray(max_rate_mps, dtype=float)
    if target.ndim != 1 or current.shape != target.shape or limit.shape != target.shape:
        raise ValueError("Target, current target, and max rate must be matching 1D arrays.")
    if np.any(limit <= 0.0):
        raise ValueError("max_rate_mps must be positive.")
    return np.clip((target - current) / float(dt), -limit, limit)


def normalize_target_mm(
    value: object,
    minimum_mm: float,
    maximum_mm: float,
    fallback_mm: float,
) -> float:
    """Parse and clip one finite tendon target expressed in millimetres."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback_mm)
    if not np.isfinite(parsed):
        parsed = float(fallback_mm)
    return float(np.clip(parsed, minimum_mm, maximum_mm))


def normalize_curvature_target(value: object, fallback: float) -> float:
    """Parse and clip one finite curvature target expressed in 1/m."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if not np.isfinite(parsed):
        parsed = float(fallback)
    return float(
        np.clip(parsed, CURVATURE_MIN_1_PER_M, CURVATURE_MAX_1_PER_M)
    )


def named_system_target(
    name: str,
    template: Mapping[str, np.ndarray],
    *,
    single_pull_m: float,
    triplet_pull_m: float,
) -> dict[str, np.ndarray]:
    """Build one of the baseline named tendon targets using named arms."""

    result = {
        arm_name: np.zeros_like(np.asarray(values, dtype=float))
        for arm_name, values in template.items()
    }
    if name == "zero":
        return result
    if name == "all_tendons_pull":
        for values in result.values():
            values[:] = triplet_pull_m
        return result

    suffixes = {
        "_tendon_1_pull": (slice(0, 1), single_pull_m),
        "_segment_1_triplet": (slice(0, 3), triplet_pull_m),
    }
    for suffix, (indices, value) in suffixes.items():
        if name.endswith(suffix):
            arm_name = name[: -len(suffix)]
            if arm_name not in result:
                raise ValueError(f"Named target references unavailable arm {arm_name!r}.")
            if result[arm_name].size < indices.stop:
                raise ValueError(f"Arm {arm_name!r} has fewer than {indices.stop} tendons.")
            result[arm_name][indices] = value
            return result
    raise ValueError(f"Unknown named tendon target {name!r}.")


def available_named_targets(arm_names: tuple[str, ...]) -> tuple[str, ...]:
    commands = ["zero"]
    for arm_name in arm_names:
        commands.extend(
            (
                f"{arm_name}_tendon_1_pull",
                f"{arm_name}_segment_1_triplet",
            )
        )
    commands.append("all_tendons_pull")
    return tuple(commands)


def bounded_compatible_target(
    current_m: np.ndarray,
    candidate_m: np.ndarray,
    lower_m: np.ndarray,
    upper_m: np.ndarray,
) -> np.ndarray:
    """Move toward a compatible candidate without breaking tendon bounds."""

    current = np.asarray(current_m, dtype=float)
    candidate = np.asarray(candidate_m, dtype=float)
    lower = np.asarray(lower_m, dtype=float)
    upper = np.asarray(upper_m, dtype=float)
    if not (
        current.shape == candidate.shape == lower.shape == upper.shape
        and current.ndim == 1
    ):
        raise ValueError("Compatible target arrays must be matching 1D vectors.")
    current = np.clip(current, lower, upper)
    delta = candidate - current
    scale = 1.0
    for value, change, minimum, maximum in zip(
        current, delta, lower, upper, strict=True
    ):
        if change > 0.0:
            scale = min(scale, float((maximum - value) / change))
        elif change < 0.0:
            scale = min(scale, float((minimum - value) / change))
    return current + np.clip(scale, 0.0, 1.0) * delta


class MujocoSystemDebugViewer:
    """Mode-specific Matplotlib controls for a composed MuJoCo system backend."""

    def __init__(
        self,
        backend: MujocoSystemBackend,
        *,
        control_dt_s: float,
        n_substeps: int,
        state_update_callback: Callable[[RobotSystemState], None] | None = None,
        diagnostic_text_provider: (
            Callable[[RobotSystemState, str], str] | None
        ) = None,
        control_mode: str = "curvature",
        panel_fps: float = 15.0,
        status_fps: float = 5.0,
        show_tendon_monitor: bool = False,
        simulation_lock=None,
        runtime_timing=None,
        clock=perf_counter,
    ) -> None:
        from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

        if control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive.")
        if n_substeps <= 0:
            raise ValueError("n_substeps must be positive.")
        if len(backend.assembly.enabled_arms) > 2:
            raise ValueError("The debug viewer supports at most two enabled arms.")
        if control_mode not in {"curvature", "tendon"}:
            raise ValueError("control_mode must be 'curvature' or 'tendon'.")
        if not np.isfinite(panel_fps) or panel_fps <= 0.0:
            raise ValueError("panel_fps must be positive and finite.")
        if not np.isfinite(status_fps) or status_fps <= 0.0:
            raise ValueError("status_fps must be positive and finite.")

        self.backend = backend
        self.control_dt_s = float(control_dt_s)
        self.n_substeps = int(n_substeps)
        self.state_update_callback = state_update_callback
        self.diagnostic_text_provider = diagnostic_text_provider
        self.control_mode = str(control_mode)
        self.panel_fps = float(panel_fps)
        self.status_fps = float(status_fps)
        self.show_tendon_monitor = bool(show_tendon_monitor)
        self._clock = clock
        self._simulation_lock = RLock() if simulation_lock is None else simulation_lock
        self._target_lock = RLock()
        self.runtime_timing = runtime_timing
        if runtime_timing is not None:
            backend.set_runtime_timing(runtime_timing)
        self.state = backend.reset_system()
        self.arm_names = tuple(self.state.arms)
        self.targets = {
            name: np.asarray(arm.tendon_target_m, dtype=float).copy()
            for name, arm in self.state.arms.items()
        }
        self._target_limits = {
            arm.name: (
                np.asarray(
                    arm.spatial_arm.limits.tendon_displacement_min_m,
                    dtype=float,
                ).copy(),
                np.asarray(
                    arm.spatial_arm.limits.tendon_displacement_max_m,
                    dtype=float,
                ).copy(),
            )
            for arm in backend.assembly.enabled_arms
        }
        self._target_slot = LatestValueSlot(_copy_targets(self.targets))
        self._state_slot = LatestValueSlot(self.state)
        self._running = False
        self._updating_controls = False
        self._views_dirty = False
        self._dirty_target_arms: set[str] = set()
        self._dirty_curvature_components: dict[str, set[int]] = {}
        self._base_controls_dirty = False
        self._worker_error: BaseException | None = None
        self._worker_error_reported = False
        self.control_space = (
            "bending_compatible"
            if self.control_mode == "curvature"
            else "raw_tendon_debug"
        )
        self.panel = _ManualControlPanel(
            title=(
                "continuum_sim curvature control"
                if self.control_mode == "curvature"
                else "continuum_sim tendon control"
            )
        )
        self.status_panel = SystemStatusPanel(
            title="continuum_sim manual control status"
        )
        self.tendon_panel = (
            SystemTendonMonitorPanel(
                title="continuum_sim tendon length and force",
                show_info=False,
            )
            if self.show_tendon_monitor
            else None
        )
        self._status_gate = TimeRateGate(
            1.0 / self.status_fps,
            clock=self._clock,
        )

        self.sliders: dict[str, list[Slider]] = {}
        self.target_inputs: dict[str, list[TextBox]] = {}
        if self.control_mode == "tendon":
            for arm_index, arm in enumerate(backend.assembly.enabled_arms):
                sliders, target_inputs = self._build_arm_controls(
                    Slider,
                    TextBox,
                    arm_index,
                    arm.name,
                    arm.spatial_arm.limits.tendon_displacement_min_m,
                    arm.spatial_arm.limits.tendon_displacement_max_m,
                )
                self.sliders[arm.name] = sliders
                self.target_inputs[arm.name] = target_inputs

        self.curvature_sliders: dict[str, list[Slider]] = {}
        self.curvature_inputs: dict[str, list[TextBox]] = {}
        if self.control_mode == "curvature":
            for arm_index, arm_name in enumerate(self.arm_names):
                sliders, inputs = self._build_curvature_controls(
                    Slider,
                    TextBox,
                    arm_index,
                    arm_name,
                )
                self.curvature_sliders[arm_name] = sliders
                self.curvature_inputs[arm_name] = inputs

        self._configure_base_controls()
        self.base_buttons, self.base_target_inputs = self._build_base_controls(
            Button,
            TextBox,
        )

        self.reset_button = Button(
            self.panel.fig.add_axes((0.04, 0.045, 0.08, 0.04)),
            "Reset",
            **_disable_widget_blit(Button),
        )
        self.zero_button = Button(
            self.panel.fig.add_axes((0.13, 0.045, 0.08, 0.04)),
            "Zero arms",
            **_disable_widget_blit(Button),
        )
        self.zero_base_button = Button(
            self.panel.fig.add_axes((0.22, 0.045, 0.08, 0.04)),
            "Zero base",
            **_disable_widget_blit(Button),
        )
        self.step_button = Button(
            self.panel.fig.add_axes((0.31, 0.045, 0.08, 0.04)),
            "Step",
            **_disable_widget_blit(Button),
        )
        self.run_button = Button(
            self.panel.fig.add_axes((0.40, 0.045, 0.08, 0.04)),
            "Run",
            **_disable_widget_blit(Button),
        )
        self.named_targets = available_named_targets(self.arm_names)
        self.radio = None
        if self.control_mode == "tendon":
            self.radio = RadioButtons(
                self.panel.fig.add_axes((0.79, 0.025, 0.17, 0.12)),
                self.named_targets,
                active=0,
                **_disable_widget_blit(RadioButtons),
            )
        self._control_worker = MonotonicRateRunner(
            self.control_dt_s,
            self.step,
            self._on_control_worker_error,
            name="continuum-sim-control",
        )
        self.timer = self.panel.fig.canvas.new_timer(
            interval=max(1, round(1000.0 / self.panel_fps))
        )
        self.timer.add_callback(self._on_timer)
        self._closed = False
        self._close_cid = self.panel.fig.canvas.mpl_connect(
            "close_event",
            self._on_control_window_closed,
        )
        self._connect_controls()
        self._update_status(redraw=False)
        if self.tendon_panel is not None:
            self.tendon_panel.update(self.state, redraw=False)
        self._notify_state_updated()

    def _build_arm_controls(
        self,
        slider_type,
        text_box_type,
        arm_index: int,
        arm_name: str,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[list, list]:
        sliders = []
        target_inputs = []
        x0 = 0.04 + 0.31 * arm_index
        y0 = 0.44
        role = "MAIN / EXECUTOR" if arm_name == "executor" else "OBSERVER"
        self.panel.fig.text(x0, y0 + 0.065, role, fontsize=10, weight="bold")
        self.panel.fig.text(x0, y0 + 0.043, "tendon target [mm]", fontsize=8)
        for tendon_index, (minimum, maximum) in enumerate(
            zip(lower, upper, strict=True)
        ):
            y = y0 - 0.040 * tendon_index
            slider_axis = self.panel.fig.add_axes(
                (x0, y, 0.205, 0.018)
            )
            slider = slider_type(
                ax=slider_axis,
                label=f"S{tendon_index // 3 + 1}-T{tendon_index % 3 + 1}",
                valmin=1000.0 * float(minimum),
                valmax=1000.0 * float(maximum),
                valinit=0.0,
                valfmt="% .3f",
                **_disable_widget_blit(slider_type),
            )
            slider.valtext.set_visible(False)
            input_axis = self.panel.fig.add_axes(
                (x0 + 0.22, y - 0.003, 0.065, 0.025)
            )
            target_input = text_box_type(
                input_axis,
                "",
                initial="0.000",
                textalignment="center",
                **_disable_widget_blit(text_box_type),
            )
            _disconnect_textbox_resize(target_input)
            sliders.append(slider)
            target_inputs.append(target_input)
        return sliders, target_inputs

    def _build_curvature_controls(
        self,
        slider_type,
        text_box_type,
        arm_index: int,
        arm_name: str,
    ) -> tuple[list, list]:
        sliders = []
        inputs = []
        x0 = 0.04 + 0.31 * arm_index
        role = "MAIN / EXECUTOR" if arm_name == "executor" else "OBSERVER"
        self.panel.fig.text(x0, 0.625, role, fontsize=10, weight="bold")
        self.panel.fig.text(x0, 0.603, "segment-local curvature [1/m]", fontsize=8)
        model = self.backend.layout.bending_models[arm_name]
        bending = model.estimate(model.project(self.targets[arm_name]))
        for component_index, value in enumerate(bending):
            segment_index = component_index // 2
            component_name = "kx" if component_index % 2 == 0 else "ky"
            y = 0.555 - 0.050 * component_index
            slider = slider_type(
                ax=self.panel.fig.add_axes((x0, y, 0.205, 0.020)),
                label=f"S{segment_index + 1}-{component_name}",
                valmin=CURVATURE_MIN_1_PER_M,
                valmax=CURVATURE_MAX_1_PER_M,
                valinit=float(value),
                valfmt="% .3f",
                **_disable_widget_blit(slider_type),
            )
            slider.valtext.set_visible(False)
            slider.drawon = False
            target_input = text_box_type(
                self.panel.fig.add_axes((x0 + 0.22, y - 0.003, 0.065, 0.026)),
                "",
                initial=_format_curvature_target(value),
                textalignment="center",
                **_disable_widget_blit(text_box_type),
            )
            _disconnect_textbox_resize(target_input)
            sliders.append(slider)
            inputs.append(target_input)
        return sliders, inputs

    def _configure_base_controls(self) -> None:
        self.base_control_enabled = self.backend.assembly.base.control_mode != "fixed"
        self.base_target_pose_rpy = _pose_xyz_rpy(self.state.base.pose)
        self.base_initial_pose_rpy = self.base_target_pose_rpy.copy()
        assembly_base = self.backend.assembly.base
        position_lower = np.asarray(assembly_base.position_min_m, dtype=float)
        position_upper = np.asarray(assembly_base.position_max_m, dtype=float)
        rpy_lower = np.full(3, -np.pi, dtype=float)
        rpy_upper = np.full(3, np.pi, dtype=float)
        self.base_translation_step_m = 0.01
        self.base_fine_translation_step_m = 0.002
        self.base_rotation_step_rad = np.deg2rad(2.0)
        self.base_fine_rotation_step_rad = np.deg2rad(0.5)
        path = self.backend.config.mobile_base_config_path
        if path is not None:
            mobile = load_mobile_base_mount_config(path).mobile_base
            position_lower = np.maximum(position_lower, mobile.limits.position_min_m)
            position_upper = np.minimum(position_upper, mobile.limits.position_max_m)
            rpy_lower = np.deg2rad(mobile.limits.rpy_min_deg)
            rpy_upper = np.deg2rad(mobile.limits.rpy_max_deg)
            manual = mobile.manual_control
            self.base_translation_step_m = float(manual.translation_step_m)
            self.base_fine_translation_step_m = float(
                manual.fine_translation_step_m
            )
            self.base_rotation_step_rad = np.deg2rad(manual.rotation_step_deg)
            self.base_fine_rotation_step_rad = np.deg2rad(
                manual.fine_rotation_step_deg
            )
        self.base_lower_pose_rpy = np.concatenate((position_lower, rpy_lower))
        self.base_upper_pose_rpy = np.concatenate((position_upper, rpy_upper))
        self.base_fine_mode = False

    def _build_base_controls(self, button_type, text_box_type):
        labels = ("X [m]", "Y [m]", "Z [m]", "Roll [deg]", "Pitch [deg]", "Yaw [deg]")
        self.panel.fig.text(0.68, 0.625, "BASE 6-DOF (world frame)", fontsize=10, weight="bold")
        self.base_step_button = button_type(
            self.panel.fig.add_axes((0.885, 0.605, 0.075, 0.032)),
            "coarse",
            **_disable_widget_blit(button_type),
        )
        self.base_step_button.on_clicked(lambda _event: self._toggle_base_step())
        buttons = []
        inputs = []
        for index, label in enumerate(labels):
            y = 0.555 - 0.042 * index
            self.panel.fig.text(0.68, y + 0.006, label, fontsize=8)
            minus = button_type(
                self.panel.fig.add_axes((0.755, y, 0.040, 0.028)),
                "-",
                **_disable_widget_blit(button_type),
            )
            target_input = text_box_type(
                self.panel.fig.add_axes((0.802, y, 0.090, 0.028)),
                "",
                initial=_format_base_target(self.base_target_pose_rpy[index], index),
                textalignment="center",
                **_disable_widget_blit(text_box_type),
            )
            _disconnect_textbox_resize(target_input)
            plus = button_type(
                self.panel.fig.add_axes((0.900, y, 0.040, 0.028)),
                "+",
                **_disable_widget_blit(button_type),
            )
            minus.on_clicked(
                lambda _event, axis=index: self.adjust_base_target(axis, -1.0)
            )
            plus.on_clicked(
                lambda _event, axis=index: self.adjust_base_target(axis, 1.0)
            )
            target_input.on_submit(
                lambda text, axis=index: self._on_base_target_input(axis, text)
            )
            buttons.append((minus, plus))
            inputs.append(target_input)
        return buttons, inputs

    def _toggle_base_step(self) -> None:
        self.base_fine_mode = not self.base_fine_mode
        self.base_step_button.label.set_text(
            "fine" if self.base_fine_mode else "coarse"
        )

    def adjust_base_target(self, component_index: int, direction: float) -> np.ndarray:
        if component_index not in range(6):
            raise ValueError("Base component index must be in 0..5.")
        if direction not in (-1.0, 1.0):
            raise ValueError("direction must be -1 or 1.")
        if not self.base_control_enabled:
            return self.base_target_pose_rpy.copy()
        if component_index < 3:
            step = (
                self.base_fine_translation_step_m
                if self.base_fine_mode
                else self.base_translation_step_m
            )
        else:
            step = (
                self.base_fine_rotation_step_rad
                if self.base_fine_mode
                else self.base_rotation_step_rad
            )
        target = self.base_target_pose_rpy.copy()
        target[component_index] += direction * step
        self.set_base_target(target)
        return self.base_target_pose_rpy.copy()

    def _on_base_target_input(self, component_index: int, text: str) -> None:
        if self._updating_controls or not self.base_control_enabled:
            return
        try:
            value = float(text)
        except (TypeError, ValueError):
            value = self.base_target_pose_rpy[component_index]
        if component_index >= 3:
            value = np.deg2rad(value)
        target = self.base_target_pose_rpy.copy()
        target[component_index] = value
        self.set_base_target(target)

    def set_base_target(self, target_pose_rpy: np.ndarray) -> None:
        values = np.asarray(target_pose_rpy, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("Base target must be one finite xyz-rpy vector.")
        with self._target_lock:
            clipped = np.clip(
                values,
                self.base_lower_pose_rpy,
                self.base_upper_pose_rpy,
            )
            if np.array_equal(clipped, self.base_target_pose_rpy):
                return
            self.base_target_pose_rpy = clipped
            self._base_controls_dirty = True
            self._views_dirty = True

    def _sync_base_controls(self) -> None:
        with self._target_lock:
            if not self._base_controls_dirty:
                return
            values = self.base_target_pose_rpy.copy()
            self._base_controls_dirty = False
        updates = []
        for index, target_input in enumerate(self.base_target_inputs):
            formatted = _format_base_target(values[index], index)
            if target_input.text != formatted:
                updates.append((target_input, formatted))
        self._apply_widget_updates(updates)

    def _sync_curvature_controls(self) -> None:
        with self._target_lock:
            dirty_components = {
                arm_name: tuple(sorted(component_indices))
                for arm_name, component_indices in (
                    self._dirty_curvature_components.items()
                )
                if component_indices
            }
            if not dirty_components:
                return
            values_by_arm = {
                arm_name: self.backend.layout.bending_models[arm_name].estimate(
                    self.backend.layout.bending_models[arm_name].project(
                        self.targets[arm_name]
                    )
                )
                for arm_name in dirty_components
            }
            self._dirty_curvature_components.clear()
        updates = []
        for arm_name, values in values_by_arm.items():
            for component_index in dirty_components[arm_name]:
                slider = self.curvature_sliders[arm_name][component_index]
                target_input = self.curvature_inputs[arm_name][component_index]
                value = values[component_index]
                value = float(value)
                if float(slider.val) != value:
                    updates.append((slider, value))
                formatted = _format_curvature_target(value)
                if target_input.text != formatted:
                    updates.append((target_input, formatted))
        self._apply_widget_updates(updates)

    def zero_base_target(self) -> None:
        self.set_base_target(self.base_initial_pose_rpy)

    def _connect_controls(self) -> None:
        for arm_name, sliders in self.sliders.items():
            for tendon_index, (slider, target_input) in enumerate(
                zip(sliders, self.target_inputs[arm_name], strict=True)
            ):
                slider.on_changed(
                    lambda value, name=arm_name, index=tendon_index: (
                        self._on_slider(name, index, value)
                    )
                )
                target_input.on_submit(
                    lambda text, name=arm_name, index=tendon_index: (
                        self._on_target_input(name, index, text)
                    )
                )
        for arm_name, sliders in self.curvature_sliders.items():
            for component_index, (slider, target_input) in enumerate(
                zip(sliders, self.curvature_inputs[arm_name], strict=True)
            ):
                slider.on_changed(
                    lambda value, name=arm_name, index=component_index: (
                        self._on_curvature_slider(name, index, value)
                    )
                )
                target_input.on_submit(
                    lambda text, name=arm_name, index=component_index: (
                        self._on_curvature_input(name, index, text)
                    )
                )
        self.reset_button.on_clicked(lambda _event: self.reset())
        self.zero_button.on_clicked(lambda _event: self.zero_targets())
        self.zero_base_button.on_clicked(lambda _event: self.zero_base_target())
        self.step_button.on_clicked(lambda _event: self._step_once())
        self.run_button.on_clicked(lambda _event: self.toggle_run())
        if self.radio is not None:
            self.radio.on_clicked(self.apply_named_target)

    def _on_curvature_slider(
        self,
        arm_name: str,
        component_index: int,
        value_1_per_m: float,
    ) -> None:
        if self._updating_controls:
            return
        self.set_curvature_component(
            arm_name,
            component_index,
            value_1_per_m,
            input_source="slider",
        )

    def _on_curvature_input(
        self,
        arm_name: str,
        component_index: int,
        text: str,
    ) -> None:
        if self._updating_controls:
            return
        with self._target_lock:
            current = self.backend.layout.bending_models[arm_name].estimate(
                self.backend.layout.bending_models[arm_name].project(
                    self.targets[arm_name]
                )
            )
            value = normalize_curvature_target(text, current[component_index])
        self.set_curvature_component(
            arm_name,
            component_index,
            value,
            input_source="text",
        )

    def set_curvature_component(
        self,
        arm_name: str,
        component_index: int,
        value_1_per_m: float,
        *,
        input_source: str = "api",
    ) -> np.ndarray:
        """Set one absolute segment-local kx/ky target and publish tendons."""

        if arm_name not in self.targets:
            raise KeyError(f"Unknown arm {arm_name!r}.")
        if component_index not in range(6):
            raise ValueError("component_index must be in 0..5.")
        timing = getattr(self, "runtime_timing", None)
        with (
            nullcontext()
            if timing is None
            else timing.measure("input.callback")
        ):
            with self._target_lock:
                if timing is not None:
                    segment_index = component_index // 2
                    component = "kx" if component_index % 2 == 0 else "ky"
                    timing.mark_input(
                        f"{arm_name}:S{segment_index + 1}:{component}:{input_source}"
                    )
                model = self.backend.layout.bending_models[arm_name]
                current = model.project(self.targets[arm_name])
                bending = model.estimate(current)
                value = normalize_curvature_target(
                    value_1_per_m,
                    bending[component_index],
                )
                bending[component_index] = value
                candidate = model.to_tendon(bending)
                arm = next(
                    item
                    for item in self.backend.assembly.enabled_arms
                    if item.name == arm_name
                )
                bounded = bounded_compatible_target(
                    current,
                    candidate,
                    arm.spatial_arm.limits.tendon_displacement_min_m,
                    arm.spatial_arm.limits.tendon_displacement_max_m,
                )
                self.set_targets({arm_name: bounded})
                self._dirty_curvature_components.setdefault(arm_name, set()).add(
                    component_index
                )
                self._views_dirty = True
                return model.estimate(bounded)

    def _on_slider(self, arm_name: str, tendon_index: int, value_mm: float) -> None:
        if self._updating_controls:
            return
        timing = getattr(self, "runtime_timing", None)
        with (
            nullcontext()
            if timing is None
            else timing.measure("input.callback")
        ):
            with self._target_lock:
                if timing is not None:
                    timing.mark_input(f"{arm_name}:T{tendon_index + 1}:slider")
                self.targets[arm_name][tendon_index] = 0.001 * float(value_mm)
                self._dirty_target_arms.add(arm_name)
                self._views_dirty = True
                self._project_targets_if_compatible(arm_name)
                self._publish_targets_locked()

    def _on_target_input(
        self,
        arm_name: str,
        tendon_index: int,
        text: str,
    ) -> None:
        if self._updating_controls:
            return
        slider = self.sliders[arm_name][tendon_index]
        timing = getattr(self, "runtime_timing", None)
        with (
            nullcontext()
            if timing is None
            else timing.measure("input.callback")
        ):
            with self._target_lock:
                if timing is not None:
                    timing.mark_input(f"{arm_name}:T{tendon_index + 1}:text")
                current_mm = 1000.0 * self.targets[arm_name][tendon_index]
                value_mm = normalize_target_mm(
                    text,
                    float(slider.valmin),
                    float(slider.valmax),
                    current_mm,
                )
                self.targets[arm_name][tendon_index] = 0.001 * value_mm
                self._dirty_target_arms.add(arm_name)
                self._views_dirty = True
                self._project_targets_if_compatible(arm_name)
                self._publish_targets_locked()

    def _project_targets_if_compatible(self, arm_name: str) -> None:
        if self.control_space != "bending_compatible":
            return
        projected = self.backend.layout.bending_models[arm_name].project(
            self.targets[arm_name]
        )
        self.set_targets({arm_name: projected})

    def reset(self) -> RobotSystemState:
        self.pause()
        with self._simulation_lock:
            state = self.backend.reset_system()
        self.state = state
        self._state_slot.publish(state)
        with self._target_lock:
            self.base_target_pose_rpy = _pose_xyz_rpy(self.state.base.pose)
            self.base_initial_pose_rpy = self.base_target_pose_rpy.copy()
            self._base_controls_dirty = True
            self._views_dirty = True
        self.set_targets(
            {name: np.zeros_like(values) for name, values in self.targets.items()}
        )
        return state

    def zero_targets(self) -> None:
        self.set_targets(
            {name: np.zeros_like(values) for name, values in self.targets.items()}
        )

    def set_targets(self, targets: Mapping[str, np.ndarray]) -> None:
        unknown = set(targets).difference(self.targets)
        if unknown:
            raise ValueError(f"Unknown target arm names: {sorted(unknown)}.")
        with self._target_lock:
            changed = False
            for arm_name, values in targets.items():
                array = np.asarray(values, dtype=float)
                if array.shape != self.targets[arm_name].shape:
                    raise ValueError(
                        f"Target for arm {arm_name!r} has shape {array.shape}, "
                        f"expected {self.targets[arm_name].shape}."
                    )
                lower, upper = self._target_limits_for(arm_name)
                normalized = np.clip(array, lower, upper)
                if np.array_equal(normalized, self.targets[arm_name]):
                    continue
                previous = self.targets[arm_name]
                self.targets[arm_name] = normalized
                self._dirty_target_arms.add(arm_name)
                if getattr(self, "control_mode", "tendon") == "curvature":
                    model = self.backend.layout.bending_models[arm_name]
                    previous_bending = model.estimate(model.project(previous))
                    normalized_bending = model.estimate(model.project(normalized))
                    changed_components = np.flatnonzero(
                        ~np.isclose(
                            previous_bending,
                            normalized_bending,
                            rtol=1.0e-12,
                            atol=1.0e-12,
                        )
                    )
                    self._dirty_curvature_components.setdefault(
                        arm_name,
                        set(),
                    ).update(int(index) for index in changed_components)
                self._views_dirty = True
                changed = True
            if changed:
                self._publish_targets_locked()

    def _publish_targets_locked(self) -> None:
        self._target_slot.publish(_copy_targets(self.targets))

    def _target_limits_for(self, arm_name: str) -> tuple[np.ndarray, np.ndarray]:
        limits = getattr(self, "_target_limits", None)
        if limits is not None:
            return limits[arm_name]
        sliders = self.sliders[arm_name]
        return (
            0.001 * np.asarray([slider.valmin for slider in sliders], dtype=float),
            0.001 * np.asarray([slider.valmax for slider in sliders], dtype=float),
        )

    def _sync_target_controls(self) -> None:
        with self._target_lock:
            dirty_arms = tuple(self._dirty_target_arms)
            target_values = {
                arm_name: self.targets[arm_name].copy()
                for arm_name in dirty_arms
            }
            self._dirty_target_arms.clear()
        updates = []
        for arm_name, values in target_values.items():
            if arm_name not in self.sliders:
                continue
            for slider, target_input, value_m in zip(
                self.sliders[arm_name],
                self.target_inputs[arm_name],
                values,
                strict=True,
            ):
                value_mm = 1000.0 * float(value_m)
                if float(slider.val) != value_mm:
                    updates.append((slider, value_mm))
                formatted = _format_target_mm(value_mm)
                if target_input.text != formatted:
                    updates.append((target_input, formatted))
        self._apply_widget_updates(updates)

    def _apply_widget_updates(self, updates: list[tuple[object, object]]) -> None:
        if not updates:
            return
        saved_states = []
        self._updating_controls = True
        try:
            for widget, _value in updates:
                drawon = getattr(widget, "drawon", None)
                eventson = getattr(widget, "eventson", None)
                saved_states.append((widget, drawon, eventson))
                if drawon is not None:
                    widget.drawon = False
                if eventson is not None:
                    widget.eventson = False
            for widget, value in updates:
                widget.set_val(value)
        finally:
            for widget, drawon, eventson in saved_states:
                if drawon is not None:
                    widget.drawon = drawon
                if eventson is not None:
                    widget.eventson = eventson
            self._updating_controls = False

    def apply_named_target(self, name: str) -> RobotSystemState:
        smoke = self.backend.config.smoke_tests
        self.set_targets(
            named_system_target(
                name,
                self.targets,
                single_pull_m=float(smoke.single_tendon_delta_m),
                triplet_pull_m=float(smoke.symmetric_tendon_delta_m),
            )
        )
        if not self._running:
            return self.step()
        return self._state_slot.snapshot()[0]

    def _step_once(self) -> RobotSystemState:
        if self._running:
            return self._state_slot.snapshot()[0]
        return self.step()

    def step(self) -> RobotSystemState:
        timing = self.runtime_timing
        try:
            if timing is not None:
                timing.start_cycle()
            state = self._state_slot.snapshot()[0]
            targets = self._target_slot.snapshot()[0]
            with self._target_lock:
                control_space = self.control_space
                base_target_pose_rpy = self.base_target_pose_rpy.copy()
            with (
                nullcontext()
                if timing is None
                else timing.measure("control.command")
            ):
                commands: dict[str, ArmTendonRateCommand] = {}
                arms_by_name = {
                    arm.name: arm for arm in self.backend.assembly.enabled_arms
                }
                for arm_name, arm_state in state.arms.items():
                    max_rate = arms_by_name[
                        arm_name
                    ].spatial_arm.limits.max_tendon_rate_mps
                    target = targets[arm_name]
                    model = self.backend.layout.bending_models[arm_name]
                    if control_space == "bending_compatible":
                        target = model.project(target)
                        requested = (
                            target
                            - np.asarray(arm_state.tendon_target_m, dtype=float)
                        ) / self.control_dt_s
                        requested = model.project(requested)
                        ratios = np.divide(
                            max_rate,
                            np.abs(requested),
                            out=np.full_like(max_rate, np.inf),
                            where=np.abs(requested) > 0.0,
                        )
                        rates = min(1.0, float(np.min(ratios))) * requested
                    else:
                        rates = target_rates(
                            target,
                            np.asarray(arm_state.tendon_target_m, dtype=float),
                            max_rate,
                            dt=self.control_dt_s,
                        )
                    commands[arm_name] = ArmTendonRateCommand(
                        rates,
                        control_space=control_space,
                    )
                base_twist = self._base_target_twist(
                    state,
                    base_target_pose_rpy,
                )
            with self._simulation_lock:
                next_state = self.backend.step_system(
                    RobotSystemCommand(
                        base_twist_world=base_twist,
                        arms=commands,
                        metadata={
                            "source": "mujoco_system_debug_viewer",
                            "enforce_backend_base_speed_limits": True,
                        },
                    ),
                    dt=self.control_dt_s,
                    n_substeps=self.n_substeps,
                )
            self.state = next_state
            self._state_slot.publish(next_state)
            return next_state
        finally:
            if timing is not None:
                timing.finish_cycle()

    def _base_target_twist(
        self,
        state: RobotSystemState,
        base_target_pose_rpy: np.ndarray,
    ) -> np.ndarray:
        if not self.base_control_enabled:
            return np.zeros(6, dtype=float)
        actual = _pose_xyz_rpy(state.base.pose)
        error = base_target_pose_rpy - actual
        error[3:] = _wrap_angles(error[3:])
        twist = error / self.control_dt_s
        base = self.backend.assembly.base
        twist[:3] = np.clip(
            twist[:3],
            -float(base.max_linear_speed_mps),
            float(base.max_linear_speed_mps),
        )
        twist[3:] = np.clip(
            twist[3:],
            -float(base.max_angular_speed_rad_s),
            float(base.max_angular_speed_rad_s),
        )
        return twist

    def toggle_run(self) -> None:
        if self._running:
            self.pause()
            return
        if self._control_worker.is_alive:
            return
        self._worker_error = None
        self._worker_error_reported = False
        self._control_worker.start()
        self._running = self._control_worker.is_alive
        self.run_button.label.set_text("Pause" if self._running else "Run")

    def pause(self) -> None:
        self._running = False
        self._control_worker.stop()
        self.run_button.label.set_text("Run")

    def show(self) -> None:
        self.status_panel.show(block=False)
        if self.tendon_panel is not None:
            self.tendon_panel.show(block=False)
        self._status_gate.reset(self._clock())
        self.timer.start()
        try:
            self.panel.show(block=True)
        finally:
            self.timer.stop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.pause()
        self.timer.stop()
        self.status_panel.close()
        if self.tendon_panel is not None:
            self.tendon_panel.close()
        self.panel.close()

    def _on_control_window_closed(self, _event) -> None:
        if self._closed:
            return
        self._closed = True
        self.pause()
        self.timer.stop()
        self.status_panel.close()
        if self.tendon_panel is not None:
            self.tendon_panel.close()

    def refresh(self) -> None:
        """Refresh controls, read-only windows, and external views."""

        self._update_views(force=True)

    def _on_timer(self) -> bool:
        if not self.panel.is_open():
            self.pause()
            self.timer.stop()
            return False
        if self._worker_error is not None and not self._worker_error_reported:
            self._worker_error_reported = True
            traceback.print_exception(self._worker_error)
            self.run_button.label.set_text("Run")
        self._update_views(force=False)
        return self.panel.is_open()

    def _on_control_worker_error(self, error: BaseException) -> None:
        self._worker_error = error
        self._running = False

    def _update_views(self, *, force: bool = False) -> None:
        with self._target_lock:
            sync_targets = bool(self._dirty_target_arms)
            update_controls = (
                force
                or self._views_dirty
                or self._base_controls_dirty
                or bool(self._dirty_curvature_components)
            )
            if update_controls:
                self._views_dirty = False
        state = self._state_slot.snapshot()[0]
        timing = self.runtime_timing
        if sync_targets:
            self._sync_target_controls()
        if update_controls:
            self._sync_base_controls()
            self._sync_curvature_controls()
            with (
                nullcontext()
                if timing is None
                else timing.measure("ui.controls")
            ):
                self.panel.fig.canvas.draw_idle()
        now_s = self._clock()
        if force or self._status_gate.due(now_s):
            if self.status_panel.is_open():
                with (
                    nullcontext()
                    if timing is None
                    else timing.measure("ui.status")
                ):
                    self._update_status(state=state)
            if self.tendon_panel is not None and self.tendon_panel.is_open():
                with (
                    nullcontext()
                    if timing is None
                    else timing.measure("ui.tendon_monitor")
                ):
                    self.tendon_panel.update(state)
        with (
            nullcontext()
            if timing is None
            else timing.measure("windows.total")
        ):
            self._notify_state_updated(state)

    def _update_status(
        self,
        *,
        redraw: bool = True,
        state: RobotSystemState | None = None,
    ) -> None:
        state = self._state_slot.snapshot()[0] if state is None else state
        with self._target_lock:
            control_space = self.control_space
            base_target = self.base_target_pose_rpy.copy()
            targets = {
                arm_name: values.copy()
                for arm_name, values in self.targets.items()
            }
        info_text = self._manual_diagnostic_text(
            state,
            control_space=control_space,
            base_target_pose_rpy=base_target,
            targets=targets,
        )
        if self.diagnostic_text_provider is not None:
            extra = self.diagnostic_text_provider(
                state,
                control_space,
            )
            if extra:
                info_text = f"{info_text}\n\n{extra}"
        if self.status_panel.is_open():
            self.status_panel.update(info_text, redraw=redraw)

    def _manual_diagnostic_text(
        self,
        state: RobotSystemState | None = None,
        *,
        control_space: str | None = None,
        base_target_pose_rpy: np.ndarray | None = None,
        targets: Mapping[str, np.ndarray] | None = None,
    ) -> str:
        state = self._state_slot.snapshot()[0] if state is None else state
        control_space = self.control_space if control_space is None else control_space
        base_target_pose_rpy = (
            self.base_target_pose_rpy
            if base_target_pose_rpy is None
            else base_target_pose_rpy
        )
        targets = self.targets if targets is None else targets
        base_actual = _pose_xyz_rpy(state.base.pose)
        lines = [
            f"time: {state.time_s:.3f} s",
            f"mode: {control_space}",
            "base target xyz: "
            + " ".join(f"{value:+.3f}" for value in base_target_pose_rpy[:3]),
            "base actual xyz: "
            + " ".join(f"{value:+.3f}" for value in base_actual[:3]),
            "base target rpy: "
            + " ".join(f"{value:+.1f}" for value in np.rad2deg(base_target_pose_rpy[3:])),
            "base actual rpy: "
            + " ".join(f"{value:+.1f}" for value in np.rad2deg(base_actual[3:])),
        ]
        for arm_name, arm in state.arms.items():
            model = self.backend.layout.bending_models[arm_name]
            target_bending = model.estimate(model.project(targets[arm_name]))
            actual_bending = model.estimate(arm.tendon_displacement_m)
            lines.append(arm_name)
            for index, pose in enumerate(arm.segment_poses_world):
                position = pose[:3, 3]
                lines.append(
                    f" S{index + 1} k=({target_bending[2*index]:+.3f},"
                    f"{target_bending[2*index+1]:+.3f}) "
                    f"act=({actual_bending[2*index]:+.3f},"
                    f"{actual_bending[2*index+1]:+.3f}) "
                    f"pW=({position[0]:+.4f},{position[1]:+.4f},"
                    f"{position[2]:+.4f}) m"
                )
            if arm.tool_wrench is not None:
                force = arm.tool_wrench.force_sensor_n
                torque = arm.tool_wrench.torque_sensor_nm
                lines.append(
                    f" F=({force[0]:+.3f},{force[1]:+.3f},{force[2]:+.3f}) N "
                    f"M=({torque[0]:+.4f},{torque[1]:+.4f},"
                    f"{torque[2]:+.4f}) Nm "
                    f"{'SATURATED' if arm.tool_wrench.saturated else 'ok'}"
                )
        return "\n".join(lines)

    def _notify_state_updated(self, state: RobotSystemState | None = None) -> None:
        if self.state_update_callback is not None:
            latest = self._state_slot.snapshot()[0] if state is None else state
            self.state_update_callback(latest)


def _format_target_mm(value_mm: float) -> str:
    return f"{float(value_mm):.3f}"


def _format_curvature_target(value_1_per_m: float) -> str:
    return f"{float(value_1_per_m):.3f}"


def _copy_targets(
    targets: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        arm_name: np.asarray(values, dtype=float).copy()
        for arm_name, values in targets.items()
    }


def _format_base_target(value: float, component_index: int) -> str:
    displayed = np.rad2deg(value) if component_index >= 3 else value
    return f"{float(displayed):.3f}"


def _pose_xyz_rpy(pose) -> np.ndarray:
    rotation = quaternion_wxyz_to_rotation_matrix(pose.quat)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1.0e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.concatenate(
        (np.asarray(pose.position, dtype=float), np.array([roll, pitch, yaw]))
    )


def _wrap_angles(values: np.ndarray) -> np.ndarray:
    angles = np.asarray(values, dtype=float)
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def _disable_widget_blit(widget_type) -> dict[str, bool]:
    """Disable Matplotlib widget blitting when the installed version supports it."""

    import inspect

    if "useblit" not in inspect.signature(widget_type).parameters:
        return {}
    return {"useblit": False}


def _disconnect_textbox_resize(text_box) -> None:
    """Drop the Matplotlib TextBox resize hook that breaks on ResizeEvent."""

    canvas = getattr(text_box, "canvas", None)
    callbacks = getattr(canvas, "callbacks", None)
    callback_map = getattr(callbacks, "callbacks", {}) if callbacks is not None else {}
    resize_callbacks = callback_map.get("resize_event", {})
    for cid in tuple(getattr(text_box, "_cids", ())):
        reference = resize_callbacks.get(cid)
        callback = reference() if callable(reference) else None
        if getattr(callback, "__name__", "") != "_resize":
            continue
        canvas.mpl_disconnect(cid)
        text_box._cids.remove(cid)


__all__ = [
    "MujocoSystemDebugViewer",
    "available_named_targets",
    "bounded_compatible_target",
    "named_system_target",
    "normalize_curvature_target",
    "normalize_target_mm",
    "target_rates",
]
