"""Interactive tendon target controls for the scenario-based MuJoCo system."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from threading import Event, RLock, Thread, current_thread
from time import perf_counter
import traceback

import numpy as np

from continuum_sim.backends.mujoco_system_backend import MujocoSystemBackend
from continuum_sim.model.base_pose import quaternion_wxyz_to_rotation_matrix
from continuum_sim.model.mount_frame import load_mobile_base_mount_config
from continuum_sim.system.types import (
    ArmTendonRateCommand,
    RobotSystemCommand,
    RobotSystemState,
)
from continuum_sim.visualization.system_tendon_debug import SystemTendonMonitorPanel


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


class _FixedRateControlWorker:
    """Run control work on a monotonic fixed-rate clock outside the GUI thread."""

    def __init__(
        self,
        interval_s: float,
        callback: Callable[[], object],
        error_callback: Callable[[BaseException], None],
    ) -> None:
        self.interval_s = float(interval_s)
        self._callback = callback
        self._error_callback = error_callback
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._run,
            name="continuum-sim-control",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()
        if thread is None or not thread.is_alive():
            self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        next_deadline_s = perf_counter()
        try:
            while not self._stop_event.is_set():
                self._callback()
                next_deadline_s += self.interval_s
                now_s = perf_counter()
                if next_deadline_s <= now_s:
                    missed = int((now_s - next_deadline_s) // self.interval_s) + 1
                    next_deadline_s += missed * self.interval_s
                self._stop_event.wait(max(0.0, next_deadline_s - perf_counter()))
        except BaseException as exc:  # noqa: BLE001 - report worker failures to UI.
            self._error_callback(exc)


class MujocoSystemDebugViewer:
    """Matplotlib controls and diagnostics for a composed MuJoCo system backend."""

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
        curvature_step_1_per_m: float = 0.5,
        panel_fps: float = 15.0,
        simulation_lock=None,
        runtime_timing=None,
    ) -> None:
        from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

        if control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive.")
        if n_substeps <= 0:
            raise ValueError("n_substeps must be positive.")
        if len(backend.assembly.enabled_arms) > 2:
            raise ValueError("The debug viewer supports at most two enabled arms.")
        if not np.isfinite(curvature_step_1_per_m) or curvature_step_1_per_m <= 0.0:
            raise ValueError("curvature_step_1_per_m must be positive and finite.")
        if not np.isfinite(panel_fps) or panel_fps <= 0.0:
            raise ValueError("panel_fps must be positive and finite.")

        self.backend = backend
        self.control_dt_s = float(control_dt_s)
        self.n_substeps = int(n_substeps)
        self.state_update_callback = state_update_callback
        self.diagnostic_text_provider = diagnostic_text_provider
        self.panel_fps = float(panel_fps)
        self._simulation_lock = RLock() if simulation_lock is None else simulation_lock
        self.runtime_timing = runtime_timing
        if runtime_timing is not None:
            backend.set_runtime_timing(runtime_timing)
        self.state = backend.reset_system()
        self.arm_names = tuple(self.state.arms)
        self.targets = {
            name: np.asarray(arm.tendon_target_m, dtype=float).copy()
            for name, arm in self.state.arms.items()
        }
        self._running = False
        self._updating_controls = False
        self._views_dirty = False
        self._dirty_target_arms: set[str] = set()
        self._base_controls_dirty = False
        self._curvature_control_dirty = False
        self._worker_error: BaseException | None = None
        self._worker_error_reported = False
        self.control_space = "bending_compatible"
        self.curvature_step_1_per_m = float(curvature_step_1_per_m)

        self.panel = SystemTendonMonitorPanel(
            title="continuum_sim MuJoCo system tendon debug"
        )
        self.panel.fig.set_size_inches(18.0, 10.0, forward=True)
        self.panel.length_ax.set_position((0.04, 0.72, 0.58, 0.24))
        self.panel.force_ax.set_position((0.67, 0.72, 0.29, 0.24))
        self.panel.info_ax.set_position((0.68, 0.11, 0.28, 0.20))
        self.panel._info_text.set_fontsize(6.2)

        self.sliders: dict[str, list[Slider]] = {}
        self.target_inputs: dict[str, list[TextBox]] = {}
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

        self.segment_buttons: dict[str, list[tuple[Button, Button, Button, Button]]] = {}
        for arm_index, arm_name in enumerate(self.arm_names):
            self.segment_buttons[arm_name] = self._build_segment_controls(
                Button,
                arm_index,
                arm_name,
            )

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
        self.radio = RadioButtons(
            self.panel.fig.add_axes((0.79, 0.025, 0.17, 0.09)),
            self.named_targets,
            active=0,
            **_disable_widget_blit(RadioButtons),
        )
        self.mode_radio = RadioButtons(
            self.panel.fig.add_axes((0.50, 0.025, 0.11, 0.09)),
            ("compatible", "raw tendon"),
            active=0,
            **_disable_widget_blit(RadioButtons),
        )
        self.panel.fig.text(0.625, 0.084, "curvature step [1/m]", fontsize=8)
        self.curvature_step_input = TextBox(
            self.panel.fig.add_axes((0.66, 0.045, 0.08, 0.035)),
            "",
            initial=f"{self.curvature_step_1_per_m:g}",
            textalignment="center",
            **_disable_widget_blit(TextBox),
        )
        _disconnect_textbox_resize(self.curvature_step_input)
        self._control_worker = _FixedRateControlWorker(
            self.control_dt_s,
            self.step,
            self._on_control_worker_error,
        )
        self.timer = self.panel.fig.canvas.new_timer(
            interval=max(1, round(1000.0 / self.panel_fps))
        )
        self.timer.add_callback(self._on_timer)
        self._connect_controls()
        self._update_panel(redraw=False)
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

    def _build_segment_controls(
        self,
        button_type,
        arm_index: int,
        arm_name: str,
    ) -> list[tuple]:
        controls = []
        x0 = 0.04 + 0.31 * arm_index
        role = "MAIN / EXECUTOR" if arm_name == "executor" else "CAMERA / OBSERVER"
        self.panel.fig.text(x0, 0.625, role, fontsize=10, weight="bold")
        self.panel.fig.text(x0, 0.603, "segment-local curvature [1/m]", fontsize=8)
        for segment_index in range(3):
            y = 0.565 - 0.034 * segment_index
            self.panel.fig.text(x0, y + 0.004, f"S{segment_index + 1}", fontsize=8)
            buttons = []
            for button_index, (label, axis, direction) in enumerate(
                (
                    ("+kx", 0, 1.0),
                    ("-kx", 0, -1.0),
                    ("+ky", 1, 1.0),
                    ("-ky", 1, -1.0),
                )
            ):
                button = button_type(
                    self.panel.fig.add_axes(
                        (x0 + 0.040 + 0.058 * button_index, y, 0.052, 0.025)
                    ),
                    label,
                    **_disable_widget_blit(button_type),
                )
                button.on_clicked(
                    lambda _event, name=arm_name, segment=segment_index,
                    component=axis, sign=direction: self.adjust_segment_bending(
                        name,
                        segment,
                        component,
                        sign,
                    )
                )
                buttons.append(button)
            controls.append(tuple(buttons))
        return controls

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
        with self._simulation_lock:
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
        with self._simulation_lock:
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

    def _sync_curvature_control(self) -> None:
        with self._simulation_lock:
            if not self._curvature_control_dirty:
                return
            value = self.curvature_step_1_per_m
            self._curvature_control_dirty = False
        formatted = f"{value:g}"
        if self.curvature_step_input.text != formatted:
            self._apply_widget_updates([(self.curvature_step_input, formatted)])

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
        self.reset_button.on_clicked(lambda _event: self.reset())
        self.zero_button.on_clicked(lambda _event: self.zero_targets())
        self.zero_base_button.on_clicked(lambda _event: self.zero_base_target())
        self.step_button.on_clicked(lambda _event: self._step_once())
        self.run_button.on_clicked(lambda _event: self.toggle_run())
        self.radio.on_clicked(self.apply_named_target)
        self.mode_radio.on_clicked(self._set_control_mode)
        self.curvature_step_input.on_submit(self._set_curvature_step)

    def _set_curvature_step(self, text: str) -> None:
        with self._simulation_lock:
            try:
                value = float(text)
            except (TypeError, ValueError):
                value = self.curvature_step_1_per_m
            if not np.isfinite(value) or value <= 0.0:
                value = self.curvature_step_1_per_m
            self.curvature_step_1_per_m = float(value)
            self._curvature_control_dirty = True
            self._views_dirty = True

    def adjust_segment_bending(
        self,
        arm_name: str,
        segment_index: int,
        component_index: int,
        direction: float,
    ) -> np.ndarray:
        """Increment one segment-local kx/ky component and update tendon targets."""

        if arm_name not in self.targets:
            raise KeyError(f"Unknown arm {arm_name!r}.")
        if segment_index not in range(3):
            raise ValueError("segment_index must be 0, 1, or 2.")
        if component_index not in (0, 1):
            raise ValueError("component_index must be 0 (kx) or 1 (ky).")
        if direction not in (-1.0, 1.0):
            raise ValueError("direction must be -1 or 1.")
        timing = getattr(self, "runtime_timing", None)
        with (
            nullcontext()
            if timing is None
            else timing.measure("input.callback")
        ):
            with self._simulation_lock:
                if timing is not None:
                    component = "kx" if component_index == 0 else "ky"
                    sign = "+" if direction > 0.0 else "-"
                    timing.mark_input(
                        f"{arm_name}:S{segment_index + 1}:{sign}{component}"
                    )
                model = self.backend.layout.bending_models[arm_name]
                current = model.project(self.targets[arm_name])
                bending = model.estimate(current)
                bending[2 * segment_index + component_index] += (
                    direction * self.curvature_step_1_per_m
                )
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
            with self._simulation_lock:
                if timing is not None:
                    timing.mark_input(f"{arm_name}:T{tendon_index + 1}:slider")
                self.targets[arm_name][tendon_index] = 0.001 * float(value_mm)
                self._dirty_target_arms.add(arm_name)
                self._views_dirty = True
                self._project_targets_if_compatible(arm_name)

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
            with self._simulation_lock:
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
            self.state = self.backend.reset_system()
            self.base_target_pose_rpy = _pose_xyz_rpy(self.state.base.pose)
            self.base_initial_pose_rpy = self.base_target_pose_rpy.copy()
            self._base_controls_dirty = True
            self.set_targets(
                {name: np.zeros_like(values) for name, values in self.targets.items()}
            )
            self._views_dirty = True
            return self.state

    def zero_targets(self) -> None:
        self.set_targets(
            {name: np.zeros_like(values) for name, values in self.targets.items()}
        )

    def set_targets(self, targets: Mapping[str, np.ndarray]) -> None:
        unknown = set(targets).difference(self.targets)
        if unknown:
            raise ValueError(f"Unknown target arm names: {sorted(unknown)}.")
        with self._simulation_lock:
            for arm_name, values in targets.items():
                array = np.asarray(values, dtype=float)
                if array.shape != self.targets[arm_name].shape:
                    raise ValueError(
                        f"Target for arm {arm_name!r} has shape {array.shape}, "
                        f"expected {self.targets[arm_name].shape}."
                    )
                normalized = np.empty_like(array)
                for index, (slider, value_m) in enumerate(zip(
                    self.sliders[arm_name],
                    array,
                    strict=True,
                )):
                    value_mm = normalize_target_mm(
                        1000.0 * float(value_m),
                        float(slider.valmin),
                        float(slider.valmax),
                        1000.0 * self.targets[arm_name][index],
                    )
                    normalized[index] = 0.001 * value_mm
                if np.array_equal(normalized, self.targets[arm_name]):
                    continue
                self.targets[arm_name] = normalized
                self._dirty_target_arms.add(arm_name)
                self._views_dirty = True

    def _sync_target_controls(self) -> None:
        with self._simulation_lock:
            dirty_arms = tuple(self._dirty_target_arms)
            target_values = {
                arm_name: self.targets[arm_name].copy()
                for arm_name in dirty_arms
            }
            self._dirty_target_arms.clear()
        updates = []
        for arm_name, values in target_values.items():
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
        with self._simulation_lock:
            return self.state

    def _step_once(self) -> RobotSystemState:
        if self._running:
            with self._simulation_lock:
                return self.state
        return self.step()

    def step(self) -> RobotSystemState:
        timing = self.runtime_timing
        try:
            with self._simulation_lock:
                if timing is not None:
                    timing.start_cycle()
                with (
                    nullcontext()
                    if timing is None
                    else timing.measure("control.command")
                ):
                    commands: dict[str, ArmTendonRateCommand] = {}
                    arms_by_name = {
                        arm.name: arm for arm in self.backend.assembly.enabled_arms
                    }
                    for arm_name, arm_state in self.state.arms.items():
                        max_rate = arms_by_name[
                            arm_name
                        ].spatial_arm.limits.max_tendon_rate_mps
                        target = self.targets[arm_name]
                        model = self.backend.layout.bending_models[arm_name]
                        if self.control_space == "bending_compatible":
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
                            control_space=self.control_space,
                        )
                    base_twist = self._base_target_twist()
                self.state = self.backend.step_system(
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
                self._views_dirty = True
                return self.state
        finally:
            if timing is not None:
                timing.finish_cycle()

    def _base_target_twist(self) -> np.ndarray:
        if not self.base_control_enabled:
            return np.zeros(6, dtype=float)
        actual = _pose_xyz_rpy(self.state.base.pose)
        error = self.base_target_pose_rpy - actual
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

    def _set_control_mode(self, label: str) -> None:
        with self._simulation_lock:
            self.control_space = (
                "bending_compatible" if label == "compatible" else "raw_tendon_debug"
            )
            if self.control_space == "bending_compatible":
                projected = {
                    arm_name: self.backend.layout.bending_models[arm_name].project(values)
                    for arm_name, values in self.targets.items()
                }
                self.set_targets(projected)
            self._views_dirty = True

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
        self.timer.start()
        try:
            self.panel.show(block=True)
        finally:
            self.timer.stop()

    def close(self) -> None:
        self.pause()
        self.timer.stop()
        self.panel.close()

    def refresh(self) -> None:
        """Redraw the panel and notify external diagnostic views."""

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
        with self._simulation_lock:
            update_panel = force or self._views_dirty or bool(
                self._dirty_target_arms
            ) or self._base_controls_dirty or self._curvature_control_dirty
            state = self.state
            if update_panel:
                self._views_dirty = False
        if update_panel:
            self._sync_target_controls()
            self._sync_base_controls()
            self._sync_curvature_control()
        timing = self.runtime_timing
        if update_panel:
            with (
                nullcontext()
                if timing is None
                else timing.measure("ui.panel")
            ):
                self._update_panel(state=state)
        with (
            nullcontext()
            if timing is None
            else timing.measure("windows.total")
        ):
            self._notify_state_updated(state)

    def _update_panel(
        self,
        *,
        redraw: bool = True,
        state: RobotSystemState | None = None,
    ) -> None:
        state = self.state if state is None else state
        with self._simulation_lock:
            control_space = self.control_space
            curvature_step = self.curvature_step_1_per_m
            base_target = self.base_target_pose_rpy.copy()
            targets = {
                arm_name: values.copy()
                for arm_name, values in self.targets.items()
            }
        info_text = self._manual_diagnostic_text(
            state,
            control_space=control_space,
            curvature_step_1_per_m=curvature_step,
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
        self.panel.update(
            state,
            redraw=redraw,
            info_text=info_text,
        )

    def _manual_diagnostic_text(
        self,
        state: RobotSystemState | None = None,
        *,
        control_space: str | None = None,
        curvature_step_1_per_m: float | None = None,
        base_target_pose_rpy: np.ndarray | None = None,
        targets: Mapping[str, np.ndarray] | None = None,
    ) -> str:
        state = self.state if state is None else state
        control_space = self.control_space if control_space is None else control_space
        curvature_step_1_per_m = (
            self.curvature_step_1_per_m
            if curvature_step_1_per_m is None
            else curvature_step_1_per_m
        )
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
            f"curvature step: {curvature_step_1_per_m:g} 1/m",
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
            self.state_update_callback(self.state if state is None else state)


def _format_target_mm(value_mm: float) -> str:
    return f"{float(value_mm):.3f}"


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
    "normalize_target_mm",
    "target_rates",
]
