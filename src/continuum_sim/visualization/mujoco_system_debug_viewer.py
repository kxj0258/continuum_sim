"""Interactive tendon target controls for the scenario-based MuJoCo system."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from continuum_sim.backends.mujoco_system_backend import MujocoSystemBackend
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


class MujocoSystemDebugViewer:
    """Matplotlib controls and diagnostics for a composed MuJoCo system backend."""

    def __init__(
        self,
        backend: MujocoSystemBackend,
        *,
        control_dt_s: float,
        n_substeps: int,
        state_update_callback: Callable[[RobotSystemState], None] | None = None,
    ) -> None:
        from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

        if control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive.")
        if n_substeps <= 0:
            raise ValueError("n_substeps must be positive.")
        if len(backend.assembly.enabled_arms) > 2:
            raise ValueError("The debug viewer supports at most two enabled arms.")

        self.backend = backend
        self.control_dt_s = float(control_dt_s)
        self.n_substeps = int(n_substeps)
        self.state_update_callback = state_update_callback
        self.state = backend.reset_system()
        self.arm_names = tuple(self.state.arms)
        self.targets = {
            name: np.zeros_like(arm.tendon_displacement_m)
            for name, arm in self.state.arms.items()
        }
        self._running = False
        self._updating_controls = False

        self.panel = SystemTendonMonitorPanel(
            title="continuum_sim MuJoCo system tendon debug"
        )
        self.panel.info_ax.set_position((0.68, 0.31, 0.29, 0.20))
        self.panel._info_text.set_fontsize(7.5)

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

        self.reset_button = Button(
            self.panel.fig.add_axes((0.06, 0.07, 0.10, 0.04)),
            "Reset",
        )
        self.zero_button = Button(
            self.panel.fig.add_axes((0.18, 0.07, 0.10, 0.04)),
            "Zero",
        )
        self.step_button = Button(
            self.panel.fig.add_axes((0.30, 0.07, 0.10, 0.04)),
            "Step",
        )
        self.run_button = Button(
            self.panel.fig.add_axes((0.42, 0.07, 0.10, 0.04)),
            "Run",
        )
        self.named_targets = available_named_targets(self.arm_names)
        self.radio = RadioButtons(
            self.panel.fig.add_axes((0.68, 0.07, 0.29, 0.19)),
            self.named_targets,
            active=0,
        )
        self.timer = self.panel.fig.canvas.new_timer(
            interval=max(1, round(1000.0 * self.control_dt_s))
        )
        self.timer.add_callback(self._on_timer)
        self._connect_controls()
        self.panel.update(self.state, redraw=False)
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
        x0 = 0.06 + 0.30 * arm_index
        y0 = 0.49
        for tendon_index, (minimum, maximum) in enumerate(
            zip(lower, upper, strict=True)
        ):
            y = y0 - 0.040 * tendon_index
            slider_axis = self.panel.fig.add_axes(
                (x0, y, 0.20, 0.018)
            )
            slider = slider_type(
                ax=slider_axis,
                label=f"{arm_name}:{tendon_index + 1} [mm]",
                valmin=1000.0 * float(minimum),
                valmax=1000.0 * float(maximum),
                valinit=0.0,
                valfmt="% .3f",
            )
            slider.valtext.set_visible(False)
            input_axis = self.panel.fig.add_axes(
                (x0 + 0.215, y - 0.003, 0.07, 0.025)
            )
            target_input = text_box_type(
                input_axis,
                "",
                initial="0.000",
                textalignment="center",
            )
            sliders.append(slider)
            target_inputs.append(target_input)
        return sliders, target_inputs

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
        self.step_button.on_clicked(lambda _event: self.step())
        self.run_button.on_clicked(lambda _event: self.toggle_run())
        self.radio.on_clicked(self.apply_named_target)

    def _on_slider(self, arm_name: str, tendon_index: int, value_mm: float) -> None:
        if self._updating_controls:
            return
        self.targets[arm_name][tendon_index] = 0.001 * float(value_mm)
        self._updating_controls = True
        try:
            self.target_inputs[arm_name][tendon_index].set_val(
                _format_target_mm(value_mm)
            )
        finally:
            self._updating_controls = False

    def _on_target_input(
        self,
        arm_name: str,
        tendon_index: int,
        text: str,
    ) -> None:
        if self._updating_controls:
            return
        slider = self.sliders[arm_name][tendon_index]
        current_mm = 1000.0 * self.targets[arm_name][tendon_index]
        value_mm = normalize_target_mm(
            text,
            float(slider.valmin),
            float(slider.valmax),
            current_mm,
        )
        self._updating_controls = True
        try:
            self.targets[arm_name][tendon_index] = 0.001 * value_mm
            slider.set_val(value_mm)
            self.target_inputs[arm_name][tendon_index].set_val(
                _format_target_mm(value_mm)
            )
        finally:
            self._updating_controls = False

    def reset(self) -> RobotSystemState:
        self.pause()
        self.state = self.backend.reset_system()
        self.set_targets(
            {name: np.zeros_like(values) for name, values in self.targets.items()}
        )
        self._update_views()
        return self.state

    def zero_targets(self) -> None:
        self.set_targets(
            {name: np.zeros_like(values) for name, values in self.targets.items()}
        )

    def set_targets(self, targets: Mapping[str, np.ndarray]) -> None:
        if set(targets) != set(self.targets):
            raise ValueError("Targets must exactly match the enabled arm names.")
        self._updating_controls = True
        try:
            for arm_name, values in targets.items():
                array = np.asarray(values, dtype=float)
                if array.shape != self.targets[arm_name].shape:
                    raise ValueError(
                        f"Target for arm {arm_name!r} has shape {array.shape}, "
                        f"expected {self.targets[arm_name].shape}."
                    )
                normalized = np.empty_like(array)
                for index, (slider, target_input, value_m) in enumerate(zip(
                    self.sliders[arm_name],
                    self.target_inputs[arm_name],
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
                    slider.set_val(value_mm)
                    target_input.set_val(_format_target_mm(value_mm))
                self.targets[arm_name] = normalized
        finally:
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
        return self.step()

    def step(self) -> RobotSystemState:
        commands: dict[str, ArmTendonRateCommand] = {}
        arms_by_name = {
            arm.name: arm for arm in self.backend.assembly.enabled_arms
        }
        for arm_name, arm_state in self.state.arms.items():
            max_rate = arms_by_name[
                arm_name
            ].spatial_arm.limits.max_tendon_rate_mps
            commands[arm_name] = ArmTendonRateCommand(
                target_rates(
                    self.targets[arm_name],
                    np.asarray(arm_state.tendon_target_m, dtype=float),
                    max_rate,
                    dt=self.control_dt_s,
                )
            )
        self.state = self.backend.step_system(
            RobotSystemCommand(
                base_twist_world=np.zeros(6, dtype=float),
                arms=commands,
                metadata={"source": "mujoco_system_debug_viewer"},
            ),
            dt=self.control_dt_s,
            n_substeps=self.n_substeps,
        )
        self._update_views()
        return self.state

    def toggle_run(self) -> None:
        if self._running:
            self.pause()
            return
        self._running = True
        self.run_button.label.set_text("Pause")
        self.timer.start()

    def pause(self) -> None:
        self._running = False
        self.timer.stop()
        self.run_button.label.set_text("Run")

    def show(self) -> None:
        self.panel.show(block=True)

    def close(self) -> None:
        self.pause()
        self.panel.close()

    def _on_timer(self) -> bool:
        if self._running and self.panel.is_open():
            self.step()
        elif self._running:
            self.pause()
        return True

    def _update_views(self) -> None:
        self.panel.update(self.state)
        self.panel.flush_events()
        self._notify_state_updated()

    def _notify_state_updated(self) -> None:
        if self.state_update_callback is not None:
            self.state_update_callback(self.state)


def _format_target_mm(value_mm: float) -> str:
    return f"{float(value_mm):.3f}"


__all__ = [
    "MujocoSystemDebugViewer",
    "available_named_targets",
    "named_system_target",
    "normalize_target_mm",
    "target_rates",
]
