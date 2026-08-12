"""Named-arm tendon diagnostics shared by scenario hooks and debug controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.system.types import RobotSystemState


@dataclass(frozen=True)
class SystemTendonViewData:
    """Flat, labelled tendon diagnostics derived from a named system state."""

    time_s: float
    labels: tuple[str, ...]
    arm_boundaries: tuple[int, ...]
    target_m: np.ndarray
    actual_m: np.ndarray
    error_m: np.ndarray
    force_n: np.ndarray
    saturation_summary: str


def system_tendon_view_data(state: RobotSystemState) -> SystemTendonViewData:
    """Flatten per-arm diagnostics without assuming fixed tendon offsets."""

    labels: list[str] = []
    targets: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    boundaries: list[int] = []
    tendon_count = 0
    for arm_index, (arm_name, arm) in enumerate(state.arms.items()):
        count = arm.tendon_displacement_m.size
        labels.extend(f"{arm_name}:{index}" for index in range(1, count + 1))
        targets.append(np.asarray(arm.tendon_target_m, dtype=float))
        actuals.append(arm.tendon_displacement_m)
        forces.append(np.asarray(arm.actuator_force_n, dtype=float))
        tendon_count += count
        if arm_index < len(state.arms) - 1:
            boundaries.append(tendon_count)

    target = _concatenate(targets)
    actual = _concatenate(actuals)
    force = _concatenate(forces)
    return SystemTendonViewData(
        time_s=float(state.time_s),
        labels=tuple(labels),
        arm_boundaries=tuple(boundaries),
        target_m=target,
        actual_m=actual,
        error_m=target - actual,
        force_n=force,
        saturation_summary=_saturation_summary(state),
    )


class SystemTendonMonitorPanel:
    """Read-only Matplotlib monitor for named system tendon diagnostics."""

    def __init__(self, *, title: str = "continuum_sim tendon monitor") -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self.fig = plt.figure(figsize=(15.5, 9.0))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(title)
        self.length_ax = self.fig.add_axes((0.06, 0.58, 0.56, 0.34))
        self.force_ax = self.fig.add_axes((0.68, 0.58, 0.29, 0.34))
        self.info_ax = self.fig.add_axes((0.68, 0.08, 0.29, 0.43))
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
        self.last_view_data: SystemTendonViewData | None = None
        self._labels: tuple[str, ...] | None = None
        self._target_bars = None
        self._actual_bars = None
        self._force_bars = None

    def update(
        self,
        state: RobotSystemState,
        *,
        redraw: bool = True,
        info_text: str | None = None,
    ) -> SystemTendonViewData:
        view = system_tendon_view_data(state)
        self.last_view_data = view
        self._draw_lengths(view)
        self._draw_forces(view)
        self._info_text.set_text(
            _format_info(view) if info_text is None else str(info_text)
        )
        if redraw:
            self.fig.canvas.draw_idle()
        return view

    def show(self, *, block: bool = False) -> None:
        if _is_noninteractive_backend(self._plt.get_backend()):
            return
        self._plt.show(block=block)
        if not block:
            self.flush_events()

    def flush_events(self) -> None:
        if not self.is_open():
            return
        flush = getattr(self.fig.canvas, "flush_events", None)
        if callable(flush):
            flush()

    def is_open(self) -> bool:
        return bool(self._plt.fignum_exists(self.fig.number))

    def close(self) -> None:
        self._plt.close(self.fig)

    def _draw_lengths(self, view: SystemTendonViewData) -> None:
        indices = np.arange(1, len(view.labels) + 1, dtype=float)
        if self._target_bars is None or self._actual_bars is None:
            self._labels = view.labels
            self._target_bars = self.length_ax.bar(
                indices - 0.19,
                1000.0 * view.target_m,
                width=0.36,
                color="tab:orange",
                alpha=0.65,
                label="target",
            )
            self._actual_bars = self.length_ax.bar(
                indices + 0.19,
                1000.0 * view.actual_m,
                width=0.36,
                color="tab:blue",
                alpha=0.75,
                label="current",
            )
            self._decorate_axis(self.length_ax, view, indices)
            self.length_ax.set_ylabel("tendon displacement [mm]")
            self.length_ax.set_title("Target vs current tendon length")
            self.length_ax.legend(loc="upper right", fontsize=8)
        else:
            self._require_stable_labels(view)
            for bar, height in zip(
                self._target_bars,
                1000.0 * view.target_m,
                strict=True,
            ):
                bar.set_height(float(height))
            for bar, height in zip(
                self._actual_bars,
                1000.0 * view.actual_m,
                strict=True,
            ):
                bar.set_height(float(height))
        self._rescale_y(self.length_ax)

    def _draw_forces(self, view: SystemTendonViewData) -> None:
        indices = np.arange(1, len(view.labels) + 1, dtype=float)
        if self._force_bars is None:
            self._labels = view.labels
            self._force_bars = self.force_ax.bar(
                indices,
                view.force_n,
                width=0.60,
                color="tab:red",
                alpha=0.75,
            )
            self._decorate_axis(self.force_ax, view, indices)
            self.force_ax.axhline(0.0, color="0.35", linewidth=0.8)
            self.force_ax.set_ylabel("actuator force [N]")
            self.force_ax.set_title("Tendon actuator force")
        else:
            self._require_stable_labels(view)
            for bar, height in zip(self._force_bars, view.force_n, strict=True):
                bar.set_height(float(height))
        self._rescale_y(self.force_ax)

    def _require_stable_labels(self, view: SystemTendonViewData) -> None:
        if view.labels != self._labels:
            raise ValueError("Tendon labels changed after monitor artists were created.")

    @staticmethod
    def _rescale_y(axis) -> None:
        axis.relim()
        axis.autoscale_view(scalex=False, scaley=True)

    @staticmethod
    def _decorate_axis(axis, view: SystemTendonViewData, indices: np.ndarray) -> None:
        axis.set_xlim(0.3, len(view.labels) + 0.7)
        axis.set_xticks(indices, view.labels, rotation=55, ha="right", fontsize=8)
        axis.grid(True, axis="y", alpha=0.25)
        for boundary in view.arm_boundaries:
            axis.axvline(boundary + 0.5, color="0.45", linestyle="--", linewidth=0.8)


def _concatenate(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=float)
    return np.concatenate(values).astype(float, copy=True)


def _saturation_summary(state: RobotSystemState) -> str:
    saturation = state.metadata.get("saturation")
    if not isinstance(saturation, dict):
        return "none"
    lines: list[str] = []
    for arm_name, arm_values in saturation.items():
        if not isinstance(arm_values, dict):
            continue
        rate = np.asarray(arm_values.get("rate", []), dtype=bool)
        displacement = np.asarray(arm_values.get("displacement", []), dtype=bool)
        residual = np.asarray(
            arm_values.get("compatibility_residual_mps", []),
            dtype=float,
        )
        scale = arm_values.get("common_scale", 1.0)
        mode = "raw" if bool(arm_values.get("raw_debug", False)) else "compatible"
        target_mode = str(arm_values.get("target_mode", "unknown"))
        lines.append(
            f"{arm_name}: rate {int(np.count_nonzero(rate))}/{rate.size}, "
            f"displacement {int(np.count_nonzero(displacement))}/{displacement.size}\n"
            f"  mode {mode}, scale {float(scale):.4f}, "
            f"target {target_mode}, residual {float(np.linalg.norm(residual)):.3e} m/s"
        )
    return "\n".join(lines) if lines else "none"


def _format_info(view: SystemTendonViewData) -> str:
    lines = [
        f"time_s: {view.time_s:.4f}",
        "",
        "error [mm]",
    ]
    lines.extend(
        f"{label:>12}: {1000.0 * error: .4f}"
        for label, error in zip(view.labels, view.error_m, strict=True)
    )
    lines.extend(("", "saturation", view.saturation_summary))
    return "\n".join(lines)


def _is_noninteractive_backend(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
        "module://matplotlib_inline.backend_inline",
    }


__all__ = [
    "SystemTendonMonitorPanel",
    "SystemTendonViewData",
    "system_tendon_view_data",
]
