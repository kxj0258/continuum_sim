"""Interactive matplotlib viewer for the PCC continuum-arm model."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from continuum_sim.kinematics.pcc import PCCForwardKinematicsResult, forward_kinematics
from continuum_sim.kinematics.tendon_mapping import q_to_tendon_delta
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.visualization.axis_limits import (
    apply_axis_limits,
    default_robot_axis_limits,
)

Q_LABELS = ("kx1", "ky1", "eps1", "kx2", "ky2", "eps2", "kx3", "ky3", "eps3")
NAMED_Q_STATES = ("straight", "bend_x", "three_segment")
SEGMENT_COLORS = ("tab:blue", "tab:orange", "tab:green")


@dataclass(frozen=True)
class PCCViewData:
    """Computed data needed by the interactive plot."""

    q: np.ndarray
    tendon_delta: np.ndarray
    fk: PCCForwardKinematicsResult
    axis_limit: float

    @property
    def tip_position(self) -> np.ndarray:
        return self.fk.tip_pose[:3, 3]


def named_q(name: str) -> np.ndarray:
    """Return a named q vector."""
    states = {
        "straight": np.zeros(9, dtype=float),
        "bend_x": np.array([18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "three_segment": np.array([16.0, 0.0, 0.0, 0.0, -14.0, 0.0, -10.0, 12.0, 0.02]),
    }
    try:
        return states[name].copy()
    except KeyError as exc:
        raise ValueError(f"Unknown named q state {name!r}. Choose one of {tuple(states)}.") from exc


def compute_view_data(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    *,
    samples_per_segment: int = 40,
) -> PCCViewData:
    """Compute FK, tendon deltas, and plot scale for a q vector."""
    q_array = np.asarray(q, dtype=float)
    if q_array.shape != (9,):
        raise ValueError(f"Expected q with shape (9,), got {q_array.shape}.")
    fk = forward_kinematics(q_array, params, samples_per_segment=samples_per_segment)
    tendon_delta = q_to_tendon_delta(q_array, params)
    total_length = float(np.sum(params.segment_lengths))
    centerline_extent = float(np.max(np.abs(fk.centerline))) if fk.centerline.size else 0.0
    axis_limit = max(total_length * 0.75, centerline_extent * 1.2, 0.01)
    return PCCViewData(q=q_array.copy(), tendon_delta=tendon_delta, fk=fk, axis_limit=axis_limit)


class PCCInteractiveViewer:
    """Matplotlib UI for live PCC state exploration."""

    def __init__(
        self,
        params: ThreeSegmentRobotParams,
        *,
        initial_q: np.ndarray | None = None,
        samples_per_segment: int = 40,
    ) -> None:
        self.params = params
        self.samples_per_segment = samples_per_segment
        self.axis_limits = default_robot_axis_limits(params)
        self.q = np.asarray(initial_q if initial_q is not None else named_q("straight"), dtype=float)
        if self.q.shape != (9,):
            raise ValueError(f"Expected initial_q with shape (9,), got {self.q.shape}.")
        self._updating_sliders = False

        self.fig = plt.figure(figsize=(13.5, 8.0))
        self.fig.canvas.manager.set_window_title("continuum_sim PCC viewer")
        self.ax = self.fig.add_axes((0.05, 0.13, 0.52, 0.80), projection="3d")
        self.info_ax = self.fig.add_axes((0.60, 0.56, 0.36, 0.37))
        self.info_ax.axis("off")

        self._segment_lines = []
        self._tip_artist = None
        self._frame_artists = []
        self._info_text = self.info_ax.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )

        self.sliders = self._build_sliders()
        self.reset_button = self._build_reset_button()
        self.radio = self._build_preset_radio()
        self._connect_controls()
        self.update_plot(redraw=False)

    def _build_sliders(self) -> list[Slider]:
        sliders = []
        y0 = 0.48
        dy = 0.045
        for index, label in enumerate(Q_LABELS):
            axis = self.fig.add_axes((0.64, y0 - index * dy, 0.28, 0.025))
            value_min, value_max = (-0.1, 0.1) if label.startswith("eps") else (-30.0, 30.0)
            slider = Slider(
                ax=axis,
                label=label,
                valmin=value_min,
                valmax=value_max,
                valinit=float(self.q[index]),
                valstep=None,
            )
            sliders.append(slider)
        return sliders

    def _build_reset_button(self) -> Button:
        axis = self.fig.add_axes((0.64, 0.04, 0.12, 0.045))
        return Button(axis, "Reset")

    def _build_preset_radio(self) -> RadioButtons:
        axis = self.fig.add_axes((0.79, 0.025, 0.17, 0.085))
        return RadioButtons(axis, NAMED_Q_STATES, active=0)

    def _connect_controls(self) -> None:
        for slider in self.sliders:
            slider.on_changed(self._on_slider_changed)
        self.reset_button.on_clicked(lambda _event: self.set_q(named_q("straight")))
        self.radio.on_clicked(lambda name: self.set_q(named_q(str(name))))

    def _on_slider_changed(self, _value: float) -> None:
        if self._updating_sliders:
            return
        self.q = np.array([slider.val for slider in self.sliders], dtype=float)
        self.update_plot()

    def set_q(self, q: np.ndarray) -> None:
        """Set q, synchronize sliders, and redraw."""
        q_array = np.asarray(q, dtype=float)
        if q_array.shape != (9,):
            raise ValueError(f"Expected q with shape (9,), got {q_array.shape}.")
        self.q = q_array.copy()
        self._updating_sliders = True
        try:
            for slider, value in zip(self.sliders, self.q, strict=True):
                slider.set_val(float(value))
        finally:
            self._updating_sliders = False
        self.update_plot()

    def update_plot(self, redraw: bool = True) -> PCCViewData:
        """Refresh the plot from the current q and return computed view data."""
        view_data = compute_view_data(
            self.q,
            self.params,
            samples_per_segment=self.samples_per_segment,
        )
        self.ax.cla()
        self._draw_centerlines(view_data)
        self._draw_frames(view_data)
        self._format_axes(view_data)
        self._info_text.set_text(_format_info_text(view_data))
        if redraw:
            self.fig.canvas.draw_idle()
        return view_data

    def _draw_centerlines(self, view_data: PCCViewData) -> None:
        for points, color in zip(view_data.fk.segment_centerlines, SEGMENT_COLORS, strict=True):
            self.ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=2.5)
            self.ax.scatter(points[0, 0], points[0, 1], points[0, 2], color=color, s=18)
        tip = view_data.tip_position
        self.ax.scatter(tip[0], tip[1], tip[2], color="black", s=45, label="tip")

    def _draw_frames(self, view_data: PCCViewData) -> None:
        total_length = float(np.sum(self.params.segment_lengths))
        scale = max(total_length * 0.12, 0.005)
        self._draw_frame(np.eye(4), scale, "base")
        self._draw_frame(view_data.fk.tip_pose, scale, "tip")

    def _draw_frame(self, transform: np.ndarray, scale: float, label: str) -> None:
        origin = transform[:3, 3]
        axes = transform[:3, :3]
        colors = ("tab:red", "tab:green", "tab:blue")
        names = ("x", "y", "z")
        for axis_index, color, name in zip(range(3), colors, names, strict=True):
            direction = axes[:, axis_index] * scale
            self.ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                color=color,
                linewidth=1.3,
                arrow_length_ratio=0.25,
            )
            end = origin + direction * 1.12
            self.ax.text(end[0], end[1], end[2], f"{label}_{name}", color=color, fontsize=8)

    def _format_axes(self, view_data: PCCViewData) -> None:
        apply_axis_limits(self.ax, self.axis_limits)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")
        self.ax.set_title("PCC centerline")
        self.ax.grid(True)
        self.ax.view_init(elev=24, azim=-60)
        self.ax.set_box_aspect((1.0, 1.0, 1.4))

    def show(self) -> None:
        """Start the matplotlib event loop."""
        plt.show()

    def close(self) -> None:
        """Close the viewer figure."""
        plt.close(self.fig)


def _format_info_text(view_data: PCCViewData) -> str:
    tip = view_data.tip_position
    delta_m = view_data.tendon_delta
    delta_mm = delta_m * 1000.0
    lines = [
        "Tip position [m]",
        f"  x={tip[0]: .5f}  y={tip[1]: .5f}  z={tip[2]: .5f}",
        "",
        "q = [kx, ky, eps] x 3",
    ]
    q_segments = view_data.q.reshape(3, 3)
    for index, q_segment in enumerate(q_segments, start=1):
        lines.append(
            f"  seg{index}: kx={q_segment[0]: .3f}, ky={q_segment[1]: .3f}, eps={q_segment[2]: .4f}"
        )
    lines.extend(["", "tendon delta"])
    delta_segments_m = delta_m.reshape(3, 3)
    delta_segments_mm = delta_mm.reshape(3, 3)
    for index, (segment_m, segment_mm) in enumerate(
        zip(delta_segments_m, delta_segments_mm, strict=True),
        start=1,
    ):
        lines.append(
            "  seg{}: [{: .5f}, {: .5f}, {: .5f}] m".format(index, *segment_m)
        )
        lines.append(
            "        [{: .2f}, {: .2f}, {: .2f}] mm".format(*segment_mm)
        )
    return "\n".join(lines)
