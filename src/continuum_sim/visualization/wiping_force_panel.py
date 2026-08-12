"""Live matplotlib monitor for MuJoCo wiping contact force diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


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
class WipingForceViewData:
    """Single live diagnostic sample from the wiping runtime."""

    time_s: float
    normal_force_n: float
    target_normal_force_n: float
    force_error_n: float
    contact_proxy_m: float
    phase: str
    waypoint_index: int
    contact_source: str
    in_contact: bool


class WipingForceMonitorPanel:
    """Read-only panel for wiping normal force and contact proxy monitoring."""

    def __init__(
        self,
        *,
        target_normal_force_n: float,
        history_points: int = 300,
        title: str = "continuum_sim MuJoCo wiping force monitor",
    ) -> None:
        if history_points <= 0:
            raise ValueError(f"history_points must be positive, got {history_points}.")

        self.target_normal_force_n = float(target_normal_force_n)
        self.history_points = int(history_points)
        self.time_s: list[float] = []
        self.normal_force_n: list[float] = []
        self.target_force_n: list[float] = []
        self.force_error_n: list[float] = []
        self.contact_proxy_m: list[float] = []
        self.phase: list[str] = []
        self.waypoint_index: list[int] = []
        self.contact_source: list[str] = []
        self.in_contact: list[bool] = []

        self.fig = plt.figure(figsize=(12.5, 7.2))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(title)
        self.force_ax = self.fig.add_axes((0.08, 0.56, 0.56, 0.35))
        self.proxy_ax = self.fig.add_axes((0.08, 0.14, 0.56, 0.30))
        self.info_ax = self.fig.add_axes((0.70, 0.14, 0.26, 0.77))
        self.info_ax.axis("off")
        self._info_text = self.info_ax.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=9.0,
        )
        (self._normal_force_line,) = self.force_ax.plot(
            [], [], color="tab:blue", linewidth=1.8, label="normal force"
        )
        (self._force_error_line,) = self.force_ax.plot(
            [], [], color="tab:red", linewidth=1.2, alpha=0.75,
            label="force error",
        )
        self._target_force_line = self.force_ax.axhline(
            self.target_normal_force_n,
            color="tab:green",
            linewidth=1.2,
            linestyle="--",
            label="target force",
        )
        self.force_ax.set_xlabel("time [s]")
        self.force_ax.set_ylabel("force [N]")
        self.force_ax.set_title("Wiping normal force")
        self.force_ax.grid(True, alpha=0.25)
        self.force_ax.legend(loc="upper right", fontsize=8)
        (self._proxy_line,) = self.proxy_ax.plot(
            [], [], color="tab:purple", linewidth=1.6,
            label="signed contact proxy",
        )
        (self._penetration_line,) = self.proxy_ax.plot(
            [], [], color="tab:orange", linewidth=1.2, alpha=0.8,
            label="penetration proxy",
        )
        self.proxy_ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
        self.proxy_ax.set_xlabel("time [s]")
        self.proxy_ax.set_ylabel("proxy [mm]")
        self.proxy_ax.set_title("Contact distance / penetration proxy")
        self.proxy_ax.grid(True, alpha=0.25)
        self.proxy_ax.legend(loc="upper right", fontsize=8)

    def update(
        self,
        *,
        time_s: float,
        normal_force_n: float,
        force_error_n: float,
        contact_proxy_m: float,
        phase: str,
        waypoint_index: int,
        contact_source: str,
        in_contact: bool,
        target_normal_force_n: float | None = None,
        redraw: bool = True,
    ) -> WipingForceViewData:
        """Append a diagnostic sample and refresh the live charts."""

        if target_normal_force_n is not None:
            self.target_normal_force_n = float(target_normal_force_n)
        view_data = WipingForceViewData(
            time_s=float(time_s),
            normal_force_n=float(normal_force_n),
            target_normal_force_n=self.target_normal_force_n,
            force_error_n=float(force_error_n),
            contact_proxy_m=float(contact_proxy_m),
            phase=str(phase),
            waypoint_index=int(waypoint_index),
            contact_source=str(contact_source),
            in_contact=bool(in_contact),
        )
        self._append_sample(view_data)
        self._draw_force_chart()
        self._draw_proxy_chart()
        self._info_text.set_text(_format_force_info_text(view_data))
        if redraw:
            self.fig.canvas.draw_idle()
        return view_data

    def show(self, *, block: bool = True) -> None:
        """Show the monitor figure when the active backend supports it."""

        if _is_noninteractive_matplotlib_backend(plt.get_backend()):
            return
        plt.show(block=block)

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

    def _append_sample(self, view_data: WipingForceViewData) -> None:
        self.time_s.append(view_data.time_s)
        self.normal_force_n.append(view_data.normal_force_n)
        self.target_force_n.append(view_data.target_normal_force_n)
        self.force_error_n.append(view_data.force_error_n)
        self.contact_proxy_m.append(view_data.contact_proxy_m)
        self.phase.append(view_data.phase)
        self.waypoint_index.append(view_data.waypoint_index)
        self.contact_source.append(view_data.contact_source)
        self.in_contact.append(view_data.in_contact)
        extra = len(self.time_s) - self.history_points
        if extra <= 0:
            return
        del self.time_s[:extra]
        del self.normal_force_n[:extra]
        del self.target_force_n[:extra]
        del self.force_error_n[:extra]
        del self.contact_proxy_m[:extra]
        del self.phase[:extra]
        del self.waypoint_index[:extra]
        del self.contact_source[:extra]
        del self.in_contact[:extra]

    def _draw_force_chart(self) -> None:
        if not self.time_s:
            return
        time_s = np.asarray(self.time_s, dtype=float)
        normal_force = np.asarray(self.normal_force_n, dtype=float)
        force_error = np.asarray(self.force_error_n, dtype=float)
        self._normal_force_line.set_data(time_s, normal_force)
        self._force_error_line.set_data(time_s, force_error)
        self._target_force_line.set_ydata(
            [self.target_normal_force_n, self.target_normal_force_n]
        )
        lower = min(0.0, float(np.min(normal_force)), float(np.min(force_error)))
        upper = max(
            self.target_normal_force_n,
            float(np.max(normal_force)),
            float(np.max(force_error)),
        )
        margin = max(0.1, 0.12 * max(abs(lower), abs(upper), 1.0))
        self.force_ax.set_ylim(lower - margin, upper + margin)
        self.force_ax.set_xlim(float(time_s[0]), max(float(time_s[-1]), float(time_s[0]) + 1e-6))

    def _draw_proxy_chart(self) -> None:
        if not self.time_s:
            return
        time_s = np.asarray(self.time_s, dtype=float)
        proxy_mm = 1000.0 * np.asarray(self.contact_proxy_m, dtype=float)
        penetration_mm = 1000.0 * np.maximum(
            0.0,
            -np.asarray(self.contact_proxy_m, dtype=float),
        )
        self._proxy_line.set_data(time_s, proxy_mm)
        self._penetration_line.set_data(time_s, penetration_mm)
        lower = min(0.0, float(np.min(proxy_mm)), float(np.min(penetration_mm)))
        upper = max(0.0, float(np.max(proxy_mm)), float(np.max(penetration_mm)))
        margin = max(0.1, 0.12 * max(abs(lower), abs(upper), 1.0))
        self.proxy_ax.set_ylim(lower - margin, upper + margin)
        self.proxy_ax.set_xlim(float(time_s[0]), max(float(time_s[-1]), float(time_s[0]) + 1e-6))


def _format_force_info_text(view_data: WipingForceViewData) -> str:
    contact_state = "yes" if view_data.in_contact else "no"
    return "\n".join(
        [
            f"time_s: {view_data.time_s: .4f}",
            f"phase: {view_data.phase}",
            f"waypoint_index: {view_data.waypoint_index}",
            "",
            f"normal_force_n: {view_data.normal_force_n: .4f}",
            f"target_force_n: {view_data.target_normal_force_n: .4f}",
            f"force_error_n: {view_data.force_error_n: .4f}",
            "",
            f"contact_proxy_m: {view_data.contact_proxy_m: .6f}",
            f"penetration_mm: {1000.0 * max(0.0, -view_data.contact_proxy_m): .3f}",
            f"contact_source: {view_data.contact_source}",
            f"in_contact: {contact_state}",
        ]
    )


def _is_noninteractive_matplotlib_backend(backend_name: str) -> bool:
    """Return True when the active matplotlib backend cannot open a live GUI window."""

    return backend_name.strip().lower() in _NONINTERACTIVE_MATPLOTLIB_BACKENDS


__all__ = [
    "WipingForceMonitorPanel",
    "WipingForceViewData",
]
