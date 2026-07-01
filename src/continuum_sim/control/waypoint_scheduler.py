"""Waypoint advancement policies for scenario controllers."""

from __future__ import annotations

from dataclasses import dataclass


WAYPOINT_ADVANCE_MODES = ("tolerance", "time")


@dataclass
class WaypointScheduler:
    """Advance waypoint indices by tolerance or fixed controller-step cadence."""

    waypoint_count: int
    mode: str
    tolerance_m: float
    loop: bool
    controller_dt_s: float
    step_interval: int | None = None
    time_interval_s: float | None = None
    active_index: int = 0
    done: bool = False
    _updates: int = 0

    def __post_init__(self) -> None:
        if self.waypoint_count <= 0:
            raise ValueError("waypoint_count must be positive.")
        if self.mode not in WAYPOINT_ADVANCE_MODES:
            raise ValueError(f"mode must be one of {WAYPOINT_ADVANCE_MODES}.")
        if self.tolerance_m < 0.0:
            raise ValueError("tolerance_m must be non-negative.")
        if self.controller_dt_s <= 0.0:
            raise ValueError("controller_dt_s must be positive.")
        if self.step_interval is None and self.time_interval_s is not None:
            self.step_interval = max(1, int(round(self.time_interval_s / self.controller_dt_s)))
        if self.mode == "time" and self.step_interval is None:
            self.step_interval = 1
        if self.step_interval is not None and self.step_interval <= 0:
            raise ValueError("step_interval must be positive.")

    def update(self, *, error_norm_m: float) -> int:
        """Update scheduler state and return the active waypoint index."""

        if self.done:
            return self.active_index
        if self.mode == "tolerance":
            if error_norm_m <= self.tolerance_m:
                self.advance()
            return self.active_index
        self._updates += 1
        if self._updates >= int(self.step_interval):
            self._updates = 0
            self.advance()
        return self.active_index

    def advance(self) -> None:
        """Advance one waypoint, respecting loop and completion settings."""

        if self.active_index < self.waypoint_count - 1:
            self.active_index += 1
        elif self.loop:
            self.active_index = 0
        else:
            self.done = True
