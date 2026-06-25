"""M6 executor engine-cleaning task-space controller scaffold.

This module consumes M5 `CleaningWaypoint` objects and executor TCP feedback,
then produces a desired TCP velocity in world coordinates. It intentionally
stops at task-space intent: no Jacobian/qdot conversion, no tendon or motor
commands, no MuJoCo runtime coupling, no visual servo, and no dual-arm
avoidance are implemented here.

Sign convention: `CleaningWaypoint.normal` is the outward surface normal.
The signed gap is `dot(tcp_position - waypoint.position, waypoint.normal)`.
During contact, a positive gap error or low compression force drives velocity
along `-normal`; excessive but safe force drives velocity along `+normal`.
Measured normal force is positive in compression, in Newtons.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    choice_value,
    nonnegative_float,
    positive_float,
    required,
    section,
)
from continuum_sim.control.engine_cleaning_types import (
    EngineCleaningCommand,
    EngineCleaningControllerGains,
    EngineCleaningFeedback,
)
from continuum_sim.tasks.engine_surface_path import CleaningWaypoint


SUPPORTED_CONTROLLER_TYPES = ("engine_cleaning_task_space",)


class EngineCleaningController:
    """Waypoint-following controller that outputs only TCP velocity intent."""

    def __init__(
        self,
        gains: EngineCleaningControllerGains,
        waypoints: list[CleaningWaypoint] | tuple[CleaningWaypoint, ...],
    ) -> None:
        if not waypoints:
            raise ValueError("waypoints must contain at least one CleaningWaypoint.")
        self.gains = gains
        self.waypoints = tuple(waypoints)
        self.active_index = 0
        self.done = False
        self.safety_stop = False
        self.stop_reason: str | None = None

    def reset(self) -> None:
        """Reset controller progress and clear safety state."""

        self.active_index = 0
        self.done = False
        self.safety_stop = False
        self.stop_reason = None

    def current_waypoint(self) -> CleaningWaypoint:
        """Return the active waypoint, clamping to the last waypoint after completion."""

        return self.waypoints[min(self.active_index, len(self.waypoints) - 1)]

    def is_done(self) -> bool:
        """Return whether all waypoints have been completed."""

        return self.done

    def step(self, feedback: EngineCleaningFeedback) -> EngineCleaningCommand:
        """Advance one controller update and return a task-space command."""

        if self.safety_stop:
            return self._stopped_command(feedback, self.current_waypoint(), reached=False)
        command_index = self.active_index
        waypoint = self.current_waypoint()
        if feedback.measured_normal_force_n > self.gains.max_contact_force_n:
            self.safety_stop = True
            self.stop_reason = "max_contact_force_exceeded"
            return self._stopped_command(feedback, waypoint, reached=False)

        if waypoint.phase == "approach":
            velocity, metadata = self._approach_velocity(feedback, waypoint)
            reached = _position_error_norm(feedback, waypoint) <= self.gains.waypoint_tolerance_m
        elif waypoint.phase == "contact":
            velocity, metadata = self._contact_velocity(feedback, waypoint)
            reached = metadata["tangential_error_norm_m"] <= self.gains.waypoint_tolerance_m
        elif waypoint.phase == "retreat":
            velocity, metadata = self._retreat_velocity(feedback, waypoint)
            reached = _position_error_norm(feedback, waypoint) <= self.gains.waypoint_tolerance_m
        else:
            raise ValueError(f"Unsupported waypoint phase {waypoint.phase!r}.")

        velocity = limit_tcp_velocity(
            velocity,
            waypoint.normal,
            max_tcp_speed_mps=self.gains.max_tcp_speed_mps,
            max_normal_speed_mps=self.gains.max_normal_speed_mps,
        )
        if reached:
            self.advance_if_reached(feedback, already_reached=True)
        metadata = {
            **metadata,
            "waypoint_path_index": waypoint.index,
            "done_after_step": self.done,
        }
        return EngineCleaningCommand(
            desired_tcp_velocity_world=velocity,
            active_waypoint_index=command_index,
            phase=waypoint.phase,
            waypoint_reached=reached,
            safety_stop=False,
            stop_reason=None,
            metadata=metadata,
        )

    def advance_if_reached(
        self,
        feedback: EngineCleaningFeedback,
        *,
        already_reached: bool | None = None,
    ) -> bool:
        """Advance to the next waypoint when the active waypoint is reached."""

        if self.safety_stop or self.done:
            return False
        waypoint = self.current_waypoint()
        reached = (
            self._is_waypoint_reached(feedback, waypoint)
            if already_reached is None
            else already_reached
        )
        if not reached:
            return False
        if self.active_index >= len(self.waypoints) - 1:
            self.done = True
        else:
            self.active_index += 1
        return True

    def _is_waypoint_reached(
        self,
        feedback: EngineCleaningFeedback,
        waypoint: CleaningWaypoint,
    ) -> bool:
        if waypoint.phase == "contact":
            return (
                _tangential_error(feedback, waypoint)[1]
                <= self.gains.waypoint_tolerance_m
            )
        return _position_error_norm(feedback, waypoint) <= self.gains.waypoint_tolerance_m

    def _approach_velocity(
        self,
        feedback: EngineCleaningFeedback,
        waypoint: CleaningWaypoint,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        error = waypoint.position - feedback.tcp_pose.position
        return self.gains.approach_position_gain * error, {
            "position_error": error,
            "position_error_norm_m": float(np.linalg.norm(error)),
        }

    def _retreat_velocity(
        self,
        feedback: EngineCleaningFeedback,
        waypoint: CleaningWaypoint,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        error = waypoint.position - feedback.tcp_pose.position
        return self.gains.retreat_position_gain * error, {
            "position_error": error,
            "position_error_norm_m": float(np.linalg.norm(error)),
        }

    def _contact_velocity(
        self,
        feedback: EngineCleaningFeedback,
        waypoint: CleaningWaypoint,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        tangent_error, tangent_error_norm = _tangential_error(feedback, waypoint)
        tangent_velocity = self.gains.tangential_position_gain * tangent_error
        signed_gap = float(np.dot(feedback.tcp_pose.position - waypoint.position, waypoint.normal))
        gap_error = signed_gap - waypoint.standoff_distance_m
        force_error = waypoint.target_force_n - feedback.measured_normal_force_n
        if abs(force_error) <= self.gains.force_deadband_n:
            force_error = 0.0
        normal_scalar = (
            -self.gains.normal_position_gain * gap_error
            - self.gains.normal_force_gain * force_error
        )
        normal_velocity = normal_scalar * waypoint.normal
        return tangent_velocity + normal_velocity, {
            "tangent_velocity": tangent_velocity,
            "normal_velocity": normal_velocity,
            "tangential_error": tangent_error,
            "tangential_error_norm_m": tangent_error_norm,
            "signed_gap_m": signed_gap,
            "gap_error_m": gap_error,
            "force_error_n": force_error,
            "normal_velocity_scalar_mps": float(normal_scalar),
        }

    def _stopped_command(
        self,
        feedback: EngineCleaningFeedback,
        waypoint: CleaningWaypoint,
        *,
        reached: bool,
    ) -> EngineCleaningCommand:
        del feedback
        return EngineCleaningCommand(
            desired_tcp_velocity_world=np.zeros(3, dtype=float),
            active_waypoint_index=self.active_index,
            phase=waypoint.phase,
            waypoint_reached=reached,
            safety_stop=True,
            stop_reason=self.stop_reason,
            metadata={"waypoint_path_index": waypoint.index},
        )


def load_engine_cleaning_controller_config(path: str) -> dict[str, Any]:
    """Load the standalone M6 controller YAML and return the controller section."""

    raw = load_yaml(path)
    return section(raw, "controller")


def validate_engine_cleaning_controller_config(config: dict[str, Any]) -> None:
    """Validate controller type and gain fields."""

    choice_value(required(config, "type"), "controller.type", SUPPORTED_CONTROLLER_TYPES)
    gains = section(config, "gains")
    for name in (
        "tangential_position_gain",
        "normal_position_gain",
        "normal_force_gain",
        "approach_position_gain",
        "retreat_position_gain",
        "waypoint_tolerance_m",
        "force_deadband_n",
        "min_clearance_m",
    ):
        nonnegative_float(gains, name)
    for name in ("max_tcp_speed_mps", "max_normal_speed_mps", "max_contact_force_n"):
        positive_float(gains, name)


def build_engine_cleaning_gains_from_config(
    config: dict[str, Any],
) -> EngineCleaningControllerGains:
    """Build validated controller gains from a loaded controller config."""

    validate_engine_cleaning_controller_config(config)
    gains = section(config, "gains")
    return EngineCleaningControllerGains(
        tangential_position_gain=float(required(gains, "tangential_position_gain")),
        normal_position_gain=float(required(gains, "normal_position_gain")),
        normal_force_gain=float(required(gains, "normal_force_gain")),
        approach_position_gain=float(required(gains, "approach_position_gain")),
        retreat_position_gain=float(required(gains, "retreat_position_gain")),
        max_tcp_speed_mps=float(required(gains, "max_tcp_speed_mps")),
        max_normal_speed_mps=float(required(gains, "max_normal_speed_mps")),
        waypoint_tolerance_m=float(required(gains, "waypoint_tolerance_m")),
        max_contact_force_n=float(required(gains, "max_contact_force_n")),
        force_deadband_n=float(required(gains, "force_deadband_n")),
        min_clearance_m=float(gains.get("min_clearance_m", 0.0)),
    )


def limit_tcp_velocity(
    velocity: np.ndarray,
    normal: np.ndarray,
    *,
    max_tcp_speed_mps: float,
    max_normal_speed_mps: float,
) -> np.ndarray:
    """Limit total TCP speed and the component along the surface normal."""

    result = np.asarray(velocity, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"Expected velocity with shape (3,), got {result.shape}.")
    normal_unit = _normalize(normal, "normal")
    normal_scalar = float(np.dot(result, normal_unit))
    clipped_normal_scalar = float(
        np.clip(normal_scalar, -max_normal_speed_mps, max_normal_speed_mps)
    )
    result = result + (clipped_normal_scalar - normal_scalar) * normal_unit
    speed = float(np.linalg.norm(result))
    if speed > max_tcp_speed_mps:
        result = result * (max_tcp_speed_mps / speed)
    return result


def _position_error_norm(
    feedback: EngineCleaningFeedback,
    waypoint: CleaningWaypoint,
) -> float:
    return float(np.linalg.norm(waypoint.position - feedback.tcp_pose.position))


def _tangential_error(
    feedback: EngineCleaningFeedback,
    waypoint: CleaningWaypoint,
) -> tuple[np.ndarray, float]:
    error = waypoint.position - feedback.tcp_pose.position
    normal = _normalize(waypoint.normal, "waypoint.normal")
    tangent_error = error - np.dot(error, normal) * normal
    return tangent_error, float(np.linalg.norm(tangent_error))


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected {name} with shape (3,), got {array.shape}.")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must have non-zero length.")
    return array / norm
