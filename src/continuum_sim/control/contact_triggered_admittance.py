"""Contact-triggered admittance logic for wiping force-position tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ContactTriggeredAdmittanceConfig:
    """Parameters for contact-triggered force-position waypoint tracking."""

    target_normal_force_n: float
    contact_force_threshold_n: float = 0.1
    tangent_tolerance_m: float = 1.0e-3
    force_tolerance_n: float = 0.08
    stable_steps_required: int = 1
    max_steps_per_target: int = 100
    position_gain: float = 10.0
    kp_force: float = 0.5
    ki_force: float = 0.012
    admittance_mass: float = 1.0
    admittance_damping: float = 20.0
    admittance_stiffness: float = 5.0
    admittance_clip_m: float = 0.012
    force_deadband_n: float = 0.03
    force_filter_alpha: float = 0.1
    max_tangent_velocity_m_s: float = 0.012
    max_normal_velocity_m_s: float = 0.010
    enforce_velocity_limits: bool = False

    def __post_init__(self) -> None:
        _validate_nonnegative("target_normal_force_n", self.target_normal_force_n)
        _validate_nonnegative("contact_force_threshold_n", self.contact_force_threshold_n)
        _validate_nonnegative("tangent_tolerance_m", self.tangent_tolerance_m)
        _validate_nonnegative("force_tolerance_n", self.force_tolerance_n)
        if self.stable_steps_required <= 0:
            raise ValueError("stable_steps_required must be positive.")
        if self.max_steps_per_target <= 0:
            raise ValueError("max_steps_per_target must be positive.")
        _validate_nonnegative("position_gain", self.position_gain)
        _validate_nonnegative("kp_force", self.kp_force)
        _validate_nonnegative("ki_force", self.ki_force)
        _validate_positive("admittance_mass", self.admittance_mass)
        _validate_nonnegative("admittance_damping", self.admittance_damping)
        _validate_nonnegative("admittance_stiffness", self.admittance_stiffness)
        _validate_nonnegative("admittance_clip_m", self.admittance_clip_m)
        _validate_nonnegative("force_deadband_n", self.force_deadband_n)
        if not 0.0 < self.force_filter_alpha <= 1.0:
            raise ValueError("force_filter_alpha must be in (0, 1].")
        _validate_positive("max_tangent_velocity_m_s", self.max_tangent_velocity_m_s)
        _validate_positive("max_normal_velocity_m_s", self.max_normal_velocity_m_s)


@dataclass(frozen=True)
class ContactTriggeredAdmittanceCommand:
    """One update produced by ContactTriggeredAdmittanceTracker."""

    target_index: int
    target_position: np.ndarray
    corrected_target_position: np.ndarray
    desired_velocity: np.ndarray
    target_normal_force_n: float
    filtered_force_n: float
    force_error_n: float
    tangent_error_m: float
    raw_normal_error_m: float
    corrected_error_m: float
    admittance_position_m: float
    admittance_velocity_m_s: float
    contact_active: bool
    waypoint_advanced: bool
    advance_reason: str


@dataclass
class ContactTriggeredAdmittanceTracker:
    """Stateful waypoint and admittance tracker independent of MuJoCo."""

    config: ContactTriggeredAdmittanceConfig
    target_index: int = 0
    contact_active: bool = False
    hold_steps: int = 0
    stable_steps: int = 0
    filtered_force_n: float = 0.0
    force_integral_n_s: float = 0.0
    admittance_position_m: float = 0.0
    admittance_velocity_m_s: float = 0.0
    _last_command: ContactTriggeredAdmittanceCommand | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def last_command(self) -> ContactTriggeredAdmittanceCommand | None:
        return self._last_command

    def reset(self, *, target_index: int = 0) -> None:
        if target_index < 0:
            raise ValueError("target_index must be non-negative.")
        self.target_index = int(target_index)
        self.contact_active = False
        self.hold_steps = 0
        self.stable_steps = 0
        self.filtered_force_n = 0.0
        self.force_integral_n_s = 0.0
        self.admittance_position_m = 0.0
        self.admittance_velocity_m_s = 0.0
        self._last_command = None

    def step(
        self,
        *,
        tip_position: np.ndarray,
        target_positions: np.ndarray,
        normal: np.ndarray,
        measured_normal_force_n: float,
        dt: float,
    ) -> ContactTriggeredAdmittanceCommand:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        targets = _target_array(target_positions)
        if self.target_index >= len(targets):
            self.target_index = len(targets) - 1
        tip = _vector3(tip_position, "tip_position")
        normal_unit = _unit_vector(normal, "normal")
        target = targets[self.target_index]
        command_target_index = self.target_index
        command_target = target.copy()
        measured_force = max(0.0, float(measured_normal_force_n))

        self.filtered_force_n = (
            self.config.force_filter_alpha * measured_force
            + (1.0 - self.config.force_filter_alpha) * self.filtered_force_n
        )
        if (
            not self.contact_active
            and measured_force >= self.config.contact_force_threshold_n
        ):
            self.contact_active = True
            self.hold_steps = 0
            self.stable_steps = 0

        target_force = (
            self.config.target_normal_force_n if self.contact_active else 0.0
        )
        if self.contact_active:
            force_error_for_admittance = target_force - self.filtered_force_n
            if abs(force_error_for_admittance) < self.config.force_deadband_n:
                force_error_for_admittance = 0.0
            self.force_integral_n_s += force_error_for_admittance * dt
            force_drive = (
                self.config.kp_force * force_error_for_admittance
                + self.config.ki_force * self.force_integral_n_s
            )
            admittance_acc = (
                -force_drive
                - self.config.admittance_damping * self.admittance_velocity_m_s
                - self.config.admittance_stiffness * self.admittance_position_m
            ) / self.config.admittance_mass
            self.admittance_velocity_m_s += admittance_acc * dt
            self.admittance_position_m += self.admittance_velocity_m_s * dt
            self.admittance_position_m = float(
                np.clip(
                    self.admittance_position_m,
                    -self.config.admittance_clip_m,
                    self.config.admittance_clip_m,
                )
            )
            if (
                abs(self.admittance_position_m) >= self.config.admittance_clip_m
                and np.sign(self.admittance_velocity_m_s)
                == np.sign(self.admittance_position_m)
            ):
                self.admittance_velocity_m_s = 0.0

        corrected_target = target + self.admittance_position_m * normal_unit
        raw_error = tip - target
        raw_normal_error = float(np.dot(raw_error, normal_unit))
        tangent_error_vec = raw_error - raw_normal_error * normal_unit
        corrected_error_vec = tip - corrected_target
        corrected_normal_error = float(np.dot(corrected_error_vec, normal_unit))
        corrected_tangent_error = corrected_error_vec - corrected_normal_error * normal_unit
        tangent_velocity = -self.config.position_gain * corrected_tangent_error
        normal_velocity = -self.config.position_gain * corrected_normal_error * normal_unit
        if self.config.enforce_velocity_limits:
            tangent_velocity = _clip_norm(
                tangent_velocity,
                self.config.max_tangent_velocity_m_s,
            )
            normal_velocity = _clip_norm(
                normal_velocity,
                self.config.max_normal_velocity_m_s,
            )
        desired_velocity = tangent_velocity + normal_velocity

        force_error = target_force - measured_force
        waypoint_advanced = False
        advance_reason = ""
        if self.contact_active:
            self.hold_steps += 1
            tangent_ok = (
                float(np.linalg.norm(tangent_error_vec))
                <= self.config.tangent_tolerance_m
            )
            force_ok = abs(force_error) <= self.config.force_tolerance_n
            if tangent_ok and force_ok:
                self.stable_steps += 1
            else:
                self.stable_steps = 0
            reached = self.stable_steps >= self.config.stable_steps_required
            timed_out = self.hold_steps >= self.config.max_steps_per_target
            if reached or timed_out:
                if self.target_index < len(targets) - 1:
                    self.target_index += 1
                    waypoint_advanced = True
                    advance_reason = "stable" if reached else "max_steps"
                    self.hold_steps = 0
                    self.stable_steps = 0
                elif timed_out:
                    advance_reason = "final_target_max_steps"

        command = ContactTriggeredAdmittanceCommand(
            target_index=command_target_index,
            target_position=command_target,
            corrected_target_position=corrected_target.copy(),
            desired_velocity=desired_velocity,
            target_normal_force_n=float(target_force),
            filtered_force_n=float(self.filtered_force_n),
            force_error_n=float(force_error),
            tangent_error_m=float(np.linalg.norm(tangent_error_vec)),
            raw_normal_error_m=float(raw_normal_error),
            corrected_error_m=float(np.linalg.norm(corrected_error_vec)),
            admittance_position_m=float(self.admittance_position_m),
            admittance_velocity_m_s=float(self.admittance_velocity_m_s),
            contact_active=bool(self.contact_active),
            waypoint_advanced=bool(waypoint_advanced),
            advance_reason=advance_reason,
        )
        self._last_command = command
        return command


def _target_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(
            "target_positions must have shape (n, 3) with at least one target."
        )
    return array


def _vector3(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    return array


def _unit_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = _vector3(values, name)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return vector / norm


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 0.0:
        return np.asarray(vector, dtype=float)
    return np.asarray(vector, dtype=float) * (max_norm / norm)


def _validate_nonnegative(name: str, value: float) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def _validate_positive(name: str, value: float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
