"""Rate-limited direct tendon-length target integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.control.cbf_qp_kinematics import CBFQPConfig, solve_cbf_qp_velocity
from continuum_sim.model.bending_space import BendingSpaceModel


DEFAULT_TARGET_LEAD_M = 0.0005
_BENDING_LIMIT_PROJECTION_CONFIG = CBFQPConfig(max_projection_iterations=32)
_BENDING_LIMIT_FEASIBILITY_TOLERANCE = (
    _BENDING_LIMIT_PROJECTION_CONFIG.tolerance
)
TENDON_TARGET_MODES = ("protected", "free_integrated", "actual_anchored")
TENDON_INNER_LOOP_MODES = ("legacy", "bending_rate_servo")
ZERO_COMMAND_MODES = ("hold", "relax")


@dataclass(frozen=True)
class TendonRateLimits:
    """Per-tendon displacement and rate limits."""

    displacement_min_m: np.ndarray
    displacement_max_m: np.ndarray
    max_rate_mps: np.ndarray
    target_lead_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        lower = _as_vector(self.displacement_min_m, "displacement_min_m")
        upper = _as_vector(self.displacement_max_m, "displacement_max_m", lower.size)
        max_rate = _as_vector(self.max_rate_mps, "max_rate_mps", lower.size)
        target_lead = (
            np.full(lower.shape, DEFAULT_TARGET_LEAD_M, dtype=float)
            if self.target_lead_m is None
            else _as_vector_or_scalar(self.target_lead_m, "target_lead_m", lower.size)
        )
        if np.any(lower >= upper):
            raise ValueError("Tendon displacement lower limits must be below upper limits.")
        if np.any(max_rate <= 0.0):
            raise ValueError("Tendon max rates must be positive.")
        if np.any(target_lead <= 0.0):
            raise ValueError("Tendon target lead limits must be positive.")
        object.__setattr__(self, "displacement_min_m", lower)
        object.__setattr__(self, "displacement_max_m", upper)
        object.__setattr__(self, "max_rate_mps", max_rate)
        object.__setattr__(self, "target_lead_m", target_lead)


@dataclass(frozen=True)
class TendonRateStep:
    """One saturated integration result."""

    requested_rate_mps: np.ndarray
    applied_rate_mps: np.ndarray
    displacement_m: np.ndarray
    rate_saturated: np.ndarray
    displacement_saturated: np.ndarray


class TendonRateIntegrator:
    """Integrate tendon-length rates without accumulating beyond target limits."""

    def __init__(
        self,
        limits: TendonRateLimits,
        initial_displacement_m: np.ndarray | None = None,
    ) -> None:
        self.limits = limits
        self._displacement_m = np.zeros_like(limits.max_rate_mps)
        self.reset(initial_displacement_m)

    @property
    def displacement_m(self) -> np.ndarray:
        return self._displacement_m.copy()

    def reset(self, displacement_m: np.ndarray | None = None) -> np.ndarray:
        values = (
            np.zeros_like(self.limits.max_rate_mps)
            if displacement_m is None
            else _as_vector(
                displacement_m,
                "displacement_m",
                self.limits.max_rate_mps.size,
            )
        )
        self._displacement_m = np.clip(
            values,
            self.limits.displacement_min_m,
            self.limits.displacement_max_m,
        )
        return self.displacement_m

    def step(self, requested_rate_mps: np.ndarray, dt: float) -> TendonRateStep:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        requested = _as_vector(
            requested_rate_mps,
            "requested_rate_mps",
            self.limits.max_rate_mps.size,
        )
        rate = np.clip(
            requested,
            -self.limits.max_rate_mps,
            self.limits.max_rate_mps,
        )
        unconstrained = self._displacement_m + float(dt) * rate
        displacement = np.clip(
            unconstrained,
            self.limits.displacement_min_m,
            self.limits.displacement_max_m,
        )
        applied_rate = (displacement - self._displacement_m) / float(dt)
        result = TendonRateStep(
            requested_rate_mps=requested.copy(),
            applied_rate_mps=applied_rate,
            displacement_m=displacement.copy(),
            rate_saturated=np.abs(rate - requested) > 1.0e-15,
            displacement_saturated=np.abs(displacement - unconstrained) > 1.0e-15,
        )
        self._displacement_m = displacement
        return result


@dataclass(frozen=True)
class CompatibleTendonRateStep:
    """One compatible or explicitly raw integration result."""

    requested_rate_mps: np.ndarray
    applied_rate_mps: np.ndarray
    bending_rate: np.ndarray
    bending_displacement: np.ndarray
    displacement_m: np.ndarray
    common_scale: float
    compatibility_residual_mps: np.ndarray
    rate_saturated: np.ndarray
    displacement_saturated: np.ndarray
    raw_debug: bool


@dataclass(frozen=True)
class BendingRateServoConfig:
    """Configuration for the bending-space tendon rate servo."""

    rate_filter_time_constant_s: float = 0.06
    feedforward_lead_time_s: float = 0.0
    rate_proportional_time_s: float = 0.0
    rate_integral_gain: float = 1.0
    anti_windup_gain: float = 1.0
    enforce_target_lead_limit: bool = True
    max_target_lead_m: float | np.ndarray | None = None
    soft_force_limit_n: float | None = None
    hard_force_limit_n: float | None = None
    zero_command_mode: str = "hold"
    zero_rate_tolerance_mps: float = 1.0e-7

    def __post_init__(self) -> None:
        nonnegative = {
            "rate_filter_time_constant_s": self.rate_filter_time_constant_s,
            "feedforward_lead_time_s": self.feedforward_lead_time_s,
            "rate_proportional_time_s": self.rate_proportional_time_s,
            "rate_integral_gain": self.rate_integral_gain,
            "anti_windup_gain": self.anti_windup_gain,
            "zero_rate_tolerance_mps": self.zero_rate_tolerance_mps,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.anti_windup_gain > 1.0:
            raise ValueError("anti_windup_gain must be in [0, 1].")
        if not isinstance(self.enforce_target_lead_limit, bool):
            raise ValueError("enforce_target_lead_limit must be a boolean.")
        if self.max_target_lead_m is not None:
            lead = np.asarray(self.max_target_lead_m, dtype=float)
            if not np.all(np.isfinite(lead)) or np.any(lead <= 0.0):
                raise ValueError("max_target_lead_m must contain positive finite values.")
            object.__setattr__(self, "max_target_lead_m", lead.copy())
        for name, value in (
            ("soft_force_limit_n", self.soft_force_limit_n),
            ("hard_force_limit_n", self.hard_force_limit_n),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite when set.")
        if (
            self.soft_force_limit_n is not None
            and self.hard_force_limit_n is not None
            and self.soft_force_limit_n > self.hard_force_limit_n
        ):
            raise ValueError("soft_force_limit_n cannot exceed hard_force_limit_n.")
        if self.zero_command_mode not in ZERO_COMMAND_MODES:
            raise ValueError(
                f"zero_command_mode must be one of {ZERO_COMMAND_MODES}."
            )


@dataclass(frozen=True)
class CompatibleBendingRateServoStep:
    """One bending-space rate-servo result and its anti-windup diagnostics."""

    requested_rate_mps: np.ndarray
    compatible_rate_mps: np.ndarray
    constrained_rate_mps: np.ndarray
    measured_rate_mps: np.ndarray
    measured_rate_raw_mps: np.ndarray
    target_rate_mps: np.ndarray
    raw_target_m: np.ndarray
    displacement_m: np.ndarray
    target_lead_m: np.ndarray
    target_lead_limit_m: np.ndarray
    bending_requested_rate: np.ndarray
    bending_constrained_rate: np.ndarray
    bending_measured_rate: np.ndarray
    bending_rate_error: np.ndarray
    bending_integral: np.ndarray
    anti_windup_correction: np.ndarray
    common_scale: float
    compatibility_residual_mps: np.ndarray
    rate_saturated: np.ndarray
    target_rate_saturated: np.ndarray
    displacement_saturated: np.ndarray
    lead_saturated: np.ndarray
    force_saturated: np.ndarray
    hard_force_saturated: np.ndarray
    force_scale: float
    guard_feasible: bool
    max_constraint_violation_m: float
    compatibility_bypassed_for_safety: bool
    hold_target_retained: bool


class CompatibleBendingRateServo:
    """Track compatible tendon rates using measured bending-rate feedback.

    The actuator target is anchored to the measured tendon displacement.  A
    persistent bending-space integral stores only the lead required to close
    the rate error, so motion that MuJoCo could not realize in one controller
    period is retained instead of being discarded.  Tracking targets are
    projected through compatible tendon bounds.  Cross-cycle slew limits apply
    to the actuator lead rather than the absolute target, so measured tendon
    motion can re-anchor the target without making the compatible projection
    infeasible.  HOLD preserves the last full target, while an infeasible guard
    actively unwinds the lead toward the measured displacement.
    """

    def __init__(
        self,
        model: BendingSpaceModel,
        limits: TendonRateLimits,
        config: BendingRateServoConfig | None = None,
    ) -> None:
        if limits.max_rate_mps.shape != (model.tendon_count,):
            raise ValueError("Bending model and tendon limits must have matching sizes.")
        self.model = model
        self.limits = limits
        self.config = BendingRateServoConfig() if config is None else config
        configured_lead = (
            np.full(model.tendon_count, np.inf, dtype=float)
            if not self.config.enforce_target_lead_limit
            else (
                limits.target_lead_m
                if self.config.max_target_lead_m is None
                else _as_vector_or_scalar(
                    self.config.max_target_lead_m,
                    "max_target_lead_m",
                    model.tendon_count,
                )
            )
        )
        self._target_lead_limit_m = (
            configured_lead
            if not self.config.enforce_target_lead_limit
            else np.minimum(limits.target_lead_m, configured_lead)
        )
        self._integral_bending = np.zeros(model.bending_size, dtype=float)
        self._filtered_bending_rate = np.zeros(model.bending_size, dtype=float)
        self._previous_actual_bending = np.zeros(model.bending_size, dtype=float)
        self._target_m = np.zeros(model.tendon_count, dtype=float)
        self._previous_target_m = np.zeros(model.tendon_count, dtype=float)
        self._previous_lead_m = np.zeros(model.tendon_count, dtype=float)
        self._has_actual_observation = False
        self.reset()

    @property
    def displacement_m(self) -> np.ndarray:
        return self._target_m.copy()

    @property
    def bending_integral(self) -> np.ndarray:
        return self._integral_bending.copy()

    @property
    def target_lead_limit_m(self) -> np.ndarray:
        return self._target_lead_limit_m.copy()

    def reset(
        self,
        actual_displacement_m: np.ndarray | None = None,
        *,
        retained_target_m: np.ndarray | None = None,
    ) -> np.ndarray:
        actual = (
            np.zeros(self.model.tendon_count, dtype=float)
            if actual_displacement_m is None
            else _as_vector(
                actual_displacement_m,
                "actual_displacement_m",
                self.model.tendon_count,
            )
        )
        target = (
            actual
            if retained_target_m is None
            else _as_vector(
                retained_target_m,
                "retained_target_m",
                self.model.tendon_count,
            )
        )
        self._integral_bending.fill(0.0)
        self._filtered_bending_rate.fill(0.0)
        self._previous_actual_bending = self.model.estimate(actual)
        self._target_m = target.copy()
        self._previous_target_m = target.copy()
        self._previous_lead_m = target - actual
        self._has_actual_observation = actual_displacement_m is not None
        return self.displacement_m

    def step(
        self,
        requested_rate_mps: np.ndarray,
        dt: float,
        *,
        actual_displacement_m: np.ndarray,
        actuator_force_n: np.ndarray | None = None,
    ) -> CompatibleBendingRateServoStep:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        requested = _as_vector(
            requested_rate_mps,
            "requested_rate_mps",
            self.model.tendon_count,
        )
        actual = _as_vector(
            actual_displacement_m,
            "actual_displacement_m",
            self.model.tendon_count,
        )
        force = (
            np.zeros(self.model.tendon_count, dtype=float)
            if actuator_force_n is None
            else _as_vector(
                actuator_force_n,
                "actuator_force_n",
                self.model.tendon_count,
            )
        )
        residual = self.model.residual(requested)
        if not self.model.is_compatible(requested):
            raise ValueError(
                "Normal tendon-rate command is outside bending space: "
                f"residual={np.linalg.norm(residual):.6e} m/s, "
                f"tolerance={self.model.compatibility_tolerance(requested):.6e} m/s."
            )

        bending_requested = self.model.estimate(requested)
        compatible_rate = self.model.to_tendon(bending_requested)
        actual_bending = self.model.estimate(actual)
        if self._has_actual_observation:
            measured_bending_raw = (
                actual_bending - self._previous_actual_bending
            ) / float(dt)
        else:
            measured_bending_raw = np.zeros(self.model.bending_size, dtype=float)
            self._has_actual_observation = True
        filter_tau = self.config.rate_filter_time_constant_s
        filter_alpha = 1.0 if filter_tau <= 0.0 else float(dt) / (filter_tau + float(dt))
        self._filtered_bending_rate = self._filtered_bending_rate + filter_alpha * (
            measured_bending_raw - self._filtered_bending_rate
        )

        (
            bending_constrained,
            rate_constraint_violation,
            rate_guard_feasible,
        ) = _project_bending_rate_checked(
            self.model,
            bending_requested,
            actual,
            float(dt),
            self.limits.displacement_min_m,
            self.limits.displacement_max_m,
            self.limits.max_rate_mps,
            actual,
            self._target_lead_limit_m,
        )
        constrained_rate = self.model.to_tendon(bending_constrained)
        measured_rate_raw = self.model.to_tendon(measured_bending_raw)
        measured_rate = self.model.to_tendon(self._filtered_bending_rate)
        bending_rate_error = bending_constrained - self._filtered_bending_rate

        soft_force_saturated = (
            np.zeros(self.model.tendon_count, dtype=bool)
            if self.config.soft_force_limit_n is None
            else np.abs(force) >= self.config.soft_force_limit_n
        )
        hard_force_saturated = (
            np.zeros(self.model.tendon_count, dtype=bool)
            if self.config.hard_force_limit_n is None
            else np.abs(force) >= self.config.hard_force_limit_n
        )
        request_is_zero = (
            np.max(np.abs(compatible_rate)) <= self.config.zero_rate_tolerance_mps
        )
        relax_requested = (
            self.config.zero_command_mode == "relax" and request_is_zero
        )
        if relax_requested:
            integral_candidate = np.zeros_like(self._integral_bending)
        elif np.any(soft_force_saturated):
            integral_candidate = self._integral_bending.copy()
        elif request_is_zero:
            integral_candidate = self._integral_bending.copy()
        else:
            integral_candidate = (
                self._integral_bending + float(dt) * bending_rate_error
            )

        feedforward_lead = (
            self.config.feedforward_lead_time_s * bending_constrained
        )
        proportional_lead = (
            self.config.rate_proportional_time_s * bending_rate_error
        )
        integral_lead = self.config.rate_integral_gain * integral_candidate
        raw_lead_bending = feedforward_lead + proportional_lead + integral_lead
        if relax_requested:
            feedforward_lead.fill(0.0)
            proportional_lead.fill(0.0)
            raw_lead_bending.fill(0.0)
        elif self.config.zero_command_mode == "hold" and request_is_zero:
            feedforward_lead.fill(0.0)
            proportional_lead.fill(0.0)
            raw_lead_bending = self.model.estimate(
                self._previous_target_m - actual
            )

        lead_lower = np.maximum(
            self.limits.displacement_min_m - actual,
            -self._target_lead_limit_m,
        )
        lead_lower = np.maximum(
            lead_lower,
            self._previous_lead_m - float(dt) * self.limits.max_rate_mps,
        )
        lead_upper = np.minimum(
            self.limits.displacement_max_m - actual,
            self._target_lead_limit_m,
        )
        lead_upper = np.minimum(
            lead_upper,
            self._previous_lead_m + float(dt) * self.limits.max_rate_mps,
        )
        raw_lead = self.model.to_tendon(raw_lead_bending)
        raw_target = actual + raw_lead
        peak_force = float(np.max(np.abs(force)))
        force_scale = 1.0
        hard_force_recovery = (
            self.config.hard_force_limit_n is not None
            and peak_force >= self.config.hard_force_limit_n
        )
        hold_requested = (
            self.config.zero_command_mode == "hold" and request_is_zero
        )
        retained_lead = self._previous_target_m - actual
        hold_target_retained = bool(
            hold_requested
            and not hard_force_recovery
            and np.all(
                self._previous_target_m >= self.limits.displacement_min_m
            )
            and np.all(
                self._previous_target_m <= self.limits.displacement_max_m
            )
            and np.all(np.abs(retained_lead) <= self._target_lead_limit_m)
        )
        compatibility_bypassed_for_safety = False
        if hold_target_retained:
            projected_lead = retained_lead.copy()
            projected_lead_bending = self.model.estimate(projected_lead)
            target = self._previous_target_m.copy()
            raw_lead = retained_lead.copy()
            raw_target = target.copy()
            guard_feasible = True
            max_constraint_violation = 0.0
        else:
            (
                projected_lead_bending,
                lead_constraint_violation,
                lead_guard_feasible,
            ) = _project_with_tendon_bounds_checked(
                self.model,
                raw_lead_bending,
                lead_lower,
                lead_upper,
            )
            guard_feasible = lead_guard_feasible and rate_guard_feasible
            max_constraint_violation = max(
                lead_constraint_violation,
                rate_constraint_violation * float(dt),
            )
            projected_lead = self.model.to_tendon(projected_lead_bending)
            target = actual + projected_lead
        if hard_force_recovery:
            recovery_force = (
                0.9 * self.config.hard_force_limit_n
                if self.config.soft_force_limit_n is None
                else self.config.soft_force_limit_n
            )
            force_scale = float(recovery_force / peak_force)
            previous_guard_feasible = guard_feasible
            previous_constraint_violation = max_constraint_violation
            (
                projected_lead_bending,
                recovery_constraint_violation,
                recovery_guard_feasible,
            ) = _project_with_tendon_bounds_checked(
                self.model,
                force_scale * projected_lead_bending,
                lead_lower,
                lead_upper,
            )
            guard_feasible = (
                previous_guard_feasible
                and recovery_guard_feasible
                and rate_guard_feasible
            )
            max_constraint_violation = max(
                previous_constraint_violation,
                recovery_constraint_violation,
                rate_constraint_violation * float(dt),
            )
            projected_lead = self.model.to_tendon(projected_lead_bending)
            target = actual + projected_lead
        if not guard_feasible:
            safe_lead_lower = np.maximum(
                self.limits.displacement_min_m - actual,
                -self._target_lead_limit_m,
            )
            safe_lead_upper = np.minimum(
                self.limits.displacement_max_m - actual,
                self._target_lead_limit_m,
            )
            rate_lead_lower = (
                self._previous_lead_m - float(dt) * self.limits.max_rate_mps
            )
            rate_lead_upper = (
                self._previous_lead_m + float(dt) * self.limits.max_rate_mps
            )
            bounded_lead_lower = np.maximum(
                safe_lead_lower,
                rate_lead_lower,
            )
            bounded_lead_upper = np.minimum(
                safe_lead_upper,
                rate_lead_upper,
            )
            safe_box_feasible = safe_lead_lower <= safe_lead_upper
            bounded_box_feasible = safe_box_feasible & (
                bounded_lead_lower <= bounded_lead_upper
            )
            projected_lead = np.zeros(self.model.tendon_count, dtype=float)
            if np.any(bounded_box_feasible):
                projected_lead[bounded_box_feasible] = np.clip(
                    0.0,
                    bounded_lead_lower[bounded_box_feasible],
                    bounded_lead_upper[bounded_box_feasible],
                )
            safety_only = safe_box_feasible & ~bounded_box_feasible
            if np.any(safety_only):
                # Lead/force safety takes precedence where actual tendon motion
                # makes the cross-cycle lead-slew box temporarily infeasible.
                projected_lead[safety_only] = np.clip(
                    0.0,
                    safe_lead_lower[safety_only],
                    safe_lead_upper[safety_only],
                )
            target = actual + projected_lead
            projected_lead_bending = self.model.estimate(projected_lead)
            lead_rate_violation_m = float(
                np.max(
                    np.maximum(
                        np.abs(projected_lead - self._previous_lead_m)
                        - float(dt) * self.limits.max_rate_mps,
                        0.0,
                    )
                )
            )
            max_constraint_violation = max(
                max_constraint_violation,
                lead_rate_violation_m,
                _maximum_box_violation(
                    target,
                    self.limits.displacement_min_m,
                    self.limits.displacement_max_m,
                ),
                _maximum_box_violation(
                    projected_lead,
                    -self._target_lead_limit_m,
                    self._target_lead_limit_m,
                ),
            )
            compatibility_bypassed_for_safety = not self.model.is_compatible(
                projected_lead
            )
        anti_windup_correction = np.zeros_like(self._integral_bending)
        if relax_requested:
            self._integral_bending.fill(0.0)
        elif self.config.rate_integral_gain > 0.0:
            desired_integral = (
                projected_lead_bending - feedforward_lead - proportional_lead
            ) / self.config.rate_integral_gain
            anti_windup_correction = self.config.anti_windup_gain * (
                desired_integral - integral_candidate
            )
            self._integral_bending = (
                integral_candidate + anti_windup_correction
            )
        else:
            self._integral_bending.fill(0.0)

        target_rate = (target - self._previous_target_m) / float(dt)
        raw_target_rate = (raw_target - self._previous_target_m) / float(dt)
        tolerance = 1.0e-12
        rate_saturated = np.abs(constrained_rate - compatible_rate) > tolerance
        target_rate_saturated = np.abs(target_rate - raw_target_rate) > tolerance
        displacement_saturated = (
            target <= self.limits.displacement_min_m + tolerance
        ) | (target >= self.limits.displacement_max_m - tolerance)
        lead_saturated = (
            np.abs(projected_lead - raw_lead) > tolerance
        ) | (
            np.abs(projected_lead) >= self._target_lead_limit_m - tolerance
        )
        result = CompatibleBendingRateServoStep(
            requested_rate_mps=requested.copy(),
            compatible_rate_mps=compatible_rate,
            constrained_rate_mps=constrained_rate,
            measured_rate_mps=measured_rate,
            measured_rate_raw_mps=measured_rate_raw,
            target_rate_mps=target_rate,
            raw_target_m=raw_target,
            displacement_m=target.copy(),
            target_lead_m=projected_lead,
            target_lead_limit_m=self._target_lead_limit_m.copy(),
            bending_requested_rate=bending_requested,
            bending_constrained_rate=bending_constrained,
            bending_measured_rate=self._filtered_bending_rate.copy(),
            bending_rate_error=bending_rate_error,
            bending_integral=self._integral_bending.copy(),
            anti_windup_correction=anti_windup_correction,
            common_scale=_effective_scale(compatible_rate, constrained_rate),
            compatibility_residual_mps=residual,
            rate_saturated=rate_saturated,
            target_rate_saturated=target_rate_saturated,
            displacement_saturated=displacement_saturated,
            lead_saturated=lead_saturated,
            force_saturated=soft_force_saturated | hard_force_saturated,
            hard_force_saturated=hard_force_saturated,
            force_scale=force_scale,
            guard_feasible=guard_feasible,
            max_constraint_violation_m=max_constraint_violation,
            compatibility_bypassed_for_safety=compatibility_bypassed_for_safety,
            hold_target_retained=hold_target_retained,
        )
        self._previous_actual_bending = actual_bending
        self._previous_target_m = target.copy()
        self._previous_lead_m = projected_lead.copy()
        self._target_m = target.copy()
        return result


class CompatibleTendonRateIntegrator:
    """Integrate bending-compatible tendon targets with common-scale limits."""

    def __init__(
        self,
        model: BendingSpaceModel,
        limits: TendonRateLimits,
    ) -> None:
        if limits.max_rate_mps.shape != (model.tendon_count,):
            raise ValueError("Bending model and tendon limits must have matching sizes.")
        self.model = model
        self.limits = limits
        self._bending = np.zeros(model.bending_size, dtype=float)
        self._raw_target = np.zeros(model.tendon_count, dtype=float)
        self._raw_mode = False

    @property
    def displacement_m(self) -> np.ndarray:
        if self._raw_mode:
            return self._raw_target.copy()
        return self.model.to_tendon(self._bending)

    @property
    def bending_displacement(self) -> np.ndarray:
        return self.model.estimate(self.displacement_m)

    def reset(self, displacement_m: np.ndarray | None = None) -> np.ndarray:
        values = (
            np.zeros(self.model.tendon_count, dtype=float)
            if displacement_m is None
            else _as_vector(displacement_m, "displacement_m", self.model.tendon_count)
        )
        values = np.clip(
            values,
            self.limits.displacement_min_m,
            self.limits.displacement_max_m,
        )
        self._bending = self.model.estimate(values)
        self._raw_target = values.copy()
        self._raw_mode = False
        return self.displacement_m

    def reset_raw(self, displacement_m: np.ndarray) -> np.ndarray:
        """Rebase raw-debug integration without creating a target jump."""

        values = _as_vector(
            displacement_m,
            "displacement_m",
            self.model.tendon_count,
        )
        values = np.clip(
            values,
            self.limits.displacement_min_m,
            self.limits.displacement_max_m,
        )
        self._bending = self.model.estimate(values)
        self._raw_target = values.copy()
        self._raw_mode = True
        return self.displacement_m

    def step(
        self,
        requested_rate_mps: np.ndarray,
        dt: float,
        *,
        raw_debug: bool = False,
        actual_displacement_m: np.ndarray | None = None,
        enforce_limits: bool = True,
        target_mode: str = "protected",
    ) -> CompatibleTendonRateStep:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if target_mode not in TENDON_TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TENDON_TARGET_MODES}.")
        requested = _as_vector(
            requested_rate_mps,
            "requested_rate_mps",
            self.model.tendon_count,
        )
        if raw_debug:
            return self._step_raw(requested, float(dt))
        actual = (
            None
            if actual_displacement_m is None
            else _as_vector(
                actual_displacement_m,
                "actual_displacement_m",
                self.model.tendon_count,
            )
        )
        mode = target_mode if enforce_limits else "actual_anchored"
        return self._step_compatible(requested, float(dt), actual, mode)

    def _step_compatible(
        self,
        requested: np.ndarray,
        dt: float,
        actual_displacement_m: np.ndarray | None,
        target_mode: str,
    ) -> CompatibleTendonRateStep:
        if self._raw_mode:
            self._bending = self.model.estimate(self._raw_target)
            self._raw_mode = False
        residual = self.model.residual(requested)
        if not self.model.is_compatible(requested):
            raise ValueError(
                "Normal tendon-rate command is outside bending space: "
                f"residual={np.linalg.norm(residual):.6e} m/s, "
                f"tolerance={self.model.compatibility_tolerance(requested):.6e} m/s."
            )
        bending_rate = self.model.estimate(requested)
        compatible_rate = self.model.to_tendon(bending_rate)
        if target_mode == "actual_anchored":
            anchor = (
                self.displacement_m
                if actual_displacement_m is None
                else actual_displacement_m
            )
            displacement = anchor + dt * compatible_rate
            self._raw_target = displacement.copy()
            self._bending = self.model.estimate(displacement)
            self._raw_mode = True
            return CompatibleTendonRateStep(
                requested_rate_mps=requested.copy(),
                applied_rate_mps=compatible_rate,
                bending_rate=bending_rate,
                bending_displacement=self._bending.copy(),
                displacement_m=displacement.copy(),
                common_scale=1.0,
                compatibility_residual_mps=residual,
                rate_saturated=np.zeros(self.model.tendon_count, dtype=bool),
                displacement_saturated=np.zeros(self.model.tendon_count, dtype=bool),
                raw_debug=False,
            )
        if target_mode == "free_integrated":
            self._bending = self._bending + dt * bending_rate
            displacement = self.model.to_tendon(self._bending)
            return CompatibleTendonRateStep(
                requested_rate_mps=requested.copy(),
                applied_rate_mps=compatible_rate,
                bending_rate=bending_rate,
                bending_displacement=self._bending.copy(),
                displacement_m=displacement,
                common_scale=1.0,
                compatibility_residual_mps=residual,
                rate_saturated=np.zeros(self.model.tendon_count, dtype=bool),
                displacement_saturated=np.zeros(self.model.tendon_count, dtype=bool),
                raw_debug=False,
            )
        if actual_displacement_m is not None:
            self._bending = _project_bending_displacement(
                self.model,
                self._bending,
                self.limits.displacement_min_m,
                self.limits.displacement_max_m,
                actual_displacement_m,
                self.limits.target_lead_m,
            )
        current = self.model.to_tendon(self._bending)
        applied_bending_rate = _project_bending_rate(
            self.model,
            bending_rate,
            current,
            dt,
            self.limits.displacement_min_m,
            self.limits.displacement_max_m,
            self.limits.max_rate_mps,
            actual_displacement_m,
            self.limits.target_lead_m,
        )
        applied_rate = self.model.to_tendon(applied_bending_rate)
        self._bending = self._bending + dt * applied_bending_rate
        displacement = self.model.to_tendon(self._bending)
        effective_scale = _effective_scale(compatible_rate, applied_rate)
        return CompatibleTendonRateStep(
            requested_rate_mps=requested.copy(),
            applied_rate_mps=applied_rate,
            bending_rate=applied_bending_rate,
            bending_displacement=self._bending.copy(),
            displacement_m=displacement,
            common_scale=effective_scale,
            compatibility_residual_mps=residual,
            rate_saturated=np.abs(applied_rate - compatible_rate) > 1.0e-15,
            displacement_saturated=_displacement_saturated(
                displacement,
                self.limits.displacement_min_m,
                self.limits.displacement_max_m,
                actual_displacement_m,
                self.limits.target_lead_m,
            ),
            raw_debug=False,
        )

    def _step_raw(
        self,
        requested: np.ndarray,
        dt: float,
    ) -> CompatibleTendonRateStep:
        if not self._raw_mode:
            self._raw_target = self.model.to_tendon(self._bending)
            self._raw_mode = True
        rate = np.clip(requested, -self.limits.max_rate_mps, self.limits.max_rate_mps)
        unconstrained = self._raw_target + dt * rate
        displacement = np.clip(
            unconstrained,
            self.limits.displacement_min_m,
            self.limits.displacement_max_m,
        )
        applied_rate = (displacement - self._raw_target) / dt
        self._raw_target = displacement
        bending = self.model.estimate(displacement)
        return CompatibleTendonRateStep(
            requested_rate_mps=requested.copy(),
            applied_rate_mps=applied_rate,
            bending_rate=self.model.estimate(applied_rate),
            bending_displacement=bending,
            displacement_m=displacement.copy(),
            common_scale=float("nan"),
            compatibility_residual_mps=self.model.residual(applied_rate),
            rate_saturated=np.abs(rate - requested) > 1.0e-15,
            displacement_saturated=np.abs(displacement - unconstrained) > 1.0e-15,
            raw_debug=True,
        )


def _common_rate_scale(rate: np.ndarray, limit: np.ndarray) -> float:
    ratios = np.divide(
        limit,
        np.abs(rate),
        out=np.full_like(limit, np.inf),
        where=np.abs(rate) > 0.0,
    )
    return float(min(1.0, np.min(ratios)))


def _common_displacement_scale(
    current: np.ndarray,
    rate: np.ndarray,
    dt: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    scales = np.ones_like(rate)
    positive = rate > 0.0
    negative = rate < 0.0
    scales[positive] = (upper[positive] - current[positive]) / (
        dt * rate[positive]
    )
    scales[negative] = (lower[negative] - current[negative]) / (
        dt * rate[negative]
    )
    return float(np.clip(np.min(scales), 0.0, 1.0))


def _project_bending_displacement(
    model: BendingSpaceModel,
    bending: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
    target_lead: np.ndarray,
) -> np.ndarray:
    finite_lead = np.isfinite(target_lead)
    if not np.any(finite_lead):
        return bending
    lead_lower = lower.copy()
    lead_upper = upper.copy()
    lead_lower[finite_lead] = np.maximum(
        lead_lower[finite_lead],
        actual[finite_lead] - target_lead[finite_lead],
    )
    lead_upper[finite_lead] = np.minimum(
        lead_upper[finite_lead],
        actual[finite_lead] + target_lead[finite_lead],
    )
    lead_lower, lead_upper = _ensure_feasible_bounds(
        lead_lower,
        lead_upper,
        actual,
    )
    return _project_with_tendon_bounds(
        model,
        bending,
        lead_lower,
        lead_upper,
    )


def _project_bending_rate(
    model: BendingSpaceModel,
    bending_rate: np.ndarray,
    current: np.ndarray,
    dt: float,
    lower: np.ndarray,
    upper: np.ndarray,
    max_rate: np.ndarray,
    actual: np.ndarray | None,
    target_lead: np.ndarray,
) -> np.ndarray:
    rate_lower, rate_upper = _bending_rate_bounds(
        current,
        dt,
        lower,
        upper,
        max_rate,
        actual,
        target_lead,
    )
    rate_lower, rate_upper = _ensure_feasible_bounds(
        rate_lower,
        rate_upper,
        np.zeros_like(rate_lower),
    )
    return _project_with_tendon_bounds(
        model,
        bending_rate,
        rate_lower,
        rate_upper,
    )


def _project_with_tendon_bounds(
    model: BendingSpaceModel,
    reference_bending: np.ndarray,
    tendon_lower: np.ndarray,
    tendon_upper: np.ndarray,
) -> np.ndarray:
    coupling = model.coupling_matrix
    barrier_jacobian = np.vstack((coupling, -coupling))
    barrier_lower_bound = np.concatenate((tendon_lower, -tendon_upper))
    return solve_cbf_qp_velocity(
        reference_bending,
        barrier_jacobian=barrier_jacobian,
        barrier_lower_bound=barrier_lower_bound,
        config=_BENDING_LIMIT_PROJECTION_CONFIG,
    )


def _project_bending_rate_checked(
    model: BendingSpaceModel,
    bending_rate: np.ndarray,
    current: np.ndarray,
    dt: float,
    lower: np.ndarray,
    upper: np.ndarray,
    max_rate: np.ndarray,
    actual: np.ndarray | None,
    target_lead: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    rate_lower, rate_upper = _bending_rate_bounds(
        current,
        dt,
        lower,
        upper,
        max_rate,
        actual,
        target_lead,
    )
    return _project_with_tendon_bounds_checked(
        model,
        bending_rate,
        rate_lower,
        rate_upper,
    )


def _bending_rate_bounds(
    current: np.ndarray,
    dt: float,
    lower: np.ndarray,
    upper: np.ndarray,
    max_rate: np.ndarray,
    actual: np.ndarray | None,
    target_lead: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rate_lower = np.maximum(-max_rate, (lower - current) / dt)
    rate_upper = np.minimum(max_rate, (upper - current) / dt)
    if actual is not None:
        finite_lead = np.isfinite(target_lead)
        rate_lower[finite_lead] = np.maximum(
            rate_lower[finite_lead],
            (actual[finite_lead] - target_lead[finite_lead] - current[finite_lead])
            / dt,
        )
        rate_upper[finite_lead] = np.minimum(
            rate_upper[finite_lead],
            (actual[finite_lead] + target_lead[finite_lead] - current[finite_lead])
            / dt,
        )
    return rate_lower, rate_upper


def _project_with_tendon_bounds_checked(
    model: BendingSpaceModel,
    reference_bending: np.ndarray,
    tendon_lower: np.ndarray,
    tendon_upper: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Project a bending vector and report whether its tendon box is feasible."""

    tolerance = _BENDING_LIMIT_FEASIBILITY_TOLERANCE
    if np.any(tendon_lower > tendon_upper + tolerance):
        zero = np.zeros(model.bending_size, dtype=float)
        violation = _maximum_box_violation(
            model.to_tendon(zero),
            tendon_lower,
            tendon_upper,
        )
        return zero, violation, False
    tiny_conflict = tendon_lower > tendon_upper
    if np.any(tiny_conflict):
        midpoint = 0.5 * (
            tendon_lower[tiny_conflict] + tendon_upper[tiny_conflict]
        )
        tendon_lower = tendon_lower.copy()
        tendon_upper = tendon_upper.copy()
        tendon_lower[tiny_conflict] = midpoint
        tendon_upper[tiny_conflict] = midpoint
    projected = _project_with_tendon_bounds(
        model,
        reference_bending,
        tendon_lower,
        tendon_upper,
    )
    projected_tendon = model.to_tendon(projected)
    violation = _maximum_box_violation(
        projected_tendon,
        tendon_lower,
        tendon_upper,
    )
    if violation <= tolerance:
        return projected, violation, True

    zero_is_feasible = bool(
        np.all(tendon_lower <= tolerance) and np.all(tendon_upper >= -tolerance)
    )
    if zero_is_feasible:
        reference_tendon = model.to_tendon(reference_bending)
        scale = _common_box_scale_from_zero(
            reference_tendon,
            tendon_lower,
            tendon_upper,
        )
        fallback = scale * reference_bending
        fallback_violation = _maximum_box_violation(
            model.to_tendon(fallback),
            tendon_lower,
            tendon_upper,
        )
        return fallback, fallback_violation, fallback_violation <= tolerance

    zero = np.zeros(model.bending_size, dtype=float)
    zero_violation = _maximum_box_violation(
        model.to_tendon(zero),
        tendon_lower,
        tendon_upper,
    )
    return zero, zero_violation, False


def _common_box_scale_from_zero(
    reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    scale = 1.0
    positive = reference > 0.0
    negative = reference < 0.0
    if np.any(positive):
        scale = min(scale, float(np.min(upper[positive] / reference[positive])))
    if np.any(negative):
        scale = min(scale, float(np.min(lower[negative] / reference[negative])))
    return float(np.clip(scale, 0.0, 1.0))


def _maximum_box_violation(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
    ):
        return float("inf")
    return float(
        max(
            0.0,
            float(np.max(lower - values)),
            float(np.max(values - upper)),
        )
    )


def _ensure_feasible_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixed = lower > upper
    if not np.any(fixed):
        return lower, upper
    midpoint = np.clip(fallback[fixed], upper[fixed], lower[fixed])
    lower = lower.copy()
    upper = upper.copy()
    lower[fixed] = midpoint
    upper[fixed] = midpoint
    return lower, upper


def _effective_scale(reference_rate: np.ndarray, applied_rate: np.ndarray) -> float:
    denom = float(reference_rate @ reference_rate)
    if denom <= 1.0e-30:
        return 1.0
    return float(np.clip((applied_rate @ reference_rate) / denom, 0.0, 1.0))


def _displacement_saturated(
    displacement: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray | None,
    target_lead: np.ndarray,
) -> np.ndarray:
    tolerance = 1.0e-12
    saturated = (displacement <= lower + tolerance) | (
        displacement >= upper - tolerance
    )
    if actual is not None:
        finite_lead = np.isfinite(target_lead)
        saturated[finite_lead] |= (
            displacement[finite_lead]
            <= actual[finite_lead] - target_lead[finite_lead] + tolerance
        ) | (
            displacement[finite_lead]
            >= actual[finite_lead] + target_lead[finite_lead] - tolerance
        )
    return saturated


def _as_vector(values: np.ndarray, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or (size is not None and result.shape != (size,)):
        expected = "a 1D vector" if size is None else f"shape ({size},)"
        raise ValueError(f"{name} must have {expected}, got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result.copy()


def _as_vector_or_scalar(values: np.ndarray, name: str, size: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim == 0:
        result = np.full((size,), float(result), dtype=float)
    elif result.shape != (size,):
        raise ValueError(f"{name} must be a scalar or have shape ({size},), got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result.copy()
