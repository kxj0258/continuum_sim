"""Rate-limited direct tendon-length target integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from continuum_sim.control.cbf_qp_kinematics import CBFQPConfig, solve_cbf_qp_velocity
from continuum_sim.model.bending_space import BendingSpaceModel


DEFAULT_TARGET_LEAD_M = 0.0005
_BENDING_LIMIT_PROJECTION_CONFIG = CBFQPConfig(max_projection_iterations=32)
TENDON_TARGET_MODES = ("protected", "free_integrated", "actual_anchored")


def resolve_tendon_target_policy(
    metadata: Mapping[str, object],
) -> tuple[bool, str]:
    """Resolve explicit target mode first, then legacy boolean metadata."""

    explicit_mode = metadata.get("backend_tendon_target_mode")
    if explicit_mode is not None:
        mode = str(explicit_mode)
        if mode not in TENDON_TARGET_MODES:
            raise ValueError(f"backend_tendon_target_mode must be one of {TENDON_TARGET_MODES}.")
        return True, mode
    if "enforce_backend_tendon_limits" in metadata:
        enforce_limits = bool(metadata["enforce_backend_tendon_limits"])
    elif "disable_backend_tendon_limits" in metadata:
        enforce_limits = not bool(metadata["disable_backend_tendon_limits"])
    else:
        enforce_limits = True
    return enforce_limits, ("protected" if enforce_limits else "actual_anchored")


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
