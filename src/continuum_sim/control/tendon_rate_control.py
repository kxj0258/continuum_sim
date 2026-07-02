"""Rate-limited direct tendon-length target integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.model.bending_space import BendingSpaceModel

@dataclass(frozen=True)
class TendonRateLimits:
    """Per-tendon displacement and rate limits."""

    displacement_min_m: np.ndarray
    displacement_max_m: np.ndarray
    max_rate_mps: np.ndarray

    def __post_init__(self) -> None:
        lower = _as_vector(self.displacement_min_m, "displacement_min_m")
        upper = _as_vector(self.displacement_max_m, "displacement_max_m", lower.size)
        max_rate = _as_vector(self.max_rate_mps, "max_rate_mps", lower.size)
        if np.any(lower >= upper):
            raise ValueError("Tendon displacement lower limits must be below upper limits.")
        if np.any(max_rate <= 0.0):
            raise ValueError("Tendon max rates must be positive.")
        object.__setattr__(self, "displacement_min_m", lower)
        object.__setattr__(self, "displacement_max_m", upper)
        object.__setattr__(self, "max_rate_mps", max_rate)


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
    ) -> CompatibleTendonRateStep:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        requested = _as_vector(
            requested_rate_mps,
            "requested_rate_mps",
            self.model.tendon_count,
        )
        if raw_debug:
            return self._step_raw(requested, float(dt))
        return self._step_compatible(requested, float(dt))

    def _step_compatible(
        self,
        requested: np.ndarray,
        dt: float,
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
        current = self.model.to_tendon(self._bending)
        rate_scale = _common_rate_scale(compatible_rate, self.limits.max_rate_mps)
        displacement_scale = _common_displacement_scale(
            current,
            compatible_rate,
            dt,
            self.limits.displacement_min_m,
            self.limits.displacement_max_m,
        )
        scale = min(rate_scale, displacement_scale)
        applied_bending_rate = scale * bending_rate
        applied_rate = self.model.to_tendon(applied_bending_rate)
        self._bending = self._bending + dt * applied_bending_rate
        displacement = self.model.to_tendon(self._bending)
        return CompatibleTendonRateStep(
            requested_rate_mps=requested.copy(),
            applied_rate_mps=applied_rate,
            bending_rate=applied_bending_rate,
            bending_displacement=self._bending.copy(),
            displacement_m=displacement,
            common_scale=float(scale),
            compatibility_residual_mps=residual,
            rate_saturated=np.abs(applied_rate - compatible_rate) > 1.0e-15,
            displacement_saturated=np.full(
                self.model.tendon_count,
                displacement_scale < rate_scale,
                dtype=bool,
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


def _as_vector(values: np.ndarray, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or (size is not None and result.shape != (size,)):
        expected = "a 1D vector" if size is None else f"shape ({size},)"
        raise ValueError(f"{name} must have {expected}, got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result.copy()
