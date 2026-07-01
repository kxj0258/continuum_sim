"""Rate-limited direct tendon-length target integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def _as_vector(values: np.ndarray, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or (size is not None and result.shape != (size,)):
        expected = "a 1D vector" if size is None else f"shape ({size},)"
        raise ValueError(f"{name} must have {expected}, got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result.copy()

