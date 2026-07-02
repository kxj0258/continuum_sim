"""Bending-only coordinates for physically compatible tendon control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.model.robot_params import (
    PCC_VALUES_PER_SEGMENT,
    ThreeSegmentRobotParams,
)
from continuum_sim.model.tendon_coupling import TendonPathLike, build_coupling_matrix


@dataclass(frozen=True)
class BendingSpaceModel:
    """Map segment bending coordinates to the compatible tendon subspace."""

    selection_matrix: np.ndarray
    coupling_matrix: np.ndarray
    pseudoinverse: np.ndarray
    rank: int
    condition_number: float
    absolute_tolerance_m: float = 1.0e-9
    relative_tolerance: float = 1.0e-6

    @classmethod
    def from_arm(
        cls,
        params: ThreeSegmentRobotParams,
        physical_tendons: tuple[TendonPathLike, ...],
        *,
        absolute_tolerance_m: float = 1.0e-9,
        relative_tolerance: float = 1.0e-6,
    ) -> "BendingSpaceModel":
        if absolute_tolerance_m < 0.0 or relative_tolerance < 0.0:
            raise ValueError("Compatibility tolerances must be non-negative.")
        selection = np.zeros((params.q_size, 2 * params.segment_count), dtype=float)
        for segment_index in range(params.segment_count):
            q_start = PCC_VALUES_PER_SEGMENT * segment_index
            bending_start = 2 * segment_index
            selection[q_start : q_start + 2, bending_start : bending_start + 2] = (
                np.eye(2, dtype=float)
            )
        coupling = build_coupling_matrix(params, physical_tendons) @ selection
        rank = int(np.linalg.matrix_rank(coupling))
        if rank != coupling.shape[1]:
            raise ValueError(
                "Bending coupling matrix must have full column rank: "
                f"expected {coupling.shape[1]}, got {rank}."
            )
        return cls(
            selection_matrix=selection,
            coupling_matrix=coupling,
            pseudoinverse=np.linalg.pinv(coupling),
            rank=rank,
            condition_number=float(np.linalg.cond(coupling)),
            absolute_tolerance_m=float(absolute_tolerance_m),
            relative_tolerance=float(relative_tolerance),
        )

    @property
    def bending_size(self) -> int:
        return self.coupling_matrix.shape[1]

    @property
    def tendon_count(self) -> int:
        return self.coupling_matrix.shape[0]

    def to_q(self, bending: np.ndarray) -> np.ndarray:
        return self.selection_matrix @ self._bending_vector(bending)

    def to_tendon(self, bending: np.ndarray) -> np.ndarray:
        return self.coupling_matrix @ self._bending_vector(bending)

    def estimate(self, tendon_delta: np.ndarray) -> np.ndarray:
        return self.pseudoinverse @ self._tendon_vector(tendon_delta)

    def project(self, tendon_delta: np.ndarray) -> np.ndarray:
        values = self._tendon_vector(tendon_delta)
        return self.coupling_matrix @ (self.pseudoinverse @ values)

    def residual(self, tendon_delta: np.ndarray) -> np.ndarray:
        values = self._tendon_vector(tendon_delta)
        return values - self.project(values)

    def residual_norm(self, tendon_delta: np.ndarray) -> float:
        return float(np.linalg.norm(self.residual(tendon_delta)))

    def compatibility_tolerance(self, tendon_delta: np.ndarray) -> float:
        values = self._tendon_vector(tendon_delta)
        return self.absolute_tolerance_m + self.relative_tolerance * float(
            np.linalg.norm(values)
        )

    def is_compatible(self, tendon_delta: np.ndarray) -> bool:
        return self.residual_norm(tendon_delta) <= self.compatibility_tolerance(
            tendon_delta
        )

    def _bending_vector(self, values: np.ndarray) -> np.ndarray:
        return _finite_vector(values, self.bending_size, "bending")

    def _tendon_vector(self, values: np.ndarray) -> np.ndarray:
        return _finite_vector(values, self.tendon_count, "tendon_delta")


def _finite_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return result
