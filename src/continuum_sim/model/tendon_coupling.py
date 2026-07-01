"""Coupling between physical tendon length deltas and PCC state q."""

from __future__ import annotations

import numpy as np

from typing import Protocol
from continuum_sim.model.robot_params import PCC_VALUES_PER_SEGMENT, ThreeSegmentRobotParams


class TendonPathLike(Protocol):
    id: str
    global_index: int
    angle_deg: float
    radial_offset: float
    path_segment_indices: tuple[int, ...]


def build_coupling_matrix(
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[TendonPathLike, ...],
) -> np.ndarray:
    """Build the matrix mapping PCC q to physical tendon length deltas."""
    tendon_count = len(physical_tendons)
    if tendon_count <= 0:
        raise ValueError("Expected at least one physical tendon.")
    expected_indices = list(range(tendon_count))
    global_indices = sorted(tendon.global_index for tendon in physical_tendons)
    if global_indices != expected_indices:
        raise ValueError(
            "Expected physical tendon global_index values to cover "
            f"0..{tendon_count - 1}, got {global_indices}."
        )

    C = np.zeros((tendon_count, params.q_size), dtype=float)
    for tendon in physical_tendons:
        theta = np.deg2rad(tendon.angle_deg)
        for segment_index in tendon.path_segment_indices:
            if segment_index < 0 or segment_index >= len(params.segments):
                raise ValueError(
                    f"{tendon.id} references invalid segment index {segment_index}."
                )
            length = params.segments[segment_index].length
            column = PCC_VALUES_PER_SEGMENT * segment_index
            C[tendon.global_index, column : column + PCC_VALUES_PER_SEGMENT] = np.array(
                [
                    -length * tendon.radial_offset * np.cos(theta),
                    -length * tendon.radial_offset * np.sin(theta),
                    length,
                ],
                dtype=float,
            )
    return C


def q_to_physical_tendon_delta(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[TendonPathLike, ...],
) -> np.ndarray:
    """Map PCC state q to physical tendon length deltas."""
    q_array = _as_vector(q, "q", expected_size=params.q_size)
    return build_coupling_matrix(params, physical_tendons) @ q_array


def physical_tendon_delta_to_q(
    tendon_delta: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[TendonPathLike, ...],
    regularization: float = 0.0,
) -> np.ndarray:
    """Estimate PCC state q from physical tendon length deltas."""
    delta_array = _as_vector(
        tendon_delta,
        "tendon_delta",
        expected_size=len(physical_tendons),
    )
    C = build_coupling_matrix(params, physical_tendons)

    if regularization < 0.0:
        raise ValueError(f"regularization must be non-negative, got {regularization}.")
    if regularization > 0.0:
        lhs = C.T @ C + regularization * np.eye(C.shape[1], dtype=float)
        rhs = C.T @ delta_array
        return np.linalg.solve(lhs, rhs)
    return np.linalg.pinv(C) @ delta_array


def coupling_diagnostics(C: np.ndarray) -> dict[str, float | int | bool]:
    """Return numerical diagnostics for a coupling matrix."""
    C_array = np.asarray(C, dtype=float)
    if C_array.ndim != 2:
        raise ValueError(f"Expected C to be 2D, got shape {C_array.shape}.")

    rank = int(np.linalg.matrix_rank(C_array))
    return {
        "rank": rank,
        "condition_number": float(np.linalg.cond(C_array)),
        "is_full_rank": rank == C_array.shape[1],
    }


def _as_vector(values: np.ndarray, name: str, *, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"Expected {name} with shape ({expected_size},), got {array.shape}.")
    return array
