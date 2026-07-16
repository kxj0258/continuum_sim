"""Mappings between PCC state q and tendon length deltas."""

from __future__ import annotations

import numpy as np

from continuum_sim.model.robot_params import PCC_VALUES_PER_SEGMENT, ThreeSegmentRobotParams


def _as_segment_q(q: np.ndarray, params: ThreeSegmentRobotParams) -> np.ndarray:
    q_array = np.asarray(q, dtype=float)
    if q_array.shape != (params.q_size,):
        raise ValueError(f"Expected q with shape ({params.q_size},), got {q_array.shape}.")
    return q_array.reshape(params.segment_count, PCC_VALUES_PER_SEGMENT)


def _as_segment_delta(tendon_delta: np.ndarray, params: ThreeSegmentRobotParams) -> np.ndarray:
    delta_array = np.asarray(tendon_delta, dtype=float)
    if delta_array.shape != (params.tendon_count,):
        raise ValueError(
            f"Expected tendon_delta with shape ({params.tendon_count},), got {delta_array.shape}."
        )
    return delta_array.reshape(params.segment_count, -1)


def segment_q_to_tendon_delta(
    q_segment: np.ndarray,
    length: float,
    radius: float,
    angles_rad: np.ndarray,
) -> np.ndarray:
    """Map one segment state [kx, ky, eps] to three tendon length deltas."""
    kx, ky, eps = np.asarray(q_segment, dtype=float)
    return length * eps - length * radius * (
        kx * np.cos(angles_rad) + ky * np.sin(angles_rad)
    )


def segment_tendon_delta_to_q(
    tendon_delta_segment: np.ndarray,
    length: float,
    radius: float,
    angles_rad: np.ndarray,
) -> np.ndarray:
    """Least-squares inverse of the one-segment tendon mapping."""
    tendon_delta_segment = np.asarray(tendon_delta_segment, dtype=float)
    design = np.column_stack(
        (
            -length * radius * np.cos(angles_rad),
            -length * radius * np.sin(angles_rad),
            np.full(angles_rad.shape, length, dtype=float),
        )
    )
    q_segment, *_ = np.linalg.lstsq(design, tendon_delta_segment, rcond=None)
    return q_segment


def q_to_tendon_delta(
    q: np.ndarray,
    params: ThreeSegmentRobotParams | None = None,
) -> np.ndarray:
    """Map a PCC state vector to segment-local tendon length deltas.

    The state ordering is [kx1, ky1, eps1, kx2, ky2, eps2, kx3, ky3, eps3].
    The output ordering is segment-major: three tendon deltas per segment.
    """
    params = params or ThreeSegmentRobotParams.default()
    q_segments = _as_segment_q(q, params)
    deltas = []
    for q_segment, segment in zip(q_segments, params.segments, strict=True):
        deltas.append(
            segment_q_to_tendon_delta(
                q_segment=q_segment,
                length=segment.effective_flexure_length,
                radius=segment.tendon_radius,
                angles_rad=segment.routing.angles_rad,
            )
        )
    return np.concatenate(deltas)


def tendon_delta_to_q(
    tendon_delta: np.ndarray,
    params: ThreeSegmentRobotParams | None = None,
) -> np.ndarray:
    """Map segment-local tendon length deltas back to the PCC state vector."""
    params = params or ThreeSegmentRobotParams.default()
    delta_segments = _as_segment_delta(tendon_delta, params)
    q_segments = []
    for delta_segment, segment in zip(delta_segments, params.segments, strict=True):
        q_segments.append(
            segment_tendon_delta_to_q(
                tendon_delta_segment=delta_segment,
                length=segment.effective_flexure_length,
                radius=segment.tendon_radius,
                angles_rad=segment.routing.angles_rad,
            )
        )
    return np.concatenate(q_segments)
