"""Map PCC states onto the reduced-order MuJoCo joint target layout."""

from __future__ import annotations

import numpy as np

from continuum_sim.model import ThreeSegmentRobotParams


def pcc_q_to_joint_targets(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    links_per_segment: int,
) -> np.ndarray:
    """Return MuJoCo position-actuator targets for a 9D PCC state.

    ``q`` is ordered as ``[kx1, ky1, eps1, kx2, ky2, eps2, kx3, ky3, eps3]``.
    The reduced-order model has x/y hinge pairs for each link, ordered by
    segment then link. Axial strain ``eps`` is intentionally ignored in this
    first reduced-order mapping because the MJCF has no axial prismatic DOFs.
    """

    q_array = np.asarray(q, dtype=float)
    if q_array.shape != (9,):
        raise ValueError(f"Expected q with shape (9,), got {q_array.shape}.")
    if links_per_segment <= 0:
        raise ValueError(f"links_per_segment must be positive, got {links_per_segment}.")

    q_segments = q_array.reshape(3, 3)
    targets = np.zeros((3, links_per_segment, 2), dtype=float)
    for segment_index, (segment_q, segment) in enumerate(
        zip(q_segments, params.segments, strict=True)
    ):
        kx, ky, _eps = segment_q
        per_link_angle = segment.length / float(links_per_segment)
        targets[segment_index, :, 0] = -ky * per_link_angle
        targets[segment_index, :, 1] = kx * per_link_angle

    return targets.reshape(-1)
