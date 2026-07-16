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
    segment then link. Only the physical flexure axis for each link is driven
    for the current Y/X/Y/X structure. Axial strain ``eps`` is intentionally
    ignored because the MJCF has no axial prismatic DOFs.
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
        axes = segment.flexure_joint_axes
        if len(axes) != links_per_segment:
            axes = tuple(axes[index % len(axes)] for index in range(links_per_segment))
        x_count = max(1, axes.count("x"))
        y_count = max(1, axes.count("y"))
        x_angle = -ky * segment.effective_flexure_length / float(x_count)
        y_angle = kx * segment.effective_flexure_length / float(y_count)
        for link_index, axis in enumerate(axes):
            if axis == "x":
                targets[segment_index, link_index, 0] = x_angle
            elif axis == "y":
                targets[segment_index, link_index, 1] = y_angle
            else:
                raise ValueError(f"Unsupported flexure joint axis {axis!r}.")

    return targets.reshape(-1)
