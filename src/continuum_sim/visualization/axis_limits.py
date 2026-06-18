"""Shared axis-limit helpers for matplotlib viewers."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from continuum_sim.model.robot_params import ThreeSegmentRobotParams

AxisLimits: TypeAlias = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def default_robot_axis_limits(params: ThreeSegmentRobotParams) -> AxisLimits:
    """Return a stable plotting box large enough for the default three-segment arm."""
    total_length = float(np.sum(params.segment_lengths))
    lateral_limit = max(total_length * 1.15, 0.01)
    z_upper = max(total_length * 1.25, 0.01)
    return (
        (-lateral_limit, lateral_limit),
        (-lateral_limit, lateral_limit),
        (0.0, z_upper),
    )


def apply_axis_limits(axis, axis_limits: AxisLimits) -> None:
    """Apply x/y/z limits to a matplotlib 3D axis."""
    axis.set_xlim(*axis_limits[0])
    axis.set_ylim(*axis_limits[1])
    axis.set_zlim(*axis_limits[2])
