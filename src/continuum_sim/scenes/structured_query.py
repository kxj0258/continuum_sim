"""Adapter from navigation-scene primitives to the system scene-query port."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.scenes.primitives import DistanceQuery, nearest_clearance
from continuum_sim.scenes.scene_config import NavigationSceneConfig


@dataclass(frozen=True)
class StructuredSceneQuery:
    """Control-facing clearance queries for structured baseline scenes."""

    config: NavigationSceneConfig

    def nearest_distance(self, point_world: np.ndarray) -> DistanceQuery:
        return nearest_clearance(
            np.asarray(point_world, dtype=float),
            self.config.clearance_primitives,
        )

    def nearest_centerline_clearance(self, centerline_world: np.ndarray) -> DistanceQuery:
        centerline = np.asarray(centerline_world, dtype=float)
        if centerline.ndim != 2 or centerline.shape[1] != 3:
            raise ValueError("centerline_world must have shape (N, 3).")
        if centerline.shape[0] == 0:
            return self.nearest_distance(np.zeros(3, dtype=float))
        return min(
            (self.nearest_distance(point) for point in centerline),
            key=lambda query: query.distance_m,
        )

