"""Scenario-native navigation mission target resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.scenes.scene_config import NavigationSceneConfig


@dataclass(frozen=True)
class NavigationMissionSpec:
    """Ordered inspection mission expressed by structured-scene target ids."""

    waypoint_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "NavigationMissionSpec":
        raw_ids = values.get("waypoint_ids", ())
        if not isinstance(raw_ids, list | tuple):
            raise ValueError("mission.waypoint_ids must be a list.")
        waypoint_ids = tuple(str(value) for value in raw_ids)
        if not waypoint_ids:
            raise ValueError("mission.waypoint_ids must contain at least one id.")
        return cls(waypoint_ids=waypoint_ids)


def resolve_navigation_waypoints(
    spec: NavigationMissionSpec,
    scene: NavigationSceneConfig,
) -> np.ndarray:
    """Return world-frame waypoints for a structured-scene mission."""

    return scene.target_positions(spec.waypoint_ids)
