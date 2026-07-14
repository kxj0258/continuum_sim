"""Parameter objects for the three-segment tendon-driven continuum arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.model.tendon_routing import TendonRouting

PCC_VALUES_PER_SEGMENT = 3


@dataclass(frozen=True)
class SegmentParams:
    """Geometry, routing, and optional physical data for one segment."""

    length: float
    tendon_radius: float
    tendon_angles_deg: tuple[float, ...] = (0.0, 120.0, 240.0)
    collision_radius: float | None = None
    mass: float | None = None
    bending_stiffness: float | None = None

    @property
    def effective_collision_radius(self) -> float:
        """Return the collision radius, falling back to the tendon radius."""
        return self.tendon_radius if self.collision_radius is None else self.collision_radius

    @property
    def routing(self) -> TendonRouting:
        return TendonRouting(self.tendon_angles_deg, self.tendon_radius)


@dataclass(frozen=True)
class ThreeSegmentRobotParams:
    """PCC-relevant parameters for the full three-segment robot."""

    segments: tuple[SegmentParams, SegmentParams, SegmentParams]

    @classmethod
    def default(cls) -> "ThreeSegmentRobotParams":
        """Return the placeholder 3x40 mm, 5 mm tendon-radius robot."""
        segment = SegmentParams(length=0.04, tendon_radius=0.005)
        return cls(segments=(segment, segment, segment))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ThreeSegmentRobotParams":
        """Create robot parameters from the project's robot YAML file."""
        from continuum_sim.model.dual_arm_robot import is_dual_arm_robot_config, load_dual_arm_robot_config

        if is_dual_arm_robot_config(path):
            return load_dual_arm_robot_config(path).default_arm_params
        config = load_yaml(path)
        segments = []
        for segment in config["segments"]:
            segments.append(
                SegmentParams(
                    length=float(segment["length"]),
                    tendon_radius=float(segment["tendon_radius"]),
                    tendon_angles_deg=tuple(float(v) for v in segment["tendon_angles_deg"]),
                    collision_radius=(
                        None
                        if segment.get("collision_radius") is None
                        else float(segment["collision_radius"])
                    ),
                    mass=None if segment.get("mass") is None else float(segment["mass"]),
                    bending_stiffness=(
                        None
                        if segment.get("bending_stiffness") is None
                        else float(segment["bending_stiffness"])
                    ),
                )
            )
        if len(segments) != 3:
            raise ValueError(f"Expected 3 segments, got {len(segments)}.")
        return cls(segments=tuple(segments))  # type: ignore[arg-type]

    @property
    def segment_lengths(self) -> np.ndarray:
        return np.array([segment.length for segment in self.segments], dtype=float)

    @property
    def tendon_radii(self) -> np.ndarray:
        return np.array([segment.tendon_radius for segment in self.segments], dtype=float)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def q_size(self) -> int:
        return self.segment_count * PCC_VALUES_PER_SEGMENT

    @property
    def tendon_count(self) -> int:
        return sum(len(segment.tendon_angles_deg) for segment in self.segments)
