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
    flexure_length: float | None = None
    distal_straight_length: float = 0.0
    flexure_joint_axes: tuple[str, ...] = ("y", "x", "y", "x")
    collision_radius: float | None = None
    mass: float | None = None
    bending_stiffness: float | None = None

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("segment length must be positive.")
        if self.tendon_radius <= 0.0:
            raise ValueError("tendon_radius must be positive.")
        flexure_length = self.effective_flexure_length
        distal_straight_length = self.length - flexure_length
        if flexure_length <= 0.0:
            raise ValueError("segment flexure_length must be positive.")
        if distal_straight_length < 0.0:
            raise ValueError("segment distal straight length cannot be negative.")
        if (
            self.flexure_length is not None
            and self.distal_straight_length != 0.0
            and abs(self.distal_straight_length - distal_straight_length) > 1.0e-12
        ):
            raise ValueError(
                "segment flexure_length and distal_straight_length must sum to length."
            )
        if not self.flexure_joint_axes:
            raise ValueError("segment flexure_joint_axes cannot be empty.")
        normalized_axes = tuple(str(axis).lower() for axis in self.flexure_joint_axes)
        if any(axis not in ("x", "y") for axis in normalized_axes):
            raise ValueError("segment flexure_joint_axes may only contain 'x' and 'y'.")
        object.__setattr__(self, "flexure_joint_axes", normalized_axes)
        object.__setattr__(self, "distal_straight_length", distal_straight_length)

    @property
    def effective_flexure_length(self) -> float:
        """Return the length occupied by the articulated flexure cells."""
        if self.flexure_length is not None:
            return self.flexure_length
        return self.length - self.distal_straight_length

    @property
    def effective_distal_straight_length(self) -> float:
        """Return the rigid spacer length after the segment flexure cells."""
        return self.length - self.effective_flexure_length

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
        segment = SegmentParams(
            length=0.04,
            tendon_radius=0.005,
            flexure_length=0.0365,
            distal_straight_length=0.0035,
        )
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
                    flexure_length=(
                        None
                        if segment.get("flexure_length") is None
                        else float(segment["flexure_length"])
                    ),
                    distal_straight_length=float(
                        segment.get("distal_straight_length", 0.0)
                    ),
                    flexure_joint_axes=tuple(
                        str(axis).lower()
                        for axis in segment.get(
                            "flexure_joint_axes",
                            ("y", "x", "y", "x"),
                        )
                    ),
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
