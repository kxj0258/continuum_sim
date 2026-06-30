"""Physical tendon path definitions for the three-segment robot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from continuum_sim.config import load_yaml


@dataclass(frozen=True)
class PhysicalTendonPath:
    """A base-routed physical tendon and the segments it traverses."""

    id: str
    global_index: int
    motor_index: int
    anchor_segment_index: int
    angle_deg: float
    radial_offset: float
    path_segment_indices: tuple[int, ...]
    hole_index: int | None = None


def load_physical_tendons_from_yaml(path: str | Path) -> tuple[PhysicalTendonPath, ...]:
    """Load and validate physical tendon paths from the robot YAML file."""
    from continuum_sim.model.dual_arm_robot import is_dual_arm_robot_config, load_dual_arm_robot_config

    if is_dual_arm_robot_config(path):
        return tuple(
            PhysicalTendonPath(
                id=tendon.id,
                global_index=index,
                motor_index=index,
                anchor_segment_index=tendon.anchor_segment_index,
                angle_deg=tendon.angle_deg,
                radial_offset=tendon.radial_offset,
                path_segment_indices=tendon.path_segment_indices,
                hole_index=tendon.hole_index,
            )
            for index, tendon in enumerate(load_dual_arm_robot_config(path).default_arm_tendons)
        )
    config = load_yaml(path)
    robot = config.get("robot", {})
    segment_count = int(robot.get("segment_count", len(config.get("segments", []))))
    expected_tendon_count = int(robot.get("total_tendon_count", 0))
    tendon_items = config.get("physical_tendons", [])
    if expected_tendon_count <= 0:
        expected_tendon_count = len(tendon_items)
    if len(tendon_items) != expected_tendon_count:
        raise ValueError(
            f"Expected {expected_tendon_count} physical tendons, got {len(tendon_items)}."
        )

    tendons = tuple(_physical_tendon_from_dict(item) for item in tendon_items)
    tendons = tuple(sorted(tendons, key=lambda tendon: tendon.global_index))

    global_indices = [tendon.global_index for tendon in tendons]
    expected_indices = list(range(expected_tendon_count))
    if global_indices != expected_indices:
        expected_label = (
            "none"
            if not expected_indices
            else f"{expected_indices[0]}..{expected_indices[-1]}"
        )
        raise ValueError(
            "Expected physical tendon global_index values "
            f"{expected_label}, got {global_indices}."
        )

    for tendon in tendons:
        if not tendon.path_segment_indices:
            raise ValueError(f"{tendon.id} must define at least one path segment.")
        invalid_indices = [
            segment_index
            for segment_index in tendon.path_segment_indices
            if segment_index < 0 or segment_index >= segment_count
        ]
        if invalid_indices:
            raise ValueError(
                f"{tendon.id} has path_segment_indices outside "
                f"0..{segment_count - 1}: {invalid_indices}."
            )

    return tendons


def _physical_tendon_from_dict(item: dict[str, object]) -> PhysicalTendonPath:
    return PhysicalTendonPath(
        id=str(item["id"]),
        global_index=int(item["global_index"]),
        motor_index=int(item["motor_index"]),
        anchor_segment_index=int(item["anchor_segment_index"]),
        angle_deg=float(item["angle_deg"]),
        radial_offset=float(item["radial_offset"]),
        path_segment_indices=tuple(int(v) for v in item["path_segment_indices"]),  # type: ignore[index]
        hole_index=(None if "hole_index" not in item else int(item["hole_index"])),
    )
