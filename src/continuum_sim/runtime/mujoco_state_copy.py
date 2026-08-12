"""Copy the dynamic inputs needed to reconstruct an independent MuJoCo state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MUJOCO_DYNAMIC_ARRAY_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "mocap_pos",
    "mocap_quat",
    "userdata",
)


def copy_mujoco_dynamic_state(source: object, destination: object) -> None:
    """Copy live integration inputs; call ``mj_forward`` on the destination next."""

    destination.time = source.time
    for field_name in MUJOCO_DYNAMIC_ARRAY_FIELDS:
        source_values = getattr(source, field_name, None)
        destination_values = getattr(destination, field_name, None)
        if source_values is None or destination_values is None:
            continue
        np.copyto(destination_values, source_values)


@dataclass(frozen=True)
class MujocoDynamicStateSnapshot:
    """Owned dynamic arrays safe to hand to a renderer thread."""

    time: float
    arrays: dict[str, np.ndarray]

    def apply_to(self, destination: object) -> None:
        destination.time = self.time
        for field_name, values in self.arrays.items():
            destination_values = getattr(destination, field_name, None)
            if destination_values is not None:
                np.copyto(destination_values, values)


def capture_mujoco_dynamic_state(source: object) -> MujocoDynamicStateSnapshot:
    """Capture an owned, immutable-by-convention dynamic-state payload."""

    return MujocoDynamicStateSnapshot(
        time=float(source.time),
        arrays={
            field_name: np.asarray(values).copy()
            for field_name in MUJOCO_DYNAMIC_ARRAY_FIELDS
            if (values := getattr(source, field_name, None)) is not None
        },
    )


__all__ = [
    "MUJOCO_DYNAMIC_ARRAY_FIELDS",
    "MujocoDynamicStateSnapshot",
    "capture_mujoco_dynamic_state",
    "copy_mujoco_dynamic_state",
]
