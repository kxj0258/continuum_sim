"""Copy the dynamic inputs needed to reconstruct an independent MuJoCo state."""

from __future__ import annotations

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


__all__ = ["MUJOCO_DYNAMIC_ARRAY_FIELDS", "copy_mujoco_dynamic_state"]
