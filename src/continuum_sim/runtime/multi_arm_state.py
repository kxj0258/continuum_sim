"""Runtime state scaffold for future dual-continuum arm tasks.

The state helpers only compose externally supplied local tip poses and
centerlines into world frame. They do not implement dual-arm controllers,
visual servo, collision avoidance, MuJoCo integration, or snake-arm dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.kinematics.world_kinematics import (
    compose_world_tip_pose,
    transform_centerline_to_world,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.multi_arm import MultiArmConfig, get_arms_by_role


@dataclass
class ArmRuntimeState:
    """Runtime pose and optional shape state for one continuum arm."""

    name: str
    role: str
    local_tip_pose: Pose6D
    world_tip_pose: Pose6D | None = None
    local_centerline: np.ndarray | None = None
    world_centerline: np.ndarray | None = None
    q: np.ndarray | None = None
    tendon_lengths: np.ndarray | None = None
    attachment_pose: Pose6D | None = None


@dataclass
class MultiArmRuntimeState:
    """Runtime state for all continuum arms mounted on a shared base pose."""

    base_pose: Pose6D
    arm_states: dict[str, ArmRuntimeState]


def update_world_poses(config: MultiArmConfig, state: MultiArmRuntimeState) -> None:
    """Update each arm state with `base * mount * local_tip` world poses."""

    for arm_name, arm_state in state.arm_states.items():
        arm_config = config.arms.get(arm_name)
        if arm_config is None:
            raise KeyError(f"Arm state {arm_name!r} has no matching config arm.")
        arm_state.world_tip_pose = compose_world_tip_pose(
            state.base_pose,
            arm_config.mount.pose,
            arm_state.local_tip_pose,
        )


def update_world_centerlines(config: MultiArmConfig, state: MultiArmRuntimeState) -> None:
    """Update each arm state with world-frame centerline points when present."""

    for arm_name, arm_state in state.arm_states.items():
        if arm_state.local_centerline is None:
            arm_state.world_centerline = None
            continue
        arm_config = config.arms.get(arm_name)
        if arm_config is None:
            raise KeyError(f"Arm state {arm_name!r} has no matching config arm.")
        arm_state.world_centerline = transform_centerline_to_world(
            state.base_pose,
            arm_config.mount.pose,
            arm_state.local_centerline,
        )


def get_arm_state(state: MultiArmRuntimeState, name: str) -> ArmRuntimeState:
    """Return one arm runtime state by name."""

    try:
        return state.arm_states[name]
    except KeyError as exc:
        raise KeyError(f"Unknown arm state {name!r}.") from exc


def get_role_state(
    config: MultiArmConfig,
    state: MultiArmRuntimeState,
    role: str,
) -> list[ArmRuntimeState]:
    """Return runtime states for all configured arms matching `role`."""

    result: list[ArmRuntimeState] = []
    for arm_config in get_arms_by_role(config, role):
        if arm_config.name in state.arm_states:
            result.append(state.arm_states[arm_config.name])
    return result
