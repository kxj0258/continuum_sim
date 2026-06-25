from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.multi_arm import load_multi_arm_config
from continuum_sim.runtime.multi_arm_state import (
    ArmRuntimeState,
    MultiArmRuntimeState,
    get_arm_state,
    get_role_state,
    update_world_centerlines,
    update_world_poses,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "robots" / "dual_continuum.yaml"


def test_update_world_poses_identity_base_uses_mount_then_local_tip() -> None:
    config = load_multi_arm_config(CONFIG_PATH)
    state = MultiArmRuntimeState(
        base_pose=Pose6D.identity(),
        arm_states={
            "observer": _state("observer", "observer", [0.0, 0.0, 0.10]),
            "executor": _state("executor", "executor", [0.0, 0.0, 0.10]),
        },
    )

    update_world_poses(config, state)

    assert_allclose(get_arm_state(state, "observer").world_tip_pose.position, [0.0, 0.035, 0.10])
    assert_allclose(get_arm_state(state, "executor").world_tip_pose.position, [0.0, -0.035, 0.10])


def test_update_world_poses_translates_all_arm_tips_by_base_pose() -> None:
    config = load_multi_arm_config(CONFIG_PATH)
    state = MultiArmRuntimeState(
        base_pose=Pose6D.from_dict({"position": [0.3, -0.2, 0.4], "quat": [1.0, 0.0, 0.0, 0.0]}),
        arm_states={
            "observer": _state("observer", "observer", [0.0, 0.0, 0.10]),
            "executor": _state("executor", "executor", [0.0, 0.0, 0.12]),
        },
    )

    update_world_poses(config, state)

    assert_allclose(get_arm_state(state, "observer").world_tip_pose.position, [0.3, -0.165, 0.5])
    assert_allclose(get_arm_state(state, "executor").world_tip_pose.position, [0.3, -0.235, 0.52])


def test_mount_offsets_make_observer_and_executor_world_tips_different() -> None:
    config = load_multi_arm_config(CONFIG_PATH)
    state = MultiArmRuntimeState(
        base_pose=Pose6D.identity(),
        arm_states={
            "observer": _state("observer", "observer", [0.0, 0.0, 0.10]),
            "executor": _state("executor", "executor", [0.0, 0.0, 0.10]),
        },
    )

    update_world_poses(config, state)

    observer_tip = get_arm_state(state, "observer").world_tip_pose.position
    executor_tip = get_arm_state(state, "executor").world_tip_pose.position
    assert not np.allclose(observer_tip, executor_tip)


def test_update_world_centerlines_transforms_each_local_centerline_to_world() -> None:
    config = load_multi_arm_config(CONFIG_PATH)
    local_centerline = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, 0.10],
        ],
        dtype=float,
    )
    state = MultiArmRuntimeState(
        base_pose=Pose6D.from_dict({"position": [0.1, 0.0, 0.2], "quat": [1.0, 0.0, 0.0, 0.0]}),
        arm_states={
            "observer": _state("observer", "observer", [0.0, 0.0, 0.10], local_centerline=local_centerline),
            "executor": _state("executor", "executor", [0.0, 0.0, 0.10], local_centerline=local_centerline),
        },
    )

    update_world_centerlines(config, state)

    assert_allclose(
        get_arm_state(state, "observer").world_centerline,
        [[0.1, 0.035, 0.2], [0.1, 0.035, 0.25], [0.1, 0.035, 0.3]],
    )
    assert_allclose(
        get_arm_state(state, "executor").world_centerline,
        [[0.1, -0.035, 0.2], [0.1, -0.035, 0.25], [0.1, -0.035, 0.3]],
    )


def test_get_role_state_returns_all_states_for_matching_role() -> None:
    config = load_multi_arm_config(CONFIG_PATH)
    state = MultiArmRuntimeState(
        base_pose=Pose6D.identity(),
        arm_states={
            "observer": _state("observer", "observer", [0.0, 0.0, 0.10]),
            "executor": _state("executor", "executor", [0.0, 0.0, 0.10]),
        },
    )

    observer_states = get_role_state(config, state, "observer")

    assert [arm_state.name for arm_state in observer_states] == ["observer"]


def test_get_arm_state_raises_clear_error_for_unknown_state() -> None:
    state = MultiArmRuntimeState(base_pose=Pose6D.identity(), arm_states={})

    with pytest.raises(KeyError, match="missing"):
        get_arm_state(state, "missing")


def _state(
    name: str,
    role: str,
    position: list[float],
    *,
    local_centerline: np.ndarray | None = None,
) -> ArmRuntimeState:
    return ArmRuntimeState(
        name=name,
        role=role,
        local_tip_pose=Pose6D.from_dict({"position": position, "quat": [1.0, 0.0, 0.0, 0.0]}),
        local_centerline=local_centerline,
    )
