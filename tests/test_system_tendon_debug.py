from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pytest

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.system.types import ArmSystemState, BaseSystemState, RobotSystemState
from continuum_sim.visualization.system_tendon_debug import system_tendon_view_data


def test_arm_system_state_validates_target_and_force_shapes() -> None:
    with pytest.raises(ValueError, match="matching 1D arrays"):
        _arm_state(
            "executor",
            target=[0.0, 0.001],
            actual=[0.0, 0.001],
            force=[1.0],
        )


def test_arm_system_state_defaults_diagnostics_for_legacy_callers() -> None:
    state = _arm_state(
        "executor",
        target=None,
        actual=[0.0, 0.001],
        force=None,
    )

    assert_allclose(state.tendon_target_m, state.tendon_displacement_m)
    assert_allclose(state.actuator_force_n, np.zeros(2))


def test_system_tendon_view_data_flattens_named_arms_in_state_order() -> None:
    state = RobotSystemState(
        time_s=0.2,
        base=BaseSystemState(Pose6D.identity()),
        arms={
            "executor": _arm_state(
                "executor",
                target=[0.001, 0.002],
                actual=[0.0005, 0.0015],
                force=[1.0, 2.0],
            ),
            "observer": _arm_state(
                "observer",
                target=[-0.001, -0.002],
                actual=[-0.0008, -0.0018],
                force=[0.5, 0.75],
            ),
        },
        metadata={
            "saturation": {
                "executor": {
                    "rate": np.array([True, False]),
                    "displacement": np.array([False, False]),
                }
            }
        },
    )

    view = system_tendon_view_data(state)

    assert view.labels == (
        "executor:1",
        "executor:2",
        "observer:1",
        "observer:2",
    )
    assert view.arm_boundaries == (2,)
    assert_allclose(view.target_m, [0.001, 0.002, -0.001, -0.002])
    assert_allclose(view.actual_m, [0.0005, 0.0015, -0.0008, -0.0018])
    assert_allclose(view.error_m, view.target_m - view.actual_m)
    assert_allclose(view.force_n, [1.0, 2.0, 0.5, 0.75])
    assert view.saturation_summary == (
        "executor: rate 1/2, displacement 0/2\n"
        "  mode compatible, scale 1.0000, residual 0.000e+00 m/s"
    )


def _arm_state(
    name: str,
    *,
    target: list[float] | None,
    actual: list[float],
    force: list[float] | None,
) -> ArmSystemState:
    actual_values = np.asarray(actual, dtype=float)
    return ArmSystemState(
        name=name,
        role=name,
        tip_pose_world=Pose6D.identity(),
        segment_poses_world=np.repeat(np.eye(4)[None, :, :], 3, axis=0),
        tendon_displacement_m=actual_values,
        tendon_velocity_mps=np.zeros_like(actual_values),
        tendon_target_m=None if target is None else np.asarray(target, dtype=float),
        actuator_force_n=None if force is None else np.asarray(force, dtype=float),
    )
