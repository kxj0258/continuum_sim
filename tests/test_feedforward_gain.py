from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from continuum_sim import load_mujoco_config
from continuum_sim.application.scenario import (
    ScenarioTrackingControlConfig,
    _load_tracking_control_config,
    load_scenario_config,
)
from continuum_sim.application.application import _tracking_coordinated_config
from continuum_sim.backends.analytic_system_backend import AnalyticSystemBackend
from continuum_sim.backends.mujoco_system_backend import (
    _bound_servo_config_to_actuator,
)
from continuum_sim.control.coordinated_tracking import CoordinatedTrackingConfig
from continuum_sim.control.tendon_rate_control import BendingRateServoConfig
from continuum_sim.control.task_intent import (
    CartesianTaskIntent,
    SystemTaskIntent,
    TaskStatus,
    TaskStep,
)
from continuum_sim.control.unified_low_level import UnifiedLowLevelController
from continuum_sim.model.robot_assembly import load_robot_assembly_config


def test_mujoco_low_level_profile_loads_unit_feedforward_gain() -> None:
    config = load_scenario_config("configs/scenarios/single_mujoco_tracking.yaml")

    assert config.task.tracking_control.feedforward_gain == 1.0
    inner_loop = config.task.tracking_control.tendon_inner_loop
    assert inner_loop.mode == "bending_rate_servo"
    assert inner_loop.max_target_lead_m == 0.000285
    assert inner_loop.soft_force_limit_n == 24.0


def test_mujoco_servo_static_lead_bound_uses_hard_force_limit() -> None:
    mujoco_config = load_mujoco_config(
        "configs/mujoco_dual.yaml",
        require_xml=False,
        require_visual_meshes=False,
    )
    bounded = _bound_servo_config_to_actuator(
        BendingRateServoConfig(
            max_target_lead_m=0.0005,
            soft_force_limit_n=24.0,
            hard_force_limit_n=30.0,
        ),
        mujoco_config,
    )

    assert bounded is not None
    assert_allclose(bounded.max_target_lead_m, 0.0003)
    assert bounded.soft_force_limit_n == 24.0
    assert bounded.hard_force_limit_n == 30.0


def test_task_feedforward_gain_overrides_low_level_profile() -> None:
    config = _load_tracking_control_config(
        {"tracking_control": {"feedforward_gain": 0.25}},
        {"feedforward_gain": 0.75},
    )

    assert config.feedforward_gain == 0.25


def test_scenario_feedforward_gain_reaches_coordinated_control() -> None:
    coordinated = _tracking_coordinated_config(
        ScenarioTrackingControlConfig(feedforward_gain=0.25)
    )

    assert coordinated.feedforward_gain == 0.25


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf])
def test_feedforward_gain_must_be_non_negative_and_finite(value: float) -> None:
    with pytest.raises(ValueError, match="feedforward_gain"):
        ScenarioTrackingControlConfig(feedforward_gain=value)

    with pytest.raises(ValueError, match="feedforward_gain"):
        CoordinatedTrackingConfig(feedforward_gain=value)


def test_position_mode_scales_feedforward_and_retains_raw_intent() -> None:
    assembly = load_robot_assembly_config(
        "configs/robots/assemblies/single_spatial.yaml"
    )
    state = AnalyticSystemBackend(assembly).reset_system()
    current_tip = state.arms["executor"].tip_pose_world.position
    raw_velocity = np.array([0.004, 0.0, 0.0], dtype=float)
    controller = UnifiedLowLevelController(
        assembly,
        coordinated_config=CoordinatedTrackingConfig(feedforward_gain=0.25),
    )

    command = controller.compute_command(
        state,
        _task_step(current_tip, raw_velocity, control_mode="position"),
    )

    assert command.metadata["executor_feedforward_gain"] == 0.25
    assert_allclose(command.metadata["task_intent_velocity_world"], raw_velocity)
    assert_allclose(
        command.metadata["executor_scaled_feedforward_velocity_world"],
        [0.001, 0.0, 0.0],
    )
    assert_allclose(
        command.metadata["executor_target_velocity_world"],
        [0.001, 0.0, 0.0],
    )


def test_velocity_mode_does_not_scale_direct_velocity_command() -> None:
    assembly = load_robot_assembly_config(
        "configs/robots/assemblies/single_spatial.yaml"
    )
    state = AnalyticSystemBackend(assembly).reset_system()
    current_tip = state.arms["executor"].tip_pose_world.position
    direct_velocity = np.array([0.004, 0.0, 0.0], dtype=float)
    controller = UnifiedLowLevelController(
        assembly,
        coordinated_config=CoordinatedTrackingConfig(feedforward_gain=0.25),
    )

    command = controller.compute_command(
        state,
        _task_step(
            current_tip + np.array([0.02, 0.0, 0.0], dtype=float),
            direct_velocity,
            control_mode="velocity",
        ),
    )

    assert command.metadata["executor_feedforward_gain"] == 0.25
    assert_allclose(
        command.metadata["executor_scaled_feedforward_velocity_world"],
        direct_velocity,
    )
    assert_allclose(
        command.metadata["executor_target_velocity_world"],
        direct_velocity,
    )


def _task_step(
    target_position_world: np.ndarray,
    velocity_world: np.ndarray,
    *,
    control_mode: str,
) -> TaskStep:
    return TaskStep(
        intent=SystemTaskIntent(
            executor=CartesianTaskIntent(
                target_position_world=target_position_world,
                feedforward_velocity_world=velocity_world,
                control_mode=control_mode,
            )
        ),
        status=TaskStatus(task_type="tracking", phase="tracking"),
    )
