from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from continuum_sim.application import SimulationApplication
from continuum_sim.system.types import ArmTendonRateCommand, RobotSystemCommand
from continuum_sim.visualization.mujoco_system_debug_viewer import (
    MujocoSystemDebugViewer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = PROJECT_ROOT / "configs" / "scenarios" / "mujoco_manual_control.yaml"


@pytest.mark.mujoco
def test_manual_scenario_compiles_tool_sensor_and_advances_exact_control_time() -> None:
    pytest.importorskip("mujoco")
    application = SimulationApplication.from_yaml(SCENARIO)
    backend = application.loop.backend
    state = backend.reset_system()
    mujoco = pytest.importorskip("mujoco")
    model = backend.physics.model

    def geom_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    assert geom_id("executor_ft_sensor_visual") >= 0
    assert geom_id("executor_wiping_sphere") >= 0
    assert geom_id("observer_ft_sensor_visual") == -1
    assert geom_id("observer_wiping_sphere") == -1
    assert geom_id("observer_eye_camera_dome_visual") >= 0
    assert geom_id("observer_eye_camera_lens_visual") >= 0

    engine_mesh_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_MESH,
            f"engine_visual_mesh_part_{index}",
        )
        for index in (1, 2)
    ]
    assert all(mesh_id >= 0 for mesh_id in engine_mesh_ids)
    assert sum(int(model.mesh_facenum[mesh_id]) for mesh_id in engine_mesh_ids) == 253_026
    for index in (1, 2):
        engine_geom_id = geom_id(f"engine_visual_part_{index}")
        assert engine_geom_id >= 0
        assert_allclose(
            model.geom_rgba[engine_geom_id],
            [0.66, 0.68, 0.71, 1.0],
            atol=1.0e-6,
        )

    executor = state.arms["executor"]
    assert executor.tool_pose_world is not None
    assert executor.tool_wrench is not None
    arm_tip = np.asarray(executor.metadata["arm_tip_pose_world"], dtype=float)
    assert np.linalg.norm(executor.tool_pose_world.position - arm_tip[:3, 3]) == pytest.approx(
        0.018
    )
    assert_allclose(executor.tool_wrench.force_sensor_n, np.zeros(3), atol=1.0e-10)
    assert_allclose(executor.tool_wrench.torque_sensor_nm, np.zeros(3), atol=1.0e-10)

    command = RobotSystemCommand(
        base_twist_world=np.zeros(6),
        arms={
            name: ArmTendonRateCommand(np.zeros_like(arm.tendon_displacement_m))
            for name, arm in state.arms.items()
        },
    )
    next_state = backend.step_system(command, dt=0.02, n_substeps=20)

    assert next_state.time_s == pytest.approx(0.02)


@pytest.mark.mujoco
def test_wiping_scenario_uses_tool_wrench_feedback_source() -> None:
    pytest.importorskip("mujoco")
    application = SimulationApplication.from_yaml(
        PROJECT_ROOT / "configs" / "scenarios" / "mujoco_wiping.yaml"
    )
    state = application.loop.backend.reset_system()

    command = application.loop.controller.compute_command(state)

    assert command.metadata["force_feedback_mode"] == "tool_wrench_sensor"
    assert command.metadata["normal_force_source"] == "tool_wrench_sensor"
    assert np.isfinite(command.metadata["measured_normal_force_n"])


@pytest.mark.mujoco
def test_manual_viewer_moves_mobile_base_in_translation_and_rotation() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("matplotlib")
    application = SimulationApplication.from_yaml(SCENARIO)
    viewer = MujocoSystemDebugViewer(
        application.loop.backend,
        control_dt_s=0.02,
        n_substeps=20,
    )
    try:
        assert viewer.base_control_enabled is True
        viewer.adjust_base_target(0, 1.0)
        viewer.adjust_base_target(5, 1.0)
        for _ in range(12):
            state = viewer.step()
        assert state.base.pose.position[0] == pytest.approx(0.01, abs=1.0e-6)
        assert abs(state.base.pose.quat[3]) > 0.0
    finally:
        viewer.close()
