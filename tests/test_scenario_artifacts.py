from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import continuum_sim.io.scenario_artifacts as scenario_artifacts
from continuum_sim.application.scenario import (
    ScenarioArtifactConfig,
    ScenarioBackendConfig,
    ScenarioConfig,
    ScenarioHookConfig,
    ScenarioRuntimeConfig,
    ScenarioSceneConfig,
    ScenarioTaskConfig,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.runtime.simulation_loop import SimulationLoopResult
from continuum_sim.system.types import (
    ArmSystemState,
    ArmTendonRateCommand,
    BaseSystemState,
    RobotSystemCommand,
    RobotSystemState,
)


def test_scenario_artifacts_keep_npz_metadata_and_plots_when_gif_fails(
    tmp_path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    assembly_path = tmp_path / "assembly.yaml"
    scenario_path.write_text("schema_version: 1\n", encoding="utf-8")
    assembly_path.write_text("schema_version: 1\n", encoding="utf-8")
    config = ScenarioConfig(
        path=scenario_path,
        name="gif_failure",
        assembly_config_path=assembly_path,
        backend=ScenarioBackendConfig(type="analytic"),
        scene=ScenarioSceneConfig(),
        task=ScenarioTaskConfig(
            type="tracking",
            waypoints_world=np.zeros((1, 3), dtype=float),
            waypoint_tolerance_m=0.001,
            observer_roi_world=None,
            loop=False,
            min_clearance_m=0.01,
            terminate_on_clearance_violation=True,
            surface_normal_world=np.array([0.0, 0.0, 1.0], dtype=float),
            target_contact_distance_m=0.0,
            contact_tolerance_m=0.002,
        ),
        runtime=ScenarioRuntimeConfig(
            controller_dt_s=0.02,
            n_substeps=1,
            max_steps=1,
        ),
        hooks=ScenarioHookConfig(
            recorder=True,
            tendon_debug=False,
            tendon_debug_stride=1,
            viewer="none",
            keep_viewer_open=False,
        ),
        artifacts=ScenarioArtifactConfig(
            enabled=True,
            output_root=tmp_path / "runs",
            save_npz=True,
            save_plots=True,
            save_gif=True,
            save_model=True,
            video_fps=20,
            video_stride=10,
        ),
    )
    state0 = _state(0.0, [0.0, 0.0, 0.10])
    state1 = _state(0.02, [0.01, 0.0, 0.10])
    result = SimulationLoopResult(
        states=(state0, state1),
        commands=(
            RobotSystemCommand(
                base_twist_world=np.zeros(6, dtype=float),
                arms={"executor": ArmTendonRateCommand(np.zeros(9, dtype=float))},
            ),
        ),
        stopped_early=False,
    )
    recorder = SimpleNamespace(
        target_position_m=[np.array([0.01, 0.0, 0.10], dtype=float)],
        tracking_error_m=[0.0],
        achieved_waypoint_error_m=[0.0005],
        waypoint_advanced=[True],
        tracking_complete=[True],
        tracking_approach=[False],
        arm_saturation_scale={"executor": [0.75]},
        arm_tendon_target_error_norm_m={"executor": [0.0004]},
        arm_tendon_target_error_max_m={"executor": [0.0003]},
        arm_peak_actuator_force_n={"executor": [3.0]},
        waypoint_index=[0],
        min_clearance_m=[np.nan],
        contact_distance_m=[np.nan],
        contact_error_m=[np.nan],
        target_force_n=[np.nan],
        estimated_force_n=[np.nan],
        force_error_n=[np.nan],
        task_phase=["tracking"],
        engine_navigation_phase=["base_approach"],
        engine_navigation_terminal_reason=[""],
        engine_navigation_progress=[0.0],
        base_target_position_m=[np.array([0.1, 0.2, 0.3])],
        base_position_error_m=[0.1],
        base_orientation_error_rad=[0.2],
    )
    application = SimpleNamespace(
        config=config,
        hooks_by_name={"recorder": recorder},
        loop=SimpleNamespace(backend=SimpleNamespace()),
    )

    def fail_video(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("gif encoder unavailable")

    monkeypatch.setattr(scenario_artifacts, "save_replay_video", fail_video)

    paths = scenario_artifacts.save_scenario_artifacts(application, result)

    assert paths is not None
    assert paths.result_npz.is_file()
    assert paths.metadata_json.is_file()
    assert (paths.plots_dir / "trajectory.png").is_file()
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["video"] is None
    assert metadata["errors"] == ["video: RuntimeError: gif encoder unavailable"]
    assert metadata["metrics"]["final_achieved_waypoint_error_m"] == 0.0005
    with np.load(paths.result_npz) as arrays:
        assert arrays["arm_executor_saturation_scale"].tolist() == [0.75]
        assert arrays["arm_executor_peak_actuator_force_n"].tolist() == [3.0]
        assert arrays["engine_navigation_phase"].tolist() == ["base_approach"]
        assert arrays["base_target_position_m"].shape == (1, 3)
        assert arrays["base_position_error_m"].tolist() == [0.1]


def _state(time_s: float, tip_position: list[float]) -> RobotSystemState:
    pose = Pose6D(position=np.asarray(tip_position, dtype=float), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    return RobotSystemState(
        time_s=time_s,
        base=BaseSystemState(Pose6D.identity()),
        arms={
            "executor": ArmSystemState(
                name="executor",
                role="executor",
                tip_pose_world=pose,
                segment_poses_world=np.repeat(np.eye(4)[None, :, :], 2, axis=0),
                tendon_displacement_m=np.zeros(9, dtype=float),
                tendon_velocity_mps=np.zeros(9, dtype=float),
            )
        },
    )
