from __future__ import annotations

from dataclasses import replace

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.runtime.simulation_loop import SimulationLoop, SimulationLoopConfig
from continuum_sim.system.types import (
    ArmSystemState,
    BaseSystemState,
    RobotSystemCommand,
    RobotSystemState,
)


class _Backend:
    def __init__(self) -> None:
        self.step_count = 0

    def reset_system(self) -> RobotSystemState:
        return _state(self.step_count)

    def step_system(
        self,
        command: RobotSystemCommand,
        *,
        dt: float,
        n_substeps: int = 1,
    ) -> RobotSystemState:
        del command, dt, n_substeps
        self.step_count += 1
        return _state(self.step_count)


class _Controller:
    def __init__(self) -> None:
        self.seen = []

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        self.seen.append(state.metadata.get("visual_servo_target_visible"))
        return RobotSystemCommand.zeros({"observer": 3})


class _Enricher:
    def enrich_state(self, state: RobotSystemState) -> RobotSystemState:
        return replace(
            state,
            metadata={**state.metadata, "visual_servo_target_visible": True},
        )


def test_simulation_loop_enriches_state_before_controller() -> None:
    controller = _Controller()
    loop = SimulationLoop(
        _Backend(),
        controller,
        SimulationLoopConfig(controller_dt_s=0.02, n_substeps=1, max_steps=1),
        hooks=(_Enricher(),),
    )

    result = loop.run()

    assert controller.seen == [True]
    assert result.states[0].metadata["visual_servo_target_visible"] is True
    assert result.states[-1].metadata["visual_servo_target_visible"] is True


def _state(step: int) -> RobotSystemState:
    return RobotSystemState(
        time_s=0.02 * step,
        base=BaseSystemState(pose=Pose6D.identity()),
        arms={
            "observer": ArmSystemState(
                name="observer",
                role="observer",
                tip_pose_world=Pose6D.identity(),
                segment_poses_world=np.repeat(np.eye(4)[None, :, :], 3, axis=0),
                tendon_displacement_m=np.zeros(3, dtype=float),
                tendon_velocity_mps=np.zeros(3, dtype=float),
            )
        },
    )
