"""Route automatic executor tasks through the mounted tool TCP."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


EXECUTOR_CONTROL_FRAME = "tool_tcp"


def with_executor_tool_tcp(state: RobotSystemState) -> RobotSystemState:
    """Return a controller-facing state whose executor task pose is the tool TCP."""

    updated = dict(state.arms)
    executor_count = 0
    for name, arm in state.arms.items():
        if arm.role != "executor":
            continue
        executor_count += 1
        if arm.tool_pose_world is None:
            raise RuntimeError(
                f"Executor arm {name!r} has no MuJoCo tool TCP site. "
                f"Automatic tasks require {name}_tool_tcp."
            )
        updated[name] = replace(
            arm,
            tip_pose_world=arm.tool_pose_world,
            metadata={
                **arm.metadata,
                "executor_control_frame": EXECUTOR_CONTROL_FRAME,
            },
        )
    if executor_count != 1:
        raise RuntimeError(
            "Automatic tasks require exactly one enabled executor arm, "
            f"got {executor_count}."
        )
    return replace(state, arms=updated)


class ExecutorToolTcpController:
    """Apply one TCP task frame to every automatic controller implementation."""

    def __init__(self, controller: object) -> None:
        self.controller = controller

    def __getattr__(self, name: str):
        return getattr(self.controller, name)

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        control_state = with_executor_tool_tcp(state)
        command = self.controller.compute_command(control_state)
        executor = next(
            arm for arm in control_state.arms.values() if arm.role == "executor"
        )
        actual = executor.tip_pose_world.position.copy()
        metadata = {
            **command.metadata,
            "executor_control_frame": EXECUTOR_CONTROL_FRAME,
            "executor_actual_world": actual,
        }
        target = _metadata_point(metadata.get("executor_target_world"))
        if target is not None:
            error = target - actual
            metadata["executor_error_world"] = error
            metadata["executor_error_m"] = float(np.linalg.norm(error))
        return replace(command, metadata=metadata)


def _metadata_point(value: object) -> np.ndarray | None:
    if value is None:
        return None
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return point.copy()


__all__ = [
    "EXECUTOR_CONTROL_FRAME",
    "ExecutorToolTcpController",
    "with_executor_tool_tcp",
]
