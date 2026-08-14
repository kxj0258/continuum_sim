"""Reference poses for executor TCP task planning."""

from __future__ import annotations

import numpy as np

from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.tools.attachments import attachment_config_path, load_attachment_config
from continuum_sim.tools.tool_frames import compute_tool_tcp_pose


def base_to_straight_executor_tcp_pose(
    assembly: RobotAssemblyConfig,
) -> Pose6D:
    """Return the base-frame pose of the executor TCP at zero curvature."""

    executors = [arm for arm in assembly.enabled_arms if arm.role == "executor"]
    if len(executors) != 1:
        raise ValueError(
            "TCP task planning requires exactly one enabled executor arm, "
            f"got {len(executors)}."
        )
    executor = executors[0]
    if executor.attachment is None:
        raise ValueError(
            "The executor arm requires a contact tool attachment with a TCP."
        )
    attachment_path = attachment_config_path(assembly.path, executor.attachment)
    if attachment_path is None:
        raise FileNotFoundError(
            f"Attachment config {executor.attachment!r} for executor arm "
            f"{executor.name!r} was not found."
        )
    tool = load_attachment_config(attachment_path)
    local_tip = Pose6D.from_matrix(
        forward_kinematics(
            np.zeros(executor.spatial_arm.params.q_size, dtype=float),
            executor.spatial_arm.params,
        ).tip_pose
    )
    base_tip = executor.mount_pose.compose(local_tip)
    return compute_tool_tcp_pose(base_tip, tool)


def straight_executor_tcp_world_pose(
    assembly: RobotAssemblyConfig,
) -> Pose6D:
    """Return the initial world-frame executor TCP pose at zero curvature."""

    return assembly.base.initial_pose.compose(
        base_to_straight_executor_tcp_pose(assembly)
    )


__all__ = [
    "base_to_straight_executor_tcp_pose",
    "straight_executor_tcp_world_pose",
]
