"""World-frame pose composition helpers for mobile-base-mounted continuum arms."""

from __future__ import annotations

import numpy as np

from continuum_sim.model.base_pose import Pose6D


def compose_base_mount_pose(base_pose: Pose6D, mount_pose: Pose6D) -> Pose6D:
    """Return ``T_world_mount = T_world_mobile_base * T_mobile_base_mount``."""

    return base_pose.compose(mount_pose)


def compute_world_tip_pose(
    base_pose: Pose6D,
    mount_pose: Pose6D,
    local_tip_pose: Pose6D,
) -> Pose6D:
    """Return ``T_world_tip = T_world_mobile_base * T_mobile_base_mount * T_mount_tip``."""

    return compose_base_mount_pose(base_pose, mount_pose).compose(local_tip_pose)


def compose_world_tip_pose(
    base_pose: Pose6D,
    mount_pose: Pose6D,
    local_tip_pose: Pose6D,
) -> Pose6D:
    """Backward-compatible alias for :func:`compute_world_tip_pose`."""

    return compute_world_tip_pose(base_pose, mount_pose, local_tip_pose)


def transform_pose_to_world(
    base_pose: Pose6D,
    mount_pose: Pose6D,
    local_pose: Pose6D,
) -> Pose6D:
    """Transform any local mount-frame pose into world frame."""

    return compute_world_tip_pose(base_pose, mount_pose, local_pose)


def transform_centerline_to_world(
    base_pose: Pose6D,
    mount_pose: Pose6D,
    local_centerline: np.ndarray,
) -> np.ndarray:
    """Transform an ``N x 3`` centerline from mount frame into world frame."""

    return compose_base_mount_pose(base_pose, mount_pose).transform_points(local_centerline)


def compose_world_tool_pose(
    base_pose: Pose6D,
    mount_pose: Pose6D,
    local_tip_pose: Pose6D,
    tip_to_tool_pose: Pose6D,
) -> Pose6D:
    """Compose world tip pose and a fixed tip-to-tool transform."""

    return compute_world_tip_pose(base_pose, mount_pose, local_tip_pose).compose(tip_to_tool_pose)
