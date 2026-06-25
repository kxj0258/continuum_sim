"""Attachment frame composition helpers.

These functions only compose fixed `Pose6D` transforms. They do not perform
control, rendering, perception, or collision checks.
"""

from __future__ import annotations

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.tools.attachments import AttachmentConfig


def compute_attachment_pose(
    world_tip_pose: Pose6D,
    attachment_config: AttachmentConfig,
) -> Pose6D:
    """Return `T_world_attachment = T_world_tip * T_tip_attachment`."""

    return world_tip_pose.compose(attachment_config.tip_to_attachment)


def compute_tool_tcp_pose(
    world_tip_pose: Pose6D,
    tool_config: AttachmentConfig,
) -> Pose6D:
    """Return `T_world_tcp` for a contact sphere tool."""

    if tool_config.type != "contact_sphere_tool" or tool_config.tcp_pose is None:
        raise ValueError("compute_tool_tcp_pose requires a contact_sphere_tool with tcp_pose.")
    return compute_attachment_pose(world_tip_pose, tool_config).compose(tool_config.tcp_pose)


def compute_camera_pose(
    world_tip_pose: Pose6D,
    camera_airgun_config: AttachmentConfig,
) -> Pose6D:
    """Return `T_world_camera = T_world_tip * T_tip_camera`."""

    if camera_airgun_config.type != "camera_airgun" or camera_airgun_config.camera is None:
        raise ValueError("compute_camera_pose requires a camera_airgun with camera config.")
    return world_tip_pose.compose(camera_airgun_config.camera.tip_to_camera)


def compute_nozzle_pose(
    world_tip_pose: Pose6D,
    camera_airgun_config: AttachmentConfig,
) -> Pose6D:
    """Return `T_world_nozzle = T_world_tip * T_tip_nozzle`."""

    if camera_airgun_config.type != "camera_airgun" or camera_airgun_config.nozzle_pose is None:
        raise ValueError("compute_nozzle_pose requires a camera_airgun with nozzle_pose.")
    return world_tip_pose.compose(camera_airgun_config.nozzle_pose)


def compute_all_attachment_frames(
    world_tip_pose: Pose6D,
    attachment_config: AttachmentConfig,
) -> dict[str, Pose6D]:
    """Return all supported world-frame poses for one attachment."""

    frames = {"attachment": compute_attachment_pose(world_tip_pose, attachment_config)}
    if attachment_config.type == "contact_sphere_tool":
        frames["tcp"] = compute_tool_tcp_pose(world_tip_pose, attachment_config)
    elif attachment_config.type == "camera_airgun":
        frames["camera"] = compute_camera_pose(world_tip_pose, attachment_config)
        frames["nozzle"] = compute_nozzle_pose(world_tip_pose, attachment_config)
    else:
        raise ValueError(f"Unknown attachment type {attachment_config.type!r}.")
    return frames
