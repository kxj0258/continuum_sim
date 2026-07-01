"""Shared mobile-base + arm frame context helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.mount_frame import MobileBaseMountConfig, load_mobile_base_mount_config


@dataclass(frozen=True)
class MobileBaseArmContext:
    """Static frame context for one mobile base and one arm mount."""

    base_pose: Pose6D
    mount_pose: Pose6D
    mount_name: str = "arm_mount"

    @classmethod
    def identity(cls) -> "MobileBaseArmContext":
        return cls(base_pose=Pose6D.identity(), mount_pose=Pose6D.identity())

    @classmethod
    def from_config(
        cls,
        config: MobileBaseMountConfig,
        *,
        mount_name: str | None = None,
    ) -> "MobileBaseArmContext":
        name = mount_name or ("arm_mount" if "arm_mount" in config.mounts else config.mount.name)
        mount = config.mounts.get(name)
        if mount is None:
            raise KeyError(f"Unknown mount {name!r} in mobile base config.")
        return cls(
            base_pose=config.mobile_base.pose,
            mount_pose=mount.pose,
            mount_name=name,
        )

    @classmethod
    def from_config_path(
        cls,
        path: str | Path | None,
        *,
        mount_name: str | None = None,
    ) -> "MobileBaseArmContext":
        if path is None:
            return cls.identity()
        return cls.from_config(load_mobile_base_mount_config(path), mount_name=mount_name)

    @property
    def world_mount_pose(self) -> Pose6D:
        return self.base_pose.compose(self.mount_pose)

    @property
    def local_from_world_pose(self) -> Pose6D:
        return self.world_mount_pose.inverse()

    def local_point_to_world(self, point: np.ndarray) -> np.ndarray:
        return self.world_mount_pose.transform_point(point)

    def local_points_to_world(self, points: np.ndarray) -> np.ndarray:
        return self.world_mount_pose.transform_points(points)

    def world_point_to_local(self, point: np.ndarray) -> np.ndarray:
        return self.local_from_world_pose.transform_point(point)

    def world_points_to_local(self, points: np.ndarray) -> np.ndarray:
        return self.local_from_world_pose.transform_points(points)

    def local_pose_to_world(self, pose: Pose6D) -> Pose6D:
        return self.world_mount_pose.transform_pose(pose)

    def world_pose_to_local(self, pose: Pose6D) -> Pose6D:
        return self.local_from_world_pose.transform_pose(pose)
