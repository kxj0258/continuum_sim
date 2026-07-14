"""Live PCC-versus-MuJoCo geometry diagnostics for manual tendon debugging."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.bending_space import BendingSpaceModel
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.system.types import RobotSystemState


@dataclass(frozen=True)
class ArmPccMujocoComparison:
    """One arm's synchronized PCC and MuJoCo geometry."""

    arm_name: str
    tendon_displacement_m: np.ndarray
    bending_state_rad_per_m: np.ndarray
    pcc_centerline_world_m: np.ndarray
    mujoco_centerline_world_m: np.ndarray
    pcc_tip_world_m: np.ndarray
    mujoco_tip_world_m: np.ndarray
    tip_error_world_m: np.ndarray
    tip_error_mount_m: np.ndarray
    tip_error_norm_m: float
    compatibility_residual_norm_m: float


def compare_pcc_mujoco_state(
    assembly: RobotAssemblyConfig,
    state: RobotSystemState,
    *,
    samples_per_segment: int = 21,
) -> dict[str, ArmPccMujocoComparison]:
    """Compare PCC FK with measured MuJoCo sites for every enabled arm."""

    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2.")
    expected_names = tuple(arm.name for arm in assembly.enabled_arms)
    if set(state.arms) != set(expected_names):
        raise ValueError(
            "MuJoCo state arms must match the enabled assembly arms: "
            f"expected {sorted(expected_names)}, got {sorted(state.arms)}."
        )
    base_pose = _mujoco_base_pose(state)
    result: dict[str, ArmPccMujocoComparison] = {}
    for arm_config in assembly.enabled_arms:
        arm_name = arm_config.name
        arm_state = state.arms[arm_name]
        tendon = _finite_vector(
            arm_state.tendon_displacement_m,
            arm_config.spatial_arm.tendon_count,
            f"{arm_name} tendon displacement",
        )
        mujoco_tip = _finite_vector(
            arm_state.tip_pose_world.position,
            3,
            f"{arm_name} MuJoCo tip position",
        )
        mujoco_centerline = _finite_points(
            arm_state.centerline_world,
            f"{arm_name} MuJoCo centerline",
        )
        model = BendingSpaceModel.from_arm(
            arm_config.spatial_arm.params,
            arm_config.spatial_arm.tendons,
        )
        bending = model.estimate(tendon)
        fk = forward_kinematics(
            model.to_q(bending),
            arm_config.spatial_arm.params,
            samples_per_segment=samples_per_segment,
        )
        world_mount = base_pose.compose(arm_config.mount_pose)
        pcc_centerline = world_mount.transform_points(fk.centerline)
        pcc_tip = world_mount.transform_point(fk.tip_pose[:3, 3])
        error_world = pcc_tip - mujoco_tip
        rotation_world_from_mount = world_mount.as_matrix()[:3, :3]
        error_mount = rotation_world_from_mount.T @ error_world
        result[arm_name] = ArmPccMujocoComparison(
            arm_name=arm_name,
            tendon_displacement_m=tendon.copy(),
            bending_state_rad_per_m=bending.copy(),
            pcc_centerline_world_m=pcc_centerline,
            mujoco_centerline_world_m=mujoco_centerline,
            pcc_tip_world_m=pcc_tip,
            mujoco_tip_world_m=mujoco_tip,
            tip_error_world_m=error_world,
            tip_error_mount_m=error_mount,
            tip_error_norm_m=float(np.linalg.norm(error_world)),
            compatibility_residual_norm_m=model.residual_norm(tendon),
        )
    return result


def format_pcc_mujoco_diagnostics(
    state: RobotSystemState,
    comparisons: Mapping[str, ArmPccMujocoComparison],
    *,
    control_space: str,
    sample_count: int,
    status: str = "",
) -> str:
    """Format compact coordinates and errors for the tendon-control panel."""

    mode = "compatible" if control_space == "bending_compatible" else "raw tendon"
    lines = [
        f"time: {state.time_s:.4f} s   mode: {mode}",
        f"samples: {sample_count}",
    ]
    for arm_name, comparison in comparisons.items():
        pcc_mm = 1000.0 * comparison.pcc_tip_world_m
        mujoco_mm = 1000.0 * comparison.mujoco_tip_world_m
        error_mm = 1000.0 * comparison.tip_error_mount_m
        lines.extend(
            (
                "",
                f"[{arm_name}] world position [mm]",
                f"  PCC: {pcc_mm[0]: 8.3f} {pcc_mm[1]: 8.3f} {pcc_mm[2]: 8.3f}",
                f"   MJ: {mujoco_mm[0]: 8.3f} {mujoco_mm[1]: 8.3f} {mujoco_mm[2]: 8.3f}",
                f"  dM : {error_mm[0]: 8.3f} {error_mm[1]: 8.3f} {error_mm[2]: 8.3f}",
                f"  |d|: {1000.0 * comparison.tip_error_norm_m:8.3f} mm",
                "  compatibility residual: "
                f"{1000.0 * comparison.compatibility_residual_norm_m:.6f} mm",
            )
        )
    if status:
        lines.extend(("", status))
    return "\n".join(lines)


class MujocoPccOverlay:
    """Draw synchronized PCC and measured MuJoCo geometry in a user scene."""

    _PCC_COLORS = (
        np.array([0.90, 0.10, 1.00, 1.00], dtype=np.float32),
        np.array([0.58, 0.20, 0.88, 1.00], dtype=np.float32),
    )
    _MUJOCO_COLORS = (
        np.array([0.00, 0.95, 1.00, 1.00], dtype=np.float32),
        np.array([0.00, 0.62, 0.76, 1.00], dtype=np.float32),
    )
    _ERROR_COLOR = np.array([1.00, 0.08, 0.04, 0.95], dtype=np.float32)

    def __init__(self, viewer, mujoco_module) -> None:
        self.viewer = viewer
        self.mujoco = mujoco_module

    def update(
        self,
        comparisons: Mapping[str, ArmPccMujocoComparison],
    ) -> None:
        scene = getattr(self.viewer, "user_scn", None)
        if scene is None:
            raise RuntimeError("The MuJoCo viewer does not expose user_scn overlays.")
        scene.ngeom = 0
        for arm_index, comparison in enumerate(comparisons.values()):
            color_index = min(arm_index, len(self._PCC_COLORS) - 1)
            self._add_polyline(
                scene,
                comparison.mujoco_centerline_world_m,
                radius=0.00055,
                rgba=self._MUJOCO_COLORS[color_index],
            )
            self._add_polyline(
                scene,
                comparison.pcc_centerline_world_m,
                radius=0.00075,
                rgba=self._PCC_COLORS[color_index],
            )
            self._add_sphere(
                scene,
                comparison.mujoco_tip_world_m,
                radius=0.0030,
                rgba=self._MUJOCO_COLORS[color_index],
            )
            self._add_sphere(
                scene,
                comparison.pcc_tip_world_m,
                radius=0.0030,
                rgba=self._PCC_COLORS[color_index],
            )
            self._add_connector(
                scene,
                comparison.mujoco_tip_world_m,
                comparison.pcc_tip_world_m,
                radius=0.0008,
                rgba=self._ERROR_COLOR,
            )

    def _add_sphere(self, scene, position, *, radius: float, rgba: np.ndarray) -> None:
        geom = self._next_geom(scene)
        self.mujoco.mjv_initGeom(
            geom,
            self.mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([radius, 0.0, 0.0], dtype=float),
            np.asarray(position, dtype=float),
            np.eye(3, dtype=float).reshape(9),
            rgba,
        )

    def _add_polyline(
        self,
        scene,
        points: np.ndarray,
        *,
        radius: float,
        rgba: np.ndarray,
    ) -> None:
        for start, end in zip(points[:-1], points[1:], strict=True):
            if float(np.linalg.norm(end - start)) <= 1.0e-12:
                continue
            self._add_connector(
                scene,
                start,
                end,
                radius=radius,
                rgba=rgba,
            )

    def _add_connector(
        self,
        scene,
        start: np.ndarray,
        end: np.ndarray,
        *,
        radius: float,
        rgba: np.ndarray,
    ) -> None:
        if float(np.linalg.norm(np.asarray(end) - np.asarray(start))) <= 1.0e-12:
            return
        geom = self._next_geom(scene)
        self.mujoco.mjv_connector(
            geom,
            self.mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(radius),
            np.ascontiguousarray(start, dtype=np.float64),
            np.ascontiguousarray(end, dtype=np.float64),
        )
        geom.rgba[:] = rgba

    @staticmethod
    def _next_geom(scene):
        if int(scene.ngeom) >= int(scene.maxgeom):
            raise RuntimeError(
                "MuJoCo user-scene overlay capacity was exhausted; reduce "
                "--samples-per-segment."
            )
        geom = scene.geoms[int(scene.ngeom)]
        scene.ngeom += 1
        return geom


class PccMujocoSampleRecorder:
    """Keep synchronized samples in memory and save only on explicit request."""

    def __init__(self, *, scenario_name: str) -> None:
        self.scenario_name = str(scenario_name)
        self._rows: list[dict[str, object]] = []
        self._last_state_identity: int | None = None
        self._last_time_s: float | None = None
        self._session = 0
        self._state_sample_count = 0

    @property
    def sample_count(self) -> int:
        return self._state_sample_count

    def record(
        self,
        state: RobotSystemState,
        comparisons: Mapping[str, ArmPccMujocoComparison],
        *,
        control_space: str,
    ) -> None:
        state_identity = id(state)
        if state_identity == self._last_state_identity:
            return
        time_s = float(state.time_s)
        if self._last_time_s is not None and time_s < self._last_time_s - 1.0e-12:
            self._session += 1
        for comparison in comparisons.values():
            row: dict[str, object] = {
                "scenario": self.scenario_name,
                "session": self._session,
                "time_s": time_s,
                "arm": comparison.arm_name,
                "control_space": control_space,
            }
            for index, value in enumerate(
                comparison.tendon_displacement_m,
                start=1,
            ):
                row[f"tendon_{index}_m"] = float(value)
            _add_xyz(row, "pcc_tip_world_m", comparison.pcc_tip_world_m)
            _add_xyz(row, "mujoco_tip_world_m", comparison.mujoco_tip_world_m)
            _add_xyz(row, "error_world_m", comparison.tip_error_world_m)
            _add_xyz(row, "error_mount_m", comparison.tip_error_mount_m)
            row["error_norm_m"] = comparison.tip_error_norm_m
            row["compatibility_residual_norm_m"] = (
                comparison.compatibility_residual_norm_m
            )
            self._rows.append(row)
        self._last_state_identity = state_identity
        self._last_time_s = time_s
        self._state_sample_count += 1

    def save_csv(self, path: str | Path) -> Path:
        if not self._rows:
            raise ValueError("No PCC/MuJoCo samples are available to save.")
        output_path = Path(path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self._rows[0])
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)
        return output_path


def default_pcc_debug_csv_path(project_root: str | Path) -> Path:
    """Return a timestamped path without creating it or its directory."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(project_root) / "output" / "diagnostics" / (
        f"mujoco_pcc_manual_{timestamp}.csv"
    )


def _mujoco_base_pose(state: RobotSystemState) -> Pose6D:
    raw_pose = state.metadata.get("mujoco_mobile_base_frame_pose")
    if raw_pose is None:
        raise ValueError(
            "MuJoCo state metadata is missing 'mujoco_mobile_base_frame_pose'."
        )
    matrix = np.asarray(raw_pose, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            "mujoco_mobile_base_frame_pose must be a finite 4x4 transform."
        )
    return Pose6D.from_matrix(matrix)


def _finite_vector(values: object, size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite vector with shape ({size},).")
    return result


def _finite_points(values: object, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] != 3:
        raise ValueError(f"{label} must have shape (N, 3) with N >= 2.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values.")
    return result.copy()


def _add_xyz(row: dict[str, object], prefix: str, values: np.ndarray) -> None:
    for axis, value in zip("xyz", values, strict=True):
        row[f"{prefix}_{axis}"] = float(value)


__all__ = [
    "ArmPccMujocoComparison",
    "MujocoPccOverlay",
    "PccMujocoSampleRecorder",
    "compare_pcc_mujoco_state",
    "default_pcc_debug_csv_path",
    "format_pcc_mujoco_diagnostics",
]
