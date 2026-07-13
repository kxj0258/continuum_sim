"""Reduced-order MuJoCo backend for the three-segment arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from continuum_sim.backends.base_types import BackendState
from continuum_sim.config import MujocoConfig, load_mujoco_config
from continuum_sim.control.dual_arm_adapter import DualArmCommandAdapter
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.model.segment_followers import (
    SEGMENT_2DOF_Q_SIZE,
    sample_segment_followers,
    segment_2dof_forward_kinematics,
)


class MujocoBackend:
    """Thin wrapper around the optional MuJoCo Python bindings."""

    def __init__(
        self,
        config: MujocoConfig,
        mujoco_module: Any | None = None,
        *,
        xml_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self._mujoco = mujoco_module or _import_mujoco()
        model_xml_path = (
            Path(xml_path).resolve()
            if xml_path is not None
            else _model_xml_path_for_control_mode(config)
        )
        self.model = self._mujoco.MjModel.from_xml_path(str(model_xml_path))
        self._validate_loaded_model_matches_control_mode()
        self.model.opt.timestep = config.solver.timestep
        self.model.opt.iterations = config.solver.iterations
        if hasattr(self._mujoco.mjtIntegrator, "mjINT_IMPLICITFAST"):
            integrators = {
                "euler": self._mujoco.mjtIntegrator.mjINT_EULER,
                "implicit": self._mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "implicitfast": self._mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                "rk4": self._mujoco.mjtIntegrator.mjINT_RK4,
            }
        else:
            integrators = {
                "euler": self._mujoco.mjtIntegrator.mjINT_EULER,
                "implicit": self._mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "implicitfast": self._mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "rk4": self._mujoco.mjtIntegrator.mjINT_RK4,
            }
        self.model.opt.integrator = integrators[config.solver.integrator]
        self.model.opt.gravity[:] = np.asarray(
            _effective_gravity_vector(config),
            dtype=float,
        )

        self.data = self._mujoco.MjData(self.model)
        self._neutral_tendon_length = np.zeros((getattr(self.model, "ntendon", 0),), dtype=float)
        self._segment_site_ids = tuple(
            self._site_id(name) for name in config.site_names.segments
        )
        self._tip_site_id = self._site_id(config.site_names.tip)
        self._base_site_id = self._site_id(config.site_names.base)
        self._mobile_base_qpos_addr = self._joint_qpos_addr("mobile_base_freejoint")
        self._mobile_base_qvel_addr = self._joint_qvel_addr("mobile_base_freejoint")
        self._neutral_mobile_base_pose_rpy = np.zeros((6,), dtype=float)
        self._robot_params = (
            ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
            if self._uses_segment_2dof_followers()
            else None
        )
        self._dual_arm_command_adapter = (
            DualArmCommandAdapter.from_robot_config(config.robot_config_path)
            if config.model.type == "dual_distributed_links"
            else None
        )
        self._follower_mocap_ids = self._resolve_follower_mocap_ids()
        self.reset()

    @classmethod
    def from_config(
        cls,
        config: str | Path | MujocoConfig,
        *,
        override_xml_path: str | Path | None = None,
    ) -> "MujocoBackend":
        """Create a backend from a YAML path or an already loaded config."""

        if isinstance(config, MujocoConfig):
            mujoco_config = config
        else:
            mujoco_config = load_mujoco_config(config)
        return cls(mujoco_config, xml_path=override_xml_path)

    def reset(self) -> BackendState:
        """Reset simulation state and return the initial backend state."""

        self._mujoco.mj_resetData(self.model, self.data)
        self.update_follower_poses()
        self._mujoco.mj_forward(self.model, self.data)
        self._neutral_tendon_length = self._absolute_tendon_length()
        self._neutral_mobile_base_pose_rpy = self._absolute_mobile_base_pose_rpy()
        self.update_follower_poses()
        self._mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, control: np.ndarray, n_substeps: int = 20) -> BackendState:
        """Apply a control vector and advance the simulation."""

        control_array, base_pose_rpy = self._split_control_array(control)
        if control_array.shape != (self.model.nu,):
            raise ValueError(f"Expected control with shape ({self.model.nu},), got {control_array.shape}.")
        if n_substeps <= 0:
            raise ValueError(f"n_substeps must be positive, got {n_substeps}.")

        if base_pose_rpy is not None:
            self.set_mobile_base_pose_rpy(base_pose_rpy)
        self.data.ctrl[:] = self._absolute_control_for_model(control_array)
        self.update_follower_poses()
        for _ in range(n_substeps):
            if base_pose_rpy is not None:
                self.set_mobile_base_pose_rpy(base_pose_rpy)
            self._mujoco.mj_step(self.model, self.data)
            if base_pose_rpy is not None:
                self.set_mobile_base_pose_rpy(base_pose_rpy)
            self.update_follower_poses()
            self._mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def get_state(self) -> BackendState:
        """Return a snapshot of time, poses, and generalized coordinates."""

        return BackendState(
            time=float(self.data.time),
            tip_pose=self.get_tip_pose(),
            segment_poses=self.get_segment_poses(),
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            tendon_length=self.get_tendon_length(),
            tendon_velocity=self.get_tendon_velocity(),
            actuator_force=self.get_actuator_force(),
            mocap_pos=self.data.mocap_pos.copy(),
            mocap_quat=self.data.mocap_quat.copy(),
        )

    def get_tip_pose(self) -> np.ndarray:
        """Return the whole-arm tip site pose as a 4x4 SE(3) matrix."""

        if (
            self._uses_segment_2dof_followers()
            and self.config.model.pose_source == "pcc_fk"
        ):
            tip_pose, _segment_poses = self._segment_2dof_fk_poses()
            return tip_pose
        return self._site_pose(self._tip_site_id)

    def get_segment_poses(self) -> np.ndarray:
        """Return segment tip poses in segment_1, segment_2, segment_3 order."""

        if (
            self._uses_segment_2dof_followers()
            and self.config.model.pose_source == "pcc_fk"
        ):
            _tip_pose, segment_poses = self._segment_2dof_fk_poses()
            return np.asarray(segment_poses, dtype=float)
        poses = np.zeros((3, 4, 4), dtype=float)
        for index, site_id in enumerate(self._segment_site_ids):
            poses[index] = self._site_pose(site_id)
        return poses

    def get_site_pose(self, name: str) -> np.ndarray | None:
        """Return a named MuJoCo site pose, or ``None`` when the site is absent."""

        site_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_SITE,
            name,
        )
        if site_id < 0:
            return None
        return self._site_pose(int(site_id))

    def update_follower_poses(self) -> None:
        """Write runtime follower mocap poses from the current 6DOF qpos."""

        if not self._uses_segment_2dof_followers() or self._robot_params is None:
            return
        if not self._follower_mocap_ids:
            return
        q_segment = self._segment_2dof_qpos()
        followers = sample_segment_followers(
            q_segment,
            self._robot_params,
            self.config.model.follower_samples_per_segment,
        )
        if len(followers) != len(self._follower_mocap_ids):
            raise ValueError(
                "Follower pose count does not match loaded MuJoCo mocap bodies: "
                f"{len(followers)} and {len(self._follower_mocap_ids)}."
            )
        for follower, mocap_id in zip(followers, self._follower_mocap_ids, strict=True):
            self.data.mocap_pos[mocap_id] = follower.center_position
            self.data.mocap_quat[mocap_id] = _rotation_matrix_to_quat_wxyz(
                follower.orientation
            )

    def get_tendon_length(self) -> np.ndarray:
        """Return tendon length deltas from the reset neutral length."""

        values = self._absolute_tendon_length()
        if values.size == 0:
            return values
        return values - self._neutral_tendon_length

    def _absolute_tendon_length(self) -> np.ndarray:
        """Return MuJoCo absolute tendon lengths, or an empty array."""

        values = getattr(self.data, "ten_length", None)
        if values is None:
            return np.zeros((0,), dtype=float)
        return np.asarray(values, dtype=float).copy()

    def get_tendon_velocity(self) -> np.ndarray:
        """Return MuJoCo tendon velocities, or an empty array when unavailable."""

        values = getattr(self.data, "ten_velocity", None)
        if values is None:
            return np.zeros((0,), dtype=float)
        return np.asarray(values, dtype=float).copy()

    def get_actuator_force(self) -> np.ndarray:
        """Return actuator forces reported by MuJoCo."""

        values = getattr(self.data, "actuator_force", None)
        if values is None:
            return np.zeros((0,), dtype=float)
        return np.asarray(values, dtype=float).copy()

    def _site_pose(self, site_id: int) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = self.data.site_xmat[site_id].reshape(3, 3).copy()
        pose[:3, 3] = self.data.site_xpos[site_id].copy()
        return pose

    def _site_id(self, name: str) -> int:
        site_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"MuJoCo model is missing site {name!r}.")
        return int(site_id)

    def set_mobile_base_pose_rpy(self, pose_rpy: np.ndarray) -> None:
        """Set the optional mobile-base freejoint pose as xyz + roll/pitch/yaw."""

        if self._mobile_base_qpos_addr is None:
            raise ValueError("MuJoCo model does not define mobile_base_freejoint.")
        values = np.asarray(pose_rpy, dtype=float)
        if values.shape != (6,):
            raise ValueError(f"Expected mobile base pose with shape (6,), got {values.shape}.")
        qpos_addr = self._mobile_base_qpos_addr
        target_pose = self._neutral_mobile_base_pose_rpy + values
        self.data.qpos[qpos_addr : qpos_addr + 3] = target_pose[:3]
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = _rpy_to_quat_wxyz(target_pose[3:])
        if self._mobile_base_qvel_addr is not None:
            qvel_addr = self._mobile_base_qvel_addr
            self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def get_mobile_base_pose_rpy(self) -> np.ndarray:
        """Return optional mobile-base freejoint pose as xyz + roll/pitch/yaw."""

        if self._mobile_base_qpos_addr is None:
            return np.zeros((6,), dtype=float)
        return self._absolute_mobile_base_pose_rpy() - self._neutral_mobile_base_pose_rpy

    def _absolute_mobile_base_pose_rpy(self) -> np.ndarray:
        if self._mobile_base_qpos_addr is None:
            return np.zeros((6,), dtype=float)
        qpos_addr = self._mobile_base_qpos_addr
        xyz = np.asarray(self.data.qpos[qpos_addr : qpos_addr + 3], dtype=float)
        quat = np.asarray(self.data.qpos[qpos_addr + 3 : qpos_addr + 7], dtype=float)
        return np.concatenate((xyz, _quat_wxyz_to_rpy(quat)))

    def _validate_loaded_model_matches_control_mode(self) -> None:
        if self.config.control_mode == "tendon_position":
            if self.model.nu != self.config.tendon_model.count:
                raise ValueError(
                    "tendon_position mode expected "
                    f"{self.config.tendon_model.count} actuators, got {self.model.nu}."
                )
            if getattr(self.model, "ntendon", 0) != self.config.tendon_model.count:
                raise ValueError(
                    "tendon_position mode expected "
                    f"{self.config.tendon_model.count} tendons, got "
                    f"{getattr(self.model, 'ntendon', 0)}."
                )
        if self._uses_segment_2dof_followers() and self.model.nv != SEGMENT_2DOF_Q_SIZE:
            raise ValueError(
                "segment_2dof_followers expected 6 generalized velocities, "
                f"got {self.model.nv}."
            )

    def _uses_segment_2dof_followers(self) -> bool:
        return self.config.model.type == "segment_2dof_followers"

    def _split_control_array(self, control: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        control_array = np.asarray(control, dtype=float)
        if (
            self._mobile_base_qpos_addr is None
            and control_array.shape == (self.config.tendon_model.count + 6,)
        ):
            raise ValueError(
                "Received tendon+mobile-base control, but the loaded MuJoCo XML "
                "does not define mobile_base_freejoint. Regenerate the mobile-base wrapped XML."
            )
        if (
            self._mobile_base_qpos_addr is not None
            and control_array.shape == (self.config.tendon_model.count + 6,)
        ):
            tendon_control = control_array[: self.config.tendon_model.count]
            return self._control_array_for_model(tendon_control), control_array[-6:]
        return self._control_array_for_model(control_array), None

    def _control_array_for_model(self, control: np.ndarray) -> np.ndarray:
        control_array = np.asarray(control, dtype=float)
        if self._dual_arm_command_adapter is not None:
            return self._dual_arm_command_adapter.adapt(control_array)
        return control_array

    def _absolute_control_for_model(self, tendon_delta_control: np.ndarray) -> np.ndarray:
        if (
            self.config.control_mode == "tendon_position"
            and self.config.tendon_model.type == "spatial"
        ):
            if self._neutral_tendon_length.shape != tendon_delta_control.shape:
                raise ValueError(
                    "Neutral tendon length shape does not match control shape: "
                    f"{self._neutral_tendon_length.shape} and {tendon_delta_control.shape}."
                )
            return self._neutral_tendon_length + tendon_delta_control
        return tendon_delta_control

    def _segment_2dof_qpos(self) -> np.ndarray:
        qpos = np.asarray(self.data.qpos, dtype=float)
        if qpos.shape[0] < SEGMENT_2DOF_Q_SIZE:
            raise ValueError(
                "segment_2dof_followers expected at least 6 qpos entries, "
                f"got {qpos.shape[0]}."
            )
        return qpos[:SEGMENT_2DOF_Q_SIZE].copy()

    def _segment_2dof_fk_poses(self) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        if self._robot_params is None:
            raise RuntimeError("Robot params are not loaded for segment_2dof_followers.")
        return segment_2dof_forward_kinematics(
            self._segment_2dof_qpos(),
            self._robot_params,
        )

    def _resolve_follower_mocap_ids(self) -> tuple[int, ...]:
        if not self._uses_segment_2dof_followers():
            return ()
        ids: list[int] = []
        expected = self.config.model.follower_samples_per_segment * 3
        for segment_index in range(3):
            for sample_index in range(self.config.model.follower_samples_per_segment):
                name = f"follower_segment_{segment_index + 1}_sample_{sample_index + 1}"
                body_id = self._body_id(name)
                mocap_id = int(self.model.body_mocapid[body_id])
                if mocap_id < 0:
                    raise ValueError(f"Follower body {name!r} is not a mocap body.")
                ids.append(mocap_id)
        if len(ids) != expected:
            raise ValueError(f"Expected {expected} follower bodies, got {len(ids)}.")
        return tuple(ids)

    def _body_id(self, name: str) -> int:
        body_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_BODY,
            name,
        )
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing body {name!r}.")
        return int(body_id)

    def _joint_qpos_addr(self, name: str) -> int | None:
        joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            return None
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_qvel_addr(self, name: str) -> int | None:
        joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            return None
        return int(self.model.jnt_dofadr[joint_id])


def _import_mujoco() -> Any:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo backend requires the optional 'mujoco' package. "
            "Install it with `pip install -e .[mujoco]`."
        ) from exc
    return mujoco


def _model_xml_path_for_control_mode(config: MujocoConfig) -> Path:
    if config.control_mode == "position_joint":
        return config.xml_path
    if config.control_mode == "tendon_position":
        return config.tendon_xml_path
    raise ValueError(f"Unsupported MuJoCo control_mode {config.control_mode!r}.")


def _effective_gravity_vector(config: MujocoConfig) -> tuple[float, float, float]:
    if not config.gravity.enabled:
        return (0.0, 0.0, 0.0)
    return config.gravity.vector_m_s2


def _rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected rotation with shape (3, 3), got {matrix.shape}.")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=float,
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=float,
            )
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr = np.cos(0.5 * roll)
    sr = np.sin(0.5 * roll)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.zeros((3,), dtype=float)
    w, x, y, z = q / norm
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)
