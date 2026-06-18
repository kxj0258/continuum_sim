"""Project follower MuJoCo contact wrenches into segment-2DOF coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.model.segment_followers import (
    SEGMENT_2DOF_Q_SIZE,
    sample_segment_followers,
)
from continuum_sim.scenes.contact_surfaces import WorkSurfaceConfig


FOLLOWER_CONTACT_SOURCE = "mujoco_follower_contact_projection"


@dataclass(frozen=True)
class ContactProjectionResult:
    """Aggregated contact force feedback and generalized-force projection."""

    normal_force_n: float
    total_force_world: np.ndarray
    projected_generalized_force_q: np.ndarray
    contact_count: int
    contact_points_world: np.ndarray
    contact_geom_names: tuple[str, ...]
    source: str = FOLLOWER_CONTACT_SOURCE


def zero_contact_projection_result(
    *,
    source: str = FOLLOWER_CONTACT_SOURCE,
) -> ContactProjectionResult:
    """Return a no-contact result with stable shapes."""

    return ContactProjectionResult(
        normal_force_n=0.0,
        total_force_world=np.zeros(3, dtype=float),
        projected_generalized_force_q=np.zeros(SEGMENT_2DOF_Q_SIZE, dtype=float),
        contact_count=0,
        contact_points_world=np.zeros((0, 3), dtype=float),
        contact_geom_names=(),
        source=source,
    )


def project_follower_contacts(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    q_segment: np.ndarray,
    params: ThreeSegmentRobotParams,
    samples_per_segment: int,
    surface: WorkSurfaceConfig,
    scene_geom_prefixes: tuple[str, ...] = ("scene_",),
    finite_difference_step: float = 1.0e-6,
) -> ContactProjectionResult:
    """Project follower contacts from MuJoCo contact wrenches into 6D q space."""

    if finite_difference_step <= 0.0:
        raise ValueError(
            f"finite_difference_step must be positive, got {finite_difference_step}."
        )
    q_array = np.asarray(q_segment, dtype=float)
    if q_array.shape != (SEGMENT_2DOF_Q_SIZE,):
        raise ValueError(
            f"Expected q_segment with shape ({SEGMENT_2DOF_Q_SIZE},), got {q_array.shape}."
        )
    contact_force = getattr(mujoco_module, "mj_contactForce", None)
    if contact_force is None:
        return zero_contact_projection_result()

    followers = sample_segment_followers(q_array, params, samples_per_segment)
    follower_lookup = {follower.name: follower for follower in followers}
    total_force_world = np.zeros(3, dtype=float)
    tau_q = np.zeros(SEGMENT_2DOF_Q_SIZE, dtype=float)
    normal_force = 0.0
    points: list[np.ndarray] = []
    geom_names: list[str] = []

    for contact_id in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[contact_id]
        geom1_name = _geom_name(model, int(contact.geom1))
        geom2_name = _geom_name(model, int(contact.geom2))
        follower_geom_name, other_geom_name = _follower_scene_pair(
            geom1_name,
            geom2_name,
            scene_geom_prefixes,
        )
        if follower_geom_name is None or other_geom_name is None:
            continue

        follower_name = follower_geom_name[: -len("_collision")]
        follower = follower_lookup.get(follower_name)
        if follower is None:
            continue
        wrench_contact = np.zeros(6, dtype=float)
        contact_force(model, data, contact_id, wrench_contact)
        force_world, torque_world = contact_wrench_to_world(contact, wrench_contact)
        point_world = np.asarray(contact.pos, dtype=float).reshape(3).copy()
        offset_local = follower.orientation.T @ (point_world - follower.center_position)
        jacobian_v, jacobian_w = finite_difference_follower_jacobian(
            q_array,
            params,
            samples_per_segment,
            follower.segment_index,
            follower.sample_index,
            offset_local,
            step=finite_difference_step,
        )
        total_force_world += force_world
        tau_q += jacobian_v.T @ force_world + jacobian_w.T @ torque_world
        normal_force += max(0.0, float(np.dot(force_world, surface.normal)))
        points.append(point_world)
        geom_names.append(f"{follower_geom_name}:{other_geom_name}")

    if not points:
        return zero_contact_projection_result()
    return ContactProjectionResult(
        normal_force_n=float(normal_force),
        total_force_world=total_force_world,
        projected_generalized_force_q=tau_q,
        contact_count=len(points),
        contact_points_world=np.asarray(points, dtype=float),
        contact_geom_names=tuple(geom_names),
    )


def contact_wrench_to_world(contact: Any, wrench_contact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform a MuJoCo contact-frame wrench into world coordinates."""

    wrench = np.asarray(wrench_contact, dtype=float)
    if wrench.shape != (6,):
        raise ValueError(f"Expected wrench_contact with shape (6,), got {wrench.shape}.")
    frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
    rotation_contact_to_world = frame.T
    return (
        rotation_contact_to_world @ wrench[:3],
        rotation_contact_to_world @ wrench[3:],
    )


def finite_difference_follower_jacobian(
    q_segment: np.ndarray,
    params: ThreeSegmentRobotParams,
    samples_per_segment: int,
    segment_index: int,
    sample_index: int,
    point_offset_local: np.ndarray | None = None,
    *,
    step: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-difference translational and angular Jacobians for a follower."""

    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}.")
    q_array = np.asarray(q_segment, dtype=float)
    if q_array.shape != (SEGMENT_2DOF_Q_SIZE,):
        raise ValueError(
            f"Expected q_segment with shape ({SEGMENT_2DOF_Q_SIZE},), got {q_array.shape}."
        )
    offset = (
        np.zeros(3, dtype=float)
        if point_offset_local is None
        else np.asarray(point_offset_local, dtype=float)
    )
    if offset.shape != (3,):
        raise ValueError(
            f"Expected point_offset_local with shape (3,), got {offset.shape}."
        )
    base_pose = _selected_follower_pose(
        q_array,
        params,
        samples_per_segment,
        segment_index,
        sample_index,
    )
    base_rotation = base_pose[:3, :3]
    base_point = base_pose[:3, 3] + base_rotation @ offset
    jacobian_v = np.zeros((3, SEGMENT_2DOF_Q_SIZE), dtype=float)
    jacobian_w = np.zeros((3, SEGMENT_2DOF_Q_SIZE), dtype=float)

    for dof_index in range(SEGMENT_2DOF_Q_SIZE):
        q_plus = q_array.copy()
        q_minus = q_array.copy()
        q_plus[dof_index] += step
        q_minus[dof_index] -= step
        pose_plus = _selected_follower_pose(
            q_plus,
            params,
            samples_per_segment,
            segment_index,
            sample_index,
        )
        pose_minus = _selected_follower_pose(
            q_minus,
            params,
            samples_per_segment,
            segment_index,
            sample_index,
        )
        point_plus = pose_plus[:3, 3] + pose_plus[:3, :3] @ offset
        point_minus = pose_minus[:3, 3] + pose_minus[:3, :3] @ offset
        jacobian_v[:, dof_index] = (point_plus - point_minus) / (2.0 * step)

        rotation_delta = pose_plus[:3, :3] @ pose_minus[:3, :3].T
        jacobian_w[:, dof_index] = _rotation_vector_from_matrix(rotation_delta) / (
            2.0 * step
        )

    del base_point, base_rotation
    return jacobian_v, jacobian_w


def apply_projected_qfrc(
    data: Any,
    projected_generalized_force_q: np.ndarray,
    *,
    dof_count: int = SEGMENT_2DOF_Q_SIZE,
) -> None:
    """Write projected follower contact force into MuJoCo qfrc_applied."""

    qfrc_applied = getattr(data, "qfrc_applied", None)
    if qfrc_applied is None:
        return
    tau = np.asarray(projected_generalized_force_q, dtype=float)
    if tau.shape != (dof_count,):
        raise ValueError(f"Expected projected qfrc shape ({dof_count},), got {tau.shape}.")
    qfrc_applied[:dof_count] = tau


def _selected_follower_pose(
    q_segment: np.ndarray,
    params: ThreeSegmentRobotParams,
    samples_per_segment: int,
    segment_index: int,
    sample_index: int,
) -> np.ndarray:
    followers = sample_segment_followers(q_segment, params, samples_per_segment)
    for follower in followers:
        if follower.segment_index == segment_index and follower.sample_index == sample_index:
            return follower.pose
    raise ValueError(
        "No follower pose for "
        f"segment_index={segment_index}, sample_index={sample_index}."
    )


def _geom_name(model: Any, geom_id: int) -> str:
    names = getattr(model, "geom_names", None)
    if names is not None:
        name = names[geom_id]
        if isinstance(name, bytes):
            return name.decode("utf-8")
        return str(name)
    names_blob = getattr(model, "names", None)
    name_addr = getattr(model, "name_geomadr", None)
    if names_blob is None or name_addr is None:
        return f"geom_{geom_id}"
    start = int(name_addr[geom_id])
    if isinstance(names_blob, bytes):
        end = names_blob.find(b"\x00", start)
        return names_blob[start:end].decode("utf-8")
    raw = bytes(names_blob)
    end = raw.find(b"\x00", start)
    return raw[start:end].decode("utf-8")


def _follower_scene_pair(
    geom1_name: str,
    geom2_name: str,
    scene_geom_prefixes: tuple[str, ...],
) -> tuple[str | None, str | None]:
    geom1_is_follower = _is_follower_collision_geom(geom1_name)
    geom2_is_follower = _is_follower_collision_geom(geom2_name)
    geom1_is_scene = _is_scene_geom(geom1_name, scene_geom_prefixes)
    geom2_is_scene = _is_scene_geom(geom2_name, scene_geom_prefixes)
    if geom1_is_follower and geom2_is_scene:
        return geom1_name, geom2_name
    if geom2_is_follower and geom1_is_scene:
        return geom2_name, geom1_name
    return None, None


def _is_follower_collision_geom(name: str) -> bool:
    return name.startswith("follower_segment_") and name.endswith("_collision")


def _is_scene_geom(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected rotation with shape (3, 3), got {matrix.shape}.")
    cos_theta = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1.0e-12:
        return np.array(
            [
                0.5 * (matrix[2, 1] - matrix[1, 2]),
                0.5 * (matrix[0, 2] - matrix[2, 0]),
                0.5 * (matrix[1, 0] - matrix[0, 1]),
            ],
            dtype=float,
        )
    scale = theta / (2.0 * np.sin(theta))
    return scale * np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=float,
    )
