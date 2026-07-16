"""Post-run PCC diagnostics for MuJoCo system trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from continuum_sim.kinematics.pcc import (
    DEFAULT_PCC_KINEMATICS_MODE,
    PCCKinematicsMode,
    forward_kinematics,
)
from continuum_sim.kinematics.whole_body import (
    assemble_whole_body_jacobian,
    base_point_jacobian_world,
    bending_position_jacobian,
    rotate_position_jacobian_to_world,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.bending_space import BendingSpaceModel
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.system.control_layout import ControlLayout
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


def build_mujoco_pcc_diagnostic_arrays(
    assembly: RobotAssemblyConfig,
    states: Sequence[RobotSystemState],
    commands: Sequence[RobotSystemCommand],
    *,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """Compare measured MuJoCo motion with PCC FK and Jacobian predictions."""

    if not states:
        return {}
    if stride < 1:
        raise ValueError("mujoco_pcc_diagnostics stride must be >= 1.")
    state_indices = tuple(range(0, len(states), stride))
    sampled_states = tuple(states[index] for index in state_indices)
    sampled_command_indices = tuple(
        index for index in state_indices if index < len(commands)
    )
    sampled_commands = tuple(commands[index] for index in sampled_command_indices)
    arrays: dict[str, np.ndarray] = {}
    arrays["mujoco_pcc_diagnostics_state_index"] = np.asarray(
        state_indices,
        dtype=int,
    )
    arrays["mujoco_pcc_diagnostics_time_s"] = np.asarray(
        [state.time_s for state in sampled_states],
        dtype=float,
    )
    arrays["mujoco_pcc_diagnostics_command_index"] = np.asarray(
        sampled_command_indices,
        dtype=int,
    )
    arrays["mujoco_pcc_diagnostics_command_time_s"] = np.asarray(
        [sampled_states[index].time_s for index in range(len(sampled_commands))],
        dtype=float,
    )
    available_arms = set(sampled_states[-1].arms)
    for arm in assembly.enabled_arms:
        if arm.name not in available_arms:
            continue
        arrays.update(
            _arm_diagnostic_arrays(
                assembly,
                arm.name,
                sampled_states,
                sampled_commands,
                kinematics_mode=kinematics_mode,
            )
        )
    return arrays


def _arm_diagnostic_arrays(
    assembly: RobotAssemblyConfig,
    arm_name: str,
    states: Sequence[RobotSystemState],
    commands: Sequence[RobotSystemCommand],
    *,
    kinematics_mode: PCCKinematicsMode,
) -> dict[str, np.ndarray]:
    arm_config = assembly.arms[arm_name]
    params = arm_config.spatial_arm.params
    tendons = arm_config.spatial_arm.tendons
    model = BendingSpaceModel.from_arm(params, tendons)
    layout = ControlLayout.from_assembly(assembly)
    state_count = len(states)

    bending_state = np.empty((state_count, model.bending_size), dtype=float)
    mujoco_tip_world = np.empty((state_count, 3), dtype=float)
    mujoco_tip_mount = np.empty((state_count, 3), dtype=float)
    pcc_tip_mount = np.empty((state_count, 3), dtype=float)
    pcc_tip_world = np.empty((state_count, 3), dtype=float)
    bending_jacobian_mount = np.empty(
        (state_count, 3, model.bending_size),
        dtype=float,
    )
    bending_jacobian_world = np.empty_like(bending_jacobian_mount)
    software_frame_bending_jacobian_world = np.empty_like(
        bending_jacobian_mount
    )
    reconstructed_system_jacobian_world = np.empty(
        (state_count, 3, layout.size),
        dtype=float,
    )
    tendon_jacobian_mount = np.empty(
        (state_count, 3, model.tendon_count),
        dtype=float,
    )
    tendon_jacobian_world = np.empty_like(tendon_jacobian_mount)
    singular_values = np.empty((state_count, 3), dtype=float)
    jacobian_rank = np.empty(state_count, dtype=int)
    jacobian_condition = np.empty(state_count, dtype=float)
    jacobian_mount_row_norm = np.empty((state_count, 3), dtype=float)
    jacobian_world_row_norm = np.empty((state_count, 3), dtype=float)
    measured_arm_velocity_mount = np.empty((state_count, 3), dtype=float)
    measured_arm_velocity_world = np.empty((state_count, 3), dtype=float)
    measured_base_velocity_world = np.empty((state_count, 3), dtype=float)
    measured_total_velocity_world = np.empty((state_count, 3), dtype=float)

    for index, state in enumerate(states):
        arm_state = state.arms[arm_name]
        mujoco_base_pose = _mujoco_base_pose(state)
        mount = mujoco_base_pose.compose(arm_config.mount_pose)
        mount_matrix = mount.as_matrix()
        rotation_world_from_mount = mount_matrix[:3, :3]
        measured_world = arm_state.tip_pose_world.position
        measured_mount = rotation_world_from_mount.T @ (
            measured_world - mount.position
        )
        bending = model.estimate(arm_state.tendon_displacement_m)
        q = model.to_q(bending)
        pcc_mount = forward_kinematics(
            q,
            params,
            kinematics_mode=kinematics_mode,
        ).tip_pose[:3, 3]
        jacobian_mount = bending_position_jacobian(
            q,
            params,
            tendons,
            kinematics_mode=kinematics_mode,
        )
        jacobian_world = rotate_position_jacobian_to_world(
            jacobian_mount,
            rotation_world_from_mount,
        )
        software_mount = state.base.pose.compose(arm_config.mount_pose)
        software_frame_jacobian_world = rotate_position_jacobian_to_world(
            jacobian_mount,
            software_mount.as_matrix()[:3, :3],
        )
        software_base_jacobian = base_point_jacobian_world(
            measured_world,
            state.base.pose.position,
        )
        reconstructed_system_jacobian = assemble_whole_body_jacobian(
            layout,
            arm_name,
            software_base_jacobian,
            software_frame_jacobian_world,
        )
        physical_jacobian_mount = jacobian_mount @ model.pseudoinverse
        physical_jacobian_world = jacobian_world @ model.pseudoinverse
        values = np.linalg.svd(jacobian_mount, compute_uv=False)
        arm_velocity_mount = physical_jacobian_mount @ arm_state.tendon_velocity_mps
        arm_velocity_world = rotation_world_from_mount @ arm_velocity_mount
        base_velocity_world = base_point_jacobian_world(
            measured_world,
            mujoco_base_pose.position,
        ) @ state.base.twist_world

        bending_state[index] = bending
        mujoco_tip_world[index] = measured_world
        mujoco_tip_mount[index] = measured_mount
        pcc_tip_mount[index] = pcc_mount
        pcc_tip_world[index] = mount.transform_point(pcc_mount)
        bending_jacobian_mount[index] = jacobian_mount
        bending_jacobian_world[index] = jacobian_world
        software_frame_bending_jacobian_world[index] = (
            software_frame_jacobian_world
        )
        reconstructed_system_jacobian_world[index] = (
            reconstructed_system_jacobian
        )
        tendon_jacobian_mount[index] = physical_jacobian_mount
        tendon_jacobian_world[index] = physical_jacobian_world
        singular_values[index] = values
        jacobian_rank[index] = int(np.linalg.matrix_rank(jacobian_mount))
        jacobian_condition[index] = float(np.linalg.cond(jacobian_mount))
        jacobian_mount_row_norm[index] = np.linalg.norm(jacobian_mount, axis=1)
        jacobian_world_row_norm[index] = np.linalg.norm(jacobian_world, axis=1)
        measured_arm_velocity_mount[index] = arm_velocity_mount
        measured_arm_velocity_world[index] = arm_velocity_world
        measured_base_velocity_world[index] = base_velocity_world
        measured_total_velocity_world[index] = (
            base_velocity_world + arm_velocity_world
        )

    position_error_mount = pcc_tip_mount - mujoco_tip_mount
    position_error_world = pcc_tip_world - mujoco_tip_world
    transition_count = min(len(commands), state_count - 1)
    transition = _transition_diagnostics(
        arm_name=arm_name,
        model=model,
        states=states,
        commands=commands,
        transition_count=transition_count,
        mujoco_tip_mount=mujoco_tip_mount,
        pcc_tip_mount=pcc_tip_mount,
        bending_jacobian_mount=bending_jacobian_mount,
        software_frame_bending_jacobian_world=(
            software_frame_bending_jacobian_world
        ),
        reconstructed_system_jacobian_world=(
            reconstructed_system_jacobian_world
        ),
        layout=layout,
    )
    prefix = f"arm_{arm_name}"
    result = {
        f"{prefix}_mujoco_tip_position_world_m": mujoco_tip_world,
        f"{prefix}_mujoco_tip_position_mount_m": mujoco_tip_mount,
        f"{prefix}_pcc_bending_state_rad_per_m": bending_state,
        f"{prefix}_pcc_tip_position_mount_m": pcc_tip_mount,
        f"{prefix}_pcc_tip_position_world_m": pcc_tip_world,
        f"{prefix}_pcc_mujoco_tip_error_mount_m": position_error_mount,
        f"{prefix}_pcc_mujoco_tip_error_world_m": position_error_world,
        f"{prefix}_pcc_mujoco_tip_error_norm_m": np.linalg.norm(
            position_error_mount,
            axis=1,
        ),
        f"{prefix}_pcc_bending_jacobian_mount": bending_jacobian_mount,
        f"{prefix}_pcc_bending_jacobian_world": bending_jacobian_world,
        f"{prefix}_pcc_software_frame_bending_jacobian_world": (
            software_frame_bending_jacobian_world
        ),
        f"{prefix}_pcc_reconstructed_system_jacobian_world": (
            reconstructed_system_jacobian_world
        ),
        f"{prefix}_pcc_tendon_jacobian_mount": tendon_jacobian_mount,
        f"{prefix}_pcc_tendon_jacobian_world": tendon_jacobian_world,
        f"{prefix}_pcc_jacobian_singular_values": singular_values,
        f"{prefix}_pcc_jacobian_rank": jacobian_rank,
        f"{prefix}_pcc_jacobian_condition_number": jacobian_condition,
        f"{prefix}_pcc_jacobian_mount_row_norm": jacobian_mount_row_norm,
        f"{prefix}_pcc_jacobian_world_row_norm": jacobian_world_row_norm,
        f"{prefix}_pcc_tip_velocity_from_measured_tendon_mount_mps": (
            measured_arm_velocity_mount
        ),
        f"{prefix}_pcc_tip_velocity_from_measured_tendon_world_mps": (
            measured_arm_velocity_world
        ),
        f"{prefix}_base_tip_velocity_from_measured_twist_world_mps": (
            measured_base_velocity_world
        ),
        f"{prefix}_pcc_total_velocity_from_measured_state_world_mps": (
            measured_total_velocity_world
        ),
    }
    result.update({f"{prefix}_{key}": value for key, value in transition.items()})
    return result


def _transition_diagnostics(
    *,
    arm_name: str,
    model: BendingSpaceModel,
    states: Sequence[RobotSystemState],
    commands: Sequence[RobotSystemCommand],
    transition_count: int,
    mujoco_tip_mount: np.ndarray,
    pcc_tip_mount: np.ndarray,
    bending_jacobian_mount: np.ndarray,
    software_frame_bending_jacobian_world: np.ndarray,
    reconstructed_system_jacobian_world: np.ndarray,
    layout: ControlLayout,
) -> dict[str, np.ndarray]:
    shape = (transition_count, 3)
    mujoco_velocity_mount = np.full(shape, np.nan, dtype=float)
    mujoco_velocity_world = np.full(shape, np.nan, dtype=float)
    pcc_fk_velocity_mount = np.full(shape, np.nan, dtype=float)
    jacobian_velocity_from_delta = np.full(shape, np.nan, dtype=float)
    command_arm_velocity_mount = np.full(shape, np.nan, dtype=float)
    command_arm_velocity_world = np.full(shape, np.nan, dtype=float)
    command_base_velocity_world = np.full(shape, np.nan, dtype=float)
    command_total_velocity_world = np.full(shape, np.nan, dtype=float)

    for index in range(transition_count):
        current = states[index]
        following = states[index + 1]
        dt = float(following.time_s - current.time_s)
        if not np.isfinite(dt) or dt <= 0.0:
            continue
        current_arm = current.arms[arm_name]
        following_arm = following.arms[arm_name]
        command = commands[index]
        tendon_rate_from_delta = (
            following_arm.tendon_displacement_m
            - current_arm.tendon_displacement_m
        ) / dt
        bending_rate_from_delta = model.estimate(tendon_rate_from_delta)
        command_bending_rate = model.estimate(
            command.arms[arm_name].tendon_rate_mps
        )
        actual_tip_world = current_arm.tip_pose_world.position
        base_velocity = base_point_jacobian_world(
            actual_tip_world,
            current.base.pose.position,
        ) @ command.base_twist_world

        mujoco_velocity_mount[index] = (
            mujoco_tip_mount[index + 1] - mujoco_tip_mount[index]
        ) / dt
        mujoco_velocity_world[index] = (
            following_arm.tip_pose_world.position - actual_tip_world
        ) / dt
        pcc_fk_velocity_mount[index] = (
            pcc_tip_mount[index + 1] - pcc_tip_mount[index]
        ) / dt
        jacobian_velocity_from_delta[index] = (
            bending_jacobian_mount[index] @ bending_rate_from_delta
        )
        command_arm_velocity_mount[index] = (
            bending_jacobian_mount[index] @ command_bending_rate
        )
        command_arm_velocity_world[index] = (
            software_frame_bending_jacobian_world[index] @ command_bending_rate
        )
        command_base_velocity_world[index] = base_velocity
        system_velocity = np.zeros(layout.size, dtype=float)
        if layout.base_size:
            system_velocity[layout.base] = command.base_twist_world
        system_velocity[layout.arms[arm_name]] = command_bending_rate
        command_total_velocity_world[index] = (
            reconstructed_system_jacobian_world[index] @ system_velocity
        )

    jacobian_linearization_residual = (
        jacobian_velocity_from_delta - pcc_fk_velocity_mount
    )
    pcc_mujoco_model_velocity_residual = (
        pcc_fk_velocity_mount - mujoco_velocity_mount
    )
    command_mujoco_velocity_residual = (
        command_total_velocity_world - mujoco_velocity_world
    )
    return {
        "mujoco_tip_velocity_fd_mount_mps": mujoco_velocity_mount,
        "mujoco_tip_velocity_fd_world_mps": mujoco_velocity_world,
        "pcc_tip_velocity_fd_mount_mps": pcc_fk_velocity_mount,
        "pcc_tip_velocity_from_tendon_delta_mount_mps": (
            jacobian_velocity_from_delta
        ),
        "pcc_jacobian_linearization_residual_mount_mps": (
            jacobian_linearization_residual
        ),
        "pcc_jacobian_linearization_residual_norm_mps": np.linalg.norm(
            jacobian_linearization_residual,
            axis=1,
        ),
        "pcc_mujoco_model_velocity_residual_mount_mps": (
            pcc_mujoco_model_velocity_residual
        ),
        "pcc_mujoco_model_velocity_residual_norm_mps": np.linalg.norm(
            pcc_mujoco_model_velocity_residual,
            axis=1,
        ),
        "pcc_command_arm_tip_velocity_mount_mps": command_arm_velocity_mount,
        "pcc_command_arm_tip_velocity_world_mps": command_arm_velocity_world,
        "base_command_tip_velocity_world_mps": command_base_velocity_world,
        "pcc_command_total_tip_velocity_world_mps": command_total_velocity_world,
        "pcc_command_mujoco_velocity_residual_world_mps": (
            command_mujoco_velocity_residual
        ),
        "pcc_command_mujoco_velocity_residual_norm_mps": np.linalg.norm(
            command_mujoco_velocity_residual,
            axis=1,
        ),
    }


def _mujoco_base_pose(state: RobotSystemState) -> Pose6D:
    raw_pose = state.metadata.get("mujoco_mobile_base_frame_pose")
    if raw_pose is not None:
        matrix = np.asarray(raw_pose, dtype=float)
        if matrix.shape == (4, 4) and np.all(np.isfinite(matrix)):
            return Pose6D.from_matrix(matrix)
    return state.base.pose
