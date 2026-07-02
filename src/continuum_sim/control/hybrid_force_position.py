"""Hybrid force-position wiping controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.actuation.motor_mapping import MotorParams
from continuum_sim.control.differential_ik import damped_least_squares
from continuum_sim.kinematics.differential import (
    bending_position_jacobian,
    bending_rate_to_motor_velocity,
    motor_position_jacobian,
)
from continuum_sim.model.bending_space import BendingSpaceModel
from continuum_sim.model.physical_tendon import PhysicalTendonPath
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.scenes.contact_surfaces import WorkSurfaceConfig
from continuum_sim.tasks.wiping_config import WipingControllerConfig


@dataclass(frozen=True)
class ContactMeasurement:
    """Normal contact measurement with a replaceable source."""

    normal_force_n: float
    signed_distance_m: float
    source: str
    in_contact: bool


def compute_wiping_motor_velocity_command_from_observation(
    actual_tip_position: np.ndarray,
    actual_tendon_delta: np.ndarray,
    target_position: np.ndarray,
    surface: WorkSurfaceConfig,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    config: WipingControllerConfig,
    *,
    measured_normal_force_n: float | None = None,
    contact_radius_m: float = 0.0,
    force_control_enabled: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str | bool]]:
    """Compute a motor velocity command from MuJoCo-observed contact/tendon state."""

    tendon_delta = _as_vector(
        actual_tendon_delta,
        "actual_tendon_delta",
        expected_size=len(physical_tendons),
    )
    bending_model = BendingSpaceModel.from_arm(params, physical_tendons)
    q_est = bending_model.to_q(bending_model.estimate(tendon_delta))
    return _compute_wiping_command_from_state(
        tip_position=_as_position(actual_tip_position, "actual_tip_position"),
        q_est=q_est,
        target_position=_as_position(target_position, "target_position"),
        surface=surface,
        params=params,
        physical_tendons=physical_tendons,
        motor_params=motor_params,
        config=config,
        measured_normal_force_n=measured_normal_force_n,
        contact_radius_m=contact_radius_m,
        force_control_enabled=force_control_enabled,
    )


def compute_wiping_motor_velocity_command_from_state(
    tip_position: np.ndarray,
    q_est: np.ndarray,
    target_position: np.ndarray,
    surface: WorkSurfaceConfig,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    config: WipingControllerConfig,
    *,
    measured_normal_force_n: float | None = None,
    contact_radius_m: float = 0.0,
    force_control_enabled: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str | bool]]:
    """Compute a wiping command from an already-estimated PCC state."""

    return _compute_wiping_command_from_state(
        tip_position=_as_position(tip_position, "tip_position"),
        q_est=_as_vector(q_est, "q_est", expected_size=params.q_size),
        target_position=_as_position(target_position, "target_position"),
        surface=surface,
        params=params,
        physical_tendons=physical_tendons,
        motor_params=motor_params,
        config=config,
        measured_normal_force_n=measured_normal_force_n,
        contact_radius_m=contact_radius_m,
        force_control_enabled=force_control_enabled,
    )


def contact_measurement_from_surface_proxy(
    contact_position: np.ndarray,
    surface: WorkSurfaceConfig,
    config: WipingControllerConfig,
    *,
    measured_normal_force_n: float | None = None,
    contact_radius_m: float = 0.0,
) -> ContactMeasurement:
    """Return true force when available, otherwise a pad-surface distance proxy."""

    center_distance = surface.signed_distance(
        _as_position(contact_position, "contact_position")
    )
    signed_distance = center_distance - max(0.0, float(contact_radius_m))
    if measured_normal_force_n is not None and np.isfinite(measured_normal_force_n):
        force = max(0.0, float(measured_normal_force_n))
        return ContactMeasurement(
            normal_force_n=force,
            signed_distance_m=signed_distance,
            source="mujoco_contact_force",
            in_contact=force > 0.0 or signed_distance <= 0.0,
        )
    penetration = max(0.0, -signed_distance)
    return ContactMeasurement(
        normal_force_n=config.force_proxy_stiffness_n_m * penetration,
        signed_distance_m=signed_distance,
        source="distance_proxy",
        in_contact=signed_distance <= 0.0,
    )


def desired_hybrid_tip_velocity(
    tip_position: np.ndarray,
    target_position: np.ndarray,
    surface: WorkSurfaceConfig,
    config: WipingControllerConfig,
    *,
    measured_normal_force_n: float | None = None,
    contact_radius_m: float = 0.0,
    force_control_enabled: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str | bool]]:
    """Split position and force regulation into tangent and normal velocities."""

    _validate_config(config)
    tip = _as_position(tip_position, "tip_position")
    target = _as_position(target_position, "target_position")
    position_error = target - tip
    normal = surface.normal
    tangent_error = position_error - np.dot(position_error, normal) * normal
    tangent_velocity = config.tangent_position_gain * tangent_error
    tangent_speed = float(np.linalg.norm(tangent_velocity))
    if tangent_speed > config.max_tangent_velocity_m_s:
        tangent_velocity *= config.max_tangent_velocity_m_s / tangent_speed

    contact = contact_measurement_from_surface_proxy(
        tip,
        surface,
        config,
        measured_normal_force_n=measured_normal_force_n,
        contact_radius_m=contact_radius_m,
    )
    force_error = config.target_normal_force_n - contact.normal_force_n
    if force_control_enabled:
        force_velocity = -config.normal_force_gain * force_error
        distance_error = config.target_contact_distance_m - contact.signed_distance_m
        distance_velocity = config.normal_position_gain * distance_error
        normal_scalar_velocity = force_velocity + distance_velocity
    else:
        force_velocity = 0.0
        distance_error = 0.0
        distance_velocity = 0.0
        normal_scalar_velocity = config.tangent_position_gain * float(
            np.dot(position_error, normal)
        )
    normal_scalar_velocity = float(
        np.clip(
            normal_scalar_velocity,
            -config.max_normal_velocity_m_s,
            config.max_normal_velocity_m_s,
        )
    )
    normal_velocity = normal_scalar_velocity * normal
    desired_velocity = tangent_velocity + normal_velocity
    info: dict[str, np.ndarray | float | str | bool] = {
        "position_error": position_error,
        "tangent_position_error": tangent_error,
        "normal_position_error_m": float(np.dot(position_error, normal)),
        "tangent_velocity": tangent_velocity,
        "normal_velocity": normal_velocity,
        "desired_tip_velocity": desired_velocity,
        "normal_force_n": float(contact.normal_force_n),
        "contact_proxy_m": float(contact.signed_distance_m),
        "force_error_n": float(force_error),
        "contact_radius_m": float(max(0.0, contact_radius_m)),
        "force_velocity_m_s": float(force_velocity),
        "distance_error_m": float(distance_error),
        "distance_velocity_m_s": float(distance_velocity),
        "force_control_enabled": bool(force_control_enabled),
        "contact_source": contact.source,
        "in_contact": bool(contact.in_contact),
    }
    return desired_velocity, info


def _compute_wiping_command_from_state(
    *,
    tip_position: np.ndarray,
    q_est: np.ndarray,
    target_position: np.ndarray,
    surface: WorkSurfaceConfig,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    config: WipingControllerConfig,
    measured_normal_force_n: float | None,
    contact_radius_m: float,
    force_control_enabled: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str | bool]]:
    desired_velocity, info = desired_hybrid_tip_velocity(
        tip_position,
        target_position,
        surface,
        config,
        measured_normal_force_n=measured_normal_force_n,
        contact_radius_m=contact_radius_m,
        force_control_enabled=force_control_enabled,
    )
    jacobian = bending_position_jacobian(
        q_est,
        params,
        physical_tendons,
        step=config.finite_difference_step_rad,
    )
    bending_rate = damped_least_squares(jacobian, desired_velocity, config.damping)
    motor_velocity_cmd = bending_rate_to_motor_velocity(
        bending_rate,
        params,
        physical_tendons,
        motor_params,
        max_motor_velocity_rad_s=config.max_motor_velocity_rad_s,
    )
    info.update(
        {
            "q_est": np.asarray(q_est, dtype=float).copy(),
            "tip_position": np.asarray(tip_position, dtype=float).copy(),
            "target_position": np.asarray(target_position, dtype=float).copy(),
            "error_norm": float(np.linalg.norm(target_position - tip_position)),
            "J_bending": jacobian,
            "J_motor": motor_position_jacobian(
                q_est,
                params,
                physical_tendons,
                motor_params,
                step=config.finite_difference_step_rad,
            ),
            "bending_rate": bending_rate,
        }
    )
    return motor_velocity_cmd, info


def _validate_config(config: WipingControllerConfig) -> None:
    if config.damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {config.damping}.")
    if config.tangent_position_gain < 0.0:
        raise ValueError("tangent_position_gain must be non-negative.")
    if config.normal_force_gain < 0.0:
        raise ValueError("normal_force_gain must be non-negative.")
    if config.normal_position_gain < 0.0:
        raise ValueError("normal_position_gain must be non-negative.")
    if config.force_proxy_stiffness_n_m <= 0.0:
        raise ValueError("force_proxy_stiffness_n_m must be positive.")
    if config.max_normal_velocity_m_s <= 0.0:
        raise ValueError("max_normal_velocity_m_s must be positive.")
    if config.max_tangent_velocity_m_s <= 0.0:
        raise ValueError("max_tangent_velocity_m_s must be positive.")
    if config.max_motor_velocity_rad_s <= 0.0:
        raise ValueError("max_motor_velocity_rad_s must be positive.")
    if config.max_contact_force_n <= 0.0:
        raise ValueError("max_contact_force_n must be positive.")
    if config.finite_difference_step_rad <= 0.0:
        raise ValueError("finite_difference_step_rad must be positive.")


def _as_position(values: np.ndarray, name: str) -> np.ndarray:
    return _as_vector(values, name, expected_size=3)


def _as_vector(values: np.ndarray, name: str, *, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"Expected {name} with shape ({expected_size},), got {array.shape}.")
    return array
