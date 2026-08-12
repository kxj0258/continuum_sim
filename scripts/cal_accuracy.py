"""Single-pose worst-case encoder-accuracy analysis.

The six encoder channels are ordered as
``[theta_kx_1, theta_ky_1, ..., theta_kx_3, theta_ky_3]`` and represent
segment-total bending angles, not PCC curvatures. Project configuration is the
only source of robot geometry, kinematics mode, joint limits and tool TCP.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np

from continuum_sim.application.scenario import load_scenario_config
from continuum_sim.config import load_mujoco_config
from continuum_sim.model.base_pose import Pose6D, quaternion_wxyz_to_rotation_matrix
from continuum_sim.model.bending_space import BendingSpaceModel
from continuum_sim.model.robot_assembly import SpatialArmConfig, load_spatial_arm_config
from continuum_sim.model.segment_followers import segment_2dof_forward_kinematics
from continuum_sim.tools.attachments import load_attachment_config
from continuum_sim.tools.tool_frames import compute_tool_tcp_pose


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARM_CONFIG = ROOT / "configs" / "robots" / "spatial_arm_executor.yaml"
DEFAULT_TOOL_CONFIG = ROOT / "configs" / "tools" / "carbon_remover.yaml"
DEFAULT_SCENARIO_CONFIG = ROOT / "configs" / "scenarios" / "mujoco_manual_control.yaml"
DEFAULT_ACCURACY_DEG = (2.0, 1.0, 0.5, 0.25, 0.1, 0.05)
ENCODER_CHANNELS = 6
SIGNS = np.asarray(tuple(product((-1.0, 1.0), repeat=ENCODER_CHANNELS)))


@dataclass(frozen=True)
class AccuracyModel:
    """Project-backed geometry and limits shared by both accuracy scripts."""

    arm: SpatialArmConfig
    bending_model: BendingSpaceModel
    kinematics_mode: str
    encoder_angle_lower_deg: np.ndarray
    encoder_angle_upper_deg: np.ndarray
    tcp_pose_from_tip: Pose6D

    @property
    def params(self):
        return self.arm.params

    @property
    def flexure_lengths_m(self) -> np.ndarray:
        return np.repeat(
            [segment.effective_flexure_length for segment in self.params.segments],
            2,
        )


@dataclass(frozen=True)
class PoseError:
    position_mm: float
    orientation_deg: float


def load_accuracy_model(
    arm_config: str | Path = DEFAULT_ARM_CONFIG,
    tool_config: str | Path = DEFAULT_TOOL_CONFIG,
    scenario_config: str | Path = DEFAULT_SCENARIO_CONFIG,
) -> AccuracyModel:
    """Load all analysis parameters from the same configs as the simulator."""

    arm = load_spatial_arm_config(arm_config)
    scenario = load_scenario_config(scenario_config)
    if scenario.backend.mujoco_config_path is None:
        raise ValueError("Accuracy analysis requires a MuJoCo-backed scenario.")
    mujoco_config = load_mujoco_config(
        scenario.backend.mujoco_config_path,
        require_xml=False,
        require_visual_meshes=False,
    )
    hinge_lower, hinge_upper = mujoco_config.joints.hinge.range_rad
    lower_angles = []
    upper_angles = []
    for segment in arm.params.segments:
        # Positive kx is distributed across Y hinges. Positive ky maps to
        # negative X-hinge rotation according to the project's sign convention.
        y_count = segment.flexure_joint_axes.count("y")
        x_count = segment.flexure_joint_axes.count("x")
        if x_count <= 0 or y_count <= 0:
            raise ValueError("Each segment must contain both X and Y flexure axes.")
        lower_angles.extend((y_count * hinge_lower, -x_count * hinge_upper))
        upper_angles.extend((y_count * hinge_upper, -x_count * hinge_lower))
    tool = load_attachment_config(tool_config)
    tcp = compute_tool_tcp_pose(Pose6D.identity(), tool)
    return AccuracyModel(
        arm=arm,
        bending_model=BendingSpaceModel.from_arm(arm.params, arm.tendons),
        kinematics_mode=scenario.backend.kinematics_mode,
        encoder_angle_lower_deg=np.rad2deg(np.asarray(lower_angles, dtype=float)),
        encoder_angle_upper_deg=np.rad2deg(np.asarray(upper_angles, dtype=float)),
        tcp_pose_from_tip=tcp,
    )


def encoder_angles_to_segment_q(theta_deg: Sequence[float]) -> np.ndarray:
    """Convert encoder total angles to ``[hinge_x, hinge_y]`` per segment."""

    theta = _finite_vector(theta_deg, ENCODER_CHANNELS, "theta_deg")
    theta_rad = np.deg2rad(theta).reshape(3, 2)
    q_2dof = np.empty((3, 2), dtype=float)
    q_2dof[:, 0] = -theta_rad[:, 1]
    q_2dof[:, 1] = theta_rad[:, 0]
    return q_2dof.reshape(-1)


def encoder_pose(
    theta_deg: Sequence[float],
    model: AccuracyModel | None = None,
    *,
    target: str = "tcp",
) -> np.ndarray:
    """Return the bare-tip or tool-TCP pose for six encoder angles."""

    model = model or load_accuracy_model()
    tip_transform, _ = segment_2dof_forward_kinematics(
        encoder_angles_to_segment_q(theta_deg),
        model.params,
        samples_per_segment=2,
        kinematics_mode=model.kinematics_mode,
    )
    if target == "bare":
        return tip_transform
    if target != "tcp":
        raise ValueError("target must be 'bare' or 'tcp'.")
    return (Pose6D.from_matrix(tip_transform).compose(model.tcp_pose_from_tip)).as_matrix()


def batch_encoder_poses(
    theta_deg: np.ndarray,
    model: AccuracyModel,
    *,
    target: str = "tcp",
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of :func:`encoder_pose` for workspace studies."""

    theta = np.asarray(theta_deg, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != ENCODER_CHANNELS or not np.all(
        np.isfinite(theta)
    ):
        raise ValueError("theta_deg must have finite shape (N, 6).")
    if model.kinematics_mode != "discrete_hinge":
        transforms = np.asarray(
            [encoder_pose(values, model, target=target) for values in theta],
            dtype=float,
        )
        return transforms[:, :3, 3], transforms[:, :3, :3]
    count = theta.shape[0]
    rotation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    position = np.zeros((count, 3), dtype=float)
    theta_rad = np.deg2rad(theta).reshape(count, 3, 2)
    for segment_index, segment in enumerate(model.params.segments):
        axes = segment.flexure_joint_axes
        y_count = max(1, axes.count("y"))
        x_count = max(1, axes.count("x"))
        y_angle = theta_rad[:, segment_index, 0] / float(y_count)
        x_angle = -theta_rad[:, segment_index, 1] / float(x_count)
        cell_length = segment.effective_flexure_length / float(len(axes))
        for axis in axes:
            local_rotation = _batch_axis_rotation(axis, y_angle if axis == "y" else x_angle)
            rotation = rotation @ local_rotation
            position += cell_length * rotation[:, :, 2]
        position += segment.effective_distal_straight_length * rotation[:, :, 2]
    if target == "bare":
        return position, rotation
    if target != "tcp":
        raise ValueError("target must be 'bare' or 'tcp'.")
    position += np.einsum("nij,j->ni", rotation, model.tcp_pose_from_tip.position)
    rotation = rotation @ quaternion_wxyz_to_rotation_matrix(
        model.tcp_pose_from_tip.quat
    )
    return position, rotation


def encoder_angles_to_tendon_delta(
    theta_deg: Sequence[float], model: AccuracyModel
) -> np.ndarray:
    """Map the six total angles to the project's nine physical tendon deltas."""

    theta = _finite_vector(theta_deg, ENCODER_CHANNELS, "theta_deg")
    bending = np.deg2rad(theta) / model.flexure_lengths_m
    return model.bending_model.to_tendon(bending)


def is_pose_within_project_limits(
    theta_deg: Sequence[float], model: AccuracyModel, *, atol: float = 1.0e-12
) -> bool:
    """Check configured angle and all nine physical-tendon displacement limits."""

    theta = _finite_vector(theta_deg, ENCODER_CHANNELS, "theta_deg")
    if np.any(theta < model.encoder_angle_lower_deg - atol) or np.any(
        theta > model.encoder_angle_upper_deg + atol
    ):
        return False
    tendon = encoder_angles_to_tendon_delta(theta, model)
    limits = model.arm.limits
    return bool(
        np.all(tendon >= limits.tendon_displacement_min_m - atol)
        and np.all(tendon <= limits.tendon_displacement_max_m + atol)
    )


def orientation_error_deg(reference_rotation: np.ndarray, measured_rotation: np.ndarray) -> float:
    relative = np.asarray(reference_rotation).T @ np.asarray(measured_rotation)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def pose_error(reference: np.ndarray, measured: np.ndarray) -> PoseError:
    return PoseError(
        position_mm=1000.0
        * float(np.linalg.norm(measured[:3, 3] - reference[:3, 3])),
        orientation_deg=orientation_error_deg(reference[:3, :3], measured[:3, :3]),
    )


def worst_case_error(
    theta_true_deg: Sequence[float],
    angle_error_bound_deg: float | Sequence[float],
    model: AccuracyModel | None = None,
    *,
    target: str = "tcp",
) -> PoseError:
    """Enumerate all 64 error-bound corners at one true pose."""

    model = model or load_accuracy_model()
    theta_true = _finite_vector(theta_true_deg, ENCODER_CHANNELS, "theta_true_deg")
    error_bound = _error_bound_vector(angle_error_bound_deg)
    reference = encoder_pose(theta_true, model, target=target)
    maximum = PoseError(0.0, 0.0)
    for signed_error in SIGNS * error_bound:
        error = pose_error(
            reference,
            encoder_pose(theta_true + signed_error, model, target=target),
        )
        maximum = PoseError(
            max(maximum.position_mm, error.position_mm),
            max(maximum.orientation_deg, error.orientation_deg),
        )
    return maximum


def _error_bound_vector(value: float | Sequence[float]) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(ENCODER_CHANNELS, float(values))
    if values.shape != (ENCODER_CHANNELS,) or not np.all(np.isfinite(values)):
        raise ValueError("angle error bound must be a finite scalar or six-vector.")
    if np.any(values < 0.0):
        raise ValueError("angle error bounds must be non-negative.")
    return values


def _batch_axis_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result = np.zeros((angle.size, 3, 3), dtype=float)
    if axis == "x":
        result[:, 0, 0] = 1.0
        result[:, 1, 1] = cosine
        result[:, 1, 2] = -sine
        result[:, 2, 1] = sine
        result[:, 2, 2] = cosine
    elif axis == "y":
        result[:, 1, 1] = 1.0
        result[:, 0, 0] = cosine
        result[:, 0, 2] = sine
        result[:, 2, 0] = -sine
        result[:, 2, 2] = cosine
    else:
        raise ValueError(f"Unsupported flexure joint axis {axis!r}.")
    return result


def _finite_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return array


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-deg", nargs="+", type=float, default=DEFAULT_ACCURACY_DEG)
    parser.add_argument("--theta-deg", nargs=6, type=float, default=np.zeros(6))
    parser.add_argument("--arm-config", type=Path, default=DEFAULT_ARM_CONFIG)
    parser.add_argument("--tool-config", type=Path, default=DEFAULT_TOOL_CONFIG)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model = load_accuracy_model(args.arm_config, args.tool_config, args.scenario_config)
    theta = np.asarray(args.theta_deg, dtype=float)
    if not is_pose_within_project_limits(theta, model):
        raise ValueError("--theta-deg is outside configured angle or tendon limits.")
    print(f"kinematics={model.kinematics_mode} | TCP={1000*np.linalg.norm(model.tcp_pose_from_tip.position):.3f} mm")
    print(
        "encoder workspace [deg]: "
        f"{model.encoder_angle_lower_deg[0]:.3f} to {model.encoder_angle_upper_deg[0]:.3f}"
    )
    for accuracy in args.accuracy_deg:
        bare = worst_case_error(theta, accuracy, model, target="bare")
        tool = worst_case_error(theta, accuracy, model, target="tcp")
        print(
            f"accuracy=+/-{accuracy:.3f} deg | "
            f"bare={bare.position_mm:.3f} mm, {bare.orientation_deg:.3f} deg | "
            f"tcp={tool.position_mm:.3f} mm, {tool.orientation_deg:.3f} deg"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
