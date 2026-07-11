"""Weighted whole-body velocity solver for base-plus-spatial-arms systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.kinematics.whole_body import (
    SingularityConfig,
    SingularityReport,
    analyze_singularity,
)
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.system.control_layout import ControlLayout
from continuum_sim.system.types import RobotSystemCommand


@dataclass(frozen=True)
class WholeBodyTask:
    """One weighted linear velocity objective ``J v ~= target``."""

    name: str
    jacobian: np.ndarray
    target_velocity: np.ndarray
    weight: float

    def __post_init__(self) -> None:
        jacobian = np.asarray(self.jacobian, dtype=float)
        target = np.asarray(self.target_velocity, dtype=float)
        if jacobian.ndim != 2:
            raise ValueError("WholeBodyTask.jacobian must be 2D.")
        if target.shape != (jacobian.shape[0],):
            raise ValueError("WholeBodyTask target size must match Jacobian rows.")
        if self.weight <= 0.0:
            raise ValueError("WholeBodyTask.weight must be positive.")
        object.__setattr__(self, "jacobian", jacobian.copy())
        object.__setattr__(self, "target_velocity", target.copy())


@dataclass(frozen=True)
class WholeBodyControllerConfig:
    """Numerical and objective weighting settings."""

    executor_tracking_weight: float = 100.0
    observer_tracking_weight: float = 40.0
    executor_collision_avoidance_weight: float = 80.0
    observer_collision_avoidance_weight: float | None = None
    base_regularization_weight: float = 1.0
    tendon_regularization_weight: float = 0.2
    singularity: SingularityConfig = SingularityConfig()
    decouple_arm_singularity: bool = False
    singularity_strategy: str = "svd_projection"
    enforce_base_velocity_limits: bool = False
    enforce_tendon_rate_limits: bool = False


@dataclass(frozen=True)
class WholeBodySolveResult:
    """Named command and solver diagnostics."""

    command: RobotSystemCommand
    system_velocity: np.ndarray
    singularity: SingularityReport
    residual_norm: float
    arm_diagnostics: dict[str, dict[str, object]]
    arm_singularities: dict[str, SingularityReport]
    solver_diagnostics: dict[str, object] | None = None


@dataclass(frozen=True)
class SingularityProtection:
    """Per-variable damping/scaling plus global and per-arm reports."""

    damping: np.ndarray
    velocity_scale: np.ndarray
    global_report: SingularityReport
    arm_reports: dict[str, SingularityReport]


class WholeBodyController:
    """Solve base-plus-bending tasks and return compatible tendon rates."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        config: WholeBodyControllerConfig = WholeBodyControllerConfig(),
    ) -> None:
        self.assembly = assembly
        self.config = config
        self.layout = ControlLayout.from_assembly(assembly)

    def solve(self, tasks: list[WholeBodyTask] | tuple[WholeBodyTask, ...]) -> WholeBodySolveResult:
        if not tasks:
            command = RobotSystemCommand.zeros(
                {
                    arm.name: arm.spatial_arm.tendon_count
                    for arm in self.assembly.enabled_arms
                }
            )
            return WholeBodySolveResult(
                command=command,
                system_velocity=self.layout.flatten(command),
                singularity=analyze_singularity(
                    np.zeros((0, self.layout.size), dtype=float),
                    self.config.singularity,
                ),
                residual_norm=0.0,
                arm_diagnostics={
                    arm.name: self._arm_diagnostics(
                        arm.name,
                        np.zeros(
                            self.layout.arms[arm.name].stop
                            - self.layout.arms[arm.name].start,
                            dtype=float,
                        ),
                        1.0,
                    )
                    for arm in self.assembly.enabled_arms
                },
                arm_singularities={
                    name: analyze_singularity(
                        np.zeros(
                            (0, arm_slice.stop - arm_slice.start),
                            dtype=float,
                        ),
                        self.config.singularity,
                    )
                    for name, arm_slice in self.layout.arms.items()
                },
            )
        for task in tasks:
            if task.jacobian.shape[1] != self.layout.size:
                raise ValueError(
                    f"Task {task.name!r} Jacobian must have {self.layout.size} columns."
                )
        weighted_jacobian = np.vstack(
            [np.sqrt(task.weight) * task.jacobian for task in tasks]
        )
        weighted_target = np.concatenate(
            [np.sqrt(task.weight) * task.target_velocity for task in tasks]
        )
        target_projection_residual_norm = 0.0
        if self.config.singularity_strategy == "svd_projection":
            projected_target = _project_target_to_controllable_subspace(
                weighted_jacobian,
                weighted_target,
                self.config.singularity.minimum_singular_value,
            )
            target_projection_residual_norm = float(
                np.linalg.norm(weighted_target - projected_target)
            )
            weighted_target = projected_target
        elif self.config.singularity_strategy != "damping_scale":
            raise ValueError(
                "WholeBodyControllerConfig.singularity_strategy must be "
                "'damping_scale' or 'svd_projection'."
            )
        regularization = self._regularization_matrix()
        augmented_jacobian = np.vstack((weighted_jacobian, regularization))
        augmented_target = np.concatenate(
            (weighted_target, np.zeros(regularization.shape[0], dtype=float))
        )
        protection = self._singularity_protection(weighted_jacobian)
        singularity = protection.global_report
        lhs = (
            augmented_jacobian.T @ augmented_jacobian
            + np.diag(protection.damping**2)
        )
        rhs = augmented_jacobian.T @ augmented_target
        velocity = np.linalg.solve(lhs, rhs) * protection.velocity_scale
        velocity, arm_scales = self._apply_limits(velocity)
        command = self.layout.unflatten(velocity)
        residual = weighted_jacobian @ velocity - weighted_target
        return WholeBodySolveResult(
            command=command,
            system_velocity=velocity,
            singularity=singularity,
            residual_norm=float(np.linalg.norm(residual)),
            arm_diagnostics={
                name: self._arm_diagnostics(name, velocity[arm_slice], arm_scales[name])
                for name, arm_slice in self.layout.arms.items()
            },
            arm_singularities=protection.arm_reports,
            solver_diagnostics={
                "singularity_strategy": self.config.singularity_strategy,
                "target_projection_residual_norm": target_projection_residual_norm,
                "enforce_base_velocity_limits": self.config.enforce_base_velocity_limits,
                "enforce_tendon_rate_limits": self.config.enforce_tendon_rate_limits,
            },
        )

    def _singularity_protection(
        self,
        weighted_jacobian: np.ndarray,
    ) -> SingularityProtection:
        global_report = analyze_singularity(
            weighted_jacobian,
            self.config.singularity,
        )
        if self.config.singularity_strategy == "svd_projection":
            damping = np.zeros(self.layout.size, dtype=float)
            velocity_scale = np.ones(self.layout.size, dtype=float)
            arm_reports = {
                name: analyze_singularity(
                    weighted_jacobian[:, arm_slice],
                    self.config.singularity,
                )
                for name, arm_slice in self.layout.arms.items()
            }
            return SingularityProtection(
                damping=damping,
                velocity_scale=velocity_scale,
                global_report=global_report,
                arm_reports=arm_reports,
            )
        damping = np.full(
            self.layout.size,
            global_report.damping,
            dtype=float,
        )
        velocity_scale = np.full(
            self.layout.size,
            global_report.velocity_scale,
            dtype=float,
        )
        arm_reports = {
            name: analyze_singularity(
                weighted_jacobian[:, arm_slice],
                self.config.singularity,
            )
            for name, arm_slice in self.layout.arms.items()
        }
        if (
            self.config.decouple_arm_singularity
            and self.assembly.base.control_mode == "fixed"
        ):
            for name, arm_slice in self.layout.arms.items():
                report = arm_reports[name]
                damping[arm_slice] = report.damping
                velocity_scale[arm_slice] = report.velocity_scale
        return SingularityProtection(
            damping=damping,
            velocity_scale=velocity_scale,
            global_report=global_report,
            arm_reports=arm_reports,
        )

    def weight_for(self, objective: str) -> float:
        """Return configured weights for standard executor/observer objectives."""

        weights = {
            "executor_tracking": self.config.executor_tracking_weight,
            "observer_tracking": self.config.observer_tracking_weight,
            "executor_collision_avoidance": (
                self.config.executor_collision_avoidance_weight
            ),
            "observer_collision_avoidance": (
                self.config.executor_collision_avoidance_weight
                if self.config.observer_collision_avoidance_weight is None
                else self.config.observer_collision_avoidance_weight
            ),
        }
        try:
            return weights[objective]
        except KeyError as exc:
            raise KeyError(f"Unknown whole-body objective {objective!r}.") from exc

    def _regularization_matrix(self) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.layout.base_size:
            base = np.zeros((self.layout.base_size, self.layout.size), dtype=float)
            base[:, self.layout.base] = (
                np.sqrt(self.config.base_regularization_weight)
                * np.eye(self.layout.base_size, dtype=float)
            )
            blocks.append(base)
        for arm_name, arm_slice in self.layout.arms.items():
            model = self.layout.bending_models[arm_name]
            tendon_effort = np.zeros(
                (model.tendon_count, self.layout.size),
                dtype=float,
            )
            tendon_effort[:, arm_slice] = (
                np.sqrt(self.config.tendon_regularization_weight)
                * model.coupling_matrix
            )
            blocks.append(tendon_effort)
        return np.vstack(blocks)

    def _apply_limits(
        self,
        velocity: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        result = np.asarray(velocity, dtype=float).copy()
        arm_scales: dict[str, float] = {}
        if self.assembly.base.control_mode == "fixed":
            result[self.layout.base] = 0.0
        elif self.config.enforce_base_velocity_limits:
            result[self.layout.base.start : self.layout.base.start + 3] = np.clip(
                result[self.layout.base.start : self.layout.base.start + 3],
                -self.assembly.base.max_linear_speed_mps,
                self.assembly.base.max_linear_speed_mps,
            )
            result[self.layout.base.start + 3 : self.layout.base.stop] = np.clip(
                result[self.layout.base.start + 3 : self.layout.base.stop],
                -self.assembly.base.max_angular_speed_rad_s,
                self.assembly.base.max_angular_speed_rad_s,
            )
        for arm in self.assembly.enabled_arms:
            arm_slice = self.layout.arms[arm.name]
            model = self.layout.bending_models[arm.name]
            tendon_rate = model.to_tendon(result[arm_slice])
            scale = 1.0
            if self.config.enforce_tendon_rate_limits:
                rate_limit = arm.spatial_arm.limits.max_tendon_rate_mps
                ratios = np.divide(
                    rate_limit,
                    np.abs(tendon_rate),
                    out=np.full_like(rate_limit, np.inf),
                    where=np.abs(tendon_rate) > 0.0,
                )
                scale = float(min(1.0, np.min(ratios)))
                result[arm_slice] *= scale
            arm_scales[arm.name] = scale
        return result, arm_scales

    def _arm_diagnostics(
        self,
        arm_name: str,
        bending_rate: np.ndarray,
        scale: float,
    ) -> dict[str, object]:
        model = self.layout.bending_models[arm_name]
        tendon_rate = model.to_tendon(bending_rate)
        return {
            "bending_rate_rad_per_m_s": np.asarray(bending_rate, dtype=float).copy(),
            "tendon_rate_mps": tendon_rate,
            "compatibility_residual_mps": model.residual(tendon_rate),
            "compatibility_residual_norm_mps": model.residual_norm(tendon_rate),
            "limit_scale": float(scale),
            "bending_mapping_rank": model.rank,
            "bending_mapping_condition_number": model.condition_number,
        }


def _project_target_to_controllable_subspace(
    jacobian: np.ndarray,
    target: np.ndarray,
    minimum_singular_value: float,
) -> np.ndarray:
    """Drop task-space target components along weak SVD directions."""

    if jacobian.size == 0:
        return target.copy()
    u, singular_values, _ = np.linalg.svd(jacobian, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros_like(target)
    keep = singular_values >= minimum_singular_value
    if not np.any(keep):
        return np.zeros_like(target)
    controllable = u[:, keep]
    return controllable @ (controllable.T @ target)
