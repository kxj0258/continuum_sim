"""Dual-arm robot YAML helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from continuum_sim.config import load_yaml
from continuum_sim.model.physical_tendon import PhysicalTendonPath
from continuum_sim.model.robot_params import SegmentParams, ThreeSegmentRobotParams


@dataclass(frozen=True)
class DualArmMotorParams:
    """Motor-to-tendon transmission parameters for one dual-arm actuator."""

    id: str
    motor_index: int
    tendon_global_index: int
    spool_radius: float
    gear_ratio: float
    direction_sign: float
    zero_position: float = 0.0


@dataclass(frozen=True)
class DualArmRobotConfig:
    """Parsed dual-arm robot description with spatial tendons."""

    path: Path
    default_arm: str
    arm_names: tuple[str, ...]
    params_by_arm: dict[str, ThreeSegmentRobotParams]
    tendons_by_arm: dict[str, tuple[PhysicalTendonPath, ...]]
    motors_by_arm: dict[str, tuple[DualArmMotorParams, ...]]
    total_tendon_count: int
    total_motor_count: int
    tendons_per_arm: int

    @property
    def physical_tendons(self) -> tuple[PhysicalTendonPath, ...]:
        tendons: list[PhysicalTendonPath] = []
        for arm_name in self.arm_names:
            tendons.extend(self.tendons_by_arm[arm_name])
        return tuple(tendons)

    @property
    def motor_params(self) -> tuple[DualArmMotorParams, ...]:
        motors: list[DualArmMotorParams] = []
        for arm_name in self.arm_names:
            motors.extend(self.motors_by_arm[arm_name])
        return tuple(motors)

    @property
    def default_arm_params(self) -> ThreeSegmentRobotParams:
        return self.params_by_arm[self.default_arm]

    @property
    def default_arm_tendons(self) -> tuple[PhysicalTendonPath, ...]:
        return self.tendons_by_arm[self.default_arm]

    @property
    def default_arm_motors(self) -> tuple[DualArmMotorParams, ...]:
        return self.motors_by_arm[self.default_arm]


def is_dual_arm_robot_config(path: str | Path) -> bool:
    raw = load_yaml(path)
    return "dual_robot" in raw


def load_dual_arm_robot_config(path: str | Path) -> DualArmRobotConfig:
    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    dual_raw = raw.get("dual_robot")
    if not isinstance(dual_raw, dict):
        raise ValueError("dual_robot must be present for a dual-arm robot config.")
    arms_raw = dual_raw.get("arms")
    if not isinstance(arms_raw, dict) or not arms_raw:
        raise ValueError("dual_robot.arms must be a non-empty mapping.")
    default_arm = str(dual_raw.get("default_arm", next(iter(arms_raw))))
    if default_arm not in arms_raw:
        raise ValueError(f"dual_robot.default_arm {default_arm!r} is not in dual_robot.arms.")

    arm_names = tuple(str(name) for name in arms_raw.keys())
    total_tendon_count = int(dual_raw.get("total_tendon_count", 0))
    total_motor_count = int(dual_raw.get("total_motor_count", total_tendon_count))
    tendons_per_arm = int(dual_raw.get("tendons_per_arm", 0))
    if total_tendon_count <= 0:
        raise ValueError("dual_robot.total_tendon_count must be positive.")
    if total_motor_count <= 0:
        raise ValueError("dual_robot.total_motor_count must be positive.")
    if tendons_per_arm <= 0:
        raise ValueError("dual_robot.tendons_per_arm must be positive.")

    params_by_arm: dict[str, ThreeSegmentRobotParams] = {}
    tendons_by_arm: dict[str, tuple[PhysicalTendonPath, ...]] = {}
    motors_by_arm: dict[str, tuple[DualArmMotorParams, ...]] = {}
    for arm_name in arm_names:
        arm_raw = arms_raw[arm_name]
        if not isinstance(arm_raw, dict):
            raise ValueError(f"dual_robot.arms.{arm_name} must be a mapping.")
        params_by_arm[arm_name] = _load_arm_params(arm_name, arm_raw)
        tendons_by_arm[arm_name] = _load_arm_tendons(
            arm_name,
            arm_raw,
            tendons_per_arm=tendons_per_arm,
        )
        motors_by_arm[arm_name] = _load_arm_motors(
            arm_name,
            arm_raw,
            tendons_per_arm=tendons_per_arm,
        )

    config = DualArmRobotConfig(
        path=config_path,
        default_arm=default_arm,
        arm_names=arm_names,
        params_by_arm=params_by_arm,
        tendons_by_arm=tendons_by_arm,
        motors_by_arm=motors_by_arm,
        total_tendon_count=total_tendon_count,
        total_motor_count=total_motor_count,
        tendons_per_arm=tendons_per_arm,
    )
    _validate_dual_indices(config, dual_raw)
    return config


def _load_arm_params(arm_name: str, arm_raw: dict[str, object]) -> ThreeSegmentRobotParams:
    segments_raw = arm_raw.get("segments")
    if not isinstance(segments_raw, list) or len(segments_raw) != 3:
        raise ValueError(f"dual_robot.arms.{arm_name}.segments must contain exactly 3 items.")
    segments = []
    for segment_raw in segments_raw:
        if not isinstance(segment_raw, dict):
            raise ValueError(f"dual_robot.arms.{arm_name}.segments items must be mappings.")
        segments.append(
            SegmentParams(
                length=float(segment_raw["length"]),
                tendon_radius=float(segment_raw["tendon_radius"]),
                tendon_angles_deg=tuple(float(v) for v in segment_raw["tendon_angles_deg"]),
            )
        )
    return ThreeSegmentRobotParams(segments=tuple(segments))  # type: ignore[arg-type]


def _load_arm_tendons(
    arm_name: str,
    arm_raw: dict[str, object],
    *,
    tendons_per_arm: int,
) -> tuple[PhysicalTendonPath, ...]:
    tendon_items = arm_raw.get("physical_tendons")
    if not isinstance(tendon_items, list) or len(tendon_items) != tendons_per_arm:
        raise ValueError(
            f"dual_robot.arms.{arm_name}.physical_tendons must contain "
            f"{tendons_per_arm} items."
        )
    tendons = tuple(_physical_tendon_from_dict(item) for item in tendon_items)
    global_indices = [tendon.global_index for tendon in tendons]
    if len(set(global_indices)) != len(global_indices):
        raise ValueError(f"dual_robot.arms.{arm_name}.physical_tendons has duplicate global_index values.")
    motor_indices = [tendon.motor_index for tendon in tendons]
    if len(set(motor_indices)) != len(motor_indices):
        raise ValueError(f"dual_robot.arms.{arm_name}.physical_tendons has duplicate motor_index values.")
    return tuple(sorted(tendons, key=lambda tendon: tendon.global_index))


def _load_arm_motors(
    arm_name: str,
    arm_raw: dict[str, object],
    *,
    tendons_per_arm: int,
) -> tuple[DualArmMotorParams, ...]:
    motors_raw = arm_raw.get("motors")
    if not isinstance(motors_raw, dict):
        raise ValueError(f"dual_robot.arms.{arm_name}.motors must be a mapping.")
    motor_items = motors_raw.get("items")
    if not isinstance(motor_items, list) or len(motor_items) != tendons_per_arm:
        raise ValueError(
            f"dual_robot.arms.{arm_name}.motors.items must contain "
            f"{tendons_per_arm} items."
        )
    motors = tuple(_motor_params_from_dict(item) for item in motor_items)
    motor_indices = [motor.motor_index for motor in motors]
    if len(set(motor_indices)) != len(motor_indices):
        raise ValueError(f"dual_robot.arms.{arm_name}.motors.items has duplicate motor_index values.")
    return tuple(sorted(motors, key=lambda motor: motor.motor_index))


def _validate_dual_indices(
    config: DualArmRobotConfig,
    dual_raw: dict[str, object],
) -> None:
    total_tendon_count = config.total_tendon_count
    total_motor_count = config.total_motor_count
    if total_tendon_count != len(config.arm_names) * config.tendons_per_arm:
        raise ValueError(
            "dual_robot.total_tendon_count must equal arm_count * tendons_per_arm, got "
            f"{total_tendon_count} and {len(config.arm_names)} * {config.tendons_per_arm}."
        )
    tendon_indices = sorted(tendon.global_index for tendon in config.physical_tendons)
    expected_tendons = list(range(total_tendon_count))
    if tendon_indices != expected_tendons:
        raise ValueError(
            "dual_robot physical tendon global_index values must cover "
            f"0..{total_tendon_count - 1}, got {tendon_indices}."
        )
    tendon_motor_indices = sorted(tendon.motor_index for tendon in config.physical_tendons)
    if tendon_motor_indices != expected_tendons:
        raise ValueError(
            "dual_robot physical tendon motor_index values must cover "
            f"0..{total_tendon_count - 1}, got {tendon_motor_indices}."
        )
    motor_indices = sorted(motor.motor_index for motor in config.motor_params)
    expected_motors = list(range(total_motor_count))
    if motor_indices != expected_motors:
        raise ValueError(
            "dual_robot motor_index values must cover "
            f"0..{total_motor_count - 1}, got {motor_indices}."
        )
    motor_tendon_indices = sorted(motor.tendon_global_index for motor in config.motor_params)
    if motor_tendon_indices != expected_tendons:
        raise ValueError(
            "dual_robot motor tendon_global_index values must cover "
            f"0..{total_tendon_count - 1}, got {motor_tendon_indices}."
        )


def _physical_tendon_from_dict(item: object) -> PhysicalTendonPath:
    if not isinstance(item, dict):
        raise ValueError("physical_tendons items must be mappings.")
    return PhysicalTendonPath(
        id=str(item["id"]),
        global_index=int(item["global_index"]),
        motor_index=int(item["motor_index"]),
        anchor_segment_index=int(item["anchor_segment_index"]),
        angle_deg=float(item["angle_deg"]),
        radial_offset=float(item["radial_offset"]),
        path_segment_indices=tuple(int(v) for v in item["path_segment_indices"]),  # type: ignore[index]
        hole_index=(None if "hole_index" not in item else int(item["hole_index"])),
    )


def _motor_params_from_dict(item: object) -> DualArmMotorParams:
    if not isinstance(item, dict):
        raise ValueError("motors.items entries must be mappings.")
    return DualArmMotorParams(
        id=str(item["id"]),
        motor_index=int(item["motor_index"]),
        tendon_global_index=int(item["tendon_global_index"]),
        spool_radius=float(item["spool_radius"]),
        gear_ratio=float(item["gear_ratio"]),
        direction_sign=float(item["direction_sign"]),
        zero_position=float(item.get("zero_position", 0.0)),
    )
