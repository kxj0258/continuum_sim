"""Small YAML validation helpers shared by config loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_path(config_path: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    parent_candidate = (config_path.parent / path).resolve()
    if parent_candidate.exists():
        return parent_candidate
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    portable_candidate = _resolve_portable_project_path(path)
    if portable_candidate is not None and portable_candidate.exists():
        return portable_candidate
    return parent_candidate


def _resolve_portable_project_path(path: Path) -> Path | None:
    parts = path.parts
    while parts and parts[0] == "..":
        parts = parts[1:]
    if not parts or parts[0] not in {"assets", "configs"}:
        return None
    portable_path = Path(*parts)
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        candidate = (root / portable_path).resolve()
        if candidate.exists():
            return candidate
    return None


def section(values: dict, name: str) -> dict:
    result = values.get(name)
    if not isinstance(result, dict):
        raise ValueError(f"Expected section {name!r} to be a mapping.")
    return result


def optional_section(values: dict, name: str) -> dict:
    result = values.get(name, {})
    if not isinstance(result, dict):
        raise ValueError(f"Expected section {name!r} to be a mapping.")
    return result


def required(values: dict, name: str) -> object:
    if name not in values:
        raise ValueError(f"Missing required config field {name!r}.")
    return values[name]


def positive_float(values: dict, name: str) -> float:
    return positive_float_value(required(values, name), name)


def positive_float_value(raw_value: object, name: str) -> float:
    value = float(raw_value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def nonnegative_float(values: dict, name: str) -> float:
    return nonnegative_float_value(required(values, name), name)


def nonnegative_float_value(raw_value: object, name: str) -> float:
    value = float(raw_value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def positive_int(values: dict, name: str) -> int:
    return positive_int_value(required(values, name), name)


def positive_int_value(raw_value: object, name: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def nonnegative_int_value(raw_value: object, name: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def choice(values: dict, name: str, choices: tuple[str, ...]) -> str:
    return choice_value(required(values, name), name, choices)


def choice_value(raw_value: object, name: str, choices: tuple[str, ...]) -> str:
    value = str(raw_value)
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}.")
    return value


def bool_value(raw_value: object, name: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    raise ValueError(f"{name} must be a boolean, got {raw_value!r}.")


def bool_field(values: dict, name: str, *, default: bool | None = None) -> bool:
    if default is None:
        return bool_value(required(values, name), name)
    return bool_value(values.get(name, default), name)


def float_value(raw_value: object, name: str) -> float:
    del name
    return float(raw_value)


def float_tuple(raw_value: object, name: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(raw_value, list | tuple):
        raise ValueError(f"{name} must be a list of {length} numbers.")
    result = tuple(float(value) for value in raw_value)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers.")
    return result


def range_tuple(raw_value: object, name: str) -> tuple[float, float]:
    result = float_tuple(raw_value, name, length=2)
    if result[0] >= result[1]:
        raise ValueError(f"{name} lower bound must be < upper bound, got {result}.")
    return result  # type: ignore[return-value]


def rgba_tuple(raw_value: object, name: str) -> tuple[float, float, float, float]:
    if not isinstance(raw_value, list | tuple):
        raise ValueError(f"{name} must be a list of four floats.")
    result = tuple(float(value) for value in raw_value)
    if len(result) != 4:
        raise ValueError(f"{name} must contain exactly four values.")
    if any(value < 0.0 or value > 1.0 for value in result):
        raise ValueError(f"{name} values must be in [0, 1], got {result}.")
    return result  # type: ignore[return-value]


def position_vector(raw_value: object, name: str) -> np.ndarray:
    array = np.asarray(raw_value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected {name} with shape (3,), got {array.shape}.")
    return array


def motor_vector(values: object, name: str, *, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"Expected {name} with shape ({expected_size},), got {array.shape}.")
    return array


def string_tuple(
    raw_value: object,
    name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(raw_value, list | tuple):
        raise ValueError(f"{name} must be a list of strings.")
    result = tuple(str(value) for value in raw_value)
    if not allow_empty and (not result or any(not value for value in result)):
        raise ValueError(f"{name} must contain at least one non-empty filename.")
    return result


def geom_group(raw_value: object, name: str) -> int:
    value = int(raw_value)
    if value < 0 or value > 5:
        raise ValueError(f"{name} must be a MuJoCo geom group in [0, 5], got {value}.")
    return value
