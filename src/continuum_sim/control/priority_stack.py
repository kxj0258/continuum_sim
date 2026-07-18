"""Declarative whole-body task priority stack configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


KNOWN_PRIORITY_TASKS = frozenset(
    {
        "position_servo",
        "orientation_servo",
        "normal_force_control",
        "scene_avoidance",
        "interarm_avoidance",
        "observer_tracking",
        "look_at",
    }
)


@dataclass(frozen=True)
class PriorityLevelConfig:
    """One nullspace priority level made of one or more named task groups."""

    tasks: tuple[str, ...]

    def __post_init__(self) -> None:
        tasks = tuple(str(task) for task in self.tasks)
        if not tasks:
            raise ValueError("PriorityLevelConfig.tasks must be non-empty.")
        unknown = [task for task in tasks if task not in KNOWN_PRIORITY_TASKS]
        if unknown:
            raise ValueError(f"Unknown priority task group(s): {unknown}.")
        object.__setattr__(self, "tasks", tasks)


@dataclass(frozen=True)
class PriorityStackConfig:
    """Executor and observer priority stacks consumed by the intent resolver."""

    executor: tuple[PriorityLevelConfig, ...] = field(
        default_factory=lambda: (
            PriorityLevelConfig(("position_servo",)),
            PriorityLevelConfig(("normal_force_control",)),
            PriorityLevelConfig(("orientation_servo",)),
            PriorityLevelConfig(("scene_avoidance",)),
        )
    )
    observer: tuple[PriorityLevelConfig, ...] = field(
        default_factory=lambda: (
            PriorityLevelConfig(("interarm_avoidance",)),
            PriorityLevelConfig(("observer_tracking",)),
            PriorityLevelConfig(("look_at",)),
            PriorityLevelConfig(("scene_avoidance",)),
        )
    )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "PriorityStackConfig":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("priority_stack must be a mapping.")
        default = cls()
        return cls(
            executor=_levels_from_mapping(values.get("executor"), default.executor),
            observer=_levels_from_mapping(values.get("observer"), default.observer),
        )

    def as_metadata(self) -> dict[str, tuple[tuple[str, ...], ...]]:
        return {
            "executor": tuple(level.tasks for level in self.executor),
            "observer": tuple(level.tasks for level in self.observer),
        }


def _levels_from_mapping(
    value: object,
    default: tuple[PriorityLevelConfig, ...],
) -> tuple[PriorityLevelConfig, ...]:
    if value is None:
        return default
    if not isinstance(value, list | tuple):
        raise ValueError("priority stack arm entries must be a list.")
    levels: list[PriorityLevelConfig] = []
    for item in value:
        if isinstance(item, str):
            levels.append(PriorityLevelConfig((item,)))
            continue
        if not isinstance(item, Mapping):
            raise ValueError("priority stack levels must be strings or mappings.")
        tasks = item.get("tasks")
        if isinstance(tasks, str):
            levels.append(PriorityLevelConfig((tasks,)))
        elif isinstance(tasks, list | tuple):
            levels.append(PriorityLevelConfig(tuple(str(task) for task in tasks)))
        else:
            raise ValueError("priority stack level.tasks must be a string or list.")
    if not levels:
        raise ValueError("priority stack arm entries must not be empty.")
    return tuple(levels)
