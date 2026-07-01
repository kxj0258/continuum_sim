"""Backend-independent closed-loop orchestration for composable robot systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from continuum_sim.backends.base_types import SystemBackendProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


class SystemControllerProtocol(Protocol):
    """Controller boundary consumed by the generic simulation loop."""

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        """Return one world-base and named tendon-rate command."""


class SimulationHookProtocol(Protocol):
    """Optional observer for recording, viewer sync, or task progression."""

    def on_reset(self, state: RobotSystemState) -> None:
        """Observe the reset state."""

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        """Observe one completed control step."""

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        """Return whether the simulation loop should stop."""

    def on_finish(self, state: RobotSystemState) -> None:
        """Release optional resources after the loop."""


@dataclass(frozen=True)
class SimulationLoopConfig:
    """Timing and bounded-run configuration."""

    controller_dt_s: float
    n_substeps: int
    max_steps: int

    def __post_init__(self) -> None:
        if self.controller_dt_s <= 0.0:
            raise ValueError("controller_dt_s must be positive.")
        if self.n_substeps <= 0:
            raise ValueError("n_substeps must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")


@dataclass(frozen=True)
class SimulationLoopResult:
    """Named state/command history returned by the generic loop."""

    states: tuple[RobotSystemState, ...]
    commands: tuple[RobotSystemCommand, ...]
    stopped_early: bool
    metadata: dict[str, object] = field(default_factory=dict)


class SimulationLoop:
    """Run controller/backend composition without importing MuJoCo."""

    def __init__(
        self,
        backend: SystemBackendProtocol,
        controller: SystemControllerProtocol,
        config: SimulationLoopConfig,
        hooks: tuple[SimulationHookProtocol, ...] = (),
    ) -> None:
        self.backend = backend
        self.controller = controller
        self.config = config
        self.hooks = hooks

    def run(self) -> SimulationLoopResult:
        initial = self.backend.reset_system()
        states = [initial]
        commands: list[RobotSystemCommand] = []
        for hook in self.hooks:
            hook.on_reset(initial)
        stopped_early = False
        for step_index in range(self.config.max_steps):
            current = states[-1]
            if any(hook.should_stop(current, step_index) for hook in self.hooks):
                stopped_early = True
                break
            command = self.controller.compute_command(current)
            next_state = self.backend.step_system(
                command,
                dt=self.config.controller_dt_s,
                n_substeps=self.config.n_substeps,
            )
            commands.append(command)
            states.append(next_state)
            for hook in self.hooks:
                hook.on_step(next_state, command, step_index)
        for hook in self.hooks:
            hook.on_finish(states[-1])
        return SimulationLoopResult(
            states=tuple(states),
            commands=tuple(commands),
            stopped_early=stopped_early,
        )
