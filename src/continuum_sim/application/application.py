"""Scenario composition root and primary simulation application API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread

from continuum_sim.application.scenario import (
    ScenarioConfig,
    load_scenario_config,
)
from continuum_sim.application.backend_factory import build_mujoco_backend
from continuum_sim.application.controller_factory import build_controller
from continuum_sim.application.hook_factory import build_runtime_hooks
from continuum_sim.application.task_plan_factory import resolve_task_plan
from continuum_sim.model.robot_assembly import load_robot_assembly_config
from continuum_sim.io.scenario_artifacts import save_scenario_artifacts
from continuum_sim.runtime.simulation_loop import (
    SimulationLoop,
    SimulationLoopConfig,
    SimulationLoopResult,
)
from continuum_sim.scenes.engine_query import EnginePrimitiveSceneQuery
from continuum_sim.scenes.engine_scene import load_engine_scene_config
from continuum_sim.scenes.scene_config import load_navigation_scene_config
from continuum_sim.scenes.structured_query import StructuredSceneQuery


@dataclass
class SimulationApplication:
    """Fully composed backend, controller, and hook lifecycle."""

    config: ScenarioConfig
    loop: SimulationLoop
    hooks_by_name: dict[str, object]
    last_artifacts: object | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationApplication":
        return cls.from_config(load_scenario_config(path))

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> "SimulationApplication":
        assembly = load_robot_assembly_config(config.assembly_config_path)
        engine_scene = (
            None
            if config.scene.engine_config_path is None
            else load_engine_scene_config(config.scene.engine_config_path)
        )
        structured_scene = (
            None
            if config.scene.structured_config_path is None
            else load_navigation_scene_config(config.scene.structured_config_path)
        )
        if engine_scene is not None and structured_scene is not None:
            raise ValueError("A scenario cannot select engine and structured scenes together.")
        backend = build_mujoco_backend(
            config,
            assembly,
            engine_scene,
            structured_scene,
        )
        if engine_scene is not None:
            scene_query = EnginePrimitiveSceneQuery(engine_scene)
        elif structured_scene is not None:
            scene_query = StructuredSceneQuery(structured_scene)
        else:
            scene_query = None
        task_plan = resolve_task_plan(config, assembly, engine_scene, structured_scene)
        controller_build = build_controller(
            config=config,
            assembly=assembly,
            engine_scene=engine_scene,
            scene_query=scene_query,
            task_plan=task_plan,
        )
        controller = controller_build.controller
        observer_camera_target_world = controller_build.observer_camera_target_world
        hooks_by_name = build_runtime_hooks(
            config=config,
            backend=backend,
            controller=controller,
            assembly=assembly,
            observer_camera_target_world=observer_camera_target_world,
        )
        loop = SimulationLoop(
            backend,
            controller,
            SimulationLoopConfig(
                controller_dt_s=config.runtime.controller_dt_s,
                n_substeps=config.runtime.n_substeps,
                max_steps=config.runtime.max_steps,
            ),
            hooks=tuple(hooks_by_name.values()),
        )
        return cls(config=config, loop=loop, hooks_by_name=hooks_by_name)

    def run(self) -> SimulationLoopResult:
        gui_hooks = tuple(
            hook
            for hook in self.hooks_by_name.values()
            if bool(getattr(hook, "requires_gui_main_thread", False))
        )
        result = (
            self._run_with_gui_presentations(gui_hooks)
            if gui_hooks
            else self.loop.run()
        )
        self.last_artifacts = save_scenario_artifacts(self, result)
        return result

    def _run_with_gui_presentations(self, gui_hooks):
        completed = Event()
        result_holder = []
        failure_holder = []

        def run_loop() -> None:
            try:
                result_holder.append(self.loop.run())
            except BaseException as exc:  # noqa: BLE001 - re-raised on main thread.
                failure_holder.append(exc)
            finally:
                completed.set()

        worker = Thread(
            target=run_loop,
            name="continuum-sim-scenario",
            daemon=False,
        )
        worker.start()
        try:
            while not completed.wait(0.01):
                for hook in gui_hooks:
                    hook.present_pending()
                _pump_matplotlib_events()
            worker.join()
            for hook in gui_hooks:
                hook.present_pending(force=True)
            if failure_holder:
                raise failure_holder[0]
            return result_holder[0]
        finally:
            for hook in reversed(gui_hooks):
                hook.close_presentation()


def _pump_matplotlib_events() -> None:
    try:
        import matplotlib.pyplot as plt

        if plt.get_fignums():
            plt.pause(0.001)
    except Exception:
        pass


