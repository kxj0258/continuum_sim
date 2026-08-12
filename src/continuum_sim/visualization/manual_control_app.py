"""Three-window manual control application for the dual MuJoCo system."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from threading import Event, RLock, Thread, current_thread
from time import perf_counter

from continuum_sim.application import SimulationApplication
from continuum_sim.application.hook_factory import observer_camera_attachment_config
from continuum_sim.runtime.concurrency import (
    AsyncLinePrinter,
    LatestValueSlot,
    TimeRateGate,
)
from continuum_sim.runtime.mujoco_state_copy import copy_mujoco_dynamic_state
from continuum_sim.runtime.observer_camera_hooks import MujocoObserverCameraFeedbackHook
from continuum_sim.runtime.viewer_hooks import _configure_mujoco_viewer
from continuum_sim.system.types import RobotSystemState
from continuum_sim.utils.runtime_timing import RuntimeTimingReporter
from continuum_sim.visualization.mujoco_system_debug_viewer import (
    MujocoSystemDebugViewer,
)


class _LatestCameraRenderWorker:
    """Render only the newest submitted state on a dedicated MuJoCo context."""

    def __init__(self, hook_factory) -> None:
        self._hook_factory = hook_factory
        self._states: LatestValueSlot[RobotSystemState] | None = None
        self._frames = LatestValueSlot(None)
        self._wake_event = Event()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._failure: BaseException | None = None

    def start(self, state: RobotSystemState) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._states = LatestValueSlot(state)
        self._failure = None
        self._stop_event.clear()
        self._wake_event.clear()
        self._thread = Thread(
            target=self._run,
            name="continuum-sim-camera-render",
            daemon=True,
        )
        self._thread.start()
        self.submit(state)

    def submit(self, state: RobotSystemState) -> None:
        if self._states is None or self._stop_event.is_set():
            return
        self._states.publish(state)
        self._wake_event.set()

    def consume_frame_after(self, version: int):
        return self._frames.consume_after(version)

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()
        self._thread = None

    def _run(self) -> None:
        hook = self._hook_factory()
        state = self._states.snapshot()[0]
        consumed_version = -1
        try:
            hook.on_reset(state)
            while not self._stop_event.is_set():
                self._wake_event.wait()
                self._wake_event.clear()
                if self._stop_event.is_set():
                    break
                item = self._states.consume_after(consumed_version)
                if item is None:
                    continue
                state, consumed_version = item
                self._frames.publish(hook.render_frame(state))
        except BaseException as exc:  # noqa: BLE001 - presentation failure is isolated.
            self._failure = exc
        finally:
            hook.on_finish(state)


class _MatplotlibCameraPresenter:
    """Own all Matplotlib camera artists on the GUI thread."""

    def __init__(self, title: str) -> None:
        self.title = str(title)
        self._plt = None
        self._figure = None
        self._axis = None
        self._image = None

    def present_frame(self, frame) -> None:
        if self._plt is None:
            import matplotlib.pyplot as plt

            self._plt = plt
            plt.ion()
            self._figure, self._axis = plt.subplots()
            self._figure.canvas.manager.set_window_title(self.title)
        if self._image is None:
            self._image = self._axis.imshow(frame)
            self._axis.set_axis_off()
        else:
            self._image.set_data(frame)
        self._figure.canvas.draw_idle()

    def close(self) -> None:
        if self._plt is not None and self._figure is not None:
            self._plt.close(self._figure)
        self._image = None


class ManualControlWindows:
    """Own the passive MuJoCo and observer-camera windows beside the controls."""

    def __init__(
        self,
        backend,
        *,
        control_dt_s: float,
        viewer_fps: float = 15.0,
        camera_fps: float = 10.0,
        simulation_lock=None,
        clock=perf_counter,
        runtime_timing=None,
    ) -> None:
        if viewer_fps <= 0.0:
            raise ValueError("viewer_fps must be positive.")
        if camera_fps <= 0.0:
            raise ValueError("camera_fps must be positive.")
        self.backend = backend
        self.control_dt_s = float(control_dt_s)
        self.viewer_fps = float(viewer_fps)
        self.camera_fps = float(camera_fps)
        self._simulation_lock = RLock() if simulation_lock is None else simulation_lock
        self._clock = clock
        self.runtime_timing = runtime_timing
        self._mujoco = None
        self._viewer = None
        self._viewer_data = None
        self._camera_worker = None
        self._camera_presenter = None
        self._camera_frame_version = 0
        self._active = False
        self._last_state: RobotSystemState | None = None
        self._viewer_gate = TimeRateGate(
            1.0 / self.viewer_fps,
            clock=self._clock,
        )
        self._camera_gate = TimeRateGate(
            1.0 / self.camera_fps,
            clock=self._clock,
        )

    def start(self, state: RobotSystemState) -> None:
        import mujoco
        import mujoco.viewer

        self._last_state = state
        self._mujoco = mujoco
        self._viewer_data = mujoco.MjData(self.backend.physics.model)
        self._viewer = mujoco.viewer.launch_passive(
            self.backend.physics.model,
            self._viewer_data,
        )
        _configure_mujoco_viewer(self._viewer, self.backend.config)
        self._sync_viewer()

        camera_attachment = observer_camera_attachment_config(self.backend.assembly)
        if camera_attachment is not None and camera_attachment.camera is not None:
            camera_name = camera_attachment.camera.name
            intrinsics = camera_attachment.camera.intrinsics
            self._camera_worker = _LatestCameraRenderWorker(
                lambda: MujocoObserverCameraFeedbackHook(
                    self.backend,
                    camera_name=camera_name,
                    intrinsics=intrinsics,
                    show_window=False,
                    stride=1,
                    video_output_paths=None,
                    data_lock=self._simulation_lock,
                    runtime_timing=self.runtime_timing,
                )
            )
            self._camera_presenter = _MatplotlibCameraPresenter(camera_name)
            self._camera_worker.start(state)
        self._active = True
        now_s = self._clock()
        self._viewer_gate.reset(now_s)
        self._camera_gate.reset(now_s)

    def update(self, state: RobotSystemState) -> None:
        self._last_state = state
        if not self._active:
            return
        timing = self.runtime_timing
        now_s = self._clock()
        if self._viewer_gate.due(now_s):
            with (
                nullcontext()
                if timing is None
                else timing.measure("viewer.sync")
            ):
                if self._viewer is not None and self._viewer.is_running():
                    self._sync_viewer()
        if self._camera_gate.due(now_s):
            with (
                nullcontext()
                if timing is None
                else timing.measure("camera.total")
            ):
                if self._camera_worker is not None:
                    self._camera_worker.submit(state)
        if self._camera_worker is not None and self._camera_presenter is not None:
            item = self._camera_worker.consume_frame_after(
                self._camera_frame_version
            )
            if item is not None:
                frame, self._camera_frame_version = item
                if frame is not None:
                    self._camera_presenter.present_frame(frame)

    def _sync_viewer(self) -> None:
        with self._simulation_lock:
            copy_mujoco_dynamic_state(
                self.backend.physics.data,
                self._viewer_data,
            )
        self._mujoco.mj_forward(self.backend.physics.model, self._viewer_data)
        self._viewer.sync()

    def close(self) -> None:
        self._active = False
        if self._camera_worker is not None:
            self._camera_worker.stop()
        self._camera_worker = None
        if self._camera_presenter is not None:
            self._camera_presenter.close()
        self._camera_presenter = None
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = None
        self._viewer_data = None


def run_manual_control(
    scenario_path: str | Path,
    *,
    panel_fps: float = 15.0,
    viewer_fps: float = 15.0,
    camera_fps: float = 10.0,
    curvature_step_1_per_m: float = 0.5,
) -> None:
    """Compose a scenario backend and run all three manual-control windows."""

    application = SimulationApplication.from_yaml(scenario_path)
    backend = application.loop.backend
    simulation_lock = RLock()
    async_printer = AsyncLinePrinter(lambda line: print(line, flush=True))
    runtime_timing = RuntimeTimingReporter(
        report_interval_s=0.5,
        printer=async_printer.write,
    )
    windows = ManualControlWindows(
        backend,
        control_dt_s=application.config.runtime.controller_dt_s,
        viewer_fps=viewer_fps,
        camera_fps=camera_fps,
        simulation_lock=simulation_lock,
        runtime_timing=runtime_timing,
    )
    viewer = MujocoSystemDebugViewer(
        backend,
        control_dt_s=application.config.runtime.controller_dt_s,
        n_substeps=application.config.runtime.n_substeps,
        state_update_callback=windows.update,
        curvature_step_1_per_m=curvature_step_1_per_m,
        panel_fps=panel_fps,
        simulation_lock=simulation_lock,
        runtime_timing=runtime_timing,
    )
    windows.start(viewer.state)
    runtime_timing.reset()
    async_printer.write(
        "[manual-timing] enabled; summary interval=0.5 s; "
        "stage values are avg/max milliseconds"
    )
    try:
        viewer.show()
    finally:
        viewer.close()
        windows.close()
        async_printer.close(drain=True)


__all__ = ["ManualControlWindows", "run_manual_control"]
