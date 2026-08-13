from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import continuum_sim.visualization.manual_control_app as manual_control_app
from continuum_sim.visualization.manual_control_app import ManualControlWindows
from continuum_sim.runtime.concurrency import TimeRateGate


def test_manual_windows_schedule_only_passive_viewer() -> None:
    clock = _Clock()
    windows = ManualControlWindows.__new__(ManualControlWindows)
    windows.viewer_fps = 15.0
    windows._clock = clock
    windows._simulation_lock = _Lock()
    windows.runtime_timing = None
    windows._mujoco = _Mujoco()
    windows.backend = SimpleNamespace(
        physics=SimpleNamespace(model=object(), data=_Data(2.0)),
    )
    windows._viewer_data = _Data(0.0)
    windows._viewer = _Viewer()
    windows._active = True
    windows._last_state = None
    windows._viewer_gate = TimeRateGate(
        1.0 / windows.viewer_fps,
        clock=clock,
        start_s=0.0,
    )
    state = SimpleNamespace(time_s=0.0)

    for now_s in (0.00, 0.05, 0.07, 0.10, 0.14, 0.20):
        clock.now_s = now_s
        windows.update(state)

    assert windows._viewer.sync_calls == 3
    assert not hasattr(windows, "_camera_worker")


def test_viewer_snapshot_releases_live_data_lock_before_forward_and_sync() -> None:
    events: list[str] = []
    windows = ManualControlWindows.__new__(ManualControlWindows)
    windows.backend = SimpleNamespace(
        physics=SimpleNamespace(model=object(), data=_Data(4.0))
    )
    windows._viewer_data = _Data(0.0)
    windows._simulation_lock = _Lock(events)
    windows._mujoco = _Mujoco(events)
    windows._viewer = _Viewer(events)

    windows._sync_viewer()

    assert windows._viewer_data.time == 4.0
    assert np.array_equal(windows._viewer_data.qpos, [4.0, 5.0])
    assert events == ["lock.enter", "lock.exit", "forward", "viewer.sync"]


def test_curvature_entry_fixes_compatible_control_mode(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        manual_control_app,
        "_run_manual_control",
        lambda scenario_path, **options: calls.append((scenario_path, options)),
    )

    manual_control_app.run_manual_curvature_control("scenario.yaml")

    assert calls[0][1]["control_mode"] == "curvature"
    assert calls[0][1]["show_tendon_monitor"] is False


def test_tendon_entry_fixes_raw_control_mode(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        manual_control_app,
        "_run_manual_control",
        lambda scenario_path, **options: calls.append((scenario_path, options)),
    )

    manual_control_app.run_manual_tendon_control("scenario.yaml")

    assert calls[0][1]["control_mode"] == "tendon"
    assert calls[0][1]["show_tendon_monitor"] is False


class _Clock:
    now_s = 0.0

    def __call__(self) -> float:
        return self.now_s


class _Lock:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events

    def __enter__(self):
        if self.events is not None:
            self.events.append("lock.enter")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self.events is not None:
            self.events.append("lock.exit")


class _Viewer:
    def __init__(self, events: list[str] | None = None) -> None:
        self.sync_calls = 0
        self.events = events

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        self.sync_calls += 1
        if self.events is not None:
            self.events.append("viewer.sync")


class _Data:
    def __init__(self, value: float) -> None:
        self.time = value
        self.qpos = np.array([value, value + 1.0])
        self.qvel = np.array([value + 2.0, value + 3.0])
        self.act = np.zeros(0)
        self.ctrl = np.zeros(0)
        self.mocap_pos = np.zeros((0, 3))
        self.mocap_quat = np.zeros((0, 4))
        self.userdata = np.zeros(0)


class _Mujoco:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events

    def mj_forward(self, model, data) -> None:
        del model, data
        if self.events is not None:
            self.events.append("forward")
