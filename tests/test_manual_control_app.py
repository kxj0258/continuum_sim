from __future__ import annotations

from types import SimpleNamespace

from continuum_sim.visualization.manual_control_app import ManualControlWindows


def test_manual_windows_schedule_viewer_and_camera_independently() -> None:
    clock = _Clock()
    windows = ManualControlWindows.__new__(ManualControlWindows)
    windows.viewer_fps = 15.0
    windows.camera_fps = 10.0
    windows._clock = clock
    windows._simulation_lock = _Lock()
    windows.runtime_timing = None
    windows._viewer = _Viewer()
    windows._camera_hook = _Camera()
    windows._active = True
    windows._last_state = None
    windows._next_viewer_update_s = 1.0 / windows.viewer_fps
    windows._next_camera_update_s = 1.0 / windows.camera_fps
    state = SimpleNamespace(time_s=0.0)

    for now_s in (0.00, 0.05, 0.07, 0.10, 0.14, 0.20):
        clock.now_s = now_s
        windows.update(state)

    assert windows._viewer.sync_calls == 3
    assert windows._camera_hook.update_calls == 2


class _Clock:
    now_s = 0.0

    def __call__(self) -> float:
        return self.now_s


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback


class _Viewer:
    def __init__(self) -> None:
        self.sync_calls = 0

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        self.sync_calls += 1


class _Camera:
    def __init__(self) -> None:
        self.update_calls = 0

    def enrich_state(self, state):
        self.update_calls += 1
        return state
