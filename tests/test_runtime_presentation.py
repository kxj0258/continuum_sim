from __future__ import annotations

from threading import get_ident
from types import SimpleNamespace

import numpy as np

from continuum_sim.application.application import SimulationApplication
from continuum_sim.runtime.viewer_hooks import MujocoViewerHook, RealtimePacerHook


def test_realtime_pacer_is_independent_from_viewer_refresh() -> None:
    sleeps: list[tuple[float, float, float, float]] = []
    pacer = RealtimePacerHook(
        realtime_factor=2.0,
        clock=lambda: 10.0,
        sleeper=lambda start_wall, start_sim, current_sim, factor: sleeps.append(
            (start_wall, start_sim, current_sim, factor)
        ),
    )
    state = SimpleNamespace(time_s=1.0)
    pacer.on_reset(state)

    pacer.on_step(SimpleNamespace(time_s=1.02), object(), 0)
    pacer.on_step(SimpleNamespace(time_s=1.04), object(), 1)

    assert sleeps == [
        (10.0, 1.0, 1.02, 2.0),
        (10.0, 1.0, 1.04, 2.0),
    ]


def test_mujoco_viewer_uses_private_data_and_time_gate() -> None:
    live_data = _Data(1.0)
    viewer_data = _Data(0.0)
    viewer = _Viewer()
    mujoco = _Mujoco(viewer_data)
    backend = SimpleNamespace(
        physics=SimpleNamespace(model=object(), data=live_data),
        config=SimpleNamespace(
            viewer=SimpleNamespace(
                overlays=SimpleNamespace(trail_max_points=10),
                camera=SimpleNamespace(),
            ),
            visuals=SimpleNamespace(
                visual_geom_group=0,
                collision_geom_group=1,
            ),
        ),
    )
    hook = MujocoViewerHook(backend, display_interval_s=0.05)
    def open_viewer():
        viewer.launched_data = viewer_data
        return mujoco, viewer, viewer_data

    hook._open_viewer = open_viewer
    hook._draw_overlay = lambda state, command: None

    hook.on_reset(SimpleNamespace(time_s=0.0))
    for time_s in (0.02, 0.04, 0.06, 0.08, 0.10):
        live_data.time = time_s
        live_data.qpos[:] = time_s
        hook.on_step(
            SimpleNamespace(time_s=time_s, arms={}, metadata={}),
            SimpleNamespace(metadata={}),
            0,
        )
    hook.on_finish(SimpleNamespace(time_s=0.10))

    assert viewer.launched_data is viewer_data
    assert viewer_data is not live_data
    assert 2 <= viewer.sync_calls <= 3  # reset plus latest-value refreshes
    assert mujoco.forward_calls == viewer.sync_calls
    assert np.array_equal(viewer_data.qpos, [0.10, 0.10])


def test_application_runs_simulation_worker_while_presenting_on_main_thread(
    monkeypatch,
) -> None:
    main_thread_id = get_ident()
    hook = _PresentationHook()
    loop = _Loop(hook)
    application = SimulationApplication(
        config=SimpleNamespace(),
        loop=loop,
        hooks_by_name={"panel": hook},
    )
    monkeypatch.setattr(
        "continuum_sim.application.application.save_scenario_artifacts",
        lambda app, result: None,
    )

    result = application.run()

    assert result == "result"
    assert loop.thread_id != main_thread_id
    assert hook.presentation_thread_ids
    assert set(hook.presentation_thread_ids) == {main_thread_id}
    assert hook.closed is True


class _Data:
    def __init__(self, value: float) -> None:
        self.time = value
        self.qpos = np.full(2, value)
        self.qvel = np.full(2, value)
        self.act = np.zeros(0)
        self.ctrl = np.zeros(0)
        self.mocap_pos = np.zeros((0, 3))
        self.mocap_quat = np.zeros((0, 4))
        self.userdata = np.zeros(0)


class _Viewer:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.launched_data = None

    def sync(self) -> None:
        self.sync_calls += 1

    def is_running(self) -> bool:
        return True

    def close(self) -> None:
        pass


class _Mujoco:
    def __init__(self, viewer_data: _Data) -> None:
        self.viewer_data = viewer_data
        self.forward_calls = 0

    def mj_forward(self, model, data) -> None:
        del model
        assert data is self.viewer_data
        self.forward_calls += 1


class _PresentationHook:
    requires_gui_main_thread = True

    def __init__(self) -> None:
        self.presentation_thread_ids: list[int] = []
        self.closed = False

    def present_pending(self, *, force: bool = False) -> None:
        del force
        self.presentation_thread_ids.append(get_ident())

    def close_presentation(self) -> None:
        self.closed = True


class _Loop:
    def __init__(self, hook: _PresentationHook) -> None:
        self.hooks = (hook,)
        self.thread_id = None

    def run(self):
        self.thread_id = get_ident()
        return "result"
