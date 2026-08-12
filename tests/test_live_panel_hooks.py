from __future__ import annotations

import inspect

import numpy as np

from continuum_sim.runtime.live_panel_hooks import (
    LiveDiagnosticsPanelHook,
    LiveTendonPanelHook,
    LiveWipingForcePanelHook,
)
from continuum_sim.runtime.matplotlib_artists import PersistentAxisArtists
from continuum_sim.runtime.viewer_hooks import MatplotlibSystemViewerHook


def test_persistent_axis_reuses_line_artist_between_frames() -> None:
    axis = _Axis()
    artists = PersistentAxisArtists(axis)

    artists.begin_frame()
    first = artists.plot([0.0, 1.0], [1.0, 2.0], label="force")[0]
    artists.begin_frame()
    second = artists.plot([0.0, 1.0], [3.0, 4.0], label="force")[0]

    assert first is second
    assert axis.plot_calls == 1
    assert first.set_data_calls == 1
    assert np.array_equal(first.y, [3.0, 4.0])


def test_runtime_live_panels_do_not_clear_axes_or_flush_events() -> None:
    source = "\n".join(
        inspect.getsource(value)
        for value in (
            LiveTendonPanelHook,
            LiveWipingForcePanelHook,
            LiveDiagnosticsPanelHook,
            MatplotlibSystemViewerHook,
        )
    )

    assert "axis.cla()" not in source
    assert "axis.clear()" not in source
    assert "flush_events(" not in source


class _Line:
    def __init__(self, x, y) -> None:
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.set_data_calls = 0
        self.visible = True

    def set_data(self, x, y) -> None:
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.set_data_calls += 1

    def set_visible(self, visible: bool) -> None:
        self.visible = visible

    def set(self, **kwargs) -> None:
        del kwargs


class _Axis:
    def __init__(self) -> None:
        self.plot_calls = 0

    def plot(self, x, y, *args, **kwargs):
        del args, kwargs
        self.plot_calls += 1
        return [_Line(x, y)]
