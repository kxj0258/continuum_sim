import matplotlib
import numpy as np
from numpy.testing import assert_allclose

matplotlib.use("Agg")

from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.visualization.pcc_viewer import (
    PCCInteractiveViewer,
    compute_view_data,
    named_q,
)


def test_compute_view_data_from_default_q() -> None:
    params = ThreeSegmentRobotParams.default()
    view_data = compute_view_data(np.zeros(9), params, samples_per_segment=8)

    assert view_data.fk.centerline.shape == (22, 3)
    assert view_data.tendon_delta.shape == (9,)
    assert view_data.axis_limit > 0.0
    assert_allclose(view_data.tip_position, [0.0, 0.0, 0.12], atol=1.0e-14)
    assert_allclose(view_data.tendon_delta, np.zeros(9), atol=1.0e-14)


def test_named_q_states_have_expected_shape_and_effect() -> None:
    params = ThreeSegmentRobotParams.default()

    straight = named_q("straight")
    bend_x = named_q("bend_x")
    three_segment = named_q("three_segment")

    assert straight.shape == (9,)
    assert bend_x.shape == (9,)
    assert three_segment.shape == (9,)
    assert_allclose(straight, np.zeros(9), atol=1.0e-14)

    bend_data = compute_view_data(bend_x, params)
    three_segment_data = compute_view_data(three_segment, params)
    assert bend_data.tip_position[0] > 0.0
    assert np.linalg.norm(three_segment_data.tip_position[:2]) > 0.0


def test_viewer_update_runs_without_gui_display() -> None:
    params = ThreeSegmentRobotParams.default()
    viewer = PCCInteractiveViewer(params, samples_per_segment=6)
    try:
        straight_data = viewer.update_plot(redraw=False)
        assert straight_data.fk.centerline.shape == (16, 3)

        viewer.set_q(named_q("three_segment"))
        three_segment_data = viewer.update_plot(redraw=False)
        assert np.linalg.norm(three_segment_data.tip_position[:2]) > 0.0
    finally:
        viewer.close()


def test_viewer_axes_stay_fixed_across_shape_updates() -> None:
    params = ThreeSegmentRobotParams.default()
    viewer = PCCInteractiveViewer(params, samples_per_segment=6)
    try:
        viewer.update_plot(redraw=False)
        initial_limits = (
            viewer.ax.get_xlim(),
            viewer.ax.get_ylim(),
            viewer.ax.get_zlim(),
        )

        viewer.set_q(named_q("three_segment"))
        updated_limits = (
            viewer.ax.get_xlim(),
            viewer.ax.get_ylim(),
            viewer.ax.get_zlim(),
        )

        assert_allclose(updated_limits, initial_limits)
    finally:
        viewer.close()
