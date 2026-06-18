import types

import numpy as np
import pytest
from numpy.testing import assert_allclose

from continuum_sim.model import ThreeSegmentRobotParams
from continuum_sim.runtime.mujoco_contact_projection import (
    FOLLOWER_CONTACT_SOURCE,
    apply_projected_qfrc,
    contact_wrench_to_world,
    finite_difference_follower_jacobian,
    project_follower_contacts,
)
from continuum_sim.scenes.contact_surfaces import make_work_surface


def test_contact_wrench_to_world_uses_contact_frame_transpose() -> None:
    rotation = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    contact = types.SimpleNamespace(frame=rotation.reshape(-1))

    force_world, torque_world = contact_wrench_to_world(
        contact,
        np.array([2.0, 0.0, 1.0, 0.0, 3.0, 4.0], dtype=float),
    )

    assert_allclose(force_world, rotation.T @ [2.0, 0.0, 1.0])
    assert_allclose(torque_world, rotation.T @ [0.0, 3.0, 4.0])


def test_finite_difference_follower_jacobian_has_expected_shapes() -> None:
    params = ThreeSegmentRobotParams.default()

    jacobian_v, jacobian_w = finite_difference_follower_jacobian(
        np.array([0.03, -0.01, 0.02, 0.04, -0.03, 0.01], dtype=float),
        params,
        samples_per_segment=4,
        segment_index=1,
        sample_index=2,
        point_offset_local=np.array([0.001, 0.0, 0.0], dtype=float),
    )

    assert jacobian_v.shape == (3, 6)
    assert jacobian_w.shape == (3, 6)
    assert np.all(np.isfinite(jacobian_v))
    assert np.all(np.isfinite(jacobian_w))


def test_project_follower_contacts_filters_contacts_and_projects_force() -> None:
    params = ThreeSegmentRobotParams.default()
    surface = make_work_surface(
        id="board",
        primitive_id="board_surface_geom",
        center_m=np.zeros(3),
        normal=np.array([1.0, 0.0, 0.0], dtype=float),
        tangent_u=np.array([0.0, 1.0, 0.0], dtype=float),
        width_m=0.1,
        height_m=0.1,
    )
    model = types.SimpleNamespace(
        geom_names=[
            "follower_segment_1_sample_1_collision",
            "scene_board_surface_geom",
            "unrelated_geom",
        ],
    )
    contact = types.SimpleNamespace(
        geom1=0,
        geom2=1,
        pos=np.array([0.0, 0.0, 0.005], dtype=float),
        frame=np.eye(3, dtype=float).reshape(-1),
    )
    data = types.SimpleNamespace(ncon=1, contact=[contact])
    mujoco_module = types.SimpleNamespace(
        mj_contactForce=lambda model_arg, data_arg, index, out: out.__setitem__(
            slice(None),
            np.array([2.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        )
    )

    result = project_follower_contacts(
        mujoco_module=mujoco_module,
        model=model,
        data=data,
        q_segment=np.zeros(6, dtype=float),
        params=params,
        samples_per_segment=4,
        surface=surface,
    )

    assert result.source == FOLLOWER_CONTACT_SOURCE
    assert result.contact_count == 1
    assert result.normal_force_n == pytest.approx(2.5)
    assert_allclose(result.total_force_world, [2.5, 0.0, 0.0])
    assert result.projected_generalized_force_q.shape == (6,)
    assert result.contact_points_world.shape == (1, 3)
    assert result.contact_geom_names == (
        "follower_segment_1_sample_1_collision:scene_board_surface_geom",
    )


def test_project_follower_contacts_returns_zero_without_contacts() -> None:
    params = ThreeSegmentRobotParams.default()
    surface = make_work_surface(
        id="board",
        primitive_id="board_surface_geom",
        center_m=np.zeros(3),
        normal=np.array([1.0, 0.0, 0.0], dtype=float),
        tangent_u=np.array([0.0, 1.0, 0.0], dtype=float),
        width_m=0.1,
        height_m=0.1,
    )

    result = project_follower_contacts(
        mujoco_module=types.SimpleNamespace(mj_contactForce=lambda *args: None),
        model=types.SimpleNamespace(geom_names=[]),
        data=types.SimpleNamespace(ncon=0, contact=[]),
        q_segment=np.zeros(6, dtype=float),
        params=params,
        samples_per_segment=4,
        surface=surface,
    )

    assert result.contact_count == 0
    assert result.normal_force_n == pytest.approx(0.0)
    assert_allclose(result.total_force_world, np.zeros(3))
    assert_allclose(result.projected_generalized_force_q, np.zeros(6))
    assert result.source == FOLLOWER_CONTACT_SOURCE


def test_apply_projected_qfrc_writes_first_six_entries() -> None:
    data = types.SimpleNamespace(qfrc_applied=np.ones(8, dtype=float))

    apply_projected_qfrc(data, np.arange(6, dtype=float))

    assert_allclose(data.qfrc_applied[:6], np.arange(6, dtype=float))
    assert_allclose(data.qfrc_applied[6:], np.ones(2, dtype=float))
