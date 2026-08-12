from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.runtime.mujoco_state_copy import copy_mujoco_dynamic_state


def test_copy_mujoco_dynamic_state_copies_supported_fields() -> None:
    source = _data(1.25, offset=10.0)
    destination = _data(0.0, offset=0.0)

    copy_mujoco_dynamic_state(source, destination)

    assert destination.time == 1.25
    for field_name in (
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "mocap_pos",
        "mocap_quat",
        "userdata",
    ):
        assert_allclose(getattr(destination, field_name), getattr(source, field_name))
        assert not np.shares_memory(
            getattr(destination, field_name), getattr(source, field_name)
        )

    source.qpos[0] = -99.0
    assert destination.qpos[0] != -99.0


def test_copy_mujoco_dynamic_state_accepts_zero_length_arrays() -> None:
    source = SimpleNamespace(
        time=2.0,
        qpos=np.zeros(0),
        qvel=np.zeros(0),
        act=np.zeros(0),
        ctrl=np.zeros(0),
        mocap_pos=np.zeros((0, 3)),
        mocap_quat=np.zeros((0, 4)),
        userdata=np.zeros(0),
    )
    destination = SimpleNamespace(
        time=0.0,
        qpos=np.zeros(0),
        qvel=np.zeros(0),
        act=np.zeros(0),
        ctrl=np.zeros(0),
        mocap_pos=np.zeros((0, 3)),
        mocap_quat=np.zeros((0, 4)),
        userdata=np.zeros(0),
    )

    copy_mujoco_dynamic_state(source, destination)

    assert destination.time == 2.0


def _data(time_s: float, *, offset: float) -> SimpleNamespace:
    return SimpleNamespace(
        time=time_s,
        qpos=np.arange(3, dtype=float) + offset,
        qvel=np.arange(3, dtype=float) + offset,
        act=np.arange(2, dtype=float) + offset,
        ctrl=np.arange(2, dtype=float) + offset,
        mocap_pos=np.arange(6, dtype=float).reshape(2, 3) + offset,
        mocap_quat=np.arange(8, dtype=float).reshape(2, 4) + offset,
        userdata=np.arange(4, dtype=float) + offset,
    )
