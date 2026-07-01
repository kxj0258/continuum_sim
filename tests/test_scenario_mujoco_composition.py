from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from continuum_sim.runtime.hooks import _configure_mujoco_viewer
from continuum_sim.scenes.engine_mjcf_adapter import retain_spatial_arm


def test_retain_executor_removes_observer_actuator_sensor_references() -> None:
    root = ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <body name="executor_base"/>
            <body name="observer_base"/>
          </worldbody>
          <tendon>
            <spatial name="executor_tendon_1"/>
            <spatial name="observer_tendon_1"/>
          </tendon>
          <actuator>
            <position name="act_executor_tendon_1" tendon="executor_tendon_1"/>
            <position name="act_observer_tendon_1" tendon="observer_tendon_1"/>
          </actuator>
          <sensor>
            <actuatorfrc
              name="sensor_observer_tendon_1_actuator_force"
              actuator="act_observer_tendon_1"
            />
          </sensor>
        </mujoco>
        """
    )

    retain_spatial_arm(root, "executor")

    names = {element.get("name") for element in root.iter()}
    assert "executor_base" in names
    assert "act_executor_tendon_1" in names
    assert all(
        "observer" not in value
        for element in root.iter()
        for value in element.attrib.values()
    )


def test_configure_mujoco_viewer_applies_camera_and_geom_groups() -> None:
    viewer = SimpleNamespace(
        cam=SimpleNamespace(
            lookat=np.zeros(3),
            distance=0.0,
            azimuth=0.0,
            elevation=0.0,
        ),
        opt=SimpleNamespace(geomgroup=np.zeros(6, dtype=int)),
    )
    config = SimpleNamespace(
        visuals=SimpleNamespace(visual_geom_group=1, collision_geom_group=0),
        viewer=SimpleNamespace(
            show_collision_geoms=True,
            camera=SimpleNamespace(
                lookat=(0.025, 0.0, 0.095),
                distance=0.25,
                azimuth=315.0,
                elevation=-25.0,
            ),
        ),
    )

    _configure_mujoco_viewer(viewer, config)

    np.testing.assert_allclose(viewer.cam.lookat, config.viewer.camera.lookat)
    assert viewer.cam.distance == 0.25
    assert viewer.cam.azimuth == 315.0
    assert viewer.cam.elevation == -25.0
    assert viewer.opt.geomgroup[1] == 1
    assert viewer.opt.geomgroup[0] == 1
