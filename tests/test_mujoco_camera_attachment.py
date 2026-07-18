from __future__ import annotations

from xml.etree import ElementTree as ET

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.scenes.scene_builder import inject_tip_camera


def test_inject_tip_camera_mounts_camera_on_tip_site_parent_body() -> None:
    root = ET.fromstring(
        """
<mujoco>
  <worldbody>
    <body name="arm">
      <body name="tip_link">
        <site name="observer_tip" pos="0 0 0.01" quat="1 0 0 0" />
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )

    inject_tip_camera(
        root,
        tip_site_name="observer_tip",
        camera_name="observer_eye_camera",
        tip_to_camera=Pose6D.from_dict(
            {"position": [0.0, 0.0, 0.04], "quat": [1.0, 0.0, 0.0, 0.0]}
        ),
        fovy_deg=60.0,
    )

    camera = root.find(".//camera[@name='observer_eye_camera']")

    assert camera is not None
    assert camera.get("pos") == "0 0 0.04"
    assert camera.get("quat") == "0 1 0 0"
    assert camera.get("fovy") == "60"
