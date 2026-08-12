from pathlib import Path
import xml.etree.ElementTree as ET

from numpy.testing import assert_allclose

from continuum_sim.scenes.tool_mjcf_adapter import inject_force_sensor_sphere_tool
from continuum_sim.tools.attachments import load_attachment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_CONFIG = PROJECT_ROOT / "configs" / "tools" / "carbon_remover.yaml"


def test_force_sensor_sphere_tool_injects_geometry_tcp_and_six_axis_sensors() -> None:
    root = ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <body name="executor_last_link">
              <site name="executor_tip" pos="0 0 0.01" quat="1 0 0 0"/>
            </body>
          </worldbody>
          <sensor/>
        </mujoco>
        """
    )
    config = load_attachment_config(TOOL_CONFIG)

    inject_force_sensor_sphere_tool(
        root,
        arm_name="executor",
        tip_site_name="executor_tip",
        config=config,
    )

    sensor_body = root.find(".//body[@name='executor_ft_sensor_body']")
    assert sensor_body is not None
    assert_allclose(_vector(sensor_body.get("pos")), [0.0, 0.0, 0.014])
    sensor_geom = sensor_body.find("./geom[@name='executor_ft_sensor_visual']")
    assert sensor_geom is not None
    assert_allclose(_vector(sensor_geom.get("size")), [0.0075, 0.0075, 0.004])
    assert sensor_geom.get("contype") == "0"
    assert sensor_geom.get("conaffinity") == "0"

    sphere = root.find(".//geom[@name='executor_wiping_sphere']")
    assert sphere is not None
    assert sphere.get("size") == "0.009"
    assert sphere.get("contype") == "1"
    assert sphere.get("conaffinity") == "1"
    tcp = root.find(".//site[@name='executor_tool_tcp']")
    assert tcp is not None
    assert_allclose(_vector(tcp.get("pos")), [0.0, 0.0, 0.009])

    force = root.find("./sensor/force[@name='executor_ft_force']")
    torque = root.find("./sensor/torque[@name='executor_ft_torque']")
    assert force is not None and force.get("site") == "executor_ft_sensor_site"
    assert torque is not None and torque.get("site") == "executor_ft_sensor_site"


def _vector(raw: str | None):
    assert raw is not None
    return [float(value) for value in raw.split()]
