from pathlib import Path
import struct
import xml.etree.ElementTree as ET

from numpy.testing import assert_allclose

from continuum_sim.scenes.engine_mjcf_adapter import (
    inject_engine_scene,
    prepare_mujoco_stl_parts,
)
from continuum_sim.scenes.engine_scene import load_engine_scene_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_engine_scene_uses_opaque_silver_native_material() -> None:
    config = load_engine_scene_config(
        PROJECT_ROOT / "configs" / "scenes" / "engine_scene.yaml"
    )
    root = ET.fromstring("<mujoco><asset/><worldbody/></mujoco>")

    inject_engine_scene(
        root,
        config,
        output_dir=None,
        include_visual_mesh=True,
        include_collision_mesh=False,
        include_control_primitives=False,
    )

    material = root.find("./asset/material[@name='engine_silver']")
    geom = root.find(".//geom[@name='engine_visual']")
    assert material is not None
    assert geom is not None
    assert geom.get("material") == "engine_silver"
    assert_allclose(
        [float(v) for v in geom.get("rgba").split()],
        [0.66, 0.68, 0.71, 1.0],
    )
    assert_allclose([float(v) for v in material.get("rgba").split()], [0.66, 0.68, 0.71, 1.0])
    assert material.get("specular") == "0.72"
    assert material.get("shininess") == "0.48"


def test_large_engine_stl_is_losslessly_split_into_mujoco_parts(tmp_path: Path) -> None:
    source = tmp_path / "large.stl"
    face_count = 7
    header = b"lossless split".ljust(80, b"\0")
    records = [bytes([index]) * 50 for index in range(face_count)]
    source.write_bytes(header + struct.pack("<I", face_count) + b"".join(records))

    parts = prepare_mujoco_stl_parts(source, tmp_path, "engine", max_faces=3)

    assert [struct.unpack_from("<I", part.read_bytes(), 80)[0] for part in parts] == [
        3,
        3,
        1,
    ]
    payload = b"".join(part.read_bytes()[84:] for part in parts)
    assert payload == b"".join(records)
