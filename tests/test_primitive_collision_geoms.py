from __future__ import annotations

import pytest

from continuum_sim.scenes.primitive_collision import (
    load_primitive_collision_geoms,
    primitive_geom_bbox,
    validate_primitive_collision_geoms,
)


def test_capsule_fromto_bbox_uses_radius() -> None:
    geom = load_primitive_collision_geoms(
        [
            {
                "name": "capsule_hint",
                "type": "capsule",
                "fromto_m": [-1.0, 0.0, 0.5, 1.0, 0.0, 0.5],
                "radius_m": 0.2,
            }
        ]
    )[0]

    bbox = primitive_geom_bbox(geom)

    assert bbox.minimum == pytest.approx((-1.2, -0.2, 0.3))
    assert bbox.maximum == pytest.approx((1.2, 0.2, 0.7))


def test_box_position_size_bbox() -> None:
    geom = load_primitive_collision_geoms(
        [
            {
                "name": "box_hint",
                "type": "box",
                "position_m": [1.0, 2.0, 3.0],
                "size_m": [2.0, 4.0, 6.0],
            }
        ]
    )[0]

    bbox = primitive_geom_bbox(geom)

    assert bbox.minimum == pytest.approx((0.0, 0.0, 0.0))
    assert bbox.maximum == pytest.approx((2.0, 4.0, 6.0))


def test_disabled_false_is_default_and_valid() -> None:
    geom = load_primitive_collision_geoms(
        [
            {
                "name": "sphere_hint",
                "type": "sphere",
                "position_m": [0.0, 0.0, 0.0],
                "radius_m": 0.1,
            }
        ]
    )[0]

    assert geom.enabled is False
    validate_primitive_collision_geoms([geom])


def test_invalid_primitive_type_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="primitive_collision_geoms.*type"):
        load_primitive_collision_geoms([{"name": "bad", "type": "mesh"}])


def test_invalid_radius_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="radius_m"):
        load_primitive_collision_geoms(
            [
                {
                    "name": "bad_capsule",
                    "type": "capsule",
                    "fromto_m": [0, 0, 0, 1, 0, 0],
                    "radius_m": 0.0,
                }
            ]
        )


def test_missing_primitive_geoms_section_is_allowed() -> None:
    assert load_primitive_collision_geoms(None) == []
