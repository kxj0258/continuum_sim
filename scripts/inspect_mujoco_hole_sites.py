"""Print MuJoCo world coordinates for hole in/out sites on selected bodies."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = (
    PROJECT_ROOT
    / "assets"
    / "mujoco"
    / "dual_three_segment_arm_tendon_with_visuals_mobile_base.xml"
)
DEFAULT_BODIES = ("executor_base", "executor_segment_1_link_1", "executor_segment_1_link_2")
HOLE_SITE_PATTERN = re.compile(r"_hole_(\d+)_(in|out)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML_PATH,
        help="Path to the MuJoCo XML to inspect.",
    )
    parser.add_argument(
        "--bodies",
        nargs="+",
        default=list(DEFAULT_BODIES),
        help="Body names whose direct hole sites should be printed.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=9,
        help="Decimal precision for printed coordinates.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print machine-readable CSV instead of aligned text.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a passive MuJoCo viewer after printing coordinates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_path = args.xml.resolve()
    if not xml_path.is_file():
        print(f"XML file does not exist: {xml_path}", file=sys.stderr)
        return 1

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        print(
            "Failed to import mujoco. Activate the project environment first, "
            "for example: conda activate continuum_sim",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rows = _collect_hole_site_rows(
        mujoco,
        model,
        data,
        body_names=tuple(args.bodies),
    )
    _print_rows(rows, precision=args.precision, csv=args.csv, xml_path=xml_path)

    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)
    return 0


def _collect_hole_site_rows(
    mujoco,
    model,
    data,
    *,
    body_names: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo body does not exist: {body_name!r}")
        body_pos = np.asarray(data.xpos[body_id], dtype=float).copy()
        body_mat = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
        site_rows = _hole_sites_for_body(
            mujoco,
            model,
            data,
            body_name=body_name,
            body_id=body_id,
        )
        rows.append(
            {
                "kind": "body",
                "body": body_name,
                "site": "",
                "hole": "",
                "side": "",
                "world_pos": body_pos,
                "body_x_axis": body_mat[:, 0],
                "body_y_axis": body_mat[:, 1],
                "body_z_axis": body_mat[:, 2],
            }
        )
        rows.extend(site_rows)
    return rows


def _hole_sites_for_body(
    mujoco,
    model,
    data,
    *,
    body_name: str,
    body_id: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for site_id in range(model.nsite):
        if int(model.site_bodyid[site_id]) != body_id:
            continue
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if site_name is None:
            continue
        match = HOLE_SITE_PATTERN.search(site_name)
        if match is None:
            continue
        hole_index = int(match.group(1))
        side = match.group(2)
        rows.append(
            {
                "kind": "site",
                "body": body_name,
                "site": site_name,
                "hole": hole_index,
                "side": side,
                "world_pos": np.asarray(data.site_xpos[site_id], dtype=float).copy(),
                "local_pos": np.asarray(model.site_pos[site_id], dtype=float).copy(),
            }
        )
    rows.sort(key=lambda row: (int(row["hole"]), 0 if row["side"] == "in" else 1))
    return rows


def _print_rows(
    rows: list[dict[str, object]],
    *,
    precision: int,
    csv: bool,
    xml_path: Path,
) -> None:
    if csv:
        print("kind,body,site,hole,side,world_x,world_y,world_z,local_x,local_y,local_z")
        for row in rows:
            world_pos = row["world_pos"]
            local_pos = row.get("local_pos", np.asarray((np.nan, np.nan, np.nan)))
            print(
                ",".join(
                    (
                        str(row["kind"]),
                        str(row["body"]),
                        str(row["site"]),
                        str(row["hole"]),
                        str(row["side"]),
                        *_format_values(world_pos, precision),
                        *_format_values(local_pos, precision),
                    )
                )
            )
        return

    print(f"xml: {xml_path}")
    current_body = None
    for row in rows:
        if row["kind"] == "body":
            current_body = row["body"]
            print()
            print(f"[body] {current_body}")
            print(f"  world_pos: {_format_vec(row['world_pos'], precision)}")
            print(f"  x_axis:    {_format_vec(row['body_x_axis'], precision)}")
            print(f"  y_axis:    {_format_vec(row['body_y_axis'], precision)}")
            print(f"  z_axis:    {_format_vec(row['body_z_axis'], precision)}")
            continue
        print(
            f"  hole_{int(row['hole']):02d}_{row['side']:<3} "
            f"world={_format_vec(row['world_pos'], precision)} "
            f"local={_format_vec(row['local_pos'], precision)} "
            f"site={row['site']}"
        )


def _format_vec(values: object, precision: int) -> str:
    return "[" + ", ".join(_format_values(values, precision)) + "]"


def _format_values(values: object, precision: int) -> list[str]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return [f"{value:.{precision}f}" for value in array]


if __name__ == "__main__":
    raise SystemExit(main())
