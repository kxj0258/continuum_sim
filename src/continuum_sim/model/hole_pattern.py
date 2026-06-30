"""Shared link-local tendon hole pattern loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from continuum_sim.config import load_yaml


HOLE_DISPLAY_MODES = ("none", "routed", "all")


@dataclass(frozen=True)
class TendonHole:
    """One hole with independently configured link-local in/out centers."""

    id: str
    index: int
    angle_deg: float
    xy_m: tuple[float, float]
    in_z_m: float
    out_z_m: float


@dataclass(frozen=True)
class TendonHoleEndpoint:
    """One independently defined in/out site center."""

    id: str
    index: int
    xy_m: tuple[float, float]
    z_m: float


@dataclass(frozen=True)
class TendonHoleSiteGeneration:
    """Site-generation settings for each repeated hole."""

    site_size_m: float
    in_site_rgba: tuple[float, float, float, float]
    out_site_rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class TendonHoleVisualization:
    """Visibility policy that does not affect tendon physics."""

    hole_display: str
    show_tendons: bool


@dataclass(frozen=True)
class TendonSegmentTerminalLink:
    """Per-arm outlet geometry for the final link of one segment."""

    segment_number: int
    link_number: int
    in_holes_from: str
    exclusive_out_holes: bool
    out_holes_by_arm: dict[str, tuple[TendonHoleEndpoint, ...]]


@dataclass(frozen=True)
class TendonHolePattern:
    """Odd/even link-hole templates shared by both generated arms.

    The YAML values are the geometry source of truth: callers consume each
    hole's ``in_z_m`` and ``out_z_m`` independently instead of deriving one
    from the other.
    """

    path: Path
    link_length_m: float
    base_height_m: float
    link_odd_holes: tuple[TendonHole, ...]
    link_even_holes: tuple[TendonHole, ...]
    base_holes: tuple[TendonHole, ...]
    site_generation: TendonHoleSiteGeneration
    visualization: TendonHoleVisualization
    segment_terminal_links: tuple[TendonSegmentTerminalLink, ...]

    def link_holes_for_number(self, link_number: int) -> tuple[TendonHole, ...]:
        if link_number <= 0:
            raise ValueError(f"Link number must be positive, got {link_number}.")
        if link_number % 2 == 1:
            return self.link_odd_holes
        return self.link_even_holes

    def link_hole_by_index(self, index: int, *, link_number: int) -> TendonHole:
        for hole in self.link_holes_for_number(link_number):
            if hole.index == index:
                return hole
        raise KeyError(
            f"Unknown tendon hole index {index} for link number {link_number}."
        )

    def link_in_endpoints(
        self,
        *,
        global_link_number: int,
        segment_number: int,
        segment_link_number: int,
    ) -> tuple[TendonHoleEndpoint, ...]:
        terminal_link = self._segment_terminal_link(
            segment_number,
            segment_link_number,
        )
        if terminal_link is not None:
            holes = self._template_holes(terminal_link.in_holes_from)
        else:
            holes = self.link_holes_for_number(global_link_number)
        return tuple(
            TendonHoleEndpoint(
                id=hole.id,
                index=hole.index,
                xy_m=hole.xy_m,
                z_m=hole.in_z_m,
            )
            for hole in holes
        )

    def link_out_endpoints(
        self,
        *,
        arm_name: str,
        global_link_number: int,
        segment_number: int,
        segment_link_number: int,
    ) -> tuple[TendonHoleEndpoint, ...]:
        terminal_link = self._segment_terminal_link(
            segment_number,
            segment_link_number,
        )
        template_holes = self.link_holes_for_number(global_link_number)
        if terminal_link is None:
            return tuple(
                TendonHoleEndpoint(
                    id=hole.id,
                    index=hole.index,
                    xy_m=hole.xy_m,
                    z_m=hole.out_z_m,
                )
                for hole in template_holes
            )
        try:
            overridden_endpoints = terminal_link.out_holes_by_arm[arm_name]
        except KeyError as exc:
            raise KeyError(
                "hole_pattern.segment_terminal_links has no outlet entry "
                f"for arm {arm_name!r} at segment_{segment_number}_"
                f"link_{segment_link_number}."
            ) from exc
        if terminal_link.exclusive_out_holes:
            return overridden_endpoints
        overridden_by_index = {
            endpoint.index: endpoint
            for endpoint in overridden_endpoints
        }
        return tuple(
            overridden_by_index.get(
                hole.index,
                TendonHoleEndpoint(
                    id=hole.id,
                    index=hole.index,
                    xy_m=hole.xy_m,
                    z_m=hole.out_z_m,
                ),
            )
            for hole in template_holes
        )

    def link_endpoint_by_index(
        self,
        index: int,
        *,
        arm_name: str,
        suffix: str,
        global_link_number: int,
        segment_number: int,
        segment_link_number: int,
    ) -> TendonHoleEndpoint:
        if suffix == "in":
            endpoints = self.link_in_endpoints(
                global_link_number=global_link_number,
                segment_number=segment_number,
                segment_link_number=segment_link_number,
            )
        elif suffix == "out":
            endpoints = self.link_out_endpoints(
                arm_name=arm_name,
                global_link_number=global_link_number,
                segment_number=segment_number,
                segment_link_number=segment_link_number,
            )
        else:
            raise ValueError(f"Unknown hole endpoint suffix {suffix!r}.")
        for endpoint in endpoints:
            if endpoint.index == index:
                return endpoint
        raise KeyError(
            f"Link segment_{segment_number}_link_{segment_link_number} "
            f"has no {suffix} endpoint for hole index {index}."
        )

    def base_hole_by_index(self, index: int) -> TendonHole:
        for hole in self.base_holes:
            if hole.index == index:
                return hole
        raise KeyError(f"Unknown base tendon hole index {index}.")

    def _segment_terminal_link(
        self,
        segment_number: int,
        link_number: int,
    ) -> TendonSegmentTerminalLink | None:
        for terminal_link in self.segment_terminal_links:
            if (
                segment_number == terminal_link.segment_number
                and link_number == terminal_link.link_number
            ):
                return terminal_link
        return None

    def _template_holes(self, template_name: str) -> tuple[TendonHole, ...]:
        if template_name == "link_odd":
            return self.link_odd_holes
        if template_name == "link_even":
            return self.link_even_holes
        raise ValueError(f"Unknown link-hole template {template_name!r}.")

def load_tendon_hole_pattern(path: str | Path) -> TendonHolePattern:
    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    pattern_raw = raw.get("hole_pattern")
    if not isinstance(pattern_raw, dict):
        raise ValueError("hole_pattern must be a mapping.")
    link_odd_raw = pattern_raw.get("link_odd")
    if not isinstance(link_odd_raw, dict):
        raise ValueError("hole_pattern.link_odd must be a mapping.")
    link_even_raw = pattern_raw.get("link_even")
    if not isinstance(link_even_raw, dict):
        raise ValueError("hole_pattern.link_even must be a mapping.")
    base_raw = pattern_raw.get("base")
    if not isinstance(base_raw, dict):
        raise ValueError("hole_pattern.base must be a mapping.")
    link_odd_holes = _holes_from_section(
        link_odd_raw,
        "hole_pattern.link_odd.holes",
    )
    link_even_holes = _holes_from_section(
        link_even_raw,
        "hole_pattern.link_even.holes",
    )
    base_holes = _holes_from_section(base_raw, "hole_pattern.base.holes")
    site_raw = pattern_raw.get("site_generation", {})
    if not isinstance(site_raw, dict):
        raise ValueError("hole_pattern.site_generation must be a mapping.")
    visualization_raw = pattern_raw.get("visualization")
    if not isinstance(visualization_raw, dict):
        raise ValueError("hole_pattern.visualization must be a mapping.")
    segment_terminal_links_raw = pattern_raw.get("segment_terminal_links")
    if not isinstance(segment_terminal_links_raw, list):
        raise ValueError("hole_pattern.segment_terminal_links must be a list.")
    link_odd_length_m = float(link_odd_raw["link_length_m"])
    link_even_length_m = float(link_even_raw["link_length_m"])
    if link_odd_length_m != link_even_length_m:
        raise ValueError(
            "hole_pattern.link_odd.link_length_m and "
            "hole_pattern.link_even.link_length_m must match."
        )
    return TendonHolePattern(
        path=config_path,
        link_length_m=link_odd_length_m,
        base_height_m=float(base_raw["height_m"]),
        link_odd_holes=link_odd_holes,
        link_even_holes=link_even_holes,
        base_holes=base_holes,
        site_generation=TendonHoleSiteGeneration(
            site_size_m=float(site_raw.get("site_size_m", 0.0007)),
            in_site_rgba=_rgba_tuple(site_raw["in_site_rgba"], "in_site_rgba"),
            out_site_rgba=_rgba_tuple(site_raw["out_site_rgba"], "out_site_rgba"),
        ),
        visualization=_visualization_from_dict(visualization_raw),
        segment_terminal_links=_segment_terminal_links_from_list(
            segment_terminal_links_raw,
            link_odd_holes=link_odd_holes,
            link_even_holes=link_even_holes,
        ),
    )


def _holes_from_section(section: dict[str, object], field_name: str) -> tuple[TendonHole, ...]:
    holes_raw = section.get("holes")
    if not isinstance(holes_raw, list) or len(holes_raw) != 12:
        raise ValueError(f"{field_name} must contain exactly 12 items.")
    holes = tuple(sorted((_hole_from_dict(item) for item in holes_raw), key=lambda hole: hole.index))
    indices = [hole.index for hole in holes]
    if indices != list(range(12)):
        raise ValueError(f"{field_name} index values must cover 0..11, got {indices}.")
    return holes


def _hole_from_dict(item: object) -> TendonHole:
    if not isinstance(item, dict):
        raise ValueError("hole_pattern link/base hole items must be mappings.")
    xy = item["xy_m"]
    if not isinstance(xy, list | tuple) or len(xy) != 2:
        raise ValueError("hole_pattern link/base holes[].xy_m must contain exactly 2 numbers.")
    return TendonHole(
        id=str(item["id"]),
        index=int(item["index"]),
        angle_deg=float(item.get("angle_deg", 0.0)),
        xy_m=(float(xy[0]), float(xy[1])),
        in_z_m=float(item["in_z_m"]),
        out_z_m=float(item["out_z_m"]),
    )


def _visualization_from_dict(
    values: dict[str, object],
) -> TendonHoleVisualization:
    hole_display = str(values.get("hole_display", "routed"))
    if hole_display not in HOLE_DISPLAY_MODES:
        raise ValueError(
            "hole_pattern.visualization.hole_display must be one of "
            f"{HOLE_DISPLAY_MODES}, got {hole_display!r}."
        )
    show_tendons = values.get("show_tendons", True)
    if not isinstance(show_tendons, bool):
        raise ValueError(
            "hole_pattern.visualization.show_tendons must be a boolean."
        )
    return TendonHoleVisualization(
        hole_display=hole_display,
        show_tendons=show_tendons,
    )


def _segment_terminal_links_from_list(
    items: list[object],
    *,
    link_odd_holes: tuple[TendonHole, ...],
    link_even_holes: tuple[TendonHole, ...],
) -> tuple[TendonSegmentTerminalLink, ...]:
    templates = {
        "link_odd": link_odd_holes,
        "link_even": link_even_holes,
    }
    terminal_links: list[TendonSegmentTerminalLink] = []
    seen_links: set[tuple[int, int]] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(
                "hole_pattern.segment_terminal_links items must be mappings."
            )
        segment_number = int(item["segment_number"])
        link_number = int(item["link_number"])
        if segment_number <= 0 or link_number <= 0:
            raise ValueError(
                "hole_pattern.segment_terminal_links segment_number and "
                "link_number must be positive."
            )
        link_key = (segment_number, link_number)
        if link_key in seen_links:
            raise ValueError(
                "hole_pattern.segment_terminal_links contains duplicate link "
                f"{link_key}."
            )
        seen_links.add(link_key)
        in_holes_from = str(item["in_holes_from"])
        if in_holes_from not in templates:
            raise ValueError(
                "hole_pattern.segment_terminal_links.in_holes_from must be "
                "'link_odd' or 'link_even'."
            )
        exclusive_out_holes = item.get("exclusive_out_holes")
        if not isinstance(exclusive_out_holes, bool):
            raise ValueError(
                "hole_pattern.segment_terminal_links.exclusive_out_holes "
                "must be a boolean."
            )
        source_by_index = {
            hole.index: hole
            for hole in templates[in_holes_from]
        }
        out_holes_raw = item.get("out_holes")
        if not isinstance(out_holes_raw, dict):
            raise ValueError(
                "hole_pattern.segment_terminal_links.out_holes must be an "
                "arm-to-list mapping."
            )
        arm_names = {str(name) for name in out_holes_raw}
        expected_arm_names = {"executor", "observer"}
        if arm_names != expected_arm_names:
            raise ValueError(
                "hole_pattern.segment_terminal_links.out_holes must define "
                f"{sorted(expected_arm_names)}, got {sorted(arm_names)}."
            )
        out_holes_by_arm: dict[str, tuple[TendonHoleEndpoint, ...]] = {}
        for arm_name, arm_holes_raw in out_holes_raw.items():
            if not isinstance(arm_holes_raw, list) or not arm_holes_raw:
                raise ValueError(
                    "hole_pattern.segment_terminal_links.out_holes arm "
                    "entries must be non-empty lists."
                )
            endpoints: list[TendonHoleEndpoint] = []
            seen_indices: set[int] = set()
            for hole_raw in arm_holes_raw:
                if not isinstance(hole_raw, dict):
                    raise ValueError(
                        "hole_pattern.segment_terminal_links.out_holes "
                        "items must be mappings."
                    )
                hole_index = int(hole_raw["index"])
                if hole_index in seen_indices:
                    raise ValueError(
                        "hole_pattern.segment_terminal_links.out_holes "
                        f"contains duplicate index {hole_index} for "
                        f"arm {arm_name!r}."
                    )
                seen_indices.add(hole_index)
                try:
                    source_hole = source_by_index[hole_index]
                except KeyError as exc:
                    raise ValueError(
                        "hole_pattern.segment_terminal_links.out_holes "
                        f"references unknown index {hole_index}."
                    ) from exc
                hole_id = str(hole_raw["id"])
                if hole_id != source_hole.id:
                    raise ValueError(
                        "hole_pattern.segment_terminal_links.out_holes "
                        f"id/index mismatch: index {hole_index} is "
                        f"{source_hole.id!r}, got {hole_id!r}."
                    )
                endpoints.append(
                    TendonHoleEndpoint(
                        id=hole_id,
                        index=hole_index,
                        xy_m=source_hole.xy_m,
                        z_m=float(hole_raw["z_m"]),
                    )
                )
            out_holes_by_arm[str(arm_name)] = tuple(
                sorted(endpoints, key=lambda endpoint: endpoint.index)
            )
        terminal_links.append(
            TendonSegmentTerminalLink(
                segment_number=segment_number,
                link_number=link_number,
                in_holes_from=in_holes_from,
                exclusive_out_holes=exclusive_out_holes,
                out_holes_by_arm=out_holes_by_arm,
            )
        )
    return tuple(terminal_links)


def _rgba_tuple(
    value: object,
    field_name: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"{field_name} must contain exactly 4 numbers.")
    return tuple(float(component) for component in value)  # type: ignore[return-value]
