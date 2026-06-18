import matplotlib
import pytest

matplotlib.use("Agg", force=True)

from continuum_sim.visualization.wiping_force_panel import WipingForceMonitorPanel


def test_wiping_force_panel_updates_and_closes_under_agg() -> None:
    panel = WipingForceMonitorPanel(target_normal_force_n=1.5, history_points=2)

    sample0 = panel.update(
        time_s=0.0,
        normal_force_n=0.0,
        force_error_n=1.5,
        contact_proxy_m=0.003,
        phase="approach",
        waypoint_index=0,
        contact_source="distance_proxy",
        in_contact=False,
        redraw=False,
    )
    sample1 = panel.update(
        time_s=0.02,
        normal_force_n=1.25,
        force_error_n=0.25,
        contact_proxy_m=-0.002,
        phase="contact",
        waypoint_index=1,
        contact_source="mujoco_contact_force",
        in_contact=True,
        redraw=False,
    )
    panel.update(
        time_s=0.04,
        normal_force_n=1.4,
        force_error_n=0.1,
        contact_proxy_m=-0.0022,
        phase="contact",
        waypoint_index=2,
        contact_source="distance_proxy",
        in_contact=True,
        redraw=False,
    )

    assert sample0.in_contact is False
    assert sample1.contact_source == "mujoco_contact_force"
    assert panel.time_s == [0.02, 0.04]
    assert panel.normal_force_n == pytest.approx([1.25, 1.4])
    assert panel.contact_proxy_m == pytest.approx([-0.002, -0.0022])

    panel.show(block=False)
    panel.flush_events()
    panel.close()
