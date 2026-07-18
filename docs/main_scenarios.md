# Main Scenario Commands

The primary MuJoCo tasks use one YAML per task. Switch between the dual-arm
and single-arm variants by editing `scenario.arm_mode`.

```yaml
scenario:
  arm_mode: dual   # dual or single
```

In `dual` mode, the scenario uses the executor and observer assembly. In
`single` mode, it uses the executor-only assembly and the generated MuJoCo XML
retains only the executor arm. Task logic, controller gains, runtime settings,
and artifact settings stay with the same YAML.

## Recommended Main Tasks

```powershell
python scripts/run_scenario.py configs/scenarios/mujoco_tracking.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_navigation.yaml
python scripts/run_scenario.py configs/scenarios/engine_navigation.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_wiping.yaml
python scripts/run_scenario.py configs/scenarios/mujoco_point_servo.yaml
```

The default `arm_mode` values are:

| Scenario | Default | Baseline |
| --- | --- | --- |
| `mujoco_tracking.yaml` | `dual` | `dual_mujoco_tracking.yaml` |
| `mujoco_navigation.yaml` | `dual` | `dual_mujoco_navigation.yaml` |
| `engine_navigation.yaml` | `dual` | `dual_engine_navigation.yaml` |
| `mujoco_wiping.yaml` | `dual` | `dual_mujoco_wiping.yaml` |
| `mujoco_point_servo.yaml` | `single` | `single_mujoco_point_servo.yaml` |

`mujoco_point_servo.yaml` keeps the executor point-servo behavior from the
single-arm task. When switched to `dual`, the observer arm uses the same
collision-avoidance control style as the tracking task.

## Reference And Debug Tasks

The older `dual_*`, `single_*`, and `*_analytic_*` scenario files remain in the
repository for comparison, migration checks, and debugging reference. Analytic
scenarios are not part of the recommended main task list.
