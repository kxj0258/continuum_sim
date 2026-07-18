# Unified Task-Intent Control Architecture

The dual MuJoCo tracking, navigation, wiping, and engine-navigation tasks now
share the same control boundary:

1. Task planners resolve scenario YAML and scene annotations into `TaskPlan`.
2. Task controllers advance the plan and emit typed task intents.
3. Low-level control resolves intents into raw tendon-rate references, then the
   execution layer makes those references bending-compatible for MuJoCo tendon
   position actuators.

## Task Planner

Planner input is scenario configuration, robot assembly, scene geometry, and
initial task annotations. Planner output is `TaskPlan`, which carries:

- `waypoints_world`
- optional waypoint orientations
- per-waypoint phase labels
- surface normal and surface point
- target normal force
- per-waypoint normals and standoff distances
- approach masks and source waypoint indices
- clearance and base-approach constraints

Controllers no longer consume anonymous plan dictionaries.

## Task Controller

Task controllers are responsible for phase/state progression only. They consume
`TaskPlan` or a task-specific resolved plan and produce `TaskStep`, which wraps:

- `CartesianTaskIntent` for executor position/orientation servoing
- `ObserverTaskIntent` for observer tracking or collision avoidance
- `ContactForceIntent` for generic force-position contact tasks
- `SafetyTaskIntent` for diagnostics and supervisors
- `TaskStatus` for lifecycle state

Wiping contact control uses the generic contact intent. The current force
feedback mode is `proxy_distance`; the same intent supports
`measured_contact_force` when contact-force sensing is routed through the state.

## Intent Resolver

`IntentResolver` is the first low-level sublayer. It converts task intent into
task-space velocity, then invokes the coordinated whole-body solver.

Input:

- `RobotSystemState`
- `TaskStep`
- `PriorityStackConfig`
- task-space servo gains
- whole-body solver gains
- optional scene query

Output:

- raw executor/observer tendon-rate references
- optional base twist
- task-space and whole-body diagnostics

The default priority stack is declarative:

- executor: position servo, normal force control, orientation servo, scene
  avoidance
- observer: inter-arm avoidance, observer tracking, look-at, scene avoidance

## Actuation Compatibility Layer

`ActuationCompatibilityLayer` is the second low-level sublayer. It receives raw
tendon-rate references and produces executable MuJoCo tendon-position targets.

Input:

- raw `RobotSystemCommand`
- actual tendon displacement
- actuator force feedback
- tendon displacement/rate/lead limits
- bending-space model
- tendon inner-loop configuration

Output:

- tendon position targets for MuJoCo
- compatible and constrained tendon rates
- target lead, force, saturation, and bending residual diagnostics

This layer owns bending compatibility, antagonistic tendon coordination,
tendon-rate limiting, target-lead limiting, force limiting, and actuator tracking
anti-windup. The task resolver is not allowed to bypass it.

## Engine Observer Parameters

Engine navigation now derives inter-arm observer collision parameters from the
top-level `task.observer_control`. The engine-specific subtree is limited to
`observer_control_overrides`, which keeps task-local settings such as observer
offset, ROI blend, and centerline sampling.
