# Bending-Space Control Design

## Purpose

All task controllers shall solve continuum-arm motion in a physically compatible
bending space instead of treating the nine tendon lengths of each arm as nine
independent control degrees of freedom.

The change covers:

- scenario tasks: idle, tracking, navigation, wiping, and engine cleaning;
- whole-body single-arm and dual-arm control;
- executor tracking, observer tracking, inter-arm avoidance, and scene avoidance;
- analytic and MuJoCo system backends;
- retained offline and legacy motor-space tracking, navigation, and wiping APIs;
- the system tendon debug interface.

Normal task control must generate compatible tendon commands by construction.
The debug interface retains an explicitly selected raw-tendon diagnostic mode.

## Physical Model

Each three-segment arm uses a six-dimensional bending coordinate

```text
b = [kx_1, ky_1, kx_2, ky_2, kx_3, ky_3]
```

Axial strain is excluded. The corresponding nine-dimensional PCC coordinate is

```text
q = S_b b
```

where `S_b` inserts a zero axial-strain component for every segment. With the
existing full tendon coupling matrix `C_q`,

```text
delta_l = C_q q = C_b b
C_b = C_q S_b
```

The compatible tendon subspace is `range(C_b)`. The least-squares bending state,
compatible projection, and incompatibility residual are

```text
b_hat = pinv(C_b) delta_l
delta_l_compatible = C_b b_hat
r = delta_l - delta_l_compatible
```

The residual norm is diagnostic information. It may be nonzero for measured
MuJoCo state because of elasticity, solver tolerances, or model mismatch, but it
must be zero to numerical precision for normal commanded targets.

This bending-only model matches MuJoCo configurations where
`include_axial_strain: false`. The analytic backend shall use the same assumption
instead of estimating artificial axial strain from tendon measurements.

## Architecture

### Bending model utilities

A focused model module owns:

- construction of `S_b` and `C_b`;
- bending-to-PCC and bending-to-tendon mappings;
- tendon-to-bending least-squares estimation;
- compatible projection and residual calculation;
- mapping diagnostics such as rank and condition number.

The module validates vector sizes, finite values, tendon geometry, and full
column rank. It does not know about controllers or backends.

### Solver layout

The whole-body optimization layout becomes:

```text
[base twist, executor bending rate, observer bending rate, ...]
```

Each enabled arm contributes six columns rather than nine tendon columns.
Fixed-base systems contribute no active base columns, as today.

Task Jacobians are built directly with respect to bending coordinates:

```text
J_b = J_q S_b
```

Tip tracking, centerline collision avoidance, observer-only projection, and
scene avoidance all use the same bending layout. The weighted least-squares,
singularity damping, task priorities, and base handling remain conceptually
unchanged.

The solver maps each arm's solved bending rate to tendon rate only after solving:

```text
delta_l_dot = C_b b_dot
```

`RobotSystemCommand` remains the runtime boundary so hooks, recorders, and
artifacts do not require a parallel command hierarchy.

### Command modes

Each arm command carries an explicit control-space mode:

- `bending_compatible`: default for every normal controller and zero command;
- `raw_tendon_debug`: accepted only for explicit diagnostic operation.

Normal commands are checked for compatibility at the backend boundary. Invalid
dimensions, non-finite values, or a material compatibility residual raise a
clear error rather than being silently accepted.

The raw mode is intentionally not used by scenario tasks. It preserves the
ability to pull one tendon independently when diagnosing routing, signs,
actuator direction, and force response.

## Compatibility-Preserving Limits and Integration

Independent tendon clipping is replaced for compatible commands.

For a requested compatible tendon rate, a single nonnegative arm scale `alpha`
is selected so that:

```text
abs(alpha * delta_l_dot_i) <= tendon_rate_limit_i
lower_i <= delta_l_i + dt * alpha * delta_l_dot_i <= upper_i
```

The minimum admissible scale across all tendons is applied to the complete arm
vector. This preserves the tendon ratios and therefore preserves membership in
`range(C_b)`.

Normal backends integrate a six-dimensional bending target and derive the
nine-dimensional tendon target from `C_b` after each step. Reset establishes a
zero bending target. If a backend must synchronize from a measured tendon
target, it uses the least-squares bending projection and records the discarded
residual.

Raw debug commands retain independent tendon integration and clipping because
their purpose is to leave the compatible subspace deliberately. Switching from
raw mode back to compatible mode reinitializes the compatible target from the
projection of the current target to avoid a discontinuous hidden state.

## Scenario and Legacy Coverage

### Unified scenario path

`tracking`, `navigation`, `wiping`, and `engine_cleaning` continue to construct
Cartesian task objectives through the coordinated controller. Their task
semantics and YAML targets do not change. The common whole-body solver makes all
of them bending-space controllers.

`idle` emits a zero compatible command.

### Retained motor-space path

Legacy differential IK, navigation, hybrid force-position wiping, dynamic
adaptive wiping, and engine-cleaning helpers retain their public motor-velocity
return types.

Internally they:

1. estimate the bending state from tendon measurements;
2. form Cartesian Jacobians with respect to bending coordinates;
3. solve for bending rate;
4. map bending rate through `C_b` to compatible tendon rate;
5. map tendon rate to motor rate using the configured spool/sign mapping;
6. apply one common scale needed to satisfy motor/tendon limits.

This retains caller compatibility while removing incompatible motor commands.

## State Estimation and Kinematics

All task-facing PCC state estimates are formed from bending estimates with zero
axial strain. Forward kinematics still receives the existing nine-dimensional
PCC vector, but its axial entries are exactly zero.

New bending Jacobian helpers coexist with lower-level full-PCC helpers. Existing
full tendon/PCC conversion functions may remain for model inspection and raw
debugging, but normal control paths must not use their unconstrained
nine-dimensional inverse as a control state.

## Diagnostics and Artifacts

Controller or backend metadata shall expose, per arm:

- requested and applied bending rate;
- requested and applied tendon rate;
- compatibility residual vector and norm;
- compatibility tolerance and pass/fail state;
- common limit scale;
- rate-limit and displacement-limit activity;
- bending mapping rank and condition number.

Existing tendon target/current/force panels remain available. The debug panel
adds:

- a default bending-compatible mode;
- per-segment `kx` and `ky` inputs or equivalent compatible presets;
- an explicit raw-tendon diagnostic mode;
- compatibility residual and warning status;
- clear units for curvature, tendon displacement, and force.

## Configuration

No new third-party solver dependency is introduced.

Existing robot tendon geometry, rate limits, displacement limits, task targets,
and MuJoCo XML routing remain authoritative. A small control configuration
section may expose:

- compatibility absolute and relative tolerances;
- whether backend compatibility enforcement is enabled, defaulting to true;
- debug startup mode, defaulting to compatible;
- optional maximum bending-rate magnitude when a project-level limit is needed.

Raw debug bypass must not be configurable as the default for scenario runs.

## Failure Handling

- Rank-deficient `C_b` fails during model/controller construction with the arm
  name and computed rank.
- Non-finite states or commands fail at their owning boundary.
- Normal commands outside the compatible tolerance fail before integration.
- An infeasible displacement step produces zero scale for the blocked direction
  and reports saturation; it does not independently distort tendon components.
- Measured incompatibility is projected for state estimation and reported, not
  treated as a commanded axial degree of freedom.
- Singular Cartesian Jacobians continue to use the current damping and velocity
  scaling mechanism.

## Test Adaptation

Tests shall be added or updated without changing the physical expectations:

- exact `S_b`, `C_b`, forward mapping, inverse estimation, and projection;
- zero axial-strain reconstruction;
- compatible command generation and residual rejection;
- common-scale rate and displacement limiting;
- switching between compatible and raw debug modes;
- whole-body layout and Jacobian dimensions for fixed/mobile, single/dual arms;
- tracking, navigation, wiping, engine cleaning, and idle scenario controllers;
- analytic and MuJoCo backend command handling and metadata;
- retained motor-space APIs producing compatible tendon-equivalent commands;
- debug UI mode and input synchronization.

Per user instruction, implementation work will not automatically execute tests,
builds, linters, formatters, installers, viewers, or simulations. The completed
handoff will provide commands for manual validation.

## Documentation

Update:

- `README.md` with control semantics, debug modes, and manual commands;
- `docs/architecture_overview.md` with the bending-space data flow;
- `docs/coordinate_conventions.md` with coordinate ordering, units, and mapping;
- relevant module docstrings and configuration comments.

Documentation must distinguish commanded compatibility from measured physical
residuals and must state that raw debug mode is intentionally nonphysical as a
multi-tendon control policy.

## Compatibility and Migration

The external scenario YAML task schema and `RobotSystemCommand` container remain
compatible. Existing callers that construct `ArmTendonRateCommand` obtain the
safe compatible mode by default; callers intentionally commanding one tendon
must opt into `raw_tendon_debug`.

This default may expose previously hidden incompatible commands as errors. That
is intentional for normal operation and is the principal migration risk. All
known internal raw-tendon uses must be marked explicitly during implementation.
