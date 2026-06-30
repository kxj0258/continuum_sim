# In/Out Hole Site Colors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate tendon entry and exit hole sites with independently configured RGBA colors.

**Architecture:** Replace the shared site color in the hole-pattern data model with two required colors. The MuJoCo XML generator selects the appropriate color while it emits each base or link site.

**Tech Stack:** Python dataclasses, YAML configuration, `xml.etree.ElementTree`, Markdown

## Global Constraints

- Remove `site_rgba` without a compatibility fallback.
- Require four-component `in_site_rgba` and `out_site_rgba` values.
- Apply the same in/out mapping to base and link hole sites.
- Do not run tests, verification, lint, format, build, install, viewer, demo, or simulation commands.

---

### Task 1: Load Separate Site Colors

**Files:**
- Modify: `src/continuum_sim/model/hole_pattern.py`
- Modify: `configs/robots/dual_arm_hole_pattern.yaml`

**Interfaces:**
- Consumes: `hole_pattern.site_generation.in_site_rgba` and `out_site_rgba`
- Produces: `TendonHoleSiteGeneration.in_site_rgba` and `.out_site_rgba`

- [ ] **Step 1: Replace the data-model field**

Define two RGBA tuple fields:

```python
in_site_rgba: tuple[float, float, float, float]
out_site_rgba: tuple[float, float, float, float]
```

- [ ] **Step 2: Require both YAML fields**

Load both values with direct indexing so a missing new field is an error:

```python
in_site_rgba=_rgba_tuple(site_raw["in_site_rgba"], "in_site_rgba"),
out_site_rgba=_rgba_tuple(site_raw["out_site_rgba"], "out_site_rgba"),
```

Make `_rgba_tuple` accept the field name and include it in malformed-value errors.

- [ ] **Step 3: Update the project configuration**

Replace `site_rgba` with visually distinct `in_site_rgba` and `out_site_rgba`.

### Task 2: Emit Side-Specific XML Colors

**Files:**
- Modify: `scripts/build_mujoco_dual_arm_model.py`

**Interfaces:**
- Consumes: `TendonHoleSiteGeneration.in_site_rgba` and `.out_site_rgba`
- Produces: `rgba` attributes on generated MuJoCo base/link hole sites

- [ ] **Step 1: Add a side-to-color helper**

```python
def _hole_site_rgba(
    hole_pattern: TendonHolePattern,
    suffix: str,
) -> tuple[float, float, float, float]:
    if suffix == "in":
        return hole_pattern.site_generation.in_site_rgba
    return hole_pattern.site_generation.out_site_rgba
```

- [ ] **Step 2: Use the helper for both site types**

In `_append_tendon_base_sites` and `_append_tendon_link_sites`, format the helper result for each emitted site's `rgba` attribute.

### Task 3: Document the Configuration

**Files:**
- Modify: `docs/dual_arm_mujoco_landing.md`

**Interfaces:**
- Consumes: final YAML field names and generator behavior
- Produces: user-facing configuration reference

- [ ] **Step 1: Describe both required colors**

Add a `site_generation` example and state that `in` and `out` sites use their corresponding RGBA fields for both base and link holes.

### Task 4: Manual Validation Handoff

**Files:**
- No file changes

- [ ] **Step 1: Do not execute checks**

Leave all validation to the user, as required.

- [ ] **Step 2: Suggest manual commands**

Recommend model generation and targeted tests in the final handoff, clearly noting that Codex did not run them.
