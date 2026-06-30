# In/Out Hole Site Colors

## Goal

Allow generated MuJoCo hole sites to use one color for tendon entry (`in`) sites and another color for tendon exit (`out`) sites.

## Configuration

Replace the single `hole_pattern.site_generation.site_rgba` field with two required fields:

```yaml
hole_pattern:
  site_generation:
    site_size_m: 0.0006
    in_site_rgba: [1.0, 0.2, 0.1, 1.0]
    out_site_rgba: [0.1, 0.4, 1.0, 1.0]
```

This is an intentional breaking configuration change. The loader will not accept or fall back to the old `site_rgba` field. Missing or malformed new fields will raise a configuration error.

## Data Model and Loading

`TendonHoleSiteGeneration` will expose `in_site_rgba` and `out_site_rgba`. The hole-pattern loader will parse both required four-component RGBA values from `site_generation`.

## MuJoCo XML Generation

The dual-arm model generator will select the RGBA value from the site suffix:

- `*_hole_*_in` sites use `in_site_rgba`.
- `*_hole_*_out` sites use `out_site_rgba`.

The same rule applies to both arm-base holes and continuum-link holes. Site size and all hole coordinates remain unchanged.

## Documentation

Update the dual-arm hole-pattern configuration and its landing documentation to describe the two required color fields and their in/out mapping.

## Validation

No tests, builds, linters, formatters, installers, viewers, simulations, or verification commands will be run automatically. Suggested manual checks will be provided after implementation.
