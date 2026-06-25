# Engine Assets

This directory holds engine geometry for the engine-cleaning scene scaffold.

## Directory Layout

- `raw/`: CAD source files such as STEP, SLDPRT, and SLDASM.
- `meshes/`: visual meshes for rendering, such as STL, OBJ, or MSH.
- `collision/`: simplified collision meshes, or notes for MuJoCo primitive collision.

STEP, SLDPRT, and SLDASM files cannot be loaded directly by MuJoCo. Export them
from SolidWorks or another CAD tool to STL, OBJ, or MSH before using them in a
scene config.

Keep visual and collision assets separate. A detailed full-engine visual mesh is
useful for preview, but it is usually too complex to use as the only collision
model. Prefer simplified collision meshes or MuJoCo primitives around the actual
contact surfaces.

If SolidWorks exports in millimeters, set the engine scene `scale` to `0.001`.
If the exported mesh is already in meters, use `1.0`. Confirm with:

```bash
python scripts/check_engine_assets.py --config configs/scenes/engine_cleaning.yaml
```

Large CAD and mesh files should not be committed directly to git unless the
project intentionally tracks them. Use Git LFS when these assets need to be
versioned.

MuJoCo may reject STL files with more than 200,000 faces. The preview script can
create a temporary, subsampled STL for loading checks, but production assets
should still be exported at an appropriate visual and collision resolution.

If the nozzle is the main collision target, export a separate nozzle visual mesh
and then build a simplified nozzle collision mesh or primitive collision model.
