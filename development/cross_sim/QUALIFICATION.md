# Cross-simulator qualification

Use [the declared limits](qualification_v1.json) and the [development plan](PLAN.md).
The limits apply at the native image size. Task accuracy has a separate evaluation.

Render the same source arrangement in both engines. Use its nominal clear reference at each stop.
Keep the four query states separate. Preserve the physical intervention even when the camera cannot see its effect.

Read cameras and object transforms from the compiled scene. Compare them with the common input.
World rotations use column vectors. Image coordinates increase rightwards and downwards.
Project the original mesh independently for the mask comparison. Ignore a two-pixel band at rasterized boundaries.

[MuJoCo](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera) and [OpenUSD](https://openusd.org/release/api/class_usd_geom_camera.html) document their camera conventions.
Test the installed engines against the declared values.

Check the target and blocker masks in every state. A removed target must have zero pixels.
Measure the clear-state RGB change over the eroded target mask. Repeat the clear-present render after the other states.
Inspect both engines' RGB images, reference boxes and texture orientation before accepting the adapters.
All these checks use evaluator-only information.

Release the paired pilot only after both engines pass. Keep each failed attempt and its input hashes.
Diagnose a failed engineering check before changing code. Record any change to the scientific criteria before another attempt.

## Reproduce the check

Use Python 3.13.5 and NumPy 2.5.3 for cohort generation.
Keep the dated manifest and its arrangement files together. The historical arrangement inventory contributes to their hashes.
Use MuJoCo 3.17.0 with EGL for the MuJoCo adapter. Use the recorded Isaac Sim 6.2 environment for the Isaac adapter.
Each engine needs the same source models, textures and tag files.

List the capture arguments in the appropriate runtime.

```sh
python -m development.cross_sim.cohort --help
python -m development.cross_sim.mujoco_capture --help
python -m development.cross_sim.isaac_driver --help
```

Run the independent check in a Python environment with NumPy, Pillow and OpenCV.
Set the variables below to the manifest, asset root, completed captures, Isaac binding and reviewer output directory.

```sh
python -m development.cross_sim.qualification --engine mujoco \
  --manifest "$manifest" --assets "$assets" --renders "$mujoco_renders" \
  --out "$review/mujoco-audit.json"
python -m development.cross_sim.qualification --engine isaac \
  --manifest "$manifest" --assets "$assets" --renders "$isaac_renders" \
  --binding "$isaac_binding" --out "$review/isaac-audit.json"
```

Record the visual and operator review with its split, status and evidence paths.
Store the final audits where the scorer can read them. The release records their absolute paths and hashes.

```sh
python -m development.cross_sim.qualification_release --manifest "$manifest" \
  --visual-review "$review/visual-review.json" --isaac-audit "$review/isaac-audit.json" \
  --mujoco-audit "$review/mujoco-audit.json" --out "$review/release.json"
```
