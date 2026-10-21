# Hardware

Printable parts and markers for the capture rig. The generator scripts are the source of truth.
Rendered previews and the other angle variants are not committed; rebuild them with the commands below.

| Folder | What | Build |
|---|---|---|
| `roll_shim/` | Wedge shim that corrects the camera case's roll. `roll_shim_4deg.stl` is the printed one (thick side on the camera's left). | `uv run --no-sync --with trimesh --with shapely --with mapbox-earcut python make_roll_shim.py` (in the folder) |
| `pi_case/` | Rigid case for the Pi 5 + AI HAT+ (16 mm stack) + Camera Module 3, in two parts: `pi_case_base.stl` and `pi_case_lid.stl`. The Pi stack sits unscrewed in a cradle and the lid holds it down. The camera is screwed flat to the front wall (4 x M2.5 x 8 + captured nuts), and the lid to the base (4 x M2.5 x 8 + captured nuts). A 1/4"-20 tripod nut is in the floor, and the sides are 72 mm apart for a phone clamp. The BOM, assembly order and fit checks are in its README. | `uv run --no-sync --with trimesh --with shapely --with mapbox-earcut --with manifold3d python hardware/pi_case/make_pi_case.py` (from the repo root) |
| `fiducials/` | Four 5 cm ArUco tags (DICT_4X4_50, IDs 0-3, plus spares) for the corners of the study compartment. They are used to register each view and crop it to the shelf. Print at 100 %; the 10 cm bar checks the scale. | `uv run --no-sync python hardware/fiducials/make_tags.py` |
