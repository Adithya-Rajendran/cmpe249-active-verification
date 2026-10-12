# Hardware

Printable parts and markers for the capture rig. The generator scripts are the source of truth.
Rendered previews and the other angle variants are not committed; rebuild them with the commands below.

| Folder | What | Build |
|---|---|---|
| `roll_shim/` | Wedge shim that corrects the camera case's roll. `roll_shim_4deg.stl` is the printed one (thick side on the camera's left). | `uv run --no-sync --with trimesh --with shapely --with mapbox-earcut python make_roll_shim.py` (in the folder) |
| `fiducials/` | Four 5 cm ArUco tags (DICT_4X4_50, IDs 0-3, plus spares) for the corners of the study compartment. They are used to register each view and crop it to the shelf. Print at 100 %; the 10 cm bar checks the scale. | `uv run --no-sync python hardware/fiducials/make_tags.py` |
