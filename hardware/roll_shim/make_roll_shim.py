"""Roll shims that level the Pi-in-case camera about its viewing axis.

The case sits flat on the shim's inclined top. The incline runs ACROSS the camera
(left-right), so the shim corrects roll without changing pitch. Thick side = camera's
LEFT as seen from behind the camera (2026-10-05 frames: shelf edges slope down to the
right, so the camera's left side must come up). A lip on the low (right) edge stops
the case sliding. Grooves under the thick edge count the degrees (1 groove = 1 deg).

Profile in the XZ plane (X = across the camera, left -> right; Z = up), extruded
along Y (the viewing direction). Prints flat, bottom on the bed, no supports.

Run: uv run --no-sync --with trimesh --with shapely --with mapbox-earcut python make_roll_shim.py
"""
import math
import numpy as np
import trimesh
from shapely.geometry import Polygon, box

ANGLES = (1.0, 2.0, 3.0, 4.0, 5.0)  # degrees of roll correction
WIDTH = 80.0       # across the camera (mm); Pi 5 case is ~60-70 mm wide (same as wedge40)
LENGTH = 105.0     # along the viewing direction (mm); same bed length as wedge40
T_MIN = 2.0        # thickness at the low (right) edge (mm)
LIP_H = 5.0        # lip height above the incline at the low edge (mm)
LIP_T = 3.0        # lip thickness (mm)
GROOVE_W, GROOVE_D, GROOVE_PITCH, GROOVE_X0 = 1.2, 0.6, 3.0, 4.0  # degree-count grooves


def profile(angle_deg):
    rise = WIDTH * math.tan(math.radians(angle_deg))
    t = math.radians(angle_deg)
    hi = np.array([0.0, T_MIN + rise])            # top of the thick (left) edge
    lo = np.array([WIDTH, T_MIN])                 # lip root at the low (right) edge
    n = np.array([math.sin(t), math.cos(t)])      # incline normal (up, tilted toward the low side)
    lip_top = lo + LIP_H * n
    lip_out = np.array([WIDTH + LIP_T, lip_top[1] - LIP_T * math.tan(t)])
    outer = [(0, 0), (WIDTH + LIP_T, 0), tuple(lip_out), tuple(lip_top), tuple(lo), tuple(hi)]
    poly = Polygon(outer).buffer(0)
    for i in range(int(round(angle_deg))):
        x = GROOVE_X0 + i * GROOVE_PITCH
        poly = poly.difference(box(x, -1.0, x + GROOVE_W, GROOVE_D))
    assert poly.is_valid and poly.geom_type == "Polygon"
    return poly, rise


def build(angle_deg):
    poly, rise = profile(angle_deg)
    m = trimesh.creation.extrude_polygon(poly, LENGTH)   # XY profile, extruded along +Z
    # rotate so the profile's Y (up) becomes Z and the extrusion runs along +Y (print flat)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    m.apply_translation(-m.bounds[0])
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0
    # sanity: the incline is the up-facing tilt with the most area (the lip top is a tiny facet)
    up = m.face_normals[:, 2] > 0.9
    tilt = np.round(np.degrees(np.arccos(np.clip(m.face_normals[up][:, 2], -1, 1))), 2)
    areas = {}
    for t, ar in zip(tilt, m.area_faces[up]):
        areas[t] = areas.get(t, 0.0) + ar
    return m, rise, float(max(areas, key=areas.get))


if __name__ == "__main__":
    for a in ANGLES:
        m, rise, meas = build(a)
        name = f"roll_shim_{a:.0f}deg.stl"
        m.export(name)
        ext = m.bounds[1] - m.bounds[0]
        print(f"{name}: incline {meas:.2f} deg (target {a}), rise {rise:.2f} mm over {WIDTH:.0f} mm, "
              f"extents {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm, {m.volume/1000:.1f} cm^3 "
              f"(~{m.volume/1000*1.24:.0f} g PLA solid), grooves {int(round(a))}")
        assert abs(meas - a) < 0.01, (a, meas)
