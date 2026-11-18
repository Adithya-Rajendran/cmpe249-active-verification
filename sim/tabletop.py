"""Simulator-agnostic description of the simulated TABLETOP environment: table, room, workspace, cameras, lighting. numpy
only (like sim/rig.py, which it reuses for boxes and cameras).

Why: the second simulated environment (decision 2026-11-04, progress/drafts/2026-11-04-tabletop-design.md) tests whether
evidence v0.3's shelf-motivated background cue transfers to a realistic table. This module turns
configs/sim/tabletop_geometry.json plus a positions file into the same kind of plain data sim/rig.py gives for the shelf
(boxes, a mesh, cameras) and exports it in the existing scene schema (cmpe249.sim-scene.v1) for backends that cannot import
it (Isaac Sim's own Python). The shelf code path is untouched: nothing in sim/rig.py's export changes, and the optional
fields the tabletop adds to the scene spec (mesh 'uv' / 'normals', material 'texture' / 'metallic', 'lighting',
'workspace', 'environment') are never emitted for the shelf.

Frame (tabletop_geometry.json 'frame'): metres; origin at the workspace centre on the table top (z = 0 is the top
surface); x right as seen from the start stop P3, y away from P3, z up. sim/rig.py's arc_stops works unchanged with
arc_centre_behind_front_cm = 0: a stop at azimuth a sits at (R sin a, -R cos a, height) and faces the origin.

Scene parts:
  - the table top: ONE closed slab mesh (24 vertices, flat normals, UVs) carrying the fixed wood texture
    (sim/wood_texture.py): u = (x - x_min) / size_x, v = (y - y_min) / size_y on the top and bottom, the slab's sides on thin
    strips of the same image (near / far edges: the bottom / top strip of height thickness / size_y; left / right ends: the
    left / right strip), so every UV is in [0, 1] and the texture is never repeated;
  - four legs, the floor, the back wall 1 m behind the far edge, two side walls, a baseboard along each wall (boxes);
  - the workspace outline: 24 points on the 0.20 m circle at z = 0 and at z = 0.12 (the frame check's outline);
  - lighting: the shelf presets (sim/isaac_rig.py LIGHTING) with the same numbers relative to the workspace centre, aimed at
    (0, 0, 0.05); carried in the scene spec ('lighting'), since the renderer's own presets are the shelf's.

Positions (python -m sim.tabletop positions): radius 45 cm, lens 38 cm above the table, azimuths -60 / -30 / 0 / 30 / 60,
start P3, F order P1 P2 P4 P5, roll 0 (positions file 'roll_deg', read by sim/rig.py cameras_from_positions). Per stop, the
downward pitch maximises the smallest k1-distorted frame margin of the workspace outline (0.01 deg grid, refined to 0.001,
stored to 0.01); accepted only if every stop's acceptable-pitch window (margin >= 0) is >= 2 deg wide. The margins are also
verified WITH the revisit offsets (sim/arrange.py perturb_camera; ranges of configs/sim/arrange_shelf_v3.json 'revisit':
+-10 mm per world axis, +-0.5 deg yaw / pitch / roll): the worst case over the 64 corners of that box (the margin is a minimum
of nearly linear functions of the offsets, so to first order its minimum over the box is at a corner) and over a seeded
uniform sample, whose share below min_margin_px is the expected redraw rate of the outline check.

    python -m sim.tabletop positions [--out configs/sim/positions_tabletop_A.json]
    python -m sim.tabletop scene [--positions configs/sim/positions_tabletop_A.json] --scene-out scene.json
    python -m sim.tabletop texture --out PATH.png          # = python -m sim.wood_texture
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim import rig as R

GEOMETRY_SCHEMA = "cmpe249.tabletop_geometry.v1"
ENVIRONMENT = "tabletop"
DEFAULT_GEOMETRY_PATH = R.REPO_ROOT / "configs" / "sim" / "tabletop_geometry.json"
DEFAULT_POSITIONS_PATH = R.REPO_ROOT / "configs" / "sim" / "positions_tabletop_A.json"
REVISIT_CONFIG_PATH = R.REPO_ROOT / "configs" / "sim" / "arrange_shelf_v3.json"     # the revisit ranges (shared protocol)
LABELS = ["table", "room"]          # object labels of the static scene (arrangement objects are added by the generator)

# the positions plan (design proposal section 2 = the recorded D2b tabletop plan, DECISIONS 2026-09-26 evening)
ARC = {"radius_cm": 45.0, "arc_centre_behind_front_cm": 0.0, "arc_rotation_deg": 0.0, "height_cm": 38.0, "roll_deg": 0.0,
       "azimuths_deg": [-60.0, -30.0, 0.0, 30.0, 60.0], "names": ["P1", "P2", "P3", "P4", "P5"], "start_stop": "P3",
       "f_order": ["P1", "P2", "P4", "P5"]}
PITCH_GRID = (20.0, 60.0)           # coarse search range of the pitch optimum, deg
WINDOW_GRID = (0.0, 89.0)           # range scanned for the acceptable-pitch window, deg
MIN_WINDOW_DEG = 2.0                # acceptance: every stop's window at least this wide
REVISIT_SAMPLE_N = 2000
REVISIT_SAMPLE_ENTROPY = 7_000_000  # sample draws use Draw([REVISIT_SAMPLE_ENTROPY + k, stream_salt, stop, 0]) (not a seed)


# ---------------------------------------------------------------- geometry

def _read(src: str | Path | Mapping) -> dict:
    if isinstance(src, Mapping):
        return json.loads(json.dumps(src))
    return json.loads(Path(src).read_text())


def load_geometry(src: str | Path | Mapping = DEFAULT_GEOMETRY_PATH) -> dict:
    g = _read(src)
    if g.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError(f"tabletop geometry schema must be {GEOMETRY_SCHEMA!r}, got {g.get('schema')!r}")
    return g


def table_extent(g: Mapping) -> dict:
    """The table top's rectangle in the frame: x_min / x_max / y_min (near edge) / y_max (far edge), z_top = 0, z_bottom."""
    t = g["table"]
    return {"x_min": -t["size_x"] / 2, "x_max": t["size_x"] / 2, "y_min": -t["workspace_centre_from_near_edge"],
            "y_max": t["size_y"] - t["workspace_centre_from_near_edge"], "z_top": 0.0, "z_bottom": -t["slab_thickness"]}


def floor_z(g: Mapping) -> float:
    return -float(g["table"]["top_height_above_floor"])


def wall_y(g: Mapping) -> float:
    """y of the back wall's front face."""
    return table_extent(g)["y_max"] + float(g["room"]["wall_gap_behind_far_edge"])


@dataclass(frozen=True)
class TexturedMesh:
    """A mesh with per-vertex normals and UVs, in world coordinates. faces counter-clockwise seen from outside."""
    name: str
    role: str
    label: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    material: str


def _quad(corners: Sequence[Sequence[float]], normal: Sequence[float]) -> list[np.ndarray]:
    """4 corners in cyclic order, reordered so the quad faces `normal` (counter-clockwise seen from outside)."""
    c = [np.asarray(p, float) for p in corners]
    if float(np.dot(np.cross(c[1] - c[0], c[2] - c[0]), normal)) < 0:
        c = c[::-1]
    return c


def table_top_mesh(g: Mapping) -> TexturedMesh:
    """The slab (top at z = 0, thickness from the geometry) as 6 quads = 24 vertices / 12 triangles with flat normals and
    UVs (module docstring)."""
    e = table_extent(g)
    x0, x1, y0, y1, zt, zb = e["x_min"], e["x_max"], e["y_min"], e["y_max"], e["z_top"], e["z_bottom"]
    sx, sy, th = x1 - x0, y1 - y0, zt - zb

    def uv_xy(p):
        return ((p[0] - x0) / sx, (p[1] - y0) / sy)

    faces = [  # (corners, outward normal, uv function)
        ([(x0, y0, zt), (x1, y0, zt), (x1, y1, zt), (x0, y1, zt)], (0, 0, 1), uv_xy),
        ([(x0, y0, zb), (x1, y0, zb), (x1, y1, zb), (x0, y1, zb)], (0, 0, -1), uv_xy),
        ([(x0, y0, zb), (x1, y0, zb), (x1, y0, zt), (x0, y0, zt)], (0, -1, 0), lambda p: ((p[0] - x0) / sx, (p[2] - zb) / sy)),
        ([(x0, y1, zb), (x1, y1, zb), (x1, y1, zt), (x0, y1, zt)], (0, 1, 0), lambda p: ((p[0] - x0) / sx, 1 - (zt - p[2]) / sy)),
        ([(x0, y0, zb), (x0, y1, zb), (x0, y1, zt), (x0, y0, zt)], (-1, 0, 0), lambda p: ((p[2] - zb) / sx, (p[1] - y0) / sy)),
        ([(x1, y0, zb), (x1, y1, zb), (x1, y1, zt), (x1, y0, zt)], (1, 0, 0), lambda p: (1 - (zt - p[2]) / sx, (p[1] - y0) / sy)),
    ]
    V, N, UV, F = [], [], [], []
    for corners, n, uvf in faces:
        q = _quad(corners, n)
        k = len(V)
        V += q
        N += [n] * 4
        UV += [uvf(p) for p in q]
        F += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    if th <= 0:
        raise ValueError("slab thickness must be positive")
    return TexturedMesh("table_top", "table", "table", np.asarray(V, float), np.asarray(F, np.int64), np.asarray(N, float),
                        np.asarray(UV, float), "table_top")


def table_legs(g: Mapping) -> list[R.Box]:
    e = table_extent(g)
    lg = g["table"]["legs"]
    s, inset = float(lg["size"]), float(lg["inset_from_corner"])
    out = []
    for name, sx, sy in (("leg_near_left", -1, -1), ("leg_near_right", 1, -1), ("leg_far_left", -1, 1), ("leg_far_right", 1, 1)):
        xo = e["x_min"] + inset if sx < 0 else e["x_max"] - inset - s
        yo = e["y_min"] + inset if sy < 0 else e["y_max"] - inset - s
        out.append(R.Box.from_bounds(name, "table", (xo, yo, floor_z(g)), (xo + s, yo + s, e["z_bottom"]),
                                     tuple(g["materials"]["table_leg"]["rgba"])))
    return out


def room_boxes(g: Mapping) -> list[R.Box]:
    """Floor, back wall, side walls, baseboards (label 'room'); all axis-aligned."""
    r, m = g["room"], g["materials"]
    fz, wy, sw = floor_z(g), wall_y(g), float(r["side_wall_x"])
    wt, wh, ny = float(r["wall_thickness"]), float(r["wall_height"]), float(r["floor_near_y"])
    bh, bt = float(r["baseboard"]["height"]), float(r["baseboard"]["thickness"])
    col = {k: tuple(m[k]["rgba"]) for k in ("floor", "wall", "baseboard")}
    return [
        R.Box.from_bounds("floor", "room", (-sw, ny, fz - r["floor_thickness"]), (sw, wy, fz), col["floor"]),
        R.Box.from_bounds("wall_back", "room", (-sw - wt, wy, fz), (sw + wt, wy + wt, fz + wh), col["wall"]),
        R.Box.from_bounds("wall_left", "room", (-sw - wt, ny, fz), (-sw, wy, fz + wh), col["wall"]),
        R.Box.from_bounds("wall_right", "room", (sw, ny, fz), (sw + wt, wy, fz + wh), col["wall"]),
        R.Box.from_bounds("baseboard_back", "room", (-sw, wy - bt, fz), (sw, wy, fz + bh), col["baseboard"]),
        R.Box.from_bounds("baseboard_left", "room", (-sw, ny, fz), (-sw + bt, wy - bt, fz + bh), col["baseboard"]),
        R.Box.from_bounds("baseboard_right", "room", (sw - bt, ny, fz), (sw, wy - bt, fz + bh), col["baseboard"]),
    ]


def box_material(b: R.Box) -> str:
    if b.label == "table":
        return "table_leg"
    return "wall" if b.name.startswith("wall") else ("baseboard" if b.name.startswith("baseboard") else "floor")


def workspace_outline(g: Mapping) -> np.ndarray:
    """(2 x n) x 3 points: n points every 360 / n deg on the workspace circle at each outline height (z = 0, 0.12)."""
    w = g["workspace"]
    n, rad = int(w["outline_points_per_circle"]), float(w["radius"])
    th = 2 * math.pi * np.arange(n) / n
    return np.asarray([(rad * math.cos(a), rad * math.sin(a), float(z)) for z in w["outline_heights"] for a in th], float)


def lighting_spec(g: Mapping) -> dict:
    """The lighting block of the scene spec: presets in the frame, each rect light's forward direction toward 'aim'."""
    L = g["lighting"]
    aim = np.asarray(L["aim"], float)
    presets = {}
    for k, p in L["presets"].items():
        lights = []
        for lt in p["lights"]:
            pos = np.asarray(lt["position"], float)
            d = (aim - pos) / np.linalg.norm(aim - pos)
            lights.append({"name": lt["name"], "type": "rect", "position": R._vec(pos), "forward": R._vec(d),
                           "size_m": lt["size_m"], "intensity": lt["intensity"], "on": bool(lt["on"])})
        presets[k] = {"dome_intensity": p["dome_intensity"], "lights": lights}
    return {"aim": [float(v) for v in aim], "up_hint": [0.0, 0.0, 1.0], "presets": presets,
            "note": "rect lights emit along 'forward' (toward 'aim'); an 'on': false light has intensity 0 in that preset. "
                    "Same numbers as the shelf presets (sim/isaac_rig.py LIGHTING), relative to the workspace centre."}


# ---------------------------------------------------------------- cameras and the scene

def cameras(g: Mapping, positions: Mapping, roll_deg: float | None = None) -> list[R.Camera]:
    """The stops of a positions file as sim/rig.py Cameras (the geometry's 'camera' intrinsics; roll from the file's
    'roll_deg' unless overridden)."""
    if roll_deg is None and positions.get("roll_deg") is None:
        raise ValueError("a tabletop positions file must give roll_deg (there is no fitted tabletop roll)")
    return R.cameras_from_positions(positions, g, roll_deg)


def outline_margin(g: Mapping, cam: R.Camera, distort: bool = True) -> float:
    """Smallest frame margin (px, negative = outside) of the workspace outline; k1 applied when `distort`."""
    try:
        return cam.frame_margin(workspace_outline(g), distort)
    except ValueError:            # a point behind the camera: far outside any frame
        return -math.inf


@dataclass
class TabletopScene:
    geometry: dict
    positions: dict | None
    top: TexturedMesh
    legs: list[R.Box]
    room: list[R.Box]
    cameras: list[R.Camera]
    sources: dict = field(default_factory=dict)

    def all_boxes(self) -> list[R.Box]:
        return list(self.legs) + list(self.room)

    def camera(self, name: str) -> R.Camera:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(name)

    def outline(self) -> np.ndarray:
        return workspace_outline(self.geometry)


def load_scene(geometry: str | Path | Mapping = DEFAULT_GEOMETRY_PATH, positions: str | Path | Mapping | None = None,
               roll_deg: float | None = None) -> TabletopScene:
    g = load_geometry(geometry)
    pos = None if positions is None else _read(positions)
    cams = [] if pos is None else cameras(g, pos, roll_deg)
    src = {k: (None if v is None else "<in-memory>" if isinstance(v, Mapping) else str(v))
           for k, v in (("geometry", geometry), ("positions", positions))}
    return TabletopScene(g, pos, table_top_mesh(g), table_legs(g), room_boxes(g), cams, src)


def camera_spec(c: R.Camera) -> dict:
    """One camera in the scene-spec format of sim/rig.py export_scene_spec."""
    right, up, fwd = c.axes()
    return {"name": c.name, "position": R._vec(c.position), "forward": R._vec(fwd), "up": R._vec(up), "right": R._vec(right),
            "f_px": c.f_px, "width": c.width, "height": c.height, "principal_point_px": [c.width / 2, c.height / 2],
            "fovy_deg": round(c.fovy_deg, 9), "heading_deg": c.heading_deg, "pitch_down_deg": c.pitch_down_deg,
            "roll_deg": c.roll_deg, "k1_fit_not_rendered": c.k1}


def _sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def export_scene_spec(scene: TabletopScene) -> dict:
    """The tabletop as plain JSON-able data in the shelf's scene schema (cmpe249.sim-scene.v1), plus the optional fields the
    shelf never emits: 'environment', mesh 'normals' / 'uv', material 'texture' / 'metallic', 'lighting', 'workspace',
    'tags_enabled' false. Boxes and meshes have the shelf export's keys; 'label' is the object a part belongs to ('table'
    or 'room'; segmentation groups by it)."""
    g = scene.geometry
    boxes = [{"name": b.name, "role": b.label, "label": b.label, "center": R._vec(b.centre), "half": R._vec(b.half_size),
              "quat_wxyz": R._vec(b.quat), "material": box_material(b)} for b in scene.all_boxes()]
    t = scene.top
    meshes = [{"name": t.name, "role": t.role, "label": t.label, "vertices": [R._vec(v) for v in t.vertices],
               "faces": [[int(i) for i in f] for f in t.faces], "normals": [R._vec(n) for n in t.normals],
               "uv": [R._vec(u) for u in t.uv], "uv_interpolation": "vertex", "material": t.material}]
    mats = {}
    for k, m in g["materials"].items():
        mats[k] = {"rgba": list(m["rgba"]), "roughness": m["roughness"]}
        if "metallic" in m:
            mats[k]["metallic"] = m["metallic"]
        if "texture" in m:
            mats[k]["texture"] = dict(m["texture"])
    w = g["workspace"]
    return {
        "schema": R.SCENE_SCHEMA,
        "environment": ENVIRONMENT,
        "units": "m",
        "frame": g["frame"]["axes"] + " Origin: " + g["frame"]["origin"] + ".",
        "camera_model": "pinhole, square pixels, principal point (W/2, H/2), pixel i covers [i, i + 1); image x along "
                        "'right', image y along -'up', looking along 'forward' (right = forward x up)",
        "labels": list(LABELS),
        "materials": mats,
        "boxes": boxes,
        "meshes": meshes,
        "tags": [],
        "tags_enabled": False,
        "cameras": [camera_spec(c) for c in scene.cameras],
        "lighting": lighting_spec(g),
        "workspace": {"radius": w["radius"], "outline_heights": list(w["outline_heights"]),
                      "outline": [R._vec(p) for p in scene.outline()]},
        "mug": None,
        "occluder": None,
        "visibility_classes": {k: g["visibility_classes"][k] for k in ("H_max", "C_min")},
        "provenance": {
            "geometry_file": scene.sources.get("geometry"),
            "positions_file": scene.sources.get("positions"),
            "table": g["table"]["provenance"],
            "room": g["room"]["provenance"],
            "lighting": g["lighting"]["provenance"],
            "cameras": {"poses": "positions file: arc radius / centre / rotation / azimuths, height_cm, per-stop tilt_deg, roll_deg",
                        "intrinsics": g["camera"]["provenance"]},
            "exported_by": "sim/tabletop.py export_scene_spec",
        },
    }


# ---------------------------------------------------------------- ray check: every frame ray meets the modelled room

def _ray_box(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Distance along unit ray (o, d) to an axis-aligned box, inf if missed."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    tmin = np.nanmax(np.minimum(t1, t2))
    tmax = np.nanmin(np.maximum(t1, t2))
    return float(tmin) if tmax >= max(tmin, 0.0) and tmin > 0 else math.inf


def frame_rays(scene: TabletopScene, cam: R.Camera, step_px: int = 48) -> dict:
    """Cast pinhole rays through the frame border (every step_px) and the centre; each must meet a scene part (the slab's box,
    a leg, the floor, a wall, a baseboard) within 20 m. Returns {'n', 'missed': [(u, v)], 'parts': {part: n}}."""
    right, up, fwd = cam.axes()
    W, H = cam.width, cam.height
    border = ([(u, 0) for u in range(0, W + 1, step_px)] + [(u, H) for u in range(0, W + 1, step_px)]
              + [(0, v) for v in range(0, H + 1, step_px)] + [(W, v) for v in range(0, H + 1, step_px)] + [(W / 2, H / 2)])
    e = table_extent(scene.geometry)
    parts = [("table_top", np.array([e["x_min"], e["y_min"], e["z_bottom"]]), np.array([e["x_max"], e["y_max"], e["z_top"]]))]
    parts += [(b.name, *b.bounds()) for b in scene.all_boxes()]
    o = np.asarray(cam.position, float)
    missed, hits = [], {}
    for u, v in border:
        d = fwd + ((u - W / 2) / cam.f_px) * right - ((v - H / 2) / cam.f_px) * up
        d = d / np.linalg.norm(d)
        best = min(((_ray_box(o, d, lo, hi), n) for n, lo, hi in parts), key=lambda x: x[0])
        if best[0] > 20.0:
            missed.append((u, v))
        else:
            hits[best[1]] = hits.get(best[1], 0) + 1
    return {"n": len(border), "missed": missed, "parts": dict(sorted(hits.items()))}


# ---------------------------------------------------------------- pitch rule and the revisit check

def _with_pitch(cam: R.Camera, p: float) -> R.Camera:
    return dataclasses.replace(cam, pitch_down_deg=float(p))


def best_pitch(g: Mapping, cam: R.Camera, grid: tuple[float, float] = PITCH_GRID) -> tuple[float, float]:
    """(pitch, margin): the downward pitch maximising the smallest k1 margin of the workspace outline; 0.01 deg grid over
    `grid`, then 0.001 deg within +-0.02 of the best."""
    coarse = np.round(np.arange(grid[0], grid[1] + 1e-9, 0.01), 4)
    vals = np.array([outline_margin(g, _with_pitch(cam, p)) for p in coarse])
    p0 = float(coarse[int(np.argmax(vals))])
    fine = np.round(np.arange(p0 - 0.02, p0 + 0.02 + 1e-9, 0.001), 5)
    fv = np.array([outline_margin(g, _with_pitch(cam, p)) for p in fine])
    return float(fine[int(np.argmax(fv))]), float(fv.max())


def pitch_window(g: Mapping, cam: R.Camera, floor_px: float = 0.0, grid: tuple[float, float] = WINDOW_GRID) -> tuple[float, float]:
    """(lowest, highest) pitch on a 0.01 deg grid with outline margin >= floor_px (k1); (nan, nan) if none. The acceptable
    pitches form one interval here (checked: no gap inside the returned range)."""
    ps = np.round(np.arange(grid[0], grid[1] + 1e-9, 0.01), 4)
    ok = np.array([outline_margin(g, _with_pitch(cam, p)) >= floor_px for p in ps])
    if not ok.any():
        return float("nan"), float("nan")
    idx = np.nonzero(ok)[0]
    if idx[-1] - idx[0] + 1 != len(idx):
        raise ValueError(f"{cam.name}: the acceptable pitches are not one interval")
    return float(ps[idx[0]]), float(ps[idx[-1]])


def revisit_ranges(config: str | Path | Mapping = REVISIT_CONFIG_PATH) -> dict:
    rc = _read(config)["revisit"]
    return {k: rc[k] for k in ("translation_m", "rotation_deg", "min_margin_px", "distort", "max_redraws", "stream_salt")}


def revisit_check(g: Mapping, cam: R.Camera, stop_index: int, rc: Mapping, n_sample: int = REVISIT_SAMPLE_N) -> dict:
    """Outline margins (k1 if rc['distort']) at the revisit offsets: the worst of the 64 corners of the +-translation /
    +-rotation box (with its delta) and of n_sample seeded uniform draws (sim/arrange.py revisit_draw with entropy
    [REVISIT_SAMPLE_ENTROPY + k], the given stop index, attempt 0), and the sample's share below rc['min_margin_px'] (the
    expected redraw rate of the outline check)."""
    from sim import arrange as A      # lazy: sim/arrange.py may import this module

    distort, floor = bool(rc["distort"]), float(rc["min_margin_px"])
    tr, rot = float(rc["translation_m"]), float(rc["rotation_deg"])
    keys = ("dx_m", "dy_m", "dz_m", "dyaw_deg", "dpitch_deg", "droll_deg")
    corners = []
    for signs in itertools.product((-1.0, 1.0), repeat=6):
        delta = dict(zip(keys, [s * tr for s in signs[:3]] + [s * rot for s in signs[3:]]))
        corners.append((outline_margin(g, A.perturb_camera(cam, delta), distort), delta))
    worst = min(corners, key=lambda x: x[0])
    sample = []
    for k in range(n_sample):
        delta = A.revisit_draw(rc, [REVISIT_SAMPLE_ENTROPY + k], stop_index, 0)
        sample.append(outline_margin(g, A.perturb_camera(cam, delta), distort))
    sample = np.asarray(sample)
    return {"corners_worst_margin_px": round(float(worst[0]), 2), "corners_worst_delta": worst[1],
            "corners_best_margin_px": round(float(max(c[0] for c in corners)), 2),
            "sample_n": int(n_sample), "sample_min_margin_px": round(float(sample.min()), 2),
            "sample_median_margin_px": round(float(np.median(sample)), 2),
            "sample_below_floor": int((sample < floor).sum()), "expected_redraw_rate": float((sample < floor).mean())}


def positions_base() -> dict:
    """The tabletop positions plan without pitches (schema cmpe249.positions.v1)."""
    return {"schema": R.POSITIONS_SCHEMA, "radius_cm": ARC["radius_cm"], "arc_centre_behind_front_cm": ARC["arc_centre_behind_front_cm"],
            "arc_rotation_deg": ARC["arc_rotation_deg"], "height_cm": ARC["height_cm"], "roll_deg": ARC["roll_deg"],
            "stops": [{"name": n, "azimuth_deg": a} for n, a in zip(ARC["names"], ARC["azimuths_deg"])],
            "start_stop": ARC["start_stop"], "f_order": list(ARC["f_order"])}


def make_positions(geometry: str | Path | Mapping = DEFAULT_GEOMETRY_PATH, revisit_config: str | Path | Mapping = REVISIT_CONFIG_PATH,
                   n_sample: int = REVISIT_SAMPLE_N) -> tuple[dict, list[str]]:
    """(positions file, report lines). Raises ValueError if a stop's acceptable-pitch window is narrower than
    MIN_WINDOW_DEG (the acceptance rule) or the stored pitch fails the frame check with the revisit offsets."""
    g = load_geometry(geometry)
    rc = revisit_ranges(revisit_config)
    base = positions_base()
    base["tilt_deg"] = 0.0                           # placeholder for the search only (removed below)
    cams = cameras(g, base)
    stops, checks, lines = [], {}, []
    for i, (s, cam) in enumerate(zip(base["stops"], cams)):
        p_opt, m_opt = best_pitch(g, cam)
        p = round(p_opt, 2)
        c = _with_pitch(cam, p)
        lo, hi = pitch_window(g, cam)
        rv = revisit_check(g, c, i, rc, n_sample)
        checks[s["name"]] = {"tilt_deg": p, "optimum_deg": round(p_opt, 3), "k1_min_margin_px": round(outline_margin(g, c, True), 2),
                             "pinhole_min_margin_px": round(outline_margin(g, c, False), 2), "k1_window_deg": [lo, hi],
                             "k1_window_width_deg": round(hi - lo, 2), "revisit": rv}
        stops.append({"name": s["name"], "azimuth_deg": s["azimuth_deg"], "tilt_deg": p})
        lines.append(f"{s['name']}: az {s['azimuth_deg']:+.0f} pitch {p:.2f} deg (optimum {p_opt:.3f}, {m_opt:.2f} px) | k1 margin "
                     f"{checks[s['name']]['k1_min_margin_px']:.1f} px, pinhole {checks[s['name']]['pinhole_min_margin_px']:.1f} px | "
                     f"window {lo:.2f}-{hi:.2f} deg ({hi - lo:.2f}) | revisit corners worst {rv['corners_worst_margin_px']:.1f} px, "
                     f"sample min {rv['sample_min_margin_px']:.1f} px, redraw rate {rv['expected_redraw_rate']:.4f}")
    bad = [n for n, c in checks.items() if c["k1_window_width_deg"] < MIN_WINDOW_DEG]
    if bad:
        raise ValueError(f"acceptable-pitch window narrower than {MIN_WINDOW_DEG} deg at {bad}: {checks}")
    if any(c["revisit"]["corners_worst_margin_px"] < rc["min_margin_px"] for c in checks.values()):
        lines.append("NOTE: a revisit corner puts the outline outside the frame at some stop (redraws will occur)")
    out = {
        "schema": R.POSITIONS_SCHEMA,
        "environment": ENVIRONMENT,
        "status": "SIM 2026-11-04: the simulated TABLETOP's stops (DEV, not frozen). Written by `python -m sim.tabletop positions`.",
        "note": "Frame: configs/sim/tabletop_geometry.json (origin at the workspace centre on the table top, y away from P3). Lens on "
                "a 45 cm arc about the workspace centre, 38 cm above the table, azimuths -60..60 deg in 30 deg steps, each stop "
                "facing the centre, roll 0. Each stop's tilt_deg (downward pitch) maximises the smallest frame margin of the "
                "workspace outline (24 points on the 0.20 m circle at z = 0 and z = 0.12) with the fitted k1; accepted only if every "
                "stop's acceptable-pitch window (margin >= 0) is >= 2 deg wide. There is no file-level tilt_deg: readers use the "
                "per-stop values (sim/rig.py stop_tilt_deg).",
        "radius_cm": base["radius_cm"], "arc_centre_behind_front_cm": base["arc_centre_behind_front_cm"],
        "arc_centre_note": "the arc centre is the frame origin (the workspace centre); the key name is the shelf schema's",
        "arc_rotation_deg": base["arc_rotation_deg"], "height_cm": base["height_cm"],
        "height_note": "lens height above the table top (z = 0)",
        "roll_deg": base["roll_deg"],
        "roll_note": "every stop; read by sim/rig.py cameras_from_positions (the shelf file has no roll_deg and keeps the P3 fitted 1.98 deg)",
        "tilt_note": "per stop (stops[i].tilt_deg)",
        "stops": stops, "start_stop": base["start_stop"], "f_order": base["f_order"],
        "frame_check": checks,
        "revisit_check": {**{k: rc[k] for k in ("translation_m", "rotation_deg", "min_margin_px", "distort")},
                          "ranges_from": "configs/sim/arrange_shelf_v3.json revisit (the shared revisit protocol)",
                          "method": "worst outline margin over the 64 corners of the +-translation (world x, y, z) / +-rotation (yaw, "
                                    "pitch, roll) box (sim/arrange.py perturb_camera), and over a seeded uniform sample "
                                    f"(revisit_draw, entropy [{REVISIT_SAMPLE_ENTROPY} + k], k < {n_sample}, stream_salt "
                                    f"{rc['stream_salt']}, attempt 0); expected_redraw_rate = the sample's share below min_margin_px. "
                                    "The arrangement frame check will add the target's hull (sim/arrange.py frame_check)."},
        "acceptance": {"rule": f"every stop's k1 acceptable-pitch window >= {MIN_WINDOW_DEG} deg",
                       "min_window_deg": min(c["k1_window_width_deg"] for c in checks.values()), "ok": True},
        "provenance": {"plan": "design proposal section 2 = the recorded D2b tabletop plan (DECISIONS 2026-09-26 evening); "
                               "researcher's decision 2026-11-04 item 1",
                       "tilt_deg": "sim/tabletop.py make_positions (this file's frame_check block is its output)",
                       "geometry": f"configs/sim/tabletop_geometry.json (workspace outline, camera f_px / k1) sha256 "
                                   f"{_sha256_file(DEFAULT_GEOMETRY_PATH) if not isinstance(geometry, Mapping) else '<in-memory>'}"},
    }
    return out, lines


# ---------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m sim.tabletop", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("positions", help="write the tabletop positions file (per-stop pitch, revisit check)")
    p.add_argument("--geometry", default=str(DEFAULT_GEOMETRY_PATH))
    p.add_argument("--out", default=str(DEFAULT_POSITIONS_PATH))
    s = sub.add_parser("scene", help="write the scene spec (cmpe249.sim-scene.v1)")
    s.add_argument("--geometry", default=str(DEFAULT_GEOMETRY_PATH))
    s.add_argument("--positions", default=str(DEFAULT_POSITIONS_PATH), help="'none' for no cameras")
    s.add_argument("--scene-out", required=True)
    t = sub.add_parser("texture", help="write the fixed wood texture PNG (sim/wood_texture.py)")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "positions":
        pos, lines = make_positions(a.geometry)
        Path(a.out).write_text(json.dumps(pos, indent=1) + "\n")
        print("\n".join(lines + [f"wrote {a.out}"]))
    elif a.cmd == "scene":
        sc = load_scene(a.geometry, None if a.positions == "none" else a.positions)
        spec = export_scene_spec(sc)
        Path(a.scene_out).write_text(json.dumps(spec, indent=1) + "\n")
        print(f"wrote {a.scene_out}: {len(spec['boxes'])} boxes, {len(spec['meshes'])} mesh(es), {len(spec['cameras'])} cameras")
    else:
        from sim import wood_texture as WT

        info = WT.write(a.out)
        print(" ".join(f"{k}={v}" for k, v in info.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
