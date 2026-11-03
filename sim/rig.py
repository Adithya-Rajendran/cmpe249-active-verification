"""Simulator-agnostic description of the shelf rig: shelf unit, ArUco tags, study mug, design occluders, arc cameras.

Why this exists: the far-spot camera arcs and occluders (data/raw/2026-10-10_far-camera/design_far.md) were designed
with an analytic ray model that treats the mug as a flat card plus a rim disc. Before the student tapes stops and cuts
occluders, a renderer should confirm the same patterns on a real 3-D mesh, and later produce visibility labels for
candidate views. This module turns configs/rig_geometry.json plus a positions draft into plain dataclasses (boxes,
meshes, tags, cameras) that any backend can consume: sim/mujoco_rig.py now, an Isaac Sim backend later. It must stay
free of simulator imports (numpy only).

What it is NOT: an appearance model. Colours here are render defaults so previews are readable; textures, materials,
lighting and lens effects are not modelled, so nothing photometric (detector scores, DINOv2 matches) should be
trusted from a render.

Frame (configs/rig_geometry.json 'frame'): metres, x lateral (+ right as seen from the camera), y depth behind the
#3 front plane (+ into the shelf), z height above the mug-shelf top; origin on the #3 centre line in the front plane at
the mug-shelf top. Right-handed. The analytic design frame (x_a lateral, y_a height, z_a depth; cm) maps as
(x, y, z) = (x_a, z_a, y_a) / 100, so "the mug at z = 24" in the design documents is mug_z = 0.24 here (the y
coordinate).

Cameras: a pinhole with square pixels and the principal point at (W/2, H/2), in continuous pixel coordinates where
pixel i covers [i, i + 1). Stops come from a positions file (schema cmpe249.positions.v1): lens on a horizontal arc of
radius R about a centre on the #3 centre line, rotated by arc_rotation_deg, each stop yawed to face the centre, one
downward tilt for all stops, or a per-stop stops[i].tilt_deg where the file gives one (configs/sim/positions_shelf_A.json,
2026-11-03). Roll is not in the drafts; by default every stop gets the roll fitted at P3, as
arc_far.py assumed. The fitted radial k1 is applied only on request (frame checks); renderers draw a pinhole.

Visibility: visible fraction = target pixels in the full scene / target pixels with everything else hidden, computed
from integer label masks (one id per object, -1 = background). Classes follow geom_measured.py: H <= 0.05, C >= 0.90,
else p; a pattern is the classes over P1..P5, e.g. CCHCC.

Hand-off to backends that cannot import this module (Isaac Sim runs in its own Python 3.12 env): export_scene_spec(rig)
returns the same scene MuJoCo builds from the same Rig object as plain JSON (schema cmpe249.sim-scene.v1):
  python -m sim.rig --positions <draft.json> [--occluder pick_S8] [--mug-z 0.24] --scene-out scene.json
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GEOMETRY_PATH = REPO_ROOT / "configs" / "rig_geometry.json"
GEOMETRY_SCHEMA = "cmpe249.rig_geometry.v1"
POSITIONS_SCHEMA = "cmpe249.positions.v1"
SCENE_SCHEMA = "cmpe249.sim-scene.v1"          # export_scene_spec(): plain-JSON hand-off to other backends
BACKGROUND = -1
FACE_OFFSET = 0.0003    # m: tag cards sit this far in front of the front plane, markers twice as far (no z-fighting)

# render defaults only (appearance is not modelled)
RGBA = {"shelf": (0.93, 0.93, 0.90, 1.0), "back": (0.85, 0.85, 0.82, 1.0), "mug": (0.90, 0.87, 0.75, 1.0),
        "occluder": (0.75, 0.80, 0.88, 1.0), "tag": (0.05, 0.05, 0.05, 1.0), "card": (0.98, 0.98, 0.98, 1.0)}


# ---------------------------------------------------------------- parts

@dataclass(frozen=True)
class Box:
    """A box part. centre / half_size in metres (world frame); quat (w, x, y, z) rotates the box about its centre."""
    name: str
    label: str
    centre: tuple[float, float, float]
    half_size: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_bounds(cls, name: str, label: str, lo: Sequence[float], hi: Sequence[float], rgba=(0.8, 0.8, 0.8, 1.0)) -> Box:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if np.any(hi <= lo):
            raise ValueError(f"{name}: max must exceed min on every axis ({lo} / {hi})")
        return cls(name, label, tuple((lo + hi) / 2), tuple((hi - lo) / 2), tuple(rgba))

    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.quat)

    def corners(self) -> np.ndarray:
        """8 x 3 world corners."""
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * self.half_size
        return np.asarray(self.centre) + s @ self.rotation().T

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.corners()
        return c.min(0), c.max(0)


@dataclass(frozen=True)
class Mesh:
    """A closed triangle mesh. vertices are in the object frame (metres); pos places that frame's origin in the world
    (no rotation: the mug and its handle are built already oriented). faces are counter-clockwise seen from outside."""
    name: str
    label: str
    vertices: np.ndarray
    faces: np.ndarray
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)

    def world_vertices(self) -> np.ndarray:
        return self.vertices + np.asarray(self.pos)


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def revolve_profile(heights: Sequence[float], radii: Sequence[float], n_theta: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Closed solid of revolution about the local z axis: one ring of n_theta vertices per profile row, flat caps at the
    first and last rows. Radius is linear between rows, as geom_measured.width(..., 'frame') interpolates. The top cap
    stands in for the rim opening (the analytic model's rim disc), so the mesh is a closed solid, not an open cup."""
    z, r = np.asarray(heights, float), np.asarray(radii, float)
    if z.ndim != 1 or z.shape != r.shape or len(z) < 2 or np.any(np.diff(z) < 0) or np.any(r <= 0):
        raise ValueError("profile needs >= 2 rows, non-decreasing heights and positive radii")
    th = 2 * math.pi * np.arange(n_theta) / n_theta
    rings = np.stack([np.c_[ri * np.cos(th), ri * np.sin(th), np.full(n_theta, zi)] for zi, ri in zip(z, r)])
    V = np.vstack([rings.reshape(-1, 3), [[0.0, 0.0, z[0]], [0.0, 0.0, z[-1]]]])
    nr = len(z)
    bottom, top = nr * n_theta, nr * n_theta + 1
    F = []
    for k in range(nr - 1):
        for j in range(n_theta):
            a, b = k * n_theta + j, k * n_theta + (j + 1) % n_theta
            c, d = b + n_theta, a + n_theta
            F += [(a, b, c), (a, c, d)]
    for j in range(n_theta):
        a, b = j, (j + 1) % n_theta
        F.append((bottom, b, a))
        F.append((top, (nr - 1) * n_theta + a, (nr - 1) * n_theta + b))
    return V, np.asarray(F, np.int64)


def handle_plate(heights: Sequence[float], radii: Sequence[float], z_range: Sequence[float], protrusion: float,
                 thickness: float, direction: Sequence[float], embed: float) -> tuple[np.ndarray, np.ndarray]:
    """The analytic handle (geom_measured.target: a plate along the handle direction, body silhouette + protrusion,
    z_range high) as a closed prism. At each height the plate runs from `embed` inside the body surface out to body
    radius + protrusion, `thickness` across the handle direction. Heights are the profile rows inside z_range plus its
    two ends, so the outer edge follows the body profile exactly like the analytic samples do."""
    z, r = np.asarray(heights, float), np.asarray(radii, float)
    z0, z1 = float(z_range[0]), float(z_range[1])
    zs = np.unique(np.r_[z0, z[(z > z0) & (z < z1)], z1])
    rs = np.interp(zs, z, r)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    dvec = np.array([d[0], d[1], 0.0])
    pvec = np.array([-d[1], d[0], 0.0])           # (dvec, pvec, z) is right-handed
    h = thickness / 2
    rings = []
    for zi, ri in zip(zs, rs):
        s_in, s_out = ri - embed, ri + protrusion
        rings.append([s_in * dvec - h * pvec, s_out * dvec - h * pvec, s_out * dvec + h * pvec, s_in * dvec + h * pvec])
    V = np.asarray(rings, float)
    V[..., 2] = zs[:, None]
    V = V.reshape(-1, 3)
    n = len(zs)
    F = []
    for k in range(n - 1):
        for j in range(4):
            a, b = 4 * k + j, 4 * k + (j + 1) % 4
            F += [(a, b, b + 4), (a, b + 4, a + 4)]
    F += [(0, 2, 1), (0, 3, 2)]
    t = 4 * (n - 1)
    F += [(t, t + 1, t + 2), (t, t + 2, t + 3)]
    return V, np.asarray(F, np.int64)


def mesh_stats(V: np.ndarray, F: np.ndarray) -> dict:
    """Closedness checks: every undirected edge used by exactly two faces, every directed edge once (consistent
    orientation), Euler characteristic, signed volume (positive = outward faces)."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, dcount = np.unique(e, axis=0, return_counts=True)
    _, ucount = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6)
    return {"n_vertices": len(V), "n_faces": len(F), "n_edges": len(ucount),
            "boundary_edges": int((ucount == 1).sum()), "nonmanifold_edges": int((ucount > 2).sum()),
            "repeated_directed_edges": int((dcount > 1).sum()), "euler": len(V) - len(ucount) + len(F), "volume": vol}


# ---------------------------------------------------------------- tags, occluders, mug

@dataclass(frozen=True)
class Tag:
    """An ArUco marker flush with the front plane: centre (x, 0, z), square side `size`, in-plane rotation (+ = counter-
    clockwise as seen from the camera), on a white card `card_margin` wider on every side."""
    id: int
    centre: tuple[float, float, float]
    size: float
    inplane_rot_deg: float = 0.0
    card_margin: float = 0.0

    def _quat(self) -> tuple[float, float, float, float]:
        a = math.radians(self.inplane_rot_deg)       # CCW seen from the camera (looking along +y) = rotation about -y
        return (math.cos(a / 2), 0.0, -math.sin(a / 2), 0.0)

    def corners(self) -> np.ndarray:
        """4 x 3 marker corners in ArUco order TL, TR, BR, BL as seen from the camera."""
        h = self.size / 2
        local = np.array([[-h, 0, h], [h, 0, h], [h, 0, -h], [-h, 0, -h]], float)
        return np.asarray(self.centre) + local @ quat_to_matrix(self._quat()).T

    def boxes(self) -> list[Box]:
        cx, _, cz = self.centre
        q = self._quat()
        out = [Box(f"tag{self.id}", f"tag{self.id}", (cx, -2 * FACE_OFFSET, cz), (self.size / 2, FACE_OFFSET / 2,
                   self.size / 2), RGBA["tag"], q)]
        if self.card_margin > 0:
            hc = self.size / 2 + self.card_margin
            out.append(Box(f"card{self.id}", "tag_cards", (cx, -FACE_OFFSET, cz), (hc, FACE_OFFSET / 2, hc), RGBA["card"], q))
        return out


@dataclass(frozen=True)
class OccluderSpec:
    """A design occluder: a box on the mug shelf, `width` along x, `height` tall, `thickness` deep, back face at the
    absolute depth y_back, x centre = mug x + shift (occ_box() in geom_measured.py)."""
    name: str
    width: float
    height: float
    thickness: float
    y_back: float
    shift: float
    intended_pattern: str | None = None

    def box(self, mug_x: float = 0.0) -> Box:
        y0 = max(0.0, self.y_back - self.thickness)    # occ_box clamps the front face at the front plane too
        return Box.from_bounds(self.name, "occluder", (mug_x + self.shift - self.width / 2, y0, 0.0),
                               (mug_x + self.shift + self.width / 2, self.y_back, self.height), RGBA["occluder"])

    def perturbed(self, dshift: float = 0.0, dy_back: float = 0.0) -> OccluderSpec:
        return dataclasses.replace(self, shift=self.shift + dshift, y_back=self.y_back + dy_back,
                                   name=f"{self.name}{'' if not (dshift or dy_back) else f'[ds{dshift:+.3f},dy{dy_back:+.3f}]'}")


@dataclass(frozen=True)
class MugSpec:
    """Study mug: silhouette profile revolved about a vertical axis at (x, y), handle plate along `handle` (a key of the
    geometry's handle directions, or None for no handle)."""
    profile_z: tuple[float, ...]
    profile_r: tuple[float, ...]
    x: float
    y: float
    handle: str | None
    handle_dirs: Mapping[str, tuple[float, float]]
    handle_protrusion: float
    handle_z_range: tuple[float, float]
    handle_thickness: float
    handle_embed: float
    n_theta: int = 96

    @property
    def rim_height(self) -> float:
        return self.profile_z[-1]

    @property
    def max_radius(self) -> float:
        return max(self.profile_r)

    def meshes(self) -> list[Mesh]:
        V, F = revolve_profile(self.profile_z, self.profile_r, self.n_theta)
        out = [Mesh("mug_body", "mug", V, F, (self.x, self.y, 0.0), RGBA["mug"])]
        if self.handle is not None:
            Vh, Fh = handle_plate(self.profile_z, self.profile_r, self.handle_z_range, self.handle_protrusion,
                                  self.handle_thickness, self.handle_dirs[self.handle], self.handle_embed)
            out.append(Mesh("mug_handle", "mug", Vh, Fh, (self.x, self.y, 0.0), RGBA["mug"]))
        return out


# ---------------------------------------------------------------- cameras

@dataclass(frozen=True)
class Camera:
    """Pinhole camera. heading_deg: + = turned right (toward +x), 0 = looking straight into the shelf (+y);
    pitch_down_deg: + = looking down; roll_deg: pose_fit_far.py's sign (+ rotates image content counter-clockwise)."""
    name: str
    position: tuple[float, float, float]
    heading_deg: float
    pitch_down_deg: float
    roll_deg: float
    f_px: float
    width: int
    height: int
    k1: float = 0.0

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(right, up, forward) unit vectors in the world frame."""
        h, p, r = (math.radians(v) for v in (self.heading_deg, self.pitch_down_deg, self.roll_deg))
        fwd = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), -math.sin(p)])
        right = np.array([math.cos(h), -math.sin(h), 0.0])
        up = np.cross(right, fwd)
        right, up = math.cos(r) * right - math.sin(r) * up, math.sin(r) * right + math.cos(r) * up
        return right, up, fwd

    @property
    def fovy_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / 2 / self.f_px))

    def project(self, points: np.ndarray, distort: bool = False) -> np.ndarray:
        """World points (N x 3) -> (N x 2) continuous pixel coordinates (u right, v down). distort applies k1."""
        right, up, fwd = self.axes()
        d = np.atleast_2d(np.asarray(points, float)) - np.asarray(self.position)
        zc = d @ fwd
        if np.any(zc <= 0):
            raise ValueError(f"{self.name}: point behind the camera")
        x, y = (d @ right) / zc, -(d @ up) / zc
        if distort and self.k1:
            q = 1 + self.k1 * (x * x + y * y)
            x, y = x * q, y * q
        return np.c_[self.width / 2 + self.f_px * x, self.height / 2 + self.f_px * y]

    def frame_margin(self, points: np.ndarray, distort: bool = False) -> float:
        """Smallest distance (px) from the projected points to the frame edge; negative = outside (arc_far margin())."""
        uv = self.project(points, distort)
        return float(min(uv[:, 0].min(), uv[:, 1].min(), self.width - uv[:, 0].max(), self.height - uv[:, 1].max()))


# ---------------------------------------------------------------- the rig

@dataclass
class Rig:
    boxes: list[Box]
    tags: list[Tag]
    mug: MugSpec
    occluder: OccluderSpec | None
    cameras: list[Camera]
    geometry: dict = field(repr=False)
    positions: dict | None = field(default=None, repr=False)
    sources: dict = field(default_factory=dict)      # {'geometry': path, 'positions': path} as given to load_rig

    def occluder_box(self) -> Box | None:
        return None if self.occluder is None else self.occluder.box(self.mug.x)

    def all_boxes(self) -> list[Box]:
        """Every box part (shelf, tag markers and cards, occluder); with meshes() this is the whole scene."""
        out = list(self.boxes)
        for t in self.tags:
            out += t.boxes()
        ob = self.occluder_box()
        return out + ([ob] if ob is not None else [])

    def meshes(self) -> list[Mesh]:
        return self.mug.meshes()

    def labels(self) -> list[str]:
        """Object labels in a fixed order; label masks store the index into this list (-1 = background)."""
        return ["shelf", "mug", "occluder"] + [f"tag{t.id}" for t in self.tags] + ["tag_cards"]

    def camera(self, name: str) -> Camera:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(name)

    def with_occluder(self, occluder: OccluderSpec | str | None) -> Rig:
        return dataclasses.replace(self, occluder=resolve_occluder(self.geometry, occluder))

    def with_mug(self, **changes: Any) -> Rig:
        return dataclasses.replace(self, mug=dataclasses.replace(self.mug, **changes))

    def c3_outline(self) -> np.ndarray:
        """12 points arc_far.py used for the #3 bbox: opening corners at the front and back wall, plus the outer
        front-face corners (side-panel outer faces x floor-board bottom / mid-board top)."""
        s = self.geometry["shelf"]
        hw, d, h3 = s["c3_inner"]["half_width"], s["c3_inner"]["depth"], s["c3_inner"]["clear_height"]
        t, ho = s["board_thickness"], s["unit_half_outer"]
        pts = [(sx * hw, y, z) for sx in (-1, 1) for y in (0.0, d) for z in (0.0, h3)]
        pts += [(sx * ho, 0.0, z) for sx in (-1, 1) for z in (-t, h3 + t)]
        return np.asarray(pts, float)

    def tag_corners(self) -> dict[int, np.ndarray]:
        return {t.id: t.corners() for t in self.tags}


def _read(src: str | Path | Mapping) -> dict:
    if isinstance(src, Mapping):
        return dict(src)
    return json.loads(Path(src).read_text())


def load_geometry(src: str | Path | Mapping = DEFAULT_GEOMETRY_PATH) -> dict:
    g = _read(src)
    if g.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError(f"rig geometry schema must be {GEOMETRY_SCHEMA!r}, got {g.get('schema')!r}")
    return g


def occluder_spec(geometry: Mapping, name: str) -> OccluderSpec:
    for o in geometry["occluders"]["items"]:
        if o["name"] == name:
            return OccluderSpec(o["name"], o["width"], o["height"], o["thickness"], o["y_back"], o["shift"],
                                o.get("intended_pattern"))
    raise KeyError(f"no occluder {name!r} in the rig geometry")


def occluder_names(geometry: Mapping) -> list[str]:
    return [o["name"] for o in geometry["occluders"]["items"]]


def resolve_occluder(geometry: Mapping, occluder: OccluderSpec | str | None) -> OccluderSpec | None:
    """None, an OccluderSpec, or the name of one of the geometry's occluders -> OccluderSpec | None."""
    if occluder is None or isinstance(occluder, OccluderSpec):
        return occluder
    return occluder_spec(geometry, occluder)


def stop_tilt_deg(positions: Mapping, stop: Mapping) -> float:
    """Downward pitch of one stop: the stop's own `tilt_deg` if it has one (per-stop pitch, e.g.
    configs/sim/positions_shelf_A.json), else the file's single `tilt_deg` (the 2026-10-10 drafts)."""
    if stop.get("tilt_deg") is not None:
        return float(stop["tilt_deg"])
    if positions.get("tilt_deg") is None:
        raise ValueError(f"stop {stop.get('name')!r} has no tilt_deg and the positions file has no default tilt_deg")
    return float(positions["tilt_deg"])


def arc_stops(positions: Mapping) -> list[dict]:
    """Lens positions of a positions file: x lateral, y = -(out from the front plane), heading facing the arc centre.
    Mirrors arc_far.make_stops: the stop at azimuth a sits at a + arc_rotation_deg about a centre on x = 0.
    tilt_deg is the stop's pitch (stop_tilt_deg: per stop if given, else the file's single tilt)."""
    if positions.get("schema") != POSITIONS_SCHEMA:
        raise ValueError(f"positions schema must be {POSITIONS_SCHEMA!r}")
    radius = positions["radius_cm"] / 100
    yc = positions["arc_centre_behind_front_cm"] / 100
    rot = positions.get("arc_rotation_deg", 0.0)
    out = []
    for s in positions["stops"]:
        az = math.radians(s["azimuth_deg"] + rot)
        out.append({"name": s["name"], "x": radius * math.sin(az), "y": yc - radius * math.cos(az),
                    "heading_deg": -math.degrees(az), "tilt_deg": stop_tilt_deg(positions, s)})
    return out


def cameras_from_positions(positions: Mapping, geometry: Mapping, roll_deg: float | None = None) -> list[Camera]:
    cam = geometry["camera"]
    roll = cam["p3_fit"]["roll_deg"] if roll_deg is None else roll_deg
    z = positions["height_cm"] / 100
    return [Camera(s["name"], (s["x"], s["y"], z), s["heading_deg"], s["tilt_deg"], roll, cam["f_px"],
                   cam["width_px"], cam["height_px"], cam["k1"]) for s in arc_stops(positions)]


def frame_margins(rig: Rig, cam: Camera, distort: bool = True) -> dict:
    """In-frame margins (px, negative = outside) of the #3 outline (rig.c3_outline(), arc_far.py's 12 points) and of
    each ArUco tag's 4 corners at one camera, with the fitted k1 by default (the frame check of
    data/raw/2026-11-02_sim/frame_check.py). 'min' is the smallest of them all."""
    out = {"c3": cam.frame_margin(rig.c3_outline(), distort),
           "tags": {t.id: cam.frame_margin(t.corners(), distort) for t in rig.tags}}
    out["min"] = min([out["c3"], *out["tags"].values()])
    return out


def p3_fit_camera(geometry: Mapping, name: str = "P3_fit") -> Camera:
    """The real P3 camera as fitted on far_p3.jpg (pose_fit_far.json k1), including its yaw and roll."""
    cam, p = geometry["camera"], geometry["camera"]["p3_fit"]
    return Camera(name, (p["x"], p["y"], p["z"]), p["heading_deg"], p["pitch_down_deg"], p["roll_deg"], cam["f_px"],
                  cam["width_px"], cam["height_px"], cam["k1"])


def mug_spec(geometry: Mapping, mug_z: float = 0.24, mug_x: float = 0.0, handle: str | None = "toward",
             handle_thickness: float | None = None, n_theta: int = 96) -> MugSpec:
    m = geometry["mug"]
    h = m["handle"]
    if handle is not None and handle not in h["directions"]:
        raise ValueError(f"handle must be one of {sorted(h['directions'])} or None")
    return MugSpec(tuple(m["profile"]["z"]), tuple(m["profile"]["r"]), mug_x, mug_z, handle,
                   {k: tuple(v) for k, v in h["directions"].items()}, h["protrusion"], tuple(h["z_range"]),
                   h["thickness"] if handle_thickness is None else handle_thickness, h["embed"], n_theta)


def shelf_boxes(geometry: Mapping) -> list[Box]:
    out = []
    for p in geometry["shelf"]["parts"]:
        out.append(Box.from_bounds(p["name"], "shelf", p["min"], p["max"],
                                   RGBA["back"] if p["name"] == "back_panel" else RGBA["shelf"]))
    return out


def tags_from_geometry(geometry: Mapping) -> list[Tag]:
    t = geometry["tags"]
    return [Tag(i["id"], tuple(i["centre"]), t["marker_size"], i["inplane_rot_deg"], t["card_margin"]) for i in t["items"]]


def load_rig(geometry_json: str | Path | Mapping = DEFAULT_GEOMETRY_PATH, positions_json: str | Path | Mapping | None = None,
             occluder: OccluderSpec | str | None = None, mug_z: float = 0.24, mug_x: float = 0.0,
             handle: str | None = "toward", handle_thickness: float | None = None, roll_deg: float | None = None,
             n_theta: int = 96) -> Rig:
    """Build the rig. mug_z is the mug axis depth behind the #3 front plane in metres (the design documents' 'z'; the
    sim's y coordinate); occluder is a name from the geometry (e.g. 'pick_S8'), an OccluderSpec or None. Without a
    positions file the rig has no cameras (add them with dataclasses.replace)."""
    g = load_geometry(geometry_json)
    pos = None if positions_json is None else _read(positions_json)
    cams = [] if pos is None else cameras_from_positions(pos, g, roll_deg)
    src = {k: (None if v is None else "<in-memory>" if isinstance(v, Mapping) else str(v))
           for k, v in (("geometry", geometry_json), ("positions", positions_json))}
    return Rig(shelf_boxes(g), tags_from_geometry(g), mug_spec(g, mug_z, mug_x, handle, handle_thickness, n_theta),
               resolve_occluder(g, occluder), cams, g, pos, src)


# ---------------------------------------------------------------- label-mask helpers (any backend)

def visible_fraction(full: np.ndarray, alone: np.ndarray, label: int) -> float | None:
    """Share of the target's unoccluded pixels (label in `alone`, everything else hidden) that are still the target in
    `full` (nothing hidden). None if the target has no pixels in frame."""
    a = alone == label
    n = int(np.count_nonzero(a))
    if n == 0:
        return None
    return int(np.count_nonzero(a & (full == label))) / n


def classify(v: float | None, h_max: float = 0.05, c_min: float = 0.90) -> str:
    if v is None:
        return "?"
    return "H" if v <= h_max else ("C" if v >= c_min else "p")


def pattern(fracs: Sequence[float | None], h_max: float = 0.05, c_min: float = 0.90) -> str:
    return "".join(classify(v, h_max, c_min) for v in fracs)


def label_bbox(labels: np.ndarray, label: int) -> tuple[int, int, int, int] | None:
    """(u0, v0, u1, v1) inclusive pixel indices of a label, or None."""
    vs, us = np.nonzero(labels == label)
    if len(us) == 0:
        return None
    return int(us.min()), int(vs.min()), int(us.max()), int(vs.max())


def bbox_margin_px(bbox: Sequence[int], width: int, height: int) -> int:
    """Pixels between an inclusive-index bbox and the nearest frame edge (0 = touching)."""
    u0, v0, u1, v1 = bbox
    return int(min(u0, v0, width - 1 - u1, height - 1 - v1))


# ---------------------------------------------------------------- scene-spec export (other simulators, other Pythons)

def _vec(v: Sequence[float]) -> list[float]:
    return [round(float(x), 9) for x in v]


def _box_role(b: Box) -> tuple[str, str]:
    """(role, material) of a box part. Materials are short names for the render defaults in RGBA."""
    if b.label == "occluder":
        return "occluder", "occluder"
    if b.label == "tag_cards":
        return "tag_card", "card"
    if b.label.startswith("tag"):
        return "tag_marker", "tag"
    return "shelf", "back" if b.name == "back_panel" else "shelf"


def export_scene_spec(rig: Rig) -> dict:
    """The rig as plain JSON-able data (schema cmpe249.sim-scene.v1) for backends that cannot import this module, e.g.
    Isaac Sim in its own Python 3.12 env. Same geometry MuJoCo gets from build_mjcf(rig): every box but the tag
    markers (those are 'tags', as corner quads in the front plane; MuJoCo draws them as 0.3 mm plates 0.6 mm in
    front, cards 0.3 mm in front, to avoid z-fighting), the mug meshes in world coordinates, the cameras as
    position / forward / up. 'label' is the object a part belongs to: segmentation groups by it (rig.labels())."""
    boxes = []
    for b in rig.all_boxes():
        role, material = _box_role(b)
        if role == "tag_marker":
            continue
        boxes.append({"name": "occluder" if role == "occluder" else b.name, "role": role, "label": b.label,
                      "center": _vec(b.centre), "half": _vec(b.half_size),
                      "quat_wxyz": _vec(b.quat), "material": material})
    meshes = [{"name": m.name, "role": "target", "label": m.label, "vertices": [_vec(v) for v in m.world_vertices()],
               "faces": [[int(i) for i in f] for f in m.faces], "material": "mug"} for m in rig.meshes()]
    tg = rig.geometry["tags"]
    tags = [{"id": t.id, "label": f"tag{t.id}", "dictionary": tg["dictionary"], "corners": [_vec(c) for c in t.corners()],
             "size_m": t.size, "card_margin_m": t.card_margin, "inplane_rot_deg": t.inplane_rot_deg} for t in rig.tags]
    cams = []
    for c in rig.cameras:
        right, up, fwd = c.axes()
        cams.append({"name": c.name, "position": _vec(c.position), "forward": _vec(fwd), "up": _vec(up), "right": _vec(right),
                     "f_px": c.f_px, "width": c.width, "height": c.height, "principal_point_px": [c.width / 2, c.height / 2],
                     "fovy_deg": round(c.fovy_deg, 9), "heading_deg": c.heading_deg, "pitch_down_deg": c.pitch_down_deg,
                     "roll_deg": c.roll_deg, "k1_fit_not_rendered": c.k1})
    g = rig.geometry
    return {
        "schema": SCENE_SCHEMA,
        "units": "m",
        "frame": g["frame"]["axes"] + " Origin: " + g["frame"]["origin"] + ".",
        "camera_model": "pinhole, square pixels, principal point (W/2, H/2), pixel i covers [i, i + 1); image x along "
                        "'right', image y along -'up', looking along 'forward' (right = forward x up)",
        "labels": rig.labels(),
        "materials": {"shelf": {"rgba": list(RGBA["shelf"])}, "back": {"rgba": list(RGBA["back"])},
                      "mug": {"rgba": list(RGBA["mug"])}, "occluder": {"rgba": list(RGBA["occluder"])},
                      "tag": {"rgba": list(RGBA["tag"])}, "card": {"rgba": list(RGBA["card"])}},
        "boxes": boxes,
        "meshes": meshes,
        "tags": tags,
        "cameras": cams,
        "mug": {"axis_xy": [rig.mug.x, rig.mug.y], "rim_height": rig.mug.rim_height, "handle": rig.mug.handle,
                "handle_thickness": rig.mug.handle_thickness if rig.mug.handle else None},
        "occluder": None if rig.occluder is None else dataclasses.asdict(rig.occluder),
        "visibility_classes": {k: g["visibility_classes"][k] for k in ("H_max", "C_min")},
        "provenance": {
            "geometry_file": rig.sources.get("geometry"),
            "positions_file": rig.sources.get("positions"),
            "boxes (shelf)": g["shelf"]["provenance"],
            "boxes (occluder)": g["occluders"]["provenance"],
            "tags": g["tags"]["provenance"],
            "meshes (mug)": g["mug"]["provenance"],
            "cameras": {"poses": "positions file: arc radius / centre / rotation / azimuths, height_cm, tilt_deg (per stop "
                                 "where given); roll "
                                 f"{rig.cameras[0].roll_deg if rig.cameras else None} deg (P3 fit unless overridden)",
                        **g["camera"]["provenance"]},
            "exported_by": "sim/rig.py export_scene_spec",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m sim.rig",
                                 description="Write the rig as a JSON scene spec (cmpe249.sim-scene.v1) for other backends.")
    ap.add_argument("--geometry", default=str(DEFAULT_GEOMETRY_PATH))
    ap.add_argument("--positions", default=None, help="positions file (cmpe249.positions.v1); without it: no cameras")
    ap.add_argument("--occluder", default="none", help="an occluder name from the geometry (e.g. pick_S8) or 'none'")
    ap.add_argument("--mug-z", type=float, default=0.24, help="mug axis depth behind the #3 front plane, m (design 'z')")
    ap.add_argument("--mug-x", type=float, default=0.0)
    ap.add_argument("--handle", default="toward", choices=["toward", "away", "left", "right", "none"])
    ap.add_argument("--handle-thickness", type=float, default=None, help="m; default from the geometry")
    ap.add_argument("--roll-deg", type=float, default=None, help="default: the P3 fitted roll for every stop")
    ap.add_argument("--scene-out", required=True)
    a = ap.parse_args(argv)
    g = load_geometry(a.geometry)
    if a.occluder != "none" and a.occluder not in occluder_names(g):
        ap.error(f"--occluder must be 'none' or one of {occluder_names(g)}")
    rig = load_rig(a.geometry, a.positions, occluder=None if a.occluder == "none" else a.occluder, mug_z=a.mug_z,
                   mug_x=a.mug_x, handle=None if a.handle == "none" else a.handle, handle_thickness=a.handle_thickness,
                   roll_deg=a.roll_deg)
    spec = export_scene_spec(rig)
    Path(a.scene_out).write_text(json.dumps(spec, indent=1) + "\n")
    print(f"wrote {a.scene_out}: {len(spec['boxes'])} boxes, {len(spec['meshes'])} meshes, {len(spec['tags'])} tags, "
          f"{len(spec['cameras'])} cameras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
