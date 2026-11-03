"""Seeded arrangement generator for the simulated shelf: (seed, config) -> arrangement. numpy only.

Why: docs/SIMULATION_STUDY.md moves the course study into simulation, where arrangements come from a seed and a fixed
generator instead of being hand-built. One arrangement mirrors one physical arrangement of capture/run_arrangement.py:
  - one study target in compartment #3, at a random lateral position, depth and yaw where every stop sees it whole;
  - 2-5 distractor objects (other YCB-Video models) in #3 and #4, with non-overlapping footprints, never occluding the
    target from any stop;
  - one occluder of one family (design boards S8 / M16 / IND28, a plain white box over the target's footprint, i.e.
    the pilot-2 failure case, or a YCB box object). It has a 'placed' pose in front of the target, chosen so the
    start stop P3 is mostly blocked, and a 'parked' pose that occludes nothing;
  - the four phases, in capture order (PHASES = capture/run_arrangement.py PHASES):
      reference  target in place, occluder parked       (the per-stop references)
      visible    scene unchanged                         (query, condition visible)
      occluded   occluder placed                         (query, condition occluded)
      removed    target removed, occluder as in occluded (query, condition removed: the matched layout)
    each phase at every stop under L1 and L2.
The four option-A design cases (design_arrangement(0..3), seeds "design-0".."design-3") are fixed, predeclared dev
arrangements: the measured mug at design z 0.24 with pick_S8 / pick_M16+3 / pick_M16-5 / pick_IND28.

Geometry is decided by projecting shapes through the pinhole cameras of the positions file (as rendered), so the
generator needs no meshes and no renderer: each model's convex hull (YCB-V hull points, configs/sim/ycbv_hulls.json;
the measured mug's mesh) for visibility, its rest-pose bounding box (models_info) for footprints. Both visibility tests
are conservative in the right direction: 'hidden at P3' means the occluder's projected hull covers the target's
projected hull (p3_coverage_min), and a 'clear' stop means the two hulls do not overlap at all. Every family except
IND28 must leave >= 2 clear stops (so another view resolves the occlusion and a removed episode can reach the k = 2
clear-miss rule); IND28 must cover every stop (UNKNOWN is the right answer, predeclared). The real visibility labels
always come from the rendered instance masks (sim/isaac_batch.py), never from this file.

Integrity (docs/SIMULATION_STUDY.md): dev seeds are 0-9999 and are the only integers accepted. Seeds >= 100000 are
reserved for test ([open]) and refused (check_dev_seed). Every arrangement records the sha256 of this file and of
sim/rig.py, the config, the hull file, the geometry and the positions file, plus the numpy version. Random draws use
only Generator.random() (PCG64 doubles), so the stream does not depend on numpy's distribution algorithms.

    python -m sim.arrange --seeds 0-5 --design --out DIR      # DIR/<ARR>.json + DIR/index.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sim import rig as R

GENERATOR_VERSION = "0.1.0"
ARRANGEMENT_SCHEMA = "cmpe249.sim-arrangement.v0"
CONFIG_SCHEMA = "cmpe249.sim-arrange-config.v0"
INDEX_SCHEMA = "cmpe249.sim-arrangement-index.v0"
DEV_SEED_RANGE = (0, 9999)
TEST_SEED_MIN = 100000           # reserved for test ([open]); never generated, rendered, scored or viewed before the freeze
DEFAULT_CONFIG_PATH = R.REPO_ROOT / "configs" / "sim" / "arrange_shelf_v0.json"
THIS_FILE = Path(__file__).resolve()
RIG_FILE = Path(R.__file__).resolve()

# (phase, role, condition): capture/run_arrangement.py PHASES (tests/test_sim_arrange.py checks they agree)
PHASES = (("reference", "reference", "visible"), ("visible", "query", "visible"),
          ("occluded", "query", "occluded"), ("removed", "query", "removed"))
PHASE_STATE = {"reference": (True, "parked"), "visible": (True, "parked"),       # (target present, occluder pose)
               "occluded": (True, "placed"), "removed": (False, "placed")}
CONDITIONS = ("visible", "occluded", "removed")
LIGHTINGS = ("L1", "L2")


class SeedError(ValueError):
    """A seed outside the dev range (test seeds are reserved)."""


def check_dev_seed(seed: Any) -> int:
    """The seed as an int if it is a dev seed (0-9999); SeedError otherwise. Seeds >= 100000 are the reserved test
    range and are refused with that reason; 10000-99999 are unassigned."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise SeedError(f"a dev seed is an integer, got {seed!r}")
    seed = int(seed)
    if seed >= TEST_SEED_MIN:
        raise SeedError(f"seed {seed} is in the reserved TEST range (>= {TEST_SEED_MIN}, [open] in "
                        "docs/SIMULATION_STUDY.md): test seeds are never generated, rendered, scored or viewed before the freeze")
    lo, hi = DEV_SEED_RANGE
    if not lo <= seed <= hi:
        raise SeedError(f"seed {seed} is not a dev seed ({lo}-{hi})")
    return seed


def arrangement_id(seed: int, prefix: str = "SD") -> str:
    return f"{prefix}{check_dev_seed(seed):04d}"


def parse_seeds(text: str) -> list[int]:
    """'0-5' or '0,3,7' (or a mix) -> checked dev seeds, in order, no repeats."""
    out: list[int] = []
    for part in (p.strip() for p in str(text).split(",") if p.strip()):
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise SeedError(f"bad seed range {part!r}")
            if hi >= TEST_SEED_MIN:
                check_dev_seed(hi)
            vals = range(lo, hi + 1)
        else:
            vals = [int(part)]
        for v in vals:
            v = check_dev_seed(v)
            if v not in out:
                out.append(v)
    return out


# ---------------------------------------------------------------- hashing / io

def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else R.REPO_ROOT / p


# ---------------------------------------------------------------- random draws

class Draw:
    """Uniform draws built only on Generator.random() (stable PCG64 doubles)."""

    def __init__(self, entropy: int | Sequence[int]):
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))

    def u(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(lo) + (float(hi) - float(lo)) * float(self.rng.random())

    def index(self, n: int) -> int:
        if n <= 0:
            raise ValueError("empty choice")
        return min(int(self.rng.random() * n), n - 1)

    def choice(self, seq: Sequence) -> Any:
        return seq[self.index(len(seq))]

    def integer(self, lo: int, hi: int) -> int:
        """lo..hi inclusive."""
        return int(lo) + self.index(int(hi) - int(lo) + 1)

    def weighted_order(self, weights: Mapping[str, float]) -> list[str]:
        """Keys in a weighted random order (successive draws without replacement); zero weights are left out."""
        left = {k: float(w) for k, w in weights.items() if float(w) > 0}
        out = []
        while left:
            keys = list(left)
            c = np.cumsum([left[k] for k in keys])
            x = self.u(0.0, float(c[-1]))
            k = keys[min(int(np.searchsorted(c, x, side="right")), len(keys) - 1)]
            out.append(k)
            del left[k]
        return out


# ---------------------------------------------------------------- bodies and 2-D geometry

@dataclass(frozen=True)
class Body:
    """An object: bounding-box extents `size` (sx, sy, sz) in its own frame, the bottom-centre of that box at `base`
    (x, y, z), yawed by yaw_deg about z (+ = counter-clockwise seen from above). The renderer places a model the same
    way: rotated bounding box centred on base x/y, lowest point on base z. `local_points` (N x 3, body frame) are the
    model's convex-hull points when known; visibility tests use them (shape_points), footprints use the box."""
    size: tuple[float, float, float]
    base: tuple[float, float, float]
    yaw_deg: float = 0.0
    local_points: np.ndarray | None = field(default=None, compare=False, repr=False)

    def shape_points(self) -> np.ndarray:
        """World points whose convex hull is the object's (its hull points, else the 8 box corners)."""
        if self.local_points is None:
            return self.corners()
        a = math.radians(self.yaw_deg)
        c, s = math.cos(a), math.sin(a)
        return self.local_points @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) + np.asarray(self.base, float)

    def footprint(self) -> np.ndarray:
        """4 x 2 corners in the xy plane, counter-clockwise."""
        sx, sy, _ = self.size
        a = math.radians(self.yaw_deg)
        c, s = math.cos(a), math.sin(a)
        loc = np.array([[-sx / 2, -sy / 2], [sx / 2, -sy / 2], [sx / 2, sy / 2], [-sx / 2, sy / 2]])
        return loc @ np.array([[c, s], [-s, c]]) + np.asarray(self.base[:2], float)

    def corners(self) -> np.ndarray:
        fp = self.footprint()
        z0 = float(self.base[2])
        return np.vstack([np.c_[fp, np.full(4, z0)], np.c_[fp, np.full(4, z0 + self.size[2])]])

    def centre(self) -> np.ndarray:
        return np.array([self.base[0], self.base[1], self.base[2] + self.size[2] / 2], float)

    @property
    def top(self) -> float:
        return float(self.base[2]) + float(self.size[2])


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; counter-clockwise in a y-up sense (signed area > 0), no repeated end point."""
    pts = sorted(set(map(tuple, np.round(np.asarray(points, float), 9))))
    if len(pts) <= 2:
        return np.asarray(pts, float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], float)


def polygon_area(poly) -> float:
    """Signed shoelace area (> 0 counter-clockwise in a y-up sense)."""
    p = [(float(x), float(y)) for x, y in poly]
    n = len(p)
    if n < 3:
        return 0.0
    return 0.5 * sum(p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1] for i in range(n))


def clip_polygon(subject, clip) -> np.ndarray:
    """Sutherland-Hodgman: the part of `subject` inside the convex polygon `clip` (either orientation)."""
    cl = [(float(x), float(y)) for x, y in clip]
    out = [(float(x), float(y)) for x, y in subject]
    if len(cl) < 3 or len(out) < 3:
        return np.zeros((0, 2))
    sgn = 1.0 if polygon_area(cl) > 0 else -1.0
    for i in range(len(cl)):
        ax, ay = cl[i]
        bx, by = cl[(i + 1) % len(cl)]
        ex, ey = bx - ax, by - ay
        inp, out = out, []
        if not inp:
            break
        prev = inp[-1]
        sp = sgn * (ex * (prev[1] - ay) - ey * (prev[0] - ax))
        for cur in inp:
            sc = sgn * (ex * (cur[1] - ay) - ey * (cur[0] - ax))
            if sc >= 0:
                if sp < 0:
                    f = sp / (sp - sc)
                    out.append((prev[0] + (cur[0] - prev[0]) * f, prev[1] + (cur[1] - prev[1]) * f))
                out.append(cur)
            elif sp >= 0:
                f = sp / (sp - sc)
                out.append((prev[0] + (cur[0] - prev[0]) * f, prev[1] + (cur[1] - prev[1]) * f))
            prev, sp = cur, sc
    return np.asarray(out, float) if out else np.zeros((0, 2))


def overlap_area(a: np.ndarray, b: np.ndarray) -> float:
    """Area of the intersection of two convex polygons."""
    return abs(polygon_area(clip_polygon(a, b)))


def separated(p: np.ndarray, q: np.ndarray, clearance: float = 0.0) -> bool:
    """Separating-axis test for two convex polygons: True if some edge normal separates them by >= clearance."""
    for poly in (p, q):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            axis = np.array([-e[1], e[0]], float)
            nn = np.linalg.norm(axis)
            if nn == 0:
                continue
            axis /= nn
            pa, qa = p @ axis, q @ axis
            if pa.max() + clearance <= qa.min() + 1e-12 or qa.max() + clearance <= pa.min() + 1e-12:
                return True
    return False


_AXES: dict = {}


def project(cam: R.Camera, pts: np.ndarray) -> np.ndarray:
    """cam.project(pts) (pinhole, as rendered) with the camera axes cached: the generator projects ~10^5 boxes."""
    key = (cam.name, cam.position, cam.heading_deg, cam.pitch_down_deg, cam.roll_deg)
    if key not in _AXES:
        right, up, fwd = cam.axes()
        _AXES[key] = (np.asarray(cam.position, float), np.stack([right, -up, fwd], 1))
    pos, m = _AXES[key]
    d = (np.atleast_2d(np.asarray(pts, float)) - pos) @ m
    if np.any(d[:, 2] <= 0):
        raise ValueError(f"{cam.name}: point behind the camera")
    return np.c_[cam.width / 2 + cam.f_px * d[:, 0] / d[:, 2], cam.height / 2 + cam.f_px * d[:, 1] / d[:, 2]]


def projected_hull(cam: R.Camera, pts: np.ndarray) -> np.ndarray:
    return convex_hull(project(cam, pts))


def sphere_directions(n: int = 2000) -> np.ndarray:
    """n Fibonacci-sphere unit vectors plus the 6 axis directions (so axis extremes, i.e. the bounding box, are exact)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    d = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    return np.vstack([np.eye(3), -np.eye(3), d])


def support_points(vertices: np.ndarray, n_dirs: int = 2000) -> np.ndarray:
    """The vertices that are extreme along any of sphere_directions(n_dirs): a subset of the convex-hull vertices whose
    hull matches the model's within R (1 - cos d) for direction spacing d (about 0.3 mm on a 10 cm model at 2000).
    Perspective maps convex sets to convex sets, so their projected hull is the model's projected hull."""
    v = np.asarray(vertices, float)
    idx = np.unique(np.argmax(v @ sphere_directions(n_dirs).T, axis=0))
    return v[idx]


def coverage(cam: R.Camera, target: Body, occ: Body) -> float:
    """Share of the target's projected convex hull covered by the occluder's projected hull (pinhole, as rendered).
    1 means hidden (the occluder stands wholly in front: see place_occluder); 0 means the hulls do not overlap."""
    t = projected_hull(cam, target.shape_points())
    at = abs(polygon_area(t))
    return 0.0 if at == 0 else min(1.0, overlap_area(t, projected_hull(cam, occ.shape_points())) / at)


def _depths(cam: R.Camera, pts: np.ndarray) -> np.ndarray:
    _, _, fwd = cam.axes()
    return (np.asarray(pts, float) - np.asarray(cam.position)) @ fwd


def may_occlude(cam: R.Camera, front: Body, back: Body, min_px2: float = 1.0) -> bool:
    """Conservative: `front` may hide part of `back` at this camera unless their projected box hulls are disjoint
    (overlap below min_px2) or `front` lies wholly behind `back` in camera depth."""
    if overlap_area(projected_hull(cam, front.shape_points()), projected_hull(cam, back.shape_points())) < min_px2:
        return False
    return not _depths(cam, front.shape_points()).min() > _depths(cam, back.shape_points()).max()


def sightline_x(cam: R.Camera, point: np.ndarray, y: float) -> float:
    """x where the line from the camera centre to `point` crosses the depth plane y."""
    c = np.asarray(cam.position, float)
    p = np.asarray(point, float)
    t = (y - c[1]) / (p[1] - c[1])
    return float(c[0] + t * (p[0] - c[0]))


# ---------------------------------------------------------------- context

@dataclass
class Context:
    config: dict
    config_path: str | None
    geometry: dict
    positions: dict
    rig: R.Rig
    comps: dict
    provenance: dict

    @property
    def cams(self) -> list[R.Camera]:
        return self.rig.cameras

    @property
    def start_cam(self) -> R.Camera:
        return self.rig.camera(self.positions.get("start_stop", "P3"))

    hulls: dict = field(default_factory=dict)

    def item(self, obj: int | str) -> dict:
        return self.config["objects"]["items"][str(int(obj))]

    def rest(self, name: str) -> np.ndarray:
        return np.asarray(self.config["rest_poses"][name], float)

    def local_points(self, obj: int, rest: str) -> np.ndarray | None:
        """The model's hull points in a body frame (rest-rotated, bounding box centred in x/y, bottom at z = 0, metres),
        exactly as the renderer centres the full mesh (sim/isaac_batch.py asset_local); None without hull data."""
        pts = self.hulls.get(str(int(obj)))
        if pts is None:
            return None
        rm = self.rest(rest)
        item = self.item(obj)
        lo, size = np.asarray(item["min_mm"], float), np.asarray(item["size_mm"], float)
        corners = np.array([[lo[i] + (size[i] if (k >> i) & 1 else 0.0) for i in range(3)] for k in range(8)]) @ rm.T
        cmin, cmax = corners.min(0), corners.max(0)
        off = np.array([(cmin[0] + cmax[0]) / 2, (cmin[1] + cmax[1]) / 2, cmin[2]])
        return (np.asarray(pts, float) @ rm.T - off) / 1000.0


def compartments(geometry: Mapping) -> dict:
    s = geometry["shelf"]
    c3, c4 = s["c3_inner"], s["c4_inner"]
    return {"c3": {"x": (-c3["half_width"], c3["half_width"]), "y": (0.0, c3["depth"]), "z": (0.0, c3["clear_height"])},
            "c4": {"x": (-c4["half_width"], c4["half_width"]), "y": (0.0, c4["depth"]), "z": (c4["z_min"], c4["z_max"])}}


def load_config(config: str | Path | Mapping = DEFAULT_CONFIG_PATH) -> tuple[dict, str | None, str]:
    """(config dict, path or None, sha256). The sha256 is of the file bytes, or of canonical JSON for a dict."""
    if isinstance(config, Mapping):
        cfg, path, sha = dict(config), None, sha256_json(config)
    else:
        p = _repo_path(config)
        cfg, path, sha = json.loads(p.read_text()), str(config), sha256_file(p)
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA!r}, got {cfg.get('schema')!r}")
    lo, hi = cfg["seeds"]["dev"]
    if (lo, hi) != DEV_SEED_RANGE or int(cfg["seeds"]["test_reserved_min"]) != TEST_SEED_MIN:
        raise ValueError(f"config seed ranges {cfg['seeds']} disagree with the generator's dev {DEV_SEED_RANGE} / test "
                         f">= {TEST_SEED_MIN}; changing them is a protocol decision")
    return cfg, path, sha


def load_context(config: str | Path | Mapping = DEFAULT_CONFIG_PATH) -> Context:
    cfg, cfg_path, cfg_sha = load_config(config)
    gpath, ppath = _repo_path(cfg["geometry"]), _repo_path(cfg["positions"])
    geometry = R.load_geometry(gpath)
    positions = json.loads(ppath.read_text())
    rig = R.load_rig(gpath, ppath, occluder=None)
    hulls, hull_prov = {}, None
    hc = cfg["objects"].get("hulls")
    if hc:
        hpath = _repo_path(hc["path"])
        sha = sha256_file(hpath)
        if sha != hc["sha256"]:
            raise ValueError(f"{hc['path']}: sha256 {sha} != the config's {hc['sha256']}")
        hulls = json.loads(hpath.read_text())["items"]
        hull_prov = {"path": hc["path"], "sha256": sha}
    prov = {
        "generator": {"path": "sim/arrange.py", "version": GENERATOR_VERSION, "sha256": sha256_file(THIS_FILE)},
        "rig_py": {"path": "sim/rig.py", "sha256": sha256_file(RIG_FILE)},
        "config": {"path": cfg_path, "name": cfg.get("name"), "sha256": cfg_sha},
        "geometry": {"path": cfg["geometry"], "sha256": sha256_file(gpath)},
        "positions": {"path": cfg["positions"], "sha256": sha256_file(ppath)},
        "hulls": hull_prov,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    return Context(cfg, cfg_path, geometry, positions, rig, compartments(geometry), prov, hulls)


def rest_extents(item: Mapping, rest: np.ndarray) -> tuple[float, float, float]:
    """Bounding-box extents (m) of a model after its rest rotation (models_info min/size in mm)."""
    lo, size = np.asarray(item["min_mm"], float), np.asarray(item["size_mm"], float)
    corners = np.array([[lo[i] + (size[i] if (k >> i) & 1 else 0.0) for i in range(3)] for k in range(8)])
    r = corners @ np.asarray(rest, float).T
    ext = (r.max(0) - r.min(0)) / 1000.0
    return float(ext[0]), float(ext[1]), float(ext[2])


# ---------------------------------------------------------------- placement rules

def inside(ctx: Context, body: Body, comp: str, front: float | None = None) -> bool:
    """Footprint inside the compartment's walls with the configured clearances, standing on its floor, below its
    ceiling clearance."""
    c, cl = ctx.comps[comp], ctx.config["clearances_m"]
    fp = body.footprint()
    front = cl["front"] if front is None else front
    if abs(float(body.base[2]) - c["z"][0]) > 1e-9:
        return False
    if body.top > c["z"][1] - cl["ceiling"] + 1e-12:
        return False
    return bool(fp[:, 0].min() >= c["x"][0] + cl["wall"] and fp[:, 0].max() <= c["x"][1] - cl["wall"]
                and fp[:, 1].min() >= c["y"][0] + front and fp[:, 1].max() <= c["y"][1] - cl["back"])


def seen_whole(ctx: Context, cam: R.Camera, body: Body) -> bool:
    """Every point of the body's convex hull (shape_points) has a sight line through the #3 opening (front plane) with
    opening_margin_m to spare and projects frame_margin_px inside the frame: the stop sees the object whole."""
    tc = ctx.config["target"]
    m = tc["opening_margin_m"]
    c3 = ctx.comps["c3"]
    c = np.asarray(cam.position, float)
    pts = body.shape_points()
    q = c + ((0.0 - c[1]) / (pts[:, 1] - c[1]))[:, None] * (pts - c)
    if not (np.all(q[:, 0] >= c3["x"][0] + m) and np.all(q[:, 0] <= c3["x"][1] - m)
            and np.all(q[:, 2] >= c3["z"][0] + m) and np.all(q[:, 2] <= c3["z"][1] - m)):
        return False
    uv = project(cam, pts)
    return float(min(uv[:, 0].min(), uv[:, 1].min(), cam.width - uv[:, 0].max(), cam.height - uv[:, 1].max())) >= tc["frame_margin_px"]


def _clear_of(fp: np.ndarray, others: Sequence[np.ndarray], clearance: float) -> bool:
    return all(separated(fp, o, clearance) for o in others)


def place_target(ctx: Context, d: Draw, obj: int) -> dict | None:
    """A position and yaw for target model `obj` in #3 where every stop sees it whole (seen_whole)."""
    tc = ctx.config["target"]
    item = ctx.item(obj)
    size = rest_extents(item, ctx.rest("upright"))
    if size[2] > tc["max_height_m"]:
        raise ValueError(f"target object {obj} ({item['name']}) is {size[2]:.3f} m tall > max_height_m")
    pts = ctx.local_points(obj, "upright")
    for _ in range(int(tc["tries"])):
        body = Body(size, (d.u(*tc["region_m"]["x"]), d.u(*tc["region_m"]["y"]), 0.0), d.u(0.0, 360.0), pts)
        if not inside(ctx, body, "c3") or body.footprint()[:, 1].min() < tc["min_front_y_m"]:
            continue
        if all(seen_whole(ctx, cam, body) for cam in ctx.cams):
            return {"obj_id": obj, "item": item, "rest": "upright", "body": body}
    return None


def _occluder_ok(ctx: Context, body: Body, target: Body) -> bool:
    cl = ctx.config["clearances_m"]
    return (inside(ctx, body, "c3", front=cl["occluder_front"])
            and separated(body.footprint(), target.footprint(), 0.0))


def _gap(ctx: Context, d: Draw, spec: Mapping, target: Body, depth: float) -> float | None:
    """Distance from the target's front to the occluder's back face, drawn from the family's gap_m range clipped to
    the room in front of the target (the occluder's front face stays occluder_front behind the front plane)."""
    room = float(target.footprint()[:, 1].min()) - ctx.config["clearances_m"]["occluder_front"] - depth
    lo, hi = float(spec["gap_m"][0]), min(float(spec["gap_m"][1]), room)
    return None if hi < lo else d.u(lo, hi)


def _board(ctx: Context, d: Draw, spec: Mapping, target: Body) -> dict | None:
    o = R.occluder_spec(ctx.geometry, spec["geometry_occluder"])
    gap = _gap(ctx, d, spec, target, o.thickness)
    if gap is None:
        return None
    shift = d.u(*spec["shift_m"])
    y_back = float(target.footprint()[:, 1].min()) - gap
    y_front = y_back - o.thickness
    yc = (y_front + y_back) / 2
    x = sightline_x(ctx.start_cam, target.centre(), yc) + shift
    body = Body((o.width, o.thickness, o.height), (x, yc, 0.0), 0.0)
    return {"kind": "box", "material": "occluder", "geometry_occluder": o.name, "body": body,
            "gap_m": gap, "shift_m": shift}


def _white_box(ctx: Context, d: Draw, spec: Mapping, target: Body, cov_min: float) -> dict | None:
    cl = ctx.config["clearances_m"]
    gap = _gap(ctx, d, spec, target, 0.02)
    if gap is None:
        return None
    depth, shift = d.u(*spec["depth_m"]), d.u(*spec["shift_m"])
    y_back = float(target.footprint()[:, 1].min()) - gap
    y_front = max(y_back - depth, cl["occluder_front"])
    if y_back - y_front < 0.02:
        return None
    yc = (y_front + y_back) / 2
    # the smallest front face that hides the target's convex hull from the start stop: back-project its hull points
    # along their sight lines onto the plane y = y_front, then pad (so the box's P3 image footprint ~ the target's)
    c = np.asarray(ctx.start_cam.position, float)
    pts = target.shape_points()
    q = c + ((y_front - c[1]) / (pts[:, 1] - c[1]))[:, None] * (pts - c)
    pad_w, pad_h = d.u(*spec["pad_m"]), d.u(*spec["pad_m"])
    w = float(q[:, 0].max() - q[:, 0].min()) + 2 * pad_w
    h = float(q[:, 2].max()) + pad_h
    if h > ctx.comps["c3"]["z"][1] - cl["ceiling"]:
        return None
    x = float(q[:, 0].max() + q[:, 0].min()) / 2 + max(-pad_w, min(pad_w, shift))
    return {"kind": "box", "material": spec.get("material", "white_box"),
            "body": Body((w, y_back - y_front, h), (x, yc, 0.0), 0.0), "gap_m": gap, "shift_m": shift}


def _ycb_box(ctx: Context, d: Draw, spec: Mapping, target_obj: int, target: Body) -> dict | None:
    choices = [c for c in spec["choices"] if int(c[0]) != target_obj]
    if not choices:
        return None
    obj, rest = d.choice(choices)
    item = ctx.item(obj)
    w, dep, h = rest_extents(item, ctx.rest(rest))
    gap = _gap(ctx, d, spec, target, dep)
    if gap is None:
        return None
    shift = d.u(*spec["shift_m"])
    y_back = float(target.footprint()[:, 1].min()) - gap
    yc = y_back - dep / 2
    x = sightline_x(ctx.start_cam, target.centre(), yc) + shift
    return {"kind": "ycbv", "obj_id": int(obj), "item": item, "rest": rest, "material": None,
            "body": Body((w, dep, h), (x, yc, 0.0), 0.0, ctx.local_points(obj, rest)), "gap_m": gap, "shift_m": shift}


def place_occluder(ctx: Context, d: Draw, target: dict, fam: str) -> dict | None:
    """A placed pose for an occluder of family `fam` in front of the target: P3 coverage >= p3_coverage_min and
    the family's stop rule (stop_rule_ok)."""
    oc = ctx.config["occluder"]
    cov_min = float(oc["p3_coverage_min"])
    tb = target["body"]
    spec = oc["families"][fam]
    ind = bool(spec.get("predeclared_indistinguishable", False))
    for _ in range(int(oc["tries_per_family"])):
        if spec["kind"] == "board":
            cand = _board(ctx, d, spec, tb)
        elif spec["kind"] == "white_box":
            cand = _white_box(ctx, d, spec, tb, cov_min)
        elif spec["kind"] == "ycbv":
            cand = _ycb_box(ctx, d, spec, target["obj_id"], tb)
        else:
            raise ValueError(f"unknown occluder kind {spec['kind']!r}")
        if cand is None or not _occluder_ok(ctx, cand["body"], tb):
            continue
        cov = coverage(ctx.start_cam, tb, cand["body"])
        if cov < cov_min:
            continue
        covs = {c.name: coverage(c, tb, cand["body"]) for c in ctx.cams}
        if not stop_rule_ok(covs, ind, oc):
            continue
        cand.update(family=fam, p3_coverage=cov, predeclared_indistinguishable=ind)
        return cand
    return None


def stop_rule_ok(covs: Mapping[str, float], indistinguishable: bool, oc: Mapping) -> bool:
    """The family's rule over all stops (convex-hull coverage of the target by the placed occluder, per stop).
    Indistinguishable families (IND28) must cover the target at EVERY stop (UNKNOWN is the right answer there);
    every other family must leave at least min_clear_stops stops with no overlap at all, so the occluded episode can
    be resolved by another view and the removed episode can reach the k = 2 clear-miss rule."""
    if indistinguishable:
        return min(covs.values()) >= float(oc["p3_coverage_min"])
    return sum(v <= float(oc["clear_coverage_max"]) for v in covs.values()) >= int(oc["min_clear_stops"])


def place_distractors(ctx: Context, d: Draw, target: dict, occ: dict) -> list[dict] | None:
    dc = ctx.config["distractors"]
    clr = ctx.config["clearances_m"]["between_objects"]
    n = d.integer(*dc["count"])
    used = {target["obj_id"]} | ({occ["obj_id"]} if occ["kind"] == "ycbv" else set())
    fps = [target["body"].footprint(), occ["body"].footprint()]       # #3 footprints (placed occluder included)
    fps4: list[np.ndarray] = []
    out: list[dict] = []
    for _ in range(n):
        for _ in range(int(dc["tries_each"])):
            comp = "c3" if d.u() < float(dc["p_c3"]) else "c4"
            pool = [int(o) for o in dc[f"{comp}_objects"] if int(o) not in used]
            if not pool:
                continue
            obj = d.choice(pool)
            item = ctx.item(obj)
            size = rest_extents(item, ctx.rest("upright"))
            c = ctx.comps[comp]
            body = Body(size, (d.u(*c["x"]), d.u(*c["y"]), c["z"][0]), d.u(0.0, 360.0), ctx.local_points(obj, "upright"))
            if not inside(ctx, body, comp):
                continue
            if not _clear_of(body.footprint(), fps if comp == "c3" else fps4, clr):
                continue
            if comp == "c3" and any(may_occlude(cam, body, target["body"]) for cam in ctx.cams):
                continue
            out.append({"obj_id": obj, "item": item, "rest": "upright", "body": body, "compartment": comp})
            used.add(obj)
            (fps if comp == "c3" else fps4).append(body.footprint())
            break
    return out if len(out) >= int(dc["count"][0]) else None


def target_window(cam: R.Camera, target: Body, context: float) -> np.ndarray:
    """The bounding box of the target's projected hull, grown by `context` x its size on every side (perception/evidence.py
    Params.context = 1.0 is the DINOv2 window), as a 4-point polygon in pixels."""
    uv = project(cam, target.shape_points())
    (u0, v0), (u1, v1) = uv.min(0), uv.max(0)
    w, h = u1 - u0, v1 - v0
    u0, v0, u1, v1 = u0 - context * w, v0 - context * h, u1 + context * w, v1 + context * h
    return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], float)


def park_occluder(ctx: Context, d: Draw, occ: dict, target: Body, others: Sequence[Body]) -> tuple[Body, str] | None:
    """A pose for the occluder that occludes nothing: inside a compartment, footprint clear of every object, hiding no
    object (target, distractors) from any stop, and outside the target's context window at every stop (so moving it
    later does not change the target's own background between phases)."""
    pc = ctx.config["occluder"]["parked"]
    clr = ctx.config["clearances_m"]["between_objects"]
    size, pts = occ["body"].size, occ["body"].local_points
    objs = [target, *others]
    windows = [target_window(cam, target, float(pc["target_window_context"])) for cam in ctx.cams]
    for comp in pc["compartments"]:
        c = ctx.comps[comp]
        same = [b.footprint() for b in objs if abs(b.base[2] - c["z"][0]) < 1e-9]
        for _ in range(int(pc["tries"])):
            body = Body(size, (d.u(*c["x"]), d.u(*c["y"]), c["z"][0]), 0.0, pts)
            if not inside(ctx, body, comp) or not _clear_of(body.footprint(), same, clr):
                continue
            if any(overlap_area(projected_hull(cam, body.shape_points()), win) >= 1.0 for cam, win in zip(ctx.cams, windows)):
                continue
            if any(may_occlude(cam, body, b) for cam in ctx.cams for b in objs):
                continue
            return body, comp
    return None


# ---------------------------------------------------------------- records

def _r(v: Any, nd: int = 6) -> Any:
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, nd) for x in v]
    return round(float(v), nd)


def _pose(body: Body, comp: str) -> dict:
    return {"base": _r(body.base), "yaw_deg": _r(body.yaw_deg, 4), "compartment": comp, "footprint": _r(body.footprint())}


def _object(name: str, role: str, label: str, kind: str, body: Body, comp: str, *, obj_id: int | None = None,
            item: Mapping | None = None, rest: str | None = None, rest_matrix: Any = None, material: str | None = None,
            **extra: Any) -> dict:
    rec = {"name": name, "role": role, "label": label, "kind": kind, "size_m": _r(body.size)}
    if kind == "ycbv":
        rec.update(obj_id=int(obj_id), ycb=item["ycb"], ply=item["ply"], rest=rest,
                   rest_matrix=[[int(v) for v in row] for row in np.asarray(rest_matrix)])
    if material is not None:
        rec["material"] = material
    rec["pose"] = _pose(body, comp)
    rec.update(extra)
    return rec


def body_of(rec: Mapping, pose: str = "pose", ctx: Context | None = None) -> Body:
    """The Body of an object record (pose = 'pose', or 'placed' / 'parked' for the occluder); with ctx, a YCB-Video
    object gets its hull points (the shape the generator tested)."""
    p = rec["pose"] if pose == "pose" else rec["poses"][pose]
    pts = ctx.local_points(rec["obj_id"], rec["rest"]) if ctx is not None and rec.get("kind") == "ycbv" else None
    if ctx is not None and rec.get("kind") == "mesh" and "local_points" in rec:
        pts = np.asarray(rec["local_points"], float)
    return Body(tuple(rec["size_m"]), tuple(p["base"]), float(p["yaw_deg"]), pts)


def _phases(ctx: Context) -> list[dict]:
    lights = list(LIGHTINGS)
    return [{"phase": ph, "role": role, "condition": cond, "target_present": PHASE_STATE[ph][0],
             "occluder_pose": PHASE_STATE[ph][1], "lightings": lights} for ph, role, cond in PHASES]


def _analytic(ctx: Context, target: Body, placed: Body, parked: Body) -> dict:
    return {"note": "convex-hull coverage of the target by the occluder through the pinhole cameras; NOT visibility labels (those come from the render)",
            "placed_coverage": {c.name: round(coverage(c, target, placed), 4) for c in ctx.cams},
            "parked_coverage": {c.name: round(coverage(c, target, parked), 4) for c in ctx.cams}}


def _scene(ctx: Context, rig: R.Rig, keep_mug: bool) -> dict:
    spec = R.export_scene_spec(rig)
    spec["boxes"] = [b for b in spec["boxes"] if b["role"] != "occluder"]
    if not keep_mug:
        spec["meshes"], spec["mug"] = [], None
    spec["occluder"] = None
    spec["materials"].update({k: {"rgba": v["rgba"]} for k, v in ctx.config.get("materials", {}).items()})
    spec["provenance"]["geometry_file"] = ctx.config["geometry"]
    spec["provenance"]["positions_file"] = ctx.config["positions"]
    return spec


def _finish(ctx: Context, arr: dict, rig: R.Rig, keep_mug: bool) -> dict:
    arr["phases"] = _phases(ctx)
    arr["lightings"] = list(LIGHTINGS)
    arr["stops"] = [c.name for c in ctx.cams]
    arr["start_stop"] = ctx.positions.get("start_stop", "P3")
    arr["provenance"] = ctx.provenance
    arr["scene"] = _scene(ctx, rig, keep_mug)
    arr["arrangement_sha256"] = sha256_json({k: v for k, v in arr.items() if k != "arrangement_sha256"})
    return arr


def generate(seed: int, config: str | Path | Mapping = DEFAULT_CONFIG_PATH, ctx: Context | None = None) -> dict:
    """The arrangement for a dev seed (deterministic in (seed, config, geometry, positions, numpy's PCG64))."""
    seed = check_dev_seed(seed)
    ctx = ctx or load_context(config)
    d = Draw([seed])
    # the target model and the occluder family are drawn once, so their frequencies follow the config; the next model
    # or family in the drawn order is used only if a pair admits no valid arrangement (recorded in 'draws')
    objs = [int(k) for k in d.weighted_order({str(o): 1.0 for o in ctx.config["target"]["objects"]})]
    fams = d.weighted_order(ctx.config["occluder"]["family_weights"])
    n_att = int(ctx.config["occluder"]["attempts_per_family"])
    for oi, obj in enumerate(objs):
        for fi, fam in enumerate(fams):
            for attempt in range(n_att):
                target = place_target(ctx, d, obj)
                if target is None:
                    continue
                occ = place_occluder(ctx, d, target, fam)
                if occ is None:
                    continue
                dis = place_distractors(ctx, d, target, occ)
                if dis is None:
                    continue
                parked = park_occluder(ctx, d, occ, target["body"], [x["body"] for x in dis])
                if parked is None:
                    continue
                draws = {"target_order": objs, "family_order": fams, "target_rank": oi, "family_rank": fi,
                         "attempt": attempt}
                return _assemble(ctx, seed, target, occ, dis, parked, draws)
    raise RuntimeError(f"seed {seed}: no valid arrangement for any target model / occluder family")


def _assemble(ctx: Context, seed: int, target: dict, occ: dict, dis: list[dict], parked: tuple[Body, str],
              draws: dict) -> dict:
    tb = target["body"]
    t_rec = _object("target", "target", target["item"]["name"], "ycbv", tb, "c3", obj_id=target["obj_id"],
                    item=target["item"], rest=target["rest"], rest_matrix=ctx.rest(target["rest"]))
    o_kind = occ["kind"]
    o_rec = _object("occluder", "occluder", occ["family"] if o_kind == "box" else occ["item"]["name"], o_kind,
                    occ["body"], "c3", obj_id=occ.get("obj_id"), item=occ.get("item"), rest=occ.get("rest"),
                    rest_matrix=None if o_kind != "ycbv" else ctx.rest(occ["rest"]), material=occ.get("material"),
                    family=occ["family"], gap_m=_r(occ["gap_m"]), shift_m=_r(occ["shift_m"]),
                    p3_coverage=_r(occ["p3_coverage"], 4))
    if "geometry_occluder" in occ:
        o_rec["geometry_occluder"] = occ["geometry_occluder"]
    o_rec["poses"] = {"placed": o_rec.pop("pose"), "parked": _pose(parked[0], parked[1])}
    d_recs = [_object(f"distractor_{i}", "distractor", x["item"]["name"], "ycbv", x["body"], x["compartment"],
                      obj_id=x["obj_id"], item=x["item"], rest=x["rest"], rest_matrix=ctx.rest(x["rest"]))
              for i, x in enumerate(dis)]
    arr = {"schema": ARRANGEMENT_SCHEMA, "arrangement_id": arrangement_id(seed), "seed": seed, "split": "dev",
           "kind": "random", "environment": ctx.config["environment"], "target_id": t_rec["label"],
           "predeclared_indistinguishable": bool(occ["predeclared_indistinguishable"]),
           "objects": {"target": t_rec, "occluder": o_rec, "distractors": d_recs},
           "analytic": _analytic(ctx, tb, occ["body"], parked[0]), "draws": draws}
    return _finish(ctx, arr, ctx.rig, keep_mug=False)


def design_arrangement(index: int, config: str | Path | Mapping = DEFAULT_CONFIG_PATH, ctx: Context | None = None) -> dict:
    """Design case `index` (0..3): option A, measured mug at design z 0.24 (handle toward the camera), design occluder
    placed as in configs/rig_geometry.json, parked in #4 at the configured spot. No distractors."""
    ctx = ctx or load_context(config)
    dc = ctx.config["design_cases"]
    case = dc["cases"][index]
    rig = R.load_rig(_repo_path(ctx.config["geometry"]), _repo_path(ctx.config["positions"]), occluder=case["occluder"],
                     mug_z=dc["mug_z"], mug_x=dc["mug_x"], handle=dc["handle"])
    ob = rig.occluder_box()
    size = tuple(2 * float(h) for h in ob.half_size)
    placed = Body(size, (ob.centre[0], ob.centre[1], ob.centre[2] - size[2] / 2), 0.0)
    verts = np.vstack([m.world_vertices() for m in rig.meshes()])
    lo, hi = verts.min(0), verts.max(0)
    base = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    mug = Body(tuple(hi - lo), tuple(base), 0.0, support_points(verts) - base)
    pk = dc["parked"]
    parked = Body(size, (pk["x"], pk["y"], ctx.comps[pk["compartment"]]["z"][0]), 0.0)
    ctxw = float(ctx.config["occluder"]["parked"]["target_window_context"])
    if (not inside(ctx, parked, pk["compartment"]) or any(may_occlude(c, parked, mug) for c in ctx.cams)
            or any(overlap_area(projected_hull(c, parked.shape_points()), target_window(c, mug, ctxw)) >= 1.0 for c in ctx.cams)):
        raise ValueError(f"design parking spot {pk} is not valid for {case['occluder']}")
    t_rec = _object("target", "target", dc["target_name"], "mesh", mug, "c3", meshes=[m.name for m in rig.meshes()],
                    mug={"x": dc["mug_x"], "z_design": dc["mug_z"], "handle": dc["handle"]},
                    local_points=_r(mug.local_points))
    o_rec = _object("occluder", "occluder", case["occluder"], "box", placed, "c3", material="occluder",
                    family=case["occluder"], geometry_occluder=case["occluder"],
                    p3_coverage=_r(coverage(ctx.start_cam, mug, placed), 4),
                    intended_pattern=rig.occluder.intended_pattern)
    o_rec["poses"] = {"placed": o_rec.pop("pose"), "parked": _pose(parked, pk["compartment"])}
    arr = {"schema": ARRANGEMENT_SCHEMA, "arrangement_id": case["arrangement_id"], "seed": case["seed"], "split": "dev",
           "kind": "design", "environment": ctx.config["environment"], "target_id": dc["target_name"],
           "predeclared_indistinguishable": bool(case.get("predeclared_indistinguishable", False)),
           "objects": {"target": t_rec, "occluder": o_rec, "distractors": []},
           "analytic": _analytic(ctx, mug, placed, parked), "design_note": dc["note"]}
    return _finish(ctx, arr, rig, keep_mug=True)


def n_design_cases(config: str | Path | Mapping = DEFAULT_CONFIG_PATH) -> int:
    return len(load_config(config)[0]["design_cases"]["cases"])


# ---------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.arrange", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.relative_to(R.REPO_ROOT)))
    ap.add_argument("--seeds", default="", help="dev seeds, e.g. '0-5' or '0,3,7' (0-9999 only)")
    ap.add_argument("--design", action="store_true", help="also write the design cases DES0..")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        seeds = parse_seeds(a.seeds) if a.seeds else []
    except (SeedError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    ctx = load_context(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arrs = [design_arrangement(i, ctx=ctx) for i in range(len(ctx.config["design_cases"]["cases"]))] if a.design else []
    arrs += [generate(s, ctx=ctx) for s in seeds]
    index = []
    for arr in arrs:
        p = out / f"{arr['arrangement_id']}.json"
        p.write_text(json.dumps(arr, indent=1) + "\n")
        index.append({"arrangement_id": arr["arrangement_id"], "seed": arr["seed"], "kind": arr["kind"],
                      "target_id": arr["target_id"], "occluder_family": arr["objects"]["occluder"]["family"],
                      "n_distractors": len(arr["objects"]["distractors"]), "file": p.name, "sha256": sha256_file(p),
                      "arrangement_sha256": arr["arrangement_sha256"]})
        print(f"{arr['arrangement_id']}: seed {arr['seed']}, target {arr['target_id']}, occluder "
              f"{arr['objects']['occluder']['family']} (P3 hull coverage {arr['objects']['occluder']['p3_coverage']}), "
              f"{len(arr['objects']['distractors'])} distractors")
    (out / "index.json").write_text(json.dumps({"schema": INDEX_SCHEMA, "provenance": ctx.provenance,
                                                "arrangements": index}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
