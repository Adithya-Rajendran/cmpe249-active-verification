"""Seeded arrangement generator for the simulated shelf: slot list + (seed, attempt) -> arrangement. numpy only.

Why: docs/SIMULATION_STUDY.md moves the course study into simulation, where arrangements come from a seed and a fixed
generator instead of being hand-built. One arrangement mirrors one physical arrangement of capture/run_arrangement.py:
  - one study target in compartment #3, at a random lateral position, depth and yaw where every stop sees it whole;
  - 2-5 distractor objects (other YCB-Video models) in #3 and #4, with non-overlapping footprints, never occluding the
    target from any stop;
  - one occluder of one family (design boards S8 / M16 / IND28, a plain white box over the target's footprint, i.e.
    the pilot-2 failure case, or a YCB box object). It has a 'placed' pose in front of the target, chosen so the
    start stop P3 is mostly blocked, and a 'parked' pose that occludes nothing;
  - the four phases, in capture order (PHASES = capture/run_arrangement.py PHASES):
      reference  target in place, occluder parked       (the per-stop references, nominal camera poses)
      visible    scene unchanged                         (query, condition visible)
      occluded   occluder placed                         (query, condition occluded)
      removed    target removed, occluder as in occluded (query, condition removed: the matched layout)
    each phase at every stop under L1 and L2; the three query phases use the REVISIT camera poses (below).
The four option-A design cases (design_arrangement(0..3), seeds "design-0".."design-3") are fixed, predeclared dev
arrangements outside the 60 slots: the measured mug at design z 0.24 with pick_S8 / pick_M16+3 / pick_M16-5 / pick_IND28.

Quota slots (docs/SIM_PREREGISTRATION.md section 2). The occluder family of an arrangement is NOT drawn: slot_list(ctx,
split) expands the config's quota table (shelf dev: S8 12, M16 18 = 9 L + 9 R, IND28 6, white_box 12, ycb_box 12; test 20 /
30 / 10 / 20 / 20), permutes it with a seeded shuffle and gives slot i the primary seed primary_seed_start + i. Placement
failure retries deterministically within the SAME family: try (seed, attempt) for attempt 0..retry_max (attempt 0 is the
plain seed; 1..retry_max are the retries, retry_max = 20 in the config); after that the next unused seed of the family's
reserve list is tried the same way. A slot NEVER falls back to another family or M16 side (the old family_weights fallback
turned SD0001 of shelf_dev_v0 into IND28). Every failed try and replacement is logged in the arrangement's 'slot' record
and in the set-level generation log (generate_set).

M16 is asymmetric. SIDE: seen from the cameras (the rig frame, +x = image right at the start stop P3), the board's side is
the sign of its lateral displacement from the sight line P3 -> target centre, measured on the board's plane: R = board
centre displaced toward +x, L = toward -x, by a magnitude drawn from the config's side_shift_m. The covered stops then lie
toward that side of the arc: R hides P3 and P4 (pattern CCHHC), L hides P2 and P3 (CHHCC). The slot gives the side; a draw
is accepted only if side_rule_ok (config occluder.families.M16.side_rule) holds on the analytic hull coverage: the start
stop hidden, the adjacent stop on the board's side covered >= adjacent_min, the nearest opposite stop covered <=
opposite_near_max and every further opposite stop clear. DES1 (pick_M16+3) is side R, DES2 (pick_M16-5) side L.

Revisit noise (query captures only). Every arrangement records one bounded uniform draw per stop (revisit_cameras): the
camera moves by +-translation_m on each world axis and by +-rotation_deg about its own yaw (heading), pitch and roll axes,
seeded from the arrangement seed and the stop, and the same pose serves the visible / occluded / removed phases, both
lightings and every policy. A draw that would put the #3 outline or a tag outside the frame (sim/rig.py frame_margins,
fitted k1, min margin >= min_margin_px, the nominal check) is redrawn with attempt + 1; rejected draws are logged. The
renderer reads scene.revisit_cameras; references use scene.cameras (nominal).

Geometry is decided by projecting shapes through the pinhole cameras of the positions file (as rendered), so the
generator needs no meshes and no renderer: each model's convex hull (YCB-V hull points, configs/sim/ycbv_hulls.json;
the measured mug's mesh) for visibility, its rest-pose bounding box (models_info) for footprints. Both visibility tests
are conservative in the right direction: 'hidden at P3' means the occluder's projected hull covers the target's
projected hull (p3_coverage_min), and a 'clear' stop means the two hulls do not overlap at all. Every family except
IND28 must leave >= 2 clear stops (so another view resolves the occlusion and a removed episode can reach the k = 2
clear-miss rule); IND28 must cover every stop (UNKNOWN is the right answer, predeclared). The real visibility labels
always come from the rendered instance masks (sim/isaac_batch.py), never from this file. The analytic coverage is at the
NOMINAL poses; the revisit poses move the images by a few tens of pixels.

Integrity (docs/SIMULATION_STUDY.md): dev seeds are 0-9999 and are the only integers accepted. Seeds >= 100000 are
reserved for test ([open]) and refused (check_dev_seed): slot_list(ctx, 'test') works, but no test arrangement is ever
generated. arrangement_sha256 is the hash of the arrangement without its 'unhashed' block (the caller-spelled config path
and the Python version), so it depends only on (seed, attempt, family, side, config content, the code and input files whose
sha256 the provenance records, numpy's version). Random draws use only Generator.random() (PCG64 doubles), so the stream
does not depend on numpy's distribution algorithms.

    python -m sim.arrange --split dev --slots all|first-per-class|0-5,9 [--design] --out DIR
        # DIR/<ARR>.json; DIR/index/{index,slots,generation_log}.json (a subfolder, so DIR/*.json is only arrangements)
"""
from __future__ import annotations

import argparse
import dataclasses
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

GENERATOR_VERSION = "0.2.0"
ARRANGEMENT_SCHEMA = "cmpe249.sim-arrangement.v1"
CONFIG_SCHEMA = "cmpe249.sim-arrange-config.v1"
INDEX_SCHEMA = "cmpe249.sim-arrangement-index.v0"
SLOTS_SCHEMA = "cmpe249.sim-slots.v0"
GENERATION_LOG_SCHEMA = "cmpe249.sim-generation-log.v0"
REVISIT_SCHEMA = "cmpe249.sim-revisit.v0"
NON_ARRANGEMENT_SCHEMAS = (INDEX_SCHEMA, SLOTS_SCHEMA, GENERATION_LOG_SCHEMA)   # files in DIR/index/ (isaac_batch skips them)
FAMILIES = ("S8", "M16", "IND28", "white_box", "ycb_box")
SIDES = ("L", "R")
UNHASHED_KEYS = ("arrangement_sha256", "unhashed")
DEV_SEED_RANGE = (0, 9999)
TEST_SEED_MIN = 100000           # reserved for test ([open]); never generated, rendered, scored or viewed before the freeze
DEFAULT_CONFIG_PATH = R.REPO_ROOT / "configs" / "sim" / "arrange_shelf_v1.json"
THIS_FILE = Path(__file__).resolve()
RIG_FILE = Path(R.__file__).resolve()

# (phase, role, condition): capture/run_arrangement.py PHASES (tests/test_sim_arrange.py checks they agree)
PHASES = (("reference", "reference", "visible"), ("visible", "query", "visible"),
          ("occluded", "query", "occluded"), ("removed", "query", "removed"))
PHASE_STATE = {"reference": (True, "parked"), "visible": (True, "parked"),       # (target present, occluder pose)
               "occluded": (True, "placed"), "removed": (False, "placed")}
PHASE_CAMERA = {"reference": "nominal", "visible": "revisit", "occluded": "revisit", "removed": "revisit"}   # revisit noise: queries only
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

    def shuffle(self, seq: Sequence) -> list:
        """Fisher-Yates permutation of seq (a new list), driven only by index()."""
        out = list(seq)
        for i in range(len(out) - 1, 0, -1):
            j = self.index(i + 1)
            out[i], out[j] = out[j], out[i]
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
    provenance: dict            # hashed into arrangement_sha256: content hashes only
    unhashed: dict = field(default_factory=dict)       # caller-dependent metadata (config path as spelled, file hash, Python)

    @property
    def cams(self) -> list[R.Camera]:
        return self.rig.cameras

    @property
    def start_cam(self) -> R.Camera:
        return self.rig.camera(self.positions.get("start_stop", "P3"))

    @property
    def start_index(self) -> int:
        return [c.name for c in self.rig.cameras].index(self.positions.get("start_stop", "P3"))

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
    """(config dict, path or None, sha256). The sha256 is of the config's canonical JSON (sorted keys, so it ignores
    formatting and key order, for a file and an in-memory dict alike); the file's own byte hash goes to the unhashed
    metadata (load_context)."""
    if isinstance(config, Mapping):
        cfg, path = json.loads(json.dumps(config)), None
    else:
        p = _repo_path(config)
        cfg, path = json.loads(p.read_text()), str(config)
    sha = sha256_json(cfg)
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA!r}, got {cfg.get('schema')!r}")
    lo, hi = cfg["seeds"]["dev"]
    if (lo, hi) != DEV_SEED_RANGE or int(cfg["seeds"]["test_reserved_min"]) != TEST_SEED_MIN:
        raise ValueError(f"config seed ranges {cfg['seeds']} disagree with the generator's dev {DEV_SEED_RANGE} / test "
                         f">= {TEST_SEED_MIN}; changing them is a protocol decision")
    check_slot_config(cfg)
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
        "config": {"name": cfg.get("name"), "sha256": cfg_sha},
        "geometry": {"path": cfg["geometry"], "sha256": sha256_file(gpath)},
        "positions": {"path": cfg["positions"], "sha256": sha256_file(ppath)},
        "hulls": hull_prov,
        "numpy": np.__version__,
    }
    unhashed = {"config_path": cfg_path, "config_file_sha256": None if cfg_path is None else sha256_file(_repo_path(cfg_path)),
                "python": platform.python_version(),
                "note": "caller-dependent metadata; deliberately outside arrangement_sha256"}
    return Context(cfg, cfg_path, geometry, positions, rig, compartments(geometry), prov, unhashed, hulls)


# ---------------------------------------------------------------- quota slots

def _quota_entries(quota: Mapping) -> list[tuple[str, str | None]]:
    """The quota table as (family, side) entries in sorted key order (family, then side): the order must not depend on the
    dict's insertion order, so the permutation is a function of the config's content only."""
    out: list[tuple[str, str | None]] = []
    for fam in sorted(quota):
        q = quota[fam]
        if isinstance(q, Mapping):
            for side in sorted(q):
                out += [(fam, side)] * int(q[side])
        else:
            out += [(fam, None)] * int(q)
    return out


def check_slot_config(cfg: Mapping) -> None:
    """Quota families known, M16 (and only M16) has sides, primary and reserve seeds disjoint and inside the split's seed
    range (dev: 0-9999; test: >= test_reserved_min), reserve lists per family, retry_max a non-negative int."""
    sl = cfg["slots"]
    if int(sl["retry_max"]) < 0:
        raise ValueError("slots.retry_max must be >= 0")
    seen: dict[int, str] = {}
    for split in ("dev", "test"):
        blk = sl[split]
        quota = blk["quota"]
        if set(quota) != set(FAMILIES) or set(blk["reserve"]) != set(FAMILIES):
            raise ValueError(f"slots.{split}: quota and reserve must list exactly {FAMILIES}")
        for fam, q in quota.items():
            if isinstance(q, Mapping) != (fam == "M16") or (isinstance(q, Mapping) and set(q) != set(SIDES)):
                raise ValueError(f"slots.{split}.quota.{fam}: only M16 has sides, and it needs exactly {SIDES}")
        entries = _quota_entries(quota)
        seeds = [(int(blk["primary_seed_start"]) + i, f"{split} primary") for i in range(len(entries))]
        seeds += [(int(v), f"{split} reserve {fam}") for fam in sorted(blk["reserve"]) for v in blk["reserve"][fam]]
        for v, what in seeds:
            ok = v >= TEST_SEED_MIN if split == "test" else DEV_SEED_RANGE[0] <= v <= DEV_SEED_RANGE[1]
            if not ok:
                raise ValueError(f"slots: {what} seed {v} is outside the {split} range")
            if v in seen:
                raise ValueError(f"slots: seed {v} is used twice ({seen[v]} and {what})")
            seen[v] = what


def slot_list(ctx: Context | Mapping, split: str = "dev") -> list[dict]:
    """The fixed slot list of a split: one dict per slot, {'index', 'split', 'family', 'm16_side', 'primary_seed'}. A
    seeded permutation of the quota table (Fisher-Yates driven by Draw([permutation_seed])); a function of the config
    content only. split 'test' works (to inspect counts) but no test arrangement is generated."""
    cfg = ctx.config if isinstance(ctx, Context) else ctx
    if split not in ("dev", "test"):
        raise ValueError(f"split must be dev or test, got {split!r}")
    blk = cfg["slots"][split]
    order = Draw([int(blk["permutation_seed"])]).shuffle(_quota_entries(blk["quota"]))
    return [{"index": i, "split": split, "family": fam, "m16_side": side, "primary_seed": int(blk["primary_seed_start"]) + i}
            for i, (fam, side) in enumerate(order)]


def first_slot_per_class(slots: Sequence[Mapping]) -> list[int]:
    """Indices of the first slot of each class (S8, M16-L, M16-R, IND28, white_box, ycb_box) in list order."""
    seen, out = set(), []
    for sl in slots:
        key = (sl["family"], sl["m16_side"])
        if key not in seen:
            seen.add(key)
            out.append(int(sl["index"]))
    return out


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


def side_sign(side: str | None) -> float:
    """+1 for R (board displaced toward +x, image right at the start stop), -1 for L."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return 1.0 if side == "R" else -1.0


def _board(ctx: Context, d: Draw, spec: Mapping, target: Body, side: str | None = None) -> dict | None:
    o = R.occluder_spec(ctx.geometry, spec["geometry_occluder"])
    gap = _gap(ctx, d, spec, target, o.thickness)
    if gap is None:
        return None
    # a sided family (M16): shift = side sign x a magnitude from side_shift_m; else a signed range about the sight line
    shift = side_sign(side) * d.u(*spec["side_shift_m"]) if "side_shift_m" in spec else d.u(*spec["shift_m"])
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


def place_occluder(ctx: Context, d: Draw, target: dict, fam: str, side: str | None = None) -> dict | None:
    """A placed pose for an occluder of family `fam` in front of the target: P3 coverage >= p3_coverage_min and
    the family's stop rule (stop_rule_ok) and, for a sided family (M16), its side rule (side_rule_ok)."""
    oc = ctx.config["occluder"]
    cov_min = float(oc["p3_coverage_min"])
    tb = target["body"]
    spec = oc["families"][fam]
    ind = bool(spec.get("predeclared_indistinguishable", False))
    if ("side_rule" in spec) != (side is not None):
        raise ValueError(f"family {fam}: a side ({side!r}) is given if and only if the family has a side_rule")
    for _ in range(int(spec.get("tries", oc["tries_per_family"]))):
        if spec["kind"] == "board":
            cand = _board(ctx, d, spec, tb, side)
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
        if side is not None and not side_rule_ok(list(covs.values()), ctx.start_index, side, spec["side_rule"],
                                                 float(oc["clear_coverage_max"])):
            continue
        cand.update(family=fam, side=side, p3_coverage=cov, predeclared_indistinguishable=ind)
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


def side_rule_ok(covs: Sequence[float], start: int, side: str, rule: Mapping, clear_max: float) -> bool:
    """One-sidedness of a sided board, on the analytic hull coverage per stop (arc order, `start` = the start stop).
    The board's side runs toward higher stop indices for R and lower for L. Accepted if the adjacent stop on that side is
    covered >= rule['adjacent_min'], the nearest stop on the other side is covered <= rule['opposite_near_max'], and every
    further stop on the other side is clear (<= clear_max). Needs a stop on each side of the start stop."""
    step = int(side_sign(side))
    n = len(covs)
    same = [covs[i] for i in range(start + step, n if step > 0 else -1, step)]
    opp = [covs[i] for i in range(start - step, -1 if step > 0 else n, -step)]
    if not same or not opp:
        return False
    return bool(same[0] >= float(rule["adjacent_min"]) and opp[0] <= float(rule["opposite_near_max"])
                and all(v <= clear_max for v in opp[1:]))


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
             "occluder_pose": PHASE_STATE[ph][1], "camera": PHASE_CAMERA[ph], "lightings": lights} for ph, role, cond in PHASES]


def _analytic(ctx: Context, target: Body, placed: Body, parked: Body) -> dict:
    return {"note": "convex-hull coverage of the target by the occluder through the NOMINAL pinhole cameras; NOT visibility labels (those come from the render at the revisit poses)",
            "placed_coverage": {c.name: round(coverage(c, target, placed), 4) for c in ctx.cams},
            "parked_coverage": {c.name: round(coverage(c, target, parked), 4) for c in ctx.cams}}


# ---------------------------------------------------------------- revisit noise

def revisit_entropy(seed: int | str) -> list[int]:
    """Entropy of an arrangement's revisit stream: [seed] for a dev seed, [2**31 + n] for the design case 'design-n'."""
    if isinstance(seed, str):
        if not (seed.startswith("design-") and seed[7:].isdigit()):
            raise SeedError(f"{seed!r} is not a design seed")
        return [2 ** 31 + int(seed[7:])]
    return [check_dev_seed(seed)]


def _cam_pose(cam: R.Camera) -> dict:
    return {"position": _r(cam.position, 9), "heading_deg": round(cam.heading_deg, 9),
            "pitch_down_deg": round(cam.pitch_down_deg, 9), "roll_deg": round(cam.roll_deg, 9)}


def perturb_camera(cam: R.Camera, delta: Mapping[str, float]) -> R.Camera:
    """The camera moved by `delta` (dx_m, dy_m, dz_m along the world axes; dyaw_deg, dpitch_deg, droll_deg added to its
    heading, downward pitch and roll, i.e. rotations about its own yaw, pitch and roll axes), rounded to 1e-9 like the
    record."""
    x, y, z = cam.position
    return dataclasses.replace(cam, position=(round(x + delta["dx_m"], 9), round(y + delta["dy_m"], 9), round(z + delta["dz_m"], 9)),
                               heading_deg=round(cam.heading_deg + delta["dyaw_deg"], 9),
                               pitch_down_deg=round(cam.pitch_down_deg + delta["dpitch_deg"], 9),
                               roll_deg=round(cam.roll_deg + delta["droll_deg"], 9))


def revisit_draw(rc: Mapping, entropy: Sequence[int], stop_index: int, attempt: int) -> dict:
    """One bounded uniform draw (six numbers) for (arrangement, stop, attempt): +-translation_m per world axis,
    +-rotation_deg about the camera's yaw, pitch and roll, rounded to 1e-9."""
    d = Draw([*entropy, int(rc["stream_salt"]), int(stop_index), int(attempt)])
    tr, rot = float(rc["translation_m"]), float(rc["rotation_deg"])
    delta = {"dx_m": d.u(-tr, tr), "dy_m": d.u(-tr, tr), "dz_m": d.u(-tr, tr),
             "dyaw_deg": d.u(-rot, rot), "dpitch_deg": d.u(-rot, rot), "droll_deg": d.u(-rot, rot)}
    return {k: round(v, 9) for k, v in delta.items()}


def revisit_cameras(ctx: Context, rig: R.Rig, seed: int | str) -> tuple[dict, list[R.Camera]]:
    """(record, cameras): one revisit pose per stop. The first draw whose pose keeps the #3 outline and all four tags in
    the frame (sim/rig.py frame_margins, fitted k1, min margin >= min_margin_px: the nominal check) is used; each rejected
    draw is logged with its margin. RuntimeError if max_redraws draws are all rejected."""
    rc = ctx.config["revisit"]
    entropy = revisit_entropy(seed)
    distort, floor = bool(rc["distort"]), float(rc["min_margin_px"])
    stops, cams = {}, []
    for i, cam in enumerate(rig.cameras):
        rejected = []
        for attempt in range(int(rc["max_redraws"]) + 1):
            delta = revisit_draw(rc, entropy, i, attempt)
            pc = perturb_camera(cam, delta)
            m = R.frame_margins(rig, pc, distort=distort)
            if m["min"] >= floor:
                break
            rejected.append({"attempt": attempt, "delta": delta, "min_margin_px": round(float(m["min"]), 3)})
        else:
            raise RuntimeError(f"{seed}: no revisit pose for {cam.name} inside the frame in {rc['max_redraws'] + 1} draws")
        nm = R.frame_margins(rig, cam, distort=distort)
        stops[cam.name] = {
            "attempt": attempt, "n_rejected": len(rejected), "rejected": rejected, "delta": delta,
            "nominal": _cam_pose(cam), "pose": _cam_pose(pc),
            "margin_px": {"c3": round(float(m["c3"]), 3), "tags": {str(k): round(float(v), 3) for k, v in m["tags"].items()},
                          "min": round(float(m["min"]), 3)},
            "nominal_margin_px": {"c3": round(float(nm["c3"]), 3), "tags": {str(k): round(float(v), 3) for k, v in nm["tags"].items()},
                                  "min": round(float(nm["min"]), 3)}}
        cams.append(pc)
    rec = {"schema": REVISIT_SCHEMA,
           "applies_to": "query captures only (phases visible, occluded, removed; L1 and L2); the reference captures use the nominal cameras",
           "ranges": {"translation_m": rc["translation_m"], "rotation_deg": rc["rotation_deg"], "min_margin_px": rc["min_margin_px"],
                      "distort": rc["distort"], "max_redraws": rc["max_redraws"]},
           "stream": "Draw([*entropy, stream_salt, stop_index, attempt]); six uniform draws: dx, dy, dz (world axes, m), dyaw (heading), "
                     "dpitch (pitch_down), droll (deg)",
           "entropy": entropy, "stream_salt": int(rc["stream_salt"]), "stops": stops}
    return rec, cams


def _scene(ctx: Context, rig: R.Rig, keep_mug: bool, revisit_cams: Sequence[R.Camera]) -> dict:
    spec = R.export_scene_spec(rig)
    spec["boxes"] = [b for b in spec["boxes"] if b["role"] != "occluder"]
    if not keep_mug:
        spec["meshes"], spec["mug"] = [], None
    spec["occluder"] = None
    spec["materials"].update({k: {"rgba": v["rgba"]} for k, v in ctx.config.get("materials", {}).items()})
    spec["provenance"]["geometry_file"] = ctx.config["geometry"]
    spec["provenance"]["positions_file"] = ctx.config["positions"]
    # the query (revisit) cameras, in the format of 'cameras' (the nominal ones, used for the references)
    spec["revisit_cameras"] = R.export_scene_spec(dataclasses.replace(rig, cameras=list(revisit_cams)))["cameras"]
    return spec


def arrangement_hash(arr: Mapping) -> str:
    """sha256 of the arrangement without its UNHASHED_KEYS (itself and the caller-dependent 'unhashed' block)."""
    return sha256_json({k: v for k, v in arr.items() if k not in UNHASHED_KEYS})


def _finish(ctx: Context, arr: dict, rig: R.Rig, keep_mug: bool) -> dict:
    arr["phases"] = _phases(ctx)
    arr["lightings"] = list(LIGHTINGS)
    arr["stops"] = [c.name for c in ctx.cams]
    arr["start_stop"] = ctx.positions.get("start_stop", "P3")
    arr["provenance"] = ctx.provenance
    arr["revisit"], rcams = revisit_cameras(ctx, rig, arr["seed"])
    arr["scene"] = _scene(ctx, rig, keep_mug, rcams)
    arr["unhashed"] = dict(ctx.unhashed)
    arr["arrangement_sha256"] = arrangement_hash(arr)
    return arr


# ---------------------------------------------------------------- generation: tries, slots, sets

class PlacementFailure(RuntimeError):
    """One (seed, attempt) could not place its slot's family (reason: target / occluder / distractors / parked)."""

    def __init__(self, seed: int, attempt: int, reason: str):
        super().__init__(f"seed {seed} attempt {attempt}: placement failed at {reason}")
        self.seed, self.attempt, self.reason = seed, attempt, reason


class ReserveExhausted(RuntimeError):
    """A slot used up its retries and every seed of its family's reserve list."""


def try_place(ctx: Context, seed: int, attempt: int, family: str, side: str | None) -> tuple[dict | None, str | None, str]:
    """One try for slot class (family, side) at (seed, attempt): attempt 0 is the plain seed, attempts 1..retry_max the
    retries (entropy [seed, attempt]). Returns (parts, None, target_name) or (None, reason, target_name). The target
    model is drawn first, uniformly. No other family is ever tried."""
    seed = check_dev_seed(seed)
    if family not in ctx.config["occluder"]["families"]:
        raise ValueError(f"unknown occluder family {family!r}")
    d = Draw([seed] if attempt == 0 else [seed, int(attempt)])
    obj = int(d.choice(ctx.config["target"]["objects"]))
    name = ctx.item(obj)["name"]
    target = place_target(ctx, d, obj)
    if target is None:
        return None, "target", name
    occ = place_occluder(ctx, d, target, family, side)
    if occ is None:
        return None, "occluder", name
    dis = place_distractors(ctx, d, target, occ)
    if dis is None:
        return None, "distractors", name
    parked = park_occluder(ctx, d, occ, target["body"], [x["body"] for x in dis])
    if parked is None:
        return None, "parked", name
    return {"target": target, "occ": occ, "dis": dis, "parked": parked}, None, name


def generate(seed: int, family: str, m16_side: str | None = None, attempt: int = 0,
             config: str | Path | Mapping = DEFAULT_CONFIG_PATH, ctx: Context | None = None, slot: Mapping | None = None) -> dict:
    """The arrangement of one try (seed, attempt) of family `family` (and M16 side), deterministic in (seed, attempt, family,
    side, config, code, inputs). PlacementFailure if the try fails. Slot runs go through generate_slot."""
    ctx = ctx or load_context(config)
    parts, reason, _ = try_place(ctx, seed, attempt, family, m16_side)
    if parts is None:
        raise PlacementFailure(int(seed), int(attempt), reason)
    return _assemble(ctx, int(seed), parts["target"], parts["occ"], parts["dis"], parts["parked"],
                     {"seed": int(seed), "attempt": int(attempt), "family": family, "m16_side": m16_side}, slot)


def generate_slot(ctx: Context, slot: Mapping, reserve_used: dict[str, int]) -> tuple[dict, dict]:
    """The arrangement of one slot, with retries within its family and then reserve seeds of that family. Never another
    family or side. reserve_used[family] counts the reserve seeds consumed so far (updated). Returns (arrangement, log
    entry); the log entry lists every failed try and replacement."""
    split, fam, side = slot["split"], slot["family"], slot["m16_side"]
    retry_max = int(ctx.config["slots"]["retry_max"])
    reserve = [int(v) for v in ctx.config["slots"][split]["reserve"][fam]]
    seed, rank = int(slot["primary_seed"]), None
    failed: list[dict] = []
    replacements: list[dict] = []
    while True:
        for attempt in range(retry_max + 1):
            parts, reason, tname = try_place(ctx, seed, attempt, fam, side)
            if parts is not None:
                rec = {"index": slot["index"], "split": split, "family": fam, "m16_side": side,
                       "primary_seed": int(slot["primary_seed"]), "used_seed": seed, "used_attempt": attempt, "reserve_rank": rank,
                       "retry_max": retry_max, "failed_tries": failed, "replacements": replacements}
                arr = _assemble(ctx, seed, parts["target"], parts["occ"], parts["dis"], parts["parked"],
                                {"seed": seed, "attempt": attempt, "family": fam, "m16_side": side}, rec)
                return arr, {**rec, "arrangement_id": arr["arrangement_id"], "n_failed_tries": len(failed),
                             "replaced": rank is not None, "p3_coverage": arr["objects"]["occluder"]["p3_coverage"]}
            failed.append({"seed": seed, "attempt": attempt, "reason": reason, "target": tname})
        k = reserve_used.get(fam, 0)
        if k >= len(reserve):
            raise ReserveExhausted(f"slot {slot['index']} ({fam}{'' if side is None else '-' + side}): primary seed "
                                   f"{slot['primary_seed']} and all {len(reserve)} reserve seeds failed")
        reserve_used[fam] = k + 1
        replacements.append({"from_seed": seed, "to_seed": reserve[k], "reserve_rank": k,
                             "after_failed_tries": len(failed)})
        seed, rank = reserve[k], k


def generate_set(ctx: Context, split: str = "dev", indices: Sequence[int] | None = None) -> tuple[list[dict], dict]:
    """(arrangements, generation log) for the chosen slot indices of a split (None = all), in slot order. The reserve
    list of a family is consumed in slot order, so slots of a selected family before the last selected one are resolved
    too (their arrangements are discarded): a subset reproduces exactly what the full set would contain."""
    slots = slot_list(ctx, split)
    want = set(range(len(slots))) if indices is None else {int(i) for i in indices}
    bad = sorted(want - set(range(len(slots))))
    if bad:
        raise ValueError(f"slot indices {bad} are outside 0..{len(slots) - 1}")
    last = {}
    for i in sorted(want):
        last[slots[i]["family"]] = i
    reserve_used: dict[str, int] = {}
    arrs, entries = [], []
    for sl in slots:
        if sl["index"] > last.get(sl["family"], -1):
            continue
        arr, entry = generate_slot(ctx, sl, reserve_used)
        entry["selected"] = sl["index"] in want
        entries.append(entry)
        if entry["selected"]:
            arrs.append(arr)
    sel = [e for e in entries if e["selected"]]
    counts = {}
    for e in sel:
        c = counts.setdefault(e["family"] + ("" if e["m16_side"] is None else "-" + e["m16_side"]),
                              {"slots": 0, "with_retries": 0, "replaced": 0, "failed_tries": 0})
        c["slots"] += 1
        c["with_retries"] += int(e["n_failed_tries"] > 0)
        c["replaced"] += int(e["replaced"])
        c["failed_tries"] += e["n_failed_tries"]
    log = {"schema": GENERATION_LOG_SCHEMA, "split": split, "config": ctx.provenance["config"],
           "generator_sha256": ctx.provenance["generator"]["sha256"], "retry_max": int(ctx.config["slots"]["retry_max"]),
           "n_slots_in_list": len(slots), "n_selected": len(sel),
           "note": "entries cover the selected slots plus earlier slots of the same family that were resolved only to advance the "
                   "family's reserve list (selected = false)",
           "counts_by_class": counts, "reserve_used": dict(sorted(reserve_used.items())),
           "slots": entries,
           "retries": [{"slot": e["index"], "family": e["family"], "m16_side": e["m16_side"], **t}
                       for e in entries for t in e["failed_tries"]],
           "replacements": [{"slot": e["index"], "family": e["family"], "m16_side": e["m16_side"], **r}
                            for e in entries for r in e["replacements"]]}
    return arrs, log


def _assemble(ctx: Context, seed: int, target: dict, occ: dict, dis: list[dict], parked: tuple[Body, str],
              draws: dict, slot: Mapping | None = None) -> dict:
    tb = target["body"]
    t_rec = _object("target", "target", target["item"]["name"], "ycbv", tb, "c3", obj_id=target["obj_id"],
                    item=target["item"], rest=target["rest"], rest_matrix=ctx.rest(target["rest"]))
    o_kind = occ["kind"]
    o_rec = _object("occluder", "occluder", occ["family"] if o_kind == "box" else occ["item"]["name"], o_kind,
                    occ["body"], "c3", obj_id=occ.get("obj_id"), item=occ.get("item"), rest=occ.get("rest"),
                    rest_matrix=None if o_kind != "ycbv" else ctx.rest(occ["rest"]), material=occ.get("material"),
                    family=occ["family"], side=occ.get("side"), gap_m=_r(occ["gap_m"]), shift_m=_r(occ["shift_m"]),
                    p3_coverage=_r(occ["p3_coverage"], 4))
    if "geometry_occluder" in occ:
        o_rec["geometry_occluder"] = occ["geometry_occluder"]
    o_rec["poses"] = {"placed": o_rec.pop("pose"), "parked": _pose(parked[0], parked[1])}
    d_recs = [_object(f"distractor_{i}", "distractor", x["item"]["name"], "ycbv", x["body"], x["compartment"],
                      obj_id=x["obj_id"], item=x["item"], rest=x["rest"], rest_matrix=ctx.rest(x["rest"]))
              for i, x in enumerate(dis)]
    arr = {"schema": ARRANGEMENT_SCHEMA, "arrangement_id": arrangement_id(seed), "seed": seed, "split": "dev",
           "kind": "random", "environment": ctx.config["environment"], "target_id": t_rec["label"],
           "occluder_family": occ["family"], "m16_side": occ.get("side"), "design_case": False,
           "predeclared_indistinguishable": bool(occ["predeclared_indistinguishable"]),
           "objects": {"target": t_rec, "occluder": o_rec, "distractors": d_recs},
           "analytic": _analytic(ctx, tb, occ["body"], parked[0]), "draws": draws, "slot": None if slot is None else dict(slot)}
    return _finish(ctx, arr, ctx.rig, keep_mug=False)


def design_arrangement(index: int, config: str | Path | Mapping = DEFAULT_CONFIG_PATH, ctx: Context | None = None) -> dict:
    """Design case `index` (0..3): option A, measured mug at design z 0.24 (handle toward the camera), design occluder
    placed as in configs/rig_geometry.json, parked in #4 at the configured spot. No distractors. Predeclared dev
    arrangements outside the 60 slots (design_case true); with revisit cameras like every arrangement."""
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
                    family=case["family"], side=case["m16_side"], geometry_occluder=case["occluder"],
                    p3_coverage=_r(coverage(ctx.start_cam, mug, placed), 4),
                    intended_pattern=rig.occluder.intended_pattern)
    o_rec["poses"] = {"placed": o_rec.pop("pose"), "parked": _pose(parked, pk["compartment"])}
    arr = {"schema": ARRANGEMENT_SCHEMA, "arrangement_id": case["arrangement_id"], "seed": case["seed"], "split": "dev",
           "kind": "design", "environment": ctx.config["environment"], "target_id": dc["target_name"],
           "occluder_family": case["family"], "m16_side": case["m16_side"], "design_case": True,
           "predeclared_indistinguishable": bool(case.get("predeclared_indistinguishable", False)),
           "objects": {"target": t_rec, "occluder": o_rec, "distractors": []},
           "analytic": _analytic(ctx, mug, placed, parked), "design_note": dc["note"]}
    return _finish(ctx, arr, rig, keep_mug=True)


def n_design_cases(config: str | Path | Mapping = DEFAULT_CONFIG_PATH) -> int:
    return len(load_config(config)[0]["design_cases"]["cases"])


# ---------------------------------------------------------------- CLI

def parse_slot_indices(text: str, slots: Sequence[Mapping]) -> list[int]:
    """'all', 'first-per-class', or '0-5,9' -> sorted unique slot indices."""
    t = text.strip()
    if t == "all":
        return list(range(len(slots)))
    if t == "first-per-class":
        return sorted(first_slot_per_class(slots))
    out: set[int] = set()
    for part in (p.strip() for p in t.split(",") if p.strip()):
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = sorted(i for i in out if not 0 <= i < len(slots))
    if bad:
        raise ValueError(f"slot indices {bad} are outside 0..{len(slots) - 1}")
    return sorted(out)


def _index_row(arr: Mapping, p: Path) -> dict:
    return {"arrangement_id": arr["arrangement_id"], "seed": arr["seed"], "kind": arr["kind"], "target_id": arr["target_id"],
            "occluder_family": arr["occluder_family"], "m16_side": arr["m16_side"], "design_case": arr["design_case"],
            "slot": None if not arr.get("slot") else arr["slot"]["index"],
            "n_distractors": len(arr["objects"]["distractors"]), "file": p.name, "sha256": sha256_file(p),
            "arrangement_sha256": arr["arrangement_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.arrange", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.relative_to(R.REPO_ROOT)))
    ap.add_argument("--split", default="dev", choices=("dev", "test"),
                    help="slot list to use; 'test' is refused (no test arrangement is generated before the freeze)")
    ap.add_argument("--slots", default="", help="'all', 'first-per-class', or slot indices like '0-5,9' (none: no slot arrangements)")
    ap.add_argument("--design", action="store_true", help="also write the design cases DES0.. (outside the slots)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ctx = load_context(a.config)
    slots = slot_list(ctx, a.split)
    try:
        if a.split != "dev":
            check_dev_seed(slots[0]["primary_seed"])           # refuses with the test-range message; nothing is written
        indices = parse_slot_indices(a.slots, slots) if a.slots else []
    except (SeedError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index").mkdir(exist_ok=True)
    arrs = [design_arrangement(i, ctx=ctx) for i in range(len(ctx.config["design_cases"]["cases"]))] if a.design else []
    log = None
    if indices:
        slot_arrs, log = generate_set(ctx, a.split, indices)
        arrs += slot_arrs
    rows = []
    for arr in arrs:
        p = out / f"{arr['arrangement_id']}.json"
        p.write_text(json.dumps(arr, indent=1) + "\n")
        rows.append(_index_row(arr, p))
        o = arr["objects"]["occluder"]
        sl = arr.get("slot")
        print(f"{arr['arrangement_id']}: {'slot ' + str(sl['index']) if sl else 'design'}, seed {arr['seed']}, target {arr['target_id']}, "
              f"{arr['occluder_family']}{'' if arr['m16_side'] is None else '-' + arr['m16_side']} "
              f"(P3 hull coverage {o['p3_coverage']}), {len(arr['objects']['distractors'])} distractors"
              + ("" if not sl or (not sl['failed_tries']) else f", {len(sl['failed_tries'])} failed tries"
                 + (f", replaced by reserve {sl['reserve_rank']}" if sl["reserve_rank"] is not None else "")))
    prov = {"provenance": ctx.provenance, "unhashed": ctx.unhashed}
    (out / "index" / "index.json").write_text(json.dumps({"schema": INDEX_SCHEMA, **prov, "arrangements": rows}, indent=1) + "\n")
    (out / "index" / "slots.json").write_text(json.dumps({"schema": SLOTS_SCHEMA, **prov, "split": a.split, "retry_max": ctx.config["slots"]["retry_max"],
                                                           "slots": slots}, indent=1) + "\n")
    if log is not None:
        (out / "index" / "generation_log.json").write_text(json.dumps({**log, "unhashed": ctx.unhashed}, indent=1) + "\n")
        print(f"generation log: {len(log['retries'])} failed tries, {len(log['replacements'])} replacements over {log['n_selected']} slots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
