"""Batch Isaac Sim renderer for generated arrangements (sim/arrange.py): one Isaac process, a new stage per arrangement.

Why: the simulated capture must mirror the physical one (capture/run_arrangement.py): phases reference / visible /
occluded / removed, each at every stop under L1 and L2, with the occluder parked for reference and visible and the
same placed pose for occluded and removed (the matched layout). Simulation adds what the physical capture lacked:
exact instance masks, so visibility and diagnostic labels are computed, not annotated.

Revisit noise (docs/SIM_PREREGISTRATION.md, 2026-11-03): the REFERENCE captures use the nominal cameras
(scene.cameras); the three QUERY phases (visible, occluded, removed; both lightings) use the arrangement's revisit
cameras (scene.revisit_cameras: one bounded draw per stop, sim/arrange.py revisit_cameras), the same pose for every
query capture of the arrangement. Labels come from the renders at those poses.

Runs in the Isaac Sim 6.2 env on the GB10 (Python 3.12), like sim/isaac_rig.py. Isaac, omni and pxr are imported
only inside the functions that need them; everything else is numpy and is unit-tested anywhere
(tests/test_sim_batch.py). The USD helpers come from sim/isaac_rig.py (a copy sits next to this file in the GB10 run
dir).

    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 OMNI_KIT_ACCEPT_EULA=YES \\
      ~/.local/bin/uv run --no-project --python ~/cmpe249-sim/isaacsim-6.2/.venv/bin/python isaac_batch.py \\
      --arrangements arr/*.json --assets-dir ../assets/ycbv/models --tags-dir tags --out OUT

(arr/index/ holds the generator's index, slot list and log; non-arrangement JSON files that do get passed are skipped.)

Per arrangement, OUT/<ARR>/ gets:
  arrangement.json          the input, byte for byte (seed, generator and config hashes inside)
  rgb/<phase>_<stop>_<L>.png  RGB for every phase x stop x lighting (40 for 4 phases, 5 stops, L1/L2); each PNG
                            carries a tEXt chunk 'cmpe249' with the arrangement id, seed, phase, stop, lighting, camera
                            (nominal / revisit) and hashes
  masks/target_<phase>_<stop>.png   target pixels (0/255) for reference / visible / occluded (from the L1 render)
  masks/target_alone_<stop>.png     the target with everything else hidden at the NOMINAL camera (reference denominator)
  masks/target_alone_revisit_<stop>.png   the same at the REVISIT camera (denominator of the query views)
  masks/classes_<phase>_<stop>.png  uint8 class map: 0 shelf/tags/background, 1 target, 2 occluder, 10+i distractor i
  labels.json               visible fractions per (phase, stop), occluder coverage of the target's region (the
                            visible-phase mask at the same revisit pose) for occluded and removed, H/p/C patterns, truth,
                            diagnostic labels per (condition, stop), reference boxes from each lighting's reference mask,
                            the revisit deltas, L1/L2 mask agreement, hashes
  render.json               settings, Isaac version, per-step timings, hashes of this file and isaac_rig.py

Writing is atomic per arrangement: everything is rendered into OUT/<ARR>.partial and renamed to OUT/<ARR> when complete
(an existing folder is replaced). --skip-existing skips an arrangement only if its labels.json records the sha256 of the
very arrangement file given now (labels.hashes.arrangement_file_sha256); otherwise it is re-rendered. A leftover
OUT/<ARR>.partial marks an interrupted render and makes sim/episodes.py refuse the set until it is rendered again.

Render history: the real-time RTX renderer keeps per-pixel history in each render product. A target hidden between
phases left a target-shaped trace of +1.5 to +2.7 grey levels in the next phase's images for 100+ frames, also on an
occluder's face in front of it (data/raw/2026-11-03_sim_pilot/ghost_test*.py: unchanged with temporal AA off). Every
(phase, lighting) capture therefore uses NEW render products (empty history) and CAPTURE_UPDATES app updates before
the capture; with fresh render products the trace was within noise (-0.08 to +0.25 grey levels after 90 updates).
Segmentation has no history.

Labels (docs/SIM_PREREGISTRATION.md section 3): a present-target view (visible, occluded) is diagnostic when the
target's visible fraction is >= 0.90 (target pixels in the full scene / target pixels with everything else hidden, both at
that view's camera pose); a removed view is diagnostic when its expected region is unobstructed, i.e. the occluder covers
< 10 % of the target's region at that stop, where the region is the target mask of the VISIBLE-phase capture at the same
revisit pose (target present, occluder parked), not the nominal reference mask. The occlusionRatio annotator returns -1
on 6.2 and is not used.

Dev seeds only: an arrangement whose integer seed is outside 0-9999 (test seeds start at 100000) or whose split is
not 'dev' is refused before Isaac starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

try:
    from sim import isaac_rig as ir
except ImportError:  # run as a script from the GB10 run dir, next to a copy of isaac_rig.py
    import isaac_rig as ir

BATCH_VERSION = "0.2.0"
ARRANGEMENT_SCHEMA = "cmpe249.sim-arrangement.v1"
LABELS_SCHEMA = "cmpe249.sim-view-labels.v1"
NON_ARRANGEMENT_SCHEMAS = ("cmpe249.sim-arrangement-index.v0", "cmpe249.sim-slots.v0", "cmpe249.sim-generation-log.v0")   # = sim/arrange.py
RENDER_SCHEMA = "cmpe249.sim-render.v0"
DEV_SEED_RANGE = (0, 9999)       # = sim/arrange.py (tests/test_sim_batch.py checks)
TEST_SEED_MIN = 100000
PRESENT_DIAGNOSTIC_MIN = 0.90    # visible fraction for a diagnostic present-target view (the plan's C class)
REMOVED_OBSTRUCTION_MAX = 0.10   # occluder share of the target's reference region for a diagnostic removed view
CLASS_IDS = {"background": 0, "target": 1, "occluder": 2, "distractor0": 10}
PHASE_ORDER = ("reference", "visible", "occluded", "removed")
CAPTURE_UPDATES = 100            # app updates on fresh render products before each capture (ghost_test2.json)
ALONE_UPDATES = 10               # the target-alone pass reads segmentation only
CAMERA_SETS = ("nominal", "revisit")       # arrangement['phases'][i]['camera']: references nominal, queries revisit
CONDITION_TRUTH = {"visible": "present", "occluded": "present", "removed": "removed"}
THIS_FILE = Path(__file__).resolve()


# --------------------------------------------------------------------------- pure helpers (numpy only)
class RefusedArrangement(ValueError):
    pass


def check_arrangement(arr: Mapping) -> None:
    """Refuse anything but a dev arrangement: schema, split 'dev', and an integer seed in 0-9999 or a design seed
    'design-<n>'. Seeds >= 100000 are the reserved test range."""
    if arr.get("schema") != ARRANGEMENT_SCHEMA:
        raise RefusedArrangement(f"schema {arr.get('schema')!r}, expected {ARRANGEMENT_SCHEMA!r}")
    if arr.get("split") != "dev":
        raise RefusedArrangement(f"{arr.get('arrangement_id')}: split {arr.get('split')!r}; only dev arrangements render")
    seed = arr.get("seed")
    if isinstance(seed, str):
        if not (arr.get("kind") == "design" and seed.startswith("design-") and seed[7:].isdigit()):
            raise RefusedArrangement(f"{arr.get('arrangement_id')}: seed {seed!r} is not a design seed")
        return
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise RefusedArrangement(f"{arr.get('arrangement_id')}: seed {seed!r} is not an integer")
    if int(seed) >= TEST_SEED_MIN:
        raise RefusedArrangement(f"{arr.get('arrangement_id')}: seed {seed} is in the reserved TEST range (>= {TEST_SEED_MIN}); "
                                 "test seeds are never rendered before the freeze")
    if not DEV_SEED_RANGE[0] <= int(seed) <= DEV_SEED_RANGE[1]:
        raise RefusedArrangement(f"{arr.get('arrangement_id')}: seed {seed} is not a dev seed {DEV_SEED_RANGE}")


def check_scene(arr: Mapping) -> None:
    """The scene must carry both camera sets (nominal for references, revisit for queries) with the same stops, and every
    phase must name its camera set."""
    scene = arr.get("scene", {})
    names = [c["name"] for c in scene.get("cameras", [])]
    rnames = [c["name"] for c in scene.get("revisit_cameras", [])]
    if not names or names != rnames or names != list(arr.get("stops", [])):
        raise RefusedArrangement(f"{arr.get('arrangement_id')}: scene.cameras {names}, scene.revisit_cameras {rnames} and stops "
                                 f"{arr.get('stops')} must be the same non-empty list")
    for ph in arr.get("phases", []):
        if ph.get("camera") not in CAMERA_SETS:
            raise RefusedArrangement(f"{arr.get('arrangement_id')}: phase {ph.get('phase')!r} has camera {ph.get('camera')!r}")


def load_arrangement_files(paths: Sequence[str | Path]) -> tuple[list[tuple[dict, bytes]], list[tuple[str, str]]]:
    """([(arrangement, bytes)], [(file, skipped schema)]). A JSON file with one of NON_ARRANGEMENT_SCHEMAS (the generator's
    index / slots / log) is skipped, so a shell glob over the arrangement folder is harmless. Everything else must pass
    check_arrangement and check_scene (dev only, revisit cameras present) or the whole batch is refused."""
    items, skipped = [], []
    for f in paths:
        b = Path(f).read_bytes()
        arr = json.loads(b)
        if arr.get("schema") in NON_ARRANGEMENT_SCHEMAS:
            skipped.append((str(f), arr["schema"]))
            continue
        check_arrangement(arr)
        check_scene(arr)
        items.append((arr, b))
    return items, skipped


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def pose_matrix(base: Sequence[float], yaw_deg: float) -> np.ndarray:
    """4x4 USD local-to-world matrix (row vectors): rotate by yaw about z, then move the local origin to base."""
    m = np.eye(4)
    m[:3, :3] = rot_z(yaw_deg).T
    m[3, :3] = np.asarray(base, float)
    return m


def box_local_matrix(size: Sequence[float]) -> np.ndarray:
    """A size-2 cube (-1..1) scaled to `size` with its bottom face on the local origin (row-vector matrix)."""
    m = np.eye(4)
    m[:3, :3] = np.diag(np.asarray(size, float) / 2)
    m[3, 2] = float(size[2]) / 2
    return m


def asset_local(vertices_mm: np.ndarray, rest: np.ndarray, normals: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray | None]:
    """BOP model vertices (mm) -> metres, rotated by the rest pose (world = R @ model), bounding box centred in x/y,
    lowest point at z = 0: the frame sim/arrange.py's Body assumes (base = bottom-centre)."""
    R = np.asarray(rest, float)
    v = np.asarray(vertices_mm, float) / 1000.0 @ R.T
    lo, hi = v.min(0), v.max(0)
    v = v - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    n = None if normals is None else np.asarray(normals, float) @ R.T
    return v, n


def matches(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def class_map(ids: np.ndarray, id_to_labels: Mapping, roles: Mapping[str, int]) -> np.ndarray:
    """instance_id_segmentation ids + idToLabels (id -> prim path) -> uint8 class map; roles maps a prim-path prefix
    to a class id. Unlisted prims (shelf, tags, background) are 0."""
    out = np.zeros(np.asarray(ids).shape, np.uint8)
    for k, v in id_to_labels.items():
        for prefix, cls in roles.items():
            if matches(str(v), prefix):
                out[np.asarray(ids) == int(k)] = cls
                break
    return out


def mask_box(mask: np.ndarray) -> list[int] | None:
    """[x0, y0, x1, y1] in continuous pixel coordinates (pixel i covers [i, i + 1)) of a boolean mask, or None."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def classify(v: float | None, h_max: float = 0.05, c_min: float = 0.90) -> str:
    """sim/rig.py classify(): H <= 0.05, C >= 0.90, else p; '?' without pixels."""
    if v is None:
        return "?"
    return "H" if v <= h_max else ("C" if v >= c_min else "p")


def compute_labels(stops: Sequence[str], target: Mapping[tuple[str, str], np.ndarray], alone_ref: Mapping[str, np.ndarray],
                   alone_query: Mapping[str, np.ndarray], occluder: Mapping[tuple[str, str], np.ndarray],
                   h_max: float = 0.05, c_min: float = 0.90, present_min: float = PRESENT_DIAGNOSTIC_MIN,
                   removed_max: float = REMOVED_OBSTRUCTION_MAX) -> dict:
    """Labels from boolean masks. target[(phase, stop)] = target pixels in the full scene (reference, visible, occluded);
    alone_ref[stop] / alone_query[stop] = target pixels with everything else hidden at the NOMINAL / REVISIT camera;
    occluder[(phase, stop)] = visible occluder pixels (occluded, removed). The reference phase is measured at the nominal
    pose; visible, occluded and removed at the revisit pose of their stop.
      visible fraction = target pixels with the scene / target pixels without it (alone), at that view's pose;
      region = the target mask of the VISIBLE phase at the same revisit pose (target present, occluder parked);
      obstruction (occluded, removed) = share of the region covered by that phase's occluder pixels.
    Returns visible fractions, obstruction, patterns, truth and diagnostic labels per condition and stop (lighting does
    not change geometry)."""
    vis, px, obstruction = {}, {}, {}
    alone = {"reference": alone_ref, "visible": alone_query, "occluded": alone_query}
    for ph in ("reference", "visible", "occluded"):
        vis[ph], px[ph] = {}, {}
        for s in stops:
            a = int(np.count_nonzero(alone[ph][s]))
            t = int(np.count_nonzero(target[(ph, s)] & alone[ph][s]))
            px[ph][s] = int(np.count_nonzero(target[(ph, s)]))
            vis[ph][s] = None if a == 0 else round(t / a, 6)
    region_px = {s: int(np.count_nonzero(target[("visible", s)])) for s in stops}
    for ph in ("occluded", "removed"):
        obstruction[ph] = {}
        for s in stops:
            n = region_px[s]
            obstruction[ph][s] = None if n == 0 else round(int(np.count_nonzero(target[("visible", s)] & occluder[(ph, s)])) / n, 6)
    diag = {"visible": {s: vis["visible"][s] is not None and vis["visible"][s] >= present_min for s in stops},
            "occluded": {s: vis["occluded"][s] is not None and vis["occluded"][s] >= present_min for s in stops},
            "removed": {s: obstruction["removed"][s] is not None and obstruction["removed"][s] < removed_max for s in stops}}
    return {
        "visible_fraction": vis,
        "target_px": px,
        "alone_px": {"reference": {s: int(np.count_nonzero(alone_ref[s])) for s in stops},
                     "query": {s: int(np.count_nonzero(alone_query[s])) for s in stops}},
        "region_px": region_px,
        "region_obstruction": obstruction,
        "patterns": {ph: "".join(classify(vis[ph][s], h_max, c_min) for s in stops) for ph in ("reference", "visible", "occluded")},
        "truth": dict(CONDITION_TRUTH),
        "diagnostic": diag,
        "n_diagnostic": {c: int(sum(v.values())) for c, v in diag.items()},
        "indistinguishable_by_masks": not any(diag["occluded"].values()) and not any(diag["removed"].values()),
        "rules": {"present": f"visible fraction >= {present_min} (target pixels with the scene / target pixels with everything else "
                             "hidden, both at the view's own camera pose)",
                  "removed": f"occluder covers < {removed_max} of the target's region at that stop, the region being the target mask "
                             "of the visible-phase capture at the same revisit pose (target present, occluder parked)",
                  "classes": f"H <= {h_max}, C >= {c_min}, else p",
                  "poses": "reference: nominal cameras; visible, occluded, removed: revisit cameras (one draw per stop)"},
    }


def png_meta(arr: Mapping, phase: str, stop: str, lighting: str, camera: str = "nominal") -> dict:
    prov = arr.get("provenance", {})
    delta = arr.get("revisit", {}).get("stops", {}).get(stop, {}).get("delta") if camera == "revisit" else None
    return {"arrangement_id": arr["arrangement_id"], "seed": arr["seed"], "split": arr["split"], "phase": phase,
            "stop": stop, "lighting": lighting, "camera": camera, "revisit_delta": delta,
            "arrangement_sha256": arr.get("arrangement_sha256"),
            "generator_sha256": prov.get("generator", {}).get("sha256"), "config_sha256": prov.get("config", {}).get("sha256"),
            "renderer": f"sim/isaac_batch.py {BATCH_VERSION}", "simulation": True}


def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def partial_dir(out: Path, arrangement_id: str) -> Path:
    return Path(out) / f"{arrangement_id}.partial"


def needs_render(folder: Path, arr_bytes: bytes) -> tuple[bool, str]:
    """(True, reason) unless `folder` holds a COMPLETE render of exactly these arrangement bytes: arrangement.json equal to
    the input, a readable labels.json of the current schema whose hashes.arrangement_file_sha256 is the input's sha256,
    and render.json. --skip-existing skips only on (False, ...)."""
    want = hashlib.sha256(arr_bytes).hexdigest()
    lab = Path(folder) / "labels.json"
    if not lab.is_file():
        return True, "no labels.json"
    try:
        doc = json.loads(lab.read_text())
    except ValueError:
        return True, "labels.json unreadable"
    if doc.get("schema") != LABELS_SCHEMA:
        return True, f"labels schema {doc.get('schema')!r} is not {LABELS_SCHEMA!r}"
    have = doc.get("hashes", {}).get("arrangement_file_sha256")
    if have != want:
        return True, f"labels.json was rendered from arrangement file {str(have)[:12]}, input is {want[:12]}"
    if not (Path(folder) / "render.json").is_file():
        return True, "no render.json"
    af = Path(folder) / "arrangement.json"
    if not af.is_file() or hashlib.sha256(af.read_bytes()).hexdigest() != want:
        return True, "arrangement.json differs from the input"
    return False, "complete render of the same arrangement file"


def install_rendered(tmp: Path, final: Path) -> None:
    """Rename a finished render into place. An existing folder is first renamed aside and removed after the new one is in
    place, so `final` is never a mix of two renders (a crash between the renames leaves final.old<pid> and no final)."""
    import shutil

    tmp, final = Path(tmp), Path(final)
    old = None
    if final.exists():
        old = final.with_name(final.name + f".old{os.getpid()}")
        final.rename(old)
    tmp.rename(final)
    if old is not None:
        shutil.rmtree(old)


# --------------------------------------------------------------------------- USD side (Isaac env only)
def _xform(stage, path: str, matrix: np.ndarray | None = None):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, path)
    op = x.AddTransformOp()
    op.Set(Gf.Matrix4d(*(np.eye(4) if matrix is None else matrix).reshape(-1).tolist()))
    return x.GetPrim(), op


def _cube(stage, path: str, local: np.ndarray, mat) -> None:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(2.0)
    UsdGeom.Xformable(cube.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*local.reshape(-1).tolist()))
    ir._bind(cube.GetPrim(), mat)


def build_stage(arr: dict, assets_dir: Path, tags_dir: Path, ply_cache: dict) -> dict:
    """A new stage for one arrangement. Static parts (shelf, tag cards, tags, lights, cameras) come from the
    arrangement's scene spec; objects go under /World/Objects with their pose on an Xform, so phases only toggle
    visibility and the occluder's pose. Returns prim paths and the occluder's two pose matrices."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    spec = arr["scene"]
    mats = {k: ir._material(stage, k, rgb, r) for k, (rgb, r) in ir.MATERIALS.items()}
    for k, m in (spec.get("materials") or {}).items():
        rgba = m.get("rgba", [0.6, 0.6, 0.6, 1.0])
        mats[k] = ir._material(stage, f"spec_{ir._safe(k)}", rgba[:3], m.get("roughness", 0.6))
    out = {"cameras": {}, "cameras_revisit": {}, "lights": {}, "shelf": "/World/Shelf", "objects": "/World/Objects", "distractors": []}

    UsdGeom.Xform.Define(stage, "/World/Shelf")
    for b in spec.get("boxes", []):
        if b.get("role") == "occluder":
            raise ValueError("scene boxes must not hold the occluder (it is an arrangement object)")
        p = f"/World/Shelf/{ir._safe(b['role'])}_{ir._safe(b['name'])}"
        cube = UsdGeom.Cube.Define(stage, p)
        cube.GetSizeAttr().Set(2.0)
        m = ir.box_matrix(b["center"], b["half"], b.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0)))
        UsdGeom.Xformable(cube.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*m.reshape(-1).tolist()))
        ir._bind(cube.GetPrim(), mats.get(b.get("material", "grey"), mats["grey"]))
    for t in spec.get("tags", []):
        c = ir.tag_quad(t["corners"])
        png = Path(tags_dir) / f"aruco_4x4_50_id{t['id']}.png"
        prim = ir._mesh(stage, f"/World/Shelf/tag_{t['id']}", c, [[0, 3, 2], [0, 2, 1]], uv=[(0, 1), (1, 1), (1, 0), (0, 0)])
        UsdGeom.Mesh(prim).GetDoubleSidedAttr().Set(True)
        ir._bind(prim, ir._material(stage, f"tag_{t['id']}", (1, 1, 1), 0.8, str(png)))

    UsdGeom.Xform.Define(stage, "/World/Objects")
    objs = arr["objects"]

    def ycbv_mesh(parent: str, rec: dict, idx: str):
        ply = Path(assets_dir) / rec["ply"]
        if str(ply) not in ply_cache:
            ply_cache[str(ply)] = ir.read_ply(ply)
        d = ply_cache[str(ply)]
        v, n = asset_local(d["vertices"], np.asarray(rec["rest_matrix"], float), d["normals"])
        prim = ir._mesh(stage, f"{parent}/mesh", v, d["faces"], n, d["uv"])
        tex = None if not d["texture"] else str(ply.parent / d["texture"])
        ir._bind(prim, ir._material(stage, f"obj_{idx}", (0.8, 0.8, 0.8), 0.4, tex))

    # target
    t = objs["target"]
    tp = "/World/Objects/target"
    if t["kind"] == "mesh":
        _xform(stage, tp)
        for m in spec["meshes"]:
            if m["name"] in t["meshes"]:
                prim = ir._mesh(stage, f"{tp}/{ir._safe(m['name'])}", m["vertices"], m["faces"])
                ir._bind(prim, mats.get(m.get("material", "mug"), mats["grey"]))
    elif t["kind"] == "ycbv":
        _xform(stage, tp, pose_matrix(t["pose"]["base"], t["pose"]["yaw_deg"]))
        ycbv_mesh(tp, t, "target")
    else:
        raise ValueError(f"target kind {t['kind']!r}")
    out["target"] = tp
    # occluder: two poses, the Xform's transform op is switched per phase
    o = objs["occluder"]
    op_path = "/World/Objects/occluder"
    poses = {k: pose_matrix(p["base"], p["yaw_deg"]) for k, p in o["poses"].items()}
    _, xop = _xform(stage, op_path, poses["parked"])
    if o["kind"] == "box":
        _cube(stage, f"{op_path}/box", box_local_matrix(o["size_m"]), mats.get(o.get("material", "occluder"), mats["grey"]))
    elif o["kind"] == "ycbv":
        ycbv_mesh(op_path, o, "occluder")
    else:
        raise ValueError(f"occluder kind {o['kind']!r}")
    out["occluder"], out["occluder_op"], out["occluder_poses"] = op_path, xop, poses
    for i, dr in enumerate(objs["distractors"]):
        p = f"/World/Objects/distractor_{i}"
        _xform(stage, p, pose_matrix(dr["pose"]["base"], dr["pose"]["yaw_deg"]))
        ycbv_mesh(p, dr, f"distractor_{i}")
        out["distractors"].append(p)

    UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    for name, pos, size, _, _ in ir.LIGHTING["L1"][1]:
        lt = UsdLux.RectLight.Define(stage, f"/World/Lights/{name}")
        lt.GetWidthAttr().Set(size)
        lt.GetHeightAttr().Set(size)
        m = ir.camera_matrix(pos, np.array([0.0, 0.15, 0.1]) - np.asarray(pos), (0, 0, 1))
        UsdGeom.Xformable(lt.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*m.reshape(-1).tolist()))
        out["lights"][name] = lt.GetPrim().GetPath().pathString
    ir.set_lighting(stage, out, "L1")
    # two camera sets: nominal (the references) and revisit (the query phases: visible, occluded, removed)
    for key, root, cams in (("cameras", "Cameras", spec["cameras"]), ("cameras_revisit", "CamerasRevisit", spec["revisit_cameras"])):
        for cam in cams:
            p = f"/World/{root}/{cam['name']}"
            c = UsdGeom.Camera.Define(stage, p)
            h_ap, v_ap = ir.apertures(cam["f_px"], cam["width"], cam["height"])
            c.GetFocalLengthAttr().Set(ir.FOCAL_MM)
            c.GetHorizontalApertureAttr().Set(h_ap)
            c.GetVerticalApertureAttr().Set(v_ap)
            c.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
            m = ir.camera_matrix(cam["position"], cam["forward"], cam["up"])
            UsdGeom.Xformable(c.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*m.reshape(-1).tolist()))
            out[key][cam["name"]] = p
    return out


def _visible(stage, path: str, on: bool) -> None:
    from pxr import UsdGeom

    im = UsdGeom.Imageable(stage.GetPrimAtPath(path))
    im.MakeVisible() if on else im.MakeInvisible()


def render_arrangement(app, arr: dict, arr_bytes: bytes, out_dir: Path, assets_dir: Path, tags_dir: Path,
                       lightings: Sequence[str], rt_subframes: int, settle: int, capture_updates: int, ply_cache: dict,
                       pool) -> dict:
    """Render one arrangement into `out_dir` (main() passes a fresh OUT/<ARR>.partial and installs it when this returns)."""
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image, PngImagePlugin
    from pxr import Gf

    t0 = time.time()
    timings: dict = {}
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "arrangement.json").write_bytes(arr_bytes)
    built = build_stage(arr, assets_dir, tags_dir, ply_cache)
    stage = omni.usd.get_context().get_stage()
    spec = arr["scene"]
    stops = [c["name"] for c in spec["cameras"]]
    if stops != list(arr["stops"]) or stops != [c["name"] for c in spec["revisit_cameras"]]:
        raise ValueError(f"camera order {stops} != arrangement stops {arr['stops']}")
    cams = {c["name"]: c for c in spec["cameras"]}
    cam_paths = {"nominal": built["cameras"], "revisit": built["cameras_revisit"]}

    def attach(cam_set: str):
        """New render products and annotators for every camera of a set: a new render product has no history, so nothing
        rendered before (a target that has since been hidden, an occluder that has moved) can linger in its images."""
        rps, anns = [], {}
        for name, path in cam_paths[cam_set].items():
            rp = rep.create.render_product(path, (cams[name]["width"], cams[name]["height"]))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            seg = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
            rgb.attach(rp)
            seg.attach(rp)
            rps.append(rp)
            anns[name] = (rgb, seg)
        return rps, anns

    def detach(rps, anns):
        for rgb, seg in anns.values():
            rgb.detach()
            seg.detach()
        for rp in rps:
            rp.destroy()

    timings["build_s"] = round(time.time() - t0, 2)
    t1 = time.time()
    rps, anns = attach("nominal")        # load textures / compile shaders on the new stage
    for _ in range(settle):
        app.update()
    detach(rps, anns)
    timings["settle_s"] = round(time.time() - t1, 2)
    roles = {built["target"]: CLASS_IDS["target"], built["occluder"]: CLASS_IDS["occluder"]}
    roles.update({p: CLASS_IDS["distractor0"] + i for i, p in enumerate(built["distractors"])})

    cls_maps: dict = {}
    futures = []
    steps = []
    phases = {p["phase"]: p for p in arr["phases"]}
    for ph in PHASE_ORDER:
        st = phases[ph]
        cam_set = st["camera"]
        _visible(stage, built["target"], bool(st["target_present"]))
        built["occluder_op"].Set(Gf.Matrix4d(*built["occluder_poses"][st["occluder_pose"]].reshape(-1).tolist()))
        for lighting in lightings:
            ts = time.time()
            ir.set_lighting(stage, built, lighting)
            rps, anns = attach(cam_set)  # fresh history for every capture (ghost_test2: hidden targets linger otherwise)
            for _ in range(capture_updates):
                app.update()
            rep.orchestrator.step(rt_subframes=rt_subframes)
            for name, (rgb, seg) in anns.items():
                img = np.asarray(rgb.get_data())[..., :3].copy()
                info = PngImagePlugin.PngInfo()
                info.add_text("cmpe249", json.dumps(png_meta(arr, ph, name, lighting, cam_set)))
                futures.append(pool.submit(Image.fromarray(img).save, out_dir / "rgb" / f"{ph}_{name}_{lighting}.png",
                                           pnginfo=info, compress_level=3))
                sd = seg.get_data()
                cls_maps[(ph, name, lighting)] = class_map(sd["data"], sd["info"].get("idToLabels", {}), roles)
            detach(rps, anns)
            steps.append({"phase": ph, "camera": cam_set, "lighting": lighting, "s": round(time.time() - ts, 2)})
    # the target alone (denominator of the visible fraction): hide everything else; segmentation only, which has no
    # temporal history, so a short settle suffices. Once at the nominal cameras (the references' denominator) and once
    # at the revisit cameras (the query views')
    ts = time.time()
    hide = [built["shelf"], built["occluder"], *built["distractors"]]
    for p in hide:
        _visible(stage, p, False)
    _visible(stage, built["target"], True)
    ir.set_lighting(stage, built, lightings[0])
    alone_by_set = {}
    for cam_set in CAMERA_SETS:
        rps, anns = attach(cam_set)
        for _ in range(ALONE_UPDATES):
            app.update()
        rep.orchestrator.step(rt_subframes=1)
        alone_by_set[cam_set] = {}
        for name, (_, seg) in anns.items():
            sd = seg.get_data()
            alone_by_set[cam_set][name] = class_map(sd["data"], sd["info"].get("idToLabels", {}), roles) == CLASS_IDS["target"]
        detach(rps, anns)
    for p in hide:
        _visible(stage, p, True)
    alone, alone_q = alone_by_set["nominal"], alone_by_set["revisit"]
    steps.append({"phase": "alone", "camera": "nominal+revisit", "lighting": lightings[0], "s": round(time.time() - ts, 2)})
    for f in futures:
        f.result()

    L0 = lightings[0]
    tgt = {(ph, s): cls_maps[(ph, s, L0)] == CLASS_IDS["target"] for ph in ("reference", "visible", "occluded") for s in stops}
    occ = {(ph, s): cls_maps[(ph, s, L0)] == CLASS_IDS["occluder"] for ph in ("occluded", "removed") for s in stops}
    vc = spec.get("visibility_classes", {"H_max": 0.05, "C_min": 0.90})
    labels = compute_labels(stops, tgt, alone, alone_q, occ, vc["H_max"], vc["C_min"])
    # every class map must agree between lightings (segmentation is geometry); record the differences
    labels["lighting_mask_agreement"] = {f"{ph}_{s}": {L: int(np.count_nonzero(cls_maps[(ph, s, L)] != cls_maps[(ph, s, L0)]))
                                                       for L in lightings[1:]} for ph in PHASE_ORDER for s in stops}
    labels["reference_box"] = {L: {s: mask_box(cls_maps[("reference", s, L)] == CLASS_IDS["target"]) for s in stops}
                               for L in lightings}
    labels["query_box_visible"] = {L: {s: mask_box(cls_maps[("visible", s, L)] == CLASS_IDS["target"]) for s in stops}
                                   for L in lightings}
    labels["removed_target_px"] = {s: int(np.count_nonzero(cls_maps[("removed", s, L0)] == CLASS_IDS["target"])) for s in stops}
    rv = arr.get("revisit", {})
    labels["revisit"] = {"schema": rv.get("schema"), "ranges": rv.get("ranges"),
                         "stops": {s: {"delta": rv["stops"][s]["delta"], "attempt": rv["stops"][s]["attempt"],
                                       "n_rejected": rv["stops"][s]["n_rejected"],
                                       "min_margin_px": rv["stops"][s]["margin_px"]["min"],
                                       "nominal_min_margin_px": rv["stops"][s]["nominal_margin_px"]["min"]} for s in stops}}
    for ph in ("reference", "visible", "occluded"):
        for s in stops:
            Image.fromarray(tgt[(ph, s)].astype(np.uint8) * 255).save(out_dir / "masks" / f"target_{ph}_{s}.png")
    for s in stops:
        Image.fromarray(alone[s].astype(np.uint8) * 255).save(out_dir / "masks" / f"target_alone_{s}.png")
        Image.fromarray(alone_q[s].astype(np.uint8) * 255).save(out_dir / "masks" / f"target_alone_revisit_{s}.png")
        for ph in PHASE_ORDER:
            Image.fromarray(cls_maps[(ph, s, L0)]).save(out_dir / "masks" / f"classes_{ph}_{s}.png")
    prov = arr.get("provenance", {})
    labels.update({
        "schema": LABELS_SCHEMA, "arrangement_id": arr["arrangement_id"], "seed": arr["seed"], "split": arr["split"],
        "kind": arr["kind"], "target_id": arr["target_id"], "stops": stops, "lightings": list(lightings),
        "predeclared_indistinguishable": bool(arr.get("predeclared_indistinguishable", False)),
        "occluder_family": arr["objects"]["occluder"]["family"], "m16_side": arr.get("m16_side"),
        "design_case": bool(arr.get("design_case", False)),
        "intended_pattern": arr["objects"]["occluder"].get("intended_pattern"),
        "class_ids": {"background (shelf, tags, dome)": 0, "target": 1, "occluder": 2,
                      **{f"distractor_{i}": CLASS_IDS["distractor0"] + i for i in range(len(built["distractors"]))}},
        "masks_from_lighting": L0, "simulation": True,
        "hashes": {"arrangement_file_sha256": hashlib.sha256(arr_bytes).hexdigest(),
                   "arrangement_sha256": arr.get("arrangement_sha256"),
                   "generator_sha256": prov.get("generator", {}).get("sha256"),
                   "config_sha256": prov.get("config", {}).get("sha256"),
                   "isaac_batch_sha256": sha256_file(THIS_FILE), "isaac_rig_sha256": sha256_file(ir.__file__)},
    })
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1) + "\n")
    timings["steps"] = steps
    timings["total_s"] = round(time.time() - t0, 2)
    return {"timings": timings, "labels": labels}


def isaac_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("isaacsim")
    except Exception:  # noqa: BLE001 - informational only
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrangements", nargs="+", required=True,
                    help="arrangement JSON files (sim/arrange.py); the generator's index/slots/log JSON files are skipped")
    ap.add_argument("--assets-dir", required=True, help="YCB-Video BOP models/ (obj_0000NN.ply + .png)")
    ap.add_argument("--tags-dir", required=True, help="aruco_4x4_50_id<N>.png files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lighting", nargs="+", default=["L1", "L2"], choices=sorted(ir.LIGHTING))
    ap.add_argument("--renderer", default="RaytracedLighting", choices=["RaytracedLighting", "PathTracing"])
    ap.add_argument("--rt-subframes", type=int, default=16)
    ap.add_argument("--settle", type=int, default=ir.SETTLE_UPDATES, help="app updates after building each stage")
    ap.add_argument("--capture-updates", type=int, default=CAPTURE_UPDATES,
                    help="app updates on fresh render products before each (phase, lighting) capture")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip an arrangement only if OUT/<ARR> holds a complete render of exactly this arrangement file "
                         "(labels.hashes.arrangement_file_sha256 equals its sha256); otherwise re-render")
    a = ap.parse_args(argv)
    items, skipped = load_arrangement_files(a.arrangements)     # refuses test / malformed arrangements before Isaac starts
    for f, schema in skipped:
        print(f"skip {f} (not an arrangement: {schema})", flush=True)
    if not items:
        raise SystemExit("no arrangement files given")
    ids = [arr["arrangement_id"] for arr, _ in items]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate arrangement ids: {ids}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cam0 = items[0][0]["scene"]["cameras"][0]
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from concurrent.futures import ThreadPoolExecutor

    t_boot = time.time()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": a.renderer, "width": cam0["width"], "height": cam0["height"]})
    boot_s = round(time.time() - t_boot, 2)
    report = {"schema": RENDER_SCHEMA, "batch_version": BATCH_VERSION, "isaacsim": isaac_version(), "renderer": a.renderer,
              "rt_subframes": a.rt_subframes, "settle_updates": a.settle, "capture_updates": a.capture_updates,
              "history": "fresh render products for every (phase, lighting) capture (ghost_test2: a hidden target otherwise "
                         "leaves a 1.5-2.7 grey-level trace for 100+ frames)",
              "lighting": a.lighting, "lighting_presets": {k: ir.LIGHTING[k] for k in a.lighting},
              "materials": ir.MATERIALS, "focal_mm": ir.FOCAL_MM, "boot_s": boot_s,
              "isaac_batch_sha256": sha256_file(THIS_FILE), "isaac_rig_sha256": sha256_file(ir.__file__),
              "assets_dir": str(a.assets_dir), "models_info_sha256": sha256_file(Path(a.assets_dir) / "models_info.json")
              if (Path(a.assets_dir) / "models_info.json").exists() else None, "arrangements": {}}
    ply_cache: dict = {}
    try:
        ir._tonemap()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for arr, b in items:
                d = out / arr["arrangement_id"]
                redo, why = needs_render(d, b)
                if a.skip_existing and not redo:
                    print(f"skip {arr['arrangement_id']} ({why})", flush=True)
                    continue
                if a.skip_existing and d.exists():
                    print(f"re-render {arr['arrangement_id']}: {why}", flush=True)
                tmp = partial_dir(out, arr["arrangement_id"])
                if tmp.exists():                    # an interrupted earlier render
                    import shutil

                    shutil.rmtree(tmp)
                r = render_arrangement(app, arr, b, tmp, Path(a.assets_dir), Path(a.tags_dir), a.lighting, a.rt_subframes,
                                       a.settle, a.capture_updates, ply_cache, pool)
                rec = {k: v for k, v in report.items() if k != "arrangements"}
                rec.update(arrangement_id=arr["arrangement_id"], timings=r["timings"])
                (tmp / "render.json").write_text(json.dumps(rec, indent=1) + "\n")
                install_rendered(tmp, d)            # complete folder or none: never a mix of two renders
                report["arrangements"][arr["arrangement_id"]] = r["timings"]
                lab = r["labels"]
                print("ISAAC_BATCH " + json.dumps({"arrangement_id": arr["arrangement_id"], "total_s": r["timings"]["total_s"],
                                                   "occluded_vf": lab["visible_fraction"]["occluded"],
                                                   "patterns": lab["patterns"], "n_diagnostic": lab["n_diagnostic"]}), flush=True)
        (out / f"render_report_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(report, indent=1) + "\n")
    except BaseException:
        # SimulationApp.close() ends the process (fast shutdown) before Python would print the exception
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
