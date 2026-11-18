"""Shelf goldens (tabletop wave, step 0): canonical hashes of every shelf output, so that the tabletop changes to sim/ can be
shown to leave the shelf byte-for-byte unchanged. A helper module, not a test module: tests/test_shelf_goldens.py runs the
checks and tests/golden/make_shelf_goldens.py wrote tests/golden/shelf_goldens.json at origin/main 51cf0a3 (before any
tabletop edit).

Canonical hash: sha256 of json.dumps(obj, sort_keys=True, separators=(",", ":")) (= sim/arrange.py sha256_json) after
masking ONLY the fields that hash code files, which the tabletop wave changes by construction:
  - an arrangement (sim/arrange.py): provenance.generator.sha256 (arrange.py), provenance.rig_py.sha256 (rig.py) and the
    arrangement_sha256 computed over them (arrangement_hash). The golden also keeps the stored code hashes, so a
    regenerated arrangement with them restored must give the stored arrangement_sha256 again (restore_code_hashes);
  - the generator's index files: the same two provenance fields, generation_log.generator_sha256, and index.json's per-row
    file sha256 / arrangement_sha256 (hashes of the arrangement files, which embed the code hashes);
  - sim/episodes.py outputs: every 'episodes_py_sha256' key, and set.json's 'files' values (sha256 of the files written,
    which embed episodes_py_sha256 and absolute paths; every one of those files has its own golden);
  - sim/isaac_batch.py (stub render, tests/fake_isaac.py): labels.json hashes.isaac_batch_sha256 / isaac_rig_sha256 and
    render_settings.isaacsim (the installed Isaac Sim version, None off the GB10).
Nothing else is masked. sim/check_batch.py output has no code hash and is compared unmasked.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO / "tests" / "golden" / "shelf_goldens.json"
SIM_SHARE = Path("/mnt/bench-data/datasets/cmpe249-active-verification/sim")
DEV60 = SIM_SHARE / "shelf_v3_geometry" / "dev60"          # the 64 shelf_v3 dev arrangements (python -m sim.arrange, make_v3.py)
SHELF_DEV_V3 = SIM_SHARE / "shelf_dev_v3"                    # the rendered shelf_v3 dev set (isaac_batch + episodes + check_batch)
CONFIG = "configs/sim/arrange_shelf_v3.json"                 # spelled as make_v3 / dev60 spelled it (unhashed.config_path)
POSITIONS = "configs/sim/positions_shelf_A.json"
MASK = "<masked: code hash>"
# the slots the default test regenerates (the full 64 under CMPE249_GOLDEN_FULL=1): the first slot of each class
# (S8, M16-L, M16-R, IND28, white_box, ycb_box = slots 0, 1, 2, 5, 7, 10 of the dev list) plus four others (3 = S8 with 4
# distractors, 4 = M16-L, 8 = ycb_box with one failed try, 12 = M16-R), and the four design cases
REGEN_SUBSET = [0, 1, 2, 3, 4, 5, 7, 8, 10, 12]
CHECK_SUBSET = ["DES0", "SD1000", "SD1008"]                  # check_batch.check_arrangement per-arrangement goldens (fast)
FAKE_ARRANGEMENT = "SD1000"                                  # slot 0: M16-L, foam brick, 2 distractors (fake_isaac + stage log)
STAGE_LOG_ARRANGEMENTS = ["SD1000", "DES0"]                  # build_stage call logs: a YCB-V target and the study-mug mesh target


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


# ---------------------------------------------------------------- masking

def _mask_path(obj: dict, keys: tuple[str, ...]) -> None:
    d = obj
    for k in keys[:-1]:
        if not isinstance(d, dict) or k not in d:
            return
        d = d[k]
    if isinstance(d, dict) and keys[-1] in d:
        d[keys[-1]] = MASK


def mask_provenance(prov_holder: dict) -> None:
    _mask_path(prov_holder, ("provenance", "generator", "sha256"))
    _mask_path(prov_holder, ("provenance", "rig_py", "sha256"))


def canonical_arrangement(arr: Mapping) -> dict:
    a = copy.deepcopy(dict(arr))
    mask_provenance(a)
    if "arrangement_sha256" in a:
        a["arrangement_sha256"] = MASK
    return a


def canonical_index_file(name: str, doc: Mapping) -> dict:
    d = copy.deepcopy(dict(doc))
    mask_provenance(d)
    if name == "generation_log.json":
        d["generator_sha256"] = MASK
    if name == "index.json":
        for row in d.get("arrangements", []):
            row["sha256"] = MASK
            row["arrangement_sha256"] = MASK
    return d


def mask_keys(obj, keys: tuple[str, ...] = ("episodes_py_sha256",)):
    """A copy of obj with the value of every dict key in `keys` (at any depth) masked."""
    if isinstance(obj, dict):
        return {k: (MASK if k in keys else mask_keys(v, keys)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask_keys(v, keys) for v in obj]
    return obj


def replace_strings(obj, old: str, new: str):
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, dict):
        return {replace_strings(k, old, new): replace_strings(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_strings(v, old, new) for v in obj]
    return obj


def restore_code_hashes(arr: Mapping, generator_sha256: str, rig_py_sha256: str) -> str:
    """arrangement_sha256 of `arr` recomputed with the golden's code hashes put back into its provenance."""
    from sim import arrange as A

    a = copy.deepcopy(dict(arr))
    a["provenance"]["generator"]["sha256"] = generator_sha256
    a["provenance"]["rig_py"]["sha256"] = rig_py_sha256
    return A.arrangement_hash(a)


# ---------------------------------------------------------------- generator

def generator_context():
    from sim import arrange as A

    return A.load_context(CONFIG)


def slot_list_hashes(ctx) -> dict:
    from sim import arrange as A

    return {split: {"sha256": sha256_json(A.slot_list(ctx, split)), "n": len(A.slot_list(ctx, split))} for split in ("dev", "test")}


def regenerate(ctx, slot_indices, design: bool = True) -> dict[str, dict]:
    """{arrangement id: arrangement} for the dev slots (each with an empty reserve cursor: valid because the dev60 generation
    log records no reserve replacement, which the golden pins) and, if `design`, the design cases."""
    from sim import arrange as A

    slots = A.slot_list(ctx, "dev")
    out = {}
    for i in slot_indices:
        arr, _ = A.generate_slot(ctx, slots[i], {})
        out[arr["arrangement_id"]] = arr
    if design:
        for k in range(A.n_design_cases(CONFIG)):
            arr = A.design_arrangement(k, ctx=ctx)
            out[arr["arrangement_id"]] = arr
    return out


# ---------------------------------------------------------------- rendered set: episodes (on a JSON-only copy)

def copy_set_json(src_root: Path, dst_root: Path) -> None:
    """The inputs sim/episodes.py reads: <ARR>/arrangement.json + labels.json per rendered folder; the set's
    generation_log.json goes NEXT TO dst_root (build copies it into the set)."""
    dst_root.mkdir(parents=True)
    for d in sorted(Path(src_root).iterdir()):
        if d.is_dir() and (d / "labels.json").is_file():
            (dst_root / d.name).mkdir()
            for fn in ("arrangement.json", "labels.json"):
                shutil.copyfile(d / fn, dst_root / d.name / fn)
    shutil.copyfile(Path(src_root) / "generation_log.json", dst_root.parent / "generation_log.json")


EPISODES_FILES = ("jobs_refL1.json", "jobs_same.json", "manifest.json", "truth.json", "view_labels.json", "true_alignment.json",
                  "summary.md", "jobs_design_refL1.json", "jobs_design_same.json", "manifest_design.json", "truth_design.json",
                  "view_labels_design.json", "true_alignment_design.json", "summary_design.md", "positions_shelf_A.json",
                  "generation_log.json", "set.json", "README.md")


def _canonical_file(p: Path, root: Path, as_root: Path) -> str:
    text = p.read_text()
    if p.suffix != ".json":
        return hashlib.sha256(text.replace(str(root), str(as_root)).encode()).hexdigest()
    doc = mask_keys(replace_strings(json.loads(text), str(root), str(as_root)))
    if p.name == "set.json":
        doc["files"] = {k: MASK for k in doc["files"]}
    return sha256_json(doc)


def episodes_hashes(root: Path, as_root: Path | None = None) -> dict:
    """Canonical hashes of a set's sim/episodes.py outputs (EPISODES_FILES and every <ARR>/annotations/*.json, digested per
    arrangement). `as_root`: the set root the paths inside the files are rewritten to (a copy is compared as the share)."""
    root = Path(root)
    as_root = Path(as_root or root)
    files = {fn: _canonical_file(root / fn, root, as_root) for fn in EPISODES_FILES}
    ann = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "annotations").is_dir():
            lines = [f"{p.name}:{_canonical_file(p, root, as_root)}" for p in sorted((d / "annotations").glob("*.json"))]
            ann[d.name] = {"n": len(lines), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}
    return {"files": files, "annotations": ann}


def rebuild_episodes(src_root: Path, work: Path) -> dict:
    """Run sim/episodes.py build (as the set was built: both reference pairings, the shelf positions file, the set's
    generation log) on a JSON-only copy of src_root under work/<src name>, and return episodes_hashes of the copy with its
    paths rewritten to src_root. The share is only read."""
    from sim import episodes as E

    dst = Path(work) / Path(src_root).name               # same folder name: README.md names the set by it
    copy_set_json(src_root, dst)
    E.build(dst, E.REF_LIGHTINGS, POSITIONS, sheets=False, generation_log=dst.parent / "generation_log.json")
    return episodes_hashes(dst, src_root)


# ---------------------------------------------------------------- rendered set: check_batch

def check_arrangement_hashes(root: Path, ids) -> dict:
    from sim import check_batch as CB

    tags = CB._tags_module()
    out = {}
    for i in ids:
        a = CB.check_arrangement(Path(root) / i, tags)
        a["rendered_side"] = CB.side_of_pattern(a["labels"]["patterns"]["occluded"])      # as check_set adds it
        out[i] = sha256_json(json.loads(json.dumps(a)))                                   # as written to check.json
    return out


def check_set_hash(root: Path) -> str:
    """Canonical hash of sim/check_batch.py's full result (check_set) on a set root; ~85 s on the Jetson for shelf_dev_v3."""
    from sim import check_batch as CB

    return sha256_json(json.loads(json.dumps(CB.check_set(Path(root)))))


def check_file_hashes(doc: Mapping) -> dict:
    """{'check.json': canonical hash, 'arrangements': {id: canonical hash of its entry}} of a check_batch result."""
    return {"check.json": sha256_json(doc), "arrangements": {a["arrangement_id"]: sha256_json(a) for a in doc["arrangements"]}}


# ---------------------------------------------------------------- stub render (tests/fake_isaac.py)

def fake_input_arrangement(golden: Mapping, arr_id: str = FAKE_ARRANGEMENT, ctx=None) -> dict:
    """A stub-render / stage-log input: arrangement `arr_id` (a dev slot 'SDnnnn' or a design case 'DESn') regenerated by
    the current code, with the golden's code hashes and stored arrangement_sha256 restored, so the input bytes stay those of
    dev60/<arr_id>.json whatever the code files' hashes are (the generator golden separately pins that everything else in it
    is unchanged)."""
    from sim import arrange as A

    ctx = ctx or generator_context()
    if arr_id.startswith("DES"):
        arr = A.design_arrangement(int(arr_id[3:]), ctx=ctx)
    else:
        sl = next(s for s in A.slot_list(ctx, "dev") if A.arrangement_id(s["primary_seed"]) == arr_id)
        arr, _ = A.generate_slot(ctx, sl, {})
    g = golden["code_hashes"]
    arr["provenance"]["generator"]["sha256"] = g["sim/arrange.py"]
    arr["provenance"]["rig_py"]["sha256"] = g["sim/rig.py"]
    arr["arrangement_sha256"] = golden["generator"]["arrangements"][arr_id]["arrangement_sha256"]
    return arr


def fake_render(arr: Mapping, work: Path) -> dict:
    """Render `arr` with sim/isaac_batch.py main() against the stub Isaac (tests/fake_isaac.py) into work/out and return
    canonical hashes of the call log (every annotator read: step, lighting, kind, render product, camera path), the PNG
    metadata, the render settings (render.json) and labels.json."""
    from sim import isaac_batch as B
    from tests import fake_isaac as FI

    work = Path(work)
    (work / "assets").mkdir(parents=True)
    f = work / f"{arr['arrangement_id']}.json"
    f.write_text(json.dumps(arr, indent=1) + "\n")
    out = work / "out"
    FI.reset(1)
    with contextlib.ExitStack() as st:
        for p in FI.patches(B):
            st.enter_context(p)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            B.main(["--arrangements", str(f), "--assets-dir", str(work / "assets"), "--tags-dir", "tags", "--out", str(out)])
    d = out / arr["arrangement_id"]
    rec = json.loads((d / "render.json").read_text())
    settings = dict(rec["settings"])
    settings["isaacsim"] = MASK
    lab = json.loads((d / "labels.json").read_text())
    for k in ("isaac_batch_sha256", "isaac_rig_sha256"):
        lab["hashes"][k] = MASK
    lab["hashes"]["render_settings_sha256"] = MASK      # = sha256 of the settings (isaacsim included), pinned above
    png = {str(Path(k).relative_to(B.partial_dir(out, arr["arrangement_id"]))): v for k, v in FI.PNG_TEXT.items()}
    return {"captures": {"n": len(FI.STATE["captures"]), "sha256": sha256_json(FI.STATE["captures"])},
            "png_meta_sha256": sha256_json(png), "render_settings_sha256": sha256_json(settings),
            "labels_sha256": sha256_json(lab), "file_sha256": sha256_file(f)}


class _Rec:
    """A stand-in for every pxr / omni object: attribute access names a path, a call appends [callee, args, kwargs] to the
    shared log and returns a new object named after the call's index ('#12'), so later calls on it reference that index."""

    def __init__(self, log: list, name: str):
        self._log, self._name = log, name

    def __getattr__(self, k: str):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Rec(self._log, f"{self._name}.{k}")

    def __call__(self, *args, **kwargs):
        self._log.append([self._name, [_canon(a) for a in args], {k: _canon(v) for k, v in sorted(kwargs.items())}])
        return _Rec(self._log, f"#{len(self._log) - 1}")


def _canon(v):
    if isinstance(v, _Rec):
        return v._name
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 9)
    if isinstance(v, np.ndarray):
        return _canon(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, Path):
        return str(v)
    return repr(v)


def _fake_ply(path) -> dict:
    """A stand-in for ir.read_ply: one tetrahedron in mm, with a texture name (the real BOP models' layout)."""
    return {"vertices": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
            "normals": None, "uv": None, "faces": np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
            "texture": f"{Path(path).stem}.png"}


def stage_log(arr: Mapping) -> list:
    """Every pxr / omni call sim/isaac_batch.py build_stage makes for `arr` (then set_lighting L2 and back to L1), recorded
    with _Rec stand-ins; the PLY reader is replaced by _fake_ply, assets_dir is the fixed '/assets'."""
    from sim import isaac_batch as B

    log: list = []
    mods = {"pxr": _Rec(log, "pxr"), "omni": _Rec(log, "omni"), "omni.usd": _Rec(log, "omni.usd"), "carb": _Rec(log, "carb")}
    with mock.patch.dict(sys.modules, mods), mock.patch.object(B.ir, "read_ply", _fake_ply):
        built = B.build_stage(json.loads(json.dumps(arr)), Path("/assets"), Path("/tags"), {})
        stage = _Rec(log, "stage")
        B.ir.set_lighting(stage, built, "L2")
        B.ir.set_lighting(stage, built, "L1")
        log.append(["built", _canon({k: v for k, v in built.items() if k != "occluder_op"}), {}])
    return log


def stage_log_hash(arr: Mapping) -> dict:
    log = stage_log(arr)
    return {"n_calls": len(log), "sha256": sha256_json(log),
            "defined_paths_sha256": sha256_json([e[1][1] for e in log if e[0].endswith(".Define") and len(e[1]) > 1])}
