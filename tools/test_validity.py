"""The frozen test's validity requirements (docs/FREEZE.md section 3), checked on operational outputs before any outcome
is read. Read-only.

It reads:
- the committed FREEZE.md and its record (sim/freeze.py require_freeze);
- the generated arrangements and their index (sim/testsplit.py --out);
- each rendered arrangement's labels.json hashes;
- the set's manifest and set.json;
- the evidence job's provenance;
- verify.run's config.json;
- optionally, a `sha256sum` listing of the GB10 checkpoints.

It never reads an evidence score, a decision or a label value. Each group runs only when its input is given; a given
input that is missing or unreadable fails. Exit 0 when every check that ran passed, 1 otherwise, 2 when FREEZE.md itself
does not pass.

    uv run --offline --no-project --python 3.13.5 --with numpy==2.5.3 --with opencv-python-headless==5.0.0.93 \\
      python tools/test_validity.py --freeze docs/FREEZE.md --env shelf \\
        --geometry <share>/sim/shelf_test_geometry/arr --set <share>/sim/shelf_test \\
        [--evidence-provenance runs/shelf_test/evidence_provenance.json] [--verify-run runs/<date>-test/shelf] \\
        [--gb10-checkpoints gb10_checkpoints.sha256] [--out report.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sim import arrange as A  # noqa: E402
from sim import freeze as F  # noqa: E402
from sim import testsplit as TS  # noqa: E402

CONFIGS = {"shelf": "configs/sim/arrange_shelf_v3.json", "tabletop": "configs/sim/arrange_tabletop_v1.json"}
POSITIONS = {"shelf": "configs/sim/positions_shelf_A.json", "tabletop": "configs/sim/positions_tabletop_A.json"}
# The development sets' render-settings hash per environment (every random development arrangement, labels.json), which
# the test renders must reproduce (notes/2026-11-06-pooled-calibration.md; FREEZE.md section 4).
RENDER_SETTINGS = {"shelf": "a1faaceb1a240293ff56a2695232f90fe87201d5aaabe0fcbc4a8a69afc1a954",
                   "tabletop": "09244c0c8a80c91766c428409913a611d6b3585f495a4476a547f97aabc984d6"}
# Evidence model files on the GB10 (checkpoints dir; FREEZE.md section 1). DINOv2 is pinned by revision in
# perception/evidence.py, which the record covers.
CHECKPOINTS = {"yolov8s-worldv2.pt": "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792",
               "clip/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"}
EVIDENCE_FROZEN = ("perception/evidence_v03.py", "perception/evidence.py", "perception/common.py",
                   "perception/side_descriptor.py", "perception/registration_v032.py", "verify/schema.py",
                   "verify/stopping.py", "pyproject.toml", "uv.lock")   # = tools/pipeline.py EVIDENCE_FROZEN
RUN_CONFIG_VERSION = "real-v6"   # = verify/run.py REAL_CONFIG_VERSION


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, group: str, name: str, ok: bool, detail: Any = None) -> None:
        self.checks.append({"group": group, "check": name, "ok": bool(ok), "detail": detail})

    def run(self, group: str, name: str, fn: Callable[[], tuple[bool, Any]]) -> None:
        try:
            ok, detail = fn()
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:   # a given input that cannot be read fails
            ok, detail = False, f"{type(e).__name__}: {e}"
        self.add(group, name, ok, detail)

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)


def _all(failures: list) -> tuple[bool, Any]:
    return (not failures, failures[:10] or None)


def check_generation(rep: Report, rec: dict, freeze_sha: str, env: str, geometry: Path,
                     ctx: A.Context | None = None) -> dict[str, dict]:
    """The generated arrangements; returns the index rows by arrangement id. ctx: the frozen config's context (tests
    pass an in-memory synthetic block)."""
    g = "generation"
    files = rec["files"]
    rep.run(g, "config is the frozen one", lambda: (sha(REPO / CONFIGS[env]) == files.get(CONFIGS[env]), CONFIGS[env]))
    rows = {r["arrangement_id"]: r for r in json.loads((geometry / "index" / "index.json").read_text())["arrangements"]}
    ctx = ctx or A.load_context(REPO / CONFIGS[env])
    block = set(TS.test_block_seeds(ctx))
    slots = json.loads((geometry / "index" / "slots.json").read_text())["slots"]
    expected = len(A.slot_list(ctx, "test"))
    rep.add(g, "one arrangement per test slot", len(rows) == len(slots) == expected,
            {"arrangements": len(rows), "slots": len(slots), "expected": expected})
    bad: dict[str, list] = {k: [] for k in ("file", "split", "seed", "hash", "freeze", "slot")}
    seen_slots = []
    for aid, r in sorted(rows.items()):
        p = geometry / r["file"]
        if sha(p) != r["sha256"]:
            bad["file"].append(aid)
        arr = json.loads(p.read_text())
        if (arr.get("split"), arr.get("kind"), arr.get("environment")) != ("test", "random", env):
            bad["split"].append(aid)
        if arr.get("seed") not in block:
            bad["seed"].append(aid)
        if not (arr.get("arrangement_sha256") == A.arrangement_hash(arr) == r["arrangement_sha256"]):
            bad["hash"].append(aid)
        fz = arr.get("freeze") or {}
        if (fz.get("sha256"), fz.get("freeze_id"), fz.get("testsplit_sha256")) != \
                (freeze_sha, rec["freeze_id"], rec["files"].get("sim/testsplit.py")):
            bad["freeze"].append(aid)
        sl = arr.get("slot") or {}
        if (sl.get("split"), sl.get("family"), sl.get("m16_side")) != ("test", arr.get("occluder_family"),
                                                                       arr.get("m16_side")):
            bad["slot"].append(aid)
        seen_slots.append(sl.get("index"))
    count = Counter(seen_slots)
    off = sorted((s for s in count if s not in range(expected) or count[s] > 1), key=str)
    bad["slot"] += [f"slot {s}" for s in off]
    for k, v in bad.items():
        rep.add(g, {"file": "each file has its index sha256", "split": "split test, kind random, this environment",
                    "seed": "each seed is in the config's test block", "hash": "arrangement_sha256 recomputes",
                    "freeze": "each records this FREEZE.md, its freeze_id and the frozen testsplit",
                    "slot": "each is a test slot of its family, and no slot twice"}[k], not v, v or None)
    return rows


def check_render(rep: Report, rec: dict, env: str, rows: dict[str, dict], set_root: Path) -> None:
    g = "render"
    files = rec["files"]
    present = {p.parent.name for p in set_root.glob("*/labels.json")}
    rep.add(g, "the set holds exactly the generated arrangements", present == set(rows),
            {"missing": sorted(set(rows) - present)[:10], "extra": sorted(present - set(rows))[:10]})
    bad: dict[str, list] = {k: [] for k in ("arrangement", "renderer", "settings")}
    for aid in sorted(present & set(rows)):
        h = json.loads((set_root / aid / "labels.json").read_text()).get("hashes", {})
        if (h.get("arrangement_file_sha256"), h.get("arrangement_sha256")) != \
                (rows[aid]["sha256"], rows[aid]["arrangement_sha256"]):
            bad["arrangement"].append(aid)
        if (h.get("isaac_batch_sha256"), h.get("isaac_rig_sha256")) != \
                (files.get("sim/isaac_batch.py"), files.get("sim/isaac_rig.py")):
            bad["renderer"].append(aid)
        if h.get("render_settings_sha256") != RENDER_SETTINGS[env]:
            bad["settings"].append(aid)
    rep.add(g, "each render is of its generated arrangement file", not bad["arrangement"], bad["arrangement"] or None)
    rep.add(g, "each render ran the frozen isaac_batch.py and isaac_rig.py", not bad["renderer"],
            bad["renderer"] or None)
    rep.add(g, "each render has the development render settings", not bad["settings"], bad["settings"] or None)

    def manifest() -> tuple[bool, Any]:
        m = json.loads((set_root / "manifest.json").read_text())
        arrs = m["arrangements"]
        off = [a["arrangement_id"] for a in arrs if (a.get("split"), a.get("environment")) != ("test", env)]
        ids = {a["arrangement_id"] for a in arrs}
        return (not off and ids == set(rows), {"not test/this environment": off[:10], "ids_equal": ids == set(rows)})
    rep.run("episodes", "manifest: the generated arrangements, split test, this environment", manifest)
    rep.run("episodes", "set.json: split test, this environment", lambda: (
        (lambda s: ((s.get("split"), s.get("environment")) == ("test", env), {k: s.get(k) for k in ("split",
                                                                                                    "environment")}))
        (json.loads((set_root / "set.json").read_text()))))


def check_evidence(rep: Report, rec: dict, set_root: Path, prov_path: Path) -> dict[str, dict]:
    """The evidence job; returns its outputs {path: {records, sha256}}."""
    g = "evidence"
    files = rec["files"]
    prov = json.loads(prov_path.read_text())
    rep.add(g, "the GB10 env's uv.lock is the frozen one", prov["gb10_env"].get("uv_lock_sha256") == files.get("uv.lock"),
            prov["gb10_env"].get("uv_lock_sha256"))

    def shipped() -> tuple[bool, Any]:
        bad = []
        for rel in EVIDENCE_FROZEN:
            res = subprocess.run(["git", "-C", str(REPO), "show", f"{prov['code_commit']}:{rel}"], capture_output=True)
            if res.returncode != 0 or hashlib.sha256(res.stdout).hexdigest() != files.get(rel):
                bad.append(rel)
        return _all(bad)
    rep.run(g, "the shipped code commit holds the frozen evidence code", shipped)
    rep.add(g, "the job ran on this set", Path(prov["set_root"]) == set_root, prov["set_root"])
    n_jobs = len(json.loads((set_root / "jobs_refL1.json").read_text())["jobs"])
    outs = prov["outputs"]
    wanted = ("v03_image_refL1.jsonl", "c0_refL1.jsonl", "side_refL1.jsonl")
    got = {w: [v for k, v in outs.items() if Path(k).name == w] for w in wanted}
    rep.add(g, "every primary output has one record per L1-reference job",
            all(len(v) == 1 and v[0]["records"] == n_jobs for v in got.values()),
            {w: [x["records"] for x in v] for w, v in got.items()} | {"jobs": n_jobs})
    return outs


def check_run(rep: Report, rec: dict, env: str, set_root: Path, run_dir: Path, outs: dict[str, dict] | None) -> None:
    g = "verify.run"
    files = rec["files"]
    c = json.loads((run_dir / "config.json").read_text())["config"]
    want = {
        "config_version": RUN_CONFIG_VERSION, "environment": env, "scoring_split": "test", "model_id": "v03_image",
        "bootstrap_B": 10000, "candidate_sha256": files.get("frozen/frozen_candidate.json"),
        "positions_file_sha256": files.get(POSITIONS[env]),
        "manifest_sha256": sha(set_root / "manifest.json"), "view_labels_sha256": sha(set_root / "view_labels.json"),
    }
    off = {k: {"run": c.get(k), "want": v} for k, v in want.items() if c.get(k) != v}
    rep.add(g, "frozen candidate, positions, set files and settings", not off, off or None)
    models = c.get("ranker_models") or {}
    mod = {n: (models.get(n) or {}).get("sha256") for n in ("A", "A-nocue")}
    rep.add(g, "the frozen A and A-nocue model files", mod == {"A": files.get("frozen/model_A.json"),
                                                              "A-nocue": files.get("frozen/model_A-nocue.json")}, mod)
    cand = json.loads((REPO / "frozen" / "frozen_candidate.json").read_text())["thresholds"]
    th = {k: c["thresholds"].get(k) for k in ("tau_q", "tau_occ", "tau_match", "k", "lam")}
    rep.add(g, "the frozen thresholds", th == {k: cand[k] for k in th}, th)
    if outs is not None:
        by_name = {Path(k).name: v["sha256"] for k, v in outs.items()}
        ev = {"evidence_sha256": by_name.get("v03_image_refL1.jsonl"), "side_file_sha256": by_name.get("side_refL1.jsonl")}
        off = {k: {"run": c.get(k), "provenance": v} for k, v in ev.items() if c.get(k) != v}
        rep.add(g, "the evidence of record and the descriptors of this set's evidence job", not off, off or None)


def check_checkpoints(rep: Report, listing: Path) -> None:
    got = {}
    for line in listing.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            got[parts[1].strip().lstrip("*").removeprefix("./")] = parts[0]
    off = {k: got.get(k) for k, v in CHECKPOINTS.items() if got.get(k) != v}
    rep.add("checkpoints", "the GB10 detector and CLIP weights", not off, off or None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--freeze", required=True, help="the committed docs/FREEZE.md")
    ap.add_argument("--env", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--geometry", type=Path, help="sim.testsplit --out (the generated arrangements and index/)")
    ap.add_argument("--set", type=Path, help="the rendered set root (episodes built)")
    ap.add_argument("--evidence-provenance", type=Path, help="the pipeline run dir's evidence_provenance.json")
    ap.add_argument("--verify-run", type=Path, help="the verify.run output dir")
    ap.add_argument("--gb10-checkpoints", type=Path, help="`sha256sum` output, run in the GB10 checkpoints dir")
    ap.add_argument("--out", type=Path, help="write the report here (JSON)")
    a = ap.parse_args(argv)
    try:
        fz = F.require_freeze(a.freeze)
    except F.FreezeError as e:
        print(f"FREEZE.md does not pass: {e}", file=sys.stderr)
        return 2
    rec = fz["record"]
    rep = Report()
    rows = None
    if a.geometry:
        try:
            rows = check_generation(rep, rec, fz["sha256"], a.env, a.geometry)
        except (OSError, ValueError, KeyError) as e:
            rep.add("generation", "the generated set is readable", False, f"{type(e).__name__}: {e}")
    if a.set:
        if rows is None:
            rep.add("render", "needs --geometry (the index)", False)
        else:
            check_render(rep, rec, a.env, rows, a.set)
    outs = None
    if a.evidence_provenance:
        if not a.set:
            rep.add("evidence", "needs --set", False)
        else:
            try:
                outs = check_evidence(rep, rec, a.set, a.evidence_provenance)
            except (OSError, ValueError, KeyError) as e:
                rep.add("evidence", "the provenance is readable", False, f"{type(e).__name__}: {e}")
    if a.verify_run:
        if not a.set:
            rep.add("verify.run", "needs --set", False)
        else:
            try:
                check_run(rep, rec, a.env, a.set, a.verify_run, outs)
            except (OSError, ValueError, KeyError) as e:
                rep.add("verify.run", "the run config is readable", False, f"{type(e).__name__}: {e}")
    if a.gb10_checkpoints:
        try:
            check_checkpoints(rep, a.gb10_checkpoints)
        except OSError as e:
            rep.add("checkpoints", "the listing is readable", False, str(e))
    doc = {"freeze": {"path": fz["path"], "sha256": fz["sha256"], "freeze_id": rec["freeze_id"]}, "env": a.env,
           "ok": rep.ok, "checks": rep.checks}
    if a.out:
        a.out.write_text(json.dumps(doc, indent=1) + "\n")
    for c in rep.checks:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['group']}: {c['check']}"
              + ("" if c["ok"] or c["detail"] is None else f"  {json.dumps(c['detail'])[:300]}"))
    print(f"{'ALL PASS' if rep.ok else 'FAILED'}: {sum(c['ok'] for c in rep.checks)} / {len(rep.checks)} checks")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
