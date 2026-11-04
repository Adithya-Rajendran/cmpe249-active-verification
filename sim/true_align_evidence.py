"""EVALUATION ONLY: evidence v0.3 with the simulator's TRUE query -> reference homography instead of image registration.

The homographies come from true_alignment*.json (cmpe249.sim-true-alignment.v1), computed from the renderer's camera
poses, on the shelf front plane (y = 0) or the target's plane; for a tabletop set (true_alignment*.json with
"environment": "tabletop", sim/alignment.py) on the table plane (z = 0) or the target's horizontal plane (z = the target's
centre height). `--plane` is checked against the alignment document: front|target for a shelf set, table|target for a
tabletop set. This is never an evidence step a robot could run and
its output is never a policy or calibration input. It separates registration error from cue error, and in particular
measures how much v0.2.1's quality term (sharpness on a nearest-neighbour resampled ring) moves when the warp moves
by a pixel or two (notes/2026-11-03-evidence-v03-align.md). Each record is perception.evidence_v03.evidence_for,
unchanged, given the true H; the job loop repeats evidence_v03.run_jobs's checks (job stop = annotation stop, the same
job path resolution) and the diagnostics record the plane.

Usage (repo root, GB10 project env):
    python -m sim.true_align_evidence --set SET_DIR --jobs JOBS.json --plane target|front|table --out OUT.jsonl \\
        --diag DIAG.jsonl [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from perception import evidence as e021
from perception import evidence_v03 as v03
from verify.schema import EvidenceStore

TRUE_ALIGNMENT_FILES = ("true_alignment.json", "true_alignment_design.json")
PLANES = {"shelf": ("front", "target"), "tabletop": ("table", "target")}     # by the alignment document's "environment" (absent: shelf)


def load_true_h(set_dir: Path, plane: str) -> dict[tuple[str, int], np.ndarray]:
    """(arrangement_id, stop index) -> the true query -> reference homography on `plane`."""
    out, envs = {}, set()
    for name in TRUE_ALIGNMENT_FILES:
        path = set_dir / name
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        env = doc.get("environment", "shelf")
        if plane not in PLANES.get(env, ()):
            raise ValueError(f"{path}: --plane {plane!r} is not a plane of a {env} set (choose from {PLANES.get(env, ())})")
        envs.add(env)
        if len(envs) > 1:
            raise ValueError(f"{set_dir}: true-alignment files of different environments {sorted(envs)}")
        for arr, a in doc["arrangements"].items():
            for s in a["stops"].values():
                out[(arr, int(s["stop_index"]))] = np.asarray(s[f"H_{plane}_plane"], np.float64)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sim.true_align_evidence", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, type=Path, help="set root holding true_alignment*.json")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--plane", required=True, choices=("target", "front", "table"),
                    help="front|target for a shelf set, table|target for a tabletop set (checked against the alignment document)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    doc = json.loads(e021._path(args.jobs).read_text())
    if doc.get("schema") != e021.JOBS_SCHEMA:
        print(f"{args.jobs}: schema is {doc.get('schema')!r}, expected {e021.JOBS_SCHEMA!r}", file=sys.stderr)
        return 2
    true_h = load_true_h(args.set, args.plane)
    enc, det = e021.DinoV2Patches(args.device), e021.YoloWorld(args.device)
    store = EvidenceStore(synthetic=False, source=f"sim.true_align_evidence: evidence_v03 {v03.EVIDENCE_VERSION} on the "
                                                  f"TRUE {args.plane}-plane alignment (evaluation only)")
    anns: dict[str, dict] = {}
    refs: dict[str, np.ndarray] = {}
    with open(args.diag, "w") as fh:
        for j in doc["jobs"]:
            ann = anns.setdefault(j["annotation"], e021.load_annotation(j["annotation"]))
            if int(ann["stop"]) != int(j["stop"]):
                raise ValueError(f"job stop {j['stop']} != annotation stop {ann['stop']} ({j['annotation']})")
            ref = refs.setdefault(ann["reference"], e021.read_bgr(ann["reference"]))
            key = (j["episode_id"].split("-")[0], int(j["stop"]))
            if key not in true_h:
                raise KeyError(f"no true alignment for {key} in {args.set}/{TRUE_ALIGNMENT_FILES}")
            H = true_h[key]
            rec, diag = v03.evidence_for(ref, ann["targets"][j["target"]]["box"], e021.read_bgr(j["query"]),
                                         int(j["stop"]), enc, det, align=lambda r, q, H=H: H)
            diag["registration"] = {"method": f"true-{args.plane}-plane (evaluation only)"}
            store.put(j["episode_id"], rec)
            fh.write(json.dumps({**j, "match_score": rec.match_score, "quality": rec.quality,
                                 "obstruction": rec.obstruction, "evidence_version": v03.EVIDENCE_VERSION,
                                 "flags_placeholder_thresholds": e021.flags(rec), **diag}) + "\n")
    store.to_jsonl(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
