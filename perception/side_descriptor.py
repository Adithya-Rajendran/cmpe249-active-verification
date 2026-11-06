"""Left/right obstruction descriptor of one requested view (DEVELOPMENT; notes/2026-11-05-side-descriptor.md).

Why: under a tight travel cap a policy has to choose a side of the arc, and v0.3's obstruction is a single number (how
much of the expected box an occluding surface covers), not on which side of the target that surface lies. This
descriptor splits v0.3's own obstruction map into the left and right parts of the target's context window. It is an
input of ranker A and of the direction-rule control only: it never enters the stopping rule, v0.3's scores or the
evidence of record, and a policy sees it only for views it has requested (verify/side.py).

Definition, fixed in the commit that adds this file, before the descriptor was computed on any rendered view:
  registration  C0 unchanged (notes/2026-11-04-registration-repair-plan.md, Amendment 2): v0.3 `image_align` with its
                default AlignParams, the query resampled into reference pixels by v0.2.1's `register`, as the evidence
                step does it. A refused registration gives NO descriptor (missing: "registration refused"), never 0.
  map           v0.3's background-obstruction map with its own steps and parameters (`v03.background_obstruction`):
                v0.2.1's change_sigma, change_delta, search_pad and context window grow(box, context = 1.0), v0.3's
                robust gain and its row-wise prediction of the background behind the target. A window pixel is
                OBSTRUCTED when it changed against the gain-corrected reference AND is not the expected background
                (inside the search box the row-interpolated reference, outside it the reference itself, so there
                obstructed = changed). Invalid pixels (registered from outside the query frame) are never obstructed.
  split         the window's pixel columns whose centre lies left of the box centre, x + 0.5 < (x0 + x1) / 2 in
                reference pixels, and the rest.
  shares        u_left, u_right = obstructed / VALID pixels of each part.
  support       missing ("support") when either part has fewer than half of its pixels valid.
  descriptor    d = u_left - u_right, in [-1, 1].
The in-box share of the same map is recomputed for every view and must equal v0.3's background_obstruction.

Reading (camera geometry and the generator's design, not data): image +x points toward increasing azimuth at every stop
of both environments (sim/rig.py cameras of the pinned positions files), and an occluding surface in front of the
target hides the stops toward its own displaced side (parallax; the generator's M16 side rule, sim/arrange.py). So
d > 0 (more obstruction left of the target) points to the stops at larger azimuth, d < 0 to smaller azimuth. It
describes observed change, not a measured occlusion mask and not proof that the target is visible from anywhere.

Usage (repo root; numpy and OpenCV only, no detector or encoder):
    uv run --no-sync python -m perception.side_descriptor --jobs JOBS.json --out side.jsonl [--workers N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from perception import evidence as e021
from perception import evidence_v03 as v03

DESCRIPTOR_VERSION = "side-0.1-dev"
SUPPORT_MIN = 0.5     # a part with fewer valid pixels than this share gives no descriptor
SCHEMA = "cmpe249.side-descriptor.v0"
Box = e021.Box


def obstruction_map(ref_bgr: np.ndarray, qry_bgr: np.ndarray, valid: np.ndarray, box: Box,
                    p: e021.Params = e021.Params(), v: v03.V03Params = v03.V03Params()
                    ) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """(window, obstructed, valid, inbox), the last three in window pixels: the steps of v03.background_obstruction,
    unchanged. `qry_bgr` is the query already registered into reference pixels, `valid` its fill mask."""
    h, w = ref_bgr.shape[:2]
    gr = cv2.GaussianBlur(ref_bgr.astype(np.float32), (0, 0), p.change_sigma)
    gq = cv2.GaussianBlur(qry_bgr.astype(np.float32), (0, 0), p.change_sigma)
    g = v03.robust_gain(gr, gq, valid, box, p, v)
    win = e021.grow(box, p.context, w, h)
    sl = (slice(win[1], win[3]), slice(win[0], win[2]))
    r, q, val = gr[sl], gq[sl] * g, valid[sl]
    inbox = e021.box_mask(box, win)
    sx0, sy0, sx1, sy1 = v03.search_box(box, w, h, p)
    hole = (sx0 - win[0], sy0 - win[1], sx1 - win[0], sy1 - win[1])
    changed = (np.abs(q - r).max(axis=2) > p.change_delta) & val
    bg = v03.row_interpolate(r, hole, v.edge_px)
    not_bg = np.abs(q - bg).max(axis=2) > p.change_delta
    return win, changed & not_bg, val, inbox


def descriptor_registered(ref_bgr: np.ndarray, qry_bgr: np.ndarray, valid: np.ndarray, box: Box,
                          p: e021.Params = e021.Params(), v: v03.V03Params = v03.V03Params()) -> dict[str, Any]:
    """The descriptor of a query already registered into reference pixels (see the module docstring)."""
    box = tuple(float(x) for x in box)
    win, obs, val, inbox = obstruction_map(ref_bgr, qry_bgr, valid, box, p, v)
    x = np.arange(win[0], win[2], dtype=np.float64) + 0.5
    left = np.broadcast_to((x < (box[0] + box[2]) / 2)[None, :], obs.shape)
    out: dict[str, Any] = {"window": list(win),
                           "background_obstruction": float(obs[inbox].mean()) if inbox.any() else 0.0}
    shares = {}
    for name, part in (("left", left), ("right", ~left)):
        n, nv = int(part.sum()), int((part & val).sum())
        out[f"valid_{name}"] = nv / n if n else 0.0
        shares[name] = float((obs & part).sum()) / nv if nv else None
    if min(out["valid_left"], out["valid_right"]) < SUPPORT_MIN:
        return {**out, "u_left": None, "u_right": None, "d": None, "missing": "support"}
    return {**out, "u_left": shares["left"], "u_right": shares["right"], "d": shares["left"] - shares["right"],
            "missing": None}


def view_descriptor(ref_bgr: np.ndarray, qry_bgr: np.ndarray, box: Box, ap: v03.AlignParams = v03.AlignParams(),
                    p: e021.Params = e021.Params(), v: v03.V03Params = v03.V03Params()) -> dict[str, Any]:
    """C0 registration, then the descriptor; no descriptor when C0 refuses. Checks the in-box share against
    v0.3's background_obstruction on the same registration."""
    box = tuple(float(x) for x in box)
    H, info = v03.image_align(ref_bgr, qry_bgr, box, ap)
    if H is None:
        return {"d": None, "u_left": None, "u_right": None, "missing": "registration refused", "registration": info,
                "H": None}
    qry, valid, _ = e021.register(ref_bgr, qry_bgr, lambda r, q: H)
    out = descriptor_registered(ref_bgr, qry, valid, box, p, v)
    u_v03, _ = v03.background_obstruction(ref_bgr, qry, valid, box, p, v)
    if out["background_obstruction"] != u_v03:
        raise AssertionError(f"in-box share {out['background_obstruction']} != v0.3 background_obstruction {u_v03}")
    return {**out, "registration": info, "H": H.tolist()}


# ---------------------------------------------------------------- jobs

_ANNS: dict[str, dict] = {}
_REFS: dict[str, np.ndarray] = {}


def job_descriptor(j: dict) -> dict[str, Any]:
    """One jobs-file entry (cmpe249.evidence-jobs.v0) -> one output row: the job's keys and the descriptor."""
    ann = _ANNS.setdefault(j["annotation"], e021.load_annotation(j["annotation"]))
    if int(ann["stop"]) != int(j["stop"]):
        raise ValueError(f"job stop {j['stop']} != annotation stop {ann['stop']} ({j['annotation']})")
    ref = _REFS.setdefault(ann["reference"], e021.read_bgr(ann["reference"]))
    d = view_descriptor(ref, e021.read_bgr(j["query"]), ann["targets"][j["target"]]["box"])
    return {**j, "descriptor_version": DESCRIPTOR_VERSION, **d}


def _init_worker() -> None:
    cv2.setNumThreads(1)


def run_jobs(jobs: Sequence[dict], workers: int = 1) -> list[dict[str, Any]]:
    """Rows in the jobs' order. workers > 1: spawned processes, order kept (imap)."""
    if workers <= 1:
        return [job_descriptor(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        return list(pool.imap(job_descriptor, jobs, chunksize=4))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perception.side_descriptor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True, help="one JSON row per job (schema, the job's keys, the descriptor)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    doc = json.loads(e021._path(args.jobs).read_text())
    if doc.get("schema") != e021.JOBS_SCHEMA:
        print(f"{args.jobs}: schema is {doc.get('schema')!r}, expected {e021.JOBS_SCHEMA!r}", file=sys.stderr)
        return 2
    rows = run_jobs(doc["jobs"], args.workers)
    keys = [(r["episode_id"], int(r["stop"])) for r in rows]
    if len(set(keys)) != len(keys):
        print("duplicate (episode_id, stop) in the jobs", file=sys.stderr)
        return 2
    with open(args.out, "w") as f:      # one row per job and nothing else (tools/pipeline.py counts the lines)
        for r in rows:
            f.write(json.dumps({"schema": SCHEMA, **r}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
