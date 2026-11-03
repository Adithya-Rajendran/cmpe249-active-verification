"""Per-view evidence v0.3 (DEVELOPMENT, not frozen): v0.2.1 plus a background-consistency occlusion cue.

v0.2.1 (perception/evidence.py, frozen, sha256 d0e0927f...) stays the baseline and is imported read-only:
match_score, quality and the v0.2.1 obstruction come from its own `evidence_for`, unchanged. v0.3 only adds
one obstruction cue and combines it by max:

    obstruction_v03 = max(obstruction_v0.2.1, background_obstruction)

Why (notes/2026-11-03-evidence-v03-dev.md). On the rendered check batch, 13 of 22 fully hidden non-diagnostic
occluded views passed the gate with v0.2.1 obstruction 0.00-0.10. Their box pixels DID change against the
reference (0.80-1.00 of the box), but neither v0.2.1 cue turned that into obstruction: the detector found no
box for a plain board, and DINOv2 novelty was low because a plain white board is explained by the plain shelf
surfaces in the reference window. A change in the box alone cannot be the cue: a removed target changes the
box as much. What separates them is WHAT the box shows now:
  - removed: the background that was behind the target, which continues the background visible around it in
    the reference;
  - occluded: a surface in front, which is not that background (a lit board in front of a shadowed shelf).

background_obstruction: share of the expected box whose pixels
  (a) CHANGED against the reference (max over B, G, R of |gain * query - reference| > change_delta, both
      Gaussian-blurred with change_sigma, as in v0.2.1), AND
  (b) are NOT the expected background: they differ by more than change_delta from a prediction of the
      background behind the target made from the REFERENCE alone, by row-wise linear interpolation across
      the search box (grow(box, search_pad)) between the mean of `edge_px` reference pixels just left and
      just right of it.
Only the reference image and box, the query and the tag registration are used. No mask, label, geometry or
phase is read.

Choices, each fixed once and written down (no knob was swept over a range; the alternatives that were tried
on the check batch, dev data, are recorded in the dev note with their numbers):
  - change_delta, change_sigma, search_pad: v0.2.1's own values, reused, not re-tuned.
  - Gain: per-channel MEDIAN of reference / query over grow(box, gain_context) minus the search box, valid
    pixels only. v0.2.1 uses the MEAN over the ring of its context window; an occluder covering most of that
    ring pulls it (up to 28 % from the visible-condition gain on the check batch), while the median over a
    window twice as wide stayed within 1.1 % (L1) and 6.2 % (L2). gain_context = 2.0 is one value, chosen
    because a board several times the target's size still covers a minority of that window.
  - Row-wise interpolation, not cv2.inpaint (Telea) and not column-wise: a shelf's background (floor / back-wall
    junction, shelf lips) runs across the target horizontally. Telea filled the 180 x 165 px holes with an
    X-shaped artefact (removed diagnostic views read up to 0.70); column-wise interpolation crosses the
    floor / wall junction (removed diagnostic median 0.37).
  - edge_px = 4: averages a little sensor / render noise at the interpolation ends; not tuned.
  - Invalid query pixels (registered from outside the query frame) never count as obstruction; v0.2.1's
    quality already counts them through `valid`.

Known limits (check batch, 2026-11-03; see the dev note):
  - a background that is not horizontally continuous across the target (a vertical divider edge just behind
    it, a different object right beside it in the reference) is predicted wrongly, and a REMOVED target then
    reads as obstructed (4 of 44 removed diagnostic views read 0.71-0.76: the safe direction, but it costs
    removal recall);
  - an occluder whose colour, after gain, is within change_delta of the background behind the target (a grey
    board in front of a grey wall) is missed exactly like v0.2.1 misses it;
  - removing a target also removes its contact shadow, which brightens the floor right below it; with the
    blur and change_delta that read at most 0.18 on the other 40 removed diagnostic views;
  - registration is a front-plane homography; parallax of deeper structure is left to the blur.

Usage (repo root), the same CLI shape as perception/evidence.py:
  uv run --no-sync python -m perception.evidence_v03 --jobs JOBS.json --out evidence.jsonl \\
      [--diag diagnostics.jsonl] [--device cuda]
The diagnostics add `obstruction_v021` and `v03_parts` (background_obstruction, box_changed, ring_changed,
gain) to every v0.2.1 diagnostic field. ring_changed (share of the context ring, outside the search box, that
changed under the robust gain) is the ring-change cue that was tried and NOT adopted: an occluder standing
beside a removed target changes the ring too.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from perception import evidence as e021
from verify.schema import EvidenceRecord, EvidenceStore

EVIDENCE_VERSION = "0.3.0-dev"
Box = e021.Box


@dataclass(frozen=True)
class V03Params:
    """The v0.3 additions. Everything else is v0.2.1's `Params` (passed as `base`), unchanged."""
    gain_context: float = 2.0   # robust gain window = grow(box, gain_context) minus the search box
    edge_px: int = 4            # reference pixels averaged at each end of a row interpolation


# ---------------------------------------------------------------- pieces (pure numpy / cv2)

def search_box(box: Box, w: int, h: int, p: e021.Params = e021.Params()) -> tuple[int, int, int, int]:
    return e021.grow(box, p.search_pad, w, h)


def robust_gain(ref_blur: np.ndarray, qry_blur: np.ndarray, valid: np.ndarray, box: Box,
                p: e021.Params = e021.Params(), v: V03Params = V03Params()) -> np.ndarray:
    """Per-channel median of reference / query over grow(box, gain_context) minus the search box, valid pixels
    only. Multiplies the query. 1.0 where nothing is measurable."""
    h, w = ref_blur.shape[:2]
    m = np.zeros((h, w), bool)
    x0, y0, x1, y1 = e021.grow(box, v.gain_context, w, h)
    m[y0:y1, x0:x1] = True
    x0, y0, x1, y1 = search_box(box, w, h, p)
    m[y0:y1, x0:x1] = False
    m &= valid
    if not m.any():
        return np.ones(3, np.float32)
    g = [np.median(ref_blur[..., c][m] / np.maximum(qry_blur[..., c][m], 1.0)) for c in range(3)]
    return np.asarray(g, np.float32)


def row_interpolate(img: np.ndarray, hole: tuple[int, int, int, int], edge_px: int = 4) -> np.ndarray:
    """Copy of `img` (H x W x C float) with the rectangle hole = (x0, y0, x1, y1) replaced, row by row, by the
    straight line between the mean of the `edge_px` pixels just left of it and just right of it. A side cut off
    by the image edge uses the other side alone (a constant row)."""
    out = img.astype(np.float32, copy=True)
    x0, y0, x1, y1 = hole
    W = img.shape[1]
    if x1 <= x0 or y1 <= y0:
        return out
    lx = np.arange(max(0, x0 - edge_px), x0)
    rx = np.arange(x1, min(W, x1 + edge_px))
    if not len(lx) and not len(rx):
        return out                                          # nothing to interpolate from: leave it
    if not len(lx) or not len(rx):                          # one side cut off by the image edge: constant rows
        out[y0:y1, x0:x1] = out[y0:y1][:, lx if len(lx) else rx].mean(axis=1)[:, None, :]
        return out
    L, R = out[y0:y1][:, lx].mean(axis=1), out[y0:y1][:, rx].mean(axis=1)
    cl, cr = float(lx.mean()), float(rx.mean())             # the two means sit at these columns
    t = ((np.arange(x0, x1, dtype=np.float32) - cl) / np.float32(cr - cl))[None, :, None]
    out[y0:y1, x0:x1] = L[:, None, :] * (1 - t) + R[:, None, :] * t
    return out


def background_obstruction(ref_bgr: np.ndarray, qry_bgr: np.ndarray, valid: np.ndarray, box: Box,
                           p: e021.Params = e021.Params(),
                           v: V03Params = V03Params()) -> tuple[float, dict[str, Any]]:
    """(share of the box that changed AND is not the expected background, parts). `qry_bgr` is the query
    already registered into reference pixels, `valid` its fill mask (see e021.register)."""
    h, w = ref_bgr.shape[:2]
    gr = cv2.GaussianBlur(ref_bgr.astype(np.float32), (0, 0), p.change_sigma)
    gq = cv2.GaussianBlur(qry_bgr.astype(np.float32), (0, 0), p.change_sigma)
    g = robust_gain(gr, gq, valid, box, p, v)
    win = e021.grow(box, p.context, w, h)
    sl = (slice(win[1], win[3]), slice(win[0], win[2]))
    r, q, val = gr[sl], gq[sl] * g, valid[sl]
    inbox = e021.box_mask(box, win)
    sx0, sy0, sx1, sy1 = search_box(box, w, h, p)
    hole = (sx0 - win[0], sy0 - win[1], sx1 - win[0], sy1 - win[1])
    ring = ~e021.box_mask((sx0, sy0, sx1, sy1), win)
    changed = (np.abs(q - r).max(axis=2) > p.change_delta) & val
    bg = row_interpolate(r, hole, v.edge_px)
    not_bg = np.abs(q - bg).max(axis=2) > p.change_delta
    obstructed = changed & not_bg
    u = float(obstructed[inbox].mean()) if inbox.any() else 0.0
    parts = {"background_obstruction": u,
             "box_changed": float(changed[inbox].mean()) if inbox.any() else 0.0,
             "ring_changed": float(changed[ring].mean()) if ring.any() else 0.0,
             "gain": [round(float(x), 4) for x in g]}
    return u, parts


# ---------------------------------------------------------------- one view

def evidence_for(ref_bgr: np.ndarray, box: Box, query_bgr: np.ndarray, stop: int, encoder: e021.PatchEncoder,
                 detector: e021.Detector, p: e021.Params = e021.Params(), v: V03Params = V03Params(),
                 align: e021.Aligner = e021.tag_align) -> tuple[EvidenceRecord, dict[str, Any]]:
    """v0.2.1's record and diagnostics for this view, with obstruction replaced by max(v0.2.1, background cue).
    match_score and quality are v0.2.1's, computed by v0.2.1's own code on the same registration."""
    box = tuple(float(x) for x in box)
    H = align(ref_bgr, query_bgr)   # once, shared by v0.2.1 and the v0.3 cue
    rec021, diag = e021.evidence_for(ref_bgr, box, query_bgr, stop, encoder, detector, p, align=lambda r, q: H)
    qry, valid, _ = e021.register(ref_bgr, query_bgr, lambda r, q: H)
    u_bg, parts = background_obstruction(ref_bgr, qry, valid, box, p, v)
    rec = EvidenceRecord(stop, rec021.match_score, rec021.quality, max(rec021.obstruction, u_bg))
    diag = {**diag, "obstruction_v021": rec021.obstruction, "v03_parts": parts}
    return rec, diag


# ---------------------------------------------------------------- cli

def run_jobs(jobs: Sequence[dict], encoder: e021.PatchEncoder, detector: e021.Detector,
             p: e021.Params = e021.Params(), v: V03Params = V03Params(),
             align: e021.Aligner = e021.tag_align) -> tuple[EvidenceStore, list[dict[str, Any]]]:
    store = EvidenceStore(synthetic=False, source=f"perception.evidence_v03 {EVIDENCE_VERSION}")
    diags = []
    anns: dict[str, dict] = {}
    refs: dict[str, np.ndarray] = {}
    for j in jobs:
        ann = anns.setdefault(j["annotation"], e021.load_annotation(j["annotation"]))
        if int(ann["stop"]) != int(j["stop"]):
            raise ValueError(f"job stop {j['stop']} != annotation stop {ann['stop']} ({j['annotation']})")
        ref = refs.setdefault(ann["reference"], e021.read_bgr(ann["reference"]))
        rec, diag = evidence_for(ref, ann["targets"][j["target"]]["box"], e021.read_bgr(j["query"]), int(j["stop"]),
                                 encoder, detector, p, v, align)
        store.put(j["episode_id"], rec)
        diags.append({**j, **dataclasses.asdict(rec), "evidence_version": EVIDENCE_VERSION,
                      "flags_placeholder_thresholds": e021.flags(rec), **diag})
    return store, diags


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perception.evidence_v03", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True, help="evidence.jsonl (EvidenceStore format)")
    ap.add_argument("--diag", default=None, help="per-view diagnostics jsonl (v0.2.1 parts plus v03_parts)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    doc = json.loads(e021._path(args.jobs).read_text())
    if doc.get("schema") != e021.JOBS_SCHEMA:
        print(f"{args.jobs}: schema is {doc.get('schema')!r}, expected {e021.JOBS_SCHEMA!r}", file=sys.stderr)
        return 2
    store, diags = run_jobs(doc["jobs"], e021.DinoV2Patches(args.device), e021.YoloWorld(args.device))
    store.to_jsonl(args.out)
    if args.diag:
        with open(args.diag, "w") as f:
            for d in diags:
                f.write(json.dumps(d) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
