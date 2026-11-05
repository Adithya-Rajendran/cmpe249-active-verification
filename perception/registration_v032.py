"""Registration candidates for evidence 0.3.2-dev (DEVELOPMENT, not frozen; notes/2026-11-04-registration-repair-plan.md).

The plan is pinned (commit b144e04): candidates, rules and pass criteria were fixed before any of them was scored.
This module holds the candidates; v0.3 (perception/evidence_v03.py, 0.3.1-dev, `image_align` = C0) and the frozen v0.2.1
(perception/evidence.py) are imported unchanged. Only the homography leaves an aligner: the evidence step is
v0.3's `evidence_for`, unchanged, given that H (`align=lambda r, q: H`), so pixels left out of a fit still enter
v0.2.1's quality / match and v0.3's change / obstruction exactly as before. A refused view gets H = None, which the
evidence step scores as v0.3 scores a refused registration (identity, align_ok False, quality 0).

C1 = exactly the direction-corrected 2026-11-03 prototype (runs/2026-11-04-block-rejection-repair/align_lib.py,
sha256 689ad6d1...; `ecc_global(..., "homography", "edges", robust=True)`):
  pass 1  v0.3's edge images, fit mask (frame minus a 24 px border minus grow(box, 0.5)) and coarse-to-fine ECC
          (homography, levels 4 .. 1) from identity; NO refusal check on it, its warp is kept for the repair;
  repair  32 px (full resolution) = 16 px (level 1) blocks whose median |z(reference edges) - z(query edges warped
          into reference coordinates by W)| exceeds median + 4 max(MAD, 1e-3) are dropped; ECC is refitted at
          level 1 from the pass-1 warp on the retained mask;
  checks  once, on the final warp: v0.3's (refit cc >= 0.5, box centre shift <= 100 px, local scale within 0.1),
          and the support guard (retained >= 50 % of the pass-1 level-1 fit-mask pixels; >= 6 of the 9 cells of a
          3 x 3 grid over the level-1 frame keep >= 25 % of their pass-1 fit pixels). A cv2.error in either pass
          refuses the view (no fallback to the pass-1 warp).
Its outlier threshold is relative only, so it drops textured blocks of correct fits too (the known limit pinned in
runs/2026-11-04-block-rejection-repair/test_direction.py); that is part of C1 and is not changed.

C0 = v0.3 `image_align`, unchanged, run through this same harness so that C0 and a candidate are timed as comparable
jobs (plan, Amendment 1). Every diagnostic records `H` (the warp the evidence step used; None if refused) and `timing`: registration_s (the aligner call alone) and evidence_s
(the whole evidence step of the view, registration included), perf_counter. `--workers N` shards the jobs round-robin
over N spawned processes (each loads its own encoder and detector; cv2 one thread, torch / OMP `--torch-threads`);
the merged outputs are in the jobs file's order and must hold every (episode_id, stop) key of the jobs exactly once.
`--timing PATH` writes the batch record: settings, host, wall clock, views per hour.

Usage (repo root, GB10 project env), the CLI shape of perception/evidence_v03.py:
    python -m perception.registration_v032 --candidate C1 --jobs JOBS.json --out evidence.jsonl --diag diag.jsonl \
        [--workers 8] [--torch-threads 1] [--timing timing.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from perception import evidence as e021
from perception import evidence_v03 as v03
from verify.schema import EvidenceRecord, EvidenceStore

EVIDENCE_VERSION = "0.3.2-dev"
Box = e021.Box
AP = v03.AlignParams()          # v0.3's values; the final checks use min_cc, max_shift_px, max_scale_dev from it
SUPPORT_SHARE = 0.5             # retained / pass-1 fit-mask pixels (level 1)
CELL_SHARE, CELLS_MIN = 0.25, 6  # of the 3 x 3 cells over the level-1 frame


# ---------------------------------------------------------------- C1: the prototype's numerics, verbatim

def _prep(bgr, sigma, mode):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigma)
    if mode == "edges":
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        g = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), sigma)
    return g


def _scale(W, s):
    S = np.diag([s, s, 1.0]).astype(np.float64)
    return (S @ W.astype(np.float64) @ np.linalg.inv(S)).astype(np.float32)


def _ecc_pyr(tmpl, inp, mask, levels, W0, iters=200, eps=1e-6):
    """Coarse-to-fine homography ECC; W maps template (query) coords -> input (reference) coords. (cc, W) or cv2.error."""
    T, I, M = [tmpl], [inp], [mask]
    for _ in range(levels):
        T.append(cv2.pyrDown(T[-1])); I.append(cv2.pyrDown(I[-1]))
        M.append((cv2.resize(M[-1].astype(np.float32), (I[-1].shape[1], I[-1].shape[0]), interpolation=cv2.INTER_AREA) > 0.999).astype(np.uint8))
    W = _scale(W0, 0.5 ** levels)
    cc = float("nan")
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    for lev in range(levels, -1, -1):
        if int(M[lev].sum()) < 200:
            raise cv2.error("too few mask pixels")
        cc, w = cv2.findTransformECC(T[lev], I[lev], W.astype(np.float32), cv2.MOTION_HOMOGRAPHY, crit, M[lev], 1)
        W = w
        if lev > 0:
            W = _scale(W, 2.0)
    return float(cc), W


def _block_inliers(tmpl, inp, mask, W, block=32, k=4.0):
    """mask minus blocks (input = reference coords) whose |z(ref) - z(warped query)| median is an outlier."""
    h, w = inp.shape
    # query in reference coordinates: tmpl(W^-1 y) = cv2.warpPerspective(tmpl, W) (the 2026-11-04 direction fix)
    wt = cv2.warpPerspective(tmpl, W.astype(np.float64), (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    vt = cv2.warpPerspective(np.ones_like(tmpl), W.astype(np.float64), (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    sel = (mask > 0) & vt
    zr = (inp - inp[sel].mean()) / (inp[sel].std() + 1e-6)
    zq = (wt - wt[sel].mean()) / (wt[sel].std() + 1e-6)
    res = np.abs(zr - zq)
    stats = []
    for y0 in range(0, h, block):
        for x0 in range(0, w, block):
            bm = sel[y0:y0 + block, x0:x0 + block]
            if bm.sum() >= 0.25 * bm.size:
                stats.append((y0, x0, float(np.median(res[y0:y0 + block, x0:x0 + block][bm]))))
    out = (mask > 0).astype(np.uint8)
    if not stats:
        return out, 0
    v = np.array([s[2] for s in stats]); med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
    thr = med + k * max(mad, 1e-3)
    n = 0
    for y0, x0, r in stats:
        if r > thr:
            out[y0:y0 + block, x0:x0 + block] = 0; n += 1
    return out, n


def _c1_fit(ref_bgr, qry_bgr, box, levels=4, stop_level=1, excl_f=0.5, sigma=1.5, border=24, W0=None):
    """The prototype's ecc_global(robust=True) at level `stop_level`: (W_level, info, pass-1 mask, retained mask).
    W0 (full-resolution start warp) is None = identity, as in the prototype."""
    h, w = ref_bgr.shape[:2]
    R, Q = _prep(ref_bgr, sigma, "edges"), _prep(qry_bgr, sigma, "edges")
    m = np.zeros((h, w), np.uint8); m[border:h - border, border:w - border] = 1
    ex0, ey0, ex1, ey1 = e021.grow(box, excl_f, w, h); m[ey0:ey1, ex0:ex1] = 0
    for _ in range(stop_level):
        R, Q = cv2.pyrDown(R), cv2.pyrDown(Q)
        m = (cv2.resize(m.astype(np.float32), (R.shape[1], R.shape[0]), interpolation=cv2.INTER_AREA) > 0.999).astype(np.uint8)
    start = np.eye(3, dtype=np.float32) if W0 is None else _scale(np.asarray(W0, np.float64), 0.5 ** stop_level)
    cc1, W = _ecc_pyr(Q, R, m, levels - stop_level, start)
    m2, n = _block_inliers(Q, R, m, W, block=max(8, 32 >> stop_level))
    cc, W = _ecc_pyr(Q, R, m2, 0, W)
    return W, {"cc1": round(cc1, 6), "blocks_dropped": n, "cc": round(cc, 6)}, m, m2


def support(m: np.ndarray, m2: np.ndarray) -> dict[str, Any]:
    """The support guard on the level-1 pass-1 fit mask m and the retained mask m2."""
    h, w = m.shape
    share = float(m2.sum()) / max(float(m.sum()), 1.0)
    cells = 0
    for i in range(3):
        for j in range(3):
            sl = (slice(i * h // 3, (i + 1) * h // 3), slice(j * w // 3, (j + 1) * w // 3))
            base = float(m[sl].sum())
            cells += base > 0 and float(m2[sl].sum()) >= CELL_SHARE * base
    return {"support_share": round(share, 4), "support_cells": int(cells),
            "support_ok": share >= SUPPORT_SHARE and cells >= CELLS_MIN}


def final_checks(H: np.ndarray, cc: float, box: Box, sup: dict[str, Any]) -> str | None:
    """None if the final warp passes v0.3's three checks and the support guard, else the reason."""
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1.0])
    Hi = np.linalg.inv(H)
    q = Hi @ c
    q = q[:2] / q[2]
    shift = float(np.hypot(*(q - c[:2])))
    scale = float(np.sqrt(abs(np.linalg.det(v03._jacobian(Hi, c[:2])))))
    if not cc >= AP.min_cc:
        return f"cc {cc:.3f} < {AP.min_cc}"
    if shift > AP.max_shift_px:
        return f"box moves {shift:.1f} px > {AP.max_shift_px}"
    if abs(scale - 1.0) > AP.max_scale_dev:
        return f"local scale {scale:.3f} off by more than {AP.max_scale_dev}"
    if not sup["support_ok"]:
        return (f"support {sup['support_share']:.2f} of the fit pixels, {sup['support_cells']} of 9 cells "
                f"(needs {SUPPORT_SHARE}, {CELLS_MIN})")
    return None


def c1_align(ref_bgr: np.ndarray, qry_bgr: np.ndarray, box: Box) -> tuple[np.ndarray | None, dict[str, Any]]:
    """C1: (H query -> reference at full resolution, or None if refused; diagnostics). Images and box only."""
    nt = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        if qry_bgr.shape[:2] != ref_bgr.shape[:2]:
            return None, {"method": "C1", "refused": f"size {qry_bgr.shape[:2]} != {ref_bgr.shape[:2]}"}
        try:
            W, info, m, m2 = _c1_fit(ref_bgr, qry_bgr, box)
        except cv2.error as err:
            return None, {"method": "C1", "refused": "ECC: " + str(err).strip().splitlines()[-1][:160]}
        H = _scale(W, 2.0).astype(np.float64)
        if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
            return None, {"method": "C1", **info, "refused": "non-finite warp"}
        H = H / H[2, 2]
        sup = support(m, m2)
        c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1.0])
        Hi = np.linalg.inv(H)
        q = Hi @ c
        info = {"method": "C1", **info, **sup,
                "box_shift_px": round(float(np.hypot(*(q[:2] / q[2] - c[:2]))), 3),
                "box_scale": round(float(np.sqrt(abs(np.linalg.det(v03._jacobian(Hi, c[:2]))))), 5),
                "H_unchecked": np.round(H, 10).tolist()}
        reason = final_checks(H, info["cc"], box, sup)
        if reason:
            return None, {**info, "refused": reason}
        return H, info
    finally:
        cv2.setNumThreads(nt)


# ---------------------------------------------------------------- C1+M: two starts, one winner (plan, C1+M)

PC_LEVEL, PC_MAX_PX, TIE_MARGIN, BLOCK_L1 = 4, 100.0, 0.01, 16


def phase_start(ref_bgr: np.ndarray, qry_bgr: np.ndarray, box: Box) -> np.ndarray | None:
    """S1: a full-resolution translation start from phase correlation of the level-4 edge images, each multiplied by
    the level-4 fit mask and a Hann window; t with query(x) ~ ref(x + t) (cv2.phaseCorrelate(query, ref), sign pinned
    by a test). None if |16 t| > 100 px."""
    h, w = ref_bgr.shape[:2]
    R, Q = v03._edges(ref_bgr, AP.sigma), v03._edges(qry_bgr, AP.sigma)
    m = np.zeros((h, w), np.uint8)
    m[AP.border:h - AP.border, AP.border:w - AP.border] = 1
    x0, y0, x1, y1 = e021.grow(box, AP.excl_f, w, h)
    m[y0:y1, x0:x1] = 0
    for _ in range(PC_LEVEL):
        R, Q = cv2.pyrDown(R), cv2.pyrDown(Q)
        m = v03._pyr_down_mask(m, R.shape)
    win = cv2.createHanningWindow((R.shape[1], R.shape[0]), cv2.CV_32F)
    mf = m.astype(np.float32)
    (tx, ty), _ = cv2.phaseCorrelate(np.ascontiguousarray(Q * mf), np.ascontiguousarray(R * mf), win)
    t = np.array([tx, ty]) * 2.0 ** PC_LEVEL
    if float(np.hypot(*t)) > PC_MAX_PX:
        return None
    return np.array([[1.0, 0, t[0]], [0, 1.0, t[1]], [0, 0, 1.0]])


def _blocks(m: np.ndarray) -> list[tuple[int, int]]:
    """Level-1 blocks (16 px, grid at (0, 0)) with >= 25 % of their pixels in the mask m."""
    h, w = m.shape
    b = BLOCK_L1
    return [(y, x) for y in range(0, h, b) for x in range(0, w, b)
            if m[y:y + b, x:x + b].sum() >= 0.25 * m[y:y + b, x:x + b].size]


def _cells(blocks: Sequence[tuple[int, int]], base: Sequence[tuple[int, int]], shape: tuple[int, int]) -> int:
    """Cells of a 3 x 3 grid over the level-1 frame that keep >= 25 % of their `base` blocks."""
    h, w = shape
    cell = lambda yx: (min(2, yx[0] * 3 // h), min(2, yx[1] * 3 // w))
    keep, tot = {}, {}
    for yx in base:
        tot[cell(yx)] = tot.get(cell(yx), 0) + 1
    for yx in blocks:
        keep[cell(yx)] = keep.get(cell(yx), 0) + 1
    return sum(1 for c, n in tot.items() if keep.get(c, 0) >= CELL_SHARE * n)


def validation_scores(R1: np.ndarray, Q1: np.ndarray, m1: np.ndarray, warps: dict[str, np.ndarray]) -> dict[str, Any]:
    """The comparable robust validation score: median block ZNCC of reference edges and query edges warped into
    reference coordinates (bilinear) by each level-1 warp, over the COMMON blocks: S0 pass-1 fit-mask blocks usable
    (>= 25 % in the mask and valid) under EVERY warp, with reference edge std > 1e-6."""
    h, w = R1.shape
    base = _blocks(m1)
    warped, valid = {}, {}
    for k, W in warps.items():
        warped[k] = cv2.warpPerspective(Q1, W, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        valid[k] = cv2.warpPerspective(np.ones_like(Q1), W, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    b = BLOCK_L1
    common = []
    for y, x in base:
        sl = (slice(y, y + b), slice(x, x + b))
        if R1[sl][m1[sl] > 0].std() <= 1e-6:
            continue
        if all(((m1[sl] > 0) & valid[k][sl]).sum() >= 0.25 * m1[sl].size for k in warps):
            common.append((y, x))
    scores = {}
    for k in warps:
        z = []
        for y, x in common:
            sl = (slice(y, y + b), slice(x, x + b))
            sel = (m1[sl] > 0) & valid[k][sl]
            a, q = R1[sl][sel].astype(np.float64), warped[k][sl][sel].astype(np.float64)
            a, q = a - a.mean(), q - q.mean()
            den = np.sqrt((a * a).sum() * (q * q).sum())
            z.append(float((a * q).sum() / den) if den > 0 else 0.0)
        scores[k] = float(np.median(z)) if z else float("nan")
    ok = len(common) >= SUPPORT_SHARE * len(base) and _cells(common, base, (h, w)) >= CELLS_MIN
    return {"scores": scores, "common_blocks": len(common), "base_blocks": len(base),
            "common_cells": _cells(common, base, (h, w)), "comparable": bool(ok)}


def choose(eligible: dict[str, bool], val: dict[str, Any] | None) -> str | None:
    """The pinned ranking: only eligible starts; none -> None (refused); one -> it; both: S0 unless the scores are
    comparable and S1's exceeds S0's by more than 0.01; not comparable -> S0."""
    el = [k for k in ("S0", "S1") if eligible.get(k)]
    if not el:
        return None
    if len(el) == 1:
        return el[0]
    if not val["comparable"]:
        return "S0"
    sc = val["scores"]
    return "S1" if sc["S1"] > sc["S0"] + TIE_MARGIN else "S0"


def _c1_from(ref_bgr, qry_bgr, box, W0):
    """One C1 run from start W0 (None = identity): (H or None, info, level-1 W, pass-1 mask)."""
    try:
        W, info, m, m2 = _c1_fit(ref_bgr, qry_bgr, box, W0=W0)
    except cv2.error as err:
        return None, {"refused": "ECC: " + str(err).strip().splitlines()[-1][:160]}, None, None
    H = _scale(W, 2.0).astype(np.float64)
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None, {**info, "refused": "non-finite warp"}, None, None
    H = H / H[2, 2]
    sup = support(m, m2)
    reason = final_checks(H, info["cc"], box, sup)
    info = {**info, **sup, "H_unchecked": np.round(H, 10).tolist()}
    return (None if reason else H), ({**info, "refused": reason} if reason else info), W, m


def c1m_align(ref_bgr: np.ndarray, qry_bgr: np.ndarray, box: Box) -> tuple[np.ndarray | None, dict[str, Any]]:
    """C1+M: C1 from S0 (identity) and S1 (phase-correlation translation); the pinned ranking picks one."""
    nt = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        if qry_bgr.shape[:2] != ref_bgr.shape[:2]:
            return None, {"method": "C1+M", "refused": f"size {qry_bgr.shape[:2]} != {ref_bgr.shape[:2]}"}
        runs = {"S0": _c1_from(ref_bgr, qry_bgr, box, None)}
        s1 = phase_start(ref_bgr, qry_bgr, box)
        if s1 is not None:
            runs["S1"] = _c1_from(ref_bgr, qry_bgr, box, s1)
        eligible = {k: r[0] is not None for k, r in runs.items()}
        val = None
        if sum(eligible.values()) == 2:
            m1 = runs["S0"][3]
            R1 = cv2.pyrDown(_prep(ref_bgr, AP.sigma, "edges"))
            Q1 = cv2.pyrDown(_prep(qry_bgr, AP.sigma, "edges"))
            val = validation_scores(R1, Q1, m1, {k: runs[k][2].astype(np.float64) for k in ("S0", "S1")})
        win = choose(eligible, val)
        info = {"method": "C1+M", "s1_start": None if s1 is None else np.round(s1, 4).tolist(),
                "starts": {k: r[1] for k, r in runs.items()}, "validation": val, "winner": win}
        if win is None:
            return None, {**info, "refused": runs["S0"][1].get("refused")}
        return runs[win][0], {**info, "cc": runs[win][1]["cc"]}
    finally:
        cv2.setNumThreads(nt)


def c0_align(ref_bgr: np.ndarray, qry_bgr: np.ndarray, box: Box) -> tuple[np.ndarray | None, dict[str, Any]]:
    """C0: v0.3's image_align at its default AlignParams, unchanged."""
    return v03.image_align(ref_bgr, qry_bgr, box)


CANDIDATES = {"C0": c0_align, "C1": c1_align, "C1M": c1m_align}


# ---------------------------------------------------------------- evidence with a candidate's warp

def evidence_for(ref_bgr: np.ndarray, box: Box, query_bgr: np.ndarray, stop: int, encoder: e021.PatchEncoder,
                 detector: e021.Detector, candidate: str) -> tuple[Any, dict[str, Any]]:
    """v0.3's evidence_for, unchanged, on the candidate's warp (only H is passed)."""
    t0 = time.perf_counter()
    box = tuple(float(x) for x in box)
    H, ainfo = CANDIDATES[candidate](ref_bgr, query_bgr, box)
    t1 = time.perf_counter()
    rec, diag = v03.evidence_for(ref_bgr, box, query_bgr, stop, encoder, detector, align=lambda r, q: H)
    t2 = time.perf_counter()
    return rec, {**diag, "registration": ainfo, "H": None if H is None else np.round(H, 10).tolist(),
                 "timing": {"registration_s": round(t1 - t0, 4), "evidence_s": round(t2 - t0, 4)}}


def run_jobs(jobs: Sequence[dict], encoder: e021.PatchEncoder, detector: e021.Detector,
             candidate: str) -> tuple[EvidenceStore, list[dict[str, Any]]]:
    store = EvidenceStore(synthetic=False, source=f"perception.registration_v032 {EVIDENCE_VERSION} {candidate}")
    diags = []
    anns: dict[str, dict] = {}
    refs: dict[str, np.ndarray] = {}
    for j in jobs:
        ann = anns.setdefault(j["annotation"], e021.load_annotation(j["annotation"]))
        if int(ann["stop"]) != int(j["stop"]):
            raise ValueError(f"job stop {j['stop']} != annotation stop {ann['stop']} ({j['annotation']})")
        ref = refs.setdefault(ann["reference"], e021.read_bgr(ann["reference"]))
        rec, diag = evidence_for(ref, ann["targets"][j["target"]]["box"], e021.read_bgr(j["query"]), int(j["stop"]),
                                 encoder, detector, candidate)
        store.put(j["episode_id"], rec)
        diags.append({**j, **dataclasses.asdict(rec), "evidence_version": f"{EVIDENCE_VERSION} {candidate}",
                      "flags_placeholder_thresholds": e021.flags(rec), **diag})
    return store, diags


def merge_shards(jobs: Sequence[dict], shards: Sequence[Sequence[dict]]) -> list[dict]:
    """The shards' diagnostics in the jobs' order; every (episode_id, stop) key of the jobs exactly once, or ValueError."""
    want = [(j["episode_id"], int(j["stop"])) for j in jobs]
    if len(set(want)) != len(want):
        raise ValueError("the jobs file repeats an (episode_id, stop) key")
    got: dict[tuple[str, int], dict] = {}
    for sh in shards:
        for d in sh:
            k = (d["episode_id"], int(d["stop"]))
            if k in got:
                raise ValueError(f"key {k} produced twice")
            got[k] = d
    missing, extra = set(want) - set(got), set(got) - set(want)
    if missing or extra:
        raise ValueError(f"{len(missing)} keys missing (e.g. {sorted(missing)[:3]}), {len(extra)} unexpected")
    return [got[k] for k in want]


def _worker(args: tuple) -> list[dict]:
    jobs, candidate, device, torch_threads = args
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(torch_threads)
    _, diags = run_jobs(jobs, e021.DinoV2Patches(device), e021.YoloWorld(device), candidate)
    return diags


def run_sharded(jobs: Sequence[dict], candidate: str, device: str, workers: int,
                torch_threads: int) -> list[dict]:
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)     # inherited by the spawned workers
    shards = [list(jobs[i::workers]) for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        out = pool.map(_worker, [(sh, candidate, device, torch_threads) for sh in shards])
    return merge_shards(jobs, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perception.registration_v032", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True, help="evidence.jsonl (EvidenceStore format)")
    ap.add_argument("--diag", default=None, help="per-view diagnostics jsonl (v0.3's fields; registration = the candidate's)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=1, help="spawned worker processes (jobs sharded round-robin)")
    ap.add_argument("--torch-threads", type=int, default=1, help="torch.set_num_threads and OMP_NUM_THREADS per worker")
    ap.add_argument("--timing", default=None, help="batch timing record (json)")
    args = ap.parse_args(argv)
    doc = json.loads(e021._path(args.jobs).read_text())
    if doc.get("schema") != e021.JOBS_SCHEMA:
        print(f"{args.jobs}: schema is {doc.get('schema')!r}, expected {e021.JOBS_SCHEMA!r}", file=sys.stderr)
        return 2
    start, wall0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.perf_counter()
    diags = run_sharded(doc["jobs"], args.candidate, args.device, args.workers, args.torch_threads)
    wall = time.perf_counter() - wall0
    store = EvidenceStore(synthetic=False, source=f"perception.registration_v032 {EVIDENCE_VERSION} {args.candidate}")
    for d in diags:
        store.put(d["episode_id"], EvidenceRecord(int(d["stop"]), d["match_score"], d["quality"], d["obstruction"]))
    store.to_jsonl(args.out)
    if args.timing:
        gpu = None
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001 - the record says what it could not read
            gpu = "unknown"
        with open(args.timing, "w") as f:
            json.dump({"candidate": args.candidate, "jobs": str(args.jobs), "views": len(diags),
                       "workers": args.workers, "torch_threads": args.torch_threads,
                       "omp_num_threads": args.torch_threads, "cv2_threads": 1, "device": args.device,
                       "host": platform.node(), "cpu_count": os.cpu_count(), "gpu": gpu,
                       "started_utc": start, "wall_s": round(wall, 2),
                       "views_per_hour": round(len(diags) / wall * 3600, 1)}, f, indent=1)
    if args.diag:
        with open(args.diag, "w") as f:
            for d in diags:
                f.write(json.dumps(d) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
