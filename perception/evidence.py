"""Per-view evidence v0: (per-stop reference + expected box, query at the same stop) -> EvidenceRecord.

This is the step from real captures to what verify/ consumes (verify/README.md section 0):
`EvidenceRecord(stop, match_score, quality, obstruction)`, all in [0, 1], none a probability. The
present / clear-miss / occluded flags are not stored; `verify.stopping.StoppingRule` derives them
from these three numbers, and `flags()` here calls that same rule so diagnostics never disagree.

For one (target, view):
  1. Register the query onto the reference with `align(ref, query) -> H | None` (H maps query pixels
     to reference pixels). The default is capture/tags.py (ArUco corner tags); if it is missing or
     refuses, the identity is used, `align_ok` is False in the diagnostics and quality is set to 0
     (an unregistered view can never count as a clear miss).
  2. match_score: DINOv2-small patch tokens of the reference and the registered query, each computed
     on the same context window around the expected box at NATIVE resolution (the window is only
     trimmed to a multiple of 14 px, never resized; a far-spot mug is ~90 px, i.e. ~6x6 patches).
     For every reference patch inside the box, take the best cosine among query patches inside the
     box grown by `search_pad` (stops repeat to a few cm); the score is the mean, clipped to [0, 1].
     Look-alikes are told apart by location only: the expected region comes from this target's
     reference box.
  3. quality: min(sharpness, exposure, valid) on the context window. sharpness = Laplacian variance
     of the query relative to the reference, on the ring outside the search region (so the target's
     own change does not count), capped at 1. Since 0.2.0 the ring's CHANGED pixels are left out too
     (a neighbour taken away removes texture that is not blur; pilot 2026-10-14: chess clock removed
     together with the Clorox read 0.32), unless fewer than `min_ring_kept` of the ring remain. The
     query side is resampled nearest-neighbour, because bilinear resampling alone (a 0.5 px shift)
     cuts Laplacian variance to ~0.2x. exposure = 1 - share of clipped pixels. valid = share of the
     window that registration filled from inside the query frame.
  4. obstruction = max of two cues:
     - detector cue: share of the expected box whose pixels CHANGED against the reference (gain
       normalised, blurred, max over B/G/R) AND lie inside a detected non-target box (YOLO-World;
       detections with IoU >= target_iou against the box count as the target itself). A removed
       target changes pixels but adds no detection; static clutter is detected but unchanged.
     - novelty cue (0.2.0): share of query patches in the box that match nothing in the reference
       window (target or background; best DINOv2 cosine < novel_cos). A removed target leaves
       background; an occluder the detector misses (a book: pilot 2026-10-14, novelty 0.84/0.86 vs
       <= 0.13 for the four removals) does not. An occluder that also appears in the reference's
       background (the Clorox beside the cup) is explained by it, so the detector cue is still needed.

Every number in `Params` is a placeholder to be selected on dev data later. Nothing was tuned on
the 2026-10-12 smoke frames. novel_cos 0.5 was picked from three printed cutoffs (0.4/0.5/0.6) on the
2026-10-14 pilot (dev data, 12 queries): it is a dev choice, not a test result. Its passing window on
that pilot is about 0.37-0.59, set at each end by a single example (review 2026-10-16).
Known limits (review 2026-10-16, evidence in progress/LOG.md):
  - an occluder that looks like something already in the reference window (background-like, or a
    copy of a neighbour) and that the detector misses gives low novelty and no detection: a WRONG
    clear miss. Real pilot cases (the Clorox) were saved only by the detector;
  - a removal that reveals content absent from the reference window (a different object behind, a
    deep shadow) reads as novel: flagged occluded, never a clear miss (unknown, the safe direction);
  - sharpness is a Laplacian-variance ratio: noise raises it, so heavy blur plus sensor noise can still
    read as sharp.

Usage (repo root):
  uv run --no-sync python -m perception.evidence --jobs JOBS.json --out evidence.jsonl \
      [--diag diagnostics.jsonl] [--device cuda]

JOBS.json: {"schema": "cmpe249.evidence-jobs.v0", "jobs": [{"episode_id", "stop", "target",
"annotation", "query"}, ...]}. An annotation (one per stop reference) is
{"schema": "cmpe249.evidence-annotation.v0", "reference": <image path>, "stop": <int>,
 "targets": {<name>: {"box": [x0, y0, x1, y1]}}} in reference pixels. Relative paths resolve
against the repo root. --out is EvidenceStore.to_jsonl format, which verify.run --evidence reads.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from verify.schema import REPO_ROOT, EvidenceRecord, EvidenceStore, Observation, Positions, Thresholds
from verify.stopping import StoppingRule

EVIDENCE_VERSION = "0.2.1"
ANNOTATION_SCHEMA = "cmpe249.evidence-annotation.v0"
JOBS_SCHEMA = "cmpe249.evidence-jobs.v0"
DINOV2_REPO = "facebook/dinov2-small"
DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"   # same pin as perception/match_check.py
DETECTOR_WEIGHTS = "yolov8s-worldv2.pt"
# Open-vocabulary prompt: targets plus household occluders/clutter. Placeholder list.
CATALOG = ("cup", "mug", "clock", "plush toy", "bottle", "canister", "can", "box", "book", "jar",
           "figurine", "toy")

Box = tuple[float, float, float, float]
Aligner = Callable[[np.ndarray, np.ndarray], "np.ndarray | None"]


@dataclass(frozen=True)
class Params:
    """PLACEHOLDERS, to be selected on dev data later. Exceptions, chosen on the 2026-10-14 pilot (dev):
    novel_cos, and the decision to add the novelty cue and change-masked sharpness at all (they fix
    the 3 errors 0.1.1 made on that pilot). sharp_dilate and min_ring_kept are unvalidated guesses:
    the pilot gives the same verdicts for sharp_dilate 0..9 and never hits the min_ring_kept fallback."""
    context: float = 1.0           # DINOv2 window = box grown by context x box size on every side
    search_pad: float = 0.25       # query patches searched within box grown by this fraction
    det_context: float = 2.0       # detector window = box grown by this (an occluder can be taller)
    det_conf: float = 0.25
    det_imgsz: int = 640
    target_iou: float = 0.5        # a detection this overlapping the box is the target, not an occluder
    max_area_ratio: float = 10.0   # ignore detections larger than this x box area (shelf-sized boxes)
    change_delta: float = 25.0     # grey levels (0-255) after gain normalisation and blur
    change_sigma: float = 2.0
    clip_lo: int = 4
    clip_hi: int = 251
    novel_cos: float = 0.5         # a box patch explained by neither the reference target nor the background
    min_ring_kept: float = 0.2     # sharpness ring: fall back to the full ring if fewer unchanged pixels remain
    sharp_dilate: int = 9          # px; changed pixels are dilated before leaving them out of the sharpness ring
                                   # (the blurred change test misses the edges of a removed texture, which carry
                                   # most of its Laplacian energy)


@dataclass(frozen=True)
class Detection:
    box: Box
    cls: str
    conf: float


class PatchEncoder(Protocol):
    patch: int

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 HxWx3 RGB, H and W multiples of `patch` -> (H/patch, W/patch, D) unit-norm features."""


class Detector(Protocol):
    def __call__(self, bgr: np.ndarray) -> list[Detection]:
        """uint8 HxWx3 BGR -> detections in that image's pixels."""


# ---------------------------------------------------------------- io

def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def read_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(_path(path)), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return img


def load_annotation(path: str | Path) -> dict[str, Any]:
    d = json.loads(_path(path).read_text())
    if d.get("schema") != ANNOTATION_SCHEMA:
        raise ValueError(f"{path}: schema is {d.get('schema')!r}, expected {ANNOTATION_SCHEMA!r}")
    for name, t in d["targets"].items():
        x0, y0, x1, y1 = t["box"]
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"{path}: bad box for {name}: {t['box']}")
    return d


# ---------------------------------------------------------------- geometry

def grow(box: Box, f: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Box grown by f x its size on every side, rounded outwards and clipped to a w x h image."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return (max(0, int(np.floor(x0 - f * bw))), max(0, int(np.floor(y0 - f * bh))),
            min(w, int(np.ceil(x1 + f * bw))), min(h, int(np.ceil(y1 + f * bh))))


def box_mask(box: Box, win: tuple[int, int, int, int]) -> np.ndarray:
    """Bool mask (window pixels) of the part of `box` inside `win`."""
    m = np.zeros((win[3] - win[1], win[2] - win[0]), bool)
    x0, y0 = max(0, int(round(box[0])) - win[0]), max(0, int(round(box[1])) - win[1])
    x1, y1 = min(m.shape[1], int(round(box[2])) - win[0]), min(m.shape[0], int(round(box[3])) - win[1])
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


# ---------------------------------------------------------------- registration

def tag_align(ref: np.ndarray, query: np.ndarray) -> np.ndarray | None:
    """capture/tags.py registration (query -> ref homography), or None if unavailable or refused."""
    capture_dir = str(REPO_ROOT / "capture")
    if capture_dir not in sys.path:
        sys.path.insert(0, capture_dir)
    try:
        import tags
    except ImportError:
        return None
    a = tags.align(ref, query)
    return a.H if a.ok else None


def identity_align(ref: np.ndarray, query: np.ndarray) -> np.ndarray | None:
    """Explicit identity: the query is already in reference pixels (tests, pre-registered crops)."""
    return np.eye(3)


def register(ref: np.ndarray, query: np.ndarray, align: Aligner,
             interp: int = cv2.INTER_LINEAR) -> tuple[np.ndarray, np.ndarray, bool]:
    """(query resampled into ref pixels, valid mask, align_ok). Identity when align returns None."""
    H = align(ref, query)
    h, w = ref.shape[:2]
    if H is None:
        if query.shape[:2] != (h, w):
            raise ValueError(f"query {query.shape[:2]} and reference {(h, w)} differ and registration failed")
        return query, np.ones((h, w), bool), False
    warped = cv2.warpPerspective(query, H, (w, h), flags=interp, borderValue=0)
    ones = np.ones(query.shape[:2], np.uint8)
    valid = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    return warped, valid, True


# ---------------------------------------------------------------- the three scores

def match_score(ref_rgb: np.ndarray, qry_rgb: np.ndarray, box: Box, encoder: PatchEncoder,
                p: Params = Params()) -> float:
    """Mean over reference patches in the box of the best cosine among query patches in the search box."""
    return match_and_novelty(ref_rgb, qry_rgb, box, encoder, p)[0]


def match_and_novelty(ref_rgb: np.ndarray, qry_rgb: np.ndarray, box: Box, encoder: PatchEncoder,
                      p: Params = Params()) -> tuple[float, float]:
    """(match_score, novelty) from one pair of encoder calls on the context window.

    novelty: share of QUERY patches inside the box whose best cosine to the REFERENCE's patches, both
    the target's (box) and the rest of the reference window (0.2.1: including the search-pad ring, so
    a surface seen only right around the target, e.g. a mat under it, explains a removal), is below
    `novel_cos`.
    Something new now stands in front of the target. A removed target leaves the box looking like
    background (low novelty); an occluder the detector does not know (a book) does not."""
    h, w = ref_rgb.shape[:2]
    x0, y0, x1, y1 = grow(box, p.context, w, h)
    k = encoder.patch
    x1, y1 = x0 + (x1 - x0) // k * k, y0 + (y1 - y0) // k * k   # trim, never resize
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"context window around {box} is smaller than one {k} px patch")
    fr = encoder(ref_rgb[y0:y1, x0:x1])
    fq = encoder(qry_rgb[y0:y1, x0:x1])
    gh, gw = fr.shape[:2]
    cx = x0 + (np.arange(gw) + 0.5) * k
    cy = y0 + (np.arange(gh) + 0.5) * k

    def centres_in(b: Box) -> np.ndarray:
        return ((cy[:, None] >= b[1]) & (cy[:, None] < b[3]) & (cx[None, :] >= b[0]) & (cx[None, :] < b[2]))

    in_ref = centres_in(box)
    if not in_ref.any():   # box smaller than a patch: use the patch holding its centre
        in_ref[min(gh - 1, int(((box[1] + box[3]) / 2 - y0) // k)), min(gw - 1, int(((box[0] + box[2]) / 2 - x0) // k))] = True
    sx0, sy0, sx1, sy1 = grow(box, p.search_pad, w, h)
    in_qry = centres_in((sx0, sy0, sx1, sy1)) | in_ref
    sims = fr[in_ref] @ fq[in_qry].T
    m = float(np.clip(sims.max(axis=1).mean(), 0.0, 1.0))
    best = (fq[in_ref] @ fr.reshape(-1, fr.shape[-1]).T).max(axis=1)   # whole reference window explains
    return m, float((best < p.novel_cos).mean())


def _grey(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def quality_score(ref_bgr: np.ndarray, qry_bgr: np.ndarray, valid: np.ndarray, box: Box,
                  p: Params = Params(), sharp_bgr: np.ndarray | None = None,
                  changed_win: np.ndarray | None = None) -> tuple[float, dict[str, float]]:
    """min(sharpness, exposure, valid) on the context window; also returns the parts. `sharp_bgr` is
    the query registered nearest-neighbour (default `qry_bgr`), used for sharpness only. `changed_win`
    (window_change_mask) drops changed ring pixels from the sharpness comparison, so a neighbour that
    moved away (its texture gone) is not read as blur; if fewer than `min_ring_kept` of the ring remain,
    the full ring is used."""
    h, w = ref_bgr.shape[:2]
    win = grow(box, p.context, w, h)
    ring = ~box_mask(grow(box, p.search_pad, w, h), win)
    use = ring
    if changed_win is not None:
        k = 2 * (p.sharp_dilate // 2) + 1
        grown = cv2.dilate(changed_win.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        keep = ring & ~grown
        if keep.sum() >= p.min_ring_kept * ring.sum():
            use = keep
    gr, gq = _grey(ref_bgr), _grey(qry_bgr)
    sl = (slice(win[1], win[3]), slice(win[0], win[2]))
    gs = _grey(qry_bgr if sharp_bgr is None else sharp_bgr)
    if not use.any():
        sharp = 0.0   # no ring to compare (box fills the window): unmeasured, never counts as sharp
    else:
        # 0.2.1: gain-normalised (Laplacian variance scales with gain^2; a dimmer lighting is not blur)
        mr, mq = float(gr[sl][use].mean()), float(gs[sl][use].mean())
        g2 = (mr / mq) ** 2 if mq > 1e-6 else 1.0
        lr = cv2.Laplacian(gr, cv2.CV_32F)[sl][use].var()
        lq = cv2.Laplacian(gs, cv2.CV_32F)[sl][use].var() * g2
        sharp = 1.0 if lr <= 1e-6 else min(1.0, float(lq / lr))
    q = gq[sl]
    expo = 1.0 - float(((q <= p.clip_lo) | (q >= p.clip_hi)).mean())
    val = float(valid[sl].mean())
    parts = {"sharpness": sharp, "exposure": expo, "valid": val,
             "ring_kept": float(use.sum() / max(1, ring.sum()))}
    return min(sharp, expo, val), parts


def change_mask(ref_bgr: np.ndarray, qry_bgr: np.ndarray, box: Box, p: Params = Params()) -> np.ndarray:
    """Bool mask over the box's pixels: max over B, G, R of |query - reference| > change_delta, after
    per-channel gain normalisation (ratio of ring means, so a lighting change is not a change) and a
    Gaussian blur. Per channel, because an occluder can match the target's brightness but not its colour."""
    h, w = ref_bgr.shape[:2]
    win = grow(box, p.context, w, h)
    inbox = box_mask(box, win)
    return window_change_mask(ref_bgr, qry_bgr, box, p)[inbox].reshape(
        int(inbox.any(axis=1).sum()), int(inbox.any(axis=0).sum()))


def window_change_mask(ref_bgr: np.ndarray, qry_bgr: np.ndarray, box: Box, p: Params = Params()) -> np.ndarray:
    """change_mask over the whole context window (grow(box, context))."""
    h, w = ref_bgr.shape[:2]
    win = grow(box, p.context, w, h)
    sl = (slice(win[1], win[3]), slice(win[0], win[2]))
    gr = cv2.GaussianBlur(ref_bgr[sl].astype(np.float32), (0, 0), p.change_sigma)
    gq = cv2.GaussianBlur(qry_bgr[sl].astype(np.float32), (0, 0), p.change_sigma)
    ring = ~box_mask(grow(box, p.search_pad, w, h), win)
    if not ring.any():
        ring = np.ones(ring.shape, bool)
    mq, mr = gq[ring].mean(axis=0), gr[ring].mean(axis=0)
    gain = np.where(mq > 1e-6, mr / np.maximum(mq, 1e-6), 1.0)
    return np.abs(gq * gain - gr).max(axis=2) > p.change_delta


def obstruction_score(changed: np.ndarray, dets: Sequence[Detection], box: Box,
                      p: Params = Params(), novelty: float = 0.0) -> tuple[float, dict[str, float]]:
    """max(share of the box that changed AND is inside a non-target detection, novelty); also returns
    the parts. The detector cue catches a known occluder even when it also appears in the reference
    elsewhere (the Clorox next to the cup); novelty catches an occluder the detector misses (the book;
    pilot 2026-10-14)."""
    bx0, by0 = int(round(box[0])), int(round(box[1]))
    hh, ww = changed.shape
    area = (box[2] - box[0]) * (box[3] - box[1])
    cover = np.zeros((hh, ww), bool)
    for d in dets:
        darea = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
        if iou(d.box, box) >= p.target_iou or darea > p.max_area_ratio * area:
            continue
        x0, y0 = max(0, int(round(d.box[0])) - bx0), max(0, int(round(d.box[1])) - by0)
        x1, y1 = min(ww, int(round(d.box[2])) - bx0), min(hh, int(round(d.box[3])) - by0)
        if x1 > x0 and y1 > y0:
            cover[y0:y1, x0:x1] = True
    det_u = float((cover & changed).mean())
    parts = {"det_cover": float(cover.mean()), "changed": float(changed.mean()), "det_obstruction": det_u,
             "novelty": float(novelty)}
    return max(det_u, float(novelty)), parts


# ---------------------------------------------------------------- one view

def evidence_for(ref_bgr: np.ndarray, box: Box, query_bgr: np.ndarray, stop: int, encoder: PatchEncoder,
                 detector: Detector, p: Params = Params(),
                 align: Aligner = tag_align) -> tuple[EvidenceRecord, dict[str, Any]]:
    """EvidenceRecord for one (target, view) plus a diagnostics dict (parts, detections, align_ok)."""
    box = tuple(float(v) for v in box)
    H = align(ref_bgr, query_bgr)   # once; reused for both resamplings
    qry, valid, align_ok = register(ref_bgr, query_bgr, lambda r, q: H)
    qry_nn, _, _ = register(ref_bgr, query_bgr, lambda r, q: H, cv2.INTER_NEAREST)
    h, w = ref_bgr.shape[:2]
    m, nov = match_and_novelty(ref_bgr[:, :, ::-1], qry[:, :, ::-1], box, encoder, p)
    chw = window_change_mask(ref_bgr, qry, box, p)
    q, qparts = quality_score(ref_bgr, qry, valid, box, p, sharp_bgr=qry_nn, changed_win=chw)
    if not align_ok:
        q = 0.0
    dx0, dy0, dx1, dy1 = grow(box, p.det_context, w, h)
    dets = [Detection((d.box[0] + dx0, d.box[1] + dy0, d.box[2] + dx0, d.box[3] + dy0), d.cls, d.conf)
            for d in detector(np.ascontiguousarray(qry[dy0:dy1, dx0:dx1]))]
    u, uparts = obstruction_score(change_mask(ref_bgr, qry, box, p), dets, box, p, novelty=nov)
    rec = EvidenceRecord(stop, m, q, u)
    diag = {"align_ok": align_ok, "quality_parts": qparts, "obstruction_parts": uparts,
            "detections": [{"box": [round(v, 1) for v in d.box], "cls": d.cls, "conf": round(d.conf, 3),
                            "iou_box": round(iou(d.box, box), 3)} for d in dets]}
    return rec, diag


def flags(rec: EvidenceRecord, th: Thresholds = Thresholds()) -> dict[str, bool]:
    """The stopping rule's per-view flags for this record (thresholds are placeholders)."""
    o = Observation.from_record(rec, 0, Positions.default())
    rule = StoppingRule(th)
    return {"strong_match": rule.is_strong_match(o), "clear_miss": rule.is_clear_miss(o),
            "flagged_occluded": rule.is_flagged_occluded(o)}


# ---------------------------------------------------------------- real models (lazy imports)

class DinoV2Patches:
    """DINOv2-small patch tokens (last_hidden_state minus CLS, already layer-normed) at native size."""
    patch = 14

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(DINOV2_REPO, revision=DINOV2_REVISION).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(DINOV2_REPO, revision=DINOV2_REVISION)
        self.mean = np.array(proc.image_mean, np.float32)
        self.std = np.array(proc.image_std, np.float32)
        self.id = f"{DINOV2_REPO}@{DINOV2_REVISION[:12]}"

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = rgb.shape[:2]
        a = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(a.transpose(2, 0, 1)[None].copy()).to(self.device)
        with torch.no_grad():
            t = self.model(pixel_values=x).last_hidden_state[0, 1:].float()
        t = torch.nn.functional.normalize(t, dim=-1)
        return t.reshape(h // self.patch, w // self.patch, -1).cpu().numpy()


class YoloWorld:
    """YOLO-World v2-s prompted with CATALOG (the proposal's small open-vocabulary detector)."""

    def __init__(self, device: str = "cuda", catalog: Sequence[str] = CATALOG, p: Params = Params()) -> None:
        from perception.common import CHECKPOINTS, configure_ultralytics, pin_ultralytics_settings

        configure_ultralytics()
        from ultralytics import YOLO

        pin_ultralytics_settings()
        self.model = YOLO(str(CHECKPOINTS / DETECTOR_WEIGHTS))
        self.model.set_classes(list(catalog))
        self.device, self.p = device, p
        self.id = f"{DETECTOR_WEIGHTS} catalog={','.join(catalog)}"

    def __call__(self, bgr: np.ndarray) -> list[Detection]:
        r = self.model.predict(bgr, imgsz=self.p.det_imgsz, conf=self.p.det_conf, device=self.device,
                               verbose=False)[0]
        return [Detection(tuple(float(v) for v in b), r.names[int(c)], float(s))
                for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy())]


# ---------------------------------------------------------------- cli

def run_jobs(jobs: Sequence[dict], encoder: PatchEncoder, detector: Detector, p: Params = Params(),
             align: Aligner = tag_align) -> tuple[EvidenceStore, list[dict[str, Any]]]:
    store = EvidenceStore(synthetic=False, source=f"perception.evidence {EVIDENCE_VERSION}")
    diags = []
    anns: dict[str, dict] = {}
    refs: dict[str, np.ndarray] = {}
    for j in jobs:
        ann = anns.setdefault(j["annotation"], load_annotation(j["annotation"]))
        if int(ann["stop"]) != int(j["stop"]):
            raise ValueError(f"job stop {j['stop']} != annotation stop {ann['stop']} ({j['annotation']})")
        ref = refs.setdefault(ann["reference"], read_bgr(ann["reference"]))
        rec, diag = evidence_for(ref, ann["targets"][j["target"]]["box"], read_bgr(j["query"]), int(j["stop"]),
                                 encoder, detector, p, align)
        store.put(j["episode_id"], rec)
        diags.append({**j, **dataclasses.asdict(rec), "flags_placeholder_thresholds": flags(rec), **diag})
    return store, diags


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perception.evidence", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True, help="evidence.jsonl (EvidenceStore format)")
    ap.add_argument("--diag", default=None, help="per-view diagnostics jsonl (parts, detections, flags)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    doc = json.loads(_path(args.jobs).read_text())
    if doc.get("schema") != JOBS_SCHEMA:
        print(f"{args.jobs}: schema is {doc.get('schema')!r}, expected {JOBS_SCHEMA!r}", file=sys.stderr)
        return 2
    store, diags = run_jobs(doc["jobs"], DinoV2Patches(args.device), YoloWorld(args.device))
    store.to_jsonl(args.out)
    if args.diag:
        with open(args.diag, "w") as f:
            for d in diags:
                f.write(json.dumps(d) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
