"""OCID -> evidence-step DEVELOPMENT episodes (visible / occluded / removed) with label-derived truth.

Why: the evidence step (perception/evidence.py: match / quality / obstruction) needs many more labelled
PRESENT / OCCLUDED / REMOVED views than our own pilots give, in particular the known failure where an
occluder sits right over the target's footprint. OCID (Suchi et al., ICRA 2019) is a public clutter
dataset where objects are ADDED one at a time and never taken away, with a per-pixel instance label
for every frame. Later objects occlude earlier ones, so the labels give truth for free:
  - an object whose labelled area shrank, and whose lost pixels are now other objects, is OCCLUDED;
  - an object whose area did not shrink is PRESENT (visible);
  - running time backwards (query = a frame from BEFORE the object was placed) gives REMOVED.
This is development data only. It is never the held-out test: the study's test set is our own capture.

Data facts this module relies on (checked on the full unpacked dataset, public_data/README.md):
  - label/*.png is uint16, 480 x 640. 0 = no valid depth (exactly the depth==0 pixels; unlabelled).
    The SUPPORT SURFACE is not one fixed id: in floor sequences 1 = the floor; in table sequences
    1 = the background (wall / floor around the table) and 2 = the TABLE, present alone in frame 0
    (empty table, hence 11 / 21 frames); in ARID10/floor/top/curved/seq05 there is no 1 and 256 is
    the floor. So surface ids are found per sequence: LABEL_SURFACE plus every id present in frame 0
    covering >= SURFACE_MIN_SHARE of the image (real data: surfaces >= 0.51, frame-0 objects <= 0.12).
    All other ids are object instances, stable across the frames of one sequence; normally numbered
    in order of placement, but objects are ordered by FIRST APPEARANCE, never by id value.
  - rgb/ and label/ share file names result_<YYYY-MM-DD-HH-MM-SS>.png; sorted names = time order.
    At most one object is added per frame (asserted in tracks()). An object can vanish from the label
    when fully hidden (one case); the nudge rule below guards against moved masks (it excludes
    nothing on the real data).
  - top/ and bottom/ cameras of the same <set>/<surface>/[<category>/]seqNN record the same
    physical arrangement at the same moments (equal frame counts, each bottom frame between the
    neighbouring top frames). They are two stops of ONE arrangement: stop = CAMERA_STOPS[camera],
    and the dev/val split is shared by both cameras. build() asserts that, per arrangement, both
    cameras have the same object first-appearance frames, so objKK names one physical object. Each camera is static within a sequence, so
    query and reference are already registered (public_data/run_evidence.py uses identity_align).

Episode = (sequence = one camera, reference frame r, target t, query frame q); box = bbox of t's
mask in r (x1, y1 exclusive). Truth rules, frozen as the constants below BEFORE any evidence was
run on these episodes (do not tune them on evidence output; bump LOADER_VERSION if they change):
  reference: t must be clear in r: area_r(t) >= REF_MIN_SHARE x t's max area over the sequence and
    >= REF_MIN_PIXELS; targets whose mask comes within BORDER_MARGIN_PX of the image border in ANY
    frame are skipped (cut-off objects make the ratio meaningless).
  q after r (t is still physically there): v = area_q(t) / area_r(t).
    nudged: pixels of mask_q(t) outside mask_r(t) dilated by NUDGE_DILATE_PX exceed NUDGE_MAX_OUTSIDE
      x area_r(t) -> excluded (the object moved; neither label is clean).
    PRESENT ('visible') if v >= PRESENT_MIN.
    OCCLUDED if v <= OCCLUDED_MAX AND at least OCCLUDER_COVER_MIN of the lost pixels
      (mask_r(t) minus mask_q(t)) are labelled as other objects in q. Occluders = objects holding
      >= OCCLUDER_MIN_SHARE of the lost pixels.
    otherwise excluded as ambiguous (partial loss, or lost pixels that became surface / no-depth).
  REMOVED (time reversed): q < first frame of t (t not yet placed), r = a clear reference frame of t.
    Visible and occluded records also hold target_lost_share = |mask_r - mask_q| / area_r and
    box_new_object_share = share of the box covered in q by objects placed after r, because a
    'visible' target can still have a new object resting partly on it (truth is unchanged by them).
    Removed records hold box_other_new_share = share of the box in r covered by objects placed after q
    (other than t), and box_changed_share = share of box pixels outside mask_r(t) whose label differs
    between q and r. box_other_change = box_other_new_share > REMOVED_BOX_CHANGE, so removals where
    the box region also holds other changes are identifiable.
  same_footprint (occluded only): some occluder's bbox in q has IoU >= SAME_FOOTPRINT_IOU with the
    target's box: the known failure case, tagged in the truth record.

Sampling (deterministic, no RNG state): candidates are ranked by derive_seed(SAMPLE_SALT,
episode_id), occluded ones with same_footprint first (the failure we most need examples of); then at
most CAP_PER_TARGET[label] per (target, label) and CAP_PER_SEQ[label] per sequence are kept. Removed
candidates are NOT ranked by box_other_change, so both clean removals and removals with other changes
in the box occur (about 43 % of removed candidates have them; the flag is in truth_detail). Occlusions
are rare in OCID (72 of 178 sequences have any, mostly of one or two objects), so occluded gets a
larger cap and two episodes per target, visible and removed fewer. Split: by ARRANGEMENT = <set>-<surface>-seqNN (shared by top and bottom), val if
sha256(SPLIT_SALT|arrangement) as a fraction of 2^64 < VAL_FRACTION, else dev. Each arrangement's
split depends only on its own id, so adding or dropping sequences never moves another one.

Output (<out>, absolute paths everywhere): annotations/<seq_key>/<frame>.json (one per reference
frame, cmpe249.evidence-annotation.v0, stop = camera index), jobs.json (cmpe249.evidence-jobs.v0),
truth.json (episode_id -> visible|occluded|removed, as in pilot 2), truth_detail.jsonl (every
recorded number per episode) and MANIFEST.json (version, constants, counts, source, sha256 of this file).
Episode ids (<seq_key>:objKK:rNN:qNN) do not spell out the label, but they are NOT blind: q < r iff
removed (and the query file's timestamp shows the same). The frozen evidence step never reads ids;
anything that may not see truth must not parse them.

Usage (repo root):
  python -m public_data.ocid [--root OCID-dataset] [--out DIR] [--limit-sequences N] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from verify.schema import derive_seed

LOADER_VERSION = "0.2.0"   # 0.2.0: support surface found per sequence (was: table = obj01)
ANNOTATION_SCHEMA = "cmpe249.evidence-annotation.v0"   # same strings as perception/evidence.py
JOBS_SCHEMA = "cmpe249.evidence-jobs.v0"
DETAIL_SCHEMA = "cmpe249.ocid-truth-detail.v0"
MANIFEST_SCHEMA = "cmpe249.ocid-episodes-manifest.v0"
DATA_BASE = Path("/mnt/bench-data/datasets/cmpe249-active-verification/ocid")
DEFAULT_ROOT = DATA_BASE / "extracted" / "OCID-dataset"
DEFAULT_OUT_BASE = DATA_BASE / "episodes"

# ---- label encoding (verified, README)
LABEL_NO_DEPTH = 0
LABEL_SURFACE = 1             # always surface/background; more surface ids are found per sequence
SURFACE_MIN_SHARE = 0.4       # an id present in frame 0 covering >= this share of the image is surface
CAMERA_STOPS = {"top": 0, "bottom": 1}
CONDITIONS = ("visible", "occluded", "removed")

# ---- truth rules (frozen before any evidence was run; module docstring)
REF_MIN_SHARE = 0.9          # area in r >= this x the target's max area in the sequence
REF_MIN_PIXELS = 400         # and at least this many pixels (~20 x 20)
BORDER_MARGIN_PX = 2         # mask within this many px of the image border in any frame -> skip target
NUDGE_DILATE_PX = 5          # tolerance for label jitter around mask_r(t)
NUDGE_MAX_OUTSIDE = 0.05     # mask_q(t) pixels outside the dilation, as a share of area_r(t) -> nudged
PRESENT_MIN = 0.9            # v >= this -> visible
OCCLUDED_MAX = 0.5           # v <= this (and lost pixels covered by objects) -> occluded
OCCLUDER_COVER_MIN = 0.8     # share of lost pixels that must now be other objects
OCCLUDER_MIN_SHARE = 0.1     # an object holding this share of the lost pixels is an occluder
SAME_FOOTPRINT_IOU = 0.5     # occluder bbox IoU with the target box -> same_footprint
REMOVED_BOX_CHANGE = 0.05    # removed: box share covered by objects placed after q -> box_other_change

# ---- sampling and split
CAP_PER_SEQ = {"visible": 2, "occluded": 4, "removed": 2}
CAP_PER_TARGET = {"visible": 1, "occluded": 2, "removed": 1}
SAMPLE_SALT = "ocid-sample-v0"
SPLIT_SALT = "ocid-split-v0"
VAL_FRACTION = 0.25

CONSTANTS = {k: v for k, v in dict(globals()).items()
             if k in ("LABEL_NO_DEPTH", "LABEL_SURFACE", "SURFACE_MIN_SHARE", "CAMERA_STOPS", "REF_MIN_SHARE",
                      "REF_MIN_PIXELS", "BORDER_MARGIN_PX", "NUDGE_DILATE_PX", "NUDGE_MAX_OUTSIDE", "PRESENT_MIN",
                      "OCCLUDED_MAX", "OCCLUDER_COVER_MIN", "OCCLUDER_MIN_SHARE", "SAME_FOOTPRINT_IOU",
                      "REMOVED_BOX_CHANGE", "CAP_PER_SEQ", "CAP_PER_TARGET", "SAMPLE_SALT", "SPLIT_SALT",
                      "VAL_FRACTION")}

Box = tuple[int, int, int, int]


# ---------------------------------------------------------------- discovery

@dataclass(frozen=True)
class OcidSequence:
    """One camera's recording of one arrangement. `parts` = (set, surface, camera, [category,] seqNN)."""
    root: Path
    rel: str                     # path below the dataset root, '/'-separated
    frames: tuple[str, ...]      # label/rgb file names, time order
    n_unmatched: int = 0         # .png names present in only one of label/ and rgb/ (dropped)

    @property
    def parts(self) -> list[str]:
        return self.rel.split("/")

    @property
    def set(self) -> str:
        return self.parts[0]

    @property
    def surface(self) -> str:
        return self.parts[1]

    @property
    def camera(self) -> str:
        return self.parts[2]

    @property
    def category(self) -> str | None:
        return self.parts[3] if len(self.parts) == 5 else None

    @property
    def seq(self) -> str:
        return self.parts[-1]

    @property
    def key(self) -> str:
        return "-".join(self.parts)

    @property
    def arrangement(self) -> str:
        """Shared by the top and bottom camera of the same physical scene."""
        return f"{self.set}-{self.surface}-{self.seq}"

    @property
    def stop(self) -> int:
        return CAMERA_STOPS[self.camera]

    def label_path(self, i: int) -> Path:
        return self.root / self.rel / "label" / self.frames[i]

    def rgb_path(self, i: int) -> Path:
        return self.root / self.rel / "rgb" / self.frames[i]


def discover(root: str | Path) -> list[OcidSequence]:
    """Every directory below root holding label/ and rgb/, sorted by relative path. Frames present in
    only one of label/ and rgb/ are dropped and counted (n_unmatched, summed in MANIFEST; 0 in the
    release we have)."""
    root = Path(root).resolve()
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames.sort()
        if "label" in dirnames and "rgb" in dirnames:
            d = Path(dirpath)
            rel = d.relative_to(root).as_posix()
            if len(rel.split("/")) not in (4, 5) or rel.split("/")[2] not in CAMERA_STOPS:
                raise ValueError(f"unexpected OCID layout: {rel}")
            lab = {n for n in os.listdir(d / "label") if n.endswith(".png")}
            rgb = {n for n in os.listdir(d / "rgb") if n.endswith(".png")}
            out.append(OcidSequence(root, rel, tuple(sorted(lab & rgb)), len(lab ^ rgb)))
    return sorted(out, key=lambda s: s.rel)


def load_labels(seq: OcidSequence) -> np.ndarray:
    """(n_frames, H, W) label stack (int32)."""
    if not seq.frames:
        raise ValueError(f"{seq.rel}: no frames (no .png name shared by label/ and rgb/)")
    frames = []
    for i in range(len(seq.frames)):
        lab = cv2.imread(str(seq.label_path(i)), cv2.IMREAD_UNCHANGED)
        if lab is None or lab.ndim != 2:
            raise ValueError(f"cannot read label {seq.label_path(i)}")
        frames.append(lab)
    if len({f.shape for f in frames}) != 1:
        raise ValueError(f"{seq.rel}: label frames differ in size")
    return np.stack(frames).astype(np.int32)


def split_of(arrangement: str) -> str:
    h = int.from_bytes(hashlib.sha256(f"{SPLIT_SALT}|{arrangement}".encode()).digest()[:8], "big")
    return "val" if h / 2.0 ** 64 < VAL_FRACTION else "dev"


# ---------------------------------------------------------------- per-sequence truth

@dataclass
class ObjectTrack:
    label_id: int
    name: str                    # objNN, NN = order of placement (1-based), surface excluded; same on both cameras
    first: int                   # first frame index where the label is present
    areas: np.ndarray            # per-frame pixel count
    boxes: list[Box | None]      # per-frame bbox (x0, y0, x1, y1), exclusive ends
    border: bool = False


def _bbox(mask: np.ndarray) -> Box | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def surface_ids(labels: np.ndarray) -> set[int]:
    """LABEL_SURFACE plus every id present in frame 0 covering >= SURFACE_MIN_SHARE of the image
    (the table in table sequences, 256 = floor in ARID10/floor/top/curved/seq05)."""
    ids, cnt = np.unique(labels[0], return_counts=True)
    big = {int(i) for i, c in zip(ids.tolist(), cnt.tolist())
           if i != LABEL_NO_DEPTH and c >= SURFACE_MIN_SHARE * labels[0].size}
    return {LABEL_SURFACE} | big


def object_ids(labels: np.ndarray) -> list[int]:
    surf = surface_ids(labels)
    return sorted(int(i) for i in np.unique(labels) if i != LABEL_NO_DEPTH and int(i) not in surf)


def tracks(labels: np.ndarray) -> list[ObjectTrack]:
    """Object tracks (support surface excluded) ordered by first appearance, with per-frame areas
    and boxes. Raises if two objects first appear in the same frame (OCID adds one per frame)."""
    n, h, w = labels.shape
    ids = object_ids(labels)
    raw = []
    for k in ids:
        areas = np.array([(labels[f] == k).sum() for f in range(n)])
        boxes = [_bbox(labels[f] == k) if areas[f] else None for f in range(n)]
        first = int(np.flatnonzero(areas)[0])
        m = BORDER_MARGIN_PX
        border = any(b is not None and (b[0] < m or b[1] < m or b[2] > w - m or b[3] > h - m) for b in boxes)
        raw.append((first, k, areas, boxes, border))
    raw.sort(key=lambda r: (r[0], r[1]))
    firsts = [r[0] for r in raw]
    if len(set(firsts)) != len(firsts):
        dup = sorted(f for f, c in Counter(firsts).items() if c > 1)
        raise ValueError(f"more than one object first appears in frame(s) {dup}: surface not separated?")
    return [ObjectTrack(k, f"obj{i + 1:02d}", first, areas, boxes, border)
            for i, (first, k, areas, boxes, border) in enumerate(raw)]


def clear_reference_frames(t: ObjectTrack) -> list[int]:
    mx = int(t.areas.max())
    return [int(f) for f in np.flatnonzero((t.areas >= REF_MIN_SHARE * mx) & (t.areas >= REF_MIN_PIXELS))]


@dataclass
class Candidate:
    episode_id: str
    label: str
    seq: OcidSequence
    target: ObjectTrack
    r: int
    q: int
    box: Box
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def priority(self) -> tuple[int, int]:
        """Lower sorts first: same_footprint occlusions, then the stable hash."""
        tier = int(self.label == "occluded" and not self.detail.get("same_footprint"))
        return tier, derive_seed(SAMPLE_SALT, self.episode_id)


def episode_id(seq: OcidSequence, t: ObjectTrack, r: int, q: int) -> str:
    return f"{seq.key}:{t.name}:r{r:02d}:q{q:02d}"


def sequence_candidates(seq: OcidSequence, labels: np.ndarray) -> tuple[list[Candidate], Counter]:
    """All labelled (r, t, q) candidates in one sequence plus counts of exclusions by reason."""
    n = labels.shape[0]
    tr = tracks(labels)
    by_id = {t.label_id: t for t in tr}
    obj_ids = np.array(sorted(by_id), dtype=np.int64)
    surf_ids = np.array(sorted(surface_ids(labels)), dtype=np.int64)
    excl: Counter = Counter()
    out: list[Candidate] = []
    k = 2 * NUDGE_DILATE_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for t in tr:
        if t.border:
            excl["target_touches_border"] += 1
            continue
        refs = clear_reference_frames(t)
        if not refs:
            excl["target_no_clear_reference"] += 1
            continue
        for r in refs:
            mask_r = labels[r] == t.label_id
            box = t.boxes[r]
            area_r = int(t.areas[r])
            base = {"area_r": area_r, "max_area": int(t.areas.max()), "ref_share": area_r / int(t.areas.max())}
            dil = cv2.dilate(mask_r.astype(np.uint8), kernel) > 0
            x0, y0, x1, y1 = box
            # ---- q after r: visible / occluded
            for q in range(r + 1, n):
                mask_q = labels[q] == t.label_id
                area_q = int(t.areas[q])
                outside = area_q - int((mask_q & dil).sum())
                v = area_q / area_r
                after_r = [o.label_id for o in tr if o.first > r]
                d: dict[str, Any] = {**base, "area_q": area_q, "v": v, "nudge_outside_px": outside,
                                     "nudge_outside_share": outside / area_r,
                                     "target_lost_share": int((mask_r & ~mask_q).sum()) / area_r,
                                     "box_new_object_share": float(np.isin(labels[q, y0:y1, x0:x1], after_r).mean())
                                     if after_r else 0.0}
                if outside > NUDGE_MAX_OUTSIDE * area_r:
                    excl["nudged"] += 1
                    continue
                if v >= PRESENT_MIN:
                    out.append(Candidate(episode_id(seq, t, r, q), "visible", seq, t, r, q, box, d))
                    continue
                if v > OCCLUDED_MAX:
                    excl["ambiguous_partial"] += 1
                    continue
                lost = mask_r & ~mask_q
                lab_lost = labels[q][lost]
                is_obj = np.isin(lab_lost, obj_ids) & (lab_lost != t.label_id)
                cover = float(is_obj.mean()) if lab_lost.size else 0.0
                d["lost_px"] = int(lab_lost.size)
                d["lost_covered_by_objects"] = cover
                d["lost_no_depth_share"] = float((lab_lost == LABEL_NO_DEPTH).mean()) if lab_lost.size else 0.0
                d["lost_surface_share"] = float(np.isin(lab_lost, surf_ids).mean()) if lab_lost.size else 0.0
                if cover < OCCLUDER_COVER_MIN:
                    excl["ambiguous_lost_not_objects"] += 1
                    continue
                ids, cnt = np.unique(lab_lost[is_obj], return_counts=True)
                occ = []
                for oid, c in sorted(zip(ids.tolist(), cnt.tolist()), key=lambda x: (-x[1], x[0])):
                    share = c / lab_lost.size
                    if share < OCCLUDER_MIN_SHARE:
                        continue
                    o = by_id[oid]
                    ob = o.boxes[q]
                    occ.append({"label_id": oid, "name": o.name, "share_of_lost": share,
                                "box_q": list(ob), "box_iou": box_iou(ob, box)})
                d["occluders"] = occ
                d["max_occluder_box_iou"] = max((o["box_iou"] for o in occ), default=0.0)
                d["same_footprint"] = d["max_occluder_box_iou"] >= SAME_FOOTPRINT_IOU
                out.append(Candidate(episode_id(seq, t, r, q), "occluded", seq, t, r, q, box, d))
            # ---- reversed time: removed
            in_box_r = labels[r, y0:y1, x0:x1]
            not_t = in_box_r != t.label_id
            for q in range(0, t.first):
                in_box_q = labels[q, y0:y1, x0:x1]
                later = {o.label_id for o in tr if o.first > q and o.label_id != t.label_id}
                new_share = float(np.isin(in_box_r, list(later)).mean()) if later else 0.0
                changed = float((not_t & (in_box_r != in_box_q)).sum() / in_box_r.size)
                d = {**base, "box_other_new_share": new_share, "box_changed_share": changed,
                     "box_other_change": new_share > REMOVED_BOX_CHANGE,
                     "frames_between": r - q}
                out.append(Candidate(episode_id(seq, t, r, q), "removed", seq, t, r, q, box, d))
    return out, excl


def sample(cands: Iterable[Candidate]) -> list[Candidate]:
    """Deterministic caps: per (target, label) and per sequence and label (module docstring)."""
    cands = list(cands)          # read once per label
    kept: list[Candidate] = []
    for label in CONDITIONS:
        per_target: Counter = Counter()
        n = 0
        for c in sorted((c for c in cands if c.label == label), key=lambda c: c.priority):
            if n >= CAP_PER_SEQ[label]:
                break
            if per_target[c.target.label_id] >= CAP_PER_TARGET[label]:
                continue
            per_target[c.target.label_id] += 1
            kept.append(c)
            n += 1
    return kept


# ---------------------------------------------------------------- output

def detail_record(c: Candidate) -> dict[str, Any]:
    s = c.seq
    return {"schema": DETAIL_SCHEMA, "episode_id": c.episode_id, "label": c.label,
            "split": split_of(s.arrangement), "arrangement": s.arrangement, "sequence": s.key,
            "set": s.set, "surface": s.surface, "camera": s.camera, "category": s.category, "stop": s.stop,
            "target": c.target.name, "target_label_id": c.target.label_id, "target_first_frame": c.target.first,
            "ref_index": c.r, "query_index": c.q, "ref_frame": s.frames[c.r], "query_frame": s.frames[c.q],
            "box": list(c.box), **c.detail}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_camera_pairs(firsts: dict[str, tuple[str, str, tuple[int, ...]]]) -> None:
    """firsts: seq_key -> (arrangement, camera, first-appearance frames in name order). Raises unless
    both cameras of every arrangement that has both agree, so objKK is one physical object on both."""
    by_arr: dict[str, dict[str, tuple[int, ...]]] = {}
    for arr, cam, f in firsts.values():
        if cam in by_arr.setdefault(arr, {}):
            raise ValueError(f"{arr}: two {cam} sequences")
        by_arr[arr][cam] = f
    bad = {a: c for a, c in by_arr.items() if len(c) == 2 and len(set(c.values())) != 1}
    if bad:
        raise ValueError(f"top/bottom object first frames differ (objKK would name different objects): {bad}")


# output files build() owns inside <out>; evidence runs default to <out>/evidence/ (run_evidence.py)
OWNED_DIRS = ("annotations", "evidence")


def build(root: str | Path, out: str | Path, limit_sequences: int | None = None,
          sequences: Sequence[OcidSequence] | None = None, log=print, force: bool = False) -> dict[str, Any]:
    """Write annotations, jobs.json, truth.json, truth_detail.jsonl and MANIFEST.json; return the manifest.
    An <out> already holding a MANIFEST.json is refused unless force; with force, the MANIFEST must be
    ours (MANIFEST_SCHEMA) and its annotations/ and evidence/ (now stale) are deleted first."""
    root = Path(root).resolve()
    out = Path(out).resolve()
    old = out / "MANIFEST.json"
    if old.exists():
        if not force:
            raise FileExistsError(f"{out} already holds episodes (MANIFEST.json); use force to overwrite")
        try:
            schema = json.loads(old.read_text()).get("schema")
        except (ValueError, AttributeError):
            schema = None
        if schema != MANIFEST_SCHEMA:
            raise ValueError(f"{old} is not a {MANIFEST_SCHEMA} manifest ({schema!r}); refusing to delete anything")
    seqs = list(sequences) if sequences is not None else discover(root)
    if limit_sequences is not None:
        seqs = seqs[:limit_sequences]
    kept: list[Candidate] = []
    excl: Counter = Counter()
    n_cand: Counter = Counter()
    shapes: Counter = Counter()
    firsts: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    n_surface: Counter = Counter()
    for i, s in enumerate(seqs):
        labels = load_labels(s)
        shapes[f"{labels.shape[2]}x{labels.shape[1]}"] += 1
        firsts[s.key] = (s.arrangement, s.camera, tuple(t.first for t in tracks(labels)))
        n_surface[len(surface_ids(labels))] += 1
        cands, e = sequence_candidates(s, labels)
        excl.update(e)
        n_cand.update(c.label for c in cands)
        kept.extend(sample(cands))
        log(f"[{i + 1}/{len(seqs)}] {s.key}: {len(cands)} candidates")
    kept.sort(key=lambda c: c.episode_id)
    check_camera_pairs(firsts)

    ann_dir = out / "annotations"
    if old.exists():             # force, and checked above that it is our own output
        for sub in OWNED_DIRS:
            if (out / sub).is_dir():
                shutil.rmtree(out / sub)
    out.mkdir(parents=True, exist_ok=True)
    annotations: dict[tuple[str, int], dict[str, Any]] = {}
    jobs, truth, details = [], {}, []
    for c in kept:
        key = (c.seq.key, c.r)
        ann = annotations.setdefault(key, {
            "schema": ANNOTATION_SCHEMA, "reference": str(c.seq.rgb_path(c.r)), "stop": c.seq.stop,
            "note": f"public_data.ocid {LOADER_VERSION}: boxes are the bbox of the target's OCID label mask "
                    f"in this frame; development data, never the held-out test.",
            "targets": {}})
        ann["targets"][c.target.name] = {"box": list(c.box), "ocid_label_id": c.target.label_id}
        ann_path = ann_dir / c.seq.key / f"{Path(c.seq.frames[c.r]).stem}.json"
        jobs.append({"episode_id": c.episode_id, "stop": c.seq.stop, "target": c.target.name,
                     "annotation": str(ann_path), "query": str(c.seq.rgb_path(c.q))})
        truth[c.episode_id] = c.label
        details.append(detail_record(c))
    for (skey, r), ann in annotations.items():
        p = ann_dir / skey / f"{Path(ann['reference']).stem}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ann, indent=1, sort_keys=True) + "\n")
    (out / "jobs.json").write_text(json.dumps(
        {"schema": JOBS_SCHEMA, "source": f"public_data.ocid {LOADER_VERSION}", "jobs": jobs}, indent=1) + "\n")
    (out / "truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    with open(out / "truth_detail.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d, sort_keys=True) + "\n")

    def counts(keyf) -> dict[str, dict[str, int]]:
        res: dict[str, Counter] = {}
        for d in details:
            res.setdefault(keyf(d), Counter())[d["label"]] += 1
        return {k: dict(sorted(v.items())) for k, v in sorted(res.items())}

    arrs = sorted({s.arrangement for s in seqs})
    manifest = {
        "schema": MANIFEST_SCHEMA, "loader": "public_data.ocid", "loader_version": LOADER_VERSION,
        "loader_sha256": file_sha256(__file__), "source_root": str(root),
        "limit_sequences": limit_sequences, "n_sequences": len(seqs), "frame_sizes_wxh": dict(shapes),
        "n_unmatched_frames": sum(s.n_unmatched for s in seqs),
        "sequences_by_n_surface_ids": {str(k): v for k, v in sorted(n_surface.items())},
        "n_annotations": len(annotations), "n_episodes": len(details),
        "constants": CONSTANTS,
        "counts": {"by_label": dict(sorted(Counter(d["label"] for d in details).items())),
                   "by_split": counts(lambda d: d["split"]), "by_set": counts(lambda d: d["set"]),
                   "by_camera": counts(lambda d: d["camera"]),
                   "occluded_same_footprint": sum(1 for d in details if d.get("same_footprint")),
                   "removed_box_other_change": sum(1 for d in details if d.get("box_other_change"))},
        "candidates_before_sampling": dict(sorted(n_cand.items())),
        "exclusions": dict(sorted(excl.items())),
        "arrangements": {sp: [a for a in arrs if split_of(a) == sp] for sp in ("dev", "val")},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="public_data.ocid", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="unpacked OCID-dataset directory")
    ap.add_argument("--out", default=str(DEFAULT_OUT_BASE / LOADER_VERSION), help="output directory")
    ap.add_argument("--limit-sequences", type=int, default=None, help="only the first N sequences (sorted)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output of this loader (deletes its annotations/ and evidence/)")
    args = ap.parse_args(argv)
    if (Path(args.out) / "MANIFEST.json").exists() and not args.force:
        print(f"{args.out} already holds episodes (MANIFEST.json); pass --force to overwrite "
              f"(deletes its annotations/ and evidence/)", file=sys.stderr)
        return 2
    m = build(args.root, args.out, args.limit_sequences, force=args.force)
    print(json.dumps({k: m[k] for k in ("n_sequences", "n_episodes", "n_annotations", "counts",
                                         "candidates_before_sampling", "exclusions")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
