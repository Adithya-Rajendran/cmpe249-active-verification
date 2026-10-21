#!/usr/bin/env python3
"""ArUco tag registration of a query image onto a per-stop reference (stdlib + numpy + cv2).

Runs unchanged on the Pi (system CPython 3.13.5, OpenCV 4.10) and on the GB10 (uv env, OpenCV 5.0);
both have `cv2.aruco.ArucoDetector`. Import it with the `capture/` directory on `sys.path`
(`import tags`), like `ecc.py`.

Rig (2026-10-12): four DICT_4X4_50 tags, IDs 0-3, 5.0 cm black square, taped flush with the shelf
front plane (0 lower-left, 1 lower-right below the #3 floor; 2 upper-left, 3 upper-right at the top
of #4). The tag rectangle is the monitored area (compartments #3 and #4).

  detect(img)            -> {id: 4x2 float64 corners}, ArUco order TL, TR, BR, BL of each tag.
  align(ref, query)      -> Alignment: homography query -> ref fitted on all shared tag corners
                            (least squares), residuals in ref pixels, implied shift/rotation/scale.
  default_rect(ref)      -> (x0, y0, x1, y1) in ref pixels: the tag rectangle plus a margin.
  warp_crop(query, H, r) -> the query resampled into ref coordinates, cropped to rectangle r.

Refusal rule (loose since 0.2.0: only "roughly the same spot" is needed): ok = False only when fewer
than MIN_TAGS tags are shared by both images, or the
residual is gross: in-fit RMS > max_rms_px (MAX_RMS_PX) or, with 4+ tags, leave-one-tag-out
prediction error > max_loo_px (MAX_LOO_PX). Otherwise ok = True and the residuals are recorded.
Outliers: while more than MIN_TAGS tags remain, the tag with the largest per-tag in-fit RMS is
dropped and the fit repeated if that RMS exceeds OUTLIER_PX (a misdetected or partly covered corner);
dropped IDs are reported.
Why the leave-one-out check: four tags are only one tag more than a homography needs, so a WHOLE tag
that moved between reference and query (tape knocked) is absorbed by the fit (12 px move -> in-fit
RMS 0.6 px on now.jpg). Predicting each tag from the other three exposes it (12 px -> 11.5-13.6 px vs
a synthetic noise floor <= 2.0 px), but with four tags it cannot say reliably WHICH tag moved, so it
refuses instead of dropping. With exactly 3 shared tags a moved tag cannot be detected at all.
With exactly 2 shared tags (e.g. two tags cut off at a frame edge) a similarity transform (shift,
rotation, scale; no keystone) is fitted instead of a homography.

Geometry caveat: the homography is exact only for the front plane the tags lie on. Objects deeper in
the shelf (up to 30.5 cm behind it) show a small residual parallax that grows with the error in stop
placement (design_far.md measured ~3 px per cm of lateral lens error between the #3 front plane and
the back wall). With per-stop references and stops that repeat to a few cm this is acceptable; the
evidence step must tolerate a few pixels.

Summary convention (as in ecc.py): shift (dx, dy) is where the reference's tag-rectangle centre
appears in the query (dx > 0 = content moved right); rot_deg > 0 = query content rotated clockwise
on screen; scale > 1 = query content larger. All are taken from the query <- ref map at that centre.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import NamedTuple

import cv2
import numpy as np

TAGS_VERSION = "0.2.0"
DICTIONARY = cv2.aruco.DICT_4X4_50
# 0.2.0 (2026-10-13, DECISIONS: "roughly the same spot" is enough): refuse only gross failures.
# Was MIN_TAGS 3 / RMS 3 / LOO 6 / outlier 3 px. Noise floors on now.jpg: in-fit RMS <= 0.28 px,
# leave-one-out <= 2.0 px; real cross-viewpoint pairs reach LOO 6.7 px (2026-10-13 tripod check2).
MIN_TAGS = 2        # 2 tags -> similarity fit (shift, rotation, scale); 3+ -> homography
MAX_RMS_PX = 10.0   # gross in-fit RMS
MAX_LOO_PX = 30.0   # gross leave-one-tag-out error (a tag knocked well off its spot)
OUTLIER_PX = 10.0   # per-tag RMS above this -> drop that tag (only while > MIN_TAGS tags remain)
DEFAULT_MARGIN = 40  # px added around the tag rectangle by default_rect


class Alignment(NamedTuple):
    ok: bool
    reason: str          # "" when ok
    H: np.ndarray | None  # 3x3, maps query pixels -> ref pixels
    ids_used: list
    ids_dropped: list
    rms_px: float | None
    max_px: float | None
    per_tag_rms: dict    # id -> RMS residual (px, ref coords) under the final fit
    loo_max_px: float | None  # worst leave-one-tag-out prediction RMS (None with < 4 tags)
    dx: float | None
    dy: float | None
    rot_deg: float | None
    scale: float | None

    def to_json(self) -> dict:
        d = self._asdict()
        d["H"] = None if self.H is None else [[round(float(v), 9) for v in row] for row in self.H]
        for k in ("rms_px", "max_px", "loo_max_px", "dx", "dy", "rot_deg", "scale"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 4)
        d["per_tag_rms"] = {str(k): round(float(v), 4) for k, v in self.per_tag_rms.items()}
        return d


def _detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), params)


def detect(img: np.ndarray) -> dict:
    """{id: 4x2 float64 corners (TL, TR, BR, BL)} for every tag found exactly once."""
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector().detectMarkers(grey)
    if ids is None:
        return {}
    ids = [int(i) for i in ids.ravel()]
    return {i: c.reshape(4, 2).astype(np.float64)
            for i, c in zip(ids, corners) if ids.count(i) == 1}  # duplicated id = ambiguous, skip


def _corners(x) -> dict:
    return x if isinstance(x, dict) else detect(x)


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity src -> dst (Umeyama, no reflection) as a 3x3 matrix."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - ms, dst - md
    U, S, Vt = np.linalg.svd(d.T @ s / len(src))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    c = float(np.trace(np.diag(S) @ D) / np.mean(np.sum(s ** 2, axis=1)))
    H = np.eye(3)
    H[:2, :2] = c * R
    H[:2, 2] = md - c * R @ ms
    return H


def _fit(ref_c: dict, qry_c: dict, ids: list):
    src = np.concatenate([qry_c[i] for i in ids]).astype(np.float64)
    dst = np.concatenate([ref_c[i] for i in ids]).astype(np.float64)
    if len(ids) < 3:
        H = _similarity(src, dst)
    else:
        H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        return None, None
    mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(mapped - dst, axis=1)
    return H, err


def _loo_max(ref_c: dict, qry_c: dict, ids: list):
    """Worst RMS error of predicting one tag's ref corners from a fit on the other tags."""
    if len(ids) < 4:  # predict each tag from a homography on the other three
        return None
    worst = 0.0
    for t in ids:
        H, _ = _fit(ref_c, qry_c, [i for i in ids if i != t])
        if H is None:
            return math.inf
        p = cv2.perspectiveTransform(qry_c[t].reshape(-1, 1, 2), H).reshape(-1, 2)
        worst = max(worst, float(np.sqrt(np.mean(np.sum((p - ref_c[t]) ** 2, axis=1)))))
    return worst


def _summary(H: np.ndarray, ref_c: dict, ids: list):
    """Shift / rotation / scale of the ref -> query map at the centre of the shared tags."""
    Hi = np.linalg.inv(H)
    c = np.concatenate([ref_c[i] for i in ids]).mean(axis=0)
    q = cv2.perspectiveTransform(c.reshape(1, 1, 2), Hi).reshape(2)
    e = 1.0  # numerical Jacobian of ref -> query at c
    qx = cv2.perspectiveTransform((c + [e, 0]).reshape(1, 1, 2), Hi).reshape(2) - q
    qy = cv2.perspectiveTransform((c + [0, e]).reshape(1, 1, 2), Hi).reshape(2) - q
    a, b, cc, d = qx[0], qy[0], qx[1], qy[1]
    rot = math.degrees(math.atan2(cc - b, a + d))
    scale = math.sqrt(abs(a * d - b * cc))
    return float(q[0] - c[0]), float(q[1] - c[1]), rot, scale


def align(ref, query, max_rms_px: float = MAX_RMS_PX, max_loo_px: float = MAX_LOO_PX,
          outlier_px: float = OUTLIER_PX) -> Alignment:
    """Fit query -> ref. `ref` and `query` are images or detect() dicts."""
    ref_c, qry_c = _corners(ref), _corners(query)
    ids = sorted(set(ref_c) & set(qry_c))

    def refuse(reason, **kw):
        base = dict(H=None, ids_used=ids, ids_dropped=[], rms_px=None, max_px=None,
                    per_tag_rms={}, loo_max_px=None, dx=None, dy=None, rot_deg=None, scale=None)
        base.update(kw)
        return Alignment(False, reason, **base)

    if len(ids) < MIN_TAGS:
        return refuse(f"only {len(ids)} shared tags (ref {sorted(ref_c)}, query {sorted(qry_c)}); "
                      f"need {MIN_TAGS}")
    dropped = []
    while True:
        H, err = _fit(ref_c, qry_c, ids)
        if H is None:
            return refuse("homography fit failed", ids_dropped=dropped)
        per_tag = {i: float(np.sqrt(np.mean(err[4 * k:4 * k + 4] ** 2))) for k, i in enumerate(ids)}
        worst = max(per_tag, key=per_tag.get)
        if len(ids) > MIN_TAGS and per_tag[worst] > outlier_px:
            dropped.append(worst)
            ids = [i for i in ids if i != worst]
            continue
        break
    rms, mx = float(np.sqrt(np.mean(err ** 2))), float(err.max())
    loo = _loo_max(ref_c, qry_c, ids)
    dx, dy, rot, scale = _summary(H, ref_c, ids)
    reason = ""
    if rms > max_rms_px:
        reason = f"RMS residual {rms:.2f} px > {max_rms_px} px"
    elif loo is not None and loo > max_loo_px:
        reason = f"tags inconsistent: leave-one-out error {loo:.2f} px > {max_loo_px} px (a tag moved?)"
    return Alignment(not reason, reason, H, ids, dropped, rms, mx, per_tag, loo, dx, dy, rot, scale)


def default_rect(ref, margin: int = DEFAULT_MARGIN) -> tuple:
    """Bounding box of all ref tag corners plus `margin` px, as ints (x0, y0, x1, y1)."""
    pts = np.concatenate(list(_corners(ref).values()))
    x0, y0 = np.floor(pts.min(axis=0)) - margin
    x1, y1 = np.ceil(pts.max(axis=0)) + margin
    return int(x0), int(y0), int(x1), int(y1)


def warp_crop(query: np.ndarray, H: np.ndarray, rect: tuple) -> np.ndarray:
    """Resample `query` into ref coordinates through H (query -> ref) and cut out `rect`
    (x0, y0, x1, y1 in ref pixels). Pixels from outside the query frame are black."""
    x0, y0, x1, y1 = rect
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    return cv2.warpPerspective(query, T @ H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _load_ref(path: str):
    if path.endswith(".json"):  # the --json output of a previous run (its "corners")
        with open(path) as f:
            return {int(k): np.array(v, np.float64) for k, v in json.load(f)["corners"].items()}
    return _read(path)


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"tags.py: cannot read image {path}")
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Detect ArUco tags and register IMG onto a reference.")
    ap.add_argument("img")
    ap.add_argument("--ref", help="reference image, or a --json output file (default: IMG itself)")
    ap.add_argument("--out-crop", help="write the registered crop (default rect: tags + margin)")
    ap.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    ap.add_argument("--max-rms", type=float, default=MAX_RMS_PX)
    ap.add_argument("--max-loo", type=float, default=MAX_LOO_PX)
    ap.add_argument("--json", action="store_true", help="print one JSON object")
    a = ap.parse_args(argv)

    img = _read(a.img)
    qry_c = detect(img)
    ref = _load_ref(a.ref) if a.ref else qry_c
    ref_c = _corners(ref)
    al = align(ref_c, qry_c, max_rms_px=a.max_rms, max_loo_px=a.max_loo)
    rect = default_rect(ref_c, a.margin) if ref_c else None
    if al.ok and a.out_crop:
        if not cv2.imwrite(a.out_crop, warp_crop(img, al.H, rect)):
            raise SystemExit(f"tags.py: cannot write {a.out_crop}")
    out = {"version": TAGS_VERSION, "img": a.img, "ref": a.ref,
           "corners": {str(k): v.round(3).tolist() for k, v in sorted(qry_c.items())},
           "crop_rect": rect, "out_crop": a.out_crop if al.ok else None, **al.to_json()}
    if a.json:
        print(json.dumps(out))
    else:
        print(f"tags found: {sorted(qry_c)}" + (f"  ref: {sorted(ref_c)}" if a.ref else ""))
        if al.ok:
            loo = "-" if al.loo_max_px is None else f"{al.loo_max_px:.3f}"
            print(f"ok  used {al.ids_used} dropped {al.ids_dropped}  rms {al.rms_px:.3f} px  "
                  f"max {al.max_px:.3f} px  loo {loo} px  shift ({al.dx:+.2f}, {al.dy:+.2f}) px  "
                  f"rot {al.rot_deg:+.3f} deg  scale {al.scale:.4f}  crop {rect}")
        else:
            print(f"REFUSED: {al.reason}")
    return 0 if al.ok else 1


if __name__ == "__main__":
    sys.exit(main())
