"""ECC image registration shared by the Pi tools and the GB10 analysis (stdlib + numpy + cv2).

Runs unchanged on the Pi (system CPython 3.13.5, apt numpy 2.2 / OpenCV 4.10) and on the GB10
(uv env, numpy 2.5 / OpenCV 5.0). Import it with the `capture/` directory on `sys.path`
(`import ecc`); that is how `pose_check.py`, `qc_session.py`, the tests and
`camera_check/analyze_camcheck.py` load it.

Conventions (all functions):
  * `ref` is the reference (golden / per-stop reference), `img` the image being checked.
  * A returned shift (dx, dy) is where `ref`'s content appears in `img`: img(x + dx, y + dy) ~ ref(x, y).
    dx > 0 = content moved right, dy > 0 = content moved down, in pixels of the input images.
  * `mask` is in `ref` coordinates: nonzero = pixel may be used. It is honoured exactly on both
    OpenCV versions (internally `img` is the ECC template and `ref` the warped input, because
    `findTransformECC`'s only mask argument is the input-image mask on OpenCV 4.10).
  * Inputs are never modified: every array is copied (as float32) before any OpenCV call.
  * mode="intensity" registers smoothed grey levels; mode="edges" registers the gradient magnitude
    of the smoothed image. What each mode tolerates (pinned in tests/test_ecc.py, TestToneAndShadow;
    real-frame numbers in MODULE1 section 5):
      - a spatially varying brightness change (a soft shadow, a lamp moved): edges is much less
        biased (synthetic 45 % shadow: 0.004-0.006 px vs 0.11 px for intensity; cc 0.955 vs 0.875);
      - a global non-linear tone change (gamma, clipping at 255): intensity is somewhat MORE
        accurate and keeps a higher cc (gamma 0.6 x 1.4 with clipping: 0.005 vs 0.02 px, cc 0.976
        vs 0.75). Both stay far below the 2 px QC/pose flag; ECC's correlation already removes a
        linear gain + offset in either mode.
    So edges is NOT generally "less sensitive to exposure": it is chosen for shadows / local
    lighting differences, and its cc drops faster under clipping (min_cc gates in pose_check).
  * Coarse-to-fine: `levels` pyramid levels (factor 2 each) are run first so that shifts of
    10-20 px converge; levels=0 is a single full-resolution run.
  * A `border` (default 24 px) at the image edges is always excluded (ANDed with `mask`). Without
    it OpenCV 4.10 lets pixels warped in from outside the frame bias the estimate (measured on the
    Pi: 0.3 px recovered as 0.295 / -0.7 as -0.690 unmasked); OpenCV 5.0 is less affected.

Failures (no convergence, empty mask, size mismatch) raise EccError.
"""
from __future__ import annotations

import hashlib
import math
from typing import NamedTuple

import cv2
import numpy as np

ECC_VERSION = "1.0.0"
MODES = ("intensity", "edges")
DEFAULT_SIGMA = 1.5
DEFAULT_ITERS = 200
DEFAULT_EPS = 1e-6
DEFAULT_LEVELS = 3
DEFAULT_BORDER = 24  # px (full resolution) always excluded at the image edges, see _run
MIN_MASK_PX = 500  # fewer usable pixels than this at any level -> EccError


class EccError(RuntimeError):
    """Registration could not be computed (bad input, empty mask, or ECC did not converge)."""


class Shift(NamedTuple):
    dx: float
    dy: float
    cc: float  # ECC correlation coefficient at full resolution (1.0 = identical up to gain/offset)


class Rigid(NamedTuple):
    dx: float       # displacement of the image centre (ref centre -> where it appears in img), px
    dy: float
    rot_deg: float  # rotation of img content relative to ref; + = clockwise on screen (y axis down)
    cc: float


# ----------------------------------------------------------------------------- preprocessing
def to_gray_f32(a: np.ndarray) -> np.ndarray:
    """Copy of `a` as a contiguous float32 grey image. Accepts 2-D or 3-channel BGR arrays
    (uint8/uint16/float). Never returns a view of the input."""
    arr = np.asarray(a)
    if arr.ndim == 3 and arr.shape[2] == 3:
        f = arr.astype(np.float32, copy=True)
        g = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]  # BGR (cv2.imread order)
    elif arr.ndim == 2:
        g = arr.astype(np.float32, copy=True)
    else:
        raise EccError(f"expected a 2-D grey or HxWx3 BGR image, got shape {arr.shape}")
    return np.ascontiguousarray(g, dtype=np.float32)


def _blur(g: np.ndarray, sigma: float) -> np.ndarray:
    if sigma and sigma > 0:
        return cv2.GaussianBlur(g, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
                                borderType=cv2.BORDER_REFLECT)
    return g.copy()


def gradient_magnitude(g: np.ndarray, sigma: float = DEFAULT_SIGMA) -> np.ndarray:
    """Gradient magnitude of the sigma-smoothed image (float32, same size). Input is copied."""
    s = _blur(to_gray_f32(g), sigma)
    gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    return cv2.magnitude(gx, gy)


def _prepare(a: np.ndarray, sigma: float, mode: str) -> np.ndarray:
    if mode not in MODES:
        raise EccError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "edges":
        return _blur(gradient_magnitude(a, sigma), sigma)
    return _blur(to_gray_f32(a), sigma)


def _mask_u8(mask, shape) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.shape != tuple(shape):
        raise EccError(f"mask shape {m.shape} does not match image shape {tuple(shape)}")
    return np.ascontiguousarray((m > 0).astype(np.uint8))  # copy, 0/1


def _pyr(a: np.ndarray, levels: int) -> list[np.ndarray]:
    out = [a]
    for _ in range(levels):
        out.append(cv2.pyrDown(out[-1]))
    return out


def _mask_pyr(m: np.ndarray | None, shapes: list[tuple[int, int]]) -> list[np.ndarray | None]:
    if m is None:
        return [None] * len(shapes)
    out = [m]
    for h, w in shapes[1:]:
        # a coarse pixel is usable only if every fine pixel under it is (INTER_AREA mean == 1)
        small = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        out.append(np.ascontiguousarray((small > 0.999).astype(np.uint8)))
    return out


def _find(template: np.ndarray, inp: np.ndarray, warp: np.ndarray, motion: int,
          iters: int, eps: float, mask: np.ndarray | None) -> tuple[float, np.ndarray]:
    if mask is not None and int(mask.sum()) < MIN_MASK_PX:
        raise EccError(f"mask leaves only {int(mask.sum())} usable pixels (< {MIN_MASK_PX})")
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    try:
        # gaussFiltSize=1: smoothing was already applied with the requested sigma
        cc, w = cv2.findTransformECC(template, inp, warp.copy(), motion, crit, mask, 1)
    except cv2.error as e:
        raise EccError(f"ECC did not converge: {str(e).strip().splitlines()[-1]}") from None
    if not np.all(np.isfinite(w)):
        raise EccError("ECC returned a non-finite warp")
    return float(cc), w


def _bilinear(a: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """a sampled at float positions (exact bilinear weights in float64; positions must be inside)."""
    x0 = np.floor(xs).astype(np.intp)
    y0 = np.floor(ys).astype(np.intp)
    fx, fy = xs - x0, ys - y0
    a00, a01 = a[y0, x0], a[y0, x0 + 1]
    a10, a11 = a[y0 + 1, x0], a[y0 + 1, x0 + 1]
    return (a00 * (1 - fx) + a01 * fx) * (1 - fy) + (a10 * (1 - fx) + a11 * fx) * fy


def _refine_translation(r: np.ndarray, i: np.ndarray, m: np.ndarray, dx: float, dy: float,
                        iters: int = 30, tol: float = 1e-5) -> tuple[float, float, float]:
    """Gauss-Newton refinement of a translation in numpy (version-independent, no fixed-point warp).

    Model: r(x) ~ a * i(x + d) + b over the mask (ref coords), linearised in d each iteration;
    i is sampled with exact float bilinear weights. Plain least squares (no trimming: on real
    frames trimming discarded the edge pixels that carry the shift). OpenCV's ECC warps with
    1/32-px fixed-point positions and, on 4.10, carried a masked bias of ~0.012 px (median,
    synthetic tests); this step removes both. Raises EccError if it moves more than 1 px away
    from the starting estimate. Returns (dx, dy, cc)."""
    ys, xs = np.nonzero(m)
    R = r[ys, xs].astype(np.float64)
    i64 = i.astype(np.float64)
    gy, gx = np.gradient(i64)
    h, w = i.shape
    dx0, dy0 = dx, dy
    for _ in range(int(iters)):
        px, py = xs + dx, ys + dy
        ok = (px >= 0) & (py >= 0) & (px < w - 1) & (py < h - 1)
        if ok.sum() < MIN_MASK_PX:
            raise EccError("refinement: too few usable mask pixels")
        px, py, Rk = px[ok], py[ok], R[ok]
        Ik = _bilinear(i64, px, py)
        A = np.stack([Ik, np.ones_like(Ik), _bilinear(gx, px, py), _bilinear(gy, px, py)], axis=1)
        sol, *_ = np.linalg.lstsq(A, Rk, rcond=None)
        a = sol[0]
        if not np.isfinite(sol).all() or abs(a) < 1e-9:
            raise EccError("refinement: degenerate solution (flat image or mask?)")
        ddx, ddy = sol[2] / a, sol[3] / a
        dx, dy = dx + ddx, dy + ddy
        if math.hypot(dx - dx0, dy - dy0) > 1.0:
            raise EccError(f"refinement moved {math.hypot(dx - dx0, dy - dy0):.2f} px from the ECC estimate")
        if math.hypot(ddx, ddy) < tol:
            break
    px, py = xs + dx, ys + dy
    ok = (px >= 0) & (py >= 0) & (px < w - 1) & (py < h - 1)
    Ik, Rk = _bilinear(i64, px[ok], py[ok]), R[ok]
    Ik, Rk = Ik - Ik.mean(), Rk - Rk.mean()
    cc = float((Ik * Rk).sum() / (math.sqrt((Ik * Ik).sum() * (Rk * Rk).sum()) + 1e-12))
    return float(dx), float(dy), cc


def _run(ref, img, mask, sigma, iters, eps, mode, levels, motion, init_warp_inv, border):
    r, i = np.asarray(ref), np.asarray(img)
    if r.shape[:2] != i.shape[:2]:
        raise EccError(f"ref {r.shape[:2]} and img {i.shape[:2]} differ in size")
    levels = int(levels)
    if levels < 0:
        raise EccError("levels must be >= 0")
    pr, pi = _prepare(r, sigma, mode), _prepare(i, sigma, mode)
    m = _mask_u8(mask, pr.shape)
    b = int(border)
    if b > 0:
        if 2 * b >= min(pr.shape):
            raise EccError(f"border {b} px leaves no image")
        if m is None:
            m = np.zeros(pr.shape, np.uint8)
            m[b:-b, b:-b] = 1
        else:
            m[:b, :] = 0
            m[-b:, :] = 0
            m[:, :b] = 0
            m[:, -b:] = 0
    rp, ip = _pyr(pr, levels), _pyr(pi, levels)
    mp = _mask_pyr(m, [a.shape for a in rp])
    # internal warp W maps img (template) coords -> ref (input) coords: ref(W x) ~ img(x)
    w = init_warp_inv.astype(np.float32).copy()
    w[:, 2] /= 2.0 ** levels
    cc = float("nan")
    for lev in range(levels, -1, -1):
        cc, w = _find(ip[lev], rp[lev], w, motion, iters, eps, mp[lev])
        if lev > 0:
            w[:, 2] *= 2.0
    return cc, w, pr, pi, mp[0]


def ecc_translation(ref: np.ndarray, img: np.ndarray, mask: np.ndarray | None = None,
                    sigma: float = DEFAULT_SIGMA, iters: int = DEFAULT_ITERS, eps: float = DEFAULT_EPS,
                    mode: str = "intensity", levels: int = DEFAULT_LEVELS,
                    init: tuple[float, float] = (0.0, 0.0), border: int = DEFAULT_BORDER,
                    refine: bool = True) -> Shift:
    """Translation of `img` relative to `ref` (see module conventions). Returns Shift(dx, dy, cc).
    refine=True (default) finishes with _refine_translation at full resolution; cc is then the
    normalised correlation of the refined fit over the mask."""
    w0 = np.array([[1, 0, -init[0]], [0, 1, -init[1]]], dtype=np.float32)
    cc, w, pr, pi, m = _run(ref, img, mask, sigma, iters, eps, mode, levels, cv2.MOTION_TRANSLATION, w0, border)
    dx, dy = float(-w[0, 2]), float(-w[1, 2])
    if refine:
        mm = m if m is not None else np.ones(pr.shape, np.uint8)
        dx, dy, cc = _refine_translation(pr, pi, mm > 0, dx, dy)
    return Shift(dx, dy, cc)


def ecc_euclidean(ref: np.ndarray, img: np.ndarray, mask: np.ndarray | None = None,
                  sigma: float = DEFAULT_SIGMA, iters: int = DEFAULT_ITERS, eps: float = DEFAULT_EPS,
                  mode: str = "intensity", levels: int = DEFAULT_LEVELS,
                  init: tuple[float, float] = (0.0, 0.0), border: int = DEFAULT_BORDER) -> Rigid:
    """Rigid (rotation + translation) registration. dx, dy are the displacement of the image
    centre; rot_deg the in-plane rotation (camera roll shows up here). Returns Rigid(dx, dy, rot_deg, cc)."""
    w0 = np.array([[1, 0, -init[0]], [0, 1, -init[1]]], dtype=np.float32)
    cc, w, _, _, _ = _run(ref, img, mask, sigma, iters, eps, mode, levels, cv2.MOTION_EUCLIDEAN, w0, border)
    fwd = cv2.invertAffineTransform(w.astype(np.float64))  # ref coords -> img coords
    h, wd = np.asarray(ref).shape[:2]
    c = np.array([(wd - 1) / 2.0, (h - 1) / 2.0, 1.0])
    p = fwd @ c
    rot = math.degrees(math.atan2(fwd[1, 0], fwd[0, 0]))
    return Rigid(float(p[0] - c[0]), float(p[1] - c[1]), float(rot), cc)


def warp_to_ref(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """`img` translated back onto `ref`'s pixel grid (float32 grey copy; borders replicated)."""
    g = to_gray_f32(img)
    m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(g, m, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------------- robust variants
def residual_mask(ref: np.ndarray, img: np.ndarray, dx: float, dy: float,
                  mask: np.ndarray | None = None, block: int = 32, k: float = 4.0,
                  sigma: float = DEFAULT_SIGMA, grow: int = 1) -> np.ndarray:
    """uint8 0/1 mask (ref coords) = `mask` minus the blocks whose registered residual is an
    outlier: median |z(ref) - z(img warped back)| per block > median + k * 1.4826 * MAD over the
    usable blocks (z = zero-mean, unit-variance over the usable pixels); each outlier block also
    removes its neighbours within `grow` blocks (edges of a changed region straddle blocks). Used to drop regions that
    changed between the two images (moved objects, an occluder, a hand) before a second ECC pass."""
    r = _blur(to_gray_f32(ref), sigma)
    i = _blur(warp_to_ref(img, dx, dy), sigma)
    h, w = r.shape
    m = np.ones((h, w), np.uint8) if mask is None else _mask_u8(mask, (h, w))
    sel = m > 0
    if sel.sum() < MIN_MASK_PX:
        raise EccError("mask leaves too few pixels for residual_mask")
    zr = (r - r[sel].mean()) / (r[sel].std() + 1e-6)
    zi = (i - i[sel].mean()) / (i[sel].std() + 1e-6)
    res = np.abs(zr - zi)
    out = m.copy()
    stats = []
    for y0 in range(0, h, block):
        for x0 in range(0, w, block):
            bm = sel[y0:y0 + block, x0:x0 + block]
            if bm.sum() >= 0.25 * bm.size:
                stats.append((y0, x0, float(np.median(res[y0:y0 + block, x0:x0 + block][bm]))))
    if not stats:
        return out
    vals = np.array([s[2] for s in stats])
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    thr = med + k * max(mad, 1e-3)
    g = max(0, int(grow)) * block
    for y0, x0, v in stats:
        if v > thr:
            out[max(0, y0 - g):y0 + block + g, max(0, x0 - g):x0 + block + g] = 0
    return out


def ecc_translation_robust(ref: np.ndarray, img: np.ndarray, mask: np.ndarray | None = None,
                           block: int = 32, k: float = 4.0, grow: int = 1, **kw) -> tuple[Shift, np.ndarray]:
    """Two-pass translation: OpenCV ECC on `mask` (no refinement), drop outlier blocks with
    residual_mask, then ECC + refinement again on the reduced mask starting from the first
    estimate. Returns (Shift, the mask used in pass 2)."""
    kw1 = dict(kw)
    kw1["refine"] = False  # pass 1 is only a starting point; the changed region may still be in the mask
    first = ecc_translation(ref, img, mask=mask, **kw1)
    m2 = residual_mask(ref, img, first.dx, first.dy, mask=mask, block=block, k=k,
                       sigma=kw.get("sigma", DEFAULT_SIGMA), grow=grow)
    kw2 = dict(kw)
    kw2["init"] = (first.dx, first.dy)
    kw2["levels"] = 0
    return ecc_translation(ref, img, mask=m2, **kw2), m2


# ------------------------------------------------------------------------------ masks
def regions_mask(spec: dict, width: int, height: int) -> np.ndarray:
    """Rasterise a region spec (configs/capture_lock.json format) to a uint8 0/1 mask.

    spec = {"base": "all" | "none",
            "include": [region, ...] (optional), "exclude": [region, ...] (optional)}
    region = {"rect": [x0, y0, x1, y1]}            (x1, y1 exclusive, pixels)
           | {"polygon": [[x, y], [x, y], ...]}   (filled by fill_polygon: pixel (x, y) is inside
                                                  when its integer coordinate passes the even-odd
                                                  scanline rule; pure Python/numpy, identical on
                                                  every host - cv2.fillPoly 4.10 and 5.0 differ)
    Order: base, then every include is set to 1, then every exclude to 0."""
    base = spec.get("base", "all")
    if base not in ("all", "none"):
        raise ValueError(f"mask base must be 'all' or 'none', got {base!r}")
    m = np.full((height, width), 1 if base == "all" else 0, np.uint8)

    def paint(region: dict, val: int) -> None:
        if "rect" in region:
            x0, y0, x1, y1 = (int(v) for v in region["rect"])
            m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = val
        elif "polygon" in region:
            m[fill_polygon(region["polygon"], width, height)] = val
        else:
            raise ValueError(f"region needs 'rect' or 'polygon': {region}")

    for r in spec.get("include", []):
        paint(r, 1)
    for r in spec.get("exclude", []):
        paint(r, 0)
    return m


def fill_polygon(points, width: int, height: int) -> np.ndarray:
    """Boolean (height, width) array: pixel (x, y) is True when the point (x, y) lies inside the
    polygon by the even-odd scanline rule with half-open edges (y0 <= y < y1; x from ceil(xa) to
    ceil(xb) - 1). Deterministic across numpy/OpenCV versions."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    out = np.zeros((height, width), bool)
    ys = [p[1] for p in pts]
    for y in range(max(0, math.floor(min(ys))), min(height, math.ceil(max(ys)) + 1)):
        xs = []
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
            if (y0 <= y < y1) or (y1 <= y < y0):
                xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
        xs.sort()
        for xa, xb in zip(xs[0::2], xs[1::2]):
            a, b = max(0, math.ceil(xa)), min(width, math.ceil(xb))
            if b > a:
                out[y, a:b] = True
    return out


def mask_sha256(mask: np.ndarray) -> str:
    """sha256 of the mask as C-order uint8 0/1 bytes (identical across numpy/OpenCV versions)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(mask) > 0, dtype=np.uint8).tobytes()).hexdigest()
