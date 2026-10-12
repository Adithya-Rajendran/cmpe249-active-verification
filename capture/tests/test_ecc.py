"""Tests for capture/ecc.py (numpy + cv2; runs on the GB10 via uv and on the Pi's system python3).

Run from the repo root (GB10):  uv run --no-sync python -m unittest discover -s capture/tests -v
Run on the Pi:                  cd ~/cmpe249 && python3 -m unittest discover -s capture/tests -v

Sub-pixel test shifts are made by an exact Fourier shift of a LARGER canvas that is then cropped
(`shifted`), so content enters and leaves at the borders as in a real camera translation (not a
circular np.roll of the test image itself). cv2.warpAffine is NOT used as the ground-truth
generator for the 0.01 px checks: it quantises sub-pixel positions to 1/32 px (INTER_BITS=5) and
its interpolation kernels add a fraction-dependent bias; on this texture INTER_CUBIC put 0.3 px at
0.328 px and INTER_LINEAR at 0.288 px (measured while writing these tests). warpAffine is used
only for the rotation test, with the tolerances stated there.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ecc  # noqa: E402

H, W = 360, 480
QUAD_PX, QUAD_SHA = 14403, "c22c1445689229fcfb4a8f3f3d09c67089abe372b35adadf8c65b8eed03695f5"
BORDER = 24  # pixels near the border are excluded from the error check masks (warp fill)


PAD = 40  # canvas margin cropped away after the Fourier shift


def canvas(seed: int = 0, h: int = H + 2 * PAD, w: int = W + 2 * PAD) -> np.ndarray:
    """Smooth multi-scale random texture, float64, roughly 30..225 (the larger, uncropped canvas)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for s, amp in ((2.0, 1.0), (5.0, 1.5), (12.0, 2.0)):
        n = rng.standard_normal((h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), s)
        acc += amp * n / (n.std() + 1e-6)
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    return 30.0 + 195.0 * acc.astype(np.float64)


def texture(seed: int = 0) -> np.ndarray:
    """The H x W reference: the canvas interior, uint8."""
    return np.round(canvas(seed)[PAD:-PAD, PAD:-PAD]).astype(np.uint8)


def shifted(seed: int, dx: float, dy: float) -> np.ndarray:
    """The same scene with its content moved by (dx, dy) px: exact Fourier shift of the canvas,
    then the same interior crop as `texture` (non-circular at the crop borders), uint8."""
    big = canvas(seed)
    ky = np.fft.fftfreq(big.shape[0])[:, None]
    kx = np.fft.fftfreq(big.shape[1])[None, :]
    out = np.fft.ifft2(np.fft.fft2(big) * np.exp(-2j * np.pi * (kx * dx + ky * dy))).real
    return np.round(np.clip(out[PAD:-PAD, PAD:-PAD], 0, 255)).astype(np.uint8)


def warped(img: np.ndarray, dx: float, dy: float, rot_deg: float = 0.0) -> np.ndarray:
    """cv2.warpAffine version: content moved by (dx, dy) and rotated by rot_deg about the image
    centre (+ = clockwise on screen). Sub-pixel accuracy limited to ~1/32 px (see module doc)."""
    h, w = img.shape[:2]
    c = ((w - 1) / 2.0, (h - 1) / 2.0)
    m = cv2.getRotationMatrix2D(c, -rot_deg, 1.0)  # OpenCV: + angle = counter-clockwise on screen
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def inner_mask(h: int = H, w: int = W, b: int = BORDER) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[b:h - b, b:w - b] = 1
    return m


class TestTranslation(unittest.TestCase):
    def setUp(self):
        self.ref = texture(0)
        self.mask = inner_mask()

    def test_subpixel_shifts_recovered(self):
        for dx, dy in [(0.0, 0.0), (0.3, -0.7), (-1.25, 0.4), (2.6, 1.9), (-0.05, 0.08)]:
            img = shifted(0, dx, dy)
            r = ecc.ecc_translation(self.ref, img)
            self.assertAlmostEqual(r.dx, dx, delta=0.01, msg=f"dx for {(dx, dy)}: {r}")
            self.assertAlmostEqual(r.dy, dy, delta=0.01, msg=f"dy for {(dx, dy)}: {r}")
            self.assertGreater(r.cc, 0.99)
            # Masked too: ecc.py finishes with a numpy Gauss-Newton step, so OpenCV's fixed-point
            # warp and 4.10's masked bias (median 0.012 px before the refinement step, measured on
            # the Pi) do not reach the result.
            rm = ecc.ecc_translation(self.ref, img, mask=self.mask)
            self.assertAlmostEqual(rm.dx, dx, delta=0.01, msg=f"masked dx for {(dx, dy)}: {rm}")
            self.assertAlmostEqual(rm.dy, dy, delta=0.01, msg=f"masked dy for {(dx, dy)}: {rm}")

    def test_large_shift_with_pyramid(self):
        dx, dy = 9.4, -6.2
        img = shifted(0, dx, dy)
        r = ecc.ecc_translation(self.ref, img, mask=self.mask, levels=3)
        self.assertAlmostEqual(r.dx, dx, delta=0.01)
        self.assertAlmostEqual(r.dy, dy, delta=0.01)

    def test_edges_mode_recovers_shift_under_gain_and_gamma_change(self):
        dx, dy = -1.7, 0.45
        img = shifted(0, dx, dy).astype(np.float32)
        img = np.clip(255.0 * (img / 255.0) ** 0.8 * 0.85 + 6.0, 0, 255).astype(np.uint8)  # nonlinear exposure change
        r = ecc.ecc_translation(self.ref, img, mask=self.mask, mode="edges")
        self.assertAlmostEqual(r.dx, dx, delta=0.01)
        self.assertAlmostEqual(r.dy, dy, delta=0.01)

    def test_inputs_unchanged(self):
        img = shifted(0, 0.6, -0.3)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        copies = [a.copy() for a in (self.ref, img, bgr, self.mask)]
        flags = [a.flags.writeable for a in (self.ref, img, bgr, self.mask)]
        ecc.ecc_translation(self.ref, img, mask=self.mask)
        ecc.ecc_translation(self.ref, bgr, mask=self.mask, mode="edges")
        ecc.ecc_euclidean(self.ref, img, mask=self.mask)
        ecc.ecc_translation_robust(self.ref, img, mask=self.mask)
        for a, c, f in zip((self.ref, img, bgr, self.mask), copies, flags):
            self.assertTrue(np.array_equal(a, c))
            self.assertEqual(a.dtype, c.dtype)
            self.assertEqual(a.flags.writeable, f)

    def test_read_only_inputs_accepted(self):
        img = shifted(0, 0.5, 0.5)
        ro_ref, ro_img = self.ref.copy(), img.copy()
        ro_ref.flags.writeable = False
        ro_img.flags.writeable = False
        r = ecc.ecc_translation(ro_ref, ro_img, mask=self.mask)
        self.assertAlmostEqual(r.dx, 0.5, delta=0.01)

    def test_masked_registration_ignores_moving_region(self):
        dx, dy = 0.8, -0.6
        img = shifted(0, dx, dy)
        # a large patch that moved differently (an "occluder" placed in the scene)
        y0, y1, x0, x1 = 100, 260, 150, 330
        img[y0:y1, x0:x1] = shifted(7, 5.0, 3.0)[y0:y1, x0:x1]
        m = self.mask.copy()
        m[y0 - 12:y1 + 12, x0 - 12:x1 + 12] = 0
        r = ecc.ecc_translation(self.ref, img, mask=m)
        self.assertAlmostEqual(r.dx, dx, delta=0.01)
        self.assertAlmostEqual(r.dy, dy, delta=0.01)
        # without the mask, OpenCV's ECC alone (refine=False) is biased by the moving region (or fails)
        try:
            bad = ecc.ecc_translation(self.ref, img, mask=self.mask, refine=False)
            self.assertGreater(math.hypot(bad.dx - dx, bad.dy - dy), 0.01)
        except ecc.EccError:
            pass

    def test_mask_is_in_ref_coordinates(self):
        """The mask excludes exactly the ref-frame footprint of a region that changed. The changed
        content sits at that footprint + the shift in img, so a mask taken in img coordinates would
        let a 10 px strip of it leak in (16 px shift minus the 6 px margin kept for the blur)."""
        dx, dy = 16.0, 0.0
        img = shifted(0, dx, dy)
        y0, y1, x0, x1 = 100, 260, 150, 330
        img[y0:y1, x0 + 16:x1 + 16] = 250  # flat bright block where the ref region now appears
        m = self.mask.copy()
        m[y0 - 6:y1 + 6, x0 - 6:x1 + 6] = 0
        r = ecc.ecc_translation(self.ref, img, mask=m)
        self.assertAlmostEqual(r.dx, dx, delta=0.01)
        self.assertAlmostEqual(r.dy, dy, delta=0.01)

    def test_robust_two_pass_drops_changed_region_without_a_mask(self):
        dx, dy = -0.9, 0.35
        img = shifted(0, dx, dy)
        y0, y1, x0, x1 = 120, 240, 180, 300
        img[y0:y1, x0:x1] = 240  # flat bright occluder
        r, m2 = ecc.ecc_translation_robust(self.ref, img, mask=self.mask)
        self.assertAlmostEqual(r.dx, dx, delta=0.01)
        self.assertAlmostEqual(r.dy, dy, delta=0.01)
        self.assertEqual(int(m2[y0 + 10:y1 - 10, x0 + 10:x1 - 10].sum()), 0)

    def test_errors(self):
        with self.assertRaises(ecc.EccError):
            ecc.ecc_translation(self.ref, self.ref[:-1], mask=None)
        with self.assertRaises(ecc.EccError):
            ecc.ecc_translation(self.ref, self.ref, mask=np.zeros_like(self.mask))
        with self.assertRaises(ecc.EccError):
            ecc.ecc_translation(self.ref, self.ref, mode="phase")
        with self.assertRaises(ecc.EccError):
            ecc.ecc_translation(self.ref, self.ref, mask=self.mask[:-1])
        flat = np.full_like(self.ref, 128)
        with self.assertRaises(ecc.EccError):
            ecc.ecc_translation(self.ref, flat)


def tone(img: np.ndarray, gamma: float = 1.0, gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    return np.clip(255.0 * (img.astype(np.float32) / 255.0) ** gamma * gain + offset, 0, 255).astype(np.uint8)


def soft_shadow(img: np.ndarray, depth: float) -> np.ndarray:
    """Multiply by 1 - depth * Gaussian blob (left third of the frame): a spatially varying darkening."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f = 1.0 - depth * np.exp(-(((xx - w * 0.3) / (w * 0.15)) ** 2 + ((yy - h * 0.5) / (h * 0.25)) ** 2))
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


class TestToneAndShadow(unittest.TestCase):
    """Pins what each registration mode tolerates (review finding ECC-EDGES-ROBUSTNESS-CLAIM).
    edges is chosen for shadows / local lighting changes, NOT for global tone changes, where
    intensity is at least as accurate and keeps a higher cc."""

    def setUp(self):
        self.ref = texture(0)
        self.mask = inner_mask()
        self.dx, self.dy = -1.7, 0.45

    def err(self, img, mode):
        r = ecc.ecc_translation(self.ref, img, mask=self.mask, mode=mode)
        return math.hypot(r.dx - self.dx, r.dy - self.dy), r.cc

    def test_linear_gain_offset_both_modes_exact(self):
        img = tone(shifted(0, self.dx, self.dy), gain=0.7, offset=10)
        for mode in ecc.MODES:
            e, cc = self.err(img, mode)
            self.assertLess(e, 0.01, mode)
            self.assertGreater(cc, 0.99, mode)

    def test_global_gamma_and_clipping_intensity_not_worse(self):
        for gamma, gain in ((0.6, 1.0), (1.5, 1.0), (0.6, 1.4)):  # (0.6, 1.4) clips the highlights
            img = tone(shifted(0, self.dx, self.dy), gamma=gamma, gain=gain)
            ei, cci = self.err(img, "intensity")
            ee, cce = self.err(img, "edges")
            msg = f"gamma {gamma} gain {gain}: intensity {ei:.4f}/{cci:.4f}, edges {ee:.4f}/{cce:.4f}"
            self.assertLess(ei, 0.02, msg)
            self.assertLess(ee, 0.05, msg)  # far below the 2 px flag, but not better than intensity
            self.assertGreater(cci, cce, msg)  # edges' cc drops faster (matters for pose_check min_cc)

    def test_soft_shadow_edges_much_less_biased(self):
        img = soft_shadow(shifted(0, self.dx, self.dy), 0.45)
        ei, cci = self.err(img, "intensity")
        ee, cce = self.err(img, "edges")
        msg = f"intensity {ei:.4f}/{cci:.4f}, edges {ee:.4f}/{cce:.4f}"
        self.assertLess(ee, 0.02, msg)
        self.assertGreater(ei, 0.05, msg)
        self.assertGreater(ei, 5 * ee, msg)
        self.assertGreater(cce, cci, msg)


class TestEuclidean(unittest.TestCase):
    def test_roll_and_centre_shift_recovered(self):
        ref = texture(3)
        m = inner_mask(b=40)
        for rot, dx, dy in [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (-0.8, 1.2, -0.4), (0.25, -2.0, 1.0)]:
            img = warped(ref, dx, dy, rot)
            r = ecc.ecc_euclidean(ref, img, mask=m)
            self.assertAlmostEqual(r.rot_deg, rot, delta=0.01, msg=f"{(rot, dx, dy)} -> {r}")
            self.assertAlmostEqual(r.dx, dx, delta=0.02, msg=f"{(rot, dx, dy)} -> {r}")
            self.assertAlmostEqual(r.dy, dy, delta=0.02, msg=f"{(rot, dx, dy)} -> {r}")


class TestMasks(unittest.TestCase):
    def test_regions_mask(self):
        spec = {"base": "all",
                "exclude": [{"rect": [10, 20, 30, 25]}, {"polygon": [[50, 50], [60, 50], [60, 60], [50, 60]]}],
                "include": [{"rect": [0, 0, 5, 5]}]}
        m = ecc.regions_mask(spec, 100, 80)
        self.assertEqual(m.shape, (80, 100))
        self.assertEqual(m.dtype, np.uint8)
        self.assertEqual(int(m[20:25, 10:30].sum()), 0)
        self.assertEqual(int(m[19, 10]), 1)
        self.assertEqual(int(m[20, 30]), 1)   # x1 exclusive
        self.assertEqual(int(m[55, 55]), 0)
        self.assertEqual(int(m[0, 0]), 1)
        n = ecc.regions_mask({"base": "none", "include": [{"rect": [0, 0, 10, 10]}]}, 100, 80)
        self.assertEqual(int(n.sum()), 100)
        self.assertEqual(ecc.mask_sha256(m), ecc.mask_sha256(m * 255))
        with self.assertRaises(ValueError):
            ecc.regions_mask({"base": "some"}, 10, 10)

    def test_fill_polygon_rule_and_known_hash(self):
        tri = ecc.fill_polygon([[0, 0], [10, 0], [0, 10]], 12, 12)
        self.assertTrue(tri[0, 0])
        self.assertTrue(tri[0, 9])
        self.assertFalse(tri[0, 10])     # right end exclusive
        self.assertTrue(tri[5, 4])
        self.assertFalse(tri[5, 5])      # on the hypotenuse x = 10 - y: exclusive
        self.assertFalse(tri[10, 0])     # bottom vertex row: half-open in y
        self.assertEqual(int(tri.sum()), 55)
        # a slanted quad like the mug-compartment polygon; the hash pins the rasterisation so the
        # GB10 (OpenCV 5.0) and the Pi (OpenCV 4.10) must agree
        q = ecc.fill_polygon([[11, 3], [203, 4], [200, 80], [13, 79]], 220, 90)
        self.assertEqual(int(q.sum()), QUAD_PX)
        self.assertEqual(ecc.mask_sha256(q), QUAD_SHA)


if __name__ == "__main__":
    unittest.main()
