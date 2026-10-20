"""Tests for tags.py: synthetic tag scenes with known homographies, refusals, the CLI, and (GB10
only) the real 2026-10-12 frame.

Synthetic scene: smooth random texture with four DICT_4X4_50 markers (IDs 0-3, 60 px, white quiet
zone) near the corners, like the shelf tags. Queries are the reference warped by a known
homography (cv2.warpPerspective), optionally blurred, noised and JPEG re-encoded. Accuracy is
checked as the mapping error of the fitted query -> ref homography over a grid covering the
default crop rectangle, against the true inverse.
"""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tags  # noqa: E402

H_IMG, W_IMG = 720, 960
MARK, PAD = 60, 20
NOW_JPG = HERE.parents[1] / "data" / "raw" / "2026-10-12_scene" / "now.jpg"


def scene(seed: int = 0, ids=(0, 1, 2, 3)) -> np.ndarray:
    """BGR reference: texture plus markers at lower-left (0), lower-right (1), upper-left (2),
    upper-right (3), as on the shelf."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((H_IMG, W_IMG), np.float32)
    for s in (3.0, 8.0, 20.0):
        n = cv2.GaussianBlur(rng.standard_normal((H_IMG, W_IMG)).astype(np.float32), (0, 0), s)
        acc += n / (n.std() + 1e-6)
    g = (60 + 140 * (acc - acc.min()) / (acc.max() - acc.min())).astype(np.uint8)
    d = cv2.aruco.getPredefinedDictionary(tags.DICTIONARY)
    pos = {0: (150, 560), 1: (750, 560), 2: (150, 100), 3: (750, 100)}  # top-left of each marker
    for i in ids:
        x, y = pos[i]
        g[y - PAD:y + MARK + PAD, x - PAD:x + MARK + PAD] = 255
        g[y:y + MARK, x:x + MARK] = cv2.aruco.generateImageMarker(d, i, MARK, borderBits=1)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def similarity(dx, dy, roll_deg, s):
    """ref -> query: rotate by roll_deg and scale by s about the image centre, then shift."""
    a = math.radians(roll_deg)
    A = s * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.array([W_IMG / 2, H_IMG / 2])
    t = c - A @ c + [dx, dy]
    return np.array([[A[0, 0], A[0, 1], t[0]], [A[1, 0], A[1, 1], t[1]], [0, 0, 1]])


def warp(img, Ht, degrade=False, seed=0):
    q = cv2.warpPerspective(img, Ht, (W_IMG, H_IMG), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    if degrade:
        q = cv2.GaussianBlur(q, (0, 0), 1.0)
        q = np.clip(q + np.random.default_rng(seed).normal(0, 3, q.shape), 0, 255).astype(np.uint8)
        q = cv2.imdecode(cv2.imencode(".jpg", q, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
    return q


def grid_error(H_fit, Ht, rect):
    """Max |H_fit(Ht(p)) - p| over a grid of ref points p in rect."""
    gx, gy = np.meshgrid(np.linspace(rect[0], rect[2], 15), np.linspace(rect[1], rect[3], 15))
    p = np.stack([gx.ravel(), gy.ravel()], 1).reshape(-1, 1, 2)
    back = cv2.perspectiveTransform(cv2.perspectiveTransform(p, Ht), H_fit)
    return float(np.linalg.norm((back - p).reshape(-1, 2), axis=1).max())


class TestDetect(unittest.TestCase):
    def test_finds_all_four_at_expected_corners(self):
        c = tags.detect(scene())
        self.assertEqual(sorted(c), [0, 1, 2, 3])
        for i, (x, y) in {0: (150, 560), 1: (750, 560), 2: (150, 100), 3: (750, 100)}.items():
            self.assertEqual(c[i].shape, (4, 2))
            # outer black square corners, TL first; pixel-centre convention puts the edge at -0.5
            np.testing.assert_allclose(c[i][0], (x - 0.5, y - 0.5), atol=0.6)
            np.testing.assert_allclose(c[i][2], (x + MARK - 0.5, y + MARK - 0.5), atol=0.6)

    def test_grey_input_and_no_tags(self):
        self.assertEqual(sorted(tags.detect(cv2.cvtColor(scene(), cv2.COLOR_BGR2GRAY))), [0, 1, 2, 3])
        self.assertEqual(tags.detect(np.full((200, 200, 3), 128, np.uint8)), {})


class TestAlign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = scene()
        cls.rc = tags.detect(cls.ref)
        cls.rect = tags.default_rect(cls.rc)

    def check(self, Ht, degrade, tol):
        q = warp(self.ref, Ht, degrade)
        al = tags.align(self.rc, q)
        self.assertTrue(al.ok, al.reason)
        self.assertEqual(al.ids_used, [0, 1, 2, 3])
        self.assertEqual(al.ids_dropped, [])
        self.assertLess(grid_error(al.H, Ht, self.rect), tol)
        self.assertLess(al.rms_px, 1.0)
        return al

    def test_identity_is_exact(self):
        al = tags.align(self.ref, self.ref)
        self.assertTrue(al.ok)
        np.testing.assert_allclose(al.H / al.H[2, 2], np.eye(3), atol=1e-9)
        self.assertAlmostEqual(al.rms_px, 0.0, places=6)
        self.assertAlmostEqual(al.loo_max_px, 0.0, places=6)

    def test_round_trip_shift_roll_scale(self):
        for degrade in (False, True):
            with self.subTest(degrade=degrade):
                al = self.check(similarity(20, -12, 1.0, 1.02), degrade, tol=0.5)
                self.assertAlmostEqual(al.rot_deg, 1.0, delta=0.05)
                self.assertAlmostEqual(al.scale, 1.02, delta=0.002)

    def test_pure_shift_summary(self):
        al = self.check(similarity(20, -12, 0, 1), True, tol=0.5)
        self.assertAlmostEqual(al.dx, 20, delta=0.2)
        self.assertAlmostEqual(al.dy, -12, delta=0.2)

    def test_keystone(self):
        Ht = np.array([[1, 0, 0], [0, 1, 0], [3e-5, 0, 1]]) @ similarity(-15, 10, -1, 0.98)
        self.check(Ht, True, tol=0.5)

    def test_warp_crop_matches_reference(self):
        q = warp(self.ref, similarity(20, -12, 1.0, 1.02), degrade=True)
        al = tags.align(self.rc, q)
        a = tags.warp_crop(self.ref, np.eye(3), self.rect).astype(float)
        b = tags.warp_crop(q, al.H, self.rect).astype(float)
        x0, y0, x1, y1 = self.rect
        self.assertEqual(b.shape, (y1 - y0, x1 - x0, 3))
        self.assertLess(np.abs(a - b).mean(), 6.0)

    def test_two_tags_similarity(self):
        Ht = similarity(20, -12, 1.0, 1.02)
        al = tags.align(self.rc, warp(scene(ids=(2, 3)), Ht, True))
        self.assertTrue(al.ok, al.reason)
        self.assertEqual(al.ids_used, [2, 3])
        self.assertIsNone(al.loo_max_px)
        self.assertLess(grid_error(al.H, Ht, self.rect), 2.0)
        self.assertAlmostEqual(al.rot_deg, 1.0, delta=0.1)

    def test_refuses_with_one_tag(self):
        al = tags.align(self.rc, scene(ids=(3,)))
        self.assertFalse(al.ok)
        self.assertIn("1 shared tags", al.reason)
        self.assertIsNone(al.H)
        self.assertFalse(tags.align(scene(ids=(2,)), self.ref).ok)  # either image

    def test_three_tags_ok(self):
        al = tags.align(self.rc, warp(scene(ids=(0, 1, 2)), similarity(10, 5, 0.5, 1.0), True))
        self.assertTrue(al.ok, al.reason)
        self.assertEqual(al.ids_used, [0, 1, 2])
        self.assertIsNone(al.loo_max_px)

    def test_bad_corner_dropped(self):
        qc = {k: v.copy() for k, v in self.rc.items()}
        qc[1][2] += [30, 15]
        al = tags.align(self.rc, qc)
        self.assertTrue(al.ok)
        self.assertEqual(al.ids_dropped, [1])
        self.assertEqual(al.ids_used, [0, 2, 3])

    def test_moved_tag_refused(self):
        qc = {k: v.copy() for k, v in self.rc.items()}
        qc[3] += [40, 0]  # whole tag moved: in-fit RMS stays small, leave-one-out exposes it
        al = tags.align(self.rc, qc)
        self.assertLess(al.rms_px, tags.MAX_RMS_PX)
        self.assertFalse(al.ok)
        self.assertIn("leave-one-out", al.reason)

    def test_gross_rms_refused(self):
        qc = {k: v.copy() for k, v in self.rc.items() if k in (2, 3)}
        qc[3] = np.roll(qc[3], 1, axis=0)  # tag 3 turned 90 deg; only 2 tags, so nothing can be dropped
        al = tags.align(self.rc, qc)
        self.assertFalse(al.ok)
        self.assertIn("RMS residual", al.reason)


class TestCli(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(HERE.parent / "tags.py"), *args],
                              capture_output=True, text=True, timeout=60)

    def test_json_crop_and_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            ref, qry, two = (str(Path(d) / n) for n in ("ref.png", "q.png", "two.png"))
            cv2.imwrite(ref, scene())
            cv2.imwrite(qry, warp(scene(), similarity(20, -12, 1.0, 1.02), True))
            cv2.imwrite(two, scene(ids=(1,)))  # one tag: refused
            crop = str(Path(d) / "crop.png")
            r = self.run_cli(qry, "--ref", ref, "--out-crop", crop, "--json")
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout)
            self.assertTrue(out["ok"])
            self.assertEqual(sorted(out["corners"]), ["0", "1", "2", "3"])
            x0, y0, x1, y1 = out["crop_rect"]
            self.assertEqual(cv2.imread(crop).shape[:2], (y1 - y0, x1 - x0))
            # a saved --json output can serve as the reference
            refjson = Path(d) / "ref.json"
            refjson.write_text(self.run_cli(ref, "--json").stdout)
            r = self.run_cli(qry, "--ref", str(refjson), "--json")
            self.assertEqual(r.returncode, 0, r.stderr)
            r = self.run_cli(two, "--ref", ref)
            self.assertEqual(r.returncode, 1)
            self.assertIn("REFUSED", r.stdout)


@unittest.skipUnless(NOW_JPG.exists(), "GB10 only: needs data/raw/2026-10-12_scene/now.jpg")
class TestRealFrame(unittest.TestCase):
    def test_now_jpg_ids_0_to_3(self):
        c = tags.detect(cv2.imread(str(NOW_JPG)))
        self.assertEqual(sorted(c), [0, 1, 2, 3])
        expected = {0: (778, 1107), 1: (1807, 1073), 2: (713.5, 185), 3: (1810, 125.5)}
        for i, xy in expected.items():
            np.testing.assert_allclose(c[i].mean(axis=0), xy, atol=1.5)


if __name__ == "__main__":
    unittest.main()
