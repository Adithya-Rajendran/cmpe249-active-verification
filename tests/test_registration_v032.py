"""perception/registration_v032.py (candidate C1 of notes/2026-11-04-registration-repair-plan.md) on synthetic images.

Pins: the direction (W maps query pixels to reference pixels and the query is warped by W, not W^-1); C1's numerics
equal the pinned prototype copy (runs/2026-11-04-block-rejection-repair/align_lib.py, when present); C1's pass 1 is
v0.3's fit; the support guard; final checks on the post-repair warp only; and only H reaches the evidence step.
"""
from __future__ import annotations

import hashlib
import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np

from perception import evidence_v03 as v03
from perception import registration_v032 as r32
from tests.test_evidence_v03 import BOX as SMALL_BOX, MeanColour, background, no_detections, with_target

REPO = Path(__file__).resolve().parents[1]
PROTO = REPO / "runs" / "2026-11-04-block-rejection-repair" / "align_lib.py"
PROTO_SHA = "689ad6d1971561beee8fd89b85a061d35076e3bef652918df98183aad194ebf6"
H_, W_ = 360, 640
BOX = (280.0, 150.0, 360.0, 210.0)


def scene(seed: int = 0) -> np.ndarray:
    """A weakly textured surface with strongly textured patches (objects), BGR uint8."""
    rng = np.random.default_rng(seed)
    g = 110 + 35 * cv2.GaussianBlur(rng.normal(0, 1, (H_, W_)).astype(np.float32), (0, 0), 10.0)
    for y0, x0 in ((40, 60), (60, 470), (250, 90), (240, 480), (40, 280), (280, 300)):
        g[y0:y0 + 60, x0:x0 + 60] += 40 * cv2.GaussianBlur(rng.normal(0, 1, (60, 60)).astype(np.float32), (0, 0), 2.0)
    g = np.clip(cv2.GaussianBlur(g, (0, 0), 1.0), 0, 255).astype(np.uint8)
    return cv2.merge([g, g, g])


def query_of(ref: np.ndarray, W: np.ndarray) -> np.ndarray:
    """query(x) = ref(W x): W maps query pixels to reference pixels."""
    return cv2.warpPerspective(ref, W, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)


SHIFT = np.array([[1.0, 0, 6.0], [0, 1.0, -4.0], [0, 0, 1]])
HOMOG = np.array([[1.004, 0.003, 5.0], [-0.002, 0.998, 3.0], [4e-6, -3e-6, 1]])


def corner_err(Ha, Hb, box=BOX) -> float:
    c = np.array([[box[0], box[1], 1], [box[2], box[1], 1], [box[2], box[3], 1], [box[0], box[3], 1]], float).T
    a, b = Ha @ c, Hb @ c
    return float(np.abs(a[:2] / a[2] - b[:2] / b[2]).max())


class TestC1(unittest.TestCase):
    def test_recovers_a_known_warp_in_the_query_to_reference_direction(self):
        ref = scene()
        for W in (SHIFT, HOMOG):
            H, info = r32.c1_align(ref, query_of(ref, W), BOX)
            self.assertIsNotNone(H, info)
            self.assertLess(corner_err(H, W), 0.3, info)
            self.assertTrue(info["support_ok"])

    @unittest.skipUnless(PROTO.exists(), "pinned prototype copy not present (untracked run folder)")
    def test_numerics_equal_the_pinned_prototype_copy(self):
        self.assertEqual(hashlib.sha256(PROTO.read_bytes()).hexdigest(), PROTO_SHA)
        spec = importlib.util.spec_from_file_location("align_lib_repaired", PROTO)
        A = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(A)
        ref = scene(1)
        cv2.setNumThreads(1)
        for W in (SHIFT, HOMOG):
            q = query_of(ref, W)
            Hp, ip = A.ecc_global(ref, q, BOX, "homography", "edges", robust=True)
            Wl, info, _, _ = r32._c1_fit(ref, q, BOX)
            Hc = r32._scale(Wl, 2.0).astype(np.float64)
            self.assertTrue(np.array_equal(Hp, Hc / Hc[2, 2]))
            self.assertEqual(ip["blocks_dropped"], info["blocks_dropped"])

    def test_pass_one_is_v03s_fit(self):
        ref = scene(2)
        q = query_of(ref, HOMOG)
        nr = v03.AlignParams(min_cc=-2.0, max_shift_px=1e9, max_scale_dev=1e9)
        Hv, _ = v03.image_align(ref, q, BOX, nr)
        cv2.setNumThreads(1)
        R, Q = r32._prep(ref, 1.5, "edges"), r32._prep(q, 1.5, "edges")
        m = np.zeros(ref.shape[:2], np.uint8); m[24:-24, 24:-24] = 1
        x0, y0, x1, y1 = v03.e021.grow(BOX, 0.5, W_, H_); m[y0:y1, x0:x1] = 0
        R, Q = cv2.pyrDown(R), cv2.pyrDown(Q)
        m = (cv2.resize(m.astype(np.float32), (R.shape[1], R.shape[0]), interpolation=cv2.INTER_AREA) > 0.999).astype(np.uint8)
        _, W1 = r32._ecc_pyr(Q, R, m, 3, np.eye(3, dtype=np.float32))
        H1 = r32._scale(W1, 2.0).astype(np.float64)
        self.assertLess(corner_err(Hv, H1 / H1[2, 2]), 1e-3)

    def test_support_guard(self):
        m = np.ones((90, 90), np.uint8)
        self.assertTrue(r32.support(m, m)["support_ok"])
        five = m.copy(); five[60:, :] = 0; five[30:60, 60:] = 0  # 5 whole cells kept: 56 %, but 5 of 9 cells
        s = r32.support(m, five)
        self.assertEqual((s["support_share"], s["support_cells"]), (0.5556, 5))
        self.assertFalse(s["support_ok"])
        half = m.copy(); half[:, 45:] = 0                       # 50 %; the middle column keeps half: 6 cells
        self.assertEqual(r32.support(m, half)["support_cells"], 6)
        spread = m.copy(); spread[::2] = 0                      # 50 % kept, every cell keeps 50 %
        self.assertTrue(r32.support(m, spread)["support_ok"])
        sparse = m.copy(); sparse[1::3] = 0; sparse[2::3] = 0  # every cell keeps 1/3, 33 % overall
        self.assertFalse(r32.support(m, sparse)["support_ok"])

    def test_final_checks_run_on_the_final_warp_only(self):
        ok = {"support_ok": True, "support_share": 1.0, "support_cells": 9}
        self.assertIsNone(r32.final_checks(np.eye(3), 0.9, BOX, ok))
        self.assertIn("cc", r32.final_checks(np.eye(3), 0.4, BOX, ok))
        self.assertIn("box moves", r32.final_checks(np.array([[1, 0, 150.0], [0, 1, 0], [0, 0, 1]]), 0.9, BOX, ok))
        self.assertIn("local scale", r32.final_checks(np.diag([1.2, 1.2, 1.0]), 0.9, BOX, ok))
        self.assertIn("support", r32.final_checks(np.eye(3), 0.9, BOX, {**ok, "support_ok": False}))

    def test_a_refusal_records_the_unchecked_warp_and_no_fallback(self):
        ref = scene(3)
        rng = np.random.default_rng(9)
        unrelated = np.clip(rng.normal(128, 60, ref.shape), 0, 255).astype(np.uint8)
        H, info = r32.c1_align(ref, unrelated, BOX)
        self.assertIsNone(H)
        self.assertIn("refused", info)


class TestOnlyTheWarpReachesTheEvidence(unittest.TestCase):
    def test_same_warp_same_record_whatever_the_fit_excluded(self):
        ref = with_target(background(0))
        qry = with_target(background(1))
        H = np.array([[1.0, 0, 1.0], [0, 1.0, 0.5], [0, 0, 1]])
        enc = MeanColour()
        base, _ = v03.evidence_for(ref, SMALL_BOX, qry, 0, enc, no_detections, align=lambda r, q: H)
        saved = dict(r32.CANDIDATES)
        try:
            for excluded in (0.0, 0.5, 0.9):     # stand-in candidates that left different shares out of their fit
                r32.CANDIDATES["stub"] = lambda r, q, b, e=excluded: (H, {"method": "stub", "excluded": e})
                rec, diag = r32.evidence_for(ref, SMALL_BOX, qry, 0, enc, no_detections, "stub")
                self.assertEqual(rec, base)
                self.assertEqual(diag["registration"]["excluded"], excluded)
            r32.CANDIDATES["stub"] = lambda r, q, b: (None, {"method": "stub", "refused": "x"})
            ref_rec, _ = v03.evidence_for(ref, SMALL_BOX, qry, 0, enc, no_detections, align=lambda r, q: None)
            rec, _ = r32.evidence_for(ref, SMALL_BOX, qry, 0, enc, no_detections, "stub")
            self.assertEqual(rec, ref_rec)
            self.assertEqual(rec.quality, 0.0)
        finally:
            r32.CANDIDATES.clear()
            r32.CANDIDATES.update(saved)


class TestHarness(unittest.TestCase):
    def test_c0_is_v03_unchanged_and_views_are_timed(self):
        ref = scene(4)
        qry = query_of(ref, SHIFT)
        enc = MeanColour()
        base, bdiag = v03.evidence_for(ref, BOX, qry, 0, enc, no_detections)
        rec, diag = r32.evidence_for(ref, BOX, qry, 0, enc, no_detections, "C0")
        self.assertEqual(rec, base)
        self.assertEqual(diag["registration"], bdiag["registration"])
        t = diag["timing"]
        self.assertGreaterEqual(t["evidence_s"], t["registration_s"])
        self.assertGreater(t["registration_s"], 0.0)

    def test_merge_keeps_the_jobs_order_and_every_key_exactly_once(self):
        jobs = [{"episode_id": e, "stop": s} for e in ("A-visible-L1", "A-removed-L1") for s in range(3)]
        shards = [[dict(j) for j in jobs[i::2]] for i in range(2)]
        merged = r32.merge_shards(jobs, shards)
        self.assertEqual([(d["episode_id"], d["stop"]) for d in merged], [(j["episode_id"], j["stop"]) for j in jobs])
        with self.assertRaisesRegex(ValueError, "twice"):
            r32.merge_shards(jobs, shards + [[dict(jobs[0])]])
        with self.assertRaisesRegex(ValueError, "missing"):
            r32.merge_shards(jobs, shards[:1])
        with self.assertRaisesRegex(ValueError, "repeats"):
            r32.merge_shards(jobs + [jobs[0]], shards)


if __name__ == "__main__":
    unittest.main()
