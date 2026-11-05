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


class TestC1M(unittest.TestCase):
    def test_phase_start_sign_and_size(self):
        ref = scene(5)
        for t in ((40.0, -24.0), (-32.0, 16.0)):
            W = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1.0]])
            s1 = r32.phase_start(ref, query_of(ref, W), BOX)
            self.assertIsNotNone(s1)
            self.assertLess(float(np.hypot(*(s1[:2, 2] - np.array(t)))), 16.0, s1)   # one level-4 pixel
        far = np.array([[1, 0, 130.0], [0, 1, 0], [0, 0, 1.0]])
        self.assertIsNone(r32.phase_start(ref, query_of(ref, far), BOX))

    def test_ranking_rule(self):
        comp = {"comparable": True, "scores": {"S0": 0.80, "S1": 0.805}}
        self.assertIsNone(r32.choose({"S0": False, "S1": False}, None))
        self.assertEqual(r32.choose({"S0": False, "S1": True}, None), "S1")
        self.assertEqual(r32.choose({"S0": True}, None), "S0")
        self.assertEqual(r32.choose({"S0": True, "S1": True}, comp), "S0")              # within the 0.01 tie
        self.assertEqual(r32.choose({"S0": True, "S1": True}, {**comp, "scores": {"S0": 0.80, "S1": 0.82}}), "S1")
        self.assertEqual(r32.choose({"S0": True, "S1": True},
                                    {"comparable": False, "scores": {"S0": 0.1, "S1": 0.9}}), "S0")

    def test_validation_prefers_the_true_warp_on_common_support(self):
        ref = scene(6)
        q = query_of(ref, HOMOG)
        R1, Q1 = cv2.pyrDown(r32._prep(ref, 1.5, "edges")), cv2.pyrDown(r32._prep(q, 1.5, "edges"))
        m1 = np.zeros(R1.shape, np.uint8); m1[12:-12, 12:-12] = 1
        good = r32._scale(HOMOG, 0.5).astype(np.float64)
        bad = r32._scale(np.array([[1, 0, 14.0], [0, 1, 9.0], [0, 0, 1]]) @ HOMOG, 0.5).astype(np.float64)
        v = r32.validation_scores(R1, Q1, m1, {"S0": bad, "S1": good})
        self.assertTrue(v["comparable"])
        self.assertGreater(v["scores"]["S1"], v["scores"]["S0"] + 0.1)

    def test_c1m_on_a_known_warp(self):
        ref = scene(7)
        H, info = r32.c1m_align(ref, query_of(ref, HOMOG), BOX)
        self.assertIsNotNone(H, info)
        self.assertLess(corner_err(H, HOMOG), 0.3)
        self.assertIn(info["winner"], ("S0", "S1"))


class TestC2(unittest.TestCase):
    def test_weighted_ecc_with_unit_weights_matches_cv2(self):
        """The plan's pre-scoring validation: <= 0.05 px at the box corners against cv2.findTransformECC."""
        cv2.setNumThreads(1)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        for seed, W in ((0, SHIFT), (1, HOMOG), (2, HOMOG)):
            ref = scene(seed)
            R, Q = r32._prep(ref, 1.5, "edges"), r32._prep(query_of(ref, W), 1.5, "edges")
            m = np.zeros(R.shape, np.uint8); m[24:-24, 24:-24] = 1
            _, Wc = cv2.findTransformECC(Q, R, np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY, crit, m, 1)
            cc, Wn = r32.weighted_ecc(Q, R, m.astype(np.float32), np.eye(3))
            self.assertLessEqual(corner_err(Wc.astype(np.float64), Wn), 0.05)
            self.assertGreater(cc, 0.99)

    def test_block_weight_formula(self):
        rng = np.random.default_rng(0)
        R1 = cv2.GaussianBlur(rng.normal(0, 1, (96, 96)).astype(np.float32), (0, 0), 1.0)
        Q1 = R1.copy()
        Q1[0:16, 0:16] = rng.normal(0, 1, (16, 16)).astype(np.float32)       # one changed block
        m1 = np.ones(R1.shape, np.uint8)
        wmap, info = r32.block_weights(R1, Q1, m1, np.eye(3))
        self.assertEqual(info["blocks"], 36)
        self.assertEqual(info["residual_scale"], 0.02)                       # MAD 0: the floor
        self.assertAlmostEqual(float(wmap[40, 40]), 1.0, places=6)           # unchanged: residual 0 = the median
        r = 1.0 - r32._wzncc(R1[:16, :16].astype(np.float64), Q1[:16, :16].astype(np.float64), np.ones((16, 16)))
        self.assertAlmostEqual(float(wmap[5, 5]), 1.0 / (1.0 + (r / (2.385 * 0.02)) ** 2), places=5)
        self.assertLess(float(wmap[5, 5]), 0.01)

    def test_weighted_support(self):
        m = np.ones((90, 90), np.uint8)
        self.assertTrue(r32.weighted_support(m, m.astype(np.float32))["support_ok"])
        half = np.full((90, 90), 0.5, np.float32)                            # uniform halves keep N_eff = all
        s = r32.weighted_support(m, half)
        self.assertEqual(s["support_share"], 1.0)
        self.assertTrue(s["support_ok"])
        few = np.zeros((90, 90), np.float32); few[:30, :] = 1.0             # one row of cells: 3 cells, 33 %
        self.assertFalse(r32.weighted_support(m, few)["support_ok"])

    def test_c2_recovers_a_known_warp_and_downweights_an_occluder(self):
        ref = scene(8)
        q = query_of(ref, HOMOG)
        rng = np.random.default_rng(3)
        for rect in ((40, 100, 480, 560), (30, 130, 440, 600)):                # warm the generator as when tuned
            rng.normal(0, 1, (rect[1] - rect[0], rect[3] - rect[2]))
        tex = 110 + 40 * cv2.GaussianBlur(rng.normal(0, 1, (130, 160)).astype(np.float32), (0, 0), 2.0)
        q[200:330, 40:200] = np.clip(tex, 0, 255).astype(np.uint8)[..., None]   # a textured object in front
        H0, _ = r32.c0_align(ref, q, BOX)
        H, info = r32.c2_align(ref, q, BOX)
        self.assertIsNotNone(H, info)
        self.assertEqual(len(info["rounds"]), 3)
        self.assertLess(corner_err(H, HOMOG), 0.1)
        self.assertLess(corner_err(H, HOMOG), corner_err(H0, HOMOG) / 4)
        self.assertGreater(info["cc"], 0.95)


class TestC2M(unittest.TestCase):
    def test_c2m_on_a_known_warp(self):
        ref = scene(9)
        H, info = r32.c2m_align(ref, query_of(ref, HOMOG), BOX)
        self.assertIsNotNone(H, info)
        self.assertLess(corner_err(H, HOMOG), 0.3)
        self.assertEqual(info["method"], "C2+M")
        self.assertIn(info["winner"], ("S0", "S1"))


class TestHarness(unittest.TestCase):
    def test_c0_is_v03_unchanged_and_views_are_timed(self):
        ref = scene(4)
        qry = query_of(ref, SHIFT)
        enc = MeanColour()
        base, bdiag = v03.evidence_for(ref, BOX, qry, 0, enc, no_detections)
        rec, diag = r32.evidence_for(ref, BOX, qry, 0, enc, no_detections, "C0")
        self.assertEqual(rec, base)
        self.assertEqual(diag["registration"], bdiag["registration"])
        H, _ = v03.image_align(ref, qry, BOX)
        self.assertEqual(diag["H"], np.round(H, 10).tolist())
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
