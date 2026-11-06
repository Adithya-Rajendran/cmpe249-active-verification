"""tools/registration_compare.py: the pinned criteria's arithmetic on a tiny synthetic set (no renders)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import registration_compare as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

BOX = [40, 40, 80, 80]
T5 = [[1, 0, 5.0], [0, 1, 0], [0, 0, 1]]


class TestPieces(unittest.TestCase):
    def test_refusal_boundary(self):
        self.assertTrue(rc.refusals_ok(8, 165))
        self.assertFalse(rc.refusals_ok(9, 165))
        self.assertTrue(rc.refusals_ok(44, 900))
        self.assertFalse(rc.refusals_ok(45, 900))

    def test_corner_error_and_silhouette(self):
        self.assertAlmostEqual(rc.corner_err(np.eye(3), np.array(T5), BOX), 5.0)
        m = np.zeros((120, 120), np.uint8)
        m[40:80, 40:80] = 255
        self.assertAlmostEqual(rc.iou(m, m, np.eye(3)), 1.0)
        self.assertAlmostEqual(rc.iou(m, m, np.array(T5)), 35 * 40 / (45 * 40))


class TestScore(unittest.TestCase):
    """One arrangement, one stop, visible / occluded / removed x L1 / L2 (6 views); truth H = a 5 px shift."""

    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.t)
        s = self.t / "set"
        (s / "TX1" / "masks").mkdir(parents=True)
        m = np.zeros((120, 120), np.uint8)
        m[40:80, 40:80] = 255
        for n in ("target_alone_P1.png", "target_alone_revisit_P1.png"):
            cv2.imwrite(str(s / "TX1" / "masks" / n), m)
        ann = s / "TX1" / "ann_P1_L1.json"
        ann.write_text(json.dumps({"stop": 0, "stop_name": "P1", "targets": {"t": {"box": BOX}}}))
        eps = {}
        jobs = []
        for c, truth in (("visible", "present"), ("occluded", "present"), ("removed", "removed")):
            for L in ("L1", "L2"):
                e = f"TX1-{c}-{L}"
                eps[e] = {"arrangement_id": "TX1", "condition": c, "lighting": L, "truth": truth,
                          "stops": [{"stop": 0, "name": "P1", "visible_fraction": 1.0, "diagnostic": True}]}
                jobs.append({"episode_id": e, "stop": 0, "target": "t", "annotation": str(ann)})
        (s / "view_labels.json").write_text(json.dumps({"episodes": eps}))
        (s / "true_alignment.json").write_text(json.dumps({"environment": "tabletop", "arrangements": {"TX1": {"stops": {
            "P1": {"stop_index": 0, "H_target_plane": T5, "H_table_plane": T5}}}}}))
        (s / "jobs_refL1.json").write_text(json.dumps({"jobs": jobs}))
        self.set, self.jobs = s, jobs

    def diag(self, name, per_view):
        p = self.t / f"{name}.jsonl"
        with p.open("w") as f:
            for j in self.jobs:
                H, refused, match, q, obs = per_view(j["episode_id"])
                f.write(json.dumps({"episode_id": j["episode_id"], "stop": 0, "match_score": match, "quality": q,
                                    "obstruction": obs, "H": H, "registration": {"refused": refused},
                                    "timing": {"registration_s": 1.0, "evidence_s": 2.0}}) + "\n")
        return str(p)

    def score(self, c0, cand):
        a = SimpleNamespace(set=str(self.set), jobs="jobs_refL1.json", c0=c0, cand=cand, name="CX", design=False,
                            c0_timing=None, cand_timing=None, original_c0=None)
        return rc.score(a)

    def test_criteria(self):
        # C0: removed L1 refused (identity, quality 0); every other view on the truth
        def c0(e):
            if e == "TX1-removed-L1":
                return None, "cc", 0.2, 0.0, 0.0
            return T5, None, (0.2 if "removed" in e else 0.9), 0.9, 0.1
        # candidate: recovers removed L1 with a 1 px error (a clear miss), and is 1 px off on visible L2
        T6 = [[1, 0, 6.0], [0, 1, 0], [0, 0, 1]]
        def cand(e):
            if e == "TX1-removed-L1":
                return T6, None, 0.2, 0.9, 0.1
            if e == "TX1-visible-L2":
                return T6, None, 0.9, 0.9, 0.1
            return c0(e)
        o = self.score(self.diag("c0", c0), self.diag("cand", cand))
        L1, L2 = o["per_lighting"]["L1"], o["per_lighting"]["L2"]
        self.assertEqual((L1["c0"]["refused"], L1["cand"]["refused"]), (1, 0))
        self.assertEqual((L1["c0"]["diag_removal_yield"], L1["cand"]["diag_removal_yield"]), (0, 1))
        self.assertEqual([k for k in L1["newly_accepted"]], [["TX1-removed-L1", 0]])
        self.assertAlmostEqual(L1["newly_accepted_err_target"][0][2], 1.0)
        self.assertTrue(all(L1["criteria"].values()))
        # L2: common-accepted max error grows by 1 px (<= 2), median of 3 views stays 0: geometry passes
        self.assertAlmostEqual(L2["geometry_common_target"]["cand"]["max"], 1.0)
        self.assertTrue(L2["criteria"]["5_common_geometry"])
        self.assertTrue(o["pass"])

    def test_a_present_clear_miss_or_lost_yield_fails(self):
        def c0(e):
            return T5, None, (0.2 if "removed" in e else 0.9), 0.9, 0.1
        def cand(e):
            if e == "TX1-occluded-L2":
                return T5, None, 0.2, 0.9, 0.1          # a present-labelled clear miss
            if e == "TX1-removed-L1":
                return None, "support", 0.2, 0.0, 0.0   # newly refused: the diagnostic removal is lost
            return c0(e)
        o = self.score(self.diag("c0b", c0), self.diag("candb", cand))
        self.assertFalse(o["per_lighting"]["L2"]["criteria"]["2_no_present_clear_miss"])
        self.assertFalse(o["per_lighting"]["L1"]["criteria"]["4_diag_yield_ge_c0"])
        self.assertEqual(o["per_lighting"]["L1"]["newly_refused"], [["TX1-removed-L1", 0, "support"]])
        self.assertFalse(o["pass"])

    def test_report_only_with_the_frozen_candidate(self):
        # the test set's view-level reporting: C0 alone, thresholds from frozen/ (sidecar-checked), no criteria
        frozen = REPO / "frozen" / "frozen_candidate.json"
        def c0(e):
            return T5, None, (0.68 if "removed" in e else 0.9), 0.9, 0.38
        p = self.diag("c0r", c0)
        a = SimpleNamespace(set=str(self.set), jobs="jobs_refL1.json", c0=p, cand=p, name="C0", design=False,
                            c0_timing=None, cand_timing=None, original_c0=None, thresholds=str(frozen),
                            report_only=True, data_label="TEST data")
        o = rc.score(a)
        self.assertEqual(o["thresholds_sha256"], rc.sha(frozen))
        self.assertFalse(o["gating"])
        # tau_match 0.7 and tau_occ 0.4: match 0.68 with obstruction 0.38 is a clear miss here (the shelf
        # candidate's tau_match 0.65 would make it a strong match, and its tau_occ 0.35 would fail the gate)
        self.assertEqual(o["per_lighting"]["L1"]["c0"]["diag_removal_yield"], 1)
        self.assertEqual(o["per_lighting"]["L1"]["c0"]["removed_strong_match"], 0)
        self.assertNotIn("pass", o)                                     # no criteria for C0 alone
        self.assertNotIn("criteria", o["per_lighting"]["L1"])
        md = rc.markdown(o)
        self.assertTrue(md.startswith("# C0 views: set"))
        self.assertIn("SIMULATION, TEST data.", md)
        self.assertNotIn("Criteria", md)
        self.assertNotIn("PASS", md)

    def test_a_candidate_failing_its_sidecar_is_refused(self):
        bad = self.t / "frozen_candidate.json"
        shutil.copy(REPO / "frozen" / "frozen_candidate.json", bad)
        (self.t / "frozen_candidate.json.sha256").write_text("0" * 64 + "  frozen_candidate.json\n")
        with self.assertRaises(SystemExit):
            rc.thresholds_rule(str(bad))


if __name__ == "__main__":
    unittest.main()
