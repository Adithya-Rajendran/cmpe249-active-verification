"""verify.run's frozen path (real-v5): thresholds only from an eligible frozen candidate, side descriptors with Dir and
the frozen rankers, the primary contrast rows and the predeclared common-target analysis (analysis/common_target.py).
Hand-built sim sets with invented scores; no number here is a result."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.bootstrap import paired_diff
from analysis.common_target import common_target_diff
from tests.test_bootstrap import CONDITIONS, LIGHTINGS, P, R, U, res, truth_for
from tests.test_ranker_side import write_side
from tests.test_select_sim import write_sim_set
from verify import ranker as rk
from verify import run
from verify import select as sel
from verify.policies import NOCUE_FEATURES, RANKER_FEATURES
from verify.schema import Thresholds

SELECT_QUICK = ["--cap-steps", "2", "--r-seeds", "1", "--k-sensitivity", "none", "--lam-sensitivity", "none",
                "--bootstrap-B", "20", "--tau-q-grid", "0.5,0.6", "--tau-occ-grid", "0.3", "--tau-match-grid", "0.5,0.6"]
RUN_QUICK = ["--budgets", "1,3", "--cap-steps", "2", "--r-seeds", "2", "--bootstrap-B", "50", "--primary-budget", "3"]
DEV_SLOTS = (("SD0000", "S8", None), ("SD0001", "M16", "L"), ("SD0002", "M16", "R"), ("SD0003", "IND28", None),
             ("SD0004", "white_box", None), ("SD0005", "ycb_box", None), ("SD0006", "S8", None))
TEST_SLOTS = tuple((f"ST{i:04d}", f, s) for i, (f, s) in enumerate(
    [("S8", None), ("S8", None), ("M16", "L"), ("M16", "L"), ("M16", "R"), ("M16", "R"), ("white_box", None),
     ("white_box", None), ("ycb_box", None), ("ycb_box", None)]))
TARGETS = ("foam_brick", "tuna_fish_can")


def retarget(d: Path) -> None:
    """Alternate the two common targets over the arrangements (write_sim_set gives every arrangement a mug)."""
    m = json.loads((d / "manifest.json").read_text())
    for i, a in enumerate(m["arrangements"]):
        a["target_id"] = TARGETS[i % 2]
    (d / "manifest.json").write_text(json.dumps(m, indent=2))


def quiet():
    st = contextlib.ExitStack()
    st.enter_context(contextlib.redirect_stdout(io.StringIO()))
    st.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return st


class FrozenRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        dev_args = write_sim_set(d / "dev", DEV_SLOTS)
        dev_side = write_side(d / "dev", DEV_SLOTS)
        with quiet():
            cls.select_rc = sel.main([*dev_args, "--side", str(dev_side), *SELECT_QUICK, "--out", str(d / "cand")])
        cls.cand = d / "cand" / "frozen_candidate.json"
        doc = sel.require_freeze_eligible(cls.cand)
        cls.th = Thresholds(**{k: doc["thresholds"][k] for k in ("tau_match", "tau_q", "tau_occ", "k", "lam",
                                                                  "travel_norm_cm")})
        dev = sel.sim_dev_data(*[d / "dev" / f for f in ("manifest.json", "evidence.jsonl", "view_labels.json",
                                                          "positions.json")], dev_side)
        cls.models = {}
        for name, feats in (("A", RANKER_FEATURES), ("A-nocue", NOCUE_FEATURES)):
            fitted = rk.fit_with_selection(rk.training_rows(dev, cls.th, feats))
            cls.models[name] = d / f"model_{name}.json"
            rk.save_model(cls.models[name], name, fitted, cls.th, {"note": "test"})
        cls.test_args = write_sim_set(d / "test", TEST_SLOTS, split="test")
        retarget(d / "test")
        cls.test_side = write_side(d / "test", TEST_SLOTS)
        cls.d = d
        cls.out = d / "out"
        with quiet():
            cls.rc = run.main([*cls.test_args, *RUN_QUICK, "--candidate", str(cls.cand), "--side", str(cls.test_side),
                               "--model", f"A={cls.models['A']}", "--model", f"A-nocue={cls.models['A-nocue']}",
                               "--out", str(cls.out)])
        cls.res = json.loads((cls.out / "results.json").read_text()) if cls.rc == 0 else None

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def main(self, *extra: str, args=None) -> tuple[int, str]:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            rc = run.main([*(args or self.test_args), *RUN_QUICK, *extra, "--out", str(self.d / "o2")])
        return rc, err.getvalue()

    def test_thresholds_come_from_the_candidate(self) -> None:
        self.assertEqual(self.select_rc, 0)
        self.assertEqual(self.rc, 0)
        cfg = self.res["config"]
        self.assertEqual(cfg["thresholds_source"], "frozen candidate")
        self.assertEqual(cfg["thresholds"], self.th.to_dict())
        self.assertEqual(self.res["label"], run.SIM_FROZEN_LABEL)
        self.assertEqual(set(cfg["ranker_models"]), {"A", "A-nocue"})

    def test_a_test_manifest_needs_a_candidate(self) -> None:
        rc, err = self.main()
        self.assertEqual(rc, 2)
        self.assertIn("needs --candidate", err)

    def test_a_tampered_candidate_is_refused(self) -> None:
        bad = self.d / "bad_cand"
        shutil.copytree(self.cand.parent, bad, dirs_exist_ok=True)
        doc = json.loads((bad / "frozen_candidate.json").read_text())
        doc["thresholds"]["tau_q"] = 0.0
        (bad / "frozen_candidate.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        rc, err = self.main("--candidate", str(bad / "frozen_candidate.json"))
        self.assertEqual(rc, 2)
        self.assertIn("--candidate refused", err)

    def test_a_model_trained_on_other_thresholds_is_refused(self) -> None:
        doc = json.loads(self.models["A"].read_text())
        doc["thresholds"]["tau_occ"] = 0.99
        other = self.d / "model_other.json"
        other.write_text(json.dumps(doc))
        rc, err = self.main("--candidate", str(self.cand), "--side", str(self.test_side), "--model", f"A={other}")
        self.assertEqual(rc, 2)
        self.assertIn("trained with thresholds", err)
        rc, _ = self.main("--candidate", str(self.cand), "--model", f"A={self.models['A']}")   # no --side
        self.assertEqual(rc, 2)

    def test_every_policy_ran(self) -> None:
        pols = {m["policy"] for m in self.res["metrics"]}
        self.assertTrue({"A", "A-nocue", "Dir", "A-fcm", "G", "G-lookahead"} <= pols, pols)

    def test_primary_rows_and_their_reading(self) -> None:
        prim = self.res["primary"]
        self.assertEqual(sorted((r["lighting"], r["metric"]) for r in prim),
                         sorted((l, m) for l in ("L1", "L2") for m in run.BOOTSTRAP_METRICS))
        for r in prim:
            self.assertEqual((r["policy_a"], r["policy_b"], r["travel_cap_cm"], r["budget"]), ("A", "G", None, 3))
            if r["ci_lo"] is not None:
                self.assertEqual(r["inconclusive"], r["ci_lo"] <= 0.0 <= r["ci_hi"])
        table = (self.out / "table.md").read_text()
        self.assertIn("PRIMARY CONTRAST: A - G", table)
        self.assertIn("never reported as equivalence", table)

    def test_a_minus_g_lookahead_at_every_cap(self) -> None:
        caps = {r["travel_cap_cm"] for r in self.res["bootstrap"]
                if (r["policy_a"], r["policy_b"]) == ("A", "G-lookahead")}
        self.assertEqual(caps, set(self.res["config"]["travel_caps_cm"]))
        self.assertTrue([r for r in self.res["bootstrap"] if (r["policy_a"], r["policy_b"]) == ("A", "Dir")])

    def test_common_target_rows(self) -> None:
        ct = self.res["common_target"]
        self.assertEqual(sorted((r["lighting"], r["metric"]) for r in ct),
                         sorted((l, m) for l in ("L1", "L2") for m in run.BOOTSTRAP_METRICS))
        for r in ct:
            self.assertEqual(r["targets"], list(TARGETS))
            self.assertEqual(r["n_arrangements"], len(TEST_SLOTS))


def results_for(arrs, decide, policy):
    return [res(a, c, l, decide(a, c, l), policy) for a in arrs for c in CONDITIONS for l in LIGHTINGS[:1]]


class CommonTargetTest(unittest.TestCase):
    def test_balanced_cells_reduce_to_the_plain_paired_estimate(self) -> None:
        arrs = [f"A{i}" for i in range(8)]
        fam = {a: ("x" if i < 4 else "y") for i, a in enumerate(arrs)}
        tgt = {a: TARGETS[i % 2] for i, a in enumerate(arrs)}
        a = results_for(arrs, lambda x, c, l: R if c.value == "removed" and x in ("A0", "A5") else P, "A")
        b = results_for(arrs, lambda x, c, l: U if c.value == "removed" else P, "G")
        truth = truth_for(a)
        got = common_target_diff(a, b, truth, fam, tgt, "removal_recall", B=200, seed=0)
        want = paired_diff(a, b, truth, "removal_recall", B=200, seed=0)
        self.assertAlmostEqual(got["estimate"], want.estimate)
        self.assertEqual(got["single_target_families"], [])

    def test_unbalanced_targets_get_equal_weight_within_a_family(self) -> None:
        # family x: three foam-brick arrangements (A removes none) and one tuna can (A removes it): equal target weight
        # gives A's removal recall (0 + 1) / 2 = 0.5 there, where the unweighted share would be 1 / 4
        arrs = ["A0", "A1", "A2", "A3"]
        fam = {x: "x" for x in arrs}
        tgt = {"A0": TARGETS[0], "A1": TARGETS[0], "A2": TARGETS[0], "A3": TARGETS[1]}
        a = results_for(arrs, lambda x, c, l: R if c.value == "removed" and x == "A3" else P, "A")
        b = results_for(arrs, lambda x, c, l: U if c.value == "removed" else P, "G")
        got = common_target_diff(a, b, truth_for(a), fam, tgt, "removal_recall", B=100, seed=0)
        self.assertAlmostEqual(got["estimate"], 0.5)
        self.assertEqual(got["arrangements_per_family_target"], {"x|foam_brick": 3, "x|tuna_fish_can": 1})

    def test_other_targets_are_left_out_and_single_target_families_flagged(self) -> None:
        arrs = ["A0", "A1", "A2"]
        fam = {"A0": "x", "A1": "x", "A2": "y"}
        tgt = {"A0": TARGETS[0], "A1": "mug", "A2": TARGETS[1]}
        a = results_for(arrs, lambda x, c, l: P, "A")
        got = common_target_diff(a, a, truth_for(a), fam, tgt, "coverage", B=50, seed=0)
        self.assertEqual(got["n_arrangements"], 2)
        self.assertEqual(got["single_target_families"], ["x", "y"])
        self.assertEqual((got["estimate"], got["ci_lo"], got["ci_hi"]), (0.0, 0.0, 0.0))

    def test_draws_follow_the_strata(self) -> None:
        arrs = [f"A{i}" for i in range(6)]
        fam = {a: "x" for a in arrs}
        tgt = {a: TARGETS[i % 2] for i, a in enumerate(arrs)}
        rng = np.random.default_rng(3)
        flips = {a: rng.uniform() < 0.5 for a in arrs}
        a = results_for(arrs, lambda x, c, l: R if c.value == "removed" and flips[x] else P, "A")
        b = results_for(arrs, lambda x, c, l: U if c.value == "removed" else P, "G")
        s1 = common_target_diff(a, b, truth_for(a), fam, tgt, "removal_recall", B=300, seed=1,
                                strata={x: ("s1" if i < 3 else "s2") for i, x in enumerate(arrs)})
        self.assertTrue(s1["stratified"])
        self.assertLessEqual(s1["ci_lo"], s1["estimate"])
        self.assertLessEqual(s1["estimate"], s1["ci_hi"])


if __name__ == "__main__":
    unittest.main()
