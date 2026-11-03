"""Checkpoint 2 tests for verify/run.py on manifests: the SIMULATION label, the family-stratified paired bootstrap,
FRR per arrangement and the geometric no-diagnostic-view share (context, never a ceiling). Hand-built sets only."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis.bootstrap import paired_diff
from tests.test_select_sim import SLOTS, write_sim_set
from verify import run
from verify.schema import GroundTruth, Split

BASE = ["--budgets", "0,3", "--primary-budget", "3", "--cap-steps", "none", "--r-seeds", "2", "--bootstrap-B", "50"]


def run_set(d: Path, out: str, labels: bool = True, **kw) -> tuple[int, Path]:
    args = write_sim_set(d / f"set-{out}", **kw)
    lab = ["--view-labels", args[5]] if labels else []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rc = run.main([*args[:4], *args[6:], *BASE, *lab, "--out", str(d / out)])
    return rc, d / out


class SimRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.rc, cls.out = run_set(d, "sim", split="test")
        cls.res = json.loads((cls.out / "results.json").read_text())
        cls.cfg = json.loads((cls.out / "config.json").read_text())
        cls.table = (cls.out / "table.md").read_text()
        cls.rc2, cls.out2 = run_set(d, "legacy", False, split="test", simulation=False, labels_simulation=False,
                                    family=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_simulation_label_everywhere(self) -> None:
        self.assertEqual(self.rc, 0)
        self.assertEqual(self.res["label"], run.SIM_LABEL)
        self.assertEqual(self.cfg["label"], run.SIM_LABEL)
        self.assertTrue(self.res["simulation"] and self.cfg["config"]["simulation"])
        self.assertTrue(self.table.startswith(f"# {run.SIM_LABEL}"))
        self.assertTrue(self.table.rstrip().endswith(f"_{run.SIM_LABEL}._"))
        for text in (self.table, json.dumps(self.res), json.dumps(self.cfg)):
            self.assertNotIn("REAL PILOT", text)
        legacy = json.loads((self.out2 / "results.json").read_text())
        self.assertEqual((self.rc2, legacy["label"], legacy["config"]["simulation"]), (0, run.REAL_LABEL, False))
        # the flag is in the hashed config
        cfg = dict(self.cfg["config"])
        self.assertEqual(run.config_sha256(cfg), self.cfg["config_sha256"])
        self.assertNotEqual(run.config_sha256({**cfg, "simulation": False}), self.cfg["config_sha256"])

    def test_sim_manifest_needs_positions(self) -> None:
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as err:
            args = write_sim_set(Path(d) / "s", split="test")
            self.assertEqual(run.main([*args[:4], *BASE, "--out", str(Path(d) / "o")]), 2)
        self.assertIn("needs --positions", err.getvalue())

    def test_stratified_bootstrap_matches_a_direct_call(self) -> None:
        st = self.res["bootstrap_stratification"]
        self.assertTrue(st["stratified"])
        self.assertEqual(st["strata"], {"IND28": 1, "M16-L": 1, "M16-R": 1, "S8": 2, "white_box": 1, "ycb_box": 1})
        self.assertTrue(self.res["bootstrap"] and all(b["stratified"] for b in self.res["bootstrap"]))
        self.assertIn("Paired bootstrap stratified by quota slot class", self.table)
        logs = [run.EpisodeLog.from_dict(json.loads(x)) for x in (self.out / "logs.jsonl").read_text().splitlines()]
        truth = GroundTruth.load(self.out / "ground_truth.json")
        groups = run.group_results(logs)
        strata = {a: (f"M16-{s}" if f == "M16" else f) for a, f, s in SLOTS}
        row = next(b for b in self.res["bootstrap"] if (b["policy_a"], b["policy_b"], b["metric"], b["lighting"]) ==
                   ("G", "F", "coverage", "L2"))
        sub = {k: [r for r in v if r.lighting.value == "L2"] for k, v in groups.items()}
        want = paired_diff(sub[("G", 3, None)], sub[("F", 3, None)], truth, "coverage", B=50, seed=0, strata=strata)
        self.assertEqual((row["ci_lo"], row["ci_hi"], row["estimate"]), (want.ci_lo, want.ci_hi, want.estimate))
        legacy = json.loads((self.out2 / "results.json").read_text())
        self.assertFalse(legacy["bootstrap_stratification"]["stratified"])
        self.assertTrue(all(not b["stratified"] for b in legacy["bootstrap"]))
        self.assertIn("Paired bootstrap UNSTRATIFIED", (self.out2 / "table.md").read_text())

    def test_no_diagnostic_share_from_labels_and_fallback(self) -> None:
        # by construction only SD0003 (all five stops blocked) has occluded and removed episodes with no diagnostic stop
        s = next(m for m in self.res["metrics"] if (m["scope"], m["policy"], m["lighting"]) == ("test", "S", "L1"))
        o = s["overall"]
        self.assertEqual((o["n_no_diagnostic_view"], o["n_episodes"]), (2, 3 * len(SLOTS)))
        self.assertIn("view labels", o["no_diagnostic_view_source"])
        self.assertIn("geometric no-diagnostic-view share (test, L1 queries): 0.095 (2 of 21 episodes", self.table)
        self.assertIn("context for coverage, not a ceiling", self.table)
        self.assertNotIn("max achievable", self.table.lower())
        legacy = json.loads((self.out2 / "results.json").read_text())
        lo = next(m for m in legacy["metrics"] if (m["scope"], m["policy"], m["lighting"]) == ("test", "S", "L1"))
        # no labels given for the legacy run: the predeclared flag (IND28 arrangement), labelled as the fallback
        self.assertEqual(lo["overall"]["n_no_diagnostic_view"], 2)   # SD0003 occluded + removed, flagged IND28
        self.assertIn("predeclared indistinguishable flag", lo["overall"]["no_diagnostic_view_source"])

    def test_view_labels_are_read_only_after_every_policy_ran(self) -> None:
        events = []
        real_stage, real_load = run.run_stage, run.load_geometric_labels

        def stage(*a, **kw):
            events.append("run")
            return real_stage(*a, **kw)

        def load(*a, **kw):
            events.append("labels")
            return real_load(*a, **kw)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(run, "run_stage", stage), \
                mock.patch.object(run, "load_geometric_labels", load):
            self.assertEqual(run_set(Path(d), "o", split="test")[0], 0)
        self.assertEqual(events, ["run", "labels"])

    def test_frr_per_arrangement_in_metrics_and_table(self) -> None:
        for m in self.res["metrics"]:
            o = m["overall"]
            self.assertIn("false_removal_rate_arrangement", o)
            self.assertEqual(o["n_arr_with_present"], len({a for a, _, _ in SLOTS}))
        self.assertIn("FRR per arrangement (x/n)", self.table)

    def test_r_frr_per_arrangement_is_per_seed_from_the_logs(self) -> None:
        # By hand from logs.jsonl: per base seed, an arrangement counts 1 if any present episode of the row (its
        # lighting, or both for pooled) was removed under that seed; mean over seeds, summed over arrangements.
        with tempfile.TemporaryDirectory() as d:
            rc, out = run_set(Path(d), "camo", split="test", camouflaged=True)
            logs = [json.loads(x) for x in (out / "logs.jsonl").read_text().splitlines()]
            truth = json.loads((out / "ground_truth.json").read_text())["states"]
            res = json.loads((out / "results.json").read_text())
        self.assertEqual(rc, 0)
        nonzero = []
        for pol in ("R", "R-fcm"):
            for light in ("L1", "L2", "pooled"):
                rows = [g for g in logs if g["policy"] == pol and g["budget_additional_views"] == 3
                        and (light == "pooled" or g["lighting"] == light) and truth[g["episode_id"]] == "present"]
                seeds = sorted({g["seed"] for g in rows})
                want = 0.0
                for a in sorted({g["arrangement_id"] for g in rows}):
                    want += sum(any(g["decision"] == "removed" for g in rows
                                    if g["arrangement_id"] == a and g["seed"] == sd) for sd in seeds) / len(seeds)
                m = next(m for m in res["metrics"] if (m["scope"], m["policy"], m["budget"], m["lighting"]) ==
                         ("test", pol, 3, light))["overall"]
                nonzero.append(want)
                with self.subTest(policy=pol, lighting=light):
                    self.assertAlmostEqual(m["arrangement_false_removal"], want, places=12)
                    self.assertFalse(m["arrangement_frr_binomial"])
        self.assertTrue(any(0 < w < 7 for w in nonzero), nonzero)   # not vacuous: R does falsely remove here
        # deterministic policies keep 0/1 counts
        g = next(m for m in res["metrics"] if (m["scope"], m["policy"], m["budget"], m["lighting"]) ==
                 ("test", "G", 3, "L1"))["overall"]
        self.assertTrue(g["arrangement_frr_binomial"])
        self.assertIsInstance(g["arrangement_false_removal"], int)

    def test_dev_only_manifest_is_scored_on_dev(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rc, out = run_set(Path(d), "dev")
            table = (out / "table.md").read_text()
        self.assertEqual(rc, 0)
        self.assertIn("scored on the **dev** split (the manifest has no test arrangement)", table)
        self.assertIn("## L1 queries - Primary table", table)
        self.assertIn("the bootstrap runs on the test split only", table)
        self.assertEqual(run.table_scope([{"scope": Split.DEV.value}]), "dev")


if __name__ == "__main__":
    unittest.main()
