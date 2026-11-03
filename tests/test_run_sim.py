"""Checkpoint 2 tests for verify/run.py on manifests: the SIMULATION label, the family-stratified paired bootstrap,
FRR per arrangement and the geometric no-diagnostic-view share (context, never a ceiling). Hand-built sets only."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis.bootstrap import paired_diff, rate_intervals
from tests.test_select_sim import SLOTS, write_sim_set
from verify import run
from verify.schema import GroundTruth, Split

BASE = ["--budgets", "0,3", "--primary-budget", "3", "--cap-steps", "none", "--r-seeds", "2", "--bootstrap-B", "50"]
# every quota slot class twice: no singleton stratum, so the bootstrap stays stratified
SLOTS2 = (*SLOTS, ("SD0007", "M16", "L"), ("SD0008", "M16", "R"), ("SD0009", "IND28", None),
          ("SD0010", "white_box", None), ("SD0011", "ycb_box", None))


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
        cls.rc3, cls.out3 = run_set(d, "strat", split="test", slots=SLOTS2)

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
        self.assertEqual(self.rc3, 0)
        res = json.loads((self.out3 / "results.json").read_text())
        table = (self.out3 / "table.md").read_text()
        st = res["bootstrap_stratification"]
        self.assertTrue(st["stratified"] and st["ran"])
        self.assertEqual(st["strata"], {"IND28": 2, "M16-L": 2, "M16-R": 2, "S8": 2, "white_box": 2, "ycb_box": 2})
        self.assertTrue(json.loads((self.out3 / "config.json").read_text())["config"]["bootstrap_stratified"])
        self.assertTrue(res["bootstrap"] and all(b["stratified"] for b in res["bootstrap"]))
        self.assertIn("Paired bootstrap stratified by quota slot class", table)
        logs = [run.EpisodeLog.from_dict(json.loads(x)) for x in (self.out3 / "logs.jsonl").read_text().splitlines()]
        truth = GroundTruth.load(self.out3 / "ground_truth.json")
        groups = run.group_results(logs)
        strata = {a: (f"M16-{s}" if f == "M16" else f) for a, f, s in SLOTS2}
        row = next(b for b in res["bootstrap"] if (b["policy_a"], b["policy_b"], b["metric"], b["lighting"]) ==
                   ("G", "F", "coverage", "L2"))
        sub = {k: [r for r in v if r.lighting.value == "L2"] for k, v in groups.items()}
        want = paired_diff(sub[("G", 3, None)], sub[("F", 3, None)], truth, "coverage", B=50, seed=0, strata=strata)
        self.assertTrue(want.stratified)
        self.assertEqual((row["ci_lo"], row["ci_hi"], row["estimate"]), (want.ci_lo, want.ci_hi, want.estimate))
        # the single rates use the same stratified bootstrap
        m = next(m for m in res["metrics"] if (m["scope"], m["policy"], m["budget"], m["lighting"]) ==
                 ("test", "G", 3, "L2"))
        iv = rate_intervals(sub[("G", 3, None)], truth, B=50, seed=0, strata=strata)["coverage"]
        self.assertTrue(m["rate_ci95"]["stratified"])
        self.assertEqual((m["rate_ci95"]["coverage"]["ci_lo"], m["rate_ci95"]["coverage"]["ci_hi"]),
                         (iv.ci_lo, iv.ci_hi))

    def test_singleton_strata_fall_back_and_are_flagged(self) -> None:
        # Review: on SLOTS (five of six slot classes with one arrangement) the stratified CI was zero-width for any
        # contrast driven by a singleton; it now falls back to unstratified, flagged in results, rows and table
        st = self.res["bootstrap_stratification"]
        self.assertFalse(st["stratified"])
        self.assertEqual(st["strata"], {"IND28": 1, "M16-L": 1, "M16-R": 1, "S8": 2, "white_box": 1, "ycb_box": 1})
        self.assertEqual(st["singleton_strata"], ["IND28", "M16-L", "M16-R", "white_box", "ycb_box"])
        self.assertTrue(st["note"].startswith("unstratified: singleton strata IND28"))
        self.assertFalse(self.cfg["config"]["bootstrap_stratified"])
        self.assertTrue(self.res["bootstrap"] and all(not b["stratified"] and b["singleton_strata"]
                                                      for b in self.res["bootstrap"]))
        self.assertIn("**Paired bootstrap UNSTRATIFIED** (IND28 1, M16-L 1, M16-R 1, S8 2, white_box 1, ycb_box 1 test "
                      "arrangements): unstratified: singleton strata", self.table)
        self.assertIn("| unstratified: singleton strata IND28, M16-L, M16-R, white_box, ycb_box |", self.table)
        rows = [m for m in self.res["metrics"] if m.get("rate_ci95")]
        self.assertTrue(rows and all(not m["rate_ci95"]["stratified"] for m in rows))
        logs = [run.EpisodeLog.from_dict(json.loads(x)) for x in (self.out / "logs.jsonl").read_text().splitlines()]
        truth = GroundTruth.load(self.out / "ground_truth.json")
        sub = {k: [r for r in v if r.lighting.value == "L2"] for k, v in run.group_results(logs).items()}
        row = next(b for b in self.res["bootstrap"] if (b["policy_a"], b["policy_b"], b["metric"], b["lighting"]) ==
                   ("G", "F", "coverage", "L2"))
        want = paired_diff(sub[("G", 3, None)], sub[("F", 3, None)], truth, "coverage", B=50, seed=0)   # unstratified
        self.assertEqual((row["ci_lo"], row["ci_hi"]), (want.ci_lo, want.ci_hi))
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

    def test_no_diagnostic_wording_follows_its_source(self) -> None:
        # Review: without --view-labels the line said "episodes have no geometrically diagnostic stop" under a
        # "geometric" heading while the count is the predeclared flag; the per-condition column named no source
        self.assertIn("| geometric no-diagnostic-view share (context; view labels) |", self.table)
        legacy = (self.out2 / "table.md").read_text()
        self.assertIn("predeclared indistinguishable share (no view labels given) (test, L1 queries): 0.095 (2 of 21 "
                      "episodes are occluded or removed episodes of arrangements flagged indistinguishable", legacy)
        self.assertIn("| predeclared indistinguishable share (context; no view labels given) |", legacy)
        self.assertNotIn("geometrically diagnostic stop", legacy)
        self.assertNotIn("geometric no-diagnostic-view share (test", legacy)

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

    def test_view_labels_edited_during_the_run_are_refused(self) -> None:
        # verify.run hashes --view-labels into the config, reads it only after the run and re-hashes it: a file edited
        # in between must not be scored (exit 2, no results.json / table.md).
        real_stage = run.run_stage
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            args = write_sim_set(d / "set", split="test")
            labels, out = Path(args[5]), d / "out"

            def stage(*a, **kw):
                logs = real_stage(*a, **kw)
                labels.write_text(labels.read_text() + "\n")   # still valid JSON, different bytes
                return logs
            with mock.patch.object(run, "run_stage", stage), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                rc = run.main([*args, *BASE, "--out", str(out)])
            self.assertEqual(rc, 2)
            self.assertIn("changed during the run", err.getvalue())
            self.assertFalse((out / "results.json").exists() or (out / "table.md").exists())

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
            res = json.loads((out / "results.json").read_text())
            cfg = json.loads((out / "config.json").read_text())["config"]
        self.assertEqual(rc, 0)
        self.assertIn("scored on the **dev** split (the manifest has no test arrangement)", table)
        self.assertIn("## L1 queries - Primary table", table)
        self.assertEqual(run.table_scope([{"scope": Split.DEV.value}]), "dev")
        # Review: it said "Paired bootstrap stratified by quota slot class ( test arrangements)" and results.json
        # {stratified: true, strata: {}} although no bootstrap ran. It now says plainly that none ran.
        self.assertEqual(res["bootstrap"], [])
        st = res["bootstrap_stratification"]
        self.assertEqual((st["ran"], st["stratified"], st["strata"]), (False, False, {}))
        self.assertIn("no bootstrap ran", st["note"])
        self.assertFalse(cfg["bootstrap_stratified"])
        self.assertIn("- **No bootstrap ran** (paired or single-rate): this run has no test arrangement", table)
        self.assertEqual(table.count("**No bootstrap ran:** this run has no test arrangement"), 2)
        self.assertNotIn("test arrangements)", table)
        self.assertNotIn("stratified by quota slot class", table)
        self.assertNotIn("| setting | A - B |", table)
        self.assertFalse([m for m in res["metrics"] if "rate_ci95" in m])   # no single-rate interval on dev either

    def test_single_rate_intervals_in_results_and_table(self) -> None:
        # prereg section 8: percentile 95% intervals for single rates too, on test, per lighting (not pooled)
        for m in self.res["metrics"]:
            self.assertEqual("rate_ci95" in m, m["scope"] == "test" and m["lighting"] != "pooled", m["lighting"])
        m = next(m for m in self.res["metrics"] if (m["scope"], m["policy"], m["budget"], m["lighting"]) ==
                 ("test", "G", 3, "L1"))
        ci, o = m["rate_ci95"], m["overall"]
        for metric in ("false_removal_rate", "removal_recall", "coverage"):
            self.assertEqual(ci[metric]["estimate"], o[metric])
        self.assertEqual((ci["B"], ci["seed"], ci["n_arrangements"], ci["n_episodes"]), (50, 0, len(SLOTS), 21))
        cell = (f"| G | {run.fmt(o['false_removal_rate'])} [{run.fmt(ci['false_removal_rate']['ci_lo'])}, "
                f"{run.fmt(ci['false_removal_rate']['ci_hi'])}] |")
        self.assertIn(cell, self.table)
        self.assertIn("**Single-rate intervals.**", self.table)
        self.assertIn("seeds are not", self.table)

    def test_failed_run_leaves_no_stale_outputs(self) -> None:
        # Review: main_real removed only ground_truth.json and table.md up front, so a run that failed at the
        # post-run labels check (exit 2) left the previous run's results.json next to its new config.json.
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            args = write_sim_set(d / "set", split="test")
            out = d / "out"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run.main([*args, *BASE, "--out", str(out)]), 0)
                self.assertTrue((out / "results.json").is_file())
                lab = json.loads(Path(args[5]).read_text())
                lab["episodes"].pop(next(iter(lab["episodes"])))
                bad = d / "labels_missing.json"
                bad.write_text(json.dumps(lab))
                rc = run.main([*args[:4], *args[6:], *BASE, "--seed", "1", "--view-labels", str(bad),
                               "--out", str(out)])
            self.assertEqual(rc, 2)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["config.json", "ground_truth.json", "logs.jsonl"])
            self.assertEqual(json.loads((out / "config.json").read_text())["config"]["seed"], 1)
            # an input that is one of the outputs is refused before anything is removed
            shutil.copyfile(args[1], out / "results.json")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                rc = run.main([*args[:1], str(out / "results.json"), *args[2:], *BASE, "--out", str(out)])
            self.assertEqual(rc, 2)
            self.assertIn("is an output of this run", err.getvalue())
            self.assertTrue((out / "results.json").is_file() and (out / "config.json").is_file())

    def test_synthetic_failure_leaves_no_stale_results(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "syn"
            out.mkdir()
            for name in run.RUN_OUTPUTS:
                (out / name).write_text("stale")

            def boom(*a, **kw):
                raise RuntimeError("run failed")
            with mock.patch.object(run, "run_stage", boom), mock.patch("builtins.print"), \
                    self.assertRaises(RuntimeError):
                run.main(["--synthetic", "--n-arr", "4", "--r-seeds", "1", "--bootstrap-B", "10", "--out", str(out)])
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["config.json"])
            self.assertNotEqual((out / "config.json").read_text(), "stale")

    def test_simulation_flags_must_agree_and_logs_and_truth_are_labelled(self) -> None:
        # Review: verify.run never compared the labels file's "simulation" flag with the manifest's (verify.select
        # does), and logs.jsonl / ground_truth.json of a simulation run carried no SIMULATION label
        with tempfile.TemporaryDirectory() as d:
            args = write_sim_set(Path(d) / "set", split="test", labels_simulation=False)
            out = Path(d) / "out"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                rc = run.main([*args, *BASE, "--out", str(out)])
            self.assertEqual(rc, 2)
            self.assertIn("the manifest says simulation=True but", err.getvalue())
            self.assertFalse((out / "results.json").exists())
        logs = [json.loads(x) for x in (self.out / "logs.jsonl").read_text().splitlines()]
        self.assertTrue(logs and all(g["label"] == run.SIM_LABEL and g["simulation"] is True for g in logs))
        gt = json.loads((self.out / "ground_truth.json").read_text())
        self.assertEqual((gt["label"], gt["simulation"], gt["synthetic"]), (run.SIM_LABEL, True, False))
        self.assertEqual(GroundTruth.load(self.out / "ground_truth.json").states,
                         GroundTruth.load(self.out2 / "ground_truth.json").states)
        legacy_logs = [json.loads(x) for x in (self.out2 / "logs.jsonl").read_text().splitlines()]
        self.assertFalse([g for g in legacy_logs if "label" in g or "simulation" in g])   # a real pilot: no label
        self.assertEqual(json.loads((self.out2 / "ground_truth.json").read_text())["label"], "")


if __name__ == "__main__":
    unittest.main()
