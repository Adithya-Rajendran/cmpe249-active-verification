"""End-to-end: verify.run on a tiny synthetic set (README section 8)."""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis.bootstrap import paired_diff
from verify import policies, run
from verify.schema import (
    OBSERVATION_FIELDS,
    SYNTHETIC_LABEL,
    EpisodeSpec,
    EvidenceStore,
    GroundTruth,
    Observation,
    Positions,
    config_sha256,
)
from verify.stopping import StoppingRule

TINY = ["--synthetic", "--n-arr", "6", "--seed", "3", "--r-seeds", "3", "--bootstrap-B", "200",
        "--budgets", "0..4", "--cap-steps", "2"]
# verify.run reads configs/positions.json by default, so the expected cap comes from that file
# (measured shelf arc 2026-10-09: 2 x 35.34 = 70.69 cm; it was 47.12 cm on the design template).
CAP2 = 2 * Positions.load().nominal_step_cm()
REQUIRED_LOG_FIELDS = {"episode_id", "arrangement_id", "split", "target_id", "condition", "lighting",
                       "model_id", "config_version", "config_sha256", "policy", "budget_additional_views",
                       "travel_cap_cm", "seed", "requested_stops", "evidence", "decision", "travel_cm",
                       "n_views", "n_additional_views", "synthetic", "label"}
FORBIDDEN = (GroundTruth, EpisodeSpec, EvidenceStore)


def _keys(obj) -> set[str]:
    """All dict keys, recursively."""
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in _keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in _keys(v)}
    return set()


def _run(tmp: str, name: str, extra: list[str] | None = None) -> Path:
    out = Path(tmp) / name
    with mock.patch("builtins.print"):
        rc = run.main([*TINY, *(extra or []), "--out", str(out)])
    assert rc == 0
    return out


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = _run(cls.tmp.name, "a")
        cls.logs = [json.loads(x) for x in (cls.out / "logs.jsonl").read_text().splitlines()]
        cls.results = json.loads((cls.out / "results.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_files_and_labels(self) -> None:
        for n in ("table.md", "results.json", "logs.jsonl", "config.json", "ground_truth.json", "episodes.json"):
            self.assertTrue((self.out / n).is_file(), n)
        self.assertTrue((self.out / "table.md").read_text().startswith(f"# {SYNTHETIC_LABEL}"))
        self.assertIn("SYNTHETIC - not study results", (self.out / "table.md").read_text())
        self.assertEqual(self.results["label"], SYNTHETIC_LABEL)
        self.assertTrue(self.results["synthetic"])
        self.assertTrue(json.loads((self.out / "ground_truth.json").read_text())["synthetic"])

    def test_config_hash_in_every_log(self) -> None:
        cfg = json.loads((self.out / "config.json").read_text())
        sha = config_sha256(cfg["config"])
        self.assertEqual(cfg["config_sha256"], sha)
        self.assertEqual(self.results["config_sha256"], sha)
        self.assertTrue(all(g["config_sha256"] == sha for g in self.logs))

    def test_logs_have_required_fields_and_no_truth(self) -> None:
        for g in self.logs:
            self.assertLessEqual(REQUIRED_LOG_FIELDS, set(g))
            self.assertEqual(g["label"], SYNTHETIC_LABEL)
            self.assertEqual(len(g["requested_stops"]), g["n_views"])
            self.assertEqual([o["stop"] for o in g["evidence"]], g["requested_stops"])
            for o in g["evidence"]:
                self.assertEqual(set(o), OBSERVATION_FIELDS)
            self.assertFalse({k for k in _keys(g) if "truth" in k.lower()}, g["episode_id"])

    def test_log_counts_and_effort(self) -> None:
        n_ep, budgets, caps, seeds = 36, 5, 2, 3   # deterministic multi-view: F, G, G-myopic
        self.assertEqual(len(self.logs), n_ep * (2 + caps * budgets * 2 * (3 + seeds)))
        self.assertEqual(self.results["n_logs"], len(self.logs))
        by = {}
        for g in self.logs:
            by.setdefault(g["policy"], []).append(g)
        self.assertEqual(set(by), {"S", "FULL", "F", "G", "R", "G-myopic", "F-fcm", "G-fcm", "R-fcm",
                                   "G-myopic-fcm"})
        self.assertTrue(all(g["n_views"] == 1 and g["travel_cm"] == 0 for g in by["S"]))
        self.assertTrue(all(g["n_views"] == 5 for g in by["FULL"]))
        self.assertEqual({g["seed"] for g in by["R"]}, {0, 1, 2})
        self.assertTrue(all(g["seed"] is None for g in by["F"] + by["G"]))
        for g in self.logs:
            self.assertLessEqual(g["n_additional_views"], g["budget_additional_views"])
            if g["travel_cap_cm"] is not None:
                self.assertLessEqual(g["travel_cm"], g["travel_cap_cm"] + 1e-6)

    def test_scored_on_test_split_with_counts(self) -> None:
        eps = json.loads((self.out / "episodes.json").read_text())["episodes"]
        test_arr = {e["arrangement_id"] for e in eps if e["split"] == "test"}
        primary = [m for m in self.results["metrics"] if m["scope"] == "test"]
        self.assertTrue(primary)
        for m in primary:
            self.assertEqual(m["overall"]["n_arrangements"], len(test_arr))
            self.assertEqual(m["overall"]["n_episodes"], 6 * len(test_arr))
        pairs = {(b["policy_a"], b["policy_b"], b["metric"], b["travel_cap_cm"]) for b in self.results["bootstrap"]}
        # (R, S) and the H1 pairs X vs X-fcm were added after review finding P3
        for a, b in (("G", "F"), ("G", "R"), ("G", "S"), ("F", "S"), ("R", "S"), ("G", "G-myopic"),
                     ("F", "F-fcm"), ("R", "R-fcm"), ("G", "G-fcm"), ("G-myopic", "G-myopic-fcm")):
            for metric in ("false_removal_rate", "removal_recall", "coverage"):
                self.assertIn((a, b, metric, None), pairs)
                self.assertIn((a, b, metric, CAP2), pairs)
        for b in self.results["bootstrap"]:
            self.assertEqual(b["n_arrangements"], len(test_arr))
            self.assertEqual(b["B"], 200)

    def test_bootstrap_matches_direct_call(self) -> None:
        logs = [run.EpisodeLog.from_dict(g) for g in self.logs]
        groups = run.group_results(logs)
        truth = GroundTruth.load(self.out / "ground_truth.json")
        test = {k: [r for r in v if r.split.value == "test"] for k, v in groups.items()}
        direct = paired_diff(test[("G", 3, None)], test[("F", 3, None)], truth, "coverage", B=200, seed=3)
        row = next(b for b in self.results["bootstrap"] if (b["policy_a"], b["policy_b"], b["metric"],
                                                              b["travel_cap_cm"]) == ("G", "F", "coverage", None))
        self.assertEqual((row["estimate"], row["ci_lo"], row["ci_hi"]), (direct.estimate, direct.ci_lo, direct.ci_hi))

    def test_reproducible_by_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            again = _run(tmp, "b")
            other = _run(tmp, "c", ["--seed", "4"])
            for n in ("results.json", "logs.jsonl", "table.md"):
                self.assertEqual((self.out / n).read_bytes(), (again / n).read_bytes(), n)
            self.assertNotEqual((self.out / "results.json").read_bytes(), (other / "results.json").read_bytes())

    def test_reachable_view_sets_hand_checked(self) -> None:
        pos = Positions.default()
        step = pos.nominal_step_cm()
        r = run.reachable_view_sets(pos, 2, 2 * step)        # one extreme, or two stops on one side
        self.assertEqual((r["max_views"], r["max_view_sets"]), (3, [[0, 1, 2], [2, 3, 4]]))
        r = run.reachable_view_sets(pos, 2, 3 * step)        # + both neighbours (P2 then P4)
        self.assertEqual(r["max_view_sets"], [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        r = run.reachable_view_sets(pos, 4, 6 * step)        # both extremes: all 5 stops
        self.assertEqual((r["max_views"], r["n_max_view_sets"]), (5, 1))
        r = run.reachable_view_sets(pos, 4, 4 * step)        # P3 -> P2 -> P1 -> ... P4 needs 3 more
        self.assertEqual((r["max_views"], r["max_view_sets"]), (4, [[0, 1, 2, 3], [1, 2, 3, 4]]))
        r = run.reachable_view_sets(pos, 2, None)
        self.assertEqual((r["max_views"], r["n_max_view_sets"]), (3, 6))
        r = run.reachable_view_sets(pos, 3, 1 * step)        # only one neighbour fits
        self.assertEqual((r["max_views"], r["n_max_view_sets"]), (2, 2))

    def test_diagnostics_in_results(self) -> None:
        d = self.results["diagnostics"]
        self.assertEqual({x["policy"] for x in d["used_view_sets"]}, {"F", "R", "G", "G-myopic"})
        dis = {(x["scope"], x["policy_a"], x["policy_b"], x["budget"], x["travel_cap_cm"]): x
               for x in d["disagreement"]}
        for x in dis.values():
            self.assertTrue(0.0 <= x["path_disagreement_rate"] <= 1.0)
            self.assertTrue(0.0 <= x["decision_disagreement_rate"] <= x["path_disagreement_rate"] + 1e-12)
        # recompute G vs F at the 2-step cap, b = 3, from the logs
        paths = {}
        for g in self.logs:
            if g["budget_additional_views"] == 3 and g["travel_cap_cm"] == CAP2 and g["policy"] in ("F", "G"):
                paths.setdefault(g["episode_id"], {})[g["policy"]] = g["requested_stops"]
        n_diff = sum(p["F"] != p["G"] for p in paths.values())
        row = dis[("all", "G", "F", 3, CAP2)]
        self.assertEqual((row["path_differs"], row["n_episodes"]), (n_diff, len(paths)))
        self.assertGreater(n_diff, 0)   # cap-aware G leaves F's P3 -> P1 path at 2 steps

    def test_without_synthetic_exits_2(self) -> None:
        with mock.patch("sys.stderr"):
            self.assertEqual(run.main(["--n-arr", "6"]), 2)

    def test_parse_budgets(self) -> None:
        self.assertEqual(run.parse_budgets("0..4"), (0, 1, 2, 3, 4))
        self.assertEqual(run.parse_budgets("3,1,1"), (1, 3))
        self.assertEqual(run.parse_cap_steps("none"), ())
        self.assertEqual(run.parse_cap_steps("6,1,2,inf,2"), (1.0, 2.0, 6.0))
        for bad in ("0", "-1", "x"):
            with self.assertRaises(argparse.ArgumentTypeError):
                run.parse_cap_steps(bad)
        with self.assertRaises(argparse.ArgumentTypeError):
            run.parse_budgets("0..5")


class TestReviewFixes(unittest.TestCase):
    """Regressions for the offline-core review (LEAK-2, SPEC-1..5, SPEC-7, DESIGN-1)."""

    def test_default_cap_grid_in_steps(self) -> None:
        # D-cap: {1, 2, 3, 4, 6} steps plus uncapped; caps are multiples of the nominal step, no slack
        args = run.build_parser().parse_args(["--synthetic"])
        self.assertEqual(args.cap_steps, (1, 2, 3, 4, 6))
        pos = Positions.default()
        cfg = run.build_config(args, pos, run.Thresholds(), run.SyntheticParams())
        self.assertEqual(cfg["cap_steps"], [None, 1, 2, 3, 4, 6])
        self.assertEqual(cfg["travel_caps_cm"][0], None)
        for steps, cm in zip(cfg["cap_steps"][1:], cfg["travel_caps_cm"][1:]):
            self.assertEqual(cm, steps * pos.nominal_step_cm())
        self.assertAlmostEqual(cfg["nominal_step_cm"], 23.561944901923447, places=12)
        self.assertEqual(cfg["travel_tol_cm"], 1e-6)
        self.assertFalse(any(abs(c - 50.0) < 1 for c in cfg["travel_caps_cm"][1:]))   # old 50 cm example is gone

    def test_no_scoring_files_and_no_spec_in_env_during_run(self) -> None:
        seen_files: list[set[str]] = []
        env_args: list[tuple] = []
        real_run_stage, real_env = run.run_stage, run.ReplayEnv

        def spy_run_stage(*a, **kw):
            seen_files.append({p.name for p in spy_run_stage.out.iterdir()})
            return real_run_stage(*a, **kw)

        def spy_env(*a, **kw):
            env_args.append(a + tuple(kw.values()))
            return real_env(*a, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spy"
            out.mkdir()
            (out / "ground_truth.json").write_text("{}")  # stale file from an earlier run is removed first
            spy_run_stage.out = out
            with mock.patch.object(run, "run_stage", spy_run_stage), mock.patch.object(run, "ReplayEnv", spy_env), \
                    mock.patch("builtins.print"):
                self.assertEqual(run.main([*TINY, "--out", str(out)]), 0)
            self.assertTrue((out / "ground_truth.json").is_file())
        self.assertEqual(len(seen_files), 1)
        self.assertFalse(seen_files[0] & {"ground_truth.json", "evidence.jsonl", "episodes.json"})
        self.assertTrue(env_args)
        for args in env_args:
            for a in args:
                self.assertNotIsInstance(a, (EpisodeSpec, GroundTruth, str))
                for cond in ("visible", "occluded", "removed"):
                    self.assertNotIn(cond, repr(a))

    def test_table_caveats_and_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = _run(tmp, "t")
            table = (out / "table.md").read_text()
            res = json.loads((out / "results.json").read_text())
            logs = [run.EpisodeLog.from_dict(json.loads(x)) for x in (out / "logs.jsonl").read_text().splitlines()]
            anywhere = _run(tmp, "w", ["--block-anywhere"])
            table_w = (anywhere / "table.md").read_text()
            cfg_w = json.loads((anywhere / "config.json").read_text())["config"]
        self.assertIn("Generator caveat (regime: start-blocked)", table)
        self.assertIn("operating condition, reported as curves, not tuned", table)
        self.assertIn(f"travel cap 2 steps ({CAP2:.2f} cm)", table)
        self.assertNotIn("50 cm", table)
        self.assertIn("## Cap curve, b = 3", table)
        self.assertIn("## Diagnostic: distinct view sets per budget and cap", table)
        self.assertIn("## Diagnostic: path and decision disagreement", table)
        self.assertIn("max achievable coverage (test)", table)
        self.assertIn("max achievable cov", table)
        self.assertIn("FULL (uncapped)", table)
        self.assertIn("uncapped reference (exceeds cap)", table)
        section = ""
        for line in table.splitlines():  # FULL under a capped heading is always marked as uncapped
            if line.startswith("## "):
                section = line
            if line.startswith("| FULL"):
                capped = "travel cap" in section and "no travel cap" not in section
                self.assertEqual(capped, line.startswith("| FULL (uncapped) |"), section)
        self.assertTrue(all("identical_runs" in b for b in res["bootstrap"]))
        self.assertTrue(res["config"]["generator"]["block_includes_start"])
        self.assertFalse(cfg_w["generator"]["block_includes_start"])
        self.assertIn("block-anywhere", table_w)
        self.assertNotIn("regime: start-blocked", table_w)
        self.assertEqual(res["config"]["positions_file"], "configs/positions.json")
        self.assertEqual(len(res["config"]["positions_file_sha256"]), 64)
        groups = run.group_results(logs)  # DESIGN-1: degenerate pairs are detected and flagged
        self.assertTrue(run.identical_runs(groups[("F", 3, None)], groups[("F", 3, None)]))
        self.assertFalse(run.identical_runs(groups[("S", 0, None)], groups[("FULL", 4, None)]))
        for b in res["bootstrap"]:
            ka = (b["policy_a"], 3, b["travel_cap_cm"]) if b["policy_a"] not in ("S", "FULL") else ("S", 0, None)
            kb = (b["policy_b"], 3, b["travel_cap_cm"]) if b["policy_b"] not in ("S", "FULL") else ("S", 0, None)
            test = lambda k: [r for r in groups[k] if r.split.value == "test"]
            self.assertEqual(b["identical_runs"], run.identical_runs(test(ka), test(kb)))

    def test_bad_positions_file_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "p.json"
            bad.write_text(json.dumps({**Positions.default().to_dict(), "schema": "bogus"}))
            with mock.patch("sys.stderr"):
                self.assertEqual(run.main([*TINY, "--positions", str(bad), "--out", str(Path(tmp) / "o")]), 2)


class TestNoTruthReachesPolicies(unittest.TestCase):
    def test_run_stage_signature_has_no_truth(self) -> None:
        sig = inspect.signature(run.run_stage)
        self.assertFalse([p for p in sig.parameters if "truth" in p.lower()])
        self.assertNotIn("GroundTruth", str(sig))
        self.assertNotIn("GroundTruth", str(inspect.signature(run.generate_stage).return_annotation))

    def test_policies_and_rule_see_only_observations(self) -> None:
        calls = {"select": 0, "decide": 0}
        fields = {f.name for f in dataclasses.fields(Observation)}

        def check_obs(tc: unittest.TestCase, observations) -> None:
            tc.assertIsInstance(observations, tuple)
            for o in observations:
                tc.assertIs(type(o), Observation)
                tc.assertEqual(set(vars(o)), OBSERVATION_FIELDS)
                tc.assertEqual(fields, OBSERVATION_FIELDS)

        def wrap_select(orig):
            def select(self_, observations, candidates, positions, current_stop):
                calls["select"] += 1
                check_obs(self, observations)
                self.assertTrue(candidates and all(type(c) is int for c in candidates))
                self.assertIs(type(positions), Positions)
                self.assertIs(type(current_stop), int)
                for a in (observations, candidates, positions, current_stop):
                    self.assertNotIsInstance(a, FORBIDDEN)
                visited = {o.stop for o in observations}
                self.assertFalse(visited & set(candidates))
                return orig(self_, observations, candidates, positions, current_stop)
            return select

        orig_decide = StoppingRule.decide

        def decide(self_, observations, can_continue):
            calls["decide"] += 1
            check_obs(self, tuple(observations))
            self.assertIs(type(can_continue), bool)
            return orig_decide(self_, observations, can_continue)

        patches = [mock.patch.object(cls, "select", wrap_select(cls.select))
                   for cls in (policies.F, policies.G, policies.GMyopic, policies.R, policies.FullSweep)]
        patches.append(mock.patch.object(StoppingRule, "decide", decide))
        for p in patches:
            p.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _run(tmp, "spy")
        finally:
            for p in patches:
                p.stop()
        self.assertGreater(calls["select"], 100)
        self.assertGreater(calls["decide"], 100)


if __name__ == "__main__":
    unittest.main()
