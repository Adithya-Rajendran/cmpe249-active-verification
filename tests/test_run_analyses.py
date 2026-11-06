"""verify.run's declared test analyses (real-v6): each multi-view policy against S, the per-family breakdowns with their
target composition and A - G per family, the occlusion-gate-off ablation (F and R, tau_occ = 1.0), the tau_match
risk-coverage sweep over the frozen candidate's grid, the geometric diagnostic-view coverage and the gate's per-view
rates against the geometric labels (verify.select's definitions). Hand-built sets and synthetic evidence only; no number
here is a result."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from analysis.bootstrap import paired_diff
from analysis.metrics import compute_all
from tests.test_ranker_side import write_side
from tests.test_run_freeze import DEV_SLOTS, RUN_QUICK, SELECT_QUICK, TEST_SLOTS, quiet, retarget
from tests.test_select_sim import SLOTS, write_sim_set
from verify import ranker as rk
from verify import run
from verify import select as sel
from verify.manifest import ArrangementRecord, GeometricLabels, episodes_from_arrangements, load_geometric_labels
from verify.policies import NOCUE_FEATURES, RANKER_FEATURES
from verify.schema import (
    Condition,
    EpisodeLog,
    EvidenceStore,
    GroundTruth,
    Lighting,
    Positions,
    RunContext,
    Split,
    Thresholds,
    file_sha256,
)
from verify.synthetic import SyntheticParams

# one IND28 test arrangement: a singleton family, whose A - G bootstrap must be skipped
SLOTS_SINGLE = (*TEST_SLOTS, ("ST0010", "IND28", None))
ALL_FLAGS = ("--gate-off-ablation", "--tau-match-sweep")


def logs_of(out: Path) -> list[EpisodeLog]:
    return [EpisodeLog.from_dict(json.loads(x)) for x in (out / "logs.jsonl").read_text().splitlines()]


class AnalysesRunTest(unittest.TestCase):
    """One frozen run with --side, both rankers, --gate-off-ablation and --tau-match-sweep on a test set with a
    singleton family."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        dev_args = write_sim_set(d / "dev", DEV_SLOTS)
        dev_side = write_side(d / "dev", DEV_SLOTS)
        with quiet():
            assert sel.main([*dev_args, "--side", str(dev_side), *SELECT_QUICK, "--out", str(d / "cand")]) == 0
        cls.cand = d / "cand" / "frozen_candidate.json"
        cls.th = sel.candidate_thresholds(sel.require_freeze_eligible(cls.cand))
        dev = sel.sim_dev_data(*[d / "dev" / f for f in ("manifest.json", "evidence.jsonl", "view_labels.json",
                                                          "positions.json")], dev_side)
        cls.models = {}
        prov = {"candidate_sha256": file_sha256(cls.cand), "note": "test"}
        for name, feats in (("A", RANKER_FEATURES), ("A-nocue", NOCUE_FEATURES)):
            cls.models[name] = d / f"model_{name}.json"
            rk.save_model(cls.models[name], name, rk.fit_with_selection(rk.training_rows(dev, cls.th, feats),
                                                                       dev.arrangements), cls.th, prov)
        cls.test_args = write_sim_set(d / "test", SLOTS_SINGLE, split="test")
        retarget(d / "test")
        cls.test_side = write_side(d / "test", SLOTS_SINGLE)
        cls.frozen = [*cls.test_args, *RUN_QUICK, "--candidate", str(cls.cand)]
        cls.out = d / "out"
        with quiet():
            cls.rc = run.main([*cls.frozen, "--side", str(cls.test_side), "--model", f"A={cls.models['A']}",
                               "--model", f"A-nocue={cls.models['A-nocue']}", *ALL_FLAGS, "--out", str(cls.out)])
        cls.res = json.loads((cls.out / "results.json").read_text())
        cls.table = (cls.out / "table.md").read_text()
        cls.logs = logs_of(cls.out)
        cls.truth = GroundTruth.load(cls.out / "ground_truth.json")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def main(self, *extra: str, out: str = "o2") -> tuple[int, str]:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            rc = run.main([*self.frozen, *extra, "--out", str(self.d / out)])
        return rc, err.getvalue()

    def pairs(self, res: dict | None = None) -> set[tuple[str, str]]:
        return {(b["policy_a"], b["policy_b"]) for b in (res or self.res)["bootstrap"]}

    # ------------------------------------------------------------ item 1: each multi-view policy against S

    def test_every_multi_view_policy_against_s(self) -> None:
        self.assertEqual(self.rc, 0)
        for p in ("G", "F", "R", "A", "A-nocue", "Dir", "G-lookahead", "G-myopic"):
            for v in ("", "-fcm"):
                self.assertIn((p + v, "S"), self.pairs(), p + v)
        self.assertEqual(self.res["config"]["config_version"], "real-v6")
        self.assertIn(["A", "S"], self.res["config"]["bootstrap_pairs"])
        rows = [b for b in self.res["bootstrap"] if (b["policy_a"], b["policy_b"]) == ("Dir", "S")]
        self.assertEqual({b["travel_cap_cm"] for b in rows}, set(self.res["config"]["travel_caps_cm"]))
        self.assertIn("| no travel cap | A - S | coverage |", self.table)

    def test_pairs_of_policies_that_did_not_run_are_skipped(self) -> None:
        rc, _ = self.main()   # no --side, no --model: no Dir, A or A-nocue
        self.assertEqual(rc, 0)
        res = json.loads((self.d / "o2" / "results.json").read_text())
        pairs = self.pairs(res)
        self.assertTrue({("G-lookahead", "S"), ("G-myopic", "S"), ("G", "S")} <= pairs)
        self.assertFalse({p for p in pairs if {"A", "A-nocue", "Dir"} & set(p)})
        self.assertFalse({p for p in pairs if "gateoff" in p[1]})   # no --gate-off-ablation
        self.assertNotIn("tau_match_sweep", res)
        self.assertIsNone(res["config"]["gate_off_ablation"])
        self.assertIsNone(res["config"]["tau_match_sweep"])

    # ------------------------------------------------------------ item 2: per-family breakdowns

    def test_family_rows_are_compute_all_on_the_family_subset(self) -> None:
        fam = self.res["by_family"]
        self.assertTrue(fam["ran"])
        self.assertEqual(fam["families"], ["IND28", "M16-L", "M16-R", "S8", "white_box", "ycb_box"])
        pols = {m["policy"] for m in fam["metrics"]}
        self.assertTrue({"S", "FULL", "A", "A-fcm", "Dir", "R-fcm", "G-myopic"} <= pols, pols)
        self.assertFalse({p for p in pols if run._is_sensitivity(p) or "gateoff" in p or "-tm" in p}, pols)
        self.assertEqual({m["lighting"] for m in fam["metrics"]}, {"L1", "L2"})
        self.assertEqual({m["budget"] for m in fam["metrics"] if m["policy"] not in ("S", "FULL")}, {3})
        self.assertEqual({m["travel_cap_cm"] for m in fam["metrics"] if m["policy"] == "R"},
                         set(self.res["config"]["travel_caps_cm"]))
        groups = run.group_results(self.logs)
        m = next(m for m in fam["metrics"] if (m["scope"], m["lighting"], m["family"], m["policy"],
                                               m["travel_cap_cm"]) == ("test", "L2", "M16-L", "R", None))
        sub = [r for r in groups[("R", 3, None)] if r.lighting is Lighting.L2 and r.arrangement_id in
               ("ST0002", "ST0003")]
        want = compute_all(sub, self.truth, load_geometric_labels(self.test_args[5], 5).no_diagnostic_view(),
                           run._seed_removed(self.logs)[("R", 3, None)])
        self.assertEqual(json.loads(json.dumps(run.to_jsonable(want))), {k: m[k] for k in want})
        self.assertIn("by_condition", m)
        self.assertEqual(m["overall"]["n_arrangements"], 2)

    def test_target_composition_from_the_run(self) -> None:
        comp = {c["family"]: c for c in self.res["by_family"]["composition"] if c["scope"] == "test"}
        self.assertEqual(comp["IND28"]["targets"], {"foam_brick": 1})                       # ST0010 (index 10)
        self.assertEqual(comp["M16-L"]["targets"], {"foam_brick": 1, "tuna_fish_can": 1})   # ST0002, ST0003
        self.assertIn("**IND28** (1 test arrangements; targets: foam_brick 1)", self.table)

    def test_family_bootstrap_and_the_singleton_skip(self) -> None:
        fam = self.res["by_family"]
        skipped = {(r["lighting"], r["family"]) for r in fam["bootstrap_skipped"]}
        self.assertEqual(skipped, {("L1", "IND28"), ("L2", "IND28")})
        self.assertEqual({r["family"] for r in fam["bootstrap"]}, {"M16-L", "M16-R", "S8", "white_box", "ycb_box"})
        self.assertEqual(len(fam["bootstrap"]), 5 * 2 * 3)
        self.assertEqual(fam["bootstrap_seed"], run.family_bootstrap_seed(0))
        self.assertNotEqual(fam["bootstrap_seed"], 0)
        groups = run.group_results(self.logs)
        row = next(r for r in fam["bootstrap"] if (r["lighting"], r["family"], r["metric"]) ==
                   ("L1", "S8", "coverage"))
        s8 = ("ST0000", "ST0001")
        pick = lambda k: [r for r in groups[k] if r.lighting is Lighting.L1 and r.arrangement_id in s8]   # noqa: E731
        want = paired_diff(pick(("A", 3, None)), pick(("G", 3, None)), self.truth, "coverage", B=50,
                           seed=run.family_bootstrap_seed(0))
        self.assertFalse(want.stratified)
        self.assertEqual((row["estimate"], row["ci_lo"], row["ci_hi"], row["B"]),
                         (want.estimate, want.ci_lo, want.ci_hi, 50))
        self.assertEqual(row["inconclusive"], row["ci_lo"] <= 0.0 <= row["ci_hi"])
        self.assertIn("A - G on IND28, L1 queries: not bootstrapped (1 test arrangement)", self.table)
        self.assertIn("## L2 queries - By family, b = 3, uncapped (test split; descriptive, H3)", self.table)

    def test_family_breakdown_needs_every_family(self) -> None:
        d = self.d / "nofam"
        args = write_sim_set(d, TEST_SLOTS, split="test", family=False)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = run.main([*args, *RUN_QUICK, "--candidate", str(self.cand), "--out", str(d / "o")])
        self.assertEqual(rc, 0)
        res = json.loads((d / "o" / "results.json").read_text())
        self.assertFalse(res["by_family"]["ran"])
        self.assertIn("occluder_family", res["by_family"]["note"])
        self.assertIn("By family, b = 3, uncapped (test split; descriptive, H3)\n\nNot computed",
                      (d / "o" / "table.md").read_text())

    # ------------------------------------------------------------ item 3: the occlusion-gate-off ablation

    def test_gate_off_rows_are_f_and_r_only(self) -> None:
        names = {g.policy for g in self.logs if "gateoff" in g.policy}
        self.assertEqual(names, {"F-gateoff", "R-gateoff"})
        self.assertEqual({g.budget_additional_views for g in self.logs if g.policy in names}, {3})
        caps = {g.travel_cap_cm for g in self.logs if g.policy == "R-gateoff"}
        self.assertEqual(caps, set(self.res["config"]["travel_caps_cm"]))
        self.assertEqual({g.seed for g in self.logs if g.policy == "R-gateoff"}, {0, 1})
        cfg = self.res["config"]["gate_off_ablation"]
        self.assertEqual((cfg["policies"], cfg["tau_occ"]), (["F", "R"], 1.0))
        rows = {(b["policy_a"], b["policy_b"], b["travel_cap_cm"]) for b in self.res["bootstrap"]
                if "gateoff" in b["policy_b"]}
        self.assertEqual(rows, {("F", "F-gateoff", None), ("R", "R-gateoff", None)})   # primary setting only
        self.assertIn("Occlusion-gate-off ablation, b = 3 (test split; F and R only", self.table)
        self.assertIn("| F-gateoff | no travel cap |", self.table)
        self.assertFalse(run._is_sensitivity("F-gateoff"))

    def test_gate_off_setting_changes_tau_occ_only(self) -> None:
        suffix, t = run.gate_off_setting(self.th)
        self.assertEqual((suffix, t.tau_occ), ("gateoff", 1.0))
        self.assertEqual(dataclasses.replace(t, tau_occ=self.th.tau_occ), self.th)

    # ------------------------------------------------------------ item 4: the tau_match sweep

    def test_sweep_log_names_and_settings(self) -> None:
        grid = self.res["tau_match_sweep"]["grid"]
        frozen = self.th.tau_match
        self.assertEqual(grid, sel.require_freeze_eligible(self.cand)["settings"]["tau_match_grid"])
        self.assertIn(frozen, grid)
        others = [v for v in grid if v != frozen]
        self.assertTrue(others)
        want = {f"{p}-tm{v:g}" for p in run.TM_SWEEP_POLICIES for v in others}
        got = {g.policy for g in self.logs if "-tm" in g.policy}
        self.assertEqual(got, want)
        for g in self.logs:
            if g.policy in got:
                self.assertIsNone(g.travel_cap_cm)
                self.assertEqual(g.budget_additional_views, {"S": 0, "FULL": 4}.get(g.policy.split("-tm")[0], 3))
        self.assertEqual(self.res["config"]["tau_match_sweep"]["values_run"], others)

    def test_sweep_curves_mark_the_frozen_point(self) -> None:
        curves = self.res["tau_match_sweep"]["curves"]
        self.assertEqual({c["policy"] for c in curves}, set(run.TM_SWEEP_POLICIES))
        for c in curves:
            frozen = [x for x in c["points"] if x["frozen"]]
            self.assertEqual(len(frozen), 1, c["policy"])
            self.assertEqual((frozen[0]["row"], frozen[0]["tau_match"]), (c["policy"], self.th.tau_match))
            for x in c["points"]:
                m = next(m for m in self.res["metrics"] if (m["scope"], m["lighting"], m["policy"]) ==
                         (c["scope"], c["lighting"], x["row"]) and m["travel_cap_cm"] is None
                         and m["budget"] == c["budget"])["overall"]
                self.assertEqual(x["coverage"], m["coverage"])
                fp = m["confusion"]["removed"]["present"]
                self.assertEqual(x["false_present_rate_removed"], fp / m["n_removed"])
        self.assertIn(f"| A | **{self.th.tau_match:g} (frozen)** |", self.table)

    def test_sweep_grid_comes_from_the_candidate(self) -> None:
        th = Thresholds(tau_match=0.5)
        self.assertEqual(run.tau_match_grid_of({"settings": {"tau_match_grid": [0.6, 0.5, 0.4]}}, th),
                         (0.4, 0.5, 0.6))
        self.assertEqual([s for s, _ in run.tau_match_sweep_settings(th, (0.4, 0.5, 0.6))], ["tm0.4", "tm0.6"])
        self.assertEqual({t.tau_match for _, t in run.tau_match_sweep_settings(th, (0.4, 0.5))}, {0.4})
        for doc in ({}, {"settings": {}}, {"settings": {"tau_match_grid": []}},
                    {"settings": {"tau_match_grid": [0.4, 0.6]}},          # the frozen value is not in it
                    {"settings": {"tau_match_grid": [0.5, "x"]}}, {"settings": {"tau_match_grid": [0.5, 1.5]}}):
            with self.assertRaises(run.SweepRefused, msg=str(doc)):
                run.tau_match_grid_of(doc, th)

    def test_sweep_refuses_without_a_candidate_or_a_grid(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            rc = run.main([*self.test_args, *RUN_QUICK, "--placeholder-thresholds", "--tau-match-sweep",
                           "--out", str(self.d / "o3")])
        self.assertEqual(rc, 2)
        self.assertIn("--tau-match-sweep needs --candidate", err.getvalue())
        bad = self.d / "nogrid"
        shutil.copytree(self.cand.parent, bad, dirs_exist_ok=True)
        doc = json.loads((bad / "frozen_candidate.json").read_text())
        del doc["settings"]["tau_match_grid"]
        text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
        (bad / "frozen_candidate.json").write_text(text)
        (bad / "frozen_candidate.json.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
        self.assertEqual(sel.require_freeze_eligible(bad / "frozen_candidate.json")["thresholds"],
                         self.th.to_dict())   # an eligible candidate, only without its grid
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            rc = run.main([*self.test_args, *RUN_QUICK, "--candidate", str(bad / "frozen_candidate.json"),
                           "--tau-match-sweep", "--out", str(self.d / "o3")])
        self.assertEqual(rc, 2)
        self.assertIn("--tau-match-sweep refused: the candidate records no predeclared tau_match grid", err.getvalue())
        self.assertFalse((self.d / "o3" / "logs.jsonl").exists())

    def test_new_options_are_refused_with_synthetic(self) -> None:
        for flag in ALL_FLAGS:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                rc = run.main(["--synthetic", "--n-arr", "4", flag, "--out", str(self.d / "syn")])
            self.assertEqual(rc, 2, flag)
            self.assertIn("not --synthetic", err.getvalue())

    def test_new_options_are_in_the_config_hash(self) -> None:
        cfg = json.loads((self.out / "config.json").read_text())
        self.assertEqual(run.config_sha256(cfg["config"]), cfg["config_sha256"])
        for key in ("gate_off_ablation", "tau_match_sweep", "by_family", "view_diagnostics", "bootstrap_pairs"):
            self.assertIn(key, cfg["config"])
            self.assertNotEqual(run.config_sha256({**cfg["config"], key: None}), cfg["config_sha256"], key)

    # ------------------------------------------------------------ items 5 and 6: the view diagnostics

    def test_view_diagnostics_in_results_and_table(self) -> None:
        cov, rates = self.res["diagnostic_view_coverage"], self.res["gate_view_rates"]
        self.assertTrue(cov["ran"] and rates["ran"])
        self.assertEqual((cov["start_stop"], cov["k"]), (2, 2))
        row = next(r for r in cov["rows"] if (r["scope"], r["lighting"], r["family"]) == ("test", "L1", "IND28"))
        self.assertEqual(row["occluded_with_diagnostic_non_start"], {"x": 0, "n": 1, "rate": 0.0})   # all blocked
        self.assertIn("## L1 queries - Geometric diagnostic-view coverage (test split)", self.table)
        self.assertIn("## L2 queries - Gate per-view rates against the geometric labels (test split)", self.table)
        # the test rows equal verify.select's ViewRates on the same rows
        labels = load_geometric_labels(self.test_args[5], 5)
        specs, _ = episodes_from_arrangements([ArrangementRecord.from_dict(a) for a in
                                               json.loads(Path(self.test_args[1]).read_text())["arrangements"]])
        store = EvidenceStore.from_jsonl(self.test_args[3], 5)
        pos = Positions.load(self.test_args[7])
        rows = sel.labelled_view_rows(specs, store, {(e, s): d for e, ds in labels.episodes.items()
                                                     for s, d in enumerate(ds)}, lambda _a: pos)
        want = {k: v.to_dict() for k, v in sel.rates_by_lighting(rows, self.th).items()}
        got = {r["lighting"]: r["rates"] for r in rates["rows"] if r["scope"] == "test"}
        self.assertEqual(got, want)

    def test_no_view_labels_no_view_diagnostics(self) -> None:
        args = [*self.test_args[:4], *self.test_args[6:]]   # the same set without --view-labels
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = run.main([*args, *RUN_QUICK, "--candidate", str(self.cand), "--out", str(self.d / "o4")])
        self.assertEqual(rc, 0)
        res = json.loads((self.d / "o4" / "results.json").read_text())
        for key in ("diagnostic_view_coverage", "gate_view_rates"):
            self.assertFalse(res[key]["ran"])
            self.assertIn("--view-labels", res[key]["note"])
        self.assertIn("Geometric diagnostic-view coverage (test split)\n\nNot computed",
                      (self.d / "o4" / "table.md").read_text())


class GateOffStageTest(unittest.TestCase):
    """run_stage on synthetic evidence: the gate-off setting runs F and R exactly as with tau_occ = 1.0, and leaves
    the k / lam sensitivity runs unchanged."""

    def stage(self, th: Thresholds, sensitivity=(), seed: int = 3) -> list[EpisodeLog]:
        pos = Positions.default()
        specs, store = run.generate_stage(4, seed, SyntheticParams(), pos)
        cap = 2 * pos.nominal_step_cm()
        return run.run_stage(specs, store, pos, th, (1, 3), (None, cap), (0, 1), RunContext(synthetic=True),
                             sensitivity=sensitivity, primary_budget=3)

    def test_gate_off_equals_a_run_with_tau_occ_one(self) -> None:
        th = Thresholds()
        logs = self.stage(th, [run.gate_off_setting(th)])
        off = self.stage(dataclasses.replace(th, tau_occ=1.0))
        key = lambda g: (g.episode_id, g.budget_additional_views, g.travel_cap_cm, g.seed)
        for p in ("F", "R"):
            got = {key(g): (g.decision, g.requested_stops) for g in logs if g.policy == f"{p}-gateoff"}
            want = {key(g): (g.decision, g.requested_stops) for g in off if g.policy == p
                    and g.budget_additional_views == 3}
            self.assertEqual(got, want, p)
        # not vacuous: the gate matters for F on this evidence
        plain = {key(g): g.decision for g in logs if g.policy == "F" and g.budget_additional_views == 3}
        self.assertNotEqual(plain, {key(g): g.decision for g in logs if g.policy == "F-gateoff"})

    def test_k_and_lam_runs_are_unchanged(self) -> None:
        th = Thresholds()
        sens = run.sensitivity_settings(th, (3,), (0.7,))
        a = [g.to_dict() for g in self.stage(th, sens)]
        b = [g.to_dict() for g in self.stage(th, [*sens, run.gate_off_setting(th)]) if "gateoff" not in g.policy]
        self.assertEqual(a, b)


class FamilyAndCoverageUnitTest(unittest.TestCase):
    """The family split, the target composition and the diagnostic-view coverage on hand-built records and labels."""

    RECORDS = (ArrangementRecord("SD0000", "mug", Split.DEV, occluder_family="S8"),
               ArrangementRecord("SD0001", "foam_brick", Split.DEV, occluder_family="M16", m16_side="L"),
               ArrangementRecord("SD0002", "tuna_fish_can", Split.DEV, occluder_family="M16", m16_side="R"),
               ArrangementRecord("SD0003", "foam_brick", Split.DEV, occluder_family="M16", m16_side="L"),
               ArrangementRecord("DES0", "mug", Split.DEV, occluder_family="S8", design_case=True),
               ArrangementRecord("ST0000", "mug", Split.TEST, occluder_family="S8"),
               ArrangementRecord("ST0001", "tuna_fish_can", Split.TEST, occluder_family="S8"),
               ArrangementRecord("ST0002", "mug", Split.TEST, occluder_family="white_box"))

    def test_family_is_the_quota_slot_class(self) -> None:
        fam = run.family_of_records(self.RECORDS)
        self.assertEqual(fam, {a.arrangement_id: a.stratum for a in self.RECORDS})
        self.assertEqual((fam["SD0001"], fam["SD0002"], fam["DES0"]), ("M16-L", "M16-R", "design:S8"))
        tab = ArrangementRecord("SD0009", "mug", Split.DEV, occluder_family="S8", environment="tabletop")
        self.assertEqual(run.family_of_records([tab]), {"SD0009": "tabletop:S8"})
        self.assertIsNone(run.family_of_records([*self.RECORDS, ArrangementRecord("SD0010", "mug", Split.DEV)]))

    def test_target_composition(self) -> None:
        comp = run.target_composition(self.RECORDS, run.family_of_records(self.RECORDS))
        self.assertEqual(comp, [
            {"scope": "test", "family": "S8", "n_arrangements": 2, "targets": {"mug": 1, "tuna_fish_can": 1}},
            {"scope": "test", "family": "white_box", "n_arrangements": 1, "targets": {"mug": 1}},
            {"scope": "dev", "family": "M16-L", "n_arrangements": 2, "targets": {"foam_brick": 2}},
            {"scope": "dev", "family": "M16-R", "n_arrangements": 1, "targets": {"tuna_fish_can": 1}},
            {"scope": "dev", "family": "S8", "n_arrangements": 1, "targets": {"mug": 1}},
            {"scope": "dev", "family": "design:S8", "n_arrangements": 1, "targets": {"mug": 1}}])

    def test_diagnostic_view_coverage_excludes_the_start_stop(self) -> None:
        recs = self.RECORDS[:2]
        specs, _ = episodes_from_arrangements(recs)
        diag = {  # (arrangement, condition) -> diagnostic stops; the start stop is 2 (P3); L1 and L2 alike
            ("SD0000", Condition.OCCLUDED): {2},          # only the start stop: not covered
            ("SD0000", Condition.REMOVED): {0, 2},        # one non-start stop: < k = 2
            ("SD0001", Condition.OCCLUDED): {4},          # covered
            ("SD0001", Condition.REMOVED): {0, 1, 2}}     # two non-start stops: covered
        eps = {}
        for s in specs:
            stops = diag.get((s.arrangement_id, s.condition), set(range(5)))
            eps[s.episode_id] = tuple(st in stops for st in range(5))
        labels = GeometricLabels(eps, {s.episode_id: s.arrangement_id for s in specs},
                                 {s.episode_id: s.lighting for s in specs}, {s.episode_id: s.condition for s in specs},
                                 {}, {}, True)
        fam = run.family_of_records(recs)
        rows = run.diagnostic_view_coverage(specs, labels, 2, 2, fam)
        get = lambda light, f: next(r for r in rows if (r["scope"], r["lighting"], r["family"]) == ("dev", light, f))
        for light, n in (("L1", 1), ("L2", 1), ("pooled", 2)):
            r = get(light, "all")
            self.assertEqual(r["occluded_with_diagnostic_non_start"], {"x": n, "n": 2 * n, "rate": 0.5})
            self.assertEqual(r["removed_with_k_diagnostic_non_start"], {"x": n, "n": 2 * n, "rate": 0.5, "k": 2})
            s8, m16 = get(light, "S8"), get(light, "M16-L")
            self.assertEqual((s8["occluded_with_diagnostic_non_start"]["x"],
                              s8["removed_with_k_diagnostic_non_start"]["x"]), (0, 0))
            self.assertEqual((m16["occluded_with_diagnostic_non_start"]["x"],
                              m16["removed_with_k_diagnostic_non_start"]["x"]), (n, n))
        # with k = 1 the removed SD0000 episodes count too; with start stop 0 SD0001's removed ones keep 2 stops
        r = next(r for r in run.diagnostic_view_coverage(specs, labels, 2, 1) if r["lighting"] == "L1")
        self.assertEqual(r["removed_with_k_diagnostic_non_start"]["x"], 2)
        r = next(r for r in run.diagnostic_view_coverage(specs, labels, 0, 2) if r["lighting"] == "L1")
        self.assertEqual((r["occluded_with_diagnostic_non_start"]["x"], r["removed_with_k_diagnostic_non_start"]["x"]),
                         (2, 1))
        self.assertEqual({r["family"] for r in run.diagnostic_view_coverage(specs, labels, 2, 2)}, {"all"})


class ViewRatesMatchSelectTest(unittest.TestCase):
    """The gate's per-view rates in verify.run are verify.select's, on the same dev rows."""

    def test_dev_rates_equal_select(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            args = write_sim_set(d / "dev", SLOTS, camouflaged=True)
            dev = sel.sim_dev_data(*(Path(args[i]) for i in (1, 3, 5, 7)))
            th = Thresholds(tau_match=0.5, tau_q=0.6, tau_occ=0.3, k=2, lam=0.0)
            rows = sel.view_rows(dev)
            labels = load_geometric_labels(args[5], 5)
            store = EvidenceStore.from_jsonl(args[3], 5)
            specs = sorted(dev.specs, key=lambda s: s.episode_id)
            # the factored-out row builder gives verify.select's own rows
            self.assertEqual(sel.labelled_view_rows(dev.specs, store, {(l.episode_id, l.stop): l.diagnostic
                                                                       for l in dev.view_labels},
                                                    lambda _a: dev.positions), rows)
            got = run.gate_view_rates(specs, store, labels, dev.positions, th)
            want = {k: v.to_dict() for k, v in sel.rates_by_lighting(rows, th).items()}
            self.assertEqual({r["lighting"]: r["rates"] for r in got}, want)
            self.assertEqual({r["scope"] for r in got}, {"dev"})
            self.assertGreater(want["L1"]["false_clear"]["x"], 0)   # camouflaged: not vacuous
            # and with a run through main on the same dev set
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = run.main([*args, "--budgets", "3", "--cap-steps", "none", "--r-seeds", "1",
                               "--k-sensitivity", "none", "--lam-sensitivity", "none", "--out", str(d / "o")])
            self.assertEqual(rc, 0)
            res = json.loads((d / "o" / "results.json").read_text())
            th0 = Thresholds(**res["thresholds"])
            want0 = {k: v.to_dict() for k, v in sel.rates_by_lighting(rows, th0).items()}
            self.assertEqual({r["lighting"]: r["rates"] for r in res["gate_view_rates"]["rows"]}, want0)
            self.assertEqual(res["by_family"]["bootstrap"], [])
            self.assertIn("no bootstrap ran", res["by_family"]["bootstrap_note"])


if __name__ == "__main__":
    unittest.main()
