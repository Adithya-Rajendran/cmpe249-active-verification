"""Tests for verify/select.py (D-select). Synthetic data exercises logic only; no value here is a threshold."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify import select as sel
from verify.schema import (
    Condition,
    Decision,
    EpisodeSpec,
    EvidenceRecord,
    GroundTruth,
    Lighting,
    Positions,
    RunContext,
    Split,
    SYNTHETIC_LABEL,
    Thresholds,
)
from verify.synthetic import SyntheticParams, generate, generate_layouts

POS = Positions.default()
CTX = RunContext(synthetic=True)


def small_dev(n_arr: int = 9, seed: int = 0) -> sel.DevData:
    return sel.synthetic_dev_data(n_arr, seed, SyntheticParams(), POS)


def row(cond: Condition, diag: bool, q: float, o: float, arr: str = "A01") -> sel.ViewRow:
    return sel.ViewRow(arr, f"{arr}-{cond.value}-L1", cond, 0, q, o, diag)


def counts(fr: int, n_present: int = 40, tr: int = 10, n_removed: int = 20, resolved: int = 40,
           travel: float = 0.0) -> sel.Counts:
    return sel.Counts(n_episodes=n_present + n_removed, n_present=n_present, n_false_removal=fr,
                      n_removed=n_removed, n_true_removal=tr, n_resolved=resolved, total_travel_cm=travel)


class FakeEvaluator:
    """counts() from a table keyed by (k, tau_match, lam); records the arrangements it is asked about."""

    def __init__(self, table: dict[tuple[int, float, float], sel.Counts]) -> None:
        self.table = table
        self.seen: list[frozenset[str]] = []

    def counts(self, th: Thresholds, arrangements) -> sel.Counts:
        self.seen.append(frozenset(arrangements))
        return self.table[(th.k, th.tau_match, th.lam)]


class SplitRefusalTest(unittest.TestCase):
    def test_refuses_test_and_pilot_specs(self) -> None:
        for split in (Split.TEST, Split.PILOT):
            with self.assertRaises(sel.SplitRefused):
                sel.refuse_non_dev([EpisodeSpec("A01", Condition.VISIBLE, Lighting.L1, "t", Split.DEV),
                                    EpisodeSpec("A02", Condition.VISIBLE, Lighting.L1, "t", split)])

    def test_devdata_refuses_test_arrangement(self) -> None:
        specs, store, truth = generate(9, 0, SyntheticParams(), POS)
        factory = sel.RecordStoreFactory.from_store(store, [s.episode_id for s in specs])
        layouts = generate_layouts(9, 0, SyntheticParams(), n_stops=5, start_index=2)
        labels = sel.synthetic_view_labels(specs, layouts, 5)
        with self.assertRaises(sel.SplitRefused):  # specs include TEST arrangements
            sel.DevData(tuple(specs), factory, truth, tuple(labels), POS, True, "x")

    def test_devdata_refuses_store_with_test_episodes(self) -> None:
        specs, store, truth = generate(9, 0, SyntheticParams(), POS)
        dev = [s for s in specs if s.split is Split.DEV]
        factory = sel.RecordStoreFactory.from_store(store, [s.episode_id for s in specs])  # dev + test evidence
        layouts = generate_layouts(9, 0, SyntheticParams(), n_stops=5, start_index=2)
        labels = sel.synthetic_view_labels(dev, layouts, 5)
        with self.assertRaises(sel.SplitRefused):
            sel.DevData(tuple(dev), factory, truth, tuple(labels), POS, True, "x")

    def test_synthetic_dev_data_is_dev_only(self) -> None:
        dev = small_dev()
        self.assertTrue(all(s.split is Split.DEV for s in dev.specs))
        self.assertEqual(len(dev.arrangements), 3)
        self.assertEqual(len(dev.specs), 18)
        self.assertEqual(set(dev.store_factory().episode_ids()), {s.episode_id for s in dev.specs})


class StoreFactoryTest(unittest.TestCase):
    def test_fresh_unsealed_identical(self) -> None:
        dev = small_dev()
        a = dev.store_factory()
        eid = dev.specs[0].episode_id
        rec = a.get(eid, 3)
        a.bind(a.replay_key(eid))
        self.assertTrue(a.sealed)
        b = dev.store_factory()
        self.assertFalse(b.sealed)
        self.assertEqual(b.get(eid, 3), rec)
        self.assertIsNot(a, b)

    def test_evaluator_uses_one_fresh_store_per_config(self) -> None:
        dev = small_dev()
        calls = []
        inner = dev.store_factory

        class Counting:
            n_stops = inner.n_stops

            def __call__(self):
                calls.append(1)
                return inner()
        dev2 = sel.DevData(dev.specs, Counting(), dev.truth, dev.view_labels, POS, True, dev.label_source)
        n0 = len(calls)
        ev = sel.Evaluator(dev2, sel.SelectionSettings(), CTX)
        ev.outcomes(Thresholds(k=2))
        ev.outcomes(Thresholds(k=2))       # cached: no new store
        ev.outcomes(Thresholds(k=3))
        self.assertEqual(len(calls) - n0, 2)
        self.assertEqual(ev.n_runs, 2 * len(dev.specs))


class SyntheticLabelsTest(unittest.TestCase):
    def test_labels_follow_hidden_layout(self) -> None:
        dev = small_dev()
        layouts = generate_layouts(9, 0, SyntheticParams(), n_stops=5, start_index=2)
        lab = {(l.episode_id, l.stop): l.diagnostic for l in dev.view_labels}
        for s in dev.specs:
            lay = layouts[s.arrangement_id]
            for stop in range(5):
                d = lab[(s.episode_id, stop)]
                if s.condition is Condition.VISIBLE:
                    self.assertTrue(d)
                elif stop in lay.blocked or stop in lay.edges(5):
                    self.assertFalse(d)
                else:
                    self.assertTrue(d)

    def test_label_file_roundtrip(self) -> None:
        labels = [sel.ViewLabel("A01-occluded-L1", 0, False), sel.ViewLabel("A01-occluded-L1", 1, True)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.json"
            sel.save_view_labels(labels, p, source="test")
            self.assertEqual(sel.load_view_labels(p), labels)


class GateTest(unittest.TestCase):
    def test_np_rule(self) -> None:
        occ, rem = Condition.OCCLUDED, Condition.REMOVED
        rows = [row(occ, False, 0.9, 0.40), row(occ, False, 0.9, 0.25),     # null set
                row(occ, True, 0.9, 0.05),                                  # diagnostic present: ignored
                row(rem, True, 0.9, 0.05), row(rem, True, 0.9, 0.20), row(rem, True, 0.45, 0.05),
                row(rem, False, 0.9, 0.9)]                                  # non-diagnostic removed: ignored
        # alpha 0: no null view may pass -> tau_occ < 0.25. Best clear rate: tau_occ 0.2, tau_q <= 0.45.
        g = sel.choose_gate(rows, (0.4, 0.45, 0.5), (0.1, 0.2, 0.3, 0.5), 0.0)
        self.assertEqual((g.tau_q, g.tau_occ), (0.45, 0.2))   # tie on clear rate 3/3 -> higher tau_q
        self.assertEqual(g.false_clear, (0, 2))
        self.assertEqual(g.clear, (3, 3))
        self.assertTrue(g.met_alpha)
        # alpha 0.5 lets one null view pass: tau_occ 0.3 also has clear 3/3, but the tie-break prefers lower FCR.
        g2 = sel.choose_gate(rows, (0.4, 0.45, 0.5), (0.1, 0.2, 0.3, 0.5), 0.5)
        self.assertEqual((g2.tau_q, g2.tau_occ), (0.45, 0.2))

    def test_infeasible_falls_back_and_warns(self) -> None:
        rows = [row(Condition.OCCLUDED, False, 0.9, 0.05), row(Condition.REMOVED, True, 0.9, 0.05)]
        warn: list[str] = []
        with self.assertLogs("verify.select", level="WARNING"):
            g = sel.choose_gate(rows, (0.5,), (0.1, 0.2), 0.0, warn)
        self.assertFalse(g.met_alpha)
        self.assertEqual(g.false_clear, (1, 1))
        self.assertTrue(any("alpha_clear" in w for w in warn))


class D10Test(unittest.TestCase):
    def cfg(self, order: int) -> sel.GridConfig:
        return sel.GridConfig(order, order, 2, 0.5, 0.0)

    def test_recall_constraint_then_coverage(self) -> None:
        a = (self.cfg(0), counts(0, tr=17, resolved=50))   # recall 0.85, coverage 50/60
        b = (self.cfg(1), counts(0, tr=10, resolved=58))   # higher coverage but recall 0.5
        c = (self.cfg(2), counts(0, tr=16, resolved=52))   # recall 0.80 (boundary counts), coverage 52/60
        cfg, _, ok = sel.d10_choose([a, b, c], 0.80)
        self.assertTrue(ok)
        self.assertEqual(cfg.order, 2)

    def test_fallback_when_no_recall(self) -> None:
        a = (self.cfg(0), counts(0, tr=5, resolved=50))
        b = (self.cfg(1), counts(0, tr=4, resolved=55))
        cfg, _, ok = sel.d10_choose([a, b], 0.80)
        self.assertFalse(ok)
        self.assertEqual(cfg.order, 1)

    def test_tie_breaks(self) -> None:
        a = (self.cfg(0), counts(0, tr=17, resolved=50, travel=900.0))
        b = (self.cfg(1), counts(0, tr=18, resolved=50, travel=990.0))   # higher recall wins
        c = (self.cfg(2), counts(0, tr=18, resolved=50, travel=600.0))   # same recall, less travel wins
        d = (self.cfg(3), counts(0, tr=18, resolved=50, travel=600.0))   # identical -> grid order
        cfg, _, _ = sel.d10_choose([d, a, b, c], 0.80)
        self.assertEqual(cfg.order, 2)


class WalkTest(unittest.TestCase):
    settings = sel.SelectionSettings(tau_match_grid=(0.4, 0.5, 0.6), k_grid=(3, 2), lam_grid=(0.0, 1.0))
    gate = sel.GateChoice(0.5, 0.3, (0, 1), (1, 1), True, 1, 1)

    def table(self, frr: dict[tuple[int, float], tuple[int, int]]) -> dict:
        return {(k, t, lam): counts(frr[(k, t)][i]) for (k, t) in frr for i, lam in enumerate((0.0, 1.0))}

    def test_stops_at_first_failure_and_flags_nonmonotone(self) -> None:
        # step order: (3,.4) (3,.5) (3,.6) (2,.4) (2,.5) (2,.6); FRR limit 0.10 = 4/40
        frr = {(3, 0.4): (0, 0), (3, 0.5): (3, 2), (3, 0.6): (3, 5),   # worst-over-lam 5/40 fails
               (2, 0.4): (1, 1), (2, 0.5): (2, 2), (2, 0.6): (9, 9)}   # (2,.4),(2,.5) pass but come after the failure
        ev = FakeEvaluator(self.table(frr))
        walk = sel.run_walk(ev, self.gate, self.settings, ["A01"])
        self.assertEqual([w.passed for w in walk], [True, True, False, True, True, False])
        self.assertEqual([w.accepted for w in walk], [True, True, False, False, False, False])
        self.assertAlmostEqual(walk[2].worst_frr, 5 / 40)
        warn = sel.monotonicity_warnings(walk, self.settings)
        self.assertTrue(any("later step(s) would pass" in w for w in warn))
        self.assertTrue(any("FRR higher at k=3 than k=2" in w for w in warn))

    def test_monotone_grid_has_no_warnings(self) -> None:
        frr = {(3, 0.4): (0, 0), (3, 0.5): (0, 1), (3, 0.6): (1, 1),
               (2, 0.4): (1, 1), (2, 0.5): (1, 2), (2, 0.6): (6, 6)}
        walk = sel.run_walk(FakeEvaluator(self.table(frr)), self.gate, self.settings, ["A01"])
        self.assertEqual(sel.monotonicity_warnings(walk, self.settings), [])

    def test_walk_order_conservative_first(self) -> None:
        grid = sel.walk_order(self.settings)
        self.assertEqual([(g.k, g.tau_match, g.lam) for g in grid[:3]], [(3, 0.4, 0.0), (3, 0.4, 1.0), (3, 0.5, 0.0)])
        self.assertEqual(len(grid), 12)
        f = sel.SelectionSettings(policy="F", tau_match_grid=(0.4, 0.5), k_grid=(2,))
        self.assertEqual(len(sel.walk_order(f)), 2)  # lam collapses for F


class CountsAndBoundsTest(unittest.TestCase):
    def test_counts_arrangement_level(self) -> None:
        O = sel.Outcome
        outs = [O("A-v", "A", True, Decision.REMOVED, 10.0, 2), O("A-o", "A", True, Decision.PRESENT, 0.0, 1),
                O("A-r", "A", False, Decision.REMOVED, 20.0, 3), O("B-v", "B", True, Decision.UNKNOWN, 30.0, 4),
                O("B-r", "B", False, Decision.UNKNOWN, 0.0, 1)]
        c = sel.Counts.of(outs)
        self.assertEqual((c.n_present, c.n_false_removal, c.n_removed, c.n_true_removal, c.n_resolved),
                         (3, 1, 2, 1, 3))
        self.assertEqual((c.n_arrangements, c.n_arr_with_present, c.n_arr_with_false_removal), (2, 2, 1))
        self.assertAlmostEqual(c.mean_travel_cm, 12.0)
        b = sel.bounds(c, 0.95)
        self.assertAlmostEqual(b["frr_arrangement"]["upper"]["cp"], 0.9747, places=4)  # 1/2 one-sided
        self.assertEqual(b["frr_episode"]["x"], 1)

    def test_bound_on_zero_of_ten_arrangements(self) -> None:
        c = sel.Counts(n_episodes=60, n_present=40, n_removed=20, n_true_removal=16, n_resolved=50,
                       n_arrangements=10, n_arr_with_present=10, n_arr_with_false_removal=0)
        b = sel.bounds(c, 0.95)
        self.assertAlmostEqual(b["frr_episode"]["upper"]["cp"], 0.0722, places=4)
        self.assertAlmostEqual(b["frr_arrangement"]["upper"]["cp"], 0.2589, places=4)
        self.assertAlmostEqual(b["recall"]["lower"]["cp"], 0.5990, places=4)

    def test_cluster_bootstrap_degenerate(self) -> None:
        O = sel.Outcome
        outs = [O(f"{a}-{i}", a, i < 2, Decision.PRESENT if i < 2 else Decision.REMOVED, 0.0, 1)
                for a in ("A", "B", "C") for i in range(3)]
        bt = sel.cluster_bootstrap(outs, 200, 0)
        self.assertEqual(bt["frr"]["ci95"], [0.0, 0.0])
        self.assertEqual(bt["recall"]["ci95"], [1.0, 1.0])


class NestedCVTest(unittest.TestCase):
    def test_folds_never_see_held_out(self) -> None:
        dev = small_dev(15)
        settings = sel.SelectionSettings(tau_match_grid=(0.5, 0.6), lam_grid=(0.0, 0.5), bootstrap_B=50)
        ev = sel.Evaluator(dev, settings, CTX)
        seen: list[frozenset[str]] = []
        orig = ev.counts

        def spy(th, arrangements):
            arrangements = list(arrangements)
            seen.append(frozenset(arrangements))
            return orig(th, arrangements)
        ev.counts = spy  # type: ignore[method-assign]
        rows = sel.view_rows(dev)
        folds, pooled = sel.nested_cv(ev, rows, settings, dev.arrangements)
        self.assertEqual(len(folds), len(dev.arrangements))
        self.assertEqual(sorted(o.episode_id for o in pooled), sorted(s.episode_id for s in dev.specs))
        per_fold = len(seen) // len(folds)
        for i, f in enumerate(folds):
            for s in seen[i * per_fold:(i + 1) * per_fold]:
                self.assertNotIn(f.held_out, s)
            self.assertEqual(f.held_counts.n_episodes, 6)

    def test_evaluator_matches_direct_replay(self) -> None:
        from verify.policies import G, run_episode
        from verify.replay import ReplayEnv
        from verify.schema import Budget
        from verify.stopping import StoppingRule
        dev = small_dev()
        th = Thresholds(tau_match=0.55, tau_q=0.45, tau_occ=0.2, k=2, lam=0.25)
        ev = sel.Evaluator(dev, sel.SelectionSettings(budget=3), CTX)
        got = ev.outcomes(th)
        store = dev.store_factory()
        keys = store.replay_keys()
        for s in dev.specs:
            run = run_episode(G.from_thresholds(th), ReplayEnv(keys[s.episode_id], store, POS, Budget(3), CTX),
                              StoppingRule(th))
            self.assertEqual((got[s.episode_id].decision, got[s.episode_id].travel_cm), (run.decision, run.travel_cm))


class EndToEndTest(unittest.TestCase):
    def test_cli_writes_candidate_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            shas = []
            for name in ("a", "b"):
                out = Path(d) / name
                rc = sel.main(["--synthetic", "--n-arr", "12", "--seed", "1", "--bootstrap-B", "100",
                               "--tau-match-grid", "0.5,0.6", "--lam-grid", "0,0.5", "--cap-steps", "3",
                               "--out", str(out)])
                self.assertEqual(rc, 0)
                text = (out / "frozen_candidate.json").read_text()
                sha = hashlib.sha256(text.encode()).hexdigest()
                self.assertEqual((out / "frozen_candidate.json.sha256").read_text().split()[0], sha)
                shas.append(sha)
                doc = json.loads(text)
                self.assertTrue(doc["synthetic"])
                self.assertEqual(doc["label"], SYNTHETIC_LABEL)
                self.assertIn("CANDIDATE", doc["status"])
                self.assertLessEqual({"tau_match", "tau_q", "tau_occ", "k", "lam"}, set(doc["thresholds"]))
                self.assertAlmostEqual(doc["travel_cap_cm"], 3 * Positions.load().nominal_step_cm())  # CLI reads configs/positions.json
                rep = (out / "selection_report.md").read_text()
                self.assertTrue(rep.startswith(f"# {SYNTHETIC_LABEL}"))
                self.assertIn("proposed default - confirm", rep)
                self.assertIn("nested grouped CV", rep)
            self.assertEqual(shas[0], shas[1])

    def test_synthetic_provenance_in_candidate_and_settings_hash(self) -> None:
        # Review F4: the generator regime must be recorded and must change settings_sha256.
        with tempfile.TemporaryDirectory() as d:
            docs = {}
            for name, extra in (("default", []), ("anywhere", ["--block-anywhere"])):
                out = Path(d) / name
                rc = sel.main(["--synthetic", "--n-arr", "9", "--seed", "2", "--bootstrap-B", "20",
                               "--tau-match-grid", "0.5", "--lam-grid", "0", "--out", str(out), *extra])
                self.assertEqual(rc, 0)
                docs[name] = json.loads((out / "frozen_candidate.json").read_text())
            for name, doc in docs.items():
                prov = doc["synthetic_provenance"]
                self.assertEqual((prov["n_arr"], prov["n_dev"], prov["seed"]), (9, len(doc["dev_arrangements"]), 2))
                self.assertEqual(prov["generator"]["block_includes_start"], name == "default")
            self.assertNotEqual(docs["default"]["settings_sha256"], docs["anywhere"]["settings_sha256"])

    def test_lam_grid_check_in_candidate(self) -> None:
        res = sel.run_selection(small_dev(), sel.SelectionSettings(tau_match_grid=(0.5,), lam_grid=(0.0, 0.5),
                                                                   bootstrap_B=20), CTX)
        chk = res["lam_grid_check"]
        self.assertEqual(chk.classes, ((0.0, 0.5),))   # both below the first breakpoint 2/3
        self.assertTrue(chk.warnings)
        f = sel.run_selection(small_dev(), sel.SelectionSettings(policy="F", tau_match_grid=(0.5,), bootstrap_B=20),
                              CTX)
        self.assertIsNone(f["lam_grid_check"])

    def test_cli_requires_synthetic(self) -> None:
        self.assertEqual(sel.main([]), 2)

    def test_custom_devdata_with_hand_labels(self) -> None:
        # A hand-built two-arrangement dev set (the real-data path): every view clean, target visible
        # at stops != 2 when present. Selection must run and never touch anything but this data.
        recs, specs, states, labels = {}, [], {}, []
        for a in ("A01", "A02"):
            for cond in (Condition.VISIBLE, Condition.REMOVED):
                s = EpisodeSpec(a, cond, Lighting.L1, "t", Split.DEV)
                specs.append(s)
                states[s.episode_id] = "removed" if cond is Condition.REMOVED else "present"
                m = 0.05 if cond is Condition.REMOVED else 0.9
                recs[s.episode_id] = [EvidenceRecord(st, m, 0.9, 0.05) for st in range(5)]
                labels += [sel.ViewLabel(s.episode_id, st, True) for st in range(5)]
        dev = sel.DevData(tuple(specs), sel.RecordStoreFactory(recs, 5, synthetic=False), GroundTruth(states),
                          tuple(labels), POS, False, "hand labels (test)")
        res = sel.run_selection(dev, sel.SelectionSettings(tau_match_grid=(0.5,), lam_grid=(0.5,), bootstrap_B=20),
                                RunContext(synthetic=False))
        c = res["final"].chosen_counts
        self.assertEqual((c.n_false_removal, c.n_true_removal, c.n_resolved), (0, 2, 4))
        # coverage and recall tie between k = 3 and k = 2; k = 2 needs one fewer move on removed -> less travel
        self.assertEqual(res["final"].chosen.k, 2)
