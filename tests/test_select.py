"""Tests for verify/select.py (D-select, procedure select-v4). Synthetic data exercises logic only; no value here
is a threshold."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import itertools
import json
import random
import tempfile
import unittest
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from unittest import mock

import verify.policies as pol
from analysis.metrics import compute
from verify import select as sel
from verify.schema import (
    SYNTHETIC_LABEL,
    Condition,
    Decision,
    EpisodeSpec,
    EvidenceRecord,
    GroundTruth,
    Lighting,
    Observation,
    Positions,
    RunContext,
    Split,
    Thresholds,
    config_sha256,
)
from verify.stopping import StoppingRule
from verify.synthetic import SyntheticParams, generate, generate_layouts

POS = Positions.default()
CTX = RunContext(synthetic=True)
# Fast settings: uncapped plus one cap, two R seeds, a short bootstrap, one lam sensitivity value.
FAST = sel.SelectionSettings(cap_steps=(2,), r_seeds=2, bootstrap_B=40, lam_sensitivity=(0.7,))
LIGHTS = ("L1", "L2")
# For hand_dev(): one tau_match, uncapped only, one R seed, no sensitivity runs.
HAND = sel.SelectionSettings(tau_match_grid=(0.5,), cap_steps=(), r_seeds=1, bootstrap_B=20, k_sensitivity=(),
                             lam_sensitivity=())
REAL = RunContext(synthetic=False)


def small_dev(n_arr: int = 9, seed: int = 0, **params) -> sel.DevData:
    return sel.synthetic_dev_data(n_arr, seed, SyntheticParams(**params), POS)


def hand_dev(flip: tuple[str, int] | None = None, **fields) -> sel.DevData:
    """A hand-built two-arrangement dev set (the real-data path), L1 only. An occluder blocks P2 and P3 in the occluded
    and removed episodes (obstruction 0.9, non-diagnostic); every other view is clean and diagnostic. flip inverts the
    label of one (episode_id, stop), on the same evidence; fields replace DevData arguments."""
    recs, specs, states, labels = {}, [], {}, []
    for a in ("A01", "A02"):
        for cond in (Condition.VISIBLE, Condition.OCCLUDED, Condition.REMOVED):
            s = EpisodeSpec(a, cond, Lighting.L1, "t", Split.DEV)
            specs.append(s)
            states[s.episode_id] = "removed" if cond is Condition.REMOVED else "present"
            blocked = set() if cond is Condition.VISIBLE else {1, 2}
            m = 0.05 if cond is Condition.REMOVED else 0.9
            recs[s.episode_id] = [EvidenceRecord(st, 0.1 if st in blocked and m > 0.5 else m, 0.9,
                                                 0.9 if st in blocked else 0.05) for st in range(5)]
            labels += [sel.ViewLabel(s.episode_id, st, (st not in blocked) != ((s.episode_id, st) == flip))
                       for st in range(5)]
    factory = sel.RecordStoreFactory(recs, 5, synthetic=False)
    args = dict(specs=tuple(specs), store_factory=factory, truth=GroundTruth(states), view_labels=tuple(labels),
                positions=POS, synthetic=False, label_source="hand labels (test)",
                evidence_sha256=factory.evidence_sha256())
    return sel.DevData(**{**args, **fields})


def vrow(cond: Condition, diag: bool, m: float, q: float = 0.9, o: float = 0.05,
         light: Lighting = Lighting.L1, arr: str = "A01", stop: int = 0) -> sel.ViewRow:
    obs = Observation(step=0, stop=stop, azimuth_deg=0.0, match_score=m, quality=q, obstruction=o)
    return sel.ViewRow(arr, f"{arr}-{cond.value}-{light.value}", cond, light, diag, obs)


Wrap = Callable[[Callable], Callable]


@contextlib.contextmanager
def patched_runs(wrap: Wrap) -> Iterator[None]:
    """Wrap run_episode where select calls it and where run_r_seeds (R) calls it."""
    real = pol.run_episode
    with mock.patch.object(sel, "run_episode", wrap(real)), mock.patch.object(pol, "run_episode", wrap(real)):
        yield


def flip_to_removed(real: Callable) -> Callable:
    """A run_episode that reports REMOVED whatever the evidence said."""
    def f(policy, env, rule, **kw):
        return dataclasses.replace(real(policy, env, rule, **kw), decision=Decision.REMOVED)
    return f


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

    def test_run_selection_refuses_a_test_episode_itself(self) -> None:
        dev = small_dev()
        test_spec = dataclasses.replace(dev.specs[0], split=Split.TEST)
        object.__setattr__(dev, "specs", (*dev.specs[1:], test_spec))   # smuggled past DevData's own check
        with self.assertRaises(sel.SplitRefused):
            sel.run_selection(dev, FAST, CTX)

    def test_synthetic_dev_data_is_dev_only(self) -> None:
        dev = small_dev()
        self.assertTrue(all(s.split is Split.DEV for s in dev.specs))
        self.assertEqual(len(dev.arrangements), 3)
        self.assertEqual(len(dev.specs), 18)
        self.assertEqual(set(dev.store_factory().episode_ids()), {s.episode_id for s in dev.specs})

    def test_devdata_refuses_a_synthetic_flag_that_disagrees_with_its_data(self) -> None:
        # Review: synthetic evidence with synthetic=False was accepted, and its candidate and report had no SYNTHETIC
        # label at all. The flag must agree with the evidence store and with the truth, in both directions.
        syn, real = small_dev(), hand_dev()
        with self.assertRaisesRegex(ValueError, "the evidence store says synthetic=True; the ground truth says "
                                                "synthetic=True"):
            sel.DevData(syn.specs, syn.store_factory, syn.truth, syn.view_labels, POS, False, "real labels (claimed)",
                        syn.evidence_sha256)
        with self.assertRaisesRegex(ValueError, "the evidence store says synthetic=False; the ground truth says "
                                                "synthetic=False"):
            hand_dev(synthetic=True)
        real_truth = GroundTruth(syn.truth.states, syn.truth.indistinguishable)   # synthetic store, real truth
        with self.assertRaisesRegex(ValueError, r"disagrees with its data: the ground truth says synthetic=False$"):
            sel.DevData(syn.specs, syn.store_factory, real_truth, syn.view_labels, POS, True, syn.label_source,
                        syn.evidence_sha256)
        self.assertFalse(real.synthetic)   # agreeing flags are accepted either way

    def test_real_devdata_needs_its_evidence_sha256(self) -> None:
        with self.assertRaisesRegex(ValueError, "need evidence_sha256"):
            hand_dev(evidence_sha256="")
        dev = small_dev()   # synthetic data may leave it empty (synthetic_dev_data fills it in)
        self.assertTrue(sel.DevData(dev.specs, dev.store_factory, dev.truth, dev.view_labels, POS, True, "x").synthetic)


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

    def test_runner_uses_one_fresh_store_per_setting(self) -> None:
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
        runner = sel.PolicyRunner(dev2, FAST, CTX)
        th = sel.frozen_thresholds(0.6, 0.5, 0.3)
        b3 = sel.run_budget("G", None, FAST, POS)
        runner.results("G", th, b3)
        runner.results("G", th, b3)       # cached: no new store
        runner.results("F", th, b3)
        self.assertEqual(len(calls) - n0, 2)
        self.assertEqual((runner.n_runs, runner.n_settings), (2 * len(dev.specs), 2))


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

    def test_view_labels_hash(self) -> None:
        # Review: the labels set the thresholds as much as the evidence, but only the evidence was hashed. The labels
        # hash is the canonical JSON of the rows in (episode_id, stop) order: the "labels" list of a saved file.
        dev = hand_dev()
        other = hand_dev(flip=("A01-occluded-L1", 0))   # the same evidence, one label changed
        self.assertEqual(dev.evidence_sha256, other.evidence_sha256)
        self.assertNotEqual(dev.view_labels_sha256, other.view_labels_sha256)
        self.assertEqual(sel.view_labels_sha256(reversed(dev.view_labels)), dev.view_labels_sha256)   # order-free
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.json"
            sel.save_view_labels(reversed(dev.view_labels), p, source="test")
            self.assertEqual(config_sha256(json.loads(p.read_text())["labels"]), dev.view_labels_sha256)
        self.assertEqual(len(dev.view_labels_sha256), 64)

    def test_view_rows_carry_lighting_and_the_rules_observation(self) -> None:
        dev = small_dev()
        rows = sel.view_rows(dev)
        self.assertEqual(len(rows), 5 * len(dev.specs))
        store = dev.store_factory()
        for r in rows[:10]:
            rec = store.get(r.episode_id, r.obs.stop)
            self.assertEqual((r.obs.match_score, r.obs.quality, r.obs.obstruction),
                             (rec.match_score, rec.quality, rec.obstruction))
            self.assertEqual(r.lighting.value, r.episode_id.rsplit("-", 1)[1])
        self.assertEqual({r.lighting for r in rows}, set(Lighting))


def brute_force(rows: list[sel.ViewRow], tq_grid, to_grid, tm_grid, alpha: float):
    """An independent enumeration of the joint search (no ViewTable, no GridPoint): every (tau_q, tau_occ, tau_match)
    with fresh StoppingRule predicates, exact Fraction comparisons against alpha, and the declared order written out
    again: max correct clear misses, then lower false clear-miss rate, lower false-clear rate, lower tau_match, higher
    tau_q, lower tau_occ. Returns (best setting or None, number of feasible settings)."""
    null_g = [r for r in rows if r.condition is Condition.OCCLUDED and not r.diagnostic]
    pres = [r for r in rows if r.condition is not Condition.REMOVED and r.diagnostic]
    rem = [r for r in rows if r.condition is Condition.REMOVED and r.diagnostic]
    feas = []
    for tq, to, tm in itertools.product(tq_grid, to_grid, tm_grid):
        rule = StoppingRule(Thresholds(tau_match=tm, tau_q=tq, tau_occ=to))
        fc = sum(rule.passes_gate(r.obs) for r in null_g)
        fcm = sum(rule.is_clear_miss(r.obs) for r in pres)
        cm = sum(rule.is_clear_miss(r.obs) for r in rem)
        if null_g and pres and Fraction(fc, len(null_g)) <= Fraction(alpha).limit_denominator(10 ** 6) and \
                Fraction(fcm, len(pres)) <= Fraction(alpha).limit_denominator(10 ** 6):
            feas.append(((-cm, Fraction(fcm, len(pres)), Fraction(fc, len(null_g)), tm, -tq, to), (tq, to, tm)))
    if not feas:
        return None, 0
    return min(feas)[1], len(feas)


def random_rows(rng: random.Random, n_arr: int, values: tuple[float, ...]) -> list[sel.ViewRow]:
    """Random views whose scores sit ON the grid values half the time (the boundaries are inclusive)."""
    def score() -> float:
        return rng.choice(values) if rng.random() < 0.5 else round(rng.random(), 3)
    rows = []
    for a in range(n_arr):
        for cond in Condition:
            for light in Lighting:
                for st in range(rng.randint(1, 4)):
                    rows.append(vrow(cond, rng.random() < 0.6, score(), score(), score(), light, f"A{a:02d}", st))
    return rows


def counterexample_rows() -> list[sel.ViewRow]:
    """The sequential counterexample (runs/2026-11-03-pr4-review/select1/probe_sequential_infeasible.py, copied):
    hand-built scores modelling L2 queries against the L1 reference, dim (quality 0.55) and poorly matched (0.30)."""
    rows = []
    for a in range(10):
        arr = f"A{a:02d}"
        for light, q in ((Lighting.L1, 0.9), (Lighting.L2, 0.55)):
            m_present = 0.9 if light is Lighting.L1 else 0.30
            for st in range(5):
                blocked = st in (1, 2)
                rows.append(vrow(Condition.VISIBLE, True, m_present, q, 0.05, light, arr, st))
                rows.append(vrow(Condition.OCCLUDED, not blocked, 0.1 if blocked else m_present, q,
                                 0.9 if blocked else 0.05, light, arr, st))
                rows.append(vrow(Condition.REMOVED, not blocked, 0.05, q, 0.9 if blocked else 0.05, light, arr, st))
    return rows


class JointSearchTest(unittest.TestCase):
    """Step 1: one joint search over tau_q x tau_occ x tau_match (revision 2026-11-03)."""

    def test_matches_brute_force_on_random_cases(self) -> None:
        rng = random.Random(20261103)
        n_feasible = n_infeasible = 0
        for case in range(150):
            tq = tuple(sorted(rng.sample((0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), rng.randint(1, 4))))
            to = tuple(sorted(rng.sample((0.1, 0.2, 0.3, 0.4, 0.5), rng.randint(1, 3))))
            tm = tuple(sorted(rng.sample((0.3, 0.4, 0.5, 0.6, 0.7), rng.randint(1, 4))))
            alpha = rng.choice((0.0, 0.1, 0.2, 0.25, 1 / 3, 0.5))
            rows = random_rows(rng, rng.randint(1, 4), tq + to + tm)
            s = sel.SelectionSettings(tau_q_grid=tq, tau_occ_grid=to, tau_match_grid=tm, alpha_clear=alpha)
            want, n_want = brute_force(rows, tq, to, tm, alpha)
            with self.subTest(case=case):
                if want is None:
                    n_infeasible += 1
                    with self.assertRaises(sel.CalibrationInfeasible):
                        sel.calibrate(rows, s, log_warnings=False)
                    continue
                n_feasible += 1
                cal = sel.calibrate(rows, s, log_warnings=False)
                th = cal.thresholds
                self.assertEqual((th.tau_q, th.tau_occ, th.tau_match), want)
                self.assertEqual(cal.choice.n_feasible, n_want)
                self.assertEqual(cal.choice.n_grid, len(tq) * len(to) * len(tm))
        self.assertGreater(n_feasible, 30)    # both outcomes are exercised
        self.assertGreater(n_infeasible, 10)

    def test_grid_order_never_breaks_a_tie(self) -> None:
        # Permuting the grid points (and the rows) gives the same choice: the order is total, not iteration order.
        rows = known_rows() + counterexample_rows()[:60]
        s = sel.SelectionSettings(tau_q_grid=(0.4, 0.5, 0.6), tau_occ_grid=(0.1, 0.3), tau_match_grid=KNOWN_GRID,
                                  alpha_clear=0.3)
        table = sel.ViewTable(rows, s)
        pts = table.points()
        ref = sel.choose_joint(pts, s.alpha_clear, table.grid)
        self.assertGreater(ref.n_tied_at_max, 1)   # a tie the declared order must break
        rng = random.Random(7)
        for _ in range(25):
            perm = pts[:]
            rng.shuffle(perm)
            got = sel.choose_joint(perm, s.alpha_clear, table.grid)
            self.assertEqual((got.chosen, got.n_tied_at_max, got.decided_by),
                             (ref.chosen, ref.n_tied_at_max, ref.decided_by))
            shuffled = rows[:]
            rng.shuffle(shuffled)
            self.assertEqual(sel.calibrate(shuffled, s, log_warnings=False).thresholds,
                             sel.calibrate(rows, s, log_warnings=False).thresholds)

    def test_each_tie_level_can_decide(self) -> None:
        # Hand-built: two removed views (one clear miss each wherever the gate passes), tie pressure by construction.
        rem = [vrow(Condition.REMOVED, True, 0.1, 0.9, 0.05), vrow(Condition.REMOVED, True, 0.1, 0.9, 0.05)]
        occ_null = [vrow(Condition.OCCLUDED, False, 0.1, 0.9, 0.25)]   # passes the gate iff tau_occ >= 0.25
        cases = [
            # (present diagnostic views, tau_q grid, tau_occ grid, tau_match grid, alpha, want, decided_by)
            # a present view matching 0.45: a false clear miss at tau_match 0.5, not at 0.4 -> lower false clear-miss
            ([vrow(Condition.VISIBLE, True, 0.45)], (0.5,), (0.2,), (0.4, 0.5), 1.0, (0.5, 0.2, 0.4),
             "false clear-miss rate"),
            # tau_occ 0.3 lets the null view through (false clear 1/1) -> lower false-clear rate wins
            ([vrow(Condition.VISIBLE, True, 0.9)], (0.5,), (0.2, 0.3), (0.4,), 1.0, (0.5, 0.2, 0.4),
             "false-clear rate"),
            ([vrow(Condition.VISIBLE, True, 0.9)], (0.5,), (0.2,), (0.4, 0.5, 0.6), 1.0, (0.5, 0.2, 0.4),
             "tau_match (lower)"),
            ([vrow(Condition.VISIBLE, True, 0.9)], (0.4, 0.5), (0.2,), (0.4,), 1.0, (0.5, 0.2, 0.4), "tau_q (higher)"),
            ([vrow(Condition.VISIBLE, True, 0.9)], (0.5,), (0.1, 0.2), (0.4,), 1.0, (0.5, 0.1, 0.4),
             "tau_occ (lower)"),
        ]
        for present, tq, to, tm, alpha, want, level in cases:
            with self.subTest(level=level):
                rows = rem + occ_null + present
                s = sel.SelectionSettings(tau_q_grid=tq, tau_occ_grid=to, tau_match_grid=tm, alpha_clear=alpha)
                cal = sel.calibrate(rows, s, log_warnings=False)
                th = cal.thresholds
                self.assertEqual((th.tau_q, th.tau_occ, th.tau_match), want)
                self.assertEqual(cal.choice.decided_by, level)
                self.assertTrue(cal.choice.tie_break_decided)
                self.assertEqual((th.tau_q, th.tau_occ, th.tau_match), brute_force(rows, tq, to, tm, alpha)[0])
        # a unique maximum: the objective alone decides
        rows = rem[:1] + [vrow(Condition.REMOVED, True, 0.1, 0.45, 0.05)] + occ_null + [vrow(Condition.VISIBLE, True, 0.9)]
        cal = sel.calibrate(rows, sel.SelectionSettings(tau_q_grid=(0.4, 0.5), tau_occ_grid=(0.2,),
                                                        tau_match_grid=(0.4,), alpha_clear=1.0), log_warnings=False)
        self.assertEqual((cal.thresholds.tau_q, cal.choice.decided_by, cal.choice.n_tied_at_max), (0.4, None, 1))

    def test_sequential_counterexample_is_feasible_and_shows_the_l2_rejection(self) -> None:
        rows = counterexample_rows()
        s = sel.SelectionSettings()   # the predeclared default grids and alpha_clear 0.10
        cal = sel.calibrate(rows, s, log_warnings=False)
        want, n = brute_force(rows, s.tau_q_grid, s.tau_occ_grid, s.tau_match_grid, s.alpha_clear)
        th = cal.thresholds
        self.assertEqual((th.tau_q, th.tau_occ, th.tau_match), want)
        self.assertEqual(want, (0.8, 0.1, 0.4))
        self.assertEqual((cal.choice.n_feasible, n, cal.choice.n_grid), (315, 315, 693))   # as Codex counted
        c = cal.choice.chosen
        self.assertEqual((c.false_clear, c.false_clear_miss, c.clear_miss), ((0, 40), (0, 160), (30, 60)))
        gp = sel.gate_pass_by_lighting(rows, th)
        self.assertEqual((gp["L1"]["removed_diagnostic"], gp["L2"]["removed_diagnostic"]), ((30, 30), (0, 30)))
        self.assertEqual((gp["L1"]["all"], gp["L2"]["all"]), ((110, 150), (0, 150)))   # as in audit.json
        flags = sel.gate_pass_flags(gp, "here")
        self.assertIn("here: the gate rejects ALL L2 removed diagnostic views (0/30 pass tau_q / tau_occ)", flags)
        self.assertIn("here: the gate rejects ALL L2 present diagnostic views (0/80 pass tau_q / tau_occ)", flags)
        self.assertFalse([f for f in flags if " L1 " in f])
        self.assertIn("here: the gate rejects ALL L2 views (all) (0/150 pass tau_q / tau_occ)", flags)
        # and every feasible setting rejects every L2 view at the gate (the reason the diagnostic exists)
        for p in cal.choice.points:
            if p.feasible(s.alpha_clear):
                self.assertFalse(StoppingRule(Thresholds(tau_q=p.tau_q, tau_occ=p.tau_occ)).passes_gate(
                    Observation(0, 0, 0.0, 0.3, 0.55, 0.05)))

    def test_gate_pass_flags_cover_all_views_and_empty_populations(self) -> None:
        # Review (checkpoint 2), section 7 lists three populations: removed diagnostic, present diagnostic and all
        # views. An L2 all-views share of 3/120 = 0.025 was not flagged (only the diagnostic populations were), and a
        # population with no view was skipped without a word.
        gp = {"L1": {"removed_diagnostic": (10, 12), "present_diagnostic": (0, 0), "all": (90, 120)},
              "L2": {"removed_diagnostic": (1, 12), "present_diagnostic": (20, 30), "all": (3, 120)},
              "pooled": {"removed_diagnostic": (11, 24), "present_diagnostic": (20, 30), "all": (93, 240)}}
        flags = sel.gate_pass_flags(gp, "here")
        self.assertEqual(flags, [
            "here: no views: there are no L1 present diagnostic views (n = 0), so their gate pass share is undefined "
            "there",
            "here: the gate passes nearly no L2 views (all) (3/120 = 0.025 <= 0.05)"])

    def test_gate_pass_flag_boundary_is_inclusive_for_every_population(self) -> None:
        # The rule is share <= 0.05 (GATE_PASS_LOW): exactly 1/20 is flagged, 1/19 = 0.0526 is not, and the all-views
        # population is held to the same boundary as the two diagnostic ones. A strict < slipped through unnoticed.
        gp = {"L1": {"removed_diagnostic": (1, 20), "present_diagnostic": (1, 20), "all": (1, 20)},
              "L2": {"removed_diagnostic": (1, 19), "present_diagnostic": (1, 19), "all": (1, 19)}}
        flags = sel.gate_pass_flags(gp, "here")
        self.assertEqual(flags, [
            "here: the gate passes nearly no L1 removed diagnostic views (1/20 = 0.050 <= 0.05)",
            "here: the gate passes nearly no L1 present diagnostic views (1/20 = 0.050 <= 0.05)",
            "here: the gate passes nearly no L1 views (all) (1/20 = 0.050 <= 0.05)"])
        self.assertFalse([f for f in flags if " L2 " in f])

    def test_known_tau_match_answers_under_one_gate(self) -> None:
        # select-v4's old Step 1b known answers still hold when the grid has one gate (the joint search reduces to it)
        for alpha, want in ((0.10, 0.40), (0.20, 0.45), (0.35, 0.50), (0.50, 0.55), (1.0, 0.65)):
            with self.subTest(alpha=alpha):
                s = sel.SelectionSettings(tau_q_grid=(0.5,), tau_occ_grid=(0.3,), tau_match_grid=KNOWN_GRID,
                                          alpha_clear=alpha)
                cal = sel.calibrate(known_rows(), s, log_warnings=False)
                self.assertEqual(cal.thresholds.tau_match, want)
        curve = sel.threshold_curves(sel.calibrate(known_rows(), sel.SelectionSettings(
            tau_q_grid=(0.5,), tau_occ_grid=(0.3,), tau_match_grid=KNOWN_GRID), log_warnings=False).choice)
        tm = curve["tau_match"]["points"]
        self.assertEqual([p["false_clear_miss"] for p in tm], [[i, 10] for i in range(1, 8)])
        self.assertEqual([p["correct_clear_miss"][0] for p in tm], [5, 6, 7, 8, 8, 9, 9])
        self.assertEqual([p["chosen"] for p in tm], [True] + [False] * 6)
        self.assertFalse(curve["tau_match"]["correct_clear_miss_flat"])
        self.assertTrue(curve["tau_q"]["correct_clear_miss_flat"])   # a one-point grid is flat: reported, not forced

    def test_curves_hold_the_other_two_at_the_chosen_values(self) -> None:
        rows = counterexample_rows()
        cal = sel.calibrate(rows, sel.SelectionSettings(), log_warnings=False)
        curves = sel.threshold_curves(cal.choice)
        th = cal.thresholds
        for axis, grid in (("tau_q", sel.DEFAULT_TAU_Q_GRID), ("tau_occ", sel.DEFAULT_TAU_OCC_GRID),
                           ("tau_match", sel.DEFAULT_TAU_MATCH_GRID)):
            c = curves[axis]
            self.assertEqual([p["value"] for p in c["points"]], list(grid))
            self.assertEqual(sum(p["chosen"] for p in c["points"]), 1)
            for p in c["points"]:
                setting = {"tau_q": th.tau_q, "tau_occ": th.tau_occ, "tau_match": th.tau_match, axis: p["value"]}
                rule = StoppingRule(Thresholds(**setting))
                rem = sel.removed_diagnostic_views(rows)
                pres = sel.present_diagnostic_views(rows)
                self.assertEqual(p["correct_clear_miss"], [sum(rule.is_clear_miss(r.obs) for r in rem), len(rem)])
                self.assertEqual(p["false_clear_miss"], [sum(rule.is_clear_miss(r.obs) for r in pres), len(pres)])
        # tau_match does not move the clear-miss count here (every L1 removed view matches 0.05): flat, kept flat
        self.assertTrue(curves["tau_match"]["correct_clear_miss_flat"])
        self.assertTrue(curves["tau_match"]["tie_break_decided_on_curve"])

    def test_table_on_a_subset_equals_a_recount(self) -> None:
        rows = sel.view_rows(small_dev(15))
        s = sel.SelectionSettings(tau_q_grid=(0.4, 0.6), tau_occ_grid=(0.2, 0.4), tau_match_grid=(0.5, 0.6))
        table = sel.ViewTable(rows, s)
        keep = table.arrangements[1:4]
        self.assertEqual(table.points(keep), sel.ViewTable([r for r in rows if r.arrangement_id in keep], s).points())

    def test_calibration_calls_the_rules_methods(self) -> None:
        names = ("is_clear_miss", "is_strong_match", "passes_gate")
        calls = dict.fromkeys(names, 0)
        orig = {n: getattr(StoppingRule, n) for n in names}

        def spy(name: str) -> Callable:
            def f(rule, o):
                calls[name] += 1
                return orig[name](rule, o)
            return f
        settings = sel.SelectionSettings(tau_match_grid=KNOWN_GRID, tau_q_grid=(0.5,), tau_occ_grid=(0.3,))
        with mock.patch.object(StoppingRule, "is_clear_miss", spy("is_clear_miss")), \
                mock.patch.object(StoppingRule, "is_strong_match", spy("is_strong_match")), \
                mock.patch.object(StoppingRule, "passes_gate", spy("passes_gate")):
            cal = sel.calibrate(known_rows(), settings)
        self.assertEqual(calls["is_clear_miss"], len(KNOWN_GRID) * 20)   # 10 present + 10 removed diagnostic views
        self.assertEqual(calls["is_strong_match"], len(KNOWN_GRID) * 10)
        self.assertEqual(calls["passes_gate"], 11 + len(KNOWN_GRID) * 20)   # 1 null + 10 removed, then via clear miss
        self.assertEqual(cal.thresholds, Thresholds(tau_match=0.40, tau_q=0.5, tau_occ=0.3, k=2, lam=0.0))

    def test_no_power_view_warns_and_takes_the_most_conservative_setting(self) -> None:
        rows = [vrow(Condition.OCCLUDED, False, 0.1, 0.9, 0.9), vrow(Condition.VISIBLE, True, 0.9)]
        settings = sel.SelectionSettings(tau_q_grid=(0.4, 0.5), tau_occ_grid=(0.1, 0.2))
        with self.assertLogs("verify.select", level="WARNING"):
            cal = sel.calibrate(rows, settings)
        self.assertEqual((cal.thresholds.tau_q, cal.thresholds.tau_occ, cal.thresholds.tau_match), (0.5, 0.1, 0.40))
        self.assertTrue(any(w.startswith("Step 1: no diagnostic removed-episode views") for w in cal.warnings))


# Known answers along tau_match (gate tau_q 0.5, tau_occ 0.3; every diagnostic view below passes it).
KNOWN_PRESENT = (0.35, 0.42, 0.48, 0.52, 0.58, 0.62, 0.66, 0.72, 0.78, 0.85)   # diagnostic views, target present
KNOWN_REMOVED = (0.05, 0.10, 0.20, 0.30, 0.38, 0.44, 0.47, 0.53, 0.61, 0.70)   # diagnostic views, target removed
KNOWN_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def known_rows(extra: tuple[sel.ViewRow, ...] = ()) -> list[sel.ViewRow]:
    rows = [vrow(Condition.VISIBLE if i % 2 else Condition.OCCLUDED, True, m, stop=i % 5)
            for i, m in enumerate(KNOWN_PRESENT)]
    rows += [vrow(Condition.REMOVED, True, m, stop=i % 5) for i, m in enumerate(KNOWN_REMOVED)]
    # non-diagnostic views are in neither null set of tau_match; the occluded one (flagged) is the false-clear null
    rows += [vrow(Condition.REMOVED, False, 0.9), vrow(Condition.OCCLUDED, False, 0.0, o=0.9)]
    return rows + list(extra)


class InfeasibleTest(unittest.TestCase):
    """No grid setting meets both constraints: Step 1 fails loudly, with no fallback and no candidate."""

    def infeasible(self, rows, tq, to, tm, alpha=0.10) -> sel.Infeasibility:
        s = sel.SelectionSettings(tau_q_grid=tq, tau_occ_grid=to, tau_match_grid=tm, alpha_clear=alpha)
        self.assertIsNone(brute_force(rows, tq, to, tm, alpha)[0])
        with self.assertRaises(sel.CalibrationInfeasible) as cm:
            sel.calibrate(rows, s, log_warnings=False)
        return cm.exception.failure

    def test_binding_constraint_and_best_rates(self) -> None:
        occ, rem, vis = Condition.OCCLUDED, Condition.REMOVED, Condition.VISIBLE
        nulls = [vrow(occ, False, 0.1, 0.55, 0.05), vrow(occ, False, 0.1, 0.9, 0.05)]   # pass at tau_q 0.5; 1 at 0.6
        # false-clear binds: every gate passes a null view; the present view never clear-misses (match 0.9)
        f = self.infeasible(nulls + [vrow(rem, True, 0.1), vrow(vis, True, 0.9)], (0.5, 0.6), (0.1, 0.2), (0.5,))
        self.assertEqual(f.binding, "false-clear")
        self.assertEqual((f.best_false_clear["rate"], f.best_false_clear["x"], f.best_false_clear["n"]), (0.5, 1, 2))
        self.assertEqual(f.best_false_clear["settings"], [{"tau_q": 0.6, "tau_occ": 0.1, "tau_match": 0.5},
                                                          {"tau_q": 0.6, "tau_occ": 0.2, "tau_match": 0.5}])
        self.assertEqual(f.best_false_clear_miss["rate"], 0.0)
        self.assertIsNone(f.best_false_clear_miss_meeting_false_clear)
        self.assertEqual(len(f.grid_rates), 4)
        msg = str(f.message())
        for needle in ("Step 1 is infeasible: no setting on the predeclared grid meets both", "Binding: the "
                       "false-clear constraint", "0.5000 (1/2)", "Grid searched (jointly)", sel.NO_FALLBACK):
            self.assertIn(needle, msg)
        # false clear-miss binds: present view with match 0.3 clear-misses under every passing gate; no null view passes
        f = self.infeasible([vrow(occ, False, 0.1, 0.9, 0.9), vrow(rem, True, 0.1), vrow(vis, True, 0.3)],
                      (0.5,), (0.3,), (0.4, 0.5))
        self.assertEqual((f.binding, f.best_false_clear_miss["rate"]), ("false clear-miss", 1.0))
        # both bind
        f = self.infeasible(nulls + [vrow(vis, True, 0.3)], (0.5,), (0.3,), (0.5,))
        self.assertEqual(f.binding, "both")

    def test_the_most_conservative_corner_minimises_both_false_rates(self) -> None:
        # The gate is monotone (a higher tau_q or lower tau_occ rejects more) and a lower tau_match makes fewer clear
        # misses, so (max tau_q, min tau_occ, min tau_match) has the lowest false-clear AND false clear-miss rates on
        # the grid: if each constraint is met somewhere, both are met there. "jointly" is kept as a defensive label;
        # on random cases it never occurs.
        rng = random.Random(3)
        seen = set()
        for _ in range(150):
            tq = tuple(sorted(rng.sample((0.2, 0.4, 0.6, 0.8), rng.randint(1, 3))))
            to = tuple(sorted(rng.sample((0.1, 0.3, 0.5), rng.randint(1, 3))))
            tm = tuple(sorted(rng.sample((0.3, 0.5, 0.7), rng.randint(1, 3))))
            rows = random_rows(rng, 2, tq + to + tm)
            s = sel.SelectionSettings(tau_q_grid=tq, tau_occ_grid=to, tau_match_grid=tm, alpha_clear=0.1)
            pts = sel.ViewTable(rows, s).points()
            corner = next(p for p in pts if (p.tau_q, p.tau_occ, p.tau_match) == (tq[-1], to[0], tm[0]))
            self.assertEqual(corner.false_clear[0], min(p.false_clear[0] for p in pts))
            self.assertEqual(corner.false_clear_miss[0], min(p.false_clear_miss[0] for p in pts))
            try:
                sel.calibrate(rows, s, log_warnings=False)
            except sel.CalibrationInfeasible as e:
                seen.add(e.failure.binding)
        self.assertNotIn("jointly", seen)
        self.assertTrue(seen)

    def test_no_view_to_measure_a_constraint_on_is_infeasible(self) -> None:
        f = self.infeasible([vrow(Condition.REMOVED, True, 0.1), vrow(Condition.VISIBLE, True, 0.9)], (0.5,), (0.3,), (0.5,))
        self.assertEqual((f.n_gate_null_views, f.best_false_clear), (0, None))
        self.assertIn("no view to measure the false-clear rate on", f.binding_text)
        f = self.infeasible([vrow(Condition.REMOVED, True, 0.1), vrow(Condition.OCCLUDED, False, 0.1, o=0.9)],
                      (0.5,), (0.3,), (0.5,))
        self.assertEqual((f.n_match_null_views, f.binding), (0, "false clear-miss"))

    def test_run_selection_fails_before_any_fold_or_policy(self) -> None:
        def boom(*a, **kw):
            raise AssertionError("something ran after an infeasible calibration")
        with mock.patch.object(sel, "grouped_cv", boom), mock.patch.object(sel, "PolicyRunner", boom), \
                mock.patch.object(sel, "run_episode", boom), mock.patch.object(pol, "run_episode", boom), \
                self.assertRaises(sel.CalibrationInfeasible) as cm:
            sel.run_selection(small_dev(), dataclasses.replace(FAST, tau_q_grid=(0.0,), tau_occ_grid=(1.0,),
                                                               tau_match_grid=(0.95,)), CTX)
        self.assertEqual(cm.exception.failure.binding, "both")

    def test_cli_exits_3_and_leaves_no_candidate(self) -> None:
        base = ["--synthetic", "--n-arr", "9", "--seed", "0", "--bootstrap-B", "20", "--cap-steps", "none",
                "--r-seeds", "1", "--k-sensitivity", "none", "--lam-sensitivity", "none"]
        bad = ["--tau-q-grid", "0", "--tau-occ-grid", "1", "--tau-match-grid", "0.95"]   # the gate passes everything
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            out = Path(d) / "run"
            self.assertEqual(sel.main([*base, "--out", str(out)]), 0)   # a feasible candidate first
            self.assertTrue((out / "frozen_candidate.json").exists())
            rc = sel.main([*base, *bad, "--out", str(out)])
            gone = [n for n in sel.CANDIDATE_FILES if not (out / n).exists()]
            doc = json.loads((out / sel.INFEASIBLE_FILE).read_text())
            rep = (out / "selection_report.md").read_text()
            self.assertEqual(sel.main([*base, "--out", str(out)]), 0)   # a later feasible run clears the record
            cleared = not (out / sel.INFEASIBLE_FILE).exists()
        self.assertEqual(rc, sel.EXIT_INFEASIBLE)
        self.assertEqual(gone, list(sel.CANDIDATE_FILES))   # the earlier candidate cannot be picked up
        self.assertTrue(cleared)
        self.assertEqual((doc["status"], doc["schema"]), ("infeasible", sel.INFEASIBLE_SCHEMA))
        self.assertEqual(doc["procedure_revision"], sel.PROCEDURE_REVISION)
        self.assertNotIn("thresholds", doc)
        f = doc["failure"]
        self.assertEqual((f["binding"], f["grid"]["tau_match_grid"], f["alpha_clear"]), ("both", [0.95], 0.1))
        self.assertGreater(f["best_false_clear"]["rate"], 0.1)
        self.assertIn(sel.NO_FALLBACK, doc["no_candidate"])
        self.assertTrue(rep.startswith(f"# {SYNTHETIC_LABEL}"))
        for needle in ("INFEASIBLE", "No frozen candidate was written", "**binding**", "searched jointly",
                       "best achievable false-clear rate over the grid", f"labels sha256 `{doc['view_labels_sha256']}`"):
            self.assertIn(needle, rep)
        self.assertNotIn("Step 1b", rep)
        self.assertIn("CALIBRATION INFEASIBLE", err.getvalue())

    def test_infeasible_fold_is_recorded_without_a_fallback_and_blocks_freeze(self) -> None:
        rows: list[sel.ViewRow] = []
        for arr, lows, n in (("A01", 0, 20), ("A02", 3, 10), ("A03", 0, 10)):
            rows += [vrow(Condition.VISIBLE, True, 0.35 if i < lows else 0.9, arr=arr, stop=i % 5) for i in range(n)]
            rows += [vrow(Condition.REMOVED, True, 0.1, arr=arr, stop=st) for st in range(3)]
            rows.append(vrow(Condition.OCCLUDED, False, 0.1, o=0.9, arr=arr))
        settings = sel.SelectionSettings(tau_q_grid=(0.5,), tau_occ_grid=(0.3,), tau_match_grid=(0.4, 0.5))
        self.assertEqual(sel.calibrate(rows, settings, log_warnings=False).thresholds.tau_match, 0.4)   # 3/40 <= 0.10
        folds = {f.held_out: f for f in sel.grouped_cv(rows, settings, ["A01", "A02", "A03"])}
        bad = folds["A01"]   # without A01 the false clear-miss rate is 3/20 > 0.10 at every tau_match
        self.assertIsNone(bad.thresholds)
        self.assertIsNone(bad.held)
        self.assertEqual(bad.infeasible.binding, "false clear-miss")
        self.assertEqual(bad.infeasible.best_false_clear_miss["x"], 3)
        self.assertEqual(bad.infeasible.best_false_clear_miss["n"], 20)
        self.assertTrue(folds["A02"].thresholds and folds["A03"].thresholds)
        self.assertEqual(sel.pool_folds(list(folds.values()))["pooled"].false_clear_miss[1], 20)   # A02 + A03 only
        self.assertEqual(bad.to_dict()["infeasible"]["binding"], "false clear-miss")
        ok, reasons = sel.freeze_eligibility(list(folds.values()), ["A01", "A02", "A03"])
        self.assertFalse(ok)
        self.assertEqual([(r["fold"], r["constraint"]) for r in reasons], [("A01", "false clear-miss")])
        ok, reasons = sel.freeze_eligibility([], ["A01"])
        self.assertFalse(ok)
        self.assertIn("fewer than 2 dev arrangements", reasons[0]["reason"])
        self.assertEqual(sel.freeze_eligibility([folds["A02"], folds["A03"]], ["A02", "A03"]), (True, []))


class FixedKAndLamTest(unittest.TestCase):
    def test_k_and_lam_fixed_whatever_the_data(self) -> None:
        devs = [small_dev(9, 0), small_dev(12, 3, block_includes_start=False), small_dev(9, 5, indistinguishable_frac=0.0)]
        settings = [sel.SelectionSettings(), sel.SelectionSettings(alpha_clear=0.05),
                    sel.SelectionSettings(alpha_clear=1.0, tau_match_grid=(0.1, 0.9))]
        seen = set()
        for dev in devs:
            rows = sel.view_rows(dev)
            for s in settings:
                th = sel.calibrate(rows, s, log_warnings=False).thresholds
                seen.add((th.tau_q, th.tau_occ, th.tau_match))
                self.assertEqual((th.k, th.lam), (2, 0.0))
        self.assertGreater(len(seen), 1)   # the calibrated values do move with the data; k and lam never do
        self.assertEqual((sel.K_FIXED, sel.LAM_FIXED), (2, 0.0))

    def test_no_setting_or_option_selects_k_or_lam(self) -> None:
        names = {f.name for f in dataclasses.fields(sel.SelectionSettings)}
        self.assertFalse(names & {"k", "lam", "k_grid", "lam_grid", "policy", "recall_target"})
        for bad in (["--k-grid", "3,2"], ["--lam-grid", "0,0.7"], ["--policy", "G"], ["--recall-target", "0.8"]):
            with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sel.build_parser().parse_args(["--synthetic", *bad])
        with self.assertRaises(ValueError):
            sel.SelectionSettings(k_sensitivity=(2,))
        with self.assertRaises(ValueError):
            sel.SelectionSettings(lam_sensitivity=(0.0, 0.7))
        self.assertEqual(sel.SelectionSettings().to_dict()["fixed_a_priori"], {"k": 2, "lam": 0.0})
        self.assertEqual(sel.DEFAULT_LAM_GRID, (0.0, *sel.DEFAULT_LAM_SENSITIVITY))


class SettingsTest(unittest.TestCase):
    def test_settings_hash_does_not_depend_on_spelling(self) -> None:
        # Review: the default cap_steps were ints and --cap-steps 1,2,3,4,6 parsed to floats, so one setting had two
        # settings hashes (and two candidates). Every number has one type now.
        want = sel.settings_sha256(sel.SelectionSettings())
        args = sel.build_parser().parse_args(["--synthetic", "--cap-steps", "6,4,3,2,1,inf", "--k-sensitivity", "3",
                                              "--lam-sensitivity", "0.7,0.9,1.2", "--alpha-clear", "0.1"])
        spelled = {"parsed": sel.SelectionSettings(cap_steps=args.cap_steps, k_sensitivity=args.k_sensitivity,
                                                   lam_sensitivity=args.lam_sensitivity, alpha_clear=args.alpha_clear),
                   "ints": sel.SelectionSettings(cap_steps=(1, 2, 3, 4, 6), k_sensitivity=(3.0,)),
                   "floats": sel.SelectionSettings(cap_steps=(1.0, 2.0, 3.0, 4.0, 6.0))}
        for name, s in spelled.items():
            with self.subTest(spelling=name):
                self.assertEqual(sel.settings_sha256(s), want)
                self.assertTrue(all(type(c) is float for c in s.cap_steps))
                self.assertTrue(all(type(k) is int for k in s.k_sensitivity))
        self.assertEqual(sel.settings_sha256(sel.SelectionSettings(alpha_clear=0)),
                         sel.settings_sha256(sel.SelectionSettings(alpha_clear=0.0)))

    def test_explicit_default_caps_give_the_same_candidate(self) -> None:
        args = ["--synthetic", "--n-arr", "9", "--seed", "0", "--bootstrap-B", "20", "--tau-match-grid", "0.5",
                "--r-seeds", "1", "--k-sensitivity", "none", "--lam-sensitivity", "none"]
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            files = []
            for name, extra in (("default", []), ("explicit", ["--cap-steps", "1,2,3,4,6"]),
                                ("spelled", ["--cap-steps", "6,4.0,3,2,1,inf"])):
                out = Path(d) / name
                self.assertEqual(sel.main([*args, *extra, "--out", str(out)]), 0)
                files.append({n: (out / n).read_bytes() for n in (*sel.CANDIDATE_FILES, sel.REPORT_FILE)})
        self.assertEqual(files[1], files[0])
        self.assertEqual(files[2], files[0])

    def test_gate_grids_must_be_strictly_ascending(self) -> None:
        # Review: tau_q_grid / tau_occ_grid took duplicates, which were counted as extra grid gates ("6 of 6") and
        # listed repeatedly in an infeasible record, while tau_match_grid was already checked.
        for n in ("tau_q_grid", "tau_occ_grid", "tau_match_grid"):
            for bad in ((0.3, 0.3), (0.4, 0.3), (0.2, 0.3, 0.3)):
                with self.subTest(grid=n, values=bad), \
                        self.assertRaisesRegex(ValueError, f"{n} must be strictly ascending, without duplicates"):
                    sel.SelectionSettings(**{n: bad})

    def test_invalid_settings_are_usage_errors(self) -> None:
        # Review: these crashed with a ValueError traceback (exit 1), and argparse called a bad k list "invalid f value"
        cases = {("--lam-sensitivity", "0"): "lam = 0 is fixed a priori",
                 ("--k-sensitivity", "2"): "k = 2 is fixed a priori",
                 ("--budget", "0"): "budget must be 1..4",
                 ("--lam-sensitivity", "1.2,0.7"): "lam_sensitivity must be strictly ascending",
                 ("--tau-q-grid", "0.6,0.6,0.6"): "tau_q_grid must be strictly ascending",
                 ("--tau-occ-grid", "0.3,0.2"): "tau_occ_grid must be strictly ascending",
                 ("--k-sensitivity", "3.5"): "argument --k-sensitivity: invalid int list value: '3.5'",
                 ("--lam-sensitivity", "x"): "argument --lam-sensitivity: invalid float list value: 'x'"}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "never"
            for extra, needle in cases.items():
                with self.subTest(args=extra):
                    with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as cm:
                        sel.main(["--synthetic", "--n-arr", "6", *extra, "--out", str(out)])
                    self.assertEqual(cm.exception.code, 2)
                    self.assertIn("verify.select: error: ", err.getvalue())
                    self.assertIn(needle, err.getvalue())
                    self.assertNotIn("invalid f value", err.getvalue())
            self.assertFalse(out.exists())   # a usage error stops before anything runs or is written


class FreezeBeforeRunsTest(unittest.TestCase):
    """No policy runs before the thresholds are fixed, and no policy result reaches them."""

    def test_no_policy_runs_before_the_thresholds_are_frozen(self) -> None:
        events: list[tuple[str, Thresholds]] = []
        real_cal = sel.calibrate

        def cal_spy(*a, **kw):
            cal = real_cal(*a, **kw)
            events.append(("calibrate", cal.thresholds))
            return cal

        def record(real: Callable) -> Callable:
            def f(policy, env, rule, **kw):
                events.append(("run", rule.thresholds))
                return real(policy, env, rule, **kw)
            return f
        with mock.patch.object(sel, "calibrate", cal_spy), patched_runs(record):
            res = sel.run_selection(small_dev(), FAST, CTX)
        kinds = [k for k, _ in events]
        first_run = kinds.index("run")
        self.assertGreater(first_run, 1)                           # the candidate and every CV fold came first
        self.assertNotIn("calibrate", kinds[first_run:])
        frozen = res["thresholds"]
        self.assertEqual(events[0], ("calibrate", frozen))         # the first calibration is the candidate
        runs = [th for k, th in events if k == "run"]
        self.assertEqual(len(runs), res["n_episode_runs"] + res["n_heldout_runs"])
        # the check and the sensitivity only ever see the frozen taus; the held-out check (last) only its folds'
        check_runs, held_runs = runs[:res["n_episode_runs"]], runs[res["n_episode_runs"]:]
        for th in check_runs:
            self.assertEqual((th.tau_q, th.tau_occ, th.tau_match), (frozen.tau_q, frozen.tau_occ, frozen.tau_match))
        self.assertEqual({(th.k, th.lam) for th in check_runs}, {(2, 0.0), (3, 0.0), (2, 0.7)})
        fold_taus = {(f.thresholds.tau_q, f.thresholds.tau_occ, f.thresholds.tau_match) for f in res["folds"]}
        self.assertTrue(held_runs)
        for th in held_runs:
            self.assertIn((th.tau_q, th.tau_occ, th.tau_match), fold_taus)
            self.assertIn((th.k, th.lam), {(2, 0.0), (1, 0.0)})   # the k rule, and the -fcm / S rule

    def test_calibration_and_cv_never_touch_the_replay(self) -> None:
        def boom(*a, **kw):
            raise AssertionError("a policy or an env was used while calibrating")
        dev = small_dev()
        rows = sel.view_rows(dev)
        with mock.patch.object(sel, "run_episode", boom), mock.patch.object(pol, "run_episode", boom), \
                mock.patch.object(sel, "ReplayEnv", boom), mock.patch.object(sel, "PolicyRunner", boom):
            cal = sel.calibrate(rows, FAST, log_warnings=False)
            folds = sel.grouped_cv(rows, FAST, dev.arrangements)
        self.assertEqual(len(folds), len(dev.arrangements))
        self.assertEqual((cal.thresholds.k, cal.thresholds.lam), (2, 0.0))

    def test_thresholds_ignore_policy_outcomes(self) -> None:
        dev = small_dev()
        clean = sel.run_selection(dev, FAST, CTX)
        with patched_runs(flip_to_removed):
            dirty = sel.run_selection(dev, FAST, CTX)
        for key in ("thresholds", "calibration", "folds", "cv_held_out", "view_rates"):
            self.assertEqual(dirty[key], clean[key], key)
        # every present episode is now a false removal: every L1 / L2 row is flagged, and nothing was changed
        self.assertTrue(all(r.counts.frr == 1.0 for r in dirty["check"]))
        self.assertEqual(len(dirty["flags"]), sum(r.lighting in LIGHTS for r in dirty["check"]))

    def test_frr_limit_is_a_check_not_an_objective(self) -> None:
        # Review: on small_dev(15) every FRR was 0, so no limit could flag anything and this test could not fail. With
        # one tau_match this dev set has false removals (S and R on L2): the limit flags them and changes nothing.
        dev = small_dev(9, 10)
        base = dataclasses.replace(FAST, tau_match_grid=(0.6,), k_sensitivity=(), lam_sensitivity=())
        runs = {lim: sel.run_selection(dev, dataclasses.replace(base, frr_limit=lim), CTX) for lim in (0.0, 0.1, 1.0)}
        ref = runs[1.0]   # a limit nothing can exceed
        for lim, res in runs.items():
            with self.subTest(frr_limit=lim):
                over = [r for r in res["check"] if r.lighting in LIGHTS and r.counts.frr > lim + 1e-9]
                self.assertEqual(len(res["flags"]), len(over))
                for key in ("thresholds", "calibration", "folds", "view_rates"):
                    self.assertEqual(res[key], ref[key], key)
                self.assertEqual([r.counts for r in res["check"]], [r.counts for r in ref["check"]])
        self.assertGreater(len(runs[0.1]["flags"]), 0)   # the default limit does flag here: the check is not vacuous
        self.assertGreater(len(runs[0.0]["flags"]), len(runs[0.1]["flags"]))
        self.assertEqual(ref["flags"], [])
        self.assertIs(runs[0.1]["all_within_limit"], False)
        self.assertIs(ref["all_within_limit"], True)


class SensitivityTest(unittest.TestCase):
    def test_sensitivity_never_changes_the_frozen_candidate(self) -> None:
        dev = small_dev()
        none = dataclasses.replace(FAST, k_sensitivity=(), lam_sensitivity=())
        full = dataclasses.replace(FAST, k_sensitivity=(3,), lam_sensitivity=(0.7, 1.2))
        a, b = sel.run_selection(dev, none, CTX), sel.run_selection(dev, full, CTX)

        def corrupt(real: Callable) -> Callable:   # spoil only the sensitivity runs (k = 3 or lam != 0)
            def f(policy, env, rule, **kw):
                run = real(policy, env, rule, **kw)
                if rule.k == 3 or rule.thresholds.lam != 0.0:
                    return dataclasses.replace(run, decision=Decision.REMOVED)
                return run
            return f
        with patched_runs(corrupt):
            c = sel.run_selection(dev, full, CTX)
        self.assertEqual(a["sensitivity"], {"k": [], "lam": []})
        self.assertTrue(b["sensitivity"]["k"] and b["sensitivity"]["lam"])
        self.assertNotEqual([r.counts for r in c["sensitivity"]["k"]], [r.counts for r in b["sensitivity"]["k"]])
        docs = {n: sel.candidate_doc(dev, s, r, "x") for n, s, r in (("a", none, a), ("b", full, b), ("c", full, c))}
        for other in ("b", "c"):
            for key in ("thresholds", "fixed_a_priori", "calibration", "dev_check"):
                self.assertEqual(docs[other][key], docs["a"][key], (other, key))

    def test_sensitivity_rows(self) -> None:
        res = sel.run_selection(small_dev(), dataclasses.replace(FAST, lam_sensitivity=(0.7, 1.2)), CTX)
        k_rows, lam_rows = res["sensitivity"]["k"], res["sensitivity"]["lam"]
        self.assertEqual({r.policy for r in k_rows}, set(sel.K_SENSITIVITY_POLICIES))   # S is k = 1 by construction
        self.assertEqual({(r.k, r.lam) for r in k_rows}, {(3, 0.0)})
        self.assertEqual({r.policy for r in lam_rows}, {"G", "G-lookahead"})          # the only readers of lam
        self.assertEqual({(r.k, r.lam) for r in lam_rows}, {(2, 0.7), (2, 1.2)})
        self.assertEqual({(r.lighting, r.cap_steps) for r in lam_rows},
                         {(light, cap) for light in (*LIGHTS, "pooled") for cap in (None, 2)})
        self.assertTrue(all(r.bounds is None and r.bootstrap is None for r in k_rows + lam_rows))

    def test_lam_values_are_checked_against_g_behaviour(self) -> None:
        res = sel.run_selection(small_dev(), dataclasses.replace(FAST, k_sensitivity=(), lam_sensitivity=(0.5,)), CTX)
        chk = res["lam_grid_check"]
        self.assertEqual(chk.classes, ((0.0, 0.5),))   # 0.5 is below the first breakpoint 2/3: it behaves like 0
        self.assertTrue(chk.warnings)
        self.assertTrue(set(chk.warnings) <= set(res["warnings"]))


class DevCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dev = small_dev()
        cls.settings = dataclasses.replace(FAST, k_sensitivity=(), lam_sensitivity=())
        cls.res = sel.run_selection(cls.dev, cls.settings, CTX)

    def row(self, policy: str, cap: float | None, light: str) -> sel.CheckRow:
        return next(r for r in self.res["check"] if (r.policy, r.cap_steps, r.lighting) == (policy, cap, light))

    def test_every_policy_is_reported_for_both_lightings(self) -> None:
        n_arr = len(self.dev.arrangements)
        self.assertEqual(sel.CHECK_POLICIES, ("S", "F", "R", "G", "G-lookahead", "G-myopic"))
        self.assertEqual(len(self.res["check"]), 3 * (1 + 5 * 2))
        for p in sel.CHECK_POLICIES:
            for cap in ((None,) if p == "S" else (None, 2)):
                for light in LIGHTS:   # an L1 row holds that lighting's episodes only: 3 per arrangement
                    r = self.row(p, cap, light)
                    self.assertEqual((r.counts.n_episodes, r.counts.n_arrangements), (3 * n_arr, n_arr))
                    self.assertEqual((r.k, r.lam, r.budget), (1, 0.0, 0) if p == "S" else (2, 0.0, 3))
                    self.assertEqual(r.bounds["conf"], 0.95)
                    self.assertEqual(r.bootstrap["n_arrangements"], n_arr)
                l1, l2, pooled = (self.row(p, cap, x).counts for x in (*LIGHTS, "pooled"))
                self.assertEqual(pooled.n_episodes, l1.n_episodes + l2.n_episodes)
                self.assertAlmostEqual(pooled.n_resolved, l1.n_resolved + l2.n_resolved)

    def test_matches_analysis_metrics_by_lighting(self) -> None:
        runner, th = self.res["runner"], self.res["thresholds"]
        for p, cap in (("S", None), ("F", 2), ("R", None), ("G", None), ("G-lookahead", 2), ("G-myopic", None)):
            by = compute(runner.results(p, th, sel.run_budget(p, cap, self.settings, POS)), self.dev.truth,
                         group_by="lighting")
            for light in LIGHTS:
                c, m = self.row(p, cap, light).counts, by[light]
                with self.subTest(policy=p, lighting=light):
                    self.assertAlmostEqual(c.frr, m["false_removal_rate"])
                    self.assertAlmostEqual(c.recall, m["removal_recall"])
                    self.assertAlmostEqual(c.coverage, m["coverage"])
                    self.assertAlmostEqual(c.mean_travel_cm, m["mean_travel_cm"])

    def test_r_gets_the_bootstrap_but_no_binomial_bounds(self) -> None:
        r = self.row("R", None, "L1")
        self.assertFalse(r.counts.binomial)
        self.assertIsNone(r.counts.n_arr_with_false_removal)
        self.assertIsNone(r.bounds["frr_episode"]["upper"])
        self.assertIn("not binomial", r.bounds["note"])
        self.assertEqual(r.bootstrap["B"], self.settings.bootstrap_B)
        g = self.row("G", None, "L1")
        self.assertTrue(g.counts.binomial)
        self.assertIsNotNone(g.bounds["frr_episode"]["upper"])

    def test_flags_name_every_policy_above_the_limit(self) -> None:
        hi = sel.Counts(n_episodes=6, n_present=4, n_false_removal=1, n_removed=2, n_arrangements=2,
                        n_arr_with_present=2, n_arr_with_false_removal=1)
        lo = dataclasses.replace(hi, n_false_removal=0, n_arr_with_false_removal=0)

        def mk(policy: str, light: str, c: sel.Counts) -> sel.CheckRow:
            return sel.CheckRow(policy, 2, 0.0, 3, None, None, light, c, None, None, c.frr > 0.10)
        rows = [mk("G", "L1", hi), mk("G", "L2", lo), mk("F", "pooled", hi), mk("R", "L2", hi)]
        flags = sel.check_flags(rows, 0.10)
        self.assertEqual(len(flags), 2)   # the pooled row is secondary: never a flag
        self.assertTrue(flags[0].startswith("dev check: G, b = 3, uncapped, L1 queries: FRR 0.250 (1/4) > frr_limit"))
        self.assertTrue(flags[1].startswith("dev check: R, b = 3, uncapped, L2 queries"))
        self.assertTrue(all("not changed" in f for f in flags))

    def test_calibration_rates_reported_per_lighting(self) -> None:
        rates = self.res["view_rates"]
        self.assertEqual(set(rates), {"L1", "L2", "pooled"})
        self.assertEqual(rates["L1"] + rates["L2"], rates["pooled"])
        rows = sel.view_rows(self.dev)
        self.assertEqual(rates["L2"], sel.ViewRates.of([r for r in rows if r.lighting is Lighting.L2],
                                                        self.res["thresholds"]))


class GroupedCVTest(unittest.TestCase):
    def test_each_fold_calibrates_without_its_held_out_arrangement(self) -> None:
        dev = small_dev(15)
        rows = sel.view_rows(dev)
        seen: list[tuple[str, ...]] = []
        real = sel.calibrate

        def spy(rows_, settings, arrangements=None, **kw):
            cal = real(rows_, settings, arrangements, **kw)
            seen.append(cal.arrangements)
            return cal
        with mock.patch.object(sel, "calibrate", spy):
            folds = sel.grouped_cv(rows, FAST, dev.arrangements)
        self.assertEqual([f.held_out for f in folds], list(dev.arrangements))
        for f, arrs in zip(folds, seen):
            self.assertNotIn(f.held_out, arrs)
            self.assertEqual(set(arrs) | {f.held_out}, set(dev.arrangements))
            train = [r for r in rows if r.arrangement_id != f.held_out]
            self.assertEqual(f.thresholds, real(train, FAST, log_warnings=False).thresholds)
            held = [r for r in rows if r.arrangement_id == f.held_out]
            self.assertEqual(f.held["pooled"], sel.ViewRates.of(held, f.thresholds))

    def test_pooled_held_out_covers_every_view_once(self) -> None:
        dev = small_dev(15)
        rows = sel.view_rows(dev)
        pooled = sel.pool_folds(sel.grouped_cv(rows, FAST, dev.arrangements))
        p = pooled["pooled"]
        self.assertEqual((p.false_clear[1], p.false_clear_miss[1], p.clear_miss[1]),
                         (len(sel.gate_null_views(rows)), len(sel.present_diagnostic_views(rows)),
                          len(sel.removed_diagnostic_views(rows))))
        self.assertEqual(pooled["L1"] + pooled["L2"], p)

    def test_fewer_than_two_arrangements_has_no_folds(self) -> None:
        rows = sel.view_rows(small_dev())
        self.assertEqual(sel.grouped_cv(rows, FAST, [rows[0].arrangement_id]), [])


class CountsAndBoundsTest(unittest.TestCase):
    @staticmethod
    def o(eid: str, arr: str, present: bool, removed: float, resolved: float, travel: float = 0.0,
          views: float = 1.0) -> sel.Outcome:
        return sel.Outcome(eid, arr, Lighting.L1, present, removed, resolved, travel, views)

    def test_counts_arrangement_level(self) -> None:
        o = self.o
        outs = [o("A-v", "A", True, 1.0, 1.0, 10.0, 2), o("A-o", "A", True, 0.0, 1.0, 0.0, 1),
                o("A-r", "A", False, 1.0, 1.0, 20.0, 3), o("B-v", "B", True, 0.0, 0.0, 30.0, 4),
                o("B-r", "B", False, 0.0, 0.0, 0.0, 1)]
        c = sel.Counts.of(outs)
        self.assertEqual((c.n_present, c.n_false_removal, c.n_removed, c.n_true_removal, c.n_resolved),
                         (3, 1, 2, 1, 3))
        self.assertIsInstance(c.n_false_removal, int)
        self.assertEqual((c.n_arrangements, c.n_arr_with_present, c.n_arr_with_false_removal), (2, 2, 1))
        self.assertAlmostEqual(c.mean_travel_cm, 12.0)
        self.assertAlmostEqual(c.mean_views, 2.2)
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

    def test_fractional_counts_get_no_binomial_bounds(self) -> None:
        outs = [self.o("A-v", "A", True, 0.25, 1.0), self.o("A-r", "A", False, 0.5, 0.75)]
        with self.assertRaises(ValueError):
            sel.Counts.of(outs)   # binomial counts need 0/1 decisions
        c = sel.Counts.of(outs, binomial=False)
        self.assertAlmostEqual(c.frr, 0.25)
        self.assertIsNone(c.n_arr_with_false_removal)
        b = sel.bounds(c, 0.95)
        self.assertIsNone(b["frr_episode"]["upper"])
        self.assertIsNone(b["recall"]["lower"])

    def test_cluster_bootstrap(self) -> None:
        outs = [self.o(f"{a}-{i}", a, i < 2, 0.0 if i < 2 else 1.0, 1.0) for a in ("A", "B", "C") for i in range(3)]
        bt = sel.cluster_bootstrap(outs, 200, 0)
        self.assertEqual(bt["frr"]["ci95"], [0.0, 0.0])
        self.assertEqual(bt["recall"]["ci95"], [1.0, 1.0])
        empty = sel.cluster_bootstrap([], 10, 0)
        self.assertEqual((empty["frr"]["ci95"], empty["n_arrangements"]), ([None, None], 0))


class WriteOrderTest(unittest.TestCase):
    """Review: write_outputs wrote the candidate and its .sha256 first, so a failure before the report and details left
    a new candidate next to the previous run's report and details."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dev = small_dev()
        cls.a_settings = dataclasses.replace(FAST, k_sensitivity=(), lam_sensitivity=())   # an earlier run, A
        cls.b_settings = dataclasses.replace(cls.a_settings, tau_match_grid=(0.5,))        # the run that fails, B
        cls.a = sel.run_selection(cls.dev, cls.a_settings, CTX)
        cls.b = sel.run_selection(cls.dev, cls.b_settings, CTX)

    @staticmethod
    def files(out: Path) -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in sorted(out.iterdir())}

    def test_a_render_failure_leaves_the_earlier_run_as_it_was(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            sel.write_outputs(out, self.dev, self.a_settings, self.a)
            before = self.files(out)
            self.assertEqual(set(before), {*sel.CANDIDATE_FILES, sel.REPORT_FILE})
            for target in ("candidate_doc", "render_report", "check_episodes"):   # every text is rendered first
                with self.subTest(failing=target):
                    with mock.patch.object(sel, target, side_effect=RuntimeError("boom")), \
                            self.assertRaises(RuntimeError):
                        sel.write_outputs(out, self.dev, self.b_settings, self.b)
                    self.assertEqual(self.files(out), before)

    def test_a_write_failure_never_leaves_a_candidate_beside_another_runs_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ref = Path(d) / "ref"
            sel.write_outputs(ref, self.dev, self.b_settings, self.b)
            want = self.files(ref)   # B's complete output
            real = sel._write_atomic
            for name in want:   # fail on each file in turn
                out = Path(d) / f"out-{name}"
                sel.write_outputs(out, self.dev, self.a_settings, self.a)
                (out / sel.INFEASIBLE_FILE).write_text("{}")   # a stale failure record too

                def failing(path: Path, text: str, name: str = name) -> None:
                    if path.name == name:
                        raise OSError("disk full")
                    real(path, text)
                with self.subTest(failing=name):
                    with mock.patch.object(sel, "_write_atomic", failing), self.assertRaises(OSError):
                        sel.write_outputs(out, self.dev, self.b_settings, self.b)
                    got = self.files(out)
                    self.assertNotIn("frozen_candidate.json", got)   # the candidate is written last
                    self.assertLessEqual(set(got), set(want))         # A's files and the failure record are gone
                    self.assertEqual(got, {n: want[n] for n in got})  # what is there is B's

    def test_atomic_write_replaces_or_keeps_the_old_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.txt"
            p.write_text("old")
            with mock.patch.object(sel.os, "replace", side_effect=OSError("boom")), self.assertRaises(OSError):
                sel._write_atomic(p, "new")
            self.assertEqual((p.read_text(), [x.name for x in Path(d).iterdir()]), ("old", ["f.txt"]))   # no temp
            sel._write_atomic(p, "new")
            self.assertEqual((p.read_text(), [x.name for x in Path(d).iterdir()]), ("new", ["f.txt"]))


class EndToEndTest(unittest.TestCase):
    ARGS =["--synthetic", "--n-arr", "12", "--seed", "1", "--bootstrap-B", "60", "--tau-match-grid", "0.5,0.6",
            "--cap-steps", "3", "--r-seeds", "2", "--lam-sensitivity", "0.7"]

    def test_cli_writes_candidate_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            shas = []
            for name in ("a", "b"):
                out = Path(d) / name
                self.assertEqual(sel.main([*self.ARGS, "--out", str(out)]), 0)
                text = (out / "frozen_candidate.json").read_text()
                sha = hashlib.sha256(text.encode()).hexdigest()
                self.assertEqual((out / "frozen_candidate.json.sha256").read_text().split()[0], sha)
                shas.append(sha)
            self.assertEqual(shas[0], shas[1])
            doc = json.loads(text)
            rep = (out / "selection_report.md").read_text()
            details = json.loads((out / "selection_details.json").read_text())
        self.assertTrue(doc["synthetic"])
        self.assertEqual(doc["label"], SYNTHETIC_LABEL)
        self.assertIn("CANDIDATE", doc["status"])
        self.assertEqual((doc["procedure_version"], doc["schema"]), ("select-v4", "cmpe249.frozen_candidate.v3"))
        self.assertEqual(doc["procedure_revision"], sel.PROCEDURE_REVISION)
        self.assertIs(doc["eligible_for_freeze"], True)
        self.assertEqual(doc["ineligible_reasons"], [])
        th = doc["thresholds"]
        self.assertEqual((th["k"], th["lam"]), (2, 0.0))
        self.assertIn(th["tau_match"], (0.5, 0.6))
        self.assertEqual((doc["fixed_a_priori"]["k"], doc["fixed_a_priori"]["lam"]), (2, 0.0))
        self.assertIs(doc["selected_on_policy_results"], False)
        self.assertEqual(doc["alpha_clear"], 0.1)
        cal = doc["calibration"]
        self.assertEqual(set(cal["per_view_rates"]), {"L1", "L2", "pooled"})
        self.assertEqual(set(cal["grouped_cv"]["held_out_per_view_rates"]), {"L1", "L2", "pooled"})
        self.assertEqual(len(cal["grouped_cv"]["folds"]), len(doc["dev_arrangements"]))
        self.assertEqual((cal["chosen"]["tau_q"], cal["chosen"]["tau_occ"], cal["chosen"]["tau_match"]),
                         (th["tau_q"], th["tau_occ"], th["tau_match"]))
        self.assertEqual(cal["order"], sel.FULL_ORDER)
        self.assertEqual(set(cal["curves"]), {"tau_q", "tau_occ", "tau_match"})
        self.assertEqual(set(cal["gate_pass_by_lighting"]) - {"note"}, {"L1", "L2", "pooled"})
        self.assertFalse(doc["bootstrap_stratification"]["stratified"])
        chk = doc["dev_check"]
        self.assertEqual(chk["policies"], list(sel.CHECK_POLICIES))
        self.assertEqual({(r["policy"], r["lighting"]) for r in chk["rows"]},
                         {(p, x) for p in sel.CHECK_POLICIES for x in (*LIGHTS, "pooled")})
        self.assertAlmostEqual(chk["travel_caps_cm"][1], 3 * Positions.load().nominal_step_cm())  # configs/positions
        self.assertEqual(chk["unchecked"], [])   # both lightings have present-target episodes: every row checked
        self.assertEqual(chk["all_within_limit"], not chk["flags"])
        self.assertEqual(chk["g_lookahead"], {"G": False, "G-lookahead": True})
        self.assertEqual(len(doc["view_labels_sha256"]), 64)
        self.assertIn(f"labels sha256 `{doc['view_labels_sha256']}`", rep)
        self.assertIn(sel.frr_bound_note(len(doc["dev_arrangements"]), 0.1, 0.95), rep)
        self.assertNotIn("about 10", rep)
        sens = doc["sensitivity"]
        self.assertEqual((sens["k_values"], sens["lam_values"]), ([3], [0.7]))
        self.assertTrue(sens["k"] and sens["lam"])
        self.assertIn("lam_grid_check", doc)
        self.assertTrue(rep.startswith(f"# {SYNTHETIC_LABEL}"))
        for needle in ("select-v4", "proposed default - confirm", "## Step 1: joint gate + tau_match search", "grouped cross-validation",
                       "### tau_match curve", "Gate pass rates per lighting", sel.FULL_ORDER,
                       "### (b) Held-out policy check", "**UNSTRATIFIED**",
                       "### L1 queries", "### L2 queries", "Step 5: sensitivity (reported only, never selected)",
                       "select-v3 (replaced)", "Secondary, not a headline",
                       "grouped cross-validation (Step 3): complete", "does not turn it into alpha squared"):
            self.assertIn(needle, rep)
        self.assertEqual({e["lighting"] for e in details["check_episodes"]}, set(LIGHTS))

    def test_synthetic_provenance_in_candidate_and_settings_hash(self) -> None:
        # Review F4: the generator regime must be recorded and must change settings_sha256.
        args = ["--synthetic", "--n-arr", "9", "--seed", "2", "--bootstrap-B", "20", "--tau-match-grid", "0.5",
                "--cap-steps", "none", "--r-seeds", "1", "--k-sensitivity", "none", "--lam-sensitivity", "none"]
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            docs = {}
            for name, extra in (("default", []), ("anywhere", ["--block-anywhere"])):
                out = Path(d) / name
                self.assertEqual(sel.main([*args, "--out", str(out), *extra]), 0)
                docs[name] = json.loads((out / "frozen_candidate.json").read_text())
        for name, doc in docs.items():
            prov = doc["synthetic_provenance"]
            self.assertEqual((prov["n_arr"], prov["n_dev"], prov["seed"]), (9, len(doc["dev_arrangements"]), 2))
            self.assertEqual(prov["generator"]["block_includes_start"], name == "default")
            self.assertEqual(doc["dev_check"]["cap_steps"], [None])
            self.assertEqual(doc["sensitivity"]["k"], [])
        self.assertNotEqual(docs["default"]["settings_sha256"], docs["anywhere"]["settings_sha256"])

    def test_report_says_plainly_when_no_sensitivity_or_cap_ran(self) -> None:
        # Review: empty lists gave "k = (none) runs only as a reported sensitivity.", "lam in {} runs ..." and the
        # settings row "uncapped,  arc steps".
        args = ["--synthetic", "--n-arr", "9", "--seed", "2", "--bootstrap-B", "20", "--tau-match-grid", "0.5",
                "--cap-steps", "none", "--r-seeds", "1", "--k-sensitivity", "none", "--lam-sensitivity", "none"]
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = Path(d) / "run"
            self.assertEqual(sel.main([*args, "--out", str(out)]), 0)
            rep = (out / sel.REPORT_FILE).read_text()
            doc = json.loads((out / "frozen_candidate.json").read_text())
        for bad in ("(none)", "{}", "uncapped,  arc steps", "uncapped and under every cap", "k = 3"):
            self.assertNotIn(bad, rep)
        for needle in ("\"at least two distinct positions\"; no k sensitivity run was requested.",
                       "reported as curves; no lam sensitivity run was requested.",
                       "| cap_steps | uncapped only (no capped setting was run) |",
                       "primary budget b = 3, uncapped only (no capped setting was run) (caps are",
                       "- no k sensitivity run was requested", "- no lam sensitivity run was requested"):
            self.assertIn(needle, rep)
        self.assertNotIn("k = 3", doc["fixed_a_priori"]["k_note"])   # it said "k = 3 runs" with none requested
        self.assertEqual((doc["sensitivity"]["k_values"], doc["sensitivity"]["lam_values"]), ([], []))

    def test_report_text_for_given_values(self) -> None:
        s = dataclasses.replace(FAST, k_sensitivity=(3, 4), lam_sensitivity=(0.7,), cap_steps=(1, 2.5))
        self.assertEqual(sel._k_sensitivity_text(s), "k in {3, 4} runs only as a reported sensitivity")
        self.assertEqual(sel._lam_sensitivity_text(s), "lam = 0.7 runs only as a reported sensitivity")
        self.assertEqual(sel._caps_text(s), "uncapped, 1, 2.5 arc steps")

    def test_bound_note_follows_the_number_of_dev_arrangements(self) -> None:
        # Review: "no formal guarantee on FRR is possible with about 10 dev arrangements" was a literal in the module
        # docstring and every report, false for the 60-arrangement sim dev sets. The note is built from the count.
        self.assertNotIn("about 10", sel.__doc__)
        ten, sixty = sel.frr_bound_note(10, 0.10, 0.95), sel.frr_bound_note(60, 0.10, 0.95)
        self.assertIn("- With 10 dev arrangements, the best one-sided 95% arrangement-level FRR bound", ten)
        self.assertIn("is 0.259, above frr_limit = 0.1, so no dev result can bring that bound below the limit.", ten)
        self.assertIn("- With 60 dev arrangements, the best one-sided 95% arrangement-level FRR bound", sixty)
        self.assertIn("is 0.049. The check", sixty)
        self.assertNotIn("above frr_limit", sixty)
        for note in (ten, sixty):
            self.assertIn("goals, not guarantees", note)

    def test_cli_requires_synthetic(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(sel.main([]), 2)

    def test_custom_devdata_with_hand_labels(self) -> None:
        # The real-data path on hand_dev(): calibration and the check must run and never touch anything but this data.
        dev = hand_dev()
        res = sel.run_selection(dev, HAND, REAL)
        self.assertEqual(res["thresholds"], Thresholds(tau_match=0.5, tau_q=0.8, tau_occ=0.1, k=2, lam=0.0))
        g = next(r for r in res["check"] if (r.policy, r.lighting) == ("G", "L1"))
        self.assertEqual((g.counts.n_false_removal, g.counts.n_true_removal, g.counts.n_resolved), (0, 2, 6))
        g2 = next(r for r in res["check"] if (r.policy, r.lighting) == ("G", "L2"))
        self.assertEqual((g2.counts.n_episodes, g2.over_limit), (0, None))   # no L2 episodes: reported, undefined
        doc = sel.candidate_doc(dev, HAND, res, "x")
        self.assertEqual((doc["synthetic"], doc["label"]), (False, ""))
        self.assertEqual((doc["dev_evidence_sha256"], doc["view_labels_sha256"]),
                         (dev.store_factory.evidence_sha256(), dev.view_labels_sha256))
        self.assertIn(f"labels sha256 `{dev.view_labels_sha256}`", sel.render_report(dev, HAND, res, doc, "y"))

    def test_an_unchecked_lighting_makes_the_check_incomplete_not_passed(self) -> None:
        # Review: hand_dev() has no L2 episode, so FRR is undefined in every L2 row of the check; all_within_limit was
        # `not flags`, i.e. true, and the report said that no policy is above the limit on L1 or L2 queries.
        dev = hand_dev()
        res = sel.run_selection(dev, HAND, REAL)
        l2 = [r for r in res["check"] if r.lighting == "L2"]
        self.assertTrue(l2 and all(r.over_limit is None for r in l2))
        self.assertEqual(res["flags"], [])
        self.assertEqual(len(res["unchecked"]), len(l2))
        self.assertTrue(all("L2 queries: FRR undefined (no present-target episode)" in u for u in res["unchecked"]))
        self.assertIsNone(res["all_within_limit"])
        doc = sel.candidate_doc(dev, HAND, res, "x")
        self.assertIsNone(doc["dev_check"]["all_within_limit"])
        self.assertEqual(doc["dev_check"]["unchecked"], res["unchecked"])
        rep = sel.render_report(dev, HAND, res, doc, "y")
        self.assertNotIn("no policy is above FRR 0.1 on L1 or L2 dev queries at any setting", rep)
        for needle in (f"- **NOT CHECKED**: L2 queries: FRR is undefined in {len(l2)} of {len(l2)} rows",
                       f"**INCOMPLETE**: FRR undefined in {len(l2)} L1 / L2 row(s)", "`all_within_limit` null",
                       "where FRR is defined, but the check is incomplete"):
            self.assertIn(needle, rep)
        # a flag settles the answer whatever was not checked: false, not null
        with patched_runs(flip_to_removed):
            flagged = sel.run_selection(dev, HAND, REAL)
        self.assertTrue(flagged["flags"] and flagged["unchecked"])
        self.assertIs(flagged["all_within_limit"], False)
        self.assertEqual([sel.all_within_limit(f, u) for f, u in (([], []), (["f"], []), ([], ["u"]), (["f"], ["u"]))],
                         [True, False, None, False])


if __name__ == "__main__":
    unittest.main()
