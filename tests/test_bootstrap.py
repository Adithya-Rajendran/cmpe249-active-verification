"""Tests for analysis/bootstrap.py: paired, arrangement-level resampling (inputs built with schema types)."""
from __future__ import annotations

import unittest
from collections import Counter
from unittest import mock

import numpy as np

from analysis import bootstrap
from analysis.bootstrap import (
    arrangement_draws,
    draw_diffs,
    pair,
    paired_diff,
    rate_intervals,
    resolve_strata,
    sample_draws,
    stratified_arrangement_draws,
)
from analysis.metrics import metrics_on
from verify.schema import (
    CONDITIONS,
    LIGHTINGS,
    SYNTHETIC_LABEL,
    Decision,
    EpisodeResult,
    GroundTruth,
    Split,
    episode_id,
    truth_from_condition,
)

P, R, U = Decision.PRESENT, Decision.REMOVED, Decision.UNKNOWN
EPISODES_PER_ARR = len(CONDITIONS) * len(LIGHTINGS)  # 6


def res(arr: str, cond, light, d: Decision | dict, policy: str, synthetic: bool = False) -> EpisodeResult:
    frac = {x: (1.0 if x is d else 0.0) for x in Decision} if isinstance(d, Decision) else \
        {x: float(d.get(x, 0.0)) for x in Decision}
    return EpisodeResult(episode_id=episode_id(arr, cond, light), arrangement_id=arr, split=Split.TEST,
                         condition=cond, lighting=light, policy=policy, budget_additional_views=3,
                         travel_cap_cm=None, decision_frac=frac, mean_views=2.0, mean_additional_views=1.0,
                         mean_travel_cm=23.56, seeds=(None,), synthetic=synthetic)


def make(n_arr: int, decide, policy: str, synthetic: bool = False) -> list[EpisodeResult]:
    """decide(arr_index, condition, lighting) -> Decision | frac dict."""
    return [res(f"A{a:02d}", c, l, decide(a, c, l), policy, synthetic)
            for a in range(n_arr) for c in CONDITIONS for l in LIGHTINGS]


def truth_for(results) -> GroundTruth:
    return GroundTruth({r.episode_id: truth_from_condition(r.condition) for r in results})


def correct(a, c, l) -> Decision:
    return R if c.value == "removed" else P


def noisy(a, c, l) -> Decision:
    """Correct except a varying, arrangement-dependent set of mistakes / unknowns."""
    k = (a * 7 + CONDITIONS.index(c) * 3 + LIGHTINGS.index(l)) % 5
    if k == 0:
        return U
    if k == 1:
        return P if c.value == "removed" else R
    return correct(a, c, l)


class TestIdentical(unittest.TestCase):
    def test_identical_policies_zero_ci(self) -> None:
        a = make(8, noisy, "G")
        b = make(8, noisy, "G")
        truth = truth_for(a)
        for metric in ("false_removal_rate", "removal_recall", "coverage", "resolved_accuracy",
                       "mean_travel_cm"):
            r = paired_diff(a, b, truth, metric, B=500, seed=1)
            self.assertEqual((r.estimate, r.ci_lo, r.ci_hi), (0.0, 0.0, 0.0), metric)
            self.assertEqual(r.n_undefined, 0)
            self.assertEqual((r.n_arrangements, r.n_episodes, r.B, r.seed), (8, 48, 500, 1))


class TestKnownDifference(unittest.TestCase):
    def test_constant_difference(self) -> None:
        """B says unknown on exactly 1 of 6 episodes per arrangement -> coverage diff 1/6 in every draw."""
        a = make(5, correct, "G")
        b = make(5, lambda i, c, l: U if (c.value, l.value) == ("visible", "L1") else correct(i, c, l), "F")
        r = paired_diff(a, b, truth_for(a), "coverage", B=300, seed=0)
        self.assertEqual((r.policy_a, r.policy_b), ("G", "F"))
        self.assertAlmostEqual(r.estimate, 1 / 6)
        self.assertAlmostEqual(r.ci_lo, 1 / 6)
        self.assertAlmostEqual(r.ci_hi, 1 / 6)

    def test_estimate_and_ci_bracket(self) -> None:
        """A false-removes every present episode of arrangement 0 only: FRR diff = 4 / (4 * 6)."""
        a = make(6, lambda i, c, l: R if i == 0 else correct(i, c, l), "G")
        b = make(6, correct, "F")
        r = paired_diff(a, b, truth_for(a), "false_removal_rate", B=2000, seed=3)
        self.assertAlmostEqual(r.estimate, 4 / 24)
        self.assertEqual(r.ci_lo, 0.0)          # draws without arrangement 0 exist
        self.assertGreater(r.ci_hi, r.estimate)
        self.assertEqual(r.n_undefined, 0)


class TestResamplingUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.a = make(7, noisy, "G")
        self.b = make(7, correct, "F")
        self.truth = truth_for(self.a)
        self.sample = pair(self.a, self.b, self.truth)

    def test_arrangement_never_split(self) -> None:
        draws = arrangement_draws(len(self.sample.arrangement_ids), 200, seed=5)
        self.assertEqual(draws.shape, (200, 7))
        for draw in draws:
            ra, rb = self.sample.expand(draw)
            self.assertEqual(len(ra), EPISODES_PER_ARR * len(draw))
            self.assertEqual([r.episode_id for r in ra], [r.episode_id for r in rb])   # paired
            per_arr = Counter(r.arrangement_id for r in ra)
            per_ep = Counter(r.episode_id for r in ra)
            drawn = Counter(self.sample.arrangement_ids[i] for i in draw)
            for arr, n in per_arr.items():
                self.assertEqual(n % EPISODES_PER_ARR, 0)
                self.assertEqual(n, EPISODES_PER_ARR * drawn[arr])
            for eid, n in per_ep.items():
                # every episode of a drawn arrangement appears exactly as often as it was drawn
                self.assertEqual(n, drawn[eid.split("-")[0]])
            self.assertEqual(set(per_arr), set(drawn))

    def test_vectorised_draws_match_multiset_metrics(self) -> None:
        draws = arrangement_draws(len(self.sample.arrangement_ids), 50, seed=9)
        for metric in ("false_removal_rate", "coverage", "resolved_accuracy"):
            diffs = draw_diffs(self.sample, self.truth, metric, draws)
            for draw, d in zip(draws, diffs):
                ra, rb = self.sample.expand(draw)
                ma, mb = metrics_on(ra, self.truth)[metric], metrics_on(rb, self.truth)[metric]
                self.assertAlmostEqual(d, ma - mb, places=12)

    def test_paired_diff_uses_arrangement_draws(self) -> None:
        with mock.patch.object(bootstrap, "arrangement_draws", wraps=arrangement_draws) as spy:
            r = paired_diff(self.a, self.b, self.truth, "coverage", B=400, seed=11)
        spy.assert_called_once_with(7, 400, 11)
        diffs = draw_diffs(self.sample, self.truth, "coverage", arrangement_draws(7, 400, 11))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        self.assertEqual((r.ci_lo, r.ci_hi), (float(lo), float(hi)))

    def test_sample_grouping(self) -> None:
        self.assertEqual(self.sample.arrangement_ids, tuple(f"A{i:02d}" for i in range(7)))
        self.assertTrue(all(len(g) == EPISODES_PER_ARR for g in self.sample.groups))


class TestDeterminism(unittest.TestCase):
    def test_seed_reproducible(self) -> None:
        a, b = make(9, noisy, "G"), make(9, correct, "F")
        truth = truth_for(a)
        r1 = paired_diff(a, b, truth, "removal_recall", B=1000, seed=42)
        r2 = paired_diff(list(reversed(a)), b, truth, "removal_recall", B=1000, seed=42)
        self.assertEqual(r1, r2)                  # input order doesn't matter
        self.assertTrue(np.array_equal(arrangement_draws(9, 1000, 42), arrangement_draws(9, 1000, 42)))
        # a different seed gives different draws (percentile CIs of discrete diffs may still coincide)
        self.assertFalse(np.array_equal(arrangement_draws(9, 1000, 42), arrangement_draws(9, 1000, 43)))


class TestUndefined(unittest.TestCase):
    def test_undefined_draws_counted(self) -> None:
        """Only arrangement 0 has present episodes, so FRR is undefined in draws that miss it."""
        a = [x for x in make(3, noisy, "G") if x.arrangement_id == "A00" or x.condition.value == "removed"]
        b = [x for x in make(3, correct, "F") if x.arrangement_id == "A00" or x.condition.value == "removed"]
        truth = truth_for(make(3, correct, "F"))
        B, seed = 2000, 7
        r = paired_diff(a, b, truth, "false_removal_rate", B=B, seed=seed)
        draws = arrangement_draws(3, B, seed)
        expected = int(sum(0 not in d for d in draws))
        self.assertGreater(expected, 0)
        self.assertEqual(r.n_undefined, expected)
        self.assertIsNotNone(r.ci_lo)
        self.assertEqual((r.n_arrangements, r.n_episodes), (3, 6 + 2 + 2))

    def test_all_undefined(self) -> None:
        a = make(3, lambda *_: U, "G")
        b = make(3, lambda *_: U, "F")
        r = paired_diff(a, b, truth_for(a), "resolved_accuracy", B=100, seed=0)
        self.assertIsNone(r.estimate)
        self.assertIsNone(r.ci_lo)
        self.assertIsNone(r.ci_hi)
        self.assertEqual(r.n_undefined, 100)


class TestErrorsAndLabel(unittest.TestCase):
    def test_episode_sets_must_match(self) -> None:
        a, b = make(3, correct, "G"), make(3, correct, "F")
        with self.assertRaises(ValueError):
            paired_diff(a, b[1:], truth_for(a), "coverage", B=10)

    def test_bad_metric_and_B(self) -> None:
        a = make(2, correct, "G")
        with self.assertRaises(ValueError):
            paired_diff(a, a, truth_for(a), "accuracy", B=10)
        with self.assertRaises(ValueError):
            paired_diff(a, a, truth_for(a), "coverage", B=0)

    def test_missing_truth(self) -> None:
        a = make(2, correct, "G")
        with self.assertRaises(KeyError):
            paired_diff(a, a, truth_for(a[1:]), "coverage", B=10)

    def test_synthetic_label(self) -> None:
        a = make(2, correct, "G", synthetic=True)
        d = paired_diff(a, make(2, correct, "F"), truth_for(a), "coverage", B=10).to_dict()
        self.assertTrue(d["synthetic"])
        self.assertEqual(d["label"], SYNTHETIC_LABEL)
        self.assertEqual(d["B"], 10)


class TestStratified(unittest.TestCase):
    """Family-stratified cluster bootstrap (prereg section 8): resample within strata, keeping each stratum's count."""
    # A quota-like design: S8 x3, M16-L x2, M16-R x2, IND28 x1, white_box x2, ycb_box x2 (12 arrangements, unsorted)
    STRATA = ["S8", "M16-L", "white_box", "S8", "M16-R", "ycb_box", "IND28", "M16-L", "S8", "white_box", "M16-R",
              "ycb_box"]

    def test_every_replicate_keeps_every_stratum_count(self) -> None:
        B = 3000
        draws = stratified_arrangement_draws(self.STRATA, B, seed=4)
        self.assertEqual(draws.shape, (B, len(self.STRATA)))
        want = Counter(self.STRATA)
        lab = np.array(self.STRATA)
        for d in draws:
            self.assertEqual(Counter(lab[d]), want)            # every stratum's count, in every replicate
        for j, st in enumerate(self.STRATA):                    # column j only ever holds members of j's stratum
            self.assertTrue(set(lab[draws[:, j]]) == {st})

    def test_members_are_drawn_uniformly_within_their_stratum(self) -> None:
        draws = stratified_arrangement_draws(self.STRATA, 20000, seed=1)
        s8 = [j for j, x in enumerate(self.STRATA) if x == "S8"]
        freq = Counter(draws[:, s8].ravel().tolist())
        self.assertEqual(set(freq), set(s8))
        for j in s8:   # 60000 draws over 3 members: 1/3 each within a few standard errors (0.0019)
            self.assertAlmostEqual(freq[j] / (20000 * 3), 1 / 3, delta=0.01)
        ind = self.STRATA.index("IND28")   # a singleton stratum always resamples itself
        self.assertTrue((draws[:, ind] == ind).all())

    def test_stratum_draws_do_not_depend_on_the_order_arrangements_are_listed_in(self) -> None:
        # Strata are visited in sorted order, so a stratum's draws depend on its name and size, not on where the
        # manifest happens to list its arrangements (first-appearance order would change the stream).
        a = self.STRATA                                    # S8 first
        b = sorted(self.STRATA, reverse=True)              # ycb_box first, strata contiguous
        c = [x for pair in zip(sorted(a)[:6], sorted(a)[6:]) for x in pair]   # interleaved strata
        B, seed = 200, 7
        base = stratified_arrangement_draws(a, B, seed)

        def per_stratum(labels: list[str], draws: np.ndarray) -> dict[str, np.ndarray]:
            out = {}
            for st in set(labels):
                members = [j for j, x in enumerate(labels) if x == st]
                out[st] = np.stack([np.searchsorted(members, draws[:, j]) for j in members], axis=1)
            return out
        want = per_stratum(a, base)
        for labels in (b, c):
            self.assertNotEqual(list(dict.fromkeys(labels)), list(dict.fromkeys(a)))
            got = per_stratum(labels, stratified_arrangement_draws(labels, B, seed))
            self.assertEqual(got.keys(), want.keys())
            for st in want:
                self.assertTrue(np.array_equal(got[st], want[st]), st)

    def test_one_stratum_is_the_unstratified_draw(self) -> None:
        self.assertTrue(np.array_equal(stratified_arrangement_draws(["x"] * 9, 500, 42), arrangement_draws(9, 500, 42)))
        ids = [f"A{i:02d}" for i in range(9)]
        self.assertTrue(np.array_equal(sample_draws(ids, 500, 42, None), arrangement_draws(9, 500, 42)))

    def test_missing_stratum_raises(self) -> None:
        with self.assertRaises(ValueError):
            sample_draws(["A00", "A01"], 10, 0, {"A00": "S8"})

    # The same 12 arrangements with no singleton stratum (each class at least twice)
    STRATA2 = ["S8", "M16-L", "white_box", "S8", "M16-R", "ycb_box", "M16-L", "M16-R", "S8", "white_box", "ycb_box",
               "S8"]

    def test_singleton_strata_fall_back_to_unstratified(self) -> None:
        # Review (checkpoint 2): with every arrangement in a stratum of its own, every stratified replicate is the
        # sample itself and the CI collapsed to the point estimate (4/24, 4/24), still called stratified. A stratum of
        # fewer than two arrangements now makes the bootstrap unstratified, and the result says why.
        a = make(6, lambda i, c, l: R if i == 0 else correct(i, c, l), "G")
        b = make(6, correct, "F")
        u = paired_diff(a, b, truth_for(a), "false_removal_rate", B=500, seed=3)
        self.assertFalse(u.stratified)
        self.assertLess(u.ci_lo, u.ci_hi)
        for strata, singles in (({f"A{i:02d}": f"s{i}" for i in range(6)}, ("s0", "s1", "s2", "s3", "s4", "s5")),
                                ({f"A{i:02d}": "big" if i < 5 else "lone" for i in range(6)}, ("lone",))):
            r = paired_diff(a, b, truth_for(a), "false_removal_rate", B=500, seed=3, strata=strata)
            with self.subTest(singles=singles):
                self.assertAlmostEqual(r.estimate, 4 / 24)
                self.assertFalse(r.stratified)
                self.assertEqual(r.singleton_strata, singles)
                self.assertTrue(r.stratification_note.startswith("unstratified: singleton strata " + singles[0]))
                self.assertEqual((r.ci_lo, r.ci_hi), (u.ci_lo, u.ci_hi))   # exactly the unstratified bootstrap
                self.assertEqual(r.to_dict()["singleton_strata"], list(singles))
        self.assertEqual(resolve_strata(["A", "B", "C", "D"], {"A": "x", "B": "x", "C": "y", "D": "y"}),
                         ({"A": "x", "B": "x", "C": "y", "D": "y"}, (), ""))
        self.assertEqual(resolve_strata(["A", "B"], None), (None, (), ""))
        with self.assertRaises(ValueError):
            resolve_strata(["A", "B"], {"A": "x"})

    def test_paired_diff_uses_the_stratified_draws(self) -> None:
        a, b = make(12, noisy, "G"), make(12, correct, "F")
        truth = truth_for(a)
        strata = {f"A{i:02d}": st for i, st in enumerate(self.STRATA2)}
        r = paired_diff(a, b, truth, "coverage", B=400, seed=11, strata=strata)
        sample = pair(a, b, truth)
        draws = stratified_arrangement_draws([strata[x] for x in sample.arrangement_ids], 400, 11)
        diffs = draw_diffs(sample, truth, "coverage", draws)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        self.assertEqual((r.ci_lo, r.ci_hi), (float(lo), float(hi)))
        self.assertTrue(r.to_dict()["stratified"])
        self.assertEqual((r.singleton_strata, r.stratification_note), ((), ""))
        # and each draw's difference is the metric recomputed on the expanded multiset
        for draw, d in list(zip(draws, diffs))[:25]:
            ra, rb = sample.expand(draw)
            self.assertAlmostEqual(d, metrics_on(ra, truth)["coverage"] - metrics_on(rb, truth)["coverage"], places=12)



class TestRateIntervals(unittest.TestCase):
    """Single-rate percentile intervals (prereg section 8: "for contrasts and for single rates")."""

    def by_hand(self, results, truth, metric, draws) -> tuple[float, float]:
        ids = sorted({r.arrangement_id for r in results})
        vals = []
        for d in draws:
            multiset = [r for i in d for r in results if r.arrangement_id == ids[i]]
            v = metrics_on(multiset, truth)[metric]
            if v is not None:
                vals.append(v)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    def test_matches_a_hand_bootstrap_stratified_and_not(self) -> None:
        a = make(12, noisy, "G")
        truth = truth_for(a)
        strata = {f"A{i:02d}": st for i, st in enumerate(TestStratified.STRATA2)}
        ids = sorted(strata)
        for st in (None, strata):
            iv = rate_intervals(a, truth, B=300, seed=5, strata=st)
            draws = sample_draws(ids, 300, 5, st)
            for metric in ("false_removal_rate", "removal_recall", "coverage"):
                with self.subTest(stratified=st is not None, metric=metric):
                    r = iv[metric]
                    lo, hi = self.by_hand(a, truth, metric, draws)
                    self.assertAlmostEqual(r.ci_lo, lo, places=12)
                    self.assertAlmostEqual(r.ci_hi, hi, places=12)
                    self.assertEqual(r.estimate, metrics_on(a, truth)[metric])
                    self.assertEqual((r.stratified, r.n_arrangements, r.n_episodes, r.B, r.seed, r.policy),
                                     (st is not None, 12, 72, 300, 5, "G"))
                    self.assertLess(r.ci_lo, r.ci_hi)

    def test_seed_averaged_fractions_and_singleton_fallback(self) -> None:
        # R's seed-averaged decision fractions enter as they are (one result per episode; seeds are not resampled)
        half = {R: 0.5, P: 0.5}
        a = make(6, lambda i, c, l: half if i < 3 and c.value == "visible" else correct(i, c, l), "R")
        truth = truth_for(a)
        iv = rate_intervals(a, truth, ("false_removal_rate",), B=400, seed=2)
        self.assertAlmostEqual(iv["false_removal_rate"].estimate, (3 * 2 * 0.5) / 24)
        lo, hi = self.by_hand(a, truth, "false_removal_rate", sample_draws(sorted({r.arrangement_id for r in a}),
                                                                             400, 2))
        self.assertAlmostEqual(iv["false_removal_rate"].ci_lo, lo, places=12)
        self.assertAlmostEqual(iv["false_removal_rate"].ci_hi, hi, places=12)
        lone = {f"A{i:02d}": f"s{i}" for i in range(6)}
        fb = rate_intervals(a, truth, ("false_removal_rate",), B=400, seed=2, strata=lone)["false_removal_rate"]
        self.assertFalse(fb.stratified)
        self.assertEqual(len(fb.singleton_strata), 6)
        self.assertEqual((fb.ci_lo, fb.ci_hi), (iv["false_removal_rate"].ci_lo, iv["false_removal_rate"].ci_hi))
        with self.assertRaises(ValueError):   # one result per episode
            rate_intervals(a + a[:1], truth, B=10)


if __name__ == "__main__":
    unittest.main()
