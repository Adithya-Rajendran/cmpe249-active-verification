"""Known-answer tests for analysis/metrics.py (hand-computed; inputs built with schema types)."""
from __future__ import annotations

import unittest

from analysis.metrics import METRICS, compute, compute_all
from verify.schema import (
    SYNTHETIC_LABEL,
    Condition,
    Decision,
    EpisodeLog,
    EpisodeResult,
    GroundTruth,
    Lighting,
    Split,
    episode_id,
    truth_from_condition,
)

P, R, U = Decision.PRESENT, Decision.REMOVED, Decision.UNKNOWN


def res(arr: str, cond: str, light: str, frac: dict, *, views: float = 2.0, travel: float = 23.56,
        policy: str = "G", budget: int = 2, split: str = "test", synthetic: bool = False) -> EpisodeResult:
    full = {d: float(frac.get(d, 0.0)) for d in Decision}
    return EpisodeResult(episode_id=episode_id(arr, cond, light), arrangement_id=arr, split=Split(split),
                         condition=Condition(cond), lighting=Lighting(light), policy=policy,
                         budget_additional_views=budget, travel_cap_cm=None, decision_frac=full,
                         mean_views=views, mean_additional_views=views - 1, mean_travel_cm=travel,
                         seeds=(None,), synthetic=synthetic)


def truth_for(results, indistinguishable=(), synthetic=False) -> GroundTruth:
    return GroundTruth({r.episode_id: truth_from_condition(r.condition) for r in results},
                       frozenset(indistinguishable), synthetic)


def log(arr: str, cond: str, light: str, decision: Decision, seed: int, n_views: int = 3,
        travel: float = 47.12) -> EpisodeLog:
    return EpisodeLog(episode_id=episode_id(arr, cond, light), arrangement_id=arr, split=Split.TEST,
                      target_id="t", condition=Condition(cond), lighting=Lighting(light), model_id="m",
                      config_version="v", config_sha256="x", policy="R", budget_additional_views=2,
                      travel_cap_cm=None, seed=seed, requested_stops=tuple(range(n_views)), evidence=(),
                      decision=decision, travel_cm=travel, n_views=n_views, n_additional_views=n_views - 1,
                      synthetic=True)


# Two arrangements x 6 episodes, decisions chosen by hand:
#   A01: vis-L1 P, vis-L2 P, occ-L1 U, occ-L2 R (false removal), rem-L1 R, rem-L2 U
#   A02: vis-L1 P, vis-L2 U, occ-L1 P, occ-L2 P, rem-L1 R, rem-L2 P (wrong)
TABLE = [("A01", "visible", "L1", P), ("A01", "visible", "L2", P), ("A01", "occluded", "L1", U),
         ("A01", "occluded", "L2", R), ("A01", "removed", "L1", R), ("A01", "removed", "L2", U),
         ("A02", "visible", "L1", P), ("A02", "visible", "L2", U), ("A02", "occluded", "L1", P),
         ("A02", "occluded", "L2", P), ("A02", "removed", "L1", R), ("A02", "removed", "L2", P)]


def table_results() -> list[EpisodeResult]:
    out = []
    for i, (a, c, l, d) in enumerate(TABLE):
        views = 4.0 if i == 0 else 2.0
        out.append(res(a, c, l, {d: 1.0}, views=views, travel=10.0 * i))
    return out


class TestOverall(unittest.TestCase):
    def setUp(self) -> None:
        self.results = table_results()
        self.truth = truth_for(self.results)
        self.m = compute(self.results, self.truth)

    def test_counts(self) -> None:
        m = self.m
        self.assertEqual((m["n_arrangements"], m["n_episodes"], m["n_present"], m["n_removed"],
                          m["n_indistinguishable"]), (2, 12, 8, 4, 0))

    def test_rates(self) -> None:
        m = self.m
        self.assertAlmostEqual(m["false_removal_rate"], 1 / 8)   # 1 removed decision on 8 present
        self.assertAlmostEqual(m["removal_recall"], 2 / 4)       # 2 of 4 removed episodes
        self.assertAlmostEqual(m["coverage"], 9 / 12)            # 3 unknown stay in the denominator
        self.assertAlmostEqual(m["resolved_accuracy"], 7 / 9)    # 5 correct present + 2 correct removed
        self.assertAlmostEqual(m["max_achievable_coverage"], 1.0)
        self.assertTrue(all(m["defined"][k] for k in METRICS))

    def test_effort(self) -> None:
        m = self.m
        self.assertAlmostEqual(m["mean_views"], (4 + 11 * 2) / 12)
        self.assertAlmostEqual(m["mean_additional_views"], (3 + 11 * 1) / 12)
        self.assertAlmostEqual(m["mean_travel_cm"], 10.0 * sum(range(12)) / 12)

    def test_confusion(self) -> None:
        self.assertEqual(self.m["confusion"], {"present": {"present": 5.0, "removed": 1.0, "unknown": 2.0},
                                               "removed": {"present": 1.0, "removed": 2.0, "unknown": 1.0}})

    def test_provenance(self) -> None:
        m = self.m
        self.assertEqual((m["policy"], m["budget_additional_views"], m["travel_cap_cm"]), ("G", 2, None))
        self.assertFalse(m["synthetic"])
        self.assertNotIn("label", m)


class TestGrouping(unittest.TestCase):
    def setUp(self) -> None:
        self.results = table_results()
        self.truth = truth_for(self.results)

    def test_by_condition(self) -> None:
        g = compute(self.results, self.truth, group_by="condition")
        self.assertEqual(list(g), ["visible", "occluded", "removed"])
        for v in g.values():
            self.assertEqual((v["n_arrangements"], v["n_episodes"]), (2, 4))
        vis, occ, rem = g["visible"], g["occluded"], g["removed"]
        self.assertEqual(vis["false_removal_rate"], 0.0)
        self.assertIsNone(vis["removal_recall"])
        self.assertFalse(vis["defined"]["removal_recall"])
        self.assertAlmostEqual(vis["coverage"], 3 / 4)
        self.assertAlmostEqual(vis["resolved_accuracy"], 1.0)
        self.assertAlmostEqual(occ["false_removal_rate"], 1 / 4)
        self.assertAlmostEqual(occ["coverage"], 3 / 4)
        self.assertAlmostEqual(occ["resolved_accuracy"], 2 / 3)
        # removed-only group: FRR is undefined (None), never 0
        self.assertIsNone(rem["false_removal_rate"])
        self.assertFalse(rem["defined"]["false_removal_rate"])
        self.assertEqual((rem["n_present"], rem["n_removed"]), (0, 4))
        self.assertAlmostEqual(rem["removal_recall"], 2 / 4)
        self.assertAlmostEqual(rem["coverage"], 3 / 4)
        self.assertAlmostEqual(rem["resolved_accuracy"], 2 / 3)

    def test_by_lighting(self) -> None:
        g = compute(self.results, self.truth, group_by="lighting")
        self.assertEqual(list(g), ["L1", "L2"])
        l1, l2 = g["L1"], g["L2"]
        for v in (l1, l2):
            self.assertEqual((v["n_arrangements"], v["n_episodes"], v["n_present"], v["n_removed"]),
                             (2, 6, 4, 2))
        self.assertEqual(l1["false_removal_rate"], 0.0)
        self.assertAlmostEqual(l1["removal_recall"], 1.0)
        self.assertAlmostEqual(l1["coverage"], 5 / 6)
        self.assertAlmostEqual(l1["resolved_accuracy"], 1.0)
        self.assertAlmostEqual(l2["false_removal_rate"], 1 / 4)
        self.assertEqual(l2["removal_recall"], 0.0)
        self.assertAlmostEqual(l2["coverage"], 4 / 6)
        self.assertAlmostEqual(l2["resolved_accuracy"], 2 / 4)

    def test_only_present_groups_and_split(self) -> None:
        sub = [r for r in self.results if r.lighting is Lighting.L2]
        self.assertEqual(list(compute(sub, truth_for(sub), group_by="lighting")), ["L2"])
        self.assertEqual(list(compute(self.results, self.truth, group_by="split")), ["test"])

    def test_compute_all(self) -> None:
        a = compute_all(self.results, self.truth)
        self.assertEqual(set(a), {"overall", "by_condition", "by_lighting"})
        self.assertAlmostEqual(a["overall"]["coverage"], 9 / 12)
        self.assertEqual(a["by_lighting"]["L1"]["n_episodes"], 6)

    def test_bad_group_key(self) -> None:
        with self.assertRaises(ValueError):
            compute(self.results, self.truth, group_by="arrangement")


class TestFractional(unittest.TestCase):
    def test_r_quarter_removed(self) -> None:
        r = [res("A01", "visible", "L1", {R: 0.25, P: 0.75})]
        m = compute(r, truth_for(r))
        self.assertAlmostEqual(m["false_removal_rate"], 0.25)
        self.assertAlmostEqual(m["coverage"], 1.0)
        self.assertAlmostEqual(m["resolved_accuracy"], 0.75)
        self.assertAlmostEqual(m["confusion"]["present"]["removed"], 0.25)

    def test_from_r_logs(self) -> None:
        """Four seeds on one present episode, one REMOVED: averaged within the episode -> FRR 0.25."""
        logs = [log("A01", "occluded", "L1", d, s) for s, d in enumerate((R, P, P, P))]
        logs += [log("A01", "removed", "L1", d, s, n_views=5, travel=94.25) for s, d in enumerate((R, R, U, U))]
        truth = GroundTruth({episode_id("A01", "occluded", "L1"): "present",
                             episode_id("A01", "removed", "L1"): "removed"})
        m = compute(logs, truth)
        self.assertEqual(m["n_episodes"], 2)                    # seeds add no samples
        self.assertAlmostEqual(m["false_removal_rate"], 0.25)
        self.assertAlmostEqual(m["removal_recall"], 0.5)
        self.assertAlmostEqual(m["coverage"], (1.0 + 0.5) / 2)
        self.assertAlmostEqual(m["resolved_accuracy"], (0.75 + 0.5) / 1.5)
        self.assertAlmostEqual(m["mean_views"], 4.0)
        self.assertAlmostEqual(m["mean_travel_cm"], (47.12 + 94.25) / 2)
        self.assertTrue(m["synthetic"])
        self.assertEqual(m["label"], SYNTHETIC_LABEL)

    def test_mixed_unknown(self) -> None:
        r = [res("A01", "removed", "L1", {R: 0.5, U: 0.5}), res("A01", "visible", "L1", {P: 1.0})]
        m = compute(r, truth_for(r))
        self.assertAlmostEqual(m["coverage"], 0.75)
        self.assertAlmostEqual(m["resolved_accuracy"], 1.0)
        self.assertAlmostEqual(m["removal_recall"], 0.5)


class TestUndefinedAndErrors(unittest.TestCase):
    def test_all_unknown(self) -> None:
        r = [res(a, c, l, {U: 1.0}) for a, c, l, _ in TABLE]
        m = compute(r, truth_for(r))
        self.assertEqual(m["coverage"], 0.0)
        self.assertIsNone(m["resolved_accuracy"])
        self.assertFalse(m["defined"]["resolved_accuracy"])
        self.assertEqual(m["false_removal_rate"], 0.0)
        self.assertEqual(m["removal_recall"], 0.0)
        self.assertTrue(m["defined"]["coverage"])

    def test_indistinguishable(self) -> None:
        r = table_results()
        ind = [episode_id("A02", "occluded", "L1"), episode_id("A02", "removed", "L2")]
        m = compute(r, truth_for(r, ind, synthetic=True))
        self.assertEqual(m["n_indistinguishable"], 2)
        self.assertAlmostEqual(m["max_achievable_coverage"], 10 / 12)
        self.assertTrue(m["synthetic"])

    def test_mixed_policy_or_budget(self) -> None:
        r = table_results()
        other = res("A03", "visible", "L1", {P: 1.0}, policy="F")
        with self.assertRaises(ValueError):
            compute(r + [other], truth_for(r + [other]))
        other = res("A03", "visible", "L1", {P: 1.0}, budget=3)
        with self.assertRaises(ValueError):
            compute(r + [other], truth_for(r + [other]))

    def test_missing_truth(self) -> None:
        r = table_results()
        with self.assertRaises(KeyError):
            compute(r, truth_for(r[1:]))
        with self.assertRaises(KeyError):
            compute(r, truth_for(r[1:]), group_by="condition")

    def test_duplicate_and_empty(self) -> None:
        r = table_results()
        with self.assertRaises(ValueError):
            compute(r + r[:1], truth_for(r))
        with self.assertRaises(ValueError):
            compute([], truth_for(r))

    def test_bad_fractions(self) -> None:
        r = [res("A01", "visible", "L1", {P: 0.5, R: 0.2})]
        with self.assertRaises(ValueError):
            compute(r, truth_for(r))


if __name__ == "__main__":
    unittest.main()
