"""Tests for verify/manifest.py (stdlib unittest). Run from the repo root:
uv run --no-sync python -m unittest tests.test_manifest -v
"""
import dataclasses
import tempfile
import unittest
from pathlib import Path

from verify import manifest as m
from verify.schema import (
    Condition,
    EpisodeSpec,
    GroundTruth,
    Lighting,
    ManifestError,
    Split,
    TruthState,
    episode_id,
)


def arrs(n: int = 6, n_dev: int = 2, seed: int = 0) -> list[m.ArrangementRecord]:
    ids = [f"A{i:02d}" for i in range(1, n + 1)]
    splits = m.assign_splits(ids, n_dev, seed)
    return [m.ArrangementRecord(a, f"mug-{a}", splits[a], indistinguishable=(a == "A02")) for a in ids]


class TestEpisodes(unittest.TestCase):
    def test_six_episodes_per_arrangement(self):
        specs, truth = m.episodes_from_arrangements(arrs())
        self.assertEqual(len(specs), 36)
        for a in {s.arrangement_id for s in specs}:
            got = {(s.condition, s.lighting) for s in specs if s.arrangement_id == a}
            self.assertEqual(len(got), 6)
            self.assertEqual(got, {(c, l) for c in Condition for l in Lighting})

    def test_l1_only_gives_three(self):
        specs, _ = m.episodes_from_arrangements([m.ArrangementRecord("P01", "t", Split.PILOT, (Lighting.L1,))])
        self.assertEqual([s.condition for s in specs], list(Condition))

    def test_order_and_truth(self):
        specs, truth = m.episodes_from_arrangements(arrs(2, 1))
        self.assertEqual([s.episode_id for s in specs[:6]],
                         ["A01-visible-L1", "A01-visible-L2", "A01-occluded-L1", "A01-occluded-L2",
                          "A01-removed-L1", "A01-removed-L2"])
        self.assertIs(truth["A01-occluded-L2"], TruthState.PRESENT)
        self.assertIs(truth["A01-removed-L1"], TruthState.REMOVED)
        self.assertEqual(truth.indistinguishable, {"A02-occluded-L1", "A02-occluded-L2",
                                                   "A02-removed-L1", "A02-removed-L2"})

    def test_spec_has_no_truth(self):
        names = {f.name for f in dataclasses.fields(EpisodeSpec)}
        self.assertFalse(names & {"truth", "ground_truth", "state"})


class TestSplit(unittest.TestCase):
    def test_split_never_crosses_arrangements(self):
        for seed in range(20):
            specs, _ = m.episodes_from_arrangements(arrs(30, 10, seed))
            by_arr = {}
            for s in specs:
                by_arr.setdefault(s.arrangement_id, set()).add(s.split)
            self.assertTrue(all(len(v) == 1 for v in by_arr.values()))
            dev = {s.arrangement_id for s in m.filter_split(specs, Split.DEV)}
            test = {s.arrangement_id for s in m.filter_split(specs, Split.TEST)}
            self.assertFalse(dev & test)
            self.assertEqual(len(dev), 10)
            self.assertEqual(len(test), 20)

    def test_assign_splits_deterministic(self):
        ids = [f"A{i:02d}" for i in range(1, 31)]
        self.assertEqual(m.assign_splits(ids, 10, 3), m.assign_splits(list(reversed(ids)), 10, 3))

    def test_summary(self):
        specs, _ = m.episodes_from_arrangements(arrs(30, 10))
        self.assertEqual(m.summary(specs), {"n_arrangements": 30, "n_episodes": 180, "n_dev_arrangements": 10,
                                            "n_test_arrangements": 20, "n_dev_episodes": 60,
                                            "n_test_episodes": 120})


class TestValidate(unittest.TestCase):
    def setUp(self):
        self.specs, self.truth = m.episodes_from_arrangements(arrs(3, 1))

    def bad(self, specs, truth=None, msg=""):
        with self.assertRaises(ManifestError) as cm:
            m.validate(specs, truth)
        self.assertIn(msg, str(cm.exception))

    def test_ok(self):
        m.validate(self.specs, self.truth)

    def test_duplicate(self):
        self.bad(self.specs + [self.specs[0]], msg="duplicate")

    def test_mixed_split(self):
        s = list(self.specs)
        other = Split.TEST if s[0].split is Split.DEV else Split.DEV
        s[0] = dataclasses.replace(s[0], split=other)
        self.bad(s, msg="more than one split")

    def test_target_differs(self):
        s = list(self.specs)
        s[1] = dataclasses.replace(s[1], target_id="other")
        self.bad(s, msg="target_id")

    def test_incomplete(self):
        self.bad(self.specs[1:], msg="incomplete")

    def test_uneven_lighting_sets(self):
        s = [x for x in self.specs if not (x.arrangement_id == "A01" and x.condition is Condition.REMOVED
                                           and x.lighting is Lighting.L2)]
        self.bad(s, msg="incomplete")

    def test_truth_keys(self):
        t = GroundTruth({k: v for k, v in self.truth.states.items() if k != self.specs[0].episode_id})
        self.bad(self.specs, t, msg="truth keys")

    def test_truth_disagrees(self):
        st = dict(self.truth.states)
        st["A01-occluded-L1"] = TruthState.REMOVED
        self.bad(self.specs, GroundTruth(st), msg="condition says")

    def test_visible_indistinguishable(self):
        t = GroundTruth(self.truth.states, frozenset({episode_id("A01", "visible", "L1")}))
        self.bad(self.specs, t, msg="visible")

    def test_bad_regex(self):
        s = [dataclasses.replace(x, arrangement_id="a01") if x.arrangement_id == "A01" else x for x in self.specs]
        self.bad(s, msg="bad arrangement id")
        with self.assertRaises(ManifestError):
            m.ArrangementRecord("1AB", "t", Split.DEV)


class TestRoundTrip(unittest.TestCase):
    def test_save_load_identity(self):
        a = arrs(8, 3, seed=5)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "arrangements.json"
            m.save_manifest(a, p)
            self.assertEqual(m.load_arrangements(p), a)
            specs, truth = m.load_manifest(p)
        specs0, truth0 = m.episodes_from_arrangements(a)
        self.assertEqual(specs, specs0)
        self.assertEqual(truth, truth0)

    def test_wrong_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text('{"schema": "nope", "arrangements": []}')
            with self.assertRaises(ManifestError):
                m.load_manifest(p)


if __name__ == "__main__":
    unittest.main()
