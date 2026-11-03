"""Tests for verify/manifest.py (stdlib unittest). Run from the repo root:
uv run --no-sync python -m unittest tests.test_manifest -v
"""
import dataclasses
import json
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


class TestQuotaSlotsAndSimulation(unittest.TestCase):
    """Checkpoint 2: occluder_family / m16_side / design_case, the top-level simulation flag, the sim view labels."""

    def test_slot_fields_validate_and_give_strata(self):
        R = m.ArrangementRecord
        self.assertEqual(R("SD01", "t", Split.DEV, occluder_family="M16", m16_side="L").stratum, "M16-L")
        self.assertEqual(R("SD02", "t", Split.DEV, occluder_family="S8").stratum, "S8")
        self.assertEqual(R("DES1", "t", Split.DEV, occluder_family="white_box", design_case=True).stratum,
                         "design:white_box")
        self.assertIsNone(R("SD03", "t", Split.DEV).stratum)
        for bad in (dict(occluder_family="M16"), dict(occluder_family="M16", m16_side="X"),
                    dict(occluder_family="S8", m16_side="L"), dict(m16_side="R"), dict(occluder_family="box")):
            with self.subTest(bad=bad), self.assertRaises(ManifestError):
                R("SD04", "t", Split.DEV, **bad)
        with self.assertRaises(ManifestError):   # a JSON string "false" must not read as True
            R.from_dict({"arrangement_id": "SD05", "target_id": "t", "split": "dev", "design_case": "false"})

    def test_strata_need_every_arrangement(self):
        R = m.ArrangementRecord
        full = [R("SD01", "t", Split.DEV, occluder_family="M16", m16_side="R"), R("SD02", "t", Split.DEV,
                                                                                   occluder_family="IND28")]
        self.assertEqual(m.bootstrap_strata(full), {"SD01": "M16-R", "SD02": "IND28"})
        self.assertIsNone(m.bootstrap_strata([*full, R("SD03", "t", Split.DEV)]))   # partial info: unstratified
        self.assertIsNone(m.bootstrap_strata([]))

    def test_round_trip_and_legacy_shape(self):
        R = m.ArrangementRecord
        arrs = [R("SD01", "t", Split.DEV, occluder_family="M16", m16_side="L"), R("SD02", "t", Split.DEV,
                                                                                   occluder_family="ycb_box")]
        legacy = [R("A01", "t", Split.DEV), R("A02", "t", Split.TEST)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            m.save_manifest(arrs, p, simulation=True)
            info = m.load_manifest_info(p)
            self.assertEqual((info.arrangements, info.simulation), (tuple(arrs), True))
            self.assertEqual(info.strata(), {"SD01": "M16-L", "SD02": "ycb_box"})
            m.save_manifest(legacy, p)
            doc = json.loads(p.read_text())
            self.assertNotIn("simulation", doc)
            self.assertEqual(set(doc["arrangements"][0]), {"arrangement_id", "target_id", "split", "lightings",
                                                           "indistinguishable"})   # the legacy shape is unchanged
            self.assertFalse(m.load_manifest_info(p).simulation)
            doc["simulation"] = "true"
            p.write_text(json.dumps(doc))
            with self.assertRaises(ManifestError):
                m.load_manifest_info(p)

    def test_geometric_labels_loader(self):
        def doc(**over):
            stops = [{"stop": i, "diagnostic": i != 2} for i in range(5)]
            e = {"arrangement_id": "SD01", "condition": "occluded", "lighting": "L2", "truth": "present",
                 "stops": stops, **over}
            return {"schema": m.SIM_VIEW_LABELS_SCHEMA, "simulation": True, "episodes": {"SD01-occluded-L2": e}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.json"
            p.write_text(json.dumps(doc()))
            g = m.load_geometric_labels(p, 5)
            self.assertEqual(g.episodes["SD01-occluded-L2"], (True, True, False, True, True))
            self.assertEqual(g.no_diagnostic_view(), {"SD01-occluded-L2": False})
            for bad in (dict(lighting="L1"), dict(truth="removed"),
                        dict(stops=[{"stop": i, "diagnostic": True} for i in range(4)]),
                        dict(stops=[{"stop": i, "diagnostic": 1} for i in range(5)])):
                p.write_text(json.dumps(doc(**bad)))
                with self.subTest(bad=list(bad)), self.assertRaises(ManifestError):
                    m.load_geometric_labels(p, 5)
            p.write_text(json.dumps({**doc(), "schema": "cmpe249.view_labels.v1"}))
            with self.assertRaises(ManifestError):
                m.load_geometric_labels(p, 5)
