"""Tests for verify/synthetic.py (SYNTHETIC generator)."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from verify.schema import (
    Condition,
    EpisodeSpec,
    EvidenceRecord,
    GroundTruth,
    Lighting,
    Positions,
    Split,
    Thresholds,
    TruthState,
    episode_id,
)
from verify.synthetic import (
    SyntheticParams,
    arrangement_ids,
    generate,
    generate_layouts,
)

N, SEED = 30, 7


def _dump(store) -> list[tuple[str, EvidenceRecord]]:
    return list(store.items())


class TestSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs, cls.store, cls.truth = generate(N, SEED)
        cls.layouts = generate_layouts(N, SEED)
        cls.n_stops = Positions.default().n_stops

    def test_determinism_by_seed(self) -> None:
        specs, store, truth = generate(N, SEED)
        self.assertEqual(specs, self.specs)
        self.assertEqual(_dump(store), _dump(self.store))
        self.assertEqual(truth, self.truth)
        self.assertEqual(generate_layouts(N, SEED), self.layouts)
        _, other, _ = generate(N, SEED + 1)
        self.assertNotEqual(_dump(other), _dump(self.store))

    def test_ids_and_six_episodes_per_arrangement(self) -> None:
        self.assertEqual(arrangement_ids(3), ["SYN01", "SYN02", "SYN03"])
        self.assertEqual(arrangement_ids(100)[0], "SYN001")
        self.assertEqual(len(self.specs), 6 * N)
        per: dict[str, set] = {}
        for s in self.specs:
            per.setdefault(s.arrangement_id, set()).add((s.condition, s.lighting))
        self.assertEqual(len(per), N)
        for a, keys in per.items():
            self.assertEqual(len(keys), 6, a)
        self.assertEqual(set(self.store.episode_ids()), {s.episode_id for s in self.specs})

    def test_spec_order_and_split(self) -> None:
        cr = {c: i for i, c in enumerate(Condition)}
        lr = {l: i for i, l in enumerate(Lighting)}
        keys = [(s.arrangement_id, cr[s.condition], lr[s.lighting]) for s in self.specs]
        self.assertEqual(keys, sorted(keys))
        split_of: dict[str, Split] = {}
        for s in self.specs:
            self.assertEqual(split_of.setdefault(s.arrangement_id, s.split), s.split)  # never mixed
        self.assertEqual(sum(v is Split.DEV for v in split_of.values()), 10)  # round(30 / 3)

    def test_layouts_contiguous_and_include_start(self) -> None:
        p = SyntheticParams()
        for a, lay in self.layouts.items():
            b = lay.blocked
            self.assertEqual(list(b), list(range(b[0], b[0] + len(b))), a)
            self.assertIn(2, b)
            if lay.indistinguishable:
                self.assertEqual(b, tuple(range(self.n_stops)))
            else:
                self.assertIn(len(b), p.block_widths)
        free = generate_layouts(200, 3, SyntheticParams(block_includes_start=False, indistinguishable_frac=0.0))
        self.assertTrue(any(2 not in lay.blocked for lay in free.values()))

    def test_matched_layout_occluded_vs_removed(self) -> None:
        p = SyntheticParams()
        for a, lay in self.layouts.items():
            for light in Lighting:
                occ = [self.store.get(episode_id(a, Condition.OCCLUDED, light), s) for s in range(self.n_stops)]
                rem = [self.store.get(episode_id(a, Condition.REMOVED, light), s) for s in range(self.n_stops)]
                vis = [self.store.get(episode_id(a, Condition.VISIBLE, light), s) for s in range(self.n_stops)]
                self.assertEqual([r.obstruction for r in occ], [r.obstruction for r in rem])
                for s in range(self.n_stops):
                    if s in lay.blocked:
                        self.assertGreaterEqual(occ[s].obstruction, p.blocked_obstruction[0])
                    self.assertLessEqual(vis[s].obstruction, p.clear_obstruction[1])

    def test_indistinguishable_subset(self) -> None:
        ind = [a for a, lay in self.layouts.items() if lay.indistinguishable]
        self.assertEqual(len(ind), 3)  # 10% of 30
        expected = {episode_id(a, c, l) for a in ind for c in (Condition.OCCLUDED, Condition.REMOVED)
                    for l in Lighting}
        self.assertEqual(self.truth.indistinguishable, expected)
        th = Thresholds()
        for eid in expected:  # no strong match and no clear miss anywhere
            for s in range(self.n_stops):
                r = self.store.get(eid, s)
                self.assertGreater(r.obstruction, th.tau_occ)
        self.assertEqual(sum(l.indistinguishable for l in generate_layouts(3, 0).values()), 1)
        self.assertEqual(sum(l.indistinguishable for l in generate_layouts(2, 0).values()), 0)

    def test_indistinguishable_only_in_truth(self) -> None:
        names = {f.name for f in dataclasses.fields(EpisodeSpec)} | {f.name for f in dataclasses.fields(EvidenceRecord)}
        for n in names:
            self.assertNotIn("indisting", n)
            self.assertNotIn("truth", n)
        self.assertEqual({f.name for f in dataclasses.fields(EvidenceRecord)},
                         {"stop", "match_score", "quality", "obstruction"})

    def test_truth_not_reachable_from_specs_or_store(self) -> None:
        for s in self.specs:
            for v in dataclasses.asdict(s).values():
                self.assertNotIsInstance(v, (GroundTruth, TruthState))
        for n in type(self.store).__slots__:  # slotted store: check every slot, mangled ones included
            attr = f"_EvidenceStore{n}" if n.startswith("__") else n
            self.assertNotIsInstance(getattr(self.store, attr), (GroundTruth, TruthState))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ev.jsonl"
            self.store.to_jsonl(path)
            for line in path.read_text().splitlines():
                self.assertEqual(set(json.loads(line)),
                                 {"episode_id", "stop", "match_score", "quality", "obstruction", "synthetic"})
        self.assertIsInstance(self.truth, GroundTruth)
        for s in self.specs:
            want = TruthState.REMOVED if s.condition is Condition.REMOVED else TruthState.PRESENT
            self.assertEqual(self.truth[s.episode_id], want)

    def test_values_in_unit_interval(self) -> None:
        for _, r in self.store.items():
            for v in (r.match_score, r.quality, r.obstruction):
                self.assertTrue(0.0 <= v <= 1.0)
        wide = SyntheticParams(match_present=(0.9, 0.5), quality_L2=(0.5, 0.6), match_absent=(0.05, 0.4))
        _, st, _ = generate(10, 1, wide)
        for _, r in st.items():
            for v in (r.match_score, r.quality, r.obstruction):
                self.assertTrue(0.0 <= v <= 1.0)

    def test_l2_lowers_quality_and_signal_separation(self) -> None:
        q = {l: [] for l in Lighting}
        m = {c: [] for c in Condition}
        for s in self.specs:
            for st in range(self.n_stops):
                r = self.store.get(s.episode_id, st)
                q[s.lighting].append(r.quality)
                m[s.condition].append(r.match_score)
        self.assertLess(np.mean(q[Lighting.L2]), np.mean(q[Lighting.L1]))
        self.assertGreater(np.mean(m[Condition.VISIBLE]), np.mean(m[Condition.REMOVED]) + 0.4)
        # every distinguishable occluded episode has an unblocked stop where the target shows
        for a, lay in self.layouts.items():
            if not lay.indistinguishable:
                self.assertLess(len(lay.blocked), self.n_stops)

    def test_labelling_and_unsealed(self) -> None:
        self.assertFalse(self.store.sealed)
        self.assertTrue(self.store.synthetic)
        self.assertEqual(self.store.source, f"verify.synthetic seed={SEED}")
        self.assertTrue(self.truth.synthetic)
        import verify.synthetic as mod
        self.assertIn("SYNTHETIC", mod.__doc__)

    def test_bad_params(self) -> None:
        with self.assertRaises(ValueError):
            SyntheticParams(block_width_probs=(0.5, 0.5, 0.5, 0.5))
        with self.assertRaises(ValueError):
            generate_layouts(5, 0, SyntheticParams(block_widths=(5,), block_width_probs=(1.0,)))


if __name__ == "__main__":
    unittest.main()
