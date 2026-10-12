"""Truth-table tests for verify/stopping.py (contract: verify/README.md section 1)."""
from __future__ import annotations

import itertools
import unittest

from verify.schema import (
    CONTINUE,
    Continue,
    Decision,
    Observation,
    Positions,
    Thresholds,
)
from verify.stopping import StoppingRule

POS = Positions.default()
TH = Thresholds()  # defaults 0.6 / 0.5 / 0.3, k = 2

P, R, U, C = Decision.PRESENT, Decision.REMOVED, Decision.UNKNOWN, CONTINUE


def obs(*views: tuple[int, float, float, float]) -> tuple[Observation, ...]:
    """Build observations from (stop, match, quality, obstruction) in request order."""
    return tuple(Observation(step=i, stop=s, azimuth_deg=POS.azimuth(s), match_score=m,
                             quality=q, obstruction=b) for i, (s, m, q, b) in enumerate(views))


# Per-view kinds as (match, quality, obstruction) under the default thresholds.
KINDS: dict[str, tuple[float, float, float]] = {
    "S": (0.90, 0.90, 0.10),  # strong match, clean view
    "X": (0.70, 0.20, 0.90),  # strong match despite low quality and heavy obstruction
    "C": (0.10, 0.90, 0.10),  # clear miss
    "O": (0.10, 0.90, 0.80),  # miss, flagged occluded (not a clear miss)
    "Q": (0.10, 0.20, 0.10),  # miss, low quality (not a clear miss)
}


def kobs(spec: str) -> tuple[Observation, ...]:
    """'C2 O0' -> clear miss at stop 2 then occluded miss at stop 0."""
    return obs(*((int(t[1:]), *KINDS[t[0]]) for t in spec.split()))


# ---------------------------------------------------------------- README truth table (verbatim)

README_TABLE = [
    # (#, obs, k, can_continue, expected)
    (1, [(2, 0.60, 0.9, 0.0)], 2, True, P),
    (2, [(2, 0.59, 0.9, 0.0)], 2, True, C),
    (3, [(2, 0.59, 0.9, 0.0)], 2, False, U),
    (4, [(2, 0.1, 0.9, 0.1), (0, 0.1, 0.9, 0.1)], 2, True, R),
    (5, [(2, 0.1, 0.9, 0.1), (2, 0.1, 0.9, 0.1)], 2, False, U),
    (6, [(2, 0.1, 0.9, 0.1), (0, 0.1, 0.9, 0.1), (4, 0.7, 0.9, 0.9)], 2, False, P),
    (7, [(2, 0.1, 0.49, 0.1), (0, 0.1, 0.9, 0.1)], 2, False, U),
    (8, [(2, 0.1, 0.5, 0.3), (0, 0.1, 0.9, 0.1)], 2, False, R),
    (9, [(2, 0.1, 0.9, 0.31), (0, 0.1, 0.9, 0.1)], 2, True, C),
    (10, [(2, 0.1, 0.9, 0.1)], 1, True, R),
    (11, [(2, 0.1, 0.9, 0.9)], 1, True, C),
    (12, [(2, 0.1, 0.9, 0.9)], 1, False, U),
]

# ---------------------------------------------------------------- kind truth table (explicit)
# (views, k, expected if can_continue=True, expected if can_continue=False)

KIND_TABLE = [
    # one view
    ("S2", 1, P, P), ("S2", 2, P, P), ("S2", 3, P, P),
    ("X2", 1, P, P), ("X2", 2, P, P), ("X2", 3, P, P),
    ("C2", 1, R, R), ("C2", 2, C, U), ("C2", 3, C, U),
    ("O2", 1, C, U), ("O2", 2, C, U), ("O2", 3, C, U),
    ("Q2", 1, C, U), ("Q2", 2, C, U), ("Q2", 3, C, U),
    # two views, distinct stops
    ("C2 C0", 1, R, R), ("C2 C0", 2, R, R), ("C2 C0", 3, C, U),
    ("C2 O0", 1, R, R), ("C2 O0", 2, C, U), ("C2 O0", 3, C, U),
    ("O2 C0", 1, R, R), ("O2 C0", 2, C, U), ("O2 C0", 3, C, U),
    ("C2 Q0", 1, R, R), ("C2 Q0", 2, C, U), ("C2 Q0", 3, C, U),
    ("Q2 C0", 1, R, R), ("Q2 C0", 2, C, U), ("Q2 C0", 3, C, U),
    ("O2 O0", 1, C, U), ("O2 O0", 2, C, U), ("O2 O0", 3, C, U),
    ("O2 Q0", 1, C, U), ("O2 Q0", 2, C, U), ("O2 Q0", 3, C, U),
    ("Q2 Q0", 1, C, U), ("Q2 Q0", 2, C, U), ("Q2 Q0", 3, C, U),
    ("C2 S0", 1, P, P), ("C2 S0", 2, P, P), ("C2 S0", 3, P, P),
    ("S2 C0", 1, P, P), ("S2 C0", 2, P, P), ("S2 C0", 3, P, P),
    ("O2 S0", 1, P, P), ("O2 S0", 2, P, P), ("O2 S0", 3, P, P),
    ("Q2 X0", 1, P, P), ("Q2 X0", 2, P, P), ("Q2 X0", 3, P, P),
    ("S2 S0", 1, P, P), ("S2 S0", 2, P, P), ("S2 S0", 3, P, P),
    # two views, same stop (clear misses at one stop count once)
    ("C2 C2", 1, R, R), ("C2 C2", 2, C, U), ("C2 C2", 3, C, U),
    ("O2 C2", 1, R, R), ("O2 C2", 2, C, U), ("O2 C2", 3, C, U),
    ("C2 O2", 1, R, R), ("C2 O2", 2, C, U), ("C2 O2", 3, C, U),
    ("O2 O2", 1, C, U), ("O2 O2", 2, C, U), ("O2 O2", 3, C, U),
    ("C2 S2", 1, P, P), ("C2 S2", 2, P, P), ("C2 S2", 3, P, P),
    # three views, distinct stops
    ("C2 C0 C1", 1, R, R), ("C2 C0 C1", 2, R, R), ("C2 C0 C1", 3, R, R),
    ("C2 C0 O4", 1, R, R), ("C2 C0 O4", 2, R, R), ("C2 C0 O4", 3, C, U),
    ("O2 C0 C4", 1, R, R), ("O2 C0 C4", 2, R, R), ("O2 C0 C4", 3, C, U),
    ("C2 O0 Q4", 1, R, R), ("C2 O0 Q4", 2, C, U), ("C2 O0 Q4", 3, C, U),
    ("O2 O0 O4", 1, C, U), ("O2 O0 O4", 2, C, U), ("O2 O0 O4", 3, C, U),
    ("Q2 Q0 Q4", 1, C, U), ("Q2 Q0 Q4", 2, C, U), ("Q2 Q0 Q4", 3, C, U),
    ("O2 Q0 O4", 1, C, U), ("O2 Q0 O4", 2, C, U), ("O2 Q0 O4", 3, C, U),
    ("C2 C0 S4", 1, P, P), ("C2 C0 S4", 2, P, P), ("C2 C0 S4", 3, P, P),
    ("S2 C0 C1", 1, P, P), ("S2 C0 C1", 2, P, P), ("S2 C0 C1", 3, P, P),
    ("C2 C0 X1", 1, P, P), ("C2 C0 X1", 2, P, P), ("C2 C0 X1", 3, P, P),
    ("O2 O0 X4", 1, P, P), ("O2 O0 X4", 2, P, P), ("O2 O0 X4", 3, P, P),
    # three views with repeated stops
    ("C2 C0 C0", 1, R, R), ("C2 C0 C0", 2, R, R), ("C2 C0 C0", 3, C, U),
    ("C2 C2 C2", 1, R, R), ("C2 C2 C2", 2, C, U), ("C2 C2 C2", 3, C, U),
    ("C2 C2 C0", 1, R, R), ("C2 C2 C0", 2, R, R), ("C2 C2 C0", 3, C, U),
    ("C2 O2 C0", 1, R, R), ("C2 O2 C0", 2, R, R), ("C2 O2 C0", 3, C, U),
    ("O2 C0 O0", 1, R, R), ("O2 C0 O0", 2, C, U), ("O2 C0 O0", 3, C, U),
    ("C2 C0 S0", 1, P, P), ("C2 C0 S0", 2, P, P), ("C2 C0 S0", 3, P, P),
]


class TestReadmeTruthTable(unittest.TestCase):
    def test_rows(self) -> None:
        for n, views, k, cc, expected in README_TABLE:
            with self.subTest(row=n):
                self.assertIs(StoppingRule(TH, k=k).decide(obs(*views), cc), expected)

    def test_row13_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            StoppingRule(TH, k=2).decide([], True)
        with self.assertRaises(ValueError):
            StoppingRule(TH, k=1).decide((), False)


class TestKindTruthTable(unittest.TestCase):
    def test_table_is_complete(self) -> None:
        keys = [(v, k) for v, k, _, _ in KIND_TABLE]
        self.assertEqual(len(keys), len(set(keys)), "duplicate table rows")
        for v in {v for v, _ in keys}:
            self.assertEqual({k for vv, k in keys if vv == v}, {1, 2, 3}, v)

    def test_rows(self) -> None:
        for views, k, if_cc, if_not in KIND_TABLE:
            for cc, expected in ((True, if_cc), (False, if_not)):
                with self.subTest(views=views, k=k, can_continue=cc):
                    self.assertIs(StoppingRule(TH, k=k).decide(kobs(views), cc), expected)

    def test_order_invariant(self) -> None:
        for views, k, _, _ in KIND_TABLE:
            o = kobs(views)
            rule = StoppingRule(TH, k=k)
            for cc in (True, False):
                got = {rule.decide(p, cc) for p in itertools.permutations(o)}
                self.assertEqual(len(got), 1, (views, k, cc))


class TestExhaustive(unittest.TestCase):
    """Every kind sequence of 1-3 views over stops {0, 2, 4} (repeats allowed), k=1..3, both cc."""

    def test_all_combinations(self) -> None:
        kinds = "SXCOQ"
        n = 0
        for length in (1, 2, 3):
            for ks in itertools.product(kinds, repeat=length):
                for stops in itertools.product((0, 2, 4), repeat=length):
                    o = obs(*((s, *KINDS[c]) for c, s in zip(ks, stops)))
                    present = any(c in "SX" for c in ks)
                    miss_stops = {s for c, s in zip(ks, stops) if c == "C"}
                    for k in (1, 2, 3):
                        rule = StoppingRule(TH, k=k)
                        for cc in (True, False):
                            if present:
                                exp = P
                            elif len(miss_stops) >= k:
                                exp = R
                            else:
                                exp = C if cc else U
                            self.assertIs(rule.decide(o, cc), exp, (ks, stops, k, cc))
                            n += 1
        self.assertEqual(n, (5 * 3 + 25 * 9 + 125 * 27) * 6)


class TestBoundaries(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = StoppingRule(TH)

    def one(self, m: float, q: float, b: float, stop: int = 2) -> Observation:
        return obs((stop, m, q, b))[0]

    def test_match_boundary(self) -> None:
        self.assertTrue(self.rule.is_strong_match(self.one(0.6, 0.9, 0.0)))
        self.assertFalse(self.rule.is_strong_match(self.one(0.5999999, 0.9, 0.0)))
        self.assertTrue(self.rule.is_clear_miss(self.one(0.5999999, 0.9, 0.0)))
        self.assertFalse(self.rule.is_clear_miss(self.one(0.6, 0.9, 0.0)))
        # a strong match counts whatever its quality and obstruction
        self.assertIs(self.rule.decide([self.one(0.6, 0.0, 1.0)], False), P)

    def test_quality_boundary(self) -> None:
        self.assertTrue(self.rule.is_clear_miss(self.one(0.1, 0.5, 0.1)))
        self.assertFalse(self.rule.is_clear_miss(self.one(0.1, 0.4999999, 0.1)))

    def test_obstruction_boundary(self) -> None:
        at, above = self.one(0.1, 0.9, 0.3), self.one(0.1, 0.9, 0.3000001)
        self.assertTrue(self.rule.is_clear_miss(at))
        self.assertFalse(self.rule.is_flagged_occluded(at))
        self.assertFalse(self.rule.is_clear_miss(above))
        self.assertTrue(self.rule.is_flagged_occluded(above))

    def test_all_three_at_boundary_still_removed(self) -> None:
        o = obs((2, 0.5999999, 0.5, 0.3), (4, 0.5999999, 0.5, 0.3))
        self.assertIs(self.rule.decide(o, True), R)

    def test_extreme_thresholds(self) -> None:
        # tau_occ = 1: nothing is ever flagged; obstruction 1.0 still allows a clear miss
        r = StoppingRule(Thresholds(tau_occ=1.0), k=1)
        self.assertFalse(r.is_flagged_occluded(self.one(0.1, 0.9, 1.0)))
        self.assertIs(r.decide([self.one(0.1, 0.9, 1.0)], True), R)
        # tau_occ = 0: any obstruction > 0 is flagged, obstruction exactly 0 is clear
        r = StoppingRule(Thresholds(tau_occ=0.0), k=1)
        self.assertTrue(r.is_flagged_occluded(self.one(0.1, 0.9, 1e-9)))
        self.assertIs(r.decide([self.one(0.1, 0.9, 0.0)], True), R)
        # tau_match = 1: only a perfect score is a strong match
        r = StoppingRule(Thresholds(tau_match=1.0))
        self.assertIs(r.decide([self.one(1.0, 0.0, 1.0)], True), P)
        self.assertIs(r.decide([self.one(0.99, 0.9, 0.9)], True), C)
        # tau_match = 0: every view is a strong match, so nothing can be a clear miss
        r = StoppingRule(Thresholds(tau_match=0.0), k=1)
        self.assertIs(r.decide([self.one(0.0, 0.9, 0.0)], True), P)

    def test_flagged_and_clear_miss_exclusive(self) -> None:
        grid = (0.0, 0.1, 0.2999999, 0.3, 0.3000001, 0.4999999, 0.5, 0.5000001,
                0.5999999, 0.6, 0.6000001, 0.9, 1.0)
        for m, q, b in itertools.product(grid, repeat=3):
            o = self.one(m, q, b)
            with self.subTest(m=m, q=q, b=b):
                self.assertFalse(self.rule.is_flagged_occluded(o) and self.rule.is_clear_miss(o))
                self.assertFalse(self.rule.is_strong_match(o) and self.rule.is_clear_miss(o))


class TestClearMissStops(unittest.TestCase):
    def test_distinct(self) -> None:
        rule = StoppingRule(TH)
        self.assertEqual(rule.clear_miss_stops(kobs("C2 C2 O0 C4 Q1 S3")), frozenset({2, 4}))
        self.assertEqual(rule.clear_miss_stops(kobs("O2 Q0")), frozenset())
        self.assertIsInstance(rule.clear_miss_stops(kobs("C2")), frozenset)


class TestConstruction(unittest.TestCase):
    def test_k_defaults_to_thresholds(self) -> None:
        self.assertEqual(StoppingRule(TH).k, 2)
        self.assertEqual(StoppingRule(Thresholds(k=3)).k, 3)
        self.assertEqual(StoppingRule(Thresholds(k=3), k=1).k, 1)

    def test_bad_k(self) -> None:
        for k in (0, -1):
            with self.subTest(k=k), self.assertRaises(ValueError):
                StoppingRule(TH, k=k)

    def test_first_clear_miss(self) -> None:
        r = StoppingRule.first_clear_miss(Thresholds(k=3))
        self.assertEqual(r.k, 1)
        self.assertEqual(r.name, "shared-k1")
        self.assertIs(r.decide(kobs("C2"), True), R)

    def test_name_and_thresholds(self) -> None:
        r = StoppingRule(TH)
        self.assertEqual(r.name, "shared-k2")
        self.assertIs(r.thresholds, TH)

    def test_read_only(self) -> None:
        r = StoppingRule(TH)
        for attr, val in (("k", 1), ("thresholds", Thresholds()), ("name", "x")):
            with self.subTest(attr=attr), self.assertRaises(AttributeError):
                setattr(r, attr, val)


class TestStateless(unittest.TestCase):
    def test_repeat_and_no_mutation(self) -> None:
        rule = StoppingRule(TH)
        o = list(kobs("C2 O0"))
        snapshot = list(o)
        first = rule.decide(o, True)
        self.assertIs(first, C)
        self.assertIsInstance(first, Continue)
        self.assertNotIsInstance(first, Decision)
        # a later, longer call does not carry state from the earlier one
        self.assertIs(rule.decide(kobs("C4"), False), U)
        self.assertIs(rule.decide(o, True), first)
        self.assertEqual(o, snapshot)


if __name__ == "__main__":
    unittest.main()
