"""Tests for verify/lamgrid.py (review F1) and lam validation (review F3). Synthetic data checks logic only."""
from __future__ import annotations

import math
import unittest

from verify import select as sel
from verify.lamgrid import check_lam_grid, g_choices, g_lam_breakpoints, g_states
from verify.policies import G, GMyopic, run_episode
from verify.replay import ReplayEnv
from verify.schema import Budget, Positions, RunContext, Thresholds, cap_cm_from_steps
from verify.stopping import StoppingRule
from verify.synthetic import SyntheticParams, generate

POS = Positions.default()
CTX = RunContext(synthetic=True)


def irregular() -> Positions:
    return Positions.from_dict({
        "schema": "cmpe249.positions.v1", "radius_cm": 45.0, "start_stop": "P3",
        "f_order": ["P1", "P2", "P4", "P5"],
        "stops": [{"name": n, "azimuth_deg": a} for n, a in
                  zip(("P1", "P2", "P3", "P4", "P5"), (-55.0, -30.0, 0.0, 20.0, 65.0))]})


class RecordingG(G):
    """G that records every (current stop, ref, pool) it scores, to check g_states covers them."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.seen: set = set()

    def select(self, observations, candidates, positions, current_stop):
        pick = super().select(observations, candidates, positions, current_stop)
        if len(self.last_pool) >= 2:
            self.seen.add((current_stop, self.reference_stops(observations), tuple(self.last_pool)))
        return pick


def run_g_paths(lam: float, positions: Positions, *, recorder: set | None = None) -> list:
    """G's (path, decision) on every synthetic episode for k in {1,2,3}, b in 1..4, caps {1,2,3,4,6,inf} steps."""
    out = []
    for block_start in (True, False):
        specs, store, _truth = generate(12, 3, SyntheticParams(block_includes_start=block_start), positions)
        keys = store.replay_keys()
        for k in (1, 2, 3):
            th = Thresholds(k=k, lam=lam)
            for b in (1, 2, 3, 4):
                for steps in (1, 2, 3, 4, 6, None):
                    budget = Budget(b, cap_cm_from_steps(steps, positions))
                    for s in specs:
                        g = RecordingG(th.lam, th.tau_occ)
                        run = run_episode(g, ReplayEnv(keys[s.episode_id], store, positions, budget, CTX),
                                          StoppingRule(th))
                        out.append((s.episode_id, k, b, steps, run.requested_stops, run.decision))
                        if recorder is not None:
                            recorder |= g.seen
    return out


class DesignBreakpointsTest(unittest.TestCase):
    def test_design_breakpoints(self) -> None:
        got = [b.lam for b in g_lam_breakpoints(POS)]
        want = [2 / 3, 3 / 4, 1.0, 4 / 3, 3 / 2, 2.0]
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=6)

    def test_old_grid_collapses_and_new_default_does_not(self) -> None:
        old = check_lam_grid((0.0, 0.25, 0.5, 1.0), POS)
        self.assertIn((0.0, 0.25, 0.5), old.classes)
        self.assertEqual(old.on_breakpoint, (1.0,))
        self.assertEqual(len(old.warnings), 2)
        new = check_lam_grid(sel.DEFAULT_LAM_GRID, POS)
        self.assertEqual(len(new.classes), len(sel.DEFAULT_LAM_GRID))
        self.assertEqual(new.on_breakpoint, ())
        self.assertEqual(new.warnings, ())

    def test_placeholder_lam_behaves_like_zero_at_design_geometry(self) -> None:
        st = g_states(POS)
        self.assertEqual(g_choices(Thresholds().lam, st, POS), g_choices(0.0, st, POS))

    def test_states_cover_every_state_g_reaches(self) -> None:
        seen: set = set()
        run_g_paths(0.9, POS, recorder=seen)
        self.assertTrue(seen)
        self.assertLessEqual(seen, set(g_states(POS)))

    def test_same_interval_same_episodes_different_interval_differs(self) -> None:
        a, b, c = run_g_paths(0.0, POS), run_g_paths(0.5, POS), run_g_paths(1.2, POS)
        self.assertEqual(a, b)          # both below 2/3
        self.assertNotEqual(a, c)


class IrregularGeometryTest(unittest.TestCase):
    def test_breakpoints_move_and_intervals_are_constant(self) -> None:
        pos = irregular()
        bps = [b.lam for b in g_lam_breakpoints(pos)]
        self.assertNotEqual([round(x, 6) for x in bps],
                            [round(x, 6) for x in (2 / 3, 3 / 4, 1.0, 4 / 3, 3 / 2, 2.0)])
        st = g_states(pos)
        edges = [0.0, *bps, bps[-1] + 1.0]
        for lo, hi in zip(edges, edges[1:]):
            inside = [lo + (hi - lo) * f for f in (0.1, 0.5, 0.9)]
            ch = {g_choices(x, st, pos) for x in inside}
            self.assertEqual(len(ch), 1, (lo, hi))
        for x in bps:
            self.assertNotEqual(g_choices(x - 1e-6, st, pos), g_choices(x + 1e-6, st, pos))

    def test_reached_states_covered(self) -> None:
        pos = irregular()
        seen: set = set()
        run_g_paths(0.7, pos, recorder=seen)
        self.assertLessEqual(seen, set(g_states(pos)))


class LamValidationTest(unittest.TestCase):
    def test_non_finite_lam_refused(self) -> None:
        for bad in (math.nan, math.inf, -1.0):
            with self.assertRaises(ValueError):
                G(bad, 0.3)
            with self.assertRaises(ValueError):
                Thresholds(lam=bad)
            with self.assertRaises(ValueError):
                GMyopic(bad, 94.0, 0.3)
            with self.assertRaises(ValueError):
                sel.SelectionSettings(lam_grid=(0.0, bad) if bad > 0 else (bad, 0.0))
        with self.assertRaises(ValueError):
            GMyopic(30.0, math.nan, 0.3)
        with self.assertRaises(ValueError):
            Thresholds(travel_norm_cm=math.inf)
        with self.assertRaises(ValueError):
            sel.SelectionSettings(tau_match_grid=(0.5, math.nan))

    def test_cli_rejects_nan_lam_grid(self) -> None:
        with self.assertRaises(ValueError):
            sel.main(["--synthetic", "--n-arr", "6", "--lam-grid", "nan"])


if __name__ == "__main__":
    unittest.main()
