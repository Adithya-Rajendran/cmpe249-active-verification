"""Where does G's travel weight lam actually change G's choice? (review finding F1)

G's D-norm score is score(c) = a(c) - lam * b(c), with a = sep/max_sep and b = travel/max_travel over the
pool (verify/policies.py). On a fixed geometry there are only finitely many decision states, so G's
behaviour is piecewise constant in lam. It changes only at a few *breakpoints*. Two lam values between
the same breakpoints give identical choices in every episode, so a lam grid with two values in one
interval can not select lam at all.

This module enumerates every decision state G can be in on the given Positions and returns the
breakpoints. It is exact, cheap (a few hundred states on 5 stops) and depends only on the geometry, so
it must be re-run when the measured azimuths replace the design template in configs/positions.json.

A state is (current stop, reference set, pool):
- current stop: the last stop of any simple path from the start stop (the start itself if no move yet);
- reference set: any non-empty subset of the visited stops (G uses the visited stops flagged occluded,
  else all visited ones, and any subset can be flagged);
- pool: what G.select scores, i.e. the feasible candidates if any, else the affordable ones, for every
  remaining travel allowance, remaining view count and number of clear misses still needed.
This is a slight superset of the states G reaches (it ignores which paths G itself would take), so a
breakpoint listed here may be one that no real episode hits. It never misses a breakpoint.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations

from verify.policies import G, _argmax, can_visit_more
from verify.schema import TRAVEL_TOL_CM, Positions

State = tuple[int, tuple[int, ...], tuple[int, ...]]   # (current stop, ref, pool)


def _paths_from(start: int, positions: Positions) -> Iterable[tuple[int, ...]]:
    others = [s for s in range(positions.n_stops) if s != start]
    for r in range(0, len(others) + 1):
        for perm in permutations(others, r):
            yield (start, *perm)


def _remaining_allowances(cur: int, unvisited: Sequence[int], positions: Positions) -> list[float | None]:
    """Every travel allowance at which some pool can change: the cost of each simple path from cur over
    unvisited stops (pools are constant between these values), plus uncapped (None)."""
    vals: set[float] = {0.0}
    for r in range(1, len(unvisited) + 1):
        for perm in permutations(unvisited, r):
            vals.add(round(positions.path_travel_cm((cur, *perm)), 9))
    return [*sorted(vals), None]


def _pools(path: tuple[int, ...], positions: Positions) -> set[tuple[int, ...]]:
    cur, visited = path[-1], frozenset(path)
    unvisited = [s for s in range(positions.n_stops) if s not in visited]
    out: set[tuple[int, ...]] = set()
    n = positions.n_stops
    for left in _remaining_allowances(cur, unvisited, positions):
        afford = tuple(c for c in unvisited
                       if left is None or positions.travel_cm(cur, c) <= left + TRAVEL_TOL_CM)
        if len(afford) < 2:
            continue
        for views_left in range(1, n):            # views left before the move
            for more in range(0, n):              # clear-miss stops still needed after c
                feas = tuple(c for c in afford if can_visit_more(
                    c, visited | {c}, more,
                    None if left is None else left - positions.travel_cm(cur, c),
                    views_left - 1, positions))
                pool = feas or afford
                if len(pool) >= 2:
                    out.add(pool)
    return out


def g_states(positions: Positions) -> list[State]:
    """Every (current stop, reference set, pool) decision state of G on this geometry (sorted, unique)."""
    states: set[State] = set()
    for path in _paths_from(positions.start_index, positions):
        visited = tuple(sorted(path))
        pools = _pools(path, positions)
        if not pools:
            continue
        for r in range(1, len(visited) + 1):
            for ref in combinations(visited, r):
                for pool in pools:
                    states.add((path[-1], ref, pool))
    return sorted(states)


def g_choices(lam: float, states: Sequence[State], positions: Positions) -> tuple[int, ...]:
    """G's choice in each state at this lam (same scoring and tie-break code as G.select)."""
    g = G(lam, 0.0)
    return tuple(_argmax(g.scores(pool, ref, positions, cur), positions, cur) for cur, ref, pool in states)


@dataclass(frozen=True)
class Breakpoint:
    lam: float
    n_states_changed: int     # states whose choice differs just below vs just above
    resolves_to: str          # "above" / "below" / "neither": which side the exact value behaves like


def g_lam_breakpoints(positions: Positions, *, lam_max: float = 4.0, delta: float = 1e-7) -> list[Breakpoint]:
    """Exact lam values in (0, lam_max] at which G's choice changes in at least one state."""
    states = g_states(positions)
    g = G(0.0, 0.0)
    cands: set[float] = set()
    for cur, ref, pool in states:
        sep = {c: min(positions.angle_deg(c, v) for v in ref) for c in pool}
        trav = {c: positions.travel_cm(cur, c) for c in pool}
        ms, mt = max(sep.values()), max(trav.values())
        a = {c: (sep[c] / ms if ms else 0.0) for c in pool}
        b = {c: (trav[c] / mt if mt else 0.0) for c in pool}
        for c, d in combinations(pool, 2):
            if abs(b[c] - b[d]) > 1e-12:
                x = (a[c] - a[d]) / (b[c] - b[d])
                if 0.0 < x <= lam_max:
                    cands.add(round(x, 9))
    del g
    out = []
    for x in sorted(cands):
        lo, at, hi = (g_choices(v, states, positions) for v in (x - delta, x, x + delta))
        if lo == hi:
            continue
        side = "above" if at == hi else "below" if at == lo else "neither"
        out.append(Breakpoint(x, sum(p != q for p, q in zip(lo, hi)), side))
    return out


@dataclass(frozen=True)
class LamGridCheck:
    breakpoints: tuple[Breakpoint, ...]
    classes: tuple[tuple[float, ...], ...]   # grid values that give identical G choices in every state
    on_breakpoint: tuple[float, ...]         # grid values sitting exactly on a breakpoint
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"breakpoints": [{"lam": b.lam, "n_states_changed": b.n_states_changed,
                                 "exact_value_behaves_like": b.resolves_to} for b in self.breakpoints],
                "equivalence_classes": [list(c) for c in self.classes],
                "on_breakpoint": list(self.on_breakpoint), "warnings": list(self.warnings)}


def check_lam_grid(grid: Sequence[float], positions: Positions, *, tol: float = 1e-6) -> LamGridCheck:
    """Which grid values are behaviourally identical for G, and which sit on a breakpoint."""
    lam_max = max(4.0, max(grid, default=0.0) + 1.0)
    bps = tuple(g_lam_breakpoints(positions, lam_max=lam_max))
    states = g_states(positions)
    by_choice: dict[tuple[int, ...], list[float]] = {}
    for lam in grid:
        by_choice.setdefault(g_choices(lam, states, positions), []).append(float(lam))
    classes = tuple(tuple(v) for v in by_choice.values())
    on_bp = tuple(float(lam) for lam in grid if any(abs(lam - b.lam) <= tol for b in bps))
    warns = []
    for c in classes:
        if len(c) > 1:
            warns.append(f"lam grid values {list(c)} give identical G choices in every decision state on this "
                         "geometry: only one of them can be told apart by selection")
    for lam in on_bp:
        b = next(b for b in bps if abs(lam - b.lam) <= tol)
        warns.append(f"lam = {lam} sits on a G breakpoint (lam = {b.lam:.6f}); its choices come from the "
                     f"exact tie-break and match the values just {b.resolves_to} it")
    return LamGridCheck(bps, classes, on_bp, tuple(warns))
