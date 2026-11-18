"""View-selection policies (S, F, R, G, G-lookahead, G-myopic, FULL; Dir and Ranker A in development) and the episode
driver (verify/README.md sec. 3).

A policy only ever sees the Observations of stops it requested, the affordable candidate
stops, the Positions and the current stop. It never sees the ReplayEnv, the evidence store,
the EpisodeSpec or the ground truth. The decision always comes from the StoppingRule object
the caller passes to run_episode, so every multi-view policy shares one rule.

Primary G (DECISIONS 2026-10-12, approved for the code 2026-11-02) is the normalised selector
(D-norm) with the reachability lookahead OFF. The lookahead (D-G) is kept as the ablation
"G-lookahead" on the same scorer, so G vs G-lookahead isolates the lookahead. G-myopic is the
legacy pre-D-G selector (different normalisation, no lookahead), a separate ablation.

Lookahead hook. A policy may define begin_episode(limits: EpisodeLimits); run_episode then
calls it once per episode with the env's public budget (additional views, travel cap), the
rule's k and the rule's own per-view clear-miss predicate. None of this is evidence or truth.
Only G defines it, and only G-lookahead reads it, as part of its *selection*; primary G, F and
R choose from the affordable set as specified, and the decision is still rule.decide.

This module must stay free of introspection and I/O (gc, sys, inspect, open, closures, mangled
names); tests/test_policy_lint.py enforces that statically.
"""
from __future__ import annotations

import numbers
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np

from verify.schema import (
    CONTINUE,
    EPS,
    TRAVEL_TOL_CM,
    Budget,
    EpisodeLog,
    EpisodeResult,
    EpisodeRun,
    EpisodeSpec,
    Observation,
    Positions,
    ReplayError,
    Thresholds,
    derive_seed,
)

if TYPE_CHECKING:  # typing only: the driver uses just the public ReplayEnv / StoppingRule API
    from verify.replay import ReplayEnv
    from verify.stopping import StoppingRule

DEFAULT_R_SEEDS: tuple[int, ...] = tuple(range(20))
FULL_SWEEP_VIEWS = 4  # additional views FullSweep needs (all 5 stops)
# Legacy travel weight of the G-myopic ablation: degrees of separation per unit of travel / travel_norm_cm
# (the pre-D-norm G default). Placeholder, like every synthetic setting.
GMYOPIC_LAM_DEG = 30.0


class Policy(Protocol):
    name: str
    max_additional_views: int | None  # 0 for S; None = use the env budget
    early_stop: bool                  # False only for FullSweep

    def select(self, observations: Sequence[Observation], candidates: Sequence[int],
               positions: Positions, current_stop: int) -> int: ...


# ---------------------------------------------------------------- policies

class S:
    """Passive single view: the initial observation only. select is never called."""
    name = "S"
    max_additional_views: int | None = 0
    early_stop = True

    def select(self, observations: Sequence[Observation], candidates: Sequence[int],
               positions: Positions, current_stop: int) -> int:
        raise RuntimeError("S never moves; select must not be called")


def _first_in_f_order(candidates: Sequence[int], positions: Positions) -> int:
    allowed = set(candidates)
    for s in positions.f_order:
        if s in allowed:
            return s
    raise ReplayError(f"no f_order stop among candidates {tuple(candidates)}")


class F:
    """Fixed order: the first stop of positions.f_order that is affordable and unvisited."""
    name = "F"
    max_additional_views: int | None = None
    early_stop = True

    def select(self, observations: Sequence[Observation], candidates: Sequence[int],
               positions: Positions, current_stop: int) -> int:
        return _first_in_f_order(candidates, positions)


class R:
    """Uniform random affordable unvisited stop. Use a fresh R per (episode, seed) run."""
    name = "R"
    max_additional_views: int | None = None
    early_stop = True

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def select(self, observations: Sequence[Observation], candidates: Sequence[int],
               positions: Positions, current_stop: int) -> int:
        return int(self.rng.choice(sorted(candidates)))


class EpisodeLimits:
    """Public per-episode limits handed to policies that define begin_episode (only G does).

    additional_views and travel_cap_cm come from the env's Budget; k and is_clear_miss come from
    the shared StoppingRule, so G-lookahead counts clear misses exactly as the rule does. Primary
    G (lookahead off) receives them too but never reads them.
    """
    __slots__ = ("additional_views", "travel_cap_cm", "k", "is_clear_miss")

    def __init__(self, additional_views: int, travel_cap_cm: float | None, k: int,
                 is_clear_miss: Callable[[Observation], bool]) -> None:
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError(f"k must be an integer >= 1, got {k!r}")
        self.additional_views = int(additional_views)
        self.travel_cap_cm = None if travel_cap_cm is None else float(travel_cap_cm)
        self.k = int(k)
        self.is_clear_miss = is_clear_miss

    @classmethod
    def of(cls, budget: Budget, rule: StoppingRule) -> EpisodeLimits:
        return cls(budget.additional_views, budget.travel_cap_cm, rule.k, rule.is_clear_miss)

    def __repr__(self) -> str:
        return (f"EpisodeLimits(additional_views={self.additional_views}, "
                f"travel_cap_cm={self.travel_cap_cm}, k={self.k})")


def can_visit_more(start: int, visited: frozenset[int] | set[int], n: int, travel_left_cm: float | None,
                   views_left: int, positions: Positions) -> bool:
    """Exact: is there an ordered sequence of n distinct stops outside `visited`, starting from `start`,
    whose total travel is <= travel_left_cm + TRAVEL_TOL_CM, with n <= views_left?

    travel_left_cm None means no travel cap. Depth-first enumeration of every such path (at most
    4! = 24 orders on 5 stops), pruned once the running travel exceeds the allowance; exact, not greedy.
    """
    if n <= 0:
        return True
    if n > views_left:
        return False
    free = tuple(s for s in range(positions.n_stops) if s not in visited)
    if len(free) < n:
        return False
    if travel_left_cm is None:
        return True
    limit = travel_left_cm + TRAVEL_TOL_CM

    def search(cur: int, used: frozenset[int], spent: float, todo: int) -> bool:
        if todo == 0:
            return True
        for s in free:
            if s in used:
                continue
            t = spent + positions.travel_cm(cur, s)
            if t <= limit and search(s, used | {s}, t, todo - 1):
                return True
        return False
    return search(start, frozenset(), 0.0, n)


def _norm(x: float, m: float) -> float:
    """x / m with 0/0 defined as 0 (m is a max over non-negative values, so m == 0 implies x == 0)."""
    return 0.0 if m == 0 else x / m


def reference_stops(observations: Sequence[Observation], tau_occ: float) -> tuple[int, ...]:
    """Visited stops flagged occluded (obstruction > tau_occ), or all visited if none is."""
    flagged = tuple(sorted({o.stop for o in observations if o.obstruction > tau_occ}))
    return flagged or tuple(sorted({o.stop for o in observations}))


G_NAME = "G"                      # primary G: D-norm scorer, lookahead off (DECISIONS 2026-10-12)
G_LOOKAHEAD_NAME = "G-lookahead"  # ablation: the same scorer with the reachability lookahead (D-G)


class G:
    """Geometric parallax with per-decision normalisation (D-norm); reachability lookahead (D-G) optional.

    lookahead=False (the default) is primary G, name "G". lookahead=True is the ablation
    "G-lookahead": the same scorer, settings and tie-break, plus step 2's feasibility filter.
    1. ref = visited stops flagged occluded (obstruction > tau_occ), else all visited stops.
    2. Pool. Primary G: every affordable candidate. G-lookahead: the feasible candidates, where c
       is feasible if, after moving to c, REMOVED is still reachable: needed = k - #distinct
       clear-miss stops so far; c may supply one; the other needed - 1 must be reachable as
       distinct unvisited stops within the remaining travel cap AND remaining view budget
       (can_visit_more, exact). If no candidate is feasible G-lookahead falls back to all
       affordable candidates; it can still find PRESENT.
    3. Over that pool: score(c) = sep(c) / max_sep - lam * travel(c) / max_travel, sep(c) = min angle
       from c to ref, both maxima over the pool, 0/0 = 0; lam is dimensionless.
    4. Argmax; scores within EPS tie -> shorter travel -> lower stop index.

    The lookahead is G-lookahead's own selection intelligence, not a shared rule: F and R are unchanged.
    """
    max_additional_views: int | None = None
    early_stop = True

    def __init__(self, lam: float, tau_occ: float, *, lookahead: bool = False) -> None:
        lam = float(lam)
        if not (np.isfinite(lam) and lam >= 0):   # NaN/inf would silently tie every score (review F3)
            raise ValueError(f"lam must be finite and >= 0, got {lam!r}")
        self.lam = lam
        self.tau_occ = float(tau_occ)
        self.lookahead = bool(lookahead)
        self.name = G_LOOKAHEAD_NAME if self.lookahead else G_NAME
        self.limits: EpisodeLimits | None = None
        self.last_pool: tuple[int, ...] = ()   # inspection only: the pool of the last select
        self.last_fallback = False             # True if the last select found no feasible candidate

    @classmethod
    def from_thresholds(cls, th: Thresholds, *, lookahead: bool = False) -> G:
        """Primary G by default; lookahead=True gives the G-lookahead ablation with the same settings."""
        return cls(th.lam, th.tau_occ, lookahead=lookahead)

    def begin_episode(self, limits: EpisodeLimits) -> None:
        self.limits = limits

    def reference_stops(self, observations: Sequence[Observation]) -> tuple[int, ...]:
        return reference_stops(observations, self.tau_occ)

    def feasible(self, c: int, observations: Sequence[Observation], positions: Positions,
                 current_stop: int) -> bool:
        """Could the episode still reach REMOVED after moving to c (exact, optimistic about evidence)?"""
        lim = self.limits
        if lim is None:
            raise ReplayError("G.begin_episode(limits) must be called before select (run_episode does this)")
        visited = frozenset(o.stop for o in observations)
        needed = lim.k - len({o.stop for o in observations if lim.is_clear_miss(o)})
        views_left = lim.additional_views - (len(observations) - 1) - 1   # after the move to c
        if lim.travel_cap_cm is None:
            travel_left = None
        else:
            spent = positions.path_travel_cm([o.stop for o in observations])
            travel_left = lim.travel_cap_cm - spent - positions.travel_cm(current_stop, c)
        return can_visit_more(c, visited | {c}, needed - 1, travel_left, views_left, positions)

    def scores(self, pool: Sequence[int], ref: Sequence[int], positions: Positions,
               current_stop: int) -> dict[int, float]:
        """D-norm score of every stop in pool (normalised over the pool)."""
        sep = {c: min(positions.angle_deg(c, v) for v in ref) for c in pool}
        trav = {c: positions.travel_cm(current_stop, c) for c in pool}
        ms, mt = max(sep.values()), max(trav.values())
        return {c: _norm(sep[c], ms) - self.lam * _norm(trav[c], mt) for c in pool}

    def select(self, observations: Sequence[Observation], candidates: Sequence[int],
               positions: Positions, current_stop: int) -> int:
        if not observations:
            raise ValueError("G needs at least the initial observation")
        cands = tuple(sorted(candidates))
        feas = tuple(c for c in cands if self.feasible(c, observations, positions, current_stop)) \
            if self.lookahead else ()
        self.last_fallback = self.lookahead and not feas
        pool = feas or cands
        self.last_pool = pool
        sc = self.scores(pool, self.reference_stops(observations), positions, current_stop)
        return _argmax(sc, positions, current_stop)


class GMyopic:
    """Legacy ablation: the pre-D-G G, unchanged. No lookahead; fixed travel normaliser; lam in degrees.

    score(c) = min angle(c, ref) - lam_deg * travel(c) / travel_norm_cm over all affordable candidates.
    It differs from primary G in the normalisation, not the lookahead (neither has one), so G vs
    G-myopic is not the lookahead contrast; G vs G-lookahead is.
    """
    name = "G-myopic"
    max_additional_views: int | None = None
    early_stop = True

    def __init__(self, lam: float, travel_norm_cm: float, tau_occ: float) -> None:
        if not (np.isfinite(lam) and lam >= 0 and np.isfinite(travel_norm_cm) and travel_norm_cm > 0):
            raise ValueError("lam must be finite and >= 0, travel_norm_cm finite and > 0")
        self.lam = float(lam)
        self.travel_norm_cm = float(travel_norm_cm)
        self.tau_occ = float(tau_occ)

    @classmethod
    def from_thresholds(cls, th: Thresholds, lam_deg: float = GMYOPIC_LAM_DEG) -> GMyopic:
        """Thresholds.lam is G's dimensionless weight, so G-myopic takes its own lam_deg (legacy 30)."""
        return cls(lam_deg, th.travel_norm_cm, th.tau_occ)

    def reference_stops(self, observations: Sequence[Observation]) -> tuple[int, ...]:
        return reference_stops(observations, self.tau_occ)

    def score(self, c: int, ref: Sequence[int], positions: Positions, current_stop: int) -> float:
        sep = min(positions.angle_deg(c, v) for v in ref)
        return sep - self.lam * positions.travel_cm(current_stop, c) / self.travel_norm_cm

    def select(self, observations: Sequence[Observation], candidates: Sequence[int],
               positions: Positions, current_stop: int) -> int:
        if not observations:
            raise ValueError("G needs at least the initial observation")
        ref = self.reference_stops(observations)
        return _argmax({c: self.score(c, ref, positions, current_stop) for c in candidates},
                       positions, current_stop)


def _argmax(scores: dict[int, float], positions: Positions, current_stop: int) -> int:
    best: tuple[float, float, int] | None = None  # (score, travel, index)
    for c in sorted(scores):
        cand = (scores[c], positions.travel_cm(current_stop, c), c)
        if best is None or _g_better(cand, best):
            best = cand
    assert best is not None
    return best[2]


def _g_better(a: tuple[float, float, int], b: tuple[float, float, int]) -> bool:
    """Higher score; scores within EPS tie -> shorter travel -> lower stop index."""
    if abs(a[0] - b[0]) > EPS:
        return a[0] > b[0]
    if abs(a[1] - b[1]) > EPS:
        return a[1] < b[1]
    return a[2] < b[2]


# ---------------------------------------------------------------- descriptor-reading policies (DEVELOPMENT)
# notes/2026-11-05-side-descriptor.md and notes/2026-11-05-ranker-a-design.md. They read `side` from the observations:
# verify/side.py's SideEnv builds a SideObservation, with the descriptor of its own (requested) stop, for every view it
# reveals. A plain Observation has no `side`, so running them without descriptors fails loudly.

DIR_NAME = "Dir"


def _start_observation(observations: Sequence[Observation]) -> Observation:
    for o in observations:
        if o.step == 0:
            return o
    raise ValueError("no initial observation (step 0)")


class Dir:
    """Direction-rule control (explanatory; post-hoc development origin, notes/2026-11-05-side-descriptor.md).

    1. d3 = the start view's descriptor.
    2. d3 missing or exactly 0: choose as primary G (lambda 0, lookahead off) at every step.
    3. Otherwise prefer increasing azimuth if d3 > 0, decreasing if d3 < 0: among the affordable candidates on that
       side of the start stop, the one nearest the current stop (ties: lower stop index); none there: as primary G.
    """
    name = DIR_NAME
    max_additional_views: int | None = None
    early_stop = True

    def __init__(self, tau_occ: float) -> None:
        self.g = G(0.0, tau_occ)

    @classmethod
    def from_thresholds(cls, th: Thresholds) -> Dir:
        return cls(th.tau_occ)

    def select(self, observations: Sequence[Observation], candidates: Sequence[int],
               positions: Positions, current_stop: int) -> int:
        if not observations:
            raise ValueError("Dir needs at least the initial observation")
        start = _start_observation(observations)
        d3 = start.side
        if d3 is not None and d3 != 0.0:
            a0 = positions.azimuth(start.stop)
            side = [c for c in sorted(candidates) if (positions.azimuth(c) - a0) * d3 > 0]
            if side:
                return min(side, key=lambda c: (positions.travel_cm(current_stop, c), c))
        return self.g.select(observations, candidates, positions, current_stop)


# Ranker A's features (the design note's table, in its order) and A-nocue's subset: A-nocue drops the side descriptor
# and every other obstruction-derived feature.
RANKER_FEATURES = ("az_from_start", "sep_visited", "sep_occluded", "obstruction_start", "side_toward", "side_missing",
                   "match_max", "quality_min")
NOCUE_FEATURES = ("az_from_start", "sep_visited", "match_max", "quality_min")
OBSTRUCTION_DERIVED = tuple(f for f in RANKER_FEATURES if f not in NOCUE_FEATURES)


def half_span_deg(positions: Positions) -> float:
    """The arc's half-span S: the largest |azimuth - start azimuth| over the stops."""
    a0 = positions.azimuth(positions.start_index)
    return max(abs(positions.azimuth(s) - a0) for s in range(positions.n_stops))


def ranker_features(observations: Sequence[Observation], c: int, positions: Positions,
                    tau_occ: float) -> dict[str, float]:
    """The eight features of candidate stop c (notes/2026-11-05-ranker-a-design.md), from the observations of
    requested views and the geometry only. Nothing of c's own view is read."""
    if not observations:
        raise ValueError("ranker features need at least the initial observation")
    if any(o.stop == c for o in observations):
        raise ValueError(f"stop {c} has been visited; features are for unvisited candidates")
    span = half_span_deg(positions)
    az_c = positions.azimuth(c)
    start = _start_observation(observations)

    def dist(stop: int) -> float:
        return abs(az_c - positions.azimuth(stop)) / span
    sep_visited = min(dist(o.stop) for o in observations)
    occluded = [o for o in observations if o.obstruction > tau_occ]
    with_side = [o for o in observations if o.side is not None]
    toward = [o.side * (1.0 if az_c > positions.azimuth(o.stop) else -1.0) for o in with_side]
    return {"az_from_start": dist(start.stop),
            "sep_visited": sep_visited,
            "sep_occluded": min(dist(o.stop) for o in occluded) if occluded else sep_visited,
            "obstruction_start": float(start.obstruction),
            "side_toward": sum(toward) / len(toward) if toward else 0.0,
            "side_missing": 0.0 if with_side else 1.0,
            "match_max": max(float(o.match_score) for o in observations),
            "quality_min": min(float(o.quality) for o in observations)}


def _sigmoid(z: float) -> float:
    if z >= 0:
        return float(1.0 / (1.0 + np.exp(-z)))
    e = float(np.exp(z))
    return e / (1.0 + e)


class Ranker:
    """Ranker A (or A-nocue) at decision time: a fitted logistic model over ranker_features of each affordable
    candidate (G's candidate set, no lookahead), argmax of the probability with G's tie-break (scores within EPS ->
    shorter travel -> lower stop index); lambda 0. Fitted by verify/ranker.py; nothing is learned here."""
    max_additional_views: int | None = None
    early_stop = True

    def __init__(self, name: str, features: Sequence[str], mean: Sequence[float], scale: Sequence[float],
                 coef: Sequence[float], intercept: float, tau_occ: float) -> None:
        self.name = str(name)
        self.features = tuple(features)
        if not self.features or any(f not in RANKER_FEATURES for f in self.features) \
                or len(set(self.features)) != len(self.features):
            raise ValueError(f"features must be distinct names of RANKER_FEATURES, got {self.features}")
        vals = [tuple(float(x) for x in v) for v in (mean, scale, coef)]
        if any(len(v) != len(self.features) for v in vals):
            raise ValueError("mean, scale and coef need one value per feature")
        self.mean, self.scale, self.coef = vals
        self.intercept = float(intercept)
        self.tau_occ = float(tau_occ)
        if not all(np.isfinite(x) for v in vals for x in v) or not np.isfinite(self.intercept) \
                or any(s <= 0 for s in self.scale):
            raise ValueError("the model's numbers must be finite and every scale > 0")

    def probability(self, observations: Sequence[Observation], c: int, positions: Positions) -> float:
        f = ranker_features(observations, c, positions, self.tau_occ)
        z = self.intercept + sum(w * (f[n] - m) / s
                                 for n, m, s, w in zip(self.features, self.mean, self.scale, self.coef))
        return _sigmoid(z)

    def select(self, observations: Sequence[Observation], candidates: Sequence[int],
               positions: Positions, current_stop: int) -> int:
        if not observations:
            raise ValueError("the ranker needs at least the initial observation")
        return _argmax({c: self.probability(observations, c, positions) for c in candidates}, positions, current_stop)


class FullSweep:
    """Higher-effort reference: visit every remaining stop in F order, decide on all evidence."""
    name = "FULL"
    max_additional_views: int | None = None
    early_stop = False

    def select(self, observations: Sequence[Observation], candidates: Sequence[int],
               positions: Positions, current_stop: int) -> int:
        return _first_in_f_order(candidates, positions)


# ---------------------------------------------------------------- driver

def default_name(policy: Policy, rule: StoppingRule) -> str:
    """policy.name, with '-fcm' for the first-clear-miss rule (k = 1) on any policy except S."""
    if rule.k == 1 and not isinstance(policy, S):
        return f"{policy.name}-fcm"
    return policy.name


def _as_stop(stop: object, candidates: tuple[int, ...]) -> int:
    if isinstance(stop, bool) or not isinstance(stop, numbers.Integral):
        raise ReplayError(f"policy returned {stop!r}; expected an int stop index")
    s = int(stop)
    if s not in candidates:
        raise ReplayError(f"policy chose stop {s}, not in candidates {candidates}")
    return s


def run_episode(policy: Policy, env: ReplayEnv, rule: StoppingRule, *, seed: int | None = None,
                name: str | None = None) -> EpisodeRun:
    """Drive one policy through one fresh env with the given (shared) stopping rule.

    Returns the header-free EpisodeRun; join the header with EpisodeLog.from_run(spec, run).
    Exceptions from the policy, the env or the rule are never caught.
    """
    if not policy.early_stop and (env.budget.additional_views < FULL_SWEEP_VIEWS
                                  or env.budget.travel_cap_cm is not None):
        raise ReplayError(f"{policy.name} needs Budget({FULL_SWEEP_VIEWS}, None) (all stops, no travel cap); "
                          f"got {env.budget}")
    if isinstance(policy, S) and rule.k != 1:
        raise ReplayError(f"S decides with k = 1 semantics (removed on one clear miss); got k = {rule.k}")
    if hasattr(policy, "begin_episode"):   # G: public budget + the rule's k and predicate (G-lookahead reads them)
        policy.begin_episode(EpisodeLimits.of(env.budget, rule))
    obs: list[Observation] = [env.reset()]
    while True:
        cap = policy.max_additional_views
        cap_ok = cap is None or env.n_additional_views < cap
        can_continue = cap_ok and len(env.candidates()) > 0
        if not policy.early_stop and can_continue:
            decision = CONTINUE
        else:
            decision = rule.decide(tuple(obs), can_continue)
        if decision is CONTINUE:
            if not can_continue:
                raise ReplayError("stopping rule returned CONTINUE with no affordable stop")
        else:
            break
        candidates = tuple(env.candidates())
        stop = _as_stop(policy.select(tuple(obs), candidates, env.positions, env.current_stop), candidates)
        obs.append(env.request(stop))
    return env.finish(decision, policy=name or default_name(policy, rule), seed=seed)


def r_seed_salt(spec: EpisodeSpec) -> str:
    """Condition-free R salt: (arrangement, lighting). Matched visible/occluded/removed episodes of
    one arrangement and lighting share R's random stream (common random numbers)."""
    return f"{spec.arrangement_id}|{spec.lighting.value}"


def run_r_seeds(make_env: Callable[[], ReplayEnv], rule: StoppingRule, seeds: Sequence[int],
                seed_salt: str, *, spec: EpisodeSpec,
                name: str | None = None) -> tuple[list[EpisodeLog], EpisodeResult]:
    """R once per base seed (fresh env, fresh R seeded by derive_seed(s, salt)); average within episode.

    Pass r_seed_salt(spec) as seed_salt. spec is used only to join the log header after each run;
    it never reaches the env or the policy. DEFAULT_R_SEEDS is range(20).
    """
    if not seeds:
        raise ValueError("need at least one seed")
    logs = [EpisodeLog.from_run(spec, run_episode(R(derive_seed(s, seed_salt)), make_env(), rule, seed=s, name=name))
            for s in seeds]
    return logs, EpisodeResult.from_logs(logs)
