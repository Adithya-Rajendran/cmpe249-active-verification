"""SYNTHETIC episode generator for testing the offline verification pipeline.

Every number this module produces is invented. The parameters below were picked so that view
selection matters (an occluder hides a contiguous block of stops, the target is visible
elsewhere); they say nothing about real perception, and results computed from them are not
study results (schema.SYNTHETIC_LABEL).

Layout per arrangement: an occluder blocks a contiguous range of stops (by default always
including the start stop P3). The occluded-present and removed episodes of an arrangement
share the same layout and the same obstruction values (per lighting), so the occluder does
not leak the label. Visible episodes have the occluder present but not blocking. A predeclared
subset of arrangements has every stop blocked; there, UNKNOWN is the intended answer, and only
GroundTruth.indistinguishable says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from verify.schema import (
    CONDITIONS,
    LIGHTINGS,
    Condition,
    EpisodeSpec,
    EvidenceRecord,
    EvidenceStore,
    GroundTruth,
    Lighting,
    Positions,
    derive_seed,
    split_arrangements,
)

Range = tuple[float, float]


@dataclass(frozen=True)
class SyntheticParams:
    n_dev: int | None = None            # None -> round(n_arrangements / 3)
    indistinguishable_frac: float = 0.1 # arrangements whose occluder blocks all 5 stops (at least 1 if n >= 3)
    block_widths: tuple[int, ...] = (1, 2, 3, 4)      # contiguous blocked-stop count (distinguishable arrangements)
    block_width_probs: tuple[float, ...] = (0.25, 0.35, 0.25, 0.15)
    block_includes_start: bool = True   # the occluder always hides the start stop P3
    edge_obstruction: Range = (0.15, 0.45)   # U range at stops adjacent to the block
    blocked_obstruction: Range = (0.5, 0.95) # U range at blocked stops
    clear_obstruction: Range = (0.0, 0.15)   # U range elsewhere (and everywhere in visible)
    match_present: Range = (0.80, 0.08)      # N(mean, sd) for an unblocked present target
    match_absent: Range = (0.15, 0.08)       # N for removed targets / blocked present targets
    match_edge_present: Range = (0.55, 0.15) # N at edge stops when present
    quality_L1: Range = (0.85, 0.05)         # N
    quality_L2: Range = (0.60, 0.12)         # N (L2 lowers quality; some views fail tau_q)
    l2_match_drop: float = 0.08                            # subtracted from present-match means under L2
    target_id: str = "synthetic-target"

    def __post_init__(self) -> None:
        if not 0.0 <= self.indistinguishable_frac <= 1.0:
            raise ValueError("indistinguishable_frac must be in [0, 1]")
        if not self.block_widths or len(self.block_widths) != len(self.block_width_probs):
            raise ValueError("block_widths and block_width_probs must be non-empty and the same length")
        if any(p < 0 for p in self.block_width_probs) or not math.isclose(sum(self.block_width_probs), 1.0):
            raise ValueError("block_width_probs must be >= 0 and sum to 1")
        if any(w < 1 for w in self.block_widths):
            raise ValueError("block widths must be >= 1")
        for n in ("edge_obstruction", "blocked_obstruction", "clear_obstruction"):
            lo, hi = getattr(self, n)
            if not 0.0 <= lo <= hi <= 1.0:
                raise ValueError(f"{n} must be a range inside [0, 1]")
        for n in ("match_present", "match_absent", "match_edge_present", "quality_L1", "quality_L2"):
            if getattr(self, n)[1] < 0:
                raise ValueError(f"{n} sd must be >= 0")


@dataclass(frozen=True)
class OccluderLayout:
    arrangement_id: str
    blocked: tuple[int, ...]   # contiguous, ascending stop indices
    indistinguishable: bool    # True iff blocked == all stops

    def edges(self, n_stops: int) -> tuple[int, ...]:
        """Stops next to the block but not in it."""
        if not self.blocked:
            return ()
        cand = (self.blocked[0] - 1, self.blocked[-1] + 1)
        return tuple(s for s in cand if 0 <= s < n_stops)


def arrangement_ids(n: int) -> list[str]:
    """SYN01, SYN02, ... (3 digits when n >= 100)."""
    width = 3 if n >= 100 else 2
    return [f"SYN{i:0{width}d}" for i in range(1, n + 1)]


def n_indistinguishable(n_arrangements: int, frac: float) -> int:
    """round(frac * n) (half up), at least 1 when n >= 3 and frac > 0."""
    k = math.floor(frac * n_arrangements + 0.5)
    if n_arrangements >= 3 and frac > 0:
        k = max(k, 1)
    return min(k, n_arrangements)


def generate_layouts(n_arrangements: int, seed: int, params: SyntheticParams = SyntheticParams(),
                     n_stops: int = 5, start_index: int = 2) -> dict[str, OccluderLayout]:
    """One occluder layout per arrangement. Uses default_rng(seed); indistinguishable ones are chosen first."""
    if n_arrangements < 1:
        raise ValueError("n_arrangements must be >= 1")
    if not 0 <= start_index < n_stops:
        raise ValueError("start_index out of range")
    if any(w >= n_stops for w in params.block_widths):
        raise ValueError("distinguishable block widths must be < n_stops")
    rng = np.random.default_rng(seed)
    ids = arrangement_ids(n_arrangements)
    k = n_indistinguishable(n_arrangements, params.indistinguishable_frac)
    ind = {ids[int(i)] for i in rng.choice(n_arrangements, size=k, replace=False)} if k else set()
    out: dict[str, OccluderLayout] = {}
    for a in ids:
        if a in ind:
            out[a] = OccluderLayout(a, tuple(range(n_stops)), True)
            continue
        w = int(rng.choice(params.block_widths, p=params.block_width_probs))
        if params.block_includes_start:
            starts = list(range(max(0, start_index - w + 1), min(start_index, n_stops - w) + 1))
        else:
            starts = list(range(0, n_stops - w + 1))
        lo = starts[int(rng.integers(len(starts)))]
        out[a] = OccluderLayout(a, tuple(range(lo, lo + w)), False)
    return out


def _uniform(rng: np.random.Generator, r: Range) -> float:
    return float(rng.uniform(r[0], r[1]))


def _normal(rng: np.random.Generator, mean: float, sd: float) -> float:
    return float(rng.normal(mean, sd))


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def _obstruction(rng: np.random.Generator, layout: OccluderLayout, n_stops: int, p: SyntheticParams) -> list[float]:
    """Blocking obstruction per stop for one (arrangement, lighting); shared by occluded and removed."""
    edges = set(layout.edges(n_stops))
    out = []
    for s in range(n_stops):
        r = p.blocked_obstruction if s in layout.blocked else p.edge_obstruction if s in edges else p.clear_obstruction
        out.append(_clip(_uniform(rng, r)))
    return out


def _match(rng: np.random.Generator, cond: Condition, light: Lighting, stop: int,
           layout: OccluderLayout, n_stops: int, p: SyntheticParams) -> float:
    drop = p.l2_match_drop if light is Lighting.L2 else 0.0
    if cond is Condition.REMOVED:
        mean, sd = p.match_absent
    elif cond is Condition.VISIBLE:
        mean, sd = p.match_present[0] - drop, p.match_present[1]
    elif stop in layout.blocked:
        mean, sd = p.match_absent
    elif stop in layout.edges(n_stops):
        mean, sd = p.match_edge_present[0] - drop, p.match_edge_present[1]
    else:
        mean, sd = p.match_present[0] - drop, p.match_present[1]
    return _clip(_normal(rng, mean, sd))


def generate(n_arrangements: int, seed: int, params: SyntheticParams = SyntheticParams(),
             positions: Positions | None = None) -> tuple[list[EpisodeSpec], EvidenceStore, GroundTruth]:
    """SYNTHETIC specs, an unsealed evidence store and the scoring-only ground truth."""
    pos = positions if positions is not None else Positions.default()
    n_stops = pos.n_stops
    layouts = generate_layouts(n_arrangements, seed, params, n_stops=n_stops, start_index=pos.start_index)
    ids = list(layouts)
    n_dev = params.n_dev if params.n_dev is not None else math.floor(n_arrangements / 3 + 0.5)
    splits = split_arrangements(ids, n_dev, seed)

    rng = np.random.default_rng(derive_seed("evidence", seed))
    store = EvidenceStore(n_stops, synthetic=True, source=f"verify.synthetic seed={seed}")
    specs: list[EpisodeSpec] = []
    for a in ids:
        lay = layouts[a]
        for light in LIGHTINGS:
            blocking = _obstruction(rng, lay, n_stops, params)   # same values for occluded and removed
            for cond in CONDITIONS:
                spec = EpisodeSpec(a, cond, light, params.target_id, splits[a])
                q_mean, q_sd = params.quality_L1 if light is Lighting.L1 else params.quality_L2
                for s in range(n_stops):
                    obs = (_clip(_uniform(rng, params.clear_obstruction)) if cond is Condition.VISIBLE
                           else blocking[s])
                    store.put(spec.episode_id, EvidenceRecord(
                        stop=s, match_score=_match(rng, cond, light, s, lay, n_stops, params),
                        quality=_clip(_normal(rng, q_mean, q_sd)), obstruction=obs))
                specs.append(spec)
    cond_rank = {c: i for i, c in enumerate(CONDITIONS)}
    light_rank = {l: i for i, l in enumerate(LIGHTINGS)}
    specs.sort(key=lambda s: (s.arrangement_id, cond_rank[s.condition], light_rank[s.lighting]))
    truth = GroundTruth.from_specs(specs, [a for a, l in layouts.items() if l.indistinguishable], synthetic=True)
    return specs, store, truth
