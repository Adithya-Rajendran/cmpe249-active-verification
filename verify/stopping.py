"""Shared stopping rule for every multi-view policy (see verify/README.md section 1).

PRESENT if any view is a strong match; REMOVED once clear misses come from >= k distinct
stops; otherwise CONTINUE while the caller says another view is possible, else UNKNOWN.
Stateless and deterministic: it reads only the observations it is given and never combines
scores (they are not calibrated probabilities).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from verify.schema import CONTINUE, Continue, Decision, Observation, Thresholds


class StoppingRule:
    """One rule object shared by F, R, G (and A later); k=1 is the first-clear-miss variant."""

    __slots__ = ("_th", "_k")

    def __init__(self, thresholds: Thresholds, k: int | None = None) -> None:
        k = thresholds.k if k is None else k
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError(f"k must be an integer >= 1, got {k!r}")
        self._th = thresholds
        self._k = int(k)

    @classmethod
    def first_clear_miss(cls, thresholds: Thresholds) -> StoppingRule:
        return cls(thresholds, k=1)

    @property
    def k(self) -> int:
        return self._k

    @property
    def thresholds(self) -> Thresholds:
        return self._th

    @property
    def name(self) -> str:
        return f"shared-k{self._k}"

    def __repr__(self) -> str:
        return f"StoppingRule(k={self._k}, thresholds={self._th!r})"

    # ------------------------------------------------------------ per-view predicates
    # Boundaries are exact as written in the contract (no EPS): >= tau_match, >= tau_q,
    # <= tau_occ, and flagged occluded means > tau_occ.

    def is_strong_match(self, o: Observation) -> bool:
        return o.match_score >= self._th.tau_match

    def is_clear_miss(self, o: Observation) -> bool:
        return (o.quality >= self._th.tau_q
                and o.obstruction <= self._th.tau_occ
                and o.match_score < self._th.tau_match)

    def is_flagged_occluded(self, o: Observation) -> bool:
        return o.obstruction > self._th.tau_occ

    def clear_miss_stops(self, observations: Iterable[Observation]) -> frozenset[int]:
        """Distinct stops with at least one clear miss (repeat views of a stop count once)."""
        return frozenset(o.stop for o in observations if self.is_clear_miss(o))

    # ------------------------------------------------------------ decision

    def decide(self, observations: Sequence[Observation], can_continue: bool) -> Decision | Continue:
        obs = tuple(observations)
        if not obs:
            raise ValueError("decide needs at least one observation (the initial view)")
        if any(self.is_strong_match(o) for o in obs):
            return Decision.PRESENT
        if len(self.clear_miss_stops(obs)) >= self._k:
            return Decision.REMOVED
        return CONTINUE if can_continue else Decision.UNKNOWN
