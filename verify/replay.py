"""Offline replay of one episode: the only holder of evidence during a run (verify/README.md section 2).

A policy sees an Observation only for a stop it requested through request(). The env is built
from an opaque ReplayKey, never from the EpisodeSpec, so it holds nothing that encodes the
condition or the truth; finish() returns a header-free EpisodeRun and the driver joins the log
header outside the env. The env keeps no reference to the EvidenceStore (binding seals it).

Guarantee: no public or single-underscore name leads to unvisited evidence, and the sealed store
refuses every direct read with LeakError. Deliberate introspection (name-mangled slots, closures,
frames, gc, reading files) is out of scope: in-process Python cannot be sandboxed, so policy code
is kept honest by review plus the static lint in tests/test_policy_lint.py.
"""
from __future__ import annotations

import numbers
from collections.abc import Callable

from verify.schema import (
    TRAVEL_TOL_CM,
    Budget,
    Decision,
    EpisodeRun,
    EpisodeSpec,
    EvidenceRecord,
    EvidenceStore,
    GroundTruth,
    Observation,
    Positions,
    ReplayError,
    ReplayKey,
    RunContext,
)


class ReplayEnv:
    """Replays one recorded episode under a view budget and an optional travel cap."""

    __slots__ = ("__reveal", "__positions", "__budget", "__context",
                 "__visited", "__evidence", "__travel", "__reset", "__done")

    def __init__(self, key: ReplayKey, evidence_store: EvidenceStore, positions: Positions,
                 budget: Budget, context: RunContext = RunContext()) -> None:
        args = (key, evidence_store, positions, budget, context)
        if any(isinstance(a, GroundTruth) for a in args):
            raise TypeError("ground truth is scoring-only and must never be passed to ReplayEnv")
        if any(isinstance(a, EpisodeSpec) for a in args):
            raise TypeError("pass store.replay_key(spec.episode_id), not the spec: the spec encodes the condition")
        for name, val, typ in (("key", key, ReplayKey), ("evidence_store", evidence_store, EvidenceStore),
                               ("positions", positions, Positions), ("budget", budget, Budget),
                               ("context", context, RunContext)):
            if not isinstance(val, typ):
                raise TypeError(f"{name} must be {typ.__name__}, got {type(val).__name__}")
        if evidence_store.n_stops != positions.n_stops:
            raise ReplayError(f"store has {evidence_store.n_stops} stops, positions has {positions.n_stops}")
        reveal: Callable[[int], EvidenceRecord] = evidence_store.bind(key)  # seals the store
        self.__reveal = reveal
        self.__positions = positions
        self.__budget = budget
        self.__context = context
        self.__visited: list[int] = []
        self.__evidence: list[Observation] = []
        self.__travel = 0.0
        self.__reset = False
        self.__done = False

    # ------------------------------------------------------------ read-only state

    @property
    def positions(self) -> Positions:
        return self.__positions

    @property
    def budget(self) -> Budget:
        return self.__budget

    @property
    def current_stop(self) -> int:
        if not self.__visited:
            raise ReplayError("env not reset")
        return self.__visited[-1]

    @property
    def visited(self) -> tuple[int, ...]:
        return tuple(self.__visited)

    @property
    def n_views(self) -> int:
        return len(self.__visited)

    @property
    def n_additional_views(self) -> int:
        return max(len(self.__visited) - 1, 0)

    @property
    def travel_cm(self) -> float:
        return self.__travel

    @property
    def remaining_travel_cm(self) -> float | None:
        cap = self.__budget.travel_cap_cm
        return None if cap is None else max(cap - self.__travel, 0.0)

    @property
    def done(self) -> bool:
        return self.__done

    # ------------------------------------------------------------ actions

    def reset(self) -> Observation:
        """Take the initial view at the start stop (zero travel). Only once."""
        if self.__reset:
            raise ReplayError("reset() may be called only once per env")
        self.__reset = True
        return self.__visit(self.__positions.start_index, 0.0)

    def candidates(self) -> tuple[int, ...]:
        """Unvisited stops affordable under the remaining view budget and travel cap, ascending."""
        if not self.__reset or self.__done or self.n_additional_views >= self.__budget.additional_views:
            return ()
        cur, rem = self.__visited[-1], self.remaining_travel_cm
        return tuple(s for s in range(self.__positions.n_stops)
                     if s not in self.__visited
                     and (rem is None or self.__positions.travel_cm(cur, s) <= rem + TRAVEL_TOL_CM))

    def request(self, stop: int) -> Observation:
        """Move to an affordable unvisited stop and reveal its evidence. A rejected request changes nothing."""
        if not self.__reset:
            raise ReplayError("request() before reset()")
        if self.__done:
            raise ReplayError("request() after finish()")
        if isinstance(stop, bool) or not isinstance(stop, numbers.Integral):
            raise ReplayError(f"stop must be an int, got {type(stop).__name__}")
        stop = int(stop)
        if not 0 <= stop < self.__positions.n_stops:
            raise ReplayError(f"stop {stop} out of range 0..{self.__positions.n_stops - 1}")
        if stop in self.__visited:
            raise ReplayError(f"stop {stop} already visited")
        if self.n_additional_views >= self.__budget.additional_views:
            raise ReplayError(f"view budget used up ({self.__budget.additional_views} additional views)")
        if stop not in self.candidates():
            raise ReplayError(f"stop {stop} unaffordable: travel {self.__positions.travel_cm(self.current_stop, stop):.2f} cm "
                              f"> remaining {self.remaining_travel_cm} cm")
        return self.__visit(stop, self.__positions.travel_cm(self.__visited[-1], stop))

    def finish(self, decision: Decision, *, policy: str, seed: int | None = None) -> EpisodeRun:
        """Close the episode with a terminal decision; return the header-free run (no ids, no truth)."""
        if not self.__reset:
            raise ReplayError("finish() before reset()")
        if self.__done:
            raise ReplayError("finish() may be called only once")
        if not isinstance(decision, Decision):
            raise ReplayError(f"decision must be a Decision, got {decision!r}")
        if not isinstance(policy, str) or not policy:
            raise ReplayError("policy name must be a non-empty string")
        self.__done = True
        ctx, b = self.__context, self.__budget
        return EpisodeRun(model_id=ctx.model_id, config_version=ctx.config_version,
                          config_sha256=ctx.config_sha256, policy=policy,
                          budget_additional_views=b.additional_views, travel_cap_cm=b.travel_cap_cm,
                          seed=seed, requested_stops=tuple(self.__visited), evidence=tuple(self.__evidence),
                          decision=decision, travel_cm=self.__travel, n_views=self.n_views,
                          n_additional_views=self.n_additional_views, synthetic=ctx.synthetic)

    def __visit(self, stop: int, travel: float) -> Observation:
        obs = Observation.from_record(self.__reveal(stop), step=len(self.__visited), positions=self.__positions)
        if obs.stop != stop:
            raise ReplayError(f"evidence for stop {stop} is labelled stop {obs.stop}")
        self.__travel += travel
        self.__visited.append(stop)
        self.__evidence.append(obs)
        return obs

    def __repr__(self) -> str:
        return f"ReplayEnv(visited={self.visited}, travel_cm={self.__travel:.2f}, done={self.__done})"

