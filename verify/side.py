"""The left/right descriptor on the replay path (DEVELOPMENT; notes/2026-11-05-side-descriptor.md and
notes/2026-11-05-ranker-a-design.md).

A policy that reads the descriptor (ranker A, the direction-rule control Dir) must see it only for stops it requested,
exactly as it sees evidence. The pinned evidence types in verify/schema.py (an import of C0, Amendment 2) stay
unchanged: the descriptor travels in its own sealed store, and SideEnv wraps a ReplayEnv so that each Observation the
env reveals (reset, request) comes back as a SideObservation carrying the descriptor of THAT stop and no other.

    SideStore        episode_id -> stop -> descriptor (a float in [-1, 1], or None = missing: refused registration or
                     weak support). The only holder during a run. Before sealing it is an ordinary container; the
                     first bind() seals it (put, get, episode_ids and complete then raise LeakError) and returns
                     reveal(stop) for one episode. State lives in name-mangled slots, as in EvidenceStore: that stops
                     accidental and single-underscore access, not deliberate introspection (verify/README.md section 2).
    SideData         an immutable snapshot (DevData.side): a fresh unsealed SideStore per call, subset() for a CV
                     fold (no reference back to the full snapshot), and a provenance hash.
    SideObservation  Observation plus `side`, the descriptor of its own stop (None if missing).
    SideEnv          ReplayEnv's public API; reset() and request() return SideObservations, nothing else changes. The
                     run's log (EpisodeRun) is the inner env's, so the logged evidence is unchanged.

Rows on disk are perception/side_descriptor.py's output (schema cmpe249.side-descriptor.v0), one per (episode_id, stop).
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from verify.replay import ReplayEnv
from verify.schema import Budget, Decision, EpisodeRun, LeakError, Observation, Positions, config_sha256

SIDE_SCHEMA = "cmpe249.side-descriptor.v0"
SIDE_VERSIONS = ("side-0.1-dev",)


def _check_value(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not (math.isfinite(v) and -1.0 <= v <= 1.0):
        raise ValueError(f"a descriptor is None or a finite float in [-1, 1], got {value!r}")
    return v


class SideStore:
    """episode_id -> stop -> descriptor or None. See the module docstring."""

    __slots__ = ("__data", "__n_stops", "__sealed", "source")

    def __init__(self, n_stops: int = 5, *, source: str = "") -> None:
        self.__data: dict[str, dict[int, float | None]] = {}
        self.__n_stops = n_stops
        self.__sealed = False
        self.source = source

    @property
    def sealed(self) -> bool:
        return self.__sealed

    @property
    def n_stops(self) -> int:
        return self.__n_stops

    def _check_open(self, what: str) -> None:
        if self.__sealed:
            raise LeakError(f"SideStore is sealed; {what} is not allowed (use SideEnv)")

    def put(self, episode_id: str, stop: int, value: float | None) -> None:
        self._check_open("put")
        if not 0 <= int(stop) < self.__n_stops:
            raise ValueError(f"stop {stop} out of range")
        per = self.__data.setdefault(episode_id, {})
        if int(stop) in per:
            raise ValueError(f"duplicate descriptor for {episode_id} stop {stop}")
        per[int(stop)] = _check_value(value)

    def get(self, episode_id: str, stop: int) -> float | None:
        self._check_open("get")
        return self.__data[episode_id][stop]

    def episode_ids(self) -> tuple[str, ...]:
        self._check_open("episode_ids")
        return tuple(sorted(self.__data))

    def complete(self, episode_id: str) -> bool:
        self._check_open("complete")
        return sorted(self.__data.get(episode_id, {})) == list(range(self.__n_stops))

    def seal(self) -> None:
        self.__sealed = True

    def bind(self, episode_id: str) -> Callable[[int], float | None]:
        """Seal the store and return reveal(stop) for one episode (the policy runner only; never a policy)."""
        if episode_id not in self.__data:
            raise LeakError(f"no descriptor for episode {episode_id!r}")
        self.__sealed = True
        frozen = dict(self.__data[episode_id])

        def reveal(stop: int) -> float | None:
            return frozen[stop]
        return reveal

    @classmethod
    def from_jsonl(cls, paths: str | Path | Sequence[str | Path], n_stops: int = 5, *,
                   episode_ids: Iterable[str] | None = None) -> SideStore:
        """Rows of perception/side_descriptor.py output files; with episode_ids, only those episodes are kept and
        each must be complete. Refuses another schema or descriptor version and a duplicate (episode_id, stop)."""
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        keep = None if episode_ids is None else set(episode_ids)
        store = cls(n_stops, source="; ".join(str(p) for p in paths))
        for p in paths:
            for line in Path(p).read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("schema") != SIDE_SCHEMA or d.get("descriptor_version") not in SIDE_VERSIONS:
                    raise ValueError(f"{p}: not a side-descriptor row ({d.get('schema')!r}, "
                                     f"{d.get('descriptor_version')!r})")
                if keep is None or d["episode_id"] in keep:
                    store.put(d["episode_id"], int(d["stop"]), d["d"])
        if keep is not None:
            bad = sorted(e for e in keep if not store.complete(e))
            if bad:
                raise ValueError(f"{store.source}: no complete descriptor for {len(bad)} episodes, e.g. {bad[:3]}")
        return store


class SideData:
    """Immutable snapshot of descriptors (DevData.side): {episode_id: (value per stop)}."""

    def __init__(self, values: Mapping[str, Sequence[float | None]], n_stops: int, *, source: str = "") -> None:
        self._values = {e: tuple(_check_value(v) for v in vals) for e, vals in sorted(values.items())}
        bad = [e for e, vals in self._values.items() if len(vals) != n_stops]
        if bad:
            raise ValueError(f"descriptors need {n_stops} stops per episode, e.g. {bad[:3]}")
        self.n_stops = n_stops
        self.source = source

    @classmethod
    def from_store(cls, store: SideStore, episode_ids: Iterable[str], *, source: str = "") -> SideData:
        ids = sorted(set(episode_ids))
        return cls({e: [store.get(e, s) for s in range(store.n_stops)] for e in ids}, store.n_stops,
                   source=source or store.source)

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._values)

    def sha256(self) -> str:
        return config_sha256(self._values)

    def subset(self, episode_ids: Iterable[str]) -> SideData:
        """Only the listed episodes; the result keeps no reference to this snapshot (a CV training fold)."""
        ids = set(episode_ids)
        missing = ids - set(self._values)
        if missing:
            raise ValueError(f"no descriptor for {sorted(missing)[:3]}")
        return SideData({e: v for e, v in self._values.items() if e in ids}, self.n_stops,
                        source=f"subset of the descriptors ({len(ids)} episodes; source file not recorded)")

    def __call__(self) -> SideStore:
        store = SideStore(self.n_stops, source=self.source)
        for e, vals in self._values.items():
            for s, v in enumerate(vals):
                store.put(e, s, v)
        return store


@dataclass(frozen=True)
class SideObservation(Observation):
    """An Observation of a requested stop plus that stop's descriptor (None if missing)."""
    side: float | None = None


class SideEnv:
    """A ReplayEnv whose revealed observations also carry the descriptor of the revealed stop."""

    __slots__ = ("__env", "__reveal")

    def __init__(self, env: ReplayEnv, reveal: Callable[[int], float | None]) -> None:
        if not isinstance(env, ReplayEnv):
            raise TypeError(f"env must be a ReplayEnv, got {type(env).__name__}")
        self.__env = env
        self.__reveal = reveal

    @property
    def positions(self) -> Positions:
        return self.__env.positions

    @property
    def budget(self) -> Budget:
        return self.__env.budget

    @property
    def current_stop(self) -> int:
        return self.__env.current_stop

    @property
    def visited(self) -> tuple[int, ...]:
        return self.__env.visited

    @property
    def n_views(self) -> int:
        return self.__env.n_views

    @property
    def n_additional_views(self) -> int:
        return self.__env.n_additional_views

    @property
    def travel_cm(self) -> float:
        return self.__env.travel_cm

    @property
    def remaining_travel_cm(self) -> float | None:
        return self.__env.remaining_travel_cm

    @property
    def done(self) -> bool:
        return self.__env.done

    def reset(self) -> SideObservation:
        return self.__with_side(self.__env.reset())

    def candidates(self) -> tuple[int, ...]:
        return self.__env.candidates()

    def request(self, stop: int) -> SideObservation:
        return self.__with_side(self.__env.request(stop))

    def finish(self, decision: Decision, *, policy: str, seed: int | None = None) -> EpisodeRun:
        return self.__env.finish(decision, policy=policy, seed=seed)

    def __with_side(self, obs: Observation) -> SideObservation:
        return SideObservation(**dataclasses.asdict(obs), side=self.__reveal(obs.stop))

    def __repr__(self) -> str:
        return f"SideEnv({self.__env!r})"
