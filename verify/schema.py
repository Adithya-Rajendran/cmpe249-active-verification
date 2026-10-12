"""Shared types for the offline verification core (see verify/README.md for the contract).

Everything here is plain Python + stdlib. Policies may import Observation, Positions and the
enums; they never receive an EvidenceStore, an EpisodeSpec or a GroundTruth.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import secrets
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSITIONS_PATH = REPO_ROOT / "configs" / "positions.json"

LOG_SCHEMA = "cmpe249.episode_log.v1"
RESULT_SCHEMA = "cmpe249.episode_result.v1"
TRUTH_SCHEMA = "cmpe249.ground_truth.v1"
POSITIONS_SCHEMA = "cmpe249.positions.v1"
SYNTHETIC_LABEL = "SYNTHETIC - generated evidence, not study results"
EPS = 1e-9  # tolerance for score / rate comparisons (travel affordability uses TRAVEL_TOL_CM)
TRAVEL_TOL_CM = 1e-6  # tolerance for "is this move/path affordable under the cap" (D-cap; env candidates() and G lookahead); no other slack
CAP_STEPS_DEFAULT: tuple[float, ...] = (1, 2, 3, 4, 6, math.inf)  # D-cap grid in arc steps; inf = uncapped


# ---------------------------------------------------------------- errors

class LeakError(RuntimeError):
    """Something tried to read evidence or ground truth it must not see."""


class ReplayError(ValueError):
    """Illegal ReplayEnv call (visited / unaffordable / over budget / after finish)."""


class ManifestError(ValueError):
    """Episode list or split violates the protocol."""


# ---------------------------------------------------------------- enums

class Condition(StrEnum):
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    REMOVED = "removed"


class Lighting(StrEnum):
    L1 = "L1"
    L2 = "L2"


class Split(StrEnum):
    DEV = "dev"
    TEST = "test"
    PILOT = "pilot"


class TruthState(StrEnum):
    """Physical target state. 'unknown' is never a truth class."""
    PRESENT = "present"
    REMOVED = "removed"


class Decision(StrEnum):
    PRESENT = "present"
    REMOVED = "removed"
    UNKNOWN = "unknown"


class Continue(StrEnum):
    """Non-terminal output of StoppingRule.decide: acquire another view."""
    CONTINUE = "continue"


CONTINUE = Continue.CONTINUE
CONDITIONS: tuple[Condition, ...] = tuple(Condition)
LIGHTINGS: tuple[Lighting, ...] = tuple(Lighting)


def truth_from_condition(condition: Condition | str) -> TruthState:
    """visible/occluded -> PRESENT, removed -> REMOVED (scoring side only)."""
    return TruthState.REMOVED if Condition(condition) is Condition.REMOVED else TruthState.PRESENT


# ---------------------------------------------------------------- geometry

@dataclass(frozen=True)
class Stop:
    index: int          # 0-based, P1 -> 0
    name: str           # "P1".."P5"
    azimuth_deg: float  # about the workspace centre


@dataclass(frozen=True)
class Positions:
    """Camera stops on the arc. Travel is arc length |d az| (rad) x radius."""
    stops: tuple[Stop, ...]
    radius_cm: float
    start_index: int
    f_order: tuple[int, ...]              # stop indices, excludes the start stop
    height_cm: float | None = None
    tilt_deg: float | None = None
    status: str = "design values"

    def __post_init__(self) -> None:
        if [s.index for s in self.stops] != list(range(len(self.stops))):
            raise ValueError("stop indices must be 0..n-1 in order")
        if not 0 <= self.start_index < len(self.stops):
            raise ValueError("start_index out of range")
        if sorted(self.f_order) != sorted(set(range(len(self.stops))) - {self.start_index}):
            raise ValueError("f_order must list every non-start stop exactly once")
        if self.radius_cm <= 0:
            raise ValueError("radius_cm must be > 0")

    @classmethod
    def default(cls) -> Positions:
        """The design values (P1..P5 at -60..+60 deg, r = 45 cm, start P3, F = P1,P2,P4,P5)."""
        stops = tuple(Stop(i, f"P{i + 1}", az) for i, az in enumerate((-60.0, -30.0, 0.0, 30.0, 60.0)))
        return cls(stops=stops, radius_cm=45.0, start_index=2, f_order=(0, 1, 3, 4))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Positions:
        if d.get("schema") != POSITIONS_SCHEMA:
            raise ValueError(f"positions schema must be {POSITIONS_SCHEMA!r}, got {d.get('schema')!r}")
        stops = tuple(Stop(i, s["name"], float(s["azimuth_deg"])) for i, s in enumerate(d["stops"]))
        idx = {s.name: s.index for s in stops}
        return cls(stops=stops, radius_cm=float(d["radius_cm"]), start_index=idx[d["start_stop"]],
                   f_order=tuple(idx[n] for n in d["f_order"]), height_cm=d.get("height_cm"),
                   tilt_deg=d.get("tilt_deg"), status=str(d.get("status", "")))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POSITIONS_PATH) -> Positions:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": POSITIONS_SCHEMA, "radius_cm": self.radius_cm, "height_cm": self.height_cm, "tilt_deg": self.tilt_deg,
                "stops": [{"name": s.name, "azimuth_deg": s.azimuth_deg} for s in self.stops],
                "start_stop": self.stops[self.start_index].name,
                "f_order": [self.stops[i].name for i in self.f_order], "status": self.status}

    @property
    def n_stops(self) -> int:
        return len(self.stops)

    def azimuth(self, i: int) -> float:
        return self.stops[i].azimuth_deg

    def name(self, i: int) -> str:
        return self.stops[i].name

    def angle_deg(self, i: int, j: int) -> float:
        """Absolute angular separation between two stops, degrees."""
        return abs(self.stops[i].azimuth_deg - self.stops[j].azimuth_deg)

    def travel_cm(self, i: int, j: int) -> float:
        """Commanded travel between two stops (arc length along the template)."""
        return math.radians(self.angle_deg(i, j)) * self.radius_cm

    def path_travel_cm(self, seq: Sequence[int]) -> float:
        return sum(self.travel_cm(a, b) for a, b in zip(seq, seq[1:]))

    def span_cm(self) -> float:
        """Largest single move (end to end); the default travel normaliser."""
        return max(self.travel_cm(i, j) for i in range(self.n_stops) for j in range(self.n_stops))

    def nominal_step_cm(self) -> float:
        """Nominal arc step: the arc from the lowest to the highest azimuth divided by n_stops - 1.

        With the design values (30 deg spacing, r = 45 cm) this is 23.5619 cm, exactly the
        adjacent-stop travel. Travel caps are defined as multiples of it (cap_cm_from_steps).
        """
        if self.n_stops < 2:
            raise ValueError("need at least 2 stops for a step")
        az = sorted(s.azimuth_deg for s in self.stops)
        return math.radians(az[-1] - az[0]) * self.radius_cm / (self.n_stops - 1)

    def adjacent_step_deviation_cm(self) -> float:
        """Largest |adjacent-stop travel - nominal step| (0 for evenly spaced stops).

        A non-zero value means a cap of N steps no longer buys exactly N adjacent moves.
        """
        order = sorted(range(self.n_stops), key=self.azimuth)
        step = self.nominal_step_cm()
        return max(abs(self.travel_cm(a, b) - step) for a, b in zip(order, order[1:]))


def cap_cm_from_steps(steps: float | None, positions: Positions) -> float | None:
    """Travel cap in cm for a cap given in arc steps; None or inf -> None (uncapped). No slack is added:
    affordability checks use TRAVEL_TOL_CM instead."""
    if steps is None or math.isinf(steps):
        return None
    if steps < 0 or math.isnan(steps):
        raise ValueError(f"cap steps must be >= 0, got {steps}")
    return float(steps) * positions.nominal_step_cm()


def cap_steps_from_cm(cap_cm: float | None, positions: Positions) -> float | None:
    """Inverse of cap_cm_from_steps (rounded to 6 decimals for labels); None -> None."""
    return None if cap_cm is None else round(cap_cm / positions.nominal_step_cm(), 6)


# ---------------------------------------------------------------- evidence

def _unit(name: str, x: float) -> float:
    x = float(x)
    if not (0.0 <= x <= 1.0) or math.isnan(x):
        raise ValueError(f"{name} must be in [0, 1], got {x}")
    return x


@dataclass(frozen=True)
class EvidenceRecord:
    """One view's evidence (perception/evidence.py later; synthetic now). Scores are not probabilities."""
    stop: int
    match_score: float   # best reference match in/near the expected region
    quality: float       # exposure / focus
    obstruction: float   # occlusion cue: non-target overlap with the expected region

    def __post_init__(self) -> None:
        for n in ("match_score", "quality", "obstruction"):
            object.__setattr__(self, n, _unit(n, getattr(self, n)))
        if int(self.stop) != self.stop or self.stop < 0:
            raise ValueError("stop must be a non-negative int")


@dataclass(frozen=True)
class Observation:
    """What a policy sees for a stop it requested. No paths, no ids, no truth."""
    step: int            # 0 = initial view, 1.. = additional views in request order
    stop: int
    azimuth_deg: float
    match_score: float
    quality: float
    obstruction: float

    @classmethod
    def from_record(cls, rec: EvidenceRecord, step: int, positions: Positions) -> Observation:
        return cls(step=step, stop=rec.stop, azimuth_deg=positions.azimuth(rec.stop),
                   match_score=rec.match_score, quality=rec.quality, obstruction=rec.obstruction)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Observation:
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


OBSERVATION_FIELDS = frozenset(f.name for f in dataclasses.fields(Observation))


@dataclass(frozen=True)
class ReplayKey:
    """Opaque handle to one episode's evidence (README section 2).

    The token is random and says nothing about the episode (no id, no condition), so a
    ReplayEnv built from it never holds anything truth-bearing. Mint keys with
    EvidenceStore.replay_key(s) before the store is sealed.
    """
    token: str


class EvidenceStore:
    """episode_id -> stop -> EvidenceRecord. The only holder of evidence.

    Before sealing it is an ordinary container (generators, loaders, tests) and can mint
    opaque ReplayKeys. The first bind() seals it: after that get(), items(), episode_ids(),
    put() and minting raise LeakError, and bind() only accepts a key minted before sealing.
    State lives in name-mangled slots; that stops accidental and single-underscore access,
    not deliberate introspection (README section 2, "Guarantee and its boundary").
    """

    __slots__ = ("__data", "__n_stops", "__sealed", "__keys", "__by_token", "synthetic", "source")

    def __init__(self, n_stops: int = 5, *, synthetic: bool = False, source: str = "") -> None:
        self.__data: dict[str, dict[int, EvidenceRecord]] = {}
        self.__n_stops = n_stops
        self.__sealed = False
        self.__keys: dict[str, ReplayKey] = {}
        self.__by_token: dict[str, str] = {}
        self.synthetic = synthetic
        self.source = source

    @property
    def sealed(self) -> bool:
        return self.__sealed

    @property
    def n_stops(self) -> int:
        return self.__n_stops

    def _check_open(self, what: str) -> None:
        if self.__sealed:
            raise LeakError(f"EvidenceStore is sealed; {what} is not allowed (use ReplayEnv.request)")

    def put(self, episode_id: str, record: EvidenceRecord) -> None:
        self._check_open("put")
        if not 0 <= record.stop < self.__n_stops:
            raise ValueError(f"stop {record.stop} out of range")
        per = self.__data.setdefault(episode_id, {})
        if record.stop in per:
            raise ValueError(f"duplicate evidence for {episode_id} stop {record.stop}")
        per[record.stop] = record

    def get(self, episode_id: str, stop: int) -> EvidenceRecord:
        self._check_open("get")
        return self.__data[episode_id][stop]

    def episode_ids(self) -> tuple[str, ...]:
        self._check_open("episode_ids")
        return tuple(sorted(self.__data))

    def items(self) -> Iterable[tuple[str, EvidenceRecord]]:
        self._check_open("items")
        for eid in sorted(self.__data):
            for s in sorted(self.__data[eid]):
                yield eid, self.__data[eid][s]

    def __contains__(self, episode_id: object) -> bool:
        return episode_id in self.__data

    def __len__(self) -> int:
        return len(self.__data)

    def seal(self) -> None:
        self.__sealed = True

    def replay_key(self, episode_id: str) -> ReplayKey:
        """Opaque key for one complete episode (same key on repeat calls). Unsealed store only."""
        self._check_open("replay_key")
        per = self.__data.get(episode_id)
        if per is None or sorted(per) != list(range(self.__n_stops)):
            raise ReplayError(f"no complete evidence for episode {episode_id!r}")
        key = self.__keys.get(episode_id)
        if key is None:
            key = ReplayKey(secrets.token_hex(16))
            self.__keys[episode_id] = key
            self.__by_token[key.token] = episode_id
        return key

    def replay_keys(self) -> dict[str, ReplayKey]:
        """episode_id -> ReplayKey for every episode. Unsealed store only (mint before the run)."""
        self._check_open("replay_keys")
        return {eid: self.replay_key(eid) for eid in sorted(self.__data)}

    def bind(self, key: ReplayKey) -> Callable[[int], EvidenceRecord]:
        """Seal the store and return reveal(stop) for the keyed episode (ReplayEnv only)."""
        if not isinstance(key, ReplayKey):
            raise LeakError(f"bind needs a ReplayKey minted before sealing, got {type(key).__name__}")
        eid = self.__by_token.get(key.token)
        if eid is None:
            raise LeakError("unknown ReplayKey (keys are minted by this store before it is sealed)")
        self.__sealed = True
        frozen = dict(self.__data[eid])

        def reveal(stop: int) -> EvidenceRecord:
            return frozen[stop]
        return reveal

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            for eid, rec in self.items():
                f.write(json.dumps({"episode_id": eid, **dataclasses.asdict(rec),
                                    "synthetic": self.synthetic}) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path, n_stops: int = 5, *, source: str = "") -> EvidenceStore:
        store: EvidenceStore | None = None
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if store is None:
                store = cls(n_stops, synthetic=bool(d.get("synthetic", False)), source=source or str(path))
            store.put(d["episode_id"], EvidenceRecord(d["stop"], d["match_score"], d["quality"], d["obstruction"]))
        return store if store is not None else cls(n_stops, source=source or str(path))


# ---------------------------------------------------------------- episodes and truth

def episode_id(arrangement_id: str, condition: Condition | str, lighting: Lighting | str) -> str:
    """Stable, readable id, e.g. 'A01-occluded-L2'. Never given to policies."""
    return f"{arrangement_id}-{Condition(condition).value}-{Lighting(lighting).value}"


@dataclass(frozen=True)
class EpisodeKey:
    arrangement_id: str
    condition: Condition
    lighting: Lighting

    @property
    def episode_id(self) -> str:
        return episode_id(self.arrangement_id, self.condition, self.lighting)


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode for replay and logging. Deliberately carries no ground-truth field."""
    arrangement_id: str
    condition: Condition
    lighting: Lighting
    target_id: str
    split: Split

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", Condition(self.condition))
        object.__setattr__(self, "lighting", Lighting(self.lighting))
        object.__setattr__(self, "split", Split(self.split))

    @property
    def key(self) -> EpisodeKey:
        return EpisodeKey(self.arrangement_id, self.condition, self.lighting)

    @property
    def episode_id(self) -> str:
        return episode_id(self.arrangement_id, self.condition, self.lighting)

    def to_dict(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id, "arrangement_id": self.arrangement_id,
                "condition": self.condition.value, "lighting": self.lighting.value,
                "target_id": self.target_id, "split": self.split.value}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EpisodeSpec:
        return cls(d["arrangement_id"], d["condition"], d["lighting"], d["target_id"], d["split"])


@dataclass(frozen=True)
class GroundTruth:
    """Scoring-only: episode_id -> TruthState, plus the predeclared indistinguishable subset."""
    states: Mapping[str, TruthState]
    indistinguishable: frozenset[str] = frozenset()
    synthetic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", {k: TruthState(v) for k, v in self.states.items()})
        object.__setattr__(self, "indistinguishable", frozenset(self.indistinguishable))
        extra = self.indistinguishable - set(self.states)
        if extra:
            raise ValueError(f"indistinguishable ids without truth: {sorted(extra)}")

    @classmethod
    def from_specs(cls, specs: Iterable[EpisodeSpec], indistinguishable_arrangements: Iterable[str] = (),
                   *, synthetic: bool = False) -> GroundTruth:
        """Truth from condition; indistinguishable = occluded + removed episodes of the listed arrangements."""
        specs = list(specs)
        ind_arr = set(indistinguishable_arrangements)
        ind = {s.episode_id for s in specs
               if s.arrangement_id in ind_arr and s.condition in (Condition.OCCLUDED, Condition.REMOVED)}
        return cls({s.episode_id: truth_from_condition(s.condition) for s in specs}, frozenset(ind), synthetic)

    def __getitem__(self, eid: str) -> TruthState:
        return self.states[eid]

    def __contains__(self, eid: object) -> bool:
        return eid in self.states

    def is_indistinguishable(self, eid: str) -> bool:
        return eid in self.indistinguishable

    def to_dict(self) -> dict[str, Any]:
        return {"schema": TRUTH_SCHEMA, "synthetic": self.synthetic,
                "label": SYNTHETIC_LABEL if self.synthetic else "",
                "states": {k: v.value for k, v in sorted(self.states.items())},
                "indistinguishable": sorted(self.indistinguishable)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GroundTruth:
        return cls(d["states"], frozenset(d.get("indistinguishable", ())), bool(d.get("synthetic", False)))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> GroundTruth:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- settings

@dataclass(frozen=True)
class Thresholds:
    """Stopping-rule and G settings. Defaults are SYNTHETIC placeholders until dev selection (D10)."""
    tau_match: float = 0.6        # present if any match_score >= tau_match
    tau_q: float = 0.5            # clear miss needs quality >= tau_q
    tau_occ: float = 0.3          # clear miss needs obstruction <= tau_occ; flagged occluded if > tau_occ
    k: int = 2                    # distinct clear-miss stops needed for REMOVED (k=1: first-clear-miss)
    # G travel weight, DIMENSIONLESS (D-norm): G scores sep/max_sep - lam * travel/max_travel, both maxima
    # taken per decision over G's candidate set. 0.5 is a placeholder, chosen on dev (verify/select.py grid).
    # At the design geometry every lam < 2/3 gives the same choices, so 0.5 behaves exactly like lam = 0
    # (no travel penalty); see verify/lamgrid.py (review F1).
    lam: float = 0.5
    # Fixed travel normaliser used ONLY by the legacy G-myopic ablation (P1<->P5 arc at r = 45 cm).
    travel_norm_cm: float = 94.24777960769379

    def __post_init__(self) -> None:
        for n in ("tau_match", "tau_q", "tau_occ"):
            _unit(n, getattr(self, n))
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if not (math.isfinite(self.travel_norm_cm) and self.travel_norm_cm > 0):
            raise ValueError("travel_norm_cm must be finite and > 0")
        if not (math.isfinite(self.lam) and self.lam >= 0):   # NaN/inf would make G silently tie (review F3)
            raise ValueError(f"lam must be finite and >= 0, got {self.lam!r}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Budget:
    """additional_views: views beyond the initial one (0..4). travel_cap_cm: None = unlimited."""
    additional_views: int
    travel_cap_cm: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.additional_views <= 4:
            raise ValueError("additional_views must be in 0..4")
        if self.travel_cap_cm is not None and self.travel_cap_cm < 0:
            raise ValueError("travel_cap_cm must be >= 0 or None")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RunContext:
    """Provenance copied into every EpisodeLog."""
    model_id: str = "synthetic"           # detector/encoder checkpoint id (+sha) for real data
    config_version: str = "synthetic-v0"
    config_sha256: str = ""
    synthetic: bool = True


# ---------------------------------------------------------------- logs and results

@dataclass(frozen=True)
class EpisodeRun:
    """What ReplayEnv.finish returns: one run with no episode header (the env never sees the spec).

    The driver joins the header with EpisodeLog.from_run(spec, run) outside the env.
    """
    model_id: str
    config_version: str
    config_sha256: str
    policy: str
    budget_additional_views: int
    travel_cap_cm: float | None
    seed: int | None
    requested_stops: tuple[int, ...]
    evidence: tuple[Observation, ...]
    decision: Decision
    travel_cm: float
    n_views: int
    n_additional_views: int
    synthetic: bool = False


@dataclass(frozen=True)
class EpisodeLog:
    """One run of one policy on one episode (EXPERIMENT_PLAN 'Logs'). No ground truth."""
    episode_id: str
    arrangement_id: str
    split: Split
    target_id: str
    condition: Condition
    lighting: Lighting
    model_id: str
    config_version: str
    config_sha256: str
    policy: str
    budget_additional_views: int
    travel_cap_cm: float | None
    seed: int | None                       # base seed (R); None for deterministic policies
    requested_stops: tuple[int, ...]       # in order, includes the start stop
    evidence: tuple[Observation, ...]      # revealed evidence, same order
    decision: Decision
    travel_cm: float
    n_views: int                           # total, includes the initial view
    n_additional_views: int
    synthetic: bool = False
    schema: str = LOG_SCHEMA

    @classmethod
    def from_run(cls, spec: EpisodeSpec, run: EpisodeRun) -> EpisodeLog:
        """Join the log header (outside the env) onto a header-free run."""
        if not isinstance(spec, EpisodeSpec) or not isinstance(run, EpisodeRun):
            raise TypeError("from_run needs (EpisodeSpec, EpisodeRun)")
        return cls(episode_id=spec.episode_id, arrangement_id=spec.arrangement_id, split=spec.split,
                   target_id=spec.target_id, condition=spec.condition, lighting=spec.lighting,
                   **{f.name: getattr(run, f.name) for f in dataclasses.fields(EpisodeRun)})

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d["evidence"] = [o.to_dict() for o in self.evidence]
        d["requested_stops"] = list(self.requested_stops)
        for k in ("split", "condition", "lighting", "decision"):
            d[k] = d[k].value
        if self.synthetic:
            d["label"] = SYNTHETIC_LABEL
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EpisodeLog:
        kw = {f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d}
        kw["split"], kw["condition"] = Split(d["split"]), Condition(d["condition"])
        kw["lighting"], kw["decision"] = Lighting(d["lighting"]), Decision(d["decision"])
        kw["requested_stops"] = tuple(d["requested_stops"])
        kw["evidence"] = tuple(Observation.from_dict(o) for o in d["evidence"])
        return cls(**kw)


@dataclass(frozen=True)
class EpisodeResult:
    """One policy x budget x episode, aggregated over runs (R: seeds averaged within episode).

    decision_frac maps every Decision to the fraction of runs that returned it (sums to 1).
    Deterministic policies have one run and 0/1 fractions. Metrics use this type.
    """
    episode_id: str
    arrangement_id: str
    split: Split
    condition: Condition
    lighting: Lighting
    policy: str
    budget_additional_views: int
    travel_cap_cm: float | None
    decision_frac: Mapping[Decision, float]
    mean_views: float
    mean_additional_views: float
    mean_travel_cm: float
    seeds: tuple[int | None, ...]
    synthetic: bool = False

    @classmethod
    def from_logs(cls, logs: Sequence[EpisodeLog]) -> EpisodeResult:
        if not logs:
            raise ValueError("no logs")
        a = logs[0]
        for g in logs[1:]:
            if (g.episode_id, g.policy, g.budget_additional_views, g.travel_cap_cm) != \
                    (a.episode_id, a.policy, a.budget_additional_views, a.travel_cap_cm):
                raise ValueError("logs mix episodes, policies or budgets")
        n = len(logs)
        frac = {d: sum(g.decision is d for g in logs) / n for d in Decision}
        return cls(episode_id=a.episode_id, arrangement_id=a.arrangement_id, split=a.split,
                   condition=a.condition, lighting=a.lighting, policy=a.policy,
                   budget_additional_views=a.budget_additional_views, travel_cap_cm=a.travel_cap_cm,
                   decision_frac=frac, mean_views=sum(g.n_views for g in logs) / n,
                   mean_additional_views=sum(g.n_additional_views for g in logs) / n,
                   mean_travel_cm=sum(g.travel_cm for g in logs) / n,
                   seeds=tuple(g.seed for g in logs), synthetic=all(g.synthetic for g in logs))

    def frac(self, d: Decision) -> float:
        return float(self.decision_frac.get(d, 0.0))

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        for k in ("split", "condition", "lighting"):
            d[k] = d[k].value
        d["decision_frac"] = {k.value: v for k, v in self.decision_frac.items()}
        d["seeds"] = list(self.seeds)
        d["schema"] = RESULT_SCHEMA
        return d


def as_results(items: Iterable[EpisodeLog | EpisodeResult]) -> list[EpisodeResult]:
    """Normalise to EpisodeResult. Bare EpisodeLogs are grouped by (episode, policy, budget, cap)."""
    out: list[EpisodeResult] = []
    groups: dict[tuple, list[EpisodeLog]] = {}
    for x in items:
        if isinstance(x, EpisodeResult):
            out.append(x)
        elif isinstance(x, EpisodeLog):
            groups.setdefault((x.episode_id, x.policy, x.budget_additional_views, x.travel_cap_cm), []).append(x)
        else:
            raise TypeError(f"expected EpisodeLog or EpisodeResult, got {type(x).__name__}")
    out.extend(EpisodeResult.from_logs(g) for g in groups.values())
    return out


# ---------------------------------------------------------------- hashing and seeds

def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace); enums and dataclasses flattened."""
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"))


def config_sha256(config: Any) -> str:
    """sha256 hex of canonical_json(config)."""
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_seed(*parts: Any) -> int:
    """Stable 63-bit seed from any parts (e.g. base seed + episode_id); same on every machine."""
    return int.from_bytes(hashlib.sha256(canonical_json(list(parts)).encode()).digest()[:8], "big") >> 1


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "to_dict") and not isinstance(obj, type):
        return to_jsonable(obj.to_dict())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, StrEnum):
        return obj.value
    if isinstance(obj, Mapping):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        return obj.item()
    return obj


def split_arrangements(arrangement_ids: Sequence[str], n_dev: int, seed: int) -> dict[str, Split]:
    """Deterministic arrangement-level dev/test split: n_dev arrangements to DEV, the rest to TEST."""
    ids = sorted(set(arrangement_ids))
    if len(ids) != len(arrangement_ids):
        raise ManifestError("duplicate arrangement ids")
    if not 0 <= n_dev <= len(ids):
        raise ManifestError("n_dev out of range")
    ranked = sorted(ids, key=lambda a: derive_seed("split", seed, a))
    dev = set(ranked[:n_dev])
    return {a: (Split.DEV if a in dev else Split.TEST) for a in ids}

