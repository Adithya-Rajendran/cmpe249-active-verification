"""Dev-only threshold selection (D-select; lit report notes/2026-10-02-caps-thresholds-literature.md sec. 3).

    uv run --no-sync python -m verify.select --synthetic [--n-arr 30] [--seed 0] [--policy G] [--budget 3]
        [--cap-steps inf] [--tau-match-grid 0.40,...] [--k-grid 3,2] [--lam-grid 0,0.7,0.9,1.2]
        [--tau-q-grid ...] [--tau-occ-grid ...] [--alpha-clear 0.10] [--frr-limit 0.10]
        [--recall-target 0.80] [--bootstrap-B 2000] [--tag TAG] [--out runs/select-<tag>/]
    uv run --no-sync python -m verify.select --lam-breakpoints [--positions configs/positions.json]
        (prints the lam values at which G's choice changes on that geometry, and checks --lam-grid)

Procedure (every number it produces on --synthetic data is SYNTHETIC and is never a real threshold):

  Step 2  tau_q / tau_occ from per-view diagnostic labels, Neyman-Pearson style: over a fixed
          (tau_q, tau_occ) grid, keep gates whose false-clear rate on non-diagnostic views of
          occluded-present episodes is <= alpha_clear, then maximise the clear rate on diagnostic
          views of removed episodes. Ties: lower false-clear rate, higher tau_q, lower tau_occ.
  Step 3  Fixed-sequence walk (Learn-then-Test style) over a predeclared conservative -> liberal
          order of (k, tau_match): k in the given order (default 3 then 2), tau_match ascending
          within k. A step passes if its WORST false-removal rate over the lam grid is <= frr_limit,
          so the later lam choice can never be what rescues the FRR constraint. The walk accepts the
          prefix of steps before the first failure. Among accepted (k, tau_match, lam) configs the
          D10 objective keeps those with recall >= recall_target if any, maximises coverage, then
          breaks ties by recall, lower mean travel, then grid order.
  Step 4  Nested grouped CV: outer leave-one-arrangement-out; the inner selection is the same
          rule (Steps 2 + 3) run on the remaining arrangements only. The held-out decisions,
          pooled over folds, are the honest dev estimate. The candidate to freeze is the rule
          applied to all dev arrangements.

Bounds: one-sided 95% Clopper-Pearson upper bound on FRR at the episode level and at the
arrangement level ("any false removal in the arrangement", the 0/10-style bound), lower bound
on recall (and coverage), the same with Jeffreys, plus a percentile cluster bootstrap over
arrangements. Episodes of one arrangement are correlated, so the episode-level bound is
optimistic; no formal (alpha, delta) guarantee is possible with about 10 dev arrangements.

Data access: this module is offline dev analysis, not a policy. It reads every dev view for
Step 2 (from a fresh unsealed store), and joins ground truth to score dev runs. Policies still
see only requested views: every config is run through the public ReplayEnv / run_episode API on
a fresh EvidenceStore from a StoreFactory (one store per config; keys are minted before sealing).
Only DEV-split episodes are accepted; any test (or pilot) episode raises SplitRefused.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from analysis.bootstrap import arrangement_draws
from analysis.intervals import (
    clopper_pearson_lower,
    clopper_pearson_upper,
    jeffreys_lower,
    jeffreys_upper,
)
from verify.lamgrid import LamGridCheck, check_lam_grid
from verify.policies import F, G, run_episode
from verify.replay import ReplayEnv
from verify.schema import (
    EPS,
    REPO_ROOT,
    SYNTHETIC_LABEL,
    Budget,
    Condition,
    Decision,
    EpisodeSpec,
    EvidenceRecord,
    EvidenceStore,
    GroundTruth,
    ManifestError,
    Positions,
    RunContext,
    Split,
    Thresholds,
    TruthState,
    cap_cm_from_steps,
    config_sha256,
    to_jsonable,
)
from verify.stopping import StoppingRule
from verify.synthetic import SyntheticParams, generate, generate_layouts

log = logging.getLogger("verify.select")

PROCEDURE_VERSION = "select-v2"   # v2: lam grid from breakpoints, lam_grid_check, synthetic provenance (review F1, F4)
CANDIDATE_SCHEMA = "cmpe249.frozen_candidate.v1"
VIEW_LABELS_SCHEMA = "cmpe249.view_labels.v1"
PROPOSED = "proposed default - confirm"
SYNTHETIC_LABEL_SOURCE = ("SYNTHETIC stand-in: derived from the generator's hidden occluder layout "
                          "(blocked or edge stop -> non-diagnostic), not from human annotation")

# Proposed defaults (all "proposed default - confirm"; the score grids are placeholders until
# the catalog sweep shows real score distributions).
DEFAULT_TAU_MATCH_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
DEFAULT_K_GRID = (3, 2)                  # walk order: conservative (3) first
# lam grid (review F1): G's choice only changes at a few lam breakpoints (verify/lamgrid.py). At the design
# geometry they are 2/3, 3/4, 1, 4/3, 3/2, 2, so {0, 0.25, 0.5} would be one behaviour and 1.0 sits on a
# breakpoint. One value per interval, off the breakpoints. Re-derive with --lam-breakpoints once the
# measured azimuths are in configs/positions.json (proposed default - confirm).
DEFAULT_LAM_GRID = (0.0, 0.7, 0.9, 1.2)
DEFAULT_TAU_Q_GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(11))    # 0.30 .. 0.80
DEFAULT_TAU_OCC_GRID = tuple(round(0.10 + 0.05 * i, 2) for i in range(9))   # 0.10 .. 0.50
SELECTION_POLICIES = ("G", "F")


class SplitRefused(ManifestError):
    """Selection was handed a non-dev (test or pilot) episode."""


# ---------------------------------------------------------------- data interfaces

class StoreFactory(Protocol):
    """Returns a FRESH, UNSEALED EvidenceStore holding exactly the dev episodes, on every call.

    A ReplayEnv seals its store on bind, so selection asks for a new store per config (and one
    for the Step 2 view table). Every call must return identical evidence.
    """
    n_stops: int

    def __call__(self) -> EvidenceStore: ...


class RecordStoreFactory:
    """StoreFactory over an immutable snapshot of EvidenceRecords (copied once, rebuilt per call)."""

    def __init__(self, records: Mapping[str, Sequence[EvidenceRecord]], n_stops: int, *,
                 synthetic: bool, source: str = "") -> None:
        self._records = {eid: tuple(sorted(recs, key=lambda r: r.stop)) for eid, recs in sorted(records.items())}
        self.n_stops = n_stops
        self.synthetic = synthetic
        self.source = source

    @classmethod
    def from_store(cls, store: EvidenceStore, episode_ids: Iterable[str], *, source: str = "") -> RecordStoreFactory:
        """Snapshot the listed episodes from an UNSEALED store (e.g. a loader or the generator)."""
        ids = sorted(set(episode_ids))
        recs = {eid: [store.get(eid, s) for s in range(store.n_stops)] for eid in ids}
        return cls(recs, store.n_stops, synthetic=store.synthetic, source=source or store.source)

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def evidence_sha256(self) -> str:
        """Provenance hash of the dev evidence (canonical JSON of every record)."""
        return config_sha256({eid: [dataclasses.asdict(r) for r in recs] for eid, recs in self._records.items()})

    def __call__(self) -> EvidenceStore:
        store = EvidenceStore(self.n_stops, synthetic=self.synthetic, source=self.source)
        for eid, recs in self._records.items():
            for r in recs:
                store.put(eid, r)
        return store


@dataclass(frozen=True)
class ViewLabel:
    """Dev annotation for one view: would the target be verifiable here if present (present
    episodes), or is the expected region verifiably unobstructed (removed episodes)? Ambiguous
    views are non-diagnostic (EXPERIMENT_PLAN 'Initial algorithm specification')."""
    episode_id: str
    stop: int
    diagnostic: bool


def save_view_labels(labels: Iterable[ViewLabel], path: str | Path, *, source: str) -> None:
    doc = {"schema": VIEW_LABELS_SCHEMA, "source": source,
           "labels": [dataclasses.asdict(l) for l in sorted(labels, key=lambda l: (l.episode_id, l.stop))]}
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")


def load_view_labels(path: str | Path) -> list[ViewLabel]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != VIEW_LABELS_SCHEMA:
        raise ValueError(f"view labels schema must be {VIEW_LABELS_SCHEMA!r}, got {doc.get('schema')!r}")
    return [ViewLabel(d["episode_id"], int(d["stop"]), bool(d["diagnostic"])) for d in doc["labels"]]


@dataclass(frozen=True)
class DevData:
    """Everything selection may use. Construction refuses any non-DEV episode."""
    specs: tuple[EpisodeSpec, ...]
    store_factory: StoreFactory
    truth: GroundTruth
    view_labels: tuple[ViewLabel, ...]
    positions: Positions
    synthetic: bool
    label_source: str
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", tuple(sorted(self.specs, key=lambda s: s.episode_id)))
        object.__setattr__(self, "view_labels", tuple(self.view_labels))
        refuse_non_dev(self.specs)
        ids = {s.episode_id for s in self.specs}
        if not ids:
            raise ValueError("no dev episodes")
        store = self.store_factory()
        if store.sealed:
            raise ValueError("StoreFactory must return an unsealed store")
        extra = set(store.episode_ids()) - ids
        if extra:
            raise SplitRefused(f"store holds episodes outside the dev specs: {sorted(extra)[:5]}")
        missing = ids - set(store.episode_ids())
        if missing:
            raise ValueError(f"store lacks dev episodes: {sorted(missing)[:5]}")
        missing_truth = [e for e in ids if e not in self.truth]
        if missing_truth:
            raise KeyError(f"no ground truth for {sorted(missing_truth)[:5]}")
        lab = {(l.episode_id, l.stop) for l in self.view_labels}
        if len(lab) != len(self.view_labels):
            raise ValueError("duplicate view labels")
        if not {l.episode_id for l in self.view_labels} <= ids:
            raise SplitRefused("view labels for episodes outside the dev specs")
        need = {(e, s) for e in ids for s in range(store.n_stops)}
        if need - lab:
            raise ValueError(f"missing view labels, e.g. {sorted(need - lab)[:3]}")

    @property
    def arrangements(self) -> tuple[str, ...]:
        return tuple(sorted({s.arrangement_id for s in self.specs}))


def refuse_non_dev(specs: Iterable[EpisodeSpec]) -> None:
    """Raise SplitRefused if any spec is not DEV (test above all: never tune on test)."""
    bad = [s.episode_id for s in specs if s.split is not Split.DEV]
    if bad:
        raise SplitRefused(f"selection reads DEV arrangements only; refused {len(bad)} non-dev episodes "
                           f"(e.g. {bad[:3]})")


def synthetic_view_labels(specs: Sequence[EpisodeSpec], layouts: Mapping[str, Any], n_stops: int) -> list[ViewLabel]:
    """SYNTHETIC stand-in for annotations: visible -> all diagnostic; occluded/removed -> diagnostic
    iff the stop is neither blocked nor next to the block (edge views are partial = ambiguous)."""
    out = []
    for s in specs:
        lay = layouts[s.arrangement_id]
        hidden = set(lay.blocked) | set(lay.edges(n_stops))
        for stop in range(n_stops):
            diag = True if s.condition is Condition.VISIBLE else stop not in hidden
            out.append(ViewLabel(s.episode_id, stop, diag))
    return out


def synthetic_dev_data(n_arr: int, seed: int, params: SyntheticParams, positions: Positions) -> DevData:
    """Dev split of the synthetic generator, with stand-in labels from its hidden layout."""
    specs, store, truth = generate(n_arr, seed, params, positions)
    dev = [s for s in specs if s.split is Split.DEV]
    ids = [s.episode_id for s in dev]
    factory = RecordStoreFactory.from_store(store, ids, source=f"verify.synthetic seed={seed} (dev only)")
    dev_truth = GroundTruth({e: truth[e] for e in ids}, truth.indistinguishable & set(ids), synthetic=True)
    layouts = generate_layouts(n_arr, seed, params, n_stops=positions.n_stops, start_index=positions.start_index)
    labels = synthetic_view_labels(dev, layouts, positions.n_stops)
    return DevData(tuple(dev), factory, dev_truth, tuple(labels), positions, True, SYNTHETIC_LABEL_SOURCE,
                   factory.evidence_sha256())


# ---------------------------------------------------------------- settings

@dataclass(frozen=True)
class SelectionSettings:
    policy: str = "G"
    budget: int = 3
    cap_steps: float | None = None          # None = uncapped
    tau_match_grid: tuple[float, ...] = DEFAULT_TAU_MATCH_GRID
    k_grid: tuple[int, ...] = DEFAULT_K_GRID
    lam_grid: tuple[float, ...] = DEFAULT_LAM_GRID
    tau_q_grid: tuple[float, ...] = DEFAULT_TAU_Q_GRID
    tau_occ_grid: tuple[float, ...] = DEFAULT_TAU_OCC_GRID
    alpha_clear: float = 0.10
    frr_limit: float = 0.10
    recall_target: float = 0.80
    conf: float = 0.95
    bootstrap_B: int = 2000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.policy not in SELECTION_POLICIES:
            raise ValueError(f"policy must be one of {SELECTION_POLICIES} (deterministic), got {self.policy!r}")
        if not 1 <= self.budget <= 4:
            raise ValueError("budget must be 1..4 (selection needs at least one additional view)")
        if not self.tau_match_grid or not self.k_grid or not self.lam_grid:
            raise ValueError("grids must be non-empty")
        if len(set(self.k_grid)) != len(self.k_grid) or any(k < 1 for k in self.k_grid):
            raise ValueError("k_grid must be unique integers >= 1")
        if list(self.tau_match_grid) != sorted(set(self.tau_match_grid)):
            raise ValueError("tau_match_grid must be strictly ascending (walked conservative -> liberal)")
        if list(self.lam_grid) != sorted(set(self.lam_grid)):
            raise ValueError("lam_grid must be strictly ascending")
        for n in ("tau_match_grid", "lam_grid", "tau_q_grid", "tau_occ_grid"):
            if not all(math.isfinite(x) for x in getattr(self, n)):   # NaN breaks ordering and G's scores (F3)
                raise ValueError(f"{n} must contain finite numbers only")
        if any(x < 0 for x in self.lam_grid):
            raise ValueError("lam_grid values must be >= 0")
        if self.cap_steps is not None and not (math.isfinite(self.cap_steps) and self.cap_steps >= 0):
            raise ValueError("cap_steps must be finite and >= 0 (None = uncapped)")
        for n in ("alpha_clear", "frr_limit", "recall_target"):
            if not 0.0 <= getattr(self, n) <= 1.0:
                raise ValueError(f"{n} must be in [0, 1]")

    @property
    def effective_lam_grid(self) -> tuple[float, ...]:
        """F ignores lam: collapse the grid to the schema default so lam is not 'selected'."""
        return self.lam_grid if self.policy == "G" else (Thresholds().lam,)

    def budget_obj(self, positions: Positions) -> Budget:
        return Budget(self.budget, cap_cm_from_steps(self.cap_steps, positions))

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["cap_steps"] = "uncapped" if self.cap_steps is None else self.cap_steps
        return d


# ---------------------------------------------------------------- evaluation

@dataclass(frozen=True)
class Outcome:
    """One dev episode under one config (truth joined; scoring side only)."""
    episode_id: str
    arrangement_id: str
    present: bool
    decision: Decision
    travel_cm: float
    n_views: int


@dataclass(frozen=True)
class Counts:
    n_episodes: int = 0
    n_present: int = 0
    n_false_removal: int = 0
    n_removed: int = 0
    n_true_removal: int = 0
    n_resolved: int = 0
    total_travel_cm: float = 0.0
    n_arrangements: int = 0
    n_arr_with_present: int = 0
    n_arr_with_false_removal: int = 0

    @staticmethod
    def of(outcomes: Iterable[Outcome]) -> Counts:
        outs = list(outcomes)
        arr_present: dict[str, bool] = {}
        arr_fr: dict[str, bool] = {}
        for o in outs:
            arr_present[o.arrangement_id] = arr_present.get(o.arrangement_id, False) or o.present
            fr = o.present and o.decision is Decision.REMOVED
            arr_fr[o.arrangement_id] = arr_fr.get(o.arrangement_id, False) or fr
        return Counts(
            n_episodes=len(outs),
            n_present=sum(o.present for o in outs),
            n_false_removal=sum(o.present and o.decision is Decision.REMOVED for o in outs),
            n_removed=sum(not o.present for o in outs),
            n_true_removal=sum((not o.present) and o.decision is Decision.REMOVED for o in outs),
            n_resolved=sum(o.decision is not Decision.UNKNOWN for o in outs),
            total_travel_cm=sum(o.travel_cm for o in outs),
            n_arrangements=len(arr_present),
            n_arr_with_present=sum(arr_present.values()),
            n_arr_with_false_removal=sum(arr_fr.values()))

    @property
    def frr(self) -> float | None:
        return self.n_false_removal / self.n_present if self.n_present else None

    @property
    def recall(self) -> float | None:
        return self.n_true_removal / self.n_removed if self.n_removed else None

    @property
    def coverage(self) -> float | None:
        return self.n_resolved / self.n_episodes if self.n_episodes else None

    @property
    def mean_travel_cm(self) -> float | None:
        return self.total_travel_cm / self.n_episodes if self.n_episodes else None

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "frr": self.frr, "recall": self.recall,
                "coverage": self.coverage, "mean_travel_cm": self.mean_travel_cm}


def make_policy(name: str, th: Thresholds) -> Any:
    """Build a fresh policy through its public constructor (G gets lam/thresholds via from_thresholds)."""
    if name == "G":
        return G.from_thresholds(th)
    if name == "F":
        return F()
    raise ValueError(f"unsupported selection policy {name!r}")


class Evaluator:
    """Runs (and caches) every dev episode under a Thresholds config via the public replay API."""

    def __init__(self, dev: DevData, settings: SelectionSettings, context: RunContext) -> None:
        self.dev = dev
        self.policy = settings.policy
        self.budget = settings.budget_obj(dev.positions)
        self.context = context
        self._cache: dict[Thresholds, dict[str, Outcome]] = {}
        self.n_runs = 0

    def outcomes(self, th: Thresholds) -> dict[str, Outcome]:
        hit = self._cache.get(th)
        if hit is not None:
            return hit
        store = self.dev.store_factory()           # fresh store per config
        keys = store.replay_keys()                 # minted before the first bind seals it
        rule = StoppingRule(th)
        out: dict[str, Outcome] = {}
        for spec in self.dev.specs:
            env = ReplayEnv(keys[spec.episode_id], store, self.dev.positions, self.budget, self.context)
            run = run_episode(make_policy(self.policy, th), env, rule)
            out[spec.episode_id] = Outcome(spec.episode_id, spec.arrangement_id,
                                           self.dev.truth[spec.episode_id] is TruthState.PRESENT,
                                           run.decision, run.travel_cm, run.n_views)
            self.n_runs += 1
        self._cache[th] = out
        return out

    def subset(self, th: Thresholds, arrangements: Iterable[str]) -> list[Outcome]:
        keep = set(arrangements)
        return [o for o in self.outcomes(th).values() if o.arrangement_id in keep]

    def counts(self, th: Thresholds, arrangements: Iterable[str]) -> Counts:
        return Counts.of(self.subset(th, arrangements))


# ---------------------------------------------------------------- Step 2: visibility gate

@dataclass(frozen=True)
class ViewRow:
    arrangement_id: str
    episode_id: str
    condition: Condition
    stop: int
    quality: float
    obstruction: float
    diagnostic: bool


def view_rows(dev: DevData) -> list[ViewRow]:
    """Every dev view with its label (offline dev analysis, read from a fresh unsealed store)."""
    store = dev.store_factory()
    lab = {(l.episode_id, l.stop): l.diagnostic for l in dev.view_labels}
    rows = []
    for s in dev.specs:
        for stop in range(store.n_stops):
            r = store.get(s.episode_id, stop)
            rows.append(ViewRow(s.arrangement_id, s.episode_id, s.condition, stop, r.quality, r.obstruction,
                                lab[(s.episode_id, stop)]))
    return rows


@dataclass(frozen=True)
class GateChoice:
    tau_q: float
    tau_occ: float
    false_clear: tuple[int, int]     # (passing, total) non-diagnostic occluded-present views
    clear: tuple[int, int]           # (passing, total) diagnostic removed-episode views
    met_alpha: bool
    n_feasible: int
    n_grid: int

    @property
    def false_clear_rate(self) -> float | None:
        return self.false_clear[0] / self.false_clear[1] if self.false_clear[1] else None

    @property
    def clear_rate(self) -> float | None:
        return self.clear[0] / self.clear[1] if self.clear[1] else None

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "false_clear_rate": self.false_clear_rate, "clear_rate": self.clear_rate}


def passes_gate(r: ViewRow, tau_q: float, tau_occ: float) -> bool:
    """Same boundaries as StoppingRule.is_clear_miss's gate: quality >= tau_q and obstruction <= tau_occ."""
    return r.quality >= tau_q and r.obstruction <= tau_occ


def choose_gate(rows: Sequence[ViewRow], tau_q_grid: Sequence[float], tau_occ_grid: Sequence[float],
                alpha_clear: float, warnings: list[str] | None = None) -> GateChoice:
    """NP rule: false-clear rate <= alpha_clear, then max clear rate; ties lower FCR, higher tau_q, lower tau_occ.
    If no gate meets alpha_clear, take the lowest false-clear rate (then the same tie-breaks) and warn."""
    null = [r for r in rows if r.condition is Condition.OCCLUDED and not r.diagnostic]
    power = [r for r in rows if r.condition is Condition.REMOVED and r.diagnostic]
    warn = warnings if warnings is not None else []
    if not null:
        warn.append("Step 2: no non-diagnostic occluded-present views; the false-clear constraint is vacuous")
    if not power:
        warn.append("Step 2: no diagnostic removed-episode views; clear rate is 0 for every gate")
    cands = []
    for tq in tau_q_grid:
        for to in tau_occ_grid:
            fc = sum(passes_gate(r, tq, to) for r in null)
            cl = sum(passes_gate(r, tq, to) for r in power)
            fcr = fc / len(null) if null else 0.0
            clr = cl / len(power) if power else 0.0
            cands.append((fcr, clr, tq, to, fc, cl))
    feasible = [c for c in cands if c[0] <= alpha_clear + EPS]
    if feasible:
        best = min(feasible, key=lambda c: (-c[1], c[0], -c[2], c[3]))
    else:
        best = min(cands, key=lambda c: (c[0], -c[1], -c[2], c[3]))
        warn.append(f"Step 2: no gate meets alpha_clear = {alpha_clear}; took the lowest false-clear rate "
                    f"({best[0]:.3f})")
    for w in warn:
        log.warning(w)
    return GateChoice(best[2], best[3], (best[4], len(null)), (best[5], len(power)), bool(feasible),
                      len(feasible), len(cands))


# ---------------------------------------------------------------- Step 3: fixed-sequence walk + D10

@dataclass(frozen=True)
class GridConfig:
    order: int          # position in the predeclared walk (k-major, tau_match ascending, lam ascending)
    step: int           # position of the (k, tau_match) step
    k: int
    tau_match: float
    lam: float

    def thresholds(self, gate: GateChoice) -> Thresholds:
        return Thresholds(tau_match=self.tau_match, tau_q=gate.tau_q, tau_occ=gate.tau_occ, k=self.k, lam=self.lam)


def walk_order(settings: SelectionSettings) -> list[GridConfig]:
    out, step = [], 0
    for k in settings.k_grid:
        for tm in settings.tau_match_grid:
            for lam in settings.effective_lam_grid:
                out.append(GridConfig(len(out), step, k, tm, lam))
            step += 1
    return out


@dataclass(frozen=True)
class WalkStep:
    step: int
    k: int
    tau_match: float
    rows: tuple[tuple[GridConfig, Counts], ...]
    worst_frr: float | None
    passed: bool
    accepted: bool      # inside the accepted prefix (no earlier failure)


@dataclass
class Selection:
    gate: GateChoice
    walk: list[WalkStep]
    chosen: GridConfig
    chosen_counts: Counts
    meets_frr_limit: bool
    recall_constraint_met: bool
    arrangements: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)

    @property
    def thresholds(self) -> Thresholds:
        return self.chosen.thresholds(self.gate)


def d10_key(cfg: GridConfig, c: Counts) -> tuple:
    cov = c.coverage if c.coverage is not None else -1.0
    rec = c.recall if c.recall is not None else -1.0
    trav = c.mean_travel_cm if c.mean_travel_cm is not None else math.inf
    return (-round(cov, 12), -round(rec, 12), round(trav, 9), cfg.order)


def d10_choose(cands: Sequence[tuple[GridConfig, Counts]], recall_target: float) -> tuple[GridConfig, Counts, bool]:
    """Keep recall >= target if any; maximise coverage; tie-break recall, lower mean travel, grid order."""
    if not cands:
        raise ValueError("no candidates")
    ok = [c for c in cands if c[1].recall is not None and c[1].recall >= recall_target - EPS]
    pool = ok or list(cands)
    cfg, cnt = min(pool, key=lambda c: d10_key(*c))
    return cfg, cnt, bool(ok)


def run_walk(ev: Evaluator, gate: GateChoice, settings: SelectionSettings,
             arrangements: Sequence[str]) -> list[WalkStep]:
    grid = walk_order(settings)
    steps: dict[int, list[GridConfig]] = {}
    for g in grid:
        steps.setdefault(g.step, []).append(g)
    out, failed = [], False
    for s, cfgs in sorted(steps.items()):
        rows = tuple((g, ev.counts(g.thresholds(gate), arrangements)) for g in cfgs)
        frrs = [c.frr for _, c in rows]
        worst = None if any(f is None for f in frrs) else max(frrs)  # type: ignore[type-var]
        passed = worst is not None and worst <= settings.frr_limit + EPS
        accepted = passed and not failed
        failed = failed or not passed
        out.append(WalkStep(s, cfgs[0].k, cfgs[0].tau_match, rows, worst, passed, accepted))
    return out


def monotonicity_warnings(walk: Sequence[WalkStep], settings: SelectionSettings) -> list[str]:
    """Empirical checks of the orderings the fixed-sequence walk assumes (never assumed)."""
    warn = []
    first_fail = next((w.step for w in walk if not w.passed), None)
    if first_fail is not None:
        late = [w for w in walk if w.step > first_fail and w.passed]
        if late:
            warn.append(f"walk stopped at step {first_fail} (k={walk[first_fail].k}, tau_match="
                        f"{walk[first_fail].tau_match}) but {len(late)} later step(s) would pass the FRR limit, e.g. "
                        f"k={late[0].k}, tau_match={late[0].tau_match}: FRR is not monotone along the walk")
    for a, b in zip(walk, walk[1:]):
        if a.worst_frr is not None and b.worst_frr is not None and b.worst_frr < a.worst_frr - EPS:
            warn.append(f"worst FRR decreases along the walk: step {a.step} (k={a.k}, tau_match={a.tau_match}) "
                        f"{a.worst_frr:.3f} -> step {b.step} (k={b.k}, tau_match={b.tau_match}) {b.worst_frr:.3f}")
    table = {(g.k, g.tau_match, g.lam): c.frr for w in walk for g, c in w.rows}
    for lam in settings.effective_lam_grid:
        for k in settings.k_grid:
            seq = [table[(k, t, lam)] for t in settings.tau_match_grid]
            for t0, t1, f0, f1 in zip(settings.tau_match_grid, settings.tau_match_grid[1:], seq, seq[1:]):
                if f0 is not None and f1 is not None and f1 < f0 - EPS:
                    warn.append(f"FRR not non-decreasing in tau_match (k={k}, lam={lam}): "
                                f"{t0} -> {f0:.3f}, {t1} -> {f1:.3f}")
        ks = sorted(settings.k_grid)
        for t in settings.tau_match_grid:
            for k_lo, k_hi in zip(ks, ks[1:]):
                f_lo, f_hi = table[(k_lo, t, lam)], table[(k_hi, t, lam)]
                if f_lo is not None and f_hi is not None and f_hi > f_lo + EPS:
                    warn.append(f"FRR higher at k={k_hi} than k={k_lo} (tau_match={t}, lam={lam}): "
                                f"{f_hi:.3f} > {f_lo:.3f}")
    return warn


def select(ev: Evaluator, rows: Sequence[ViewRow], settings: SelectionSettings,
           arrangements: Sequence[str], *, check_monotone: bool = False) -> Selection:
    """Steps 2 + 3 on the given arrangements only (the inner rule of the nested CV)."""
    arrangements = tuple(sorted(arrangements))
    keep = set(arrangements)
    warnings: list[str] = []
    gate = choose_gate([r for r in rows if r.arrangement_id in keep], settings.tau_q_grid,
                       settings.tau_occ_grid, settings.alpha_clear, warnings)
    walk = run_walk(ev, gate, settings, arrangements)
    accepted = [(g, c) for w in walk if w.accepted for g, c in w.rows]
    if accepted:
        cfg, cnt, rec_ok = d10_choose(accepted, settings.recall_target)
        meets = True
    else:
        # Nothing meets the FRR limit: fall back to the most conservative step, lam by D10, and say so.
        cfg, cnt, rec_ok = d10_choose(list(walk[0].rows), settings.recall_target)
        meets = False
        warnings.append("Step 3: no (k, tau_match) step meets the FRR limit; candidate is the most conservative "
                        "step and does NOT satisfy D10")
    if check_monotone:
        for w in monotonicity_warnings(walk, settings):
            log.warning(w)
            warnings.append(w)
    return Selection(gate, walk, cfg, cnt, meets, rec_ok, arrangements, warnings)


# ---------------------------------------------------------------- Step 4: nested CV + bounds

@dataclass(frozen=True)
class Fold:
    held_out: str
    selection_thresholds: Thresholds
    meets_frr_limit: bool
    train_counts: Counts
    held_counts: Counts


def nested_cv(ev: Evaluator, rows: Sequence[ViewRow], settings: SelectionSettings,
              arrangements: Sequence[str]) -> tuple[list[Fold], list[Outcome]]:
    """Outer leave-one-arrangement-out; inner = select() on the other arrangements only."""
    folds, pooled = [], []
    for a in sorted(arrangements):
        train = [x for x in arrangements if x != a]
        sel = select(ev, rows, settings, train)
        held = ev.subset(sel.thresholds, [a])
        pooled.extend(held)
        folds.append(Fold(a, sel.thresholds, sel.meets_frr_limit, sel.chosen_counts, Counts.of(held)))
    return folds, pooled


def bounds(c: Counts, conf: float) -> dict[str, Any]:
    """One-sided CP and Jeffreys bounds; None where the denominator is 0."""
    def up(x: int, n: int) -> dict[str, float] | None:
        return None if n == 0 else {"cp": clopper_pearson_upper(x, n, conf), "jeffreys": jeffreys_upper(x, n, conf)}

    def lo(x: int, n: int) -> dict[str, float] | None:
        return None if n == 0 else {"cp": clopper_pearson_lower(x, n, conf), "jeffreys": jeffreys_lower(x, n, conf)}
    return {
        "conf": conf,
        "frr_episode": {"x": c.n_false_removal, "n": c.n_present, "upper": up(c.n_false_removal, c.n_present)},
        "frr_arrangement": {"x": c.n_arr_with_false_removal, "n": c.n_arr_with_present,
                            "upper": up(c.n_arr_with_false_removal, c.n_arr_with_present)},
        "recall": {"x": c.n_true_removal, "n": c.n_removed, "lower": lo(c.n_true_removal, c.n_removed)},
        "coverage": {"x": c.n_resolved, "n": c.n_episodes, "lower": lo(c.n_resolved, c.n_episodes)},
    }


def cluster_bootstrap(outcomes: Sequence[Outcome], B: int, seed: int) -> dict[str, Any]:
    """Percentile 95% CIs for FRR, recall, coverage, resampling arrangements with replacement."""
    arrs = sorted({o.arrangement_id for o in outcomes})
    idx = {a: i for i, a in enumerate(arrs)}
    terms = {m: (np.zeros(len(arrs)), np.zeros(len(arrs))) for m in ("frr", "recall", "coverage")}
    for o in outcomes:
        i = idx[o.arrangement_id]
        rem = o.decision is Decision.REMOVED
        if o.present:
            terms["frr"][0][i] += rem
            terms["frr"][1][i] += 1
        else:
            terms["recall"][0][i] += rem
            terms["recall"][1][i] += 1
        terms["coverage"][0][i] += o.decision is not Decision.UNKNOWN
        terms["coverage"][1][i] += 1
    draws = arrangement_draws(len(arrs), B, seed)
    counts = np.stack([np.bincount(d, minlength=len(arrs)) for d in draws]).astype(float)
    out: dict[str, Any] = {"B": B, "seed": seed, "n_arrangements": len(arrs)}
    for m, (num, den) in terms.items():
        n, d = counts @ num, counts @ den
        ok = d > EPS
        vals = n[ok] / d[ok]
        ci = [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals.size else [None, None]
        out[m] = {"ci95": ci, "n_undefined": int((~ok).sum())}
    return out


# ---------------------------------------------------------------- the whole procedure

def run_selection(dev: DevData, settings: SelectionSettings, context: RunContext) -> dict[str, Any]:
    """Steps 2-4 on dev; returns everything the report and the frozen candidate need."""
    refuse_non_dev(dev.specs)
    ev = Evaluator(dev, settings, context)
    rows = view_rows(dev)
    arrs = dev.arrangements
    final = select(ev, rows, settings, arrs, check_monotone=True)
    folds, pooled = nested_cv(ev, rows, settings, arrs)
    pooled_counts = Counts.of(pooled)
    final_outcomes = ev.subset(final.thresholds, arrs)
    lam_check = check_lam_grid(settings.effective_lam_grid, dev.positions) if settings.policy == "G" else None
    for w in (lam_check.warnings if lam_check else ()):
        log.warning(w)
    return {
        "lam_grid_check": lam_check,
        "final": final, "folds": folds, "pooled": pooled, "pooled_counts": pooled_counts,
        "cv_bounds": bounds(pooled_counts, settings.conf),
        "cv_bootstrap": cluster_bootstrap(pooled, settings.bootstrap_B, settings.seed),
        "resub_bounds": bounds(final.chosen_counts, settings.conf),
        "resub_bootstrap": cluster_bootstrap(final_outcomes, settings.bootstrap_B, settings.seed),
        "n_episode_runs": ev.n_runs, "n_configs_evaluated": len(ev._cache),
    }


def candidate_doc(dev: DevData, settings: SelectionSettings, res: dict[str, Any], settings_sha: str,
                  provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """provenance: for --synthetic runs, the generator parameters, n_arr and n_dev (review F4); None on real data,
    where dev_evidence_sha256 already identifies the evidence."""
    final: Selection = res["final"]
    th = final.thresholds
    doc = {
        "schema": CANDIDATE_SCHEMA, "procedure_version": PROCEDURE_VERSION,
        "status": "CANDIDATE - not frozen until the student signs FREEZE.md",
        "synthetic": dev.synthetic, "label": SYNTHETIC_LABEL if dev.synthetic else "",
        "settings": settings.to_dict(), "settings_sha256": settings_sha, "proposed_defaults_note": PROPOSED,
        "dev_arrangements": list(dev.arrangements), "n_dev_episodes": len(dev.specs),
        "dev_evidence_sha256": dev.evidence_sha256, "view_label_source": dev.label_source,
        "positions": dev.positions.to_dict(),
        "travel_cap_cm": settings.budget_obj(dev.positions).travel_cap_cm,
        "thresholds": th.to_dict(),
        "lam_selected": settings.policy == "G",
        "lam_grid_check": res["lam_grid_check"].to_dict() if res.get("lam_grid_check") else None,
        "step2_gate": final.gate.to_dict(),
        "meets_frr_limit": final.meets_frr_limit, "recall_constraint_met": final.recall_constraint_met,
        "dev_counts_resubstitution": final.chosen_counts.to_dict(),
        "nested_cv_counts": res["pooled_counts"].to_dict(),
        "nested_cv_bounds": res["cv_bounds"], "nested_cv_bootstrap": res["cv_bootstrap"],
        "warnings": final.warnings,
    }
    if provenance is not None:
        doc["synthetic_provenance"] = dict(provenance)
    return to_jsonable(doc)


# ---------------------------------------------------------------- report

def _f(x: float | None, nd: int = 3) -> str:
    return "undef" if x is None else f"{x:.{nd}f}"


def _b(d: dict | None, key: str) -> str:
    return "undef" if d is None else f"{d[key]:.4f}"


def render_report(dev: DevData, settings: SelectionSettings, res: dict[str, Any], cand: dict[str, Any],
                  cand_sha: str) -> str:
    final: Selection = res["final"]
    g = final.gate
    syn = dev.synthetic
    L = []
    if syn:
        L += [f"# {SYNTHETIC_LABEL}", "",
              "**SYNTHETIC - not study results and not threshold values.** The evidence and the per-view "
              "diagnostic labels come from `verify.synthetic` (labels are a stand-in derived from the "
              "generator's hidden occluder layout). This run only exercises the selection procedure; "
              "nothing here may be carried into FREEZE.md.", ""]
    L += ["# Threshold selection report (D-select)", "",
          f"- procedure `{PROCEDURE_VERSION}`; settings sha256 `{cand['settings_sha256']}`",
          f"- candidate `frozen_candidate.json` sha256 `{cand_sha}` (status: candidate, not frozen)",
          f"- dev: {len(dev.arrangements)} arrangements, {len(dev.specs)} episodes; evidence sha256 "
          f"`{dev.evidence_sha256}`",
          f"- view labels: {dev.label_source}",
          f"- {res['n_configs_evaluated']} configs evaluated, {res['n_episode_runs']} episode replays "
          "(each config on a fresh EvidenceStore through ReplayEnv/run_episode)", "",
          "## Settings (every value is a *proposed default - confirm*)", "",
          "| setting | value |", "| --- | --- |"]
    sd = settings.to_dict()
    for k in ("policy", "budget", "cap_steps", "tau_match_grid", "k_grid", "lam_grid", "tau_q_grid", "tau_occ_grid",
              "alpha_clear", "frr_limit", "recall_target", "conf", "bootstrap_B", "seed"):
        L.append(f"| {k} | {sd[k]} |")
    if settings.policy != "G":
        L.append(f"| lam | not selected: {settings.policy} ignores it; the placeholder {Thresholds().lam:g} is used |")
    prov = cand.get("synthetic_provenance")
    if prov:
        L.append(f"| synthetic generator | n_arr {prov['n_arr']}, n_dev {prov['n_dev']}, block_includes_start "
                 f"{prov['generator']['block_includes_start']} |")
    lc: LamGridCheck | None = res.get("lam_grid_check")
    if lc is not None:
        L += ["", "## lam grid check (geometry only; verify/lamgrid.py)", "",
              "G's choice changes with lam only at these breakpoints on this `positions.json`: "
              + (", ".join(f"{b.lam:.4f}" for b in lc.breakpoints) or "none") + ". "
              "Grid values between the same two breakpoints give identical G choices in every decision state.", "",
              "- behaviour classes of the grid: " + "; ".join("{" + ", ".join(f"{v:g}" for v in c) + "}"
                                                             for c in lc.classes)]
        L += [f"- WARNING: {w}" for w in lc.warnings] or ["- every grid value is a distinct G behaviour, none on a breakpoint"]
    L += ["", "## Step 2: visibility gate (tau_q, tau_occ) from diagnostic labels", "",
          "Null set: non-diagnostic views of occluded-present episodes (a pass here is a false clear). "
          "Power set: diagnostic views of removed episodes. Gate = quality >= tau_q and obstruction <= tau_occ.", "",
          f"- chosen tau_q = {g.tau_q}, tau_occ = {g.tau_occ} ({g.n_feasible} of {g.n_grid} grid gates met "
          f"alpha_clear = {settings.alpha_clear}; met: {g.met_alpha})",
          f"- false-clear rate {_f(g.false_clear_rate)} ({g.false_clear[0]}/{g.false_clear[1]} views"
          + (f", one-sided {settings.conf:.0%} CP upper {clopper_pearson_upper(*g.false_clear, settings.conf):.4f}"
             if g.false_clear[1] else "") + "; views of one arrangement are correlated, so this bound is optimistic)",
          f"- clear rate on diagnostic removed views {_f(g.clear_rate)} ({g.clear[0]}/{g.clear[1]})", "",
          "## Step 3: fixed-sequence walk (conservative -> liberal)", "",
          f"A step passes if its worst FRR over the lam grid is <= {settings.frr_limit}; the accepted prefix "
          "ends at the first failing step.", "",
          "| step | k | tau_match | FRR per lam | worst FRR | pass | accepted |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for w in final.walk:
        per = ", ".join(f"{gc.lam:g}: {c.n_false_removal}/{c.n_present}" for gc, c in w.rows)
        L.append(f"| {w.step} | {w.k} | {w.tau_match} | {per} | {_f(w.worst_frr)} | {'yes' if w.passed else 'no'} | "
                 f"{'yes' if w.accepted else 'no'} |")
    L += ["", "### Accepted configurations (D10 objective input)", "",
          "| order | k | tau_match | lam | FRR | recall | coverage | mean travel cm |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for w in final.walk:
        if w.accepted:
            for gc, c in w.rows:
                mark = " **chosen**" if gc == final.chosen else ""
                L.append(f"| {gc.order}{mark} | {gc.k} | {gc.tau_match} | {gc.lam:g} | {_f(c.frr)} | {_f(c.recall)} | "
                         f"{_f(c.coverage)} | {_f(c.mean_travel_cm, 1)} |")
    if not any(w.accepted for w in final.walk):
        L.append("| - | none | | | | | | |")
    c = final.chosen_counts
    L += ["", f"- **candidate**: k = {final.chosen.k}, tau_match = {final.chosen.tau_match}, "
              f"lam = {final.chosen.lam:g}, tau_q = {g.tau_q}, tau_occ = {g.tau_occ}",
          f"- meets FRR limit: {final.meets_frr_limit}; recall >= {settings.recall_target} available: "
          f"{final.recall_constraint_met}",
          f"- dev (resubstitution, optimistic): FRR {_f(c.frr)} ({c.n_false_removal}/{c.n_present}), recall "
          f"{_f(c.recall)} ({c.n_true_removal}/{c.n_removed}), coverage {_f(c.coverage)} "
          f"({c.n_resolved}/{c.n_episodes}), mean travel {_f(c.mean_travel_cm, 1)} cm", "",
          "### Monotonicity checks (empirical)", ""]
    mono = [w for w in final.warnings if not w.startswith("Step")]
    L += [f"- {w}" for w in mono] or ["- no violations found on this data"]
    other = [w for w in final.warnings if w.startswith("Step")]
    if other:
        L += ["", "### Other warnings", "", *[f"- {w}" for w in other]]
    L += ["", "## Step 4: nested grouped CV (outer leave-one-arrangement-out)", "",
          "| held out | tau_q | tau_occ | k | tau_match | lam | inner meets FRR | held FRR | held recall | held coverage |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in res["folds"]:
        t, h = f.selection_thresholds, f.held_counts
        L.append(f"| {f.held_out} | {t.tau_q} | {t.tau_occ} | {t.k} | {t.tau_match} | {t.lam:g} | {f.meets_frr_limit} | "
                 f"{h.n_false_removal}/{h.n_present} | {h.n_true_removal}/{h.n_removed} | {h.n_resolved}/{h.n_episodes} |")
    distinct = len({f.selection_thresholds for f in res["folds"]})
    L += ["", f"Selections differ across folds: {distinct} distinct configs over {len(res['folds'])} folds.", ""]

    def bound_block(title: str, cnt: Counts, bd: dict, boot: dict) -> list[str]:
        fe, fa, rc, cv = bd["frr_episode"], bd["frr_arrangement"], bd["recall"], bd["coverage"]
        pct = f"{bd['conf']:.0%}"
        return [f"### {title}", "",
                "| quantity | estimate | x/n | one-sided CP | one-sided Jeffreys | cluster bootstrap 95% CI |",
                "| --- | --- | --- | --- | --- | --- |",
                f"| FRR (episode) upper {pct} | {_f(cnt.frr)} | {fe['x']}/{fe['n']} | {_b(fe['upper'], 'cp')} | "
                f"{_b(fe['upper'], 'jeffreys')} | [{_f(boot['frr']['ci95'][0])}, {_f(boot['frr']['ci95'][1])}] |",
                f"| FRR (arrangement: any false removal) upper {pct} | "
                f"{_f(fa['x'] / fa['n'] if fa['n'] else None)} | {fa['x']}/{fa['n']} | {_b(fa['upper'], 'cp')} | "
                f"{_b(fa['upper'], 'jeffreys')} | - |",
                f"| recall lower {pct} | {_f(cnt.recall)} | {rc['x']}/{rc['n']} | {_b(rc['lower'], 'cp')} | "
                f"{_b(rc['lower'], 'jeffreys')} | [{_f(boot['recall']['ci95'][0])}, {_f(boot['recall']['ci95'][1])}] |",
                f"| coverage lower {pct} | {_f(cnt.coverage)} | {cv['x']}/{cv['n']} | {_b(cv['lower'], 'cp')} | "
                f"{_b(cv['lower'], 'jeffreys')} | [{_f(boot['coverage']['ci95'][0])}, "
                f"{_f(boot['coverage']['ci95'][1])}] |", "",
                f"Bootstrap: B = {boot['B']}, seed {boot['seed']}, {boot['n_arrangements']} arrangements; undefined "
                f"draws FRR {boot['frr']['n_undefined']}, recall {boot['recall']['n_undefined']}.", ""]
    L += bound_block("Honest dev estimate: pooled held-out decisions", res["pooled_counts"], res["cv_bounds"],
                     res["cv_bootstrap"])
    L += bound_block("Candidate on all dev arrangements (resubstitution; optimistic, for reference)",
                     final.chosen_counts, res["resub_bounds"], res["resub_bootstrap"])
    L += ["## Reading these numbers", "",
          "- Episodes of one arrangement share an occluder and lighting, so they are not independent: the "
          "arrangement-level bound is the more honest of the two FRR bounds.",
          "- With about 10 dev arrangements no formal (alpha, delta) guarantee on FRR is achievable; the dev "
          "targets are goals and the selection rule is a selector (lit report sec. 3).",
          "- The resubstitution numbers reuse the data that chose the config; quote the nested-CV numbers.", ""]
    if syn:
        L += [f"_{SYNTHETIC_LABEL}._", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- CLI

def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def _cap_steps(text: str) -> float | None:
    t = text.strip().lower()
    if t in ("inf", "none", "uncapped"):
        return None
    v = float(t)
    if v < 0:
        raise argparse.ArgumentTypeError("cap steps must be >= 0")
    return v


def build_parser() -> argparse.ArgumentParser:
    d = SelectionSettings()
    p = argparse.ArgumentParser(prog="verify.select", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="select on the synthetic generator's DEV split")
    p.add_argument("--n-arr", type=int, default=30)
    p.add_argument("--n-dev", type=int, default=None)
    p.add_argument("--seed", type=int, default=0, help="generator / split / bootstrap seed")
    p.add_argument("--block-anywhere", action="store_true")
    p.add_argument("--policy", choices=SELECTION_POLICIES, default=d.policy)
    p.add_argument("--budget", type=int, default=d.budget)
    p.add_argument("--cap-steps", type=_cap_steps, default=None, help="travel cap in arc steps, or inf (default)")
    p.add_argument("--tau-match-grid", type=_floats, default=d.tau_match_grid)
    p.add_argument("--k-grid", type=_ints, default=d.k_grid, help="walk order, conservative first (default 3,2)")
    p.add_argument("--lam-grid", type=_floats, default=d.lam_grid)
    p.add_argument("--tau-q-grid", type=_floats, default=d.tau_q_grid)
    p.add_argument("--tau-occ-grid", type=_floats, default=d.tau_occ_grid)
    p.add_argument("--alpha-clear", type=float, default=d.alpha_clear)
    p.add_argument("--frr-limit", type=float, default=d.frr_limit)
    p.add_argument("--recall-target", type=float, default=d.recall_target)
    p.add_argument("--bootstrap-B", type=int, default=d.bootstrap_B)
    p.add_argument("--positions", default=None, help="positions JSON (default configs/positions.json)")
    p.add_argument("--lam-breakpoints", action="store_true",
                   help="print G's lam breakpoints on the positions file, check --lam-grid against them, and exit")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None, help="output dir (default runs/select-<tag>/)")
    return p


def write_outputs(out: Path, dev: DevData, settings: SelectionSettings, res: dict[str, Any],
                  provenance: Mapping[str, Any] | None = None) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    hashed: dict[str, Any] = {"settings": settings.to_dict(), "procedure": PROCEDURE_VERSION}
    if provenance is not None:   # synthetic runs: the generator regime is part of what was selected on (F4)
        hashed["synthetic_provenance"] = to_jsonable(dict(provenance))
    sha_settings = config_sha256(hashed)
    cand = candidate_doc(dev, settings, res, sha_settings, provenance)
    text = json.dumps(cand, indent=1, sort_keys=True) + "\n"
    (out / "frozen_candidate.json").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    (out / "frozen_candidate.json.sha256").write_text(f"{sha}  frozen_candidate.json\n")
    (out / "selection_report.md").write_text(render_report(dev, settings, res, cand, sha))
    details = {"label": SYNTHETIC_LABEL if dev.synthetic else "", "settings": settings.to_dict(),
               "walk": [{"step": w.step, "k": w.k, "tau_match": w.tau_match, "worst_frr": w.worst_frr,
                         "passed": w.passed, "accepted": w.accepted,
                         "configs": [{"order": g.order, "lam": g.lam, **c.to_dict()} for g, c in w.rows]}
                        for w in res["final"].walk],
               "folds": [{"held_out": f.held_out, "thresholds": f.selection_thresholds.to_dict(),
                          "meets_frr_limit": f.meets_frr_limit, "train": f.train_counts.to_dict(),
                          "held": f.held_counts.to_dict()} for f in res["folds"]]}
    (out / "selection_details.json").write_text(json.dumps(to_jsonable(details), indent=1, sort_keys=True) + "\n")
    return {"frozen_candidate.json": sha}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.lam_breakpoints:
        pos = Positions.load(args.positions) if args.positions else Positions.load()
        chk = check_lam_grid(args.lam_grid, pos)
        print("G lam breakpoints (geometry only): "
              + ", ".join(f"{b.lam:.6f} ({b.n_states_changed} states)" for b in chk.breakpoints))
        print("grid behaviour classes: " + "; ".join(str(list(c)) for c in chk.classes))
        for w in chk.warnings:
            print("WARNING: " + w)
        return 0
    if not args.synthetic:
        print("real dev data not available yet (use --synthetic); build a DevData and call run_selection",
              file=sys.stderr)
        return 2
    positions = Positions.load(args.positions) if args.positions else Positions.load()
    settings = SelectionSettings(policy=args.policy, budget=args.budget, cap_steps=args.cap_steps,
                                 tau_match_grid=args.tau_match_grid, k_grid=args.k_grid, lam_grid=args.lam_grid,
                                 tau_q_grid=args.tau_q_grid, tau_occ_grid=args.tau_occ_grid,
                                 alpha_clear=args.alpha_clear, frr_limit=args.frr_limit,
                                 recall_target=args.recall_target, bootstrap_B=args.bootstrap_B, seed=args.seed)
    params = SyntheticParams(n_dev=args.n_dev, block_includes_start=not args.block_anywhere)
    dev = synthetic_dev_data(args.n_arr, args.seed, params, positions)
    context = RunContext(model_id="synthetic-no-model", config_version=PROCEDURE_VERSION,
                         config_sha256=config_sha256(settings.to_dict()), synthetic=True)
    res = run_selection(dev, settings, context)
    tag = args.tag or f"synthetic-n{args.n_arr}-s{args.seed}"
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"select-{tag}"
    n_dev_used = len(dev.arrangements)
    provenance = {"generator": dataclasses.asdict(params), "n_arr": args.n_arr, "n_dev": n_dev_used,
                  "seed": args.seed}
    shas = write_outputs(out, dev, settings, res, provenance)
    print(f"{SYNTHETIC_LABEL}: wrote {out} (frozen_candidate.json sha256 {shas['frozen_candidate.json']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
