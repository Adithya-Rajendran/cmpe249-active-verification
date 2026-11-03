"""Dev-only threshold calibration (D-select), procedure select-v4 (approved by the student 2026-11-02).

    uv run --no-sync python -m verify.select --synthetic [--n-arr 30] [--seed 0] [--block-anywhere]
        [--budget 3] [--cap-steps 1,2,3,4,6 | none] [--r-seeds 20] [--tau-match-grid 0.40,...]
        [--tau-q-grid ...] [--tau-occ-grid ...] [--alpha-clear 0.10] [--frr-limit 0.10]
        [--k-sensitivity 3 | none] [--lam-sensitivity 0.7,0.9,1.2 | none] [--bootstrap-B 2000]
        [--tag TAG] [--out runs/select-v4-<tag>/]
    uv run --no-sync python -m verify.select --lam-breakpoints [--positions configs/positions.json]
        [--lam-sensitivity ...]
        (prints the lam values at which primary G's choice changes on that geometry, and checks lam = 0
        and the sensitivity values against them)

No compared policy's results set any shared setting (DECISIONS 2026-10-12: "Shared thresholds are
selected by a documented procedure that does not optimise them on G's results"). The thresholds come
from per-view labels alone, k and lam are fixed in advance, and every policy (S, F, R, G, G-lookahead,
G-myopic; A later) then runs with the same frozen settings.

Procedure select-v4. Every number it produces on --synthetic data is SYNTHETIC and is never a real
threshold.

  Step 1  Per-view calibration on the dev views of BOTH lightings together: one frozen threshold set
          serves L1 and L2 queries, and its per-view rates are reported separately for each. No policy
          is run. Every per-view criterion is the shared StoppingRule's own predicate (passes_gate,
          is_clear_miss, is_strong_match), never a copy of it.
    1a    Gate (tau_q, tau_occ), as in select-v3: over the fixed grid, keep the gates whose false-clear
          rate (gate passes on non-diagnostic views of occluded-present episodes) is <= alpha_clear,
          then maximise the clear rate (gate passes on diagnostic views of removed episodes). Ties:
          lower false-clear rate, higher tau_q, lower tau_occ.
    1b    tau_match, with that gate: keep the grid values whose false clear-miss rate (rule clear
          misses on diagnostic views of present episodes, where the target is there and visible) is
          <= alpha_clear, the same predeclared alpha, then maximise the clear-miss rate on diagnostic
          views of removed episodes. Ties: lower false clear-miss rate, then the lower, more
          conservative tau_match.
    Fail loudly. If no gate on the grid meets alpha_clear in 1a, or no tau_match on the grid meets it in
          1b under the gate 1a chose (or there is no view to measure the false rate on), calibration raises
          CalibrationInfeasible. There is then no frozen candidate: the CLI removes any earlier
          frozen_candidate.json from its output directory, writes calibration_infeasible.json (status
          "infeasible": the constraint that failed, the best achievable rate and the grid searched) and
          exits with status 3. alpha_clear is never relaxed, the grid is never widened, and no "least bad"
          setting is used (select-v3 fell back to the lowest false rate). A 1b failure does not show that
          no gate and tau_match on the grid meet both constraints (see Known gaps).
  Step 2  Fixed a priori, never selected: k = 2 (the plan's "at least two distinct positions") and
          lam = 0 (G and A choose by spread or informativeness alone; travel is limited by the caps,
          which are reported as curves).
  Step 3  Honest per-view estimate: grouped cross-validation of Step 1, leaving one arrangement out.
          A held-out arrangement's per-view rates come from thresholds calibrated without it. A fold
          whose calibration is infeasible gets no thresholds and no held-out rates (no fallback); it is
          reported and the held-out estimate is marked incomplete. A candidate is written whenever Step 1
          is feasible on all dev arrangements, but one whose grouped CV is incomplete must not be frozen
          (docs/SIM_PREREGISTRATION.md section 7; not yet marked in the candidate, see Known gaps).
  Step 4  FRR <= frr_limit is a CHECK, not an objective. With the frozen thresholds (k = 2, lam = 0)
          every policy (S, F, R, G, G-lookahead, G-myopic) runs on dev at the primary budget, uncapped
          and under every cap of the grid. FRR, recall and coverage are reported separately for L1
          and L2 queries, with one-sided Clopper-Pearson and Jeffreys bounds and a cluster bootstrap
          over arrangements. A policy above the limit is flagged; nothing is changed to fix it. A row
          with no present-target episode (FRR undefined) is not checked, and the check is then
          incomplete (all_within_limit null), never passed.
  Step 5  Sensitivity, reported and never selected: k = 3 (F, R, G, G-lookahead, G-myopic) and lam in
          {0.7, 0.9, 1.2} (G and G-lookahead, the only policies that read lam), with the frozen taus.

alpha_clear (0.10 by default) is a predeclared, EMPIRICAL per-view calibration constraint on the dev
views: one value for both Step 1 false rates, measured on the dev views of both lightings together. It
is not a bound on any policy's false removal rate, which Step 4 only checks, and requiring k = 2
clear-miss stops does not turn it into alpha squared.

Order. Steps 1-3 read views only. The first policy run happens after the thresholds are frozen, and
Steps 4-5 hand the policies nothing but the frozen Thresholds (or a copy with k or lam replaced), so
neither the check nor the sensitivity runs can reach the candidate.

Replaced: select-v3 (until 2026-11-02). Its Step 3 walked (k, tau_match) in a fixed conservative ->
liberal sequence under the FRR limit and then picked (k, tau_match, lam) by RUNNING primary G on dev
and maximising its coverage (the D10 objective, with a recall >= 0.80 filter and a travel tie-break);
its Step 4 re-ran that selection inside nested CV. Picking shared settings on G's own dev results
contradicted DECISIONS 2026-10-12, and it would have handed G-tuned settings to ranker A, tilting the
primary A-vs-G comparison. The study now runs in simulation (docs/SIMULATION_STUDY.md), where dev
arrangements come with exact per-view labels in any number, so per-view calibration is cheap and
reliable.

Bounds. One-sided Clopper-Pearson and Jeffreys bounds at the episode level and at the arrangement
level ("any false removal in the arrangement"), plus a percentile cluster bootstrap over arrangements.
Views and episodes of one arrangement are correlated, so view- and episode-level bounds are
optimistic and the arrangement-level bound is the honest one. How low that bound can go depends on
the number of dev arrangements, which the report states with the best bound it allows. The check
runs on the arrangements whose views set the thresholds, so its bounds are mildly optimistic too; the
FRR limit is a goal, never a guarantee. R's seed-averaged decisions are not binomial counts, so R gets
the bootstrap only.

Data access. This module is offline dev analysis, not a policy. It reads every dev view for Step 1
(from a fresh unsealed store) and joins dev ground truth to score Steps 4-5. Policies still see only
requested views: every run goes through the public ReplayEnv / run_episode API on a fresh
EvidenceStore from a StoreFactory (one store per run setting; keys are minted before sealing). Only
DEV-split episodes are accepted; any test (or pilot) episode raises SplitRefused.

Known gaps, next checkpoint (docs/SIM_PREREGISTRATION.md section 7; verify/README.md section 10):
  - Held-out policy check. Each fold's thresholds are not yet used to run the policies on the held-out
    arrangement. Only the per-view CV rates of Step 3 exist, and the Step 4 policy rows are resubstitution.
  - Freeze eligibility. A candidate whose grouped CV is incomplete is not yet marked
    eligible_for_freeze: false. Until it is, a candidate with grouped_cv.complete = false must not be frozen.
  - Sim dev input. This module cannot read the sim set files yet (their view_labels.json is
    cmpe249.sim-view-labels-set.v0), and the CLI is --synthetic only.
  - Gate and tau_match are chosen in sequence. Step 1b searches tau_match only under the gate Step 1a
    picked, so it can report "infeasible" while other grid settings meet both alpha constraints. The next
    checkpoint replaces this with a joint search: maximise correct clear misses subject to both
    constraints, with a declared conservative tie-break.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import logging
import math
import os
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
from verify.policies import (
    DEFAULT_R_SEEDS,
    GMYOPIC_LAM_DEG,
    F,
    G,
    GMyopic,
    S,
    r_seed_salt,
    run_episode,
    run_r_seeds,
)
from verify.replay import ReplayEnv
from verify.schema import (
    CAP_STEPS_DEFAULT,
    EPS,
    REPO_ROOT,
    SYNTHETIC_LABEL,
    Budget,
    Condition,
    Decision,
    EpisodeLog,
    EpisodeResult,
    EpisodeSpec,
    EvidenceRecord,
    EvidenceStore,
    GroundTruth,
    Lighting,
    ManifestError,
    Observation,
    Positions,
    RunContext,
    Split,
    Thresholds,
    TruthState,
    cap_cm_from_steps,
    config_sha256,
    to_jsonable,
    truth_from_condition,
)
from verify.stopping import StoppingRule
from verify.synthetic import SyntheticParams, generate, generate_layouts

log = logging.getLogger("verify.select")

PROCEDURE_VERSION = "select-v4"   # v4 (2026-11-02): per-view calibration only; k = 2 and lam = 0 fixed; FRR a check
# v3: G is primary G (lookahead off); v2: lam grid from breakpoints, lam_grid_check, synthetic provenance (F1, F4)
CANDIDATE_SCHEMA = "cmpe249.frozen_candidate.v2"   # v2 (select-v4): calibration, dev check and sensitivity sections
VIEW_LABELS_SCHEMA = "cmpe249.view_labels.v1"
PROPOSED = "proposed default - confirm"
SYNTHETIC_LABEL_SOURCE = ("SYNTHETIC stand-in: derived from the generator's hidden occluder layout "
                          "(blocked or edge stop -> non-diagnostic), not from human annotation")

# Fixed a priori (Step 2). They are module constants, not settings: nothing in this module can select them.
K_FIXED = 2       # the plan's "at least two distinct positions" (EXPERIMENT_PLAN 'Stopping')
LAM_FIXED = 0.0   # G and A choose by spread / informativeness alone; travel is limited by the caps (curves)

# Proposed defaults (all "proposed default - confirm"; the score grids are placeholders until the
# catalog sweep shows real score distributions).
DEFAULT_TAU_MATCH_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
DEFAULT_TAU_Q_GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(11))    # 0.30 .. 0.80
DEFAULT_TAU_OCC_GRID = tuple(round(0.10 + 0.05 * i, 2) for i in range(9))   # 0.10 .. 0.50
DEFAULT_CAP_STEPS = tuple(float(c) for c in CAP_STEPS_DEFAULT if math.isfinite(c))  # 1..6; uncapped always runs
DEFAULT_K_SENSITIVITY = (3,)
# lam sensitivity (review F1): G's choice changes only at a few lam breakpoints (verify/lamgrid.py; at the design
# geometry 2/3, 3/4, 1, 4/3, 3/2, 2). One value inside each of the intervals (2/3, 3/4), (3/4, 1), (1, 4/3), none on a
# breakpoint; the fixed lam = 0 stands for [0, 2/3). Re-derive with --lam-breakpoints for a new geometry.
DEFAULT_LAM_SENSITIVITY = (0.7, 0.9, 1.2)
DEFAULT_LAM_GRID = (LAM_FIXED, *DEFAULT_LAM_SENSITIVITY)   # every lam that runs: the fixed value, then the sensitivity

CHECK_POLICIES = ("S", "F", "R", "G", "G-lookahead", "G-myopic")
K_SENSITIVITY_POLICIES = ("F", "R", "G", "G-lookahead", "G-myopic")   # S decides with k = 1 by construction
LAM_SENSITIVITY_POLICIES = ("G", "G-lookahead")   # the only policies that read Thresholds.lam (G-myopic: lam_deg)
REPORT_LIGHTINGS = tuple(x.value for x in Lighting)   # "L1", "L2": the headline split (DECISIONS 2026-11-02)
POOLED = "pooled"   # both lightings: the set Step 1 calibrates on; for policy results a labelled secondary line
NOT_BINOMIAL = ("seed-averaged decisions (R) are not binomial counts: no Clopper-Pearson / Jeffreys bound; "
                "see the cluster bootstrap")
ALPHA_NOTE = ("alpha_clear is a predeclared, empirical per-view calibration constraint on the dev views: the chosen "
              "thresholds must keep the false-clear rate (Step 1a) and the false clear-miss rate (Step 1b), measured "
              "on the dev views of both lightings together, at or below it. It is not a bound on any policy's false "
              "removal rate (FRR is only checked, Step 4), and requiring k = 2 clear-miss stops does not turn it into "
              "alpha squared.")
GATE_CONSTRAINT = ("false-clear rate (StoppingRule.passes_gate on non-diagnostic views of occluded-present dev "
                   "episodes) <= alpha_clear")
MATCH_CONSTRAINT = ("false clear-miss rate (StoppingRule.is_clear_miss, with the Step 1a gate, on diagnostic views of "
                    "present dev episodes) <= alpha_clear")
NO_FALLBACK = "alpha_clear was not relaxed, the grid was not widened and no fallback setting was used"
INFEASIBLE_FILE = "calibration_infeasible.json"
INFEASIBLE_SCHEMA = "cmpe249.calibration_infeasible.v1"
EXIT_INFEASIBLE = 3   # CLI exit status when Step 1 is infeasible (2 = usage error)
CANDIDATE_FILES = ("frozen_candidate.json", "frozen_candidate.json.sha256", "selection_details.json")
REPORT_FILE = "selection_report.md"
OUTPUT_FILES = (*CANDIDATE_FILES, REPORT_FILE, INFEASIBLE_FILE)   # everything a run writes into its output dir


class SplitRefused(ManifestError):
    """Selection was handed a non-dev (test or pilot) episode."""


@dataclass(frozen=True)
class Infeasibility:
    """Why Step 1 failed: no gate on the grid meets alpha_clear (1a), no tau_match on the grid meets it under the gate
    1a chose (1b), or there is no view to measure the false rate on. Nothing was relaxed, widened or substituted.

    A 1b failure covers that one gate only: other gates on the grid were never tried with the tau_match grid, so it
    does not show that no grid setting meets both constraints (a known gap; the next checkpoint searches jointly)."""
    step: str                                      # "1a" (gate) or "1b" (tau_match)
    constraint: str                                # the constraint no grid setting met
    alpha_clear: float
    n_null_views: int                              # the views the false rate is measured on
    best_rate: float | None                        # lowest false rate on the grid (None: no view to measure it on)
    best_counts: tuple[int, int] | None
    best_settings: tuple[Mapping[str, float], ...] # every grid setting that reaches best_rate
    grid: Mapping[str, Any]                        # the grid searched (Step 1b: with the Step 1a gate it used)
    grid_rates: tuple[Mapping[str, Any], ...]      # every grid setting with its false rate (x, n)
    arrangements: tuple[str, ...]

    def summary(self) -> str:
        where = " (tau_match under the Step 1a gate)" if self.step == "1b" else ""
        if self.best_rate is None or self.best_counts is None:
            return f"Step {self.step}{where}: no view to measure the false rate on"
        x, n = self.best_counts
        return (f"Step {self.step}{where}: best achievable false rate {self.best_rate:.4f} ({x}/{n}) > alpha_clear "
                f"{self.alpha_clear:g}")

    def searched(self) -> tuple[str, str]:
        """(what failed, under what) for the messages: 1a searched the whole gate grid, 1b only the tau_match grid
        under the one gate Step 1a chose."""
        if self.step == "1b":
            return "no tau_match on the grid", f" under the gate Step 1a chose {_grid_text(self.grid.get('gate', {}))}"
        return "no gate (tau_q, tau_occ) on the grid", ""

    def message(self) -> str:
        if self.best_rate is None:
            best = "there is no view to measure the false rate on (0 views)"
        else:
            at = [", ".join(f"{k} {v:g}" for k, v in st.items()) for st in self.best_settings]
            more = f" and {len(at) - 5} more" if len(at) > 5 else ""
            best = (f"the best achievable rate is {self.best_rate:.4f} ({self.best_counts[0]}/{self.best_counts[1]} "
                    f"views), at {'; '.join(at[:5])}{more}")
        grid = "; ".join(f"{k} {_grid_text(v)}" for k, v in self.grid.items())
        # Review: "no setting on the grid" overstated a 1b failure, which tried the tau_match grid under one gate only.
        sequential = (" Step 1b searches tau_match only under the gate Step 1a picked, so other grid settings may "
                      "still meet both alpha constraints (a known gap: the next checkpoint searches gate and tau_match "
                      "jointly)." if self.step == "1b" else "")
        what, under = self.searched()
        return (f"Step {self.step} is infeasible: {what} meets '{self.constraint}' with alpha_clear = "
                f"{self.alpha_clear:g}{under}; {best}. Grid searched: {grid}.{sequential} Dev views of "
                f"{len(self.arrangements)} arrangements, both lightings. {NO_FALLBACK}.")

    def to_dict(self) -> dict[str, Any]:
        return {**{f.name: getattr(self, f.name) for f in dataclasses.fields(self)}, "message": self.message()}


def _grid_text(v: Any) -> str:
    if isinstance(v, Mapping):
        return "(" + ", ".join(f"{k} {x:g}" for k, x in v.items()) + ")"
    return "[" + ", ".join(f"{x:g}" for x in v) + "]"


class CalibrationInfeasible(RuntimeError):
    """Step 1 failed loudly: no gate on the grid meets alpha_clear (1a), or no tau_match does under the gate 1a chose
    (1b). There is no candidate; alpha_clear is never relaxed, the grid never widened and no "least bad" setting is
    used. failure says what failed."""

    def __init__(self, failure: Infeasibility) -> None:
        super().__init__(failure.message())
        self.failure = failure


def infeasibility(step: str, constraint: str, alpha: float, n_null: int,
                  points: Sequence[tuple[Mapping[str, float], tuple[int, int]]], grid: Mapping[str, Any],
                  rows: Iterable[ViewRow]) -> Infeasibility:
    """The Infeasibility record of a step whose grid points (setting, false-rate counts) all miss alpha."""
    rated = [(st, xn, _ratio(xn)) for st, xn in points]
    defined = [r for _, _, r in rated if r is not None]
    best = min(defined) if defined else None
    best_pts = [(st, xn) for st, xn, r in rated if best is not None and r == best]
    return Infeasibility(step, constraint, alpha, n_null, best, best_pts[0][1] if best_pts else None,
                         tuple(dict(st) for st, _ in best_pts), dict(grid),
                         tuple({**st, "x": xn[0], "n": xn[1], "rate": r} for st, xn, r in rated),
                         tuple(sorted({r.arrangement_id for r in rows})))


# ---------------------------------------------------------------- data interfaces

class StoreFactory(Protocol):
    """Returns a FRESH, UNSEALED EvidenceStore holding exactly the dev episodes, on every call.

    A ReplayEnv seals its store on bind, so selection asks for a new store per run setting (and one
    for the Step 1 view table). Every call must return identical evidence.
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


def _label_rows(labels: Iterable[ViewLabel]) -> list[dict[str, Any]]:
    """The label rows in their stable order, (episode_id, stop): the order files and the labels hash use."""
    return [dataclasses.asdict(l) for l in sorted(labels, key=lambda l: (l.episode_id, l.stop))]


def view_labels_sha256(labels: Iterable[ViewLabel]) -> str:
    """Provenance hash of the per-view labels: sha256 of the canonical JSON of the label rows in (episode_id, stop)
    order, i.e. of the "labels" list save_view_labels writes. The labels set the thresholds as much as the evidence
    does, so a candidate records both hashes (review: only the evidence was hashed, next to a free-text source)."""
    return config_sha256(_label_rows(labels))


def save_view_labels(labels: Iterable[ViewLabel], path: str | Path, *, source: str) -> None:
    doc = {"schema": VIEW_LABELS_SCHEMA, "source": source, "labels": _label_rows(labels)}
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")


def load_view_labels(path: str | Path) -> list[ViewLabel]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != VIEW_LABELS_SCHEMA:
        raise ValueError(f"view labels schema must be {VIEW_LABELS_SCHEMA!r}, got {doc.get('schema')!r}")
    return [ViewLabel(d["episode_id"], int(d["stop"]), bool(d["diagnostic"])) for d in doc["labels"]]


@dataclass(frozen=True)
class DevData:
    """Everything selection may use. Construction refuses any non-DEV episode, a synthetic flag that disagrees with
    the evidence store's or the truth's, and real (non-synthetic) data without an evidence_sha256."""
    specs: tuple[EpisodeSpec, ...]
    store_factory: StoreFactory
    truth: GroundTruth
    view_labels: tuple[ViewLabel, ...]
    positions: Positions
    synthetic: bool
    label_source: str
    evidence_sha256: str = ""   # required on real data; synthetic_dev_data fills it in too

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
        # The candidate's SYNTHETIC label comes from this flag alone, so it must agree with the data it describes:
        # synthetic evidence can never yield an unlabelled candidate (review; verify/run.py's real path has the same
        # guard). A real store or truth called synthetic is refused too: a disagreement means a wrong input.
        disagree = [f"the {what} says synthetic={flag}" for what, flag in
                    (("evidence store", store.synthetic), ("ground truth", self.truth.synthetic))
                    if bool(flag) != bool(self.synthetic)]
        if disagree:
            raise ValueError(f"DevData(synthetic={self.synthetic}) disagrees with its data: {'; '.join(disagree)}")
        if not self.synthetic and not (self.evidence_sha256 or "").strip():
            raise ValueError("real (non-synthetic) dev data need evidence_sha256, the provenance hash of the dev "
                             "evidence (e.g. RecordStoreFactory.evidence_sha256()); the candidate records it")

    @property
    def arrangements(self) -> tuple[str, ...]:
        return tuple(sorted({s.arrangement_id for s in self.specs}))

    @property
    def view_labels_sha256(self) -> str:
        """sha256 of the per-view labels (view_labels_sha256), recorded next to evidence_sha256."""
        return view_labels_sha256(self.view_labels)


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
    """Predeclared settings of select-v4; every value is a *proposed default - confirm*.

    k and lam are deliberately absent: they are fixed a priori (K_FIXED, LAM_FIXED), and the only other
    values of them that ever run are the reported sensitivity values below.
    """
    budget: int = 3                                    # primary budget of the dev check (additional views)
    cap_steps: tuple[float, ...] = DEFAULT_CAP_STEPS   # dev-check travel caps in arc steps; uncapped always runs too
    r_seeds: int = len(DEFAULT_R_SEEDS)                # R seeds per episode in the dev check
    tau_match_grid: tuple[float, ...] = DEFAULT_TAU_MATCH_GRID
    tau_q_grid: tuple[float, ...] = DEFAULT_TAU_Q_GRID
    tau_occ_grid: tuple[float, ...] = DEFAULT_TAU_OCC_GRID
    alpha_clear: float = 0.10     # empirical per-view constraint on dev views for both Step 1 false rates (not an
    #                               FRR bound); a step with no grid value meeting it fails (CalibrationInfeasible)
    frr_limit: float = 0.10       # the dev check's limit: a check, never an objective
    k_sensitivity: tuple[int, ...] = DEFAULT_K_SENSITIVITY
    lam_sensitivity: tuple[float, ...] = DEFAULT_LAM_SENSITIVITY
    conf: float = 0.95
    bootstrap_B: int = 2000
    seed: int = 0

    def __post_init__(self) -> None:
        # One spelling per value, so settings_sha256 and the candidate bytes do not depend on how a number was written
        # (review: the default cap_steps were ints and --cap-steps 1,2,3,4,6 parsed to floats, two hashes for one
        # setting). Grids, caps, lam values and the rates are floats; k values are ints once validated below.
        for n in ("cap_steps", "tau_match_grid", "tau_q_grid", "tau_occ_grid", "lam_sensitivity"):
            object.__setattr__(self, n, tuple(float(x) for x in getattr(self, n)))
        for n in ("alpha_clear", "frr_limit", "conf"):
            object.__setattr__(self, n, float(getattr(self, n)))
        if not 1 <= self.budget <= 4:
            raise ValueError("budget must be 1..4 (the dev check needs at least one additional view)")
        for n in ("tau_match_grid", "tau_q_grid", "tau_occ_grid"):
            g = getattr(self, n)
            if not g:
                raise ValueError(f"{n} must be non-empty")
            if not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in g):   # NaN breaks the ordering (review F3)
                raise ValueError(f"{n} must contain finite numbers in [0, 1]")
            # A repeated value would count as another grid setting in the reported gate counts, the infeasible record
            # and the settings hash for the same search (review: only tau_match_grid was checked).
            if list(g) != sorted(set(g)):
                raise ValueError(f"{n} must be strictly ascending, without duplicates")
        if not all(math.isfinite(c) and c > 0 for c in self.cap_steps) or \
                list(self.cap_steps) != sorted(set(self.cap_steps)):
            raise ValueError("cap_steps must be strictly ascending, finite and > 0 (the uncapped setting always runs)")
        if isinstance(self.r_seeds, bool) or int(self.r_seeds) != self.r_seeds or self.r_seeds < 1:
            raise ValueError("r_seeds must be an integer >= 1")
        if any(isinstance(k, bool) or int(k) != k or k < 1 for k in self.k_sensitivity) or \
                len(set(self.k_sensitivity)) != len(self.k_sensitivity):
            raise ValueError("k_sensitivity must be unique integers >= 1")
        object.__setattr__(self, "k_sensitivity", tuple(int(k) for k in self.k_sensitivity))
        if K_FIXED in self.k_sensitivity:
            raise ValueError(f"k = {K_FIXED} is fixed a priori; it is not a sensitivity value")
        if not all(math.isfinite(x) and x >= 0 for x in self.lam_sensitivity):   # review F3
            raise ValueError("lam_sensitivity values must be finite and >= 0")
        if list(self.lam_sensitivity) != sorted(set(self.lam_sensitivity)):
            raise ValueError("lam_sensitivity must be strictly ascending")
        if any(abs(x - LAM_FIXED) <= EPS for x in self.lam_sensitivity):
            raise ValueError(f"lam = {LAM_FIXED:g} is fixed a priori; it is not a sensitivity value")
        for n in ("alpha_clear", "frr_limit"):
            if not 0.0 <= getattr(self, n) <= 1.0:
                raise ValueError(f"{n} must be in [0, 1]")
        if not 0.0 < self.conf < 1.0:
            raise ValueError("conf must be in (0, 1)")
        if self.bootstrap_B < 1:
            raise ValueError("bootstrap_B must be >= 1")

    @property
    def cap_settings(self) -> tuple[float | None, ...]:
        """The dev check's travel caps in arc steps: None (uncapped) first, then the grid."""
        return (None, *self.cap_steps)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["fixed_a_priori"] = {"k": K_FIXED, "lam": LAM_FIXED}
        return d


# ---------------------------------------------------------------- Step 1: per-view calibration (views only)

@dataclass(frozen=True)
class ViewRow:
    """One dev view with its diagnostic label; obs is the view exactly as the stopping rule would see it."""
    arrangement_id: str
    episode_id: str
    condition: Condition
    lighting: Lighting
    diagnostic: bool
    obs: Observation

    @property
    def present(self) -> bool:
        return truth_from_condition(self.condition) is TruthState.PRESENT


def view_rows(dev: DevData) -> list[ViewRow]:
    """Every dev view with its label (offline dev analysis, read from a fresh unsealed store)."""
    store = dev.store_factory()
    lab = {(l.episode_id, l.stop): l.diagnostic for l in dev.view_labels}
    rows = []
    for s in dev.specs:
        for stop in range(store.n_stops):
            # step is read by no per-view predicate; 0 marks "a view on its own"
            obs = Observation.from_record(store.get(s.episode_id, stop), step=0, positions=dev.positions)
            rows.append(ViewRow(s.arrangement_id, s.episode_id, s.condition, s.lighting,
                                lab[(s.episode_id, stop)], obs))
    return rows


# The per-view populations. A predicate that holds on a "null" view is an error towards false removal.
def gate_null_views(rows: Iterable[ViewRow]) -> list[ViewRow]:
    """1a's null set: non-diagnostic views of occluded-present episodes (a gate pass here is a false clear)."""
    return [r for r in rows if r.condition is Condition.OCCLUDED and not r.diagnostic]


def present_diagnostic_views(rows: Iterable[ViewRow]) -> list[ViewRow]:
    """1b's null set: diagnostic views of present episodes, i.e. the target is there and visible."""
    return [r for r in rows if r.present and r.diagnostic]


def removed_diagnostic_views(rows: Iterable[ViewRow]) -> list[ViewRow]:
    """The power set of 1a and 1b: diagnostic views of removed episodes."""
    return [r for r in rows if not r.present and r.diagnostic]


def count(rows: Sequence[ViewRow], predicate: Callable[[Observation], bool]) -> tuple[int, int]:
    """(views where the rule's predicate holds, views)."""
    return sum(bool(predicate(r.obs)) for r in rows), len(rows)


def _ratio(xn: tuple[float, int]) -> float | None:
    return xn[0] / xn[1] if xn[1] else None


def frozen_thresholds(tau_match: float, tau_q: float, tau_occ: float) -> Thresholds:
    """The three calibrated values with k and lam fixed a priori (every Thresholds that calibration produces)."""
    return Thresholds(tau_match=tau_match, tau_q=tau_q, tau_occ=tau_occ, k=K_FIXED, lam=LAM_FIXED)


@dataclass(frozen=True)
class GateChoice:
    tau_q: float
    tau_occ: float
    false_clear: tuple[int, int]     # (gate passes, views) non-diagnostic occluded-present views
    clear: tuple[int, int]           # (gate passes, views) diagnostic removed-episode views
    n_feasible: int                  # grid gates that met alpha_clear (a GateChoice exists only if >= 1 did)
    n_grid: int
    # every grid gate: (tau_q, tau_occ, false_clear, clear); selection_details.json only
    grid: tuple[tuple[float, float, tuple[int, int], tuple[int, int]], ...] = field(default=(), repr=False,
                                                                                     compare=False)

    @property
    def false_clear_rate(self) -> float | None:
        return _ratio(self.false_clear)

    @property
    def clear_rate(self) -> float | None:
        return _ratio(self.clear)

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "grid"}
        return {**d, "false_clear_rate": self.false_clear_rate, "clear_rate": self.clear_rate}


def choose_gate(rows: Sequence[ViewRow], tau_q_grid: Sequence[float], tau_occ_grid: Sequence[float],
                alpha_clear: float, warnings: list[str] | None = None) -> GateChoice:
    """Step 1a (select-v3's Step 2 rule): false-clear rate <= alpha_clear, then max clear rate; ties lower false-clear
    rate, higher tau_q, lower tau_occ. A gate pass is the rule's own StoppingRule.passes_gate.

    Raises CalibrationInfeasible if no gate on the grid meets alpha_clear, or if there is no null view to measure the
    false-clear rate on. Nothing is relaxed and there is no fallback (select-v3 took the lowest false-clear rate)."""
    null, power = gate_null_views(rows), removed_diagnostic_views(rows)
    warn = warnings if warnings is not None else []
    cands = []
    for tq in tau_q_grid:
        for to in tau_occ_grid:
            rule = StoppingRule(frozen_thresholds(Thresholds().tau_match, tq, to))   # the gate ignores tau_match
            cands.append((tq, to, count(null, rule.passes_gate), count(power, rule.passes_gate)))
    feasible = [c for c in cands if null and c[2][0] / c[2][1] <= alpha_clear + EPS]
    if not feasible:
        raise CalibrationInfeasible(infeasibility(
            "1a", GATE_CONSTRAINT, alpha_clear, len(null), [({"tau_q": tq, "tau_occ": to}, fc) for tq, to, fc, _ in cands],
            {"tau_q_grid": list(tau_q_grid), "tau_occ_grid": list(tau_occ_grid)}, rows))
    if not power:
        warn.append("Step 1a: no diagnostic removed-episode views; the clear rate is 0 for every gate, so the tie-breaks "
                    "pick the most conservative feasible gate")
    best = min(feasible, key=lambda c: (-(_ratio(c[3]) or 0.0), _ratio(c[2]), -c[0], c[1]))
    return GateChoice(best[0], best[1], best[2], best[3], len(feasible), len(cands), tuple(cands))


@dataclass(frozen=True)
class MatchPoint:
    """Per-view counts at one tau_match (gate fixed), each from the rule's own predicate."""
    tau_match: float
    false_clear_miss: tuple[int, int]     # is_clear_miss on diagnostic views of present episodes
    clear_miss: tuple[int, int]           # is_clear_miss on diagnostic views of removed episodes
    false_strong_match: tuple[int, int]   # is_strong_match on diagnostic views of removed episodes (-> PRESENT)
    feasible: bool                        # false clear-miss rate <= alpha_clear (and measurable)

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "false_clear_miss_rate": _ratio(self.false_clear_miss),
                "clear_miss_rate": _ratio(self.clear_miss), "false_strong_match_rate": _ratio(self.false_strong_match)}


@dataclass(frozen=True)
class MatchChoice:
    tau_match: float
    n_feasible: int                 # grid values that met alpha_clear (a MatchChoice exists only if >= 1 did)
    curve: tuple[MatchPoint, ...]   # every grid value, ascending

    @property
    def chosen(self) -> MatchPoint:
        return next(p for p in self.curve if p.tau_match == self.tau_match)

    def to_dict(self) -> dict[str, Any]:
        c = self.chosen
        return {"tau_match": self.tau_match, "n_feasible": self.n_feasible,
                "n_grid": len(self.curve), "false_clear_miss": c.false_clear_miss,
                "false_clear_miss_rate": _ratio(c.false_clear_miss), "clear_miss": c.clear_miss,
                "clear_miss_rate": _ratio(c.clear_miss), "false_strong_match": c.false_strong_match,
                "false_strong_match_rate": _ratio(c.false_strong_match),
                "curve": [p.to_dict() for p in self.curve]}


def choose_tau_match(rows: Sequence[ViewRow], gate: GateChoice, tau_match_grid: Sequence[float], alpha: float,
                     warnings: list[str] | None = None) -> MatchChoice:
    """Step 1b, with the Step 1a gate: false clear-miss rate (diagnostic views of present episodes) <= alpha, then
    max clear-miss rate on diagnostic views of removed episodes; ties lower false clear-miss rate, then the lower
    (more conservative) tau_match.

    Every count is StoppingRule.is_clear_miss / is_strong_match at the candidate Thresholds: a strong match and a
    clear miss both read tau_match (verify/stopping.py), so the per-view criteria are exactly the rule's.
    Raises CalibrationInfeasible if no grid value meets alpha under this gate, or if there is no diagnostic view of a
    present episode to measure the false clear-miss rate on. Nothing is relaxed and there is no fallback, and no
    other gate is tried (a known gap: the next checkpoint searches gate and tau_match jointly).
    """
    null, power = present_diagnostic_views(rows), removed_diagnostic_views(rows)
    warn = warnings if warnings is not None else []
    pts = []
    for tm in tau_match_grid:
        rule = StoppingRule(frozen_thresholds(tm, gate.tau_q, gate.tau_occ))
        fcm = count(null, rule.is_clear_miss)
        pts.append(MatchPoint(tm, fcm, count(power, rule.is_clear_miss), count(power, rule.is_strong_match),
                              bool(null) and fcm[0] / fcm[1] <= alpha + EPS))
    feasible = [p for p in pts if p.feasible]
    if not feasible:
        raise CalibrationInfeasible(infeasibility(
            "1b", MATCH_CONSTRAINT, alpha, len(null), [({"tau_match": p.tau_match}, p.false_clear_miss) for p in pts],
            {"tau_match_grid": list(tau_match_grid), "gate": {"tau_q": gate.tau_q, "tau_occ": gate.tau_occ}}, rows))
    if not power:
        warn.append("Step 1b: no diagnostic removed-episode views; the clear-miss rate is 0 for every tau_match, so the "
                    "tie-breaks pick the lowest feasible tau_match")
    best = min(feasible, key=lambda p: (-(_ratio(p.clear_miss) or 0.0), _ratio(p.false_clear_miss), p.tau_match))
    return MatchChoice(best.tau_match, len(feasible), tuple(pts))


@dataclass(frozen=True)
class Calibration:
    """Step 1 on one set of arrangements: the gate, then tau_match with that gate; k and lam fixed."""
    gate: GateChoice
    match: MatchChoice
    arrangements: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def thresholds(self) -> Thresholds:
        return frozen_thresholds(self.match.tau_match, self.gate.tau_q, self.gate.tau_occ)


def calibrate(rows: Sequence[ViewRow], settings: SelectionSettings, arrangements: Iterable[str] | None = None,
              *, log_warnings: bool = True) -> Calibration:
    """Steps 1-2 on the views of the given arrangements (default: every row). It reads views and their labels only:
    it never builds an env, runs a policy or sees an episode decision. Raises CalibrationInfeasible when no gate on
    the grid meets alpha_clear (Step 1a), or no tau_match on the grid does under the chosen gate (Step 1b)."""
    if arrangements is not None:
        keep = set(arrangements)
        rows = [r for r in rows if r.arrangement_id in keep]
    warnings: list[str] = []
    gate = choose_gate(rows, settings.tau_q_grid, settings.tau_occ_grid, settings.alpha_clear, warnings)
    match = choose_tau_match(rows, gate, settings.tau_match_grid, settings.alpha_clear, warnings)
    if log_warnings:
        for w in warnings:
            log.warning(w)
    return Calibration(gate, match, tuple(sorted({r.arrangement_id for r in rows})), tuple(warnings))


VIEW_RATES = ("false_clear", "clear", "false_clear_miss", "clear_miss", "false_strong_match")
VIEW_ERRORS = ("false_clear", "false_clear_miss", "false_strong_match")   # reported with upper bounds


@dataclass(frozen=True)
class ViewRates:
    """Per-view counts (x, n) at one threshold setting, each from the rule's own predicate:

    false_clear         passes_gate on non-diagnostic views of occluded-present episodes (1a's null set)
    clear               passes_gate on diagnostic views of removed episodes (1a's power set)
    false_clear_miss    is_clear_miss on diagnostic views of present episodes (1b's null set)
    clear_miss          is_clear_miss on diagnostic views of removed episodes (1b's power set)
    false_strong_match  is_strong_match on diagnostic views of removed episodes (each one decides PRESENT)
    """
    false_clear: tuple[int, int] = (0, 0)
    clear: tuple[int, int] = (0, 0)
    false_clear_miss: tuple[int, int] = (0, 0)
    clear_miss: tuple[int, int] = (0, 0)
    false_strong_match: tuple[int, int] = (0, 0)

    @staticmethod
    def of(rows: Sequence[ViewRow], th: Thresholds) -> ViewRates:
        rule = StoppingRule(th)
        null, present, removed = gate_null_views(rows), present_diagnostic_views(rows), removed_diagnostic_views(rows)
        return ViewRates(count(null, rule.passes_gate), count(removed, rule.passes_gate),
                         count(present, rule.is_clear_miss), count(removed, rule.is_clear_miss),
                         count(removed, rule.is_strong_match))

    def __add__(self, other: ViewRates) -> ViewRates:
        return ViewRates(*((a[0] + b[0], a[1] + b[1])
                           for a, b in zip(dataclasses.astuple(self), dataclasses.astuple(other))))

    def rate(self, name: str) -> float | None:
        return _ratio(getattr(self, name))

    def to_dict(self, conf: float | None = None) -> dict[str, Any]:
        """{name: {x, n, rate}}, plus one-sided CP / Jeffreys bounds when conf is given (views of one
        arrangement are correlated, so these are optimistic)."""
        out: dict[str, Any] = {}
        for name in VIEW_RATES:
            x, n = getattr(self, name)
            d: dict[str, Any] = {"x": x, "n": n, "rate": self.rate(name)}
            if conf is not None and n:
                if name in VIEW_ERRORS:
                    d["upper"] = {"cp": clopper_pearson_upper(x, n, conf), "jeffreys": jeffreys_upper(x, n, conf)}
                else:
                    d["lower"] = {"cp": clopper_pearson_lower(x, n, conf), "jeffreys": jeffreys_lower(x, n, conf)}
            out[name] = d
        return out


def rates_by_lighting(rows: Sequence[ViewRow], th: Thresholds) -> dict[str, ViewRates]:
    """{"L1", "L2", "pooled"}: per-view rates on L1 views, L2 views and both (the calibration set)."""
    out = {light: ViewRates.of([r for r in rows if r.lighting.value == light], th) for light in REPORT_LIGHTINGS}
    out[POOLED] = ViewRates.of(rows, th)
    return out


def alpha_warnings(rates: Mapping[str, ViewRates], alpha: float, what: str, keys: Sequence[str]) -> list[str]:
    """Per-view false rates above alpha_clear. Reported only: the thresholds are never adjusted for them."""
    out = []
    for key in keys:
        for name, label in (("false_clear", "false-clear rate (gate)"),
                            ("false_clear_miss", "false clear-miss rate (tau_match)")):
            x, n = getattr(rates[key], name)
            if n and x / n > alpha + EPS:
                views = "L1 + L2" if key == POOLED else key
                out.append(f"{what}: {label} on {views} views is {x / n:.3f} ({x}/{n}) > alpha_clear = {alpha:g}; "
                           "reported, not adjusted (one threshold set serves both lightings)")
    return out


# ---------------------------------------------------------------- Step 3: grouped CV of the calibration

@dataclass(frozen=True)
class ViewFold:
    """One leave-one-arrangement-out fold: Step 1 without held_out, scored on held_out's views. If Step 1 is
    infeasible without held_out, the fold has no thresholds and no held-out rates (no fallback setting)."""
    held_out: str
    thresholds: Thresholds | None              # None: calibrating without held_out was infeasible
    held: Mapping[str, ViewRates] | None       # "L1", "L2", "pooled"; None when infeasible
    infeasible: Infeasibility | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"held_out": self.held_out,
                "thresholds": None if self.thresholds is None else self.thresholds.to_dict(),
                "held": None if self.held is None else {k: v.to_dict() for k, v in self.held.items()},
                "infeasible": None if self.infeasible is None else {
                    "step": self.infeasible.step, "summary": self.infeasible.summary(),
                    "best_rate": self.infeasible.best_rate, "best_counts": self.infeasible.best_counts}}


def grouped_cv(rows: Sequence[ViewRow], settings: SelectionSettings, arrangements: Iterable[str]) -> list[ViewFold]:
    """Step 3 (views only): for each arrangement, calibrate on the other arrangements' views and score its own.
    A fold whose calibration is infeasible is recorded as such, with no thresholds. Fewer than two arrangements:
    no folds."""
    arrs = sorted(set(arrangements))
    if len(arrs) < 2:
        return []
    folds = []
    for a in arrs:
        try:
            cal = calibrate(rows, settings, [x for x in arrs if x != a], log_warnings=False)
        except CalibrationInfeasible as e:
            folds.append(ViewFold(a, None, None, e.failure))
            continue
        held = [r for r in rows if r.arrangement_id == a]
        folds.append(ViewFold(a, cal.thresholds, rates_by_lighting(held, cal.thresholds)))
    return folds


def pool_folds(folds: Sequence[ViewFold]) -> dict[str, ViewRates]:
    """Held-out counts summed over the feasible folds: each of their arrangements' views is held out exactly once."""
    out = {k: ViewRates() for k in (*REPORT_LIGHTINGS, POOLED)}
    for f in folds:
        if f.held is None:
            continue
        for k in out:
            out[k] = out[k] + f.held[k]
    return out


# ---------------------------------------------------------------- Steps 4-5: policy runs (after the freeze)

def make_policy(name: str, th: Thresholds) -> Any:
    """A fresh deterministic policy through its public constructor (R is built per seed by run_r_seeds).

    "G" is primary G (D-norm, lookahead off; DECISIONS 2026-10-12), "G-lookahead" the same scorer with the
    reachability lookahead; both read th.lam. G-myopic keeps its legacy travel weight (GMYOPIC_LAM_DEG degrees),
    as in verify.run. S, F and R ignore lam.
    """
    if name == "S":
        return S()
    if name == "F":
        return F()
    if name == "G":
        return G.from_thresholds(th, lookahead=False)
    if name == "G-lookahead":
        return G.from_thresholds(th, lookahead=True)
    if name == "G-myopic":
        return GMyopic.from_thresholds(th)
    raise ValueError(f"unsupported policy {name!r} (R runs through run_r_seeds)")


def run_budget(policy: str, cap_steps: float | None, settings: SelectionSettings, positions: Positions) -> Budget:
    """S: Budget(0) (one view; budget- and cap-independent). Every other policy: the primary budget under the cap."""
    return Budget(0) if policy == "S" else Budget(settings.budget, cap_cm_from_steps(cap_steps, positions))


class PolicyRunner:
    """Runs one policy on every dev episode through the public replay API (Steps 4-5 only).

    run_selection builds it only after the thresholds are frozen, and it only ever receives Thresholds. One fresh
    store per run setting (all keys minted before the first bind seals it); results are cached per setting.
    """

    def __init__(self, dev: DevData, settings: SelectionSettings, context: RunContext) -> None:
        self.dev = dev
        self.r_seeds = tuple(range(settings.r_seeds))
        self.context = context
        self._cache: dict[tuple[str, Thresholds, Budget], tuple[EpisodeResult, ...]] = {}
        self.n_runs = 0

    @property
    def n_settings(self) -> int:
        return len(self._cache)

    def results(self, policy: str, th: Thresholds, budget: Budget) -> tuple[EpisodeResult, ...]:
        """One EpisodeResult per dev episode (R: averaged over its seeds within the episode)."""
        key = (policy, th, budget)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        rule = StoppingRule.first_clear_miss(th) if policy == "S" else StoppingRule(th)
        store = self.dev.store_factory()       # fresh store per setting
        keys = store.replay_keys()             # minted before the first bind seals it
        out = []
        for spec in self.dev.specs:
            def make_env(key=keys[spec.episode_id]) -> ReplayEnv:
                return ReplayEnv(key, store, self.dev.positions, budget, self.context)
            if policy == "R":
                logs, res = run_r_seeds(make_env, rule, self.r_seeds, r_seed_salt(spec), spec=spec)
                self.n_runs += len(logs)
            else:
                run = run_episode(make_policy(policy, th), make_env(), rule)
                res = EpisodeResult.from_logs([EpisodeLog.from_run(spec, run)])
                self.n_runs += 1
            out.append(res)
        self._cache[key] = tuple(out)
        return self._cache[key]


@dataclass(frozen=True)
class Outcome:
    """One dev episode under one run setting (truth joined; scoring side only). removed and resolved are decision
    fractions: 0 or 1 for a deterministic policy, the mean over seeds for R."""
    episode_id: str
    arrangement_id: str
    lighting: Lighting
    present: bool
    removed: float
    resolved: float
    travel_cm: float
    n_views: float

    @classmethod
    def of(cls, res: EpisodeResult, truth: GroundTruth) -> Outcome:
        return cls(res.episode_id, res.arrangement_id, res.lighting, truth[res.episode_id] is TruthState.PRESENT,
                   res.frac(Decision.REMOVED), 1.0 - res.frac(Decision.UNKNOWN), res.mean_travel_cm, res.mean_views)


@dataclass(frozen=True)
class Counts:
    n_episodes: int = 0
    n_present: int = 0
    n_false_removal: float = 0
    n_removed: int = 0
    n_true_removal: float = 0
    n_resolved: float = 0
    total_travel_cm: float = 0.0
    total_views: float = 0.0
    n_arrangements: int = 0
    n_arr_with_present: int = 0
    n_arr_with_false_removal: int | None = 0   # None when the decisions are seed-averaged (R)
    binomial: bool = True                      # integer counts: one deterministic run per episode

    @staticmethod
    def of(outcomes: Iterable[Outcome], *, binomial: bool = True) -> Counts:
        outs = list(outcomes)
        if binomial and any(o.removed not in (0.0, 1.0) or o.resolved not in (0.0, 1.0) for o in outs):
            raise ValueError("binomial counts need 0/1 decisions (one deterministic run per episode)")
        arr_present: dict[str, bool] = {}
        arr_fr: dict[str, bool] = {}
        for o in outs:
            arr_present[o.arrangement_id] = arr_present.get(o.arrangement_id, False) or o.present
            arr_fr[o.arrangement_id] = arr_fr.get(o.arrangement_id, False) or (o.present and o.removed > 0)
        fr = sum(o.removed for o in outs if o.present)
        tr = sum(o.removed for o in outs if not o.present)
        res = sum(o.resolved for o in outs)
        if binomial:
            fr, tr, res = round(fr), round(tr), round(res)
        return Counts(n_episodes=len(outs), n_present=sum(o.present for o in outs), n_false_removal=fr,
                      n_removed=sum(not o.present for o in outs), n_true_removal=tr, n_resolved=res,
                      total_travel_cm=sum(o.travel_cm for o in outs), total_views=sum(o.n_views for o in outs),
                      n_arrangements=len(arr_present), n_arr_with_present=sum(arr_present.values()),
                      n_arr_with_false_removal=sum(arr_fr.values()) if binomial else None, binomial=binomial)

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

    @property
    def mean_views(self) -> float | None:
        return self.total_views / self.n_episodes if self.n_episodes else None

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "frr": self.frr, "recall": self.recall, "coverage": self.coverage,
                "mean_travel_cm": self.mean_travel_cm, "mean_views": self.mean_views}


def bounds(c: Counts, conf: float) -> dict[str, Any]:
    """One-sided CP and Jeffreys bounds; None where the denominator is 0 or the counts are not binomial (R)."""
    def up(x: Any, n: int) -> dict[str, float] | None:
        if n == 0 or x is None or not c.binomial:
            return None
        return {"cp": clopper_pearson_upper(x, n, conf), "jeffreys": jeffreys_upper(x, n, conf)}

    def lo(x: Any, n: int) -> dict[str, float] | None:
        if n == 0 or not c.binomial:
            return None
        return {"cp": clopper_pearson_lower(x, n, conf), "jeffreys": jeffreys_lower(x, n, conf)}
    out: dict[str, Any] = {
        "conf": conf, "binomial": c.binomial,
        "frr_episode": {"x": c.n_false_removal, "n": c.n_present, "upper": up(c.n_false_removal, c.n_present)},
        "frr_arrangement": {"x": c.n_arr_with_false_removal, "n": c.n_arr_with_present,
                            "upper": up(c.n_arr_with_false_removal, c.n_arr_with_present)},
        "recall": {"x": c.n_true_removal, "n": c.n_removed, "lower": lo(c.n_true_removal, c.n_removed)},
        "coverage": {"x": c.n_resolved, "n": c.n_episodes, "lower": lo(c.n_resolved, c.n_episodes)},
    }
    if not c.binomial:
        out["note"] = NOT_BINOMIAL
    return out


@functools.lru_cache(maxsize=64)
def _draw_counts(n_arrangements: int, B: int, seed: int) -> np.ndarray:
    """(B, n) multiplicity of each arrangement in each bootstrap draw (analysis.bootstrap.arrangement_draws)."""
    draws = arrangement_draws(n_arrangements, B, seed)
    counts = np.zeros((B, n_arrangements))
    np.add.at(counts, (np.arange(B)[:, None], draws), 1.0)
    counts.setflags(write=False)
    return counts


def cluster_bootstrap(outcomes: Sequence[Outcome], B: int, seed: int) -> dict[str, Any]:
    """Percentile 95% CIs for FRR, recall, coverage, resampling arrangements with replacement (fractions for R)."""
    arrs = sorted({o.arrangement_id for o in outcomes})
    out: dict[str, Any] = {"B": B, "seed": seed, "n_arrangements": len(arrs)}
    if not arrs:
        return {**out, **{m: {"ci95": [None, None], "n_undefined": B} for m in ("frr", "recall", "coverage")}}
    idx = {a: i for i, a in enumerate(arrs)}
    terms = {m: (np.zeros(len(arrs)), np.zeros(len(arrs))) for m in ("frr", "recall", "coverage")}
    for o in outcomes:
        i = idx[o.arrangement_id]
        m = "frr" if o.present else "recall"
        terms[m][0][i] += o.removed
        terms[m][1][i] += 1
        terms["coverage"][0][i] += o.resolved
        terms["coverage"][1][i] += 1
    counts = _draw_counts(len(arrs), B, seed)
    for m, (num, den) in terms.items():
        n, d = counts @ num, counts @ den
        ok = d > EPS
        vals = n[ok] / d[ok]
        ci = [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals.size else [None, None]
        out[m] = {"ci95": ci, "n_undefined": int((~ok).sum())}
    return out


@dataclass(frozen=True)
class CheckRow:
    """One policy at one run setting, one query lighting (Steps 4-5). Lighting "pooled" is a secondary line."""
    policy: str
    k: int                          # the rule's k (S: 1, its one-view rule)
    lam: float                      # Thresholds.lam (read by G and G-lookahead only)
    budget: int                     # additional views (S: 0)
    cap_steps: float | None         # None = uncapped
    travel_cap_cm: float | None
    lighting: str                   # "L1", "L2" or "pooled"
    counts: Counts
    bounds: Mapping[str, Any] | None
    bootstrap: Mapping[str, Any] | None
    over_limit: bool | None         # FRR > frr_limit (None when FRR is undefined)

    def to_dict(self) -> dict[str, Any]:
        return {"policy": self.policy, "k": self.k, "lam": self.lam, "budget": self.budget,
                "cap_steps": self.cap_steps, "travel_cap_cm": self.travel_cap_cm, "lighting": self.lighting,
                **self.counts.to_dict(), "bounds": self.bounds, "bootstrap": self.bootstrap,
                "frr_over_limit": self.over_limit}


def setting_rows(runner: PolicyRunner, policy: str, th: Thresholds, cap_steps: float | None,
                 settings: SelectionSettings, *, with_bounds: bool = True) -> list[CheckRow]:
    """Run one policy at (th, primary budget, cap) on every dev episode; one row per L1, L2 and pooled."""
    budget = run_budget(policy, cap_steps, settings, runner.dev.positions)
    if policy == "S":
        cap_steps = None
    outs = [Outcome.of(r, runner.dev.truth) for r in runner.results(policy, th, budget)]
    k = 1 if policy == "S" else th.k
    rows = []
    for light in (*REPORT_LIGHTINGS, POOLED):
        sub = outs if light == POOLED else [o for o in outs if o.lighting.value == light]
        c = Counts.of(sub, binomial=policy != "R")
        rows.append(CheckRow(policy, k, th.lam, budget.additional_views, cap_steps, budget.travel_cap_cm, light, c,
                             bounds(c, settings.conf) if with_bounds else None,
                             cluster_bootstrap(sub, settings.bootstrap_B, settings.seed) if with_bounds else None,
                             None if c.frr is None else c.frr > settings.frr_limit + EPS))
    return rows


def run_check(runner: PolicyRunner, th: Thresholds, settings: SelectionSettings) -> list[CheckRow]:
    """Step 4: every policy at the frozen thresholds, the primary budget, uncapped and every cap (S once)."""
    rows = []
    for p in CHECK_POLICIES:
        for cap in ((None,) if p == "S" else settings.cap_settings):
            rows += setting_rows(runner, p, th, cap, settings)
    return rows


def run_sensitivity(runner: PolicyRunner, th: Thresholds, settings: SelectionSettings) -> dict[str, list[CheckRow]]:
    """Step 5, reported only and never selected: each k_sensitivity value (lam = 0) and each lam_sensitivity value
    (k = 2), always with the frozen taus. Point estimates only (the check carries the bounds)."""
    k_rows: list[CheckRow] = []
    lam_rows: list[CheckRow] = []
    for k in settings.k_sensitivity:
        t = dataclasses.replace(th, k=k)
        for p in K_SENSITIVITY_POLICIES:
            for cap in settings.cap_settings:
                k_rows += setting_rows(runner, p, t, cap, settings, with_bounds=False)
    for lam in settings.lam_sensitivity:
        t = dataclasses.replace(th, lam=lam)
        for p in LAM_SENSITIVITY_POLICIES:
            for cap in settings.cap_settings:
                lam_rows += setting_rows(runner, p, t, cap, settings, with_bounds=False)
    return {"k": k_rows, "lam": lam_rows}


def _num(x: float | int | None) -> str:
    if x is None:
        return "undef"
    return f"{x:g}" if float(x).is_integer() else f"{x:.2f}"


def setting_label(r: CheckRow) -> str:
    if r.policy == "S":
        return "1 view (S: no budget, no cap)"
    cap = "uncapped" if r.cap_steps is None else f"cap {_steps(r.cap_steps)} ({r.travel_cap_cm:.2f} cm)"
    return f"b = {r.budget}, {cap}"


def check_flags(rows: Sequence[CheckRow], frr_limit: float) -> list[str]:
    """Every (policy, setting, lighting) of the dev check whose FRR is above the limit (L1 and L2 rows)."""
    return [f"dev check: {r.policy}, {setting_label(r)}, {r.lighting} queries: FRR {r.counts.frr:.3f} "
            f"({_num(r.counts.n_false_removal)}/{r.counts.n_present}) > frr_limit {frr_limit:g}; the thresholds "
            "were not changed (FRR is a check, not an objective)"
            for r in rows if r.lighting in REPORT_LIGHTINGS and r.over_limit]


def check_unchecked(rows: Sequence[CheckRow]) -> list[str]:
    """Every L1 / L2 row of the dev check whose FRR is undefined (no present-target episode at that lighting): the FRR
    limit was not checked there, so the check cannot pass on it."""
    return [f"dev check: {r.policy}, {setting_label(r)}, {r.lighting} queries: FRR undefined (no present-target "
            "episode), so the FRR limit was not checked"
            for r in rows if r.lighting in REPORT_LIGHTINGS and r.over_limit is None]


def all_within_limit(flags: Sequence[str], unchecked: Sequence[str]) -> bool | None:
    """dev_check.all_within_limit: True only if every L1 / L2 row was checked and none is above the limit. False if a
    row is flagged (with or without unchecked rows, the answer is then known). None if no row is flagged but some row
    was not checked: an L1-only dev set used to pass "all within limit" with its L2 check never run (review)."""
    if flags:
        return False
    return None if unchecked else True


# ---------------------------------------------------------------- the whole procedure

def run_selection(dev: DevData, settings: SelectionSettings, context: RunContext) -> dict[str, Any]:
    """select-v4 on dev; returns everything the report and the frozen candidate need.

    The order is the guarantee: Steps 1-3 read views only; the PolicyRunner is created after the thresholds are
    frozen and only ever receives them (or, in Step 5, a copy with k or lam replaced). If Step 1 is infeasible on
    the dev arrangements, CalibrationInfeasible propagates before any fold or policy runs.
    """
    refuse_non_dev(dev.specs)
    rows = view_rows(dev)
    arrs = dev.arrangements
    cal = calibrate(rows, settings)                       # Steps 1-2: views only; raises CalibrationInfeasible
    th = cal.thresholds                                   # frozen from here on
    rates = rates_by_lighting(rows, th)
    extra = alpha_warnings(rates, settings.alpha_clear, "Step 1 (calibration set)", REPORT_LIGHTINGS)
    folds = grouped_cv(rows, settings, arrs)              # Step 3: views only
    held = pool_folds(folds)
    bad = [f for f in folds if f.infeasible is not None]
    for f in bad:
        extra.append(f"Step 3: calibrating without {f.held_out} is infeasible ({f.infeasible.summary()}); that fold "
                     f"has no thresholds and no held-out rates (no fallback), so the held-out estimate covers "
                     f"{len(folds) - len(bad)} of {len(folds)} arrangements")
    if folds:
        extra += alpha_warnings(held, settings.alpha_clear, "Step 3 (held out)", (*REPORT_LIGHTINGS, POOLED))
    else:
        extra.append("Step 3: fewer than 2 dev arrangements; no grouped cross-validation")
    for w in extra:
        log.warning(w)
    runner = PolicyRunner(dev, settings, context)         # Steps 4-5: the frozen thresholds only
    check = run_check(runner, th, settings)
    flags = check_flags(check, settings.frr_limit)
    for f in flags:
        log.warning(f)
    unchecked = check_unchecked(check)
    if unchecked:
        log.warning(f"Step 4: FRR undefined in {len(unchecked)} L1 / L2 row(s) of the dev check (no present-target "
                    "episode); the FRR limit was not checked there")
    sensitivity = run_sensitivity(runner, th, settings)
    lam_check = check_lam_grid((LAM_FIXED, *settings.lam_sensitivity), dev.positions, lookahead=False)
    for w in lam_check.warnings:
        log.warning(w)
    return {"calibration": cal, "thresholds": th, "view_rates": rates, "folds": folds, "cv_held_out": held,
            "check": check, "flags": flags, "unchecked": unchecked,
            "all_within_limit": all_within_limit(flags, unchecked),
            "sensitivity": sensitivity, "lam_grid_check": lam_check,
            "cv_complete": bool(folds) and not bad,
            "warnings": [*cal.warnings, *extra, *lam_check.warnings], "runner": runner,
            "n_view_rows": len(rows), "n_episode_runs": runner.n_runs, "n_run_settings": runner.n_settings}


# Settings in words for the report. An empty list says so plainly (review: "k = (none) runs only as a reported
# sensitivity", "lam in {} ..." and "uncapped,  arc steps").
def _values_text(name: str, values: Sequence[float]) -> str:
    text = [f"{v:g}" for v in values]
    return f"{name} = {text[0]}" if len(text) == 1 else f"{name} in {{{', '.join(text)}}}"


def _k_sensitivity_text(settings: SelectionSettings) -> str:
    if not settings.k_sensitivity:
        return "no k sensitivity run was requested"
    return f"{_values_text('k', settings.k_sensitivity)} runs only as a reported sensitivity"


def _lam_sensitivity_text(settings: SelectionSettings) -> str:
    if not settings.lam_sensitivity:
        return "no lam sensitivity run was requested"
    return f"{_values_text('lam', settings.lam_sensitivity)} runs only as a reported sensitivity"


def _caps_text(settings: SelectionSettings) -> str:
    if not settings.cap_steps:
        return "uncapped only (no capped setting was run)"
    return "uncapped, " + ", ".join(f"{c:g}" for c in settings.cap_steps) + " arc steps"


def candidate_doc(dev: DevData, settings: SelectionSettings, res: dict[str, Any], settings_sha: str,
                  provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """provenance: for --synthetic runs, the generator parameters, n_arr and n_dev (review F4); None on real data,
    where dev_evidence_sha256 already identifies the evidence."""
    cal: Calibration = res["calibration"]
    th: Thresholds = res["thresholds"]
    conf = settings.conf
    doc = {
        "schema": CANDIDATE_SCHEMA, "procedure_version": PROCEDURE_VERSION,
        "status": "CANDIDATE - not frozen until the student signs FREEZE.md",
        "synthetic": dev.synthetic, "label": SYNTHETIC_LABEL if dev.synthetic else "",
        "settings": settings.to_dict(), "settings_sha256": settings_sha, "proposed_defaults_note": PROPOSED,
        "dev_arrangements": list(dev.arrangements), "n_dev_episodes": len(dev.specs), "n_dev_views": res["n_view_rows"],
        "dev_evidence_sha256": dev.evidence_sha256, "view_labels_sha256": dev.view_labels_sha256,
        "view_label_source": dev.label_source,
        "positions": dev.positions.to_dict(),
        "thresholds": th.to_dict(),
        "calibrated_per_view": ["tau_q", "tau_occ", "tau_match"],
        # The notes name no sensitivity value (it said "k = 3" even with none requested): this section must not
        # depend on the sensitivity settings at all, and sensitivity.k_values / lam_values list what ran.
        "fixed_a_priori": {
            "k": K_FIXED, "lam": LAM_FIXED,
            "k_note": ("the plan's 'at least two distinct positions'; any other k runs only as a reported sensitivity "
                       "(sensitivity.k_values)"),
            "lam_note": ("G and A choose by spread or informativeness alone; travel is limited by the caps, reported "
                         "as curves; any other lam runs only as a reported sensitivity (sensitivity.lam_values)")},
        "selected_on_policy_results": False,
        "alpha_clear": settings.alpha_clear,
        "alpha_note": ALPHA_NOTE,
        "calibration": {
            "gate": cal.gate.to_dict(),
            "tau_match": cal.match.to_dict(),
            "per_view_rates": {k: v.to_dict(conf) for k, v in res["view_rates"].items()},
            "grouped_cv": {
                "complete": res["cv_complete"],
                "infeasible_folds": [f.held_out for f in res["folds"] if f.infeasible is not None],
                "held_out_per_view_rates": {k: v.to_dict(conf) for k, v in res["cv_held_out"].items()},
                "folds": [f.to_dict() for f in res["folds"]],
                "n_distinct_threshold_sets": len({f.thresholds for f in res["folds"] if f.thresholds is not None})},
        },
        "dev_check": {
            "frr_limit": settings.frr_limit, "primary_budget": settings.budget,
            "cap_steps": list(settings.cap_settings),
            "travel_caps_cm": [cap_cm_from_steps(c, dev.positions) for c in settings.cap_settings],
            "policies": list(CHECK_POLICIES), "g_lookahead": {"G": False, "G-lookahead": True},
            "g_myopic_lam_deg": GMYOPIC_LAM_DEG, "r_seeds": settings.r_seeds,
            "note": ("FRR <= frr_limit is a check, not an objective: flags are reported and nothing is changed to "
                     "fix them. Rows for L1 and L2 queries are the result; 'pooled' is a secondary line. "
                     "all_within_limit is true only if every L1 and L2 row was checked and none is flagged, false "
                     "if one is flagged, and null if none is but FRR is undefined in some row (listed in "
                     "'unchecked'): the check is then incomplete."),
            "flags": res["flags"], "unchecked": res["unchecked"], "all_within_limit": res["all_within_limit"],
            "rows": [r.to_dict() for r in res["check"]]},
        "sensitivity": {
            "note": "reported only, never selected; the frozen taus with k or lam replaced (point estimates)",
            "k_values": list(settings.k_sensitivity), "k_policies": list(K_SENSITIVITY_POLICIES),
            "lam_values": list(settings.lam_sensitivity), "lam_policies": list(LAM_SENSITIVITY_POLICIES),
            "k": [r.to_dict() for r in res["sensitivity"]["k"]],
            "lam": [r.to_dict() for r in res["sensitivity"]["lam"]]},
        "lam_grid_check": res["lam_grid_check"].to_dict(),
        "warnings": res["warnings"],
    }
    if provenance is not None:
        doc["synthetic_provenance"] = dict(provenance)
    return to_jsonable(doc)


# ---------------------------------------------------------------- report

def _f(x: float | None, nd: int = 3) -> str:
    return "undef" if x is None else f"{x:.{nd}f}"


def _xn(xn: tuple[Any, int]) -> str:
    return f"{_num(xn[0])}/{xn[1]}"


def _rate_xn(xn: tuple[Any, int]) -> str:
    return f"{_f(_ratio(xn))} ({_xn(xn)})"


def _cp_up(xn: tuple[int, int], conf: float) -> str:
    return f"{clopper_pearson_upper(xn[0], xn[1], conf):.4f}" if xn[1] else "undef"


def _bound(b: Mapping[str, Any] | None, side: str) -> str:
    if b is None or b.get(side) is None:
        return "n/a"
    return f"{b[side]['cp']:.3f} / {b[side]['jeffreys']:.3f}"


def _ci(boot: Mapping[str, Any] | None, m: str) -> str:
    if boot is None:
        return "-"
    lo, hi = boot[m]["ci95"]
    return f"[{_f(lo)}, {_f(hi)}]"


def _steps(steps: float) -> str:
    return f"{steps:g} step" + ("" if steps == 1 else "s")


def _cap_cell(r: CheckRow) -> str:
    if r.policy == "S":
        return "n/a (1 view)"
    return "uncapped" if r.cap_steps is None else f"{_steps(r.cap_steps)} ({r.travel_cap_cm:.1f} cm)"


def _check_table(rows: Sequence[CheckRow], conf: float) -> list[str]:
    pct = f"{conf:.0%}"
    L = [f"| policy | travel cap | FRR (x/n) | FRR {pct} upper, episode: CP / Jeffreys "
         f"| FRR {pct} upper, arrangement: CP / Jeffreys | FRR bootstrap 95% CI | recall (x/n) "
         f"| recall {pct} lower: CP / Jeffreys | recall bootstrap 95% CI | coverage (x/n) "
         f"| coverage {pct} lower: CP / Jeffreys | coverage bootstrap 95% CI | mean views | mean travel cm "
         "| FRR > limit |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        c, b = r.counts, r.bounds
        fe = _bound(b["frr_episode"] if b else None, "upper")
        fa = _bound(b["frr_arrangement"] if b else None, "upper")
        flag = "undef" if r.over_limit is None else ("**FLAG**" if r.over_limit else "no")
        L.append(f"| {r.policy} | {_cap_cell(r)} | {_f(c.frr)} ({_num(c.n_false_removal)}/{c.n_present}) | {fe} | {fa} "
                 f"| {_ci(r.bootstrap, 'frr')} | {_f(c.recall)} ({_num(c.n_true_removal)}/{c.n_removed}) "
                 f"| {_bound(b['recall'] if b else None, 'lower')} | {_ci(r.bootstrap, 'recall')} "
                 f"| {_f(c.coverage)} ({_num(c.n_resolved)}/{c.n_episodes}) "
                 f"| {_bound(b['coverage'] if b else None, 'lower')} | {_ci(r.bootstrap, 'coverage')} "
                 f"| {_f(c.mean_views, 2)} | {_f(c.mean_travel_cm, 1)} | {flag} |")
    return L


def _sens_table(rows: Sequence[CheckRow], primary: Mapping[tuple[str, float | None, str], CheckRow]) -> list[str]:
    L = ["| policy | travel cap | FRR | recall | coverage | mean travel cm | FRR > limit |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        p = primary.get((r.policy, r.cap_steps, r.lighting))

        def arrow(get: Callable[[CheckRow], float | None], nd: int = 3) -> str:
            return f"{_f(get(p), nd) if p else '-'} -> {_f(get(r), nd)}"
        flag = "undef" if r.over_limit is None else ("yes" if r.over_limit else "no")
        L.append(f"| {r.policy} | {_cap_cell(r)} | {arrow(lambda x: x.counts.frr)} | {arrow(lambda x: x.counts.recall)} "
                 f"| {arrow(lambda x: x.counts.coverage)} | {arrow(lambda x: x.counts.mean_travel_cm, 1)} | {flag} |")
    return L


def _view_table(rates: Mapping[str, ViewRates], conf: float, keys: Sequence[tuple[str, str]]) -> list[str]:
    pct = f"{conf:.0%}"
    L = [f"| views | false-clear rate (x/n) | {pct} CP upper | clear rate (x/n) | false clear-miss rate (x/n) "
         f"| {pct} CP upper | clear-miss rate on removed (x/n) | false strong-match rate on removed (x/n) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, label in keys:
        v = rates[key]
        L.append(f"| {label} | {_rate_xn(v.false_clear)} | {_cp_up(v.false_clear, conf)} | {_rate_xn(v.clear)} "
                 f"| {_rate_xn(v.false_clear_miss)} | {_cp_up(v.false_clear_miss, conf)} | {_rate_xn(v.clear_miss)} "
                 f"| {_rate_xn(v.false_strong_match)} |")
    return L


def frr_bound_note(n_arr: int, frr_limit: float, conf: float) -> str:
    """What the dev check can show, from the run's own number of dev arrangements (review: a literal "about 10 dev
    arrangements" was false for a 60-arrangement sim dev set). The best arrangement-level bound is the one-sided CP
    upper bound when no arrangement has a false removal; if even that is above frr_limit, no dev result gets below
    it."""
    best = clopper_pearson_upper(0, n_arr, conf)
    beyond = (f", above frr_limit = {frr_limit:g}, so no dev result can bring that bound below the limit"
              if best > frr_limit + EPS else "")
    return (f"- With {n_arr} dev arrangement{'' if n_arr == 1 else 's'}, the best one-sided {conf:.0%} "
            "arrangement-level FRR bound (Clopper-Pearson, no arrangement with a false removal) is "
            f"{best:.3f}{beyond}. The check also runs on the arrangements whose views set the thresholds. The FRR "
            "limit and the plan's recall and coverage targets are goals, not guarantees.")


def render_report(dev: DevData, settings: SelectionSettings, res: dict[str, Any], cand: dict[str, Any],
                  cand_sha: str) -> str:
    cal: Calibration = res["calibration"]
    th: Thresholds = res["thresholds"]
    g, m = cal.gate, cal.match
    conf = settings.conf
    syn = dev.synthetic
    L: list[str] = []
    if syn:
        L += [f"# {SYNTHETIC_LABEL}", "",
              "**SYNTHETIC - not study results and not threshold values.** The evidence and the per-view "
              "diagnostic labels come from `verify.synthetic` (labels are a stand-in derived from the "
              "generator's hidden occluder layout). This run only exercises the calibration procedure; "
              "nothing here may be carried into FREEZE.md.", ""]
    flags, unchecked = res["flags"], res["unchecked"]
    within = {True: "true", False: "false", None: "null"}[res["all_within_limit"]]
    L += ["# Threshold calibration report (D-select, select-v4)", "",
          f"- procedure `{PROCEDURE_VERSION}`; settings sha256 `{cand['settings_sha256']}`",
          f"- candidate `frozen_candidate.json` sha256 `{cand_sha}` (status: candidate, not frozen)",
          f"- dev: {len(dev.arrangements)} arrangements, {len(dev.specs)} episodes, {res['n_view_rows']} views; "
          f"evidence sha256 `{dev.evidence_sha256}`",
          f"- view labels: {dev.label_source}; labels sha256 `{dev.view_labels_sha256}`",
          "- **No policy result set any threshold.** Steps 1-3 read the dev views and their labels only. The "
          f"{res['n_episode_runs']} episode replays ({res['n_run_settings']} run settings, each on a fresh "
          "EvidenceStore through ReplayEnv/run_episode) all ran after the thresholds were frozen: the Step 4 "
          "check and the Step 5 sensitivity are reported, never fed back.",
          "- grouped cross-validation of the calibration (Step 3): " + (
              "complete" if res["cv_complete"] else "**INCOMPLETE** (" + (
                  "infeasible without " + ", ".join(f.held_out for f in res["folds"] if f.infeasible is not None)
                  if res["folds"] else "fewer than 2 dev arrangements") + "; see Step 3). A candidate whose grouped "
              "CV is incomplete must not be frozen (docs/SIM_PREREGISTRATION.md section 7); it is not yet marked "
              "`eligible_for_freeze: false` (a known gap)"),
          f"- dev check (Step 4): {len(flags)} flag(s) above FRR {settings.frr_limit:g}"
          + (f"; **INCOMPLETE**: FRR undefined in {len(unchecked)} L1 / L2 row(s), which were not checked"
             if unchecked else "") + f"; `all_within_limit` {within}", "",
          "## Frozen candidate", "",
          "| tau_q | tau_occ | tau_match | k | lam |", "| --- | --- | --- | --- | --- |",
          f"| {th.tau_q} | {th.tau_occ} | {th.tau_match} | {th.k} (fixed a priori) | {th.lam:g} (fixed a priori) |", "",
          "- tau_q, tau_occ and tau_match are calibrated per view on the dev views of both lightings together "
          "(one threshold set serves L1 and L2 queries).",
          f"- alpha_clear = {settings.alpha_clear:g}: {ALPHA_NOTE}",
          f"- k = 2 is the plan's \"at least two distinct positions\"; {_k_sensitivity_text(settings)}.",
          "- lam = 0: G and A choose by spread or informativeness alone; travel is limited by the caps, which are "
          f"reported as curves; {_lam_sensitivity_text(settings)}.", "",
          "## Settings (every value is a *proposed default - confirm*)", "",
          "| setting | value |", "| --- | --- |"]
    sd = settings.to_dict()
    for k in ("budget", "cap_steps", "r_seeds", "tau_match_grid", "tau_q_grid", "tau_occ_grid", "alpha_clear",
              "frr_limit", "k_sensitivity", "lam_sensitivity", "conf", "bootstrap_B", "seed"):
        v = sd[k]
        if k == "cap_steps":
            v = _caps_text(settings)
        L.append(f"| {k} | {v} |")
    L.append(f"| fixed a priori | k = {K_FIXED}, lam = {LAM_FIXED:g} |")
    prov = cand.get("synthetic_provenance")
    if prov:
        L.append(f"| synthetic generator | n_arr {prov['n_arr']}, n_dev {prov['n_dev']}, block_includes_start "
                 f"{prov['generator']['block_includes_start']} |")
    lights = [("pooled", "calibration set: L1 + L2 views"), ("L1", "L1 views"), ("L2", "L2 views")]
    L += ["", "## Step 1a: visibility gate (tau_q, tau_occ) from diagnostic labels", "",
          "Null set: non-diagnostic views of occluded-present episodes (a gate pass there is a false clear). Power "
          "set: diagnostic views of removed episodes. The gate is the rule's own `StoppingRule.passes_gate` "
          "(quality >= tau_q and obstruction <= tau_occ).", "",
          f"- chosen tau_q = {g.tau_q}, tau_occ = {g.tau_occ} ({g.n_feasible} of {g.n_grid} grid gates met "
          f"alpha_clear = {settings.alpha_clear:g} on the dev views; with none, calibration fails, see below)",
          f"- calibration set: false-clear rate {_rate_xn(g.false_clear)}, clear rate {_rate_xn(g.clear)}", "",
          "## Step 1b: tau_match, with the Step 1a gate", "",
          "A strong match (match >= tau_match) and a clear miss (gate passed and match < tau_match) both read "
          "tau_match (`verify/stopping.py`): on a gate-passing view they are complements, so one value trades false "
          "clear misses where the target is present and visible against clear misses (and fewer false strong "
          "matches) where it was removed. Null set: diagnostic views of present episodes; power set: diagnostic "
          "views of removed episodes. Counts are `StoppingRule.is_clear_miss` / `is_strong_match` at the candidate "
          "thresholds. Ties go to the lower, more conservative tau_match.", "",
          "| tau_match | false clear-miss rate (x/n) | <= alpha | clear-miss rate on removed (x/n) "
          "| false strong-match rate on removed (x/n) | |", "| --- | --- | --- | --- | --- | --- |"]
    for p in m.curve:
        L.append(f"| {p.tau_match} | {_rate_xn(p.false_clear_miss)} | {'yes' if p.feasible else 'no'} "
                 f"| {_rate_xn(p.clear_miss)} | {_rate_xn(p.false_strong_match)} "
                 f"| {'**chosen**' if p.tau_match == m.tau_match else ''} |")
    L += ["", f"- chosen tau_match = {m.tau_match} ({m.n_feasible} of {len(m.curve)} values met alpha_clear = "
          f"{settings.alpha_clear:g} on the dev views)",
          "- Fail loudly: had no gate on the grid (1a), or no tau_match on the grid under the chosen gate (1b), met "
          f"alpha_clear, this run would have written no candidate, only `{INFEASIBLE_FILE}`, and exited with status "
          f"{EXIT_INFEASIBLE}; alpha_clear is never relaxed, the grid never widened and no fallback setting is "
          "used.", "",
          "### Per-view rates at the frozen thresholds, by lighting (resubstitution: the views that chose them)", ""]
    L += _view_table(res["view_rates"], conf, lights)
    L += ["", "Views of one arrangement are correlated, so the view-level CP bounds are optimistic.", ""]
    st1 = [w for w in res["warnings"] if w.startswith("Step 1")]
    if st1:
        L += ["### Step 1 warnings", "", *[f"- {w}" for w in st1], ""]
    L += ["## Step 3: grouped cross-validation of the calibration (leave one arrangement out)", "",
          "Each fold calibrates Steps 1a-1b on the other arrangements' views and scores the held-out arrangement's "
          "views; no policy runs. The pooled held-out rates are the honest per-view estimate. The held-out policy "
          "check of docs/SIM_PREREGISTRATION.md section 7 (every policy run on the held-out arrangement with its "
          "fold's thresholds) is not implemented yet, a known gap: the Step 4 policy rows are resubstitution.", ""]
    folds: list[ViewFold] = res["folds"]
    if folds:
        bad = [f for f in folds if f.infeasible is not None]
        if bad:
            L += [f"**INCOMPLETE: calibrating without {', '.join(f.held_out for f in bad)} is infeasible.** Those folds "
                  "have no thresholds and no held-out rates (no fallback setting), so the held-out estimate below "
                  f"covers {len(folds) - len(bad)} of {len(folds)} arrangements. A candidate whose grouped CV is "
                  "incomplete must not be frozen (docs/SIM_PREREGISTRATION.md section 7); it is not yet marked "
                  "`eligible_for_freeze: false` (a known gap).", ""]
        L += ["| held out | tau_q | tau_occ | tau_match | false clear (x/n) | clear (x/n) | false clear miss (x/n) "
              "| clear miss on removed (x/n) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for f in folds:
            if f.thresholds is None or f.held is None:
                L.append(f"| {f.held_out} | **infeasible**: {f.infeasible.summary() if f.infeasible else ''} "
                         "| - | - | - | - | - | - |")
                continue
            t, h = f.thresholds, f.held[POOLED]
            L.append(f"| {f.held_out} | {t.tau_q} | {t.tau_occ} | {t.tau_match} | {_xn(h.false_clear)} | {_xn(h.clear)} "
                     f"| {_xn(h.false_clear_miss)} | {_xn(h.clear_miss)} |")
        n_sets = len({f.thresholds for f in folds if f.thresholds is not None})
        L += ["", f"Calibrations differ across folds: {n_sets} distinct threshold sets over "
              f"{len(folds) - len(bad)} feasible folds.", "", "### Held-out per-view rates (honest estimate)", ""]
        L += _view_table(res["cv_held_out"], conf, [("L1", "L1 views"), ("L2", "L2 views"),
                                                     ("pooled", "L1 + L2 views")])
        st3 = [w for w in res["warnings"] if w.startswith("Step 3")]
        L += ["", *[f"- {w}" for w in st3], ""] if st3 else [""]
    else:
        L += ["- fewer than 2 dev arrangements: no grouped cross-validation. A candidate without a complete grouped "
              "CV must not be frozen (docs/SIM_PREREGISTRATION.md section 7; not yet marked in the candidate, a known "
              "gap).", ""]
    check: list[CheckRow] = res["check"]
    caps = "uncapped and under every cap" if settings.cap_steps else "uncapped only (no capped setting was run)"
    L += [f"## Step 4: dev check at the frozen thresholds (FRR <= {settings.frr_limit:g} is a check, not an "
          "objective)", "",
          f"Every policy at k = {th.k}, lam = {th.lam:g}, primary budget b = {settings.budget}, {caps} (caps are an "
          "operating condition reported as curves, never tuned). S is one view with its "
          "k = 1 rule; G is primary G (lookahead off), G-lookahead the same scorer with the lookahead; G-myopic "
          f"keeps its legacy weight ({GMYOPIC_LAM_DEG:g} deg). R is averaged over {settings.r_seeds} seeds within "
          "each episode, so it has no binomial bounds. A flag is reported and never fixed: changing a threshold to "
          "clear it would tune the thresholds on policy results.", ""]
    L += [f"- **FLAG** {f}" for f in flags]
    for light in REPORT_LIGHTINGS:   # FRR is undefined where a lighting has no present-target dev episode
        rows = [r for r in check if r.lighting == light]
        n = sum(r.over_limit is None for r in rows)
        if n:
            L.append(f"- **NOT CHECKED**: {light} queries: FRR is undefined in {n} of {len(rows)} rows (no "
                     f"present-target {light} dev episode), so the FRR limit was not checked there")
    if not flags and not unchecked:
        L.append(f"- no policy is above FRR {settings.frr_limit:g} on L1 or L2 dev queries at any setting")
    elif not flags:   # an incomplete check never reads as a pass (review: an L1-only dev set did)
        L.append(f"- no policy is above FRR {settings.frr_limit:g} where FRR is defined, but the check is incomplete "
                 "(`all_within_limit` is null): it says nothing about the rows not checked")
    for light in REPORT_LIGHTINGS:
        L += ["", f"### {light} queries", ""]
        L += _check_table([r for r in check if r.lighting == light], conf)
    L += ["", "## Step 5: sensitivity (reported only, never selected)", "",
          "The frozen taus with only k or lam replaced. Each cell is primary (k = 2, lam = 0) -> sensitivity.", ""]
    primary = {(r.policy, r.cap_steps, r.lighting): r for r in check}
    sens = res["sensitivity"]
    for k in settings.k_sensitivity:
        for light in REPORT_LIGHTINGS:
            L += [f"### k = {k} ({', '.join(K_SENSITIVITY_POLICIES)}), {light} queries", ""]
            L += _sens_table([r for r in sens["k"] if r.k == k and r.lighting == light], primary)
            L.append("")
    for lam in settings.lam_sensitivity:
        for light in REPORT_LIGHTINGS:
            L += [f"### lam = {lam:g} ({', '.join(LAM_SENSITIVITY_POLICIES)}), {light} queries", ""]
            L += _sens_table([r for r in sens["lam"] if r.lam == lam and r.lighting == light], primary)
            L.append("")
    for given, text in ((settings.k_sensitivity, _k_sensitivity_text(settings)),
                        (settings.lam_sensitivity, _lam_sensitivity_text(settings))):
        if not given:
            L += [f"- {text}", ""]
    lc: LamGridCheck = res["lam_grid_check"]
    L += ["## lam values check (geometry only; verify/lamgrid.py)", "",
          "Primary G's choice changes with lam only at these breakpoints on this `positions.json`: "
          + (", ".join(f"{b.lam:.4f}" for b in lc.breakpoints) or "none") + ". Values between the same two "
          "breakpoints give identical G choices in every decision state.", "",
          "- behaviour classes of lam = 0 and the sensitivity values: "
          + "; ".join("{" + ", ".join(f"{v:g}" for v in c) + "}" for c in lc.classes)]
    L += [f"- WARNING: {w}" for w in lc.warnings] or ["- every value is a distinct G behaviour, none on a breakpoint"]
    L += ["", f"## Secondary, not a headline: dev check pooled over L1 and L2 queries (b = {settings.budget})", "",
          "| policy | travel cap | FRR (x/n) | recall (x/n) | coverage (x/n) | mean travel cm |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in check:
        if r.lighting == POOLED:
            c = r.counts
            L.append(f"| {r.policy} | {_cap_cell(r)} | {_f(c.frr)} ({_num(c.n_false_removal)}/{c.n_present}) "
                     f"| {_f(c.recall)} ({_num(c.n_true_removal)}/{c.n_removed}) "
                     f"| {_f(c.coverage)} ({_num(c.n_resolved)}/{c.n_episodes}) | {_f(c.mean_travel_cm, 1)} |")
    L += ["", "## Reading these numbers", "",
          "- Views and episodes of one arrangement share an occluder and lighting, so they are not independent: "
          "view- and episode-level bounds are optimistic, and the arrangement-level FRR bound is the more honest one.",
          "- Quote the Step 3 held-out per-view rates, not the resubstitution rates of Step 1. The Step 4 check runs "
          "on the same dev arrangements whose views were calibrated, so it is mildly optimistic too, although no "
          "policy result influenced any threshold.",
          f"- alpha_clear = {settings.alpha_clear:g} constrains per-view rates on the dev views only: it does not "
          "guarantee that any policy's FRR is at most that value, and needing k = 2 clear-miss stops does not turn it "
          "into alpha squared. The FRR of each policy is only checked (Step 4).",
          frr_bound_note(len(dev.arrangements), settings.frr_limit, conf),
          "- select-v3 (replaced) chose (k, tau_match, lam) by running primary G on dev and maximising its coverage "
          "after a fixed-sequence FRR walk (D10). That tuned shared settings on G's own results, contrary to "
          "DECISIONS 2026-10-12, and would have handed G-tuned settings to ranker A.", ""]
    other = [w for w in res["warnings"] if not w.startswith(("Step 1", "Step 3")) and w not in lc.warnings]
    if other:
        L += ["### Other warnings", "", *[f"- {w}" for w in other], ""]
    if syn:
        L += [f"_{SYNTHETIC_LABEL}._", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- CLI

def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def _optional(parse: Callable[[str], tuple], name: str) -> Callable[[str], tuple]:
    """parse, with '' or 'none' -> (). name is what argparse calls the type in its error, "invalid <name> value: ..."
    (review: the bare inner function made it say "invalid f value")."""
    def f(text: str) -> tuple:
        return () if text.strip().lower() in ("", "none") else parse(text)
    f.__name__ = f.__qualname__ = name
    return f


def _cap_steps_list(text: str) -> tuple[float, ...]:
    """'1,2,3,4,6' -> finite caps in arc steps, sorted ('inf' is dropped: uncapped always runs); 'none' -> ()."""
    if text.strip().lower() in ("", "none"):
        return ()
    try:
        vals = {float(x) for x in text.split(",") if x.strip()}
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"cap steps must be numbers or 'inf', got {text!r}") from e
    if any(v != v or v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("cap steps must be > 0 (or 'inf')")
    return tuple(sorted(v for v in vals if not math.isinf(v)))


def build_parser() -> argparse.ArgumentParser:
    d = SelectionSettings()
    p = argparse.ArgumentParser(prog="verify.select", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="calibrate on the synthetic generator's DEV split")
    p.add_argument("--n-arr", type=int, default=30)
    p.add_argument("--n-dev", type=int, default=None)
    p.add_argument("--seed", type=int, default=0, help="generator / split / bootstrap seed")
    p.add_argument("--block-anywhere", action="store_true")
    p.add_argument("--budget", type=int, default=d.budget, help="primary budget of the dev check (default 3)")
    p.add_argument("--cap-steps", type=_cap_steps_list, default=d.cap_steps,
                   help="dev-check travel caps in arc steps, comma list or 'none' (default 1,2,3,4,6); the uncapped "
                        "setting always runs. Caps are an operating condition reported as curves, never tuned")
    p.add_argument("--r-seeds", type=int, default=d.r_seeds, help="R seeds per episode in the dev check (default 20)")
    p.add_argument("--tau-match-grid", type=_floats, default=d.tau_match_grid)
    p.add_argument("--tau-q-grid", type=_floats, default=d.tau_q_grid)
    p.add_argument("--tau-occ-grid", type=_floats, default=d.tau_occ_grid)
    p.add_argument("--alpha-clear", type=float, default=d.alpha_clear,
                   help="predeclared empirical per-view constraint on the dev views for both Step 1 false rates "
                        "(default 0.10); not a bound on any policy's FRR. If no gate (Step 1a), or no tau_match under "
                        f"the chosen gate (Step 1b), meets it, calibration fails: no candidate, {INFEASIBLE_FILE}, "
                        f"exit status {EXIT_INFEASIBLE}")
    p.add_argument("--frr-limit", type=float, default=d.frr_limit,
                   help="FRR limit of the dev check: a check that flags, never an objective (default 0.10)")
    p.add_argument("--k-sensitivity", type=_optional(_ints, "int list"), default=d.k_sensitivity,
                   help="k values run as a reported sensitivity only (default 3; 'none'); k = 2 is fixed")
    p.add_argument("--lam-sensitivity", type=_optional(_floats, "float list"), default=d.lam_sensitivity,
                   help="lam values run as a reported sensitivity only (default 0.7,0.9,1.2; 'none'); lam = 0 is fixed")
    p.add_argument("--bootstrap-B", type=int, default=d.bootstrap_B)
    p.add_argument("--positions", default=None, help="positions JSON (default configs/positions.json)")
    p.add_argument("--lam-breakpoints", action="store_true",
                   help="print primary G's lam breakpoints on the positions file, check lam = 0 and --lam-sensitivity "
                        "against them, and exit")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None, help="output dir (default runs/select-v4-<tag>/)")
    return p


def check_episodes(runner: PolicyRunner, th: Thresholds, settings: SelectionSettings) -> list[dict[str, Any]]:
    """Per-episode decisions of the Step 4 check (cached runs; for selection_details.json)."""
    out = []
    for p in CHECK_POLICIES:
        for cap in ((None,) if p == "S" else settings.cap_settings):
            budget = run_budget(p, cap, settings, runner.dev.positions)
            for r in runner.results(p, th, budget):
                out.append({"policy": p, "cap_steps": cap, "travel_cap_cm": budget.travel_cap_cm,
                            "episode_id": r.episode_id, "lighting": r.lighting.value,
                            "decision_frac": {d.value: r.frac(d) for d in Decision},
                            "mean_views": r.mean_views, "mean_travel_cm": r.mean_travel_cm})
    return out


def settings_sha256(settings: SelectionSettings, provenance: Mapping[str, Any] | None = None) -> str:
    hashed: dict[str, Any] = {"settings": settings.to_dict(), "procedure": PROCEDURE_VERSION}
    if provenance is not None:   # synthetic runs: the generator regime is part of what was calibrated on (F4)
        hashed["synthetic_provenance"] = to_jsonable(dict(provenance))
    return config_sha256(hashed)


def _write_atomic(path: Path, text: str) -> None:
    """Write text to a temp file next to path, then os.replace it over path: a reader never sees half a file."""
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_outputs(out: Path, dev: DevData, settings: SelectionSettings, res: dict[str, Any],
                  provenance: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Write the candidate, its .sha256, the report and the details. Every text is rendered before anything is
    written, so a failure while rendering leaves out as it was. Then every file an earlier run left (including a
    stale calibration_infeasible.json) is removed, and each file is written as a temp file plus os.replace, with
    frozen_candidate.json last: a failure while writing leaves no candidate, never a new candidate next to another
    run's report or details (review: the candidate and its .sha256 used to be written first)."""
    sha_settings = settings_sha256(settings, provenance)
    cand = candidate_doc(dev, settings, res, sha_settings, provenance)
    text = json.dumps(cand, indent=1, sort_keys=True) + "\n"
    sha = hashlib.sha256(text.encode()).hexdigest()
    cal: Calibration = res["calibration"]
    details = {"label": SYNTHETIC_LABEL if dev.synthetic else "", "procedure_version": PROCEDURE_VERSION,
               "settings": settings.to_dict(),
               "gate_grid": [{"tau_q": tq, "tau_occ": to, "false_clear": fc, "clear": cl,
                              "feasible": bool(fc[1]) and fc[0] / fc[1] <= settings.alpha_clear + EPS}
                             for tq, to, fc, cl in cal.gate.grid],
               "tau_match_curve": [p.to_dict() for p in cal.match.curve],
               "folds": [f.to_dict() for f in res["folds"]],
               "check_episodes": check_episodes(res["runner"], res["thresholds"], settings)}
    texts = {"selection_details.json": json.dumps(to_jsonable(details), indent=1, sort_keys=True) + "\n",
             REPORT_FILE: render_report(dev, settings, res, cand, sha),
             "frozen_candidate.json.sha256": f"{sha}  frozen_candidate.json\n",
             "frozen_candidate.json": text}   # written in this order: the candidate last
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (out / name).unlink(missing_ok=True)
    for name, t in texts.items():
        _write_atomic(out / name, t)
    return {"frozen_candidate.json": sha}


def write_infeasible(out: Path, dev: DevData, settings: SelectionSettings, failure: Infeasibility,
                     provenance: Mapping[str, Any] | None = None) -> Path:
    """Step 1 failed: first remove any frozen candidate (with its .sha256 and details) and the report an earlier run
    left in out, so nothing downstream can pick a candidate up for this dev data even if what follows fails. Then
    write the report and calibration_infeasible.json (last), each as a temp file plus os.replace. Returns the path."""
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (out / name).unlink(missing_ok=True)
    sha_settings = settings_sha256(settings, provenance)
    doc: dict[str, Any] = {
        "schema": INFEASIBLE_SCHEMA, "status": "infeasible", "procedure_version": PROCEDURE_VERSION,
        "synthetic": dev.synthetic, "label": SYNTHETIC_LABEL if dev.synthetic else "",
        "no_candidate": (f"no frozen_candidate.json was written and any earlier one in this directory was removed; "
                         f"{NO_FALLBACK}"),
        "failure": failure.to_dict(),
        "alpha_clear": settings.alpha_clear, "alpha_note": ALPHA_NOTE,
        "settings": settings.to_dict(), "settings_sha256": sha_settings, "proposed_defaults_note": PROPOSED,
        "dev_arrangements": list(dev.arrangements), "n_dev_episodes": len(dev.specs),
        "dev_evidence_sha256": dev.evidence_sha256, "view_labels_sha256": dev.view_labels_sha256,
        "view_label_source": dev.label_source,
    }
    if provenance is not None:
        doc["synthetic_provenance"] = dict(provenance)
    record = json.dumps(to_jsonable(doc), indent=1, sort_keys=True) + "\n"
    report = render_infeasible_report(dev, settings, failure, sha_settings)
    path = out / INFEASIBLE_FILE
    _write_atomic(out / REPORT_FILE, report)
    _write_atomic(path, record)
    return path


def render_infeasible_report(dev: DevData, settings: SelectionSettings, failure: Infeasibility,
                             sha_settings: str) -> str:
    L: list[str] = []
    if dev.synthetic:
        L += [f"# {SYNTHETIC_LABEL}", "",
              "**SYNTHETIC - not study results and not threshold values.** This run only exercises the calibration "
              "procedure.", ""]
    # Step 1b searched tau_match under the one gate Step 1a chose, so its failure must not read as "no setting on the
    # predeclared grid" (review; the next checkpoint searches gate and tau_match jointly).
    sequential = (["**Step 1b searches tau_match only under the gate Step 1a picked**, so other gate and tau_match "
                   "settings on the grid were never tried together and may still meet both alpha constraints (a "
                   "known gap: the next checkpoint replaces the sequence with a joint search that maximises correct "
                   "clear misses subject to both constraints, with a declared conservative tie-break).", ""]
                  if failure.step == "1b" else [])
    what, under = failure.searched()
    L += ["# Threshold calibration report (D-select, select-v4): INFEASIBLE", "",
          f"**No frozen candidate was written.** Step {failure.step} found {what} that meets "
          f"alpha_clear = {failure.alpha_clear:g}{under}, and {NO_FALLBACK}. Any earlier "
          f"`frozen_candidate.json` in this directory was removed. The failure record is `{INFEASIBLE_FILE}` (status "
          f"\"infeasible\"); the CLI exits with status {EXIT_INFEASIBLE}.", "", *sequential,
          f"- procedure `{PROCEDURE_VERSION}`; settings sha256 `{sha_settings}`",
          f"- dev: {len(dev.arrangements)} arrangements, {len(dev.specs)} episodes; evidence sha256 "
          f"`{dev.evidence_sha256}`; view labels: {dev.label_source}; labels sha256 `{dev.view_labels_sha256}`",
          f"- constraint that failed (Step {failure.step}): {failure.constraint}",
          f"- views it is measured on: {failure.n_null_views} (dev views of both lightings)",
          "- best achievable rate among the settings searched: " + (
              "none (no view to measure it on)" if failure.best_rate is None or failure.best_counts is None else
              f"{failure.best_rate:.4f} ({failure.best_counts[0]}/{failure.best_counts[1]}), at "
              + "; ".join(", ".join(f"{k} {v:g}" for k, v in st.items()) for st in failure.best_settings)),
          "- grid searched: " + "; ".join(f"{k} {_grid_text(v)}" for k, v in failure.grid.items()),
          f"- alpha_clear = {failure.alpha_clear:g}: {ALPHA_NOTE}", "",
          "Changing alpha_clear or a grid is a change to the predeclared settings, for the student to decide and "
          "record; this procedure never does it.", "",
          "## Every setting searched" + (" (tau_match, under the Step 1a gate)" if failure.step == "1b" else ""), "",
          "| setting | false rate (x/n) |", "| --- | --- |"]
    for g in failure.grid_rates:
        name = ", ".join(f"{k} {v:g}" for k, v in g.items() if k not in ("x", "n", "rate"))
        L.append(f"| {name} | {_f(g['rate'], 4)} ({g['x']}/{g['n']}) |")
    if dev.synthetic:
        L += ["", f"_{SYNTHETIC_LABEL}._"]
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.lam_breakpoints:
        pos = Positions.load(args.positions) if args.positions else Positions.load()
        chk = check_lam_grid((LAM_FIXED, *args.lam_sensitivity), pos, lookahead=False)   # primary G
        print("G (primary, lookahead off) lam breakpoints (geometry only): "
              + ", ".join(f"{b.lam:.6f} ({b.n_states_changed} states)" for b in chk.breakpoints))
        print("behaviour classes of lam = 0 and the sensitivity values: "
              + "; ".join(str(list(c)) for c in chk.classes))
        for w in chk.warnings:
            print("WARNING: " + w)
        return 0
    if not args.synthetic:
        print("real dev data not available yet (use --synthetic); build a DevData and call run_selection",
              file=sys.stderr)
        return 2
    try:
        settings = SelectionSettings(budget=args.budget, cap_steps=args.cap_steps, r_seeds=args.r_seeds,
                                     tau_match_grid=args.tau_match_grid, tau_q_grid=args.tau_q_grid,
                                     tau_occ_grid=args.tau_occ_grid, alpha_clear=args.alpha_clear,
                                     frr_limit=args.frr_limit, k_sensitivity=args.k_sensitivity,
                                     lam_sensitivity=args.lam_sensitivity, bootstrap_B=args.bootstrap_B,
                                     seed=args.seed)
    except ValueError as e:   # an invalid setting is a usage error (exit 2), not a crash (review: traceback, exit 1)
        parser.error(str(e))
    positions = Positions.load(args.positions) if args.positions else Positions.load()
    params = SyntheticParams(n_dev=args.n_dev, block_includes_start=not args.block_anywhere)
    dev = synthetic_dev_data(args.n_arr, args.seed, params, positions)
    context = RunContext(model_id="synthetic-no-model", config_version=PROCEDURE_VERSION,
                         config_sha256=config_sha256(settings.to_dict()), synthetic=True)
    tag = args.tag or f"synthetic-n{args.n_arr}-s{args.seed}"
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"{PROCEDURE_VERSION}-{tag}"
    provenance = {"generator": dataclasses.asdict(params), "n_arr": args.n_arr, "n_dev": len(dev.arrangements),
                  "seed": args.seed}
    try:
        res = run_selection(dev, settings, context)
    except CalibrationInfeasible as e:   # fail loudly: no candidate, a failure record, a non-zero exit
        path = write_infeasible(out, dev, settings, e.failure, provenance)
        print(f"{SYNTHETIC_LABEL}: CALIBRATION INFEASIBLE, no frozen candidate written. {e}", file=sys.stderr)
        print(f"{SYNTHETIC_LABEL}: failure record {path}; report {out / REPORT_FILE}", file=sys.stderr)
        return EXIT_INFEASIBLE
    shas = write_outputs(out, dev, settings, res, provenance)
    th = res["thresholds"]
    print(f"{SYNTHETIC_LABEL}: wrote {out} (frozen_candidate.json sha256 {shas['frozen_candidate.json']})")
    print(f"{SYNTHETIC_LABEL}: candidate tau_q {th.tau_q}, tau_occ {th.tau_occ}, tau_match {th.tau_match}, "
          f"k {th.k} (fixed), lam {th.lam:g} (fixed); dev check: {len(res['flags'])} flag(s) above FRR "
          f"{settings.frr_limit:g}" + (f", {len(res['unchecked'])} row(s) not checked (FRR undefined)"
                                       if res["unchecked"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
