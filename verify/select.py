"""Dev-only threshold calibration (D-select), procedure select-v4 (approved by the student 2026-11-02; the joint
search, the held-out policy check and freeze eligibility 2026-11-03, before any test data and before the freeze).

    uv run --no-sync python -m verify.select --synthetic [--n-arr 30] [--seed 0] [--block-anywhere]
        [--budget 3] [--cap-steps 1,2,3,4,6 | none] [--r-seeds 20] [--tau-match-grid 0.40,...]
        [--tau-q-grid ...] [--tau-occ-grid ...] [--alpha-clear 0.10] [--frr-limit 0.10]
        [--k-sensitivity 3 | none] [--lam-sensitivity 0.7,0.9,1.2 | none] [--bootstrap-B 2000]
        [--tag TAG] [--out runs/select-v4-<tag>/]
    uv run --no-sync python -m verify.select --manifest SET/manifest.json --evidence SET/evidence_refL1.jsonl
        --view-labels SET/view_labels.json --positions configs/sim/positions_shelf_A.json [the same options]
        (sim dev data: the set files sim/episodes.py writes, with evidence from perception/evidence.py run on
        jobs_refL1.json; only dev arrangements are accepted)
    uv run --no-sync python -m verify.select --lam-breakpoints [--positions configs/positions.json]
        [--lam-sensitivity ...]
        (prints the lam values at which primary G's choice changes on that geometry, and checks lam = 0
        and the sensitivity values against them)

Exit status: 0 candidate written and eligible for freeze; 2 usage or input error; 3 calibration infeasible on all
dev data (no candidate, calibration_infeasible.json); 4 candidate written but NOT eligible for freeze (a planned
grouped-CV fold is infeasible, or there are fewer than two dev arrangements).

No compared policy's results set any shared setting (DECISIONS 2026-10-12: "Shared thresholds are
selected by a documented procedure that does not optimise them on G's results"). The thresholds come
from per-view labels alone, k and lam are fixed in advance, and every policy (S, F, R, G, G-lookahead,
G-myopic; A later) then runs with the same frozen settings.

Procedure select-v4. Every number it produces on --synthetic data is SYNTHETIC and is never a real
threshold; every number from a manifest with "simulation": true is a SIMULATION result.

  Step 1  Per-view calibration on the dev views of BOTH lightings together: one frozen threshold set
          serves L1 and L2 queries, and its per-view rates are reported separately for each. No policy
          is run. Every per-view criterion is the shared StoppingRule's own predicate (passes_gate,
          is_clear_miss, is_strong_match), never a copy of it.
          ONE Neyman-Pearson search over the whole predeclared grid tau_q x tau_occ x tau_match: maximise
          correct clear misses (is_clear_miss on diagnostic views of removed episodes) subject to BOTH
            the false-clear rate (passes_gate on non-diagnostic views of occluded-present episodes) <= alpha_clear
            the false clear-miss rate (is_clear_miss on diagnostic views of present episodes) <= alpha_clear.
          Declared total-order tie-break toward the conservative setting: lower false clear-miss rate, then
          lower false-clear rate, then lower tau_match, then higher tau_q, then lower tau_occ. The candidate
          records whether the tie-break decided the choice and at which level.
          (Procedure revision 2, 2026-11-03, found in review: the intent is unchanged, the implemented method
          changed. Revision 1 chose the gate first (1a) and then tau_match under that one gate (1b). 1b could fail "infeasible" while hundreds of grid settings met both constraints,
          e.g. L2 queries against the L1 reference, the sim study's primary setting.)
    Curves. For each calibrated threshold, the correct and the false clear-miss curves over its grid (the other
          two held at the chosen values), with counts, the chosen point marked, "flat" where a curve does not
          move and whether the tie-break decided along it. A flat curve is reported as flat; no variation is forced.
    Fail loudly. If no grid setting meets both constraints (or there is no view to measure a false rate on),
          calibration raises CalibrationInfeasible: which constraint binds, the best achievable rates and the grid.
          There is then no frozen candidate: the CLI removes any earlier frozen_candidate.json from its output
          directory, writes calibration_infeasible.json and exits with status 3. alpha_clear is never relaxed, the
          grid is never widened, and no "least bad" setting is used.
  Step 2  Fixed a priori, never selected: k = 2 (the plan's "at least two distinct positions") and
          lam = 0 (G and A choose by spread or informativeness alone; travel is limited by the caps,
          which are reported as curves).
  Step 3  Grouped cross-validation, leaving one arrangement out (all its views, conditions and lightings
          together). Every fold calibrates Step 1 without the held-out arrangement (the same joint search).
          (a) Per-view: the held-out arrangement's per-view rates at the fold's thresholds.
          (b) Held-out policy check: every policy (S, F, R, G, G-lookahead, G-myopic, FULL and the -fcm variants
              verify.run reports, plus any fold-trained policy, e.g. A later) runs on the held-out arrangement's
              episodes with THAT FOLD'S thresholds, at the primary budget, uncapped and under every cap. A fold-trained
              policy is built fresh per fold by its factory, in that fold's own process, and trained on a DevData
              holding the training arrangements only (a snapshot with no path back to the full dev data), so its
              training and model selection exclude the held-out arrangement by construction, and nothing it learned
              in one fold reaches another. Rates per policy and lighting
              (pooled only as a secondary line): FRR per episode and per arrangement, recall, coverage, mean views,
              mean travel, with counts and the stratified cluster bootstrap.
          A fold whose calibration is infeasible gets no thresholds, no held-out rates and no held-out runs (no
          fallback); it is reported.
    Freeze eligibility. A candidate is eligible only if calibration succeeds on all dev data AND in every planned CV
          fold (docs/SIM_PREREGISTRATION.md section 7). Otherwise the candidate is still written for investigation,
          marked "eligible_for_freeze": false with ineligible_reasons, and the CLI exits with status 4. Held-out
          arrangements or policies missing alpha / the FRR limit do not make a candidate ineligible: those are
          reported checks. require_freeze_eligible(path) is what the freeze step calls; it takes the candidate file
          only (checked against its .sha256 sidecar) and refuses an ineligible, unmarked, synthetic (by its flag,
          label or synthetic_provenance) or tampered candidate.
  Step 4  FRR <= frr_limit is a CHECK, not an objective. With the frozen thresholds (k = 2, lam = 0)
          every policy (S, F, R, G, G-lookahead, G-myopic) runs on dev at the primary budget, uncapped
          and under every cap of the grid. FRR (per episode and per arrangement), recall and coverage are reported
          separately for L1 and L2 queries, with one-sided Clopper-Pearson and Jeffreys bounds and a cluster
          bootstrap over arrangements, stratified by quota slot class (occluder family, M16 split by side) when
          the manifest has it and every class holds at least two of the arrangements resampled, else unstratified
          and flagged ("unstratified: singleton strata" when a class has one). A policy above the limit is flagged;
          nothing is changed to fix it. A row with no present-target episode (FRR undefined) is not checked, and the
          check is then incomplete (all_within_limit null), never passed.
  Step 5  Sensitivity, reported and never selected: k = 3 (F, R, G, G-lookahead, G-myopic) and lam in
          {0.7, 0.9, 1.2} (G and G-lookahead, the only policies that read lam), with the frozen taus.

alpha_clear (0.10 by default) is a predeclared, EMPIRICAL per-view calibration constraint on the dev
views: one value for both Step 1 false rates, measured on the dev views of both lightings together. It
is not a bound on any policy's false removal rate, which Step 4 only checks, and requiring k = 2
clear-miss stops does not turn it into alpha squared.

Order. Steps 1-3 calibrate on views only: every threshold set (the candidate and every fold's) is fixed before the
first policy runs. Step 4 and Step 5 hand the policies nothing but the frozen Thresholds (or a copy with k or lam
replaced); the held-out check hands each fold's policies that fold's Thresholds. No run can reach a threshold.

Replaced: select-v3 (until 2026-11-02). It picked (k, tau_match, lam) by RUNNING primary G on dev and
maximising its coverage, contrary to DECISIONS 2026-10-12.

Data access. This module is offline dev analysis, not a policy. It reads every dev view for Step 1
(from a fresh unsealed store) and joins dev ground truth to score Steps 3b-5. Policies still see only
requested views: every run goes through the public ReplayEnv / run_episode API on a fresh
EvidenceStore from a StoreFactory (one store per run setting; keys are minted before sealing). Only
DEV-split arrangements are accepted; any test (or pilot) arrangement raises SplitRefused. The geometric view
labels are calibration and scoring inputs only, never a policy input.

Known gaps (docs/SIM_PREREGISTRATION.md; verify/README.md section 10): ranker A is not built (the held-out check
has the per-fold training hook it will plug into); results are not yet split by environment; verify.run still
replays with placeholder thresholds, not a frozen candidate.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import logging
import math
import multiprocessing
import os
import sys
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from analysis.bootstrap import arrangement_draws, resolve_strata, stratified_arrangement_draws
from analysis.intervals import (
    clopper_pearson_lower,
    clopper_pearson_upper,
    jeffreys_lower,
    jeffreys_upper,
)
from verify.lamgrid import LamGridCheck, check_lam_grid
from verify.manifest import (
    SIM_VIEW_LABELS_SCHEMA,
    SIMULATION_LABEL,
    UNSTRATIFIED_NOTE,
    episodes_from_arrangements,
    load_geometric_labels,
    load_manifest_info,
)
from verify.policies import (
    DEFAULT_R_SEEDS,
    FULL_SWEEP_VIEWS,
    GMYOPIC_LAM_DEG,
    F,
    FullSweep,
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
    file_sha256,
    to_jsonable,
    truth_from_condition,
)
from verify.stopping import StoppingRule
from verify.synthetic import SyntheticParams, generate, generate_layouts

log = logging.getLogger("verify.select")

PROCEDURE_VERSION = "select-v4"   # v4 (2026-11-02): per-view calibration only; k = 2 and lam = 0 fixed; FRR a check
# 2026-11-03 corrections within select-v4 (before any test data and before the freeze): joint gate + tau_match search,
# held-out policy check, freeze eligibility. v3: G is primary G (lookahead off); v2: lam grid from breakpoints (F1, F4)
# The procedure revision (2026-11-03): the intent of select-v4 is unchanged, the implemented method changed (joint
# search, the joint search repeated in every CV fold, held-out policy check, freeze eligibility). Recorded in the
# candidate, the infeasible record and the settings hash, so a revision-1 candidate can never pass for this one.
PROCEDURE_REVISION = "select-v4 revision 2 (2026-11-03): joint gate + tau_match search"
CANDIDATE_SCHEMA = "cmpe249.frozen_candidate.v3"   # v3: joint search, curves, held-out check, eligible_for_freeze
VIEW_LABELS_SCHEMA = "cmpe249.view_labels.v1"
PROPOSED = "proposed default - confirm"
SYNTHETIC_LABEL_SOURCE = ("SYNTHETIC stand-in: derived from the generator's hidden occluder layout "
                          "(blocked or edge stop -> non-diagnostic), not from human annotation")
SIM_LABEL_SOURCE = (f"geometric labels from the renderer's instance masks ({SIM_VIEW_LABELS_SCHEMA}): a present view "
                    "is diagnostic at visible fraction >= 0.90, a removed view when the occluder covers < 10 % of the "
                    "target's region; never a policy input")

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
# The held-out policy check runs every policy verify.run reports (run.ORDER): the k-rule policies, their -fcm variants
# (first-clear-miss rule, k = 1) and FULL (all stops, uncapped).
HELDOUT_POLICIES = ("S", "F", "R", "G", "G-lookahead", "G-myopic", "F-fcm", "R-fcm", "G-fcm", "G-lookahead-fcm",
                    "G-myopic-fcm", "FULL")
BUILTIN_POLICIES = ("S", "F", "R", "G", "G-lookahead", "G-myopic", "FULL")
K_SENSITIVITY_POLICIES = ("F", "R", "G", "G-lookahead", "G-myopic")   # S decides with k = 1 by construction
LAM_SENSITIVITY_POLICIES = ("G", "G-lookahead")   # the only policies that read Thresholds.lam (G-myopic: lam_deg)
REPORT_LIGHTINGS = tuple(x.value for x in Lighting)   # "L1", "L2": the headline split (DECISIONS 2026-11-02)
POOLED = "pooled"   # both lightings: the set Step 1 calibrates on; for policy results a labelled secondary line
NOT_BINOMIAL = ("seed-averaged decisions (R) are not binomial counts: no Clopper-Pearson / Jeffreys bound; "
                "see the cluster bootstrap")
ALPHA_NOTE = ("alpha_clear is a predeclared, empirical per-view calibration constraint on the dev views: the chosen "
              "thresholds must keep the false-clear rate and the false clear-miss rate, both measured on the dev "
              "views of both lightings together, at or below it. It is not a bound on any policy's false removal "
              "rate (FRR is only checked, Step 4), and requiring k = 2 clear-miss stops does not turn it into alpha "
              "squared.")
GATE_CONSTRAINT = ("false-clear rate (StoppingRule.passes_gate on non-diagnostic views of occluded-present dev "
                   "episodes) <= alpha_clear")
MATCH_CONSTRAINT = ("false clear-miss rate (StoppingRule.is_clear_miss on diagnostic views of present dev episodes) "
                    "<= alpha_clear")
JOINT_OBJECTIVE = ("maximise correct clear misses: StoppingRule.is_clear_miss on diagnostic views of removed dev "
                   "episodes")
TIE_BREAK = ("lower false clear-miss rate, then lower false-clear rate, then lower tau_match, then higher tau_q, then "
             "lower tau_occ")
FULL_ORDER = f"maximise correct clear misses, then {TIE_BREAK}"   # the complete total order of the joint search
TIE_LEVELS = ("false clear-miss rate", "false-clear rate", "tau_match (lower)", "tau_q (higher)", "tau_occ (lower)")
JOINT_SEARCH = ("one joint Neyman-Pearson search over the whole predeclared grid tau_q x tau_occ x tau_match "
                "(select-v4 correction found in review, 2026-11-03, before any test data and before the freeze: the "
                "gate was chosen first and tau_match only under that one gate, so calibration could fail while other "
                "grid settings met both constraints)")
NO_FALLBACK = "alpha_clear was not relaxed, the grid was not widened and no fallback setting was used"
ELIGIBILITY_RULE = ("eligible only if calibration succeeds on all dev data and in every planned grouped-CV fold "
                    "(docs/SIM_PREREGISTRATION.md section 7). Held-out arrangements or policies missing alpha_clear or "
                    "the FRR limit do not make a candidate ineligible: those are reported checks.")
NO_DIAG_CONTEXT = ("context for coverage, not a ceiling: a policy can resolve an episode with no diagnostic view, and "
                   "one with diagnostic views can stay unknown")
INFEASIBLE_FILE = "calibration_infeasible.json"
INFEASIBLE_SCHEMA = "cmpe249.calibration_infeasible.v2"   # v2: joint search (binding constraint, best rates)
EXIT_INFEASIBLE = 3   # CLI exit status when Step 1 is infeasible on all dev data (2 = usage or input error)
EXIT_INELIGIBLE = 4   # candidate written, but not eligible for freeze
CANDIDATE_FILES = ("frozen_candidate.json", "frozen_candidate.json.sha256", "selection_details.json")
REPORT_FILE = "selection_report.md"
OUTPUT_FILES = (*CANDIDATE_FILES, REPORT_FILE, INFEASIBLE_FILE)   # everything a run writes into its output dir


class SplitRefused(ManifestError):
    """Selection was handed a non-dev (test or pilot) episode."""


def _ratio(xn: tuple[float, int]) -> float | None:
    return xn[0] / xn[1] if xn[1] else None


def _meets(xn: tuple[int, int], alpha: float) -> bool:
    """x / n <= alpha (EPS tolerance), and measurable (n > 0)."""
    return bool(xn[1]) and xn[0] / xn[1] <= alpha + EPS


def _grid_text(v: Any) -> str:
    if isinstance(v, Mapping):
        return "(" + ", ".join(f"{k} {x:g}" for k, x in v.items()) + ")"
    return "[" + ", ".join(f"{x:g}" for x in v) + "]"


def _setting_text(st: Mapping[str, float]) -> str:
    return ", ".join(f"{k} {v:g}" for k, v in st.items())


# ---------------------------------------------------------------- Step 1: the joint grid

@dataclass(frozen=True)
class GridPoint:
    """Per-view counts (x, n) at one grid setting, each from the rule's own predicate:

    false_clear         passes_gate on non-diagnostic views of occluded-present episodes (constraint 1)
    clear               passes_gate on diagnostic views of removed episodes
    false_clear_miss    is_clear_miss on diagnostic views of present episodes (constraint 2)
    clear_miss          is_clear_miss on diagnostic views of removed episodes (the objective)
    false_strong_match  is_strong_match on diagnostic views of removed episodes (each one decides PRESENT)
    """
    tau_q: float
    tau_occ: float
    tau_match: float
    false_clear: tuple[int, int]
    clear: tuple[int, int]
    false_clear_miss: tuple[int, int]
    clear_miss: tuple[int, int]
    false_strong_match: tuple[int, int]

    @property
    def setting(self) -> dict[str, float]:
        return {"tau_q": self.tau_q, "tau_occ": self.tau_occ, "tau_match": self.tau_match}

    def gate_ok(self, alpha: float) -> bool:
        return _meets(self.false_clear, alpha)

    def match_ok(self, alpha: float) -> bool:
        return _meets(self.false_clear_miss, alpha)

    def feasible(self, alpha: float) -> bool:
        return self.gate_ok(alpha) and self.match_ok(alpha)

    def order_key(self) -> tuple:
        """The declared total order, best first: max correct clear misses, then the tie-break (TIE_BREAK). Counts
        compare like rates because every point of one calibration shares the denominators."""
        return (-self.clear_miss[0], self.false_clear_miss[0], self.false_clear[0], self.tau_match, -self.tau_q,
                self.tau_occ)

    def to_dict(self, alpha: float | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {**self.setting}
        for name in ("false_clear", "clear", "false_clear_miss", "clear_miss", "false_strong_match"):
            xn = getattr(self, name)
            d[name] = {"x": xn[0], "n": xn[1], "rate": _ratio(xn)}
        if alpha is not None:
            d.update(meets_false_clear=self.gate_ok(alpha), meets_false_clear_miss=self.match_ok(alpha),
                     feasible=self.feasible(alpha))
        return d


class ViewTable:
    """Every per-view predicate at every grid setting, counted per arrangement once (with the rule's own
    passes_gate / is_clear_miss / is_strong_match), so a calibration on any set of arrangements, the whole dev set
    or one CV fold, is a sum over those arrangements. It reads views and labels only."""

    def __init__(self, rows: Sequence[ViewRow], settings: SelectionSettings) -> None:
        self.tau_q_grid, self.tau_occ_grid = tuple(settings.tau_q_grid), tuple(settings.tau_occ_grid)
        self.tau_match_grid = tuple(settings.tau_match_grid)
        self.arrangements = tuple(sorted({r.arrangement_id for r in rows}))
        idx = {a: i for i, a in enumerate(self.arrangements)}
        A, Q, O, T = len(self.arrangements), len(self.tau_q_grid), len(self.tau_occ_grid), len(self.tau_match_grid)
        null_g, pres, rem = gate_null_views(rows), present_diagnostic_views(rows), removed_diagnostic_views(rows)
        self.n_gate_null = np.zeros(A, dtype=np.int64)
        self.n_present_diag = np.zeros(A, dtype=np.int64)
        self.n_removed_diag = np.zeros(A, dtype=np.int64)
        for pop, n in ((null_g, self.n_gate_null), (pres, self.n_present_diag), (rem, self.n_removed_diag)):
            for r in pop:
                n[idx[r.arrangement_id]] += 1
        self.fc = np.zeros((A, Q, O), dtype=np.int64)
        self.clear = np.zeros((A, Q, O), dtype=np.int64)
        self.fcm = np.zeros((A, Q, O, T), dtype=np.int64)
        self.cm = np.zeros((A, Q, O, T), dtype=np.int64)
        self.fsm = np.zeros((A, T), dtype=np.int64)
        for qi, tq in enumerate(self.tau_q_grid):
            for oi, to in enumerate(self.tau_occ_grid):
                gate = StoppingRule(frozen_thresholds(Thresholds().tau_match, tq, to))   # the gate ignores tau_match
                for r in null_g:
                    self.fc[idx[r.arrangement_id], qi, oi] += gate.passes_gate(r.obs)
                for r in rem:
                    self.clear[idx[r.arrangement_id], qi, oi] += gate.passes_gate(r.obs)
                for ti, tm in enumerate(self.tau_match_grid):
                    rule = StoppingRule(frozen_thresholds(tm, tq, to))
                    for r in pres:
                        self.fcm[idx[r.arrangement_id], qi, oi, ti] += rule.is_clear_miss(r.obs)
                    for r in rem:
                        self.cm[idx[r.arrangement_id], qi, oi, ti] += rule.is_clear_miss(r.obs)
        for ti, tm in enumerate(self.tau_match_grid):   # a strong match reads tau_match only
            rule = StoppingRule(frozen_thresholds(tm, self.tau_q_grid[0], self.tau_occ_grid[0]))
            for r in rem:
                self.fsm[idx[r.arrangement_id], ti] += rule.is_strong_match(r.obs)

    @property
    def grid(self) -> dict[str, list[float]]:
        return {"tau_q_grid": list(self.tau_q_grid), "tau_occ_grid": list(self.tau_occ_grid),
                "tau_match_grid": list(self.tau_match_grid)}

    def points(self, arrangements: Iterable[str] | None = None) -> list[GridPoint]:
        """Every grid setting's counts summed over the given arrangements (default all), in grid order
        (tau_q, tau_occ, tau_match)."""
        keep = self.arrangements if arrangements is None else tuple(sorted(set(arrangements)))
        unknown = set(keep) - set(self.arrangements)
        if unknown:
            raise ValueError(f"arrangements not in the view table: {sorted(unknown)[:5]}")
        m = np.array([a in set(keep) for a in self.arrangements], dtype=bool)
        n_null, n_pres, n_rem = (int(v[m].sum()) for v in (self.n_gate_null, self.n_present_diag, self.n_removed_diag))
        fc, clear, fcm, cm, fsm = (v[m].sum(axis=0) for v in (self.fc, self.clear, self.fcm, self.cm, self.fsm))
        out = []
        for qi, tq in enumerate(self.tau_q_grid):
            for oi, to in enumerate(self.tau_occ_grid):
                for ti, tm in enumerate(self.tau_match_grid):
                    out.append(GridPoint(tq, to, tm, (int(fc[qi, oi]), n_null), (int(clear[qi, oi]), n_rem),
                                         (int(fcm[qi, oi, ti]), n_pres), (int(cm[qi, oi, ti]), n_rem),
                                         (int(fsm[ti]), n_rem)))
        return out


def _best(points: Sequence[GridPoint], name: str) -> dict[str, Any] | None:
    """The lowest false rate `name` among points (with every setting that reaches it); None if not measurable."""
    rated = [(p, _ratio(getattr(p, name))) for p in points]
    rated = [(p, r) for p, r in rated if r is not None]
    if not rated:
        return None
    best = min(r for _, r in rated)
    at = [p for p, r in rated if r == best]
    x, n = getattr(at[0], name)
    return {"rate": best, "x": x, "n": n, "settings": [p.setting for p in at]}


@dataclass(frozen=True)
class Infeasibility:
    """Why the joint search failed: no setting on the grid meets both constraints. binding says which constraint
    binds; the best achievable rates are given over the whole grid and within each constraint's feasible set.
    Nothing was relaxed, widened or substituted."""
    binding: str                                   # "false-clear", "false clear-miss", "both", "jointly"
    binding_text: str
    alpha_clear: float
    n_gate_null_views: int
    n_match_null_views: int
    best_false_clear: Mapping[str, Any] | None                 # over the whole grid
    best_false_clear_miss: Mapping[str, Any] | None
    best_false_clear_miss_meeting_false_clear: Mapping[str, Any] | None   # among settings meeting constraint 1
    best_false_clear_meeting_false_clear_miss: Mapping[str, Any] | None   # among settings meeting constraint 2
    grid: Mapping[str, Any]
    grid_rates: tuple[Mapping[str, Any], ...]      # every setting with both false rates
    arrangements: tuple[str, ...]

    @staticmethod
    def of(points: Sequence[GridPoint], alpha: float, grid: Mapping[str, Any],
           arrangements: Iterable[str]) -> Infeasibility:
        gate_any = any(p.gate_ok(alpha) for p in points)
        match_any = any(p.match_ok(alpha) for p in points)
        n_null = points[0].false_clear[1] if points else 0
        n_pres = points[0].false_clear_miss[1] if points else 0
        if not gate_any and not match_any:
            binding, text = "both", ("both constraints: no grid setting meets the false-clear constraint and none "
                                     "meets the false clear-miss constraint, each on its own")
        elif not gate_any:
            binding, text = "false-clear", f"the false-clear constraint: no grid setting meets '{GATE_CONSTRAINT}'"
        elif not match_any:
            binding, text = ("false clear-miss", "the false clear-miss constraint: no grid setting meets "
                             f"'{MATCH_CONSTRAINT}'")
        else:
            binding, text = "jointly", ("the two constraints jointly: grid settings meet each constraint on its own, "
                                        "but none meets both")
        for what, n in (("false-clear", n_null), ("false clear-miss", n_pres)):
            if n == 0:
                text += f"; there is no view to measure the {what} rate on (0 views)"
        return Infeasibility(
            binding, text, alpha, n_null, n_pres, _best(points, "false_clear"), _best(points, "false_clear_miss"),
            _best([p for p in points if p.gate_ok(alpha)], "false_clear_miss"),
            _best([p for p in points if p.match_ok(alpha)], "false_clear"), dict(grid),
            tuple({**p.setting, "false_clear": list(p.false_clear), "false_clear_miss": list(p.false_clear_miss)}
                  for p in points), tuple(sorted(set(arrangements))))

    @staticmethod
    def _best_text(b: Mapping[str, Any] | None) -> str:
        if b is None:
            return "none (no view, or no setting, to measure it on)"
        at = [_setting_text(st) for st in b["settings"]]
        more = f" and {len(at) - 3} more" if len(at) > 3 else ""
        return f"{b['rate']:.4f} ({b['x']}/{b['n']}) at {'; '.join(at[:3])}{more}"

    def summary(self) -> str:
        fc = "undef" if self.best_false_clear is None else f"{self.best_false_clear['rate']:.4f}"
        fcm = "undef" if self.best_false_clear_miss is None else f"{self.best_false_clear_miss['rate']:.4f}"
        return (f"joint search: no grid setting meets both constraints (binding: {self.binding}); best false-clear "
                f"{fc}, best false clear-miss {fcm} > alpha_clear {self.alpha_clear:g} where binding")

    def message(self) -> str:
        grid = "; ".join(f"{k} {_grid_text(v)}" for k, v in self.grid.items())
        return (f"Step 1 is infeasible: no setting on the predeclared grid meets both '{GATE_CONSTRAINT}' and "
                f"'{MATCH_CONSTRAINT}' with alpha_clear = {self.alpha_clear:g}. Binding: {self.binding_text}. Best "
                f"achievable false-clear rate over the grid: {self._best_text(self.best_false_clear)}; best false "
                f"clear-miss rate over the grid: {self._best_text(self.best_false_clear_miss)}; best false clear-miss "
                f"rate among settings meeting the false-clear constraint: "
                f"{self._best_text(self.best_false_clear_miss_meeting_false_clear)}; best false-clear rate among "
                f"settings meeting the false clear-miss constraint: "
                f"{self._best_text(self.best_false_clear_meeting_false_clear_miss)}. Grid searched (jointly): {grid}. "
                f"Dev views of {len(self.arrangements)} arrangements, both lightings. {NO_FALLBACK}.")

    def to_dict(self) -> dict[str, Any]:
        return {**{f.name: getattr(self, f.name) for f in dataclasses.fields(self)}, "message": self.message(),
                "summary": self.summary(), "search": JOINT_SEARCH}


class CalibrationInfeasible(RuntimeError):
    """Step 1 failed loudly: no setting on the joint grid meets both alpha_clear constraints. There is no candidate;
    alpha_clear is never relaxed, the grid never widened and no "least bad" setting is used. failure says what bound
    and the best achievable rates."""

    def __init__(self, failure: Infeasibility) -> None:
        super().__init__(failure.message())
        self.failure = failure


@dataclass(frozen=True)
class JointChoice:
    """The joint search's result on one set of arrangements."""
    chosen: GridPoint
    alpha_clear: float
    n_grid: int
    n_feasible: int                 # settings meeting both constraints (a JointChoice exists only if >= 1 does)
    n_meeting_false_clear: int
    n_meeting_false_clear_miss: int
    n_tied_at_max: int              # feasible settings with the chosen (maximal) correct clear-miss count
    decided_by: str | None          # None: the objective alone decided; else the first TIE_LEVELS entry that did
    points: tuple[GridPoint, ...] = field(default=(), repr=False, compare=False)

    @property
    def tie_break_decided(self) -> bool:
        return self.n_tied_at_max > 1

    def to_dict(self) -> dict[str, Any]:
        return {"chosen": self.chosen.to_dict(self.alpha_clear), "n_grid": self.n_grid, "n_feasible": self.n_feasible,
                "n_meeting_false_clear": self.n_meeting_false_clear,
                "n_meeting_false_clear_miss": self.n_meeting_false_clear_miss, "n_tied_at_max": self.n_tied_at_max,
                "tie_break_decided": self.tie_break_decided, "decided_by": self.decided_by}


def choose_joint(points: Sequence[GridPoint], alpha: float, grid: Mapping[str, Any],
                 arrangements: Iterable[str] = ()) -> JointChoice:
    """Step 1: among the settings meeting BOTH constraints, the first in the declared total order
    (GridPoint.order_key: max correct clear misses, then TIE_BREAK). Raises CalibrationInfeasible if none does."""
    feasible = [p for p in points if p.feasible(alpha)]
    if not feasible:
        raise CalibrationInfeasible(Infeasibility.of(points, alpha, grid, arrangements))
    ordered = sorted(feasible, key=GridPoint.order_key)
    best = ordered[0]
    tied = [p for p in feasible if p.clear_miss[0] == best.clear_miss[0]]
    decided_by = None
    if len(tied) > 1:
        rest = tied
        for level in range(1, len(TIE_LEVELS) + 1):   # order_key()[level] is TIE_LEVELS[level - 1]
            top = min(p.order_key()[level] for p in rest)
            rest = [p for p in rest if p.order_key()[level] == top]
            if len(rest) == 1:
                decided_by = TIE_LEVELS[level - 1]
                break
    return JointChoice(best, alpha, len(points), len(feasible), sum(p.gate_ok(alpha) for p in points),
                       sum(p.match_ok(alpha) for p in points), len(tied), decided_by, tuple(points))


CURVE_AXES = ("tau_q", "tau_occ", "tau_match")


def threshold_curves(choice: JointChoice) -> dict[str, dict[str, Any]]:
    """For each calibrated threshold, its grid with the other two held at the chosen values: the correct clear-miss
    curve and the false clear-miss curve (and the false-clear rate), with counts, the chosen point marked, "flat" when
    a curve does not move, and whether the declared tie-break decided along this curve (another feasible point on it
    has the same correct clear-miss count). A flat curve is reported as flat; no variation is forced."""
    c, alpha = choice.chosen, choice.alpha_clear
    out = {}
    for axis in CURVE_AXES:
        fixed = {k: v for k, v in c.setting.items() if k != axis}
        pts = sorted((p for p in choice.points if all(p.setting[k] == v for k, v in fixed.items())),
                     key=lambda p: p.setting[axis])
        feas = [p for p in pts if p.feasible(alpha)]
        cms = [p.clear_miss[0] for p in pts]
        fcms = [p.false_clear_miss[0] for p in pts]
        out[axis] = {
            "threshold": axis, "held_fixed": fixed, "chosen_value": c.setting[axis],
            "points": [{"value": p.setting[axis], "correct_clear_miss": list(p.clear_miss),
                        "correct_clear_miss_rate": _ratio(p.clear_miss), "false_clear_miss": list(p.false_clear_miss),
                        "false_clear_miss_rate": _ratio(p.false_clear_miss), "false_clear": list(p.false_clear),
                        "false_clear_rate": _ratio(p.false_clear), "feasible": p.feasible(alpha),
                        "chosen": p == c} for p in pts],
            "correct_clear_miss_flat": len(set(cms)) <= 1, "false_clear_miss_flat": len(set(fcms)) <= 1,
            "tie_break_decided_on_curve": sum(p.clear_miss[0] == c.clear_miss[0] for p in feas) > 1}
    return out


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


class SubsetStoreFactory(RecordStoreFactory):
    """A StoreFactory holding only some episodes of another one: a snapshot of the listed episodes' records, taken
    once at construction from one fresh store of the base factory, which is then dropped. It keeps no reference to the
    base (review, checkpoint 2: it used to keep the full factory in self._base and rebuild from it on every call, so a
    fold-trained policy could call train.store_factory._base() and read the held-out arrangement's evidence). Nothing
    but the listed episodes is reachable through it, by construction. Its source names no file: the base's source
    can carry the evidence file's path, and that file holds every dev arrangement."""

    def __init__(self, base: StoreFactory, episode_ids: Iterable[str]) -> None:
        src = base()
        ids = sorted(set(episode_ids))
        super().__init__({eid: [src.get(eid, s) for s in range(src.n_stops)] for eid in ids}, src.n_stops,
                         synthetic=src.synthetic,
                         source=f"subset of the dev evidence ({len(ids)} episodes; source file not recorded)")


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
    the evidence store's or the truth's, and real (non-synthetic) data without an evidence_sha256.

    simulation: the manifest's top-level "simulation" flag (every output is then labelled SIMULATION).
    strata: {arrangement_id: quota slot class} for the stratified cluster bootstrap, or None (unstratified, flagged).
    inputs: the input files and their sha256 (the sim dev path), recorded in the candidate."""
    specs: tuple[EpisodeSpec, ...]
    store_factory: StoreFactory
    truth: GroundTruth
    view_labels: tuple[ViewLabel, ...]
    positions: Positions
    synthetic: bool
    label_source: str
    evidence_sha256: str = ""   # required on real data; synthetic_dev_data fills it in too
    simulation: bool = False
    strata: Mapping[str, str] | None = None
    inputs: Mapping[str, Any] | None = None

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
        if self.synthetic and self.simulation:
            raise ValueError("DevData cannot be both synthetic and simulation")
        if not self.synthetic and not (self.evidence_sha256 or "").strip():
            raise ValueError("real (non-synthetic) dev data need evidence_sha256, the provenance hash of the dev "
                             "evidence (e.g. RecordStoreFactory.evidence_sha256()); the candidate records it")
        if self.strata is not None:
            object.__setattr__(self, "strata", dict(sorted(self.strata.items())))
            if set(self.strata) != {s.arrangement_id for s in self.specs}:
                raise ValueError("strata must name exactly the dev arrangements")

    @property
    def arrangements(self) -> tuple[str, ...]:
        return tuple(sorted({s.arrangement_id for s in self.specs}))

    @property
    def view_labels_sha256(self) -> str:
        """sha256 of the per-view labels (view_labels_sha256), recorded next to evidence_sha256."""
        return view_labels_sha256(self.view_labels)

    @property
    def label(self) -> str:
        """The run label: SYNTHETIC, SIMULATION, or "" for real (photo) data."""
        return SYNTHETIC_LABEL if self.synthetic else (SIMULATION_LABEL if self.simulation else "")

    def subset(self, arrangements: Iterable[str]) -> DevData:
        """The same dev data restricted to the given arrangements: specs, evidence (a SubsetStoreFactory snapshot with
        no reference to this store factory), truth, labels and strata hold nothing else, and evidence_sha256 is the
        subset's own. Used per CV fold, so a fold-trained policy never sees the held-out one.

        inputs keeps only the input files' sha256 (subset_inputs): the file paths are dropped, because the evidence and
        labels files hold every dev arrangement, the held-out one included (review, checkpoint 2: json.load of
        train.inputs["view_labels"]["file"] gave the held-out arrangement's geometric labels)."""
        keep = set(arrangements)
        unknown = keep - set(self.arrangements)
        if unknown or not keep:
            raise ValueError(f"subset needs dev arrangements; unknown: {sorted(unknown)[:5]}")
        specs = tuple(s for s in self.specs if s.arrangement_id in keep)
        ids = {s.episode_id for s in specs}
        truth = GroundTruth({e: self.truth[e] for e in sorted(ids)}, self.truth.indistinguishable & ids,
                            synthetic=self.truth.synthetic)
        factory = SubsetStoreFactory(self.store_factory, ids)
        return DevData(specs, factory, truth,
                       tuple(l for l in self.view_labels if l.episode_id in ids), self.positions, self.synthetic,
                       self.label_source, factory.evidence_sha256(), simulation=self.simulation,
                       strata=None if self.strata is None else {a: self.strata[a] for a in sorted(keep)},
                       inputs=subset_inputs(self.inputs))


SUBSET_INPUTS_NOTE = ("a subset of the dev data (e.g. a CV training fold): the input files hold every dev arrangement, "
                      "so their paths are dropped and only their sha256 is kept")


def subset_inputs(inputs: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A subset's inputs: {name: {"sha256"}} for every input file, with no path (and no ignored_episodes list), plus
    a note. Nothing in it leads back to a file that holds another arrangement's evidence or labels."""
    if inputs is None:
        return None
    out: dict[str, Any] = {k: {"sha256": v["sha256"]} for k, v in sorted(inputs.items())
                           if isinstance(v, Mapping) and "sha256" in v}
    out["note"] = SUBSET_INPUTS_NOTE
    return out


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
    """Dev split of the synthetic generator, with stand-in labels from its hidden layout (no family: unstratified)."""
    specs, store, truth = generate(n_arr, seed, params, positions)
    dev = [s for s in specs if s.split is Split.DEV]
    ids = [s.episode_id for s in dev]
    factory = RecordStoreFactory.from_store(store, ids, source=f"verify.synthetic seed={seed} (dev only)")
    dev_truth = GroundTruth({e: truth[e] for e in ids}, truth.indistinguishable & set(ids), synthetic=True)
    layouts = generate_layouts(n_arr, seed, params, n_stops=positions.n_stops, start_index=positions.start_index)
    labels = synthetic_view_labels(dev, layouts, positions.n_stops)
    return DevData(tuple(dev), factory, dev_truth, tuple(labels), positions, True, SYNTHETIC_LABEL_SOURCE,
                   factory.evidence_sha256())


def _rel(path: str | Path) -> str:
    """Repo-relative path when inside the repo (keeps recorded paths machine-independent)."""
    p = Path(path).resolve()
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)


def sim_dev_data(manifest: str | Path, evidence: str | Path, view_labels: str | Path,
                 positions: str | Path) -> DevData:
    """DevData from the set files sim/episodes.py writes (requirement 1, 2026-11-03):

    manifest     manifest.json (cmpe249.arrangements.v1): every arrangement must be split dev (SplitRefused otherwise,
                 before any other file is read); "simulation", occluder_family / m16_side / design_case are read.
    evidence     evidence.jsonl from perception/evidence.py on jobs_refL1.json (EvidenceStore format).
    view_labels  view_labels.json (cmpe249.sim-view-labels-set.v1): the geometric labels, the only label source.
    positions    the stops, e.g. configs/sim/positions_shelf_A.json.

    The evidence and labels files may hold episodes of other arrangements of the same rendered set (one jobs file
    covers the random and the design manifests); only the manifest's episodes are read, and the others are counted
    in inputs["ignored_episodes"]. Every manifest episode needs evidence and labels at every stop. The sha256 of all
    four files is recorded (inputs)."""
    info = load_manifest_info(manifest)
    bad = sorted(a.arrangement_id for a in info.arrangements if a.split is not Split.DEV)
    if bad:
        raise SplitRefused(f"{manifest}: selection reads DEV arrangements only; refused {len(bad)} non-dev "
                           f"arrangements (e.g. {bad[:3]})")
    specs, truth = episodes_from_arrangements(info.arrangements)
    pos = Positions.load(positions)
    store = EvidenceStore.from_jsonl(evidence, pos.n_stops, source=f"perception.evidence ({_rel(evidence)})")
    if store.synthetic:
        raise ValueError(f"{evidence} is synthetic evidence; use --synthetic")
    ids = [s.episode_id for s in specs]
    missing = sorted(set(ids) - set(store.episode_ids()))
    if missing:
        raise ValueError(f"{evidence} has no evidence for {len(missing)} manifest episodes, e.g. {missing[:3]}")
    factory = RecordStoreFactory.from_store(store, ids)   # KeyError if a stop is missing
    glab = load_geometric_labels(view_labels, pos.n_stops)
    missing = sorted(set(ids) - set(glab.episodes))
    if missing:
        raise ValueError(f"{view_labels} has no labels for {len(missing)} manifest episodes, e.g. {missing[:3]}")
    if glab.simulation != info.simulation:
        raise ValueError(f"the manifest says simulation={info.simulation} but {view_labels} says "
                         f"simulation={glab.simulation}")
    labels = tuple(ViewLabel(e, st, d) for e in ids for st, d in enumerate(glab.episodes[e]))
    ign_ev = sorted(set(store.episode_ids()) - set(ids))
    ign_lab = sorted(set(glab.episodes) - set(ids))
    inputs = {name: {"file": _rel(p), "sha256": file_sha256(p)} for name, p in
              (("manifest", manifest), ("evidence", evidence), ("view_labels", view_labels), ("positions", positions))}
    inputs["ignored_episodes"] = {
        "note": "episodes in the evidence or labels file that the manifest does not list; never read",
        "evidence": {"n": len(ign_ev), "arrangements": sorted({e.rsplit("-", 2)[0] for e in ign_ev})},
        "view_labels": {"n": len(ign_lab), "arrangements": sorted({glab.arrangement_of[e] for e in ign_lab})}}
    if ign_ev or ign_lab:
        log.warning(f"sim dev input: {len(ign_ev)} evidence and {len(ign_lab)} label episodes are not in {manifest} "
                    "and were ignored")
    return DevData(tuple(specs), factory, truth, labels, pos, False, SIM_LABEL_SOURCE if info.simulation else
                   f"geometric labels ({SIM_VIEW_LABELS_SCHEMA})", factory.evidence_sha256(),
                   simulation=info.simulation, strata=info.strata(), inputs=inputs)


def no_diagnostic_shares(dev: DevData) -> dict[str, dict[str, Any]]:
    """Geometric no-diagnostic-view share per lighting (and pooled): episodes with no diagnostic stop, from the dev
    view labels. Context for coverage, never a ceiling."""
    diag: dict[str, bool] = {}
    for l in dev.view_labels:
        diag[l.episode_id] = diag.get(l.episode_id, False) or l.diagnostic
    out = {}
    for light in (*REPORT_LIGHTINGS, POOLED):
        eps = [s.episode_id for s in dev.specs if light == POOLED or s.lighting.value == light]
        x = sum(not diag[e] for e in eps)
        out[light] = {"x": x, "n": len(eps), "share": _ratio((x, len(eps)))}
    return out


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
    #                               FRR bound); with no joint grid setting meeting both, calibration fails
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
            # A repeated value would count as another grid setting in the reported counts, the infeasible record
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
    """The false-clear null set: non-diagnostic views of occluded-present episodes (a gate pass here is a false
    clear)."""
    return [r for r in rows if r.condition is Condition.OCCLUDED and not r.diagnostic]


def present_diagnostic_views(rows: Iterable[ViewRow]) -> list[ViewRow]:
    """The false clear-miss null set: diagnostic views of present episodes, i.e. the target is there and visible."""
    return [r for r in rows if r.present and r.diagnostic]


def removed_diagnostic_views(rows: Iterable[ViewRow]) -> list[ViewRow]:
    """The objective's set: diagnostic views of removed episodes."""
    return [r for r in rows if not r.present and r.diagnostic]


def count(rows: Sequence[ViewRow], predicate: Callable[[Observation], bool]) -> tuple[int, int]:
    """(views where the rule's predicate holds, views)."""
    return sum(bool(predicate(r.obs)) for r in rows), len(rows)


def frozen_thresholds(tau_match: float, tau_q: float, tau_occ: float) -> Thresholds:
    """The three calibrated values with k and lam fixed a priori (every Thresholds that calibration produces)."""
    return Thresholds(tau_match=tau_match, tau_q=tau_q, tau_occ=tau_occ, k=K_FIXED, lam=LAM_FIXED)


@dataclass(frozen=True)
class Calibration:
    """Step 1 on one set of arrangements: the joint search; k and lam fixed."""
    choice: JointChoice
    arrangements: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def thresholds(self) -> Thresholds:
        c = self.choice.chosen
        return frozen_thresholds(c.tau_match, c.tau_q, c.tau_occ)


def calibrate(rows: Sequence[ViewRow], settings: SelectionSettings, arrangements: Iterable[str] | None = None,
              *, log_warnings: bool = True, table: ViewTable | None = None) -> Calibration:
    """Steps 1-2 on the views of the given arrangements (default: every row): the joint search over tau_q x tau_occ x
    tau_match. It reads views and their labels only: it never builds an env, runs a policy or sees an episode
    decision. table (a ViewTable of a superset of these rows, built with the same grids) saves recounting; the result
    is the same. Raises CalibrationInfeasible when no grid setting meets both alpha_clear constraints."""
    if arrangements is not None:
        keep = set(arrangements)
        rows = [r for r in rows if r.arrangement_id in keep]
    arrs = tuple(sorted({r.arrangement_id for r in rows}))
    if table is None or (table.tau_q_grid, table.tau_occ_grid, table.tau_match_grid) != \
            (settings.tau_q_grid, settings.tau_occ_grid, settings.tau_match_grid):
        table = ViewTable(rows, settings)
    points = table.points(arrs)
    choice = choose_joint(points, settings.alpha_clear, table.grid, arrs)
    warnings: list[str] = []
    if not points[0].clear_miss[1]:
        warnings.append("Step 1: no diagnostic removed-episode views; the correct clear-miss count is 0 at every "
                        "setting, so the declared tie-break picks the most conservative feasible setting")
    if log_warnings:
        for w in warnings:
            log.warning(w)
    return Calibration(choice, arrs, tuple(warnings))


# Gate pass rates per lighting (Codex review, 2026-11-03): a feasible setting can meet both constraints by rejecting a
# whole lighting at the quality gate (every L2 view fails tau_q), so feasibility alone does not show robustness. The
# share of views passing the tau_q / tau_occ gate is reported per lighting at the chosen setting and in every fold,
# and a lighting where it is at or below GATE_PASS_LOW is flagged.
GATE_PASS_POPULATIONS = ("removed_diagnostic", "present_diagnostic", "all")
GATE_PASS_TEXT = {"removed_diagnostic": "removed diagnostic views", "present_diagnostic": "present diagnostic views",
                  "all": "views (all)"}
GATE_PASS_LOW = 0.05   # "nearly zero": at or below this share of a lighting's views pass the gate (flagged)


def gate_pass_by_lighting(rows: Sequence[ViewRow], th: Thresholds) -> dict[str, dict[str, tuple[int, int]]]:
    """{"L1", "L2", "pooled": {population: (views passing StoppingRule.passes_gate, views)}} for diagnostic removed
    views, diagnostic present views and all views."""
    rule = StoppingRule(th)
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for light in (*REPORT_LIGHTINGS, POOLED):
        sub = [r for r in rows if light == POOLED or r.lighting.value == light]
        pops = {"removed_diagnostic": removed_diagnostic_views(sub), "present_diagnostic": present_diagnostic_views(sub),
                "all": sub}
        out[light] = {k: count(pops[k], rule.passes_gate) for k in GATE_PASS_POPULATIONS}
    return out


def gate_pass_flags(gp: Mapping[str, Mapping[str, Sequence[int]]], where: str) -> list[str]:
    """One line per lighting and population (diagnostic removed views, diagnostic present views and all views,
    docs/SIM_PREREGISTRATION.md section 7) where (nearly) no view passes the gate, and one line per lighting and
    population that has no view at all: thresholds that are feasible by rejecting a whole lighting must be obvious,
    and a share that could not be measured must say so (review, checkpoint 2: only the two diagnostic populations
    were flagged, and an empty one was skipped without a word)."""
    out = []
    for light in REPORT_LIGHTINGS:
        for pop in GATE_PASS_POPULATIONS:
            x, n = gp[light][pop]
            if not n:
                out.append(f"{where}: no views: there are no {light} {GATE_PASS_TEXT[pop]} (n = 0), so their gate "
                           "pass share is undefined there")
                continue
            if x == 0:
                out.append(f"{where}: the gate rejects ALL {light} {GATE_PASS_TEXT[pop]} (0/{n} pass tau_q / tau_occ)")
            elif x / n <= GATE_PASS_LOW + EPS:
                out.append(f"{where}: the gate passes nearly no {light} {GATE_PASS_TEXT[pop]} ({x}/{n} = {x / n:.3f} "
                           f"<= {GATE_PASS_LOW:g})")
    return out


def _gate_pass_dict(gp: Mapping[str, Mapping[str, Sequence[int]]]) -> dict[str, Any]:
    return {light: {pop: {"x": xn[0], "n": xn[1], "share": _ratio((xn[0], xn[1]))} for pop, xn in d.items()}
            for light, d in gp.items()}


VIEW_RATES = ("false_clear", "clear", "false_clear_miss", "clear_miss", "false_strong_match")
VIEW_ERRORS = ("false_clear", "false_clear_miss", "false_strong_match")   # reported with upper bounds


@dataclass(frozen=True)
class ViewRates:
    """Per-view counts (x, n) at one threshold setting, each from the rule's own predicate:

    false_clear         passes_gate on non-diagnostic views of occluded-present episodes (constraint 1's null set)
    clear               passes_gate on diagnostic views of removed episodes
    false_clear_miss    is_clear_miss on diagnostic views of present episodes (constraint 2's null set)
    clear_miss          is_clear_miss on diagnostic views of removed episodes (the objective)
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
    gate_pass_train: Mapping[str, Any] | None = None   # gate_pass_by_lighting on the fold's training views
    gate_pass_held: Mapping[str, Any] | None = None    # ... and on the held-out arrangement's views

    def to_dict(self) -> dict[str, Any]:
        return {"held_out": self.held_out,
                "thresholds": None if self.thresholds is None else self.thresholds.to_dict(),
                "held": None if self.held is None else {k: v.to_dict() for k, v in self.held.items()},
                "gate_pass_train": None if self.gate_pass_train is None else _gate_pass_dict(self.gate_pass_train),
                "gate_pass_held": None if self.gate_pass_held is None else _gate_pass_dict(self.gate_pass_held),
                "infeasible": None if self.infeasible is None else {
                    "binding": self.infeasible.binding, "summary": self.infeasible.summary(),
                    "best_false_clear": self.infeasible.best_false_clear,
                    "best_false_clear_miss": self.infeasible.best_false_clear_miss}}


def grouped_cv(rows: Sequence[ViewRow], settings: SelectionSettings, arrangements: Iterable[str],
               table: ViewTable | None = None) -> list[ViewFold]:
    """Step 3 (views only): for each arrangement, calibrate (the same joint search) on the other arrangements' views
    and score its own. A fold whose calibration is infeasible is recorded as such, with no thresholds. Fewer than two
    arrangements: no folds."""
    arrs = sorted(set(arrangements))
    if len(arrs) < 2:
        return []
    folds = []
    for a in arrs:
        try:
            cal = calibrate(rows, settings, [x for x in arrs if x != a], log_warnings=False, table=table)
        except CalibrationInfeasible as e:
            folds.append(ViewFold(a, None, None, e.failure))
            continue
        held = [r for r in rows if r.arrangement_id == a]
        train = [r for r in rows if r.arrangement_id != a and r.arrangement_id in arrs]
        folds.append(ViewFold(a, cal.thresholds, rates_by_lighting(held, cal.thresholds), None,
                              gate_pass_by_lighting(train, cal.thresholds), gate_pass_by_lighting(held, cal.thresholds)))
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


def freeze_eligibility(folds: Sequence[ViewFold], arrangements: Sequence[str]) -> tuple[bool, list[dict[str, Any]]]:
    """(eligible_for_freeze, ineligible_reasons). Calibration on all dev data has succeeded when this is called (else
    there is no candidate). Eligible only if every planned CV fold is feasible; fewer than two dev arrangements plan no
    fold at all, which is not eligible either (there is no held-out estimate)."""
    reasons: list[dict[str, Any]] = []
    if len(set(arrangements)) < 2:
        reasons.append({"reason": "no grouped cross-validation: fewer than 2 dev arrangements", "fold": None,
                        "constraint": None})
    for f in folds:
        if f.infeasible is not None:
            reasons.append({"reason": f"calibrating without {f.held_out} is infeasible", "fold": f.held_out,
                            "constraint": f.infeasible.binding, "summary": f.infeasible.summary(),
                            "best_false_clear": f.infeasible.best_false_clear,
                            "best_false_clear_miss": f.infeasible.best_false_clear_miss})
    return not reasons, reasons


# ---------------------------------------------------------------- Steps 3b-5: policy runs (after every threshold is fixed)

class FoldTrainable(Protocol):
    """A policy trained per CV fold (ranker A later). train() receives a DevData holding the fold's TRAINING
    arrangements only (DevData.subset: its store, truth, labels and strata have nothing of the held-out arrangement,
    and its store factory and inputs lead nowhere else) and the fold's thresholds, and returns a factory of fresh
    policy instances, which then run on the held-out arrangement through the public replay API with the fold's
    stopping rule."""
    name: str

    def train(self, train: DevData, thresholds: Thresholds) -> Callable[[], Any]: ...


class FoldTrainableFactory(Protocol):
    """What run_selection / held_out_check take for a fold-trained policy: a callable with a `name` that builds a
    FRESH FoldTrainable (a FoldTrainable class is one). Each fold calls it once, in a process of its own
    (fold_isolated), and trains and runs the new trainable there; the parent process never calls it. State a
    trainable keeps, on itself, on its class or in its module, therefore ends with its fold: fold j can never inherit
    what fold i (whose training set holds j) learned (review, checkpoint 2: one object was trained in every fold)."""
    name: str

    def __call__(self) -> FoldTrainable: ...


def check_trainable_factories(trainables: Sequence[Any]) -> None:
    """TypeError unless every entry is a FoldTrainableFactory: callable, with a str name, and not itself a trainable
    instance (an instance passed here would be trained in every fold, carrying its state across them)."""
    for t in trainables:
        if not isinstance(t, type) and callable(getattr(t, "train", None)):
            raise TypeError(f"{t!r} is a FoldTrainable instance; pass a factory that builds a fresh one per fold "
                            "(e.g. its class), so no state can carry over from one fold to the next")
        if not callable(t) or not isinstance(getattr(t, "name", None), str):
            raise TypeError(f"{t!r} is not a FoldTrainableFactory (a callable with a str name)")


def fold_isolated(fn: Callable[..., Any], *args: Any) -> Any:
    """fn(*args) in a forked child process; returns its (pickled) result or raises RuntimeError with the child's
    traceback. The child starts from the parent's memory, in which no fold-trained code has run (the parent never
    calls a FoldTrainableFactory), and everything it changes dies with it."""
    ctx = multiprocessing.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)

    def child() -> None:
        recv.close()
        try:
            msg = ("ok", fn(*args))
        except BaseException:   # report everything, then exit: nothing may run on in the child
            msg = ("error", traceback.format_exc())
        try:
            send.send(msg)
        except Exception:   # e.g. an unpicklable result
            send.send(("error", traceback.format_exc()))
        send.close()
    proc = ctx.Process(target=child)
    proc.start()
    send.close()
    try:
        status, value = recv.recv()   # before join: a large result would fill the pipe and block the child
    except EOFError:
        status, value = "error", "the fold process exited without a result"
    finally:
        recv.close()
        proc.join()
    if status != "ok":
        raise RuntimeError(f"fold-isolated run failed:\n{value}")
    return value


def _train_and_run_fold(factory: FoldTrainableFactory, train: DevData, held: DevData, thresholds: Thresholds,
                        settings: SelectionSettings, context: RunContext
                        ) -> tuple[dict[tuple[str, float | None], tuple[EpisodeResult, ...]], int]:
    """In a fold's own process (fold_isolated): build a fresh trainable, train it on the fold's training subset, and
    run it (k rule and -fcm) on the held-out arrangement's episodes, uncapped and under every cap. Returns
    {(policy, cap_steps): results} and the number of replays."""
    trained = {factory.name: factory().train(train, thresholds)}
    runner = PolicyRunner(held, settings, context, trained)
    out = {}
    for p in (factory.name, f"{factory.name}-fcm"):
        for cap in policy_caps(p, settings):
            out[(p, cap)] = runner.results(p, thresholds, run_budget(p, cap, settings, held.positions))
    return out, runner.n_runs


def split_policy(name: str) -> tuple[str, bool]:
    """'G-fcm' -> ('G', True); 'G' -> ('G', False)."""
    return (name[:-4], True) if name.endswith("-fcm") else (name, False)


def make_policy(name: str, th: Thresholds, trained: Mapping[str, Callable[[], Any]] | None = None) -> Any:
    """A fresh deterministic policy through its public constructor (R is built per seed by run_r_seeds). name may
    carry '-fcm' (the rule, not the policy, differs).

    "G" is primary G (D-norm, lookahead off; DECISIONS 2026-10-12), "G-lookahead" the same scorer with the
    reachability lookahead; both read th.lam. G-myopic keeps its legacy travel weight (GMYOPIC_LAM_DEG degrees),
    as in verify.run. S, F, R and FULL ignore lam. trained: fold-trained policy factories (FoldTrainable).
    """
    base, _ = split_policy(name)
    if trained and base in trained:
        return trained[base]()
    if base == "S":
        return S()
    if base == "F":
        return F()
    if base == "FULL":
        return FullSweep()
    if base == "G":
        return G.from_thresholds(th, lookahead=False)
    if base == "G-lookahead":
        return G.from_thresholds(th, lookahead=True)
    if base == "G-myopic":
        return GMyopic.from_thresholds(th)
    raise ValueError(f"unsupported policy {name!r} (R runs through run_r_seeds)")


def run_budget(policy: str, cap_steps: float | None, settings: SelectionSettings, positions: Positions) -> Budget:
    """S: Budget(0) (one view; budget- and cap-independent). FULL: Budget(4), uncapped (every stop). Every other policy:
    the primary budget under the cap."""
    base, _ = split_policy(policy)
    if base == "S":
        return Budget(0)
    if base == "FULL":
        return Budget(FULL_SWEEP_VIEWS)
    return Budget(settings.budget, cap_cm_from_steps(cap_steps, positions))


def policy_caps(policy: str, settings: SelectionSettings) -> tuple[float | None, ...]:
    """S and FULL run once (uncapped); every other policy uncapped and under every cap."""
    return (None,) if split_policy(policy)[0] in ("S", "FULL") else settings.cap_settings


class PolicyRunner:
    """Runs one policy on every episode of dev through the public replay API (Steps 3b-5 only).

    run_selection builds it only after every threshold set is fixed, and it only ever receives Thresholds. One fresh
    store per run setting (all keys minted before the first bind seals it); results are cached per setting.
    """

    def __init__(self, dev: DevData, settings: SelectionSettings, context: RunContext,
                 trained: Mapping[str, Callable[[], Any]] | None = None) -> None:
        self.dev = dev
        self.r_seeds = tuple(range(settings.r_seeds))
        self.context = context
        self.trained = dict(trained or {})
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
        base, fcm = split_policy(policy)
        rule = StoppingRule.first_clear_miss(th) if base == "S" or fcm else StoppingRule(th)
        store = self.dev.store_factory()       # fresh store per setting
        keys = store.replay_keys()             # minted before the first bind seals it
        out = []
        for spec in self.dev.specs:
            def make_env(key=keys[spec.episode_id]) -> ReplayEnv:
                return ReplayEnv(key, store, self.dev.positions, budget, self.context)
            if base == "R" and "R" not in self.trained:
                logs, res = run_r_seeds(make_env, rule, self.r_seeds, r_seed_salt(spec), spec=spec)
                self.n_runs += len(logs)
            else:
                # name=policy: a fold-trained policy is logged under its own name, whatever its class calls itself
                run = run_episode(make_policy(policy, th, self.trained), make_env(), rule, name=policy)
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
    def frr_arrangement(self) -> float | None:
        """Share of arrangements with a present episode that have any false removal (None for R or when undefined)."""
        if self.n_arr_with_false_removal is None or not self.n_arr_with_present:
            return None
        return self.n_arr_with_false_removal / self.n_arr_with_present

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
        return {**dataclasses.asdict(self), "frr": self.frr, "frr_arrangement": self.frr_arrangement,
                "recall": self.recall, "coverage": self.coverage, "mean_travel_cm": self.mean_travel_cm,
                "mean_views": self.mean_views}


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
def _draw_counts(n_arrangements: int, B: int, seed: int, strata: tuple[str, ...] | None = None) -> np.ndarray:
    """(B, n) multiplicity of each arrangement in each bootstrap draw (analysis.bootstrap: arrangement_draws, or
    stratified_arrangement_draws when strata gives each arrangement's stratum)."""
    draws = arrangement_draws(n_arrangements, B, seed) if strata is None else \
        stratified_arrangement_draws(strata, B, seed)
    counts = np.zeros((B, n_arrangements))
    np.add.at(counts, (np.arange(B)[:, None], draws), 1.0)
    counts.setflags(write=False)
    return counts


def stratification(strata: Mapping[str, str] | None, arrangements: Iterable[str]) -> dict[str, Any]:
    """{"stratified", "strata": {stratum: n_arrangements}, "singleton_strata", "note"} over the given arrangements.
    With a stratum of fewer than two of them the bootstrap is unstratified (analysis.bootstrap.resolve_strata): a
    stratum of one always redraws itself and adds no variance, and the note says "unstratified: singleton strata"."""
    arrs = sorted(set(arrangements))
    if strata is None:
        return {"stratified": False, "strata": {}, "singleton_strata": [], "note": UNSTRATIFIED_NOTE}
    used, singles, note = resolve_strata(arrs, strata)
    counts: dict[str, int] = {}
    for a in arrs:
        counts[strata[a]] = counts.get(strata[a], 0) + 1
    if used is None:
        return {"stratified": False, "strata": dict(sorted(counts.items())), "singleton_strata": list(singles),
                "note": note}
    return {"stratified": True, "strata": dict(sorted(counts.items())), "singleton_strata": [],
            "note": "resampled within quota slot classes (occluder family, M16 by side), keeping each class's count"}


def cluster_bootstrap(outcomes: Sequence[Outcome], B: int, seed: int,
                      strata: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Percentile 95% CIs for FRR, recall, coverage, resampling arrangements with replacement (fractions for R);
    within each stratum, keeping its count, when strata ({arrangement_id: stratum}) is given and no stratum of these
    arrangements is a singleton (else unstratified, flagged: stratification)."""
    arrs = sorted({o.arrangement_id for o in outcomes})
    out: dict[str, Any] = {"B": B, "seed": seed, "n_arrangements": len(arrs), **stratification(strata, arrs)}
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
    counts = _draw_counts(len(arrs), B, seed, tuple(strata[a] for a in arrs) if out["stratified"] else None)
    for m, (num, den) in terms.items():
        n, d = counts @ num, counts @ den
        ok = d > EPS
        vals = n[ok] / d[ok]
        ci = [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals.size else [None, None]
        out[m] = {"ci95": ci, "n_undefined": int((~ok).sum())}
    return out


@dataclass(frozen=True)
class CheckRow:
    """One policy at one run setting, one query lighting (Steps 3b-5). Lighting "pooled" is a secondary line."""
    policy: str
    k: int                          # the rule's k (S and -fcm: 1)
    lam: float                      # Thresholds.lam (read by G and G-lookahead only)
    budget: int                     # additional views (S: 0, FULL: 4)
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


def rows_from_outcomes(policy: str, k: int, lam: float, budget: Budget, cap_steps: float | None,
                       outs: Sequence[Outcome], settings: SelectionSettings, strata: Mapping[str, str] | None,
                       *, with_bounds: bool = True) -> list[CheckRow]:
    """One CheckRow per L1, L2 and pooled from one policy's outcomes."""
    rows = []
    for light in (*REPORT_LIGHTINGS, POOLED):
        sub = outs if light == POOLED else [o for o in outs if o.lighting.value == light]
        c = Counts.of(sub, binomial=split_policy(policy)[0] != "R")
        rows.append(CheckRow(policy, k, lam, budget.additional_views, cap_steps, budget.travel_cap_cm, light, c,
                             bounds(c, settings.conf) if with_bounds else None,
                             cluster_bootstrap(sub, settings.bootstrap_B, settings.seed, strata) if with_bounds
                             else None,
                             None if c.frr is None else c.frr > settings.frr_limit + EPS))
    return rows


def rule_k(policy: str, th: Thresholds) -> int:
    base, fcm = split_policy(policy)
    return 1 if base == "S" or fcm else th.k


def setting_rows(runner: PolicyRunner, policy: str, th: Thresholds, cap_steps: float | None,
                 settings: SelectionSettings, *, with_bounds: bool = True) -> list[CheckRow]:
    """Run one policy at (th, primary budget, cap) on every dev episode; one row per L1, L2 and pooled."""
    budget = run_budget(policy, cap_steps, settings, runner.dev.positions)
    if split_policy(policy)[0] in ("S", "FULL"):
        cap_steps = None
    outs = [Outcome.of(r, runner.dev.truth) for r in runner.results(policy, th, budget)]
    return rows_from_outcomes(policy, rule_k(policy, th), th.lam, budget, cap_steps, outs, settings,
                              runner.dev.strata, with_bounds=with_bounds)


def run_check(runner: PolicyRunner, th: Thresholds, settings: SelectionSettings) -> list[CheckRow]:
    """Step 4: every policy at the frozen thresholds, the primary budget, uncapped and every cap (S once)."""
    rows = []
    for p in CHECK_POLICIES:
        for cap in policy_caps(p, settings):
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


def heldout_policies(trainables: Sequence[FoldTrainableFactory]) -> tuple[str, ...]:
    """HELDOUT_POLICIES, then every fold-trained policy with its -fcm variant."""
    check_trainable_factories(trainables)
    names = [t.name for t in trainables]
    for n in names:
        if n in BUILTIN_POLICIES or n.endswith("-fcm") or not n:
            raise ValueError(f"fold-trained policy name {n!r} clashes with a built-in policy or the -fcm suffix")
    if len(set(names)) != len(names):
        raise ValueError("fold-trained policy names must be unique")
    return (*HELDOUT_POLICIES, *(x for n in names for x in (n, f"{n}-fcm")))


def held_out_check(dev: DevData, settings: SelectionSettings, context: RunContext, folds: Sequence[ViewFold],
                   trainables: Sequence[FoldTrainableFactory] = ()) -> dict[str, Any]:
    """Step 3b, the held-out policy check: in every feasible fold, every policy runs on the held-out arrangement's
    episodes (all conditions and lightings) with that fold's thresholds; a fold-trained policy is built fresh by its
    factory, trained on dev.subset(training arrangements) only and run on the held-out episodes, all in the fold's
    own process (fold_isolated), so nothing it keeps reaches another fold. Outcomes are pooled over folds into rows
    per policy, cap and lighting (FRR per episode and per arrangement, recall, coverage, mean views and travel, counts,
    bounds and the stratified bootstrap). FRR above the limit is flagged as a reported check; it never touches
    eligibility or thresholds."""
    policies = heldout_policies(trainables)
    outs: dict[tuple[str, float | None], list[Outcome]] = {}
    fold_log: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    n_runs = 0
    for f in folds:
        if f.thresholds is None:
            fold_log.append({"held_out": f.held_out, "ran": False, "thresholds": None,
                             "reason": f"calibration without it is infeasible ({f.infeasible.summary()})"
                             if f.infeasible else "no thresholds"})
            continue
        train_arrs = [a for a in dev.arrangements if a != f.held_out]
        train = dev.subset(train_arrs)
        held = dev.subset([f.held_out])
        fold_trained: dict[tuple[str, float | None], tuple[EpisodeResult, ...]] = {}
        for t in trainables:   # one process per fold and trainable: no state outlives its fold
            got, n = fold_isolated(_train_and_run_fold, t, train, held, f.thresholds, settings, context)
            fold_trained.update(got)
            n_runs += n
        runner = PolicyRunner(held, settings, context)   # built-in policies only
        for p in policies:
            for cap in policy_caps(p, settings):
                budget = run_budget(p, cap, settings, dev.positions)
                got = fold_trained.get((p, cap))
                for r in (got if got is not None else runner.results(p, f.thresholds, budget)):
                    outs.setdefault((p, cap), []).append(Outcome.of(r, held.truth))
                    episodes.append({"fold": f.held_out, "policy": p, "cap_steps": cap,
                                     "travel_cap_cm": budget.travel_cap_cm, "episode_id": r.episode_id,
                                     "lighting": r.lighting.value,
                                     "decision_frac": {d.value: r.frac(d) for d in Decision},
                                     "mean_views": r.mean_views, "mean_travel_cm": r.mean_travel_cm})
        n_runs += runner.n_runs
        fold_log.append({"held_out": f.held_out, "ran": True, "thresholds": f.thresholds.to_dict(),
                         "training_arrangements": train_arrs, "trained_policies": sorted(t.name for t in trainables)})
    rows: list[CheckRow] = []
    th0 = frozen_thresholds(0.5, 0.5, 0.5)   # only k and lam are read from it (the same in every fold)
    for p in policies:
        for cap in policy_caps(p, settings):
            if (p, cap) in outs:
                rows += rows_from_outcomes(p, rule_k(p, th0), LAM_FIXED, run_budget(p, cap, settings, dev.positions),
                                           cap, outs[(p, cap)], settings, dev.strata)
    ran = [x["held_out"] for x in fold_log if x["ran"]]
    flags = [f"held-out check: {r.policy}, {setting_label(r)}, {r.lighting} queries: FRR {r.counts.frr:.3f} "
             f"({_num(r.counts.n_false_removal)}/{r.counts.n_present}) > frr_limit {settings.frr_limit:g} (a reported "
             "check: it does not affect eligibility or any threshold)"
             for r in rows if r.lighting in REPORT_LIGHTINGS and r.over_limit]
    return {"rows": rows, "folds": fold_log, "flags": flags, "episodes": episodes, "n_runs": n_runs,
            "policies": list(policies), "trainable_policies": [t.name for t in trainables],
            "arrangements_run": ran, "complete": bool(folds) and len(ran) == len(folds)}


def _num(x: float | int | None) -> str:
    if x is None:
        return "undef"
    return f"{x:g}" if float(x).is_integer() else f"{x:.2f}"


def setting_label(r: CheckRow) -> str:
    base, _ = split_policy(r.policy)
    if base == "S":
        return "1 view (S: no budget, no cap)"
    if base == "FULL":
        return "all 5 views (FULL: b = 4, uncapped)"
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

def run_selection(dev: DevData, settings: SelectionSettings, context: RunContext,
                  trainables: Sequence[FoldTrainableFactory] = ()) -> dict[str, Any]:
    """select-v4 on dev; returns everything the report and the frozen candidate need.

    The order is the guarantee: Steps 1-3a read views only and fix every threshold set (the candidate's and every
    fold's) before any policy runs. The Step 4 / 5 PolicyRunner only ever receives the frozen thresholds (or, in
    Step 5, a copy with k or lam replaced); the held-out check gives each fold's runner that fold's thresholds. If
    Step 1 is infeasible on all dev arrangements, CalibrationInfeasible propagates before any fold or policy runs.
    trainables: factories of fold-trained policies for the held-out check (FoldTrainableFactory, e.g. a FoldTrainable
    class; ranker A later), each built fresh and trained in every fold's own process.
    """
    refuse_non_dev(dev.specs)
    heldout_policies(trainables)                          # a bad name fails before anything runs
    rows = view_rows(dev)
    arrs = dev.arrangements
    table = ViewTable(rows, settings)
    cal = calibrate(rows, settings, table=table)          # Steps 1-2: views only; raises CalibrationInfeasible
    th = cal.thresholds                                   # frozen from here on
    curves = threshold_curves(cal.choice)
    rates = rates_by_lighting(rows, th)
    gate_pass = gate_pass_by_lighting(rows, th)
    extra = alpha_warnings(rates, settings.alpha_clear, "Step 1 (calibration set)", REPORT_LIGHTINGS)
    gate_flags = gate_pass_flags(gate_pass, "Step 1 (chosen setting, all dev views)")
    folds = grouped_cv(rows, settings, arrs, table)       # Step 3a: views only, the same joint search per fold
    held = pool_folds(folds)
    bad = [f for f in folds if f.infeasible is not None]
    for f in bad:
        extra.append(f"Step 3: calibrating without {f.held_out} is infeasible ({f.infeasible.summary()}); that fold "
                     f"has no thresholds, no held-out rates and no held-out policy runs (no fallback), so the "
                     f"held-out estimate covers {len(folds) - len(bad)} of {len(folds)} arrangements and the "
                     "candidate is NOT eligible for freeze")
    for f in folds:
        if f.gate_pass_train is not None:
            gate_flags += gate_pass_flags(f.gate_pass_train, f"Step 3 fold without {f.held_out} (training views)")
            gate_flags += gate_pass_flags(f.gate_pass_held, f"Step 3 fold without {f.held_out} (held-out views)")
    extra += gate_flags
    if folds:
        extra += alpha_warnings(held, settings.alpha_clear, "Step 3 (held out)", (*REPORT_LIGHTINGS, POOLED))
    else:
        extra.append("Step 3: fewer than 2 dev arrangements; no grouped cross-validation, so the candidate is NOT "
                     "eligible for freeze")
    eligible, reasons = freeze_eligibility(folds, arrs)
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
    heldout = held_out_check(dev, settings, context, folds, trainables)   # Step 3b: each fold's own thresholds
    lam_check = check_lam_grid((LAM_FIXED, *settings.lam_sensitivity), dev.positions, lookahead=False)
    for w in lam_check.warnings:
        log.warning(w)
    st = stratification(dev.strata, dev.arrangements)
    if not st["stratified"]:
        extra.append(f"bootstrap: {st['note']}")
    return {"calibration": cal, "thresholds": th, "curves": curves, "view_rates": rates, "folds": folds,
            "gate_pass": gate_pass, "gate_pass_flags": gate_flags,
            "cv_held_out": held, "check": check, "flags": flags, "unchecked": unchecked,
            "all_within_limit": all_within_limit(flags, unchecked),
            "sensitivity": sensitivity, "lam_grid_check": lam_check, "held_out_check": heldout,
            "cv_complete": bool(folds) and not bad, "eligible_for_freeze": eligible, "ineligible_reasons": reasons,
            "no_diagnostic_view_share": no_diagnostic_shares(dev),
            "warnings": [*cal.warnings, *extra, *lam_check.warnings], "runner": runner,
            "n_view_rows": len(rows), "n_episode_runs": runner.n_runs, "n_heldout_runs": heldout["n_runs"],
            "n_run_settings": runner.n_settings}


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
    """provenance: for --synthetic runs, the generator parameters, n_arr and n_dev (review F4); None on real or sim
    data, where dev_evidence_sha256 and the input file hashes (inputs) identify the data."""
    cal: Calibration = res["calibration"]
    th: Thresholds = res["thresholds"]
    conf = settings.conf
    ho = res["held_out_check"]
    doc = {
        "schema": CANDIDATE_SCHEMA, "procedure_version": PROCEDURE_VERSION, "procedure_revision": PROCEDURE_REVISION,
        "status": "CANDIDATE - not frozen until the student signs FREEZE.md",
        "synthetic": dev.synthetic, "simulation": dev.simulation, "label": dev.label,
        "eligible_for_freeze": res["eligible_for_freeze"], "ineligible_reasons": res["ineligible_reasons"],
        "eligibility_rule": ELIGIBILITY_RULE,
        "settings": settings.to_dict(), "settings_sha256": settings_sha, "proposed_defaults_note": PROPOSED,
        "dev_arrangements": list(dev.arrangements), "n_dev_episodes": len(dev.specs), "n_dev_views": res["n_view_rows"],
        "dev_evidence_sha256": dev.evidence_sha256, "view_labels_sha256": dev.view_labels_sha256,
        "view_label_source": dev.label_source, "inputs": dev.inputs,
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
            "search": JOINT_SEARCH, "objective": JOINT_OBJECTIVE, "constraints": [GATE_CONSTRAINT, MATCH_CONSTRAINT],
            "tie_break": TIE_BREAK, "order": FULL_ORDER,
            **cal.choice.to_dict(),
            "curves": res["curves"],
            "per_view_rates": {k: v.to_dict(conf) for k, v in res["view_rates"].items()},
            "gate_pass_by_lighting": {
                "note": ("share of views passing the tau_q / tau_occ gate (StoppingRule.passes_gate) at the chosen "
                         "setting, per lighting; feasible thresholds can reject a whole lighting at the gate, so this "
                         f"is reported, and a share at or below {GATE_PASS_LOW:g} is flagged. Per fold: "
                         "grouped_cv.folds[].gate_pass_train / gate_pass_held. Held-out recall and coverage per "
                         "lighting remain the evidence of lighting robustness."),
                **_gate_pass_dict(res["gate_pass"])},
            "gate_pass_flags": res["gate_pass_flags"],
            "grouped_cv": {
                "complete": res["cv_complete"],
                "infeasible_folds": [f.held_out for f in res["folds"] if f.infeasible is not None],
                "held_out_per_view_rates": {k: v.to_dict(conf) for k, v in res["cv_held_out"].items()},
                "folds": [f.to_dict() for f in res["folds"]],
                "n_distinct_threshold_sets": len({f.thresholds for f in res["folds"] if f.thresholds is not None})},
        },
        "held_out_check": {
            "note": ("each fold's thresholds, calibrated without the held-out arrangement, run every policy on that "
                     "arrangement's episodes; outcomes pooled over the folds. Rates above the limit are reported "
                     "checks: they never touch eligibility or any threshold. Rows for L1 and L2 queries are the "
                     "result; 'pooled' is a secondary line."),
            "policies": ho["policies"], "trainable_policies": ho["trainable_policies"],
            "primary_budget": settings.budget, "cap_steps": list(settings.cap_settings),
            "complete": ho["complete"], "arrangements_run": ho["arrangements_run"], "folds": ho["folds"],
            "flags": ho["flags"], "rows": [r.to_dict() for r in ho["rows"]]},
        "no_diagnostic_view_share": {"note": f"geometric no-diagnostic-view share of the dev episodes: {NO_DIAG_CONTEXT}",
                                     "source": dev.label_source, **res["no_diagnostic_view_share"]},
        "bootstrap_stratification": stratification(dev.strata, dev.arrangements),
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
                     "'unchecked'): the check is then incomplete. The rows are resubstitution (the arrangements whose "
                     "views set the thresholds); the held-out check is the honest estimate."),
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


# ---------------------------------------------------------------- the freeze step's gate

class NotEligibleForFreeze(RuntimeError):
    """require_freeze_eligible refused a candidate."""


def require_freeze_eligible(candidate: str | Path) -> dict[str, Any]:
    """What the freeze step calls before anything is frozen. It takes the candidate FILE only: the file must match
    its .sha256 sidecar (written by write_outputs). A parsed document is refused, because it cannot be checked against
    the sidecar (review, checkpoint 2: a dict skipped the tamper check, so an edited tau_q was accepted). Refuses
    (NotEligibleForFreeze) a candidate that is not this schema / procedure, is unmarked (no eligible_for_freeze), is
    marked anything but exactly true, or is synthetic by ANY of its marks: the "synthetic" flag not exactly false, a
    SYNTHETIC label, or a synthetic_provenance record (a flipped and re-signed flag must not pass while the label still
    says SYNTHETIC). Returns the document."""
    if not isinstance(candidate, (str, os.PathLike)):
        raise NotEligibleForFreeze(f"require_freeze_eligible takes the candidate file's path, not a "
                                   f"{type(candidate).__name__}: only a file can be checked against its .sha256 "
                                   "sidecar")
    path = Path(candidate)
    text = path.read_text()
    side = path.with_name(path.name + ".sha256")
    if not side.is_file():
        raise NotEligibleForFreeze(f"{path}: no {side.name} next to it; refusing an unverifiable candidate")
    want = side.read_text().split()[0] if side.read_text().split() else ""
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != want:
        raise NotEligibleForFreeze(f"{path}: sha256 {got} does not match {side.name} ({want})")
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise NotEligibleForFreeze(f"{path}: not JSON ({e})") from e
    if not isinstance(doc, dict):
        raise NotEligibleForFreeze(f"{path}: not a candidate document (a JSON {type(doc).__name__})")
    if (doc.get("schema"), doc.get("procedure_version"), doc.get("procedure_revision")) != \
            (CANDIDATE_SCHEMA, PROCEDURE_VERSION, PROCEDURE_REVISION):
        raise NotEligibleForFreeze(f"candidate schema/procedure/revision {doc.get('schema')!r}/"
                                   f"{doc.get('procedure_version')!r}/{doc.get('procedure_revision')!r} is not "
                                   f"{CANDIDATE_SCHEMA!r}/{PROCEDURE_VERSION!r}/{PROCEDURE_REVISION!r}")
    if "eligible_for_freeze" not in doc:
        raise NotEligibleForFreeze("candidate is unmarked: no eligible_for_freeze field")
    if doc["eligible_for_freeze"] is not True:
        why = "; ".join(r.get("reason", "?") for r in doc.get("ineligible_reasons") or []) or "no reason recorded"
        raise NotEligibleForFreeze(f"candidate is not eligible for freeze (eligible_for_freeze = "
                                   f"{doc['eligible_for_freeze']!r}): {why}")
    marks = [m for m, hit in (("synthetic is not exactly false", doc.get("synthetic") is not False),
                              (f"label {doc.get('label')!r}", "SYNTHETIC" in str(doc.get("label") or "").upper()),
                              ("it carries synthetic_provenance", "synthetic_provenance" in doc)) if hit]
    if marks:
        raise NotEligibleForFreeze("a synthetic candidate is never frozen (its numbers are not thresholds): "
                                   + "; ".join(marks))
    return doc


# ---------------------------------------------------------------- report

def _f(x: float | None, nd: int = 3) -> str:
    return "undef" if x is None else f"{x:.{nd}f}"


def _xn(xn: Sequence[Any]) -> str:
    return f"{_num(xn[0])}/{xn[1]}"


def _rate_xn(xn: Sequence[Any]) -> str:
    return f"{_f(_ratio((xn[0], xn[1])))} ({_xn(xn)})"


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
    # a row whose strata had a singleton fell back to the unstratified bootstrap: say so next to the interval
    return f"[{_f(lo)}, {_f(hi)}]" + (" (unstratified: singleton strata)" if boot.get("singleton_strata") else "")


def _steps(steps: float) -> str:
    return f"{steps:g} step" + ("" if steps == 1 else "s")


def _cap_cell(r: CheckRow) -> str:
    base = split_policy(r.policy)[0]
    if base == "S":
        return "n/a (1 view)"
    if base == "FULL":
        return "n/a (FULL, uncapped)"
    return "uncapped" if r.cap_steps is None else f"{_steps(r.cap_steps)} ({r.travel_cap_cm:.1f} cm)"


def _arr_frr(c: Counts) -> str:
    if c.n_arr_with_false_removal is None:
        return "n/a (R)"
    return f"{_f(c.frr_arrangement)} ({c.n_arr_with_false_removal}/{c.n_arr_with_present})"


def _check_table(rows: Sequence[CheckRow], conf: float) -> list[str]:
    pct = f"{conf:.0%}"
    L = [f"| policy | travel cap | FRR (x/n) | FRR {pct} upper, episode: CP / Jeffreys | FRR per arrangement (x/n) "
         f"| FRR {pct} upper, arrangement: CP / Jeffreys | FRR bootstrap 95% CI | recall (x/n) "
         f"| recall {pct} lower: CP / Jeffreys | recall bootstrap 95% CI | coverage (x/n) "
         f"| coverage {pct} lower: CP / Jeffreys | coverage bootstrap 95% CI | mean views | mean travel cm "
         "| FRR > limit |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        c, b = r.counts, r.bounds
        fe = _bound(b["frr_episode"] if b else None, "upper")
        fa = _bound(b["frr_arrangement"] if b else None, "upper")
        flag = "undef" if r.over_limit is None else ("**FLAG**" if r.over_limit else "no")
        L.append(f"| {r.policy} | {_cap_cell(r)} | {_f(c.frr)} ({_num(c.n_false_removal)}/{c.n_present}) | {fe} "
                 f"| {_arr_frr(c)} | {fa} "
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


def _curve_lines(curves: Mapping[str, Mapping[str, Any]]) -> list[str]:
    L: list[str] = []
    for axis in CURVE_AXES:
        c = curves[axis]
        fixed = _setting_text(c["held_fixed"])
        flat = [name for name, key in (("correct clear-miss", "correct_clear_miss_flat"),
                                       ("false clear-miss", "false_clear_miss_flat")) if c[key]]
        L += [f"### {axis} curve ({fixed} held at the chosen values)", "",
              f"| {axis} | correct clear-miss rate on removed (x/n) | false clear-miss rate (x/n) | false-clear rate "
              "(x/n) | both constraints met | |", "| --- | --- | --- | --- | --- | --- |"]
        for p in c["points"]:
            L.append(f"| {p['value']:g} | {_rate_xn(p['correct_clear_miss'])} | {_rate_xn(p['false_clear_miss'])} "
                     f"| {_rate_xn(p['false_clear'])} | {'yes' if p['feasible'] else 'no'} "
                     f"| {'**chosen**' if p['chosen'] else ''} |")
        L += ["", "- " + ("; ".join(f"the {n} curve is **flat** over this grid" for n in flat) + " (reported as flat; "
                          "no variation is forced)" if flat else "neither curve is flat over this grid")
              + ("; the declared tie-break decided the choice along this curve (another feasible point has the same "
                 "correct clear-miss count)" if c["tie_break_decided_on_curve"] else
                 "; the objective, not the tie-break, decided along this curve"), ""]
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


def _strat_text(dev: DevData) -> str:
    st = stratification(dev.strata, dev.arrangements)
    classes = ", ".join(f"{k} {v}" for k, v in st["strata"].items())
    if st["stratified"]:
        return f"stratified by quota slot class ({classes} arrangements): each replicate keeps every class's count"
    if st["singleton_strata"]:
        return f"**UNSTRATIFIED** ({classes} dev arrangements): {st['note']}"
    return f"**UNSTRATIFIED**: {st['note']}"


def _label_header(dev: DevData) -> list[str]:
    if dev.synthetic:
        return [f"# {SYNTHETIC_LABEL}", "",
                "**SYNTHETIC - not study results and not threshold values.** The evidence and the per-view "
                "diagnostic labels come from `verify.synthetic` (labels are a stand-in derived from the "
                "generator's hidden occluder layout). This run only exercises the calibration procedure; "
                "nothing here may be carried into FREEZE.md.", ""]
    if dev.simulation:
        return [f"# {SIMULATION_LABEL}", "",
                "**SIMULATION.** The dev evidence comes from `perception/evidence.py` on Isaac Sim renders and the "
                "per-view labels are geometric labels from the renderer's instance masks. Every number here is a "
                "simulation result.", ""]
    return []


def render_report(dev: DevData, settings: SelectionSettings, res: dict[str, Any], cand: dict[str, Any],
                  cand_sha: str) -> str:
    cal: Calibration = res["calibration"]
    th: Thresholds = res["thresholds"]
    ch = cal.choice
    conf = settings.conf
    L: list[str] = _label_header(dev)
    flags, unchecked = res["flags"], res["unchecked"]
    within = {True: "true", False: "false", None: "null"}[res["all_within_limit"]]
    elig = ("**eligible_for_freeze: true** (calibration succeeded on all dev data and in every planned CV fold)"
            if res["eligible_for_freeze"] else
            "**eligible_for_freeze: FALSE** (" + "; ".join(r["reason"] for r in res["ineligible_reasons"])
            + "); the candidate is saved for investigation only and the CLI exits with status "
            f"{EXIT_INELIGIBLE}")
    L += ["# Threshold calibration report (D-select, select-v4)", "",
          f"- procedure `{PROCEDURE_VERSION}`; settings sha256 `{cand['settings_sha256']}`",
          f"- candidate `frozen_candidate.json` sha256 `{cand_sha}` (status: candidate, not frozen)",
          f"- freeze: {elig}",
          f"- dev: {len(dev.arrangements)} arrangements, {len(dev.specs)} episodes, {res['n_view_rows']} views; "
          f"evidence sha256 `{dev.evidence_sha256}`",
          f"- view labels: {dev.label_source}; labels sha256 `{dev.view_labels_sha256}`"]
    if dev.inputs:
        L.append("- input files: " + "; ".join(f"{k} `{v.get('file')}` sha256 `{v['sha256']}`" for k, v in
                                               dev.inputs.items() if isinstance(v, Mapping) and "sha256" in v))
        ign = dev.inputs.get("ignored_episodes")
        if ign and (ign["evidence"]["n"] or ign["view_labels"]["n"]):
            L.append(f"- ignored (not in the manifest, never read): {ign['evidence']['n']} evidence episodes "
                     f"({', '.join(ign['evidence']['arrangements'])}), {ign['view_labels']['n']} label episodes "
                     f"({', '.join(ign['view_labels']['arrangements'])})")
    L += ["- **No policy result set any threshold.** Steps 1-3a read the dev views and their labels only, and every "
          f"threshold set (the candidate's and each fold's) was fixed before the first policy ran. The "
          f"{res['n_episode_runs']} dev-check replays ({res['n_run_settings']} run settings) and the "
          f"{res['n_heldout_runs']} held-out replays each ran on a fresh EvidenceStore through ReplayEnv/run_episode; "
          "they are reported, never fed back.",
          "- grouped cross-validation (Step 3): " + (
              "complete" if res["cv_complete"] else "**INCOMPLETE** (" + (
                  "infeasible without " + ", ".join(f.held_out for f in res["folds"] if f.infeasible is not None)
                  if res["folds"] else "fewer than 2 dev arrangements") + "; see Step 3), so the candidate is NOT "
              "eligible for freeze"),
          f"- dev check (Step 4): {len(flags)} flag(s) above FRR {settings.frr_limit:g}"
          + (f"; **INCOMPLETE**: FRR undefined in {len(unchecked)} L1 / L2 row(s), which were not checked"
             if unchecked else "") + f"; `all_within_limit` {within}",
          f"- cluster bootstrap: {_strat_text(dev)}", "",
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
    c = ch.chosen
    decided = ("the objective alone (a unique maximum of correct clear misses)" if not ch.tie_break_decided else
               f"the declared tie-break, at the level '{ch.decided_by}' ({ch.n_tied_at_max} feasible settings tied "
               "on correct clear misses)")
    L += ["", "## Step 1: joint gate + tau_match search over the predeclared grid", "",
          f"One Neyman-Pearson search over tau_q x tau_occ x tau_match: {JOINT_OBJECTIVE}, subject to "
          f"'{GATE_CONSTRAINT}' and '{MATCH_CONSTRAINT}'. Every count is the rule's own `StoppingRule.passes_gate` / "
          "`is_clear_miss` / `is_strong_match`.", "",
          f"- complete declared total order (ties never fall to iteration order): {FULL_ORDER}; the tie-break leans "
          "toward the conservative setting",
          f"- {ch.n_feasible} of {ch.n_grid} grid settings meet both constraints ({ch.n_meeting_false_clear} meet the "
          f"false-clear constraint, {ch.n_meeting_false_clear_miss} the false clear-miss constraint)",
          f"- chosen: tau_q {c.tau_q:g}, tau_occ {c.tau_occ:g}, tau_match {c.tau_match:g}; correct clear misses "
          f"{_rate_xn(c.clear_miss)}, false clear-miss rate {_rate_xn(c.false_clear_miss)}, false-clear rate "
          f"{_rate_xn(c.false_clear)}, false strong matches on removed {_rate_xn(c.false_strong_match)}",
          f"- decided by: {decided}",
          f"- procedure revision: {PROCEDURE_REVISION}; {JOINT_SEARCH}.",
          "- Fail loudly: had no grid setting met both constraints, this run would have written no candidate, only "
          f"`{INFEASIBLE_FILE}` (which constraint binds, the best achievable rates), and exited with status "
          f"{EXIT_INFEASIBLE}; alpha_clear is never relaxed, the grid never widened and no fallback setting is used.",
          "", "## Threshold curves (each threshold over its grid, the other two at the chosen values)", ""]
    L += _curve_lines(res["curves"])
    gp = res["gate_pass"]
    L += ["### Gate pass rates per lighting at the chosen setting (all dev views)", "",
          "Share of views passing the tau_q / tau_occ gate. Thresholds can meet both constraints by rejecting a whole "
          "lighting at the gate, so feasibility alone does not show lighting robustness; the held-out recall and "
          "coverage per lighting (Step 3b) are the evidence.", "",
          "| lighting | removed diagnostic views passing (x/n) | present diagnostic views passing (x/n) "
          "| all views passing (x/n) |", "| --- | --- | --- | --- |",
          *(f"| {k} | {_rate_xn(gp[k]['removed_diagnostic'])} | {_rate_xn(gp[k]['present_diagnostic'])} "
            f"| {_rate_xn(gp[k]['all'])} |" for k in (*REPORT_LIGHTINGS, POOLED)), ""]
    gflags = res["gate_pass_flags"]
    L += [f"- **GATE FLAG** {w}" for w in gflags] or [
        f"- in no lighting does (nearly) no view pass the gate (share <= {GATE_PASS_LOW:g}; removed diagnostic, "
        "present diagnostic and all views), and no population is empty, at the chosen setting or in any fold"]
    L += ["", "### Per-view rates at the frozen thresholds, by lighting (resubstitution: the views that chose them)", ""]
    L += _view_table(res["view_rates"], conf, lights)
    L += ["", "Views of one arrangement are correlated, so the view-level CP bounds are optimistic.", ""]
    nd = res["no_diagnostic_view_share"]
    L += ["### Geometric no-diagnostic-view share of the dev episodes (context, not a ceiling)", "",
          "| lighting | episodes with no diagnostic stop (x/n) |", "| --- | --- |",
          *(f"| {k} | {_f(nd[k]['share'])} ({nd[k]['x']}/{nd[k]['n']}) |" for k in (*REPORT_LIGHTINGS, POOLED)), "",
          f"- It is {NO_DIAG_CONTEXT}. Source: {dev.label_source}.", ""]
    st1 = [w for w in res["warnings"] if w.startswith("Step 1") and w not in gflags]
    if st1:
        L += ["### Step 1 warnings", "", *[f"- {w}" for w in st1], ""]
    L += ["## Step 3: grouped cross-validation (leave one arrangement out)", "",
          "Each fold runs the same joint search on the other arrangements' views. (a) It scores the held-out "
          "arrangement's views; the pooled held-out per-view rates are the honest per-view estimate. (b) The "
          "held-out policy check runs every policy on the held-out arrangement with that fold's thresholds.", ""]
    folds: list[ViewFold] = res["folds"]
    if folds:
        bad = [f for f in folds if f.infeasible is not None]
        if bad:
            L += [f"**INCOMPLETE: calibrating without {', '.join(f.held_out for f in bad)} is infeasible.** Those folds "
                  "have no thresholds, no held-out rates and no held-out policy runs (no fallback setting), so the "
                  f"held-out estimates cover {len(folds) - len(bad)} of {len(folds)} arrangements, and the candidate "
                  f"is marked `eligible_for_freeze: false` (exit status {EXIT_INELIGIBLE}).", ""]
        L += ["| held out | tau_q | tau_occ | tau_match | false clear (x/n) | clear (x/n) | false clear miss (x/n) "
              "| clear miss on removed (x/n) | gate pass, training removed diag. L1 / L2 "
              "| gate pass, training present diag. L1 / L2 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for f in folds:
            if f.thresholds is None or f.held is None:
                L.append(f"| {f.held_out} | **infeasible** (binding: "
                         f"{f.infeasible.binding if f.infeasible else '?'}): "
                         f"{f.infeasible.summary() if f.infeasible else ''} | - | - | - | - | - | - | - | - |")
                continue
            t, h, g = f.thresholds, f.held[POOLED], f.gate_pass_train
            gcell = (lambda pop: " / ".join(_xn(g[x][pop]) for x in REPORT_LIGHTINGS)) if g else (lambda pop: "-")
            L.append(f"| {f.held_out} | {t.tau_q} | {t.tau_occ} | {t.tau_match} | {_xn(h.false_clear)} | {_xn(h.clear)} "
                     f"| {_xn(h.false_clear_miss)} | {_xn(h.clear_miss)} | {gcell('removed_diagnostic')} "
                     f"| {gcell('present_diagnostic')} |")
        n_sets = len({f.thresholds for f in folds if f.thresholds is not None})
        L += ["", f"Calibrations differ across folds: {n_sets} distinct threshold sets over "
              f"{len(folds) - len(bad)} feasible folds.", "", "### (a) Held-out per-view rates (honest estimate)", ""]
        L += _view_table(res["cv_held_out"], conf, [("L1", "L1 views"), ("L2", "L2 views"),
                                                     ("pooled", "L1 + L2 views")])
        st3 = [w for w in res["warnings"] if w.startswith("Step 3") and w not in gflags]
        L += ["", *[f"- {w}" for w in st3], ""] if st3 else [""]
    else:
        L += ["- fewer than 2 dev arrangements: no grouped cross-validation, so the candidate is marked "
              f"`eligible_for_freeze: false` (exit status {EXIT_INELIGIBLE}).", ""]
    ho = res["held_out_check"]
    L += ["### (b) Held-out policy check (each fold's thresholds on its held-out arrangement)", "",
          f"Policies: {', '.join(ho['policies'])}; at b = {settings.budget}, {_caps_text(settings)} (S one view, FULL "
          "all five, uncapped). Fold-trained policies: "
          + (", ".join(ho["trainable_policies"]) or "none yet (ranker A plugs in through the FoldTrainable hook)")
          + f". Outcomes are pooled over the {len(ho['arrangements_run'])} folds that ran"
          + ("" if ho["complete"] else " (**INCOMPLETE**: infeasible folds did not run)")
          + f". Rates above FRR {settings.frr_limit:g} are reported checks: they never touch eligibility or a "
          "threshold.", ""]
    L += [f"- **FLAG** {f}" for f in ho["flags"]]
    for light in REPORT_LIGHTINGS:
        L += ["", f"#### Held out, {light} queries", ""]
        L += _check_table([r for r in ho["rows"] if r.lighting == light], conf)
    check: list[CheckRow] = res["check"]
    caps = "uncapped and under every cap" if settings.cap_steps else "uncapped only (no capped setting was run)"
    L += ["", f"## Step 4: dev check at the frozen thresholds (FRR <= {settings.frr_limit:g} is a check, not an "
          "objective)", "",
          f"Every policy at k = {th.k}, lam = {th.lam:g}, primary budget b = {settings.budget}, {caps} (caps are an "
          "operating condition reported as curves, never tuned). S is one view with its "
          "k = 1 rule; G is primary G (lookahead off), G-lookahead the same scorer with the lookahead; G-myopic "
          f"keeps its legacy weight ({GMYOPIC_LAM_DEG:g} deg). R is averaged over {settings.r_seeds} seeds within "
          "each episode, so it has no binomial bounds. A flag is reported and never fixed: changing a threshold to "
          "clear it would tune the thresholds on policy results. These rows are resubstitution; Step 3b is the "
          "held-out estimate.", ""]
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
    L += ["", "## Secondary, not a headline: pooled over L1 and L2 queries", "",
          f"### Dev check (b = {settings.budget})", "",
          "| policy | travel cap | FRR (x/n) | FRR per arrangement (x/n) | recall (x/n) | coverage (x/n) "
          "| mean travel cm |", "| --- | --- | --- | --- | --- | --- | --- |"]
    pooled_line = (lambda r: f"| {r.policy} | {_cap_cell(r)} | {_f(r.counts.frr)} ({_num(r.counts.n_false_removal)}/"
                   f"{r.counts.n_present}) | {_arr_frr(r.counts)} | {_f(r.counts.recall)} "
                   f"({_num(r.counts.n_true_removal)}/{r.counts.n_removed}) | {_f(r.counts.coverage)} "
                   f"({_num(r.counts.n_resolved)}/{r.counts.n_episodes}) | {_f(r.counts.mean_travel_cm, 1)} |")
    L += [pooled_line(r) for r in check if r.lighting == POOLED]
    L += ["", "### Held-out policy check", "",
          "| policy | travel cap | FRR (x/n) | FRR per arrangement (x/n) | recall (x/n) | coverage (x/n) "
          "| mean travel cm |", "| --- | --- | --- | --- | --- | --- | --- |"]
    L += [pooled_line(r) for r in ho["rows"] if r.lighting == POOLED]
    L += ["", "## Reading these numbers", "",
          "- Views and episodes of one arrangement share an occluder and lighting, so they are not independent: "
          "view- and episode-level bounds are optimistic, and the arrangement-level FRR bound is the more honest one.",
          "- Quote the Step 3 held-out rates (per view and per policy), not the resubstitution rates of Steps 1 and 4.",
          f"- alpha_clear = {settings.alpha_clear:g} constrains per-view rates on the dev views only: it does not "
          "guarantee that any policy's FRR is at most that value, and needing k = 2 clear-miss stops does not turn it "
          "into alpha squared. The FRR of each policy is only checked (Steps 3b and 4).",
          frr_bound_note(len(dev.arrangements), settings.frr_limit, conf),
          "- select-v3 (replaced) chose (k, tau_match, lam) by running primary G on dev and maximising its coverage "
          "after a fixed-sequence FRR walk (D10). That tuned shared settings on G's own results, contrary to "
          "DECISIONS 2026-10-12, and would have handed G-tuned settings to ranker A.", ""]
    other = [w for w in res["warnings"] if not w.startswith(("Step 1", "Step 3")) and w not in lc.warnings]
    if other:
        L += ["### Other warnings", "", *[f"- {w}" for w in other], ""]
    if dev.label:
        L += [f"_{dev.label}._", ""]
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
    p.add_argument("--manifest", default=None,
                   help="sim/real dev data: manifest.json (every arrangement must be split dev); needs --evidence, "
                        "--view-labels and --positions")
    p.add_argument("--evidence", default=None, help="evidence.jsonl from perception/evidence.py (jobs_refL1.json)")
    p.add_argument("--view-labels", default=None,
                   help=f"view_labels.json ({SIM_VIEW_LABELS_SCHEMA}): the geometric per-view labels")
    p.add_argument("--model-id", default="perception.evidence-v0", help="sim/real data: perception model id (logs)")
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
                        "(default 0.10); not a bound on any policy's FRR. If no setting of the joint grid meets both, "
                        f"calibration fails: no candidate, {INFEASIBLE_FILE}, exit status {EXIT_INFEASIBLE}")
    p.add_argument("--frr-limit", type=float, default=d.frr_limit,
                   help="FRR limit of the dev check: a check that flags, never an objective (default 0.10)")
    p.add_argument("--k-sensitivity", type=_optional(_ints, "int list"), default=d.k_sensitivity,
                   help="k values run as a reported sensitivity only (default 3; 'none'); k = 2 is fixed")
    p.add_argument("--lam-sensitivity", type=_optional(_floats, "float list"), default=d.lam_sensitivity,
                   help="lam values run as a reported sensitivity only (default 0.7,0.9,1.2; 'none'); lam = 0 is fixed")
    p.add_argument("--bootstrap-B", type=int, default=d.bootstrap_B)
    p.add_argument("--positions", default=None,
                   help="positions JSON (default configs/positions.json; required with --manifest)")
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
        for cap in policy_caps(p, settings):
            budget = run_budget(p, cap, settings, runner.dev.positions)
            for r in runner.results(p, th, budget):
                out.append({"policy": p, "cap_steps": cap, "travel_cap_cm": budget.travel_cap_cm,
                            "episode_id": r.episode_id, "lighting": r.lighting.value,
                            "decision_frac": {d.value: r.frac(d) for d in Decision},
                            "mean_views": r.mean_views, "mean_travel_cm": r.mean_travel_cm})
    return out


def settings_sha256(settings: SelectionSettings, provenance: Mapping[str, Any] | None = None,
                    data: Mapping[str, Any] | None = None) -> str:
    """sha256 of the predeclared settings and the procedure; synthetic runs add the generator regime (F4), sim / real
    runs add data_provenance (the simulation flag and the input files' sha256)."""
    hashed: dict[str, Any] = {"settings": settings.to_dict(), "procedure": PROCEDURE_VERSION,
                              "revision": PROCEDURE_REVISION}
    if provenance is not None:   # synthetic runs: the generator regime is part of what was calibrated on (F4)
        hashed["synthetic_provenance"] = to_jsonable(dict(provenance))
    if data is not None:
        hashed["data_provenance"] = to_jsonable(dict(data))
    return config_sha256(hashed)


def data_provenance(dev: DevData) -> dict[str, Any] | None:
    """What the settings hash records about sim / real dev data: the simulation flag and the input file hashes."""
    if dev.synthetic:
        return None
    return {"simulation": dev.simulation,
            "inputs": {k: v["sha256"] for k, v in (dev.inputs or {}).items() if isinstance(v, Mapping) and "sha256" in v},
            "dev_evidence_sha256": dev.evidence_sha256, "view_labels_sha256": dev.view_labels_sha256}


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
    sha_settings = settings_sha256(settings, provenance, data_provenance(dev))
    cand = candidate_doc(dev, settings, res, sha_settings, provenance)
    text = json.dumps(cand, indent=1, sort_keys=True) + "\n"
    sha = hashlib.sha256(text.encode()).hexdigest()
    cal: Calibration = res["calibration"]
    alpha = settings.alpha_clear
    details = {"label": dev.label, "procedure_version": PROCEDURE_VERSION,
               "settings": settings.to_dict(),
               "joint_grid": [p.to_dict(alpha) for p in cal.choice.points],
               "curves": res["curves"],
               "folds": [f.to_dict() for f in res["folds"]],
               "check_episodes": check_episodes(res["runner"], res["thresholds"], settings),
               "held_out_episodes": res["held_out_check"]["episodes"]}
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
    sha_settings = settings_sha256(settings, provenance, data_provenance(dev))
    doc: dict[str, Any] = {
        "schema": INFEASIBLE_SCHEMA, "status": "infeasible", "procedure_version": PROCEDURE_VERSION,
        "procedure_revision": PROCEDURE_REVISION,
        "synthetic": dev.synthetic, "simulation": dev.simulation, "label": dev.label,
        "no_candidate": (f"no frozen_candidate.json was written and any earlier one in this directory was removed; "
                         f"{NO_FALLBACK}"),
        "failure": failure.to_dict(),
        "alpha_clear": settings.alpha_clear, "alpha_note": ALPHA_NOTE,
        "settings": settings.to_dict(), "settings_sha256": sha_settings, "proposed_defaults_note": PROPOSED,
        "dev_arrangements": list(dev.arrangements), "n_dev_episodes": len(dev.specs),
        "dev_evidence_sha256": dev.evidence_sha256, "view_labels_sha256": dev.view_labels_sha256,
        "view_label_source": dev.label_source, "inputs": dev.inputs,
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
    L: list[str] = _label_header(dev)
    if dev.synthetic:
        L[2] = ("**SYNTHETIC - not study results and not threshold values.** This run only exercises the calibration "
                "procedure.")
    bt = Infeasibility._best_text
    L += ["# Threshold calibration report (D-select, select-v4): INFEASIBLE", "",
          "**No frozen candidate was written.** The joint search found no setting on the predeclared grid (tau_q x "
          f"tau_occ x tau_match, searched jointly) that meets both alpha_clear = {failure.alpha_clear:g} constraints, "
          f"and {NO_FALLBACK}. Any earlier `frozen_candidate.json` in this directory was removed. The failure record "
          f"is `{INFEASIBLE_FILE}` (status \"infeasible\"); the CLI exits with status {EXIT_INFEASIBLE}.", "",
          f"- procedure `{PROCEDURE_VERSION}`; settings sha256 `{sha_settings}`",
          f"- dev: {len(dev.arrangements)} arrangements, {len(dev.specs)} episodes; evidence sha256 "
          f"`{dev.evidence_sha256}`; view labels: {dev.label_source}; labels sha256 `{dev.view_labels_sha256}`",
          f"- **binding**: {failure.binding_text}",
          f"- constraints: '{GATE_CONSTRAINT}' (measured on {failure.n_gate_null_views} views) and "
          f"'{MATCH_CONSTRAINT}' (measured on {failure.n_match_null_views} views), dev views of both lightings",
          f"- best achievable false-clear rate over the grid: {bt(failure.best_false_clear)}",
          f"- best achievable false clear-miss rate over the grid: {bt(failure.best_false_clear_miss)}",
          "- best false clear-miss rate among settings meeting the false-clear constraint: "
          f"{bt(failure.best_false_clear_miss_meeting_false_clear)}",
          "- best false-clear rate among settings meeting the false clear-miss constraint: "
          f"{bt(failure.best_false_clear_meeting_false_clear_miss)}",
          "- grid searched (jointly): " + "; ".join(f"{k} {_grid_text(v)}" for k, v in failure.grid.items()),
          f"- alpha_clear = {failure.alpha_clear:g}: {ALPHA_NOTE}", "",
          "Changing alpha_clear or a grid is a change to the predeclared settings, for the student to decide and "
          "record; this procedure never does it.", "",
          "## Every setting searched", "",
          "| tau_q | tau_occ | tau_match | false-clear rate (x/n) | false clear-miss rate (x/n) |",
          "| --- | --- | --- | --- | --- |"]
    for g in failure.grid_rates:
        L.append(f"| {g['tau_q']:g} | {g['tau_occ']:g} | {g['tau_match']:g} | {_rate_xn(g['false_clear'])} "
                 f"| {_rate_xn(g['false_clear_miss'])} |")
    if dev.label:
        L += ["", f"_{dev.label}._"]
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
    files = (args.manifest, args.evidence, args.view_labels)
    if args.synthetic == any(x is not None for x in files):
        print("use either --synthetic, or --manifest with --evidence, --view-labels and --positions (sim/real dev "
              "data)", file=sys.stderr)
        return 2
    if not args.synthetic and (None in files or args.positions is None):
        print("--manifest needs --evidence, --view-labels and --positions", file=sys.stderr)
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
    provenance: dict[str, Any] | None = None
    if args.synthetic:
        positions = Positions.load(args.positions) if args.positions else Positions.load()
        params = SyntheticParams(n_dev=args.n_dev, block_includes_start=not args.block_anywhere)
        dev = synthetic_dev_data(args.n_arr, args.seed, params, positions)
        tag = args.tag or f"synthetic-n{args.n_arr}-s{args.seed}"
        provenance = {"generator": dataclasses.asdict(params), "n_arr": args.n_arr, "n_dev": len(dev.arrangements),
                      "seed": args.seed}
        context = RunContext(model_id="synthetic-no-model", config_version=PROCEDURE_VERSION,
                             config_sha256=settings_sha256(settings, provenance), synthetic=True)
    else:
        try:
            dev = sim_dev_data(args.manifest, args.evidence, args.view_labels, args.positions)
        except (OSError, ValueError, KeyError) as e:   # SplitRefused is a ManifestError, i.e. a ValueError
            print(f"bad dev input: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        tag = args.tag or f"{'sim' if dev.simulation else 'real'}-{Path(args.manifest).resolve().parent.name}"
        context = RunContext(model_id=args.model_id, config_version=PROCEDURE_VERSION,
                             config_sha256=settings_sha256(settings, None, data_provenance(dev)), synthetic=False)
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"{PROCEDURE_VERSION}-{tag}"
    who = dev.label or "REAL DEV DATA"
    try:
        res = run_selection(dev, settings, context)
    except CalibrationInfeasible as e:   # fail loudly: no candidate, a failure record, a non-zero exit
        path = write_infeasible(out, dev, settings, e.failure, provenance)
        print(f"{who}: CALIBRATION INFEASIBLE, no frozen candidate written. {e}", file=sys.stderr)
        print(f"{who}: failure record {path}; report {out / REPORT_FILE}", file=sys.stderr)
        return EXIT_INFEASIBLE
    shas = write_outputs(out, dev, settings, res, provenance)
    th = res["thresholds"]
    print(f"{who}: wrote {out} (frozen_candidate.json sha256 {shas['frozen_candidate.json']})")
    print(f"{who}: candidate tau_q {th.tau_q}, tau_occ {th.tau_occ}, tau_match {th.tau_match}, "
          f"k {th.k} (fixed), lam {th.lam:g} (fixed); dev check: {len(res['flags'])} flag(s) above FRR "
          f"{settings.frr_limit:g}" + (f", {len(res['unchecked'])} row(s) not checked (FRR undefined)"
                                       if res["unchecked"] else "")
          + f"; held-out check: {len(res['held_out_check']['flags'])} flag(s)")
    if not res["eligible_for_freeze"]:
        print(f"{who}: candidate NOT eligible for freeze: "
              + "; ".join(r["reason"] for r in res["ineligible_reasons"]), file=sys.stderr)
        return EXIT_INELIGIBLE
    print(f"{who}: eligible_for_freeze true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
