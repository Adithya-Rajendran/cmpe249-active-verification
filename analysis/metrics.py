"""Episode-level metrics (EXPERIMENT_PLAN 'Metrics and reporting').

Ground truth is joined here (and in analysis/bootstrap.py) and nowhere else. Decisions are
fractional: an EpisodeResult contributes frac(d) to every decision d, so R's seeds are
averaged within the episode before anything is counted. 'unknown' stays in the coverage
denominator. A metric whose denominator is 0 is None with defined[metric] = False.

Every metric is a ratio of per-episode sums (numerator, denominator), so the same terms serve
compute() on unique episodes and the bootstrap on a resampled multiset.

Context, not a ceiling (docs/SIM_PREREGISTRATION.md sections 3 and 9): geometric_no_diagnostic_view_share is the
share of episodes with no geometrically diagnostic stop. It is computed from the per-view geometric labels when
the caller passes them (no_diagnostic: episode_id -> True when no stop is diagnostic), and otherwise falls back to
the arrangement-level predeclared indistinguishable flag (labelled as such in no_diagnostic_view_source). It is
never a maximum coverage: a policy can resolve an episode with no diagnostic view (rightly or wrongly), and one
with diagnostic views can stay unknown. (It replaces max_achievable_coverage = 1 - that share, 2026-11-03.)

FRR per arrangement (section 8), from compute() only (unique episodes): the share of arrangements with a present
episode that have any false removal. With seed-averaged decisions (R) an arrangement contributes
1 - prod(1 - frac_removed) over its present episodes: R's seeds are salted per episode, so this is the probability
of any false removal in the arrangement with independently randomised episodes.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from verify.schema import (
    CONDITIONS,
    EPS,
    LIGHTINGS,
    SYNTHETIC_LABEL,
    Decision,
    EpisodeLog,
    EpisodeResult,
    GroundTruth,
    Split,
    TruthState,
    as_results,
)

METRICS = ("false_removal_rate", "removal_recall", "coverage", "resolved_accuracy",
           "mean_views", "mean_additional_views", "mean_travel_cm")
NO_DIAG_SOURCE_LABELS = "geometric view labels (episodes with no diagnostic stop)"
NO_DIAG_SOURCE_FLAG = ("predeclared indistinguishable flag (legacy fallback: arrangement-level, occluded and removed "
                       "episodes of flagged arrangements; no view labels given)")
GROUP_KEYS = {"condition": CONDITIONS, "lighting": LIGHTINGS, "split": tuple(Split)}
FRAC_TOL = 1e-6  # decision_frac must sum to 1 within this


def check_single_run(results: Sequence[EpisodeResult]) -> tuple[str, int, float | None]:
    """Results must share one (policy, budget, travel cap); returns it. ValueError otherwise."""
    if not results:
        raise ValueError("no results")
    keys = {(r.policy, r.budget_additional_views, r.travel_cap_cm) for r in results}
    if len(keys) != 1:
        raise ValueError(f"results mix policies/budgets/caps: {sorted(keys, key=str)}")
    return next(iter(keys))


def check_truth(results: Iterable[EpisodeResult], truth: GroundTruth) -> None:
    """KeyError if any episode has no truth; ValueError if a decision_frac doesn't sum to 1."""
    for r in results:
        if r.episode_id not in truth:
            raise KeyError(f"no ground truth for episode {r.episode_id!r}")
        total = sum(r.frac(d) for d in Decision)
        if abs(total - 1.0) > FRAC_TOL:
            raise ValueError(f"decision_frac of {r.episode_id!r} sums to {total}, not 1")


def episode_terms(r: EpisodeResult, truth: GroundTruth) -> dict[str, tuple[float, float]]:
    """(numerator, denominator) contribution of one episode to each metric in METRICS."""
    present = truth[r.episode_id] is TruthState.PRESENT
    removed_frac = r.frac(Decision.REMOVED)
    resolved = 1.0 - r.frac(Decision.UNKNOWN)
    correct = r.frac(Decision.PRESENT) if present else removed_frac
    return {
        "false_removal_rate": (removed_frac if present else 0.0, 1.0 if present else 0.0),
        "removal_recall": (0.0 if present else removed_frac, 0.0 if present else 1.0),
        "coverage": (resolved, 1.0),
        "resolved_accuracy": (correct, resolved),
        "mean_views": (float(r.mean_views), 1.0),
        "mean_additional_views": (float(r.mean_additional_views), 1.0),
        "mean_travel_cm": (float(r.mean_travel_cm), 1.0),
    }


def ratio(num: float, den: float) -> float | None:
    """num / den, or None when the denominator is (numerically) zero."""
    return float(num / den) if den > EPS else None


def metrics_on(results: Sequence[EpisodeResult], truth: GroundTruth,
               no_diagnostic: Mapping[str, bool] | None = None) -> dict[str, Any]:
    """Metrics on a multiset of results (duplicates count with multiplicity). No grouping. no_diagnostic: episode_id
    -> True when the episode has no geometrically diagnostic stop (from view labels); None falls back to the
    predeclared indistinguishable flag."""
    policy, budget, cap = check_single_run(results)
    check_truth(results, truth)
    sums = {m: [0.0, 0.0] for m in METRICS}
    confusion = {t.value: {d.value: 0.0 for d in Decision} for t in TruthState}
    n_present = n_removed = n_ind = n_nd = 0
    for r in results:
        for m, (num, den) in episode_terms(r, truth).items():
            sums[m][0] += num
            sums[m][1] += den
        t = truth[r.episode_id]
        n_present += t is TruthState.PRESENT
        n_removed += t is TruthState.REMOVED
        n_ind += truth.is_indistinguishable(r.episode_id)
        if no_diagnostic is not None:
            if r.episode_id not in no_diagnostic:
                raise KeyError(f"no view labels for episode {r.episode_id!r}")
            n_nd += bool(no_diagnostic[r.episode_id])
        for d in Decision:
            confusion[t.value][d.value] += r.frac(d)
    n = len(results)
    out: dict[str, Any] = {
        "policy": policy, "budget_additional_views": budget, "travel_cap_cm": cap,
        "n_arrangements": len({r.arrangement_id for r in results}), "n_episodes": n,
        "n_present": n_present, "n_removed": n_removed, "n_indistinguishable": n_ind,
    }
    for m in METRICS:
        out[m] = ratio(*sums[m])
    if no_diagnostic is None:
        n_nd = n_ind
    out["n_no_diagnostic_view"] = n_nd
    out["geometric_no_diagnostic_view_share"] = ratio(n_nd, n)
    out["no_diagnostic_view_source"] = NO_DIAG_SOURCE_FLAG if no_diagnostic is None else NO_DIAG_SOURCE_LABELS
    out["defined"] = {m: out[m] is not None for m in (*METRICS, "geometric_no_diagnostic_view_share")}
    out["confusion"] = confusion
    synthetic = truth.synthetic or any(r.synthetic for r in results)
    out["synthetic"] = synthetic
    if synthetic:
        out["label"] = SYNTHETIC_LABEL
    return out


def arrangement_frr(results: Sequence[EpisodeResult], truth: GroundTruth) -> dict[str, Any]:
    """FRR per arrangement on UNIQUE episodes: {n_arr_with_present, arrangement_false_removal,
    false_removal_rate_arrangement, arrangement_frr_binomial}. An arrangement with a present episode contributes
    1 - prod(1 - frac_removed) over its present episodes (0/1 for a deterministic policy: any false removal)."""
    keep: dict[str, float] = {}
    for r in results:
        if truth[r.episode_id] is TruthState.PRESENT:
            keep[r.arrangement_id] = keep.get(r.arrangement_id, 1.0) * (1.0 - r.frac(Decision.REMOVED))
    any_fr = {a: 1.0 - v for a, v in keep.items()}
    binomial = all(r.frac(Decision.REMOVED) in (0.0, 1.0) for r in results)
    total = sum(any_fr.values())
    if binomial:
        total = round(total)
    return {"n_arr_with_present": len(any_fr), "arrangement_false_removal": total,
            "false_removal_rate_arrangement": ratio(total, len(any_fr)), "arrangement_frr_binomial": binomial}


def _with_arrangement_frr(res: Sequence[EpisodeResult], truth: GroundTruth,
                          no_diagnostic: Mapping[str, bool] | None) -> dict[str, Any]:
    out = metrics_on(res, truth, no_diagnostic)
    out.update(arrangement_frr(res, truth))
    out["defined"]["false_removal_rate_arrangement"] = out["false_removal_rate_arrangement"] is not None
    return out


def compute(results: Iterable[EpisodeLog | EpisodeResult], truth: GroundTruth,
            group_by: str | None = None, no_diagnostic: Mapping[str, bool] | None = None) -> dict:
    """Metrics for one policy x budget; with group_by, {group value: metrics} in enum order. Unique episodes only,
    so the per-arrangement FRR is added here (arrangement_frr)."""
    res = as_results(results)
    check_single_run(res)
    ids = [r.episode_id for r in res]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode_id in results (one result per episode expected)")
    if group_by is None:
        return _with_arrangement_frr(res, truth, no_diagnostic)
    if group_by not in GROUP_KEYS:
        raise ValueError(f"group_by must be None or one of {sorted(GROUP_KEYS)}, got {group_by!r}")
    check_truth(res, truth)
    out: dict[str, dict] = {}
    for g in GROUP_KEYS[group_by]:
        sub = [r for r in res if getattr(r, group_by) == g]
        if sub:
            out[g.value] = _with_arrangement_frr(sub, truth, no_diagnostic)
    return out


def compute_all(results: Iterable[EpisodeLog | EpisodeResult], truth: GroundTruth,
                no_diagnostic: Mapping[str, bool] | None = None) -> dict:
    """{"overall", "by_condition", "by_lighting"} for one policy x budget."""
    res = as_results(results)
    return {"overall": compute(res, truth, no_diagnostic=no_diagnostic),
            "by_condition": compute(res, truth, "condition", no_diagnostic),
            "by_lighting": compute(res, truth, "lighting", no_diagnostic)}
