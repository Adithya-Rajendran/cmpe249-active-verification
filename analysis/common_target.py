"""The predeclared secondary contrast on common targets (docs/SIM_PREREGISTRATION.md section 9, 2026-11-03).

A - G restricted to the episodes whose target is one of COMMON_TARGETS: the foam brick and the tuna can, the two targets
every occluder family supports on the shelf (on the tabletop every family supports all five, so the restriction is the
same two targets there). Consistent target weighting: within each occluder family the targets carry equal weight, so a
difference between families is not a difference in their target mix. With N_ft drawn arrangements of family f and
target t, N_f their sum over t and T_f the number of common targets family f has in the draw,

    w(f, t) = N_f / (T_f * N_ft)        (each family keeps its weight N_f, split equally over its targets)

and a metric is sum(w * numerator) / sum(w * denominator) over the drawn episodes (analysis.metrics.episode_terms). The
weights are recomputed on every bootstrap replicate from that replicate's arrangements: the estimator is applied to the
replicate. The resampling is analysis.bootstrap's paired arrangement-level cluster bootstrap with the same strata rule
and the same draws as paired_diff for the same arrangements, B and seed. A family that has only one of the targets in
the sample is listed under `single_target_families`: its weight then rests on that target alone.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from analysis.bootstrap import _terms, pair, resolve_strata, sample_draws
from analysis.metrics import METRICS
from verify.schema import EPS, EpisodeLog, EpisodeResult, GroundTruth, as_results

COMMON_TARGETS = ("foam_brick", "tuna_fish_can")


def _weighted(counts: np.ndarray, cell: np.ndarray, family: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Per row of counts (B, n): the weighted ratio over arrangements (NaN where the denominator is 0)."""
    n_cells = int(cell.max()) + 1
    N_c = np.stack([counts[:, cell == c].sum(1) for c in range(n_cells)], 1)           # (B, cells)
    fam_of_cell = np.array([family[cell == c][0] for c in range(n_cells)])
    w_c = np.zeros_like(N_c)
    for f in np.unique(fam_of_cell):
        cs = fam_of_cell == f
        N_f = N_c[:, cs].sum(1, keepdims=True)
        T_f = (N_c[:, cs] > 0).sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            w_c[:, cs] = np.where(N_c[:, cs] > 0, N_f / np.maximum(T_f * N_c[:, cs], 1e-300), 0.0)
    W = w_c[:, cell] * counts                                                           # (B, n)
    a, b = W @ num, W @ den
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > EPS, a / np.where(b > EPS, b, 1.0), np.nan)


def common_target_diff(results_a: Iterable[EpisodeLog | EpisodeResult], results_b: Iterable[EpisodeLog | EpisodeResult],
                       truth: GroundTruth, family_of: Mapping[str, str], target_of: Mapping[str, str], metric: str,
                       B: int = 10000, seed: int = 0, strata: Mapping[str, str] | None = None,
                       targets: Sequence[str] = COMMON_TARGETS) -> dict[str, Any]:
    """metric(A) - metric(B) on the common-target episodes with consistent target weighting, and its percentile 95%
    paired cluster-bootstrap interval. family_of and target_of map arrangement ids."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if B < 1:
        raise ValueError("B must be >= 1")
    keep = set(targets)
    ra = [r for r in as_results(results_a) if target_of[r.arrangement_id] in keep]
    rb = [r for r in as_results(results_b) if target_of[r.arrangement_id] in keep]
    if not ra:
        raise ValueError("no episode with a common target")
    sample = pair(ra, rb, truth)
    arrs = sample.arrangement_ids
    fam = np.array([family_of[a] for a in arrs])
    tgt = [target_of[a] for a in arrs]
    keys = sorted({(f, t) for f, t in zip(fam, tgt)})
    cell = np.array([keys.index((f, t)) for f, t in zip(fam, tgt)])
    terms = [_terms(sample.groups, res, truth, metric) for res in (sample.results_a, sample.results_b)]
    one = np.ones((1, len(arrs)))
    est = [_weighted(one, cell, fam, n, d)[0] for n, d in terms]
    estimate = None if np.isnan(est[0]) or np.isnan(est[1]) else float(est[0] - est[1])
    used, singles, note = resolve_strata(arrs, strata)
    draws = sample_draws(arrs, B, seed, used)
    counts = np.zeros((B, len(arrs)))
    np.add.at(counts, (np.arange(B)[:, None], draws), 1.0)
    diffs = _weighted(counts, cell, fam, *terms[0]) - _weighted(counts, cell, fam, *terms[1])
    ok = diffs[~np.isnan(diffs)]
    lo = hi = None
    if ok.size:
        lo, hi = (float(x) for x in np.percentile(ok, [2.5, 97.5]))
    cells = {f"{f}|{t}": int((cell == keys.index((f, t))).sum()) for f, t in keys}
    single = sorted(f for f in set(fam) if sum(1 for ff, _ in keys if ff == f) < len(keep))
    return {"metric": metric, "policy_a": sample.results_a[0].policy, "policy_b": sample.results_b[0].policy,
            "targets": list(targets), "estimate": estimate, "ci_lo": lo, "ci_hi": hi,
            "inconclusive": None if lo is None else bool(lo <= 0.0 <= hi),
            "n_arrangements": len(arrs), "n_episodes": sample.n_episodes, "arrangements_per_family_target": cells,
            "single_target_families": single, "B": B, "seed": seed, "n_undefined": int(diffs.size - ok.size),
            "stratified": used is not None, "singleton_strata": list(singles), "stratification_note": note,
            "weighting": "within each family the common targets carry equal weight; weights recomputed per replicate"}
