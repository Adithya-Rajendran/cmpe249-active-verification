"""Paired, arrangement-level bootstrap for metric differences between two policies.

Each draw resamples the unique arrangement ids with replacement and takes all episodes of
every drawn arrangement (duplicates repeated), so an arrangement's episodes are never split.
Both policies are scored on the same multiset; the CI is the 2.5/97.5 percentile interval of
the defined A - B draws. Draws where either side is undefined are counted and excluded.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from analysis.metrics import (
    METRICS,
    check_single_run,
    check_truth,
    episode_terms,
    metrics_on,
)
from verify.schema import (
    EPS,
    SYNTHETIC_LABEL,
    EpisodeLog,
    EpisodeResult,
    GroundTruth,
    as_results,
)


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    policy_a: str
    policy_b: str
    estimate: float | None     # metric(a) - metric(b) on the full sample (None if undefined)
    ci_lo: float | None        # 2.5th percentile of defined draws (None if no draw is defined)
    ci_hi: float | None        # 97.5th percentile
    n_arrangements: int
    n_episodes: int
    B: int
    n_undefined: int           # draws where either side was undefined (excluded)
    seed: int
    synthetic: bool

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        if self.synthetic:
            d["label"] = SYNTHETIC_LABEL
        return d


@dataclass(frozen=True)
class PairedSample:
    """Two policies' results aligned on the same episodes, grouped by arrangement."""
    arrangement_ids: tuple[str, ...]          # sorted unique ids (the resampling unit)
    groups: tuple[tuple[int, ...], ...]       # per arrangement: indices into results_a/b
    results_a: tuple[EpisodeResult, ...]      # sorted by episode_id
    results_b: tuple[EpisodeResult, ...]      # same episode order as results_a

    @property
    def n_episodes(self) -> int:
        return len(self.results_a)

    def expand(self, draw: Sequence[int]) -> tuple[list[EpisodeResult], list[EpisodeResult]]:
        """The multiset of (A, B) results for one draw of arrangement indices."""
        idx = [i for a in draw for i in self.groups[a]]
        return [self.results_a[i] for i in idx], [self.results_b[i] for i in idx]


def pair(results_a: Iterable[EpisodeLog | EpisodeResult], results_b: Iterable[EpisodeLog | EpisodeResult],
         truth: GroundTruth) -> PairedSample:
    """Normalise and align two result sets; ValueError unless they cover the same episodes."""
    ra = sorted(as_results(results_a), key=lambda r: r.episode_id)
    rb = sorted(as_results(results_b), key=lambda r: r.episode_id)
    check_single_run(ra)
    check_single_run(rb)
    ids_a, ids_b = [r.episode_id for r in ra], [r.episode_id for r in rb]
    for ids in (ids_a, ids_b):
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate episode_id in results")
    if ids_a != ids_b:
        raise ValueError("policies must cover the same episode_id set for a paired bootstrap")
    for x, y in zip(ra, rb):
        if x.arrangement_id != y.arrangement_id:
            raise ValueError(f"arrangement mismatch for {x.episode_id!r}")
    check_truth(ra, truth)
    check_truth(rb, truth)
    arr_ids = tuple(sorted({r.arrangement_id for r in ra}))
    groups = tuple(tuple(i for i, r in enumerate(ra) if r.arrangement_id == a) for a in arr_ids)
    return PairedSample(arr_ids, groups, tuple(ra), tuple(rb))


def arrangement_draws(n_arrangements: int, B: int, seed: int) -> np.ndarray:
    """(B, n_arrangements) indices into the sorted arrangement id list, with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_arrangements, size=(B, n_arrangements))


def _group_terms(sample: PairedSample, results: Sequence[EpisodeResult], truth: GroundTruth,
                 metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-arrangement summed (numerator, denominator) of the metric terms."""
    num = np.zeros(len(sample.groups))
    den = np.zeros(len(sample.groups))
    for g, idx in enumerate(sample.groups):
        for i in idx:
            n, d = episode_terms(results[i], truth)[metric]
            num[g] += n
            den[g] += d
    return num, den


def draw_diffs(sample: PairedSample, truth: GroundTruth, metric: str, draws: np.ndarray) -> np.ndarray:
    """A - B per draw (NaN where either side is undefined). Same as metrics_on on sample.expand(draw)."""
    counts = np.stack([np.bincount(d, minlength=len(sample.groups)) for d in draws]).astype(float) \
        if len(draws) else np.zeros((0, len(sample.groups)))
    vals = []
    for res in (sample.results_a, sample.results_b):
        num, den = _group_terms(sample, res, truth, metric)
        n, d = counts @ num, counts @ den
        with np.errstate(divide="ignore", invalid="ignore"):
            vals.append(np.where(d > EPS, n / np.where(d > EPS, d, 1.0), np.nan))
    return vals[0] - vals[1]


def paired_diff(results_a: Iterable[EpisodeLog | EpisodeResult], results_b: Iterable[EpisodeLog | EpisodeResult],
                truth: GroundTruth, metric: str, B: int = 10000, seed: int = 0) -> BootstrapResult:
    """Paired arrangement-level bootstrap of metric(A) - metric(B) with a percentile 95% CI."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if B < 1:
        raise ValueError("B must be >= 1")
    sample = pair(results_a, results_b, truth)
    ma = metrics_on(list(sample.results_a), truth)
    mb = metrics_on(list(sample.results_b), truth)
    estimate = None if ma[metric] is None or mb[metric] is None else ma[metric] - mb[metric]
    diffs = draw_diffs(sample, truth, metric, arrangement_draws(len(sample.arrangement_ids), B, seed))
    ok = diffs[~np.isnan(diffs)]
    lo = hi = None
    if ok.size:
        lo, hi = (float(x) for x in np.percentile(ok, [2.5, 97.5]))
    return BootstrapResult(metric=metric, policy_a=ma["policy"], policy_b=mb["policy"], estimate=estimate,
                           ci_lo=lo, ci_hi=hi, n_arrangements=len(sample.arrangement_ids),
                           n_episodes=sample.n_episodes, B=B, n_undefined=int(diffs.size - ok.size),
                           seed=seed, synthetic=bool(ma["synthetic"] or mb["synthetic"]))
