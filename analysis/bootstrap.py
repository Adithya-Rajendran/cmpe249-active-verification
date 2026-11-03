"""Paired, arrangement-level bootstrap for metric differences between two policies.

Each draw resamples the unique arrangement ids with replacement and takes all episodes of
every drawn arrangement (duplicates repeated), so an arrangement's episodes are never split.
Both policies are scored on the same multiset; the CI is the 2.5/97.5 percentile interval of
the defined A - B draws. Draws where either side is undefined are counted and excluded.

Stratified (docs/SIM_PREREGISTRATION.md section 8, the primary uncertainty): given strata
{arrangement_id: stratum} (verify.manifest.bootstrap_strata: the occluder family, M16 split by
side), each draw resamples arrangements WITHIN each stratum, with replacement, keeping every
stratum's count, so each replicate keeps the design's quota counts. Without strata the draw is
the unstratified one above, unchanged.

Singleton strata (review, checkpoint 2): a stratum with one arrangement of the sample always redraws that
arrangement, so it adds no variance, and a sample whose strata are mostly singletons (the 6-arrangement check batch
has 6 slot classes) gets a near zero-width "stratified" CI. So when any stratum holds fewer than MIN_STRATUM_SIZE
arrangements of the sample, the bootstrap falls back to the unstratified draw and says so (resolve_strata:
stratified False, singleton_strata, stratification_note "unstratified: singleton strata ..."). The planned designs
have at least 6 arrangements per stratum and stay stratified.

Single rates (docs/SIM_PREREGISTRATION.md section 8: "Percentile 95 % intervals for contrasts and for single
rates"): rate_intervals gives FRR, removal recall and coverage of one policy the same percentile interval over the
same draws (same sorted arrangement ids, B, seed and strata give the same draws as paired_diff). R enters with its
seed-averaged decision fractions, as in the point estimate: arrangements are resampled, seeds are not, so the
interval is for R's expected rate over its own randomisation (seeds add no samples), and R's common-random-number
episodes (one stream per arrangement and lighting, verify.policies.r_seed_salt) stay together in every draw.
"""
from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterable, Mapping, Sequence
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
    stratified: bool = False   # True: arrangements resampled within strata (stratified_arrangement_draws)
    singleton_strata: tuple[str, ...] = ()   # strata given but with < MIN_STRATUM_SIZE arrangements: fell back
    stratification_note: str = ""            # "unstratified: singleton strata ..." when it fell back, else ""

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["singleton_strata"] = list(self.singleton_strata)
        if self.synthetic:
            d["label"] = SYNTHETIC_LABEL
        return d


@dataclass(frozen=True)
class RateInterval:
    """Percentile 95% interval of one policy's single rate (rate_intervals): the same arrangement-level cluster
    bootstrap as paired_diff, on one policy's results."""
    metric: str
    policy: str
    estimate: float | None     # the metric on the full sample (None if undefined)
    ci_lo: float | None        # 2.5th percentile of the defined draws (None if no draw is defined)
    ci_hi: float | None        # 97.5th percentile
    n_arrangements: int
    n_episodes: int
    B: int
    n_undefined: int           # draws where the rate is undefined (denominator 0; excluded)
    seed: int
    synthetic: bool
    stratified: bool = False
    singleton_strata: tuple[str, ...] = ()
    stratification_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["singleton_strata"] = list(self.singleton_strata)
        if self.synthetic:
            d["label"] = SYNTHETIC_LABEL
        return d


MIN_STRATUM_SIZE = 2   # a stratum of one always redraws itself: it adds no variance (falls back to unstratified)
SINGLETON_PREFIX = "unstratified: singleton strata"


def singleton_note(singletons: Sequence[str]) -> str:
    """The flag a fallback carries in results.json, the bootstrap rows and the tables."""
    return (f"{SINGLETON_PREFIX} {', '.join(singletons)} (fewer than {MIN_STRATUM_SIZE} arrangements each: a stratum "
            "of one always redraws itself and adds no variance, so a stratified CI would be too narrow); the "
            "unstratified arrangement-level cluster bootstrap ran instead")


def resolve_strata(arrangement_ids: Sequence[str], strata: Mapping[str, str] | None
                   ) -> tuple[Mapping[str, str] | None, tuple[str, ...], str]:
    """(strata to resample within, singleton strata, note) for a sample's arrangements. None in, None out (no note:
    the caller flags a missing family itself). If any stratum holds fewer than MIN_STRATUM_SIZE of the sample's
    arrangements, the strata are dropped (unstratified) and the singletons and the note say why. An arrangement
    without a stratum raises ValueError, as sample_draws does."""
    if strata is None:
        return None, (), ""
    missing = [a for a in arrangement_ids if a not in strata]
    if missing:
        raise ValueError(f"no bootstrap stratum for arrangements {missing[:5]}")
    counts: dict[str, int] = {}
    for a in set(arrangement_ids):
        counts[strata[a]] = counts.get(strata[a], 0) + 1
    singles = tuple(sorted(k for k, v in counts.items() if v < MIN_STRATUM_SIZE))
    if singles:
        return None, singles, singleton_note(singles)
    return strata, (), ""


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


def stratified_arrangement_draws(strata: Sequence[str], B: int, seed: int) -> np.ndarray:
    """(B, n) indices into the sorted arrangement id list; strata[j] is arrangement j's stratum. Column j of every
    draw is filled from arrangement j's own stratum, so each replicate holds exactly as many arrangements of every
    stratum as the sample does. Strata are visited in sorted order with one default_rng(seed) stream. With a single
    stratum this is exactly arrangement_draws(n, B, seed)."""
    labels = list(strata)
    n = len(labels)
    rng = np.random.default_rng(seed)
    out = np.empty((B, n), dtype=np.int64)
    for st in sorted(set(labels)):
        members = np.array([j for j, x in enumerate(labels) if x == st], dtype=np.int64)
        out[:, members] = members[rng.integers(0, len(members), size=(B, len(members)))]
    return out


def sample_draws(arrangement_ids: Sequence[str], B: int, seed: int,
                 strata: Mapping[str, str] | None = None) -> np.ndarray:
    """Unstratified arrangement_draws, or stratified_arrangement_draws when strata are given (every arrangement must
    have one: ValueError otherwise, the caller decides any fallback)."""
    if strata is None:
        return arrangement_draws(len(arrangement_ids), B, seed)
    missing = [a for a in arrangement_ids if a not in strata]
    if missing:
        raise ValueError(f"no bootstrap stratum for arrangements {missing[:5]}")
    return stratified_arrangement_draws([strata[a] for a in arrangement_ids], B, seed)


def _group_terms(sample: PairedSample, results: Sequence[EpisodeResult], truth: GroundTruth,
                 metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-arrangement summed (numerator, denominator) of the metric terms."""
    return _terms(sample.groups, results, truth, metric)


def _terms(groups: Sequence[Sequence[int]], results: Sequence[EpisodeResult], truth: GroundTruth,
           metric: str) -> tuple[np.ndarray, np.ndarray]:
    num = np.zeros(len(groups))
    den = np.zeros(len(groups))
    for g, idx in enumerate(groups):
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
                truth: GroundTruth, metric: str, B: int = 10000, seed: int = 0,
                strata: Mapping[str, str] | None = None) -> BootstrapResult:
    """Paired arrangement-level bootstrap of metric(A) - metric(B) with a percentile 95% CI; stratified within the
    given {arrangement_id: stratum} when strata is not None."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if B < 1:
        raise ValueError("B must be >= 1")
    sample = pair(results_a, results_b, truth)
    ma = metrics_on(list(sample.results_a), truth)
    mb = metrics_on(list(sample.results_b), truth)
    estimate = None if ma[metric] is None or mb[metric] is None else ma[metric] - mb[metric]
    used, singles, note = resolve_strata(sample.arrangement_ids, strata)
    diffs = draw_diffs(sample, truth, metric, sample_draws(sample.arrangement_ids, B, seed, used))
    ok = diffs[~np.isnan(diffs)]
    lo = hi = None
    if ok.size:
        lo, hi = (float(x) for x in np.percentile(ok, [2.5, 97.5]))
    return BootstrapResult(metric=metric, policy_a=ma["policy"], policy_b=mb["policy"], estimate=estimate,
                           ci_lo=lo, ci_hi=hi, n_arrangements=len(sample.arrangement_ids),
                           n_episodes=sample.n_episodes, B=B, n_undefined=int(diffs.size - ok.size),
                           seed=seed, synthetic=bool(ma["synthetic"] or mb["synthetic"]), stratified=used is not None,
                           singleton_strata=singles, stratification_note=note)


@functools.lru_cache(maxsize=16)
def _draw_counts(arrangement_ids: tuple[str, ...], B: int, seed: int,
                 strata: tuple[str, ...] | None) -> np.ndarray:
    """(B, n) multiplicity of each arrangement in each draw of sample_draws (read-only; cached, since every policy row
    of one lighting resamples the same arrangements with the same draws)."""
    st = None if strata is None else dict(zip(arrangement_ids, strata))
    draws = sample_draws(arrangement_ids, B, seed, st)
    counts = np.zeros((B, len(arrangement_ids)))
    np.add.at(counts, (np.arange(B)[:, None], draws), 1.0)
    counts.setflags(write=False)
    return counts


RATE_METRICS = ("false_removal_rate", "removal_recall", "coverage")


def rate_intervals(results: Iterable[EpisodeLog | EpisodeResult], truth: GroundTruth,
                   metrics: Sequence[str] = RATE_METRICS, B: int = 10000, seed: int = 0,
                   strata: Mapping[str, str] | None = None) -> dict[str, RateInterval]:
    """{metric: RateInterval}: percentile 95% intervals of one policy's single rates from the arrangement-level
    cluster bootstrap, stratified within the given {arrangement_id: stratum} unless a stratum of the sample is a
    singleton (resolve_strata). The draws are sample_draws over the sorted arrangement ids, the same draws paired_diff
    uses for the same arrangements, B and seed. One result per episode (R: seeds averaged within the episode)."""
    if B < 1:
        raise ValueError("B must be >= 1")
    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {m!r}")
    res = sorted(as_results(results), key=lambda r: r.episode_id)
    policy, _, _ = check_single_run(res)
    ids = [r.episode_id for r in res]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode_id in results")
    check_truth(res, truth)
    arr_ids = tuple(sorted({r.arrangement_id for r in res}))
    groups = tuple(tuple(i for i, r in enumerate(res) if r.arrangement_id == a) for a in arr_ids)
    used, singles, note = resolve_strata(arr_ids, strata)
    counts = _draw_counts(arr_ids, B, seed, None if used is None else tuple(used[a] for a in arr_ids))
    full = metrics_on(res, truth)
    out = {}
    for m in metrics:
        num, den = _terms(groups, res, truth, m)
        n, d = counts @ num, counts @ den
        ok = d > EPS
        vals = n[ok] / d[ok]
        lo = hi = None
        if vals.size:
            lo, hi = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
        out[m] = RateInterval(metric=m, policy=policy, estimate=full[m], ci_lo=lo, ci_hi=hi,
                              n_arrangements=len(arr_ids), n_episodes=len(res), B=B, n_undefined=int((~ok).sum()),
                              seed=seed, synthetic=bool(full["synthetic"]), stratified=used is not None,
                              singleton_strata=singles, stratification_note=note)
    return out
