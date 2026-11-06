"""Ranker A and A-nocue: training (DEVELOPMENT; notes/2026-11-05-ranker-a-design.md is the specification).

The decision-time policy (verify.policies.Ranker) and the feature function it shares with training
(verify.policies.ranker_features) live in the linted policy module. This module turns a fold's TRAINING DevData into
labelled rows, fits the L2 logistic model, picks C by inner arrangement-grouped cross-validation on the label's
log-loss, and returns a factory of fresh Ranker instances: the FoldTrainable protocol of verify/select.py, so the
held-out policy check builds A afresh in every fold's own process on dev.subset(training arrangements).

    rows        every training episode, every ordered history H of 0..MAX_HISTORY distinct non-start stops after which
                the shared stopping rule (the fold's thresholds) says CONTINUE, one row per unvisited stop: the
                features of that stop given P3 + H (requested views only) and its geometric diagnostic label.
    weights     each arrangement 1 in total, split equally over its episodes with rows, then over that episode's rows.
    fit         standardised features (weighted mean and deviation of the fitting rows; zero deviation: centred
                only), unpenalised intercept, sum w * logloss + ||beta||^2 / (2 C); Newton-Raphson from zero with step
                halving if the objective would rise, until the largest step < 1e-10 (at most 100 steps, else an error).
    selection   INNER_FOLDS folds of the sorted training arrangements (fold = position mod INNER_FOLDS); per C the mean
                over folds of the held-in fold's weighted mean log-loss; lowest wins, ties within 1e-12 to the smaller C.
"""
from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verify.policies import NOCUE_FEATURES, RANKER_FEATURES, Ranker, ranker_features
from verify.schema import CONTINUE, Thresholds
from verify.side import SideObservation
from verify.stopping import StoppingRule

C_GRID = (0.01, 0.1, 1.0, 10.0)
INNER_FOLDS = 5
MAX_HISTORY = 2            # b - 1 at the primary budget b = 3: the states of the first, second and third choice
MAX_NEWTON = 100
STEP_TOL = 1e-10
TIE_TOL = 1e-12
MODEL_SCHEMA = "cmpe249.ranker-model.v1"


@dataclass(frozen=True)
class Rows:
    X: np.ndarray             # (n, p) raw features, in `features` order
    y: np.ndarray             # (n,) 1.0 = the candidate stop is geometrically diagnostic
    w: np.ndarray             # (n,) weights: each arrangement sums to 1
    arrangement: tuple[str, ...]
    features: tuple[str, ...]


def histories(n_stops: int, start: int, max_len: int = MAX_HISTORY) -> list[tuple[int, ...]]:
    """Every ordered sequence of 0..max_len distinct non-start stops."""
    free = [s for s in range(n_stops) if s != start]
    return [h for k in range(max_len + 1) for h in itertools.permutations(free, k)]


def training_rows(dev: Any, thresholds: Thresholds, features: Sequence[str]) -> Rows:
    """The design note's rows and weights from `dev` (a DevData of training arrangements only)."""
    features = tuple(features)
    if dev.side is None:   # A reads it; A-nocue runs on the same SideEnv observations and ignores it
        raise ValueError("the rankers run on dev data with side descriptors (DevData.side; verify.select --side)")
    store = dev.store_factory()
    side = dev.side()
    labels = {(lab.episode_id, lab.stop): lab.diagnostic for lab in dev.view_labels}
    rule = StoppingRule(thresholds)
    per_episode: dict[str, list[tuple[list[float], float]]] = {}
    arrangement_of: dict[str, str] = {}
    for spec in dev.specs:
        pos = dev.positions_of(spec.arrangement_id)
        n, start = pos.n_stops, pos.start_index
        recs = [store.get(spec.episode_id, s) for s in range(n)]
        sides = [side.get(spec.episode_id, s) for s in range(n)]

        def observe(stops: Sequence[int]) -> tuple[SideObservation, ...]:
            return tuple(SideObservation(step=i, stop=s, azimuth_deg=pos.azimuth(s), match_score=recs[s].match_score,
                                         quality=recs[s].quality, obstruction=recs[s].obstruction, side=sides[s])
                         for i, s in enumerate(stops))
        rows = []
        for h in histories(n, start):
            obs = observe((start, *h))
            if rule.decide(obs, True) is not CONTINUE:
                continue
            for c in range(n):
                if c == start or c in h:
                    continue
                f = ranker_features(obs, c, pos, thresholds.tau_occ)
                rows.append(([f[k] for k in features], 1.0 if labels[(spec.episode_id, c)] else 0.0))
        if rows:
            per_episode[spec.episode_id] = rows
            arrangement_of[spec.episode_id] = spec.arrangement_id
    if not per_episode:
        raise ValueError("no training rows: every training episode stopped at its first view")
    eps_of: dict[str, list[str]] = {}
    for e, a in arrangement_of.items():
        eps_of.setdefault(a, []).append(e)
    X, y, w, arr = [], [], [], []
    for a in sorted(eps_of):
        for e in sorted(eps_of[a]):
            rows = per_episode[e]
            for x, lab in rows:
                X.append(x)
                y.append(lab)
                w.append(1.0 / len(eps_of[a]) / len(rows))
                arr.append(a)
    return Rows(np.asarray(X, float), np.asarray(y, float), np.asarray(w, float), tuple(arr), features)


def standardise(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and deviation per column; a zero deviation becomes 1 (the column is only centred)."""
    m = (w[:, None] * X).sum(0) / w.sum()
    sd = np.sqrt((w[:, None] * (X - m) ** 2).sum(0) / w.sum())
    return m, np.where(sd > 0, sd, 1.0)


def _objective(beta: np.ndarray, Xa: np.ndarray, y: np.ndarray, w: np.ndarray, pen: np.ndarray) -> float:
    z = Xa @ beta
    return float((w * (np.logaddexp(0.0, z) - y * z)).sum() + 0.5 * (pen * beta ** 2).sum())


def fit_logistic(Z: np.ndarray, y: np.ndarray, w: np.ndarray, C: float) -> tuple[float, np.ndarray]:
    """(intercept, coefficients) on standardised features Z. See the module docstring."""
    n, p = Z.shape
    Xa = np.hstack([np.ones((n, 1)), Z])
    pen = np.r_[0.0, np.full(p, 1.0 / C)]
    beta = np.zeros(p + 1)
    obj = _objective(beta, Xa, y, w, pen)
    for _ in range(MAX_NEWTON):
        mu = 0.5 * (1.0 + np.tanh(0.5 * (Xa @ beta)))          # the logistic function, without overflow
        grad = Xa.T @ (w * (mu - y)) + pen * beta
        hess = (Xa * (w * mu * (1.0 - mu))[:, None]).T @ Xa + np.diag(pen)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError as e:   # e.g. one label class only: the intercept runs off to infinity
            raise RuntimeError(f"the logistic fit is degenerate (C = {C}): {e}") from e
        t = 1.0
        while True:
            new = beta - t * step
            new_obj = _objective(new, Xa, y, w, pen)
            if new_obj <= obj + 1e-12 or t < 1e-8:
                break
            t *= 0.5
        beta, obj = new, new_obj
        if np.max(np.abs(t * step)) < STEP_TOL:
            return float(beta[0]), beta[1:]
    raise RuntimeError(f"the logistic fit did not converge in {MAX_NEWTON} Newton steps (C = {C})")


def weighted_logloss(intercept: float, coef: np.ndarray, Z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    z = intercept + Z @ coef
    return float((w * (np.logaddexp(0.0, z) - y * z)).sum() / w.sum())


def inner_folds(arrangements: Sequence[str], k: int = INNER_FOLDS) -> list[set[str]]:
    arrs = sorted(set(arrangements))
    if len(arrs) < k:
        raise ValueError(f"inner cross-validation needs at least {k} training arrangements, got {len(arrs)}")
    return [{a for i, a in enumerate(arrs) if i % k == j} for j in range(k)]


@dataclass(frozen=True)
class Fitted:
    features: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    C: float
    inner_logloss: dict[str, float]
    n_rows: int
    n_arrangements: int
    label_rate: float


def fit_with_selection(rows: Rows, c_grid: Sequence[float] = C_GRID) -> Fitted:
    """Inner arrangement-grouped selection of C, then the refit on every row."""
    arr = np.asarray(rows.arrangement)
    losses = {}
    for C in c_grid:
        per = []
        for held in inner_folds(rows.arrangement):
            te = np.isin(arr, sorted(held))
            m, s = standardise(rows.X[~te], rows.w[~te])
            b0, b = fit_logistic((rows.X[~te] - m) / s, rows.y[~te], rows.w[~te], C)
            per.append(weighted_logloss(b0, b, (rows.X[te] - m) / s, rows.y[te], rows.w[te]))
        losses[C] = float(np.mean(per))
    best = min(c_grid, key=lambda C: (losses[C], C))
    best = min((C for C in c_grid if losses[C] <= losses[best] + TIE_TOL), default=best)
    m, s = standardise(rows.X, rows.w)
    b0, b = fit_logistic((rows.X - m) / s, rows.y, rows.w, best)
    return Fitted(rows.features, tuple(map(float, m)), tuple(map(float, s)), tuple(map(float, b)), b0, float(best),
                  {f"{C:g}": losses[C] for C in c_grid}, len(rows.y), len(set(rows.arrangement)),
                  float((rows.w * rows.y).sum() / rows.w.sum()))


def policy_of(name: str, fitted: Fitted, tau_occ: float) -> Ranker:
    return Ranker(name, fitted.features, fitted.mean, fitted.scale, fitted.coef, fitted.intercept, tau_occ)


class RankerA:
    """FoldTrainable for ranker A (class = factory: the held-out check builds one per fold, in the fold's process)."""
    name = "A"
    features: tuple[str, ...] = RANKER_FEATURES

    def __init__(self) -> None:
        self.report: dict[str, Any] | None = None

    def train(self, train: Any, thresholds: Thresholds) -> Callable[[], Ranker]:
        fitted = fit_with_selection(training_rows(train, thresholds, self.features))
        self.report = report_of(fitted, train.arrangements)
        name, tau_occ = self.name, thresholds.tau_occ
        return lambda: policy_of(name, fitted, tau_occ)


class RankerANoCue(RankerA):
    """FoldTrainable for A-nocue: A without the side descriptor and every other obstruction-derived feature."""
    name = "A-nocue"
    features = NOCUE_FEATURES


RANKERS = {"A": RankerA, "A-nocue": RankerANoCue}


def report_of(fitted: Fitted, arrangements: Sequence[str]) -> dict[str, Any]:
    return {"features": list(fitted.features), "C": fitted.C, "inner_logloss": fitted.inner_logloss,
            "intercept": fitted.intercept, "coef": dict(zip(fitted.features, fitted.coef)),
            "mean": dict(zip(fitted.features, fitted.mean)), "scale": dict(zip(fitted.features, fitted.scale)),
            "n_rows": fitted.n_rows, "n_arrangements": fitted.n_arrangements, "label_rate": fitted.label_rate,
            "n_training_arrangements": len(arrangements)}


def save_model(path: str | Path, name: str, fitted: Fitted, thresholds: Thresholds, provenance: dict[str, Any]) -> None:
    """The frozen model file (FREEZE.md): everything Ranker needs, plus how it was trained."""
    doc = {"schema": MODEL_SCHEMA, "name": name, "tau_occ": thresholds.tau_occ, "features": list(fitted.features),
           "mean": list(fitted.mean), "scale": list(fitted.scale), "coef": list(fitted.coef),
           "intercept": fitted.intercept, "C": fitted.C, "inner_logloss": fitted.inner_logloss,
           "n_rows": fitted.n_rows, "n_arrangements": fitted.n_arrangements, "label_rate": fitted.label_rate,
           "thresholds": thresholds.to_dict(), "provenance": provenance}
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def load_model(path: str | Path) -> Ranker:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {MODEL_SCHEMA!r}")
    return Ranker(doc["name"], doc["features"], doc["mean"], doc["scale"], doc["coef"], doc["intercept"], doc["tau_occ"])
