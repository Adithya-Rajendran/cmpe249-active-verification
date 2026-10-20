"""Offline replay runner: generate -> run (no truth) -> score (see verify/README.md section 8).

    uv run --no-sync python -m verify.run --synthetic --n-arr 30 --seed 0 --budgets 0..4 \\
        [--cap-steps 1,2,3,4,6,inf | none] [--block-anywhere] [--positions configs/positions.json] \\
        [--r-seeds 20] [--bootstrap-B 10000] [--primary-budget 3] [--tag TAG] [--out runs/synthetic-<tag>/]

Every --synthetic output is labelled SYNTHETIC: the generator parameters and thresholds are
placeholders, and the numbers are pipeline checks, not study results.

Real data (v0, pilot): --manifest configs/arrangements.json --evidence evidence.jsonl, where the
evidence file comes from perception/evidence.py (EvidenceStore.to_jsonl format). The same run stage
replays it with no truth; the manifest is re-read for truth only after the run. Thresholds are still
the placeholders; it writes config.json, logs.jsonl, ground_truth.json and results.json (no table.md).

Stages are separate functions, in this order:
  1. generate_stage returns the specs and the (unsealed) evidence store. It writes nothing
     truth-bearing and does not return the truth.
  2. run_stage has no truth parameter. It mints opaque ReplayKeys, drives every policy through
     a fresh ReplayEnv built from a key (never the spec) and joins the log header afterwards.
  3. write_scoring_files regenerates the episodes deterministically after the run, checks every
     revealed observation against them, and only then writes episodes.json, evidence.jsonl and
     ground_truth.json. score_stage loads the truth from disk and computes metrics and bootstraps.

Travel caps (D-cap) are an operating condition reported as curves, never tuned: a cap is given in
arc steps and cap_cm = steps x Positions.nominal_step_cm() (23.56 cm on the design geometry), with
no added slack. The default grid is {1, 2, 3, 4, 6} steps plus uncapped.

Scoring is on the TEST split only (dev is reserved for threshold selection, D10, which the
synthetic placeholders skip). Dev metrics are kept in results.json with scope "dev".
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from analysis.bootstrap import paired_diff
from analysis.metrics import compute_all
from verify.manifest import load_manifest
from verify.policies import (
    DEFAULT_R_SEEDS,
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
    DEFAULT_POSITIONS_PATH,
    TRAVEL_TOL_CM,
    REPO_ROOT,
    SYNTHETIC_LABEL,
    Budget,
    Decision,
    EpisodeLog,
    EpisodeResult,
    EpisodeSpec,
    EvidenceStore,
    GroundTruth,
    Positions,
    ReplayError,
    RunContext,
    Split,
    Thresholds,
    as_results,
    cap_cm_from_steps,
    config_sha256,
    file_sha256,
    to_jsonable,
)
from verify.stopping import StoppingRule
from verify.synthetic import SyntheticParams, generate

CONFIG_VERSION = "synthetic-v3"
MODEL_ID = "synthetic-no-model"
MULTI_VIEW = ("F", "R", "G", "G-myopic")
BOOTSTRAP_PAIRS = (("G", "F"), ("G", "R"), ("G", "S"), ("F", "S"), ("R", "S"), ("G", "G-myopic"))
# H1 (prereg sec. 10): the same selector with the k-distinct rule vs the first-clear-miss rule (k = 1).
H1_PAIRS = tuple((x, f"{x}-fcm") for x in MULTI_VIEW)
# "selection can matter" diagnostics (lit report sec. 4): G's path against these, per budget and cap
DISAGREE_PAIRS = (("G", "F"), ("G", "R"), ("G", "G-myopic"), ("G-myopic", "F"))
BOOTSTRAP_METRICS = ("false_removal_rate", "removal_recall", "coverage")
SCOPES = (Split.TEST, Split.DEV)
ROLE = {"S": "actual effort (1 view, budget-independent)",
        "FULL": "higher-effort reference (5 views, ignores budget)"}

RunKey = tuple[str, int, float | None]  # (policy log name, budget, travel cap)


# ---------------------------------------------------------------- arguments and config

def parse_budgets(text: str) -> tuple[int, ...]:
    """'0,1,3' or '0..4' -> sorted unique budgets in 0..4."""
    text = text.strip()
    if ".." in text:
        lo, hi = (int(x) for x in text.split("..", 1))
        vals = list(range(lo, hi + 1))
    else:
        vals = [int(x) for x in text.split(",") if x.strip()]
    if not vals or any(not 0 <= b <= 4 for b in vals):
        raise argparse.ArgumentTypeError(f"budgets must be in 0..4, got {text!r}")
    return tuple(sorted(set(vals)))


def parse_cap_steps(text: str) -> tuple[float, ...]:
    """'1,2,3,4,6,inf' -> finite cap settings in arc steps, sorted; 'inf' (uncapped) always runs anyway.

    'none' -> no capped setting. Caps are given in steps only (D-cap): no cm input, no ad-hoc slack.
    """
    if text.strip().lower() in ("", "none"):
        return ()
    try:
        vals = {float(x) for x in text.split(",") if x.strip()}
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"cap steps must be numbers or 'inf', got {text!r}") from e
    if any(v <= 0 or v != v for v in vals):
        raise argparse.ArgumentTypeError("cap steps must be > 0 (or 'inf')")
    return tuple(sorted(v for v in vals if v != float("inf")))


def _steps_str(steps: float) -> str:
    return f"{steps:g}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verify.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="use the synthetic generator")
    p.add_argument("--manifest", default=None, help="real data: arrangements manifest (with --evidence)")
    p.add_argument("--evidence", default=None, help="real data: evidence.jsonl from perception/evidence.py")
    p.add_argument("--n-arr", type=int, default=30, help="number of arrangements (default 30)")
    p.add_argument("--n-dev", type=int, default=None, help="dev arrangements (default round(n/3))")
    p.add_argument("--seed", type=int, default=0, help="generator / split / bootstrap seed")
    p.add_argument("--budgets", type=parse_budgets, default=(0, 1, 2, 3, 4), help="'0..4' or '0,1,3'")
    p.add_argument("--cap-steps", type=parse_cap_steps,
                   default=tuple(c for c in CAP_STEPS_DEFAULT if c != float("inf")),
                   help="travel caps in arc steps (cap_cm = steps x nominal step from --positions), comma list, "
                        "or 'none'; default 1,2,3,4,6. The uncapped setting always runs. Caps are an operating "
                        "condition reported as curves, not tuned (D-cap)")
    p.add_argument("--block-anywhere", action="store_true",
                   help="synthetic regime: occluder block placed anywhere (default: always covers the start stop)")
    p.add_argument("--positions", default=str(DEFAULT_POSITIONS_PATH),
                   help="positions JSON (default configs/positions.json); its contents enter the config hash")
    p.add_argument("--r-seeds", type=int, default=len(DEFAULT_R_SEEDS), help="R seeds per episode (default 20)")
    p.add_argument("--bootstrap-B", type=int, default=10000, help="bootstrap draws (default 10000)")
    p.add_argument("--primary-budget", type=int, default=3, help="budget for the primary table (default 3)")
    p.add_argument("--tag", default=None, help="run tag (default n<N>-s<seed>)")
    p.add_argument("--out", default=None, help="output dir (default runs/synthetic-<tag>/ or runs/real-<tag>/)")
    p.add_argument("--model-id", default="perception.evidence-v0", help="real data: perception model id for the logs")
    return p


def _rel(path: Path) -> str:
    """Repo-relative path when inside the repo (keeps the config hash machine-independent)."""
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def build_config(args: argparse.Namespace, positions: Positions, th: Thresholds,
                 params: SyntheticParams) -> dict[str, Any]:
    """Everything that determines the run; hashed into every log line."""
    pos_path = Path(args.positions)
    return {"config_version": CONFIG_VERSION, "model_id": MODEL_ID, "synthetic": True,
            "positions_file": _rel(pos_path), "positions_file_sha256": file_sha256(pos_path),
            "n_arr": args.n_arr, "seed": args.seed, "budgets": list(args.budgets),
            "nominal_step_cm": positions.nominal_step_cm(), "travel_tol_cm": TRAVEL_TOL_CM,
            "cap_steps": [None, *args.cap_steps],
            "travel_caps_cm": [None, *(cap_cm_from_steps(c, positions) for c in args.cap_steps)],
            "g_myopic": {"lam_deg": GMYOPIC_LAM_DEG, "travel_norm_cm": th.travel_norm_cm},
            "r_seeds": list(range(args.r_seeds)),
            "bootstrap_B": args.bootstrap_B, "primary_budget": args.primary_budget,
            "thresholds": th.to_dict(), "positions": positions.to_dict(),
            "generator": {"module": "verify.synthetic", **dataclasses.asdict(params)},
            "scoring_split": Split.TEST.value}


# ---------------------------------------------------------------- stage 1: generate

def generate_stage(n_arr: int, seed: int, params: SyntheticParams,
                   positions: Positions) -> tuple[list[EpisodeSpec], EvidenceStore]:
    """Generate synthetic episodes; return specs and store only. Writes nothing to disk.

    Truth and the full evidence dump are written after the run (write_scoring_files), so no
    truth-bearing or unvisited-evidence file exists in the output dir while policies run.
    """
    specs, store, truth = generate(n_arr, seed, params, positions)
    del truth  # the run stage never sees it
    return specs, store


def write_scoring_files(n_arr: int, seed: int, params: SyntheticParams, positions: Positions,
                        specs: Sequence[EpisodeSpec], logs: Sequence[EpisodeLog], out: Path) -> None:
    """After the run: regenerate deterministically, check it matches what was replayed, then write
    episodes.json, evidence.jsonl and ground_truth.json (scoring-only)."""
    specs2, store2, truth = generate(n_arr, seed, params, positions)
    if [s.to_dict() for s in specs2] != [s.to_dict() for s in specs]:
        raise ReplayError("regenerated episodes differ from the replayed ones")
    for g in logs:
        for o in g.evidence:
            rec = store2.get(g.episode_id, o.stop)
            if (rec.match_score, rec.quality, rec.obstruction) != (o.match_score, o.quality, o.obstruction):
                raise ReplayError(f"regenerated evidence differs for {g.episode_id} stop {o.stop}")
    write_json(out / "episodes.json", {"label": SYNTHETIC_LABEL, "synthetic": True,
                                       "episodes": [s.to_dict() for s in specs]})
    store2.to_jsonl(out / "evidence.jsonl")
    truth.save(out / "ground_truth.json")


# ---------------------------------------------------------------- stage 2: run (no truth)

def run_stage(specs: Sequence[EpisodeSpec], store: EvidenceStore, positions: Positions, th: Thresholds,
              budgets: Sequence[int], caps: Sequence[float | None], r_seeds: Sequence[int],
              context: RunContext) -> list[EpisodeLog]:
    """Every policy x budget x cap on every episode, each run on a fresh ReplayEnv.

    Envs are built from opaque ReplayKeys (minted here, before the store is sealed), never from
    the spec; the log header is joined onto each header-free run afterwards.
    """
    rule = StoppingRule(th)
    fcm = StoppingRule.first_clear_miss(th)  # F/G/R-fcm, and S (one view, removed on a single clear miss)
    keys = store.replay_keys()
    logs: list[EpisodeLog] = []

    for spec in specs:
        key = keys[spec.episode_id]

        def env(budget: Budget, key=key) -> ReplayEnv:
            return ReplayEnv(key, store, positions, budget, context)

        logs.append(EpisodeLog.from_run(spec, run_episode(S(), env(Budget(0)), fcm)))
        logs.append(EpisodeLog.from_run(spec, run_episode(FullSweep(), env(Budget(4)), rule)))
        for cap in caps:
            for b in budgets:
                budget = Budget(b, cap)
                for r in (rule, fcm):
                    for pol in (F(), G.from_thresholds(th), GMyopic.from_thresholds(th)):
                        logs.append(EpisodeLog.from_run(spec, run_episode(pol, env(budget), r)))
                    seed_logs, _ = run_r_seeds(lambda bu=budget, e=env: e(bu), r, r_seeds, r_seed_salt(spec),
                                               spec=spec)
                    logs.extend(seed_logs)
    return logs


# ---------------------------------------------------------------- stage 3: score

def group_results(logs: Sequence[EpisodeLog]) -> dict[RunKey, list[EpisodeResult]]:
    """(policy, budget, cap) -> per-episode results (R seeds averaged within the episode)."""
    out: dict[RunKey, list[EpisodeResult]] = {}
    for r in as_results(logs):
        out.setdefault((r.policy, r.budget_additional_views, r.travel_cap_cm), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r.episode_id)
    return out


def _cap_key(cap: float | None) -> tuple[int, float]:
    return (0, 0.0) if cap is None else (1, cap)


def score_stage(logs: Sequence[EpisodeLog], truth: GroundTruth, primary_budget: int,
                B: int, seed: int) -> dict[str, Any]:
    """Metrics per (scope, policy, budget, cap) and paired bootstraps at the primary budget."""
    groups = group_results(logs)
    caps = sorted({k[2] for k in groups}, key=_cap_key)
    metrics: list[dict[str, Any]] = []
    for scope in SCOPES:
        for key in sorted(groups, key=lambda k: (k[0], k[1], _cap_key(k[2]))):
            res = [r for r in groups[key] if r.split is scope]
            if not res:
                continue
            m = compute_all(res, truth)
            metrics.append({"scope": scope.value, "policy": key[0], "budget": key[1], "travel_cap_cm": key[2],
                            "role": ROLE.get(key[0], ""), **m})
    boots: list[dict[str, Any]] = []
    test = {k: [r for r in v if r.split is Split.TEST] for k, v in groups.items()}
    for cap in caps:
        pairs = [(a + v if a != "S" else a, b + v if b != "S" else b)
                 for v in ("", "-fcm") for a, b in BOOTSTRAP_PAIRS] + list(H1_PAIRS)
        for a, b in pairs:
            ka = _lookup(test, a, primary_budget, cap)
            kb = _lookup(test, b, primary_budget, cap)
            if ka is None or kb is None or not test[ka] or not test[kb]:
                continue
            same = identical_runs(test[ka], test[kb])
            for metric in BOOTSTRAP_METRICS:
                br = paired_diff(test[ka], test[kb], truth, metric, B=B, seed=seed)
                boots.append({"scope": Split.TEST.value, "travel_cap_cm": cap, "budget": primary_budget,
                              "identical_runs": same, **br.to_dict()})
    return {"metrics": metrics, "bootstrap": boots}


def identical_runs(a: Sequence[EpisodeResult], b: Sequence[EpisodeResult]) -> bool:
    """True if A and B made the same decisions with the same effort on every episode (degenerate CI)."""
    da = {r.episode_id: r for r in a}
    db = {r.episode_id: r for r in b}
    if set(da) != set(db):
        return False
    return all({d: da[e].frac(d) for d in Decision} == {d: db[e].frac(d) for d in Decision}
               and da[e].mean_travel_cm == db[e].mean_travel_cm and da[e].mean_views == db[e].mean_views
               for e in da)


def _lookup(groups: dict[RunKey, list[EpisodeResult]], policy: str, budget: int,
            cap: float | None) -> RunKey | None:
    """S and FULL are budget- and cap-independent: one run each, reused at every setting."""
    if policy == "S":
        return ("S", 0, None) if ("S", 0, None) in groups else None
    if policy == "FULL":
        return ("FULL", 4, None) if ("FULL", 4, None) in groups else None
    k = (policy, budget, cap)
    return k if k in groups else None


# ---------------------------------------------------------------- diagnostics (no truth)

def reachable_view_sets(positions: Positions, budget: int, cap_cm: float | None) -> dict[str, Any]:
    """Geometry only: every set of stops reachable from the start with <= budget moves within the cap.

    Returns the largest reachable number of views and the number of distinct view sets of that size
    (the sets a policy that uses its whole budget could end with), plus all reachable sets.
    """
    sets: set[frozenset[int]] = set()

    def walk(cur: int, seen: frozenset[int], spent: float, moves: int) -> None:
        sets.add(seen)
        if moves == budget:
            return
        for s in range(positions.n_stops):
            if s in seen:
                continue
            t = spent + positions.travel_cm(cur, s)
            if cap_cm is None or t <= cap_cm + TRAVEL_TOL_CM:
                walk(s, seen | {s}, t, moves + 1)
    st = positions.start_index
    walk(st, frozenset({st}), 0.0, 0)
    top = max(len(x) for x in sets)
    full = sorted(sorted(x) for x in sets if len(x) == top)
    return {"max_views": top, "n_max_view_sets": len(full), "max_view_sets": full, "n_reachable_sets": len(sets)}


def diagnostics_stage(logs: Sequence[EpisodeLog], positions: Positions, budgets: Sequence[int],
                      caps: Sequence[float | None]) -> dict[str, Any]:
    """'Can selection matter?' diagnostics (logs only, no truth): reachable view sets per (b, cap),
    distinct view sets each policy actually used, and path / decision disagreement between policies.

    Disagreement for (A, B) is per episode: deterministic vs deterministic is 0/1; a pair with R is the
    fraction of R's seeds whose path (or decision) differs, so it is averaged within episode first.
    """
    paths: dict[tuple[str, int, float | None, str], list[tuple[tuple[int, ...], str]]] = {}
    split_of: dict[str, str] = {}
    for g in logs:
        paths.setdefault((g.policy, g.budget_additional_views, g.travel_cap_cm, g.episode_id), []).append(
            (tuple(g.requested_stops), g.decision.value))
        split_of[g.episode_id] = g.split.value
    episodes = sorted(split_of)
    reach, used, disagree = [], [], []
    for cap in caps:
        for b in budgets:
            reach.append({"budget": b, "travel_cap_cm": cap, **reachable_view_sets(positions, b, cap)})
            for scope in ("test", "dev", "all"):
                eps = [e for e in episodes if scope == "all" or split_of[e] == scope]
                for pol in MULTI_VIEW:
                    sets = {frozenset(pth) for e in eps for pth, _ in paths.get((pol, b, cap, e), [])}
                    if sets:
                        used.append({"scope": scope, "policy": pol, "budget": b, "travel_cap_cm": cap,
                                     "n_view_sets": len(sets)})
                for a, bb in DISAGREE_PAIRS:
                    path_d, dec_d, n = 0.0, 0.0, 0
                    for e in eps:
                        ra, rb = paths.get((a, b, cap, e)), paths.get((bb, b, cap, e))
                        if not ra or not rb:
                            continue
                        pairs = [(x, y) for x in ra for y in rb]
                        path_d += sum(x[0] != y[0] for x, y in pairs) / len(pairs)
                        dec_d += sum(x[1] != y[1] for x, y in pairs) / len(pairs)
                        n += 1
                    if n:
                        disagree.append({"scope": scope, "policy_a": a, "policy_b": bb, "budget": b,
                                         "travel_cap_cm": cap, "n_episodes": n, "path_differs": path_d,
                                         "path_disagreement_rate": path_d / n, "decision_differs": dec_d,
                                         "decision_disagreement_rate": dec_d / n})
    return {"reachable_view_sets": reach, "used_view_sets": used, "disagreement": disagree}


# ---------------------------------------------------------------- table.md

def fmt(x: float | None, nd: int = 3) -> str:
    return "undef" if x is None else f"{x:.{nd}f}"


def _find(metrics: Sequence[dict], scope: str, policy: str, budget: int, cap: float | None) -> dict | None:
    if policy == "S":
        budget, cap = 0, None
    elif policy == "FULL":
        budget, cap = 4, None
    for m in metrics:
        if (m["scope"], m["policy"], m["budget"], m["travel_cap_cm"]) == (scope, policy, budget, cap):
            return m
    return None


def _cap_label(cap: float | None, step_cm: float | None = None) -> str:
    """'no travel cap' or 'travel cap N steps (X cm)' (cm only when step_cm is unknown)."""
    if cap is None:
        return "no travel cap"
    if not step_cm:
        return f"travel cap {cap:.2f} cm"
    return f"travel cap {_steps_str(round(cap / step_cm, 6))} steps ({cap:.2f} cm)"


def _cells(o: dict) -> list[str]:
    return [fmt(o["false_removal_rate"]), fmt(o["removal_recall"]), fmt(o["coverage"]),
            fmt(o["resolved_accuracy"]), fmt(o["mean_views"], 2), fmt(o["mean_travel_cm"], 1),
            f"{o['n_arrangements']}/{o['n_episodes']}"]


def _md(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


HEAD = ("| policy | FRR | recall | coverage | resolved acc | mean views | mean travel cm | n_arr/n_ep | note |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
ORDER = ("S", "F", "R", "G", "G-myopic", "F-fcm", "R-fcm", "G-fcm", "G-myopic-fcm", "FULL")
FULL_OVER_CAP = "uncapped reference (exceeds cap)"


def _note(p: str, cap: float | None) -> str:
    return FULL_OVER_CAP if p == "FULL" and cap is not None else ROLE.get(p, "")


def _label(p: str, cap: float | None) -> str:
    return "FULL (uncapped)" if p == "FULL" and cap is not None else p


def _generator_caveat(gen: dict) -> str:
    if gen.get("block_includes_start", True):
        return ("- **Generator caveat (regime: start-blocked).** Every distinguishable occluder is one contiguous "
                "block that always contains the start stop P3, and obstruction falls off with distance from it. "
                "That is exactly G's max-angular-separation prior, and it makes S's single P3 view flagged "
                "occluded in every occluded and removed episode, so S cannot resolve them (recall 0 by "
                "construction). On this regime S rows and the G - S, F - S and G - F differences reflect how the "
                "data were generated, not the policies. Compare the `--block-anywhere` regime.")
    return ("- **Generator regime: block-anywhere.** The occluder block is placed anywhere on the arc (it need not "
            "cover P3). Still synthetic; the obstruction-falls-off-with-distance shape remains G-friendly.")


def _cap_notes(cfg: dict, pos: Positions) -> list[str]:
    step = cfg["nominal_step_cm"]
    caps = [c for c in cfg["travel_caps_cm"] if c is not None]
    out = [(f"- **Travel caps are an operating condition, reported as curves, not tuned (D-cap).** "
            f"Cap = N arc steps x {step:.2f} cm (nominal step of the positions file); grid "
            f"{', '.join(_steps_str(s) for s in cfg['cap_steps'] if s is not None) or 'none'} steps plus uncapped. "
            f"Affordability uses a {cfg['travel_tol_cm']:g} cm tolerance and no other slack. FULL is never capped.")]
    dev = pos.adjacent_step_deviation_cm()
    if dev > cfg["travel_tol_cm"]:
        out.append(f"- **Warning:** adjacent stops deviate from the nominal step by up to {dev:.3f} cm, so a cap "
                   "of N steps does not buy exactly N adjacent moves on this positions file.")
    if caps:
        out.append("- G is cap-aware (reachability lookahead, D-G) with per-decision normalisation (D-norm); "
                   f"G-myopic is the pre-D-G G kept as an ablation (no lookahead, lam "
                   f"{cfg['g_myopic']['lam_deg']:g} deg per {cfg['g_myopic']['travel_norm_cm']:.2f} cm). "
                   "F and R are naive by design.")
    return out


def _maxcov(m: dict) -> str:
    o = m["overall"]
    return (f"- max achievable coverage (test): {fmt(o['max_achievable_coverage'])} "
            f"({o['n_indistinguishable']} of {o['n_episodes']} episodes are predeclared indistinguishable, "
            f"{o['n_arrangements']} arrangements)")


def render_table(results: dict[str, Any]) -> str:
    cfg, metrics = results["config"], results["metrics"]
    pb, scope = cfg["primary_budget"], Split.TEST.value
    caps = cfg["travel_caps_cm"]
    step = cfg.get("nominal_step_cm")

    def lab(cap: float | None) -> str:
        return _cap_label(cap, step)
    lines = [f"# {SYNTHETIC_LABEL}", "",
             ("**SYNTHETIC - not study results.** Evidence comes from `verify.synthetic` with invented "
              "parameters and placeholder thresholds; these numbers only exercise the pipeline."), "",
             f"- config `{cfg['config_version']}`, sha256 `{results['config_sha256']}`",
             (f"- {cfg['n_arr']} arrangements, seed {cfg['seed']}; scored on the **test** split only "
              "(dev metrics are in results.json, scope \"dev\")"),
             (f"- thresholds: tau_match {cfg['thresholds']['tau_match']}, tau_q {cfg['thresholds']['tau_q']}, "
              f"tau_occ {cfg['thresholds']['tau_occ']}, k {cfg['thresholds']['k']}, "
              f"lam {cfg['thresholds']['lam']}; R averaged over {len(cfg['r_seeds'])} seeds within each episode"),
             ("- FRR = false removal rate; `-fcm` = first-clear-miss rule (k = 1); S is plotted at its actual "
              "effort; FULL is a higher-effort reference, not a budget-matched competitor"),
             _generator_caveat(cfg.get("generator", {})), *_cap_notes(cfg, Positions.from_dict(results["positions"]))]
    ref = _find(metrics, scope, "S", pb, None)
    if ref:
        lines.append(_maxcov(ref))
    lines.append("")
    for cap in caps:
        lines += [f"## Primary table: b = {pb} additional views, {lab(cap)}", "", HEAD]
        for p in ORDER:
            m = _find(metrics, scope, p, pb, cap)
            if m:
                lines.append(_md([_label(p, cap), *_cells(m["overall"]), _note(p, cap)]))
        lines.append("")
    for cap in caps:
        lines += [f"## Budget curve, {lab(cap)}", "",
                  "| policy | b | FRR | recall | coverage | resolved acc | mean views | mean travel cm | n_arr/n_ep |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in ORDER:
            if p in ("S", "FULL"):
                continue
            for b in cfg["budgets"]:
                m = _find(metrics, scope, p, b, cap)
                if m:
                    lines.append(_md([p, str(b), *_cells(m["overall"])]))
        lines.append("")
    for cap in caps:
        lines += [f"## Per condition, b = {pb}, {lab(cap)}", "",
                  "Decision fractions are averaged within episode (R) and summed over episodes.", "",
                  ("| policy | condition | n_arr/n_ep | present | removed | unknown | coverage | max achievable cov | "
                   "mean views | mean travel cm |"), "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in ORDER:
            m = _find(metrics, scope, p, pb, cap)
            if not m:
                continue
            for cond, o in m["by_condition"].items():
                t = "removed" if cond == "removed" else "present"
                c = o["confusion"][t]
                lines.append(f"| {_label(p, cap)} | {cond} | {o['n_arrangements']}/{o['n_episodes']} | "
                             f"{c['present']:.2f} | {c['removed']:.2f} | {c['unknown']:.2f} | {fmt(o['coverage'])} | "
                             f"{fmt(o['max_achievable_coverage'])} | "
                             f"{fmt(o['mean_views'], 2)} | {fmt(o['mean_travel_cm'], 1)} |")
        lines.append("")
    lines += [f"## Cap curve, b = {pb} (test split)", "",
              "Each policy across the cap grid at the primary budget. Caps are an operating condition, not tuned.", "",
              "| policy | travel cap | FRR | recall | coverage | resolved acc | mean views | mean travel cm | n_arr/n_ep |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in ORDER:
        if p in ("S", "FULL"):
            continue
        for cap in caps:
            m = _find(metrics, scope, p, pb, cap)
            if m:
                lines.append(_md([p, lab(cap).replace("travel cap ", ""), *_cells(m["overall"])]))
    lines.append("")
    diag = results.get("diagnostics")
    if diag:
        lines += _diagnostic_lines(diag, cfg, lab)
    lines += [f"## Per lighting, b = {pb}, no travel cap", "",
              ("| policy | lighting | FRR | recall | coverage | resolved acc | mean views | mean travel cm | "
               "n_arr/n_ep |"), "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in ORDER:
        m = _find(metrics, scope, p, pb, None)
        if m:
            for light, o in m["by_lighting"].items():
                lines.append(_md([p, light, *_cells(o)]))
    lines += ["", f"## Paired arrangement-level bootstrap, b = {pb} (test split)", "",
              "Difference A - B; percentile 95% CI over resampled arrangements (all 6 episodes kept together).",
              ("A row marked *identical* means A and B made the same decisions with the same effort on every "
               "episode: its [0, 0] CI is degenerate and is not evidence of equal performance."),
              "", "| setting | A - B | metric | estimate | 95% CI | n_arr | n_ep | B | undefined draws | note |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results["bootstrap"]:
        note = "identical: A == B on every episode" if r.get("identical_runs") else ""
        lines.append(f"| {lab(r['travel_cap_cm'])} | {r['policy_a']} - {r['policy_b']} | {r['metric']} | "
                     f"{fmt(r['estimate'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | {r['n_arrangements']} | "
                     f"{r['n_episodes']} | {r['B']} | {r['n_undefined']} | {note} |")
    lines += ["", f"_{SYNTHETIC_LABEL}._", ""]
    return "\n".join(lines)


def _diagnostic_lines(diag: dict, cfg: dict, lab) -> list[str]:
    caps, budgets = cfg["travel_caps_cm"], cfg["budgets"]
    used = {(u["policy"], u["budget"], u["travel_cap_cm"]): u["n_view_sets"]
            for u in diag["used_view_sets"] if u["scope"] == "test"}
    lines = ["## Diagnostic: distinct view sets per budget and cap", "",
             ("Can selection matter at all? *Reachable* is geometry only: the most views reachable from the start "
              "within b moves and the cap, and how many distinct view sets of that size exist. *Used* counts the "
              "distinct view sets each policy actually ended with on the test episodes (early stopping included)."),
             "", "| travel cap | b | max views | reachable sets at max | used: F | used: G | used: G-myopic | used: R |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in diag["reachable_view_sets"]:
        cap, b = r["travel_cap_cm"], r["budget"]
        lines.append(_md([lab(cap).replace("travel cap ", ""), str(b), str(r["max_views"]), str(r["n_max_view_sets"]),
                          *(str(used.get((p, b, cap), "-")) for p in ("F", "G", "G-myopic", "R"))]))
    lines += ["", "## Diagnostic: path and decision disagreement (test split, k-rule policies)", "",
              ("Fraction of test episodes where A's requested path (or final decision) differs from B's. Pairs with "
               "R average over R's seeds within each episode. 0 means the two policies are indistinguishable at "
               "that setting, so any paired difference there is degenerate. Counts are in results.json."), "",
              "| travel cap | b | " + " | ".join(f"{a} vs {b} path / decision" for a, b in DISAGREE_PAIRS) + " |",
              "| --- | --- | " + " | ".join("---" for _ in DISAGREE_PAIRS) + " |"]
    dis = {(d["policy_a"], d["policy_b"], d["budget"], d["travel_cap_cm"]): d
           for d in diag["disagreement"] if d["scope"] == "test"}
    for cap in caps:
        for b in budgets:
            if b == 0:
                continue
            cells = []
            for a, bb in DISAGREE_PAIRS:
                d = dis.get((a, bb, b, cap))
                cells.append("-" if d is None else
                             f"{d['path_disagreement_rate']:.2f} / {d['decision_disagreement_rate']:.2f}")
            lines.append(_md([lab(cap).replace("travel cap ", ""), str(b), *cells]))
    lines.append("")
    return lines


# ---------------------------------------------------------------- io and main

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=1, sort_keys=True) + "\n")


def write_logs(path: Path, logs: Sequence[EpisodeLog]) -> None:
    with path.open("w") as f:
        for g in logs:
            f.write(json.dumps(to_jsonable(g.to_dict()), sort_keys=True, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.synthetic:
        if args.manifest and args.evidence:
            return main_real(args)
        print("need --synthetic, or --manifest and --evidence for real data", file=sys.stderr)
        return 2
    if args.n_arr < 2 or args.r_seeds < 1 or args.bootstrap_B < 1:
        print("need --n-arr >= 2, --r-seeds >= 1, --bootstrap-B >= 1", file=sys.stderr)
        return 2
    if args.primary_budget not in args.budgets:
        print(f"--primary-budget {args.primary_budget} not in --budgets {args.budgets}", file=sys.stderr)
        return 2
    tag = args.tag or f"n{args.n_arr}-s{args.seed}"
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"synthetic-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("ground_truth.json", "evidence.jsonl", "episodes.json"):  # never present while policies run
        (out / stale).unlink(missing_ok=True)

    try:
        positions = Positions.load(args.positions)
    except (OSError, ValueError, KeyError) as e:
        print(f"bad --positions {args.positions}: {e}", file=sys.stderr)
        return 2
    th = Thresholds()
    params = SyntheticParams(n_dev=args.n_dev, block_includes_start=not args.block_anywhere)
    config = build_config(args, positions, th, params)
    sha = config_sha256(config)
    write_json(out / "config.json", {"label": SYNTHETIC_LABEL, "config": config, "config_sha256": sha})
    context = RunContext(model_id=MODEL_ID, config_version=CONFIG_VERSION, config_sha256=sha, synthetic=True)

    specs, store = generate_stage(args.n_arr, args.seed, params, positions)
    logs = run_stage(specs, store, positions, th, args.budgets, config["travel_caps_cm"],
                     config["r_seeds"], context)
    del store
    write_logs(out / "logs.jsonl", logs)
    write_scoring_files(args.n_arr, args.seed, params, positions, specs, logs, out)

    truth = GroundTruth.load(out / "ground_truth.json")
    scored = score_stage(logs, truth, args.primary_budget, args.bootstrap_B, args.seed)
    diagnostics = diagnostics_stage(logs, positions, args.budgets, config["travel_caps_cm"])
    results = {"label": SYNTHETIC_LABEL, "synthetic": True, "config": config, "config_sha256": sha,
               "positions": positions.to_dict(), "thresholds": th.to_dict(), "generator": config["generator"],
               "files": {n: file_sha256(out / n) for n in ("episodes.json", "ground_truth.json",
                                                           "evidence.jsonl", "logs.jsonl")},
               "n_logs": len(logs), **scored, "diagnostics": diagnostics}
    write_json(out / "results.json", results)
    (out / "table.md").write_text(render_table(results))
    print(f"{SYNTHETIC_LABEL}: wrote {out}")
    return 0


REAL_CONFIG_VERSION = "real-v0"
REAL_LABEL = "REAL PILOT - placeholder thresholds, not frozen"


def main_real(args: argparse.Namespace) -> int:
    """Run S/FULL/F/R/G/G-myopic on real evidence: same run and score stages as --synthetic."""
    if args.primary_budget not in args.budgets or args.r_seeds < 1 or args.bootstrap_B < 1:
        print("need --primary-budget in --budgets, --r-seeds >= 1, --bootstrap-B >= 1", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"real-{args.tag or Path(args.evidence).stem}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ground_truth.json").unlink(missing_ok=True)   # never present while policies run
    try:
        positions = Positions.load(args.positions)
        specs, truth = load_manifest(args.manifest)
        del truth  # the run stage never sees it; re-read after the run
        store = EvidenceStore.from_jsonl(args.evidence)
        if store.synthetic:
            raise ValueError(f"{args.evidence} is synthetic evidence; use --synthetic")
        for s in specs:
            store.replay_key(s.episode_id)   # ReplayError unless all stops have evidence
    except (OSError, ValueError, KeyError) as e:
        print(f"bad real-data input: {e}", file=sys.stderr)
        return 2
    th = Thresholds()
    config = {"config_version": REAL_CONFIG_VERSION, "model_id": args.model_id, "synthetic": False,
              "positions_file": _rel(Path(args.positions)), "positions_file_sha256": file_sha256(args.positions),
              "manifest_file": _rel(Path(args.manifest)), "manifest_sha256": file_sha256(args.manifest),
              "evidence_file": _rel(Path(args.evidence)), "evidence_sha256": file_sha256(args.evidence),
              "seed": args.seed, "budgets": list(args.budgets), "nominal_step_cm": positions.nominal_step_cm(),
              "travel_tol_cm": TRAVEL_TOL_CM, "cap_steps": [None, *args.cap_steps],
              "travel_caps_cm": [None, *(cap_cm_from_steps(c, positions) for c in args.cap_steps)],
              "g_myopic": {"lam_deg": GMYOPIC_LAM_DEG, "travel_norm_cm": th.travel_norm_cm},
              "r_seeds": list(range(args.r_seeds)), "bootstrap_B": args.bootstrap_B,
              "primary_budget": args.primary_budget, "thresholds": th.to_dict(), "positions": positions.to_dict(),
              "scoring_split": Split.TEST.value}
    sha = config_sha256(config)
    write_json(out / "config.json", {"label": REAL_LABEL, "config": config, "config_sha256": sha})
    context = RunContext(model_id=args.model_id, config_version=REAL_CONFIG_VERSION, config_sha256=sha,
                         synthetic=False)
    logs = run_stage(specs, store, positions, th, args.budgets, config["travel_caps_cm"], config["r_seeds"], context)
    del store
    write_logs(out / "logs.jsonl", logs)
    _, truth = load_manifest(args.manifest)
    truth.save(out / "ground_truth.json")
    scored = score_stage(logs, truth, args.primary_budget, args.bootstrap_B, args.seed)
    results = {"label": REAL_LABEL, "synthetic": False, "config": config, "config_sha256": sha,
               "positions": positions.to_dict(), "thresholds": th.to_dict(),
               "files": {n: file_sha256(out / n) for n in ("ground_truth.json", "logs.jsonl")},
               "n_logs": len(logs), **scored,
               "diagnostics": diagnostics_stage(logs, positions, args.budgets, config["travel_caps_cm"])}
    write_json(out / "results.json", results)
    print(f"{REAL_LABEL}: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
