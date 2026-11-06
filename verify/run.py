"""Offline replay runner: generate -> run (no truth) -> score (see verify/README.md section 8).

    uv run --no-sync python -m verify.run --synthetic --n-arr 30 --seed 0 --budgets 0..4 \\
        [--cap-steps 1,2,3,4,6,inf | none] [--block-anywhere] [--positions configs/positions.json] \\
        [--r-seeds 20] [--bootstrap-B 10000] [--primary-budget 3] [--tag TAG] [--out runs/synthetic-<tag>/]

Every --synthetic output is labelled SYNTHETIC: the generator parameters and thresholds are
placeholders, and the numbers are pipeline checks, not study results.

Real or simulated data:

    uv run --no-sync python -m verify.run --manifest MANIFEST.json --evidence evidence.jsonl \\
        [--view-labels view_labels.json] --positions P [--candidate frozen_candidate.json | --placeholder-thresholds] \\
        [--side side.jsonl [--model A=model_A.json --model A-nocue=model_A-nocue.json]] [budgets, caps, B as above] \\
        [--gate-off-ablation] [--tau-match-sweep]

The evidence file comes from perception/evidence.py or evidence_v03.py (EvidenceStore.to_jsonl format). The same run
stage replays it with no truth; the manifest is re-read for truth only after the run, and the optional sim view labels
(cmpe249.sim-view-labels-set.v1) are read only after the run too, for the geometric no-diagnostic-view share.
Thresholds: --candidate reads a frozen select-v4 candidate through verify.select.require_freeze_eligible (sidecar,
eligibility, synthetic marks); without it the placeholders are used, which a simulation manifest with test arrangements
refuses unless --placeholder-thresholds is given (pipeline tests on hand-built sets only). --side gives every env the
left/right descriptor of each requested view (verify/side.py) and adds the direction-rule control Dir; each --model
adds a frozen ranker (verify/ranker.py) whose recorded features, thresholds and candidate must match. A manifest with a
top-level "simulation": true (sim/episodes.py) is labelled SIMULATION everywhere; any other is a REAL PILOT. It writes
config.json, logs.jsonl, ground_truth.json, results.json and table.md (scored on test, or on dev when the manifest has
no test arrangement). results.json and table.md lead with the primary contrast A - G (b = 3, uncapped, per lighting;
inconclusive when its 95 % interval contains 0) and carry the predeclared common-target analysis.

Uncertainty (docs/SIM_PREREGISTRATION.md section 8): the paired arrangement-level cluster bootstrap and the
percentile intervals of the single rates (FRR, removal recall, coverage of each policy row on test, per lighting)
are stratified by quota slot class (occluder family, M16 split by side) when every arrangement of the manifest
carries its family and every stratum holds at least two test arrangements; otherwise (the synthetic generator,
legacy pilot manifests, singleton strata) they are unstratified and flagged as such. With no test arrangement no
bootstrap runs, and the outputs say so. FRR is reported per episode and per arrangement (any false removal).

Declared test analyses (real-v6; docs/SIM_PREREGISTRATION.md sections 1, 4, 9 and 10), all reported, none selected:
each multi-view policy against S (BOOTSTRAP_PAIRS); per-family breakdowns at two levels, the occluder family
(`by_occluder_family`, M16's sides pooled) and the quota slot class (`by_family`), each with metrics, the target
composition and A - G per family on test; the occlusion-gate-off ablation (--gate-off-ablation: F and R with
tau_occ = 1.0, logged as F-gateoff / R-gateoff, with F - F-gateoff and R - R-gateoff on test at the primary setting);
the risk-coverage sweep of tau_match over the frozen candidate's predeclared grid plus a predeclared extension above
the frozen point (--tau-match-sweep, needs --candidate; logged as <policy>-tm<value>); and, with --view-labels, the
geometric diagnostic-view coverage and the gate's per-view rates against the geometric labels (verify.select's
ViewRates). They are computed after the run, from the truth, the labels and the evidence file read again from disk.

Every output is removed from the output directory before a run starts, so a run that fails part-way never leaves
an earlier run's results next to its own config (RUN_OUTPUTS, REAL_OUTPUTS).

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

Scoring is on the TEST split only (dev is reserved for threshold calibration, verify/select.py
procedure select-v4, which the synthetic placeholders skip). Dev metrics are kept in results.json with
scope "dev".
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from analysis.bootstrap import MIN_STRATUM_SIZE, paired_diff, rate_intervals, resolve_strata
from analysis.common_target import COMMON_TARGETS, common_target_diff
from analysis.metrics import NO_DIAG_SOURCE_LABELS, compute_all, ratio, seed_decisions
from verify.manifest import (
    DEFAULT_ENVIRONMENT,
    UNSTRATIFIED_NOTE,
    ArrangementRecord,
    GeometricLabels,
    episodes_from_arrangements,
    load_geometric_labels,
    load_manifest_info,
)
from verify.policies import (
    DEFAULT_R_SEEDS,
    GMYOPIC_LAM_DEG,
    Dir,
    F,
    FullSweep,
    G,
    GMyopic,
    Ranker,
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
    Condition,
    Decision,
    EpisodeLog,
    EpisodeResult,
    EpisodeSpec,
    EvidenceStore,
    GroundTruth,
    Lighting,
    Positions,
    ReplayError,
    RunContext,
    Split,
    Thresholds,
    as_results,
    cap_cm_from_steps,
    config_sha256,
    derive_seed,
    file_sha256,
    to_jsonable,
)
from verify.side import SideData, SideEnv, SideStore
from verify.stopping import StoppingRule
from verify.synthetic import SyntheticParams, generate

CONFIG_VERSION = "synthetic-v8"   # v8 (2026-11-06): the new pairs against S (G-lookahead - S, G-myopic - S)
# v7 (2026-11-03): single-rate CIs; singleton strata fall back to unstratified
# v6 (2026-11-03): simulation flag hashed; stratified-bootstrap flag; FRR per arr.
# v5 (2026-11-02): cap_steps hashed as floats; diagnostics per lighting
# v4 (2026-11-02): primary G = lookahead off, G-lookahead added; per-lighting results
MODEL_ID = "synthetic-no-model"
# G is primary G (D-norm, lookahead off; DECISIONS 2026-10-12). G-lookahead is the same scorer with the
# reachability lookahead, so G - G-lookahead isolates the lookahead. G-myopic is the legacy pre-D-G ablation.
# Dir (the direction-rule control) runs with side descriptors (--side); A and A-nocue with their frozen model files
# (--model). A pair or row whose policy did not run is skipped.
MULTI_VIEW = ("F", "R", "G", "G-lookahead", "G-myopic", "Dir", "A", "A-nocue")
OPTIONAL_POLICIES = ("Dir", "A", "A-nocue")   # run only with --side / --model
G_LOOKAHEAD = {"G": False, "G-lookahead": True}   # recorded in the config (and its hash)
# A - G at the primary budget, uncapped, is THE primary contrast (SIM_PREREGISTRATION section 1); every other pair and
# every capped setting is secondary. A - G-lookahead across the caps: researcher's decision 2026-11-05 item 4. A - Dir:
# explanatory (post-hoc development origin, notes/2026-11-05-side-descriptor.md).
PRIMARY_PAIR = ("A", "G")
# Each multi-view policy against S (prereg sections 1 and 9): A, A-nocue, Dir, G-lookahead and G-myopic - S added in
# real-v6 next to G, F and R - S.
BOOTSTRAP_PAIRS = (PRIMARY_PAIR, ("A", "F"), ("A", "R"), ("A", "A-nocue"), ("A", "G-lookahead"), ("A", "Dir"),
                   ("G", "F"), ("G", "R"), ("G", "S"), ("F", "S"), ("R", "S"), ("A", "S"), ("A-nocue", "S"),
                   ("Dir", "S"), ("G-lookahead", "S"), ("G-myopic", "S"), ("G", "G-lookahead"),
                   ("G", "G-myopic"))
# Occlusion-gate-off ablation (prereg sections 4 and 10.5; --gate-off-ablation): F and R only, with tau_occ = 1.0, so
# no view is ever flagged occluded and the gate is the quality cut alone. G is left out because tau_occ is also G's
# occlusion flag (it would change G's selection, not only the gate). Reported only, never selected; the paired
# contrasts run on test at the primary setting only (primary budget, uncapped).
GATE_OFF_SUFFIX = "gateoff"
GATE_OFF_TAU_OCC = 1.0
GATE_OFF_POLICIES = ("F", "R")
GATE_OFF_PAIRS = tuple((p, f"{p}-{GATE_OFF_SUFFIX}") for p in GATE_OFF_POLICIES)
# Risk-coverage sweep of tau_match (prereg section 9; --tau-match-sweep): every value of the frozen candidate's
# predeclared tau_match grid except the frozen one, every other threshold frozen, at the primary budget, uncapped only;
# logged as "<policy>-tm<value>". The frozen point of each curve is the policy's own primary row. Reported only.
TM_SWEEP_PREFIX = "tm"
# The sweep's extension above the frozen point, declared on 2026-11-06, before the freeze: the frozen tau_match 0.7 is
# the top of the candidate's grid, so the grid alone gives curves that stop at the frozen point. The extension
# continues the grid's 0.05 step up to 0.9, so the risk-coverage curve shows both sides. Reported only, never selected.
TAU_MATCH_SWEEP_ABOVE = (0.75, 0.8, 0.85, 0.9)
TM_SWEEP_POLICIES = ("S", "F", "R", "G", "G-lookahead", "G-myopic", "Dir", "A", "A-nocue", "FULL")   # those that run
# Per-family breakdowns (prereg section 9; H3) at two levels: the occluder family (S8, M16 with both sides pooled,
# IND28, white_box, ycb_box; design cases apart) and the quota slot class, the bootstrap stratum
# (verify.manifest.ArrangementRecord.stratum: M16 split by side). table.md shows these policies' uncapped
# primary-budget rows.
FAMILY_TABLE_POLICIES = ("F", "R", "G", "G-lookahead", "Dir", "A", "A-nocue", "FULL")
# H1 (prereg sec. 10): the same selector with the k-distinct rule vs the first-clear-miss rule (k = 1).
H1_PAIRS = tuple((x, f"{x}-fcm") for x in MULTI_VIEW)
# "selection can matter" diagnostics (lit report sec. 4): G's path against these, per budget and cap
DISAGREE_PAIRS = (("G", "F"), ("G", "R"), ("G", "G-lookahead"), ("G", "G-myopic"), ("G-myopic", "F"), ("A", "G"))
BOOTSTRAP_METRICS = ("false_removal_rate", "removal_recall", "coverage")
RATE_CI_METRICS = BOOTSTRAP_METRICS   # single rates with a percentile 95% interval (prereg section 8), test rows only
SCOPES = (Split.TEST, Split.DEV)
# Results are reported separately for L1 and L2 queries (DECISIONS 2026-11-02); "lighting" is the query's
# lighting. A number pooled over lighting is a labelled secondary line only, never a headline.
REPORT_LIGHTINGS = tuple(x.value for x in Lighting)
POOLED = "pooled"
POOLED_DIAGNOSTIC = POOLED   # the lighting of a diagnostics row over both lightings (secondary section), as for metrics
ROLE = {"S": "actual effort (1 view, budget-independent)",
        "FULL": "higher-effort reference (5 views, ignores budget)",
        "A": "learned ranker (frozen model)", "A-nocue": "ablation: A without the obstruction-derived features",
        "Dir": "explanatory control (post-hoc development origin)"}

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
    p.add_argument("--view-labels", default=None,
                   help="real/sim data: view_labels.json (cmpe249.sim-view-labels-set.v1), read only after the run, for "
                        "the geometric no-diagnostic-view share (context, not a ceiling); without it the share falls "
                        "back to the manifest's predeclared indistinguishable flag, labelled as such")
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
    p.add_argument("--positions", default=None,
                   help="positions JSON (default configs/positions.json; required for a simulation manifest); its "
                        "contents enter the config hash")
    p.add_argument("--r-seeds", type=int, default=len(DEFAULT_R_SEEDS), help="R seeds per episode (default 20)")
    p.add_argument("--bootstrap-B", type=int, default=10000, help="bootstrap draws (default 10000)")
    p.add_argument("--primary-budget", type=int, default=3, help="budget for the primary table (default 3)")
    p.add_argument("--tag", default=None, help="run tag (default n<N>-s<seed>)")
    p.add_argument("--out", default=None, help="output dir (default runs/synthetic-<tag>/ or runs/real-<tag>/)")
    p.add_argument("--model-id", default="perception.evidence-v0", help="real data: perception model id for the logs")
    p.add_argument("--candidate", default=None,
                   help="real/sim data: frozen_candidate.json from verify.select, read only through "
                        "require_freeze_eligible (its .sha256 sidecar must match and it must be eligible); its "
                        "thresholds replace the placeholders. Required for a simulation manifest with test "
                        "arrangements")
    p.add_argument("--k-sensitivity", type=lambda x: () if x == "none" else tuple(int(v) for v in x.split(",")),
                   default=(3,), help="real/sim data: k values of the declared sensitivity runs (default 3, or none)")
    p.add_argument("--lam-sensitivity", type=lambda x: () if x == "none" else tuple(float(v) for v in x.split(",")),
                   default=(0.7, 0.9, 1.2),
                   help="real/sim data: lam values of the declared sensitivity runs for G and G-lookahead "
                        "(default 0.7,0.9,1.2, or none)")
    p.add_argument("--placeholder-thresholds", action="store_true",
                   help="real/sim data: allow a simulation manifest with test arrangements to run on the placeholder "
                        "thresholds (pipeline tests on hand-built sets only; every output stays labelled 'placeholder "
                        "thresholds, not frozen')")
    p.add_argument("--side", default=None,
                   help="real/sim data: side descriptors (perception/side_descriptor.py rows on the same jobs): every "
                        "env then reveals each requested view's own descriptor, and the direction-rule control Dir "
                        "runs")
    p.add_argument("--model", action="append", default=None,
                   help="real/sim data: NAME=PATH, a frozen ranker model file (verify/ranker.py save_model), e.g. "
                        "A=model_A.json; needs --side and --candidate (the model's features, thresholds and "
                        "candidate must match); "
                        "repeatable")
    p.add_argument("--gate-off-ablation", action="store_true",
                   help="real/sim data: the occlusion-gate-off ablation, F and R only with tau_occ = 1.0 (logged as "
                        "F-gateoff and R-gateoff), at the primary budget under every cap; reported only, never "
                        "selected")
    p.add_argument("--tau-match-sweep", action="store_true",
                   help="real/sim data: the risk-coverage sweep of tau_match over the frozen candidate's predeclared "
                        "tau_match_grid (needs --candidate), every other threshold frozen, at the primary budget, "
                        "uncapped (logged as <policy>-tm<value>); reported only, never selected")
    return p


def _rel(path: Path) -> str:
    """Repo-relative path when inside the repo (keeps the config hash machine-independent)."""
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _cap_steps_config(cap_steps: Sequence[float]) -> list[float | None]:
    """[None (uncapped), *caps as floats] for the hashed config. One spelling per cap: the parser default was ints and
    --cap-steps 1,2,3,4,6 parsed to floats, so the same caps hashed two ways (review; synthetic-v5 / real-v2)."""
    return [None, *(float(c) for c in cap_steps)]


def build_config(args: argparse.Namespace, positions: Positions, th: Thresholds,
                 params: SyntheticParams) -> dict[str, Any]:
    """Everything that determines the run; hashed into every log line."""
    pos_path = Path(args.positions or DEFAULT_POSITIONS_PATH)
    return {"config_version": CONFIG_VERSION, "model_id": MODEL_ID, "synthetic": True, "simulation": False,
            "bootstrap_stratified": False,
            "positions_file": _rel(pos_path), "positions_file_sha256": file_sha256(pos_path),
            "n_arr": args.n_arr, "seed": args.seed, "budgets": list(args.budgets),
            "nominal_step_cm": positions.nominal_step_cm(), "travel_tol_cm": TRAVEL_TOL_CM,
            "cap_steps": _cap_steps_config(args.cap_steps),
            "travel_caps_cm": [None, *(cap_cm_from_steps(c, positions) for c in args.cap_steps)],
            "g_lookahead": dict(G_LOOKAHEAD),
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
              context: RunContext, side: SideData | None = None,
              rankers: Sequence[Ranker] = (), sensitivity: Sequence[tuple[str, Thresholds]] = (),
              primary_budget: int | None = None,
              tau_match_sweep: Sequence[tuple[str, Thresholds]] = ()) -> list[EpisodeLog]:
    """Every policy x budget x cap on every episode, each run on a fresh ReplayEnv.

    Envs are built from opaque ReplayKeys (minted here, before the store is sealed), never from
    the spec; the log header is joined onto each header-free run afterwards. With side descriptors every env is a
    SideEnv (each revealed view carries its own stop's descriptor, nothing else), and Dir runs; each frozen ranker
    (A, A-nocue) runs like G, with the same rules, budgets and caps. sensitivity: (suffix, thresholds) pairs run at the
    primary budget under every cap and logged as "<policy>-<suffix>" (SENSITIVITY_SUFFIX): a k value runs every
    multi-view policy with that k-rule; a lam value runs G and G-lookahead with that weight (reported only); the
    GATE_OFF_SUFFIX setting (gate_off_setting) runs F and R only. tau_match_sweep: (suffix, thresholds) pairs from
    tau_match_sweep_settings, each run once at the primary budget, uncapped: S (first-clear-miss rule) and FULL at their
    own budgets, then every multi-view policy with the k-rule, logged as "<policy>-<suffix>" (reported only).
    """
    if rankers and side is None:
        raise ValueError("the rankers read the side descriptor: pass side descriptors with ranker models")
    rule = StoppingRule(th)
    fcm = StoppingRule.first_clear_miss(th)  # F/G/R-fcm, and S (one view, removed on a single clear miss)
    keys = store.replay_keys()
    side_store = side() if side is not None else None
    logs: list[EpisodeLog] = []

    for spec in specs:
        key = keys[spec.episode_id]
        reveal = side_store.bind(spec.episode_id) if side_store is not None else None

        def env(budget: Budget, key=key, reveal=reveal) -> ReplayEnv | SideEnv:
            e = ReplayEnv(key, store, positions, budget, context)
            return e if reveal is None else SideEnv(e, reveal)

        logs.append(EpisodeLog.from_run(spec, run_episode(S(), env(Budget(0)), fcm)))
        logs.append(EpisodeLog.from_run(spec, run_episode(FullSweep(), env(Budget(4)), rule)))
        for cap in caps:
            for b in budgets:
                budget = Budget(b, cap)
                for r in (rule, fcm):
                    for pol in (F(), G.from_thresholds(th, lookahead=G_LOOKAHEAD["G"]),
                                G.from_thresholds(th, lookahead=G_LOOKAHEAD["G-lookahead"]),
                                GMyopic.from_thresholds(th),
                                *((Dir.from_thresholds(th),) if side_store is not None else ()), *rankers):
                        logs.append(EpisodeLog.from_run(spec, run_episode(pol, env(budget), r)))
                    seed_logs, _ = run_r_seeds(lambda bu=budget, e=env: e(bu), r, r_seeds, r_seed_salt(spec),
                                               spec=spec)
                    logs.extend(seed_logs)
        for suffix, t in sensitivity:
            k_run = suffix.startswith("k")
            gate_off = suffix == GATE_OFF_SUFFIX   # F and R only (G reads tau_occ as its occlusion flag)
            r = StoppingRule(t)
            for cap in caps:
                budget = Budget(primary_budget, cap)
                pols = ((F(), G.from_thresholds(t, lookahead=False), G.from_thresholds(t, lookahead=True),
                         GMyopic.from_thresholds(t), *((Dir.from_thresholds(t),) if side_store is not None else ()),
                         *rankers) if k_run
                        else (F(),) if gate_off
                        else (G.from_thresholds(t, lookahead=False), G.from_thresholds(t, lookahead=True)))
                for pol in pols:
                    run = run_episode(pol, env(budget), r, name=f"{pol.name}-{suffix}")
                    logs.append(EpisodeLog.from_run(spec, run))
                if k_run or gate_off:
                    seed_logs, _ = run_r_seeds(lambda bu=budget, e=env: e(bu), r, r_seeds, r_seed_salt(spec),
                                               spec=spec, name=f"R-{suffix}")
                    logs.extend(seed_logs)
        for suffix, t in tau_match_sweep:   # only tau_match differs from th: the policies keep th's tau_occ and lam
            r, r1 = StoppingRule(t), StoppingRule.first_clear_miss(t)
            logs.append(EpisodeLog.from_run(spec, run_episode(S(), env(Budget(0)), r1, name=f"S-{suffix}")))
            logs.append(EpisodeLog.from_run(spec, run_episode(FullSweep(), env(Budget(4)), r, name=f"FULL-{suffix}")))
            budget = Budget(primary_budget, None)
            for pol in (F(), G.from_thresholds(t, lookahead=G_LOOKAHEAD["G"]),
                        G.from_thresholds(t, lookahead=G_LOOKAHEAD["G-lookahead"]), GMyopic.from_thresholds(t),
                        *((Dir.from_thresholds(t),) if side_store is not None else ()), *rankers):
                logs.append(EpisodeLog.from_run(spec, run_episode(pol, env(budget), r, name=f"{pol.name}-{suffix}")))
            seed_logs, _ = run_r_seeds(lambda bu=budget, e=env: e(bu), r, r_seeds, r_seed_salt(spec),
                                       spec=spec, name=f"R-{suffix}")
            logs.extend(seed_logs)
    return logs


def sensitivity_settings(th: Thresholds, ks: Sequence[int], lams: Sequence[float]) -> list[tuple[str, Thresholds]]:
    """The declared sensitivity runs (SIM_PREREGISTRATION §0, §7): k in ks with lam fixed, lam in lams with k fixed."""
    return ([(f"k{k}", dataclasses.replace(th, k=k)) for k in ks]
            + [(f"lam{lam:g}", dataclasses.replace(th, lam=lam)) for lam in lams])


def gate_off_setting(th: Thresholds) -> tuple[str, Thresholds]:
    """The occlusion-gate-off ablation as a sensitivity setting (run_stage runs it for F and R only): tau_occ = 1.0,
    so no view is flagged occluded (obstruction <= 1 always passes) and the clear-miss gate is the quality cut alone."""
    return GATE_OFF_SUFFIX, dataclasses.replace(th, tau_occ=GATE_OFF_TAU_OCC)


class SweepRefused(ValueError):
    """--tau-match-sweep refused: no predeclared tau_match grid in the candidate, one that does not hold the frozen
    value, or an extension value at or below the frozen value."""


def tau_match_grid_of(candidate: Mapping[str, Any], th: Thresholds) -> tuple[float, ...]:
    """The frozen candidate's predeclared tau_match grid (settings.tau_match_grid), sorted. SweepRefused when the
    candidate has none, when a value is not a number in [0, 1], or when the frozen tau_match is not one of its values
    (the candidate was chosen from that grid, so a grid without it is not the candidate's)."""
    grid = (candidate.get("settings") or {}).get("tau_match_grid")
    if not isinstance(grid, list) or not grid:
        raise SweepRefused("the candidate records no predeclared tau_match grid (settings.tau_match_grid); the sweep "
                           "never invents one")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0 for v in grid):
        raise SweepRefused(f"the candidate's tau_match_grid {grid} is not a list of numbers in [0, 1]")
    if th.tau_match not in grid:
        raise SweepRefused(f"the frozen tau_match {th.tau_match:g} is not in the candidate's tau_match_grid {grid}")
    return tuple(sorted({float(v) for v in grid}))


def tau_match_sweep_values(th: Thresholds, grid: Sequence[float],
                           above: Sequence[float] = TAU_MATCH_SWEEP_ABOVE) -> tuple[float, ...]:
    """The tau_match values the sweep runs, sorted: the candidate's grid (tau_match_grid_of) without the frozen value,
    plus the predeclared extension above it (TAU_MATCH_SWEEP_ABOVE). SweepRefused when an extension value is not a
    number in [0, 1] or is at or below the frozen tau_match (the extension only continues the curve upwards)."""
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0 for v in above):
        raise SweepRefused(f"the sweep extension {list(above)} is not a list of numbers in [0, 1]")
    low = [v for v in above if v <= th.tau_match]
    if low:
        raise SweepRefused(f"the sweep extension value(s) {low} are at or below the frozen tau_match {th.tau_match:g}: "
                           "the extension continues the curve above the frozen point only")
    return tuple(sorted({float(v) for v in (*grid, *above)} - {th.tau_match}))


def tau_match_sweep_settings(th: Thresholds, grid: Sequence[float]) -> list[tuple[str, Thresholds]]:
    """("tm<value>", th with that tau_match) for every given value except the frozen th.tau_match, in the given order
    (main_real passes tau_match_sweep_values): the other thresholds, k and lam stay frozen."""
    return [(f"{TM_SWEEP_PREFIX}{v:g}", dataclasses.replace(th, tau_match=float(v)))
            for v in grid if float(v) != th.tau_match]


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


NO_BOOTSTRAP_REASON = ("this run has no test arrangement, and the paired and single-rate bootstraps run on the test "
                       "split only")
NO_BOOTSTRAP_NOTE = f"no bootstrap ran: {NO_BOOTSTRAP_REASON}"
STRATIFIED_NOTE = ("stratified by quota slot class (occluder family, M16 split by side; design cases apart): each "
                   "replicate resamples arrangements within every stratum and keeps its count")


def stratification_info(strata: Mapping[str, str] | None, arrangement_ids: Sequence[str]) -> dict[str, Any]:
    """How the bootstrap ran over the given (test) arrangements: {"ran", "stratified", "strata": {stratum:
    n_arrangements}, "singleton_strata", "note"}. With no arrangement no bootstrap runs ("ran" false; review: a dev-only
    run said "stratified" with empty strata). With a stratum of fewer than two arrangements the bootstrap falls back
    to unstratified (analysis.bootstrap.resolve_strata) and the note says "unstratified: singleton strata"."""
    arrs = sorted(set(arrangement_ids))
    if not arrs:
        return {"ran": False, "stratified": False, "strata": {}, "singleton_strata": [], "note": NO_BOOTSTRAP_NOTE}
    if strata is None:
        return {"ran": True, "stratified": False, "strata": {}, "singleton_strata": [], "note": UNSTRATIFIED_NOTE}
    used, singles, note = resolve_strata(arrs, strata)
    counts: dict[str, int] = {}
    for a in arrs:
        counts[strata[a]] = counts.get(strata[a], 0) + 1
    return {"ran": True, "stratified": used is not None, "strata": dict(sorted(counts.items())),
            "singleton_strata": list(singles), "note": note or STRATIFIED_NOTE}


def score_stage(logs: Sequence[EpisodeLog], truth: GroundTruth, primary_budget: int,
                B: int, seed: int, *, strata: Mapping[str, str] | None = None,
                no_diagnostic: Mapping[str, bool] | None = None,
                common_target: Mapping[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    """Metrics per (scope, lighting, policy, budget, cap) and paired bootstraps at the primary budget, per lighting.

    lighting is the query's lighting (the episode's). The "L1" and "L2" rows are the primary report
    (DECISIONS 2026-11-02); a "pooled" row (both lightings) is kept only as a labelled secondary line.
    The bootstrap runs within each lighting; its resampling unit is still the arrangement (with its
    condition episodes at that lighting), and with strata ({arrangement_id: stratum}, verify.manifest.bootstrap_strata)
    it resamples within each stratum, keeping the design's quota counts, unless a stratum is a singleton (then it
    is unstratified and flagged). No bootstrap is run on pooled data.
    Every L1 / L2 test metrics row also carries "rate_ci95": percentile 95% intervals of its FRR, removal recall and
    coverage from the same arrangement-level cluster bootstrap (analysis.bootstrap.rate_intervals: the same draws as
    the paired rows of that lighting). R enters with its seed-averaged decisions, as in its point estimate: the
    bootstrap resamples arrangements, never seeds, so R's interval is for its expected rate over its own
    randomisation (seeds add no samples; with 20 seeds averaged the seed noise is small next to the arrangement
    spread), and R's episodes that share a random stream (one per arrangement and lighting) stay together.
    no_diagnostic (episode_id -> no diagnostic stop, from the view labels) feeds the geometric no-diagnostic-view
    share; None falls back to the predeclared indistinguishable flag (analysis.metrics).
    "primary" lists the primary contrast's rows (PRIMARY_PAIR at the primary budget, uncapped, per lighting) with
    "inconclusive" = the 95% interval contains 0 (never read as equivalence). common_target ({arrangement_id: (occluder
    family, target id)}) adds the predeclared secondary A - G on common targets (analysis.common_target) at the same
    setting, per lighting. GATE_OFF_PAIRS run with the other pairs at the primary budget, uncapped only, when the
    gate-off rows exist.
    """
    groups = group_results(logs)
    # R's per-seed decisions, for its per-arrangement FRR (computed per base seed: R's condition-free salt makes the
    # episodes of one arrangement and lighting share a random stream, so they are not independent)
    seed_logs: dict[RunKey, list[EpisodeLog]] = {}
    for g in logs:
        if g.seed is not None:
            seed_logs.setdefault((g.policy, g.budget_additional_views, g.travel_cap_cm), []).append(g)
    seed_removed = {k: seed_decisions(v) for k, v in seed_logs.items()}
    caps = sorted({k[2] for k in groups}, key=_cap_key)
    metrics: list[dict[str, Any]] = []
    for scope in SCOPES:
        for key in sorted(groups, key=lambda k: (k[0], k[1], _cap_key(k[2]))):
            res = [r for r in groups[key] if r.split is scope]
            for light in (*Lighting, None):            # None = pooled over lighting (secondary)
                sub = res if light is None else [r for r in res if r.lighting is light]
                if not sub:
                    continue
                m = compute_all(sub, truth, no_diagnostic, seed_removed.get(key))
                row = {"scope": scope.value, "lighting": POOLED if light is None else light.value,
                       "policy": key[0], "budget": key[1], "travel_cap_cm": key[2], "role": ROLE.get(key[0], ""), **m}
                if scope is Split.TEST and light is not None:   # single-rate intervals: test, per lighting only
                    row["rate_ci95"] = _rate_ci(sub, truth, B, seed, strata)
                metrics.append(row)
    boots: list[dict[str, Any]] = []
    test = {k: [r for r in v if r.split is Split.TEST] for k, v in groups.items()}
    pairs = [(a + v if a != "S" else a, b + v if b != "S" else b)
             for v in ("", "-fcm") for a, b in BOOTSTRAP_PAIRS] + list(H1_PAIRS)
    for light in Lighting:
        sub = {k: [r for r in v if r.lighting is light] for k, v in test.items()}
        for cap in caps:
            # the gate-off contrasts at the primary setting only (skipped, as any pair, when the ablation did not run)
            for a, b in (pairs + list(GATE_OFF_PAIRS) if cap is None else pairs):
                ka = _lookup(sub, a, primary_budget, cap)
                kb = _lookup(sub, b, primary_budget, cap)
                if ka is None or kb is None or not sub[ka] or not sub[kb]:
                    continue
                same = identical_runs(sub[ka], sub[kb])
                for metric in BOOTSTRAP_METRICS:
                    br = paired_diff(sub[ka], sub[kb], truth, metric, B=B, seed=seed, strata=strata)
                    boots.append({"scope": Split.TEST.value, "lighting": light.value, "travel_cap_cm": cap,
                                  "budget": primary_budget, "identical_runs": same, **br.to_dict()})
    test_arrs = sorted({r.arrangement_id for v in test.values() for r in v})
    primary = [{**r, "inconclusive": None if r["ci_lo"] is None else bool(r["ci_lo"] <= 0.0 <= r["ci_hi"])}
               for r in boots if (r["policy_a"], r["policy_b"]) == PRIMARY_PAIR and r["travel_cap_cm"] is None]
    common: list[dict[str, Any]] = []
    if common_target is not None:
        for light in Lighting:
            sub = {k: [r for r in v if r.lighting is light] for k, v in test.items()}
            ka = _lookup(sub, PRIMARY_PAIR[0], primary_budget, None)
            kb = _lookup(sub, PRIMARY_PAIR[1], primary_budget, None)
            if ka is None or kb is None or not sub[ka] or not sub[kb] or \
                    not any(common_target[r.arrangement_id][1] in COMMON_TARGETS for r in sub[ka]):
                continue
            for metric in BOOTSTRAP_METRICS:
                common.append({"scope": Split.TEST.value, "lighting": light.value, "budget": primary_budget,
                               "travel_cap_cm": None, **common_target_diff(
                                   sub[ka], sub[kb], truth, {a: f for a, (f, _) in common_target.items()},
                                   {a: t for a, (_, t) in common_target.items()}, metric, B=B, seed=seed,
                                   strata=strata)})
    return {"metrics": metrics, "bootstrap": boots, "primary": primary, "common_target": common,
            "bootstrap_stratification": stratification_info(strata, test_arrs)}


def _rate_ci(results: Sequence[EpisodeResult], truth: GroundTruth, B: int, seed: int,
             strata: Mapping[str, str] | None) -> dict[str, Any]:
    """{metric: {"estimate", "ci_lo", "ci_hi", "n_undefined"}, "B", "seed", "n_arrangements", "n_episodes",
    "stratified", "singleton_strata", "stratification_note"} for one policy row (rate_intervals)."""
    iv = rate_intervals(results, truth, RATE_CI_METRICS, B=B, seed=seed, strata=strata)
    one = next(iter(iv.values()))
    return {**{m: {"estimate": r.estimate, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "n_undefined": r.n_undefined}
               for m, r in iv.items()},
            "B": B, "seed": seed, "n_arrangements": one.n_arrangements, "n_episodes": one.n_episodes,
            "stratified": one.stratified, "singleton_strata": list(one.singleton_strata),
            "stratification_note": one.stratification_note}


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


# ---------------------------------------------------------------- declared test analyses (real-v6, after the run)

FAMILY_ALL = "all"   # the row over every family in the diagnostic-view coverage
SLOT_CLASS, OCCLUDER_FAMILY = "slot_class", "occluder_family"   # the two family levels (results keys below)
FAMILY_LEVELS = {OCCLUDER_FAMILY: "by_occluder_family", SLOT_CLASS: "by_family"}
FAMILY_LEVEL_TEXT = {
    OCCLUDER_FAMILY: ("the occluder family (S8, M16 with both sides pooled, IND28, white_box, ycb_box; design cases "
                      "and tabletop records apart, as in the strata)"),
    SLOT_CLASS: ("the quota slot class (the bootstrap stratum: occluder family, M16 split by side, design cases "
                 "apart)")}
FAMILY_NOTE = ("Descriptive evidence for H3 (prereg section 1), not a new primary. The family is {level}. Each family "
               "carries its target composition: a difference between families can reflect their targets as well as "
               "their occluders")
FAMILY_SKIPPED_NOTE = ("the manifest lacks occluder_family for some arrangement, so there is no per-family breakdown "
                       "(the family is the occluder family or the quota slot class)")
FAMILY_BOOTSTRAP_NOTE = ("paired arrangement-level cluster bootstrap of A - G per family and lighting at the primary "
                         "budget, uncapped, on test, {strata}; the main bootstrap's B with a seed per family and "
                         "lighting derived from the main seed (family_bootstrap_seed); inconclusive when the 95% "
                         "interval contains 0. Descriptive (H3), not a new primary")
FAMILY_STRATA_TEXT = {
    OCCLUDER_FAMILY: ("stratified by slot class within the family (by side for M16, falling back to unstratified, "
                      "flagged, when a side has fewer than 2 arrangements; a single stratum, i.e. unstratified, for "
                      "the others)"),
    SLOT_CLASS: "unstratified within the family (one slot class)"}
FAMILY_ALL_SKIPPED_NOTE = (f"no per-family A - G bootstrap ran: every family with test episodes has fewer than "
                           f"{MIN_STRATUM_SIZE} test arrangements")
FAMILY_SINGLETON_REASON = (f"fewer than {MIN_STRATUM_SIZE} test arrangements: a cluster bootstrap over a single "
                           "arrangement only redraws it and has no interval")
NO_LABELS_NOTE = "needs --view-labels (the geometric labels, read after the run): not computed"


def family_bootstrap_seed(seed: int, family: str, lighting: str) -> int:
    """The per-family A - G bootstrap seed of one family and lighting, derived deterministically from the main seed
    (verify.schema.derive_seed), so equal-size families never share draws. The same family name gets the same seed at
    both levels (S8 is the same arrangements there)."""
    return derive_seed(seed, "by-family", family, lighting)


def _seed_removed(logs: Sequence[EpisodeLog]) -> dict[RunKey, dict[str, dict[int, bool]]]:
    """(policy, budget, cap) -> R's per-seed decisions (seed_decisions), as score_stage builds them."""
    seed_logs: dict[RunKey, list[EpisodeLog]] = {}
    for g in logs:
        if g.seed is not None:
            seed_logs.setdefault((g.policy, g.budget_additional_views, g.travel_cap_cm), []).append(g)
    return {k: seed_decisions(v) for k, v in seed_logs.items()}


def _occluder_family(a: ArrangementRecord) -> str:
    """The record's occluder family with the stratum's prefixes (tabletop:, design:) but M16's sides pooled."""
    base = a.occluder_family if a.env == DEFAULT_ENVIRONMENT else f"{a.env}:{a.occluder_family}"
    return f"design:{base}" if a.design_case else base   # type: ignore[return-value]


def family_of_records(records: Iterable[ArrangementRecord], level: str = SLOT_CLASS) -> dict[str, str] | None:
    """{arrangement_id: family}: at SLOT_CLASS the record's quota slot class (ArrangementRecord.stratum, the bootstrap
    stratum); at OCCLUDER_FAMILY its occluder family (M16-L and M16-R pooled as M16, the same prefixes). None when any
    record lacks its occluder family."""
    records = list(records)
    if not records or any(not a.has_slot for a in records):
        return None
    if level == SLOT_CLASS:
        return {a.arrangement_id: a.stratum for a in records}   # type: ignore[misc]
    if level == OCCLUDER_FAMILY:
        return {a.arrangement_id: _occluder_family(a) for a in records}
    raise ValueError(f"family level must be one of {sorted(FAMILY_LEVELS)}, got {level!r}")


def target_composition(records: Sequence[ArrangementRecord], family_of: Mapping[str, str]) -> list[dict[str, Any]]:
    """[{"scope", "family", "n_arrangements", "targets": {target_id: n_arrangements}}] per split and family, read from
    the manifest (the targets are assigned per slot within what each family supports, prereg section 2)."""
    out = []
    for scope in SCOPES:
        recs = [a for a in records if a.split is scope]
        for fam in sorted({family_of[a.arrangement_id] for a in recs}):
            mine = [a for a in recs if family_of[a.arrangement_id] == fam]
            out.append({"scope": scope.value, "family": fam, "n_arrangements": len(mine),
                        "targets": dict(sorted(Counter(a.target_id for a in mine).items()))})
    return out


def family_stage(logs: Sequence[EpisodeLog], truth: GroundTruth, records: Sequence[ArrangementRecord],
                 primary_budget: int, B: int, seed: int, no_diagnostic: Mapping[str, bool] | None = None,
                 level: str = SLOT_CLASS, groups: Mapping[RunKey, list[EpisodeResult]] | None = None,
                 seed_removed: Mapping[RunKey, Mapping[str, Mapping[int, bool]]] | None = None) -> dict[str, Any]:
    """Per-family breakdowns (prereg section 9) at one level (OCCLUDER_FAMILY or SLOT_CLASS): compute_all per scope,
    lighting (L1, L2) and family for every policy row in ORDER (S and FULL at their own effort, the others at the
    primary budget, uncapped and under every cap; not the sensitivity, gate-off or sweep rows), each family's target
    composition, and on test the paired cluster bootstrap of A - G per family and lighting (FAMILY_BOOTSTRAP_NOTE;
    at OCCLUDER_FAMILY stratified by slot class within the family when it has more than one, i.e. by side for M16),
    skipped and listed for a family with fewer than MIN_STRATUM_SIZE test arrangements. With a record that lacks its
    family nothing is computed ("ran" false). groups / seed_removed: group_results / _seed_removed of the logs, when
    the caller has them already."""
    family_of = family_of_records(records, level)
    if family_of is None:
        return {"ran": False, "level": level, "note": FAMILY_SKIPPED_NOTE, "composition": [], "metrics": [],
                "bootstrap": [], "bootstrap_skipped": []}
    slot_of = family_of_records(records, SLOT_CLASS)
    groups = group_results(logs) if groups is None else groups
    seed_removed = _seed_removed(logs) if seed_removed is None else seed_removed
    fams = sorted(set(family_of.values()))
    keys = sorted((k for k in groups if k[0] in ORDER and (k[1] == primary_budget or k[0] in ("S", "FULL"))),
                  key=lambda k: (k[0], k[1], _cap_key(k[2])))
    metrics: list[dict[str, Any]] = []
    for scope in SCOPES:
        for key in keys:
            res = [r for r in groups[key] if r.split is scope]
            for fam in fams:
                for light in Lighting:
                    sub = [r for r in res if r.lighting is light and family_of[r.arrangement_id] == fam]
                    if sub:
                        metrics.append({"scope": scope.value, "lighting": light.value, "family": fam,
                                        "policy": key[0], "budget": key[1], "travel_cap_cm": key[2],
                                        "role": ROLE.get(key[0], ""),
                                        **compute_all(sub, truth, no_diagnostic, seed_removed.get(key))})
    boots: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    ka, kg = ((p, primary_budget, None) for p in PRIMARY_PAIR)
    if not any(a.split is Split.TEST for a in records):
        how = NO_BOOTSTRAP_NOTE
    elif ka not in groups or kg not in groups:
        how = f"no per-family {' - '.join(PRIMARY_PAIR)} bootstrap: {PRIMARY_PAIR[0]} or {PRIMARY_PAIR[1]} did not run"
    else:
        for light in Lighting:
            for fam in fams:
                a, g = ([r for r in groups[k] if r.split is Split.TEST and r.lighting is light
                         and family_of[r.arrangement_id] == fam] for k in (ka, kg))
                if not a or not g:
                    continue
                arrs = sorted({r.arrangement_id for r in a})
                if len(arrs) < MIN_STRATUM_SIZE:
                    skipped.append({"scope": Split.TEST.value, "lighting": light.value, "family": fam,
                                    "n_arrangements": len(arrs), "reason": FAMILY_SINGLETON_REASON})
                    continue
                strata = {x: slot_of[x] for x in arrs}   # type: ignore[index]
                strata = strata if len(set(strata.values())) > 1 else None   # one slot class: unstratified
                same = identical_runs(a, g)
                fseed = family_bootstrap_seed(seed, fam, light.value)
                for metric in BOOTSTRAP_METRICS:
                    br = paired_diff(a, g, truth, metric, B=B, seed=fseed, strata=strata).to_dict()
                    boots.append({"scope": Split.TEST.value, "lighting": light.value, "family": fam,
                                  "travel_cap_cm": None, "budget": primary_budget, "identical_runs": same, **br,
                                  "inconclusive": None if br["ci_lo"] is None else
                                  bool(br["ci_lo"] <= 0.0 <= br["ci_hi"])})
        how = (FAMILY_BOOTSTRAP_NOTE.format(strata=FAMILY_STRATA_TEXT[level]) if boots or not skipped
               else FAMILY_ALL_SKIPPED_NOTE)
    return {"ran": True, "level": level, "note": FAMILY_NOTE.format(level=FAMILY_LEVEL_TEXT[level]), "families": fams,
            "composition": target_composition(records, family_of), "metrics": metrics, "bootstrap": boots,
            "bootstrap_skipped": skipped, "bootstrap_note": how, "bootstrap_B": B}


def tau_match_sweep_curves(metrics: Sequence[Mapping[str, Any]], th: Thresholds, grid: Sequence[float],
                           primary_budget: int) -> list[dict[str, Any]]:
    """Risk-coverage curves over tau_match (prereg section 9) from the scored metrics rows: per scope, lighting (L1,
    L2, pooled) and policy in TM_SWEEP_POLICIES, one point per grid value, the frozen value being the policy's own
    primary row ("frozen": true; S and FULL at their own effort). Each point has FRR, removal recall, coverage, resolved
    accuracy and the false-present rate among removed episodes (confusion removed -> present over n_removed), with
    counts. A policy with no sweep row has no curve."""
    index = {(m["scope"], m["lighting"], m["policy"], m["budget"], m["travel_cap_cm"]): m for m in metrics}
    scopes = [s.value for s in SCOPES if any(m["scope"] == s.value for m in metrics)]
    out = []
    for scope in scopes:
        for light in (*REPORT_LIGHTINGS, POOLED):
            for p in TM_SWEEP_POLICIES:
                b = 0 if p == "S" else 4 if p == "FULL" else primary_budget
                points = []
                for v in grid:
                    name = p if v == th.tau_match else f"{p}-{TM_SWEEP_PREFIX}{v:g}"
                    m = index.get((scope, light, name, b, None))
                    if m is None:
                        continue
                    o = m["overall"]
                    fp = o["confusion"]["removed"][Decision.PRESENT.value]
                    points.append({"tau_match": v, "frozen": v == th.tau_match, "row": name,
                                   **{k: o[k] for k in ("false_removal_rate", "removal_recall", "coverage",
                                                        "resolved_accuracy")},
                                   "false_present_removed": fp, "false_present_rate_removed": ratio(fp, o["n_removed"]),
                                   **{k: o[k] for k in ("n_present", "n_removed", "n_arrangements", "n_episodes")}})
                if any(not x["frozen"] for x in points):
                    out.append({"scope": scope, "lighting": light, "policy": p, "budget": b, "travel_cap_cm": None,
                                "points": points})
    return out


def diagnostic_view_coverage(specs: Sequence[EpisodeSpec], labels: GeometricLabels, start_stop: int, k: int,
                             family_of: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Geometric diagnostic-view coverage (prereg section 10.3), from the sim view labels (scoring only): per scope,
    lighting (L1, L2, pooled) and family (FAMILY_ALL, then each family when family_of is given), the occluded-present
    episodes with at least one geometrically diagnostic stop other than the start stop, and the removed episodes with
    at least k such stops, with counts."""
    def non_start(eid: str) -> int:
        return sum(1 for st, d in enumerate(labels.episodes[eid]) if d and st != start_stop)
    fams = [FAMILY_ALL, *(sorted(set(family_of.values())) if family_of else ())]
    out = []
    for scope in SCOPES:
        for light in (*REPORT_LIGHTINGS, POOLED):
            for fam in fams:
                eps = [s for s in specs if s.split is scope and (light == POOLED or s.lighting.value == light)
                       and (fam == FAMILY_ALL or family_of[s.arrangement_id] == fam)]   # type: ignore[index]
                if not eps:
                    continue
                occ = [non_start(s.episode_id) for s in eps if s.condition is Condition.OCCLUDED]
                rem = [non_start(s.episode_id) for s in eps if s.condition is Condition.REMOVED]
                xo, xr = sum(n >= 1 for n in occ), sum(n >= k for n in rem)
                out.append({"scope": scope.value, "lighting": light, "family": fam,
                            "n_arrangements": len({s.arrangement_id for s in eps}),
                            "occluded_with_diagnostic_non_start": {"x": xo, "n": len(occ), "rate": ratio(xo, len(occ))},
                            "removed_with_k_diagnostic_non_start": {"x": xr, "n": len(rem), "rate": ratio(xr, len(rem)),
                                                                    "k": k}})
    return out


def family_coverage(specs: Sequence[EpisodeSpec], labels: GeometricLabels, start_stop: int, k: int,
                    records: Sequence[ArrangementRecord]) -> list[dict[str, Any]]:
    """diagnostic_view_coverage over every family (FAMILY_ALL), each occluder family, then each slot class that is not
    an occluder family too (M16-L, M16-R); "level" says which. Without families: the FAMILY_ALL rows only."""
    occ, slot = family_of_records(records, OCCLUDER_FAMILY), family_of_records(records, SLOT_CLASS)
    rows = [{**r, "level": None if r["family"] == FAMILY_ALL else OCCLUDER_FAMILY}
            for r in diagnostic_view_coverage(specs, labels, start_stop, k, occ)]
    if slot is not None:
        have = set(occ.values())   # type: ignore[union-attr]
        rows += [{**r, "level": SLOT_CLASS} for r in diagnostic_view_coverage(specs, labels, start_stop, k, slot)
                 if r["family"] != FAMILY_ALL and r["family"] not in have]
    return rows


GATE_RATES = (("false_clear", "false-clear rate: views passing the tau_q / tau_occ gate", "non-diagnostic views of "
               "occluded-present episodes"),
              ("false_clear_miss", "false clear-miss rate", "diagnostic views of present episodes"),
              ("clear_miss", "clear-miss rate", "diagnostic views of removed episodes"),
              ("false_strong_match", "false strong-match rate (verify.select's: diagnostic views only)",
               "diagnostic views of removed episodes"))
STRONG_MATCH_ALL_REMOVED = "strong_match"   # gate_view_rates "all_removed_views": every view of a removed episode
STRONG_MATCH_ALL_TEXT = ("strong-match rate on ALL views of removed episodes (each one decides PRESENT)",
                         "every view of removed episodes, diagnostic or not")


def gate_view_rates(specs: Sequence[EpisodeSpec], store: EvidenceStore, labels: GeometricLabels, positions: Positions,
                    th: Thresholds) -> list[dict[str, Any]]:
    """The gate's per-view rates against the geometric labels (prereg section 10.4) at the run's thresholds: per scope
    and lighting (L1, L2, pooled), verify.select's ViewRates (rates_by_lighting) on the rows labelled_view_rows builds
    from the (unsealed, re-read) evidence store and the labels, the same populations and StoppingRule predicates as the
    calibration. Every view of the manifest's episodes carries a label (load_geometric_labels and main_real check).
    Beside verify.select's false_strong_match (diagnostic removed views only), "all_removed_views" counts strong matches
    on EVERY view of the removed episodes, diagnostic or not (STRONG_MATCH_ALL_REMOVED): any one decides PRESENT."""
    from verify.select import count, labelled_view_rows, rates_by_lighting   # heavy; as for --candidate
    rule = StoppingRule(th)
    diagnostic = {(e, st): d for e, ds in labels.episodes.items() for st, d in enumerate(ds)}
    rows = labelled_view_rows(specs, store, diagnostic, lambda _arr: positions)
    out = []
    for scope in SCOPES:
        ids = {s.episode_id for s in specs if s.split is scope}
        sub = [r for r in rows if r.episode_id in ids]
        if not sub:
            continue
        for light, vr in rates_by_lighting(sub, th).items():
            mine = [r for r in sub if light == POOLED or r.lighting.value == light]
            x, n = count([r for r in mine if not r.present], rule.is_strong_match)
            out.append({"scope": scope.value, "lighting": light, "n_views": len(mine), "rates": vr.to_dict(),
                        "all_removed_views": {STRONG_MATCH_ALL_REMOVED: {"x": x, "n": n, "rate": ratio(x, n)}}})
    return out


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

    The used view sets and the disagreement are computed per query lighting, like the metrics: every row has
    `lighting` "L1" or "L2", plus a "pooled" row over both lightings that is only reported in the
    secondary section (review: they were pooled over L1 and L2, with no lighting key). The reachable sets are
    geometry only and have no lighting.

    Disagreement for (A, B) is per episode: deterministic vs deterministic is 0/1; a pair with R is the
    fraction of R's seeds whose path (or decision) differs, so it is averaged within episode first.
    """
    paths: dict[tuple[str, int, float | None, str], list[tuple[tuple[int, ...], str]]] = {}
    split_of: dict[str, str] = {}
    light_of: dict[str, str] = {}
    for g in logs:
        paths.setdefault((g.policy, g.budget_additional_views, g.travel_cap_cm, g.episode_id), []).append(
            (tuple(g.requested_stops), g.decision.value))
        split_of[g.episode_id] = g.split.value
        light_of[g.episode_id] = g.lighting.value
    episodes = sorted(split_of)
    reach, used, disagree = [], [], []
    for cap in caps:
        for b in budgets:
            reach.append({"budget": b, "travel_cap_cm": cap, **reachable_view_sets(positions, b, cap)})
            for scope in ("test", "dev", "all"):
                for light in (*REPORT_LIGHTINGS, POOLED_DIAGNOSTIC):
                    eps = [e for e in episodes if (scope == "all" or split_of[e] == scope)
                           and (light == POOLED_DIAGNOSTIC or light_of[e] == light)]
                    for pol in MULTI_VIEW:
                        sets = {frozenset(pth) for e in eps for pth, _ in paths.get((pol, b, cap, e), [])}
                        if sets:
                            used.append({"scope": scope, "lighting": light, "policy": pol, "budget": b,
                                         "travel_cap_cm": cap, "n_view_sets": len(sets)})
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
                            disagree.append({"scope": scope, "lighting": light, "policy_a": a, "policy_b": bb,
                                             "budget": b, "travel_cap_cm": cap, "n_episodes": n,
                                             "path_differs": path_d, "path_disagreement_rate": path_d / n,
                                             "decision_differs": dec_d, "decision_disagreement_rate": dec_d / n})
    return {"reachable_view_sets": reach, "used_view_sets": used, "disagreement": disagree}


# ---------------------------------------------------------------- table.md

def fmt(x: float | None, nd: int = 3) -> str:
    return "undef" if x is None else f"{x:.{nd}f}"


def _find(metrics: Sequence[dict], scope: str, policy: str, budget: int, cap: float | None,
          lighting: str) -> dict | None:
    """The metrics row for one (scope, policy, budget, cap, lighting); lighting is "L1", "L2" or POOLED."""
    if policy == "S":
        budget, cap = 0, None
    elif policy == "FULL":
        budget, cap = 4, None
    for m in metrics:
        if (m["scope"], m["policy"], m["budget"], m["travel_cap_cm"], m["lighting"]) == \
                (scope, policy, budget, cap, lighting):
            return m
    return None


def _cap_label(cap: float | None, step_cm: float | None = None) -> str:
    """'no travel cap' or 'travel cap N steps (X cm)' (cm only when step_cm is unknown)."""
    if cap is None:
        return "no travel cap"
    if not step_cm:
        return f"travel cap {cap:.2f} cm"
    return f"travel cap {_steps_str(round(cap / step_cm, 6))} steps ({cap:.2f} cm)"


def _cnt(x: float | int) -> str:
    return f"{x:g}" if float(x).is_integer() else f"{x:.2f}"


def _arr_frr(o: dict) -> str:
    """FRR per arrangement (any false removal), with its counts; R's is computed per base seed and averaged over the
    seeds (analysis.metrics)."""
    x = o["arrangement_false_removal"]
    return f"{fmt(o['false_removal_rate_arrangement'])} ({'undef' if x is None else _cnt(x)}/{o['n_arr_with_present']})"


def _with_ci(o: dict, metric: str, ci: dict | None) -> str:
    """The rate, then its percentile 95% interval when the row has one (test rows, per lighting)."""
    if not ci:
        return fmt(o[metric])
    r = ci[metric]
    return f"{fmt(o[metric])} [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]"


def _cells(o: dict, ci: dict | None = None) -> list[str]:
    return [_with_ci(o, "false_removal_rate", ci), _arr_frr(o), _with_ci(o, "removal_recall", ci),
            _with_ci(o, "coverage", ci), fmt(o["resolved_accuracy"]), fmt(o["mean_views"], 2),
            fmt(o["mean_travel_cm"], 1), f"{o['n_arrangements']}/{o['n_episodes']}"]


def _row_cells(m: dict) -> list[str]:
    """_cells of a metrics row's overall numbers, with its single-rate intervals when it has them."""
    return _cells(m["overall"], m.get("rate_ci95"))


def _md(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


HEAD = ("| policy | FRR | FRR per arrangement (x/n) | recall | coverage | resolved acc | mean views | mean travel cm "
        "| n_arr/n_ep | note |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
ORDER = ("S", "F", "R", "G", "G-lookahead", "G-myopic", "Dir", "A", "A-nocue", "F-fcm", "R-fcm", "G-fcm",
         "G-lookahead-fcm", "G-myopic-fcm", "Dir-fcm", "A-fcm", "A-nocue-fcm", "FULL")   # rows that ran; others skipped
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
    grid = ", ".join(_steps_str(s) for s in cfg["cap_steps"] if s is not None)
    out = [(f"- **Travel caps are an operating condition, reported as curves, not tuned (D-cap).** "
            f"Cap = N arc steps x {step:.2f} cm (nominal step of the positions file); "
            + (f"grid {grid} steps plus uncapped" if grid else "no capped setting ran, only uncapped travel") +
            f". Affordability uses a {cfg['travel_tol_cm']:g} cm tolerance and no other slack. FULL is never capped.")]
    dev = pos.adjacent_step_deviation_cm()
    if dev > cfg["travel_tol_cm"]:
        out.append(f"- **Warning:** adjacent stops deviate from the nominal step by up to {dev:.3f} cm, so a cap "
                   "of N steps does not buy exactly N adjacent moves on this positions file.")
    # Which G is primary must be said with or without caps (review: it sat under `if caps:`, so a --cap-steps none
    # table never said that G is now the lookahead-off variant).
    out.append("- G is the primary geometric selector: per-decision normalisation (D-norm), reachability "
               "lookahead off (DECISIONS 2026-10-12). G-lookahead is the same scorer and settings with the "
               "lookahead (D-G), so G - G-lookahead isolates the lookahead. G-myopic is the legacy pre-D-G G, "
               f"kept as an extra ablation (no lookahead, lam {cfg['g_myopic']['lam_deg']:g} deg per "
               f"{cfg['g_myopic']['travel_norm_cm']:.2f} cm): G - G-myopic changes the normalisation, not the "
               "lookahead. F and R are naive by design.")
    if not caps:
        out.append("- Uncapped, the lookahead finds every candidate equally feasible, so G-lookahead takes G's path on "
                   "every episode: with no capped setting, every G - G-lookahead row is identical.")
    return out


NO_DIAG_CONTEXT = ("context for coverage, not a ceiling: a policy can resolve an episode with no diagnostic view, and "
                   "one with diagnostic views can stay unknown")


def _from_labels(o: dict) -> bool:
    """True when the no-diagnostic-view count comes from the geometric view labels, False for the fallback (the
    predeclared indistinguishable flag, no --view-labels)."""
    return o.get("no_diagnostic_view_source") == NO_DIAG_SOURCE_LABELS


def _no_diag_line(m: dict, lighting: str, scope: str) -> str:
    """Worded by its source (review: without --view-labels the line still said "episodes have no geometrically
    diagnostic stop", a geometric fact the run never measured, about the predeclared flag count)."""
    o = m["overall"]
    if _from_labels(o):
        return (f"- geometric no-diagnostic-view share ({scope}, {lighting} queries): "
                f"{fmt(o['geometric_no_diagnostic_view_share'])} ({o['n_no_diagnostic_view']} of {o['n_episodes']} "
                f"episodes have no geometrically diagnostic stop; {o['n_arrangements']} arrangements; source: "
                f"{o['no_diagnostic_view_source']}). It is {NO_DIAG_CONTEXT}.")
    return (f"- predeclared indistinguishable share (no view labels given) ({scope}, {lighting} queries): "
            f"{fmt(o['geometric_no_diagnostic_view_share'])} ({o['n_no_diagnostic_view']} of {o['n_episodes']} "
            f"episodes are occluded or removed episodes of arrangements flagged indistinguishable in the manifest; "
            f"{o['n_arrangements']} arrangements; source: {o['no_diagnostic_view_source']}). No geometry was "
            f"measured; it stands in for the geometric no-diagnostic-view share, which needs --view-labels. It is "
            f"{NO_DIAG_CONTEXT}.")


def _no_diag_column(metrics: Sequence[dict]) -> str:
    """The per-condition table's context column, named by the source of its numbers."""
    labelled = any(_from_labels(m["overall"]) for m in metrics)
    return ("geometric no-diagnostic-view share (context; view labels)" if labelled else
            "predeclared indistinguishable share (context; no view labels given)")


CURVE_HEAD = ("| policy | travel cap | FRR | FRR per arrangement (x/n) | recall | coverage | resolved acc | mean views "
              "| mean travel cm | n_arr/n_ep |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def table_scope(metrics: Sequence[dict]) -> str:
    """The split table.md scores: test, or dev when the run has no test episode (a dev-only sim manifest)."""
    return Split.TEST.value if any(m["scope"] == Split.TEST.value for m in metrics) else Split.DEV.value


def _lighting_lines(results: dict[str, Any], light: str, lab) -> list[str]:
    """Primary tables, budget curves, per-condition tables, cap curve and bootstrap for one query lighting."""
    cfg, metrics = results["config"], results["metrics"]
    pb, scope, caps = cfg["primary_budget"], table_scope(metrics), cfg["travel_caps_cm"]
    h = f"## {light} queries -"
    lines: list[str] = []
    prim = [r for r in results.get("primary", []) if r["lighting"] == light]
    if prim:
        lines += [f"{h} PRIMARY CONTRAST: {' - '.join(PRIMARY_PAIR)} at b = {pb}, uncapped ({scope} split)", "",
                  ("The preregistered primary comparison (SIM_PREREGISTRATION section 1), paired arrangement-level "
                   "cluster bootstrap. A contrast whose 95% interval contains 0 is INCONCLUSIVE: it is never reported "
                   "as equivalence, and no equivalence margin is declared. Every other row below is secondary."), "",
                  "| metric | estimate | 95% CI | n_arr | n_ep | reading |", "| --- | --- | --- | --- | --- | --- |"]
        for r in prim:
            lines.append(f"| {r['metric']} | {fmt(r['estimate'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | "
                         f"{r['n_arrangements']} | {r['n_episodes']} | {_reading(r)} |")
        lines.append("")
    for cap in caps:
        lines += [f"{h} Primary table: b = {pb} additional views, {lab(cap)}", "", HEAD]
        for p in ORDER:
            m = _find(metrics, scope, p, pb, cap, light)
            if m:
                lines.append(_md([_label(p, cap), *_row_cells(m), _note(p, cap)]))
        lines.append("")
    for cap in caps:
        lines += [f"{h} Budget curve, {lab(cap)}", "",
                  "| policy | b | FRR | FRR per arrangement (x/n) | recall | coverage | resolved acc | mean views "
                  "| mean travel cm | n_arr/n_ep |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in ORDER:
            if p in ("S", "FULL"):
                continue
            for b in cfg["budgets"]:
                m = _find(metrics, scope, p, b, cap, light)
                if m:
                    lines.append(_md([p, str(b), *_row_cells(m)]))
        lines.append("")
    for cap in caps:
        lines += [f"{h} Per condition, b = {pb}, {lab(cap)}", "",
                  "Decision fractions are averaged within episode (R) and summed over episodes.", "",
                  ("| policy | condition | n_arr/n_ep | present | removed | unknown | coverage | "
                   f"{_no_diag_column(metrics)} | mean views | mean travel cm |"),
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in ORDER:
            m = _find(metrics, scope, p, pb, cap, light)
            if not m:
                continue
            for cond, o in m["by_condition"].items():
                t = "removed" if cond == "removed" else "present"
                c = o["confusion"][t]
                lines.append(f"| {_label(p, cap)} | {cond} | {o['n_arrangements']}/{o['n_episodes']} | "
                             f"{c['present']:.2f} | {c['removed']:.2f} | {c['unknown']:.2f} | {fmt(o['coverage'])} | "
                             f"{fmt(o['geometric_no_diagnostic_view_share'])} | "
                             f"{fmt(o['mean_views'], 2)} | {fmt(o['mean_travel_cm'], 1)} |")
        lines.append("")
    lines += [f"{h} Cap curve, b = {pb} ({scope} split)", "",
              "Each policy across the cap grid at the primary budget. Caps are an operating condition, not tuned.", "",
              CURVE_HEAD]
    for p in ORDER:
        if p in ("S", "FULL"):
            continue
        for cap in caps:
            m = _find(metrics, scope, p, pb, cap, light)
            if m:
                lines.append(_md([p, lab(cap).replace("travel cap ", ""), *_row_cells(m)]))
    sens = cfg.get("sensitivity") or {}
    sens_pols = sorted({m["policy"] for m in metrics if _is_sensitivity(m["policy"])})
    if sens_pols:
        lines += ["", f"{h} Sensitivity runs, b = {pb} ({scope} split; reported only, never selected)", "",
                  (f"k = {', '.join(map(str, sens.get('k', [])))} with lam fixed (every multi-view policy, '-k<k>'); "
                   f"lam = {', '.join(f'{x:g}' for x in sens.get('lam', []))} with k fixed (G and G-lookahead, "
                   "'-lam<lam>'). The primary setting is k = 2, lam = 0."), "", CURVE_HEAD]
        for p in sens_pols:
            for cap in caps:
                m = _find(metrics, scope, p, pb, cap, light)
                if m:
                    lines.append(_md([p, lab(cap).replace("travel cap ", ""), *_row_cells(m)]))
    strat = _stratification(results)
    lines += ["", f"{h} Paired arrangement-level bootstrap, b = {pb} (test split)", ""]
    if not strat.get("ran", True):
        lines += [f"**No bootstrap ran:** {NO_BOOTSTRAP_REASON}.", ""]
    else:
        how = ("resampled within quota slot strata, each stratum keeping its count" if strat["stratified"]
               else "**UNSTRATIFIED**: " + strat["note"])
        lines += [(f"Difference A - B on {light} queries only; percentile 95% CI over resampled arrangements (an "
                   f"arrangement's {light} episodes are kept together; {how})."),
                  ("A row marked *identical* means A and B made the same decisions with the same effort on every "
                   "episode: its [0, 0] CI is degenerate and is not evidence of equal performance."),
                  "", "| setting | A - B | metric | estimate | 95% CI | n_arr | n_ep | B | undefined draws | note |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in results["bootstrap"]:
            if r["lighting"] != light:
                continue
            notes = (["identical: A == B on every episode"] if r.get("identical_runs") else []) + \
                (["unstratified: singleton strata " + ", ".join(r["singleton_strata"])]
                 if r.get("singleton_strata") else [])
            lines.append(f"| {lab(r['travel_cap_cm'])} | {r['policy_a']} - {r['policy_b']} | {r['metric']} | "
                         f"{fmt(r['estimate'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | {r['n_arrangements']} | "
                         f"{r['n_episodes']} | {r['B']} | {r['n_undefined']} | {'; '.join(notes)} |")
        lines.append("")
    ct = [r for r in results.get("common_target", []) if r["lighting"] == light]
    if ct:
        lines += [f"{h} Secondary (predeclared): {' - '.join(PRIMARY_PAIR)} on common targets, b = {pb}, uncapped", "",
                  (f"Episodes of {', '.join(ct[0]['targets'])} only; within each occluder family these targets carry "
                   "equal weight, recomputed on every bootstrap replicate (analysis/common_target.py). Secondary: it "
                   "does not replace the primary contrast."), "",
                  "| metric | estimate | 95% CI | n_arr | n_ep | single-target families | reading |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in ct:
            lines.append(f"| {r['metric']} | {fmt(r['estimate'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | "
                         f"{r['n_arrangements']} | {r['n_episodes']} | "
                         f"{', '.join(r['single_target_families']) or '-'} | "
                         f"{_reading(r)} |")
        lines.append("")
    diag = results.get("diagnostics")
    if diag:
        lines += _diagnostic_lines(diag, cfg, lab, light, f"{h} Diagnostic:", scope)
    return lines


SENSITIVITY_SUFFIX = ("-k", "-lam")   # run_stage's log names for the declared sensitivity runs


def _is_sensitivity(policy: str) -> bool:
    return any(s in policy and policy.rsplit(s, 1)[1][:1].isdigit() for s in SENSITIVITY_SUFFIX)


def _reading(r: Mapping[str, Any]) -> str:
    """The declared reading of one contrast row: inconclusive when its 95% interval contains 0, never equivalence."""
    if r.get("ci_lo") is None or r.get("estimate") is None:
        return "undefined"
    if r["ci_lo"] <= 0.0 <= r["ci_hi"]:
        return "inconclusive (the interval contains 0; not equivalence)"
    return ("first policy higher" if r["estimate"] > 0 else "first policy lower") + " (the interval excludes 0)"


def _pooled_lines(results: dict[str, Any], lab) -> list[str]:
    """The only pooled-over-lighting numbers in table.md: one labelled secondary section at the end, with the
    pooled policy table and the pooled diagnostics."""
    cfg, metrics = results["config"], results["metrics"]
    pb, scope, caps = cfg["primary_budget"], table_scope(metrics), cfg["travel_caps_cm"]
    lines = [f"## Secondary, not a headline: pooled over L1 and L2 queries ({scope} split)", "",
             ("Both lightings together, for reference only. The results are the per-lighting sections above; "
              "no bootstrap is run on pooled data."), "", f"### Policies at b = {pb}, pooled", "", CURVE_HEAD]
    for p in ORDER:
        for cap in ((None,) if p in ("S", "FULL") else caps):
            m = _find(metrics, scope, p, pb, cap, POOLED)
            if m:
                lines.append(_md([p, lab(cap).replace("travel cap ", ""), *_row_cells(m)]))
    lines.append("")
    diag = results.get("diagnostics")
    if diag:
        lines += _diagnostic_lines(diag, cfg, lab, POOLED_DIAGNOSTIC, "### Diagnostic, pooled:", scope)
    return lines


def _table_intro(results: dict[str, Any], scope: str) -> list[str]:
    """The label line, the data paragraph and the data line of table.md: SYNTHETIC, SIMULATION or REAL PILOT."""
    cfg = results["config"]
    if results.get("synthetic"):
        what = [("**SYNTHETIC - not study results.** Evidence comes from `verify.synthetic` with invented "
                 "parameters and placeholder thresholds; these numbers only exercise the pipeline.")]
        data = (f"- {cfg['n_arr']} arrangements, seed {cfg['seed']}; scored on the **test** split only "
                "(dev metrics are in results.json, scope \"dev\")")
    else:
        frozen = cfg.get("thresholds_source") == "frozen candidate"
        th_text = (f"The thresholds are the frozen select-v4 candidate's (`{cfg.get('candidate_file')}`, sha256 "
                   f"`{cfg.get('candidate_sha256')}`)." if frozen
                   else "The thresholds are the placeholders, not a frozen select-v4 candidate.")
        if cfg.get("simulation"):
            what = [("**SIMULATION.** Evidence from `perception/evidence.py` on Isaac Sim renders (the manifest says "
                     "`\"simulation\": true`); every number here is a simulation result. " + th_text)]
        else:
            what = [("**REAL PILOT.** Evidence from `perception/evidence.py` on real photos; not study results. "
                     + th_text)]
        other = "dev" if scope == Split.TEST.value else "test"
        data = (f"- environment **{cfg.get('environment', 'shelf')}**; "
                f"manifest `{cfg['manifest_file']}` (sha256 `{cfg['manifest_sha256']}`), evidence "
                f"`{cfg['evidence_file']}` (sha256 `{cfg['evidence_sha256']}`); scored on the **{scope}** split"
                + (" (the manifest has no test arrangement)" if scope != Split.TEST.value else "")
                + f" ({other} metrics, if any, are in results.json)")
    return [f"# {results['label']}", "", *what, "", f"- config `{cfg['config_version']}`, sha256 "
            f"`{results['config_sha256']}`", data]


def _stratification(results: dict[str, Any]) -> dict[str, Any]:
    return results.get("bootstrap_stratification") or {"ran": True, "stratified": False, "strata": {},
                                                        "singleton_strata": [], "note": UNSTRATIFIED_NOTE}


def _strat_line(results: dict[str, Any]) -> str:
    st = _stratification(results)
    parts = ", ".join(f"{k} {v}" for k, v in st["strata"].items())
    if not st.get("ran", True):
        return f"- **No bootstrap ran** (paired or single-rate): {NO_BOOTSTRAP_REASON}."
    if st["stratified"]:
        return f"- **Paired bootstrap stratified by quota slot class** ({parts} test arrangements): {st['note']}."
    if st.get("singleton_strata"):
        return f"- **Paired bootstrap UNSTRATIFIED** ({parts} test arrangements): {st['note']}."
    return f"- **Paired bootstrap UNSTRATIFIED**: {st['note']}."


def _rate_ci_line(results: dict[str, Any]) -> str | None:
    """The header note on the single-rate intervals in the tables, or None when no row has one."""
    if not any(m.get("rate_ci95") for m in results["metrics"]):
        return None
    st = _stratification(results)
    how = "stratified as the paired bootstrap" if st["stratified"] else "unstratified, as the paired bootstrap"
    return ("- **Single-rate intervals.** On the test split each L1 / L2 row gives FRR, recall and coverage as "
            "`rate [2.5%, 97.5%]`: the percentile 95% interval of the same arrangement-level cluster bootstrap as "
            f"the paired rows ({how}; B = {results['config']['bootstrap_B']}). R's interval is over its "
            "seed-averaged decisions: arrangements are resampled, seeds are not (results.json `rate_ci95`).")


FAMILY_HEAD = ("| policy | FRR | FRR per arrangement (x/n) | recall | coverage | resolved acc | mean views "
               "| mean travel cm | n_arr/n_ep |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
CONTRAST_HEAD = "| metric | estimate | 95% CI | n_arr | n_ep | reading |\n| --- | --- | --- | --- | --- | --- |"


def _xn(d: Mapping[str, Any]) -> str:
    return f"{_cnt(d['x'])}/{d['n']}"


def _gate_off_lines(results: dict[str, Any], light: str, lab, h: str) -> list[str]:
    """The gate-off rows next to F and R across the caps, when the ablation ran."""
    cfg, metrics = results["config"], results["metrics"]
    pb, scope = cfg["primary_budget"], table_scope(metrics)
    if not any(m["policy"].endswith(f"-{GATE_OFF_SUFFIX}") for m in metrics):
        return []
    lines = [f"{h} Occlusion-gate-off ablation, b = {pb} ({scope} split; F and R only; reported only, never selected)",
             "", (f"tau_occ = {GATE_OFF_TAU_OCC:.1f}: no view is flagged occluded, so a clear miss needs only "
                  "quality >= tau_q and a match below tau_match. G is not run this way: tau_occ is also G's "
                  f"occlusion flag. The paired F - F-{GATE_OFF_SUFFIX} and R - R-{GATE_OFF_SUFFIX} contrasts (test, "
                  f"b = {pb}, uncapped) are in the bootstrap table."), "", CURVE_HEAD]
    for p in (x for g in GATE_OFF_POLICIES for x in (g, f"{g}-{GATE_OFF_SUFFIX}")):
        for cap in cfg["travel_caps_cm"]:
            m = _find(metrics, scope, p, pb, cap, light)
            if m:
                lines.append(_md([p, lab(cap).replace("travel cap ", ""), *_row_cells(m)]))
    return [*lines, ""]


def _sweep_lines(results: dict[str, Any], light: str, h: str) -> list[str]:
    """The tau_match risk-coverage curves of one lighting, frozen point marked."""
    sw = results.get("tau_match_sweep")
    scope = table_scope(results["metrics"])
    curves = [c for c in (sw or {}).get("curves", []) if c["scope"] == scope and c["lighting"] == light]
    if not curves:
        return []
    lines = [f"{h} Risk-coverage sweep of tau_match, b = {results['config']['primary_budget']}, uncapped ({scope} "
             "split; reported only, never selected)", "",
             (f"The frozen candidate's predeclared grid {', '.join(f'{v:g}' for v in sw['grid'])}, extended above "
              f"the frozen point by {', '.join(f'{v:g}' for v in sw.get('extension', []))} (declared 2026-11-06, "
              "before the freeze), every other threshold frozen. The frozen point (marked) is the policy's own "
              "primary row; S is at its actual effort and FULL at all stops. False-present rate = removed episodes "
              "decided present (x/n removed)."),
             "", "| policy | tau_match | FRR | recall | coverage | resolved acc | false-present rate among removed "
             "(x/n) | n_arr/n_ep |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in curves:
        for x in c["points"]:
            tm = f"**{x['tau_match']:g} (frozen)**" if x["frozen"] else f"{x['tau_match']:g}"
            lines.append(_md([c["policy"], tm, fmt(x["false_removal_rate"]), fmt(x["removal_recall"]),
                              fmt(x["coverage"]), fmt(x["resolved_accuracy"]),
                              f"{fmt(x['false_present_rate_removed'])} ({_cnt(x['false_present_removed'])}/"
                              f"{x['n_removed']})", f"{x['n_arrangements']}/{x['n_episodes']}"]))
    return [*lines, ""]


def _arrangements(n: int, scope: str) -> str:
    return f"{n} {scope} arrangement{'' if n == 1 else 's'}"


def _family_lines(results: dict[str, Any], light: str, h: str, key: str, title: str,
                  only: Sequence[str] | None = None) -> list[str]:
    """One family level (results[key]): per family the uncapped primary-budget rows of FAMILY_TABLE_POLICIES, the
    target composition and A - G; only: the families to show (None: all)."""
    fam = results.get(key)
    if not fam:
        return []
    cfg = results["config"]
    pb, scope = cfg["primary_budget"], table_scope(results["metrics"])
    head = f"{h} {title}, b = {pb}, uncapped ({scope} split; descriptive, H3)"
    if not fam["ran"]:
        return [head, "", f"Not computed: {fam['note']}.", ""]
    shown = [f for f in fam["families"] if only is None or f in only]
    if not shown:
        return []
    idx = {(m["scope"], m["lighting"], m["family"], m["policy"], m["budget"], m["travel_cap_cm"]): m
           for m in fam["metrics"]}
    comp = {(c["scope"], c["family"]): c for c in fam["composition"]}
    lines = [head, "", f"{fam['note']}. Every cap is in results.json (`{key}`).", ""]
    if scope == Split.TEST.value:
        lines += [f"A - G per family: {fam.get('bootstrap_note', '')}.", ""]
    for f in shown:
        c = comp.get((scope, f))
        if c is None:
            continue
        lines += [f"**{f}** ({_arrangements(c['n_arrangements'], scope)}; targets: "
                  + ", ".join(f"{t} {n}" for t, n in c["targets"].items()) + ")", "", FAMILY_HEAD]
        for p in FAMILY_TABLE_POLICIES:
            b = 4 if p == "FULL" else pb
            m = idx.get((scope, light, f, p, b, None))
            if m:
                lines.append(_md([p, *_cells(m["overall"])]))
        lines.append("")
        if scope != Split.TEST.value:
            continue
        rows = [r for r in fam["bootstrap"] if r["lighting"] == light and r["family"] == f]
        skip = [r for r in fam["bootstrap_skipped"] if r["lighting"] == light and r["family"] == f]
        if rows:
            how = ("stratified by slot class" if rows[0].get("stratified") else
                   "unstratified: " + rows[0]["stratification_note"] if rows[0].get("singleton_strata") else
                   "unstratified within the family")
            lines += [f"{' - '.join(PRIMARY_PAIR)} on {f}, {light} queries (B = {rows[0]['B']}, {how}):", "",
                      CONTRAST_HEAD]
            for r in rows:
                note = " (identical runs)" if r.get("identical_runs") else ""
                lines.append(f"| {r['metric']} | {fmt(r['estimate'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | "
                             f"{r['n_arrangements']} | {r['n_episodes']} | {_reading(r)}{note} |")
            lines.append("")
        for r in skip:
            lines += [f"{' - '.join(PRIMARY_PAIR)} on {f}, {light} queries: not bootstrapped "
                      f"({_arrangements(r['n_arrangements'], 'test')}): {r['reason']}.", ""]
    return lines


def _families_lines(results: dict[str, Any], light: str, h: str) -> list[str]:
    """The five occluder families first, then the slot classes that are not an occluder family (M16-L, M16-R)."""
    occ = results.get(FAMILY_LEVELS[OCCLUDER_FAMILY]) or {}
    lines = _family_lines(results, light, h, FAMILY_LEVELS[OCCLUDER_FAMILY], "By occluder family")
    split = None if not occ.get("ran") else [f for f in (results.get(FAMILY_LEVELS[SLOT_CLASS]) or {}).get(
        "families", []) if f not in occ["families"]]
    if occ.get("ran") or not occ:   # a family-less manifest says "Not computed" once, under the occluder families
        lines += _family_lines(results, light, h, FAMILY_LEVELS[SLOT_CLASS],
                               "By slot class (the occluder families split by M16 side)", split)
    return lines


def _view_diagnostic_lines(results: dict[str, Any], light: str, h: str) -> list[str]:
    """Geometric diagnostic-view coverage (prereg 10.3) and the gate's per-view rates (10.4) for one lighting."""
    cov, rates = results.get("diagnostic_view_coverage"), results.get("gate_view_rates")
    if not cov or not rates:
        return []
    scope = table_scope(results["metrics"])
    start = results["positions"]["start_stop"]
    head1 = f"{h} Geometric diagnostic-view coverage ({scope} split)"
    head2 = f"{h} Gate per-view rates against the geometric labels ({scope} split)"
    if not cov["ran"]:
        return [head1, "", f"Not computed: {cov['note']}.", "", head2, "", f"Not computed: {rates['note']}.", ""]
    k = cov["k"]
    lines = [head1, "",
             (f"Occluded (present) episodes with at least one geometrically diagnostic stop other than the start stop "
              f"{start}, and removed episodes with at least k = {k} such stops, from the view labels (geometric "
              "labels, not recognisability; context, not a ceiling)."), "",
             f"| family | n_arr | occluded: >= 1 diagnostic non-start stop (x/n) | share | removed: >= {k} diagnostic "
             "non-start stops (x/n) | share |", "| --- | --- | --- | --- | --- | --- |"]
    for r in cov["rows"]:
        if r["scope"] == scope and r["lighting"] == light:
            o, m = r["occluded_with_diagnostic_non_start"], r["removed_with_k_diagnostic_non_start"]
            lines.append(_md([r["family"], str(r["n_arrangements"]), _xn(o), fmt(o["rate"]), _xn(m), fmt(m["rate"])]))
    th = rates["thresholds"]
    lines += ["", head2, "",
              (f"At the run's thresholds (tau_match {th['tau_match']:g}, tau_q {th['tau_q']:g}, tau_occ "
               f"{th['tau_occ']:g}), with the calibration's view populations and StoppingRule predicates "
               "(verify.select ViewRates). Per view: the views of one arrangement are correlated."), "",
              "| rate | views | x/n | rate |", "| --- | --- | --- | --- |"]
    for r in rates["rows"]:
        if r["scope"] == scope and r["lighting"] == light:
            for name, what, pop in GATE_RATES:
                d = r["rates"][name]
                lines.append(_md([what, pop, _xn(d), fmt(d["rate"])]))
            d = (r.get("all_removed_views") or {}).get(STRONG_MATCH_ALL_REMOVED)
            if d:
                lines.append(_md([*STRONG_MATCH_ALL_TEXT, _xn(d), fmt(d["rate"])]))
    return [*lines, ""]


def _test_analysis_lines(results: dict[str, Any], light: str, lab) -> list[str]:
    """The real-v6 sections of one lighting: gate-off ablation, tau_match sweep, by family, view diagnostics."""
    h = f"## {light} queries -"
    return [*_gate_off_lines(results, light, lab, h), *_sweep_lines(results, light, h),
            *_families_lines(results, light, h), *_view_diagnostic_lines(results, light, h)]


def render_table(results: dict[str, Any]) -> str:
    cfg, metrics = results["config"], results["metrics"]
    pb, scope = cfg["primary_budget"], table_scope(metrics)
    step = cfg.get("nominal_step_cm")

    def lab(cap: float | None) -> str:
        return _cap_label(cap, step)
    lights = [x for x in REPORT_LIGHTINGS if any(m["lighting"] == x for m in metrics)]
    lines = [*_table_intro(results, scope),
             (f"- thresholds: tau_match {cfg['thresholds']['tau_match']}, tau_q {cfg['thresholds']['tau_q']}, "
              f"tau_occ {cfg['thresholds']['tau_occ']}, k {cfg['thresholds']['k']}, "
              f"lam {cfg['thresholds']['lam']}; R averaged over {len(cfg['r_seeds'])} seeds within each episode"),
             ("- FRR = false removal rate per episode; FRR per arrangement = share of arrangements with a present "
              "episode that have any false removal (for R, computed per base seed and averaged over the seeds); "
              "`-fcm` = first-clear-miss rule (k = 1); S is plotted at its actual effort; FULL is a higher-effort "
              "reference, not a budget-matched competitor"),
             ("- **Reported separately for L1 and L2 queries** (the lighting of the query view; DECISIONS "
              "2026-11-02): each lighting has its own primary tables, budget curves, per-condition tables, cap "
              "curve and paired bootstrap. A number pooled over lighting appears only in the secondary section "
              "at the end."),
             *([_generator_caveat(cfg.get("generator", {}))] if results.get("synthetic") else []),
             *_cap_notes(cfg, Positions.from_dict(results["positions"])), _strat_line(results),
             *(x for x in (_rate_ci_line(results),) if x)]
    for light in lights:
        ref = _find(metrics, scope, "S", pb, None, light)
        if ref:
            lines.append(_no_diag_line(ref, light, scope))
    lines.append("")
    for light in lights:
        lines += _lighting_lines(results, light, lab)
        lines += _test_analysis_lines(results, light, lab)
    lines += _pooled_lines(results, lab)
    lines += [f"_{results['label']}._", ""]
    return "\n".join(lines)


USED_COLS = ("F", "G", "G-lookahead", "G-myopic", "R", "Dir", "A", "A-nocue")


def _diagnostic_lines(diag: dict, cfg: dict, lab, light: str, head: str, scope: str = "test") -> list[str]:
    """The two 'can selection matter?' tables for one query lighting ("L1", "L2", or POOLED_DIAGNOSTIC in the
    secondary section); head starts each heading."""
    caps, budgets = cfg["travel_caps_cm"], cfg["budgets"]
    # columns only for the policies and pairs that ran (A and Dir need --model / --side), the same in every table
    ran = {u["policy"] for u in diag["used_view_sets"]}
    used_cols = [p for p in USED_COLS if p in ran or p not in OPTIONAL_POLICIES]
    pairs_ran = {(d["policy_a"], d["policy_b"]) for d in diag["disagreement"]}
    disagree_pairs = [p for p in DISAGREE_PAIRS if p in pairs_ran or not set(p) & set(OPTIONAL_POLICIES)]
    eps = f"{scope} episodes of both lightings" if light == POOLED_DIAGNOSTIC else f"{scope} episodes with {light} queries"
    used = {(u["policy"], u["budget"], u["travel_cap_cm"]): u["n_view_sets"]
            for u in diag["used_view_sets"] if u["scope"] == scope and u["lighting"] == light}
    lines = [f"{head} distinct view sets per budget and cap ({scope} split)", "",
             ("Can selection matter at all? *Reachable* is geometry only (the same for every lighting): the most "
              "views reachable from the start within b moves and the cap, and how many distinct view sets of that "
              f"size exist. *Used* counts the distinct view sets each policy actually ended with on the {eps} "
              "(early stopping included). This table describes paths only."),
             "", "| travel cap | b | max views | reachable sets at max | "
             + " | ".join(f"used: {p}" for p in used_cols) + " |",
             "| --- | --- | --- | --- | " + " | ".join("---" for _ in used_cols) + " |"]
    for r in diag["reachable_view_sets"]:
        cap, b = r["travel_cap_cm"], r["budget"]
        lines.append(_md([lab(cap).replace("travel cap ", ""), str(b), str(r["max_views"]), str(r["n_max_view_sets"]),
                          *(str(used.get((p, b, cap), "-")) for p in used_cols)]))
    # Review: "the diagnostics describe paths, not outcomes" was wrong for the decision column, which compares
    # outcomes (without truth: it says how often two policies decide differently, not which one is right).
    lines += ["", f"{head} path and decision disagreement ({scope} split, k-rule policies)", "",
              (f"Fraction of {eps} where A's requested path (or final decision) differs from B's. "
               "Pairs with R average over R's seeds within each episode. Path disagreement compares where two "
               "policies looked; decision disagreement compares what they decided, an outcome (read without truth, "
               "so it does not say which decision was right). 0 means the two policies are indistinguishable at "
               "that setting, so any paired difference there is degenerate. Counts are in results.json."), "",
              "| travel cap | b | " + " | ".join(f"{a} vs {b} path / decision" for a, b in disagree_pairs) + " |",
              "| --- | --- | " + " | ".join("---" for _ in disagree_pairs) + " |"]
    dis = {(d["policy_a"], d["policy_b"], d["budget"], d["travel_cap_cm"]): d
           for d in diag["disagreement"] if d["scope"] == scope and d["lighting"] == light}
    for cap in caps:
        for b in budgets:
            if b == 0:
                continue
            cells = []
            for a, bb in disagree_pairs:
                d = dis.get((a, bb, b, cap))
                cells.append("-" if d is None else
                             f"{d['path_disagreement_rate']:.2f} / {d['decision_disagreement_rate']:.2f}")
            lines.append(_md([lab(cap).replace("travel cap ", ""), str(b), *cells]))
    lines.append("")
    return lines


# ---------------------------------------------------------------- io and main

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=1, sort_keys=True) + "\n")


def write_logs(path: Path, logs: Sequence[EpisodeLog], *, simulation_label: str | None = None) -> None:
    """One log per line. simulation_label: a simulation run's label, added to every line with "simulation": true
    (EpisodeLog.from_dict ignores both keys)."""
    with path.open("w") as f:
        for g in logs:
            d = g.to_dict()
            if simulation_label:
                d.update(label=simulation_label, simulation=True)
            f.write(json.dumps(to_jsonable(d), sort_keys=True, separators=(",", ":")) + "\n")


def save_truth(path: Path, truth: GroundTruth, *, simulation_label: str | None = None) -> None:
    """GroundTruth.save, plus the label and "simulation": true for a simulation run (GroundTruth.load ignores them)."""
    d = truth.to_dict()
    if simulation_label:
        d.update(label=simulation_label, simulation=True)
    path.write_text(json.dumps(d, indent=2) + "\n")


# Everything a run writes into its output directory, removed before it starts (review: a run that failed after the
# policies ran, e.g. on --view-labels, left the previous run's results.json beside its own config.json).
RUN_OUTPUTS = ("config.json", "logs.jsonl", "episodes.json", "evidence.jsonl", "ground_truth.json", "results.json",
               "table.md")
REAL_OUTPUTS = ("config.json", "logs.jsonl", "ground_truth.json", "results.json", "table.md")


def clear_outputs(out: Path, names: Sequence[str], inputs: Sequence[str | None] = ()) -> None:
    """Remove the run's own output files from out. ValueError, before anything is removed, if an input file is one
    of them (an evidence file named evidence.jsonl in the output dir must never be deleted)."""
    targets = {(out / n).resolve() for n in names}
    clash = [p for p in inputs if p is not None and Path(p).resolve() in targets]
    if clash:
        raise ValueError(f"input file {clash[0]} is an output of this run in {out}; use another --out")
    for n in names:
        (out / n).unlink(missing_ok=True)


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
    given = [f for f, v in (("--candidate", args.candidate), ("--placeholder-thresholds", args.placeholder_thresholds),
                            ("--side", args.side), ("--model", args.model),
                            ("--gate-off-ablation", args.gate_off_ablation),
                            ("--tau-match-sweep", args.tau_match_sweep)) if v]
    if given:   # the synthetic path never reads them: refuse instead of silently ignoring them
        print(f"{', '.join(given)}: real/sim data only, not --synthetic", file=sys.stderr)
        return 2
    if args.primary_budget not in args.budgets:
        print(f"--primary-budget {args.primary_budget} not in --budgets {args.budgets}", file=sys.stderr)
        return 2
    tag = args.tag or f"n{args.n_arr}-s{args.seed}"
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"synthetic-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    try:   # every output, so no truth-bearing file exists while policies run and no stale result survives a failure
        clear_outputs(out, RUN_OUTPUTS, (args.positions,))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    args.positions = args.positions or str(DEFAULT_POSITIONS_PATH)
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
    results = {"label": SYNTHETIC_LABEL, "synthetic": True, "simulation": False, "config": config,
               "config_sha256": sha,
               "positions": positions.to_dict(), "thresholds": th.to_dict(), "generator": config["generator"],
               "files": {n: file_sha256(out / n) for n in ("episodes.json", "ground_truth.json",
                                                           "evidence.jsonl", "logs.jsonl")},
               "n_logs": len(logs), **scored, "diagnostics": diagnostics}
    write_json(out / "results.json", results)
    (out / "table.md").write_text(render_table(results))
    print(f"{SYNTHETIC_LABEL}: wrote {out}")
    return 0


REAL_CONFIG_VERSION = "real-v6"
# v6 (2026-11-05): multi-view policies vs S, by-family breakdowns, gate-off ablation, tau_match sweep, view diagnostics
#   (review 2026-11-06, before any v6 test run: occluder-family level, sweep extension, per-family seeds; code_sha256)
ANALYSIS_CODE = ("verify/*.py", "analysis/*.py")   # every module this run can import from the repository


def code_sha256() -> dict[str, str]:
    """{repo-relative path: sha256} of ANALYSIS_CODE as this run loads it (tools/test_validity.py compares it with the
    freeze record)."""
    return {p.relative_to(REPO_ROOT).as_posix(): file_sha256(p)
            for pattern in ANALYSIS_CODE for p in sorted(REPO_ROOT.glob(pattern))}
# v5 (2026-11-05): frozen candidate thresholds (--candidate), side descriptors, Dir and frozen rankers (--side,
# --model), the primary contrast and the common-target secondary analysis
# v4 (2026-11-03): bootstrap_stratified is what the test bootstrap uses (singleton strata fall back); single-rate CIs;
# SIMULATION-labelled logs and truth
# v3 (2026-11-03): simulation flag, view labels file and bootstrap strata hashed
# v2 (2026-11-02): cap_steps hashed as floats; diagnostics per lighting
# v1 (2026-11-02): primary G = lookahead off, G-lookahead added; per-lighting results
REAL_LABEL = "REAL PILOT - placeholder thresholds, not frozen"
SIM_LABEL = "SIMULATION - placeholder thresholds, not frozen"
REAL_FROZEN_LABEL = "REAL PILOT - frozen candidate thresholds"
SIM_FROZEN_LABEL = "SIMULATION - frozen candidate thresholds"


def main_real(args: argparse.Namespace) -> int:
    """Run S/FULL/F/R/G/G-lookahead/G-myopic, with --side also Dir and with --model the frozen rankers, on real or
    simulated evidence: same run and score stages as --synthetic.
    A manifest with "simulation": true is labelled SIMULATION (config.json, results.json, table.md); the bootstrap
    is stratified by the manifest's quota slot classes when every arrangement has one. After the run it adds the
    declared test analyses (real-v6): by_occluder_family, by_family, tau_match_sweep (with --tau-match-sweep),
    diagnostic_view_coverage and gate_view_rates (with --view-labels; otherwise "ran": false and a note)."""
    if args.primary_budget not in args.budgets or args.r_seeds < 1 or args.bootstrap_B < 1:
        print("need --primary-budget in --budgets, --r-seeds >= 1, --bootstrap-B >= 1", file=sys.stderr)
        return 2
    try:
        info = load_manifest_info(args.manifest)
        specs, truth = episodes_from_arrangements(info.arrangements)
        del truth  # the run stage never sees it; re-read after the run
    except (OSError, ValueError, KeyError) as e:
        print(f"bad real-data input: {e}", file=sys.stderr)
        return 2
    if len(info.environments) != 1:   # split by environment (DECISIONS 2026-11-04, choice 4): one per run
        print(f"{args.manifest}: one environment per run, this manifest has {list(info.environments)}; run each "
              "environment's manifest with its own positions file", file=sys.stderr)
        return 2
    if info.simulation and args.positions is None:
        # configs/positions.json is the old close-spot arc: travel and caps would be wrong for a sim set
        print("a simulation manifest needs --positions (e.g. configs/sim/positions_shelf_A.json)", file=sys.stderr)
        return 2
    pos_path = args.positions or str(DEFAULT_POSITIONS_PATH)
    label = SIM_LABEL if info.simulation else REAL_LABEL
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"real-{args.tag or Path(args.evidence).stem}"
    out.mkdir(parents=True, exist_ok=True)
    try:   # every output first: no truth while policies run, no earlier results after a failure (e.g. --view-labels)
        clear_outputs(out, REAL_OUTPUTS, (args.manifest, args.evidence, args.view_labels, pos_path))
    except ValueError as e:
        print(f"bad real-data input: {e}", file=sys.stderr)
        return 2
    try:
        positions = Positions.load(pos_path)
        store = EvidenceStore.from_jsonl(args.evidence, positions.n_stops)
        if store.synthetic:
            raise ValueError(f"{args.evidence} is synthetic evidence; use --synthetic")
        for s in specs:
            store.replay_key(s.episode_id)   # ReplayError unless all stops have evidence
    except (OSError, ValueError, KeyError) as e:
        print(f"bad real-data input: {e}", file=sys.stderr)
        return 2
    strata = info.strata()
    # the stratification the test bootstrap will use, from the manifest's splits and families (no truth): singleton
    # strata fall back to unstratified, and with no test arrangement no bootstrap runs
    planned = stratification_info(strata, [s.arrangement_id for s in specs if s.split is Split.TEST])
    has_test = any(s.split is Split.TEST for s in specs)
    if info.simulation and has_test and args.candidate is None and not args.placeholder_thresholds:
        print("a simulation manifest with test arrangements needs --candidate: test is scored only with frozen "
              "thresholds, never with the placeholders (--placeholder-thresholds: pipeline tests only)",
              file=sys.stderr)
        return 2
    if args.candidate is not None and args.placeholder_thresholds:
        print("--candidate and --placeholder-thresholds exclude each other", file=sys.stderr)
        return 2
    if args.model and (args.side is None or args.candidate is None):
        print("--model needs --side (the rankers read the descriptor) and --candidate", file=sys.stderr)
        return 2
    if args.tau_match_sweep and args.candidate is None:
        print("--tau-match-sweep needs --candidate: it sweeps the frozen candidate's predeclared tau_match grid",
              file=sys.stderr)
        return 2
    th = Thresholds()
    cand_doc = None
    tm_grid: tuple[float, ...] = ()
    tm_values: tuple[float, ...] = ()
    side_data: SideData | None = None
    rankers: list[Ranker] = []
    models: dict[str, dict[str, str]] = {}
    try:
        if args.candidate is not None:
            from verify.select import NotEligibleForFreeze, candidate_thresholds, require_freeze_eligible   # heavy
            try:
                cand_doc = require_freeze_eligible(args.candidate)
            except NotEligibleForFreeze as e:
                raise ValueError(f"--candidate refused: {e}") from e
            if bool(cand_doc.get("simulation")) != info.simulation:
                raise ValueError(f"the candidate says simulation={cand_doc.get('simulation')} but the manifest says "
                                 f"simulation={info.simulation}")
            th = candidate_thresholds(cand_doc)
            if args.tau_match_sweep:
                try:
                    tm_grid = tau_match_grid_of(cand_doc, th)
                    tm_values = tau_match_sweep_values(th, tm_grid)
                except SweepRefused as e:
                    raise ValueError(f"--tau-match-sweep refused: {e}") from e
        if args.side is not None:
            side_data = SideData.from_store(SideStore.from_jsonl(args.side, positions.n_stops,
                                                                 episode_ids=[s.episode_id for s in specs]),
                                            [s.episode_id for s in specs], source=_rel(Path(args.side)))
        for item in args.model or ():
            from verify.ranker import RANKERS, load_model
            name, sep, path = item.partition("=")
            if not sep or name not in RANKERS or not path or name in models:
                raise ValueError(f"--model takes NAME=PATH once per name, NAME one of {sorted(RANKERS)}; got {item!r}")
            doc = json.loads(Path(path).read_text())
            if doc.get("thresholds") != th.to_dict():
                raise ValueError(f"model {path} was trained with thresholds {doc.get('thresholds')}, not the "
                                 f"candidate's {th.to_dict()}")
            if (doc.get("provenance") or {}).get("candidate_sha256") != file_sha256(args.candidate):
                raise ValueError(f"model {path} was trained for another candidate than {args.candidate}")
            r = load_model(path)
            if r.name != name or r.features != RANKERS[name].features:
                raise ValueError(f"model {path} is {r.name!r} with features {r.features}, not {name!r} with "
                                 f"{RANKERS[name].features}")
            rankers.append(r)
            models[name] = {"file": _rel(Path(path)), "sha256": file_sha256(path)}
    except (OSError, ValueError, KeyError) as e:
        print(f"bad real-data input: {e}", file=sys.stderr)
        return 2
    config = {"config_version": REAL_CONFIG_VERSION, "model_id": args.model_id, "synthetic": False,
              "simulation": info.simulation,
              "positions_file": _rel(Path(pos_path)), "positions_file_sha256": file_sha256(pos_path),
              "manifest_file": _rel(Path(args.manifest)), "manifest_sha256": file_sha256(args.manifest),
              "evidence_file": _rel(Path(args.evidence)), "evidence_sha256": file_sha256(args.evidence),
              # only the hash before the run: the labels are read after it, for scoring context only
              "view_labels_file": None if args.view_labels is None else _rel(Path(args.view_labels)),
              "view_labels_sha256": None if args.view_labels is None else file_sha256(args.view_labels),
              "bootstrap_stratified": planned["stratified"],
              "seed": args.seed, "budgets": list(args.budgets), "nominal_step_cm": positions.nominal_step_cm(),
              "travel_tol_cm": TRAVEL_TOL_CM, "cap_steps": _cap_steps_config(args.cap_steps),
              "travel_caps_cm": [None, *(cap_cm_from_steps(c, positions) for c in args.cap_steps)],
              "g_lookahead": dict(G_LOOKAHEAD),
              "g_myopic": {"lam_deg": GMYOPIC_LAM_DEG, "travel_norm_cm": th.travel_norm_cm},
              "r_seeds": list(range(args.r_seeds)), "bootstrap_B": args.bootstrap_B,
              "primary_budget": args.primary_budget, "thresholds": th.to_dict(), "positions": positions.to_dict(),
              "scoring_split": Split.TEST.value, "environment": info.environments[0],
              "thresholds_source": "frozen candidate" if cand_doc is not None else "placeholders",
              "candidate_file": None if args.candidate is None else _rel(Path(args.candidate)),
              "candidate_sha256": None if args.candidate is None else file_sha256(args.candidate),
              "side_file": None if args.side is None else _rel(Path(args.side)),
              "side_file_sha256": None if args.side is None else file_sha256(args.side),
              "ranker_models": models, "primary_contrast": list(PRIMARY_PAIR),
              "sensitivity": {"k": list(args.k_sensitivity), "lam": list(args.lam_sensitivity),
                              "note": "reported only, never selected; primary budget, every cap"},
              "common_targets": list(COMMON_TARGETS),
              # real-v6: the declared test analyses (prereg sections 1, 4, 9 and 10)
              "bootstrap_pairs": [list(x) for x in BOOTSTRAP_PAIRS],
              "by_family": {"levels": {FAMILY_LEVELS[lv]: {"family": FAMILY_LEVEL_TEXT[lv],
                                                           "bootstrap_strata": FAMILY_STRATA_TEXT[lv]}
                                       for lv in (OCCLUDER_FAMILY, SLOT_CLASS)},
                            "lightings": list(REPORT_LIGHTINGS), "bootstrap_pair": list(PRIMARY_PAIR),
                            "bootstrap_seed": f"derive_seed({args.seed}, 'by-family', family, lighting)",
                            "bootstrap_min_arrangements": MIN_STRATUM_SIZE,
                            "note": "descriptive (H3), not a new primary"},
              "gate_off_ablation": None if not args.gate_off_ablation else {
                  "policies": list(GATE_OFF_POLICIES), "tau_occ": GATE_OFF_TAU_OCC,
                  "pairs": [list(x) for x in GATE_OFF_PAIRS],
                  "note": "reported only, never selected; primary budget, every cap; contrasts at the primary setting"},
              "tau_match_sweep": None if not args.tau_match_sweep else {
                  "grid": list(tm_grid), "frozen_tau_match": th.tau_match, "extension": list(TAU_MATCH_SWEEP_ABOVE),
                  "extension_note": ("declared 2026-11-06, before the freeze: the curve on both sides of the "
                                     "frozen point"),
                  "values_run": list(tm_values),
                  "policies": list(TM_SWEEP_POLICIES),
                  "note": "reported only, never selected; primary budget, uncapped; other thresholds frozen"},
              "view_diagnostics": {"start_stop": positions.start_index, "k": th.k,
                                   "source": "view labels and evidence file, read after the run"},
              "code_sha256": code_sha256()}
    if cand_doc is not None:
        label = SIM_FROZEN_LABEL if info.simulation else REAL_FROZEN_LABEL
    sha = config_sha256(config)
    write_json(out / "config.json", {"label": label, "config": config, "config_sha256": sha})
    context = RunContext(model_id=args.model_id, config_version=REAL_CONFIG_VERSION, config_sha256=sha,
                         synthetic=False)
    logs = run_stage(specs, store, positions, th, args.budgets, config["travel_caps_cm"], config["r_seeds"], context,
                     side=side_data, rankers=rankers,
                     sensitivity=[*sensitivity_settings(th, args.k_sensitivity, args.lam_sensitivity),
                                  *((gate_off_setting(th),) if args.gate_off_ablation else ())],
                     primary_budget=args.primary_budget,
                     tau_match_sweep=tau_match_sweep_settings(th, tm_values) if args.tau_match_sweep else ())
    del store
    sim_label = label if info.simulation else None   # every output of a simulation manifest is labelled SIMULATION
    write_logs(out / "logs.jsonl", logs, simulation_label=sim_label)
    _, truth = episodes_from_arrangements(load_manifest_info(args.manifest).arrangements)
    save_truth(out / "ground_truth.json", truth, simulation_label=sim_label)
    no_diag = None
    labels: GeometricLabels | None = None
    if args.view_labels is not None:   # scoring context only, read after every policy has run
        try:
            if file_sha256(args.view_labels) != config["view_labels_sha256"]:
                raise ValueError(f"{args.view_labels} changed during the run")
            labels = load_geometric_labels(args.view_labels, positions.n_stops)
            if labels.simulation != info.simulation:   # as verify.select refuses it (sim_dev_data)
                raise ValueError(f"the manifest says simulation={info.simulation} but {args.view_labels} says "
                                 f"simulation={labels.simulation}")
            missing = sorted({s.episode_id for s in specs} - set(labels.episodes))
            if missing:
                raise ValueError(f"no view labels for {len(missing)} episodes, e.g. {missing[:3]}")
        except (OSError, ValueError, KeyError) as e:
            print(f"bad --view-labels: {e}", file=sys.stderr)
            return 2
        no_diag = labels.no_diagnostic_view()
        try:   # the gate's per-view rates read the evidence again, after the run, from the same (unchanged) file
            if file_sha256(args.evidence) != config["evidence_sha256"]:
                raise ValueError(f"{args.evidence} changed during the run")
            view_store = EvidenceStore.from_jsonl(args.evidence, positions.n_stops)
        except (OSError, ValueError, KeyError) as e:
            print(f"bad --evidence after the run: {e}", file=sys.stderr)
            return 2
    common = ({a.arrangement_id: (a.occluder_family, a.target_id) for a in info.arrangements}
              if all(a.occluder_family is not None for a in info.arrangements) else None)
    scored = score_stage(logs, truth, args.primary_budget, args.bootstrap_B, args.seed, strata=strata,
                         no_diagnostic=no_diag, common_target=common)
    if common is None and any(m["policy"] == PRIMARY_PAIR[0] for m in scored["metrics"]):
        scored["common_target_skipped"] = ("the manifest lacks occluder_family for some arrangement, so the "
                                           "predeclared common-target analysis could not run")
    # real-v6: the declared test analyses, all after the run (truth, labels and evidence re-read from disk)
    groups, seed_removed = group_results(logs), _seed_removed(logs)
    analyses: dict[str, Any] = {
        key: family_stage(logs, truth, info.arrangements, args.primary_budget, args.bootstrap_B, args.seed, no_diag,
                          level, groups, seed_removed) for level, key in FAMILY_LEVELS.items()}
    if args.tau_match_sweep:
        analyses["tau_match_sweep"] = {**config["tau_match_sweep"], "curves": tau_match_sweep_curves(
            scored["metrics"], th, sorted({*tm_values, th.tau_match}), args.primary_budget)}
    view_diag = {"start_stop": positions.start_index, "k": th.k}
    if labels is None:
        analyses["diagnostic_view_coverage"] = {"ran": False, "note": NO_LABELS_NOTE, **view_diag, "rows": []}
        analyses["gate_view_rates"] = {"ran": False, "note": NO_LABELS_NOTE, "thresholds": th.to_dict(), "rows": []}
    else:
        analyses["diagnostic_view_coverage"] = {
            "ran": True, **view_diag,
            "rows": family_coverage(specs, labels, positions.start_index, th.k, info.arrangements)}
        analyses["gate_view_rates"] = {"ran": True, "thresholds": th.to_dict(),
                                       "rows": gate_view_rates(specs, view_store, labels, positions, th)}
    results = {"label": label, "synthetic": False, "simulation": info.simulation, "config": config,
               "config_sha256": sha, "positions": positions.to_dict(), "thresholds": th.to_dict(),
               "files": {n: file_sha256(out / n) for n in ("ground_truth.json", "logs.jsonl")},
               "n_logs": len(logs), **scored,
               "diagnostics": diagnostics_stage(logs, positions, args.budgets, config["travel_caps_cm"]), **analyses}
    write_json(out / "results.json", results)
    (out / "table.md").write_text(render_table(results))
    print(f"{label}: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
