"""Geometry checks behind configs/sim/arrange_shelf_v2.json / v3 and the tabletop config: how tall the M16 board must be, and which (target model, slot
class) pairs can be placed at all. numpy only; slow (minutes), run on request, never in the unit tests except as a tiny smoke.

Why: the student's 2026-11-03 decision was a taller simulation M16 so that it can hide every target model, with the balanced
per-slot target assignment (sim/arrange.py assign_targets) built on the pairs that are feasible. "Feasible" here is the
generator's own rule set, not an opinion:
  - the target sits in #3 where every stop sees it whole (place_target);
  - the board stands in #3 inside the walls and under the ceiling clearance (the board's top <= clear height - ceiling);
  - it covers the target's projected hull at the start stop (p3_coverage_min), leaves min_clear_stops stops where the hulls do
    not touch (clear_coverage_max), and satisfies the side rule (adjacent same-side stop covered >= adjacent_min, nearest
    opposite stop <= opposite_near_max, further opposite stops clear).

Two tools:
  scan      for each target model, side and board height: does ANY placement exist? The target is placed by place_target
            (n_placements random draws, the same ones for every height), the board's gap and lateral shift are swept on a grid
            (gap_step, shift_step) over the family's ranges. Existence on the grid proves feasibility; absence is bounded by
            the sampling (n_placements, grid), which the output records. `--ceiling-free` sets clearances_m.ceiling to -1 so the
            board may be as tall as asked (the height the geometry would need if #3 were taller); `--set a.b.c=value` overrides
            any config value (e.g. a relaxed side rule) to see which rule is binding.
  matrix    for each (target model, slot class): `tries` real generator tries (try_place, attempt 0, probe seeds 3000..: dev
            seeds that no slot or reserve list uses) and where each failed (target / occluder / distractors / parked). This
            is the per-try success rate that decides how many retries a slot costs.

Both work on either environment (config 'environment'; sim/arrange.py board_dims / _front_limit), and the tabletop adds:
  heightscan  the tabletop board-height scan of a COMMITTED plan file (data/raw/2026-11-04_tabletop_scan/scan_plan.json, fixed
            before the run): every cell (family x M16 side x target x height, IND28 x width, white_box and ycb_box x choice x
            target) gets the same fixed number of real generator tries (try_place, attempt 0, the plan's probe seeds); the
            base config's sha256 is checked against the plan. Cells are logged as they finish.
  decide    applies the plan's decision rule (decide_height: admitted = >= 1 success; (a) M16 admits foam brick and tuna can on
            both sides; (b) S8 / IND28 retain the best target support of the whole grid, identities and success counts, a count
            not significantly lower than the best by a one-sided Fisher exact test at the plan's alpha) and picks H_T, the IND28
            width and the ycb_box choices, or reports that no height satisfies the rule.

    python -m sim.feasibility --config configs/sim/arrange_shelf_v2.json scan --heights 0.10,0.12,0.14,0.15 --out scan.json
    python -m sim.feasibility --config configs/sim/arrange_shelf_v2.json matrix --tries 30 --out matrix.json
    python -m sim.feasibility --workers 5 heightscan --plan data/raw/2026-11-04_tabletop_scan/scan_plan.json --log cells.jsonl --out results.json
    python -m sim.feasibility decide --plan data/raw/2026-11-04_tabletop_scan/scan_plan.json --results results.json --out decision.json
    python -m sim.feasibility --config C matrix --tries 100 --seed0 7200 --set occluder.families.S8.height_m=0.2 --out matrix.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections.abc import Mapping, Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from sim import arrange as A
from sim import rig as R

PROBE_SEED_START = 3000          # dev seeds not used by any primary or reserve seed of shelf_v1 / shelf_v2


def apply_overrides(cfg: Mapping, overrides: Mapping[str, Any]) -> dict:
    """A deep copy of cfg with 'a.b.c': value overrides applied (a missing key along the path is an error)."""
    out = copy.deepcopy(dict(cfg))
    for path, value in overrides.items():
        node = out
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[k]
        if keys[-1] not in node:
            raise KeyError(f"override {path!r}: no such key")
        node[keys[-1]] = value
    return out


def board_search(cfg: Mapping, tid: int, side: str, heights: Sequence[float], n_placements: int, seed: int = 123,
                 gap_step: float = 0.01, shift_step: float = 0.005, family: str = "M16", min_y: float | None = None) -> dict:
    """For each board height: how many of `n_placements` target placements (place_target, Draw([seed, tid])) admit at least one
    (gap, shift) on the grid that satisfies every rule of `family` (see the module docstring), and how many (placement, gap,
    shift) combinations do in total. min_y keeps only placements whose base-centre y is at least that (deep targets)."""
    ctx = A.load_context(dict(cfg))
    oc = ctx.config["occluder"]
    spec = oc["families"][family]
    _, o_width, o_thickness, _ = A.board_dims(ctx, spec)         # shelf: the geometry file's board; tabletop: the family's own
    sgn = A.side_sign(side)
    lo_s, hi_s = (float(v) for v in spec["side_shift_m"])
    d = A.Draw([seed, int(tid)])
    placements = []
    for _ in range(100 * n_placements):
        if len(placements) == n_placements:
            break
        t = A.place_target(ctx, d, int(tid))
        if t is not None and (min_y is None or t["body"].base[1] >= min_y):
            placements.append(t)
    res = {h: {"placements_feasible": 0, "combos": 0, "example": None} for h in heights}
    for t in placements:
        tb = t["body"]
        hulls = [A.projected_hull(c, tb.shape_points()) for c in ctx.cams]
        areas = [abs(A.polygon_area(h)) for h in hulls]
        front = float(tb.footprint()[:, 1].min())
        gap_max = min(float(spec["gap_m"][1]), front - A._front_limit(ctx) - o_thickness)
        gaps = np.arange(float(spec["gap_m"][0]), gap_max + 1e-9, gap_step)
        shifts = np.arange(lo_s, hi_s + 1e-9, shift_step)
        for h in heights:
            n = 0
            for gap in gaps:
                yc = front - gap - o_thickness / 2
                sl = A.sightline_x(ctx.start_cam, tb.centre(), yc)
                for sh in shifts:
                    body = A.Body((o_width, o_thickness, float(h)), (sl + sgn * sh, yc, 0.0))
                    if not A._occluder_ok(ctx, body, tb):
                        continue
                    covs = [min(1.0, A.overlap_area(hu, A.projected_hull(c, body.shape_points())) / a)
                            for c, hu, a in zip(ctx.cams, hulls, areas)]
                    if covs[ctx.start_index] < float(oc["p3_coverage_min"]):
                        continue
                    if not A.stop_rule_ok(dict(zip((c.name for c in ctx.cams), covs)), False, oc):
                        continue
                    if "side_rule" in spec and not A.side_rule_ok(covs, ctx.start_index, side, spec["side_rule"],
                                                                  float(oc["clear_coverage_max"])):
                        continue
                    n += 1
                    if res[h]["example"] is None:
                        res[h]["example"] = {"target_base": [round(float(v), 4) for v in tb.base], "yaw_deg": round(float(tb.yaw_deg), 1),
                                             "gap_m": round(float(gap), 4), "shift_m": round(float(sh), 4),
                                             "coverage": [round(v, 3) for v in covs]}
            res[h]["combos"] += n
            res[h]["placements_feasible"] += int(n > 0)
    return {"target": int(tid), "name": ctx.item(tid)["name"], "side": side, "family": family, "n_placements": len(placements),
            "gap_step": gap_step, "shift_step": shift_step, "min_y": min_y,
            "heights": {f"{h:g}": v for h, v in res.items()}}


def _scan_job(a: dict) -> dict:
    return board_search(a["cfg"], a["tid"], a["side"], a["heights"], a["n"], a["seed"], a["gap_step"], a["shift_step"],
                        min_y=a.get("min_y"))


def height_scan(cfg: Mapping, targets: Sequence[int], heights: Sequence[float], n_placements: int, workers: int = 4,
                overrides: Mapping[str, Any] | None = None, seed: int = 123, gap_step: float = 0.01, shift_step: float = 0.005,
                min_y: float | None = None) -> dict:
    """board_search over targets x {L, R}; the smallest height with a feasible combination per (target, side), or None."""
    c = apply_overrides(cfg, overrides or {})
    jobs = [{"cfg": c, "tid": int(t), "side": s, "heights": list(heights), "n": n_placements, "seed": seed,
             "gap_step": gap_step, "shift_step": shift_step, "min_y": min_y} for t in targets for s in A.SIDES]
    if workers > 1:
        with Pool(workers) as p:
            rows = p.map(_scan_job, jobs, chunksize=1)
    else:
        rows = [_scan_job(j) for j in jobs]
    for r in rows:
        feas = [float(h) for h, v in r["heights"].items() if v["combos"] > 0]
        r["min_feasible_height_m"] = min(feas) if feas else None
    return {"overrides": dict(overrides or {}), "heights": list(heights), "rows": rows}


def slot_classes(cfg: Mapping) -> list[tuple[str, str | None]]:
    q = cfg["slots"]["dev"]["quota"]
    return sorted(set(A._quota_entries(q)), key=lambda e: (e[0], e[1] or ""))


def _matrix_job(a: dict) -> dict:
    ctx = A.load_context(a["cfg"])
    fam, side, tid = a["family"], a["side"], a["tid"]
    fails: dict[str, int] = {}
    ok = 0
    t0 = time.time()
    for k in range(a["tries"]):
        parts, reason, _ = A.try_place(ctx, a["seed0"] + k, 0, fam, side, tid)
        if parts is not None:
            ok += 1
        else:
            fails[reason] = fails.get(reason, 0) + 1
    return {"target": tid, "name": ctx.item(tid)["name"], "class": A.class_key(fam, side), "tries": a["tries"], "ok": ok,
            "fail_reasons": fails, "seconds": round(time.time() - t0, 1)}


def class_matrix(cfg: Mapping, tries: int, workers: int = 4, targets: Sequence[int] | None = None,
                 classes: Sequence[str] | None = None, seed0: int = PROBE_SEED_START) -> dict:
    """The (target x slot class) table of real generator tries. Needs a config with slots.target_assignment (the target is
    given to try_place)."""
    if cfg["slots"].get("target_assignment") is None:
        raise ValueError("class_matrix needs a config with slots.target_assignment")
    tids = [int(t) for t in (targets if targets is not None else cfg["target"]["objects"])]
    jobs = [{"cfg": dict(cfg), "family": f, "side": s, "tid": t, "tries": tries, "seed0": seed0}
            for f, s in slot_classes(cfg) if classes is None or A.class_key(f, s) in classes for t in tids]
    if workers > 1:
        with Pool(workers) as p:
            rows = p.map(_matrix_job, jobs, chunksize=1)
    else:
        rows = [_matrix_job(j) for j in jobs]
    return {"tries_per_cell": tries, "probe_seeds": [seed0, seed0 + tries - 1], "rows": rows}


# ---------------------------------------------------------------- tabletop board-height scan (plan -> results -> decision)

PLAN_SCHEMA = "cmpe249.tabletop-height-scan-plan.v1"
RESULTS_SCHEMA = "cmpe249.tabletop-height-scan-results.v1"
DECISION_SCHEMA = "cmpe249.tabletop-height-scan-decision.v1"


def scan_cells(plan: Mapping) -> list[dict]:
    """Every cell of a height-scan plan, in a fixed order: for each height, S8 x target, M16 x side x target, IND28 x width x
    target; then the height-independent cells (white_box x target, ycb_box x choice x target). A cell is (family, side,
    target, height or None, width or None, ycb choice or None) plus its config overrides."""
    cells = []
    fams = plan["board_families"]
    for h in plan["heights_m"]:
        hov = {f"occluder.families.{f}.height_m": float(h) for f in fams}
        for t in plan["targets"]:
            cells.append({"family": "S8", "side": None, "target": int(t), "height_m": float(h), "width_m": None, "choice": None,
                          "overrides": dict(hov)})
        for side in plan["m16_sides"]:
            for t in plan["targets"]:
                cells.append({"family": "M16", "side": side, "target": int(t), "height_m": float(h), "width_m": None,
                              "choice": None, "overrides": dict(hov)})
        for w in plan["ind28_widths_m"]:
            for t in plan["targets"]:
                cells.append({"family": "IND28", "side": None, "target": int(t), "height_m": float(h), "width_m": float(w),
                              "choice": None, "overrides": {**hov, "occluder.families.IND28.width_m": float(w)}})
    h0 = {f"occluder.families.{f}.height_m": float(plan["heights_m"][0]) for f in fams}    # unused by these families
    for t in plan["targets"]:
        cells.append({"family": "white_box", "side": None, "target": int(t), "height_m": None, "width_m": None, "choice": None,
                      "overrides": dict(h0)})
    for ch in plan["ycb_box_choices"]:
        for t in plan["targets"]:
            cells.append({"family": "ycb_box", "side": None, "target": int(t), "height_m": None, "width_m": None, "choice": list(ch),
                          "overrides": {**h0, "occluder.families.ycb_box.choices": [list(ch)]}})
    for i, c in enumerate(cells):
        c["cell"] = i
    return cells


def _cell_job(a: dict) -> dict:
    cfg = apply_overrides(a["cfg"], a["cell"]["overrides"])
    ctx = A.load_context(cfg)
    c = a["cell"]
    fails: dict[str, int] = {}
    ok_seeds = []
    t0 = time.time()
    for k in range(a["tries"]):
        seed = a["seed0"] + k
        parts, reason, _ = A.try_place(ctx, seed, 0, c["family"], c["side"], c["target"])
        if parts is not None:
            ok_seeds.append(seed)
        else:
            fails[reason] = fails.get(reason, 0) + 1
    return {**{k: v for k, v in c.items() if k != "overrides"}, "name": ctx.item(c["target"])["name"], "tries": a["tries"],
            "successes": len(ok_seeds), "rate": len(ok_seeds) / a["tries"], "ok_seeds": ok_seeds, "fail_reasons": fails,
            "seconds": round(time.time() - t0, 1)}


def check_plan(plan: Mapping, plan_dir: Path) -> dict:
    """The plan's base config (sha256 of the file checked against the plan) as a dict; refuses a plan whose probe seeds are
    not dev seeds or overlap the base config's slot seeds."""
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan schema must be {PLAN_SCHEMA!r}")
    bpath = (plan_dir / plan["base_config"]["file"]).resolve()
    sha = A.sha256_file(bpath)
    if sha != plan["base_config"]["sha256"]:
        raise ValueError(f"{bpath}: sha256 {sha} != the plan's {plan['base_config']['sha256']}")
    cfg = json.loads(bpath.read_text())
    used = set()
    for split in ("dev",):
        blk = cfg["slots"][split]
        used |= set(range(int(blk["primary_seed_start"]), int(blk["primary_seed_start"]) + len(A._quota_entries(blk["quota"]))))
        used |= {int(v) for vs in blk["reserve"].values() for v in vs}
    for key in ("probe_seeds", "class_matrix_probe_seeds"):
        s0, n = int(plan[key]["start"]), int(plan[key]["n"])
        seeds = range(s0, s0 + n)
        for v in seeds:
            A.check_dev_seed(v)
        if used & set(seeds):
            raise ValueError(f"plan {key} overlap the base config's slot seeds")
    return cfg


def run_height_scan(plan_path: str | Path, workers: int, log_path: str | Path | None = None) -> dict:
    """Runs every cell of the plan (scan_cells) with tries_per_cell real generator tries each (try_place, attempt 0, probe seeds
    start..start + n - 1, the same seeds in every cell). Cells are written to log_path (JSON lines) as they finish."""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    cfg = check_plan(plan, plan_path.parent)
    tries, seed0 = int(plan["tries_per_cell"]), int(plan["probe_seeds"]["start"])
    if tries != int(plan["probe_seeds"]["n"]):
        raise ValueError("tries_per_cell must equal probe_seeds.n")
    cells = scan_cells(plan)
    jobs = [{"cfg": cfg, "cell": c, "tries": tries, "seed0": seed0} for c in cells]
    t0 = time.time()
    rows = []
    log = open(log_path, "a") if log_path else None
    try:
        with Pool(workers) as p:
            for r in p.imap_unordered(_cell_job, jobs, chunksize=1):
                rows.append(r)
                if log:
                    log.write(json.dumps(r) + "\n")
                    log.flush()
                print(f"[{time.time() - t0:7.0f} s] cell {r['cell']:3d}/{len(cells)} {r['family']}{'-' + r['side'] if r['side'] else ''} "
                      f"h={r['height_m']} w={r['width_m']} choice={r['choice']} {r['name']}: {r['successes']}/{r['tries']} "
                      f"{r['fail_reasons']} ({r['seconds']} s)", flush=True)
    finally:
        if log:
            log.close()
    rows.sort(key=lambda r: r["cell"])
    return {"schema": RESULTS_SCHEMA, "plan": {"file": plan_path.name, "sha256": A.sha256_file(plan_path)},
            "base_config_sha256": plan["base_config"]["sha256"],
            "code": {"sim/arrange.py": A.sha256_file(A.THIS_FILE), "sim/feasibility.py": A.sha256_file(Path(__file__).resolve()),
                     "sim/tabletop.py": A.sha256_file(A.TABLETOP_FILE), "sim/rig.py": A.sha256_file(A.RIG_FILE)},
            "numpy": np.__version__, "workers": workers, "seconds": round(time.time() - t0, 1), "n_cells": len(rows), "rows": rows}


def fisher_one_sided_lower(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided Fisher exact p-value that group 1's success rate (k1 of n1) is LOWER than group 2's (k2 of n2): P(X <= k1)
    for X hypergeometric with the table's margins. Pure Python (math.comb)."""
    import math

    K, N = k1 + k2, n1 + n2
    lo = max(0, K - n2)
    den = math.comb(N, K)
    return sum(math.comb(n1, x) * math.comb(n2, K - x) for x in range(lo, k1 + 1)) / den


def _support(rows: Sequence[Mapping]) -> dict:
    """{target: successes} of a list of cells."""
    return {int(r["target"]): int(r["successes"]) for r in rows}


def decide_height(plan: Mapping, results: Mapping) -> dict:
    """The plan's decision rule (plan['decision_rule'], fixed before the run) applied to the results. Admitted at a cell =
    successes >= 1. (a) M16: foam brick (21) and tuna can (6) admitted on both sides. (b) S8 / IND28 retain the best support of
    the whole grid: the best identities B_F = every target admitted anywhere in F's grid (S8: heights; IND28: heights x widths);
    a height h retains it if (for IND28: for some width W) every target of B_F is admitted at h and its success count at h is
    not significantly lower than its largest count anywhere in the grid (one-sided Fisher exact p >= alpha). H_T = the
    smallest height meeting (a) and (b); None if none does. IND28's width at H_T = the narrowest width meeting (b) there.
    ycb_box: a choice is kept if it admits at least one target."""
    rule = plan["decision_rule"]
    alpha = float(rule["alpha"])
    rows = results["rows"]
    N = int(plan["tries_per_cell"])
    heights = [float(h) for h in plan["heights_m"]]
    widths = [float(w) for w in plan["ind28_widths_m"]]
    must = [int(t) for t in rule["m16_must_admit"]]

    def cells(fam, h=None, side=None, w=None, choice=None):
        return [r for r in rows if r["family"] == fam and (h is None or r["height_m"] == h) and r["side"] == side
                and (w is None or r["width_m"] == w) and (choice is None or r["choice"] == choice)]

    per_height = {}
    s8_best = {t: max(int(r["successes"]) for r in cells("S8") if r["target"] == t) for t in plan["targets"]}
    ind_best = {t: max(int(r["successes"]) for r in cells("IND28") if r["target"] == t) for t in plan["targets"]}
    B_s8 = sorted(t for t, k in s8_best.items() if k > 0)
    B_ind = sorted(t for t, k in ind_best.items() if k > 0)

    def retains(sup: Mapping[int, int], best: Mapping[int, int], B: Sequence[int]) -> tuple[bool, dict]:
        detail = {}
        ok = True
        for t in B:
            k = sup.get(t, 0)
            p = fisher_one_sided_lower(k, N, best[t], N)
            good = k >= 1 and p >= alpha
            detail[str(t)] = {"successes": k, "best_successes": best[t], "p_lower": round(p, 6), "retained": good}
            ok &= good
        return ok, detail

    for h in heights:
        m16 = {s: _support(cells("M16", h, s)) for s in plan["m16_sides"]}
        a_ok = all(m16[s].get(t, 0) >= 1 for s in plan["m16_sides"] for t in must)
        s8_ok, s8_detail = retains(_support(cells("S8", h)), s8_best, B_s8)
        ind = {}
        ind_w = None
        for w in widths:
            ok_w, det = retains(_support(cells("IND28", h, w=w)), ind_best, B_ind)
            ind[f"{w:g}"] = {"ok": ok_w, "detail": det,
                             "admitted": sorted(t for t, k in _support(cells("IND28", h, w=w)).items() if k >= 1)}
            if ok_w and ind_w is None:
                ind_w = w
        per_height[f"{h:g}"] = {
            "a_m16_ok": a_ok,
            "m16_admitted": {s: sorted(t for t, k in m16[s].items() if k >= 1) for s in plan["m16_sides"]},
            "b_s8_ok": s8_ok, "s8_admitted": sorted(t for t, k in _support(cells("S8", h)).items() if k >= 1), "s8_detail": s8_detail,
            "b_ind28_ok": ind_w is not None, "ind28_narrowest_width_m": ind_w, "ind28_by_width": ind,
            "ok": bool(a_ok and s8_ok and ind_w is not None)}
    chosen = next((h for h in heights if per_height[f"{h:g}"]["ok"]), None)
    ycb = {}
    for ch in plan["ycb_box_choices"]:
        sup = _support(cells("ycb_box", choice=list(ch)))
        ycb[f"{ch[0]}:{ch[1]}"] = {"admitted": sorted(t for t, k in sup.items() if k >= 1), "successes": sup,
                                   "kept": any(k >= 1 for k in sup.values())}
    wb = _support(cells("white_box"))
    return {"schema": DECISION_SCHEMA, "plan_sha256": results["plan"]["sha256"], "results_n_cells": results["n_cells"],
            "best_support": {"S8": {"identities": B_s8, "best_successes": s8_best},
                             "IND28": {"identities": B_ind, "best_successes": ind_best}},
            "per_height": per_height, "H_T_m": chosen,
            "ind28_width_m": None if chosen is None else per_height[f"{chosen:g}"]["ind28_narrowest_width_m"],
            "ycb_box": ycb, "white_box": {"successes": wb, "admitted": sorted(t for t, k in wb.items() if k >= 1)},
            "outcome": ("no height satisfies (a) and (b)" if chosen is None else
                        f"H_T = {chosen:g} m: the smallest scanned height meeting (a) and (b)")}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.feasibility", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(A.DEFAULT_CONFIG_PATH.relative_to(R.REPO_ROOT)))
    ap.add_argument("--workers", type=int, default=4)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="does a feasible M16 placement exist, per target, side and board height")
    sc.add_argument("--heights", default="0.10,0.12,0.14,0.15")
    sc.add_argument("--targets", default="", help="comma list of model ids (default: the config's target.objects)")
    sc.add_argument("--placements", type=int, default=60)
    sc.add_argument("--min-y", type=float, default=None, help="keep only target placements at least this deep (base-centre y, m)")
    sc.add_argument("--gap-step", type=float, default=0.01)
    sc.add_argument("--shift-step", type=float, default=0.005)
    sc.add_argument("--ceiling-free", action="store_true", help="clearances_m.ceiling = -1: the board may be taller than #3 allows")
    sc.add_argument("--set", action="append", default=[], metavar="a.b.c=JSON", help="override a config value (repeatable)")
    sc.add_argument("--out", required=True)
    mx = sub.add_parser("matrix", help="real generator tries per (target model, slot class)")
    mx.add_argument("--tries", type=int, default=30)
    mx.add_argument("--classes", default="", help="comma list like S8,M16-L (default: all)")
    mx.add_argument("--targets", default="", help="comma list of model ids (default: the config's target.objects)")
    mx.add_argument("--seed0", type=int, default=PROBE_SEED_START, help="first probe seed (dev seeds no slot or reserve list uses)")
    mx.add_argument("--set", action="append", default=[], metavar="a.b.c=JSON", help="override a config value (repeatable)")
    mx.add_argument("--out", required=True)
    hs = sub.add_parser("heightscan", help="tabletop board-height scan: every cell of a committed plan (JSON) with real generator tries")
    hs.add_argument("--plan", required=True)
    hs.add_argument("--log", default=None, help="JSON lines, one per finished cell (appended)")
    hs.add_argument("--out", required=True)
    de = sub.add_parser("decide", help="apply the plan's decision rule to height-scan results")
    de.add_argument("--plan", required=True)
    de.add_argument("--results", required=True)
    de.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "heightscan":
        res = run_height_scan(a.plan, a.workers, a.log)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print(f"wrote {a.out} ({res['seconds']} s)")
        return 0
    if a.cmd == "decide":
        plan, results = json.loads(Path(a.plan).read_text()), json.loads(Path(a.results).read_text())
        if results["plan"]["sha256"] != A.sha256_file(a.plan):
            print("refused: the results were not made from this plan file", file=sys.stderr)
            return 2
        res = decide_height(plan, results)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print(res["outcome"])
        return 0
    cfg = json.loads(A._repo_path(a.config).read_text())
    if a.cmd == "scan":
        ov = {"clearances_m.ceiling": -1.0} if a.ceiling_free else {}
        for kv in a.set:
            k, v = kv.split("=", 1)
            ov[k] = json.loads(v)
        targets = [int(x) for x in a.targets.split(",") if x] or cfg["target"]["objects"]
        res = height_scan(cfg, targets, [float(x) for x in a.heights.split(",")], a.placements, a.workers, ov,
                          gap_step=a.gap_step, shift_step=a.shift_step, min_y=a.min_y)
        for r in res["rows"]:
            print(f"{r['name']:16s} {r['side']}  placements {r['n_placements']}  min feasible height {r['min_feasible_height_m']}  "
                  + "  ".join(f"{h}:{v['combos']}" for h, v in r["heights"].items()))
    else:
        ov = {}
        for kv in a.set:
            k, v = kv.split("=", 1)
            ov[k] = json.loads(v)
        cfg = apply_overrides(cfg, ov)
        res = class_matrix(cfg, a.tries, a.workers, targets=[int(x) for x in a.targets.split(",") if x] or None,
                           classes=[c for c in a.classes.split(",") if c] or None, seed0=a.seed0)
        res["overrides"] = ov
        for r in res["rows"]:
            print(f"{r['class']:10s} {r['name']:16s} ok {r['ok']}/{r['tries']}  {r['fail_reasons']}")
    res["config"] = {"path": a.config, "sha256": A.sha256_json(cfg)}
    res["generator_sha256"] = A.sha256_file(A.THIS_FILE)
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {a.out} ({res['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
