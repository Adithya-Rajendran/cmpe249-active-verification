"""Geometry checks behind configs/sim/arrange_shelf_v2.json: how tall the M16 board must be, and which (target model, slot
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

    python -m sim.feasibility --config configs/sim/arrange_shelf_v2.json scan --heights 0.10,0.12,0.14,0.15 --out scan.json
    python -m sim.feasibility --config configs/sim/arrange_shelf_v2.json matrix --tries 30 --out matrix.json
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
    o = R.occluder_spec(ctx.geometry, spec["geometry_occluder"])
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
        gap_max = min(float(spec["gap_m"][1]), front - ctx.config["clearances_m"]["occluder_front"] - o.thickness)
        gaps = np.arange(float(spec["gap_m"][0]), gap_max + 1e-9, gap_step)
        shifts = np.arange(lo_s, hi_s + 1e-9, shift_step)
        for h in heights:
            n = 0
            for gap in gaps:
                yc = front - gap - o.thickness / 2
                sl = A.sightline_x(ctx.start_cam, tb.centre(), yc)
                for sh in shifts:
                    body = A.Body((o.width, o.thickness, float(h)), (sl + sgn * sh, yc, 0.0))
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
    mx.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(A._repo_path(a.config).read_text())
    t0 = time.time()
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
        res = class_matrix(cfg, a.tries, a.workers, targets=[int(x) for x in a.targets.split(",") if x] or None,
                           classes=[c for c in a.classes.split(",") if c] or None)
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
