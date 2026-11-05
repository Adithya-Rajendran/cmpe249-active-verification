"""Score a registration candidate against C0 with the pinned criteria (notes/2026-11-04-registration-repair-plan.md,
pinned b144e04, Amendment 1). Evaluation only: the simulator's labels, true homographies and target-alone masks are
read here and never by the evidence step.

Per lighting, on every view of the jobs (fixed, complete denominators; a refused view stays in, scored as the
evidence step scored it):
  1. refusals < 5 % (count x 20 < n);
  2. present-labelled (visible and occluded episodes) clear misses: 0;
  3. removed-labelled strong matches: no more than C0;
  4. diagnostic removal yield (clear misses on removed diagnostic views): no lower than C0;
  5. common-accepted views (accepted by both): box-corner error vs the true TARGET plane, candidate median <= C0 + 0.5
     px, p90 <= C0 + 1 px, max <= C0 + 2 px, both summaries on exactly that view set (numpy linear quantiles);
  6. newly accepted views (refused by C0): each error <= the max over C0-accepted views + 2 px.
Reported, never gating: the other plane (shelf front, tabletop table), target-alone silhouette IoU, newly refused /
recovered views, and run time (Amendment 1: registration and complete evidence latency median / p90 / max with the
candidate / C0 ratio, and batch throughput from the *.timing.json records). Thresholds: the shelf v03_image candidate,
sha256-checked. Box-corner error = max over the 4 reference-box corners of |H^-1 c - H_true^-1 c| (query px).
Silhouette IoU = target_alone_<stop> (reference pose) against target_alone_revisit_<stop> warped by H (nearest).

Usage (repo root):
    python tools/registration_compare.py --set SET --jobs jobs_refL1.json --c0 C0.diag.jsonl --cand C1.diag.jsonl \\
        --name C1 [--c0-timing T0.json --cand-timing T1.json] [--original-c0 v03_image_refL1.diag.jsonl] \\
        [--design] --out-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from verify.schema import EvidenceRecord, Observation, Positions, Thresholds  # noqa: E402
from verify.stopping import StoppingRule  # noqa: E402

CANDIDATE = Path("/home/jetson/Projects/SJSU/Fall26/CMPE249/cmpe249-active-verification/runs/shelf_dev_v3_evidence/"
                 "select/v03_image/frozen_candidate.json")
CANDIDATE_SHA = "4a06de0eebe407aad27b58fdfce3751460b5357924aba93a6f6830095b93b983"
BUDGET = {"median": 0.5, "p90": 1.0, "max": 2.0}
NEW_MARGIN = 2.0
LIGHTINGS = ("L1", "L2")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path) -> dict[tuple[str, int], dict]:
    with p.open() as f:
        return {(d["episode_id"], int(d["stop"])): d for d in (json.loads(x) for x in f if x.strip())}


def rule() -> StoppingRule:
    if sha(CANDIDATE) != CANDIDATE_SHA:
        raise SystemExit(f"{CANDIDATE}: sha256 is not the recorded {CANDIDATE_SHA}")
    t = json.loads(CANDIDATE.read_text())["thresholds"]
    return StoppingRule(Thresholds(tau_match=t["tau_match"], tau_q=t["tau_q"], tau_occ=t["tau_occ"], k=t["k"]))


def decisions(d: dict, r: StoppingRule) -> dict[str, bool]:
    o = Observation.from_record(EvidenceRecord(int(d["stop"]), d["match_score"], d["quality"], d["obstruction"]), 0,
                                Positions.default())
    return {"gate": r.passes_gate(o), "clear_miss": r.is_clear_miss(o), "strong_match": r.is_strong_match(o)}


def refusals_ok(refused: int, n: int) -> bool:
    """Criterion 1: refusals < 5 % of the fixed denominator (tabletop <= 8 / 165, shelf <= 44 / 900)."""
    return refused * 20 < n


def corner_err(H: np.ndarray, Ht: np.ndarray, box) -> float:
    c = np.array([[box[0], box[1], 1], [box[2], box[1], 1], [box[2], box[3], 1], [box[0], box[3], 1]], float).T
    a, b = np.linalg.inv(H) @ c, np.linalg.inv(Ht) @ c
    return float(np.linalg.norm((a[:2] / a[2] - b[:2] / b[2]).T, axis=1).max())


def iou(ref_m: np.ndarray, qry_m: np.ndarray, H: np.ndarray) -> float:
    q = cv2.warpPerspective(qry_m, H, (ref_m.shape[1], ref_m.shape[0]), flags=cv2.INTER_NEAREST) > 0
    r = ref_m > 0
    return float(np.count_nonzero(r & q) / max(np.count_nonzero(r | q), 1))


def q3(xs) -> dict[str, float] | None:
    if not len(xs):
        return None
    a = np.asarray(xs, float)
    return {"n": int(len(a)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def view_meta(set_dir: Path, design: bool) -> tuple[dict, dict, str]:
    labels = json.loads((set_dir / ("view_labels_design.json" if design else "view_labels.json")).read_text())["episodes"]
    ta = json.loads((set_dir / ("true_alignment_design.json" if design else "true_alignment.json")).read_text())
    env = ta.get("environment", "shelf")
    truth = {}
    for arr, a in ta["arrangements"].items():
        for s in a["stops"].values():
            other = "H_table_plane" if env == "tabletop" else "H_front_plane"
            truth[(arr, int(s["stop_index"]))] = (np.array(s["H_target_plane"]), np.array(s[other]))
    return labels, truth, env


def view_class(lab: dict, stop: int) -> str:
    s = lab["stops"][stop]
    if lab["condition"] == "visible":
        return "present"
    if lab["condition"] == "occluded":
        vf = s["visible_fraction"]
        return "occluded C" if vf >= 0.9 else "occluded H" if vf <= 0.05 else "occluded p"
    return "removed diag" if s["diagnostic"] else "removed non-diag"


def score(args) -> dict:
    set_dir = Path(args.set)
    jobs = json.loads((set_dir / args.jobs).read_text())["jobs"]
    keys = [(j["episode_id"], int(j["stop"])) for j in jobs]
    c0, cd = load(Path(args.c0)), load(Path(args.cand))
    for name, d in (("C0", c0), (args.name, cd)):
        if sorted(d) != sorted(keys) or len(d) != len(keys):
            raise SystemExit(f"{name}: the diagnostics do not hold every job key exactly once")
    labels, truth, env = view_meta(set_dir, args.design)
    r = rule()
    masks: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for j, k in zip(jobs, keys):
        arr = k[0].split("-")[0]
        ann = json.loads(Path(j["annotation"]).read_text())
        box = ann["targets"][j["target"]]["box"]
        sname = ann["stop_name"]
        if (arr, sname) not in masks:
            md = set_dir / arr / "masks"
            masks[(arr, sname)] = (cv2.imread(str(md / f"target_alone_{sname}.png"), cv2.IMREAD_GRAYSCALE),
                                   cv2.imread(str(md / f"target_alone_revisit_{sname}.png"), cv2.IMREAD_GRAYSCALE))
        rm, qm = masks[(arr, sname)]
        Ht, Ho = truth[(arr, k[1])]
        lab = labels[k[0]]
        row = {"key": list(k), "lighting": lab["lighting"], "truth": lab["truth"], "class": view_class(lab, k[1]),
               "silhouette_iou_true_target": iou(rm, qm, Ht)}
        for tag, d in (("c0", c0[k]), ("cand", cd[k])):
            H = None if d.get("H") is None else np.array(d["H"])
            row[tag] = {"refused": d["registration"].get("refused"), **decisions(d, r),
                        "registration_s": d["timing"]["registration_s"], "evidence_s": d["timing"]["evidence_s"],
                        "err_target": None if H is None else corner_err(H, Ht, box),
                        "err_other": None if H is None else corner_err(H, Ho, box),
                        "silhouette_iou": None if H is None else iou(rm, qm, H)}
        rows.append(row)
    out = {"set": str(set_dir), "jobs": args.jobs, "design": args.design, "environment": env, "candidate": args.name,
           "inputs": {p: sha(Path(p)) for p in (args.c0, args.cand, str(set_dir / args.jobs))},
           "thresholds_sha256": CANDIDATE_SHA, "gating": not args.design, "per_lighting": {}}
    for L in LIGHTINGS:
        rs = [x for x in rows if x["lighting"] == L]
        n = len(rs)
        res: dict = {"n": n, "denominators": {
            "refused": n, "present_clear_miss": sum(x["truth"] == "present" for x in rs),
            "removed_strong_match": sum(x["truth"] == "removed" for x in rs),
            "diag_removal_yield": sum(x["class"] == "removed diag" for x in rs),
            "present_gate_pass": sum(x["class"] == "present" for x in rs)}}
        for tag in ("c0", "cand"):
            res[tag] = {
                "refused": sum(x[tag]["refused"] is not None for x in rs),
                "present_clear_miss": sum(x["truth"] == "present" and x[tag]["clear_miss"] for x in rs),
                "removed_strong_match": sum(x["truth"] == "removed" and x[tag]["strong_match"] for x in rs),
                "diag_removal_yield": sum(x["class"] == "removed diag" and x[tag]["clear_miss"] for x in rs),
                "present_gate_pass": sum(x["class"] == "present" and x[tag]["gate"] for x in rs),
                "registration_s": q3([x[tag]["registration_s"] for x in rs]),
                "evidence_s": q3([x[tag]["evidence_s"] for x in rs])}
        common = [x for x in rs if x["c0"]["refused"] is None and x["cand"]["refused"] is None]
        new = [x for x in rs if x["c0"]["refused"] is not None and x["cand"]["refused"] is None]
        lost = [x for x in rs if x["c0"]["refused"] is None and x["cand"]["refused"] is not None]
        g0, g1 = q3([x["c0"]["err_target"] for x in common]), q3([x["cand"]["err_target"] for x in common])
        c0_acc_max = max((x["c0"]["err_target"] for x in rs if x["c0"]["refused"] is None), default=float("nan"))
        res.update({
            "common_accepted": len(common), "newly_accepted": [x["key"] for x in new],
            "newly_refused": [[*x["key"], x["cand"]["refused"]] for x in lost],
            "geometry_common_target": {"c0": g0, "cand": g1},
            "geometry_common_other": {"c0": q3([x["c0"]["err_other"] for x in common]),
                                      "cand": q3([x["cand"]["err_other"] for x in common])},
            "silhouette_common": {"c0": q3([x["c0"]["silhouette_iou"] for x in common]),
                                  "cand": q3([x["cand"]["silhouette_iou"] for x in common]),
                                  "true_target": q3([x["silhouette_iou_true_target"] for x in common])},
            "newly_accepted_err_target": [[*x["key"], x["cand"]["err_target"], x["cand"]["silhouette_iou"]] for x in new],
            "c0_accepted_max_err_target": c0_acc_max})
        crit = {
            "1_refusals_lt_5pct": refusals_ok(res["cand"]["refused"], n),
            "2_no_present_clear_miss": res["cand"]["present_clear_miss"] == 0,
            "3_removed_strong_match_le_c0": res["cand"]["removed_strong_match"] <= res["c0"]["removed_strong_match"],
            "4_diag_yield_ge_c0": res["cand"]["diag_removal_yield"] >= res["c0"]["diag_removal_yield"],
            "5_common_geometry": bool(g0 and g1 and all(g1[s] <= g0[s] + b for s, b in BUDGET.items())),
            "6_newly_accepted_geometry": all(x["cand"]["err_target"] <= c0_acc_max + NEW_MARGIN for x in new)}
        res["criteria"] = crit
        res["timing_ratio"] = {m: {s: (res["cand"][m][s] / res["c0"][m][s]) if res["c0"][m][s] else None
                                   for s in ("median", "p90", "max")} for m in ("registration_s", "evidence_s")}
        out["per_lighting"][L] = res
    out["pass"] = all(all(v["criteria"].values()) for v in out["per_lighting"].values())
    for tag, p in (("c0", args.c0_timing), ("cand", args.cand_timing)):
        if p:
            out[f"batch_{tag}"] = json.loads(Path(p).read_text())
    if args.original_c0:
        orig = load(Path(args.original_c0))
        out["c0_vs_original"] = {
            "max_abs_diff": {f: max(abs(c0[k][f] - orig[k][f]) for k in keys) for f in ("match_score", "quality", "obstruction")},
            "refusals_equal": all((c0[k]["registration"].get("refused") is None) == (orig[k]["registration"].get("refused") is None) for k in keys)}
    out["rows"] = rows
    return out


def fmt(g: dict | None) -> str:
    return "-" if not g else f"{g['median']:.2f} / {g['p90']:.2f} / {g['max']:.2f}"


def markdown(o: dict) -> str:
    c = o["candidate"]
    md = [f"# {c} against C0: {Path(o['set']).name} {'(design cases, NOT gating)' if o['design'] else ''}", "",
          "SIMULATION, development data. Pinned plan notes/2026-11-04-registration-repair-plan.md (b144e04, Amendment 1). "
          "Fixed shelf v03_image thresholds (sha256 4a06de0e...). Simulator truth for scoring only.", ""]
    if "c0_vs_original" in o:
        md += [f"C0 through the harness vs the original v0.3 evidence run: {o['c0_vs_original']}", ""]
    md += [f"| | {' | '.join(f'{L} C0 | {L} {c}' for L in LIGHTINGS)} |", "|---|" + "---|" * 4]
    P = o["per_lighting"]
    for key, lbl in (("refused", "refused"), ("present_clear_miss", "present-labelled clear misses"),
                     ("removed_strong_match", "removed-labelled strong matches"),
                     ("diag_removal_yield", "diagnostic removal yield"), ("present_gate_pass", "present gate pass")):
        md.append(f"| {lbl} | " + " | ".join(f"{P[L]['c0'][key]} / {P[L]['denominators'][key]} | "
                                             f"{P[L]['cand'][key]} / {P[L]['denominators'][key]}" for L in LIGHTINGS) + " |")
    md.append("| target-plane error, common accepted (median / p90 / max px) | " + " | ".join(
        f"{fmt(P[L]['geometry_common_target']['c0'])} | {fmt(P[L]['geometry_common_target']['cand'])}" for L in LIGHTINGS) + " |")
    md.append(f"| {'table' if o['environment'] == 'tabletop' else 'front'}-plane error, common accepted | " + " | ".join(
        f"{fmt(P[L]['geometry_common_other']['c0'])} | {fmt(P[L]['geometry_common_other']['cand'])}" for L in LIGHTINGS) + " |")
    md.append("| silhouette IoU, common accepted (median / p90 / max) | " + " | ".join(
        f"{fmt(P[L]['silhouette_common']['c0'])} | {fmt(P[L]['silhouette_common']['cand'])}" for L in LIGHTINGS) + " |")
    md.append("| registration s/view (median / p90 / max) | " + " | ".join(
        f"{fmt(P[L]['c0']['registration_s'])} | {fmt(P[L]['cand']['registration_s'])}" for L in LIGHTINGS) + " |")
    md.append("| complete evidence s/view (median / p90 / max) | " + " | ".join(
        f"{fmt(P[L]['c0']['evidence_s'])} | {fmt(P[L]['cand']['evidence_s'])}" for L in LIGHTINGS) + " |")
    md += ["", "Criteria (gating only for the random arrangements):", ""]
    for L in LIGHTINGS:
        x = P[L]
        md.append(f"- {L}: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in x["criteria"].items()) +
                  f"; common accepted {x['common_accepted']}, newly accepted {len(x['newly_accepted'])}, newly refused "
                  f"{len(x['newly_refused'])}; C0-accepted max target error {x['c0_accepted_max_err_target']:.2f} px; "
                  f"timing ratio {c}/C0 registration median {x['timing_ratio']['registration_s']['median']:.2f}, "
                  f"evidence median {x['timing_ratio']['evidence_s']['median']:.2f}")
    md += ["", f"**Overall: {'PASS' if o['pass'] else 'FAIL'}** (all criteria, both lightings)" if o["gating"] else
           "", ""]
    for L in LIGHTINGS:
        if P[L]["newly_accepted_err_target"]:
            md.append(f"Newly accepted, {L} (view, stop, target-plane error px, silhouette IoU): " +
                      "; ".join(f"{a} {b} {e:.2f} {i:.3f}" for a, b, e, i in P[L]["newly_accepted_err_target"]))
        if P[L]["newly_refused"]:
            md.append(f"Newly refused, {L}: " + "; ".join(f"{a} {b} ({why})" for a, b, why in P[L]["newly_refused"]))
    for tag in ("c0", "cand"):
        b = o.get(f"batch_{tag}")
        if b:
            md.append(f"Batch {b['candidate']}: {b['views']} views, {b['workers']} workers, torch/OMP {b['torch_threads']} "
                      f"thread, cv2 {b['cv2_threads']}, host {b['host']} ({b['cpu_count']} CPUs, {b['gpu']}), wall "
                      f"{b['wall_s']} s, {b['views_per_hour']} views/hour (wall clock includes model loading)")
    return "\n".join(md) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--c0", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--c0-timing")
    ap.add_argument("--cand-timing")
    ap.add_argument("--original-c0")
    ap.add_argument("--design", action="store_true", help="design cases: reported, not gating")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    o = score(args)
    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_{Path(args.jobs).stem.removeprefix('jobs_')}"
    (d / f"{stem}.json").write_text(json.dumps(o, indent=1))
    (d / f"{stem}.md").write_text(markdown(o))
    print(f"wrote {d / stem}.md; " + (f"overall {'PASS' if o['pass'] else 'FAIL'}" if o["gating"] else "design cases: not gating"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
