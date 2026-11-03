"""Score per-view evidence on public-data episodes against label-derived truth, the way pilot 2 was scored.

Per view, nothing combined across stops: the verdict comes from the frozen module's own placeholder-threshold
flags in diagnostics.jsonl (`flags_placeholder_thresholds`, i.e. perception.evidence.flags with the
verify.schema.Thresholds defaults):
  strong_match -> present; else clear_miss -> removed; else flagged_occluded -> occluded; else unknown.
That is the single-view (k=1) reading pilots 1 and 2 were scored with (data/raw/2026-10-20-pilot2/analysis/
pilot2_table.md). The study's StoppingRule needs clear misses from k=2 distinct stops before REMOVED, so these
are view-level numbers, not decisions of any policy. As a check, the flags are recomputed from the recorded
scores with verify.stopping (mismatches are counted; expected 0).

Metrics, per view, with the EXPERIMENT_PLAN definitions (truth visible and occluded are both 'present'):
  agreement           visible->present, occluded->occluded, removed->removed (pilot 2's '51/52');
  false-removal rate  removed verdicts / views of present targets (visible + occluded);
  removal recall      removed verdicts / views of removed targets;
  coverage            present or removed verdicts / all views (an occluded flag or unknown decides nothing).
Also: confusion matrices overall / per set / per camera stop, the same_footprint occluded subset (with the
detections the target_iou rule treats as the target itself), removed split by box_other_change, quantiles of
match / quality / obstruction per truth label, and a few stratifications.

Inputs: <evidence-dir>/diagnostics.jsonl and evidence_run.json (public_data.run_evidence), and truth.json and
truth_detail.jsonl from the episode directory (default: the directory of the jobs.json that evidence_run.json
names). Refused: truth files whose sha256 differs from the one the run recorded, a view without truth, a
duplicate view, and val views unless --allow-val. Writes <out-dir>/analysis.json and <out-dir>/analysis.md
(out-dir defaults to the evidence dir; existing files are refused unless --force).

Usage (repo root; stdlib and verify/ only, no GPU):
  python -m public_data.score_views --evidence-dir EPISODES/evidence/0.2.1/dev [--episodes DIR] \
      [--out-dir DIR] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verify.schema import Observation, Thresholds
from verify.stopping import StoppingRule

SCHEMA = "cmpe249.ocid-view-scores.v0"
LABELS = ("visible", "occluded", "removed")
VERDICTS = ("present", "occluded", "removed", "unknown")
EXPECTED = {"visible": "present", "occluded": "occluded", "removed": "removed"}
SCORES = ("match_score", "quality", "obstruction")
PARTS = ("novelty", "det_obstruction", "changed", "sharpness")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# Our rig, pilot 2 (2026-10-20, data/raw/2026-10-20-pilot2/analysis/diagnostics.jsonl + truth.json):
# min / max match per truth label (44 visible, 4 occluded, 4 removed views). Quoted for comparison only.
RIG_MATCH = {"visible": (0.8888, 0.9407, 44), "occluded": (0.254, 0.485, 4), "removed": (0.200, 0.276, 4)}
OCCLUDER_BOX_IOU = 0.5     # a detection with this box IoU against a truth occluder's query box is counted as it
AREA_BINS = (("< 2500 px", 0, 2500), ("2500-10000 px", 2500, 10000), (">= 10000 px", 10000, math.inf))
# fill = target mask area in the reference / its box area: how much of the box is the target (the rest is
# background or other objects, which the match score also compares)
FILL_BINS = (("< 0.5", 0.0, 0.5), ("0.5-0.75", 0.5, 0.75), (">= 0.75", 0.75, math.inf))
MAX_MD_ROWS = 60


# ---------------------------------------------------------------- small pieces

def verdict(flags: Mapping[str, bool]) -> str:
    """Pilot 2's single-view verdict from the stopping rule's per-view flags."""
    if flags["strong_match"]:
        return "present"
    if flags["clear_miss"]:
        return "removed"
    if flags["flagged_occluded"]:
        return "occluded"
    return "unknown"


def quantile(xs: Iterable[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default method, Hyndman-Fan type 7)."""
    s = sorted(float(x) for x in xs)
    if not s:
        raise ValueError("quantile of no values")
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summary(xs: Iterable[float]) -> dict[str, Any]:
    v = [float(x) for x in xs]
    if not v:
        return {"n": 0}
    out: dict[str, Any] = {"n": len(v), "min": min(v)}
    out.update({f"p{round(q * 100):02d}": quantile(v, q) for q in QUANTILES})
    out["max"] = max(v)
    return out


def ratio(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": k / n if n else None}


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def verdict_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    c = Counter(r["verdict"] for r in rows)
    return {v: c[v] for v in VERDICTS}


def confusion(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {t: verdict_counts(r for r in rows if r["truth"] == t) for t in LABELS}


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    c = Counter((r["truth"], r["verdict"]) for r in rows)
    nt = Counter(r["truth"] for r in rows)
    n = len(rows)
    return {"n": n, "n_by_truth": {t: nt[t] for t in LABELS},
            "agreement": ratio(sum(c[(t, EXPECTED[t])] for t in LABELS), n),
            "false_removal_rate": ratio(c[("visible", "removed")] + c[("occluded", "removed")],
                                        nt["visible"] + nt["occluded"]),
            "false_removal_rate_visible": ratio(c[("visible", "removed")], nt["visible"]),
            "false_removal_rate_occluded": ratio(c[("occluded", "removed")], nt["occluded"]),
            "removal_recall": ratio(c[("removed", "removed")], nt["removed"]),
            "coverage": ratio(sum(k for (_, v), k in c.items() if v in ("present", "removed")), n),
            "not_unknown": ratio(sum(k for (_, v), k in c.items() if v != "unknown"), n),
            "agreement_by_truth": {t: ratio(c[(t, EXPECTED[t])], nt[t]) for t in LABELS}}


def grouped(rows: Sequence[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, list[dict[str, Any]]]:
    g: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    return dict(sorted(g.items()))


def camera_key(r: Mapping[str, Any]) -> str:
    return f"stop {r['stop']} ({r['camera']})"


def area_bin(box: Sequence[float]) -> str:
    a = (box[2] - box[0]) * (box[3] - box[1])
    return next(name for name, lo, hi in AREA_BINS if lo <= a < hi)


def fill_bin(detail: Mapping[str, Any]) -> str:
    x0, y0, x1, y1 = detail["box"]
    f = detail["area_r"] / ((x1 - x0) * (y1 - y0))
    return next(name for name, lo, hi in FILL_BINS if lo <= f < hi)


def stratum(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"metrics": metrics(rows), "confusion": confusion(rows),
            "removed_match": summary(r["match_score"] for r in rows if r["truth"] == "removed")}


NEW_SHARE_BINS = ("0", "(0, 0.1]", "> 0.1")


def new_share_bin(x: float) -> str:
    return NEW_SHARE_BINS[0] if x == 0 else NEW_SHARE_BINS[1] if x <= 0.1 else NEW_SHARE_BINS[2]


def in_order(g: Mapping[str, Any], order: Iterable[str]) -> dict[str, Any]:
    """g's items in the given key order (bins read low to high), keys absent from g skipped."""
    return {k: g[k] for k in order if k in g}


def target_iou_detections(r: Mapping[str, Any], target_iou: float) -> list[dict[str, Any]]:
    """Detections the frozen obstruction rule drops as 'the target itself' (recorded iou_box >= target_iou)."""
    occ = [o["box_q"] for o in r["detail"].get("occluders", [])]
    return [{"cls": d["cls"], "conf": d["conf"], "iou_box": d["iou_box"], "box": d["box"],
             "max_iou_with_occluder_box": max((iou(d["box"], b) for b in occ), default=None)}
            for d in r["detections"] if d["iou_box"] >= target_iou]


def cue(r: Mapping[str, Any], tau: float) -> str:
    det = r["obstruction_parts"]["det_obstruction"] > tau
    nov = r["obstruction_parts"]["novelty"] > tau
    return "both" if det and nov else "detector only" if det else "novelty only" if nov else "neither"


def quality_limiter(r: Mapping[str, Any]) -> str:
    q = r["quality_parts"]
    return min(("sharpness", "exposure", "valid"), key=lambda k: q[k])


def view_brief(r: Mapping[str, Any], target_iou: float) -> dict[str, Any]:
    d = r["detail"]
    return {"episode_id": r["episode_id"], "truth": r["truth"], "verdict": r["verdict"], "set": r["set"],
            "camera": r["camera"], "match_score": r["match_score"], "quality": r["quality"],
            "obstruction": r["obstruction"], "det_obstruction": r["obstruction_parts"]["det_obstruction"],
            "novelty": r["obstruction_parts"]["novelty"], "changed": r["obstruction_parts"]["changed"],
            "box": d["box"], "same_footprint": d.get("same_footprint"),
            "truth_max_occluder_box_iou": d.get("max_occluder_box_iou"),
            "box_new_object_share": d.get("box_new_object_share"), "n_detections": len(r["detections"]),
            "target_iou_detections": target_iou_detections(r, target_iou)}


# ---------------------------------------------------------------- load and score

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_views(evidence_dir: str | Path, episodes_dir: str | Path | None = None,
               allow_val: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One row per scored view (diagnostics joined with truth and truth_detail) plus provenance."""
    ev_dir = Path(evidence_dir).resolve()
    run_path, diag_path = ev_dir / "evidence_run.json", ev_dir / "diagnostics.jsonl"
    for p in (run_path, diag_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found")
    run = json.loads(run_path.read_text())
    ep_dir = Path(episodes_dir).resolve() if episodes_dir else Path(run["jobs"]).resolve().parent
    truth_path, detail_path = ep_dir / "truth.json", ep_dir / "truth_detail.jsonl"
    for p in (truth_path, detail_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found")
    hashes = {"truth.json": _sha256(truth_path), "truth_detail.jsonl": _sha256(detail_path),
              "diagnostics.jsonl": _sha256(diag_path), "evidence_run.json": _sha256(run_path)}
    checked = {}
    for name, key in (("truth.json", "truth_sha256"), ("truth_detail.jsonl", "truth_detail_sha256")):
        rec = run.get(key)
        if rec is not None and rec != hashes[name]:
            raise ValueError(f"{ep_dir / name} has sha256 {hashes[name]}, but {run_path} recorded {rec}")
        checked[name] = rec is not None
    truth = json.loads(truth_path.read_text())
    details = {d["episode_id"]: d for d in _jsonl(detail_path)}
    rows: list[dict[str, Any]] = []
    for d in _jsonl(diag_path):
        eid = d["episode_id"]
        if eid not in truth or eid not in details:
            raise ValueError(f"{diag_path}: no truth for {eid}")
        det = details[eid]
        if det["label"] != truth[eid]:
            raise ValueError(f"truth.json and truth_detail.jsonl disagree on {eid}")
        if int(det["stop"]) != int(d["stop"]):
            raise ValueError(f"{eid}: stop {d['stop']} in diagnostics, {det['stop']} in truth_detail")
        rows.append({"episode_id": eid, "truth": truth[eid], "verdict": verdict(d["flags_placeholder_thresholds"]),
                     "flags": d["flags_placeholder_thresholds"], "split": det["split"], "set": det["set"],
                     "camera": det["camera"], "stop": int(d["stop"]), "match_score": d["match_score"],
                     "quality": d["quality"], "obstruction": d["obstruction"], "quality_parts": d["quality_parts"],
                     "obstruction_parts": d["obstruction_parts"], "detections": d["detections"],
                     "align_ok": d.get("align_ok"), "detail": det})
    ids = Counter(r["episode_id"] for r in rows)
    dup = sorted(e for e, k in ids.items() if k > 1)
    if dup:
        raise ValueError(f"{diag_path}: {len(dup)} episode(s) appear more than once, e.g. {dup[:3]}")
    splits = Counter(r["split"] for r in rows)
    if splits.get("val") and not allow_val:
        raise ValueError(f"{diag_path} holds {splits['val']} val view(s); val is kept back "
                         f"(pass --allow-val to score it)")
    if run.get("split") is not None and set(splits) != {run["split"]}:
        raise ValueError(f"{run_path} says split {run['split']!r}, but the views are {dict(splits)}")
    in_truth = {e for e, d in details.items() if d["split"] in splits}
    ctx = {"evidence_dir": str(ev_dir), "episodes_dir": str(ep_dir), "run": run, "sha256": hashes,
           "truth_hashes_checked_against_run": checked, "splits": dict(splits),
           "n_split_episodes_in_truth": len(in_truth), "missing_from_evidence": sorted(in_truth - set(ids))}
    return rows, ctx


def score(evidence_dir: str | Path, episodes_dir: str | Path | None = None,
          allow_val: bool = False) -> dict[str, Any]:
    rows, ctx = load_views(evidence_dir, episodes_dir, allow_val)
    run = ctx["run"]
    th = Thresholds()
    rule = StoppingRule(th)
    tau = th.tau_occ
    target_iou = float((run.get("params") or {}).get("target_iou", 0.5))
    flag_mismatch = []
    for r in rows:
        o = Observation(step=0, stop=r["stop"], azimuth_deg=0.0, match_score=r["match_score"],
                        quality=r["quality"], obstruction=r["obstruction"])
        mine = {"strong_match": rule.is_strong_match(o), "clear_miss": rule.is_clear_miss(o),
                "flagged_occluded": rule.is_flagged_occluded(o)}
        if mine != {k: r["flags"][k] for k in mine}:
            flag_mismatch.append(r["episode_id"])
    occ = [r for r in rows if r["truth"] == "occluded"]
    sf = [r for r in occ if r["detail"].get("same_footprint")]
    occ_other = [r for r in occ if not r["detail"].get("same_footprint")]
    sf_briefs = [view_brief(r, target_iou) for r in sf]
    with_t = [b for b in sf_briefs if b["target_iou_detections"]]
    rem = [r for r in rows if r["truth"] == "removed"]
    vis = [r for r in rows if r["truth"] == "visible"]
    unknown = [r for r in rows if r["verdict"] == "unknown"]
    near_boundary = sum(1 for r in rows for d in r["detections"] if abs(d["iou_box"] - target_iou) < 5e-4)

    def tiou_rate(rs: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return ratio(sum(1 for r in rs if target_iou_detections(r, target_iou)), len(rs))

    def removed_group(flag: bool) -> dict[str, Any]:
        g = [r for r in rem if bool(r["detail"].get("box_other_change")) is flag]
        return {"n": len(g), "verdicts": verdict_counts(g),
                "removal_recall": ratio(sum(1 for r in g if r["verdict"] == "removed"), len(g))}

    a: dict[str, Any] = {
        "schema": SCHEMA, "scorer": "public_data.score_views", "scorer_sha256": _sha256(Path(__file__)),
        "evidence_dir": ctx["evidence_dir"], "episodes_dir": ctx["episodes_dir"], "sha256": ctx["sha256"],
        "truth_hashes_checked_against_run": ctx["truth_hashes_checked_against_run"],
        "evidence_run": {k: run.get(k) for k in ("split", "n_jobs", "n_jobs_in_file", "n_jobs_by_split",
                                                  "evidence_version", "evidence_sha256", "loader_version",
                                                  "loader_sha256", "align", "encoder", "detector", "params")},
        "split": run.get("split"), "splits_scored": ctx["splits"], "n_views": len(rows),
        "n_split_episodes_in_truth": ctx["n_split_episodes_in_truth"],
        "missing_from_evidence": ctx["missing_from_evidence"],
        "verdict_rule": "strong_match -> present; else clear_miss -> removed; else flagged_occluded -> "
                        "occluded; else unknown (flags_placeholder_thresholds; single view, k=1)",
        "thresholds": th.to_dict(), "target_iou": target_iou,
        "flag_recompute_mismatches": flag_mismatch,
        "strong_match_and_flagged_occluded": sum(1 for r in rows if r["flags"]["strong_match"]
                                                 and r["flags"]["flagged_occluded"]),
        "align_not_ok": sum(1 for r in rows if r["align_ok"] is False),
        "metrics": {"overall": metrics(rows),
                    "by_set": {k: metrics(v) for k, v in grouped(rows, lambda r: r["set"]).items()},
                    "by_camera_stop": {k: metrics(v) for k, v in grouped(rows, camera_key).items()}},
        "confusion": {"overall": confusion(rows),
                      "by_set": {k: confusion(v) for k, v in grouped(rows, lambda r: r["set"]).items()},
                      "by_camera_stop": {k: confusion(v) for k, v in grouped(rows, camera_key).items()}},
        "same_footprint": {
            "n": len(sf), "verdicts": verdict_counts(sf),
            "with_target_iou_detection": len(with_t),
            "with_target_iou_detection_matching_occluder_box": sum(
                1 for b in with_t if any((t["max_iou_with_occluder_box"] or 0) >= OCCLUDER_BOX_IOU
                                         for t in b["target_iou_detections"])),
            "with_target_iou_detection_closer_to_occluder": sum(
                1 for b in with_t if any((t["max_iou_with_occluder_box"] or 0) > t["iou_box"]
                                         for t in b["target_iou_detections"])),
            "verdicts_with_target_iou_detection": verdict_counts(with_t),
            "verdicts_without_target_iou_detection": verdict_counts(b for b in sf_briefs
                                                                    if not b["target_iou_detections"]),
            "with_any_detection": sum(1 for b in sf_briefs if b["n_detections"]),
            "match_score": summary(r["match_score"] for r in sf),
            "det_obstruction": summary(r["obstruction_parts"]["det_obstruction"] for r in sf),
            "novelty": summary(r["obstruction_parts"]["novelty"] for r in sf),
            "occluder_box_iou_rule": OCCLUDER_BOX_IOU,
            "detections_within_rounding_of_target_iou": near_boundary,
            "episodes": sf_briefs},
        "target_iou_detection_rate": {"visible": tiou_rate(vis), "occluded same_footprint": tiou_rate(sf),
                                      "occluded other": tiou_rate(occ_other), "removed": tiou_rate(rem)},
        "occluded_cues": {"all": dict(sorted(Counter(cue(r, tau) for r in occ).items())),
                          "same_footprint": dict(sorted(Counter(cue(r, tau) for r in sf).items())),
                          "other": dict(sorted(Counter(cue(r, tau) for r in occ_other).items()))},
        "removed_by_box_other_change": {"false": removed_group(False), "true": removed_group(True)},
        "visible_by_box_new_object_share": in_order({k: {"n": len(g), "verdicts": verdict_counts(g)} for k, g in grouped(
            vis, lambda r: new_share_bin(r["detail"].get("box_new_object_share", 0.0))).items()}, NEW_SHARE_BINS),
        "by_target_box_area": in_order({k: stratum(v) for k, v in grouped(rows, lambda r: area_bin(r["detail"]["box"])).items()},
                                       (name for name, _, _ in AREA_BINS)),
        "by_target_fill": in_order({k: stratum(v) for k, v in grouped(rows, lambda r: fill_bin(r["detail"])).items()},
                                   (name for name, _, _ in FILL_BINS)),
        "unknown_ring_kept": summary(r["quality_parts"]["ring_kept"] for r in unknown),
        "unknown_sharpness": summary(r["quality_parts"]["sharpness"] for r in unknown),
        "score_quantiles": {t: {s: summary(r[s] for r in rows if r["truth"] == t) for s in SCORES} for t in LABELS},
        "part_quantiles": {t: {p: summary((r["obstruction_parts"] | r["quality_parts"])[p] for r in rows
                                          if r["truth"] == t) for p in PARTS} for t in LABELS},
        "rig_pilot2_match": {t: {"min": lo, "max": hi, "n": n} for t, (lo, hi, n) in RIG_MATCH.items()},
        "visible_match_below_rig_min": ratio(sum(1 for r in vis if r["match_score"] < RIG_MATCH["visible"][0]),
                                             len(vis)),
        "unknown_quality_limiter": dict(sorted(Counter(quality_limiter(r) for r in unknown).items())),
        "false_removals": [view_brief(r, target_iou) for r in rows
                           if r["truth"] != "removed" and r["verdict"] == "removed"],
        "views": [[r["episode_id"], r["truth"], r["verdict"]] for r in rows],
    }
    return a


# ---------------------------------------------------------------- markdown

def _r(d: Mapping[str, Any]) -> str:
    return f"{d['k']}/{d['n']} ({d['rate']:.3f})" if d["n"] else "0/0"


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _vc(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{v} {counts[v]}" for v in VERDICTS)


def _kv(d: Mapping[str, Any]) -> str:
    return ", ".join(f"{k} {'yes' if v is True else 'no' if v is False else v}" for k, v in d.items()) or "none"


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(str(c) for c in row) + " |" for row in rows)]


def _q_row(name: str, s: Mapping[str, Any]) -> list[str]:
    if not s["n"]:
        return [name, "0", *["n/a"] * 7]
    return [name, str(s["n"]), *(_f(s[k]) for k in ("min", "p05", "p25", "p50", "p75", "p95", "max"))]


def _views_table(briefs: Sequence[Mapping[str, Any]]) -> list[str]:
    def tdets(b: Mapping[str, Any]) -> str:
        return "; ".join(f"{t['cls']} {t['conf']:.2f} iou {t['iou_box']:.2f}"
                         + (f" (occluder iou {t['max_iou_with_occluder_box']:.2f})"
                            if t["max_iou_with_occluder_box"] is not None else "")
                         for t in b["target_iou_detections"]) or "-"
    rows = [[b["episode_id"], b["truth"], b["verdict"], _f(b["match_score"]), _f(b["quality"]),
             _f(b["det_obstruction"]), _f(b["novelty"]), _f(b["changed"]), b["n_detections"], tdets(b)]
            for b in briefs[:MAX_MD_ROWS]]
    out = _table(["episode", "truth", "verdict", "match", "quality", "det obstr", "novelty", "changed",
                  "dets", "target-IoU detections (cls conf iou)"], rows)
    if len(briefs) > MAX_MD_ROWS:
        out.append(f"\n({len(briefs) - MAX_MD_ROWS} more in analysis.json)")
    return out


def render_md(a: Mapping[str, Any]) -> str:
    run, th, m = a["evidence_run"], a["thresholds"], a["metrics"]
    L: list[str] = []
    L += [f"# OCID {a['split']}: per-view scores of frozen evidence {run['evidence_version']}", "",
          "Development data (public OCID, label-derived truth); never the held-out test. Generated by "
          f"`public_data/score_views.py` (sha256 {a['scorer_sha256'][:12]}...); every number below is "
          "computed from the files listed here.", "",
          f"- Evidence: `{a['evidence_dir']}`: split **{a['split']}**, {run['n_jobs']} jobs run "
          f"(jobs.json holds {_kv(run['n_jobs_by_split'] or {})}), evidence {run['evidence_version']} "
          f"(sha256 `{run['evidence_sha256']}`), align {run['align']}.",
          f"- Models: encoder `{run['encoder']}`, detector `{run['detector']}`.",
          f"- Episodes: `{a['episodes_dir']}`, loader {run['loader_version']} (sha256 `{run['loader_sha256']}`). "
          f"truth.json `{a['sha256']['truth.json'][:12]}...`, truth_detail.jsonl "
          f"`{a['sha256']['truth_detail.jsonl'][:12]}...` (equal to the run's recorded sha256: "
          f"{_kv(a['truth_hashes_checked_against_run'])}). diagnostics.jsonl "
          f"`{a['sha256']['diagnostics.jsonl'][:12]}...`.",
          f"- Views scored: {a['n_views']} ({_kv(a['splits_scored'])}); {a['split']} episodes in truth: "
          f"{a['n_split_episodes_in_truth']}; missing from the evidence: {len(a['missing_from_evidence'])}.",
          f"- Verdict per view (pilot 2 rule): {a['verdict_rule']}. Placeholder thresholds: tau_match "
          f"{th['tau_match']}, tau_q {th['tau_q']}, tau_occ {th['tau_occ']}. Flags recomputed from the recorded "
          f"scores with verify.stopping: {len(a['flag_recompute_mismatches'])} mismatches. Views with both "
          f"strong_match and flagged_occluded (counted present): {a['strong_match_and_flagged_occluded']}. "
          f"Views with align_ok false: {a['align_not_ok']}.",
          "- Metrics (EXPERIMENT_PLAN definitions, per view): agreement = visible->present, "
          "occluded->occluded, removed->removed; false-removal rate = removed verdicts / views of present "
          "targets (visible + occluded); removal recall = removed verdicts / removed views; coverage = present "
          "or removed verdicts / all views (occluded flag and unknown decide nothing); 'not unknown' also "
          "counts occluded flags.", ""]

    L += ["## Headline metrics", ""]
    groups = [("overall", m["overall"]), *m["by_set"].items(), *m["by_camera_stop"].items()]
    L += _table(["group", "n (vis/occ/rem)", "agreement", "false removal (present)", "false removal visible",
                 "false removal occluded", "removal recall", "coverage", "not unknown"],
                [[g, f"{x['n']} ({x['n_by_truth']['visible']}/{x['n_by_truth']['occluded']}/"
                     f"{x['n_by_truth']['removed']})", _r(x["agreement"]), _r(x["false_removal_rate"]),
                  _r(x["false_removal_rate_visible"]), _r(x["false_removal_rate_occluded"]),
                  _r(x["removal_recall"]), _r(x["coverage"]), _r(x["not_unknown"])] for g, x in groups])
    L += ["", "## Confusion (truth x verdict)", ""]
    cgroups = [("overall", a["confusion"]["overall"]), *a["confusion"]["by_set"].items(),
               *a["confusion"]["by_camera_stop"].items()]
    L += _table(["group", "truth \\ verdict", *VERDICTS, "n"],
                [[g, t, *(cm[t][v] for v in VERDICTS), sum(cm[t].values())] for g, cm in cgroups for t in LABELS])

    sf = a["same_footprint"]
    L += ["", "## Occluded, same_footprint subset", "",
          f"Occluded views whose truth occluder's box has IoU >= 0.5 with the target box (the pilot 2 clock case): "
          f"**{sf['n']}**. Verdicts: {_vc(sf['verdicts'])}.", "",
          f"- With at least one detection whose recorded IoU with the target box is >= target_iou "
          f"({a['target_iou']}), i.e. a detection the obstruction rule drops as the target itself: "
          f"**{sf['with_target_iou_detection']}** of {sf['n']}. Of those, the dropped detection also has box IoU "
          f">= {sf['occluder_box_iou_rule']} with a truth occluder's query box: "
          f"{sf['with_target_iou_detection_matching_occluder_box']}; and overlaps that occluder's box more than "
          f"the target box: {sf['with_target_iou_detection_closer_to_occluder']}.",
          f"- Verdicts with such a detection: {_vc(sf['verdicts_with_target_iou_detection'])}; without: "
          f"{_vc(sf['verdicts_without_target_iou_detection'])}. Views with any detection at all: "
          f"{sf['with_any_detection']}.",
          f"- Recorded IoUs within rounding (5e-4) of target_iou, over all views: "
          f"{sf['detections_within_rounding_of_target_iou']} (diagnostics round iou_box to 3 decimals).", ""]
    L += _table(["score", "n", "min", "p05", "p25", "p50", "p75", "p95", "max"],
                [_q_row(k, sf[k]) for k in ("match_score", "det_obstruction", "novelty")])
    L += ["", "Share of views with a target-IoU detection, by truth:", ""]
    L += _table(["truth", "views with a detection IoU >= target_iou"],
                [[k, _r(v)] for k, v in a["target_iou_detection_rate"].items()])
    L += ["", f"Which obstruction cue exceeded tau_occ ({th['tau_occ']}) on occluded views:", ""]
    cues = ("detector only", "novelty only", "both", "neither")
    L += _table(["occluded subset", *cues], [[k, *(v.get(c, 0) for c in cues)] for k, v in a["occluded_cues"].items()])
    L += ["", "Same-footprint views:", ""]
    L += _views_table(sf["episodes"])

    L += ["", "## Removed, split by box_other_change", "",
          "box_other_change: objects placed after the query frame cover > 5 % of the target box in the reference "
          "(the time-reversed query also lacks them).", ""]
    L += _table(["box_other_change", "n", *VERDICTS, "removal recall"],
                [[k, v["n"], *(v["verdicts"][x] for x in VERDICTS), _r(v["removal_recall"])]
                 for k, v in a["removed_by_box_other_change"].items()])

    L += ["", "## Score quantiles per truth label", ""]
    L += _table(["truth", "score", "n", "min", "p05", "p25", "p50", "p75", "p95", "max"],
                [[t, *_q_row(s, a["score_quantiles"][t][s])] for t in LABELS for s in SCORES])
    rig = a["rig_pilot2_match"]
    L += ["", "Rig comparison (pilot 2, our shelf, same frozen code): match " + "; ".join(
        f"{t} {_f(v['min'])}-{_f(v['max'])} (n={v['n']})" for t, v in rig.items()) + ". "
        f"OCID {a['split']} visible views with match below the rig's visible minimum ({_f(rig['visible']['min'])}): "
        f"{_r(a['visible_match_below_rig_min'])}.", "",
        "Parts (novelty and det_obstruction are the two obstruction cues, obstruction = max of them; changed = "
        "share of the box whose pixels changed; sharpness is a quality part):", ""]
    L += _table(["truth", "part", "n", "min", "p05", "p25", "p50", "p75", "p95", "max"],
                [[t, *_q_row(p, a["part_quantiles"][t][p])] for t in LABELS for p in PARTS])

    L += ["", "## Stratifications", "", "Visible views by box_new_object_share (share of the box covered in the "
          "query by objects placed after the reference; truth stays visible):", ""]
    L += _table(["box_new_object_share", "n", *VERDICTS],
                [[k, v["n"], *(v["verdicts"][x] for x in VERDICTS)]
                 for k, v in a["visible_by_box_new_object_share"].items()])
    def strata(title: str, key: str, label: str) -> list[str]:
        out = ["", title, ""]
        out += _table([label, "n (vis/occ/rem)", "visible->present", "occluded->occluded", "removed->removed",
                       "removed->present", "removed match p50", "false removal (present)", "coverage"],
                      [[k, f"{x['metrics']['n']} ({x['metrics']['n_by_truth']['visible']}/"
                           f"{x['metrics']['n_by_truth']['occluded']}/{x['metrics']['n_by_truth']['removed']})",
                        _r(x["metrics"]["agreement_by_truth"]["visible"]),
                        _r(x["metrics"]["agreement_by_truth"]["occluded"]),
                        _r(x["metrics"]["agreement_by_truth"]["removed"]),
                        _r(ratio(x["confusion"]["removed"]["present"], x["metrics"]["n_by_truth"]["removed"])),
                        _f(x["removed_match"].get("p50")), _r(x["metrics"]["false_removal_rate"]),
                        _r(x["metrics"]["coverage"])]
                       for k, x in a[key].items()])
        return out

    L += strata("By target box area:", "by_target_box_area", "box area")
    L += strata("By target fill (target mask area in the reference / box area; the rest of the box is background "
                "or other objects, which the match score compares too):", "by_target_fill", "fill")
    L += ["", f"Unknown verdicts (match < tau_match, obstruction <= tau_occ, quality < tau_q), by the quality part "
          f"that is lowest: {_kv(a['unknown_quality_limiter'])}. Their sharpness: median "
          f"{_f(a['unknown_sharpness'].get('p50'))}, max {_f(a['unknown_sharpness'].get('max'))}; share of the "
          f"sharpness ring kept after leaving out changed pixels: min {_f(a['unknown_ring_kept'].get('min'))}, "
          f"median {_f(a['unknown_ring_kept'].get('p50'))}, max {_f(a['unknown_ring_kept'].get('max'))} "
          f"(1.0 = nothing left out, or the full-ring fallback).", ""]

    L += [f"## False removals ({len(a['false_removals'])})", "",
          "Present targets (truth visible or occluded) given a removed verdict:", ""]
    L += _views_table(a["false_removals"]) if a["false_removals"] else ["(none)"]
    L += ["", "## Limits", "",
          "- Per view with k=1, as pilots 1 and 2: the study's rule needs clear misses from two distinct stops "
          "before REMOVED. OCID's top and bottom cameras are two stops of one arrangement, but they are scored "
          "separately here.",
          "- Thresholds are placeholders, not selected on this data. Truth comes from OCID's visible (modal) label "
          "masks; removals are simulated by time reversal (see public_data/README.md).", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- cli

def write(a: Mapping[str, Any], out_dir: str | Path, force: bool = False) -> tuple[Path, Path]:
    out = Path(out_dir)
    paths = (out / "analysis.json", out / "analysis.md")
    if not force and any(p.exists() for p in paths):
        raise FileExistsError(f"{out} already holds analysis.json / analysis.md; use force to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(a, indent=1) + "\n")
    paths[1].write_text(render_md(a))
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="public_data.score_views", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence-dir", required=True, help="holds diagnostics.jsonl and evidence_run.json")
    ap.add_argument("--episodes", default=None,
                    help="holds truth.json and truth_detail.jsonl (default: dir of the jobs.json the run names)")
    ap.add_argument("--out-dir", default=None, help="default: --evidence-dir")
    ap.add_argument("--allow-val", action="store_true", help="score val views (refused otherwise)")
    ap.add_argument("--force", action="store_true", help="overwrite existing analysis.json / analysis.md")
    args = ap.parse_args(argv)
    out = Path(args.out_dir or args.evidence_dir)
    if not args.force and any((out / n).exists() for n in ("analysis.json", "analysis.md")):
        print(f"{out} already holds analysis.json / analysis.md; pass --force to overwrite", file=sys.stderr)
        return 2
    try:
        a = score(args.evidence_dir, args.episodes, allow_val=args.allow_val)
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 2
    paths = write(a, out, force=args.force)
    o = a["metrics"]["overall"]
    print(json.dumps({"written": [str(p) for p in paths], "split": a["split"], "n_views": a["n_views"],
                      "agreement": o["agreement"], "false_removal_rate": o["false_removal_rate"],
                      "removal_recall": o["removal_recall"], "coverage": o["coverage"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
