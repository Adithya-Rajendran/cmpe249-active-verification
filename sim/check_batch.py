"""Checks on a rendered sim set (sim/isaac_batch.py output), from the files only: frame margins, tag detection and
registration of every query view onto its reference, revisit deltas, visible fractions and patterns, how far the
visible-phase queries differ from their references, and render seconds. numpy and cv2; runs on the Jetson or the GB10.

Why: the checkpoint-2 generator changes (quota slots, one-sided M16, revisit noise, labels from perturbed renders) are
verified on a small batch before the full dev batch is rendered. Every number printed here is computed from the rendered
folders; the full result is written to --out as JSON so a note can cite the file.

    uv run --no-project --python 3.13.5 --with numpy==2.5.3 --with opencv-python-headless==5.0.0.93 \\
        python -m sim.check_batch --root /mnt/bench-data/.../sim/shelf_dev_v1_check [--out ROOT/check/check.json]

Registration is capture/tags.py align() query -> L1 reference, the same call perception/evidence.py makes (tag_align). Its
quality is also measured at the target: the visible-phase target mask is warped through H into reference pixels and
compared with the reference mask (IoU and centroid offset). Tags are 4 ArUco markers in the shelf's front plane; the
target stands 5-30 cm behind it, so a camera move of a few mm leaves a small parallax residual at the target.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

from sim import isaac_batch as B
from sim import rig as R

REPO = R.REPO_ROOT
PHASES = ("reference", "visible", "occluded", "removed")
CONDITIONS = ("visible", "occluded", "removed")
LIGHTINGS = ("L1", "L2")
N_TAGS = 4
LEFTOVER = re.compile(r"\.(partial|old\d*)$")     # <ARR>.partial (interrupted render), <ARR>.old<pid> (crash while installing)


def _tags_module():
    d = str(REPO / "capture")
    if d not in sys.path:
        sys.path.insert(0, d)
    import tags

    return tags


def border_margin(corners: dict, width: int, height: int) -> float | None:
    """Smallest distance (px) from any detected tag corner to the frame edge, or None without tags."""
    if not corners:
        return None
    pts = np.concatenate(list(corners.values()))
    return float(min(pts[:, 0].min(), pts[:, 1].min(), width - pts[:, 0].max(), height - pts[:, 1].max()))


def mask_stats(m: np.ndarray) -> dict | None:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return {"n": int(len(xs)), "cx": float(xs.mean() + 0.5), "cy": float(ys.mean() + 0.5)}


def warped_mask_agreement(ref_mask: np.ndarray, qry_mask: np.ndarray, H: np.ndarray) -> dict:
    """IoU and centroid offset (px, reference coordinates) between the reference mask and the query mask warped by H."""
    h, w = ref_mask.shape
    warped = cv2.warpPerspective(qry_mask.astype(np.uint8), H, (w, h), flags=cv2.INTER_NEAREST) > 0
    inter, union = int(np.count_nonzero(ref_mask & warped)), int(np.count_nonzero(ref_mask | warped))
    a, b = mask_stats(ref_mask), mask_stats(warped)
    off = None if a is None or b is None else float(np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]))
    raw = int(np.count_nonzero(ref_mask & qry_mask)) / max(1, int(np.count_nonzero(ref_mask | qry_mask)))
    return {"iou_after_registration": None if union == 0 else inter / union, "iou_without_registration": raw,
            "centroid_offset_px": off}


def summarize(values: list[float]) -> dict | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v), "mean": statistics.fmean(v)}


def read_png(p: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(str(p), flags)
    if img is None:
        raise FileNotFoundError(p)
    return img


def check_arrangement(folder: Path, tags) -> dict:
    arr = json.loads((folder / "arrangement.json").read_text())
    lab = json.loads((folder / "labels.json").read_text())
    render = json.loads((folder / "render.json").read_text())
    if arr["arrangement_id"] != folder.name or lab.get("arrangement_id") != folder.name:
        raise ValueError(f"{folder}: arrangement.json has id {arr['arrangement_id']!r} and labels.json {lab.get('arrangement_id')!r}; "
                         "a folder must be named after its arrangement (the same check as sim/episodes.py load_rendered)")
    stops = lab["stops"]
    out: dict = {"arrangement_id": arr["arrangement_id"], "seed": arr["seed"], "family": arr["occluder_family"],
                 "m16_side": arr.get("m16_side"), "design_case": arr.get("design_case"), "target": arr["target_id"],
                 "slot": (arr.get("slot") or {}).get("index"), "render_s": render["timings"]["total_s"],
                 "render_steps": render["timings"]["steps"]}
    # revisit deltas and margins from the record
    out["revisit"] = {s: {"delta": v["delta"], "attempt": v["attempt"], "n_rejected": v["n_rejected"],
                          "analytic_min_margin_px": v["margin_px"]["min"], "nominal_min_margin_px": v["nominal_margin_px"]["min"]}
                      for s, v in arr["revisit"]["stops"].items()}
    # per capture: tags detected and their distance to the frame edge, registration onto the L1 reference
    caps, reg = [], []
    refs = {}
    for s in stops:
        for L in LIGHTINGS:
            refs[(s, L)] = read_png(folder / "rgb" / f"reference_{s}_{L}.png")
    for ph in PHASES:
        for s in stops:
            for L in LIGHTINGS:
                img = refs[(s, L)] if ph == "reference" else read_png(folder / "rgb" / f"{ph}_{s}_{L}.png")
                det = tags.detect(img)
                h, w = img.shape[:2]
                caps.append({"phase": ph, "stop": s, "lighting": L, "camera": "nominal" if ph == "reference" else "revisit",
                             "tags": sorted(det), "all_four": set(range(N_TAGS)) <= set(det),
                             "extra_ids": sorted(set(det) - set(range(N_TAGS))),
                             "tag_corner_border_px": border_margin(det, w, h)})
                if ph == "reference":
                    continue
                a = tags.align(refs[(s, "L1")], img)
                rec = {"phase": ph, "stop": s, "lighting": L, "ok": bool(a.ok), "reason": a.reason, "ids_used": a.ids_used,
                       "ids_dropped": a.ids_dropped, "rms_px": a.rms_px, "max_px": a.max_px, "loo_max_px": a.loo_max_px,
                       "dx": a.dx, "dy": a.dy, "rot_deg": a.rot_deg, "scale": a.scale}
                if a.ok:
                    ref_m = read_png(folder / "masks" / f"target_reference_{s}.png", cv2.IMREAD_GRAYSCALE) > 0
                    qry_m = read_png(folder / "masks" / f"target_visible_{s}.png", cv2.IMREAD_GRAYSCALE) > 0   # same pose as every query
                    rec["target"] = warped_mask_agreement(ref_m, qry_m, a.H)
                reg.append(rec)
    out["captures"] = caps
    out["registration"] = reg
    # visible-phase queries vs their references (each lighting against its own reference: same scene and lighting, other pose)
    diffs = []
    for s in stops:
        for L in LIGHTINGS:
            ref = cv2.cvtColor(refs[(s, L)], cv2.COLOR_BGR2GRAY).astype(np.int16)
            qimg = read_png(folder / "rgb" / f"visible_{s}_{L}.png")
            qry = cv2.cvtColor(qimg, cv2.COLOR_BGR2GRAY).astype(np.int16)
            d = np.abs(ref - qry)
            rec = {"stop": s, "lighting": L, "identical": bool(not d.any()), "mean_abs_grey": float(d.mean()),
                   "frac_pixels_differing": float((d > 0).mean()), "frac_pixels_over_8": float((d > 8).mean()),
                   "max_abs_grey": int(d.max())}
            # after tag registration, inside the tag rectangle (where both views are valid): what is left is parallax of
            # the shelf interior, the target's offset and render noise
            al = tags.align(refs[(s, L)], qimg)
            if al.ok:
                h, w = ref.shape
                x0, y0, x1, y1 = tags.default_rect(refs[(s, L)])
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                warped = cv2.warpPerspective(qry.astype(np.float32), al.H, (w, h), flags=cv2.INTER_LINEAR)
                valid = cv2.warpPerspective(np.ones((h, w), np.uint8), al.H, (w, h), flags=cv2.INTER_NEAREST) > 0
                dd = np.abs(ref.astype(np.float32) - warped)[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
                rec["registered_mean_abs_grey_in_tag_rect"] = float(dd.mean())
                rec["registered_frac_over_8_in_tag_rect"] = float((dd > 8).mean())
            diffs.append(rec)
    out["visible_vs_reference"] = diffs
    # labels
    out["labels"] = {"patterns": lab["patterns"], "n_diagnostic": lab["n_diagnostic"], "intended_pattern": lab.get("intended_pattern"),
                     "visible_fraction": lab["visible_fraction"], "region_obstruction": lab["region_obstruction"],
                     "predeclared_indistinguishable": lab["predeclared_indistinguishable"],
                     "indistinguishable_by_masks": lab["indistinguishable_by_masks"],
                     "analytic_placed_coverage": arr["analytic"]["placed_coverage"]}
    # label sanity, recomputed from the stored pixel counts (older renders have no label_sanity block): the visible-phase
    # target pixels must be at least 0.9 x the target-alone pixels at the same revisit pose, at every stop
    out["label_sanity"] = B.label_sanity(stops, lab["target_px"]["visible"], lab["alone_px"]["query"])
    out["label_sanity_recorded"] = lab.get("label_sanity", {}).get("ok")
    return out


def side_of_pattern(pattern: str, start: int = 2) -> str | None:
    """Which side of the start stop the non-C stops of a rendered pattern lie on: 'L', 'R', 'both', or None (none)."""
    left = any(c != "C" for c in pattern[:start])
    right = any(c != "C" for c in pattern[start + 1:])
    return "both" if left and right else ("L" if left else ("R" if right else None))


def list_folders(root: Path) -> tuple[list[Path], list[str]]:
    """(rendered arrangement folders, leftover folder names). A leftover is <ARR>.partial or <ARR>.old<pid>: crash debris
    that sim/episodes.py refuses and that must not be counted here (an .old folder is a complete copy of its arrangement)."""
    folders, leftovers = [], []
    for p in sorted(Path(root).iterdir()):
        if not p.is_dir():
            continue
        if LEFTOVER.search(p.name):
            leftovers.append(p.name)
        elif (p / "labels.json").is_file():
            folders.append(p)
    return folders, leftovers


def check_set(root: Path) -> dict:
    tags = _tags_module()
    folders, leftovers = list_folders(root)
    arrs = [check_arrangement(f, tags) for f in folders]
    caps = [c for a in arrs for c in a["captures"]]
    reg = [r for a in arrs for r in a["registration"]]
    summary = {
        "n_arrangements": len(arrs), "n_captures": len(caps), "n_query_views": len(reg),
        "leftover_folders_skipped": leftovers,
        "label_sanity_flagged": {a["arrangement_id"]: a["label_sanity"]["flagged_stops"] for a in arrs if not a["label_sanity"]["ok"]},
        "extra_tag_ids": [(a["arrangement_id"], c["phase"], c["stop"], c["lighting"], c["extra_ids"]) for a in arrs for c in a["captures"] if c["extra_ids"]],
        "tags_all_four": {"captures": sum(c["all_four"] for c in caps), "of": len(caps),
                          "nominal": [sum(c["all_four"] for c in caps if c["camera"] == "nominal"), sum(c["camera"] == "nominal" for c in caps)],
                          "revisit": [sum(c["all_four"] for c in caps if c["camera"] == "revisit"), sum(c["camera"] == "revisit" for c in caps)]},
        "tag_corner_border_px": {k: summarize([c["tag_corner_border_px"] for c in caps if c["camera"] == k]) for k in ("nominal", "revisit")},
        "registration_ok": {"views": sum(r["ok"] for r in reg), "of": len(reg), "failed": [(a["arrangement_id"], r["phase"], r["stop"], r["lighting"], r["reason"])
                                                                                     for a in arrs for r in a["registration"] if not r["ok"]]},
        "registration_rms_px": summarize([r["rms_px"] for r in reg]),
        "registration_loo_px": summarize([r["loo_max_px"] for r in reg]),
        "target_iou_after_registration": summarize([r["target"]["iou_after_registration"] for r in reg if "target" in r]),
        "target_iou_without_registration": summarize([r["target"]["iou_without_registration"] for r in reg if "target" in r]),
        "target_centroid_offset_px": summarize([r["target"]["centroid_offset_px"] for r in reg if "target" in r]),
        "visible_vs_reference": {"views": sum(len(a["visible_vs_reference"]) for a in arrs),
                                 "identical": sum(d["identical"] for a in arrs for d in a["visible_vs_reference"]),
                                 "mean_abs_grey": summarize([d["mean_abs_grey"] for a in arrs for d in a["visible_vs_reference"]]),
                                 "frac_pixels_differing": summarize([d["frac_pixels_differing"] for a in arrs for d in a["visible_vs_reference"]]),
                                 "registered_mean_abs_grey_in_tag_rect": summarize([d.get("registered_mean_abs_grey_in_tag_rect") for a in arrs
                                                                                    for d in a["visible_vs_reference"]])},
        "render_s": summarize([a["render_s"] for a in arrs]),
        "revisit_attempts": {"stops": sum(len(a["revisit"]) for a in arrs),
                             "redrawn": sum(v["n_rejected"] > 0 for a in arrs for v in a["revisit"].values()),
                             "rejected_total": sum(v["n_rejected"] for a in arrs for v in a["revisit"].values())},
    }
    for k in ("dx_m", "dy_m", "dz_m", "dyaw_deg", "dpitch_deg", "droll_deg"):
        summary.setdefault("revisit_delta", {})[k] = summarize([v["delta"][k] for a in arrs for v in a["revisit"].values()])
    for a in arrs:
        a["rendered_side"] = side_of_pattern(a["labels"]["patterns"]["occluded"])
    return {"root": str(root), "summary": summary, "arrangements": arrs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.check_batch", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None, help="default ROOT/check/check.json")
    a = ap.parse_args(argv)
    res = check_set(Path(a.root))
    out = Path(a.out) if a.out else Path(a.root) / "check" / "check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["summary"], indent=1))
    if res["summary"]["leftover_folders_skipped"]:
        print(f"NOTE: skipped crash leftovers (not checked, and sim/episodes.py refuses them): {res['summary']['leftover_folders_skipped']}")
    flagged = res["summary"]["label_sanity_flagged"]
    for arr_id, stops in flagged.items():
        print(f"WARNING LABEL SANITY {arr_id}: visible-phase target pixels are below {B.LABEL_SANITY_MIN} x the target-alone pixels "
              f"at {stops}: the shelf or a distractor hides the target, which the generator rules prevent; its removed views get a "
              "small region. Do not use this arrangement until it is understood.")
    for arr in res["arrangements"]:
        lb = arr["labels"]
        print(f"{arr['arrangement_id']:7s} {arr['family']:9s} {arr['m16_side'] or '-':1s} patterns ref/vis/occ "
              f"{lb['patterns']['reference']}/{lb['patterns']['visible']}/{lb['patterns']['occluded']} rendered side {arr['rendered_side']} "
              f"n_diag {lb['n_diagnostic']} render {arr['render_s']} s")
    print(f"wrote {out}")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
