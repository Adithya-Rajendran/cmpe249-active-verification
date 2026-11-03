"""Rendered sim arrangements -> evidence annotations and jobs, the verify manifest, truth and view labels.

Why: the next stage runs perception/evidence.py (frozen v0.2.1) on every query view and replays S/F/R/G through
verify/. Both read fixed formats, so this step writes exactly those (no conversion later):
  - one annotation (cmpe249.evidence-annotation.v0) per (arrangement, stop, reference lighting): the reference image
    and the target's box at that stop, from that lighting's reference mask (reference pixels, continuous coordinates);
  - jobs (cmpe249.evidence-jobs.v0) for every query view, visible / occluded / removed x L1 / L2 x 5 stops, each
    against its stop's reference:
      jobs_refL1.json  PRIMARY (docs/SIMULATION_STUDY.md, "Reference pairing", 2026-11-03): every query, L1 and L2,
                       against the L1 reference at its stop (memory recorded once; the lighting may differ at revisit);
      jobs_same.json   SECONDARY: each query against the reference at its own lighting. A separate run reported
                       separately, never a relabelling of the primary (the reference enters the evidence scores, so it
                       can change decisions and the views a policy requests);
    episode_id = verify.schema.episode_id(arrangement, condition, lighting), so results can be reported per lighting;
    stop = index into the positions file;
  - manifest.json via verify.manifest.save_manifest (split dev; indistinguishable = the arrangement's predeclared
    flag, i.e. the IND28 occluder family);
  - truth.json (scoring only), view_labels.json (exact per-view visible fractions and diagnostic labels from the
    masks, for development analysis only: never a policy input), summary.md, set.json (hashes);
  - optional contact sheets (cv2) per arrangement and lighting, and a crop sheet around the target.

Paths in annotations and jobs are absolute: the share (/mnt/bench-data/...) has the same path on the Jetson and the
GB10. Only dev arrangements are accepted (integer seeds 0-9999 or the design seeds), as in sim/arrange.py.

    python -m sim.episodes --root /mnt/bench-data/datasets/cmpe249-active-verification/sim/shelf_dev_v0 \\
        [--ref-lighting same L1] [--contact-sheets] [--positions configs/sim/positions_shelf_A.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from sim import arrange as A
from verify.schema import episode_id

ANNOTATION_SCHEMA = "cmpe249.evidence-annotation.v0"   # perception/evidence.py (frozen; not imported: it needs cv2)
JOBS_SCHEMA = "cmpe249.evidence-jobs.v0"
SET_SCHEMA = "cmpe249.sim-set.v0"
VIEW_LABELS_SCHEMA = "cmpe249.sim-view-labels-set.v0"
CONDITIONS = ("visible", "occluded", "removed")
REF_LIGHTINGS = ("L1", "same")
JOBS_FILES = {"L1": "jobs_refL1.json", "same": "jobs_same.json"}
JOBS_ROLE = {"L1": "primary", "same": "secondary"}
THIS_FILE = Path(__file__).resolve()


def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_dev(arr: Mapping) -> None:
    """Integer seeds must be dev seeds (sim/arrange.py check_dev_seed); string seeds must be design seeds."""
    if arr.get("split") != "dev":
        raise A.SeedError(f"{arr.get('arrangement_id')}: split {arr.get('split')!r} (dev only)")
    seed = arr.get("seed")
    if isinstance(seed, str):
        if arr.get("kind") != "design" or not seed.startswith("design-"):
            raise A.SeedError(f"{arr.get('arrangement_id')}: seed {seed!r} is not a design seed")
    else:
        A.check_dev_seed(seed)


def load_rendered(root: str | Path) -> list[tuple[dict, dict, Path]]:
    """(arrangement, labels, folder) for every rendered arrangement under root, sorted by id."""
    out = []
    for d in sorted(Path(root).iterdir()):
        if d.is_dir() and (d / "labels.json").is_file() and (d / "arrangement.json").is_file():
            arr = json.loads((d / "arrangement.json").read_text())
            lab = json.loads((d / "labels.json").read_text())
            check_dev(arr)
            if lab["arrangement_id"] != arr["arrangement_id"] or d.name != arr["arrangement_id"]:
                raise ValueError(f"{d}: arrangement id mismatch")
            out.append((arr, lab, d.resolve()))
    return out


def rgb_path(folder: Path, phase: str, stop: str, lighting: str) -> Path:
    return folder / "rgb" / f"{phase}_{stop}_{lighting}.png"


def annotation_doc(arr: Mapping, labels: Mapping, folder: Path, stop_index: int, lighting: str) -> dict:
    stop = labels["stops"][stop_index]
    box = labels["reference_box"][lighting][stop]
    if box is None:
        raise ValueError(f"{arr['arrangement_id']} {stop} {lighting}: the target has no pixels in the reference")
    prov = arr.get("provenance", {})
    return {"schema": ANNOTATION_SCHEMA,
            "reference": str(rgb_path(folder, "reference", stop, lighting)),
            "stop": int(stop_index),
            "targets": {arr["target_id"]: {"box": [int(v) for v in box],
                                           "where": f"{stop}, compartment #3, from the {lighting} reference instance mask"}},
            "note": "SIMULATION (Isaac Sim render). Box = bounding box of the target's instance mask in this lighting's "
                    "reference render, continuous pixel coordinates (pixel i covers [i, i + 1)). Written by sim/episodes.py "
                    "before any evidence run.",
            "simulation": True, "arrangement_id": arr["arrangement_id"], "stop_name": stop, "lighting": lighting,
            "provenance": {"generator_sha256": prov.get("generator", {}).get("sha256"),
                           "config_sha256": prov.get("config", {}).get("sha256"),
                           "arrangement_sha256": arr.get("arrangement_sha256"),
                           "labels_sha256": labels.get("_file_sha256"), "episodes_py_sha256": sha256_file(THIS_FILE)}}


def annotation_path(folder: Path, stop: str, lighting: str) -> Path:
    return folder / "annotations" / f"ann_{stop}_{lighting}.json"


def jobs_for(arr: Mapping, labels: Mapping, folder: Path, ref_lighting: str) -> list[dict]:
    """Every query view of one arrangement: condition x lighting x stop, against its stop's reference annotation."""
    if ref_lighting not in REF_LIGHTINGS:
        raise ValueError(f"ref_lighting must be one of {REF_LIGHTINGS}")
    jobs = []
    for cond in CONDITIONS:
        for lighting in labels["lightings"]:
            ref_l = lighting if ref_lighting == "same" else "L1"
            for i, stop in enumerate(labels["stops"]):
                jobs.append({"episode_id": episode_id(arr["arrangement_id"], cond, lighting), "stop": i,
                             "target": arr["target_id"], "annotation": str(annotation_path(folder, stop, ref_l)),
                             "query": str(rgb_path(folder, cond, stop, lighting))})
    return jobs


def jobs_doc(rendered: Sequence[tuple[dict, dict, Path]], ref_lighting: str, provenance: Mapping) -> dict:
    jobs = [j for arr, lab, d in rendered for j in jobs_for(arr, lab, d, ref_lighting)]
    note = ("PRIMARY run (docs/SIMULATION_STUDY.md, Reference pairing): every query, L1 and L2, against the L1 reference "
            "at its stop. Report L1 and L2 queries separately (the lighting is in every episode_id)."
            if ref_lighting == "L1" else
            "SECONDARY run: each query against the reference at its own lighting (L1-L1, L2-L2). A separate run, reported "
            "separately; never a relabelling of the primary run (jobs_refL1.json).")
    return {"schema": JOBS_SCHEMA, "simulation": True, "role": JOBS_ROLE[ref_lighting], "ref_lighting": ref_lighting,
            "note": "SIMULATION. " + note, "provenance": dict(provenance), "jobs": jobs}


def manifest_records(rendered: Sequence[tuple[dict, dict, Path]]):
    from verify.manifest import ArrangementRecord

    return [ArrangementRecord(arr["arrangement_id"], arr["target_id"], "dev", tuple(lab["lightings"]),
                              bool(arr.get("predeclared_indistinguishable", False))) for arr, lab, _ in rendered]


def truth_doc(rendered: Sequence[tuple[dict, dict, Path]]) -> dict:
    states = {}
    for arr, lab, _ in rendered:
        for cond in CONDITIONS:
            for lighting in lab["lightings"]:
                states[episode_id(arr["arrangement_id"], cond, lighting)] = lab["truth"][cond]
    return {"schema": "cmpe249.sim-truth.v0", "simulation": True, "scoring_only": True, "states": states,
            "predeclared_indistinguishable": sorted(a["arrangement_id"] for a, _, _ in rendered
                                                    if a.get("predeclared_indistinguishable"))}


def view_labels_doc(rendered: Sequence[tuple[dict, dict, Path]]) -> dict:
    eps = {}
    for arr, lab, _ in rendered:
        for cond in CONDITIONS:
            src = {"visible": "visible", "occluded": "occluded", "removed": None}[cond]
            for lighting in lab["lightings"]:
                eps[episode_id(arr["arrangement_id"], cond, lighting)] = {
                    "arrangement_id": arr["arrangement_id"], "condition": cond, "lighting": lighting,
                    "truth": lab["truth"][cond],
                    "stops": [{"stop": i, "name": s,
                               "visible_fraction": None if src is None else lab["visible_fraction"][src][s],
                               "occluder_share_of_reference_region": (lab["reference_region_obstruction"][cond][s]
                                                                      if cond in ("occluded", "removed") else None),
                               "diagnostic": bool(lab["diagnostic"][cond][s])} for i, s in enumerate(lab["stops"])]}
    return {"schema": VIEW_LABELS_SCHEMA, "simulation": True,
            "note": "Exact labels from the renderer's instance masks, for development analysis and scoring only; never a "
                    "policy input (docs/EXPERIMENT_PLAN.md: visibility annotations are stored separately from policy inputs).",
            "rules": rendered[0][1]["rules"] if rendered else {}, "episodes": eps}


def summary_md(rendered: Sequence[tuple[dict, dict, Path]]) -> str:
    stops = rendered[0][1]["stops"] if rendered else []
    head = ["| Arrangement | Seed | Target | Occluder | Occluded visible fraction " + " / ".join(stops)
            + " | Pattern | Intended | Diagnostic views (visible / occluded / removed) | Ref. vf min | Removed: occluder share "
            + " / ".join(stops) + " |", "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for arr, lab, _ in rendered:
        o = arr["objects"]["occluder"]
        occ = lab["visible_fraction"]["occluded"]
        ref = [v for v in lab["visible_fraction"]["reference"].values() if v is not None]
        rem = lab["reference_region_obstruction"]["removed"]
        n = lab["n_diagnostic"]
        rows.append(f"| {arr['arrangement_id']} | {arr['seed']} | {arr['target_id']} | {o['family']}"
                    f"{'' if o['kind'] == 'box' else ' (' + o['label'] + ')'} | "
                    + " / ".join("-" if occ[s] is None else f"{occ[s]:.3f}" for s in stops)
                    + f" | {lab['patterns']['occluded']} | {lab.get('intended_pattern') or ''} | "
                    f"{n['visible']} / {n['occluded']} / {n['removed']} | {min(ref):.4f} | "
                    + " / ".join("-" if rem[s] is None else f"{rem[s]:.3f}" for s in stops) + " |")
    return "\n".join(["SIMULATION (Isaac Sim renders; labels from instance masks). Diagnostic: present views with visible "
                      "fraction >= 0.90; removed views where the occluder covers < 10 % of the target's reference-mask "
                      "pixels.", "", *head, *rows, ""])


def contact_sheets(folder: Path, labels: Mapping, tile_w: int = 460) -> list[Path]:
    """Full-frame sheets (rows: phases, columns: stops) per lighting, and a crop sheet around the target (L1)."""
    import cv2
    import numpy as np

    stops = labels["stops"]
    phases = ("reference", "visible", "occluded", "removed")
    out = []

    def tag(img, text, scale):
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (0, 0), (w + 8, h + base + 8), (255, 255, 255), -1)
        cv2.putText(img, text, (4, h + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)

    def outline(img, mask, colour, scale):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = [(c * scale).astype(np.int32) for c in cs]
        cv2.drawContours(img, cs, -1, colour, 2)

    def label_text(ph, s):
        if ph == "removed":
            v = labels["reference_region_obstruction"]["removed"][s]
            return f"{ph} {s} occ {v:.2f} {'D' if labels['diagnostic']['removed'][s] else 'nd'}"
        v = labels["visible_fraction"][ph][s]
        d = labels["diagnostic"].get(ph, {}).get(s)
        return f"{ph} {s} vf {v:.2f}" + ("" if d is None else (" D" if d else " nd"))

    for lighting in labels["lightings"]:
        rows = []
        for ph in phases:
            tiles = []
            for s in stops:
                img = cv2.imread(str(rgb_path(folder, ph, s, lighting)))
                sc = tile_w / img.shape[1]
                t = cv2.resize(img, (tile_w, round(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                alone = cv2.imread(str(folder / "masks" / f"target_alone_{s}.png"), cv2.IMREAD_GRAYSCALE) > 0
                cls = cv2.imread(str(folder / "masks" / f"classes_{ph}_{s}.png"), cv2.IMREAD_GRAYSCALE)
                outline(t, alone, (0, 200, 0), sc)
                if ph in ("occluded", "removed"):
                    outline(t, cls == 2, (0, 0, 230), sc)
                tag(t, label_text(ph, s), 0.55)
                tiles.append(t)
            rows.append(np.hstack(tiles))
        sheet = np.vstack(rows)
        bar = np.full((40, sheet.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"SIMULATION {labels['arrangement_id']} seed {labels['seed']} target {labels['target_id']} "
                    f"occluder {labels['occluder_family']} {lighting}; green = target (alone mask), red = occluder",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        p = folder / f"contact_{lighting}.jpg"
        cv2.imwrite(str(p), np.vstack([bar, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(p)
    # crops around the target at every stop (L1): reference / occluded / removed
    L = labels["lightings"][0]
    rows = []
    for ph in ("reference", "occluded", "removed"):
        tiles = []
        for s in stops:
            img = cv2.imread(str(rgb_path(folder, ph, s, L)))
            x0, y0, x1, y1 = labels["reference_box"][L][s]
            w, h = x1 - x0, y1 - y0
            g = max(w, h)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            a, b = int(max(0, cx - 1.5 * g)), int(max(0, cy - 1.5 * g))
            c, d = int(min(img.shape[1], cx + 1.5 * g)), int(min(img.shape[0], cy + 1.5 * g))
            crop = cv2.resize(img[b:d, a:c], (300, 300), interpolation=cv2.INTER_AREA)
            tag(crop, label_text(ph, s), 0.45)
            tiles.append(crop)
        rows.append(np.hstack(tiles))
    p = folder / f"crops_{L}.jpg"
    cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    out.append(p)
    return out


def build(root: str | Path, ref_lightings: Sequence[str] = REF_LIGHTINGS, positions: str | Path | None = None,
          sheets: bool = False) -> dict:
    from verify.manifest import save_manifest

    root = Path(root).resolve()
    rendered = load_rendered(root)
    if not rendered:
        raise ValueError(f"no rendered arrangements under {root}")
    for arr, lab, d in rendered:
        lab["_file_sha256"] = sha256_file(d / "labels.json")
        for i, stop in enumerate(lab["stops"]):
            for lighting in lab["lightings"]:
                p = annotation_path(d, stop, lighting)
                p.parent.mkdir(exist_ok=True)
                p.write_text(json.dumps(annotation_doc(arr, lab, d, i, lighting), indent=1) + "\n")
    gens = {a.get("provenance", {}).get("generator", {}).get("sha256") for a, _, _ in rendered if a["kind"] == "random"}
    cfgs = {a.get("provenance", {}).get("config", {}).get("sha256") for a, _, _ in rendered}
    prov = {"generator_sha256": sorted(x for x in gens if x), "config_sha256": sorted(x for x in cfgs if x),
            "episodes_py_sha256": sha256_file(THIS_FILE),
            "arrangements": {a["arrangement_id"]: {"seed": a["seed"], "arrangement_sha256": a.get("arrangement_sha256"),
                                                   "labels_sha256": lab["_file_sha256"]} for a, lab, _ in rendered}}
    written = {}
    for rl in ref_lightings:
        p = root / JOBS_FILES[rl]
        p.write_text(json.dumps(jobs_doc(rendered, rl, prov), indent=1) + "\n")
        written[p.name] = sha256_file(p)
    save_manifest(manifest_records(rendered), root / "manifest.json")
    written["manifest.json"] = sha256_file(root / "manifest.json")
    for name, doc in (("truth.json", truth_doc(rendered)), ("view_labels.json", view_labels_doc(rendered))):
        (root / name).write_text(json.dumps(doc, indent=1) + "\n")
        written[name] = sha256_file(root / name)
    (root / "summary.md").write_text(summary_md(rendered))
    if positions is not None:
        dst = root / Path(positions).name
        shutil.copyfile(A._repo_path(positions), dst)
        written[dst.name] = sha256_file(dst)
    if sheets:
        for _, lab, d in rendered:
            contact_sheets(d, lab)
    doc = {"schema": SET_SCHEMA, "simulation": True, "split": "dev", "root": str(root),
           "n_arrangements": len(rendered), "n_episodes": len(rendered) * len(CONDITIONS) * 2,
           "jobs": {JOBS_ROLE[rl]: JOBS_FILES[rl] for rl in ref_lightings},
           "jobs_note": "primary = jobs_refL1.json (every query vs the L1 reference at its stop); secondary = jobs_same.json "
                        "(a separate run, reported separately, never a relabelling of the primary); report L1 and L2 queries "
                        "separately",
           "files": written, "provenance": prov}
    (root / "set.json").write_text(json.dumps(doc, indent=1) + "\n")
    (root / "README.md").write_text(set_readme(root, rendered, ref_lightings))
    return doc


def set_readme(root: Path, rendered: Sequence[tuple[dict, dict, Path]], ref_lightings: Sequence[str]) -> str:
    ids = ", ".join(a["arrangement_id"] for a, _, _ in rendered)
    ind = ", ".join(a["arrangement_id"] for a, _, _ in rendered if a.get("predeclared_indistinguishable")) or "none"
    jobs = []
    if "L1" in ref_lightings:
        jobs.append("- `jobs_refL1.json`: **PRIMARY** evidence jobs. Every query, L1 and L2, is paired with the **L1 reference** at "
                    "its stop (docs/SIMULATION_STUDY.md, Reference pairing). Report L1 and L2 queries separately; the lighting is "
                    "in every episode_id (`<ARR>-<condition>-<L>`).")
    if "same" in ref_lightings:
        jobs.append("- `jobs_same.json`: **SECONDARY** evidence jobs. Each query is paired with the reference at its own lighting "
                    "(L1-L1, L2-L2). It is a separate run, reported separately, and never a relabelling of the primary run: the "
                    "reference enters the evidence scores, so it can change decisions and the views a policy requests.")
    return "\n".join([
        f"# {root.name}: simulated dev set (SIMULATION)", "",
        "Isaac Sim 6.2 RTX renders of generated shelf arrangements: `sim/arrange.py` (generator) -> `sim/isaac_batch.py` "
        "(renderer, GB10) -> `sim/episodes.py` (this folder's jobs, annotations and manifest). Dev split only; every "
        "result from it is a simulation result.", "",
        f"- Arrangements ({len(rendered)}): {ids}. Predeclared indistinguishable (IND28 occluder family): {ind}.",
        f"- Episodes: {len(rendered) * len(CONDITIONS) * 2} (3 conditions x 2 lightings each); 5 stops per episode, "
        "stop index = position in `positions_shelf_A.json` (P3 = 2, the start stop).", "",
        "## Files", "",
        *jobs,
        "- `manifest.json`: verify manifest (`verify.manifest`), split dev.",
        "- `truth.json`: present / removed per episode (scoring only).",
        "- `view_labels.json`: exact per-view visible fractions and diagnostic labels from the instance masks (development "
        "analysis and scoring only; never a policy input).",
        "- `summary.md`: per-arrangement table. `set.json`: file hashes and generator / config hashes.",
        "- `positions_shelf_A.json`: the stops (copy of `configs/sim/positions_shelf_A.json`, per-stop pitch).",
        "- `<ARR>/`: `arrangement.json` (seed, objects, phases, generator and config hashes), `rgb/<phase>_<stop>_<L>.png`, "
        "`masks/` (target masks, target-alone masks, class maps), `annotations/ann_<stop>_<L>.json` (one per stop and "
        "reference lighting; boxes from that lighting's reference mask), `labels.json`, `render.json`, contact sheets.", "",
        "Diagnostic (docs/EXPERIMENT_PLAN.md): a present-target view with visible fraction >= 0.90; a removed view whose "
        "expected region is unobstructed (the occluder covers < 10 % of the target's reference-mask pixels).", ""])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.episodes", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="folder with one rendered <ARR>/ per arrangement (sim/isaac_batch.py)")
    ap.add_argument("--ref-lighting", nargs="+", default=list(REF_LIGHTINGS), choices=REF_LIGHTINGS,
                    help="reference pairing(s) to write: L1 -> jobs_refL1.json (PRIMARY), same -> jobs_same.json "
                         "(SECONDARY, a separate run); default both")
    ap.add_argument("--positions", default="configs/sim/positions_shelf_A.json", help="copied into the set")
    ap.add_argument("--contact-sheets", action="store_true", help="also write contact sheets (needs cv2)")
    a = ap.parse_args(argv)
    doc = build(a.root, a.ref_lighting, a.positions, a.contact_sheets)
    print(json.dumps({k: doc[k] for k in ("root", "n_arrangements", "n_episodes", "files")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
