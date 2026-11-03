"""Rendered sim arrangements -> evidence annotations and jobs, the verify manifest, truth and view labels.

Why: the next stage runs perception/evidence.py (frozen v0.2.1) on every query view and replays S/F/R/G through
verify/. Both read fixed formats, so this step writes exactly those (no conversion later):
  - one annotation (cmpe249.evidence-annotation.v0) per (arrangement, stop, reference lighting): the reference image
    and the target's box at that stop, from that lighting's reference mask (reference pixels, continuous coordinates,
    NOMINAL camera). Query views are taken at the arrangement's revisit poses (small seeded camera perturbations); the
    evidence step registers each query to its reference through the four tags, so the expected box stays the reference's;
  - jobs (cmpe249.evidence-jobs.v0) for every query view, visible / occluded / removed x L1 / L2 x 5 stops, each
    against its stop's reference:
      jobs_refL1.json  PRIMARY (docs/SIMULATION_STUDY.md, "Reference pairing", 2026-11-03): every query, L1 and L2,
                       against the L1 reference at its stop (memory recorded once; the lighting may differ at revisit);
      jobs_same.json   SECONDARY: each query against the reference at its own lighting. A separate run reported
                       separately, never a relabelling of the primary (the reference enters the evidence scores, so it
                       can change decisions and the views a policy requests);
    episode_id = verify.schema.episode_id(arrangement, condition, lighting), so results can be reported per lighting;
    stop = index into the positions file;
  - manifest.json (cmpe249.arrangements.v1 plus the top-level key "simulation": true, so verify can label the run
    SIMULATION; each entry = verify's ArrangementRecord.to_dict() plus the plain keys occluder_family, m16_side and
    design_case); split dev; indistinguishable = the arrangement's predeclared flag, i.e. the IND28 occluder family;
  - truth.json (scoring only), view_labels.json (exact per-view visible fractions and diagnostic labels from the
    masks, for development analysis only: never a policy input), true_alignment.json (the true query-to-reference homographies
    of the shelf front plane and of the target's depth plane from the known camera poses, sim/alignment.py: evaluation-only,
    never an evidence or policy input; the corner tags are gone from shelf_v3 scenes), summary.md, set.json (hashes, revisit deltas);
  - optional contact sheets (cv2) per arrangement and lighting, and a crop sheet around the target.

The random slots and the predeclared design cases DES0..3 are separate groups: the files above (manifest, jobs, truth,
view labels, summary) describe the random slots only; a set with design cases also gets manifest_design.json,
jobs_design_refL1.json, jobs_design_same.json, truth_design.json, view_labels_design.json and summary_design.md.

Integrity checks (PR #4 review): load_rendered raises on a folder with arrangement.json but no labels.json (or a leftover
<ARR>.partial), and on a folder whose labels.json was rendered from another arrangement file (labels.hashes.
arrangement_file_sha256 != sha256(arrangement.json)); build refuses a --positions file whose sha256 differs from any
arrangement's recorded positions sha256 or whose stop names differ from labels['stops'].

Paths in annotations and jobs are absolute: the share (/mnt/bench-data/...) has the same path on the Jetson and the
GB10. Only dev arrangements are accepted (integer seeds 0-9999 or the design seeds), as in sim/arrange.py.

    python -m sim.episodes --root /mnt/bench-data/datasets/cmpe249-active-verification/sim/shelf_dev_v1 \\
        [--ref-lighting same L1] [--contact-sheets] [--positions configs/sim/positions_shelf_A.json] [--generation-log LOG]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from sim import alignment as AL
from sim import arrange as A
from verify.schema import episode_id

ANNOTATION_SCHEMA = "cmpe249.evidence-annotation.v0"   # perception/evidence.py (frozen; not imported: it needs cv2)
JOBS_SCHEMA = "cmpe249.evidence-jobs.v0"
SET_SCHEMA = "cmpe249.sim-set.v1"
VIEW_LABELS_SCHEMA = "cmpe249.sim-view-labels-set.v1"
LABELS_SCHEMA = "cmpe249.sim-view-labels.v1"           # = sim/isaac_batch.py (tests/test_sim_batch.py checks)
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


class IncompleteRender(RuntimeError):
    """A folder under the set root is a render that never finished (or is mixed): the set must not be built from it."""


def load_rendered(root: str | Path) -> list[tuple[dict, dict, Path]]:
    """(arrangement, labels, folder) for every rendered arrangement under root, sorted by id.
    Raises IncompleteRender, listing every offender, for a folder with arrangement.json but no labels.json (a crashed
    render), a leftover <ARR>.partial (an interrupted render), or labels.json without arrangement.json; raises ValueError
    for a folder whose labels.json was rendered from a different arrangement file (labels.hashes.arrangement_file_sha256),
    whose id does not match its name, or whose labels have another schema. Nothing is skipped silently."""
    out, incomplete = [], []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        has_arr, has_lab = (d / "arrangement.json").is_file(), (d / "labels.json").is_file()
        if d.name.endswith(".partial"):
            incomplete.append(f"{d.name}: interrupted render (leftover .partial folder)")
        elif has_arr and not has_lab:
            incomplete.append(f"{d.name}: arrangement.json but no labels.json (render did not finish)")
        elif has_lab and not has_arr:
            incomplete.append(f"{d.name}: labels.json but no arrangement.json")
        elif has_arr and has_lab:
            arr_bytes = (d / "arrangement.json").read_bytes()
            arr = json.loads(arr_bytes)
            lab = json.loads((d / "labels.json").read_text())
            check_dev(arr)
            if lab.get("schema") != LABELS_SCHEMA:
                raise ValueError(f"{d}: labels schema {lab.get('schema')!r}, expected {LABELS_SCHEMA!r} (re-render)")
            if lab["arrangement_id"] != arr["arrangement_id"] or d.name != arr["arrangement_id"]:
                raise ValueError(f"{d}: arrangement id mismatch")
            have = lab.get("hashes", {}).get("arrangement_file_sha256")
            if have != hashlib.sha256(arr_bytes).hexdigest():
                raise ValueError(f"{d}: labels.json was rendered from arrangement file {str(have)[:12]}, but arrangement.json is "
                                 f"{hashlib.sha256(arr_bytes).hexdigest()[:12]} (a mixed folder: render it again)")
            out.append((arr, lab, d.resolve()))
    if incomplete:
        raise IncompleteRender(f"{root}: {len(incomplete)} incomplete render folder(s); render them again or remove them:\n  "
                               + "\n  ".join(incomplete))
    return out


def rgb_path(folder: Path, phase: str, stop: str, lighting: str) -> Path:
    return folder / "rgb" / f"{phase}_{stop}_{lighting}.png"


def tag_free(arr: Mapping) -> bool:
    """True for a scene without the corner ArUco tags (shelf_v3: scene.tags false -> scene.tags_enabled false)."""
    return arr.get("scene", {}).get("tags_enabled", True) is False


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
                    "reference render (nominal camera), continuous pixel coordinates (pixel i covers [i, i + 1)). Query views "
                    "are rendered at the arrangement's revisit pose (a small seeded camera perturbation); "
                    + ("the scene has NO corner tags: the evidence step must align the query to this reference itself (or fall back to "
                       "no alignment); the true alignment is in true_alignment.json, for scoring only. "
                       if tag_free(arr) else "the evidence step registers the query to this reference through the tags. ")
                    + "Written by sim/episodes.py before any evidence run.",
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


def jobs_doc(rendered: Sequence[tuple[dict, dict, Path]], ref_lighting: str, provenance: Mapping, group: str = "random") -> dict:
    jobs = [j for arr, lab, d in rendered for j in jobs_for(arr, lab, d, ref_lighting)]
    note = ("PRIMARY run (docs/SIMULATION_STUDY.md, Reference pairing): every query, L1 and L2, against the L1 reference "
            "at its stop. Report L1 and L2 queries separately (the lighting is in every episode_id)."
            if ref_lighting == "L1" else
            "SECONDARY run: each query against the reference at its own lighting (L1-L1, L2-L2). A separate run, reported "
            "separately; never a relabelling of the primary run (jobs_refL1.json).")
    return {"schema": JOBS_SCHEMA, "simulation": True, "role": JOBS_ROLE[ref_lighting], "ref_lighting": ref_lighting,
            "group": group, "note": "SIMULATION. " + note, "provenance": dict(provenance), "jobs": jobs}


def manifest_records(rendered: Sequence[tuple[dict, dict, Path]]):
    from verify.manifest import ArrangementRecord

    return [ArrangementRecord(arr["arrangement_id"], arr["target_id"], "dev", tuple(lab["lightings"]),
                              bool(arr.get("predeclared_indistinguishable", False))) for arr, lab, _ in rendered]


def manifest_doc(rendered: Sequence[tuple[dict, dict, Path]]) -> dict:
    """The verify manifest (cmpe249.arrangements.v1) with "simulation": true at the top level and, per arrangement, the
    plain keys occluder_family, m16_side ("L" / "R" / null) and design_case next to ArrangementRecord.to_dict()'s.
    Validated with verify.manifest first (complete arrangements, no duplicates)."""
    from verify.manifest import MANIFEST_SCHEMA, episodes_from_arrangements

    recs = manifest_records(rendered)
    episodes_from_arrangements(recs)
    rows = [{**rec.to_dict(), "occluder_family": arr["occluder_family"], "m16_side": arr.get("m16_side"),
             "design_case": bool(arr.get("design_case", False))} for rec, (arr, _, _) in zip(recs, rendered)]
    return {"schema": MANIFEST_SCHEMA, "simulation": True, "arrangements": rows}


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
                               "occluder_share_of_region": (lab["region_obstruction"][cond][s]
                                                            if cond in ("occluded", "removed") else None),
                               "diagnostic": bool(lab["diagnostic"][cond][s])} for i, s in enumerate(lab["stops"])]}
    return {"schema": VIEW_LABELS_SCHEMA, "simulation": True,
            "note": "Exact labels from the renderer's instance masks at each query view's own (revisit) camera pose, for "
                    "development analysis and scoring only; never a policy input (docs/EXPERIMENT_PLAN.md: visibility "
                    "annotations are stored separately from policy inputs).",
            "rules": rendered[0][1]["rules"] if rendered else {}, "episodes": eps}


def summary_md(rendered: Sequence[tuple[dict, dict, Path]]) -> str:
    stops = rendered[0][1]["stops"] if rendered else []
    head = ["| Arrangement | Seed | Target | Occluder | Side | Occluded visible fraction " + " / ".join(stops)
            + " | Pattern | Intended | Diagnostic views (visible / occluded / removed) | Ref. vf min | Removed: occluder share of region "
            + " / ".join(stops) + " |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for arr, lab, _ in rendered:
        o = arr["objects"]["occluder"]
        occ = lab["visible_fraction"]["occluded"]
        ref = [v for v in lab["visible_fraction"]["reference"].values() if v is not None]
        rem = lab["region_obstruction"]["removed"]
        n = lab["n_diagnostic"]
        rows.append(f"| {arr['arrangement_id']} | {arr['seed']} | {arr['target_id']} | {arr['occluder_family']}"
                    f"{'' if o['kind'] == 'box' else ' (' + o['label'] + ')'} | {arr.get('m16_side') or ''} | "
                    + " / ".join("-" if occ[s] is None else f"{occ[s]:.3f}" for s in stops)
                    + f" | {lab['patterns']['occluded']} | {lab.get('intended_pattern') or ''} | "
                    f"{n['visible']} / {n['occluded']} / {n['removed']} | {'-' if not ref else format(min(ref), '.4f')} | "
                    + " / ".join("-" if rem[s] is None else f"{rem[s]:.3f}" for s in stops) + " |")
    return "\n".join(["SIMULATION (Isaac Sim renders; labels from instance masks at the revisit poses of the query views). "
                      "Diagnostic: present views with visible fraction >= 0.90; removed views where the occluder covers < 10 % "
                      "of the target's region (the visible-phase mask at the same pose).", "", *head, *rows, ""])


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

    def num(v):
        return "-" if v is None else f"{v:.2f}"       # None: no target pixels to divide by (visible fraction / obstruction)

    def label_text(ph, s):
        if ph == "removed":
            v = labels["region_obstruction"]["removed"][s]
            return f"{ph} {s} occ {num(v)} {'D' if labels['diagnostic']['removed'][s] else 'nd'}"
        v = labels["visible_fraction"][ph][s]
        d = labels["diagnostic"].get(ph, {}).get(s)
        return f"{ph} {s} vf {num(v)}" + ("" if d is None else (" D" if d else " nd"))

    def alone_mask(ph, s):
        name = f"target_alone_{s}.png" if ph == "reference" else f"target_alone_revisit_{s}.png"
        return cv2.imread(str(folder / "masks" / name), cv2.IMREAD_GRAYSCALE) > 0

    for lighting in labels["lightings"]:
        rows = []
        for ph in phases:
            tiles = []
            for s in stops:
                img = cv2.imread(str(rgb_path(folder, ph, s, lighting)))
                sc = tile_w / img.shape[1]
                t = cv2.resize(img, (tile_w, round(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                cls = cv2.imread(str(folder / "masks" / f"classes_{ph}_{s}.png"), cv2.IMREAD_GRAYSCALE)
                outline(t, alone_mask(ph, s), (0, 200, 0), sc)
                if ph in ("occluded", "removed"):
                    outline(t, cls == 2, (0, 0, 230), sc)
                tag(t, label_text(ph, s), 0.55)
                tiles.append(t)
            rows.append(np.hstack(tiles))
        sheet = np.vstack(rows)
        bar = np.full((40, sheet.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"SIMULATION {labels['arrangement_id']} seed {labels['seed']} target {labels['target_id']} "
                    f"occluder {labels['occluder_family']}{'' if not labels.get('m16_side') else '-' + labels['m16_side']} {lighting}; "
                    "green = target (alone mask), red = occluder; queries at revisit poses",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        p = folder / f"contact_{lighting}.jpg"
        cv2.imwrite(str(p), np.vstack([bar, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(p)
    # crops around the target at every stop (L1): reference / visible / occluded / removed, each around the target's
    # mask box of its own pose (reference box for the reference, the visible-phase box for the queries)
    L = labels["lightings"][0]
    rows = []
    for ph in ("reference", "occluded", "removed"):
        tiles = []
        for s in stops:
            img = cv2.imread(str(rgb_path(folder, ph, s, L)))
            box = labels["reference_box"][L][s] if ph == "reference" else labels["query_box_visible"][L][s]
            if box is None:                       # no target pixels at this view: show the whole frame
                box = [0, 0, img.shape[1], img.shape[0]]
            x0, y0, x1, y1 = box
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


def check_positions(positions: str | Path, rendered: Sequence[tuple[dict, dict, Path]]) -> str:
    """The positions file to copy into the set must be the one every arrangement was generated and rendered with: its
    sha256 must equal each arrangement's provenance.positions.sha256 and its stop names must equal labels['stops'].
    Returns its sha256; ValueError otherwise."""
    path = A._repo_path(positions)
    sha = sha256_file(path)
    names = [st["name"] for st in json.loads(path.read_text())["stops"]]
    bad = [arr["arrangement_id"] for arr, _, _ in rendered
           if arr.get("provenance", {}).get("positions", {}).get("sha256") != sha]
    if bad:
        raise ValueError(f"{positions}: sha256 {sha[:12]} differs from the positions file recorded by {bad[:5]}"
                         f"{' ...' if len(bad) > 5 else ''} ({len(bad)} arrangements); the set's stops would not be the "
                         "cameras the images were rendered from")
    bad = [arr["arrangement_id"] for arr, lab, _ in rendered if list(lab["stops"]) != names]
    if bad:
        raise ValueError(f"{positions}: stop names {names} differ from labels['stops'] of {bad[:5]}")
    return sha


def _group_files(group: str) -> dict[str, str]:
    sfx = "" if group == "random" else "_design"
    return {"manifest": f"manifest{sfx}.json", "truth": f"truth{sfx}.json", "view_labels": f"view_labels{sfx}.json",
            "true_alignment": f"true_alignment{sfx}.json",
            "summary": f"summary{sfx}.md", "jobs_L1": f"jobs{sfx}_refL1.json", "jobs_same": f"jobs{sfx}_same.json"}


def revisit_summary(arr: Mapping) -> dict:
    return {s: {"delta": v["delta"], "attempt": v["attempt"], "n_rejected": v["n_rejected"], "min_margin_px": v["margin_px"]["min"]}
            for s, v in arr["revisit"]["stops"].items()}


def check_single_version(rendered: Sequence[tuple[dict, dict, Path]]) -> tuple[set, set]:
    """({generator sha256}, {config sha256}) of the set; ValueError if the arrangements (the random slots and the design
    cases) come from more than one generator sha256 or config sha256: a set built from two generator or config versions
    would silently mix them in the manifest, jobs and view labels (regenerate and re-render the odd ones out). An
    arrangement without a recorded hash counts as its own version."""
    gens = {a.get("provenance", {}).get("generator", {}).get("sha256") for a, _, _ in rendered}
    cfgs = {a.get("provenance", {}).get("config", {}).get("sha256") for a, _, _ in rendered}
    for what, vals, key in (("generator", gens, "generator"), ("config", cfgs, "config")):
        if len(vals) > 1:
            by = {}
            for a, _, _ in rendered:
                by.setdefault(str(a.get("provenance", {}).get(key, {}).get("sha256"))[:12], []).append(a["arrangement_id"])
            raise ValueError(f"the arrangements come from {len(vals)} {what} versions, not one: "
                             + "; ".join(f"{sha}: {ids[:6]}{' ...' if len(ids) > 6 else ''}" for sha, ids in sorted(by.items()))
                             + ". Regenerate and re-render the odd ones out.")
    return gens, cfgs


def build(root: str | Path, ref_lightings: Sequence[str] = REF_LIGHTINGS, positions: str | Path | None = None,
          sheets: bool = False, generation_log: str | Path | None = None) -> dict:
    root = Path(root).resolve()
    rendered = load_rendered(root)
    if not rendered:
        raise ValueError(f"no rendered arrangements under {root}")
    gens, cfgs = check_single_version(rendered)           # before anything is written
    pos_sha = check_positions(positions, rendered) if positions is not None else None
    for arr, lab, d in rendered:
        lab["_file_sha256"] = sha256_file(d / "labels.json")
        for i, stop in enumerate(lab["stops"]):
            for lighting in lab["lightings"]:
                p = annotation_path(d, stop, lighting)
                p.parent.mkdir(exist_ok=True)
                p.write_text(json.dumps(annotation_doc(arr, lab, d, i, lighting), indent=1) + "\n")
    prov = {"generator_sha256": sorted(x for x in gens if x), "config_sha256": sorted(x for x in cfgs if x),
            "episodes_py_sha256": sha256_file(THIS_FILE), "positions_sha256": pos_sha,
            "arrangements": {a["arrangement_id"]: {"seed": a["seed"], "arrangement_sha256": a.get("arrangement_sha256"),
                                                   "labels_sha256": lab["_file_sha256"], "occluder_family": a["occluder_family"],
                                                   "m16_side": a.get("m16_side"), "design_case": bool(a.get("design_case", False)),
                                                   "slot": (a.get("slot") or {}).get("index"),
                                                   "revisit": revisit_summary(a)} for a, lab, _ in rendered}}
    groups = {"random": [x for x in rendered if not x[0].get("design_case", False)],
              "design": [x for x in rendered if x[0].get("design_case", False)]}
    written, group_docs = {}, {}
    for g, items in groups.items():
        if not items:
            continue
        fn = _group_files(g)
        for rl in ref_lightings:
            p = root / fn["jobs_" + ("L1" if rl == "L1" else "same")]
            p.write_text(json.dumps(jobs_doc(items, rl, prov, g), indent=1) + "\n")
            written[p.name] = sha256_file(p)
        for key, doc in (("manifest", manifest_doc(items)), ("truth", truth_doc(items)), ("view_labels", view_labels_doc(items))):
            (root / fn[key]).write_text(json.dumps(doc, indent=1 if key != "manifest" else 2) + "\n")
            written[fn[key]] = sha256_file(root / fn[key])
        if all(a.get("scene", {}).get("revisit_cameras") and "forward" in a["scene"]["revisit_cameras"][0] for a, _, _ in items):
            (root / fn["true_alignment"]).write_text(json.dumps(AL.doc([(a, lab) for a, lab, _ in items], g)) + "\n")
            written[fn["true_alignment"]] = sha256_file(root / fn["true_alignment"])
        (root / fn["summary"]).write_text(summary_md(items))
        group_docs[g] = {"arrangements": [a["arrangement_id"] for a, _, _ in items], "n_arrangements": len(items),
                         "n_episodes": sum(len(CONDITIONS) * len(lab["lightings"]) for _, lab, _ in items),
                         "jobs": {JOBS_ROLE[rl]: fn["jobs_" + ("L1" if rl == "L1" else "same")] for rl in ref_lightings},
                         "manifest": fn["manifest"]}
    if positions is not None:
        dst = root / Path(positions).name
        shutil.copyfile(A._repo_path(positions), dst)
        written[dst.name] = sha256_file(dst)
    if generation_log is not None:
        dst = root / "generation_log.json"
        shutil.copyfile(generation_log, dst)
        written[dst.name] = sha256_file(dst)
    if sheets:
        for _, lab, d in rendered:
            contact_sheets(d, lab)
    rnd = group_docs.get("random", {"arrangements": [], "n_arrangements": 0, "n_episodes": 0, "jobs": {}})
    doc = {"schema": SET_SCHEMA, "simulation": True, "split": "dev", "root": str(root),
           "n_arrangements": rnd["n_arrangements"], "n_episodes": rnd["n_episodes"], "jobs": rnd["jobs"], "groups": group_docs,
           "jobs_note": "primary = jobs_refL1.json (every query vs the L1 reference at its stop); secondary = jobs_same.json "
                        "(a separate run, reported separately, never a relabelling of the primary); report L1 and L2 queries "
                        "separately. The random slots and the design cases DES0..3 are separate groups (files with _design).",
           "label_sanity_flagged": {a["arrangement_id"]: lab["label_sanity"]["flagged_stops"] for a, lab, _ in rendered
                                    if lab.get("label_sanity") and not lab["label_sanity"]["ok"]},
           "files": written, "provenance": prov}
    (root / "set.json").write_text(json.dumps(doc, indent=1) + "\n")
    (root / "README.md").write_text(set_readme(root, groups, ref_lightings))
    return doc


def set_readme(root: Path, groups: Mapping[str, Sequence[tuple[dict, dict, Path]]], ref_lightings: Sequence[str]) -> str:
    rendered = [x for g in groups.values() for x in g]
    ids = ", ".join(a["arrangement_id"] for a, _, _ in groups["random"]) or "none"
    dids = ", ".join(a["arrangement_id"] for a, _, _ in groups["design"]) or "none"
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
        f"- Random slots ({len(groups['random'])}): {ids}.",
        f"- Design cases, outside the slots ({len(groups['design'])}): {dids}. They have their own `*_design*` files.",
        f"- Predeclared indistinguishable (IND28 occluder family, DES3): {ind}.",
        "- Episodes: 3 conditions x 2 lightings per arrangement; 5 stops per episode, stop index = position in "
        "`positions_shelf_A.json` (P3 = 2, the start stop).",
        "- Query views (visible / occluded / removed, both lightings) are rendered at the arrangement's revisit poses (a small "
        "seeded camera perturbation, one draw per stop); the references at the nominal poses.",
        ("- The scenes have no corner ArUco tags (shelf_v3): the query is aligned to its reference by the evidence step itself "
         "(or not at all); `true_alignment.json` is the ground truth for scoring that." if any(tag_free(a) for a, _, _ in rendered)
         else "- The scenes carry the four corner ArUco tags (shelf_v1 / v2)."), "",
        "## Files", "",
        *jobs,
        "- `manifest.json`: verify manifest (`verify.manifest`), split dev, `\"simulation\": true`, with `occluder_family`, "
        "`m16_side` and `design_case` per arrangement. `manifest_design.json` is the same for the design cases.",
        "- `truth.json`: present / removed per episode (scoring only).",
        "- `true_alignment.json`: the true query-to-reference homographies (front plane y = 0 and the target's depth plane) per "
        "arrangement and stop, with the revisit pose deltas and the reference-box corner displacement; from the simulator's "
        "camera poses (sim/alignment.py). Evaluation-only, never an evidence or policy input. The corner tags are not in "
        "shelf_v3 scenes; shelf_v1 / v2 sets still have them.",
        "- `view_labels.json`: exact per-view visible fractions and diagnostic labels from the instance masks at each view's "
        "own pose (development analysis and scoring only; never a policy input).",
        "- `summary.md`: per-arrangement table. `set.json`: file hashes, generator / config hashes, revisit deltas.",
        "- `positions_shelf_A.json`: the stops (copy of `configs/sim/positions_shelf_A.json`, per-stop pitch).",
        "- `generation_log.json` (if given): the generator's set-level log of failed tries and reserve replacements.",
        "- `<ARR>/`: `arrangement.json` (seed, objects, phases, revisit cameras, generator and config hashes), "
        "`rgb/<phase>_<stop>_<L>.png`, `masks/` (target masks, target-alone masks at the nominal and revisit poses, class maps), "
        "`annotations/ann_<stop>_<L>.json` (one per stop and reference lighting; boxes from that lighting's nominal reference "
        "mask), `labels.json`, `render.json`, contact sheets.", "",
        "Diagnostic (docs/SIM_PREREGISTRATION.md section 3): a present-target view with visible fraction >= 0.90; a removed view "
        "whose expected region is unobstructed (the occluder covers < 10 % of the target's region: the visible-phase mask at the "
        "same revisit pose).", ""])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.episodes", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="folder with one rendered <ARR>/ per arrangement (sim/isaac_batch.py)")
    ap.add_argument("--ref-lighting", nargs="+", default=list(REF_LIGHTINGS), choices=REF_LIGHTINGS,
                    help="reference pairing(s) to write: L1 -> jobs_refL1.json (PRIMARY), same -> jobs_same.json "
                         "(SECONDARY, a separate run); default both")
    ap.add_argument("--positions", default="configs/sim/positions_shelf_A.json",
                    help="copied into the set; must be the file the arrangements were generated with (sha256 checked; shelf_v1, v2 "
                         "and v3 all use positions_shelf_A.json)")
    ap.add_argument("--generation-log", default=None, help="the generator's generation_log.json, copied into the set")
    ap.add_argument("--contact-sheets", action="store_true", help="also write contact sheets (needs cv2)")
    a = ap.parse_args(argv)
    try:
        doc = build(a.root, a.ref_lighting, a.positions, a.contact_sheets, a.generation_log)
    except (IncompleteRender, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: doc[k] for k in ("root", "n_arrangements", "n_episodes", "files")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
