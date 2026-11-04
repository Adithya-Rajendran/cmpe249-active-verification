"""Summarise an evidence + select-v4 run per variant and per lighting (the `summary` stage of tools/pipeline.py).

For every evidence variant in the spec's `summary` section it reports, from files already written:
    - registration refusals (align_ok false) per lighting;
    - the gate (quality >= tau_q and obstruction <= tau_occ) pass rate per lighting and view class, at the
      variant's own select-v4 thresholds (when calibration produced a candidate) and at the fixed illustrative
      setting of the spec (the same for every variant);
    - select-v4's outcome and, from selection_details.json, the held-out policy FRR, removal recall,
      false present (a removed target answered 'present') and coverage per lighting (uncapped, primary k and lam);
    - the DES design cases separately (they are never part of calibration).
View classes come from view_labels*.json (renderer masks; evaluation only): present = a visible-condition stop;
occluded C / p / H = an occluded-condition stop with visible fraction >= 0.9 / between / <= 0.05; removed diag /
removed non-diag = a removed-condition stop whose region the occluder covers < 0.1 / more.

Writes RUN_DIR/summary.md and RUN_DIR/summary.json. Stdlib only. Usage:
    python tools/evidence_summary.py --spec configs/pipeline/shelf_dev_v3_evidence.json --run-dir runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CLASSES = ("present", "occluded C", "occluded p", "occluded H", "removed diag", "removed non-diag")
LIGHTINGS = ("L1", "L2")


def view_class(ep: dict, st: dict) -> str:
    if ep["condition"] == "visible":
        return "present"
    if ep["condition"] == "removed":
        return "removed diag" if st["diagnostic"] else "removed non-diag"
    vf = st["visible_fraction"]
    return "occluded C" if vf >= 0.9 else "occluded H" if vf <= 0.05 else "occluded p"


def load_views(diag_path: Path, labels: dict, obstruction_key: str) -> list[dict]:
    views = []
    for line in diag_path.open():
        if not line.strip():
            continue
        d = json.loads(line)
        ep = labels[d["episode_id"]]
        st = next(s for s in ep["stops"] if s["stop"] == int(d["stop"]))
        views.append({"episode_id": d["episode_id"], "arrangement": ep["arrangement_id"], "stop": int(d["stop"]),
                      "lighting": ep["lighting"], "condition": ep["condition"], "cls": view_class(ep, st),
                      "quality": d["quality"], "obstruction": d[obstruction_key], "match": d["match_score"],
                      "align_ok": bool(d.get("align_ok", True))})
    return views


def gate(v: dict, th: dict) -> bool:
    return v["quality"] >= th["tau_q"] and v["obstruction"] <= th["tau_occ"]


def frac(x: float, n: int) -> str:
    return f"{x / n:.3f} ({x:g}/{n:g})" if n else "n/a"


def gate_table(views: list[dict], th: dict) -> dict:
    out = {}
    for L in LIGHTINGS:
        for c in CLASSES:
            vs = [v for v in views if v["lighting"] == L and v["cls"] == c]
            out[f"{L}|{c}"] = [sum(gate(v, th) for v in vs), len(vs)]
    return out


def held_out(details: dict, truth: dict) -> dict:
    """policy -> lighting -> FRR, recall, false present, coverage, the confusion counts (truth -> decision) and
    the accuracy among resolved episodes (uncapped episodes of the primary setting)."""
    conf: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    acc: dict = defaultdict(lambda: defaultdict(lambda: {"frr": [0.0, 0], "recall": [0.0, 0], "false_present": [0.0, 0],
                                                         "coverage": [0.0, 0]}))
    for e in details["held_out_episodes"]:
        if e.get("cap_steps") is not None:
            continue
        a = acc[e["policy"]][e["lighting"]]
        f = e["decision_frac"]
        key = "frr" if truth[e["episode_id"]] == "present" else "recall"
        a[key][0] += f["removed"]
        a[key][1] += 1
        if key == "recall":   # a removed target answered "present": wrong, yet counted as covered
            a["false_present"][0] += f["present"]
            a["false_present"][1] += 1
        a["coverage"][0] += 1.0 - f["unknown"]
        a["coverage"][1] += 1
        t = truth[e["episode_id"]]
        for decided in ("present", "removed", "unknown"):   # the full confusion counts, truth x decision
            conf[e["policy"]][e["lighting"]][f"{t}->{decided}"] += f[decided]
    out = {p: {L: {k: [round(x, 4), n] for k, (x, n) in d.items()} for L, d in by.items()} for p, by in acc.items()}
    for p, by in conf.items():
        for L, c in by.items():
            right = c["present->present"] + c["removed->removed"]
            resolved = right + c["present->removed"] + c["removed->present"]
            out[p][L]["confusion"] = {k: round(v, 4) for k, v in c.items()}
            out[p][L]["accuracy_resolved"] = [round(right, 4), round(resolved, 4)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    root, out_dir, sm = Path(spec["set_root"]), spec["evidence"]["out_dir"], spec["summary"]
    labels = json.loads((root / "view_labels.json").read_text())["episodes"]
    labels_des = json.loads((root / "view_labels_design.json").read_text())["episodes"]
    truth = {k: v["truth"] for k, v in labels.items()}
    illustrative = sm["illustrative"]
    report = {"set_root": str(root), "illustrative": illustrative, "variants": {}}
    md = [f"# Evidence variants on {root.name}", "",
          "SIMULATION - Isaac Sim renders; every result is a simulation result. Development data only.", "",
          f"Gate = quality >= tau_q and obstruction <= tau_occ. Illustrative setting (the same for every variant): "
          f"tau_q {illustrative['tau_q']}, tau_occ {illustrative['tau_occ']}. View classes from the renderer's "
          f"masks (evaluation only).", ""]
    for name, v in sm["variants"].items():
        diag = Path(v["diag"].format(out=out_dir, part="refL1"))
        diag_des = Path(v["diag"].format(out=out_dir, part="design_refL1"))
        views = load_views(diag, labels, v["obstruction"])
        des = load_views(diag_des, labels_des, v["obstruction"])
        sel = args.run_dir / "select" / name
        cand_path, details_path = sel / "frozen_candidate.json", sel / "selection_details.json"
        chosen = None
        if cand_path.exists():
            c = json.loads(cand_path.read_text())
            th = c.get("thresholds", c)
            chosen = {k: th[k] for k in ("tau_q", "tau_occ", "tau_match") if k in th}
        outcome = ("calibration infeasible" if (sel / "calibration_infeasible.json").exists()
                   else "candidate written" if chosen else "no select output")
        r = {"diag": str(diag), "outcome": outcome, "chosen": chosen,
             "align_refused": {L: [sum(not x["align_ok"] for x in views if x["lighting"] == L),
                                   sum(x["lighting"] == L for x in views)] for L in LIGHTINGS},
             "gate_illustrative": gate_table(views, illustrative),
             "gate_chosen": gate_table(views, chosen) if chosen else None,
             "held_out": held_out(json.loads(details_path.read_text()), truth) if details_path.exists() else None}
        report["variants"][name] = r
        md += [f"## {name}", "", f"- evidence: `{diag}` (obstruction field `{v['obstruction']}`)",
               f"- select-v4: {outcome}" + (f"; thresholds {chosen}" if chosen else ""),
               "- registration refused: " + ", ".join(f"{L} {frac(*r['align_refused'][L])}" for L in LIGHTINGS), ""]
        for label, tab in (("variant's select-v4 thresholds", r["gate_chosen"]),
                           ("illustrative setting", r["gate_illustrative"])):
            if tab is None:
                continue
            md += [f"Gate pass, {label}:", "", "| lighting | " + " | ".join(CLASSES) + " |",
                   "|---" * (len(CLASSES) + 1) + "|"]
            md += [f"| {L} | " + " | ".join(frac(*tab[f"{L}|{c}"]) for c in CLASSES) + " |" for L in LIGHTINGS]
            md.append("")
        if r["held_out"]:
            md += ["Held-out policies (leave-one-arrangement-out thresholds; uncapped; FRR on present episodes, "
                   "recall and false present on removed, coverage on all):", "",
                   "| policy | L1 FRR | L1 recall | L1 false present | L1 coverage | L2 FRR | L2 recall | "
                   "L2 false present | L2 coverage |", "|---" * 9 + "|"]
            for pol, by in r["held_out"].items():
                cells = [frac(*by[L][k]) if L in by else "n/a" for L in LIGHTINGS
                         for k in ("frr", "recall", "false_present", "coverage")]
                md.append(f"| {pol} | " + " | ".join(cells) + " |")
            md += ["", "Confusion counts (truth -> decision; R-type policies average fractional decisions over seeds) "
                   "and accuracy among resolved episodes (correct / decided present or removed):", "",
                   "| policy | lighting | present->present | present->removed | present->unknown | removed->removed | "
                   "removed->present | removed->unknown | accuracy among resolved |", "|---" * 9 + "|"]
            for pol, by in r["held_out"].items():
                for L in LIGHTINGS:
                    if L not in by:
                        continue
                    c = by[L]["confusion"]
                    cells = [f"{c.get(k, 0):g}" for k in ("present->present", "present->removed", "present->unknown",
                                                          "removed->removed", "removed->present", "removed->unknown")]
                    x, n = by[L]["accuracy_resolved"]
                    md.append(f"| {pol} | {L} | " + " | ".join(cells) + f" | {frac(x, n)} |")
            md.append("")
        if chosen:
            md += ["DES cases (not calibrated on), gate at the variant's thresholds; per stop P1..P5 the class "
                   "(C/p/H present-occluded, R removed) and + where the gate passes:", "",
                   "| episode | stops |", "|---|---|"]
            by_ep: dict = defaultdict(list)
            for x in des:
                by_ep[x["episode_id"]].append(x)
            for eid in sorted(by_ep):
                cells = []
                for x in sorted(by_ep[eid], key=lambda x: x["stop"]):
                    k = {"present": "C", "occluded C": "C", "occluded p": "p", "occluded H": "H"}.get(x["cls"], "R")
                    cells.append(k + ("+" if gate(x, chosen) else ""))
                md.append(f"| {eid} | {' '.join(cells)} |")
            md.append("")
    (args.run_dir / "summary.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    (args.run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {args.run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
