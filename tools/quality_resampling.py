"""How much does each view's evidence move between image registration and the simulator's true alignment?

Evaluation only (the `summary` stage of configs/pipeline/shelf_dev_v3_truealign.json). Compares, view by view, the
v0.3 diagnostics computed on ECC image registration (the deployable step) with the same code on the TRUE homography
of the target plane and of the front plane (sim/true_align_evidence.py). Reports per lighting:
    - quality differences (|delta| > 0.05 / 0.1 / 0.3) and which quality part moved (sharpness, exposure, valid);
    - flips of the gate (quality >= tau_q and obstruction <= tau_occ), of a clear miss (gate and match < tau_match)
      and of a strong match (match >= tau_match), at the v0.3 select-v4 thresholds and at the illustrative setting;
    - the views with the largest quality difference.
The true-alignment diagnostics are read through RUN_DIR/evidence_provenance.json and the image diagnostics through
the provenance of the evidence run summary.image_run (tools/provenance.py: recorded outputs with their recorded
sha256 only). Writes RUN_DIR/quality_resampling.md and .json. Stdlib only. Usage:
    python tools/quality_resampling.py --spec configs/pipeline/shelf_dev_v3_truealign.json --run-dir runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import provenance

LIGHTINGS = ("L1", "L2")
PARTS = ("sharpness", "exposure", "valid")


def load(path: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for line in path.open():
        if line.strip():
            d = json.loads(line)
            out[(d["episode_id"], int(d["stop"]))] = d
    return out


def decisions(d: dict, th: dict) -> dict[str, bool]:
    gate = d["quality"] >= th["tau_q"] and d["obstruction"] <= th["tau_occ"]
    return {"gate": gate, "clear_miss": gate and d["match_score"] < th["tau_match"],
            "strong_match": d["match_score"] >= th["tau_match"]}


def compare(base: dict, other: dict, labels: dict, settings: dict[str, dict]) -> dict:
    res: dict = {L: {"n": 0, "dq": {"0.05": 0, "0.1": 0, "0.3": 0}, "part_moved": {p: 0 for p in PARTS},
                     "flips": {s: {"gate": 0, "clear_miss": 0, "strong_match": 0} for s in settings}}
                 for L in LIGHTINGS}
    rows = []
    for key, b in base.items():
        o = other[key]
        L = labels[key[0]]["lighting"]
        r = res[L]
        r["n"] += 1
        dq = abs(b["quality"] - o["quality"])
        for t in r["dq"]:
            r["dq"][t] += dq > float(t)
        if dq > 0.1:
            for p in PARTS:
                r["part_moved"][p] += abs(b["quality_parts"][p] - o["quality_parts"][p]) > 0.1
        for s, th in settings.items():
            db, do = decisions(b, th), decisions(o, th)
            for k in db:
                r["flips"][s][k] += db[k] != do[k]
        rows.append({"episode_id": key[0], "stop": key[1], "dq": round(dq, 4),
                     "quality": [round(b["quality"], 4), round(o["quality"], 4)],
                     "sharpness": [round(b["quality_parts"]["sharpness"], 4), round(o["quality_parts"]["sharpness"], 4)],
                     "match": [round(b["match_score"], 4), round(o["match_score"], 4)],
                     "registration": b.get("registration", {})})
    rows.sort(key=lambda x: -x["dq"])
    return {"per_lighting": res, "largest": rows[:12]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    sm, root = spec["summary"], Path(spec["set_root"])
    prov = provenance.load(args.run_dir)
    image_prov = provenance.load(Path(__file__).resolve().parents[1] / sm["image_run"])
    labels = json.loads((root / "view_labels.json").read_text())["episodes"]
    cand = json.loads(Path(sm["v03_candidate"]).read_text())["thresholds"]
    settings = {"v0.3 select-v4": {k: cand[k] for k in ("tau_q", "tau_occ", "tau_match")},
                "illustrative": sm["illustrative"]}
    image_diag = provenance.output(image_prov, sm["image_diag"])
    base = load(image_diag)
    report = {"settings": settings, "image_diag": str(image_diag), "planes": {}}
    md = ["# Evidence under image registration vs the true alignment (evaluation only)", "",
          f"SIMULATION, development data: {root.name}, random dev arrangements, {len(base)} views. Base: v0.3 on ECC "
          f"image registration. Compared: the same code on the simulator's true homography. Settings: {settings}.", ""]
    for plane, path in sm["true_diags"].items():
        other = load(provenance.output(prov, path))
        missing = set(base) ^ set(other)
        if missing:
            raise SystemExit(f"{plane}: {len(missing)} views differ between the files, e.g. {sorted(missing)[:3]}")
        c = compare(base, other, labels, settings)
        report["planes"][plane] = c
        md += [f"## True {plane} plane", "", "| lighting | views | abs dq > 0.05 | > 0.1 | > 0.3 | of > 0.1: sharpness "
               "moved | exposure | valid |", "|---|---|---|---|---|---|---|---|"]
        for L in LIGHTINGS:
            r = c["per_lighting"][L]
            md.append(f"| {L} | {r['n']} | " + " | ".join(str(r["dq"][t]) for t in ("0.05", "0.1", "0.3")) + " | "
                      + " | ".join(str(r["part_moved"][p]) for p in PARTS) + " |")
        md += ["", "Decision flips (views whose decision differs between the two alignments):", "",
               "| lighting | setting | gate | clear miss | strong match |", "|---|---|---|---|---|"]
        for L in LIGHTINGS:
            for s in settings:
                f = c["per_lighting"][L]["flips"][s]
                md.append(f"| {L} | {s} | {f['gate']} | {f['clear_miss']} | {f['strong_match']} |")
        md += ["", "Largest quality differences (image, true):", "",
               "| view | dq | quality | sharpness | match | image registration cc / box shift px |",
               "|---|---|---|---|---|---|"]
        for x in c["largest"]:
            reg = x["registration"]
            md.append(f"| {x['episode_id']} P{x['stop'] + 1} | {x['dq']} | {x['quality']} | {x['sharpness']} | "
                      f"{x['match']} | {reg.get('cc')} / {reg.get('box_shift_px')} |")
        md.append("")
    (args.run_dir / "quality_resampling.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    (args.run_dir / "quality_resampling.md").write_text("\n".join(md) + "\n")
    print(f"wrote {args.run_dir / 'quality_resampling.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
