"""Evaluate all level-shelf decisions and matched historical replay controls."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from development.level_shelf.prepare import STATES, ENGINES, LAYOUTS, STOPS
from development.paper_audit.inputs_build import sha, box_from_mask
from development.paper_audit.evaluate_witness import primary_rgb, mask, exposure

OLD = Path("runs/2026-11-18-paper-audit-v1/evaluation/final")
HISTORICAL = {("PAM01", "P5"), ("PAM02", "P4"), ("PAM05", "P4"), ("PAM05", "P5")}
OLD_STOPS = {"P1": "P4", "P2": "P4", "P4": "P5", "P5": "P5"}


def pixel(path):
    with Image.open(path) as image:
        a = np.asarray(image.convert("RGB"))
    return hashlib.sha256(
        json.dumps(list(a.shape), separators=(",", ":")).encode() + b"\0" + a.tobytes()
    ).hexdigest()


def evaluate(packet, ledger, methods, out):
    packet = Path(packet)
    ledger = json.loads(Path(ledger).read_text())
    cases = json.loads((packet / "cases.json").read_text())
    if sha(packet / "cases.json") != ledger["cases_sha256"]:
        raise ValueError("case binding mismatch")
    for proof in ledger["verification"]:
        if sha(proof["path"]) != proof["sha256"]:
            raise ValueError("verification changed")
    public = {r["id"]: r for r in cases}
    private = {r["id"]: r for r in ledger["rows"]}
    if len(public) != 320 or len(private) != 320 or set(public) != set(private):
        raise ValueError("case inventory mismatch")
    predictions = {}
    bindings = {}
    for method, path in methods.items():
        data = [
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
        if (
            len(data) != 320
            or len({r["id"] for r in data}) != 320
            or {r["id"] for r in data} != set(public)
        ):
            raise ValueError("scorer inventory mismatch")
        binding = json.loads((Path(path).parent / "binding.json").read_text())
        if binding["cases_sha256"] != sha(packet / "cases.json"):
            raise ValueError("scorer packet mismatch")
        old_binding = json.loads(
            (
                Path("runs/2026-11-18-paper-audit-v1/scoring") / method / "binding.json"
            ).read_text()
        )
        keys = (
            (
                "script_sha256",
                "model_revision",
                "model_shards",
                "geometry_checkpoint_sha256",
                "spec_sha256",
                "torch",
                "transformers",
            )
            if method == "vggt_qwen"
            else (
                "script_sha256",
                "cross_sim_scorer_sha256",
                "geometry_config_sha256",
                "protocol_sha256",
                "method",
                "params",
                "v03_params",
                "align_params",
                "thresholds",
                "checkpoint_digests",
                "torch",
            )
        )
        if any(
            binding.get(k) != old_binding.get(k)
            or k not in binding
            or k not in old_binding
            for k in keys
        ):
            raise ValueError("scorer settings or checkpoint mismatch: " + method)
        if method == "geometry" and binding.get("provider", {}).get(
            "provenance"
        ) != old_binding.get("provider", {}).get("provenance"):
            raise ValueError("geometry provenance mismatch")
        predictions[method] = {r["id"]: r for r in data}
        bindings[method] = binding
    if set(methods) != {"baseline_200", "geometry", "vggt_qwen"}:
        raise ValueError("all three frozen methods required")
    old = list(csv.DictReader((OLD / "cases.csv").open()))
    archive = {
        (r["engine"], r["arrangement"], r["stop"], r["state"], r["method"]): r
        for r in old
        if r["level"] == "D6" and r["arrangement"] in LAYOUTS
    }
    old_pairs = list(csv.DictReader((OLD / "pairs.csv").open()))
    certified = {
        (r["engine"], r["arrangement"], r["stop"], r["placement"]): r
        for r in old_pairs
        if r["level"] == "D6" and r["arrangement"] in LAYOUTS
    }
    by_key = {
        (r["engine"], r["arrangement"], r["stop"], r["state"]): r
        for r in private.values()
    }
    if len(by_key) != 320 or set(by_key) != {
        (e, a, s, t) for e in ENGINES for a in LAYOUTS for s in STOPS for t in STATES
    }:
        raise ValueError("declared conditions mismatch")
    rows = []
    pairs = []
    for key, p in sorted(by_key.items()):
        engine, arrangement, stop, state = key
        uid = p["id"]
        c = public[uid]
        files = p["files"]
        for rec in files.values():
            if sha(rec["path"]) != rec["sha256"]:
                raise ValueError("native input changed")
        for role in ("reference", "query"):
            path = packet / c[role]
            if (
                path.resolve().parent != packet.resolve()
                or sha(path) != c[role + "_sha256"]
            ):
                raise ValueError("prepared image binding mismatch")
            with Image.open(path) as image:
                if (
                    image.mode != "RGB"
                    or image.size != (1200, 675)
                    or image.info
                    or not np.array_equal(
                        np.asarray(image), primary_rgb(files[role]["path"])
                    )
                ):
                    raise ValueError("native export mismatch")
        if c["box_xyxy"] != box_from_mask(files["reference_mask"]["path"]):
            raise ValueError("reference box mismatch")
        if state.endswith("removed") and mask(files["query_mask"]["path"]).any():
            raise ValueError("removed target mask nonempty")
        present = by_key[
            (engine, arrangement, stop, state.replace("removed", "present"))
        ]
        visible = mask(present["files"]["query_mask"]["path"])
        alone = mask(files["alone_mask"]["path"])
        refmask = mask(files["reference_mask"]["path"])
        vis = exposure(visible, alone)
        placed_mask = mask(
            by_key[(engine, arrangement, stop, "placed_present")]["files"][
                "query_mask"
            ]["path"]
        )
        placed_vis = exposure(placed_mask, alone)
        truth = "REMOVED" if state.endswith("removed") else "PRESENT"
        for method, records in predictions.items():
            score = records[uid]
            if any(
                score[k] != c[k]
                for k in ("box_xyxy", "reference_sha256", "query_sha256")
            ):
                raise ValueError("scorer input mismatch")
            if score["answer"] not in ("PRESENT", "REMOVED", "UNKNOWN"):
                raise ValueError("invalid scorer answer")
            old_stop = OLD_STOPS.get(stop)
            prior = (
                archive.get((engine, arrangement, old_stop, state, method))
                if old_stop
                else None
            )
            row = {
                "engine": engine,
                "arrangement": arrangement,
                "stop": stop,
                "state": state,
                "id": uid,
                "method": method,
                "truth": truth,
                "answer": score["answer"],
                "correct": score["answer"] == truth,
                "failure": score.get("failure"),
                "raw": score.get("raw"),
                "valid_fraction": score.get("valid_fraction"),
                "reference_target_pixels": int(refmask.sum()),
                "retained_target_pixels": vis["visible_pixels"],
                "placed_target_pixels": placed_vis["visible_pixels"],
                "placed_visible_fraction": placed_vis["visible_fraction_of_alone"],
                "isolated_target_pixels": vis["alone_pixels"],
                "visible_fraction": vis["visible_fraction_of_alone"],
                "visible_outside_isolated": vis["visible_outside_alone_pixels"],
                "original_stop": old_stop,
                "original_id": prior["id"] if prior else None,
                "original_answer": prior["answer"] if prior else None,
                "original_reference_file_equal": (
                    prior["reference_sha256"] == c["reference_sha256"]
                )
                if prior
                else None,
                "original_query_file_equal": (
                    prior["query_sha256"] == c["query_sha256"]
                )
                if prior
                else None,
                "original_reference_decoded_equal": (
                    prior["delivered_reference_pixel_sha256_shape_bound"]
                    == pixel(packet / c["reference"])
                )
                if prior
                else None,
                "original_query_decoded_equal": (
                    prior["delivered_query_pixel_sha256_shape_bound"]
                    == pixel(packet / c["query"])
                )
                if prior
                else None,
                "original_native_reference_equal": (
                    prior["native_reference_pixel_sha256_shape_bound"]
                    == pixel(files["reference"]["path"])
                )
                if prior
                else None,
                "original_native_query_equal": (
                    prior["native_query_pixel_sha256_shape_bound"]
                    == pixel(files["query"]["path"])
                )
                if prior
                else None,
                "original_box_equal": (json.loads(prior["box_xyxy"]) == c["box_xyxy"])
                if prior
                else None,
                "historical_false_pair": engine == "mujoco"
                and state.startswith("placed_")
                and (arrangement, old_stop) in HISTORICAL,
                "historical_layout_origins": ",".join(
                    s for a, s in sorted(HISTORICAL) if a == arrangement
                )
                if engine == "mujoco"
                else "",
                "score_record": json.dumps(score, sort_keys=True),
            }
            rows.append(row)
    by_decision = {
        (r["engine"], r["arrangement"], r["stop"], r["state"], r["method"]): r
        for r in rows
    }
    for engine in ENGINES:
        for arrangement in LAYOUTS:
            for stop in STOPS:
                for placement in ("clear", "placed"):
                    a = by_key[(engine, arrangement, stop, placement + "_present")]
                    b = by_key[(engine, arrangement, stop, placement + "_removed")]
                    ca, cb = public[a["id"]], public[b["id"]]
                    native = pixel(a["files"]["query"]["path"]) == pixel(
                        b["files"]["query"]["path"]
                    )
                    delivered = pixel(packet / ca["query"]) == pixel(
                        packet / cb["query"]
                    )
                    equal = (
                        delivered
                        and pixel(packet / ca["reference"])
                        == pixel(packet / cb["reference"])
                        and ca["box_xyxy"] == cb["box_xyxy"]
                    )
                    prior = (
                        certified.get((engine, arrangement, OLD_STOPS[stop], placement))
                        if stop in OLD_STOPS
                        else None
                    )
                    for method in methods:
                        x = by_decision[
                            (engine, arrangement, stop, placement + "_present", method)
                        ]
                        y = by_decision[
                            (engine, arrangement, stop, placement + "_removed", method)
                        ]
                        pairs.append(
                            {
                                "engine": engine,
                                "arrangement": arrangement,
                                "stop": stop,
                                "placement": placement,
                                "method": method,
                                "present_id": a["id"],
                                "removed_id": b["id"],
                                "present_answer": x["answer"],
                                "removed_answer": y["answer"],
                                "native_query_equal": native,
                                "delivered_query_equal": delivered,
                                "complete_input_equal": equal,
                                "commits": x["answer"] != "UNKNOWN"
                                or y["answer"] != "UNKNOWN",
                                "inconsistent_answers": x["answer"] != y["answer"],
                                "visible_fraction": x["visible_fraction"],
                                "visible_pixels": x["retained_target_pixels"],
                                "reference_pixels": x["reference_target_pixels"],
                                "original_stop": OLD_STOPS.get(stop),
                                "original_complete_input_equal": prior[
                                    "complete_input_equal"
                                ]
                                == "True"
                                if prior
                                else None,
                                "historical_false_pair": engine == "mujoco"
                                and placement == "placed"
                                and (arrangement, OLD_STOPS.get(stop)) in HISTORICAL,
                                "historical_layout_origins": ",".join(
                                    s for a, s in sorted(HISTORICAL) if a == arrangement
                                )
                                if engine == "mujoco"
                                else "",
                            }
                        )
    groups = defaultdict(list)
    for r in rows:
        groups[(r["engine"], r["stop"], r["state"], r["method"])].append(r)
    summary = []
    for key, group in sorted(groups.items()):
        ans = Counter(r["answer"] for r in group)
        summary.append(
            dict(
                zip(("engine", "stop", "state", "method"), key),
                n=len(group),
                correct=sum(r["correct"] for r in group),
                definite=sum(r["answer"] != "UNKNOWN" for r in group),
                wrong_definite=sum(
                    r["answer"] not in ("UNKNOWN", r["truth"]) for r in group
                ),
                answers=dict(ans),
                registration_gate=sum(
                    r["failure"] == "registration_gate" for r in group
                ),
            )
        )
    pair_groups = defaultdict(list)
    for r in pairs:
        pair_groups[(r["engine"], r["stop"], r["placement"], r["method"])].append(r)
    pair_summary = []
    for key, group in sorted(pair_groups.items()):
        exact = [r for r in group if r["complete_input_equal"]]
        pair_summary.append(
            dict(
                zip(("engine", "stop", "placement", "method"), key),
                n=len(group),
                exact_pairs=len(exact),
                exact_committing_pairs=sum(r["commits"] for r in exact),
                exact_inconsistent_pairs=sum(r["inconsistent_answers"] for r in exact),
                zero_visible=sum(r["visible_pixels"] == 0 for r in group),
                partially_visible=sum(
                    r["visible_fraction"] is not None and 0 < r["visible_fraction"] < 1
                    for r in group
                ),
                fully_visible=sum(r["visible_fraction"] == 1 for r in group),
                missing_visible_fraction=sum(
                    r["visible_fraction"] is None for r in group
                ),
            )
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "decisions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out / "pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "case_count": len(cases),
                "layout_count_per_engine": len(LAYOUTS),
                "groups": summary,
                "pair_groups": pair_summary,
                "cases_sha256": sha(packet / "cases.json"),
                "records_sha256": {k: sha(v) for k, v in methods.items()},
                "evaluator_sha256": sha(__file__),
                "note": "Descriptive sensitivity check on reused layouts; cases are not independent.",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--packet", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--method", action="append", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    evaluate(a.packet, a.ledger, dict(x.split("=", 1) for x in a.method), a.out)
