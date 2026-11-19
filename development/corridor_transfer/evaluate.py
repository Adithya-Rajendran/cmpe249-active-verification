"""Evaluate the locked corridor panel with pairs and layout clusters retained."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from development.corridor_transfer.prepare import ENGINES, LAYOUTS, STATES, STOPS
from development.paper_audit.evaluate_witness import exposure, mask, primary_rgb
from development.paper_audit.inputs_build import box_from_mask, sha


def decoded_hash(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    shape = json.dumps(list(pixels.shape), separators=(",", ":")).encode()
    return hashlib.sha256(shape + b"\0" + pixels.tobytes()).hexdigest()


def call_type(score):
    failure = score.get("failure")
    if failure == "registration_gate":
        return "registration_gate_unknown"
    if failure == "invalid_model_output":
        return "invalid_model_output"
    if failure not in (None, False, ""):
        return "execution_failure"
    if score["answer"] == "UNKNOWN":
        return "qwen_unknown" if score.get("raw") == "UNKNOWN" else "rule_unknown"
    return score["answer"].lower()


def recurrence_label(opportunity_count, failure_count):
    if not 0 <= failure_count <= opportunity_count:
        raise ValueError("failure count exceeds exact opportunities")
    if opportunity_count == 0:
        return "no_opportunity"
    if failure_count >= 2:
        return "recurrent"
    if failure_count == 1:
        return "isolated"
    return "none_observed"


def evaluate(packet, ledger_path, methods, endpoint_lock, output):
    packet = Path(packet)
    ledger = json.loads(Path(ledger_path).read_text())
    cases = json.loads((packet / "cases.json").read_text())
    if sha(packet / "cases.json") != ledger["cases_sha256"]:
        raise ValueError("packet binding mismatch")
    if sha(endpoint_lock) != ledger["endpoint_lock_sha256"]:
        raise ValueError("endpoint lock changed")
    for proof in ledger["verification"]:
        if sha(proof["path"]) != proof["sha256"]:
            raise ValueError("verification changed")
    public = {r["id"]: r for r in cases}
    private = {r["id"]: r for r in ledger["rows"]}
    if (
        len(cases) != 192
        or len(public) != 192
        or len(private) != 192
        or set(public) != set(private)
    ):
        raise ValueError("packet inventory mismatch")
    if set(methods) != {"baseline_200", "geometry", "vggt_qwen"}:
        raise ValueError("all three frozen methods required")
    predictions = {}
    for method, filename in methods.items():
        records = [
            json.loads(line)
            for line in Path(filename).read_text().splitlines()
            if line.strip()
        ]
        if (
            len(records) != 192
            or len({r["id"] for r in records}) != 192
            or {r["id"] for r in records} != set(public)
        ):
            raise ValueError("scorer inventory mismatch")
        binding = json.loads((Path(filename).parent / "binding.json").read_text())
        old = json.loads(
            (
                Path("runs/2026-11-18-paper-audit-v1/scoring") / method / "binding.json"
            ).read_text()
        )
        if binding["cases_sha256"] != sha(packet / "cases.json"):
            raise ValueError("scorer packet mismatch")
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
        if any(k not in binding or k not in old or binding[k] != old[k] for k in keys):
            raise ValueError("scorer settings or checkpoint mismatch")
        if method == "geometry" and binding.get("provider", {}).get(
            "provenance"
        ) != old.get("provider", {}).get("provenance"):
            raise ValueError("geometry provenance mismatch")
        predictions[method] = {r["id"]: r for r in records}
    run = Path(endpoint_lock).resolve().parent
    angle_path = run / "camera-angles.csv"
    camera_receipt = json.loads((run / "camera-verification.json").read_text())
    if sha(angle_path) != camera_receipt["angles_csv_sha256"]:
        raise ValueError("camera angle receipt mismatch")
    angle_rows = list(csv.DictReader(angle_path.open()))
    angles = {
        (r["engine"], r["layout"], r["level"], r["camera"], r["slot"]): float(
            r["edge_angle_deg"]
        )
        for r in angle_rows
    }
    if len(angles) != len(angle_rows):
        raise ValueError("duplicate camera angle")
    by_key = {
        (r["engine"], r["arrangement"], r["stop"], r["state"]): r
        for r in private.values()
    }
    expected = {
        (e, a, s, t) for e in ENGINES for a in LAYOUTS for s in STOPS for t in STATES
    }
    if len(by_key) != 192 or set(by_key) != expected:
        raise ValueError("declared conditions mismatch")
    rows = []
    for key, item in sorted(by_key.items()):
        engine, arrangement, stop, state = key
        uid = item["id"]
        public_row = public[uid]
        files = item["files"]
        for source in files.values():
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("native source changed")
        for role in ("reference", "query"):
            path = packet / public_row[role]
            if (
                path.resolve().parent != packet.resolve()
                or sha(path) != public_row[role + "_sha256"]
            ):
                raise ValueError("packet image binding mismatch")
            with Image.open(path) as image:
                if (
                    image.mode != "RGB"
                    or image.size != (1200, 675)
                    or image.info
                    or not np.array_equal(
                        np.asarray(image), primary_rgb(files[role]["path"])
                    )
                ):
                    raise ValueError("delivered RGB differs from native export")
        if public_row["box_xyxy"] != box_from_mask(files["reference_mask"]["path"]):
            raise ValueError("reference box mismatch")
        if state.endswith("removed") and mask(files["query_mask"]["path"]).any():
            raise ValueError("removed target mask not empty")
        retained = by_key[
            (engine, arrangement, stop, state.replace("removed", "present"))
        ]
        visible = mask(retained["files"]["query_mask"]["path"])
        alone = mask(files["alone_mask"]["path"])
        visibility = exposure(visible, alone)
        truth = "REMOVED" if state.endswith("removed") else "PRESENT"
        for method, scores in predictions.items():
            score = scores[uid]
            if any(
                score[k] != public_row[k]
                for k in ("box_xyxy", "reference_sha256", "query_sha256")
            ):
                raise ValueError("score input binding mismatch")
            if score["answer"] not in ("PRESENT", "REMOVED", "UNKNOWN"):
                raise ValueError("invalid score answer")
            rows.append(
                {
                    "engine": engine,
                    "arrangement": arrangement,
                    "target": item.get("target"),
                    "target_bop_id": item.get("target_bop_id"),
                    "reference_feature_angle_deg": angles[
                        (engine, arrangement, "D0", "nominal", stop)
                    ],
                    "query_feature_angle_deg": angles[
                        (engine, arrangement, "D6", "revisit", stop)
                    ],
                    "stop": stop,
                    "state": state,
                    "id": uid,
                    "method": method,
                    "truth": truth,
                    "answer": score["answer"],
                    "correct": score["answer"] == truth,
                    "call_type": call_type(score),
                    "raw": score.get("raw"),
                    "failure": score.get("failure"),
                    "valid_fraction": score.get("valid_fraction"),
                    "reference_pixels": int(
                        mask(files["reference_mask"]["path"]).sum()
                    ),
                    "retained_pixels": visibility["visible_pixels"],
                    "isolated_pixels": visibility["alone_pixels"],
                    "retained_fraction": visibility["visible_fraction_of_alone"],
                    "visible_outside_isolated": visibility[
                        "visible_outside_alone_pixels"
                    ],
                    "raw_score": json.dumps(score, sort_keys=True),
                }
            )
    decisions = {
        (r["engine"], r["arrangement"], r["stop"], r["state"], r["method"]): r
        for r in rows
    }
    pairs = []
    for engine in ENGINES:
        for arrangement in LAYOUTS:
            for stop in STOPS:
                for placement in ("clear", "placed"):
                    a = by_key[(engine, arrangement, stop, placement + "_present")]
                    b = by_key[(engine, arrangement, stop, placement + "_removed")]
                    ca, cb = public[a["id"]], public[b["id"]]
                    native_equal = decoded_hash(
                        a["files"]["query"]["path"]
                    ) == decoded_hash(b["files"]["query"]["path"])
                    query_equal = decoded_hash(packet / ca["query"]) == decoded_hash(
                        packet / cb["query"]
                    )
                    ref_equal = decoded_hash(packet / ca["reference"]) == decoded_hash(
                        packet / cb["reference"]
                    )
                    delivered_complete_equal = (
                        query_equal and ref_equal and ca["box_xyxy"] == cb["box_xyxy"]
                    )
                    native_ref_equal = decoded_hash(
                        a["files"]["reference"]["path"]
                    ) == decoded_hash(b["files"]["reference"]["path"])
                    complete_equal = (
                        native_equal and native_ref_equal and delivered_complete_equal
                    )
                    for method in methods:
                        x = decisions[
                            (engine, arrangement, stop, placement + "_present", method)
                        ]
                        y = decisions[
                            (engine, arrangement, stop, placement + "_removed", method)
                        ]
                        pairs.append(
                            {
                                "engine": engine,
                                "arrangement": arrangement,
                                "target": x["target"],
                                "target_bop_id": x["target_bop_id"],
                                "reference_feature_angle_deg": x[
                                    "reference_feature_angle_deg"
                                ],
                                "query_feature_angle_deg": x["query_feature_angle_deg"],
                                "stop": stop,
                                "placement": placement,
                                "method": method,
                                "present_id": a["id"],
                                "removed_id": b["id"],
                                "present_answer": x["answer"],
                                "removed_answer": y["answer"],
                                "present_call_type": x["call_type"],
                                "removed_call_type": y["call_type"],
                                "visible_pixels": x["retained_pixels"],
                                "visible_fraction": x["retained_fraction"],
                                "native_query_equal": native_equal,
                                "delivered_query_equal": query_equal,
                                "reference_equal": ref_equal,
                                "native_reference_equal": native_ref_equal,
                                "delivered_complete_input_equal": delivered_complete_equal,
                                "box_equal": ca["box_xyxy"] == cb["box_xyxy"],
                                "complete_input_equal": complete_equal,
                                "commits": x["answer"] != "UNKNOWN"
                                or y["answer"] != "UNKNOWN",
                            }
                        )
    groups = defaultdict(list)
    for row in rows:
        groups[(row["engine"], row["stop"], row["state"], row["method"])].append(row)
    summary = []
    for key, group in sorted(groups.items()):
        calls = Counter(r["call_type"] for r in group)
        summary.append(
            dict(
                zip(("engine", "stop", "state", "method"), key),
                n=len(group),
                correct=sum(r["correct"] for r in group),
                definite=sum(r["answer"] != "UNKNOWN" for r in group),
                coverage=sum(r["answer"] != "UNKNOWN" for r in group) / len(group),
                correct_all=sum(r["correct"] for r in group) / len(group),
                wrong_definite=sum(
                    r["answer"] not in ("UNKNOWN", r["truth"]) for r in group
                ),
                fully_visible_retained=sum(
                    r["truth"] == "PRESENT"
                    and r["retained_fraction"] is not None
                    and r["retained_fraction"] >= 0.95
                    for r in group
                ),
                fully_visible_false_removal=sum(
                    r["truth"] == "PRESENT"
                    and r["retained_fraction"] is not None
                    and r["retained_fraction"] >= 0.95
                    and r["answer"] == "REMOVED"
                    for r in group
                ),
                calls=dict(calls),
            )
        )
    pair_groups = defaultdict(list)
    for pair in pairs:
        pair_groups[
            (pair["engine"], pair["stop"], pair["placement"], pair["method"])
        ].append(pair)
    pair_summary = []
    for key, group in sorted(pair_groups.items()):
        exact = [r for r in group if r["complete_input_equal"]]
        pair_summary.append(
            dict(
                zip(("engine", "stop", "placement", "method"), key),
                n=len(group),
                zero_visible=sum(r["visible_pixels"] == 0 for r in group),
                partially_visible=sum(
                    r["visible_fraction"] is not None and 0 < r["visible_fraction"] < 1
                    for r in group
                ),
                fully_visible=sum(r["visible_fraction"] == 1 for r in group),
                unequal_pairs=sum(not r["complete_input_equal"] for r in group),
                zero_visible_but_unequal=sum(
                    r["visible_pixels"] == 0 and not r["complete_input_equal"]
                    for r in group
                ),
                exact_pairs=len(exact),
                committing_exact_pairs=sum(r["commits"] for r in exact),
                both_removed_exact_pairs=sum(
                    r["present_answer"] == r["removed_answer"] == "REMOVED"
                    for r in exact
                ),
            )
        )
    qwen = {
        (r["engine"], r["arrangement"], r["stop"]): r
        for r in pairs
        if r["method"] == "vggt_qwen" and r["placement"] == "placed"
    }
    opportunities = [
        qwen[("mujoco", a, "P1")]
        for a in LAYOUTS
        if qwen[("mujoco", a, "P1")]["complete_input_equal"]
    ]
    recurring = [
        qwen[("mujoco", a, "P1")]
        for a in LAYOUTS
        if qwen[("mujoco", a, "P1")]["complete_input_equal"]
        and qwen[("mujoco", a, "P1")]["present_answer"]
        == qwen[("mujoco", a, "P1")]["removed_answer"]
        == "REMOVED"
    ]
    roll = []
    for original in recurring:
        leveled = qwen[("mujoco", original["arrangement"], "P2")]
        roll.append(
            {
                "arrangement": original["arrangement"],
                "P2_complete_input_equal": leveled["complete_input_equal"],
                "P2_present_call_type": leveled["present_call_type"],
                "P2_removed_call_type": leveled["removed_call_type"],
                "P2_gate_unknown_both": leveled["complete_input_equal"]
                and leveled["present_call_type"]
                == leveled["removed_call_type"]
                == "registration_gate_unknown",
            }
        )
    target_groups = defaultdict(list)
    for row in rows:
        target_groups[
            (row["engine"], row["target"], row["stop"], row["state"], row["method"])
        ].append(row)
    target_summary = []
    for key, group in sorted(target_groups.items()):
        target_summary.append(
            dict(
                zip(("engine", "target", "stop", "state", "method"), key),
                n=len(group),
                correct=sum(r["correct"] for r in group),
                definite=sum(r["answer"] != "UNKNOWN" for r in group),
                false_removals=sum(
                    r["truth"] == "PRESENT" and r["answer"] == "REMOVED" for r in group
                ),
            )
        )
    paired_index = {
        (r["engine"], r["arrangement"], r["stop"], r["placement"], r["method"]): r
        for r in pairs
    }
    pose_comparisons = []
    for engine in ENGINES:
        for arrangement in LAYOUTS:
            for placement in ("clear", "placed"):
                for method in methods:
                    original = paired_index[
                        (engine, arrangement, "P1", placement, method)
                    ]
                    leveled = paired_index[
                        (engine, arrangement, "P2", placement, method)
                    ]
                    pose_comparisons.append(
                        {
                            "engine": engine,
                            "arrangement": arrangement,
                            "target": original["target"],
                            "placement": placement,
                            "method": method,
                            "P1_query_feature_angle_deg": original[
                                "query_feature_angle_deg"
                            ],
                            "P2_query_feature_angle_deg": leveled[
                                "query_feature_angle_deg"
                            ],
                            "P1_present": original["present_answer"],
                            "P1_removed": original["removed_answer"],
                            "P1_present_call": original["present_call_type"],
                            "P1_removed_call": original["removed_call_type"],
                            "P1_exact": original["complete_input_equal"],
                            "P1_visible_pixels": original["visible_pixels"],
                            "P2_present": leveled["present_answer"],
                            "P2_removed": leveled["removed_answer"],
                            "P2_present_call": leveled["present_call_type"],
                            "P2_removed_call": leveled["removed_call_type"],
                            "P2_exact": leveled["complete_input_equal"],
                            "P2_visible_pixels": leveled["visible_pixels"],
                        }
                    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in (
        ("decisions.csv", rows),
        ("pairs.csv", pairs),
        ("pose-comparisons.csv", pose_comparisons),
    ):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "case_count": len(cases),
                "layout_count_per_engine": len(LAYOUTS),
                "groups": summary,
                "pair_groups": pair_summary,
                "target_groups": target_summary,
                "primary_muJoCo_P1_exact_opportunity_layouts": [
                    r["arrangement"] for r in opportunities
                ],
                "primary_muJoCo_P1_both_removed_exact_layouts": [
                    r["arrangement"] for r in recurring
                ],
                "primary_recurrence": recurrence_label(
                    len(opportunities), len(recurring)
                ),
                "conditional_roll": roll,
                "cases_sha256": sha(packet / "cases.json"),
                "endpoint_lock_sha256": sha(endpoint_lock),
                "records_sha256": {k: sha(v) for k, v in methods.items()},
                "evaluator_sha256": sha(__file__),
                "note": "Descriptive transfer panel; retain layout clusters. Zero visible pixels alone do not establish identical inputs.",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--endpoint-lock", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluate(
        args.packet,
        args.ledger,
        dict(x.split("=", 1) for x in args.method),
        args.endpoint_lock,
        args.out,
    )
