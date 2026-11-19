"""Recount saved answers and matched images from the original scorer records."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

METHODS = {"baseline_200": "homography", "geometry": "dense", "vggt_qwen": "registered"}
PROJECT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@cache
def pixels(path):
    with Image.open(path) as src:
        image = src.convert("RGB")
        return image.size, hashlib.sha256(image.tobytes()).digest()


def mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def require(value, message):
    if not value:
        raise ValueError(message)


def independent_review(run, tables):
    receipt = read(run / "evaluation/scorer-ready.json")
    public_file, private_file = (
        Path(receipt["cases_path"]),
        Path(receipt["private_ledger_path"]),
    )
    require(
        sha(public_file) == receipt["cases_sha256"]
        and sha(private_file) == receipt["private_ledger_sha256"],
        "Input binding differs",
    )
    require(
        sha(run / "endpoint-lock.json") == receipt["endpoint_lock_sha256"],
        "Endpoint lock differs",
    )
    packet = {r["id"]: r for r in read(public_file)}
    ledger = {r["id"]: r for r in read(private_file)["rows"]}
    require(
        len(packet) == 192 and set(packet) == set(ledger), "Input inventories differ"
    )
    out = csv_rows(tables / "decisions.csv")
    per_answer = {(r["id"], r["method"]): r for r in out}
    pair_table = csv_rows(tables / "pairs.csv")
    eval_pairs = {
        (r["present_id"], r["removed_id"], r["method"]): r for r in pair_table
    }
    require(
        len(out) == len(per_answer) == 576
        and len(pair_table) == len(eval_pairs) == 288,
        "Missing decisions or pairs",
    )
    key_to_id = {
        (r["engine"], r["arrangement"], r["stop"], r["state"]): u
        for u, r in ledger.items()
    }
    records = {}
    for method, name in METHODS.items():
        path = run / "scoring" / method
        baseline = read(PROJECT / "paper/evidence/raw-scores" / name / "binding.json")
        binding = read(path / "binding.json")
        require(
            binding.pop("cases_sha256") == sha(public_file), "Wrong input inventory"
        )
        baseline.pop("cases_sha256")
        require(
            binding == baseline,
            f"{method} differs from the paper scoring configuration",
        )
        original = [
            json.loads(line)
            for line in (path / "records.jsonl").read_text().splitlines()
        ]
        records[method] = {r["id"]: r for r in original}
        require(
            len(original) == len(records[method]) == 192
            and set(records[method]) == set(packet),
            "Incomplete scorer",
        )
    counts, per_object = defaultdict(Counter), defaultdict(Counter)
    geometry_comparison = defaultdict(Counter)
    for uid, spec in ledger.items():
        c = packet[uid]
        engine, layout, slot, state = [
            spec[k] for k in ("engine", "arrangement", "stop", "state")
        ]
        truth = "PRESENT" if state.endswith("_present") else "REMOVED"
        homography = records["baseline_200"][uid]["answer"]
        dense = records["geometry"][uid]["answer"]
        comparison = geometry_comparison[engine]
        comparison["queries"] += 1
        comparison["new_wrong_definite"] += dense not in (
            truth,
            "UNKNOWN",
        ) and homography in (truth, "UNKNOWN")
        comparison["old_wrong_definite_avoided"] += homography not in (
            truth,
            "UNKNOWN",
        ) and dense in (truth, "UNKNOWN")
        if state == "clear_removed":
            comparison["clear_removed"] += 1
            comparison["clear_removed_gained"] += dense == truth and homography != truth
            comparison["clear_removed_lost"] += dense != truth and homography == truth
        for method in METHODS:
            score = records[method][uid]
            row = per_answer[uid, method]
            require(
                all(
                    score[k] == c[k]
                    for k in ("reference_sha256", "query_sha256", "box_xyxy")
                ),
                "Score attached to wrong images",
            )
            require(
                row["truth"] == truth
                and row["answer"] == score["answer"]
                and row["state"] == state,
                "Decision differs",
            )
            require(
                (row["engine"], row["arrangement"], row["stop"])
                == (engine, layout, slot),
                "Case location differs",
            )
            group = counts[engine, slot, method]
            group["queries"] += 1
            group["definite"] += score["answer"] != "UNKNOWN"
            group["correct"] += score["answer"] == truth
            group["retained"] += truth == "PRESENT"
            group["false_removals"] += (
                truth == "PRESENT" and score["answer"] == "REMOVED"
            )
            group["clear_removals"] += state == "clear_removed"
            group["clear_removals_correct"] += (
                state == "clear_removed" and score["answer"] == "REMOVED"
            )
            group["clear_retained"] += state == "clear_present"
            group["clear_retained_correct"] += (
                state == "clear_present" and score["answer"] == "PRESENT"
            )
            failure = score.get("failure")
            group["technical_failures"] += bool(failure) and failure not in (
                "registration_gate",
                "invalid_model_output",
            )
            group["invalid_outputs"] += failure == "invalid_model_output"
            if method == "vggt_qwen":
                group["registration_stops"] += failure == "registration_gate"
                group["model_calls"] += score.get("raw") is not None
                group["model_unknown"] += score.get("raw") == "UNKNOWN"
                per_object[engine, spec["target"], slot]["false_removals"] += (
                    truth == "PRESENT" and score["answer"] == "REMOVED"
                )
                per_object[engine, spec["target"], slot]["clear_removals_correct"] += (
                    state == "clear_removed" and score["answer"] == "REMOVED"
                )
    examples, primary, eligible, roll_stops, new_on_roll, exposed_errors = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for (engine, layout, slot, state), uid in sorted(key_to_id.items()):
        if not state.endswith("_present"):
            continue
        placement = state.split("_")[0]
        other = key_to_id[engine, layout, slot, placement + "_removed"]
        retained, removed = ledger[uid]["files"], ledger[other]["files"]
        a, b = packet[uid], packet[other]
        native = all(
            pixels(retained[k]["path"]) == pixels(removed[k]["path"])
            for k in ("reference", "query")
        )
        delivered = all(
            pixels(public_file.parent / a[k]) == pixels(public_file.parent / b[k])
            for k in ("reference", "query")
        )
        exact = native and delivered and a["box_xyxy"] == b["box_xyxy"]
        visible, alone = (
            mask(retained["query_mask"]["path"]),
            mask(retained["alone_mask"]["path"]),
        )
        require(
            alone.any() and not mask(removed["query_mask"]["path"]).any(),
            "Unexpected target intervention",
        )
        fraction = np.count_nonzero(visible & alone) / int(alone.sum())
        for method in METHODS:
            arow, brow = records[method][uid], records[method][other]
            ra, rb = arow["answer"], brow["answer"]
            row = eval_pairs[uid, other, method]
            require(
                row["present_answer"] == ra and row["removed_answer"] == rb,
                "Pair decisions differ",
            )
            require(
                (row["complete_input_equal"] == "True") == exact,
                "Independent equality certificate differs",
            )
            require(
                int(row["visible_pixels"]) == int(visible.sum())
                and abs(float(row["visible_fraction"]) - fraction) < 1e-12,
                "Independent mask measurement differs",
            )
            group = counts[engine, slot, method]
            if placement == "clear":
                group["open_exposed_pairs"] += fraction >= 0.95
                group["open_exposed_false_removal"] += (
                    fraction >= 0.95 and ra == "REMOVED"
                )
                if (
                    slot == "P3"
                    and method == "vggt_qwen"
                    and fraction >= 0.95
                    and ra == "REMOVED"
                ):
                    exposed_errors.append(
                        {
                            "engine": engine,
                            "layout": layout,
                            "target": ledger[uid]["target"],
                            "id": uid,
                        }
                    )
                continue
            group["door_closed"] += 1
            group["closed_zero_mask"] += not visible.any()
            group["closed_exact"] += exact
            group["closed_exact_definite_pairs"] += exact and (
                ra != "UNKNOWN" or rb != "UNKNOWN"
            )
            group["closed_exact_double_removed"] += exact and ra == rb == "REMOVED"
            group["closed_exact_double_present"] += exact and ra == rb == "PRESENT"
            group["closed_exact_double_unknown"] += exact and ra == rb == "UNKNOWN"
            group["closed_exact_discordant"] += exact and ra != rb
            group["closed_double_removed"] += ra == rb == "REMOVED"
            if method == "vggt_qwen":
                group["closed_exact_both_gate"] += (
                    exact
                    and arow.get("failure")
                    == brow.get("failure")
                    == "registration_gate"
                )
                if engine == "mujoco":
                    examples.append(
                        {
                            "layout": layout,
                            "view": slot,
                            "target": ledger[uid]["target"],
                            "exact": exact,
                            "answers": [ra, rb],
                            "gate": arow.get("failure")
                            == brow.get("failure")
                            == "registration_gate",
                        }
                    )
    by_view = {(v["layout"], v["view"]): v for v in examples}
    for layout in sorted({v["layout"] for v in examples}):
        old, leveled = by_view[layout, "P1"], by_view[layout, "P2"]
        if old["exact"]:
            eligible.append(layout)
            if old["answers"] == ["REMOVED", "REMOVED"]:
                primary.append(layout)
                if leveled["exact"] and leveled["gate"]:
                    roll_stops.append(layout)
        if (
            leveled["exact"]
            and leveled["answers"] == ["REMOVED", "REMOVED"]
            and not (old["exact"] and old["answers"] == ["REMOVED", "REMOVED"])
        ):
            new_on_roll.append(layout)
    summary = read(tables / "summary.json")
    require(
        eligible == summary["primary_muJoCo_P1_exact_opportunity_layouts"]
        and primary == summary["primary_muJoCo_P1_both_removed_exact_layouts"],
        "Primary result differs",
    )
    require(
        roll_stops
        == [
            r["arrangement"]
            for r in summary["conditional_roll"]
            if r["P2_gate_unknown_both"]
        ],
        "Conditional roll result differs",
    )
    return {
        "checked_queries": len(packet),
        "checked_method_decisions": len(per_answer),
        "checked_opposite_state_pairs": len(eval_pairs) // 3,
        "all_three_scorer_bindings_match_paper_except_cases": True,
        "primary_eligible": eligible,
        "primary_events": primary,
        "primary_gate_stops_after_roll": roll_stops,
        "new_exact_double_removals_after_roll": new_on_roll,
        "dense_vs_homography": dict(sorted(geometry_comparison.items())),
        "open_frontal_exposed_qwen_false_removals": exposed_errors,
        "mujoco_closed_qwen": examples,
        "groups": [
            dict(engine=e, view=v, method=m, **g)
            for (e, v, m), g in sorted(counts.items())
        ],
        "qwen_object_groups": [
            dict(engine=e, target=t, view=s, **g)
            for (e, t, s), g in sorted(per_object.items())
        ],
        "checked_table_sha256": {
            n: sha(tables / n) for n in ("decisions.csv", "pairs.csv", "summary.json")
        },
        "reviewer_sha256": sha(Path(__file__)),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("tables", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = independent_review(args.run, args.tables)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Verified",
        result["checked_method_decisions"],
        "decisions and",
        result["checked_opposite_state_pairs"],
        "pairs.",
    )
