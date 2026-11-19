"""Join frozen blind decisions to physical interventions and compare the matched corridors."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from development.corridor_transfer.evaluate import call_type, recurrence_label
from development.workshop_realism.prepare import csv_file
from development.workshop_realism.release import (
    HERE,
    METHODS,
    OLD,
    PROJECT,
    read,
    require,
    sha,
    write_once,
)

PRIOR_PAIRS = PROJECT / "notes/figures/2026-11-18-corridor-transfer/pairs.csv"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def assert_index(rows, expected):
    result = {row["id"]: row for row in rows}
    require(
        len(result) == len(rows) == expected,
        "Case identifiers are not unique or complete",
    )
    return result


def compare(old, new):
    was = (
        old["complete_input_equal"] == "True"
        and old["present_answer"] == old["removed_answer"] == "REMOVED"
    )
    now = new["exact"] and new["present_answer"] == new["removed_answer"] == "REMOVED"
    if was:
        if now:
            return "persistent"
        return (
            "lost_exact_opportunity" if not new["exact"] else "resolved_on_exact_input"
        )
    return "new" if now else "neither"


def select(rows, engine, method, view=None):
    return [
        r
        for r in rows
        if r["engine"] == engine
        and r["method"] == method
        and (view is None or r["view"] == view)
    ]


def statistics(rows, paired, method):
    exposed = [
        r for r in paired if r["door"] == "clear" and r["visible_fraction"] >= 0.95
    ]
    covered = [r for r in paired if r["door"] == "placed"]
    exact = [r for r in covered if r["exact"]]
    c = Counter(r["call_type"] for r in rows)
    return {
        "queries": len(rows),
        "definite": sum(r["answer"] != "UNKNOWN" for r in rows),
        "correct": sum(r["correct"] for r in rows),
        "retained": sum(r["truth"] == "PRESENT" for r in rows),
        "false_removals": sum(
            r["answer"] == "REMOVED" and r["truth"] == "PRESENT" for r in rows
        ),
        "open_exposed": len(exposed),
        "open_retained_present": sum(r["present_answer"] == "PRESENT" for r in exposed),
        "open_retained_false_removals": sum(
            r["present_answer"] == "REMOVED" for r in exposed
        ),
        "open_removals_detected": sum(
            r["removed_answer"] == "REMOVED" for r in exposed
        ),
        "closed_pairs": len(covered),
        "closed_zero_mask": sum(r["target_pixels"] == 0 for r in covered),
        "closed_retained_false_removals": sum(
            r["present_answer"] == "REMOVED" for r in covered
        ),
        "closed_exact": len(exact),
        "closed_exact_double_removed": sum(r["double_removed"] for r in exact),
        "closed_exact_double_present": sum(
            r["present_answer"] == r["removed_answer"] == "PRESENT" for r in exact
        ),
        "closed_exact_double_unknown": sum(
            r["present_answer"] == r["removed_answer"] == "UNKNOWN" for r in exact
        ),
        "closed_exact_mixed_answers": sum(
            r["present_answer"] != r["removed_answer"] for r in exact
        ),
        "registration_stops": c["registration_gate_unknown"]
        if method == "vggt_qwen"
        else None,
        "model_calls_with_output": sum(r["raw"] is not None for r in rows)
        if method == "vggt_qwen"
        else None,
        "model_unknown": c["qwen_unknown"] if method == "vggt_qwen" else None,
        "invalid_outputs": c["invalid_model_output"],
        "technical_failures": c["execution_failure"],
    }


def evaluate(run, out):
    packet = run / "scoring/fresh-packet"
    qualification = run / "qualification"
    ready = read(qualification / "scorer-ready.json")
    lock = read(run / "endpoint-lock.json")
    prior = read_csv(PRIOR_PAIRS)
    earlier_registered = assert_index(
        [
            json.loads(line)
            for line in (OLD / "scoring/vggt_qwen/records.jsonl")
            .read_text()
            .splitlines()
        ],
        192,
    )
    require(
        sha(PRIOR_PAIRS) == lock["prior_pairs_sha256"],
        "Earlier corridor evidence changed",
    )
    for name, path in [
        ("cases", packet / "cases.json"),
        ("private_ledger", qualification / "private-ledger.json"),
        ("pairs", qualification / "input-pairs.csv"),
        ("design_release", run / "design-release.json"),
        ("cameras", qualification / "measured-cameras.csv"),
    ]:
        require(
            ready[name + "_sha256"] == sha(path), "Different pre-score file: " + name
        )
    require(
        ready["task_sha256"]
        == lock["original_task_sha256"]
        == sha(PROJECT / "paper/evidence/design/specification.json"),
        "Different task",
    )
    public = assert_index(read(packet / "cases.json"), 320)
    private = assert_index(read(qualification / "private-ledger.json"), 320)
    require(public.keys() == private.keys(), "Missing private state")
    inputs = read_csv(qualification / "input-pairs.csv")
    require(len(inputs) == 160, "Missing intervention pair")
    decisions = []
    table = []
    binding_checks = {}
    for method, original_name in METHODS.items():
        folder = run / "scoring" / method
        original = (
            PROJECT / "paper/evidence/raw-scores" / original_name / "binding.json"
        )
        require(
            sha(original) == lock["original_bindings"][original_name],
            "Paper binding changed",
        )
        reference_binding = read(original)
        binding = read(folder / "binding.json")
        require(
            binding.pop("cases_sha256") == ready["cases_sha256"],
            "Different model packet",
        )
        reference_binding.pop("cases_sha256")
        require(binding == reference_binding, "Different scorer settings")
        binding_checks[method] = {
            "settings_match_paper": True,
            "records_sha256": sha(folder / "records.jsonl"),
            "binding_sha256": sha(folder / "binding.json"),
        }
        outputs = assert_index(
            [
                json.loads(l)
                for l in (folder / "records.jsonl").read_text().splitlines()
            ],
            320,
        )
        require(outputs.keys() == public.keys(), "Missing replies")
        for uid, row in outputs.items():
            case = public[uid]
            meta = private[uid]
            answer = row["answer"]
            state = meta["state"]
            require(
                all(
                    case[k] == row[k]
                    for k in ("query_sha256", "reference_sha256", "box_xyxy")
                )
                and case["box_xyxy"] == meta["box_xyxy"],
                "Scored input differs",
            )
            require(answer in ("PRESENT", "REMOVED", "UNKNOWN"), "Unexpected answer")
            truth = state.partition("_")[2].upper()
            decisions.append(
                {
                    "engine": meta["engine"],
                    "layout": meta["layout"],
                    "target": meta["target"],
                    "view": meta["view"],
                    "state": state,
                    "truth": truth,
                    "id": uid,
                    "method": method,
                    "answer": answer,
                    "correct": answer == truth,
                    "call_type": call_type(row),
                    "raw": row.get("raw"),
                    "failure": row.get("failure"),
                    "valid_fraction": row.get("valid_fraction"),
                    "visible": meta["visible"],
                    "isolated": meta["isolated"],
                    "visible_overlap": meta["overlap"],
                    "native_reference_pixels_sha256": meta["native_reference"],
                    "native_query_pixels_sha256": meta["native_query"],
                    "reference_png_sha256": case["reference_sha256"],
                    "query_png_sha256": case["query_sha256"],
                    "box_xyxy": case["box_xyxy"],
                }
            )
        for row in inputs:
            p = outputs[row["present_id"]]
            g = outputs[row["removed_id"]]
            pair = {
                **row,
                "method": method,
                **{
                    k: row[k] == "True"
                    for k in ("exact", "native_exact", "delivered_exact")
                },
                **{
                    k: int(row[k])
                    for k in (
                        "target_pixels",
                        "isolated_pixels",
                        "changed_native_pixels",
                    )
                },
                "visible_fraction": float(row["visible_fraction"]),
                "present_answer": p["answer"],
                "removed_answer": g["answer"],
                "present_usable_fraction": p.get("valid_fraction"),
                "removed_usable_fraction": g.get("valid_fraction"),
                "present_call": call_type(p),
                "removed_call": call_type(g),
                "double_removed": p["answer"] == g["answer"] == "REMOVED",
            }
            table.append(pair)
    old = {
        (r["engine"], r["arrangement"], r["stop"], r["placement"], r["method"]): r
        for r in prior
    }
    matched = []
    for r in table:
        if r["view"] not in ("P1", "P2", "P3"):
            continue
        keys = ("engine", "layout", "view", "door", "method")
        o = old[tuple(r[k] for k in keys)]
        matched.append(
            {
                **{k: r[k] for k in keys},
                "old_exact": o["complete_input_equal"],
                "new_exact": r["exact"],
                "old_present": o["present_answer"],
                "old_removed": o["removed_answer"],
                "new_present": r["present_answer"],
                "new_removed": r["removed_answer"],
                "old_present_usable_fraction": earlier_registered[o["present_id"]][
                    "valid_fraction"
                ]
                if r["method"] == "vggt_qwen"
                else None,
                "new_present_usable_fraction": r["present_usable_fraction"],
                "old_removed_usable_fraction": earlier_registered[o["removed_id"]][
                    "valid_fraction"
                ]
                if r["method"] == "vggt_qwen"
                else None,
                "new_removed_usable_fraction": r["removed_usable_fraction"],
                "old_present_call": o["present_call_type"],
                "new_present_call": r["present_call"],
                "old_removed_call": o["removed_call_type"],
                "new_removed_call": r["removed_call"],
                "double_removed_change": compare(o, r),
            }
        )
    require(len(matched) == len(old) == 288, "Missing matched decision")
    prior_primary = [
        r
        for r in matched
        if (r["engine"], r["view"], r["door"], r["method"])
        == ("mujoco", "P3", "placed", "vggt_qwen")
    ]
    original_events = sorted(
        r["layout"]
        for r in prior_primary
        if r["old_exact"] == "True"
        and r["old_present"] == r["old_removed"] == "REMOVED"
    )
    require(
        original_events == read(HERE / "declaration.json")["prior_frontal_events"],
        "Earlier primary counts do not match declaration",
    )
    events = sorted(
        r["layout"]
        for r in prior_primary
        if r["new_exact"] and r["new_present"] == r["new_removed"] == "REMOVED"
    )
    eligible = sum(r["new_exact"] for r in prior_primary)
    primary = {
        "layouts": len(prior_primary),
        "exact_opportunities": eligible,
        "double_removed": len(events),
        "event_layouts": events,
        "classification": recurrence_label(eligible, len(events)),
        "plain_events": original_events,
        "paired_transitions": {
            k: sorted(
                r["layout"] for r in prior_primary if r["double_removed_change"] == k
            )
            for k in (
                "persistent",
                "resolved_on_exact_input",
                "lost_exact_opportunity",
                "new",
                "neither",
            )
        },
    }
    result = []
    pooled = []
    for engine in ("isaac", "mujoco"):
        for method in METHODS:
            pooled.append(
                {
                    "engine": engine,
                    "method": method,
                    **statistics(
                        select(decisions, engine, method),
                        select(table, engine, method),
                        method,
                    ),
                }
            )
            for view in ("P1", "P2", "P3", "P4", "P5"):
                result.append(
                    {
                        "engine": engine,
                        "view": view,
                        "method": method,
                        **statistics(
                            select(decisions, engine, method, view),
                            select(table, engine, method, view),
                            method,
                        ),
                    }
                )
    decisions.sort(
        key=lambda r: tuple(
            r[k] for k in ("engine", "layout", "view", "state", "method")
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    csv_file(out / "decisions.csv", decisions)
    csv_file(out / "pairs.csv", table)
    csv_file(out / "method-by-view.csv", result)
    csv_file(out / "paired-changes.csv", matched)
    write_once(
        out / "summary.json",
        {
            "scope": "same eight layouts in two simulated rooms",
            "cases": len(public),
            "decisions": len(decisions),
            "opposite_pairs": len(inputs),
            "primary": primary,
            "all_views": pooled,
            "original_binding_checks": binding_checks,
            "input_cases_sha256": ready["cases_sha256"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    evaluate(args.run, args.out)
