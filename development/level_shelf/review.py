"""Independently check the reported decisions and pool the two oblique stops by treatment."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from PIL import Image


TREATMENT = {"P1": "oblique", "P2": "edge_leveled", "P3": "frontal", "P4": "oblique", "P5": "edge_leveled"}
METHODS = ("baseline_200", "geometry", "vggt_qwen")
ARCHIVE = Path(__file__).resolve().parents[2] / "paper/evidence/raw-scores"
PAPER_METHOD = {"baseline_200": "homography", "geometry": "dense", "vggt_qwen": "registered"}


@lru_cache(maxsize=None)
def pixels(path):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def review(run, released, destination):
    with (run / "evaluation/scorer-ready.json").open() as f:
        release = json.load(f)
    cases_file = Path(release["cases_path"])
    ledger_file = Path(release["private_ledger_path"])
    cases = {r["id"]: r for r in json.loads(cases_file.read_text())}
    ledger = {r["id"]: r for r in json.loads(ledger_file.read_text())["rows"]}
    scores = {}
    for method in METHODS:
        path = run / "scoring" / method / "records.jsonl"
        binding = json.loads((path.parent / "binding.json").read_text())
        old_binding = json.loads((ARCHIVE / PAPER_METHOD[method] / "binding.json").read_text())
        if binding.pop("cases_sha256") != hashlib.sha256(cases_file.read_bytes()).hexdigest():
            raise ValueError("Scorer input binding differs")
        old_binding.pop("cases_sha256")
        if binding != old_binding:
            raise ValueError(f"{method} scorer differs from the original run")
        scores[method] = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
        if scores[method].keys() != cases.keys():
            raise ValueError("Different scorer IDs")
    if cases.keys() != ledger.keys() or len(cases) != 320:
        raise ValueError("Unexpected case inventory")
    evaluated = {(r["id"], r["method"]): r for r in read_csv(released / "decisions.csv")}
    if len(evaluated) != 960:
        raise ValueError("Missing evaluator rows")
    pairs = defaultdict(dict)
    indexed = {}
    counts = defaultdict(Counter)
    archived_replay = defaultdict(Counter)
    for uid, r in ledger.items():
        engine, layout, slot, state = (r[k] for k in ("engine", "arrangement", "stop", "state"))
        placement, presence = state.split("_")
        truth = "PRESENT" if presence == "present" else "REMOVED"
        pairs[engine, layout, slot, placement][presence] = uid
        indexed[engine, layout, slot, state] = uid
        for method in METHODS:
            record = scores[method][uid]
            decision = record["answer"]
            expected = evaluated[uid, method]
            if (decision, truth, engine, layout, slot, state) != tuple(expected[k] for k in
                    ("answer", "truth", "engine", "arrangement", "stop", "state")):
                raise ValueError("Independent decision recount differs")
            technical_failure = record["failure"] not in (None, False, "registration_gate")
            if technical_failure and decision != "UNKNOWN":
                raise ValueError(f"Definite answer recorded after an inference failure for {uid} {method}")
            if TREATMENT[slot] == "oblique":
                replay = archived_replay[engine, method]
                replay["query_count"] += 1
                replay["same_prediction"] += expected["original_answer"] == decision
                replay["same_complete_input"] += all(expected[k] == "True" for k in
                        ("original_reference_decoded_equal", "original_query_decoded_equal", "original_box_equal"))
            group = counts[engine, TREATMENT[slot], method]
            group["queries"] += 1
            group["technical_failures"] += technical_failure
            group["correct"] += decision == truth
            group["definite"] += decision != "UNKNOWN"
            group["retained"] += truth == "PRESENT"
            group["false_removals"] += truth == "PRESENT" and decision == "REMOVED"
            group["clear_false_removals"] += state == "clear_present" and decision == "REMOVED"
            group["false_presences"] += truth == "REMOVED" and decision == "PRESENT"
            group["clear_removals"] += state == "clear_removed"
            group["correct_clear_removals"] += state == "clear_removed" and decision == "REMOVED"
            if method == "vggt_qwen":
                group["upstream_unknown"] += record["failure"] == "registration_gate"
                group["qwen_unknown"] += record["raw"] == "UNKNOWN"
    evaluated_pairs = {(r["present_id"], r["removed_id"], r["method"]): r for r in read_csv(released / "pairs.csv")}
    if len(pairs) != 160 or len(evaluated_pairs) != 480:
        raise ValueError("Missing pair rows")
    for (engine, layout, slot, placement), members in pairs.items():
        a, b = members["present"], members["removed"]
        native = all(pixels(ledger[a]["files"][k]["path"]) == pixels(ledger[b]["files"][k]["path"])
                     for k in ("reference", "query"))
        same = all(pixels(cases_file.parent / cases[a][k]) == pixels(cases_file.parent / cases[b][k])
                   for k in ("reference", "query")) and cases[a]["box_xyxy"] == cases[b]["box_xyxy"]
        for method in METHODS:
            evaluated_pair = evaluated_pairs[a, b, method]
            if same != (evaluated_pair["complete_input_equal"] == "True") or native != (evaluated_pair["native_query_equal"] == "True"):
                raise ValueError("Independent pixel certificate differs")
            decisions = [scores[method][uid]["answer"] for uid in (a, b)]
            if same and decisions[0] != decisions[1]:
                raise ValueError("Different decisions on equal scorer inputs")
            group = counts[engine, TREATMENT[slot], method]
            group["matched_pairs"] += 1
            group["exact_pairs"] += same
            group["exact_pair_commitments"] += same and decisions[0] != "UNKNOWN"
            if method == "vggt_qwen" and same:
                group["exact_pairs_passed_to_qwen"] += scores[method][a]["failure"] != "registration_gate"
    transitions = defaultdict(Counter)
    for (engine, layout, src, state), uid in indexed.items():
        if src not in ("P1", "P4") or not state.endswith("_present"):
            continue
        dst = "P2" if src == "P1" else "P5"
        other = indexed[engine, layout, dst, state]
        for method in METHODS:
            before = scores[method][uid]["answer"]
            after = scores[method][other]["answer"]
            changes = transitions[engine, method]
            changes["retained_comparisons"] += 1
            changes["false_removal_both"] += before == "REMOVED" and after == "REMOVED"
            changes["false_removal_to_present"] += before == "REMOVED" and after == "PRESENT"
            changes["false_removal_to_unknown"] += before == "REMOVED" and after == "UNKNOWN"
            changes["new_false_removal"] += before != "REMOVED" and after == "REMOVED"
    groups = [{"engine": e, "treatment": t, "method": m, **values} for (e, t, m), values in sorted(counts.items())]
    result = {"checked_queries": len(cases), "checked_decisions": len(evaluated), "checked_pairs": len(pairs),
              "all_three_scorer_bindings_match_original_except_input_inventory": True,
              "verified_evaluator_decisions_and_input_certificates": True, "groups": groups,
              "archived_replay": [{"engine": e, "method": m, **v} for (e, m), v in sorted(archived_replay.items())],
              "paired_effect_of_roll": [{"engine": e, "method": m, **v} for (e, m), v in sorted(transitions.items())]}
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(evaluated)} decisions and {len(pairs)} opposite-state pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("released", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    review(args.run, args.released, args.destination)
