"""Cross-check the independent evaluator against the portable paper tables."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    if path.is_file():
        stream = path.open(newline="")
    else:
        stream = gzip.open(path.with_suffix(".csv.gz"), "rt", newline="")
    with stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def verify(evidence, official, names):
    study = read_json(evidence / "summary.json")
    export = read_json(evidence / "export-receipt.json")
    evaluator = read_json(official / "summary.json")
    names = dict(names)
    for ours, theirs in names.items():
        check(export["score_sha256"][ours] == evaluator["scorer_jsonl_sha256"][theirs],
              "Raw prediction streams differ")
    by_name = {theirs: ours for ours, theirs in names.items()}
    queries = {r["id"]: r for r in read_csv(evidence / "queries.csv")}
    pairs = {(r["present_id"], r["removed_id"]): r for r in read_csv(evidence / "pairs.csv")}
    checked = set()
    for decision in read_csv(official / "cases.csv"):
        uid = decision["id"]
        q = queries[uid]
        method = decision["method"]
        key = uid, method
        check(key not in checked, f"Repeated decision {key}")
        checked.add(key)
        for name in ("engine", "environment", "arrangement", "level", "state", "stop", "truth"):
            check(q[name] == decision[name], f"Case {uid} differs on {name}")
        for role in ("reference", "query"):
            check(q[role + "_file_sha256"] == decision[role + "_sha256"], "File digest differs")
        check(json.loads(q["box"]) == json.loads(decision["box_xyxy"]), "Reference box differs")
        if q["truth"] == "PRESENT":
            visible, alone = int(q["visible_pixels"]), int(q["isolated_pixels"])
            overlap = int(q["visible_isolated_overlap_pixels"])
            check((visible, alone, visible - overlap) ==
                  (int(decision["visible_pixels"]), int(decision["alone_pixels"]), int(decision["visible_outside_alone_pixels"])),
                  "Native exposed and isolated masks differ")
        if method in by_name:
            expected = q[by_name[method]]
        else:
            expected = {"always_UNKNOWN": "UNKNOWN", "always_REMOVED": "REMOVED"}[method]
        check(decision["answer"] == expected, f"Decision differs for {key}")
    check(len(checked) == len(queries) * (len(names) + 2), "Missing case decisions")
    pair_keys = set()
    for pair in read_csv(official / "pairs.csv"):
        key = pair["present_id"], pair["removed_id"]
        check(key not in pair_keys, "Repeated pair")
        pair_keys.add(key)
        p = pairs[key]
        check((pair["complete_input_equal"] == "True") == (p["exact"] == "1"), "Certificate differs")
        raw_equal = pair["native_reference_equal"] == "True" and pair["native_query_equal"] == "True"
        check(raw_equal == (p["native_exact"] == "1"), "Native comparison differs")
    check(pair_keys == pairs.keys(), "Missing pair")
    primary_checks = []
    for row in read_json(official / "paired_comparisons.json"):
        methods = row["first"], row["second"]
        if (set(methods) != {names["homography"], names["dense"]}
                or row["level"] != "D6" or row["state"] != "clear_removed"):
            continue
        order = 1 if methods[1] == names["dense"] else -1
        p = study["engines"][row["engine"]]["environments"][row["environment"]]["primary"]
        gain_key, loss_key = ("second_gained", "second_lost") if order == 1 else ("second_lost", "second_gained")
        check((p["n"], p["gained"], p["lost"]) == (row["n"], row[gain_key], row[loss_key]),
              "Paired endpoint differs")
        expected = [p["delta_pp"], *p["delta_interval_pp"]]
        boot_interval = sorted(x * 100 * order for x in row["bootstrap_95"])
        actual = [row["accuracy_difference_second_minus_first"] * 100 * order, *boot_interval]
        check(all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(actual, expected)),
              "Paired estimate or resampling interval differs")
        primary_checks.append([row["engine"], row["environment"]])
    check(len(set(map(tuple, primary_checks))) == 6, "Missing primary comparison")
    criteria = read_json(official / "criteria.json")
    for result in criteria["engines"]:
        engine = study["engines"][result["engine"]]
        overall = engine["environments"]["all"]
        table_net = engine["environments"]["tabletop"]["primary"]["layout_net"]
        cert = engine["certificates"]
        check(result["primary_gain_correct_cases"] == overall["primary"]["net"]
              and sorted(result["tabletop_gain_layouts"]) == sorted(k for k, v in table_net.items() if v > 0)
              and result["newly_wrong_definite_all_cases"] == overall["overall"]["newly_wrong"]
              and result["certified_pairs"] == cert["exact"]
              and result["geometry_committing_certified_pairs"] == cert["dense"]["commit"]
              and result["zero_certified_commitments"] == (cert["dense"]["commit"] == 0 if cert["exact"] else None),
              "Predeclared improvement criterion differs")
    return {
        "queries": len(queries),
        "method_decisions": len(queries) * len(names),
        "analytic_control_decisions": len(queries) * 2,
        "native_and_delivered_pairs": len(pair_keys),
        "paired_estimates_and_intervals": sorted(primary_checks),
        "improvement_criteria_checked": sorted(r["engine"] for r in criteria["engines"]),
        "score_sha256": export["score_sha256"],
        "portable_summary_sha256": digest(evidence / "summary.json"),
        "evaluator_summary_sha256": digest(official / "summary.json"),
        "result": "All decision, input, pair and primary interval comparisons agreed."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("evaluator", type=Path)
    for flag in ("homography", "dense", "registered"):
        parser.add_argument("--" + flag, required=True, help="Method name in the evaluator report")
    args = parser.parse_args()
    labels = [(m, getattr(args, m)) for m in ("homography", "dense", "registered")]
    print(json.dumps(verify(args.evidence, args.evaluator, labels), indent=2))
