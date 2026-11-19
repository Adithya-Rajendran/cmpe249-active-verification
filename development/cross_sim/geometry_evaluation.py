"""Evaluate the registered RGB comparison after the complete candidate run."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
from development.cross_sim.paired_intervals import bootstrap_indices, interval
from development.stress.prepare_inputs import load, sha


PROTOCOL = Path(__file__).with_name("geometry_protocol.json")
ENGINES = ("isaac", "mujoco")
ENVIRONMENTS = ("shelf", "tabletop")
LEVELS = ("D0", "D4", "D6")
STATES = ("clear_present", "clear_removed", "placed_present", "placed_removed")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bound(path, expected_hash):
    require(sha(path) == expected_hash, f"Changed file: {path}")
    return load(path)


def index_by_id(rows, expected_count):
    indexed = {row["id"]: row for row in rows}
    require(len(indexed) == len(rows) == expected_count, "Duplicate or missing cases")
    return indexed


def decision(record, thresholds):
    evidence = record["evidence"]
    align_ok = record["diagnostics"]["align_ok"]
    require(type(align_ok) is bool and record["failure"] is False, "Invalid case result")
    q, match, obstruction = (evidence[key] for key in ("quality", "match_score", "obstruction"))
    require(all(math.isfinite(x) and 0 <= x <= 1 for x in (q, match, obstruction)), "Invalid evidence")
    if not align_ok:
        return "UNKNOWN", record["refusal"]
    if q < thresholds["quality"]:
        return "UNKNOWN", "low_quality"
    if match >= thresholds["present"]:
        return "PRESENT", "accepted"
    if match > thresholds["absent"]:
        return "UNKNOWN", "intermediate_match"
    if obstruction > thresholds["obstruction"]:
        return "UNKNOWN", "obstruction"
    return "REMOVED", "accepted"


def read_record(path, expected_binding, job, thresholds):
    record = load(path)
    reported_hash = record.pop("record_payload_sha256")
    serialized = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(hashlib.sha256(serialized).hexdigest() == reported_hash, "Changed record payload")
    require(record["id"] == job["id"] and record["binding_sha256"] == expected_binding, "Record binding mismatch")
    require(all(record[k] == job[k] for k in ("reference_sha256", "query_sha256")), "Record input mismatch")
    answer, cause = decision(record, thresholds)
    require(record["answer"] == answer, "Stored answer differs from its evidence")
    return record, answer, cause


def compare(rows):
    result = Counter()
    for row in rows:
        truth = row["physical_state"]
        prior, candidate = row["baseline"], row["candidate"]
        result["cases"] += 1
        for key, answer in (("baseline", prior), ("candidate", candidate)):
            result[key + "_correct"] += answer == truth
            result[key + "_unknown"] += answer == "UNKNOWN"
            result[key + "_false_present"] += answer == "PRESENT" and truth == "REMOVED"
            result[key + "_false_removed"] += answer == "REMOVED" and truth == "PRESENT"
        result["gained_correct"] += prior != truth and candidate == truth
        result["lost_correct"] += prior == truth and candidate != truth
        prior_wrong = prior not in ("UNKNOWN", truth)
        candidate_wrong = candidate not in ("UNKNOWN", truth)
        result["newly_wrong"] += candidate_wrong and not prior_wrong
        result["prior_errors_avoided"] += prior_wrong and not candidate_wrong
    result["net_correct"] = result["gained_correct"] - result["lost_correct"]
    return dict(result)


def evaluate_criteria(rows, exact_pairs, rule):
    by_id = {r["id"]: r for r in rows}
    total = compare(rows)
    cells = {}
    for engine in ENGINES:
        subset = [r for r in rows if (r["engine"], r["environment"], r["level"], r["state"]) ==
                  (engine, *rule["primary_cell"][1:])]
        cells[engine] = compare(subset)
        require(len(subset) == 25, "Incomplete primary cell")
    require(cells["mujoco"]["baseline_correct"] == rule["primary_correct_strictly_above"] and
            cells["isaac"]["baseline_correct"] == rule["paired_isaac_correct_at_least"], "Primary baseline changed")
    mu_features = [r for r in rows if [r["engine"], r["environment"], r["level"], r["state"]] == rule["primary_cell"]]
    primary_by_base = {b: compare([r for r in mu_features if r["base_id"] == b])
                       for b in sorted({r["base_id"] for r in mu_features})}
    require(len(primary_by_base) == 5 and all(c["cases"] == 5 for c in primary_by_base.values()),
            "Primary arrangement groups changed")
    improved = [b for b, c in primary_by_base.items() if c["net_correct"] > 0]
    pairs = []
    for a_id, b_id in exact_pairs:
        a, b = by_id[a_id], by_id[b_id]
        require(a["pair_id"] == b["pair_id"] and a["physical_state"] != b["physical_state"]
                and a["engine"] == b["engine"] == "mujoco", "Invalid certified pair")
        members = sorted((a, b), key=lambda row: row["state"])
        pairs.append({"pair_id": a["pair_id"], "engine": a["engine"], "base_id": a["base_id"],
                      "level": a["level"], "stop": a["stop"], "ids": [r["id"] for r in members],
                      "states": [r["state"] for r in members], "baseline": [r["baseline"] for r in members],
                      "candidate": [r["candidate"] for r in members]})
    n_pairs = rule["certified_pairs_requiring_two_unknowns"]
    require(len(pairs) == len({p["pair_id"] for p in pairs}) == n_pairs, "Unexpected certified pair set")
    both_unknown = sum(p["candidate"] == ["UNKNOWN", "UNKNOWN"] for p in pairs)
    goals = {
        "more_clear_mujoco_removals_across_arrangements": cells["mujoco"]["candidate_correct"] > rule["primary_correct_strictly_above"]
            and len(improved) >= rule["min_arrangements_with_positive_net_gain"],
        "retain_clear_isaac_removals": cells["isaac"]["candidate_correct"] >= rule["paired_isaac_correct_at_least"],
        "no_new_physical_errors": total["newly_wrong"] <= rule["max_newly_wrong_definite_cases"],
        "all_certified_pairs_both_unknown": both_unknown == n_pairs,
    }
    return {"criteria": goals, "decision": "replicate_on_fresh_arrangements" if all(goals.values()) else "stop_this_revision",
            "clear_tabletop_d6_removed": cells, "mujoco_primary_by_arrangement": primary_by_base,
            "improved_mujoco_arrangements": improved, "candidate_both_unknown": both_unknown, "certified_pairs": pairs}


def read_study(pilot, run_root, release, protocol):
    inputs = protocol["input"]
    prepared = pilot / "prepared"
    analysis = read_bound(pilot / "analysis.json", inputs["source_analysis_sha256"])
    pre = read_bound(prepared / "preparation.json", inputs["preparation_sha256"])
    require(pre["jobs_sha256"]["primary"] == inputs["jobs_sha256"] and pre["ledger_sha256"] == inputs["ledger_sha256"],
            "Earlier preparation binding changed")
    ledger = index_by_id(read_bound(prepared / "ledger.json", inputs["ledger_sha256"]), inputs["cases"])
    jobs = index_by_id(read_bound(prepared / "verifier/primary/jobs.json", inputs["jobs_sha256"]), inputs["cases"])
    prior_rows = index_by_id(analysis["rows"]["relaxed_guard"], inputs["cases"])
    require(jobs.keys() == ledger.keys() == prior_rows.keys(), "Different case IDs")
    require(analysis["control_manifest_sha256"]["relaxed_guard"] == inputs["baseline_manifest_sha256"],
            "Earlier analysis mismatch")
    roots = {"baseline": run_root / "baseline-replay-v1", "candidate": run_root / "candidate-v1"}
    manifests = {}
    expected = {ident + ".json" for ident in jobs}
    for arm, folder in roots.items():
        require((run_root / (arm + "-exit.txt")).read_text().strip() == "0", f"{arm} did not finish successfully")
        manifest_hash = sha(folder / "manifest.json")
        bound = load(folder / "manifest.json")
        require(bound["study_id"] == protocol["study_id"] and bound["jobs_sha256"] == inputs["jobs_sha256"]
                and bound["reference_manifest_sha256"] == inputs["baseline_manifest_sha256"]
                and bound["thresholds"] == protocol["geometry"]["thresholds"], f"Invalid {arm} binding")
        require(bound["source_sha256"] == release["source_sha256"]["development/cross_sim/geometry_run.py"],
                "Runner differs from released version")
        if arm == "baseline":
            require(manifest_hash == release["replay_manifest_sha256"] and bound["config"] == {"method": "baseline_200"},
                    "Unqualified baseline replay")
        else:
            geo = protocol["geometry"]
            provenance = bound["provider"]["provenance"]
            require(bound["config_sha256"] == release["candidate_config_sha256"]
                    and bound["baseline_record_inventory_sha256"] is None
                    and bound["config"]["min_target_valid_fraction"] == geo["min_target_valid_fraction"]
                    and bound["provider"]["source_sha256"] == release["source_sha256"]["development/cross_sim/geoscd_geometry.py"]
                    and provenance["checkpoint_sha256"] == geo["safetensors_sha256"]
                    and provenance["geometry_revision"] == geo["geometry_revision"]
                    and provenance["model_revision"] == geo["model_revision"], "Unreleased geometry candidate")
        require({p.name for p in (folder / "records").iterdir()} == expected, f"Incomplete {arm} record set")
        manifests[arm] = manifest_hash
    rows = []
    record_hashes = {k: hashlib.sha256() for k in roots}
    thresholds = protocol["geometry"]["thresholds"]
    for ident, item in sorted(ledger.items()):
        row = {k: item[k] for k in ("id", "engine", "environment", "base_id", "stop", "level", "state", "pair_id", "physical_state")}
        require(row["physical_state"] == ("PRESENT" if row["state"].endswith("_present") else "REMOVED"),
                "Physical intervention mismatch")
        require(all(prior_rows[ident][k] == v for k, v in row.items()), "Prior study ledger mismatch")
        for arm, folder in roots.items():
            path = folder / "records" / (ident + ".json")
            record, answer, cause = read_record(path, manifests[arm], jobs[ident], thresholds)
            record_hashes[arm].update((ident + " " + sha(path) + "\n").encode())
            row[arm], row[arm + "_gate"] = answer, cause
            if arm == "baseline":
                require(record["baseline_parity"] == "PASS" and answer == prior_rows[ident]["answer"], "Baseline parity lost")
            else:
                reg = record["diagnostics"]["registration"]
                usable = reg["target_valid_fraction"]
                require(0 <= usable <= 1 and usable == reg["box_usable_fraction"] and
                        record["diagnostics"]["align_ok"] == (reg["registration_ok"] and usable >= protocol["geometry"]["min_target_valid_fraction"]),
                        "Invalid geometry gate")
                row["geometry_box_valid_fraction"] = usable
            row[arm + "_seconds"] = record["runtime_seconds"]
        rows.append(row)
    keys = {(r["engine"], r["environment"], r["base_id"], r["level"], r["state"], r["stop"]) for r in rows}
    bases = {e: sorted({r["base_id"] for r in rows if r["environment"] == e}) for e in ENVIRONMENTS}
    require(all(len(v) == 5 for v in bases.values()) and set(bases["shelf"]).isdisjoint(bases["tabletop"]),
            "Invalid arrangements")
    require(keys == {(eng, env, base, level, state, f"P{stop}") for eng in ENGINES for env in ENVIRONMENTS
                     for base in bases[env] for level in LEVELS for state in STATES for stop in range(1, 6)}, "Broken paired design")
    exact = [c["ids"] for c in analysis["certificates"]["relaxed_guard"] if c["equal_all_supplied_inputs"]]
    return rows, bases, exact, manifests, {k: v.hexdigest() for k, v in record_hashes.items()}


def tables(rows, bases, bootstrap):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["engine"], r["environment"], r["level"], r["state"], r["base_id"]].append(r)
    samples = bootstrap_indices(bases, draws=bootstrap["draws"], seed=bootstrap["seed"])
    cells, arrangements = [], []
    for eng in ENGINES:
        for env in ENVIRONMENTS:
            for level in LEVELS:
                for state in STATES:
                    group = {"engine": eng, "environment": env, "level": level, "state": state}
                    case_sets = [grouped[eng, env, level, state, b] for b in bases[env]]
                    base_counts = [compare(part) for part in case_sets]
                    for b, counts in zip(bases[env], base_counts, strict=True):
                        arrangements.append({**group, "base_id": b, **counts})
                    gains = np.array([c["net_correct"] / c["cases"] for c in base_counts])
                    delta = interval(gains, samples[env])
                    cells.append({**group, **compare([r for part in case_sets for r in part]),
                                  "correct_yield_difference": delta["estimate"],
                                  "ci_low": delta["interval_95_percentile"][0], "ci_high": delta["interval_95_percentile"][1]})
    all_level_gains = {}
    for eng in ENGINES:
        point, draws = [], []
        for env in ENVIRONMENTS:
            values = []
            for b in bases[env]:
                part = [r for r in rows if r["engine"] == eng and r["base_id"] == b]
                counts = compare(part)
                values.append(counts["net_correct"] / counts["cases"])
            rates = np.asarray(values)
            point.append(rates.mean())
            draws.append(rates[samples[env]].mean(axis=1))
        all_level_gains[eng] = (np.mean(point), np.mean(draws, axis=0))
    all_level_gains["combined"] = (np.mean([pair[0] for pair in all_level_gains.values()]),
                                   np.mean([pair[1] for pair in all_level_gains.values()], axis=0))
    headline = {eng: {"estimate": float(estimate), "interval_95_percentile": np.percentile(
                draws, [2.5, 97.5], method="linear").tolist()} for eng, (estimate, draws) in all_level_gains.items()}
    return cells, arrangements, headline


def evaluate(pilot, run_root, release_path, output):
    protocol = load(PROTOCOL)
    release = load(release_path)
    require(release["stage"] == "candidate-pilot-released" and release["protocol_sha256"] == sha(PROTOCOL),
            "Release does not match the comparison protocol")
    rows, bases, exact, manifests, inventories = read_study(pilot, run_root, release, protocol)
    criteria = evaluate_criteria(rows, exact, protocol["decision"])
    cells, arrangements, headline = tables(rows, bases, protocol["report"]["bootstrap"])
    recurring_stops = {("CSPT03", "P3"), ("CSPT04", "P5"), ("CSPT05", "P4")}
    prior_refusals = [r for r in rows if r["level"] == "D6" and r["state"] == "clear_removed"
                      and (r["base_id"], r["stop"]) in recurring_stops]
    require(len(prior_refusals) == 6, "Missing matched views for the prior refusals")
    result = {
        "study_id": protocol["study_id"], "scope": protocol["scope"], **criteria, "totals": compare(rows),
        "by_engine": {e: compare([r for r in rows if r["engine"] == e]) for e in ENGINES},
        "prior_recurring_refusals_and_matching_isaac_views": prior_refusals,
        "balanced_change_in_correct_yield": headline,
        "candidate_abstention_gates": dict(Counter(r["candidate_gate"] for r in rows if r["candidate"] == "UNKNOWN")),
        "timing_seconds_per_input": {arm: median(r[arm + "_seconds"] for r in rows) for arm in ("baseline", "candidate")},
        "transition_counts": dict(sorted(Counter(r["baseline"] + "_to_" + r["candidate"] for r in rows).items())),
        "wrong_definite_cases": [r for r in rows if r["baseline"] not in (r["physical_state"], "UNKNOWN")
                                 or r["candidate"] not in (r["physical_state"], "UNKNOWN")],
        "protocol_sha256": sha(PROTOCOL), "release_sha256": sha(release_path), "manifest_sha256": manifests,
        "record_inventory_sha256": inventories, "evaluator_sha256": sha(__file__),
        "bootstrap": protocol["report"]["bootstrap"],
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename, records in (("cells.csv", cells), ("arrangements.csv", arrangements), ("case_decisions.csv", rows)):
        with (output / filename).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    result["output_sha256"] = {p.name: sha(p) for p in sorted(output.glob("*.csv"))}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "run-root", "release", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.pilot, args.run_root, args.release, args.out)
    print(json.dumps({"decision": result["decision"], "criteria": result["criteria"], "totals": result["totals"]}))
