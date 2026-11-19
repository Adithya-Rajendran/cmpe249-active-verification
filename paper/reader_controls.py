"""Reproduce the post hoc checks on the diagnostic language-model reader."""

import json
from collections import Counter
from statistics import median

from recount import DATA, METHODS, check_raw_decisions, load_csv


def records(path):
    with (DATA / path).open() as src:
        return [json.loads(line) for line in src]


def box_area(box):
    left, top, right, bottom = box
    return (right - left) * (bottom - top)


def main():
    cases = load_csv("queries.csv")
    check_raw_decisions({r["id"]: r for r in cases})
    same = [r for r in cases if r["reference_pixels_sha256"] == r["query_pixels_sha256"]]
    retained = [r for r in cases if r["truth"] == "PRESENT"]
    errors = [r for r in retained if r["registered"] == "REMOVED"]
    exposed = [r for r in retained if int(r["visible_isolated_overlap_pixels"]) * 10000
               >= int(r["isolated_pixels"]) * 9995]
    # Keep source-table order for tied areas. The upper boundary splits a tie.
    exposed.sort(key=lambda r: int(r["isolated_pixels"]))
    thirds = []
    for i in range(3):
        group = exposed[i * len(exposed) // 3 : (i + 1) * len(exposed) // 3]
        thirds.append({"retained": len(group),
                       "false_deletions": sum(r["registered"] == "REMOVED" for r in group)})
    admitted = [r for r in cases if r["registered_gate"] == "accepted"]
    if any((r["dense_gate"] == "accepted") != (r["registered_gate"] == "accepted") for r in cases):
        raise ValueError("Geometric and language readers have different admission rules")
    dense = {r["id"]: r for r in records("raw-scores/dense/records.jsonl")}
    threshold_calls = Counter(r["truth"] for r in admitted if dense[r["id"]]["evidence"]["match_score"] < .60)
    return {
        "false_deletions": len(errors),
        "fully_exposed_errors": sum(int(r["visible_isolated_overlap_pixels"]) == int(r["isolated_pixels"]) for r in errors),
        "fully_hidden_errors": sum(int(r["visible_pixels"]) == 0 for r in errors),
        "nominal_camera_errors": sum(r["level"] == "D0" for r in errors),
        "identical_reference_and_revisit": {m: dict(Counter(r[m] for r in same)) for m in METHODS},
        "nearly_or_fully_exposed_sorted_by_silhouette_area": thirds,
        "median_reference_box_area": {
            "archived_controls": median(box_area(r["box_xyxy"]) for r in records("qualification/records.jsonl")),
            "main_admitted": median(box_area(json.loads(r["box"])) for r in admitted),
        },
        "post_hoc_match_below_0.60": {
            "admitted_queries": len(admitted),
            "actual_removals": sum(r["truth"] == "REMOVED" for r in admitted),
            "removals_found": threshold_calls["REMOVED"],
            "false_deletions": threshold_calls["PRESENT"],
        },
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
