"""Recompute the paper's paired estimates from portable per-query and per-pair evidence."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "evidence"
METHODS = ("homography", "dense", "registered")
ENGINES = ("isaac", "mujoco")
ENVIRONMENTS = ("shelf", "tabletop")
STATES = ("clear_present", "clear_removed", "placed_present", "placed_removed")
LEVELS = ("D0", "D6")


def load_csv(name):
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def flag(value):
    if value not in ("0", "1"):
        raise ValueError(f"Expected binary field; received {value!r}")
    return value == "1"


def interval(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def compare(rows, layout_ids, draws):
    """Count every input once and retain the paired predictions on each layout."""
    by_layout = {name: Counter() for name in layout_ids}
    totals = Counter()
    for r in rows:
        truth = r["truth"]
        if truth not in ("PRESENT", "REMOVED"):
            raise ValueError("Unknown physical state")
        ok = {}
        for m in METHODS:
            ans = r[m]
            if ans not in ("UNKNOWN", "PRESENT", "REMOVED"):
                raise ValueError("Unknown decision")
            ok[m] = ans == truth
            totals[f"{m}_correct"] += ok[m]
            totals[f"{m}_definite"] += ans != "UNKNOWN"
            totals[f"{m}_wrong"] += ans != "UNKNOWN" and ans != truth
            totals[f"{m}_false_removal"] += ans == "REMOVED" and truth == "PRESENT"
            totals[f"{m}_false_presence"] += ans == "PRESENT" and truth == "REMOVED"
        totals["newly_wrong"] += r["dense"] not in ("UNKNOWN", truth) and r[
            "homography"
        ] in ("UNKNOWN", truth)
        totals["avoided_wrong"] += r["homography"] not in ("UNKNOWN", truth) and r[
            "dense"
        ] in ("UNKNOWN", truth)
        totals["gained"] += ok["dense"] and not ok["homography"]
        totals["lost"] += ok["homography"] and not ok["dense"]
        totals["n"] += 1
        by_layout[r["arrangement"]]["net"] += int(ok["dense"]) - int(ok["homography"])
        by_layout[r["arrangement"]]["registered_net"] += int(ok["registered"]) - int(
            ok["homography"]
        )
        by_layout[r["arrangement"]]["n"] += 1
    delta = np.array([by_layout[k]["net"] for k in layout_ids])
    counts = np.array([by_layout[k]["n"] for k in layout_ids])
    numerator = delta[draws].sum(axis=1)
    denominator = counts[draws].sum(axis=1)
    if not (denominator > 0).all():
        raise ValueError("Resampling lost a design stratum")
    reg_net = np.array([by_layout[k]["registered_net"] for k in layout_ids])
    totals = dict(totals)
    totals.update(
        net=totals["gained"] - totals["lost"],
        layouts=int(np.count_nonzero(counts)),
        delta_pp=100 * delta.sum() / counts.sum(),
        delta_interval_pp=interval(100 * numerator / denominator),
        layouts_positive=int((delta > 0).sum()),
        layouts_negative=int((delta < 0).sum()),
        registered_delta_pp=100 * reg_net.sum() / counts.sum(),
        registered_delta_interval_pp=interval(
            100 * reg_net[draws].sum(axis=1) / denominator
        ),
        layout_net={k: by_layout[k]["net"] for k in layout_ids if by_layout[k]["n"]},
    )
    return totals


def check_example():
    from PIL import Image

    source = DATA / "illustration"
    casefile = json.loads((source / "cases.json").read_text())
    counts = {}
    for r in casefile["rows"]:
        for record in r["files"]:
            path = source / record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["file_sha256"]:
                raise ValueError("Illustration source mismatch")
            with Image.open(path) as im:
                if (
                    hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
                    != record["decoded_sha256"]
                ):
                    raise ValueError("Decoded illustration mismatch")
        pred = [v["candidate"] for v in r["cases"]]
        pa, pb = [
            np.asarray(Image.open(source / (c["state"] + ".png")).convert("RGB"))
            for c in r["cases"]
        ]
        changed = int(np.count_nonzero(np.any(pa != pb, axis=2)))
        if changed != r["revisit_pixels_changed_in_full_1200x675_rgb"]:
            raise ValueError("Illustrated case equality mismatch")
        counts[r["setting"]] = {"changed_pixels": changed, "geometry_answers": pred}
    return counts


def check_raw_decisions(by_id):
    export = json.loads((DATA / "export-receipt.json").read_text())
    for method in METHODS:
        folder = DATA / "raw-scores" / method
        raw = (folder / "records.jsonl").read_bytes()
        binding = json.loads((folder / "binding.json").read_text())
        if hashlib.sha256(raw).hexdigest() != export["score_sha256"][method]:
            raise ValueError(f"Raw score checksum changed for {method}")
        if binding["cases_sha256"] != export["packet_inventory_sha256"]:
            raise ValueError(f"Scored packet changed for {method}")
        seen = set()
        for line in raw.splitlines():
            reported = json.loads(line)
            uid = reported["id"]
            row = by_id[uid]
            if (uid in seen or reported["answer"] != row[method]
                    or reported["box_xyxy"] != json.loads(row["box"])
                    or any(reported[r + "_sha256"] != row[r + "_file_sha256"]
                           for r in ("reference", "query"))):
                raise ValueError(f"Raw record differs for {method}, {uid}")
            seen.add(uid)
        if seen != by_id.keys():
            raise ValueError(f"Incomplete raw score inventory for {method}")


def compare_readers(rows):
    admitted = []
    for row in rows:
        dense_passed = row["dense_gate"] == "accepted"
        qwen_passed = row["registered_gate"] == "accepted"
        if dense_passed != qwen_passed:
            raise ValueError(f"Readers used different admission decisions for {row['id']}")
        if dense_passed:
            admitted.append(row)

    result = {}
    for name, cases in (("all_queries", rows), ("shared_registration", admitted)):
        by_engine = {}
        for engine in (*ENGINES, "pooled"):
            group = cases if engine == "pooled" else [r for r in cases if r["engine"] == engine]
            table = Counter((r["truth"], r["dense"], r["registered"]) for r in group)
            counts = {"removed": sum(r["truth"] == "REMOVED" for r in group),
                      "retained": sum(r["truth"] == "PRESENT" for r in group)}
            for method in ("dense", "registered"):
                counts[method] = {
                    "missed_removals": sum(r["truth"] == "REMOVED" and r[method] != "REMOVED" for r in group),
                    "false_deletions": sum(r["truth"] == "PRESENT" and r[method] == "REMOVED" for r in group),
                }
            counts["additional_qwen_false_deletions"] = {
                "dense_answered_present": table["PRESENT", "PRESENT", "REMOVED"],
                "dense_abstained": table["PRESENT", "UNKNOWN", "REMOVED"],
            }
            by_engine[engine] = counts
        result[name] = by_engine
    return result


def run(removals=False):
    rows, pairs = load_csv("queries.csv"), load_csv("pairs.csv")
    identifiers = {r["id"] for r in rows}
    physical_keys = {
        (r["engine"], r["arrangement"], r["stop"], r["level"], r["state"]) for r in rows
    }
    if len(rows) != len(identifiers) or len(rows) != len(physical_keys):
        raise ValueError("Duplicate case")
    by_id = {row["id"]: row for row in rows}
    check_raw_decisions(by_id)
    for r in rows:
        if r["truth"] != ("PRESENT" if r["state"].endswith("_present") else "REMOVED"):
            raise ValueError("Intervention mismatch")
    if removals:
        return compare_readers(rows)
    layout_ids = sorted({r["arrangement"] for r in rows})
    rng = np.random.default_rng(249)
    strata = [
        sorted({r["arrangement"] for r in rows if r["environment"] == e})
        for e in ENVIRONMENTS
    ]
    if (
        len(layout_ids) != 16
        or list(map(len, strata)) != [8, 8]
        or len(set(strata[0]) & set(strata[1]))
    ):
        raise ValueError("Fresh layout panel differs from the complete declared design")
    draws = np.hstack(
        [
            rng.choice(
                [layout_ids.index(k) for k in s], size=(10000, len(s)), replace=True
            )
            for s in strata
        ]
    )
    output = {
        "layouts": len(layout_ids),
        "n": len(rows),
        "engines": {},
        "exploratory_example": check_example(),
    }
    addressed = set()
    for e in ENGINES:
        engine_rows = [r for r in rows if r["engine"] == e]
        if len(engine_rows) != 640:
            raise ValueError(f"Incomplete engine {e}")
        by_env = {}
        for env in ("all", *ENVIRONMENTS):
            current = [
                r for r in engine_rows if env == "all" or r["environment"] == env
            ]
            endpoint = [
                r
                for r in current
                if r["level"] == "D6" and r["state"] == "clear_removed"
            ]
            by_env[env] = {
                "overall": compare(current, layout_ids, draws),
                "primary": compare(endpoint, layout_ids, draws),
            }
        cross = []
        prescore = {
            (r["arrangement"], r["level"], r["stop"], r["placement"]): r
            for r in load_csv(e + "-input-pairs.csv")
        }
        if len(prescore) != 320:
            raise ValueError("Incomplete pre-scoring input audit")
        for p in [p for p in pairs if p["engine"] == e]:
            left, right = by_id[p["present_id"]], by_id[p["removed_id"]]
            if (left["truth"], right["truth"]) != ("PRESENT", "REMOVED"):
                raise ValueError("Unbalanced counterfactual")
            for key in ("engine", "arrangement", "stop", "level", "environment"):
                if left[key] != right[key] or left[key] != p[key]:
                    raise ValueError("Invalid pairing")
            if left["id"] in addressed or right["id"] in addressed:
                raise ValueError("Reused member of a pair")
            addressed.update([left["id"], right["id"]])

            exact = all(
                left[k] == right[k]
                for k in (
                    "reference_pixels_sha256",
                    "query_pixels_sha256",
                    "box",
                    "task",
                )
            )
            if exact != flag(p["exact"]):
                raise ValueError("Published complete-input certificate mismatch")
            independent = prescore[(p["arrangement"], p["level"], p["stop"], p["placement"])]
            if not (
                exact == flag(independent["primary_identical"])
                and flag(p["native_exact"]) == flag(independent["native_identical"])
                and independent["shared_reference_digest"] == left["reference_pixels_sha256"]
                and json.loads(independent["box_json"]) == json.loads(left["box"])
                and independent["target_native_pixels"] == left["visible_pixels"]
            ):
                raise ValueError("Fresh input differs from its pre-scoring audit")
            cross.append((p, left, right, exact))
        cert = [v for v in cross if v[-1]]
        certificates = {
            "pairs": len(cross),
            "exact": len(cert),
            "layouts": len({p["arrangement"] for p, _, _, _ in cert}),
            "native_exact": sum(flag(p["native_exact"]) for p, _, _, _ in cross),
            "fully_hidden_placed": sum(
                p["placement"] == "placed" and int(a["visible_pixels"]) == 0
                for p, a, _, _ in cross
            ),
        }
        for m in METHODS:
            diff = sum(a[m] != b[m] for _, a, b, _ in cert)
            definite = sum(
                a[m] != "UNKNOWN" or b[m] != "UNKNOWN" for _, a, b, _ in cert
            )
            certificates[m] = {
                "commit": definite,
                "abstain": len(cert) - definite,
                "different_answers": diff,
                "unsupported_decisions": sum(
                    int(a[m] != "UNKNOWN") + int(b[m] != "UNKNOWN")
                    for _, a, b, _ in cert
                ),
                "errors": sum(
                    a[m] != a["truth"] and a[m] != "UNKNOWN" for _, a, _, _ in cert
                )
                + sum(b[m] != b["truth"] and b[m] != "UNKNOWN" for _, _, b, _ in cert),
            }
            if diff == 0 and certificates[m]["errors"] != definite:
                raise ValueError("Certified error count violates the paired identity")

        def decisions(items):
            comparable = [
                r
                for r in items
                if r["dense_valid_fraction"] and r["registered_valid_fraction"]
            ]
            return {
                "comparable_maps": len(comparable),
                "execution_failures": {
                    m: sum(bool(r.get(m + "_execution_error")) for r in items)
                    for m in METHODS
                },
                "n": len(items),
                "blocked_before_language": sum(
                    r["registered_gate"] == "registration_gate" for r in items
                ),
                "model_unknown": sum(
                    r["registered_gate"] == "accepted"
                    and r["registered_raw"] == "UNKNOWN"
                    for r in items
                ),
                "unparsed": sum(
                    r["registered_gate"] == "invalid_model_output" for r in items
                ),
                "map_gate_disagreements": sum(
                    (r["dense_gate"] == "accepted")
                    != (r["registered_gate"] != "registration_gate")
                    for r in comparable
                ),
                "max_absolute_map_fraction_difference": max(
                    (
                        abs(
                            float(r["dense_valid_fraction"])
                            - float(r["registered_valid_fraction"])
                        )
                        for r in comparable
                    ),
                    default=None,
                ),
            }

        output["engines"][e] = {
            "environments": by_env,
            "certificates": certificates,
            "retained_target_visibility": {},
            "registered_gate": {
                "all_queries": decisions(engine_rows),
                "certified_queries": decisions(
                    [case for _, a, b, _ in cert for case in (a, b)]
                ),
            },
        }
        for placement in ("clear", "placed"):
            members = [a for p, a, _, _ in cross if p["placement"] == placement]
            fractions = [
                int(a["visible_isolated_overlap_pixels"]) / int(a["isolated_pixels"])
                for a in members
            ]
            output["engines"][e]["retained_target_visibility"][placement] = {
                "pairs": len(members),
                "no_visible_target_pixels": sum(int(a["visible_pixels"]) == 0 for a in members),
                "part_of_isolated_mask_exposed": sum(0 < x < 1 for x in fractions),
                "all_of_isolated_mask_exposed": sum(x == 1 for x in fractions),
                "minimum_exposed_fraction": min(fractions),
                "median_exposed_fraction": float(np.median(fractions)),
            }
    if addressed != identifiers:
        raise ValueError("Unpaired observation")
    hist = load_csv("exploratory-decisions.csv")
    if len(hist) != 1200:
        raise ValueError("Incomplete exploratory data")
    output["exploratory"] = {
        e: {
            m: sum(r[m] == r["physical_state"] for r in hist if r["engine"] == e)
            for m in ("baseline", "candidate")
        }
        for e in ENGINES
    }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--removals", action="store_true", help="Recount missed removals and false deletions on the full and admitted panels.")
    summary = run(removals=parser.parse_args().removals)
    print(json.dumps(summary, indent=2, allow_nan=False))
