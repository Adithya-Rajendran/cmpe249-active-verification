"""Seal the independent reviews, measure cited cues and apply the pilot rules."""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from development.cross_sim.support_labels.prepare import ENGINES, PROTOCOL, load, sha
from development.cross_sim.support_labels.witness import case_evidence


DEFINITE = {"PRESENT", "REMOVED"}
ANSWERS = DEFINITE | {"UNKNOWN", "DISPUTED"}


def store_new(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def seal(run, role):
    prepared, out = run/"prepared", run/"sealed"
    release = load(prepared/"release.json")
    if release["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Qualification protocol changed")
    out.mkdir(exist_ok=True)
    packets = [release["main"]] if role == "main" else release["packets"]
    votes, hashes = {}, []
    for packet in packets:
        folder = Path(packet["path"])
        casefile = folder/"cases.json"
        if sha(casefile) != packet["sha256"] or sha(folder/"Instructions.md") != release["instructions_sha256"]:
            raise ValueError("Presented cases or instructions changed")
        cases = {r["id"]:r for r in load(casefile)}
        answer_file = folder/"votes.json"
        entries = load(answer_file)
        if not isinstance(entries,list) or len(entries) != len(cases) or {r["id"] for r in entries} != set(cases):
            raise ValueError("Incomplete reviewer inventory")
        for r in entries:
            uid = r["id"]
            if uid in votes or r["answer"] not in ANSWERS:
                raise ValueError("Repeated or invalid judgment")
            if any(not isinstance(r.get(k),str) or not r[k].strip() for k in ("target","anchor","cue","uncertainty")):
                raise ValueError("Missing qualitative evidence")
            pt = r["point"]
            if pt is None and r["answer"] in DEFINITE:
                raise ValueError("Missing point on a definite answer")
            if pt is not None and (type(pt) is not list or len(pt) != 2 or any(type(c) is not int for c in pt)
                                   or not (0 <= pt[0] < 1200 and 0 <= pt[1] < 675)):
                raise ValueError("Invalid revisit point")
            for k in ("card", "reference", "query"):
                if sha(folder / (uid+"-"+k+".png")) != cases[uid][k+"_sha256"]:
                    raise ValueError("Presented image changed")
            votes[uid] = r
        hashes.append({"packet_sha256": sha(casefile), "votes_sha256": sha(answer_file)})
    if len(votes) != release["main"]["cases"]:
        raise ValueError("Incomplete review")
    store_new(out/(role+".json"), {"role":role, "utc":datetime.now(timezone.utc).isoformat(), "votes":votes,
                                 "packets":hashes, "release_sha256":sha(prepared/"release.json")})


def measure(run, roots):
    prep, locked = run/"prepared", run/"sealed"
    release = load(prep/"release.json")
    if (release["protocol_sha256"] != sha(PROTOCOL) or release["ledger_sha256"] != sha(prep/"ledger.json")):
        raise ValueError("Release changed")
    readers = {r:load(locked/(r+".json")) for r in ("terra","main")}
    ledger = load(prep/"ledger.json")
    for reader in readers.values():
        if reader["release_sha256"] != sha(prep/"release.json") or set(reader["votes"]) != {r["id"] for r in ledger}:
            raise ValueError("Votes bind different cases")
    rows = []
    for row in ledger:
        votes = {r:reader["votes"][row["id"]] for r,reader in readers.items()}
        evidence = case_evidence(row,votes,roots)
        if evidence["mask_counts_native"]["removed_target"]:
            raise ValueError("Removed object still has target-mask pixels")
        rows.append({"case":row,"votes":votes,"evidence":evidence})
    store_new(run/"measurements.json", {"rows":rows, "reader_sha256":{r:sha(locked/(r+".json")) for r in readers},
              "protocol_sha256":sha(PROTOCOL),"witness_sha256":sha(Path(__file__).with_name("witness.py"))})


def patch_pass(m):
    return all(m[k] for k in ("mask_majority","intervention_change_ge_8","intervention_minus_repeat_ge_4"))


def case_result(entry, review):
    c, v, e = entry["case"], entry["votes"], entry["evidence"]
    truth = "PRESENT" if c["state"].endswith("_present") else "REMOVED"
    result = {k:c[k] for k in ("id","engine","base_id","level","stop","environment","condition")}
    result["physical_state"] = truth
    result["target_fraction"] = c["measurements"]["fraction_visible"] if c["condition"] != "clear" else 1.
    result["exact_ambiguity"] = exact = e["exact_opposite_input_certificate"]
    for name in ("terra","main"):
        result.update({name+"_"+k:v[name][k] for k in ("answer","point","target","anchor","cue","uncertainty")})
        metrics = e["reviewer_points"][name]
        for k in ("mask_pixels","present_removed_mean_abs_channel_delta","clear_repeat_mean_abs_channel_delta"):
            result[name+"_"+k] = metrics[k]
        result[name+"_point_pass"] = patch_pass(metrics)
    terra, main = (v[name]["answer"] for name in ("terra","main"))
    result["same_answer"] = terra == main
    result["terra_wrong_definite"] = terra in DEFINITE and terra != truth
    result["terra_definite_on_exact"] = exact and terra in DEFINITE
    result["qualitative_accept"] = review["accept_visual_cues"]
    result["reviewer_note"] = review["reason"]
    if type(review["accept_visual_cues"]) is not bool or not review["reason"].strip():
        raise ValueError("A per-case evidence review is required")
    result["corroborated_definite"] = (terra == main == truth and not exact and result["terra_point_pass"]
                                      and result["main_point_pass"] and result["qualitative_accept"])
    result["certificate_confirmed_unknown"] = exact and terra == main == "UNKNOWN"
    result["corroborated"] = result["corroborated_definite"] or result["certificate_confirmed_unknown"]
    return result


def counts(rows):
    return {"n":len(rows), "terra":dict(Counter(r["terra_answer"] for r in rows)),
            "physical_states":dict(Counter(r["physical_state"] for r in rows)),
            **{k:sum(r[k] for r in rows) for k in ("same_answer","corroborated_definite","certificate_confirmed_unknown",
              "corroborated","exact_ambiguity","terra_wrong_definite","terra_definite_on_exact")}}


def screen(rows):
    partial = [r for r in rows if r["condition"]=="partial"]
    clear = [r for r in rows if r["condition"]=="clear"]
    partial_answers = Counter(r["terra_answer"] for r in partial if r["corroborated"])
    checks = {"planned_counts": len(partial)==8 and len(clear)==4,
              "minimum_partial":sum(partial_answers.values()) >= 6, "minimum_clear":sum(r["corroborated"] for r in clear)>=3,
              "both_partial_definite_classes": all(partial_answers[k]>0 for k in DEFINITE),
              "no_wrong_definite":not any(r["terra_wrong_definite"] for r in rows),
              "no_definite_on_exact_ambiguity": not any(r["terra_definite_on_exact"] for r in rows)}
    if not checks["no_wrong_definite"] or not checks["no_definite_on_exact_ambiguity"]:
        outcome = "fail"
    elif not checks["planned_counts"]:
        outcome = "inconclusive"
    else:
        outcome = "pass" if all(checks.values()) else "fail"
    return {"outcome":outcome,"checks":checks,"partial_corroborated_by_answer":dict(partial_answers)}


def report(run, dest):
    measured=load(run/"measurements.json")
    reviews=load(run/"adjudication.json")
    if set(reviews) != {r["case"]["id"] for r in measured["rows"]}:
        raise ValueError("Missing or extra per-case reviews")
    rows = sorted((case_result(e, reviews[e["case"]["id"]]) for e in measured["rows"]),
                  key=lambda r: (r["engine"],r["base_id"],r["condition"],r["stop"]))
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/"case-review.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    data = {"overall":counts(rows), "unit":"base arrangement, paired across engines", "arrangements":len({r["base_id"] for r in rows}),
            "engines":{}, "case_review_sha256":sha(dest/"case-review.csv"), "measurements_sha256":sha(run/"measurements.json"),
            "adjudication_sha256":sha(run/"adjudication.json"), "reader_sha256":measured["reader_sha256"],
            "protocol_sha256":measured["protocol_sha256"], "witness_sha256":measured["witness_sha256"],
            "release_sha256":sha(run/"prepared/release.json")}
    for engine in ENGINES:
        subset=[r for r in rows if r["engine"]==engine]
        data["engines"][engine] = {**screen(subset),"overall":counts(subset),
            "conditions":{c:counts([r for r in subset if r["condition"]==c]) for c in ("partial","clear","hidden")}}
    store_new(dest/"results.json",data)
    return data


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("main","terra","measure","report"))
    ap.add_argument("run",type=Path)
    for k in ("isaac","mujoco","dest"):
        ap.add_argument("--"+k,type=Path)
    opt=ap.parse_args()
    if opt.mode in ("main","terra"):
        seal(opt.run,opt.mode)
    elif opt.mode=="measure":
        measure(opt.run,{"isaac":opt.isaac,"mujoco":opt.mujoco})
    else:
        print(json.dumps(report(opt.run,opt.dest),indent=2))
