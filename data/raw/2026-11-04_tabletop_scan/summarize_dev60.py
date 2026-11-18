"""Summary of a generated tabletop dev set (python -m sim.arrange --slots all --design --out DIR): per-class slots, retries,
reserve use, failed tries by reason, targets, M16 analytic patterns, revisit redraw rate, parking refusals incl. the shadow
screen / ray check. Usage: python summarize_dev60.py DIR OUT.json"""
import collections
import json
import sys
from pathlib import Path

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
log = json.loads((D / "index" / "generation_log.json").read_text())
arrs = [json.loads(p.read_text()) for p in sorted(D.glob("*.json"))]
pat = lambda c: "".join("H" if v >= 0.95 else ("C" if v <= 0.10 else "p") for v in c)
out = {"n_arrangements": len(arrs), "n_design": sum(a["design_case"] for a in arrs),
       "counts_by_class": log["counts_by_class"], "reserve_used": log["reserve_used"],
       "n_failed_tries": len(log["retries"]), "n_replacements": len(log["replacements"]),
       "failed_by_class_reason": {}, "targets_by_class": {}, "m16_patterns": {}, "revisit": {}, "parking": {}}
fr = collections.defaultdict(collections.Counter)
for r in log["retries"]:
    fr[r["family"] + ("-" + r["m16_side"] if r["m16_side"] else "")][r["reason"]] += 1
out["failed_by_class_reason"] = {k: dict(v) for k, v in sorted(fr.items())}
tg = collections.defaultdict(collections.Counter)
mp = collections.defaultdict(collections.Counter)
rej_by_stop, stops_redrawn, n_stops = collections.Counter(), 0, 0
park = collections.Counter()
draws = 0
for a in arrs:
    key = a["occluder_family"] + ("-" + a["m16_side"] if a["m16_side"] else "")
    if not a["design_case"]:
        tg[key][a["target_id"]] += 1
        if a["occluder_family"] == "M16":
            mp[key][pat([a["analytic"]["placed_coverage"][s] for s in a["stops"]])] += 1
        pk = a["objects"]["occluder"]["parking"]
        park.update(pk["refused"])
        draws += pk["draws"]
    for s, st in a["revisit"]["stops"].items():
        n_stops += 1
        rej_by_stop[s] += st["n_rejected"]
        stops_redrawn += st["n_rejected"] > 0
out["targets_by_class"] = {k: dict(v) for k, v in sorted(tg.items())}
out["m16_patterns"] = {k: dict(v) for k, v in sorted(mp.items())}
out["revisit"] = {"stops": n_stops, "stops_with_a_redraw": stops_redrawn, "rejected_draws": sum(rej_by_stop.values()),
                  "rejected_by_stop": dict(rej_by_stop), "redraw_rate_per_stop": stops_redrawn / n_stops}
out["parking"] = {"random_arrangements": sum(not a["design_case"] for a in arrs), "draws_total": draws, "refused_by_reason": dict(park),
                  "shadow_rays_rejections_after_passing_the_screen": park["shadow_rays"]}
out["design"] = {a["arrangement_id"]: {"pattern": pat([a["analytic"]["placed_coverage"][s] for s in a["stops"]]),
                                       "margin_px": a["objects"]["occluder"]["design_margin_px"],
                                       "parked": a["objects"]["occluder"]["poses"]["parked"]["base"],
                                       "shadow_check": a["objects"]["occluder"]["parking"]["shadow_check"]}
                 for a in arrs if a["design_case"]}
OUT.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
