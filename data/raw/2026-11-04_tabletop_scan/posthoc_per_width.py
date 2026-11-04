"""POST HOC sensitivity (written AFTER the scan; not part of the plan, decides nothing): the plan's rule (b) for IND28 with
the best support taken per width (each width's own 5-height grid) instead of over the whole height x width grid.
Usage: python posthoc_per_width.py results.json"""
import json
import sys
sys.path.insert(0, ".")
from sim.feasibility import fisher_one_sided_lower as fisher

r = json.load(open(sys.argv[1]))
rows, N = r["rows"], r["rows"][0]["tries"]
ind = [x for x in rows if x["family"] == "IND28"]
for w in sorted({x["width_m"] for x in ind}):
    cw = [x for x in ind if x["width_m"] == w]
    best = {t: max(x["successes"] for x in cw if x["target"] == t) for t in {x["target"] for x in cw}}
    B = sorted(t for t, k in best.items() if k > 0)
    for h in sorted({x["height_m"] for x in cw}):
        sup = {x["target"]: x["successes"] for x in cw if x["height_m"] == h}
        fails = [t for t in B if not (sup[t] >= 1 and fisher(sup[t], N, best[t], N) >= 0.05)]
        print(f"IND28 w={w:g} h={h:g}: retains its own-width best support: {not fails}" + (f" (fails {fails})" if fails else ""))
