"""Text table of results.json (successes of 100 tries per cell). Usage: python summarize_scan.py results.json > scan_table.txt"""
import json
import sys

r = json.load(open(sys.argv[1]))
T = [4, 9, 14, 21, 6]
N = {4: "soup", 9: "meat", 14: "mug", 21: "brick", 6: "tuna"}
rows = r["rows"]
print(f"successes / {rows[0]['tries']} tries per cell (targets: " + ", ".join(f"{t}={N[t]}" for t in T) + ")")
print(f"{'cell':22s}" + "".join(f"{N[t]:>7s}" for t in T))
key = lambda x: (x["family"], x["side"] or "", x["width_m"] or 0, str(x["choice"]))
groups = {}
for x in rows:
    groups.setdefault((x["height_m"], key(x)), {})[x["target"]] = x["successes"]
for (h, k), v in sorted(groups.items(), key=lambda e: (e[0][0] is not None, e[0][0] or 0, e[0][1])):
    fam, side, w, ch = k
    lab = (f"h={h:g} " if h is not None else "") + fam + (f"-{side}" if side else "") + (f" w={w:g}" if w else "") + (f" {ch}" if ch != "None" else "")
    print(f"{lab:22s}" + "".join(f"{v.get(t, '-'):>7}" for t in T))
print(f"seconds {r['seconds']} workers {r['workers']}")
