"""Post-hoc headroom check for a one-extra-view study, on the public paper's main panel.

Each layout has five camera stops (P1..P5), each with its own reference. An episode starts at one stop.
If the verifier answers UNKNOWN there, a policy picks ONE other stop (same layout, engine, camera level,
blocker condition and physical state) and the answer at that stop is final. Travel cost is ignored.
Policies: none; next stop (P1->P2 ... P5->P4); random other stop (expected counts); visibility (the stop
where the retained twin shows the largest share of the target, ties to the nearer stop; this uses the
evaluator's masks); oracle (best outcome that
any single other stop can give).
Run from the repository root: python3 paper/archived_stop_replay.py
Inspected development data. Not a study result.
"""
import collections
import csv
from pathlib import Path

with (Path(__file__).resolve().parent / "evidence" / "queries.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
STOPS = ["P1", "P2", "P3", "P4", "P5"]


def frac(r):
    iso = int(r["isolated_pixels"])
    return int(r["visible_isolated_overlap_pixels"]) / iso if iso else 0.0


cell = collections.defaultdict(dict)      # (engine, layout, level, blocker, truth) -> stop -> row
vis = {}                                  # (engine, layout, level, blocker, stop) -> visible share of retained twin
for r in rows:
    blocker = r["state"].split("_")[0]
    cell[(r["engine"], r["arrangement"], r["level"], blocker, r["truth"])][r["stop"]] = r
    if r["truth"] == "PRESENT":
        vis[(r["engine"], r["arrangement"], r["level"], blocker, r["stop"])] = frac(r)


def outcome(r, m):
    return "unknown" if r[m] == "UNKNOWN" else ("correct" if r[m] == r["truth"] else "wrong")


RANK = {"correct": 2, "unknown": 1, "wrong": 0}


def run(m, select):
    tot = {p: collections.Counter() for p in ["none", "next", "random", "visibility", "oracle"]}
    n = 0
    for (eng, lay, lvl, blk, truth), by_stop in cell.items():
        if not select(eng, lay, lvl, blk, truth):
            continue
        for s in STOPS:
            n += 1
            first = outcome(by_stop[s], m)
            tot["none"][first] += 1
            if first != "unknown":
                for p in ["next", "random", "visibility", "oracle"]:
                    tot[p][first] += 1
                continue
            others = [t for t in STOPS if t != s]
            i = STOPS.index(s)
            nxt = STOPS[i + 1] if i < 4 else STOPS[3]
            tot["next"][outcome(by_stop[nxt], m)] += 1
            for t in others:
                tot["random"][outcome(by_stop[t], m)] += 1 / len(others)
            best_vis = max(others, key=lambda t: (vis[(eng, lay, lvl, blk, t)], -abs(STOPS.index(t) - i)))
            tot["visibility"][outcome(by_stop[best_vis], m)] += 1
            tot["oracle"][max((outcome(by_stop[t], m) for t in others), key=RANK.get)] += 1
    return n, tot


def show(title, select):
    print(f"\n{title}")
    for m in ["homography", "dense"]:
        n, tot = run(m, select)
        print(f"  {m}: {n} starting queries")
        for p, c in tot.items():
            print(f"    {p:10s} correct {c['correct']:6.1f} ({100 * c['correct'] / n:5.1f}%)  wrong {c['wrong']:5.1f}"
                  f"  unknown {c['unknown']:6.1f}")


show("A. Blocker placed (the occlusion cases), both states, both engines, both camera levels",
     lambda e, l, v, b, t: b == "placed")
show("B. Blocker placed, by engine: Isaac", lambda e, l, v, b, t: b == "placed" and e == "isaac")
show("C. Blocker placed, by engine: MuJoCo", lambda e, l, v, b, t: b == "placed" and e == "mujoco")
show("D. All queries", lambda e, l, v, b, t: True)

print("\nE. Layouts where no stop shows more than half of the target with the blocker placed (per engine, level):")
c = collections.Counter()
for (eng, lay, lvl, blk, stop), f in vis.items():
    if blk == "placed":
        c[(eng, lay, lvl)] = max(c[(eng, lay, lvl)], f)
for eng in ["isaac", "mujoco"]:
    for lvl in ["D0", "D6"]:
        vals = [v for (e, l, v_), v in c.items() if e == eng and v_ == lvl]
        print(f"  {eng} {lvl}: {sum(v <= 0.5 for v in vals)} of {len(vals)} layouts; fully clear from some stop: {sum(v >= 0.9995 for v in vals)}")
