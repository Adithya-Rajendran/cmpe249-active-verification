"""POST HOC diagnostic (after the scan; decides nothing): why parking fails for tall / wide boards. Replays try_place's stages
for the first 20 probe seeds of a few scan cells and sums the parked draws refused by reason (sim/arrange.py parked_reason)."""
import collections
import json
import sys
sys.path.insert(0, ".")
from sim import arrange as A
from sim import feasibility as F

base = json.load(open("data/raw/2026-11-04_tabletop_scan/base_config.json"))
for fam, side, h, w, tid in [("IND28", None, 0.22, 0.36, 4), ("IND28", None, 0.25, 0.28, 4), ("M16", "L", 0.25, 0.28, 4), ("IND28", None, 0.18, 0.28, 4)]:
    cfg = F.apply_overrides(base, {**{f"occluder.families.{f}.height_m": h for f in ("S8", "M16", "IND28")}, "occluder.families.IND28.width_m": w})
    ctx = A.load_context(cfg)
    tot, nfail = collections.Counter(), 0
    for seed in range(7000, 7020):
        d = A.Draw(A.try_entropy(seed, 0, "tabletop"))
        t = A.place_target(ctx, d, tid)
        occ = A.place_occluder(ctx, d, t, fam, side) if t else None
        dis = A.place_distractors(ctx, d, t, occ) if occ else None
        if dis is None:
            continue
        if A.park_occluder(ctx, d, occ, t["body"], [x["body"] for x in dis]) is None:
            nfail += 1
            tot.update(occ["parking"]["refused"])
    print(f"{fam}{'-' + side if side else ''} h={h} w={w} target {tid}: {nfail} failed parks of 20 seeds; refused draws by reason {dict(tot)}")
