"""Select camera-balanced cases and issue matched, opaque reader packets."""
import argparse
import hashlib
import itertools
import json
import random
import secrets
from collections import Counter
from pathlib import Path
from shutil import copyfile

from development.cross_sim.scoring_prepare import box_from_mask
from development.cross_sim.support_labels import prepare as first
from development.cross_sim.support_labels.presentation import card


HERE = Path(__file__).resolve().parent
STUDY = "support-localization-comparison-v1"
ORIGINAL = first.PROTOCOL.with_name("REVIEWER.md")
LABELS = ("present", "removed")
BASES = ("LFS01", "LFS02", "LFT01", "LFT02")
LEVELS = ("D0", "D6")
AUDIT_FIELDS = ("id", "box_xyxy", "reference_sha256", "query_sha256", "card_sha256")


def rank(*pieces):
    encoded = json.dumps([STUDY, *pieces], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def eligible(pool):
    clear = [k for k, both in pool.items() if all(
        m["clear"] >= 64 and m["alone"] > 0 and m["clear"] / m["alone"] >= .80
        for m in both.values())]
    partial = [k for k in clear if all(.10 <= m["fraction_visible"] <= .90 and
               m["fraction_blocker"] >= .08 for m in pool[k].values())]
    hidden = [k for k in clear if all(m["placed"] == 0 and m["fraction_blocker"] >= .90
              for m in pool[k].values())]
    return {"clear": clear, "partial": partial, "hidden": hidden}


def alternatives(keys):
    if len(keys) >= 2:
        return list(itertools.permutations(keys, 2))
    if keys:
        return [(keys[0], None), (None, keys[0])]
    return [(None, None)]


def balanced_pairs(environment, bases, candid):
    def score(pairs):
        counts = Counter((key[0], state) for pair in pairs
                         for state, key in zip(LABELS, pair) if key)
        deviation = sum(abs(counts[lv, state]-1) for lv in LEVELS for state in LABELS)
        paired = [pair for pair in pairs if all(pair)]
        levels = sum(p[0][0] != p[1][0] for p in paired)
        stops = sum(p[0][1] != p[1][1] for p in paired)
        return (-sum(counts.values()), deviation, -levels, -stops,
                rank(environment, list(zip(bases,pairs))))

    options = [alternatives(candid[b]["partial"]) for b in bases]
    return min(itertools.product(*options), key=score)


def select(pools):
    eligible_by_base = {b: eligible(pool) for b, pool in pools.items()}
    choices = {b: [] for b in BASES}
    for environment, bases in (("shelf", BASES[:2]), ("tabletop", BASES[2:])):
        pairs = balanced_pairs(environment, bases, eligible_by_base)
        for i, (base, pair) in enumerate(zip(bases, pairs)):
            choices[base] = [(key, "partial", "placed_"+state) for key,state in zip(pair,LABELS) if key]
            occupied = {k for k,_,_ in choices[base]}
            clear_level = "D0" if base in ("LFS01", "LFT02") else "D6"
            for group, state, preferred in (
                    ("hidden","placed_"+LABELS[1-i],"D6" if clear_level=="D0" else "D0"),
                    ("clear","clear_"+LABELS[i],clear_level)):
                remaining = [k for k in eligible_by_base[base][group] if k not in occupied]
                if remaining:
                    key = min(remaining, key=lambda k: (k[0] != preferred, rank(base,group,k)))
                    choices[base].append((key, group, state))
                    occupied.add(key)
    return choices, eligible_by_base


def prepare(run, roots, neutral):
    path = run/"cohort"/"manifest.json"
    cohort = first.load(path)
    if (cohort["study_id"] != STUDY or cohort["split"] != "qualification" or
            [b["base_id"] for b in cohort["arrangements"]] != list(BASES)
            or cohort["levels"] != list(LEVELS)):
        raise ValueError("Unexpected cohort or camera levels")
    declaration = first.load(run/"design-release.json")
    for name, digest in declaration["sha256"].items():
        if first.sha(Path(name)) != digest:
            raise ValueError("Design release changed: " + name)
    if first.load(run/"cohort"/"validation.json")["status"].upper() != "PASS":
        raise ValueError("Cohort validation failed")
    for engine in roots:
        if first.load(run/(engine+"-camera-review.json"))["status"] != "pass":
            raise ValueError("Independent camera checks failed for " + engine)
    if (run/"prepared").exists() or neutral.exists():
        raise FileExistsError("Review destination exists")
    marks = first.completions(roots, cohort, path)
    pools = {base: {(lv, stop): {e: first.masks(r/base/lv/"masks",stop) for e,r in roots.items()}
             for lv in LEVELS for stop in cohort["stops"]} for base in BASES}
    choices, eligibility = select(pools)
    work = run/"prepared"
    private = work/"images"
    private.mkdir(parents=True)
    neutral.mkdir(parents=True)
    rng = random.SystemRandom()
    rows = []
    for spec in cohort["arrangements"]:
        base = spec["base_id"]
        for (lv,stop), condition,state in choices[base]:
            for engine,root in roots.items():
                uid = secrets.token_hex(6)
                dest = private/uid
                dest.mkdir()
                nominal, revisit = root/base/"D0", root/base/lv
                ref, query, aid = (dest/(k+".png") for k in first.ROLES)
                first.clean_image(nominal/"rgb"/("reference_"+stop+".png"),ref,first.SIZE)
                first.clean_image(revisit/"rgb"/(state+"_"+stop+".png"),query,first.SIZE)
                box = [round(x*1200/2304) for x in box_from_mask(nominal/"masks"/("target_reference_"+stop+".png"))]
                crop = card(ref, query, box, aid)
                rows.append({"id":uid,"engine":engine,"base_id":base,"environment":spec["environment"],
                             "level":lv,"stop":stop,"condition":condition,"state":state,"box_xyxy":box,
                             "prepared_reference":str(ref.resolve()),"prepared_query":str(query.resolve()),
                             "reference_sha256":first.sha(ref),"query_sha256":first.sha(query),"card_sha256":first.sha(aid),
                             "viewing_crop_xyxy":crop,"measurements":pools[base][lv,stop][engine]})

    def packet(group, folder, instruction):
        folder.mkdir()
        listings = []
        for row in group:
            uid = row["id"]
            for k in first.ROLES:
                copyfile(private/uid/(k+".png"), folder/(uid+"-"+k+".png"))
            listings.append({k:row[k] for k in AUDIT_FIELDS})
        copyfile(instruction, folder/"Instructions.md")
        first.save(folder/"cases.json",listings)
        copyfile(folder/"cases.json", work/(folder.name+"-cases.json"))
        return {"path":str(folder.resolve()),"n":len(group),"sha256":first.sha(folder/"cases.json"),
                "instructions_sha256":first.sha(folder/"Instructions.md")}

    n_groups = max(sum(r["base_id"]==b for r in rows) for b in BASES)
    groups = [[] for _ in range(n_groups)]
    for base in BASES:
        taking = [r for r in rows if r["base_id"] == base]
        rng.shuffle(taking)
        for idx,row in zip(rng.sample(range(n_groups),len(taking)), taking):
            groups[idx].append(row)
    dispatch = []
    for pair, group in enumerate(groups):
        rng.shuffle(group)
        arms = list(("original","two_step"))
        rng.shuffle(arms)
        for arm in arms:
            instruction = ORIGINAL if arm=="original" else HERE/"LOCATE.md"
            dispatch.append({"pair":pair,"arm":arm,**packet(group,neutral/("p-"+secrets.token_hex(4)),instruction)})
    main = packet(rng.sample(rows,len(rows)),neutral/"main",ORIGINAL)
    rng.shuffle(rows)
    first.save(work/"ledger.json",rows)
    first.save(work/"selection.json",{b:{"eligible":eligibility[b],"selected":choices[b],
               "all_measures":{"/".join(k):v for k,v in pools[b].items()}} for b in BASES})
    release = {"study":STUDY,"cohort_sha256":first.sha(path),"design_sha256":first.sha(run/"design-release.json"),
               "builder_sha256":first.sha(__file__),"camera_markers":marks,"ledger_sha256":first.sha(work/"ledger.json"),
               "selection_sha256":first.sha(work/"selection.json"),"packets":dispatch,"main":main}
    first.save(work/"release.json",release)
    inventory = {b: Counter() for b in (s["environment"] for s in cohort["arrangements"])}
    for r in rows:
        if r["engine"] == "isaac" and r["condition"]=="partial":
            inventory[r["environment"]][r["level"]+"/"+r["state"]]+=1
    print(json.dumps({"n":len(rows),"packets":len(dispatch),"partial_inventory":inventory}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run","isaac","mujoco","neutral"):
        parser.add_argument("--"+key,required=True,type=Path)
    opt = parser.parse_args()
    prepare(opt.run,{"isaac":opt.isaac,"mujoco":opt.mujoco},opt.neutral)
