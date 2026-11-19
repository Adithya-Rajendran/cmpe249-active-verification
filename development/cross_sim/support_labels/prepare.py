"""Select development views from matched masks and make two kinds of blind packets."""
import argparse
import hashlib
import itertools
import json
import secrets
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image

from development.cross_sim.scoring_prepare import box_from_mask
from development.cross_sim.support_labels.presentation import card
from development.stress.prepare_inputs import clean_image, load, sha


STUDY = "support-label-qualification-v1"
PROTOCOL = Path(__file__).with_name("PROTOCOL.md")
ENGINES = ("isaac", "mujoco")
SIZE = (1200, 675)
ROLES = ("reference", "query", "card")


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def rank(base, group, key):
    return hashlib.sha256("|".join((STUDY, base, group, *key)).encode()).hexdigest()


def masks(folder, stop):
    def binary(name):
        with Image.open(folder / (name + "_" + stop + ".png")) as image:
            if image.size != (2304, 1296):
                raise ValueError("Wrong native image dimensions")
            return np.asarray(image.convert("L")) > 0
    clear = binary("target_clear_present")
    placed = binary("target_placed_present")
    blocker = binary("blocker_placed_present")
    alone = binary("alone")
    if binary("target_placed_removed").any() or binary("target_clear_removed").any():
        raise ValueError("Removed target produced target pixels")
    n = int(clear.sum())
    return {"clear": n, "alone": int(alone.sum()), "placed": int(placed.sum()),
            "fraction_visible": float(np.count_nonzero(clear & placed) / n) if n else 0.,
            "fraction_blocker": float(np.count_nonzero(clear & blocker) / n) if n else 0.}


def select(base, candidates):
    available = {k: v for k, v in candidates.items() if all(
        d["clear"] >= 64 and d["alone"] > 0 and d["clear"]/d["alone"] >= .80 for d in v.values())}
    partial = [k for k, v in available.items() if all(
        .10 <= d["fraction_visible"] <= .90 and d["fraction_blocker"] >= .08 for d in v.values())]
    partial.sort(key=lambda k: rank(base, "partial", k))
    if len(partial) > 1:
        pair = min(itertools.combinations(partial, 2), key=lambda ab: (
            ab[0][1] == ab[1][1], ab[0][0] == ab[1][0],
            rank(base, "partial", ab[0]), rank(base, "partial", ab[1])))
        picked = list(pair)
    else:
        picked = list(partial)
    odd = int(base[-2:]) % 2 == 1
    states = ["placed_present", "placed_removed"] if len(picked) == 2 else ["placed_present" if odd else "placed_removed"]
    choices = [(key, "partial", state) for key, state in zip(picked, states)]
    hidden = [k for k, v in available.items() if k not in picked and all(
        d["placed"] == 0 and d["fraction_blocker"] >= .90 for d in v.values())]
    if hidden:
        key = min(hidden, key=lambda k: rank(base, "hidden", k))
        picked.append(key)
        choices.append((key, "hidden", "placed_removed" if odd else "placed_present"))
    clear = [k for k in available if k not in picked]
    if clear:
        key = min(clear, key=lambda k: rank(base, "clear", k))
        choices.append((key, "clear", "clear_present" if odd else "clear_removed"))
    return choices, {"positions": len(candidates), "clear_eligible": len(available),
                     "partial_eligible": len(partial), "hidden_eligible": len(hidden)}


def completions(roots, cohort, cohort_path):
    markers = {}
    for engine, root in roots.items():
        for base in cohort["arrangements"]:
            for lv, source in base["levels"].items():
                if sha(cohort_path.parent / source["path"]) != source["sha256"]:
                    raise ValueError("Cohort arrangement binding changed")
                directory = root / base["base_id"] / lv
                name = "completion.json" if engine == "isaac" else "complete.json"
                marker = load(directory / name)
                if engine == "mujoco" and marker["identity"]["manifest_sha256"] != sha(cohort_path):
                    raise ValueError("MuJoCo cohort binding changed")
                observed = load(directory / "observed.json")
                if sha(directory / "observed.json") != marker["outputs"]["observed.json"]:
                    raise ValueError("Observed geometry file does not match the capture")
                if engine == "mujoco" and marker["identity"]["arrangement_sha256"] != source["sha256"]:
                    raise ValueError("MuJoCo arrangement binding changed")
                if engine == "isaac" and (observed["arrangement_sha256"] != source["sha256"]
                                           or observed["binding_sha256"] != marker["binding_sha256"]):
                    raise ValueError("Isaac cohort binding changed")
                outputs = marker["outputs"]
                required = ["rgb", "masks"]
                files = {str(f.relative_to(directory)) for sub in required for f in (directory/sub).glob("*.png")}
                if not files or not files <= outputs.keys() or any(sha(directory/f) != outputs[f] for f in files):
                    raise ValueError("Unverified capture files in " + str(directory))
                markers[engine + "/" + base["base_id"] + "/" + lv] = sha(directory / name)
    return markers


def prepare(cohort_path, roots, out, neutral):
    cohort = load(cohort_path)
    if (cohort["study_id"] != STUDY or cohort["split"] != "qualification"
            or len(cohort["arrangements"]) != 4 or cohort["levels"] != ["D0", "D6"]):
        raise ValueError("Expected the fresh qualification cohort")
    if out.exists() or neutral.exists():
        raise FileExistsError("A review output already exists")
    marks = completions(roots, cohort, cohort_path)
    out.mkdir(parents=True)
    private = out / "images"
    main = neutral / "main"
    private.mkdir()
    main.mkdir(parents=True)
    rows, summary = [], {}
    rng = secrets.SystemRandom()
    for base in cohort["arrangements"]:
        bid = base["base_id"]
        pool = {(lv, stop): {e: masks(root/bid/lv/"masks", stop) for e, root in roots.items()}
                for lv in cohort["levels"] for stop in cohort["stops"]}
        choices, counts = select(bid, pool)
        summary[bid] = {**counts, "selected": choices, "measurements": {"/".join(k): v for k,v in pool.items()}}
        for (lv, stop), group, state in choices:
            for engine, root in roots.items():
                ident = secrets.token_hex(6)
                dest = private / ident
                dest.mkdir()
                reference = root / bid / "D0"
                query = root / bid / lv
                ref_file, q_file = dest / "reference.png", dest / "query.png"
                clean_image(reference/"rgb"/("reference_"+stop+".png"), ref_file, SIZE)
                clean_image(query/"rgb"/(state+"_"+stop+".png"), q_file, SIZE)
                box = [round(x*1200/2304) for x in box_from_mask(reference/"masks"/("target_reference_"+stop+".png"))]
                aid = dest / "card.png"
                crop = card(ref_file, q_file, box, aid)
                rows.append({"id": ident, "engine": engine, "base_id": bid, "environment": base["environment"],
                             "level": lv, "stop": stop, "condition": group, "state": state, "box_xyxy": box,
                             "prepared_reference": str(ref_file.resolve()), "prepared_query": str(q_file.resolve()),
                             "reference_sha256": sha(ref_file), "query_sha256": sha(q_file), "card_sha256": sha(aid),
                             "viewing_crop_xyxy": crop, "measurements": pool[lv, stop][engine]})
    if not rows:
        raise ValueError("No cases were eligible")
    packets = [[] for _ in range(max(sum(r["base_id"] == b for r in rows) for b in summary))]
    for base in summary:
        taking = [r for r in rows if r["base_id"] == base]
        rng.shuffle(taking)
        for index, row in zip(rng.sample(range(len(packets)), len(taking)), taking):
            packets[index].append(row)
    def write_packet(folder, group):
        folder.mkdir(exist_ok=True)
        copyfile(PROTOCOL.with_name("REVIEWER.md"), folder / "Instructions.md")
        ordered = rng.sample(group, len(group))
        jobs = []
        for row in ordered:
            uid = row["id"]
            for role in ROLES:
                copyfile(private / uid / (role + ".png"), folder / (uid + "-" + role + ".png"))
            jobs.append({k: row[k] for k in ("id", "box_xyxy", "reference_sha256", "query_sha256", "card_sha256")})
        save(folder / "cases.json", jobs)
        copyfile(folder / "cases.json", out / (folder.name + "-cases.json"))
        return {"path": str(folder.resolve()), "cases": len(jobs), "sha256": sha(folder/"cases.json")}
    packet_records = []
    for group in packets:
        if group:
            if len({r["base_id"] for r in group}) != len(group):
                raise ValueError("A reviewer would see a repeated arrangement")
            packet_records.append(write_packet(neutral / ("p-" + secrets.token_hex(3)), group))
    ma = write_packet(main, rows)
    rng.shuffle(rows)
    save(out/"ledger.json", rows)
    save(out/"selection.json", summary)
    save(out/"release.json", {"study_id": STUDY, "protocol_sha256": sha(PROTOCOL), "cohort_sha256": sha(cohort_path),
         "native_markers": marks, "selection_sha256": sha(out/"selection.json"), "ledger_sha256": sha(out/"ledger.json"),
         "builder_sha256": sha(__file__), "card_builder_sha256": sha(Path(__file__).with_name("presentation.py")),
         "instructions_sha256": sha(PROTOCOL.with_name("REVIEWER.md")),
         "packets": packet_records, "main": ma})
    print(json.dumps({"cases": len(rows), "packets": packet_records, "main": ma}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "isaac", "mujoco", "out", "neutral"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.cohort, {"isaac": args.isaac, "mujoco": args.mujoco}, args.out, args.neutral)
