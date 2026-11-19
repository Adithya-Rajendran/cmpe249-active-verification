"""Build a portable input archive or recount it using Python, Pillow and NumPy."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
from PIL import Image

METHODS = {"baseline_200": "homography", "geometry": "dense", "vggt_qwen": "registered"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(flag, message):
    if not flag:
        raise ValueError(message)


def native_sweep(observed):
    objects = observed["objects"]
    parked = np.asarray(objects["blocker_parked"]["bounds"])
    closed = np.asarray(objects["blocker_placed"]["bounds"])
    swept_min = np.minimum(parked[0], closed[0])
    swept_max = np.maximum(parked[1], closed[1])
    groups = {
        "architecture": observed["static_bounds"],
        "objects": {
            k: v["bounds"] for k, v in objects.items() if not k.startswith("blocker_")
        },
    }
    results = {}
    for kind, shapes in groups.items():
        separations = {}
        for name, (low, high) in shapes.items():
            distance = np.maximum(
                np.asarray(low) - swept_max, swept_min - np.asarray(high)
            )
            separations[name] = float(np.linalg.norm(np.maximum(distance, 0.0)))
        nearest = min(separations, key=separations.get)
        check(separations[nearest] > 0.0, "Physical door sweep touches " + nearest)
        results[kind] = {
            "count": len(shapes),
            "nearest": nearest,
            "gap_m": separations[nearest],
        }
    return results


def call_from_record(record):
    error = record.get("failure")
    if error == "registration_gate":
        return "registration_gate_unknown"
    if error:
        return (
            "invalid_model_output"
            if error == "invalid_model_output"
            else "execution_failure"
        )
    if record["answer"] == "UNKNOWN":
        return "qwen_unknown" if record.get("raw") == "UNKNOWN" else "rule_unknown"
    return record["answer"].lower()


def check_changes(prior_pairs, prior_raw, changes, pairs, actual):
    old = {
        (r["engine"], r["arrangement"], r["stop"], r["placement"], r["method"]): r
        for r in prior_pairs
    }
    seen = set()
    for row in changes:
        key = tuple(row[k] for k in ("engine", "layout", "view", "door", "method"))
        check(key in old and key not in seen, "Unknown or duplicate matched pair")
        seen.add(key)
        previous = old[key]
        current = pairs[key]
        check(
            row["old_exact"] == previous["complete_input_equal"]
            and row["new_exact"] == current["exact"],
            "Matched comparison uses a different equality check",
        )
        for member in ("present", "removed"):
            output = actual[(key[-1], current[member + "_id"])]
            check(
                row["old_" + member] == previous[member + "_answer"]
                and row["new_" + member] == output["answer"]
                and row["old_" + member + "_call"] == previous[member + "_call_type"]
                and row["new_" + member + "_call"] == call_from_record(output),
                "Matched comparison uses different answers or stopping reasons",
            )
            if key[-1] == "vggt_qwen":
                earlier = prior_raw[previous[member + "_id"]]
                check(
                    earlier["answer"] == previous[member + "_answer"]
                    and call_from_record(earlier) == previous[member + "_call_type"]
                    and float(row["old_" + member + "_usable_fraction"])
                    == earlier["valid_fraction"]
                    and float(row["new_" + member + "_usable_fraction"])
                    == output["valid_fraction"],
                    "The geometry fractions or language-model records do not match",
                )
        before = (
            row["old_exact"] == "True"
            and row["old_present"] == row["old_removed"] == "REMOVED"
        )
        after = (
            row["new_exact"] == "True"
            and row["new_present"] == row["new_removed"] == "REMOVED"
        )
        label = "new" if after else "neither"
        if before:
            label = (
                "persistent"
                if after
                else "resolved_on_exact_input"
                if row["new_exact"] == "True"
                else "lost_exact_opportunity"
            )
        check(row["double_removed_change"] == label, "Wrong paired transition")
    check(len(changes) == len(old) == len(seen) == 288, "Incomplete matched comparison")
    return len(seen)


def build(run, tables, bundle):
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).resolve().parent
    files = {}

    def add(name, path):
        if name in files:
            return
        check(path.is_file(), "Missing " + str(path))
        files[name] = path

    for name in json.loads((run / "design-release.json").read_bytes())["source_files"]:
        add("design/" + name, source / name)
    for p in (run / "cohort").iterdir():
        add("cohort/" + p.name, p)
    for p in (source / "materials").iterdir():
        add("design/materials/" + p.name, p)
    for name in [
        "PROTOCOL.md",
        "declaration.json",
        "scene.py",
        "capture.py",
        "materials.py",
        "ASSETS.md",
        "REPRODUCE.md",
        "release.py",
        "prepare.py",
        "run_scores.py",
        "evaluate.py",
        "supplement.py",
    ]:
        add("design/" + name, source / name)
    for n in [
        "endpoint-lock.json",
        "design-release.json",
        "fixture-review.json",
        "fixture-exposure-criteria.json",
    ]:
        add(n, run / n)
    add("paper/specification.json", root / "paper/evidence/design/specification.json")
    add(
        "paper/previous-pairs.csv",
        root / "notes/figures/2026-11-18-corridor-transfer/pairs.csv",
    )
    add(
        "paper/registered-prior-outputs.jsonl",
        root / "runs/2026-11-18-corridor-transfer-v1/scoring/vggt_qwen/records.jsonl",
    )
    add("paper/ASSET-CREDITS.md", root / "paper/evidence/ASSET-CREDITS.md")
    for p in (run / "qualification").iterdir():
        add("qualification/" + p.name, p)
    for n in [
        "summary.json",
        "decisions.csv",
        "pairs.csv",
        "paired-changes.csv",
        "method-by-view.csv",
    ]:
        add("results/" + n, tables / n)
    add("inputs/cases.json", run / "scoring/fresh-packet/cases.json")
    for row in json.loads(files["inputs/cases.json"].read_bytes()):
        for role in ["reference", "query"]:
            add(
                "images/" + row[role + "_sha256"] + ".png",
                run / "scoring/fresh-packet" / row[role],
            )
    for row in json.loads(files["qualification/private-ledger.json"].read_bytes()):
        for item in row["sources"].values():
            add("images/" + item["sha256"] + ".png", run / item["path"])
    for method, slug in METHODS.items():
        for suffix in ["-guard.json", "-verified.json"]:
            add("scoring/" + method + suffix, run / "scoring" / (method + suffix))
        for n in ["records.jsonl", "binding.json"]:
            add("scoring/" + method + "/" + n, run / "scoring" / method / n)
        add(
            "paper/" + method + "-binding.json",
            root / "paper/evidence/raw-scores" / slug / "binding.json",
        )
    for p in (run / "operator").glob("*/COR*/D*/*.json"):
        if p.parts[-4] in ("mujoco", "isaac"):
            add("capture/" + "/".join(p.parts[-4:]), p)
    add("capture/capture-receipt.json", run / "operator/capture-receipt.json")
    add("capture/isaac-binding.json", run / "operator/isaac-binding.json")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_suffix(".partial")
    check(not bundle.exists(), "Archive already exists")
    with ZipFile(temporary, "x", allowZip64=True) as z:
        for name, path in sorted(files.items()):
            z.write(
                path,
                name,
                compress_type=ZIP_STORED if name.endswith(".png") else ZIP_DEFLATED,
            )
    temporary.rename(bundle)


def recount(bundle):
    with ZipFile(bundle) as z:

        def read(name):
            return json.loads(z.read(name))

        def rows(name):
            return list(csv.DictReader(io.StringIO(z.read(name).decode())))

        @lru_cache(maxsize=12)
        def image(key, mode):
            data = z.read("images/" + key + ".png")
            check(digest(data) == key, "Image checksum mismatch")
            with Image.open(io.BytesIO(data)) as im:
                check(im.mode == mode, "Unexpected image mode")
                return np.asarray(im).copy()

        @lru_cache(maxsize=2000)
        def raster_digest(key):
            im = image(key, "RGB")
            return digest(im.tobytes())

        @lru_cache(maxsize=800)
        def verify_delivery(native_key, delivered_key):
            source = image(native_key, "RGB")
            check(source.shape == (1296, 2304, 3), "Wrong native RGB dimensions")
            rebuilt = Image.fromarray(source).resize(
                (1200, 675), getattr(Image, "Resampling", Image).LANCZOS
            )
            check(
                digest(np.asarray(rebuilt).tobytes()) == raster_digest(delivered_key),
                "Delivered pixels differ from original native render",
            )

        @lru_cache(maxsize=140)
        def bbox(key):
            im = image(key, "L")
            check(im.shape == (1296, 2304), "Native reference size")
            y, x = np.nonzero(im)
            return [
                (int(x.min()) * 1200) // 2304,
                (int(y.min()) * 675) // 1296,
                ((int(x.max()) + 1) * 1200 + 2303) // 2304,
                ((int(y.max()) + 1) * 675 + 1295) // 1296,
            ]

        lock = read("endpoint-lock.json")
        design = read("design-release.json")
        ready = read("qualification/scorer-ready.json")
        declaration = lock.get("declaration_file", "declaration.json")
        check(
            design["endpoint_lock_sha256"] == digest(z.read("endpoint-lock.json"))
            and ready["design_release_sha256"] == digest(z.read("design-release.json"))
            and lock["declaration_sha256"] == digest(z.read("design/" + declaration))
            and read("fixture-review.json")["criteria_sha256"]
            == digest(z.read("fixture-exposure-criteria.json"))
            and lock["prior_pairs_sha256"]
            == digest(z.read("paper/previous-pairs.csv")),
            "Different design, fixture acceptance criteria or original corridor comparison",
        )
        if "door_amendment_sha256" in lock:
            check(
                lock["door_amendment_sha256"]
                == digest(z.read("design/DOOR_AMENDMENT.md"))
                and lock["original_declaration_sha256"]
                == digest(z.read("design/declaration.json")),
                "Changed door amendment",
            )
        check(
            lock["protocol_sha256"] == digest(z.read("design/PROTOCOL.md"))
            and lock["original_task_sha256"]
            == digest(z.read("paper/specification.json")),
            "Protocol drift",
        )
        for name, h in design["source_files"].items():
            check(h == digest(z.read("design/" + name)), "Changed scene source")
        check(
            design["cohort_manifest_sha256"] == digest(z.read("cohort/manifest.json"))
            and design["fixture_review_sha256"]
            == digest(z.read("fixture-review.json")),
            "Changed design",
        )
        native_files = {}
        native_sweeps = []
        for layout in read("cohort/manifest.json")["arrangements"]:
            for level, item in layout["levels"].items():
                recipe = read("cohort/" + item["path"])
                check(
                    digest(z.read("cohort/" + item["path"])) == item["sha256"],
                    "Changed scene recipe",
                )
                for engine, marker in [
                    ("isaac", "completion.json"),
                    ("mujoco", "complete.json"),
                ]:
                    base = f"capture/{engine}/{layout['base_id']}/{level}/"
                    proof = read(base + marker)
                    observed = read(base + "observed.json")
                    if "door_amendment_sha256" in lock:
                        native_sweeps.append(native_sweep(observed))
                    check(
                        ready["completion_hashes"][
                            "operator/" + base[len("capture/") : -1]
                        ]
                        == digest(z.read(base + marker)),
                        "Changed completion record",
                    )
                    check(
                        proof["outputs"]["observed.json"]
                        == digest(z.read(base + "observed.json")),
                        "Changed native transforms",
                    )
                    for state in (
                        "clear_present",
                        "clear_removed",
                        "placed_present",
                        "placed_removed",
                    ):
                        door, truth = state.split("_")
                        if engine == "mujoco":
                            state_data = observed["states"][state]
                            enabled = state_data["target_enabled"]
                            position = state_data["blocker"]["position"]
                        else:
                            state_data = observed["usd_per_state"][state]
                            enabled = (
                                state_data["/World/Objects/target"]["visibility"]
                                != "invisible"
                            )
                            position = state_data["/World/Objects/occluder"][
                                "world_from_local_row"
                            ][3][:3]
                        poses = recipe["objects"]["occluder"]["poses"]
                        goal = poses["parked" if door == "clear" else "placed"]["base"]
                        check(
                            enabled == (truth == "present")
                            and np.allclose(position, goal, atol=1e-5, rtol=0),
                            "The stored simulation state disagrees with the intervention",
                        )
                    for name, image_hash in proof["outputs"].items():
                        native_files["operator/" + base[len("capture/") :] + name] = (
                            image_hash
                        )
        by_id = {r["id"]: r for r in read("inputs/cases.json")}
        private = {r["id"]: r for r in read("qualification/private-ledger.json")}
        check(
            len(private) == len(by_id) == 320 and private.keys() == by_id.keys(),
            "Missing inputs",
        )
        check(
            ready["cases_sha256"] == digest(z.read("inputs/cases.json"))
            and ready["private_ledger_sha256"]
            == digest(z.read("qualification/private-ledger.json")),
            "Packet binding",
        )
        actual = {}
        for method, slug in METHODS.items():
            receipt = read("scoring/" + method + "-verified.json")
            check(
                receipt["cases_sha256"] == ready["cases_sha256"]
                and receipt["records_sha256"]
                == digest(z.read("scoring/" + method + "/records.jsonl"))
                and receipt["binding_sha256"]
                == digest(z.read("scoring/" + method + "/binding.json")),
                "Score files differ from the verifier's completion receipt",
            )
            records = [
                json.loads(l)
                for l in z.read("scoring/" + method + "/records.jsonl").splitlines()
            ]
            observed = {r["id"]: r for r in records}
            check(
                len(records) == len(observed) == 320
                and observed.keys() == by_id.keys(),
                "Missing method outputs",
            )
            b = read("scoring/" + method + "/binding.json")
            orig = read("paper/" + method + "-binding.json")
            check(
                digest(z.read("paper/" + method + "-binding.json"))
                == lock["original_bindings"][slug],
                "Paper binding drift",
            )
            check(
                b.pop("cases_sha256") == ready["cases_sha256"],
                "Method saw different cases",
            )
            orig.pop("cases_sha256")
            check(b == orig, "Changed model or settings")
            for uid, r in observed.items():
                check(
                    all(
                        by_id[uid][k] == r[k]
                        for k in ["box_xyxy", "reference_sha256", "query_sha256"]
                    ),
                    "Different scored input",
                )
                actual[(method, uid)] = r
        # Read intervention files directly. No pair table or decision table provides an answer here.
        joined = defaultdict(dict)
        counts = Counter()
        ids = set()
        gates = Counter()
        for uid, meta in private.items():
            c = by_id[uid]
            src = meta["sources"]
            base = f"operator/{meta['engine']}/{meta['layout']}/"
            view, state = meta["view"], meta["state"]
            names = {
                "reference": f"D0/rgb/reference_{view}.png",
                "reference_mask": f"D0/masks/target_reference_{view}.png",
                "query": f"D6/rgb/{state}_{view}.png",
                "query_mask": f"D6/masks/target_{state}_{view}.png",
                "blocker_mask": f"D6/masks/blocker_{state}_{view}.png",
                "alone_mask": f"D6/masks/alone_{view}.png",
            }
            check(
                all(
                    src[k]["path"] == base + v
                    and src[k]["sha256"] == native_files[base + v]
                    for k, v in names.items()
                ),
                "A scored source points to a different physical intervention",
            )
            box = bbox(src["reference_mask"]["sha256"])
            check(box == c["box_xyxy"], "Wrong reference box")
            natives = []
            delivered = []
            for role in ["reference", "query"]:
                target = c[role + "_sha256"]
                original = src[role]["sha256"]
                pix = image(target, "RGB")
                check(pix.shape == (675, 1200, 3), "Wrong delivered image size")
                verify_delivery(original, target)
                natives.append(raster_digest(original))
                delivered.append(raster_digest(target))
            visible = image(src["query_mask"]["sha256"], "L") > 0
            alone = image(src["alone_mask"]["sha256"], "L") > 0
            image(src["blocker_mask"]["sha256"], "L")
            door, truth = meta["state"].split("_")
            n = int(visible.sum())
            a = int(alone.sum())
            overlap = int(np.count_nonzero(visible & alone))
            check(
                (n, a, overlap) == (meta["visible"], meta["isolated"], meta["overlap"]),
                "Native exposure differs",
            )
            key = (meta["engine"], meta["layout"], meta["view"], door)
            check(truth not in joined[key], "Duplicate intervention")
            joined[key][truth] = (uid, tuple(natives + delivered + box), n, overlap / a)
            for method in METHODS:
                u = actual[(method, uid)]
                ans = u["answer"]
                counts[(meta["engine"], method, "false_removals")] += (
                    truth == "present" and ans == "REMOVED"
                )
                counts[(meta["engine"], method, "definite")] += ans != "UNKNOWN"
                counts[(meta["engine"], method, "correct")] += ans == truth.upper()
                if method == "vggt_qwen":
                    gates[(meta["engine"], u.get("failure") or "passed")] += 1
            ids.add(uid)
        expected = {
            (r["engine"], r["layout"], r["view"], r["door"], r["method"]): r
            for r in rows("results/pairs.csv")
        }
        check(len(joined) == 160 and len(expected) == 480, "Missing pairs")
        prim = []
        primary_exact = []
        mask_counts = Counter()
        for key, pair in joined.items():
            engine, layout, view, door = key
            held, removed = pair["present"], pair["removed"]
            same = held[1] == removed[1]
            check(removed[2] == 0, "Removed mask nonzero")
            mask_counts[(engine, door, "qualified")] += int(
                held[3] >= 0.95 if door == "clear" else held[2] == 0
            )
            if engine == "mujoco" and view == "P3" and door == "placed" and same:
                primary_exact.append(layout)
            for method in METHODS:
                r = expected[(*key, method)]
                pa = actual[(method, held[0])]["answer"]
                ga = actual[(method, removed[0])]["answer"]
                check(
                    (
                        r["present_id"],
                        r["removed_id"],
                        r["present_answer"],
                        r["removed_answer"],
                        r["exact"],
                    )
                    == (held[0], removed[0], pa, ga, str(same)),
                    "Pair table mismatch",
                )
                if (
                    (engine, view, door, method)
                    == ("mujoco", "P3", "placed", "vggt_qwen")
                    and same
                    and pa == ga == "REMOVED"
                ):
                    prim.append(layout)
        published = rows("results/decisions.csv")
        check(len(published) == len(actual) == 960, "Missing decisions")
        seen = set()
        for d in published:
            k = (d["method"], d["id"])
            check(
                k not in seen
                and actual[k]["answer"] == d["answer"]
                and d["truth"] == private[d["id"]]["state"].split("_")[1].upper(),
                "Decision table mismatch",
            )
            seen.add(k)
        summary = read("results/summary.json")
        prior_raw = {
            r["id"]: r
            for r in map(
                json.loads, z.read("paper/registered-prior-outputs.jsonl").splitlines()
            )
        }
        check(len(prior_raw) == 192, "Incomplete original language-model output")
        compared = check_changes(
            rows("paper/previous-pairs.csv"),
            prior_raw,
            rows("results/paired-changes.csv"),
            expected,
            actual,
        )
        check(
            sorted(prim) == summary["primary"]["event_layouts"]
            and len(primary_exact) == summary["primary"]["exact_opportunities"],
            "Primary differs",
        )
        for row in summary["all_views"]:
            for metric in ["false_removals", "definite", "correct"]:
                check(
                    row[metric] == counts[(row["engine"], row["method"], metric)],
                    "Pooled count mismatch",
                )
        return {
            "archive_bytes": bundle.stat().st_size,
            "archive_sha256": digest_file(bundle),
            "case_ids": len(ids),
            "native_and_delivered_pairs": len(joined),
            "unique_native_to_delivered_exports_verified": verify_delivery.cache_info().misses,
            "method_pair_rows": len(expected),
            "native_clearance_checks": len(native_sweeps),
            "matched_corridor_transitions": compared,
            "original_decisions": len(actual),
            "bound_models": list(METHODS),
            "primary_exact_layouts": sorted(primary_exact),
            "primary_double_removed_layouts": sorted(prim),
            "mask_checks": {"/".join(k): n for k, n in sorted(mask_counts.items())},
            "registered_gate_outcomes": {
                "/".join(k): n for k, n in sorted(gates.items())
            },
            "checks": "recounted from decoded native and delivered images, simulator masks, numeric boxes, original model outputs and paper settings",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--build", nargs=2, type=Path, metavar=("RUN", "TABLES"))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.build:
        build(*args.build, args.bundle)
    output = json.dumps(recount(args.bundle), indent=2) + "\n"
    if args.receipt:
        with args.receipt.open("x") as stream:
            stream.write(output)
    print(output, end="")
