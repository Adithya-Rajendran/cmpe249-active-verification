"""Bundle and verify the scored images, native images and masks, raw answers, and corridor recipes."""

import argparse
import csv
import hashlib
import io
import json
import tempfile
import zipfile
from functools import cache
from pathlib import Path

from PIL import Image, ImageChops

PAPER_NAMES = {
    "baseline_200": "homography",
    "geometry": "dense",
    "vggt_qwen": "registered",
}
ARCHIVE_GUIDE = """Service corridor transfer study

The simulations place a YCB bleach bottle or power drill behind a sliding door.
Isaac Sim and MuJoCo use the same eight corridor layouts.
ASSET-CREDITS.md identifies the object models and their license.

Open PROTOCOL.md for the comparison rules and evidence/summary.json for the findings.
Open evidence/corridor-examples.png to see the two targets and the answers from both simulators.
Open evidence/corridor-angle-example.png to compare three angles in the same MuJoCo layout.
Use evidence/decisions.csv to find each answer and its anonymous case ID.
Use inputs/cases.json to find the two PNGs and the target box for that ID.
Only inputs/ went to the scorers. The native/ index links each case to the original RGB images and masks.
Open evidence/pairs.csv to compare the retained and removed states of the same scene.
Open scores/ for every raw verifier response, its settings and the corresponding paper settings.
Use scenes/ to recover the nominal and moved camera definitions for each layout.
To repeat the checks, extract verify_bundle.py and run it with Python and Pillow:
python3 verify_bundle.py verify corridor-transfer-inputs.zip
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_native(archive, pairs, case_ids):
    cases = json.loads(archive.read("native/cases.json"))
    if set(cases) != set(case_ids):
        raise ValueError("Incomplete native inventory")

    def load_image(digest):
        data = archive.read("native/images/" + digest + ".png")
        if sha(data) != digest:
            raise ValueError("Native image differs from its source digest")
        with Image.open(io.BytesIO(data)) as image:
            if image.size != (2304, 1296):
                raise ValueError("Unexpected native resolution")
            return image.copy()

    @cache
    def rgb(digest):
        image = load_image(digest)
        if image.mode != "RGB":
            raise ValueError("Invalid native RGB encoding")
        return sha(image.tobytes())

    @cache
    def area(digest):
        image = load_image(digest)
        if image.mode != "L":
            raise ValueError("Invalid mask encoding")
        return sum(image.histogram()[1:])

    for item in cases.values():
        for key in ("reference", "query"):
            rgb(item[key])
        for key in ("reference_mask", "query_mask", "blocker_mask", "alone_mask"):
            area(item[key])
    exact = 0
    for row in pairs:
        a, b = cases[row["present_id"]], cases[row["removed_id"]]
        same_reference = rgb(a["reference"]) == rgb(b["reference"])
        same_query = rgb(a["query"]) == rgb(b["query"])
        if (same_reference, same_query) != (
            row["native_reference_equal"] == "True", row["native_query_equal"] == "True"
        ):
            raise ValueError("Native image equality differs from the pair certificate")
        same = same_reference and same_query and row["delivered_complete_input_equal"] == "True"
        if same != (row["complete_input_equal"] == "True"):
            raise ValueError("Full input equality differs from the pair certificate")
        exact += same
        if area(b["query_mask"]) or area(a["query_mask"]) != int(row["visible_pixels"]):
            raise ValueError("Target mask differs from the intervention record")
        overlap = ImageChops.darker(load_image(a["query_mask"]), load_image(a["alone_mask"]))
        fraction = sum(overlap.histogram()[1:]) / area(a["alone_mask"])
        if abs(fraction - float(row["visible_fraction"])) > 1e-12:
            raise ValueError("Target visibility differs from the evaluator")
    return exact


def verify(bundle):
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt archive")
        case_data = z.read("inputs/cases.json")
        cases = json.loads(case_data)
        keys = {c["id"]: c for c in cases}
        if len(cases) != 192 or len(keys) != 192:
            raise ValueError("Incomplete case inventory")
        lock_data = z.read("evidence/endpoint-lock.json")
        lock = json.loads(lock_data)
        design = json.loads(z.read("evidence/design-release.json"))
        if (
            design["endpoint_lock_sha256"] != sha(lock_data)
            or design["manifest_sha256"] != sha(z.read("scenes/manifest.json"))
            or lock["protocol_sha256"] != sha(z.read("PROTOCOL.md"))
            or lock["declaration_sha256"] != sha(z.read("declaration.json"))
        ):
            raise ValueError("Archived design differs from the pre-score release")
        if sha(z.read("scorer-specification.json")) != lock["original_task_sha256"]:
            raise ValueError("Verifier instructions differ from the locked task")

        @cache
        def decoded(filename, digest):
            data = z.read("inputs/" + filename)
            if sha(data) != digest:
                raise ValueError("Image bytes differ from case inventory")
            with Image.open(io.BytesIO(data)) as im:
                if im.mode != "RGB" or im.size != (1200, 675) or im.info:
                    raise ValueError("Unexpected image format")
                return sha(im.tobytes())

        for c in cases:
            for role in ("reference", "query"):
                decoded(c[role], c[role + "_sha256"])
        decision_rows = csv.DictReader(io.StringIO(z.read("evidence/decisions.csv").decode()))
        answers = {}
        for row in decision_rows:
            key = row["method"], row["id"]
            if key in answers:
                raise ValueError("Duplicate decision in evidence table")
            answers[key] = row["answer"]
        if set(answers) != {(method, uid) for method in PAPER_NAMES for uid in keys}:
            raise ValueError("Incomplete decision table")
        summary = json.loads(z.read("evidence/summary.json"))
        for method, paper_name in PAPER_NAMES.items():
            base = "scores/" + method + "/"
            original = z.read(base + "paper-binding.json")
            binding = json.loads(z.read(base + "binding.json"))
            prior = json.loads(original)
            if sha(original) != lock["paper_binding_sha256"][paper_name]:
                raise ValueError("Original scorer settings differ from the endpoint lock")
            if binding.pop("cases_sha256") != sha(case_data):
                raise ValueError("Scorer received a different input packet")
            prior.pop("cases_sha256")
            if binding != prior:
                raise ValueError("Corridor scorer settings differ from the paper")
            data = z.read(base + "records.jsonl")
            if sha(data) != summary["records_sha256"][method]:
                raise ValueError("Raw scores differ from the evaluation")
            records = [json.loads(line) for line in data.splitlines()]
            if len(records) != len(keys) or {r["id"] for r in records} != set(keys):
                raise ValueError("Incomplete raw scorer output")
            for r in records:
                c = keys[r["id"]]
                if r["answer"] != answers[method, r["id"]] or any(
                    r[k] != c[k] for k in ("reference_sha256", "query_sha256", "box_xyxy")
                ):
                    raise ValueError("Raw answer or input differs from the published table")
        rows = list(csv.DictReader(io.StringIO(z.read("evidence/pairs.csv").decode())))
        pairs = [r for r in rows if r["method"] == "vggt_qwen"]
        if (
            len(pairs) != 96
            or len({(r["present_id"], r["removed_id"]) for r in pairs}) != 96
        ):
            raise ValueError("Incomplete pair inventory")
        exact = 0
        for r in pairs:
            a, b = keys[r["present_id"]], keys[r["removed_id"]]
            same = a["box_xyxy"] == b["box_xyxy"] and all(
                decoded(a[k], a[k + "_sha256"]) == decoded(b[k], b[k + "_sha256"])
                for k in ("reference", "query")
            )
            if same != (r["delivered_complete_input_equal"] == "True"):
                raise ValueError(
                    "Decoded pair disagrees with the independent evaluation"
                )
            exact += same
        exact_native_and_delivered = check_native(z, pairs, keys)
        manifest = json.loads(z.read("scenes/manifest.json"))
        scenes = [
            level for a in manifest["arrangements"] for level in a["levels"].values()
        ]
        if len(scenes) != 16 or any(
            sha(z.read("scenes/" + v["path"])) != v["sha256"] for v in scenes
        ):
            raise ValueError("Invalid scene recipes")
        return {
            "verified_cases": len(cases),
            "verified_delivered_pairs": len(pairs),
            "exact_delivered_pairs": exact,
            "verified_native_pairs": len(pairs),
            "exact_native_and_delivered_pairs": exact_native_and_delivered,
            "verified_scene_recipes": len(scenes),
            "verified_raw_decisions": len(answers),
            "cases_sha256": sha(case_data),
        }


def pack(run, tables, dest):
    project = Path(__file__).resolve().parents[2]
    release = read(run / "evaluation/scorer-ready.json")
    packet = Path(release["cases_path"]).parent
    ledger = Path(release["private_ledger_path"])
    if sha(ledger.read_bytes()) != release["private_ledger_sha256"]:
        raise ValueError("Native inventory differs from the pre-score release")
    cases = read(packet / "cases.json")
    files = {
        "inputs/cases.json": packet / "cases.json",
        "scenes/manifest.json": run / "cohort/manifest.json",
        "scorer-specification.json": project
        / "paper/evidence/design/specification.json",
        "ASSET-CREDITS.md": project / "paper/evidence/ASSET-CREDITS.md",
        "PROTOCOL.md": Path(__file__).with_name("PROTOCOL.md"),
        "declaration.json": Path(__file__).with_name("declaration.json"),
        "verify_bundle.py": Path(__file__),
    }
    native = {}
    for row in read(ledger)["rows"]:
        native[row["id"]] = {}
        for role, image in row["files"].items():
            digest = image["sha256"]
            native[row["id"]][role] = digest
            files["native/images/" + digest + ".png"] = Path(image["path"])
    for row in cases:
        for key in ("reference", "query"):
            name = row[key]
            if Path(name).name != name:
                raise ValueError("Invalid image path")
            files["inputs/" + name] = packet / name
    for row in read(run / "cohort/manifest.json")["arrangements"]:
        for item in row["levels"].values():
            name = item["path"]
            if Path(name).name != name:
                raise ValueError("Invalid recipe path")
            files["scenes/" + name] = run / "cohort" / name
    for name in (
        "decisions.csv",
        "pairs.csv",
        "summary.json",
        "pose-comparisons.csv",
        "method-by-view.csv",
        "independent-review.json",
        "camera-angles.csv",
        "camera-verification.json",
        "capture-receipt.json",
        "input-qualification.json",
        "corridor-examples.png",
        "illustrated-cases.json",
        "corridor-angle-example.png",
        "illustrated-angles.json",
    ):
        files["evidence/" + name] = tables / name
    for name in ("design-release.json", "endpoint-lock.json"):
        files["evidence/" + name] = run / name
    for method, paper_name in PAPER_NAMES.items():
        base = "scores/" + method + "/"
        for name in ("binding.json", "records.jsonl"):
            files[base + name] = run / "scoring" / method / name
        files[base + "paper-binding.json"] = (
            project / "paper/evidence/raw-scores" / paper_name / "binding.json"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        intermediate = Path(tmp) / dest.name
        with zipfile.ZipFile(
            intermediate, "w", zipfile.ZIP_DEFLATED, compresslevel=3
        ) as z:
            z.writestr("README.txt", ARCHIVE_GUIDE)
            z.writestr("native/cases.json", json.dumps(native, sort_keys=True, indent=2) + "\n")
            for name, path in sorted(files.items()):
                z.write(path, name)
        result = verify(intermediate)
        intermediate.rename(dest)
    return {
        **result,
        "bytes": dest.stat().st_size,
        "archive_sha256": sha(dest.read_bytes()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    make = sub.add_parser("pack")
    make.add_argument("run", type=Path)
    make.add_argument("tables", type=Path)
    make.add_argument("bundle", type=Path)
    check = sub.add_parser("verify")
    check.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = (
        verify(args.bundle)
        if args.mode == "verify"
        else pack(args.run, args.tables, args.bundle)
    )
    print(json.dumps(result, indent=2))
