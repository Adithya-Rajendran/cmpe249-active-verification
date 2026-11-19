"""Independently bind the three raw prediction streams, native captures and shareable audit tables."""

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
        return h.hexdigest()


@lru_cache(maxsize=6000)
def pixels(file, width, height):
    with Image.open(file) as im:
        if im.size != (width, height):
            raise ValueError(f"Unexpected dimensions: {im.size}")
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


def native_mask(file):
    with Image.open(file) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        a = np.any(a[:, :, :3] != 0, axis=2)
    return a != 0


@lru_cache(maxsize=1400)
def exposure(file, isolated_file):
    target = native_mask(file)
    isolated = native_mask(isolated_file)
    checked(target.shape == isolated.shape == (1296, 2304), "Native mask size mismatch")
    area = int(np.count_nonzero(isolated))
    checked(area > 0, "Empty isolated target mask")
    return (int(np.count_nonzero(target)), area, int(np.count_nonzero(target & isolated)))


def table(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def export(packet, ledger, methods, evidence, image_path):
    cases = json.loads((packet / "cases.json").read_text())
    public = {row["id"]: row for row in cases}
    private = json.loads(ledger.read_text())
    checked(
        private["cases_sha256"] == sha(packet / "cases.json"),
        "Private ledger binding failed",
    )
    ids = set(public)
    checked(len(cases) == len(ids) == 1280, "Incomplete public inventory")
    streams = {}
    for method, file in methods.items():
        raw = [
            json.loads(line) for line in file.read_text().splitlines() if line.strip()
        ]
        stream = {r["id"]: r for r in raw}
        checked(
            set(stream) == ids and len(raw) == len(ids), f"Incomplete scores: {method}"
        )
        streams[method] = stream
    with (evidence / "arrangements.csv").open() as f:
        layouts = {r["arrangement"]: r for r in csv.DictReader(f)}
    query_rows = []
    asset_table = {}
    pairs = defaultdict(dict)
    evidence.mkdir(exist_ok=True, parents=True)

    def add_image(role, row):
        path = packet / row[role]
        digest = row[role + "_sha256"]
        if sha(path) != digest:
            raise ValueError("PNG checksum mismatch")
        if digest not in asset_table:
            asset_table[digest] = path
        return pixels(path, 1200, 675)

    for item in private["rows"]:
        uid = item["id"]
        c = public[uid]
        cap = item["capture"]
        state = item["state"]
        data = cap["images"][state]
        meta = layouts[cap["arrangement"]]
        checked(meta["environment"] == cap["environment"], "Layout environment changed")
        box = json.dumps(c["box_xyxy"], separators=(",", ":"))
        visible, isolated, overlap = exposure(data["target_mask"]["path"], cap["isolated_target_mask"]["path"])
        rr = {
            "id": uid,
            "engine": cap["engine"],
            "environment": cap["environment"],
            "arrangement": cap["arrangement"],
            "target": meta["target"],
            "blocker": meta["blocker"],
            "stop": cap["stop"],
            "level": cap["level"],
            "state": state,
            "truth": "PRESENT" if state.endswith("_present") else "REMOVED",
            "box": box,
            "task": "paper-audit-v1 fixed method-specific task",
            "reference_file_sha256": c["reference_sha256"],
            "query_file_sha256": c["query_sha256"],
            "reference_pixels_sha256": add_image("reference", c),
            "query_pixels_sha256": add_image("query", c),
            "reference_native_pixels_sha256": pixels(
                cap["reference"]["rgb"]["path"], 2304, 1296
            ),
            "query_native_pixels_sha256": pixels(data["rgb"]["path"], 2304, 1296),
            "visible_pixels": visible,
            "isolated_pixels": isolated,
            "visible_isolated_overlap_pixels": overlap,
        }
        if rr["truth"] == "REMOVED" and rr["visible_pixels"]:
            raise ValueError("Removed target still in target mask")
        for method, stream in streams.items():
            score = stream[uid]
            checked(
                all(
                    score[k] == c[k]
                    for k in ("box_xyxy", "reference_sha256", "query_sha256")
                ),
                "Scored different input",
            )
            answer = score["answer"]
            failure = score.get("failure")
            execution_error = bool(failure) and failure not in (
                "registration_gate",
                "invalid_model_output",
            )
            rr[method + "_execution_error"] = str(failure) if execution_error else ""
            if execution_error:
                checked(
                    answer == "UNKNOWN", "Execution failure returned a definite answer"
                )
                rr[method + "_gate"] = "execution_error"
                if method in ("homography", "dense"):
                    for key in ("quality", "match_score", "obstruction"):
                        rr[method + "_" + key] = None
                if method in ("dense", "registered"):
                    rr[method + "_valid_fraction"] = score.get("valid_fraction")
                if method == "registered":
                    rr["registered_raw"] = score.get("raw") or ""
            elif method in ("homography", "dense"):
                ev = score["evidence"]
                aligned = score["diagnostics"]["align_ok"]
                check = "UNKNOWN"
                if aligned and ev["quality"] >= 0.30:
                    if ev["match_score"] >= 0.70:
                        check = "PRESENT"
                    elif ev["match_score"] <= 0.60 and ev["obstruction"] <= 0.40:
                        check = "REMOVED"
                checked(answer == check, "Independent technical decision disagrees")
                for key in ("quality", "match_score", "obstruction"):
                    rr[method + "_" + key] = ev[key]
                rr[method + "_gate"] = score["refusal"]
                if method == "dense":
                    rr["dense_valid_fraction"] = score["diagnostics"]["registration"][
                        "target_valid_fraction"
                    ]
            else:
                if score["failure"] == "registration_gate":
                    check = "UNKNOWN"
                elif score.get("raw") in ("PRESENT", "REMOVED", "UNKNOWN"):
                    check = score["raw"]
                elif score["failure"] == "invalid_model_output":
                    check = "UNKNOWN"
                else:
                    raise ValueError(
                        f"Unexplained model failure: {score.get('failure')}"
                    )
                checked(check == answer, "Independent model output parse differs")
                rr["registered_raw"] = score.get("raw") or ""
                rr["registered_valid_fraction"] = score["valid_fraction"]
                rr["registered_gate"] = score.get("failure") or "accepted"
            rr[method] = answer
            rr[method + "_seconds"] = score["runtime_seconds"]
        key = tuple(
            rr[k] for k in ("engine", "environment", "arrangement", "level", "stop")
        ) + (state.split("_")[0],)
        checked(rr["truth"] not in pairs[key], "Duplicate counterfactual")
        pairs[key][rr["truth"]] = rr
        query_rows.append(rr)
    checked(
        len(query_rows) == 1280 and len(pairs) == 640, "Incomplete private inventory"
    )
    pair_rows = []
    for key, states in sorted(pairs.items()):
        a, b = states["PRESENT"], states["REMOVED"]
        shared = ("reference_pixels_sha256", "query_pixels_sha256", "box", "task")
        exact = all(a[k] == b[k] for k in shared)
        native = all(
            a[k] == b[k]
            for k in ("reference_native_pixels_sha256", "query_native_pixels_sha256")
        )
        rec = dict(
            zip(
                ("engine", "environment", "arrangement", "level", "stop", "placement"),
                key,
            )
        )
        rec.update(
            present_id=a["id"],
            removed_id=b["id"],
            exact=int(exact),
            native_exact=int(native),
        )
        if exact:
            for role in ("reference", "query"):
                with (
                    Image.open(packet / public[a["id"]][role]) as im_a,
                    Image.open(packet / public[b["id"]][role]) as im_b,
                ):
                    checked(
                        im_a.convert("RGB").tobytes() == im_b.convert("RGB").tobytes(),
                        "Direct byte check failed",
                    )
        pair_rows.append(rec)
    table(
        evidence / "queries.csv",
        sorted(
            query_rows,
            key=lambda r: (
                r["engine"],
                r["arrangement"],
                r["level"],
                r["stop"],
                r["state"],
            ),
        ),
    )
    table(evidence / "pairs.csv", pair_rows)
    if image_path:
        with zipfile.ZipFile(
            image_path, "x", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as z:
            for digest, file in sorted(asset_table.items()):
                z.write(file, "rgb/" + digest + ".png")
            z.write(packet / "cases.json", "cases.json")
            for file in ("queries.csv", "pairs.csv", "arrangements.csv"):
                z.write(evidence / file, file)
            z.writestr(
                "README.txt",
                "Full delivered 1200 by 675 input RGBs. Join reference_file_sha256 and query_file_sha256 in queries.csv to rgb/<hash>.png. Copying a reference to each case is unnecessary. Predictions and truth are for evaluation only.\n",
            )
    receipt = {
        "case_count": len(query_rows),
        "pairs": len(pair_rows),
        "unique_png_files": len(asset_table),
        "packet_inventory_sha256": sha(packet / "cases.json"),
        "ledger_sha256": sha(ledger),
        "score_sha256": {m: sha(f) for m, f in methods.items()},
        "queries_sha256": sha(evidence / "queries.csv"),
        "pairs_sha256": sha(evidence / "pairs.csv"),
    }
    if image_path:
        receipt.update(
            rgb_archive_sha256=sha(image_path),
            rgb_archive_bytes=image_path.stat().st_size,
        )
    (evidence / "export-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "ledger", "homography", "dense", "registered", "evidence"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--image-archive", type=Path)
    a = p.parse_args()
    export(
        a.packet,
        a.ledger,
        {m: getattr(a, m) for m in ("homography", "dense", "registered")},
        a.evidence,
        a.image_archive,
    )
