"""Prepare one blinded packet from completed, verified level-shelf captures."""

from __future__ import annotations
import argparse
import json
import platform
import secrets
from pathlib import Path
import numpy as np
import PIL
from development.paper_audit.inputs_build import sha, box_from_mask, clean
from development.paper_audit.evaluate_calibration import check_geometry

STATES = ("clear_present", "clear_removed", "placed_present", "placed_removed")
ENGINES = ("isaac", "mujoco")
LAYOUTS = tuple(f"PAM{i:02}" for i in range(1, 9))
STOPS = tuple(f"P{i}" for i in range(1, 6))


def index_capture(root):
    rows = []
    for engine in ENGINES:
        for arrangement in LAYOUTS:
            completions = {}
            for level in ("D0", "D6"):
                folder = Path(root) / engine / arrangement / level
                marker = folder / (
                    "completion.json" if engine == "isaac" else "complete.json"
                )
                data = json.loads(marker.read_text())
                for name, digest in data["outputs"].items():
                    file = (folder / name).resolve()
                    if folder.resolve() not in file.parents or sha(file) != digest:
                        raise ValueError("capture completion mismatch")
                completions[level] = data["outputs"]
                observed = folder / "observed.json"
                if data["outputs"].get("observed.json") != sha(observed):
                    raise ValueError("observed file not bound")
            nominal = Path(root) / engine / arrangement / "D0"
            query = Path(root) / engine / arrangement / "D6"
            for stop in STOPS:
                ref = nominal / "rgb" / f"reference_{stop}.png"
                refmask = nominal / "masks" / f"target_reference_{stop}.png"
                for state in STATES:
                    rgb = query / "rgb" / f"{state}_{stop}.png"
                    tm = query / "masks" / f"target_{state}_{stop}.png"
                    bm = query / "masks" / f"blocker_{state}_{stop}.png"
                    alone = query / "masks" / f"alone_{stop}.png"
                    files = {
                        "reference": ref,
                        "reference_mask": refmask,
                        "query": rgb,
                        "query_mask": tm,
                        "blocker_mask": bm,
                        "alone_mask": alone,
                    }
                    records = {}
                    for role, path in files.items():
                        level = (
                            "D0" if role in ("reference", "reference_mask") else "D6"
                        )
                        rel = str(path.relative_to(nominal if level == "D0" else query))
                        digest = sha(path)
                        if completions[level].get(rel) != digest:
                            raise ValueError("input absent from completion receipt")
                        records[role] = {"path": str(path.resolve()), "sha256": digest}
                    rows.append(
                        {
                            "engine": engine,
                            "arrangement": arrangement,
                            "stop": stop,
                            "state": state,
                            "files": records,
                        }
                    )
    if len(rows) != 320:
        raise ValueError("incomplete declared cohort")
    return rows


def prepare(root, packet, ledger, verification):
    packet = Path(packet)
    ledger = Path(ledger)
    if packet.exists() or ledger.exists():
        raise FileExistsError("destination exists")
    if packet.resolve() in ledger.resolve().parents:
        raise ValueError("private ledger inside packet")
    run = Path(root).resolve().parent
    release = run / "design-release.json"
    design = json.loads(release.read_text())
    if (
        sha(run / "cohort" / "manifest.json") != design["manifest_sha256"]
        or sha("development/level_shelf/PROTOCOL.md") != design["protocol_sha256"]
        or sha("development/level_shelf/declaration.json")
        != design["declaration_sha256"]
    ):
        raise ValueError("frozen design changed")
    checks = []
    accepted = False
    geometry_engines = set()
    mask_engines = set()
    for path in verification:
        p = Path(path)
        data = json.loads(p.read_text())
        if (
            data.get("errors")
            or any(r.get("failures") for r in data.get("rows", []))
            or data.get("status") in ("fail", "FAIL")
        ):
            raise ValueError("capture integrity check failed")
        if data.get("passed") is True:
            if (
                data.get("manifest_sha256") != design["manifest_sha256"]
                or data.get("engines") != list(ENGINES)
                or data.get("layouts") != list(LAYOUTS)
                or data.get("source_records") != 32
                or data.get("camera_poses_checked") != 320
                or not 0 <= data.get("max_leveled_edge_abs_deg", float("inf")) < 0.05
            ):
                raise ValueError("camera verification incomplete")
            observed = data.get("observed_files", {})
            expected = {
                f"operator/{e}/{a}/{lv}/observed.json"
                for e in ENGINES
                for a in LAYOUTS
                for lv in ("D0", "D6")
            }
            if set(observed) != expected or any(
                sha(run / k) != v for k, v in observed.items()
            ):
                raise ValueError("camera receipt observed binding mismatch")
            accepted = True
        if "inputs_sha256" in data and "rows" in data:
            verified = check_geometry([p])
            for engine in ENGINES:
                expected_observed = {
                    str((Path(root) / engine / a / lv / "observed.json").resolve())
                    for a in LAYOUTS
                    for lv in ("D0", "D6")
                }
                if expected_observed <= set(verified):
                    geometry_engines.add(engine)
        if "errors" in data and "rows" in data:
            expected_rows = {
                (a, lv, s) for a in LAYOUTS for lv in ("D0", "D6") for s in STOPS
            }
            actual = {(r["base_id"], r["level"], r["stop"]) for r in data["rows"]}
            if len(data["rows"]) != 80 or actual != expected_rows:
                raise ValueError("native integrity audit incomplete")
            for engine in ENGINES:
                if p.name.startswith(engine + "-"):
                    mask_engines.add(engine)
        checks.append({"path": str(p.resolve()), "sha256": sha(p)})
    if not accepted or geometry_engines != set(ENGINES) or mask_engines != set(ENGINES):
        raise ValueError("full camera, geometry and native integrity receipts required")
    rows = index_capture(root)
    packet.mkdir(parents=True)
    public = []
    private = []
    for row in rows:
        uid = secrets.token_hex(12)
        reference = packet / (uid + "-reference.png")
        query = packet / (uid + "-query.png")
        clean(row["files"]["reference"]["path"], reference)
        clean(row["files"]["query"]["path"], query)
        box = box_from_mask(row["files"]["reference_mask"]["path"])
        public.append(
            {
                "id": uid,
                "reference": reference.name,
                "query": query.name,
                "box_xyxy": box,
                "reference_sha256": sha(reference),
                "query_sha256": sha(query),
            }
        )
        private.append({"id": uid, **row})
    secrets.SystemRandom().shuffle(public)
    (packet / "cases.json").write_text(json.dumps(public, indent=2) + "\n")
    metadata = {
        "python": platform.python_version(),
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "cases_sha256": sha(packet / "cases.json"),
    }
    (packet / "packet-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "design_release_sha256": sha(release),
                "cases_sha256": sha(packet / "cases.json"),
                "builder_sha256": sha(__file__),
                "verification": checks,
                "rows": private,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--packet", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--verification", action="append", required=True)
    a = p.parse_args()
    prepare(a.root, a.packet, a.ledger, a.verification)
