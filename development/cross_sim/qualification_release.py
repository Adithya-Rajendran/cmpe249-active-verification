"""Bind a completed reviewer decision and two full engine audits to the experimental split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from development.cross_sim.qualification import expected_outputs


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def make_release(manifest_path, protocol_path, visual_path, audit_paths):
    manifest, protocol, visual = map(read, (manifest_path, protocol_path, visual_path))
    split = manifest["split"]
    stages = {"qualification": "qualification-accepted", "pilot": "pilot-inputs-accepted"}
    if split not in stages or manifest["study_id"] != protocol["study_id"]:
        raise ValueError("unknown study or split")
    if visual.get("status") != "PASS" or visual.get("split") != split:
        raise ValueError("completed visual and operator review required")
    if set(audit_paths) != {"isaac", "mujoco"}:
        raise ValueError("expected two engine audits")

    manifest_sha, protocol_sha = digest(manifest_path), digest(protocol_path)
    stops = set(protocol["stops"])
    expected_units = {f"{record['base_id']}/{level}" for record in manifest["arrangements"]
                      for level in protocol[split + "_levels"]}
    engines = {}
    for engine, path in audit_paths.items():
        path = Path(path).resolve()
        audit = read(path)
        if (audit.get("status") != "PASS" or audit["failures"] or audit["engine"] != engine
                or audit["split"] != split or audit["manifest_sha256"] != manifest_sha
                or audit["protocol_sha256"] != protocol_sha or set(audit["units"]) != expected_units):
            raise ValueError(f"incomplete or mismatched {engine} audit")
        root = Path(audit["root"])
        if not root.is_absolute() or not root.is_dir():
            raise ValueError(f"invalid capture root in {engine} audit")
        expected_files = set()
        for unit, record in audit["units"].items():
            if record["failures"] or set(record["views"]) != stops:
                raise ValueError(f"incomplete review of {engine}/{unit}")
            states = list(protocol["states"])
            if unit.endswith("/D0"):
                states.append("reference")
            if split == "qualification":
                states.append("repeat_clear_present")
            marker = "completion.json" if engine == "isaac" else "complete.json"
            expected_files.update(f"{unit}/{name}" for name in expected_outputs(stops, states) | {marker})
        if set(audit["files"]) != expected_files:
            raise ValueError(f"incomplete file ledger in {engine} audit")
        engines[engine] = {"root": str(root), "audit_path": str(path), "audit_sha256": digest(path)}
    return {"study_id": manifest["study_id"], "stage": stages[split], "status": "PASS",
            "manifest_sha256": manifest_sha, "protocol_sha256": protocol_sha,
            "visual_review_sha256": digest(visual_path), "engines": engines,
            "timestamp": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "visual-review", "isaac-audit", "mujoco-audit", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = make_release(args.manifest, Path(__file__).with_name("qualification_v1.json"),
                          args.visual_review, {"isaac": args.isaac_audit, "mujoco": args.mujoco_audit})
    with args.out.open("x") as dest:
        json.dump(result, dest, indent=2)
        dest.write("\n")


if __name__ == "__main__":
    main()
