"""Verify the complete blind packet and publish its scorer-ready receipt."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from development.level_shelf.prepare import ENGINES, LAYOUTS, STOPS, STATES
from development.paper_audit.inputs_build import sha, box_from_mask
from development.paper_audit.evaluate_witness import primary_rgb, mask


def validate(packet, ledger_path, output):
    packet = Path(packet)
    ledger_path = Path(ledger_path)
    ledger = json.loads(ledger_path.read_text())
    cases = json.loads((packet / "cases.json").read_text())
    run = packet.resolve().parents[1]
    release = run / "design-release.json"
    if (
        sha(release) != ledger["design_release_sha256"]
        or sha(packet / "cases.json") != ledger["cases_sha256"]
    ):
        raise ValueError("release binding mismatch")
    for proof in ledger["verification"]:
        if sha(proof["path"]) != proof["sha256"]:
            raise ValueError("verification changed")
    public = {c["id"]: c for c in cases}
    private = {c["id"]: c for c in ledger["rows"]}
    if (
        len(cases) != 320
        or len(public) != 320
        or len(private) != 320
        or set(public) != set(private)
    ):
        raise ValueError("inventory mismatch")
    expected = {
        (e, a, s, t) for e in ENGINES for a in LAYOUTS for s in STOPS for t in STATES
    }
    found = set()
    for uid, c in public.items():
        if set(c) != {
            "id",
            "reference",
            "query",
            "box_xyxy",
            "reference_sha256",
            "query_sha256",
        }:
            raise ValueError("unexpected public field")
        p = private[uid]
        key = (p["engine"], p["arrangement"], p["stop"], p["state"])
        if key in found:
            raise ValueError("duplicate physical case")
        found.add(key)
        files = p["files"]
        for rec in files.values():
            if sha(rec["path"]) != rec["sha256"]:
                raise ValueError("native file changed")
        if c["box_xyxy"] != box_from_mask(files["reference_mask"]["path"]):
            raise ValueError("reference box mismatch")
        if p["state"].endswith("removed") and mask(files["query_mask"]["path"]).any():
            raise ValueError("removed target mask not empty")
        for role in ("reference", "query"):
            path = (packet / c[role]).resolve()
            if path.parent != packet.resolve() or sha(path) != c[role + "_sha256"]:
                raise ValueError("packet image mismatch")
            with Image.open(path) as im:
                if (
                    im.size != (1200, 675)
                    or im.mode != "RGB"
                    or im.info
                    or not np.array_equal(
                        np.asarray(im), primary_rgb(files[role]["path"])
                    )
                ):
                    raise ValueError("native image export mismatch")
    if found != expected:
        raise ValueError("declared conditions incomplete")
    spec = Path("runs/2026-11-18-paper-audit-v1/scoring/specification.json")
    if sha(spec) != "db57d40b5d55ba40b0b3fcdcb7a449c322bb861d4cb401c29361f6c5f5f045ed":
        raise ValueError("original task changed")
    out = Path(output)
    if out.exists():
        raise FileExistsError("receipt exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            {
                "status": "pass",
                "cases_path": str((packet / "cases.json").resolve()),
                "cases_sha256": sha(packet / "cases.json"),
                "private_ledger_path": str(ledger_path.resolve()),
                "private_ledger_sha256": sha(ledger_path),
                "total_cases": len(cases),
                "original_task_sha256": sha(spec),
                "design_release_sha256": sha(release),
                "verifications": ledger["verification"],
                "validator_sha256": sha(__file__),
            },
            indent=2,
        )
        + "\n"
    )
    temp.replace(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--packet", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    validate(a.packet, a.ledger, a.out)
