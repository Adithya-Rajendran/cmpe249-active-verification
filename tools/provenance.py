"""Read evidence outputs only through the provenance the evidence stage recorded (stdlib).

The evidence stage writes RUN_DIR/evidence_provenance.json: the job's output directory ("out") and the sha256 and
record count of every output. Downstream stages (select, the summaries) resolve "{out}" templates against that
directory and refuse a file the job did not write or whose bytes changed since it was recorded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(run_dir: Path) -> dict:
    path = Path(run_dir) / "evidence_provenance.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the evidence stage for this run first")
    prov = json.loads(path.read_text())
    if "out" not in prov:   # runs before per-job output dirs wrote every output to one shared out_dir
        dirs = {os.path.dirname(o) for o in prov["outputs"]}
        if len(dirs) != 1:
            raise SystemExit(f"{path}: no 'out' and the outputs are in {len(dirs)} directories")
        prov["out"] = dirs.pop()
    return prov


def output(prov: dict, template: str, **fmt: str) -> Path:
    """The evidence file `template` names ({out} = the job's output dir), after checking it is one of the job's
    recorded outputs and still has the recorded sha256."""
    try:
        path = template.format(out=prov["out"], **fmt)
    except KeyError as e:
        raise SystemExit(f"{template}: unknown placeholder {e}")
    rec = prov["outputs"].get(path)
    if rec is None:
        raise SystemExit(f"{path} is not an output of evidence job {prov.get('key')}")
    if not Path(path).exists():
        raise SystemExit(f"{path} was recorded by evidence job {prov.get('key')} but is gone")
    if sha256(Path(path)) != rec["sha256"]:
        raise SystemExit(f"{path} changed since evidence job {prov.get('key')} recorded it (sha256 differs)")
    return Path(path)
