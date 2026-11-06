"""Write the machine-readable record of docs/FREEZE.md (SIM_PREREGISTRATION §12): sha256 of every frozen file.

The record is the one fenced JSON block sim/freeze.py reads: {"schema", "freeze_id", "commit", "files": {path: sha256}}.
FROZEN lists the repository files the test path runs or reads (generator, renderer and its test-set launcher,
episodes, evidence, descriptor, policies, selection and analysis code, the run driver and its test specs, configs,
geometry, the frozen candidate and the A / A-nocue model files in frozen/, the preregistration and the pinned notes);
--extra adds more. The tracked tree must be clean, so the commit names exactly what is frozen. This tool only prints
the block; the freeze is the commit of docs/FREEZE.md that contains it, after the researcher's review.

    uv run --no-sync python tools/make_freeze.py --freeze-id simulation-freeze-1 [--extra PATH ...] > record.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FREEZE_SCHEMA = "cmpe249.freeze.v1"   # = sim/freeze.py
FROZEN = (
    "pyproject.toml", "uv.lock",
    "sim/arrange.py", "sim/rig.py", "sim/tabletop.py", "sim/freeze.py", "sim/testsplit.py", "sim/episodes.py",
    "sim/isaac_batch.py", "sim/isaac_rig.py", "sim/render_test.py", "sim/run_render.sh", "sim/run_render_test.sh",
    "sim/alignment.py", "sim/check_batch.py", "sim/true_align_evidence.py", "sim/wood_texture.py",
    "perception/evidence.py", "perception/evidence_v03.py", "perception/common.py",
    "perception/side_descriptor.py", "perception/registration_v032.py",
    "sim/__init__.py", "verify/__init__.py", "analysis/__init__.py",
    "verify/schema.py", "verify/manifest.py", "verify/replay.py", "verify/stopping.py", "verify/policies.py",
    "verify/side.py", "verify/ranker.py", "verify/select.py", "verify/run.py", "verify/lamgrid.py",
    "verify/synthetic.py",
    "analysis/metrics.py", "analysis/bootstrap.py", "analysis/intervals.py", "analysis/common_target.py",
    "tools/pipeline.py", "tools/provenance.py", "tools/watch.py", "tools/merge_dev_sets.py",
    "tools/registration_compare.py", "tools/evidence_summary.py", "tools/test_validity.py",
    "configs/sim/arrange_shelf_v3.json", "configs/sim/arrange_tabletop_v1.json", "configs/rig_geometry.json",
    "configs/sim/tabletop_geometry.json", "configs/sim/positions_shelf_A.json", "configs/sim/positions_tabletop_A.json",
    "configs/sim/ycbv_hulls.json", "configs/sim/tabletop_dev49.json",
    "configs/pipeline/shelf_test.json", "configs/pipeline/tabletop_test.json",
    "configs/pipeline/tabletop_dev49_evidence.json",
    "frozen/frozen_candidate.json", "frozen/frozen_candidate.json.sha256", "frozen/model_A.json",
    "frozen/model_A-nocue.json", "frozen/README.md",
    "docs/SIM_PREREGISTRATION.md", "docs/EXPERIMENT_PLAN.md", "docs/SIMULATION_STUDY.md",
    "notes/2026-11-02-simulation.md", "notes/2026-11-04-registration-repair-plan.md",
    "notes/2026-11-05-side-descriptor.md", "notes/2026-11-05-ranker-a-design.md",
    "notes/2026-11-06-tabletop-dev49-transfer.md", "notes/2026-11-06-pooled-calibration.md",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(freeze_id: str, extra: list[str]) -> dict:
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("the tracked tree is not clean: commit first, so the record names one commit")
    files = {}
    for rel in (*FROZEN, *extra):
        p = REPO / rel
        if not p.is_file():
            raise SystemExit(f"{rel}: no such file")
        files[rel] = sha(p)
    return {"schema": FREEZE_SCHEMA, "freeze_id": freeze_id, "commit": git("rev-parse", "HEAD").strip(),
            "files": files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--freeze-id", required=True)
    ap.add_argument("--extra", action="append", default=[], help="another repository file to freeze (repeatable)")
    a = ap.parse_args(argv)
    print(json.dumps(record(a.freeze_id, a.extra), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
