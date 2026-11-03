"""Run the frozen evidence step (perception/evidence.py, unmodified) on ONE split of public-data episodes.

Why a separate runner: perception.evidence.main always registers with capture/tags.py (ArUco tags our
shelf has and OCID does not). Each OCID camera is static within a sequence, so query and reference
are already in the same pixels; this runner calls the same `run_jobs` with
`perception.evidence.identity_align` instead. Everything else (models, Params, scores, diagnostics)
is the frozen module's own code, so evidence on OCID is evidence from exactly the version under study.
evidence_run.json ties a run to both sides: sha256 of perception/evidence.py, and sha256 of the
jobs.json / truth.json / truth_detail.jsonl / MANIFEST.json it scored (plus the loader version and
sha256 from MANIFEST).

One split per run. Every episode is dev or val by arrangement (public_data/ocid.py). The split of each
job is read from truth_detail.jsonl next to jobs.json and checked against MANIFEST.json's arrangement
lists; a job without a detail record is an error. `--split` is required, and val is refused unless
`--allow-val` is given too: val is kept back to check choices made on dev, so running it is a decision,
never a default.

Usage (repo root, project env on the GB10):
  uv run --no-sync python -m public_data.run_evidence --jobs EPISODES/jobs.json --split dev \
      [--out-dir DIR] [--device cuda] [--force]
  (--split val also needs --allow-val)
Writes <out-dir>/evidence.jsonl (EvidenceStore format), <out-dir>/diagnostics.jsonl and
<out-dir>/evidence_run.json (with the split, its job count and the job count per split in jobs.json).
--out-dir defaults to <jobs dir>/evidence/<EVIDENCE_VERSION>/<split>/ (a rebuild of the episodes with
--force deletes <jobs dir>/evidence/). An existing evidence.jsonl is refused unless force.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from perception import evidence as ev

RUN_SCHEMA = "cmpe249.ocid-evidence-run.v0"
SPLITS = ("dev", "val")


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _check_split(split: str, allow_val: bool) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "val" and not allow_val:
        raise ValueError("refusing to run the val split without allow_val (--allow-val): val is kept back "
                         "to check choices made on dev")


def default_out_dir(jobs_path: str | Path, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return Path(jobs_path).resolve().parent / "evidence" / ev.EVIDENCE_VERSION / split


def episode_splits(ep_dir: str | Path) -> dict[str, str]:
    """episode_id -> split, from <ep_dir>/truth_detail.jsonl. When <ep_dir>/MANIFEST.json lists the
    arrangements per split, every record's split must agree with its arrangement's entry there."""
    ep_dir = Path(ep_dir)
    detail_path = ep_dir / "truth_detail.jsonl"
    if not detail_path.is_file():
        raise FileNotFoundError(f"{detail_path} not found: the split of each job is read from it")
    manifest_path = ep_dir / "MANIFEST.json"
    arr_split: dict[str, str] = {}
    if manifest_path.is_file():
        for sp, arrs in (json.loads(manifest_path.read_text()).get("arrangements") or {}).items():
            for a in arrs:
                arr_split[a] = sp
    out: dict[str, str] = {}
    for n, line in enumerate(detail_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        eid, sp = d["episode_id"], d.get("split")
        if sp not in SPLITS:
            raise ValueError(f"{detail_path}:{n}: {eid} has split {sp!r}")
        if arr_split and arr_split.get(d.get("arrangement")) != sp:
            raise ValueError(f"{detail_path}:{n}: {eid} is {sp} but MANIFEST.json puts arrangement "
                             f"{d.get('arrangement')!r} in {arr_split.get(d.get('arrangement'))!r}")
        if out.setdefault(eid, sp) != sp:
            raise ValueError(f"{detail_path}:{n}: {eid} appears with two splits")
    return out


def select_jobs(jobs_path: str | Path, split: str, allow_val: bool = False
                ) -> tuple[dict[str, Any], bytes, list[dict[str, Any]], dict[str, int]]:
    """(jobs doc, its bytes, this split's jobs in file order, job count per split in the file)."""
    _check_split(split, allow_val)
    jobs_path = Path(jobs_path).resolve()
    jobs_bytes = jobs_path.read_bytes()
    doc = json.loads(jobs_bytes)
    if doc.get("schema") != ev.JOBS_SCHEMA:
        raise ValueError(f"{jobs_path}: schema is {doc.get('schema')!r}, expected {ev.JOBS_SCHEMA!r}")
    splits = episode_splits(jobs_path.parent)
    missing = [j["episode_id"] for j in doc["jobs"] if j["episode_id"] not in splits]
    if missing:
        raise ValueError(f"{len(missing)} job(s) in {jobs_path} have no truth_detail record, so their split "
                         f"is unknown (e.g. {missing[:3]})")
    jobs = [j for j in doc["jobs"] if splits[j["episode_id"]] == split]
    if not jobs:
        raise ValueError(f"{jobs_path} has no {split} jobs")
    per_split = Counter(splits[j["episode_id"]] for j in doc["jobs"])
    return doc, jobs_bytes, jobs, {sp: per_split.get(sp, 0) for sp in SPLITS}


def run(jobs_path: str | Path, out_dir: str | Path, encoder: ev.PatchEncoder, detector: ev.Detector,
        p: ev.Params = ev.Params(), force: bool = False, *, split: str,
        allow_val: bool = False) -> dict[str, Any]:
    """run_jobs with identity_align on one split's jobs; writes evidence.jsonl, diagnostics.jsonl,
    evidence_run.json. val is refused unless allow_val."""
    jobs_path = Path(jobs_path).resolve()
    doc, jobs_bytes, jobs, per_split = select_jobs(jobs_path, split, allow_val)
    out = Path(out_dir).resolve()
    if (out / "evidence.jsonl").exists() and not force:
        raise FileExistsError(f"{out}/evidence.jsonl exists; use force to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    ep_dir = jobs_path.parent
    manifest_path = ep_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    store, diags = ev.run_jobs(jobs, encoder, detector, p, align=ev.identity_align)
    store.to_jsonl(str(out / "evidence.jsonl"))
    with open(out / "diagnostics.jsonl", "w") as f:
        for d in diags:
            f.write(json.dumps(d) + "\n")
    meta = {"schema": RUN_SCHEMA, "jobs": str(jobs_path), "split": split, "n_jobs": len(jobs),
            "n_jobs_in_file": len(doc["jobs"]), "n_jobs_by_split": per_split,
            "split_source": "truth_detail.jsonl (checked against MANIFEST.json arrangements)",
            "jobs_source": doc.get("source"), "jobs_sha256": hashlib.sha256(jobs_bytes).hexdigest(),
            "truth_sha256": _sha256(ep_dir / "truth.json"),
            "truth_detail_sha256": _sha256(ep_dir / "truth_detail.jsonl"),
            "manifest_sha256": _sha256(manifest_path),
            "loader_version": manifest.get("loader_version"), "loader_sha256": manifest.get("loader_sha256"),
            "evidence_version": ev.EVIDENCE_VERSION,
            "evidence_sha256": hashlib.sha256(Path(ev.__file__).read_bytes()).hexdigest(),
            "align": "identity_align", "params": dataclasses.asdict(p),
            "encoder": getattr(encoder, "id", type(encoder).__name__),
            "detector": getattr(detector, "id", type(detector).__name__)}
    (out / "evidence_run.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="public_data.run_evidence", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True, help="jobs.json written by public_data.ocid")
    ap.add_argument("--split", required=True, choices=SPLITS,
                    help="run only this split's jobs (split per episode from truth_detail.jsonl)")
    ap.add_argument("--allow-val", action="store_true", help="needed with --split val (val is kept back)")
    ap.add_argument("--out-dir", default=None, help="default: <jobs dir>/evidence/<EVIDENCE_VERSION>/<split>/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="overwrite an existing evidence.jsonl")
    args = ap.parse_args(argv)
    try:                         # every refusal comes before any model loads
        _, _, jobs, _ = select_jobs(args.jobs, args.split, args.allow_val)
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 2
    out = Path(args.out_dir) if args.out_dir else default_out_dir(args.jobs, args.split)
    if (out / "evidence.jsonl").exists() and not args.force:
        print(f"{out}/evidence.jsonl exists; pass --force to overwrite", file=sys.stderr)
        return 2
    print(f"{args.split}: {len(jobs)} jobs -> {out}", file=sys.stderr)
    meta = run(args.jobs, out, ev.DinoV2Patches(args.device), ev.YoloWorld(args.device), force=args.force,
               split=args.split, allow_val=args.allow_val)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
