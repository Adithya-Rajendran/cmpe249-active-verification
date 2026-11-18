"""Merge rendered dev sets of ONE environment into one set for verify.select (e.g. the tabletop's first 11 arrangements
and the remaining 49 into the 60-arrangement tabletop dev set of the final pooled calibration).

verify.select pools environments, one set each; the tabletop dev arrangements were rendered as two sets. This writes
one set directory from them, checking before anything is written (or created) that:
  - every manifest has the same schema, simulation flag and environment, and the arrangement ids are disjoint;
  - the positions files are byte-identical (one geometry);
  - the view-label files have the same schema, simulation flag and rules, and label exactly their manifest's episodes;
  - each evidence file (EvidenceStore format, real) and each side-descriptor file (perception/side_descriptor.py rows)
    holds every manifest episode at every stop and nothing else;
  - each evidence file agrees with its diagnostics file (the same views and scores), and every diagnostics row of
    every set carries the same evidence_version (one evidence build);
and records every source file's sha256 in merge_provenance.json. Contents are copied, not changed: evidence rows,
descriptor rows and label records are concatenated in set order. Nothing is calibrated or scored here.

    uv run --no-sync python tools/merge_dev_sets.py --set ROOT EVIDENCE DIAG SIDE [--set ...] --out DIR
ROOT holds manifest.json, view_labels.json and one positions_*.json; EVIDENCE, its DIAG(nostics) and SIDE are that set's
files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def positions_of(root: Path) -> Path:
    found = sorted(root.glob("positions_*.json"))
    if len(found) != 1:
        raise ValueError(f"{root}: expected one positions_*.json, found {[p.name for p in found]}")
    return found[0]


def merge(sets: list[tuple[Path, Path, Path, Path]], out: Path) -> dict:
    mans = [json.loads((r / "manifest.json").read_text()) for r, *_ in sets]
    labs = [json.loads((r / "view_labels.json").read_text()) for r, *_ in sets]
    if len({(m["schema"], m.get("simulation")) for m in mans}) != 1:
        raise ValueError("the manifests differ in schema or simulation flag")
    envs = {a.get("environment", "shelf") for m in mans for a in m["arrangements"]}
    if len(envs) != 1:
        raise ValueError(f"one environment only, got {sorted(envs)}")
    ids = [a["arrangement_id"] for m in mans for a in m["arrangements"]]
    if len(set(ids)) != len(ids):
        raise ValueError("an arrangement id is in two sets")
    pos = [positions_of(r) for r, *_ in sets]
    if len({sha(p) for p in pos}) != 1 or len({p.name for p in pos}) != 1:
        raise ValueError("the sets' positions files differ")
    if len({(l["schema"], l["simulation"], json.dumps(l["rules"], sort_keys=True)) for l in labs}) != 1:
        raise ValueError("the view-label files differ in schema, simulation flag or rules")
    n_stops = len(json.loads(pos[0].read_text())["stops"])
    ev_rows, side_rows, versions = [], [], set()
    for (root, ev, diag, side), m, l in zip(sets, mans, labs):
        eps = {f"{a['arrangement_id']}-{c}-{li}" for a in m["arrangements"] for c in ("visible", "occluded", "removed")
               for li in a["lightings"]}
        if set(l["episodes"]) != eps:
            raise ValueError(f"{root}: view_labels.json does not label exactly the manifest's episodes")
        want = {(e, s) for e in eps for s in range(n_stops)}
        for name, path, check in (("evidence", ev, lambda d: d.get("synthetic") is False),
                                  ("side", side, lambda d: d.get("schema") == "cmpe249.side-descriptor.v0")):
            rs = rows(path)
            keys = [(d["episode_id"], int(d["stop"])) for d in rs]
            if set(keys) != want or len(keys) != len(want):
                raise ValueError(f"{path}: {name} rows are not exactly the manifest's episodes at every stop")
            if not all(check(d) for d in rs):
                raise ValueError(f"{path}: not {name} rows of the expected kind")
            (ev_rows if name == "evidence" else side_rows).extend(rs)
        scores = {(d["episode_id"], int(d["stop"])): (d["match_score"], d["quality"], d["obstruction"])
                  for d in rows(ev)}
        drows = rows(diag)
        if {(d["episode_id"], int(d["stop"])): (d["match_score"], d["quality"], d["obstruction"])
                for d in drows} != scores or len(drows) != len(scores):
            raise ValueError(f"{diag} is not the diagnostics of {ev} (other views or scores)")
        versions |= {d.get("evidence_version") for d in drows}
    if len(versions) != 1 or None in versions:
        raise ValueError(f"the sets come from different evidence builds: evidence_version {sorted(map(str, versions))}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"{out} is not an empty directory")
    out.mkdir(parents=True, exist_ok=True)
    man = {**mans[0], "arrangements": [a for m in mans for a in m["arrangements"]]}
    lab = {**labs[0], "episodes": {k: v for l in labs for k, v in l["episodes"].items()}}
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    (out / "view_labels.json").write_text(json.dumps(lab, indent=1) + "\n")
    (out / "evidence.jsonl").write_text("".join(json.dumps(d) + "\n" for d in ev_rows))
    (out / "side.jsonl").write_text("".join(json.dumps(d) + "\n" for d in side_rows))
    (out / pos[0].name).write_bytes(pos[0].read_bytes())
    prov = {"note": "merged dev sets of one environment (tools/merge_dev_sets.py); contents copied unchanged",
            "environment": envs.pop(), "n_arrangements": len(ids), "evidence_version": versions.pop(),
            "sources": [{"root": str(r), "arrangements": len(m["arrangements"]),
                         "files": {str(p): sha(p) for p in (r / "manifest.json", r / "view_labels.json",
                                                             positions_of(r), ev, diag, side)}}
                        for (r, ev, diag, side), m in zip(sets, mans)],
            "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != "merge_provenance.json"}}
    (out / "merge_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    return prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].replace("\n", " "))
    ap.add_argument("--set", nargs=4, action="append", required=True, metavar=("ROOT", "EVIDENCE", "DIAG", "SIDE"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.set) < 2:
        print("merging needs at least two sets", file=sys.stderr)
        return 2
    try:
        prov = merge([tuple(Path(x) for x in s) for s in a.set], Path(a.out))
    except KeyError as e:
        print(f"refused: a source file lacks the field {e}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(f"merged {prov['n_arrangements']} {prov['environment']} arrangements into {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
