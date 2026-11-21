"""Assemble the portable paper supplement from its selected publication files."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

MATERIAL = (".typ", ".bib", ".md", ".py", ".csv", ".gz", ".json", ".jsonl", ".png", ".txt")
SKIP = {"__pycache__", ".git", ".venv"}
FOLLOWUPS = {
    "level-shelf": "level_shelf",
    "corridor-transfer": "corridor_transfer",
    "workshop-clearance": "workshop_realism",
}


def digest(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
        return h.hexdigest()


def followup_files(paper):
    for name, design in FOLLOWUPS.items():
        prefix = Path("followups") / name
        exports = paper.parent / "notes" / "figures" / f"2026-11-18-{name}"
        protocol = paper.parent / "development" / design
        sources = {
            "pairs.csv": exports,
            "decisions.csv": exports,
            "summary.json": exports,
            "design-release.json": exports,
            "independent-review.json": exports,
            "PROTOCOL.md": protocol,
            "declaration.json": protocol,
        }
        if name == "workshop-clearance":
            sources["DOOR_AMENDMENT.md"] = protocol
            sources["declaration-door.json"] = protocol
        for filename, source in sources.items():
            relative = prefix / filename
            packaged = paper / relative
            yield relative.as_posix(), packaged if packaged.is_file() else source / filename


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--software", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    targets = {}
    for f in here.iterdir():
        if f.name == "SHA256SUMS.txt":
            continue
        if f.is_file() and (
            f.suffix in MATERIAL
            or f.name == "Rajendran-counterfactual-scene-memory.pdf"
        ):
            targets[f.name] = f
    for folder in ("figures", "evidence"):
        for f in sorted((here / folder).rglob("*")):
            if (
                f.is_file()
                and f.suffix in MATERIAL
                and not set(f.relative_to(here).parts) & SKIP
            ):
                targets[f.relative_to(here).as_posix()] = f
    # A standalone supplement keeps the trained verifier's portable archive beside its tables.
    still_archive = Path("evidence/trained-still-comparison.zip")
    source = here / still_archive
    if not source.is_file():
        source = here.parent / "notes/figures/2026-11-20-still-verifier" / still_archive.name
    targets[still_archive.as_posix()] = source
    targets.update(followup_files(here))
    for f in sorted(a.software.rglob("*")):
        if f.is_file() and not set(f.relative_to(a.software).parts) & SKIP:
            targets["software/" + f.relative_to(a.software).as_posix()] = f
    if (
        not {
            "evidence/queries.csv",
            "evidence/pairs.csv",
            "evidence/calibration.json",
            "evidence/summary.json",
            "Rajendran-counterfactual-scene-memory.pdf",
            "software/README.md",
        }
        <= targets.keys()
    ):
        raise ValueError("Missing required paper product")
    listing = {name: digest(f) for name, f in sorted(targets.items())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        a.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for name, f in sorted(targets.items()):
            z.write(f, name)
        z.writestr("SHA256SUMS.txt", "".join(f"{h}  {n}\n" for n, h in listing.items()))
    with zipfile.ZipFile(a.out) as z:
        if z.testzip() is not None:
            raise ValueError("Damaged supplement")
    print(
        json.dumps(
            {
                "files": len(listing),
                "sha256": digest(a.out),
                "bytes": a.out.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
