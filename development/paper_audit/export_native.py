"""Create an offline raw-data archive from completed, hash-bound study captures."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def digest(path):
    m = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            m.update(chunk)
    return m.hexdigest()


def pack(source, output):
    included = {}

    def include(path, name, expected=None):
        h = digest(path)
        if expected is not None and h != expected:
            raise ValueError(f"Original completed capture changed: {name}")
        if name in included and included[name] != (path, h):
            raise ValueError(f"Archive path collision: {name}")
        included[name] = (path, h)

    for split, n in (("calibration", 4), ("evaluation", 16)):
        cohort = source / (split + "-cohort")
        m = json.loads((cohort / "manifest.json").read_text())
        if m["study_id"] != "paper-audit-v1" or len(m["arrangements"]) != n:
            raise ValueError("Unexpected layout manifest")
        for file in ("manifest.json", "generation_log.json"):
            include(cohort / file, f"cohorts/{split}/{file}")
        for a in m["arrangements"]:
            for level, info in a["levels"].items():
                include(
                    cohort / info["path"],
                    f"cohorts/{split}/" + info["path"],
                    info["sha256"],
                )
                for engine in ("isaac", "mujoco"):
                    base = source / (engine + "-" + split) / a["base_id"] / level
                    dst = f"{engine}/{split}/{a['base_id']}/{level}"
                    marker = "completion.json" if engine == "isaac" else "complete.json"
                    observed = json.loads((base / marker).read_text())
                    if not observed["outputs"]:
                        raise ValueError("Empty capture receipt")
                    for filename, h in observed["outputs"].items():
                        p = (base / filename).resolve()
                        if base.resolve() not in p.parents:
                            raise ValueError("Invalid capture path")
                        include(p, f"{dst}/{filename}", h)
                    include(base / marker, f"{dst}/{marker}")
    manifest = {name: h for name, (_, h) in sorted(included.items())}
    # Images are already compressed; text geometry and observations compress well.
    with zipfile.ZipFile(output, "x", allowZip64=True) as z:
        for name, (path, _) in sorted(included.items()):
            compression = (
                zipfile.ZIP_STORED if path.suffix == ".png" else zipfile.ZIP_DEFLATED
            )
            z.write(path, name, compress_type=compression)
        z.writestr(
            "SHA256SUMS.txt",
            "".join(h + "  " + n + "\n" for n, h in manifest.items()),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        z.writestr(
            "README.txt",
            "Native simulation captures for paper-audit-v1. The calibration and evaluation directories use disjoint arrangements. Evaluator-only poses and masks must remain hidden during scoring. Each completion file records the original per-file SHA-256 value. Use the paper supplement to check the analysis.\n",
        )
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError("Raw-data archive integrity check failed")
    print(
        json.dumps(
            {
                "files": len(included),
                "bytes": output.stat().st_size,
                "sha256": digest(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture_directory", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    pack(a.capture_directory, a.output)
