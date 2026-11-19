"""Restore the original opaque scorer packet from content-addressed PNGs."""

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path


def restore(archive, out):
    out.mkdir(exist_ok=False, parents=True)
    with zipfile.ZipFile(archive) as z:
        manifest = z.read("cases.json")
        cases = json.loads(manifest)
        seen = {}
        ids = set()
        for r in cases:
            uid = r["id"]
            if not uid.isalnum() or uid in ids:
                raise ValueError("Invalid or repeated opaque case identifier")
            ids.add(uid)
            for role in ("reference", "query"):
                name = r[role]
                if Path(name).name != name or name in (".", ".."):
                    raise ValueError("Invalid packet filename")
                dest = out / name
                if dest.exists():
                    raise ValueError("Duplicate destination")
                h = r[role + "_sha256"]
                if len(h) != 64 or any(ch not in "0123456789abcdef" for ch in h):
                    raise ValueError("Invalid image checksum")
                if h in seen:
                    try:
                        os.link(seen[h], dest)
                    except OSError:
                        shutil.copyfile(seen[h], dest)
                    continue
                with z.open("rgb/" + h + ".png") as src, dest.open("xb") as target:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: src.read(2**20), b""):
                        target.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != h:
                    raise ValueError("Damaged input image")
                seen[h] = dest
        (out / "cases.json").write_bytes(manifest)
    print(
        f"{len(ids)} cases, {len(seen)} distinct image files, inventory SHA-256 {hashlib.sha256(manifest).hexdigest()}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("new_directory", type=Path)
    a = p.parse_args()
    restore(a.archive, a.new_directory)
