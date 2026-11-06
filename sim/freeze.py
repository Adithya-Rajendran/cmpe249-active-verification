"""The freeze gate: test seeds (>= 100000) open only against the committed FREEZE.md (docs/SIM_PREREGISTRATION.md §12).

FREEZE.md carries exactly one fenced ```json block whose "schema" is FREEZE_SCHEMA:

    {"schema": "cmpe249.freeze.v1", "freeze_id": "...", "commit": "<the frozen commit>",
     "files": {"<repo-relative path>": "<sha256>", ...}}

`require_freeze` (a checkout with git, where arrangements are generated and episodes built) refuses unless the file is
tracked and committed unchanged at HEAD, its record parses, the record lists the files the caller needs frozen, and
every listed file still has its listed sha256. It returns the record and FREEZE.md's own sha256, which every test arrangement
then records. `matching_freeze` (the GB10 render directory, no git) only checks that a given copy of FREEZE.md has the
sha256 an arrangement recorded. No dev path calls either.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCHEMA = "cmpe249.freeze.v1"
_BLOCK = re.compile(r"```json\s*\n(.*?)\n```", re.S)


class FreezeError(RuntimeError):
    """No valid freeze: test seeds stay closed."""


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_record(text: str) -> dict[str, Any]:
    """The one fenced json block with "schema": FREEZE_SCHEMA."""
    found = []
    for block in _BLOCK.findall(text):
        try:
            doc = json.loads(block)
        except ValueError:
            continue
        if isinstance(doc, dict) and doc.get("schema") == FREEZE_SCHEMA:
            found.append(doc)
    if len(found) != 1:
        raise FreezeError(f"FREEZE.md needs exactly one ```json block with schema {FREEZE_SCHEMA!r}; "
                          f"found {len(found)}")
    rec = found[0]
    files = rec.get("files")
    if not (isinstance(rec.get("freeze_id"), str) and rec["freeze_id"] and isinstance(rec.get("commit"), str)
            and isinstance(files, dict) and files
            and all(isinstance(k, str) and isinstance(v, str) and len(v) == 64 for k, v in files.items())):
        raise FreezeError("the freeze record needs a freeze_id, a commit and a non-empty {path: sha256} files map")
    return rec


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def require_freeze(path: str | Path, needed: Iterable[str] = (), repo: str | Path = REPO_ROOT) -> dict[str, Any]:
    """{"record", "sha256", "path"} of a committed, unchanged FREEZE.md whose listed files all match; FreezeError
    otherwise. needed: repo-relative paths the caller requires to be among the frozen files."""
    repo = Path(repo).resolve()
    p = Path(path).resolve()
    try:
        rel = p.relative_to(repo).as_posix()
    except ValueError as e:
        raise FreezeError(f"{p} is not inside the repository {repo}") from e
    if not p.is_file():
        raise FreezeError(f"{p}: no such file")
    if _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        raise FreezeError(f"{rel} is not tracked by git: a freeze must be committed")
    if _git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise FreezeError(f"{rel} differs from HEAD: commit the freeze first")
    text = p.read_text()
    rec = parse_record(text)
    missing = sorted(set(needed) - set(rec["files"]))
    if missing:
        raise FreezeError(f"the freeze does not list {missing}")
    changed = []
    for f, want in sorted(rec["files"].items()):
        fp = repo / f
        if not fp.is_file() or file_sha256(fp) != want:
            changed.append(f)
    if changed:
        raise FreezeError(f"frozen files changed or missing since the freeze: {changed[:5]}")
    return {"record": rec, "sha256": hashlib.sha256(text.encode()).hexdigest(), "path": rel}


def matching_freeze(path: str | Path, want_sha256: str) -> str:
    """The sha256 of the given FREEZE.md copy if it equals want_sha256 (what an arrangement recorded); FreezeError
    otherwise. For a host without the repository (the GB10 render directory)."""
    p = Path(path)
    if not p.is_file():
        raise FreezeError(f"{p}: no such file")
    got = file_sha256(p)
    if got != want_sha256:
        raise FreezeError(f"{p}: sha256 {got} is not the arrangement's freeze {want_sha256}")
    parse_record(p.read_text())
    return got
