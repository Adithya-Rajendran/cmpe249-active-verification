"""Render frozen test arrangements with sim/isaac_batch.py, byte-identical, under the freeze gate.

isaac_batch.py refuses every test arrangement (check_arrangement), and it stays exactly the renderer of the development
sets. This wrapper, copied next to it in the GB10 run directory with a copy of the committed FREEZE.md, accepts a test
arrangement only if it is a random arrangement with a test-range seed, split "test", and a freeze record whose sha256
equals the given FREEZE.md copy's (sim/testsplit.py writes that record; sim/freeze.py guards the generating side). Any
other arrangement goes through isaac_batch's own check unchanged, so a development arrangement renders exactly as
before.
Then isaac_batch.main runs with the remaining arguments. Usage (GB10 run dir, Isaac's python, only after the freeze):

    python render_test.py --freeze FREEZE.md --arrangements ARR/*.json --assets-dir ... --out OUT [isaac_batch options]
"""
from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from sim import isaac_batch as B
except ImportError:  # the GB10 run dir, next to a copy of isaac_batch.py
    import isaac_batch as B


def gate(freeze_sha256: str):
    """isaac_batch.check_arrangement, plus test arrangements of this freeze."""
    original = B.check_arrangement

    def check(arr: Mapping, *args, **kw) -> None:
        if arr.get("split") != "test":
            return original(arr, *args, **kw)
        seed = arr.get("seed")
        if arr.get("schema") != B.ARRANGEMENT_SCHEMA or arr.get("kind") != "random" or isinstance(seed, bool) \
                or not isinstance(seed, int) or seed < B.TEST_SEED_MIN:
            raise B.RefusedArrangement(f"{arr.get('arrangement_id')}: not a test-range random arrangement")
        if (arr.get("freeze") or {}).get("sha256") != freeze_sha256:
            raise B.RefusedArrangement(f"{arr.get('arrangement_id')}: its freeze is not the given FREEZE.md")
    return check


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--freeze" not in argv or argv.index("--freeze") + 1 >= len(argv):
        print("render_test.py needs --freeze FREEZE.md (a copy of the committed freeze)", file=sys.stderr)
        return 2
    i = argv.index("--freeze")
    path = Path(argv[i + 1])
    del argv[i:i + 2]
    if not path.is_file():
        print(f"{path}: no such file", file=sys.stderr)
        return 2
    B.check_arrangement = gate(hashlib.sha256(path.read_bytes()).hexdigest())
    return B.main(argv) or 0


if __name__ == "__main__":
    raise SystemExit(main())
