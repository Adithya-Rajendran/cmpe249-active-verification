"""Generate the frozen test arrangements: sim/arrange.py, byte-identical, run under the freeze gate (sim/freeze.py).

sim/arrange.py stays exactly as it generated the development sets: an arrangement's sha256 covers the generator's own
sha256, so editing that file would break the reproduction of every pinned development arrangement. This module opens
the test seeds around it instead, and only inside `unlocked`, which `main` enters after require_freeze has accepted the
committed FREEZE.md. Inside it, sim.arrange's own module functions are wrapped:
    check_dev_seed  also accepts the seeds of the frozen test block (primary seeds and reserves), nothing else;
    arrangement_id  gives a test seed the id <first letter of the dev prefix>T<6-digit seed> (ST100000, TT110000);
    _finish         marks a test-block random arrangement split "test" and adds its freeze record
                    {sha256 of FREEZE.md, freeze_id, sha256 of this module} BEFORE the arrangement is hashed.
Everything else (placement, retries, reserves, targets, revisit draws, hashing) is sim.arrange's unchanged code, and
every wrapper is restored when the block ends. Usage, only after the freeze (repo root):

    uv run --no-sync python -m sim.testsplit --config configs/sim/arrange_tabletop_v1.json --freeze docs/FREEZE.md \\
        --slots all --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sim import arrange as A
from sim import freeze as F

THIS_FILE = Path(__file__).resolve()
NEEDED = ("sim/arrange.py", "sim/rig.py", "sim/tabletop.py", "sim/freeze.py", "sim/testsplit.py")
FREEZE_REL = "docs/FREEZE.md"
# Seeds from here up form the synthetic blocks of the unit tests (in-memory configs). Every lower test-range seed
# (>= A.TEST_SEED_MIN), including both study blocks and their reserves, needs the committed docs/FREEZE.md itself.
SYNTHETIC_SEED_MIN = 900000
# Captured at import: a test that patches sim.freeze.require_freeze cannot open the study's test seeds.
_REQUIRE_FREEZE = F.require_freeze


def test_block_seeds(ctx: A.Context) -> list[int]:
    """Every seed the config's test block may use: the slots' primary seeds and every family's reserve list."""
    blk = ctx.config["slots"]["test"]
    primary = [sl["primary_seed"] for sl in A.slot_list(ctx, "test")]
    return primary + [int(v) for r in blk["reserve"].values() for v in r]


@contextlib.contextmanager
def unlocked(freeze: Mapping[str, Any], seeds: Sequence[int]) -> Iterator[None]:
    """Accept these test seeds in sim.arrange while the block runs (freeze: require_freeze's result).
    - A seed below SYNTHETIC_SEED_MIN (a study block or its reserves) opens only if the committed docs/FREEZE.md passes
      the require_freeze captured at import and has the freeze's sha256. A mocked or stand-in record therefore cannot
      open a study seed, not even with a matching file elsewhere.
    - A synthetic test block needs the file the freeze names, with its sha256, on disk.
    The gate guards against accidents, such as a test with a mocked freeze, not against deliberate in-process patching
    of this module or of sim.freeze."""
    if A.check_dev_seed.__module__ != A.__name__:
        raise RuntimeError("test seeds are already unlocked")
    allowed = frozenset(int(s) for s in seeds)
    if not allowed or min(allowed) < A.TEST_SEED_MIN:
        raise A.SeedError(f"a test block holds only seeds >= {A.TEST_SEED_MIN}")
    if min(allowed) < SYNTHETIC_SEED_MIN:
        try:
            committed = _REQUIRE_FREEZE(F.REPO_ROOT / FREEZE_REL, needed=NEEDED)
        except F.FreezeError as e:
            raise A.SeedError(f"study test seeds need the committed {FREEZE_REL}: {e}") from e
        if committed["sha256"] != freeze["sha256"]:
            raise A.SeedError(f"the freeze's sha256 is not the committed {FREEZE_REL}'s; test seeds stay closed")
    else:
        fp = Path(freeze["path"])
        fp = fp if fp.is_absolute() else A.R.REPO_ROOT / fp
        if not fp.is_file() or hashlib.sha256(fp.read_bytes()).hexdigest() != freeze["sha256"]:
            raise A.SeedError(f"{fp}: no FREEZE.md with the freeze's sha256; test seeds stay closed")
    record = {"sha256": freeze["sha256"], "freeze_id": freeze["record"]["freeze_id"],
              "testsplit_sha256": hashlib.sha256(THIS_FILE.read_bytes()).hexdigest()}
    real_check, real_id, real_finish = A.check_dev_seed, A.arrangement_id, A._finish

    def check(seed: Any) -> int:
        if not isinstance(seed, bool) and isinstance(seed, (int, np.integer)) and int(seed) in allowed:
            return int(seed)
        return real_check(seed)

    def arrangement_id(seed: int, prefix: str = "SD") -> str:
        seed = check(seed)
        return f"{prefix[0]}T{seed:06d}" if seed >= A.TEST_SEED_MIN else real_id(seed, prefix)

    def finish(ctx: A.Context, arr: dict, rig: Any, keep_mug: bool, *args: Any, **kw: Any) -> dict:
        if arr.get("kind") == "random" and isinstance(arr.get("seed"), int) and arr["seed"] in allowed:
            arr["split"] = "test"
            arr["freeze"] = dict(record)
        return real_finish(ctx, arr, rig, keep_mug, *args, **kw)

    check.__module__ = arrangement_id.__module__ = finish.__module__ = __name__
    A.check_dev_seed, A.arrangement_id, A._finish = check, arrangement_id, finish
    try:
        yield
    finally:
        A.check_dev_seed, A.arrangement_id, A._finish = real_check, real_id, real_finish


def write_set(out: Path, ctx: A.Context, arrs: Sequence[dict], slots: Sequence[dict], log: Mapping | None) -> None:
    """The files sim.arrange.main writes: DIR/<ARR>.json and DIR/index/{index,slots,generation_log}.json."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "index").mkdir(exist_ok=True)
    rows = []
    for arr in arrs:
        p = out / f"{arr['arrangement_id']}.json"
        p.write_text(json.dumps(arr, indent=1) + "\n")
        rows.append(A._index_row(arr, p))
    prov = {"provenance": ctx.provenance, "unhashed": ctx.unhashed}
    (out / "index" / "index.json").write_text(json.dumps({"schema": A.INDEX_SCHEMA, **prov, "arrangements": rows},
                                                         indent=1) + "\n")
    (out / "index" / "slots.json").write_text(json.dumps({"schema": A.SLOTS_SCHEMA, **prov, "split": "test",
                                                          "retry_max": ctx.config["slots"]["retry_max"],
                                                          "slots": list(slots)}, indent=1) + "\n")
    if log is not None:
        (out / "index" / "generation_log.json").write_text(json.dumps({**log, "unhashed": ctx.unhashed}, indent=1)
                                                           + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.testsplit", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--freeze", required=True, help="the committed docs/FREEZE.md (sim/freeze.py require_freeze)")
    ap.add_argument("--slots", required=True, help="'all' or test slot indices like '0-5,9'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    try:
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise ValueError(f"{out} is not empty")
        cfg_rel = Path(a.config).resolve().relative_to(A.R.REPO_ROOT).as_posix()
        freeze = F.require_freeze(a.freeze, needed=(*NEEDED, cfg_rel))
        ctx = A.load_context(a.config)
        slots = A.slot_list(ctx, "test")
        indices = A.parse_slot_indices(a.slots, slots)
        with unlocked(freeze, test_block_seeds(ctx)):
            arrs, log = A.generate_set(ctx, "test", indices)
    except (A.SeedError, ValueError, RuntimeError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    write_set(out, ctx, arrs, slots, log)
    print(f"wrote {len(arrs)} test arrangements to {out} under freeze {freeze['record']['freeze_id']} "
          f"(FREEZE.md sha256 {freeze['sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
