"""The test-seed gate (sim/freeze.py and its three users): test seeds open only against a committed, unchanged
FREEZE.md whose listed files all match; generation, episode building and rendering each refuse a test arrangement
without it. Dev behaviour is pinned by the goldens, not here.

RULE FOR THESE TESTS: never generate an arrangement from the study's test blocks (shelf 100000+, tabletop 110000+ and
their reserves) before the freeze. Generation is exercised only on a synthetic block (SYNTHETIC_START) of a modified
in-memory config. (On 2026-11-05 a first version of this file generated the shelf block's slot 0 in a temporary
directory under a mocked freeze; nothing of it was viewed, and sim.testsplit.unlocked now re-checks the freeze file.)"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sim import arrange as A
from sim import episodes as E
from sim import freeze as F
from sim import isaac_batch as B
from sim import render_test as RT
from sim import testsplit as TS

REPO = Path(__file__).resolve().parents[1]


def record(files: dict[str, str], **kw) -> str:
    doc = {"schema": F.FREEZE_SCHEMA, "freeze_id": "freeze-test", "commit": "abc", "files": files, **kw}
    return "# FREEZE\n\ntext\n\n```json\n" + json.dumps(doc, indent=1) + "\n```\n"


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


class ParseTest(unittest.TestCase):
    def test_exactly_one_valid_record(self) -> None:
        self.assertEqual(F.parse_record(record({"a.txt": "0" * 64}))["freeze_id"], "freeze-test")
        for text in ("no block", record({"a.txt": "0" * 64}) * 2, record({}), record({"a.txt": "short"}),
                     "```json\n{\"schema\": \"other\"}\n```"):
            with self.assertRaises(F.FreezeError):
                F.parse_record(text)


class RequireTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "t@example.com")
        git(self.repo, "config", "user.name", "t")
        (self.repo / "a.txt").write_text("frozen\n")
        self.freeze = self.repo / "FREEZE.md"
        self.freeze.write_text(record({"a.txt": F.file_sha256(self.repo / "a.txt")}))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_committed_and_unchanged(self) -> None:
        with self.assertRaises(F.FreezeError):                     # untracked
            F.require_freeze(self.freeze, repo=self.repo)
        git(self.repo, "add", "a.txt", "FREEZE.md")
        git(self.repo, "commit", "-q", "-m", "freeze")
        got = F.require_freeze(self.freeze, needed=("a.txt",), repo=self.repo)
        self.assertEqual(got["sha256"], F.file_sha256(self.freeze))
        self.assertEqual(got["record"]["freeze_id"], "freeze-test")
        with self.assertRaises(F.FreezeError):                     # a needed file the freeze does not list
            F.require_freeze(self.freeze, needed=("b.txt",), repo=self.repo)
        (self.repo / "a.txt").write_text("changed\n")
        with self.assertRaises(F.FreezeError):                     # a frozen file changed after the freeze
            F.require_freeze(self.freeze, repo=self.repo)
        (self.repo / "a.txt").write_text("frozen\n")
        self.freeze.write_text(self.freeze.read_text() + "\nedited\n")
        with self.assertRaises(F.FreezeError):                     # FREEZE.md edited after its commit
            F.require_freeze(self.freeze, repo=self.repo)

    def test_matching_copy(self) -> None:
        sha = F.file_sha256(self.freeze)
        self.assertEqual(F.matching_freeze(self.freeze, sha), sha)
        with self.assertRaises(F.FreezeError):
            F.matching_freeze(self.freeze, "0" * 64)


FAKE = {"record": {"freeze_id": "freeze-test"}, "sha256": "f" * 64, "path": "docs/FREEZE.md"}
SYNTHETIC_START = 900000   # a synthetic seed block for tests: far from the study's test blocks and their reserves


def synthetic_ctx() -> tuple[A.Context, list[int]]:
    """The shelf config with its test block moved to SYNTHETIC_START (in memory): never the study's test seeds."""
    cfg = json.loads((REPO / "configs/sim/arrange_shelf_v3.json").read_text())
    blk = cfg["slots"]["test"]
    blk["primary_seed_start"] = SYNTHETIC_START
    blk["reserve"] = {f: [v + SYNTHETIC_START - 99000 for v in r] for f, r in blk["reserve"].items()}
    ctx = A.load_context(cfg)
    seeds = TS.test_block_seeds(ctx)
    assert min(seeds) >= SYNTHETIC_START
    return ctx, seeds


def real_freeze(t: Path) -> dict:
    fz = Path(t) / "FREEZE.md"
    fz.write_text(record({"x": "0" * 64}))
    return {**FAKE, "path": str(fz), "sha256": F.file_sha256(fz)}


class GeneratorTest(unittest.TestCase):
    def run_main(self, *args: str) -> tuple[int, str]:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            rc = TS.main(list(args))
        return rc, err.getvalue()

    def test_main_needs_the_committed_freeze(self) -> None:
        seen = {}

        def refuse(path, needed=(), repo=None):
            seen["needed"] = needed
            raise F.FreezeError("not committed")
        with tempfile.TemporaryDirectory() as t, mock.patch.object(F, "require_freeze", side_effect=refuse):
            rc, err = self.run_main("--config", "configs/sim/arrange_shelf_v3.json", "--freeze", "x.md",
                                    "--slots", "0", "--out", str(Path(t) / "o"))
            self.assertFalse((Path(t) / "o").exists())
        self.assertEqual(rc, 2)
        self.assertIn("not committed", err)
        self.assertIn("configs/sim/arrange_shelf_v3.json", seen["needed"])
        self.assertTrue(set(TS.NEEDED) <= set(seen["needed"]))

    def test_a_test_arrangement_records_its_freeze(self) -> None:
        ctx, seeds = synthetic_ctx()
        with tempfile.TemporaryDirectory() as t:
            real = real_freeze(t)
            with TS.unlocked(real, seeds):
                arrs, _ = A.generate_set(ctx, "test", [0])
        arr = arrs[0]
        self.assertEqual(arr["split"], "test")
        self.assertGreaterEqual(arr["seed"], SYNTHETIC_START)
        self.assertEqual(arr["arrangement_id"], f"ST{arr['seed']:06d}")
        self.assertEqual(arr["freeze"]["sha256"], real["sha256"])
        self.assertEqual(arr["freeze"]["testsplit_sha256"], F.file_sha256(REPO / "sim/testsplit.py"))
        self.assertEqual(A.check_dev_seed.__module__, A.__name__)       # every wrapper restored
        with self.assertRaises(A.SeedError):                             # and without the gate, no test arrangement
            A.generate_set(ctx, "test", [0])

    def test_a_stand_in_freeze_cannot_unlock(self) -> None:
        _, seeds = synthetic_ctx()
        with self.assertRaises(A.SeedError):                             # FAKE names a FREEZE.md that does not exist
            with TS.unlocked(FAKE, seeds):
                pass
        self.assertEqual(A.check_dev_seed.__module__, A.__name__)

    def test_the_unlock_is_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            real = real_freeze(t)
            with self.assertRaises(A.SeedError):
                A.check_dev_seed(SYNTHETIC_START)
            with TS.unlocked(real, [SYNTHETIC_START]):
                self.assertEqual(A.check_dev_seed(SYNTHETIC_START), SYNTHETIC_START)
                self.assertEqual(A.arrangement_id(SYNTHETIC_START, "TD"), f"TT{SYNTHETIC_START:06d}")
                self.assertEqual(A.arrangement_id(5, "TD"), "TD0005")     # dev ids unchanged inside the block
                with self.assertRaises(A.SeedError):
                    A.check_dev_seed(SYNTHETIC_START + 1)                  # not in the unlocked block
                with self.assertRaises(RuntimeError):
                    with TS.unlocked(real, [SYNTHETIC_START]):
                        pass
            with self.assertRaises(A.SeedError):
                A.check_dev_seed(SYNTHETIC_START)
            with self.assertRaises(A.SeedError):
                with TS.unlocked(real, [5]):                               # a dev seed is never "unlocked"
                    pass


def test_arr(**kw) -> dict:
    return {"schema": B.ARRANGEMENT_SCHEMA, "arrangement_id": "ST900000", "seed": SYNTHETIC_START, "split": "test",
            "kind": "random", "freeze": {"sha256": "f" * 64, "freeze_id": "freeze-test"}, **kw}


class ConsumerTest(unittest.TestCase):
    def test_episodes_check(self) -> None:
        E.check_split(test_arr(), "f" * 64)
        for arr, sha in ((test_arr(), None), (test_arr(), "0" * 64), (test_arr(seed=5), "f" * 64),
                         (test_arr(kind="design"), "f" * 64)):
            with self.assertRaises(A.SeedError):
                E.check_split(arr, sha)
        E.check_split({"split": "dev", "seed": 5, "arrangement_id": "SD0005"})   # dev as before

    def test_renderer_gate(self) -> None:
        check = RT.gate("f" * 64)
        check(test_arr())
        for arr in (test_arr(freeze={"sha256": "0" * 64}), test_arr(seed=99999), test_arr(kind="design")):
            with self.assertRaises(B.RefusedArrangement):
                check(arr)
        with self.assertRaises(B.RefusedArrangement):                     # isaac_batch's own check: still refuses
            B.check_arrangement(test_arr())
        with self.assertRaises(B.RefusedArrangement):                     # a dev-split test seed: the original check
            check({**test_arr(), "split": "dev"})
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(RT.main(["--arrangements", "x.json"]), 2)   # no --freeze



class MakeFreezeTest(unittest.TestCase):
    def test_the_frozen_list_exists_and_uses_the_gates_schema(self) -> None:
        import sys
        sys.path.insert(0, str(REPO / "tools"))
        import make_freeze as MF
        self.assertEqual(MF.FREEZE_SCHEMA, F.FREEZE_SCHEMA)
        self.assertEqual(len(set(MF.FROZEN)), len(MF.FROZEN))
        self.assertEqual([f for f in MF.FROZEN if not (REPO / f).is_file()], [])
        for needed in (*TS.NEEDED, "sim/episodes.py", "sim/isaac_batch.py", "sim/render_test.py"):
            self.assertIn(needed, MF.FROZEN)


if __name__ == "__main__":
    unittest.main()
