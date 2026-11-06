"""tools/merge_dev_sets.py: two hand-built tabletop sets merge into one set verify.select reads as the union of both;
overlapping arrangements, another geometry and a mixed environment are refused before anything is written."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_ranker_side import write_side
from tests.test_select_pooled import TAB_POS, write_tabletop_set
from verify import select as sel
from verify.schema import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))
import merge_dev_sets as mds  # noqa: E402

A = (("TD5000", "S8", None), ("TD5001", "M16", "L"), ("TD5002", "M16", "R"))
B = (("TD5003", "IND28", None), ("TD5004", "white_box", None), ("TD5005", "ycb_box", None), ("TD5006", "S8", None))


def make(d: Path, slots) -> tuple[Path, Path, Path]:
    write_tabletop_set(d, slots)
    (d / "positions.json").rename(d / "positions_tabletop_A.json")
    return d, d / "evidence.jsonl", write_side(d, slots)


class MergeTest(unittest.TestCase):
    def test_union_of_two_sets(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            prov = mds.merge([make(t / "a", A), make(t / "b", B)], t / "out")
            self.assertEqual(prov["n_arrangements"], 7)
            o = t / "out"
            dev = sel.sim_dev_data(o / "manifest.json", o / "evidence.jsonl", o / "view_labels.json",
                                   o / "positions_tabletop_A.json", o / "side.jsonl")
            self.assertEqual(dev.arrangements, tuple(sorted(a for a, _, _ in A + B)))
            self.assertEqual(set(dev.side.episode_ids), {s.episode_id for s in dev.specs})
            self.assertEqual(json.loads((o / "merge_provenance.json").read_text())["environment"], "tabletop")

    def test_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            a = make(t / "a", A)
            with self.assertRaises(ValueError):                      # the same arrangements twice
                mds.merge([a, make(t / "a2", A)], t / "o1")
            other = make(t / "c", B)
            pos = json.loads(TAB_POS.read_text())
            pos["radius_cm"] = pos["radius_cm"] + 1
            (t / "c" / "positions_tabletop_A.json").write_text(json.dumps(pos))
            with self.assertRaises(ValueError):                      # another geometry
                mds.merge([a, other], t / "o2")
            self.assertFalse((t / "o1").exists() or (t / "o2").exists())

    def test_cli(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            sa, sb = make(t / "a", A), make(t / "b", B)
            self.assertEqual(mds.main(["--set", *map(str, sa), "--set", *map(str, sb), "--out", str(t / "o")]), 0)
            self.assertEqual(mds.main(["--set", *map(str, sa), "--out", str(t / "o3")]), 2)


if __name__ == "__main__":
    unittest.main()
