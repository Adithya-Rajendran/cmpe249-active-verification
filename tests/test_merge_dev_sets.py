"""tools/merge_dev_sets.py: two hand-built tabletop sets merge into one set verify.select reads as the union of both;
overlapping arrangements, another geometry, a mixed environment, other label rules, incomplete or synthetic evidence,
diagnostics that disagree with the evidence, two evidence builds and a non-empty output are refused before anything is
written."""
from __future__ import annotations

import contextlib
import io
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


def write_diag(d: Path, version: str = "0.3.1-dev") -> Path:
    """Diagnostics rows that agree with the set's evidence file (the evidence step writes both from one pass)."""
    rows = [json.loads(x) for x in (d / "evidence.jsonl").read_text().splitlines() if x.strip()]
    p = d / "evidence.diag.jsonl"
    p.write_text("".join(json.dumps({**r, "evidence_version": version}) + "\n" for r in rows))
    return p


def make(d: Path, slots, **kw) -> tuple[Path, Path, Path, Path]:
    write_tabletop_set(d, slots, **kw)
    (d / "positions.json").rename(d / "positions_tabletop_A.json")
    return d, d / "evidence.jsonl", write_diag(d), write_side(d, slots)


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
            doc = json.loads((o / "merge_provenance.json").read_text())
            self.assertEqual((doc["environment"], doc["evidence_version"]), ("tabletop", "0.3.1-dev"))
            self.assertEqual([s["arrangements"] for s in doc["sources"]], [len(A), len(B)])
            self.assertEqual(doc["outputs"]["evidence.jsonl"], mds.sha(o / "evidence.jsonl"))
            self.assertTrue(all(len(s["files"]) == 6 for s in doc["sources"]))

    def refused(self, sets, t: Path, name: str) -> None:
        with self.assertRaises(ValueError):
            mds.merge(sets, t / name)
        self.assertFalse((t / name).exists(), name)

    def test_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            a = make(t / "a", A)
            self.refused([a, make(t / "a2", A)], t, "o_same")                      # the same arrangements twice
            geo = make(t / "geo", B)
            pos = json.loads(TAB_POS.read_text())
            pos["radius_cm"] = pos["radius_cm"] + 1
            (t / "geo" / "positions_tabletop_A.json").write_text(json.dumps(pos))
            self.refused([a, geo], t, "o_geo")                                     # another geometry
            shelf = make(t / "shelf", B, environment=None)
            self.refused([a, shelf], t, "o_env")                                   # a shelf set among tabletop ones
            rules = make(t / "rules", B)
            lab = json.loads((t / "rules" / "view_labels.json").read_text())
            lab["rules"] = {"present": "something else"}
            (t / "rules" / "view_labels.json").write_text(json.dumps(lab))
            self.refused([a, rules], t, "o_rules")                                 # other label rules
            short = make(t / "short", B)
            ev = (t / "short" / "evidence.jsonl").read_text().splitlines()
            (t / "short" / "evidence.jsonl").write_text("\n".join(ev[:-1]) + "\n")
            self.refused([a, (t / "short", t / "short" / "evidence.jsonl", write_diag(t / "short"), short[3])], t,
                         "o_short")                                                # an evidence row missing
            syn = make(t / "syn", B)
            rows = [json.loads(x) for x in (t / "syn" / "evidence.jsonl").read_text().splitlines()]
            (t / "syn" / "evidence.jsonl").write_text("".join(json.dumps({**r, "synthetic": True}) + "\n"
                                                              for r in rows))
            self.refused([a, (t / "syn", t / "syn" / "evidence.jsonl", write_diag(t / "syn"), syn[3])], t, "o_syn")
            other = make(t / "other", B)
            self.refused([a, (*other[:2], write_diag(t / "other", "0.2.1"), other[3])], t, "o_version")
            bad = make(t / "bad", B)
            d = [json.loads(x) for x in bad[2].read_text().splitlines()]
            d[0]["match_score"] = 0.123456
            bad[2].write_text("".join(json.dumps(x) + "\n" for x in d))
            self.refused([a, bad], t, "o_diag")                                    # diagnostics of other evidence
            (t / "full").mkdir()
            (t / "full" / "x.txt").write_text("x")
            with self.assertRaises(ValueError):
                mds.merge([a, make(t / "b", B)], t / "full")                      # a non-empty output

    def test_cli(self) -> None:
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            t = Path(t)
            sa, sb = make(t / "a", A), make(t / "b", B)
            self.assertEqual(mds.main(["--set", *map(str, sa), "--set", *map(str, sb), "--out", str(t / "o")]), 0)
            self.assertEqual(mds.main(["--set", *map(str, sa), "--out", str(t / "o3")]), 2)
            self.assertIn("at least two sets", err.getvalue())


if __name__ == "__main__":
    unittest.main()
