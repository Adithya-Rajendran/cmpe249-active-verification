"""Shelf goldens (tabletop wave, step 0): every shelf output must stay what it was at origin/main 51cf0a3.

tests/golden/shelf_goldens.json (written by tests/golden/make_shelf_goldens.py before any tabletop edit) holds canonical
hashes with only the code-hash fields masked (tests/shelf_golden.py). Pinned here:
  - the generator: slot_list dev and test; the 64 shelf_v3 dev arrangements (regenerated: REGEN_SUBSET + the 4 design
    cases by default, all 60 slots with CMPE249_GOLDEN_FULL=1), and each one's arrangement_sha256 recomputed with the
    golden's code hashes restored;
  - the share's files (skipped without the share): dev60 arrangements and index files; shelf_dev_v3's sim/episodes.py
    outputs, rebuilt on a JSON-only scratch copy (the share is never written); sim/check_batch.py per-arrangement results for
    CHECK_SUBSET (the full check_set, ~85 s, with CMPE249_GOLDEN_SLOW=1);
  - the renderer without Isaac: sim/isaac_batch.py main() against tests/fake_isaac.py for SD1000 (annotator call log, PNG
    metadata, render settings, labels) and the pxr call log of the real build_stage for SD1000 and DES0.
"""
import json
import os
import platform
import tempfile
import unittest
from pathlib import Path

from sim import arrange as A
from tests import shelf_golden as G

GOLD = G.load_golden()
HAVE_SHARE = G.DEV60.is_dir() and G.SHELF_DEV_V3.is_dir()
SAME_PYTHON = platform.python_version() == GOLD["made"]["python"]
FULL = os.environ.get("CMPE249_GOLDEN_FULL") == "1"
SLOW = os.environ.get("CMPE249_GOLDEN_SLOW") == "1"


class TestGoldenFile(unittest.TestCase):
    def test_complete(self):
        g = GOLD["generator"]
        self.assertEqual(len(g["arrangements"]), 64)
        self.assertEqual({k for k in g["arrangements"] if k.startswith("DES")}, {"DES0", "DES1", "DES2", "DES3"})
        self.assertEqual((g["slot_list"]["dev"]["n"], g["slot_list"]["test"]["n"]), (60, 100))
        self.assertEqual(set(G.EPISODES_FILES), set(GOLD["rendered_set"]["episodes"]["files"]))
        self.assertEqual(len(GOLD["rendered_set"]["episodes"]["annotations"]), 64)
        self.assertTrue(set(G.CHECK_SUBSET) <= set(GOLD["rendered_set"]["check_batch"]["arrangements"]))

    def test_masks_touch_only_code_hashes(self):
        a = {"provenance": {"generator": {"sha256": "x", "version": "v"}, "rig_py": {"sha256": "y"}, "config": {"sha256": "c"}},
             "arrangement_sha256": "h", "seed": 3}
        c = G.canonical_arrangement(a)
        self.assertEqual(c["provenance"]["config"]["sha256"], "c")
        self.assertEqual((c["provenance"]["generator"]["sha256"], c["provenance"]["rig_py"]["sha256"], c["arrangement_sha256"]),
                         (G.MASK,) * 3)
        self.assertEqual((c["seed"], c["provenance"]["generator"]["version"]), (3, "v"))
        self.assertEqual(G.mask_keys({"a": [{"episodes_py_sha256": 1, "b": 2}]}), {"a": [{"episodes_py_sha256": G.MASK, "b": 2}]})


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = G.generator_context()

    def test_slot_lists(self):
        self.assertEqual(G.slot_list_hashes(self.ctx), GOLD["generator"]["slot_list"])

    @unittest.skipUnless(SAME_PYTHON, "the arrangements record the Python version (unhashed.python); run with the golden's Python")
    def test_regenerated_arrangements_match(self):
        """REGEN_SUBSET (the first slot of each class + 4 others) and the design cases; all 60 with CMPE249_GOLDEN_FULL=1."""
        g = GOLD["generator"]
        regen = G.regenerate(self.ctx, range(60) if FULL else G.REGEN_SUBSET)
        self.assertEqual(len(regen), (60 if FULL else len(G.REGEN_SUBSET)) + 4)
        code = GOLD["code_hashes"]
        for aid, arr in regen.items():
            with self.subTest(arrangement=aid):
                self.assertEqual(G.sha256_json(G.canonical_arrangement(arr)), g["arrangements"][aid]["canonical_sha256"])
                self.assertEqual(G.restore_code_hashes(arr, code["sim/arrange.py"], code["sim/rig.py"]),
                                 g["arrangements"][aid]["arrangement_sha256"])


@unittest.skipUnless(HAVE_SHARE, f"needs the share ({G.SIM_SHARE})")
class TestShareFiles(unittest.TestCase):
    def test_dev60_arrangements_and_index(self):
        g = GOLD["generator"]
        files = sorted(G.DEV60.glob("*.json"))
        self.assertEqual(len(files), 64)
        for p in files:
            a = json.loads(p.read_text())
            with self.subTest(arrangement=p.stem):
                self.assertEqual(G.sha256_json(G.canonical_arrangement(a)), g["arrangements"][p.stem]["canonical_sha256"])
                self.assertEqual(A.arrangement_hash(a), a["arrangement_sha256"])
                self.assertEqual(a["arrangement_sha256"], g["arrangements"][p.stem]["arrangement_sha256"])
        for fn, sha in g["index_files"].items():
            self.assertEqual(G.sha256_json(G.canonical_index_file(fn, json.loads((G.DEV60 / "index" / fn).read_text()))), sha, fn)

    def test_episodes_rebuilt_on_a_copy(self):
        self.assertEqual(G.episodes_hashes(G.SHELF_DEV_V3), GOLD["rendered_set"]["episodes"])
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(G.rebuild_episodes(G.SHELF_DEV_V3, Path(t)), GOLD["rendered_set"]["episodes"])

    def test_check_batch(self):
        cb = GOLD["rendered_set"]["check_batch"]
        doc = json.loads((G.SHELF_DEV_V3 / "check" / "check.json").read_text())
        self.assertEqual(G.check_file_hashes(doc), {"check.json": cb["check.json"], "arrangements": cb["arrangements"]})
        got = G.check_arrangement_hashes(G.SHELF_DEV_V3, G.CHECK_SUBSET)
        self.assertEqual(got, {i: cb["arrangements"][i] for i in G.CHECK_SUBSET})

    @unittest.skipUnless(SLOW, "the full check_set takes ~85 s: set CMPE249_GOLDEN_SLOW=1")
    def test_check_batch_full(self):
        self.assertEqual(G.check_set_hash(G.SHELF_DEV_V3), GOLD["rendered_set"]["check_batch"]["check.json"])


@unittest.skipUnless(SAME_PYTHON, "the stub-render input records the Python version (unhashed.python)")
class TestStubRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx = G.generator_context()
        cls.inputs = {i: G.fake_input_arrangement(GOLD, i, ctx) for i in G.STAGE_LOG_ARRANGEMENTS}

    def test_fake_isaac_call_log_settings_and_labels(self):
        with tempfile.TemporaryDirectory() as t:
            got = G.fake_render(self.inputs[G.FAKE_ARRANGEMENT], Path(t))
        want = {k: v for k, v in GOLD["fake_isaac"].items() if k != "arrangement"}
        self.assertEqual(got, want)

    def test_build_stage_call_log(self):
        for i in G.STAGE_LOG_ARRANGEMENTS:
            with self.subTest(arrangement=i):
                self.assertEqual(G.stage_log_hash(self.inputs[i]), GOLD["stage_log"][i])

    def test_the_call_log_sees_a_lighting_change(self):
        """The golden is sensitive: moving one shelf light by 1 mm changes the recorded calls."""
        from unittest import mock

        from sim import isaac_rig as IR

        dome, lights = IR.LIGHTING["L1"]
        moved = [(n, (p[0] + 0.001, p[1], p[2]) if n == "key_left" else p, s, i, on) for n, p, s, i, on in lights]
        with mock.patch.dict(IR.LIGHTING, {"L1": (dome, moved)}):
            self.assertNotEqual(G.stage_log_hash(self.inputs[G.FAKE_ARRANGEMENT])["sha256"],
                                GOLD["stage_log"][G.FAKE_ARRANGEMENT]["sha256"])


if __name__ == "__main__":
    unittest.main()
