"""The pinned 11-arrangement transfer diagnostic (configs/sim/tabletop_transfer11.json, tabletop check (c)).

Pins: the selection is slots 0-10 (TD5000-TD5010), the shortest prefix of the dev slot order covering all six
classes; the expected count is 11; no design case is in it; the shelf thresholds are the recorded ones and nothing
is calibrated on the subset. When the untracked dev60 output is present its slot list and files must still have the
pinned hashes, and (CMPE249_GOLDEN_SLOW=1) a fresh generation of slots 0-10 must reproduce every arrangement_sha256.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "sim" / "tabletop_transfer11.json"
DEV60 = REPO / "runs" / "2026-11-04-tabletop-dev60" / "dev60"
CLASSES = {"IND28", "M16-L", "M16-R", "S8", "white_box", "ycb_box"}
SLOW = os.environ.get("CMPE249_GOLDEN_SLOW") == "1"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _cls(slot: dict) -> str:
    return slot["family"] + (f"-{slot['m16_side']}" if slot["m16_side"] else "")


class TestSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(CFG.read_text())
        cls.arr = cls.cfg["arrangements"]

    def test_the_pinned_identities(self):
        self.assertEqual(self.cfg["expected_count"], 11)
        self.assertEqual([a["slot"] for a in self.arr], list(range(11)))
        self.assertEqual([a["arrangement_id"] for a in self.arr], [f"TD{5000 + i}" for i in range(11)])
        self.assertEqual([a["seed"] for a in self.arr], list(range(5000, 5011)))
        self.assertTrue(self.cfg["label"].startswith("11-ARRANGEMENT TRANSFER DIAGNOSTIC"))
        self.assertFalse(any(a["arrangement_id"].startswith("TDES") for a in self.arr))

    def test_it_is_the_shortest_all_class_prefix(self):
        classes = [a["class"] for a in self.arr]
        self.assertEqual(set(classes), CLASSES)
        self.assertLess(len(set(classes[:-1])), len(CLASSES))
        self.assertEqual(self.cfg["class_counts"], {c: classes.count(c) for c in sorted(CLASSES)})

    def test_thresholds_are_the_shelf_ones(self):
        th = self.cfg["shelf_thresholds"]
        v03 = th["v03_image"]
        self.assertEqual((v03["tau_q"], v03["tau_occ"], v03["tau_match"], v03["k"], v03["lam"]), (0.3, 0.35, 0.65, 2, 0.0))
        self.assertEqual(v03["sha256"], "4a06de0eebe407aad27b58fdfce3751460b5357924aba93a6f6830095b93b983")
        self.assertIsNone(th["v021_image"])
        self.assertEqual(self.cfg["source"]["arrange_config_sha256"], _sha(REPO / self.cfg["source"]["arrange_config"]))

    @unittest.skipUnless(DEV60.is_dir(), "untracked dev60 output not present")
    def test_the_dev60_output_still_matches(self):
        slots = DEV60 / "index" / "slots.json"
        self.assertEqual(_sha(slots), self.cfg["source"]["slot_list_sha256"])
        order = json.loads(slots.read_text())["slots"]
        for a in self.arr:
            s = order[a["slot"]]
            self.assertEqual((s["primary_seed"], _cls(s), s["target"]), (a["seed"], a["class"], a["slot_target"]))
            f = DEV60 / f"{a['arrangement_id']}.json"
            self.assertEqual(_sha(f), a["file_sha256"], f.name)
            self.assertEqual(json.loads(f.read_text())["arrangement_sha256"], a["arrangement_sha256"], f.name)

    @unittest.skipUnless(SLOW, "generating 11 arrangements takes about a minute: set CMPE249_GOLDEN_SLOW=1")
    def test_a_fresh_generation_reproduces_the_pins(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "gen"
            subprocess.run([sys.executable, "-m", "sim.arrange", "--config", self.cfg["source"]["arrange_config"],
                            "--split", "dev", "--slots", "0-10", "--out", str(out)], cwd=REPO, check=True,
                           capture_output=True)
            got = sorted(p.name for p in out.glob("*.json"))
            self.assertEqual(got, [f"{a['arrangement_id']}.json" for a in self.arr])
            for a in self.arr:
                doc = json.loads((out / f"{a['arrangement_id']}.json").read_text())
                self.assertEqual(doc["arrangement_sha256"], a["arrangement_sha256"], a["arrangement_id"])
                self.assertEqual(doc["target_id"], a["target_id"], a["arrangement_id"])


if __name__ == "__main__":
    unittest.main()
