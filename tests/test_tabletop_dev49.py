"""The pinned 49-arrangement tabletop development remainder (configs/sim/tabletop_dev49.json, Amendment 2).

Pins: the remainder is slots 11-59 of the dev slot order (TD5011-TD5050, TD6200 for slot 51 from its reserve list,
TD5052-TD5059), the expected count is 49, and together with the 11-arrangement transfer diagnostic it is exactly the
60-slot dev quota with no overlap; the design cases TDES0-TDES3 are listed apart; the shelf thresholds are the
recorded ones. When the untracked dev60 output is present its slot list and files must still have the pinned hashes,
and (CMPE249_GOLDEN_SLOW=1) a fresh generation of slots 11-59 with the design cases must reproduce every
arrangement_sha256.
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
CFG = REPO / "configs" / "sim" / "tabletop_dev49.json"
CFG11 = REPO / "configs" / "sim" / "tabletop_transfer11.json"
DEV60 = REPO / "runs" / "2026-11-04-tabletop-dev60" / "dev60"
SLOW = os.environ.get("CMPE249_GOLDEN_SLOW") == "1"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _cls(slot: dict) -> str:
    return slot["family"] + (f"-{slot['m16_side']}" if slot["m16_side"] else "")


class TestRemainder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(CFG.read_text())
        cls.arr = cls.cfg["arrangements"]

    def test_the_pinned_identities(self):
        self.assertEqual(self.cfg["expected_count"], 49)
        self.assertEqual([a["slot"] for a in self.arr], list(range(11, 60)))
        want = [f"TD{5000 + i}" for i in range(11, 51)] + ["TD6200"] + [f"TD{5000 + i}" for i in range(52, 60)]
        self.assertEqual([a["arrangement_id"] for a in self.arr], want)
        self.assertEqual([a["seed"] for a in self.arr], [int(i[2:]) for i in want])
        self.assertTrue(self.cfg["label"].startswith("49-ARRANGEMENT TABLETOP DEVELOPMENT REMAINDER"))
        self.assertEqual([d["arrangement_id"] for d in self.cfg["design_cases"]], [f"TDES{i}" for i in range(4)])
        self.assertFalse(any(a["arrangement_id"].startswith("TDES") for a in self.arr))

    def test_with_the_first_11_it_is_the_whole_dev_quota(self):
        first = json.loads(CFG11.read_text())["arrangements"]
        self.assertEqual(sorted(a["slot"] for a in first + self.arr), list(range(60)))
        quota = json.loads((REPO / self.cfg["source"]["arrange_config"]).read_text())["slots"]["dev"]["quota"]
        want = {}
        for fam, n in quota.items():
            for side, m in (n.items() if isinstance(n, dict) else [(None, n)]):
                want[fam + (f"-{side}" if side else "")] = m
        got = {}
        for a in first + self.arr:
            got[a["class"]] = got.get(a["class"], 0) + 1
        self.assertEqual(got, want)
        self.assertEqual(self.cfg["class_counts"], {c: sum(a["class"] == c for a in self.arr) for c in sorted(got)})

    def test_thresholds_are_the_shelf_ones(self):
        th = self.cfg["shelf_thresholds"]
        v03 = th["v03_image"]
        self.assertEqual((v03["tau_q"], v03["tau_occ"], v03["tau_match"], v03["k"], v03["lam"]),
                         (0.3, 0.35, 0.65, 2, 0.0))
        self.assertEqual(v03["sha256"], "4a06de0eebe407aad27b58fdfce3751460b5357924aba93a6f6830095b93b983")
        self.assertIsNone(th["v021_image"])
        self.assertEqual(self.cfg["source"]["arrange_config_sha256"], _sha(REPO / self.cfg["source"]["arrange_config"]))
        self.assertEqual(self.cfg["source"]["slot_list_sha256"],
                         json.loads(CFG11.read_text())["source"]["slot_list_sha256"])

    @unittest.skipUnless(DEV60.is_dir(), "untracked dev60 output not present")
    def test_the_dev60_output_still_matches(self):
        slots = DEV60 / "index" / "slots.json"
        self.assertEqual(_sha(slots), self.cfg["source"]["slot_list_sha256"])
        order = json.loads(slots.read_text())["slots"]
        for a in self.arr + self.cfg["design_cases"]:
            f = DEV60 / f"{a['arrangement_id']}.json"
            self.assertEqual(_sha(f), a["file_sha256"], f.name)
            doc = json.loads(f.read_text())
            self.assertEqual(doc["arrangement_sha256"], a["arrangement_sha256"], f.name)
            if "slot" in a:
                s = order[a["slot"]]
                self.assertEqual((doc["slot"]["index"], doc["slot"]["used_seed"]), (a["slot"], a["seed"]), f.name)
                self.assertEqual((_cls(s), s["target"]), (a["class"], a["slot_target"]), f.name)

    @unittest.skipUnless(SLOW, "generating 53 arrangements takes about three minutes: set CMPE249_GOLDEN_SLOW=1")
    def test_a_fresh_generation_reproduces_the_pins(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "gen"
            subprocess.run([sys.executable, "-m", "sim.arrange", "--config", self.cfg["source"]["arrange_config"],
                            "--split", "dev", "--slots", "11-59", "--design", "--out", str(out)], cwd=REPO, check=True,
                           capture_output=True)
            pinned = self.arr + self.cfg["design_cases"]
            got = sorted(p.name for p in out.glob("*.json"))
            self.assertEqual(got, sorted(f"{a['arrangement_id']}.json" for a in pinned))
            for a in pinned:
                doc = json.loads((out / f"{a['arrangement_id']}.json").read_text())
                self.assertEqual(doc["arrangement_sha256"], a["arrangement_sha256"], a["arrangement_id"])
                self.assertEqual(doc["target_id"], a["target_id"], a["arrangement_id"])


if __name__ == "__main__":
    unittest.main()
