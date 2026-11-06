"""The descriptor and 49-arrangement evidence specs (configs/pipeline): they parse, use only the placeholders
tools/pipeline.py fills in, call perception.side_descriptor with options it has, and the 49's spec points at the set and
jobs the pin (configs/sim/tabletop_dev49.json) describes."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from perception import side_descriptor as sd

REPO = Path(__file__).resolve().parents[1]
SPECS = ("shelf_dev_v3_side", "tabletop_transfer11_side", "tabletop_dev49_evidence")
KNOWN = {"jobs", "out", "part", "set"}
CLI_OPTIONS = {"--jobs", "--out", "--workers"}   # perception/side_descriptor.py main()


class SideSpecTest(unittest.TestCase):
    def test_specs(self) -> None:
        for name in SPECS:
            with self.subTest(spec=name):
                spec = json.loads((REPO / "configs" / "pipeline" / f"{name}.json").read_text())
                ev = spec["evidence"]
                self.assertTrue(ev["out_dir"].startswith(spec["set_root"]))
                for v in ev["variants"].values():
                    for text in (*v["args"], *v["outputs"]):
                        self.assertLessEqual(set(re.findall(r"{(\w+)}", text)), KNOWN, text)
                side = ev["variants"].get("side")
                self.assertIsNotNone(side)
                self.assertEqual(side["args"][:2], ["-m", sd.__name__])
                flags = {a for a in side["args"] if a.startswith("--")}
                self.assertLessEqual(flags, CLI_OPTIONS)
                self.assertEqual(side["outputs"], ["{out}/side_{part}.jsonl"])

    def test_the_49s_spec_matches_its_pin(self) -> None:
        spec = json.loads((REPO / "configs" / "pipeline" / "tabletop_dev49_evidence.json").read_text())
        pin = json.loads((REPO / "configs" / "sim" / "tabletop_dev49.json").read_text())
        self.assertTrue(spec["set_root"].endswith("/sim/tabletop_dev49"))
        self.assertEqual(spec["evidence"]["jobs"], ["jobs_refL1.json", "jobs_design_refL1.json"])
        self.assertEqual(len(pin["design_cases"]), 4)                  # the design jobs file holds TDES0-TDES3
        self.assertNotIn("select", spec)
        self.assertNotIn("summary", spec)                              # sealed until A's design was pinned
        self.assertEqual(spec["evidence"]["variants"]["v03_image"]["args"][:4],
                         ["-m", "perception.evidence_v03", "--align", "image"])


if __name__ == "__main__":
    unittest.main()
