"""Test failure paths of the independent capture checks."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from development.cross_sim import qualification as audit
from development.cross_sim import qualification_release as release


class QualificationTest(unittest.TestCase):
    def test_empty_masks_are_valid_but_nonbinary_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.png"
            Image.fromarray(np.zeros((3, 5), np.uint8)).save(path)
            self.assertFalse(audit.get_image(path, (5, 3), mask=True).any())
            Image.fromarray(np.full((3, 5), 180, np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "nonbinary"):
                audit.get_image(path, (5, 3), mask=True)

    def test_reference_and_repeat_are_distinct_outputs(self):
        states = ["reference", "clear_present", "clear_removed", "placed_present",
                  "placed_removed", "repeat_clear_present"]
        files = audit.expected_outputs(["P3"], states)
        self.assertEqual(len(files), 20)
        self.assertIn("masks/blocker_clear_removed_P3.png", files)
        self.assertIn("rgb/repeat_clear_present_P3.png", files)
        self.assertIn("rgb/reference_P3.png", files)

    def test_self_declared_pass_without_files_cannot_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = json.loads(audit.PROTOCOL.read_text())
            manifest = {"study_id": protocol["study_id"], "split": "qualification",
                        "arrangements": [{"base_id": "case"}]}
            manifest_path, review_path = root / "manifest.json", root / "review.json"
            manifest_path.write_text(json.dumps(manifest))
            review_path.write_text(json.dumps({"split": "qualification", "status": "PASS"}))
            units = {f"case/{level}": {"failures": [], "views": dict.fromkeys(protocol["stops"])}
                     for level in protocol["qualification_levels"]}
            paths = {}
            for engine in ("isaac", "mujoco"):
                record = {"status": "PASS", "split": "qualification", "engine": engine,
                          "manifest_sha256": release.digest(manifest_path), "root": str(root),
                          "protocol_sha256": release.digest(audit.PROTOCOL), "files": {},
                          "units": units, "failures": []}
                paths[engine] = root / (engine + ".json")
                paths[engine].write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "incomplete file ledger"):
                release.make_release(manifest_path, audit.PROTOCOL, review_path, paths)

    def test_both_bound_directions_are_inclusive(self):
        failures = []
        audit.limit(failures, "repeat", 3, 3)
        audit.limit(failures, "change", 6, 6, minimum=True)
        self.assertFalse(failures)
        audit.limit(failures, "repeat", 3.001, 3)
        audit.limit(failures, "change", 5.999, 6, minimum=True)
        self.assertEqual([r["check"] for r in failures], ["repeat", "change"])


if __name__ == "__main__":
    unittest.main()
