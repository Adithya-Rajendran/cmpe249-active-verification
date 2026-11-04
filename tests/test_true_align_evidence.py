"""sim/true_align_evidence.py: the evaluation-only loader of the simulator's true homographies."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim import true_align_evidence as tae


class TestLoadTrueH(unittest.TestCase):
    def test_reads_both_files_and_the_chosen_plane(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            H_front, H_target = np.eye(3) * 2, np.eye(3) * 3
            stop = {"stop_index": 2, "H_front_plane": H_front.tolist(), "H_target_plane": H_target.tolist()}
            (t / "true_alignment.json").write_text(json.dumps({"arrangements": {"SD1000": {"stops": {"P3": stop}}}}))
            (t / "true_alignment_design.json").write_text(
                json.dumps({"arrangements": {"DES0": {"stops": {"P3": stop}}}}))
            front, target = tae.load_true_h(t, "front"), tae.load_true_h(t, "target")
            self.assertEqual(set(front), {("SD1000", 2), ("DES0", 2)})
            np.testing.assert_array_equal(front[("SD1000", 2)], H_front)
            np.testing.assert_array_equal(target[("DES0", 2)], H_target)

    def test_a_missing_design_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            stop = {"stop_index": 0, "H_front_plane": np.eye(3).tolist(), "H_target_plane": np.eye(3).tolist()}
            (t / "true_alignment.json").write_text(json.dumps({"arrangements": {"SD1001": {"stops": {"P1": stop}}}}))
            self.assertEqual(list(tae.load_true_h(t, "target")), [("SD1001", 0)])


if __name__ == "__main__":
    unittest.main()
