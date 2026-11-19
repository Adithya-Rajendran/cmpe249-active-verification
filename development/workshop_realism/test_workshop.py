"""Regress the paired comparison and the physical camera mounts."""

import copy
import unittest

import numpy as np

from development.corridor_transfer.evaluate import recurrence_label
from development.corridor_transfer.scene import build as plain
from development.workshop_realism.evaluate import compare, statistics
from development.workshop_realism.scene import arrange


class WorkshopTest(unittest.TestCase):
    def test_legacy_views_and_target_are_exact(self):
        for i in range(1, 9):
            for level in ("D0", "D6"):
                with self.subTest(i=i, level=level):
                    a = plain(4170 + i, f"COR{i:02}", level)
                    b = arrange(a)
                    self.assertEqual(a["objects"]["target"], b["objects"]["target"])
                    self.assertEqual(
                        a["objects"]["occluder"]["poses"],
                        b["objects"]["occluder"]["poses"],
                    )
                    for group in ("cameras", "revisit_cameras"):
                        self.assertEqual(a["scene"][group], b["scene"][group][:3])

    def test_forward_motion_is_hardware_relative(self):
        plain_spec = plain(4171, "COR01", "D6")
        new = arrange(plain_spec)
        for before, after, height, pitch, degrees in zip(
            new["scene"]["cameras"][3:],
            new["scene"]["revisit_cameras"][3:],
            [0.95, 0.48],
            [13, 8],
            [1, 0],
        ):
            self.assertEqual(before["position"][2], height)
            self.assertEqual(after["pitch_down_deg"], pitch)
            offset = np.subtract(after["position"], before["position"])
            flat = np.array(before["forward"][:2])
            flat /= np.linalg.norm(flat)
            np.testing.assert_allclose(offset[:2], 0.06 * flat, atol=1e-9)
            self.assertAlmostEqual(
                abs(after["heading_deg"] - before["heading_deg"]), degrees, places=8
            )

    def test_object_position_cannot_aim_added_cameras(self):
        a = plain(4171, "COR01", "D6")
        b = copy.deepcopy(a)
        b["objects"]["target"]["pose"]["base"][0] += 0.14
        first = arrange(a)
        second = arrange(b)
        for group in ("cameras", "revisit_cameras"):
            self.assertEqual(first["scene"][group][3:], second["scene"][group][3:])

    def test_exact_opportunity_and_nonrecurrence_are_distinct(self):
        self.assertEqual(recurrence_label(0, 0), "no_opportunity")
        self.assertEqual(recurrence_label(8, 0), "none_observed")
        self.assertEqual(recurrence_label(8, 1), "isolated")
        self.assertEqual(recurrence_label(8, 2), "recurrent")
        old = {
            "complete_input_equal": "True",
            "present_answer": "REMOVED",
            "removed_answer": "REMOVED",
        }
        new = {"exact": False, "present_answer": "REMOVED", "removed_answer": "REMOVED"}
        self.assertEqual(compare(old, new), "lost_exact_opportunity")
        new.update(exact=True, present_answer="UNKNOWN")
        self.assertEqual(compare(old, new), "resolved_on_exact_input")
        new.update(present_answer="REMOVED")
        self.assertEqual(compare(old, new), "persistent")
        old.update(present_answer="UNKNOWN")
        self.assertEqual(compare(old, new), "new")

    def test_count_unknowns_and_exposure_controls(self):
        decisions = [
            {
                "answer": "UNKNOWN",
                "correct": False,
                "truth": "PRESENT",
                "call_type": "registration_gate_unknown",
                "raw": None,
            },
            {
                "answer": "UNKNOWN",
                "correct": False,
                "truth": "REMOVED",
                "call_type": "qwen_unknown",
                "raw": "UNKNOWN",
            },
        ]
        pairs = [
            {
                "door": "clear",
                "visible_fraction": 0.8,
                "present_answer": "UNKNOWN",
                "removed_answer": "UNKNOWN",
            }
        ]
        score = statistics(decisions, pairs, "vggt_qwen")
        self.assertEqual(
            (
                score["queries"],
                score["registration_stops"],
                score["model_unknown"],
                score["open_exposed"],
                score["definite"],
            ),
            (2, 1, 1, 0, 0),
        )


if __name__ == "__main__":
    unittest.main()
