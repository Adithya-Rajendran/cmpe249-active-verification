"""Check the continuous door travel and two-sided layout mirror."""

import unittest

from development.corridor_transfer.scene import build
from development.workshop_realism.door_clearance import SILL_GAP, modify, sweep
from development.workshop_realism.scene import arrange


class ClearanceTest(unittest.TestCase):
    def test_former_wall_overlap_and_repaired_sweep(self):
        for i in range(1, 9):
            with self.subTest(layout=i):
                old = arrange(build(4170 + i, f"COR{i:02}", "D6"))
                self.assertTrue(
                    any(n in sweep(old)["hits"] for n in ["left_wall", "right_wall"])
                )
                updated = modify(old)
                metrics = sweep(updated)
                self.assertEqual(metrics["hits"], [])
                self.assertGreaterEqual(metrics["minimum_static_clearance_m"], 0.01)
                self.assertAlmostEqual(
                    metrics["gap_to_parking_wall_m"], 0.225, places=8
                )
                self.assertEqual(
                    updated["objects"]["occluder"]["poses"]["parked"]["base"][2],
                    SILL_GAP,
                )
                self.assertEqual(sweep(updated)["door_open_edge_to_center_m"], 0.505)

    def test_continuous_proof_detects_obstacle_between_clear_endpoints(self):
        a = modify(arrange(build(4171, "COR01", "D6")))
        p = a["objects"]["occluder"]["poses"]["placed"]["base"]
        xyz = [-0.49, p[1], 0.35]
        # The 3 cm space between the two endpoint boxes is swept while the leaf moves.
        a["scene"]["boxes"].append(
            {
                "name": "mid_travel_obstacle",
                "center": xyz,
                "half": [0.001, 0.001, 0.001],
            }
        )
        for state in ("parked", "placed"):
            self.assertGreater(
                abs(xyz[0] - a["objects"]["occluder"]["poses"][state]["base"][0]),
                0.475 + 0.001,
            )
        self.assertIn("mid_travel_obstacle", sweep(a)["hits"])


if __name__ == "__main__":
    unittest.main()
