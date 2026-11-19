import unittest

from development.corridor_transfer.scene import build
from development.corridor_transfer.verify_geometry import (
    camera_groups,
    check_rig,
    source_bounds,
)
from sim.isaac_rig import spec_render_settings


class CorridorGeometryTest(unittest.TestCase):
    def test_frontal_travel_is_lateral(self):
        scene = build(4169, "CORFIX", "D6")["scene"]
        cameras = camera_groups(scene)
        bounds = source_bounds(scene)
        self.assertEqual(len(check_rig(cameras, bounds, "D6")), 6)

        frontal = cameras["revisit"]["P3"]
        position = list(cameras["nominal"]["P3"]["position"])
        position[1] += 0.06
        frontal["position"] = position
        with self.assertRaisesRegex(ValueError, "not lateral"):
            check_rig(cameras, bounds, "D6")

    def test_native_capture_metadata_and_lights(self):
        for level in ("D0", "D6"):
            arr = build(4169, "CORFIX", level)
            self.assertEqual(arr["level"], level)
            self.assertEqual(arr["scene"]["tags_enabled"], False)
            settings = spec_render_settings(arr["scene"], ["L1"])
            self.assertIn("L1", settings["lighting_presets"])


if __name__ == "__main__":
    unittest.main()
