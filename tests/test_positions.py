"""configs/positions.json schema validation and geometry (PLAN section 5, module 0)."""
import math
import unittest

from verify.schema import DEFAULT_POSITIONS_PATH, POSITIONS_SCHEMA, Positions


class TestPositionsFile(unittest.TestCase):
    def test_file_matches_measured_shelf_geometry(self):
        # configs/positions.json holds the measured shelf arc (2026-10-09: R = 90 cm, stops -45..+45 deg
        # in 22.5 deg steps; evidence data/raw/2026-10-09_measured-geometry/arc_measured.txt).
        # Positions.default() keeps the design template (R = 45 cm, -60..+60 deg) for the synthetic tests.
        loaded, default = Positions.load(), Positions.default()
        self.assertEqual([s.name for s in loaded.stops], [s.name for s in default.stops])
        self.assertEqual([s.azimuth_deg for s in loaded.stops], [-45.0, -22.5, 0.0, 22.5, 45.0])
        self.assertEqual((loaded.radius_cm, loaded.start_index, loaded.f_order),
                         (90.0, default.start_index, default.f_order))
        self.assertAlmostEqual(loaded.travel_cm(2, 3), math.radians(22.5) * 90.0)
        self.assertAlmostEqual(loaded.travel_cm(2, 3), 35.34, places=2)
        self.assertAlmostEqual(loaded.travel_cm(2, 3), loaded.nominal_step_cm())   # equal spacing
        self.assertAlmostEqual(loaded.path_travel_cm((loaded.start_index, *loaded.f_order)), 212.06, places=2)
        self.assertEqual((default.radius_cm, default.azimuth(0)), (45.0, -60.0))   # template unchanged

    def test_schema_is_checked(self):
        d = Positions.default().to_dict()
        self.assertEqual(d["schema"], POSITIONS_SCHEMA)
        self.assertEqual(Positions.from_dict(d), Positions.default())  # round trip
        for bad in ("bogus", None):
            with self.subTest(schema=bad), self.assertRaises(ValueError):
                Positions.from_dict({**d, "schema": bad})
        no_schema = {k: v for k, v in d.items() if k != "schema"}
        with self.assertRaises(ValueError):
            Positions.from_dict(no_schema)

    def test_default_path_is_repo_config(self):
        self.assertEqual(DEFAULT_POSITIONS_PATH.name, "positions.json")
        self.assertTrue(DEFAULT_POSITIONS_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
