import unittest
from development.cross_sim.support_labels.prepare import select


def unit(fractions):
    return {e: {"clear": 100, "alone": 100, "placed": round(100*f),
                "fraction_visible": f, "fraction_blocker": 1-f}
            for e, f in zip(("isaac", "mujoco"), fractions)}


class SelectionTest(unittest.TestCase):
    def test_paired_partial_and_unique_positions(self):
        pool = {("D0","P1"): unit((.1,.90)), ("D6","P2"): unit((.5,.4)),
                ("D0","P3"): unit((0,0)), ("D0","P4"): unit((1,1))}
        rows, counts = select("LQS01", pool)
        self.assertEqual(counts["partial_eligible"], 2)
        self.assertEqual(len({k for k, _, _ in rows}), len(rows))
        self.assertEqual({s for _, g, s in rows if g == "partial"}, {"placed_present","placed_removed"})
        self.assertIn((("D0","P3"),"hidden","placed_removed"), rows)
        self.assertIn((("D0","P4"),"clear","clear_present"), rows)

    def test_one_engine_cannot_supply_other_engines_partial_view(self):
        pool = {("D0","P1"): unit((.40,0)), ("D6","P2"): unit((.5,.4))}
        rows, counts = select("LQT02", pool)
        self.assertEqual(counts["partial_eligible"], 1)
        self.assertEqual([s for _, g, s in rows if g == "partial"], ["placed_removed"])
        self.assertFalse(any(g == "hidden" for _, g, _ in rows))

    def test_distinct_stops_and_levels_have_priority(self):
        pool = {(lv, stop): unit((.5,.5)) for lv in ("D0","D6") for stop in ("P1","P2")}
        partial = [k for k, group, _ in select("LQS01", pool)[0] if group == "partial"]
        self.assertNotEqual(partial[0][0], partial[1][0])
        self.assertNotEqual(partial[0][1], partial[1][1])


if __name__ == "__main__":
    unittest.main()
