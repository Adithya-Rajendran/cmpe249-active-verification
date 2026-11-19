import unittest
from development.cross_sim.support_labels.evaluate import screen


def entry(condition, answer):
    return dict(condition=condition, terra_answer=answer, corroborated=True,
                terra_wrong_definite=False, terra_definite_on_exact=False)


class ScreenTest(unittest.TestCase):
    def setUp(self):
        self.rows = [entry("partial", ans) for ans in ("PRESENT", "REMOVED") for _ in range(4)]
        self.rows += [entry("clear", ans) for ans in ("PRESENT", "REMOVED") for _ in range(2)]

    def test_complete_and_short_samples(self):
        self.assertEqual(screen(self.rows)["outcome"], "pass")
        self.assertEqual(screen(self.rows[:-1])["outcome"], "inconclusive")

    def test_definite_on_identical_inputs_overrides_other_passes(self):
        hidden = entry("hidden", "REMOVED")
        hidden["terra_definite_on_exact"] = True
        self.assertEqual(screen(self.rows+[hidden])["outcome"], "fail")

    def test_unresolved_stays_in_denominator(self):
        for row in self.rows[:3]:
            row["corroborated"] = False
        self.assertEqual(screen(self.rows)["outcome"], "fail")


if __name__ == "__main__":
    unittest.main()
