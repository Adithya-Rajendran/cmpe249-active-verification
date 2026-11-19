import tempfile
import unittest
from pathlib import Path

from PIL import Image

from development.corridor_transfer.evaluate import (
    call_type,
    decoded_hash,
    recurrence_label,
)
from development.corridor_transfer.prepare import ENGINES, LAYOUTS, STATES, STOPS


class CorridorEvaluatorTest(unittest.TestCase):
    def test_panel_size_and_unknown_sources(self):
        self.assertEqual(len(ENGINES) * len(LAYOUTS) * len(STOPS) * len(STATES), 192)
        self.assertEqual(
            call_type({"answer": "UNKNOWN", "failure": "registration_gate"}),
            "registration_gate_unknown",
        )
        self.assertEqual(
            call_type({"answer": "UNKNOWN", "failure": None, "raw": "UNKNOWN"}),
            "qwen_unknown",
        )
        self.assertEqual(
            call_type({"answer": "UNKNOWN", "failure": "RuntimeError: fail"}),
            "execution_failure",
        )
        self.assertEqual(
            call_type({"answer": "UNKNOWN", "failure": False}), "rule_unknown"
        )

    def test_zero_opportunities_and_distinct_recurrence(self):
        self.assertEqual(recurrence_label(0, 0), "no_opportunity")
        self.assertEqual(recurrence_label(3, 0), "none_observed")
        self.assertEqual(recurrence_label(3, 1), "isolated")
        self.assertEqual(recurrence_label(3, 2), "recurrent")
        with self.assertRaises(ValueError):
            recurrence_label(1, 2)

    def test_identical_input_requires_decoded_pixels_and_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a.png", Path(folder) / "b.png"
            Image.new("RGB", (8, 4), (1, 2, 3)).save(a, compress_level=0)
            Image.new("RGB", (8, 4), (1, 2, 3)).save(b, compress_level=9)
            self.assertEqual(decoded_hash(a), decoded_hash(b))
            Image.new("RGB", (4, 8), (1, 2, 3)).save(b)
            self.assertNotEqual(decoded_hash(a), decoded_hash(b))


if __name__ == "__main__":
    unittest.main()
