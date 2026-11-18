"""Regression checks for the scientific error in the old magnitude-only diagnostic."""
import unittest
import numpy as np
from development.stress.alignment_diagnostics import (
    load_alignment_only, observe_fit, point_errors, rendered_to_input_homography,
)


class AlignmentDiagnosticTests(unittest.TestCase):
    def test_equal_lengths_opposite_directions_are_wrong(self):
        left = np.array([[1., 0, 25], [0, 1, 0], [0, 0, 1]])
        right = np.array([[1., 0, -25], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(point_errors(left, right, [[100, 100]])["errors_px"], [50.0])

    def test_background_correctness_does_not_imply_target_correctness(self):
        identity = np.eye(3)
        target = np.array([[1., 0, 8], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(point_errors(identity, identity, [[10, 20]])["errors_px"], [0.0])
        self.assertEqual(point_errors(identity, target, [[10, 20]])["errors_px"], [8.0])

    def test_resize_and_half_pixel_conversion(self):
        h = np.array([[1., 0, 10], [0, 1, 4], [0, 0, 1]])
        self.assertTrue(np.allclose(rendered_to_input_homography(h, .5), [[1, 0, 5], [0, 1, 2], [0, 0, 1]]))

    def test_observation_preserves_real_frozen_guard_decision(self):
        import cv2
        module = load_alignment_only()
        rng = np.random.default_rng(62)
        ref = cv2.resize(rng.integers(0, 256, (50, 75, 3), dtype=np.uint8),
                         (600, 400), interpolation=cv2.INTER_CUBIC)
        query = cv2.warpPerspective(ref, np.array([[1., 0, 10], [0, 1, 0], [0, 0, 1]]), (600, 400))
        params = module.AlignParams(levels=2, stop_level=0, max_shift_px=2)
        box = (240, 140, 310, 230)
        plain_h, plain_info = module.image_align(ref, query, box, params)
        observed_h, observed_info, raw = observe_fit(module, ref, query, box, params)
        self.assertIsNone(plain_h)
        self.assertIsNone(observed_h)
        self.assertEqual(plain_info, observed_info)
        self.assertTrue(plain_info['refused'].startswith('box moves'))
        self.assertIsNotNone(raw)
        truth = np.array([[1., 0, -10], [0, 1, 0], [0, 0, 1]])
        self.assertLess(point_errors(raw, truth, [[275, 185]])['errors_px'][0], 1.0)


if __name__ == '__main__':
    unittest.main()
