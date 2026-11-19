import unittest

import cv2
import numpy as np

from development.cross_sim.dense_alignment import warp_query


class DenseAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        y, x = np.indices(self.image.shape[:2])
        self.grid = np.stack((x, y), axis=-1).astype(float)
        self.valid = np.ones(self.image.shape[:2], bool)
        self.occ = np.zeros(self.image.shape[:2], bool)

    def warp(self, coords, valid=None, occ=None, transform=None):
        return warp_query(self.image, coords, self.valid if valid is None else valid,
                          self.occ if occ is None else occ, np.eye(3) if transform is None else transform)

    def test_identity(self):
        out, valid = self.warp(self.grid)
        np.testing.assert_array_equal(out, self.image)
        self.assertTrue(valid.all())

    def test_direction_and_sign(self):
        coords = self.grid.copy()
        coords[..., 0] += 1
        out, valid = self.warp(coords)
        np.testing.assert_array_equal(out[:, :-1], self.image[:, 1:])
        self.assertFalse(valid[:, -1].any())
        self.assertFalse(out[:, -1].any())

    def test_explicit_non_square_padded_query_coordinates(self):
        # Original 8x6 query centered in an 8x8 canvas, followed by 2x resize.
        # Pixel-center resize convention: (x + pad + 0.5) * 2 - 0.5.
        transform = np.array([[2, 0, .5], [0, 2, 2.5], [0, 0, 1]])
        padded = self.grid @ transform[:2, :2].T + transform[:2, 2]
        out, valid = self.warp(padded, transform=transform)
        np.testing.assert_array_equal(out, self.image)
        self.assertTrue(valid.all())
        padded[0, 0] = [0, 0]
        _, valid = self.warp(padded, transform=transform)
        self.assertFalse(valid[0, 0])

    def test_invalid_occluded_nonfinite_and_outside(self):
        coords = self.grid.copy()
        coords[0, 0] = [np.nan, 0]
        coords[0, 1] = [np.inf, 0]
        coords[0, 2] = [-0.01, 1]
        coords[0, 3] = [7.01, 1]
        valid, occ = self.valid.copy(), self.occ.copy()
        valid[0, 4] = False
        occ[0, 5] = True
        out, got = self.warp(coords, valid, occ)
        self.assertFalse(got[0, :6].any())
        self.assertFalse(out[0, :6].any())
        self.assertTrue(got[1:].all())

    def test_matches_existing_perspective_warp_for_integer_translation(self):
        from perception.evidence import register
        homography = np.array([[1., 0., -1.], [0., 1., 0.], [0., 0., 1.]])
        old, old_valid, ok = register(self.image, self.image, lambda *_: homography)
        coords = self.grid.copy()
        coords[..., 0] += 1
        new, new_valid = self.warp(coords)
        self.assertTrue(ok)
        np.testing.assert_array_equal(new, old)
        np.testing.assert_array_equal(new_valid, old_valid)

    def test_rejects_ambiguous_or_malformed_inputs(self):
        for kwargs in ({'valid': self.valid.astype(np.uint8)}, {'occ': self.occ.astype(np.uint8)},
                       {'transform': np.zeros((3, 3))}, {'transform': np.eye(3, dtype=complex)}):
            with self.assertRaises(ValueError):
                self.warp(self.grid, **kwargs)
        with self.assertRaises(ValueError):
            self.warp(self.grid.astype(complex))


if __name__ == '__main__':
    unittest.main()
