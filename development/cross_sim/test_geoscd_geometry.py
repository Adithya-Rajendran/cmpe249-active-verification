import unittest

import numpy as np

from development.cross_sim.geoscd_geometry import GeometryProvider, project_geometry, upsample_displacement


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((12, 16), 2.)
        self.k = np.array([[8., 0, 7.5], [0, 8., 5.5], [0, 0, 1.]])
        self.transform = np.eye(4)

    def test_identity_and_native_non_square(self):
        coords, valid, _ = project_geometry(self.depth, self.depth, self.k, self.k, self.transform)
        y, x = np.indices(self.depth.shape)
        np.testing.assert_allclose(coords, np.stack([x, y], axis=-1), atol=1e-12)
        self.assertTrue(valid.all())
        mapping, good = upsample_displacement(coords, valid, (24, 48), (24, 48))
        yy, xx = np.indices(good.shape)
        np.testing.assert_allclose(mapping[good], np.stack([xx, yy], axis=-1)[good], atol=1e-12)
        self.assertFalse(good[0].any())
        self.assertFalse(good[:, 0].any())
        self.assertTrue(good[12, 24])

    def test_translation_direction(self):
        self.transform[0, 3] = .25
        coords, valid, _ = project_geometry(self.depth, self.depth, self.k, self.k, self.transform)
        self.assertAlmostEqual(coords[5, 5, 0], 6.)
        self.assertTrue(valid[5, 5])
        self.assertFalse(valid[5, -1])
        mapping, good = upsample_displacement(coords, valid, (24, 48), (24, 48))
        self.assertAlmostEqual(mapping[10, 15, 0], 18.)
        self.assertTrue(good[10, 15])

    def test_occlusion_and_missing_depth(self):
        query = self.depth.copy()
        query[4:7, 5:8] = .5
        ref = self.depth.copy()
        ref[1, 1] = np.nan
        query[1, 3] = 0
        coords, valid, _ = project_geometry(ref, query, self.k, self.k, self.transform)
        self.assertFalse(valid[5, 6])
        self.assertFalse(valid[1, 1])
        self.assertFalse(valid[1, 3])
        self.assertFalse(valid[5, 3])  # occlusion dilation
        _, native_valid = upsample_displacement(coords, valid, (24, 48), (24, 48))
        self.assertFalse(native_valid[10, 19])

    def test_provider_boundary_and_nearest(self):
        provider = GeometryProvider.__new__(GeometryProvider)
        provider._predict = lambda ref, query: (np.stack([self.depth, self.depth]), np.stack([self.k, self.k]), self.transform)
        provider.precision, provider.device = 'bfloat16', 'cuda'
        provider.checkpoint_sha256 = 'fixture'
        class TorchFixture:
            __version__ = 'fixture'
        provider.torch = TorchFixture()
        image = np.arange(24 * 48 * 3, dtype=np.uint8).reshape(24, 48, 3)
        warped, valid, diag = provider.register(image, image, (5, 5, 40, 20))
        np.testing.assert_array_equal(warped[valid], image[valid])
        np.testing.assert_array_equal(diag['warped_nearest_bgr'][valid], image[valid])
        self.assertTrue(diag['registration_ok'])
        self.assertEqual(diag['reason'], 'ok')
        self.assertGreater(diag['box_usable_fraction'], 0)
        artifact = provider.geometry_artifact()
        self.assertEqual(artifact['bounds_xyxy'], [4, 4, 41, 21])
        self.assertEqual(artifact['query_xy'].shape, (17, 37, 2))
        np.testing.assert_array_equal(artifact['valid'], valid[4:21, 4:41])
        self.assertEqual(provider.provenance()['geometry_revision'], 'dd31369654e96d6843cc4bbcebce854a8bc2159b')

    def test_behind_camera(self):
        self.transform[2, 3] = -3
        _, valid, diagnostic = project_geometry(self.depth, self.depth, self.k, self.k, self.transform)
        self.assertFalse(valid.any())
        self.assertIsNone(diagnostic['depth_threshold'])

    def test_zero_weight_invalid_neighbor_does_not_reject_exact_sample(self):
        y, x = np.indices((4, 4))
        coords = np.stack((x, y), -1).astype(float)
        valid = np.ones((4, 4), bool)
        valid[1, 2] = False
        _, native_valid = upsample_displacement(coords, valid, (4, 4), (4, 4))
        self.assertTrue(native_valid[1, 1])
        self.assertFalse(native_valid[1, 2])

    def test_missing_coarse_corner_is_conservatively_rejected(self):
        y, x = np.indices((4, 4))
        coords = np.stack((x, y), -1).astype(float)
        valid = np.ones((4, 4), bool)
        valid[1, 1] = False
        coords[1, 1] = np.nan
        _, native_valid = upsample_displacement(coords, valid, (8, 8), (8, 8))
        self.assertFalse(native_valid[3, 3])
        self.assertTrue(native_valid[5, 5])

if __name__ == '__main__':
    unittest.main()
