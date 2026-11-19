import unittest
import numpy as np
from development.cross_sim import geometry_residuals as g


class GeometryResidualTests(unittest.TestCase):
    def test_inverse_baseline_and_matched_translation(self):
        y, x = np.indices((7, 8))
        xy = np.stack((x + 2, y - 1), axis=-1).astype(np.float32)
        ref = np.array([[1., 1.], [2.25, 3.5], [5., 4.]])
        truth = ref + [2, -1]
        h = np.array([[1., 0, -2.], [0, 1., 1.], [0, 0, 1.]])
        result = g.compare_points(ref, truth, [0, 0, 8, 7], h, xy, np.ones((7,8), bool), [0,0,8,7])
        self.assertEqual(result['common_vertices'], 3)
        self.assertEqual(result['candidate_common']['p90_px'], 0)
        self.assertEqual(result['baseline_common']['p90_px'], 0)
        wrong = g.compare_points(ref, truth, [0,0,8,7], np.linalg.inv(h), xy, np.ones((7,8), bool), [0,0,8,7])
        self.assertGreater(wrong['baseline_common']['median_px'], 4)

    def test_mask_positive_weights_invalid_and_outside(self):
        y, x = np.indices((4,4))
        xy = np.stack((x, y), -1).astype(np.float32)
        valid = np.ones((4,4), bool); valid[1,2] = False
        points = np.array([[1.,1.], [1.5,1.], [2.,1.], [3.01,1.], [np.nan,1.]])
        mapped, good, inside = g.sample_map(xy, valid, [0,0,4,4], points)
        np.testing.assert_array_equal(good, [True,False,False,False,False])
        np.testing.assert_array_equal(inside, [True,True,True,False,False])
        np.testing.assert_array_equal(mapped[0], [1,1])
        self.assertTrue(np.isnan(mapped[1:]).all())

    def test_missing_no_zero_error_and_common(self):
        y, x = np.indices((5,5)); xy = np.stack((x,y),-1).astype(np.float32)
        valid = np.ones((5,5), bool); valid[2,2] = False
        pts = np.array([[1.,1.],[2.,2.],[3.,3.],[9.,9.]])
        result = g.compare_points(pts, pts, [0,0,5,5], None, xy, valid, [0,0,5,5])
        self.assertEqual(result['reference_box_vertices'], 3)
        self.assertEqual(result['candidate_missing_or_occluded'], 1)
        self.assertEqual(result['candidate_available'], 2)
        self.assertEqual(result['baseline_available'], 0)
        self.assertEqual(result['baseline']['median_px'], None)
        self.assertEqual(result['baseline_common']['count'], 0)
        self.assertEqual(result['candidate_common']['median_px'], None)

    def test_malformed_geometry_and_singular_h(self):
        mapped, valid, status = g.project_homography(np.zeros((3,3)), np.array([[1.,1.]]))
        self.assertEqual(status, 'invalid_matrix')
        self.assertFalse(valid[0])
        with self.assertRaises(ValueError):
            g.sample_map(np.zeros((2,2,2), np.float64), np.ones((2,2), bool), [0,0,2,2], np.zeros((1,2)))

    def test_summary_stratifies_outcomes_and_missing(self):
        y,x = np.indices((3,3)); xy = np.stack((x,y),-1).astype(np.float32)
        fit = g.compare_points(np.array([[1.,1.]]), np.array([[1.,1.]]), [0,0,3,3], np.eye(3), xy, np.ones((3,3), bool), [0,0,3,3])
        rows = [dict(engine='isaac',environment='shelf',level='D0',state='clear_removed',candidate_outcome='unknown',baseline_outcome='correct',geometry=fit)]
        groups = g.summarize(rows)
        self.assertTrue(any(r['group'].get('candidate_outcome') == 'unknown' and r['common_vertices'] == 1 for r in groups))


if __name__ == '__main__': unittest.main()
