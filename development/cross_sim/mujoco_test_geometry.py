"""Checks against the existing Isaac geometry conversion."""
import unittest
import numpy as np
from development.cross_sim import mujoco_geometry as mg
from sim import isaac_batch as ib
from sim import isaac_rig as ir


class GeometryTest(unittest.TestCase):
    def test_ycb_conversion_matches_isaac(self):
        v = np.array([[10, 20, 30], [70, 90, 110], [20, 80, 40]])
        r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for actual, expected in zip(mg.local_ycb(v, r, np.eye(3)), ib.asset_local(v, r, np.eye(3))):
            np.testing.assert_allclose(actual, expected)
        self.assertEqual(mg.local_ycb(v, r)[0][:, 2].min(), 0)

    def test_camera_matches_isaac(self):
        forward, up = [0.3, 1, -0.2], [0, 0, 1]
        np.testing.assert_allclose(mg.camera_axes(forward, up), ir.camera_matrix([1, 2, 3], forward, up)[:3, :3])
        self.assertAlmostEqual(mg.fovy_degrees(1000, 1000), 53.13010235415598)
        with self.assertRaises(ValueError):
            mg.camera_axes([0, 0, 1], [0, 0, 2])

    def test_world_pose_matches_isaac(self):
        v = np.array([[1, 2, 3], [-2, 0, 1]])
        m = ib.pose_matrix([2, 3, 4], 37)
        expected = np.c_[v, np.ones(2)] @ m
        np.testing.assert_allclose(mg.world_yaw(v, [2, 3, 4], 37), expected[:, :3])


if __name__ == '__main__':
    unittest.main()
