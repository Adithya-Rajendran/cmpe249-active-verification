"""Tests for the pure parts of sim/isaac_rig.py (no Isaac Sim needed: numpy only).

The USD/RTX side runs only in the Isaac Sim env on the GB10. It was checked end to end on 2026-11-02:
on the toy scene its visible fractions matched MuJoCo's to the pixel at P1 (13163 vs 13162). These
tests pin the camera math and the PLY reader that check depends on.
"""
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim import isaac_rig as ir


def _ply_ascii(path: Path) -> None:
    path.write_text(
        "ply\nformat ascii 1.0\ncomment TextureFile tex.png\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\n"
        "property float nz\nproperty float texture_u\nproperty float texture_v\nelement face 1\n"
        "property list uchar int vertex_indices\nend_header\n"
        "0 0 0 0 0 1 0 0\n1 0 0 0 0 1 1 0\n1 1 0 0 0 1 1 1\n0 1 0 0 0 1 0 1\n4 0 1 2 3\n")


def _ply_binary(path: Path) -> None:
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
            "property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n").encode()
    body = struct.pack("<9f", 0, 0, 0, 2, 0, 0, 0, 3, 0) + bytes([3]) + struct.pack("<3i", 0, 1, 2)
    path.write_bytes(head + body)


class TestPly(unittest.TestCase):
    def test_ascii_quad_is_fan_triangulated_with_normals_uv_texture(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "q.ply"
            _ply_ascii(p)
            d = ir.read_ply(p)
        self.assertEqual(d["vertices"].shape, (4, 3))
        self.assertEqual(d["faces"].tolist(), [[0, 1, 2], [0, 2, 3]])
        self.assertEqual(d["normals"].shape, (4, 3))
        self.assertEqual(d["uv"].tolist()[2], [1.0, 1.0])
        self.assertEqual(d["texture"], "tex.png")

    def test_binary_little_endian(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "b.ply"
            _ply_binary(p)
            d = ir.read_ply(p)
        self.assertEqual(d["vertices"].tolist(), [[0, 0, 0], [2, 0, 0], [0, 3, 0]])
        self.assertEqual(d["faces"].tolist(), [[0, 1, 2]])
        self.assertIsNone(d["normals"])
        self.assertIsNone(d["uv"])

    def test_bad_format_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.ply"
            p.write_text("ply\nformat binary_big_endian 1.0\nelement vertex 0\nend_header\n")
            with self.assertRaises(ValueError):
                ir.read_ply(p)


class TestCameraMath(unittest.TestCase):
    def test_camera_looks_along_forward_with_up_up(self):
        pos, fwd = np.array([0.3, -0.8, 0.4]), np.array([-0.3, 0.9, -0.1])
        m = ir.camera_matrix(pos, fwd, (0, 0, 1))
        r, u, back, t = m[0, :3], m[1, :3], m[2, :3], m[3, :3]
        self.assertTrue(np.allclose(np.stack([r, u, back]) @ np.stack([r, u, back]).T, np.eye(3), atol=1e-12))
        self.assertTrue(np.allclose(-back, fwd / np.linalg.norm(fwd)))  # USD cameras look down local -Z
        self.assertGreater(u[2], 0.9)  # image up stays world-up for a near-level camera
        self.assertTrue(np.allclose(t, pos))
        self.assertAlmostEqual(np.linalg.det(np.stack([r, u, back])), 1.0, places=12)  # right-handed

    def test_parallel_forward_and_up_refused(self):
        with self.assertRaises(ValueError):
            ir.camera_matrix((0, 0, 0), (0, 0, 2), (0, 0, 1))

    def test_apertures_reproduce_the_rig_fov(self):
        h, v = ir.apertures(1731.26, 2304, 1296)
        self.assertAlmostEqual(math.degrees(2 * math.atan(h / 2 / ir.FOCAL_MM)), 67.28, places=2)
        self.assertAlmostEqual(math.degrees(2 * math.atan(v / 2 / ir.FOCAL_MM)), 41.04, places=2)
        self.assertAlmostEqual(h / v, 2304 / 1296, places=12)  # square pixels

    def test_quat_matrix_is_a_rotation_and_matches_a_known_turn(self):
        a = math.radians(30)
        r = ir.quat_matrix((math.cos(a / 2), 0, math.sin(a / 2), 0))  # 30 deg about +y
        self.assertTrue(np.allclose(r @ r.T, np.eye(3)))
        self.assertTrue(np.allclose(r @ [1, 0, 0], [math.cos(a), 0, -math.sin(a)]))
        self.assertTrue(np.allclose(ir.quat_matrix((2, 0, 0, 0)), np.eye(3)))  # normalised

    def test_box_matrix_maps_the_unit_cube_corners(self):
        q = (math.cos(math.radians(45) / 2), 0, 0, math.sin(math.radians(45) / 2))  # 45 deg about z
        m = ir.box_matrix((1, 2, 3), (0.5, 0.2, 0.1), q)
        corner = np.array([1, 1, 1, 1.0]) @ m  # row-vector convention, as USD
        expect = ir.quat_matrix(q) @ np.array([0.5, 0.2, 0.1]) + [1, 2, 3]
        self.assertTrue(np.allclose(corner[:3], expect))
        self.assertTrue(np.allclose((np.array([0, 0, 0, 1.0]) @ m)[:3], [1, 2, 3]))

    def test_tag_quad_moves_toward_the_viewer(self):
        corners = [[-0.03, 0.0, 0.02], [0.02, 0.0, 0.02], [0.02, 0.0, -0.03], [-0.03, 0.0, -0.03]]  # TL TR BR BL
        q = ir.tag_quad(corners)
        self.assertTrue(np.allclose(q[:, 1], -ir.TAG_FRONT_OFFSET_M))  # -y = out of the shelf, toward the camera
        self.assertTrue(np.allclose(q[:, [0, 2]], np.asarray(corners)[:, [0, 2]]))

    def test_load_scene_checks_schema(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "s.json"
            p.write_text('{"schema": "something-else"}')
            with self.assertRaises(ValueError):
                ir.load_scene(p)
            p.write_text('{"schema": "%s", "cameras": []}' % ir.SCENE_SCHEMA)
            self.assertEqual(ir.load_scene(p)["cameras"], [])


if __name__ == "__main__":
    unittest.main()
