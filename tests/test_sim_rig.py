"""sim/rig.py: simulator-free rig geometry (numpy only).

Reference numbers are copied from data/raw (gitignored, so a clean checkout has no copy):
  DRAFTS     data/raw/2026-10-10_far-camera/positions_far_{A,B20,B15}_draft.json (arc parameters + floor_coords_cm)
  SELFTEST   data/raw/2026-10-10_far-camera/geom_far_selftest.txt: the analytic far-pose model's projection of the #3
             opening corners at the fitted P3 pose (pose_fit_far.json k1), which matched pixel_budget.json to 0.22 px.
If the draft files are present (the working copy), they are checked directly as well.
"""
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim import rig as R

FAR = R.REPO_ROOT / "data" / "raw" / "2026-10-10_far-camera"


def draft(radius, centre, rot, step, floor):
    return {"schema": "cmpe249.positions.v1", "radius_cm": radius, "arc_centre_behind_front_cm": centre,
            "arc_rotation_deg": rot, "height_cm": 40.78, "tilt_deg": 6.83,
            "stops": [{"name": f"P{i + 1}", "azimuth_deg": (i - 2) * step} for i in range(5)], "floor_coords_cm": floor}


# (lateral_cm, out_from_front_cm, heading_deg_right) per stop, from the drafts' floor_coords_cm
DRAFTS = {
    "A": draft(126.8, 10.89, -2.36, 22.5, {"P1": (-93.27, 75.0, 47.36), "P2": (-53.3, 104.16, 24.86), "P3": (-5.22, 115.8, 2.36),
                                           "P4": (43.66, 108.15, -20.14), "P5": (85.89, 82.38, -42.64)}),
    "B20": draft(126.8, 10.89, -2.36, 20.0, {"P1": (-85.43, 82.8, 42.36), "P2": (-48.23, 106.37, 22.36), "P3": (-5.22, 115.8, 2.36),
                                             "P4": (38.43, 109.94, -17.64), "P5": (77.44, 89.51, -37.64)}),
    "B15": draft(126.8, 10.89, -2.36, 15.0, {"P1": (-67.86, 96.22, 32.36), "P2": (-37.83, 110.13, 17.36), "P3": (-5.22, 115.8, 2.36),
                                             "P4": (27.75, 112.83, -12.64), "P5": (58.82, 101.44, -27.64)}),
}
# geom_far_selftest.txt 'model' column: (sign x, depth 0/1, height 0/1) -> (u, v) px, f 1731.26, k1, P3 as fitted
SELFTEST = {(-1, 0, 0): (695.1, 1050.0), (1, 0, 0): (1848.8, 1018.0), (-1, 0, 1): (680.2, 815.2), (1, 0, 1): (1850.6, 779.2),
            (-1, 1, 0): (792.0, 926.2), (1, 1, 0): (1709.1, 899.2), (-1, 1, 1): (781.1, 738.4), (1, 1, 1): (1709.3, 708.9)}


class TestGeometryFile(unittest.TestCase):
    def setUp(self):
        self.g = R.load_geometry()

    def test_schema_is_checked(self):
        self.assertEqual(self.g["schema"], R.GEOMETRY_SCHEMA)
        with self.assertRaises(ValueError):
            R.load_geometry({**self.g, "schema": "bogus"})

    def test_every_block_has_provenance(self):
        for block in ("shelf", "camera", "tags", "mug", "occluders", "visibility_classes"):
            with self.subTest(block=block):
                self.assertTrue(self.g[block]["provenance"])

    def test_shelf_parts_match_the_measured_opening(self):
        s = self.g["shelf"]
        hw, d, h3 = s["c3_inner"]["half_width"], s["c3_inner"]["depth"], s["c3_inner"]["clear_height"]
        self.assertEqual((hw, d, h3), (0.395, 0.305, 0.165))     # geom_measured.py HALF_W, DEPTH, CLEAR (cm / 100)
        p = {b.name: b.bounds() for b in R.shelf_boxes(self.g)}
        self.assertAlmostEqual(p["c3_floor_board"][1][2], 0.0)             # mug-shelf top is z = 0
        self.assertAlmostEqual(p["mid_board"][0][2], h3)                   # #3 ceiling
        self.assertAlmostEqual(p["side_left"][1][0], -hw)                  # inner faces bound the opening
        self.assertAlmostEqual(p["side_right"][0][0], hw)
        self.assertAlmostEqual(p["back_panel"][0][1], d)                   # back wall at the inner depth
        self.assertAlmostEqual(p["c4_top_board"][0][2], self.g["shelf"]["c4_inner"]["z_max"])
        for name, (lo, _) in p.items():
            with self.subTest(part=name):
                self.assertAlmostEqual(lo[1], 0.0 if name != "back_panel" else d)   # flush with the front plane
        self.assertAlmostEqual(s["board_thickness"], 0.018727, places=6)   # pose_fit_far.json k1 board_t_cm 1.8727
        self.assertAlmostEqual(s["unit_half_outer"], hw + s["side_thickness"], places=6)

    def test_occluder_catalogue(self):
        names = R.occluder_names(self.g)
        self.assertEqual(names, ["pick_S8", "pick_M16+3", "pick_M16-5", "pick_IND28",
                                 "plan_S8", "plan_M18+5", "plan_M18-5", "plan_IND26"])
        o = R.occluder_spec(self.g, "pick_M16-5")
        self.assertEqual((o.width, o.y_back, o.shift, o.height, o.thickness, o.intended_pattern),
                         (0.16, 0.04, -0.05, 0.12, 0.005, "CHHCC"))
        self.assertEqual(R.occluder_spec(self.g, "plan_IND26").y_back, 0.18)   # 6 cm in front of the z-24 cup axis
        with self.assertRaises(KeyError):
            R.occluder_spec(self.g, "S9")


class TestCameras(unittest.TestCase):
    def test_draft_cameras_match_floor_coords(self):
        g = R.load_geometry()
        sources = dict(DRAFTS)
        for key in DRAFTS:
            f = FAR / f"positions_far_{key}_draft.json"
            if f.is_file():
                sources[f"{key} (file)"] = json.loads(f.read_text())
        for key, pos in sources.items():
            cams = R.cameras_from_positions(pos, g)
            self.assertEqual([c.name for c in cams], ["P1", "P2", "P3", "P4", "P5"])
            for c in cams:
                fc = pos["floor_coords_cm"][c.name]
                lat, out, head = (fc["lateral_cm"], fc["out_from_front_cm"], fc["heading_deg_right"]) if isinstance(fc, dict) else fc
                with self.subTest(draft=key, stop=c.name):
                    # drafts round R to 0.1 cm and the rotation to 0.01 deg; floor coords came from unrounded values
                    self.assertAlmostEqual(c.position[0] * 100, lat, delta=0.02)
                    self.assertAlmostEqual(-c.position[1] * 100, out, delta=0.02)
                    self.assertAlmostEqual(c.heading_deg, head, places=6)
                    self.assertAlmostEqual(c.position[2], 0.4078)
                    self.assertAlmostEqual(c.pitch_down_deg, 6.83)
                    self.assertAlmostEqual(c.roll_deg, g["camera"]["p3_fit"]["roll_deg"])
                    _, _, fwd = c.axes()                              # each stop faces the arc centre
                    to_centre = np.array([0.0, pos["arc_centre_behind_front_cm"] / 100]) - np.array(c.position[:2])
                    cosang = fwd[:2] @ to_centre / np.linalg.norm(fwd[:2]) / np.linalg.norm(to_centre)
                    self.assertAlmostEqual(cosang, 1.0, places=9)

    def test_p3_fit_reproduces_the_analytic_projection(self):
        """Frame mapping, heading / pitch / roll signs and k1 together: the sim camera at the fitted P3 pose must put
        the #3 opening corners where geom_far.py did (0.1 px rounding in the reference)."""
        g = R.load_geometry()
        cam = R.p3_fit_camera(g)
        c3 = g["shelf"]["c3_inner"]
        pts = np.array([(sx * c3["half_width"], sd * c3["depth"], sh * c3["clear_height"]) for sx, sd, sh in SELFTEST])
        uv = cam.project(pts, distort=True)
        self.assertLess(np.abs(uv - np.array(list(SELFTEST.values()))).max(), 0.1)
        self.assertGreater(np.abs(cam.project(pts) - uv).max(), 5)           # k1 moves these corners by up to 7.9 px

    def test_projection_conventions(self):
        cam = R.Camera("t", (0.0, -1.0, 0.0), 0.0, 0.0, 0.0, 1731.26, 2304, 1296)
        self.assertAlmostEqual(cam.fovy_deg, 41.04, places=2)                 # Camera Module 3 at 2304x1296
        np.testing.assert_allclose(cam.project([[0, 0, 0]])[0], (1152, 648))
        u, v = cam.project([[0.1, 0, 0.1]])[0]
        self.assertGreater(u, 1152)                                            # +x is right in the image
        self.assertLess(v, 648)                                                # +z is up
        right, up, fwd = cam.axes()
        np.testing.assert_allclose(np.cross(right, up), -fwd, atol=1e-12)     # MuJoCo: camera looks along -z
        rolled = R.Camera("r", (0.0, -1.0, 0.0), 0.0, 0.0, 5.0, 1731.26, 2304, 1296)
        self.assertLess(rolled.project([[0.2, 0, 0]])[0][1], 648)              # + roll: content turns counter-clockwise
        down = R.Camera("d", (0.0, -1.0, 0.0), 0.0, 10.0, 0.0, 1731.26, 2304, 1296)
        self.assertLess(down.project([[0, 0, 0]])[0][1], 648)                  # looking down moves content up
        with self.assertRaises(ValueError):
            cam.project([[0, -2, 0]])                                          # behind the camera


class TestMugMesh(unittest.TestCase):
    def setUp(self):
        self.g = R.load_geometry()
        self.mug = R.mug_spec(self.g, mug_z=0.24)

    def test_body_is_closed_and_matches_the_profile(self):
        z, r = np.array(self.mug.profile_z), np.array(self.mug.profile_r)
        V, F = R.revolve_profile(z, r, 96)
        st = R.mesh_stats(V, F)
        self.assertEqual((st["boundary_edges"], st["nonmanifold_edges"], st["repeated_directed_edges"], st["euler"]),
                         (0, 0, 0, 2))
        # volume of the piecewise-linear solid of revolution (frustums), times the 96-gon area factor
        frustum = sum(math.pi * (z1 - z0) * (r0 * r0 + r0 * r1 + r1 * r1) / 3
                      for z0, z1, r0, r1 in zip(z[:-1], z[1:], r[:-1], r[1:]))
        self.assertAlmostEqual(st["volume"] / (frustum * 96 / (2 * math.pi) * math.sin(2 * math.pi / 96)), 1.0, places=9)
        rad = np.hypot(V[:, 0], V[:, 1])
        self.assertAlmostEqual(V[:, 2].min(), 0.0)
        self.assertAlmostEqual(V[:, 2].max(), self.g["mug"]["rim_height"])          # 5.669 cm (mug_from_frame.json)
        self.assertAlmostEqual(rad.max(), self.g["mug"]["max_radius"])              # 3.859 cm, 'frame' profile max
        for zi, ri in zip(z, r):
            ring = np.isclose(V[:, 2], zi) & (rad > 1e-9)
            with self.subTest(z=zi):
                np.testing.assert_allclose(rad[ring], ri, rtol=1e-12)
        self.assertAlmostEqual(r[0], 0.02069, places=5)     # foot ~ the student's 4 cm base
        self.assertAlmostEqual(r[-1], 0.03449, places=5)    # fitted rim radius (student tape 3.5)

    def test_handle_plate(self):
        h = self.g["mug"]["handle"]
        V, F = R.handle_plate(self.mug.profile_z, self.mug.profile_r, h["z_range"], h["protrusion"], h["thickness"],
                              h["directions"]["toward"], h["embed"])
        st = R.mesh_stats(V, F)
        self.assertEqual((st["boundary_edges"], st["nonmanifold_edges"], st["repeated_directed_edges"], st["euler"]),
                         (0, 0, 0, 2))
        self.assertGreater(st["volume"], 0)
        np.testing.assert_allclose([V[:, 2].min(), V[:, 2].max()], h["z_range"])
        np.testing.assert_allclose([V[:, 0].min(), V[:, 0].max()], [-h["thickness"] / 2, h["thickness"] / 2])
        self.assertTrue(np.all(V[:, 1] < 0))                                     # 'toward' = toward the camera (-y)
        r_at = np.interp(V[:, 2], self.mug.profile_z, self.mug.profile_r)       # body radius at each vertex height
        np.testing.assert_allclose(np.unique(np.round(-V[:, 1] - r_at, 9)), [-h["embed"], h["protrusion"]], atol=1e-9)

    def test_meshes_are_placed_at_the_mug_axis(self):
        ms = self.mug.meshes()
        self.assertEqual([m.name for m in ms], ["mug_body", "mug_handle"])
        Wv = ms[0].world_vertices()
        np.testing.assert_allclose(Wv[:, :2].mean(0), (0.0, 0.24), atol=1e-12)
        self.assertEqual(len(R.mug_spec(self.g, handle=None).meshes()), 1)
        with self.assertRaises(ValueError):
            R.mug_spec(self.g, handle="up")


class TestRigAndTags(unittest.TestCase):
    def test_load_rig_and_occluder_box(self):
        rig = R.load_rig(R.DEFAULT_GEOMETRY_PATH, DRAFTS["A"], occluder="pick_S8")
        self.assertEqual(len(rig.cameras), 5)
        b = rig.occluder_box()
        lo, hi = b.bounds()
        np.testing.assert_allclose(lo, (-0.05, 0.015, 0.0), atol=1e-12)        # shift -1 cm, w 8, back face z 2
        np.testing.assert_allclose(hi, (0.03, 0.02, 0.12), atol=1e-12)
        moved = rig.with_mug(x=0.05).occluder_box().bounds()[0]
        self.assertAlmostEqual(moved[0], 0.0)                                  # follows the mug laterally
        p = rig.occluder.perturbed(0.01, -0.01)
        self.assertAlmostEqual(p.shift, 0.0)
        self.assertAlmostEqual(p.y_back, 0.01)
        self.assertIsNone(rig.with_occluder(None).occluder_box())
        self.assertEqual(rig.labels()[:3], ["shelf", "mug", "occluder"])
        self.assertEqual(len(rig.c3_outline()), 12)

    def test_tag_corners(self):
        g = R.load_geometry()
        tags = R.tags_from_geometry(g)
        self.assertEqual([t.id for t in tags], [0, 1, 2, 3])
        for t in tags:
            c = t.corners()
            with self.subTest(tag=t.id):
                np.testing.assert_allclose(np.linalg.norm(c - np.roll(c, -1, 0), axis=1), 0.05)
                np.testing.assert_allclose(c[:, 1], 0.0)                       # in the front plane
                self.assertLess(c[0, 0], c[1, 0])                              # TL left of TR
                self.assertEqual(c[1, 2] > c[0, 2], t.inplane_rot_deg > 0)     # + rotation: TR above TL (CCW)
        self.assertLess(tags[0].centre[2], -g["shelf"]["board_thickness"])     # 0 / 1 below the #3 floor board
        self.assertGreater(tags[2].centre[2], g["shelf"]["c4_inner"]["z_max"])  # 2 / 3 at the top of #4


class TestLabelHelpers(unittest.TestCase):
    def test_visible_fraction_and_classes(self):
        alone = np.full((10, 10), -1)
        alone[2:8, 2:8] = 1                                                    # 36 target pixels
        full = alone.copy()
        full[2:8, 2:5] = 2                                                     # half covered by label 2
        self.assertAlmostEqual(R.visible_fraction(full, alone, 1), 0.5)
        self.assertEqual(R.visible_fraction(full, alone, 1), 18 / 36)
        self.assertIsNone(R.visible_fraction(full, alone, 7))
        self.assertEqual([R.classify(v) for v in (0.0, 0.05, 0.0501, 0.8999, 0.90, 1.0, None)],
                         ["H", "H", "p", "p", "C", "C", "?"])
        self.assertEqual(R.pattern([1.0, 0.95, 0.0, 0.5, 0.9]), "CCHpC")
        self.assertEqual(R.label_bbox(full, 1), (5, 2, 7, 7))
        self.assertIsNone(R.label_bbox(full, 9))
        self.assertEqual(R.bbox_margin_px((5, 2, 7, 7), 10, 10), 2)


class TestSceneSpec(unittest.TestCase):
    """export_scene_spec: the JSON hand-off to backends that cannot import sim/rig.py (Isaac Sim, Python 3.12)."""

    def test_export_round_trips(self):
        rig = R.load_rig(R.DEFAULT_GEOMETRY_PATH, DRAFTS["A"], occluder="pick_S8")
        spec = json.loads(json.dumps(R.export_scene_spec(rig)))
        self.assertEqual(spec["schema"], "cmpe249.sim-scene.v1")
        self.assertEqual([c["name"] for c in spec["cameras"]], ["P1", "P2", "P3", "P4", "P5"])
        for c in spec["cameras"]:
            f, u, r = (np.array(c[k]) for k in ("forward", "up", "right"))
            with self.subTest(camera=c["name"]):
                for v in (f, u, r):
                    self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=7)
                self.assertAlmostEqual(float(f @ u), 0.0, places=7)
                np.testing.assert_allclose(np.cross(f, u), r, atol=1e-7)       # right = forward x up
                self.assertEqual((c["f_px"], c["width"], c["height"]), (1731.26, 2304, 1296))
        roles = [b["role"] for b in spec["boxes"]]
        self.assertEqual((roles.count("shelf"), roles.count("tag_card"), roles.count("occluder")), (6, 4, 1))
        self.assertEqual([len(t["corners"]) for t in spec["tags"]], [4, 4, 4, 4])
        self.assertEqual({t["dictionary"] for t in spec["tags"]}, {"DICT_4X4_50"})
        for m in spec["meshes"]:
            with self.subTest(mesh=m["name"]):
                self.assertEqual(m["role"], "target")
                self.assertEqual(max(max(f) for f in m["faces"]) + 1, len(m["vertices"]))
        body = np.array(spec["meshes"][0]["vertices"])
        np.testing.assert_allclose(body[:, :2].mean(0), (0.0, 0.24), atol=1e-9)      # world coordinates
        occ = next(b for b in spec["boxes"] if b["role"] == "occluder")
        np.testing.assert_allclose(occ["center"], (-0.01, 0.0175, 0.06), atol=1e-9)
        np.testing.assert_allclose(occ["half"], (0.04, 0.0025, 0.06), atol=1e-9)
        self.assertTrue(spec["provenance"]["boxes (shelf)"] and spec["provenance"]["tags"])

    def test_cli_writes_the_spec(self):
        with tempfile.TemporaryDirectory() as d:
            pos, out = Path(d) / "positions.json", Path(d) / "scene.json"
            pos.write_text(json.dumps(DRAFTS["A"]))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(R.main(["--positions", str(pos), "--occluder", "pick_M16-5", "--scene-out", str(out)]), 0)
            spec = json.loads(out.read_text())
            self.assertEqual(spec["occluder"]["name"], "pick_M16-5")
            self.assertEqual(spec["provenance"]["positions_file"], str(pos))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                R.main(["--occluder", "S9", "--scene-out", str(out)])


if __name__ == "__main__":
    unittest.main()
