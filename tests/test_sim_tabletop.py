"""sim/tabletop.py, sim/wood_texture.py, configs/sim/tabletop_geometry.json and configs/sim/positions_tabletop_A.json (numpy only).

Pins: the table / room / workspace dimensions of the design (decision 2026-11-04); the slab mesh (closed, outward, UVs in
[0, 1] with the grain along x); the fixed texture (pixel sha256 recorded in the geometry before any evidence result); the
lighting = the shelf presets' numbers relative to the workspace centre; every frame ray meets the modelled room; the scene
export in cmpe249.sim-scene.v1 with optional fields the shelf export never emits; the positions file (per-stop pitch, windows
>= 2 deg, symmetric stops, margins with the revisit offsets recomputed) and sim/rig.py's optional roll_deg (the shelf's
cameras unchanged).
"""
import contextlib
import dataclasses
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim import isaac_rig as IR
from sim import rig as R
from sim import tabletop as T
from sim import wood_texture as WT

G = T.load_geometry()
POS_PATH = T.DEFAULT_POSITIONS_PATH
SHELF_POS = R.REPO_ROOT / "configs" / "sim" / "positions_shelf_A.json"
SHELF_POS_SHA256 = "695dbd3339039679df03755c2613ddb84b9c09906b22763c363b6219e0190dcf"     # recorded by every shelf_v3 arrangement


class TestGeometry(unittest.TestCase):
    def test_schema_is_checked(self):
        with self.assertRaises(ValueError):
            T.load_geometry({**G, "schema": "cmpe249.rig_geometry.v1"})

    def test_table_and_room_dimensions(self):
        e = T.table_extent(G)
        self.assertEqual((e["x_min"], e["x_max"]), (-0.60, 0.60))
        self.assertAlmostEqual(e["y_min"], -0.30)
        self.assertAlmostEqual(e["y_max"], 0.50)
        self.assertEqual((e["z_top"], e["z_bottom"]), (0.0, -0.03))
        self.assertAlmostEqual(T.floor_z(G), -0.75)
        self.assertAlmostEqual(T.wall_y(G), 1.50)                         # 1 m behind the far edge
        self.assertFalse(G["room"]["ceiling"])
        for leg in T.table_legs(G):
            lo, hi = leg.bounds()
            self.assertAlmostEqual(lo[2], -0.75)
            self.assertAlmostEqual(hi[2], -0.03)
            self.assertTrue(e["x_min"] < lo[0] < hi[0] < e["x_max"] and e["y_min"] < lo[1] < hi[1] < e["y_max"], leg.name)
        boxes = {b.name: b.bounds() for b in T.room_boxes(G)}
        self.assertAlmostEqual(boxes["floor"][1][2], -0.75)               # floor top = table legs' feet
        self.assertAlmostEqual(boxes["wall_back"][0][1], 1.50)
        self.assertGreater(boxes["wall_back"][1][2] - (-0.75), 2.0)

    def test_camera_intrinsics_are_the_shelf_cameras(self):
        rc = R.load_geometry()["camera"]
        for k in ("f_px", "width_px", "height_px", "k1"):
            self.assertEqual(G["camera"][k], rc[k], k)
        self.assertEqual(G["visibility_classes"]["H_max"], R.load_geometry()["visibility_classes"]["H_max"])
        self.assertEqual(G["visibility_classes"]["C_min"], R.load_geometry()["visibility_classes"]["C_min"])

    def test_lighting_is_the_shelf_presets(self):
        L = G["lighting"]
        self.assertEqual(L["aim"], [0.0, 0.0, 0.05])
        self.assertEqual(set(L["presets"]), set(IR.LIGHTING))
        for k, (dome, lights) in IR.LIGHTING.items():
            p = L["presets"][k]
            self.assertEqual(p["dome_intensity"], dome)
            self.assertEqual([(lt["name"], tuple(lt["position"]), lt["size_m"], lt["intensity"], lt["on"]) for lt in p["lights"]],
                             [(n, tuple(pos), s, i, on) for n, pos, s, i, on in lights])
        self.assertEqual([lt["on"] for lt in L["presets"]["L2"]["lights"]], [True, False])     # L2 = key_right off
        spec = T.lighting_spec(G)
        for p in spec["presets"].values():
            for lt in p["lights"]:
                d = np.asarray(spec["aim"]) - np.asarray(lt["position"])
                np.testing.assert_allclose(lt["forward"], d / np.linalg.norm(d), atol=1e-8)

    def test_floor_colour_is_the_shelf_render_floor(self):
        rgb, rough = IR.MATERIALS["floor"]
        self.assertEqual(tuple(G["materials"]["floor"]["rgba"][:3]), rgb)
        self.assertEqual(G["materials"]["floor"]["roughness"], rough)
        self.assertEqual(G["materials"]["table_top"]["roughness"], 0.5)

    def test_workspace_outline(self):
        pts = T.workspace_outline(G)
        self.assertEqual(pts.shape, (48, 3))
        np.testing.assert_allclose(np.hypot(pts[:, 0], pts[:, 1]), 0.20)
        self.assertEqual(sorted(set(np.round(pts[:, 2], 9))), [0.0, 0.12])


class TestTableTopMesh(unittest.TestCase):
    def setUp(self):
        self.m = T.table_top_mesh(G)

    def test_closed_slab_facing_out(self):
        st = R.mesh_stats(self.m.vertices, self.m.faces)
        self.assertEqual((st["n_vertices"], st["n_faces"]), (24, 12))
        self.assertAlmostEqual(st["volume"], 1.20 * 0.80 * 0.03, places=12)   # positive: every face winds outward
        for f in self.m.faces:
            a, b, c = self.m.vertices[f]
            n = np.cross(b - a, c - a)
            np.testing.assert_allclose(n / np.linalg.norm(n), self.m.normals[f[0]], atol=1e-12)

    def test_uvs_cover_the_top_once_with_the_grain_along_x(self):
        e = T.table_extent(G)
        self.assertTrue(np.all(self.m.uv >= 0) and np.all(self.m.uv <= 1))
        top = np.isclose(self.m.vertices[:, 2], 0.0) & np.isclose(self.m.normals[:, 2], 1.0)
        self.assertEqual(int(top.sum()), 4)
        np.testing.assert_allclose(self.m.uv[top, 0], (self.m.vertices[top, 0] - e["x_min"]) / 1.20)     # u (image width) along x
        np.testing.assert_allclose(self.m.uv[top, 1], (self.m.vertices[top, 1] - e["y_min"]) / 0.80)
        self.assertEqual(sorted(map(tuple, np.round(self.m.uv[top], 9))), [(0, 0), (0, 1), (1, 0), (1, 1)])
        tex = G["materials"]["table_top"]["texture"]
        self.assertEqual(tex["physical_size_m"], [1.20, 0.80])
        self.assertEqual(tex["grain_orientation"].split()[1], "+x")

    def test_side_strips_keep_the_physical_scale(self):
        for n, extent in (((0, -1, 0), 0.03 / 0.80), ((0, 1, 0), 0.03 / 0.80), ((-1, 0, 0), 0.03 / 1.20), ((1, 0, 0), 0.03 / 1.20)):
            sel = np.all(np.isclose(self.m.normals, n), axis=1)
            k = 1 if n[1] else 0                                            # the UV axis that spans the slab's thickness
            self.assertAlmostEqual(float(np.ptp(self.m.uv[sel, k])), extent, places=12)


class TestTexture(unittest.TestCase):
    def test_fixed_texture_pixels(self):
        tex = G["materials"]["table_top"]["texture"]
        self.assertEqual(WT.PARAMS["name"], tex["name"])
        self.assertEqual(WT.PARAMS["size_m"], tex["physical_size_m"])
        self.assertEqual(WT.PARAMS["m_per_px"], tex["m_per_px"])
        self.assertEqual(WT.PARAMS["grain_axis"], "x")
        img = WT.generate()
        self.assertEqual(img.shape, (tex["height_px"], tex["width_px"], 3))
        self.assertEqual(WT.pixels_sha256(img), tex["pixels_sha256"])
        mean = img.reshape(-1, 3).mean(0) / 255
        np.testing.assert_allclose(G["materials"]["table_top"]["rgba"][:3], mean, atol=0.002)
        # grain along x: neighbouring pixels differ much less along a row than down a column
        a = img.astype(float).mean(2)
        self.assertLess(np.abs(np.diff(a, axis=1)).mean() * 3, np.abs(np.diff(a, axis=0)).mean())

    def test_png_round_trip_and_determinism(self):
        small = {**WT.PARAMS, "size_m": [0.06, 0.04]}
        a, b = WT.generate(small), WT.generate(small)
        self.assertEqual(WT.pixels_sha256(a), WT.pixels_sha256(b))
        with tempfile.TemporaryDirectory() as t:
            info = WT.write(Path(t) / "x.png", small)
            np.testing.assert_array_equal(WT.read_png_pixels(Path(t) / "x.png"), a)
            self.assertEqual(info["pixels_sha256"], WT.pixels_sha256(a))
            self.assertEqual(WT.write(Path(t) / "y.png", small)["file_sha256"], info["file_sha256"])
        self.assertNotEqual(WT.pixels_sha256(WT.generate({**small, "seed": 1})), WT.pixels_sha256(a))


class TestSceneExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = T.load_scene(T.DEFAULT_GEOMETRY_PATH, POS_PATH)
        cls.spec = T.export_scene_spec(cls.scene)

    def test_schema_and_parts(self):
        s = json.loads(json.dumps(self.spec))
        self.assertEqual(s["schema"], R.SCENE_SCHEMA)
        self.assertEqual(s["environment"], "tabletop")
        self.assertEqual((s["tags"], s["tags_enabled"], s["mug"], s["occluder"]), ([], False, None, None))
        self.assertEqual(s["labels"], ["table", "room"])
        for b in s["boxes"]:
            self.assertIn(b["material"], s["materials"])
            self.assertIn(b["label"], s["labels"])
        (m,) = s["meshes"]
        self.assertEqual((m["name"], m["material"], len(m["vertices"]), len(m["uv"]), len(m["normals"])), ("table_top", "table_top", 24, 24, 24))
        self.assertEqual(s["materials"]["table_top"]["texture"]["pixels_sha256"], G["materials"]["table_top"]["texture"]["pixels_sha256"])
        self.assertEqual([c["name"] for c in s["cameras"]], ["P1", "P2", "P3", "P4", "P5"])
        self.assertEqual(len(s["workspace"]["outline"]), 48)
        self.assertEqual(set(s["lighting"]["presets"]), {"L1", "L2"})

    def test_cameras_in_the_shelf_export_format(self):
        shelf = R.export_scene_spec(R.load_rig(positions_json=SHELF_POS))
        self.assertEqual(T.camera_spec(R.load_rig(positions_json=SHELF_POS).cameras[0]), shelf["cameras"][0])
        self.assertEqual(set(self.spec["cameras"][0]), set(shelf["cameras"][0]))

    def test_the_shelf_export_has_none_of_the_optional_fields(self):
        shelf = R.export_scene_spec(R.load_rig(positions_json=SHELF_POS))
        for k in ("environment", "lighting", "workspace", "tags_enabled"):
            self.assertNotIn(k, shelf)
        self.assertFalse(any("uv" in m or "normals" in m for m in shelf["meshes"]))
        self.assertFalse(any("texture" in m or "metallic" in m for m in shelf["materials"].values()))

    def test_cli_writes_the_spec(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            T.main(["scene", "--scene-out", str(Path(t) / "s.json")])
            self.assertEqual(json.loads((Path(t) / "s.json").read_text()), json.loads(json.dumps(self.spec)))

    def test_every_frame_ray_meets_the_room(self):
        """No frame sees past the room (no ceiling, so every ray must point down onto the floor, a wall or the table), at the
        nominal poses and at the extreme revisit offsets."""
        rc = T.revisit_ranges()
        tr, rot = rc["translation_m"], rc["rotation_deg"]
        from sim import arrange as A

        for cam in self.scene.cameras:
            for sgn in (0, -1, 1):
                d = {"dx_m": sgn * tr, "dy_m": sgn * tr, "dz_m": sgn * tr, "dyaw_deg": sgn * rot, "dpitch_deg": -sgn * rot, "droll_deg": sgn * rot}
                c = A.perturb_camera(cam, d)
                r = T.frame_rays(self.scene, c)
                with self.subTest(stop=cam.name, offset=sgn):
                    self.assertEqual(r["missed"], [])
                    self.assertIn("table_top", r["parts"])


class TestPositions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pos = json.loads(POS_PATH.read_text())
        cls.cams = T.cameras(G, cls.pos)

    def test_plan(self):
        p = self.pos
        self.assertEqual(p["schema"], R.POSITIONS_SCHEMA)
        self.assertEqual((p["radius_cm"], p["height_cm"], p["arc_centre_behind_front_cm"], p["arc_rotation_deg"], p["roll_deg"]),
                         (45.0, 38.0, 0.0, 0.0, 0.0))
        self.assertEqual([s["azimuth_deg"] for s in p["stops"]], [-60.0, -30.0, 0.0, 30.0, 60.0])
        self.assertEqual((p["start_stop"], p["f_order"]), ("P3", ["P1", "P2", "P4", "P5"]))
        self.assertNotIn("tilt_deg", p)
        for c in self.cams:
            self.assertEqual(c.roll_deg, 0.0)
            self.assertAlmostEqual(c.position[2], 0.38)
            self.assertAlmostEqual(math.hypot(c.position[0], c.position[1]), 0.45)
            fwd = c.axes()[2]
            np.testing.assert_allclose(fwd[:2] / np.linalg.norm(fwd[:2]), -np.asarray(c.position[:2]) / 0.45, atol=1e-12)   # faces the centre

    def test_windows_at_least_2_deg_and_margins_recompute(self):
        for c in self.cams:
            fc = self.pos["frame_check"][c.name]
            with self.subTest(stop=c.name):
                self.assertGreaterEqual(fc["k1_window_width_deg"], T.MIN_WINDOW_DEG)
                lo, hi = fc["k1_window_deg"]
                self.assertTrue(lo < c.pitch_down_deg < hi)
                self.assertGreaterEqual(T.outline_margin(G, dataclasses.replace(c, pitch_down_deg=lo)), 0)
                self.assertLess(T.outline_margin(G, dataclasses.replace(c, pitch_down_deg=lo - 0.01)), 0)
                self.assertGreaterEqual(T.outline_margin(G, dataclasses.replace(c, pitch_down_deg=hi)), 0)
                self.assertLess(T.outline_margin(G, dataclasses.replace(c, pitch_down_deg=hi + 0.01)), 0)
                self.assertAlmostEqual(T.outline_margin(G, c), fc["k1_min_margin_px"], places=2)
                self.assertAlmostEqual(T.outline_margin(G, c, False), fc["pinhole_min_margin_px"], places=2)
                self.assertTrue(36.0 <= c.pitch_down_deg <= 42.0)              # the design's expected window
        self.assertTrue(self.pos["acceptance"]["ok"])

    def test_stops_are_symmetric(self):
        by = {c.name: c for c in self.cams}
        for a, b in (("P1", "P5"), ("P2", "P4")):
            ca, cb = by[a], by[b]
            self.assertEqual(ca.pitch_down_deg, cb.pitch_down_deg)
            self.assertAlmostEqual(ca.position[0], -cb.position[0], places=12)
            self.assertAlmostEqual(ca.position[1], cb.position[1], places=12)
            self.assertAlmostEqual(ca.heading_deg, -cb.heading_deg, places=12)
            fa, fb = self.pos["frame_check"][a], self.pos["frame_check"][b]
            self.assertEqual(fa["k1_window_deg"], fb["k1_window_deg"])
            self.assertAlmostEqual(fa["k1_min_margin_px"], fb["k1_min_margin_px"], places=2)

    def test_revisit_corners_recompute(self):
        rc = T.revisit_ranges()
        self.assertEqual((rc["translation_m"], rc["rotation_deg"]), (0.01, 0.5))
        for i, c in enumerate(self.cams):
            rv = T.revisit_check(G, c, i, rc, n_sample=50)
            fc = self.pos["frame_check"][c.name]["revisit"]
            with self.subTest(stop=c.name):
                self.assertEqual(rv["corners_worst_margin_px"], fc["corners_worst_margin_px"])
                self.assertEqual(rv["corners_worst_delta"], fc["corners_worst_delta"])
                self.assertGreaterEqual(fc["corners_worst_margin_px"], rc["min_margin_px"])
                self.assertGreaterEqual(fc["sample_min_margin_px"], fc["corners_worst_margin_px"] - 1.0)   # corners bound the sample
                self.assertEqual(fc["expected_redraw_rate"], 0.0)

    def test_the_file_is_the_generators_output(self):
        """make_positions reproduces the committed file (a reduced revisit sample: the sample fields are compared to a
        fresh run with the same reduced size, everything else must be identical)."""
        out, _ = T.make_positions(n_sample=20)
        ref = json.loads(POS_PATH.read_text())
        strip = lambda d: {k: v for k, v in d.items() if not k.startswith("sample_") and k != "expected_redraw_rate"}
        for name in ref["frame_check"]:
            out["frame_check"][name]["revisit"] = strip(out["frame_check"][name]["revisit"])
            ref["frame_check"][name]["revisit"] = strip(ref["frame_check"][name]["revisit"])
        out["revisit_check"]["method"] = ref["revisit_check"]["method"] = None
        self.assertEqual(json.loads(json.dumps(out)), ref)


class TestRollFromPositions(unittest.TestCase):
    def test_shelf_positions_unaffected(self):
        self.assertEqual(T._sha256_file(SHELF_POS), SHELF_POS_SHA256)
        shelf = json.loads(SHELF_POS.read_text())
        self.assertNotIn("roll_deg", shelf)
        rig = R.load_rig(positions_json=SHELF_POS)
        roll = R.load_geometry()["camera"]["p3_fit"]["roll_deg"]
        self.assertEqual([c.roll_deg for c in rig.cameras], [roll] * 5)
        self.assertEqual([c.pitch_down_deg for c in rig.cameras], [s["tilt_deg"] for s in shelf["stops"]])

    def test_roll_deg_precedence(self):
        g = R.load_geometry()
        shelf = json.loads(SHELF_POS.read_text())
        self.assertEqual(R.cameras_from_positions({**shelf, "roll_deg": 0.25}, g)[0].roll_deg, 0.25)        # the file's roll
        self.assertEqual(R.cameras_from_positions({**shelf, "roll_deg": 0.25}, g, 0.5)[0].roll_deg, 0.5)    # the argument wins
        self.assertEqual(R.cameras_from_positions(shelf, g, 0.5)[0].roll_deg, 0.5)
        with self.assertRaises(ValueError):
            T.cameras(G, {k: v for k, v in self.tabletop().items() if k != "roll_deg"})

    @staticmethod
    def tabletop():
        return json.loads(POS_PATH.read_text())


if __name__ == "__main__":
    unittest.main()
