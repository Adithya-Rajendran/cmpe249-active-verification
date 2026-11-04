"""sim/alignment.py: the true query-to-reference homography from the known camera poses, and its use in sim/episodes.py
(true_alignment.json) and sim/check_batch.py (tag-free sets). numpy and cv2 only.

The tags are gone from shelf_v3 scenes; the simulator knows both camera poses, so the homography of a world plane between the
reference and a query view is exact. Checked here by projecting points of the plane through both cameras and mapping them
through H, independently of the matrix algebra in the module.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim import alignment as AL
from sim import arrange as A
from sim import check_batch as CB
from sim import episodes as E
from sim import rig as R
from tests.test_sim_batch import STOPS, _labels

REPO = Path(__file__).resolve().parents[1]
V3 = "configs/sim/arrange_shelf_v3.json"


def _spec(cam: R.Camera) -> dict:
    right, up, fwd = cam.axes()
    return {"name": cam.name, "position": list(cam.position), "right": list(right), "up": list(up), "forward": list(fwd),
            "f_px": cam.f_px, "width": cam.width, "height": cam.height}


def _project(spec: dict, pts) -> np.ndarray:
    """Pinhole projection written out from the spec's vectors (independent of plane_matrix)."""
    d = np.asarray(pts, float) - np.asarray(spec["position"], float)
    z = d @ np.asarray(spec["forward"])
    return np.c_[spec["width"] / 2 + spec["f_px"] * (d @ np.asarray(spec["right"])) / z,
                 spec["height"] / 2 - spec["f_px"] * (d @ np.asarray(spec["up"])) / z]


class TestPlaneHomography(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = A.load_context(V3)
        cls.nominal = [_spec(c) for c in cls.ctx.rig.cameras]
        rng = np.random.default_rng(5)
        cls.moved = []
        for c in cls.ctx.rig.cameras:
            d = {"dx_m": rng.uniform(-.01, .01), "dy_m": rng.uniform(-.01, .01), "dz_m": rng.uniform(-.01, .01),
                 "dyaw_deg": rng.uniform(-.5, .5), "dpitch_deg": rng.uniform(-.5, .5), "droll_deg": rng.uniform(-.5, .5)}
            cls.moved.append(_spec(A.perturb_camera(c, d)))

    def test_points_of_the_plane_map_through_h(self):
        rng = np.random.default_rng(1)
        for y0 in (0.0, 0.12, 0.25):
            for ref, qry in zip(self.nominal, self.moved):
                pts = np.c_[rng.uniform(-0.3, 0.3, 40), np.full(40, y0), rng.uniform(0.0, 0.15, 40)]
                uq, ur = _project(qry, pts), _project(ref, pts)
                h = AL.plane_homography(qry, ref, y0)
                self.assertAlmostEqual(h[2, 2], 1.0)
                self.assertLess(float(np.abs(AL.apply(h, uq) - ur).max()), 1e-6)
                back = AL.plane_homography(ref, qry, y0)
                self.assertLess(float(np.abs(AL.apply(back, ur) - uq).max()), 1e-6)
                prod = h @ back
                self.assertLess(float(np.abs(prod / prod[2, 2] - np.eye(3)).max()), 1e-9)

    def test_same_pose_is_the_identity_and_planes_differ(self):
        for cam in self.nominal:
            self.assertLess(float(np.abs(AL.plane_homography(cam, cam, 0.1) - np.eye(3)).max()), 1e-9)
        h0, h1 = (AL.plane_homography(self.moved[2], self.nominal[2], y) for y in (0.0, 0.2))
        self.assertGreater(float(np.abs(h0 - h1).max()), 1e-5)                 # parallax: the planes are aligned differently
        pt_off = np.array([[0.05, 0.2, 0.05]])                                  # a point off the front plane
        uq, ur = _project(self.moved[2], pt_off), _project(self.nominal[2], pt_off)
        self.assertGreater(float(np.abs(AL.apply(h0, uq) - ur).max()), 0.5)    # the front-plane H is not exact for it ...
        self.assertLess(float(np.abs(AL.apply(h1, uq) - ur).max()), 1e-6)      # ... the plane through it is

    def test_corner_displacement(self):
        box = [100, 80, 180, 140]
        self.assertEqual(AL.corner_displacement(np.eye(3), box)["max_px"], 0.0)
        shift = np.array([[1, 0, 5.0], [0, 1, -2.0], [0, 0, 1]])                # query -> ref moves everything by (5, -2)
        d = AL.corner_displacement(shift, box)
        self.assertAlmostEqual(d["max_px"], float(np.hypot(5, 2)), places=3)
        self.assertEqual(len(d["corners_px"]), 4)
        self.assertEqual(AL.box_corners(box).tolist(), [[100, 80], [180, 80], [180, 140], [100, 140]])

    def test_a_real_arrangement(self):
        arr = A.design_arrangement(1, ctx=self.ctx)
        lab = _labels("DES1", "design-1", "study_mug", b"x", design=True)
        al = AL.arrangement_alignment(arr, lab)
        self.assertEqual(set(al["stops"]), set(STOPS))
        self.assertAlmostEqual(al["target_plane_y_m"], arr["objects"]["target"]["pose"]["base"][1], places=5)
        for s, st in al["stops"].items():
            self.assertEqual(st["pose_delta"], arr["revisit"]["stops"][s]["delta"])
            ref = next(c for c in arr["scene"]["cameras"] if c["name"] == s)
            qry = next(c for c in arr["scene"]["revisit_cameras"] if c["name"] == s)
            pts = np.array([[x, 0.0, z] for x in (-0.3, 0.0, 0.3) for z in (0.02, 0.1)])
            self.assertLess(float(np.abs(AL.apply(np.array(st["H_front_plane"]), _project(qry, pts)) - _project(ref, pts)).max()), 1e-5)
            self.assertGreater(st["box_corner_displacement_px"]["front_plane"]["max_px"], 0.0)       # the pose moved
            self.assertGreater(st["box_corner_displacement_px"]["target_plane"]["max_px"], 0.0)


class TestGeneralPlane(unittest.TestCase):
    """The general-plane path (n . X = d) and the tabletop planes. Points are projected through both cameras by the
    independent pinhole `_project`, then mapped through H."""

    @classmethod
    def setUpClass(cls):
        from sim import tabletop as T

        cls.scene = T.load_scene(positions=T.DEFAULT_POSITIONS_PATH)
        rng = np.random.default_rng(11)
        cls.nominal, cls.moved, cls.deltas = [], [], []
        for c in cls.scene.cameras:
            d = {"dx_m": rng.uniform(-.01, .01), "dy_m": rng.uniform(-.01, .01), "dz_m": rng.uniform(-.01, .01),
                 "dyaw_deg": rng.uniform(-.5, .5), "dpitch_deg": rng.uniform(-.5, .5), "droll_deg": rng.uniform(-.5, .5)}
            cls.nominal.append(_spec(c))
            cls.moved.append(_spec(A.perturb_camera(c, d)))
            cls.deltas.append(d)

    def test_z_plane_points_map_through_h_subpixel(self):
        rng = np.random.default_rng(2)
        for z0 in (0.0, 0.03, 0.0275, 0.12):                 # the table top, and heights of target centres
            for ref, qry in zip(self.nominal, self.moved):
                pts = np.c_[rng.uniform(-0.2, 0.2, 40), rng.uniform(-0.1, 0.2, 40), np.full(40, z0)]
                uq, ur = _project(qry, pts), _project(ref, pts)
                h = AL.plane_homography_general(qry, ref, AL.Z_UP, z0)
                self.assertAlmostEqual(h[2, 2], 1.0)
                self.assertLess(float(np.abs(AL.apply(h, uq) - ur).max()), 1e-6)         # far below one pixel
                back = AL.plane_homography_general(ref, qry, AL.Z_UP, z0)
                self.assertLess(float(np.abs(AL.apply(back, ur) - uq).max()), 1e-6)

    def test_a_point_off_the_plane_moves_by_parallax(self):
        ref, qry = self.nominal[0], self.moved[0]
        h0 = AL.plane_homography_general(qry, ref, AL.Z_UP, 0.0)
        pt = np.array([[0.05, 0.05, 0.10]])                  # 10 cm above the table
        off = float(np.abs(AL.apply(h0, _project(qry, pt)) - _project(ref, pt)).max())
        self.assertGreater(off, 0.5)
        h1 = AL.plane_homography_general(qry, ref, AL.Z_UP, 0.10)
        self.assertLess(float(np.abs(AL.apply(h1, _project(qry, pt)) - _project(ref, pt)).max()), 1e-6)

    def test_general_equals_the_y_plane_path_and_is_scale_invariant(self):
        for ref, qry in zip(self.nominal, self.moved):
            for y0 in (0.0, 0.2):
                a = AL.plane_homography(qry, ref, y0)
                self.assertLess(float(np.abs(AL.plane_homography_general(qry, ref, (0, 1, 0), y0) - a).max()), 1e-9)
                self.assertLess(float(np.abs(AL.plane_homography_general(qry, ref, (0, 5, 0), 5 * y0) - a).max()), 1e-9)

    def test_a_tilted_plane(self):
        n = np.array([0.3, -0.5, 0.8])
        n /= np.linalg.norm(n)
        d = 0.04
        e1, e2 = AL._plane_basis(n)
        self.assertAlmostEqual(abs(float(e1 @ n)), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(np.cross(e1, e2))), 1.0)
        rng = np.random.default_rng(3)
        st = rng.uniform(-0.15, 0.15, (30, 2))
        pts = n * d + st[:, :1] * e1 + st[:, 1:] * e2
        ref, qry = self.nominal[1], self.moved[1]
        h = AL.plane_homography_general(qry, ref, n, d)
        self.assertLess(float(np.abs(AL.apply(h, _project(qry, pts)) - _project(ref, pts)).max()), 1e-6)
        with self.assertRaises(ValueError):
            AL.plane_matrix_general(ref, (0, 0, 0), 0.0)

    def _tabletop_arr(self, size_z=0.055, base_z=0.0):
        from sim import tabletop as T

        spec = T.export_scene_spec(self.scene)
        spec["revisit_cameras"] = [dict(c) for c in spec["cameras"]]
        stops = [c["name"] for c in spec["cameras"]]
        for c, m in zip(spec["revisit_cameras"], self.moved):
            c.update({k: m[k] for k in ("position", "right", "up", "forward")})
        return {"arrangement_id": "TD1000", "target_id": "foam_brick", "stops": stops, "scene": spec,
                "objects": {"target": {"size_m": [0.05, 0.08, size_z], "pose": {"base": [0.02, 0.05, base_z], "yaw_deg": 0.0}}},
                "revisit": {"stops": {s: {"delta": dict(d)} for s, d in zip(stops, self.deltas)}}}

    def test_tabletop_arrangement_alignment(self):
        arr = self._tabletop_arr()
        self.assertEqual(AL.environment(arr), "tabletop")
        self.assertAlmostEqual(AL.target_plane_z(arr), 0.0275)
        self.assertAlmostEqual(AL.target_plane_z(self._tabletop_arr(0.2, 0.01)), 0.11)
        box = [900, 500, 1000, 600]
        lab = {"reference_box": {"L1": {s: box for s in arr["stops"]}}}
        al = AL.arrangement_alignment(arr, lab)
        self.assertEqual((al["target"], al["target_plane_z_m"]), ("foam_brick", 0.0275))
        self.assertNotIn("target_plane_y_m", al)
        for i, (s, st) in enumerate(al["stops"].items()):
            self.assertEqual(st["stop_index"], i)
            self.assertEqual(set(st) - {"box_corner_displacement_px"},
                             {"stop_index", "pose_delta", "H_table_plane", "H_target_plane", "reference_box"})
            self.assertEqual(set(st["box_corner_displacement_px"]), {"table_plane", "target_plane"})
            ref, qry = self.nominal[i], self.moved[i]
            tab = np.c_[np.random.default_rng(i).uniform(-0.2, 0.2, (20, 2)), np.zeros(20)]
            self.assertLess(float(np.abs(AL.apply(np.array(st["H_table_plane"]), _project(qry, tab)) - _project(ref, tab)).max()), 1e-5)
            mid = tab + [0, 0, 0.0275]
            self.assertLess(float(np.abs(AL.apply(np.array(st["H_target_plane"]), _project(qry, mid)) - _project(ref, mid)).max()), 1e-5)
            self.assertGreater(st["box_corner_displacement_px"]["table_plane"]["max_px"], 0.0)
        al2 = AL.arrangement_alignment(arr, None)
        self.assertIsNone(al2["stops"]["P3"]["reference_box"])
        self.assertNotIn("box_corner_displacement_px", al2["stops"]["P3"])

    def test_tabletop_doc_names_its_planes_and_a_mixed_set_is_refused(self):
        arr = self._tabletop_arr()
        lab = {"reference_box": {"L1": {s: [900, 500, 1000, 600] for s in arr["stops"]}}}
        d = AL.doc([(arr, lab)])
        self.assertEqual((d["schema"], d["environment"]), (AL.SCHEMA, "tabletop"))
        self.assertTrue(d["simulation"] and d["evaluation_only"] and d["never_an_evidence_or_policy_input"])
        self.assertEqual(set(d["convention"]["planes"]), {"table_plane", "target_plane"})
        self.assertIn("QUERY pixel", d["convention"]["H"])
        self.assertIn("TD1000", d["arrangements"])
        json.dumps(d)
        shelf = A.design_arrangement(1, ctx=A.load_context(V3))
        self.assertEqual(AL.environment(shelf), "shelf")
        with self.assertRaises(ValueError):
            AL.doc([(arr, lab), (shelf, _labels("DES1", "design-1", "study_mug", b"x", design=True))])

    def test_shelf_document_is_unchanged(self):
        """No 'environment' key, the shelf planes and keys: the episodes golden pins the bytes; this pins the shape."""
        arr = A.design_arrangement(1, ctx=A.load_context(V3))
        d = AL.doc([(arr, _labels("DES1", "design-1", "study_mug", b"x", design=True))])
        self.assertEqual(list(d), ["schema", "simulation", "evaluation_only", "never_an_evidence_or_policy_input", "group", "note",
                                   "convention", "arrangements"])
        self.assertEqual(set(d["convention"]["planes"]), {"front_plane", "target_plane"})
        st = d["arrangements"]["DES1"]
        self.assertEqual(list(st)[:2], ["target", "target_plane_y_m"])
        self.assertEqual(set(st["stops"]["P3"]) - {"box_corner_displacement_px"},
                         {"stop_index", "pose_delta", "H_front_plane", "H_target_plane", "reference_box"})



class TestTrueAlignmentFile(unittest.TestCase):
    def _set(self, root: Path):
        ctx = A.load_context(V3)
        arr = A.design_arrangement(1, ctx=ctx)
        b = json.dumps(arr).encode()
        d = root / "DES1"
        d.mkdir(parents=True)
        (d / "arrangement.json").write_bytes(b)
        lab = _labels("DES1", "design-1", "study_mug", b, design=True, family="M16", side="R")
        (d / "labels.json").write_text(json.dumps(lab))
        return arr, lab, d

    def test_episodes_writes_it_with_the_convention_and_the_evaluation_only_marks(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            arr, lab, _ = self._set(root)
            doc = E.build(root, positions=None)
            self.assertIn("true_alignment_design.json", doc["files"])
            ta = json.loads((root / "true_alignment_design.json").read_text())
            self.assertEqual(ta["schema"], AL.SCHEMA)
            self.assertTrue(ta["simulation"] and ta["evaluation_only"] and ta["never_an_evidence_or_policy_input"])
            self.assertIn("QUERY pixel", ta["convention"]["H"])
            st = ta["arrangements"]["DES1"]["stops"]["P3"]
            self.assertEqual(np.array(st["H_front_plane"]).shape, (3, 3))
            self.assertEqual(st["stop_index"], 2)
            self.assertEqual(st["reference_box"], lab["reference_box"]["L1"]["P3"])
            self.assertEqual(set(st["pose_delta"]), {"dx_m", "dy_m", "dz_m", "dyaw_deg", "dpitch_deg", "droll_deg"})
            ann = json.loads((root / "DES1" / "annotations" / "ann_P3_L1.json").read_text())
            self.assertIn("NO corner tags", ann["note"])                                         # tag-free scene
            self.assertIn("no corner ArUco tags", (root / "README.md").read_text())

    def test_sets_without_camera_vectors_get_no_file(self):
        from tests.test_sim_batch import _write_set

        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False)])
            E.build(root, positions=None)
            self.assertFalse((root / "true_alignment.json").exists())

    def test_check_batch_skips_the_tag_checks_of_a_tag_free_set(self):
        import cv2

        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            arr, lab, d = self._set(root)
            (d / "rgb").mkdir()
            rng = np.random.default_rng(0)
            for ph in ("reference", "visible", "occluded", "removed"):
                for s in STOPS:
                    for L in ("L1", "L2"):
                        cv2.imwrite(str(d / "rgb" / f"{ph}_{s}_{L}.png"), rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
            lab.update(indistinguishable_by_masks=False, target_px={"visible": {s: 100 for s in STOPS}},
                       alone_px={"query": {s: 100 for s in STOPS}})
            (d / "labels.json").write_text(json.dumps(lab))
            (d / "render.json").write_text(json.dumps({"timings": {"total_s": 1.0, "steps": []}}))
            res = CB.check_set(root)
            sm = res["summary"]
            self.assertTrue(sm["tag_checks"].startswith("SKIPPED"))
            self.assertEqual((sm["tag_free_arrangements"], sm["n_query_views"], sm["tags_all_four"]["of"]), (1, 0, 0))
            ta = sm["true_alignment_corner_displacement_px"]
            self.assertEqual(ta["n_views"], 5)
            self.assertGreater(ta["front_plane"]["min"], 0.0)
            self.assertIsNone(sm["registration_rms_px"])
            self.assertEqual(sm["label_sanity_flagged"], {})
            self.assertEqual(sm["visible_vs_reference"]["views"], 10)                           # the raw difference is still reported
            self.assertIsNone(sm["visible_vs_reference"]["registered_mean_abs_grey_in_tag_rect"])


if __name__ == "__main__":
    unittest.main()
