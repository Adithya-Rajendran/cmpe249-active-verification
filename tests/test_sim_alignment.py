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
