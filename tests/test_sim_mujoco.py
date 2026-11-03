"""sim/mujoco_rig.py: MJCF build, label renders and visible fractions (skipped when MuJoCo is not installed).

MuJoCo is not a project dependency; run with it in a throwaway env (sim/README.md):
  MUJOCO_GL=egl uv run --no-project --python 3.13.5 --with numpy==2.5.3 --with mujoco==3.17.0 \
      python -m unittest tests.test_sim_mujoco
The cameras are the option-A draft stops at quarter resolution (576x324, f / 4: the same field of view) so the whole
class renders in about a second. The rig is the real configs/rig_geometry.json.
"""
import dataclasses
import importlib.util
import os
import unittest

import numpy as np

from sim import rig as R

SKIP = None
if importlib.util.find_spec("mujoco") is None:
    SKIP = "mujoco not installed (the sim runs in a throwaway uv env; see sim/README.md)"
else:
    try:
        from sim import mujoco_rig as MR    # first: it defaults MUJOCO_GL to egl before mujoco is imported

        import mujoco
    except Exception as e:      # mujoco present but its GL backend (MUJOCO_GL) failed to load
        SKIP = f"mujoco import failed with MUJOCO_GL={os.environ.get('MUJOCO_GL')}: {e}"

A_DRAFT = {"schema": "cmpe249.positions.v1", "radius_cm": 126.8, "arc_centre_behind_front_cm": 10.89,
           "arc_rotation_deg": -2.36, "height_cm": 40.78, "tilt_deg": 6.83,
           "stops": [{"name": f"P{i + 1}", "azimuth_deg": (i - 2) * 22.5} for i in range(5)]}   # positions_far_A_draft.json


def quarter(rig: R.Rig) -> R.Rig:
    cams = [dataclasses.replace(c, width=c.width // 4, height=c.height // 4, f_px=c.f_px / 4) for c in rig.cameras]
    return dataclasses.replace(rig, cameras=cams)


@unittest.skipIf(SKIP, SKIP)
class TestMujocoRig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:                                    # an offscreen GL context at all? (skip, not fail, on GL-less hosts)
            probe = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8)
            probe.close()
        except Exception as e:
            raise unittest.SkipTest(f"no offscreen GL context (MUJOCO_GL={os.environ.get('MUJOCO_GL')}): {e}")
        cls.rig = quarter(R.load_rig(R.DEFAULT_GEOMETRY_PATH, A_DRAFT, mug_z=0.24))
        cls.mr = MR.MujocoRig(cls.rig)
        cls.stops = [c.name for c in cls.rig.cameras]

    @classmethod
    def tearDownClass(cls):
        cls.mr.close()

    def setUp(self):
        self.mr.set_mug(x=0.0, y=0.24)
        self.mr.set_occluder(None)

    def fractions(self):
        return [r["fraction"] for r in self.mr.fractions("mug")]

    def test_mjcf_cameras(self):
        m = self.mr.model
        self.assertEqual(m.ncam, 5)
        for i, cam in enumerate(self.rig.cameras):
            self.assertAlmostEqual(float(m.cam_fovy[i]), cam.fovy_deg, places=6)
            np.testing.assert_allclose(self.mr.data.cam_xpos[i], cam.position, atol=1e-9)
        self.assertIn('mocap="true"', self.mr.xml)

    def test_no_occluder_is_fully_visible(self):
        self.assertEqual(self.fractions(), [1.0] * 5)

    def test_full_occluder_hides_the_mug(self):
        wall = R.OccluderSpec("wall", width=0.79, height=0.165, thickness=0.005, y_back=0.10, shift=0.0)
        self.mr.set_occluder(wall)
        self.assertEqual(self.fractions(), [0.0] * 5)
        self.assertEqual(R.pattern(self.fractions()), "HHHHH")

    def test_design_picks_keep_their_patterns(self):
        for name in ("pick_S8", "pick_M16+3", "pick_M16-5", "pick_IND28"):
            self.mr.set_occluder(name)
            with self.subTest(occluder=name):
                self.assertEqual(R.pattern(self.fractions()), R.occluder_spec(self.rig.geometry, name).intended_pattern)

    def test_renders_agree_with_the_projection(self):
        """MJCF camera conventions and mesh placement (MuJoCo recentres mesh assets) against sim/rig.py's projection."""
        V = np.vstack([m.world_vertices() for m in self.rig.meshes()])
        for cam in self.rig.cameras:
            lab = self.mr.render_labels(cam.name)
            boxes = {f"tag{t.id}": t.corners() for t in self.rig.tags}
            boxes["mug"] = V
            for label, pts in boxes.items():
                uv = cam.project(pts)
                pb = np.floor([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
                if pb[0] < 0 or pb[1] < 0 or pb[2] >= cam.width or pb[3] >= cam.height:
                    continue                                   # clipped by the frame edge
                src = self.mr.alone_mask("mug", cam.name) if label == "mug" else lab
                bb = R.label_bbox(src, self.mr.label_id(label))
                with self.subTest(camera=cam.name, label=label):
                    self.assertIsNotNone(bb)
                    self.assertLessEqual(np.abs(np.array(bb) - pb).max(), 1)

    def test_export_matches_the_mjcf(self):
        """Both backends must see one geometry: every box of export_scene_spec(rig) is a MuJoCo geom of the same size
        and place, the tags sit where MuJoCo draws the markers, and the exported cameras are MuJoCo's cameras."""
        self.mr.set_occluder("pick_S8")
        spec = R.export_scene_spec(self.mr.rig)
        m, d = self.mr.model, self.mr.data
        for b in spec["boxes"]:
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, b["name"])
            with self.subTest(box=b["name"]):
                self.assertGreaterEqual(g, 0)
                np.testing.assert_allclose(m.geom_size[g], b["half"], atol=1e-6)
                np.testing.assert_allclose(d.geom_xpos[g], b["center"], atol=1e-6)
        for t in spec["tags"]:
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"tag{t['id']}")
            np.testing.assert_allclose(d.geom_xpos[g], np.mean(t["corners"], axis=0), atol=1e-3)   # 0.6 mm render offset
            self.assertAlmostEqual(2 * float(m.geom_size[g][0]), t["size_m"], places=9)
        self.assertEqual(sorted(x["name"] for x in spec["meshes"]), ["mug_body", "mug_handle"])
        for i, c in enumerate(spec["cameras"]):
            xm = d.cam_xmat[i].reshape(3, 3)          # columns: camera x (right), y (up), z (backward) in the world
            with self.subTest(camera=c["name"]):
                np.testing.assert_allclose(d.cam_xpos[i], c["position"], atol=1e-6)
                np.testing.assert_allclose(xm[:, 0], c["right"], atol=1e-6)
                np.testing.assert_allclose(xm[:, 1], c["up"], atol=1e-6)
                np.testing.assert_allclose(-xm[:, 2], c["forward"], atol=1e-6)

    def test_moving_the_mug_and_cache(self):
        before = self.mr.visible_fraction("mug", "P3")["alone_px"]
        self.mr.set_mug(y=0.12)                                # closer to the camera: more pixels
        after = self.mr.visible_fraction("mug", "P3")["alone_px"]
        self.assertGreater(after, before)
        self.mr.set_occluder("plan_IND26")                     # front face at y 0.175: now behind the mug (back 0.159)
        self.assertEqual(self.fractions(), [1.0] * 5)

    def test_hiding_restores_groups(self):
        groups = self.mr.model.geom_group.copy()
        self.mr.render_labels("P3", only=["mug"])
        np.testing.assert_array_equal(self.mr.model.geom_group, groups)

    def test_multisampled_context_refuses_labels(self):
        with MR.MujocoRig(self.rig, offsamples=4) as ms:
            self.assertEqual(ms.render_rgb("P3").shape, (324, 576, 3))
            with self.assertRaises(RuntimeError):
                ms.render_labels("P3")


if __name__ == "__main__":
    unittest.main()
