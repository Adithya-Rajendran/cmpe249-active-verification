"""A tabletop set end to end without Isaac (tabletop wave, step 8): tests/fake_isaac.py renders a tabletop arrangement through
sim/isaac_batch.py main(), then sim/episodes.py builds the set and sim/check_batch.py checks it. numpy and cv2 only.

Pinned here (the shelf's outputs are pinned by tests/test_shelf_goldens.py, which must pass unchanged):
  - the environment comes from the records (top level and scene spec; neither = the shelf); a set or a check over two
    environments is refused with a clear error, before anything is written;
  - annotations name the table workspace, manifest rows carry "environment": "tabletop" (verify reads it, strata
    "tabletop:<family>"), set.json and the README say tabletop, true_alignment.json has the table and target planes;
  - check_batch takes its plane keys from the alignment (table_plane / target_plane) and reports the clipped-table-pixel
    share per capture and stop: table pixels = camera rays meeting the table top AND class 0 at the same pose; grey >=
    perception/evidence.py's clip_hi (251); the pre-set criterion < 1 % per stop.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from perception.evidence import Params
from sim import alignment as AL
from sim import arrange as A
from sim import check_batch as CB
from sim import episodes as E
from sim import isaac_batch as B
from tests import fake_isaac as FI
from tests.test_tabletop_render import _TexDir, tabletop_arrangement, tabletop_spec
from verify.manifest import load_manifest_info

REPO = Path(__file__).resolve().parents[1]
POS_TABLETOP = REPO / "configs" / "sim" / "positions_tabletop_A.json"
STOPS = FI.STOPS


def _small(cam: dict, dx: float = 0.0) -> dict:
    """A scene-spec camera scaled to the fake renderer's 16 x 12 frames (same pose and horizontal field of view), moved dx m
    along x (a revisit pose)."""
    c = copy.deepcopy(cam)
    c["f_px"] = c["f_px"] * FI.W / c["width"]
    c["width"], c["height"] = FI.W, FI.H
    c["principal_point_px"] = [FI.W / 2, FI.H / 2]
    c["position"] = [c["position"][0] + dx, *c["position"][1:]]
    return c


def fake_tabletop_arrangement(spec: dict, arr_id: str = "TD5000", seed: int = 5000) -> dict:
    """tests/test_tabletop_render.py's tabletop arrangement made renderable by the fake Isaac and complete enough for
    episodes and check_batch: 16 x 12 cameras, revisit cameras 2 mm off, the generator's top-level environment, the
    target's size, the analytic coverage and the tabletop positions file's sha256."""
    arr = tabletop_arrangement(spec)
    arr["arrangement_id"], arr["seed"] = arr_id, seed
    arr["environment"] = "tabletop"
    arr["scene"]["cameras"] = [_small(c) for c in spec["cameras"]]
    arr["scene"]["revisit_cameras"] = [_small(c, 0.002) for c in spec["cameras"]]
    arr["objects"]["target"]["size_m"] = [0.06, 0.06, 0.10]
    arr["analytic"] = {"placed_coverage": {s: 0.5 for s in STOPS}}
    arr["provenance"]["positions"] = {"path": "configs/sim/positions_tabletop_A.json", "sha256": A.sha256_file(POS_TABLETOP)}
    return arr


class _Rendered:
    """Fake-renders arrangements into a scratch set root (sim/isaac_batch.py main against tests/fake_isaac.py)."""

    def __init__(self, arrs):
        self.tex = _TexDir()
        self.tmp = tempfile.TemporaryDirectory()
        work = Path(self.tmp.name)
        (work / "assets").mkdir()
        files = []
        for a in arrs:
            f = work / f"{a['arrangement_id']}.json"
            f.write_text(json.dumps(a) + "\n")
            files.append(f)
        self.root = work / "set"
        FI.reset(1)
        with contextlib.ExitStack() as st:
            for p in FI.patches(B):
                st.enter_context(p)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                B.main(["--arrangements", *map(str, files), "--assets-dir", str(work / "assets"), "--tags-dir", "tags",
                        "--textures-dir", str(self.tex.dir), "--out", str(self.root)])

    def close(self):
        self.tmp.cleanup()
        self.tex.close()


class TestEnvironmentOfRecords(unittest.TestCase):
    def test_from_the_top_level_or_the_scene_and_absent_is_the_shelf(self):
        self.assertEqual(E.environment({}), "shelf")
        self.assertEqual(E.environment({"environment": "shelf", "scene": {}}), "shelf")
        self.assertEqual(E.environment({"scene": {"environment": "tabletop"}}), "tabletop")
        self.assertEqual(E.environment({"environment": "tabletop", "scene": {"environment": "tabletop"}}), "tabletop")
        with self.assertRaisesRegex(ValueError, "scene spec says"):
            E.environment({"arrangement_id": "X", "environment": "shelf", "scene": {"environment": "tabletop"}})
        with self.assertRaisesRegex(ValueError, "must be one of"):
            E.environment({"environment": "kitchen"})

    def test_a_mixed_list_is_refused_naming_both(self):
        rendered = [({"arrangement_id": "SD1000"}, {}, Path(".")), ({"arrangement_id": "TD5000", "environment": "tabletop"}, {}, Path("."))]
        with self.assertRaisesRegex(ValueError, r"2 environments.*shelf: \['SD1000'\].*tabletop: \['TD5000'\]"):
            E.check_single_environment(rendered)
        self.assertEqual(E.check_single_environment(rendered[1:]), "tabletop")


class TestTableRayMask(unittest.TestCase):
    RECT = (-0.5, 0.5, -0.5, 0.5, 0.0)

    def test_straight_down_sees_the_table_square(self):
        cam = {"position": [0.0, 0.0, 1.0], "forward": [0.0, 0.0, -1.0], "right": [1.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0],
               "f_px": 10.0, "width": 20, "height": 20}
        m = CB.table_ray_mask(cam, self.RECT)
        want = np.zeros((20, 20), bool)
        want[5:15, 5:15] = True                     # |pixel centre - 10| / 10 <= 0.5 at 1 m
        np.testing.assert_array_equal(m, want)

    def test_nothing_behind_the_camera(self):
        cam = {"position": [0.0, 0.0, 1.0], "forward": [0.0, 0.0, 1.0], "right": [1.0, 0.0, 0.0], "up": [0.0, -1.0, 0.0],
               "f_px": 10.0, "width": 20, "height": 20}
        self.assertFalse(CB.table_ray_mask(cam, self.RECT).any())

    def test_top_rect_from_the_spec_mesh(self):
        spec = tabletop_spec()
        self.assertEqual(CB.table_top_rect({"arrangement_id": "X", "scene": spec}), (-0.6, 0.6, -0.3, 0.5, 0.0))
        with self.assertRaisesRegex(ValueError, "table_top"):
            CB.table_top_rect({"arrangement_id": "X", "scene": {"meshes": []}})

    def test_clip_hi_is_the_frozen_evidence_value(self):
        self.assertEqual(CB.CLIP_HI, Params().clip_hi)
        self.assertEqual(CB.CLIP_HI, 251)
        self.assertEqual(CB.TABLE_CLIP_LIMIT, 0.01)


class TestTabletopSetEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tex_probe = _TexDir()
        cls.spec = tabletop_spec(tex_sha=cls.tex_probe.sha)        # every _TexDir holds the same stand-in bytes
        cls.r = _Rendered([fake_tabletop_arrangement(cls.spec)])
        cls.root = cls.r.root
        cls.doc = E.build(cls.root, E.REF_LIGHTINGS, POS_TABLETOP)
        cls.check = CB.check_set(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.r.close()
        cls.tex_probe.close()

    def test_annotations_name_the_table_workspace(self):
        ann = json.loads((self.root / "TD5000" / "annotations" / "ann_P3_L1.json").read_text())
        where = ann["targets"]["tomato_soup_can"]["where"]
        self.assertEqual(where, "P3, on the table (workspace), from the L1 reference instance mask")
        self.assertNotIn("compartment", where)
        self.assertIn("NO corner tags", ann["note"])
        self.assertEqual(ann["targets"]["tomato_soup_can"]["box"], [4, 4, 8, 8])

    def test_manifest_carries_the_environment_and_verify_reads_it(self):
        rows = json.loads((self.root / "manifest.json").read_text())["arrangements"]
        self.assertEqual([r["environment"] for r in rows], ["tabletop"])
        self.assertEqual(list(rows[0])[-1], "environment")
        info = load_manifest_info(self.root / "manifest.json")
        self.assertEqual(info.environments, ("tabletop",))
        self.assertEqual(info.strata(), {"TD5000": "tabletop:white_box"})

    def test_set_json_readme_and_positions(self):
        s = json.loads((self.root / "set.json").read_text())
        self.assertEqual((list(s)[:2], s["environment"]), (["schema", "environment"], "tabletop"))
        self.assertNotIn("DES0", s["jobs_note"])
        self.assertTrue((self.root / "positions_tabletop_A.json").is_file())
        readme = (self.root / "README.md").read_text()
        self.assertIn("TABLETOP", readme)
        self.assertIn("positions_tabletop_A.json", readme)
        self.assertNotIn("positions_shelf_A.json", readme)
        self.assertNotIn("compartment", readme)

    def test_true_alignment_has_the_table_and_target_planes(self):
        ta = json.loads((self.root / "true_alignment.json").read_text())
        self.assertEqual(ta["environment"], "tabletop")
        st = ta["arrangements"]["TD5000"]["stops"]["P3"]
        self.assertEqual({k for k in st if k.startswith("H_")}, {"H_table_plane", "H_target_plane"})
        self.assertEqual(ta["arrangements"]["TD5000"]["target_plane_z_m"], 0.05)
        # the revisit camera is 2 mm along x: the table plane's homography is a pure shift, not the identity
        self.assertFalse(np.allclose(np.array(st["H_table_plane"]), np.eye(3)))

    def test_check_batch_plane_keys_come_from_the_alignment(self):
        a = self.check["arrangements"][0]
        self.assertEqual(set(a["true_alignment"]["P3"]), {"pose_delta", "table_plane_max_corner_px", "target_plane_max_corner_px"})
        ta = self.check["summary"]["true_alignment_corner_displacement_px"]
        self.assertEqual({k for k in ta if k.endswith("_plane")}, {"table_plane", "target_plane"})
        self.assertEqual(ta["n_views"], 5)
        self.assertEqual(self.check["summary"]["environment"], "tabletop")
        self.assertTrue(a["tag_free"])

    def test_table_pixels_are_ray_hits_minus_objects(self):
        arr = json.loads((self.root / "TD5000" / "arrangement.json").read_text())
        rect = CB.table_top_rect(arr)
        caps = {(c["phase"], c["stop"], c["lighting"]): c for c in self.check["arrangements"][0]["table_clipping"]["captures"]}
        self.assertEqual(len(caps), 4 * 5 * 2)
        for ph in CB.PHASES:
            cams = arr["scene"]["cameras"] if ph == "reference" else arr["scene"]["revisit_cameras"]
            for cam in cams:
                cls = cv2.imread(str(self.root / "TD5000" / "masks" / f"classes_{ph}_{cam['name']}.png"), cv2.IMREAD_GRAYSCALE)
                ray = CB.table_ray_mask(cam, rect)
                want = int(np.count_nonzero(ray & (cls == 0)))
                self.assertGreater(want, 0)
                self.assertLess(want, int(np.count_nonzero(ray)) + (0 if (cls > 0).any() else 1))   # objects are left out
                for L in ("L1", "L2"):
                    self.assertEqual(caps[(ph, cam["name"], L)]["n_table_px"], want)
                    self.assertEqual(caps[(ph, cam["name"], L)]["share"], 0.0)       # fake grey 101-211 < 251
        tc = self.check["summary"]["table_clipping"]
        self.assertEqual((tc["pass"], tc["max_share"], tc["captures_at_or_over_limit"]), (True, 0.0, []))

    def test_clipped_table_pixels_fail_the_criterion_at_their_stop(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            import shutil

            shutil.copytree(self.root, root)
            d = root / "TD5000"
            arr = json.loads((d / "arrangement.json").read_text())
            cam = next(c for c in arr["scene"]["cameras"] if c["name"] == "P2")
            cls = cv2.imread(str(d / "masks" / "classes_reference_P2.png"), cv2.IMREAD_GRAYSCALE)
            table = CB.table_ray_mask(cam, CB.table_top_rect(arr)) & (cls == 0)
            img = cv2.imread(str(d / "rgb" / "reference_P2_L1.png"))
            ys, xs = np.nonzero(table)
            img[ys[:3], xs[:3]] = 255                             # 3 table pixels blown out, all else untouched
            img[cls > 0] = 255                                    # object pixels do not count, however bright
            cv2.imwrite(str(d / "rgb" / "reference_P2_L1.png"), img)
            res = CB.check_set(root)
        tc = res["summary"]["table_clipping"]
        n = int(table.sum())
        self.assertAlmostEqual(tc["max_share_per_stop"]["P2"], 3 / n)
        self.assertEqual({s: v for s, v in tc["max_share_per_stop"].items() if s != "P2"}, {s: 0.0 for s in STOPS if s != "P2"})
        self.assertEqual([x[:4] for x in tc["captures_at_or_over_limit"]], [("TD5000", "reference", "P2", "L1")])
        self.assertFalse(tc["pass"])

    def test_cli_defaults_to_the_tabletop_positions(self):
        with tempfile.TemporaryDirectory() as t:
            import shutil

            root = Path(t) / "set"
            shutil.copytree(self.root, root)
            (root / "positions_tabletop_A.json").unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(E.main(["--root", str(root)]), 0)
            self.assertTrue((root / "positions_tabletop_A.json").is_file())
            self.assertFalse((root / "positions_shelf_A.json").exists())


class TestMixedSetsAreRefused(unittest.TestCase):
    def test_episodes_and_check_batch_refuse_two_environments(self):
        tex = _TexDir()
        self.addCleanup(tex.close)
        r = _Rendered([fake_tabletop_arrangement(tabletop_spec(tex_sha=tex.sha)), FI.arrangement()])
        self.addCleanup(r.close)
        self.assertEqual(sorted(p.name for p in r.root.iterdir() if p.is_dir()), ["SD1003", "TD5000"])
        with self.assertRaisesRegex(ValueError, "2 environments"):
            E.build(r.root, positions=None)
        self.assertFalse((r.root / "manifest.json").exists())
        self.assertFalse((r.root / "TD5000" / "annotations").exists())     # refused before anything is written
        with self.assertRaisesRegex(ValueError, "2 environments"):
            CB.check_set(r.root)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(E.main(["--root", str(r.root)]), 2)
        self.assertIn("environments", err.getvalue())

    def test_the_true_alignment_document_refuses_them_too(self):
        tab = fake_tabletop_arrangement(tabletop_spec())
        shelf = copy.deepcopy(tab)
        del shelf["scene"]["environment"]
        with self.assertRaisesRegex(ValueError, "different environments"):
            AL.doc([(tab, None), (shelf, None)])


if __name__ == "__main__":
    unittest.main()
