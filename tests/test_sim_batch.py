"""Tests for the pure parts of sim/isaac_batch.py and for sim/episodes.py (numpy only; no Isaac Sim, no images).

Pins: the dev/test guard in the renderer, pose and asset transforms agreeing with sim/arrange.py's Body, segmentation
-> class maps, mask boxes, the visibility / diagnostic labels, and the evidence annotation, jobs (primary refL1 and
secondary same-lighting) and manifest formats the next stage reads.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim import arrange as A
from sim import episodes as E
from sim import isaac_batch as B
from verify.manifest import RE_ARRANGEMENT, load_manifest
from verify.schema import Condition, Lighting, episode_id

STOPS = ["P1", "P2", "P3", "P4", "P5"]


def _labels(arr_id: str, seed, target: str, ind: bool = False) -> dict:
    vf = {"reference": {s: 1.0 for s in STOPS}, "visible": {s: 1.0 for s in STOPS},
          "occluded": dict(zip(STOPS, [1.0, 0.95, 0.0, 0.4, 1.0]))}
    obst = {"occluded": dict(zip(STOPS, [0.0, 0.05, 1.0, 0.6, 0.0])), "removed": dict(zip(STOPS, [0.0, 0.05, 1.0, 0.6, 0.0]))}
    diag = {"visible": {s: True for s in STOPS}, "occluded": dict(zip(STOPS, [True, True, False, False, True])),
            "removed": dict(zip(STOPS, [True, True, False, False, True]))}
    boxes = {L: {s: [100 + i, 200, 180 + i, 260] for i, s in enumerate(STOPS)} for L in ("L1", "L2")}
    return {"schema": B.LABELS_SCHEMA, "arrangement_id": arr_id, "seed": seed, "split": "dev", "target_id": target,
            "stops": STOPS, "lightings": ["L1", "L2"], "visible_fraction": vf, "reference_region_obstruction": obst,
            "diagnostic": diag, "n_diagnostic": {k: sum(v.values()) for k, v in diag.items()},
            "patterns": {"reference": "CCCCC", "visible": "CCCCC", "occluded": "CCHpC"},
            "truth": dict(B.CONDITION_TRUTH), "reference_box": boxes, "rules": {"present": "x"},
            "occluder_family": "S8", "predeclared_indistinguishable": ind}


def _write_set(root: Path, items) -> None:
    for arr_id, seed, kind, target, ind in items:
        d = root / arr_id
        d.mkdir(parents=True)
        arr = {"schema": A.ARRANGEMENT_SCHEMA, "arrangement_id": arr_id, "seed": seed, "split": "dev", "kind": kind,
               "target_id": target, "predeclared_indistinguishable": ind,
               "objects": {"occluder": {"family": "S8", "kind": "box", "label": "S8"}},
               "provenance": {"generator": {"sha256": "g" * 64}, "config": {"sha256": "c" * 64}}, "arrangement_sha256": "a" * 64}
        (d / "arrangement.json").write_text(json.dumps(arr))
        (d / "labels.json").write_text(json.dumps(_labels(arr_id, seed, target, ind)))


class TestRendererGuard(unittest.TestCase):
    def _arr(self, **kw):
        a = {"schema": A.ARRANGEMENT_SCHEMA, "arrangement_id": "SD0001", "seed": 1, "split": "dev", "kind": "random"}
        a.update(kw)
        return a

    def test_constants_match_the_generator(self):
        self.assertEqual(B.DEV_SEED_RANGE, A.DEV_SEED_RANGE)
        self.assertEqual(B.TEST_SEED_MIN, A.TEST_SEED_MIN)
        self.assertEqual(B.PHASE_ORDER, tuple(p for p, _, _ in A.PHASES))

    def test_dev_and_design_pass(self):
        B.check_arrangement(self._arr())
        B.check_arrangement(self._arr(seed=9999))
        B.check_arrangement(self._arr(seed="design-3", kind="design", arrangement_id="DES3"))

    def test_test_and_other_seeds_refused(self):
        for kw in ({"seed": 100000}, {"seed": 123456}, {"seed": 10000}, {"seed": -1}, {"seed": "7"}, {"seed": True},
                   {"split": "test"}, {"seed": "design-1"}, {"schema": "other"}):
            with self.subTest(**kw), self.assertRaises(B.RefusedArrangement):
                B.check_arrangement(self._arr(**kw))
        with self.assertRaisesRegex(B.RefusedArrangement, "TEST"):
            B.check_arrangement(self._arr(seed=100000))


class TestTransforms(unittest.TestCase):
    def test_box_and_pose_match_body_corners(self):
        body = A.Body((0.16, 0.005, 0.12), (0.03, 0.0325, 0.0), 25.0)
        m = B.box_local_matrix(body.size) @ B.pose_matrix(body.base, body.yaw_deg)
        cube = np.array([[x, y, z, 1.0] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        world = (cube @ m)[:, :3]
        want = body.corners()
        self.assertTrue(all(np.min(np.linalg.norm(want - w, axis=1)) < 1e-9 for w in world))

    def test_asset_local_matches_generator_frame(self):
        ctx = A.load_context()
        item = ctx.item(3)                       # sugar box: 50 x 94 x 176 mm, laid on its long side
        lo = np.asarray(item["min_mm"])
        hi = lo + np.asarray(item["size_mm"])
        verts = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        v, _ = B.asset_local(verts, ctx.rest("side"))
        ext = A.rest_extents(item, ctx.rest("side"))
        self.assertTrue(np.allclose(v.max(0) - v.min(0), ext))
        self.assertTrue(np.allclose([(v[:, 0].max() + v[:, 0].min()) / 2, (v[:, 1].max() + v[:, 1].min()) / 2, v[:, 2].min()], 0))
        # the generator's hull points land in the same frame
        pts = ctx.local_points(3, "side")
        self.assertTrue(np.allclose(pts.min(0), v.min(0), atol=1e-6) and np.allclose(pts.max(0), v.max(0), atol=1e-6))


class TestMasksAndLabels(unittest.TestCase):
    def test_class_map_prefixes(self):
        ids = np.array([[0, 1, 2, 3], [4, 5, 5, 0]], np.uint32)
        info = {"1": "/World/Objects/target/mesh", "2": "/World/Objects/occluder/box", "3": "/World/Objects/distractor_1/mesh",
                "4": "/World/Objects/distractor_10/mesh", "5": "/World/Shelf/shelf_mid_board"}
        roles = {"/World/Objects/target": 1, "/World/Objects/occluder": 2, "/World/Objects/distractor_1": 11,
                 "/World/Objects/distractor_10": 20}
        cm = B.class_map(ids, info, roles)
        self.assertEqual(cm.tolist(), [[0, 1, 2, 11], [20, 0, 0, 0]])
        self.assertEqual(cm.dtype, np.uint8)

    def test_mask_box_is_continuous(self):
        m = np.zeros((10, 20), bool)
        m[3:6, 4:9] = True
        self.assertEqual(B.mask_box(m), [4, 3, 9, 6])
        self.assertIsNone(B.mask_box(np.zeros((3, 3), bool)))

    def test_compute_labels(self):
        h, w = 20, 20
        alone = {s: np.zeros((h, w), bool) for s in STOPS}
        for s in STOPS:
            alone[s][5:15, 5:15] = True                       # 100 target pixels everywhere
        tgt, occ = {}, {}
        hidden = dict(zip(STOPS, [0, 5, 100, 50, 11]))        # occluded pixels per stop
        for s in STOPS:
            tgt[("reference", s)] = alone[s].copy()
            tgt[("visible", s)] = alone[s].copy()
            t = alone[s].copy()
            o = np.zeros((h, w), bool)
            flat_t, flat_o = t.reshape(-1), o.reshape(-1)
            idx = np.flatnonzero(flat_t)[:hidden[s]]
            flat_t[idx] = False
            flat_o[idx] = True
            tgt[("occluded", s)] = t
            occ[("occluded", s)] = o
            occ[("removed", s)] = o.copy()
        lab = B.compute_labels(STOPS, tgt, alone, occ)
        self.assertEqual(lab["visible_fraction"]["occluded"], dict(zip(STOPS, [1.0, 0.95, 0.0, 0.5, 0.89])))
        self.assertEqual(lab["patterns"]["occluded"], "CCHpp")
        self.assertEqual(lab["diagnostic"]["occluded"], dict(zip(STOPS, [True, True, False, False, False])))
        self.assertEqual(lab["reference_region_obstruction"]["removed"], dict(zip(STOPS, [0.0, 0.05, 1.0, 0.5, 0.11])))
        self.assertEqual(lab["diagnostic"]["removed"], dict(zip(STOPS, [True, True, False, False, False])))
        self.assertEqual(lab["n_diagnostic"], {"visible": 5, "occluded": 2, "removed": 2})
        self.assertEqual(lab["truth"], {"visible": "present", "occluded": "present", "removed": "removed"})
        self.assertFalse(lab["indistinguishable_by_masks"])
        for s in STOPS:                                       # cover every stop -> no diagnostic view in either
            occ[("removed", s)] = alone[s].copy()
            tgt[("occluded", s)] = np.zeros((h, w), bool)
        self.assertTrue(B.compute_labels(STOPS, tgt, alone, occ)["indistinguishable_by_masks"])


class TestEpisodes(unittest.TestCase):
    def test_build_writes_the_formats_the_next_stage_reads(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("DES3", "design-3", "design", "study_mug", True), ("SD0002", 2, "random", "mug", False)])
            doc = E.build(root, positions=None)
            self.assertEqual(doc["jobs"], {"primary": "jobs_refL1.json", "secondary": "jobs_same.json"})
            prim = json.loads((root / "jobs_refL1.json").read_text())
            sec = json.loads((root / "jobs_same.json").read_text())
            self.assertEqual((prim["role"], prim["ref_lighting"], sec["role"], sec["ref_lighting"]),
                             ("primary", "L1", "secondary", "same"))
            for jd in (prim, sec):
                self.assertEqual(jd["schema"], "cmpe249.evidence-jobs.v0")
                self.assertEqual(len(jd["jobs"]), 2 * 3 * 2 * 5)
                for j in jd["jobs"]:
                    self.assertEqual(set(j), {"episode_id", "stop", "target", "annotation", "query"})
                    arr, cond, light = j["episode_id"].split("-")
                    self.assertRegex(arr, RE_ARRANGEMENT)
                    self.assertEqual(j["episode_id"], episode_id(arr, Condition(cond), Lighting(light)))
                    self.assertIsInstance(j["stop"], int)
                    self.assertTrue(0 <= j["stop"] < 5)
                    self.assertTrue(Path(j["annotation"]).is_absolute() and Path(j["query"]).is_absolute())
                    self.assertTrue(j["query"].endswith(f"/rgb/{cond}_{STOPS[j['stop']]}_{light}.png"))
                    ann = json.loads(Path(j["annotation"]).read_text())
                    self.assertEqual(ann["schema"], "cmpe249.evidence-annotation.v0")
                    self.assertEqual(ann["stop"], j["stop"])
                    x0, y0, x1, y1 = ann["targets"][j["target"]]["box"]
                    self.assertTrue(x1 > x0 and y1 > y0)
                    ref_l = "L1" if jd is prim else light
                    self.assertTrue(ann["reference"].endswith(f"/rgb/reference_{STOPS[j['stop']]}_{ref_l}.png"))
                    self.assertEqual(ann["lighting"], ref_l)
            # one annotation per (arrangement, stop, reference lighting)
            self.assertEqual(len(list(root.glob("*/annotations/ann_*_L*.json"))), 2 * 5 * 2)
            specs, truth = load_manifest(root / "manifest.json")
            self.assertEqual(len(specs), 12)
            self.assertEqual({s.split.value for s in specs}, {"dev"})
            self.assertEqual(sorted(truth.indistinguishable), ["DES3-occluded-L1", "DES3-occluded-L2",
                                                                "DES3-removed-L1", "DES3-removed-L2"])
            tr = json.loads((root / "truth.json").read_text())["states"]
            self.assertEqual(tr["SD0002-occluded-L2"], "present")
            self.assertEqual(tr["SD0002-removed-L1"], "removed")
            vl = json.loads((root / "view_labels.json").read_text())["episodes"]
            self.assertEqual([s["diagnostic"] for s in vl["SD0002-occluded-L1"]["stops"]], [True, True, False, False, True])

    def test_annotations_load_with_the_frozen_evidence_reader(self):
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("cv2 not installed (perception/evidence.py imports it)")
        from perception.evidence import load_annotation

        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD0004", 4, "random", "tuna_fish_can", False)])
            E.build(root, positions=None)
            for p in root.glob("SD0004/annotations/*.json"):
                self.assertIn("tuna_fish_can", load_annotation(p)["targets"])

    def test_non_dev_arrangements_are_refused(self):
        for seed, kind in ((100000, "random"), (12345, "random"), ("design-0", "random")):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as t:
                root = Path(t) / "set"
                _write_set(root, [("SD0009", seed, kind, "mug", False)])
                with self.assertRaises(A.SeedError):
                    E.build(root, positions=None)


if __name__ == "__main__":
    unittest.main()
