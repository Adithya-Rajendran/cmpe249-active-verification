"""Tests for sim/arrange.py (the seeded shelf arrangement generator) and configs/sim/positions_shelf_A.json. numpy only.

Pins: determinism, valid placements (inside the compartments, no overlapping footprints, the target fits #3), the
matched occluded/removed layout, the dev/test seed guard, the design cases, and the per-stop pitch file (loads, every
stop's in-frame margin with the fitted k1 is positive).
"""
import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim import arrange as A
from sim import rig as R

REPO = Path(__file__).resolve().parents[1]
POS = REPO / "configs" / "sim" / "positions_shelf_A.json"
DRAFT_A = REPO / "data" / "raw" / "2026-10-10_far-camera" / "positions_far_A_draft.json"
SEEDS = list(range(8))


def _ctx():
    if not hasattr(_ctx, "c"):
        _ctx.c = A.load_context()
    return _ctx.c


def _gen(seed):
    if not hasattr(_gen, "cache"):
        _gen.cache = {}
    if seed not in _gen.cache:
        _gen.cache[seed] = A.generate(seed, ctx=_ctx())
    return _gen.cache[seed]


class TestSeedGuard(unittest.TestCase):
    def test_dev_seeds_pass(self):
        self.assertEqual([A.check_dev_seed(s) for s in (0, 1, 9999)], [0, 1, 9999])
        self.assertEqual(A.arrangement_id(3), "SD0003")
        self.assertEqual(A.arrangement_id(9999), "SD9999")
        self.assertEqual(A.parse_seeds("0-2,7"), [0, 1, 2, 7])

    def test_test_seeds_are_refused(self):
        for s in (100000, 100001, 123456, 10**9):
            with self.subTest(seed=s), self.assertRaisesRegex(A.SeedError, "TEST"):
                A.check_dev_seed(s)
            with self.subTest(seed=s), self.assertRaisesRegex(A.SeedError, "TEST"):
                A.generate(s, ctx=_ctx())
        with self.assertRaisesRegex(A.SeedError, "TEST"):
            A.parse_seeds("99990-100002")

    def test_other_non_dev_seeds_are_refused(self):
        for s in (-1, 10000, 99999, "3", 3.0, True, None):
            with self.subTest(seed=s), self.assertRaises(A.SeedError):
                A.check_dev_seed(s)

    def test_cli_refuses_test_seeds_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "out"
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(A.main(["--seeds", "100000", "--out", str(out)]), 2)
            self.assertIn("TEST", err.getvalue())
            self.assertFalse(out.exists())

    def test_config_seed_ranges_are_the_generators(self):
        cfg = json.loads(A.DEFAULT_CONFIG_PATH.read_text())
        self.assertEqual(tuple(cfg["seeds"]["dev"]), A.DEV_SEED_RANGE)
        self.assertEqual(cfg["seeds"]["test_reserved_min"], A.TEST_SEED_MIN)
        bad = json.loads(json.dumps(cfg))
        bad["seeds"]["test_reserved_min"] = 50000
        with self.assertRaises(ValueError):
            A.load_config(bad)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_arrangement(self):
        a = A.generate(7, ctx=_ctx())
        b = A.generate(7, ctx=A.load_context())          # fresh context, same files
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertEqual(a["arrangement_sha256"], b["arrangement_sha256"])

    def test_different_seeds_differ(self):
        sigs = {json.dumps(_gen(s)["objects"], sort_keys=True) for s in SEEDS}
        self.assertEqual(len(sigs), len(SEEDS))

    def test_provenance_hashes(self):
        a = _gen(0)
        p = a["provenance"]
        self.assertEqual(p["generator"]["sha256"], A.sha256_file(REPO / "sim" / "arrange.py"))
        self.assertEqual(p["config"]["sha256"], A.sha256_file(A.DEFAULT_CONFIG_PATH))
        self.assertEqual(p["positions"]["sha256"], A.sha256_file(POS))
        self.assertEqual(a["seed"], 0)
        self.assertEqual(a["split"], "dev")


class TestPlacements(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.cfg = self.ctx.config
        self.cl = self.cfg["clearances_m"]

    def _inside(self, body, comp, front=None):
        c = self.ctx.comps[comp]
        fp = body.footprint()
        front = self.cl["front"] if front is None else front
        tol = 1e-6
        self.assertAlmostEqual(body.base[2], c["z"][0], places=6)
        self.assertLessEqual(body.top, c["z"][1] - self.cl["ceiling"] + tol)
        self.assertGreaterEqual(fp[:, 0].min(), c["x"][0] + self.cl["wall"] - tol)
        self.assertLessEqual(fp[:, 0].max(), c["x"][1] - self.cl["wall"] + tol)
        self.assertGreaterEqual(fp[:, 1].min(), c["y"][0] + front - tol)
        self.assertLessEqual(fp[:, 1].max(), c["y"][1] - self.cl["back"] + tol)

    def test_every_seed(self):
        cov_min = self.cfg["occluder"]["p3_coverage_min"]
        for seed in SEEDS:
            with self.subTest(seed=seed):
                a = _gen(seed)
                o = a["objects"]
                t, occ, dis = o["target"], o["occluder"], o["distractors"]
                tb = A.body_of(t, ctx=self.ctx)
                # target: a configured object, fits #3, seen whole from every stop
                self.assertIn(t["obj_id"], self.cfg["target"]["objects"])
                self.assertLessEqual(t["size_m"][2], min(self.cfg["target"]["max_height_m"], 0.165 - self.cl["ceiling"]))
                self._inside(tb, "c3")
                self.assertTrue(all(A.seen_whole(self.ctx, c, tb) for c in self.ctx.cams))
                # occluder: placed in #3 in front of the target, covering it at the start stop
                pb, kb = A.body_of(occ, "placed", self.ctx), A.body_of(occ, "parked", self.ctx)
                self._inside(pb, "c3", front=self.cl["occluder_front"])
                self.assertLess(pb.footprint()[:, 1].max(), tb.footprint()[:, 1].min())
                self.assertTrue(A.separated(pb.footprint(), tb.footprint()))
                self.assertGreaterEqual(A.coverage(self.ctx.start_cam, tb, pb), cov_min)
                covs = {c.name: A.coverage(c, tb, pb) for c in self.ctx.cams}
                oc = self.cfg["occluder"]
                if occ["family"] == "IND28":                    # predeclared indistinguishable: every stop covered
                    self.assertTrue(a["predeclared_indistinguishable"])
                    self.assertGreaterEqual(min(covs.values()), cov_min)
                else:                                           # resolvable: >= 2 stops where the hulls do not touch
                    self.assertFalse(a["predeclared_indistinguishable"])
                    self.assertGreaterEqual(sum(v <= oc["clear_coverage_max"] for v in covs.values()), oc["min_clear_stops"])
                self.assertIn(occ["family"], self.cfg["occluder"]["families"])
                # distractors: 2-5 distinct other models, inside, clear of each other / the target / the placed occluder
                lo, hi = self.cfg["distractors"]["count"]
                self.assertTrue(lo <= len(dis) <= hi)
                ids = [d["obj_id"] for d in dis]
                self.assertEqual(len(set(ids)), len(ids))
                self.assertNotIn(t["obj_id"], ids)
                if occ["kind"] == "ycbv":
                    self.assertNotIn(occ["obj_id"], ids)
                bodies = []
                for d in dis:
                    comp = d["pose"]["compartment"]
                    db = A.body_of(d, ctx=self.ctx)
                    self._inside(db, comp)
                    self.assertIn(d["obj_id"], self.cfg["distractors"][f"{comp}_objects"])
                    if comp == "c3":
                        self.assertTrue(A.separated(db.footprint(), tb.footprint(), self.cl["between_objects"]))
                        self.assertTrue(A.separated(db.footprint(), pb.footprint(), self.cl["between_objects"]))
                        self.assertFalse(any(A.may_occlude(c, db, tb) for c in self.ctx.cams))
                    for comp2, b2 in bodies:
                        if comp2 == comp:
                            self.assertTrue(A.separated(db.footprint(), b2.footprint(), self.cl["between_objects"]))
                    bodies.append((comp, db))
                # parked: occludes nothing from any stop, outside the target's context window
                comp = occ["poses"]["parked"]["compartment"]
                self._inside(kb, comp)
                for c in self.ctx.cams:
                    self.assertFalse(A.may_occlude(c, kb, tb))
                    self.assertFalse(any(A.may_occlude(c, kb, b) for _, b in bodies))
                    self.assertLess(A.overlap_area(A.projected_hull(c, kb.corners()), A.target_window(c, tb, 1.0)), 1.0)

    def test_targets_fit_compartment_3(self):
        clear = self.ctx.comps["c3"]["z"][1] - self.cl["ceiling"]
        for obj in self.cfg["target"]["objects"]:
            ext = A.rest_extents(self.ctx.item(obj), self.ctx.rest("upright"))
            self.assertLessEqual(ext[2], clear, self.ctx.item(obj)["name"])
        for obj in self.cfg["distractors"]["c3_objects"]:
            self.assertLessEqual(A.rest_extents(self.ctx.item(obj), self.ctx.rest("upright"))[2], clear)

    def test_rest_poses_are_rotations(self):
        for name, m in self.cfg["rest_poses"].items():
            if name == "note":
                continue
            m = np.asarray(m, float)
            self.assertTrue(np.allclose(m @ m.T, np.eye(3)), name)
            self.assertAlmostEqual(float(np.linalg.det(m)), 1.0, msg=name)
        # sugar box on its long side: 17.6 wide, 5.0 deep, 9.4 tall
        w, d, h = A.rest_extents(self.ctx.item(3), self.ctx.rest("side"))
        self.assertEqual((round(w, 3), round(d, 3), round(h, 3)), (0.176, 0.05, 0.094))


class TestMatchedLayout(unittest.TestCase):
    def test_phases_mirror_the_physical_capture(self):
        spec = importlib.util.spec_from_file_location("run_arrangement", REPO / "capture" / "run_arrangement.py")
        ra = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ra)
        self.assertEqual(A.PHASES, ra.PHASES)
        a = _gen(1)
        ph = {p["phase"]: p for p in a["phases"]}
        self.assertEqual([p["phase"] for p in a["phases"]], [p for p, _, _ in ra.PHASES])
        self.assertEqual((ph["reference"]["target_present"], ph["reference"]["occluder_pose"]), (True, "parked"))
        self.assertEqual((ph["visible"]["target_present"], ph["visible"]["occluder_pose"]), (True, "parked"))
        self.assertEqual((ph["occluded"]["target_present"], ph["occluded"]["occluder_pose"]), (True, "placed"))
        self.assertEqual((ph["removed"]["target_present"], ph["removed"]["occluder_pose"]), (False, "placed"))
        for p in a["phases"]:
            self.assertEqual(p["lightings"], ["L1", "L2"])

    def test_removed_is_occluded_minus_the_target(self):
        for seed in SEEDS[:4]:
            a = _gen(seed)
            ph = {p["phase"]: p for p in a["phases"]}
            occ, rem = dict(ph["occluded"]), dict(ph["removed"])
            self.assertEqual(occ["occluder_pose"], rem["occluder_pose"])
            for k in ("phase", "condition", "target_present"):
                occ.pop(k), rem.pop(k)
            self.assertEqual(occ, rem)
            # one occluder object with one placed pose serves both phases
            self.assertEqual(set(a["objects"]["occluder"]["poses"]), {"placed", "parked"})


class TestDesignCases(unittest.TestCase):
    def test_four_option_a_cases(self):
        ctx = _ctx()
        g = ctx.geometry
        want = {"DES0": ("pick_S8", "CCHCC"), "DES1": ("pick_M16+3", "CCHHC"), "DES2": ("pick_M16-5", "CHHCC"),
                "DES3": ("pick_IND28", "HHHHH")}
        self.assertEqual(A.n_design_cases(), 4)
        for i in range(4):
            a = A.design_arrangement(i, ctx=ctx)
            name, pat = want[a["arrangement_id"]]
            o = a["objects"]["occluder"]
            self.assertEqual((a["seed"], a["kind"], a["split"]), (f"design-{i}", "design", "dev"))
            self.assertEqual((o["geometry_occluder"], o["intended_pattern"]), (name, pat))
            box = R.occluder_spec(g, name).box(0.0)
            self.assertTrue(np.allclose(np.asarray(o["size_m"]), 2 * np.asarray(box.half_size), atol=1e-6))
            pb = A.body_of(o, "placed")
            self.assertTrue(np.allclose(pb.centre(), box.centre, atol=1e-6))
            self.assertEqual(a["objects"]["target"]["kind"], "mesh")
            self.assertEqual(a["objects"]["distractors"], [])
            self.assertEqual({m["name"] for m in a["scene"]["meshes"]}, {"mug_body", "mug_handle"})
            self.assertEqual(a["scene"]["mug"]["axis_xy"], [0.0, 0.24])
            self.assertFalse(any(b["role"] == "occluder" for b in a["scene"]["boxes"]))
            self.assertEqual(a["predeclared_indistinguishable"], name == "pick_IND28")
            self.assertEqual(a["target_id"], "study_mug")


class TestPositionsFile(unittest.TestCase):
    def test_per_stop_pitch_loads_and_margins_are_positive(self):
        pos = json.loads(POS.read_text())
        self.assertNotIn("tilt_deg", pos)                    # no silent single-tilt fallback
        rig = R.load_rig(R.DEFAULT_GEOMETRY_PATH, POS)
        self.assertEqual([c.name for c in rig.cameras], ["P1", "P2", "P3", "P4", "P5"])
        for s, cam in zip(pos["stops"], rig.cameras):
            self.assertAlmostEqual(cam.pitch_down_deg, s["tilt_deg"])
            m = R.frame_margins(rig, cam, distort=True)
            self.assertGreater(m["min"], 0.0, cam.name)
            self.assertGreater(R.frame_margins(rig, cam, distort=False)["min"], 0.0, cam.name)
            self.assertAlmostEqual(m["min"], pos["frame_check"][cam.name]["k1_min_margin_px"], delta=0.06)

    def test_same_lens_positions_as_option_a(self):
        draft = R.load_rig(R.DEFAULT_GEOMETRY_PATH, DRAFT_A)
        sim = R.load_rig(R.DEFAULT_GEOMETRY_PATH, POS)
        for a, b in zip(draft.cameras, sim.cameras):
            self.assertTrue(np.allclose(a.position, b.position))
            self.assertAlmostEqual(a.heading_deg, b.heading_deg)
            self.assertAlmostEqual(a.roll_deg, b.roll_deg)
        # the draft's P1 tag 0 was outside the frame (frame check); per-stop pitch brings it in
        self.assertLess(R.frame_margins(draft, draft.cameras[0])["min"], 0.0)

    def test_verify_reads_it(self):
        from verify.schema import Positions

        p = Positions.load(POS)
        self.assertEqual((p.n_stops, p.name(p.start_index), p.f_order), (5, "P3", (0, 1, 3, 4)))
        self.assertAlmostEqual(p.travel_cm(2, 3), math.radians(22.5) * 126.8)

    def test_tilt_fallback_and_override(self):
        d = json.loads(DRAFT_A.read_text())
        g = R.load_geometry()
        self.assertEqual({c.pitch_down_deg for c in R.cameras_from_positions(d, g)}, {6.83})
        d["stops"][0]["tilt_deg"] = 7.5
        self.assertEqual([c.pitch_down_deg for c in R.cameras_from_positions(d, g)], [7.5, 6.83, 6.83, 6.83, 6.83])
        del d["tilt_deg"]
        with self.assertRaises(ValueError):
            R.cameras_from_positions(d, g)


class TestGeometryPrimitives(unittest.TestCase):
    def test_polygons(self):
        sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
        sh = sq + [1, 1]
        self.assertAlmostEqual(A.polygon_area(A.convex_hull(np.vstack([sq, [[1, 1]]]))), 4.0)
        self.assertAlmostEqual(A.overlap_area(sq, sh), 1.0)
        self.assertAlmostEqual(A.overlap_area(sq, sh[::-1]), 1.0)           # either orientation
        self.assertAlmostEqual(A.overlap_area(sq, sq + [3, 0]), 0.0)
        self.assertTrue(A.separated(sq, sq + [2.5, 0], 0.4))
        self.assertFalse(A.separated(sq, sq + [2.5, 0], 0.6))
        self.assertFalse(A.separated(sq, sh))

    def test_body_and_coverage(self):
        b = A.Body((0.1, 0.06, 0.08), (0.0, 0.2, 0.0), 90.0)
        fp = b.footprint()
        self.assertTrue(np.allclose(np.ptp(fp, axis=0), [0.06, 0.1]))
        self.assertEqual(b.corners().shape, (8, 3))
        cam = _ctx().start_cam
        self.assertAlmostEqual(A.coverage(cam, b, b), 1.0)
        front = A.Body((0.3, 0.005, 0.15), (0.0, 0.1, 0.0))
        self.assertAlmostEqual(A.coverage(cam, b, front), 1.0)
        self.assertTrue(A.may_occlude(cam, front, b))
        self.assertFalse(A.may_occlude(cam, b, front) and A.coverage(cam, front, b) == 0.0)
        # hull points: the projected hull of the support points is the projected hull of all the vertices
        rng = np.random.default_rng(0)
        box = A.Body((0.1, 0.06, 0.08), (0.02, 0.2, 0.0), 20.0)
        w = rng.random((300, 8)) + 0.05
        v = np.vstack([box.corners(), (w / w.sum(1, keepdims=True)) @ box.corners()])        # corners + strictly inner points
        sp = A.support_points(v)
        self.assertEqual(sorted(map(tuple, np.round(sp, 9))), sorted(map(tuple, np.round(box.corners(), 9))))
        self.assertAlmostEqual(A.polygon_area(A.projected_hull(cam, sp)), A.polygon_area(A.projected_hull(cam, v)))
        self.assertTrue(np.allclose(A.project(cam, v), cam.project(v)))


if __name__ == "__main__":
    unittest.main()
