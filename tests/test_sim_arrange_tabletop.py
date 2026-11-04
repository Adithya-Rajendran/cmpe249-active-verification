"""sim/arrange.py's tabletop environment adapter and sim/feasibility.py's tabletop scan tools (numpy only).

Pins: the environment switch (config 'environment'; the shelf branch is pinned byte-for-byte by tests/test_shelf_goldens.py),
ID prefixes read from the config (TD / TDES; shelf SD / DES unchanged), environment-salted random streams (equal seeds in
the two environments never share draws; the shelf's entropy lists unchanged), tabletop placements (target whole in frame
with 40 px margin at every stop, no opening test, no ceiling, the occluder on the table, distractors 2-5 upright half
inside the 0.20 m circle with 1 cm / 2 cm clearances and never occluding the target, parking in frame and outside the
target window), the parked-occluder shadow nuisance control (light-centre screen and the analytic area-light ray check
against the target's full height), the tag-free frame check with the workspace outline, the scene spec, the provenance
key for sim/tabletop.py, the TDES design cases, and the height-scan decision rule.

The arrangement tests use tests/fixtures/tabletop/arrange_tabletop_fixture.json (a TEST fixture: the tabletop base config with
every board at 0.20 m, IND28 0.44 m wide, design cases from the margin scan; written by
data/raw/2026-11-04_tabletop_scan/make_tabletop_config.py fixture). The study config is configs/sim/arrange_tabletop_v1.json
(tests/test_sim_tabletop_config.py).
"""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim import arrange as A
from sim import feasibility as F
from sim import rig as R
from sim import tabletop as T

REPO = Path(__file__).resolve().parents[1]
FIXTURE = "tests/fixtures/tabletop/arrange_tabletop_fixture.json"
SHELF_V3 = "configs/sim/arrange_shelf_v3.json"
SHELF_V1 = "configs/sim/arrange_shelf_v1.json"


def _ctx():
    if not hasattr(_ctx, "c"):
        _ctx.c = A.load_context(FIXTURE)
    return _ctx.c


def _cfg():
    return json.loads((REPO / FIXTURE).read_text())


def _first_per_class():
    """{class key: (slot, arrangement, log entry)} for the first dev slot of each class of the fixture, cached."""
    if not hasattr(_first_per_class, "c"):
        ctx = _ctx()
        slots = A.slot_list(ctx, "dev")
        out = {}
        for i in A.first_slot_per_class(slots):
            sl = slots[i]
            arr, e = A.generate_slot(ctx, sl, {})
            out[A.class_key(sl["family"], sl["m16_side"])] = (sl, arr, e)
        _first_per_class.c = out
    return _first_per_class.c


def _designs():
    if not hasattr(_designs, "c"):
        _designs.c = [A.design_arrangement(i, ctx=_ctx()) for i in range(4)]
    return _designs.c


class TestEnvironmentAndConfig(unittest.TestCase):
    def test_tabletop_context(self):
        ctx = _ctx()
        self.assertEqual((ctx.env, ctx.target_comp), ("tabletop", "table"))
        self.assertIsInstance(ctx.rig, T.TabletopScene)
        self.assertEqual([c.name for c in ctx.cams], ["P1", "P2", "P3", "P4", "P5"])
        self.assertEqual({c.roll_deg for c in ctx.cams}, {0.0})
        self.assertEqual(ctx.start_cam.name, "P3")
        self.assertEqual(ctx.comps["table"]["x"], (-0.6, 0.6))
        self.assertEqual(ctx.comps["table"]["z"], (0.0, math.inf))

    def test_shelf_context_is_the_shelf(self):
        ctx = A.load_context(SHELF_V3)
        self.assertEqual((ctx.env, ctx.target_comp), ("shelf", "c3"))
        self.assertIsInstance(ctx.rig, R.Rig)
        self.assertNotIn("tabletop_py", ctx.provenance)
        self.assertNotIn("mug_geometry", ctx.provenance)

    def test_provenance_records_sim_tabletop_py(self):
        p = _ctx().provenance
        self.assertEqual(p["tabletop_py"], {"path": "sim/tabletop.py", "sha256": A.sha256_file(REPO / "sim" / "tabletop.py")})
        self.assertEqual(p["mug_geometry"]["path"], "configs/rig_geometry.json")
        self.assertEqual(p["geometry"]["path"], "configs/sim/tabletop_geometry.json")
        self.assertEqual(p["positions"]["path"], "configs/sim/positions_tabletop_A.json")

    def test_bad_tabletop_configs_are_refused(self):
        base = _cfg()
        bad = []
        for path, v in (("ids.dev_prefix", "SD"), ("ids.design_prefix", "DES"), ("ids.design_prefix", "TD"), ("scene.tags", True),
                        ("environment", "kitchen")):
            c = copy.deepcopy(base)
            k = path.split(".")
            (c[k[0]] if len(k) == 2 else c).__setitem__(k[-1], v)
            bad.append((path, c))
        c = copy.deepcopy(base)
        del c["occluder"]["families"]["S8"]["width_m"]
        bad.append(("S8 width", c))
        c = copy.deepcopy(base)
        c["design_cases"]["cases"][0]["arrangement_id"] = "DES0"
        bad.append(("design id", c))
        for what, c in bad:
            with self.subTest(what=what), self.assertRaises((ValueError, KeyError)):
                A.load_config(c)

    def test_frame_check_uses_the_workspace_outline(self):
        ctx = _ctx()
        cam = ctx.start_cam
        _, arr, _ = _first_per_class()["S8"]
        tp = A.body_of(arr["objects"]["target"], ctx=ctx).shape_points()
        m = A.frame_check(ctx, ctx.rig, cam, tp, True)
        self.assertEqual(set(m), {"workspace", "target", "min"})
        self.assertAlmostEqual(m["workspace"], cam.frame_margin(T.workspace_outline(ctx.geometry), True))
        self.assertAlmostEqual(m["target"], cam.frame_margin(tp, True))
        with self.assertRaises(ValueError):
            A.frame_check(ctx, ctx.rig, cam, None, True)


class TestIdsAndEntropy(unittest.TestCase):
    def test_prefixes_are_read_from_the_config(self):
        self.assertEqual(A.dev_prefix(_cfg()), "TD")
        self.assertEqual(A.dev_prefix(json.loads((REPO / SHELF_V3).read_text())), "SD")
        self.assertEqual(A.dev_prefix({}), "SD")
        for key, (sl, arr, _) in _first_per_class().items():
            self.assertEqual(arr["arrangement_id"], f"TD{arr['seed']:04d}", key)
        self.assertEqual([a["arrangement_id"] for a in _designs()], ["TDES0", "TDES1", "TDES2", "TDES3"])

    def test_a_shelf_config_with_another_prefix_names_its_arrangements_so(self):
        """arrangement_id used to hard-code SD; the shelf's own configs still say SD, so shelf ids are unchanged."""
        cfg = json.loads((REPO / SHELF_V1).read_text())
        cfg["ids"]["dev_prefix"] = "XS"
        a = A.generate(1002, "S8", None, 0, config=cfg)
        self.assertEqual(a["arrangement_id"], "XS1002")
        self.assertEqual(A.generate(1002, "S8", None, 0, config=SHELF_V1)["arrangement_id"], "SD1002")

    def test_shelf_entropy_lists_are_unchanged(self):
        self.assertEqual(A.revisit_entropy(1000), [1000])
        self.assertEqual(A.revisit_entropy("design-3"), [2 ** 31 + 3])
        self.assertEqual(A.try_entropy(1000, 0), [1000])
        self.assertEqual(A.try_entropy(1000, 7), [1000, 7])

    def test_tabletop_streams_are_salted_first(self):
        s = A.ENTROPY_SALT["tabletop"]
        self.assertEqual(A.revisit_entropy(5000, "tabletop"), [s, 5000])
        self.assertEqual(A.revisit_entropy("design-4", "tabletop"), [s, 2 ** 31 + 4])
        self.assertEqual(A.try_entropy(5000, 0, "tabletop"), [s, 5000])
        self.assertEqual(A.try_entropy(5000, 3, "tabletop"), [s, 5000, 3])
        self.assertTrue(s > A.DEV_SEED_RANGE[1] and s < 2 ** 31)        # never a dev seed or a shelf design entropy
        with self.assertRaises(ValueError):
            A.revisit_entropy(5000, "kitchen")
        with self.assertRaises(A.SeedError):
            A.revisit_entropy(110000, "tabletop")

    def test_numpy_ignores_trailing_zeros_so_the_salt_must_lead(self):
        """Why the salt is the FIRST element: SeedSequence([s]) and SeedSequence([s, 0]) are the same stream."""
        self.assertEqual(A.Draw([7]).u(), A.Draw([7, 0]).u())
        self.assertNotEqual(A.Draw([7]).u(), A.Draw([A.ENTROPY_SALT["tabletop"], 7]).u())

    def test_equal_seeds_never_share_draws(self):
        rc = _cfg()["revisit"]
        for seed in (0, 1, 5000, 6000, 7000, 9999):
            for attempt in (0, 1, 20):
                with self.subTest(seed=seed, attempt=attempt):
                    a = A.Draw(A.try_entropy(seed, attempt, "shelf")).rng.random(8)
                    b = A.Draw(A.try_entropy(seed, attempt, "tabletop")).rng.random(8)
                    self.assertFalse(np.any(np.isclose(a, b)))
            for stop in range(5):
                da = A.revisit_draw(rc, A.revisit_entropy(seed, "shelf"), stop, 0)
                db = A.revisit_draw(rc, A.revisit_entropy(seed, "tabletop"), stop, 0)
                self.assertNotEqual(da, db)

    def test_the_arrangement_records_the_salted_entropy(self):
        _, arr, _ = _first_per_class()["S8"]
        self.assertEqual(arr["revisit"]["entropy"], [A.ENTROPY_SALT["tabletop"], arr["seed"]])
        self.assertEqual(arr["revisit"]["stream_salt"], _cfg()["revisit"]["stream_salt"])
        self.assertNotEqual(arr["revisit"]["stream_salt"], json.loads((REPO / SHELF_V3).read_text())["revisit"]["stream_salt"])


class TestTabletopPlacements(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.cfg = self.ctx.config
        self.cl = self.cfg["clearances_m"]

    def _on_table(self, body):
        e = T.table_extent(self.ctx.geometry)
        fp, m = body.footprint(), self.cl["table_edge"] - 1e-9
        self.assertAlmostEqual(body.base[2], 0.0, places=9)
        self.assertGreaterEqual(fp[:, 0].min(), e["x_min"] + m)
        self.assertLessEqual(fp[:, 0].max(), e["x_max"] - m)
        self.assertGreaterEqual(fp[:, 1].min(), e["y_min"] + m)
        self.assertLessEqual(fp[:, 1].max(), e["y_max"] - m)
        self.assertTrue(A.in_front_of_cameras(self.ctx, body))

    def test_every_slot_class(self):
        oc = self.cfg["occluder"]
        got = _first_per_class()
        self.assertEqual(set(got), {"S8", "M16-L", "M16-R", "IND28", "white_box", "ycb_box"})
        for key, (sl, a, _) in got.items():
            with self.subTest(cls=key):
                o = a["objects"]
                t, occ, dis = o["target"], o["occluder"], o["distractors"]
                self.assertEqual((a["environment"], a["occluder_family"], a["m16_side"]), ("tabletop", sl["family"], sl["m16_side"]))
                tb = A.body_of(t, ctx=self.ctx)
                self.assertEqual(t["obj_id"], sl["target"])
                self.assertEqual(t["pose"]["compartment"], "table")
                self._on_table(tb)
                # whole in frame with 40 px margin at every stop, pinhole and k1 (no opening test on the table)
                for cam in self.ctx.cams:
                    self.assertGreaterEqual(cam.frame_margin(tb.shape_points(), False), 40.0)
                    self.assertGreaterEqual(cam.frame_margin(tb.shape_points(), True), 40.0)
                # occluder placed on the table, in front of the target, hiding it at P3, family rules
                pb, kb = A.body_of(occ, "placed", self.ctx), A.body_of(occ, "parked", self.ctx)
                self._on_table(pb)
                self.assertLess(pb.footprint()[:, 1].max(), tb.footprint()[:, 1].min())
                covs = {c.name: A.coverage(c, tb, pb) for c in self.ctx.cams}
                self.assertGreaterEqual(covs["P3"], oc["p3_coverage_min"])
                if sl["family"] == "IND28":
                    self.assertGreaterEqual(min(covs.values()), oc["p3_coverage_min"])
                else:
                    self.assertGreaterEqual(sum(v <= oc["clear_coverage_max"] for v in covs.values()), oc["min_clear_stops"])
                if sl["family"] == "M16":
                    self.assertTrue(A.side_rule_ok(list(covs.values()), 2, sl["m16_side"], oc["families"]["M16"]["side_rule"],
                                                   oc["clear_coverage_max"]))
                if occ["kind"] == "box" and sl["family"] != "white_box":
                    fam = oc["families"][sl["family"]]
                    self.assertEqual(occ["size_m"], [fam["width_m"], fam["thickness_m"], fam["height_m"]])
                    self.assertNotIn("geometry_occluder", occ)
                # distractors: 2-5, distinct, never the target's / occluder's model, regions, clearances, never occluding
                lo, hi = self.cfg["distractors"]["count"]
                self.assertTrue(lo <= len(dis) <= hi)
                ids = [d["obj_id"] for d in dis]
                self.assertEqual(len(set(ids)), len(ids))
                self.assertNotIn(t["obj_id"], ids)
                if occ["kind"] == "ycbv":
                    self.assertNotIn(occ["obj_id"], ids)
                bodies = []
                for d in dis:
                    db = A.body_of(d, ctx=self.ctx)
                    self._on_table(db)
                    self.assertEqual(d["rest"], "upright")
                    r = math.hypot(db.base[0], db.base[1])
                    if d["region"] == "inner":
                        self.assertLessEqual(r, self.cfg["distractors"]["inner_radius_m"] + 1e-9)
                    else:
                        self.assertGreater(r, self.cfg["distractors"]["inner_radius_m"])
                        self.assertGreaterEqual(sum(A.base_in_frame(c, db) for c in self.ctx.cams), 1)
                    for other in [tb, pb] + bodies:
                        self.assertTrue(A.separated(db.footprint(), other.footprint(), self.cl["between_objects"]))
                    self.assertFalse(any(A.may_occlude(c, db, tb) for c in self.ctx.cams))
                    bodies.append(db)
                # parked: on the table, clear, seen, outside the window, hiding nothing, no shadow on the target
                self.assertEqual(occ["poses"]["parked"]["compartment"], "table")
                self._on_table(kb)
                ss = A.shadow_setup(self.ctx, tb)
                self.assertIsNone(A.parked_reason(self.ctx, kb, tb, bodies, ss))
                self.assertFalse(A.shadow_screen(ss.lights, kb, ss.feet))
                self.assertEqual(A.shadow_rays_blocked(ss.lsamples, ss.tsamples, kb), 0)
                self.assertEqual(occ["parking"]["shadow_check"]["rays_blocked"], 0)
                self.assertEqual(set(occ["parking"]["refused"]), set(A.PARK_REASONS))
                # revisit poses pass the tabletop frame check
                for s, st in a["revisit"]["stops"].items():
                    self.assertEqual(set(st["margin_px"]), {"workspace", "target", "min"})
                    self.assertGreaterEqual(st["margin_px"]["min"], 0.0)

    def test_no_ceiling(self):
        """Boards taller than the shelf's #3 clear height and an uncapped white box are accepted on the table."""
        got = _first_per_class()
        self.assertGreater(got["S8"][1]["objects"]["occluder"]["size_m"][2], 0.165)
        self.assertEqual(self.cfg["clearances_m"].get("ceiling"), None)
        tall = A.Body((0.08, 0.005, 0.40), (0.0, -0.2, 0.0), 0.0)
        self.assertTrue(A.inside(self.ctx, tall, "c3"))                 # comp ignored on the table: no compartment, no ceiling

    def test_off_table_and_behind_camera_refused(self):
        e = T.table_extent(self.ctx.geometry)
        for base in ((e["x_max"] - 0.01, 0.0, 0.0), (0.0, e["y_min"] + 0.005, 0.0), (0.0, 0.0, 0.01)):
            self.assertFalse(A.on_table(self.ctx, A.Body((0.05, 0.05, 0.05), base, 0.0)), base)
        self.assertTrue(A.on_table(self.ctx, A.Body((0.05, 0.05, 0.05), (0.0, 0.0, 0.0), 0.0)))
        p1 = self.ctx.rig.camera("P1")
        _, _, fwd = p1.axes()
        behind = np.asarray(p1.position) - 0.1 * np.array([fwd[0], fwd[1], 0.0]) / np.hypot(fwd[0], fwd[1])
        self.assertFalse(A.in_front_of_cameras(self.ctx, A.Body((0.02, 0.02, 0.02), (behind[0], behind[1], p1.position[2] - 0.01), 0.0)))

    def test_seen_whole_has_no_opening_test(self):
        mug = A.Body((0.06, 0.06, 0.10), (0.0, 0.0, 0.0), 0.0)
        self.assertTrue(all(A.seen_whole(self.ctx, c, mug) for c in self.ctx.cams))
        corner = A.Body((0.06, 0.06, 0.10), (0.5, 0.4, 0.0), 0.0)
        self.assertFalse(all(A.seen_whole(self.ctx, c, corner) for c in self.ctx.cams))

    def test_half_the_distractors_stand_inside_the_circle(self):
        """p_inner = 0.5 per try: over the first slots the regions are mixed (a weak check of the draw, not a rate)."""
        regions = [d["region"] for _, a, _ in _first_per_class().values() for d in a["objects"]["distractors"]]
        self.assertIn("inner", regions)
        self.assertIn("outer", regions)
        self.assertEqual(self.cfg["distractors"]["p_inner"], 0.5)
        self.assertEqual(self.cfg["distractors"]["inner_radius_m"], self.ctx.geometry["workspace"]["radius"])

    def test_deterministic(self):
        sl, a, _ = _first_per_class()["S8"]
        b, _ = A.generate_slot(self.ctx, sl, {})
        self.assertEqual(a["arrangement_sha256"], b["arrangement_sha256"])
        self.assertEqual(A.arrangement_hash(a), a["arrangement_sha256"])

    def test_scene_spec(self):
        _, a, _ = _first_per_class()["M16-L"]
        sc = a["scene"]
        self.assertEqual((sc["environment"], sc["tags"], sc["tags_enabled"], sc["mug"], sc["occluder"]), ("tabletop", [], False, None, None))
        self.assertEqual(sc["provenance"]["geometry_file"], "configs/sim/tabletop_geometry.json")
        self.assertEqual(sc["provenance"]["positions_file"], "configs/sim/positions_tabletop_A.json")
        self.assertIn("lighting", sc)
        self.assertEqual(set(sc["materials"]) >= {"occluder", "white_box", "table_top"}, True)
        self.assertEqual([c["name"] for c in sc["revisit_cameras"]], a["stops"])
        for c, st in zip(sc["revisit_cameras"], a["revisit"]["stops"].values()):
            self.assertEqual(c["position"], st["pose"]["position"])
        self.assertEqual([m["name"] for m in sc["meshes"]], ["table_top"])


class TestShadow(unittest.TestCase):
    LIGHT = {"name": "L", "position": [0.0, -1.0, 1.4], "forward": list(-np.array([0.0, -1.0, 1.35]) / np.linalg.norm([0.0, -1.0, 1.35])),
             "size_m": 0.4, "on": True}

    def test_light_samples(self):
        p = A.light_samples(self.LIGHT, 5)
        self.assertEqual(p.shape, (25, 3))
        np.testing.assert_allclose(p.mean(0), self.LIGHT["position"], atol=1e-12)
        f = np.asarray(self.LIGHT["forward"])
        np.testing.assert_allclose((p - p.mean(0)) @ f, 0.0, atol=1e-12)              # on the emitting plane
        d = np.linalg.norm(p[0] - p[-1])
        self.assertAlmostEqual(d, 0.4 * math.sqrt(2))                                 # opposite corners of the 0.4 m square

    def test_scene_lights_are_the_geometry_lights_on_in_some_preset(self):
        ls = A.scene_lights(_ctx())
        self.assertEqual([lt["name"] for lt in ls], ["key_left", "key_right"])
        self.assertEqual([lt["position"] for lt in ls], [[-0.6, -1.0, 1.4], [0.6, -1.0, 1.4]])

    def test_target_samples_span_the_full_height(self):
        t = A.Body((0.06, 0.08, 0.10), (0.1, 0.05, 0.0), 30.0)
        s = A.target_shadow_samples(t, 0.01, 6)
        self.assertAlmostEqual(s[:, 2].min(), 0.0)
        self.assertAlmostEqual(s[:, 2].max(), 0.10)
        self.assertEqual(len(np.unique(np.round(s[:, 2], 9))), 6)
        hull = A.convex_hull(t.footprint())
        self.assertTrue(np.all(A._inside_convex(hull, s[:, :2] + (s[:, :2] - [0.1, 0.05]) * -1e-9)))
        top = s[np.isclose(s[:, 2], 0.10)]
        self.assertGreater(len(top), 4 * 6)                                           # outline + interior grid on the top

    def test_segments_hit_box(self):
        box = A.Body((0.2, 0.1, 0.3), (0.0, 0.0, 0.0), 0.0)
        p0 = np.array([[-1, 0, 0.1], [-1, 0, 0.5], [0, -1, 0.15], [0, 0, 1], [-1, 0.2, 0.1], [0.05, 0.0, 0.1], [-1, 0.0, 0.3]], float)
        p1 = np.array([[1, 0, 0.1], [1, 0, 0.5], [0, 1, 0.15], [0, 0, 0.31], [1, 0.2, 0.1], [0.06, 0.0, 0.1], [1, 0.0, 0.3]], float)
        np.testing.assert_array_equal(A.segments_hit_box(p0, p1, box), [True, False, True, False, False, True, True])
        yawed = A.Body((0.4, 0.02, 0.3), (0.0, 0.0, 0.0), 90.0)                      # long side now along y
        np.testing.assert_array_equal(A.segments_hit_box(np.array([[-1, 0.15, 0.1]]), np.array([[1, 0.15, 0.1]]), yawed), [True])
        np.testing.assert_array_equal(A.segments_hit_box(np.array([[0.15, -1, 0.1]]), np.array([[0.15, 1, 0.1]]), yawed), [False])

    def test_board_between_light_and_target_is_flagged_by_both(self):
        target = A.Body((0.06, 0.06, 0.10), (0.0, 0.0, 0.0), 0.0)
        lt = self.LIGHT
        board = A.Body((0.3, 0.01, 0.6), (0.0, -0.3, 0.0), 0.0)                    # tall, straight toward the light
        ls, ts = [A.light_samples(lt, 5)], A.target_shadow_samples(target, 0.01, 6)
        self.assertGreater(A.shadow_rays_blocked(ls, ts, board), 0)
        self.assertTrue(A.shadow_screen([lt], board, [A.convex_hull(target.footprint())]))
        behind = A.Body((0.3, 0.01, 0.6), (0.0, 0.3, 0.0), 0.0)                    # behind the target, away from the light
        self.assertEqual(A.shadow_rays_blocked(ls, ts, behind), 0)
        self.assertFalse(A.shadow_screen([lt], behind, [A.convex_hull(target.footprint())]))

    def test_the_ray_check_catches_what_the_light_centre_screen_misses(self):
        """An area light: a post that only the light's edge sees past onto the target. The centre's shadow misses the target's
        footprint, the rays from the light's left part hit the target."""
        target = A.Body((0.06, 0.06, 0.10), (0.0, 0.0, 0.0), 0.0)
        lt = self.LIGHT
        post = A.Body((0.03, 0.03, 0.5), (0.09, -0.25, 0.0), 0.0)
        foot = [A.convex_hull(target.footprint())]
        self.assertFalse(A.shadow_screen([lt], post, foot))
        self.assertGreater(A.shadow_rays_blocked([A.light_samples(lt, 5)], A.target_shadow_samples(target, 0.01, 6), post), 0)
        self.assertEqual(A.shadow_rays_blocked([A.light_samples(lt, 1)], A.target_shadow_samples(target, 0.01, 6), post), 0)

    def test_full_height_matters(self):
        """A low block shades the target's lower part only: sampling the target's top alone would miss it."""
        target = A.Body((0.06, 0.06, 0.10), (0.0, 0.0, 0.0), 0.0)
        ls = [A.light_samples(self.LIGHT, 5)]
        low = A.Body((0.2, 0.02, 0.03), (0.0, -0.06, 0.0), 0.0)
        full = A.target_shadow_samples(target, 0.01, 6)
        top = full[np.isclose(full[:, 2], 0.10)]
        self.assertGreater(A.shadow_rays_blocked(ls, full, low), 0)
        self.assertEqual(A.shadow_rays_blocked(ls, top, low), 0)

    def test_window_footprint_round_trip(self):
        ctx = _ctx()
        target = A.Body((0.06, 0.06, 0.10), (0.0, 0.0, 0.0), 0.0)
        for cam in ctx.cams:
            win = A.target_window(cam, target, 1.0)
            foot = A.window_footprint(cam, win)
            uv = A.project(cam, np.c_[foot, np.zeros(len(foot))])
            np.testing.assert_allclose(np.sort(uv, 0), np.sort(win, 0), atol=1e-6)
            self.assertTrue(A._inside_convex(foot, np.array([[0.0, 0.0]]))[0])          # the target's base is in its window

    def test_shadow_polygon(self):
        sh = A.shadow_polygon([0.0, 0.0, 2.0], np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
        self.assertTrue(any(np.allclose(p, [2.0, 0.0]) for p in sh))
        with self.assertRaises(ValueError):
            A.shadow_polygon([0.0, 0.0, 1.0], np.array([[0.0, 0.0, 1.5]]))


class TestDesignCases(unittest.TestCase):
    def test_four_cases(self):
        want = [("TDES0", "design-4", "S8", None, "CCHCC"), ("TDES1", "design-5", "M16", "R", "CCHHC"),
                ("TDES2", "design-6", "M16", "L", "CHHCC"), ("TDES3", "design-7", "IND28", None, "HHHHH")]
        ctx = _ctx()
        pk = ctx.config["design_cases"]["parked"]
        for a, (aid, seed, fam, side, pat) in zip(_designs(), want):
            with self.subTest(case=aid):
                o = a["objects"]["occluder"]
                self.assertEqual((a["arrangement_id"], a["seed"], a["occluder_family"], a["m16_side"], a["kind"], a["design_case"]),
                                 (aid, seed, fam, side, "design", True))
                self.assertEqual(a["predeclared_indistinguishable"], fam == "IND28")
                cov = a["analytic"]["placed_coverage"]
                got = "".join("H" if cov[s] >= 0.95 else ("C" if cov[s] <= 0.001 else "p") for s in a["stops"])
                self.assertEqual(got, pat)
                self.assertEqual(o["intended_pattern"], pat)
                self.assertGreater(o["design_margin_px"], 0)
                self.assertEqual(o["poses"]["parked"]["base"][:2], [pk["x"], pk["y"]])
                self.assertEqual(a["objects"]["distractors"], [])
                self.assertEqual(a["revisit"]["entropy"], [A.ENTROPY_SALT["tabletop"], 2 ** 31 + int(seed[7:])])
                self.assertEqual(a["scene"]["mug"]["handle"], "toward")
                self.assertEqual([m["role"] for m in a["scene"]["meshes"]], ["table", "target", "target"])
                self.assertIn("mug", a["scene"]["labels"])
                t = a["objects"]["target"]
                self.assertEqual((t["kind"], t["mug"]["x"], t["mug"]["y"], t["pose"]["base"][0]), ("mesh", 0.0, 0.0, 0.0))

    def test_handle_points_toward_p3(self):
        mug, meshes = A.design_mug(_ctx())
        handle = [m for m in meshes if m.name == "mug_handle"][0].world_vertices()
        body = [m for m in meshes if m.name == "mug_body"][0].world_vertices()
        self.assertLess(handle[:, 1].min(), body[:, 1].min() - 0.01)                  # the handle sticks out toward -y (P3)

    def test_stored_gap_and_shift_are_the_margin_scan_result(self):
        ctx = _ctx()
        case = ctx.config["design_cases"]["cases"][0]
        best = A.design_margin_scan(ctx, case["family"], case["intended_pattern"])
        self.assertEqual((best["gap_m"], best["shift_m"], best["margin_px"]), (case["gap_m"], case["shift_m"], case["margin_px"]))

    def test_a_wrong_gap_is_refused(self):
        cfg = _cfg()
        cfg["design_cases"]["cases"][0]["gap_m"] = 0.25
        cfg["design_cases"]["cases"][0]["shift_m"] = 0.15
        with self.assertRaises(ValueError):
            A.design_arrangement(0, config=cfg)

    def test_pattern_margins_signs(self):
        ctx = _ctx()
        mug, _ = A.design_mug(ctx)
        far = A.Body((0.08, 0.005, 0.2), (0.4, -0.2, 0.0), 0.0)
        self.assertTrue(all(m > 0 for m in A.pattern_margins(ctx, mug, far, "CCCCC")))
        self.assertTrue(all(m < 0 for m in A.pattern_margins(ctx, mug, far, "HHHHH")))
        with self.assertRaises(ValueError):
            A.pattern_margins(ctx, mug, far, "CCHC")


def _results(plan, table):
    """Synthetic scan results: table[(family, side, target, height, width)] = successes (default 0)."""
    rows = []
    for c in F.scan_cells(plan):
        k = table.get((c["family"], c["side"], c["target"], c["height_m"], c["width_m"]), 0)
        if c["family"] == "ycb_box":
            k = table.get(("ycb_box", tuple(c["choice"]), c["target"]), 0)
        rows.append({**{x: v for x, v in c.items() if x != "overrides"}, "successes": k, "tries": plan["tries_per_cell"]})
    return {"plan": {"sha256": "x"}, "n_cells": len(rows), "rows": rows}


class TestHeightScanRule(unittest.TestCase):
    PLAN = {"heights_m": [0.15, 0.2, 0.25], "board_families": ["S8", "M16", "IND28"], "m16_sides": ["L", "R"],
            "ind28_widths_m": [0.28, 0.36], "targets": [21, 6, 14], "ycb_box_choices": [[2, "upright"], [7, "edge"]],
            "tries_per_cell": 50, "decision_rule": {"alpha": 0.05, "m16_must_admit": [21, 6]}}

    def _base(self):
        t = {}
        for h in self.PLAN["heights_m"]:
            for s in ("L", "R"):
                t[("M16", s, 21, h, None)] = 10
                t[("M16", s, 6, h, None)] = 10
            for tid in (21, 6, 14):
                t[("S8", None, tid, h, None)] = 20
                t[("IND28", None, tid, h, 0.36)] = 20
        return t

    def test_cells(self):
        cells = F.scan_cells(self.PLAN)
        self.assertEqual(len(cells), 3 * (3 + 6 + 6) + 3 + 2 * 3)
        self.assertEqual([c["cell"] for c in cells], list(range(len(cells))))
        ind = [c for c in cells if c["family"] == "IND28"][0]
        self.assertEqual(ind["overrides"]["occluder.families.IND28.width_m"], 0.28)
        self.assertEqual(ind["overrides"]["occluder.families.S8.height_m"], 0.15)
        y = [c for c in cells if c["family"] == "ycb_box"][0]
        self.assertEqual(y["overrides"]["occluder.families.ycb_box.choices"], [[2, "upright"]])

    def test_smallest_height_meeting_both(self):
        d = F.decide_height(self.PLAN, _results(self.PLAN, self._base()))
        self.assertEqual((d["H_T_m"], d["ind28_width_m"]), (0.15, 0.36))              # 0.28 admits nothing: the wider variant

    def test_a_missing_m16_target_moves_the_choice(self):
        t = self._base()
        t[("M16", "R", 6, 0.15, None)] = 0
        self.assertEqual(F.decide_height(self.PLAN, _results(self.PLAN, t))["H_T_m"], 0.2)

    def test_identities_and_rates_both_count(self):
        t = self._base()
        t[("S8", None, 14, 0.15, None)] = 0                       # identity lost at 0.15
        self.assertEqual(F.decide_height(self.PLAN, _results(self.PLAN, t))["H_T_m"], 0.2)
        t = self._base()
        t[("S8", None, 14, 0.25, None)] = 40                      # best rate at 0.25: 20/50 vs 40/50 is significantly lower
        d = F.decide_height(self.PLAN, _results(self.PLAN, t))
        self.assertEqual(d["H_T_m"], 0.25)
        self.assertFalse(d["per_height"]["0.15"]["s8_detail"]["14"]["retained"])
        t[("S8", None, 14, 0.25, None)] = 24                      # 20/50 vs 24/50: not significantly lower
        self.assertEqual(F.decide_height(self.PLAN, _results(self.PLAN, t))["H_T_m"], 0.15)

    def test_no_height(self):
        t = self._base()
        for h in self.PLAN["heights_m"]:
            t[("M16", "L", 21, h, None)] = 0
        d = F.decide_height(self.PLAN, _results(self.PLAN, t))
        self.assertIsNone(d["H_T_m"])
        self.assertEqual(d["outcome"], "no height satisfies (a) and (b)")

    def test_ycb_choices(self):
        t = self._base()
        t[("ycb_box", (2, "upright"), 21)] = 3
        d = F.decide_height(self.PLAN, _results(self.PLAN, t))
        self.assertEqual((d["ycb_box"]["2:upright"]["kept"], d["ycb_box"]["7:edge"]["kept"]), (True, False))

    def test_fisher(self):
        self.assertAlmostEqual(F.fisher_one_sided_lower(0, 10, 5, 10), math.comb(10, 5) / math.comb(20, 5))
        self.assertAlmostEqual(F.fisher_one_sided_lower(5, 10, 5, 10), 0.5 + math.comb(10, 5) ** 2 / math.comb(20, 10) / 2)
        self.assertEqual(F.fisher_one_sided_lower(10, 10, 0, 10), 1.0)

    def test_plan_checks(self):
        with tempfile.TemporaryDirectory() as td:
            cfgp = Path(td) / "base_config.json"
            cfgp.write_text((REPO / FIXTURE).read_text())
            plan = {"schema": F.PLAN_SCHEMA, "base_config": {"file": "base_config.json", "sha256": A.sha256_file(cfgp)},
                    "probe_seeds": {"start": 7000, "n": 100}, "class_matrix_probe_seeds": {"start": 7200, "n": 100}}
            self.assertEqual(F.check_plan(plan, Path(td))["environment"], "tabletop")
            for bad in ({"probe_seeds": {"start": 5050, "n": 20}}, {"probe_seeds": {"start": 109990, "n": 20}},
                        {"base_config": {"file": "base_config.json", "sha256": "0" * 64}}):
                with self.subTest(bad=bad), self.assertRaises((ValueError, A.SeedError)):
                    F.check_plan({**plan, **bad}, Path(td))

    def test_board_search_is_environment_aware(self):
        r = F.board_search(_cfg(), 21, "R", [0.2], 1)
        self.assertEqual((r["name"], r["side"], r["n_placements"]), ("foam_brick", "R", 1))
        self.assertGreaterEqual(r["heights"]["0.2"]["combos"], r["heights"]["0.2"]["placements_feasible"])


if __name__ == "__main__":
    unittest.main()
