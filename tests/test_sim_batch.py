"""Tests for the pure parts of sim/isaac_batch.py, for sim/episodes.py, and for the batch's control flow against a stub Isaac
(tests/fake_isaac.py). numpy and cv2 only; no Isaac Sim, no real images.

Pins: the dev/test guard in the renderer, pose and asset transforms agreeing with sim/arrange.py's Body, segmentation ->
class maps, mask boxes, the visibility / diagnostic labels taken from the perturbed (revisit) renders, atomic writing and
--skip-existing by arrangement hash, the globbed command over the generator's output folder, and the evidence annotation,
jobs (primary refL1 and secondary same-lighting), manifest (simulation flag, family / side / design_case, separate design
manifest), the incomplete-folder, mixed-folder and wrong-positions refusals.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from sim import arrange as A
from sim import check_batch as CB
from sim import episodes as E
from sim import isaac_batch as B
from tests import fake_isaac as FI
from verify.manifest import RE_ARRANGEMENT, load_manifest
from verify.schema import Condition, Lighting, episode_id

STOPS = ["P1", "P2", "P3", "P4", "P5"]
REPO = Path(__file__).resolve().parents[1]
POS = REPO / "configs" / "sim" / "positions_shelf_A.json"
V1 = "configs/sim/arrange_shelf_v1.json"       # the shelf_v1 config: these tests use the legacy (draw-per-try) generator path


def _labels(arr_id: str, seed, target: str, arr_bytes: bytes, ind: bool = False, family: str = "S8", side=None,
            design: bool = False) -> dict:
    vf = {"reference": {s: 1.0 for s in STOPS}, "visible": {s: 1.0 for s in STOPS},
          "occluded": dict(zip(STOPS, [1.0, 0.95, 0.0, 0.4, 1.0]))}
    obst = {"occluded": dict(zip(STOPS, [0.0, 0.05, 1.0, 0.6, 0.0])), "removed": dict(zip(STOPS, [0.0, 0.05, 1.0, 0.6, 0.0]))}
    diag = {"visible": {s: True for s in STOPS}, "occluded": dict(zip(STOPS, [True, True, False, False, True])),
            "removed": dict(zip(STOPS, [True, True, False, False, True]))}
    boxes = {L: {s: [100 + i, 200, 180 + i, 260] for i, s in enumerate(STOPS)} for L in ("L1", "L2")}
    return {"schema": B.LABELS_SCHEMA, "arrangement_id": arr_id, "seed": seed, "split": "dev", "target_id": target,
            "stops": STOPS, "lightings": ["L1", "L2"], "visible_fraction": vf, "region_obstruction": obst,
            "diagnostic": diag, "n_diagnostic": {k: sum(v.values()) for k, v in diag.items()},
            "patterns": {"reference": "CCCCC", "visible": "CCCCC", "occluded": "CCHpC"},
            "truth": dict(B.CONDITION_TRUTH), "reference_box": boxes, "query_box_visible": boxes, "rules": {"present": "x"},
            "occluder_family": family, "m16_side": side, "design_case": design, "predeclared_indistinguishable": ind,
            "hashes": {"arrangement_file_sha256": hashlib.sha256(arr_bytes).hexdigest(), "arrangement_sha256": "a" * 64}}


def _arr_doc(arr_id, seed, kind, target, ind, family="S8", side=None, design=False, pos_sha=None):
    revisit = {"schema": "cmpe249.sim-revisit.v0", "stops": {s: {"delta": {"dx_m": 0.001, "dy_m": 0.0, "dz_m": 0.0, "dyaw_deg": 0.1,
                                                                            "dpitch_deg": 0.0, "droll_deg": 0.0},
                                                                  "attempt": 0, "n_rejected": 0, "margin_px": {"min": 12.0}}
                                                      for s in STOPS}}
    return {"schema": A.ARRANGEMENT_SCHEMA, "arrangement_id": arr_id, "seed": seed, "split": "dev", "kind": kind,
            "target_id": target, "predeclared_indistinguishable": ind, "occluder_family": family, "m16_side": side,
            "design_case": design,
            "objects": {"occluder": {"family": family, "kind": "box", "label": family}},
            "provenance": {"generator": {"sha256": "g" * 64}, "config": {"sha256": "c" * 64},
                           "positions": {"path": "configs/sim/positions_shelf_A.json",
                                         "sha256": pos_sha or A.sha256_file(POS)}},
            "revisit": revisit, "arrangement_sha256": "a" * 64}


def _write_set(root: Path, items, pos_sha=None) -> None:
    for item in items:
        arr_id, seed, kind, target, ind = item[:5]
        extra = item[5] if len(item) > 5 else {}
        d = root / arr_id
        d.mkdir(parents=True)
        arr = _arr_doc(arr_id, seed, kind, target, ind, pos_sha=pos_sha, **extra)
        b = json.dumps(arr).encode()
        (d / "arrangement.json").write_bytes(b)
        (d / "labels.json").write_text(json.dumps(_labels(arr_id, seed, target, b, ind, **extra)))


class TestRendererGuard(unittest.TestCase):
    def _arr(self, **kw):
        a = {"schema": A.ARRANGEMENT_SCHEMA, "arrangement_id": "SD0001", "seed": 1, "split": "dev", "kind": "random"}
        a.update(kw)
        return a

    def test_constants_match_the_generator(self):
        self.assertEqual(B.DEV_SEED_RANGE, A.DEV_SEED_RANGE)
        self.assertEqual(B.TEST_SEED_MIN, A.TEST_SEED_MIN)
        self.assertEqual(B.PHASE_ORDER, tuple(p for p, _, _ in A.PHASES))
        self.assertEqual(B.ARRANGEMENT_SCHEMA, A.ARRANGEMENT_SCHEMA)
        self.assertEqual(B.NON_ARRANGEMENT_SCHEMAS, A.NON_ARRANGEMENT_SCHEMAS)
        self.assertEqual(E.LABELS_SCHEMA, B.LABELS_SCHEMA)
        self.assertEqual(set(B.CAMERA_SETS), set(A.PHASE_CAMERA.values()))

    def test_dev_and_design_pass(self):
        B.check_arrangement(self._arr())
        B.check_arrangement(self._arr(seed=9999))
        B.check_arrangement(self._arr(seed="design-3", kind="design", arrangement_id="DES3"))

    def test_test_and_other_seeds_refused(self):
        for kw in ({"seed": 100000}, {"seed": 123456}, {"seed": 10000}, {"seed": -1}, {"seed": "7"}, {"seed": True},
                   {"split": "test"}, {"seed": "design-1"}, {"schema": "other"}, {"schema": "cmpe249.sim-arrangement.v0"}):
            with self.subTest(**kw), self.assertRaises(B.RefusedArrangement):
                B.check_arrangement(self._arr(**kw))
        with self.assertRaisesRegex(B.RefusedArrangement, "TEST"):
            B.check_arrangement(self._arr(seed=100000))

    def test_scene_needs_both_camera_sets(self):
        arr = FI.arrangement()
        B.check_scene(arr)
        for mutate in (lambda a: a["scene"].pop("revisit_cameras"), lambda a: a["scene"]["revisit_cameras"].pop(),
                       lambda a: a["scene"]["revisit_cameras"].reverse(), lambda a: a["phases"][1].update(camera="other")):
            bad = json.loads(json.dumps(arr))
            mutate(bad)
            with self.assertRaises(B.RefusedArrangement):
                B.check_scene(bad)

    def test_glob_over_the_generators_folder_is_harmless(self):
        """The documented command `isaac_batch.py --arrangements ARR/*.json` must work on `python -m sim.arrange --out ARR`."""
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "ARR"
            with contextlib.redirect_stdout(io.StringIO()):
                A.main(["--config", V1, "--slots", "10", "--design", "--out", str(out)])
            files = sorted(str(p) for p in out.glob("*.json"))
            self.assertEqual(len(files), 5)
            items, skipped = B.load_arrangement_files(files)
            self.assertEqual([a["arrangement_id"] for a, _ in items], ["DES0", "DES1", "DES2", "DES3", "SD1010"])
            self.assertEqual(skipped, [])
            # the index files, if they are passed anyway, are skipped rather than refusing the batch
            idx = sorted(str(p) for p in (out / "index").glob("*.json"))
            items2, skipped2 = B.load_arrangement_files(files + idx)
            self.assertEqual(len(items2), 5)
            self.assertEqual(sorted(s for _, s in skipped2), sorted(A.NON_ARRANGEMENT_SCHEMAS))
            for a, _ in items:
                B.check_scene(a)
            # a test arrangement among them still refuses the whole batch
            bad = out / "SD100000.json"
            bad.write_text(json.dumps({**items[-1][0], "arrangement_id": "SD100000", "seed": 100000}))
            with self.assertRaisesRegex(B.RefusedArrangement, "TEST"):
                B.load_arrangement_files(files + [str(bad)])


class TestTransforms(unittest.TestCase):
    def test_box_and_pose_match_body_corners(self):
        body = A.Body((0.16, 0.005, 0.12), (0.03, 0.0325, 0.0), 25.0)
        m = B.box_local_matrix(body.size) @ B.pose_matrix(body.base, body.yaw_deg)
        cube = np.array([[x, y, z, 1.0] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        world = (cube @ m)[:, :3]
        want = body.corners()
        self.assertTrue(all(np.min(np.linalg.norm(want - w, axis=1)) < 1e-9 for w in world))

    def test_asset_local_matches_generator_frame(self):
        ctx = A.load_context(V1)
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

    @staticmethod
    def _square(r0, c0, n=10, h=20, w=30):
        m = np.zeros((h, w), bool)
        m[r0:r0 + n, c0:c0 + n] = True
        return m

    def _scene(self, hidden, shift=3):
        """Reference at the nominal pose (a 10 x 10 square at cols 5-14); the query pose is `shift` columns to the right.
        hidden[i] query-target pixels are covered by the occluder at stop i."""
        alone_ref = {s: self._square(5, 5) for s in STOPS}
        alone_q = {s: self._square(5, 5 + shift) for s in STOPS}
        tgt, occ = {}, {}
        for s, hid in zip(STOPS, hidden):
            tgt[("reference", s)] = alone_ref[s].copy()
            tgt[("visible", s)] = alone_q[s].copy()
            t = alone_q[s].copy()
            o = np.zeros_like(t)
            idx = np.flatnonzero(t.reshape(-1))[:hid]
            t.reshape(-1)[idx] = False
            o.reshape(-1)[idx] = True
            tgt[("occluded", s)] = t
            occ[("occluded", s)] = o
            occ[("removed", s)] = o.copy()
        return tgt, alone_ref, alone_q, occ

    def test_compute_labels_at_the_query_pose(self):
        tgt, alone_ref, alone_q, occ = self._scene([0, 5, 100, 50, 11])
        lab = B.compute_labels(STOPS, tgt, alone_ref, alone_q, occ)
        self.assertEqual(lab["visible_fraction"]["occluded"], dict(zip(STOPS, [1.0, 0.95, 0.0, 0.5, 0.89])))
        self.assertEqual(lab["visible_fraction"]["visible"], {s: 1.0 for s in STOPS})     # visible == its own alone mask
        self.assertEqual(lab["visible_fraction"]["reference"], {s: 1.0 for s in STOPS})
        self.assertEqual(lab["patterns"]["occluded"], "CCHpp")
        self.assertEqual(lab["diagnostic"]["occluded"], dict(zip(STOPS, [True, True, False, False, False])))
        self.assertEqual(lab["region_obstruction"]["removed"], dict(zip(STOPS, [0.0, 0.05, 1.0, 0.5, 0.11])))
        self.assertEqual(lab["region_obstruction"]["occluded"], lab["region_obstruction"]["removed"])
        self.assertEqual(lab["diagnostic"]["removed"], dict(zip(STOPS, [True, True, False, False, False])))
        self.assertEqual(lab["n_diagnostic"], {"visible": 5, "occluded": 2, "removed": 2})
        self.assertEqual(lab["truth"], {"visible": "present", "occluded": "present", "removed": "removed"})
        self.assertEqual(lab["region_px"], {s: 100 for s in STOPS})
        self.assertEqual(lab["alone_px"], {"reference": {s: 100 for s in STOPS}, "query": {s: 100 for s in STOPS}})
        self.assertFalse(lab["indistinguishable_by_masks"])
        self.assertIn("revisit", lab["rules"]["poses"])
        for s in STOPS:                                       # cover every stop -> no diagnostic view in either
            occ[("removed", s)] = alone_q[s].copy()
            tgt[("occluded", s)] = np.zeros_like(alone_q[s])
        self.assertTrue(B.compute_labels(STOPS, tgt, alone_ref, alone_q, occ)["indistinguishable_by_masks"])

    def test_visible_fraction_uses_the_denominator_at_the_same_pose(self):
        tgt, alone_ref, alone_q, occ = self._scene([20] * 5, shift=3)
        ok = B.compute_labels(STOPS, tgt, alone_ref, alone_q, occ)["visible_fraction"]["occluded"]
        self.assertEqual(ok, {s: 0.8 for s in STOPS})                                      # 80 / 100 at the query pose
        wrong = B.compute_labels(STOPS, tgt, alone_ref, alone_ref, occ)["visible_fraction"]["occluded"]
        self.assertNotEqual(wrong["P3"], ok["P3"])                                         # the nominal denominator would differ
        # the reference view is measured at the nominal pose
        lab = B.compute_labels(STOPS, tgt, alone_ref, alone_q, occ)
        self.assertEqual(lab["visible_fraction"]["reference"]["P3"], 1.0)

    def test_removed_region_comes_from_the_visible_capture_not_the_reference_mask(self):
        # nominal region: cols 5-14; query region: cols 8-17. An occluder at cols 5-9 (rows 5-14) covers 50 % of the nominal
        # region but only 20 % of the query region (cols 8-9), which is where the removed view's target would be.
        tgt, alone_ref, alone_q, occ = self._scene([0] * 5, shift=3)
        o = np.zeros_like(alone_ref["P3"])
        o[5:15, 5:10] = True
        for s in STOPS:
            occ[("removed", s)] = o.copy()
            occ[("occluded", s)] = o.copy()
        lab = B.compute_labels(STOPS, tgt, alone_ref, alone_q, occ)
        self.assertEqual(lab["region_obstruction"]["removed"]["P3"], 0.2)
        self.assertNotEqual(lab["region_obstruction"]["removed"]["P3"], 0.5)               # the nominal-mask value
        self.assertFalse(lab["diagnostic"]["removed"]["P3"])                              # 0.2 >= 0.10
        # an occluder entirely outside the query region (cols 5-7) obstructs nothing, although it covers 30 % of the nominal one
        o2 = np.zeros_like(o)
        o2[5:15, 5:8] = True
        occ[("removed", "P3")] = o2
        lab = B.compute_labels(STOPS, tgt, alone_ref, alone_q, occ)
        self.assertEqual(lab["region_obstruction"]["removed"]["P3"], 0.0)
        self.assertTrue(lab["diagnostic"]["removed"]["P3"])

    def test_region_denominator_is_the_visible_phase_mask_at_the_revisit_pose(self):
        """Revisit noise changes the target's area between the reference (nominal pose) and the query poses. The reference
        mask here has 100 px, the visible-phase mask and the target-alone mask at the revisit pose 64 px (an 8 x 8 square). The
        occluder covers 16 of those 64 px: the share is 16 / 64 = 0.25 and region_px is 64, not 16 / 100 and 100."""
        sq = self._square
        alone_ref = {s: sq(5, 5, n=10) for s in STOPS}
        alone_q = {s: sq(5, 8, n=8) for s in STOPS}
        tgt, occ = {}, {}
        for s in STOPS:
            tgt[("reference", s)] = alone_ref[s].copy()
            tgt[("visible", s)] = alone_q[s].copy()
            o = np.zeros_like(alone_q[s])
            o[5:13, 8:10] = True                      # columns 8-9 of the 8 x 8 square: 8 rows x 2 columns = 16 px
            occluded = alone_q[s] & ~o
            tgt[("occluded", s)] = occluded
            occ[("occluded", s)] = o
            occ[("removed", s)] = o.copy()
        self.assertEqual(int(np.count_nonzero(o)), 16)
        lab = B.compute_labels(STOPS, tgt, alone_ref, alone_q, occ)
        self.assertEqual(lab["region_px"], {s: 64 for s in STOPS})
        self.assertEqual(lab["region_obstruction"]["removed"], {s: 0.25 for s in STOPS})
        self.assertEqual(lab["region_obstruction"]["occluded"], {s: 0.25 for s in STOPS})
        self.assertNotEqual(lab["region_obstruction"]["removed"]["P3"], 0.16)             # 16 / 100 of the reference mask
        self.assertEqual(lab["visible_fraction"]["occluded"]["P3"], 0.75)                 # 48 / 64 at the query pose
        self.assertEqual(lab["alone_px"], {"reference": {s: 100 for s in STOPS}, "query": {s: 64 for s in STOPS}})

    def test_cutoffs_are_the_preregistered_ones(self):
        self.assertEqual((B.PRESENT_DIAGNOSTIC_MIN, B.REMOVED_OBSTRUCTION_MAX), (0.90, 0.10))
        tgt, alone_ref, alone_q, occ = self._scene([10, 11, 9, 0, 0])
        d = B.compute_labels(STOPS, tgt, alone_ref, alone_q, occ)
        self.assertEqual(d["diagnostic"]["occluded"]["P1"], True)       # 0.90 is diagnostic
        self.assertEqual(d["diagnostic"]["occluded"]["P2"], False)      # 0.89
        self.assertEqual(d["diagnostic"]["removed"]["P1"], False)       # 0.10 is not < 0.10
        self.assertEqual(d["diagnostic"]["removed"]["P3"], True)        # 0.09


class TestAtomicRender(unittest.TestCase):
    def _complete(self, folder: Path, arr_bytes: bytes, **over):
        folder.mkdir(parents=True)
        (folder / "arrangement.json").write_bytes(arr_bytes)
        lab = {"schema": B.LABELS_SCHEMA, "hashes": {"arrangement_file_sha256": hashlib.sha256(arr_bytes).hexdigest()}}
        lab.update(over)
        (folder / "labels.json").write_text(json.dumps(lab))
        (folder / "render.json").write_text("{}")

    def test_needs_render(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "SD1"
            self.assertEqual(B.needs_render(f, b"v1"), (True, "no labels.json"))
            self._complete(f, b"v1")
            self.assertFalse(B.needs_render(f, b"v1")[0])
            redo, why = B.needs_render(f, b"v2")
            self.assertTrue(redo)
            self.assertIn("input is", why)
            (f / "render.json").unlink()
            self.assertEqual(B.needs_render(f, b"v1"), (True, "no render.json"))
            (f / "render.json").write_text("{}")
            (f / "arrangement.json").write_bytes(b"v3")
            self.assertEqual(B.needs_render(f, b"v1"), (True, "arrangement.json differs from the input"))
            (f / "labels.json").write_text("{not json")
            self.assertEqual(B.needs_render(f, b"v1"), (True, "labels.json unreadable"))
            (f / "labels.json").write_text(json.dumps({"schema": "cmpe249.sim-view-labels.v0"}))
            self.assertTrue(B.needs_render(f, b"v1")[0])

    def test_install_replaces_atomically_and_leaves_no_mix(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            final, tmp = out / "SD1", B.partial_dir(out, "SD1")
            self.assertEqual(tmp.name, "SD1.partial")
            self._complete(final, b"old")
            (final / "rgb_old.png").write_text("old")
            self._complete(tmp, b"new")
            (tmp / "rgb_new.png").write_text("new")
            B.install_rendered(tmp, final)
            self.assertFalse(tmp.exists())
            self.assertEqual(sorted(p.name for p in final.iterdir()), ["arrangement.json", "labels.json", "render.json", "rgb_new.png"])
            self.assertEqual((final / "arrangement.json").read_bytes(), b"new")
            self.assertEqual([p.name for p in out.iterdir()], ["SD1"])                   # no .old leftovers


class FakeIsaacCase(unittest.TestCase):
    """Drive the real render_arrangement and main with the stub Isaac."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "assets").mkdir()
        (self.root / "arr").mkdir()
        self.stack = contextlib.ExitStack()
        for p in FI.patches(B):
            self.stack.enter_context(p)
        self.addCleanup(self.stack.close)

    def write(self, arr: dict) -> Path:
        p = self.root / "arr" / f"{arr['arrangement_id']}.json"
        p.write_text(json.dumps(arr, indent=1))
        return p

    def run_main(self, files, out: Path, run: int = 1, crash_at=None, skip=False, extra=()) -> str:
        FI.reset(run, crash_at)
        argv = ["--arrangements", *map(str, files), "--assets-dir", str(self.root / "assets"), "--tags-dir", "tags", "--out", str(out)]
        argv += ["--skip-existing"] if skip else []
        argv += list(extra)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                B.main(argv)
            return "ok:" + buf.getvalue()
        except RuntimeError as e:
            return f"crashed: {e}"

    @staticmethod
    def px(p: Path) -> int:
        import cv2

        return int(cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[0, 0, 0])


class TestRenderFlow(FakeIsaacCase):
    def test_one_complete_render(self):
        out = self.root / "out"
        f = self.write(FI.arrangement())
        self.assertTrue(self.run_main([f], out).startswith("ok:"))
        d = out / "SD1003"
        self.assertEqual(sorted(p.name for p in out.iterdir() if not p.name.startswith("render_report")), ["SD1003"])
        self.assertTrue((d / "labels.json").is_file() and (d / "render.json").is_file())
        # fresh render products for every capture: no render product or annotator is read in two orchestrator steps
        steps_of_rp, steps_of_ann = {}, {}
        for c in FI.STATE["captures"]:
            steps_of_rp.setdefault(c["rp"], set()).add(c["step"])
            steps_of_ann.setdefault(c["ann"], set()).add(c["step"])
        self.assertTrue(all(len(v) == 1 for v in steps_of_rp.values()))
        self.assertTrue(all(len(v) == 1 for v in steps_of_ann.values()))
        self.assertEqual(len({c["step"] for c in FI.STATE["captures"]}), 4 * 2 + 2)            # 8 captures + nominal / revisit alone
        # reference captures use the nominal cameras, queries the revisit cameras, with the lighting that was active
        for ph in B.PHASE_ORDER:
            for s in STOPS:
                for L in ("L1", "L2"):
                    want = (100 if L == "L1" else 200) + 1 + (0 if ph == "reference" else 10)
                    self.assertEqual(self.px(d / "rgb" / f"{ph}_{s}_{L}.png"), want, (ph, s, L))
                    meta = json.loads(FI.PNG_TEXT[str(B.partial_dir(out, "SD1003") / "rgb" / f"{ph}_{s}_{L}.png")]["cmpe249"])   # written into .partial
                    self.assertEqual((meta["camera"], meta["lighting"], meta["phase"], meta["simulation"]),
                                     ("nominal" if ph == "reference" else "revisit", L, ph, True))
                    if ph == "reference":
                        self.assertIsNone(meta["revisit_delta"])
                    else:
                        self.assertEqual(meta["revisit_delta"], FI.arrangement()["revisit"]["stops"][s]["delta"])
        # labels from the renders at the query poses: the fake target is 4 x 4 at cols 4-7 (nominal) / 5-8 (revisit); the
        # occluder covers cols 4-5 when placed
        lab = json.loads((d / "labels.json").read_text())
        self.assertEqual(lab["schema"], B.LABELS_SCHEMA)
        self.assertEqual(lab["visible_fraction"]["reference"]["P3"], 1.0)
        self.assertEqual(lab["visible_fraction"]["visible"]["P3"], 1.0)
        self.assertEqual(lab["visible_fraction"]["occluded"]["P3"], 0.75)                        # 12 / 16 at the revisit pose, not 8 / 16
        self.assertEqual(lab["region_obstruction"]["occluded"]["P3"], 0.25)                      # 4 of the 16 visible-phase pixels
        self.assertEqual(lab["region_obstruction"]["removed"]["P3"], 0.25)                       # not 8 / 16 of the reference mask
        self.assertEqual(lab["patterns"], {"reference": "CCCCC", "visible": "CCCCC", "occluded": "ppppp"})
        self.assertEqual(lab["diagnostic"]["visible"]["P3"], True)
        self.assertEqual((lab["diagnostic"]["occluded"]["P3"], lab["diagnostic"]["removed"]["P3"]), (False, False))
        self.assertEqual(lab["reference_box"]["L1"]["P3"], [4, 4, 8, 8])
        self.assertEqual(lab["query_box_visible"]["L1"]["P3"], [5, 4, 9, 8])
        self.assertEqual(lab["removed_target_px"]["P3"], 0)
        self.assertEqual(lab["alone_px"], {"reference": {s: 16 for s in STOPS}, "query": {s: 16 for s in STOPS}})
        self.assertEqual(lab["hashes"]["arrangement_file_sha256"], hashlib.sha256(f.read_bytes()).hexdigest())
        rec = json.loads((d / "render.json").read_text())
        self.assertEqual(lab["hashes"]["render_settings_sha256"], B.sha256_json(rec["settings"]))
        self.assertEqual(rec["settings"]["lightings"], ["L1", "L2"])
        self.assertTrue(lab["label_sanity"]["ok"])
        self.assertEqual(lab["label_sanity"]["ratio"], {s: 1.0 for s in STOPS})
        self.assertEqual((lab["occluder_family"], lab["m16_side"], lab["design_case"]), ("white_box", None, False))
        self.assertEqual(lab["revisit"]["stops"]["P2"]["delta"], FI.arrangement()["revisit"]["stops"]["P2"]["delta"])
        for s in STOPS:
            for name in (f"target_alone_{s}.png", f"target_alone_revisit_{s}.png", f"target_reference_{s}.png",
                         f"target_visible_{s}.png", f"classes_removed_{s}.png"):
                self.assertTrue((d / "masks" / name).is_file(), name)
        # the evidence annotation's box stays the reference's
        doc = E.build(out, positions=None)
        ann = json.loads((d / "annotations" / "ann_P3_L1.json").read_text())
        self.assertEqual(ann["targets"]["tomato_soup_can"]["box"], [4, 4, 8, 8])
        self.assertEqual(doc["n_arrangements"], 1)

    def test_interrupted_rerender_never_leaves_a_mixed_folder(self):
        out = self.root / "out"
        v1 = self.write(FI.arrangement(target="tomato_soup_can"))
        self.assertTrue(self.run_main([v1], out, run=1).startswith("ok:"))
        d = out / "SD1003"
        v1_bytes = (d / "arrangement.json").read_bytes()
        v2 = self.write(FI.arrangement(target="mug", parked_x=0.25))                           # regenerated, same id
        self.assertIn("crashed", self.run_main([v2], out, run=2, crash_at=4))
        # the crash left the old complete render untouched and the new one in .partial
        self.assertEqual((d / "arrangement.json").read_bytes(), v1_bytes)
        lab = json.loads((d / "labels.json").read_text())
        self.assertEqual(lab["hashes"]["arrangement_file_sha256"], hashlib.sha256(v1_bytes).hexdigest())
        self.assertEqual({self.px(p) % 100 for p in (d / "rgb").glob("*.png")} , {1, 11})
        self.assertTrue(B.partial_dir(out, "SD1003").is_dir())
        # the set cannot be built while the partial is there
        with self.assertRaises(E.IncompleteRender):
            E.build(out, positions=None)
        # resume with --skip-existing: the labels' arrangement hash differs from v2, so it re-renders
        res = self.run_main([v2], out, run=3, skip=True)
        self.assertIn("re-render SD1003", res)
        self.assertFalse(B.partial_dir(out, "SD1003").exists())
        lab = json.loads((d / "labels.json").read_text())
        self.assertEqual(lab["target_id"], "mug")
        self.assertEqual(lab["hashes"]["arrangement_file_sha256"], hashlib.sha256((d / "arrangement.json").read_bytes()).hexdigest())
        self.assertEqual({self.px(p) % 100 for p in (d / "rgb").glob("*.png")}, {3, 13})      # every PNG is from run 3
        self.assertEqual(E.build(out, positions=None)["n_arrangements"], 1)
        # and a second resume skips it: same arrangement file
        res = self.run_main([v2], out, run=4, skip=True)
        self.assertIn("skip SD1003 (complete render of the same arrangement file)", res)
        self.assertEqual(FI.STATE["captures"], [])
        self.assertEqual({self.px(p) % 100 for p in (d / "rgb").glob("*.png")}, {3, 13})

    def test_rerender_without_skip_replaces_the_folder_whole(self):
        out = self.root / "out"
        f = self.write(FI.arrangement())
        self.run_main([f], out, run=1)
        self.run_main([f], out, run=2)
        d = out / "SD1003"
        self.assertEqual({self.px(p) % 100 for p in (d / "rgb").glob("*.png")}, {2, 12})
        self.assertEqual(sorted(p.name for p in out.iterdir() if p.is_dir()), ["SD1003"])

    def test_a_crashed_first_render_is_not_skipped_by_episodes(self):
        out = self.root / "out"
        fa = self.write(FI.arrangement())
        fb = self.write(FI.arrangement("SD1004", 1004))
        self.assertTrue(self.run_main([fa], out, run=1).startswith("ok:"))
        self.assertIn("crashed", self.run_main([fb], out, run=1, crash_at=3))
        self.assertEqual(sorted(p.name for p in out.iterdir() if p.is_dir()), ["SD1003", "SD1004.partial"])
        with self.assertRaisesRegex(E.IncompleteRender, "SD1004"):
            E.build(out, positions=None)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(E.main(["--root", str(out), "--positions", str(POS)]), 2)

    def test_non_arrangement_files_in_the_glob_are_skipped(self):
        out = self.root / "out"
        f = self.write(FI.arrangement())
        idx = self.root / "arr" / "index.json"
        idx.write_text(json.dumps({"schema": "cmpe249.sim-arrangement-index.v0", "arrangements": []}))
        res = self.run_main([f, idx], out)
        self.assertTrue(res.startswith("ok:"))
        self.assertIn("not an arrangement", res)
        self.assertTrue((out / "SD1003" / "labels.json").is_file())

    def test_test_seed_is_refused_before_any_render(self):
        out = self.root / "out"
        f = self.write(FI.arrangement("SD100000", 100000))
        with self.assertRaises(B.RefusedArrangement):
            self.run_main([f], out)
        self.assertFalse(out.exists())


class TestEpisodes(unittest.TestCase):
    def test_build_writes_the_formats_the_next_stage_reads(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False)])
            doc = E.build(root, positions=POS)
            self.assertEqual(doc["jobs"], {"primary": "jobs_refL1.json", "secondary": "jobs_same.json"})
            prim = json.loads((root / "jobs_refL1.json").read_text())
            sec = json.loads((root / "jobs_same.json").read_text())
            self.assertEqual((prim["role"], prim["ref_lighting"], sec["role"], sec["ref_lighting"]),
                             ("primary", "L1", "secondary", "same"))
            for jd in (prim, sec):
                self.assertEqual(jd["schema"], "cmpe249.evidence-jobs.v0")
                self.assertEqual(len(jd["jobs"]), 3 * 2 * 5)
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
            self.assertEqual(len(list(root.glob("*/annotations/ann_*_L*.json"))), 5 * 2)
            specs, truth = load_manifest(root / "manifest.json")
            self.assertEqual(len(specs), 6)
            self.assertEqual({s.split.value for s in specs}, {"dev"})
            tr = json.loads((root / "truth.json").read_text())["states"]
            self.assertEqual(tr["SD1002-occluded-L2"], "present")
            self.assertEqual(tr["SD1002-removed-L1"], "removed")
            vl = json.loads((root / "view_labels.json").read_text())["episodes"]
            self.assertEqual([s["diagnostic"] for s in vl["SD1002-occluded-L1"]["stops"]], [True, True, False, False, True])
            self.assertEqual(json.loads((root / "positions_shelf_A.json").read_text()), json.loads(POS.read_text()))
            self.assertEqual(doc["provenance"]["positions_sha256"], A.sha256_file(POS))
            self.assertEqual(doc["provenance"]["arrangements"]["SD1002"]["revisit"]["P3"]["delta"]["dx_m"], 0.001)

    def test_manifest_has_the_simulation_flag_and_the_analysis_keys(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False, {"family": "S8"}),
                              ("SD1000", 1000, "random", "foam_brick", False, {"family": "M16", "side": "L"}),
                              ("SD1007", 1007, "random", "tuna_fish_can", False, {"family": "M16", "side": "R"}),
                              ("SD1001", 1001, "random", "mug", True, {"family": "IND28"}),
                              ("DES0", "design-0", "design", "study_mug", False, {"family": "S8", "design": True}),
                              ("DES1", "design-1", "design", "study_mug", False, {"family": "M16", "side": "R", "design": True}),
                              ("DES3", "design-3", "design", "study_mug", True, {"family": "IND28", "design": True})])
            doc = E.build(root, positions=POS)
            man = json.loads((root / "manifest.json").read_text())
            self.assertIs(man["simulation"], True)
            self.assertEqual(man["schema"], "cmpe249.arrangements.v1")
            rows = {r["arrangement_id"]: r for r in man["arrangements"]}
            self.assertEqual(sorted(rows), ["SD1000", "SD1001", "SD1002", "SD1007"])         # the design cases are not in it
            self.assertEqual({k: (r["occluder_family"], r["m16_side"], r["design_case"]) for k, r in rows.items()},
                             {"SD1000": ("M16", "L", False), "SD1001": ("IND28", None, False), "SD1002": ("S8", None, False),
                              "SD1007": ("M16", "R", False)})
            # verify's own fields are still there and its loader still reads the file (it ignores unknown keys)
            self.assertEqual(rows["SD1001"]["indistinguishable"], True)
            self.assertEqual(rows["SD1002"]["split"], "dev")
            specs, truth = load_manifest(root / "manifest.json")
            self.assertEqual(len(specs), 4 * 6)
            # the design cases are a separate manifest, jobs, truth and view labels
            dman = json.loads((root / "manifest_design.json").read_text())
            self.assertIs(dman["simulation"], True)
            self.assertEqual({r["arrangement_id"]: (r["occluder_family"], r["m16_side"], r["design_case"]) for r in dman["arrangements"]},
                             {"DES0": ("S8", None, True), "DES1": ("M16", "R", True), "DES3": ("IND28", None, True)})
            self.assertEqual(len(load_manifest(root / "manifest_design.json")[0]), 3 * 6)
            for name in ("jobs_design_refL1.json", "jobs_design_same.json", "truth_design.json", "view_labels_design.json", "summary_design.md"):
                self.assertTrue((root / name).is_file(), name)
            ids = {j["episode_id"].split("-")[0] for j in json.loads((root / "jobs_refL1.json").read_text())["jobs"]}
            dids = {j["episode_id"].split("-")[0] for j in json.loads((root / "jobs_design_refL1.json").read_text())["jobs"]}
            self.assertEqual((ids, dids), ({"SD1000", "SD1001", "SD1002", "SD1007"}, {"DES0", "DES1", "DES3"}))
            self.assertEqual(sorted(json.loads((root / "truth.json").read_text())["states"]) [0].split("-")[0], "SD1000")
            self.assertEqual((doc["n_arrangements"], doc["groups"]["design"]["n_arrangements"]), (4, 3))
            self.assertIn("DES1", (root / "README.md").read_text())

    def test_set_with_only_design_cases(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("DES0", "design-0", "design", "study_mug", False, {"family": "S8", "design": True})])
            doc = E.build(root, positions=None)
            self.assertEqual((doc["n_arrangements"], list(doc["groups"])), (0, ["design"]))
            self.assertFalse((root / "manifest.json").exists())
            self.assertTrue((root / "manifest_design.json").is_file())

    def test_annotations_load_with_the_frozen_evidence_reader(self):
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("cv2 not installed (perception/evidence.py imports it)")
        from perception.evidence import load_annotation

        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1004", 1004, "random", "tuna_fish_can", False)])
            E.build(root, positions=None)
            for p in root.glob("SD1004/annotations/*.json"):
                self.assertIn("tuna_fish_can", load_annotation(p)["targets"])

    def test_non_dev_arrangements_are_refused(self):
        for seed, kind in ((100000, "random"), (12345, "random"), ("design-0", "random")):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as t:
                root = Path(t) / "set"
                _write_set(root, [("SD0009", seed, kind, "mug", False)])
                with self.assertRaises(A.SeedError):
                    E.build(root, positions=None)

    def test_incomplete_folders_are_never_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False)])
            (root / "SD1003").mkdir()
            (root / "SD1003" / "arrangement.json").write_text("{}")                          # crashed: no labels.json
            (root / "SD1004.partial").mkdir()                                                # interrupted render
            (root / "SD1005").mkdir()
            (root / "SD1005" / "labels.json").write_text("{}")                               # labels without an arrangement
            (root / "notes").mkdir()                                                         # not a render folder: ignored
            with self.assertRaises(E.IncompleteRender) as cm:
                E.load_rendered(root)
            msg = str(cm.exception)
            self.assertIn("SD1003", msg)
            self.assertIn("SD1004.partial", msg)
            self.assertIn("SD1005", msg)
            self.assertNotIn("SD1002", msg)
            self.assertNotIn("notes", msg)
            with self.assertRaises(E.IncompleteRender):
                E.build(root, positions=None)
            self.assertFalse((root / "manifest.json").exists())                               # nothing was built
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(E.main(["--root", str(root)]), 2)

    def test_mixed_folder_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False)])
            self.assertEqual(len(E.load_rendered(root)), 1)
            arr = json.loads((root / "SD1002" / "arrangement.json").read_text())
            arr["target_id"] = "tuna_fish_can"                                               # a regenerated arrangement over old labels
            (root / "SD1002" / "arrangement.json").write_text(json.dumps(arr))
            with self.assertRaisesRegex(ValueError, "mixed folder"):
                E.load_rendered(root)
            with self.assertRaises(ValueError):
                E.build(root, positions=None)

    def test_old_label_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False)])
            lab = json.loads((root / "SD1002" / "labels.json").read_text())
            lab["schema"] = "cmpe249.sim-view-labels.v0"
            (root / "SD1002" / "labels.json").write_text(json.dumps(lab))
            with self.assertRaisesRegex(ValueError, "schema"):
                E.load_rendered(root)

    def test_positions_file_must_be_the_one_the_arrangements_recorded(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False)])
            other = Path(t) / "positions_shelf_A.json"
            pos = json.loads(POS.read_text())
            for st, az in zip(pos["stops"], (-60, -30, 0, 30, 60)):
                st["azimuth_deg"] = az
            other.write_text(json.dumps(pos))
            with self.assertRaisesRegex(ValueError, "differs from the positions file recorded"):
                E.build(root, positions=other)
            self.assertFalse((root / "positions_shelf_A.json").exists())
            self.assertFalse((root / "manifest.json").exists())                               # refused before anything is written
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(E.main(["--root", str(root), "--positions", str(other)]), 2)
            E.build(root, positions=POS)                                                      # the recorded one is accepted
            # one arrangement with another recorded hash is enough to refuse
            _write_set(root, [("SD1010", 1010, "random", "mug", False)], pos_sha="f" * 64)
            with self.assertRaisesRegex(ValueError, "SD1010"):
                E.build(root, positions=POS)

    def test_stop_names_must_match(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False)])
            pos = json.loads(POS.read_text())
            pos["stops"][0]["name"] = "Q1"
            other = Path(t) / "p.json"
            other.write_text(json.dumps(pos))
            # the recorded hash is of the real file, so build the set against the renamed file's hash to isolate the name check
            _write_set(root.parent / "set2", [("SD1002", 1002, "random", "mug", False)], pos_sha=A.sha256_file(other))
            with self.assertRaisesRegex(ValueError, "stop names"):
                E.build(root.parent / "set2", positions=other)

    def test_generation_log_is_copied_into_the_set(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False)])
            log = Path(t) / "log.json"
            log.write_text(json.dumps({"schema": A.GENERATION_LOG_SCHEMA}))
            doc = E.build(root, positions=POS, generation_log=log)
            self.assertEqual(json.loads((root / "generation_log.json").read_text())["schema"], A.GENERATION_LOG_SCHEMA)
            self.assertIn("generation_log.json", doc["files"])


def _expect(**over) -> dict:
    e = {"isaac_batch_sha256": "b" * 64, "isaac_rig_sha256": "r" * 64, "lightings": ["L1", "L2"], "render_settings_sha256": "s" * 64}
    e.update(over)
    return e


class TestSkipExistingChecksMoreThanTheArrangement(unittest.TestCase):
    """--skip-existing may skip only a folder that a fresh render with today's code, lightings and settings would reproduce."""

    def _folder(self, root: Path, arr_bytes: bytes = b"v1", lightings=("L1", "L2"), **hashes) -> Path:
        f = root / "SD1"
        f.mkdir(parents=True)
        (f / "arrangement.json").write_bytes(arr_bytes)
        h = {"arrangement_file_sha256": hashlib.sha256(arr_bytes).hexdigest(), "isaac_batch_sha256": "b" * 64,
             "isaac_rig_sha256": "r" * 64, "render_settings_sha256": "s" * 64}
        h.update(hashes)
        (f / "labels.json").write_text(json.dumps({"schema": B.LABELS_SCHEMA, "stops": STOPS, "lightings": list(lightings), "hashes": h}))
        (f / "render.json").write_text("{}")
        (f / "rgb").mkdir()
        for ph in B.PHASE_ORDER:
            for st in STOPS:
                for L in lightings:
                    (f / "rgb" / f"{ph}_{st}_{L}.png").write_bytes(b"x")
        return f

    def test_a_matching_folder_is_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            f = self._folder(Path(t))
            self.assertEqual(B.needs_render(f, b"v1", _expect()), (False, "complete render of the same arrangement file"))

    def test_other_code_lightings_settings_or_files_re_render(self):
        for name, expect, mutate, word in (
                ("label code", _expect(isaac_batch_sha256="c" * 64), None, "isaac_batch"),
                ("rig code", _expect(isaac_rig_sha256="c" * 64), None, "isaac_rig"),
                ("settings", _expect(render_settings_sha256="c" * 64), None, "render settings"),
                ("L1 folder, L1+L2 wanted", _expect(), "l1_only", "lightings"),
                ("L1+L2 folder, L2 wanted", _expect(lightings=["L2"]), None, "lightings"),
                ("an image is missing", _expect(), "drop_png", "missing")):
            with self.subTest(name), tempfile.TemporaryDirectory() as t:
                f = self._folder(Path(t), lightings=("L1",) if mutate == "l1_only" else ("L1", "L2"))
                if mutate == "drop_png":
                    (f / "rgb" / "removed_P3_L2.png").unlink()
                redo, why = B.needs_render(f, b"v1", expect)
                self.assertTrue(redo, name)
                self.assertIn(word, why)

    def test_folder_without_the_new_hashes_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            f = self._folder(Path(t))
            lab = json.loads((f / "labels.json").read_text())
            del lab["hashes"]["render_settings_sha256"]                         # a folder rendered before this check existed
            (f / "labels.json").write_text(json.dumps(lab))
            self.assertTrue(B.needs_render(f, b"v1", _expect())[0])

    def test_render_settings_cover_resolution_renderer_updates_and_lighting(self):
        arr = FI.arrangement()
        kw = dict(renderer="RaytracedLighting", rt_subframes=16, settle=200, capture_updates=100, lightings=["L1", "L2"],
                  assets_dir="/nonexistent", tags_dir="/nonexistent", isaacsim="6.2.0")
        base = B.sha256_json(B.render_settings(arr, **kw))
        self.assertEqual(base, B.sha256_json(B.render_settings(arr, **kw)))
        for over in ({"renderer": "PathTracing"}, {"rt_subframes": 8}, {"settle": 100}, {"capture_updates": 50},
                     {"lightings": ["L1"]}, {"isaacsim": "6.3.0"}):
            with self.subTest(**over):
                self.assertNotEqual(base, B.sha256_json(B.render_settings(arr, **{**kw, **over})))
        bigger = json.loads(json.dumps(arr))
        for c in bigger["scene"]["cameras"]:
            c["width"] *= 2
        self.assertNotEqual(base, B.sha256_json(B.render_settings(bigger, **kw)))


class TestSkipExistingFlow(FakeIsaacCase):
    def _images(self, d: Path) -> set:
        return {p.name for p in (d / "rgb").glob("*.png")}

    def test_an_l1_only_folder_resumed_with_l1_and_l2_is_re_rendered(self):
        out = self.root / "out"
        f = self.write(FI.arrangement())
        self.assertTrue(self.run_main([f], out, run=1, extra=["--lighting", "L1"]).startswith("ok:"))
        d = out / "SD1003"
        self.assertEqual(len(self._images(d)), 4 * 5)
        res = self.run_main([f], out, run=2, skip=True)                                     # default lightings: L1 L2
        self.assertIn("re-render SD1003", res)
        self.assertIn("lightings", res)
        self.assertEqual(len(self._images(d)), 4 * 5 * 2)
        self.assertEqual(json.loads((d / "labels.json").read_text())["lightings"], ["L1", "L2"])
        self.assertEqual(E.build(out, positions=None)["n_episodes"], 3 * 2)
        res = self.run_main([f], out, run=3, skip=True)                                     # now complete: skipped
        self.assertIn("skip SD1003", res)
        self.assertEqual(FI.STATE["captures"], [])

    def test_a_label_code_change_re_renders(self):
        out = self.root / "out"
        f = self.write(FI.arrangement())
        self.run_main([f], out, run=1)
        d = out / "SD1003"
        other = self.root / "isaac_batch_changed.py"
        other.write_text(B.THIS_FILE.read_text() + "\n# a compute_labels fix\n")
        with mock.patch.object(B, "THIS_FILE", other):
            res = self.run_main([f], out, run=2, skip=True)
        self.assertIn("re-render SD1003", res)
        self.assertIn("isaac_batch", res)
        self.assertEqual({self.px(p) % 100 for p in (d / "rgb").glob("*.png")}, {2, 12})
        self.assertEqual(json.loads((d / "labels.json").read_text())["hashes"]["isaac_batch_sha256"], B.sha256_file(other))

    def test_a_rig_code_change_and_a_settings_change_re_render(self):
        out = self.root / "out"
        f = self.write(FI.arrangement())
        self.run_main([f], out, run=1)
        self.assertIn("skip SD1003", self.run_main([f], out, run=2, skip=True))
        res = self.run_main([f], out, run=3, skip=True, extra=["--rt-subframes", "8"])
        self.assertIn("re-render SD1003", res)
        self.assertIn("render settings", res)
        res = self.run_main([f], out, run=4, skip=True, extra=["--rt-subframes", "8", "--settle", "50"])
        self.assertIn("re-render SD1003", res)
        other = self.root / "isaac_rig_changed.py"
        other.write_text(Path(B.ir.__file__).read_text() + "\n# changed\n")
        with mock.patch.object(B.ir, "__file__", str(other)):
            res = self.run_main([f], out, run=5, skip=True, extra=["--rt-subframes", "8", "--settle", "50"])
        self.assertIn("re-render SD1003", res)
        self.assertIn("isaac_rig", res)


class TestLabelSanity(unittest.TestCase):
    def test_visible_phase_must_show_the_target_whole(self):
        ok = B.label_sanity(STOPS, dict(zip(STOPS, [100, 95, 90, 100, 100])), {s: 100 for s in STOPS})
        self.assertTrue(ok["ok"])                                                      # 0.90 is not below 0.90
        self.assertEqual(ok["flagged_stops"], [])
        bad = B.label_sanity(STOPS, dict(zip(STOPS, [100, 95, 89, 100, 40])), {s: 100 for s in STOPS})
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["flagged_stops"], ["P3", "P5"])
        self.assertEqual(bad["ratio"]["P3"], 0.89)
        none = B.label_sanity(STOPS, {s: 0 for s in STOPS}, {**{s: 100 for s in STOPS}, "P2": 0})
        self.assertEqual(none["flagged_stops"], ["P1", "P2", "P3", "P4", "P5"])           # P2 has nothing to compare with

    def test_compute_labels_records_it(self):
        sq = TestMasksAndLabels._square
        alone = {s: sq(5, 5) for s in STOPS}
        tgt = {("reference", s): alone[s].copy() for s in STOPS}
        tgt.update({("visible", s): alone[s].copy() for s in STOPS})
        tgt.update({("occluded", s): alone[s].copy() for s in STOPS})
        occ = {(ph, s): np.zeros_like(alone[s]) for ph in ("occluded", "removed") for s in STOPS}
        self.assertTrue(B.compute_labels(STOPS, tgt, alone, alone, occ)["label_sanity"]["ok"])
        part = alone["P3"].copy()
        part[5:15, 5:10] = False                                                          # the shelf hides half the target
        tgt[("visible", "P3")] = part
        lab = B.compute_labels(STOPS, tgt, alone, alone, occ)
        self.assertFalse(lab["label_sanity"]["ok"])
        self.assertEqual(lab["label_sanity"]["flagged_stops"], ["P3"])
        self.assertEqual(lab["label_sanity"]["ratio"]["P3"], 0.5)


class TestEpisodeSetIntegrity(unittest.TestCase):
    @staticmethod
    def _retag(root: Path, arr_id: str, generator=None, config=None) -> None:
        d = root / arr_id
        arr = json.loads((d / "arrangement.json").read_text())
        if generator:
            arr["provenance"]["generator"]["sha256"] = generator
        if config:
            arr["provenance"]["config"]["sha256"] = config
        b = json.dumps(arr).encode()
        (d / "arrangement.json").write_bytes(b)
        lab = json.loads((d / "labels.json").read_text())
        lab["hashes"]["arrangement_file_sha256"] = hashlib.sha256(b).hexdigest()
        (d / "labels.json").write_text(json.dumps(lab))

    def test_a_set_from_two_generator_or_config_versions_is_refused(self):
        for what, kw in (("generator", {"generator": "h" * 64}), ("config", {"config": "d" * 64})):
            with self.subTest(what), tempfile.TemporaryDirectory() as t:
                root = Path(t) / "set"
                _write_set(root, [("SD1002", 1002, "random", "mug", False), ("SD1003", 1003, "random", "mug", False)])
                self.assertEqual(E.build(root, positions=None)["n_arrangements"], 2)          # one version: fine
                self._retag(root, "SD1003", **kw)
                for f in root.glob("*.json"):
                    f.unlink()
                with self.assertRaisesRegex(ValueError, f"2 {what} versions"):
                    E.build(root, positions=None)
                self.assertFalse((root / "manifest.json").exists())                            # refused before anything is written
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(E.main(["--root", str(root), "--positions", str(POS)]), 2)

    def test_a_design_case_from_another_version_is_refused_too(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False),
                              ("DES0", "design-0", "design", "study_mug", False, {"design": True})])
            self._retag(root, "DES0", generator="h" * 64)
            with self.assertRaisesRegex(ValueError, "generator versions"):
                E.build(root, positions=None)

    def test_n_episodes_counts_the_rendered_lightings(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "set"
            _write_set(root, [("SD1002", 1002, "random", "mug", False), ("SD1003", 1003, "random", "mug", False)])
            self.assertEqual(E.build(root, positions=None)["n_episodes"], 2 * 3 * 2)
            for arr_id in ("SD1002", "SD1003"):
                lab = json.loads((root / arr_id / "labels.json").read_text())
                lab["lightings"] = ["L1"]
                (root / arr_id / "labels.json").write_text(json.dumps(lab))
            doc = E.build(root, positions=None)
            self.assertEqual((doc["n_episodes"], doc["groups"]["random"]["n_episodes"]), (2 * 3, 2 * 3))
            jobs = json.loads((root / "jobs_refL1.json").read_text())["jobs"]
            self.assertEqual(len(jobs), 2 * 3 * 5)                                            # what the jobs hold
            self.assertEqual(len(json.loads((root / "manifest.json").read_text())["arrangements"]), 2)
            self.assertEqual(len({j["episode_id"] for j in jobs}), 2 * 3)

    def test_contact_sheets_survive_missing_fractions(self):
        import cv2

        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "SD1002"
            (f / "rgb").mkdir(parents=True)
            (f / "masks").mkdir()
            blob = np.zeros((48, 64), np.uint8)
            blob[10:20, 10:20] = 255
            for ph in B.PHASE_ORDER:
                for st in STOPS:
                    cv2.imwrite(str(f / "rgb" / f"{ph}_{st}_L1.png"), np.full((48, 64, 3), 120, np.uint8))
                    cv2.imwrite(str(f / "masks" / f"classes_{ph}_{st}.png"), (blob > 0).astype(np.uint8))
                    cv2.imwrite(str(f / "masks" / f"target_alone_{st}.png"), blob)
                    cv2.imwrite(str(f / "masks" / f"target_alone_revisit_{st}.png"), blob)
            none = {s: None for s in STOPS}
            lab = _labels("SD1002", 1002, "mug", b"x")
            lab.update(lightings=["L1"], visible_fraction={"reference": dict(none), "visible": dict(none), "occluded": dict(none)},
                       region_obstruction={"occluded": dict(none), "removed": dict(none)},
                       diagnostic={"visible": {s: False for s in STOPS}, "occluded": {s: False for s in STOPS},
                                   "removed": {s: False for s in STOPS}},
                       reference_box={"L1": dict(none)}, query_box_visible={"L1": dict(none)})
            paths = E.contact_sheets(f, lab)
            self.assertEqual(sorted(p.name for p in paths), ["contact_L1.jpg", "crops_L1.jpg"])
            self.assertTrue(all(p.is_file() and p.stat().st_size > 0 for p in paths))

    def test_summary_survives_a_set_without_reference_fractions(self):
        arr = _arr_doc("SD1002", 1002, "random", "mug", False)
        arr["objects"]["occluder"].update(kind="box", label="S8")
        lab = _labels("SD1002", 1002, "mug", b"x")
        lab["visible_fraction"]["reference"] = {s: None for s in STOPS}
        lab["visible_fraction"]["occluded"]["P2"] = None
        lab["region_obstruction"]["removed"]["P1"] = None
        self.assertIn("| SD1002 |", E.summary_md([(arr, lab, Path("."))]))


class TestCheckBatchFolders(unittest.TestCase):
    def test_crash_leftovers_are_skipped_and_reported(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for name in ("SD1002", "SD1003", "SD1003.partial", "SD1002.old12345", "SD1002.old", "check"):
                (root / name).mkdir()
                if name != "check":
                    (root / name / "labels.json").write_text("{}")
            (root / "notes.txt").write_text("x")
            folders, leftovers = CB.list_folders(root)
            self.assertEqual([p.name for p in folders], ["SD1002", "SD1003"])
            self.assertEqual(leftovers, ["SD1002.old", "SD1002.old12345", "SD1003.partial"])

    def test_the_folder_name_must_be_the_arrangement_id(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "SD1004"
            f.mkdir()
            (f / "arrangement.json").write_text(json.dumps({"arrangement_id": "SD1002"}))
            (f / "labels.json").write_text(json.dumps({"arrangement_id": "SD1002", "stops": STOPS}))
            (f / "render.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "named after its arrangement"):
                CB.check_arrangement(f, None)

    def test_label_sanity_is_recomputed_from_the_pixel_counts(self):
        lab = {"target_px": {"visible": dict(zip(STOPS, [100, 100, 80, 100, 100]))}, "alone_px": {"query": {s: 100 for s in STOPS}}}
        res = B.label_sanity(STOPS, lab["target_px"]["visible"], lab["alone_px"]["query"])
        self.assertEqual(res["flagged_stops"], ["P3"])
        self.assertEqual(CB.B.LABEL_SANITY_MIN, 0.90)


class TestCheckBatch(unittest.TestCase):
    def test_border_margin(self):
        c = {0: np.array([[10, 20], [30, 20], [30, 40], [10, 40]], float), 1: np.array([[100, 50], [120, 50], [120, 70], [100, 70]], float)}
        self.assertEqual(CB.border_margin(c, 200, 100), 10.0)
        self.assertEqual(CB.border_margin(c, 125, 100), 5.0)
        self.assertIsNone(CB.border_margin({}, 10, 10))

    def test_side_of_pattern(self):
        self.assertEqual([CB.side_of_pattern(p) for p in ("CCHHC", "CHHCC", "CpHpC", "CCHCC", "CCCCC", "HHHHH")],
                         ["R", "L", "both", None, None, "both"])

    def test_warped_mask_agreement(self):
        ref = np.zeros((40, 60), bool)
        ref[10:20, 10:20] = True
        qry = np.zeros_like(ref)
        qry[10:20, 15:25] = True                                   # the query pose shifted the target 5 px right
        shift_back = np.array([[1, 0, -5], [0, 1, 0], [0, 0, 1]], float)       # query -> reference pixels
        reg = CB.warped_mask_agreement(ref, qry, shift_back)
        self.assertEqual((reg["iou_after_registration"], reg["centroid_offset_px"]), (1.0, 0.0))
        self.assertAlmostEqual(reg["iou_without_registration"], 50 / 150)
        off = CB.warped_mask_agreement(ref, qry, np.eye(3))
        self.assertAlmostEqual(off["centroid_offset_px"], 5.0)

    def test_summarize(self):
        self.assertEqual(CB.summarize([1.0, None, 3.0, 2.0]), {"n": 3, "min": 1.0, "median": 2.0, "max": 3.0, "mean": 2.0})
        self.assertIsNone(CB.summarize([None]))


if __name__ == "__main__":
    unittest.main()
