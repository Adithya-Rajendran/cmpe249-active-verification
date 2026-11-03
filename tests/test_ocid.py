"""public_data/ocid.py, run_evidence.py and score_views.py on tiny synthetic OCID-like sequences (no real data,
no GPU: a mean-colour mock encoder and a mock detector stand in for DINOv2 and YOLO-World).

The scene (200 x 160, label 1 = surface everywhere objects are not), one object placed per frame and
painted in placement order, so later objects occlude earlier ones exactly as in OCID:
  f0 obj01 (id 2)  B: rect (120,20)-(150,50)
  f1 obj02 (id 3)  A: rect (20,20)-(60,60)
  f2 obj03 (id 4)  C: ring (18,18)-(62,62) with a 20 x 20 hole -> A 75 % hidden, same footprint
  f3 obj04 (id 5)  D: rect (110,35)-(170,80) -> covers the lower half of B (v = 0.5), not same footprint
  f4 obj05 (id 6)  E: rect (70,100)-(100,130)
  f5 obj06 (id 7)  F: rect (70,100)-(82,130) -> E keeps 60 % (ambiguous partial); F itself is too small
  f6 obj07 (id 8)  G: rect (130,100)-(160,130)
  f7 obj08 (id 9)  H: rect (0,140)-(30,160) touches the border; G is nudged 12 px right
  f8 obj09 (id 10) I: 10 x 10 (too small to be a reference); 20 rows of E become label 0 (no depth)
Real OCID surfaces are not always label 1: table sequences start with an empty-table frame in which
the table is id 2 (objects then 3, 4, ...), and one floor sequence labels the floor 256.
table_scene() and floor256_scene() build those variants of the same scene.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from perception import evidence as ev
from public_data import ocid
from public_data import run_evidence
from public_data import score_views
from verify.schema import EvidenceStore

W, H, N = 200, 160, 9


def scene_labels(id_offset: int = 0) -> np.ndarray:
    def rect(lab, x0, y0, x1, y1, v):
        lab[y0:y1, x0:x1] = v

    frames = []
    for f in range(N):
        lab = np.ones((H, W), np.int32)
        o = id_offset
        rect(lab, 120, 20, 150, 50, 2 + o)
        if f >= 1:
            rect(lab, 20, 20, 60, 60, 3 + o)
        if f >= 2:
            ring = np.zeros((H, W), bool)
            ring[18:62, 18:62] = True
            ring[30:50, 30:50] = False
            lab[ring] = 4 + o
        if f >= 3:
            rect(lab, 110, 35, 170, 80, 5 + o)
        if f >= 4:
            rect(lab, 70, 100, 100, 130, 6 + o)
        if f >= 5:
            rect(lab, 70, 100, 82, 130, 7 + o)
        if f >= 6:
            gx = 142 if f >= 7 else 130
            rect(lab, gx, 100, gx + 30, 130, 8 + o)
        if f >= 7:
            rect(lab, 0, 140, 30, 160, 9 + o)
        if f >= 8:
            rect(lab, 180, 5, 190, 15, 10 + o)
            rect(lab, 82, 100, 100, 120, ocid.LABEL_NO_DEPTH)
        frames.append(lab)
    return np.stack(frames)


TABLE = (0, 10, W, 150)        # x0, y0, x1, y1 of the synthetic table: 87.5 % of the image


def table_scene() -> np.ndarray:
    """OCID table layout: frame 0 = empty table (background 1, table id 2), objects ids 3.. from frame 1."""
    base = scene_labels(id_offset=1)
    tab = np.ones((H, W), np.int32)
    x0, y0, x1, y1 = TABLE
    tab[y0:y1, x0:x1] = 2
    out = [tab]
    for lab in base:
        f = tab.copy()
        f[lab != 1] = lab[lab != 1]
        out.append(f)
    return np.stack(out)


def floor256_scene() -> np.ndarray:
    """ARID10/floor/top/curved/seq05 layout: no label 1, floor = 256, objects 257.. (first in frame 0)."""
    lab = scene_labels(id_offset=255)
    lab[lab == 1] = 256
    return lab


def frame_names(n: int = N, start_s: int = 0) -> tuple[str, ...]:
    return tuple(f"result_2020-01-01-00-{(start_s + 20 * i) // 60:02d}-{(start_s + 20 * i) % 60:02d}.png"
                 for i in range(n))


def fake_seq(rel: str = "SYN10/table/top/seq01", root: Path = Path("/nonexistent"), n: int = N) -> ocid.OcidSequence:
    return ocid.OcidSequence(root, rel, frame_names(n))


def write_sequence(root: Path, rel: str, labels: np.ndarray, start_s: int = 0) -> None:
    rng = np.random.default_rng(0)
    palette = rng.integers(30, 225, size=(int(labels.max()) + 1, 3)).astype(np.uint8)
    for name, lab in zip(frame_names(len(labels), start_s), labels):
        for sub in ("label", "rgb", "depth"):
            (root / rel / sub).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / rel / "label" / name), lab.astype(np.uint16))
        rgb = palette[lab] + rng.integers(0, 8, size=lab.shape + (3,), dtype=np.uint8)
        cv2.imwrite(str(root / rel / "rgb" / name), rgb)


def by(cands, label=None, target=None, r=None, q=None):
    return [c for c in cands if (label is None or c.label == label) and (target is None or c.target.name == target)
            and (r is None or c.r == r) and (q is None or c.q == q)]


class MeanColour:
    """Mock PatchEncoder (as in tests/test_evidence.py): patch -> unit vector of mean RGB - 100."""
    patch = 14
    id = "mock-mean-colour"

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        f = rgb.reshape(h // 14, 14, w // 14, 14, 3).astype(np.float32).mean((1, 3)) - 100.0
        return f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-6)


class TestTracks(unittest.TestCase):
    def test_ordered_by_first_appearance_and_stable_ids(self):
        tr = ocid.tracks(scene_labels())
        self.assertEqual([t.name for t in tr], [f"obj{i:02d}" for i in range(1, 10)])
        self.assertEqual([t.label_id for t in tr], list(range(2, 11)))
        self.assertEqual([t.first for t in tr], list(range(9)))
        a = tr[1]
        self.assertEqual(a.areas[1], 1600)
        self.assertEqual(a.areas[2], 400)        # seen through C's hole

    def test_offset_ids_give_identical_episodes(self):
        """One real sequence numbers its objects 256, 257, ...: ordering is by first appearance."""
        c0, e0 = ocid.sequence_candidates(fake_seq(), scene_labels())
        c1, e1 = ocid.sequence_candidates(fake_seq(), scene_labels(id_offset=254))
        self.assertEqual([(c.episode_id, c.label) for c in c0], [(c.episode_id, c.label) for c in c1])
        self.assertEqual(e0, e1)
        self.assertEqual([t.label_id for t in ocid.tracks(scene_labels(254))][:2], [256, 257])

    def test_two_objects_in_one_frame_raises(self):
        labels = scene_labels()
        labels[0][0:5, 100:110] = 3              # obj02 also appears in frame 0
        with self.assertRaises(ValueError):
            ocid.tracks(labels)

    def test_vanished_label_counts_as_fully_hidden(self):
        labels = scene_labels()
        labels[8][labels[8] == 2] = 5            # B fully covered by D in the last frame (label disappears)
        c, _ = ocid.sequence_candidates(fake_seq(), labels)
        occ = by(c, "occluded", "obj01", q=8)
        self.assertTrue(occ)
        self.assertEqual(occ[0].detail["v"], 0.0)
        self.assertEqual(occ[0].detail["occluders"][0]["name"], "obj04")


def shifted(cands, df: int):
    """(target, label, r, q) with frame indices shifted by -df, for comparing scene variants."""
    return sorted((c.target.name, c.label, c.r - df, c.q - df) for c in cands)


class TestSupportSurface(unittest.TestCase):
    def test_surface_ids(self):
        self.assertEqual(ocid.surface_ids(scene_labels()), {1})
        self.assertEqual(ocid.surface_ids(table_scene()), {1, 2})
        self.assertEqual(ocid.surface_ids(floor256_scene()), {1, 256})

    def test_table_is_not_a_target(self):
        """Table sequence: id 2 is the table (frame 0 alone), so obj01 is the first REAL object."""
        tr = ocid.tracks(table_scene())
        self.assertEqual([t.label_id for t in tr], list(range(3, 12)))
        self.assertEqual([t.first for t in tr], list(range(1, 10)))
        c0, e0 = ocid.sequence_candidates(fake_seq(), scene_labels())
        c1, e1 = ocid.sequence_candidates(fake_seq(n=N + 1), table_scene())
        empty_table_q = [c for c in c1 if c.q == 0]      # the empty-table frame: one more removed query each
        self.assertTrue(empty_table_q and all(c.label == "removed" for c in empty_table_q))
        self.assertEqual(shifted(c0, 0), shifted([c for c in c1 if c.q > 0], 1))
        self.assertEqual(e0, e1)
        self.assertNotIn(2, {c.target.label_id for c in c1})

    def test_floor_256_is_not_a_target(self):
        tr = ocid.tracks(floor256_scene())
        self.assertEqual([t.label_id for t in tr], list(range(257, 266)))
        self.assertEqual([t.first for t in tr], list(range(9)))
        c0, _ = ocid.sequence_candidates(fake_seq(), scene_labels())
        c1, _ = ocid.sequence_candidates(fake_seq(), floor256_scene())
        self.assertEqual([(c.episode_id, c.label) for c in c0], [(c.episode_id, c.label) for c in c1])

    def test_surface_pixels_are_not_occluder_cover(self):
        """A target whose pixels become TABLE (id 2) in q is not occluded (lost pixels are not objects)."""
        labels = table_scene()
        e_mask = labels[5] == 7                  # obj05 (E, id 7) as placed at table frame 5
        for f in range(6, N + 1):
            labels[f][e_mask & (labels[f] == 7)] = 2
        c, excl = ocid.sequence_candidates(fake_seq(n=N + 1), labels)
        self.assertFalse(by(c, "occluded", "obj05"))
        self.assertGreater(excl["ambiguous_lost_not_objects"], 0)

    def test_camera_pair_check(self):
        ok = {"a-top": ("A-t-seq01", "top", (1, 2, 3)), "a-bot": ("A-t-seq01", "bottom", (1, 2, 3)),
              "b-top": ("A-t-seq02", "top", (0, 1))}   # only one camera: nothing to compare
        ocid.check_camera_pairs(ok)
        bad = dict(ok, **{"a-bot": ("A-t-seq01", "bottom", (0, 1, 2, 3))})
        with self.assertRaises(ValueError):
            ocid.check_camera_pairs(bad)


class TestTruthRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cands, cls.excl = ocid.sequence_candidates(fake_seq(), scene_labels())

    def test_present(self):
        c = by(self.cands, "visible", "obj01", r=0, q=1)
        self.assertTrue(c)
        self.assertAlmostEqual(c[0].detail["v"], 1.0)
        self.assertEqual(by(self.cands, None, "obj04", r=3, q=8)[0].label, "visible")

    def test_occluded_same_footprint(self):
        c = by(self.cands, "occluded", "obj02", r=1, q=2)
        self.assertEqual(len(c), 1)
        d = c[0].detail
        self.assertAlmostEqual(d["v"], 0.25)
        self.assertEqual([o["name"] for o in d["occluders"]], ["obj03"])
        self.assertAlmostEqual(d["lost_covered_by_objects"], 1.0)
        self.assertGreaterEqual(d["max_occluder_box_iou"], ocid.SAME_FOOTPRINT_IOU)
        self.assertTrue(d["same_footprint"])
        self.assertEqual(c[0].box, (20, 20, 60, 60))

    def test_occluded_not_same_footprint(self):
        c = by(self.cands, "occluded", "obj01", r=0, q=3)
        self.assertEqual(len(c), 1)
        d = c[0].detail
        self.assertAlmostEqual(d["v"], 0.5)
        self.assertEqual(d["occluders"][0]["name"], "obj04")
        self.assertFalse(d["same_footprint"])
        self.assertLess(d["max_occluder_box_iou"], 0.2)

    def test_ambiguous_partial_excluded(self):
        self.assertFalse(by(self.cands, None, "obj05", r=4, q=5))   # v = 0.6
        self.assertGreater(self.excl["ambiguous_partial"], 0)

    def test_lost_to_no_depth_excluded(self):
        self.assertFalse(by(self.cands, None, "obj05", r=4, q=8))   # v = 0.2, half the loss is label 0
        self.assertEqual(self.excl["ambiguous_lost_not_objects"], 1)

    def test_nudged_excluded(self):
        self.assertFalse(by(self.cands, None, "obj07", r=6, q=7))
        self.assertFalse(by(self.cands, None, "obj07", r=6, q=8))
        self.assertEqual(self.excl["nudged"], 2)
        self.assertTrue(by(self.cands, "visible", "obj07", r=7, q=8))   # after the nudge it is a fresh reference

    def test_visible_records_new_objects_on_the_box(self):
        clean = by(self.cands, "visible", "obj01", r=0, q=1)[0].detail
        self.assertEqual((clean["box_new_object_share"], clean["target_lost_share"]), (0.0, 0.0))
        # obj02 (A, box 20..60) in r=1: by q=2 obj03's ring covers the box except the 20 x 20 hole
        occ = by(self.cands, "occluded", "obj02", r=1, q=2)[0].detail
        self.assertAlmostEqual(occ["box_new_object_share"], 1 - 400 / 1600)
        self.assertAlmostEqual(occ["target_lost_share"], 0.75)

    def test_border_and_small_targets_skipped(self):
        self.assertFalse(by(self.cands, target="obj08"))
        self.assertFalse(by(self.cands, target="obj09"))
        self.assertEqual(self.excl["target_touches_border"], 1)
        self.assertFalse(by(self.cands, target="obj06"))           # F is 12 x 30 = 360 px < REF_MIN_PIXELS
        self.assertEqual(self.excl["target_no_clear_reference"], 2)

    def test_reference_must_be_clear(self):
        """obj02 is 75 % hidden from f2 on: only f1 may be its reference."""
        self.assertEqual({c.r for c in by(self.cands, target="obj02")}, {1})

    def test_removed_by_time_reversal(self):
        rem = by(self.cands, "removed", "obj03")
        self.assertEqual({(c.r, c.q) for c in rem}, {(r, q) for r in range(2, 9) for q in (0, 1)})
        clean = by(rem, r=2, q=1)[0].detail
        self.assertFalse(clean["box_other_change"])
        self.assertEqual(clean["box_other_new_share"], 0.0)
        mixed = by(rem, r=2, q=0)[0].detail          # obj02 was placed after q, inside obj03's box
        self.assertTrue(mixed["box_other_change"])
        self.assertAlmostEqual(mixed["box_other_new_share"], 400 / 44 ** 2)
        self.assertGreater(mixed["box_changed_share"], 0)
        self.assertFalse(by(self.cands, "removed", "obj01"))   # placed in the first frame: nothing before it
        for c in by(self.cands, "removed"):
            self.assertLess(c.q, c.target.first)
            self.assertGreaterEqual(c.r, c.target.first)

    def test_episode_ids_unique_and_documented_leak(self):
        """Ids do not spell the label, but q < r iff removed (documented: ids are not blind)."""
        ids = [c.episode_id for c in self.cands]
        self.assertEqual(len(ids), len(set(ids)))
        for c in self.cands:
            for word in ocid.CONDITIONS:
                self.assertNotIn(word, c.episode_id)
            self.assertEqual(c.q < c.r, c.label == "removed")
        self.assertIn("NOT blind", ocid.__doc__)


class TestSamplingAndSplit(unittest.TestCase):
    def test_sampling_caps_deterministic_and_prefers_same_footprint(self):
        cands, _ = ocid.sequence_candidates(fake_seq(), scene_labels())
        a, b = ocid.sample(cands), ocid.sample(list(reversed(cands)))
        self.assertEqual([c.episode_id for c in a], [c.episode_id for c in b])
        n = Counter(c.label for c in a)
        for label, cap in ocid.CAP_PER_SEQ.items():
            self.assertLessEqual(n[label], cap)
        per_t = Counter((c.label, c.target.name) for c in a)
        for (label, _), k in per_t.items():
            self.assertLessEqual(k, ocid.CAP_PER_TARGET[label])
        occ = [c for c in a if c.label == "occluded"]
        self.assertTrue(occ[0].detail["same_footprint"])
        g = ocid.sample(c for c in cands)                  # a one-shot iterator gives the same result
        self.assertEqual([c.episode_id for c in g], [c.episode_id for c in a])

    def test_split_stable_by_arrangement(self):
        top = fake_seq("SYN10/table/top/seq01")
        bot = fake_seq("SYN10/table/bottom/seq01")
        self.assertEqual(top.arrangement, bot.arrangement)
        self.assertEqual(top.arrangement, "SYN10-table-seq01")
        self.assertEqual(ocid.split_of(top.arrangement), ocid.split_of(bot.arrangement))
        self.assertEqual((top.stop, bot.stop), (0, 1))
        cat = fake_seq("SYN10/floor/top/box/seq07")
        self.assertEqual((cat.category, cat.arrangement, cat.key), ("box", "SYN10-floor-seq07", "SYN10-floor-top-box-seq07"))
        # pinned values: changing SPLIT_SALT or the hash would move arrangements between dev and val
        pinned = {"ARID10-table-seq01": "dev", "ARID10-floor-seq05": "dev", "ARID20-table-seq11": "dev",
                  "YCB10-table-seq22": "dev", "ARID20-floor-seq04": "val", "YCB10-table-seq25": "val"}
        self.assertEqual({a: ocid.split_of(a) for a in pinned}, pinned)
        self.assertEqual([a for a in (f"ARID20-floor-seq{i:02d}" for i in range(1, 27)) if ocid.split_of(a) == "val"],
                         ["ARID20-floor-seq04", "ARID20-floor-seq07", "ARID20-floor-seq17", "ARID20-floor-seq19"])
        frac = np.mean([ocid.split_of(f"X{i}") == "val" for i in range(4000)])
        self.assertAlmostEqual(frac, ocid.VAL_FRACTION, delta=0.03)


class TestBuildOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "OCID-dataset"
        labels = table_scene()                   # real table layout: table = id 2 from frame 0
        write_sequence(root, "SYN10/table/top/seq01", labels, 0)
        write_sequence(root, "SYN10/table/bottom/seq01", labels, 5)
        (root / "SYN10/table/top/seq01/rgb/result_2099-01-01-00-00-00.png").write_bytes(b"")   # rgb only
        cls.root, cls.out = root, Path(cls.tmp.name) / "episodes"
        cls.manifest = ocid.build(root, cls.out, log=lambda *a: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_discover(self):
        seqs = ocid.discover(self.root)
        self.assertEqual([s.key for s in seqs], ["SYN10-table-bottom-seq01", "SYN10-table-top-seq01"])
        self.assertEqual([len(s.frames) for s in seqs], [N + 1, N + 1])
        self.assertEqual([s.n_unmatched for s in seqs], [0, 1])
        self.assertEqual(self.manifest["n_unmatched_frames"], 1)
        self.assertEqual(self.manifest["sequences_by_n_surface_ids"], {"2": 2})

    def test_empty_sequence_raises_clearly(self):
        with self.assertRaisesRegex(ValueError, "no frames"):
            ocid.load_labels(ocid.OcidSequence(self.root, "SYN10/table/top/seq09", ()))

    def test_jobs_annotations_truth(self):
        jobs = json.loads((self.out / "jobs.json").read_text())
        self.assertEqual(jobs["schema"], ev.JOBS_SCHEMA)
        truth = json.loads((self.out / "truth.json").read_text())
        details = [json.loads(l) for l in (self.out / "truth_detail.jsonl").read_text().splitlines()]
        self.assertEqual({j["episode_id"] for j in jobs["jobs"]}, set(truth))
        self.assertEqual({d["episode_id"] for d in details}, set(truth))
        self.assertTrue(set(truth.values()) <= set(ocid.CONDITIONS))
        self.assertEqual(set(truth.values()), set(ocid.CONDITIONS))
        for j in jobs["jobs"]:
            self.assertEqual(set(j), {"episode_id", "stop", "target", "annotation", "query"})
            self.assertTrue(Path(j["annotation"]).is_absolute() and Path(j["query"]).is_file())
            ann = ev.load_annotation(j["annotation"])      # the frozen module's own validator
            self.assertEqual(ann["stop"], j["stop"])
            self.assertTrue(Path(ann["reference"]).is_absolute() and Path(ann["reference"]).is_file())
            self.assertIn(j["target"], ann["targets"])
            self.assertEqual(j["stop"], 0 if "-top-" in j["episode_id"] else 1)
        self.assertNotIn(2, {d["target_label_id"] for d in details})     # the table is never a target
        for d in details:
            self.assertEqual(d["schema"], ocid.DETAIL_SCHEMA)
            if d["label"] in ("visible", "occluded"):
                self.assertIn("box_new_object_share", d)
            self.assertEqual(d["label"], truth[d["episode_id"]])
            self.assertEqual(d["split"], ocid.split_of("SYN10-table-seq01"))
            if d["label"] == "occluded":
                self.assertIn("same_footprint", d)
                self.assertIn("occluders", d)
            if d["label"] == "removed":
                self.assertIn("box_other_change", d)
        m = self.manifest
        self.assertEqual(m["loader_version"], ocid.LOADER_VERSION)
        self.assertEqual(m["loader_sha256"], ocid.file_sha256(ocid.__file__))
        self.assertEqual(m["n_episodes"], len(truth))
        self.assertEqual(sum(m["counts"]["by_label"].values()), len(truth))
        self.assertEqual(m["constants"]["PRESENT_MIN"], ocid.PRESENT_MIN)
        self.assertEqual(json.loads((self.out / "MANIFEST.json").read_text())["counts"], m["counts"])

    def test_cli_refuses_to_overwrite(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ocid.main(["--root", str(self.root), "--out", str(self.out)]), 2)
        with self.assertRaises(FileExistsError):
            ocid.build(self.root, self.out, log=lambda *a: None)

    def test_force_rebuild_removes_stale_outputs(self):
        out = Path(self.tmp.name) / "episodes_force"
        ocid.build(self.root, out, log=lambda *a: None)
        stale = [out / "annotations" / "STALE-seq" / "x.json", out / "evidence" / "0.0.0" / "evidence.jsonl"]
        for p in stale:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = ocid.main(["--root", str(self.root), "--out", str(out), "--force", "--limit-sequences", "1"])
        self.assertEqual(rc, 0)
        for p in stale:
            self.assertFalse(p.exists())
        m = json.loads((out / "MANIFEST.json").read_text())
        self.assertEqual((m["limit_sequences"], m["n_sequences"]), (1, 1))
        self.assertTrue((out / "annotations").is_dir())

    def test_force_refuses_foreign_manifest(self):
        out = Path(self.tmp.name) / "foreign"
        (out / "annotations").mkdir(parents=True)
        (out / "MANIFEST.json").write_text(json.dumps({"schema": "something-else"}))
        with self.assertRaises(ValueError):
            ocid.build(self.root, out, log=lambda *a: None, force=True)
        self.assertTrue((out / "annotations").is_dir())

    def test_run_evidence_with_mocks(self):
        split = ocid.split_of("SYN10-table-seq01")          # dev; the only arrangement in this build
        out = run_evidence.default_out_dir(self.out / "jobs.json", split)
        self.assertEqual(out, self.out / "evidence" / ev.EVIDENCE_VERSION / split)
        meta = run_evidence.run(self.out / "jobs.json", out, MeanColour(), lambda bgr: [], split=split)
        truth = json.loads((self.out / "truth.json").read_text())
        self.assertEqual((meta["split"], meta["n_jobs"], meta["n_jobs_in_file"]), (split, len(truth), len(truth)))
        self.assertEqual(meta["align"], "identity_align")
        self.assertEqual(meta["jobs_sha256"], ocid.file_sha256(self.out / "jobs.json"))
        self.assertEqual(meta["truth_sha256"], ocid.file_sha256(self.out / "truth.json"))
        self.assertEqual(meta["manifest_sha256"], ocid.file_sha256(self.out / "MANIFEST.json"))
        self.assertEqual((meta["loader_version"], meta["loader_sha256"]),
                         (self.manifest["loader_version"], self.manifest["loader_sha256"]))
        store = EvidenceStore.from_jsonl(out / "evidence.jsonl")
        self.assertEqual(set(store.episode_ids()), set(truth))
        diags = [json.loads(l) for l in (out / "diagnostics.jsonl").read_text().splitlines()]
        self.assertEqual(len(diags), len(truth))
        self.assertTrue(all(d["align_ok"] for d in diags))
        with self.assertRaises(FileExistsError):
            run_evidence.run(self.out / "jobs.json", out, MeanColour(), lambda bgr: [], split=split)
        with contextlib.redirect_stderr(io.StringIO()):     # refused before any model loads
            self.assertEqual(run_evidence.main(["--jobs", str(self.out / "jobs.json"), "--split", split]), 2)
        by_id = {d["episode_id"]: d for d in diags}
        for eid, label in truth.items():   # sanity on synthetic colours: an unchanged target matches perfectly
            if label == "visible":
                self.assertGreater(by_id[eid]["match_score"], 0.99)


class TestRunEvidenceSplit(unittest.TestCase):
    """One split per run (split per episode from truth_detail.jsonl); val only with allow_val; the scorer
    scores what was run. Two arrangements: SYN10-table-seq01 (dev, top and bottom camera) and
    SYN10-table-seq02 (val, top camera only)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "OCID-dataset"
        labels = table_scene()
        write_sequence(root, "SYN10/table/top/seq01", labels, 0)
        write_sequence(root, "SYN10/table/bottom/seq01", labels, 5)
        write_sequence(root, "SYN10/table/top/seq02", labels, 0)
        cls.out = Path(cls.tmp.name) / "episodes"
        ocid.build(root, cls.out, log=lambda *a: None)
        cls.jobs = cls.out / "jobs.json"
        cls.details = [json.loads(l) for l in (cls.out / "truth_detail.jsonl").read_text().splitlines()]
        cls.ids = {sp: {d["episode_id"] for d in cls.details if d["split"] == sp} for sp in run_evidence.SPLITS}
        cls.dev_dir = run_evidence.default_out_dir(cls.jobs, "dev")
        cls.dev_meta = run_evidence.run(cls.jobs, cls.dev_dir, MeanColour(), lambda bgr: [], split="dev")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy_episode_files(self, name: str) -> Path:
        d = Path(self.tmp.name) / name
        d.mkdir()
        for f in ("jobs.json", "truth.json", "truth_detail.jsonl", "MANIFEST.json"):
            shutil.copy(self.out / f, d / f)
        return d

    def test_fixture_has_both_splits(self):
        self.assertEqual((ocid.split_of("SYN10-table-seq01"), ocid.split_of("SYN10-table-seq02")), ("dev", "val"))
        self.assertTrue(self.ids["dev"] and self.ids["val"])
        self.assertEqual(len(self.details), len(self.ids["dev"]) + len(self.ids["val"]))

    def test_default_out_dir_is_per_split(self):
        for sp in run_evidence.SPLITS:
            self.assertEqual(run_evidence.default_out_dir(self.jobs, sp),
                             self.out / "evidence" / ev.EVIDENCE_VERSION / sp)
        with self.assertRaises(ValueError):
            run_evidence.default_out_dir(self.jobs, "test")

    def test_dev_run_holds_only_dev(self):
        meta = self.dev_meta
        self.assertEqual((meta["split"], meta["n_jobs"], meta["n_jobs_in_file"]),
                         ("dev", len(self.ids["dev"]), len(self.details)))
        self.assertEqual(meta["n_jobs_by_split"], {"dev": len(self.ids["dev"]), "val": len(self.ids["val"])})
        self.assertEqual(meta["truth_detail_sha256"], ocid.file_sha256(self.out / "truth_detail.jsonl"))
        self.assertEqual(json.loads((self.dev_dir / "evidence_run.json").read_text()), meta)
        self.assertEqual(set(EvidenceStore.from_jsonl(self.dev_dir / "evidence.jsonl").episode_ids()), self.ids["dev"])
        diags = [json.loads(l) for l in (self.dev_dir / "diagnostics.jsonl").read_text().splitlines()]
        in_file_order = [j["episode_id"] for j in json.loads(self.jobs.read_text())["jobs"]
                         if j["episode_id"] in self.ids["dev"]]
        self.assertEqual([d["episode_id"] for d in diags], in_file_order)
        with contextlib.redirect_stderr(io.StringIO()):     # default dev dir is taken: refused before any model loads
            self.assertEqual(run_evidence.main(["--jobs", str(self.jobs), "--split", "dev"]), 2)

    def test_split_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            run_evidence.main(["--jobs", str(self.jobs)])
        self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(TypeError):
            run_evidence.run(self.jobs, Path(self.tmp.name) / "nosplit", MeanColour(), lambda bgr: [])
        with self.assertRaises(ValueError):
            run_evidence.run(self.jobs, Path(self.tmp.name) / "badsplit", MeanColour(), lambda bgr: [], split="all")
        self.assertFalse((Path(self.tmp.name) / "badsplit").exists())

    def test_val_refused_unless_allowed(self):
        out = Path(self.tmp.name) / "ev_val"
        with self.assertRaisesRegex(ValueError, "allow_val"):
            run_evidence.run(self.jobs, out, MeanColour(), lambda bgr: [], split="val")
        self.assertFalse(out.exists())
        with contextlib.redirect_stderr(io.StringIO()):     # refused before any model loads (no torch needed)
            self.assertEqual(run_evidence.main(["--jobs", str(self.jobs), "--split", "val"]), 2)
        self.assertFalse(run_evidence.default_out_dir(self.jobs, "val").exists())
        meta = run_evidence.run(self.jobs, out, MeanColour(), lambda bgr: [], split="val", allow_val=True)
        self.assertEqual((meta["split"], meta["n_jobs"]), ("val", len(self.ids["val"])))
        self.assertEqual(set(EvidenceStore.from_jsonl(out / "evidence.jsonl").episode_ids()), self.ids["val"])
        with self.assertRaisesRegex(ValueError, "val"):     # the scorer refuses val views the same way
            score_views.score(out)
        self.assertEqual(score_views.score(out, allow_val=True)["n_views"], len(self.ids["val"]))

    def test_job_without_detail_record_is_refused(self):
        d = self.copy_episode_files("no_detail")
        lines = (d / "truth_detail.jsonl").read_text().splitlines()
        (d / "truth_detail.jsonl").write_text("\n".join(lines[1:]) + "\n")
        with self.assertRaisesRegex(ValueError, "no truth_detail record"):
            run_evidence.select_jobs(d / "jobs.json", "dev")
        (d / "truth_detail.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            run_evidence.select_jobs(d / "jobs.json", "dev")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_evidence.main(["--jobs", str(d / "jobs.json"), "--split", "dev"]), 2)

    def test_detail_split_must_agree_with_manifest(self):
        d = self.copy_episode_files("swapped_manifest")
        m = json.loads((d / "MANIFEST.json").read_text())
        m["arrangements"] = {"dev": m["arrangements"]["val"], "val": m["arrangements"]["dev"]}
        (d / "MANIFEST.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "MANIFEST"):
            run_evidence.select_jobs(d / "jobs.json", "dev")

    def test_score_dev_run(self):
        a = score_views.score(self.dev_dir)
        n = len(self.ids["dev"])
        self.assertEqual((a["split"], a["n_views"], a["n_split_episodes_in_truth"]), ("dev", n, n))
        self.assertEqual((a["missing_from_evidence"], a["flag_recompute_mismatches"]), ([], []))
        self.assertEqual(a["truth_hashes_checked_against_run"], {"truth.json": True, "truth_detail.jsonl": True})
        diags = {}
        for line in (self.dev_dir / "diagnostics.jsonl").read_text().splitlines():
            diags[json.loads(line)["episode_id"]] = json.loads(line)
        truth = json.loads((self.out / "truth.json").read_text())
        for eid, t, v in a["views"]:
            self.assertEqual(t, truth[eid])
            self.assertEqual(v, score_views.verdict(diags[eid]["flags_placeholder_thresholds"]))
        cm = a["confusion"]["overall"]
        self.assertEqual(sum(sum(row.values()) for row in cm.values()), n)
        for g in ("by_set", "by_camera_stop"):
            self.assertEqual(sum(sum(sum(row.values()) for row in c.values()) for c in a["confusion"][g].values()), n)
        self.assertEqual(set(a["confusion"]["by_camera_stop"]), {"stop 0 (top)", "stop 1 (bottom)"})
        m = a["metrics"]["overall"]
        self.assertEqual(m["false_removal_rate"]["k"], cm["visible"]["removed"] + cm["occluded"]["removed"])
        self.assertEqual(m["false_removal_rate"]["n"], sum(cm["visible"].values()) + sum(cm["occluded"].values()))
        self.assertEqual(m["removal_recall"]["k"], cm["removed"]["removed"])
        self.assertEqual(m["coverage"]["k"], sum(cm[t]["present"] + cm[t]["removed"] for t in ocid.CONDITIONS))
        dev = [d for d in self.details if d["split"] == "dev"]
        self.assertEqual(a["same_footprint"]["n"], sum(1 for d in dev if d.get("same_footprint")))
        self.assertGreater(a["same_footprint"]["n"], 0)
        self.assertEqual(a["same_footprint"]["with_target_iou_detection"], 0)       # the mock detector finds nothing
        self.assertEqual(sum(g["n"] for g in a["removed_by_box_other_change"].values()),
                         sum(1 for d in dev if d["label"] == "removed"))
        for t in ocid.CONDITIONS:
            self.assertEqual(a["score_quantiles"][t]["match_score"]["n"], sum(1 for d in dev if d["label"] == t))
        for key in ("by_target_box_area", "by_target_fill"):       # every view falls in exactly one bin
            self.assertEqual(sum(x["metrics"]["n"] for x in a[key].values()), n)
        self.assertEqual(list(a["by_target_fill"]), [b[0] for b in score_views.FILL_BINS if b[0] in a["by_target_fill"]])
        out = Path(self.tmp.name) / "analysis_dev"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(score_views.main(["--evidence-dir", str(self.dev_dir), "--out-dir", str(out)]), 0)
        self.assertEqual(json.loads((out / "analysis.json").read_text()), json.loads(json.dumps(a)))
        self.assertIn("same_footprint", (out / "analysis.md").read_text())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(score_views.main(["--evidence-dir", str(self.dev_dir), "--out-dir", str(out)]), 2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(score_views.main(["--evidence-dir", str(self.dev_dir), "--out-dir", str(out),
                                               "--force"]), 0)

    def test_scorer_counts_detections_dropped_as_the_target(self):
        """A detection on the occluder's box (IoU >= target_iou with the target box) on every same_footprint view."""
        d = Path(self.tmp.name) / "ev_dev_injected"
        shutil.copytree(self.dev_dir, d)
        details = {x["episode_id"]: x for x in self.details}
        rows = [json.loads(l) for l in (d / "diagnostics.jsonl").read_text().splitlines()]
        n_sf = 0
        for r in rows:
            det = details[r["episode_id"]]
            if det.get("same_footprint"):
                occ = max(det["occluders"], key=lambda o: o["box_iou"])
                r["detections"] = [{"box": occ["box_q"], "cls": "box", "conf": 0.9, "iou_box": round(occ["box_iou"], 3)}]
                n_sf += 1
        (d / "diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        sf = score_views.score(d)["same_footprint"]
        self.assertGreater(n_sf, 0)
        self.assertEqual((sf["n"], sf["with_target_iou_detection"], sf["with_target_iou_detection_matching_occluder_box"],
                          sf["with_target_iou_detection_closer_to_occluder"]), (n_sf, n_sf, n_sf, n_sf))

    def test_scorer_refuses_truth_the_run_did_not_record(self):
        bad = Path(self.tmp.name) / "ev_dev_badhash"
        shutil.copytree(self.dev_dir, bad)
        meta = json.loads((bad / "evidence_run.json").read_text())
        meta["truth_sha256"] = "0" * 64
        (bad / "evidence_run.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "sha256"):
            score_views.score(bad)


class TestScoreViewsPieces(unittest.TestCase):
    def test_verdict_rule(self):
        def f(s, c, o):
            return {"strong_match": s, "clear_miss": c, "flagged_occluded": o}
        self.assertEqual(score_views.verdict(f(True, False, True)), "present")    # a strong match wins (StoppingRule)
        self.assertEqual(score_views.verdict(f(False, True, False)), "removed")
        self.assertEqual(score_views.verdict(f(False, False, True)), "occluded")
        self.assertEqual(score_views.verdict(f(False, False, False)), "unknown")

    def test_quantile_is_numpy_linear(self):
        xs = np.random.default_rng(1).random(37).tolist()
        for q in score_views.QUANTILES:
            self.assertAlmostEqual(score_views.quantile(xs, q), float(np.quantile(xs, q)), places=12)
        self.assertEqual(score_views.quantile([0.3], 0.95), 0.3)
        s = score_views.summary([0.0, 1.0])
        self.assertEqual((s["n"], s["min"], s["p50"], s["max"]), (2, 0.0, 0.5, 1.0))
        self.assertAlmostEqual(s["p05"], 0.05)
        self.assertEqual(score_views.summary([]), {"n": 0})

    def test_target_iou_detections(self):
        dets = [{"box": [10, 10, 50, 50], "cls": "box", "conf": 0.6, "iou_box": 0.62},
                {"box": [200, 10, 240, 50], "cls": "cup", "conf": 0.5, "iou_box": 0.0},
                {"box": [12, 12, 50, 52], "cls": "can", "conf": 0.4, "iou_box": 0.5}]
        t = score_views.target_iou_detections({"detections": dets, "detail": {"occluders": [{"box_q": [10, 10, 50, 50]}]}}, 0.5)
        self.assertEqual([d["cls"] for d in t], ["box", "can"])
        self.assertEqual(t[0]["max_iou_with_occluder_box"], 1.0)
        self.assertIsNone(score_views.target_iou_detections({"detections": dets, "detail": {}}, 0.5)[0]
                          ["max_iou_with_occluder_box"])


if __name__ == "__main__":
    unittest.main()
