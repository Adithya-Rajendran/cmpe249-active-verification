"""perception/evidence.py v0: the three scores on synthetic images with a mocked encoder and detector,
the jobs -> EvidenceStore path, verify.run's real-data path, and one weights-gated integration test."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from perception import evidence as ev
from verify import run
from verify.schema import REPO_ROOT, EvidenceRecord, EvidenceStore, Observation, Positions, Thresholds
from verify.stopping import StoppingRule

BOX = (130.0, 80.0, 170.0, 120.0)   # 40 x 40 target in a 300 x 200 frame
OCC = (120, 60, 160, 125)           # occluder rectangle (x0, y0, x1, y1) covering 75 % of the box


class MeanColour:
    """Mock PatchEncoder: each 14 px patch -> unit vector of (mean RGB - 100). Records input shapes."""
    patch = 14

    def __init__(self) -> None:
        self.shapes: list[tuple[int, int]] = []

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        self.shapes.append((h, w))
        k = self.patch
        f = rgb.reshape(h // k, k, w // k, k, 3).astype(np.float32).mean((1, 3)) - 100.0
        return f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-6)


class FixedDetector:
    """Mock Detector returning fixed boxes given in FULL-frame pixels (converted to the crop it gets)."""

    def __init__(self, boxes=(), frame_box=BOX, p: ev.Params = ev.Params()) -> None:
        self.boxes = boxes
        self.origin = ev.grow(frame_box, p.det_context, 300, 200)[:2]

    def __call__(self, bgr: np.ndarray) -> list[ev.Detection]:
        ox, oy = self.origin
        return [ev.Detection((b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy), "bottle", 0.9) for b in self.boxes]


def scene(seed: int = 0) -> np.ndarray:
    """Textured grey background (so the Laplacian has something to measure) with a red|green target."""
    rng = np.random.default_rng(seed)
    img = np.clip(128 + rng.normal(0, 12, (200, 300, 1)), 0, 255).repeat(3, axis=2).astype(np.uint8)
    x0, y0, x1, y1 = (int(v) for v in BOX)
    img[y0:y1, x0:x1] = (30, 30, 220)               # BGR red left half
    img[y0:y1, (x0 + x1) // 2:x1] = (30, 200, 30)   # green right half (a flat target would match any sliver)
    return img


def removed(ref: np.ndarray) -> np.ndarray:
    q = ref.copy()
    x0, y0, x1, y1 = (int(v) for v in BOX)
    q[y0:y1, x0:x1] = scene(1)[y0 - 60:y1 - 60, x0:x1]   # background texture from another noise draw
    return q


def occluded(ref: np.ndarray) -> np.ndarray:
    q = ref.copy()
    x0, y0, x1, y1 = OCC
    q[y0:y1, x0:x1] = (200, 90, 20)   # BGR blue canister
    return q


def evid(query, dets=(), align=ev.identity_align, ref=None):
    ref = scene() if ref is None else ref
    return ev.evidence_for(ref, BOX, query, 2, MeanColour(), FixedDetector(dets), align=align)


class TestScores(unittest.TestCase):
    def test_present(self) -> None:
        rec, diag = evid(scene())
        self.assertAlmostEqual(rec.match_score, 1.0, places=5)
        self.assertEqual(rec.obstruction, 0.0)
        self.assertAlmostEqual(rec.quality, 1.0, places=5)
        self.assertTrue(diag["align_ok"])    # explicit identity counts as registered

    def test_refused_registration_zeroes_quality(self) -> None:
        rec, diag = evid(removed(scene()), align=lambda r, x: None)
        self.assertFalse(diag["align_ok"])
        self.assertEqual(rec.quality, 0.0)
        self.assertFalse(ev.flags(rec)["clear_miss"])   # an unregistered view is never absence evidence

    def test_removed_is_a_clear_miss_not_occluded(self) -> None:
        rec, diag = evid(removed(scene()))
        self.assertLess(rec.match_score, 0.3)
        self.assertGreater(diag["obstruction_parts"]["changed"], 0.5)
        self.assertEqual(diag["obstruction_parts"]["novelty"], 0.0)   # the box now looks like background
        self.assertEqual(rec.obstruction, 0.0)   # change without a detection or novelty is not obstruction
        self.assertTrue(ev.flags(rec)["clear_miss"])

    def test_occluded_needs_change_and_detection(self) -> None:
        rec, diag = evid(occluded(scene()), dets=[OCC])
        self.assertAlmostEqual(rec.obstruction, 0.75, delta=0.03)
        self.assertTrue(ev.flags(rec)["flagged_occluded"])
        self.assertLess(rec.match_score, 0.6)
        # no detection (the pilot's book): the novelty cue still flags it, since the occluder looks like
        # neither the reference target nor the background
        rec2, d2 = evid(occluded(scene()), dets=[])
        self.assertEqual(d2["obstruction_parts"]["det_obstruction"], 0.0)
        self.assertGreater(d2["obstruction_parts"]["novelty"], 0.3)
        self.assertTrue(ev.flags(rec2)["flagged_occluded"])
        self.assertFalse(ev.flags(rec2)["clear_miss"])

    def test_known_looking_occluder_needs_the_detector(self) -> None:
        # an occluder that looks like background is explained by it (no novelty): only the detector cue
        # can flag it (the pilot's Clorox, which also stood beside the cup in the reference)
        q = scene().copy()
        x0, y0, x1, y1 = OCC
        q[y0:y1, x0:x1] = scene(2)[y0:y1, x0 + 100:x1 + 100]   # background-like texture in front
        rec, d = evid(q, dets=[])
        self.assertLess(d["obstruction_parts"]["novelty"], 0.3)
        self.assertFalse(ev.flags(rec)["flagged_occluded"])
        rec2, d2 = evid(q, dets=[OCC])
        self.assertTrue(ev.flags(rec2)["flagged_occluded"])

    def test_removal_onto_a_mat_seen_only_beside_the_target(self) -> None:
        # review 2026-10-16: a surface visible in the reference only inside the search-pad ring (a mat
        # under the target) must explain a removal; 0.2.0 left the ring out and read novelty 1.0
        ref = scene().copy()
        x0, y0, x1, y1 = (int(v) for v in BOX)
        ref[y0 - 8:y1 + 8, x0 - 8:x1 + 8] = (200, 60, 60)   # blue mat, 8 px past the box (pad is 10 px)
        ref[y0:y1, x0:x1] = scene()[y0:y1, x0:x1]
        q = ref.copy()
        q[y0:y1, x0:x1] = (200, 60, 60)
        rec, d = evid(q, ref=ref)
        self.assertEqual(d["obstruction_parts"]["novelty"], 0.0)
        self.assertTrue(ev.flags(rec)["clear_miss"])

    def test_dimmer_lighting_is_not_blur(self) -> None:
        # review 2026-10-16: Laplacian variance scales with gain^2; at half brightness the ungained ratio
        # was ~0.25 (quality below tau_q), so removals under the dimmer lighting L2 became unknown
        dim = (scene().astype(np.float32) * 0.5).astype(np.uint8)
        rec, d = evid(dim)
        self.assertGreater(d["quality_parts"]["sharpness"], 0.9)
        blurred_dim = cv2.GaussianBlur(dim, (0, 0), 3)
        _, d2 = evid(blurred_dim)
        self.assertLess(d2["quality_parts"]["sharpness"], 0.5)

    def test_empty_ring_is_unmeasured_not_sharp(self) -> None:
        # review 2026-10-16: a box filling the context window left an empty ring; var() was NaN and
        # min(1, nan) read as sharpness 1.0
        big = (20.0, 20.0, 280.0, 180.0)
        ref = scene()
        q, parts = ev.quality_score(ref, cv2.GaussianBlur(ref, (0, 0), 3), np.ones(ref.shape[:2], bool), big)
        self.assertEqual(parts["sharpness"], 0.0)
        self.assertEqual(q, 0.0)

    def test_sharpness_ignores_a_neighbour_that_left(self) -> None:
        # a strongly textured neighbour (8 px checkerboard, like a printed label) in the ring is replaced
        # by flat grey (the neighbour was taken away): the old ring read this as blur; the change-masked
        # ring does not
        ref = scene().copy()
        yy, xx = np.mgrid[60:140, 60:115]
        ref[60:140, 60:115] = np.where(((yy // 8) + (xx // 8)) % 2 == 0, 40, 220)[..., None].astype(np.uint8)
        q = ref.copy()
        q[60:140, 60:115] = 128
        rec, d = evid(q, ref=ref)
        self.assertGreater(d["quality_parts"]["sharpness"], 0.8)
        self.assertLess(d["quality_parts"]["ring_kept"], 1.0)

    def test_static_clutter_and_target_detection_are_not_obstruction(self) -> None:
        rec, diag = evid(scene(), dets=[OCC, BOX])   # unchanged pixels; the target's own box
        self.assertEqual(rec.obstruction, 0.0)
        self.assertGreater(diag["obstruction_parts"]["det_cover"], 0.5)
        # the target's own detection (IoU 1) is never counted, even where pixels changed
        rec2, d2 = evid(removed(scene()), dets=[BOX])
        self.assertEqual(d2["obstruction_parts"]["det_cover"], 0.0)
        self.assertEqual(rec2.obstruction, 0.0)

    def test_gain_change_is_not_a_change(self) -> None:
        q = np.clip(scene().astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
        rec, diag = evid(q)
        self.assertLess(diag["obstruction_parts"]["changed"], 0.05)
        self.assertGreater(rec.match_score, 0.95)

    def test_quality_drops_for_blur_and_clipping(self) -> None:
        blurred = cv2.GaussianBlur(scene(), (0, 0), 3)
        rec, diag = evid(blurred)
        self.assertLess(diag["quality_parts"]["sharpness"], 0.5)
        self.assertLess(rec.quality, 0.5)
        dark = np.zeros_like(scene())
        rec2, d2 = evid(dark)
        self.assertLess(d2["quality_parts"]["exposure"], 0.05)
        self.assertLess(rec2.quality, 0.05)

    def test_subpixel_registration_keeps_sharpness(self) -> None:
        # The unchanged frame registered through a 0.4 px shift: bilinear resampling alone used to read
        # as sharpness ~0.2 (now.jpg, 0.5 px), i.e. quality below tau_q on a perfect view.
        S = np.float64([[1, 0, 0.4], [0, 1, 0.4], [0, 0, 1]])
        rec, diag = evid(scene(), align=lambda r, x: S)
        self.assertGreater(diag["quality_parts"]["sharpness"], 0.9)

    def test_registration_undoes_a_shift(self) -> None:
        M = np.float32([[1, 0, 9], [0, 1, -6]])
        q = cv2.warpAffine(scene(), M, (300, 200), borderMode=cv2.BORDER_REFLECT)
        H = np.array([[1, 0, -9], [0, 1, 6], [0, 0, 1]], np.float64)   # query -> ref
        rec, diag = evid(q, align=lambda r, x: H)
        self.assertTrue(diag["align_ok"])
        self.assertGreater(rec.match_score, 0.99)
        self.assertLess(diag["obstruction_parts"]["changed"], 0.05)

    def test_size_mismatch_without_registration_raises(self) -> None:
        with self.assertRaises(ValueError):
            evid(scene()[:150], align=lambda r, x: None)

    def test_encoder_sees_native_resolution(self) -> None:
        enc = MeanColour()
        ev.match_score(scene()[:, :, ::-1], scene()[:, :, ::-1], BOX, enc)
        # context window 120 x 120 px trimmed to a multiple of 14 -> 112 x 112; never 224
        self.assertEqual(enc.shapes, [(112, 112), (112, 112)])

    def test_search_pad_tolerates_a_small_shift_without_registration(self) -> None:
        q = cv2.warpAffine(scene(), np.float32([[1, 0, 8], [0, 1, 0]]), (300, 200), borderMode=cv2.BORDER_REFLECT)
        rec, _ = evid(q)
        self.assertGreater(rec.match_score, 0.9)

    def test_flags_match_the_stopping_rule(self) -> None:
        rule = StoppingRule(Thresholds())
        for m, q, u in [(0.7, 0.9, 0.0), (0.1, 0.9, 0.1), (0.1, 0.9, 0.5), (0.1, 0.3, 0.1)]:
            rec = EvidenceRecord(2, m, q, u)
            o = Observation.from_record(rec, 0, Positions.default())
            self.assertEqual(ev.flags(rec), {"strong_match": rule.is_strong_match(o),
                                             "clear_miss": rule.is_clear_miss(o),
                                             "flagged_occluded": rule.is_flagged_occluded(o)})


class TestJobs(unittest.TestCase):
    def test_jobs_write_an_evidence_store(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cv2.imwrite(str(d / "ref.png"), scene())
            cv2.imwrite(str(d / "rem.png"), removed(scene()))
            ann = {"schema": ev.ANNOTATION_SCHEMA, "reference": str(d / "ref.png"), "stop": 2,
                   "targets": {"t": {"box": list(BOX)}}}
            (d / "ann.json").write_text(json.dumps(ann))
            jobs = [{"episode_id": "A01-visible-L1", "stop": 2, "target": "t", "annotation": str(d / "ann.json"),
                     "query": str(d / "ref.png")},
                    {"episode_id": "A01-removed-L1", "stop": 2, "target": "t", "annotation": str(d / "ann.json"),
                     "query": str(d / "rem.png")}]
            store, diags = ev.run_jobs(jobs, MeanColour(), FixedDetector(), align=ev.identity_align)
            store.to_jsonl(d / "e.jsonl")
            back = EvidenceStore.from_jsonl(d / "e.jsonl")
            self.assertFalse(back.synthetic)
            self.assertAlmostEqual(back.get("A01-visible-L1", 2).match_score, 1.0, places=5)
            self.assertLess(back.get("A01-removed-L1", 2).match_score, 0.3)
            self.assertIn("flags_placeholder_thresholds", diags[0])
            jobs[0]["stop"] = 3
            with self.assertRaises(ValueError):   # job stop must match the annotation's reference stop
                ev.run_jobs(jobs, MeanColour(), FixedDetector(), align=ev.identity_align)

    def test_bad_annotation_schema(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            p.write_text(json.dumps({"schema": "x", "targets": {}}))
            with self.assertRaises(ValueError):
                ev.load_annotation(p)


class TestRunRealPath(unittest.TestCase):
    """verify.run --manifest --evidence: replays real-format evidence through the same stages."""

    def _write(self, d: Path, drop_one: bool = False, synthetic: bool = False) -> list[str]:
        man = {"schema": "cmpe249.arrangements.v1", "arrangements": [
            {"arrangement_id": "PA01", "target_id": "study_mug", "split": "dev", "lightings": ["L1"],
             "indistinguishable": False},
            {"arrangement_id": "PA02", "target_id": "study_mug", "split": "test", "lightings": ["L1"],
             "indistinguishable": False}]}
        (d / "arr.json").write_text(json.dumps(man))
        store = EvidenceStore(synthetic=synthetic)
        for a in ("PA01", "PA02"):
            for cond in ("visible", "occluded", "removed"):
                for s in range(5):
                    if drop_one and (a, cond, s) == ("PA02", "removed", 4):
                        continue
                    m = 0.9 if cond == "visible" or (cond == "occluded" and s in (0, 4)) else 0.1
                    u = 0.8 if cond in ("occluded", "removed") and s in (1, 2, 3) else 0.0
                    store.put(f"{a}-{cond}-L1", EvidenceRecord(s, m, 0.9, u))
        store.to_jsonl(d / "ev.jsonl")
        return ["--manifest", str(d / "arr.json"), "--evidence", str(d / "ev.jsonl"), "--budgets", "0,2",
                "--primary-budget", "2", "--cap-steps", "none", "--r-seeds", "2", "--bootstrap-B", "20",
                "--out", str(d / "out")]

    def test_real_run_writes_results(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with mock.patch("builtins.print"):
                self.assertEqual(run.main(self._write(d)), 0)
            res = json.loads((d / "out" / "results.json").read_text())
            self.assertFalse(res["synthetic"])
            self.assertEqual(res["label"], run.REAL_LABEL)
            logs = [json.loads(x) for x in (d / "out" / "logs.jsonl").read_text().splitlines()]
            self.assertTrue(logs and all("truth" not in g and not g["synthetic"] for g in logs))
            full = {g["episode_id"]: g["decision"] for g in logs if g["policy"] == "FULL"}
            self.assertEqual(full["PA02-visible-L1"], "present")
            self.assertEqual(full["PA02-occluded-L1"], "present")
            self.assertEqual(full["PA02-removed-L1"], "removed")   # clear misses at P1 and P5 (k = 2)

    def test_incomplete_evidence_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("sys.stderr"):
                self.assertEqual(run.main(self._write(Path(d), drop_one=True)), 2)

    def test_synthetic_evidence_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("sys.stderr"):
                self.assertEqual(run.main(self._write(Path(d), synthetic=True)), 2)

    def test_neither_mode_exits_2(self) -> None:
        with mock.patch("sys.stderr"):
            self.assertEqual(run.main(["--manifest", "x.json"]), 2)


def _weights_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    hf = Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--dinov2-small"
    return (torch.cuda.is_available() and hf.exists() and (REPO_ROOT / "checkpoints" / ev.DETECTOR_WEIGHTS).exists()
            and importlib.util.find_spec("ultralytics") is not None
            and (REPO_ROOT / "data/raw/2026-10-12_scene/now.jpg").exists())


@unittest.skipUnless(_weights_available(), "needs CUDA, DINOv2-small, YOLO-World weights and now.jpg")
class TestIntegration(unittest.TestCase):
    def test_now_jpg_against_itself(self) -> None:
        img = ev.read_bgr("data/raw/2026-10-12_scene/now.jpg")
        box = (1212, 875, 1305, 968)   # study mug, hand box (data/raw/2026-10-12_pilot-prep/evidence/)
        rec, diag = ev.evidence_for(img, box, img.copy(), 2, ev.DinoV2Patches("cuda"), ev.YoloWorld("cuda"))
        self.assertTrue(diag["align_ok"])          # capture/tags.py registers the frame onto itself
        self.assertGreater(rec.match_score, 0.99)
        self.assertEqual(rec.obstruction, 0.0)


if __name__ == "__main__":
    unittest.main()
