"""perception/evidence_v03.py: the background-consistency obstruction cue on synthetic shelf images (pure numpy /
cv2), its combination with the frozen v0.2.1 evidence through mocked encoder and detector, and the jobs path."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from perception import evidence as e021
from perception import evidence_v03 as v03
from verify.schema import EvidenceStore

W, H = 300, 200
BOX = (130.0, 80.0, 170.0, 120.0)   # 40 x 40 target; search box (120, 70, 180, 130); context window (90, 40, 210, 160)
JUNCTION = 110                      # back wall above, floor below: the junction runs across the target
BOARD = (200, 215, 225)             # BGR, a plain light board


class MeanColour:
    """Mock PatchEncoder (as in tests/test_evidence.py): each 14 px patch -> unit vector of (mean RGB - 100).
    Plain grey and plain white patches both map to (1, 1, 1)/sqrt(3): it is blind to a white-on-white board,
    like DINOv2 was on the check batch."""
    patch = 14

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        k = self.patch
        f = rgb.reshape(h // k, k, w // k, k, 3).astype(np.float32).mean((1, 3)) - 100.0
        return f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-6)


def no_detections(bgr: np.ndarray) -> list[e021.Detection]:
    return []


def background(seed: int = 0) -> np.ndarray:
    """Shelf background: back wall (110) above the junction, floor (170) below, mild noise for the sharpness ring."""
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.float32)
    img[:JUNCTION] = 110.0
    img[JUNCTION:] = 170.0
    img += rng.normal(0, 4, (H, W, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def with_target(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = (int(v) for v in BOX)
    out[y0:y1, x0:x1] = (30, 30, 150)   # dark red
    out[y0:y1, (x0 + x1) // 2:x1] = (30, 120, 30)
    return out


def with_board(img: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = rect
    out[y0:y1, x0:x1] = BOARD
    return out


def cue(query: np.ndarray, ref: np.ndarray | None = None, valid: np.ndarray | None = None) -> tuple[float, dict]:
    ref = with_target(background()) if ref is None else ref
    valid = np.ones((H, W), bool) if valid is None else valid
    return v03.background_obstruction(ref, query, valid, BOX)


class TestBackgroundCue(unittest.TestCase):
    def test_present_target_is_not_obstructed(self) -> None:
        u, parts = cue(with_target(background(1)))
        self.assertEqual(u, 0.0)
        self.assertEqual(parts["box_changed"], 0.0)

    def test_removed_target_reads_background(self) -> None:
        # the box changed (the target is gone), but what it shows now continues the reference's background
        u, parts = cue(background(1))
        self.assertGreater(parts["box_changed"], 0.9)
        self.assertLess(u, 0.05)

    def test_plain_board_in_front_is_obstruction(self) -> None:
        u, parts = cue(with_board(with_target(background(1)), (95, 50, 205, 150)))
        self.assertGreater(u, 0.95)
        self.assertGreater(parts["ring_changed"], 0.5)

    def test_plain_board_that_v021_misses(self) -> None:
        # the case the cue was built for: no detection, and novelty explains the white board by the grey wall
        ref = with_target(background())
        q = with_board(with_target(background(1)), (95, 50, 205, 150))
        rec021, _ = e021.evidence_for(ref, BOX, q, 2, MeanColour(), no_detections, align=e021.identity_align)
        rec, diag = v03.evidence_for(ref, BOX, q, 2, MeanColour(), no_detections, align=e021.identity_align)
        self.assertLess(rec021.obstruction, 0.1)
        self.assertGreater(rec.obstruction, 0.95)
        self.assertEqual(diag["obstruction_v021"], rec021.obstruction)

    def test_partial_board_reads_its_share(self) -> None:
        u, _ = cue(with_board(with_target(background(1)), (150, 50, 240, 150)))   # right half of the box
        self.assertGreater(u, 0.35)
        self.assertLess(u, 0.65)

    def test_board_beside_a_removed_target(self) -> None:
        # the occluder stands beside the empty spot (a diagnostic removed view): the ring changes, the box does not
        # show the board, so the cue stays low. A ring-change cue would flag this view.
        u, parts = cue(with_board(background(1), (185, 50, 240, 150)))
        self.assertLess(u, 0.05)
        self.assertGreater(parts["ring_changed"], 0.2)

    def test_lighting_gain_is_not_obstruction(self) -> None:
        brighter = np.clip(background(1).astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        u, parts = cue(brighter)
        self.assertLess(u, 0.05)
        self.assertAlmostEqual(float(np.mean(parts["gain"])), 1 / 1.3, delta=0.03)

    def test_robust_gain_is_not_pulled_by_a_large_board(self) -> None:
        ref = with_target(background())
        q = with_board(with_target(background(1)), (95, 50, 205, 150))
        import cv2
        gr = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), 2.0)
        gq = cv2.GaussianBlur(q.astype(np.float32), (0, 0), 2.0)
        g = v03.robust_gain(gr, gq, np.ones((H, W), bool), BOX)
        self.assertTrue(np.allclose(g, 1.0, atol=0.03), g)
        # v0.2.1's ring-mean gain on the same pair is pulled well away from 1
        win = e021.grow(BOX, 1.0, W, H)
        ring = ~e021.box_mask(e021.grow(BOX, 0.25, W, H), win)
        sl = (slice(win[1], win[3]), slice(win[0], win[2]))
        g021 = gr[sl][ring].mean(axis=0) / gq[sl][ring].mean(axis=0)
        self.assertLess(float(np.mean(g021)), 0.85)

    def test_invalid_pixels_never_count(self) -> None:
        q = with_board(with_target(background(1)), (95, 50, 205, 150))
        valid = np.zeros((H, W), bool)
        self.assertEqual(cue(q, valid=valid)[0], 0.0)

    def test_known_limit_vertical_edge_behind_target(self) -> None:
        # Recorded limit, not a wish: a background that is not continuous along rows (a divider edge right behind
        # the target) is predicted wrongly, and a REMOVED target reads as obstructed. If this ever fails, the
        # limit was fixed and the dev note should say so.
        bg = background()
        bg[:, :150] = 230                    # a white divider face left of the target's middle, shelf to the right
        q = background(1)
        q[:, :150] = 230
        u, _ = cue(q, ref=with_target(bg))
        self.assertGreater(u, 0.2)

    def test_deterministic(self) -> None:
        q = with_board(with_target(background(1)), (150, 50, 240, 150))
        a, pa = cue(q)
        b, pb = cue(q.copy())
        self.assertEqual(a, b)
        self.assertEqual(json.dumps(pa), json.dumps(pb))


class TestRowInterpolate(unittest.TestCase):
    def test_reproduces_a_row_ramp_and_a_junction(self) -> None:
        img = np.zeros((20, 40, 1), np.float32)
        img[:, :, 0] = np.arange(40, dtype=np.float32)[None, :]   # a ramp along rows
        img[10:] += 100.0                                          # and a horizontal junction
        out = v03.row_interpolate(img, (10, 2, 30, 18), edge_px=1)
        np.testing.assert_allclose(out[2:18, 10:30], img[2:18, 10:30], atol=1e-4)
        np.testing.assert_array_equal(out[:2], img[:2])

    def test_hole_at_the_image_edge_uses_the_other_side(self) -> None:
        img = np.zeros((10, 20, 3), np.float32)
        img[:, 15:] = 7.0
        out = v03.row_interpolate(img, (10, 0, 20, 10), edge_px=4)   # cut off on the right
        np.testing.assert_allclose(out[:, 10:], 0.0)
        out = v03.row_interpolate(img, (0, 0, 10, 10), edge_px=4)    # cut off on the left
        np.testing.assert_allclose(out[:, :10], 0.0)

    def test_empty_hole_is_a_copy(self) -> None:
        img = np.ones((5, 5, 3), np.float32)
        np.testing.assert_array_equal(v03.row_interpolate(img, (3, 3, 3, 4)), img)


class TestRecordAndJobs(unittest.TestCase):
    def test_match_and_quality_are_v021s(self) -> None:
        ref = with_target(background())
        for q in (with_target(background(1)), background(1), with_board(with_target(background(1)), (95, 50, 205, 150))):
            r021, d021 = e021.evidence_for(ref, BOX, q, 3, MeanColour(), no_detections, align=e021.identity_align)
            r, d = v03.evidence_for(ref, BOX, q, 3, MeanColour(), no_detections, align=e021.identity_align)
            self.assertEqual((r.stop, r.match_score, r.quality), (r021.stop, r021.match_score, r021.quality))
            self.assertEqual(r.obstruction, max(r021.obstruction, d["v03_parts"]["background_obstruction"]))
            self.assertEqual(d["quality_parts"], d021["quality_parts"])

    def test_jobs_write_a_store_deterministically(self) -> None:
        import cv2
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ref = with_target(background())
            cv2.imwrite(str(t / "ref.png"), ref)
            cv2.imwrite(str(t / "removed.png"), background(1))
            cv2.imwrite(str(t / "board.png"), with_board(with_target(background(1)), (95, 50, 205, 150)))
            ann = {"schema": e021.ANNOTATION_SCHEMA, "reference": str(t / "ref.png"), "stop": 2,
                   "targets": {"brick": {"box": list(BOX)}}}
            (t / "ann.json").write_text(json.dumps(ann))
            jobs = [{"episode_id": "A-removed-L1", "stop": 2, "target": "brick", "annotation": str(t / "ann.json"),
                     "query": str(t / "removed.png")},
                    {"episode_id": "A-occluded-L1", "stop": 2, "target": "brick", "annotation": str(t / "ann.json"),
                     "query": str(t / "board.png")}]
            outs = []
            for i in range(2):
                store, diags = v03.run_jobs(jobs, MeanColour(), no_detections, align=e021.identity_align)
                store.to_jsonl(t / f"ev{i}.jsonl")
                outs.append((t / f"ev{i}.jsonl").read_bytes())
            self.assertEqual(outs[0], outs[1])
            back = EvidenceStore.from_jsonl(t / "ev0.jsonl")
            self.assertLess(back.get("A-removed-L1", 2).obstruction, 0.1)
            self.assertGreater(back.get("A-occluded-L1", 2).obstruction, 0.9)
            self.assertEqual(diags[0]["evidence_version"], v03.EVIDENCE_VERSION)
            self.assertIn("v03_parts", diags[1])

    def test_bad_jobs_schema_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "jobs.json"
            p.write_text(json.dumps({"schema": "nope", "jobs": []}))
            self.assertEqual(v03.main(["--jobs", str(p), "--out", str(Path(t) / "o.jsonl")]), 2)


if __name__ == "__main__":
    unittest.main()
