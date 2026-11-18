"""perception/side_descriptor.py on synthetic shelf images (pure numpy / cv2): the sign of the left/right split, a
removed target reading as no obstruction, the support rule, the in-box agreement with v0.3's background cue, the
refused-registration path and the jobs CLI."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from perception import evidence as e021
from perception import evidence_v03 as v03
from perception import side_descriptor as sd

W, H = 300, 200
BOX = (130.0, 80.0, 170.0, 120.0)   # 40 x 40 target; search box (120, 70, 180, 130); context window (90, 40, 210, 160)
JUNCTION = 110
BOARD = (200, 215, 225)


def background(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.float32)
    img[:JUNCTION] = 110.0
    img[JUNCTION:] = 170.0
    img += rng.normal(0, 4, (H, W, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def with_target(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = (int(v) for v in BOX)
    out[y0:y1, x0:x1] = (30, 30, 150)
    return out


def with_board(img: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = rect
    out[y0:y1, x0:x1] = BOARD
    return out


REF = with_target(background())
ALL = np.ones((H, W), bool)
LEFT_BOARD = (95, 50, 160, 150)     # covers the window's left part and the left three quarters of the box
RIGHT_BOARD = (140, 50, 205, 150)   # its mirror image about the box centre (x = 150)


def desc(query: np.ndarray, valid: np.ndarray = ALL) -> dict:
    return sd.descriptor_registered(REF, query, valid, BOX)


class TestSplit(unittest.TestCase):
    def test_a_board_on_the_left_gives_a_positive_descriptor(self) -> None:
        d = desc(with_board(with_target(background(1)), LEFT_BOARD))
        self.assertIsNone(d["missing"])
        self.assertGreater(d["u_left"], 0.5)
        self.assertLess(d["u_right"], 0.25)
        self.assertGreater(d["d"], 0.4)

    def test_the_mirror_image_gives_the_opposite_sign(self) -> None:
        left = desc(with_board(with_target(background(1)), LEFT_BOARD))
        right = desc(with_board(with_target(background(1)), RIGHT_BOARD))
        self.assertLess(right["d"], -0.4)
        self.assertAlmostEqual(right["u_left"], left["u_right"], delta=0.02)
        self.assertAlmostEqual(right["u_right"], left["u_left"], delta=0.02)

    def test_a_removed_target_reads_as_no_obstruction(self) -> None:
        # the box changed (the target is gone) but continues the reference's background: not obstructed
        d = desc(background(1))
        self.assertEqual((d["u_left"], d["u_right"], d["d"]), (0.0, 0.0, 0.0))

    def test_an_unchanged_scene_reads_zero(self) -> None:
        d = desc(with_target(background(1)))
        self.assertEqual(d["d"], 0.0)

    def test_removed_behind_a_board_keeps_the_board_side(self) -> None:
        d = desc(with_board(background(1), LEFT_BOARD))
        self.assertGreater(d["d"], 0.4)


class TestSupportAndAgreement(unittest.TestCase):
    def test_a_mostly_invalid_part_gives_no_descriptor(self) -> None:
        valid = ALL.copy()
        valid[:, :140] = False            # most of the window's left part (x 90..150) registered from outside
        d = desc(with_board(with_target(background(1)), LEFT_BOARD), valid)
        self.assertEqual(d["missing"], "support")
        self.assertIsNone(d["d"])
        self.assertLess(d["valid_left"], sd.SUPPORT_MIN)

    def test_invalid_pixels_are_left_out_of_the_shares(self) -> None:
        valid = ALL.copy()
        valid[:, 90:100] = False          # a thin invalid strip: still enough support
        d = desc(with_board(with_target(background(1)), LEFT_BOARD), valid)
        self.assertIsNone(d["missing"])
        self.assertGreaterEqual(d["valid_left"], sd.SUPPORT_MIN)

    def test_the_in_box_share_is_v03_background_obstruction(self) -> None:
        for q in (background(2), with_board(background(2), LEFT_BOARD),
                  with_board(with_target(background(3)), RIGHT_BOARD),
                  with_board(with_target(background(4)), (120, 60, 180, 140))):
            for valid in (ALL, np.pad(np.ones((H - 20, W - 30), bool), ((0, 20), (30, 0)))):
                u, _ = v03.background_obstruction(REF, q, valid, BOX)
                self.assertEqual(desc(q, valid)["background_obstruction"], u)


BIG_W, BIG_H = 640, 480                 # large enough for C0's four pyramid levels
BIG_BOX = (300.0, 220.0, 340.0, 260.0)


def textured(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(128, 60, (BIG_H, BIG_W, 3)).astype(np.float32), (0, 0), 3)
    return np.clip(noise, 0, 255).astype(np.uint8)


class TestRegistrationPath(unittest.TestCase):
    def test_a_refused_registration_gives_no_descriptor(self) -> None:
        d = sd.view_descriptor(textured(5), np.zeros((BIG_H // 2, BIG_W, 3), np.uint8), BIG_BOX)
        self.assertEqual(d["missing"], "registration refused")
        self.assertIsNone(d["d"])
        self.assertIsNone(d["H"])

    def test_an_identical_query_registers_and_reads_zero(self) -> None:
        img = textured(5)
        d = sd.view_descriptor(img, img.copy(), BIG_BOX)
        self.assertIsNone(d["missing"])
        self.assertEqual(d["d"], 0.0)
        np.testing.assert_allclose(np.asarray(d["H"]), np.eye(3), atol=1e-3)


class TestCli(unittest.TestCase):
    def test_one_row_per_job_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            img = textured(6)
            cv2.imwrite(str(t / "ref.png"), img)
            cv2.imwrite(str(t / "same.png"), img)
            cv2.imwrite(str(t / "small.png"), img[: BIG_H // 2])
            ann = {"schema": e021.ANNOTATION_SCHEMA, "reference": str(t / "ref.png"), "stop": 2,
                   "targets": {"thing": {"box": list(BIG_BOX)}}}
            (t / "ann.json").write_text(json.dumps(ann))
            jobs = [{"episode_id": f"X0001-{c}-L1", "stop": 2, "target": "thing", "annotation": str(t / "ann.json"),
                     "query": str(t / q)} for c, q in (("occluded", "same.png"), ("removed", "small.png"))]
            (t / "jobs.json").write_text(json.dumps({"schema": e021.JOBS_SCHEMA, "jobs": jobs}))
            self.assertEqual(sd.main(["--jobs", str(t / "jobs.json"), "--out", str(t / "side.jsonl")]), 0)
            rows = [json.loads(line) for line in (t / "side.jsonl").read_text().splitlines()]
            self.assertEqual([(r["episode_id"], r["stop"]) for r in rows], [(j["episode_id"], 2) for j in jobs])
            self.assertTrue(all(r["schema"] == sd.SCHEMA and r["descriptor_version"] == sd.DESCRIPTOR_VERSION
                                for r in rows))
            self.assertEqual((rows[0]["d"], rows[0]["missing"]), (0.0, None))
            self.assertEqual((rows[1]["d"], rows[1]["missing"]), (None, "registration refused"))
            # the same rows from two spawned workers, in the jobs' order
            self.assertEqual(sd.main(["--jobs", str(t / "jobs.json"), "--out", str(t / "w.jsonl"),
                                      "--workers", "2"]), 0)
            self.assertEqual((t / "w.jsonl").read_text(), (t / "side.jsonl").read_text())
            # refusals: another jobs schema, a duplicate (episode, stop), a job whose stop is not its annotation's
            (t / "bad.json").write_text(json.dumps({"schema": "nope", "jobs": jobs}))
            (t / "dup.json").write_text(json.dumps({"schema": e021.JOBS_SCHEMA, "jobs": [jobs[0], jobs[0]]}))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sd.main(["--jobs", str(t / "bad.json"), "--out", str(t / "x.jsonl")]), 2)
                self.assertEqual(sd.main(["--jobs", str(t / "dup.json"), "--out", str(t / "x.jsonl")]), 2)
            with self.assertRaises(ValueError):
                sd.job_descriptor({**jobs[0], "stop": 3})


if __name__ == "__main__":
    unittest.main()
