"""Tests for capture/pose_check.py (numpy + cv2; GB10 via uv, Pi via system python3).

Run from the repo root (GB10):  uv run --no-sync python -m unittest discover -s capture/tests -v
Run on the Pi:                  cd ~/cmpe249 && python3 -m unittest discover -s capture/tests -v

The Pi has no ref_final.jpg, so the tests synthesise their own 2304x1296 golden (a smooth
multi-scale texture written with cv2.imwrite at quality 95) and a temporary capture_lock.json
derived from configs/capture_lock.json (golden sha256 and recomputed mask sha256 filled in). If
that file is not deployed next to capture/ (Pi layout ~/cmpe249/configs/), an embedded copy of
the sections the tools read is used instead (EMBEDDED_LOCK_CONFIG, identical P3 mask specs).

Shifted images are an exact Fourier shift of a padded canvas, then cropped (non-circular), as in
test_ecc.shifted; cv2.warpAffine is used only for the roll cases (rotation of the padded canvas
before the crop), as test_ecc.py does for its rotation test.

The helpers in this module (canvas, shifted_frame, rotated_frame, write_jpeg, make_lock_config)
are also used by test_qc_session.py.
"""
from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
CAPTURE = HERE.parent
sys.path.insert(0, str(CAPTURE))
import ecc  # noqa: E402
import pose_check  # noqa: E402

W, H = 2304, 1296
PAD = 64  # canvas margin cropped away after a shift/rotation (largest test shift is 12 px)
REAL_LOCK_CONFIG = CAPTURE.parent / "configs" / "capture_lock.json"
REPO_SHELF = CAPTURE.parent / "data" / "raw" / "2026-10-05_shelf-final"

# The sections of configs/capture_lock.json (lock_version 1) that pose_check / qc_session read;
# used only when the real file is not deployed (the Pi's ~/cmpe249 has capture/ only).
EMBEDDED_LOCK_CONFIG = {
    "schema": "cmpe249.capture_lock.v1",
    "lock_version": 1,
    "lock": {"sensor_mode": "2304:1296:10:P", "size": [2304, 1296], "scaler_crop": [0, 0, 4608, 2592],
             "shutter_us": 25000, "gain": 1.381, "awb_gains": [1.5753, 2.5111], "lens_position": 1.10,
             "tolerances": {"shutter_rel": 0.005, "gain_rel": 0.02, "awb_abs": 0.001, "lens_abs": 0.01}},
    "reference": {"position": "P3", "path": "data/raw/2026-10-05_shelf-final/pi/ref_final.jpg",
                  "sha256": "8dbbfe46eb66b5669a647d70e16d19e9dcfcc22c46a207f5ed7062f2ca1d1f97",
                  "size": [2304, 1296], "lux": 203.08969116210938},
    "masks": {
        "static_background": {"by_position": {
            "P3": {"base": "all", "exclude": [
                {"name": "left_outside_unit", "rect": [0, 0, 100, 1296]},
                {"name": "top_compartment_props", "rect": [270, 0, 1800, 362]},
                {"name": "glare_margin", "rect": [836, 362, 1010, 413]},
                {"name": "mug_compartment_interior", "polygon": [[112, 381], [2044, 396], [2017, 809], [132, 804]]},
                {"name": "bottom_compartment_props", "rect": [150, 1030, 1990, 1296]}]},
            "P1": None, "P2": None, "P4": None, "P5": None}},
        "mug_region": {"by_position": {
            "P3": {"base": "none", "include": [
                {"name": "mug_compartment_opening", "polygon": [[124, 393], [2032, 408], [2010, 840], [140, 840]]}]},
            "P1": None, "P2": None, "P4": None, "P5": None}},
        "glare": {"by_position": {
            "P3": {"base": "none", "include": [{"name": "upper_board_glare", "rect": [844, 370, 1003, 406]}]},
            "P1": None, "P2": None, "P4": None, "P5": None}},
        "sha256_P3": {"static_background": "4865260e400997a0d50a1cb4d91b3fa67c7cf0cb8543582de67ecf73dea3462e",
                      "mug_region": "656b7b96f2ddd70beddcc96faac9a446adc4502812e48428f276066be99c62bb",
                      "glare": "38dce5f20850f659765e4c6daf1a511dedfca63e3696f9fec74979697c868fc6"}},
    "registration": {"mode": "edges", "sigma": 1.5, "levels": 3, "iters": 200, "eps": 1e-6},
    "pose_check": {"flag_px": 2.0, "refuse_px": 10.0, "flag_roll_deg": 0.5, "refuse_roll_deg": 2.0, "min_cc": 0.80},
    "qc": {"lux_ratio_guard": 1.25, "shift_flag_px": 2.0, "saturation_level": 250, "saturation_flag_share": 0.001},
}


# --------------------------------------------------------------------------- synthetic data
@functools.lru_cache(maxsize=4)
def canvas(seed: int = 0) -> np.ndarray:
    """Smooth multi-scale random texture on the padded (H + 2 PAD) x (W + 2 PAD) canvas, float64,
    roughly 30..225 (same recipe as test_ecc.canvas). Cached: treat as read-only."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((H + 2 * PAD, W + 2 * PAD), np.float32)
    for s, amp in ((2.0, 1.0), (5.0, 1.5), (12.0, 2.0)):
        n = rng.standard_normal(acc.shape).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), s)
        acc += amp * n / (n.std() + 1e-6)
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    out = 30.0 + 195.0 * acc.astype(np.float64)
    out.flags.writeable = False
    return out


def _crop_u8(big: np.ndarray) -> np.ndarray:
    return np.round(np.clip(big[PAD:-PAD, PAD:-PAD], 0, 255)).astype(np.uint8)


def shifted_frame(seed: int, dx: float, dy: float) -> np.ndarray:
    """W x H uint8 frame with the canvas content moved by (dx, dy) px: exact Fourier shift of the
    padded canvas, then the interior crop (non-circular at the frame borders)."""
    big = canvas(seed)
    if dx == 0 and dy == 0:
        return _crop_u8(big)
    ky = np.fft.fftfreq(big.shape[0])[:, None]
    kx = np.fft.fftfreq(big.shape[1])[None, :]
    out = np.fft.ifft2(np.fft.fft2(big) * np.exp(-2j * np.pi * (kx * dx + ky * dy))).real
    return _crop_u8(out)


def rotated_frame(seed: int, rot_deg: float) -> np.ndarray:
    """W x H uint8 frame rotated by rot_deg about the frame centre (+ = clockwise on screen), made
    with cv2.warpAffine on the padded canvas before the crop."""
    big = canvas(seed)
    c = ((big.shape[1] - 1) / 2.0, (big.shape[0] - 1) / 2.0)
    m = cv2.getRotationMatrix2D(c, -rot_deg, 1.0)  # OpenCV: + angle = counter-clockwise on screen
    return _crop_u8(cv2.warpAffine(big, m, (big.shape[1], big.shape[0]), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REFLECT))


def to_bgr(gray: np.ndarray) -> np.ndarray:
    """A colour version of a grey frame (slightly different channel gains, never >= 250)."""
    g = gray.astype(np.float32)
    return np.clip(np.stack([g * 0.92, g, g * 1.04], axis=2), 0, 245).astype(np.uint8)


def write_jpeg(path: Path, img: np.ndarray, quality: int = 95) -> Path:
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"cv2.imwrite failed for {path}")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_lock_config() -> dict:
    if REAL_LOCK_CONFIG.is_file():
        return json.loads(REAL_LOCK_CONFIG.read_text())
    return copy.deepcopy(EMBEDDED_LOCK_CONFIG)


def make_lock_config(path: Path, golden: Path, mutate=None) -> Path:
    """Write a test capture_lock.json derived from the real one: reference.sha256 = the synthetic
    golden's, mask sha256 values recomputed with ecc. `mutate(cfg)` may edit it before writing."""
    cfg = base_lock_config()
    cfg["reference"]["sha256"] = sha256_file(golden)
    cfg["reference"]["path"] = str(golden)
    for name in ("static_background", "mug_region", "glare"):
        spec = cfg["masks"][name]["by_position"]["P3"]
        cfg["masks"]["sha256_P3"][name] = ecc.mask_sha256(ecc.regions_mask(spec, W, H))
    if mutate:
        mutate(cfg)
    Path(path).write_text(json.dumps(cfg, indent=2))
    return Path(path)


def run_main(main, argv):
    """Call a tool's main(argv) in-process; returns (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as e:  # argparse
            code = e.code
    return code, out.getvalue(), err.getvalue()


# ------------------------------------------------------------------------------- tests
class PoseBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.golden = write_jpeg(cls.dir / "golden.jpg", shifted_frame(0, 0, 0))
        cls.cfg = make_lock_config(cls.dir / "capture_lock.json", cls.golden)
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def image(self, arr: np.ndarray) -> Path:
        type(self).n += 1
        return write_jpeg(self.dir / f"img_{self.n:02d}.jpg", arr)

    def check(self, image: Path, *extra, cfg: Path | None = None, golden: Path | None = None):
        argv = ["--image", str(image), "--lock-config", str(cfg or self.cfg), "--golden", str(golden or self.golden),
                *extra]
        code, out, err = run_main(pose_check.main, argv)
        rep = json.loads(out) if out.strip() else None
        return code, rep, err


class TestPoseVerdicts(PoseBase):
    def test_zero_shift_ok(self):
        code, rep, err = self.check(self.image(shifted_frame(0, 0, 0)))
        self.assertEqual(code, 0, err)
        self.assertEqual(rep["schema"], "cmpe249.pose_check.v1")
        self.assertEqual(rep["verdict"], "ok")
        self.assertEqual(rep["reasons"], [])
        self.assertEqual(rep["exit_code"], 0)
        self.assertLess(rep["translation"]["shift_px"], 0.01)
        self.assertLess(abs(rep["euclidean"]["rot_deg"]), 0.01)
        self.assertEqual(rep["judged"]["shift_source"], "translation")
        self.assertEqual(rep["mask"]["usable_px"], 1010820)
        self.assertEqual(rep["params"]["mode"], "edges")
        self.assertEqual(rep["thresholds"]["source"]["flag_px"], "config")
        self.assertEqual(rep["image"]["size"], [W, H])

    def test_three_px_flag_and_out_file(self):
        out = self.dir / "pose_3px.json"
        code, rep, err = self.check(self.image(shifted_frame(0, 1.8, -2.4)), "--out", str(out))
        self.assertEqual(code, 10, err)
        self.assertEqual(rep["verdict"], "flag")
        self.assertAlmostEqual(rep["translation"]["dx"], 1.8, delta=0.02)
        self.assertAlmostEqual(rep["translation"]["dy"], -2.4, delta=0.02)
        self.assertAlmostEqual(rep["translation"]["shift_px"], 3.0, delta=0.02)
        self.assertEqual(rep["reasons"], ["shift 3.00 px > flag 2.0 px"])
        self.assertEqual(json.loads(out.read_text()), rep)

    def test_twelve_px_refuse(self):
        code, rep, err = self.check(self.image(shifted_frame(0, 12.0, 0.0)))
        self.assertEqual(code, 11, err)
        self.assertEqual(rep["verdict"], "refuse")
        self.assertAlmostEqual(rep["translation"]["shift_px"], 12.0, delta=0.05)
        self.assertIn("shift 12.00 px > refuse 10.0 px", rep["reasons"])
        self.assertIn("shift 12.00 px > flag 2.0 px", rep["reasons"])

    def test_one_degree_roll_flag(self):
        code, rep, err = self.check(self.image(rotated_frame(0, 1.0)))
        self.assertEqual(code, 10, err)
        self.assertEqual(rep["verdict"], "flag")
        self.assertAlmostEqual(rep["euclidean"]["rot_deg"], 1.0, delta=0.02)
        self.assertIn("roll 1.00 deg > flag 0.5 deg", rep["reasons"])
        # the roll is about the frame centre, so the judged (euclidean centre) shift is ~0
        self.assertEqual(rep["judged"]["shift_source"], "euclidean")
        self.assertLess(rep["judged"]["shift_px"], 0.1)

    def test_two_and_a_half_degree_roll_refuse(self):
        code, rep, err = self.check(self.image(rotated_frame(0, -2.5)))
        self.assertEqual(code, 11, err)
        self.assertEqual(rep["verdict"], "refuse")
        self.assertAlmostEqual(rep["euclidean"]["rot_deg"], -2.5, delta=0.02)
        self.assertIn("roll 2.50 deg > refuse 2.0 deg", rep["reasons"])

    def test_flat_image_error(self):
        code, rep, err = self.check(self.image(np.full((H, W), 128, np.uint8)))
        self.assertEqual(code, 12, err)
        self.assertEqual(rep["verdict"], "error")
        self.assertEqual(rep["exit_code"], 12)
        self.assertTrue(rep["reasons"])

    def test_cli_override_recorded(self):
        code, rep, err = self.check(self.image(shifted_frame(0, 1.8, -2.4)), "--flag-px", "4", "--mode", "edges")
        self.assertEqual(code, 0, err)
        self.assertEqual(rep["verdict"], "ok")
        self.assertEqual(rep["thresholds"]["flag_px"], 4.0)
        self.assertEqual(rep["thresholds"]["source"]["flag_px"], "cli")
        self.assertEqual(rep["thresholds"]["source"]["refuse_px"], "config")
        self.assertEqual(rep["params"]["source"]["mode"], "cli")


class TestPoseBadInput(PoseBase):
    def test_wrong_golden_sha(self):
        cfg = make_lock_config(self.dir / "bad_sha.json", self.golden,
                               lambda c: c["reference"].__setitem__("sha256", "0" * 64))
        code, rep, err = self.check(self.golden, cfg=cfg)
        self.assertEqual(code, 3)
        self.assertIsNone(rep)
        self.assertIn("sha256", err)

    def test_wrong_image_size(self):
        small = self.image(np.ascontiguousarray(shifted_frame(0, 0, 0)[:720, :1280]))
        code, rep, err = self.check(small)
        self.assertEqual(code, 3)
        self.assertIn("1280x720", err)

    def test_out_exists(self):
        out = self.dir / "exists.json"
        out.write_text("keep me\n")
        code, rep, err = self.check(self.golden, "--out", str(out))
        self.assertEqual(code, 3)
        self.assertIn("already exists", err)
        self.assertEqual(out.read_text(), "keep me\n")

    def test_missing_image_bad_schema_and_mask_sha(self):
        code, _, err = self.check(self.dir / "nope.jpg")
        self.assertEqual(code, 3)
        cfg = make_lock_config(self.dir / "bad_schema.json", self.golden, lambda c: c.__setitem__("schema", "x"))
        self.assertEqual(self.check(self.golden, cfg=cfg)[0], 3)
        cfg = make_lock_config(self.dir / "bad_mask.json", self.golden,
                               lambda c: c["masks"]["sha256_P3"].__setitem__("static_background", "f" * 64))
        code, _, err = self.check(self.golden, cfg=cfg)
        self.assertEqual(code, 3)
        self.assertIn("mask sha256", err)

    def test_cli_exit_code_via_subprocess(self):
        """The script's real process exit code (not only main()'s return value)."""
        p = subprocess.run([sys.executable, str(CAPTURE / "pose_check.py"), "--image", str(self.dir / "nope.jpg"),
                            "--lock-config", str(self.cfg), "--golden", str(self.golden)],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 3, p.stderr)
        p = subprocess.run([sys.executable, str(CAPTURE / "pose_check.py")], capture_output=True, text=True,
                           timeout=120)
        self.assertEqual(p.returncode, 2)


@unittest.skipUnless((REPO_SHELF / "pi" / "ref_final.jpg").is_file() and REAL_LOCK_CONFIG.is_file(),
                     "GB10 only: needs the repo's ref_final.jpg and configs/capture_lock.json")
class TestPoseRealFrames(unittest.TestCase):
    def check(self, image: Path):
        code, out, err = run_main(pose_check.main, ["--image", str(image), "--lock-config", str(REAL_LOCK_CONFIG),
                                                    "--golden", str(REPO_SHELF / "pi" / "ref_final.jpg")])
        return code, json.loads(out), err

    def test_ref_final_against_itself_ok(self):
        code, rep, err = self.check(REPO_SHELF / "pi" / "ref_final.jpg")
        self.assertEqual(code, 0, err)
        self.assertLess(rep["translation"]["shift_px"], 0.01)

    def test_frame_e_flag(self):
        """Frame e (LensPosition 1.45, before the rigid shift): 2.808 px, cc 0.9459 (MODULE1 section 8)."""
        code, rep, err = self.check(REPO_SHELF / "geometry" / "rpi_view_2026-10-05e.jpg")
        self.assertEqual(code, 10, err)
        self.assertAlmostEqual(rep["translation"]["shift_px"], 2.808, delta=0.01)
        self.assertEqual(rep["reasons"], ["shift 2.81 px > flag 2.0 px"])

    def test_ref_lp145_ok_on_static_mask(self):
        """ref_lp1.45 (other LensPosition, same pose): focus breathing moves the periphery, but on
        the static mask the translation is 0.217 px (cc 0.9075, MODULE1 section 8) -> ok."""
        code, rep, err = self.check(REPO_SHELF / "pi" / "ref_lp1.45.jpg")
        self.assertEqual(code, 0, err)
        self.assertLess(rep["translation"]["shift_px"], 0.5)
        self.assertLess(rep["translation"]["cc"], 0.95)


if __name__ == "__main__":
    unittest.main()
