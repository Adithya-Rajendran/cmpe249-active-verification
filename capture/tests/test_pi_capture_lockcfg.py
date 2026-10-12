"""Tests for pi_capture.py 0.3.0 `calibrate --lock-config` (MODULE1 section 1) and the fake camera's
FAKE_RPICAM_IMAGE (section 1.3). Stdlib only; the fake rpicam-still is used, never the camera.

Run from the repo root:  python3 -m unittest discover -s capture/tests -v
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pi_capture as pc  # noqa: E402

FAKE = str(HERE / "fake_rpicam_still.py")
REAL_CONFIG = HERE.parent.parent / "configs" / "capture_lock.json"

CONFIG = {
    "schema": "cmpe249.capture_lock.v1", "lock_version": 1,
    "lock": {"sensor_mode": "2304:1296:10:P", "size": [2304, 1296], "scaler_crop": [0, 0, 4608, 2592],
             "shutter_us": 25000, "gain": 1.381, "awb_gains": [1.5753, 2.5111], "lens_position": 1.10,
             "frame_duration_limits_us": [60000, 60000],
             "tolerances": {"shutter_rel": 0.005, "gain_rel": 0.02, "awb_abs": 0.001, "lens_abs": 0.01}},
}
WANT_CONTROLS = {"shutter_us": 25000, "gain": 1.381, "awb_gains": [1.5753, 2.5111], "lens_position": 1.1,
                 "lens_source": "lock-config"}


class LockConfigBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.chmod(FAKE, 0o755)
        self.clear_fakes()
        self.cfg = self.write_cfg(CONFIG)

    def tearDown(self):
        self.clear_fakes()
        self.tmp.cleanup()

    @staticmethod
    def clear_fakes():
        for k in [k for k in os.environ if k.startswith("FAKE_RPICAM_")]:
            del os.environ[k]

    def write_cfg(self, cfg, name="capture_lock.json", raw=None):
        p = Path(self.root) / name
        p.write_text(raw if raw is not None else json.dumps(cfg, indent=2))
        return str(p)

    def run_tool(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = pc.main([*argv[:1], "--root", self.root, "--rpicam", FAKE, *argv[1:]])
            except SystemExit as e:  # argparse errors
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def calibrate(self, *extra, lighting="L1", cfg=None):
        return self.run_tool("calibrate", "--session", "S1", "--lighting", lighting,
                             "--lock-config", cfg or self.cfg, *extra)

    def sdir(self):
        return Path(self.root) / "S1"

    def lock(self):
        return json.loads((self.sdir() / "lock.json").read_text())


class TestLockConfigCalibrate(LockConfigBase):
    def test_lock_has_exactly_the_file_controls(self):
        code, out, err = self.calibrate()
        self.assertEqual(code, 0, err)
        e = self.lock()["locks"]["L1"]
        self.assertEqual(e["controls"], WANT_CONTROLS)
        self.assertEqual(e["capture"]["width"], 2304)
        self.assertEqual(e["capture"]["height"], 1296)
        self.assertEqual(e["capture"]["rotation"], 0)
        self.assertEqual(e["capture"]["mode"], "2304:1296:10:P")
        self.assertIsNone(e["copied_from"])
        self.assertEqual(e["calibration"]["exposure"], "manual controls from lock config")
        lc = e["lock_config"]
        self.assertEqual(lc["path"], str(Path(self.cfg).resolve()))
        self.assertEqual(lc["sha256"], hashlib.sha256(Path(self.cfg).read_bytes()).hexdigest())
        self.assertEqual(lc["schema"], "cmpe249.capture_lock.v1")
        self.assertEqual(lc["lock_version"], 1)
        self.assertEqual(e["tool_version"], "0.3.0")
        cmd = e["calibration"]["command"]
        for flag, val in (("--shutter", "25000"), ("--gain", "1.381"), ("--awbgains", "1.5753,2.5111"),
                          ("--lens-position", "1.1"), ("--mode", "2304:1296:10:P"), ("--rotation", "0")):
            self.assertEqual(cmd[cmd.index(flag) + 1], val, flag)
        self.assertNotIn("--autofocus-mode", cmd)
        self.assertNotIn("NOTE: autofocus", out)
        side = json.loads((self.sdir() / "calibration" / "L1-v1.json").read_text())
        self.assertEqual(side["lock"]["controls"], WANT_CONTROLS)

    def test_lock_check_ok_with_scaler_crop(self):
        code, out, err = self.calibrate()
        self.assertEqual(code, 0, err)
        lc = self.lock()["locks"]["L1"]["calibration"]["lock_check"]
        self.assertTrue(lc["ok"], lc)
        for k in ("shutter_us", "gain", "awb_gains", "lens_position", "scaler_crop", "image_size"):
            self.assertTrue(lc[k]["ok"], k)
        self.assertEqual(lc["scaler_crop"]["requested"], [0, 0, 4608, 2592])
        self.assertEqual(lc["scaler_crop"]["reported"], "(0, 0)/4608x2592")
        self.assertNotIn("WARNING: camera did not apply", out)

    def test_shutter_not_applied_warns_but_writes_lock(self):
        os.environ["FAKE_RPICAM_SHUTTER"] = "20000"
        code, out, err = self.calibrate()
        self.assertEqual(code, 0, err)
        lc = self.lock()["locks"]["L1"]["calibration"]["lock_check"]
        self.assertFalse(lc["ok"])
        self.assertFalse(lc["shutter_us"]["ok"])
        self.assertIn("WARNING: camera did not apply the lock-config controls ['shutter_us']", out)
        self.assertEqual(self.lock()["locks"]["L1"]["controls"]["shutter_us"], 25000)

    def test_scaler_crop_mismatch(self):
        os.environ["FAKE_RPICAM_CROP"] = "(768, 432)/3072x1728"
        code, out, err = self.calibrate()
        self.assertEqual(code, 0, err)
        lc = self.lock()["locks"]["L1"]["calibration"]["lock_check"]
        self.assertFalse(lc["scaler_crop"]["ok"])
        self.assertFalse(lc["ok"])
        self.assertIn("['scaler_crop']", out)

    def test_list_form_scaler_crop_is_normalised(self):
        os.environ["FAKE_RPICAM_CROP"] = "[0, 0, 4608, 2592]"
        code, _, err = self.calibrate()
        self.assertEqual(code, 0, err)
        self.assertTrue(self.lock()["locks"]["L1"]["calibration"]["lock_check"]["ok"])

    def test_forbidden_flag_combinations(self):
        for extra in (["--copy-from", "L2"], ["--lens-position", "1.1"], ["--af-range", "macro"],
                      ["--size", "2304x1296"], ["--rotation", "0"]):
            with self.subTest(extra=extra):
                code, _, err = self.calibrate(*extra)
                self.assertEqual(code, pc.EXIT_USAGE, err)
                self.assertIn(extra[0], err)
                self.assertIn("cannot be combined", err)
                self.assertFalse((self.sdir() / "lock.json").exists())

    def test_allowed_options(self):
        code, _, err = self.calibrate("--quality", "90", "--shoot-settle-ms", "800", "--note", "sitting 1")
        self.assertEqual(code, 0, err)
        e = self.lock()["locks"]["L1"]
        self.assertEqual(e["capture"]["quality"], 90)
        self.assertEqual(e["capture"]["settle_ms"], 800)
        self.assertEqual(e["note"], "sitting 1")

    def test_bad_configs(self):
        def mut(fn):
            c = copy.deepcopy(CONFIG)
            fn(c)
            return c
        cases = {
            "schema": mut(lambda c: c.__setitem__("schema", "cmpe249.capture_lock.v0")),
            "no_lock": mut(lambda c: c.pop("lock")),
            "size": mut(lambda c: c["lock"].__setitem__("size", [1536, 864])),
            "shutter_float": mut(lambda c: c["lock"].__setitem__("shutter_us", 25000.0)),
            "shutter_zero": mut(lambda c: c["lock"].__setitem__("shutter_us", 0)),
            "shutter_bool": mut(lambda c: c["lock"].__setitem__("shutter_us", True)),
            "gain_zero": mut(lambda c: c["lock"].__setitem__("gain", 0)),
            "gain_str": mut(lambda c: c["lock"].__setitem__("gain", "1.381")),
            "awb_one": mut(lambda c: c["lock"].__setitem__("awb_gains", [1.5])),
            "awb_neg": mut(lambda c: c["lock"].__setitem__("awb_gains", [1.5, -2.5])),
            "lens_range": mut(lambda c: c["lock"].__setitem__("lens_position", 15.5)),
            "lens_missing": mut(lambda c: c["lock"].pop("lens_position")),
            "crop": mut(lambda c: c["lock"].__setitem__("scaler_crop", [0, 0, 4608])),
        }
        for name, cfg in cases.items():
            with self.subTest(case=name):
                code, _, err = self.calibrate(cfg=self.write_cfg(cfg, f"bad_{name}.json"))
                self.assertEqual(code, pc.EXIT_USAGE, err)
                self.assertIn("--lock-config", err)
                self.assertFalse((self.sdir() / "lock.json").exists())
        code, _, err = self.calibrate(cfg=self.write_cfg(None, "bad_json.json", raw="{not json"))
        self.assertEqual(code, pc.EXIT_USAGE)
        self.assertIn("not valid JSON", err)
        code, _, err = self.calibrate(cfg=str(Path(self.root) / "missing.json"))
        self.assertEqual(code, pc.EXIT_USAGE)
        self.assertIn("cannot read", err)

    def test_dry_run_calibrate_writes_nothing(self):
        code, out, err = self.calibrate("--dry-run")
        self.assertEqual(code, 0, err)
        self.assertIn("--shutter 25000 --gain 1.381 --awbgains 1.5753,2.5111 --lens-position 1.1", out)
        self.assertFalse((self.sdir() / "lock.json").exists())

    def test_shoot_uses_lock_config_controls(self):
        self.assertEqual(self.calibrate()[0], 0)
        code, out, err = self.run_tool("shoot", "--session", "S1", "--arrangement", "A01", "--position", "P3",
                                       "--role", "reference", "--condition", "visible", "--lighting", "L1",
                                       "--dry-run")
        self.assertEqual(code, 0, err)
        self.assertIn("--shutter 25000 --gain 1.381 --awbgains 1.5753,2.5111 --lens-position 1.1", out)
        self.assertIn("--mode 2304:1296:10:P", out)
        code, out, err = self.run_tool("shoot", "--session", "S1", "--arrangement", "A01", "--position", "P3",
                                       "--role", "reference", "--condition", "visible", "--lighting", "L1")
        self.assertEqual(code, 0, err)
        self.assertNotIn("WARNING", out)
        side = json.loads(next((self.sdir() / "A01" / "reference" / "P3").glob("*_t01.json")).read_text())
        self.assertTrue(side["lock_check"]["ok"], side["lock_check"])

    def test_copy_from_on_top_of_lock_config(self):
        self.assertEqual(self.calibrate()[0], 0)
        os.environ["FAKE_RPICAM_LUX"] = "80"
        code, out, err = self.run_tool("calibrate", "--session", "S1", "--lighting", "L2", "--copy-from", "L1")
        self.assertEqual(code, 0, err)
        e = self.lock()["locks"]["L2"]
        self.assertEqual(e["controls"], WANT_CONTROLS)
        self.assertEqual(e["copied_from"], "L1-v1")
        self.assertIsNone(e["lock_config"])
        self.assertTrue(e["calibration"]["lock_check"]["ok"])
        self.assertEqual(e["calibration"]["lux"], 80.0)

    def test_recalibrate_still_needs_reason(self):
        self.assertEqual(self.calibrate()[0], 0)
        code, _, err = self.calibrate()
        self.assertEqual(code, pc.EXIT_PROTOCOL)
        code, _, err = self.calibrate("--recalibrate")
        self.assertEqual(code, pc.EXIT_USAGE)
        code, _, err = self.calibrate("--recalibrate", "--reason", "camera bumped")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.lock()["locks"]["L1"]["lock_id"], "L1-v2")

    @unittest.skipUnless(REAL_CONFIG.exists(), "configs/capture_lock.json not deployed here")
    def test_real_repo_config(self):
        code, _, err = self.calibrate(cfg=str(REAL_CONFIG))
        self.assertEqual(code, 0, err)
        e = self.lock()["locks"]["L1"]
        self.assertEqual(e["controls"], WANT_CONTROLS)
        self.assertTrue(e["calibration"]["lock_check"]["ok"])


class TestFakeImage(unittest.TestCase):
    def test_fake_rpicam_image_copies_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "src.jpg"
            src.write_bytes(b"\xff\xd8 real jpeg bytes \xff\xd9")
            env = {k: v for k, v in os.environ.items() if not k.startswith("FAKE_RPICAM_")}
            env["FAKE_RPICAM_IMAGE"] = str(src)
            out, meta = Path(t) / "o.jpg", Path(t) / "m.json"
            p = subprocess.run([sys.executable, FAKE, "-o", str(out), "--metadata", str(meta),
                                "--shutter", "25000"], env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(out.read_bytes(), src.read_bytes())
            self.assertEqual(json.loads(meta.read_text())["ExposureTime"], 25000)
            env["FAKE_RPICAM_TRUNCATE"] = "1"
            out.unlink()
            subprocess.run([sys.executable, FAKE, "-o", str(out), "--metadata", str(meta)], env=env, check=True,
                           capture_output=True)
            self.assertEqual(out.read_bytes(), src.read_bytes()[:-10])


if __name__ == "__main__":
    unittest.main()
