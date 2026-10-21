"""Unit tests for capture/pi_capture.py using a fake rpicam-still (stdlib only).

Run from the repo root:  python3 -m unittest discover -s capture/tests -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pi_capture as pc  # noqa: E402

FAKE = str(HERE / "fake_rpicam_still.py")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.chmod(FAKE, 0o755)
        self.clear_fakes()

    def tearDown(self):
        self.clear_fakes()
        self.tmp.cleanup()

    @staticmethod
    def clear_fakes():
        for k in [k for k in os.environ if k.startswith("FAKE_RPICAM_")]:
            del os.environ[k]

    def run_tool(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = pc.main([*argv[:1], "--root", self.root, "--rpicam", FAKE, *argv[1:]])
            except SystemExit as e:  # argparse errors
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def calibrate(self, lighting="L1", *extra):
        return self.run_tool("calibrate", "--session", "S1", "--lighting", lighting, *extra)

    def shoot(self, pos="P1", role="reference", cond="visible", arr="A01", lighting="L1", *extra):
        return self.run_tool("shoot", "--session", "S1", "--arrangement", arr, "--position", pos,
                             "--role", role, "--condition", cond, "--lighting", lighting, *extra)

    def sdir(self):
        return Path(self.root) / "S1"


class TestCalibrate(Base):
    def test_calibrate_writes_lock(self):
        code, out, err = self.calibrate()
        self.assertEqual(code, 0, err)
        lock = json.loads((self.sdir() / "lock.json").read_text())
        e = lock["locks"]["L1"]
        self.assertEqual(e["lock_id"], "L1-v1")
        self.assertEqual(e["controls"]["shutter_us"], 49995)
        self.assertEqual(e["controls"]["awb_gains"], [1.2469, 3.4177])
        self.assertTrue((self.sdir() / "calibration" / "L1-v1.jpg").exists())
        self.assertTrue((self.sdir() / "calibration" / "L1-v1.json").exists())

    def test_second_calibration_refused_without_reason(self):
        self.calibrate()
        code, _, err = self.calibrate()
        self.assertEqual(code, pc.EXIT_PROTOCOL)
        self.assertIn("already has lock", err)
        code, _, err = self.calibrate("L1", "--recalibrate")
        self.assertEqual(code, pc.EXIT_USAGE)
        code, _, err = self.calibrate("L1", "--recalibrate", "--reason", "lamp bulb replaced")
        self.assertEqual(code, 0, err)
        lock = json.loads((self.sdir() / "lock.json").read_text())
        self.assertEqual(lock["locks"]["L1"]["lock_id"], "L1-v2")
        self.assertEqual(lock["locks"]["L1"]["supersedes"], "L1-v1")
        self.assertEqual(len(lock["history"]), 2)

    def test_manual_lens_and_copy_from(self):
        code, _, err = self.calibrate("L1", "--lens-position", "2.5")
        self.assertEqual(code, 0, err)
        code, _, err = self.calibrate("L2", "--copy-from", "L1")
        self.assertEqual(code, 0, err)
        lock = json.loads((self.sdir() / "lock.json").read_text())
        self.assertEqual(lock["locks"]["L1"]["controls"]["lens_position"], 2.5)
        self.assertEqual(lock["locks"]["L2"]["copied_from"], "L1-v1")

    def test_camera_failure_leaves_no_lock(self):
        os.environ["FAKE_RPICAM_FAIL"] = "1"
        code, _, err = self.calibrate()
        self.assertEqual(code, pc.EXIT_CAMERA)
        self.assertFalse((self.sdir() / "lock.json").exists())
        os.environ.pop("FAKE_RPICAM_FAIL")
        code, _, err = self.calibrate()
        self.assertEqual(code, 0, err)

    def test_bad_session_and_lighting(self):
        code, _, err = self.run_tool("calibrate", "--session", "../evil", "--lighting", "L1")
        self.assertEqual(code, pc.EXIT_USAGE)
        code, _, err = self.calibrate("L_1")
        self.assertEqual(code, pc.EXIT_USAGE)

    def test_copy_from_takes_its_own_lux_reference(self):
        os.environ["FAKE_RPICAM_LUX"] = "50.0"
        self.assertEqual(self.calibrate("L1", "--lens-position", "2.5", "--rotation", "180")[0], 0)
        os.environ["FAKE_RPICAM_LUX"] = "20.0"  # the second lighting is dimmer
        code, out, err = self.calibrate("L2", "--copy-from", "L1")
        self.assertEqual(code, 0, err)
        lock = json.loads((self.sdir() / "lock.json").read_text())
        l1, l2 = lock["locks"]["L1"], lock["locks"]["L2"]
        self.assertEqual(l2["controls"], l1["controls"])
        self.assertEqual(l2["capture"], l1["capture"])
        self.assertEqual(l2["calibration"]["lux"], 20.0)
        self.assertTrue(l2["calibration"]["lock_check"]["ok"])
        self.assertTrue((self.sdir() / "calibration" / "L2-v1.jpg").exists())
        cmd = l2["calibration"]["command"]
        self.assertEqual(cmd[cmd.index("--shutter") + 1], str(l1["controls"]["shutter_us"]))
        self.assertNotIn("--autofocus-mode", cmd)
        # an L2 shot under L2 light is not flagged against L1's Lux
        code, out, err = self.shoot(lighting="L2")
        self.assertEqual(code, 0, err)
        self.assertNotIn("ambient light differs", out)

    def test_copy_from_refuses_options_it_would_ignore(self):
        self.calibrate("L1")
        for extra in (["--lens-position", "2"], ["--rotation", "180"], ["--size", "4608x2592"]):
            code, _, err = self.calibrate("L2", "--copy-from", "L1", *extra)
            self.assertEqual(code, pc.EXIT_USAGE, extra)
            self.assertIn("cannot be combined", err)
        self.assertNotIn("L2", json.loads((self.sdir() / "lock.json").read_text())["locks"])

    def test_dry_run_writes_nothing(self):
        code, out, _ = self.calibrate("L1", "--dry-run")
        self.assertEqual(code, 0)
        self.assertFalse((self.sdir() / "lock.json").exists())
        self.calibrate("L1")
        before = (self.sdir() / "lock.json").read_bytes()
        code, out, _ = self.calibrate("L2", "--copy-from", "L1", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("--shutter", out)
        self.assertEqual((self.sdir() / "lock.json").read_bytes(), before)
        self.assertFalse((self.sdir() / "calibration" / "L2-v1.jpg").exists())

    def test_calibration_without_metadata_leaves_no_lock(self):
        os.environ["FAKE_RPICAM_NO_META"] = "1"
        code, _, err = self.calibrate()
        self.assertEqual(code, pc.EXIT_CAMERA, err)
        self.assertFalse((self.sdir() / "lock.json").exists())
        self.assertFalse((self.sdir() / "calibration" / "L1-v1.jpg").exists())
        os.environ.pop("FAKE_RPICAM_NO_META")
        self.assertEqual(self.calibrate()[0], 0)

    def test_calibration_without_lux_warns(self):
        os.environ["FAKE_RPICAM_NO_LUX"] = "1"
        code, out, err = self.calibrate()
        self.assertEqual(code, 0, err)
        self.assertIn("ambient-light guard is off", out)


class TestShoot(Base):
    def setUp(self):
        super().setUp()
        self.calibrate()

    def test_shoot_writes_image_sidecar_manifest(self):
        code, out, err = self.shoot()
        self.assertEqual(code, 0, err)
        d = self.sdir() / "A01" / "reference" / "P1"
        side = json.loads((d / "A01_P1_reference_visible_L1_t01.json").read_text())
        self.assertEqual(side["labels"]["position"], "P1")
        self.assertEqual(side["lock"]["lock_id"], "L1-v1")
        self.assertTrue(side["lock_check"]["ok"])
        self.assertEqual(len(side["image"]["sha256"]), 64)
        self.assertIn("rpicam_metadata", side)
        self.assertEqual(side["tool"]["version"], pc.TOOL_VERSION)
        cmd = side["capture"]["command"]
        self.assertIn("--shutter", cmd)
        self.assertIn("--lens-position", cmd)
        self.assertEqual(len(pc.read_jsonl(self.sdir() / "manifest.jsonl")), 1)
        self.assertEqual([p.name for p in d.iterdir() if p.name.startswith(".")], [])

    def test_overwrite_refused_and_retake_allowed(self):
        self.shoot()
        img = self.sdir() / "A01/reference/P1/A01_P1_reference_visible_L1_t01.jpg"
        before = img.read_bytes()
        code, _, err = self.shoot()
        self.assertEqual(code, pc.EXIT_OVERWRITE)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual(img.read_bytes(), before)
        code, _, err = self.shoot("P1", "reference", "visible", "A01", "L1", "--take", "2", "--note", "blur")
        self.assertEqual(code, 0, err)

    def test_bad_labels_rejected(self):
        self.assertEqual(self.shoot(pos="P6")[0], 2)
        self.assertEqual(self.shoot(role="ref")[0], 2)
        self.assertEqual(self.shoot(cond="hidden")[0], 2)
        self.assertEqual(self.shoot(arr="a01")[0], pc.EXIT_USAGE)
        self.assertEqual(self.shoot(arr="A01_x")[0], pc.EXIT_USAGE)
        code, _, err = self.shoot(lighting="L9")
        self.assertEqual(code, pc.EXIT_NOLOCK)
        self.assertEqual(self.shoot("P1", "reference", "visible", "A01", "L1", "--episode", "E_1")[0], pc.EXIT_USAGE)
        self.assertFalse((self.sdir() / "manifest.jsonl").exists())

    def test_reference_must_show_unchanged_scene(self):
        code, _, err = self.shoot(cond="occluded")
        self.assertEqual(code, pc.EXIT_PROTOCOL)

    def test_query_needs_references(self):
        code, _, err = self.shoot(role="query", cond="visible")
        self.assertEqual(code, pc.EXIT_PROTOCOL)
        self.shoot("P1")
        code, _, err = self.shoot("P1", "query", "visible")
        self.assertEqual(code, pc.EXIT_PROTOCOL)  # P2..P5 references missing
        self.assertIn("all five references", err)
        code, out, err = self.shoot("P1", "query", "pilot")
        self.assertEqual(code, 0, err)  # allowed for pilot, with a warning
        self.assertIn("WARNING", out)
        # once a query exists the scene counts as changed: more references are late
        self.assertEqual(self.shoot("P2")[0], pc.EXIT_PROTOCOL)
        for p in pc.POSITIONS:
            self.assertEqual(self.shoot(p, arr="A02")[0], 0)
        code, out, err = self.shoot("P3", "query", "occluded", "A02")
        self.assertEqual(code, 0, err)
        self.assertNotIn("WARNING", out)

    def test_late_reference_refused(self):
        for p in pc.POSITIONS:
            self.shoot(p)
        self.shoot("P2", "query", "removed")
        code, _, err = self.shoot("P2", "reference", "visible", "A01", "L1", "--take", "2")
        self.assertEqual(code, pc.EXIT_PROTOCOL)
        code, _, err = self.shoot("P2", "reference", "visible", "A01", "L1", "--take", "2",
                                  "--allow-late-reference")
        self.assertEqual(code, pc.EXIT_PROTOCOL)  # needs a note
        code, out, err = self.shoot("P2", "reference", "visible", "A01", "L1", "--take", "2",
                                    "--allow-late-reference", "--note", "reference was blurred")
        self.assertEqual(code, 0, err)

    def test_recalibration_mid_arrangement_refused(self):
        for p in pc.POSITIONS:
            self.shoot(p)
        self.calibrate("L1", "--recalibrate", "--reason", "test")
        code, _, err = self.shoot("P1", "query", "visible")
        self.assertEqual(code, pc.EXIT_PROTOCOL)
        self.assertIn("re-calibrated mid-arrangement", err)
        code, _, err = self.shoot("P1", "query", "visible", "A01", "L1", "--accept-lock-change",
                                  "--note", "sun came out")
        self.assertEqual(code, 0, err)
        # a new arrangement under the new lock is fine
        self.assertEqual(self.shoot("P1", arr="A02")[0], 0)

    def test_lighting_guard_warns(self):
        os.environ["FAKE_RPICAM_LUX"] = "10.0"
        code, out, err = self.shoot()
        self.assertEqual(code, 0, err)
        self.assertIn("ambient light differs", out)
        side = json.loads((self.sdir() / "A01/reference/P1/A01_P1_reference_visible_L1_t01.json").read_text())
        self.assertFalse(side["lighting_check"]["ok"])

    def test_camera_failure_no_files(self):
        os.environ["FAKE_RPICAM_FAIL"] = "1"
        code, _, err = self.shoot()
        self.assertEqual(code, pc.EXIT_CAMERA)
        d = self.sdir() / "A01/reference/P1"
        self.assertFalse(any(p.suffix in (".jpg", ".json") for p in d.iterdir()))
        self.assertFalse((self.sdir() / "manifest.jsonl").exists())
        os.environ.pop("FAKE_RPICAM_FAIL")
        self.assertEqual(self.shoot()[0], 0)

    def assert_nothing_saved(self, sub="A01/reference/P1"):
        d = self.sdir() / sub
        if d.exists():
            self.assertEqual([p.name for p in d.iterdir() if p.suffix in (".jpg", ".json")], [])
            self.assertEqual([p.name for p in d.iterdir() if ".partial" in p.name], [])
        self.assertFalse((self.sdir() / "manifest.jsonl").exists())

    def test_missing_or_bad_metadata_is_a_camera_error(self):
        for var in ("FAKE_RPICAM_NO_META", "FAKE_RPICAM_BAD_META"):
            os.environ[var] = "1"
            code, _, err = self.shoot()
            self.assertEqual(code, pc.EXIT_CAMERA, (var, err))
            self.assertIn("metadata", err)
            self.assert_nothing_saved()
            os.environ.pop(var)
        d = self.sdir() / "A01/reference/P1"
        failed = [p for p in d.iterdir() if ".failed-" in p.name]
        self.assertEqual(len(failed), 2)  # camera logs kept
        self.assertIn(f"camera log kept as {failed[0].parent}", err)  # the message names the kept log
        self.assertNotIn(".partial.log", err)
        self.assertEqual(self.shoot()[0], 0)

    def test_missing_or_truncated_image_is_a_camera_error(self):
        for var in ("FAKE_RPICAM_NO_IMAGE", "FAKE_RPICAM_TRUNCATE"):
            os.environ[var] = "1"
            code, _, err = self.shoot()
            self.assertEqual(code, pc.EXIT_CAMERA, (var, err))
            self.assert_nothing_saved()
            os.environ.pop(var)

    def test_stale_partial_file_is_never_used(self):
        d = self.sdir() / "A01/reference/P1"
        d.mkdir(parents=True)
        stale = d / ".A01_P1_reference_visible_L1_t01.partial.jpg"
        stale.write_bytes(b"\xff\xd8\xff\xd9")  # left over from an interrupted run
        os.environ["FAKE_RPICAM_NO_IMAGE"] = "1"
        code, _, err = self.shoot()
        self.assertEqual(code, pc.EXIT_CAMERA, err)
        self.assert_nothing_saved()

    def test_controls_fov_and_size_actually_applied_are_checked(self):
        self.assertEqual(self.shoot("P1")[0], 0)
        side = json.loads((self.sdir() / "A01/reference/P1/A01_P1_reference_visible_L1_t01.json").read_text())
        self.assertTrue(side["lock_check"]["scaler_crop"]["ok"])
        self.assertEqual([side["image"]["width"], side["image"]["height"]], [2304, 1296])
        os.environ["FAKE_RPICAM_SHUTTER"] = "30000"
        os.environ["FAKE_RPICAM_CROP"] = "(768, 432)/3072x1728"
        os.environ["FAKE_RPICAM_SIZE"] = "1536x864"
        code, out, err = self.shoot("P2")
        self.assertEqual(code, 0, err)  # kept, but flagged
        self.assertIn("did not apply locked controls", out)
        side = json.loads((self.sdir() / "A01/reference/P2/A01_P2_reference_visible_L1_t01.json").read_text())
        lc = side["lock_check"]
        self.assertFalse(lc["ok"])
        self.assertFalse(lc["shutter_us"]["ok"])
        self.assertFalse(lc["scaler_crop"]["ok"])
        self.assertEqual(lc["image_size"]["reported"], [1536, 864])
        self.assertEqual([side["image"]["width"], side["image"]["height"]], [1536, 864])
        m = pc.read_jsonl(self.sdir() / "manifest.jsonl")
        self.assertEqual([r["lock_ok"] for r in m], [True, False])

    def test_missing_lux_is_reported_not_skipped(self):
        os.environ["FAKE_RPICAM_NO_LUX"] = "1"
        code, out, err = self.shoot()
        self.assertEqual(code, 0, err)
        self.assertIn("ambient light could not be checked", out)
        m = pc.read_jsonl(self.sdir() / "manifest.jsonl")
        self.assertIsNone(m[0]["light_ok"])

    def test_zero_lux_is_flagged(self):
        os.environ["FAKE_RPICAM_LUX"] = "0.0"
        code, out, err = self.shoot()
        self.assertEqual(code, 0, err)
        self.assertIn("ambient light differs", out)

    def test_status(self):
        self.shoot()
        code, out, err = self.run_tool("status", "--session", "S1")
        self.assertEqual(code, 0, err)
        self.assertIn("A01: 1 shot(s)", out)
        self.assertIn("P1:1", out)


if __name__ == "__main__":
    unittest.main()
