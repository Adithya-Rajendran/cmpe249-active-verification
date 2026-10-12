"""Tests for capture/qc_session.py (numpy + cv2; GB10 via uv, Pi via system python3).

Run from the repo root (GB10):  uv run --no-sync python -m unittest discover -s capture/tests -v
Run on the Pi:                  cd ~/cmpe249 && python3 -m unittest discover -s capture/tests -v

Synthetic sessions only, built in temporary directories through the REAL pi_capture.py with the
fake rpicam-still. Decodable 2304x1296 JPEGs (written by cv2.imwrite) reach pi_capture through
FAKE_RPICAM_IMAGE (MODULE1 section 1.3). Until builder 0's fake_rpicam_still.py supports that
variable, a small wrapper (written to the temp dir) runs the stock fake and then copies the
FAKE_RPICAM_IMAGE bytes to -o; once the fake supports it, the copy is a no-op repeat.

The sessions are calibrated with pi_capture's existing path (`calibrate --lens-position 1.1`,
fake AE values), and the test lock config's `lock` section holds those fake values, so the tests
do not depend on pi_capture 0.3.0's --lock-config. The static mask, mug region and glare box are
the real P3 ones; P1, P2, P4 and P5 are null, which exercises the fallbacks.

Two sessions are built once per class (each shot costs a fake-camera subprocess):
  clean: complete L = 1 arrangement A01 + run/sitting.json with a real pose_check record -> pass
  dirty: one query shifted 3 px at P3 (static mask) and one 3 px at P4 (auto_residual), one
         saturated block in the mug region, one query at 1.5x Lux, one query with
         FAKE_RPICAM_SHUTTER=20000, a retake of the P1 reference, and one slot never shot.
Mutations that need no image analysis run on copies with --no-images.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pi_capture as pc  # noqa: E402
import pose_check  # noqa: E402
import qc_session  # noqa: E402
from test_pose_check import (make_lock_config, run_main, shifted_frame, to_bgr,  # noqa: E402
                             write_jpeg)

FAKE = HERE / "fake_rpicam_still.py"
# what fake_rpicam_still.py reports for `calibrate --lens-position 1.1` (its AE defaults)
FAKE_LOCK = {"shutter_us": 49995, "gain": 2.6948, "awb_gains": [1.2469, 3.4177], "lens_position": 1.1}
ARR = "A01"
WRAPPER = '''#!{python}
import os, shutil, subprocess, sys
r = subprocess.run([sys.executable, {fake!r}, *sys.argv[1:]])
src, args = os.environ.get("FAKE_RPICAM_IMAGE"), sys.argv[1:]
if (r.returncode == 0 and src and "-o" in args and os.environ.get("FAKE_RPICAM_NO_IMAGE") != "1"
        and os.environ.get("FAKE_RPICAM_TRUNCATE") != "1" and os.environ.get("FAKE_RPICAM_SIZE") is None):
    shutil.copyfile(src, args[args.index("-o") + 1])
sys.exit(r.returncode)
'''


def clear_fakes():
    for k in [k for k in os.environ if k.startswith("FAKE_RPICAM_")]:
        del os.environ[k]


class QcBase(unittest.TestCase):
    """Builds the clean and dirty sessions once for all QC tests."""
    built = False

    @classmethod
    def setUpClass(cls):
        if QcBase.built:
            return
        QcBase.tmp = tempfile.TemporaryDirectory()
        d = QcBase.dir = Path(QcBase.tmp.name)
        QcBase.root = d / "data"
        QcBase.n_out = 0
        wrapper = d / "rpicam_wrapper.py"
        wrapper.write_text(WRAPPER.format(python=sys.executable, fake=str(FAKE)))
        os.chmod(wrapper, 0o755)
        os.chmod(FAKE, 0o755)
        QcBase.rpicam = str(wrapper)
        img = d / "img"
        img.mkdir()
        base = shifted_frame(0, 0, 0)
        QcBase.golden = write_jpeg(img / "golden.jpg", base)  # grey, like ref_final read as grey
        bgr = to_bgr(base)
        frames = {"base": bgr, "shift3": to_bgr(shifted_frame(0, 3.0, 0.0)),
                  "shift3b": to_bgr(shifted_frame(0, -1.8, 2.4))}
        sat = bgr.copy()
        sat[520:700, 500:900] = 255  # inside the P3 mug_region, outside the glare box and the static mask
        frames["saturated"] = sat
        QcBase.frames = {k: write_jpeg(img / f"{k}.jpg", v) for k, v in frames.items()}

        def set_lock(cfg):
            cfg["lock"].update(FAKE_LOCK)
        QcBase.cfg = make_lock_config(d / "capture_lock.json", QcBase.golden, set_lock)

        clear_fakes()
        try:
            QcBase.clean = cls._build("CLEAN", {})
            cls._write_sitting(QcBase.clean)
            dirty = {
                ("P3", "query", "occluded"): {"image": "shift3"},
                ("P4", "query", "removed"): {"image": "shift3b"},
                ("P3", "query", "visible"): {"image": "saturated"},
                ("P2", "query", "visible"): {"FAKE_RPICAM_LUX": "75.0"},
                ("P5", "query", "occluded"): {"FAKE_RPICAM_SHUTTER": "20000"},
                ("P1", "query", "removed"): {"skip": True},
                ("P1", "reference", "visible"): {"retake": True},
            }
            QcBase.dirty = cls._build("DIRTY", dirty)
        finally:
            clear_fakes()
        QcBase.built = True

    @classmethod
    def _pc(cls, *argv):
        code, out, err = run_main(pc.main, [argv[0], "--root", str(cls.root), "--rpicam", cls.rpicam, *argv[1:]])
        if code != 0:
            raise RuntimeError(f"pi_capture {argv} failed ({code}): {err}")
        return out

    @classmethod
    def _build(cls, session: str, special: dict) -> Path:
        os.environ["FAKE_RPICAM_IMAGE"] = str(cls.frames["base"])
        cls._pc("calibrate", "--session", session, "--lighting", "L1", "--lens-position", "1.1")
        slots = [(p, "reference", "visible") for p in pc.POSITIONS] + \
                [(p, "query", c) for c in ("visible", "occluded", "removed") for p in pc.POSITIONS]
        for pos, role, cond in slots:
            sp = special.get((pos, role, cond), {})
            if sp.get("skip"):
                continue
            clear_fakes()
            os.environ["FAKE_RPICAM_IMAGE"] = str(cls.frames[sp.get("image", "base")])
            for k, v in sp.items():
                if k.startswith("FAKE_RPICAM_"):
                    os.environ[k] = v
            args = ["shoot", "--session", session, "--arrangement", ARR, "--position", pos, "--role", role,
                    "--condition", cond, "--lighting", "L1"]
            cls._pc(*args)
            if sp.get("retake"):
                cls._pc(*args, "--take", "2", "--note", "retake: test")
        clear_fakes()
        return cls.root / session

    @classmethod
    def _write_sitting(cls, sdir: Path):
        run = sdir / "run"
        run.mkdir()
        pose_out = run / "pose_check_L1-v1.json"
        code, _, err = run_main(pose_check.main, ["--image", str(sdir / "calibration" / "L1-v1.jpg"),
                                                  "--lock-config", str(cls.cfg), "--golden", str(cls.golden),
                                                  "--out", str(pose_out)])
        if code != 0:
            raise RuntimeError(f"pose_check on the synthetic calibration still: exit {code} {err}")
        sitting = {"schema": "cmpe249.sitting.v1", "tool": {"name": "run_arrangement.py", "version": "1.0.0"},
                   "session": sdir.name, "ntp": {"stdout": "yes", "returncode": 0, "ok": True},
                   "lightings": ["L1"], "locks": {"L1": "L1-v1"},
                   "pose": {"file": "run/pose_check_L1-v1.json", "verdict": "ok", "exit_code": 0,
                            "accepted_flag_note": None}, "note": None}
        with open(run / "sitting.json", "x") as f:
            json.dump(sitting, f, indent=2)

    # ---------------------------------------------------------------- helpers
    def qc(self, sdir: Path, *extra):
        QcBase.n_out += 1
        out_dir = self.dir / f"qc_out_{QcBase.n_out:02d}"
        code, out, err = run_main(qc_session.main, ["--session-dir", str(sdir), "--lock-config", str(self.cfg),
                                                    "--golden", str(self.golden), "--out-dir", str(out_dir), *extra])
        reps = sorted(out_dir.glob("qc_*.json")) if out_dir.exists() else []
        rep = json.loads(reps[0].read_text()) if reps else None
        md = reps[0].with_suffix(".md").read_text() if reps else None
        return code, rep, md, err

    def copy(self, sdir: Path, name: str) -> Path:
        dst = self.dir / "copies" / name
        shutil.copytree(sdir, dst)
        return dst

    @staticmethod
    def codes(rep, stem=None, severity=None):
        return {f["code"] for f in rep["flags"]
                if (stem is None or f["stem"] == stem) and (severity is None or f["severity"] == severity)}

    @staticmethod
    def full_run(sdir: Path, status: str = "complete") -> dict:
        """A run/<ARR>.json as run_arrangement writes it, recording every manifest row of ARR (one slot
        per (role, condition, position, lighting), highest take selected)."""
        slots = {}
        for line in (sdir / "manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["arrangement"] != ARR:
                continue
            k = (r["role"], r["condition"], r["position"], r["lighting"])
            sl = slots.setdefault(k, {"index": len(slots), "phase": "reference" if r["role"] == "reference"
                                      else r["condition"], "role": r["role"], "condition": r["condition"],
                                      "position": r["position"], "lighting": r["lighting"], "takes": [],
                                      "selected_take": None})
            sl["takes"].append({"take": r["take"], "stem": r["stem"], "path": r["path"], "sha256": r["sha256"]})
            sl["selected_take"] = max(sl["selected_take"] or 0, r["take"])
        return {"schema": "cmpe249.run_arrangement.v1", "arrangement": ARR, "status": status,
                "slots": list(slots.values())}

    @staticmethod
    def stem(pos, role, cond, take=1):
        return f"{ARR}_{pos}_{role}_{cond}_L1_t{take:02d}"

    @staticmethod
    def shot(rep, stem):
        return next(s for s in rep["shots"] if s["stem"] == stem)


class TestQcClean(QcBase):
    def test_complete_session_passes(self):
        code, rep, md, err = self.qc(self.clean)
        self.assertEqual(code, 0, (err, [f for f in rep["flags"] if f["severity"] != "info"] if rep else None))
        self.assertEqual(rep["schema"], "cmpe249.qc_session.v1")
        self.assertEqual(rep["verdict"], "pass")
        self.assertEqual(rep["completeness"][ARR], {"expected": 20, "present": 20, "missing": []})
        self.assertEqual(self.codes(rep, severity="fail") | self.codes(rep, severity="flag"), set())
        self.assertTrue(rep["sitting"]["present"])
        self.assertEqual(rep["sitting"]["pose_verdict"], "ok")
        self.assertLess(rep["sitting"]["pose_shift_px"], 0.01)
        queries = [s for s in rep["shots"] if s["role"] == "query"]
        self.assertEqual(len(queries), 15)
        for s in queries:
            self.assertTrue(s["sha256_ok"] and s["lock_ok"] and s["lock_config_ok"], s["stem"])
            self.assertLess(s["registration"]["shift_px"], 0.05, s["stem"])
            self.assertEqual(s["registration"]["mask_source"],
                             "static_background" if s["position"] == "P3" else "auto_residual")
            self.assertEqual(s["saturation_region"], "mug_region" if s["position"] == "P3" else "full_frame_fallback")
            self.assertEqual(s["saturation_share"], 0.0)
            self.assertEqual(s["lux_ratio_reference"], 1.0)
        # P3 references are compared with the golden (info only)
        ref_p3 = self.shot(rep, self.stem("P3", "reference", "visible"))
        self.assertLess(ref_p3["golden_registration"]["shift_px"], 0.05)
        self.assertIn("ref_vs_golden", self.codes(rep, ref_p3["stem"], "info"))
        # inventory: lock files and qc/ excluded; images, sidecars and run/ included
        paths = {f["path"] for f in rep["files"]}
        self.assertNotIn(".session.lock", paths)
        self.assertIn("run/sitting.json", paths)
        self.assertIn(f"{ARR}/query/P3/{self.stem('P3', 'query', 'occluded')}.jpg", paths)
        self.assertTrue(md.startswith("# QC CLEAN: **PASS**"))
        self.assertIn("| A01 | query | occluded | L1 | 5 |", md)
        self.assertEqual({c["condition"] for c in rep["by_condition_all"] if c["role"] == "query"},
                         {"visible", "occluded", "removed"})

    def test_no_run_directory(self):
        s = self.copy(self.clean, "NORUN")
        shutil.rmtree(s / "run")
        code, rep, _, _ = self.qc(s, "--no-images")
        self.assertEqual(code, 10)
        self.assertEqual(self.codes(rep, severity="flag"), {"no_sequencer_log"})
        self.assertFalse(rep["sitting"]["present"])

    def test_aborted_run_and_unaccepted_take(self):
        """Integration finding: an arrangement left 'aborted' with a saved-but-never-accepted take must not
        pass silently (run_arrangement's selected take is the operator-accepted one)."""
        s = self.copy(self.clean, "ABORTED")
        run = self.full_run(s, status="aborted")
        last = next(sl for sl in run["slots"] if (sl["position"], sl["condition"]) == ("P5", "removed"))
        last["selected_take"] = None
        (s / "run" / f"{ARR}.json").write_text(json.dumps(run))
        code, rep, _, _ = self.qc(s, "--no-images")
        self.assertEqual(code, 10)
        self.assertEqual(self.codes(rep, severity="flag"), {"run_incomplete", "unaccepted_take"})
        f = next(f for f in rep["flags"] if f["code"] == "unaccepted_take")
        self.assertEqual((f["position"], f["condition"]), ("P5", "removed"))
        run["status"] = "complete"
        last["selected_take"] = 1
        (s / "run" / f"{ARR}.json").write_text(json.dumps(run))
        code, rep, _, _ = self.qc(s, "--no-images")
        self.assertEqual(code, 0, rep["flags"])

    def test_extra_and_out_of_plan_shots(self):
        """Review finding QC-EXTRA-SHOTS: two bare pi_capture shots after a complete sequencer run, (a) an
        episode-labelled query and (b) take 7 of an existing slot, plus a stray file, used to pass QC."""
        s = self.copy(self.clean, "EXTRA")
        (s / "run" / f"{ARR}.json").write_text(json.dumps(self.full_run(s)))
        code, rep, _, _ = self.qc(s, "--no-images")
        self.assertEqual(code, 0, rep["flags"])  # the complete run file itself is clean
        os.environ["FAKE_RPICAM_IMAGE"] = str(self.frames["shift3"])
        try:
            for extra in (["--position", "P3", "--role", "query", "--condition", "visible", "--episode", "E1",
                           "--take", "1"],
                          ["--position", "P2", "--role", "query", "--condition", "removed", "--take", "7",
                           "--note", "bare retake"]):
                c, _, err = run_main(pc.main, ["shoot", "--root", str(s.parent), "--rpicam", self.rpicam,
                                               "--session", s.name, "--arrangement", ARR, "--lighting", "L1",
                                               *extra])
                self.assertEqual(c, 0, err)
        finally:
            clear_fakes()
        (s / ARR / "query" / "P3" / "copy_of_something.jpg").write_bytes(b"x")
        (s / ARR / "query" / "P1" / f"{ARR}_P1_query_visible_L1_t02.failed-20261005T120000Z.log").write_text("x")
        ep = f"{ARR}_P3_query_visible_L1_E1_t01"
        t7 = self.stem("P2", "query", "removed", 7)
        code, rep, md, _ = self.qc(s)
        self.assertEqual(code, 11)
        self.assertIn("unexpected_shot", self.codes(rep, ep, "fail"))
        self.assertEqual(self.codes(rep, t7, "flag"), {"unrecorded_take"})
        e = self.shot(rep, ep)
        self.assertFalse(e["selected"])
        self.assertTrue(e["unexpected"])
        self.assertIsNone(e["registration"])  # never registered, never in the tables
        self.assertFalse(self.shot(rep, t7)["selected"])  # the run file selects take 1
        g = next(x for x in rep["by_condition"] if (x["role"], x["condition"]) == ("query", "visible"))
        self.assertEqual(g["n"], 5)
        self.assertNotIn("unexpected_shot", g["flags"])
        self.assertEqual(rep["counts"]["unexpected"], 1)
        stray = [f for f in rep["flags"] if f["code"] == "stray_file"]
        self.assertEqual([f["detail"].split()[0] for f in stray], [f"{ARR}/query/P3/copy_of_something.jpg"])
        self.assertEqual(stray[0]["severity"], "flag")
        self.assertIn("failed_capture_log", self.codes(rep, severity="info"))
        # camera logs and sidecars of real rows are not stray
        self.assertFalse(any(ep in f["detail"] for f in stray))
        # the same shots in a session without a sequencer run file: still unexpected, no unrecorded_take
        (s / "run" / f"{ARR}.json").unlink()
        code, rep, _, _ = self.qc(s, "--no-images")
        self.assertIn("unexpected_shot", self.codes(rep, ep, "fail"))
        self.assertNotIn("unrecorded_take", self.codes(rep))
        self.assertTrue(self.shot(rep, t7)["selected"])  # highest take wins without a run file

    def test_deleted_and_byte_edited_images(self):
        s = self.copy(self.clean, "BROKEN")
        gone = self.stem("P2", "query", "visible")
        edited = self.stem("P4", "reference", "visible")
        (s / ARR / "query" / "P2" / f"{gone}.jpg").unlink()
        p = s / ARR / "reference" / "P4" / f"{edited}.jpg"
        b = bytearray(p.read_bytes())
        b[len(b) // 2] ^= 0xFF
        p.write_bytes(bytes(b))
        code, rep, _, _ = self.qc(s, "--no-images")
        self.assertEqual(code, 11)
        self.assertEqual(rep["verdict"], "fail")
        self.assertIn("missing_image", self.codes(rep, gone, "fail"))
        self.assertIn("sha_mismatch", self.codes(rep, edited, "fail"))
        self.assertFalse(self.shot(rep, edited)["sha256_ok"])

    def test_sitting_ntp_and_pose_records(self):
        s = self.copy(self.clean, "SITTING")
        sit = json.loads((s / "run" / "sitting.json").read_text())
        sit["ntp"]["ok"] = False
        sit["pose"]["verdict"] = "flag"
        sit["pose"]["accepted_flag_note"] = "test note"
        (s / "run" / "sitting.json").write_text(json.dumps(sit))
        code, rep, _, _ = self.qc(s, "--no-images")
        self.assertEqual(code, 11)
        self.assertIn("ntp_not_synced", self.codes(rep, severity="fail"))
        self.assertIn("pose_flag", self.codes(rep, severity="flag"))
        (s / "run" / "pose_check_L1-v1.json").unlink()
        code, rep, _, _ = self.qc(s, "--no-images")
        self.assertIn("pose_missing", self.codes(rep, severity="fail"))

    def test_bad_input_and_no_overwrite(self):
        empty = self.dir / "copies" / "EMPTY"
        empty.mkdir(parents=True)
        code, rep, _, err = self.qc(empty)
        self.assertEqual(code, 3)
        self.assertIsNone(rep)
        bad_cfg = self.dir / "bad_cfg.json"
        bad_cfg.write_text(json.dumps({"schema": "nope"}))
        code, _, err = run_main(qc_session.main, ["--session-dir", str(self.clean), "--lock-config", str(bad_cfg),
                                                  "--out-dir", str(self.dir / "never")])
        self.assertEqual(code, 3)
        self.assertFalse((self.dir / "never").exists())
        # the clean session itself is never modified by QC (default out-dir is <session>/qc/, not used here)
        self.assertFalse((self.clean / "qc").exists())


class TestQcDirty(QcBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        code, rep, md, err = cls.qc(cls, cls.dirty)
        cls.dcode, cls.drep, cls.dmd, cls.derr = code, rep, md, err

    def test_verdict_fail(self):
        self.assertEqual(self.dcode, 11, self.derr)
        self.assertEqual(self.drep["verdict"], "fail")

    def test_shift_flags(self):
        s3 = self.stem("P3", "query", "occluded")
        self.assertIn("shift", self.codes(self.drep, s3, "flag"))
        r = self.shot(self.drep, s3)["registration"]
        self.assertAlmostEqual(r["dx"], 3.0, delta=0.05)
        self.assertEqual(r["mask_source"], "static_background")
        s4 = self.stem("P4", "query", "removed")
        self.assertIn("shift", self.codes(self.drep, s4, "flag"))
        r = self.shot(self.drep, s4)["registration"]
        self.assertAlmostEqual(r["shift_px"], 3.0, delta=0.05)
        self.assertEqual(r["mask_source"], "auto_residual")
        # unshifted queries are not flagged for shift
        self.assertNotIn("shift", self.codes(self.drep, self.stem("P2", "query", "occluded")))

    def test_saturation_flag(self):
        st = self.stem("P3", "query", "visible")
        self.assertIn("saturation", self.codes(self.drep, st, "flag"))
        s = self.shot(self.drep, st)
        self.assertEqual(s["saturation_region"], "mug_region")
        self.assertGreater(s["saturation_share"], 0.05)
        self.assertEqual(s["glare_share"], 0.0)
        # the saturated block lies outside the static mask, so the registration is unaffected
        self.assertLess(s["registration"]["shift_px"], 0.05)

    def test_lux_flags(self):
        st = self.stem("P2", "query", "visible")
        self.assertTrue({"lux_vs_calibration", "lux_vs_reference"} <= self.codes(self.drep, st, "flag"))
        s = self.shot(self.drep, st)
        self.assertEqual(s["lux_ratio_reference"], 1.5)
        self.assertEqual(s["lux_ratio_calibration"], 1.5)

    def test_lock_flags(self):
        st = self.stem("P5", "query", "occluded")
        self.assertTrue({"lock_check_failed", "lock_config_mismatch"} <= self.codes(self.drep, st, "fail"))
        detail = next(f["detail"] for f in self.drep["flags"] if f["stem"] == st and f["code"] == "lock_config_mismatch")
        self.assertIn("ExposureTime 20000 vs 49995", detail)

    def test_missing_slot_and_retake(self):
        self.assertEqual(self.drep["completeness"][ARR]["missing"], ["P1 query removed L1"])
        self.assertIn("missing_slot", self.codes(self.drep, severity="fail"))
        self.assertIn("retake", self.codes(self.drep, severity="info"))
        t1 = self.shot(self.drep, self.stem("P1", "reference", "visible", 1))
        t2 = self.shot(self.drep, self.stem("P1", "reference", "visible", 2))
        self.assertFalse(t1["selected"])
        self.assertTrue(t2["selected"])
        self.assertEqual(self.shot(self.drep, self.stem("P1", "query", "visible"))["registration"]["against"],
                         t2["stem"])

    def test_by_condition_tables(self):
        g = next(x for x in self.drep["by_condition"] if x["role"] == "query" and x["condition"] == "occluded")
        self.assertEqual(g["n"], 5)
        self.assertAlmostEqual(g["shift_px"]["max"], 3.0, delta=0.05)
        self.assertEqual(g["flags"].get("shift"), 1)
        self.assertIn("**FAIL**", self.dmd)

    def test_no_images_skips_registration_and_saturation(self):
        code, rep, _, _ = self.qc(self.dirty, "--no-images")
        self.assertEqual(code, 11)
        self.assertFalse(rep["params"]["images"])
        codes = self.codes(rep)
        self.assertFalse({"shift", "saturation", "registration_failed", "ref_vs_golden"} & codes)
        self.assertTrue({"lux_vs_calibration", "lux_vs_reference", "lock_config_mismatch", "missing_slot"} <= codes)
        self.assertTrue(all(s["registration"] is None and s["saturation_share"] is None for s in rep["shots"]))

    def test_run_manifest_selects_take_and_overrides_recorded(self):
        s = self.copy(self.dirty, "RUNSEL")
        (s / "run").mkdir()
        run = {"schema": "cmpe249.run_arrangement.v1", "slots": [
            {"index": 0, "phase": "reference", "role": "reference", "condition": "visible", "position": "P1",
             "lighting": "L1", "takes": [], "selected_take": 1}]}
        (s / "run" / f"{ARR}.json").write_text(json.dumps(run))
        code, rep, _, _ = self.qc(s, "--no-images", "--lux-guard", "2.0")
        self.assertTrue(self.shot(rep, self.stem("P1", "reference", "visible", 1))["selected"])
        self.assertFalse(self.shot(rep, self.stem("P1", "reference", "visible", 2))["selected"])
        self.assertEqual(rep["params"]["sources"]["lux_guard"], "cli")
        self.assertEqual(rep["params"]["lux_guard"], 2.0)
        self.assertNotIn("lux_vs_reference", self.codes(rep))  # 1.5x is inside a 2.0 guard


def tearDownModule():
    if QcBase.built:
        QcBase.tmp.cleanup()
        QcBase.built = False


if __name__ == "__main__":
    unittest.main()
