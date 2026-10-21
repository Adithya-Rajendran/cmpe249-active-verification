"""Tests for capture/run_arrangement.py (stdlib only; MODULE1 section 3.9).

Run from the repo root:  uv run --no-sync python -m unittest discover -s capture/tests -v   (GB10)
                         cd ~/cmpe249 && python3 -m unittest discover -s capture/tests -v   (Pi)

The sequencer runs as a subprocess, as it would on the Pi. It drives the REAL pi_capture.py with
the fake rpicam-still (tests/fake_rpicam_still.py) in a temporary data root; the camera is never
used. Its other two collaborators are replaced by small fakes that this module writes into the
temporary directory:
  fake_timedatectl.py  prints $FAKE_NTP (default "yes") and exits $FAKE_NTP_RC (default 0)
  fake_pose_check.py   writes a minimal cmpe249.pose_check.v1 JSON to --out with
                       verdict $FAKE_POSE_VERDICT (default ok) and exits 0/10/11/12 to match

pi_capture 0.3.0 adds `calibrate --lock-config` (MODULE1 section 1, builder 0). Until that lands,
`sitting` is tested through a thin test shim (written below) that runs the real pi_capture
`calibrate --lens-position <lock lens>` and then records the lock_check the 0.3.0 tool would
record (ok unless FAKE_RPICAM_SHUTTER is set). Every other call goes straight to the real
pi_capture.py. Once pi_capture.py understands --lock-config the shim is not used.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURE = HERE.parent
RUN = CAPTURE / "run_arrangement.py"
PI_CAPTURE = CAPTURE / "pi_capture.py"
FAKE_CAM = HERE / "fake_rpicam_still.py"
REPO = CAPTURE.parent
HAS_LOCK_CONFIG = "--lock-config" in PI_CAPTURE.read_text()
FORBIDDEN = ("--allow-late-reference", "--accept-lock-change", "--episode", "--recalibrate", "--dry-run")

N_PROMPTS_L2_A = 4 + 1 + 20 + 2 * 40   # phase prompts, lamp-set, toggles, shot + saved per slot
N_PROMPTS_L2_B = 4 + 1 + 4 + 2 * 40
N_PROMPTS_L1 = 4 + 1 + 0 + 2 * 20

FAKE_TIMEDATECTL = """\
import os, sys
print(os.environ.get("FAKE_NTP", "yes"))
sys.exit(int(os.environ.get("FAKE_NTP_RC", "0")))
"""

FAKE_POSE_CHECK = """\
import argparse, json, os, sys
p = argparse.ArgumentParser()
for k in ("--image", "--lock-config", "--golden", "--out"):
    p.add_argument(k)
a = p.parse_args()
verdict = os.environ.get("FAKE_POSE_VERDICT", "ok")
code = {"ok": 0, "flag": 10, "refuse": 11, "error": 12}[verdict]
shift = {"ok": 0.2, "flag": 2.81, "refuse": 12.0, "error": None}[verdict]
reasons = {"ok": [], "flag": ["shift 2.81 px > flag 2.0 px"], "refuse": ["shift 12.00 px > refuse 10.0 px"],
           "error": ["cc 0.10 < min_cc 0.8"]}[verdict]
out = {"schema": "cmpe249.pose_check.v1", "tool": {"name": "fake_pose_check.py", "version": "0"},
       "image": {"path": a.image}, "golden": {"path": a.golden},
       "translation": {"dx": 0.0, "dy": shift, "shift_px": shift, "cc": 0.99},
       "euclidean": {"dx": 0.0, "dy": shift, "rot_deg": 0.0, "cc": 0.99},
       "verdict": verdict, "reasons": reasons, "exit_code": code}
with open(a.out, "x") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out))
sys.exit(code)
"""

# Stand-in for pi_capture 0.3.0's `calibrate --lock-config` (see the module docstring).
SHIM = """\
import json, os, subprocess, sys
REAL = {real!r}
args = sys.argv[1:]
if args and args[0] == "calibrate" and "--lock-config" in args:
    i = args.index("--lock-config")
    cfg = json.load(open(args[i + 1]))
    args = args[:i] + ["--lens-position", repr(cfg["lock"]["lens_position"])] + args[i + 2:]
    p = subprocess.run([sys.executable, REAL, *args], capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    if p.returncode == 0:
        opt = lambda k: args[args.index(k) + 1]
        sdir = os.path.join(opt("--root"), opt("--session"))
        lock_path = os.path.join(sdir, "lock.json")
        lock = json.load(open(lock_path))
        e = lock["locks"][opt("--lighting")]
        ok = not os.environ.get("FAKE_RPICAM_SHUTTER")
        e["calibration"]["lock_check"] = {{"shim": {{"ok": ok}}, "ok": ok}}
        e["controls"]["lens_source"] = "lock-config (test shim for pi_capture 0.3.0)"
        lock["history"][-1] = e
        json.dump(lock, open(lock_path, "w"), indent=2)
        side = os.path.join(sdir, "calibration", e["lock_id"] + ".json")
        json.dump({{"schema": "cmpe249.capture.v1", "kind": "calibration", "lock": e}}, open(side, "w"), indent=2)
        if not ok:
            print("WARNING: camera did not apply the lock-config controls ['shutter_us'] (shim)")
    sys.exit(p.returncode)
os.execv(sys.executable, [sys.executable, REAL, *args])
"""

# rpicam wrapper that fails the first still after $FAIL_ONCE_FLAG appears (then removes the flag).
FAIL_ONCE_RPICAM = """#!{python}
import os, sys
flag = os.environ.get("FAIL_ONCE_FLAG")
if "--version" not in sys.argv and flag and os.path.exists(flag):
    os.unlink(flag)
    print("ERROR: *** fake camera failure (fail-once) ***", file=sys.stderr)
    sys.exit(1)
os.execv(sys.executable, [sys.executable, {fake!r}, *sys.argv[1:]])
"""

# rpicam wrapper: run the fake camera (metadata, header checks), then replace -o with a real JPEG.
IMAGE_RPICAM = """#!{python}
import os, shutil, subprocess, sys
p = subprocess.run([sys.executable, {fake!r}, *sys.argv[1:]])
img = os.environ.get("TEST_RPICAM_IMAGE")
if p.returncode == 0 and img and "-o" in sys.argv:
    shutil.copyfile(img, sys.argv[sys.argv.index("-o") + 1])
sys.exit(p.returncode)
"""

# rpicam wrapper: sleep, then the fake camera (an "orphaned" pi_capture still busy after its parent died).
SLOW_RPICAM = """#!{python}
import os, sys, time
if "--version" not in sys.argv:
    time.sleep(float(os.environ.get("SLOW_RPICAM_S", "2.0")))
os.execv(sys.executable, [sys.executable, {fake!r}, *sys.argv[1:]])
"""

# Holds <session>/.session.lock (as a running pi_capture would) once run_log.jsonl shows `event`.
LOCK_HOLDER = """\
import fcntl, os, sys, time
log, lock, event, hold = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
t_end = time.time() + 120
while time.time() < t_end and not (os.path.exists(log) and event in open(log).read()):
    time.sleep(0.01)
with open(lock, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    print("held", flush=True)
    time.sleep(hold)
"""

LOCK_CONFIG = {
    "schema": "cmpe249.capture_lock.v1", "lock_version": 1,
    "lock": {"sensor_mode": "2304:1296:10:P", "size": [2304, 1296], "scaler_crop": [0, 0, 4608, 2592],
             "shutter_us": 25000, "gain": 1.381, "awb_gains": [1.5753, 2.5111], "lens_position": 1.10,
             "tolerances": {"shutter_rel": 0.005, "gain_rel": 0.02, "awb_abs": 0.001, "lens_abs": 0.01}},
    "reference": {"position": "P3", "sha256": "0" * 64, "size": [2304, 1296]},
}


def first6(plan_slots):
    return [(s["phase"], s["position"], s["lighting"]) for s in plan_slots[:6]]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.t = t
        self.root = t / "data"
        self.session = "S1"
        self.sdir = self.root / self.session
        self.rundir = self.sdir / "run"
        (t / "fake_timedatectl.py").write_text(f"#!{sys.executable}\n" + FAKE_TIMEDATECTL)
        os.chmod(t / "fake_timedatectl.py", 0o755)
        (t / "fake_pose_check.py").write_text(FAKE_POSE_CHECK)
        os.chmod(FAKE_CAM, 0o755)
        if HAS_LOCK_CONFIG:
            self.pi_capture = str(PI_CAPTURE)
        else:
            (t / "pi_capture_shim.py").write_text(SHIM.format(real=str(PI_CAPTURE)))
            self.pi_capture = str(t / "pi_capture_shim.py")
        self.lock_config = t / "capture_lock.json"
        self.lock_config.write_text(json.dumps(LOCK_CONFIG, indent=2))
        self.golden = t / "golden.jpg"
        self.golden.write_bytes(b"\xff\xd8golden\xff\xd9")
        self.rpicam = str(FAKE_CAM)
        self.n_script = 0

    def tearDown(self):
        self.tmp.cleanup()

    # -- running the tool
    def env(self, extra=None):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("FAKE_", "CMPE249_")) and k not in ("TIMEDATECTL", "RPICAM_STILL")}
        env.update(extra or {})
        return env

    def run_tool(self, cmd, *args, script=None, env=None, common=True):
        argv = [sys.executable, str(RUN), cmd]
        if common:  # before `args`, so a test's own value for an option wins (argparse keeps the last)
            argv += ["--root", str(self.root), "--session", self.session, "--pi-capture", self.pi_capture,
                     "--pose-check", str(self.t / "fake_pose_check.py"), "--python", sys.executable,
                     "--rpicam", self.rpicam, "--timedatectl", str(self.t / "fake_timedatectl.py"),
                     "--lamp-settle-s", "0"]
        argv += list(args)
        if script is not None:
            self.n_script += 1
            sp = self.t / f"script{self.n_script}.json"
            sp.write_text(json.dumps(script))
            argv += ["--script", str(sp)]
        p = subprocess.run(argv, capture_output=True, text=True, env=self.env(env), timeout=900)
        return p.returncode, p.stdout, p.stderr

    def sitting(self, script=("", ""), *extra, env=None):
        return self.run_tool("sitting", "--lock-config", str(self.lock_config), "--golden", str(self.golden),
                             *extra, script=list(script), env=env)

    def arrangement(self, script, *extra, arr="A01", ordering="A", env=None):
        return self.run_tool("arrangement", "--arrangement", arr, "--ordering", ordering, *extra,
                             script=list(script), env=env)

    def plan(self, *extra):
        code, out, err = self.run_tool("plan", *extra, common=False)
        self.assertEqual(code, 0, err)
        return json.loads(out)

    # -- reading results
    def log(self):
        return [json.loads(x) for x in (self.rundir / "run_log.jsonl").read_text().splitlines() if x.strip()]

    def manifest(self, arr="A01"):
        return json.loads((self.rundir / f"{arr}.json").read_text())

    def pc_status(self):
        p = subprocess.run([sys.executable, str(PI_CAPTURE), "status", "--root", str(self.root),
                            "--session", self.session, "--json"], capture_output=True, text=True,
                           env=self.env())
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout)

    def files(self):
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())

    def assert_no_forbidden(self):
        text = (self.rundir / "run_log.jsonl").read_text()
        for flag in FORBIDDEN:
            self.assertNotIn(flag, text)
        cmds = [e["command"] for e in self.log() if e["event"] == "pi_capture"]
        self.assertTrue(cmds)
        for c in cmds:
            self.assertNotIn("pilot", c)
            for flag in FORBIDDEN:
                self.assertNotIn(flag, c)


class TestPlan(Base):
    def test_ordering_a(self):
        p = self.plan("--arrangement", "A01", "--ordering", "A")
        self.assertEqual(p["n_slots"], 40)
        self.assertEqual(p["lamp_toggles"], 20)
        self.assertEqual(first6(p["slots"]), [
            ("reference", "P3", "L1"), ("reference", "P3", "L2"), ("reference", "P1", "L2"),
            ("reference", "P1", "L1"), ("reference", "P2", "L1"), ("reference", "P2", "L2")])
        # phase 2 starts at P3 with L2, L1 (visit counter runs across phases)
        s = p["slots"][10:12]
        self.assertEqual([(x["phase"], x["position"], x["lighting"]) for x in s],
                         [("visible", "P3", "L2"), ("visible", "P3", "L1")])
        # no toggle at phase boundaries
        for k in (10, 20, 30):
            self.assertEqual(p["slots"][k - 1]["lighting"], p["slots"][k]["lighting"])

    def test_ordering_b(self):
        p = self.plan("--arrangement", "A01", "--ordering", "B")
        self.assertEqual(p["lamp_toggles"], 4)
        self.assertEqual(first6(p["slots"]), [
            ("reference", "P3", "L1"), ("reference", "P1", "L1"), ("reference", "P2", "L1"),
            ("reference", "P4", "L1"), ("reference", "P5", "L1"), ("reference", "P3", "L2")])
        self.assertEqual([x["lighting"] for x in p["slots"][10:12]], ["L2", "L2"])  # phase 2 starts L2

    def test_roles_conditions_and_counts(self):
        p = self.plan("--arrangement", "PILOT01", "--ordering", "A")
        roles = [(s["role"], s["condition"]) for s in p["slots"]]
        self.assertEqual(roles.count(("reference", "visible")), 10)
        self.assertEqual(roles.count(("query", "visible")), 10)
        self.assertEqual(roles.count(("query", "occluded")), 10)
        self.assertEqual(roles.count(("query", "removed")), 10)
        self.assertEqual([s["phase"] for s in p["slots"][::10]], ["reference", "visible", "occluded", "removed"])

    def test_one_lighting(self):
        p = self.plan("--arrangement", "A01", "--ordering", "B", "--lightings", "L1")
        self.assertEqual((p["n_slots"], p["lamp_toggles"]), (20, 0))

    def test_bad_plan_inputs(self):
        for args in (["--stops", "P1,P3,P2,P4,P5"], ["--stops", "P3,P1,P2,P4"], ["--arrangement", "B01"]):
            argv = ["--arrangement", "A01", "--ordering", "A", *args]
            code, _, _ = self.run_tool("plan", *argv, common=False)
            self.assertEqual(code, 24, args)


class TestNtp(Base):
    """NTP is a warning (DECISIONS 2026-10-12): the sitting goes on and the failed check is recorded."""
    def check_warned(self, env, timedatectl=None):
        code, _, err = (self.run_tool("sitting", "--lock-config", str(self.lock_config), "--golden",
                                      str(self.golden), "--timedatectl", timedatectl, script=["", ""], env=env)
                        if timedatectl else self.sitting(env=env))
        self.assertEqual(code, 0, err)
        self.assertIn("WARNING: NTP is not synchronised", err)
        ev = [e for e in self.log() if e["event"] == "ntp_preflight"]
        self.assertEqual(len(ev), 1)
        self.assertFalse(ev[0]["ok"])
        self.assertIn("not synchronised", ev[0]["warning"])
        ntp = json.loads((self.rundir / "sitting.json").read_text())["ntp"]
        self.assertFalse(ntp["ok"])
        self.assertEqual(ntp["warning"], ev[0]["warning"])
        return ev[0]

    def test_ntp_no(self):
        ev = self.check_warned({"FAKE_NTP": "no"})
        self.assertEqual((ev["stdout"], ev["returncode"]), ("no", 0))

    def test_ntp_nonzero_rc(self):
        ev = self.check_warned({"FAKE_NTP_RC": "1"})
        self.assertEqual((ev["stdout"], ev["returncode"]), ("yes", 1))

    def test_ntp_missing_binary(self):
        ev = self.check_warned(None, timedatectl=str(self.t / "no-such-timedatectl"))
        self.assertIn("not found", ev["error"])

    def test_ntp_ok_has_no_warning(self):
        self.assertEqual(self.sitting()[0], 0)
        ntp = json.loads((self.rundir / "sitting.json").read_text())["ntp"]
        self.assertTrue(ntp["ok"])
        self.assertNotIn("warning", ntp)

    def test_arrangement_also_checks_ntp(self):
        self.assertEqual(self.sitting()[0], 0)
        code, _, err = self.arrangement(["q"], env={"FAKE_NTP": "no"})
        self.assertEqual(code, 22, err)  # got past the NTP check to the first prompt
        self.assertIn("WARNING: NTP is not synchronised", err)
        m = self.manifest()
        self.assertFalse(m["ntp"]["ok"])
        self.assertIn("not synchronised", m["ntp"]["warning"])


class TestSitting(Base):
    def test_sitting_ok(self):
        code, out, err = self.sitting()
        self.assertEqual(code, 0, err)
        s = json.loads((self.rundir / "sitting.json").read_text())
        self.assertEqual(s["schema"], "cmpe249.sitting.v1")
        self.assertEqual(s["locks"], {"L1": "L1-v1", "L2": "L2-v1"})
        self.assertEqual(s["pose"]["verdict"], "ok")
        self.assertEqual(s["pose"]["file"], "run/pose_check_L1-v1.json")
        self.assertTrue((self.rundir / "pose_check_L1-v1.json").exists())
        self.assertTrue(s["ntp"]["ok"])
        lock = json.loads((self.sdir / "lock.json").read_text())
        self.assertEqual(lock["locks"]["L2"]["copied_from"], "L1-v1")
        cal = [e for e in self.log() if e["event"] == "pi_capture"]
        self.assertIn("--lock-config", cal[0]["command"])
        self.assertEqual(cal[1]["command"][cal[1]["command"].index("--copy-from") + 1], "L1")
        self.assertIn("Camera at P3 (floor mark), lamps set for L1. Enter = calibrate, q = abort", out)
        self.assertIn("Keep the camera at P3. Switch lamps to L2. Enter when switched", out)

    def test_second_sitting_refused(self):
        self.assertEqual(self.sitting()[0], 0)
        code, _, err = self.sitting()
        self.assertEqual(code, 24, err)
        self.assertIn("already has a sitting", err)

    def test_existing_lock_refused(self):
        p = subprocess.run([sys.executable, str(PI_CAPTURE), "calibrate", "--root", str(self.root), "--session",
                            self.session, "--lighting", "L1", "--lens-position", "1.1", "--rpicam", self.rpicam],
                           capture_output=True, text=True, env=self.env())
        self.assertEqual(p.returncode, 0, p.stderr)
        code, _, err = self.sitting()
        self.assertEqual(code, 24, err)

    def test_pose_flag_is_advisory(self):
        code, out, err = self.sitting(env={"FAKE_POSE_VERDICT": "flag"})
        self.assertEqual(code, 0, err)
        self.assertIn("WARNING: pose check FLAG vs the golden (shift 2.81 px > flag 2.0 px)", out)
        s = json.loads((self.rundir / "sitting.json").read_text())
        self.assertEqual(s["pose"]["verdict"], "flag")
        self.assertEqual(s["pose"]["exit_code"], 10)
        self.assertTrue(s["pose"]["advisory"])
        # the arrangement accepts a flagged sitting (precondition passes; then q)
        code, _, err = self.arrangement(["q"], "--card", str(self.card()))
        self.assertEqual(code, 22, err)

    def card(self, arr="A01"):
        p = self.t / f"card_{arr}.json"
        p.write_text(json.dumps({"arrangement": arr, "phases": {"occluded": "box 6 cm left of the mug"}}))
        return p

    def test_pose_refuse_is_advisory(self):
        code, out, err = self.sitting(env={"FAKE_POSE_VERDICT": "refuse"})
        self.assertEqual(code, 0, err)
        self.assertIn("WARNING: pose check REFUSE", out)
        s = json.loads((self.rundir / "sitting.json").read_text())
        self.assertEqual(s["pose"]["verdict"], "refuse")
        code, _, err = self.arrangement(["q"], "--card", str(self.card()))
        self.assertEqual(code, 22, err)  # precondition passes; then q

    def test_pose_error_and_missing_tool_are_advisory(self):
        code, out, err = self.sitting(env={"FAKE_POSE_VERDICT": "error"})
        self.assertEqual(code, 0, err)
        self.assertIn("WARNING: pose check ERROR", out)
        self.session = "S2"
        self.sdir, self.rundir = self.root / "S2", self.root / "S2" / "run"
        code, out, err = self.run_tool("sitting", "--lock-config", str(self.lock_config), "--golden",
                                       str(self.golden), "--pose-check", str(self.t / "missing_pose.py"),
                                       script=["", ""])
        self.assertEqual(code, 0, err)
        s = json.loads((self.rundir / "sitting.json").read_text())
        self.assertEqual(s["pose"]["verdict"], "error")

    def test_lock_not_applied(self):
        code, _, err = self.sitting(env={"FAKE_RPICAM_SHUTTER": "20000"})
        self.assertEqual(code, 24, err)
        self.assertIn("did not apply the lock", err)
        self.assertFalse((self.rundir / "pose_check_L1-v1.json").exists())

    def test_abort_at_first_prompt(self):
        code, _, err = self.sitting(["q"])
        self.assertEqual(code, 22, err)
        self.assertFalse((self.sdir / "lock.json").exists())
        self.assertIn("abort", [e["event"] for e in self.log()])

    def test_bad_lock_config(self):
        self.lock_config.write_text(json.dumps({"schema": "nope"}))
        code, _, err = self.sitting()
        self.assertEqual(code, 24, err)


class Full(Base):
    def setUp(self):
        super().setUp()

    def do_sitting(self, lightings="L1,L2"):
        script = [""] * (1 + len(lightings.split(",")) - 1)
        code, _, err = self.sitting(script, "--lightings", lightings)
        self.assertEqual(code, 0, err)


class TestFullRuns(Full):
    def full(self, ordering, n_prompts, toggles):
        self.do_sitting()
        code, out, err = self.run_tool("arrangement", "--arrangement", "A01", "--ordering", ordering,
                                       script=[""] * n_prompts, env=None)
        self.assertEqual(code, 0, err)
        m = self.manifest()
        self.assertEqual(m["status"], "complete")
        self.assertEqual(len(m["slots"]), 40)
        self.assertTrue(all(s["selected_take"] == 1 and len(s["takes"]) == 1 for s in m["slots"]))
        self.assertEqual(m["lamp_toggles"], toggles)
        self.assertEqual(m["n_prompts"], n_prompts)
        plan = self.plan("--arrangement", "A01", "--ordering", ordering)
        self.assertEqual([{k: s[k] for k in plan["slots"][0]} for s in m["slots"]], plan["slots"])
        # takes copy pi_capture's manifest rows (joinable on stem)
        rows = {r["stem"]: r for r in self.pc_status()["manifest"]}
        self.assertEqual(len(rows), 40)
        for s in m["slots"]:
            t = s["takes"][0]
            self.assertEqual(t["stem"], f"A01_{s['position']}_{s['role']}_{s['condition']}_{s['lighting']}_t01")
            self.assertEqual((t["sha256"], t["path"]), (rows[t["stem"]]["sha256"], rows[t["stem"]]["path"]))
        roles = [r["role"] for r in rows.values()]
        self.assertEqual((roles.count("reference"), roles.count("query")), (10, 30))
        log = self.log()
        self.assertEqual(sum(e["event"] == "lamp_set" for e in log), 1)
        self.assertEqual(sum(e["event"] == "lamp_toggle" for e in log), toggles)
        self.assertEqual(out.count("Set lamps to L1. Enter when set"), 1)
        self.assertEqual(out.count("Switch lamps to"), toggles)
        prompts = [e for e in log if e["event"] == "prompt"]
        self.assertTrue(all({"time_utc", "time_local", "monotonic_s", "slot", "text"} <= set(e) for e in prompts))
        responses = [e for e in log if e["event"] == "response"]
        self.assertTrue(all(e["source"] == "script" and "elapsed_s" in e for e in responses))
        self.assert_no_forbidden()
        return m, out

    def test_full_ordering_a(self):
        m, out = self.full("A", N_PROMPTS_L2_A, 20)
        self.assertIn("[slot 1/40] reference P3 L1: move camera to P3. Enter = shoot, q = abort", out)
        self.assertIn("[slot 2/40] reference P3 L2: camera stays at P3. Enter = shoot, q = abort", out)
        self.assertIn("PHASE 4/4 removed: Remove the target; keep the occluder exactly as in occluded. "
                      "Enter when the scene is set, q = abort", out)

    def test_full_ordering_b(self):
        self.full("B", N_PROMPTS_L2_B, 4)


class TestFlows(Full):
    def test_one_lighting(self):
        self.do_sitting("L1")
        s = json.loads((self.rundir / "sitting.json").read_text())
        self.assertEqual(s["locks"], {"L1": "L1-v1"})
        code, out, err = self.arrangement([""] * N_PROMPTS_L1, ordering="B")
        self.assertEqual(code, 0, err)
        m = self.manifest()
        self.assertEqual((len(m["slots"]), m["lamp_toggles"]), (20, 0))
        self.assertEqual(len(self.pc_status()["manifest"]), 20)
        self.assertNotIn("Switch lamps", out)
        self.assertEqual(out.count("Set lamps to L1"), 1)

    def test_retake(self):
        self.do_sitting("L1")
        script = ["", "", "", "r", "lamp moved", ""] + [""] * N_PROMPTS_L1
        code, out, err = self.arrangement(script, ordering="A")
        self.assertEqual(code, 0, err)
        s0 = self.manifest()["slots"][0]
        self.assertEqual([t["take"] for t in s0["takes"]], [1, 2])
        self.assertEqual(s0["selected_take"], 2)
        self.assertEqual(s0["takes"][1]["note"], "retake: lamp moved")
        side = json.loads((self.sdir / "A01/reference/P3/A01_P3_reference_visible_L1_t02.json").read_text())
        self.assertEqual(side["labels"]["note"], "retake: lamp moved")
        self.assertEqual(side["protocol"]["overrides"], [])
        self.assertIn("(take 2): camera stays at P3", out)

    def test_abort_and_resume(self):
        self.do_sitting("L1")
        # phase, lamp-set, 3 slots (shoot + accept), then q at slot 4's shot prompt
        code, _, err = self.arrangement(["", ""] + ["", ""] * 3 + ["q"], ordering="A")
        self.assertEqual(code, 22, err)
        m = self.manifest()
        self.assertEqual(m["status"], "aborted")
        self.assertEqual(sum(s["selected_take"] is not None for s in m["slots"]), 3)
        self.assertIn("abort", [e["event"] for e in self.log()])
        # a plain re-run is refused; --resume with another ordering is refused
        self.assertEqual(self.arrangement([""] * 10, ordering="A")[0], 24)
        self.assertEqual(self.arrangement([""] * 10, "--resume", ordering="B")[0], 24)
        code, out, err = self.arrangement([""] * N_PROMPTS_L1, "--resume", ordering="A")
        self.assertEqual(code, 0, err)
        m = self.manifest()
        self.assertEqual(m["status"], "complete")
        self.assertTrue(all(s["selected_take"] == 1 for s in m["slots"]))
        self.assertEqual(len(self.pc_status()["manifest"]), 20)
        self.assertIn("(resumed; check the scene is still as it was)", out)
        self.assertEqual(len(m["resumed_utc"]), 1)
        # resuming a complete arrangement is refused
        self.assertEqual(self.arrangement([""], "--resume", ordering="A")[0], 24)

    def test_abort_after_shot_then_resume_takes_next_take(self):
        self.do_sitting("L1")
        code, _, err = self.arrangement(["", "", "", "q"], ordering="A")  # q at the "saved" prompt
        self.assertEqual(code, 22, err)
        s0 = self.manifest()["slots"][0]
        self.assertEqual((len(s0["takes"]), s0["selected_take"]), (1, None))
        code, _, err = self.arrangement([""] * N_PROMPTS_L1, "--resume", ordering="A")
        self.assertEqual(code, 0, err)
        s0 = self.manifest()["slots"][0]
        self.assertEqual(([t["take"] for t in s0["takes"]], s0["selected_take"]), ([1, 2], 2))
        self.assertTrue(s0["takes"][1]["note"].startswith("retake: resumed after abort"))

    def test_lock_failed_shot_needs_explicit_accept(self):
        """Review finding RA-LOCK-FAIL-ACCEPTED: a shot pi_capture reports with lock_ok false is not accepted
        by plain Enter; 'a' + a note accepts it and is recorded in the take."""
        self.do_sitting("L1")
        env = {"FAKE_RPICAM_SHUTTER": "20000"}
        # phase, lamp-set, shoot slot 0, Enter (refused), a, note, then q at slot 1's shot prompt
        code, out, err = self.arrangement(["", "", "", "", "a", "lamp test only", "q"], ordering="A", env=env)
        self.assertEqual(code, 22, err)
        self.assertIn("CHECK FAILED: lock (QC will FAIL this shot). r = retake, a = accept anyway", out)
        self.assertIn("not understood; expected 'r' / 'a' or q", out)
        s0 = self.manifest()["slots"][0]
        self.assertEqual(s0["selected_take"], 1)
        self.assertIs(s0["takes"][0]["lock_ok"], False)
        self.assertEqual(s0["takes"][0]["accepted_despite"], {"failed_checks": ["lock"], "note": "lamp test only"})
        self.assertIn("accepted_despite", [e["event"] for e in self.log()])
        # a retake from that prompt works as before; a good shot needs only Enter
        code, out, err = self.arrangement(["", "", "", "r", "retry lock", "", "", "q"], "--resume", ordering="A",
                                          env=env)
        self.assertEqual(code, 22, err)
        s1 = self.manifest()["slots"][1]
        self.assertEqual(([t["take"] for t in s1["takes"]], s1["selected_take"]), ([1, 2], None))
        code, out, err = self.arrangement([""] * N_PROMPTS_L1, "--resume", ordering="A")
        self.assertEqual(code, 0, err)
        m = self.manifest()
        self.assertEqual(m["slots"][1]["selected_take"], 3)
        self.assertIsNone(m["slots"][1]["takes"][2]["accepted_despite"])
        self.assertNotIn("CHECK FAILED", out)

    def test_lighting_failed_shot_needs_explicit_accept(self):
        self.do_sitting("L1")
        code, out, err = self.arrangement(["", "", "", "a", "lamp drift noted", "q"], ordering="A",
                                          env={"FAKE_RPICAM_LUX": "80.0"})
        self.assertEqual(code, 22, err)
        self.assertIn("CHECK FAILED: lighting (QC will flag this shot)", out)
        t = self.manifest()["slots"][0]["takes"][0]
        self.assertEqual(t["accepted_despite"]["failed_checks"], ["lighting"])

    def test_resume_waits_for_orphaned_pi_capture(self):
        """Review finding RA-RESUME-ORPHAN: the sequencer was killed while its pi_capture child was still
        shooting; an immediate --resume waits for that child, adopts its shot and completes."""
        self.do_sitting("L1")
        code, _, err = self.arrangement(["", "", "", "", "q"], ordering="A")  # slot 0 done, q at slot 1
        self.assertEqual(code, 22, err)
        slow = self.t / "rpicam_slow.py"
        slow.write_text(SLOW_RPICAM.format(python=sys.executable, fake=str(FAKE_CAM)))
        os.chmod(slow, 0o755)
        # the orphan: slot 1 (reference P1 L1) take 1, as the killed sequencer had started it
        orphan = subprocess.Popen([sys.executable, self.pi_capture, "shoot", "--root", str(self.root), "--session",
                                   self.session, "--arrangement", "A01", "--position", "P1", "--role", "reference",
                                   "--condition", "visible", "--lighting", "L1", "--take", "1",
                                   "--rpicam", str(slow)], env=self.env({"SLOW_RPICAM_S": "2.5"}),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        import fcntl
        t_end = time.monotonic() + 30
        with open(self.sdir / ".session.lock", "a") as f:  # wait until the orphan holds the session lock
            while time.monotonic() < t_end:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    time.sleep(0.02)
                except BlockingIOError:
                    break
        code, out, err = self.arrangement([""] * (N_PROMPTS_L1 + 2), "--resume", ordering="A")
        o_out, o_err = orphan.communicate(timeout=60)
        self.assertEqual(orphan.returncode, 0, o_err)
        self.assertEqual(code, 0, err)
        self.assertIn("waiting for a running pi_capture.py on this session to finish", out)
        events = [e["event"] for e in self.log()]
        self.assertIn("wait_session_lock", events)
        self.assertIn("session_lock_free", events)
        s1 = self.manifest()["slots"][1]
        self.assertEqual([t["take"] for t in s1["takes"]], [1, 2])
        self.assertTrue(s1["takes"][0]["note"].startswith("adopted on resume"))
        self.assertEqual(s1["selected_take"], 2)
        self.assertEqual(self.manifest()["status"], "complete")

    def test_session_busy_mid_run_is_retryable(self):
        """pi_capture's 'another process is using session' refusal (exit 4) mid-run: prompt, wait, retry."""
        self.do_sitting("L1")
        holder = subprocess.Popen([sys.executable, "-c", LOCK_HOLDER, str(self.rundir / "run_log.jsonl"),
                                   str(self.sdir / ".session.lock"), '"lamp_set"', "2.5"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            code, out, err = self.arrangement([""] * (N_PROMPTS_L1 + 1), "--lamp-settle-s", "1.0", ordering="A")
        finally:
            h_out, h_err = holder.communicate(timeout=60)
        self.assertEqual(h_out.strip(), "held", h_err)
        self.assertEqual(code, 0, err)
        self.assertIn("another pi_capture.py is using this session, nothing was saved. Enter = wait", out)
        rcs = [e["returncode"] for e in self.log() if e["event"] == "pi_capture" and e["slot"] == 0]
        self.assertEqual(rcs, [4, 0])
        s0 = self.manifest()["slots"][0]
        self.assertEqual(([t["take"] for t in s0["takes"]], s0["selected_take"]), ([1], 1))

    def test_other_protocol_refusal_still_fatal(self):
        """Only session-lock contention is retried; any other pi_capture exit 4 still ends the run (23)."""
        self.do_sitting("L1")
        refuse = self.t / "pi_capture_refuse.py"
        refuse.write_text("import os, sys\n"
                          "if sys.argv[1:2] == ['shoot']:\n"
                          "    print('pi_capture.py: error: arrangement A01 has queries; references come first',"
                          " file=sys.stderr)\n"
                          "    sys.exit(4)\n"
                          f"os.execv(sys.executable, [sys.executable, {self.pi_capture!r}, *sys.argv[1:]])\n")
        self.pi_capture = str(refuse)
        code, out, err = self.arrangement([""] * 10, ordering="A")
        self.assertEqual(code, 23, err)
        self.assertNotIn("is using this session", out)
        rcs = [e["returncode"] for e in self.log() if e["event"] == "pi_capture" and e["slot"] == 0]
        self.assertEqual(rcs, [4])

    def test_camera_failure_retry(self):
        self.do_sitting("L1")
        wrapper = self.t / "rpicam_fail_once.py"
        wrapper.write_text(FAIL_ONCE_RPICAM.format(python=sys.executable, fake=str(FAKE_CAM)))
        os.chmod(wrapper, 0o755)
        flag = self.t / "fail.flag"
        flag.write_text("1")
        self.rpicam = str(wrapper)
        # phase, lamp-set, shoot -> camera fails -> retry prompt (Enter) -> saved (Enter), rest Enter
        code, out, err = self.arrangement([""] * (N_PROMPTS_L1 + 1), ordering="A",
                                          env={"FAIL_ONCE_FLAG": str(flag)})
        self.assertEqual(code, 0, err)
        self.assertIn("camera failure, nothing was saved. Enter = retry the same take, q = abort", out)
        s0 = self.manifest()["slots"][0]
        self.assertEqual(([t["take"] for t in s0["takes"]], s0["selected_take"]), ([1], 1))
        rcs = [e["returncode"] for e in self.log() if e["event"] == "pi_capture" and e["slot"] == 0]
        self.assertEqual(rcs, [5, 0])

    def test_capture_refusal_exits_23(self):
        self.do_sitting("L1")
        # make the first shot's image path collide: pi_capture refuses to overwrite (exit 3)
        d = self.sdir / "A01" / "reference" / "P3"
        d.mkdir(parents=True)
        (d / "A01_P3_reference_visible_L1_t01.jpg").write_bytes(b"x")
        code, _, err = self.arrangement([""] * 10, ordering="A")
        self.assertEqual(code, 23, err)
        self.assertEqual(self.manifest()["status"], "aborted")

    def test_preconditions(self):
        self.assertEqual(self.arrangement([""])[0], 24)  # no sitting yet
        self.do_sitting("L1")
        self.assertEqual(self.arrangement([""], "--stops", "P1,P3,P2,P4,P5")[0], 24)
        self.assertEqual(self.arrangement([""], arr="B01")[0], 24)
        self.assertEqual(self.arrangement([""], arr="A1")[0], 24)
        bad_card = self.t / "card.json"
        bad_card.write_text(json.dumps({"arrangement": "A02"}))
        self.assertEqual(self.arrangement([""], "--card", str(bad_card))[0], 24)
        self.assertEqual(self.arrangement([""], "--resume")[0], 24)  # nothing to resume
        self.assertFalse((self.rundir / "A01.json").exists())

    def test_card_text_and_lamp_settle(self):
        self.do_sitting("L1,L2")
        card = self.t / "card.json"
        card.write_text(json.dumps({"arrangement": "A03", "phases": {"occluded": "cereal box 6 cm in front"}}))
        argv = ["arrangement", "--arrangement", "A03", "--ordering", "B", "--card", str(card)]
        code, out, err = self.run_tool(*argv, script=[""] * N_PROMPTS_L2_B)
        self.assertEqual(code, 0, err)  # lamp settle 0 here
        self.assertIn("PHASE 3/4 occluded: cereal box 6 cm in front. Enter when", out)
        m = self.manifest("A03")
        self.assertEqual(m["card"]["path"], str(card.resolve()))

    def test_lamp_settle_waits(self):
        self.do_sitting("L1,L2")
        settle = 0.3
        argv = ["arrangement", "--arrangement", "A04", "--ordering", "B"]
        script = self.t / "s.json"
        script.write_text(json.dumps([""] * N_PROMPTS_L2_B))
        cmd = [sys.executable, str(RUN), *argv, "--root", str(self.root), "--session", self.session,
               "--pi-capture", self.pi_capture, "--python", sys.executable, "--rpicam", self.rpicam,
               "--timedatectl", str(self.t / "fake_timedatectl.py"), "--lamp-settle-s", str(settle),
               "--script", str(script)]
        p = subprocess.run(cmd, capture_output=True, text=True, env=self.env())
        self.assertEqual(p.returncode, 0, p.stderr)
        log = self.log()
        waits = [e for e in log if e["event"] == "lamp_settle_wait"]
        self.assertEqual(len(waits), 5)  # lamp-set + 4 toggles
        # each lamp change: the next pi_capture shoot starts >= settle after the lamp response
        for i, e in enumerate(log):
            if e["event"] in ("lamp_set", "lamp_toggle") and e.get("slot") is not None:
                t_lamp = e["monotonic_s"]
                wait = next(x for x in log[i:] if x["event"] == "lamp_settle_wait")
                self.assertGreater(wait["wait_s"], 0)
                self.assertLessEqual(wait["wait_s"], settle)
                # the wait is logged just before the shot; with it the shot starts >= settle later
                self.assertGreaterEqual(wait["monotonic_s"] + wait["wait_s"] - t_lamp, settle - 0.01)

    def test_script_exhausted(self):
        self.do_sitting("L1")
        code, _, err = self.arrangement(["", "", ""], ordering="A")
        self.assertEqual(code, 25, err)
        self.assertIn("script_exhausted", [e["event"] for e in self.log()])
        self.assertEqual(self.manifest()["status"], "aborted")

    def test_script_exhausted_in_sitting(self):
        code, _, err = self.sitting([])
        self.assertEqual(code, 25, err)


class TestFakeCameraGuard(Base):
    def test_fake_camera_refused_on_default_root(self):
        real = str(Path("~/cmpe249/data").expanduser())
        for root_args in ([], ["--root", real]):
            argv = [sys.executable, str(RUN), "sitting", "--session", "S1", "--fake-camera", *root_args,
                    "--timedatectl", str(self.t / "fake_timedatectl.py")]
            p = subprocess.run(argv, capture_output=True, text=True, env=self.env())
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("--fake-camera", p.stderr)

    def test_fake_camera_on_temp_root(self):
        argv = [sys.executable, str(RUN), "sitting", "--session", "S1", "--fake-camera", "--root", str(self.root),
                "--pi-capture", self.pi_capture, "--pose-check", str(self.t / "fake_pose_check.py"),
                "--timedatectl", str(self.t / "fake_timedatectl.py"), "--lamp-settle-s", "0",
                "--lock-config", str(self.lock_config), "--golden", str(self.golden), "--lightings", "L1"]
        sp = self.t / "s.json"
        sp.write_text(json.dumps([""]))
        p = subprocess.run(argv + ["--script", str(sp)], capture_output=True, text=True, env=self.env())
        self.assertEqual(p.returncode, 0, p.stderr)
        cmd = next(e for e in self.log() if e["event"] == "pi_capture")["command"]
        self.assertEqual(cmd[cmd.index("--rpicam") + 1], str(CAPTURE / "tests" / "fake_rpicam_still.py"))


# ----------------------------------------------------------------- real pose_check (GB10)
REAL_POSE = CAPTURE / "pose_check.py"
REAL_CFG = HERE / "capture_lock_v1_close_2026-10-05.json"  # the close-spot config that matches ref_final.jpg
SHELF = REPO / "data" / "raw" / "2026-10-05_shelf-final"
GOLDEN = SHELF / "pi" / "ref_final.jpg"


def _real_ready() -> str | None:
    if not REAL_POSE.exists():
        return "capture/pose_check.py not built yet"
    if not (GOLDEN.exists() and REAL_CFG.exists()):
        return "repo shelf-final data not present (Pi)"
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return "no cv2/numpy in this interpreter"
    return None


@unittest.skipIf(_real_ready() is not None, _real_ready() or "")
class TestRealPoseCheck(Base):
    """Smoke test with the real pose_check.py, the real lock config and golden, and real frames fed
    through the fake camera. Measured on the static mask (MODULE1 section 8): ref_lp1.45 is
    0.217 px from ref_final (ok), frame e is 2.808 px (flag). The frame reaches pi_capture through
    IMAGE_RPICAM (the fake camera, then the real JPEG copied to -o), which does what builder 0's
    FAKE_RPICAM_IMAGE will do without depending on it."""

    def real_sitting(self, image, script):
        wrapper = self.t / "rpicam_image.py"
        wrapper.write_text(IMAGE_RPICAM.format(python=sys.executable, fake=str(FAKE_CAM)))
        os.chmod(wrapper, 0o755)
        self.rpicam = str(wrapper)
        argv = ["sitting", "--lock-config", str(REAL_CFG), "--golden", str(GOLDEN), "--lightings", "L1",
                "--pose-check", str(REAL_POSE)]
        return self.run_tool(*argv, script=script, env={"TEST_RPICAM_IMAGE": str(image)})

    def test_ref_lp145_ok(self):
        code, out, err = self.real_sitting(SHELF / "pi" / "ref_lp1.45.jpg", [""])
        self.assertEqual(code, 0, err + out)
        pose = json.loads((self.rundir / "pose_check_L1-v1.json").read_text())
        self.assertEqual(pose["verdict"], "ok")
        self.assertLess(pose["translation"]["shift_px"], 1.0)

    def test_frame_e_flag(self):
        code, out, err = self.real_sitting(SHELF / "geometry" / "rpi_view_2026-10-05e.jpg", [""])
        self.assertEqual(code, 0, err + out)
        pose = json.loads((self.rundir / "pose_check_L1-v1.json").read_text())
        self.assertEqual(pose["verdict"], "flag")
        self.assertGreater(pose["translation"]["shift_px"], 2.0)
        self.assertLess(pose["translation"]["shift_px"], 5.0)
        s = json.loads((self.rundir / "sitting.json").read_text())
        self.assertEqual(s["pose"]["verdict"], "flag")


if __name__ == "__main__":
    unittest.main()
