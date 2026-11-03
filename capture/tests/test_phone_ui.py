"""Tests for capture/phone_ui.py (stdlib; OpenCV/numpy only for the preview and change-hint tests).

Run from the repo root (Jetson/GB10) or ~/cmpe249 (Pi), like the other capture tests:
    python -m unittest discover -s capture/tests -t .

The server runs in-process on a free port. Its job thread drives the REAL run_arrangement.py ->
pi_capture.py chain with the fake rpicam-still in a temporary data root, as test_run_arrangement.py
does, and each test plays the phone over HTTP.
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURE = HERE.parent
if str(CAPTURE) not in sys.path:
    sys.path.insert(0, str(CAPTURE))
import phone_ui  # noqa: E402
import run_arrangement as ra  # noqa: E402

HAS_LOCK_CONFIG = "--lock-config" in (CAPTURE / "pi_capture.py").read_text()
TOKEN = "testtoken123"

FAKE_TIMEDATECTL = """\
import os, sys
print(os.environ.get("FAKE_NTP", "yes"))
sys.exit(int(os.environ.get("FAKE_NTP_RC", "0")))
"""

FAKE_POSE_CHECK = """\
import argparse, json, sys
p = argparse.ArgumentParser()
for k in ("--image", "--lock-config", "--golden", "--out"):
    p.add_argument(k)
a = p.parse_args()
out = {"schema": "cmpe249.pose_check.v1", "tool": {"name": "fake_pose_check.py", "version": "0"},
       "image": {"path": a.image}, "golden": {"path": a.golden},
       "translation": {"dx": 0.0, "dy": 0.2, "shift_px": 0.2, "cc": 0.99},
       "euclidean": {"dx": 0.0, "dy": 0.2, "rot_deg": 0.0, "cc": 0.99},
       "verdict": "ok", "reasons": [], "exit_code": 0}
with open(a.out, "x") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out))
"""

LOCK_CONFIG = {
    "schema": "cmpe249.capture_lock.v1", "lock_version": 1,
    "lock": {"sensor_mode": "2304:1296:10:P", "size": [2304, 1296], "scaler_crop": [0, 0, 4608, 2592],
             "shutter_us": 25000, "gain": 1.381, "awb_gains": [1.5753, 2.5111], "lens_position": 1.10,
             "tolerances": {"shutter_rel": 0.005, "gain_rel": 0.02, "awb_abs": 0.001, "lens_abs": 0.01}},
    "reference": {"position": "P3", "sha256": "0" * 64, "size": [2304, 1296]},
}


def write_test_jpeg(path: Path, square: bool = False) -> None:
    """A decodable 2304x1296 still (gradient + boxes), optionally with one extra white square."""
    import cv2
    import numpy as np
    img = np.zeros((1296, 2304, 3), np.uint8)
    img[:] = np.linspace(40, 140, 2304, dtype=np.uint8)[None, :, None]
    cv2.rectangle(img, (300, 300), (700, 900), (30, 160, 60), -1)
    cv2.rectangle(img, (1500, 400), (1900, 1000), (180, 60, 40), -1)
    if square:
        cv2.rectangle(img, (1000, 500), (1199, 699), (255, 255, 255), -1)
    assert cv2.imwrite(str(path), img)


@unittest.skipUnless(HAS_LOCK_CONFIG, "pi_capture.py without calibrate --lock-config")
class Base(unittest.TestCase):
    extra_env: dict = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.t = Path(self.tmp.name)
        self.root = t / "data"
        (t / "fake_timedatectl.py").write_text(f"#!{sys.executable}\n" + FAKE_TIMEDATECTL)
        os.chmod(t / "fake_timedatectl.py", 0o755)
        (t / "fake_pose_check.py").write_text(FAKE_POSE_CHECK)
        os.chmod(CAPTURE / "tests" / "fake_rpicam_still.py", 0o755)
        (t / "capture_lock.json").write_text(json.dumps(LOCK_CONFIG, indent=2))
        (t / "golden.jpg").write_bytes(b"\xff\xd8golden\xff\xd9")
        self.old_env = {k: os.environ.get(k) for k in ("FAKE_RPICAM_IMAGE", "CMPE249_DATA_ROOT")}
        for k in self.old_env:
            os.environ.pop(k, None)
        os.environ.update({k: v(t) if callable(v) else v for k, v in self.extra_env.items()})
        self.app, self.httpd = phone_ui.make_server([
            "--host", "127.0.0.1", "--port", "0", "--token", TOKEN, "--root", str(self.root), "--fake-camera", "--quiet",
            "--pose-check", str(t / "fake_pose_check.py"), "--python", sys.executable,
            "--timedatectl", str(t / "fake_timedatectl.py"), "--lamp-settle-s", "0",
            "--lock-config", str(t / "capture_lock.json"), "--golden", str(t / "golden.jpg")])
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.real_stdout, self.real_stderr = sys.stdout, sys.stderr

    def tearDown(self):
        st = self.app.state()  # never leave a job thread blocked on a prompt
        t_end = time.time() + 60
        while st["job"] and st["job"]["status"] == "running" and time.time() < t_end:
            if st["prompt"]:
                self.app.ui.answer(st["prompt"]["id"], "q")
            time.sleep(0.05)
            st = self.app.state()
        self.httpd.shutdown()
        self.httpd.server_close()
        for k, v in self.old_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.tmp.cleanup()

    # -- HTTP
    def http(self, path, body=None, cookie=True, raw=False):
        headers = {"Cookie": f"{phone_ui.COOKIE}={TOKEN}"} if cookie else {}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                payload, code, hdrs = r.read(), r.status, r.headers
        except urllib.error.HTTPError as e:
            payload, code, hdrs = e.read(), e.code, e.headers
        if raw:
            return code, payload, hdrs
        return code, json.loads(payload or b"{}")

    def state(self):
        code, st = self.http("/api/state")
        self.assertEqual(code, 200, st)
        return st

    def start(self, **body):
        code, out = self.http("/api/start", body)
        self.assertEqual(code, 200, out)

    def wait_prompt(self, timeout=60):
        t_end = time.time() + timeout
        while time.time() < t_end:
            st = self.state()
            if st["prompt"] or st["job"]["status"] != "running":
                return st
            time.sleep(0.03)
        self.fail("no prompt within timeout")

    def play(self, policy=lambda st: "", timeout=300):
        """Answer every prompt with policy(state) until the job ends; returns the final state."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            st = self.wait_prompt()
            if st["job"]["status"] != "running":
                return st
            code, out = self.http("/api/answer", {"prompt_id": st["prompt"]["id"], "answer": policy(st)})
            self.assertEqual(code, 200, out)
        self.fail("job did not finish within timeout")

    # -- files
    def sdir(self, session="S1"):
        return self.root / session

    def manifest(self, arr="A01", session="S1"):
        return json.loads((self.sdir(session) / "run" / f"{arr}.json").read_text())

    def log(self, session="S1"):
        return [json.loads(x) for x in (self.sdir(session) / "run" / "run_log.jsonl").read_text().splitlines()]

    def sitting(self, session="S1"):
        self.start(cmd="sitting", session=session)
        st = self.play()
        self.assertEqual(st["job"]["status"], "done", st["lines"][-10:])
        self.assertEqual(st["job"]["exit_code"], 0)


class TestAccess(Base):
    def test_token_required(self):
        code, out = self.http("/api/state", cookie=False)
        self.assertEqual(code, 403)
        code, out = self.http("/api/start", {"cmd": "sitting", "session": "S1"}, cookie=False)
        self.assertEqual(code, 403)
        self.assertFalse(self.root.joinpath("S1").exists())
        code, body, _ = self.http("/?t=wrong", cookie=False, raw=True)
        self.assertEqual(code, 403)
        req = urllib.request.Request(self.base + f"/?t={TOKEN}")
        opener = urllib.request.build_opener(type("NoRedirect", (urllib.request.HTTPRedirectHandler,),
                                                  {"redirect_request": lambda *a, **k: None}))
        try:
            opener.open(req, timeout=10)
            self.fail("expected a redirect")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 303)
            self.assertIn(f"{phone_ui.COOKIE}={TOKEN}", e.headers["Set-Cookie"])
            self.assertIn("SameSite=Strict", e.headers["Set-Cookie"])
        code, body, hdrs = self.http("/", raw=True)
        self.assertEqual(code, 200)
        self.assertIn(b"Rig capture", body)
        self.assertNotIn(b"__DEFAULTS__", body)

    def test_post_needs_json(self):
        req = urllib.request.Request(self.base + "/api/start", data=b"cmd=sitting",
                                     headers={"Cookie": f"{phone_ui.COOKIE}={TOKEN}",
                                              "Content-Type": "application/x-www-form-urlencoded"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 415)

    def test_fake_camera_refuses_real_root(self):
        with self.assertRaises(ValueError):
            phone_ui.make_server(["--port", "0", "--token", TOKEN, "--fake-camera",
                                  "--root", str(Path(ra.DEFAULT_DATA_ROOT).expanduser())])


class TestStartValidation(Base):
    def test_bad_requests(self):
        for body, code in ((dict(cmd="nope", session="S1"), 400), (dict(cmd="sitting", session="bad id!"), 400),
                           (dict(cmd="arrangement", session="S1", arrangement="X1", ordering="A"), 400),
                           (dict(cmd="arrangement", session="S1", arrangement="A01", ordering="C"), 400),
                           (dict(cmd="arrangement", session="S1", arrangement="A01", ordering="A", card="x"), 400)):
            got, out = self.http("/api/start", body)
            self.assertEqual(got, code, (body, out))
        self.assertIsNone(self.state()["job"])

    def test_one_job_at_a_time(self):
        self.start(cmd="sitting", session="S1")
        self.wait_prompt()
        code, out = self.http("/api/start", {"cmd": "sitting", "session": "S2"})
        self.assertEqual(code, 409, out)
        self.assertFalse(self.root.joinpath("S2").exists())


class TestFlow(Base):
    def test_sitting_then_arrangement(self):
        self.sitting()
        self.assertTrue((self.sdir() / "run" / "sitting.json").exists())
        self.assertIs(sys.stdout, self.real_stdout)  # the job restored the streams
        self.assertIs(sys.stderr, self.real_stderr)
        info = self.http("/api/session?s=S1")[1]
        self.assertTrue(info["sitting"])
        self.start(cmd="arrangement", session="S1", arrangement="a01", ordering="B",
                   card={"occluded": "White box in front of the mug", "removed": "  "})
        seen = []

        def policy(st):
            seen.append(st["prompt"]["text"])
            if st["progress"]:
                self.assertEqual(st["progress"]["n"], 40)
            return ""

        st = self.play(policy)
        self.assertEqual(st["job"]["status"], "done", st["lines"][-10:])
        man = self.manifest()
        self.assertEqual(man["status"], "complete")
        self.assertEqual(man["tool"]["version"], ra.TOOL_VERSION)
        self.assertTrue(all(s["selected_take"] == 1 for s in man["slots"]))
        self.assertEqual(len(seen), 4 + 1 + 4 + 2 * 40)  # phase prompts, lamp set, toggles, shot + saved
        self.assertEqual(man["n_prompts"], len(seen))
        # the card typed on the phone: written outside the session tree, recorded with its sha256
        card = Path(man["card"]["path"])
        self.assertEqual(card.parent, self.root.parent / "cards" / "S1")
        self.assertEqual(json.loads(card.read_text())["phases"], {"occluded": "White box in front of the mug"})
        self.assertEqual(man["card"]["sha256"], ra.sha256_file(card))
        self.assertTrue(any("White box in front of the mug" in t for t in seen))
        # every answer came from the phone
        resp = [e for e in self.log() if e["event"] == "response"]
        self.assertTrue(resp)
        self.assertEqual({e["source"] for e in resp}, {"phone"})
        self.assertEqual(st["shot"]["kind"], "slot")
        self.assertEqual(st["progress"]["done"], 40)

    def test_answers_are_validated(self):
        self.start(cmd="sitting", session="S1")
        st = self.wait_prompt()
        pid = st["prompt"]["id"]
        self.assertEqual(st["prompt"]["keys"], [""])
        code, out = self.http("/api/answer", {"prompt_id": pid + 1, "answer": ""})
        self.assertEqual(code, 409, out)
        code, out = self.http("/api/answer", {"prompt_id": pid, "answer": "x"})
        self.assertEqual(code, 400, out)
        code, out = self.http("/api/answer", {"prompt_id": str(pid), "answer": ""})
        self.assertEqual(code, 400, out)
        code, out = self.http("/api/answer", {"prompt_id": pid, "answer": ""})
        self.assertEqual(code, 200, out)
        code, out = self.http("/api/answer", {"prompt_id": pid, "answer": ""})  # a double tap
        self.assertEqual(code, 409, out)
        st = self.play()
        self.assertEqual(st["job"]["status"], "done")
        self.assertEqual(sum(1 for e in self.log() if e["event"] == "response"), 2)  # no extra answers

    def test_retake_abort_and_resume(self):
        self.sitting()
        self.start(cmd="arrangement", session="S1", arrangement="A02", ordering="A",
                   card={"occluded": "Box in front of the cup"})
        state = {"retake": False, "reason": False, "saved": 0}

        def policy(st):
            p = st["prompt"]
            if p["text"].startswith("saved "):
                if not state["retake"]:
                    state["retake"] = True
                    self.assertIn("r", p["keys"])
                    return "r"
                state["saved"] += 1
                return "q" if state["saved"] == 5 else ""
            if p["keys"] is None:
                state["reason"] = True
                self.assertIn("Retake reason", p["text"])
                return "hand in frame"
            return ""

        st = self.play(policy)
        self.assertEqual(st["job"]["status"], "aborted")
        self.assertEqual(st["job"]["exit_code"], ra.EXIT_ABORT)
        self.assertTrue(state["reason"])
        man = self.manifest("A02")
        self.assertEqual(man["status"], "aborted")
        self.assertEqual(sum(1 for s in man["slots"] if s["selected_take"] is not None), 4)
        first = man["slots"][0]
        self.assertEqual([t["take"] for t in first["takes"]], [1, 2])
        self.assertEqual(first["selected_take"], 2)
        self.assertEqual(first["takes"][1]["note"], "retake: hand in frame")
        # resume from the phone, without retyping the card: the arrangement's own card is passed again
        card = Path(man["card"]["path"])
        self.start(cmd="arrangement", session="S1", arrangement="A02", ordering="A", resume=True)
        seen = []
        st = self.play(lambda st: seen.append(st["prompt"]["text"]) or "")
        self.assertEqual(st["job"]["status"], "done", st["lines"][-10:])
        self.assertTrue(any("Box in front of the cup" in t for t in seen))
        man = self.manifest("A02")
        self.assertEqual(man["status"], "complete")
        self.assertEqual(len(man["resumed_utc"]), 1)
        self.assertEqual(Path(man["card"]["path"]), card)
        self.assertTrue(all(s["selected_take"] is not None for s in man["slots"]))

    def test_resume_refuses_a_changed_card(self):
        self.sitting()
        self.start(cmd="arrangement", session="S1", arrangement="A03", ordering="B",
                   card={"removed": "Take the mug out"})
        st = self.play(lambda st: "q" if st["prompt"]["text"].startswith("saved ") else "")
        self.assertEqual(st["job"]["status"], "aborted")
        card = Path(self.manifest("A03")["card"]["path"])
        card.write_text(card.read_text().replace("mug", "cup"))
        code, out = self.http("/api/start", {"cmd": "arrangement", "session": "S1", "arrangement": "A03",
                                             "ordering": "B", "resume": True})
        self.assertEqual(code, 409, out)
        self.assertIn("missing or changed", out["error"])

    def test_failed_job_reports_the_error(self):
        self.start(cmd="arrangement", session="S9", arrangement="A01", ordering="A")  # no sitting yet
        st = self.play()
        self.assertEqual(st["job"]["status"], "failed")
        self.assertEqual(st["job"]["exit_code"], ra.EXIT_PRECOND)
        self.assertIn("no sitting", st["job"]["message"])


@unittest.skipIf(phone_ui.cv2 is None, "OpenCV not available")
class TestPreviewAndChange(Base):
    extra_env = {"FAKE_RPICAM_IMAGE": lambda t: str(t / "still.jpg")}

    def setUp(self):
        super().setUp()
        write_test_jpeg(self.t / "still.jpg")

    def test_preview_overlay_and_no_change_hint(self):
        self.sitting()
        st = self.state()
        self.assertEqual(st["shot"]["kind"], "calibration")
        self.start(cmd="arrangement", session="S1", arrangement="A01", ordering="B")
        t_end = time.time() + 120
        cmp = None
        while time.time() < t_end:
            st = self.wait_prompt()
            shot = st["shot"]
            if shot and shot.get("phase") == "visible" and st["prompt"]["text"].startswith("saved "):
                cmp = shot.get("compare")
                if cmp and cmp["status"] == "ready":
                    break
                time.sleep(0.1)
                continue
            self.http("/api/answer", {"prompt_id": st["prompt"]["id"], "answer": ""})
        self.assertIsNotNone(cmp)
        self.assertEqual(cmp["phase"], "reference")
        self.assertEqual(cmp["pct"], 0.0)  # every fake still is the same image
        self.assertTrue(cmp["no_change"])
        self.assertFalse(cmp["registered"])  # the synthetic still has no shelf tags
        self.assertNotIn("overlay", cmp)
        import cv2
        import numpy as np
        q = f"s=S1&p={urllib.request.quote(shot['path'])}"
        code, body, hdrs = self.http(f"/api/img?{q}&w=320", raw=True)
        self.assertEqual(code, 200)
        self.assertEqual(hdrs["Content-Type"], "image/jpeg")
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[1], 320)
        code, body, _ = self.http(f"/api/overlay?s=S1&a={urllib.request.quote(cmp['path'])}"
                                  f"&b={urllib.request.quote(shot['path'])}", raw=True)
        self.assertEqual(code, 200)
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR))
        for bad in ("../../etc/passwd", "run/run_log.jsonl", "/etc/hostname", "../S1/../../x.jpg"):
            code, out = self.http(f"/api/img?s=S1&p={urllib.request.quote(bad)}")
            self.assertEqual(code, 404, bad)

    def test_change_share(self):
        a, b = self.t / "a.jpg", self.t / "b.jpg"
        write_test_jpeg(a)
        write_test_jpeg(b, square=True)
        same = phone_ui.change_share(a, a)
        self.assertEqual(same["status"], "ready")
        self.assertEqual(same["pct"], 0.0)
        self.assertTrue(same["no_change"])
        diff = phone_ui.change_share(a, b)
        # a 200 x 200 px square on 2304 x 1296 is 1.34 % of the frame
        self.assertGreater(diff["pct"], 1.0)
        self.assertLess(diff["pct"], 1.8)
        self.assertFalse(diff["no_change"])
        self.assertFalse(diff["registered"])
        self.assertTrue(diff["overlay"].startswith(b"\xff\xd8"))
        broken = self.t / "broken.jpg"
        broken.write_bytes(b"\xff\xd8\xff\xd9")
        self.assertEqual(phone_ui.change_share(a, broken)["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
