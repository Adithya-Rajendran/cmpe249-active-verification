#!/usr/bin/env python3
"""Guided capture sequencer for the CMPE 249 active-verification study (capture module 1).

Runs ON the Raspberry Pi (Python 3.13, standard library only). It never touches the camera
itself: every still is taken by calling `pi_capture.py` as a subprocess, and the sitting's pose is
judged by calling `pose_check.py` as a subprocess. Contract: capture/MODULE1.md section 3.

    run_arrangement.py sitting --session S                     # NTP (warn), lock-config calibration at P3,
                                                               # pose check, copy the lock to L2
    run_arrangement.py arrangement --session S --arrangement A01 --ordering A
                                                               # guided reference/visible/occluded/
                                                               # removed sweep over P1..P5 x L1/L2
    run_arrangement.py plan --arrangement A01 --ordering B     # print the slot order (no camera)

Files it owns live in <root>/<session>/run/ (run_log.jsonl, sitting.json,
pose_check_<lock_id>.json, <ARR>.json, .run_arrangement.lock). It never writes pi_capture's
files (lock.json, manifest.jsonl, session_log.jsonl, calibration/, the image tree).

Keys at every prompt: Enter = go on, q = abort (exit 22). After a shot: r = retake; a shot whose
lock or lighting check failed needs a = accept anyway (+ note) instead of Enter.
`--script FILE` answers the prompts from a JSON list of strings (tests, rehearsals).
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

TOOL_NAME = "run_arrangement.py"
TOOL_VERSION = "1.3.0"  # 1.3.0: the pose check is advisory (recorded, never stops a sitting); 1.2.0: NTP preflight is a recorded warning, not exit 20
SITTING_SCHEMA = "cmpe249.sitting.v1"
RUN_SCHEMA = "cmpe249.run_arrangement.v1"
LOCK_CONFIG_SCHEMA = "cmpe249.capture_lock.v1"

# Copied as literals from pi_capture.py 0.2.0 (never imported: this tool only calls it).
RE_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,63}$")
RE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,15}$")
# Study and pilot arrangement ids only (a subset of pi_capture's RE_ARRANGEMENT).
RE_ARR = re.compile(r"^(A\d{2}|PILOT\d{2})$")

POSITIONS = ("P1", "P2", "P3", "P4", "P5")
# P3 followed by the F order of configs/positions.json ("f_order": P1, P2, P4, P5).
DEFAULT_STOPS = ("P3", "P1", "P2", "P4", "P5")
DEFAULT_LIGHTINGS = ("L1", "L2")
# (phase, role, condition), in the only order the protocol allows: references before the scene
# changes (pi_capture refuses references after any query), removed last.
PHASES = (("reference", "reference", "visible"), ("visible", "query", "visible"),
          ("occluded", "query", "occluded"), ("removed", "query", "removed"))
PHASE_NAMES = tuple(p for p, _, _ in PHASES)
DEFAULT_PHASE_TEXT = {
    "reference": "Arrange the scene; target in place; occluder parked",
    "visible": "Scene unchanged except as the card says",
    "occluded": "Place the occluder as the card says",
    "removed": "Remove the target; keep the occluder exactly as in occluded",
}
# pi_capture flags this tool never emits (asserted on every command before it runs).
FORBIDDEN = ("pilot", "--allow-late-reference", "--accept-lock-change", "--episode",
             "--recalibrate", "--dry-run")

EXIT_OK, EXIT_USAGE = 0, 2
EXIT_NTP, EXIT_POSE, EXIT_ABORT, EXIT_CAPTURE, EXIT_PRECOND, EXIT_SCRIPT = 20, 21, 22, 23, 24, 25
# EXIT_NTP (1.2.0) and EXIT_POSE (1.3.0) are no longer returned (warnings now); the numbers stay reserved.
PC_EXIT_CAMERA = 5  # pi_capture: camera failure, nothing saved -> the same take may be retried
PC_EXIT_PROTOCOL = 4  # pi_capture: protocol refusal; retryable only for session-lock contention:
PC_BUSY_TEXT = "process is using session"  # "another pi_capture.py process is using session S"
POSE_EXIT = {0: "ok", 10: "flag", 11: "refuse", 12: "error", 3: "bad input"}

NTP_TIMEOUT_S, PI_CAPTURE_TIMEOUT_S, POSE_TIMEOUT_S = 10, 120, 180
SESSION_IDLE_TIMEOUT_S = PI_CAPTURE_TIMEOUT_S + 30  # an orphaned pi_capture finishes within its own timeout
NOTE_MAX = 500  # pi_capture --note limit

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "~/cmpe249/data"
DEFAULT_LOCK_CONFIG = "~/cmpe249/configs/capture_lock.json"
DEFAULT_GOLDEN = "~/cmpe249/golden/ref_final.jpg"


class RunError(Exception):
    def __init__(self, msg: str, code: int):
        super().__init__(msg)
        self.code = code


# --------------------------------------------------------------------------- helpers
def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def now_local() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_exclusive(path: Path, obj) -> None:
    with open(path, "x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:  # make the rename itself durable (power loss right after a shot)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def read_json(path: Path, what: str, code: int = EXIT_PRECOND) -> dict:
    try:
        obj = json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise RunError(f"{what} not found: {path}", code) from None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RunError(f"{what} {path} is not readable JSON ({e})", code) from None
    if not isinstance(obj, dict):
        raise RunError(f"{what} {path} is not a JSON object", code)
    return obj


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def split_labels(text: str, what: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()] if text else []


def parse_lightings(text: str) -> list[str]:
    labels = split_labels(text, "lightings")
    if not labels:
        raise RunError("--lightings needs at least one label", EXIT_PRECOND)
    for lab in labels:
        if not RE_TOKEN.fullmatch(lab):
            raise RunError(f"invalid lighting label {lab!r}: must match {RE_TOKEN.pattern}", EXIT_PRECOND)
    if len(set(labels)) != len(labels):
        raise RunError(f"--lightings has a repeated label: {labels}", EXIT_PRECOND)
    return labels


def parse_stops(text: str | None) -> list[str]:
    stops = split_labels(text, "stops") if text else list(DEFAULT_STOPS)
    if sorted(stops) != sorted(POSITIONS) or len(stops) != len(POSITIONS):
        raise RunError(f"--stops must be a permutation of {','.join(POSITIONS)}; got {stops}", EXIT_PRECOND)
    if stops[0] != "P3":
        raise RunError(f"--stops must start with P3 (the sitting's calibration stop); got {stops}", EXIT_PRECOND)
    return stops


def check_arrangement(arr: str) -> str:
    if not arr or not RE_ARR.fullmatch(arr):
        raise RunError(f"invalid arrangement id {arr!r}: must match {RE_ARR.pattern} (e.g. A01, PILOT01)",
                       EXIT_PRECOND)
    return arr


def stem_of(arr: str, pos: str, role: str, cond: str, lighting: str, take: int) -> str:
    """pi_capture's stem (no episode: this tool never passes --episode)."""
    return f"{arr}_{pos}_{role}_{cond}_{lighting}_t{take:02d}"


# ------------------------------------------------------------------------------ plan
def make_plan(ordering: str, stops: list[str], lightings: list[str]) -> list[dict]:
    """Deterministic slot list (MODULE1 section 3.5).

    A: both lightings at each stop; one stop-visit counter v runs across all phases, lightings in
       order at even v and reversed at odd v (20 toggles per arrangement at L = 2).
    B: per phase, all stops under one lighting, then all stops under the next; the lighting order
       alternates by phase (4 toggles per arrangement at L = 2)."""
    if ordering not in ("A", "B"):
        raise RunError(f"--ordering must be A or B, got {ordering!r}", EXIT_PRECOND)
    slots: list[dict] = []

    def add(phase, role, cond, pos, lighting):
        slots.append({"index": len(slots), "phase": phase, "role": role, "condition": cond,
                      "position": pos, "lighting": lighting})

    v = 0
    for p_i, (phase, role, cond) in enumerate(PHASES):
        if ordering == "A":
            for pos in stops:
                for lighting in (lightings if v % 2 == 0 else lightings[::-1]):
                    add(phase, role, cond, pos, lighting)
                v += 1
        else:
            for lighting in (lightings if p_i % 2 == 0 else lightings[::-1]):
                for pos in stops:
                    add(phase, role, cond, pos, lighting)
    return slots


def count_toggles(slots: list[dict]) -> int:
    return sum(1 for a, b in zip(slots, slots[1:]) if a["lighting"] != b["lighting"])


# ----------------------------------------------------------------------- run context
class Run:
    """Session paths, the append-only run log and the prompt/response channel."""

    def __init__(self, root: Path, session: str, script: list[str] | None, script_path: str | None):
        self.root = root
        self.session = session
        self.sdir = root / session
        self.run_dir = self.sdir / "run"
        self.log_path = self.run_dir / "run_log.jsonl"
        self.script = script
        self.script_path = script_path
        self.script_pos = 0
        self.n_prompts = 0
        self.lock_file = None

    # -- log
    def log(self, event: str, **fields) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        rec = {"time_utc": now_utc(), "time_local": now_local(), "monotonic_s": round(time.monotonic(), 3),
               "event": event, **fields}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # -- one sequencer per session
    def acquire(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        f = open(self.run_dir / ".run_arrangement.lock", "a")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            raise RunError(f"another {TOOL_NAME} is running on session {self.session}", EXIT_PRECOND) from None
        self.lock_file = f

    # -- prompts
    def _response(self) -> tuple[str, str]:
        if self.script is not None:
            if self.script_pos >= len(self.script):
                self.log("script_exhausted", script=self.script_path, consumed=self.script_pos)
                raise RunError(f"script {self.script_path} ran out of responses after {self.script_pos}",
                               EXIT_SCRIPT)
            r = self.script[self.script_pos]
            self.script_pos += 1
            print(f"> {r!r}", flush=True)
            return r, "script"
        try:
            return input("> "), "keyboard"
        except EOFError:
            print(flush=True)
            return "q", "keyboard-eof"

    def ask(self, text: str, slot: int | None = None, keys: tuple[str, ...] | None = ("",)) -> str:
        """Show a prompt, log it, return the response. `keys` = accepted responses besides 'q';
        None = a free-text answer (must be non-empty). 'q' always aborts (RunError 22)."""
        self.n_prompts += 1
        print(text, flush=True)
        self.log("prompt", slot=slot, text=text)
        t0 = time.monotonic()
        while True:
            r, source = self._response()
            key = r.strip()
            ok = key == "q" or (bool(key) if keys is None else key in keys)
            self.log("response", slot=slot, key=r, source=source, accepted=ok,
                     elapsed_s=round(time.monotonic() - t0, 3))
            if key == "q":
                raise RunError("operator abort (q)", EXIT_ABORT)
            if ok:
                return key
            want = "a non-empty answer" if keys is None else " / ".join(repr(k) if k else "Enter" for k in keys)
            print(f"  not understood; expected {want} or q", flush=True)

    def wait_until(self, until: float, slot: int | None) -> None:
        rem = until - time.monotonic()
        if rem <= 0:
            return
        self.log("lamp_settle_wait", slot=slot, wait_s=round(rem, 3))
        if self.script is not None:
            time.sleep(rem)
            return
        while (rem := until - time.monotonic()) > 0:
            print(f"\r  lamp settle: {math.ceil(rem):3d} s ", end="", flush=True)
            time.sleep(min(1.0, rem))
        print("\r  lamp settle: done     ", flush=True)


# ------------------------------------------------------------------------ preflights
def ntp_preflight(run: Run, timedatectl: str) -> dict:
    """Check the clock (DECISIONS 2026-10-12: a warning, not a gate). An unsynchronised or uncheckable
    clock is printed as a WARNING and recorded (run_log `ntp_preflight`, sitting.json / <ARR>.json
    `ntp` with ok false and a `warning`); the run goes on."""
    cmd = [timedatectl, "show", "-p", "NTPSynchronized", "--value"]
    res = {"stdout": None, "returncode": None, "ok": False}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=NTP_TIMEOUT_S)
        res.update(stdout=p.stdout.strip(), returncode=p.returncode,
                   ok=p.returncode == 0 and p.stdout.strip() == "yes")
        if not res["ok"] and p.stderr.strip():
            res["stderr"] = p.stderr.strip()[-500:]
    except FileNotFoundError:
        res["error"] = f"{timedatectl} not found"
    except subprocess.TimeoutExpired:
        res["error"] = f"timed out after {NTP_TIMEOUT_S} s"
    except OSError as e:
        res["error"] = str(e)
    if not res["ok"]:
        res["warning"] = ("NTP is not synchronised (timedatectl NTPSynchronized != yes): "
                          f"{res.get('error') or repr(res['stdout'])}, rc {res['returncode']}. "
                          "Timestamps may be off; continuing (see progress/SETUP.md).")
        print(f"WARNING: {res['warning']}", file=sys.stderr, flush=True)
    run.log("ntp_preflight", command=cmd, **res)
    return {k: res[k] for k in ("stdout", "returncode", "ok", "error", "warning") if k in res}


def check_forbidden(cmd: list[str]) -> None:
    bad = [a for a in cmd if a in FORBIDDEN]
    if bad:  # a programming error, never an operator choice
        raise RunError(f"internal error: refusing to emit forbidden pi_capture argument(s) {bad}", EXIT_CAPTURE)


def run_pi_capture(run: Run, cfg: dict, args: list[str], slot: int | None, purpose: str) -> dict:
    cmd = [cfg["python"], cfg["pi_capture"], *args]
    if cfg["rpicam"]:
        cmd += ["--rpicam", cfg["rpicam"]]
    check_forbidden(cmd)
    started = now_utc()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=PI_CAPTURE_TIMEOUT_S)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = None, (e.stdout or ""), f"pi_capture.py timed out after {PI_CAPTURE_TIMEOUT_S} s"
        out = out.decode() if isinstance(out, bytes) else out
    except OSError as e:
        rc, out, err = None, "", f"could not run pi_capture.py: {e}"
    finished = now_utc()
    warnings = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("WARNING:")]
    run.log("pi_capture", slot=slot, purpose=purpose, command=cmd, returncode=rc, stdout=out, stderr=err,
            warnings=warnings, started_utc=started, finished_utc=finished)
    return {"rc": rc, "stdout": out, "stderr": err, "warnings": warnings,
            "started_utc": started, "finished_utc": finished}


def capture_failed(res: dict, what: str) -> RunError:
    print(res["stderr"].rstrip(), file=sys.stderr, flush=True)
    return RunError(f"{what}: pi_capture.py exited {res['rc']} (stderr logged in run/run_log.jsonl)",
                    EXIT_CAPTURE)


def lock_config_info(path: Path) -> dict:
    cfg = read_json(path, "lock config")
    if cfg.get("schema") != LOCK_CONFIG_SCHEMA:
        raise RunError(f"lock config {path}: schema must be {LOCK_CONFIG_SCHEMA!r}, got {cfg.get('schema')!r}",
                       EXIT_PRECOND)
    return {"path": str(path), "sha256": sha256_file(path), "lock_version": cfg.get("lock_version")}


def applied_lock(sdir: Path, lighting: str) -> str:
    """The current lock id for `lighting`, after checking its calibration sidecar says the camera
    applied the requested controls (lock_check.ok is true)."""
    lock = read_json(sdir / "lock.json", "pi_capture lock file")
    entry = lock.get("locks", {}).get(lighting)
    if not entry:
        raise RunError(f"pi_capture reported success but lock.json has no lock for {lighting!r}", EXIT_PRECOND)
    lock_id = entry["lock_id"]
    side = read_json(sdir / "calibration" / f"{lock_id}.json", "calibration sidecar")
    lc = (side.get("lock") or {}).get("calibration", {}).get("lock_check")
    if not (isinstance(lc, dict) and lc.get("ok") is True):
        bad = [k for k, v in (lc or {}).items() if isinstance(v, dict) and not v.get("ok")]
        raise RunError(f"calibration {lock_id}: the camera did not apply the lock "
                       f"(lock_check.ok is {None if lc is None else lc.get('ok')}; {bad or 'no lock_check'}). "
                       "Do not capture in this session; check the camera and start a new session id.",
                       EXIT_PRECOND)
    return lock_id


def wait_session_idle(run: Run, slot: int | None, timeout_s: float = SESSION_IDLE_TIMEOUT_S) -> None:
    """Block until no pi_capture holds the session's flock (<session>/.session.lock), then release it
    at once. Used before --resume reads manifest.jsonl (a pi_capture child of a killed sequencer may
    still be saving its shot) and before retrying a shot pi_capture refused as 'session in use'.
    It only waits for pi_capture's own lock; it never shoots while holding it."""
    lock_path = run.sdir / ".session.lock"
    if not lock_path.exists():
        return
    t0 = time.monotonic()
    logged = False
    with open(lock_path, "a") as f:  # "a": never truncates pi_capture's file
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                break
            except BlockingIOError:
                if not logged:
                    print("waiting for a running pi_capture.py on this session to finish "
                          "(an orphan of an interrupted run?) ...", flush=True)
                    run.log("wait_session_lock", slot=slot)
                    logged = True
                if time.monotonic() - t0 > timeout_s:
                    raise RunError(f"another pi_capture.py still holds session {run.session}'s lock after "
                                   f"{timeout_s:.0f} s; find it (ps -ef | grep pi_capture) before going on",
                                   EXIT_PRECOND) from None
                time.sleep(0.2)
    if logged:
        run.log("session_lock_free", slot=slot, waited_s=round(time.monotonic() - t0, 3))


# --------------------------------------------------------------------------- sitting
def cmd_sitting(a, cfg: dict, run: Run) -> int:
    ntp = ntp_preflight(run, cfg["timedatectl"])
    sitting_path = run.run_dir / "sitting.json"
    if sitting_path.exists():
        raise RunError(f"session {run.session} already has a sitting ({sitting_path}); start a new session id",
                       EXIT_PRECOND)
    if (run.sdir / "lock.json").exists():
        raise RunError(f"session {run.session} already has a lock.json; this tool makes the session's locks "
                       "itself, so start a new session id", EXIT_PRECOND)
    lightings = parse_lightings(a.lightings)
    lock_config = Path(a.lock_config).expanduser().resolve()
    golden = Path(a.golden).expanduser().resolve()
    lc_info = lock_config_info(lock_config)
    if not golden.is_file():
        raise RunError(f"golden reference not found: {golden}", EXIT_PRECOND)
    golden_info = {"path": str(golden), "sha256": sha256_file(golden)}
    run.acquire()
    run.log("sitting_start", session=run.session, lightings=lightings, lock_config=lc_info, golden=golden_info,
            tool_version=TOOL_VERSION, note=a.note)
    primary = lightings[0]

    run.ask(f"Camera at P3 (floor mark), lamps set for {primary}. Enter = calibrate, q = abort")
    res = run_pi_capture(run, cfg, ["calibrate", "--root", str(run.root), "--session", run.session,
                                    "--lighting", primary, "--lock-config", str(lock_config)],
                         None, f"calibrate {primary} (lock config)")
    if res["rc"] != 0:
        raise capture_failed(res, f"calibration of {primary}")
    for w in res["warnings"]:
        print(w, flush=True)
    locks = {primary: applied_lock(run.sdir, primary)}

    # Pose check on the primary calibration still.
    lock_id = locks[primary]
    pose_out = run.run_dir / f"pose_check_{lock_id}.json"
    cmd = [cfg["python"], cfg["pose_check"], "--image", str(run.sdir / "calibration" / f"{lock_id}.jpg"),
           "--lock-config", str(lock_config), "--golden", str(golden), "--out", str(pose_out)]
    print("pose check against the golden P3 reference (may take a minute on the Pi) ...", flush=True)
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=POSE_TIMEOUT_S)
        prc, pout, perr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        prc, pout, perr = None, "", f"pose_check.py timed out after {POSE_TIMEOUT_S} s"
    except OSError as e:
        prc, pout, perr = None, "", f"could not run pose_check.py: {e}"
    pose = None
    try:
        pose = json.loads(pose_out.read_text())
    except (OSError, json.JSONDecodeError):
        try:
            pose = json.loads(pout)
        except json.JSONDecodeError:
            pose = None
    reasons = (pose or {}).get("reasons") or []
    run.log("pose_check", command=cmd, returncode=prc, stdout=pout, stderr=perr,
            wall_s=round(time.monotonic() - t0, 3), verdict=(pose or {}).get("verdict"), reasons=reasons,
            translation=(pose or {}).get("translation"), euclidean=(pose or {}).get("euclidean"))
    # Advisory since 1.3.0 (DECISIONS 2026-10-13: "roughly the same spot" is enough; the evidence step
    # registers every view through the shelf tags). The verdict is recorded; nothing stops the sitting.
    accepted_note = None
    pose_verdict = POSE_EXIT.get(prc, "error") if prc != 3 else "error"
    if prc == 0:
        print(f"pose OK: {_pose_summary(pose)}", flush=True)
    else:
        detail = '; '.join(reasons) or (_pose_summary(pose) if pose else perr.strip()[-300:])
        print(f"WARNING: pose check {pose_verdict.upper()} vs the golden ({detail}); advisory only, "
              "continuing", flush=True)
        run.log("pose_advisory", verdict=pose_verdict, returncode=prc)

    for lk in lightings[1:]:
        run.ask(f"Keep the camera at P3. Switch lamps to {lk}. Enter when switched")
        run.log("lamp_switch_calibration", lighting=lk)
        run.wait_until(time.monotonic() + cfg["lamp_settle_s"], None)
        res = run_pi_capture(run, cfg, ["calibrate", "--root", str(run.root), "--session", run.session,
                                        "--lighting", lk, "--copy-from", primary],
                             None, f"calibrate {lk} (copy from {primary})")
        if res["rc"] != 0:
            raise capture_failed(res, f"calibration of {lk}")
        for w in res["warnings"]:
            print(w, flush=True)
        locks[lk] = applied_lock(run.sdir, lk)

    sitting = {
        "schema": SITTING_SCHEMA, "tool": {"name": TOOL_NAME, "version": TOOL_VERSION,
                                           "python": platform.python_version()},
        "session": run.session, "created_utc": now_utc(), "created_local": now_local(),
        "hostname": socket.gethostname(), "ntp": ntp, "lock_config": lc_info, "golden": golden_info,
        "lightings": lightings, "locks": locks,
        "pose": {"file": f"run/{pose_out.name}", "verdict": pose_verdict, "exit_code": prc,
                 "advisory": True, "accepted_flag_note": accepted_note},
        "note": a.note}
    write_json_exclusive(sitting_path, sitting)
    run.log("sitting_done", locks=locks, pose_verdict=sitting["pose"]["verdict"])
    print(f"sitting ready: session {run.session}, locks {locks}, pose {sitting['pose']['verdict']}", flush=True)
    return EXIT_OK


def _pose_summary(pose: dict | None) -> str:
    if not pose:
        return "no pose JSON"
    t, e = pose.get("translation") or {}, pose.get("euclidean") or {}
    return (f"shift {t.get('shift_px')} px (dx {t.get('dx')}, dy {t.get('dy')}, cc {t.get('cc')}), "
            f"roll {e.get('rot_deg')} deg")


# ----------------------------------------------------------------------- arrangement
def load_card(path: str | None, arr: str) -> tuple[dict | None, dict]:
    """Optional layout card: a JSON object with "arrangement" and per-phase texts, either under
    "phases": {"reference": "...", ...} or as top-level phase keys."""
    if not path:
        return None, {}
    p = Path(path).expanduser().resolve()
    card = read_json(p, "card")
    if card.get("arrangement") != arr:
        raise RunError(f"card {p} is for arrangement {card.get('arrangement')!r}, not {arr!r}", EXIT_PRECOND)
    phases = card.get("phases") if isinstance(card.get("phases"), dict) else card
    texts = {ph: str(phases[ph]) for ph in PHASE_NAMES if isinstance(phases.get(ph), str) and phases[ph].strip()}
    return {"path": str(p), "sha256": sha256_file(p)}, texts


def slot_key(d: dict) -> tuple:
    return d["role"], d["condition"], d["position"], d["lighting"]


def cmd_arrangement(a, cfg: dict, run: Run) -> int:
    ntp = ntp_preflight(run, cfg["timedatectl"])
    sitting_path = run.run_dir / "sitting.json"
    if not sitting_path.exists():
        raise RunError(f"no sitting for session {run.session} ({sitting_path}); run `sitting` first",
                       EXIT_PRECOND)
    sitting = read_json(sitting_path, "sitting record")
    pose = sitting.get("pose") or {}
    if pose.get("verdict") != "ok":  # advisory since 1.3.0: recorded in sitting.json, never blocks
        print(f"note: sitting pose verdict {pose.get('verdict')!r} (advisory)", flush=True)
    lightings = list(sitting.get("lightings") or [])
    lock = read_json(run.sdir / "lock.json", "pi_capture lock file")
    for lighting in lightings:
        cur = (lock.get("locks", {}).get(lighting) or {}).get("lock_id")
        if cur is None or cur != sitting["locks"].get(lighting):
            raise RunError(f"lighting {lighting}: current lock {cur!r} != the sitting's "
                           f"{sitting['locks'].get(lighting)!r} (re-calibrated outside the sequencer?)",
                           EXIT_PRECOND)
    arr = check_arrangement(a.arrangement)
    stops = parse_stops(a.stops)
    card_info, card_text = load_card(a.card, arr)
    slots = make_plan(a.ordering, stops, lightings)
    run.acquire()
    wait_session_idle(run, None)  # an orphaned pi_capture of a killed run must finish before we read

    man_path = run.run_dir / f"{arr}.json"
    rows = [m for m in read_jsonl(run.sdir / "manifest.jsonl") if m.get("arrangement") == arr]
    if not a.resume:
        if rows or man_path.exists():
            raise RunError(f"arrangement {arr} already has {len(rows)} shot(s)"
                           f"{' and ' + str(man_path) if man_path.exists() else ''}; use --resume to continue it",
                           EXIT_PRECOND)
        man = {"schema": RUN_SCHEMA,
               "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "python": platform.python_version()},
               "session": run.session, "arrangement": arr, "ordering": a.ordering, "stops": stops,
               "lightings": lightings, "phases": list(PHASE_NAMES), "card": card_info,
               "sitting_sha256": sha256_file(sitting_path),
               "lock_config_sha256": (sitting.get("lock_config") or {}).get("sha256"), "ntp": ntp,
               "started_utc": now_utc(), "finished_utc": None, "status": "in_progress",
               "lamp_toggles": 0, "n_prompts": 0,
               "slots": [{**s, "takes": [], "selected_take": None} for s in slots]}
    else:
        if not man_path.exists():
            raise RunError(f"--resume: no {man_path} to resume", EXIT_PRECOND)
        man = read_json(man_path, "arrangement manifest")
        want = {"ordering": a.ordering, "stops": stops, "lightings": lightings}
        got = {k: man.get(k) for k in want}
        if got != want or [{k: s[k] for k in ("index", "phase", "role", "condition", "position", "lighting")}
                           for s in man.get("slots", [])] != slots:
            raise RunError(f"--resume: the saved plan {got} differs from the requested {want}", EXIT_PRECOND)
        if all(s.get("selected_take") is not None for s in man["slots"]):
            raise RunError(f"--resume: arrangement {arr} is already complete", EXIT_PRECOND)
        adopt_unrecorded(run, man, rows, arr)
        man["status"] = "in_progress"
        man.pop("abort", None)
        man.setdefault("resumed_utc", []).append(now_utc())
        man.setdefault("resumed_ntp", []).append(ntp)
        if card_info:
            man["card"] = card_info
    base_prompts = man.get("n_prompts", 0)

    def save():
        man["n_prompts"] = base_prompts + run.n_prompts
        write_json_atomic(man_path, man)

    save()
    run.log("arrangement_start", arrangement=arr, ordering=a.ordering, stops=stops, lightings=lightings,
            resume=bool(a.resume), n_slots=len(slots), planned_toggles=count_toggles(slots))
    try:
        sweep(run, cfg, man, card_text, save, resumed=bool(a.resume))
    except (RunError, KeyboardInterrupt) as e:
        code = e.code if isinstance(e, RunError) else EXIT_ABORT
        msg = str(e) if isinstance(e, RunError) else "operator abort (Ctrl-C)"
        man["status"] = "aborted"
        man["abort"] = {"time_utc": now_utc(), "exit_code": code, "message": msg}
        save()
        run.log("abort", arrangement=arr, exit_code=code, message=msg)
        raise RunError(msg, code) from None
    man["status"] = "complete"
    man["finished_utc"] = now_utc()
    save()
    n = sum(len(s["takes"]) for s in man["slots"])
    run.log("arrangement_done", arrangement=arr, lamp_toggles=man["lamp_toggles"], n_takes=n)
    print(f"arrangement {arr} complete: {len(man['slots'])} slots, {n} takes, {man['lamp_toggles']} lamp toggles; "
          f"manifest {man_path}", flush=True)
    return EXIT_OK


def adopt_unrecorded(run: Run, man: dict, rows: list[dict], arr: str) -> None:
    """Manifest rows of this arrangement that the run file does not know (the sequencer died
    between pi_capture saving a shot and recording it) become unaccepted takes of their slot.
    Rows that fit no slot of the plan are refused."""
    known = {t["stem"] for s in man["slots"] for t in s["takes"]}
    by_key = {slot_key(s): s for s in man["slots"]}
    for r in rows:
        if r["stem"] in known:
            continue
        s = by_key.get(slot_key(r)) if not r.get("episode") else None
        if s is None or s.get("selected_take") is not None:
            raise RunError(f"manifest.jsonl has shot {r['stem']} of {arr} that fits no open slot of this plan "
                           "(taken with bare pi_capture?); resolve it by hand", EXIT_PRECOND)
        s["takes"].append({"take": r["take"], "stem": r["stem"], "path": r["path"], "sha256": r["sha256"],
                           "lux": r.get("lux"), "lock_ok": r.get("lock_ok"), "light_ok": r.get("light_ok"),
                           "warnings": [], "note": "adopted on resume: saved by pi_capture, not recorded by "
                           "the sequencer", "accepted_despite": None, "pi_capture_exit": 0, "prompt_utc": None,
                           "shot_started_utc": None, "shot_finished_utc": r.get("time_utc")})
        s["takes"].sort(key=lambda t: t["take"])
        run.log("adopt_take", slot=s["index"], stem=r["stem"])


def sweep(run: Run, cfg: dict, man: dict, card_text: dict, save, resumed: bool) -> None:
    slots = man["slots"]
    n = len(slots)
    start = next(i for i, s in enumerate(slots) if s["selected_take"] is None)
    cur_lighting, prev_phase, prev_pos = None, None, None
    settle_until = 0.0
    for s in slots[start:]:
        i = s["index"]
        if s["phase"] != prev_phase:
            p_no = PHASE_NAMES.index(s["phase"]) + 1
            text = card_text.get(s["phase"], DEFAULT_PHASE_TEXT[s["phase"]])
            again = " (resumed; check the scene is still as it was)" if resumed and prev_phase is None else ""
            run.ask(f"PHASE {p_no}/4 {s['phase']}{again}: {text}. Enter when the scene is set, q = abort", i)
            run.log("phase_start", slot=i, phase=s["phase"])
            prev_phase = s["phase"]
        if s["lighting"] != cur_lighting:
            if cur_lighting is None:
                run.ask(f"Set lamps to {s['lighting']}. Enter when set", i)
                run.log("lamp_set", slot=i, lighting=s["lighting"])
            else:
                run.ask(f"Switch lamps to {s['lighting']}. Enter when switched", i)
                man["lamp_toggles"] += 1
                run.log("lamp_toggle", slot=i, lighting=s["lighting"], from_lighting=cur_lighting,
                        n_toggles=man["lamp_toggles"])
                save()
            settle_until = time.monotonic() + cfg["lamp_settle_s"]
            cur_lighting = s["lighting"]
        shoot_slot(run, cfg, man, s, n, prev_pos, settle_until, save)
        prev_pos = s["position"]


def shoot_slot(run: Run, cfg: dict, man: dict, s: dict, n: int, prev_pos: str | None,
               settle_until: float, save) -> None:
    arr, sdir, i = man["arrangement"], run.sdir, s["index"]

    def next_take() -> int:
        rows = [m for m in read_jsonl(sdir / "manifest.jsonl")
                if m.get("arrangement") == arr and not m.get("episode") and slot_key(m) == slot_key(s)]
        return max([t["take"] for t in s["takes"]] + [m["take"] for m in rows] + [0]) + 1

    take = next_take()
    note = None
    if s["takes"]:  # resumed slot whose earlier take(s) were never accepted
        note = f"retake: resumed after abort; take(s) {[t['take'] for t in s['takes']]} not accepted"
    while True:
        where = f"camera stays at {s['position']}" if prev_pos == s["position"] else f"move camera to {s['position']}"
        tag = f" (take {take})" if take > 1 else ""
        prompt_utc = now_utc()
        run.ask(f"[slot {i + 1}/{n}] {s['phase']} {s['position']} {s['lighting']}{tag}: {where}. "
                "Enter = shoot, q = abort", i)
        prev_pos = s["position"]
        run.wait_until(settle_until, i)
        while True:
            args = ["shoot", "--root", str(run.root), "--session", run.session, "--arrangement", arr,
                    "--position", s["position"], "--role", s["role"], "--condition", s["condition"],
                    "--lighting", s["lighting"], "--take", str(take)]
            if note:
                args += ["--note", note]
            res = run_pi_capture(run, cfg, args, i, f"shoot slot {i}")
            if res["rc"] == 0:
                break
            if res["rc"] == PC_EXIT_CAMERA:
                print(res["stderr"].rstrip(), file=sys.stderr, flush=True)
                run.ask("camera failure, nothing was saved. Enter = retry the same take, q = abort", i)
                continue
            if res["rc"] == PC_EXIT_PROTOCOL and PC_BUSY_TEXT in res["stderr"]:
                # session-lock contention only (any other exit-4 refusal stays fatal, never overridden)
                print(res["stderr"].rstrip(), file=sys.stderr, flush=True)
                run.ask("another pi_capture.py is using this session, nothing was saved. Enter = wait for it "
                        "and retry, q = abort", i)
                wait_session_idle(run, i)
                take = next_take()  # the other process may have saved a take of this slot meanwhile
                continue
            raise capture_failed(res, f"slot {i + 1}/{n} {s['phase']} {s['position']} {s['lighting']}")
        stem = stem_of(arr, s["position"], s["role"], s["condition"], s["lighting"], take)
        row = next((m for m in read_jsonl(sdir / "manifest.jsonl") if m.get("stem") == stem), None)
        if row is None:
            raise RunError(f"pi_capture exited 0 but manifest.jsonl has no row {stem}", EXIT_CAPTURE)
        s["takes"].append({"take": take, "stem": stem, "path": row["path"], "sha256": row["sha256"],
                           "lux": row.get("lux"), "lock_ok": row.get("lock_ok"), "light_ok": row.get("light_ok"),
                           "warnings": res["warnings"], "note": note, "accepted_despite": None, "pi_capture_exit": 0,
                           "prompt_utc": prompt_utc, "shot_started_utc": res["started_utc"],
                           "shot_finished_utc": res["finished_utc"]})
        save()
        run.log("shot", slot=i, stem=stem, take=take, sha256=row["sha256"], lux=row.get("lux"),
                lock_ok=row.get("lock_ok"), light_ok=row.get("light_ok"), warnings=res["warnings"])
        warn = ("  " + "  ".join(res["warnings"])) if res["warnings"] else ""
        failed = [c for c, k in (("lock", "lock_ok"), ("lighting", "light_ok")) if row.get(k) is False]
        if failed:
            # The camera did not apply the session lock (QC fail) or Lux left the guard (QC flag):
            # plain Enter does not accept such a shot (review finding RA-LOCK-FAIL-ACCEPTED).
            key = run.ask(f"saved {stem}{warn}. CHECK FAILED: {' and '.join(failed)} (QC will "
                          f"{'FAIL' if 'lock' in failed else 'flag'} this shot). r = retake, "
                          "a = accept anyway (asks for a note), q = abort", i, keys=("r", "a"))
            if key == "a":
                why = run.ask("Note: why accept this shot despite the failed check?", i, keys=None)
                s["takes"][-1]["accepted_despite"] = {"failed_checks": failed, "note": why[:NOTE_MAX]}
                run.log("accepted_despite", slot=i, take=take, stem=stem, failed_checks=failed, note=why)
                key = ""
        else:
            key = run.ask(f"saved {stem}{warn}. Enter = next, r = retake, q = abort", i, keys=("", "r"))
        if key == "":
            s["selected_take"] = take
            save()
            run.log("take_selected", slot=i, take=take, stem=stem)
            return
        reason = run.ask("Retake reason:", i, keys=None)
        note = ("retake: " + reason)[:NOTE_MAX]
        run.log("retake", slot=i, rejected_take=take, reason=reason)
        take += 1


# ------------------------------------------------------------------------------ plan
def cmd_plan(a) -> int:
    arr = check_arrangement(a.arrangement)
    stops, lightings = parse_stops(a.stops), parse_lightings(a.lightings)
    slots = make_plan(a.ordering, stops, lightings)
    print(json.dumps({"arrangement": arr, "ordering": a.ordering, "stops": stops, "lightings": lightings,
                      "phases": list(PHASE_NAMES), "n_slots": len(slots), "lamp_toggles": count_toggles(slots),
                      "slots": slots}, indent=2))
    return EXIT_OK


# ------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=None,
                        help=f"data root (default $CMPE249_DATA_ROOT or {DEFAULT_DATA_ROOT}), as pi_capture")
    common.add_argument("--session", required=True)
    common.add_argument("--pi-capture", default=str(HERE / "pi_capture.py"))
    common.add_argument("--pose-check", default=str(HERE / "pose_check.py"))
    common.add_argument("--python", default=sys.executable, help="interpreter for both subprocesses")
    common.add_argument("--rpicam", default=None, help=argparse.SUPPRESS)
    common.add_argument("--timedatectl", default=os.environ.get("TIMEDATECTL", "timedatectl"),
                        help=argparse.SUPPRESS)
    common.add_argument("--script", help="JSON list of responses, one per prompt (non-interactive)")
    common.add_argument("--fake-camera", action="store_true",
                        help="use capture/tests/fake_rpicam_still.py (refused on the real data root)")
    common.add_argument("--lamp-settle-s", type=float, default=15.0,
                        help="minimum seconds between a lamp change and the next shot (default 15)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sitting", parents=[common], help="NTP, lock-config calibration at P3, pose check, L2 lock")
    s.add_argument("--lock-config", default=os.environ.get("CMPE249_LOCK_CONFIG", DEFAULT_LOCK_CONFIG))
    s.add_argument("--golden", default=os.environ.get("CMPE249_GOLDEN", DEFAULT_GOLDEN))
    s.add_argument("--lightings", default=",".join(DEFAULT_LIGHTINGS))
    s.add_argument("--note")
    s.set_defaults(func=cmd_sitting)

    r = sub.add_parser("arrangement", parents=[common], help="guided capture of one arrangement")
    r.add_argument("--arrangement", required=True)
    r.add_argument("--ordering", required=True, choices=("A", "B"))
    r.add_argument("--stops", help=f"default {','.join(DEFAULT_STOPS)}")
    r.add_argument("--card")
    r.add_argument("--resume", action="store_true")
    r.set_defaults(func=cmd_arrangement)

    q = sub.add_parser("plan", help="print the slot order as JSON (no session, camera or NTP)")
    q.add_argument("--arrangement", required=True)
    q.add_argument("--ordering", required=True, choices=("A", "B"))
    q.add_argument("--stops")
    q.add_argument("--lightings", default=",".join(DEFAULT_LIGHTINGS))
    q.set_defaults(func=None)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "plan":
            return cmd_plan(a)
        root_given = a.root is not None
        root = Path(a.root or os.environ.get("CMPE249_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
        if not RE_SESSION.fullmatch(a.session or ""):
            raise RunError(f"invalid session id {a.session!r}: must match {RE_SESSION.pattern}", EXIT_USAGE)
        rpicam = a.rpicam
        if a.fake_camera:
            if rpicam:
                raise RunError("--fake-camera and --rpicam cannot be combined", EXIT_USAGE)
            real = Path(DEFAULT_DATA_ROOT).expanduser().resolve()
            if not root_given or root == real:
                raise RunError(f"--fake-camera needs an explicit --root other than the real data root {real}",
                               EXIT_USAGE)
            rpicam = str(HERE / "tests" / "fake_rpicam_still.py")
        script = None
        if a.script:
            try:
                script = json.loads(Path(a.script).read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise RunError(f"--script {a.script}: {e}", EXIT_USAGE) from None
            if not (isinstance(script, list) and all(isinstance(x, str) for x in script)):
                raise RunError("--script must be a JSON list of strings", EXIT_USAGE)
        if a.lamp_settle_s < 0:
            raise RunError("--lamp-settle-s must be >= 0", EXIT_USAGE)
        cfg = {"python": a.python, "pi_capture": a.pi_capture, "pose_check": a.pose_check, "rpicam": rpicam,
               "timedatectl": a.timedatectl, "lamp_settle_s": a.lamp_settle_s}
        run = Run(root, a.session, script, a.script)
        try:
            return a.func(a, cfg, run)
        except KeyboardInterrupt:
            e = RunError("operator abort (Ctrl-C)", EXIT_ABORT)
            run.log("abort", command=a.cmd, message=str(e))
            run.log("exit", exit_code=e.code, message=str(e))
            raise e from None
        except RunError as e:
            if e.code == EXIT_ABORT and a.cmd == "sitting":  # `arrangement` logs its own abort event
                run.log("abort", command=a.cmd, message=str(e))
            run.log("exit", exit_code=e.code, message=str(e))
            raise
    except RunError as e:
        print(f"{TOOL_NAME}: error: {e}", file=sys.stderr, flush=True)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
