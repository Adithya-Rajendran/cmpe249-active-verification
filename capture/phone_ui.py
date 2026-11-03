#!/usr/bin/env python3
"""Phone front end for the guided capture (capture/run_arrangement.py), served on the LAN.

Runs ON the Raspberry Pi beside run_arrangement.py (system Python 3.13: the standard library, plus
OpenCV and numpy for previews when they import). Open the printed URL on a phone on the same Wi-Fi.
The page shows run_arrangement's current prompt and offers only the answers that prompt accepts.

Nothing about the protocol lives here. run_arrangement.py still makes the slot plan, calls
pi_capture.py for every still, applies the lock and lighting checks, handles retakes, resume and
abort, and writes every log and manifest. This file only replaces the keyboard: it passes
run_arrangement.main a `run_factory` whose Run takes its answers from the phone (run_log responses
say source "phone"). The phone-side extras are advisory and never enter the capture record:
  - a preview of each saved still (and of the calibration still during a sitting);
  - "changed since the previous phase": the same stop and lighting one phase earlier (reference ->
    visible -> occluded -> removed), registered through the shelf tags (capture/tags.py; identity
    when it refuses) and differenced at 1/4 scale (grey, 5 px blur, mean matched, |diff| > 25).
    It exists to catch a step that was not done (pilot 2, 2026-10-20: a cup "removal" shot with
    the cup still in place). It is a hint, never a gate. On pilot 2's stills (data/raw/
    2026-10-20-pilot2/analysis/phone_ui_change_calibration.txt) two stills 5 s apart in the same
    light differed in 0.00 % of the frame, but removing a small cup changed only 0.04 % and removing
    a target already hidden behind its occluder 0.05 %; a target hidden at a stop shows no change
    there by design.

    python3 capture/phone_ui.py                  # real camera; data root as run_arrangement
    python3 capture/phone_ui.py --practice       # fake camera; data root ~/cmpe249/practice/data
    python3 capture/phone_ui.py --port 8249

Access: the printed URL carries a token (?t=...); the page then keeps it in a cookie. The token is
kept in <data root>/../.phone_ui_token (mode 600) so a phone bookmark survives restarts; --token
overrides it. Layout cards typed on the phone are written to <data root>/../cards/<session>/
<ARR>_<UTC>.json, outside pi_capture's session tree, and passed to run_arrangement with --card,
which records their path and sha256 in the arrangement manifest.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import functools
import json
import os
import queue
import re
import secrets
import socket
import sys
import threading
import time
import traceback
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_arrangement as ra  # noqa: E402  (same directory; there is no package on the Pi)

try:  # previews and the change hint only
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

TOOL_NAME = "phone_ui.py"
TOOL_VERSION = "0.1.0"
DEFAULT_PORT = 8249
PRACTICE_ROOT = "~/cmpe249/practice/data"
COOKIE = "cap_t"
PREV_PHASE = {"visible": "reference", "occluded": "visible", "removed": "occluded"}
CHANGE_SCALE, CHANGE_BLUR, CHANGE_DIFF = 4, 5, 25
NO_CHANGE_PCT = 0.01  # the page says "no change" below this (pilot 2: still pair 0.00 %, cup 0.04 %)
LOG_LINES, MAX_BODY, CARD_TEXT_MAX, CACHE_ITEMS = 400, 64 * 1024, 400, 64


def _eq(a: str, b: str) -> bool:
    return secrets.compare_digest(a.encode(), b.encode())


# ------------------------------------------------------------------ previews and the change hint
def preview_jpeg(path: Path, width: int) -> bytes | None:
    """`path` scaled to at most `width` px wide, as JPEG bytes; None if it does not decode (the
    fake camera writes a header-only JPEG) or OpenCV is missing."""
    if cv2 is None:
        return None
    flag = (cv2.IMREAD_REDUCED_COLOR_4 if width <= 576 else
            cv2.IMREAD_REDUCED_COLOR_2 if width <= 1152 else cv2.IMREAD_COLOR)
    img = cv2.imread(str(path), flag)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes() if ok else None


def change_share(a_path: Path, b_path: Path) -> dict:
    """How much of the frame changed from still a (the earlier phase) to still b (this shot), in %.

    b is registered onto a through the shelf tags (identity if tags.align refuses), both are reduced
    to 1/CHANGE_SCALE, grey and blurred, b's mean is matched to a's, and a pixel counts as changed
    when |a - b| > CHANGE_DIFF. Also returns a JPEG of b with the changed pixels in red."""
    import tags  # capture/tags.py (needs OpenCV)

    A, B = cv2.imread(str(a_path)), cv2.imread(str(b_path))
    if A is None or B is None:
        return {"status": "unavailable", "reason": "not a decodable JPEG"}
    al = tags.align(A, B)
    s = CHANGE_SCALE
    w, h = A.shape[1] // s, A.shape[0] // s
    sa = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    sb = cv2.resize(B, (B.shape[1] // s, B.shape[0] // s), interpolation=cv2.INTER_AREA)
    if al.ok:
        S = np.diag([1.0 / s, 1.0 / s, 1.0])
        Hs = S @ al.H @ np.linalg.inv(S)
        valid = cv2.warpPerspective(np.ones(sb.shape[:2], np.uint8), Hs, (w, h), flags=cv2.INTER_NEAREST) > 0
        sb = cv2.warpPerspective(sb, Hs, (w, h))
    else:
        if sb.shape[:2] != (h, w):
            sb = cv2.resize(sb, (w, h), interpolation=cv2.INTER_AREA)
        valid = np.ones((h, w), bool)
    if not valid.any():
        return {"status": "unavailable", "reason": "registration left no overlap"}
    k = (CHANGE_BLUR, CHANGE_BLUR)
    ga = cv2.GaussianBlur(cv2.cvtColor(sa, cv2.COLOR_BGR2GRAY), k, 0).astype(np.float32)
    gb = cv2.GaussianBlur(cv2.cvtColor(sb, cv2.COLOR_BGR2GRAY), k, 0).astype(np.float32)
    gb *= float(ga[valid].mean()) / max(float(gb[valid].mean()), 1e-6)
    changed = (np.abs(ga - gb) > CHANGE_DIFF) & valid
    pct = 100.0 * float(changed.sum()) / float(valid.sum())
    over = sb.astype(np.float32)
    over[~valid] *= 0.3
    over[changed] = 0.35 * over[changed] + 0.65 * np.array([40.0, 40.0, 230.0], np.float32)
    ok, buf = cv2.imencode(".jpg", over.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"status": "ready", "pct": round(pct, 3), "registered": bool(al.ok),
            "align_reason": None if al.ok else al.reason, "no_change": pct < NO_CHANGE_PCT,
            "overlay": buf.tobytes() if ok else None}


# ------------------------------------------------------------------------- shared state
class Tee:
    """Stands in for sys.stdout / sys.stderr while a job runs: passes text through, keeps lines."""

    def __init__(self, real, sink):
        self.real, self.sink, self.buf = real, sink, ""  # real None: keep lines only (--quiet)

    def write(self, s: str) -> int:
        try:
            if self.real is not None:
                self.real.write(s)
        except (OSError, ValueError):
            pass
        self.buf += s
        *lines, self.buf = self.buf.split("\n")
        for ln in lines:
            ln = ln.rsplit("\r", 1)[-1]
            if ln.strip():
                self.sink(ln)
        return len(s)

    def flush(self) -> None:
        try:
            if self.real is not None:
                self.real.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self.real if self.real is not None else sys.__stdout__, name)


class UI:
    """What the HTTP threads and the job thread share (one job at a time)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.answers: queue.Queue[str] = queue.Queue()
        self.prompt: dict | None = None
        self.seq = 0
        self.countdown_until: float | None = None
        self.lines: collections.deque[str] = collections.deque(maxlen=LOG_LINES)
        self.job: dict | None = None
        self.run: PhoneRun | None = None

    def add_line(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)

    def publish(self, text: str, keys, slot) -> None:
        with self.lock:
            self.seq += 1
            self.prompt = {"id": self.seq, "text": text, "keys": None if keys is None else list(keys),
                           "slot": slot}

    def clear(self) -> None:
        with self.lock:
            self.prompt = None

    def answer(self, prompt_id: int, value: str) -> tuple[int, str]:
        """One answer for the open prompt, accepted by run_arrangement.Run.ask's own rule."""
        with self.lock:
            p = self.prompt
            if p is None or prompt_id != p["id"]:
                return 409, "that prompt is no longer open (the page was behind; it refreshes)"
            key = value.strip()
            if not (key == "q" or (bool(key) if p["keys"] is None else key in p["keys"])):
                return 400, f"not an answer this prompt accepts: {value!r}"
            self.prompt = None
            self.answers.put(value)
            return 200, "ok"


class PhoneRun(ra.Run):
    """run_arrangement.Run whose answers come from the phone (run_log responses: source "phone")."""

    def __init__(self, root, session, script, script_path, ui: UI):
        super().__init__(root, session, script, script_path)
        self.ui = ui
        ui.run = self

    def ask(self, text: str, slot: int | None = None, keys=("",)) -> str:
        self.ui.publish(text, keys, slot)
        try:
            return super().ask(text, slot, keys)
        finally:
            self.ui.clear()

    def _response(self) -> tuple[str, str]:
        if self.script is not None:  # the server never passes --script
            return super()._response()
        return self.ui.answers.get(), "phone"

    def wait_until(self, until: float, slot: int | None) -> None:
        """Run.wait_until with the countdown on the phone instead of the terminal."""
        rem = until - time.monotonic()
        if rem <= 0:
            return
        self.log("lamp_settle_wait", slot=slot, wait_s=round(rem, 3))
        with self.ui.lock:
            self.ui.countdown_until = time.time() + rem
        try:
            time.sleep(rem)
        finally:
            with self.ui.lock:
                self.ui.countdown_until = None


# ------------------------------------------------------------------------------- the app
def _read_json(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


class App:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.ui = UI()
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.cache_lock = threading.Lock()
        self.computing: set = set()
        self.thread: threading.Thread | None = None

    # -- jobs
    def start(self, body: dict) -> tuple[int, dict]:
        cmd = body.get("cmd")
        session = str(body.get("session") or "").strip()
        if cmd not in ("sitting", "arrangement"):
            return 400, {"error": "cmd must be 'sitting' or 'arrangement'"}
        if not ra.RE_SESSION.fullmatch(session):
            return 400, {"error": f"session id must match {ra.RE_SESSION.pattern} (e.g. 2026-10-23-s1)"}
        root = self.cfg["root"]
        argv = [cmd, "--root", str(root), "--session", session, *self.cfg["passthrough"]]
        job = {"cmd": cmd, "session": session, "arrangement": None, "ordering": None, "resume": False,
               "card": None}
        texts: dict = {}
        if cmd == "sitting":
            argv += self.cfg["sitting_args"]
            note = str(body.get("note") or "").strip()
            if note:
                argv += ["--note", note[:ra.NOTE_MAX]]
        else:
            arr = str(body.get("arrangement") or "").strip().upper()
            ordering = str(body.get("ordering") or "")
            if not ra.RE_ARR.fullmatch(arr):
                return 400, {"error": f"arrangement id must match {ra.RE_ARR.pattern} (e.g. A01, PILOT03)"}
            if ordering not in ("A", "B"):
                return 400, {"error": "ordering must be A or B"}
            resume = bool(body.get("resume"))
            card = body.get("card") or {}
            if not isinstance(card, dict):
                return 400, {"error": "card must be an object of phase texts"}
            texts = {ph: str(card.get(ph) or "").strip()[:CARD_TEXT_MAX] for ph in ra.PHASE_NAMES}
            texts = {k: v for k, v in texts.items() if v}
            argv += ["--arrangement", arr, "--ordering", ordering] + (["--resume"] if resume else [])
            job.update(arrangement=arr, ordering=ordering, resume=resume)
            if resume and not texts:
                # run_arrangement shows a card's texts only when --card is passed again; on the phone a
                # resume keeps the card the arrangement started with (if it is still the same file).
                rec = (_read_json(root / session / "run" / f"{arr}.json") or {}).get("card")
                if isinstance(rec, dict) and rec.get("path"):
                    cp = Path(rec["path"])
                    if not cp.is_file() or ra.sha256_file(cp) != rec.get("sha256"):
                        return 409, {"error": f"this arrangement's layout card {cp} is missing or changed; "
                                              "type the card again to resume"}
                    argv += ["--card", str(cp)]
                    job["card"] = str(cp)
        with self.ui.lock:
            if self.ui.job and self.ui.job.get("status") == "running":
                return 409, {"error": "a sitting or arrangement is already running"}
            if texts:  # written only once nothing else is running
                cdir = self.cfg["cards_root"] / session
                cdir.mkdir(parents=True, exist_ok=True)
                cpath = cdir / f"{job['arrangement']}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.json"
                with open(cpath, "x") as f:
                    json.dump({"arrangement": job["arrangement"], "phases": texts, "written_utc": ra.now_utc(),
                               "tool": {"name": TOOL_NAME, "version": TOOL_VERSION}}, f, indent=2)
                    f.write("\n")
                argv += ["--card", str(cpath)]
                job["card"] = str(cpath)
            while not self.ui.answers.empty():  # nothing stale may answer the new job's first prompt
                self.ui.answers.get_nowait()
            self.ui.prompt, self.ui.countdown_until = None, None
            self.ui.lines.append(f"=== {cmd} {session}{' ' + job['arrangement'] if job['arrangement'] else ''} "
                                 f"({ra.now_local()[:19]}) ===")
            self.ui.job = {**job, "status": "running", "started_utc": ra.now_utc(), "finished_utc": None,
                           "exit_code": None, "message": None}
        self.thread = threading.Thread(target=self._run_job, args=(argv,), daemon=True, name="capture-job")
        self.thread.start()
        return 200, {"ok": True, "job": self.ui.job}

    def _run_job(self, argv: list[str]) -> None:
        ui = self.ui
        old = sys.stdout, sys.stderr
        echo = (None, None) if self.cfg["quiet"] else old
        sys.stdout, sys.stderr = Tee(echo[0], ui.add_line), Tee(echo[1], ui.add_line)
        code, crash = None, None
        try:
            code = ra.main(argv, run_factory=functools.partial(PhoneRun, ui=ui))
        except SystemExit as e:  # argparse usage error inside run_arrangement
            code = e.code if isinstance(e.code, int) else 2
        except BaseException:  # never let a job kill the server
            crash = traceback.format_exc()
            code = 1
        finally:
            sys.stdout, sys.stderr = old
        if crash:
            for ln in crash.splitlines():
                ui.add_line(ln)
        run = ui.run
        if run is not None and run.lock_file is not None:
            try:
                run.lock_file.close()  # releases run_arrangement's per-session flock
            except OSError:
                pass
        ui.run = None
        with ui.lock:
            ui.prompt, ui.countdown_until = None, None
            msg = None
            if code:
                msg = next((ln for ln in reversed(ui.lines) if "error:" in ln or "Traceback" in ln), None)
            status = "done" if code == 0 else "aborted" if code == ra.EXIT_ABORT else "failed"
            ui.job.update(status=status, exit_code=code, message=msg, finished_utc=ra.now_utc())

    # -- reading state
    def suggest_session(self) -> str:
        day = dt.date.today().isoformat()
        n = 1
        while (self.cfg["root"] / f"{day}-s{n}").exists():
            n += 1
        return f"{day}-s{n}"

    def state(self) -> dict:
        ui = self.ui
        with ui.lock:
            job = dict(ui.job) if ui.job else None
            prompt = dict(ui.prompt) if ui.prompt else None
            cd = None if ui.countdown_until is None else max(0.0, ui.countdown_until - time.time())
            lines = list(ui.lines)[-80:]
        st = {"tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "run_arrangement": ra.TOOL_VERSION},
              "practice": self.cfg["practice"], "root": str(self.cfg["root"]), "job": job, "prompt": prompt,
              "countdown_s": cd, "lines": lines, "suggested_session": self.suggest_session(),
              "progress": None, "shot": None}
        if job:
            st.update(self._progress(job, prompt))
        return st

    def _progress(self, job: dict, prompt: dict | None) -> dict:
        sdir = self.cfg["root"] / job["session"]
        out: dict = {"progress": None, "shot": None}
        if job["cmd"] == "sitting":
            stills = sorted((sdir / "calibration").glob("*.jpg"), key=lambda p: p.stat().st_mtime)
            if stills:
                out["shot"] = {"kind": "calibration", "path": str(stills[-1].relative_to(sdir)),
                               "stem": stills[-1].stem}
            return out
        man = _read_json(sdir / "run" / f"{job['arrangement']}.json")
        if not man or not isinstance(man.get("slots"), list):
            return out
        slots = man["slots"]
        n = len(slots)
        done = sum(1 for s in slots if s.get("selected_take") is not None)
        i = prompt.get("slot") if prompt else None
        cur = slots[i] if isinstance(i, int) and 0 <= i < n else next(
            (s for s in slots if s.get("selected_take") is None), None)
        out["progress"] = {"n": n, "done": done, "status": man.get("status"),
                           "lamp_toggles": man.get("lamp_toggles"),
                           **({k: cur[k] for k in ("index", "phase", "position", "lighting")} if cur else {})}
        if cur and cur.get("takes"):
            t, s = cur["takes"][-1], cur
        else:
            allt = [(t, s) for s in slots for t in s.get("takes", [])]
            if not allt:
                return out
            t, s = max(allt, key=lambda ts: ts[0].get("shot_finished_utc") or "")
        shot = {"kind": "slot", "path": t["path"], "stem": t["stem"], "take": t["take"], "lux": t.get("lux"),
                "lock_ok": t.get("lock_ok"), "light_ok": t.get("light_ok"), "warnings": t.get("warnings") or [],
                "accepted_despite": t.get("accepted_despite"), "phase": s["phase"], "position": s["position"],
                "lighting": s["lighting"], "selected": s.get("selected_take") == t["take"]}
        prev = PREV_PHASE.get(s["phase"])
        ps = next((x for x in slots if x["phase"] == prev and x["position"] == s["position"]
                   and x["lighting"] == s["lighting"]), None) if prev else None
        if ps and ps.get("takes"):
            pt = next((x for x in ps["takes"] if x["take"] == ps.get("selected_take")), ps["takes"][-1])
            shot["compare"] = {"phase": prev, "path": pt["path"], "stem": pt["stem"],
                               **self.change(sdir, pt["path"], t["path"])}
        out["shot"] = shot
        return out

    def session_info(self, session: str) -> dict:
        if not ra.RE_SESSION.fullmatch(session or ""):
            return {"valid": False}
        sdir = self.cfg["root"] / session
        info: dict = {"valid": True, "exists": sdir.is_dir(), "sitting": None, "arrangements": []}
        sit = _read_json(sdir / "run" / "sitting.json")
        if sit:
            info["sitting"] = {"locks": sit.get("locks"), "pose": (sit.get("pose") or {}).get("verdict"),
                               "created_local": sit.get("created_local")}
        for mp in sorted((sdir / "run").glob("*.json")):
            m = _read_json(mp)
            if m and m.get("schema") == ra.RUN_SCHEMA and isinstance(m.get("slots"), list):
                info["arrangements"].append({
                    "id": m.get("arrangement"), "status": m.get("status"), "ordering": m.get("ordering"),
                    "done": sum(1 for s in m["slots"] if s.get("selected_take") is not None),
                    "n": len(m["slots"])})
        return info

    # -- images
    def session_file(self, session: str, rel: str) -> Path | None:
        """A .jpg inside the session directory, or None (no traversal, no other files)."""
        if not ra.RE_SESSION.fullmatch(session or "") or not rel:
            return None
        sdir = (self.cfg["root"] / session).resolve()
        try:
            p = (sdir / rel).resolve()
        except (OSError, ValueError):
            return None
        if p.suffix.lower() not in (".jpg", ".jpeg") or not p.is_relative_to(sdir) or not p.is_file():
            return None
        return p

    def _cache_get(self, key):
        with self.cache_lock:
            v = self.cache.get(key)
            if v is not None:
                self.cache.move_to_end(key)
            return v

    def _cache_put(self, key, value) -> None:
        with self.cache_lock:
            self.cache[key] = value
            self.cache.move_to_end(key)
            while len(self.cache) > CACHE_ITEMS:
                self.cache.popitem(last=False)

    def preview(self, path: Path, width: int) -> bytes | None:
        key = ("preview", str(path), width)
        hit = self._cache_get(key)
        if hit is None:
            hit = preview_jpeg(path, width) or b""
            self._cache_put(key, hit)
        return hit or None

    def change(self, sdir: Path, a_rel: str, b_rel: str) -> dict:
        """The change hint for (a, b): cached, computed once in the background."""
        if cv2 is None:
            return {"status": "unavailable", "reason": "OpenCV is not available here"}
        key = ("change", str(sdir), a_rel, b_rel)
        hit = self._cache_get(key)
        if hit is not None:
            return {k: v for k, v in hit.items() if k != "overlay"}
        with self.cache_lock:
            if key in self.computing:
                return {"status": "computing"}
            self.computing.add(key)

        def work():
            try:
                res = change_share(sdir / a_rel, sdir / b_rel)
            except Exception as e:  # advisory only: report, never raise into the page
                res = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}
            self._cache_put(key, res)
            with self.cache_lock:
                self.computing.discard(key)

        threading.Thread(target=work, daemon=True, name="change-hint").start()
        return {"status": "computing"}

    def overlay(self, session: str, a_rel: str, b_rel: str) -> bytes | None:
        a, b = self.session_file(session, a_rel), self.session_file(session, b_rel)
        if a is None or b is None or cv2 is None:
            return None
        key = ("change", str(self.cfg["root"] / session), a_rel, b_rel)
        hit = self._cache_get(key)
        if hit is None:
            hit = change_share(a, b)
            self._cache_put(key, hit)
        return hit.get("overlay")


# ------------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = f"phone_ui/{TOOL_VERSION}"

    def log_message(self, fmt, *args) -> None:  # quiet; the job's own output is what matters
        pass

    @property
    def app(self) -> App:
        return self.server.app

    def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None,
              cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def _authed(self) -> bool:
        tok = self.app.cfg["token"]
        jar = cookies.SimpleCookie()
        try:
            jar.load(self.headers.get("Cookie", ""))
        except cookies.CookieError:
            pass
        c = jar.get(COOKIE)
        if c is not None and _eq(c.value, tok):
            return True
        h = self.headers.get("X-Token", "")
        return bool(h) and _eq(h, tok)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        tok = self.app.cfg["token"]
        if u.path == "/":
            if q.get("t") and _eq(q["t"], tok):
                return self._send(303, b"", "text/plain", {
                    "Location": "/",
                    "Set-Cookie": f"{COOKIE}={tok}; Path=/; Max-Age=2592000; SameSite=Strict; HttpOnly"})
            if not self._authed():
                return self._send(403, NEED_TOKEN_PAGE, "text/html; charset=utf-8")
            return self._send(200, self.app.cfg["page"], "text/html; charset=utf-8")
        if u.path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if not self._authed():
            return self._json(403, {"error": "missing or wrong access token: open the link printed on the Pi"})
        try:
            if u.path == "/api/state":
                return self._json(200, self.app.state())
            if u.path == "/api/session":
                return self._json(200, self.app.session_info(q.get("s", "")))
            if u.path == "/api/img":
                p = self.app.session_file(q.get("s", ""), q.get("p", ""))
                if p is None:
                    return self._json(404, {"error": "no such image"})
                try:
                    width = min(max(int(q.get("w", "720")), 64), 4608)
                except ValueError:
                    width = 720
                data = self.app.preview(p, width)
                if data is None and cv2 is None:
                    data = p.read_bytes()  # no OpenCV: the phone scales the full still
                if data is None:
                    return self._json(415, {"error": "no preview (not a decodable JPEG, e.g. fake camera)"})
                return self._send(200, data, "image/jpeg", cache="private, max-age=86400")
            if u.path == "/api/overlay":
                data = self.app.overlay(q.get("s", ""), q.get("a", ""), q.get("b", ""))
                if not data:
                    return self._json(404, {"error": "no overlay"})
                return self._send(200, data, "image/jpeg", cache="private, max-age=86400")
        except Exception as e:
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if not self._authed():
            return self._json(403, {"error": "missing or wrong access token: open the link printed on the Pi"})
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._json(415, {"error": "JSON only"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 <= n <= MAX_BODY:
            return self._json(413, {"error": "body too large"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json(400, {"error": "body is not JSON"})
        if not isinstance(body, dict):
            return self._json(400, {"error": "body must be a JSON object"})
        if u.path == "/api/start":
            code, obj = self.app.start(body)
            return self._json(code, obj)
        if u.path == "/api/answer":
            pid, val = body.get("prompt_id"), body.get("answer")
            if not isinstance(pid, int) or not isinstance(val, str):
                return self._json(400, {"error": "need prompt_id (int) and answer (string)"})
            code, msg = self.app.ui.answer(pid, val)
            return self._json(code, {"ok": code == 200, "message" if code == 200 else "error": msg})
        return self._json(404, {"error": "not found"})


# --------------------------------------------------------------------------------- page
NEED_TOKEN_PAGE = (b"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
                   b"<body style='font:18px system-ui;padding:24px'><h2>Rig capture</h2><p>Open the link "
                   b"printed on the Pi (it ends in <code>?t=...</code>). The page remembers it after that.</p>")

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Rig capture</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--fg:#16181d;--mute:#5f6672;--line:#dfe3e8;--acc:#1f6feb;--accfg:#fff;--ok:#1a7f37;--okbg:#dafbe1;--warn:#7d4e00;--warnbg:#fff3c4;--bad:#c62828;--badbg:#ffe5e3}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--mute:#8d96a0;--line:#2d333b;--acc:#2f81f7;--accfg:#fff;--ok:#3fb950;--okbg:#12261a;--warn:#e3b341;--warnbg:#2b2111;--bad:#ff7b72;--badbg:#3b1618}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-text-size-adjust:100%}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;align-items:center;gap:10px}
header h1{font-size:1.02rem;margin:0;flex:1;font-weight:650;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill{font-size:.8rem;padding:3px 10px;border-radius:999px;background:var(--line);white-space:nowrap}
.pill.run{background:var(--acc);color:var(--accfg)}.pill.ok{background:var(--okbg);color:var(--ok)}.pill.bad{background:var(--badbg);color:var(--bad)}.pill.warn{background:var(--warnbg);color:var(--warn)}
main{max-width:620px;margin:0 auto;padding:4px 16px 48px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 14px 16px;margin:12px 0}
.banner{border-radius:12px;padding:10px 12px;margin:12px 0;font-weight:600}
.banner.warn{background:var(--warnbg);color:var(--warn)}.banner.bad{background:var(--badbg);color:var(--bad)}
.meta{color:var(--mute);font-size:.88rem}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0 2px}
.bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .3s}
.head{font-size:.82rem;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);margin-top:8px}
.say{font-size:1.4rem;font-weight:650;line-height:1.3;margin:4px 0;overflow-wrap:anywhere}
.raw{color:var(--mute);font-size:.78rem;margin-top:8px;overflow-wrap:anywhere}
.count{font-size:2rem;font-weight:700;text-align:center;margin:10px 0}
button{font:inherit}
.btn{display:block;width:100%;min-height:58px;border-radius:12px;border:0;font-size:1.15rem;font-weight:650;margin-top:10px;background:var(--acc);color:var(--accfg);touch-action:manipulation;cursor:pointer}
.btn.sec{background:transparent;color:var(--fg);border:1.5px solid var(--line)}
.btn.warn{background:var(--warnbg);color:var(--warn)}
.btn.abort{background:transparent;color:var(--bad);border:1px solid var(--bad);min-height:44px;font-size:.95rem;margin-top:18px}
.btn:disabled{opacity:.45}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.chip{border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:999px;padding:6px 12px;font-size:.9rem;cursor:pointer}
input[type=text],textarea{width:100%;font:inherit;font-size:16px;padding:11px 12px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
textarea{min-height:60px;resize:vertical}
label{display:block;font-size:.88rem;color:var(--mute);margin:12px 0 4px}
label.ck{display:flex;gap:8px;align-items:center;color:var(--fg);font-size:1rem}
.seg{display:flex;gap:6px}.seg button{flex:1;min-height:46px;border-radius:10px;border:1.5px solid var(--line);background:transparent;color:var(--fg);font-weight:600;cursor:pointer}
.seg button.on{border-color:var(--acc);background:var(--acc);color:var(--accfg)}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}.imgs.one{grid-template-columns:1fr}
figure{margin:0}figure img{width:100%;border-radius:8px;display:block;background:var(--line);aspect-ratio:16/9;object-fit:cover}
figcaption{font-size:.78rem;color:var(--mute);margin-top:3px}
.checks{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.tag{font-size:.82rem;padding:3px 9px;border-radius:999px}.tag.ok{background:var(--okbg);color:var(--ok)}.tag.bad{background:var(--badbg);color:var(--bad)}
details summary{cursor:pointer;color:var(--mute);font-size:.9rem}
pre{white-space:pre-wrap;word-break:break-word;font-size:.74rem;line-height:1.35;max-height:320px;overflow:auto;background:var(--bg);padding:8px;border-radius:8px;margin:8px 0 0}
.hide{display:none!important}
.toast{position:fixed;left:16px;right:16px;bottom:18px;background:var(--fg);color:var(--bg);padding:12px 14px;border-radius:12px;font-weight:600;z-index:9;max-width:588px;margin:0 auto}
</style>
</head>
<body>
<header><h1 id="title">Rig capture</h1><span id="pill" class="pill">…</span></header>
<main>
<div id="conn" class="banner bad hide">Lost contact with the Pi. Retrying…</div>
<div id="practice" class="banner warn hide">PRACTICE: fake camera. Nothing here is study data.</div>

<section id="run" class="card hide">
  <div class="meta" id="prog"></div>
  <div class="bar"><i id="barfill"></i></div>
  <div class="head" id="head"></div>
  <div class="say" id="say"></div>
  <div class="count hide" id="count"></div>
  <div id="btns"></div>
  <div id="free" class="hide">
    <input type="text" id="freein" autocomplete="off" maxlength="480">
    <div class="chips" id="chips"></div>
    <button class="btn" id="freesend">Send</button>
  </div>
  <div class="raw" id="raw"></div>
  <button class="btn abort hide" id="abort">Stop here (can resume later)</button>
</section>

<section id="shot" class="card hide">
  <div class="meta" id="shottitle"></div>
  <div class="imgs" id="imgs"></div>
  <div class="checks" id="checks"></div>
  <div id="change" class="meta" style="margin-top:10px"></div>
</section>

<section id="done" class="card hide">
  <div class="say" id="donesay"></div>
  <div class="meta" id="donemsg"></div>
  <button class="btn hide" id="resume">Resume this arrangement</button>
  <button class="btn sec" id="again">Start a sitting or arrangement</button>
</section>

<section id="start" class="card hide">
  <div class="say" style="font-size:1.2rem">Start</div>
  <label for="sess">Session</label>
  <input type="text" id="sess" autocapitalize="off" autocomplete="off" spellcheck="false">
  <div class="meta" id="sessinfo"></div>
  <label>What</label>
  <div class="seg" id="cmdseg"><button type="button" data-v="sitting">Sitting</button><button type="button" data-v="arrangement">Arrangement</button></div>
  <div class="meta" id="cmdhelp" style="margin-top:6px"></div>
  <div id="arrfields">
    <label for="arr">Arrangement</label>
    <input type="text" id="arr" placeholder="A01 or PILOT03" autocapitalize="characters" autocomplete="off" spellcheck="false">
    <label>Order</label>
    <div class="seg" id="ordseg"><button type="button" data-v="A">A</button><button type="button" data-v="B">B</button></div>
    <div class="meta" id="ordhelp" style="margin-top:6px"></div>
    <label class="ck"><input type="checkbox" id="resumeck"> Resume a stopped arrangement</label>
    <details id="cardbox" style="margin-top:12px"><summary>Layout card: what to do in each phase (optional, saved with the arrangement)</summary>
      <label for="c_reference">Reference</label><textarea id="c_reference"></textarea>
      <label for="c_visible">Visible</label><textarea id="c_visible"></textarea>
      <label for="c_occluded">Occluded</label><textarea id="c_occluded"></textarea>
      <label for="c_removed">Removed</label><textarea id="c_removed"></textarea>
    </details>
  </div>
  <div id="sitfields">
    <label for="note">Note (optional)</label><input type="text" id="note" maxlength="480">
  </div>
  <button class="btn" id="go">Start</button>
</section>

<details class="card" id="logbox"><summary>Pi output</summary><pre id="log"></pre></details>
<div class="meta" id="foot"></div>
</main>
<script>
const D = __DEFAULTS__;
const $ = id => document.getElementById(id);
let st = null, lastPid = null, sending = false, formOpen = false, prefilled = false;
let formCmd = 'sitting', formOrd = 'A', overlayOn = false, shotKey = null, fails = 0, sessTimer = null, wake = null;

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cap = s => s ? s[0].toUpperCase() + s.slice(1) : s;
function toast(m){ const t = document.createElement('div'); t.className = 'toast'; t.textContent = m; document.body.appendChild(t); setTimeout(() => t.remove(), 3500); }

async function api(path, body){
  const opt = body === undefined ? {cache:'no-store'} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)};
  const r = await fetch(path, opt);
  let j = {}; try { j = await r.json(); } catch(e) {}
  if (!r.ok) { const e = new Error(j.error || ('HTTP ' + r.status)); e.status = r.status; throw e; }
  return j;
}
async function poll(){
  try { st = await api('/api/state'); fails = 0; $('conn').classList.add('hide'); render(); }
  catch(e){ if (e.status === 403) { location.reload(); return; } if (++fails >= 3) $('conn').classList.remove('hide'); }
}
setInterval(poll, 1000); poll();
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });

// --- prompt text -> what the page shows (the full prompt is always shown small underneath)
function view(t){
  let m = t.match(/^PHASE (\d)\/4 (\w+)( \(resumed[^)]*\))?: (.*?)\. Enter when the scene is set/s);
  if (m) return {head:`Phase ${m[1]} of 4 · ${m[2]}${m[3] ? ' · resumed: check the scene' : ''}`, say:cap(m[4]) + '.'};
  m = t.match(/^\[slot \d+\/\d+\] (\w+) (\w+) (\w+)( \(take (\d+)\))?: (.*?)\. Enter = shoot/s);
  if (m) return {head:`${m[1]} · ${m[2]} · ${m[3]}${m[5] ? ' · take ' + m[5] : ''}`, say:cap(m[6]) + '.'};
  if (/^saved /.test(t) && /CHECK FAILED/.test(t)) return {head:'Check failed', say:'The ' + (t.match(/CHECK FAILED: (.*?) \(/) || [,'camera'])[1] + ' check failed. Retake, or accept it with a note.'};
  if (/^saved /.test(t)) return {head:'Saved', say:'Keep this photo?'};
  if (/^Retake reason/.test(t)) return {head:'Retake', say:'Why retake? (one line)'};
  if (/^Note: why accept/.test(t)) return {head:'Accept anyway', say:'Why accept it despite the failed check?'};
  return {head:'', say:cap(t.replace(/[.,]?\s*(Enter( =| when)|r = retake|q = abort).*$/s, '').trim())};
}
function label(k, p){
  if (k === 'r') return '↺ Retake';
  if (k === 'a') return 'Accept anyway…';
  const m = p.text.match(/Enter (?:= |when )([^,.]+)/);
  const s = m ? m[1].trim() : 'OK';
  if (/^shoot$/i.test(s)) return '📷 Shoot';
  if (/^next$/i.test(s)) return '✓ Keep, next';
  if (/lamps/i.test(p.text) && /^(set|switched)$/i.test(s)) return 'Lamps are set';
  return cap(s);
}

function render(){
  const job = st.job, p = st.prompt, running = !!job && job.status === 'running';
  $('practice').classList.toggle('hide', !st.practice);
  const pill = $('pill');
  if (running) { pill.textContent = p ? 'Your turn' : (st.countdown_s != null ? 'Lamp settling' : 'Working…'); pill.className = 'pill ' + (p ? 'warn' : 'run'); }
  else if (job) { pill.textContent = ({done:'Done', aborted:'Stopped', failed:'Error'})[job.status] || job.status; pill.className = 'pill ' + (job.status === 'done' ? 'ok' : 'bad'); }
  else { pill.textContent = 'Idle'; pill.className = 'pill'; }
  $('title').textContent = job ? `${job.cmd === 'sitting' ? 'Sitting' : job.arrangement} · ${job.session}` : 'Rig capture';
  renderRun(job, p, running); renderShot(job); renderDone(job, running); renderStart(job, running);
  const log = $('log'), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
  log.textContent = (st.lines || []).join('\n'); if (atEnd) log.scrollTop = log.scrollHeight;
  $('foot').textContent = `${st.tool.name} ${st.tool.version} · run_arrangement ${st.tool.run_arrangement} · ${st.root}`;
  keepAwake(running);
}

function renderRun(job, p, running){
  $('run').classList.toggle('hide', !running);
  if (!running) { lastPid = null; return; }
  const g = st.progress;
  if (g) { $('prog').textContent = `Photo ${Math.min(g.done + 1, g.n)} of ${g.n}` + (g.phase ? ` · ${g.phase} · ${g.position} · ${g.lighting}` : ''); $('barfill').style.width = (100 * g.done / g.n) + '%'; }
  else { $('prog').textContent = job.cmd === 'sitting' ? 'Sitting: camera lock and pose check' : 'Starting…'; $('barfill').style.width = '0'; }
  const cd = st.countdown_s;
  $('count').classList.toggle('hide', cd == null);
  if (cd != null) $('count').textContent = `Lamp settling… ${Math.ceil(cd)} s`;
  if (!p) {
    $('head').textContent = ''; $('say').textContent = cd != null ? 'Wait for the lamps to settle.' : 'Working… (camera or a check is running)';
    $('btns').innerHTML = ''; $('free').classList.add('hide'); $('abort').classList.add('hide'); $('raw').textContent = ''; lastPid = null; return;
  }
  if (p.id === lastPid) return;  // keep buttons and typed text stable between polls
  lastPid = p.id;
  const v = view(p.text);
  $('head').textContent = v.head; $('say').textContent = v.say; $('raw').textContent = p.text;
  const btns = $('btns'); btns.innerHTML = '';
  if (p.keys === null) {
    $('free').classList.remove('hide'); $('freein').value = ''; $('freein').placeholder = 'Type a short note';
    const chips = /^Retake reason/.test(p.text) ? ['Hand or arm in frame', 'Blurry', 'Scene step not done', 'Camera not at the stop', 'Lamps wrong'] : [];
    $('chips').innerHTML = chips.map(c => `<button class="chip" type="button">${esc(c)}</button>`).join('');
    $('chips').querySelectorAll('.chip').forEach(b => b.onclick = () => { $('freein').value = b.textContent; });
  } else {
    $('free').classList.add('hide');
    p.keys.forEach((k, i) => {
      const b = document.createElement('button');
      b.className = 'btn' + (i ? ' sec' : '') + (k === 'a' ? ' warn' : ''); b.type = 'button';
      b.textContent = label(k, p); b.onclick = () => send(k); btns.appendChild(b);
    });
  }
  $('abort').classList.remove('hide');
}

async function send(v){
  if (sending || !st || !st.prompt) return;
  sending = true; document.querySelectorAll('#btns button, #freesend').forEach(b => b.disabled = true);
  try { await api('/api/answer', {prompt_id:st.prompt.id, answer:v}); if (navigator.vibrate) navigator.vibrate(25); }
  catch(e){ toast(e.message); lastPid = null; }
  finally { sending = false; document.querySelectorAll('#freesend').forEach(b => b.disabled = false); await poll(); }
}
$('freesend').onclick = () => { const v = $('freein').value.trim(); if (!v) { toast('Type a short note first'); return; } send(v); };
$('abort').onclick = () => { if (confirm('Stop here? Everything so far is kept, and the arrangement can be resumed.')) send('q'); };

function renderShot(job){
  const s = st.shot;
  $('shot').classList.toggle('hide', !s);
  if (!s) { shotKey = null; return; }
  const ss = encodeURIComponent(job.session), c = s.compare;
  const img = (path, w) => `/api/img?s=${ss}&p=${encodeURIComponent(path)}&w=${w}`;
  const fig = (path, src, cap) => `<figure><a href="${img(path, 2304)}" target="_blank" rel="noopener"><img src="${src}" alt="" onerror="this.style.visibility='hidden'"></a><figcaption>${cap}</figcaption></figure>`;
  const key = [s.path, c ? c.path + c.status : '', overlayOn].join('|');
  if (key !== shotKey) {
    shotKey = key;
    $('shottitle').textContent = s.kind === 'calibration' ? `Calibration still ${s.stem}` : `Last photo · ${s.phase} · ${s.position} · ${s.lighting} · take ${s.take}`;
    let h;
    if (c) {
      const now = overlayOn && c.status === 'ready' ? `/api/overlay?s=${ss}&a=${encodeURIComponent(c.path)}&b=${encodeURIComponent(s.path)}` : img(s.path, 576);
      h = fig(c.path, img(c.path, 576), `${esc(c.phase)} (before)`) + fig(s.path, now, overlayOn ? 'now · changes in red' : 'now');
    } else h = fig(s.path, img(s.path, 720), esc(s.stem));
    $('imgs').className = 'imgs' + (c ? '' : ' one'); $('imgs').innerHTML = h;
  }
  let ck = '';
  const tag = (ok, txt) => `<span class="tag ${ok ? 'ok' : 'bad'}">${ok ? '✓' : '✗'} ${esc(txt)}</span>`;
  if (s.lock_ok != null) ck += tag(s.lock_ok, 'lock');
  if (s.light_ok != null) ck += tag(s.light_ok, 'light' + (s.lux != null ? ` · ${Math.round(s.lux)} lux` : ''));
  (s.warnings || []).forEach(w => ck += `<span class="tag bad">${esc(w.replace(/^WARNING:\s*/, ''))}</span>`);
  if (s.accepted_despite) ck += '<span class="tag bad">accepted despite a failed check</span>';
  $('checks').innerHTML = ck;
  let ch = '';
  if (c && c.status === 'computing') ch = `Comparing with the ${esc(c.phase)} photo from this stop…`;
  else if (c && c.status === 'ready') {
    ch = c.no_change
      ? `<div class="banner warn" style="margin:0 0 6px">No change since the ${esc(c.phase)} photo from this stop. If you did the step, this stop may just not see it. Otherwise retake.</div>`
      : `${c.pct.toFixed(2)}% of the frame changed since the ${esc(c.phase)} photo from this stop. `;
    if (!c.registered) ch += '(Not aligned: the shelf tags were not found.) ';
    ch += `<button class="chip" id="ovbtn" type="button">${overlayOn ? 'Hide' : 'Show'} changes</button>`;
  } else if (c) ch = 'No comparison: ' + esc(c.reason || 'unavailable');
  if ($('change').innerHTML !== ch) $('change').innerHTML = ch;
  const ob = $('ovbtn'); if (ob) ob.onclick = () => { overlayOn = !overlayOn; render(); };
}

function renderDone(job, running){
  const show = !!job && !running && !formOpen;
  $('done').classList.toggle('hide', !show);
  if (!show) return;
  const what = job.cmd === 'sitting' ? 'Sitting' : 'Arrangement ' + job.arrangement;
  $('donesay').textContent = ({done:`${what} complete`, aborted:`${what} stopped`, failed:`${what} failed`})[job.status] || job.status;
  $('donemsg').textContent = job.message || (job.status === 'done' ? (job.cmd === 'sitting' ? 'Locks are set. Start an arrangement in the same session next.' : 'All photos are taken.') : (job.status === 'aborted' ? 'Everything so far is kept.' : ''));
  $('resume').classList.toggle('hide', !(job.cmd === 'arrangement' && job.status !== 'done'));
}
$('resume').onclick = () => start({cmd:'arrangement', session:st.job.session, arrangement:st.job.arrangement, ordering:st.job.ordering, resume:true});
$('again').onclick = () => { formOpen = true; prefilled = false; render(); };

function renderStart(job, running){
  const show = !running && (!job || formOpen);
  $('start').classList.toggle('hide', !show);
  if (!show) { prefilled = false; return; }
  if (!prefilled) {
    prefilled = true;
    $('sess').value = (job && job.session) || st.suggested_session;
    setCmd(job ? 'arrangement' : 'sitting'); setOrd((job && job.ordering) || 'A');
    if (job && job.arrangement) $('arr').value = job.arrangement;
    $('resumeck').checked = false; sessInfo();
  }
}
function setCmd(v){ formCmd = v; $('cmdseg').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === v)); $('arrfields').classList.toggle('hide', v !== 'arrangement'); $('sitfields').classList.toggle('hide', v !== 'sitting'); $('cmdhelp').textContent = D.help[v]; }
function setOrd(v){ formOrd = v; $('ordseg').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === v)); $('ordhelp').textContent = D.order[v]; }
$('cmdseg').onclick = e => { if (e.target.dataset.v) setCmd(e.target.dataset.v); };
$('ordseg').onclick = e => { if (e.target.dataset.v) setOrd(e.target.dataset.v); };
$('sess').oninput = () => { clearTimeout(sessTimer); sessTimer = setTimeout(sessInfo, 400); };
async function sessInfo(){
  const s = $('sess').value.trim();
  try {
    const i = await api('/api/session?s=' + encodeURIComponent(s));
    if (!i.valid) { $('sessinfo').textContent = 'Letters, digits, dots and dashes only.'; return; }
    let t = i.exists ? '' : 'New session. ';
    t += i.sitting ? `Sitting done (pose ${i.sitting.pose || '?'}). ` : (i.exists ? 'No sitting yet. ' : 'Start with a sitting. ');
    if (i.arrangements.length) t += 'Arrangements: ' + i.arrangements.map(a => `${a.id} ${a.status} ${a.done}/${a.n}`).join(', ') + '.';
    $('sessinfo').textContent = t;
  } catch(e) { $('sessinfo').textContent = ''; }
}
for (const ph of ['reference', 'visible', 'occluded', 'removed']) $('c_' + ph).placeholder = D.phases[ph];
async function start(body){
  try { await api('/api/start', body); formOpen = false; prefilled = false; lastPid = null; overlayOn = false; await poll(); }
  catch(e){ toast(e.message); }
}
$('go').onclick = () => {
  const body = {cmd:formCmd, session:$('sess').value.trim()};
  if (formCmd === 'sitting') body.note = $('note').value.trim();
  else {
    body.arrangement = $('arr').value.trim().toUpperCase(); body.ordering = formOrd; body.resume = $('resumeck').checked; body.card = {};
    for (const ph of ['reference', 'visible', 'occluded', 'removed']) body.card[ph] = $('c_' + ph).value.trim();
  }
  start(body);
};
async function keepAwake(on){
  try {
    if (on && !wake && 'wakeLock' in navigator && !document.hidden) { wake = await navigator.wakeLock.request('screen'); wake.addEventListener('release', () => { wake = null; }); }
    else if (!on && wake) { await wake.release(); wake = null; }
  } catch(e) {}
}
</script>
</body>
</html>
"""


def page_bytes() -> bytes:
    plans = {}
    for o in ("A", "B"):
        slots = ra.make_plan(o, list(ra.DEFAULT_STOPS), list(ra.DEFAULT_LIGHTINGS))
        moves = 1 + sum(1 for a, b in zip(slots, slots[1:]) if a["position"] != b["position"])
        plans[o] = (len(slots), ra.count_toggles(slots), moves)
    n = plans["A"][0]
    defaults = {
        "phases": ra.DEFAULT_PHASE_TEXT,
        "help": {
            "sitting": "Once per session, before any arrangement: camera at P3, lamps at L1. Calibrates the "
                       "camera lock, checks the pose against the golden view (advisory), then makes the L2 lock.",
            "arrangement": f"Guided sweep: reference, visible, occluded, removed x {', '.join(ra.DEFAULT_STOPS)} "
                           f"x {'/'.join(ra.DEFAULT_LIGHTINGS)} = {n} photos. Needs a sitting in this session."},
        "order": {o: f"{o}: {plans[o][0]} photos, {plans[o][2]} camera placements, {plans[o][1]} lamp switches"
                     + (" (both lightings at each stop)" if o == "A" else " (one lighting per sweep)")
                  for o in ("A", "B")},
    }
    return PAGE.replace("__DEFAULTS__", json.dumps(defaults)).encode()


# --------------------------------------------------------------------------------- main
def load_token(path: Path) -> str:
    try:
        t = path.read_text().strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{8,64}", t):
            return t
    except OSError:
        pass
    t = secrets.token_urlsafe(9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(t + "\n")
    return t


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--token", default=None, help="access token (default: kept in <data root>/../.phone_ui_token)")
    p.add_argument("--root", default=None,
                   help=f"data root (default $CMPE249_DATA_ROOT or {ra.DEFAULT_DATA_ROOT}; --practice: {PRACTICE_ROOT})")
    p.add_argument("--practice", action="store_true",
                   help="fake camera whose stills are copies of the golden reference, in the practice data root")
    p.add_argument("--fake-camera", action="store_true", help="passed to run_arrangement (needs a non-real --root)")
    p.add_argument("--quiet", action="store_true", help="keep the job's output on the page only (not on stdout)")
    # passed through to run_arrangement (both commands), only when given
    p.add_argument("--pi-capture")
    p.add_argument("--pose-check")
    p.add_argument("--python")
    p.add_argument("--lamp-settle-s")
    p.add_argument("--rpicam", help=argparse.SUPPRESS)
    p.add_argument("--timedatectl", help=argparse.SUPPRESS)
    # passed to `sitting` only
    p.add_argument("--lock-config")
    p.add_argument("--golden")
    p.add_argument("--lightings")
    return p


def make_server(argv: list[str] | None = None) -> tuple[App, ThreadingHTTPServer]:
    a = build_parser().parse_args(argv)
    fake = a.fake_camera or a.practice
    root = Path(a.root or (PRACTICE_ROOT if a.practice else os.environ.get("CMPE249_DATA_ROOT", ra.DEFAULT_DATA_ROOT)))
    root = root.expanduser().resolve()
    if fake and root == Path(ra.DEFAULT_DATA_ROOT).expanduser().resolve():
        raise ValueError(f"a fake camera cannot write to the real data root {root}; use --practice or another --root")
    if a.practice and not os.environ.get("FAKE_RPICAM_IMAGE"):
        golden = Path(a.golden or os.environ.get("CMPE249_GOLDEN", ra.DEFAULT_GOLDEN)).expanduser()
        if golden.is_file():  # real, decodable stills, so previews work while practising
            os.environ["FAKE_RPICAM_IMAGE"] = str(golden.resolve())
    root.mkdir(parents=True, exist_ok=True)
    passthrough = ["--fake-camera"] if fake else []
    for opt in ("pi_capture", "pose_check", "python", "lamp_settle_s", "rpicam", "timedatectl"):
        v = getattr(a, opt)
        if v is not None:
            passthrough += ["--" + opt.replace("_", "-"), str(v)]
    sitting_args = []
    for opt in ("lock_config", "golden", "lightings"):
        v = getattr(a, opt)
        if v is not None:
            sitting_args += ["--" + opt.replace("_", "-"), str(v)]
    token = a.token or load_token(root.parent / ".phone_ui_token")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", token):
        raise ValueError("--token must be 8-64 characters of A-Z a-z 0-9 _ -")
    cfg = {"root": root, "cards_root": root.parent / "cards", "practice": bool(a.practice or a.fake_camera),
           "token": token, "passthrough": passthrough, "sitting_args": sitting_args, "page": page_bytes(),
           "quiet": a.quiet}
    app = App(cfg)
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    httpd.daemon_threads = True
    httpd.app = app
    return app, httpd


def main(argv: list[str] | None = None) -> int:
    try:
        app, httpd = make_server(argv)
    except (OSError, ValueError) as e:
        print(f"{TOOL_NAME}: error: {e}", file=sys.stderr, flush=True)
        return 2
    port = httpd.server_address[1]
    tok = app.cfg["token"]
    print(f"{TOOL_NAME} {TOOL_VERSION} (run_arrangement {ra.TOOL_VERSION}); data root {app.cfg['root']}"
          f"{'  [PRACTICE: fake camera]' if app.cfg['practice'] else ''}", flush=True)
    print(f"open on the phone:  http://{lan_ip()}:{port}/?t={tok}", flush=True)
    print(f"              or:  http://{socket.gethostname()}.local:{port}/?t={tok}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
