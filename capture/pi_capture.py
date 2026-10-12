#!/usr/bin/env python3
"""Locked-settings still capture for the CMPE 249 active-verification study.

Runs ON the Raspberry Pi (Python 3.13, standard library only) and wraps `rpicam-still`.

    pi_capture.py calibrate --session S --lighting L1          # auto AE/AWB/AF once -> session lock
    pi_capture.py calibrate --session S --lighting L1 \\
        --lock-config configs/capture_lock.json                 # or: the study's fixed manual controls
    pi_capture.py shoot --session S --arrangement A01 --position P3 \\
        --role reference --condition visible --lighting L1      # capture with the locked controls
    pi_capture.py status --session S                            # what has been captured so far

Layout (one tree per session; rsync the session directory to the GB10 as-is):

    <root>/<session>/
        lock.json                     current lock per lighting label + full lock history
        session_log.jsonl             append-only event log (calibrations, shots, refusals)
        manifest.jsonl                one line per saved shot (used for the protocol checks)
        calibration/<lock_id>.jpg/.json
        <arrangement>/<role>/<position>/<stem>.jpg + <stem>.json   (JSON sidecar per image)

    stem = <arrangement>_<position>_<role>_<condition>_<lighting>[_<episode>]_t<take>

See capture/README.md for the protocol rules this tool enforces.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

TOOL_NAME = "pi_capture.py"
TOOL_VERSION = "0.3.0"  # 0.3.0: calibrate --lock-config (fixed manual controls from configs/capture_lock.json)
# 0.2.0: validates JPEG + metadata, checks FoV/size, Lux guard fixes, copy-from takes its own Lux shot
SCHEMA = "cmpe249.capture.v1"
LOCK_CONFIG_SCHEMA = "cmpe249.capture_lock.v1"

POSITIONS = ("P1", "P2", "P3", "P4", "P5")
ROLES = ("reference", "query")
CONDITIONS = ("visible", "occluded", "removed", "pilot")
REFERENCE_CONDITIONS = ("visible", "pilot")  # references show the unchanged scene
AF_RANGES = ("normal", "macro", "full")
# Only full-field-of-view sensor modes. The 1536x864 mode is a 3072x1728 centre crop
# (see `rpicam-hello --list-cameras`), so it would silently change the field of view.
SIZES = {"2304x1296": (2304, 1296), "4608x2592": (4608, 2592)}
# Image-processing controls pinned to their current rpicam-still defaults so that a future
# change of defaults cannot alter captures mid-study.
PINNED_ISP = ["--denoise", "cdn_hq", "--sharpness", "1", "--contrast", "1",
              "--saturation", "1", "--brightness", "0"]

RE_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,63}$")
RE_ARRANGEMENT = re.compile(r"^[A-Z][A-Z0-9]{1,15}$")
RE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,15}$")  # lighting / episode (no '_': it separates stem fields)

# Tolerances for checking that the camera actually applied the locked controls.
TOL_SHUTTER_REL = 0.005   # sensor line-time quantisation: 66642 requested -> 66616 reported (0.04%)
TOL_GAIN_REL = 0.02
TOL_AWB_ABS = 1e-3
TOL_LENS_ABS = 0.01
# Ambient-light guard: rpicam's Lux estimate vs the calibration shot.
LUX_WARN_RATIO = 1.25

EXIT_USAGE, EXIT_OVERWRITE, EXIT_PROTOCOL, EXIT_CAMERA, EXIT_NOLOCK = 2, 3, 4, 5, 6
DEFAULT_SIZE, DEFAULT_QUALITY, DEFAULT_ROTATION, DEFAULT_SHOOT_SETTLE_MS, DEFAULT_AF_RANGE = "2304x1296", 95, 0, 1500, "normal"
# JPEG start-of-frame markers (baseline, extended, progressive, lossless, arithmetic); not C4/C8/CC
JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


class CaptureError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
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
    """Create a new JSON file; never overwrite (raises FileExistsError)."""
    with open(path, "x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, obj) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check(pattern: re.Pattern, value: str, what: str, example: str) -> str:
    if value is None or not pattern.fullmatch(value):
        raise CaptureError(f"invalid {what} {value!r}: must match {pattern.pattern} (e.g. {example})")
    return value


def rpicam_version(rpicam: str) -> str:
    try:
        out = subprocess.run([rpicam, "--version"], capture_output=True, text=True, timeout=10)
        return " | ".join(line.strip() for line in (out.stdout + out.stderr).splitlines() if line.strip())
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {e}"


def session_dir(root: Path, session: str) -> Path:
    check(RE_SESSION, session, "session id", "2026-09-26-pilot")
    return root / session


@contextlib.contextmanager
def session_lock(sdir: Path):
    """Serialise tool invocations on one session (manifest checks + writes)."""
    sdir.mkdir(parents=True, exist_ok=True)
    with open(sdir / ".session.lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CaptureError(f"another {TOOL_NAME} process is using session {sdir.name}", EXIT_PROTOCOL)
        yield


def run_rpicam(cmd: list[str], log_path: Path) -> float:
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise CaptureError(f"camera command not found: {cmd[0]}", EXIT_CAMERA)
    except subprocess.TimeoutExpired:
        raise CaptureError("rpicam-still timed out after 60 s (camera busy or hung?)", EXIT_CAMERA)
    wall = time.monotonic() - t0
    log_path.write_text(proc.stderr + proc.stdout)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        raise CaptureError(f"rpicam-still failed (exit {proc.returncode})\n{tail}", EXIT_CAMERA)
    return wall


def base_cmd(rpicam: str, lock_capture: dict, settle_ms: int, img: Path, meta: Path) -> list[str]:
    w, h = lock_capture["width"], lock_capture["height"]
    cmd = [rpicam, "-n", "-t", str(settle_ms), "--width", str(w), "--height", str(h),
           "--mode", f"{w}:{h}:10:P", "-q", str(lock_capture["quality"]),
           "--rotation", str(lock_capture["rotation"]), *PINNED_ISP,
           "--metadata", str(meta), "--metadata-format", "json", "-o", str(img)]
    return cmd


def cleanup(*paths: Path) -> None:
    for p in paths:
        with contextlib.suppress(FileNotFoundError):
            p.unlink()


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%dT%H%M%S%f")  # microseconds: fast failures must not collide


def jpeg_size(path: Path) -> tuple[int, int]:
    """(width, height) from the JPEG frame header. Raises CaptureError(EXIT_CAMERA) if the file is
    missing, empty, lacks the start/end-of-image markers (partial write) or has no frame header."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(1 << 20)
            f.seek(max(0, size - 2))
            tail = f.read(2)
    except FileNotFoundError:
        raise CaptureError(f"rpicam-still reported success but wrote no image ({path.name})", EXIT_CAMERA)
    if size < 4 or head[:2] != b"\xff\xd8" or tail != b"\xff\xd9":
        raise CaptureError(f"image {path.name} is empty or truncated ({size} bytes; no JPEG start/end marker)",
                           EXIT_CAMERA)
    i = 2
    while i + 9 <= len(head):  # walk the marker segments (skips the EXIF thumbnail inside APP1)
        if head[i] != 0xFF:
            break
        m = head[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m == 0x01 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m in JPEG_SOF:
            return int.from_bytes(head[i + 7:i + 9], "big"), int.from_bytes(head[i + 5:i + 7], "big")
        if m == 0xDA:  # start of scan before any frame header
            break
        i += 2 + int.from_bytes(head[i + 2:i + 4], "big")
    raise CaptureError(f"image {path.name}: no JPEG frame header found", EXIT_CAMERA)


def read_metadata(path: Path) -> dict:
    try:
        md = json.loads(path.read_text())
    except FileNotFoundError:
        raise CaptureError(f"rpicam-still reported success but wrote no metadata ({path.name})", EXIT_CAMERA)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise CaptureError(f"metadata {path.name} is not valid JSON ({e})", EXIT_CAMERA)
    if not isinstance(md, dict) or not md:
        raise CaptureError(f"metadata {path.name} is empty or not a JSON object", EXIT_CAMERA)
    return md


def take_still(cmd: list[str], odir: Path, stem: str, img: Path) -> tuple[float, dict, tuple[int, int], Path]:
    """Run rpicam-still into hidden temporary files, validate the JPEG and the metadata, then
    hard-link the image to `img` (atomic; fails if `img` exists, so nothing is ever replaced).
    On any failure the temporaries are removed and the camera log is kept as
    <stem>.failed-<time>.log. Returns (wall_s, metadata, (width, height), camera log path)."""
    tmp_img, tmp_meta, tmp_log = (odir / f".{stem}.partial{ext}" for ext in (".jpg", ".meta.json", ".log"))
    assert cmd[cmd.index("-o") + 1] == str(tmp_img) and cmd[cmd.index("--metadata") + 1] == str(tmp_meta)
    cleanup(tmp_img, tmp_meta, tmp_log)  # leftovers of an interrupted run must never be taken for new output
    try:
        wall = run_rpicam(cmd, tmp_log)
        size = jpeg_size(tmp_img)
        md = read_metadata(tmp_meta)
        os.link(tmp_img, img)
    except BaseException as e:  # includes Ctrl-C: never leave a half-written temporary behind
        cleanup(tmp_img, tmp_meta)
        kept = ""
        if tmp_log.exists():
            failed = odir / f"{stem}.failed-{stamp()}.log"
            tmp_log.rename(failed)
            kept = f"\ncamera log kept as {failed}"
        if isinstance(e, FileExistsError):
            raise CaptureError(f"refusing to overwrite {img}", EXIT_OVERWRITE) from None
        if isinstance(e, CaptureError):
            raise CaptureError(f"{e}{kept}", e.code) from None
        if isinstance(e, OSError):
            raise CaptureError(f"could not save the capture: {e}{kept}", EXIT_CAMERA) from None
        raise
    cleanup(tmp_img, tmp_meta)
    log = odir / f"{stem}.log"
    if log.exists():
        log = odir / f"{stem}.{stamp()}.log"
    tmp_log.rename(log)
    return wall, md, size, log


def locked_cmd(rpicam: str, entry: dict, img: Path, meta: Path) -> list[str]:
    """rpicam-still command for a still with the lock's manual controls."""
    c = entry["controls"]
    return base_cmd(rpicam, entry["capture"], entry["capture"]["settle_ms"], img, meta) + [
        "--shutter", str(c["shutter_us"]), "--gain", f"{c['gain']:g}",
        "--awbgains", f"{c['awb_gains'][0]:g},{c['awb_gains'][1]:g}",
        "--lens-position", f"{c['lens_position']:g}"]


def scaler_crop_ints(v) -> list[int] | None:
    """ScalerCrop as four integers: rpicam reports "(0, 0)/4608x2592", other tools [0, 0, 4608, 2592]."""
    if v is None:
        return None
    return [int(x) for x in re.findall(r"-?\d+", str(v))]


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_lock_config(path_str: str) -> tuple[dict, dict]:
    """Read and validate configs/capture_lock.json. Returns (controls, record) where `record` is
    the lock entry's `lock_config` block. Any problem is a CaptureError (exit 2)."""
    path = Path(path_str).expanduser().resolve()
    try:
        raw = path.read_bytes()
        cfg = json.loads(raw)
    except OSError as e:
        raise CaptureError(f"--lock-config {path}: cannot read ({e.strerror or e})")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise CaptureError(f"--lock-config {path}: not valid JSON ({e})")
    bad = []
    if not isinstance(cfg, dict) or cfg.get("schema") != LOCK_CONFIG_SCHEMA:
        raise CaptureError(f"--lock-config {path}: schema must be {LOCK_CONFIG_SCHEMA!r}, got "
                           f"{cfg.get('schema') if isinstance(cfg, dict) else type(cfg).__name__!r}")
    lk = cfg.get("lock")
    if not isinstance(lk, dict):
        raise CaptureError(f"--lock-config {path}: missing 'lock' object")
    size = lk.get("size")
    size_key = (f"{size[0]}x{size[1]}" if isinstance(size, list) and len(size) == 2
                and all(isinstance(x, int) and not isinstance(x, bool) for x in size) else None)
    if size_key not in SIZES:
        bad.append(f"lock.size must be one of {[list(v) for v in SIZES.values()]}, got {size!r}")
    sh = lk.get("shutter_us")
    if not (isinstance(sh, int) and not isinstance(sh, bool) and sh > 0):
        bad.append(f"lock.shutter_us must be an integer > 0, got {sh!r}")
    g = lk.get("gain")
    if not (_is_num(g) and g > 0):
        bad.append(f"lock.gain must be a number > 0, got {g!r}")
    awb = lk.get("awb_gains")
    if not (isinstance(awb, list) and len(awb) == 2 and all(_is_num(x) and x > 0 for x in awb)):
        bad.append(f"lock.awb_gains must be two numbers > 0, got {awb!r}")
    lp = lk.get("lens_position")
    if not (_is_num(lp) and 0.0 <= lp <= 15.0):
        bad.append(f"lock.lens_position must be within 0..15 dioptres, got {lp!r}")
    crop = lk.get("scaler_crop")
    if crop is not None and len(scaler_crop_ints(crop)) != 4:
        bad.append(f"lock.scaler_crop must hold four integers, got {crop!r}")
    if bad:
        raise CaptureError(f"--lock-config {path}: " + "; ".join(bad))
    controls = {"shutter_us": sh, "gain": float(g), "awb_gains": [float(awb[0]), float(awb[1])],
                "lens_position": float(lp), "lens_source": "lock-config"}
    record = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "schema": cfg["schema"],
              "lock_version": cfg.get("lock_version"), "size_key": size_key,
              "scaler_crop": scaler_crop_ints(crop),
              "not_enforced": ["frame_duration_limits_us (rpicam-still command does not set it)",
                               "lens approach direction (rpicam-still sets the lens position directly)"]}
    return controls, record


def lighting_check(cal_lux, lux) -> tuple[dict, str | None]:
    """Compare rpicam's Lux estimate with the calibration shot's. Returns (record, warning or None).
    A missing Lux value is reported, not silently skipped."""
    light = {"calibration_lux": cal_lux, "lux": lux, "ratio": None, "ok": None, "threshold_ratio": LUX_WARN_RATIO}
    if cal_lux is None or cal_lux <= 0:
        return light, "no usable Lux in the calibration metadata; the ambient-light guard is off for this lock"
    if lux is None:
        return light, "no Lux in this shot's metadata; ambient light could not be checked"
    ratio = lux / cal_lux
    light.update(ratio=round(ratio, 4), ok=(1 / LUX_WARN_RATIO) <= ratio <= LUX_WARN_RATIO)
    if not light["ok"]:
        return light, (f"ambient light differs from calibration: Lux {lux:.2f} vs {cal_lux:.2f} "
                       f"(x{ratio:.2f}); check the lamps before continuing")
    return light, None


# ------------------------------------------------------------------------- calibrate
def cmd_calibrate(a) -> int:
    root = Path(a.root).expanduser()
    sdir = session_dir(root, a.session)
    lighting = check(RE_TOKEN, a.lighting, "lighting label", "L1")
    cfg_controls = cfg_record = None
    if a.lock_config:
        # the controls, size and rotation come from the file; refuse options that would otherwise be
        # silently ignored (same pattern as --copy-from)
        given = [f for f, v in (("--copy-from", a.copy_from), ("--lens-position", a.lens_position),
                                ("--af-range", a.af_range), ("--size", a.size), ("--rotation", a.rotation))
                 if v is not None]
        if given:
            raise CaptureError(f"--lock-config sets the controls, size and rotation from the file; "
                               f"{', '.join(given)} cannot be combined with it")
        cfg_controls, cfg_record = load_lock_config(a.lock_config)
    if a.copy_from:
        # a copied lock keeps the source lock's controls and capture settings; refuse options that
        # would otherwise be silently ignored
        given = [f for f, v in (("--size", a.size), ("--quality", a.quality), ("--rotation", a.rotation),
                                ("--lens-position", a.lens_position), ("--af-range", a.af_range),
                                ("--shoot-settle-ms", a.shoot_settle_ms)) if v is not None]
        if given:
            raise CaptureError(f"--copy-from reuses the source lock's controls and capture settings; "
                               f"{', '.join(given)} cannot be combined with it")
    size = cfg_record.pop("size_key") if cfg_record else (a.size or DEFAULT_SIZE)
    quality = a.quality if a.quality is not None else DEFAULT_QUALITY
    rotation = 0 if cfg_record else (a.rotation if a.rotation is not None else DEFAULT_ROTATION)
    shoot_settle_ms = a.shoot_settle_ms if a.shoot_settle_ms is not None else DEFAULT_SHOOT_SETTLE_MS
    af_range = a.af_range or DEFAULT_AF_RANGE
    if size not in SIZES:
        raise CaptureError(f"--size must be one of {list(SIZES)}")
    if a.settle_ms < 100 or shoot_settle_ms < 100:
        raise CaptureError("--settle-ms and --shoot-settle-ms must be at least 100 ms")
    if a.recalibrate and not (a.reason and a.reason.strip()):
        raise CaptureError("--recalibrate requires --reason (it is recorded in the lock history and session log)")
    if a.lens_position is not None and not (0.0 <= a.lens_position <= 15.0):
        raise CaptureError("--lens-position must be within 0..15 dioptres (imx708 tuning-file range)")

    with session_lock(sdir):
        lock_path = sdir / "lock.json"
        lock = json.loads(lock_path.read_text()) if lock_path.exists() else {
            "schema": SCHEMA, "session": a.session, "created_utc": now_utc(), "locks": {}, "history": []}
        prev = lock["locks"].get(lighting)
        if prev and not a.recalibrate:
            raise CaptureError(
                f"session {a.session} already has lock {prev['lock_id']} for lighting {lighting!r}. "
                "Settings are locked per session; re-calibrating changes every later capture. "
                "If you really must, pass --recalibrate --reason '...'", EXIT_PROTOCOL)
        if not prev and a.recalibrate:
            raise CaptureError(f"--recalibrate given but no existing lock for lighting {lighting!r}")
        version = (prev["version"] + 1) if prev else 1
        lock_id = f"{lighting}-v{version}"
        w, h = SIZES[size]
        capture = {"width": w, "height": h, "mode": f"{w}:{h}:10:P", "quality": quality,
                   "rotation": rotation, "settle_ms": shoot_settle_ms, "pinned_isp": PINNED_ISP}
        src = None
        if a.copy_from:
            src = lock["locks"].get(check(RE_TOKEN, a.copy_from, "lighting label", "L1"))
            if not src:
                raise CaptureError(f"--copy-from {a.copy_from!r}: no lock for that lighting in this session")
            capture = dict(src["capture"])

        cal_dir = sdir / "calibration"
        img, meta = cal_dir / f"{lock_id}.jpg", cal_dir / f"{lock_id}.json"
        if img.exists() or meta.exists():
            raise CaptureError(f"refusing to overwrite existing calibration file {img}", EXIT_OVERWRITE)
        tmp_img, tmp_meta = cal_dir / f".{lock_id}.partial.jpg", cal_dir / f".{lock_id}.partial.meta.json"
        if cfg_controls:
            # The study's fixed manual controls: the same locked-still path as --copy-from (manual
            # --shutter/--gain/--awbgains/--lens-position turn AE, AWB and AF off in rpicam-still).
            cmd = locked_cmd(a.rpicam, {"controls": cfg_controls, "capture": capture}, tmp_img, tmp_meta)
        elif src:
            # Same controls as the source lock, but a still of its own under the new lighting: its Lux
            # is the reference for this lighting's ambient-light guard (the source lock's Lux was
            # measured under the other lighting).
            cmd = locked_cmd(a.rpicam, {"controls": src["controls"], "capture": capture}, tmp_img, tmp_meta)
        else:
            cmd = base_cmd(a.rpicam, capture, a.settle_ms, tmp_img, tmp_meta)
            if a.lens_position is None:
                cmd += ["--autofocus-mode", "auto", "--autofocus-range", af_range, "--autofocus-on-capture"]
                lens_source = f"autofocus ({af_range} range, triggered on capture)"
            else:
                cmd += ["--lens-position", f"{a.lens_position:g}"]
                lens_source = "manual (--lens-position)"
        if a.dry_run:
            print(" ".join(cmd))
            return 0
        cal_dir.mkdir(parents=True, exist_ok=True)
        wall, md, (iw, ih), log = take_still(cmd, cal_dir, lock_id, img)
        if cfg_controls:
            controls = dict(cfg_controls)
            lc = lock_check(controls, md, None, [iw, ih], [capture["width"], capture["height"]])
            if cfg_record["scaler_crop"] is not None:
                rep_crop = scaler_crop_ints(md.get("ScalerCrop"))
                lc["scaler_crop"] = {"requested": cfg_record["scaler_crop"], "reported": md.get("ScalerCrop"),
                                     "ok": rep_crop == cfg_record["scaler_crop"]}
                lc["ok"] = all(v["ok"] for k, v in lc.items() if isinstance(v, dict))
        elif src:
            controls = dict(src["controls"])
            lc = lock_check(controls, md, src["calibration"].get("metadata"), [iw, ih], [capture["width"], capture["height"]])
        else:
            missing = [k for k in ("ExposureTime", "AnalogueGain", "ColourGains") if md.get(k) is None]
            lens = a.lens_position if a.lens_position is not None else md.get("LensPosition")
            if missing or lens is None:
                bad = img.with_name(f"{lock_id}.rejected-{stamp()}.jpg")
                os.rename(img, bad)  # keep the evidence, but not under the lock's name
                raise CaptureError(f"calibration metadata lacks {missing or ['LensPosition']} (image kept as "
                                   f"{bad.name}); pass --lens-position if LensPosition is missing", EXIT_CAMERA)
            gain = md["AnalogueGain"] * (md.get("DigitalGain") or 1.0)
            controls = {"shutter_us": int(md["ExposureTime"]), "gain": round(gain, 4),
                        "awb_gains": [round(md["ColourGains"][0], 4), round(md["ColourGains"][1], 4)],
                        "lens_position": round(float(lens), 4), "lens_source": lens_source}
            lc = None
        entry = {
            "lock_id": lock_id, "lighting": lighting, "version": version,
            "created_utc": now_utc(), "created_local": now_local(), "hostname": socket.gethostname(),
            "controls": controls, "capture": capture,
            "calibration": {"image": str(img.relative_to(sdir)), "sha256": sha256_file(img),
                            "image_size": [iw, ih], "command": cmd, "wall_s": round(wall, 3), "metadata": md,
                            "lux": md.get("Lux"), "af_state": md.get("AfState"),
                            "focus_fom": md.get("FocusFoM"), "lock_check": lc,
                            "exposure": ("manual controls from lock config" if cfg_controls
                                         else "locked controls of " + src["lock_id"] if src else "auto AE/AWB"),
                            "rpicam_version": rpicam_version(a.rpicam)},
            "copied_from": src["lock_id"] if src else None, "lock_config": cfg_record,
            "reason": a.reason, "note": a.note,
            "supersedes": prev["lock_id"] if prev else None, "tool_version": TOOL_VERSION}
        write_json_exclusive(meta, {"schema": SCHEMA, "kind": "calibration", "lock": entry})
        lock["locks"][lighting] = entry
        lock["history"].append(entry)
        lock["updated_utc"] = now_utc()
        write_json_atomic(lock_path, lock)
        append_jsonl(sdir / "session_log.jsonl", {
            "time_utc": now_utc(), "event": "recalibrate" if prev else "calibrate", "lock_id": lock_id,
            "lighting": lighting, "reason": a.reason, "note": a.note, "controls": entry["controls"],
            "supersedes": entry["supersedes"], "copied_from": entry["copied_from"]})
    c = entry["controls"]
    print(f"locked {lock_id}: shutter {c['shutter_us']} us, gain {c['gain']}, AWB {c['awb_gains']}, "
          f"lens {c['lens_position']} D ({c['lens_source']}); lux {entry['calibration'].get('lux')}"
          + (f"; controls copied from {entry['copied_from']}" if entry["copied_from"] else ""))
    print(f"lock file: {lock_path}")
    if not entry["copied_from"] and not cfg_controls and a.lens_position is None:
        print("NOTE: autofocus may lock onto the background rather than the workspace (see "
              "notes/2026-09-26-camera-check.md); prefer --lens-position 100/<distance_cm>.")
    _, warn = lighting_check(entry["calibration"].get("lux"), entry["calibration"].get("lux"))
    if warn:
        print("WARNING:", warn)
    lc = entry["calibration"].get("lock_check")
    if lc and not lc["ok"]:
        print(f"WARNING: camera did not apply the {'lock-config' if cfg_controls else 'copied'} controls "
              f"{[k for k, v in lc.items() if isinstance(v, dict) and not v['ok']]}; see the calibration sidecar")
    return 0


# ----------------------------------------------------------------------------- shoot
def protocol_checks(a, manifest: list[dict], lock_id: str) -> tuple[list[str], list[str]]:
    """Return (warnings, overrides). Raise CaptureError on a hard protocol violation."""
    warnings, overrides = [], []
    arr = [m for m in manifest if m["arrangement"] == a.arrangement]
    queries = [m for m in arr if m["role"] == "query"]
    refs = [m for m in arr if m["role"] == "reference"]
    if a.role == "reference":
        if a.condition not in REFERENCE_CONDITIONS:
            raise CaptureError(
                f"a reference shows the unchanged scene; --condition must be one of {REFERENCE_CONDITIONS}, "
                f"got {a.condition!r}", EXIT_PROTOCOL)
        if queries:
            msg = (f"arrangement {a.arrangement} already has {len(queries)} query shot(s): references must be "
                   "taken before the scene is changed")
            if not a.allow_late_reference:
                raise CaptureError(msg + " (override with --allow-late-reference --note '...')", EXIT_PROTOCOL)
            if not a.note:
                raise CaptureError("--allow-late-reference requires --note explaining why", EXIT_PROTOCOL)
            overrides.append("late reference: " + msg)
    else:
        ref_pos = {m["position"] for m in refs}
        if a.position not in ref_pos:
            raise CaptureError(
                f"no reference for arrangement {a.arrangement} at {a.position}; take references at every "
                "position before changing the scene", EXIT_PROTOCOL)
        missing = [p for p in POSITIONS if p not in ref_pos]
        if missing:
            msg = f"arrangement {a.arrangement} has no reference at {missing}"
            if a.condition != "pilot":
                raise CaptureError(msg + "; all five references come before any query", EXIT_PROTOCOL)
            warnings.append(msg + " (allowed for condition=pilot only)")
        if not any(m["position"] == a.position and m["lighting"] == a.lighting for m in refs):
            warnings.append(f"no reference at {a.position} under lighting {a.lighting!r} (cross-lighting query)")
    other_locks = sorted({m["lock_id"] for m in arr if m["lighting"] == a.lighting and m["lock_id"] != lock_id})
    if other_locks:
        msg = (f"arrangement {a.arrangement} was started under lock {other_locks} but the current lock for "
               f"{a.lighting!r} is {lock_id} (re-calibrated mid-arrangement)")
        if not a.accept_lock_change:
            raise CaptureError(msg + " (override with --accept-lock-change --note '...')", EXIT_PROTOCOL)
        if not a.note:
            raise CaptureError("--accept-lock-change requires --note explaining why", EXIT_PROTOCOL)
        overrides.append("lock change: " + msg)
    return warnings, overrides


def lock_check(controls: dict, md: dict, cal_md: dict | None = None,
               size: list[int] | None = None, want_size: list[int] | None = None) -> dict:
    """Requested vs reported controls. Also checks the field of view (ScalerCrop, against the
    calibration still's) and the JPEG's actual pixel size when those are given."""
    res = {}
    exp = md.get("ExposureTime")
    res["shutter_us"] = {"requested": controls["shutter_us"], "reported": exp,
                         "ok": exp is not None and abs(exp - controls["shutter_us"]) <= TOL_SHUTTER_REL * controls["shutter_us"]}
    ag = md.get("AnalogueGain")
    g = ag * (md.get("DigitalGain") or 1.0) if ag is not None else None
    res["gain"] = {"requested": controls["gain"], "reported": round(g, 4) if g is not None else None,
                   "ok": g is not None and abs(g - controls["gain"]) <= TOL_GAIN_REL * controls["gain"]}
    cg = md.get("ColourGains") or [None, None]
    res["awb_gains"] = {"requested": controls["awb_gains"], "reported": cg,
                        "ok": len(cg) == 2 and None not in cg
                        and all(abs(x - y) <= TOL_AWB_ABS for x, y in zip(cg, controls["awb_gains"]))}
    lp = md.get("LensPosition")
    res["lens_position"] = {"requested": controls["lens_position"], "reported": lp,
                            "ok": lp is not None and abs(lp - controls["lens_position"]) <= TOL_LENS_ABS}
    if cal_md and cal_md.get("ScalerCrop") is not None:  # sensor crop = field of view
        res["scaler_crop"] = {"requested": cal_md["ScalerCrop"], "reported": md.get("ScalerCrop"),
                              "ok": md.get("ScalerCrop") == cal_md["ScalerCrop"]}
    if want_size is not None:
        res["image_size"] = {"requested": want_size, "reported": size, "ok": size == want_size}
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res


def cmd_shoot(a) -> int:
    root = Path(a.root).expanduser()
    sdir = session_dir(root, a.session)
    check(RE_ARRANGEMENT, a.arrangement, "arrangement id", "A01 or PILOT00")
    check(RE_TOKEN, a.lighting, "lighting label", "L1")
    if a.episode is not None:
        check(RE_TOKEN, a.episode, "episode id", "E001")
    if a.take < 1 or a.take > 99:
        raise CaptureError("--take must be 1..99")
    if a.note is not None and len(a.note) > 500:
        raise CaptureError("--note is limited to 500 characters")
    lock_path = sdir / "lock.json"
    if not lock_path.exists():
        raise CaptureError(f"no lock file at {lock_path}; run `calibrate --session {a.session} "
                           f"--lighting {a.lighting}` first", EXIT_NOLOCK)

    with session_lock(sdir):
        lock_bytes = lock_path.read_bytes()
        lock = json.loads(lock_bytes)
        entry = lock["locks"].get(a.lighting)
        if not entry:
            raise CaptureError(f"session {a.session} has no lock for lighting {a.lighting!r} "
                               f"(calibrated: {sorted(lock['locks'])}); calibrate it first", EXIT_NOLOCK)
        manifest = read_jsonl(sdir / "manifest.jsonl")
        warnings, overrides = protocol_checks(a, manifest, entry["lock_id"])

        stem = "_".join([a.arrangement, a.position, a.role, a.condition, a.lighting]
                        + ([a.episode] if a.episode else []) + [f"t{a.take:02d}"])
        odir = sdir / a.arrangement / a.role / a.position
        img, side, log = odir / f"{stem}.jpg", odir / f"{stem}.json", odir / f"{stem}.log"
        for p in (img, side):
            if p.exists():
                raise CaptureError(
                    f"refusing to overwrite {p}. Captures are never replaced; if this is a deliberate retake, "
                    f"use --take {a.take + 1} and say why in --note", EXIT_OVERWRITE)
        tmp_img, tmp_meta = odir / f".{stem}.partial.jpg", odir / f".{stem}.partial.meta.json"
        c = entry["controls"]
        cmd = locked_cmd(a.rpicam, entry, tmp_img, tmp_meta)
        if a.dry_run:
            print(" ".join(cmd))
            for w in warnings + overrides:
                print("WARNING:", w)
            return 0
        odir.mkdir(parents=True, exist_ok=True)
        started_utc, started_local = now_utc(), now_local()
        wall, md, (iw, ih), log = take_still(cmd, odir, stem, img)
        finished_utc = now_utc()
        digest = sha256_file(img)
        lc = lock_check(c, md, entry["calibration"].get("metadata"), [iw, ih],
                        [entry["capture"]["width"], entry["capture"]["height"]])
        if not lc["ok"]:
            bad = [k for k, v in lc.items() if isinstance(v, dict) and not v["ok"]]
            warnings.append(f"camera did not apply locked controls {bad}; see lock_check in the sidecar")
        light, warn = lighting_check(entry["calibration"].get("lux"), md.get("Lux"))
        if warn:
            warnings.append(warn)
        labels = {"session": a.session, "arrangement": a.arrangement, "position": a.position,
                  "role": a.role, "condition": a.condition, "lighting": a.lighting,
                  "episode": a.episode, "take": a.take, "note": a.note}
        sidecar = {
            "schema": SCHEMA, "kind": "shot", "stem": stem, "labels": labels,
            "image": {"file": img.name, "sha256": digest, "bytes": img.stat().st_size,
                      "width": iw, "height": ih},  # read from the JPEG header, not assumed from the lock
            "capture": {"started_utc": started_utc, "started_local": started_local,
                        "finished_utc": finished_utc, "wall_s": round(wall, 3),
                        "hostname": socket.gethostname(), "command": cmd,
                        "command_note": "rpicam-still wrote to the hidden .partial files named in `command`; "
                                        "the image was validated and then hard-linked to image.file",
                        "camera_log": log.name},
            "lock": entry, "lock_file_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "rpicam_metadata": md, "lock_check": lc, "lighting_check": light,
            "protocol": {"warnings": warnings, "overrides": overrides},
            "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "python": platform.python_version(),
                     "rpicam_version": rpicam_version(a.rpicam)},
        }
        write_json_exclusive(side, sidecar)
        rec = {"time_utc": finished_utc, "stem": stem, "path": str(img.relative_to(sdir)),
               "arrangement": a.arrangement, "position": a.position, "role": a.role,
               "condition": a.condition, "lighting": a.lighting, "episode": a.episode, "take": a.take,
               "lock_id": entry["lock_id"], "sha256": digest, "lux": md.get("Lux"), "lock_ok": lc["ok"],
               "light_ok": light["ok"], "warnings": len(warnings), "overrides": len(overrides)}
        append_jsonl(sdir / "manifest.jsonl", rec)
        append_jsonl(sdir / "session_log.jsonl", {"event": "shoot", **rec})
    print(f"saved {img.relative_to(root)}  ({wall:.1f} s, lock {entry['lock_id']}, sha256 {digest[:12]})")
    for w in warnings + overrides:
        print("WARNING:", w)
    return 0


# ---------------------------------------------------------------------------- status
def cmd_status(a) -> int:
    root = Path(a.root).expanduser()
    sdir = session_dir(root, a.session)
    lock_path = sdir / "lock.json"
    if not sdir.exists():
        raise CaptureError(f"no session directory {sdir}")
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {"locks": {}, "history": []}
    manifest = read_jsonl(sdir / "manifest.jsonl")
    if a.json:
        print(json.dumps({"locks": lock["locks"], "n_history": len(lock["history"]), "manifest": manifest}, indent=2))
        return 0
    print(f"session {a.session}  ({sdir})")
    for lighting, e in sorted(lock["locks"].items()):
        c = e["controls"]
        print(f"  lock {e['lock_id']}: shutter {c['shutter_us']} us, gain {c['gain']}, AWB {c['awb_gains']}, "
              f"lens {c['lens_position']} D, {e['capture']['width']}x{e['capture']['height']}, "
              f"rotation {e['capture']['rotation']}, calibrated {e['created_local']}")
    arrs = sorted({m["arrangement"] for m in manifest})
    for arr in arrs:
        if a.arrangement and arr != a.arrangement:
            continue
        rows = [m for m in manifest if m["arrangement"] == arr]
        print(f"  {arr}: {len(rows)} shot(s)")
        for role in ROLES:
            for key in sorted({(m["condition"], m["lighting"], m["episode"]) for m in rows if m["role"] == role},
                              key=lambda k: tuple(x or "" for x in k)):
                cells = []
                for p in POSITIONS:
                    n = sum(1 for m in rows if m["role"] == role and m["position"] == p
                            and (m["condition"], m["lighting"], m["episode"]) == key)
                    cells.append(f"{p}:{n}" if n else f"{p}:-")
                ep = f" episode {key[2]}" if key[2] else ""
                print(f"    {role:9s} {key[0]:8s} {key[1]}{ep}  " + " ".join(cells))
    return 0


# ------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=os.environ.get("CMPE249_DATA_ROOT", "~/cmpe249/data"),
                        help="data root (default ~/cmpe249/data or $CMPE249_DATA_ROOT)")
    common.add_argument("--session", required=True, help="session id, e.g. 2026-09-26-pilot")
    common.add_argument("--rpicam", default=os.environ.get("RPICAM_STILL", "rpicam-still"), help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", parents=[common],
                       help="auto AE/AWB/AF once (or fixed controls from --lock-config) and write the session lock")
    c.add_argument("--lighting", required=True, help="lighting label this lock is for, e.g. L1")
    # defaults are applied in cmd_calibrate so that --copy-from can refuse options it would ignore
    c.add_argument("--size", choices=list(SIZES), help=f"default {DEFAULT_SIZE}")
    c.add_argument("--quality", type=int, choices=range(50, 101), metavar="50..100", help=f"default {DEFAULT_QUALITY}")
    c.add_argument("--rotation", type=int, choices=(0, 180),
                   help="180 if the camera is mounted upside down (only 0/180 are supported; default 0)")
    c.add_argument("--af-range", choices=AF_RANGES, help=f"default {DEFAULT_AF_RANGE}")
    c.add_argument("--lens-position", type=float,
                   help="set focus manually in dioptres (100/distance_cm) instead of autofocus")
    c.add_argument("--settle-ms", type=int, default=3000, help="AE/AWB convergence time for the calibration still")
    c.add_argument("--shoot-settle-ms", type=int,
                   help=f"settle time used by every locked shot (default {DEFAULT_SHOOT_SETTLE_MS})")
    c.add_argument("--copy-from", metavar="LIGHTING",
                   help="reuse the controls of another lighting's lock; a still is taken with those controls "
                        "under the new lighting as its Lux reference")
    c.add_argument("--lock-config", metavar="PATH",
                   help="take shutter, gain, AWB gains, lens position and size from a capture_lock.json "
                        "(schema cmpe249.capture_lock.v1) instead of running AE/AWB/AF")
    c.add_argument("--recalibrate", action="store_true", help="replace an existing lock (needs --reason)")
    c.add_argument("--reason")
    c.add_argument("--note")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("shoot", parents=[common], help="capture one still with the locked controls")
    s.add_argument("--arrangement", required=True, help="physical arrangement id, e.g. A01 or PILOT00")
    s.add_argument("--position", required=True, choices=POSITIONS)
    s.add_argument("--role", required=True, choices=ROLES)
    s.add_argument("--condition", required=True, choices=CONDITIONS)
    s.add_argument("--lighting", required=True, help="lighting label; must have a lock in this session")
    s.add_argument("--episode", help="optional episode id (letters, digits, '-')")
    s.add_argument("--take", type=int, default=1, help="retake number; a retake is never an overwrite")
    s.add_argument("--note")
    s.add_argument("--allow-late-reference", action="store_true")
    s.add_argument("--accept-lock-change", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_shoot)

    t = sub.add_parser("status", parents=[common], help="summarise locks and captures in a session")
    t.add_argument("--arrangement")
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CaptureError as e:
        print(f"{TOOL_NAME}: error: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
