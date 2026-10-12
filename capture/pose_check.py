#!/usr/bin/env python3
"""Camera pose check at the start of a sitting (CMPE 249 active verification, capture module 1).

Registers a fresh P3 still (normally the sitting's L1 calibration still) to the golden P3
reference (`ref_final.jpg`) on the static-background mask with `ecc.py`, and judges the pose:

    python3 pose_check.py --image <session>/calibration/L1-v1.jpg \\
        --lock-config ~/cmpe249/configs/capture_lock.json --golden ~/cmpe249/golden/ref_final.jpg \\
        [--out run/pose_check_L1-v1.json]

Contract: capture/MODULE1.md section 2. The JSON report (schema cmpe249.pose_check.v1) is printed
to stdout and, with --out, also written to that file (created exclusively).

Exit codes: 0 ok, 2 usage, 3 bad input (missing/unreadable file, sha256 or size mismatch, bad
config, --out exists), 10 flag, 11 refuse, 12 registration error or unreliable (cc < min_cc).
Thresholds come from capture_lock.json (pose_check.*, proposed values awaiting D17); a CLI value
overrides one and is recorded with source "cli".

Runs on the Pi (system python3, apt numpy/OpenCV) and on the GB10 (uv env).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import socket
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecc  # noqa: E402

TOOL_NAME = "pose_check.py"
TOOL_VERSION = "1.0.0"
SCHEMA = "cmpe249.pose_check.v1"
LOCK_SCHEMA = "cmpe249.capture_lock.v1"
POSITION = "P3"
MASK_NAME = "static_background"

EXIT_OK, EXIT_USAGE, EXIT_INPUT, EXIT_FLAG, EXIT_REFUSE, EXIT_ERROR = 0, 2, 3, 10, 11, 12
VERDICT_EXIT = {"ok": EXIT_OK, "flag": EXIT_FLAG, "refuse": EXIT_REFUSE, "error": EXIT_ERROR}

THRESHOLD_KEYS = ("flag_px", "refuse_px", "flag_roll_deg", "refuse_roll_deg", "min_cc")
PARAM_KEYS = ("mode", "sigma", "levels", "iters", "eps")


class PoseError(Exception):
    def __init__(self, msg: str, code: int = EXIT_INPUT):
        super().__init__(msg)
        self.code = code


# --------------------------------------------------------------------------- helpers
def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_lock_config() -> str:
    return os.environ.get("CMPE249_LOCK_CONFIG") or str(Path("~/cmpe249/configs/capture_lock.json").expanduser())


def default_golden() -> str:
    return os.environ.get("CMPE249_GOLDEN") or str(Path("~/cmpe249/golden/ref_final.jpg").expanduser())


def load_lock_config(path: Path) -> tuple[dict, str]:
    """(config, sha256 of the file's bytes). PoseError(3) if missing, not JSON or wrong schema."""
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise PoseError(f"cannot read lock config {path}: {e}")
    try:
        cfg = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise PoseError(f"lock config {path} is not valid JSON ({e})")
    if not isinstance(cfg, dict) or cfg.get("schema") != LOCK_SCHEMA:
        raise PoseError(f"lock config {path}: schema must be {LOCK_SCHEMA!r}, "
                        f"got {cfg.get('schema') if isinstance(cfg, dict) else type(cfg).__name__!r}")
    return cfg, hashlib.sha256(raw).hexdigest()


def resolve_settings(cfg: dict, cli: dict) -> tuple[dict, dict, dict, dict]:
    """(thresholds, threshold sources, registration params, param sources) from the config's
    pose_check.* and registration.*, with non-None CLI values overriding (source "cli")."""
    try:
        pc_cfg, reg_cfg = cfg["pose_check"], cfg["registration"]
        thr = {k: float(pc_cfg[k]) for k in THRESHOLD_KEYS}
        params = {"mode": str(reg_cfg["mode"]), "sigma": float(reg_cfg["sigma"]), "levels": int(reg_cfg["levels"]),
                  "iters": int(reg_cfg["iters"]), "eps": float(reg_cfg["eps"])}
    except (KeyError, TypeError, ValueError) as e:
        raise PoseError(f"lock config lacks a valid pose_check/registration entry: {e!r}")
    tsrc = {k: "config" for k in THRESHOLD_KEYS}
    psrc = {k: "config" for k in PARAM_KEYS}
    for k in THRESHOLD_KEYS:
        if cli.get(k) is not None:
            thr[k], tsrc[k] = float(cli[k]), "cli"
    if cli.get("mode") is not None:
        params["mode"], psrc["mode"] = cli["mode"], "cli"
    bad = [k for k in THRESHOLD_KEYS if not (math.isfinite(thr[k]) and thr[k] >= 0)]
    if params["mode"] not in ecc.MODES:
        bad.append("mode")
    if thr["flag_px"] > thr["refuse_px"]:
        bad.append("flag_px > refuse_px")
    if thr["flag_roll_deg"] > thr["refuse_roll_deg"]:
        bad.append("flag_roll_deg > refuse_roll_deg")
    if bad:
        from_cli = any(v == "cli" for v in tsrc.values()) or psrc["mode"] == "cli"
        raise PoseError(f"invalid thresholds/params: {bad} ({thr}, mode {params['mode']!r})",
                        EXIT_USAGE if from_cli else EXIT_INPUT)
    return thr, tsrc, params, psrc


def read_gray(path: Path, what: str, want_wh: tuple[int, int]) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise PoseError(f"{what} {path} does not exist or is not a file")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise PoseError(f"{what} {path} could not be decoded as an image")
    h, w = img.shape[:2]
    if (w, h) != tuple(want_wh):
        raise PoseError(f"{what} {path} is {w}x{h}, expected {want_wh[0]}x{want_wh[1]} (reference.size)")
    return img, sha256_file(path)


def fmt(x: float) -> str:
    return str(float(x))


def judge(t: "ecc.Shift | None", e: "ecc.Rigid | None", thr: dict) -> tuple[str, list[str], str | None, float | None]:
    """(verdict, reasons, shift_source, shift_px) following MODULE1 section 2.2 steps 7-9
    (comparisons strict), with one addition for a large roll:

    When the euclidean fit is reliable (e.cc >= min_cc) and its |roll| exceeds flag_roll_deg, a
    translation-only model does not describe the pose (measured on synthetic 2304x1296 frames: a
    1 deg roll gives an edges-mode translation cc of 0.19-0.23 and a meaningless 15 px shift), so
    the shift is judged on the euclidean centre displacement (shift_source "euclidean") and a
    failed or low-cc translation is not an error. Otherwise, as specified: either fit failing or
    t.cc < min_cc is "error", and the shift is judged on the translation fit."""
    roll_model = e is not None and e.cc >= thr["min_cc"] and abs(e.rot_deg) > thr["flag_roll_deg"]
    if roll_model:
        source, shift_px = "euclidean", math.hypot(e.dx, e.dy)
    elif t is None or e is None:
        return "error", [], None, None
    elif t.cc < thr["min_cc"]:
        return "error", [f"cc {t.cc:.4f} < min_cc {fmt(thr['min_cc'])} (registration unreliable)"], None, None
    else:
        source, shift_px = "translation", math.hypot(t.dx, t.dy)
    roll = abs(e.rot_deg)
    refuse, flag = [], []
    if shift_px > thr["refuse_px"]:
        refuse.append(f"shift {shift_px:.2f} px > refuse {fmt(thr['refuse_px'])} px")
    if roll > thr["refuse_roll_deg"]:
        refuse.append(f"roll {roll:.2f} deg > refuse {fmt(thr['refuse_roll_deg'])} deg")
    if shift_px > thr["flag_px"]:
        flag.append(f"shift {shift_px:.2f} px > flag {fmt(thr['flag_px'])} px")
    if roll > thr["flag_roll_deg"]:
        flag.append(f"roll {roll:.2f} deg > flag {fmt(thr['flag_roll_deg'])} deg")
    if source == "euclidean":
        flag.append("shift judged on the euclidean centre displacement (roll above the flag threshold)")
    verdict = "refuse" if refuse else "flag" if flag else "ok"
    return verdict, refuse + flag, source, shift_px


def pose_check(image: Path, lock_config: Path, golden: Path, cli: dict) -> dict:
    """Run the check and return the report dict (raises PoseError for bad input, code 3/2)."""
    cfg, cfg_sha = load_lock_config(lock_config)
    thr, tsrc, params, psrc = resolve_settings(cfg, cli)
    try:
        ref_cfg = cfg["reference"]
        want_sha = str(ref_cfg["sha256"])
        want_wh = (int(ref_cfg["size"][0]), int(ref_cfg["size"][1]))
        mask_spec = cfg["masks"][MASK_NAME]["by_position"][POSITION]
        want_mask_sha = cfg["masks"][f"sha256_{POSITION}"][MASK_NAME]
    except (KeyError, TypeError, IndexError, ValueError) as e:
        raise PoseError(f"lock config lacks reference/mask entries: {e!r}")
    if mask_spec is None:
        raise PoseError(f"lock config has no {MASK_NAME} mask for {POSITION}")
    if not golden.is_file():
        raise PoseError(f"golden {golden} does not exist or is not a file")
    golden_sha = sha256_file(golden)
    if golden_sha != want_sha:
        raise PoseError(f"golden {golden} sha256 {golden_sha[:12]}... does not match reference.sha256 "
                        f"{want_sha[:12]}... in {lock_config}")
    g, _ = read_gray(golden, "golden", want_wh)
    img, img_sha = read_gray(image, "image", want_wh)
    w, h = want_wh
    try:
        mask = ecc.regions_mask(mask_spec, w, h)
    except (ValueError, TypeError, KeyError) as e:
        raise PoseError(f"bad {MASK_NAME} mask spec for {POSITION}: {e}")
    mask_sha = ecc.mask_sha256(mask)
    if mask_sha != want_mask_sha:
        raise PoseError(f"{MASK_NAME} mask sha256 {mask_sha[:12]}... does not match "
                        f"masks.sha256_{POSITION}.{MASK_NAME} {str(want_mask_sha)[:12]}...")

    translation = euclidean = t = e = None
    errors = []
    kw = dict(sigma=params["sigma"], iters=params["iters"], eps=params["eps"], mode=params["mode"],
              levels=params["levels"])
    try:
        t = ecc.ecc_translation(g, img, mask=mask, **kw)
        translation = {"dx": round(t.dx, 3), "dy": round(t.dy, 3), "shift_px": round(math.hypot(t.dx, t.dy), 3),
                       "cc": round(t.cc, 4)}
    except ecc.EccError as ex:
        errors.append(f"translation: {ex}")
    try:
        e = ecc.ecc_euclidean(g, img, mask=mask, **kw)
        euclidean = {"dx": round(e.dx, 3), "dy": round(e.dy, 3), "rot_deg": round(e.rot_deg, 4),
                     "cc": round(e.cc, 4)}
    except ecc.EccError as ex:
        errors.append(f"euclidean: {ex}")
    verdict, reasons, shift_source, shift_px = judge(t, e, thr)
    if errors:
        reasons = ["fit failed: " + "; ".join(errors)] + reasons

    return {
        "schema": SCHEMA,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "ecc_version": ecc.ECC_VERSION,
                 "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                 "hostname": socket.gethostname()},
        "time_utc": now_utc(),
        "image": {"path": str(image.resolve()), "sha256": img_sha, "size": [w, h]},
        "golden": {"path": str(golden.resolve()), "sha256": golden_sha, "size": [w, h]},
        "lock_config": {"path": str(lock_config.resolve()), "sha256": cfg_sha,
                        "lock_version": cfg.get("lock_version")},
        "mask": {"name": MASK_NAME, "position": POSITION, "sha256": mask_sha, "usable_px": int(mask.sum())},
        "params": {**params, "source": psrc},
        "translation": translation,
        "euclidean": euclidean,
        "thresholds": {**thr, "source": tsrc},
        "judged": {"shift_source": shift_source,
                   "shift_px": round(shift_px, 3) if shift_px is not None else None},
        "verdict": verdict,
        "reasons": reasons,
        "exit_code": VERDICT_EXIT[verdict],
    }


# ------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    p.add_argument("--image", required=True, help="fresh P3 still (e.g. <session>/calibration/L1-v1.jpg)")
    p.add_argument("--lock-config", default=None,
                   help="capture_lock.json (default $CMPE249_LOCK_CONFIG or ~/cmpe249/configs/capture_lock.json)")
    p.add_argument("--golden", default=None,
                   help="golden P3 reference (default $CMPE249_GOLDEN or ~/cmpe249/golden/ref_final.jpg)")
    p.add_argument("--out", help="also write the JSON report here (must not exist)")
    p.add_argument("--flag-px", type=float, help="override pose_check.flag_px")
    p.add_argument("--refuse-px", type=float, help="override pose_check.refuse_px")
    p.add_argument("--flag-roll-deg", type=float, help="override pose_check.flag_roll_deg")
    p.add_argument("--refuse-roll-deg", type=float, help="override pose_check.refuse_roll_deg")
    p.add_argument("--min-cc", type=float, help="override pose_check.min_cc")
    p.add_argument("--mode", choices=ecc.MODES, help="override registration.mode")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    out = Path(a.out).expanduser() if a.out else None
    try:
        if out is not None and (out.exists() or out.is_symlink()):
            raise PoseError(f"--out {out} already exists; reports are never overwritten")
        if out is not None and not out.parent.is_dir():
            raise PoseError(f"--out directory {out.parent} does not exist")
        cli = {"flag_px": a.flag_px, "refuse_px": a.refuse_px, "flag_roll_deg": a.flag_roll_deg,
               "refuse_roll_deg": a.refuse_roll_deg, "min_cc": a.min_cc, "mode": a.mode}
        report = pose_check(Path(a.image).expanduser(), Path(a.lock_config or default_lock_config()).expanduser(),
                            Path(a.golden or default_golden()).expanduser(), cli)
        text = json.dumps(report, indent=2)
        if out is not None:
            try:
                with open(out, "x") as f:
                    f.write(text + "\n")
            except FileExistsError:
                raise PoseError(f"--out {out} already exists; reports are never overwritten")
            except OSError as e:
                raise PoseError(f"cannot write --out {out}: {e}")
    except PoseError as e:
        print(f"{TOOL_NAME}: error: {e}", file=sys.stderr)
        return e.code
    print(text)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
