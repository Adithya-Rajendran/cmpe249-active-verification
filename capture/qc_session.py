#!/usr/bin/env python3
"""Quality control for one capture session (CMPE 249 active verification, capture module 1).

Reads what `pi_capture.py` and `run_arrangement.py` wrote into one session directory and writes a
JSON report (schema cmpe249.qc_session.v1) and a short markdown summary:

    python3 qc_session.py --session-dir ~/cmpe249/data/S [--lock-config PATH] [--golden PATH]
                          [--arrangement A01 ...] [--out-dir DIR] [--no-images] [--lux-guard X]
                          [--shift-flag-px X] [--saturation-level N] [--saturation-flag-share X]

It computes file hashes, metadata checks, pixel statistics and ECC registration only. It runs no
detector or matcher and writes nothing about target state, so it may run on test sessions while
test capture is still going on. It never modifies the session: the only files it creates are
qc_<UTC>.json/.md under --out-dir (default <session-dir>/qc/), created exclusively.

Contract: capture/MODULE1.md section 4. Exit codes: 0 pass, 10 flags only, 11 at least one fail,
2 usage, 3 bad input (no manifest.jsonl, bad config, unwritable/existing output).

Runs on the GB10 (uv env) or the Pi (system python3, apt numpy/OpenCV).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import socket
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecc  # noqa: E402

TOOL_NAME = "qc_session.py"
TOOL_VERSION = "1.1.1"  # 1.1.1: ntp_not_synced is a flag, not a fail
SCHEMA = "cmpe249.qc_session.v1"
LOCK_SCHEMA = "cmpe249.capture_lock.v1"
EXIT_PASS, EXIT_USAGE, EXIT_INPUT, EXIT_FLAG, EXIT_FAIL = 0, 2, 3, 10, 11

POSITIONS = ("P1", "P2", "P3", "P4", "P5")
QUERY_CONDITIONS = ("visible", "occluded", "removed")
RE_STUDY = re.compile(r"^A\d{2}$")
RE_PILOT = re.compile(r"^PILOT\d{2}$")
LOCK_FILES = (".session.lock", "run/.run_arrangement.lock")  # flock files, excluded from the inventory
SEVERITIES = ("fail", "flag", "info")
MAX_MD_FLAGS = 30


class QcError(Exception):
    def __init__(self, msg: str, code: int = EXIT_INPUT):
        super().__init__(msg)
        self.code = code


# --------------------------------------------------------------------------- helpers
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict | None:
    """Parsed JSON object, or None if the file is missing or not valid JSON."""
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def read_manifest(path: Path) -> list[dict]:
    rows = []
    try:
        with open(path) as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise QcError(f"{path}:{n} is not valid JSON ({e})")
                    if not isinstance(row, dict) or "stem" not in row:
                        raise QcError(f"{path}:{n} is not a pi_capture manifest row")
                    rows.append(row)
    except OSError as e:
        raise QcError(f"cannot read {path}: {e}")
    return rows


def default_lock_config() -> str:
    return os.environ.get("CMPE249_LOCK_CONFIG") or str(Path("~/cmpe249/configs/capture_lock.json").expanduser())


def default_golden() -> str:
    return os.environ.get("CMPE249_GOLDEN") or str(Path("~/cmpe249/golden/ref_final.jpg").expanduser())


def load_lock_config(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        cfg = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise QcError(f"cannot load lock config {path}: {e}")
    if not isinstance(cfg, dict) or cfg.get("schema") != LOCK_SCHEMA:
        raise QcError(f"lock config {path}: schema must be {LOCK_SCHEMA!r}")
    for key in ("lock", "masks", "registration", "qc", "reference"):
        if not isinstance(cfg.get(key), dict):
            raise QcError(f"lock config {path} has no {key!r} section")
    return cfg, hashlib.sha256(raw).hexdigest()


def crop4(v) -> list[int] | None:
    """ScalerCrop normalised to its four integers: "(0, 0)/4608x2592" and [0, 0, 4608, 2592] both
    give [0, 0, 4608, 2592] (MODULE1 section 1.2)."""
    if v is None:
        return None
    nums = [int(x) for x in re.findall(r"-?\d+", str(v))]
    return nums if len(nums) == 4 else None


def stats(vals: list[float]) -> dict | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"median": round(statistics.median(vals), 6), "min": round(min(vals), 6), "max": round(max(vals), 6)}


class Flags:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, code: str, severity: str, detail: str, row: dict | None = None, **kw) -> None:
        assert severity in SEVERITIES
        f = {"code": code, "severity": severity}
        for k in ("arrangement", "condition", "lighting", "position", "stem"):
            f[k] = kw.get(k, (row or {}).get(k))
        f["detail"] = detail
        self.items.append(f)

    def count(self, severity: str) -> int:
        return sum(1 for f in self.items if f["severity"] == severity)


# ------------------------------------------------------------------------ parameters
def resolve_params(cfg: dict, a) -> tuple[dict, dict]:
    q, reg = cfg["qc"], cfg["registration"]
    try:
        params = {"lux_guard": float(q["lux_ratio_guard"]), "shift_flag_px": float(q["shift_flag_px"]),
                  "saturation_level": int(q["saturation_level"]),
                  "saturation_flag_share": float(q["saturation_flag_share"]),
                  "registration": {"mode": str(reg["mode"]), "sigma": float(reg["sigma"]),
                                   "levels": int(reg["levels"]), "iters": int(reg["iters"]),
                                   "eps": float(reg["eps"])}}
    except (KeyError, TypeError, ValueError) as e:
        raise QcError(f"lock config qc/registration section is incomplete: {e!r}")
    sources = {k: "config" for k in ("lux_guard", "shift_flag_px", "saturation_level", "saturation_flag_share")}
    for k, v in (("lux_guard", a.lux_guard), ("shift_flag_px", a.shift_flag_px),
                 ("saturation_level", a.saturation_level), ("saturation_flag_share", a.saturation_flag_share)):
        if v is not None:
            params[k], sources[k] = v, "cli"
    if params["lux_guard"] < 1.0 or params["shift_flag_px"] < 0 or not (0 < params["saturation_level"] <= 255) \
            or not (0 <= params["saturation_flag_share"] <= 1) or params["registration"]["mode"] not in ecc.MODES:
        bad_cli = any(s == "cli" for s in sources.values())
        raise QcError(f"invalid QC parameters {params}", EXIT_USAGE if bad_cli else EXIT_INPUT)
    params["images"] = not a.no_images
    params["sources"] = sources
    return params, sources


class MaskBank:
    """Rasterised masks per (name, position) from capture_lock.json; sha256-checked where the
    config gives a hash (masks.sha256_<P>.<name>); a mismatch is a bad config (exit 3)."""

    def __init__(self, cfg: dict, w: int, h: int):
        self.cfg, self.w, self.h, self.cache = cfg["masks"], w, h, {}

    def get(self, name: str, pos: str) -> np.ndarray | None:
        key = (name, pos)
        if key not in self.cache:
            try:
                spec = self.cfg[name]["by_position"].get(pos)
            except (KeyError, TypeError, AttributeError):
                raise QcError(f"lock config has no masks.{name}.by_position")
            if spec is None:
                self.cache[key] = None
            else:
                try:
                    m = ecc.regions_mask(spec, self.w, self.h)
                except (ValueError, TypeError, KeyError) as e:
                    raise QcError(f"bad mask spec masks.{name}.by_position.{pos}: {e}")
                want = (self.cfg.get(f"sha256_{pos}") or {}).get(name)
                if want is not None and ecc.mask_sha256(m) != want:
                    raise QcError(f"mask {name} at {pos}: sha256 {ecc.mask_sha256(m)[:12]}... does not match "
                                  f"masks.sha256_{pos}.{name} {str(want)[:12]}...")
                self.cache[key] = m
        return self.cache[key]


# --------------------------------------------------------------------------- checks
def inventory(sdir: Path, flags: Flags) -> list[dict]:
    files = []
    for dirpath, dirnames, filenames in os.walk(sdir):
        rel_dir = Path(dirpath).relative_to(sdir)
        if rel_dir == Path("."):
            dirnames[:] = [d for d in dirnames if d != "qc"]
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            rel = (rel_dir / name).as_posix()
            if rel in LOCK_FILES or not p.is_file() or p.is_symlink():
                continue
            files.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
            if name.startswith(".") and ".partial" in name:
                flags.add("stale_partial", "flag", f"leftover temporary file {rel} (interrupted capture?)")
    return files


def slot_key(r: dict) -> tuple:
    return (r["arrangement"], r["position"], r["role"], r["condition"], r["lighting"], r.get("episode"))


def expected_slots(arr: str, lightings: list[str]) -> list[tuple]:
    """The protocol's slots for one arrangement (slot_key form, episode None; MODULE1 section 4.2)."""
    return [(arr, p, "reference", "visible", L, None) for L in lightings for p in POSITIONS] + \
           [(arr, p, "query", c, L, None) for L in lightings for c in QUERY_CONDITIONS for p in POSITIONS]


def check_unexpected(rows: list[dict], expected: set, flags: Flags) -> set:
    """Stems of manifest rows that fit no expected slot (an episode label, an unknown lighting,
    role/condition pairs the protocol never shoots, ...). They are reported and kept out of
    selection, registration and the per-condition tables. Rows with condition 'pilot' are left
    to the pilot_condition check (same severity rule) so they are not reported twice."""
    out = set()
    for r in rows:
        if slot_key(r) in expected:
            continue
        out.add(r["stem"])
        if r.get("condition") == "pilot":
            continue
        sev = "fail" if RE_STUDY.fullmatch(str(r.get("arrangement"))) else "flag"
        flags.add("unexpected_shot", sev, f"{r['path']} fits no slot of the protocol (role {r.get('role')}, "
                  f"condition {r.get('condition')}, lighting {r.get('lighting')}, episode {r.get('episode')}); "
                  "excluded from selection and registration", r)
    return out


def check_unrecorded(rows: list[dict], runs: dict, skip: set, flags: Flags) -> None:
    """When run/<ARR>.json exists, every manifest row of that arrangement must be one of its takes
    (matched on stem, or on slot + take number for older run files). A row it does not know was
    taken outside the sequencer, or is a pi_capture child that outlived a killed sequencer and was
    never adopted by --resume."""
    for arr, run in runs.items():
        if run is None:
            continue
        stems, slot_takes = set(), set()
        for sl in run.get("slots", []) or []:
            for t in sl.get("takes", []) or []:
                if t.get("stem"):
                    stems.add(t["stem"])
                slot_takes.add((sl.get("position"), sl.get("role"), sl.get("condition"), sl.get("lighting"),
                                t.get("take")))
        for r in rows:
            if r["arrangement"] != arr or r["stem"] in skip or r["stem"] in stems:
                continue
            if (r["position"], r["role"], r["condition"], r["lighting"], int(r["take"])) in slot_takes \
                    and not r.get("episode"):
                continue
            flags.add("unrecorded_take", "flag", f"{r['stem']} is in manifest.jsonl but not in run/{arr}.json "
                      "(bare pi_capture shot, or an orphaned shot not yet adopted by run_arrangement --resume)", r)


def check_stray_files(files: list[dict], all_rows: list[dict], arrs: set, flags: Flags) -> None:
    """Files under <ARR>/ that no manifest row accounts for. pi_capture writes <stem>.jpg,
    <stem>.json and camera logs <stem>[.<time>].log; a failed still leaves <stem>.failed-<time>.log
    (info when its stem has no row: nothing was saved). Hidden .partial files are stale_partial."""
    named = set()
    stems = set()
    for r in all_rows:
        p = Path(r["path"])
        named.update({p.as_posix(), p.with_suffix(".json").as_posix()})
        stems.add((p.parent.as_posix(), r["stem"]))
    for f in files:
        parts = Path(f["path"]).parts
        if len(parts) < 2 or parts[0] not in arrs or f["path"] in named:
            continue
        name, parent = parts[-1], Path(f["path"]).parent.as_posix()
        if name.startswith(".") and ".partial" in name:
            continue  # already stale_partial
        if name.endswith(".log"):
            stem = name.split(".")[0]  # pi_capture stems contain no dots
            if (parent, stem) in stems:
                continue
            if ".failed-" in name:
                flags.add("failed_capture_log", "info", f"{f['path']}: camera log of a failed still (nothing saved)",
                          arrangement=parts[0], stem=stem)
                continue
        flags.add("stray_file", "flag", f"{f['path']} is under {parts[0]}/ but no manifest.jsonl row names it",
                  arrangement=parts[0])


def selected_takes(sdir: Path, rows: list[dict], flags: Flags, skip: set) -> tuple[dict, dict]:
    """{slot key: selected take} and {arrangement: run manifest or None}. The run manifest's
    selected_take wins when present; otherwise the highest take in manifest.jsonl. Rows whose stem
    is in `skip` (unexpected shots) are never selected."""
    runs, sel = {}, {}
    for arr in sorted({r["arrangement"] for r in rows}):
        runs[arr] = read_json(sdir / "run" / f"{arr}.json")
    for r in rows:
        if r["stem"] in skip:
            continue
        k = slot_key(r)
        sel[k] = max(sel.get(k, 0), int(r["take"]))
    for arr, run in runs.items():
        if run is not None and run.get("status") != "complete":
            # run_arrangement marks an interrupted arrangement "aborted" (resumable) or "in_progress"
            flags.add("run_incomplete", "flag", f"run/{arr}.json status is {run.get('status')!r}, not 'complete' "
                      "(resume it with run_arrangement.py arrangement --resume)", arrangement=arr)
        for s in (run or {}).get("slots", []):
            if s.get("selected_take") is None:
                if s.get("takes"):
                    # saved but never accepted by the operator: the highest manifest take stands in
                    flags.add("unaccepted_take", "flag",
                              f"slot {s.get('index')} has take(s) {[t.get('take') for t in s['takes']]} but no "
                              "accepted (selected) take; using the highest take", arrangement=arr,
                              condition=s.get("condition"), lighting=s.get("lighting"), position=s.get("position"))
                continue
            k = (arr, s.get("position"), s.get("role"), s.get("condition"), s.get("lighting"), None)
            if k in sel:
                sel[k] = int(s["selected_take"])
            else:
                flags.add("missing_image", "fail", f"run/{arr}.json selects take {s['selected_take']} of a slot "
                          "with no manifest row", arrangement=arr, condition=s.get("condition"),
                          lighting=s.get("lighting"), position=s.get("position"))
    return sel, runs


def lock_config_check(md: dict, side: dict, lock: dict) -> tuple[bool, list[str]]:
    """Sidecar metadata vs capture_lock.json `lock` (MODULE1 section 4.2)."""
    tol = lock.get("tolerances", {})
    bad = []
    exp = md.get("ExposureTime")
    if exp is None or abs(exp - lock["shutter_us"]) > tol.get("shutter_rel", 0.005) * lock["shutter_us"]:
        bad.append(f"ExposureTime {exp} vs {lock['shutter_us']}")
    ag = md.get("AnalogueGain")
    g = ag * (md.get("DigitalGain") or 1.0) if ag is not None else None
    if g is None or abs(g - lock["gain"]) > tol.get("gain_rel", 0.02) * lock["gain"]:
        bad.append(f"gain {None if g is None else round(g, 4)} vs {lock['gain']}")
    cg = md.get("ColourGains")
    if not (isinstance(cg, list) and len(cg) == 2 and None not in cg
            and all(abs(x - y) <= tol.get("awb_abs", 0.001) for x, y in zip(cg, lock["awb_gains"]))):
        bad.append(f"ColourGains {cg} vs {lock['awb_gains']}")
    lp = md.get("LensPosition")
    if lp is None or abs(lp - lock["lens_position"]) > tol.get("lens_abs", 0.01):
        bad.append(f"LensPosition {lp} vs {lock['lens_position']}")
    if crop4(md.get("ScalerCrop")) != crop4(lock.get("scaler_crop")):
        bad.append(f"ScalerCrop {md.get('ScalerCrop')} vs {lock.get('scaler_crop')}")
    im = side.get("image") or {}
    if [im.get("width"), im.get("height")] != list(lock["size"]):
        bad.append(f"image size {im.get('width')}x{im.get('height')} vs {lock['size'][0]}x{lock['size'][1]}")
    return not bad, bad


def saturation(img: np.ndarray, level: int, region: np.ndarray | None, glare: np.ndarray | None) -> dict:
    sat = img.max(axis=2) >= level if img.ndim == 3 else img >= level
    if region is None:
        reg, source = np.ones(sat.shape, bool), "full_frame_fallback"
    else:
        reg, source = region > 0, "mug_region"
    if glare is not None:
        reg = reg & ~(glare > 0)
    n = int(reg.sum())
    out = {"saturation_share": round(float((sat & reg).sum()) / n, 8) if n else None, "saturation_region": source,
           "glare_share": None}
    if glare is not None and int((glare > 0).sum()):
        out["glare_share"] = round(float((sat & (glare > 0)).sum()) / int((glare > 0).sum()), 6)
    return out


def check_sitting(sdir: Path, flags: Flags) -> dict:
    run = sdir / "run"
    sitting = read_json(run / "sitting.json")
    if sitting is None:
        flags.add("no_sequencer_log", "flag", "run/sitting.json is missing: the shots were taken with bare "
                  "pi_capture (no NTP preflight or pose-check record)")
        return {"present": False, "ntp_ok": None, "pose_verdict": None, "pose_shift_px": None}
    ntp_ok = (sitting.get("ntp") or {}).get("ok") is True
    if not ntp_ok:
        # a flag, not a fail: NTP is a warning since run_arrangement 1.2.0 (DECISIONS 2026-10-12)
        flags.add("ntp_not_synced", "flag", f"sitting NTP preflight not ok: {sitting.get('ntp')}")
    pose = sitting.get("pose") or {}
    verdict, shift = pose.get("verdict"), None
    pose_file = run / Path(pose["file"]).name if pose.get("file") else None
    rep = read_json(pose_file) if pose_file else None
    if rep is not None:
        shift = ((rep.get("translation") or {}).get("shift_px"))
    if not pose or verdict is None or rep is None:
        flags.add("pose_missing", "fail", f"sitting.json has no usable pose record (pose={pose}, "
                  f"file {'missing' if rep is None else 'present'})")
    elif verdict == "flag":
        flags.add("pose_flag", "flag", f"pose check flagged and was accepted: reasons {rep.get('reasons')}, "
                  f"note {pose.get('accepted_flag_note')!r}")
    elif verdict != "ok":
        flags.add("pose_missing", "fail", f"sitting pose verdict {verdict!r} is neither ok nor flag")
    return {"present": True, "ntp_ok": ntp_ok, "pose_verdict": verdict, "pose_shift_px": shift}


# ----------------------------------------------------------------------------- core
def run_qc(sdir: Path, cfg: dict, cfg_meta: dict, params: dict, arrangements: list[str] | None,
           golden: Path) -> dict:
    flags = Flags()
    manifest = sdir / "manifest.jsonl"
    if not manifest.is_file():
        raise QcError(f"{manifest} does not exist (not a pi_capture session directory?)")
    all_rows = read_manifest(manifest)
    rows = [r for r in all_rows if not arrangements or r.get("arrangement") in arrangements]
    lock_doc = read_json(sdir / "lock.json")
    lock = cfg["lock"]
    try:
        w, h = int(lock["size"][0]), int(lock["size"][1])
    except (KeyError, TypeError, IndexError, ValueError):
        raise QcError("lock config lock.size is missing")
    masks = MaskBank(cfg, w, h)
    files = inventory(sdir, flags)
    sitting = check_sitting(sdir, flags)
    if lock_doc is None:
        flags.add("missing_slot", "fail", "lock.json is missing or unreadable: expected lightings unknown "
                  "(using the lightings seen in manifest.jsonl)")
        lightings = sorted({r["lighting"] for r in rows})
    else:
        lightings = sorted((lock_doc.get("locks") or {}).keys())
    arrs = sorted(set(arrangements or []) | {r["arrangement"] for r in rows})
    expected = {k for arr in arrs for k in expected_slots(arr, lightings)}
    unexpected = check_unexpected(rows, expected, flags)
    sel, runs = selected_takes(sdir, rows, flags, unexpected)
    check_unrecorded(rows, runs, unexpected, flags)
    tree_arrs = set(arrs) if arrangements else set(arrs) | {
        Path(f["path"]).parts[0] for f in files
        if len(Path(f["path"]).parts) > 1 and (RE_STUDY.fullmatch(Path(f["path"]).parts[0])
                                               or RE_PILOT.fullmatch(Path(f["path"]).parts[0]))}
    check_stray_files(files, all_rows, tree_arrs, flags)

    # ---- per-row integrity / metadata
    shots = []
    for r in rows:
        k = slot_key(r)
        is_sel = r["stem"] not in unexpected and sel.get(k) == int(r["take"])
        img_path = sdir / r["path"]
        side = read_json(img_path.with_suffix(".json"))
        rec = {"stem": r["stem"], "path": r["path"], "arrangement": r["arrangement"], "position": r["position"],
               "role": r["role"], "condition": r["condition"], "lighting": r["lighting"],
               "episode": r.get("episode"), "take": int(r["take"]), "selected": is_sel,
               "unexpected": r["stem"] in unexpected, "time_utc": r.get("time_utc"),
               "sha256_ok": None, "lock_ok": None, "lock_config_ok": None, "lux": r.get("lux"),
               "lux_ratio_calibration": None, "lux_ratio_reference": None, "saturation_share": None,
               "saturation_region": None, "glare_share": None, "registration": None}
        shots.append(rec)
        if not img_path.is_file():
            flags.add("missing_image", "fail", f"{r['path']} does not exist", r)
            rec["sha256_ok"] = False
        else:
            digest = sha256_file(img_path)
            side_sha = ((side or {}).get("image") or {}).get("sha256")
            rec["sha256_ok"] = digest == r.get("sha256") and digest == side_sha
            if not rec["sha256_ok"]:
                flags.add("sha_mismatch", "fail", f"sha256 {digest[:12]}... vs manifest {str(r.get('sha256'))[:12]}... "
                          f"and sidecar {str(side_sha)[:12]}...", r)
        if side is None:
            flags.add("missing_image", "fail", f"sidecar {Path(r['path']).with_suffix('.json')} is missing or "
                      "not valid JSON", r)
        else:
            if "lock_check" not in side:
                flags.add("lock_check_failed", "flag", "sidecar has no lock_check (pi_capture 0.1 sidecar)", r)
            else:
                rec["lock_ok"] = bool((side.get("lock_check") or {}).get("ok") is True)
                if not rec["lock_ok"]:
                    bad = [kk for kk, v in (side.get("lock_check") or {}).items()
                           if isinstance(v, dict) and not v.get("ok")]
                    flags.add("lock_check_failed", "fail", f"camera did not apply the session lock: {bad}", r)
            md = side.get("rpicam_metadata") or {}
            rec["lock_config_ok"], why = lock_config_check(md, side, lock)
            if not rec["lock_config_ok"]:
                flags.add("lock_config_mismatch", "fail", "; ".join(why), r)
            rec["lux"] = md.get("Lux", rec["lux"]) if md else rec["lux"]
            if md.get("Lux") is None:
                flags.add("lux_missing", "flag", "no Lux in the rpicam metadata", r)
            ratio = (side.get("lighting_check") or {}).get("ratio")
            rec["lux_ratio_calibration"] = ratio
            g = params["lux_guard"]
            if is_sel and ratio is not None and not (1 / g <= ratio <= g):
                flags.add("lux_vs_calibration", "flag", f"Lux ratio vs the calibration shot {ratio:.3f} "
                          f"outside [{1 / g:.3f}, {g:.3f}]", r)
            ov = (side.get("protocol") or {}).get("overrides") or []
            if ov:
                flags.add("protocol_override", "fail", f"pi_capture protocol override(s): {ov}", r)
        if r["condition"] == "pilot":
            if RE_PILOT.fullmatch(r["arrangement"]):
                flags.add("pilot_condition", "flag", "condition 'pilot' in a pilot arrangement", r)
            else:
                flags.add("pilot_condition", "fail", "condition 'pilot' in a study arrangement", r)

    # ---- retakes
    takes_per_slot: dict[tuple, list[int]] = {}
    for r in rows:
        if r["stem"] not in unexpected:
            takes_per_slot.setdefault(slot_key(r), []).append(int(r["take"]))
    for k, takes in sorted(takes_per_slot.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if len(takes) > 1:
            flags.add("retake", "info", f"takes {sorted(takes)}, selected {sel.get(k)}", arrangement=k[0],
                      position=k[1], condition=k[3], lighting=k[4])

    # ---- selected references and lux vs reference
    selected = [s for s in shots if s["selected"]]
    refs = {}
    for s in selected:
        if s["role"] == "reference":
            key = (s["arrangement"], s["position"], s["lighting"], s["episode"])
            if key not in refs or s["condition"] == "visible":
                refs[key] = s
    g = params["lux_guard"]
    for s in selected:
        if s["role"] != "query":
            continue
        ref = refs.get((s["arrangement"], s["position"], s["lighting"], s["episode"]))
        if ref and s["lux"] is not None and ref["lux"]:
            s["lux_ratio_reference"] = round(s["lux"] / ref["lux"], 4)
            if not (1 / g <= s["lux_ratio_reference"] <= g):
                flags.add("lux_vs_reference", "flag", f"Lux {s['lux']:.2f} vs reference {ref['stem']} "
                          f"{ref['lux']:.2f} (x{s['lux_ratio_reference']:.3f})", s)

    # ---- completeness
    completeness = {}
    for arr in arrs:
        exp_arr = expected_slots(arr, lightings)
        missing = [e for e in exp_arr if e not in sel]
        completeness[arr] = {"expected": len(exp_arr), "present": len(exp_arr) - len(missing),
                             "missing": [f"{e[1]} {e[2]} {e[3]} {e[4]}" for e in missing]}
        for e in missing:
            flags.add("missing_slot", "fail", f"no {e[2]} {e[3]} at {e[1]} under {e[4]}", arrangement=arr,
                      position=e[1], condition=e[3], lighting=e[4])

    # ---- images: saturation, shift, ref vs golden
    if params["images"]:
        image_checks(sdir, cfg, params, masks, selected, refs, flags, golden)

    # ---- summaries
    flags_by_stem: dict[str, list[str]] = {}
    for f in flags.items:
        if f["stem"]:
            flags_by_stem.setdefault(f["stem"], []).append(f["code"])
    groups: dict[tuple, list[dict]] = {}
    for s in selected:
        groups.setdefault((s["arrangement"], s["role"], s["condition"], s["lighting"]), []).append(s)
    by_condition = []
    for (arr, role, cond, light), ss in sorted(groups.items()):
        codes: dict[str, int] = {}
        for s in ss:
            for c in flags_by_stem.get(s["stem"], []):
                codes[c] = codes.get(c, 0) + 1
        shifts = [(s["registration"] or {}).get("shift_px") for s in ss]
        sats = [s["saturation_share"] for s in ss if s["saturation_share"] is not None]
        by_condition.append({"arrangement": arr, "role": role, "condition": cond, "lighting": light, "n": len(ss),
                             "lux_ratio_reference": stats([s["lux_ratio_reference"] for s in ss]),
                             "shift_px": stats(shifts), "saturation_share_max": max(sats) if sats else None,
                             "flags": codes})
    by_all: dict[tuple, list[dict]] = {}
    for s in selected:
        by_all.setdefault((s["role"], s["condition"]), []).append(s)
    by_condition_all = []
    for (role, cond), ss in sorted(by_all.items(), key=lambda kv: (kv[0][0] != "reference",
                                                                    ("visible", "occluded", "removed", "pilot").index(kv[0][1])
                                                                    if kv[0][1] in ("visible", "occluded", "removed", "pilot") else 9)):
        lr = stats([s["lux_ratio_reference"] for s in ss])
        sh = stats([(s["registration"] or {}).get("shift_px") for s in ss])
        lx = stats([s["lux"] for s in ss])
        times = sorted(s["time_utc"] for s in ss if s.get("time_utc"))
        by_condition_all.append({"role": role, "condition": cond, "n": len(ss),
                                 "lux_ratio_reference_median": lr["median"] if lr else None,
                                 "shift_px_median": sh["median"] if sh else None,
                                 "lux_median": lx["median"] if lx else None,
                                 "first_utc": times[0] if times else None, "last_utc": times[-1] if times else None})

    verdict = "fail" if flags.count("fail") else "flag" if flags.count("flag") else "pass"
    return {
        "schema": SCHEMA,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "ecc_version": ecc.ECC_VERSION,
                 "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                 "hostname": socket.gethostname()},
        "time_utc": None,  # set by main
        "session": sdir.name, "session_dir": str(sdir.resolve()),
        "arrangements_filter": arrangements,
        "lock_config": cfg_meta,
        "params": params,
        "lightings": lightings,
        "sitting": sitting,
        "files": files,
        "shots": shots,
        "completeness": completeness,
        "by_condition": by_condition,
        "by_condition_all": by_condition_all,
        "flags": flags.items,
        "counts": {"shots": len(shots), "selected": len(selected), "unexpected": len(unexpected),
                   "expected": sum(c["expected"] for c in completeness.values()),
                   "missing": sum(len(c["missing"]) for c in completeness.values()),
                   **{sev: flags.count(sev) for sev in SEVERITIES}},
        "verdict": verdict,
        "exit_code": {"pass": EXIT_PASS, "flag": EXIT_FLAG, "fail": EXIT_FAIL}[verdict],
    }


def image_checks(sdir: Path, cfg: dict, params: dict, masks: MaskBank, selected: list[dict], refs: dict,
                 flags: Flags, golden: Path) -> None:
    reg = params["registration"]
    kw = dict(sigma=reg["sigma"], iters=reg["iters"], eps=reg["eps"], mode=reg["mode"], levels=reg["levels"])
    gray_cache: dict[str, np.ndarray] = {}

    def load(s: dict) -> np.ndarray | None:
        img = cv2.imread(str(sdir / s["path"]), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            flags.add("missing_image", "fail", f"{s['path']} could not be decoded", s)
        return img

    def sat_check(s: dict, img: np.ndarray) -> None:
        pos = s["position"]
        sat = saturation(img, params["saturation_level"], masks.get("mug_region", pos), masks.get("glare", pos))
        s.update(sat)
        if sat["saturation_share"] is not None and sat["saturation_share"] > params["saturation_flag_share"]:
            flags.add("saturation", "flag", f"{sat['saturation_share']:.5f} of {sat['saturation_region']} pixels "
                      f">= {params['saturation_level']} (limit {params['saturation_flag_share']})", s)

    # references first (kept as grey float32 for the query registrations), then queries one at a time
    for s in (s for s in selected if s["role"] == "reference"):
        img = load(s)
        if img is not None:
            sat_check(s, img)
            gray_cache[s["stem"]] = ecc.to_gray_f32(img)
    for s in (s for s in selected if s["role"] == "query"):
        img = load(s)
        if img is None:
            continue
        sat_check(s, img)
        ref = refs.get((s["arrangement"], s["position"], s["lighting"], s["episode"]))
        if ref is None or ref["stem"] not in gray_cache:
            continue  # no selected reference: already a missing_slot / missing_image fail (unexpected
            #           rows never reach here, they are not selected)
        m = masks.get("static_background", s["position"])
        try:
            if m is not None:
                t = ecc.ecc_translation(gray_cache[ref["stem"]], img, mask=m, **kw)
                source = "static_background"
            else:
                t, _ = ecc.ecc_translation_robust(gray_cache[ref["stem"]], img, **kw)
                source = "auto_residual"
        except ecc.EccError as e:
            s["registration"] = {"against": ref["stem"], "error": str(e)}
            flags.add("registration_failed", "flag", f"ECC vs {ref['stem']}: {e}", s)
            continue
        shift = math.hypot(t.dx, t.dy)
        s["registration"] = {"against": ref["stem"], "dx": round(t.dx, 3), "dy": round(t.dy, 3),
                             "shift_px": round(shift, 3), "cc": round(t.cc, 4), "mask_source": source}
        if shift > params["shift_flag_px"]:
            flags.add("shift", "flag", f"{shift:.2f} px vs {ref['stem']} (limit {params['shift_flag_px']} px, "
                      f"mask {source})", s)

    # ---- P3 references vs the golden (info only)
    p3_refs = [s for s in selected if s["role"] == "reference" and s["position"] == "P3" and s["stem"] in gray_cache]
    if not p3_refs:
        return
    want = (cfg.get("reference") or {}).get("sha256")
    skip = None
    if not golden.is_file():
        skip = f"golden {golden} not available"
    elif sha256_file(golden) != want:
        skip = f"golden {golden} sha256 does not match reference.sha256 in the lock config"
    m = masks.get("static_background", "P3")
    if skip is None and m is None:
        skip = "no static_background mask for P3"
    gimg = None if skip else cv2.imread(str(golden), cv2.IMREAD_GRAYSCALE)
    if skip is None and (gimg is None or gimg.shape[:2] != m.shape):
        skip = f"golden {golden} unreadable or not {m.shape[1]}x{m.shape[0]}"
    if skip:
        flags.add("ref_vs_golden", "info", f"skipped: {skip}")
        return
    for s in p3_refs:
        try:
            t = ecc.ecc_translation(gimg, gray_cache[s["stem"]], mask=m, **kw)
            shift = math.hypot(t.dx, t.dy)
            s["golden_registration"] = {"dx": round(t.dx, 3), "dy": round(t.dy, 3), "shift_px": round(shift, 3),
                                        "cc": round(t.cc, 4), "mask_source": "static_background"}
            flags.add("ref_vs_golden", "info", f"{shift:.2f} px vs golden (cc {t.cc:.4f})", s)
        except ecc.EccError as e:
            s["golden_registration"] = {"error": str(e)}
            flags.add("ref_vs_golden", "info", f"registration vs golden failed: {e}", s)


# ------------------------------------------------------------------------- markdown
def fmt(v, nd=3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"
    return str(v)


def markdown(rep: dict) -> str:
    c = rep["counts"]
    out = [f"# QC {rep['session']}: **{rep['verdict'].upper()}** (exit {rep['exit_code']})", "",
           f"- time {rep['time_utc']}, {rep['tool']['name']} {rep['tool']['version']} on {rep['tool']['hostname']}"
           f", images {'on' if rep['params']['images'] else 'off (--no-images)'}",
           f"- shots {c['shots']} (selected {c['selected']}), expected slots {c['expected']}, missing {c['missing']}",
           f"- flags: fail {c['fail']}, flag {c['flag']}, info {c['info']}",
           f"- sitting: {'present' if rep['sitting']['present'] else 'missing'}, NTP ok {rep['sitting']['ntp_ok']}, "
           f"pose {rep['sitting']['pose_verdict']} ({fmt(rep['sitting']['pose_shift_px'])} px)", "",
           "| arrangement | role | condition | lighting | n | Lux ratio vs ref (median) | shift px median | "
           "shift px max | saturation max | flags |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in rep["by_condition"]:
        lr, sh = g["lux_ratio_reference"] or {}, g["shift_px"] or {}
        out.append(f"| {g['arrangement']} | {g['role']} | {g['condition']} | {g['lighting']} | {g['n']} | "
                   f"{fmt(lr.get('median'))} | {fmt(sh.get('median'))} | {fmt(sh.get('max'))} | "
                   f"{fmt(g['saturation_share_max'], 5)} | {sum(g['flags'].values())} |")
    out.append("")
    order = {s: i for i, s in enumerate(SEVERITIES)}
    fl = sorted(rep["flags"], key=lambda f: order[f["severity"]])
    for f in fl[:MAX_MD_FLAGS]:
        where = " ".join(str(f[k]) for k in ("stem",) if f[k]) or " ".join(
            str(f[k]) for k in ("arrangement", "position", "condition", "lighting") if f[k])
        out.append(f"- {f['severity'].upper()} `{f['code']}` {where}: {f['detail']}")
    if len(fl) > MAX_MD_FLAGS:
        out.append(f"- … {len(fl) - MAX_MD_FLAGS} more in the JSON")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    p.add_argument("--session-dir", required=True, help="<root>/<session> as written by pi_capture.py")
    p.add_argument("--lock-config", help="capture_lock.json (default $CMPE249_LOCK_CONFIG or "
                                         "~/cmpe249/configs/capture_lock.json)")
    p.add_argument("--golden", help="golden P3 reference for ref_vs_golden (default $CMPE249_GOLDEN or "
                                    "~/cmpe249/golden/ref_final.jpg)")
    p.add_argument("--arrangement", nargs="+", help="only these arrangements (default: all in the manifest)")
    p.add_argument("--out-dir", help="default <session-dir>/qc/")
    p.add_argument("--no-images", action="store_true", help="skip registration and saturation")
    p.add_argument("--lux-guard", type=float, help="override qc.lux_ratio_guard")
    p.add_argument("--shift-flag-px", type=float, help="override qc.shift_flag_px")
    p.add_argument("--saturation-level", type=int, help="override qc.saturation_level")
    p.add_argument("--saturation-flag-share", type=float, help="override qc.saturation_flag_share")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        sdir = Path(a.session_dir).expanduser()
        if not sdir.is_dir():
            raise QcError(f"--session-dir {sdir} is not a directory")
        cfg_path = Path(a.lock_config or default_lock_config()).expanduser()
        cfg, cfg_sha = load_lock_config(cfg_path)
        params, _ = resolve_params(cfg, a)
        cfg_meta = {"path": str(cfg_path.resolve()), "sha256": cfg_sha, "lock_version": cfg.get("lock_version")}
        out_dir = Path(a.out_dir).expanduser() if a.out_dir else sdir / "qc"
        t = now_utc()
        stem = f"qc_{t.strftime('%Y%m%dT%H%M%SZ')}"
        jpath, mpath = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
        if jpath.exists() or mpath.exists():
            raise QcError(f"{jpath} or {mpath} already exists; QC reports are never overwritten (retry in a second)")
        rep = run_qc(sdir, cfg, cfg_meta, params, a.arrangement, Path(a.golden or default_golden()).expanduser())
        rep["time_utc"] = iso(t)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(jpath, "x") as f:
                json.dump(rep, f, indent=2)
                f.write("\n")
            with open(mpath, "x") as f:
                f.write(markdown(rep))
        except OSError as e:
            raise QcError(f"cannot write the QC report into {out_dir}: {e}")
    except QcError as e:
        print(f"{TOOL_NAME}: error: {e}", file=sys.stderr)
        return e.code
    c = rep["counts"]
    print(f"QC {rep['session']}: {rep['verdict'].upper()} - {c['shots']} shots, {c['missing']} missing slots, "
          f"fail {c['fail']}, flag {c['flag']}, info {c['info']}")
    print(f"report: {jpath}")
    print(f"summary: {mpath}")
    return rep["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
