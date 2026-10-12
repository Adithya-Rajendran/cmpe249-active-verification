# Capture tool (`pi_capture.py`)

Locked-settings still capture for the reference and query images described in
[`docs/EXPERIMENT_PLAN.md`](../docs/EXPERIMENT_PLAN.md). It runs **on the Raspberry Pi** (Python 3.13,
standard library only) and wraps `rpicam-still`. Tested on 2026-09-26 with rpicam-apps 1.13.0,
libcamera 0.7.2 and the imx708 Camera Module 3 (standard lens). No camera is needed to run the unit tests.

Why the settings are locked, and what the camera measurements showed, is in
[`notes/2026-09-26-camera-check.md`](../notes/2026-09-26-camera-check.md). In short: autofocus picked
the background rather than the near object, it wandered by 0.2–0.3 dioptres between runs, and the
room lighting changed twice in three minutes.

## Install / update on the Pi

```bash
# from the repo root on the GB10
ssh -o BatchMode=yes ai-pi 'mkdir -p ~/cmpe249/capture'
rsync -a --exclude __pycache__ capture/pi_capture.py capture/tests ai-pi:cmpe249/capture/
ssh -o BatchMode=yes ai-pi 'cd ~/cmpe249 && python3 -m unittest discover -s capture/tests'
```

## Usage

```bash
T=~/cmpe249/capture/pi_capture.py
S=2026-10-03-A01-A05          # one session = one lock file (e.g. one sitting at the rig)

# 1. Calibrate once per session and lighting condition. Camera at the calibration position (P3),
#    lamps set for L1. Prefer manual focus at the measured slant distance to the workspace centre:
python3 $T calibrate --session $S --lighting L1 --lens-position 1.75   # ~100 / 57 cm, see notes
#    (without --lens-position it runs autofocus once, which may lock onto the background)
#    Camera mounted upside down? add --rotation 180 (only 0/180 exist; fix any roll physically).
#    Study sittings (0.3.0) instead take the fixed manual controls (shutter, gain, AWB gains, lens,
#    size) from the lock config; AE/AWB/AF are off. run_arrangement.py `sitting` does this for you:
python3 $T calibrate --session $S --lighting L1 --lock-config ~/cmpe249/configs/capture_lock.json
#    --lock-config refuses --copy-from/--lens-position/--af-range/--size/--rotation (the file's
#    values would silently win); lock_check (incl. ScalerCrop) is recorded and a WARNING is printed
#    if the camera did not apply them.

# 2. Second lighting condition: its own calibration ...
python3 $T calibrate --session $S --lighting L2 --lens-position 1.75
#    ... or keep exactly the same controls under both lightings (lamps set for L2 first: it takes one
#    still with L1's controls, whose Lux becomes L2's reference for the ambient-light guard):
python3 $T calibrate --session $S --lighting L2 --copy-from L1
#    --copy-from refuses --size/--quality/--rotation/--lens-position/--af-range/--shoot-settle-ms,
#    because the copied lock's values would silently win.

# 3. References at all five positions BEFORE changing the scene
for P in P1 P2 P3 P4 P5; do   # move the camera to $P on the template first
  python3 $T shoot --session $S --arrangement A01 --position $P --role reference \
      --condition visible --lighting L1
done

# 4. Queries: one per position per episode (condition x lighting), after the scene change
python3 $T shoot --session $S --arrangement A01 --position P3 --role query \
    --condition occluded --lighting L1 --note "cereal box 6 cm in front of mug"

# 5. What has been captured?
python3 $T status --session $S

# 6. Pull to the GB10 (idempotent; nothing on the Pi is modified)
rsync -a ai-pi:cmpe249/data/$S data/raw/sessions/
```

Labels are validated:

| Flag | Allowed values |
| --- | --- |
| `--session` | `[A-Za-z0-9][A-Za-z0-9.-]{0,63}` |
| `--arrangement` | `[A-Z][A-Z0-9]{1,15}`, e.g. `A01`, `PILOT00` |
| `--position` | `P1`..`P5` |
| `--role` | `reference`, `query` |
| `--condition` | `visible`, `occluded`, `removed`, `pilot` (references: `visible` or `pilot` only) |
| `--lighting` | `[A-Za-z0-9][A-Za-z0-9-]{0,15}`, and it must have been calibrated in this session |
| `--episode` (optional) | same pattern as lighting |
| `--take` | 1..99, default 1 |

## Directory layout

One directory per session, laid out by the unit of evaluation (arrangement → role → position).
Copy it to the GB10 as-is with `rsync`.

```
~/cmpe249/data/<session>/
  lock.json               current lock per lighting label + full history of locks
  session_log.jsonl       append-only: calibrate / recalibrate / shoot events
  manifest.jsonl          one line per saved image (labels, lock_id, sha256, Lux)
  calibration/
    L1-v1.jpg  L1-v1.json  L1-v1.log
  A01/
    reference/
      P1/ A01_P1_reference_visible_L1_t01.jpg  .json  .log
      ...
      P5/
    query/
      P1/ A01_P1_query_occluded_L1_t01.jpg  .json  .log
      ...
```

File stem: `<arrangement>_<position>_<role>_<condition>_<lighting>[_<episode>]_t<take>`.

Each image has a JSON sidecar with: the labels; the image file name, byte size, sha256 and pixel
size (read from the JPEG header, not assumed); capture start and end timestamps (UTC and local),
wall time, hostname and the exact `rpicam-still` command (it names the hidden `.partial` files
rpicam wrote to; see `command_note`); the full lock entry and the sha256 of `lock.json` at capture
time; the complete rpicam metadata (ExposureTime, AnalogueGain, ColourGains, LensPosition, Lux,
ScalerCrop, …); a `lock_check` comparing requested and reported controls, the sensor crop (field of
view) and the pixel size; a `lighting_check` comparing Lux with the calibration shot; protocol
warnings and overrides; and the tool, Python and rpicam versions. Each `manifest.jsonl` line also
carries `lock_ok` and `light_ok` (`null` = Lux missing), so flagged shots can be filtered without
opening the sidecars.

## Protocol rules the tool enforces

| Rule | What happens |
| --- | --- |
| Settings are locked per session | `shoot` refuses without a lock for its lighting label. Every shot uses the locked shutter, gain, AWB gains and lens position, plus pinned ISP controls (denoise, sharpness, contrast, saturation, brightness). The sensor mode is fixed to a full-field-of-view mode; the 1536×864 mode is a centre crop and is not offered. |
| No silent re-calibration | A second `calibrate` for the same lighting is refused unless `--recalibrate --reason "..."`. The new lock gets a new id (`L1-v2`), and the old one stays in `lock.json` history and `session_log.jsonl`. |
| Never re-calibrate mid-arrangement unless recorded | If an arrangement already has shots under an older lock for the same lighting, `shoot` refuses unless `--accept-lock-change --note "..."`. The override is written into the sidecar. |
| References before scene changes | A `query` at position P needs a reference for that arrangement at P. A query also needs references at all five positions, except for `--condition pilot`, where it is only a warning. Once an arrangement has any query, more references are refused unless `--allow-late-reference --note "..."`. |
| References show the unchanged scene | `--role reference` only accepts `--condition visible` or `pilot`. |
| Nothing is overwritten | Existing images and sidecars are never replaced. A retake must be explicit (`--take 2`), and it gets its own file. The image is written to a hidden temporary file (stale temporaries from an interrupted run are deleted first), checked, and hard-linked into place. |
| No partial or unverifiable captures | If rpicam-still exits 0 but the JPEG is missing, empty or truncated (no end-of-image marker, no frame header), or the metadata file is missing or not JSON, nothing is saved, the manifest is untouched, the camera log is kept as `<stem>.failed-<time>.log`, and the exit code is 5. |
| Lock actually applied | After each shot the reported controls are compared with the lock (shutter ±0.5 %, gain ±2 %, AWB ±0.001, lens ±0.01 D), the reported `ScalerCrop` with the calibration still's (a change means a different field of view), and the JPEG's pixel size with the lock's. A mismatch is kept but flagged: a warning, `lock_check.ok = false` in the sidecar, `lock_ok = false` in the manifest. |
| Lighting unchanged | If rpicam's Lux estimate differs from the calibration shot by more than ×1.25, the shot is kept but flagged. Check the lamps before continuing. A shot or calibration without a Lux value is also flagged rather than silently passed. A lock made with `--copy-from` has its own Lux reference, measured under its own lighting. |

Exit codes: 0 ok, 2 bad arguments / labels, 3 refused overwrite, 4 protocol violation, 5 camera
failure (rpicam error or timeout, missing/invalid image or metadata), 6 no lock. `--dry-run` prints
the rpicam command and writes nothing. Only one `pi_capture.py` may work on a session at a time (file lock).

## What the tool cannot enforce

These depend on the physical setup and remain the operator's job:

- Camera placement at P1..P5, and the same height, tilt and roll every time (use the template/jig).
- The lighting set-up behind each label. The Lux guard only catches large changes.
- Matched occluder layouts between occluded-present and removed episodes.
- Capturing all five query positions for every episode. `status` shows the gaps.

## Study capture tools (module 1)

Interface contract: `MODULE1.md`. All three run on the Pi with system Python; `qc_session.py` also runs on the GB10.

```bash
python3 run_arrangement.py sitting --session S        # NTP check, locked calibration at P3, pose check, lock copied to L2
python3 run_arrangement.py arrangement --session S --arrangement A01 --ordering A   # guided P1..P5 sweep
python3 pose_check.py --image IMG --lock-config ~/cmpe249/configs/capture_lock.json --golden ~/cmpe249/golden/ref_final.jpg
python3 qc_session.py --session-dir ~/cmpe249/data/S  # hashes, metadata, pixel stats, registration; no target labels
```

`ecc.py` holds the shared image registration (ECC translation/euclidean). It gives the same results on OpenCV 4.10 (Pi) and 5.0 (GB10).
The pose thresholds in `configs/capture_lock.json` are proposals, still awaiting decision D17.

## Tests

```bash
python3 -m unittest discover -s capture/tests -v     # Pi (system CPython 3.13.5); uses tests/fake_rpicam_still.py
uv run python -m unittest discover -s capture/tests -v   # GB10: the project's CPython 3.13.5 (bare python3 there is 3.12)
# The fake camera can inject failures (no/invalid metadata, no/truncated image, wrong shutter, crop or
# size, missing Lux); see the docstring of tests/fake_rpicam_still.py.
```

## Camera characterization helpers

`camera_check/camcheck.sh` (bash, run on the Pi) and `camera_check/analyze_camcheck.py` (GB10, via
`uv run --no-project --with numpy --with pillow --with matplotlib`) produced the 2026-09-26
measurements. Re-run `camcheck.sh sweep <dir>` once the rig is built to choose the locked lens
position for the real workspace.
