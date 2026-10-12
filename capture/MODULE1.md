# Capture Module 1: interface contract

This is the contract for `pose_check.py`, `run_arrangement.py`, `qc_session.py`, and the small
`pi_capture.py` change they depend on. Builders follow it exactly. If something here conflicts with
`docs/EXPERIMENT_PLAN.md` or with a rule that `pi_capture.py` enforces, the protocol and `pi_capture.py`
win. Report the conflict in your notes instead of resolving it silently. If you need to change a
shared file you don't own (`ecc.py`, `configs/capture_lock.json`, `fake_rpicam_still.py`, except
for the one addition given to builder 0), say so in your notes.

Status: **contract v1**. The thresholds marked *(proposed)* wait on the student's D17 decision.

## 0. Files, owners and hosts

| File | Owner | Runs on | May import |
| --- | --- | --- | --- |
| `capture/ecc.py` | done (contract author) | Pi + GB10 | stdlib, numpy, cv2 |
| `configs/capture_lock.json` | done (contract author) | data | – |
| `capture/pi_capture.py` → **0.3.0** (§1) | builder 0 | Pi | stdlib only (unchanged rule) |
| `capture/tests/fake_rpicam_still.py` (one addition, §1.3) | builder 0 | both | stdlib only |
| `capture/pose_check.py` | builder 1 | Pi (also GB10) | stdlib, numpy, cv2, `ecc` |
| `capture/run_arrangement.py` | builder 2 | Pi | **stdlib only**; it calls `pi_capture.py` and `pose_check.py` as subprocesses and never imports them |
| `capture/qc_session.py` | builder 3 | GB10 or Pi | stdlib, numpy, cv2, `ecc` |
| `capture/tests/test_<module>.py`, plus `fake_timedatectl.py` and `fake_pose_check.py` (builder 2) | the builder of that module | both | the module under test, `ecc`, numpy/cv2 if the module uses them |

- **Imports.** Put the `capture/` directory on `sys.path` and use `import ecc` (and
  `import pi_capture as pc` in its own tests). That is the layout on the Pi (`~/cmpe249/capture/`).
  Do not add a `capture/__init__.py`.
- **Hosts.** On the Pi the code runs under system CPython 3.13.5 with apt numpy 2.2.4 and OpenCV
  4.10.0. On the GB10 it runs in the uv env (CPython 3.13.5, numpy 2.5.3, OpenCV 5.0.0). Never use
  `cv2.findTransformECCWithMask`, which does not exist on 4.10. Do all registration through
  `ecc.py`.
- **Style.** Follow `pi_capture.py`: small typed functions, `argparse`, one error class carrying an
  exit code, no prints in library functions, and every JSON written with `indent=2`. Never
  overwrite an output file: create it with `open(path, "x")`.
- **Tests.** Use stdlib `unittest` only, with temporary directories. **Never touch the real camera:**
  tests use the fake `rpicam-still` (`capture/tests/fake_rpicam_still.py`). Each builder runs their
  tests in both places and reports both results:

```bash
# GB10 (repo root)
uv run --no-sync python -m unittest discover -s capture/tests -v
# Pi (after deploying, §7)
ssh -o BatchMode=yes ai-pi 'cd ~/cmpe249 && python3 -m unittest discover -s capture/tests -v'
```

Currently, `ecc.py` and its 12 tests pass on both hosts, and so do the 26 `pi_capture` tests.

## 1. Builder 0: `pi_capture.py` 0.3.0 gains `calibrate --lock-config`

### 1.1 Why the change is needed

The lock values in `configs/capture_lock.json` cannot reach the camera through `pi_capture.py`
0.2.0's real CLI. Its `calibrate` always runs auto-exposure and auto-white-balance: `base_cmd`
passes no `--shutter`, `--gain` or `--awbgains`. Only the lens can be set by hand
(`--lens-position`). So `calibrate --lens-position 1.10` would let AE choose the exposure, and
there is no reason it would land on the flicker-safe 25000 µs (final checks §B). `shoot` takes no
control flags at all, by design. `--copy-from` copies only another lock in the same session. The
fix adds a third way to make a lock, and no existing rule is weakened.

### 1.2 Interface (exact)

```
pi_capture.py calibrate --session S --lighting L1 --lock-config PATH [--shoot-settle-ms N]
                        [--quality Q] [--note TEXT] [--recalibrate --reason TEXT] [--dry-run]
```

- `--lock-config` cannot be combined with `--copy-from`, `--lens-position`, `--af-range`, `--size`
  or `--rotation`. Those values come from the file, and rotation is 0. Giving any of them is a
  `CaptureError` (exit 2) that names the flags, using the same wording pattern as `--copy-from`.
- **Reading the file** (stdlib `json`):
  - `schema` must equal `"cmpe249.capture_lock.v1"`.
  - `lock.size` must be a key of `SIZES` (`[2304, 1296]` → `"2304x1296"`).
  - `lock.shutter_us` must be an int > 0.
  - `lock.gain` must be a float > 0.
  - `lock.awb_gains` must be two floats > 0.
  - `lock.lens_position` must be in 0..15.
  - Anything else is exit 2. The sha256 of the file's bytes is recorded.
- **Command.** The calibration still is taken with `locked_cmd(rpicam, {"controls": controls,
  "capture": capture}, tmp_img, tmp_meta)`, the same code path `--copy-from` uses. Manual
  `--shutter`, `--gain`, `--awbgains` and `--lens-position` turn AE, AWB and AF off in
  rpicam-still. The values are:
  - `controls = {"shutter_us": 25000, "gain": 1.381, "awb_gains": [1.5753, 2.5111], "lens_position": 1.1, "lens_source": "lock-config"}`
  - `capture` is built exactly as today, with `--shoot-settle-ms` (default 1500) and `--quality`
    (default 95).
- **Lock entry additions:**
  - `"lock_config": {"path": <abs path>, "sha256": ..., "schema": ..., "lock_version": ...}`
  - `"calibration.exposure": "manual controls from lock config"`
  - `"calibration.lock_check"`: `lock_check(controls, md, None, [iw, ih], [w, h])`, plus a
    `scaler_crop` entry that compares the reported `ScalerCrop` with `lock.scaler_crop`. rpicam
    reports `"(0, 0)/4608x2592"`; other tools give `[0, 0, 4608, 2592]`. Normalise both by taking
    the four integers in order with `re.findall(r"-?\d+", str(v))`.
  - `copied_from: null`
- **When `lock_check.ok` is false at calibration**, the lock is still written, as with
  `--copy-from`, and the tool prints `WARNING: camera did not apply the lock-config controls [...]`.
  `run_arrangement.py` refuses to continue in that case (§3.4).
- **What cannot be enforced** (record it, don't fake it):
  - `FrameDurationLimits` 60000 µs: the command does not set it.
  - The 0.3 → 1.10 lens approach: rpicam-still sets the lens position directly. Hysteresis was
    measured below 0.1 px (final checks §C).
- **Version and tests.** Set `TOOL_VERSION = "0.3.0"`. All 26 existing tests must pass unchanged.
  New tests:
  - The lock has exactly the file's controls, and `lens_source == "lock-config"`.
  - The fake camera reports the requested values, so `lock_check.ok` is true.
  - `FAKE_RPICAM_SHUTTER=20000` → `lock_check.ok` false plus a warning.
  - Each forbidden flag combination → exit 2.
  - Bad schema → exit 2.
  - A later `shoot` uses `--shutter 25000 --gain 1.381 --awbgains 1.5753,2.5111 --lens-position 1.1`
    (check this with `--dry-run` output).
  - `calibrate --lighting L2 --copy-from L1` still works on top of a lock-config L1.
- Update the `capture/README.md` usage block with one example of the new flag.

### 1.3 Addition to `fake_rpicam_still.py` (builder 0)

Add `FAKE_RPICAM_IMAGE=<path>`. When it is set, copy that file's bytes to `-o` instead of writing
the minimal JPEG; every other behaviour stays the same. This lets pose and QC tests feed real,
decodable 2304×1296 JPEGs through `pi_capture`, whose JPEG header check still applies. The current
minimal JPEG cannot be decoded by `cv2.imread`.

## 2. `pose_check.py` (builder 1)

Pose check at the start of a sitting. It registers a fresh P3 still (the sitting's L1 calibration
still) to the golden P3 reference (`ref_final.jpg`) on the static-background mask, and judges the
camera pose.

### 2.1 CLI

```
python3 pose_check.py --image IMG --lock-config PATH --golden PATH
                      [--out JSON] [--flag-px F] [--refuse-px R] [--flag-roll-deg F] [--refuse-roll-deg R]
                      [--min-cc C] [--mode intensity|edges]
```

- **Defaults.** Every threshold and `mode`, `sigma`, `levels`, `iters` and `eps` come from
  `capture_lock.json` (`pose_check.*`, `registration.*`). A CLI value overrides the config value,
  and that threshold's `source` becomes `"cli"`. `run_arrangement.py` never passes overrides, so the
  config stays the single source for study sittings.
- **Default paths** when the flag is omitted:
  - `--lock-config`: `$CMPE249_LOCK_CONFIG`, else `~/cmpe249/configs/capture_lock.json`.
  - `--golden`: `$CMPE249_GOLDEN`, else `~/cmpe249/golden/ref_final.jpg`.

### 2.2 Algorithm (exact)

1. Load the config. Its `schema` must be `cmpe249.capture_lock.v1`.
2. Check the golden file. Its sha256 must equal `reference.sha256`; otherwise exit 3, before any
   registration.
3. Read both images with `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`. If either is `None`, exit 3.
   Both must be `reference.size` (w, h) = 2304×1296; otherwise exit 3.
4. Build `mask = ecc.regions_mask(cfg["masks"]["static_background"]["by_position"]["P3"], w, h)`.
   `ecc.mask_sha256(mask)` must equal `masks.sha256_P3.static_background`; otherwise exit 3. The
   mask is in golden (ref) coordinates.
5. Run the translation: `t = ecc.ecc_translation(golden, img, mask=mask, sigma, iters, eps, mode, levels)`,
   then `shift_px = hypot(t.dx, t.dy)`.
6. Run the roll: `e = ecc.ecc_euclidean(golden, img, mask=mask, ...)`, with the same parameters.
   Use `e.rot_deg` for roll. Report `e.dx` and `e.dy` too, but judge the shift on `t`.
7. If either call raises `ecc.EccError`, or `t.cc < min_cc`, the verdict is `"error"` (exit 12).
8. Otherwise:
   - `refuse` if `shift_px > refuse_px` or `|rot_deg| > refuse_roll_deg` (exit 11);
   - else `flag` if `shift_px > flag_px` or `|rot_deg| > flag_roll_deg` (exit 10);
   - else `ok` (exit 0).
   All comparisons are strict.
9. List every rule that fired in `reasons` (e.g. `"shift 2.81 px > flag 2.0 px"`).

### 2.3 Output

The JSON is always printed to stdout. With `--out` it is also written to that file, created
exclusively (an existing file → exit 3). Schema `cmpe249.pose_check.v1`:

```json
{
  "schema": "cmpe249.pose_check.v1",
  "tool": {"name": "pose_check.py", "version": "1.0.0", "ecc_version": "1.0.0",
           "python": "3.13.5", "opencv": "4.10.0", "numpy": "2.2.4", "hostname": "ai-pi"},
  "time_utc": "2026-10-06T17:00:00.000+00:00",
  "image":  {"path": "/abs/.../calibration/L1-v1.jpg", "sha256": "...", "size": [2304, 1296]},
  "golden": {"path": "/abs/.../ref_final.jpg", "sha256": "...", "size": [2304, 1296]},
  "lock_config": {"path": "...", "sha256": "...", "lock_version": 1},
  "mask": {"name": "static_background", "position": "P3", "sha256": "...", "usable_px": 1010820},
  "params": {"mode": "edges", "sigma": 1.5, "levels": 3, "iters": 200, "eps": 1e-06},
  "translation": {"dx": -0.282, "dy": -2.794, "shift_px": 2.808, "cc": 0.9459},
  "euclidean": {"dx": -0.291, "dy": -2.839, "rot_deg": -0.0116, "cc": 0.9460},
  "thresholds": {"flag_px": 2.0, "refuse_px": 10.0, "flag_roll_deg": 0.5, "refuse_roll_deg": 2.0,
                 "min_cc": 0.8, "source": {"flag_px": "config", "...": "config"}},
  "verdict": "flag",
  "reasons": ["shift 2.81 px > flag 2.0 px"],
  "exit_code": 10
}
```

The numbers above are the real values for frame e against `ref_final`, measured on both hosts (§8).
Sign convention: `dx`/`dy` are where the golden's content appears in the new image (+x right,
+y down). A positive `rot_deg` is clockwise on screen.

### 2.4 Exit codes

| Code | Meaning |
| --- | --- |
| 0 | ok |
| 2 | usage (argparse) |
| 3 | bad input (missing or unreadable file, sha256 or size mismatch, bad config, `--out` exists) |
| 10 | flag |
| 11 | refuse |
| 12 | registration error or unreliable (`EccError`, cc < min_cc) |

Code 1 must never be produced deliberately. An uncaught exception yields 1, and
`run_arrangement.py` treats that as an error.

### 2.5 Required tests

The tests must also run on the Pi, where `ref_final.jpg` is not present. So they synthesise their
own data:

- A 2304×1296 textured golden JPEG, written with `cv2.imwrite` at quality 95.
- A temporary `capture_lock.json` derived from the real one, with the golden's sha256 and the
  recomputed mask sha256 filled in.
- Shifted test images made with an exact Fourier shift of a padded canvas, as `test_ecc.shifted`
  does. Do not use `cv2.warpAffine`; `test_ecc.py`'s docstring explains why.

Cases:
- 0.0 px → `ok`
- 3.0 px → `flag` (10)
- 12 px → `refuse` (11)
- 1.0° rotation → `flag`
- 2.5° rotation → `refuse`
- wrong golden sha → 3
- wrong image size → 3
- an `--out` that exists → 3
- a flat grey image → 12
- CLI override recorded as `"cli"`

Plus one GB10-only test, `skipUnless` the repo file exists: `ref_final.jpg` against itself gives
`ok` with shift < 0.01 px; `geometry/rpi_view_2026-10-05e.jpg` gives `flag`.

## 3. `run_arrangement.py` (builder 2)

A guided sequencer, stdlib only. It **calls `pi_capture.py` for every capture and never bypasses
it**: no direct `rpicam-still`, and no writes to `lock.json`, `manifest.jsonl`,
`session_log.jsonl`, `calibration/` or the image tree.

### 3.1 Session files it owns

All of them live in `<root>/<session>/run/`. The lowercase name can never collide with an
arrangement id.

| File | Content |
| --- | --- |
| `run_log.jsonl` | Append-only. Every prompt, response, shot, lamp toggle, preflight result and abort; one JSON object per line, written and fsynced as it happens. |
| `sitting.json` | The sitting record (§3.4). Written once. |
| `pose_check_<lock_id>.json` | `pose_check.py --out` output. |
| `<ARR>.json` | The arrangement manifest (§3.6). Rewritten atomically after each slot (temporary file + `os.replace`). |
| `.run_arrangement.lock` | `fcntl.flock` (non-blocking): one sequencer per session. |

### 3.2 Common options

| Option | Default | Notes |
| --- | --- | --- |
| `--root` | `$CMPE249_DATA_ROOT` or `~/cmpe249/data` | Same as `pi_capture` |
| `--session` | required | Validated with `pi_capture`'s `RE_SESSION` pattern, copied as a literal, not imported |
| `--pi-capture` | `<dir of this file>/pi_capture.py` | |
| `--pose-check` | `<dir of this file>/pose_check.py` | Tests point it at `fake_pose_check.py` |
| `--python` | `sys.executable` | Interpreter for both subprocesses |
| `--rpicam` | none | Hidden. Passed through to every `pi_capture` call as `--rpicam X` |
| `--timedatectl` | `$TIMEDATECTL` or `timedatectl` | Hidden. Tests point it at `fake_timedatectl.py` |
| `--script` | none | JSON list of responses for non-interactive runs (§3.7) |
| `--fake-camera` | off | Sets `--rpicam` to `capture/tests/fake_rpicam_still.py`. **Refused (exit 2) when `--root` resolves to the default real data root**, so dry runs can never write into real sessions. |
| `--lamp-settle-s` | 15 | Minimum time between confirming a lamp toggle and the next shot ([A4]) |

### 3.3 NTP preflight

The `sitting` and `arrangement` subcommands both run the NTP preflight before anything else:

```
[timedatectl, "show", "-p", "NTPSynchronized", "--value"]   (timeout 10 s)
```

It passes only if the exit code is 0 **and** `stdout.strip() == "yes"`. Anything else, including
a missing binary or a timeout, refuses with exit 20. Every outcome is logged:
`{"event": "ntp_preflight", "stdout": ..., "returncode": ..., "ok": ...}`.

### 3.4 `sitting`: start a session at P3

```
python3 run_arrangement.py sitting --session S [--lock-config PATH] [--golden PATH]
                                   [--lightings L1,L2] [--note TEXT]
```

The default `--lock-config` and `--golden` are those of §2.1. `--lightings` defaults to `L1,L2`;
the first label is the primary lighting. Labels must match `pi_capture`'s `RE_TOKEN`.

1. Run the NTP preflight (exit 20 on failure).
2. If `run/sitting.json` exists, exit 24 ("session already has a sitting; start a new session id").
   If the session already has a `lock.json`, also exit 24: this tool makes the session's locks itself.
3. Prompt: `Camera at P3 (floor mark), lamps set for L1. Enter = calibrate, q = abort`.
4. Run `[python, pi_capture, "calibrate", "--root", root, "--session", S, "--lighting", "L1", "--lock-config", lock_config]`.
   - Exit ≠ 0 → exit 23.
   - Then read `<session>/calibration/L1-v1.json`. If `lock.calibration.lock_check.ok` is not
     true, exit 24 (camera did not apply the lock).
5. Run `[python, pose_check, "--image", <session>/calibration/L1-v1.jpg, "--lock-config", lock_config, "--golden", golden, "--out", run/pose_check_L1-v1.json]`
   with `timeout=180` s (a timeout counts as an error, exit 21).
   - Exit 0 → continue.
   - Exit 10 → prompt `Pose FLAG: <reasons>. c = continue (asks for a note), q = abort`. The note
     is required and recorded.
   - Exit 11 → exit 21 ("re-seat the camera on the P3 mark, then start a NEW session id").
   - Exit 12, 3 or any other code → exit 21.
   - Pose is never re-checked by recalibrating: this tool never passes `--recalibrate`.
6. For each further lighting Lk:
   - Prompt `Keep the camera at P3. Switch lamps to Lk. Enter when switched`.
   - Wait `--lamp-settle-s` from that response.
   - Run `[python, pi_capture, "calibrate", "--root", root, "--session", S, "--lighting", Lk, "--copy-from", "L1"]`.
   - Apply the same `lock_check.ok` check to `calibration/Lk-v1.json` (exit 24).
7. Write `run/sitting.json`:

```json
{"schema": "cmpe249.sitting.v1", "tool": {"name": "run_arrangement.py", "version": "1.0.0"},
 "session": "S", "created_utc": "...", "created_local": "...", "hostname": "ai-pi",
 "ntp": {"stdout": "yes", "returncode": 0, "ok": true},
 "lock_config": {"path": "...", "sha256": "...", "lock_version": 1},
 "golden": {"path": "...", "sha256": "..."},
 "lightings": ["L1", "L2"],
 "locks": {"L1": "L1-v1", "L2": "L2-v1"},
 "pose": {"file": "run/pose_check_L1-v1.json", "verdict": "ok|flag", "exit_code": 0,
          "accepted_flag_note": null},
 "note": null}
```

### 3.5 `arrangement`: guided capture of one arrangement

```
python3 run_arrangement.py arrangement --session S --arrangement A01 --ordering A|B
                                       [--stops P3,P1,P2,P4,P5] [--card PATH] [--resume]
python3 run_arrangement.py plan --arrangement A01 --ordering A|B [--stops ...] [--lightings L1,L2]
                                # prints the slot list as JSON; no session, camera or NTP needed
```

**Preconditions.** Any failure exits 24, except NTP (exit 20).
- The NTP preflight passes (exit 20 otherwise).
- `run/sitting.json` exists, and its pose verdict is `ok`, or `flag` with an accepted note.
- Each lighting in `sitting.lightings` has a current lock in `lock.json`, and its lock id equals
  `sitting.locks[...]`.
- The arrangement id matches `^(A\d{2}|PILOT\d{2})$`. That is a subset of `pi_capture`'s pattern;
  other ids are refused.
- The arrangement has no rows in `manifest.jsonl` and no `run/<ARR>.json`, unless `--resume` is
  given (§3.6).
- `--stops` is a permutation of P1..P5 that **starts with P3**. The default is P3 followed by the F
  order in `configs/positions.json`: `P3,P1,P2,P4,P5`.
- If a `--card` is given, its `arrangement` equals `--arrangement`.

**Plan.** The plan is deterministic. Phases run in this fixed order (EXPERIMENT_PLAN "Reference
and label policy": references are collected before the scene changes; `pi_capture` refuses
references after any query):

| # | phase | role | condition |
| --- | --- | --- | --- |
| 1 | `reference` | reference | visible |
| 2 | `visible` | query | visible |
| 3 | `occluded` | query | occluded |
| 4 | `removed` | query | removed |

With lightings `[L1, L2]` and stops `S[0..4]`:
- **Ordering A** (both lightings at each stop, lamp order alternating by stop): keep one stop-visit
  counter `v` that runs across all four phases (0..19). At visit v the lightings are taken in order
  when v is even and reversed when v is odd. Phase 1 is then P3 L1, L2 · P1 L2, L1 · P2 L1, L2 ·
  P4 L2, L1 · P5 L1, L2, and phase 2 starts at P3 with L2, L1. This gives exactly 5 lamp toggles per
  phase and none at phase boundaries: **20 toggles per arrangement**, as in PLAN §4.4. The pair at a
  stop shares one camera placement.
- **Ordering B** (lighting as the outer loop inside each phase): phase p walks all stops under one
  lighting, then all stops under the other. The lighting order alternates by phase (phase 1: L1
  then L2; phase 2: L2 then L1; and so on). That gives 1 toggle per phase and none at the
  boundaries: **4 toggles per arrangement**. PLAN Appendix C says "2 toggles" for ordering B. That
  would need lighting as the outer loop across phases, which the fixed reference → … → removed
  order rules out (the target cannot be put back after the removed sweep). Builders implement 4
  and leave the discrepancy to the main session.
- With one lighting there are no toggles.

One phase is 5 × len(lightings) slots, so 40 slots at L = 2 and 20 at L = 1 (PLAN §4.4: 5L
references + 15L queries). Lamp prompts:
- At the start of the arrangement there is always one **lamp-set** prompt
  (`Set lamps to <first lighting>`), which is not counted as a toggle.
- After that, a **lamp-toggle** prompt comes whenever the next slot's lighting differs from the
  previous shot's.
- The manifest counts only the toggles, in `lamp_toggles`.

**Prompts and keys.** Every prompt is logged with `time_utc`, `time_local`, `monotonic_s`, `slot`
(index or null) and `text`. Every response is logged with the key and `elapsed_s`.

- **Phase start:** `PHASE <n>/4 <phase>: <card text for this phase, or the default text>. Enter when the scene is set, q = abort`.
  The default texts are:
  - reference: "Arrange the scene; target in place; occluder parked"
  - visible: "Scene unchanged except as the card says"
  - occluded: "Place the occluder as the card says"
  - removed: "Remove the target; keep the occluder exactly as in occluded"
- **Lamp toggle:** `Switch lamps to <L>. Enter when switched`. The next shot is refused until
  `--lamp-settle-s` has passed since that response. The interactive prompt shows a countdown; in
  `--script` mode the tool sleeps.
- **Shot:** `[slot i/N] <phase> <P> <L>: move camera to <P>. Enter = shoot, q = abort`. When the
  stop is unchanged from the previous slot, it reads `camera stays at <P>`.
- **After each successful shot:** `saved <stem>  <pi_capture WARNING lines, if any>. Enter = next, r = retake, q = abort`.
  - **Failed lock or lighting check** (decided after review finding RA-LOCK-FAIL-ACCEPTED). When
    the shot's `manifest.jsonl` row has `lock_ok: false` (the camera did not apply the session lock;
    QC `lock_check_failed` = fail) or `light_ok: false` (Lux outside pi_capture's guard vs the
    calibration; QC `lux_vs_calibration` = flag), plain Enter is **not** accepted. The prompt reads
    `saved <stem>  <warnings>. CHECK FAILED: lock|lighting (QC will FAIL|flag this shot). r = retake, a = accept anyway (asks for a note), q = abort`.
    `a` asks `Note: why accept this shot despite the failed check?` (non-empty) and stores
    `{"failed_checks": [...], "note": ...}` in the take's `accepted_despite` (null otherwise). This
    keeps one rule for both cases: `sitting` still refuses a calibration whose lock was not applied
    (exit 24), and an arrangement never accepts such a shot silently. QC still fails/flags it.
  - `r` asks `Retake reason:` (non-empty) and reshoots the same slot with `--take <last+1> --note "retake: <reason>"`.
  - A slot's **selected take** is the last take the operator accepted with Enter.
  - Retakes are only possible here, while the camera is still at that stop. That keeps every retake
    inside its own phase, so `pi_capture`'s reference-before-query rule never needs an override.
- **`q`** at any prompt: log an `abort` event, mark the manifest `"status": "aborted"`, exit 22.
  Data already saved stays; `pi_capture` never deletes anything.

**The `pi_capture` call (exact; the only call shape `arrangement` uses):**

```python
cmd = [python, pi_capture, "shoot", "--root", root, "--session", session,
       "--arrangement", arr, "--position", pos, "--role", role,           # role: reference | query
       "--condition", cond, "--lighting", lighting, "--take", str(take)]  # cond: visible | occluded | removed
if note:  cmd += ["--note", note]          # retakes only: "retake: <reason>" (<= 500 chars)
if rpicam: cmd += ["--rpicam", rpicam]     # tests / --fake-camera only
subprocess.run(cmd, capture_output=True, text=True, timeout=120)
```

**Never emitted, by construction and by test:**
- `--condition pilot`
- `--allow-late-reference`
- `--accept-lock-change`
- `--episode` (an episode is (arrangement, condition, lighting) and is already in the stem)
- `--recalibrate`
- `--dry-run`

The test suite asserts that none of these strings appears in any logged command.

**`pi_capture` exit codes:**

| Exit | Action |
| --- | --- |
| 0 | Find the `manifest.jsonl` row whose `stem` is `<arr>_<pos>_<role>_<cond>_<L>_t<take:02d>`. Missing → exit 23. Copy it into the slot. Lines starting `WARNING:` in stdout are shown and stored. |
| 5 (camera) | Show stderr. Prompt `Enter = retry the same take, q = abort`. Nothing was saved, so the same take number is correct. |
| 4 with stderr containing `process is using session` (session-lock contention) | Show stderr. Prompt `another pi_capture.py is using this session, nothing was saved. Enter = wait for it and retry, q = abort`. On Enter, wait for `<session>/.session.lock` (below), recompute the take number from `manifest.jsonl`, and retry. |
| 3 (overwrite), 4 (any other protocol refusal), 6 (no lock), 2 (usage), any other | Show stderr, log it, exit 23. Never retry with an override. |

**Waiting for the session lock.** `pi_capture` holds an exclusive `flock` on
`<session>/.session.lock` while it shoots. If the sequencer is killed (SIGKILL, closed terminal)
during a shot, its `pi_capture` child keeps running and saves its shot. Therefore `arrangement`
(fresh or `--resume`) first waits until that lock is free (it takes the lock non-blocking in a
loop, releases it at once, and never shoots while holding it), logging `wait_session_lock` /
`session_lock_free`; it gives up with exit 24 after 150 s (pi_capture's own 120 s timeout + 30 s).
Only then does it read `manifest.jsonl`, so `--resume` adopts the orphan's shot (review finding
RA-RESUME-ORPHAN; before this, an immediate `--resume` exited 23). After a SIGKILL the run file
stays `"status": "in_progress"` (nothing can mark it); `--resume` and QC's `run_incomplete` handle
that state. `run/<ARR>.json` is written by tmp + `os.replace` + fsync of the file and directory.

### 3.6 Arrangement manifest `run/<ARR>.json` (schema `cmpe249.run_arrangement.v1`)

The slot and take field names are the ones in `pi_capture`'s `manifest.jsonl`, copied verbatim
from its rows. Consumers can join the two on `stem`.

```json
{
  "schema": "cmpe249.run_arrangement.v1",
  "tool": {"name": "run_arrangement.py", "version": "1.1.0", "python": "3.13.5"},
  "session": "S", "arrangement": "A01", "ordering": "A",
  "stops": ["P3", "P1", "P2", "P4", "P5"], "lightings": ["L1", "L2"],
  "phases": ["reference", "visible", "occluded", "removed"],
  "card": {"path": "...", "sha256": "..."},
  "sitting_sha256": "...", "lock_config_sha256": "...",
  "started_utc": "...", "finished_utc": null, "status": "in_progress|complete|aborted",
  "lamp_toggles": 0, "n_prompts": 0,
  "slots": [
    {"index": 0, "phase": "reference", "role": "reference", "condition": "visible",
     "position": "P3", "lighting": "L1",
     "takes": [{"take": 1, "stem": "A01_P3_reference_visible_L1_t01",
                "path": "A01/reference/P3/A01_P3_reference_visible_L1_t01.jpg",
                "sha256": "...", "lux": 201.9, "lock_ok": true, "light_ok": true,
                "warnings": [], "note": null, "accepted_despite": null, "pi_capture_exit": 0,
                "prompt_utc": "...", "shot_started_utc": "...", "shot_finished_utc": "..."}],
     "selected_take": 1}
  ]
}
```

`--resume` reloads this file, checks that its plan (ordering, stops, lightings) equals the
requested one (otherwise exit 24), and continues at the first slot whose `selected_take` is null.
Resuming into an earlier phase than one that already has shots is impossible by construction,
because slots are captured in plan order.

### 3.7 Script mode (`--script FILE`)

The file is a JSON list of strings, consumed one per prompt in order:
- `""` = Enter
- `"q"` = abort
- `"r"` = retake
- `"c"` = continue after a pose flag
- any other string answers a free-text prompt (retake reason, pose-flag note)

Each consumed response is logged with `"source": "script"`. If the list runs out before the tool
finishes, it logs the event and exits 25. The prompt text still goes to stdout, so tests can
assert on it.

### 3.8 Exit codes

| Code | Meaning |
| --- | --- |
| 0 | done (`plan`, `sitting` or `arrangement` complete) |
| 2 | usage |
| 20 | NTP not synchronised, or `timedatectl` failed |
| 21 | pose check refused or errored |
| 22 | operator abort (`q`) |
| 23 | a `pi_capture` call failed or refused (stderr is logged) |
| 24 | precondition failed |
| 25 | script exhausted |

### 3.9 Required tests

The fakes:
- `fake_timedatectl.py` prints `$FAKE_NTP` (default `yes`) and exits `$FAKE_NTP_RC` (default 0).
- `fake_pose_check.py` writes a minimal `cmpe249.pose_check.v1` JSON to `--out` with
  `verdict = $FAKE_POSE_VERDICT` (default `ok`) and exits 0, 10, 11 or 12 to match.
- Real `pi_capture.py` with the fake `rpicam-still`, in a temporary root.

Cases:
1. NTP `no`, non-zero rc, or missing binary → 20, and nothing is written except `run_log.jsonl`.
2. A full `sitting` + `arrangement` under ordering A and under ordering B at L = 2. Check:
   - 40 shots;
   - the exact slot order, compared against `plan` output and a hand-written expected list for the
     first 6 slots of each ordering;
   - `lamp_toggles == 20` under ordering A and `== 4` under ordering B, plus one lamp-set prompt
     each;
   - `pi_capture status --json` shows 10 references and 30 queries.
3. L = 1 (`--lightings L1`): 20 shots, no lamp prompts.
4. Retake: `r` + reason → take 2 exists, `selected_take == 2`, and the note is in the sidecar.
5. Pose `flag` with `c` + note → continues. Pose `refuse` → 21, with no L2 calibration.
6. `q` mid-phase → 22, `status: aborted`, then `--resume` completes.
7. `FAKE_RPICAM_FAIL=1` on one shot → retry prompt, then success.
8. Forbidden flags never appear in any logged command (scan `run_log.jsonl`).
9. `--stops P1,P3,...` (not starting with P3) → 24. A second `sitting` → 24. `--fake-camera` with
   the default root → 2.
10. Script exhausted → 25.
11. (review fixes) `FAKE_RPICAM_SHUTTER=20000` on a shot → Enter refused, `a` + note accepts and
    is stored in `accepted_despite`; `FAKE_RPICAM_LUX=80` → the same prompt for `lighting`.
12. (review fixes) A slow pi_capture "orphan" holding the session lock → `--resume` waits, adopts
    its take and completes; a lock held mid-run → exit 4, prompt, wait, retry (rcs `[4, 0]`); any
    other exit 4 → 23.

## 4. `qc_session.py` (builder 3)

Quality control for one session. It reads `pi_capture` and `run_arrangement` output and writes a
JSON report and a short markdown summary. **It computes pixel statistics and registration only. It
runs no detector or matcher and writes nothing about target state**, so it may run on test
sessions while test capture is still going on (PLAN module 1: "on test sessions, no perception on
query images until test capture stops").

### 4.1 CLI

```
qc_session.py --session-dir DIR [--lock-config PATH] [--arrangement A01 ...]
              [--out-dir DIR] [--no-images] [--lux-guard X] [--shift-flag-px X]
              [--saturation-level N] [--saturation-flag-share X]
```

- `--lock-config` defaults as in §2.1. Thresholds default to the config's `qc.*`; overrides are
  recorded.
- `--out-dir` defaults to `<DIR>/qc/`. The outputs are `qc_<UTC yyyymmddTHHMMSSZ>.json` and `.md`,
  created exclusively and never overwritten.
- `--no-images` skips registration and saturation; every other check still runs.

### 4.2 Checks

Each problem becomes an entry in `flags`:

```
{"code": ..., "severity": "fail"|"flag"|"info", "arrangement", "condition", "lighting", "position", "stem", "detail"}
```

The **selected take** of a slot is the one in `run/<ARR>.json` if that file exists; otherwise it is
the highest take in `manifest.jsonl`.

| Code | Severity | Rule |
| --- | --- | --- |
| `file_inventory` | – | sha256 and bytes of every regular file under DIR, excluding the `qc/` subtree and the two lock files. Hidden `.*.partial*` files raise `stale_partial` (flag). |
| `missing_image` / `sha_mismatch` | fail | Each manifest row's image exists, and its sha256 equals the manifest's and the sidecar's `image.sha256`. |
| `lock_check_failed` | fail | The sidecar's `lock_check.ok` is not true. The key is missing on 0.1 sidecars, which is a flag, not a fail. |
| `lock_config_mismatch` | fail | The sidecar's `rpicam_metadata` must match `capture_lock.json` `lock`. `ExposureTime` within `shutter_rel` of `shutter_us`; `AnalogueGain × DigitalGain` within `gain_rel` of `gain`; `ColourGains` within `awb_abs`; `LensPosition` within `lens_abs`; normalised `ScalerCrop` equal to `scaler_crop` (§1.2); image `width`/`height` equal to `size`. |
| `lux_missing` | flag | No `Lux` in the metadata. |
| `lux_vs_calibration` | flag | `lighting_check.ratio` outside [1/guard, guard]. |
| `lux_vs_reference` | flag | A query's Lux divided by its own selected reference's Lux (same arrangement, position and lighting) is outside [1/guard, guard]. |
| `missing_slot` | fail | For each arrangement and each lighting in the session's `lock.json` `locks`: references at P1..P5 (condition visible), and queries visible/occluded/removed at P1..P5. That is 10 + 30 at L = 2. |
| `pilot_condition` | fail | Any `condition == "pilot"` row in an `A\d\d` arrangement. In `PILOT\d\d`, flag only. |
| `protocol_override` | fail | Any sidecar `protocol.overrides` non-empty (late reference, lock change). |
| `retake` | info | A slot has more than one take (expected rows only). |
| `unexpected_shot` | fail in `A\d\d`, flag otherwise | A manifest row whose (arrangement, position, role, condition, lighting, episode) is not one of the expected slots of `missing_slot` (episode must be null; lighting must be in `lock.json`). Such rows are never selected, registered or counted in `by_condition`; `shots[].unexpected` is true and `counts.unexpected` counts them. Rows with condition `pilot` are left to `pilot_condition`. |
| `unrecorded_take` | flag | `run/<ARR>.json` exists, and a manifest row of that arrangement is not among its takes (matched on `stem`, or on slot + take number). A bare `pi_capture` shot, or an orphaned shot that `--resume` has not adopted yet. |
| `stray_file` | flag | A file under `<ARR>/` that no manifest row names: not `<stem>.jpg`, `<stem>.json` or a camera log `<stem>[.<time>].log` of a row. Hidden `.partial` files are `stale_partial` instead. |
| `failed_capture_log` | info | `<stem>.failed-<time>.log` of a stem with no row (a failed still; nothing saved). |
| `shift` | flag | Per query slot: `ecc.ecc_translation(ref, query, mask=M, **registration)`, where ref is that slot's selected reference at the same position and lighting. M is `static_background.by_position[P]` if not null (its sha256 checked as in §2.2; P3 only for now). Otherwise use `ecc.ecc_translation_robust(ref, query)` with the default 24 px border and `mask_source="auto_residual"`. Flag if `shift_px > shift_flag_px`. `EccError` → `registration_failed` (flag). |
| `ref_vs_golden` | info | P3 references only: shift against the golden, as in pose_check. Needs `--golden`, default as in §2.1. Skipped with an info flag if the golden is unavailable. |
| `saturation` | flag | Per image: share of pixels with `max(B, G, R) >= saturation_level` inside the region `mug_region.by_position[P]` minus `glare.by_position[P]`. If the region is null, use the full frame (`region_source="full_frame_fallback"`). Flag if the share exceeds `saturation_flag_share`. The glare box share is reported separately as info. |
| `no_sequencer_log` | flag | `run/sitting.json` is missing: the shots were taken with bare `pi_capture`. |
| `ntp_not_synced` | fail | `sitting.ntp.ok` is not true. |
| `pose_flag` / `pose_missing` | flag / fail | The sitting's pose verdict was `flag` (accepted), or no pose record exists while `sitting.json` does. |

Read images with `cv2.imread(path, cv2.IMREAD_COLOR)`; it returns BGR. Registration uses the
`registration.*` parameters from the config.

### 4.3 JSON report (`cmpe249.qc_session.v1`)

```json
{
  "schema": "cmpe249.qc_session.v1",
  "tool": {"name": "qc_session.py", "version": "1.1.0", "ecc_version": "1.0.0", "python": "...",
           "opencv": "...", "numpy": "...", "hostname": "..."},
  "time_utc": "...", "session": "S", "session_dir": "/abs/...",
  "lock_config": {"path": "...", "sha256": "...", "lock_version": 1},
  "params": {"lux_guard": 1.25, "shift_flag_px": 2.0, "saturation_level": 250,
             "saturation_flag_share": 0.001, "registration": {"mode": "edges", "sigma": 1.5, "levels": 3},
             "images": true, "sources": {"lux_guard": "config"}},
  "sitting": {"present": true, "ntp_ok": true, "pose_verdict": "ok", "pose_shift_px": 0.21},
  "files": [{"path": "A01/reference/P3/....jpg", "bytes": 612345, "sha256": "..."}],
  "shots": [{"stem": "...", "path": "...", "arrangement": "A01", "position": "P3", "role": "query",
             "condition": "occluded", "lighting": "L1", "take": 1, "selected": true,
             "sha256_ok": true, "lock_ok": true, "lock_config_ok": true, "lux": 201.2,
             "lux_ratio_calibration": 0.99, "lux_ratio_reference": 1.01,
             "saturation_share": 0.00002, "saturation_region": "mug_region",
             "glare_share": 0.49,
             "registration": {"against": "<ref stem>", "dx": 0.05, "dy": -0.11, "shift_px": 0.12,
                              "cc": 0.998, "mask_source": "static_background|auto_residual"}}],
  "completeness": {"A01": {"expected": 40, "present": 40, "missing": []}},
  "by_condition": [{"arrangement": "A01", "condition": "occluded", "lighting": "L1", "n": 5,
                    "lux_ratio_reference": {"median": 1.0, "min": 0.98, "max": 1.02},
                    "shift_px": {"median": 0.1, "max": 0.2}, "saturation_share_max": 0.0001,
                    "flags": {"shift": 0}}],
  "by_condition_all": [{"condition": "occluded", "n": 10, "lux_ratio_reference_median": 1.0,
                        "shift_px_median": 0.1}],
  "flags": [{"code": "shift", "severity": "flag", "...": "..."}],
  "verdict": "pass|flag|fail",
  "exit_code": 0
}
```

`by_condition_all` exists for the PLAN §4.4 condition-order/time confound: the removed sweep is
always last, so a trend across conditions in Lux or shift is reported, not hidden.

The markdown file holds:
- the verdict line;
- counts of shots, expected, missing and flags by severity;
- one table with a row per arrangement × condition × lighting (n, median Lux ratio vs reference,
  median and max shift, max saturation, flag count);
- up to 30 flag lines, then "… N more in the JSON".

### 4.4 Exit codes

| Code | Meaning |
| --- | --- |
| 0 | pass (no flag or fail) |
| 10 | flags only |
| 11 | at least one fail |
| 2 | usage |
| 3 | bad input (no `manifest.jsonl`, bad config) |

### 4.5 Required tests

Synthetic sessions only, built in temporary directories through the **real** `pi_capture.py` with
the fake camera and `FAKE_RPICAM_IMAGE` (§1.3). Feed it 2304×1296 textured JPEGs written by
`cv2.imwrite`. Shifted queries use the Fourier-shift helper pattern from `test_ecc.py`.

Cases:
- A complete L = 1 session → pass. This needs a test lock config with the synthetic golden; ideally
  the static mask and mug region are null for P1, P2, P4 and P5, to exercise the fallbacks.
- One query shifted 3 px → `shift` flag (10).
- One query with a saturated block in the mug region → `saturation` flag.
- `FAKE_RPICAM_LUX` 1.5× on one query → `lux_vs_calibration` and `lux_vs_reference`.
- A deleted image and a byte-edited image → `missing_image` / `sha_mismatch` (11).
- A missing slot → `missing_slot`.
- No `run/` directory → `no_sequencer_log`.
- `FAKE_RPICAM_SHUTTER=20000` → `lock_check_failed` and `lock_config_mismatch`.
- `--no-images` skips registration and saturation.
- (review fix QC-EXTRA-SHOTS) After a complete run file, two bare `pi_capture` shots (an
  `--episode E1` query and take 7 of an existing slot) plus a stray file → `unexpected_shot` (fail,
  not selected, not registered, `by_condition` n unchanged), `unrecorded_take` (flag), `stray_file`
  (flag), and `failed_capture_log` (info) for a failed-still log.

GB10 integration check, reported and not a unit test: run it on `data/raw/pilot/2026-09-26-pilot`.
That PILOT00 data comes from another rig and lock, so it must exit 11 with `lock_config_mismatch`
and `missing_slot`. That is the expected result, and it shows QC catches a wrong lock.

## 5. `capture/ecc.py` (done): interface summary

```python
ecc.ecc_translation(ref, img, mask=None, sigma=1.5, iters=200, eps=1e-6, mode="intensity",
                    levels=3, init=(0, 0), border=24, refine=True) -> Shift(dx, dy, cc)
ecc.ecc_euclidean(ref, img, mask=None, ...same..., border=24) -> Rigid(dx, dy, rot_deg, cc)   # dx/dy at image centre
ecc.ecc_translation_robust(ref, img, mask=None, block=32, k=4.0, grow=1, **kw) -> (Shift, mask_used)
ecc.residual_mask(ref, img, dx, dy, mask=None, block=32, k=4.0, sigma=1.5, grow=1) -> uint8 mask
ecc.gradient_magnitude(img, sigma=1.5) -> float32
ecc.regions_mask(spec, width, height) -> uint8 0/1   ecc.fill_polygon(points, w, h) -> bool
ecc.mask_sha256(mask) -> str   ecc.to_gray_f32(img) -> float32 copy   ecc.warp_to_ref(img, dx, dy)
ecc.EccError (RuntimeError)    ecc.ECC_VERSION == "1.0.0"
```

- **Conventions:**
  - `(dx, dy)` is where ref's content appears in img;
  - masks are in ref coordinates, nonzero = usable;
  - inputs are copied (float32) before any OpenCV call, and are never modified (tested, including
    read-only arrays);
  - BGR input is converted to grey.
- **How it works:**
  1. Coarse-to-fine OpenCV ECC (internally img is the template and ref the warped input, so that the
     4.10 `inputMask` is the ref-coordinate mask).
  2. A full-resolution numpy Gauss-Newton refinement for translations.
- **Accuracy on non-circular synthetic shifts, 30 random shifts within ±3 px:**
  - max error 0.0026 px (intensity mode) and 0.0060 px (edges mode), the same on OpenCV 4.10 and 5.0;
  - without the refinement step, OpenCV 4.10 had a masked median error of 0.012 px.
- `mode="edges"` registers the gradient magnitude. The config uses it **for shadows and spatially
  varying light, not for global exposure/tone changes** (review finding ECC-EDGES-ROBUSTNESS-CLAIM;
  the earlier "more tolerant of exposure" wording was not supported). Measured with
  `ecc_translation(..., levels=3)` on the P3 static mask, real frames `ref_final` vs
  `lp1.10/locked_00` (same lock) with a synthetic change applied to the query; "dev" is the move
  from the untouched pair's estimate. GB10 (OpenCV 5.0) and Pi (OpenCV 4.10) printed identical
  numbers: `data/raw/2026-10-05_shelf-final/masks/check_tone_modes.py` and
  `check_tone_modes_{gb10,pi}.txt`.

  | change on the query | intensity dev / cc | edges dev / cc |
  | --- | --- | --- |
  | none | 0 / 0.9999 | 0 / 0.9981 |
  | ×0.7 + 10 (linear) | 0.000 / 0.9999 | 0.001 / 0.9977 |
  | gamma 0.6 | 0.058 / 0.9971 | 0.034 / 0.9901 |
  | gamma 1.5 | 0.020 / 0.9967 | 0.029 / 0.9875 |
  | gamma 0.6 ×1.3, clipped | 0.092 / 0.9767 | 0.138 / 0.9570 |
  | ×1.5, clipped | 0.018 / 0.9887 | 0.112 / 0.9808 |
  | soft shadow (−45 % Gaussian blob, left side) | 0.126 / 0.9477 | 0.007 / 0.9957 |

  On the synthetic texture the pattern is the same (45 % shadow: intensity 0.11 px off, edges
  0.004 px; gamma 0.6 ×1.4 clipped: intensity 0.005, edges 0.02 px, edges cc 0.75). All deviations
  are far below the 2 px flag. Edges' cc falls faster under clipping, which matters for
  `pose_check.min_cc` (0.80). `TestToneAndShadow` in `tests/test_ecc.py` pins this behaviour. No
  real L1-vs-L2 or real-shadow pair exists yet: compare both modes on the first sitting's L1 and L2
  calibration stills before the mode is frozen.
- Polygons are rasterised by `fill_polygon` and not by `cv2.fillPoly`, because 4.10 and 5.0 filled
  the same polygon differently (different mask sha256). The mask hashes in `capture_lock.json` are
  verified equal on both hosts.

## 6. `configs/capture_lock.json` (done): keys consumers use

| Key | Used by |
| --- | --- |
| `schema`, `lock_version` | all |
| `lock.{shutter_us, gain, awb_gains, lens_position, size, scaler_crop, tolerances}` | pi_capture 0.3.0, qc |
| `reference.{path, sha256, size, lux}` | pose_check, qc (`path` is repo-relative; on the Pi pass `--golden`) |
| `masks.static_background.by_position.P*` (null = not drawn yet) | pose_check (P3), qc |
| `masks.mug_region.by_position.P*`, `masks.glare.by_position.P*` | qc |
| `masks.sha256_P3.*` | pose_check, qc (refuse on mismatch) |
| `registration.{mode, sigma, levels, iters, eps}` | pose_check, qc |
| `pose_check.{flag_px, refuse_px, flag_roll_deg, refuse_roll_deg, min_cc}` | pose_check *(proposed, D17)* |
| `qc.{lux_ratio_guard, shift_flag_px, saturation_level, saturation_flag_share}` | qc *(proposed)* |

Masks for P1, P2, P4 and P5 are null until they are drawn on the catalog-sweep references.
`data/raw/2026-10-05_shelf-final/masks/make_masks.py` regenerates the overlays and hashes.

## 7. Deploying to the Pi

Keep a backup of anything that would be overwritten:

```bash
ssh -o BatchMode=yes ai-pi 'cd ~/cmpe249/capture && ts=$(date +%Y%m%dT%H%M%S) && mkdir -p ~/cmpe249/backup/$ts && cp -a pi_capture.py tests ~/cmpe249/backup/$ts/'
rsync -a --exclude __pycache__ capture/pi_capture.py capture/ecc.py capture/pose_check.py \
      capture/run_arrangement.py capture/qc_session.py ai-pi:cmpe249/capture/
rsync -a --exclude __pycache__ capture/tests/ ai-pi:cmpe249/capture/tests/
ssh -o BatchMode=yes ai-pi 'mkdir -p ~/cmpe249/configs ~/cmpe249/golden'
rsync -a configs/capture_lock.json ai-pi:cmpe249/configs/
rsync -a data/raw/2026-10-05_shelf-final/pi/ref_final.jpg ai-pi:cmpe249/golden/
ssh -o BatchMode=yes ai-pi 'cd ~/cmpe249 && python3 -m unittest discover -s capture/tests -v'
```

Never run `rpicam-*` or `pi_capture.py` against the real camera while testing: the student uses
the rig. Use the fake camera.

## 8. Evidence behind the numbers in this file

The following were measured while writing this contract. Full output is in
`data/raw/2026-10-05_shelf-final/masks/check_registration_{gb10,pi}.txt`, which comes from
`check_registration.py` on both hosts.

Translation against `ref_final` on the P3 static mask, edges mode:

| Frame | Shift | cc |
| --- | --- | --- |
| `lp1.10/locked_00` | 0.240 px | 0.9981 |
| `locked_11` (the source of `ref_final`) | 0.001 px | 0.9997 |
| `rpi_view_2026-10-05e` (LensPosition 1.45, before the rigid shift) | 2.808 px | 0.9459 |
| `ref_lp1.45` | 0.217 px | 0.9075 |

- The GB10 and the Pi give the same translations to the printed 3 decimals for all 16
  translation rows. Euclidean results differ by up to 0.011 px and 0.0007°: there is no numpy
  refinement on that path, so OpenCV 4.10 and 5.0 differ slightly.
- Run times per call: 0.1–1.4 s on the GB10 and 1.5–7 s on the Pi, with some Pi euclidean calls
  at 22–25 s. Allow up to 60 s per `pose_check` on the Pi.
- The P3 static mask keeps 1,010,820 px (33.9 % of the frame).
- On `ref_final`, there are 0 px ≥ 250 inside the mask. The share of max-channel ≥ 250 pixels in
  the mug region (glare excluded) is 4.3e-5 (`masks_summary.json`).
