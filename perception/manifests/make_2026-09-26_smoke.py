"""Write perception/manifests/2026-09-26_smoke.json (inputs for smoke_test.py and match_check.py).

Boxes were hand-annotated by reading 100 px grid overlays of the frames (uncertainty about +-15 px):
  UP     the mug in upright frames of the 11:39+ camera pose (pilot, 12:16 captures, rot180 sweep/AF frames)
  NATIVE the same box in native (upside-down) sensor orientation, = rot180(UP) on a 2304x1296 frame
  PROBE  the visible, truncated mug in the 11:28 orchestrator probe (a different, earlier pose)

Run from the repo root:  uv run python perception/manifests/make_2026-09-26_smoke.py
"""

import json
from pathlib import Path

CC = "data/raw/2026-09-26_camera-check"
PI = "data/raw/pilot/2026-09-26-pilot"
NOW = "data/raw/2026-09-26_perception-smoke"
UP = [1290, 305, 2215, 993]
NATIVE = [89, 303, 1014, 991]
PROBE = [0, 830, 985, 1296]
MUG_W = UP[2] - UP[0]
REF = f"{PI}/PILOT00/reference/P1/PILOT00_P1_reference_pilot_ambient_t01.jpg"
CROP_UP = ["crop", 1090, 105, 2304, 1195]  # mug plus ~200 px margin (clipped at the right edge)

imgs = []


def add(**k):
    imgs.append(k)


# --- real frames -------------------------------------------------------------------------------
add(id="probe_full", group="probe", path=f"{CC}/orchestrator_probe_1128.jpg", ops=[], gt_src=PROBE,
    note="11:28 orchestrator probe, full auto (AF), camera pose before it was moved; mug cut off at bottom-left and defocused; scene rolled")
add(id="probe_crop", group="probe", path=f"{CC}/orchestrator_probe_1128.jpg", ops=[["crop", 0, 600, 1300, 1296]], gt_src=PROBE,
    note="hand-picked ROI around the visible part of the mug in the probe")
add(id="sweep_best_native", group="sweep_best", path=f"{CC}/sweep_run2/lp_06.0.jpg", ops=[], gt_src=NATIVE,
    note="sweep run 2, lens 6.0 D (closest step to the 6.17 D mug sharpness peak), native upside-down orientation, ~17 lux")
add(id="sweep_best_upright", group="sweep_best", path=f"{CC}/sweep_run2/lp_06.0.jpg", ops=[["rot180"]], gt_src=NATIVE,
    note="same frame rotated 180 deg (upright)")
add(id="sweep_best_upright_crop", group="sweep_best", path=f"{CC}/sweep_run2/lp_06.0.jpg", ops=[["rot180"], CROP_UP], gt_src=NATIVE,
    note="upright, cropped to the mug plus ~200 px margin")
add(glob=f"{PI}/PILOT00/*/P1/*.jpg", id_prefix="pilot_", group="pilot", ops=[], gt_src=UP,
    note="capture-tool pilot frames: locked 66.6 ms / gain 7.84 / lens 6.0 D, --rotation 180, ~4 lux")
add(id="pilot_calibration", group="pilot", path=f"{PI}/calibration/ambient-v1.jpg", ops=[], gt_src=UP,
    note="capture-tool calibration still (auto exposure/AWB, lens 6.0 D)")
add(id="pilot_ref_t01_crop", group="pilot", path=REF, ops=[CROP_UP], gt_src=UP,
    note="pilot reference t01 cropped to the mug plus margin")
add(id="now_lens6_autoexp", group="now", path=f"{NOW}/now_lens6_autoexp.jpg", ops=[], gt_src=UP,
    note="12:16 capture for this check: auto exposure/AWB, lens locked 6.0 D, --rotation 180")
add(id="now_fullauto", group="now", path=f"{NOW}/now_fullauto.jpg", ops=[], gt_src=UP,
    note="12:16 capture for this check: full auto incl. AF (chose 3.03 D, mug slightly soft), --rotation 180")
add(glob=f"{CC}/af/*.jpg", id_prefix="af_", group="af_bright", ops=[["rot180"]], gt_src=NATIVE,
    note="AF-stage stills at ~52 lux (before the lights went off); AF chose 0-3 D, so the mug is defocused; rotated upright")
add(glob=f"{CC}/sweep_run2/lp_*.jpg", id_prefix="sweep2_", group="sweep_series", ops=[["rot180"]], gt_src=NATIVE,
    note="focus sweep run 2 (exposure/AWB locked, ~17 lux; steps 14.0-15.0 D were after a light drop), rotated upright")

# --- synthetic variants of the sharp pilot reference t01 ---------------------------------------
for f in (0.25, 0.5, 0.75):
    x = round(UP[0] + f * MUG_W)
    add(id=f"syn_trunc_left_{int(f * 100)}", group="syn_truncation", path=REF, ops=[["fill", 0, 0, x, 1296, 114, 114, 114]], gt_src=UP,
        note=f"frame left of x={x} painted letterbox-grey: the left {int(f * 100)}% of the mug is 'outside the frame'; image size unchanged")
for f in (0.25, 0.5, 0.75):
    x = round(UP[0] + f * MUG_W)
    add(id=f"syn_occl_left_{int(f * 100)}", group="syn_occlusion", path=REF,
        ops=[["fill", UP[0] - 10, UP[1] - 10, x, UP[3] + 10, 45, 45, 50]], gt_src=UP,
        note=f"flat dark block over the left {int(f * 100)}% of the mug width (full mug height); synthetic, not a real occluder")
for s in (4, 8, 16, 32):
    add(id=f"syn_blur_{s}", group="syn_blur", path=REF, ops=[["blur", s]], gt_src=UP,
        note=f"Gaussian blur sigma={s} px at 2304 px width (= {s * 640 / 2304:.1f} px at the 640 px detector input)")
for d in (15, 30, 45, 90):
    add(id=f"syn_roll_{d}", group="syn_roll", path=REF, ops=[["roll", d]], gt_src=UP,
        note=f"rotated {d} deg CCW, canvas expanded with grey; target box = axis-aligned box of the rotated hand box (looser)")
add(id="syn_roll_180", group="syn_roll", path=REF, ops=[["rot180"]], gt_src=UP,
    note="rotated 180 deg (as the sensor sees it without --rotation 180)")
# brightness: the pilot frames are ~1 stop darker (mean luma 64) than the 17-52 lux frames (125-131)
add(id="syn_pilot_gain_2x", group="syn_brightness", path=REF, ops=[["gain", 2.0]], gt_src=UP,
    note="pilot reference t01 x2 digital gain (mean luma 64 -> ~128); noise is amplified too, so this is not a real exposure change")
add(id="syn_sweep_gain_0.5x", group="syn_brightness", path=f"{CC}/sweep_run2/lp_06.0.jpg", ops=[["rot180"], ["gain", 0.5]], gt_src=NATIVE,
    note="best-focus sweep frame (17 lux, mean luma 125) x0.5 digital gain -> ~62, close to the pilot frames")
add(id="syn_af_gain_0.5x", group="syn_brightness", path=f"{CC}/af/cont_normal_1.jpg", ops=[["rot180"], ["gain", 0.5]], gt_src=NATIVE,
    note="bright AF still (52 lux, mean luma 131) x0.5 digital gain -> ~65")

m = {
    "description": "2026-09-26 perception smoke test inputs: one mug, one camera pose (plus the earlier 11:28 probe pose). "
                   "Boxes are in source-image pixels; see make_2026-09-26_smoke.py for provenance.",
    "gt_note": "gt_src is the tight box around the visible mug, hand-annotated by the agent from 100 px grid overlays "
               "(not by the student); about +-15 px uncertainty.",
    "catalog": ["cup", "mug", "book", "bottle", "box"],
    "target_classes": {"open_vocab": ["cup", "mug"], "coco": ["cup"]},
    # reference crop for match_check.py (the first pilot reference, upright, at the hand box)
    "match_reference": {"path": REF, "ops": [], "box": UP},
    "images": imgs,
}
out = Path(__file__).with_name("2026-09-26_smoke.json")
out.write_text(json.dumps(m, indent=1))
print(f"wrote {out} with {len(imgs)} entries (before glob expansion)")
