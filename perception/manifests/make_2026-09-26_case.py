"""Write perception/manifests/2026-09-26_case.json (inputs for smoke_test.py and match_check.py).

Frames: the stills lane A took at 16:50-17:16 on 2026-09-26 with the camera in the student's new 3D-printed
case (notes/2026-09-26-case-check.md). One camera pose, native orientation (no --rotation), one target object
(the cream marbled cup). Raw data: data/raw/2026-09-26_case-check/pi/ (gitignored).

Target box: hand-annotated by the agent (not the student) on the capture-tool reference t01 (17:12, lens 3.8 D,
the cup's sharpness peak) from 10 px grid zooms at 3x (runs/2026-09-26-case-perception/annotation/): rim top
y 856, body left x 1120, handle right x 1708, foot bottom y 1268. About +-5 px on that frame. The agent had
already seen the Hailo's burned-in cup box in the rpicam-vid frames before annotating.

The whole view shifted between 16:56:15 and 17:09:13 (camera-side; ECC on the background with the cup masked,
lens-matched pair fine 3.75 D -> session 3.8 D: -19.1 px x, +4.3 px y at the image centre, rotation -0.02 deg;
the cup itself moved -20.7 / +3.1 px; runs/2026-09-26-case-perception/cup_vs_background_shift.json). Frames
taken before the shift (probe, calibration stills, LED, sweeps, burst) therefore use CUP_PRE = CUP moved by
(+20.7, -3.1), rounded. CUP_PRE is derived from that measurement, not re-annotated (checked by eye on fine 3.75 D,
about +-10 px). Within each pose, lane A measured shifts of <= ~1 px; focus breathing changes the image scale by
about 0.67 % per dioptre (breathing_check.json), i.e. 5-21 px at the box corners at 0 D.

The rpicam-vid Hailo frames (pi/hailo_e2e*/) are NOT inputs: object_detect_draw_cv drew the Hailo's box and
"cup NN%" label into their pixels. The "hailo_view" group instead crops clean session stills to the sensor
crop rpicam-vid used by default at 1280x720, (768, 432)/3072x1728 on the 4608x2592 sensor = (384, 216)-(1920,
1080) at 2304x1296. It does not reproduce the Hailo's 640x640 lores input (which also changes the aspect ratio).

Run from the repo root:  uv run python perception/manifests/make_2026-09-26_case.py
"""

import json
from pathlib import Path

PI = "data/raw/2026-09-26_case-check/pi"
SES = f"{PI}/data/2026-09-26-case"
CUP = [1120, 856, 1708, 1268]                 # 2304x1296, native orientation
CUP_PRE = [1141, 853, 1729, 1265]             # frames before the 16:56-17:09 view shift (see above)
CUP_4608 = [2 * v for v in CUP_PRE]           # the 4608x2592 probe (pre-shift; same full-FOV geometry, x2)
HAILO_CROP = ["crop", 384, 216, 1920, 1080]   # rpicam-vid default 1280x720 sensor crop, in 2304 px
REF = f"{SES}/CASE00/reference/P1/CASE00_P1_reference_pilot_L1_t01.jpg"

imgs = []


def add(**k):
    imgs.append(k)


add(glob=f"{SES}/CASE00/*/P1/*.jpg", id_prefix="session_", group="session", ops=[], gt_src=CUP,
    note="capture-tool session 17:12:13-17:12:21: CASE00/P1, 2 references + 2 queries, lock L1-v1 (49 995 us, gain 2.89, "
         "AWB locked, lens 3.8 D), rotation 0, ~45 lux")
add(id="session_calibration_L1-v1", group="session", path=f"{SES}/calibration/L1-v1.jpg", ops=[], gt_src=CUP,
    note="capture-tool calibration still: auto exposure/AWB, lens 3.8 D")
add(glob=f"{SES}/CASE00/*/P1/*.jpg", id_prefix="hailoview_", group="hailo_view", ops=[HAILO_CROP], gt_src=CUP,
    note="session still cropped to rpicam-vid's default 1280x720 sensor crop (768,432)/3072x1728: the cup's bottom "
         "~190 px are outside the crop, as in the Hailo default run")
add(id="auto_probe_2304", group="auto", path=f"{PI}/probe/auto_2304.jpg", ops=[], gt_src=CUP_PRE,
    note="16:50 full-auto still incl. AF (lens 1.23 D: focused on the background, cup soft)")
add(id="auto_probe_4608", group="auto", path=f"{PI}/probe/auto_4608.jpg", ops=[], gt_src=CUP_4608,
    note="16:50 full-auto still at 4608x2592 (AF 1.41 D); box = 2 x the 2304 box")
for run, lp in (("led", "1.27"), ("sweep", "1.18"), ("burst", "1.22")):
    add(id=f"auto_calib_{run}", group="auto", path=f"{PI}/{run}/calib.jpg", ops=[], gt_src=CUP_PRE,
        note=f"full-auto calibration still before the {run} series (AF chose {lp} D, cup soft)")
add(glob=f"{PI}/burst/locked_*.jpg", id_prefix="burst_", group="burst", ops=[], gt_src=CUP_PRE,
    note="16:55:57-16:56:15 burst: 12 locked stills, lens 2.5 D; locked_01 has a hand-like shape at the bottom edge "
         "(x ~700-1100), away from the cup")
add(glob=f"{PI}/drift/locked_*.jpg", id_prefix="drift_", group="drift", ops=[], gt_src=CUP,
    note="17:13:17-17:16:28 drift series: 13 stills with the session lock (lens 3.8 D)")
add(glob=f"{PI}/sweep/lp_*.jpg", id_prefix="sweep_", group="sweep", ops=[], gt_src=CUP_PRE,
    note="coarse focus sweep lens 0-15 D in 0.5 D steps, exposure/AWB locked; cup sharpness peak 3.76 D")
add(glob=f"{PI}/fine_cup/lp_*.jpg", id_prefix="fine_", group="fine", ops=[], gt_src=CUP_PRE,
    note="fine focus sweep lens 2.5-5.0 D in 0.25 D steps around the cup peak (3.93 D)")
add(glob=f"{PI}/led/base_*.jpg", id_prefix="led_", group="led_base", ops=[], gt_src=CUP_PRE,
    note="LED leak test at the calibrated exposure (50 ms, gain 2.88), lens 1.27 D (AF value; cup soft)")
add(glob=f"{PI}/led/hi_*.jpg", id_prefix="led_", group="led_hi", ops=[], gt_src=CUP_PRE,
    note="LED leak test at 200 ms / gain 8, lens 1.27 D: the cup and wall are saturated white (cup not visible)")

m = {
    "description": "2026-09-26 case check frames (camera in the 3D-printed case): one cup, one camera pose, native "
                   "orientation. Boxes are in source-image pixels; see make_2026-09-26_case.py for provenance.",
    "gt_note": "gt_src is the tight box around the visible cup incl. handle, hand-annotated by the agent from 10 px grid "
               "zooms of session reference t01 (not by the student); about +-5 px there. Frames before the 16:56-17:09 "
               "view shift use that box moved by the measured cup displacement (+20.7, -3.1 px), about +-10 px.",
    "catalog": ["cup", "mug", "book", "bottle", "box"],
    "target_classes": {"open_vocab": ["cup", "mug"], "coco": ["cup"]},
    # reference crop for match_check.py: the capture-tool reference t01 at the hand box
    "match_reference": {"path": REF, "ops": [], "box": CUP},
    "images": imgs,
}
out = Path(__file__).with_name("2026-09-26_case.json")
out.write_text(json.dumps(m, indent=1))
print(f"wrote {out} with {len(imgs)} entries (before glob expansion)")
