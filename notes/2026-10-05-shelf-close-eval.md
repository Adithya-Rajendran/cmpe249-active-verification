# Shelf setup, close pose (2026-10-05c): evaluation

Frame: `rpi_view_2026-10-05c.jpg`. Evidence is in `pi/`, `perception/` and `geometry/` under `data/raw/2026-10-05_shelf-close/`. All numbers come from one session and one camera pose: no repeats at other poses, no error bars. Geometry numbers are from a pinhole model fitted to one frame (f = 1774 px from the HFOV spec, ±5%; W = 79 cm taken as the inner width), not from tape measurements.

## 1. Verdict

- **This spot is very nearly P3, but it is not P3 yet.** The fitted lens distance is 74.2 cm in front of the shelf front plane (70.5–77.9 cm across f ±5%; 67.3–75.3 cm if W were really 75.4–76.4 cm). The recommended arc puts P3 at 74.5 cm. Fit RMS 1.67 px (`geometry/pose_fit.json`).
- To make it P3 (all from the fit, f = 1774):
  - Move the camera about **7.6 cm to the left** as you face the shelf (fit range 6.9–8.0 cm). The lens is now 7.6 cm right of the opening centre.
  - **Raise the lens about 8.8 cm** (8.6–9.0), from 0.8 cm *below* the mug-shelf top surface to about 8 cm *above* it.
  - **Tilt down about 3.9°**, from the current 1.4° up to about 2.5° down.
  - **Remove the 3.0° yaw** (the camera is now turned about 3° to the left).
  - **Reduce roll from 3.0° to 2° or less.** G2 requires roll ≤ 2°, so 3° fails it.
- Two things must be fixed before freezing the protocol:
  - There is **120 Hz banding that the far pose did not show** (see §2).
  - The **look-alike mug and other changeable props are visible in the top compartment.** G2 says nothing changeable may be visible.

## 2. Measurements (Pi, `pi/`)

| Item | Result |
|---|---|
| Focus | Fixed manual LensPosition **1.45 D**. Two sweeps put the best focus at 1.4–1.5: 1.5 for the middle compartment (worst prop 0.96 of its peak) and 1.4 for the whole frame (worst 0.95). At 1.45 the measured fractions of peak are mug 0.93, plush 0.99, figurines 0.86 and chess clock 0.81. Sweep-to-sweep repeatability is a 3–7% median difference (max 25%). Don't use AF: one AF cycle landed at 3.12 D. Your frame was taken at 1.79 D, where the middle compartment is only 0.67–0.74 of peak. |
| Registration (10 locked frames, about 45 s) | ECC shift median 0.037 px, max 0.079 px |
| Light (175 s, AE on) | Lux 200.7–202.6 (about 4x the far pose's 49–51); gain 1.499; CT 3419–3456 K; exposure 21884–22419 µs |
| Lock values | Requested exposure 22419 µs (the sensor reports 22392), gain 1.499, ColourGains [1.583, 2.493] |
| Flicker | Row-banding std is **1.81% at 1/100 s against 0.33% at 1/120 s**, the pattern expected from 120 Hz flicker. With the same metric the far pose gave 0.69% / 0.78%, which is the noise floor at gain 8, with no excess at 1/100 s. Recomputed from the 10 locked frames at 22.4 ms (`pi/raw_locked/`, an ad-hoc computation that isn't logged in the JSON), banding is still **1.10%** (max 1.22%). Most likely cause is the second light (not confirmed). |
| Saturation (locked) | Full frame ≥250: 0.004%. The main specular hotspot is on the middle shelf's front edge (x≈930–1005, y≈667–710 at ≥240). There are also small point highlights on both mugs (≥250: 0.093% of the study-mug box, 0.048% of the mug2 box). |
| Pixel sizes (manual boxes) | Study mug box 204×134 px (body about 154 px wide). That is about 1.7–1.8x the far pose's width (90 px body, 112 px with handle) and about 1.4x its height. Mug2 box 184×80; plush 205×218 |
| Framing | Spray can is cut off at the top edge (y = 0); Switch box ends 12 px from the bottom |

**Fix for the flicker (untested):** lock the exposure at a multiple of 8333 µs. For example, use **16667 µs with gain about 2.0** or **25000 µs with gain about 1.34** (same brightness as now). Both stay within the G2 gain limit of 4; 8333 µs needed 4.03. Run a short burst to confirm before freezing settings.

## 3. Two-mug discrimination (`perception/`)

- **Detection.** Full frame, yolo11s scores the study mug "cup" at 0.89–0.91. On the middle-compartment crop the score falls to 0.58.
- **Detector labels can't tell the two mugs apart.** In full-frame runs, yolo11s, yolo26s and yolov8s-worldv2 give the look-alike a cup/mug label too (0.16–0.71).
  - yoloe-26s-seg at 1280 calls *both* mugs "3d printed egg", and its masks call them "vase" and "bowl".
- **No representation has a usable absolute threshold.** The look-alike scores as close to the study-mug reference as many genuine study-mug views do. One example: DINOv2 CLS (box crop) gives 0.732 for the look-alike against 0.642 for the far-shelf view of the (assumed) same mug.
- **Only one representation worked, with a relative test** ("which stored reference is closer?"). That was DINOv2 patch nearest-neighbour on background-masked crops.
  - It was correct on 12/12 queries for each mug, with a minimum margin of only +0.047.
  - **These queries are synthetic**: white or grey boxes pasted over 30/50/70% of the mug, left or right, on crops from this same frame. They are compared with the unoccluded crop of that same frame. This is not a test with real occluders or new captures.
  - It also picked the study reference for both earlier views: far shelf 0.681 vs 0.526, pilot 0.809 vs 0.575. Both are assumed, not confirmed, to be the same mug.
  - Other DINOv2 variants reached 3–10/12 on the study-mug queries. CLIP (6/12) and colour histograms (9–12/12 but negative minimum gaps) also fail. Both mugs are the same cream colour.
- **Confound:** the mugs differ in handle direction, height, viewpoint and lighting, and from this view the second mug shows almost no marbling. The separation may come from those differences rather than from identity. Mirroring the look-alike so its handle matches did not flip any decision.
- **What this means for the study:** identity must be decided **relative to stored references of all known look-alikes**, not by a fixed "is this my mug" threshold. Treat this result as preliminary until both mugs have been captured, really occluded, at the same spot.

## 4. Arc plan from your measured shelf (`geometry/`)

**Your heights read bottom-up.**
- Three = 16.5 cm is the mug compartment, and two = 24 cm is the clock/egg/Switch compartment.
- The image fit gives 16.59 and 23.64 cm, using W = 79 cm as the inner width. Their ratio (1.426) does not depend on W or f, and matches 24/16.5 = 1.455 to +2.0%.
- Every other consecutive pair is off by 9–68% in ratio.
- That makes four = 36.5 cm the figurine compartment and one = 31 cm a compartment below the frame.
- The fitted depth is 32.2 cm (30.7–33.6), which agrees with your 31 cm.

**Recommended: R = 90 cm, centred 15.5 cm behind the shelf front, azimuths 0, ±22.5° and ±45°, lens 8 cm above the mug-shelf top, 2.5° down.**

| Stop | Lateral from centre | Out from shelf front |
|---|---|---|
| P1 / P5 | ∓63.6 cm | 48.1 cm |
| P2 / P4 | ∓34.4 cm | 67.6 cm |
| P3 | 0 | 74.5 cm |

- Chords are 35.1 cm and the arc step is 35.34 cm.
- Every stop leaves at least 161 px of margin to the outer front frame of the compartment (201 px to the inner opening). The modelled 8.5 cm mug is about 163 px wide.
- The measured body is 154 px at about the P3 distance, so the real mug is probably about 8 cm wide.
- The old R = 80 plan is too tight with the real 79 cm width. At h = 8 cm the outer front frame is clipped by 9 px at P2/P4, and the opening margins are 49–69 px. At h = 5 cm the frame margins are 0–28 px.
- **Fallback if the floor is too cramped:** R = 90 at ±30°/±15°.
  - The end stops are then ±45.0 cm lateral and 62.4 cm out, and the step is 23.56 cm.
  - The model shows **no tested occluder (10–18 cm wide) fully reveals a centred mug at any ±30° stop**. The best is 78% visible at the end stops (10 cm wide, 14 cm gap).
  - Occluders narrower than 10 cm, or larger gaps, were not modelled.

**positions.json draft** (`geometry/positions_shelf_close_draft.json`, re-checked: it loads with `verify.schema.Positions.from_dict`):
- radius 90
- 5 stops at −45/−22.5/0/22.5/45°
- start stop P3; f_order P1, P2, P4, P5
- height/tilt left null with notes (lens about 8 cm above the shelf, 2.5–3° down)
- tape_arc_cm null until you measure the chords
- travel caps 35.34 / 70.69 / 106.03 / 141.37 / 212.06 cm (1/2/3/4/6 steps)
- λ breakpoints unchanged (2/3, 3/4, 1, 4/3, 3/2, 2)
- The fallback is `geometry/positions_shelf_close_fallback_pm30.json` (step 23.56 cm).

`configs/positions.json` (R = 45, ±60°) and the PREREGISTRATION §6 caps (23.56 cm per step) would need updating if you adopt this.

**Occluder sizing** (model only; R = 90, ±45°, mug centred 20 cm back; fraction of the mug visible at P1/P2/P3/P4/P5):

| Occluder | Gap | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|---|
| 10 cm wide | 14 cm | 100% | 56% | 0% | 56% | 100% |
| 12 cm wide | 10 cm | 89% | 22% | 0% | 22% | 89% |
| 18 cm wide | 6 cm | 11% | 0% | 0% | 0% | 11% |

- **10 cm wide, 14 cm gap:** fully revealed only at the end stops, partly (56%) at P2/P4.
- **12 cm wide, 10 cm gap:** never fully revealed. 89% at the end stops and still 22% at P2/P4, so it is *not* "only at the end stops".
- **18 cm wide, 6 cm gap:** at most 11% visible from any stop, so the stops can't tell cases apart.
- With the mug 15 cm off centre, the reveal pattern becomes lopsided (see `occluder_visibility_u±15_z20` in `geometry/arc_plan.json`).
- Occluder height of 12 or 14 cm made no difference in the model. Anything must fit under the 16.5 cm clear height.
- The modelled target is 8.5×9 cm, which is assumed, not measured.

## 5. Next physical steps (in priority order)

1. **Check the floor space for the end stops.** P1/P5 are 22.3 cm beyond the outer side of the unit and 48 cm out, and there is a shelving unit on the left and a wall/pillar on the right. If they don't fit, use the ±30° fallback, keeping the occluder limitation above in mind.
2. **Move the camera to P3:**
   - 7.6 cm left
   - raise about 8.8 cm, so the lens is about 8 cm above the mug-shelf top
   - tilt to about 2.5° down
   - yaw 0
   - roll ≤ 2°
   - Confirm with a tape measure: 74.5 cm from the front plane to the lens.
3. **Fix the flicker and lock focus:**
   - Exposure 16667 µs (gain about 2.0) or 25000 µs (gain about 1.34), then a quick burst test.
   - Lock LensPosition at 1.45 D, and re-sweep it after the move.
4. **Clear or freeze the top compartment**, and move the second mug out of view, so the frame meets G2.
5. **Two-mug capture:** put each mug in turn at the same spot and orientation in the middle compartment (then swap them), with real occluders, so the identity test is fair.
6. **Tape-measure:**
   - the chords P1–P2…P4–P5 (expect 35.1 cm)
   - both mugs' body width/height and handle depth
   - the stand-base size

## 6. Questions

1. **Which board is the "fourth shelf" at 122 cm?**
   - If 122 cm is to the board *above* the figurine compartment (bottom board not counted), the base is 6.5–8.4 cm. The mug-shelf top is then about 65–67 cm above the floor, and the lens at P3 about 73–75 cm.
   - If it is to the figurine compartment's floor, the base would be 44.9 cm, which seems unlikely.
   - Please also confirm you numbered from the bottom (one = 31 cm is the lowest).
2. Is 79 cm the **inner** width between the side panels, and is 31 cm the inner depth from the front edge to the back panel? The fitted camera distance scales with W.
3. Is the marbled mug in the middle compartment the same mug as in the pilot and far-shelf frames? Is the second mug marbled on its other side?
4. How far do the neighbouring shelving unit and the wall/pillar stick out in front of the shelf front, and how big is the gap from each to the unit's sides?